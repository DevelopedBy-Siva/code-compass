import json
import os
import statistics
import time
from typing import List, Optional, Tuple
from uuid import uuid4

import numpy as np
from qdrant_client import QdrantClient, models


class QdrantVectorStore:
    def __init__(
        self,
        embedding_dim: int,
        client: Optional[QdrantClient] = None,
        collection_name: Optional[str] = None,
        url: Optional[str] = None,
        api_key: Optional[str] = None,
        timeout_seconds: Optional[int] = None,
        upsert_batch_size: Optional[int] = None,
    ):
        self.embedding_dim = int(embedding_dim)
        # Keep the model size and pooling strategy in the collection name.
        # Vectors from another model or pooling strategy are incompatible.
        self.collection_name = collection_name or os.getenv(
            "QDRANT_COLLECTION",
            "code_compass_qwen3_embedding_0_6b_last_token_cache_v2",
        )
        self.upsert_batch_size = max(
            1, upsert_batch_size or int(os.getenv("QDRANT_UPSERT_BATCH_SIZE", "64"))
        )
        self.timeout_seconds = max(
            1, timeout_seconds or int(os.getenv("QDRANT_TIMEOUT_SECONDS", "60"))
        )

        if client is None:
            self.url = (url if url is not None else os.getenv("QDRANT_URL", "")).strip()
            if not self.url:
                raise RuntimeError(
                    "QDRANT_URL is required. Set it to your Qdrant cluster endpoint."
                )
            api_key = api_key or os.getenv("QDRANT_API_KEY", "").strip() or None
            self.client = QdrantClient(
                url=self.url,
                api_key=api_key,
                timeout=self.timeout_seconds,
            )
        else:
            # Dependency injection keeps unit tests independent of cloud
            # credentials while exercising the same Qdrant operations.
            self.url = None
            self.client = client

        self._collection_ready = False
        self._ensure_collection()

    def _ensure_collection(self) -> bool:
        if self.client.collection_exists(collection_name=self.collection_name):
            collection = self.client.get_collection(
                collection_name=self.collection_name,
            )
            self._validate_collection_dimension(collection)
            self._ensure_repository_payload_index(collection)
            self._collection_ready = True
            return True

        # The evaluation runner asks for a cached-vector count before loading
        # the embedding model. At that point its dimension is intentionally 0;
        # report an empty store and let the fully initialized RAG system create
        # the collection with the real dimension.
        if self.embedding_dim <= 0:
            self._collection_ready = False
            return False

        self.client.create_collection(
            collection_name=self.collection_name,
            vectors_config=models.VectorParams(
                size=self.embedding_dim,
                distance=models.Distance.COSINE,
            ),
        )
        collection = self.client.get_collection(
            collection_name=self.collection_name,
        )
        self._ensure_repository_payload_index(collection)
        self._collection_ready = True
        return True

    def _ensure_repository_payload_index(self, collection) -> None:
        payload_schema = getattr(collection, "payload_schema", None) or {}
        required_indexes = {
            "repository_key": models.PayloadSchemaType.KEYWORD,
            "cache_generation": models.PayloadSchemaType.KEYWORD,
            "cache_ready": models.PayloadSchemaType.BOOL,
        }
        for field_name, field_schema in required_indexes.items():
            if field_name in payload_schema:
                continue
            self.client.create_payload_index(
                collection_name=self.collection_name,
                field_name=field_name,
                field_schema=field_schema,
                wait=True,
            )

    def _validate_collection_dimension(self, collection) -> None:
        if self.embedding_dim <= 0:
            return
        vector_config = collection.config.params.vectors
        existing_dim = getattr(vector_config, "size", None)
        if existing_dim is None and isinstance(vector_config, dict):
            default_config = vector_config.get("")
            existing_dim = getattr(default_config, "size", None)
        if existing_dim is not None and int(existing_dim) != self.embedding_dim:
            raise RuntimeError(
                f"Qdrant collection {self.collection_name!r} uses vector dimension "
                f"{existing_dim}, but the embedding model produces {self.embedding_dim}. "
                "Use a new QDRANT_COLLECTION name and re-index."
            )

    def _require_collection(self) -> None:
        if self._collection_ready:
            return
        if not self._ensure_collection():
            raise RuntimeError(
                "Cannot create the Qdrant collection without a positive embedding dimension."
            )

    def add_embeddings(self, embeddings: np.ndarray, metadata: List[dict]) -> List[str]:
        if embeddings.size == 0:
            return []

        embeddings = embeddings.astype("float32")
        if embeddings.ndim == 1:
            embeddings = embeddings.reshape(1, -1)
        if embeddings.ndim != 2:
            raise ValueError("Embeddings must be a one- or two-dimensional array")
        if len(metadata) != embeddings.shape[0]:
            raise ValueError(
                "Embedding and metadata counts differ: "
                f"{embeddings.shape[0]} embeddings for {len(metadata)} metadata rows"
            )
        if self.embedding_dim > 0 and embeddings.shape[1] != self.embedding_dim:
            raise ValueError(
                f"Embedding dimension {embeddings.shape[1]} does not match "
                f"the configured dimension {self.embedding_dim}"
            )
        if not np.isfinite(embeddings).all():
            bad_rows = np.where(~np.isfinite(embeddings).all(axis=1))[0][:10].tolist()
            raise ValueError(f"Embeddings contain NaN or Infinity values at rows: {bad_rows}")

        self._require_collection()
        ids = [str(uuid4()) for _ in metadata]
        total_points = len(ids)
        request_profiles = []
        upload_started_at = time.perf_counter()

        for start in range(0, total_points, self.upsert_batch_size):
            end = start + self.upsert_batch_size
            batch_ids = ids[start:end]
            batch_embeddings = embeddings[start:end]
            batch_metadata = metadata[start:end]
            points = []

            for point_id, vector, meta in zip(
                batch_ids,
                batch_embeddings,
                batch_metadata,
            ):
                payload = self._sanitize_payload(meta)
                payload["id"] = point_id
                points.append(
                    models.PointStruct(
                        id=point_id,
                        vector=vector.tolist(),
                        payload=payload,
                    )
                )

            batch_number = (start // self.upsert_batch_size) + 1
            total_batches = (
                total_points + self.upsert_batch_size - 1
            ) // self.upsert_batch_size
            print(
                f"[qdrant] Adding batch {batch_number}/{total_batches} "
                f"points={len(points)} progress={start}/{total_points}",
                flush=True,
            )
            request_started_at = time.perf_counter()
            self.client.upsert(
                collection_name=self.collection_name,
                points=points,
                wait=True,
            )
            request_seconds = time.perf_counter() - request_started_at
            request_profile = {
                "request": batch_number,
                "total_requests": total_batches,
                "points": len(points),
                "seconds": request_seconds,
                "points_per_second": (
                    len(points) / request_seconds if request_seconds else 0.0
                ),
            }
            request_profiles.append(request_profile)
            self._log_profile("qdrant_upsert", **request_profile)

        total_upload_seconds = time.perf_counter() - upload_started_at
        latencies = [item["seconds"] for item in request_profiles]
        points_per_request = [item["points"] for item in request_profiles]
        self.last_upsert_profile = {
            "upsert_calls": len(request_profiles),
            "total_points": total_points,
            "configured_batch_size": self.upsert_batch_size,
            "points_per_request": points_per_request,
            "average_points_per_request": statistics.mean(points_per_request),
            "average_request_seconds": statistics.mean(latencies),
            "total_upload_seconds": total_upload_seconds,
            "insertion_mode": (
                "individual" if max(points_per_request) == 1 else "batched"
            ),
        }
        self._log_profile("qdrant_summary", **self.last_upsert_profile)

        return ids

    @staticmethod
    def _log_profile(event: str, **fields) -> None:
        print(
            "[profile] " + json.dumps({"event": event, **fields}, sort_keys=True),
            flush=True,
        )

    def get_repository_chunks(
        self,
        repository_key: str,
        generation: Optional[str] = None,
        ready_only: bool = True,
    ) -> List[dict]:
        if not self._collection_ready and not self._ensure_collection():
            return []

        chunks = []
        offset = None
        repository_filter = self._repository_filter(
            repository_key,
            generation=generation,
            ready_only=ready_only,
        )

        while True:
            records, next_offset = self.client.scroll(
                collection_name=self.collection_name,
                scroll_filter=repository_filter,
                limit=self.upsert_batch_size,
                offset=offset,
                with_payload=True,
                with_vectors=False,
            )
            for record in records:
                payload = dict(record.payload or {})
                payload["id"] = payload.get("id") or str(record.id)
                payload["content"] = str(payload.get("content") or "")
                payload.setdefault(
                    "searchable_text",
                    self._build_searchable_text(payload),
                )
                chunks.append(payload)

            if next_offset is None:
                break
            offset = next_offset

        return chunks

    def search(
        self,
        query_embedding: np.ndarray,
        k: int = 10,
        repository_key: Optional[str] = None,
    ) -> List[Tuple[float, dict]]:
        if not self._collection_ready and not self._ensure_collection():
            return []
        if query_embedding.ndim == 1:
            query_embedding = query_embedding.reshape(1, -1)
        query_embedding = query_embedding.astype("float32")
        if query_embedding.ndim != 2 or query_embedding.shape[0] != 1:
            raise ValueError("Query embedding must contain exactly one vector")
        if self.embedding_dim > 0 and query_embedding.shape[1] != self.embedding_dim:
            raise ValueError(
                f"Query dimension {query_embedding.shape[1]} does not match "
                f"the configured dimension {self.embedding_dim}"
            )
        if not np.isfinite(query_embedding).all():
            raise ValueError("Query embedding contains NaN or Infinity values")

        query_filter = (
            self._repository_filter(repository_key)
            if repository_key is not None
            else None
        )
        response = self.client.query_points(
            collection_name=self.collection_name,
            query=query_embedding[0].tolist(),
            query_filter=query_filter,
            limit=max(1, int(k)),
            with_payload=True,
            with_vectors=False,
        )

        hits = []
        for point in response.points:
            payload = dict(point.payload or {})
            payload["id"] = payload.get("id") or str(point.id)
            payload["content"] = str(payload.get("content") or "")
            hits.append((self._normalize_score(point.score), payload))
        return hits

    def activate_repository_generation(
        self,
        repository_key: str,
        generation: str,
    ) -> None:
        """Publish one complete generation, then remove every older generation."""
        self._require_collection()
        generation_filter = self._repository_filter(
            repository_key,
            generation=generation,
            ready_only=False,
        )
        staged_count = self.client.count(
            collection_name=self.collection_name,
            count_filter=generation_filter,
            exact=True,
        ).count
        if staged_count <= 0:
            raise RuntimeError("Cannot activate an empty repository index")

        self.client.set_payload(
            collection_name=self.collection_name,
            payload={"cache_ready": True},
            points=generation_filter,
            wait=True,
        )
        self.client.delete(
            collection_name=self.collection_name,
            points_selector=models.FilterSelector(
                filter=models.Filter(
                    must=[self._match("repository_key", repository_key)],
                    must_not=[self._match("cache_generation", generation)],
                )
            ),
            wait=True,
        )

    def remove_generation(self, repository_key: str, generation: str) -> None:
        if not self._collection_ready and not self._ensure_collection():
            return
        self.client.delete(
            collection_name=self.collection_name,
            points_selector=models.FilterSelector(
                filter=self._repository_filter(
                    repository_key,
                    generation=generation,
                    ready_only=False,
                ),
            ),
            wait=True,
        )

    def delete_repository_cache(self, repository_key: str) -> None:
        if not self._collection_ready and not self._ensure_collection():
            return
        self.client.delete(
            collection_name=self.collection_name,
            points_selector=models.FilterSelector(
                filter=self._repository_filter(repository_key, ready_only=False),
            ),
            wait=True,
        )

    def clear(self):
        if self.client.collection_exists(collection_name=self.collection_name):
            self.client.delete_collection(collection_name=self.collection_name)
        self._collection_ready = False
        if self.embedding_dim > 0:
            self._ensure_collection()

    def save(self):
        # Qdrant persists successful writes server-side.
        return None

    def close(self):
        close = getattr(self.client, "close", None)
        if close:
            close()

    def load(self):
        self._ensure_collection()

    def keep_alive(self) -> dict:
        if self.client.collection_exists(collection_name=self.collection_name):
            self.client.get_collection(collection_name=self.collection_name)
        return self.get_stats()

    def get_stats(self) -> dict:
        if not self.client.collection_exists(collection_name=self.collection_name):
            total_vectors = 0
        else:
            total_vectors = self.client.count(
                collection_name=self.collection_name,
                exact=True,
            ).count
        return {
            "total_vectors": int(total_vectors),
            "embedding_dim": self.embedding_dim,
            "collection_name": self.collection_name,
            "provider": "qdrant",
            "url": self.url,
        }

    @classmethod
    def _repository_filter(
        cls,
        repository_key: str,
        generation: Optional[str] = None,
        ready_only: bool = True,
    ) -> models.Filter:
        must = [cls._match("repository_key", repository_key)]
        if generation is not None:
            must.append(cls._match("cache_generation", generation))
        if ready_only:
            must.append(cls._match("cache_ready", True))
        return models.Filter(
            must=must,
        )

    @staticmethod
    def _match(key: str, value) -> models.FieldCondition:
        return models.FieldCondition(
            key=key,
            match=models.MatchValue(value=value),
        )

    @staticmethod
    def _build_searchable_text(chunk: dict) -> str:
        return "\n".join(
            str(value or "")
            for value in (
                chunk.get("file_path"),
                chunk.get("symbol_name"),
                chunk.get("signature"),
                chunk.get("content"),
            )
        )

    @staticmethod
    def _sanitize_payload(meta: dict) -> dict:
        sanitized = {}
        for key, value in meta.items():
            if value is None:
                sanitized[key] = ""
            elif isinstance(value, (str, int, float, bool)):
                sanitized[key] = value
            else:
                sanitized[key] = str(value)
        return sanitized

    @staticmethod
    def _normalize_score(score: float) -> float:
        if score is None:
            return 0.0
        return max(0.0, min(1.0, float(score)))
