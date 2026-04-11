import os
from typing import List, Optional, Tuple
from uuid import uuid4

import numpy as np
from qdrant_client import QdrantClient, models


class QdrantVectorStore:
    def __init__(self, embedding_dim: int, index_path: str = None, persist: bool = False):
        self.embedding_dim = embedding_dim
        self.collection_name = os.getenv("QDRANT_COLLECTION", "repo_qa_chunks")
        self.upsert_batch_size = max(1, int(os.getenv("QDRANT_UPSERT_BATCH_SIZE", "64")))
        self.client = self._create_client()
        self._ensure_collection()

    def _create_client(self):
        url = self._clean_env("QDRANT_URL")
        api_key = self._clean_env("QDRANT_API_KEY")
        timeout = int(os.getenv("QDRANT_TIMEOUT_SECONDS", "120"))
        if url:
            return QdrantClient(
                url=url,
                api_key=api_key,
                timeout=timeout,
                check_compatibility=False,
            )
        return QdrantClient(":memory:")

    @staticmethod
    def _clean_env(name: str) -> Optional[str]:
        value = os.getenv(name)
        if value is None:
            return None
        cleaned = value.strip()
        return cleaned or None

    def _ensure_collection(self):
        if not self.client.collection_exists(self.collection_name):
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=models.VectorParams(
                    size=self.embedding_dim,
                    distance=models.Distance.COSINE,
                ),
            )
        self._ensure_payload_indexes()

    def _ensure_payload_indexes(self):
        self.client.create_payload_index(
            collection_name=self.collection_name,
            field_name="repository_id",
            field_schema=models.PayloadSchemaType.INTEGER,
            wait=True,
        )

    def add_embeddings(self, embeddings: np.ndarray, metadata: List[dict]) -> List[int]:
        if embeddings.size == 0:
            return []

        embeddings = embeddings.astype("float32")
        if embeddings.ndim == 1:
            embeddings = embeddings.reshape(1, -1)

        ids = [uuid4().hex for _ in metadata]
        points = []
        for idx, meta, embedding in zip(ids, metadata, embeddings):
            payload = dict(meta)
            payload["id"] = idx
            points.append(
                models.PointStruct(
                    id=idx,
                    vector=embedding.tolist(),
                    payload=payload,
                )
            )
        total_points = len(points)
        for start in range(0, total_points, self.upsert_batch_size):
            batch = points[start : start + self.upsert_batch_size]
            batch_number = (start // self.upsert_batch_size) + 1
            total_batches = (total_points + self.upsert_batch_size - 1) // self.upsert_batch_size
            print(
                f"[qdrant] Upserting batch {batch_number}/{total_batches} "
                f"points={len(batch)} progress={start}/{total_points}",
                flush=True,
            )
            self.client.upsert(
                collection_name=self.collection_name,
                wait=True,
                points=batch,
            )

        return ids

    def search(
        self,
        query_embedding: np.ndarray,
        k: int = 10,
        repo_filter: Optional[int] = None,
    ) -> List[Tuple[float, dict]]:
        if query_embedding.ndim == 1:
            query_embedding = query_embedding.reshape(1, -1)
        query_embedding = query_embedding.astype("float32")

        query_filter = None
        if repo_filter is not None:
            query_filter = models.Filter(
                must=[
                    models.FieldCondition(
                        key="repository_id",
                        match=models.MatchValue(value=repo_filter),
                    )
                ]
            )

        hits = self.client.search(
            collection_name=self.collection_name,
            query_vector=query_embedding[0].tolist(),
            query_filter=query_filter,
            limit=k,
        )

        return [(float(hit.score), dict(hit.payload or {})) for hit in hits]

    def remove_repository(self, repo_id: int):
        self.client.delete(
            collection_name=self.collection_name,
            wait=True,
            points_selector=models.FilterSelector(
                filter=models.Filter(
                    must=[
                        models.FieldCondition(
                            key="repository_id",
                            match=models.MatchValue(value=repo_id),
                        )
                    ]
                )
            ),
        )

    def clear(self):
        if self.client.collection_exists(self.collection_name):
            self.client.delete_collection(self.collection_name)
        self._ensure_collection()

    def save(self):
        return None

    def load(self):
        self._ensure_collection()

    def get_stats(self) -> dict:
        info = self.client.get_collection(self.collection_name)
        return {
            "total_vectors": info.points_count or 0,
            "embedding_dim": self.embedding_dim,
            "collection_name": self.collection_name,
        }
