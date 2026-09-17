import os
from pathlib import Path
from typing import List, Optional, Tuple
from uuid import uuid4

import numpy as np
from chromadb import Client
from chromadb.config import Settings


class ChromaVectorStore:
    def __init__(self, embedding_dim: int, index_path: str = None, persist: bool = True):
        self.embedding_dim = embedding_dim
        self.collection_name = os.getenv("CHROMA_COLLECTION", "repo_qa_chunks")
        self.upsert_batch_size = max(1, int(os.getenv("CHROMA_UPSERT_BATCH_SIZE", "64")))
        self.persist_path = os.getenv("CHROMA_PATH", index_path or "./data/chroma")
        self.persist = persist
        self.client = self._create_client()
        self.collection = self._ensure_collection()

    def _create_client(self):
        if self.persist:
            Path(self.persist_path).mkdir(parents=True, exist_ok=True)
            return Client(
                Settings(
                    is_persistent=True,
                    persist_directory=self.persist_path,
                    anonymized_telemetry=False,
                )
            )

        return Client(Settings(anonymized_telemetry=False))

    def _ensure_collection(self):
        return self.client.get_or_create_collection(
            name=self.collection_name,
            embedding_function=None,
            metadata={"hnsw:space": "cosine"},
        )

    def add_embeddings(self, embeddings: np.ndarray, metadata: List[dict]) -> List[str]:
        if embeddings.size == 0:
            return []

        embeddings = embeddings.astype("float32")
        if embeddings.ndim == 1:
            embeddings = embeddings.reshape(1, -1)
        if not np.isfinite(embeddings).all():
            bad_rows = np.where(~np.isfinite(embeddings).all(axis=1))[0][:10].tolist()
            raise ValueError(f"Embeddings contain NaN or Infinity values at rows: {bad_rows}")

        ids = [uuid4().hex for _ in metadata]
        total_points = len(ids)

        for start in range(0, total_points, self.upsert_batch_size):
            end = start + self.upsert_batch_size
            batch_ids = ids[start:end]
            batch_embeddings = embeddings[start:end].tolist()
            batch_metadata = []
            batch_documents = []

            for idx, meta in zip(batch_ids, metadata[start:end]):
                payload = self._sanitize_metadata(meta)
                payload["id"] = idx
                batch_metadata.append(payload)
                batch_documents.append(str(meta.get("content") or ""))

            batch_number = (start // self.upsert_batch_size) + 1
            total_batches = (total_points + self.upsert_batch_size - 1) // self.upsert_batch_size
            print(
                f"[chroma] Adding batch {batch_number}/{total_batches} "
                f"points={len(batch_ids)} progress={start}/{total_points}",
                flush=True,
            )
            self.collection.add(
                ids=batch_ids,
                embeddings=batch_embeddings,
                metadatas=batch_metadata,
                documents=batch_documents,
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

        where = {"repository_id": repo_filter} if repo_filter is not None else None
        results = self.collection.query(
            query_embeddings=[query_embedding[0].tolist()],
            n_results=k,
            where=where,
            include=["documents", "metadatas", "distances"],
        )

        ids = (results.get("ids") or [[]])[0]
        documents = (results.get("documents") or [[]])[0]
        metadatas = (results.get("metadatas") or [[]])[0]
        distances = (results.get("distances") or [[]])[0]

        hits = []
        for idx, document, meta, distance in zip(ids, documents, metadatas, distances):
            payload = dict(meta or {})
            payload["id"] = payload.get("id") or idx
            payload["content"] = document or ""
            hits.append((self._distance_to_score(distance), payload))
        return hits

    def remove_repository(self, repo_id: int):
        self.collection.delete(where={"repository_id": repo_id})

    def clear(self):
        try:
            self.client.delete_collection(name=self.collection_name)
        except Exception:
            pass
        self.collection = self._ensure_collection()

    def save(self):
        persist = getattr(self.client, "persist", None)
        if callable(persist):
            persist()

    def load(self):
        self.collection = self._ensure_collection()

    def keep_alive(self) -> dict:
        heartbeat = getattr(self.client, "heartbeat", None)
        if callable(heartbeat):
            heartbeat()
        return self.get_stats()

    def get_stats(self) -> dict:
        return {
            "total_vectors": self.collection.count(),
            "embedding_dim": self.embedding_dim,
            "collection_name": self.collection_name,
            "persist_path": self.persist_path if self.persist else None,
        }

    @staticmethod
    def _sanitize_metadata(meta: dict) -> dict:
        sanitized = {}
        for key, value in meta.items():
            if key == "content":
                continue
            if value is None:
                sanitized[key] = ""
            elif isinstance(value, (str, int, float, bool)):
                sanitized[key] = value
            else:
                sanitized[key] = str(value)
        return sanitized

    @staticmethod
    def _distance_to_score(distance: float) -> float:
        if distance is None:
            return 0.0
        return max(0.0, min(1.0, 1.0 - float(distance)))
