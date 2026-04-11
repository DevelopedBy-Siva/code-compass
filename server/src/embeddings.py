import os
import time
from typing import Callable, List, Optional

import numpy as np
from openai import OpenAI
from sentence_transformers import SentenceTransformer


class EmbeddingGenerator:
    def __init__(self, provider: str = None, model_name: str = None):
        configured_provider = (provider or os.getenv("EMBEDDING_PROVIDER", "auto")).lower()
        self.provider = self._resolve_provider(configured_provider)
        self.model_name = model_name or self._resolve_model_name()
        self.batch_size = int(os.getenv("EMBEDDING_BATCH_SIZE", "8"))
        self.device = os.getenv("EMBEDDING_DEVICE")
        self.client = None
        self.model = None
        self.vertex_task_type_document = os.getenv(
            "VERTEX_EMBEDDING_TASK_TYPE_DOCUMENT", "RETRIEVAL_DOCUMENT"
        )
        self.vertex_task_type_query = os.getenv(
            "VERTEX_EMBEDDING_TASK_TYPE_QUERY", "RETRIEVAL_QUERY"
        )
        self.vertex_output_dimensionality = self._optional_int(
            os.getenv("VERTEX_EMBEDDING_OUTPUT_DIMENSIONALITY")
        )
        self.query_prefix = os.getenv("EMBEDDING_QUERY_PREFIX", "").strip()
        normalized_model_name = self.model_name.lower()
        self.query_prompt_name = (
            os.getenv("EMBEDDING_QUERY_PROMPT_NAME", "query")
            if "nomic-embed-code" in normalized_model_name
            or "coderankembed" in normalized_model_name
            else None
        )

        if self.provider == "openai":
            print(
                f"[embeddings] Initializing OpenAI embeddings with model={self.model_name}",
                flush=True,
            )
            self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
            self.embedding_dim = int(os.getenv("OPENAI_EMBEDDING_DIM", "1536"))
        elif self.provider == "vertex_ai":
            print(
                f"[embeddings] Initializing Vertex AI embeddings with model={self.model_name}",
                flush=True,
            )
            try:
                from google import genai
            except ImportError as exc:
                raise RuntimeError(
                    "Vertex AI embedding support requires the `google-genai` package."
                ) from exc

            project = os.getenv("GOOGLE_CLOUD_PROJECT")
            location = os.getenv("GOOGLE_CLOUD_LOCATION", "us-central1")
            if not project:
                raise RuntimeError(
                    "GOOGLE_CLOUD_PROJECT must be set when using Vertex AI embeddings."
                )

            self.client = genai.Client(
                vertexai=True,
                project=project,
                location=location,
            )
            self.embedding_dim = int(
                os.getenv(
                    "VERTEX_EMBEDDING_DIM",
                    str(self.vertex_output_dimensionality or 3072),
                )
            )
        else:
            model_device = self.device or "cpu"
            print(
                f"[embeddings] Loading local embedding model={self.model_name} on device={model_device}",
                flush=True,
            )
            started_at = time.perf_counter()
            self.model = SentenceTransformer(
                self.model_name,
                trust_remote_code=True,
                device=model_device,
            )
            self.embedding_dim = self.model.get_sentence_embedding_dimension()
            elapsed = time.perf_counter() - started_at
            print(
                f"[embeddings] Model ready dim={self.embedding_dim} load_time={elapsed:.2f}s",
                flush=True,
            )

    def embed_text(self, text: str) -> np.ndarray:
        if self.provider == "openai":
            return self.embed_batch([text])[0]
        if self.provider == "vertex_ai":
            return self._embed_with_vertex(
                [text],
                task_type=self.vertex_task_type_query,
            )[0]
        query_text = f"{self.query_prefix}: {text}" if self.query_prefix else text
        return self._encode_with_backoff([query_text], prompt_name=self.query_prompt_name)[0]

    def embed_batch(
        self,
        texts: List[str],
        batch_size: int = None,
        progress_callback: Optional[Callable[[int, int], None]] = None,
    ) -> np.ndarray:
        if not texts:
            return np.array([], dtype="float32")

        if self.provider == "openai":
            response = self.client.embeddings.create(
                model=self.model_name or "text-embedding-3-small",
                input=texts,
            )
            embeddings = [item.embedding for item in response.data]
            if progress_callback:
                progress_callback(len(texts), len(texts))
            return np.array(embeddings, dtype="float32")
        if self.provider == "vertex_ai":
            return self._embed_batch_with_vertex(
                texts=texts,
                batch_size=batch_size,
                progress_callback=progress_callback,
            )

        effective_batch_size = max(1, batch_size or self.batch_size)
        all_embeddings = []
        total = len(texts)

        for start in range(0, total, effective_batch_size):
            batch = texts[start : start + effective_batch_size]
            batch_number = (start // effective_batch_size) + 1
            total_batches = (total + effective_batch_size - 1) // effective_batch_size
            print(
                f"[embeddings] Encoding batch {batch_number}/{total_batches} "
                f"items={len(batch)} progress={start}/{total}",
                flush=True,
            )
            started_at = time.perf_counter()
            batch_embeddings = self._encode_with_backoff(
                batch,
                batch_size=min(effective_batch_size, len(batch)),
            )
            all_embeddings.append(batch_embeddings)
            elapsed = time.perf_counter() - started_at
            print(
                f"[embeddings] Finished batch {batch_number}/{total_batches} "
                f"elapsed={elapsed:.2f}s progress={min(start + len(batch), total)}/{total}",
                flush=True,
            )
            if progress_callback:
                progress_callback(min(start + len(batch), total), total)

        return np.vstack(all_embeddings).astype("float32")

    def _embed_batch_with_vertex(
        self,
        texts: List[str],
        batch_size: int = None,
        progress_callback: Optional[Callable[[int, int], None]] = None,
    ) -> np.ndarray:
        effective_batch_size = max(1, batch_size or self.batch_size)
        all_embeddings = []
        total = len(texts)

        for start in range(0, total, effective_batch_size):
            batch = texts[start : start + effective_batch_size]
            batch_number = (start // effective_batch_size) + 1
            total_batches = (total + effective_batch_size - 1) // effective_batch_size
            print(
                f"[embeddings] Vertex batch {batch_number}/{total_batches} "
                f"items={len(batch)} progress={start}/{total}",
                flush=True,
            )
            started_at = time.perf_counter()
            batch_embeddings = self._embed_with_vertex(
                batch,
                task_type=self.vertex_task_type_document,
            )
            all_embeddings.append(batch_embeddings)
            elapsed = time.perf_counter() - started_at
            print(
                f"[embeddings] Finished Vertex batch {batch_number}/{total_batches} "
                f"elapsed={elapsed:.2f}s progress={min(start + len(batch), total)}/{total}",
                flush=True,
            )
            if progress_callback:
                progress_callback(min(start + len(batch), total), total)

        return np.vstack(all_embeddings).astype("float32")

    def _embed_with_vertex(self, texts: List[str], task_type: str) -> np.ndarray:
        config = {
            "task_type": task_type,
        }
        if self.vertex_output_dimensionality:
            config["output_dimensionality"] = self.vertex_output_dimensionality

        response = self.client.models.embed_content(
            model=self.model_name,
            contents=texts,
            config=config,
        )
        embeddings = getattr(response, "embeddings", None)
        if not embeddings:
            raise RuntimeError("Vertex AI embeddings returned an empty response.")

        values = []
        for item in embeddings:
            if hasattr(item, "values"):
                values.append(item.values)
            elif isinstance(item, dict):
                values.append(item.get("values"))
            else:
                values.append(getattr(item, "embedding", None))

        if not values or any(vector is None for vector in values):
            raise RuntimeError("Vertex AI embeddings response could not be parsed.")

        return np.array(values, dtype="float32")

    def _encode_with_backoff(
        self,
        texts: List[str],
        batch_size: int = None,
        prompt_name: str = None,
    ) -> np.ndarray:
        effective_batch_size = max(1, batch_size or self.batch_size)

        while True:
            try:
                encode_kwargs = {
                    "sentences": texts,
                    "batch_size": effective_batch_size,
                    "show_progress_bar": len(texts) > effective_batch_size,
                    "convert_to_numpy": True,
                    "normalize_embeddings": True,
                }
                if prompt_name:
                    encode_kwargs["prompt_name"] = prompt_name

                embeddings = self.model.encode(
                    **encode_kwargs,
                )
                return embeddings.astype("float32")
            except RuntimeError as exc:
                message = str(exc).lower()
                is_memory_error = "out of memory" in message or "mps" in message
                if not is_memory_error or effective_batch_size == 1:
                    raise
                print(
                    f"[embeddings] Retrying batch with smaller size due to memory pressure: "
                    f"{effective_batch_size} -> {max(1, effective_batch_size // 2)}",
                    flush=True,
                )
                effective_batch_size = max(1, effective_batch_size // 2)

    def get_embedding_dim(self) -> int:
        return self.embedding_dim

    def _resolve_provider(self, configured_provider: str) -> str:
        if configured_provider != "auto":
            return configured_provider
        if self._is_hf_space() or self._is_test_context():
            return "local"
        return "vertex_ai"

    def _resolve_model_name(self) -> str:
        explicit_model = os.getenv("EMBEDDING_MODEL")
        if explicit_model:
            return explicit_model
        if self.provider == "vertex_ai":
            return os.getenv("VERTEX_EMBEDDING_MODEL", "gemini-embedding-001")
        if self._is_hf_space() or self._is_test_context():
            return os.getenv(
                "LIGHTWEIGHT_LOCAL_EMBEDDING_MODEL",
                "sentence-transformers/all-MiniLM-L6-v2",
            )
        return os.getenv("LOCAL_EMBEDDING_MODEL", "nomic-ai/CodeRankEmbed")

    def _is_hf_space(self) -> bool:
        return bool(os.getenv("SPACE_ID") or os.getenv("HF_SPACE_ID"))

    def _is_test_context(self) -> bool:
        app_env = os.getenv("APP_ENV", os.getenv("ENVIRONMENT", "")).lower()
        return app_env == "test" or bool(os.getenv("PYTEST_CURRENT_TEST"))

    def _optional_int(self, value: Optional[str]) -> Optional[int]:
        if value is None or not str(value).strip():
            return None
        return int(value)
