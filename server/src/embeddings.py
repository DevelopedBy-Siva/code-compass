import os
import statistics
import time
from typing import Callable, List, Optional

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer

from src.app_logging import get_logger, profiling_enabled


startup_logger = get_logger("startup")

QWEN_EMBEDDING_MODEL = "Qwen/Qwen3-Embedding-0.6B"
RETRIEVAL_INSTRUCTION = (
    "Given a codebase question, retrieve source-code and documentation passages "
    "that provide the most direct evidence for the answer"
)


class EmbeddingGenerator:
    def __init__(
        self,
        provider: str = None,
        model_name: str = None,
        enable_profiling: Optional[bool] = None,
    ):
        self.model_name = model_name or os.getenv(
            "EMBEDDING_MODEL_ID", QWEN_EMBEDDING_MODEL
        )
        self.batch_size = max(1, int(os.getenv("QWEN_EMBEDDING_BATCH_SIZE", "8")))
        self.enable_profiling = (
            profiling_enabled() if enable_profiling is None else enable_profiling
        )
        self.device = self._select_device()
        cuda_available = torch.cuda.is_available()
        if self._requires_cuda() and self.device != "cuda":
            raise RuntimeError(
                "CUDA is required for embeddings, but torch.cuda.is_available() is false. "
                "Check the container CUDA runtime and SageMaker host driver compatibility."
            )

        startup_logger.info("CUDA available=%s", str(cuda_available).lower())
        startup_logger.info(
            "Device=%s",
            torch.cuda.get_device_name(0) if cuda_available else self.device,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            trust_remote_code=True,
            padding_side="left",
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModel.from_pretrained(
            self.model_name,
            trust_remote_code=True,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
        ).to(self.device)
        self.model.config.pad_token_id = self.tokenizer.pad_token_id
        self.model.eval()
        self.embedding_dim = int(self.model.config.hidden_size)
        self.model_device = str(next(self.model.parameters()).device)
        startup_logger.info("Embedding model loaded")

    def embed_text(self, text: str) -> np.ndarray:
        query = f"Instruct: {RETRIEVAL_INSTRUCTION}\nQuery: {text}"
        return self._encode([query])[0]

    def embed_batch(
        self,
        texts: List[str],
        batch_size: int = None,
        progress_callback: Optional[Callable[[int, int], None]] = None,
    ) -> np.ndarray:
        if not texts:
            return np.array([], dtype="float32")

        effective_batch_size = max(1, batch_size or self.batch_size)
        all_embeddings = []
        batch_profiles = []
        total = len(texts)

        for start in range(0, total, effective_batch_size):
            batch = texts[start : start + effective_batch_size]
            batch_number = (start // effective_batch_size) + 1
            total_batches = (total + effective_batch_size - 1) // effective_batch_size
            batch_embeddings, profile = self._encode_with_profile(batch)
            all_embeddings.append(batch_embeddings)
            elapsed = profile["total_seconds"]
            completed = min(start + len(batch), total)
            batch_profile = {
                "batch": batch_number,
                "total_batches": total_batches,
                "size": len(batch),
                **profile,
            }
            batch_profiles.append(batch_profile)
            if progress_callback:
                progress_callback(completed, total)

        total_seconds = sum(item["total_seconds"] for item in batch_profiles)
        throughputs = [item["chunks_per_second"] for item in batch_profiles]
        latencies = [item["total_seconds"] for item in batch_profiles]
        worst_index = max(range(len(latencies)), key=latencies.__getitem__)
        best_index = min(range(len(latencies)), key=latencies.__getitem__)
        gpu_profile = (
            {
                "cuda_available": torch.cuda.is_available(),
                "gpu_name": (
                    torch.cuda.get_device_name(0)
                    if torch.cuda.is_available()
                    else None
                ),
                "gpu_memory_allocated_bytes": max(
                    item["gpu_memory_allocated_bytes"] for item in batch_profiles
                ),
                "gpu_memory_peak_bytes": max(
                    item["gpu_memory_peak_bytes"] for item in batch_profiles
                ),
                "gpu_utilization_percent": self._last_non_null(
                    item["gpu_utilization_percent"] for item in batch_profiles
                ),
            }
            if self.enable_profiling
            else {}
        )
        self.last_profile = {
            "device": self.device,
            "model_device": self.model_device,
            "batch_size": effective_batch_size,
            "batches": len(batch_profiles),
            "chunks": total,
            "total_seconds": total_seconds,
            "average_batch_seconds": statistics.mean(latencies),
            "average_chunks_per_second": total / total_seconds if total_seconds else 0.0,
            "median_chunks_per_second": statistics.median(throughputs),
            "worst_batch": batch_profiles[worst_index],
            "best_batch": batch_profiles[best_index],
            **gpu_profile,
        }
        return np.vstack(all_embeddings).astype("float32")

    def get_embedding_dim(self) -> int:
        return self.embedding_dim

    def _encode(self, texts: List[str]) -> np.ndarray:
        embeddings, _ = self._encode_with_profile(texts)
        return embeddings

    def _encode_with_profile(self, texts: List[str]) -> tuple[np.ndarray, dict]:
        total_started_at = time.perf_counter()
        tokenize_started_at = time.perf_counter()
        inputs = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=8192,
            return_tensors="pt",
        )
        tokenize_seconds = time.perf_counter() - tokenize_started_at

        transfer_started_at = time.perf_counter()
        inputs = inputs.to(self.device)
        if self.device == "cuda":
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
        transfer_seconds = time.perf_counter() - transfer_started_at

        inference_started_at = time.perf_counter()
        with torch.inference_mode():
            outputs = self.model(**inputs)
            embeddings = self._last_token_pool(
                outputs.last_hidden_state,
                inputs["attention_mask"],
            ).float()
            embeddings = F.normalize(embeddings, p=2, dim=1)
        if self.device == "cuda":
            torch.cuda.synchronize()
        inference_seconds = time.perf_counter() - inference_started_at

        postprocess_started_at = time.perf_counter()
        embeddings = embeddings.detach().cpu().float().numpy()
        embeddings = self._sanitize_embeddings(embeddings)
        postprocess_seconds = time.perf_counter() - postprocess_started_at
        total_seconds = time.perf_counter() - total_started_at
        profile = {
            "tokenize_seconds": tokenize_seconds,
            "transfer_seconds": transfer_seconds,
            "inference_seconds": inference_seconds,
            "postprocess_seconds": postprocess_seconds,
            "total_seconds": total_seconds,
            "chunks_per_second": len(texts) / total_seconds if total_seconds else 0.0,
        }
        if self.enable_profiling:
            profile.update(self._gpu_stats())
        return embeddings, profile

    @staticmethod
    def _last_token_pool(
        last_hidden_state: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Pool Qwen3 embeddings from the final non-padding token.

        Qwen3-Embedding is trained for last-token pooling. Mean pooling makes
        query and document vectors incompatible with the model's training
        objective and materially hurts retrieval quality.
        """
        if bool(torch.all(attention_mask[:, -1] == 1)):
            return last_hidden_state[:, -1]

        sequence_lengths = attention_mask.sum(dim=1) - 1
        batch_indices = torch.arange(
            last_hidden_state.shape[0],
            device=last_hidden_state.device,
        )
        return last_hidden_state[batch_indices, sequence_lengths]

    @staticmethod
    def _sanitize_embeddings(embeddings: np.ndarray) -> np.ndarray:
        embeddings = np.nan_to_num(
            embeddings.astype("float32", copy=False),
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        invalid_rows = (~np.isfinite(norms[:, 0])) | (norms[:, 0] <= 0.0)
        if np.any(invalid_rows):
            embeddings[invalid_rows] = 0.0
            embeddings[invalid_rows, 0] = 1.0
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        return (embeddings / np.maximum(norms, 1e-12)).astype("float32")

    @staticmethod
    def _select_device() -> str:
        if torch.cuda.is_available():
            return "cuda"
        if torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    @staticmethod
    def _requires_cuda() -> bool:
        return os.getenv("REQUIRE_CUDA", "0").strip().lower() in {
            "1",
            "true",
            "yes",
        }

    def _gpu_stats(self) -> dict:
        if self.device != "cuda":
            return {
                "gpu_memory_allocated_bytes": 0,
                "gpu_memory_peak_bytes": 0,
                "gpu_utilization_percent": None,
            }
        try:
            utilization = torch.cuda.utilization()
        except (AttributeError, ImportError, RuntimeError, OSError):
            utilization = None
        return {
            "gpu_memory_allocated_bytes": int(torch.cuda.memory_allocated()),
            "gpu_memory_peak_bytes": int(torch.cuda.max_memory_allocated()),
            "gpu_utilization_percent": utilization,
        }

    @staticmethod
    def _last_non_null(values):
        result = None
        for value in values:
            if value is not None:
                result = value
        return result
