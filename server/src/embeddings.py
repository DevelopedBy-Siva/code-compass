import json
import os
import statistics
import time
from typing import Callable, List, Optional

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer

QWEN_EMBEDDING_MODEL = "Qwen/Qwen3-Embedding-0.6B"
RETRIEVAL_INSTRUCTION = (
    "Given a codebase question, retrieve source-code and documentation passages "
    "that provide the most direct evidence for the answer"
)


class EmbeddingGenerator:
    def __init__(self, provider: str = None, model_name: str = None):
        self.model_name = model_name or os.getenv(
            "EMBEDDING_MODEL_ID", QWEN_EMBEDDING_MODEL
        )
        self.batch_size = max(1, int(os.getenv("QWEN_EMBEDDING_BATCH_SIZE", "8")))
        self.device = self._select_device()
        cuda_available = torch.cuda.is_available()
        if self._requires_cuda() and self.device != "cuda":
            raise RuntimeError(
                "CUDA is required for embeddings, but torch.cuda.is_available() is false. "
                "Check the container CUDA runtime and SageMaker host driver compatibility."
            )

        print(
            f"[embeddings] Loading {self.model_name} on device={self.device}",
            flush=True,
        )
        started_at = time.perf_counter()
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
        elapsed = time.perf_counter() - started_at
        print(
            f"[embeddings] Model ready dim={self.embedding_dim} load_time={elapsed:.2f}s",
            flush=True,
        )
        self._log_profile(
            "embedding_device",
            cuda_available=cuda_available,
            device=self.device,
            gpu_name=(torch.cuda.get_device_name(0) if cuda_available else None),
            model_device=self.model_device,
            batch_size=self.batch_size,
            model_name=self.model_name,
        )

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
            print(
                f"[embeddings] Encoding batch {batch_number}/{total_batches} "
                f"items={len(batch)} progress={start}/{total}",
                flush=True,
            )
            batch_embeddings, profile = self._encode_with_profile(batch)
            all_embeddings.append(batch_embeddings)
            elapsed = profile["total_seconds"]
            completed = min(start + len(batch), total)
            print(
                f"[embeddings] Finished batch {batch_number}/{total_batches} "
                f"elapsed={elapsed:.2f}s progress={completed}/{total}",
                flush=True,
            )
            batch_profile = {
                "batch": batch_number,
                "total_batches": total_batches,
                "size": len(batch),
                **profile,
            }
            batch_profiles.append(batch_profile)
            self._log_profile("embedding_batch", **batch_profile)
            if progress_callback:
                progress_callback(completed, total)

        total_seconds = sum(item["total_seconds"] for item in batch_profiles)
        throughputs = [item["chunks_per_second"] for item in batch_profiles]
        latencies = [item["total_seconds"] for item in batch_profiles]
        worst_index = max(range(len(latencies)), key=latencies.__getitem__)
        best_index = min(range(len(latencies)), key=latencies.__getitem__)
        self.last_profile = {
            "device": self.device,
            "cuda_available": torch.cuda.is_available(),
            "gpu_name": (
                torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
            ),
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
        self._log_profile("embedding_summary", **self.last_profile)
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
        return embeddings, {
            "tokenize_seconds": tokenize_seconds,
            "transfer_seconds": transfer_seconds,
            "inference_seconds": inference_seconds,
            "postprocess_seconds": postprocess_seconds,
            "total_seconds": total_seconds,
            "chunks_per_second": len(texts) / total_seconds if total_seconds else 0.0,
            **self._gpu_stats(),
        }

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

    @staticmethod
    def _log_profile(event: str, **fields) -> None:
        print(
            "[profile] " + json.dumps({"event": event, **fields}, sort_keys=True),
            flush=True,
        )
