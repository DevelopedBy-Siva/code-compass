import os
import time
from typing import Callable, List, Optional

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer

QWEN_EMBEDDING_MODEL = "Qwen/Qwen3-Embedding-4B"
RETRIEVAL_INSTRUCTION = (
    "Given a codebase question, retrieve source-code and documentation passages "
    "that provide the most direct evidence for the answer"
)


class EmbeddingGenerator:
    def __init__(self, provider: str = None, model_name: str = None):
        self.model_name = QWEN_EMBEDDING_MODEL
        self.batch_size = max(1, int(os.getenv("QWEN_EMBEDDING_BATCH_SIZE", "8")))
        self.device = self._select_device()

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
        elapsed = time.perf_counter() - started_at
        print(
            f"[embeddings] Model ready dim={self.embedding_dim} load_time={elapsed:.2f}s",
            flush=True,
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
            batch_embeddings = self._encode(batch)
            all_embeddings.append(batch_embeddings)
            elapsed = time.perf_counter() - started_at
            completed = min(start + len(batch), total)
            print(
                f"[embeddings] Finished batch {batch_number}/{total_batches} "
                f"elapsed={elapsed:.2f}s progress={completed}/{total}",
                flush=True,
            )
            if progress_callback:
                progress_callback(completed, total)

        return np.vstack(all_embeddings).astype("float32")

    def get_embedding_dim(self) -> int:
        return self.embedding_dim

    def _encode(self, texts: List[str]) -> np.ndarray:
        inputs = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=8192,
            return_tensors="pt",
        ).to(self.device)

        with torch.inference_mode():
            outputs = self.model(**inputs)
            embeddings = self._last_token_pool(
                outputs.last_hidden_state,
                inputs["attention_mask"],
            ).float()
            embeddings = F.normalize(embeddings, p=2, dim=1)

        embeddings = embeddings.detach().cpu().float().numpy()
        return self._sanitize_embeddings(embeddings)

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
