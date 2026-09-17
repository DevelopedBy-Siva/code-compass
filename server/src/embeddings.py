import time
from typing import Callable, List, Optional

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer

QWEN_EMBEDDING_MODEL = "Qwen/Qwen3-Embedding-4B"


class EmbeddingGenerator:
    def __init__(self, provider: str = None, model_name: str = None):
        self.model_name = QWEN_EMBEDDING_MODEL
        self.batch_size = 4
        self.device = self._select_device()

        print(
            f"[embeddings] Loading {self.model_name} on device={self.device}",
            flush=True,
        )
        started_at = time.perf_counter()
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            trust_remote_code=True,
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
        return self.embed_batch([text])[0]

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
            token_embeddings = outputs.last_hidden_state.float()
            mask = inputs["attention_mask"].unsqueeze(-1).to(dtype=torch.float32)
            summed = (token_embeddings * mask).sum(dim=1)
            counts = mask.sum(dim=1).clamp(min=1.0)
            embeddings = F.normalize(summed / counts, p=2, dim=1)

        embeddings = embeddings.detach().cpu().float().numpy()
        return self._sanitize_embeddings(embeddings)

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
