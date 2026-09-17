import os
import time
from typing import Callable, List, Optional

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer, BitsAndBytesConfig


DEFAULT_MODEL = "Qwen/Qwen3-Embedding-4B"
EMBEDDING_DIM = 2560
DEFAULT_QUERY_INSTRUCTION = (
    "Given a natural language query about a software repository, retrieve relevant "
    "source code passages that answer the query"
)


class EmbeddingGenerator:
    def __init__(self, provider: str = None, model_name: str = None):
        """Load the local 4-bit Qwen3 embedding model.

        ``provider`` remains in the signature for downstream compatibility. Embedding
        providers are no longer selectable; embeddings always run locally with Qwen3.
        """
        if provider and provider.lower() not in {"local", "qwen3"}:
            raise ValueError(
                "Remote embedding providers were removed; use the local Qwen3 model."
            )

        self.provider = "local"
        self.model_name = model_name or os.getenv("QWEN_EMBEDDING_MODEL", DEFAULT_MODEL)
        self.batch_size = max(1, int(os.getenv("EMBEDDING_BATCH_SIZE", "2")))
        self.max_length = int(os.getenv("QWEN_EMBEDDING_MAX_LENGTH", "8192"))
        self.embedding_dim = EMBEDDING_DIM
        self.query_instruction = os.getenv(
            "QWEN_EMBEDDING_QUERY_INSTRUCTION", DEFAULT_QUERY_INSTRUCTION
        ).strip()

        print(
            f"[embeddings] Loading 4-bit local model={self.model_name}",
            flush=True,
        )
        started_at = time.perf_counter()
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=self._compute_dtype(),
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            padding_side="left",
        )
        self.model = AutoModel.from_pretrained(
            self.model_name,
            device_map=os.getenv("QWEN_DEVICE_MAP", "auto"),
            quantization_config=quantization_config,
        ).eval()

        model_dim = int(getattr(self.model.config, "hidden_size", EMBEDDING_DIM))
        if model_dim != EMBEDDING_DIM:
            raise RuntimeError(
                f"{self.model_name} produced dimension {model_dim}; expected {EMBEDDING_DIM}."
            )

        elapsed = time.perf_counter() - started_at
        print(
            f"[embeddings] Model ready dim={self.embedding_dim} load_time={elapsed:.2f}s",
            flush=True,
        )

    def embed_text(self, text: str) -> np.ndarray:
        query = (
            f"Instruct: {self.query_instruction}\nQuery: {text}"
            if self.query_instruction
            else text
        )
        return self._encode_with_backoff([query], batch_size=1)[0]

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
            batch_embeddings = self._encode_with_backoff(
                batch,
                batch_size=min(effective_batch_size, len(batch)),
            )
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

    def _encode_with_backoff(
        self,
        texts: List[str],
        batch_size: int = None,
    ) -> np.ndarray:
        effective_batch_size = max(1, batch_size or self.batch_size)

        while True:
            try:
                encoded_batches = []
                for start in range(0, len(texts), effective_batch_size):
                    batch = texts[start : start + effective_batch_size]
                    inputs = self.tokenizer(
                        batch,
                        padding=True,
                        truncation=True,
                        max_length=self.max_length,
                        return_tensors="pt",
                    ).to(self.model.device)
                    with torch.inference_mode():
                        outputs = self.model(**inputs)
                        embeddings = self._last_token_pool(
                            outputs.last_hidden_state,
                            inputs["attention_mask"],
                        )
                        embeddings = F.normalize(embeddings, p=2, dim=1)
                    encoded_batches.append(embeddings.float().cpu().numpy())
                return np.vstack(encoded_batches).astype("float32")
            except RuntimeError as exc:
                message = str(exc).lower()
                is_memory_error = "out of memory" in message or "mps" in message
                if not is_memory_error or effective_batch_size == 1:
                    raise
                smaller_batch_size = max(1, effective_batch_size // 2)
                print(
                    "[embeddings] Retrying batch with smaller size due to memory "
                    f"pressure: {effective_batch_size} -> {smaller_batch_size}",
                    flush=True,
                )
                effective_batch_size = smaller_batch_size

    def get_embedding_dim(self) -> int:
        return self.embedding_dim

    @staticmethod
    def _last_token_pool(last_hidden_states, attention_mask):
        if attention_mask[:, -1].sum() == attention_mask.shape[0]:
            return last_hidden_states[:, -1]

        sequence_lengths = attention_mask.sum(dim=1) - 1
        batch_size = last_hidden_states.shape[0]
        return last_hidden_states[
            torch.arange(batch_size, device=last_hidden_states.device),
            sequence_lengths,
        ]

    @staticmethod
    def _compute_dtype():
        configured = os.getenv("QWEN_COMPUTE_DTYPE", "float16").lower()
        dtypes = {
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
        }
        if configured not in dtypes:
            raise ValueError(
                "QWEN_COMPUTE_DTYPE must be float16, bfloat16, or float32."
            )
        return dtypes[configured]
