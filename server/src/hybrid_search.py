import os
import re
import threading
import time
from collections import defaultdict
from typing import Dict, List, Optional

from rank_bm25 import BM25Okapi
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

TOKEN_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_./:-]*")
DEFAULT_RERANKER_MODEL = "Qwen/Qwen3-Reranker-4B"
DEFAULT_RERANK_INSTRUCTION = (
    "Given a question about a software repository, retrieve relevant source code "
    "passages that answer the question"
)


class Qwen3Reranker:
    def __init__(self, model_name: str = DEFAULT_RERANKER_MODEL):
        self.model_name = model_name
        self.batch_size = max(1, int(os.getenv("RERANKER_BATCH_SIZE", "2")))
        self.max_length = int(os.getenv("QWEN_RERANKER_MAX_LENGTH", "8192"))
        self.instruction = os.getenv(
            "QWEN_RERANKER_INSTRUCTION", DEFAULT_RERANK_INSTRUCTION
        ).strip()
        self.prefix = (
            '<|im_start|>system\nJudge whether the Document meets the requirements '
            'based on the Query and the Instruct provided. Note that the answer can '
            'only be "yes" or "no".<|im_end|>\n<|im_start|>user\n'
        )
        self.suffix = (
            "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
        )

        print(f"[reranker] Loading 4-bit local model={model_name}", flush=True)
        started_at = time.perf_counter()
        compute_dtype = os.getenv("QWEN_COMPUTE_DTYPE", "float16").lower()
        dtype_map = {
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
        }
        if compute_dtype not in dtype_map:
            raise ValueError(
                "QWEN_COMPUTE_DTYPE must be float16, bfloat16, or float32."
            )
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=dtype_map[compute_dtype],
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            padding_side="left",
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map=os.getenv("QWEN_DEVICE_MAP", "auto"),
            quantization_config=quantization_config,
        ).eval()
        self.false_token_id = self.tokenizer.convert_tokens_to_ids("no")
        self.true_token_id = self.tokenizer.convert_tokens_to_ids("yes")
        self.prefix_tokens = self.tokenizer.encode(
            self.prefix, add_special_tokens=False
        )
        self.suffix_tokens = self.tokenizer.encode(
            self.suffix, add_special_tokens=False
        )
        print(
            f"[reranker] Model ready load_time={time.perf_counter() - started_at:.2f}s",
            flush=True,
        )

    def predict(self, pairs: List[List[str]]) -> List[float]:
        formatted = [self._format_pair(query, document) for query, document in pairs]
        scores = []
        content_length = self.max_length - len(self.prefix_tokens) - len(self.suffix_tokens)
        if content_length < 1:
            raise ValueError("QWEN_RERANKER_MAX_LENGTH is too small for the model prompt.")

        for start in range(0, len(formatted), self.batch_size):
            batch = formatted[start : start + self.batch_size]
            inputs = self.tokenizer(
                batch,
                padding=False,
                truncation=True,
                max_length=content_length,
                add_special_tokens=False,
                return_attention_mask=False,
            )
            inputs["input_ids"] = [
                self.prefix_tokens + token_ids + self.suffix_tokens
                for token_ids in inputs["input_ids"]
            ]
            padded = self.tokenizer.pad(
                inputs,
                padding=True,
                return_tensors="pt",
            ).to(self.model.device)
            with torch.inference_mode():
                logits = self.model(**padded).logits[:, -1, :]
                batch_scores = (
                    logits[:, self.true_token_id] - logits[:, self.false_token_id]
                )
            scores.extend(batch_scores.float().cpu().tolist())

        return scores

    def _format_pair(self, query: str, document: str) -> str:
        return (
            f"<Instruct>: {self.instruction}\n"
            f"<Query>: {query}\n"
            f"<Document>: {document}"
        )


def tokenize(text: str) -> List[str]:
    raw_tokens = TOKEN_RE.findall(text or "")
    tokens = []

    for raw in raw_tokens:
        lowered = raw.lower()
        tokens.append(lowered)

        # Keep the original code/path token, but also expose its components to
        # BM25. This makes sendDocument, seal-document.handler.ts, etc. match
        # natural-language queries much more reliably.
        pieces = re.split(r"[./:_-]+", raw)
        for piece in pieces:
            if not piece:
                continue
            tokens.append(piece.lower())
            camel_parts = re.findall(
                r"[A-Z]?[a-z]+|[A-Z]+(?=[A-Z]|$)|\d+",
                piece,
            )
            tokens.extend(part.lower() for part in camel_parts if part)

    return [token for token in tokens if token]


class HybridSearchEngine:
    def __init__(self, reranker_model: str = DEFAULT_RERANKER_MODEL):
        self.reranker = Qwen3Reranker(reranker_model)
        # Per-repo cached BM25 index so a question doesn't have to
        # re-tokenize and re-build the lexical index over every chunk in the
        # repo on every single request. Built once when indexing finishes,
        # evicted when the repo is reset/deleted/expired.
        self._repo_indexes: Dict[int, dict] = {}
        self._index_lock = threading.Lock()

    def build_for_repository(self, repo_id: int, chunks: List[dict]):
        if not chunks:
            with self._index_lock:
                self._repo_indexes.pop(repo_id, None)
            return

        corpus_tokens = [tokenize(chunk["searchable_text"]) for chunk in chunks]
        bm25 = BM25Okapi(corpus_tokens) if corpus_tokens else None
        with self._index_lock:
            self._repo_indexes[repo_id] = {"bm25": bm25, "chunks": chunks}

    def remove_repository(self, repo_id: int):
        with self._index_lock:
            self._repo_indexes.pop(repo_id, None)

    def bm25_search(
        self,
        chunks: List[dict],
        query: str,
        top_k: int = 12,
        repo_id: Optional[int] = None,
    ) -> List[dict]:
        if not chunks:
            return []
        tokens = tokenize(query)
        if not tokens:
            return []

        bm25 = None
        source_chunks = chunks

        if repo_id is not None:
            with self._index_lock:
                cached = self._repo_indexes.get(repo_id)
            # Guard against a stale cache (e.g. repo was re-indexed but the
            # cache write raced with this read) by checking the corpus size
            # still lines up before trusting it.
            if cached is not None and len(cached["chunks"]) == len(chunks):
                bm25 = cached["bm25"]
                source_chunks = cached["chunks"]

        if bm25 is None:
            # Fall back to building an ephemeral index. Keeps this method
            # correct on its own even if build_for_repository wasn't called
            # first (e.g. direct/test usage), just without the caching win.
            corpus_tokens = [tokenize(chunk["searchable_text"]) for chunk in chunks]
            bm25 = BM25Okapi(corpus_tokens) if corpus_tokens else None
            source_chunks = chunks

        if not bm25:
            return []

        scores = bm25.get_scores(tokens)
        ranked = sorted(
            zip(source_chunks, scores),
            key=lambda item: item[1],
            reverse=True,
        )[:top_k]

        results = []
        for rank, (chunk, score) in enumerate(ranked, start=1):
            chunk = dict(chunk)
            chunk["bm25_score"] = float(score)
            chunk["bm25_rank"] = rank
            results.append(chunk)
        return results

    def reciprocal_rank_fusion(
        self,
        lexical_results: List[dict],
        semantic_results: List[dict],
        top_k: int = 10,
        k: int = 60,
    ) -> List[dict]:
        fused = defaultdict(lambda: {"rrf_score": 0.0})

        for rank, item in enumerate(lexical_results, start=1):
            fused[item["id"]]["rrf_score"] += 1.0 / (k + rank)
            fused[item["id"]].update(item)

        for rank, item in enumerate(semantic_results, start=1):
            fused[item["id"]]["rrf_score"] += 1.0 / (k + rank)
            fused[item["id"]].update(item)

        merged = sorted(fused.values(), key=lambda item: item["rrf_score"], reverse=True)
        return merged[:top_k]

    def rerank(
        self,
        query: str,
        candidates: List[dict],
        top_k: Optional[int] = None,
    ) -> List[dict]:
        """Score candidates with Qwen3-Reranker-4B and optionally truncate.

        Reranking depth is intentionally separate from answer-context depth.
        Callers can rerank a broad candidate set and still send only a small
        final source set to the LLM.
        """
        if not candidates:
            return []

        pairs = [
            [query, f'{item["file_path"]}\n{item.get("signature") or ""}\n{item["content"]}']
            for item in candidates
        ]
        scores = self.reranker.predict(pairs)

        reranked = []
        for item, score in zip(candidates, scores):
            enriched = dict(item)
            enriched["rerank_score"] = float(score)
            reranked.append(enriched)

        reranked.sort(key=lambda item: item["rerank_score"], reverse=True)
        return reranked[:top_k] if top_k is not None else reranked

    @staticmethod
    def normalize_semantic_results(results: List[dict]) -> List[dict]:
        normalized = []
        for rank, item in enumerate(results, start=1):
            enriched = dict(item)
            enriched["semantic_rank"] = rank
            enriched["semantic_score"] = float(item.get("semantic_score", 0.0))
            normalized.append(enriched)
        return normalized
