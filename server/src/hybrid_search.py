import os
import re
import threading
from collections import defaultdict
from typing import Dict, List, Optional

import torch
import torch.nn.functional as F
from rank_bm25 import BM25Okapi
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.app_logging import get_logger


startup_logger = get_logger("startup")

TOKEN_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_./:-]*")
QWEN_RERANKER_ID = "Qwen/Qwen3-Reranker-0.6B"
RERANK_INSTRUCTION = (
    "Given a codebase question, determine whether the source passage provides "
    "direct evidence needed to answer it"
)
RERANK_PREFIX = (
    '<|im_start|>system\nJudge whether the Document meets the requirements based on '
    'the Query and the Instruct provided. Note that the answer can only be "yes" '
    'or "no".<|im_end|>\n<|im_start|>user\n'
)
RERANK_SUFFIX = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
RERANK_MAX_LENGTH = 8192


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
    def __init__(self, model_name: str = None, batch_size: int = None):
        self.device = self._select_device()
        self.reranker_model_name = model_name or os.getenv(
            "RERANKER_MODEL_ID", QWEN_RERANKER_ID
        )
        self.reranker_tokenizer = AutoTokenizer.from_pretrained(
            self.reranker_model_name,
            trust_remote_code=True,
            padding_side="left",
        )
        if self.reranker_tokenizer.pad_token is None:
            self.reranker_tokenizer.pad_token = self.reranker_tokenizer.eos_token
        self.reranker = AutoModelForCausalLM.from_pretrained(
            self.reranker_model_name,
            trust_remote_code=True,
            dtype=torch.float16 if self.device == "cuda" else torch.float32,
        ).to(self.device)
        self.reranker.config.pad_token_id = self.reranker_tokenizer.pad_token_id
        self.reranker.eval()
        self._yes_token_id = self._label_token_id("yes")
        self._no_token_id = self._label_token_id("no")
        self._prefix_token_ids = self.reranker_tokenizer.encode(
            RERANK_PREFIX,
            add_special_tokens=False,
        )
        self._suffix_token_ids = self.reranker_tokenizer.encode(
            RERANK_SUFFIX,
            add_special_tokens=False,
        )
        self.rerank_batch_size = max(
            1,
            batch_size or int(os.getenv("RAG_RERANK_BATCH_SIZE", "4")),
        )
        self._repo_indexes: Dict[int, dict] = {}
        self._index_lock = threading.Lock()
        startup_logger.info("Reranker model loaded")

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
            if cached is not None and len(cached["chunks"]) == len(chunks):
                bm25 = cached["bm25"]
                source_chunks = cached["chunks"]

        if bm25 is None:
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
        if not candidates:
            return []

        documents = [
            f'{item["file_path"]}\n{item.get("signature") or ""}\n{item["content"]}'
            for item in candidates
        ]
        scores = self._score_relevance_batch(query, documents)
        reranked = []
        for item, score in zip(candidates, scores):
            enriched = dict(item)
            enriched["rerank_score"] = score
            reranked.append(enriched)

        reranked.sort(key=lambda item: item["rerank_score"], reverse=True)
        return reranked[:top_k] if top_k is not None else reranked

    def _score_relevance(self, query: str, document: str) -> float:
        return self._score_relevance_batch(query, [document])[0]

    def _score_relevance_batch(self, query: str, documents: List[str]) -> List[float]:
        scores = []
        for start in range(0, len(documents), self.rerank_batch_size):
            batch = documents[start : start + self.rerank_batch_size]
            inputs = self._prepare_rerank_inputs(query, batch)
            with torch.inference_mode():
                logits = self.reranker(**inputs).logits[:, -1, :]

            label_logits = torch.stack(
                [logits[:, self._no_token_id], logits[:, self._yes_token_id]],
                dim=1,
            )
            batch_scores = F.softmax(label_logits.float(), dim=1)[:, 1]
            scores.extend(float(score) for score in batch_scores.detach().cpu())
        return scores

    def _prepare_rerank_inputs(self, query: str, documents: List[str]):
        pairs = [self._format_rerank_pair(query, document) for document in documents]
        content_limit = RERANK_MAX_LENGTH - len(self._prefix_token_ids) - len(
            self._suffix_token_ids
        )
        encoded = self.reranker_tokenizer(
            pairs,
            padding=True,
            truncation=True,
            max_length=content_limit,
            add_special_tokens=False,
            return_attention_mask=True,
            return_tensors="pt",
        )

        # Keep the fixed classifier prompt outside content truncation so the
        # final token always asks the model for its yes/no answer. Rebuild the
        # already padded batch with left padding around the complete sequence.
        batch_size, content_width = encoded["input_ids"].shape
        prefix = torch.tensor(self._prefix_token_ids, dtype=torch.long)
        suffix = torch.tensor(self._suffix_token_ids, dtype=torch.long)
        sequence_width = len(prefix) + content_width + len(suffix)
        input_ids = torch.full(
            (batch_size, sequence_width),
            self.reranker_tokenizer.pad_token_id,
            dtype=torch.long,
        )
        attention_mask = torch.zeros_like(input_ids)

        for row in range(batch_size):
            content_ids = encoded["input_ids"][row][
                encoded["attention_mask"][row].bool()
            ]
            sequence = torch.cat((prefix, content_ids, suffix))
            offset = sequence_width - len(sequence)
            input_ids[row, offset:] = sequence
            attention_mask[row, offset:] = 1

        return {
            "input_ids": input_ids.to(self.device),
            "attention_mask": attention_mask.to(self.device),
        }

    @staticmethod
    def _format_rerank_pair(query: str, document: str) -> str:
        return (
            f"<Instruct>: {RERANK_INSTRUCTION}\n"
            f"<Query>: {query}\n"
            f"<Document>: {document}"
        )

    def _label_token_id(self, label: str) -> int:
        token_id = self.reranker_tokenizer.convert_tokens_to_ids(label)
        if token_id is None or token_id == self.reranker_tokenizer.unk_token_id:
            ids = self.reranker_tokenizer.encode(label, add_special_tokens=False)
            if len(ids) != 1:
                raise RuntimeError(
                    f"Reranker label {label!r} must map to exactly one token; got {ids}."
                )
            token_id = ids[0]
        return int(token_id)

    @staticmethod
    def _select_device() -> str:
        if torch.cuda.is_available():
            return "cuda"
        if torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    @staticmethod
    def normalize_semantic_results(results: List[dict]) -> List[dict]:
        normalized = []
        for rank, item in enumerate(results, start=1):
            enriched = dict(item)
            enriched["semantic_rank"] = rank
            enriched["semantic_score"] = float(item.get("semantic_score", 0.0))
            normalized.append(enriched)
        return normalized
