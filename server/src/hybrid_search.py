import re
import threading
from collections import defaultdict
from typing import Dict, List, Optional

import torch
import torch.nn.functional as F
from rank_bm25 import BM25Okapi
from transformers import AutoModelForCausalLM, AutoTokenizer

TOKEN_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_./:-]*")
QWEN_RERANKER_ID = "Qwen/Qwen3-Reranker-4B"


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
    def __init__(self):
        self.device = self._select_device()
        print(
            f"[reranker] Loading {QWEN_RERANKER_ID} on device={self.device}",
            flush=True,
        )
        self.reranker_tokenizer = AutoTokenizer.from_pretrained(
            QWEN_RERANKER_ID,
            trust_remote_code=True,
        )
        if self.reranker_tokenizer.pad_token is None:
            self.reranker_tokenizer.pad_token = self.reranker_tokenizer.eos_token
        self.reranker = AutoModelForCausalLM.from_pretrained(
            QWEN_RERANKER_ID,
            trust_remote_code=True,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
        ).to(self.device)
        self.reranker.config.pad_token_id = self.reranker_tokenizer.pad_token_id
        self.reranker.eval()
        self._yes_token_ids = self._label_token_ids("yes")
        self._no_token_ids = self._label_token_ids("no")
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

        reranked = []
        for item in candidates:
            document = f'{item["file_path"]}\n{item.get("signature") or ""}\n{item["content"]}'
            score = self._score_relevance(query, document)
            enriched = dict(item)
            enriched["rerank_score"] = score
            reranked.append(enriched)

        reranked.sort(key=lambda item: item["rerank_score"], reverse=True)
        return reranked[:top_k] if top_k is not None else reranked

    def _score_relevance(self, query: str, document: str) -> float:
        prompt = self._build_rerank_prompt(query, document)
        inputs = self.reranker_tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=8192,
        ).to(self.device)

        with torch.inference_mode():
            logits = self.reranker(**inputs).logits[0, -1, :]

        yes_logit = torch.logsumexp(logits[self._yes_token_ids], dim=0)
        no_logit = torch.logsumexp(logits[self._no_token_ids], dim=0)
        return float(F.softmax(torch.stack([no_logit, yes_logit]), dim=0)[1].item())

    def _build_rerank_prompt(self, query: str, document: str) -> str:
        messages = [
            {
                "role": "system",
                "content": "Judge whether the document is relevant to the query. Answer only yes or no.",
            },
            {
                "role": "user",
                "content": f"Query:\n{query}\n\nDocument:\n{document}\n\nRelevant?",
            },
        ]
        return self.reranker_tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

    def _label_token_ids(self, label: str) -> torch.Tensor:
        token_ids = set()
        for variant in {label, f" {label}", label.capitalize(), f" {label.capitalize()}"}:
            ids = self.reranker_tokenizer.encode(variant, add_special_tokens=False)
            if ids:
                token_ids.add(ids[-1])
        if not token_ids:
            raise RuntimeError(f"Could not find reranker token ids for {label!r}.")
        return torch.tensor(sorted(token_ids), device=self.device)

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
