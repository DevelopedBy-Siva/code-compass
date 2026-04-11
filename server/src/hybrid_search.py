import math
import re
from collections import defaultdict
from typing import List

from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder


TOKEN_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_./:-]*")


def tokenize(text: str) -> List[str]:
    return [token.lower() for token in TOKEN_RE.findall(text)]


class HybridSearchEngine:
    def __init__(self, reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        self.reranker = CrossEncoder(reranker_model)

    def build_for_repository(self, repo_id: int, chunks: List[dict]):
        return None

    def remove_repository(self, repo_id: int):
        return None

    def bm25_search(self, chunks: List[dict], query: str, top_k: int = 12) -> List[dict]:
        if not chunks:
            return []

        tokens = tokenize(query)
        if not tokens:
            return []

        corpus_tokens = [tokenize(chunk["searchable_text"]) for chunk in chunks]
        bm25 = BM25Okapi(corpus_tokens) if corpus_tokens else None
        if not bm25:
            return []

        scores = bm25.get_scores(tokens)
        ranked = sorted(
            zip(chunks, scores),
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

    def rerank(self, query: str, candidates: List[dict], top_k: int = 6) -> List[dict]:
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
        return reranked[:top_k]

    @staticmethod
    def normalize_semantic_results(results: List[dict]) -> List[dict]:
        normalized = []
        for rank, item in enumerate(results, start=1):
            enriched = dict(item)
            enriched["semantic_rank"] = rank
            enriched["semantic_score"] = float(item.get("semantic_score", 0.0))
            normalized.append(enriched)
        return normalized
