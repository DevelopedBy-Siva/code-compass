import re
from statistics import mean
from typing import Optional


CUTOFFS = (1, 5, 8)


def normalize_path(path: str) -> str:
    normalized = (path or "").strip().replace("\\", "/").lower()
    while normalized.startswith("./"):
        normalized = normalized[2:]
    return normalized.strip("/")


def matches_expected(actual_path: str, expected_sources: list[str]) -> bool:
    """Match an indexed file to an expected file path or expected directory."""
    actual = normalize_path(actual_path)
    for expected in expected_sources:
        expected_path = normalize_path(expected)
        if not expected_path:
            continue
        if actual == expected_path:
            return True
        basename = expected_path.rsplit("/", 1)[-1]
        is_directory = "." not in basename
        if is_directory and actual.startswith(expected_path + "/"):
            return True
    return False


def score_ranking(expected_sources: list[str], ranked_paths: list[str]) -> dict:
    """Calculate Hit@1/5/8 and MRR@8 from one ordered list of file paths."""
    relevant_ranks = [
        rank
        for rank, path in enumerate(ranked_paths, start=1)
        if matches_expected(path, expected_sources)
    ]
    first_relevant_rank = relevant_ranks[0] if relevant_ranks else None
    return {
        f"hit_at_{cutoff}": int(
            first_relevant_rank is not None and first_relevant_rank <= cutoff
        )
        for cutoff in CUTOFFS
    } | {
        "mrr_at_8": (
            1.0 / first_relevant_rank
            if first_relevant_rank is not None and first_relevant_rank <= 8
            else 0.0
        ),
        "first_relevant_rank": first_relevant_rank,
    }


def summarize_rankings(records: list[dict], method: str) -> dict:
    """Average per-query scores for a ranking method."""
    if not records:
        return {"query_count": 0}
    return {
        "query_count": len(records),
        **{
            metric: round(mean(record[method][metric] for record in records), 4)
            for metric in (
                "hit_at_1",
                "hit_at_5",
                "hit_at_8",
                "mrr_at_8",
            )
        },
    }


def summarize_comparison(records: list[dict]) -> dict:
    if not records:
        return {"query_count": 0}
    methods = {
        method: summarize_rankings(records, method)
        for method in ("vector_only", "hybrid_rerank")
    }
    return {
        "query_count": len(records),
        **methods,
        "hybrid_minus_vector_only": {
            metric: round(
                methods["hybrid_rerank"][metric] - methods["vector_only"][metric],
                4,
            )
            for metric in ("hit_at_1", "hit_at_5", "hit_at_8", "mrr_at_8")
        },
    }


def answer_quality_signals(
    case: dict, result: dict, faithfulness: Optional[float]
) -> dict:
    """Compute answer-quality signals for the separate RAG evaluation."""
    answer_tokens = set(re.findall(r"[a-z0-9_]+", result.get("answer", "").lower()))
    keywords = case.get("must_include_any", [])
    matched_keywords = sum(
        bool(tokens := re.findall(r"[a-z0-9_]+", keyword.lower()))
        and all(token in answer_tokens for token in tokens)
        for keyword in keywords
    )
    citations_present = bool(result.get("citations"))
    source_retrieved = bool(result.get("final_retrieval_hit"))
    grounded = (
        source_retrieved
        and citations_present
        and (not keywords or matched_keywords > 0)
    )
    return {
        "citation_present": int(citations_present),
        "expected_source_retrieved": int(source_retrieved),
        "required_keyword_count": len(keywords),
        "matched_required_keyword_count": matched_keywords,
        "grounded_answer_proxy": int(grounded),
        "faithfulness_judge": faithfulness,
    }


def judge_faithfulness(rag_system, question: str, answer: str, context: str) -> Optional[float]:
    """Return an optional model-judged faithfulness score in [0, 1]."""
    if not answer.strip() or not context.strip():
        return None
    system_prompt = (
        "You are a strict grading assistant. Given a question, retrieved code context, and a "
        "generated answer, output ONLY a single number between 0 and 1 for how faithful the "
        "answer is to the context (1.0 = every claim is supported, 0.0 = the answer invents or "
        "contradicts facts not in the context). Output just the number."
    )
    user_prompt = (
        f"Question: {question}\n\nContext:\n{context}\n\n"
        f"Answer:\n{answer}\n\nFaithfulness score:"
    )
    try:
        response, _ = rag_system._generate_markdown_response(system_prompt, user_prompt)
        match = re.search(r"(?<![\w.])(?:0(?:\.\d+)?|1(?:\.0+)?)(?![\w.])", response)
        if not match:
            return None
        return max(0.0, min(1.0, float(match.group(0))))
    except Exception:
        return None


def summarize_answer_quality(records: list[dict]) -> dict:
    """Summarize answer and latency signals outside retrieval headlines."""
    if not records:
        return {"case_count": 0}
    scores = [
        record["faithfulness_judge"]
        for record in records
        if record["faithfulness_judge"] is not None
    ]
    latencies = sorted(record["latency_ms"] for record in records)
    p95_index = min(len(latencies) - 1, int(len(latencies) * 0.95))
    return {
        "case_count": len(records),
        "citation_present_rate": round(
            mean(record["citation_present"] for record in records), 4
        ),
        "expected_source_retrieved_rate": round(
            mean(record["expected_source_retrieved"] for record in records), 4
        ),
        "grounded_answer_proxy_rate": round(
            mean(record["grounded_answer_proxy"] for record in records), 4
        ),
        "faithfulness_judge_mean": round(mean(scores), 4) if scores else None,
        "faithfulness_judge_coverage": len(scores),
        "latency_p95_ms": round(latencies[p95_index], 1),
    }


DIAGNOSTIC_STAGES = (
    ("semantic", "semantic_rank"),
    ("lexical", "bm25_rank"),
    ("fused", "fused_rank"),
    ("path", "path_rank"),
    ("reranker", "rerank_rank"),
    ("prioritized", "prioritized_rank"),
    ("final", "final_rank"),
)


def stage_diagnostics(expected_sources: list[str], retrieval_debug: list[dict]) -> dict:
    """Score retrieval stages for debugging; these values are not headline metrics."""
    diagnostics = {}
    for stage, rank_key in DIAGNOSTIC_STAGES:
        ranked = sorted(
            (
                (item[rank_key], item.get("file_path", ""))
                for item in retrieval_debug
                if item.get(rank_key) is not None
            ),
            key=lambda row: row[0],
        )
        diagnostics[stage] = {
            "candidate_count": len(ranked),
            "ranking_metrics": score_ranking(
                expected_sources,
                [file_path for _, file_path in ranked],
            ),
        }

    candidate_hits = [
        item
        for item in retrieval_debug
        if matches_expected(item.get("file_path", ""), expected_sources)
        and any(
            item.get(rank_key) is not None
            for rank_key in ("semantic_rank", "bm25_rank", "path_rank")
        )
    ]
    diagnostics["candidate_recall"] = {
        "candidate_count": len(retrieval_debug),
        "expected_source_candidate_count": len(candidate_hits),
        "expected_source_found": int(bool(candidate_hits)),
    }
    return diagnostics


def summarize_diagnostics(records: list[dict]) -> dict:
    """Aggregate stage and candidate signals under a diagnostics-only section."""
    if not records:
        return {"case_count": 0}
    stage_summary = {}
    for stage, _ in DIAGNOSTIC_STAGES:
        stage_summary[stage] = {
            "hit_at_8": round(
                mean(
                    record["stage_scores"][stage]["ranking_metrics"]["hit_at_8"]
                    for record in records
                ),
                4,
            ),
            "mrr_at_8": round(
                mean(
                    record["stage_scores"][stage]["ranking_metrics"]["mrr_at_8"]
                    for record in records
                ),
                4,
            ),
            "mean_candidate_count": round(
                mean(record["stage_scores"][stage]["candidate_count"] for record in records),
                2,
            ),
        }
    return {
        "case_count": len(records),
        "candidate_recall_rate": round(
            mean(
                record["stage_scores"]["candidate_recall"]["expected_source_found"]
                for record in records
            ),
            4,
        ),
        "mean_candidate_count": round(
            mean(
                record["stage_scores"]["candidate_recall"]["candidate_count"]
                for record in records
            ),
            2,
        ),
        "stages": stage_summary,
    }
