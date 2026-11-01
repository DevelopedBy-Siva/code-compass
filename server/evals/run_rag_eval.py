import json
import os
import sys
import time
from pathlib import Path

import boto3

SERVER_ROOT = Path(__file__).resolve().parents[1]
if str(SERVER_ROOT) not in sys.path:
    sys.path.insert(0, str(SERVER_ROOT))

from evals.metrics import (
    answer_quality_signals,
    judge_faithfulness,
    matches_expected,
    score_ranking,
    stage_diagnostics,
    summarize_answer_quality,
    summarize_diagnostics,
)
from evals.run_eval import (
    EVAL_REPO_IDS,
    EVAL_SESSION_KEY,
    EVAL_SET_PATH,
    TOP_K,
    choose_reindex,
    get_cached_vector_count,
    load_eval_set,
    log,
    restore_or_index_repository,
    validate_eval_set,
)
from src.rag_system import BEDROCK_QWEN_MODEL_ID, CodebaseRAGSystem


ENABLE_FAITHFULNESS = os.getenv("CODEBASE_RAG_ENABLE_FAITHFULNESS", "1") == "1"
OUTPUT_PATH = os.getenv("CODEBASE_RAG_RAG_EVAL_OUTPUT")


def check_llm_available() -> None:
    region = os.getenv("AWS_REGION", os.getenv("AWS_DEFAULT_REGION", "us-east-1"))
    client = boto3.client("bedrock-runtime", region_name=region)
    try:
        response = client.converse(
            modelId=BEDROCK_QWEN_MODEL_ID,
            messages=[{"role": "user", "content": [{"text": "Reply with exactly: ok"}]}],
            inferenceConfig={"temperature": 0.0, "maxTokens": 8},
        )
    except Exception as exc:
        raise RuntimeError(f"Bedrock LLM check failed: {exc}") from exc
    content = response.get("output", {}).get("message", {}).get("content", [])
    if not "".join(block.get("text", "") for block in content).strip():
        raise RuntimeError("Bedrock LLM check returned an empty response")


def evaluate_case(rag_system, repo_id: int, case: dict) -> dict:
    started = time.perf_counter()
    result = rag_system.answer_question(
        repo_id=repo_id,
        session_key=EVAL_SESSION_KEY,
        question=case["question"],
        top_k=TOP_K,
        history=case.get("turns", []),
        debug_retrieval=True,
        retrieval_only=False,
    )
    latency_ms = (time.perf_counter() - started) * 1000
    expected_sources = case["expected_sources"]
    final_paths = [source.get("file_path", "") for source in result.get("sources", [])]
    final_metrics = score_ranking(expected_sources, final_paths)
    trace = result.get("conversation_trace", {})
    faithfulness = (
        judge_faithfulness(
            rag_system,
            case["question"],
            result.get("answer", ""),
            trace.get("generation_context", ""),
        )
        if ENABLE_FAITHFULNESS
        else None
    )
    answer_signals = answer_quality_signals(
        case,
        {**result, "final_retrieval_hit": final_metrics["hit_at_8"]},
        faithfulness,
    )
    debug = result.get("retrieval_debug", [])
    for item in debug:
        item["expected_source"] = matches_expected(
            item.get("file_path", ""), expected_sources
        )

    return {
        "id": case.get("id", case["question"]),
        "repo": case["repo"],
        "category": case.get("category", "general"),
        "question": case["question"],
        "ground_truth": case["ground_truth"],
        "answer": result.get("answer", ""),
        "citations": result.get("citations", []),
        "expected_sources": expected_sources,
        "retrieved_sources": final_paths,
        "answer_quality": {
            **answer_signals,
            "latency_ms": round(latency_ms, 1),
        },
        "diagnostics": {
            "stage_scores": stage_diagnostics(expected_sources, debug),
            "retrieval_trace": debug,
            "conversation_trace": {
                key: value
                for key, value in trace.items()
                if key != "generation_context"
            },
        },
}


def run() -> dict:
    repositories = load_eval_set()["repositories"]
    errors = validate_eval_set(repositories)
    if errors:
        raise RuntimeError("Eval set validation failed: " + "; ".join(errors))
    if EVAL_REPO_IDS:
        available = {repo["id"].lower() for repo in repositories}
        unknown = EVAL_REPO_IDS - available
        if unknown:
            raise RuntimeError("Unknown CODEBASE_RAG_EVAL_REPOS: " + ", ".join(sorted(unknown)))
        repositories = [repo for repo in repositories if repo["id"].lower() in EVAL_REPO_IDS]

    check_llm_available()
    reindex = choose_reindex(get_cached_vector_count())
    rag_system = CodebaseRAGSystem(clear_existing_index=reindex)
    records = []
    indexed_repositories = []
    try:
        for repo_number, repository in enumerate(repositories, start=1):
            repo_id = restore_or_index_repository(
                rag_system, repository, repo_number, reindex
            )
            repo_state = rag_system.repositories[repo_id]
            indexed_repositories.append(
                {
                    "id": repository["id"],
                    "github_url": repository["github_url"],
                    "branch": repo_state.branch,
                    "cache_generation": repo_state.cache_generation,
                    "cache_hit": repo_state.cache_hit,
                    "chunk_count": repo_state.chunk_count,
                }
            )
            for case in repository["cases"]:
                records.append(
                    evaluate_case(
                        rag_system,
                        repo_id,
                        {**case, "repo": repository["id"]},
                    )
                )
    finally:
        rag_system.reset_session_state()

    answer_records = [record["answer_quality"] for record in records]
    diagnostic_records = [
        {"stage_scores": record["diagnostics"]["stage_scores"]}
        for record in records
    ]
    by_repository = {}
    diagnostics_by_repository = {}
    for record in records:
        by_repository.setdefault(record["repo"], []).append(record["answer_quality"])
        diagnostics_by_repository.setdefault(record["repo"], []).append(
            {"stage_scores": record["diagnostics"]["stage_scores"]}
        )
    report = {
        "config": {
            "evaluation_set": str(EVAL_SET_PATH),
            "query_count": len(records),
            "top_k": TOP_K,
            "faithfulness_enabled": ENABLE_FAITHFULNESS,
            "vector_collection": rag_system.vector_store.collection_name,
            "qdrant_upsert_batch_size": rag_system.vector_store.upsert_batch_size,
            "embedding_model": rag_system.embedder.model_name,
            "embedding_batch_size": rag_system.embedder.batch_size,
            "reranker_model": rag_system.hybrid_search.reranker_model_name,
            "reranker_batch_size": rag_system.hybrid_search.rerank_batch_size,
            "answer_model": BEDROCK_QWEN_MODEL_ID,
            "repository_snapshots": indexed_repositories,
        },
        "answer_quality": {
            "overall": summarize_answer_quality(answer_records),
            "per_repository": {
                repo: summarize_answer_quality(items)
                for repo, items in by_repository.items()
            },
        },
        "diagnostics_summary": {
            "overall": summarize_diagnostics(diagnostic_records),
            "per_repository": {
                repo: summarize_diagnostics(items)
                for repo, items in diagnostics_by_repository.items()
            },
        },
        "cases": records,
    }
    if OUTPUT_PATH:
        target = Path(OUTPUT_PATH)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(report, indent=2), encoding="utf-8")
        log(f"Wrote RAG/answer-quality report to {target}")
    print(json.dumps(report, indent=2))
    return report


if __name__ == "__main__":
    run()
