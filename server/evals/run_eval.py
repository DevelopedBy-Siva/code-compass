import json
import os
import re
import sys
import time
from pathlib import Path
from statistics import mean

import boto3
from dotenv import load_dotenv

SERVER_ROOT = Path(__file__).resolve().parents[1]
if str(SERVER_ROOT) not in sys.path:
    sys.path.insert(0, str(SERVER_ROOT))

load_dotenv(SERVER_ROOT / ".env")

from src.rag_system import BEDROCK_QWEN_MODEL_ID, CodebaseRAGSystem
from src.vector_store import ChromaVectorStore

EVAL_SESSION_KEY = "eval-session"
TOP_K = int(os.getenv("CODEBASE_RAG_TOP_K", "8"))
EVAL_SET_PATH = Path(
    os.getenv("CODEBASE_RAG_EVAL_SET", Path(__file__).with_name("sample_eval_set.json"))
)
EVAL_OUTPUT_PATH = os.getenv("CODEBASE_RAG_EVAL_OUTPUT")
ENABLE_FAITHFULNESS = os.getenv("CODEBASE_RAG_ENABLE_FAITHFULNESS", "1") == "1"
EVAL_REPO_IDS = {
    value.strip().lower()
    for value in os.getenv("CODEBASE_RAG_EVAL_REPOS", "").split(",")
    if value.strip()
}


def log(message: str):
    print(f"[eval] {message}", file=sys.stderr, flush=True)


def load_eval_set():
    return json.loads(EVAL_SET_PATH.read_text())


def validate_eval_set(repositories):
    errors = []
    seen_ids = set()
    for repo in repositories:
        if not repo.get("github_url"):
            errors.append(f"repo {repo.get('id', '?')}: missing github_url")
        if not repo.get("cases"):
            errors.append(f"repo {repo.get('id', '?')}: has no cases")
        for case in repo.get("cases", []):
            case_id = case.get("id") or case.get("question", "?")
            if case_id in seen_ids:
                errors.append(f"duplicate case id: {case_id}")
            seen_ids.add(case_id)
            if not case.get("question", "").strip():
                errors.append(f"{case_id}: missing question")
            if not case.get("ground_truth", "").strip():
                errors.append(f"{case_id}: missing ground_truth")
            if not case.get("expected_sources"):
                errors.append(f"{case_id}: expected_sources must be a non-empty list")
    return errors


def check_llm_available():
    log(f"Checking Bedrock LLM availability: {BEDROCK_QWEN_MODEL_ID}")
    client = boto3.client(
        "bedrock-runtime",
        region_name=os.getenv("AWS_REGION", os.getenv("AWS_DEFAULT_REGION", "us-east-1")),
    )
    try:
        response = client.converse(
            modelId=BEDROCK_QWEN_MODEL_ID,
            messages=[
                {
                    "role": "user",
                    "content": [{"text": "Reply with exactly: ok"}],
                }
            ],
            inferenceConfig={
                "temperature": 0.0,
                "maxTokens": 8,
            },
        )
    except Exception as exc:
        raise RuntimeError(f"Bedrock LLM check failed: {exc}") from exc

    content_blocks = (
        response.get("output", {})
        .get("message", {})
        .get("content", [])
    )
    text = "".join(block.get("text", "") for block in content_blocks).strip()
    if not text:
        raise RuntimeError("Bedrock LLM check returned an empty response.")
    log("Bedrock LLM check passed")


def should_reindex_existing_cache(vector_count: int) -> bool:
    if vector_count <= 0:
        log("No existing Chroma embeddings found; eval will index repositories.")
        return True

    env_choice = os.getenv("CODEBASE_RAG_REINDEX")
    if env_choice is not None:
        return env_choice.strip().lower() in {"1", "true", "yes", "y"}

    prompt = (
        f"Found {vector_count} existing Chroma embeddings. "
        "Restart/re-index from scratch? [y/N]: "
    )
    if not sys.stdin.isatty():
        log("Existing embeddings found; non-interactive run defaults to re-index=no.")
        return False

    answer = input(prompt).strip().lower()
    return answer in {"y", "yes"}


def get_cached_vector_count() -> int:
    store = ChromaVectorStore(embedding_dim=0, persist=True)
    return store.get_stats()["total_vectors"]


def normalize_path(path: str) -> str:
    return path.strip().lstrip("./").lower()


def tokenize_text(text: str):
    return re.findall(r"[a-z0-9_]+", (text or "").lower())


def matches_expected(actual_path: str, expected_sources) -> bool:
    actual = normalize_path(actual_path)
    for expected in expected_sources:
        expected_norm = normalize_path(expected).rstrip("/")
        is_dir = "." not in expected_norm.rsplit("/", 1)[-1]
        if actual == expected_norm:
            return True
        if is_dir and actual.startswith(expected_norm + "/"):
            return True
    return False


def compute_retrieval_metrics(expected_sources, actual_sources):
    matching_ranks = [
        rank
        for rank, path in enumerate(actual_sources, start=1)
        if matches_expected(path, expected_sources)
    ]
    hit = bool(matching_ranks)
    top1 = bool(matching_ranks and matching_ranks[0] == 1)
    reciprocal_rank = 1.0 / matching_ranks[0] if matching_ranks else 0.0
    return {
        "retrieval_hit": int(hit),
        "top1_hit": int(top1),
        "reciprocal_rank": reciprocal_rank,
    }


def compute_debug_stage_metrics(expected_sources, retrieval_debug, rank_key, rank_limit=None):
    ranked = sorted(
        (
            (item.get(rank_key), item.get("file_path", ""))
            for item in retrieval_debug
            if item.get(rank_key) is not None
            and (rank_limit is None or item.get(rank_key) <= rank_limit)
        ),
        key=lambda pair: pair[0],
    )
    return compute_retrieval_metrics(
        expected_sources,
        [file_path for _, file_path in ranked],
    )


def keyword_hits(answer: str, keywords):
    if not keywords:
        return 0, 0
    tokens = set(tokenize_text(answer))
    matched = 0
    for keyword in keywords:
        keyword_tokens = tokenize_text(keyword)
        if keyword_tokens and all(token in tokens for token in keyword_tokens):
            matched += 1
    return matched, len(keywords)


def judge_faithfulness(rag_system, question: str, answer: str, sources: list):
    if not ENABLE_FAITHFULNESS or not answer.strip() or not sources:
        return None
    context = "\n\n".join(
        f"[{i}] {source['file_path']}\n{source['snippet'][:1500]}"
        for i, source in enumerate(sources, start=1)
    )
    system_prompt = (
        "You are a strict grading assistant. Given a question, retrieved code context, and a "
        "generated answer, output ONLY a single number between 0 and 1 for how faithful the "
        "answer is to the context (1.0 = every claim is supported, 0.0 = the answer invents or "
        "contradicts facts not in the context). Output just the number."
    )
    user_prompt = f"Question: {question}\n\nContext:\n{context}\n\nAnswer:\n{answer}\n\nFaithfulness score:"
    try:
        text, _ = rag_system._generate_markdown_response(system_prompt, user_prompt)
        match = re.search(r"(\d(?:\.\d+)?)", text)
        if not match:
            return None
        return max(0.0, min(1.0, float(match.group(1))))
    except Exception as exc:
        log(f"Faithfulness judge failed: {exc}")
        return None


def index_repo(rag_system, github_url: str, name: str):
    repo = rag_system.create_or_reset_repository(github_url, EVAL_SESSION_KEY)
    log(f"Indexing {name} ({github_url}), repo_id={repo.id}")
    rag_system.index_repository(repo.id)
    repo_state = rag_system.get_repository_for_session(repo.id, EVAL_SESSION_KEY)
    if not repo_state or repo_state["status"] != "indexed":
        detail = repo_state.get("error_message") if repo_state else "repository disappeared"
        raise RuntimeError(f"Failed to index {name}: {detail}")
    log(
        f"Indexed {name}: {repo_state['file_count']} files, "
        f"{repo_state['chunk_count']} chunks"
    )
    return repo.id


def restore_or_index_repo(
    rag_system,
    repo_config: dict,
    expected_repo_id: int,
    reindex: bool,
) -> int:
    if not reindex:
        restored_repo_id = rag_system.restore_repository_from_cache(
            repo_config["github_url"],
            EVAL_SESSION_KEY,
            expected_repo_id,
        )
        if restored_repo_id is not None:
            repo_state = rag_system.get_repository_for_session(
                restored_repo_id,
                EVAL_SESSION_KEY,
            )
            log(
                f"Using cached embeddings for {repo_config['name']}: "
                f"repo_id={restored_repo_id}, chunks={repo_state['chunk_count']}"
            )
            return restored_repo_id

        log(
            f"No cached embeddings found for {repo_config['name']} "
            f"at repo_id={expected_repo_id}; indexing it now."
        )

    return index_repo(rag_system, repo_config["github_url"], repo_config["name"])


def run_case(rag_system, repo_id: int, repo_name: str, case: dict):
    start = time.time()
    result = rag_system.answer_question(
        repo_id=repo_id,
        session_key=EVAL_SESSION_KEY,
        question=case["question"],
        top_k=TOP_K,
        history=case.get("turns", []),
        debug_retrieval=True,
    )
    elapsed_ms = (time.time() - start) * 1000

    sources = result.get("sources", [])
    cited_paths = [source["file_path"] for source in sources]
    retrieval = compute_retrieval_metrics(case.get("expected_sources", []), cited_paths)
    matched, total_keywords = keyword_hits(result.get("answer", ""), case.get("must_include_any", []))
    has_citations = bool(result.get("citations"))
    expected_source_grounded = (
        retrieval["retrieval_hit"] == 1
        and has_citations
        and (total_keywords == 0 or matched > 0)
    )

    retrieval_debug = result.get("retrieval_debug", [])
    for item in retrieval_debug:
        item["expected_source"] = matches_expected(
            item.get("file_path", ""),
            case.get("expected_sources", []),
        )

    expected_sources = case.get("expected_sources", [])
    stage_metrics = {
        stage: compute_debug_stage_metrics(
            expected_sources,
            retrieval_debug,
            rank_key,
            rank_limit,
        )
        for stage, rank_key, rank_limit in (
            ("semantic", "semantic_rank", None),
            ("lexical", "bm25_rank", None),
            ("fused", "fused_rank", None),
            ("path", "path_rank", None),
            ("reranker", "rerank_rank", None),
            ("reranker_top_k", "rerank_rank", TOP_K),
            ("prioritized", "prioritized_rank", None),
            ("prioritized_top_k", "prioritized_rank", TOP_K),
        )
    }
    candidate_retrieval_hit = int(
        any(
            item.get("expected_source")
            and any(
                item.get(rank_key) is not None
                for rank_key in ("semantic_rank", "bm25_rank", "path_rank")
            )
            for item in retrieval_debug
        )
    )

    return {
        "id": case.get("id", case["question"]),
        "repo": repo_name,
        "category": case.get("category", "general"),
        "question": case["question"],
        "answer": result.get("answer", ""),
        "citations": result.get("citations", []),
        "expected_sources": case.get("expected_sources", []),
        "retrieved_sources": cited_paths,
        "retrieval_hit": retrieval["retrieval_hit"],
        "top1_hit": retrieval["top1_hit"],
        "reciprocal_rank": round(retrieval["reciprocal_rank"], 4),
        "candidate_retrieval_hit": candidate_retrieval_hit,
        **{
            f"{stage}_hit": metrics["retrieval_hit"]
            for stage, metrics in stage_metrics.items()
        },
        **{
            f"{stage}_reciprocal_rank": round(metrics["reciprocal_rank"], 4)
            for stage, metrics in stage_metrics.items()
        },
        "expected_source_grounded": int(expected_source_grounded),
        "grounded": int(expected_source_grounded),
        "retrieval_debug": retrieval_debug,
        "faithfulness": judge_faithfulness(rag_system, case["question"], result.get("answer", ""), sources),
        "latency_ms": round(elapsed_ms, 1),
    }


def summarize(details):
    if not details:
        return {}
    latencies = sorted(item["latency_ms"] for item in details)
    p95_index = min(len(latencies) - 1, int(len(latencies) * 0.95))
    faith_scores = [item["faithfulness"] for item in details if item["faithfulness"] is not None]
    return {
        "case_count": len(details),
        "candidate_retrieval_hit_rate": round(
            mean(item["candidate_retrieval_hit"] for item in details), 4
        ),
        "semantic_hit_rate": round(mean(item["semantic_hit"] for item in details), 4),
        "lexical_hit_rate": round(mean(item["lexical_hit"] for item in details), 4),
        "fused_hit_rate": round(mean(item["fused_hit"] for item in details), 4),
        "reranker_hit_rate": round(mean(item["reranker_hit"] for item in details), 4),
        "reranker_top_k_hit_rate": round(
            mean(item["reranker_top_k_hit"] for item in details), 4
        ),
        "prioritized_hit_rate": round(mean(item["prioritized_hit"] for item in details), 4),
        "prioritized_top_k_hit_rate": round(
            mean(item["prioritized_top_k_hit"] for item in details), 4
        ),
        "retrieval_hit_rate": round(mean(item["retrieval_hit"] for item in details), 4),
        "final_context_hit_rate": round(mean(item["retrieval_hit"] for item in details), 4),
        "top1_hit_rate": round(mean(item["top1_hit"] for item in details), 4),
        "mrr": round(mean(item["reciprocal_rank"] for item in details), 4),
        "expected_source_grounded_rate": round(
            mean(item["expected_source_grounded"] for item in details), 4
        ),
        "grounded_answer_rate": round(mean(item["grounded"] for item in details), 4),
        "faithfulness": round(mean(faith_scores), 4) if faith_scores else None,
        "latency_p95_ms": round(latencies[p95_index], 1),
    }


def summarize_by_repo(details):
    grouped = {}
    for item in details:
        grouped.setdefault(item["repo"], []).append(item)
    return {repo: summarize(items) for repo, items in grouped.items()}


def summarize_by_category(details):
    grouped = {}
    for item in details:
        grouped.setdefault(item["category"], []).append(item)
    return {category: summarize(items) for category, items in sorted(grouped.items())}


def run():
    eval_set = load_eval_set()
    repositories = eval_set["repositories"]

    errors = validate_eval_set(repositories)
    if errors:
        raise RuntimeError("Eval set validation failed: " + "; ".join(errors))

    if EVAL_REPO_IDS:
        available_ids = {repo["id"].lower() for repo in repositories}
        unknown_ids = EVAL_REPO_IDS - available_ids
        if unknown_ids:
            raise RuntimeError(
                "Unknown CODEBASE_RAG_EVAL_REPOS values: "
                + ", ".join(sorted(unknown_ids))
                + ". Available repository ids: "
                + ", ".join(sorted(available_ids))
            )
        repositories = [
            repo for repo in repositories if repo["id"].lower() in EVAL_REPO_IDS
        ]

    total_cases = sum(len(repo["cases"]) for repo in repositories)
    log(f"Loaded eval set: {len(repositories)} repositories, {total_cases} cases")

    check_llm_available()
    reindex = should_reindex_existing_cache(get_cached_vector_count())

    rag_system = CodebaseRAGSystem(clear_existing_index=reindex)
    log(f"LLM provider={rag_system.llm_provider} model={rag_system.llm_model}")
    log(f"Re-index from scratch={reindex}")

    details = []
    try:
        for expected_repo_id, repo_config in enumerate(repositories, start=1):
            repo_id = restore_or_index_repo(
                rag_system,
                repo_config,
                expected_repo_id,
                reindex,
            )
            cases = repo_config["cases"]
            for index, case in enumerate(cases, start=1):
                log(f"[{repo_config['id']} {index}/{len(cases)}] {case['id']}")
                details.append(run_case(rag_system, repo_id, repo_config["name"], case))
    finally:
        rag_system.reset_session_state()

    report = {
        "config": {
            "llm_provider": rag_system.llm_provider,
            "llm_model": rag_system.llm_model,
            "embedding_model": rag_system.embedder.model_name,
            "embedding_dimension": rag_system.embedder.get_embedding_dim(),
            "reranker_model": rag_system.hybrid_search.reranker_model_name,
            "embedding_batch_size": rag_system.embedder.batch_size,
            "reranker_batch_size": rag_system.hybrid_search.rerank_batch_size,
            "top_k": TOP_K,
            "final_source_limit": max(
                1,
                min(
                    TOP_K,
                    int(os.getenv("RAG_FINAL_SOURCE_LIMIT", str(TOP_K))),
                ),
            ),
            "eval_set": str(EVAL_SET_PATH),
            "repositories": [
                {"id": repo["id"], "name": repo["name"], "github_url": repo["github_url"]}
                for repo in repositories
            ],
        },
        "headline_metrics": summarize(details),
        "repo_breakdown": summarize_by_repo(details),
        "category_breakdown": summarize_by_category(details),
        "cases": details,
    }

    if EVAL_OUTPUT_PATH:
        target = Path(EVAL_OUTPUT_PATH)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(report, indent=2))
        log(f"Wrote JSON report to {target}")

    log("Eval complete. Printing JSON report.")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    run()
