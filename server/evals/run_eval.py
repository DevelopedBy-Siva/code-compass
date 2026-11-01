import json
import os
import sys
from pathlib import Path

from dotenv import load_dotenv


SERVER_ROOT = Path(__file__).resolve().parents[1]
REPOSITORY_ROOT = SERVER_ROOT.parent
if str(SERVER_ROOT) not in sys.path:
    sys.path.insert(0, str(SERVER_ROOT))

load_dotenv(SERVER_ROOT / ".env")

# Keep evaluation indexing isolated from the application collection.
os.environ["QDRANT_COLLECTION"] = os.getenv(
    "QDRANT_EVAL_COLLECTION",
    "code_compass_eval_qwen3_embedding_0_6b_last_token_cache_v2",
)

from evals.metrics import score_ranking, summarize_comparison
from src.rag_system import CodebaseRAGSystem
from src.vector_store import QdrantVectorStore


EVAL_SESSION_KEY = "eval-session"
TOP_K = 8
EVAL_SET_PATH = Path(
    os.getenv("CODEBASE_RAG_EVAL_SET", Path(__file__).with_name("sample_eval_set.json"))
)
EVAL_OUTPUT_PATH = os.getenv("CODEBASE_RAG_EVAL_OUTPUT")
EVAL_REPO_IDS = {
    value.strip().lower()
    for value in os.getenv("CODEBASE_RAG_EVAL_REPOS", "").split(",")
    if value.strip()
}


def log(message: str) -> None:
    print(f"[eval] {message}", file=sys.stderr, flush=True)


def load_eval_set() -> dict:
    return json.loads(EVAL_SET_PATH.read_text(encoding="utf-8"))


def validate_eval_set(repositories: list[dict]) -> list[str]:
    errors = []
    seen_ids = set()
    for repository in repositories:
        repo_id = repository.get("id", "?")
        if not repository.get("github_url"):
            errors.append(f"repo {repo_id}: missing github_url")
        cases = repository.get("cases", [])
        if not cases:
            errors.append(f"repo {repo_id}: has no cases")
        for case in cases:
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


def retrieve_vector_only(rag_system, repository_key: str, query: str) -> list[str]:
    """Run dense retrieval for the shared normalized query."""
    embedding = rag_system.embedder.embed_text(query)
    hits = rag_system.vector_store.search(
        embedding,
        k=TOP_K,
        repository_key=repository_key,
    )
    return [metadata.get("file_path", "") for _, metadata in hits]


def get_cached_vector_count() -> int:
    return QdrantVectorStore(embedding_dim=0).get_stats()["total_vectors"]


def choose_reindex(vector_count: int) -> bool:
    if vector_count <= 0:
        log("No existing evaluation vectors found; repositories will be indexed.")
        return True
    env_choice = os.getenv("CODEBASE_RAG_REINDEX")
    if env_choice is not None:
        return env_choice.strip().lower() in {"1", "true", "yes", "y"}
    if not sys.stdin.isatty():
        log("Existing vectors found; non-interactive run defaults to reindex=no.")
        return False
    answer = input(f"Found {vector_count} evaluation vectors. Re-index? [y/N]: ")
    return answer.strip().lower() in {"y", "yes"}


def index_repository(rag_system, repository: dict) -> int:
    repo = rag_system.create_or_reset_repository(
        repository["github_url"], EVAL_SESSION_KEY
    )
    log(f"Indexing {repository['name']} ({repository['github_url']})")
    rag_system.index_repository(repo.id)
    state = rag_system.get_repository_for_session(repo.id, EVAL_SESSION_KEY)
    if not state or state["status"] != "indexed":
        detail = state.get("error_message") if state else "repository disappeared"
        raise RuntimeError(f"Failed to index {repository['name']}: {detail}")
    log(
        f"Indexed {repository['name']}: {state['file_count']} files, "
        f"{state['chunk_count']} chunks"
    )
    return repo.id


def restore_or_index_repository(
    rag_system, repository: dict, expected_repo_id: int, reindex: bool
) -> int:
    if not reindex:
        restored_id = rag_system.restore_repository_from_cache(
            repository["github_url"], EVAL_SESSION_KEY, expected_repo_id
        )
        if restored_id is not None:
            log(f"Using cached evaluation index for {repository['name']}")
            return restored_id
    return index_repository(rag_system, repository)


def evaluate_case(rag_system, repo_id: int, repository_key: str, case: dict) -> dict:
    expected_sources = case["expected_sources"]
    hybrid_result = rag_system.answer_question(
        repo_id=repo_id,
        session_key=EVAL_SESSION_KEY,
        question=case["question"],
        top_k=TOP_K,
        history=case.get("turns", []),
        retrieval_only=True,
        debug_retrieval=True,
    )
    conversation_trace = hybrid_result.get("conversation_trace", {})
    shared_query = conversation_trace.get("retrieval_query") or case["question"]
    vector_paths = retrieve_vector_only(rag_system, repository_key, shared_query)
    hybrid_paths = [
        source.get("file_path", "") for source in hybrid_result.get("sources", [])
    ]

    return {
        "id": case.get("id", case["question"]),
        "repo": case.get("repo"),
        "question": case["question"],
        "retrieval_query": shared_query,
        "ground_truth": case["ground_truth"],
        "expected_sources": expected_sources,
        "rankings": {
            "vector_only": vector_paths,
            "hybrid_rerank": hybrid_paths,
        },
        "metrics": {
            "vector_only": score_ranking(expected_sources, vector_paths),
            "hybrid_rerank": score_ranking(expected_sources, hybrid_paths),
        },
    }


def summarize_by_repository(records: list[dict]) -> dict:
    grouped: dict[str, list[dict]] = {}
    for record in records:
        grouped.setdefault(record["repo"], []).append(record)
    return {
        repository: summarize_comparison(repository_records)
        for repository, repository_records in grouped.items()
    }


def run() -> dict:
    eval_set = load_eval_set()
    repositories = eval_set["repositories"]
    errors = validate_eval_set(repositories)
    if errors:
        raise RuntimeError("Eval set validation failed: " + "; ".join(errors))

    if EVAL_REPO_IDS:
        available = {repository["id"].lower() for repository in repositories}
        unknown = EVAL_REPO_IDS - available
        if unknown:
            raise RuntimeError(
                "Unknown CODEBASE_RAG_EVAL_REPOS values: "
                + ", ".join(sorted(unknown))
                + ". Available repository ids: "
                + ", ".join(sorted(available))
            )
        repositories = [
            repository
            for repository in repositories
            if repository["id"].lower() in EVAL_REPO_IDS
        ]

    query_count = sum(len(repository["cases"]) for repository in repositories)
    bundled_eval_set = (
        EVAL_SET_PATH.resolve()
        == Path(__file__).with_name("sample_eval_set.json").resolve()
    )
    if bundled_eval_set and not EVAL_REPO_IDS:
        repository_counts = {
            repository["id"]: len(repository["cases"])
            for repository in repositories
        }
        if query_count != 120 or repository_counts != {
            "documenso": 40,
            "fastapi": 40,
            "django": 40,
        }:
            raise RuntimeError(
                "The bundled primary benchmark must contain 120 queries: "
                "40 each for Documenso, FastAPI, and Django."
            )
    log(f"Loaded {query_count} queries across {len(repositories)} repositories")
    final_source_limit = int(os.getenv("RAG_FINAL_SOURCE_LIMIT", str(TOP_K)))
    if final_source_limit < TOP_K:
        raise RuntimeError(
            f"RAG_FINAL_SOURCE_LIMIT must be at least {TOP_K} for this benchmark"
        )
    reindex = choose_reindex(get_cached_vector_count())
    rag_system = CodebaseRAGSystem(clear_existing_index=reindex)
    log(f"Re-index from scratch={reindex}")

    records = []
    indexed_repositories = []
    try:
        for repo_number, repository in enumerate(repositories, start=1):
            repo_id = restore_or_index_repository(
                rag_system, repository, repo_number, reindex
            )
            repo_state = rag_system.repositories[repo_id]
            repository_key = repo_state.repository_key
            indexed_repositories.append(
                {
                    "id": repository["id"],
                    "name": repository["name"],
                    "github_url": repository["github_url"],
                    "branch": repo_state.branch,
                    "cache_generation": repo_state.cache_generation,
                    "cache_hit": repo_state.cache_hit,
                    "chunk_count": repo_state.chunk_count,
                }
            )
            for index, case in enumerate(repository["cases"], start=1):
                log(f"[{repository['id']} {index}/{len(repository['cases'])}] {case['id']}")
                case_record = {**case, "repo": repository["id"]}
                records.append(
                    evaluate_case(
                        rag_system,
                        repo_id,
                        repository_key,
                        case_record,
                    )
                )
    finally:
        rag_system.reset_session_state()

    report = {
        "config": {
            "evaluation_set": str(EVAL_SET_PATH.relative_to(REPOSITORY_ROOT))
            if EVAL_SET_PATH.is_relative_to(REPOSITORY_ROOT)
            else str(EVAL_SET_PATH),
            "query_count": query_count,
            "full_120_query_benchmark": bundled_eval_set and not EVAL_REPO_IDS,
            "top_k": TOP_K,
            "metric_cutoffs": [1, 5, 8],
            "reindexed": reindex,
            "vector_store": "qdrant",
            "vector_collection": rag_system.vector_store.collection_name,
            "qdrant_upsert_batch_size": rag_system.vector_store.upsert_batch_size,
            "qdrant_timeout_seconds": rag_system.vector_store.timeout_seconds,
            "embedding_model": rag_system.embedder.model_name,
            "embedding_dimension": rag_system.embedder.get_embedding_dim(),
            "reranker_model": rag_system.hybrid_search.reranker_model_name,
            "embedding_batch_size": rag_system.embedder.batch_size,
            "reranker_batch_size": rag_system.hybrid_search.rerank_batch_size,
            "search_multiplier": int(os.getenv("RAG_SEARCH_MULTIPLIER", "4")),
            "deep_search_multiplier": int(os.getenv("RAG_DEEP_SEARCH_MULTIPLIER", "8")),
            "final_source_limit": min(TOP_K, final_source_limit),
            "repositories": indexed_repositories,
        },
        "overall": summarize_comparison(records),
        "per_repository": summarize_by_repository(records),
        "cases": records,
    }

    if EVAL_OUTPUT_PATH:
        output_path = Path(EVAL_OUTPUT_PATH)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        log(f"Wrote JSON report to {output_path}")
    print(json.dumps(report, indent=2))
    return report


if __name__ == "__main__":
    run()
