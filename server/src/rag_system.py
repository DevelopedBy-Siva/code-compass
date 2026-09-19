import json
import os
import re
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from threading import Lock, RLock
from typing import Dict, List, Optional
from uuid import uuid4

import boto3
from botocore.config import Config as BotoConfig

from src.app_logging import fields, get_logger
from src.code_parser import CodeParser
from src.config import Settings
from src.embeddings import EmbeddingGenerator
from src.generation_context import GenerationContextBuilder
from src.hybrid_search import HybridSearchEngine
from src.repo_fetcher import RepoFetcher
from src.vector_store import QdrantVectorStore

BEDROCK_QWEN_MODEL_ID = os.getenv("BEDROCK_MODEL_ID", "qwen.qwen3-coder-next")
index_logger = get_logger("index")
query_logger = get_logger("query")
error_logger = get_logger("error")


class SessionCancelledError(RuntimeError):
    pass


@dataclass
class ConversationPlan:
    route: str
    rewritten_query: str
    clarification_question: Optional[str] = None
    rewrite_prompt: Optional[str] = None


@dataclass
class Repository:
    id: int
    github_url: str
    source_url: str
    session_key: str
    session_expires_at: datetime
    owner: str
    name: str
    branch: str = "main"
    repository_key: str = ""
    cache_generation: Optional[str] = None
    cache_hit: bool = False
    reindex_requested: bool = False
    local_path: Optional[str] = None
    status: str = "queued"
    error_message: Optional[str] = None
    file_count: int = 0
    chunk_count: int = 0
    indexed_at: Optional[datetime] = None
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)


class CodebaseRAGSystem:
    def __init__(
        self,
        repo_dir: str = None,
        clear_existing_index: bool = False,
        settings: Settings = None,
    ):
        self.settings = settings or Settings.from_env()
        self.repo_fetcher = RepoFetcher(base_dir=repo_dir or self.settings.repo_cache_dir)
        self.parser = CodeParser()
        self.embedder = EmbeddingGenerator(
            model_name=self.settings.embedding_model_id,
            enable_profiling=self.settings.enable_profiling,
        )
        self.vector_store = QdrantVectorStore(
            embedding_dim=self.embedder.get_embedding_dim(),
            collection_name=self.settings.qdrant_collection,
            url=self.settings.qdrant_url,
            api_key=self.settings.qdrant_api_key,
            timeout_seconds=self.settings.qdrant_timeout_seconds,
            upsert_batch_size=self.settings.qdrant_upsert_batch_size,
            enable_profiling=self.settings.enable_profiling,
        )
        self.hybrid_search = HybridSearchEngine(
            model_name=self.settings.reranker_model_id,
            batch_size=self.settings.rerank_batch_size,
        )
        self.app_env = self.settings.app_env
        self.llm_provider = "bedrock"
        self.llm_client = None
        self.llm_model = self.settings.bedrock_model_id
        self._configure_llm()
        self.session_ttl_minutes = self.settings.session_ttl_minutes
        self.repo_lock = RLock()
        self.repositories: Dict[int, Repository] = {}
        self.repository_registry: Dict[str, int] = {}
        self.next_repo_id = 1
        self.indexing_progress: Dict[int, dict] = {}
        self.repo_chunks: Dict[int, List[dict]] = {}
        self.cancelled_repo_ids = set()
        self.index_locks: Dict[str, Lock] = {}
        if clear_existing_index:
            self.rebuild_indexes()
        else:
            self.reset_session_state()

    def rebuild_indexes(self):
        with self.repo_lock:
            self.vector_store.clear()
            self.reset_session_state()

    def reset_session_state(self):
        with self.repo_lock:
            self.repositories.clear()
            self.repository_registry.clear()
            self.next_repo_id = 1
            self.repo_chunks.clear()
            self.indexing_progress.clear()
            self.cancelled_repo_ids.clear()

    def create_or_reset_repository(
        self,
        github_url: str,
        session_key: str,
        reindex: bool = False,
    ) -> Repository:
        info = self.repo_fetcher.parse_github_url(github_url)
        repository_key = self._build_repository_key(info)
        registry_key = self._build_registry_key(session_key, repository_key)
        cached_chunks = (
            []
            if reindex
            else self.vector_store.get_repository_chunks(repository_key)
        )
        with self.repo_lock:
            self._cleanup_expired_sessions()
            repo_id = self.repository_registry.get(registry_key)
            repo = self.repositories.get(repo_id) if repo_id else None
            if repo is not None and repo.status in {"queued", "indexing"} and not reindex:
                repo.session_expires_at = self._session_expiry()
                self._mark_repo_updated(repo)
                return repo

            if repo is None:
                repo = Repository(
                    id=self.next_repo_id,
                    github_url=registry_key,
                    source_url=github_url,
                    session_key=session_key,
                    session_expires_at=self._session_expiry(),
                    owner=info["owner"],
                    name=info["repo"],
                    branch=info["branch"],
                    repository_key=repository_key,
                    status="queued",
                )
                self.next_repo_id += 1
                self.repositories[repo.id] = repo
                self.repository_registry[registry_key] = repo.id
                self.cancelled_repo_ids.discard(repo.id)
            else:
                repo.source_url = github_url
                repo.session_key = session_key
                repo.session_expires_at = self._session_expiry()
                repo.owner = info["owner"]
                repo.name = info["repo"]
                repo.branch = info["branch"]
                repo.repository_key = repository_key
                self._mark_repo_updated(repo)
                self.cancelled_repo_ids.discard(repo.id)

            repo.reindex_requested = bool(reindex)
            repo.error_message = None
            self.indexing_progress.pop(repo.id, None)

            if cached_chunks:
                self._hydrate_repository(repo, cached_chunks, cache_hit=True)
            else:
                repo.status = "queued"
                repo.cache_hit = False
                if not reindex:
                    repo.file_count = 0
                    repo.chunk_count = 0
                    repo.indexed_at = None
                    repo.cache_generation = None
                    self.hybrid_search.remove_repository(repo.id)
                    self.repo_chunks.pop(repo.id, None)

            return repo

    def index_repository(self, repo_id: int):
        with self.repo_lock:
            repo = self.repositories.get(repo_id)
            if repo is None:
                return
            index_lock = self.index_locks.setdefault(repo.repository_key, Lock())

        with index_lock:
            with self.repo_lock:
                repo = self.repositories.get(repo_id)
                if repo is None:
                    return
                # Another session may have finished indexing the same
                # repository while this task waited for the per-repo lock.
                if repo.status == "indexed" and not repo.reindex_requested:
                    return
            self._index_repository(repo_id)

    def _index_repository(self, repo_id: int):
        clone_info = None
        staged_generation = None
        generation_activated = False
        index_started_at = time.perf_counter()
        timings = {}
        filtering_profile = {}
        parsing_profile = {
            "files_parsed": 0,
            "files_without_chunks": 0,
            "file_read_seconds": 0.0,
            "tree_sitter_seconds": 0.0,
            "chunk_generation_seconds": 0.0,
        }
        try:
            with self.repo_lock:
                self._cleanup_expired_sessions()
                repo = self.repositories.get(repo_id)
                if repo is None:
                    raise ValueError("Repository not found")
                self._ensure_repo_not_cancelled(repo.id)
                repo.status = "indexing"
                repo.error_message = None
                repo.session_expires_at = self._session_expiry()
                self._mark_repo_updated(repo)
                repository_key = repo.repository_key
                staged_generation = str(uuid4())

            self._set_progress(repo.id, phase="cloning", message="Cloning repository")

            stage_started_at = time.perf_counter()
            clone_info = self.repo_fetcher.clone_repository(repo.source_url or repo.github_url)
            timings["repository_clone_seconds"] = time.perf_counter() - stage_started_at
            self._ensure_repo_not_cancelled(repo.id)
            with self.repo_lock:
                self._ensure_repo_still_exists(repo.id)
                repo.branch = clone_info["branch"]
                repo.local_path = None
                self._mark_repo_updated(repo)
            stage_started_at = time.perf_counter()
            source_files = list(
                self.repo_fetcher.iter_source_files(
                    clone_info["local_path"],
                    profile=filtering_profile,
                )
            )
            timings["repository_filtering_seconds"] = (
                time.perf_counter() - stage_started_at
            )
            total_files = len(source_files)
            self._set_progress(
                repo.id,
                phase="parsing",
                message=f"Scanning {total_files} source files",
                total_files=total_files,
                processed_files=0,
                discovered_chunks=0,
            )

            chunk_payloads = []
            file_count = 0
            for index, file_path in enumerate(source_files, start=1):
                file_profile = {}
                file_chunks = self.parser.chunk_file(
                    str(file_path),
                    clone_info["local_path"],
                    profile=file_profile,
                )
                parsing_profile["files_parsed"] += 1
                parsing_profile["file_read_seconds"] += file_profile.get(
                    "read_seconds", 0.0
                )
                parsing_profile["tree_sitter_seconds"] += file_profile.get(
                    "parse_seconds", 0.0
                )
                parsing_profile["chunk_generation_seconds"] += file_profile.get(
                    "chunk_seconds", 0.0
                )
                if not file_chunks:
                    parsing_profile["files_without_chunks"] += 1
                    self._set_progress(
                        repo.id,
                        phase="parsing",
                        message=f"Parsed {index}/{total_files} files",
                        total_files=total_files,
                        processed_files=index,
                        discovered_chunks=len(chunk_payloads),
                    )
                    continue
                file_count += 1
                chunk_payloads.extend(file_chunks)
                self._set_progress(
                    repo.id,
                    phase="parsing",
                    message=f"Parsed {index}/{total_files} files",
                    total_files=total_files,
                    processed_files=index,
                    discovered_chunks=len(chunk_payloads),
                )

            if not chunk_payloads:
                raise RuntimeError("No supported source-code chunks were found in this repository")

            searchable_texts = [chunk["searchable_text"] for chunk in chunk_payloads]
            self._set_progress(
                repo.id,
                phase="embedding",
                message=f"Embedding {len(searchable_texts)} chunks",
                total_files=total_files,
                processed_files=total_files,
                discovered_chunks=len(chunk_payloads),
                total_chunks=len(chunk_payloads),
                embedded_chunks=0,
            )
            stage_started_at = time.perf_counter()
            embeddings = self.embedder.embed_batch(
                searchable_texts,
                progress_callback=lambda completed, total: self._set_progress(
                    repo.id,
                    phase="embedding",
                    message=f"Embedding chunks ({completed}/{total})",
                    total_files=total_files,
                    processed_files=total_files,
                    discovered_chunks=len(chunk_payloads),
                    total_chunks=total,
                    embedded_chunks=completed,
                ),
            )
            timings["embedding_seconds"] = time.perf_counter() - stage_started_at
            self._ensure_repo_not_cancelled(repo.id)

            stage_started_at = time.perf_counter()
            indexed_at = datetime.utcnow()

            vector_metadata = []
            for chunk in chunk_payloads:
                vector_metadata.append(
                    {
                        "repository_key": repository_key,
                        "cache_generation": staged_generation,
                        "cache_ready": False,
                        "indexed_at": indexed_at.isoformat(),
                        "source_url": repo.source_url or repo.github_url,
                        "owner": repo.owner,
                        "repository_name": repo.name,
                        "branch": repo.branch,
                        "file_path": chunk["file_path"],
                        "language": chunk["language"],
                        "symbol_name": chunk["symbol_name"],
                        "symbol_type": chunk["symbol_type"],
                        "line_start": chunk["line_start"],
                        "line_end": chunk["line_end"],
                        "signature": chunk["signature"],
                        "content": chunk["content"],
                        "searchable_text": chunk["searchable_text"],
                    }
                )
            timings["metadata_preparation_seconds"] = (
                time.perf_counter() - stage_started_at
            )

            stage_started_at = time.perf_counter()
            embedding_ids = self.vector_store.add_embeddings(embeddings, vector_metadata)
            timings["qdrant_upserts_seconds"] = time.perf_counter() - stage_started_at
            self._set_progress(
                repo.id,
                phase="saving",
                message="Saving chunks and search indexes",
                total_files=total_files,
                processed_files=total_files,
                discovered_chunks=len(chunk_payloads),
            )

            stage_started_at = time.perf_counter()
            created_rows = []
            for chunk, embedding_id in zip(chunk_payloads, embedding_ids):
                row = {
                    **chunk,
                    "id": embedding_id,
                    "embedding_id": embedding_id,
                }
                created_rows.append(row)

            serialized = [self._serialize_chunk(chunk) for chunk in created_rows]
            self._ensure_repo_not_cancelled(repo.id)
            self.vector_store.activate_repository_generation(
                repository_key,
                staged_generation,
            )
            generation_activated = True
            with self.repo_lock:
                matching_repositories = [
                    item
                    for item in self.repositories.values()
                    if item.repository_key == repository_key
                ]
                for item in matching_repositories:
                    item.status = "indexed"
                    item.error_message = None
                    item.file_count = file_count
                    item.chunk_count = len(created_rows)
                    item.indexed_at = indexed_at
                    item.cache_generation = staged_generation
                    item.cache_hit = item.id != repo.id
                    item.reindex_requested = False
                    item.branch = repo.branch
                    item.session_expires_at = self._session_expiry()
                    self._mark_repo_updated(item)
                    self.repo_chunks[item.id] = list(serialized)
                    self.indexing_progress.pop(item.id, None)
            for item in matching_repositories:
                self.hybrid_search.build_for_repository(item.id, serialized)
            timings["metadata_persistence_seconds"] = (
                time.perf_counter() - stage_started_at
            )
            with self.repo_lock:
                self.cancelled_repo_ids.discard(repo.id)
            stage_started_at = time.perf_counter()
            self.repo_fetcher.cleanup_repository(clone_info["local_path"])
            timings["repository_cleanup_seconds"] = time.perf_counter() - stage_started_at
            timings["tree_sitter_parsing_seconds"] = parsing_profile[
                "tree_sitter_seconds"
            ]
            timings["chunk_generation_seconds"] = parsing_profile[
                "chunk_generation_seconds"
            ]
            timings["total_indexing_seconds"] = time.perf_counter() - index_started_at
            self._log_index_summary(
                repo,
                timings,
                file_count,
                len(chunk_payloads),
            )
        except Exception as exc:
            if not isinstance(exc, SessionCancelledError):
                error_logger.exception(
                    "%s",
                    fields(
                        operation="index",
                        message="Repository indexing failed",
                        repo_id=repo_id,
                        error_type=type(exc).__name__,
                    ),
                )
            recovered_from_cache = False
            if (
                staged_generation
                and not generation_activated
                and "repository_key" in locals()
            ):
                try:
                    self.vector_store.remove_generation(repository_key, staged_generation)
                except Exception as cleanup_exc:
                    error_logger.exception(
                        "%s",
                        fields(
                            operation="index_cleanup",
                            message="Failed to remove staged Qdrant generation",
                            repo_id=repo_id,
                            error_type=type(cleanup_exc).__name__,
                        ),
                    )
            fallback_chunks = []
            if "repository_key" in locals():
                try:
                    fallback_chunks = self.vector_store.get_repository_chunks(repository_key)
                except Exception:
                    fallback_chunks = []
            with self.repo_lock:
                repo = self.repositories.get(repo_id)
                if repo:
                    if repo_id in self.cancelled_repo_ids:
                        self._delete_repositories([repo], track_cancellation=False)
                    elif fallback_chunks:
                        self._hydrate_repository(repo, fallback_chunks, cache_hit=True)
                        recovered_from_cache = True
                    else:
                        self.hybrid_search.remove_repository(repo_id)
                        self.repo_chunks.pop(repo_id, None)
                        repo.status = "failed"
                        repo.error_message = str(exc)
                        self._mark_repo_updated(repo)
            try:
                if clone_info:
                    self.repo_fetcher.cleanup_repository(clone_info["local_path"])
            except Exception:
                pass
            with self.repo_lock:
                self.indexing_progress.pop(repo_id, None)
            if isinstance(exc, SessionCancelledError):
                return
            if recovered_from_cache:
                return
            # This method runs as a background task. The repository status now
            # exposes the failure to clients; re-raising would make the ASGI
            # server emit the same traceback a second time.
            return

    def _log_index_summary(
        self,
        repo: Repository,
        timings: dict,
        file_count: int,
        chunk_count: int,
    ) -> None:
        index_logger.info("%s", fields(repo=f"{repo.owner}/{repo.name}"))
        index_logger.info("%s", fields(files=file_count, chunks=chunk_count))
        index_logger.info(
            "%s", fields(embedding=f"{timings['embedding_seconds']:.2f}s")
        )
        index_logger.info(
            "%s", fields(total=f"{timings['total_indexing_seconds']:.2f}s")
        )

    def restore_repository_from_cache(
        self,
        github_url: str,
        session_key: str,
        repo_id: int,
    ) -> Optional[int]:
        info = self.repo_fetcher.parse_github_url(github_url)
        repository_key = self._build_repository_key(info)
        chunks = self.vector_store.get_repository_chunks(repository_key)
        if not chunks:
            return None

        registry_key = self._build_registry_key(session_key, repository_key)
        repo = Repository(
            id=repo_id,
            github_url=registry_key,
            source_url=github_url,
            session_key=session_key,
            session_expires_at=self._session_expiry(),
            owner=info["owner"],
            name=info["repo"],
            branch=info["branch"],
            repository_key=repository_key,
        )

        with self.repo_lock:
            self.repositories[repo.id] = repo
            self.repository_registry[registry_key] = repo.id
            self.next_repo_id = max(self.next_repo_id, repo.id + 1)
            self.cancelled_repo_ids.discard(repo.id)
            self._hydrate_repository(repo, chunks, cache_hit=True)

        return repo.id

    def list_repositories_for_session(self, session_key: str) -> List[dict]:
        with self.repo_lock:
            self._cleanup_expired_sessions()
            repos = [
                repo
                for repo in self.repositories.values()
                if repo.session_key == session_key
            ]
            repos.sort(key=lambda repo: repo.updated_at, reverse=True)
            self._touch_session(session_key)
            return [self._serialize_repo(repo) for repo in repos]

    def get_repository_for_session(self, repo_id: int, session_key: str) -> Optional[dict]:
        with self.repo_lock:
            self._cleanup_expired_sessions()
            repo = self.repositories.get(repo_id)
            if repo and repo.session_key != session_key:
                repo = None
            self._touch_session(session_key)
            return self._serialize_repo(repo) if repo else None

    def answer_question(
        self,
        repo_id: int,
        session_key: str,
        question: str,
        top_k: int = 8,
        history=None,
        debug_retrieval: bool = False,
    ) -> dict:
        with self.repo_lock:
            self._cleanup_expired_sessions()
            repo = self.repositories.get(repo_id)
            if repo and repo.session_key != session_key:
                repo = None
            if repo is None:
                raise ValueError("Repository not found")
            repo_chunks = (
                list(self.repo_chunks[repo_id])
                if repo_id in self.repo_chunks
                else None
            )
            self._touch_session(session_key)

        normalized_history = self._normalize_history(history or [])
        conversation_plan = self._plan_conversation(question, normalized_history)
        if conversation_plan.rewritten_query != question:
            query_logger.info(
                "rewritten=%s",
                json.dumps(conversation_plan.rewritten_query, ensure_ascii=False),
            )
        trace = {
            "original_query": question,
            "rewritten_query": conversation_plan.rewritten_query,
            "retrieval_query": None,
            "final_prompt": None,
        }
        if conversation_plan.rewrite_prompt:
            trace["rewrite_prompt"] = conversation_plan.rewrite_prompt

        if conversation_plan.route == "casual":
            answer = self._casual_response(repo, question)
            if debug_retrieval:
                answer["conversation_trace"] = dict(trace)
            return answer

        if conversation_plan.route == "clarify":
            answer = self._clarification_response(
                repo,
                conversation_plan.clarification_question,
            )
            if debug_retrieval:
                answer["conversation_trace"] = dict(trace)
            return answer

        if repo.status != "indexed":
            raise ValueError("Repository is not ready for questions yet")
        if repo_chunks is None:
            raise ValueError("Session cache expired. Re-index the repository and try again.")

        rewritten_query = conversation_plan.rewritten_query
        question_intent = self._question_intent(rewritten_query)
        deep_search_intents = {
            "api",
            "implementation",
            "cross_file",
            "error_handling",
            "setup",
            "tests",
        }
        deep_multiplier = int(os.getenv("RAG_DEEP_SEARCH_MULTIPLIER", "8"))
        shallow_multiplier = int(os.getenv("RAG_SEARCH_MULTIPLIER", "4"))
        search_depth = (
            top_k * deep_multiplier
            if question_intent in deep_search_intents
            else top_k * shallow_multiplier
        )
        search_depth = max(top_k, min(search_depth, 120))
        # Over-fetch before de-duplication. Large repositories often contain
        # translated copies of the same documentation and many chunks from a
        # single file; those copies must not consume the candidate budget.
        fetch_depth = min(max(search_depth * 3, search_depth), 300, len(repo_chunks))

        # Conversation references have already been resolved in rewritten_query.
        # The retrieval query builder remains responsible only for search-oriented
        # intent expansion; the retrieval pipeline below is otherwise unchanged.
        retrieval_query = self._build_retrieval_query(rewritten_query, [])
        trace["retrieval_query"] = retrieval_query
        query_embedding = self.embedder.embed_text(retrieval_query)

        semantic_hits = []
        for score, meta in self.vector_store.search(
            query_embedding,
            k=fetch_depth,
            repository_key=repo.repository_key,
        ):
            serialized = dict(meta)
            serialized["semantic_score"] = score
            semantic_hits.append(serialized)
        semantic_hits = self._deduplicate_candidates(semantic_hits, search_depth)

        lexical_hits = self.hybrid_search.bm25_search(
            repo_chunks,
            retrieval_query,
            top_k=fetch_depth,
            repo_id=repo_id,
        )
        lexical_hits = self._deduplicate_candidates(lexical_hits, search_depth)

        semantic_hits = self.hybrid_search.normalize_semantic_results(semantic_hits)
        semantic_ranks = self._rank_map(semantic_hits)
        lexical_ranks = self._rank_map(lexical_hits)

        fused = self.hybrid_search.reciprocal_rank_fusion(
            lexical_hits, semantic_hits, top_k=search_depth
        )
        fused_ranks = self._rank_map(fused)

        path_hits = self._path_intent_search(
            repo_chunks,
            rewritten_query,
            retrieval_query,
            top_k=search_depth,
        )
        path_ranks = self._rank_map(path_hits)

        merged = self._merge_ranked_candidates(fused, path_hits, top_k=search_depth)

        rerank_query = (
            retrieval_query
            if question_intent in deep_search_intents
            else rewritten_query
        )

        rerank_pool = min(len(merged), max(50, top_k * 8))
        reranked = self.hybrid_search.rerank(rerank_query, merged, top_k=rerank_pool)
        rerank_ranks = self._rank_map(reranked)

        prioritized = self._prioritize_results(
            rewritten_query, retrieval_query, reranked, top_k=rerank_pool
        )
        prioritized_ranks = self._rank_map(prioritized)

        configured_source_limit = int(os.getenv("RAG_FINAL_SOURCE_LIMIT", str(top_k)))
        final_top_k = max(1, min(top_k, configured_source_limit))
        final_sources = self._select_answer_sources(
            rewritten_query, prioritized, top_k=final_top_k
        )
        final_ranks = self._rank_map(final_sources)

        retrieval_debug = []
        if debug_retrieval:
            all_candidates = {}
            for stage_items in (semantic_hits, lexical_hits, fused, path_hits, merged, reranked, prioritized, final_sources):
                for item in stage_items:
                    all_candidates[item["id"]] = {**all_candidates.get(item["id"], {}), **item}

            for chunk_id, item in all_candidates.items():
                retrieval_debug.append(
                    {
                        "id": chunk_id,
                        "file_path": item.get("file_path"),
                        "symbol_name": item.get("symbol_name"),
                        "semantic_rank": semantic_ranks.get(chunk_id),
                        "bm25_rank": lexical_ranks.get(chunk_id),
                        "fused_rank": fused_ranks.get(chunk_id),
                        "path_rank": path_ranks.get(chunk_id),
                        "rerank_rank": rerank_ranks.get(chunk_id),
                        "prioritized_rank": prioritized_ranks.get(chunk_id),
                        "final_rank": final_ranks.get(chunk_id),
                        "semantic_score": item.get("semantic_score"),
                        "bm25_score": item.get("bm25_score"),
                        "rrf_score": item.get("rrf_score"),
                        "path_score": item.get("path_score"),
                        "rerank_score": item.get("rerank_score"),
                        "final_score": item.get("final_score"),
                    }
                )

            retrieval_debug.sort(
                key=lambda item: (
                    item["final_rank"] is None,
                    item["final_rank"] or 10**9,
                    item["rerank_rank"] or 10**9,
                )
            )

        answer = self._generate_answer(
            repo,
            question,
            final_sources,
            normalized_history,
            rewritten_query=rewritten_query,
            trace=trace,
        )
        if debug_retrieval:
            answer["retrieval_debug"] = retrieval_debug
            answer["conversation_trace"] = dict(trace)
        return answer

    def end_session(self, session_key: str):
        with self.repo_lock:
            repos = [
                repo
                for repo in self.repositories.values()
                if repo.session_key == session_key
            ]
            self._delete_repositories(repos)


    def _generate_answer(
        self,
        repo,
        question: str,
        sources: list,
        history=None,
        rewritten_query: Optional[str] = None,
        trace: Optional[dict] = None,
    ) -> dict:
        standalone_question = rewritten_query or question
        answer_mode = self._answer_mode(standalone_question)
        section_specs = self._answer_section_specs(answer_mode)
        if not sources:
            empty_answer = (
                "I could not find enough grounded evidence in the indexed codebase "
                "to answer that confidently."
            )
            return {
                "answer": empty_answer,
                "direct_answer": empty_answer,
                "answer_mode": answer_mode,
                "answer_sections": [
                    {
                        "key": section_specs[0]["key"],
                        "title": section_specs[0]["title"],
                        "type": "markdown",
                        "content": empty_answer,
                    }
                ],
                "implementation_snippets": [],
                "why_this_code_matters": "",
                "related_files": [],
                "confidence": "low",
                "sources": [],
                "repo": self._serialize_repo(repo),
            }

        generation_context = GenerationContextBuilder().build(
            sources,
            standalone_question,
            answer_mode,
        )
        context_blocks = generation_context.blocks
        if trace is not None:
            trace["generation_context"] = generation_context.joined_context
            trace["generation_context_stats"] = generation_context.diagnostics()
        slim_sources = []

        for source in sources:
            slim_sources.append(
                {
                    "file_path": source["file_path"],
                    "language": source["language"],
                    "symbol_name": source["symbol_name"],
                    "symbol_type": source["symbol_type"],
                    "line_start": source["line_start"],
                    "line_end": source["line_end"],
                    "signature": source["signature"],
                    "snippet": source["content"],
                    "semantic_score": round(float(source.get("semantic_score", 0.0)), 4),
                    "bm25_score": round(float(source.get("bm25_score", 0.0)), 4),
                    "rrf_score": round(float(source.get("rrf_score", 0.0)), 4),
                    "rerank_score": round(float(source.get("rerank_score", 0.0)), 4),
                }
            )

        uses_snippets = any(spec["type"] == "snippets" for spec in section_specs)
        implementation_snippets = (
            self._select_implementation_snippets(standalone_question, sources)
            if uses_snippets
            else []
        )
        displayed_implementations = "\n".join(
            f"- Source {snippet['source']}: `{snippet['file_path']}` "
            f"({snippet['symbol_name']}, lines {snippet['line_start']}-{snippet['line_end']})"
            for snippet in implementation_snippets
        ) or "- No concise implementation snippet was available; explain only what the sources establish."

        structure_instructions = self._answer_structure_instructions(answer_mode)
        implementation_context = ""
        if uses_snippets:
            implementation_context = f"""
Implementation snippets the application will insert at the placeholder:
{displayed_implementations}
"""

        system_prompt = f"""
You are a repository understanding assistant. Help the user build a mental model of how this codebase is implemented, like a knowledgeable teammate walking through unfamiliar code.

Rules:
1. Use ONLY the supplied repository context to answer. Do not use external knowledge.
2. Use the question-specific Markdown structure below exactly. Keep its headings in order and do not add other top-level sections.
{structure_instructions}
3. Back factual claims with inline citations such as [1] or [2][3]. Use only the supplied source numbers.
4. Be concrete about execution flow, files, symbols, boundaries, and configuration. If evidence is partial, distinguish facts from inference.
5. Do not say "Based on the provided context" or use similar throat-clearing language.
6. If the sources are insufficient, say so plainly. Never fill gaps with external knowledge.
7. Keep every section complete. Do not leave unfinished headings, bullets, or Markdown.
"""

        joined_context = "\n\n".join(context_blocks)

        user_prompt = f"""
Repository: {repo.owner}/{repo.name}

Context from the codebase:
{joined_context}

Recent conversation:
{self._format_history(history or [])}

Standalone interpretation of the question:
{rewritten_query or question}

Answer mode: {answer_mode}
{implementation_context}

Now answer this question using only the context above:
{question}
"""

        answer_text, finish_reason = self._generate_markdown_response(
            system_prompt,
            user_prompt,
            trace=trace,
        )

        if not self._has_answer_structure(answer_text, answer_mode):
            repair_instructions = self._answer_repair_instructions(answer_mode)
            answer_text, finish_reason = self._generate_markdown_response(
                system_prompt,
                f"{user_prompt.strip()}\n\n{repair_instructions}",
                trace=trace,
            )

        if self._looks_incomplete(answer_text, finish_reason):
            repair_prompt = f"""
The draft answer below appears to be cut off or incomplete.
Rewrite it into a complete final answer using the same repository context and rules.

Draft answer:
{answer_text}
"""
            answer_text, finish_reason = self._generate_markdown_response(
                system_prompt,
                f"{user_prompt.strip()}\n\n{repair_prompt.strip()}",
                trace=trace,
            )
            if self._looks_incomplete(answer_text, finish_reason):
                short_prompt = self._answer_repair_instructions(
                    answer_mode,
                    concise=True,
                )
                answer_text, _ = self._generate_markdown_response(
                    system_prompt,
                    f"{user_prompt.strip()}\n\n{short_prompt}",
                    trace=trace,
                )

        answer_text = self._finalize_answer(answer_text)
        answer_text, citations = self._attach_citations(answer_text, sources)
        presented_answer = self._parse_presented_answer(
            answer_text,
            sources,
            question=standalone_question,
            answer_mode=answer_mode,
            implementation_snippets=implementation_snippets,
        )
        confidence = self._estimate_confidence(sources)
        direct_answer = presented_answer["direct_answer"]
        summary = " ".join(direct_answer.split())[:160] if direct_answer else ""

        return {
            # Keep `answer` as the conversational text for history and older clients.
            "answer": direct_answer,
            "direct_answer": direct_answer,
            "answer_mode": answer_mode,
            "answer_sections": presented_answer["answer_sections"],
            "implementation_snippets": implementation_snippets,
            "why_this_code_matters": presented_answer["why_this_code_matters"],
            "related_files": presented_answer["related_files"],
            "confidence": confidence,
            "summary": summary,
            "citations": citations,
            "sources": slim_sources,
            "repo": self._serialize_repo(repo),
        }

    def _configure_llm(self):
        self.llm_client = boto3.client(
            "bedrock-runtime",
            region_name=self.settings.aws_region,
            config=BotoConfig(
                connect_timeout=5,
                read_timeout=70,
                retries={"max_attempts": 3, "mode": "standard"},
            ),
        )

    def close(self):
        self.vector_store.close()

    def _generate_markdown_response(
        self,
        system_prompt: str,
        user_prompt: str,
        trace: Optional[dict] = None,
    ) -> tuple[str, str]:
        if trace is not None:
            trace["final_prompt"] = self._format_bedrock_prompt(
                system_prompt,
                user_prompt,
            )
        response = self.llm_client.converse(
            modelId=self.llm_model,
            system=[{"text": system_prompt.strip()}],
            messages=[
                {
                    "role": "user",
                    "content": [{"text": user_prompt.strip()}],
                }
            ],
            inferenceConfig={
                "temperature": 0.1,
                "maxTokens": 2200,
            },
        )
        content_blocks = (
            response.get("output", {})
            .get("message", {})
            .get("content", [])
        )
        text = "".join(block.get("text", "") for block in content_blocks)
        if not text.strip():
            raise RuntimeError("Bedrock Qwen returned an empty response.")
        return self._normalize_markdown_answer(text), response.get("stopReason", "")

    @staticmethod
    def _format_bedrock_prompt(system_prompt: str, user_prompt: str) -> str:
        return (
            f"SYSTEM\n{system_prompt.strip()}\n\n"
            f"USER\n{user_prompt.strip()}"
        )

    @staticmethod
    def _normalize_markdown_answer(raw_text: str) -> str:
        cleaned = (raw_text or "").strip()
        cleaned = re.sub(r"^```(?:markdown|md)?\s*|\s*```$", "", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(
            r"\[Source\s+(\d+)\]", r"[\1]", cleaned, flags=re.IGNORECASE
        )
        cleaned = re.sub(
            r"^(?:based on the provided context[,:\s-]*|from the provided context[,:\s-]*)",
            "",
            cleaned,
            flags=re.IGNORECASE,
        ).strip()
        cleaned = re.sub(
            r"\n(?:#{1,6}|[-*])\s*$",
            "",
            cleaned,
            flags=re.MULTILINE,
        ).strip()
        cleaned = re.sub(r"(?:\n\s*){3,}", "\n\n", cleaned)
        cleaned = cleaned.strip()
        if not cleaned:
            return "I found relevant code context, but the model returned an empty response."
        return cleaned

    @staticmethod
    def _finalize_answer(answer_text: str) -> str:
        cleaned = (answer_text or "").strip()
        if not cleaned:
            return "I found relevant code context, but the model returned an empty response."

        if CodebaseRAGSystem._looks_incomplete(cleaned):
            sentence_match = re.search(r"(?s)^.*[.!?](?:['\"\)`\]]+)?", cleaned)
            if sentence_match:
                trimmed = sentence_match.group(0).strip()
                if len(trimmed.split()) >= 12:
                    return trimmed

            lines = cleaned.splitlines()
            while lines and CodebaseRAGSystem._looks_incomplete(lines[-1]):
                lines.pop()
            candidate = "\n".join(line for line in lines if line.strip()).strip()
            if candidate:
                return candidate

        return cleaned

    @staticmethod
    def _looks_incomplete(answer_text: str, finish_reason: str = "") -> bool:
        cleaned = (answer_text or "").strip()
        if not cleaned:
            return True
        finish_reason = (finish_reason or "").strip().lower()
        if finish_reason and finish_reason not in {"stop", "stopsequence", "finish_reason_unspecified"}:
            return True
        if cleaned.endswith(("#", "-", "*", ":", "(", "[", "/", "`")):
            return True
        if cleaned.endswith(("[source", "[source 1", "[source 2", "[source 3", "[source 4")):
            return True
        if cleaned.count("```") % 2 != 0:
            return True
        if cleaned.count("(") > cleaned.count(")"):
            return True
        if cleaned.count("[") > cleaned.count("]"):
            return True
        tokens = re.findall(r"\b[\w'-]+\b", cleaned.lower())
        if not tokens:
            return True
        if tokens[-1] in {"a", "an", "the", "to", "for", "with", "of", "in", "on", "from", "about"}:
            return True
        terminal_text = re.sub(r"(?:\s*\[\d+\])+\s*$", "", cleaned).rstrip()
        if (
            len(tokens) >= 20
            and terminal_text
            and terminal_text[-1] not in {".", "!", "?", "\"", "'", "`"}
        ):
            return True
        return False

    @staticmethod
    def _answer_section_specs(answer_mode: str) -> tuple[dict, ...]:
        structures = {
            "architecture": (
                {
                    "key": "execution_flow",
                    "title": "Execution flow",
                    "type": "markdown",
                    "guidance": "Trace the ordered handoffs from entry point to outcome.",
                },
                {
                    "key": "components_involved",
                    "title": "Components involved",
                    "type": "markdown",
                    "guidance": "Name each participating component and its responsibility.",
                },
                {
                    "key": "related_files",
                    "title": "Related files",
                    "type": "files",
                    "guidance": "List up to four files that establish the flow.",
                },
            ),
            "implementation": (
                {
                    "key": "explanation",
                    "title": "Explanation",
                    "type": "markdown",
                    "guidance": "Answer directly and explain the behavior owned here.",
                },
                {
                    "key": "relevant_implementation",
                    "title": "Relevant implementation",
                    "type": "snippets",
                    "guidance": "Reserve this section for the selected code snippets.",
                },
                {
                    "key": "why_this_code_matters",
                    "title": "Why this code matters",
                    "type": "markdown",
                    "guidance": "Connect the displayed code to the observed behavior.",
                },
            ),
            "configuration": (
                {
                    "key": "configuration",
                    "title": "Configuration",
                    "type": "markdown",
                    "guidance": "Explain the setting, value, default, and scope when known.",
                },
                {
                    "key": "files",
                    "title": "Files",
                    "type": "files",
                    "guidance": "List up to four files that define or consume the setting.",
                },
                {
                    "key": "runtime_impact",
                    "title": "Runtime impact",
                    "type": "markdown",
                    "guidance": "Explain when the setting is read and what behavior it changes.",
                },
            ),
            "debugging": (
                {
                    "key": "root_cause",
                    "title": "Root cause",
                    "type": "markdown",
                    "guidance": "Identify the supported cause or clearly state remaining uncertainty.",
                },
                {
                    "key": "evidence",
                    "title": "Evidence",
                    "type": "markdown",
                    "guidance": "Tie the diagnosis to concrete branches, checks, and source evidence.",
                },
                {
                    "key": "relevant_implementation",
                    "title": "Relevant implementation",
                    "type": "snippets",
                    "guidance": "Reserve this section for the code most relevant to the symptom.",
                },
            ),
        }
        return structures.get(answer_mode, structures["implementation"])

    @classmethod
    def _answer_structure_instructions(cls, answer_mode: str) -> str:
        instructions = [f"Required structure for this {answer_mode} question:"]
        for index, spec in enumerate(cls._answer_section_specs(answer_mode), start=1):
            instructions.append(f"{index}. ## {spec['title']} — {spec['guidance']}")
            if spec["type"] == "snippets":
                instructions.append(
                    "   Write exactly [IMPLEMENTATION_SNIPPETS] under this heading; "
                    "do not write or repeat code."
                )
            elif spec["type"] == "files":
                instructions.append(
                    "   Use bullets in this exact form: - `path/to/file` — "
                    "responsibility. Do not invent paths."
                )
            else:
                instructions.append("   Keep this section to 1-2 concise paragraphs.")
        return "\n".join(instructions)

    @classmethod
    def _answer_repair_instructions(
        cls,
        answer_mode: str,
        concise: bool = False,
    ) -> str:
        headings = ", ".join(
            f"## {spec['title']}" for spec in cls._answer_section_specs(answer_mode)
        )
        length_instruction = (
            "Keep each prose section to one short paragraph and each file list to "
            "at most three bullets."
            if concise
            else "Keep the answer concise and do not add facts absent from the context."
        )
        return (
            f"Rewrite the draft using these exact headings in order: {headings}. "
            "Put only [IMPLEMENTATION_SNIPPETS] under any Relevant implementation "
            f"heading. {length_instruction}"
        )

    @classmethod
    def _has_answer_structure(cls, answer_text: str, answer_mode: str) -> bool:
        expected = [
            spec["title"].lower() for spec in cls._answer_section_specs(answer_mode)
        ]
        headings = [
            match.group(1).strip().lower()
            for match in re.finditer(
                r"^#{1,3}\s+(.+?)\s*$",
                answer_text or "",
                flags=re.MULTILINE,
            )
        ]
        return headings == expected

    @classmethod
    def _parse_presented_answer(
        cls,
        answer_text: str,
        sources: List[dict],
        question: str = "",
        answer_mode: str = "implementation",
        implementation_snippets: Optional[List[dict]] = None,
    ) -> dict:
        """Turn mode-specific Markdown into a stable presentation contract."""
        specs = cls._answer_section_specs(answer_mode)
        titles = [spec["title"] for spec in specs]
        section_pattern = re.compile(
            rf"^#{{1,3}}\s+({'|'.join(re.escape(title) for title in titles)})\s*$",
            flags=re.IGNORECASE | re.MULTILINE,
        )
        matches = list(section_pattern.finditer(answer_text or ""))
        sections = {}
        for index, match in enumerate(matches):
            start = match.end()
            end = matches[index + 1].start() if index + 1 < len(matches) else len(answer_text)
            sections[match.group(1).strip().lower()] = answer_text[start:end].strip()

        prose_sections = [
            sections.get(spec["title"].lower(), "").strip()
            for spec in specs
            if spec["type"] == "markdown"
        ]
        direct_answer = next((content for content in prose_sections if content), "")
        if not direct_answer:
            direct_answer = (answer_text or "").strip()

        file_section = next(
            (
                sections.get(spec["title"].lower(), "")
                for spec in specs
                if spec["type"] == "files"
            ),
            "",
        )
        related_files = cls._parse_answer_files(file_section, sources, question)
        answer_sections = []
        snippet_items = implementation_snippets or []
        for spec in specs:
            presented = {
                "key": spec["key"],
                "title": spec["title"],
                "type": spec["type"],
            }
            if spec["type"] == "markdown":
                presented["content"] = sections.get(spec["title"].lower(), "").strip()
            elif spec["type"] == "snippets":
                presented["items"] = snippet_items
            else:
                presented["items"] = related_files
            answer_sections.append(presented)

        return {
            "direct_answer": direct_answer,
            "answer_sections": answer_sections,
            "why_this_code_matters": sections.get("why this code matters", "").strip(),
            "related_files": related_files,
        }

    @classmethod
    def _parse_answer_files(
        cls,
        section_text: str,
        sources: List[dict],
        question: str,
    ) -> List[dict]:
        source_by_path = {
            str(source.get("file_path") or ""): source
            for source in sources
            if source.get("file_path")
        }
        related_files = []
        seen_paths = set()
        allow_tests = cls._question_intent(question) == "tests"
        for raw_line in (section_text or "").splitlines():
            match = re.match(
                r"^\s*[-*]\s+`?([^`]+?)`?(?:\s+—\s+|\s+–\s+|\s+-\s+)(.+?)\s*$",
                raw_line,
            )
            if not match:
                continue
            path = match.group(1).strip().strip("`")
            if (
                path not in source_by_path
                or path in seen_paths
                or (cls._is_test_source(source_by_path[path]) and not allow_tests)
            ):
                continue
            description = match.group(2).strip()
            related_files.append(
                {
                    "file_path": path,
                    "description": description,
                    "symbol_name": source_by_path[path].get("symbol_name"),
                }
            )
            seen_paths.add(path)
            if len(related_files) == 4:
                break

        if not related_files:
            for source in sources:
                path = str(source.get("file_path") or "")
                if (
                    not path
                    or path in seen_paths
                    or (cls._is_test_source(source) and not allow_tests)
                ):
                    continue
                related_files.append(
                    {
                        "file_path": path,
                        "description": cls._default_related_file_description(source),
                        "symbol_name": source.get("symbol_name"),
                    }
                )
                seen_paths.add(path)
                if len(related_files) == 4:
                    break

        return related_files

    @staticmethod
    def _default_related_file_description(source: dict) -> str:
        symbol = source.get("symbol_name")
        category = CodebaseRAGSystem._source_category(source)
        if symbol and symbol != source.get("file_path"):
            return f"Defines the {symbol} {category} used by this behavior."
        descriptions = {
            "implementation": "Contains the implementation for this behavior.",
            "interface": "Defines the interface or type boundary.",
            "configuration": "Configures this behavior.",
            "documentation": "Documents the intended behavior.",
            "test": "Exercises this behavior.",
        }
        return descriptions.get(category, "Participates in this behavior.")

    @staticmethod
    def _answer_mode(question: str) -> str:
        normalized = " ".join((question or "").lower().split())
        intent = CodebaseRAGSystem._question_intent(normalized)
        if intent == "cross_file" or CodebaseRAGSystem._is_repo_overview_question(
            normalized
        ):
            return "architecture"
        if any(
            token in normalized
            for token in {
                "debug",
                "fail",
                "failed",
                "fails",
                "failing",
                "failure",
                "root cause",
                "broken",
                "not work",
                "not working",
                "unexpected",
                "why does",
                "why is",
            }
        ):
            return "debugging"
        if any(
            token in normalized
            for token in {"config", "setting", "environment", "env var", "configure", "configuration"}
        ):
            return "configuration"
        return "implementation"

    @classmethod
    def _select_implementation_snippets(
        cls,
        question: str,
        sources: List[dict],
    ) -> List[dict]:
        mode = cls._answer_mode(question)
        intent = cls._question_intent(question)
        snippet_limit = 2 if mode in {"architecture", "debugging"} else 1
        ranked = cls._rank_sources_for_answer(question, sources, snippets=True)
        preferred = (
            ranked
            if intent == "tests"
            else [
                source
                for source in ranked
                if cls._source_category(source) not in {"documentation", "test"}
            ]
        )
        if not preferred:
            preferred = [source for source in ranked if not cls._is_test_source(source)]
        if not preferred:
            preferred = ranked

        snippets = []
        used_paths = set()
        for source in preferred:
            path = source.get("file_path") or ""
            if not path or path in used_paths:
                continue
            snippet = cls._extract_display_snippet(
                source,
                question=question,
            )
            snippet["source"] = next(
                (
                    index
                    for index, original in enumerate(sources, start=1)
                    if original.get("id") == source.get("id")
                    or (
                        original.get("file_path") == source.get("file_path")
                        and original.get("symbol_name") == source.get("symbol_name")
                        and original.get("line_start") == source.get("line_start")
                    )
                ),
                1,
            )
            snippets.append(snippet)
            used_paths.add(path)
            if len(snippets) == snippet_limit:
                break
        return snippets

    @staticmethod
    def _extract_display_snippet(
        source: dict,
        max_lines: int = 20,
        expanded_max: int = 60,
        question: str = "",
    ) -> dict:
        """Use the generation-context anchors for a readable contiguous snippet."""
        return GenerationContextBuilder().build_display_snippet(
            source,
            question=question,
            max_lines=max_lines,
            expanded_max=expanded_max,
        )

    @staticmethod
    def _attach_citations(answer_text: str, sources: List[dict]) -> tuple[str, List[dict]]:
        max_source = len(sources)
        if max_source == 0 or not answer_text:
            return answer_text, []

        cited_numbers = set()

        def _keep_or_drop(match: "re.Match") -> str:
            number = int(match.group(1))
            if 1 <= number <= max_source:
                cited_numbers.add(number)
                return match.group(0)
            return ""

        cleaned_text = re.sub(r"\[(\d+)\]", _keep_or_drop, answer_text)
        cleaned_text = re.sub(r"[ \t]+([.,;:!?])", r"\1", cleaned_text).strip()

        numbers_to_cite = cited_numbers if cited_numbers else set(range(1, max_source + 1))

        citations = []
        for index in sorted(numbers_to_cite):
            source = sources[index - 1]
            citations.append(
                {
                    "source": index,
                    "file_path": source["file_path"],
                    "symbol_name": source.get("symbol_name"),
                    "line_start": source.get("line_start"),
                    "line_end": source.get("line_end"),
                    "location": f"{source['file_path']}:{source.get('line_start')}-{source.get('line_end')}",
                }
            )
        return cleaned_text, citations

    @staticmethod
    def _estimate_confidence(sources: List[dict]) -> str:
        if not sources:
            return "low"

        top = sources[0]
        rerank = float(top.get("rerank_score", 0.0))
        semantic = float(top.get("semantic_score", 0.0))

        if len(sources) >= 3 and (rerank >= 0.2 or semantic >= 0.75):
            return "high"
        if rerank >= 0.05 or semantic >= 0.45:
            return "medium"
        return "low"

    def _serialize_repo(self, repo: Repository) -> dict:
        payload = {
            "id": repo.id,
            "github_url": repo.source_url or repo.github_url,
            "owner": repo.owner,
            "name": repo.name,
            "branch": repo.branch,
            "cache_hit": repo.cache_hit,
            "reindex_requested": repo.reindex_requested,
            "local_path": repo.local_path,
            "status": repo.status,
            "error_message": repo.error_message,
            "file_count": repo.file_count,
            "chunk_count": repo.chunk_count,
            "indexed_at": repo.indexed_at.isoformat() if repo.indexed_at else None,
            "created_at": repo.created_at.isoformat() if repo.created_at else None,
            "updated_at": repo.updated_at.isoformat() if repo.updated_at else None,
        }
        progress = self.indexing_progress.get(repo.id)
        if progress:
            payload["progress"] = progress
        return payload

    def _set_progress(self, repo_id: int, **progress):
        with self.repo_lock:
            self.indexing_progress[repo_id] = {
                **self.indexing_progress.get(repo_id, {}),
                **progress,
                "updated_at": datetime.utcnow().isoformat(),
            }

    def _touch_session(self, session_key: str):
        expiry = self._session_expiry()
        for repo in self.repositories.values():
            if repo.session_key == session_key:
                repo.session_expires_at = expiry
                self._mark_repo_updated(repo)

    def _cleanup_expired_sessions(self):
        now = datetime.utcnow()
        expired = [
            repo
            for repo in self.repositories.values()
            if repo.session_expires_at is not None and repo.session_expires_at < now
        ]
        if not expired:
            return
        self._delete_repositories(expired)

    def _delete_repositories(
        self,
        repos: List[Repository],
        track_cancellation: bool = True,
    ):
        repo_ids = [repo.id for repo in repos]
        for repo_id in repo_ids:
            if track_cancellation:
                self.cancelled_repo_ids.add(repo_id)
            self.hybrid_search.remove_repository(repo_id)
            self.repo_chunks.pop(repo_id, None)
            self.indexing_progress.pop(repo_id, None)
            repo = self.repositories.pop(repo_id, None)
            if repo:
                self.repository_registry.pop(repo.github_url, None)

    def _ensure_repo_not_cancelled(self, repo_id: int):
        if repo_id in self.cancelled_repo_ids:
            raise SessionCancelledError("Session ended before indexing completed.")

    @staticmethod
    def _mark_repo_updated(repo: Repository):
        repo.updated_at = datetime.utcnow()

    def _plan_conversation(
        self,
        question: str,
        history: List[dict],
    ) -> ConversationPlan:
        normalized = " ".join((question or "").strip().split())
        if self._is_casual_conversation(normalized):
            return ConversationPlan(route="casual", rewritten_query=normalized)

        if not self._needs_conversation_context(normalized):
            return ConversationPlan(route="retrieve", rewritten_query=normalized)

        usable_history = [
            turn
            for turn in history[-6:]
            if turn.get("content", "").strip()
            and not self._is_casual_conversation(turn.get("content", ""))
            and (
                turn.get("role") != "assistant"
                or self._is_substantive_assistant_message(turn.get("content", ""))
            )
        ]
        if not usable_history:
            return ConversationPlan(
                route="clarify",
                rewritten_query=normalized,
                clarification_question=self._targeted_clarification(normalized),
            )

        rewrite_system_prompt = """
You resolve follow-up questions about a software repository into standalone search queries.

Return exactly one JSON object with these fields:
- rewritten_query: a concise, standalone repository question with pronouns and references resolved
- needs_clarification: true only when the history does not establish a single reasonable referent
- clarification_question: one targeted question naming what the user must identify, or an empty string

Use only the conversation supplied. Preserve file paths, symbol names, and technical terms exactly. Do not answer the question and do not add facts. If more than one referent is genuinely plausible, request clarification instead of guessing.
"""
        rewrite_user_prompt = f"""
Recent conversation:
{self._format_history(usable_history)}

Follow-up question:
{normalized}
"""
        rewrite_prompt = self._format_bedrock_prompt(
            rewrite_system_prompt,
            rewrite_user_prompt,
        )

        try:
            response = self.llm_client.converse(
                modelId=self.llm_model,
                system=[{"text": rewrite_system_prompt.strip()}],
                messages=[
                    {
                        "role": "user",
                        "content": [{"text": rewrite_user_prompt.strip()}],
                    }
                ],
                inferenceConfig={"temperature": 0.0, "maxTokens": 350},
            )
            content_blocks = (
                response.get("output", {})
                .get("message", {})
                .get("content", [])
            )
            raw_text = "".join(block.get("text", "") for block in content_blocks)
            payload = self._parse_rewrite_response(raw_text)
            needs_clarification = payload.get("needs_clarification") is True or str(
                payload.get("needs_clarification", "")
            ).lower() == "true"
            if needs_clarification:
                clarification = str(payload.get("clarification_question") or "").strip()
                generic_clarification = " ".join(clarification.lower().split())
                if len(clarification.split()) < 6 or any(
                    phrase in generic_clarification
                    for phrase in {
                        "could you clarify",
                        "please clarify",
                        "what you mean",
                        "which one",
                    }
                ):
                    clarification = self._targeted_clarification(normalized)
                return ConversationPlan(
                    route="clarify",
                    rewritten_query=normalized,
                    clarification_question=(
                        clarification or self._targeted_clarification(normalized)
                    ),
                    rewrite_prompt=rewrite_prompt,
                )

            rewritten = " ".join(
                str(payload.get("rewritten_query") or "").strip().split()
            )
            if rewritten:
                return ConversationPlan(
                    route="retrieve",
                    rewritten_query=rewritten,
                    rewrite_prompt=rewrite_prompt,
                )
        except Exception as exc:
            error_logger.exception(
                "%s",
                fields(
                    operation="query_rewrite",
                    message="Question rewrite failed; using fallback",
                    error_type=type(exc).__name__,
                ),
            )

        return ConversationPlan(
            route="retrieve",
            rewritten_query=self._fallback_conversation_rewrite(
                normalized,
                usable_history,
            ),
            rewrite_prompt=rewrite_prompt,
        )

    @staticmethod
    def _parse_rewrite_response(raw_text: str) -> dict:
        cleaned = re.sub(
            r"^```(?:json)?\s*|\s*```$",
            "",
            (raw_text or "").strip(),
            flags=re.IGNORECASE,
        )
        match = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)
        if not match:
            raise ValueError("Query rewrite did not return JSON")
        payload = json.loads(match.group(0))
        if not isinstance(payload, dict):
            raise ValueError("Query rewrite JSON must be an object")
        return payload

    @staticmethod
    def _fallback_conversation_rewrite(question: str, history: List[dict]) -> str:
        recent_user = next(
            (
                turn.get("content", "").strip()
                for turn in reversed(history)
                if turn.get("role") == "user" and turn.get("content", "").strip()
            ),
            "",
        )
        recent_assistant = next(
            (
                turn.get("content", "").strip()
                for turn in reversed(history)
                if turn.get("role") == "assistant"
                and CodebaseRAGSystem._is_substantive_assistant_message(
                    turn.get("content", "")
                )
            ),
            "",
        )
        parts = [f"Follow-up question: {question}"]
        if recent_user:
            parts.append(f"Referenced conversation topic: {recent_user[:500]}")
        if recent_assistant:
            parts.append(f"Prior answer context: {recent_assistant[:500]}")
        return "\n".join(parts)

    @staticmethod
    def _needs_conversation_context(question: str) -> bool:
        normalized = " ".join((question or "").lower().split())
        normalized = re.sub(
            r"\b(?:this|that)\s+(?:repo|repository|project|codebase)\b",
            "repository",
            normalized,
        )
        if re.search(
            r"\b(?:here|there|above|earlier|previous|former|latter|same)\b",
            normalized,
        ):
            return True
        if re.search(r"\bits\b", normalized):
            return True
        if re.search(
            r"^(?:this|that|it|these|those|they|them)\b",
            normalized,
        ):
            return True
        if re.search(
            r"^(?:how|why|where|when|what|does|do|is|are|was|were|can|"
            r"could|would|should)\s+(?:does\s+|do\s+|is\s+|are\s+|was\s+|"
            r"were\s+|can\s+|could\s+|would\s+|should\s+)?"
            r"(?:this|that|it|these|those|they|them)\b",
            normalized,
        ) and not re.match(r"^(?:is|would|could|can) it possible\b", normalized):
            return True
        if re.search(
            r"^(?:explain|show|trace|describe|summarize|find)\s+"
            r"(?:this|that|it|these|those|them)\b",
            normalized,
        ):
            return True
        if re.search(
            r"\b(?:this|that|these|those)\s+(?:one|code|function|method|"
            r"class|component|flow|behavior|file|module|implementation|endpoint)\b",
            normalized,
        ):
            return True
        if re.search(
            r"\b(?:about|of|for|with)\s+(?:this|that|it|these|those|them)\s*\??$",
            normalized,
        ):
            return True
        if re.fullmatch(
            r"(?:what|where|why|how|which|when)(?: exactly| so)?\??",
            normalized,
        ):
            return True
        if re.fullmatch(
            r"(?:show me|tell me more|can you elaborate|"
            r"give me (?:the )?(?:code|implementation|tests?|example))\??",
            normalized,
        ):
            return True
        return bool(
            re.match(
                r"^(?:and\s+)?(?:what|how) about\b|^(?:and|also)\b",
                normalized,
            )
        )

    @staticmethod
    def _is_casual_conversation(question: str) -> bool:
        normalized = re.sub(
            r"[^a-z0-9'\s]",
            "",
            " ".join((question or "").lower().split()),
        ).strip()
        casual_patterns = (
            r"(?:hi|hello|hey|hiya|yo)(?: there)?",
            r"good (?:morning|afternoon|evening)",
            r"how are you(?: doing)?",
            r"(?:thanks|thank you|thx)(?: very much| so much)?",
            r"(?:ok|okay|got it|sounds good|cool|nice)",
            r"(?:bye|goodbye|see you|talk to you later)",
            r"(?:who are you|what can you do|help|can you help(?: me)?)",
            r"(?:what'?s up|how'?s it going|nice to meet you|tell me a joke)",
        )
        return any(re.fullmatch(pattern, normalized) for pattern in casual_patterns)

    @staticmethod
    def _targeted_clarification(question: str) -> str:
        normalized = " ".join((question or "").lower().split())
        if re.search(r"\bhere\b", normalized):
            return "Which file, code section, or earlier point does “here” refer to?"
        if re.search(r"\b(?:this|that|it|its)\b", normalized):
            reference = re.search(r"\b(this|that|it|its)\b", normalized).group(1)
            return (
                f"What specific component, file, or behavior does “{reference}” "
                "refer to?"
            )
        if re.search(r"\b(?:these|those|they|them)\b", normalized):
            return "Which components or behaviors are you referring to?"
        return "Which earlier component or behavior should I use for this follow-up?"

    def _casual_response(self, repo: Repository, question: str) -> dict:
        normalized = " ".join((question or "").lower().split())
        if re.search(r"\b(?:thanks|thank you|thx)\b", normalized):
            answer = "You’re welcome! Ask another question whenever you’re ready."
        elif "how are you" in normalized:
            answer = (
                "Doing well—and ready to help you explore "
                f"`{repo.owner}/{repo.name}`."
            )
        elif re.search(r"\b(?:bye|goodbye|see you)\b", normalized):
            answer = "See you! I’ll be here when you want to explore more of the repository."
        elif "who are you" in normalized or "what can you do" in normalized or normalized == "help":
            answer = (
                "I’m Code Compass. I can explain this repository’s architecture, "
                "trace behavior across files, find implementations, and cite the relevant code."
            )
        else:
            answer = (
                f"Hi! Ask me anything about `{repo.owner}/{repo.name}`—for example, "
                "its architecture, a request flow, or where a feature is implemented."
            )
        return {
            "answer": answer,
            "confidence": "high",
            "summary": " ".join(answer.split())[:160],
            "citations": [],
            "sources": [],
            "repo": self._serialize_repo(repo),
            "response_type": "casual",
        }

    def _clarification_response(
        self,
        repo: Repository,
        clarification_question: Optional[str],
    ) -> dict:
        answer = clarification_question or (
            "Which component, file, or behavior should I focus on?"
        )
        return {
            "answer": answer,
            "confidence": "low",
            "summary": answer,
            "citations": [],
            "sources": [],
            "repo": self._serialize_repo(repo),
            "response_type": "clarification",
        }

    def _build_retrieval_query(self, question: str, history: List[dict]) -> str:
        normalized = " ".join(question.strip().split())
        if self._is_repo_overview_question(normalized):
            return "\n".join(
                [
                    normalized,
                    "repository overview purpose main workflow architecture README features stack",
                ]
            )
        if not history:
            return self._expand_query_for_intent(normalized)

        recent_user = [
            turn["content"].strip()
            for turn in reversed(history)
            if turn.get("role") == "user" and turn.get("content", "").strip()
        ]
        recent_assistant = [
            turn["content"].strip()
            for turn in reversed(history)
            if turn.get("role") == "assistant" and turn.get("content", "").strip()
            and self._is_substantive_assistant_message(turn.get("content", ""))
        ]

        is_follow_up = (
            len(normalized.split()) <= 6
            or bool(re.fullmatch(r"(give|show|where|which|how|what)(?:\s+.+)?", normalized.lower()))
            or any(token in normalized.lower() for token in {"code", "snippet", "implementation"})
        )
        if not is_follow_up or not recent_user:
            return self._expand_query_for_intent(normalized)

        parts = [self._expand_query_for_intent(normalized)]
        if recent_user:
            parts.append(f"Follow-up to: {recent_user[0]}")
        if recent_assistant:
            parts.append(f"Previous answer: {recent_assistant[0][:300]}")
        return "\n".join(parts)

    @staticmethod
    def _rank_map(items: List[dict]) -> Dict[str, int]:
        return {item["id"]: rank for rank, item in enumerate(items, start=1)}

    @staticmethod
    def _minmax(values: List[float]) -> List[float]:
        if not values:
            return []
        low = min(values)
        high = max(values)
        if high == low:
            return [0.0 for _ in values]
        return [(value - low) / (high - low) for value in values]

    @staticmethod
    def _source_family(file_path: str) -> str:
        parts = (file_path or "").strip("/").split("/")
        if not parts or not parts[0]:
            return ""
        if parts[0] in {"packages", "apps"} and len(parts) >= 2:
            return "/".join(parts[:2])
        return parts[0]

    @staticmethod
    def _translated_document_family(file_path: str) -> Optional[str]:
        """Return a locale-independent path for translated documentation."""
        normalized = (file_path or "").lower().strip("/")
        match = re.match(
            r"^docs/([a-z]{2,3}(?:-[a-z0-9]{2,8})?)/docs/(.+)$",
            normalized,
        )
        if match:
            return f"docs/{match.group(2)}"
        return None

    @classmethod
    def _deduplicate_candidates(cls, results: List[dict], top_k: int) -> List[dict]:
        """Limit file and translation-family flooding while preserving rank."""
        selected = []
        counts = {}
        translated_slots = {}
        for item in results:
            file_path = (item.get("file_path") or "").lower()
            translated_family = cls._translated_document_family(file_path)
            if translated_family in translated_slots:
                slot = translated_slots[translated_family]
                current_path = (selected[slot].get("file_path") or "").lower()
                if "/en/docs/" in file_path and "/en/docs/" not in current_path:
                    selected[slot] = item
                continue

            if len(selected) >= top_k:
                continue
            key = translated_family or file_path
            limit = 1 if translated_family else 3
            if counts.get(key, 0) >= limit:
                continue
            counts[key] = counts.get(key, 0) + 1
            selected.append(item)
            if translated_family:
                translated_slots[translated_family] = len(selected) - 1
        return selected

    def _merge_ranked_candidates(
        self,
        ranked_results: List[dict],
        path_results: List[dict],
        top_k: int,
    ) -> List[dict]:
        merged = {}

        for rank, item in enumerate(ranked_results, start=1):
            enriched = dict(item)
            enriched.setdefault("rrf_score", 0.0)
            enriched["candidate_rank"] = rank
            merged[enriched["id"]] = enriched

        for rank, item in enumerate(path_results, start=1):
            existing = merged.get(item["id"])
            # Path matching is a useful supporting signal, but it is much
            # noisier than semantic/BM25 retrieval. Keep it below one full RRF
            # channel so a generic path hit cannot outrank a candidate found
            # strongly by both primary retrievers.
            path_bonus = 0.5 / (60 + rank)
            if existing is None:
                enriched = dict(item)
                enriched["rrf_score"] = float(enriched.get("rrf_score", 0.0)) + path_bonus
                enriched["path_rank"] = rank
                merged[enriched["id"]] = enriched
                continue

            existing.update({key: value for key, value in item.items() if key not in existing})
            existing["rrf_score"] = float(existing.get("rrf_score", 0.0)) + path_bonus
            existing["path_rank"] = rank

        return sorted(
            merged.values(),
            key=lambda item: (
                float(item.get("rrf_score", 0.0)),
                float(item.get("path_score", 0.0)),
                float(item.get("semantic_score", 0.0)),
            ),
            reverse=True,
        )[:top_k]

    def _path_intent_search(
        self,
        chunks: List[dict],
        question: str,
        retrieval_query: str,
        top_k: int,
    ) -> List[dict]:
        if not chunks:
            return []

        combined_query = f"{question}\n{retrieval_query}"
        path_hints = self._domain_path_hints(combined_query)
        code_terms = self._query_code_terms(combined_query)
        if not path_hints and not code_terms:
            return []

        scored = []
        path_fragments = self._query_path_fragments(combined_query)
        for item in chunks:
            score = 0
            file_path = (item.get("file_path") or "").lower()
            text = " ".join(
                [
                    file_path,
                    str(item.get("symbol_name") or "").lower(),
                    str(item.get("signature") or "").lower(),
                    str(item.get("content") or "")[:500].lower(),
                ]
            )

            for fragment in path_fragments:
                if file_path == fragment or file_path.endswith(f"/{fragment}") or fragment in file_path:
                    score += 12

            for hint in path_hints:
                normalized_hint = hint.rstrip("/").lower()
                if file_path == normalized_hint or file_path.startswith(normalized_hint + "/"):
                    score += 10
                elif normalized_hint in file_path:
                    score += 6

            if code_terms:
                score += min(sum(1 for term in code_terms if term in text), 8)

            if score <= 0:
                continue

            score += max(self._canonical_path_priority(item, combined_query), 0)

            enriched = dict(item)
            enriched["path_score"] = float(score)
            scored.append(enriched)

        scored.sort(
            key=lambda item: (
                float(item.get("path_score", 0.0)),
                float(item.get("bm25_score", 0.0)),
                float(item.get("semantic_score", 0.0)),
            ),
            reverse=True,
        )
        return self._deduplicate_candidates(scored, top_k)

    def _prioritize_results(
        self,
        question: str,
        retrieval_query: str,
        results: List[dict],
        top_k: int,
    ) -> List[dict]:
        if not results:
            return []

        combined_query = f"{question} {retrieval_query}".lower()
        wants_code = any(
            token in combined_query
            for token in {"code", "snippet", "implementation", "function", "class", "import"}
        )
        question_intent = self._question_intent(question)
        wants_docs = self._is_documentation_query(combined_query) and question_intent in {
            "docs",
            "overview",
        }
        wants_repo_overview = self._is_repo_overview_question(
            question
        ) or self._is_repo_overview_question(retrieval_query)

        rerank_norm = self._minmax([float(item.get("rerank_score", 0.0)) for item in results])
        rrf_norm = self._minmax([float(item.get("rrf_score", 0.0)) for item in results])
        path_norm = self._minmax([float(item.get("path_score", 0.0)) for item in results])
        canonical_norm = self._minmax(
            [float(self._canonical_path_priority(item, combined_query)) for item in results]
        )

        scored = []
        for index, item in enumerate(results):
            enriched = dict(item)
            is_doc = self._is_doc_source(item)
            intent_bonus = 0.0

            if wants_repo_overview and is_doc:
                intent_bonus += 1.0
            if wants_docs and is_doc:
                intent_bonus += 0.8
            if wants_code and not is_doc:
                intent_bonus += 0.5
            if (
                question_intent
                in {"api", "implementation", "cross_file", "error_handling", "setup"}
                and not is_doc
            ):
                intent_bonus += 0.4

            enriched["final_score"] = (
                0.50 * rerank_norm[index]
                + 0.20 * rrf_norm[index]
                + 0.12 * path_norm[index]
                + 0.10 * canonical_norm[index]
                + 0.08 * min(intent_bonus, 1.0)
            )
            scored.append(enriched)

        scored.sort(key=lambda item: item["final_score"], reverse=True)
        return scored[:top_k]

    def _select_answer_sources(
        self,
        question: str,
        results: List[dict],
        top_k: int,
    ) -> List[dict]:
        if not results:
            return []

        intent = self._question_intent(question)
        results = self._rank_sources_for_answer(question, results)
        max_per_file = 2 if intent in {"overview", "docs"} else 1
        selected = []
        selected_ids = set()
        file_counts = {}
        used_families = set()

        if intent == "cross_file":
            for item in results:
                file_path = item.get("file_path", "")
                family = self._source_family(file_path)
                if family and family in used_families:
                    continue
                selected.append(item)
                selected_ids.add(item["id"])
                if family:
                    used_families.add(family)
                file_counts[file_path] = 1
                if len(selected) == top_k:
                    return selected

        for item in results:
            if item["id"] in selected_ids:
                continue
            file_path = item.get("file_path", "")
            count = file_counts.get(file_path, 0)
            if count >= max_per_file:
                continue
            selected.append(item)
            selected_ids.add(item["id"])
            file_counts[file_path] = count + 1
            if len(selected) == top_k:
                break

        return selected

    @classmethod
    def _rank_sources_for_answer(
        cls,
        question: str,
        results: List[dict],
        snippets: bool = False,
    ) -> List[dict]:
        """Rank already-retrieved evidence for explanation and snippet usefulness.

        This is deliberately downstream of retrieval and reranking. It only decides
        which of those grounded candidates best communicate the implementation.
        """
        if not results:
            return []
        mode = cls._answer_mode(question)
        intent = cls._question_intent(question)
        final_scores = cls._minmax(
            [float(item.get("final_score", item.get("rerank_score", 0.0))) for item in results]
        )
        category_weight = {
            "implementation": 5.0,
            "interface": 4.0,
            "configuration": 3.0,
            "documentation": 2.0,
            "test": 0.0,
        }

        ranked = []
        for index, item in enumerate(results):
            enriched = dict(item)
            category = cls._source_category(item)
            score = 2.0 * final_scores[index] + category_weight[category]
            if mode == "configuration" and category == "configuration":
                score += 5.0
            if mode == "architecture" and category in {"implementation", "interface"}:
                score += 0.8
            if mode == "debugging" and category == "implementation":
                score += 1.0
            if intent == "tests" and category == "test":
                score += 6.0
            if item.get("symbol_type") not in {"fallback_chunk", "file_overview", "module"}:
                score += 0.5
            if snippets and category in {"documentation", "test"}:
                score -= 1.5
            enriched["answer_rank_score"] = round(score, 6)
            ranked.append(enriched)

        ranked.sort(
            key=lambda item: (
                float(item.get("answer_rank_score", 0.0)),
                float(item.get("final_score", 0.0)),
            ),
            reverse=True,
        )
        return ranked

    @staticmethod
    def _is_test_source(item: dict) -> bool:
        path = (item.get("file_path") or "").lower()
        basename = path.rsplit("/", 1)[-1]
        return (
            path.startswith(("tests/", "test/", "spec/", "__tests__/"))
            or "/tests/" in path
            or "/test/" in path
            or "/__tests__/" in path
            or basename.startswith("test_")
            or basename.endswith("_test.py")
            or ".test." in basename
            or ".spec." in basename
        )

    @staticmethod
    def _source_category(item: dict) -> str:
        path = (item.get("file_path") or "").lower()
        basename = path.rsplit("/", 1)[-1]
        symbol_type = (item.get("symbol_type") or "").lower()
        if CodebaseRAGSystem._is_test_source(item):
            return "test"

        config_names = {
            ".env",
            ".env.example",
            "dockerfile",
            "package.json",
            "pyproject.toml",
            "requirements.txt",
            "setup.cfg",
            "tox.ini",
            "tsconfig.json",
            "vercel.json",
        }
        config_extensions = (".toml", ".yaml", ".yml", ".ini", ".cfg", ".properties")
        if (
            basename in config_names
            or basename.endswith(config_extensions)
            or basename.startswith(("config.", "settings."))
            or any(token in path for token in {"/config/", "/configs/"})
        ):
            return "configuration"
        if (
            basename.startswith("readme")
            or basename.endswith((".md", ".mdx", ".rst"))
            or path.startswith(("docs/", "documentation/"))
            or "/docs/" in path
        ):
            return "documentation"
        if (
            "interface" in symbol_type
            or "type_alias" in symbol_type
            or basename.endswith((".d.ts", ".h", ".hpp", ".proto", ".graphql"))
            or any(token in path for token in {"/interfaces/", "/types/", "/schemas/"})
        ):
            return "interface"
        return "implementation"

    @staticmethod
    def _is_documentation_query(query: str) -> bool:
        return any(
            token in query
            for token in {
                "readme",
                "docs",
                "documentation",
                "setup",
                "install",
                "installation",
                "usage",
                "overview",
                "what is this repo",
                "what is the repository about",
                "what is the repo about",
                "what does the repo do",
                "what does this repo do",
                "repo summary",
                "repository summary",
                "project summary",
                "feature",
                "features",
                "architecture",
            }
        )

    @staticmethod
    def _question_intent(question: str) -> str:
        normalized = " ".join((question or "").lower().split())
        if not normalized:
            return "general"

        def has_any(terms: set[str]) -> bool:
            return any(
                re.search(rf"(?<![a-z0-9]){re.escape(term)}(?![a-z0-9])", normalized)
                for term in terms
            )

        if CodebaseRAGSystem._is_repo_overview_question(normalized):
            return "overview"
        # A question's retrieval shape takes precedence over the subject matter
        # it happens to mention. For example, tracing how validation failures
        # become responses needs evidence from multiple files, even though the
        # same question also contains error-handling terms.
        if CodebaseRAGSystem._is_cross_file_question(normalized):
            return "cross_file"
        if has_any({"test", "tests", "pytest", "spec"}):
            return "tests"
        if has_any({"error", "invalid", "conflict", "raises", "guard against"}):
            return "error_handling"
        if has_any(
            {
                "api",
                "api v1",
                "api v2",
                "endpoint",
                "frontend",
                "backend",
                "openapi",
                "public api",
                "route",
                "router",
                "trpc",
            }
        ):
            return "api"
        if has_any(
            {
                "build",
                "configure",
                "configured",
                "configuration",
                "create",
                "database",
                "env",
                "environment",
                "install",
                "local development",
                "metadata",
                "orchestration",
                "self-host",
                "self hosting",
                "setup",
                "table",
                "workspace",
            }
        ):
            return "setup"
        if has_any(
            {
                "behavior",
                "class",
                "function",
                "implementation",
                "implemented",
                "job",
                "jobs",
                "lifecycle",
                "lives",
                "method",
                "represented",
                "signing",
                "webhook",
                "webhooks",
                "what is special",
                "where does",
                "where is",
                "where should",
                "where would",
            }
        ):
            return "implementation"
        if CodebaseRAGSystem._is_documentation_query(normalized):
            return "docs"
        return "general"

    @staticmethod
    def _is_cross_file_question(question: str) -> bool:
        normalized = " ".join((question or "").lower().split())
        if not normalized:
            return False

        def has_any(terms: set[str]) -> bool:
            return any(
                re.search(rf"(?<![a-z0-9]){re.escape(term)}(?![a-z0-9])", normalized)
                for term in terms
            )

        if has_any(
            {
                "architecture",
                "architectural",
                "flow",
                "across",
                "across files",
                "across modules",
                "connect",
                "code path",
                "call path",
                "lifecycle",
                "reach",
            }
        ):
            return True

        if has_any({"request", "incoming request"}) and has_any({"response"}):
            return True

        # Transformation questions describe a path between system states even
        # when the user does not use the word "flow".
        transformation_patterns = (
            r"\bhow\b.+\bbecomes?\b.+",
            r"\bhow\b.+\b(?:turn|turns|turned|convert|converts|converted|"
            r"transform|transforms|transformed)\b.+\binto\b.+",
        )
        return any(re.search(pattern, normalized) for pattern in transformation_patterns)

    def _expand_query_for_intent(self, question: str) -> str:
        normalized = " ".join((question or "").split())
        lowered = normalized.lower()
        hints = []
        code_terms = self._query_code_terms(normalized)

        if any(token in lowered for token in {"export", "expose", "import"}):
            hints.extend(["package exports", "__init__.py", "index", "public api", "re-export"])
        if any(token in lowered for token in {"public api", "exposed", "exported"}):
            hints.extend(["public api", "__init__.py", "index"])
        if "session.exec" in lowered or ("session" in lowered and "exec" in lowered):
            hints.extend(["session exec", "session.py", "execute", "scalars"])
        if "async" in lowered:
            hints.extend(["async", "await", "asyncio"])
        if any(token in lowered for token in {"relationship", "field", "function", "method", "class"}):
            hints.extend(["class", "function", "method", "metadata"])
        if any(token in lowered for token in {"under the hood", "implementation", "code path", "conversion"}):
            hints.extend(["implementation", "source", "call path", "class", "function"])
        if any(token in lowered for token in {"error", "invalid", "conflict", "raise", "raises", "guard"}):
            hints.extend(["raise", "raises", "exception", "validation", "guard"])
        if any(token in lowered for token in {"test", "tests", "pytest", "spec"}):
            hints.extend(["test", "tests", "pytest", "spec"])
        if any(token in lowered for token in {"create", "setup", "install", "configuration", "metadata", "table"}):
            hints.extend(["create", "setup", "configure", "initialize", "schema", "README.md", "docs"])
        if "__init__" in lowered or "exports" in lowered:
            hints.extend(["__init__.py", "package exports", "public api"])
        hints.extend(self._domain_path_hints(normalized))

        for term in sorted(code_terms):
            parts = [part for part in re.split(r"[._/-]+", term) if len(part) > 2]
            if len(parts) > 1:
                hints.append(" ".join(parts))

        if not hints:
            return normalized
        return "\n".join([normalized, " ".join(hints)])

    @staticmethod
    def _is_repo_overview_question(question: str) -> bool:
        normalized = " ".join((question or "").lower().split())
        explicit_phrases = {
            "what is the repo about",
            "what is this repo about",
            "what does the repo do",
            "what does this repo do",
            "what is the repository about",
            "what does the repository do",
            "what is this project about",
            "what does this project do",
            "repo summary",
            "repository summary",
            "project summary",
            "summarize the repo",
            "summarize this repo",
            "repo overview",
            "repository overview",
            "project overview",
        }
        if any(phrase in normalized for phrase in explicit_phrases):
            return True

        code_markers = {
            "function", "class", "method", "endpoint", "api", "router",
            "route", "implementation", "implemented", "file", "package",
            "module", "where", "tests",
        }
        if any(re.search(rf"(?<![a-z0-9]){re.escape(marker)}(?![a-z0-9])", normalized) for marker in code_markers):
            return False

        purpose_patterns = (
            r"^what is [\w.-]+(?:\s+and\s+what\s+.+)?\??$",
            r"^what does [\w.-]+ do\??$",
            r"^what problem does [\w.-]+ solve\??$",
            r"^what product problem .+ solve\??$",
            r"^(?:describe|explain) [\w.-]+\??$",
            r"^(?:what is|explain|describe) the (?:purpose|project|repository)\b",
        )
        return any(re.search(pattern, normalized) for pattern in purpose_patterns)

    @staticmethod
    def _is_doc_source(item: dict) -> bool:
        file_path = (item.get("file_path") or "").lower()
        language = (item.get("language") or "").lower()
        return language == "text" or file_path.endswith(".md") or "/readme" in file_path

    @staticmethod
    def _doc_priority(item: dict) -> int:
        file_path = (item.get("file_path") or "").lower()
        if file_path in {"readme.md", "readme"}:
            return 3
        if file_path.startswith("docs/") or "/docs/" in file_path:
            return 2
        if file_path.endswith(".md"):
            return 1
        return 0

    @staticmethod
    def _domain_path_hints(query: str) -> List[str]:
        """Map generic question concepts to the directory/file naming
        conventions real-world repos tend to use for that concept.

        This intentionally stays convention-level (not tied to any single
        project's folder layout) because this system indexes arbitrary
        GitHub repositories: a hint list hardcoded to one repo's paths would
        never match anything in any other repo, silently doing nothing for
        almost every user while looking like it's helping. These patterns
        are matched as path *substrings* by the callers, so they work across
        Python/JS/TS/Go/Java/Rust project layouts without needing to know
        the specific repo's structure in advance.
        """
        normalized = " ".join((query or "").lower().split())
        hints = []

        def has_any(terms: set[str]) -> bool:
            return any(
                bool(re.search(rf"(?<![a-z0-9]){re.escape(term)}(?![a-z0-9])", normalized))
                for term in terms
            )

        rules = [
            (
                {"auth", "authentication", "authorization", "login", "session", "bearer", "oauth", "jwt", "token"},
                ["auth", "authentication", "authorization", "login", "session", "middleware/auth"],
            ),
            (
                {"api", "endpoint", "route", "router", "controller", "handler", "rest", "graphql", "trpc"},
                ["api", "routes", "routers", "controllers", "handlers", "endpoints", "resolvers"],
            ),
            (
                {"database", "db", "sql", "orm", "migration", "migrations", "schema", "model", "models"},
                ["models", "schema", "migrations", "db", "database", "entities", "repositories", "prisma"],
            ),
            (
                {"test", "tests", "testing", "pytest", "spec", "e2e", "unit test", "integration test"},
                ["test", "tests", "__tests__", "spec", "e2e", "testing"],
            ),
            (
                {"config", "configuration", "env", "environment", "settings", "setup", "install", "installation"},
                [
                    "config",
                    "settings",
                    ".env.example",
                    "readme.md",
                    "package.json",
                    "pyproject.toml",
                    "docker-compose",
                    "dockerfile",
                ],
            ),
            (
                {"job", "jobs", "background", "worker", "workers", "queue", "task", "tasks", "cron", "scheduler"},
                ["jobs", "workers", "queue", "tasks", "scheduler"],
            ),
            (
                {"webhook", "webhooks", "callback", "callbacks"},
                ["webhook", "webhooks", "callbacks"],
            ),
            (
                {"email", "emails", "mailer", "notification", "notifications"},
                ["email", "mailer", "notifications", "templates"],
            ),
            (
                {"upload", "storage", "s3", "file", "files", "attachment", "blob"},
                ["storage", "upload", "uploads", "files", "attachments"],
            ),
            (
                {"cli", "command line", "command-line"},
                ["cli", "commands", "bin"],
            ),
            (
                {"ui", "frontend", "component", "components", "page", "pages", "view", "views"},
                ["components", "pages", "views", "ui", "frontend", "client", "app"],
            ),
            (
                {"backend", "server", "service", "services"},
                ["server", "backend", "services", "api"],
            ),
            (
                {"middleware", "interceptor"},
                ["middleware", "interceptors"],
            ),
            (
                {"docker", "container", "deployment", "deploy", "kubernetes", "helm", "ci", "cd", "pipeline"},
                [".github/workflows", "docker", "dockerfile", "docker-compose", "helm", "deploy", "deployment", "ci"],
            ),
            (
                {"docs", "documentation", "readme"},
                ["readme.md", "docs", "documentation"],
            ),
            (
                {"util", "utils", "utility", "helper", "helpers", "common", "shared"},
                ["utils", "util", "helpers", "common", "shared", "lib"],
            ),
            (
                {"type", "types", "interface", "schema"},
                ["types", "type", "interfaces", "schemas"],
            ),
        ]

        for terms, paths in rules:
            if has_any(terms):
                hints.extend(paths)

        return list(dict.fromkeys(hints))

    def _canonical_path_priority(self, item: dict, question: str) -> int:
        file_path = (item.get("file_path") or "").lower()
        source_text = " ".join(
            [
                file_path,
                str(item.get("symbol_name") or "").lower(),
                str(item.get("signature") or "").lower(),
            ]
        )
        basename = file_path.rsplit("/", 1)[-1]
        stem = basename.rsplit(".", 1)[0]
        symbol_name = str(item.get("symbol_name") or "").lower()
        signature = str(item.get("signature") or "").lower()
        intent = self._question_intent(question)
        code_terms = self._query_code_terms(question)
        path_fragments = self._query_path_fragments(question)
        path_hints = self._domain_path_hints(question)
        score = 0

        for fragment in path_fragments:
            if file_path == fragment or file_path.endswith(f"/{fragment}") or fragment in file_path:
                score += 8

        for path_hint in path_hints:
            normalized_hint = path_hint.rstrip("/").lower()
            if file_path == normalized_hint or file_path.startswith(normalized_hint + "/"):
                score += 8
            elif normalized_hint in file_path:
                score += 4

        matched_terms = {term for term in code_terms if term in source_text}
        score += min(len(matched_terms), 6)

        for term in code_terms:
            if term == basename or term == stem:
                score += 4
            elif term in basename:
                score += 3
            if term and term in symbol_name:
                score += 3
            if term and term in file_path:
                score += 2
            if term and term in signature:
                score += 1

        if intent == "api":
            if basename == "__init__.py" or stem in {"index", "public", "api"}:
                score += 4
            if any(token in file_path for token in {"api", "route", "router", "controller"}):
                score += 2
        translated_family = self._translated_document_family(file_path)
        if translated_family:
            # Localized copies should not crowd canonical implementation or
            # English/top-level documentation out of a small context window.
            score -= 2 if intent in {"docs", "overview"} else 7

        is_test_path = (
            file_path.startswith("tests/")
            or "/tests/" in file_path
            or basename.startswith("test_")
            or ".test." in basename
            or ".spec." in basename
        )
        if is_test_path and intent != "tests":
            score -= 5
        if intent in {"implementation", "cross_file"}:
            if not self._is_doc_source(item):
                score += 2
            if item.get("symbol_type") != "fallback_chunk":
                score += 1
        if intent == "tests":
            if (
                file_path.startswith("tests/")
                or "/tests/" in file_path
                or basename.startswith("test_")
                or basename.endswith("_test.py")
                or basename.endswith(".test.js")
                or basename.endswith(".spec.js")
                or basename.endswith(".test.ts")
                or basename.endswith(".spec.ts")
            ):
                score += 5
        if intent == "error_handling":
            if any(token in source_text for token in {"raise", "except", "error", "invalid", "exception"}):
                score += 3
            if "test" in file_path:
                score += 1
        if intent == "setup":
            setup_files = {
                "readme.md",
                "package.json",
                "pyproject.toml",
                "requirements.txt",
                "dockerfile",
                "docker-compose.yml",
                "compose.yml",
            }
            if basename in setup_files or any(token in file_path for token in {"config", "settings", "setup"}):
                score += 3
            if any(token in source_text for token in {"create", "configure", "initialize", "metadata", "schema"}):
                score += 1
        if intent in {"docs", "overview"} and self._is_doc_source(item):
            score += self._doc_priority(item) + 1
        if self._is_doc_source(item) and intent in {
            "api",
            "implementation",
            "cross_file",
            "error_handling",
            "tests",
        }:
            score -= 3
        if file_path.startswith(".agents/") or file_path.startswith(".opencode/"):
            score -= 8

        return score

    @staticmethod
    def _query_code_terms(text: str) -> set:
        stopwords = {
            "about",
            "against",
            "also",
            "and",
            "are",
            "between",
            "code",
            "does",
            "file",
            "for",
            "from",
            "happen",
            "happens",
            "how",
            "into",
            "main",
            "me",
            "model",
            "models",
            "path",
            "project",
            "that",
            "the",
            "this",
            "through",
            "under",
            "using",
            "what",
            "when",
            "where",
            "which",
            "with",
        }
        raw_terms = re.findall(
            r"[A-Za-z_][A-Za-z0-9_]*(?:[./-][A-Za-z_][A-Za-z0-9_]*)*",
            text or "",
        )
        terms = set()
        for raw_term in raw_terms:
            expanded = {raw_term}
            expanded.update(re.split(r"[._/-]+", raw_term))
            expanded.update(re.sub(r"(?<=[a-z0-9])(?=[A-Z])", " ", raw_term).split())

            for term in expanded:
                normalized = term.strip("_./-").lower()
                if len(normalized) < 3 or normalized in stopwords:
                    continue
                terms.add(normalized)
        return terms

    @staticmethod
    def _query_path_fragments(text: str) -> set:
        fragments = set()
        for fragment in re.findall(r"[A-Za-z0-9_./-]+(?:\.[A-Za-z0-9_./-]+|/[A-Za-z0-9_./-]+)", text or ""):
            normalized = fragment.strip().strip("./").lower()
            if "/" in normalized or "." in normalized:
                fragments.add(normalized)
        return fragments

    @staticmethod
    def _is_substantive_assistant_message(content: str) -> bool:
        normalized = " ".join((content or "").strip().lower().split())
        if len(normalized) < 24:
            return False
        if normalized in {
            "hey, what question do you have for me today?",
            "ask a question",
        }:
            return False
        if any(
            phrase in normalized
            for phrase in {
                "ask me anything about",
                "ask another question whenever",
                "i’m code compass",
                "i'm code compass",
                "ready to help you explore",
                "i’ll be here when you want to explore",
                "i'll be here when you want to explore",
            }
        ):
            return False
        return True

    @staticmethod
    def _normalize_history(history: List[object]) -> List[dict]:
        normalized = []
        for turn in history:
            if isinstance(turn, dict):
                role = turn.get("role")
                content = turn.get("content")
            else:
                role = getattr(turn, "role", None)
                content = getattr(turn, "content", None)

            if not role or not content:
                continue

            normalized.append(
                {
                    "role": str(role),
                    "content": str(content),
                }
            )
        return normalized

    @staticmethod
    def _format_history(history: List[dict]) -> str:
        if not history:
            return "None"
        lines = []
        for turn in history[-6:]:
            role = turn.get("role", "user").capitalize()
            content = " ".join(turn.get("content", "").split())
            if content:
                lines.append(f"{role}: {content[:400]}")
        return "\n".join(lines) if lines else "None"

    def _ensure_repo_still_exists(self, repo_id: int):
        if repo_id not in self.repositories:
            raise RuntimeError("Repository was removed before indexing completed.")

    def _session_expiry(self) -> datetime:
        return datetime.utcnow() + timedelta(minutes=self.session_ttl_minutes)

    @staticmethod
    def _build_repository_key(info: dict) -> str:
        owner = str(info["owner"]).strip().lower()
        name = str(info["repo"]).strip().lower()
        branch = str(info.get("branch") or "main").strip()
        return f"github:{owner}/{name}@{branch}"

    @staticmethod
    def _build_registry_key(session_key: str, repository_key: str) -> str:
        return f"{session_key}::{repository_key}"

    def _hydrate_repository(
        self,
        repo: Repository,
        chunks: List[dict],
        cache_hit: bool,
    ) -> None:
        serialized = [self._serialize_chunk(chunk) for chunk in chunks]
        first = chunks[0]
        repo.status = "indexed"
        repo.error_message = None
        repo.file_count = len(
            {chunk.get("file_path") for chunk in chunks if chunk.get("file_path")}
        )
        repo.chunk_count = len(serialized)
        repo.indexed_at = self._parse_cached_datetime(first.get("indexed_at"))
        repo.cache_generation = str(first.get("cache_generation") or "") or None
        repo.cache_hit = cache_hit
        repo.reindex_requested = False
        repo.branch = str(first.get("branch") or repo.branch)
        repo.session_expires_at = self._session_expiry()
        self._mark_repo_updated(repo)
        self.repo_chunks[repo.id] = serialized
        self.indexing_progress.pop(repo.id, None)
        self.hybrid_search.build_for_repository(repo.id, serialized)

    @staticmethod
    def _parse_cached_datetime(value) -> datetime:
        if value:
            try:
                return datetime.fromisoformat(str(value).replace("Z", "+00:00")).replace(
                    tzinfo=None
                )
            except ValueError:
                pass
        return datetime.utcnow()

    @staticmethod
    def _serialize_chunk(chunk: dict) -> dict:
        return {
            "id": chunk["id"],
            "file_path": chunk["file_path"],
            "language": chunk["language"],
            "symbol_name": chunk["symbol_name"],
            "symbol_type": chunk["symbol_type"],
            "line_start": chunk["line_start"],
            "line_end": chunk["line_end"],
            "signature": chunk["signature"],
            "content": chunk["content"],
            "searchable_text": chunk["searchable_text"],
            "metadata_json": chunk.get("metadata_json") or {},
        }
