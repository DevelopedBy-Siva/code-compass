import os
import re
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from threading import RLock
from typing import Dict, List, Optional

from openai import OpenAI

from src.code_parser import CodeParser
from src.bedrock_claude import create_bedrock_runtime_client, generate_bedrock_claude_text
from src.embeddings import EmbeddingGenerator
from src.hybrid_search import HybridSearchEngine
from src.repo_fetcher import RepoFetcher
from src.vector_store import ChromaVectorStore


class SessionCancelledError(RuntimeError):
    pass


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
        index_path: str = None,
    ):
        self.repo_fetcher = RepoFetcher(base_dir=repo_dir)
        self.parser = CodeParser()
        self.embedder = EmbeddingGenerator()
        self.vector_store = ChromaVectorStore(
            embedding_dim=self.embedder.get_embedding_dim(),
            index_path=index_path or "./data/chroma",
            persist=True,
        )
        self.hybrid_search = HybridSearchEngine(
            reranker_model=os.getenv(
                "QWEN_RERANKER_MODEL", "Qwen/Qwen3-Reranker-4B"
            )
        )
        self.app_env = os.getenv("APP_ENV", os.getenv("ENVIRONMENT", "local")).lower()
        self.llm_provider = os.getenv("LLM_PROVIDER", "bedrock").lower()
        self.llm_client = None
        self.llm_model = ""
        self._configure_llm()
        self.session_ttl_minutes = int(os.getenv("SESSION_TTL_MINUTES", "120"))
        self.repo_lock = RLock()
        self.repositories: Dict[int, Repository] = {}
        self.repository_registry: Dict[str, int] = {}
        self.next_repo_id = 1
        self.indexing_progress: Dict[int, dict] = {}
        self.repo_chunks: Dict[int, List[dict]] = {}
        self.cancelled_repo_ids = set()
        self.rebuild_indexes()

    def rebuild_indexes(self):
        with self.repo_lock:
            self.vector_store.clear()
            self.repositories.clear()
            self.repository_registry.clear()
            self.next_repo_id = 1
            self.repo_chunks.clear()
            self.indexing_progress.clear()
            self.cancelled_repo_ids.clear()

    def create_or_reset_repository(self, github_url: str, session_key: str) -> Repository:
        info = self.repo_fetcher.parse_github_url(github_url)
        registry_key = self._build_registry_key(session_key, github_url)
        with self.repo_lock:
            self._cleanup_expired_sessions()
            repo_id = self.repository_registry.get(registry_key)
            repo = self.repositories.get(repo_id) if repo_id else None
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
                repo.status = "queued"
                repo.error_message = None
                repo.file_count = 0
                repo.chunk_count = 0
                repo.indexed_at = None
                self._mark_repo_updated(repo)
                self.cancelled_repo_ids.discard(repo.id)
                self.hybrid_search.remove_repository(repo.id)
                self.vector_store.remove_repository(repo.id)
                self.repo_chunks.pop(repo.id, None)

            return repo

    def index_repository(self, repo_id: int):
        clone_info = None
        try:
            with self.repo_lock:
                self._cleanup_expired_sessions()
                repo = self.repositories.get(repo_id)
                if repo is None:
                    raise ValueError("Repository not found")
                self._ensure_repo_not_cancelled(repo.id)
                print(f"[indexing] Starting repository index repo_id={repo.id}", flush=True)

                repo.status = "indexing"
                repo.error_message = None
                repo.session_expires_at = self._session_expiry()
                self._mark_repo_updated(repo)

            self._set_progress(repo.id, phase="cloning", message="Cloning repository")

            clone_info = self.repo_fetcher.clone_repository(repo.source_url or repo.github_url)
            self._ensure_repo_not_cancelled(repo.id)
            with self.repo_lock:
                self._ensure_repo_still_exists(repo.id)
                repo.branch = clone_info["branch"]
                repo.local_path = None
                self._mark_repo_updated(repo)
            print(
                f"[indexing] Repository cloned repo_id={repo.id} branch={repo.branch} "
                f"path={clone_info['local_path']}",
                flush=True,
            )

            source_files = list(self.repo_fetcher.iter_source_files(clone_info["local_path"]))
            total_files = len(source_files)
            print(
                f"[indexing] Found {total_files} source files for repo_id={repo.id}",
                flush=True,
            )
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
                file_chunks = self.parser.chunk_file(str(file_path), clone_info["local_path"])
                if not file_chunks:
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

            searchable_texts = [chunk["searchable_text"] for chunk in chunk_payloads]
            print(
                f"[indexing] Parsed repo_id={repo.id} files={file_count} chunks={len(searchable_texts)}",
                flush=True,
            )
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
            self._ensure_repo_not_cancelled(repo.id)

            vector_metadata = []
            for chunk in chunk_payloads:
                vector_metadata.append(
                    {
                        "repository_id": repo.id,
                        "file_path": chunk["file_path"],
                        "language": chunk["language"],
                        "symbol_name": chunk["symbol_name"],
                        "symbol_type": chunk["symbol_type"],
                        "line_start": chunk["line_start"],
                        "line_end": chunk["line_end"],
                        "signature": chunk["signature"],
                        "content": chunk["content"],
                    }
                )

            embedding_ids = self.vector_store.add_embeddings(embeddings, vector_metadata)
            print(
                f"[indexing] Uploaded {len(embedding_ids)} embeddings to vector store for repo_id={repo.id}",
                flush=True,
            )
            self._set_progress(
                repo.id,
                phase="saving",
                message="Saving chunks and search indexes",
                total_files=total_files,
                processed_files=total_files,
                discovered_chunks=len(chunk_payloads),
            )

            created_rows = []
            for chunk, embedding_id in zip(chunk_payloads, embedding_ids):
                row = {
                    **chunk,
                    "id": embedding_id,
                    "repository_id": repo.id,
                    "embedding_id": embedding_id,
                }
                created_rows.append(row)

            serialized = [self._serialize_chunk(chunk) for chunk in created_rows]
            with self.repo_lock:
                self._ensure_repo_still_exists(repo.id)
                self._ensure_repo_not_cancelled(repo.id)
                repo.status = "indexed"
                repo.file_count = file_count
                repo.chunk_count = len(created_rows)
                repo.indexed_at = datetime.utcnow()
                repo.session_expires_at = self._session_expiry()
                self._mark_repo_updated(repo)
                self.repo_chunks[repo.id] = serialized
            # Build the lexical (BM25) index once now, instead of
            # re-tokenizing every chunk in the repo on every question.
            self.hybrid_search.build_for_repository(repo.id, serialized)
            self.vector_store.save()
            with self.repo_lock:
                self.indexing_progress.pop(repo.id, None)
                self.cancelled_repo_ids.discard(repo.id)
            self.repo_fetcher.cleanup_repository(clone_info["local_path"])
            print(f"[indexing] Repository index complete repo_id={repo.id}", flush=True)
        except Exception as exc:
            print(f"[indexing] Repository index failed repo_id={repo_id} error={exc}", flush=True)
            self.vector_store.remove_repository(repo_id)
            self.hybrid_search.remove_repository(repo_id)
            with self.repo_lock:
                self.repo_chunks.pop(repo_id, None)
                repo = self.repositories.get(repo_id)
                if repo:
                    if repo_id in self.cancelled_repo_ids:
                        self._delete_repositories([repo], track_cancellation=False)
                    else:
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
            raise

    def list_repositories(self) -> List[dict]:
        raise NotImplementedError

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

    def get_repository(self, repo_id: int) -> Optional[dict]:
        raise NotImplementedError

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
            if repo.status != "indexed":
                raise ValueError("Repository is not ready for questions yet")
            if repo_id not in self.repo_chunks:
                raise ValueError("Session cache expired. Re-index the repository and try again.")
            repo_chunks = list(self.repo_chunks[repo_id])
            self._touch_session(session_key)

        normalized_history = self._normalize_history(history or [])
        question_intent = self._question_intent(question)
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

        retrieval_query = self._build_retrieval_query(question, normalized_history)
        query_embedding = self.embedder.embed_text(retrieval_query)

        semantic_hits = []
        for score, meta in self.vector_store.search(query_embedding, k=search_depth, repo_filter=repo_id):
            serialized = dict(meta)
            serialized["semantic_score"] = score
            semantic_hits.append(serialized)

        lexical_hits = self.hybrid_search.bm25_search(
            repo_chunks,
            retrieval_query,
            top_k=search_depth,
            repo_id=repo_id,
        )

        semantic_hits = self.hybrid_search.normalize_semantic_results(semantic_hits)
        semantic_ranks = self._rank_map(semantic_hits)
        lexical_ranks = self._rank_map(lexical_hits)

        fused = self.hybrid_search.reciprocal_rank_fusion(
            lexical_hits, semantic_hits, top_k=search_depth
        )
        fused_ranks = self._rank_map(fused)

        path_hits = self._path_intent_search(
            repo_chunks,
            question,
            retrieval_query,
            top_k=search_depth,
        )
        path_ranks = self._rank_map(path_hits)

        merged = self._merge_ranked_candidates(fused, path_hits, top_k=search_depth)

        rerank_query = retrieval_query if question_intent in deep_search_intents else question

        # Rerank broadly for recall. This is independent of the final LLM
        # context size, which remains capped below.
        rerank_pool = min(search_depth, 50)
        reranked = self.hybrid_search.rerank(rerank_query, merged, top_k=rerank_pool)
        rerank_ranks = self._rank_map(reranked)

        prioritized = self._prioritize_results(
            question, retrieval_query, reranked, top_k=top_k
        )

        final_top_k = min(top_k, 5)
        final_sources = self._select_answer_sources(
            question, prioritized, top_k=final_top_k
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

        answer = self._generate_answer(repo, question, final_sources, normalized_history)
        if debug_retrieval:
            answer["retrieval_debug"] = retrieval_debug
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
    ) -> dict:
        if not sources:
            return {
                "answer": "I could not find enough grounded evidence in the indexed codebase to answer that confidently.",
                "confidence": "low",
                "sources": [],
                "repo": self._serialize_repo(repo),
            }

        context_blocks = []
        slim_sources = []

        for index, source in enumerate(sources, start=1):
            content_preview = source["content"][:1500]

            context_blocks.append(
                "\n".join(
                    [
                        f"[Source {index}]",
                        f"File: {source['file_path']}",
                        f"Symbol: {source['symbol_name']}",
                        f"Lines: {source['line_start']}-{source['line_end']}",
                        content_preview,
                    ]
                )
            )
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

        wants_repo_overview = self._is_repo_overview_question(question)
        question_intent = self._question_intent(question)

        system_prompt = """
You are answering questions as a knowledgeable teammate who has carefully read this repository.

Rules:
1. Use ONLY the supplied repository context to answer. Do not use external knowledge.
2. Answer conversationally and directly, as if the repo is explaining itself to the user.
3. Do not say "Based on the provided context", "The repository is about", or similar throat-clearing phrases.
4. Be concrete about files, functions, and behavior.
5. If evidence is partial, clearly separate what is certain from what is inferred.
6. Respond in Markdown, not JSON.
7. Keep the answer complete. Do not stop mid-sentence.
8. Use short sections or bullets only when they genuinely help readability.
9. Do not leave unfinished headings, dangling bullets, or trailing markdown markers like #, ##, or ###.
10. Back up factual claims with inline citations. After a sentence or clause that relies on a specific source, add its number in brackets, e.g. [1] or [2][3] if multiple sources support it. Use only the source numbers given above (Source 1, Source 2, ...) and never invent a number.
11. If you cannot answer the question using the provided context, say: "I cannot find sufficient evidence in the codebase to answer this question."
12. Prefer the most canonical source files for API and implementation questions, such as package exports, core modules, and session/query code, over tutorial prose when they disagree in specificity.
13. Keep the answer tight. Lead with the direct answer, then add only the most important supporting detail.
"""

        if wants_repo_overview:
            system_prompt += """
14. For repository overview questions, lead with a direct one or two sentence summary of what the repo does.
15. Prioritize README and top-level documentation when they are present, then use code to support the explanation.
16. Mention the main workflow, core stack, and any important product constraints the user would care about.
17. Keep the answer polished and self-contained, like the overview a real user expects when they ask what a repo is about.
"""
        elif question_intent in {"api", "implementation", "cross_file", "error_handling", "setup"}:
            system_prompt += """
14. For API, implementation, setup, and cross-file questions, prefer the smallest correct answer that is directly supported by code.
15. If a detail comes only from docs or examples and not from the canonical implementation, say that clearly instead of presenting it as core behavior.
16. When describing exports or code paths, name the file first and keep the explanation precise.
17. Default to one short paragraph plus at most 3 short bullets. Avoid long explanatory walkthroughs unless the question explicitly asks for depth.
"""

        joined_context = "\n\n".join(context_blocks)

        # FIX: context is placed BEFORE the question (prompt ordering fix).
        user_prompt = f"""
Repository: {repo.owner}/{repo.name}

Context from the codebase:
{joined_context}

Recent conversation:
{self._format_history(history or [])}

Now answer this question using only the context above:
{question}
"""

        answer_text, finish_reason = self._generate_markdown_response(system_prompt, user_prompt)

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
            )
            if self._looks_incomplete(answer_text, finish_reason):
                short_prompt = """
Answer the question again, but keep it concise and complete.
Use 2 short paragraphs or 4-6 bullets max.
Do not leave the answer unfinished.
"""
                answer_text, _ = self._generate_markdown_response(
                    system_prompt,
                    f"{user_prompt.strip()}\n\n{short_prompt.strip()}",
                )

        answer_text = self._finalize_answer(answer_text)
        answer_text, citations = self._attach_citations(answer_text, sources)
        confidence = self._estimate_confidence(sources)
        summary = " ".join(answer_text.split())[:160] if answer_text else ""

        return {
            "answer": answer_text,
            "confidence": confidence,
            "summary": summary,
            "citations": citations,
            "sources": slim_sources,
            "repo": self._serialize_repo(repo),
        }

    def _configure_llm(self):
        if self.llm_provider == "bedrock":
            self.llm_client = create_bedrock_runtime_client()
            self.llm_model = os.getenv(
                "BEDROCK_LLM_MODEL",
                "anthropic.claude-3-5-sonnet-20240620-v1:0",
            )
            return

        if self.llm_provider == "groq":
            self.llm_client = OpenAI(
                api_key=os.getenv("GROQ_API_KEY"),
                base_url=os.getenv("GROQ_BASE_URL", "https://api.groq.com/openai/v1"),
            )
            self.llm_model = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
            return

        if self.llm_provider == "vertex_ai":
            project = os.getenv("GOOGLE_CLOUD_PROJECT")
            location = os.getenv("GOOGLE_CLOUD_LOCATION", "global")
            if not project:
                raise RuntimeError(
                    "GOOGLE_CLOUD_PROJECT must be set when using Vertex AI LLMs."
                )

            self.llm_model = os.getenv("VERTEX_LLM_MODEL", "claude-3-5-sonnet@20240620")
            if self.llm_model.startswith("claude-"):
                try:
                    from anthropic import AnthropicVertex
                except ImportError as exc:
                    raise RuntimeError(
                        "Vertex AI Claude support requires the `anthropic[vertex]` package."
                    ) from exc
                self.llm_client = AnthropicVertex(project_id=project, region=location)
                return

            try:
                from google import genai
            except ImportError as exc:
                raise RuntimeError(
                    "Vertex AI Gemini support requires the `google-genai` package."
                ) from exc

            self.llm_client = genai.Client(
                vertexai=True,
                project=project,
                location=location,
            )
            return

        raise RuntimeError(f"Unsupported LLM provider: {self.llm_provider}")

    def _generate_markdown_response(self, system_prompt: str, user_prompt: str) -> tuple[str, str]:
        if self.llm_provider == "bedrock":
            text, stop_reason = generate_bedrock_claude_text(
                self.llm_client,
                self.llm_model,
                system_prompt,
                user_prompt,
                max_tokens=2200,
                temperature=0.1,
            )
            return self._normalize_markdown_answer(text), stop_reason

        if self.llm_provider == "groq":
            response = self.llm_client.chat.completions.create(
                model=self.llm_model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.1,
                max_tokens=1600,
            )
            content = response.choices[0].message.content
            finish_reason = getattr(response.choices[0], "finish_reason", "") or ""
            return self._normalize_markdown_answer(content), str(finish_reason)

        if self.llm_provider == "vertex_ai" and self.llm_model.startswith("claude-"):
            message = self.llm_client.messages.create(
                model=self.llm_model,
                system=system_prompt.strip(),
                max_tokens=2200,
                temperature=0.1,
                messages=[
                    {
                        "role": "user",
                        "content": user_prompt.strip(),
                    }
                ],
            )
            content_blocks = getattr(message, "content", None) or []
            text = "".join(
                getattr(block, "text", "") for block in content_blocks if getattr(block, "text", "")
            )
            if not text.strip():
                raise RuntimeError("Vertex AI Claude returned an empty response.")
            stop_reason = getattr(message, "stop_reason", "") or ""
            return self._normalize_markdown_answer(text), str(stop_reason)

        response = self.llm_client.models.generate_content(
            model=self.llm_model,
            contents=f"{system_prompt.strip()}\n\n{user_prompt.strip()}",
            config={
                "temperature": 0.1,
                "max_output_tokens": 2200,
            },
        )
        if not getattr(response, "text", None):
            raise RuntimeError("Vertex AI Gemini returned an empty response.")
        finish_reason = ""
        candidates = getattr(response, "candidates", None) or []
        if candidates:
            finish_reason = str(getattr(candidates[0], "finish_reason", "") or "")
        return self._normalize_markdown_answer(response.text), finish_reason

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

        # If the tail still looks truncated, trim back to the last complete sentence or list item
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
        if len(tokens) >= 20 and cleaned[-1] not in {".", "!", "?", "\"", "'", "`"}:
            return True
        return False

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
            # Drop citation markers that don't correspond to a real source.
            return ""

        cleaned_text = re.sub(r"\[(\d+)\]", _keep_or_drop, answer_text)
        cleaned_text = re.sub(r"[ \t]+([.,;:!?])", r"\1", cleaned_text).strip()

        # If the model didn't cite anything inline, fall back to listing every
        # retrieved source so the response is still grounded in a traceable way.
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
            self.vector_store.remove_repository(repo_id)
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
            path_bonus = 1.0 / (20 + rank)
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
        return scored[:top_k]

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
        if has_any({"flow", "across", "across files", "connect", "code path"}):
            return "cross_file"
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
        for turn in history[-4:]:
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
    def _build_registry_key(session_key: str, github_url: str) -> str:
        return f"{session_key}::{github_url}"

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
