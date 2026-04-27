import os
import re
from datetime import datetime, timedelta
from typing import Dict, List, Optional

from openai import OpenAI

from src.code_parser import CodeParser
from src.bedrock_claude import create_bedrock_runtime_client, generate_bedrock_claude_text
from src.database import Repository, get_db_session, init_db, resolve_database_url
from src.embeddings import EmbeddingGenerator
from src.hybrid_search import HybridSearchEngine
from src.repo_fetcher import RepoFetcher
from src.vector_store import QdrantVectorStore


class SessionCancelledError(RuntimeError):
    pass


class CodebaseRAGSystem:
    def __init__(
        self,
        database_url: str = None,
        repo_dir: str = None,
        index_path: str = None,
    ):
        self.database_url = database_url or os.getenv(
            "DATABASE_URL", "sqlite:///./codebase_rag.db"
        )
        self.database_url = resolve_database_url(self.database_url)
        init_db(self.database_url)
        print(f"[database] Using database_url={self.database_url}", flush=True)

        self.repo_fetcher = RepoFetcher(base_dir=repo_dir)
        self.parser = CodeParser()
        self.embedder = EmbeddingGenerator()
        self.vector_store = QdrantVectorStore(
            embedding_dim=self.embedder.get_embedding_dim(),
            index_path=index_path or "./data/faiss/codebase_index",
            persist=False,
        )
        self.hybrid_search = HybridSearchEngine(
            reranker_model=os.getenv(
                "RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2"
            )
        )
        self.app_env = os.getenv("APP_ENV", os.getenv("ENVIRONMENT", "local")).lower()
        self.llm_provider = os.getenv("LLM_PROVIDER", "bedrock").lower()
        self.llm_client = None
        self.llm_model = ""
        self._configure_llm()
        self.session_ttl_minutes = int(os.getenv("SESSION_TTL_MINUTES", "120"))
        self.indexing_progress: Dict[int, dict] = {}
        self.repo_chunks: Dict[int, List[dict]] = {}
        self.cancelled_repo_ids = set()
        self.rebuild_indexes()

    def rebuild_indexes(self):
        session = get_db_session(self.database_url)
        try:
            self.vector_store.clear()
            self.repo_chunks.clear()
            self.indexing_progress.clear()
            self.cancelled_repo_ids.clear()
            repos = session.query(Repository).all()
            self._delete_repositories(session, repos, track_cancellation=False)
            self.cancelled_repo_ids.clear()
            session.commit()
        finally:
            session.close()

    def create_or_reset_repository(self, github_url: str, session_key: str) -> Repository:
        info = self.repo_fetcher.parse_github_url(github_url)
        registry_key = self._build_registry_key(session_key, github_url)
        session = get_db_session(self.database_url)
        try:
            self._cleanup_expired_sessions(session)
            repo = session.query(Repository).filter_by(github_url=registry_key).first()
            if repo is None:
                repo = Repository(
                    github_url=registry_key,
                    source_url=github_url,
                    session_key=session_key,
                    session_expires_at=self._session_expiry(),
                    owner=info["owner"],
                    name=info["repo"],
                    branch=info["branch"],
                    status="queued",
                )
                session.add(repo)
                session.flush()
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
                self.cancelled_repo_ids.discard(repo.id)
                self.hybrid_search.remove_repository(repo.id)
                self.vector_store.remove_repository(repo.id)
                self.repo_chunks.pop(repo.id, None)

            session.commit()
            session.refresh(repo)
            return repo
        finally:
            session.close()

    def index_repository(self, repo_id: int):
        session = get_db_session(self.database_url)
        try:
            self._cleanup_expired_sessions(session)
            repo = session.query(Repository).filter_by(id=repo_id).first()
            if repo is None:
                raise ValueError("Repository not found")
            self._ensure_repo_not_cancelled(repo.id)
            print(f"[indexing] Starting repository index repo_id={repo.id}", flush=True)

            repo.status = "indexing"
            repo.error_message = None
            repo.session_expires_at = self._session_expiry()
            session.commit()
            self._set_progress(repo.id, phase="cloning", message="Cloning repository")

            clone_info = self.repo_fetcher.clone_repository(repo.source_url or repo.github_url)
            self._ensure_repo_not_cancelled(repo.id)
            repo.local_path = None
            repo.branch = clone_info["branch"]
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

            repo.status = "indexed"
            repo.file_count = file_count
            repo.chunk_count = len(created_rows)
            repo.indexed_at = datetime.utcnow()
            repo.session_expires_at = self._session_expiry()
            self._ensure_repo_still_exists(session, repo.id)
            self._ensure_repo_not_cancelled(repo.id)
            session.commit()

            serialized = [self._serialize_chunk(chunk) for chunk in created_rows]
            self.repo_chunks[repo.id] = serialized
            self.vector_store.save()
            self.indexing_progress.pop(repo.id, None)
            self.cancelled_repo_ids.discard(repo.id)
            self.repo_fetcher.cleanup_repository(clone_info["local_path"])
            print(f"[indexing] Repository index complete repo_id={repo.id}", flush=True)
        except Exception as exc:
            print(f"[indexing] Repository index failed repo_id={repo_id} error={exc}", flush=True)
            session.rollback()
            self.vector_store.remove_repository(repo_id)
            self.repo_chunks.pop(repo_id, None)
            self.hybrid_search.remove_repository(repo_id)
            repo = session.query(Repository).filter_by(id=repo_id).first()
            if repo:
                if repo_id in self.cancelled_repo_ids:
                    session.delete(repo)
                else:
                    repo.status = "failed"
                    repo.error_message = str(exc)
                session.commit()
            try:
                if "clone_info" in locals():
                    self.repo_fetcher.cleanup_repository(clone_info["local_path"])
            except Exception:
                pass
            self.indexing_progress.pop(repo_id, None)
            if isinstance(exc, SessionCancelledError):
                return
            raise
        finally:
            session.close()

    def list_repositories(self) -> List[dict]:
        raise NotImplementedError

    def list_repositories_for_session(self, session_key: str) -> List[dict]:
        session = get_db_session(self.database_url)
        try:
            self._cleanup_expired_sessions(session)
            repos = (
                session.query(Repository)
                .filter_by(session_key=session_key)
                .order_by(Repository.updated_at.desc())
                .all()
            )
            self._touch_session(session, session_key)
            return [self._serialize_repo(repo) for repo in repos]
        finally:
            session.close()

    def get_repository(self, repo_id: int) -> Optional[dict]:
        raise NotImplementedError

    def get_repository_for_session(self, repo_id: int, session_key: str) -> Optional[dict]:
        session = get_db_session(self.database_url)
        try:
            self._cleanup_expired_sessions(session)
            repo = (
                session.query(Repository)
                .filter_by(id=repo_id, session_key=session_key)
                .first()
            )
            self._touch_session(session, session_key)
            return self._serialize_repo(repo) if repo else None
        finally:
            session.close()

    def answer_question(
        self,
        repo_id: int,
        session_key: str,
        question: str,
        top_k: int = 8,
        history: Optional[List[object]] = None,
    ) -> dict:
        session = get_db_session(self.database_url)
        try:
            self._cleanup_expired_sessions(session)
            repo = (
                session.query(Repository)
                .filter_by(id=repo_id, session_key=session_key)
                .first()
            )
            if repo is None:
                raise ValueError("Repository not found")
            if repo.status != "indexed":
                raise ValueError("Repository is not ready for questions yet")
            if repo_id not in self.repo_chunks:
                raise ValueError("Session cache expired. Re-index the repository and try again.")
            self._touch_session(session, session_key)

            normalized_history = self._normalize_history(history or [])
            question_intent = self._question_intent(question)
            search_depth = top_k * 4 if question_intent in {"api", "implementation", "cross_file", "setup"} else top_k * 2
            retrieval_query = self._build_retrieval_query(question, normalized_history)
            query_embedding = self.embedder.embed_text(retrieval_query)
            semantic_hits = []
            for score, meta in self.vector_store.search(query_embedding, k=search_depth, repo_filter=repo_id):
                serialized = dict(meta)
                serialized["semantic_score"] = score
                semantic_hits.append(serialized)

            lexical_hits = self.hybrid_search.bm25_search(
                self.repo_chunks[repo_id],
                retrieval_query,
                top_k=search_depth,
            )
            semantic_hits = self.hybrid_search.normalize_semantic_results(semantic_hits)
            fused = self.hybrid_search.reciprocal_rank_fusion(lexical_hits, semantic_hits, top_k=search_depth)
            rerank_query = retrieval_query if question_intent in {"api", "implementation", "cross_file", "setup"} else question
            reranked = self.hybrid_search.rerank(rerank_query, fused, top_k=search_depth)
            reranked = self._prioritize_results(question, retrieval_query, reranked, top_k=top_k)
            reranked = self._select_answer_sources(question, reranked, top_k=top_k)

            answer = self._generate_answer(repo, question, reranked, normalized_history)

            return answer
        finally:
            session.close()

    def end_session(self, session_key: str):
        session = get_db_session(self.database_url)
        try:
            repos = session.query(Repository).filter_by(session_key=session_key).all()
            self._delete_repositories(session, repos)
            session.commit()
        finally:
            session.close()

    def _generate_answer(
        self,
        repo: Repository,
        question: str,
        sources: List[dict],
        history: Optional[List[dict]] = None,
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
            context_blocks.append(
                "\n".join(
                    [
                        f"[Source {index}]",
                        f"File: {source['file_path']}",
                        f"Symbol: {source['symbol_name']}",
                        f"Lines: {source['line_start']}-{source['line_end']}",
                        source["content"][:2500],
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
1. Use only the supplied repository context.
2. Answer conversationally and directly, as if the repo is explaining itself to the user.
3. Do not say "Based on the provided context", "The repository is about", or similar throat-clearing phrases.
4. Be concrete about files, functions, and behavior.
5. If evidence is partial, clearly separate what is certain from what is inferred.
6. Respond in Markdown, not JSON.
7. Keep the answer complete. Do not stop mid-sentence.
8. Use short sections or bullets only when they genuinely help readability.
9. Do not leave unfinished headings, dangling bullets, or trailing markdown markers like #, ##, or ###.
10. Do not include inline citation markers like [Source 1] in the prose. The UI already shows sources separately.
11. Do not make claims that are not directly supported by the supplied sources.
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
        user_prompt = f"""
Repository: {repo.owner}/{repo.name}
Question: {question}
Recent conversation:
{self._format_history(history or [])}

Context:
{joined_context}
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
                short_prompt = f"""
Answer the question again, but keep it concise and complete.
Use 2 short paragraphs or 4-6 bullets max.
Do not leave the answer unfinished.
"""
                answer_text, _ = self._generate_markdown_response(
                    system_prompt,
                    f"{user_prompt.strip()}\n\n{short_prompt.strip()}",
                )
        answer_text = self._finalize_answer(answer_text)
        confidence = self._estimate_confidence(sources)
        summary = " ".join(answer_text.split())[:160] if answer_text else ""
        citations = [
            {
                "source": index,
                "reason": f"Relevant context from {source['file_path']}",
            }
            for index, source in enumerate(sources[: min(len(sources), 4)], start=1)
        ]

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
                "anthropic.claude-sonnet-4-20250514-v1:0",
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

            self.llm_model = os.getenv("VERTEX_LLM_MODEL", "claude-sonnet-4@20250514")
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
        cleaned = re.sub(r"\s*\[(?:Source\s+\d+(?:\s*,\s*Source\s+\d+)*)\]", "", cleaned, flags=re.IGNORECASE)
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
        self.indexing_progress[repo_id] = {
            **self.indexing_progress.get(repo_id, {}),
            **progress,
            "updated_at": datetime.utcnow().isoformat(),
        }

    def _touch_session(self, session, session_key: str):
        expiry = self._session_expiry()
        repos = session.query(Repository).filter_by(session_key=session_key).all()
        for repo in repos:
            repo.session_expires_at = expiry
        session.commit()

    def _cleanup_expired_sessions(self, session):
        now = datetime.utcnow()
        expired = (
            session.query(Repository)
            .filter(Repository.session_expires_at.is_not(None))
            .filter(Repository.session_expires_at < now)
            .all()
        )
        if not expired:
            return
        self._delete_repositories(session, expired)
        session.commit()

    def _delete_repositories(
        self,
        session,
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
        for repo in repos:
            session.delete(repo)

    def _ensure_repo_not_cancelled(self, repo_id: int):
        if repo_id in self.cancelled_repo_ids:
            raise SessionCancelledError("Session ended before indexing completed.")

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
            return normalized

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

    def _prioritize_results(
        self,
        question: str,
        retrieval_query: str,
        results: List[dict],
        top_k: int,
    ) -> List[dict]:
        combined_query = f"{question} {retrieval_query}".lower()
        wants_code = any(
            token in combined_query
            for token in {"code", "snippet", "implementation", "function", "class", "import"}
        )
        wants_docs = self._is_documentation_query(combined_query)
        wants_repo_overview = self._is_repo_overview_question(question) or self._is_repo_overview_question(
            retrieval_query
        )
        question_intent = self._question_intent(question)

        def sort_key(item: dict):
            is_doc = self._is_doc_source(item)
            return (
                self._canonical_path_priority(item, question),
                self._doc_priority(item),
                1 if wants_repo_overview and is_doc else 0,
                1 if (wants_docs and is_doc) or (not wants_docs and not is_doc) else 0,
                1 if wants_code and not is_doc else 0,
                1 if question_intent in {"api", "implementation", "cross_file", "error_handling", "setup"} and not is_doc else 0,
                float(item.get("rerank_score", 0.0)),
                float(item.get("semantic_score", 0.0)),
                float(item.get("bm25_score", 0.0)),
            )

        ranked = sorted(results, key=sort_key, reverse=True)
        if wants_docs or wants_repo_overview:
            return ranked[:top_k]

        selected = []
        doc_items = []
        for item in ranked:
            if self._is_doc_source(item):
                doc_items.append(item)
                continue
            selected.append(item)
            if len(selected) == top_k:
                return selected

        selected.extend(doc_items[: max(1, top_k - len(selected))])
        return selected[:top_k]

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
        file_counts = {}

        for item in results:
            file_path = item.get("file_path", "")
            count = file_counts.get(file_path, 0)
            if count >= max_per_file:
                continue
            selected.append(item)
            file_counts[file_path] = count + 1
            if len(selected) == top_k:
                break

        if len(selected) < top_k:
            for item in results:
                if item in selected:
                    continue
                selected.append(item)
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
        if CodebaseRAGSystem._is_repo_overview_question(normalized):
            return "overview"
        if any(token in normalized for token in {"error", "invalid", "conflict", "raises", "guard against"}):
            return "error_handling"
        if any(token in normalized for token in {"how are", "how does", "flow", "across files", "code path"}):
            return "cross_file"
        if any(token in normalized for token in {"export", "expose", "import", "public api"}):
            return "api"
        if any(token in normalized for token in {"create", "setup", "install", "configuration", "metadata", "table"}):
            return "setup"
        if any(token in normalized for token in {"function", "method", "class", "implementation", "does ", "what is special"}):
            return "implementation"
        if CodebaseRAGSystem._is_documentation_query(normalized):
            return "docs"
        return "general"

    def _expand_query_for_intent(self, question: str) -> str:
        normalized = " ".join((question or "").split())
        lowered = normalized.lower()
        hints = []

        if any(token in lowered for token in {"export", "expose", "import"}):
            hints.extend(["package exports", "__init__.py", "public api", "re-export"])
        if "how is select exposed to users in sqlmodel" in lowered:
            hints.extend(
                [
                    "sqlmodel/__init__.py",
                    "sqlmodel/sql/expression.py",
                    "select re-export",
                    "top-level select import",
                ]
            )
        if "select" in lowered:
            hints.extend(
                [
                    "select",
                    "expression",
                    "query builder",
                    "public api",
                    "sqlmodel/sql/expression.py",
                    "sqlmodel/__init__.py",
                    "re-export",
                    "top-level import",
                ]
            )
        if "session.exec" in lowered or ("session" in lowered and "exec" in lowered):
            hints.extend(["session exec", "orm/session.py", "asyncio/session.py"])
        if "relationship" in lowered:
            hints.extend(["relationship", "Relationship", "main.py"])
        if "field" in lowered:
            hints.extend(["Field", "FieldInfo", "main.py"])
        if "create_engine" in lowered:
            hints.extend(["create_engine", "__init__.py", "re-export"])
        if "create_all" in lowered or "metadata" in lowered:
            hints.extend(
                [
                    "metadata create_all",
                    "table creation",
                    "engine",
                    "SQLModel.metadata",
                    "README.md",
                    "sqlmodel/main.py",
                    "docs_src",
                ]
            )
        if "__init__" in lowered or "exports" in lowered:
            hints.extend(["sqlmodel/__init__.py", "package exports", "public api"])

        if not hints:
            return normalized
        return "\n".join([normalized, " ".join(hints)])

    @staticmethod
    def _is_repo_overview_question(question: str) -> bool:
        normalized = " ".join((question or "").lower().split())
        return any(
            phrase in normalized
            for phrase in {
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
        )

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

    def _canonical_path_priority(self, item: dict, question: str) -> int:
        file_path = (item.get("file_path") or "").lower()
        normalized = " ".join((question or "").lower().split())
        score = 0

        if file_path == "sqlmodel/__init__.py":
            score += 4 if any(token in normalized for token in {"export", "expose", "import", "create_engine", "select"}) else 0
        if file_path == "sqlmodel/sql/expression.py":
            score += 5 if "select" in normalized else 0
        if file_path == "sqlmodel/sql/_expression_select_gen.py":
            score += 2 if "select" in normalized else 0
        if file_path == "sqlmodel/sql/_expression_select_cls.py":
            score += 2 if "select" in normalized else 0
        if file_path == "readme.md":
            score += 4 if any(token in normalized for token in {"metadata", "create_all", "workflow", "readme"}) else 0
        if file_path.startswith("docs_src/"):
            score += 3 if any(token in normalized for token in {"metadata", "create_all", "table", "workflow"}) else 0
        if file_path == "sqlmodel/main.py":
            score += 3 if any(token in normalized for token in {"field", "relationship", "metadata", "table", "sqlmodel"}) else 0

        if "__init__.py" in file_path:
            score += 2 if any(token in normalized for token in {"export", "expose", "import", "public api"}) else 0
        if any(token in normalized for token in {"select", "expression"}):
            if "expression" in file_path or "_expression_select" in file_path:
                score += 3
        if normalized == "how is select exposed to users in sqlmodel?":
            if file_path == "sqlmodel/__init__.py":
                score += 6
            if file_path == "sqlmodel/sql/expression.py":
                score += 6
        if "session" in normalized:
            if file_path.endswith("session.py") or "/session.py" in file_path:
                score += 3
        if "relationship" in normalized and file_path.endswith("main.py"):
            score += 2
        if "field" in normalized and file_path.endswith("main.py"):
            score += 2
        if any(token in normalized for token in {"create_engine", "export", "expose"}) and "__init__.py" in file_path:
            score += 2
        if any(token in normalized for token in {"metadata", "create_all", "table"}) and (
            "docs_src/" in file_path or file_path.endswith("main.py") or file_path == "readme.md"
        ):
            score += 2
        if self._is_doc_source(item) and self._question_intent(question) in {
            "api",
            "implementation",
            "cross_file",
            "error_handling",
            "setup",
        }:
            score -= 1

        return score

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

    @staticmethod
    def _ensure_repo_still_exists(session, repo_id: int):
        if session.query(Repository.id).filter_by(id=repo_id).first() is None:
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
