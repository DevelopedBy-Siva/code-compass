import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional

from dotenv import load_dotenv
from fastapi import BackgroundTasks, Depends, FastAPI, Header, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, HttpUrl, ValidationError

from src.config import Settings
from src.rag_system import CodebaseRAGSystem

load_dotenv(Path(__file__).with_name(".env"))
logger = logging.getLogger("code_compass")


class RepoIndexRequest(BaseModel):
    github_url: HttpUrl
    reindex: bool = False


class MessageTurn(BaseModel):
    role: Literal["user", "assistant"]
    content: str = Field(..., min_length=1, max_length=4000)


class QueryRequest(BaseModel):
    repo_id: int = Field(..., ge=1)
    question: str = Field(..., min_length=3)
    top_k: int = Field(8, ge=3, le=12)
    history: List[MessageTurn] = Field(default_factory=list, max_length=8)


class SageMakerInvocation(BaseModel):
    action: Literal[
        "list_repositories",
        "get_repository",
        "index_repository",
        "query",
        "end_session",
    ]
    session_id: str = Field(..., min_length=8, max_length=256)
    payload: Dict[str, Any] = Field(default_factory=dict)


@asynccontextmanager
async def lifespan(application: FastAPI):
    settings = Settings.from_env()
    logging.basicConfig(
        level=getattr(logging, settings.log_level, logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    application.state.ready = False
    logger.info("Initializing Code Compass runtime")
    application.state.rag_system = CodebaseRAGSystem(settings=settings)
    application.state.ready = True
    logger.info("Code Compass runtime is ready")
    try:
        yield
    finally:
        application.state.ready = False
        logger.info("Shutting down Code Compass runtime")
        application.state.rag_system.close()


def create_app() -> FastAPI:
    settings = Settings.from_env(require_external_services=False)
    application = FastAPI(
        title="Codebase RAG API",
        description="Index GitHub repositories and answer questions with grounded citations.",
        version="2.1.0",
        lifespan=lifespan,
    )
    application.state.ready = False
    application.add_middleware(
        CORSMiddleware,
        allow_origins=list(settings.cors_origins),
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    return application


app = create_app()


def get_rag_system(request: Request) -> CodebaseRAGSystem:
    rag_system = getattr(request.app.state, "rag_system", None)
    if rag_system is None or not request.app.state.ready:
        raise HTTPException(status_code=503, detail="Service is starting")
    return rag_system


def require_session_id(x_session_id: Optional[str] = Header(None, alias="X-Session-Id")) -> str:
    if not x_session_id or not x_session_id.strip():
        raise HTTPException(status_code=400, detail="Missing session id")
    session_id = x_session_id.strip()
    if len(session_id) > 256:
        raise HTTPException(status_code=400, detail="Session id is too long")
    return session_id


@app.get("/")
async def root():
    return {"status": "online", "message": "Codebase RAG API is running"}


def _health(request: Request):
    if not getattr(request.app.state, "ready", False):
        raise HTTPException(status_code=503, detail="Service is starting")
    return {"status": "ok"}


@app.get("/api/health")
async def health(request: Request):
    return _health(request)


@app.get("/ping")
@app.post("/ping")
async def sagemaker_health(request: Request):
    return _health(request)


@app.get("/api/repos")
async def list_repositories(
    session_id: str = Depends(require_session_id),
    rag_system: CodebaseRAGSystem = Depends(get_rag_system),
):
    return rag_system.list_repositories_for_session(session_id)


@app.get("/api/repos/{repo_id}")
async def get_repository(
    repo_id: int,
    session_id: str = Depends(require_session_id),
    rag_system: CodebaseRAGSystem = Depends(get_rag_system),
):
    repo = rag_system.get_repository_for_session(repo_id, session_id)
    if not repo:
        raise HTTPException(status_code=404, detail="Repository not found")
    return repo


@app.post("/api/repos/index")
async def queue_repository_index(
    body: RepoIndexRequest,
    background_tasks: BackgroundTasks,
    session_id: str = Depends(require_session_id),
    rag_system: CodebaseRAGSystem = Depends(get_rag_system),
):
    try:
        repo = rag_system.create_or_reset_repository(
            str(body.github_url), session_id, reindex=body.reindex
        )
        cache_hit = repo.status == "indexed" and repo.cache_hit
        if repo.status != "indexed":
            background_tasks.add_task(rag_system.index_repository, repo.id)
        return {
            "success": True,
            "message": (
                "Loaded the existing repository index"
                if cache_hit
                else "Repository indexing started"
            ),
            "repo": rag_system.get_repository_for_session(repo.id, session_id),
        }
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/api/query")
async def query_repository(
    body: QueryRequest,
    session_id: str = Depends(require_session_id),
    rag_system: CodebaseRAGSystem = Depends(get_rag_system),
):
    try:
        return rag_system.answer_question(
            repo_id=body.repo_id,
            session_key=session_id,
            question=body.question.strip(),
            top_k=body.top_k,
            history=body.history,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Query failed")
        raise HTTPException(status_code=500, detail="Query failed") from exc


@app.post("/api/session/end")
async def end_session(
    session_id: str = Query(..., min_length=8),
    rag_system: CodebaseRAGSystem = Depends(get_rag_system),
):
    rag_system.end_session(session_id)
    return {"success": True}


@app.post("/invocations")
async def invoke(
    invocation: SageMakerInvocation,
    background_tasks: BackgroundTasks,
    rag_system: CodebaseRAGSystem = Depends(get_rag_system),
):
    """Dispatch SageMaker's single inference route to the existing operations."""
    session_id = invocation.session_id.strip()
    payload = invocation.payload

    if invocation.action == "list_repositories":
        return rag_system.list_repositories_for_session(session_id)
    if invocation.action == "get_repository":
        try:
            repo_id = int(payload.get("repo_id", 0))
        except (TypeError, ValueError) as exc:
            raise HTTPException(status_code=400, detail="Invalid repository id") from exc
        repo = rag_system.get_repository_for_session(repo_id, session_id)
        if not repo:
            raise HTTPException(status_code=404, detail="Repository not found")
        return repo
    if invocation.action == "index_repository":
        try:
            body = RepoIndexRequest.model_validate(payload)
        except ValidationError as exc:
            raise HTTPException(status_code=400, detail="Invalid indexing payload") from exc
        return await queue_repository_index(body, background_tasks, session_id, rag_system)
    if invocation.action == "query":
        try:
            body = QueryRequest.model_validate(payload)
        except ValidationError as exc:
            raise HTTPException(status_code=400, detail="Invalid query payload") from exc
        return await query_repository(body, session_id, rag_system)

    rag_system.end_session(session_id)
    return {"success": True}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "server_app:app",
        host="0.0.0.0",
        port=int(os.getenv("PORT", "8080")),
        reload=False,
        workers=1,
    )
