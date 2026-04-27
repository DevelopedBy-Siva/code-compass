import os
from pathlib import Path
from typing import List, Literal, Optional

from fastapi import BackgroundTasks, Depends, FastAPI, Header, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, HttpUrl
from dotenv import load_dotenv

from src.bedrock_claude import BedrockTransientError, is_bedrock_retryable_error
from src.rag_system import CodebaseRAGSystem

load_dotenv(Path(__file__).with_name(".env"))


app = FastAPI(
    title="Codebase RAG API",
    description="Index GitHub repositories and answer natural-language questions with grounded citations.",
    version="2.0.0",
)

cors_origins = [
    origin.strip()
    for origin in os.getenv("CORS_ORIGINS", "http://localhost:3000").split(",")
    if origin.strip()
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

rag_system: Optional[CodebaseRAGSystem] = None


class RepoIndexRequest(BaseModel):
    github_url: HttpUrl


class QueryRequest(BaseModel):
    repo_id: int = Field(..., ge=1)
    question: str = Field(..., min_length=3)
    top_k: int = Field(8, ge=3, le=12)
    history: List["MessageTurn"] = Field(default_factory=list, max_length=8)


class MessageTurn(BaseModel):
    role: Literal["user", "assistant"]
    content: str = Field(..., min_length=1, max_length=4000)


def require_session_id(x_session_id: Optional[str] = Header(None, alias="X-Session-Id")) -> str:
    if not x_session_id or not x_session_id.strip():
        raise HTTPException(status_code=400, detail="Missing session id")
    return x_session_id.strip()


@app.on_event("startup")
def startup():
    global rag_system
    Path("./data").mkdir(exist_ok=True)
    rag_system = CodebaseRAGSystem()


@app.get("/")
async def root():
    return {
        "status": "online",
        "message": "Codebase RAG API is running",
    }


@app.get("/api/health")
async def health():
    return {
        "status": "ok",
    }


@app.get("/api/repos")
async def list_repositories(session_id: str = Depends(require_session_id)):
    return rag_system.list_repositories_for_session(session_id)


@app.get("/api/repos/{repo_id}")
async def get_repository(repo_id: int, session_id: str = Depends(require_session_id)):
    repo = rag_system.get_repository_for_session(repo_id, session_id)
    if not repo:
        raise HTTPException(status_code=404, detail="Repository not found")
    return repo


@app.post("/api/repos/index")
async def queue_repository_index(
    request: RepoIndexRequest,
    background_tasks: BackgroundTasks,
    session_id: str = Depends(require_session_id),
):
    try:
        repo = rag_system.create_or_reset_repository(str(request.github_url), session_id)
        background_tasks.add_task(rag_system.index_repository, repo.id)
        return {
            "success": True,
            "message": "Repository indexing started",
            "repo": rag_system.get_repository_for_session(repo.id, session_id),
        }
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@app.post("/api/query")
async def query_repository(request: QueryRequest, session_id: str = Depends(require_session_id)):
    try:
        return rag_system.answer_question(
            repo_id=request.repo_id,
            session_key=session_id,
            question=request.question.strip(),
            top_k=request.top_k,
            history=request.history,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except BedrockTransientError as exc:
        raise HTTPException(
            status_code=429,
            detail=str(exc),
            headers={"Retry-After": os.getenv("BEDROCK_HTTP_RETRY_AFTER_SECONDS", "10")},
        )
    except Exception as exc:
        if is_bedrock_retryable_error(exc):
            raise HTTPException(
                status_code=429,
                detail=f"Bedrock throttled or was temporarily unavailable: {exc}",
                headers={"Retry-After": os.getenv("BEDROCK_HTTP_RETRY_AFTER_SECONDS", "10")},
            )
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/api/session/end")
async def end_session(session_id: str = Query(..., min_length=8)):
    rag_system.end_session(session_id)
    return {"success": True}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("server_app:app", host="0.0.0.0", port=8000, reload=True)
