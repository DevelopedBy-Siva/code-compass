import os
from datetime import datetime
from pathlib import Path

from sqlalchemy import (
    JSON,
    Column,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
    create_engine,
    inspect,
    text,
)
from sqlalchemy.orm import declarative_base, relationship, sessionmaker

Base = declarative_base()
_ENGINE_CACHE = {}
_SESSION_FACTORY_CACHE = {}
SERVER_DIR = Path(__file__).resolve().parents[1]


class Repository(Base):
    __tablename__ = "repositories"

    id = Column(Integer, primary_key=True)
    github_url = Column(String(1024), nullable=False, unique=True)
    source_url = Column(String(1024))
    session_key = Column(String(255), index=True)
    session_expires_at = Column(DateTime)
    owner = Column(String(255), nullable=False)
    name = Column(String(255), nullable=False)
    branch = Column(String(255), nullable=False, default="main")
    local_path = Column(String(1024))
    status = Column(String(64), nullable=False, default="queued")
    error_message = Column(Text)
    file_count = Column(Integer, nullable=False, default=0)
    chunk_count = Column(Integer, nullable=False, default=0)
    indexed_at = Column(DateTime)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    chunks = relationship(
        "CodeChunk", back_populates="repository", cascade="all, delete-orphan"
    )
    chat_turns = relationship(
        "ChatTurn", back_populates="repository", cascade="all, delete-orphan"
    )


class CodeChunk(Base):
    __tablename__ = "code_chunks"

    id = Column(Integer, primary_key=True)
    repository_id = Column(Integer, ForeignKey("repositories.id"), nullable=False)
    file_path = Column(String(1024), nullable=False)
    language = Column(String(64), nullable=False)
    symbol_name = Column(String(255))
    symbol_type = Column(String(128), nullable=False, default="chunk")
    line_start = Column(Integer, nullable=False)
    line_end = Column(Integer, nullable=False)
    signature = Column(Text)
    content = Column(Text, nullable=False)
    searchable_text = Column(Text, nullable=False)
    metadata_json = Column(JSON, nullable=False, default=dict)
    embedding_id = Column(Integer)
    rerank_score = Column(Float)
    created_at = Column(DateTime, default=datetime.utcnow)

    repository = relationship("Repository", back_populates="chunks")


class ChatTurn(Base):
    __tablename__ = "chat_turns"

    id = Column(Integer, primary_key=True)
    repository_id = Column(Integer, ForeignKey("repositories.id"), nullable=False)
    role = Column(String(32), nullable=False)
    content = Column(Text, nullable=False)
    answer_json = Column(JSON)
    created_at = Column(DateTime, default=datetime.utcnow)

    repository = relationship("Repository", back_populates="chat_turns")


def init_db(database_url: str = None):
    if database_url is None:
        database_url = os.getenv("DATABASE_URL", "sqlite:///./codebase_rag.db")

    database_url = resolve_database_url(database_url)
    if database_url in _ENGINE_CACHE:
        return _ENGINE_CACHE[database_url], _SESSION_FACTORY_CACHE[database_url]

    connect_args = {"check_same_thread": False} if database_url.startswith("sqlite") else {}
    engine = create_engine(database_url, echo=False, connect_args=connect_args)
    Base.metadata.create_all(engine)
    _ensure_runtime_columns(engine)
    session_local = sessionmaker(bind=engine)
    _ENGINE_CACHE[database_url] = engine
    _SESSION_FACTORY_CACHE[database_url] = session_local
    return engine, session_local


def resolve_database_url(database_url: str) -> str:
    if not database_url.startswith("sqlite:///"):
        return database_url

    sqlite_path = database_url.removeprefix("sqlite:///")
    if sqlite_path == ":memory:":
        return database_url

    path = Path(sqlite_path)
    if not path.is_absolute():
        path = SERVER_DIR / path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.touch(exist_ok=True)
    return f"sqlite:///{path.resolve()}"


def _ensure_runtime_columns(engine):
    inspector = inspect(engine)
    if "repositories" not in inspector.get_table_names():
        return

    existing = {column["name"] for column in inspector.get_columns("repositories")}
    alterations = {
        "source_url": "ALTER TABLE repositories ADD COLUMN source_url VARCHAR(1024)",
        "session_key": "ALTER TABLE repositories ADD COLUMN session_key VARCHAR(255)",
        "session_expires_at": "ALTER TABLE repositories ADD COLUMN session_expires_at DATETIME",
    }

    with engine.begin() as connection:
        for column_name, statement in alterations.items():
            if column_name not in existing:
                connection.execute(text(statement))


def get_db_session(database_url: str = None):
    _, session_local = init_db(database_url)
    return session_local()
