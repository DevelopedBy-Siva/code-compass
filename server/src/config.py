import json
import os
from dataclasses import dataclass
from functools import lru_cache
from typing import Optional

import boto3


def _positive_int(name: str, default: int) -> int:
    raw_value = os.getenv(name, str(default))
    try:
        value = int(raw_value)
    except ValueError as exc:
        raise RuntimeError(f"{name} must be an integer") from exc
    if value < 1:
        raise RuntimeError(f"{name} must be greater than zero")
    return value


def _secret_value(secret_arn: str, region: str) -> str:
    response = boto3.client("secretsmanager", region_name=region).get_secret_value(
        SecretId=secret_arn
    )
    value = response.get("SecretString", "")
    if not value:
        raise RuntimeError(f"Secret {secret_arn!r} does not contain SecretString")

    try:
        payload = json.loads(value)
    except json.JSONDecodeError:
        return value
    if isinstance(payload, dict):
        for key in ("api_key", "QDRANT_API_KEY", "value"):
            if payload.get(key):
                return str(payload[key])
    raise RuntimeError(
        f"Secret {secret_arn!r} must be a plain string or contain api_key"
    )


@dataclass(frozen=True)
class Settings:
    app_env: str
    aws_region: str
    bedrock_model_id: str
    cors_origins: tuple[str, ...]
    embedding_model_id: str
    log_level: str
    qdrant_api_key: Optional[str]
    qdrant_collection: str
    qdrant_timeout_seconds: int
    qdrant_upsert_batch_size: int
    qdrant_url: str
    repo_cache_dir: str
    rerank_batch_size: int
    reranker_model_id: str
    session_ttl_minutes: int
    enable_profiling: bool = False

    @classmethod
    def from_env(cls, *, require_external_services: bool = True) -> "Settings":
        region = os.getenv(
            "AWS_REGION", os.getenv("AWS_DEFAULT_REGION", "us-east-1")
        ).strip()
        qdrant_url = os.getenv("QDRANT_URL", "").strip()
        if require_external_services and not qdrant_url:
            raise RuntimeError("QDRANT_URL is required")

        api_key = os.getenv("QDRANT_API_KEY", "").strip() or None
        secret_arn = os.getenv("QDRANT_API_KEY_SECRET_ARN", "").strip()
        if not api_key and secret_arn and require_external_services:
            api_key = _secret_value(secret_arn, region)

        origins = tuple(
            origin.strip()
            for origin in os.getenv("CORS_ORIGINS", "http://localhost:3000").split(",")
            if origin.strip()
        )
        if not origins:
            raise RuntimeError("CORS_ORIGINS must contain at least one origin")

        return cls(
            app_env=os.getenv("APP_ENV", "local").lower(),
            aws_region=region,
            bedrock_model_id=os.getenv(
                "BEDROCK_MODEL_ID", "qwen.qwen3-coder-next"
            ).strip(),
            cors_origins=origins,
            embedding_model_id=os.getenv(
                "EMBEDDING_MODEL_ID", "Qwen/Qwen3-Embedding-0.6B"
            ).strip(),
            enable_profiling=os.getenv("ENABLE_PROFILING", "false").strip().lower()
            in {"1", "true", "yes", "on"},
            log_level=os.getenv("LOG_LEVEL", "INFO").upper(),
            qdrant_api_key=api_key,
            qdrant_collection=os.getenv(
                "QDRANT_COLLECTION",
                "code_compass_qwen3_embedding_0_6b_last_token_cache_v2",
            ).strip(),
            qdrant_timeout_seconds=_positive_int("QDRANT_TIMEOUT_SECONDS", 60),
            qdrant_upsert_batch_size=_positive_int("QDRANT_UPSERT_BATCH_SIZE", 64),
            qdrant_url=qdrant_url,
            repo_cache_dir=os.getenv("REPO_CACHE_DIR", "/opt/ml/codecompass/repos"),
            rerank_batch_size=_positive_int("RAG_RERANK_BATCH_SIZE", 4),
            reranker_model_id=os.getenv(
                "RERANKER_MODEL_ID", "Qwen/Qwen3-Reranker-0.6B"
            ).strip(),
            session_ttl_minutes=_positive_int("SESSION_TTL_MINUTES", 120),
        )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings.from_env()
