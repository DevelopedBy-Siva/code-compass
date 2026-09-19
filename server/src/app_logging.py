"""Small, request-aware logging helpers for Code Compass."""

import logging
import os
import warnings
from contextvars import ContextVar, Token
from typing import Optional
from uuid import uuid4


_request_id: ContextVar[str] = ContextVar("request_id", default="system")
_configured = False

APPLICATION_CATEGORIES = frozenset({"startup", "index", "query", "error"})

# Keep dependency warnings and failures, but hide their request-level chatter.
THIRD_PARTY_LOG_LEVELS = {
    "boto3": logging.WARNING,
    "botocore": logging.WARNING,
    "huggingface_hub": logging.WARNING,
    "httpcore": logging.WARNING,
    "httpx": logging.WARNING,
    "py.warnings": logging.WARNING,
    "qdrant_client": logging.WARNING,
    "s3transfer": logging.WARNING,
    "transformers": logging.WARNING,
    "urllib3": logging.WARNING,
    "uvicorn": logging.WARNING,
    "uvicorn.access": logging.WARNING,
    "uvicorn.error": logging.WARNING,
}
_THIRD_PARTY_LOG_LEVELS_BY_SPECIFICITY = tuple(
    sorted(THIRD_PARTY_LOG_LEVELS.items(), key=lambda item: len(item[0]), reverse=True)
)


class _ContextFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        for logger_name, minimum_level in _THIRD_PARTY_LOG_LEVELS_BY_SPECIFICITY:
            if record.name == logger_name or record.name.startswith(f"{logger_name}."):
                if record.levelno < minimum_level:
                    return False
                break
        record.request_id = _request_id.get()
        if not hasattr(record, "category"):
            # Records from dependencies have no application category. If a
            # warning or error survives their configured threshold, retain it
            # in the stream under the error category instead of inventing a
            # generic application category.
            record.category = "error"
        return True


def configure_logging(level: str = "INFO") -> None:
    """Configure the process-wide console logger once."""
    global _configured
    if _configured:
        return

    handler = logging.StreamHandler()
    handler.addFilter(_ContextFilter())
    handler.setFormatter(
        logging.Formatter(
            "%(asctime)s [%(levelname)s] [%(request_id)s] [%(category)s] %(message)s",
            datefmt="%Y-%m-%dT%H:%M:%S%z",
        )
    )
    root_logger = logging.getLogger()
    root_logger.handlers.clear()
    root_logger.addHandler(handler)
    root_logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    logging.captureWarnings(True)
    warnings.filterwarnings(
        "ignore",
        message=r"Language\(path, name\) is deprecated.*",
        category=FutureWarning,
        module=r"tree_sitter",
    )
    for logger_name, logger_level in THIRD_PARTY_LOG_LEVELS.items():
        dependency_logger = logging.getLogger(logger_name)
        dependency_logger.setLevel(logger_level)
        dependency_logger.handlers.clear()
        dependency_logger.propagate = True
        for registered_name in tuple(logging.root.manager.loggerDict):
            if not registered_name.startswith(f"{logger_name}."):
                continue
            child_logger = logging.getLogger(registered_name)
            child_logger.handlers.clear()
            child_logger.propagate = True
    _configured = True


def get_logger(category: str) -> logging.LoggerAdapter:
    if category not in APPLICATION_CATEGORIES:
        allowed = ", ".join(sorted(APPLICATION_CATEGORIES))
        raise ValueError(f"Unknown log category {category!r}; expected one of: {allowed}")
    return logging.LoggerAdapter(
        logging.getLogger(f"code_compass.{category}"),
        {"category": category},
    )


def bind_request_id(request_id: Optional[str] = None) -> Token:
    return _request_id.set(request_id or uuid4().hex[:8])


def reset_request_id(token: Token) -> None:
    _request_id.reset(token)


def new_request_id() -> str:
    return uuid4().hex[:8]


def profiling_enabled() -> bool:
    return os.getenv("ENABLE_PROFILING", "false").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def fields(**values: object) -> str:
    """Render compact, single-line key/value fields."""
    rendered = []
    for key, value in values.items():
        text = " ".join(str(value).split())
        rendered.append(f"{key}={text}")
    return " ".join(rendered)


class RequestIdMiddleware:
    """Bind one generated request ID for the full HTTP request lifecycle."""

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        request_id = new_request_id()
        scope.setdefault("state", {})["request_id"] = request_id
        token = bind_request_id(request_id)

        async def send_with_request_id(message):
            if message["type"] == "http.response.start":
                headers = list(message.get("headers", []))
                headers.append((b"x-request-id", request_id.encode("ascii")))
                message["headers"] = headers
            await send(message)

        try:
            await self.app(scope, receive, send_with_request_id)
        finally:
            reset_request_id(token)
