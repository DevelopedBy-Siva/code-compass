"""Small, request-aware logging helpers for Code Compass."""

import logging
import os
from contextvars import ContextVar, Token
from typing import Optional
from uuid import uuid4


_request_id: ContextVar[str] = ContextVar("request_id", default="system")
_configured = False


class _ContextFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        record.request_id = _request_id.get()
        if not hasattr(record, "category"):
            record.category = "app"
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
    _configured = True


def get_logger(category: str) -> logging.LoggerAdapter:
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
