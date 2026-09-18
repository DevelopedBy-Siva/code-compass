import asyncio
import io
import logging
import re
import sys
import unittest
from pathlib import Path
from unittest.mock import patch


SERVER_ROOT = Path(__file__).resolve().parents[1]
if str(SERVER_ROOT) not in sys.path:
    sys.path.insert(0, str(SERVER_ROOT))

from src.app_logging import (
    RequestIdMiddleware,
    _ContextFilter,
    bind_request_id,
    get_logger,
    profiling_enabled,
    reset_request_id,
)


class LoggingTests(unittest.TestCase):
    def test_log_record_contains_request_id_and_category(self):
        stream = io.StringIO()
        handler = logging.StreamHandler(stream)
        handler.addFilter(_ContextFilter())
        handler.setFormatter(
            logging.Formatter("[%(request_id)s] [%(category)s] %(message)s")
        )
        logger = logging.getLogger("code_compass")
        logger.handlers = [handler]
        logger.propagate = False
        logger.setLevel(logging.INFO)

        token = bind_request_id("8c1d2f4a")
        try:
            get_logger("query").info("retrieved=8")
        finally:
            reset_request_id(token)
            logger.handlers = []
            logger.propagate = True

        self.assertEqual(stream.getvalue().strip(), "[8c1d2f4a] [query] retrieved=8")

    def test_middleware_generates_and_returns_request_id(self):
        observed = {}

        async def app(scope, receive, send):
            record = logging.LogRecord("test", logging.INFO, "", 0, "ok", (), None)
            _ContextFilter().filter(record)
            observed["request_id"] = record.request_id
            await send({"type": "http.response.start", "status": 200, "headers": []})
            await send({"type": "http.response.body", "body": b""})

        sent = []

        async def receive():
            return {"type": "http.request", "body": b"", "more_body": False}

        async def send(message):
            sent.append(message)

        asyncio.run(
            RequestIdMiddleware(app)(
                {"type": "http", "state": {}}, receive, send
            )
        )
        response_headers = dict(sent[0]["headers"])
        request_id = response_headers[b"x-request-id"].decode("ascii")

        self.assertRegex(request_id, re.compile(r"^[0-9a-f]{8}$"))
        self.assertEqual(observed["request_id"], request_id)

    def test_profiling_defaults_off_and_accepts_true(self):
        with patch.dict("os.environ", {}, clear=True):
            self.assertFalse(profiling_enabled())
        with patch.dict("os.environ", {"ENABLE_PROFILING": "true"}, clear=True):
            self.assertTrue(profiling_enabled())


if __name__ == "__main__":
    unittest.main()
