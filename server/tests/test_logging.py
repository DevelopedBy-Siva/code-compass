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
    APPLICATION_CATEGORIES,
    RequestIdMiddleware,
    THIRD_PARTY_LOG_LEVELS,
    _ContextFilter,
    bind_request_id,
    get_logger,
    profiling_enabled,
    reset_request_id,
)


class LoggingTests(unittest.TestCase):
    def test_only_operational_application_categories_are_allowed(self):
        self.assertEqual(
            APPLICATION_CATEGORIES,
            {"startup", "index", "query", "error"},
        )
        with self.assertRaisesRegex(ValueError, "Unknown log category"):
            get_logger("app")

    def test_uncategorized_dependency_record_is_retained_as_error(self):
        record = logging.LogRecord("httpx", logging.WARNING, "", 0, "retry", (), None)

        retained = _ContextFilter().filter(record)

        self.assertTrue(retained)
        self.assertEqual(record.request_id, "system")
        self.assertEqual(record.category, "error")

    def test_routine_dependency_record_is_dropped(self):
        record = logging.LogRecord(
            "qdrant_client.http.api_client",
            logging.INFO,
            "",
            0,
            "request",
            (),
            None,
        )

        self.assertFalse(_ContextFilter().filter(record))

    def test_noisy_dependencies_are_warning_or_higher(self):
        expected = {
            "httpx",
            "httpcore",
            "urllib3",
            "qdrant_client",
            "botocore",
            "boto3",
            "transformers",
            "huggingface_hub",
        }

        self.assertTrue(expected.issubset(THIRD_PARTY_LOG_LEVELS))
        self.assertTrue(
            all(
                THIRD_PARTY_LOG_LEVELS[name] >= logging.WARNING
                for name in expected
            )
        )

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
