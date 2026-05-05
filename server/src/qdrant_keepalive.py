import os
import threading
from typing import Optional

from src.vector_store import QdrantVectorStore


class QdrantKeepAliveScheduler:
    def __init__(self, vector_store: QdrantVectorStore):
        self.vector_store = vector_store
        self.interval_seconds = self._interval_seconds()
        self.run_on_start = self._env_flag("QDRANT_KEEPALIVE_RUN_ON_START", True)
        self.keepalive_enabled = self._env_flag("QDRANT_KEEPALIVE_ENABLED", True)
        self.enabled = self.keepalive_enabled and self.vector_store.is_remote()
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def start(self):
        if not self.enabled:
            reason = (
                "disabled by QDRANT_KEEPALIVE_ENABLED"
                if not self.keepalive_enabled
                else "set QDRANT_URL to enable remote Qdrant pings"
            )
            print(
                f"[qdrant-keepalive] Disabled; {reason}",
                flush=True,
            )
            return
        if self._thread and self._thread.is_alive():
            return

        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._run,
            name="qdrant-keepalive",
            daemon=True,
        )
        self._thread.start()
        print(
            f"[qdrant-keepalive] Started interval_seconds={self.interval_seconds}",
            flush=True,
        )

    def stop(self):
        self._stop_event.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=5)
        self._thread = None

    def _run(self):
        if self.run_on_start:
            self._ping()

        while not self._stop_event.wait(self.interval_seconds):
            self._ping()

    def _ping(self):
        try:
            stats = self.vector_store.keep_alive()
            print(
                "[qdrant-keepalive] Ping succeeded "
                f"collection={stats['collection_name']} "
                f"points={stats['total_vectors']}",
                flush=True,
            )
        except Exception as exc:
            print(f"[qdrant-keepalive] Ping failed: {exc}", flush=True)

    @staticmethod
    def _env_flag(name: str, default: bool) -> bool:
        value = os.getenv(name)
        if value is None:
            return default
        return value.strip().lower() not in {"0", "false", "no", "off"}

    @staticmethod
    def _interval_seconds() -> int:
        value = os.getenv("QDRANT_KEEPALIVE_INTERVAL_SECONDS", "43200")
        try:
            return max(60, int(value))
        except ValueError:
            return 43200
