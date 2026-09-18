#!/usr/bin/env sh
set -eu

python - <<'PY'
import json
import os
import shutil
import socket
import sys
import tempfile
import urllib.error
import urllib.request
from pathlib import Path

CHECKS = []


def ok(name, detail=""):
    CHECKS.append({"name": name, "status": "ok", "detail": detail})
    print(f"[ok] {name}{': ' + detail if detail else ''}", flush=True)


def fail(name, detail):
    CHECKS.append({"name": name, "status": "failed", "detail": detail})
    print(f"[failed] {name}: {detail}", flush=True)
    raise SystemExit(1)


def require_env(name):
    value = os.getenv(name, "").strip()
    if not value:
        fail(f"env {name}", "missing")
    ok(f"env {name}", value if name not in {"QDRANT_API_KEY"} else "set")
    return value


def assert_writable(path_value, label):
    path = Path(path_value)
    try:
        path.mkdir(parents=True, exist_ok=True)
        probe = path / ".code-compass-write-check"
        probe.write_text("ok", encoding="utf-8")
        probe.unlink()
    except Exception as exc:
        fail(f"writable {label}", f"{path}: {exc}")
    stat = path.stat()
    ok(
        f"writable {label}",
        f"{path} uid={stat.st_uid} gid={stat.st_gid} mode={oct(stat.st_mode & 0o777)}",
    )


def http_ok(url, method="GET", payload=None):
    data = None
    headers = {}
    if payload is not None:
        data = json.dumps(payload).encode("utf-8")
        headers["Content-Type"] = "application/json"
    request = urllib.request.Request(url, data=data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(request, timeout=15) as response:
            body = response.read().decode("utf-8")
            if response.status >= 400:
                fail(f"http {method} {url}", f"status={response.status} body={body}")
            return body
    except urllib.error.HTTPError as exc:
        fail(f"http {method} {url}", f"status={exc.code} body={exc.read().decode('utf-8', 'ignore')}")
    except Exception as exc:
        fail(f"http {method} {url}", str(exc))


if os.geteuid() == 0:
    fail("runtime user", "container is running as root")
if os.geteuid() != 10001:
    fail("runtime user", f"expected uid 10001, got {os.geteuid()}")
ok("runtime user", f"uid={os.geteuid()} gid={os.getegid()}")

require_env("APP_ENV")
require_env("AWS_REGION")
require_env("BEDROCK_MODEL_ID")
require_env("CORS_ORIGINS")
require_env("EMBEDDING_MODEL_ID")
require_env("HF_HOME")
require_env("PORT")
require_env("QDRANT_COLLECTION")
require_env("QDRANT_URL")
require_env("RERANKER_MODEL_ID")
require_env("REPO_CACHE_DIR")

if not os.getenv("QDRANT_API_KEY") and not os.getenv("QDRANT_API_KEY_SECRET_ARN"):
    fail("qdrant credentials", "set QDRANT_API_KEY or QDRANT_API_KEY_SECRET_ARN")
ok("qdrant credentials", "available")

for label, path in {
    "sagemaker input": "/opt/ml/input",
    "sagemaker model": "/opt/ml/model",
    "sagemaker model cache": "/opt/ml/model-cache",
    "sagemaker output": "/opt/ml/output",
    "huggingface cache": os.environ["HF_HOME"],
    "repo cache": os.environ["REPO_CACHE_DIR"],
    "tempdir": tempfile.gettempdir(),
}.items():
    assert_writable(path, label)

from src.config import Settings
from src.repo_fetcher import RepoFetcher

settings = Settings.from_env()
ok("settings", "loaded production-equivalent environment")

fetcher = RepoFetcher(base_dir=settings.repo_cache_dir)
repo_probe = fetcher.base_dir / ".repo-create-check"
try:
    if repo_probe.exists():
        shutil.rmtree(repo_probe)
    repo_probe.mkdir(parents=True)
    (repo_probe / "README.md").write_text("# ok\n", encoding="utf-8")
    shutil.rmtree(repo_probe)
except Exception as exc:
    fail("repo cache directory", str(exc))
ok("repo cache directory", str(fetcher.base_dir))

health = http_ok("http://127.0.0.1:8080/api/health")
ok("model loading", f"application lifespan is ready: {health}")

hf_cache = Path(os.environ["HF_HOME"])
cache_entries = [path.name for path in hf_cache.iterdir()] if hf_cache.exists() else []
if os.getenv("HF_HUB_OFFLINE") == "1" and not cache_entries:
    fail("huggingface cache", "offline mode is enabled but cache is empty")
ok("huggingface cache", f"{len(cache_entries)} top-level entries")

from qdrant_client import QdrantClient

try:
    client = QdrantClient(
        url=settings.qdrant_url,
        api_key=settings.qdrant_api_key,
        timeout=settings.qdrant_timeout_seconds,
    )
    client.get_collections()
except Exception as exc:
    fail("qdrant connectivity", str(exc))
ok("qdrant connectivity", settings.qdrant_url)

try:
    socket.getaddrinfo("bedrock-runtime." + settings.aws_region + ".amazonaws.com", 443)
except Exception as exc:
    fail("bedrock dns", str(exc))
ok("bedrock dns", settings.aws_region)

import boto3
from botocore.config import Config as BotoConfig

try:
    bedrock = boto3.client(
        "bedrock-runtime",
        region_name=settings.aws_region,
        config=BotoConfig(connect_timeout=5, read_timeout=30, retries={"max_attempts": 1}),
    )
    response = bedrock.converse(
        modelId=settings.bedrock_model_id,
        messages=[{"role": "user", "content": [{"text": "Reply with ok."}]}],
        inferenceConfig={"maxTokens": 4, "temperature": 0},
    )
    text = "".join(
        block.get("text", "")
        for block in response.get("output", {}).get("message", {}).get("content", [])
    ).strip()
    if not text:
        fail("bedrock connectivity", "empty response")
except Exception as exc:
    fail("bedrock connectivity", str(exc))
ok("bedrock connectivity", settings.bedrock_model_id)

print("[ok] container checks complete", flush=True)
PY
