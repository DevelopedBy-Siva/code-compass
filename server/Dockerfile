# syntax=docker/dockerfile:1.7
# The GPU runtime already contains the matching CUDA, cuDNN, and PyTorch
# libraries. A multi-stage Python build would duplicate several GB of GPU
# wheels without making the final image smaller.
FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime

ARG PRELOAD_MODELS=1
ARG EMBEDDING_MODEL_ID=Qwen/Qwen3-Embedding-0.6B
ARG RERANKER_MODEL_ID=Qwen/Qwen3-Reranker-0.6B

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=8080 \
    HF_HOME=/opt/ml/model-cache \
    REPO_CACHE_DIR=/opt/ml/codecompass/repos \
    EMBEDDING_MODEL_ID=${EMBEDDING_MODEL_ID} \
    RERANKER_MODEL_ID=${RERANKER_MODEL_ID}

RUN apt-get update \
    && apt-get install -y --no-install-recommends git \
    && rm -rf /var/lib/apt/lists/* \
    && groupadd --gid 10001 app \
    && useradd --uid 10001 --gid app --create-home --shell /usr/sbin/nologin app

WORKDIR /app
COPY requirements.txt /app/requirements.txt
# PyTorch is supplied by the CUDA base image. Keeping it pinned in the normal
# requirements file still gives local and CI environments a reproducible version.
RUN grep -v '^torch==' /app/requirements.txt > /tmp/container-requirements.txt \
    && pip install -r /tmp/container-requirements.txt \
    && rm /tmp/container-requirements.txt

COPY --chown=app:app . /app
RUN chmod +x /app/entrypoint.sh \
    && mkdir -p /opt/ml/input /opt/ml/model /opt/ml/model-cache /opt/ml/output /opt/ml/codecompass/repos \
    && chown -R app:app /opt/ml/input /opt/ml/model /opt/ml/model-cache /opt/ml/output /opt/ml/codecompass

USER app

RUN python - <<'PY'
import os
import shutil
from pathlib import Path

from src.repo_fetcher import RepoFetcher

cache_dir = Path(os.environ["REPO_CACHE_DIR"])
fetcher = RepoFetcher(base_dir=str(cache_dir))
probe_dir = fetcher.base_dir / ".write-probe"
probe_dir.mkdir(exist_ok=True)
(probe_dir / "ok").write_text("ok", encoding="utf-8")
shutil.rmtree(probe_dir)
PY

RUN if [ "$PRELOAD_MODELS" = "1" ]; then python /app/preload_models.py; fi

# Production builds disable model-hub network checks only after the snapshots
# have been downloaded. Development builds with PRELOAD_MODELS=0 stay online.
ENV HF_HUB_OFFLINE=${PRELOAD_MODELS} \
    TRANSFORMERS_OFFLINE=${PRELOAD_MODELS}

EXPOSE 8080
STOPSIGNAL SIGTERM

HEALTHCHECK --interval=30s --timeout=5s --start-period=180s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8080/ping', timeout=4)" || exit 1

ENTRYPOINT ["/app/entrypoint.sh"]
