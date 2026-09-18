"""Download model snapshots into the image during a production build."""

import os

from huggingface_hub import snapshot_download


for variable, default in (
    ("EMBEDDING_MODEL_ID", "Qwen/Qwen3-Embedding-0.6B"),
    ("RERANKER_MODEL_ID", "Qwen/Qwen3-Reranker-0.6B"),
):
    model_id = os.getenv(variable, default)
    print(f"[build] Preloading {model_id}", flush=True)
    snapshot_download(repo_id=model_id)
