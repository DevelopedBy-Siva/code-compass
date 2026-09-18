#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
image_name="${IMAGE_NAME:-code-compass-backend}"
image_tag="${IMAGE_TAG:-$(git -C "$repo_root" rev-parse --short=12 HEAD 2>/dev/null || printf 'local')}"
docker_platform="${DOCKER_PLATFORM:-linux/amd64}"
preload_models="${PRELOAD_MODELS:-1}"

docker build \
  --platform "$docker_platform" \
  --build-arg "PRELOAD_MODELS=$preload_models" \
  --build-arg "EMBEDDING_MODEL_ID=${EMBEDDING_MODEL_ID:-Qwen/Qwen3-Embedding-0.6B}" \
  --build-arg "RERANKER_MODEL_ID=${RERANKER_MODEL_ID:-Qwen/Qwen3-Reranker-0.6B}" \
  --tag "$image_name:$image_tag" \
  "$repo_root/server"

printf 'Built %s:%s for %s\n' "$image_name" "$image_tag" "$docker_platform"
