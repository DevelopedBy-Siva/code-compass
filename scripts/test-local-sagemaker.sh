#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
image_name="${IMAGE_NAME:-code-compass-backend}"
image_tag="${IMAGE_TAG:-$(git -C "$repo_root" rev-parse --short=12 HEAD 2>/dev/null || printf 'local')}"
container_name="${LOCAL_SAGEMAKER_CONTAINER_NAME:-code-compass-sagemaker-local}"
timeout_seconds="${LOCAL_SAGEMAKER_TIMEOUT_SECONDS:-${SAGEMAKER_STARTUP_TIMEOUT_SECONDS:-900}}"
ping_request_timeout_seconds="${LOCAL_SAGEMAKER_PING_TIMEOUT_SECONDS:-2}"
ping_poll_interval_seconds="${LOCAL_SAGEMAKER_PING_POLL_INTERVAL_SECONDS:-5}"
port="${PORT:-8080}"

required_env=(
  AWS_REGION
  QDRANT_URL
  CORS_ORIGINS
)

for key in "${required_env[@]}"; do
  if [[ -z "${!key:-}" ]]; then
    printf 'Set %s before running local SageMaker compatibility tests\n' "$key" >&2
    exit 1
  fi
done

if [[ -z "${QDRANT_API_KEY:-}" && -z "${QDRANT_API_KEY_SECRET_ARN:-}" ]]; then
  printf 'Set QDRANT_API_KEY_SECRET_ARN or QDRANT_API_KEY before running local SageMaker compatibility tests\n' >&2
  exit 1
fi

if ! command -v docker >/dev/null; then
  printf 'docker is required\n' >&2
  exit 1
fi

if ! command -v curl >/dev/null; then
  printf 'curl is required\n' >&2
  exit 1
fi

cleanup() {
  docker rm -f "$container_name" >/dev/null 2>&1 || true
}

print_logs() {
  printf '\n--- %s logs ---\n' "$container_name" >&2
  docker logs "$container_name" >&2 || true
  printf -- '--- end logs ---\n' >&2
}

trap cleanup EXIT

printf 'Building local SageMaker parity image %s:%s\n' "$image_name" "$image_tag"
"$repo_root/scripts/build.sh"

cleanup

env_args=(
  -e APP_ENV="${APP_ENV:-production}"
  -e REQUIRE_CUDA="${REQUIRE_CUDA:-0}"
  -e AWS_REGION="$AWS_REGION"
  -e BEDROCK_MODEL_ID="${BEDROCK_MODEL_ID:-qwen.qwen3-coder-next}"
  -e CORS_ORIGINS="$CORS_ORIGINS"
  -e EMBEDDING_MODEL_ID="${EMBEDDING_MODEL_ID:-Qwen/Qwen3-Embedding-0.6B}"
  -e HF_HOME="${HF_HOME:-/opt/ml/model-cache}"
  -e LOG_LEVEL="${LOG_LEVEL:-INFO}"
  -e PORT=8080
  -e QDRANT_COLLECTION="${QDRANT_COLLECTION:-code_compass_qwen3_embedding_0_6b_last_token_cache_v2}"
  -e QDRANT_TIMEOUT_SECONDS="${QDRANT_TIMEOUT_SECONDS:-60}"
  -e QDRANT_UPSERT_BATCH_SIZE="${QDRANT_UPSERT_BATCH_SIZE:-64}"
  -e QDRANT_URL="$QDRANT_URL"
  -e RAG_RERANK_BATCH_SIZE="${RAG_RERANK_BATCH_SIZE:-4}"
  -e RERANKER_MODEL_ID="${RERANKER_MODEL_ID:-Qwen/Qwen3-Reranker-0.6B}"
  -e REPO_CACHE_DIR="${REPO_CACHE_DIR:-/opt/ml/codecompass/repos}"
  -e SESSION_TTL_MINUTES="${SESSION_TTL_MINUTES:-120}"
)

optional_env=(
  AWS_ACCESS_KEY_ID
  AWS_SECRET_ACCESS_KEY
  AWS_SESSION_TOKEN
  AWS_PROFILE
  AWS_DEFAULT_REGION
  AWS_CONTAINER_CREDENTIALS_FULL_URI
  AWS_CONTAINER_CREDENTIALS_RELATIVE_URI
  AWS_CONTAINER_AUTHORIZATION_TOKEN
  AWS_WEB_IDENTITY_TOKEN_FILE
  AWS_ROLE_ARN
  QDRANT_API_KEY
  QDRANT_API_KEY_SECRET_ARN
  HF_HUB_OFFLINE
  TRANSFORMERS_OFFLINE
)

for key in "${optional_env[@]}"; do
  if [[ -n "${!key:-}" ]]; then
    env_args+=(-e "$key=${!key}")
  fi
done

volume_args=(
  -v "${container_name}-input:/opt/ml/input"
  -v "${container_name}-model:/opt/ml/model"
  -v "${container_name}-model-cache:/opt/ml/model-cache"
  -v "${container_name}-output:/opt/ml/output"
  -v "${container_name}-repo-cache:/opt/ml/codecompass"
)

if [[ -d "$HOME/.aws" ]]; then
  volume_args+=(-v "$HOME/.aws:/home/app/.aws:ro")
fi

printf 'Starting local SageMaker parity container on port %s\n' "$port"
docker run \
  --detach \
  --name "$container_name" \
  --user 10001:10001 \
  --publish "$port:8080" \
  "${env_args[@]}" \
  "${volume_args[@]}" \
  "$image_name:$image_tag" \
  serve >/dev/null

started_at="$(date +%s)"
printf 'Waiting up to %ss for GET /ping to return HTTP 200; per-request timeout is %ss\n' \
  "$timeout_seconds" "$ping_request_timeout_seconds"

last_ping_code=""
last_ping_body=""
while true; do
  response_file="$(mktemp)"
  set +e
  ping_code="$(
    curl \
      --silent \
      --show-error \
      --output "$response_file" \
      --write-out '%{http_code}' \
      --request GET \
      --max-time "$ping_request_timeout_seconds" \
      "http://127.0.0.1:${port}/ping" 2>/dev/null
  )"
  curl_status="$?"
  set -e
  if [[ "$curl_status" -ne 0 ]]; then
    ping_code="000"
  fi
  ping_body="$(tr '\n' ' ' < "$response_file" | head -c 500)"
  rm -f "$response_file"

  elapsed="$(( $(date +%s) - started_at ))"
  printf '%s elapsed=%ss GET /ping status=%s body=%s\n' \
    "$(date -u +"%Y-%m-%dT%H:%M:%SZ")" "$elapsed" "$ping_code" "$ping_body"

  last_ping_code="$ping_code"
  last_ping_body="$ping_body"

  if [[ "$ping_code" == "200" ]]; then
    break
  fi

  if ! docker inspect -f '{{.State.Running}}' "$container_name" 2>/dev/null | grep -q true; then
    printf 'Container exited before GET /ping returned HTTP 200\n' >&2
    print_logs
    exit 1
  fi

  if (( elapsed > timeout_seconds )); then
    printf 'Timed out after %ss waiting for GET /ping HTTP 200; last status=%s body=%s\n' \
      "$timeout_seconds" "$last_ping_code" "$last_ping_body" >&2
    print_logs
    exit 1
  fi

  sleep "$ping_poll_interval_seconds"
done

curl --fail --silent --show-error "http://127.0.0.1:${port}/api/health" >/dev/null
curl --fail --silent --show-error \
  --header 'Content-Type: application/json' \
  --data '{"action":"list_repositories","session_id":"local-sagemaker-check","payload":{}}' \
  "http://127.0.0.1:${port}/invocations" >/dev/null

printf 'Local SageMaker compatibility checks passed for %s:%s\n' "$image_name" "$image_tag"
