#!/bin/sh
set -eu

timestamp() {
  date -u +"%Y-%m-%dT%H:%M:%SZ"
}

printf '%s container_start argv=%s uid=%s gid=%s\n' \
  "$(timestamp)" "$*" "$(id -u)" "$(id -g)"
printf '%s entrypoint_start port=%s\n' "$(timestamp)" "${PORT:-8080}"

# SageMaker starts custom inference images with a positional "serve" argument.
if [ "${1:-}" = "serve" ]; then
  shift
fi

printf '%s uvicorn_exec host=0.0.0.0 port=%s workers=1\n' \
  "$(timestamp)" "${PORT:-8080}"

exec uvicorn server_app:app \
  --host 0.0.0.0 \
  --port "${PORT:-8080}" \
  --workers 1 \
  --timeout-graceful-shutdown "${GRACEFUL_SHUTDOWN_SECONDS:-30}" \
  --log-level "$(printf '%s' "${LOG_LEVEL:-info}" | tr '[:upper:]' '[:lower:]')" \
  "$@"
