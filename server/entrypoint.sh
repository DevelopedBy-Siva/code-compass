#!/bin/sh
set -eu

# SageMaker starts custom inference images with a positional "serve" argument.
if [ "${1:-}" = "serve" ]; then
  shift
fi

exec uvicorn server_app:app \
  --host 0.0.0.0 \
  --port "${PORT:-8080}" \
  --workers 1 \
  --timeout-graceful-shutdown "${GRACEFUL_SHUTDOWN_SECONDS:-30}" \
  --log-level "$(printf '%s' "${LOG_LEVEL:-info}" | tr '[:upper:]' '[:lower:]')" \
  "$@"
