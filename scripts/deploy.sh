#!/usr/bin/env bash
set -euo pipefail

: "${AWS_REGION:?Set AWS_REGION}"
: "${ECR_REPOSITORY:?Set ECR_REPOSITORY}"
: "${SAGEMAKER_ENDPOINT_NAME:?Set SAGEMAKER_ENDPOINT_NAME}"
: "${SAGEMAKER_EXECUTION_ROLE_ARN:?Set SAGEMAKER_EXECUTION_ROLE_ARN}"
: "${QDRANT_URL:?Set QDRANT_URL}"
: "${CORS_ORIGINS:?Set CORS_ORIGINS to the deployed Vercel origin}"

if [[ -z "${QDRANT_API_KEY:-}" && -z "${QDRANT_API_KEY_SECRET_ARN:-}" ]]; then
  printf 'Set QDRANT_API_KEY_SECRET_ARN (recommended) or QDRANT_API_KEY\n' >&2
  exit 1
fi
command -v jq >/dev/null || { printf 'jq is required\n' >&2; exit 1; }

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
image_tag="${IMAGE_TAG:-$(git -C "$repo_root" rev-parse --short=12 HEAD 2>/dev/null || printf 'local')}"
account_id="${AWS_ACCOUNT_ID:-$(aws sts get-caller-identity --query Account --output text)}"
image_uri="$account_id.dkr.ecr.$AWS_REGION.amazonaws.com/$ECR_REPOSITORY:$image_tag"
instance_type="${SAGEMAKER_INSTANCE_TYPE:-ml.g5.xlarge}"
instance_count="${SAGEMAKER_INSTANCE_COUNT:-1}"
startup_timeout="${SAGEMAKER_STARTUP_TIMEOUT_SECONDS:-900}"

container_env="$({
  jq -n \
    --arg app_env "${APP_ENV:-production}" \
    --arg enable_profiling "${ENABLE_PROFILING:-false}" \
    --arg log_level "${LOG_LEVEL:-INFO}" \
    --arg require_cuda "${REQUIRE_CUDA:-1}" \
    --arg aws_region "$AWS_REGION" \
    --arg bedrock_model "${BEDROCK_MODEL_ID:-qwen.qwen3-coder-next}" \
    --arg cors "${CORS_ORIGINS:-http://localhost:3000}" \
    --arg qdrant_url "$QDRANT_URL" \
    --arg qdrant_collection "${QDRANT_COLLECTION:-code_compass_qwen3_embedding_0_6b_last_token_cache_v2}" \
    --arg repo_cache_dir "${REPO_CACHE_DIR:-/opt/ml/codecompass/repos}" \
    --arg secret_arn "${QDRANT_API_KEY_SECRET_ARN:-}" \
    --arg api_key "${QDRANT_API_KEY:-}" \
    --arg session_ttl "${SESSION_TTL_MINUTES:-120}" \
    '{APP_ENV:$app_env,ENABLE_PROFILING:$enable_profiling,LOG_LEVEL:$log_level,REQUIRE_CUDA:$require_cuda,AWS_REGION:$aws_region,BEDROCK_MODEL_ID:$bedrock_model,CORS_ORIGINS:$cors,QDRANT_URL:$qdrant_url,QDRANT_COLLECTION:$qdrant_collection,REPO_CACHE_DIR:$repo_cache_dir,SESSION_TTL_MINUTES:$session_ttl}
     + (if $secret_arn != "" then {QDRANT_API_KEY_SECRET_ARN:$secret_arn} else {} end)
     + (if $api_key != "" then {QDRANT_API_KEY:$api_key} else {} end)'
})"

deployment_hash="$(printf '%s\n%s\n%s\n%s\n%s' \
  "$image_uri" "$container_env" "$instance_type" "$instance_count" "$startup_timeout" \
  | openssl dgst -sha256 | awk '{print substr($NF,1,10)}')"
name_prefix="$(printf '%s' "$SAGEMAKER_ENDPOINT_NAME" | tr -cs 'A-Za-z0-9-' '-' | sed 's/^-//;s/-$//')"
model_name="$(printf '%.52s-%s' "$name_prefix" "$deployment_hash")"
endpoint_config_name="$model_name"

if ! aws sagemaker describe-model --region "$AWS_REGION" --model-name "$model_name" >/dev/null 2>&1; then
  primary_container="$(jq -n --arg image "$image_uri" --argjson env "$container_env" '{Image:$image,Environment:$env}')"
  aws sagemaker create-model \
    --region "$AWS_REGION" \
    --model-name "$model_name" \
    --execution-role-arn "$SAGEMAKER_EXECUTION_ROLE_ARN" \
    --primary-container "$primary_container" >/dev/null
fi

if ! aws sagemaker describe-endpoint-config \
  --region "$AWS_REGION" \
  --endpoint-config-name "$endpoint_config_name" >/dev/null 2>&1; then
  production_variants="$(jq -n \
    --arg model "$model_name" \
    --arg instance_type "$instance_type" \
    --argjson instance_count "$instance_count" \
    --argjson startup_timeout "$startup_timeout" \
    '[{VariantName:"AllTraffic",ModelName:$model,InitialInstanceCount:$instance_count,InstanceType:$instance_type,InitialVariantWeight:1.0,ContainerStartupHealthCheckTimeoutInSeconds:$startup_timeout}]')"
  aws sagemaker create-endpoint-config \
    --region "$AWS_REGION" \
    --endpoint-config-name "$endpoint_config_name" \
    --production-variants "$production_variants" >/dev/null
fi

create_endpoint() {
  aws sagemaker create-endpoint \
    --region "$AWS_REGION" \
    --endpoint-name "$SAGEMAKER_ENDPOINT_NAME" \
    --endpoint-config-name "$endpoint_config_name" >/dev/null
}

if endpoint_description="$(aws sagemaker describe-endpoint \
  --region "$AWS_REGION" \
  --endpoint-name "$SAGEMAKER_ENDPOINT_NAME" 2>/dev/null)"; then
  endpoint_status="$(jq -r '.EndpointStatus' <<<"$endpoint_description")"

  if [[ "$endpoint_status" == "Failed" ]]; then
    printf 'Deleting failed SageMaker endpoint %s before recreating it\n' \
      "$SAGEMAKER_ENDPOINT_NAME"
    aws sagemaker delete-endpoint \
      --region "$AWS_REGION" \
      --endpoint-name "$SAGEMAKER_ENDPOINT_NAME"
    aws sagemaker wait endpoint-deleted \
      --region "$AWS_REGION" \
      --endpoint-name "$SAGEMAKER_ENDPOINT_NAME"
    create_endpoint
  elif [[ "$endpoint_status" == "Deleting" ]]; then
    printf 'Waiting for SageMaker endpoint %s to finish deleting before recreating it\n' \
      "$SAGEMAKER_ENDPOINT_NAME"
    aws sagemaker wait endpoint-deleted \
      --region "$AWS_REGION" \
      --endpoint-name "$SAGEMAKER_ENDPOINT_NAME"
    create_endpoint
  else
    if [[ "$endpoint_status" != "InService" ]]; then
      aws sagemaker wait endpoint-in-service \
        --region "$AWS_REGION" \
        --endpoint-name "$SAGEMAKER_ENDPOINT_NAME"
      endpoint_description="$(aws sagemaker describe-endpoint \
        --region "$AWS_REGION" \
        --endpoint-name "$SAGEMAKER_ENDPOINT_NAME")"
    fi

    current_config="$(jq -r '.EndpointConfigName' <<<"$endpoint_description")"
    if [[ "$current_config" != "$endpoint_config_name" ]]; then
      aws sagemaker update-endpoint \
        --region "$AWS_REGION" \
        --endpoint-name "$SAGEMAKER_ENDPOINT_NAME" \
        --endpoint-config-name "$endpoint_config_name" >/dev/null
    fi
  fi
else
  create_endpoint
fi

aws sagemaker wait endpoint-in-service \
  --region "$AWS_REGION" \
  --endpoint-name "$SAGEMAKER_ENDPOINT_NAME"

printf 'SageMaker endpoint %s is InService with %s\n' \
  "$SAGEMAKER_ENDPOINT_NAME" "$endpoint_config_name"
