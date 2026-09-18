#!/usr/bin/env bash
set -euo pipefail

: "${AWS_REGION:?Set AWS_REGION}"
: "${ECR_REPOSITORY:?Set ECR_REPOSITORY}"

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
image_name="${IMAGE_NAME:-code-compass-backend}"
image_tag="${IMAGE_TAG:-$(git -C "$repo_root" rev-parse --short=12 HEAD 2>/dev/null || printf 'local')}"
account_id="${AWS_ACCOUNT_ID:-$(aws sts get-caller-identity --query Account --output text)}"
registry="$account_id.dkr.ecr.$AWS_REGION.amazonaws.com"
remote_image="$registry/$ECR_REPOSITORY:$image_tag"

if ! aws ecr describe-repositories \
  --region "$AWS_REGION" \
  --repository-names "$ECR_REPOSITORY" >/dev/null 2>&1; then
  aws ecr create-repository \
    --region "$AWS_REGION" \
    --repository-name "$ECR_REPOSITORY" \
    --image-scanning-configuration scanOnPush=true >/dev/null
fi

aws ecr get-login-password --region "$AWS_REGION" \
  | docker login --username AWS --password-stdin "$registry"
docker tag "$image_name:$image_tag" "$remote_image"
docker push "$remote_image"

printf 'Pushed %s\n' "$remote_image"
