import os
import random
import time
from typing import Optional, Tuple


class BedrockTransientError(RuntimeError):
    pass


def create_bedrock_runtime_client():
    try:
        import boto3
    except ImportError as exc:
        raise RuntimeError("Bedrock Claude support requires the `boto3` package.") from exc

    max_attempts = int(os.getenv("BEDROCK_CLIENT_MAX_ATTEMPTS", "10"))
    try:
        from botocore.config import Config

        config = Config(
            retries={
                "max_attempts": max_attempts,
                "mode": os.getenv("BEDROCK_RETRY_MODE", "adaptive"),
            }
        )
    except Exception:
        config = None

    return boto3.client(
        "bedrock-runtime",
        region_name=os.getenv("AWS_REGION", os.getenv("AWS_DEFAULT_REGION", "us-east-1")),
        config=config,
    )


def is_bedrock_retryable_error(exc: Exception) -> bool:
    code = ""
    response = getattr(exc, "response", None)
    if isinstance(response, dict):
        code = str((response.get("Error") or {}).get("Code") or "")

    message = str(exc)
    normalized = f"{code} {message}".lower()
    return any(
        marker in normalized
        for marker in {
            "throttling",
            "too many requests",
            "rate exceeded",
            "serviceunavailable",
            "service unavailable",
            "timeout",
            "timed out",
            "temporarily unavailable",
        }
    )


def generate_bedrock_claude_text(
    client,
    model: str,
    system_prompt: str,
    user_prompt: str,
    *,
    max_tokens: int,
    temperature: float,
    top_p: Optional[float] = None,
) -> Tuple[str, str]:
    inference_config = {
        "maxTokens": max_tokens,
        "temperature": temperature,
    }
    if top_p is not None:
        inference_config["topP"] = top_p

    max_retries = int(os.getenv("BEDROCK_LLM_MAX_RETRIES", "6"))
    base_seconds = float(os.getenv("BEDROCK_LLM_RETRY_BASE_SECONDS", "1.5"))
    max_sleep_seconds = float(os.getenv("BEDROCK_LLM_RETRY_MAX_SECONDS", "30"))
    last_exc = None

    for attempt in range(1, max_retries + 1):
        try:
            response = client.converse(
                modelId=model,
                system=[{"text": system_prompt.strip()}],
                messages=[
                    {
                        "role": "user",
                        "content": [{"text": user_prompt.strip()}],
                    }
                ],
                inferenceConfig=inference_config,
            )
            break
        except Exception as exc:
            if not is_bedrock_retryable_error(exc) or attempt >= max_retries:
                if is_bedrock_retryable_error(exc):
                    raise BedrockTransientError(
                        f"Bedrock throttled or was temporarily unavailable after "
                        f"{attempt} attempts: {exc}"
                    ) from exc
                raise

            last_exc = exc
            sleep_seconds = min(max_sleep_seconds, base_seconds * (2 ** (attempt - 1)))
            sleep_seconds += random.uniform(0.0, min(1.0, sleep_seconds * 0.2))
            print(
                f"[bedrock] Transient Bedrock error; retrying "
                f"attempt={attempt}/{max_retries} wait={sleep_seconds:.1f}s error={exc}",
                flush=True,
            )
            time.sleep(sleep_seconds)
    else:
        raise BedrockTransientError(f"Bedrock request failed after retries: {last_exc}")

    content_blocks = (((response or {}).get("output") or {}).get("message") or {}).get("content") or []
    text = "".join(block.get("text", "") for block in content_blocks if block.get("text")).strip()
    if not text:
        raise RuntimeError("Bedrock Claude returned an empty response.")

    return text, str((response or {}).get("stopReason", "") or "")
