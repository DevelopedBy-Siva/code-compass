import os
from typing import Optional, Tuple


def create_bedrock_runtime_client():
    try:
        import boto3
    except ImportError as exc:
        raise RuntimeError("Bedrock Claude support requires the `boto3` package.") from exc

    return boto3.client(
        "bedrock-runtime",
        region_name=os.getenv("AWS_REGION", os.getenv("AWS_DEFAULT_REGION", "us-east-1")),
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

    content_blocks = (((response or {}).get("output") or {}).get("message") or {}).get("content") or []
    text = "".join(block.get("text", "") for block in content_blocks if block.get("text")).strip()
    if not text:
        raise RuntimeError("Bedrock Claude returned an empty response.")

    return text, str((response or {}).get("stopReason", "") or "")
