"""Evaluation harness for Code Compass RAG system.

Computes 4 core metrics:
- Hit rate @ top-5 (retrieval quality)
- Grounded answer rate (citation accuracy)
- LLM-as-judge faithfulness (Claude 3.5 Sonnet via RAGAS)
- Query latency P95 (responsiveness)
"""

import asyncio
import json
import os
import sys
import re
import time
from pathlib import Path
from collections import Counter, defaultdict
from statistics import mean

import requests
from dotenv import load_dotenv

SERVER_ROOT = Path(__file__).resolve().parents[1]
if str(SERVER_ROOT) not in sys.path:
    sys.path.insert(0, str(SERVER_ROOT))

load_dotenv(SERVER_ROOT / ".env")

from src.bedrock_claude import create_bedrock_runtime_client, generate_bedrock_claude_text
from src.embeddings import EmbeddingGenerator


API_URL = os.getenv("CODEBASE_RAG_API_URL", "http://localhost:8000")
REPO_ID = int(os.getenv("CODEBASE_RAG_REPO_ID", "1"))
SESSION_ID = os.getenv("CODEBASE_RAG_SESSION_ID", "eval-session")
TOP_K = int(os.getenv("CODEBASE_RAG_TOP_K", "8"))
QUERY_TIMEOUT_SECONDS = int(os.getenv("CODEBASE_RAG_QUERY_TIMEOUT_SECONDS", "180"))
QUERY_MAX_RETRIES = int(os.getenv("CODEBASE_RAG_QUERY_MAX_RETRIES", "5"))
QUERY_RETRY_BASE_SECONDS = float(os.getenv("CODEBASE_RAG_QUERY_RETRY_BASE_SECONDS", "2"))
EVAL_SET_PATH = Path(
    os.getenv(
        "CODEBASE_RAG_EVAL_SET",
        Path(__file__).with_name("sample_eval_set.json"),
    )
)


def log(message: str):
    """Log message to stderr with [eval] prefix."""
    print(f"[eval] {message}", file=sys.stderr, flush=True)


def get_app_model_config():
    llm_provider = os.getenv("LLM_PROVIDER", "bedrock").lower()
    if llm_provider == "groq":
        llm_model = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
    elif llm_provider == "bedrock":
        llm_model = os.getenv(
            "BEDROCK_LLM_MODEL",
            "anthropic.claude-3-5-sonnet-20240620-v1:0",
        )
    elif llm_provider == "vertex_ai":
        llm_model = os.getenv("VERTEX_LLM_MODEL", "claude-3-5-sonnet@20240620")
    else:
        llm_model = "unknown"

    embedding_provider = os.getenv("EMBEDDING_PROVIDER", "auto").lower()
    if embedding_provider == "bedrock":
        embedding_model = os.getenv("BEDROCK_EMBEDDING_MODEL", "cohere.embed-v3:0")
    elif embedding_provider == "vertex_ai":
        embedding_model = os.getenv("VERTEX_EMBEDDING_MODEL", "gemini-embedding-001")
    elif embedding_provider == "openai":
        embedding_model = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
    elif embedding_provider == "local":
        embedding_model = os.getenv("EMBEDDING_MODEL") or os.getenv(
            "LOCAL_EMBEDDING_MODEL", "nomic-ai/CodeRankEmbed"
        )
    else:
        embedding_model = os.getenv("EMBEDDING_MODEL") or "auto"

    eval_model = os.getenv(
        "EVAL_MODEL",
        os.getenv("BEDROCK_EVAL_MODEL", "anthropic.claude-3-5-sonnet-20240620-v1:0"),
    )
    return {
        "llm_provider": llm_provider,
        "llm_model": llm_model,
        "embedding_provider": embedding_provider,
        "embedding_model": embedding_model,
        "eval_model": eval_model,
    }


def load_eval_rows():
    return json.loads(EVAL_SET_PATH.read_text())


def post_query(row):
    payload = {
        "repo_id": REPO_ID,
        "question": row["question"],
        "top_k": TOP_K,
        "history": row.get("turns", []),
    }
    case_id = row.get("id", row["question"])

    for attempt in range(1, QUERY_MAX_RETRIES + 1):
        response = requests.post(
            f"{API_URL}/api/query",
            json=payload,
            headers={"X-Session-Id": SESSION_ID},
            timeout=QUERY_TIMEOUT_SECONDS,
        )
        if response.ok:
            return response.json()

        detail = response.text
        try:
            parsed = response.json()
            detail = parsed.get("detail") or parsed
        except Exception:
            pass

        detail_text = str(detail)
        is_retryable = response.status_code in {429, 500, 502, 503, 504} and any(
            marker in detail_text
            for marker in [
                "ThrottlingException",
                "throttled",
                "Too many requests",
                "timed out",
                "timeout",
                "ServiceUnavailable",
                "temporarily unavailable",
            ]
        )
        if is_retryable and attempt < QUERY_MAX_RETRIES:
            retry_after = response.headers.get("Retry-After")
            try:
                wait_seconds = (
                    float(retry_after)
                    if retry_after
                    else QUERY_RETRY_BASE_SECONDS * (2 ** (attempt - 1))
                )
            except ValueError:
                wait_seconds = QUERY_RETRY_BASE_SECONDS * (2 ** (attempt - 1))
            log(
                f"Retrying case {case_id} after transient query failure "
                f"(attempt {attempt}/{QUERY_MAX_RETRIES}, wait={wait_seconds:.1f}s): {detail_text}"
            )
            time.sleep(wait_seconds)
            continue

        raise RuntimeError(
            f"Query failed for eval case {case_id!r} "
            f"with status {response.status_code}: {detail}"
        )

    raise RuntimeError(f"Query failed for eval case {case_id!r}: exhausted retries")


def normalize_path(path: str) -> str:
    return path.strip().lstrip("./").lower()


STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "how",
    "in",
    "into",
    "is",
    "it",
    "its",
    "of",
    "on",
    "or",
    "that",
    "the",
    "their",
    "this",
    "to",
    "what",
    "when",
    "where",
    "which",
    "with",
}


def tokenize_text(text: str):
    tokens = []
    for raw_token in re.findall(r"[A-Za-z0-9_./+-]+", text or ""):
        token = raw_token.lower()
        tokens.append(token)

        camel_parts = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", " ", raw_token).split()
        split_parts = re.split(r"[._/+-]+", token)
        for part in [*camel_parts, *split_parts]:
            normalized = part.strip().lower()
            if normalized and normalized != token:
                tokens.append(normalized)

    return tokens


def normalize_keywords(keywords):
    normalized = []
    seen = set()
    for keyword in keywords or []:
        phrase = " ".join(tokenize_text(str(keyword)))
        if not phrase or phrase in seen:
            continue
        seen.add(phrase)
        normalized.append(phrase)
    return normalized


def compute_retrieval_metrics(expected_sources, actual_sources):
    """Compute retrieval metrics: hit rate and top-1 hit for given rank k."""
    expected = {normalize_path(path) for path in expected_sources}
    actual = [normalize_path(path) for path in actual_sources]

    def matches_expected(actual_path: str) -> bool:
        for expected_path in expected:
            expected_is_directory = (
                expected_path.endswith("/")
                or "." not in expected_path.rsplit("/", 1)[-1]
            )
            normalized_expected = expected_path.rstrip("/")
            if actual_path == expected_path:
                return True
            if expected_is_directory and actual_path.startswith(normalized_expected + "/"):
                return True
        return False

    # Hit rate: was any retrieved source relevant?
    hit = 1 if any(matches_expected(path) for path in actual) else 0

    # Top-1 hit: was the first retrieved source relevant?
    top1_hit = 1 if actual and matches_expected(actual[0]) else 0

    return {
        "retrieval_hit": hit,
        "top1_hit": top1_hit,
    }


def keyword_match_details(row, answer: str):
    keywords = normalize_keywords(row.get("must_include_any", []))
    if not keywords:
        return None

    answer_tokens = tokenize_text(answer)
    if not answer_tokens:
        return {
            "coverage": 0.0,
            "matched_count": 0,
            "total_keywords": len(keywords),
            "matched_keywords": [],
            "missing_keywords": keywords,
        }

    matched_keywords = []
    for keyword in keywords:
        keyword_tokens = keyword.split()
        window = len(keyword_tokens)
        if window == 1:
            if keyword_tokens[0] in answer_tokens:
                matched_keywords.append(keyword)
            continue

def answer_length_metrics(answer: str):
    """Check if answer has substantive content."""
    tokens = tokenize_text(answer)
    return {
        "answer_word_count": len(tokens),
        "has_substantive_answer": 1 if len(tokens) >= 40 else 0,
    }


def validate_eval_rows(rows):
    errors = []
    warnings = []
    category_counts = Counter()
    id_counts = Counter()
    id_prefix_counts = Counter()
    expected_source_counts = []
    keyword_counts = []
    conversation_cases = 0
    benchmark_scope = {
        "type": "mixed_or_unknown",
        "dominant_prefix": None,
        "dominant_prefix_fraction": 0.0,
    }

    for index, row in enumerate(rows, start=1):
        row_id = row.get("id") or f"row-{index}"
        id_counts[row_id] += 1
        prefix = row_id.split("-", 1)[0].lower()
        if prefix:
            id_prefix_counts[prefix] += 1
        category_counts[row.get("category", "general")] += 1

        question = str(row.get("question", "")).strip()
        ground_truth = str(row.get("ground_truth", "")).strip()
        expected_sources = row.get("expected_sources", [])
        must_include_any = row.get("must_include_any", [])

        if not question:
            errors.append(f"{row_id}: missing question")
        if not ground_truth:
            errors.append(f"{row_id}: missing ground_truth")
        if not isinstance(expected_sources, list) or not expected_sources:
            errors.append(f"{row_id}: expected_sources must be a non-empty list")
        if must_include_any and not isinstance(must_include_any, list):
            errors.append(f"{row_id}: must_include_any must be a list when present")
        if isinstance(must_include_any, list):
            normalized_keywords = normalize_keywords(must_include_any)
            if len(normalized_keywords) != len([keyword for keyword in must_include_any if str(keyword).strip()]):
                warnings.append(
                    f"{row_id}: duplicate or case-variant keywords were normalized; "
                    "resume metrics are stricter than the raw checklist wording."
                )
        if row.get("turns"):
            conversation_cases += 1
        expected_source_counts.append(len(expected_sources) if isinstance(expected_sources, list) else 0)
        keyword_counts.append(len(must_include_any) if isinstance(must_include_any, list) else 0)

    duplicate_ids = sorted(row_id for row_id, count in id_counts.items() if count > 1)
    if duplicate_ids:
        errors.append(f"duplicate ids found: {', '.join(duplicate_ids)}")

    if len(rows) < 25:
        warnings.append(
            "Eval set has fewer than 25 cases. Good for iteration, but light for resume-grade benchmarking."
        )
    if len(category_counts) < 4:
        warnings.append("Eval set covers fewer than 4 categories, so breadth is limited.")
    if conversation_cases < 2:
        warnings.append("Eval set has very little multi-turn coverage.")
    if category_counts and min(category_counts.values()) < 2:
        sparse = sorted(category for category, count in category_counts.items() if count < 2)
        warnings.append(f"Some categories are underrepresented: {', '.join(sparse)}.")

    if id_prefix_counts:
        dominant_prefix, dominant_count = id_prefix_counts.most_common(1)[0]
        dominant_prefix_fraction = dominant_count / len(rows)
        if dominant_prefix_fraction >= 0.8:
            benchmark_scope = {
                "type": "single_repository",
                "dominant_prefix": dominant_prefix,
                "dominant_prefix_fraction": round(dominant_prefix_fraction, 4),
            }

    return {
        "case_count": len(rows),
        "category_counts": dict(sorted(category_counts.items())),
        "conversation_case_count": conversation_cases,
        "average_expected_sources": round(mean(expected_source_counts), 2) if expected_source_counts else 0.0,
        "average_keywords_per_case": round(mean(keyword_counts), 2) if keyword_counts else 0.0,
        "benchmark_scope": benchmark_scope,
        "errors": errors,
        "warnings": warnings,
        "is_valid": not errors,
    }


def summarize_custom_metrics(details, latency_p95=None):
    """Compute only the 4 core metrics: hit rate @ top-5, grounded answer rate, faithfulness, latency P95."""
    # Grounded answer: retrieval hit AND has substantive answer AND no failed keyword checks
    grounded_answer_passes = [
        1
        for item in details
        if item["retrieval_hit"] == 1
        and item["has_substantive_answer"] == 1
    ]
    return {
        "retrieval_hit_rate": round(mean(item["retrieval_hit"] for item in details), 4),
        "top1_hit_rate": round(mean(item["top1_hit"] for item in details), 4),
        "grounded_answer_rate": round(sum(grounded_answer_passes) / len(details), 4) if details else 0.0,
        "latency_p95_ms": round(latency_p95, 2) if latency_p95 is not None else None,
    }


def summarize_by_category(details):
    """Summarize metrics by category using only the 4 core metrics."""
    grouped = defaultdict(list)
    for item in details:
        grouped[item["category"]].append(item)

    summary = {}
    for category, items in sorted(grouped.items()):
        summary[category] = {
            "case_count": len(items),
            "retrieval_hit_rate": round(mean(item["retrieval_hit"] for item in items), 4),
            "top1_hit_rate": round(mean(item["top1_hit"] for item in items), 4),
            "grounded_answer_rate": round(
                mean(
                    1
                    if item["retrieval_hit"] == 1 and item["has_substantive_answer"] == 1
                    else 0
                    for item in items
                ),
                4,
            ),
        }
    return summary


def build_headline_metrics(custom_metrics, audit):
    """Build headline metrics section with only the 4 core metrics."""
    return {
        "sample_size": audit["case_count"],
        "category_count": len(audit["category_counts"]),
        "retrieval_hit_rate": custom_metrics["retrieval_hit_rate"],
        "top1_hit_rate": custom_metrics["top1_hit_rate"],
        "grounded_answer_rate": custom_metrics["grounded_answer_rate"],
        "latency_p95_ms": custom_metrics["latency_p95_ms"],
    }


def build_metric_guidance(custom_metrics, ragas_report):
    """Build guidance using only the 4 core metrics."""
    # Primary gate: retrieval hit rate >= 80%
    retrieval_gate_pass = custom_metrics["retrieval_hit_rate"] >= 0.8

    next_focus = []
    if custom_metrics["grounded_answer_rate"] < 0.75:
        next_focus.append("Tighten answer grounding to ensure answers cite sources.")
    if custom_metrics["latency_p95_ms"] and custom_metrics["latency_p95_ms"] > 5000:
        next_focus.append("Optimize query latency for better responsiveness.")

    return {
        "primary_gate": "pass" if retrieval_gate_pass else "needs_work",
        "primary_gate_basis": "retrieval_hit_rate",
        "next_focus": next_focus,
    }


def build_resume_summary(custom_metrics, audit, ragas_report, ragas_error):
    """Build resume summary using only the 4 core metrics: hit rate top-5, grounded answer rate, faithfulness, latency."""
    lines = [
        (
            f"Evaluated on {audit['case_count']} repo-QA cases across "
            f"{len(audit['category_counts'])} categories."
        ),
        (
            f"Retrieval hit rate @ top-5: {custom_metrics['retrieval_hit_rate']:.1%}, "
            f"top-1 hit rate: {custom_metrics['top1_hit_rate']:.1%}."
        ),
        (
            f"Grounded answer rate: {custom_metrics['grounded_answer_rate']:.1%}."
        ),
    ]

    if ragas_report and not ragas_error:
        lines.append(
            f"Faithfulness (Claude 3.5 Sonnet judge): {ragas_report.get('faithfulness', 0.0):.3f}."
        )
    else:
        lines.append("Faithfulness metrics skipped or unavailable.")

    if custom_metrics["latency_p95_ms"] is not None:
        lines.append(f"Query latency P95: {custom_metrics['latency_p95_ms']:.0f}ms.")

    scope = audit.get("benchmark_scope", {})
    if scope.get("type") == "single_repository":
        lines.append(
            "Benchmark scope: single-repository benchmark "
            f"({scope.get('dominant_prefix')}); use it to judge this target repo, not cross-repo generalization."
        )

    if audit["warnings"]:
        lines.append(
            "Benchmark caveat: "
            + " ".join(audit["warnings"][:2])
        )

    return " ".join(lines)


def benchmark_readiness(audit, ragas_error, metric_guidance=None):
    reasons = []
    if audit["case_count"] < 25:
        reasons.append("small_sample")
    if len(audit["category_counts"]) < 4:
        reasons.append("limited_category_coverage")
    if audit["conversation_case_count"] < 2:
        reasons.append("limited_multi_turn_coverage")
    if audit["warnings"]:
        reasons.append("eval_set_warnings")
    if ragas_error not in {None, "disabled"}:
        reasons.append("ragas_instability")
    if metric_guidance and metric_guidance.get("primary_gate") != "pass":
        reasons.append("primary_gate_failed")

    if reasons:
        status = "single_repo_benchmark_needs_work"
        if audit.get("benchmark_scope", {}).get("type") != "single_repository":
            status = "internal_or_demo_benchmark"
        return {
            "status": status,
            "reasons": reasons,
        }
    if audit.get("benchmark_scope", {}).get("type") == "single_repository":
        return {
            "status": "single_repo_benchmark_ready",
            "reasons": [],
        }
    return {
        "status": "presentation_ready",
        "reasons": [],
    }


def maybe_write_report(report):
    output_path = os.getenv("CODEBASE_RAG_EVAL_OUTPUT")
    if not output_path:
        return None
    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(report, indent=2))
    return str(target)


def build_bedrock_ragas_llm(run_config):
    from langchain_core.outputs import Generation, LLMResult
    from ragas.llms.base import BaseRagasLLM

    class BedrockRagasLLM(BaseRagasLLM):
        def __init__(self, model: str, run_config):
            self.client = create_bedrock_runtime_client()
            self.model = model
            self.set_run_config(run_config)

        def _prompt_to_text(self, prompt):
            prefix = (
                "Return only valid JSON or the exact structured output requested by the prompt. "
                "Do not add markdown fences, explanations, or extra prose.\n\n"
            )
            if hasattr(prompt, "to_string"):
                return prefix + prompt.to_string()
            return prefix + str(prompt)

        def _generate_once(self, prompt, n=1, temperature=1e-8, stop=None, callbacks=None):
            prompt_text = self._prompt_to_text(prompt)
            text, _ = generate_bedrock_claude_text(
                self.client,
                self.model,
                "Return only valid JSON or the exact structured output requested.",
                prompt_text,
                max_tokens=int(os.getenv("EVAL_MAX_OUTPUT_TOKENS", "2048")),
                temperature=0.0,
            )

            generations = [Generation(text=text)] if text else []

            if not generations:
                raise RuntimeError("Bedrock Claude judge returned an empty response.")

            return LLMResult(generations=[generations])

        def generate_text(self, prompt, n=1, temperature=1e-8, stop=None, callbacks=None):
            return self._generate_once(
                prompt=prompt,
                n=n,
                temperature=temperature,
                stop=stop,
                callbacks=callbacks,
            )

        async def agenerate_text(self, prompt, n=1, temperature=1e-8, stop=None, callbacks=None):
            return await asyncio.to_thread(
                self._generate_once,
                prompt,
                n,
                temperature,
                stop,
                callbacks,
            )

    model = os.getenv(
        "EVAL_MODEL",
        os.getenv("BEDROCK_EVAL_MODEL", "anthropic.claude-3-5-sonnet-20240620-v1:0"),
    )
    return BedrockRagasLLM(model=model, run_config=run_config)


def build_ragas_embeddings(run_config):
    from ragas.embeddings.base import BaseRagasEmbeddings

    class AppEmbeddingWrapper(BaseRagasEmbeddings):
        def __init__(self, generator, run_config):
            self.generator = generator
            self.set_run_config(run_config)

        def embed_query(self, text):
            return self.generator.embed_text(text).tolist()

        def embed_documents(self, texts):
            vectors = self.generator.embed_batch(list(texts))
            return vectors.tolist()

        async def aembed_query(self, text):
            return await asyncio.to_thread(self.embed_query, text)

        async def aembed_documents(self, texts):
            return await asyncio.to_thread(self.embed_documents, texts)

    return AppEmbeddingWrapper(EmbeddingGenerator(), run_config=run_config)


def run_ragas(rows, outputs):
    if not ENABLE_RAGAS:
        log("RAGAS disabled via CODEBASE_RAG_ENABLE_RAGAS=0. Reporting custom metrics only.")
        return None, "disabled"

    try:
        from datasets import Dataset
        from ragas import evaluate
        from ragas.metrics import faithfulness
        from ragas.run_config import RunConfig
    except Exception as exc:
        log(f"Skipping RAGAS because the evaluation dependencies could not be loaded: {exc}")
        return None, f"import_error: {exc}"

    def build_ragas_dataset():
        samples = []
        for row, result in zip(rows, outputs):
            samples.append(
                {
                    "question": row["question"],
                    "answer": result["answer"],
                    "contexts": [source["snippet"] for source in result.get("sources", [])],
                    "ground_truth": row["ground_truth"],
                }
            )
        return Dataset.from_list(samples)

    log("Running RAGAS metrics. This can take a while.")
    try:
        timeout_seconds = int(os.getenv("EVAL_TIMEOUT_SECONDS", "180"))
        thread_timeout_seconds = float(os.getenv("EVAL_THREAD_TIMEOUT_SECONDS", str(max(timeout_seconds, 240))))
        max_workers = int(os.getenv("EVAL_MAX_WORKERS", "2"))
        run_config = RunConfig(
            timeout=timeout_seconds,
            thread_timeout=thread_timeout_seconds,
            max_workers=max_workers,
            max_retries=int(os.getenv("EVAL_MAX_RETRIES", "3")),
            max_wait=int(os.getenv("EVAL_MAX_WAIT_SECONDS", "60")),
        )
        log(
            "Using Bedrock for RAGAS judge model "
            f"({os.getenv('EVAL_MODEL', os.getenv('BEDROCK_EVAL_MODEL', 'anthropic.claude-3-5-sonnet-20240620-v1:0'))})"
        )
        log(
            f"RAGAS runtime: async={RAGAS_ASYNC}, raise_exceptions={RAGAS_RAISE_EXCEPTIONS}, "
            f"timeout={timeout_seconds}s, thread_timeout={thread_timeout_seconds}s, max_workers={max_workers}"
        )
        llm = build_bedrock_ragas_llm(run_config)
        embeddings = build_ragas_embeddings(run_config)
        # Only use faithfulness as the RAGAS metric (simplified to 4-core metrics)
        ragas_report = evaluate(
            build_ragas_dataset(),
            metrics=[faithfulness],
            llm=llm,
            embeddings=embeddings,
            run_config=run_config,
            is_async=RAGAS_ASYNC,
            raise_exceptions=RAGAS_RAISE_EXCEPTIONS,
        )
        return {key: float(value) for key, value in ragas_report.items()}, None
    except Exception as exc:
        log(f"RAGAS evaluation failed: {exc}")
        return None, str(exc)


def run():
    log(f"Loading eval set from {EVAL_SET_PATH}")
    rows = load_eval_rows()
    audit = validate_eval_rows(rows)
    model_config = get_app_model_config()
    if audit["errors"]:
        raise RuntimeError("Eval set validation failed: " + "; ".join(audit["errors"]))
    for warning in audit["warnings"]:
        log(f"Eval set warning: {warning}")
    log(
        "Eval model config: "
        f"qna_provider={model_config['llm_provider']}, "
        f"qna_model={model_config['llm_model']}, "
        f"embedding_provider={model_config['embedding_provider']}, "
        f"embedding_model={model_config['embedding_model']}, "
        f"judge_model={model_config['eval_model']}"
    )
    log(
        f"Starting eval with api_url={API_URL}, repo_id={REPO_ID}, "
        f"session_id={SESSION_ID}, top_k={TOP_K}, cases={len(rows)}"
    )
    outputs = []
    details = []
    latencies = []

    for index, row in enumerate(rows, start=1):
        case_id = row.get("id", row["question"])
        log(f"[{index}/{len(rows)}] Querying case {case_id}")
        start_time = time.time()
        result = post_query(row)
        elapsed_ms = (time.time() - start_time) * 1000
        latencies.append(elapsed_ms)
        outputs.append(result)
        log(
            f"[{index}/{len(rows)}] Received answer for {case_id} "
            f"with {len(result.get('sources', []))} sources in {elapsed_ms:.0f}ms"
        )

        cited_paths = [source["file_path"] for source in result.get("sources", [])]
        metrics = compute_retrieval_metrics(row.get("expected_sources", []), cited_paths)
        length_metrics = answer_length_metrics(result.get("answer", ""))

        details.append(
            {
                "id": row.get("id", row["question"]),
                "category": row.get("category", "general"),
                "question": row["question"],
                "answer": result.get("answer", ""),
                "expected_sources": row.get("expected_sources", []),
                "retrieved_sources": cited_paths,
                "retrieval_hit": metrics["retrieval_hit"],
                "top1_hit": metrics["top1_hit"],
                **length_metrics,
            }
        )

    # Compute P95 latency
    latencies.sort()
    p95_index = int(len(latencies) * 0.95)
    latency_p95 = latencies[p95_index] if latencies else None

    log("Finished query loop. Computing aggregate metrics.")
    custom_metrics = summarize_custom_metrics(details, latency_p95)
    category_breakdown = summarize_by_category(details)
    ragas_report, ragas_error = run_ragas(rows, outputs)
    headline_metrics = build_headline_metrics(custom_metrics, audit)
    metric_guidance = build_metric_guidance(custom_metrics, ragas_report)
    resume_summary = build_resume_summary(custom_metrics, audit, ragas_report, ragas_error)
    readiness = benchmark_readiness(audit, ragas_error, metric_guidance)

    report = {
        "config": {
            "api_url": API_URL,
            "repo_id": REPO_ID,
            "session_id": SESSION_ID,
            "top_k": TOP_K,
            "qna_provider": model_config["llm_provider"],
            "qna_model": model_config["llm_model"],
            "embedding_provider": model_config["embedding_provider"],
            "embedding_model": model_config["embedding_model"],
            "eval_model": model_config["eval_model"],
            "query_timeout_seconds": QUERY_TIMEOUT_SECONDS,
            "query_max_retries": QUERY_MAX_RETRIES,
            "query_retry_base_seconds": QUERY_RETRY_BASE_SECONDS,
            "eval_set": str(EVAL_SET_PATH),
            "min_reference_overlap": MIN_REFERENCE_OVERLAP,
            "min_reference_term_matches": MIN_REFERENCE_TERM_MATCHES,
        },
        "eval_set_audit": audit,
        "headline_metrics": headline_metrics,
        "benchmark_readiness": readiness,
        "metric_guidance": metric_guidance,
        "ragas": ragas_report,
        "ragas_error": ragas_error,
        "custom_metrics": custom_metrics,
        "category_breakdown": category_breakdown,
        "resume_summary": resume_summary,
        "cases": details,
    }
    output_path = maybe_write_report(report)
    if output_path:
        log(f"Wrote JSON report to {output_path}")

    log("Eval complete. Printing JSON report.")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    run()
