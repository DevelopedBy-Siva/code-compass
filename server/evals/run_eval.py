import json
import os
import sys
import asyncio
import re
from pathlib import Path
from collections import Counter, defaultdict
from statistics import mean

import requests
from dotenv import load_dotenv

SERVER_ROOT = Path(__file__).resolve().parents[1]
if str(SERVER_ROOT) not in sys.path:
    sys.path.insert(0, str(SERVER_ROOT))

load_dotenv(SERVER_ROOT / ".env")

from src.embeddings import EmbeddingGenerator


API_URL = os.getenv("CODEBASE_RAG_API_URL", "http://localhost:8000")
REPO_ID = int(os.getenv("CODEBASE_RAG_REPO_ID", "1"))
SESSION_ID = os.getenv("CODEBASE_RAG_SESSION_ID", "eval-session")
TOP_K = int(os.getenv("CODEBASE_RAG_TOP_K", "8"))
QUERY_TIMEOUT_SECONDS = int(os.getenv("CODEBASE_RAG_QUERY_TIMEOUT_SECONDS", "180"))
ENABLE_RAGAS = os.getenv("CODEBASE_RAG_ENABLE_RAGAS", "1").lower() not in {"0", "false", "no"}
RAGAS_ASYNC = os.getenv("CODEBASE_RAG_RAGAS_ASYNC", "0").lower() in {"1", "true", "yes"}
RAGAS_RAISE_EXCEPTIONS = os.getenv("CODEBASE_RAG_RAGAS_RAISE_EXCEPTIONS", "0").lower() in {
    "1",
    "true",
    "yes",
}
EVAL_SET_PATH = Path(
    os.getenv(
        "CODEBASE_RAG_EVAL_SET",
        Path(__file__).with_name("sample_eval_set.json"),
    )
)


def log(message: str):
    print(f"[eval] {message}", file=sys.stderr, flush=True)


def load_eval_rows():
    return json.loads(EVAL_SET_PATH.read_text())


def post_query(row):
    payload = {
        "repo_id": REPO_ID,
        "question": row["question"],
        "top_k": TOP_K,
        "history": row.get("turns", []),
    }
    response = requests.post(
        f"{API_URL}/api/query",
        json=payload,
        headers={"X-Session-Id": SESSION_ID},
        timeout=QUERY_TIMEOUT_SECONDS,
    )
    if not response.ok:
        detail = response.text
        try:
            parsed = response.json()
            detail = parsed.get("detail") or parsed
        except Exception:
            pass
        raise RuntimeError(
            f"Query failed for eval case {row.get('id', row['question'])!r} "
            f"with status {response.status_code}: {detail}"
        )
    return response.json()


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
    return re.findall(r"[a-z0-9_./+-]+", (text or "").lower())


def compute_retrieval_metrics(expected_sources, actual_sources):
    expected = {normalize_path(path) for path in expected_sources}
    actual = [normalize_path(path) for path in actual_sources]
    unique_actual = list(dict.fromkeys(actual))

    def matches_expected(actual_path: str) -> bool:
        for expected_path in expected:
            if actual_path == expected_path:
                return True
            if "/" not in expected_path and actual_path.startswith(expected_path.rstrip("/") + "/"):
                return True
        return False

    hit = 1 if any(matches_expected(path) for path in actual) else 0
    recall = 0.0
    if expected:
        matched_expected = set()
        for expected_path in expected:
            for actual_path in actual:
                if actual_path == expected_path or (
                    "/" not in expected_path and actual_path.startswith(expected_path.rstrip("/") + "/")
                ):
                    matched_expected.add(expected_path)
                    break
        recall = len(matched_expected) / len(expected)

    mrr = 0.0
    for index, path in enumerate(actual, start=1):
        if matches_expected(path):
            mrr = 1.0 / index
            break

    return {
        "retrieval_hit": hit,
        "source_recall": recall,
        "mrr": mrr,
        "top1_hit": 1 if actual and matches_expected(actual[0]) else 0,
        "unique_source_precision": (
            sum(1 for path in unique_actual if matches_expected(path)) / len(unique_actual)
            if unique_actual
            else 0.0
        ),
        "duplicate_source_rate": (
            (len(actual) - len(unique_actual)) / len(actual)
            if actual
            else 0.0
        ),
    }


def keyword_match_ratio(row, answer: str):
    keywords = [keyword.lower() for keyword in row.get("must_include_any", []) if keyword.strip()]
    if not keywords:
        return None
    lowered = answer.lower()
    matched = sum(1 for keyword in keywords if keyword in lowered)
    return matched / len(keywords)


def keyword_pass(row, answer: str, coverage: float | None):
    if coverage is None:
        return None
    minimum = int(row.get("min_keyword_matches", 1))
    keywords = [keyword for keyword in row.get("must_include_any", []) if str(keyword).strip()]
    if not keywords:
        return None
    matched = round(coverage * len(keywords))
    return 1 if matched >= minimum else 0


def answer_length_metrics(answer: str):
    tokens = tokenize_text(answer)
    return {
        "answer_word_count": len(tokens),
        "has_substantive_answer": 1 if len(tokens) >= 40 else 0,
    }


def lexical_overlap_ratio(reference: str, candidate: str):
    reference_terms = {
        token for token in tokenize_text(reference)
        if len(token) > 2 and token not in STOPWORDS
    }
    if not reference_terms:
        return None
    candidate_terms = set(tokenize_text(candidate))
    matched = sum(1 for token in reference_terms if token in candidate_terms)
    return matched / len(reference_terms)


def validate_eval_rows(rows):
    errors = []
    warnings = []
    category_counts = Counter()
    id_counts = Counter()
    id_prefix_counts = Counter()
    expected_source_counts = []
    keyword_counts = []
    conversation_cases = 0

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
        if dominant_count / len(rows) >= 0.8:
            warnings.append(
                f"Most cases share the same id prefix ({dominant_prefix}), which suggests a benchmark focused on one target project."
            )

    return {
        "case_count": len(rows),
        "category_counts": dict(sorted(category_counts.items())),
        "conversation_case_count": conversation_cases,
        "average_expected_sources": round(mean(expected_source_counts), 2) if expected_source_counts else 0.0,
        "average_keywords_per_case": round(mean(keyword_counts), 2) if keyword_counts else 0.0,
        "errors": errors,
        "warnings": warnings,
        "is_valid": not errors,
    }


def summarize_custom_metrics(details):
    keyword_coverages = [item["keyword_coverage"] for item in details if item["keyword_coverage"] is not None]
    keyword_passes = [item["keyword_pass"] for item in details if item["keyword_pass"] is not None]
    grounded_answer_passes = [
        1
        for item in details
        if item["retrieval_hit"] == 1
        and item["has_substantive_answer"] == 1
        and (item["keyword_pass"] in {None, 1})
    ]
    exact_source_recall_cases = [1 for item in details if item["source_recall"] == 1.0]
    return {
        "retrieval_hit_rate": round(mean(item["retrieval_hit"] for item in details), 4),
        "top1_hit_rate": round(mean(item["top1_hit"] for item in details), 4),
        "source_recall": round(mean(item["source_recall"] for item in details), 4),
        "mrr": round(mean(item["mrr"] for item in details), 4),
        "unique_source_precision": round(mean(item["unique_source_precision"] for item in details), 4),
        "duplicate_source_rate": round(mean(item["duplicate_source_rate"] for item in details), 4),
        "keyword_coverage": round(mean(keyword_coverages), 4) if keyword_coverages else None,
        "keyword_pass_rate": round(mean(keyword_passes), 4) if keyword_passes else None,
        "ground_truth_lexical_overlap": round(
            mean(item["ground_truth_lexical_overlap"] for item in details if item["ground_truth_lexical_overlap"] is not None),
            4,
        )
        if any(item["ground_truth_lexical_overlap"] is not None for item in details)
        else None,
        "substantive_answer_rate": round(mean(item["has_substantive_answer"] for item in details), 4),
        "grounded_answer_rate": round(sum(grounded_answer_passes) / len(details), 4) if details else 0.0,
        "exact_source_recall_rate": round(sum(exact_source_recall_cases) / len(details), 4) if details else 0.0,
    }


def summarize_by_category(details):
    grouped = defaultdict(list)
    for item in details:
        grouped[item["category"]].append(item)

    summary = {}
    for category, items in sorted(grouped.items()):
        keyword_passes = [item["keyword_pass"] for item in items if item["keyword_pass"] is not None]
        summary[category] = {
            "case_count": len(items),
            "retrieval_hit_rate": round(mean(item["retrieval_hit"] for item in items), 4),
            "top1_hit_rate": round(mean(item["top1_hit"] for item in items), 4),
            "source_recall": round(mean(item["source_recall"] for item in items), 4),
            "mrr": round(mean(item["mrr"] for item in items), 4),
            "keyword_pass_rate": round(mean(keyword_passes), 4) if keyword_passes else None,
            "grounded_answer_rate": round(
                mean(
                    1
                    if item["retrieval_hit"] == 1 and item["has_substantive_answer"] == 1 and item["keyword_pass"] in {None, 1}
                    else 0
                    for item in items
                ),
                4,
            ),
        }
    return summary


def build_headline_metrics(custom_metrics, audit):
    return {
        "sample_size": audit["case_count"],
        "category_count": len(audit["category_counts"]),
        "retrieval_hit_rate": custom_metrics["retrieval_hit_rate"],
        "top1_hit_rate": custom_metrics["top1_hit_rate"],
        "mrr": custom_metrics["mrr"],
        "source_recall": custom_metrics["source_recall"],
        "grounded_answer_rate": custom_metrics["grounded_answer_rate"],
        "keyword_pass_rate": custom_metrics["keyword_pass_rate"],
    }


def build_resume_summary(custom_metrics, audit, ragas_report, ragas_error):
    lines = [
        (
            f"Evaluated on {audit['case_count']} repo-QA cases across "
            f"{len(audit['category_counts'])} categories."
        ),
        (
            f"Deterministic retrieval metrics: hit@{TOP_K} {custom_metrics['retrieval_hit_rate']:.1%}, "
            f"top-1 hit {custom_metrics['top1_hit_rate']:.1%}, MRR {custom_metrics['mrr']:.3f}, "
            f"source recall {custom_metrics['source_recall']:.1%}."
        ),
        (
            f"Answer quality checks: grounded answer rate {custom_metrics['grounded_answer_rate']:.1%}"
            + (
                f", keyword/checklist pass rate {custom_metrics['keyword_pass_rate']:.1%}."
                if custom_metrics["keyword_pass_rate"] is not None
                else "."
            )
        ),
    ]

    if ragas_report and not ragas_error:
        lines.append(
            "LLM-judge metrics (supporting signal, not primary headline): "
            f"faithfulness {ragas_report.get('faithfulness', 0.0):.3f}, "
            f"answer relevancy {ragas_report.get('answer_relevancy', 0.0):.3f}, "
            f"context precision {ragas_report.get('context_precision', 0.0):.3f}."
        )
    else:
        lines.append("LLM-judge metrics were skipped or unstable, so headline metrics rely on deterministic checks.")

    if audit["warnings"]:
        lines.append(
            "Benchmark caveat: "
            + " ".join(audit["warnings"][:2])
        )

    return " ".join(lines)


def benchmark_readiness(audit, ragas_error):
    reasons = []
    if audit["case_count"] < 25:
        reasons.append("small_sample")
    if len(audit["category_counts"]) < 4:
        reasons.append("limited_category_coverage")
    if audit["conversation_case_count"] < 2:
        reasons.append("limited_multi_turn_coverage")
    if audit["warnings"]:
        reasons.append("dataset_scope_warnings")
    if ragas_error not in {None, "disabled"}:
        reasons.append("ragas_instability")

    if reasons:
        return {
            "status": "internal_or_demo_benchmark",
            "reasons": reasons,
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
    import boto3
    from langchain_core.outputs import Generation, LLMResult
    from ragas.llms.base import BaseRagasLLM

    class BedrockRagasLLM(BaseRagasLLM):
        def __init__(self, model: str, region: str, run_config):
            self.client = boto3.client("bedrock-runtime", region_name=region)
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
            inference_config = {
                "temperature": 0.0,
                "maxTokens": int(os.getenv("EVAL_MAX_OUTPUT_TOKENS", "2048")),
            }
            if stop:
                inference_config["stopSequences"] = stop

            response = self.client.converse(
                modelId=self.model,
                messages=[
                    {
                        "role": "user",
                        "content": [{"text": prompt_text}],
                    }
                ],
                inferenceConfig=inference_config,
            )

            generations = []
            output_message = (response.get("output") or {}).get("message") or {}
            content_blocks = output_message.get("content") or []
            text = "".join(
                block.get("text", "") for block in content_blocks if isinstance(block, dict)
            ).strip()
            if text:
                generations.append(Generation(text=text))

            if not generations:
                raise RuntimeError("AWS Bedrock judge returned an empty response.")

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

    region = os.getenv("AWS_REGION", os.getenv("AWS_DEFAULT_REGION", "us-east-1"))
    model = os.getenv(
        "EVAL_MODEL",
        os.getenv("BEDROCK_EVAL_MODEL", "us.anthropic.claude-haiku-4-5-20251001"),
    )
    return BedrockRagasLLM(model=model, region=region, run_config=run_config)


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
        from ragas.metrics import answer_relevancy, context_precision, faithfulness
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
        max_workers = int(os.getenv("EVAL_MAX_WORKERS", "4"))
        run_config = RunConfig(
            timeout=timeout_seconds,
            thread_timeout=thread_timeout_seconds,
            max_workers=max_workers,
            max_retries=int(os.getenv("EVAL_MAX_RETRIES", "3")),
            max_wait=int(os.getenv("EVAL_MAX_WAIT_SECONDS", "60")),
        )
        log(
            "Using AWS Bedrock for RAGAS judge model "
            f"({os.getenv('EVAL_MODEL', os.getenv('BEDROCK_EVAL_MODEL', 'us.anthropic.claude-haiku-4-5-20251001'))})"
        )
        log(
            f"RAGAS runtime: async={RAGAS_ASYNC}, raise_exceptions={RAGAS_RAISE_EXCEPTIONS}, "
            f"timeout={timeout_seconds}s, thread_timeout={thread_timeout_seconds}s, max_workers={max_workers}"
        )
        llm = build_bedrock_ragas_llm(run_config)
        embeddings = build_ragas_embeddings(run_config)
        ragas_report = evaluate(
            build_ragas_dataset(),
            metrics=[faithfulness, answer_relevancy, context_precision],
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
    if audit["errors"]:
        raise RuntimeError("Eval set validation failed: " + "; ".join(audit["errors"]))
    for warning in audit["warnings"]:
        log(f"Eval set warning: {warning}")
    log(
        f"Starting eval with api_url={API_URL}, repo_id={REPO_ID}, "
        f"session_id={SESSION_ID}, top_k={TOP_K}, cases={len(rows)}"
    )
    outputs = []
    details = []

    for index, row in enumerate(rows, start=1):
        case_id = row.get("id", row["question"])
        log(f"[{index}/{len(rows)}] Querying case {case_id}")
        result = post_query(row)
        outputs.append(result)
        log(
            f"[{index}/{len(rows)}] Received answer for {case_id} "
            f"with {len(result.get('sources', []))} sources"
        )

        cited_paths = [source["file_path"] for source in result.get("sources", [])]
        metrics = compute_retrieval_metrics(row.get("expected_sources", []), cited_paths)
        keyword_coverage = keyword_match_ratio(row, result.get("answer", ""))
        keyword_gate = keyword_pass(row, result.get("answer", ""), keyword_coverage)
        length_metrics = answer_length_metrics(result.get("answer", ""))
        overlap = lexical_overlap_ratio(row.get("ground_truth", ""), result.get("answer", ""))

        details.append(
            {
                "id": row.get("id", row["question"]),
                "category": row.get("category", "general"),
                "question": row["question"],
                "answer": result.get("answer", ""),
                "expected_sources": row.get("expected_sources", []),
                "retrieved_sources": cited_paths,
                "retrieval_hit": metrics["retrieval_hit"],
                "source_recall": metrics["source_recall"],
                "mrr": metrics["mrr"],
                "top1_hit": metrics["top1_hit"],
                "unique_source_precision": metrics["unique_source_precision"],
                "duplicate_source_rate": metrics["duplicate_source_rate"],
                "keyword_coverage": keyword_coverage,
                "keyword_pass": keyword_gate,
                "ground_truth_lexical_overlap": overlap,
                **length_metrics,
            }
        )

    log("Finished query loop. Computing aggregate metrics.")
    custom_metrics = summarize_custom_metrics(details)
    category_breakdown = summarize_by_category(details)
    ragas_report, ragas_error = run_ragas(rows, outputs)
    headline_metrics = build_headline_metrics(custom_metrics, audit)
    resume_summary = build_resume_summary(custom_metrics, audit, ragas_report, ragas_error)
    readiness = benchmark_readiness(audit, ragas_error)

    report = {
        "config": {
            "api_url": API_URL,
            "repo_id": REPO_ID,
            "session_id": SESSION_ID,
            "top_k": TOP_K,
            "query_timeout_seconds": QUERY_TIMEOUT_SECONDS,
            "eval_set": str(EVAL_SET_PATH),
        },
        "eval_set_audit": audit,
        "headline_metrics": headline_metrics,
        "benchmark_readiness": readiness,
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
