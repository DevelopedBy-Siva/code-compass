# Retrieval evaluation

The benchmark in [`server/evals/sample_eval_set.json`](../server/evals/sample_eval_set.json) contains 120 hand-written repository questions: 40 each for Documenso, FastAPI, and Django. Each case includes a reference answer (`ground_truth`) and one or more expected source files or directories (`expected_sources`). Relevance is binary: a result is relevant when its indexed file path matches an expected file exactly or lies under an expected directory. The same rule is used for both retrieval methods and every metric.

## Metrics

- **Hit@1**, **Hit@5**, and **Hit@8** are 1 when at least one relevant file appears within the corresponding prefix of the ranking, otherwise 0. Reported values are the mean across queries.
- **MRR@8** is the mean reciprocal rank of the first relevant file, with results after rank 8 treated as zero.

The reported, previously verified 120-query results are:

| Metric | Vector-only | Hybrid + reranking |
|---|---:|---:|
| Hit@1 | 52.5% | 70.8% |
| Hit@5 | 78.3% | 88.3% |
| Hit@8 | 85.8% | 92.5% |
| MRR@8 | 0.63 | 0.78 |

These values are provided evaluation results, not values embedded in the scoring code. Each generated JSON report includes the actual ranking and per-query metric values used to form its aggregate.

## Compared retrieval methods

Both methods run against the same evaluation index during a run:

1. The production retrieval pipeline in retrieval-only mode resolves any follow-up question and builds one normalized retrieval query without calling a language model.
2. **Vector-only** embeds that same query and searches Qdrant directly for eight results. It skips lexical search, path and intent signals, fusion, and reranking.
3. **Hybrid + reranking** uses the same normalized query to combine semantic and BM25 candidates with path and intent signals, fuse and rerank candidates, then return up to eight selected sources. Answer generation is disabled, so Bedrock generation does not influence retrieval metrics.

Both methods use the same normalized query, `expected_sources`, repository index, embedding model, and top-eight evaluation window. The report records the normalized query, evaluation collection, model identifiers, batch sizes, repository branch, cache generation, and whether an index was reused.

## Reproduce

Prerequisites: Python dependencies from `server/requirements.txt`, access to Qdrant, and enough local resources to load the configured embedding and reranking models. Configure `QDRANT_URL` and, if needed, `QDRANT_API_KEY` in `server/.env` or the environment.

From the repository root:

```bash
CODEBASE_RAG_REINDEX=1 \
CODEBASE_RAG_EVAL_OUTPUT=server/evals/results/retrieval.json \
python server/evals/run_eval.py
```

`CODEBASE_RAG_REINDEX=1` rebuilds the evaluation indexes before scoring. Use `CODEBASE_RAG_REINDEX=0` to reuse cached indexes; the runner indexes any missing repository. Set `CODEBASE_RAG_EVAL_REPOS=fastapi` to run a subset while iterating. Valid IDs are `documenso`, `fastapi`, and `django`. A subset is useful for iteration, but it is not the full 120-query benchmark. `CODEBASE_RAG_EVAL_SET` can point to a compatible JSON set, and `CODEBASE_RAG_EVAL_OUTPUT` writes the full report while JSON is also printed to stdout.

The runner selects the isolated collection named by `QDRANT_EVAL_COLLECTION` (default: `code_compass_eval_qwen3_embedding_0_6b_last_token_cache_v2`) so evaluation indexing does not replace the application's collection. Keep the same evaluation-set file, repository index generation, model IDs, and indexing configuration when comparing runs. Repository sources currently follow their configured GitHub default branch; the dataset does not pin source commit hashes, so rebuilding at a later date can index changed upstream code.

## Separate RAG and answer-quality evaluation

Run the optional answer-quality evaluation separately when Bedrock generation is available:

```bash
CODEBASE_RAG_REINDEX=0 \
CODEBASE_RAG_ENABLE_FAITHFULNESS=1 \
CODEBASE_RAG_RAG_EVAL_OUTPUT=server/evals/results/rag_quality.json \
python server/evals/run_rag_eval.py
```

This runner reports citation presence, expected-source retrieval, a grounded-answer proxy (expected source retrieved, citations present, and at least one configured `must_include_any` keyword matched), optional model-judged faithfulness, and latency. The faithfulness judge uses the same Bedrock model family as generation, so treat it as a regression signal rather than an independent quality measure. Set `CODEBASE_RAG_ENABLE_FAITHFULNESS=0` to skip the extra judge call.

Stage-level Hit and MRR scores, candidate coverage, retrieval rank traces, and the conversation trace appear under `diagnostics_summary` and each case's `diagnostics` field. They are debugging evidence and do not contribute to the primary benchmark summary. This report has a separate output path and separate `answer_quality` summary.

The answer-quality run should reuse the evaluation collection and cache generation from the retrieval run (`CODEBASE_RAG_REINDEX=0`). Its retrieval traces are useful for diagnosis but are not a second primary benchmark result. Both reports record repository branches and cache generations so the compared index can be checked.

The evaluation uses binary expected-source relevance rather than graded judgments and does not report statistical uncertainty.
