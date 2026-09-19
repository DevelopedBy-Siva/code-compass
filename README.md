# Code Compass

**Ask questions about an unfamiliar GitHub repository and get answers grounded in its source code.**

Code Compass v1.0.0 is a full-stack codebase RAG system for exploring public GitHub repositories. It builds a code-aware index, retrieves and reranks the most relevant evidence, and produces concise answers with validated file and line-level citations.

![Code Compass repository selection screen](images/landing.png)

## The problem

Understanding a large, unfamiliar repository means tracing symbols, configuration, tests, and cross-file request flows—not merely finding text that sounds related.

Embedding search alone misses exact identifiers and can over-rank duplicated documentation, translated files, tests, or semantically similar but non-canonical sources.

For an answer to be useful, retrieval must surface the actual implementation and generation must stay grounded in that evidence. Code Compass was built to make both stages observable and measurable.

## The solution

Code Compass combines **code-aware chunking**, **hybrid retrieval**, **model-based reranking**, and **grounded generation**. It returns answers with validated citations so a developer can move directly from an explanation to the supporting source.

## Architecture

```text
Browser
   │
   ▼
Vercel (React + API proxy)
   │  OIDC
   ▼
SageMaker real-time endpoint
   ├── Qwen3 Embedding
   ├── Qwen3 Reranker
   ├── Qdrant Cloud
   └── Amazon Bedrock
```

The backend clones and parses a public repository, stores reusable embeddings in Qdrant, fuses semantic and lexical candidates, reranks the strongest evidence, and asks Qwen3-Coder-Next on Bedrock to answer only from the selected context.

## Engineering highlights

### Code-aware indexing

Tree-sitter chunks source around functions, classes, methods, and declarations. Module overviews preserve file-level context, while bounded fallback chunking handles documentation and unsupported languages.

### Hybrid retrieval

Semantic search, BM25, path and intent signals, and Reciprocal Rank Fusion cover both conceptual questions and exact identifiers. Diversity-aware selection prevents repeated or translated sources from crowding out implementation files.

### Repository caching

Qdrant stores persistent, branch-aware repository indexes. Re-indexing builds a hidden generation and activates it atomically, so a failed rebuild cannot replace the last working cache.

### GPU inference on SageMaker

Qwen3-Embedding-0.6B and Qwen3-Reranker-0.6B run together on one GPU endpoint. Batched inference improves throughput while a single worker preserves the application’s process-local session and lexical state.

### CUDA compatibility debugging

The container is based on the official PyTorch CUDA runtime to keep PyTorch, CUDA, and cuDNN aligned. Production startup can require CUDA explicitly and fails fast instead of silently falling back to CPU.

### CI/CD pipeline

```text
Unit Tests
    │
    ▼
Build + Parity Test
    │
    └── Pull requests stop after validation
    │
    ▼ main push / manual dispatch
Push ECR (GitHub OIDC)
    │
    ▼
Deploy SageMaker (wait for InService)
    │
    ▼
Smoke Test (live invocation + failure diagnostics)
```

One workflow exposes five focused jobs connected by `needs`. Every push and pull request runs through unit and parity validation; ECR publishing and deployment are gated to `main` pushes or manual dispatch. The production image stays on one runner while it is built and parity-tested, then the verified image is passed once to the isolated ECR runner as a short-lived artifact. Deployment uses short-lived GitHub OIDC credentials and succeeds only after SageMaker reaches `InService` and serves a real invocation. Failures automatically print the endpoint description, `FailureReason`, and the last 100 events from the newest CloudWatch log stream. The production deployment role therefore needs `sagemaker:InvokeEndpoint`, `logs:DescribeLogStreams`, and `logs:GetLogEvents` in addition to its existing ECR and SageMaker deployment permissions.

The same-origin Vercel function exchanges its workload identity for a narrowly scoped AWS role and invokes SageMaker without exposing AWS credentials to the browser. No long-lived AWS access keys are used by either deployment path.

### Local SageMaker parity testing

The local deployment test runs the production image as its non-root user, exercises `/ping`, `/api/health`, and `/invocations`, and prints container logs on failure before an image can be pushed.

### Structured logging

Every request receives a correlation ID. Runtime logs use only `[startup]`, `[index]`, `[query]`, and `[error]` categories. Routine dependency HTTP traffic and Uvicorn access records are suppressed, while warnings and errors remain visible. Index logs report repository, file/chunk counts, embedding time, and total time; query logs report the question, any rewrite, retrieved-document count, and latency.

## Evaluation

The evaluation suite contains 24 hand-written questions across Documenso, FastAPI, and Django. It reports retrieval-stage metrics and optional model-judged faithfulness. See the [operations guide](docs/deployment.md#evaluation) for commands and methodology.

## Deployment overview

```text
GitHub ──► GitHub Actions ──► Amazon ECR ──► SageMaker
                                                 ├── Bedrock
                                                 └── Qdrant Cloud

React  ──► Vercel ── OIDC / InvokeEndpoint ──────┘
```

The deployment uses immutable SageMaker model/config versions, a stable endpoint, non-root containers, health-gated startup, and short-lived OIDC credentials for both CI/CD and runtime invocation. See the [AWS deployment guide](docs/deployment.md) for the serving contract, IAM scope, scripts, configuration, and production tradeoffs.

### Deployment proof

| GitHub Actions | SageMaker |
|---|---|
| ![Successful GitHub Actions deployment](images/cicd.png) | ![SageMaker endpoint InService](images/sagemaker.png) |

![Successful Vercel production deployment](images/vercel.png)

## Tech stack

| Layer | Technology |
|---|---|
| Frontend | React 19, Tailwind CSS, Axios, Vercel |
| API | FastAPI, Pydantic, Uvicorn |
| Parsing | tree-sitter, symbol-aware and fallback chunking |
| Retrieval | Qwen3-Embedding-0.6B, BM25, RRF, Qdrant Cloud |
| Reranking | Qwen3-Reranker-0.6B, PyTorch, Transformers |
| Generation | Qwen3-Coder-Next on Amazon Bedrock |
| Infrastructure | Docker, Amazon ECR, SageMaker, GitHub Actions OIDC |

## Local development

Prerequisites: Python 3.11+, Node.js 18+, a Qdrant instance, and AWS credentials with access to the configured Bedrock model. CUDA is strongly recommended for embedding and reranking.

### Backend

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r server/requirements.txt

cd server
export AWS_REGION=us-east-1
export QDRANT_URL=https://your-cluster.us-east.aws.cloud.qdrant.io:6333
export QDRANT_API_KEY=your-qdrant-api-key
export PORT=8000
python server_app.py
```

### Frontend

```bash
cd ui
npm ci
cp .env.example .env
npm start
```

Open `http://localhost:3000`, submit a public GitHub URL, wait for indexing, and start asking questions. For evaluation commands, API examples, runtime configuration, and AWS deployment, see [docs/deployment.md](docs/deployment.md).

## Current scope

Code Compass supports public repositories and keeps session metadata and BM25 indexes in one process.
