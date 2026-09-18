# Code Compass

### Ask questions about an unfamiliar GitHub repository and get answers grounded in its source code.

Code Compass is a full-stack codebase RAG system. Give it a public GitHub URL, let it build a code-aware index, and ask questions such as:

> Where is authentication implemented?

> How does an incoming request become a response?

> Which files participate in the document-signing flow?

Instead of answering from model memory, Code Compass retrieves evidence from the repository, reranks it, and returns a concise answer with file and line-level citations.

The project is designed around the parts of code RAG that are easy to underestimate: symbol-aware chunking, exact identifier search, cross-file questions, canonical-source ranking, noisy tests and translated documentation, conversational follow-ups, and measurable retrieval quality.

## Results

The evaluation suite contains 24 hand-written questions across three real, unfamiliar repositories:

- [Documenso](https://github.com/documenso/documenso) — large TypeScript monorepo
- [FastAPI](https://github.com/fastapi/fastapi) — Python API framework
- [Django](https://github.com/django/django) — large Python web framework

> The results below validate the 0.6B model stack. Re-run the benchmark after
> the Qdrant migration before treating the numbers as validated for the new
> vector-store backend.

### Retrieval improvement

| Metric | Baseline | Optimized | Change |
|---|---:|---:|---:|
| Final-context retrieval hit rate | 58.33% (14/24) | **91.67% (22/24)** | **+33.34 pp** |
| Top-1 hit rate | 33.33% (8/24) | **70.83% (17/24)** | **+37.50 pp** |
| Mean Reciprocal Rank | 0.441 | **0.813** | **+84.2%** |
| Candidate retrieval hit rate | — | **100% (24/24)** | — |
| Expected-source grounded rate | 58.33% | **91.67%** | **+33.34 pp** |
| LLM-judged faithfulness | — | **99.17%** | — |

### Results by repository

| Repository | Retrieval | Top-1 | MRR |
|---|---:|---:|---:|
| Documenso | **87.5%** | **75.0%** | **0.813** |
| FastAPI | **87.5%** | **62.5%** | **0.750** |
| Django | **100%** | **75.0%** | **0.875** |

The optimized results use Qwen3-Embedding-0.6B and Qwen3-Reranker-0.6B across one complete run with eight cases per repository. All 24 answers received a faithfulness score; 23 scored 1.0 and one scored 0.8. The same Bedrock model generates and judges the answers, so faithfulness is a useful regression signal rather than an independent evaluation. The benchmark is intentionally small and should be read as a regression suite, not a universal code-retrieval benchmark.

Two remaining misses are known and diagnosable: Documenso's signing-package implementation and FastAPI's OpenAPI generation. In both cases the expected implementation source enters the candidate set but falls outside the final eight sources. The generated answers remain faithful to alternative retrieved evidence, indicating that the next improvement belongs in final retrieval prioritization—not in fine-tuning the answer model.

## Product walkthrough

### Repository selection and indexing

![Code Compass landing screen](images/landing.png)

### Grounded answers with source citations

![Code Compass chat and citations screen](images/chat.png)

## How it works

```text
Public GitHub repository
          │
          ▼
 Clone, filter, and parse files
          │
          ▼
 tree-sitter symbol chunks + module overviews
          │
          ├───────────────┐
          ▼               ▼
 Qwen3 embeddings       BM25
          │               │
          ▼               │
 Qdrant vector search     │
          └──────┬────────┘
                 ▼
       Reciprocal Rank Fusion
                 │
        Path and intent signals
                 │
                 ▼
      Qwen3-Reranker-0.6B
                 │
       Diversity-aware selection
                 │
                 ▼
 Qwen3-Coder-Next on Amazon Bedrock
                 │
                 ▼
 Answer + validated source citations
```

### 1. Code-aware indexing

The backend shallow-clones a public repository, ignores generated and dependency directories, and keeps supported source, configuration, markup, and documentation files.

Tree-sitter creates chunks around functions, classes, methods, declarations, and containers. Large classes receive compact overview chunks while their methods remain independently searchable. Each parsed source file also receives a module overview containing its path, role, imports, exports, and symbols. Plain-text and unsupported-language files use bounded fallback chunking.

Repository clones are deleted after indexing; only chunks, metadata, and embeddings remain.

Qdrant is also the persistent repository cache. A canonical repository-and-branch key lets a new browser session reuse a completed index without cloning or embedding the repository again. The landing-page **Re-index repository** switch is off by default. Turning it on builds a new, hidden generation; only after every vector is stored does the backend activate it and delete the previous generation. A failed rebuild therefore leaves the last working cache intact.

### 2. Hybrid candidate retrieval

Every question uses multiple complementary signals:

- **Semantic retrieval** with Qwen3-Embedding-0.6B for conceptual similarity
- **BM25 retrieval** for filenames, symbols, framework terminology, and exact identifiers
- **Path and intent retrieval** for likely implementation locations
- **Reciprocal Rank Fusion** to combine independently ranked channels

The system over-fetches before deduplication so repeated chunks and translated copies of the same documentation cannot consume the entire candidate budget.

### 3. Model-based reranking

Qwen3-Reranker-0.6B scores query–chunk pairs using the model's native `yes`/`no` relevance format. Candidates are processed in configurable GPU batches and then combined with lexical, semantic, path, and canonical-source signals.

The final selector limits repeated chunks from the same file and adds source diversity for cross-file questions.

### 4. Grounded answer generation

The selected evidence is sent to Qwen3-Coder-Next through Amazon Bedrock. The prompt requires the model to:

- use only retrieved repository context;
- name concrete files and symbols;
- distinguish evidence from inference;
- add inline citations such as `[1]` and `[2]`;
- acknowledge when the available evidence is insufficient.

Returned citation numbers are validated against the supplied sources before the API response is sent to the client.

## What improved retrieval quality

The initial system retrieved a relevant final source in only 14 of 24 cases. Retrieval diagnostics exposed the failure stage for every question and led to several targeted fixes:

- Replaced incorrect mean pooling with Qwen3's required last-token embedding pooling.
- Added task instructions to query embeddings while leaving document embeddings unprefixed.
- Corrected the Qwen3 reranker prompt and relevance-token scoring. The broken version assigned the same score to every candidate.
- Batched reranking for GPU throughput.
- Increased candidate recall through controlled over-fetching.
- Prevented localized documentation and repeated file chunks from flooding retrieval.
- Reduced the influence of noisy path-only matches.
- Improved cross-file question classification.
- Penalized test sources for non-test questions.
- Removed an early top-eight truncation that discarded relevant sources before final selection.
- Fixed Unicode byte-offset handling in tree-sitter symbol extraction.
- Added per-stage metrics so candidate generation, fusion, reranking, and final-context failures are measured separately.

This moved the system from **58.33% to 91.67% final-context retrieval** without fine-tuning a model.

## Evaluation

The evaluation set covers repository purpose, implementation lookup, API behavior, configuration, tests, error handling, security, cross-file flows, and conversational follow-ups.

Reported metrics include:

- candidate retrieval hit rate;
- semantic, lexical, fused, and path hit rates;
- reranker and prioritized Top-K hit rates;
- final-context retrieval hit rate;
- Top-1 hit rate and Mean Reciprocal Rank;
- expected-source grounded-answer rate;
- optional LLM-judged faithfulness;

Run the complete evaluation from the repository root:

```bash
CODEBASE_RAG_REINDEX=1 \
CODEBASE_RAG_EVAL_OUTPUT=server/evals/results/full_eval.json \
python server/evals/run_eval.py
```

Run a single repository while iterating:

```bash
CODEBASE_RAG_EVAL_REPOS=fastapi \
CODEBASE_RAG_REINDEX=1 \
CODEBASE_RAG_ENABLE_FAITHFULNESS=0 \
CODEBASE_RAG_EVAL_OUTPUT=server/evals/results/fastapi.json \
python server/evals/run_eval.py
```

Valid repository IDs are `documenso`, `fastapi`, and `django`.

The cases are defined in [`server/evals/sample_eval_set.json`](server/evals/sample_eval_set.json). Retrieval regression tests live in [`server/tests/test_retrieval_quality.py`](server/tests/test_retrieval_quality.py).

## Technology stack

| Layer | Technologies |
|---|---|
| Frontend | React 19, Tailwind CSS, Axios |
| API | FastAPI, Pydantic, Uvicorn |
| Parsing | tree-sitter, language-specific syntax trees, fallback text chunking |
| Retrieval | Qwen3-Embedding-0.6B, Qdrant, BM25, Reciprocal Rank Fusion |
| Reranking | Qwen3-Reranker-0.6B, PyTorch, Hugging Face Transformers |
| Generation | Qwen3-Coder-Next through Amazon Bedrock |
| Infrastructure | Docker-ready backend, Vercel-ready frontend |

## AWS production-demo architecture

```text
Browser
  │ HTTPS
  ▼
Vercel (React static site + same-origin serverless API adapter)
  │ Vercel OIDC → short-lived AWS credentials
  ▼
SageMaker Runtime InvokeEndpoint API
  │ POST /invocations on port 8080
  ▼
SageMaker real-time endpoint (one GPU instance, one FastAPI worker)
  ├── Qwen3 embedding + reranker models baked into the ECR image
  ├── Qdrant Cloud ── persistent vectors and repository cache
  ├── Amazon Bedrock ── grounded answer generation
  └── CloudWatch Logs ── container stdout/stderr and endpoint logs

GitHub Actions (OIDC) ── build/test ──► ECR ──► SageMaker deployment
```

SageMaker real-time endpoints are AWS APIs, not public general-purpose web
servers. The Vercel function in `ui/api/[...path].js` therefore maps the
existing browser REST calls to the container's single `/invocations` route. It
uses Vercel OIDC federation and never sends AWS credentials to the browser.
The direct `/api/*` routes remain available for local development.

The container implements SageMaker's serving contract: it listens on
`0.0.0.0:8080`, accepts health checks on `/ping`, accepts inference requests on
`/invocations`, handles SageMaker's `serve` argument, and exits cleanly on
`SIGTERM`. Startup does not report healthy until the local models and Qdrant
client are ready.

### Required AWS resources

- An ECR repository. `scripts/push.sh` creates it with scan-on-push if missing.
- A SageMaker execution role trusted by `sagemaker.amazonaws.com`.
- A SageMaker real-time endpoint quota for the selected GPU instance. The
  scripts default to one `ml.g5.xlarge`; confirm regional model memory and
  quota before deployment.
- Bedrock access to the configured `BEDROCK_MODEL_ID` in the same region.
- A Qdrant Cloud cluster reachable from the endpoint.
- A Qdrant Cloud API key. The supplied workflow reads it from the
  `QDRANT_API_KEY` GitHub environment secret. For stronger production secret
  handling, store the value in AWS Secrets Manager and deploy its ARN through
  `QDRANT_API_KEY_SECRET_ARN` instead.
- GitHub and Vercel OIDC identity providers plus narrowly scoped IAM roles.

The SageMaker execution role needs:

- ECR pull: `ecr:GetAuthorizationToken`, `ecr:BatchCheckLayerAvailability`,
  `ecr:GetDownloadUrlForLayer`, and `ecr:BatchGetImage`;
- logging: `logs:CreateLogGroup`, `logs:CreateLogStream`,
  `logs:PutLogEvents`, `logs:DescribeLogStreams`, and
  `cloudwatch:PutMetricData`;
- generation: `bedrock:InvokeModel` on the selected model/inference-profile
  ARN;
- configuration, only when using `QDRANT_API_KEY_SECRET_ARN`:
  `secretsmanager:GetSecretValue` on the Qdrant secret.

The GitHub deployment role needs `ecr:GetAuthorizationToken`,
`ecr:CreateRepository`, `ecr:DescribeRepositories`,
`ecr:BatchCheckLayerAvailability`, `ecr:InitiateLayerUpload`,
`ecr:UploadLayerPart`, `ecr:CompleteLayerUpload`, and `ecr:PutImage`, plus
`sagemaker:CreateModel`,
`sagemaker:DescribeModel`, `sagemaker:CreateEndpointConfig`,
`sagemaker:DescribeEndpointConfig`, `sagemaker:CreateEndpoint`,
`sagemaker:UpdateEndpoint`, `sagemaker:DeleteEndpoint`,
`sagemaker:DescribeEndpoint`, and
`iam:PassRole` restricted to the SageMaker execution role with
`iam:PassedToService = sagemaker.amazonaws.com`. Its trust policy should
restrict GitHub's OIDC `sub` claim to this repository, the `main` branch, and
the production environment.

The Vercel runtime role needs only `sagemaker:InvokeEndpoint` on this endpoint.
Restrict its OIDC trust policy to the production Vercel project and environment.

### Deploy from a workstation

Prerequisites are Docker, AWS CLI v2, `jq`, `openssl`, and AWS credentials that
can push to ECR and deploy SageMaker resources.

```bash
export AWS_REGION=us-east-1
export ECR_REPOSITORY=code-compass-backend
export SAGEMAKER_ENDPOINT_NAME=code-compass
export SAGEMAKER_EXECUTION_ROLE_ARN=arn:aws:iam::123456789012:role/code-compass-sagemaker
export SAGEMAKER_INSTANCE_TYPE=ml.g5.xlarge
export QDRANT_URL=https://your-cluster.us-east.aws.cloud.qdrant.io:6333
export QDRANT_API_KEY=your-qdrant-api-key
export CORS_ORIGINS=https://your-project.vercel.app

./scripts/build.sh
./scripts/push.sh
./scripts/deploy.sh
```

`build.sh` produces a Linux/amd64 image and preloads both Hugging Face model
snapshots. Set `PRELOAD_MODELS=0` for a faster development build; that image
requires outbound Hugging Face access at startup and is not recommended for
production. `push.sh` logs in to ECR, tags, and pushes the image. `deploy.sh`
uses an image-and-environment hash for immutable SageMaker model/config names,
creates missing resources, updates an existing endpoint only when needed, and
waits for `InService`. If an endpoint is in `Failed`, the script deletes it,
waits for deletion to finish, and recreates it with the desired configuration.

The Dockerfile deliberately uses the official PyTorch CUDA runtime as a single
stage. A conventional Python builder stage would download and retain another
copy of several gigabytes of CUDA/PyTorch wheels without reducing the runtime
layer. The image is still large because it contains two 0.6B models plus the GPU
runtime; baking those models trades ECR storage and build time for predictable,
network-independent endpoint startup. Runtime code is copied last for useful
layer caching, package caches are removed, and the service runs as a non-root
user with only Git retained for repository cloning.

The script deliberately retains old models and endpoint configurations for
rollback. Delete unused versions periodically after confirming a deployment.

### GitHub Actions CI/CD

`.github/workflows/deploy-sagemaker.yml` runs on backend/deployment changes to
`main`: it installs dependencies, runs the backend tests, builds the image,
pushes it to ECR, and updates the endpoint. It requests `id-token: write` and
assumes the deployment role using GitHub OIDC; no AWS access-key secrets are
used.

Create a protected GitHub environment named `production` with these secrets:

| GitHub environment secret | Purpose |
|---|---|
| `AWS_GITHUB_ROLE_ARN` | OIDC deployment role assumed by Actions |
| `AWS_REGION` | ECR, SageMaker, Secrets Manager, and Bedrock region |
| `ECR_REPOSITORY` | Backend ECR repository name |
| `SAGEMAKER_ENDPOINT_NAME` | Stable endpoint name |
| `SAGEMAKER_EXECUTION_ROLE_ARN` | Runtime role passed to SageMaker |
| `SAGEMAKER_INSTANCE_TYPE` | Endpoint instance type, normally `ml.g5.xlarge` |
| `QDRANT_URL` | Qdrant Cloud HTTPS endpoint |
| `QDRANT_API_KEY` | Qdrant Cloud API key used by the supplied workflow |
| `CORS_ORIGINS` | Production Vercel origin |
| `BEDROCK_MODEL_ID` | Bedrock model or inference-profile identifier |

All values in the table are read through GitHub's `secrets` context. Environment
protection rules and required reviewers are recommended for production
deployment.

The direct `QDRANT_API_KEY` path is convenient for a portfolio environment, but
the deployment script places it in the SageMaker model's container environment.
AWS advises against putting sensitive values in `CreateModel` environment
fields. For a longer-lived or shared environment, use
`QDRANT_API_KEY_SECRET_ARN`; the container will retrieve the value at startup,
and the SageMaker execution role must have `secretsmanager:GetSecretValue` for
that ARN.

For Vercel, set `AWS_ROLE_ARN`, `SAGEMAKER_AWS_REGION`, and
`SAGEMAKER_ENDPOINT_NAME`. Enable Vercel OIDC and configure the role trust
policy. Leave `REACT_APP_API_URL` unset in production so the UI uses the
same-origin adapter. Locally, set it to `http://localhost:8000`.

## Run locally

### Prerequisites

- Python 3.11+
- Node.js 18+
- AWS credentials with access to Qwen3-Coder-Next in Amazon Bedrock
- A CUDA GPU is strongly recommended for the local embedding and reranking models

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

The container defaults to port `8080` for SageMaker. The local example sets
`PORT=8000` to match the frontend and curl examples below.

### Frontend

In another terminal:

```bash
cd ui
npm install

cat > .env <<'ENV'
REACT_APP_API_URL=http://localhost:8000
ENV

npm start
```

The UI starts at `http://localhost:3000`.

## API

All repository operations are scoped by an `X-Session-Id` header.

| Method | Endpoint | Purpose |
|---|---|---|
| `POST` | `/api/repos/index` | Reuse a cached index or asynchronously build one |
| `GET` | `/api/repos` | List repositories for the current session |
| `GET` | `/api/repos/{repo_id}` | Read indexing status and metadata |
| `POST` | `/api/query` | Ask a grounded question about an indexed repository |
| `POST` | `/api/session/end` | Clear session state; persistent Qdrant vectors remain cached |

Index a repository, reusing its cache by default:

```bash
curl -X POST http://localhost:8000/api/repos/index \
  -H 'Content-Type: application/json' \
  -H 'X-Session-Id: portfolio-demo-session' \
  -d '{
    "github_url": "https://github.com/fastapi/fastapi",
    "reindex": false
  }'
```

Set `reindex` to `true` to force a safe replacement of the stored index.

Example query:

```bash
curl -X POST http://localhost:8000/api/query \
  -H 'Content-Type: application/json' \
  -H 'X-Session-Id: portfolio-demo-session' \
  -d '{
    "repo_id": 1,
    "question": "Where is request validation handled?",
    "top_k": 8,
    "history": []
  }'
```

## Configuration

| Variable | Default | Purpose |
|---|---|---|
| `AWS_REGION` | `us-east-1` fallback | Amazon Bedrock region |
| `APP_ENV` | `local` | Runtime environment label |
| `LOG_LEVEL` | `INFO` | Application/Uvicorn log level |
| `PORT` | `8080` | Direct container/listener port |
| `BEDROCK_MODEL_ID` | `qwen.qwen3-coder-next` | Bedrock model or inference profile |
| `QDRANT_URL` | required | Qdrant cluster REST endpoint |
| `QDRANT_API_KEY` | none | Qdrant Cloud API key; omit only for an unsecured local instance |
| `QDRANT_API_KEY_SECRET_ARN` | none | Preferred AWS source for the Qdrant key; used when direct key is absent |
| `QDRANT_COLLECTION` | versioned default | Qdrant collection override |
| `QDRANT_EVAL_COLLECTION` | versioned eval default | Isolated collection used by the evaluation runner |
| `QDRANT_UPSERT_BATCH_SIZE` | `64` | Qdrant indexing batch size |
| `QDRANT_TIMEOUT_SECONDS` | `60` | Qdrant client request timeout |
| `CORS_ORIGINS` | `http://localhost:3000` | Comma-separated allowed origins |
| `SESSION_TTL_MINUTES` | `120` | Session lifetime |
| `REPO_CACHE_DIR` | `/tmp/codecompass-repos` | Writable temporary clone directory |
| `EMBEDDING_MODEL_ID` | `Qwen/Qwen3-Embedding-0.6B` | Local embedding model |
| `QWEN_EMBEDDING_BATCH_SIZE` | `8` | Embedding batch size |
| `RERANKER_MODEL_ID` | `Qwen/Qwen3-Reranker-0.6B` | Local reranking model |
| `RAG_RERANK_BATCH_SIZE` | `4` | Reranker batch size |
| `RAG_FINAL_SOURCE_LIMIT` | request `top_k` | Maximum evidence sources sent to the answer model |
| `RAG_SEARCH_MULTIPLIER` | `4` | Candidate depth for shallow questions |
| `RAG_DEEP_SEARCH_MULTIPLIER` | `8` | Candidate depth for implementation-heavy questions |

## Project structure

```text
code-compass/
├── server/
│   ├── server_app.py             # FastAPI routes and request models
│   ├── Dockerfile                # SageMaker-compatible production image
│   ├── entrypoint.sh             # Port 8080 / signal-safe startup
│   ├── evals/
│   │   ├── run_eval.py           # Evaluation runner and stage metrics
│   │   └── sample_eval_set.json  # 24-case benchmark
│   ├── src/
│   │   ├── code_parser.py        # tree-sitter and fallback chunking
│   │   ├── embeddings.py         # Qwen3 embeddings
│   │   ├── hybrid_search.py      # BM25, RRF, and reranking
│   │   ├── rag_system.py         # Indexing, retrieval, and generation orchestration
│   │   ├── repo_fetcher.py       # GitHub cloning and file filtering
│   │   └── vector_store.py       # Qdrant persistence and search
│   └── tests/
│       └── test_retrieval_quality.py
├── ui/
│   ├── api/[...path].js          # OIDC SageMaker invocation adapter
│   └── src/                      # React application
├── scripts/                      # Build, ECR push, and SageMaker deployment
├── .github/workflows/            # OIDC CI/CD workflow
└── images/                       # Portfolio screenshots
```

## Current limitations

- Only public GitHub repositories are supported.
- Repository metadata and active session state are held in memory.
- Large repositories take time to parse and embed.
- Running two local models still benefits substantially from CUDA acceleration.
- Model-based reranking improves quality but remains the largest retrieval-time cost.
- The system does not yet build a call graph or dependency graph.
- The benchmark is a focused regression suite; broader repository and language coverage is still needed.
- Two of the 24 current cases still retrieve relevant candidates but fail final ranking.

## Production engineering review

- **State and scaling:** repository/session metadata and BM25 indexes are
  process-local. The endpoint intentionally runs one worker and should start
  with one instance. Horizontal scaling can route follow-up requests to an
  instance without that state. Move session metadata and lexical indexes to a
  shared store before enabling autoscaling or multiple workers.
- **Long-running indexing:** indexing continues as an in-process background
  task after the initial invocation returns. A deployment or instance failure
  interrupts it. A durable queue and worker is the next production step, but
  is intentionally outside this portfolio deployment.
- **Request duration:** SageMaker real-time invocations have a 60-second
  response window. Keep query generation below that bound and use async
  inference or a job service if workloads grow.
- **Security and abuse:** the demo supports public GitHub URLs and bearer-like
  session IDs, not user authentication. Protect the Vercel function with rate
  limits and bot controls, validate allowed repository size, and add per-user
  quotas before opening it broadly. Never expose `QDRANT_API_KEY` or AWS
  credentials to the browser.
- **Network egress:** cloning public GitHub repositories, reaching Qdrant, and
  calling Bedrock all require egress. If the endpoint is placed in a VPC,
  provide NAT or appropriate endpoints and security-group rules. The baked
  model cache removes Hugging Face as a runtime dependency.
- **Supply chain:** model IDs and Python packages are configurable, while most
  Python dependencies are version-bounded rather than hash-locked. Pin model
  revisions, generate a hashed lock file/SBOM, sign ECR images, and enforce ECR
  scan findings for a higher-assurance deployment.
- **Frontend dependencies:** the current Create React App toolchain is legacy,
  and `npm audit` reports transitive findings through that dependency tree.
  Triage those findings and plan a focused migration to a maintained build
  tool rather than applying a breaking `npm audit fix --force` during this
  deployment change.
- **Observability:** SageMaker sends container output to CloudWatch. Add
  structured request IDs, latency/error metrics, alarms, and Qdrant/Bedrock
  dependency dashboards before treating the service as operationally mature.
- **Cost:** a real-time GPU endpoint accrues cost while `InService`, even when
  idle; Qdrant Cloud, Bedrock tokens, ECR storage, CloudWatch ingestion, NAT,
  and Vercel functions add usage charges. Delete the endpoint when the demo is
  not needed, keep old ECR/model versions under lifecycle policies, and set AWS
  Budgets alerts. Serverless inference is not a direct fit for this model size
  and startup profile.

## Why this project matters

Code Compass is more than a chat interface over embeddings. It demonstrates an end-to-end retrieval system with observable failure stages, reproducible evaluation, GPU-aware inference, grounded generation, session-scoped API design, and iterative quality improvements driven by evidence rather than prompt changes alone.
