# Code Compass: Operations and Deployment

This document holds the operational detail intentionally kept out of the portfolio-focused root README: evaluation, API usage, runtime configuration, the SageMaker serving contract, AWS resources and IAM, deployment workflows, and production tradeoffs.

## Retrieval and conversation details

The backend shallow-clones public repositories, ignores generated and dependency directories, and keeps supported source, configuration, markup, and documentation files. Tree-sitter creates chunks around functions, classes, methods, declarations, and containers. Large classes receive overview chunks while their methods remain independently searchable. Each parsed source file also receives a module overview with its path, role, imports, exports, and symbols; plain-text and unsupported-language files use bounded fallback chunking. Clones are deleted after indexing.

Qdrant is both the vector store and persistent repository cache. A canonical repository-and-branch key lets a new session reuse an index. Re-indexing uploads a hidden generation, activates it only after every vector is stored, then removes the previous generation. A failed rebuild leaves the working cache intact.

Each question uses Qwen3 semantic retrieval, BM25, path and intent signals, and Reciprocal Rank Fusion. Controlled over-fetching precedes deduplication so repeated chunks and translated documentation cannot consume the candidate budget. Qwen3-Reranker-0.6B scores candidates with its native `yes`/`no` relevance format; final selection combines semantic, lexical, path, reranker, canonical-source, and diversity signals.

Before retrieval, the conversation layer routes social messages, rewrites contextual follow-ups into standalone repository questions, and asks for clarification when a reference has no reliable antecedent. A `conversation_trace` records the original query, standalone rewrite, retrieval query, route, and final Bedrock prompt. The answer prompt requires concrete files and symbols, evidence/inference separation, inline citations, and an explicit acknowledgement when evidence is insufficient. Citation numbers are validated before the response is returned.

Repository-overview answers use consistent **Purpose**, **Architecture**, **Technologies**, **Main components**, and **Request flow** sections.

## Evaluation

The suite contains 24 hand-written questions—eight each for Documenso, FastAPI, and Django—and covers repository purpose, implementation lookup, API behavior, configuration, tests, error handling, security, cross-file flows, and conversational follow-ups.

Reported metrics include candidate, semantic, lexical, fused, and path hit rates; reranker and prioritized Top-K hit rates; final-context retrieval hit rate; Top-1 hit rate; Mean Reciprocal Rank; expected-source grounded-answer rate; and optional LLM-judged faithfulness.

The optimized 0.6B run reached 91.67% final-context retrieval, 70.83% Top-1, 0.813 MRR, 100% candidate recall, and 99.17% LLM-judged faithfulness. All 24 answers were judged: 23 scored 1.0 and one scored 0.8. Because the same Bedrock model generates and judges answers, faithfulness is a regression signal rather than an independent evaluation.

| Repository | Final-context retrieval | Top-1 | MRR |
|---|---:|---:|---:|
| Documenso | 87.5% | 75.0% | 0.813 |
| FastAPI | 87.5% | 62.5% | 0.750 |
| Django | 100% | 75.0% | 0.875 |

The expected-source grounded-answer rate moved from 58.33% to 91.67%, matching the final-context improvement.

The two remaining misses are Documenso's signing-package implementation and FastAPI's OpenAPI generation. In both cases, the expected source enters the candidate set but falls outside the final eight sources, pointing to final retrieval prioritization rather than answer-model fine-tuning.

The changes responsible for the measured improvement were:

- Qwen3 last-token pooling instead of incorrect mean pooling;
- task instructions on query embeddings while document embeddings remain unprefixed;
- the native reranker prompt and relevance-token scoring;
- GPU-batched reranking and controlled candidate over-fetching;
- deduplication of localized documentation and repeated chunks;
- lower influence for noisy path-only matches;
- stronger cross-file question classification and test-source penalties;
- removal of an early top-eight truncation;
- correct Unicode byte offsets in tree-sitter extraction;
- stage-level metrics for candidate generation, fusion, reranking, and final selection.

Run the complete evaluation from the repository root:

```bash
CODEBASE_RAG_REINDEX=1 \
CODEBASE_RAG_EVAL_OUTPUT=server/evals/results/full_eval.json \
python server/evals/run_eval.py
```

Run one repository while iterating:

```bash
CODEBASE_RAG_EVAL_REPOS=fastapi \
CODEBASE_RAG_REINDEX=1 \
CODEBASE_RAG_ENABLE_FAITHFULNESS=0 \
CODEBASE_RAG_EVAL_OUTPUT=server/evals/results/fastapi.json \
python server/evals/run_eval.py
```

Valid repository IDs are `documenso`, `fastapi`, and `django`. Cases live in [`server/evals/sample_eval_set.json`](../server/evals/sample_eval_set.json), and regression tests live in [`server/tests/test_retrieval_quality.py`](../server/tests/test_retrieval_quality.py).

## API

All repository operations are scoped by an `X-Session-Id` header.

| Method | Endpoint | Purpose |
|---|---|---|
| `POST` | `/api/repos/index` | Reuse a cached index or asynchronously build one |
| `GET` | `/api/repos` | List repositories for the current session |
| `GET` | `/api/repos/{repo_id}` | Read indexing status and metadata |
| `POST` | `/api/query` | Ask a grounded question about an indexed repository |
| `POST` | `/api/session/end` | Clear session state; persistent vectors remain cached |

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

Set `reindex` to `true` to build and atomically activate a replacement generation.

Query the completed index:

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

## Runtime configuration

| Variable | Default | Purpose |
|---|---|---|
| `AWS_REGION` | `us-east-1` fallback | Amazon Bedrock region |
| `APP_ENV` | `local` | Runtime environment label |
| `LOG_LEVEL` | `INFO` | Application/Uvicorn log level |
| `ENABLE_PROFILING` | `false` | Parsing, embedding-batch, GPU, and Qdrant timing logs |
| `PORT` | `8080` | Direct container/listener port |
| `BEDROCK_MODEL_ID` | `qwen.qwen3-coder-next` | Bedrock model or inference profile |
| `QDRANT_URL` | required | Qdrant REST endpoint |
| `QDRANT_API_KEY` | none | Cloud key; omit only for an unsecured local instance |
| `QDRANT_API_KEY_SECRET_ARN` | none | Preferred AWS key source when direct key is absent |
| `QDRANT_COLLECTION` | versioned default | Application collection override |
| `QDRANT_EVAL_COLLECTION` | versioned eval default | Isolated evaluation collection |
| `QDRANT_UPSERT_BATCH_SIZE` | `64` | Indexing batch size |
| `QDRANT_TIMEOUT_SECONDS` | `60` | Client request timeout |
| `CORS_ORIGINS` | `http://localhost:3000` | Comma-separated allowed origins |
| `SESSION_TTL_MINUTES` | `120` | Session lifetime |
| `REPO_CACHE_DIR` | `/opt/ml/codecompass/repos` | Writable clone directory |
| `EMBEDDING_MODEL_ID` | `Qwen/Qwen3-Embedding-0.6B` | Local embedding model |
| `QWEN_EMBEDDING_BATCH_SIZE` | `8` | Embedding batch size |
| `RERANKER_MODEL_ID` | `Qwen/Qwen3-Reranker-0.6B` | Local reranking model |
| `RAG_RERANK_BATCH_SIZE` | `4` | Reranker batch size |
| `RAG_FINAL_SOURCE_LIMIT` | request `top_k` | Maximum sources sent to generation |
| `RAG_SEARCH_MULTIPLIER` | `4` | Candidate depth for shallow questions |
| `RAG_DEEP_SEARCH_MULTIPLIER` | `8` | Candidate depth for implementation questions |

## SageMaker serving architecture

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
  └── CloudWatch Logs ── container and endpoint logs

GitHub Actions (OIDC) ── build/test ──► ECR ──► SageMaker deployment
```

SageMaker endpoints are AWS APIs, not public general-purpose web servers. `ui/api/[...path].js` maps browser REST calls to `/invocations`, uses Vercel OIDC, and never sends AWS credentials to the browser. Direct `/api/*` routes remain available locally.

The container listens on `0.0.0.0:8080`, responds to `/ping`, accepts `/invocations`, handles SageMaker's `serve` argument, and exits cleanly on `SIGTERM`. It reports healthy only after both local models and Qdrant are ready.

## Required AWS resources

- An ECR repository; `scripts/push.sh` creates it with scan-on-push if missing.
- A SageMaker execution role trusted by `sagemaker.amazonaws.com`.
- Real-time endpoint quota for the selected GPU instance. Scripts default to one `ml.g5.xlarge`; confirm regional memory and quota.
- Bedrock access to `BEDROCK_MODEL_ID` in the same region.
- A reachable Qdrant Cloud cluster and API key. The workflow reads `QDRANT_API_KEY` from a GitHub environment secret; `QDRANT_API_KEY_SECRET_ARN` offers stronger handling.
- GitHub and Vercel OIDC identity providers with narrowly scoped roles.

### SageMaker execution role

The runtime role needs:

- ECR pull: `ecr:GetAuthorizationToken`, `ecr:BatchCheckLayerAvailability`, `ecr:GetDownloadUrlForLayer`, and `ecr:BatchGetImage`;
- logging: `logs:CreateLogGroup`, `logs:CreateLogStream`, `logs:PutLogEvents`, `logs:DescribeLogStreams`, and `cloudwatch:PutMetricData`;
- generation: `bedrock:InvokeModel` on the chosen model or inference-profile ARN;
- optional secret loading: `secretsmanager:GetSecretValue` on the Qdrant secret ARN.

### GitHub deployment role

The CI role needs the ECR push actions `ecr:GetAuthorizationToken`, `ecr:CreateRepository`, `ecr:DescribeRepositories`, `ecr:BatchCheckLayerAvailability`, `ecr:InitiateLayerUpload`, `ecr:UploadLayerPart`, `ecr:CompleteLayerUpload`, and `ecr:PutImage`; SageMaker create, describe, update, and delete permissions for models, endpoint configurations, and the endpoint; and `iam:PassRole` restricted to the execution role with `iam:PassedToService = sagemaker.amazonaws.com`.

Restrict the GitHub OIDC `sub` claim to this repository, `main`, and the protected production environment.

### Vercel runtime role

The role needs only `sagemaker:InvokeEndpoint` on this endpoint. Restrict its OIDC trust policy to the production Vercel project and environment.

## Deploy from a workstation

Prerequisites are Docker, AWS CLI v2, `jq`, `openssl`, and AWS credentials authorized for ECR and SageMaker.

```bash
export AWS_REGION=us-east-1
export ECR_REPOSITORY=code-compass-backend
export SAGEMAKER_ENDPOINT_NAME=code-compass
export SAGEMAKER_EXECUTION_ROLE_ARN=arn:aws:iam::123456789012:role/code-compass-sagemaker
export SAGEMAKER_INSTANCE_TYPE=ml.g5.xlarge
export QDRANT_URL=https://your-cluster.us-east.aws.cloud.qdrant.io:6333
export QDRANT_API_KEY=your-qdrant-api-key
export CORS_ORIGINS=https://your-project.vercel.app

./scripts/test-local-sagemaker.sh
./scripts/push.sh
./scripts/deploy.sh
```

`test-local-sagemaker.sh` builds the production image, runs it as the non-root production user on port 8080 with SageMaker-style writable mounts, polls the real `GET /ping` status until it receives `200`, then checks `/api/health` and `/invocations`. It prints logs and fails on startup errors.

`build.sh` targets Linux/amd64 and preloads both Hugging Face snapshots. `PRELOAD_MODELS=0` produces a faster development build that needs outbound Hugging Face access at startup and is not recommended for production. `push.sh` authenticates, tags, and pushes to ECR. `deploy.sh` hashes the image and environment into immutable model/config names, updates the stable endpoint only when necessary, waits for `InService`, and recreates a `Failed` endpoint.

The Dockerfile deliberately uses the official PyTorch CUDA runtime as a single stage. A Python builder stage would retain another copy of several gigabytes of CUDA/PyTorch wheels without shrinking the runtime layer. Baking both 0.6B models trades ECR storage and build time for predictable, network-independent startup. Runtime code is copied last for layer caching, caches are removed, and the service runs as a non-root user with only Git retained for cloning.

Old models and endpoint configurations are retained for rollback; delete unused versions periodically after validating a deployment.

## GitHub Actions CI/CD

`.github/workflows/deploy-sagemaker.yml` runs on backend or deployment changes to `main`. It installs dependencies, runs tests, assumes the deployment role, validates the local SageMaker contract, pushes the verified image, and updates the endpoint. It requests `id-token: write` and uses GitHub OIDC; no AWS access keys are stored.

Create a protected GitHub environment named `production` with these secrets:

| Secret | Purpose |
|---|---|
| `AWS_GITHUB_ROLE_ARN` | OIDC deployment role |
| `AWS_REGION` | ECR, SageMaker, Secrets Manager, and Bedrock region |
| `ECR_REPOSITORY` | Backend repository name |
| `SAGEMAKER_ENDPOINT_NAME` | Stable endpoint name |
| `SAGEMAKER_EXECUTION_ROLE_ARN` | Runtime role passed to SageMaker |
| `SAGEMAKER_INSTANCE_TYPE` | Normally `ml.g5.xlarge` |
| `QDRANT_URL` | Qdrant Cloud HTTPS endpoint |
| `QDRANT_API_KEY` | Qdrant key used by the workflow |
| `CORS_ORIGINS` | Production Vercel origin |
| `BEDROCK_MODEL_ID` | Bedrock model or inference profile |

All values use GitHub's `secrets` context. Environment protection and required reviewers are recommended.

The direct Qdrant key is convenient for a portfolio environment, but places it in the SageMaker model's container environment. AWS advises against sensitive values in `CreateModel` environment fields. For shared or long-lived deployments, set `QDRANT_API_KEY_SECRET_ARN`; the container retrieves it on startup and the execution role must allow `secretsmanager:GetSecretValue` on that ARN.

For Vercel, set `AWS_ROLE_ARN`, `SAGEMAKER_AWS_REGION`, and `SAGEMAKER_ENDPOINT_NAME`. Enable OIDC and configure the trust policy. Leave `REACT_APP_API_URL` unset in production so the UI uses the same-origin adapter; set it to `http://localhost:8000` locally.

## Repository layout

```text
code-compass/
├── server/
│   ├── server_app.py             # FastAPI routes and request models
│   ├── Dockerfile                # SageMaker-compatible image
│   ├── entrypoint.sh             # Port 8080 and signal-safe startup
│   ├── evals/                    # Evaluation runner and 24-case suite
│   ├── src/                      # Parsing, retrieval, storage, orchestration
│   └── tests/                    # Retrieval and deployment regressions
├── ui/
│   ├── api/[...path].js          # OIDC SageMaker invocation adapter
│   └── src/                      # React application
├── scripts/                      # Build, test, push, and deployment scripts
├── .github/workflows/            # OIDC CI/CD workflow
└── images/                       # Portfolio assets
```

## Production engineering review

- **State and scaling:** repository/session metadata and BM25 indexes are process-local. The endpoint deliberately uses one worker and should begin with one instance. Move state to a shared store before multiple workers or autoscaling.
- **Long-running indexing:** indexing is an in-process background task after the first invocation returns. Deployment or instance failure interrupts it; a durable queue and worker is the next step.
- **Request duration:** real-time invocations have a 60-second response window. Keep generation within it or use async inference/a job service.
- **Security and abuse:** public URLs and bearer-like session IDs are not authentication. Add rate limits, bot controls, repository-size validation, and per-user quotas before broad public access. Never expose Qdrant or AWS credentials to the browser.
- **Network egress:** GitHub cloning, Qdrant, and Bedrock require egress. A VPC deployment needs NAT or suitable endpoints and security-group rules. Baked models remove Hugging Face as a runtime dependency.
- **Supply chain:** most Python dependencies are version-bounded rather than hash-locked. Pin model revisions, generate a hashed lock file and SBOM, sign images, and enforce ECR scan findings for higher assurance.
- **Frontend dependencies:** Create React App is legacy, and `npm audit` reports transitive findings in that tree. Plan a focused migration instead of applying a breaking `npm audit fix --force` during deployment work.
- **Observability:** CloudWatch receives container output. Add latency/error metrics, alarms, and Qdrant/Bedrock dashboards before treating the service as operationally mature.
- **Cost:** a GPU endpoint accrues charges while `InService`; Qdrant, Bedrock, ECR, CloudWatch, NAT, and Vercel add costs. Delete idle endpoints, apply lifecycle policies, and configure AWS Budgets alerts. Serverless inference is not a direct fit for this model size and startup profile.

## Current limitations

- Only public GitHub repositories are supported.
- Repository metadata, active sessions, and lexical indexes are process-local.
- Large repositories take time to parse and embed.
- Both local models benefit substantially from CUDA acceleration.
- Model-based reranking is the largest retrieval-time cost.
- The system does not build a call graph or dependency graph.
- The regression suite needs broader repository and language coverage.
- Two of 24 cases retrieve the expected candidate but miss it in final ranking.
