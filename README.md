# Code Compass

An end-to-end repository question answering system that indexes a public GitHub codebase, retrieves grounded code evidence, and generates cited answers through a retrieval-augmented generation pipeline.

This project includes:
- a React frontend for repository submission and conversational querying
- a FastAPI backend for indexing and answering questions
- a hybrid retrieval pipeline with semantic search, BM25, and reranking
- an evaluation harness for measuring retrieval quality and answer grounding

## What The System Does

1. A user pastes a GitHub repository URL into the UI.
2. The backend clones the repository into a temporary local directory.
3. Source files are filtered and chunked using tree-sitter and fallback text chunking.
4. The system generates embeddings for chunks and stores them in a Chroma-backed vector layer.
5. At query time, the system retrieves evidence with:
   - semantic vector search
   - lexical BM25 search
   - reciprocal rank fusion
   - cross-encoder reranking
6. The top grounded chunks are passed to the LLM to generate a concise answer.
7. The UI displays the answer with file-level citations and GitHub source links.

## App Screens


![Code Compass landing screen](images/landing.png)


![Code Compass chat and citations screen](images/chat.png)

## Architecture

```text
┌──────────────────────┐
│      React UI        │
│  repo submit + chat  │
│  citations + status  │
└──────────┬───────────┘
           │ HTTP / JSON
           ▼
┌──────────────────────┐
│    FastAPI Server    │
│   routes + session   │
│      validation      │
└──────────┬───────────┘
           │
           ▼
┌──────────────────────────────────────────────┐
│              CodebaseRAGSystem               │
│ indexing orchestration + query orchestration │
└───────┬───────────────┬───────────────┬──────┘
        │               │               │
        │               │               │
        ▼               ▼               ▼
┌──────────────┐  ┌──────────────┐  ┌──────────────┐
│ RepoFetcher  │  │ CodeParser   │  │ Embeddings   │
│ clone/filter │  │ tree-sitter  │  │ Bedrock/local│
└──────┬───────┘  │ fallback     │  └──────┬───────┘
       │          └──────┬───────┘         │
       │                 │                 │
       └────────────┬────┴────────────┬────┘
                    ▼                 ▼
           ┌──────────────┐   ┌──────────────┐
           │ In-memory    │   │ Chroma       │
           │ repo/session │   │ vector store │
           │ state        │   └──────┬───────┘
           └──────────────┘          │
                                     ▼
                           ┌──────────────────┐
                           │ Hybrid Retrieval │
                           │ semantic + BM25  │
                           │ + reranking      │
                           └────────┬─────────┘
                                    ▼
                           ┌──────────────────┐
                           │   LLM Answerer   │
                           │ grounded answer  │
                           │ + citations      │
                           └──────────────────┘
```

### Frontend

- React 19
- Tailwind CSS
- Axios for API communication

Responsibilities:
- collect the GitHub repository URL
- poll indexing state
- send chat questions and prior conversation turns
- render markdown-like answers
- display cited files, symbols, and line ranges

Main entry points:
- [`ui/src/App.js`](/Users/sivasankernp/Desktop/code-compass/ui/src/App.js)
- [`ui/src/config.js`](/Users/sivasankernp/Desktop/code-compass/ui/src/config.js)

### Backend

- FastAPI
- Pydantic
- in-memory session and repository state

Responsibilities:
- validate requests
- manage session-scoped repository state
- run indexing in the background
- execute retrieval and answer generation
- return grounded answers and source metadata

Main entry points:
- [`server/server_app.py`](/Users/sivasankernp/Desktop/code-compass/server/server_app.py)
- [`server/src/rag_system.py`](/Users/sivasankernp/Desktop/code-compass/server/src/rag_system.py)

### Retrieval Pipeline

- tree-sitter for code-aware chunking
- Amazon Bedrock or local embeddings for semantic retrieval depending on environment
- BM25 for lexical retrieval
- reciprocal rank fusion to combine retrieval channels
- a cross-encoder reranker for final source ordering
- Groq or Amazon Bedrock generation depending on environment configuration

Core modules:
- [`server/src/code_parser.py`](/Users/sivasankernp/Desktop/code-compass/server/src/code_parser.py)
- [`server/src/embeddings.py`](/Users/sivasankernp/Desktop/code-compass/server/src/embeddings.py)
- [`server/src/hybrid_search.py`](/Users/sivasankernp/Desktop/code-compass/server/src/hybrid_search.py)
- [`server/src/vector_store.py`](/Users/sivasankernp/Desktop/code-compass/server/src/vector_store.py)
- [`server/src/repo_fetcher.py`](/Users/sivasankernp/Desktop/code-compass/server/src/repo_fetcher.py)

## Data Flow

### Indexing Flow

1. `POST /api/repos/index`
2. Backend registers the repo against a session
3. Background task clones the repo
4. Files are filtered by extension, directory, and size
5. Files are chunked into code-aware segments
6. Embeddings are generated for each chunk
7. Chunks are stored in the vector layer and in in-memory retrieval state
8. Metadata and progress are exposed back to the UI

### Query Flow

1. `POST /api/query`
2. The backend validates the session and repository status
3. The question is expanded using lightweight intent heuristics
4. Semantic search retrieves candidate chunks
5. BM25 retrieves lexical matches
6. Results are fused and reranked
7. Final sources are selected and passed to the LLM
8. The backend returns:
   - `answer`
   - `confidence`
   - `sources`
   - repository metadata

## Tech Stack Decisions

### Why FastAPI

- fast iteration speed
- strong request validation through Pydantic
- simple background task support
- clean fit for JSON APIs and model-driven backend code

### Why React

- straightforward stateful UI for a single-page workflow
- easy integration with polling, chat state, and citation rendering
- strong ecosystem for incremental iteration

### Why tree-sitter

- better chunk boundaries than naive fixed-length splitting
- lets the system reason around functions, classes, and symbols
- improves retrieval quality for implementation-focused questions

### Why Hybrid Retrieval

Pure semantic search misses exact symbols and file names. Pure lexical search misses semantic intent. This project combines both because code questions often need:
- exact identifiers
- nearby implementation detail
- cross-file semantic similarity

### Why Chroma

- simple vector abstraction
- one vector database path for both local and production runtime
- persistent local storage without a separate hosted vector service
- direct support for externally generated embeddings and metadata filters

### Why In-Memory Session State

- repository/session metadata is short-lived and cleared when the backend restarts
- no separate relational database is needed for the current product flow
- the API still exposes indexing status and session-scoped repositories while keeping deployment simpler

## Runtime Environments

### Local Development

Local development is configured for higher-quality experimentation:
- Claude 3.5 Sonnet on Amazon Bedrock for answer generation
- Cohere Embed v3 on Amazon Bedrock for semantic retrieval (smaller, faster than v4)

This setup is useful for:
- higher quality local experiments
- comparing retrieval and answer quality in a managed-model environment

Recommended local runtime:
```bash
export LLM_PROVIDER=bedrock
export EMBEDDING_PROVIDER=bedrock
export AWS_REGION=us-east-1
export BEDROCK_LLM_MODEL=anthropic.claude-3-5-sonnet-20240620-v1:0
export BEDROCK_EMBEDDING_MODEL=cohere.embed-v3:0
export CHROMA_PATH=./data/chroma
```

### Production Deployment

The production deployment target is:
- frontend on Vercel
- backend on Hugging Face Spaces

Production inference uses lower-cost models:
- Groq-hosted Llama 3.1 70B for answer generation (fast inference)
- Local sentence-transformers/all-MiniLM-L6-v2 embeddings (~80MB)
- Chroma DB for vector storage

This production setup fits Hugging Face Spaces free-tier constraints while keeping the retrieval and answer pipeline intact.

Recommended production runtime:
```bash
export LLM_PROVIDER=groq
export EMBEDDING_PROVIDER=local
export GROQ_API_KEY=<your-groq-api-key>
export CHROMA_PATH=./data/chroma
```

## Deployment

### Production Topology

- Vercel hosts the React frontend
- Hugging Face Spaces hosts the FastAPI backend
- the backend is packaged and deployed as a Docker Space
- GitHub Actions syncs the backend code to the Space on pushes to `main`

### Docker

The backend is deployed with Docker using:
- [`server/Dockerfile`](/Users/sivasankernp/Desktop/code-compass/server/Dockerfile)

The container:
- installs Python dependencies
- copies the backend application
- starts the FastAPI app with Uvicorn on port `7860`

### CI/CD

Continuous deployment is handled through:
- [`.github/workflows/deploy-hf-space.yml`](/Users/sivasankernp/Desktop/code-compass/.github/workflows/deploy-hf-space.yml)

The workflow:
- runs on pushes to `main`
- syncs the `server/` directory to the Hugging Face Space
- triggers the Docker Space rebuild automatically


## Evaluation And Benchmarking

The project includes an end-to-end eval harness that calls the live API instead of mocking the retrieval pipeline.

Files:
- [`server/evals/run_eval.py`](/Users/sivasankernp/Desktop/code-compass/server/evals/run_eval.py)
- [`server/evals/sample_eval_set.json`](/Users/sivasankernp/Desktop/code-compass/server/evals/sample_eval_set.json)

We track only 4 metrics:
- **Hit rate @ top-5**: Does the system retrieve relevant code at all? (retrieval quality)
- **Grounded answer rate**: Do answers cite actual source code? (answer trustworthiness)
- **Faithfulness (LLM judge)**: Are answers factually consistent with the retrieved context? (no hallucinations)
- **Query latency P95**: Is the system responsive enough for interactive use? (user experience)

### Benchmark Snapshot

Current sample benchmark target:
- Documenso (`https://github.com/documenso/documenso.git`)
- 43 evaluation cases
- 10 categories
- Full-application coverage across architecture, docs, setup, API layers, document flows, signing, email, jobs, tests, and follow-up questions

| Metric | Result |
| --- | ---: |
| Retrieval hit rate @ top-5 | 86% |
| Top-1 hit rate | 72% |
| Grounded answer rate | 81% |
| Faithfulness (Claude 3.5 Sonnet) | 0.92 |
| Query latency P95 | 3,200ms |

What these numbers mean:
- ~86% of queries find at least one relevant code chunk in the top 5 results
- ~72% of queries have the most relevant source ranked first
- ~81% of answers are grounded in actual code evidence from the repository
- LLM judge indicates answers are highly faithful to the retrieved context
- P95 latency is acceptable for interactive use but could be optimized

Benchmark strengths:
- full-stack application benchmark rather than a library-only benchmark
- product-domain questions around documents, recipients, fields, signing, emails, jobs, and webhooks
- measurable end-to-end performance instead of anecdotal examples

Benchmark limitations:
- sample set focused on one target project; use for this repo's quality, not cross-repo generalization
- Documenso is a large TypeScript monorepo; some cross-file questions may be harder

## Project Strengths

- full-stack architecture with a clear data flow
- code-aware retrieval rather than plain document retrieval
- practical hybrid search design
- session-aware repo isolation
- source-grounded answer generation
- explicit benchmark and evaluation workflow

## Known Tradeoffs

- retrieval state is intentionally session-scoped and mostly in memory
- cloned repositories are temporary and deleted after indexing
- repository metadata is lightweight and persisted separately from vector state
- if the backend restarts, repositories must be re-indexed

## Local Setup

### Backend

```bash
cd server
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
export LLM_PROVIDER=bedrock
export EMBEDDING_PROVIDER=bedrock
export AWS_REGION=us-east-1
export BEDROCK_LLM_MODEL=anthropic.claude-3-5-sonnet-20240620-v1:0
export BEDROCK_EMBEDDING_MODEL=cohere.embed-v3:0
export CHROMA_PATH=./data/chroma
python server_app.py
```

Backend runs on `http://localhost:8000`

### Frontend

```bash
cd ui
npm install
npm start
```

Frontend runs on `http://localhost:3000`

Create `ui/.env`:

```bash
REACT_APP_API_URL=http://localhost:8000
```

## Running The Eval Harness

From the `server` directory:

```bash
CODEBASE_RAG_API_URL=http://localhost:8000 \
CODEBASE_RAG_SESSION_ID=<session-id> \
CODEBASE_RAG_REPO_ID=<repo-id> \
CODEBASE_RAG_EVAL_OUTPUT=evals/latest_eval_report.json \
python evals/run_eval.py
```

The output report includes:
- eval-set audit warnings
- headline metrics
- category breakdowns
- case-by-case detail

If you want to save the latest run as a JSON artifact:

```bash
CODEBASE_RAG_EVAL_OUTPUT=evals/latest_eval_report.json
```

## Repository Structure

```text
server/
  server_app.py
  evals/
  src/
ui/
  src/
README.md
```
