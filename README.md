# Code Compass

Code Compass lets you paste in a public GitHub repository and ask questions about the codebase in plain English.

It clones and indexes the repository, retrieves the most relevant parts of the code for each question, and generates an answer grounded in those sources. Answers include clickable citations back to the files and line ranges used.

I built this project to explore RAG on real codebases instead of documents. Code retrieval has different challenges from normal document search. Exact symbols and filenames matter, implementations are often spread across multiple files, and retrieving code that is only loosely related to the question can lead to a convincing but incorrect answer.

## Screenshots

![Code Compass landing screen](images/landing.png)

![Code Compass chat and citations screen](images/chat.png)

## How it works

1. The user submits a public GitHub repository.
2. The backend clones the repository and filters supported files.
3. Source code is parsed with tree-sitter and split around functions, classes, symbols, and module boundaries. Unsupported files fall back to plain-text chunking.
4. Chunks are embedded and stored in Chroma.
5. Each question runs through semantic vector search and BM25 lexical search.
6. Results are combined using Reciprocal Rank Fusion.
7. Qwen3-Reranker-4B reranks the strongest candidates.
8. The final context is sent to the LLM.
9. The LLM generates an answer with inline citations such as `[1]` and `[2]`.
10. Citations are validated against the retrieved sources before the response is returned.

## Architecture

```text
┌──────────────────────┐
│      React UI        │
│  repo submit + chat  │
└──────────┬───────────┘
           │
           ▼
┌──────────────────────┐
│    FastAPI Server    │
│   API + sessions     │
└──────────┬───────────┘
           │
           ▼
┌──────────────────────────────┐
│      CodebaseRAGSystem       │
│ indexing + query pipeline    │
└───────┬──────────────┬───────┘
        │              │
        ▼              ▼
┌──────────────┐ ┌──────────────┐
│ RepoFetcher  │ │  CodeParser  │
│ clone/filter │ │  tree-sitter │
└──────┬───────┘ └──────┬───────┘
       │                 │
       └────────┬────────┘
                ▼
       ┌──────────────────┐
       │ Hybrid Retrieval │
       │ vectors + BM25   │
       │ RRF + reranking  │
       └────────┬─────────┘
                ▼
       ┌──────────────────┐
       │   LLM Answerer   │
       │ answer + sources │
       └──────────────────┘
```

## Retrieval

The retrieval pipeline combines semantic and lexical search because code questions often contain a mix of natural language and exact identifiers.

### Code-aware chunking

tree-sitter splits supported source files around functions, classes, methods, and declarations instead of arbitrary fixed-length chunks.

Each source file also gets a lightweight module overview containing information such as its path, imports, exports, and symbols. This helps with architectural questions and files that mainly connect other parts of the codebase.

### Hybrid search

Semantic vector search and BM25 run in parallel.

Semantic search helps with questions such as:

```text
How does an incoming request reach a view?
```

BM25 is useful when the question contains exact code terminology such as:

```text
QuerySet
APIRouter
OAuth2AuthorizationCodeBearer
```

The two rankings are combined using Reciprocal Rank Fusion.

### Reranking

The strongest candidates are passed through Qwen3-Reranker-4B before the final context is selected.

```text
Question
   │
   ├── Semantic search
   │
   └── BM25
          │
          ▼
         RRF
          │
          ▼
   Cross-encoder
          │
          ▼
    Final context
          │
          ▼
         LLM
```

## Tech stack

**Frontend**

* React
* Tailwind CSS
* Axios
* Vercel

**Backend**

* Python
* FastAPI
* Pydantic
* Hugging Face Spaces

**RAG**

* tree-sitter
* Chroma
* BM25
* Reciprocal Rank Fusion
* Qwen3-Reranker-4B local reranking
* Qwen3-Embedding-4B local embeddings
* Qwen3-Coder-Next through Amazon Bedrock for answer generation

## Evaluation

I added an evaluation harness so retrieval changes can be tested without manually running the frontend and repeating the same questions.

```bash
python evals/run_eval.py
```

The benchmark contains **24 hand-written questions across three repositories**:

* [Documenso](https://github.com/documenso/documenso.git), a large TypeScript monorepo
* [FastAPI](https://github.com/fastapi/fastapi.git), a Python API framework
* [Django](https://github.com/django/django.git), a large Python web framework

There are eight questions per repository covering architecture, implementation lookup, cross-file flows, APIs, configuration, tests, security and error handling, and conversational follow-ups.

The evaluation checks whether expected sources are retrieved, how highly the first expected source is ranked, and whether the generated answer is supported by the retrieved context.

The evaluation setup uses Qwen3-Coder-Next through Amazon Bedrock, Qwen3-Embedding-4B, and Qwen3-Reranker-4B.

### Results

| Metric                 |    Result |
| ---------------------- | --------: |
| Retrieval Hit Rate @ 5 | **83.3%** |
| Top-1 Hit Rate         | **62.5%** |
| Mean Reciprocal Rank   |  **0.72** |
| Faithfulness           |  **0.92** |

The retrieval results show that relevant source files are usually present in the final context, while the difference between Top-5 and Top-1 reflects cases where a related documentation file, test, helper, or consumer ranks above the main implementation.

Faithfulness measures whether claims in the generated answer are supported by the retrieved context. It is evaluated separately from retrieval so that a relevant retrieval result does not automatically count as a grounded answer.

The full evaluation set is available at:

```text
server/evals/sample_eval_set.json
```

## Running locally

### Backend

```bash
cd server

python -m venv .venv
source .venv/bin/activate

pip install -r requirements.txt

export AWS_REGION=us-east-1

export CHROMA_PATH=./data/chroma

python server_app.py
```

The API runs at `http://localhost:8000`.

### Frontend

```bash
cd ui
npm install
npm start
```

Create `ui/.env`:

```bash
REACT_APP_API_URL=http://localhost:8000
```

The frontend runs at `http://localhost:3000`.

## Deployment

The frontend is deployed on Vercel.

The FastAPI backend runs as a Docker Space on Hugging Face Spaces and is deployed through GitHub Actions.

```bash
export AWS_REGION=us-east-1
export CHROMA_PATH=./data/chroma
```

## Tradeoffs

* Repository and session state is mostly kept in memory, so backend restarts require re-indexing.
* Cloned repositories are deleted after indexing.
* Large repositories can take time to index.
* Hybrid retrieval and Qwen reranking improve retrieval quality but add latency.
* Related documentation, tests, or helper code can still rank above the canonical implementation.
* Retrieval works on chunks independently and does not currently use a dependency or call graph.

## Project structure

```text
server/
  server_app.py
  evals/
    run_eval.py
    sample_eval_set.json
  src/
    code_parser.py
    embeddings.py
    hybrid_search.py
    rag_system.py
    repo_fetcher.py
    vector_store.py

ui/
  src/

README.md
```
