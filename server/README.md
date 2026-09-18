---
title: Code Compass API
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 7860
---

# Code Compass Backend

FastAPI backend for Code Compass, a personal full-stack RAG project that indexes public GitHub repositories and answers questions with grounded source citations.

## What This Demonstrates

- End-to-end AI application design, not just a prompt wrapper
- Backend API design with FastAPI, Pydantic validation, and session-scoped state
- Code-aware retrieval using tree-sitter chunking, vector search, BM25, rank fusion, and reranking
- Grounded answer generation with file-level citations
- Deployment-aware tradeoffs for cost, model choice, and free-tier infrastructure
- Evaluation workflow prepared for retrieval and answer-quality metrics

## Backend Responsibilities

- Clone a public GitHub repository into temporary storage
- Filter and chunk source files for retrieval
- Generate embeddings and persist reusable repository indexes in Qdrant
- Maintain lightweight repository and session metadata in memory
- Run indexing as a background task
- Retrieve evidence with semantic search, lexical search, fusion, and reranking
- Generate answers from the selected context and return citations to the UI
- Delete cloned repository files after indexing while retaining the Qdrant cache

## Runtime Configuration

### Model Stack
- LLM: Amazon Bedrock with Qwen3-Coder-Next
- Embeddings: local Qwen3-Embedding-0.6B
- Reranker: local Qwen3-Reranker-0.6B
- Required AWS configuration: `AWS_REGION` or `AWS_DEFAULT_REGION`

## Qdrant Storage

The backend uses Qdrant for vector storage and persistent repository caching. It creates a cosine-similarity collection for the 1,024-dimensional Qwen3-Embedding-0.6B vectors and indexes `repository_key`, `cache_generation`, and `cache_ready` for filtered search and safe replacement. Repositories created with an older vector-store backend must be indexed once into the new collection.

`POST /api/repos/index` accepts `{"github_url": "...", "reindex": false}`. With the default `false` value, a ready cache is loaded immediately. With `true`, the backend uploads a hidden replacement generation, activates it only after all chunks are stored, and then deletes older generations. Session expiry removes only in-memory BM25 and repository state; it does not delete persistent Qdrant vectors.

Configuration:

- `QDRANT_URL=https://your-cluster.us-east.aws.cloud.qdrant.io:6333`
- `QDRANT_API_KEY=your-qdrant-api-key`
- `QDRANT_COLLECTION=code_compass_qwen3_embedding_0_6b_last_token_cache_v2`
- `QDRANT_EVAL_COLLECTION=code_compass_eval_qwen3_embedding_0_6b_last_token_cache_v2`
- `QDRANT_UPSERT_BATCH_SIZE=64`
- `QDRANT_TIMEOUT_SECONDS=60`

The evaluation runner always uses `QDRANT_EVAL_COLLECTION`, preventing a
reindexing evaluation from deleting the application's collection.

## Metrics

The evaluation harness reports 4 core metrics:
- **Retrieval hit rate @ top-5**: Fraction of queries with at least one relevant source in top 5 results
- **Top-1 hit rate**: Fraction of queries where the first result is relevant
- **Grounded answer rate**: Fraction of answers that cite actual source code
- **LLM-judged faithfulness**: Answer consistency with retrieved context
