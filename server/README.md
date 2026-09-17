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
- Code-aware retrieval using tree-sitter chunking, Qwen3 embeddings, BM25, rank fusion, and Qwen3 reranking
- Grounded answer generation with file-level citations
- Deployment-aware tradeoffs for cost, model choice, and free-tier infrastructure
- Evaluation workflow prepared for retrieval and answer-quality metrics

## Backend Responsibilities

- Clone a public GitHub repository into temporary storage
- Filter and chunk source files for retrieval
- Generate embeddings and store chunks in Chroma DB
- Maintain lightweight repository and session metadata in memory
- Run indexing as a background task
- Retrieve evidence with semantic search, lexical search, fusion, and reranking
- Generate answers from the selected context and return citations to the UI
- Delete cloned repository files after indexing

## Runtime Configuration

Copy `.env.example` to `.env`. Answer generation remains on Amazon Bedrock with
Qwen3 Coder Next. Retrieval always uses these local models:

- `QWEN_EMBEDDING_MODEL=Qwen/Qwen3-Embedding-4B`
- `QWEN_RERANKER_MODEL=Qwen/Qwen3-Reranker-4B`
- `QWEN_DEVICE_MAP=auto`
- `QWEN_COMPUTE_DTYPE=float16`

Both retrieval models use 4-bit NF4 quantization and require approximately 5–6 GB
of free accelerator memory when loaded together. They download from Hugging Face
on first startup. `EMBEDDING_BATCH_SIZE` and `RERANKER_BATCH_SIZE` default to 2 and
can be reduced if memory is constrained.

## Chroma Storage

The backend uses Chroma DB for vector storage in both local development and production. By default it stores the collection under `./data/chroma`, and you can point it somewhere else with `CHROMA_PATH`.

Configuration:

- `CHROMA_PATH=./data/chroma`
- `CHROMA_COLLECTION=repo_qa_chunks`
- `CHROMA_UPSERT_BATCH_SIZE=64`

Qwen3-Embedding-4B stores 2560-dimensional vectors. Its vector space is not
compatible with the previous Cohere embeddings, so existing Chroma data must be
deleted and every repository re-indexed after this upgrade. The normal application
startup rebuild performs this reset automatically.

## Metrics

The evaluation harness reports 4 core metrics:
- **Retrieval hit rate @ top-5**: Fraction of queries with at least one relevant source in top 5 results
- **Top-1 hit rate**: Fraction of queries where the first result is relevant
- **Grounded answer rate**: Fraction of answers that cite actual source code
- **Faithfulness (RAGAS)**: LLM-as-judge score for answer consistency with retrieved context
- **Query latency P95**: 95th percentile response time in milliseconds
