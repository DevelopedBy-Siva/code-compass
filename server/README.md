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
- Generate embeddings and store chunks in Qdrant
- Maintain lightweight repository and session metadata in SQLite
- Run indexing as a background task
- Retrieve evidence with semantic search, lexical search, fusion, and reranking
- Generate answers from the selected context and return citations to the UI
- Delete cloned repository files after indexing

## Runtime Configuration

Local development is configured for higher-quality experimentation:

- `LLM_PROVIDER=bedrock`
- `EMBEDDING_PROVIDER=bedrock`
- Claude on Amazon Bedrock for answer generation
- Cohere Embed on Amazon Bedrock for semantic retrieval

Production is configured for lower-cost hosting:

- `LLM_PROVIDER=groq`
- `EMBEDDING_PROVIDER=local`
- Groq-hosted Llama for answer generation
- Local sentence-transformer embeddings for retrieval
- Qdrant Cloud for vector storage

## Qdrant Keepalive

The backend starts a lightweight Qdrant keepalive scheduler when `QDRANT_URL` is configured. It calls the configured collection every 12 hours by default so a free-tier Qdrant cluster does not become inactive while the backend process is running.

Configuration:

- `QDRANT_URL`
- `QDRANT_API_KEY`
- `QDRANT_COLLECTION=repo_qa_chunks`
- `QDRANT_KEEPALIVE_ENABLED=true`
- `QDRANT_KEEPALIVE_INTERVAL_SECONDS=43200`

The main repository also includes a GitHub Actions keepalive workflow for cases where the backend host is asleep.

## Metrics

Metrics will be added after the next benchmark rerun. The evaluation harness is set up to report retrieval hit rate, top-1 hit rate, mean reciprocal rank, source recall, grounded answer rate, checklist pass rate, and optional RAGAS judge metrics.
