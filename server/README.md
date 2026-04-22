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
- Generate embeddings and store chunks in Chroma DB
- Maintain lightweight repository and session metadata in memory
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
- Chroma DB for vector storage

## Chroma Storage

The backend uses Chroma DB for vector storage in both local development and production. By default it stores the collection under `./data/chroma`, and you can point it somewhere else with `CHROMA_PATH`.

Configuration:

- `CHROMA_PATH=./data/chroma`
- `CHROMA_COLLECTION=repo_qa_chunks`
- `CHROMA_UPSERT_BATCH_SIZE=64`

## Metrics

Metrics will be added after the next benchmark rerun. The evaluation harness is set up to report retrieval hit rate, top-1 hit rate, mean reciprocal rank, source recall, grounded answer rate, checklist pass rate, and optional RAGAS judge metrics.
