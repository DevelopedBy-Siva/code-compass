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

### Model Stack
- LLM: Amazon Bedrock with Qwen3-Coder-Next
- Embeddings: local Qwen3-Embedding-4B
- Reranker: local Qwen3-Reranker-4B
- Required AWS configuration: `AWS_REGION` or `AWS_DEFAULT_REGION`

## Chroma Storage

The backend uses Chroma DB for vector storage in both local development and production. By default it stores the collection under `./data/chroma`, and you can point it somewhere else with `CHROMA_PATH`.

Configuration:

- `CHROMA_PATH=./data/chroma`
- `CHROMA_COLLECTION=repo_qa_chunks`
- `CHROMA_UPSERT_BATCH_SIZE=64`

## Metrics

The evaluation harness reports 4 core metrics:
- **Retrieval hit rate @ top-5**: Fraction of queries with at least one relevant source in top 5 results
- **Top-1 hit rate**: Fraction of queries where the first result is relevant
- **Grounded answer rate**: Fraction of answers that cite actual source code
- **Faithfulness (RAGAS)**: LLM-as-judge score for answer consistency with retrieved context
- **Query latency P95**: 95th percentile response time in milliseconds
