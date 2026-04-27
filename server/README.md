---
title: Code Compass API
emoji: 🚀
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 7860
---

# Code Compass Backend

FastAPI backend for a session-oriented GitHub repo QA tool.

Behavior:

- Clones a public GitHub repo
- Chunks it with tree-sitter
- Builds retrieval state with a Qdrant adapter
- Answers questions with Groq-hosted Llama or Amazon Bedrock Claude depending on environment configuration
- Deletes the cloned repo after indexing
- Keeps only lightweight repo metadata in SQLite
