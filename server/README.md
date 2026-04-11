# Code Compass Backend

FastAPI backend for a session-oriented GitHub repo QA tool.

Behavior:

- Clones a public GitHub repo
- Chunks it with tree-sitter
- Builds retrieval state with a Qdrant adapter
- Answers questions with Vertex AI Gemini and grounded citations
- Deletes the cloned repo after indexing
- Keeps only lightweight repo metadata in SQLite
