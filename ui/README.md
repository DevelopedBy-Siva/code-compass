# Code Compass UI

React frontend for submitting a GitHub URL, monitoring indexing state, chatting with the indexed codebase, and reviewing cited sources.

## API target

Production requests use the same-origin `/api` Vercel function in `api/[...path].js`, which invokes the configured SageMaker endpoint. Leave `REACT_APP_API_URL` unset for deployed builds; it is honored only on `localhost` or `127.0.0.1` for direct local FastAPI development.
