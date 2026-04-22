"""
Codebase RAG backend package.
"""

from .code_parser import CodeParser
from .embeddings import EmbeddingGenerator
from .hybrid_search import HybridSearchEngine
from .rag_system import CodebaseRAGSystem
from .repo_fetcher import RepoFetcher
from .vector_store import ChromaVectorStore

__version__ = "2.0.0"
__all__ = [
    "CodeParser",
    "CodebaseRAGSystem",
    "EmbeddingGenerator",
    "ChromaVectorStore",
    "HybridSearchEngine",
    "RepoFetcher",
]
