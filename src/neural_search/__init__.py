"""Neural Search - Production-grade semantic search API."""

from neural_search.config import get_settings
from neural_search.core.embeddings import EmbeddingModel, get_embedding_model
from neural_search.core.search_engine import SearchEngine, SearchResult, SearchResponse, get_search_engine
from neural_search.storage import VectorStore, get_vector_store

__version__ = "0.1.0"

__all__ = [
    "__version__",
    "get_settings",
    "EmbeddingModel",
    "get_embedding_model",
    "SearchEngine",
    "SearchResult",
    "SearchResponse",
    "get_search_engine",
    "VectorStore",
    "get_vector_store",
]
