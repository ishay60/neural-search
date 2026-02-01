"""Pytest configuration and fixtures."""

import asyncio
from typing import AsyncGenerator, Generator

import numpy as np
import pytest
from fastapi.testclient import TestClient
from httpx import ASGITransport, AsyncClient

from neural_search.config import Settings, get_settings
from neural_search.core.embeddings import EmbeddingModel
from neural_search.core.hybrid import HybridSearcher
from neural_search.core.search_engine import SearchEngine
from neural_search.main import app
from neural_search.storage.faiss_store import FAISSStore


# Override settings for testing
@pytest.fixture
def test_settings() -> Settings:
    """Test settings with in-memory stores."""
    return Settings(
        vector_store_type="faiss",
        embedding_model="all-MiniLM-L6-v2",
        embedding_device="cpu",
        redis_url="redis://localhost:6379/15",  # Test database
    )


@pytest.fixture
def event_loop():
    """Create event loop for async tests."""
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def faiss_store() -> FAISSStore:
    """Create a fresh FAISS store for testing."""
    return FAISSStore()


@pytest.fixture
def hybrid_searcher() -> HybridSearcher:
    """Create a hybrid searcher for testing."""
    return HybridSearcher(k1=1.5, b=0.75, alpha=0.5)


@pytest.fixture
def sample_documents() -> list[dict]:
    """Sample documents for testing."""
    return [
        {
            "id": "doc1",
            "content": "Machine learning is a subset of artificial intelligence that enables systems to learn from data.",
            "metadata": {"category": "technology", "source": "wikipedia"},
        },
        {
            "id": "doc2",
            "content": "Deep learning uses neural networks with multiple layers to process complex patterns.",
            "metadata": {"category": "technology", "source": "textbook"},
        },
        {
            "id": "doc3",
            "content": "Natural language processing allows computers to understand human language.",
            "metadata": {"category": "technology", "source": "blog"},
        },
        {
            "id": "doc4",
            "content": "Computer vision enables machines to interpret visual information from the world.",
            "metadata": {"category": "technology", "source": "paper"},
        },
        {
            "id": "doc5",
            "content": "The Python programming language is popular for data science and machine learning.",
            "metadata": {"category": "programming", "source": "tutorial"},
        },
    ]


@pytest.fixture
def sample_embeddings() -> np.ndarray:
    """Sample embeddings for testing (random for speed)."""
    np.random.seed(42)
    return np.random.randn(5, 384).astype(np.float32)


@pytest.fixture
def client() -> TestClient:
    """Create test client for sync tests."""
    return TestClient(app)


@pytest.fixture
async def async_client() -> AsyncGenerator[AsyncClient, None]:
    """Create async test client."""
    async with AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://test",
    ) as client:
        yield client


class MockEmbeddingModel:
    """Mock embedding model for testing without loading real model."""

    def __init__(self, dimension: int = 384):
        self.dimension = dimension
        self.model_name = "mock-model"

    def load(self):
        pass

    def encode(
        self,
        texts: list[str],
        batch_size: int = 32,
        normalize: bool = True,
        show_progress: bool = False,
    ) -> np.ndarray:
        """Return random embeddings."""
        np.random.seed(hash(str(texts)) % (2**32))
        embeddings = np.random.randn(len(texts), self.dimension).astype(np.float32)
        if normalize:
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
            embeddings = embeddings / norms
        return embeddings

    def encode_query(self, query: str, normalize: bool = True) -> np.ndarray:
        return self.encode([query], normalize=normalize)[0]

    def encode_documents(
        self,
        documents: list[str],
        batch_size: int = 32,
        normalize: bool = True,
        show_progress: bool = False,
    ) -> np.ndarray:
        return self.encode(documents, batch_size, normalize, show_progress)


@pytest.fixture
def mock_embedding_model() -> MockEmbeddingModel:
    """Create mock embedding model."""
    return MockEmbeddingModel()


@pytest.fixture
async def search_engine_with_data(
    faiss_store: FAISSStore,
    mock_embedding_model: MockEmbeddingModel,
    sample_documents: list[dict],
) -> SearchEngine:
    """Create search engine with sample data."""
    engine = SearchEngine(
        embedding_model=mock_embedding_model,  # type: ignore
        vector_store=faiss_store,
        reranker=None,
    )

    # Create collection and index documents
    await engine.create_collection("test-collection", dimension=384)
    await engine.index_documents("test-collection", sample_documents)

    return engine
