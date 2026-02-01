"""Tests for search API endpoints."""

import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock, AsyncMock

from neural_search.main import app
from neural_search.core.search_engine import SearchResult, SearchResponse


@pytest.fixture
def client():
    """Test client fixture."""
    return TestClient(app)


class TestSearchEndpoints:
    """Tests for search API endpoints."""

    def test_search_collection_not_found(self, client: TestClient):
        """Test searching non-existent collection."""
        with patch("neural_search.api.search.get_search_engine") as mock_engine:
            mock_se = MagicMock()
            mock_se.vector_store.collection_exists = AsyncMock(return_value=False)
            mock_engine.return_value = mock_se

            response = client.post(
                "/api/v1/search",
                json={
                    "query": "test query",
                    "collection": "nonexistent"
                }
            )

            assert response.status_code == 404
            assert "not found" in response.json()["detail"].lower()

    def test_search_basic(self, client: TestClient):
        """Test basic search."""
        with patch("neural_search.api.search.get_search_engine") as mock_engine:
            mock_se = MagicMock()
            mock_se.vector_store.collection_exists = AsyncMock(return_value=True)
            mock_se.search = AsyncMock(return_value=SearchResponse(
                results=[
                    SearchResult(
                        id="doc1",
                        content="Test content",
                        score=0.95,
                        metadata={"key": "value"}
                    )
                ],
                took_ms=10.5,
                total_found=1,
                query="test query",
                collection="test"
            ))
            mock_engine.return_value = mock_se

            response = client.post(
                "/api/v1/search",
                json={
                    "query": "test query",
                    "collection": "test",
                    "top_k": 10
                }
            )

            assert response.status_code == 200
            data = response.json()
            assert len(data["results"]) == 1
            assert data["results"][0]["id"] == "doc1"
            assert data["results"][0]["score"] == 0.95
            assert data["took_ms"] == 10.5

    def test_search_with_filters(self, client: TestClient):
        """Test search with metadata filters."""
        with patch("neural_search.api.search.get_search_engine") as mock_engine:
            mock_se = MagicMock()
            mock_se.vector_store.collection_exists = AsyncMock(return_value=True)
            mock_se.search = AsyncMock(return_value=SearchResponse(
                results=[],
                took_ms=5.0,
                total_found=0,
                query="test",
                collection="test"
            ))
            mock_engine.return_value = mock_se

            response = client.post(
                "/api/v1/search",
                json={
                    "query": "test query",
                    "collection": "test",
                    "filters": {"category": "technology"}
                }
            )

            assert response.status_code == 200
            # Verify filter was passed to search
            mock_se.search.assert_called_once()
            call_kwargs = mock_se.search.call_args[1]
            assert call_kwargs["filters"] == {"category": "technology"}

    def test_search_hybrid(self, client: TestClient):
        """Test hybrid search."""
        with patch("neural_search.api.search.get_search_engine") as mock_engine:
            mock_se = MagicMock()
            mock_se.vector_store.collection_exists = AsyncMock(return_value=True)
            mock_se.search = AsyncMock(return_value=SearchResponse(
                results=[],
                took_ms=15.0,
                total_found=0,
                query="test",
                collection="test"
            ))
            mock_engine.return_value = mock_se

            response = client.post(
                "/api/v1/search",
                json={
                    "query": "test query",
                    "collection": "test",
                    "hybrid": True,
                    "alpha": 0.7
                }
            )

            assert response.status_code == 200
            call_kwargs = mock_se.search.call_args[1]
            assert call_kwargs["hybrid"] is True
            assert call_kwargs["alpha"] == 0.7

    def test_search_with_rerank(self, client: TestClient):
        """Test search with reranking."""
        with patch("neural_search.api.search.get_search_engine") as mock_engine:
            mock_se = MagicMock()
            mock_se.vector_store.collection_exists = AsyncMock(return_value=True)
            mock_se.search = AsyncMock(return_value=SearchResponse(
                results=[],
                took_ms=20.0,
                total_found=0,
                query="test",
                collection="test"
            ))
            mock_engine.return_value = mock_se

            response = client.post(
                "/api/v1/search",
                json={
                    "query": "test query",
                    "collection": "test",
                    "rerank": True
                }
            )

            assert response.status_code == 200
            call_kwargs = mock_se.search.call_args[1]
            assert call_kwargs["rerank"] is True

    def test_search_exclude_metadata(self, client: TestClient):
        """Test search with metadata excluded."""
        with patch("neural_search.api.search.get_search_engine") as mock_engine:
            mock_se = MagicMock()
            mock_se.vector_store.collection_exists = AsyncMock(return_value=True)
            mock_se.search = AsyncMock(return_value=SearchResponse(
                results=[
                    SearchResult(
                        id="doc1",
                        content="Test content",
                        score=0.95,
                        metadata={"key": "value"}
                    )
                ],
                took_ms=10.0,
                total_found=1,
                query="test",
                collection="test"
            ))
            mock_engine.return_value = mock_se

            response = client.post(
                "/api/v1/search",
                json={
                    "query": "test query",
                    "collection": "test",
                    "include_metadata": False
                }
            )

            assert response.status_code == 200
            data = response.json()
            assert data["results"][0]["metadata"] is None

    def test_batch_search(self, client: TestClient):
        """Test batch search."""
        with patch("neural_search.api.search.get_search_engine") as mock_engine:
            mock_se = MagicMock()
            mock_se.vector_store.collection_exists = AsyncMock(return_value=True)
            mock_se.search = AsyncMock(return_value=SearchResponse(
                results=[],
                took_ms=5.0,
                total_found=0,
                query="test",
                collection="test"
            ))
            mock_engine.return_value = mock_se

            response = client.post(
                "/api/v1/search/batch",
                json=[
                    {"query": "query 1", "collection": "test"},
                    {"query": "query 2", "collection": "test"}
                ]
            )

            assert response.status_code == 200
            data = response.json()
            assert len(data) == 2

    def test_batch_search_limit(self, client: TestClient):
        """Test batch search with too many queries."""
        response = client.post(
            "/api/v1/search/batch",
            json=[{"query": f"query {i}", "collection": "test"} for i in range(101)]
        )

        assert response.status_code == 400
        assert "100" in response.json()["detail"]


class TestSearchValidation:
    """Tests for search request validation."""

    def test_empty_query(self, client: TestClient):
        """Test that empty query is rejected."""
        response = client.post(
            "/api/v1/search",
            json={
                "query": "",
                "collection": "test"
            }
        )

        assert response.status_code == 422

    def test_top_k_bounds(self, client: TestClient):
        """Test top_k parameter bounds."""
        # Too low
        response = client.post(
            "/api/v1/search",
            json={
                "query": "test",
                "collection": "test",
                "top_k": 0
            }
        )
        assert response.status_code == 422

        # Too high
        response = client.post(
            "/api/v1/search",
            json={
                "query": "test",
                "collection": "test",
                "top_k": 101
            }
        )
        assert response.status_code == 422

    def test_alpha_bounds(self, client: TestClient):
        """Test alpha parameter bounds."""
        # Too low
        response = client.post(
            "/api/v1/search",
            json={
                "query": "test",
                "collection": "test",
                "alpha": -0.1
            }
        )
        assert response.status_code == 422

        # Too high
        response = client.post(
            "/api/v1/search",
            json={
                "query": "test",
                "collection": "test",
                "alpha": 1.1
            }
        )
        assert response.status_code == 422
