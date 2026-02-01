"""Tests for collection API endpoints."""

import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock, AsyncMock

from neural_search.main import app


@pytest.fixture
def client():
    """Test client fixture."""
    return TestClient(app)


class TestCollectionEndpoints:
    """Tests for collection API endpoints."""

    def test_create_collection(self, client: TestClient):
        """Test creating a collection."""
        with patch("neural_search.api.collections.get_search_engine") as mock_engine:
            mock_se = MagicMock()
            mock_se.vector_store.collection_exists = AsyncMock(return_value=False)
            mock_se.create_collection = AsyncMock(return_value=True)
            mock_engine.return_value = mock_se

            with patch("neural_search.api.collections.get_embedding_model") as mock_model:
                mock_model.return_value.dimension = 384

                response = client.post(
                    "/api/v1/collections",
                    json={
                        "name": "test-collection",
                        "distance_metric": "cosine"
                    }
                )

                assert response.status_code == 201
                data = response.json()
                assert data["name"] == "test-collection"
                assert data["dimension"] == 384
                assert data["distance_metric"] == "cosine"

    def test_create_collection_custom_dimension(self, client: TestClient):
        """Test creating collection with custom dimension."""
        with patch("neural_search.api.collections.get_search_engine") as mock_engine:
            mock_se = MagicMock()
            mock_se.vector_store.collection_exists = AsyncMock(return_value=False)
            mock_se.create_collection = AsyncMock(return_value=True)
            mock_engine.return_value = mock_se

            with patch("neural_search.api.collections.get_embedding_model") as mock_model:
                mock_model.return_value.dimension = 384

                response = client.post(
                    "/api/v1/collections",
                    json={
                        "name": "test-collection",
                        "dimension": 768
                    }
                )

                assert response.status_code == 201
                data = response.json()
                assert data["dimension"] == 768

    def test_create_collection_already_exists(self, client: TestClient):
        """Test creating collection that already exists."""
        with patch("neural_search.api.collections.get_search_engine") as mock_engine:
            mock_se = MagicMock()
            mock_se.vector_store.collection_exists = AsyncMock(return_value=True)
            mock_engine.return_value = mock_se

            response = client.post(
                "/api/v1/collections",
                json={"name": "existing-collection"}
            )

            assert response.status_code == 409
            assert "already exists" in response.json()["detail"].lower()

    def test_list_collections(self, client: TestClient):
        """Test listing collections."""
        with patch("neural_search.api.collections.get_search_engine") as mock_engine:
            mock_se = MagicMock()
            mock_se.list_collections = AsyncMock(return_value=["collection1", "collection2"])
            mock_engine.return_value = mock_se

            response = client.get("/api/v1/collections")

            assert response.status_code == 200
            data = response.json()
            assert len(data["collections"]) == 2
            assert data["count"] == 2

    def test_get_collection_stats(self, client: TestClient):
        """Test getting collection statistics."""
        with patch("neural_search.api.collections.get_search_engine") as mock_engine:
            mock_se = MagicMock()
            mock_se.vector_store.collection_exists = AsyncMock(return_value=True)
            mock_se.get_collection_stats = AsyncMock(return_value={
                "name": "test",
                "dimension": 384,
                "distance_metric": "cosine",
                "count": 1000,
                "indexed_count": 1000,
                "bm25": {"num_documents": 1000}
            })
            mock_engine.return_value = mock_se

            response = client.get("/api/v1/collections/test")

            assert response.status_code == 200
            data = response.json()
            assert data["name"] == "test"
            assert data["count"] == 1000
            assert data["bm25"]["num_documents"] == 1000

    def test_get_collection_stats_not_found(self, client: TestClient):
        """Test getting stats for non-existent collection."""
        with patch("neural_search.api.collections.get_search_engine") as mock_engine:
            mock_se = MagicMock()
            mock_se.vector_store.collection_exists = AsyncMock(return_value=False)
            mock_engine.return_value = mock_se

            response = client.get("/api/v1/collections/nonexistent")

            assert response.status_code == 404

    def test_delete_collection(self, client: TestClient):
        """Test deleting a collection."""
        with patch("neural_search.api.collections.get_search_engine") as mock_engine:
            mock_se = MagicMock()
            mock_se.vector_store.collection_exists = AsyncMock(return_value=True)
            mock_se.delete_collection = AsyncMock(return_value=True)
            mock_engine.return_value = mock_se

            response = client.delete("/api/v1/collections/test")

            assert response.status_code == 204

    def test_delete_collection_not_found(self, client: TestClient):
        """Test deleting non-existent collection."""
        with patch("neural_search.api.collections.get_search_engine") as mock_engine:
            mock_se = MagicMock()
            mock_se.vector_store.collection_exists = AsyncMock(return_value=False)
            mock_engine.return_value = mock_se

            response = client.delete("/api/v1/collections/nonexistent")

            assert response.status_code == 404


class TestCollectionValidation:
    """Tests for collection request validation."""

    def test_invalid_collection_name(self, client: TestClient):
        """Test that invalid collection names are rejected."""
        # Contains invalid characters
        response = client.post(
            "/api/v1/collections",
            json={"name": "test collection!"}
        )
        assert response.status_code == 422

        # Empty name
        response = client.post(
            "/api/v1/collections",
            json={"name": ""}
        )
        assert response.status_code == 422

    def test_invalid_distance_metric(self, client: TestClient):
        """Test that invalid distance metrics are rejected."""
        response = client.post(
            "/api/v1/collections",
            json={
                "name": "test",
                "distance_metric": "invalid"
            }
        )
        assert response.status_code == 422

    def test_invalid_dimension(self, client: TestClient):
        """Test that invalid dimensions are rejected."""
        # Too small
        response = client.post(
            "/api/v1/collections",
            json={
                "name": "test",
                "dimension": 0
            }
        )
        assert response.status_code == 422

        # Too large
        response = client.post(
            "/api/v1/collections",
            json={
                "name": "test",
                "dimension": 5000
            }
        )
        assert response.status_code == 422
