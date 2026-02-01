"""Tests for document API endpoints."""

import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock, AsyncMock

from neural_search.main import app
from neural_search.api.schemas import DocumentsCreateRequest, DocumentInput


@pytest.fixture
def client():
    """Test client fixture."""
    return TestClient(app)


class TestDocumentEndpoints:
    """Tests for document API endpoints."""

    def test_create_documents_collection_not_found(self, client: TestClient):
        """Test creating documents in non-existent collection."""
        with patch("neural_search.api.documents.get_search_engine") as mock_engine:
            mock_se = MagicMock()
            mock_se.vector_store.collection_exists = AsyncMock(return_value=False)
            mock_engine.return_value = mock_se

            response = client.post(
                "/api/v1/documents",
                json={
                    "collection": "nonexistent",
                    "documents": [
                        {"content": "Test document"}
                    ]
                }
            )

            assert response.status_code == 404
            assert "not found" in response.json()["detail"].lower()

    def test_create_documents_sync(self, client: TestClient):
        """Test synchronous document creation."""
        with patch("neural_search.api.documents.get_search_engine") as mock_engine:
            mock_se = MagicMock()
            mock_se.vector_store.collection_exists = AsyncMock(return_value=True)
            mock_se.index_documents = AsyncMock(return_value=["doc1"])
            mock_engine.return_value = mock_se

            response = client.post(
                "/api/v1/documents",
                json={
                    "collection": "test",
                    "documents": [
                        {"id": "doc1", "content": "Test document", "metadata": {"key": "value"}}
                    ]
                }
            )

            assert response.status_code == 202
            data = response.json()
            assert data["status"] == "completed"
            assert data["documents_count"] == 1
            assert "doc1" in data["document_ids"]

    def test_create_documents_async(self, client: TestClient):
        """Test asynchronous document creation."""
        with patch("neural_search.api.documents.get_search_engine") as mock_engine:
            mock_se = MagicMock()
            mock_se.vector_store.collection_exists = AsyncMock(return_value=True)
            mock_engine.return_value = mock_se

            with patch("neural_search.api.documents.index_documents_task") as mock_task:
                mock_task.delay.return_value = MagicMock(id="job-123")

                response = client.post(
                    "/api/v1/documents?async_mode=true",
                    json={
                        "collection": "test",
                        "documents": [
                            {"content": "Test document"}
                        ]
                    }
                )

                assert response.status_code == 202
                data = response.json()
                assert data["status"] == "accepted"
                assert data["job_id"] == "job-123"
                assert data["documents_count"] == 1

    def test_get_document(self, client: TestClient):
        """Test getting a document by ID."""
        with patch("neural_search.api.documents.get_search_engine") as mock_engine:
            mock_se = MagicMock()
            mock_se.get_document = AsyncMock(return_value={
                "id": "doc1",
                "content": "Test content",
                "metadata": {"key": "value"}
            })
            mock_engine.return_value = mock_se

            response = client.get("/api/v1/documents/test/doc1")

            assert response.status_code == 200
            data = response.json()
            assert data["id"] == "doc1"
            assert data["content"] == "Test content"

    def test_get_document_not_found(self, client: TestClient):
        """Test getting non-existent document."""
        with patch("neural_search.api.documents.get_search_engine") as mock_engine:
            mock_se = MagicMock()
            mock_se.get_document = AsyncMock(return_value=None)
            mock_engine.return_value = mock_se

            response = client.get("/api/v1/documents/test/nonexistent")

            assert response.status_code == 404

    def test_delete_documents(self, client: TestClient):
        """Test deleting documents."""
        with patch("neural_search.api.documents.get_search_engine") as mock_engine:
            mock_se = MagicMock()
            mock_se.vector_store.collection_exists = AsyncMock(return_value=True)
            mock_se.delete_documents = AsyncMock(return_value=True)
            mock_engine.return_value = mock_se

            response = client.request(
                "DELETE",
                "/api/v1/documents",
                json={
                    "collection": "test",
                    "document_ids": ["doc1", "doc2"]
                }
            )

            assert response.status_code == 200
            data = response.json()
            assert data["status"] == "completed"
            assert data["deleted_count"] == 2


class TestDocumentValidation:
    """Tests for document request validation."""

    def test_empty_documents_list(self, client: TestClient):
        """Test that empty documents list is rejected."""
        response = client.post(
            "/api/v1/documents",
            json={
                "collection": "test",
                "documents": []
            }
        )

        assert response.status_code == 422

    def test_empty_content(self, client: TestClient):
        """Test that empty content is rejected."""
        response = client.post(
            "/api/v1/documents",
            json={
                "collection": "test",
                "documents": [{"content": ""}]
            }
        )

        assert response.status_code == 422

    def test_missing_collection(self, client: TestClient):
        """Test that missing collection is rejected."""
        response = client.post(
            "/api/v1/documents",
            json={
                "documents": [{"content": "test"}]
            }
        )

        assert response.status_code == 422
