"""Tests for FAISS vector store."""

import numpy as np
import pytest

from neural_search.storage.faiss_store import FAISSStore


class TestFAISSStore:
    """Tests for FAISSStore class."""

    @pytest.mark.asyncio
    async def test_create_collection(self, faiss_store: FAISSStore):
        """Test creating a collection."""
        result = await faiss_store.create_collection("test", dimension=384)

        assert result is True
        assert await faiss_store.collection_exists("test")

    @pytest.mark.asyncio
    async def test_create_collection_duplicate(self, faiss_store: FAISSStore):
        """Test creating duplicate collection."""
        await faiss_store.create_collection("test", dimension=384)
        result = await faiss_store.create_collection("test", dimension=384)

        assert result is False

    @pytest.mark.asyncio
    async def test_delete_collection(self, faiss_store: FAISSStore):
        """Test deleting a collection."""
        await faiss_store.create_collection("test", dimension=384)
        result = await faiss_store.delete_collection("test")

        assert result is True
        assert not await faiss_store.collection_exists("test")

    @pytest.mark.asyncio
    async def test_delete_nonexistent_collection(self, faiss_store: FAISSStore):
        """Test deleting non-existent collection."""
        result = await faiss_store.delete_collection("nonexistent")

        assert result is False

    @pytest.mark.asyncio
    async def test_list_collections(self, faiss_store: FAISSStore):
        """Test listing collections."""
        await faiss_store.create_collection("col1", dimension=384)
        await faiss_store.create_collection("col2", dimension=384)

        collections = await faiss_store.list_collections()

        assert "col1" in collections
        assert "col2" in collections
        assert len(collections) == 2

    @pytest.mark.asyncio
    async def test_get_collection_stats(self, faiss_store: FAISSStore):
        """Test getting collection statistics."""
        await faiss_store.create_collection("test", dimension=384)

        stats = await faiss_store.get_collection_stats("test")

        assert stats["name"] == "test"
        assert stats["dimension"] == 384
        assert stats["count"] == 0

    @pytest.mark.asyncio
    async def test_upsert(self, faiss_store: FAISSStore):
        """Test upserting vectors."""
        await faiss_store.create_collection("test", dimension=384)

        vectors = [
            {
                "id": "doc1",
                "embedding": np.random.randn(384).astype(np.float32),
                "content": "Test content 1",
                "metadata": {"key": "value1"},
            },
            {
                "id": "doc2",
                "embedding": np.random.randn(384).astype(np.float32),
                "content": "Test content 2",
                "metadata": {"key": "value2"},
            },
        ]

        result = await faiss_store.upsert("test", vectors)

        assert result is True
        stats = await faiss_store.get_collection_stats("test")
        assert stats["count"] == 2

    @pytest.mark.asyncio
    async def test_upsert_nonexistent_collection(self, faiss_store: FAISSStore):
        """Test upserting to non-existent collection."""
        vectors = [{"id": "doc1", "embedding": np.random.randn(384), "content": ""}]

        with pytest.raises(ValueError):
            await faiss_store.upsert("nonexistent", vectors)

    @pytest.mark.asyncio
    async def test_delete_vectors(self, faiss_store: FAISSStore):
        """Test deleting vectors."""
        await faiss_store.create_collection("test", dimension=384)

        vectors = [
            {"id": "doc1", "embedding": np.random.randn(384), "content": "test"},
            {"id": "doc2", "embedding": np.random.randn(384), "content": "test"},
        ]
        await faiss_store.upsert("test", vectors)

        result = await faiss_store.delete("test", ["doc1"])

        assert result is True
        doc = await faiss_store.get("test", "doc1")
        assert doc is None

    @pytest.mark.asyncio
    async def test_get_vector(self, faiss_store: FAISSStore):
        """Test getting a vector by ID."""
        await faiss_store.create_collection("test", dimension=384)

        vectors = [
            {
                "id": "doc1",
                "embedding": np.random.randn(384),
                "content": "Test content",
                "metadata": {"key": "value"},
            },
        ]
        await faiss_store.upsert("test", vectors)

        doc = await faiss_store.get("test", "doc1")

        assert doc is not None
        assert doc["id"] == "doc1"
        assert doc["content"] == "Test content"
        assert doc["metadata"]["key"] == "value"

    @pytest.mark.asyncio
    async def test_get_nonexistent_vector(self, faiss_store: FAISSStore):
        """Test getting non-existent vector."""
        await faiss_store.create_collection("test", dimension=384)

        doc = await faiss_store.get("test", "nonexistent")

        assert doc is None

    @pytest.mark.asyncio
    async def test_search(self, faiss_store: FAISSStore):
        """Test searching for similar vectors."""
        await faiss_store.create_collection("test", dimension=384)

        # Create vectors with known similarity
        base_vector = np.random.randn(384).astype(np.float32)
        base_vector = base_vector / np.linalg.norm(base_vector)

        # Create similar and dissimilar vectors
        similar_vector = base_vector + np.random.randn(384).astype(np.float32) * 0.1
        similar_vector = similar_vector / np.linalg.norm(similar_vector)

        dissimilar_vector = np.random.randn(384).astype(np.float32)
        dissimilar_vector = dissimilar_vector / np.linalg.norm(dissimilar_vector)

        vectors = [
            {"id": "similar", "embedding": similar_vector, "content": "similar"},
            {"id": "dissimilar", "embedding": dissimilar_vector, "content": "dissimilar"},
        ]
        await faiss_store.upsert("test", vectors)

        results = await faiss_store.search("test", base_vector, top_k=2)

        assert len(results) == 2
        # Similar vector should have higher score
        assert results[0]["id"] == "similar"

    @pytest.mark.asyncio
    async def test_search_with_filters(self, faiss_store: FAISSStore):
        """Test search with metadata filters."""
        await faiss_store.create_collection("test", dimension=384)

        vectors = [
            {
                "id": "doc1",
                "embedding": np.random.randn(384),
                "content": "doc1",
                "metadata": {"category": "A"},
            },
            {
                "id": "doc2",
                "embedding": np.random.randn(384),
                "content": "doc2",
                "metadata": {"category": "B"},
            },
        ]
        await faiss_store.upsert("test", vectors)

        query = np.random.randn(384).astype(np.float32)
        results = await faiss_store.search(
            "test",
            query,
            top_k=10,
            filters={"category": "A"},
        )

        assert len(results) == 1
        assert results[0]["id"] == "doc1"

    @pytest.mark.asyncio
    async def test_search_filter_operators(self, faiss_store: FAISSStore):
        """Test search with various filter operators."""
        await faiss_store.create_collection("test", dimension=384)

        vectors = [
            {"id": "d1", "embedding": np.random.randn(384), "content": "", "metadata": {"score": 10}},
            {"id": "d2", "embedding": np.random.randn(384), "content": "", "metadata": {"score": 20}},
            {"id": "d3", "embedding": np.random.randn(384), "content": "", "metadata": {"score": 30}},
        ]
        await faiss_store.upsert("test", vectors)

        query = np.random.randn(384).astype(np.float32)

        # Test $gt
        results = await faiss_store.search("test", query, filters={"score": {"$gt": 15}})
        assert all(r["metadata"]["score"] > 15 for r in results)

        # Test $lte
        results = await faiss_store.search("test", query, filters={"score": {"$lte": 20}})
        assert all(r["metadata"]["score"] <= 20 for r in results)

        # Test $in
        results = await faiss_store.search("test", query, filters={"score": {"$in": [10, 30]}})
        assert all(r["metadata"]["score"] in [10, 30] for r in results)

    @pytest.mark.asyncio
    async def test_search_empty_collection(self, faiss_store: FAISSStore):
        """Test searching empty collection."""
        await faiss_store.create_collection("test", dimension=384)

        query = np.random.randn(384).astype(np.float32)
        results = await faiss_store.search("test", query)

        assert results == []


class TestFAISSStoreDistanceMetrics:
    """Tests for different distance metrics."""

    @pytest.mark.asyncio
    async def test_cosine_distance(self, faiss_store: FAISSStore):
        """Test cosine distance metric."""
        await faiss_store.create_collection("test", dimension=384, distance_metric="cosine")

        stats = await faiss_store.get_collection_stats("test")
        assert stats["distance_metric"] == "cosine"

    @pytest.mark.asyncio
    async def test_euclidean_distance(self, faiss_store: FAISSStore):
        """Test euclidean distance metric."""
        await faiss_store.create_collection("test", dimension=384, distance_metric="euclidean")

        stats = await faiss_store.get_collection_stats("test")
        assert stats["distance_metric"] == "euclidean"
