"""Tests for search engine."""

import pytest

from neural_search.core.search_engine import SearchEngine


class TestSearchEngine:
    """Tests for SearchEngine class."""

    @pytest.mark.asyncio
    async def test_create_collection(self, search_engine_with_data: SearchEngine):
        """Test that collection was created."""
        collections = await search_engine_with_data.list_collections()
        assert "test-collection" in collections

    @pytest.mark.asyncio
    async def test_search(self, search_engine_with_data: SearchEngine):
        """Test basic search."""
        response = await search_engine_with_data.search(
            query="machine learning artificial intelligence",
            collection="test-collection",
            top_k=5,
        )

        assert response.total_found > 0
        assert len(response.results) <= 5
        assert response.took_ms >= 0
        assert response.query == "machine learning artificial intelligence"
        assert response.collection == "test-collection"

    @pytest.mark.asyncio
    async def test_search_with_filters(self, search_engine_with_data: SearchEngine):
        """Test search with metadata filters."""
        response = await search_engine_with_data.search(
            query="technology",
            collection="test-collection",
            top_k=10,
            filters={"category": "technology"},
        )

        # All results should have category=technology
        for result in response.results:
            assert result.metadata.get("category") == "technology"

    @pytest.mark.asyncio
    async def test_search_top_k(self, search_engine_with_data: SearchEngine):
        """Test top_k parameter."""
        response = await search_engine_with_data.search(
            query="test",
            collection="test-collection",
            top_k=2,
        )

        assert len(response.results) <= 2

    @pytest.mark.asyncio
    async def test_get_document(self, search_engine_with_data: SearchEngine):
        """Test getting a document by ID."""
        doc = await search_engine_with_data.get_document("test-collection", "doc1")

        assert doc is not None
        assert doc["id"] == "doc1"
        assert "content" in doc

    @pytest.mark.asyncio
    async def test_get_nonexistent_document(self, search_engine_with_data: SearchEngine):
        """Test getting non-existent document."""
        doc = await search_engine_with_data.get_document("test-collection", "nonexistent")

        assert doc is None

    @pytest.mark.asyncio
    async def test_delete_documents(self, search_engine_with_data: SearchEngine):
        """Test deleting documents."""
        # Verify document exists
        doc = await search_engine_with_data.get_document("test-collection", "doc1")
        assert doc is not None

        # Delete it
        result = await search_engine_with_data.delete_documents(
            "test-collection",
            ["doc1"],
        )
        assert result is True

        # Verify it's gone
        doc = await search_engine_with_data.get_document("test-collection", "doc1")
        assert doc is None

    @pytest.mark.asyncio
    async def test_delete_collection(self, search_engine_with_data: SearchEngine):
        """Test deleting a collection."""
        result = await search_engine_with_data.delete_collection("test-collection")
        assert result is True

        collections = await search_engine_with_data.list_collections()
        assert "test-collection" not in collections

    @pytest.mark.asyncio
    async def test_get_collection_stats(self, search_engine_with_data: SearchEngine):
        """Test getting collection statistics."""
        stats = await search_engine_with_data.get_collection_stats("test-collection")

        assert stats["name"] == "test-collection"
        assert stats["dimension"] == 384
        assert stats["count"] == 5  # 5 sample documents


class TestSearchEngineHybrid:
    """Tests for hybrid search functionality."""

    @pytest.mark.asyncio
    async def test_hybrid_search(self, search_engine_with_data: SearchEngine):
        """Test hybrid search."""
        response = await search_engine_with_data.search(
            query="machine learning neural networks",
            collection="test-collection",
            top_k=5,
            hybrid=True,
        )

        assert response.total_found > 0
        assert len(response.results) <= 5

    @pytest.mark.asyncio
    async def test_hybrid_search_with_alpha(self, search_engine_with_data: SearchEngine):
        """Test hybrid search with custom alpha."""
        response = await search_engine_with_data.search(
            query="machine learning",
            collection="test-collection",
            top_k=5,
            hybrid=True,
            alpha=0.7,
        )

        assert response.total_found > 0


class TestSearchResult:
    """Tests for SearchResult dataclass."""

    @pytest.mark.asyncio
    async def test_search_result_fields(self, search_engine_with_data: SearchEngine):
        """Test that search results have all required fields."""
        response = await search_engine_with_data.search(
            query="test",
            collection="test-collection",
            top_k=1,
        )

        if response.results:
            result = response.results[0]
            assert hasattr(result, "id")
            assert hasattr(result, "content")
            assert hasattr(result, "score")
            assert hasattr(result, "metadata")

            assert isinstance(result.id, str)
            assert isinstance(result.content, str)
            assert isinstance(result.score, float)
            assert isinstance(result.metadata, dict)
