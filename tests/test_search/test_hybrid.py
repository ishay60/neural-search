"""Tests for hybrid search."""

import pytest

from neural_search.core.hybrid import HybridSearcher


class TestHybridSearcher:
    """Tests for HybridSearcher class."""

    def test_tokenize(self):
        """Test tokenization."""
        tokens = HybridSearcher.tokenize("Hello, World! This is a TEST.")

        assert "hello" in tokens
        assert "world" in tokens
        assert "test" in tokens
        assert "," not in tokens
        assert "!" not in tokens

    def test_index_documents(self, hybrid_searcher: HybridSearcher):
        """Test indexing documents."""
        documents = [
            "Machine learning is a subset of AI.",
            "Deep learning uses neural networks.",
            "Natural language processing is NLP.",
        ]

        hybrid_searcher.index_documents(documents, ["doc1", "doc2", "doc3"])

        stats = hybrid_searcher.get_stats()
        assert stats["num_documents"] == 3
        assert stats["has_index"] is True

    def test_bm25_search(self, hybrid_searcher: HybridSearcher):
        """Test BM25 search."""
        documents = [
            "Machine learning enables computers to learn from data.",
            "Deep learning is a type of machine learning.",
            "Python is a programming language.",
        ]

        hybrid_searcher.index_documents(documents, ["doc1", "doc2", "doc3"])

        results = hybrid_searcher.bm25_search("machine learning", top_k=3)

        assert len(results) > 0
        # Documents about machine learning should rank higher
        top_ids = [r[0] for r in results]
        assert "doc1" in top_ids or "doc2" in top_ids

    def test_bm25_search_empty_index(self, hybrid_searcher: HybridSearcher):
        """Test BM25 search with empty index."""
        results = hybrid_searcher.bm25_search("test query")

        assert results == []

    def test_normalize_scores(self):
        """Test score normalization."""
        scores = [("doc1", 10.0), ("doc2", 5.0), ("doc3", 0.0)]

        normalized = HybridSearcher.normalize_scores(scores)

        assert normalized[0] == ("doc1", 1.0)  # Highest score
        assert normalized[2] == ("doc3", 0.0)  # Lowest score
        assert 0.0 <= normalized[1][1] <= 1.0  # Middle score in range

    def test_normalize_scores_equal(self):
        """Test score normalization with equal scores."""
        scores = [("doc1", 5.0), ("doc2", 5.0), ("doc3", 5.0)]

        normalized = HybridSearcher.normalize_scores(scores)

        # All should be 1.0 when scores are equal
        for _, score in normalized:
            assert score == 1.0

    def test_normalize_scores_empty(self):
        """Test score normalization with empty list."""
        normalized = HybridSearcher.normalize_scores([])
        assert normalized == []

    def test_hybrid_search(self, hybrid_searcher: HybridSearcher):
        """Test hybrid search combining dense and sparse."""
        documents = [
            "Machine learning is a subset of AI.",
            "Deep learning uses neural networks.",
            "Python is a programming language.",
        ]

        hybrid_searcher.index_documents(documents, ["doc1", "doc2", "doc3"])

        # Simulated dense results
        dense_results = [
            ("doc1", 0.9),
            ("doc2", 0.8),
            ("doc3", 0.3),
        ]

        results = hybrid_searcher.hybrid_search(
            query="machine learning neural networks",
            dense_results=dense_results,
            top_k=3,
            alpha=0.5,
        )

        assert len(results) <= 3
        # Results should be doc_id, score tuples
        assert all(isinstance(r, tuple) and len(r) == 2 for r in results)

    def test_add_documents(self, hybrid_searcher: HybridSearcher):
        """Test adding documents to existing index."""
        hybrid_searcher.index_documents(["doc 1", "doc 2"], ["id1", "id2"])

        assert hybrid_searcher.get_stats()["num_documents"] == 2

        hybrid_searcher.add_documents(["doc 3", "doc 4"], ["id3", "id4"])

        assert hybrid_searcher.get_stats()["num_documents"] == 4

    def test_remove_documents(self, hybrid_searcher: HybridSearcher):
        """Test removing documents from index."""
        hybrid_searcher.index_documents(
            ["doc 1", "doc 2", "doc 3"],
            ["id1", "id2", "id3"],
        )

        hybrid_searcher.remove_documents(["id2"])

        assert hybrid_searcher.get_stats()["num_documents"] == 2
        assert hybrid_searcher.get_document("id2") is None
        assert hybrid_searcher.get_document("id1") is not None

    def test_get_document(self, hybrid_searcher: HybridSearcher):
        """Test getting document by ID."""
        hybrid_searcher.index_documents(
            ["Test document content"],
            ["test-id"],
        )

        doc = hybrid_searcher.get_document("test-id")
        assert doc == "Test document content"

        missing = hybrid_searcher.get_document("nonexistent")
        assert missing is None

    def test_reciprocal_rank_fusion(self, hybrid_searcher: HybridSearcher):
        """Test Reciprocal Rank Fusion."""
        rankings = [
            ["doc1", "doc2", "doc3"],  # First ranking
            ["doc2", "doc1", "doc4"],  # Second ranking
            ["doc1", "doc4", "doc2"],  # Third ranking
        ]

        results = hybrid_searcher.reciprocal_rank_fusion(rankings, k=60, top_k=3)

        assert len(results) <= 3
        # doc1 appears at rank 1 in two lists, should be high
        top_ids = [r[0] for r in results]
        assert "doc1" in top_ids[:2]


class TestHybridSearcherParameters:
    """Tests for HybridSearcher parameters."""

    def test_custom_bm25_parameters(self):
        """Test custom BM25 k1 and b parameters."""
        searcher = HybridSearcher(k1=2.0, b=0.5, alpha=0.7)

        assert searcher.k1 == 2.0
        assert searcher.b == 0.5
        assert searcher.alpha == 0.7

    def test_alpha_affects_hybrid_results(self):
        """Test that alpha parameter affects hybrid search results."""
        searcher = HybridSearcher(alpha=0.9)

        documents = [
            "Machine learning algorithms",
            "Deep learning neural networks",
            "Python programming basics",
        ]
        searcher.index_documents(documents, ["doc1", "doc2", "doc3"])

        dense_results = [
            ("doc3", 0.95),  # High dense score for doc3
            ("doc2", 0.5),
            ("doc1", 0.3),
        ]

        # High alpha = favor dense results
        results_high_alpha = searcher.hybrid_search(
            query="machine learning",  # BM25 favors doc1, doc2
            dense_results=dense_results,
            alpha=0.9,
        )

        # Low alpha = favor sparse (BM25) results
        results_low_alpha = searcher.hybrid_search(
            query="machine learning",
            dense_results=dense_results,
            alpha=0.1,
        )

        # With high alpha, doc3 (high dense score) should rank higher
        # With low alpha, doc1/doc2 (BM25 matches) should rank higher
        assert results_high_alpha != results_low_alpha
