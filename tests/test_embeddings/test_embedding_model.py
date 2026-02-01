"""Tests for embedding model."""

import numpy as np
import pytest

from tests.conftest import MockEmbeddingModel


class TestMockEmbeddingModel:
    """Tests for mock embedding model (used for testing)."""

    def test_encode_single_text(self):
        """Test encoding a single text."""
        model = MockEmbeddingModel(dimension=384)

        texts = ["This is a test sentence."]
        embeddings = model.encode(texts)

        assert embeddings.shape == (1, 384)
        assert embeddings.dtype == np.float32

    def test_encode_multiple_texts(self):
        """Test encoding multiple texts."""
        model = MockEmbeddingModel(dimension=384)

        texts = [
            "First sentence.",
            "Second sentence.",
            "Third sentence.",
        ]
        embeddings = model.encode(texts)

        assert embeddings.shape == (3, 384)

    def test_encode_normalized(self):
        """Test that embeddings are normalized."""
        model = MockEmbeddingModel(dimension=384)

        embeddings = model.encode(["Test text"], normalize=True)

        # Check L2 norm is approximately 1
        norm = np.linalg.norm(embeddings[0])
        assert abs(norm - 1.0) < 1e-5

    def test_encode_unnormalized(self):
        """Test that embeddings can be unnormalized."""
        model = MockEmbeddingModel(dimension=384)

        embeddings = model.encode(["Test text"], normalize=False)

        # Check L2 norm is not necessarily 1
        norm = np.linalg.norm(embeddings[0])
        # Random vectors typically don't have unit norm
        assert norm > 0

    def test_encode_query(self):
        """Test encoding a query."""
        model = MockEmbeddingModel(dimension=384)

        embedding = model.encode_query("test query")

        assert embedding.shape == (384,)
        assert embedding.dtype == np.float32

    def test_encode_documents(self):
        """Test encoding documents."""
        model = MockEmbeddingModel(dimension=384)

        documents = ["doc 1", "doc 2", "doc 3"]
        embeddings = model.encode_documents(documents)

        assert embeddings.shape == (3, 384)

    def test_deterministic_encoding(self):
        """Test that encoding is deterministic for same input."""
        model = MockEmbeddingModel(dimension=384)

        text = "Test text for determinism"
        emb1 = model.encode([text])
        emb2 = model.encode([text])

        np.testing.assert_array_equal(emb1, emb2)

    def test_different_inputs_different_embeddings(self):
        """Test that different inputs produce different embeddings."""
        model = MockEmbeddingModel(dimension=384)

        emb1 = model.encode(["First text"])
        emb2 = model.encode(["Second text"])

        assert not np.array_equal(emb1, emb2)

    def test_dimension_property(self):
        """Test dimension property."""
        model = MockEmbeddingModel(dimension=768)
        assert model.dimension == 768


class TestEmbeddingSimilarity:
    """Tests for embedding similarity calculations."""

    def test_cosine_similarity_normalized(self):
        """Test cosine similarity with normalized vectors."""
        model = MockEmbeddingModel(dimension=384)

        # Same text should have high similarity with itself
        text = "Machine learning is a subset of AI."
        emb1 = model.encode([text], normalize=True)
        emb2 = model.encode([text], normalize=True)

        similarity = np.dot(emb1[0], emb2[0])
        assert similarity == pytest.approx(1.0, abs=1e-5)

    def test_cosine_similarity_different_texts(self):
        """Test cosine similarity with different texts."""
        model = MockEmbeddingModel(dimension=384)

        emb1 = model.encode(["Text one"], normalize=True)
        emb2 = model.encode(["Text two"], normalize=True)

        similarity = np.dot(emb1[0], emb2[0])
        # Different texts should have lower similarity
        assert -1.0 <= similarity <= 1.0
