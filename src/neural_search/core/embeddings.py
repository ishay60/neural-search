"""Embedding generation using sentence-transformers."""

import logging
from functools import lru_cache
from typing import Sequence

import numpy as np
from sentence_transformers import SentenceTransformer

from neural_search.config import get_settings
from neural_search.utils.metrics import get_metrics

logger = logging.getLogger(__name__)


class EmbeddingModel:
    """Sentence transformer based embedding model."""

    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        device: str = "cpu",
        max_seq_length: int = 512,
    ):
        """Initialize embedding model.

        Args:
            model_name: Name of the sentence-transformers model
            device: Device to run model on (cpu, cuda, mps)
            max_seq_length: Maximum sequence length for input text
        """
        self.model_name = model_name
        self.device = device
        self.max_seq_length = max_seq_length
        self._model: SentenceTransformer | None = None
        self._dimension: int | None = None

    def load(self) -> None:
        """Load the model into memory."""
        if self._model is None:
            logger.info(f"Loading embedding model: {self.model_name}")
            self._model = SentenceTransformer(self.model_name, device=self.device)
            self._model.max_seq_length = self.max_seq_length
            # Get embedding dimension from model
            self._dimension = self._model.get_sentence_embedding_dimension()
            logger.info(
                f"Loaded model with dimension {self._dimension} on {self.device}"
            )

    @property
    def model(self) -> SentenceTransformer:
        """Get the model, loading if necessary."""
        if self._model is None:
            self.load()
        return self._model  # type: ignore

    @property
    def dimension(self) -> int:
        """Get embedding dimension."""
        if self._dimension is None:
            self.load()
        return self._dimension  # type: ignore

    def encode(
        self,
        texts: str | Sequence[str],
        batch_size: int = 32,
        normalize: bool = True,
        show_progress: bool = False,
    ) -> np.ndarray:
        """Generate embeddings for text(s).

        Args:
            texts: Single text or sequence of texts to encode
            batch_size: Batch size for encoding
            normalize: Whether to L2 normalize embeddings
            show_progress: Whether to show progress bar

        Returns:
            Numpy array of embeddings with shape (n_texts, dimension)
        """
        if isinstance(texts, str):
            texts = [texts]

        metrics = get_metrics()

        with metrics.measure_embedding(self.model_name, len(texts)):
            embeddings = self.model.encode(
                list(texts),
                batch_size=batch_size,
                normalize_embeddings=normalize,
                show_progress_bar=show_progress,
                convert_to_numpy=True,
            )

        metrics.embeddings_generated.labels(model=self.model_name).inc(len(texts))

        return embeddings

    def encode_query(self, query: str, normalize: bool = True) -> np.ndarray:
        """Generate embedding for a search query.

        Args:
            query: Search query text
            normalize: Whether to L2 normalize embedding

        Returns:
            Numpy array of embedding with shape (dimension,)
        """
        embeddings = self.encode([query], normalize=normalize)
        return embeddings[0]

    def encode_documents(
        self,
        documents: Sequence[str],
        batch_size: int = 32,
        normalize: bool = True,
        show_progress: bool = False,
    ) -> np.ndarray:
        """Generate embeddings for documents.

        Args:
            documents: Sequence of document texts
            batch_size: Batch size for encoding
            normalize: Whether to L2 normalize embeddings
            show_progress: Whether to show progress bar

        Returns:
            Numpy array of embeddings with shape (n_documents, dimension)
        """
        return self.encode(
            documents,
            batch_size=batch_size,
            normalize=normalize,
            show_progress=show_progress,
        )

    def similarity(
        self,
        query_embedding: np.ndarray,
        document_embeddings: np.ndarray,
    ) -> np.ndarray:
        """Compute cosine similarity between query and documents.

        Args:
            query_embedding: Query embedding with shape (dimension,)
            document_embeddings: Document embeddings with shape (n_docs, dimension)

        Returns:
            Similarity scores with shape (n_docs,)
        """
        # Ensure query is 2D for matrix multiplication
        if query_embedding.ndim == 1:
            query_embedding = query_embedding.reshape(1, -1)

        # Compute cosine similarity (assumes normalized embeddings)
        similarities = np.dot(document_embeddings, query_embedding.T).flatten()
        return similarities


@lru_cache
def get_embedding_model() -> EmbeddingModel:
    """Get cached EmbeddingModel instance."""
    settings = get_settings()
    model = EmbeddingModel(
        model_name=settings.embedding_model,
        device=settings.embedding_device,
        max_seq_length=settings.embedding_max_seq_length,
    )
    return model
