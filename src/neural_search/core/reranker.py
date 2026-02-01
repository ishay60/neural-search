"""Cross-encoder based re-ranking for search results."""

import logging
from functools import lru_cache
from typing import Sequence

from sentence_transformers import CrossEncoder

from neural_search.config import get_settings

logger = logging.getLogger(__name__)


class CrossEncoderReranker:
    """Cross-encoder based re-ranker for improving search result quality."""

    def __init__(
        self,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        device: str = "cpu",
        max_length: int = 512,
    ):
        """Initialize cross-encoder reranker.

        Args:
            model_name: Name of the cross-encoder model
            device: Device to run model on
            max_length: Maximum sequence length
        """
        self.model_name = model_name
        self.device = device
        self.max_length = max_length
        self._model: CrossEncoder | None = None

    def load(self) -> None:
        """Load the cross-encoder model."""
        if self._model is None:
            logger.info(f"Loading cross-encoder model: {self.model_name}")
            self._model = CrossEncoder(
                self.model_name,
                max_length=self.max_length,
                device=self.device,
            )
            logger.info("Cross-encoder model loaded")

    @property
    def model(self) -> CrossEncoder:
        """Get the model, loading if necessary."""
        if self._model is None:
            self.load()
        return self._model  # type: ignore

    def rerank(
        self,
        query: str,
        documents: Sequence[str],
        top_k: int | None = None,
    ) -> list[tuple[int, float]]:
        """Re-rank documents based on query relevance.

        Args:
            query: Search query
            documents: List of document texts to rerank
            top_k: Number of top results to return (None for all)

        Returns:
            List of (original_index, score) tuples, sorted by score descending
        """
        if not documents:
            return []

        # Create query-document pairs
        pairs = [[query, doc] for doc in documents]

        # Get relevance scores from cross-encoder
        scores = self.model.predict(pairs, show_progress_bar=False)

        # Create index-score pairs
        indexed_scores = [(i, float(score)) for i, score in enumerate(scores)]

        # Sort by score descending
        indexed_scores.sort(key=lambda x: x[1], reverse=True)

        # Return top_k if specified
        if top_k is not None:
            indexed_scores = indexed_scores[:top_k]

        return indexed_scores

    def score(self, query: str, document: str) -> float:
        """Get relevance score for a single query-document pair.

        Args:
            query: Search query
            document: Document text

        Returns:
            Relevance score
        """
        score = self.model.predict([[query, document]], show_progress_bar=False)
        return float(score[0])


@lru_cache
def get_reranker() -> CrossEncoderReranker:
    """Get cached CrossEncoderReranker instance."""
    settings = get_settings()
    reranker = CrossEncoderReranker(
        model_name=settings.rerank_model,
        device=settings.embedding_device,
    )
    return reranker
