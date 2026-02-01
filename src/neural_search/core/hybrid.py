"""Hybrid search combining dense vectors and sparse BM25."""

import logging
import re
from collections import defaultdict
from typing import Any

import numpy as np
from rank_bm25 import BM25Okapi

from neural_search.config import get_settings

logger = logging.getLogger(__name__)


class HybridSearcher:
    """Hybrid searcher combining dense vector search with sparse BM25."""

    def __init__(
        self,
        k1: float = 1.5,
        b: float = 0.75,
        alpha: float = 0.5,
    ):
        """Initialize hybrid searcher.

        Args:
            k1: BM25 k1 parameter (term frequency saturation)
            b: BM25 b parameter (length normalization)
            alpha: Weight for dense scores (1-alpha for sparse scores)
        """
        self.k1 = k1
        self.b = b
        self.alpha = alpha
        self._bm25: BM25Okapi | None = None
        self._documents: list[str] = []
        self._doc_ids: list[str] = []

    @staticmethod
    def tokenize(text: str) -> list[str]:
        """Simple tokenization for BM25.

        Args:
            text: Input text

        Returns:
            List of tokens
        """
        # Lowercase and extract words
        text = text.lower()
        tokens = re.findall(r"\b\w+\b", text)
        return tokens

    def index_documents(
        self,
        documents: list[str],
        doc_ids: list[str] | None = None,
    ) -> None:
        """Index documents for BM25 search.

        Args:
            documents: List of document texts
            doc_ids: Optional list of document IDs
        """
        self._documents = documents
        self._doc_ids = doc_ids or [str(i) for i in range(len(documents))]

        # Tokenize documents
        tokenized_docs = [self.tokenize(doc) for doc in documents]

        # Create BM25 index
        self._bm25 = BM25Okapi(tokenized_docs, k1=self.k1, b=self.b)
        logger.info(f"Indexed {len(documents)} documents for BM25")

    def add_documents(
        self,
        documents: list[str],
        doc_ids: list[str] | None = None,
    ) -> None:
        """Add documents to existing index.

        Args:
            documents: List of document texts to add
            doc_ids: Optional list of document IDs
        """
        if doc_ids is None:
            start_id = len(self._doc_ids)
            doc_ids = [str(i) for i in range(start_id, start_id + len(documents))]

        self._documents.extend(documents)
        self._doc_ids.extend(doc_ids)

        # Re-index all documents (BM25 doesn't support incremental updates)
        self.index_documents(self._documents, self._doc_ids)

    def remove_documents(self, doc_ids: list[str]) -> None:
        """Remove documents from index.

        Args:
            doc_ids: List of document IDs to remove
        """
        ids_to_remove = set(doc_ids)
        new_docs = []
        new_ids = []

        for doc, doc_id in zip(self._documents, self._doc_ids):
            if doc_id not in ids_to_remove:
                new_docs.append(doc)
                new_ids.append(doc_id)

        self._documents = new_docs
        self._doc_ids = new_ids

        if self._documents:
            self.index_documents(self._documents, self._doc_ids)
        else:
            self._bm25 = None

    def bm25_search(
        self,
        query: str,
        top_k: int = 10,
    ) -> list[tuple[str, float]]:
        """Perform BM25 search.

        Args:
            query: Search query
            top_k: Number of results to return

        Returns:
            List of (doc_id, score) tuples
        """
        if self._bm25 is None:
            return []

        tokenized_query = self.tokenize(query)
        scores = self._bm25.get_scores(tokenized_query)

        # Get top-k indices
        top_indices = np.argsort(scores)[-top_k:][::-1]

        results = []
        for idx in top_indices:
            if scores[idx] > 0:  # Only include non-zero scores
                results.append((self._doc_ids[idx], float(scores[idx])))

        return results

    @staticmethod
    def normalize_scores(scores: list[tuple[str, float]]) -> list[tuple[str, float]]:
        """Normalize scores to [0, 1] range using min-max normalization.

        Args:
            scores: List of (doc_id, score) tuples

        Returns:
            List of (doc_id, normalized_score) tuples
        """
        if not scores:
            return []

        min_score = min(s for _, s in scores)
        max_score = max(s for _, s in scores)

        if max_score == min_score:
            return [(doc_id, 1.0) for doc_id, _ in scores]

        normalized = []
        for doc_id, score in scores:
            norm_score = (score - min_score) / (max_score - min_score)
            normalized.append((doc_id, norm_score))

        return normalized

    def hybrid_search(
        self,
        query: str,
        dense_results: list[tuple[str, float]],
        top_k: int = 10,
        alpha: float | None = None,
    ) -> list[tuple[str, float]]:
        """Combine dense and sparse search results.

        Uses Reciprocal Rank Fusion (RRF) for combining results.

        Args:
            query: Search query (for BM25)
            dense_results: Dense search results as (doc_id, score) tuples
            top_k: Number of results to return
            alpha: Weight for dense scores (overrides instance alpha)

        Returns:
            List of (doc_id, combined_score) tuples
        """
        alpha = alpha if alpha is not None else self.alpha

        # Get BM25 results
        sparse_results = self.bm25_search(query, top_k=len(dense_results) * 2)

        # Normalize scores
        dense_normalized = self.normalize_scores(dense_results)
        sparse_normalized = self.normalize_scores(sparse_results)

        # Combine scores
        combined_scores: dict[str, float] = defaultdict(float)

        for doc_id, score in dense_normalized:
            combined_scores[doc_id] += alpha * score

        for doc_id, score in sparse_normalized:
            combined_scores[doc_id] += (1 - alpha) * score

        # Sort by combined score
        sorted_results = sorted(
            combined_scores.items(),
            key=lambda x: x[1],
            reverse=True,
        )

        return sorted_results[:top_k]

    def reciprocal_rank_fusion(
        self,
        rankings: list[list[str]],
        k: int = 60,
        top_k: int = 10,
    ) -> list[tuple[str, float]]:
        """Combine multiple rankings using Reciprocal Rank Fusion.

        Args:
            rankings: List of document ID rankings (each is a list of doc_ids)
            k: RRF constant (default 60)
            top_k: Number of results to return

        Returns:
            List of (doc_id, rrf_score) tuples
        """
        rrf_scores: dict[str, float] = defaultdict(float)

        for ranking in rankings:
            for rank, doc_id in enumerate(ranking, start=1):
                rrf_scores[doc_id] += 1.0 / (k + rank)

        # Sort by RRF score
        sorted_results = sorted(
            rrf_scores.items(),
            key=lambda x: x[1],
            reverse=True,
        )

        return sorted_results[:top_k]

    def get_document(self, doc_id: str) -> str | None:
        """Get document by ID.

        Args:
            doc_id: Document ID

        Returns:
            Document text or None if not found
        """
        try:
            idx = self._doc_ids.index(doc_id)
            return self._documents[idx]
        except ValueError:
            return None

    def get_stats(self) -> dict[str, Any]:
        """Get indexing statistics.

        Returns:
            Dictionary with index statistics
        """
        return {
            "num_documents": len(self._documents),
            "has_index": self._bm25 is not None,
            "k1": self.k1,
            "b": self.b,
            "alpha": self.alpha,
        }
