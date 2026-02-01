"""Abstract base class for vector stores."""

from abc import ABC, abstractmethod
from functools import lru_cache
from typing import Any

import numpy as np

from neural_search.config import get_settings


class VectorStore(ABC):
    """Abstract base class for vector storage backends."""

    @abstractmethod
    async def create_collection(
        self,
        name: str,
        dimension: int,
        distance_metric: str = "cosine",
    ) -> bool:
        """Create a new collection.

        Args:
            name: Collection name
            dimension: Vector dimension
            distance_metric: Distance metric (cosine, euclidean, dot)

        Returns:
            True if created successfully
        """
        pass

    @abstractmethod
    async def delete_collection(self, name: str) -> bool:
        """Delete a collection.

        Args:
            name: Collection name

        Returns:
            True if deleted successfully
        """
        pass

    @abstractmethod
    async def list_collections(self) -> list[str]:
        """List all collections.

        Returns:
            List of collection names
        """
        pass

    @abstractmethod
    async def collection_exists(self, name: str) -> bool:
        """Check if collection exists.

        Args:
            name: Collection name

        Returns:
            True if collection exists
        """
        pass

    @abstractmethod
    async def get_collection_stats(self, name: str) -> dict[str, Any]:
        """Get statistics for a collection.

        Args:
            name: Collection name

        Returns:
            Dictionary with collection statistics
        """
        pass

    @abstractmethod
    async def upsert(
        self,
        collection: str,
        vectors: list[dict[str, Any]],
    ) -> bool:
        """Insert or update vectors.

        Args:
            collection: Collection name
            vectors: List of dicts with 'id', 'embedding', 'content', 'metadata'

        Returns:
            True if successful
        """
        pass

    @abstractmethod
    async def delete(
        self,
        collection: str,
        ids: list[str],
    ) -> bool:
        """Delete vectors by ID.

        Args:
            collection: Collection name
            ids: List of vector IDs to delete

        Returns:
            True if successful
        """
        pass

    @abstractmethod
    async def get(
        self,
        collection: str,
        id: str,
    ) -> dict[str, Any] | None:
        """Get a vector by ID.

        Args:
            collection: Collection name
            id: Vector ID

        Returns:
            Vector dict or None if not found
        """
        pass

    @abstractmethod
    async def search(
        self,
        collection: str,
        query_embedding: np.ndarray,
        top_k: int = 10,
        filters: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """Search for similar vectors.

        Args:
            collection: Collection name
            query_embedding: Query vector
            top_k: Number of results to return
            filters: Optional metadata filters

        Returns:
            List of results with 'id', 'score', 'content', 'metadata'
        """
        pass


@lru_cache
def get_vector_store() -> VectorStore:
    """Get configured vector store instance."""
    settings = get_settings()

    if settings.vector_store_type == "qdrant":
        from neural_search.storage.qdrant import QdrantStore
        return QdrantStore(
            host=settings.qdrant_host,
            port=settings.qdrant_port,
            api_key=settings.qdrant_api_key,
        )
    else:
        from neural_search.storage.faiss_store import FAISSStore
        return FAISSStore()
