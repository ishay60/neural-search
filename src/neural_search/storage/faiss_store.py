"""FAISS-based in-memory vector store."""

import logging
from dataclasses import dataclass, field
from typing import Any

import faiss
import numpy as np

from neural_search.storage.vector_store import VectorStore

logger = logging.getLogger(__name__)


@dataclass
class FAISSCollection:
    """Container for a FAISS index and associated data."""

    dimension: int
    distance_metric: str
    index: faiss.Index
    id_to_idx: dict[str, int] = field(default_factory=dict)
    idx_to_id: dict[int, str] = field(default_factory=dict)
    documents: dict[str, dict[str, Any]] = field(default_factory=dict)
    next_idx: int = 0


class FAISSStore(VectorStore):
    """FAISS-based in-memory vector store implementation."""

    def __init__(self):
        """Initialize FAISS store."""
        self._collections: dict[str, FAISSCollection] = {}

    def _create_index(
        self,
        dimension: int,
        distance_metric: str = "cosine",
    ) -> faiss.Index:
        """Create a FAISS index.

        Args:
            dimension: Vector dimension
            distance_metric: Distance metric (cosine, euclidean, dot)

        Returns:
            FAISS index
        """
        if distance_metric == "cosine":
            # For cosine similarity, we use inner product with normalized vectors
            index = faiss.IndexFlatIP(dimension)
        elif distance_metric == "euclidean":
            index = faiss.IndexFlatL2(dimension)
        elif distance_metric == "dot":
            index = faiss.IndexFlatIP(dimension)
        else:
            raise ValueError(f"Unknown distance metric: {distance_metric}")

        return index

    async def create_collection(
        self,
        name: str,
        dimension: int,
        distance_metric: str = "cosine",
    ) -> bool:
        """Create a new collection."""
        if name in self._collections:
            logger.warning(f"Collection '{name}' already exists")
            return False

        index = self._create_index(dimension, distance_metric)
        self._collections[name] = FAISSCollection(
            dimension=dimension,
            distance_metric=distance_metric,
            index=index,
        )

        logger.info(f"Created collection '{name}' with dimension {dimension}")
        return True

    async def delete_collection(self, name: str) -> bool:
        """Delete a collection."""
        if name not in self._collections:
            logger.warning(f"Collection '{name}' does not exist")
            return False

        del self._collections[name]
        logger.info(f"Deleted collection '{name}'")
        return True

    async def list_collections(self) -> list[str]:
        """List all collections."""
        return list(self._collections.keys())

    async def collection_exists(self, name: str) -> bool:
        """Check if collection exists."""
        return name in self._collections

    async def get_collection_stats(self, name: str) -> dict[str, Any]:
        """Get statistics for a collection."""
        if name not in self._collections:
            return {}

        collection = self._collections[name]
        return {
            "name": name,
            "dimension": collection.dimension,
            "distance_metric": collection.distance_metric,
            "count": collection.index.ntotal,
        }

    async def upsert(
        self,
        collection: str,
        vectors: list[dict[str, Any]],
    ) -> bool:
        """Insert or update vectors."""
        if collection not in self._collections:
            raise ValueError(f"Collection '{collection}' does not exist")

        coll = self._collections[collection]

        # Process vectors
        new_embeddings = []
        new_indices = []

        for vec in vectors:
            vec_id = vec["id"]
            embedding = vec["embedding"]

            # If vector already exists, we need to handle update
            # FAISS doesn't support true updates, so we track separately
            if vec_id in coll.id_to_idx:
                # For updates, we add to the index but mark old one as invalid
                # In a production system, you'd periodically rebuild the index
                pass

            # Add new mapping
            idx = coll.next_idx
            coll.next_idx += 1

            coll.id_to_idx[vec_id] = idx
            coll.idx_to_id[idx] = vec_id

            # Store document data
            coll.documents[vec_id] = {
                "content": vec.get("content", ""),
                "metadata": vec.get("metadata", {}),
            }

            # Ensure embedding is the right shape and normalized for cosine
            emb = np.array(embedding, dtype=np.float32)
            if coll.distance_metric == "cosine":
                # Normalize for cosine similarity
                emb = emb / np.linalg.norm(emb)

            new_embeddings.append(emb)
            new_indices.append(idx)

        # Add to index
        if new_embeddings:
            embeddings_array = np.vstack(new_embeddings).astype(np.float32)
            coll.index.add(embeddings_array)

        return True

    async def delete(
        self,
        collection: str,
        ids: list[str],
    ) -> bool:
        """Delete vectors by ID.

        Note: FAISS doesn't support true deletion, so we just remove from mappings.
        The vectors remain in the index but won't be returned in searches.
        """
        if collection not in self._collections:
            raise ValueError(f"Collection '{collection}' does not exist")

        coll = self._collections[collection]

        for vec_id in ids:
            if vec_id in coll.id_to_idx:
                idx = coll.id_to_idx[vec_id]
                del coll.id_to_idx[vec_id]
                if idx in coll.idx_to_id:
                    del coll.idx_to_id[idx]
                if vec_id in coll.documents:
                    del coll.documents[vec_id]

        return True

    async def get(
        self,
        collection: str,
        id: str,
    ) -> dict[str, Any] | None:
        """Get a vector by ID."""
        if collection not in self._collections:
            return None

        coll = self._collections[collection]

        if id not in coll.documents:
            return None

        doc = coll.documents[id]
        return {
            "id": id,
            "content": doc["content"],
            "metadata": doc["metadata"],
        }

    def _matches_filters(
        self,
        metadata: dict[str, Any],
        filters: dict[str, Any],
    ) -> bool:
        """Check if metadata matches filters.

        Supports basic operators:
        - Direct equality: {"field": "value"}
        - $eq: {"field": {"$eq": "value"}}
        - $ne: {"field": {"$ne": "value"}}
        - $gt, $gte, $lt, $lte: {"field": {"$gt": 10}}
        - $in: {"field": {"$in": ["a", "b"]}}
        - $nin: {"field": {"$nin": ["a", "b"]}}
        """
        for key, condition in filters.items():
            if key not in metadata:
                return False

            value = metadata[key]

            if isinstance(condition, dict):
                for op, op_value in condition.items():
                    if op == "$eq":
                        if value != op_value:
                            return False
                    elif op == "$ne":
                        if value == op_value:
                            return False
                    elif op == "$gt":
                        if not value > op_value:
                            return False
                    elif op == "$gte":
                        if not value >= op_value:
                            return False
                    elif op == "$lt":
                        if not value < op_value:
                            return False
                    elif op == "$lte":
                        if not value <= op_value:
                            return False
                    elif op == "$in":
                        if value not in op_value:
                            return False
                    elif op == "$nin":
                        if value in op_value:
                            return False
            else:
                # Direct equality
                if value != condition:
                    return False

        return True

    async def search(
        self,
        collection: str,
        query_embedding: np.ndarray,
        top_k: int = 10,
        filters: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """Search for similar vectors."""
        if collection not in self._collections:
            raise ValueError(f"Collection '{collection}' does not exist")

        coll = self._collections[collection]

        if coll.index.ntotal == 0:
            return []

        # Prepare query
        query = query_embedding.astype(np.float32)
        if coll.distance_metric == "cosine":
            query = query / np.linalg.norm(query)
        query = query.reshape(1, -1)

        # Search with extra results if filtering (to account for filtered out results)
        search_k = top_k * 10 if filters else top_k
        search_k = min(search_k, coll.index.ntotal)

        distances, indices = coll.index.search(query, search_k)

        # Process results
        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx == -1:  # FAISS returns -1 for missing results
                continue

            vec_id = coll.idx_to_id.get(idx)
            if vec_id is None:
                continue  # Deleted vector

            doc = coll.documents.get(vec_id)
            if doc is None:
                continue

            # Apply filters
            if filters and not self._matches_filters(doc["metadata"], filters):
                continue

            # Convert distance to score
            if coll.distance_metric == "cosine":
                score = float(dist)  # Inner product of normalized vectors = cosine sim
            elif coll.distance_metric == "euclidean":
                score = 1.0 / (1.0 + float(dist))  # Convert L2 distance to similarity
            else:
                score = float(dist)

            results.append({
                "id": vec_id,
                "score": score,
                "content": doc["content"],
                "metadata": doc["metadata"],
            })

            if len(results) >= top_k:
                break

        return results
