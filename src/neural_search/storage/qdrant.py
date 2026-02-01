"""Qdrant vector store implementation."""

import logging
from typing import Any

import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.http import models
from qdrant_client.http.exceptions import UnexpectedResponse

from neural_search.storage.vector_store import VectorStore

logger = logging.getLogger(__name__)


class QdrantStore(VectorStore):
    """Qdrant vector store implementation for persistent storage."""

    def __init__(
        self,
        host: str = "localhost",
        port: int = 6333,
        api_key: str | None = None,
        prefer_grpc: bool = True,
    ):
        """Initialize Qdrant store.

        Args:
            host: Qdrant server host
            port: Qdrant server port
            api_key: Optional API key for authentication
            prefer_grpc: Whether to prefer gRPC over HTTP
        """
        self.host = host
        self.port = port
        self._client: QdrantClient | None = None
        self._api_key = api_key
        self._prefer_grpc = prefer_grpc

    @property
    def client(self) -> QdrantClient:
        """Get Qdrant client, creating if necessary."""
        if self._client is None:
            self._client = QdrantClient(
                host=self.host,
                port=self.port,
                api_key=self._api_key,
                prefer_grpc=self._prefer_grpc,
            )
            logger.info(f"Connected to Qdrant at {self.host}:{self.port}")
        return self._client

    def _get_distance(self, metric: str) -> models.Distance:
        """Convert distance metric string to Qdrant Distance enum."""
        mapping = {
            "cosine": models.Distance.COSINE,
            "euclidean": models.Distance.EUCLID,
            "dot": models.Distance.DOT,
        }
        return mapping.get(metric, models.Distance.COSINE)

    async def create_collection(
        self,
        name: str,
        dimension: int,
        distance_metric: str = "cosine",
    ) -> bool:
        """Create a new collection."""
        try:
            self.client.create_collection(
                collection_name=name,
                vectors_config=models.VectorParams(
                    size=dimension,
                    distance=self._get_distance(distance_metric),
                ),
            )
            logger.info(f"Created Qdrant collection '{name}' with dimension {dimension}")
            return True
        except UnexpectedResponse as e:
            if "already exists" in str(e):
                logger.warning(f"Collection '{name}' already exists")
                return False
            raise

    async def delete_collection(self, name: str) -> bool:
        """Delete a collection."""
        try:
            self.client.delete_collection(collection_name=name)
            logger.info(f"Deleted Qdrant collection '{name}'")
            return True
        except UnexpectedResponse as e:
            if "not found" in str(e).lower():
                logger.warning(f"Collection '{name}' not found")
                return False
            raise

    async def list_collections(self) -> list[str]:
        """List all collections."""
        collections = self.client.get_collections()
        return [c.name for c in collections.collections]

    async def collection_exists(self, name: str) -> bool:
        """Check if collection exists."""
        try:
            self.client.get_collection(collection_name=name)
            return True
        except UnexpectedResponse:
            return False

    async def get_collection_stats(self, name: str) -> dict[str, Any]:
        """Get statistics for a collection."""
        try:
            info = self.client.get_collection(collection_name=name)
            return {
                "name": name,
                "dimension": info.config.params.vectors.size,  # type: ignore
                "distance_metric": str(info.config.params.vectors.distance),  # type: ignore
                "count": info.points_count,
                "indexed_count": info.indexed_vectors_count,
                "status": str(info.status),
            }
        except UnexpectedResponse:
            return {}

    async def upsert(
        self,
        collection: str,
        vectors: list[dict[str, Any]],
    ) -> bool:
        """Insert or update vectors."""
        points = []
        for vec in vectors:
            # Store content in payload
            payload = {
                "content": vec.get("content", ""),
                **vec.get("metadata", {}),
            }

            points.append(models.PointStruct(
                id=vec["id"],
                vector=vec["embedding"].tolist() if isinstance(vec["embedding"], np.ndarray) else vec["embedding"],
                payload=payload,
            ))

        # Upsert in batches
        batch_size = 100
        for i in range(0, len(points), batch_size):
            batch = points[i:i + batch_size]
            self.client.upsert(
                collection_name=collection,
                points=batch,
            )

        return True

    async def delete(
        self,
        collection: str,
        ids: list[str],
    ) -> bool:
        """Delete vectors by ID."""
        self.client.delete(
            collection_name=collection,
            points_selector=models.PointIdsList(points=ids),
        )
        return True

    async def get(
        self,
        collection: str,
        id: str,
    ) -> dict[str, Any] | None:
        """Get a vector by ID."""
        try:
            results = self.client.retrieve(
                collection_name=collection,
                ids=[id],
                with_payload=True,
            )

            if not results:
                return None

            point = results[0]
            payload = point.payload or {}
            content = payload.pop("content", "")

            return {
                "id": str(point.id),
                "content": content,
                "metadata": payload,
            }
        except UnexpectedResponse:
            return None

    def _build_filter(self, filters: dict[str, Any]) -> models.Filter:
        """Build Qdrant filter from filter dict.

        Supports:
        - Direct equality: {"field": "value"}
        - $eq: {"field": {"$eq": "value"}}
        - $ne: {"field": {"$ne": "value"}}
        - $gt, $gte, $lt, $lte: {"field": {"$gt": 10}}
        - $in: {"field": {"$in": ["a", "b"]}}
        """
        conditions = []

        for key, condition in filters.items():
            if isinstance(condition, dict):
                for op, value in condition.items():
                    if op == "$eq":
                        conditions.append(
                            models.FieldCondition(
                                key=key,
                                match=models.MatchValue(value=value),
                            )
                        )
                    elif op == "$ne":
                        conditions.append(
                            models.FieldCondition(
                                key=key,
                                match=models.MatchExcept(**{"except": [value]}),
                            )
                        )
                    elif op == "$gt":
                        conditions.append(
                            models.FieldCondition(
                                key=key,
                                range=models.Range(gt=value),
                            )
                        )
                    elif op == "$gte":
                        conditions.append(
                            models.FieldCondition(
                                key=key,
                                range=models.Range(gte=value),
                            )
                        )
                    elif op == "$lt":
                        conditions.append(
                            models.FieldCondition(
                                key=key,
                                range=models.Range(lt=value),
                            )
                        )
                    elif op == "$lte":
                        conditions.append(
                            models.FieldCondition(
                                key=key,
                                range=models.Range(lte=value),
                            )
                        )
                    elif op == "$in":
                        conditions.append(
                            models.FieldCondition(
                                key=key,
                                match=models.MatchAny(any=value),
                            )
                        )
            else:
                # Direct equality
                conditions.append(
                    models.FieldCondition(
                        key=key,
                        match=models.MatchValue(value=condition),
                    )
                )

        return models.Filter(must=conditions)

    async def search(
        self,
        collection: str,
        query_embedding: np.ndarray,
        top_k: int = 10,
        filters: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """Search for similar vectors."""
        query_filter = self._build_filter(filters) if filters else None

        results = self.client.search(
            collection_name=collection,
            query_vector=query_embedding.tolist(),
            limit=top_k,
            query_filter=query_filter,
            with_payload=True,
        )

        search_results = []
        for point in results:
            payload = point.payload or {}
            content = payload.pop("content", "")

            search_results.append({
                "id": str(point.id),
                "score": point.score,
                "content": content,
                "metadata": payload,
            })

        return search_results
