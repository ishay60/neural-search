"""Main search engine orchestrating all search components."""

import logging
import time
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any

import numpy as np

from neural_search.config import get_settings
from neural_search.core.embeddings import EmbeddingModel, get_embedding_model
from neural_search.core.hybrid import HybridSearcher
from neural_search.core.reranker import CrossEncoderReranker, get_reranker
from neural_search.storage import VectorStore, get_vector_store
from neural_search.utils.metrics import get_metrics

logger = logging.getLogger(__name__)


@dataclass
class SearchResult:
    """Search result with document and metadata."""

    id: str
    content: str
    score: float
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class SearchResponse:
    """Response from search operation."""

    results: list[SearchResult]
    took_ms: float
    total_found: int
    query: str
    collection: str


class SearchEngine:
    """Main search engine orchestrating embeddings, vector store, and reranking."""

    def __init__(
        self,
        embedding_model: EmbeddingModel,
        vector_store: VectorStore,
        reranker: CrossEncoderReranker | None = None,
    ):
        """Initialize search engine.

        Args:
            embedding_model: Embedding model for encoding queries and documents
            vector_store: Vector store for document storage and retrieval
            reranker: Optional cross-encoder reranker
        """
        self.embedding_model = embedding_model
        self.vector_store = vector_store
        self.reranker = reranker
        self._hybrid_searchers: dict[str, HybridSearcher] = {}

    def _get_hybrid_searcher(self, collection: str) -> HybridSearcher:
        """Get or create hybrid searcher for collection."""
        if collection not in self._hybrid_searchers:
            settings = get_settings()
            self._hybrid_searchers[collection] = HybridSearcher(
                k1=settings.bm25_k1,
                b=settings.bm25_b,
                alpha=settings.hybrid_alpha,
            )
        return self._hybrid_searchers[collection]

    async def create_collection(
        self,
        name: str,
        dimension: int | None = None,
    ) -> bool:
        """Create a new collection.

        Args:
            name: Collection name
            dimension: Vector dimension (uses model dimension if not specified)

        Returns:
            True if created successfully
        """
        if dimension is None:
            dimension = self.embedding_model.dimension

        return await self.vector_store.create_collection(name, dimension)

    async def delete_collection(self, name: str) -> bool:
        """Delete a collection.

        Args:
            name: Collection name

        Returns:
            True if deleted successfully
        """
        # Also clean up hybrid searcher
        if name in self._hybrid_searchers:
            del self._hybrid_searchers[name]

        return await self.vector_store.delete_collection(name)

    async def list_collections(self) -> list[str]:
        """List all collections.

        Returns:
            List of collection names
        """
        return await self.vector_store.list_collections()

    async def get_collection_stats(self, name: str) -> dict[str, Any]:
        """Get statistics for a collection.

        Args:
            name: Collection name

        Returns:
            Dictionary with collection statistics
        """
        stats = await self.vector_store.get_collection_stats(name)

        # Add hybrid searcher stats if available
        if name in self._hybrid_searchers:
            stats["bm25"] = self._hybrid_searchers[name].get_stats()

        return stats

    async def index_documents(
        self,
        collection: str,
        documents: list[dict[str, Any]],
        batch_size: int = 32,
    ) -> list[str]:
        """Index documents into a collection.

        Args:
            collection: Collection name
            documents: List of documents with 'id', 'content', and optional 'metadata'
            batch_size: Batch size for embedding generation

        Returns:
            List of indexed document IDs
        """
        metrics = get_metrics()

        # Extract content for embedding
        contents = [doc["content"] for doc in documents]
        doc_ids = [doc.get("id", str(i)) for i, doc in enumerate(documents)]

        # Generate embeddings
        embeddings = self.embedding_model.encode_documents(
            contents,
            batch_size=batch_size,
            show_progress=len(contents) > 100,
        )

        # Prepare vectors with metadata
        vectors = []
        for i, (doc_id, embedding, doc) in enumerate(zip(doc_ids, embeddings, documents)):
            vectors.append({
                "id": doc_id,
                "embedding": embedding,
                "content": doc["content"],
                "metadata": doc.get("metadata", {}),
            })

        # Store in vector store
        await self.vector_store.upsert(collection, vectors)

        # Update BM25 index
        hybrid_searcher = self._get_hybrid_searcher(collection)
        hybrid_searcher.add_documents(contents, doc_ids)

        # Update metrics
        metrics.documents_indexed.labels(collection=collection).inc(len(documents))
        metrics.collection_size.labels(collection=collection).set(
            (await self.get_collection_stats(collection)).get("count", 0)
        )

        logger.info(f"Indexed {len(documents)} documents to collection '{collection}'")
        return doc_ids

    async def delete_documents(
        self,
        collection: str,
        doc_ids: list[str],
    ) -> bool:
        """Delete documents from a collection.

        Args:
            collection: Collection name
            doc_ids: List of document IDs to delete

        Returns:
            True if deleted successfully
        """
        metrics = get_metrics()

        # Delete from vector store
        result = await self.vector_store.delete(collection, doc_ids)

        # Update BM25 index
        if collection in self._hybrid_searchers:
            self._hybrid_searchers[collection].remove_documents(doc_ids)

        # Update metrics
        metrics.documents_deleted.labels(collection=collection).inc(len(doc_ids))

        return result

    async def get_document(
        self,
        collection: str,
        doc_id: str,
    ) -> dict[str, Any] | None:
        """Get a document by ID.

        Args:
            collection: Collection name
            doc_id: Document ID

        Returns:
            Document dict or None if not found
        """
        return await self.vector_store.get(collection, doc_id)

    async def search(
        self,
        query: str,
        collection: str,
        top_k: int = 10,
        filters: dict[str, Any] | None = None,
        hybrid: bool = False,
        rerank: bool = False,
        alpha: float | None = None,
    ) -> SearchResponse:
        """Search for similar documents.

        Args:
            query: Search query text
            collection: Collection to search
            top_k: Number of results to return
            filters: Optional metadata filters
            hybrid: Enable hybrid search (vector + BM25)
            rerank: Enable cross-encoder reranking
            alpha: Weight for dense vs sparse scores (0-1, higher = more dense)

        Returns:
            SearchResponse with results and metadata
        """
        start_time = time.perf_counter()
        metrics = get_metrics()
        settings = get_settings()

        # Determine search type for metrics
        search_type = "dense"
        if hybrid:
            search_type = "hybrid"
        if rerank:
            search_type += "+rerank"

        with metrics.measure_search(collection, search_type):
            # Generate query embedding
            query_embedding = self.embedding_model.encode_query(query)

            # Determine how many results to fetch initially
            initial_k = top_k
            if rerank:
                initial_k = min(settings.rerank_top_k, top_k * 5)
            if hybrid:
                initial_k = max(initial_k, top_k * 2)

            # Vector search
            dense_results = await self.vector_store.search(
                collection=collection,
                query_embedding=query_embedding,
                top_k=initial_k,
                filters=filters,
            )

            # Convert to (id, score) tuples for processing
            result_tuples = [(r["id"], r["score"]) for r in dense_results]
            result_docs = {r["id"]: r for r in dense_results}

            # Hybrid search if enabled
            if hybrid:
                hybrid_searcher = self._get_hybrid_searcher(collection)
                if hybrid_searcher._bm25 is not None:
                    result_tuples = hybrid_searcher.hybrid_search(
                        query=query,
                        dense_results=result_tuples,
                        top_k=initial_k if rerank else top_k,
                        alpha=alpha,
                    )

            # Rerank if enabled
            if rerank and self.reranker is not None and result_tuples:
                # Get documents for reranking
                docs_to_rerank = []
                doc_id_order = []
                for doc_id, _ in result_tuples:
                    if doc_id in result_docs:
                        docs_to_rerank.append(result_docs[doc_id]["content"])
                        doc_id_order.append(doc_id)

                if docs_to_rerank:
                    reranked = self.reranker.rerank(
                        query=query,
                        documents=docs_to_rerank,
                        top_k=top_k,
                    )
                    result_tuples = [
                        (doc_id_order[idx], score)
                        for idx, score in reranked
                    ]

            # Build final results
            results = []
            for doc_id, score in result_tuples[:top_k]:
                if doc_id in result_docs:
                    doc = result_docs[doc_id]
                    results.append(SearchResult(
                        id=doc_id,
                        content=doc["content"],
                        score=score,
                        metadata=doc.get("metadata", {}),
                    ))

        # Calculate timing
        took_ms = (time.perf_counter() - start_time) * 1000

        # Update metrics
        metrics.searches_total.labels(
            collection=collection,
            search_type=search_type,
        ).inc()
        metrics.search_results_count.labels(collection=collection).observe(len(results))

        return SearchResponse(
            results=results,
            took_ms=took_ms,
            total_found=len(results),
            query=query,
            collection=collection,
        )


@lru_cache
def get_search_engine() -> SearchEngine:
    """Get cached SearchEngine instance."""
    embedding_model = get_embedding_model()
    vector_store = get_vector_store()
    reranker = get_reranker()

    return SearchEngine(
        embedding_model=embedding_model,
        vector_store=vector_store,
        reranker=reranker,
    )
