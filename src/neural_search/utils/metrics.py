"""Prometheus metrics for Neural Search."""

import time
from contextlib import contextmanager
from functools import lru_cache
from typing import Generator

from prometheus_client import Counter, Gauge, Histogram, Info


class MetricsCollector:
    """Prometheus metrics collector for search operations."""

    def __init__(self) -> None:
        """Initialize metrics."""
        # Application info
        self.app_info = Info("neural_search", "Neural Search application information")
        self.app_info.info({
            "version": "0.1.0",
        })

        # Request metrics
        self.requests_total = Counter(
            "neural_search_requests_total",
            "Total number of requests",
            ["method", "endpoint", "status"],
        )

        self.request_latency = Histogram(
            "neural_search_request_latency_seconds",
            "Request latency in seconds",
            ["method", "endpoint"],
            buckets=(0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0),
        )

        # Embedding metrics
        self.embeddings_generated = Counter(
            "neural_search_embeddings_generated_total",
            "Total number of embeddings generated",
            ["model"],
        )

        self.embedding_latency = Histogram(
            "neural_search_embedding_latency_seconds",
            "Embedding generation latency in seconds",
            ["model", "batch_size"],
            buckets=(0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0),
        )

        # Search metrics
        self.searches_total = Counter(
            "neural_search_searches_total",
            "Total number of searches",
            ["collection", "search_type"],
        )

        self.search_latency = Histogram(
            "neural_search_search_latency_seconds",
            "Search latency in seconds",
            ["collection", "search_type"],
            buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0),
        )

        self.search_results_count = Histogram(
            "neural_search_results_count",
            "Number of search results returned",
            ["collection"],
            buckets=(0, 1, 5, 10, 25, 50, 100),
        )

        # Document metrics
        self.documents_indexed = Counter(
            "neural_search_documents_indexed_total",
            "Total number of documents indexed",
            ["collection"],
        )

        self.documents_deleted = Counter(
            "neural_search_documents_deleted_total",
            "Total number of documents deleted",
            ["collection"],
        )

        # Collection metrics
        self.collection_size = Gauge(
            "neural_search_collection_size",
            "Number of documents in collection",
            ["collection"],
        )

        # Cache metrics
        self.cache_hits = Counter(
            "neural_search_cache_hits_total",
            "Total cache hits",
            ["cache_type"],
        )

        self.cache_misses = Counter(
            "neural_search_cache_misses_total",
            "Total cache misses",
            ["cache_type"],
        )

        # Rate limiting metrics
        self.rate_limit_exceeded = Counter(
            "neural_search_rate_limit_exceeded_total",
            "Total rate limit exceeded events",
            ["client_ip"],
        )

        # Worker metrics
        self.tasks_queued = Counter(
            "neural_search_tasks_queued_total",
            "Total tasks queued",
            ["task_type"],
        )

        self.tasks_completed = Counter(
            "neural_search_tasks_completed_total",
            "Total tasks completed",
            ["task_type", "status"],
        )

        self.task_latency = Histogram(
            "neural_search_task_latency_seconds",
            "Task processing latency in seconds",
            ["task_type"],
            buckets=(0.1, 0.5, 1.0, 5.0, 10.0, 30.0, 60.0, 300.0),
        )

    @contextmanager
    def measure_request(
        self,
        method: str,
        endpoint: str,
    ) -> Generator[None, None, None]:
        """Context manager to measure request latency.

        Args:
            method: HTTP method
            endpoint: API endpoint
        """
        start = time.perf_counter()
        try:
            yield
        finally:
            duration = time.perf_counter() - start
            self.request_latency.labels(method=method, endpoint=endpoint).observe(duration)

    @contextmanager
    def measure_embedding(
        self,
        model: str,
        batch_size: int,
    ) -> Generator[None, None, None]:
        """Context manager to measure embedding generation latency.

        Args:
            model: Model name
            batch_size: Batch size
        """
        start = time.perf_counter()
        try:
            yield
        finally:
            duration = time.perf_counter() - start
            self.embedding_latency.labels(
                model=model,
                batch_size=str(batch_size),
            ).observe(duration)

    @contextmanager
    def measure_search(
        self,
        collection: str,
        search_type: str,
    ) -> Generator[None, None, None]:
        """Context manager to measure search latency.

        Args:
            collection: Collection name
            search_type: Type of search (dense, sparse, hybrid)
        """
        start = time.perf_counter()
        try:
            yield
        finally:
            duration = time.perf_counter() - start
            self.search_latency.labels(
                collection=collection,
                search_type=search_type,
            ).observe(duration)

    @contextmanager
    def measure_task(self, task_type: str) -> Generator[None, None, None]:
        """Context manager to measure task processing latency.

        Args:
            task_type: Type of task
        """
        start = time.perf_counter()
        try:
            yield
        finally:
            duration = time.perf_counter() - start
            self.task_latency.labels(task_type=task_type).observe(duration)


@lru_cache
def get_metrics() -> MetricsCollector:
    """Get cached MetricsCollector instance."""
    return MetricsCollector()
