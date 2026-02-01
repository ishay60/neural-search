"""Celery tasks for async document processing."""

import asyncio
import logging
from typing import Any

from celery import shared_task
from celery.exceptions import SoftTimeLimitExceeded

from neural_search.config import get_settings
from neural_search.core.embeddings import get_embedding_model
from neural_search.storage import get_vector_store
from neural_search.utils.metrics import get_metrics

logger = logging.getLogger(__name__)


def run_async(coro):
    """Run async coroutine in sync context."""
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


@shared_task(
    bind=True,
    autoretry_for=(Exception,),
    retry_backoff=True,
    retry_kwargs={"max_retries": 3},
)
def index_documents_task(
    self,
    collection: str,
    documents: list[dict[str, Any]],
    batch_size: int = 32,
) -> dict[str, Any]:
    """Index documents asynchronously.

    Args:
        collection: Collection name
        documents: List of documents to index
        batch_size: Batch size for embedding generation

    Returns:
        Result dict with indexed document IDs
    """
    metrics = get_metrics()
    metrics.tasks_queued.labels(task_type="index").inc()

    try:
        with metrics.measure_task("index"):
            embedding_model = get_embedding_model()
            vector_store = get_vector_store()

            # Extract content for embedding
            contents = [doc["content"] for doc in documents]
            doc_ids = [doc.get("id", str(i)) for i, doc in enumerate(documents)]

            # Generate embeddings
            embeddings = embedding_model.encode_documents(
                contents,
                batch_size=batch_size,
                show_progress=len(contents) > 100,
            )

            # Prepare vectors
            vectors = []
            for doc_id, embedding, doc in zip(doc_ids, embeddings, documents):
                vectors.append({
                    "id": doc_id,
                    "embedding": embedding,
                    "content": doc["content"],
                    "metadata": doc.get("metadata", {}),
                })

            # Store in vector store
            run_async(vector_store.upsert(collection, vectors))

            # Update metrics
            metrics.documents_indexed.labels(collection=collection).inc(len(documents))
            metrics.tasks_completed.labels(task_type="index", status="success").inc()

            return {
                "status": "completed",
                "collection": collection,
                "document_ids": doc_ids,
                "count": len(doc_ids),
            }

    except SoftTimeLimitExceeded:
        metrics.tasks_completed.labels(task_type="index", status="timeout").inc()
        raise

    except Exception as e:
        metrics.tasks_completed.labels(task_type="index", status="error").inc()
        logger.error(f"Error indexing documents: {e}")
        raise


@shared_task(
    bind=True,
    autoretry_for=(Exception,),
    retry_backoff=True,
    retry_kwargs={"max_retries": 3},
)
def delete_documents_task(
    self,
    collection: str,
    doc_ids: list[str],
) -> dict[str, Any]:
    """Delete documents asynchronously.

    Args:
        collection: Collection name
        doc_ids: List of document IDs to delete

    Returns:
        Result dict with deletion status
    """
    metrics = get_metrics()
    metrics.tasks_queued.labels(task_type="delete").inc()

    try:
        with metrics.measure_task("delete"):
            vector_store = get_vector_store()

            # Delete from vector store
            run_async(vector_store.delete(collection, doc_ids))

            # Update metrics
            metrics.documents_deleted.labels(collection=collection).inc(len(doc_ids))
            metrics.tasks_completed.labels(task_type="delete", status="success").inc()

            return {
                "status": "completed",
                "collection": collection,
                "deleted_ids": doc_ids,
                "count": len(doc_ids),
            }

    except Exception as e:
        metrics.tasks_completed.labels(task_type="delete", status="error").inc()
        logger.error(f"Error deleting documents: {e}")
        raise


@shared_task(
    bind=True,
    autoretry_for=(Exception,),
    retry_backoff=True,
    retry_kwargs={"max_retries": 3},
)
def batch_index_task(
    self,
    collection: str,
    file_path: str,
    batch_size: int = 100,
) -> dict[str, Any]:
    """Batch index documents from a file.

    Args:
        collection: Collection name
        file_path: Path to JSON/JSONL file with documents
        batch_size: Number of documents per batch

    Returns:
        Result dict with indexing statistics
    """
    import json

    metrics = get_metrics()
    metrics.tasks_queued.labels(task_type="batch_index").inc()

    try:
        with metrics.measure_task("batch_index"):
            embedding_model = get_embedding_model()
            vector_store = get_vector_store()

            total_indexed = 0
            total_errors = 0

            # Read file (supports JSON and JSONL)
            documents = []
            with open(file_path, "r") as f:
                if file_path.endswith(".jsonl"):
                    for line in f:
                        if line.strip():
                            documents.append(json.loads(line))
                else:
                    data = json.load(f)
                    if isinstance(data, list):
                        documents = data
                    elif isinstance(data, dict) and "documents" in data:
                        documents = data["documents"]

            # Process in batches
            for i in range(0, len(documents), batch_size):
                batch = documents[i:i + batch_size]

                try:
                    # Generate embeddings
                    contents = [doc["content"] for doc in batch]
                    doc_ids = [doc.get("id", str(i + j)) for j, doc in enumerate(batch)]

                    embeddings = embedding_model.encode_documents(
                        contents,
                        batch_size=32,
                    )

                    # Prepare vectors
                    vectors = []
                    for doc_id, embedding, doc in zip(doc_ids, embeddings, batch):
                        vectors.append({
                            "id": doc_id,
                            "embedding": embedding,
                            "content": doc["content"],
                            "metadata": doc.get("metadata", {}),
                        })

                    # Store
                    run_async(vector_store.upsert(collection, vectors))
                    total_indexed += len(batch)

                    # Update progress
                    self.update_state(
                        state="PROGRESS",
                        meta={
                            "current": total_indexed,
                            "total": len(documents),
                            "percent": int(total_indexed / len(documents) * 100),
                        },
                    )

                except Exception as e:
                    logger.error(f"Error processing batch: {e}")
                    total_errors += len(batch)

            # Update metrics
            metrics.documents_indexed.labels(collection=collection).inc(total_indexed)
            metrics.tasks_completed.labels(task_type="batch_index", status="success").inc()

            return {
                "status": "completed",
                "collection": collection,
                "total_documents": len(documents),
                "indexed": total_indexed,
                "errors": total_errors,
            }

    except Exception as e:
        metrics.tasks_completed.labels(task_type="batch_index", status="error").inc()
        logger.error(f"Error in batch indexing: {e}")
        raise


@shared_task
def cleanup_expired_jobs() -> dict[str, Any]:
    """Clean up expired job results and temporary files.

    Returns:
        Cleanup statistics
    """
    logger.info("Running cleanup of expired jobs")
    # In a real implementation, this would clean up:
    # - Expired Redis keys
    # - Temporary files
    # - Orphaned data
    return {"status": "completed", "message": "Cleanup finished"}
