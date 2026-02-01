"""Celery workers for async processing."""

from neural_search.workers.celery_app import celery_app
from neural_search.workers.tasks import (
    index_documents_task,
    delete_documents_task,
    batch_index_task,
)

__all__ = [
    "celery_app",
    "index_documents_task",
    "delete_documents_task",
    "batch_index_task",
]
