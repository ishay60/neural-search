"""Celery application configuration."""

from celery import Celery

from neural_search.config import get_settings

settings = get_settings()

celery_app = Celery(
    "neural_search",
    broker=settings.celery_broker_url,
    backend=settings.celery_result_backend,
    include=["neural_search.workers.tasks"],
)

# Celery configuration
celery_app.conf.update(
    # Task settings
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,

    # Task execution settings
    task_acks_late=True,
    task_reject_on_worker_lost=True,
    task_time_limit=3600,  # 1 hour max per task
    task_soft_time_limit=3300,  # Soft limit at 55 minutes

    # Worker settings
    worker_prefetch_multiplier=1,
    worker_concurrency=4,

    # Result settings
    result_expires=86400,  # Results expire after 24 hours

    # Rate limiting
    task_annotations={
        "neural_search.workers.tasks.index_documents_task": {
            "rate_limit": "100/m",
        },
        "neural_search.workers.tasks.batch_index_task": {
            "rate_limit": "10/m",
        },
    },

    # Retry settings
    task_default_retry_delay=60,
    task_max_retries=3,

    # Routing
    task_routes={
        "neural_search.workers.tasks.index_documents_task": {"queue": "indexing"},
        "neural_search.workers.tasks.delete_documents_task": {"queue": "indexing"},
        "neural_search.workers.tasks.batch_index_task": {"queue": "batch"},
    },

    # Beat schedule (for periodic tasks)
    beat_schedule={
        "cleanup-expired-jobs": {
            "task": "neural_search.workers.tasks.cleanup_expired_jobs",
            "schedule": 3600.0,  # Every hour
        },
    },
)
