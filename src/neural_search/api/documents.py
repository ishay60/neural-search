"""Document API endpoints."""

import uuid
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Query, status

from neural_search.api.schemas import (
    DocumentResponse,
    DocumentsCreateRequest,
    DocumentsCreateResponse,
    DocumentsDeleteRequest,
    DocumentsDeleteResponse,
    JobStatus,
)
from neural_search.config import get_settings
from neural_search.core.search_engine import SearchEngine, get_search_engine
from neural_search.utils.metrics import get_metrics
from neural_search.workers.tasks import delete_documents_task, index_documents_task

router = APIRouter(prefix="/documents", tags=["documents"])


@router.post(
    "",
    response_model=DocumentsCreateResponse,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Index documents",
    description="Add documents to a collection. Supports sync and async modes.",
)
async def create_documents(
    request: DocumentsCreateRequest,
    search_engine: Annotated[SearchEngine, Depends(get_search_engine)],
    async_mode: bool = Query(default=False, description="Process asynchronously"),
) -> DocumentsCreateResponse:
    """Index documents into a collection."""
    metrics = get_metrics()

    # Check collection exists
    if not await search_engine.vector_store.collection_exists(request.collection):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Collection '{request.collection}' not found",
        )

    # Prepare documents with IDs
    documents = []
    for i, doc in enumerate(request.documents):
        doc_id = doc.id or str(uuid.uuid4())
        documents.append({
            "id": doc_id,
            "content": doc.content,
            "metadata": doc.metadata,
        })

    if async_mode:
        # Queue for async processing
        task = index_documents_task.delay(
            collection=request.collection,
            documents=documents,
        )
        metrics.tasks_queued.labels(task_type="index").inc()

        return DocumentsCreateResponse(
            status="accepted",
            job_id=task.id,
            documents_count=len(documents),
        )
    else:
        # Process synchronously
        doc_ids = await search_engine.index_documents(
            collection=request.collection,
            documents=documents,
        )

        return DocumentsCreateResponse(
            status="completed",
            documents_count=len(doc_ids),
            document_ids=doc_ids,
        )


@router.get(
    "/{collection}/{doc_id}",
    response_model=DocumentResponse,
    summary="Get document",
    description="Retrieve a document by ID.",
)
async def get_document(
    collection: str,
    doc_id: str,
    search_engine: Annotated[SearchEngine, Depends(get_search_engine)],
) -> DocumentResponse:
    """Get a document by ID."""
    doc = await search_engine.get_document(collection, doc_id)

    if doc is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Document '{doc_id}' not found in collection '{collection}'",
        )

    return DocumentResponse(
        id=doc["id"],
        content=doc["content"],
        metadata=doc.get("metadata", {}),
    )


@router.delete(
    "",
    response_model=DocumentsDeleteResponse,
    summary="Delete documents",
    description="Delete documents from a collection.",
)
async def delete_documents(
    request: DocumentsDeleteRequest,
    search_engine: Annotated[SearchEngine, Depends(get_search_engine)],
    async_mode: bool = Query(default=False, description="Process asynchronously"),
) -> DocumentsDeleteResponse:
    """Delete documents from a collection."""
    metrics = get_metrics()

    # Check collection exists
    if not await search_engine.vector_store.collection_exists(request.collection):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Collection '{request.collection}' not found",
        )

    if async_mode:
        # Queue for async processing
        task = delete_documents_task.delay(
            collection=request.collection,
            doc_ids=request.document_ids,
        )
        metrics.tasks_queued.labels(task_type="delete").inc()

        return DocumentsDeleteResponse(
            status="accepted",
            deleted_count=len(request.document_ids),
            job_id=task.id,
        )
    else:
        # Process synchronously
        await search_engine.delete_documents(
            collection=request.collection,
            doc_ids=request.document_ids,
        )

        return DocumentsDeleteResponse(
            status="completed",
            deleted_count=len(request.document_ids),
        )


@router.get(
    "/jobs/{job_id}",
    response_model=JobStatus,
    summary="Get job status",
    description="Check the status of an async job.",
)
async def get_job_status(job_id: str) -> JobStatus:
    """Get the status of an async job."""
    from celery.result import AsyncResult
    from neural_search.workers.celery_app import celery_app

    result = AsyncResult(job_id, app=celery_app)

    status_map = {
        "PENDING": "pending",
        "STARTED": "running",
        "PROGRESS": "running",
        "SUCCESS": "completed",
        "FAILURE": "failed",
        "REVOKED": "cancelled",
    }

    job_status = status_map.get(result.status, "unknown")
    progress = None
    error = None
    task_result = None

    if result.status == "PROGRESS":
        progress = result.info.get("percent", 0) / 100.0

    if result.status == "SUCCESS":
        task_result = result.result

    if result.status == "FAILURE":
        error = str(result.result)

    return JobStatus(
        job_id=job_id,
        status=job_status,
        progress=progress,
        result=task_result,
        error=error,
    )
