"""Collection management API endpoints."""

from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, status

from neural_search.api.schemas import (
    CollectionCreateRequest,
    CollectionListResponse,
    CollectionResponse,
    CollectionStatsResponse,
)
from neural_search.config import get_settings
from neural_search.core.embeddings import get_embedding_model
from neural_search.core.search_engine import SearchEngine, get_search_engine

router = APIRouter(prefix="/collections", tags=["collections"])


@router.post(
    "",
    response_model=CollectionResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Create collection",
    description="Create a new collection for storing documents.",
)
async def create_collection(
    request: CollectionCreateRequest,
    search_engine: Annotated[SearchEngine, Depends(get_search_engine)],
) -> CollectionResponse:
    """Create a new collection."""
    # Check if collection already exists
    if await search_engine.vector_store.collection_exists(request.name):
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"Collection '{request.name}' already exists",
        )

    # Get dimension from embedding model if not specified
    embedding_model = get_embedding_model()
    dimension = request.dimension or embedding_model.dimension

    # Create collection
    success = await search_engine.create_collection(
        name=request.name,
        dimension=dimension,
    )

    if not success:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to create collection",
        )

    return CollectionResponse(
        name=request.name,
        dimension=dimension,
        distance_metric=request.distance_metric,
        count=0,
    )


@router.get(
    "",
    response_model=CollectionListResponse,
    summary="List collections",
    description="List all available collections.",
)
async def list_collections(
    search_engine: Annotated[SearchEngine, Depends(get_search_engine)],
) -> CollectionListResponse:
    """List all collections."""
    collections = await search_engine.list_collections()

    return CollectionListResponse(
        collections=collections,
        count=len(collections),
    )


@router.get(
    "/{name}",
    response_model=CollectionStatsResponse,
    summary="Get collection stats",
    description="Get statistics for a collection.",
)
async def get_collection_stats(
    name: str,
    search_engine: Annotated[SearchEngine, Depends(get_search_engine)],
) -> CollectionStatsResponse:
    """Get collection statistics."""
    if not await search_engine.vector_store.collection_exists(name):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Collection '{name}' not found",
        )

    stats = await search_engine.get_collection_stats(name)

    return CollectionStatsResponse(
        name=stats.get("name", name),
        dimension=stats.get("dimension", 0),
        distance_metric=stats.get("distance_metric", "cosine"),
        count=stats.get("count", 0),
        indexed_count=stats.get("indexed_count"),
        bm25=stats.get("bm25"),
    )


@router.delete(
    "/{name}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Delete collection",
    description="Delete a collection and all its documents.",
)
async def delete_collection(
    name: str,
    search_engine: Annotated[SearchEngine, Depends(get_search_engine)],
) -> None:
    """Delete a collection."""
    if not await search_engine.vector_store.collection_exists(name):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Collection '{name}' not found",
        )

    success = await search_engine.delete_collection(name)

    if not success:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to delete collection",
        )
