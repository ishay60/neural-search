"""Search API endpoints."""

from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, status

from neural_search.api.schemas import SearchRequest, SearchResponse, SearchResultItem
from neural_search.core.search_engine import SearchEngine, get_search_engine
from neural_search.utils.cache import Cache, get_cache
from neural_search.utils.metrics import get_metrics

router = APIRouter(prefix="/search", tags=["search"])


@router.post(
    "",
    response_model=SearchResponse,
    summary="Semantic search",
    description="Search for similar documents using semantic search.",
)
async def search(
    request: SearchRequest,
    search_engine: Annotated[SearchEngine, Depends(get_search_engine)],
) -> SearchResponse:
    """Perform semantic search on a collection."""
    metrics = get_metrics()

    # Check collection exists
    if not await search_engine.vector_store.collection_exists(request.collection):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Collection '{request.collection}' not found",
        )

    # Check cache (disabled for now - would need async Redis connection)
    # cache = get_cache()
    # cache_key = cache.make_search_key(
    #     query=request.query,
    #     collection=request.collection,
    #     top_k=request.top_k,
    #     filters=request.filters,
    # )
    # cached = await cache.get(cache_key)
    # if cached:
    #     metrics.cache_hits.labels(cache_type="search").inc()
    #     return SearchResponse(**cached)
    # metrics.cache_misses.labels(cache_type="search").inc()

    # Perform search
    response = await search_engine.search(
        query=request.query,
        collection=request.collection,
        top_k=request.top_k,
        filters=request.filters,
        hybrid=request.hybrid,
        rerank=request.rerank,
        alpha=request.alpha,
    )

    # Build response
    results = []
    for r in response.results:
        item = SearchResultItem(
            id=r.id,
            content=r.content,
            score=r.score,
            metadata=r.metadata if request.include_metadata else None,
        )
        results.append(item)

    search_response = SearchResponse(
        results=results,
        took_ms=response.took_ms,
        total_found=response.total_found,
    )

    # Cache result (disabled for now)
    # await cache.set(cache_key, search_response.model_dump())

    return search_response


@router.post(
    "/batch",
    response_model=list[SearchResponse],
    summary="Batch search",
    description="Perform multiple searches in a single request.",
)
async def batch_search(
    requests: list[SearchRequest],
    search_engine: Annotated[SearchEngine, Depends(get_search_engine)],
) -> list[SearchResponse]:
    """Perform batch semantic search."""
    if len(requests) > 100:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Maximum 100 queries per batch",
        )

    responses = []
    for req in requests:
        # Check collection exists
        if not await search_engine.vector_store.collection_exists(req.collection):
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Collection '{req.collection}' not found",
            )

        response = await search_engine.search(
            query=req.query,
            collection=req.collection,
            top_k=req.top_k,
            filters=req.filters,
            hybrid=req.hybrid,
            rerank=req.rerank,
            alpha=req.alpha,
        )

        results = [
            SearchResultItem(
                id=r.id,
                content=r.content,
                score=r.score,
                metadata=r.metadata if req.include_metadata else None,
            )
            for r in response.results
        ]

        responses.append(SearchResponse(
            results=results,
            took_ms=response.took_ms,
            total_found=response.total_found,
        ))

    return responses
