"""Pydantic schemas for API requests and responses."""

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field


# Document schemas
class DocumentInput(BaseModel):
    """Input schema for a document."""

    id: str | None = Field(None, description="Document ID (auto-generated if not provided)")
    content: str = Field(..., min_length=1, description="Document text content")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Document metadata")


class DocumentsCreateRequest(BaseModel):
    """Request to create/index documents."""

    documents: list[DocumentInput] = Field(..., min_length=1, max_length=1000)
    collection: str = Field(..., min_length=1, max_length=256)


class DocumentsCreateResponse(BaseModel):
    """Response from document creation."""

    status: str
    job_id: str | None = None
    documents_count: int
    document_ids: list[str] | None = None


class DocumentResponse(BaseModel):
    """Response with document data."""

    id: str
    content: str
    metadata: dict[str, Any]


class DocumentsDeleteRequest(BaseModel):
    """Request to delete documents."""

    document_ids: list[str] = Field(..., min_length=1)
    collection: str


class DocumentsDeleteResponse(BaseModel):
    """Response from document deletion."""

    status: str
    deleted_count: int
    job_id: str | None = None


# Search schemas
class SearchRequest(BaseModel):
    """Request for semantic search."""

    query: str = Field(..., min_length=1, description="Search query text")
    collection: str = Field(..., min_length=1, description="Collection to search")
    top_k: int = Field(default=10, ge=1, le=100, description="Number of results to return")
    filters: dict[str, Any] | None = Field(None, description="Metadata filters")
    include_metadata: bool = Field(default=True, description="Include metadata in results")
    hybrid: bool = Field(default=False, description="Enable hybrid search (vector + BM25)")
    rerank: bool = Field(default=False, description="Enable cross-encoder reranking")
    alpha: float | None = Field(
        None,
        ge=0.0,
        le=1.0,
        description="Weight for dense vs sparse scores (0-1, higher = more dense)",
    )


class SearchResultItem(BaseModel):
    """Single search result."""

    id: str
    content: str
    score: float
    metadata: dict[str, Any] | None = None


class SearchResponse(BaseModel):
    """Response from search query."""

    results: list[SearchResultItem]
    took_ms: float
    total_found: int


# Collection schemas
class CollectionCreateRequest(BaseModel):
    """Request to create a collection."""

    name: str = Field(..., min_length=1, max_length=256, pattern=r"^[a-zA-Z0-9_-]+$")
    embedding_model: str | None = Field(None, description="Embedding model name")
    dimension: int | None = Field(None, ge=1, le=4096, description="Vector dimension")
    distance_metric: str = Field(
        default="cosine",
        description="Distance metric",
        pattern=r"^(cosine|euclidean|dot)$",
    )


class CollectionResponse(BaseModel):
    """Response with collection info."""

    name: str
    dimension: int
    distance_metric: str
    count: int
    indexed_count: int | None = None
    status: str | None = None


class CollectionListResponse(BaseModel):
    """Response listing collections."""

    collections: list[str]
    count: int


class CollectionStatsResponse(BaseModel):
    """Response with collection statistics."""

    name: str
    dimension: int
    distance_metric: str
    count: int
    indexed_count: int | None = None
    bm25: dict[str, Any] | None = None


# Job schemas
class JobStatus(BaseModel):
    """Status of an async job."""

    job_id: str
    status: str  # pending, running, completed, failed
    progress: float | None = None
    result: dict[str, Any] | None = None
    error: str | None = None
    created_at: datetime | None = None
    completed_at: datetime | None = None


# Health schemas
class HealthResponse(BaseModel):
    """Health check response."""

    status: str
    version: str
    embedding_model: str
    vector_store: str
    collections_count: int


# Error schemas
class ErrorResponse(BaseModel):
    """Error response."""

    detail: str
    error_code: str | None = None
