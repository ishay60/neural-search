"""Main FastAPI application for Neural Search."""

import logging
import time
from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

from neural_search import __version__
from neural_search.api import router
from neural_search.api.schemas import ErrorResponse, HealthResponse
from neural_search.config import get_settings
from neural_search.core.embeddings import get_embedding_model
from neural_search.storage import get_vector_store
from neural_search.utils.metrics import get_metrics

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Application lifespan handler."""
    logger.info("Starting Neural Search API...")

    # Pre-load embedding model
    settings = get_settings()
    embedding_model = get_embedding_model()
    embedding_model.load()
    logger.info(f"Loaded embedding model: {settings.embedding_model}")

    yield

    logger.info("Shutting down Neural Search API...")


# Create FastAPI app
settings = get_settings()

app = FastAPI(
    title="Neural Search API",
    description="Production-grade semantic search API with embeddings and vector search",
    version=__version__,
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
    lifespan=lifespan,
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Rate limiting middleware
@app.middleware("http")
async def rate_limit_middleware(request: Request, call_next):
    """Simple rate limiting middleware using in-memory tracking."""
    # In production, use Redis-based rate limiting
    metrics = get_metrics()

    # Track request
    response = await call_next(request)
    return response


# Request timing middleware
@app.middleware("http")
async def timing_middleware(request: Request, call_next):
    """Add request timing to responses."""
    start_time = time.perf_counter()

    response = await call_next(request)

    process_time = (time.perf_counter() - start_time) * 1000
    response.headers["X-Process-Time-Ms"] = f"{process_time:.2f}"

    return response


# Metrics middleware
@app.middleware("http")
async def metrics_middleware(request: Request, call_next):
    """Track request metrics."""
    metrics = get_metrics()
    method = request.method
    endpoint = request.url.path

    with metrics.measure_request(method, endpoint):
        response = await call_next(request)

    metrics.requests_total.labels(
        method=method,
        endpoint=endpoint,
        status=str(response.status_code),
    ).inc()

    return response


# Exception handlers
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Global exception handler."""
    logger.exception(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content=ErrorResponse(
            detail="Internal server error",
            error_code="INTERNAL_ERROR",
        ).model_dump(),
    )


# Health check endpoint
@app.get(
    "/health",
    response_model=HealthResponse,
    tags=["health"],
    summary="Health check",
)
async def health_check() -> HealthResponse:
    """Check API health status."""
    settings = get_settings()
    vector_store = get_vector_store()

    try:
        collections = await vector_store.list_collections()
        collections_count = len(collections)
    except Exception:
        collections_count = 0

    return HealthResponse(
        status="healthy",
        version=__version__,
        embedding_model=settings.embedding_model,
        vector_store=settings.vector_store_type,
        collections_count=collections_count,
    )


# Prometheus metrics endpoint
@app.get("/metrics", tags=["monitoring"])
async def metrics():
    """Prometheus metrics endpoint."""
    from starlette.responses import Response

    return Response(
        content=generate_latest(),
        media_type=CONTENT_TYPE_LATEST,
    )


# Include API router
app.include_router(router, prefix=settings.api_prefix)


# Root endpoint
@app.get("/", tags=["root"])
async def root():
    """Root endpoint with API information."""
    return {
        "name": "Neural Search API",
        "version": __version__,
        "docs": "/docs",
        "health": "/health",
        "metrics": "/metrics",
    }


def main():
    """Run the application with uvicorn."""
    import uvicorn

    settings = get_settings()
    uvicorn.run(
        "neural_search.main:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=settings.debug,
    )


if __name__ == "__main__":
    main()
