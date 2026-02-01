"""Main API router combining all endpoints."""

from fastapi import APIRouter

from neural_search.api.collections import router as collections_router
from neural_search.api.documents import router as documents_router
from neural_search.api.search import router as search_router

router = APIRouter()

# Include sub-routers
router.include_router(documents_router)
router.include_router(search_router)
router.include_router(collections_router)
