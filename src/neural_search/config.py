"""Configuration settings for Neural Search."""

from functools import lru_cache
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings with environment variable support."""

    model_config = SettingsConfigDict(
        env_prefix="NEURAL_SEARCH_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # API Settings
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    api_prefix: str = "/api/v1"
    debug: bool = False

    # Embedding Model Settings
    embedding_model: str = "all-MiniLM-L6-v2"
    embedding_device: str = "cpu"
    embedding_batch_size: int = 32
    embedding_max_seq_length: int = 512

    # Vector Store Settings
    vector_store_type: Literal["faiss", "qdrant"] = "faiss"
    vector_dimension: int = 384

    # Qdrant Settings
    qdrant_host: str = "localhost"
    qdrant_port: int = 6333
    qdrant_grpc_port: int = 6334
    qdrant_api_key: str | None = None

    # Redis Settings
    redis_url: str = "redis://localhost:6379/0"
    cache_ttl: int = 3600  # 1 hour

    # Celery Settings
    celery_broker_url: str = "redis://localhost:6379/1"
    celery_result_backend: str = "redis://localhost:6379/2"

    # Rate Limiting
    rate_limit_requests: int = 100
    rate_limit_window: int = 60  # seconds

    # Search Settings
    default_top_k: int = 10
    max_top_k: int = 100
    hybrid_alpha: float = Field(default=0.5, ge=0.0, le=1.0)  # Balance between dense and sparse

    # Re-ranking Settings
    rerank_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    rerank_top_k: int = 50  # Number of results to rerank

    # BM25 Settings
    bm25_k1: float = 1.5
    bm25_b: float = 0.75


@lru_cache
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()
