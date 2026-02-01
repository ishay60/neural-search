"""Redis caching layer for Neural Search."""

import hashlib
import json
import logging
from functools import lru_cache
from typing import Any

import redis.asyncio as redis

from neural_search.config import get_settings

logger = logging.getLogger(__name__)


class Cache:
    """Redis-based caching layer for embeddings and search results."""

    def __init__(self, redis_url: str, ttl: int = 3600):
        """Initialize cache with Redis connection.

        Args:
            redis_url: Redis connection URL
            ttl: Default time-to-live for cache entries in seconds
        """
        self.redis_url = redis_url
        self.ttl = ttl
        self._client: redis.Redis | None = None

    async def connect(self) -> None:
        """Establish Redis connection."""
        if self._client is None:
            self._client = redis.from_url(
                self.redis_url,
                encoding="utf-8",
                decode_responses=True,
            )
            logger.info("Connected to Redis cache")

    async def disconnect(self) -> None:
        """Close Redis connection."""
        if self._client:
            await self._client.close()
            self._client = None
            logger.info("Disconnected from Redis cache")

    @property
    def client(self) -> redis.Redis:
        """Get Redis client, raising error if not connected."""
        if self._client is None:
            raise RuntimeError("Cache not connected. Call connect() first.")
        return self._client

    @staticmethod
    def _hash_key(key: str) -> str:
        """Generate a hash for cache keys."""
        return hashlib.md5(key.encode()).hexdigest()

    async def get(self, key: str) -> Any | None:
        """Get value from cache.

        Args:
            key: Cache key

        Returns:
            Cached value or None if not found
        """
        try:
            hashed_key = self._hash_key(key)
            value = await self.client.get(hashed_key)
            if value:
                return json.loads(value)
            return None
        except Exception as e:
            logger.warning(f"Cache get error: {e}")
            return None

    async def set(self, key: str, value: Any, ttl: int | None = None) -> bool:
        """Set value in cache.

        Args:
            key: Cache key
            value: Value to cache (must be JSON serializable)
            ttl: Time-to-live in seconds (uses default if not specified)

        Returns:
            True if successful, False otherwise
        """
        try:
            hashed_key = self._hash_key(key)
            serialized = json.dumps(value)
            await self.client.setex(hashed_key, ttl or self.ttl, serialized)
            return True
        except Exception as e:
            logger.warning(f"Cache set error: {e}")
            return False

    async def delete(self, key: str) -> bool:
        """Delete value from cache.

        Args:
            key: Cache key

        Returns:
            True if deleted, False otherwise
        """
        try:
            hashed_key = self._hash_key(key)
            result = await self.client.delete(hashed_key)
            return result > 0
        except Exception as e:
            logger.warning(f"Cache delete error: {e}")
            return False

    async def exists(self, key: str) -> bool:
        """Check if key exists in cache.

        Args:
            key: Cache key

        Returns:
            True if key exists, False otherwise
        """
        try:
            hashed_key = self._hash_key(key)
            return await self.client.exists(hashed_key) > 0
        except Exception as e:
            logger.warning(f"Cache exists error: {e}")
            return False

    async def clear_pattern(self, pattern: str) -> int:
        """Clear all keys matching pattern.

        Args:
            pattern: Key pattern with wildcards

        Returns:
            Number of keys deleted
        """
        try:
            cursor = 0
            deleted = 0
            while True:
                cursor, keys = await self.client.scan(cursor, match=pattern, count=100)
                if keys:
                    deleted += await self.client.delete(*keys)
                if cursor == 0:
                    break
            return deleted
        except Exception as e:
            logger.warning(f"Cache clear pattern error: {e}")
            return 0

    def make_embedding_key(self, text: str, model: str) -> str:
        """Generate cache key for embeddings.

        Args:
            text: Input text
            model: Model name

        Returns:
            Cache key string
        """
        return f"embedding:{model}:{text}"

    def make_search_key(
        self,
        query: str,
        collection: str,
        top_k: int,
        filters: dict | None = None,
    ) -> str:
        """Generate cache key for search results.

        Args:
            query: Search query
            collection: Collection name
            top_k: Number of results
            filters: Optional metadata filters

        Returns:
            Cache key string
        """
        filters_str = json.dumps(filters, sort_keys=True) if filters else ""
        return f"search:{collection}:{query}:{top_k}:{filters_str}"


@lru_cache
def get_cache() -> Cache:
    """Get cached Cache instance."""
    settings = get_settings()
    return Cache(settings.redis_url, settings.cache_ttl)
