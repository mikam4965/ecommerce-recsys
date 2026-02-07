"""Redis caching for recommendations.

Optional caching layer for:
- Popular items (refreshed periodically)
- Active user recommendations (with TTL)
"""

import json
from typing import Any

from loguru import logger

# Redis is optional
try:
    import redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False
    logger.warning("Redis not installed. Caching disabled.")


class RecommendationCache:
    """Redis-based cache for recommendations."""

    # Cache key prefixes
    PREFIX_POPULAR = "recsys:popular"
    PREFIX_USER_RECS = "recsys:user:"
    PREFIX_SIMILAR = "recsys:similar:"

    # Default TTL values (seconds)
    TTL_POPULAR = 3600  # 1 hour
    TTL_USER_RECS = 300  # 5 minutes
    TTL_SIMILAR = 1800  # 30 minutes

    def __init__(
        self,
        host: str = "localhost",
        port: int = 6379,
        db: int = 0,
        password: str | None = None,
        enabled: bool = True,
    ):
        """Initialize Redis cache.

        Args:
            host: Redis host.
            port: Redis port.
            db: Redis database number.
            password: Redis password (optional).
            enabled: Whether caching is enabled.
        """
        self.enabled = enabled and REDIS_AVAILABLE
        self.client: redis.Redis | None = None

        if self.enabled:
            try:
                self.client = redis.Redis(
                    host=host,
                    port=port,
                    db=db,
                    password=password,
                    decode_responses=True,
                    socket_connect_timeout=2,
                )
                # Test connection
                self.client.ping()
                logger.info(f"Redis cache connected: {host}:{port}")
            except Exception as e:
                logger.warning(f"Redis connection failed: {e}. Caching disabled.")
                self.enabled = False
                self.client = None

    def is_connected(self) -> bool:
        """Check if Redis is connected."""
        if not self.enabled or self.client is None:
            return False
        try:
            self.client.ping()
            return True
        except Exception:
            return False

    def get_popular_items(self, n: int = 10) -> list[dict[str, Any]] | None:
        """Get cached popular items.

        Args:
            n: Number of items.

        Returns:
            List of item dicts or None if not cached.
        """
        if not self.enabled or self.client is None:
            return None

        try:
            key = f"{self.PREFIX_POPULAR}:{n}"
            data = self.client.get(key)
            if data:
                logger.debug(f"Cache HIT: popular items (n={n})")
                return json.loads(data)
            logger.debug(f"Cache MISS: popular items (n={n})")
            return None
        except Exception as e:
            logger.warning(f"Cache get error: {e}")
            return None

    def set_popular_items(
        self,
        items: list[dict[str, Any]],
        n: int = 10,
        ttl: int | None = None,
    ) -> bool:
        """Cache popular items.

        Args:
            items: List of item dicts.
            n: Number of items (used in key).
            ttl: Cache TTL in seconds.

        Returns:
            True if cached successfully.
        """
        if not self.enabled or self.client is None:
            return False

        try:
            key = f"{self.PREFIX_POPULAR}:{n}"
            ttl = ttl or self.TTL_POPULAR
            self.client.setex(key, ttl, json.dumps(items))
            logger.debug(f"Cached popular items (n={n}, ttl={ttl}s)")
            return True
        except Exception as e:
            logger.warning(f"Cache set error: {e}")
            return False

    def get_user_recommendations(
        self,
        user_id: int,
        n: int = 10,
    ) -> list[dict[str, Any]] | None:
        """Get cached user recommendations.

        Args:
            user_id: User identifier.
            n: Number of recommendations.

        Returns:
            List of recommendation dicts or None if not cached.
        """
        if not self.enabled or self.client is None:
            return None

        try:
            key = f"{self.PREFIX_USER_RECS}{user_id}:{n}"
            data = self.client.get(key)
            if data:
                logger.debug(f"Cache HIT: user {user_id} recommendations")
                return json.loads(data)
            logger.debug(f"Cache MISS: user {user_id} recommendations")
            return None
        except Exception as e:
            logger.warning(f"Cache get error: {e}")
            return None

    def set_user_recommendations(
        self,
        user_id: int,
        recommendations: list[dict[str, Any]],
        n: int = 10,
        ttl: int | None = None,
    ) -> bool:
        """Cache user recommendations.

        Args:
            user_id: User identifier.
            recommendations: List of recommendation dicts.
            n: Number of recommendations (used in key).
            ttl: Cache TTL in seconds.

        Returns:
            True if cached successfully.
        """
        if not self.enabled or self.client is None:
            return False

        try:
            key = f"{self.PREFIX_USER_RECS}{user_id}:{n}"
            ttl = ttl or self.TTL_USER_RECS
            self.client.setex(key, ttl, json.dumps(recommendations))
            logger.debug(f"Cached recommendations for user {user_id}")
            return True
        except Exception as e:
            logger.warning(f"Cache set error: {e}")
            return False

    def get_similar_items(
        self,
        item_id: int,
        n: int = 10,
    ) -> list[dict[str, Any]] | None:
        """Get cached similar items.

        Args:
            item_id: Item identifier.
            n: Number of similar items.

        Returns:
            List of similar item dicts or None if not cached.
        """
        if not self.enabled or self.client is None:
            return None

        try:
            key = f"{self.PREFIX_SIMILAR}{item_id}:{n}"
            data = self.client.get(key)
            if data:
                logger.debug(f"Cache HIT: similar items for {item_id}")
                return json.loads(data)
            logger.debug(f"Cache MISS: similar items for {item_id}")
            return None
        except Exception as e:
            logger.warning(f"Cache get error: {e}")
            return None

    def set_similar_items(
        self,
        item_id: int,
        similar_items: list[dict[str, Any]],
        n: int = 10,
        ttl: int | None = None,
    ) -> bool:
        """Cache similar items.

        Args:
            item_id: Item identifier.
            similar_items: List of similar item dicts.
            n: Number of items (used in key).
            ttl: Cache TTL in seconds.

        Returns:
            True if cached successfully.
        """
        if not self.enabled or self.client is None:
            return False

        try:
            key = f"{self.PREFIX_SIMILAR}{item_id}:{n}"
            ttl = ttl or self.TTL_SIMILAR
            self.client.setex(key, ttl, json.dumps(similar_items))
            logger.debug(f"Cached similar items for {item_id}")
            return True
        except Exception as e:
            logger.warning(f"Cache set error: {e}")
            return False

    def invalidate_user(self, user_id: int) -> bool:
        """Invalidate all cache entries for a user.

        Args:
            user_id: User identifier.

        Returns:
            True if invalidated successfully.
        """
        if not self.enabled or self.client is None:
            return False

        try:
            pattern = f"{self.PREFIX_USER_RECS}{user_id}:*"
            keys = list(self.client.scan_iter(match=pattern))
            if keys:
                self.client.delete(*keys)
                logger.debug(f"Invalidated {len(keys)} cache entries for user {user_id}")
            return True
        except Exception as e:
            logger.warning(f"Cache invalidation error: {e}")
            return False

    def flush_all(self) -> bool:
        """Flush all recommendation cache entries.

        Returns:
            True if flushed successfully.
        """
        if not self.enabled or self.client is None:
            return False

        try:
            pattern = "recsys:*"
            keys = list(self.client.scan_iter(match=pattern))
            if keys:
                self.client.delete(*keys)
                logger.info(f"Flushed {len(keys)} cache entries")
            return True
        except Exception as e:
            logger.warning(f"Cache flush error: {e}")
            return False

    def get_stats(self) -> dict[str, Any]:
        """Get cache statistics.

        Returns:
            Dict with cache stats.
        """
        if not self.enabled or self.client is None:
            return {"enabled": False}

        try:
            info = self.client.info("memory")
            return {
                "enabled": True,
                "connected": True,
                "used_memory": info.get("used_memory_human", "N/A"),
                "keys": self.client.dbsize(),
            }
        except Exception as e:
            return {"enabled": True, "connected": False, "error": str(e)}
