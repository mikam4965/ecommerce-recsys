"""SQLite database for storing metadata and caching recommendations.

This module provides:
- User and item metadata storage
- Recommendation caching with TTL
- User interaction history logging
"""

import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import polars as pl
from loguru import logger


# SQL statements for table creation
CREATE_USERS_TABLE = """
CREATE TABLE IF NOT EXISTS users (
    user_id INTEGER PRIMARY KEY,
    segment TEXT,
    total_purchases INTEGER DEFAULT 0,
    last_activity TIMESTAMP,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
)
"""

CREATE_ITEMS_TABLE = """
CREATE TABLE IF NOT EXISTS items (
    item_id INTEGER PRIMARY KEY,
    category_id TEXT,
    price REAL,
    popularity_score REAL DEFAULT 0
)
"""

CREATE_CACHE_TABLE = """
CREATE TABLE IF NOT EXISTS recommendations_cache (
    user_id INTEGER PRIMARY KEY,
    recommendations TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    expires_at TIMESTAMP
)
"""

CREATE_HISTORY_TABLE = """
CREATE TABLE IF NOT EXISTS user_history (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id INTEGER,
    item_id INTEGER,
    event_type TEXT,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
)
"""

# Indexes for performance
CREATE_INDEXES = [
    "CREATE INDEX IF NOT EXISTS idx_users_segment ON users(segment)",
    "CREATE INDEX IF NOT EXISTS idx_items_category ON items(category_id)",
    "CREATE INDEX IF NOT EXISTS idx_cache_expires ON recommendations_cache(expires_at)",
    "CREATE INDEX IF NOT EXISTS idx_history_user ON user_history(user_id)",
    "CREATE INDEX IF NOT EXISTS idx_history_item ON user_history(item_id)",
    "CREATE INDEX IF NOT EXISTS idx_history_timestamp ON user_history(timestamp)",
]


def init_database(db_path: str | Path) -> None:
    """Initialize database with required tables and indexes.

    Args:
        db_path: Path to SQLite database file.
    """
    db_path = Path(db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info(f"Initializing database at {db_path}")

    conn = sqlite3.connect(str(db_path))
    cursor = conn.cursor()

    try:
        # Create tables
        cursor.execute(CREATE_USERS_TABLE)
        cursor.execute(CREATE_ITEMS_TABLE)
        cursor.execute(CREATE_CACHE_TABLE)
        cursor.execute(CREATE_HISTORY_TABLE)

        # Create indexes
        for index_sql in CREATE_INDEXES:
            cursor.execute(index_sql)

        conn.commit()
        logger.info("Database initialized successfully")

    except Exception as e:
        logger.error(f"Error initializing database: {e}")
        conn.rollback()
        raise

    finally:
        conn.close()


@contextmanager
def get_connection(db_path: str | Path, timeout: float = 30.0):
    """Context manager for database connections.

    Args:
        db_path: Path to SQLite database file.
        timeout: Connection timeout in seconds.

    Yields:
        sqlite3.Connection object.
    """
    conn = sqlite3.connect(
        str(db_path),
        timeout=timeout,
        check_same_thread=False,
    )
    conn.row_factory = sqlite3.Row  # Enable dict-like access
    try:
        yield conn
    finally:
        conn.close()


class DatabaseManager:
    """Manager for database operations.

    Provides methods for:
    - Inserting/retrieving user and item data
    - Caching recommendations with TTL
    - Logging user events
    """

    def __init__(
        self,
        db_path: str | Path,
        cache_ttl: int = 3600,
        timeout: float = 30.0,
    ):
        """Initialize DatabaseManager.

        Args:
            db_path: Path to SQLite database file.
            cache_ttl: Default TTL for cached recommendations (seconds).
            timeout: Connection timeout in seconds.
        """
        self.db_path = Path(db_path)
        self.cache_ttl = cache_ttl
        self.timeout = timeout

        # Ensure database exists
        if not self.db_path.exists():
            init_database(self.db_path)

        logger.info(f"DatabaseManager initialized: {self.db_path}")

    def _get_conn(self) -> sqlite3.Connection:
        """Get a database connection."""
        conn = sqlite3.connect(
            str(self.db_path),
            timeout=self.timeout,
            check_same_thread=False,
        )
        conn.row_factory = sqlite3.Row
        return conn

    # ==================== User Operations ====================

    def insert_users(self, df: pl.DataFrame) -> int:
        """Insert users from Polars DataFrame.

        Expected columns: user_id, segment (optional), total_purchases (optional),
                         last_activity (optional)

        Args:
            df: Polars DataFrame with user data.

        Returns:
            Number of rows inserted.
        """
        if df.is_empty():
            return 0

        # Ensure required column
        if "user_id" not in df.columns:
            raise ValueError("DataFrame must have 'user_id' column")

        # Add missing columns with defaults
        if "segment" not in df.columns:
            df = df.with_columns(pl.lit(None).alias("segment"))
        if "total_purchases" not in df.columns:
            df = df.with_columns(pl.lit(0).alias("total_purchases"))
        if "last_activity" not in df.columns:
            df = df.with_columns(pl.lit(None).alias("last_activity"))

        conn = self._get_conn()
        cursor = conn.cursor()

        try:
            # Use INSERT OR REPLACE for upsert behavior
            sql = """
                INSERT OR REPLACE INTO users (user_id, segment, total_purchases, last_activity)
                VALUES (?, ?, ?, ?)
            """

            rows = df.select([
                "user_id", "segment", "total_purchases", "last_activity"
            ]).to_numpy().tolist()

            cursor.executemany(sql, rows)
            conn.commit()

            count = cursor.rowcount
            logger.info(f"Inserted {len(rows)} users")
            return len(rows)

        except Exception as e:
            logger.error(f"Error inserting users: {e}")
            conn.rollback()
            raise

        finally:
            conn.close()

    def get_user(self, user_id: int) -> dict[str, Any] | None:
        """Get user by ID.

        Args:
            user_id: User identifier.

        Returns:
            User dict or None if not found.
        """
        conn = self._get_conn()
        try:
            cursor = conn.execute(
                "SELECT * FROM users WHERE user_id = ?",
                (user_id,)
            )
            row = cursor.fetchone()
            return dict(row) if row else None
        finally:
            conn.close()

    def get_users_by_segment(self, segment: str) -> list[dict[str, Any]]:
        """Get all users in a segment.

        Args:
            segment: Segment name.

        Returns:
            List of user dicts.
        """
        conn = self._get_conn()
        try:
            cursor = conn.execute(
                "SELECT * FROM users WHERE segment = ?",
                (segment,)
            )
            return [dict(row) for row in cursor.fetchall()]
        finally:
            conn.close()

    # ==================== Item Operations ====================

    def insert_items(self, df: pl.DataFrame) -> int:
        """Insert items from Polars DataFrame.

        Expected columns: item_id, category_id (optional), price (optional),
                         popularity_score (optional)

        Args:
            df: Polars DataFrame with item data.

        Returns:
            Number of rows inserted.
        """
        if df.is_empty():
            return 0

        if "item_id" not in df.columns:
            raise ValueError("DataFrame must have 'item_id' column")

        # Add missing columns with defaults
        if "category_id" not in df.columns:
            df = df.with_columns(pl.lit(None).alias("category_id"))
        if "price" not in df.columns:
            df = df.with_columns(pl.lit(None).alias("price"))
        if "popularity_score" not in df.columns:
            df = df.with_columns(pl.lit(0.0).alias("popularity_score"))

        conn = self._get_conn()
        cursor = conn.cursor()

        try:
            sql = """
                INSERT OR REPLACE INTO items (item_id, category_id, price, popularity_score)
                VALUES (?, ?, ?, ?)
            """

            rows = df.select([
                "item_id", "category_id", "price", "popularity_score"
            ]).to_numpy().tolist()

            cursor.executemany(sql, rows)
            conn.commit()

            logger.info(f"Inserted {len(rows)} items")
            return len(rows)

        except Exception as e:
            logger.error(f"Error inserting items: {e}")
            conn.rollback()
            raise

        finally:
            conn.close()

    def get_item(self, item_id: int) -> dict[str, Any] | None:
        """Get item by ID.

        Args:
            item_id: Item identifier.

        Returns:
            Item dict or None if not found.
        """
        conn = self._get_conn()
        try:
            cursor = conn.execute(
                "SELECT * FROM items WHERE item_id = ?",
                (item_id,)
            )
            row = cursor.fetchone()
            return dict(row) if row else None
        finally:
            conn.close()

    def get_items_by_category(self, category_id: str) -> list[dict[str, Any]]:
        """Get all items in a category.

        Args:
            category_id: Category identifier.

        Returns:
            List of item dicts.
        """
        conn = self._get_conn()
        try:
            cursor = conn.execute(
                "SELECT * FROM items WHERE category_id = ?",
                (category_id,)
            )
            return [dict(row) for row in cursor.fetchall()]
        finally:
            conn.close()

    # ==================== Cache Operations ====================

    def cache_recommendations(
        self,
        user_id: int,
        items: list[dict[str, Any]],
        ttl: int | None = None,
    ) -> None:
        """Cache recommendations for a user.

        Args:
            user_id: User identifier.
            items: List of recommendation dicts with item_id and score.
            ttl: Time-to-live in seconds (uses default if None).
        """
        ttl = ttl or self.cache_ttl
        expires_at = datetime.now() + timedelta(seconds=ttl)

        conn = self._get_conn()
        try:
            conn.execute(
                """
                INSERT OR REPLACE INTO recommendations_cache
                (user_id, recommendations, created_at, expires_at)
                VALUES (?, ?, CURRENT_TIMESTAMP, ?)
                """,
                (user_id, json.dumps(items), expires_at.isoformat())
            )
            conn.commit()
            logger.debug(f"Cached recommendations for user {user_id} (TTL: {ttl}s)")

        except Exception as e:
            logger.error(f"Error caching recommendations: {e}")
            conn.rollback()

        finally:
            conn.close()

    def get_cached_recommendations(
        self,
        user_id: int,
    ) -> list[dict[str, Any]] | None:
        """Get cached recommendations for a user.

        Args:
            user_id: User identifier.

        Returns:
            List of recommendation dicts or None if not found/expired.
        """
        conn = self._get_conn()
        try:
            cursor = conn.execute(
                """
                SELECT recommendations, expires_at
                FROM recommendations_cache
                WHERE user_id = ?
                """,
                (user_id,)
            )
            row = cursor.fetchone()

            if not row:
                logger.debug(f"Cache MISS: user {user_id}")
                return None

            # Check expiration
            expires_at = datetime.fromisoformat(row["expires_at"])
            if datetime.now() > expires_at:
                logger.debug(f"Cache EXPIRED: user {user_id}")
                # Delete expired entry
                conn.execute(
                    "DELETE FROM recommendations_cache WHERE user_id = ?",
                    (user_id,)
                )
                conn.commit()
                return None

            logger.debug(f"Cache HIT: user {user_id}")
            return json.loads(row["recommendations"])

        except Exception as e:
            logger.error(f"Error getting cached recommendations: {e}")
            return None

        finally:
            conn.close()

    def cleanup_expired_cache(self) -> int:
        """Remove all expired cache entries.

        Returns:
            Number of entries removed.
        """
        conn = self._get_conn()
        try:
            cursor = conn.execute(
                """
                DELETE FROM recommendations_cache
                WHERE expires_at < ?
                """,
                (datetime.now().isoformat(),)
            )
            conn.commit()
            count = cursor.rowcount
            if count > 0:
                logger.info(f"Cleaned up {count} expired cache entries")
            return count

        except Exception as e:
            logger.error(f"Error cleaning cache: {e}")
            return 0

        finally:
            conn.close()

    def invalidate_user_cache(self, user_id: int) -> bool:
        """Invalidate cache for a specific user.

        Args:
            user_id: User identifier.

        Returns:
            True if cache was invalidated.
        """
        conn = self._get_conn()
        try:
            cursor = conn.execute(
                "DELETE FROM recommendations_cache WHERE user_id = ?",
                (user_id,)
            )
            conn.commit()
            return cursor.rowcount > 0
        finally:
            conn.close()

    # ==================== History Operations ====================

    def log_event(
        self,
        user_id: int,
        item_id: int,
        event_type: str,
    ) -> None:
        """Log a user interaction event.

        Args:
            user_id: User identifier.
            item_id: Item identifier.
            event_type: Event type (view, addtocart, transaction).
        """
        conn = self._get_conn()
        try:
            conn.execute(
                """
                INSERT INTO user_history (user_id, item_id, event_type)
                VALUES (?, ?, ?)
                """,
                (user_id, item_id, event_type)
            )
            conn.commit()
            logger.debug(f"Logged event: user={user_id}, item={item_id}, type={event_type}")

        except Exception as e:
            logger.error(f"Error logging event: {e}")
            conn.rollback()

        finally:
            conn.close()

    def get_user_history(
        self,
        user_id: int,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        """Get user interaction history.

        Args:
            user_id: User identifier.
            limit: Maximum number of events to return.

        Returns:
            List of event dicts ordered by timestamp descending.
        """
        conn = self._get_conn()
        try:
            cursor = conn.execute(
                """
                SELECT * FROM user_history
                WHERE user_id = ?
                ORDER BY timestamp DESC
                LIMIT ?
                """,
                (user_id, limit)
            )
            return [dict(row) for row in cursor.fetchall()]
        finally:
            conn.close()

    # ==================== Stats Operations ====================

    def get_stats(self) -> dict[str, Any]:
        """Get database statistics.

        Returns:
            Dict with table counts and cache stats.
        """
        conn = self._get_conn()
        try:
            stats = {}

            # Table counts
            for table in ["users", "items", "recommendations_cache", "user_history"]:
                cursor = conn.execute(f"SELECT COUNT(*) FROM {table}")
                stats[f"{table}_count"] = cursor.fetchone()[0]

            # Cache hit stats (active entries)
            cursor = conn.execute(
                """
                SELECT COUNT(*) FROM recommendations_cache
                WHERE expires_at > ?
                """,
                (datetime.now().isoformat(),)
            )
            stats["active_cache_entries"] = cursor.fetchone()[0]

            # Database file size
            stats["db_size_mb"] = round(self.db_path.stat().st_size / (1024 * 1024), 2)

            return stats

        finally:
            conn.close()

    def flush_cache(self) -> int:
        """Clear all cache entries.

        Returns:
            Number of entries removed.
        """
        conn = self._get_conn()
        try:
            cursor = conn.execute("DELETE FROM recommendations_cache")
            conn.commit()
            count = cursor.rowcount
            logger.info(f"Flushed {count} cache entries")
            return count
        finally:
            conn.close()
