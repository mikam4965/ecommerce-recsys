"""Initialize SQLite database and load user/item data.

This script:
1. Creates database tables and indexes
2. Loads user data from train.parquet (aggregated stats)
3. Loads item data from train.parquet (with category and popularity)

Run: python scripts/init_database.py
"""

import sys
from pathlib import Path

import polars as pl
from loguru import logger

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.data.database import DatabaseManager, init_database


def load_user_stats(train_path: Path) -> pl.DataFrame:
    """Calculate user statistics from training data.

    Args:
        train_path: Path to train.parquet.

    Returns:
        DataFrame with user stats.
    """
    logger.info(f"Loading user stats from {train_path}")

    train = pl.read_parquet(train_path)

    # Aggregate user statistics
    users = (
        train.group_by("user_id")
        .agg([
            # Total purchases (transactions)
            pl.col("event_type")
            .filter(pl.col("event_type") == "transaction")
            .len()
            .alias("total_purchases"),
            # Last activity
            pl.col("timestamp").max().alias("last_activity"),
        ])
    )

    logger.info(f"Calculated stats for {len(users):,} users")
    return users


def load_item_stats(train_path: Path) -> pl.DataFrame:
    """Calculate item statistics from training data.

    Args:
        train_path: Path to train.parquet.

    Returns:
        DataFrame with item stats.
    """
    logger.info(f"Loading item stats from {train_path}")

    train = pl.read_parquet(train_path)

    # Get category for each item (most common)
    item_category = (
        train.filter(pl.col("category_id") != "unknown")
        .group_by("item_id")
        .agg(pl.col("category_id").mode().first().alias("category_id"))
    )

    # Calculate popularity score (interaction count, normalized)
    item_popularity = (
        train.group_by("item_id")
        .agg(pl.len().alias("interaction_count"))
    )

    max_count = item_popularity["interaction_count"].max()
    item_popularity = item_popularity.with_columns(
        (pl.col("interaction_count") / max_count).alias("popularity_score")
    )

    # Join category and popularity
    items = item_popularity.join(item_category, on="item_id", how="left")
    items = items.select(["item_id", "category_id", "popularity_score"])

    # Fill missing category
    items = items.with_columns(
        pl.col("category_id").fill_null("unknown")
    )

    logger.info(f"Calculated stats for {len(items):,} items")
    return items


def load_rfm_segments(rfm_path: Path, users: pl.DataFrame) -> pl.DataFrame:
    """Add RFM segments to user data.

    Args:
        rfm_path: Path to rfm_segmentation.parquet.
        users: User DataFrame.

    Returns:
        User DataFrame with segment column.
    """
    if not rfm_path.exists():
        logger.warning(f"RFM file not found: {rfm_path}")
        return users.with_columns(pl.lit(None).alias("segment"))

    logger.info(f"Loading RFM segments from {rfm_path}")
    rfm = pl.read_parquet(rfm_path)

    if "segment" in rfm.columns:
        users = users.join(
            rfm.select(["user_id", "segment"]),
            on="user_id",
            how="left",
        )
        logger.info(f"Added segments for {users['segment'].drop_nulls().len():,} users")
    else:
        users = users.with_columns(pl.lit(None).alias("segment"))

    return users


def main():
    """Initialize database and load data."""
    logger.info("=" * 60)
    logger.info("DATABASE INITIALIZATION")
    logger.info("=" * 60)

    # Paths
    db_path = project_root / "data" / "database.sqlite"
    train_path = project_root / "data" / "processed" / "train.parquet"
    rfm_path = project_root / "data" / "processed" / "rfm_segmentation.parquet"

    # Check data exists
    if not train_path.exists():
        logger.error(f"Training data not found: {train_path}")
        logger.error("Run preprocessing first: make preprocess")
        sys.exit(1)

    # Initialize database
    logger.info(f"Initializing database: {db_path}")
    init_database(db_path)

    # Create manager
    db = DatabaseManager(db_path)

    # Load and insert users
    users = load_user_stats(train_path)
    users = load_rfm_segments(rfm_path, users)
    n_users = db.insert_users(users)
    logger.info(f"Inserted {n_users:,} users")

    # Load and insert items
    items = load_item_stats(train_path)
    n_items = db.insert_items(items)
    logger.info(f"Inserted {n_items:,} items")

    # Print stats
    stats = db.get_stats()
    logger.info("=" * 60)
    logger.info("DATABASE STATS:")
    for key, value in stats.items():
        logger.info(f"  {key}: {value}")
    logger.info("=" * 60)

    logger.info("Database initialization complete!")


if __name__ == "__main__":
    main()
