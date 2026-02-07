"""Data preprocessing functions for RetailRocket dataset."""

import polars as pl
from loguru import logger


def reconstruct_sessions(
    df: pl.DataFrame,
    gap_minutes: int = 30,
    user_col: str = "visitorid",
    time_col: str = "timestamp",
) -> pl.DataFrame:
    """Reconstruct user sessions based on time gap.

    A new session starts when the gap between consecutive events
    for the same user exceeds the threshold.

    Args:
        df: Events DataFrame.
        gap_minutes: Maximum gap in minutes within a session.
        user_col: Column name for user identifier.
        time_col: Column name for timestamp.

    Returns:
        DataFrame with added session_id column.
    """
    logger.info(f"Reconstructing sessions with gap={gap_minutes} minutes")

    # Convert gap to milliseconds (RetailRocket timestamps are in ms)
    gap_ms = gap_minutes * 60 * 1000

    df = df.sort([user_col, time_col])

    df = df.with_columns(
        pl.col(time_col).diff().over(user_col).alias("time_diff"),
    )

    # Mark new session when gap > threshold or first event of user
    df = df.with_columns(
        (
            (pl.col("time_diff") > gap_ms) | pl.col("time_diff").is_null()
        ).alias("is_new_session")
    )

    # Cumulative sum of new session flags = session id within user
    df = df.with_columns(
        pl.col("is_new_session").cum_sum().over(user_col).alias("user_session_num")
    )

    # Create global session_id
    df = df.with_columns(
        (
            pl.col(user_col).cast(pl.Utf8)
            + "_"
            + pl.col("user_session_num").cast(pl.Utf8)
        ).alias("session_id")
    )

    # Drop temporary columns
    df = df.drop(["time_diff", "is_new_session", "user_session_num"])

    n_sessions = df["session_id"].n_unique()
    logger.info(f"Reconstructed {n_sessions:,} sessions")

    return df


def extract_item_metadata(properties_df: pl.DataFrame) -> pl.DataFrame:
    """Extract category_id from item properties.

    Note: RetailRocket dataset has anonymized properties.
    Only 'categoryid' and 'available' are readable, no price data.

    Args:
        properties_df: DataFrame with item properties (long format).

    Returns:
        DataFrame with item_id, category_id columns.
    """
    logger.info("Extracting item metadata (category_id)")

    # Filter only categoryid property
    logger.info("Filtering categoryid properties...")
    filtered = properties_df.filter(pl.col("property") == "categoryid")
    logger.info(f"Filtered to {len(filtered):,} records")

    # Get latest category per item using lazy evaluation
    logger.info("Getting latest category per item...")
    metadata = (
        filtered.lazy()
        .sort("timestamp", descending=True)
        .group_by("itemid")
        .agg(pl.col("value").first().alias("category_id"))
        .collect()
    )

    logger.info(f"Extracted metadata for {len(metadata):,} items")

    return metadata


def handle_missing_values(
    df: pl.DataFrame,
    category_col: str = "category_id",
    default_category: str = "unknown",
) -> pl.DataFrame:
    """Handle missing values in the dataset.

    Args:
        df: DataFrame with potential missing values.
        category_col: Column name for category.
        default_category: Default value for missing categories.

    Returns:
        DataFrame with filled missing values.
    """
    logger.info("Handling missing values")

    if category_col in df.columns:
        initial_nulls = df[category_col].null_count()
        df = df.with_columns(
            pl.col(category_col).fill_null(default_category)
        )
        logger.info(f"Filled {initial_nulls:,} null categories with '{default_category}'")

    return df


def create_final_dataset(
    events: pl.DataFrame,
    metadata: pl.DataFrame,
    categories: pl.DataFrame | None = None,
) -> pl.DataFrame:
    """Create final preprocessed dataset.

    Args:
        events: Events DataFrame with session_id.
        metadata: Item metadata DataFrame.
        categories: Optional category hierarchy DataFrame.

    Returns:
        Final DataFrame with all columns.
    """
    logger.info("Creating final dataset")
    logger.info(f"Events: {len(events):,} rows")
    logger.info(f"Metadata: {len(metadata):,} items")

    # Join events with item metadata
    df = events.join(
        metadata,
        left_on="itemid",
        right_on="itemid",
        how="left",
    )

    # Rename columns to standard names
    df = df.rename({
        "visitorid": "user_id",
        "itemid": "item_id",
        "event": "event_type",
    })

    # Select and order final columns
    final_columns = [
        "user_id",
        "session_id",
        "timestamp",
        "event_type",
        "item_id",
        "category_id",
    ]

    # Keep only existing columns
    existing_columns = [c for c in final_columns if c in df.columns]
    df = df.select(existing_columns)

    # Handle missing values
    df = handle_missing_values(df)

    # Sort by user and time
    df = df.sort(["user_id", "timestamp"])

    logger.info(f"Final dataset: {len(df):,} rows")
    logger.info(f"Columns: {df.columns}")

    return df
