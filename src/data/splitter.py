"""Data splitting functions for train/valid/test sets."""

import polars as pl
from loguru import logger


def temporal_split(
    df: pl.DataFrame,
    train_ratio: float = 0.7,
    valid_ratio: float = 0.1,
    test_ratio: float = 0.2,
    time_col: str = "timestamp",
) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    """Split data by time into train/valid/test sets.

    Args:
        df: DataFrame to split.
        train_ratio: Fraction of data for training.
        valid_ratio: Fraction of data for validation.
        test_ratio: Fraction of data for testing.
        time_col: Column name for timestamp.

    Returns:
        Tuple of (train_df, valid_df, test_df).

    Raises:
        ValueError: If ratios don't sum to 1.0.
    """
    total = train_ratio + valid_ratio + test_ratio
    if abs(total - 1.0) > 1e-6:
        raise ValueError(f"Ratios must sum to 1.0, got {total}")

    logger.info(f"Temporal split: train={train_ratio}, valid={valid_ratio}, test={test_ratio}")

    # Sort by time
    df = df.sort(time_col)

    # Calculate split points
    n = len(df)
    train_end = int(n * train_ratio)
    valid_end = int(n * (train_ratio + valid_ratio))

    # Split data
    train_df = df.slice(0, train_end)
    valid_df = df.slice(train_end, valid_end - train_end)
    test_df = df.slice(valid_end, n - valid_end)

    # Log statistics
    logger.info(f"Train: {len(train_df):,} rows ({len(train_df)/n*100:.1f}%)")
    logger.info(f"Valid: {len(valid_df):,} rows ({len(valid_df)/n*100:.1f}%)")
    logger.info(f"Test: {len(test_df):,} rows ({len(test_df)/n*100:.1f}%)")

    # Log time ranges
    for name, split_df in [("Train", train_df), ("Valid", valid_df), ("Test", test_df)]:
        if len(split_df) > 0:
            min_ts = split_df[time_col].min()
            max_ts = split_df[time_col].max()
            logger.info(f"{name} time range: {min_ts} - {max_ts}")

    # Validate no data leakage
    if len(train_df) > 0 and len(valid_df) > 0:
        assert train_df[time_col].max() <= valid_df[time_col].min(), "Train/valid overlap!"
    if len(valid_df) > 0 and len(test_df) > 0:
        assert valid_df[time_col].max() <= test_df[time_col].min(), "Valid/test overlap!"

    logger.info("No temporal leakage detected")

    return train_df, valid_df, test_df


def split_by_users(
    df: pl.DataFrame,
    train_ratio: float = 0.7,
    valid_ratio: float = 0.1,
    test_ratio: float = 0.2,
    user_col: str = "user_id",
    seed: int = 42,
) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    """Split data by users into train/valid/test sets.

    Each user appears in only one split.

    Args:
        df: DataFrame to split.
        train_ratio: Fraction of users for training.
        valid_ratio: Fraction of users for validation.
        test_ratio: Fraction of users for testing.
        user_col: Column name for user identifier.
        seed: Random seed for reproducibility.

    Returns:
        Tuple of (train_df, valid_df, test_df).
    """
    total = train_ratio + valid_ratio + test_ratio
    if abs(total - 1.0) > 1e-6:
        raise ValueError(f"Ratios must sum to 1.0, got {total}")

    logger.info(f"User split: train={train_ratio}, valid={valid_ratio}, test={test_ratio}")

    # Get unique users and shuffle
    users = df.select(user_col).unique().sample(fraction=1.0, seed=seed)
    n_users = len(users)

    # Calculate split points
    train_end = int(n_users * train_ratio)
    valid_end = int(n_users * (train_ratio + valid_ratio))

    train_users = users.slice(0, train_end)
    valid_users = users.slice(train_end, valid_end - train_end)
    test_users = users.slice(valid_end, n_users - valid_end)

    # Filter data by user sets
    train_df = df.join(train_users, on=user_col, how="inner")
    valid_df = df.join(valid_users, on=user_col, how="inner")
    test_df = df.join(test_users, on=user_col, how="inner")

    logger.info(f"Train: {len(train_df):,} rows, {len(train_users):,} users")
    logger.info(f"Valid: {len(valid_df):,} rows, {len(valid_users):,} users")
    logger.info(f"Test: {len(test_df):,} rows, {len(test_users):,} users")

    return train_df, valid_df, test_df
