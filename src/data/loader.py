"""Data loading functions for RetailRocket dataset."""

from pathlib import Path

import polars as pl
from loguru import logger


# Expected schemas for validation
EVENTS_SCHEMA = {
    "timestamp": pl.Int64,
    "visitorid": pl.Int64,
    "event": pl.Utf8,
    "itemid": pl.Int64,
    "transactionid": pl.Int64,
}

ITEM_PROPERTIES_SCHEMA = {
    "timestamp": pl.Int64,
    "itemid": pl.Int64,
    "property": pl.Utf8,
    "value": pl.Utf8,
}

CATEGORIES_SCHEMA = {
    "categoryid": pl.Int64,
    "parentid": pl.Int64,
}


def validate_schema(df: pl.DataFrame, expected_schema: dict[str, pl.DataType], name: str) -> None:
    """Validate DataFrame schema against expected schema.

    Args:
        df: DataFrame to validate.
        expected_schema: Expected column names and types.
        name: Dataset name for error messages.

    Raises:
        ValueError: If schema doesn't match.
    """
    actual_columns = set(df.columns)
    expected_columns = set(expected_schema.keys())

    missing = expected_columns - actual_columns
    if missing:
        raise ValueError(f"{name}: missing columns {missing}")

    for col, expected_type in expected_schema.items():
        actual_type = df.schema.get(col)
        if actual_type is None:
            continue
        # Allow nullable types
        if actual_type != expected_type and actual_type != pl.Null:
            logger.warning(f"{name}.{col}: expected {expected_type}, got {actual_type}")


def load_events(path: str | Path) -> pl.DataFrame:
    """Load events.csv file.

    Args:
        path: Path to events.csv file.

    Returns:
        DataFrame with user interactions.

    Raises:
        FileNotFoundError: If file doesn't exist.
        ValueError: If schema validation fails.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Events file not found: {path}")

    logger.info(f"Loading events from {path}")

    df = pl.read_csv(
        path,
        schema_overrides={
            "timestamp": pl.Int64,
            "visitorid": pl.Int64,
            "itemid": pl.Int64,
            "transactionid": pl.Int64,
        },
        null_values=["", "NA", "null"],
    )

    validate_schema(df, EVENTS_SCHEMA, "events")

    logger.info(f"Loaded {len(df):,} events")
    logger.info(f"Unique users: {df['visitorid'].n_unique():,}")
    logger.info(f"Unique items: {df['itemid'].n_unique():,}")
    logger.info(f"Event types: {df['event'].unique().to_list()}")

    return df


def load_item_properties(path1: str | Path, path2: str | Path) -> pl.DataFrame:
    """Load and merge item_properties files.

    Args:
        path1: Path to item_properties_part1.csv.
        path2: Path to item_properties_part2.csv.

    Returns:
        Combined DataFrame with item properties.

    Raises:
        FileNotFoundError: If any file doesn't exist.
    """
    path1 = Path(path1)
    path2 = Path(path2)

    for p in [path1, path2]:
        if not p.exists():
            raise FileNotFoundError(f"Item properties file not found: {p}")

    logger.info(f"Loading item properties from {path1} and {path2}")

    df1 = pl.read_csv(
        path1,
        schema_overrides={"timestamp": pl.Int64, "itemid": pl.Int64},
        null_values=["", "NA", "null"],
    )

    df2 = pl.read_csv(
        path2,
        schema_overrides={"timestamp": pl.Int64, "itemid": pl.Int64},
        null_values=["", "NA", "null"],
    )

    df = pl.concat([df1, df2])

    validate_schema(df, ITEM_PROPERTIES_SCHEMA, "item_properties")

    logger.info(f"Loaded {len(df):,} property records")
    logger.info(f"Unique items: {df['itemid'].n_unique():,}")
    logger.info(f"Property types: {df['property'].unique().to_list()[:10]}...")

    return df


def load_categories(path: str | Path) -> pl.DataFrame:
    """Load category_tree.csv file.

    Args:
        path: Path to category_tree.csv file.

    Returns:
        DataFrame with category hierarchy.

    Raises:
        FileNotFoundError: If file doesn't exist.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Categories file not found: {path}")

    logger.info(f"Loading categories from {path}")

    df = pl.read_csv(
        path,
        schema_overrides={"categoryid": pl.Int64, "parentid": pl.Int64},
        null_values=["", "NA", "null"],
    )

    validate_schema(df, CATEGORIES_SCHEMA, "categories")

    logger.info(f"Loaded {len(df):,} categories")

    return df
