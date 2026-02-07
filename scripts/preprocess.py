"""Preprocessing pipeline for RetailRocket dataset."""

import sys
from pathlib import Path

import yaml
from loguru import logger

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.loader import load_categories, load_events, load_item_properties
from src.data.preprocessing import (
    create_final_dataset,
    extract_item_metadata,
    reconstruct_sessions,
)
from src.data.splitter import temporal_split


def load_config(config_path: str | Path) -> dict:
    """Load configuration from YAML file."""
    with open(config_path) as f:
        return yaml.safe_load(f)


def main() -> None:
    """Run preprocessing pipeline."""
    # Configure logging
    logger.remove()
    logger.add(
        sys.stderr,
        format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | {message}",
        level="INFO",
    )

    logger.info("=" * 60)
    logger.info("RetailRocket Preprocessing Pipeline")
    logger.info("=" * 60)

    # Load config
    config_path = PROJECT_ROOT / "configs" / "data.yaml"
    config = load_config(config_path)
    logger.info(f"Loaded config from {config_path}")

    # Create output directory
    output_dir = PROJECT_ROOT / "data" / "processed"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Step 1: Load raw data
    logger.info("-" * 40)
    logger.info("Step 1: Loading raw data")
    logger.info("-" * 40)

    raw_paths = config["paths"]["raw"]

    events = load_events(PROJECT_ROOT / raw_paths["events"])
    properties = load_item_properties(
        PROJECT_ROOT / raw_paths["item_properties_1"],
        PROJECT_ROOT / raw_paths["item_properties_2"],
    )
    categories = load_categories(PROJECT_ROOT / raw_paths["categories"])

    # Step 2: Reconstruct sessions
    logger.info("-" * 40)
    logger.info("Step 2: Reconstructing sessions")
    logger.info("-" * 40)

    session_gap = config["preprocessing"]["session_gap_minutes"]
    events = reconstruct_sessions(events, gap_minutes=session_gap)

    # Step 3: Extract item metadata
    logger.info("-" * 40)
    logger.info("Step 3: Extracting item metadata")
    logger.info("-" * 40)

    metadata = extract_item_metadata(properties)

    # Step 4: Create final dataset
    logger.info("-" * 40)
    logger.info("Step 4: Creating final dataset")
    logger.info("-" * 40)

    final_df = create_final_dataset(events, metadata, categories)

    # Save processed events
    processed_path = PROJECT_ROOT / config["paths"]["processed"]["events"]
    final_df.write_parquet(processed_path)
    logger.info(f"Saved processed events to {processed_path}")

    # Step 5: Split data
    logger.info("-" * 40)
    logger.info("Step 5: Splitting data")
    logger.info("-" * 40)

    split_config = config["split"]
    train_df, valid_df, test_df = temporal_split(
        final_df,
        train_ratio=split_config["train_ratio"],
        valid_ratio=split_config["valid_ratio"],
        test_ratio=split_config["test_ratio"],
    )

    # Save splits
    processed_paths = config["paths"]["processed"]

    train_path = PROJECT_ROOT / processed_paths["train"]
    valid_path = PROJECT_ROOT / processed_paths["valid"]
    test_path = PROJECT_ROOT / processed_paths["test"]

    train_df.write_parquet(train_path)
    valid_df.write_parquet(valid_path)
    test_df.write_parquet(test_path)

    logger.info(f"Saved train to {train_path}")
    logger.info(f"Saved valid to {valid_path}")
    logger.info(f"Saved test to {test_path}")

    # Summary
    logger.info("=" * 60)
    logger.info("Preprocessing complete!")
    logger.info("=" * 60)
    logger.info(f"Total events: {len(final_df):,}")
    logger.info(f"Train: {len(train_df):,} ({len(train_df)/len(final_df)*100:.1f}%)")
    logger.info(f"Valid: {len(valid_df):,} ({len(valid_df)/len(final_df)*100:.1f}%)")
    logger.info(f"Test: {len(test_df):,} ({len(test_df)/len(final_df)*100:.1f}%)")
    logger.info(f"Output files in: {output_dir}")


if __name__ == "__main__":
    main()
