"""Script to run RFM segmentation analysis."""

import sys
from pathlib import Path

import polars as pl
from loguru import logger

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.analysis.rfm import (
    get_segment_stats,
    run_rfm_analysis,
    visualize_segments,
)


def main():
    """Run RFM analysis."""
    logger.info("=" * 70)
    logger.info("RFM сегментациясын талдау")
    logger.info("=" * 70)

    # Load training data
    train_path = project_root / "data" / "processed" / "train.parquet"

    if not train_path.exists():
        logger.error(f"Training data not found: {train_path}")
        logger.info("Run 'python scripts/preprocess.py' first")
        return

    logger.info(f"Loading data from {train_path}")
    events = pl.read_parquet(train_path)
    logger.info(f"Loaded {len(events):,} events")

    # Run RFM analysis
    results = run_rfm_analysis(events)

    if results["n_users"] == 0:
        logger.warning("No users found with transactions")
        return

    # Print segment distribution
    stats = results["segment_stats"]

    logger.info("\n" + "=" * 70)
    logger.info("СЕГМЕНТ ТАРАЛУЫ")
    logger.info("=" * 70)

    print("\n{:<20} {:>10} {:>10} {:>12} {:>12} {:>12}".format(
        "Сегмент", "Пайд-лар", "%", "Орт жақынд", "Орт жиілік", "Орт ақшалай"
    ))
    print("-" * 78)

    for row in stats.iter_rows(named=True):
        print("{:<20} {:>10,} {:>9.1f}% {:>12.1f} {:>12.1f} {:>12.1f}".format(
            row["segment"],
            row["count"],
            row["percentage"],
            row["avg_recency"],
            row["avg_frequency"],
            row["avg_monetary"],
        ))

    print("-" * 78)
    print(f"Барлық пайдаланушылар: {results['n_users']:,}")

    # Save visualization
    viz_path = project_root / "data" / "processed" / "rfm_segments.png"
    visualize_segments(results["rfm_data"], save_path=viz_path)
    logger.info(f"\nVisualization saved to {viz_path}")

    # Save RFM data
    rfm_path = project_root / "data" / "processed" / "rfm_segmentation.parquet"
    results["rfm_data"].write_parquet(rfm_path)
    logger.info(f"RFM data saved to {rfm_path}")

    # Print average metrics by segment
    logger.info("\n" + "=" * 70)
    logger.info("СЕГМЕНТ БОЙЫНША ОРТАША RFM ҰПАЙЛАРЫ")
    logger.info("=" * 70)

    print("\n{:<20} {:>10} {:>10} {:>10}".format(
        "Сегмент", "R ұпай", "F ұпай", "M ұпай"
    ))
    print("-" * 52)

    for row in stats.iter_rows(named=True):
        print("{:<20} {:>10.2f} {:>10.2f} {:>10.2f}".format(
            row["segment"],
            row["avg_r_score"],
            row["avg_f_score"],
            row["avg_m_score"],
        ))

    logger.info("\n" + "=" * 70)
    logger.info("RFM талдау аяқталды!")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
