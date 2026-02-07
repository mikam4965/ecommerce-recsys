"""Script to run category co-occurrence heatmap analysis."""

import sys
from pathlib import Path

import polars as pl
from loguru import logger

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.analysis.heatmaps import (
    plot_heatmap,
    run_cooccurrence_analysis,
)


def main():
    """Run heatmap analysis."""
    logger.info("=" * 70)
    logger.info("Санаттар бірлесу жылу картасын талдау")
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

    # Run analysis
    results = run_cooccurrence_analysis(
        events,
        top_n_categories=20,
        event_type="transaction",
    )

    # Print top categories
    logger.info("\n" + "=" * 70)
    logger.info("ТРАНЗАКЦИЯ САНЫ БОЙЫНША ТОП-20 САНАТ")
    logger.info("=" * 70)

    print("\nТалданған санаттар:", ", ".join(results["categories"]))

    # Print top pairs
    logger.info("\n" + "=" * 70)
    logger.info("БІРЛЕСУ БОЙЫНША ТОП-10 САНАТ ЖҰПТАРЫ")
    logger.info("=" * 70)

    top_pairs = results["top_pairs"]

    print("\n{:<15} {:<15} {:>15}".format(
        "1-санат", "2-санат", "Бірлескен сатып алулар"
    ))
    print("-" * 47)

    for row in top_pairs.iter_rows(named=True):
        print("{:<15} {:<15} {:>15,}".format(
            row["category_1"],
            row["category_2"],
            row["count"],
        ))

    # Save heatmap visualization
    heatmap_path = project_root / "data" / "processed" / "category_heatmap.html"
    plot_heatmap(
        results["matrix"],
        results["categories"],
        save_path=heatmap_path,
        title="Топ-20 санат: Бірлесу жылу картасы",
        show_values=False,
        colorscale="Blues",
    )

    logger.info(f"\nHeatmap saved to {heatmap_path}")

    # Print interpretation
    logger.info("\n" + "=" * 70)
    logger.info("ТОП-3 КҮШТІ БАЙЛАНЫСТАРДЫ ТҮСІНДІРУ")
    logger.info("=" * 70)

    if len(top_pairs) >= 3:
        for i, row in enumerate(top_pairs.head(3).iter_rows(named=True), 1):
            print(f"\n{i}. Санат {row['category_1']} <-> Санат {row['category_2']}")
            print(f"   Бірлескен сатып алулар: {row['count']:,} сессия")
            print(f"   Бұл санаттар бір сессияда жиі бірге сатып алынады.")

    logger.info("\n" + "=" * 70)
    logger.info("Талдау аяқталды!")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
