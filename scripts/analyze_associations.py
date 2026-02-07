"""Script to run association rules analysis on RetailRocket data."""

import sys
from pathlib import Path

import polars as pl
from loguru import logger

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.analysis.association_rules import (
    get_top_rules_by_lift,
    run_association_analysis,
    visualize_top_rules,
)


def main():
    """Run association rules analysis."""
    logger.info("=" * 60)
    logger.info("Association Rules Analysis")
    logger.info("=" * 60)

    # Load training data
    train_path = project_root / "data" / "processed" / "train.parquet"

    if not train_path.exists():
        logger.error(f"Training data not found: {train_path}")
        logger.info("Run 'python scripts/preprocess.py' first")
        return

    logger.info(f"Loading data from {train_path}")
    events = pl.read_parquet(train_path)
    logger.info(f"Loaded {len(events):,} events")

    # Check event types distribution
    event_counts = events.group_by("event_type").agg(pl.len().alias("count"))
    logger.info("Event type distribution:")
    for row in event_counts.iter_rows(named=True):
        logger.info(f"  {row['event_type']}: {row['count']:,}")

    # Run analysis on transactions first
    logger.info("\n" + "=" * 60)
    logger.info("Analyzing TRANSACTION events")
    logger.info("=" * 60)

    results_transaction = run_association_analysis(
        events,
        event_type="transaction",
        min_support=0.005,  # Lower threshold for sparse data
        min_confidence=0.1,
        min_lift=1.0,
        max_itemset_len=3,
    )

    # If no transaction rules, try addtocart
    if results_transaction["n_rules"] == 0:
        logger.info("\n" + "=" * 60)
        logger.info("No transaction rules found, analyzing ADDTOCART events")
        logger.info("=" * 60)

        results_addtocart = run_association_analysis(
            events,
            event_type="addtocart",
            min_support=0.005,
            min_confidence=0.1,
            min_lift=1.0,
            max_itemset_len=3,
        )
        results = results_addtocart
        event_type_used = "addtocart"
    else:
        results = results_transaction
        event_type_used = "transaction"

    # Print summary
    logger.info("\n" + "=" * 60)
    logger.info("Analysis Summary")
    logger.info("=" * 60)
    logger.info(f"Event type analyzed: {event_type_used}")
    logger.info(f"Transactions analyzed: {results['n_transactions']:,}")
    logger.info(f"Frequent itemsets found: {results['n_itemsets']:,}")
    logger.info(f"Association rules found: {results['n_rules']:,}")

    # Display top rules
    if results["n_rules"] > 0:
        logger.info("\n" + "=" * 60)
        logger.info("Top 10 Rules by Lift")
        logger.info("=" * 60)

        top_rules = get_top_rules_by_lift(results["rules"], top_n=10)

        for i, row in enumerate(top_rules.iter_rows(named=True), 1):
            ant = ", ".join(map(str, row["antecedents"]))
            cons = ", ".join(map(str, row["consequents"]))
            logger.info(
                f"{i:2}. {ant} -> {cons}\n"
                f"    Support: {row['support']:.4f}, "
                f"Confidence: {row['confidence']:.4f}, "
                f"Lift: {row['lift']:.2f}"
            )

        # Save visualization
        viz_path = project_root / "data" / "processed" / "association_rules.png"
        visualize_top_rules(results["rules"], top_n=20, save_path=viz_path)
        logger.info(f"\nVisualization saved to {viz_path}")

        # Save rules to parquet
        rules_path = project_root / "data" / "processed" / "association_rules.parquet"
        results["rules"].write_parquet(rules_path)
        logger.info(f"Rules saved to {rules_path}")
    else:
        logger.warning("No association rules found with current thresholds")
        logger.info("Try lowering min_support or min_confidence")

    logger.info("\n" + "=" * 60)
    logger.info("Analysis complete!")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
