"""Verification script for Stage 2: Behavioral Analysis."""

import sys
from pathlib import Path

import polars as pl
from loguru import logger

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.analysis.association_rules import run_association_analysis, get_top_rules_by_lift
from src.analysis.rfm import run_rfm_analysis, get_segment_stats
from src.analysis.funnel import (
    run_funnel_analysis,
    compare_segments,
    visualize_funnel,
    visualize_segment_comparison,
)
from src.analysis.heatmaps import run_cooccurrence_analysis, plot_heatmap


def main():
    """Run full Stage 2 verification."""
    logger.info("=" * 70)
    logger.info("STAGE 2 VERIFICATION: Behavioral Analysis")
    logger.info("=" * 70)

    # Create reports directory
    reports_dir = project_root / "reports"
    reports_dir.mkdir(exist_ok=True)

    # ========================================
    # 1. Load data
    # ========================================
    logger.info("\n" + "=" * 70)
    logger.info("STEP 1: Loading preprocessed data")
    logger.info("=" * 70)

    train_path = project_root / "data" / "processed" / "train.parquet"
    if not train_path.exists():
        logger.error(f"Training data not found: {train_path}")
        return False

    events = pl.read_parquet(train_path)
    logger.info(f"Loaded {len(events):,} events")
    logger.info(f"Columns: {events.columns}")

    # ========================================
    # 2. Association Rules
    # ========================================
    logger.info("\n" + "=" * 70)
    logger.info("STEP 2: Association Rules Analysis")
    logger.info("=" * 70)

    try:
        # Run on category level for more rules
        from src.analysis.association_rules import (
            find_frequent_itemsets,
            generate_rules,
            filter_rules,
        )

        # Prepare category transactions
        filtered = events.filter(pl.col("event_type").is_in(["transaction", "addtocart"]))
        filtered = filtered.filter(pl.col("category_id") != "unknown")

        transactions_df = (
            filtered
            .group_by("session_id")
            .agg(pl.col("category_id").unique().alias("categories"))
            .filter(pl.col("categories").list.len() >= 2)
        )

        transactions = []
        for row in transactions_df.iter_rows(named=True):
            cats = [int(c) for c in row["categories"] if c.isdigit()]
            if len(cats) >= 2:
                transactions.append(cats)

        logger.info(f"Transactions for analysis: {len(transactions):,}")

        # Find rules
        itemsets = find_frequent_itemsets(transactions, min_support=0.005, max_len=3)
        rules = generate_rules(itemsets, min_confidence=0.1)
        filtered_rules = filter_rules(rules, min_support=0.005, min_confidence=0.1, min_lift=1.0)

        logger.info(f"Rules found: {len(filtered_rules)}")

        # Top-10 by lift
        top_rules = get_top_rules_by_lift(filtered_rules, top_n=10)

        print("\nTOP-10 RULES BY LIFT:")
        print("-" * 60)
        for i, row in enumerate(top_rules.iter_rows(named=True), 1):
            ant = ", ".join(map(str, row["antecedents"]))
            cons = ", ".join(map(str, row["consequents"]))
            print(f"{i:2}. [{ant}] -> [{cons}]")
            print(f"    Support: {row['support']:.4f}, Confidence: {row['confidence']:.2%}, Lift: {row['lift']:.2f}")

        # Save to CSV
        csv_path = project_root / "data" / "processed" / "association_rules.csv"
        filtered_rules.write_csv(csv_path)
        logger.info(f"Saved rules to {csv_path}")

        assoc_ok = True
    except Exception as e:
        logger.error(f"Association rules failed: {e}")
        assoc_ok = False

    # ========================================
    # 3. RFM Segmentation
    # ========================================
    logger.info("\n" + "=" * 70)
    logger.info("STEP 3: RFM Segmentation")
    logger.info("=" * 70)

    try:
        results = run_rfm_analysis(events)

        logger.info(f"Users segmented: {results['n_users']:,}")

        # Print segment distribution
        stats = results["segment_stats"]
        print("\nSEGMENT DISTRIBUTION:")
        print("-" * 60)
        print(f"{'Segment':<20} {'Users':>10} {'%':>8} {'Avg Recency':>12}")
        print("-" * 60)
        for row in stats.iter_rows(named=True):
            print(f"{row['segment']:<20} {row['count']:>10,} {row['percentage']:>7.1f}% {row['avg_recency']:>12.1f}")

        # Save to Parquet
        parquet_path = project_root / "data" / "processed" / "user_segments.parquet"
        results["rfm_data"].write_parquet(parquet_path)
        logger.info(f"Saved segments to {parquet_path}")

        rfm_ok = True
        rfm_data = results["rfm_data"]
    except Exception as e:
        logger.error(f"RFM segmentation failed: {e}")
        rfm_ok = False
        rfm_data = None

    # ========================================
    # 4. Conversion Funnel
    # ========================================
    logger.info("\n" + "=" * 70)
    logger.info("STEP 4: Conversion Funnel")
    logger.info("=" * 70)

    try:
        results = run_funnel_analysis(events, rfm_data)

        # Overall funnel
        funnel = results["funnel"]
        rates = results["rates"]

        print("\nOVERALL FUNNEL:")
        print("-" * 40)
        for row in funnel.iter_rows(named=True):
            print(f"{row['stage']:<15} {row['users']:>12,} ({row['percentage']:.2f}%)")

        print(f"\nConversion Rates:")
        print(f"  View -> Cart:     {rates['view_to_cart']:.2f}%")
        print(f"  Cart -> Purchase: {rates['cart_to_purchase']:.2f}%")
        print(f"  View -> Purchase: {rates['view_to_purchase']:.2f}%")

        # Compare Champions vs At Risk
        if results["segment_funnel"] is not None:
            segment_funnel = results["segment_funnel"]
            comparison = compare_segments(segment_funnel, ["Champions", "At Risk"])

            print("\nCHAMPIONS vs AT RISK:")
            print("-" * 50)
            for row in comparison.iter_rows(named=True):
                print(f"{row['segment']:<15}: V->C={row['view_to_cart']:.1f}%, C->P={row['cart_to_purchase']:.1f}%, V->P={row['view_to_purchase']:.1f}%")

            # Save comparison visualization
            comparison_path = reports_dir / "funnel_analysis.html"
            visualize_segment_comparison(segment_funnel, ["Champions", "At Risk", "Lost"], save_path=comparison_path)
            logger.info(f"Saved funnel analysis to {comparison_path}")

        funnel_ok = True
    except Exception as e:
        logger.error(f"Funnel analysis failed: {e}")
        funnel_ok = False

    # ========================================
    # 5. Category Heatmap
    # ========================================
    logger.info("\n" + "=" * 70)
    logger.info("STEP 5: Category Heatmap (Top-15)")
    logger.info("=" * 70)

    try:
        results = run_cooccurrence_analysis(events, top_n_categories=15, event_type="transaction")

        logger.info(f"Categories analyzed: {results['n_categories']}")

        # Top pairs
        top_pairs = results["top_pairs"]
        print("\nTOP-5 CATEGORY PAIRS:")
        print("-" * 50)
        for row in top_pairs.head(5).iter_rows(named=True):
            print(f"  {row['category_1']} <-> {row['category_2']}: {row['count']} co-purchases")

        # Save heatmap
        heatmap_path = reports_dir / "category_heatmap.html"
        plot_heatmap(
            results["matrix"],
            results["categories"],
            save_path=heatmap_path,
            title="Top-15 Categories: Co-Occurrence Heatmap",
        )
        logger.info(f"Saved heatmap to {heatmap_path}")

        heatmap_ok = True
    except Exception as e:
        logger.error(f"Heatmap analysis failed: {e}")
        heatmap_ok = False

    # ========================================
    # Summary
    # ========================================
    logger.info("\n" + "=" * 70)
    logger.info("VERIFICATION SUMMARY")
    logger.info("=" * 70)

    results_table = [
        ("Association Rules", assoc_ok, "data/processed/association_rules.csv"),
        ("RFM Segmentation", rfm_ok, "data/processed/user_segments.parquet"),
        ("Conversion Funnel", funnel_ok, "reports/funnel_analysis.html"),
        ("Category Heatmap", heatmap_ok, "reports/category_heatmap.html"),
    ]

    print("\n{:<25} {:<10} {:<40}".format("Module", "Status", "Artifact"))
    print("-" * 77)
    for name, ok, artifact in results_table:
        status = "OK" if ok else "FAILED"
        artifact_path = project_root / artifact
        exists = "EXISTS" if artifact_path.exists() else "MISSING"
        print(f"{name:<25} {status:<10} {artifact} ({exists})")

    all_ok = all([assoc_ok, rfm_ok, funnel_ok, heatmap_ok])

    if all_ok:
        logger.info("\nALL MODULES PASSED!")
    else:
        logger.error("\nSOME MODULES FAILED!")

    return all_ok


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
