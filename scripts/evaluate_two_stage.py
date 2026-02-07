"""Compare HybridRecommender alone vs TwoStageRecommender.

This script:
1. Trains both models on the same data
2. Evaluates on test set
3. Compares NDCG@10, Precision@10, Recall@10
4. Saves results to CSV
"""

import sys
from pathlib import Path

import polars as pl
from loguru import logger

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.evaluation.evaluator import RecommenderEvaluator
from src.models.hybrid import HybridRecommender
from src.models.pipeline import TwoStageRecommender


def main():
    """Run comparison evaluation."""
    logger.info("=" * 70)
    logger.info("Two-Stage Recommender Comparison")
    logger.info("HybridRecommender vs TwoStageRecommender")
    logger.info("=" * 70)

    # Load data
    logger.info("\n--- Loading Data ---")
    train_events = pl.read_parquet(project_root / "data/processed/train.parquet")
    test_events = pl.read_parquet(project_root / "data/processed/test.parquet")

    logger.info(f"Train events: {len(train_events):,}")
    logger.info(f"Test events: {len(test_events):,}")

    # Load RFM data if available
    rfm_path = project_root / "data/processed/user_segments.parquet"
    if rfm_path.exists():
        rfm_data = pl.read_parquet(rfm_path)
        logger.info(f"RFM data: {len(rfm_data):,} users")
    else:
        rfm_path = project_root / "data/processed/rfm_segmentation.parquet"
        if rfm_path.exists():
            rfm_data = pl.read_parquet(rfm_path)
            logger.info(f"RFM data: {len(rfm_data):,} users")
        else:
            rfm_data = None
            logger.warning("No RFM data found")

    # Filter to users with transactions for better evaluation
    logger.info("\n--- Filtering to Active Users ---")
    train_tx_users = (
        train_events.filter(pl.col("event_type") == "transaction")
        .select("user_id")
        .unique()
    )
    test_tx_users = (
        test_events.filter(pl.col("event_type") == "transaction")
        .select("user_id")
        .unique()
    )

    # Users who have transactions in both train and test
    common_users = train_tx_users.join(test_tx_users, on="user_id", how="inner")
    logger.info(f"Users with transactions in both train and test: {len(common_users):,}")

    # Filter events to these users
    train_filtered = train_events.join(common_users, on="user_id", how="inner")
    test_filtered = test_events.join(common_users, on="user_id", how="inner")

    logger.info(f"Filtered train events: {len(train_filtered):,}")
    logger.info(f"Filtered test events: {len(test_filtered):,}")

    # Train HybridRecommender
    logger.info("\n" + "=" * 70)
    logger.info("Training HybridRecommender (baseline)")
    logger.info("=" * 70)

    hybrid = HybridRecommender(
        factors=64,
        iterations=15,
        cf_weight=0.7,
        feature_weight=0.3,
    )
    hybrid.fit(
        train_filtered,
        rfm_data=rfm_data,
        use_item_features=True,
        use_user_features=rfm_data is not None,
    )

    # Train TwoStageRecommender
    logger.info("\n" + "=" * 70)
    logger.info("Training TwoStageRecommender")
    logger.info("=" * 70)

    two_stage = TwoStageRecommender(
        factors=64,
        iterations=15,
        cf_weight=0.7,
        feature_weight=0.3,
        ranker_iterations=300,
        ranker_depth=6,
        n_candidates=100,
        n_negatives=4,
    )
    two_stage.fit(
        train_filtered,
        rfm_data=rfm_data,
        use_item_features=True,
        use_user_features=rfm_data is not None,
    )

    # Evaluate both models
    logger.info("\n" + "=" * 70)
    logger.info("Evaluation")
    logger.info("=" * 70)

    evaluator = RecommenderEvaluator(k_values=[5, 10, 20])

    models = {
        "HybridRecommender": hybrid,
        "TwoStageRecommender": two_stage,
    }

    results = evaluator.compare_models(
        models,
        test_filtered,
        n_users=1000,
        show_progress=True,
    )

    # Print comparison table
    print("\n" + "=" * 80)
    print("COMPARISON RESULTS")
    print("=" * 80)

    # Header
    print(f"\n{'Model':<25} {'NDCG@10':>12} {'Precision@10':>14} {'Recall@10':>12} {'HitRate@10':>12}")
    print("-" * 80)

    # Results
    for row in results.iter_rows(named=True):
        print(
            f"{row['model']:<25} "
            f"{row['NDCG@10']:>12.6f} "
            f"{row['Precision@10']:>14.6f} "
            f"{row['Recall@10']:>12.6f} "
            f"{row['HitRate@10']:>12.4f}"
        )

    # Calculate improvement
    print("\n" + "-" * 80)
    hybrid_row = results.filter(pl.col("model") == "HybridRecommender").row(0, named=True)
    twostage_row = results.filter(pl.col("model") == "TwoStageRecommender").row(0, named=True)

    if hybrid_row["NDCG@10"] > 0:
        ndcg_improvement = (twostage_row["NDCG@10"] - hybrid_row["NDCG@10"]) / hybrid_row["NDCG@10"] * 100
        prec_improvement = (twostage_row["Precision@10"] - hybrid_row["Precision@10"]) / max(hybrid_row["Precision@10"], 1e-10) * 100
        recall_improvement = (twostage_row["Recall@10"] - hybrid_row["Recall@10"]) / max(hybrid_row["Recall@10"], 1e-10) * 100

        print(f"\nTwoStageRecommender improvement over HybridRecommender:")
        print(f"  NDCG@10:      {ndcg_improvement:+.2f}%")
        print(f"  Precision@10: {prec_improvement:+.2f}%")
        print(f"  Recall@10:    {recall_improvement:+.2f}%")

    # Save results first (before printing full table)
    output_path = project_root / "data/processed/two_stage_comparison.csv"
    results.write_csv(output_path)
    logger.info(f"\nResults saved to {output_path}")

    # Full metrics table
    print("\n" + "=" * 80)
    print("FULL METRICS (all K values)")
    print("=" * 80)
    # Print columns explicitly to avoid encoding issues
    cols = results.columns
    print("  ".join(cols))
    for row in results.iter_rows():
        formatted = []
        for val in row:
            if isinstance(val, float):
                formatted.append(f"{val:.6f}")
            else:
                formatted.append(str(val))
        print("  ".join(formatted))

    return results


if __name__ == "__main__":
    main()
