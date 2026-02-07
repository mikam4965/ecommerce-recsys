"""Script to evaluate cold start performance of different models."""

import sys
from pathlib import Path

import polars as pl
from loguru import logger

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.evaluation.cold_start import (
    cold_item_analysis,
    compare_cold_start_models,
    evaluate_cold_items,
    evaluate_cold_start,
    get_user_interaction_stats,
    split_by_interaction_count,
)
from src.models.collaborative import ALSRecommender
from src.models.hybrid import HybridRecommender


def main():
    """Evaluate and compare cold start performance."""
    logger.info("=" * 70)
    logger.info("Cold Start Evaluation")
    logger.info("=" * 70)

    # ========================================
    # 1. Load data
    # ========================================
    logger.info("\n" + "=" * 70)
    logger.info("STEP 1: Loading data")
    logger.info("=" * 70)

    train_path = project_root / "data" / "processed" / "train.parquet"
    test_path = project_root / "data" / "processed" / "test.parquet"
    rfm_path = project_root / "data" / "processed" / "user_segments.parquet"

    if not train_path.exists():
        logger.error(f"Training data not found: {train_path}")
        logger.info("Run 'python scripts/preprocess.py' first")
        return

    train_events = pl.read_parquet(train_path)
    test_events = pl.read_parquet(test_path)

    logger.info(f"Training events: {len(train_events):,}")
    logger.info(f"Test events: {len(test_events):,}")

    # Load RFM data
    rfm_data = None
    if rfm_path.exists():
        rfm_data = pl.read_parquet(rfm_path)
        logger.info(f"RFM data: {len(rfm_data):,} users")

    # ========================================
    # 2. Analyze user distribution
    # ========================================
    logger.info("\n" + "=" * 70)
    logger.info("STEP 2: Analyzing user distribution")
    logger.info("=" * 70)

    user_groups = split_by_interaction_count(
        train_events,
        cold_threshold=5,
        warm_threshold=20,
    )

    print("\nUser distribution by interaction count:")
    print(f"  Cold users (<5 interactions):    {len(user_groups['cold']):,}")
    print(f"  Warm users (5-20 interactions):  {len(user_groups['warm']):,}")
    print(f"  Hot users (>20 interactions):    {len(user_groups['hot']):,}")

    # User stats
    stats = get_user_interaction_stats(train_events)
    print("\nInteraction statistics:")
    print(f"  Mean interactions per user:  {stats['n_interactions'].mean():.1f}")
    print(f"  Median interactions:         {stats['n_interactions'].median():.1f}")
    print(f"  Max interactions:            {stats['n_interactions'].max()}")

    # ========================================
    # 3. Analyze item distribution
    # ========================================
    logger.info("\n" + "=" * 70)
    logger.info("STEP 3: Analyzing item distribution")
    logger.info("=" * 70)

    item_groups = cold_item_analysis(
        train_events,
        cold_threshold=10,
        popular_threshold=100,
    )

    print("\nItem distribution by interaction count:")
    print(f"  Cold items (<10 interactions):    {len(item_groups['cold_items']):,}")
    print(f"  Warm items (10-100 interactions): {len(item_groups['warm_items']):,}")
    print(f"  Popular items (>100 interactions):{len(item_groups['popular_items']):,}")

    # ========================================
    # 4. Train models
    # ========================================
    logger.info("\n" + "=" * 70)
    logger.info("STEP 4: Training models")
    logger.info("=" * 70)

    models = {}

    # ALS baseline
    logger.info("\n--- Training ALS (baseline) ---")
    als_model = ALSRecommender(
        factors=64,
        regularization=0.01,
        iterations=15,
        random_state=42,
    )
    als_model.fit(train_events)
    models["ALS (baseline)"] = als_model

    # Hybrid (CF only)
    logger.info("\n--- Training Hybrid (CF only) ---")
    hybrid_cf = HybridRecommender(
        factors=64,
        regularization=0.01,
        iterations=15,
        random_state=42,
        cf_weight=1.0,
        feature_weight=0.0,
    )
    hybrid_cf.fit(
        train_events,
        rfm_data=None,
        use_item_features=False,
        use_user_features=False,
    )
    models["Hybrid (CF)"] = hybrid_cf

    # Hybrid (CF + item features)
    logger.info("\n--- Training Hybrid (CF + items) ---")
    hybrid_items = HybridRecommender(
        factors=64,
        regularization=0.01,
        iterations=15,
        random_state=42,
        cf_weight=0.7,
        feature_weight=0.3,
    )
    hybrid_items.fit(
        train_events,
        rfm_data=None,
        use_item_features=True,
        use_user_features=False,
    )
    models["Hybrid (CF+Items)"] = hybrid_items

    # Hybrid (full) if RFM data available
    if rfm_data is not None:
        logger.info("\n--- Training Hybrid (full) ---")
        hybrid_full = HybridRecommender(
            factors=64,
            regularization=0.01,
            iterations=15,
            random_state=42,
            cf_weight=0.7,
            feature_weight=0.3,
        )
        hybrid_full.fit(
            train_events,
            rfm_data=rfm_data,
            use_item_features=True,
            use_user_features=True,
        )
        models["Hybrid (Full)"] = hybrid_full

    # ========================================
    # 5. Evaluate cold start
    # ========================================
    logger.info("\n" + "=" * 70)
    logger.info("STEP 5: Evaluating cold start performance")
    logger.info("=" * 70)

    results = compare_cold_start_models(
        models,
        test_events,
        train_events,
        k_values=[10],
        n_users_per_group=500,
    )

    # ========================================
    # 6. Print results
    # ========================================
    logger.info("\n" + "=" * 70)
    logger.info("COLD START COMPARISON RESULTS")
    logger.info("=" * 70)

    # Pivot for better display
    print("\n{:<20} {:>8} {:>12} {:>12} {:>12} {:>12}".format(
        "Model", "Group", "N Users", "P@10", "R@10", "HR@10"
    ))
    print("-" * 80)

    for row in results.sort(["model", "group"]).iter_rows(named=True):
        print("{:<20} {:>8} {:>12} {:>12.4f} {:>12.4f} {:>12.4f}".format(
            row["model"][:20],
            row["group"],
            row.get("n_users", 0),
            row.get("Precision@10", 0),
            row.get("Recall@10", 0),
            row.get("HitRate@10", 0),
        ))

    # ========================================
    # 7. Test new user recommendations
    # ========================================
    logger.info("\n" + "=" * 70)
    logger.info("STEP 6: Testing new user recommendations")
    logger.info("=" * 70)

    # Get sample session items
    best_model = models.get("Hybrid (Full)", models.get("Hybrid (CF+Items)"))
    sample_items = list(best_model.item_to_idx.keys())[:5]

    print("\n--- recommend_for_new_user() ---")
    print(f"Session items: {sample_items}")
    new_user_recs = best_model.recommend_for_new_user(sample_items, n=10)
    print(f"Recommendations: {new_user_recs}")

    # ========================================
    # 8. Test new item recommendations
    # ========================================
    print("\n--- recommend_new_item() ---")

    # Get a sample category
    if best_model.item_categories:
        sample_category = list(best_model.item_categories.values())[0]
        print(f"New item category: {sample_category}")

        target_users = best_model.recommend_new_item(
            {"category_id": sample_category},
            n=10
        )
        print(f"Target users: {target_users}")

    # ========================================
    # 9. Cold item coverage
    # ========================================
    logger.info("\n" + "=" * 70)
    logger.info("STEP 7: Evaluating cold item coverage")
    logger.info("=" * 70)

    cold_coverage = evaluate_cold_items(
        best_model,
        test_events,
        train_events,
        item_groups["cold_items"],
        n_users=500,
        n_recs=20,
    )

    print("\nCold item coverage:")
    print(f"  Total cold items:           {cold_coverage['total_cold_items']:,}")
    print(f"  Cold items recommended:     {cold_coverage['cold_items_recommended']:,}")
    print(f"  Cold item coverage:         {cold_coverage['cold_item_coverage']:.2%}")
    print(f"  Cold items in recs ratio:   {cold_coverage['cold_items_in_recs_ratio']:.4%}")

    # ========================================
    # 10. Save results
    # ========================================
    logger.info("\n" + "=" * 70)
    logger.info("STEP 8: Saving results")
    logger.info("=" * 70)

    results_path = project_root / "data" / "processed" / "cold_start_comparison.csv"
    results.write_csv(results_path)
    logger.info(f"Results saved to {results_path}")

    logger.info("\n" + "=" * 70)
    logger.info("Evaluation complete!")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
