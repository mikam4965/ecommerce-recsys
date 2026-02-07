"""Script to train LightFM hybrid model (runs in WSL/Linux)."""

import sys
from pathlib import Path

import polars as pl
from loguru import logger

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.evaluation.evaluator import RecommenderEvaluator
from src.models.hybrid import TrueLightFMRecommender


def main():
    """Train and evaluate LightFM hybrid model."""
    logger.info("=" * 70)
    logger.info("LightFM Hybrid Model Training (WSL)")
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
        return

    if not test_path.exists():
        logger.error(f"Test data not found: {test_path}")
        return

    train_events = pl.read_parquet(train_path)
    test_events = pl.read_parquet(test_path)

    logger.info(f"Training events: {len(train_events):,}")
    logger.info(f"Test events: {len(test_events):,}")

    # Filter to users with transactions
    logger.info("Filtering to users with transactions...")
    active_users = (
        train_events
        .filter(pl.col("event_type") == "transaction")
        .select("user_id")
        .unique()
    )
    logger.info(f"Active users (with transactions): {len(active_users):,}")

    train_events = train_events.join(active_users, on="user_id", how="inner")
    test_events = test_events.join(active_users, on="user_id", how="inner")

    logger.info(f"Filtered training events: {len(train_events):,}")
    logger.info(f"Filtered test events: {len(test_events):,}")

    # Load RFM data
    rfm_data = None
    if rfm_path.exists():
        rfm_data = pl.read_parquet(rfm_path)
        logger.info(f"RFM data: {len(rfm_data):,} users")
    else:
        logger.warning("RFM data not found")

    # ========================================
    # 2. Train LightFM variants
    # ========================================
    logger.info("\n" + "=" * 70)
    logger.info("STEP 2: Training LightFM model variants")
    logger.info("=" * 70)

    models = {}

    # Variant 1: CF only (no features)
    logger.info("\n--- Training LightFM (CF only) ---")
    model_cf = TrueLightFMRecommender(
        no_components=64,
        loss="warp",
        epochs=30,
        learning_rate=0.05,
        random_state=42,
    )
    model_cf.fit(
        train_events,
        rfm_data=None,
        use_item_features=False,
        use_user_features=False,
    )
    models["LightFM (CF)"] = model_cf

    # Variant 2: CF + item features
    logger.info("\n--- Training LightFM (CF + item features) ---")
    model_item = TrueLightFMRecommender(
        no_components=64,
        loss="warp",
        epochs=30,
        learning_rate=0.05,
        random_state=42,
    )
    model_item.fit(
        train_events,
        rfm_data=None,
        use_item_features=True,
        use_user_features=False,
    )
    models["LightFM (CF+Items)"] = model_item

    # Variant 3: Full hybrid
    if rfm_data is not None:
        logger.info("\n--- Training LightFM (full hybrid) ---")
        model_hybrid = TrueLightFMRecommender(
            no_components=64,
            loss="warp",
            epochs=30,
            learning_rate=0.05,
            random_state=42,
        )
        model_hybrid.fit(
            train_events,
            rfm_data=rfm_data,
            use_item_features=True,
            use_user_features=True,
        )
        models["LightFM (Full)"] = model_hybrid

    # ========================================
    # 3. Evaluate models
    # ========================================
    logger.info("\n" + "=" * 70)
    logger.info("STEP 3: Evaluating models")
    logger.info("=" * 70)

    evaluator = RecommenderEvaluator(k_values=[5, 10, 20])

    results_df = evaluator.compare_models(
        models,
        test_events,
        n_users=1000,
        show_progress=True,
    )

    # Print results
    logger.info("\n" + "=" * 70)
    logger.info("COMPARISON RESULTS")
    logger.info("=" * 70)

    print("\n{:<25} {:>12} {:>12} {:>12} {:>12}".format(
        "Model", "Precision@10", "Recall@10", "NDCG@10", "HitRate@10"
    ))
    print("-" * 77)

    for row in results_df.iter_rows(named=True):
        print("{:<25} {:>12.4f} {:>12.4f} {:>12.4f} {:>12.4f}".format(
            row["model"][:25],
            row.get("Precision@10", 0),
            row.get("Recall@10", 0),
            row.get("NDCG@10", 0),
            row.get("HitRate@10", 0),
        ))

    # ========================================
    # 4. Test features
    # ========================================
    logger.info("\n" + "=" * 70)
    logger.info("STEP 4: Testing additional features")
    logger.info("=" * 70)

    best_model = models.get("LightFM (Full)", models.get("LightFM (CF+Items)"))
    sample_users = list(best_model.user_to_idx.keys())[:3]
    sample_items = list(best_model.item_to_idx.keys())[:3]

    print("\n--- Sample Recommendations ---")
    for user_id in sample_users:
        recs = best_model.recommend(user_id, n=5)
        print(f"User {user_id}: {recs}")

    print("\n--- Similar Items ---")
    for item_id in sample_items:
        similar = best_model.similar_items(item_id, n=5)
        print(f"Item {item_id} similar to: {similar}")

    if "LightFM (Full)" in models and rfm_data is not None:
        print("\n--- Cold Start Recommendations ---")
        segments = rfm_data["segment"].unique().to_list()[:2]

        for segment in segments:
            cold_recs = models["LightFM (Full)"].recommend_cold_start(
                {"segment": segment},
                n=5
            )
            print(f"New user ({segment}): {cold_recs}")

    # ========================================
    # 5. Save best model
    # ========================================
    logger.info("\n" + "=" * 70)
    logger.info("STEP 5: Saving model")
    logger.info("=" * 70)

    models_dir = project_root / "models"
    models_dir.mkdir(exist_ok=True)

    model_path = models_dir / "lightfm_model.pkl"
    best_model.save(model_path)

    logger.info(f"Model saved to {model_path}")

    # Save results
    results_path = project_root / "data" / "processed" / "lightfm_comparison.csv"
    results_df.write_csv(results_path)
    logger.info(f"Results saved to {results_path}")

    logger.info("\n" + "=" * 70)
    logger.info("Training complete!")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
