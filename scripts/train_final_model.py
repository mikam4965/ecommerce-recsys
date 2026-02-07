"""Train final model with best Optuna parameters.

This script loads the best hyperparameters from Optuna optimization,
trains a HybridRecommender model, evaluates it on test data,
and saves the model for deployment.
"""

import sys
from pathlib import Path
import time

import yaml
import polars as pl
from loguru import logger

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.models.hybrid import HybridRecommender
from src.evaluation.evaluator import RecommenderEvaluator


def load_best_params() -> dict:
    """Load best hyperparameters from Optuna optimization."""
    params_path = project_root / "configs" / "best_params.yaml"

    if not params_path.exists():
        raise FileNotFoundError(f"Best params not found: {params_path}")

    with open(params_path) as f:
        config = yaml.safe_load(f)

    return config["hybrid_recommender"]


def sample_data(df: pl.DataFrame, max_users: int = 50000) -> pl.DataFrame:
    """Sample top active users to manage memory."""
    top_users = (
        df.group_by("user_id")
        .agg(pl.len().alias("event_count"))
        .sort("event_count", descending=True)
        .head(max_users)
        .select("user_id")
    )
    return df.join(top_users, on="user_id", how="inner")


def main():
    """Train final model with best parameters."""
    import gc

    logger.info("=" * 70)
    logger.info("TRAINING FINAL MODEL WITH BEST PARAMETERS")
    logger.info("=" * 70)

    # Load best parameters
    logger.info("Loading best parameters...")
    best_params = load_best_params()
    logger.info(f"Best NDCG@10: {best_params.get('best_ndcg_at_10', 'N/A')}")
    logger.info(f"Parameters: factors={best_params.get('factors')}, "
                f"iterations={best_params.get('iterations')}, "
                f"regularization={best_params.get('regularization'):.6f}")

    # Load and sample train data (use scan for memory efficiency)
    logger.info("\nLoading train data...")
    train_sampled = (
        pl.scan_parquet(project_root / "data/processed/train.parquet")
        .head(500000)  # Limit to first 500K events
        .collect()
    )
    logger.info(f"Train: {len(train_sampled):,} events")
    gc.collect()

    # Load and sample test data
    logger.info("Loading test data...")
    test_sampled = (
        pl.scan_parquet(project_root / "data/processed/test.parquet")
        .head(200000)  # Limit to first 200K events
        .collect()
    )
    logger.info(f"Test: {len(test_sampled):,} events")
    gc.collect()

    # Create model with best parameters (reduced factors for memory)
    model = HybridRecommender(
        factors=min(32, best_params.get("factors", 32)),  # Cap at 32 for memory
        iterations=best_params.get("iterations", 40),
        regularization=best_params.get("regularization", 0.01),
        cf_weight=best_params.get("cf_weight", 0.7),
        feature_weight=best_params.get("feature_weight", 0.3),
        random_state=42,
    )

    # Train model
    logger.info("\nTraining model...")
    start_time = time.time()
    model.fit(
        train_sampled,
        use_item_features=best_params.get("use_item_features", False),
        use_user_features=best_params.get("use_user_features", False),
    )
    train_time = time.time() - start_time
    logger.info(f"Training completed in {train_time:.1f}s")

    # Evaluate on test data
    logger.info("\nEvaluating on test data...")
    evaluator = RecommenderEvaluator(k_values=[5, 10, 20])
    results = evaluator.evaluate(model, test_sampled, n_users=500, show_progress=True)

    logger.info("\n" + "=" * 50)
    logger.info("FINAL MODEL RESULTS")
    logger.info("=" * 50)
    for metric in ["Precision@10", "Recall@10", "NDCG@10", "HitRate@10", "MRR@10"]:
        if metric in results:
            logger.info(f"  {metric}: {results[metric]:.4f}")

    # Save model
    model_path = project_root / "models" / "hybrid_best.pkl"
    model_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info(f"\nSaving model to {model_path}...")
    model.save(str(model_path))
    logger.info("Model saved successfully!")

    # Summary
    logger.info("\n" + "=" * 70)
    logger.info("TRAINING COMPLETE")
    logger.info("=" * 70)
    logger.info(f"Model: {model_path}")
    logger.info(f"NDCG@10: {results.get('NDCG@10', 0):.4f}")
    logger.info(f"Training time: {train_time:.1f}s")


if __name__ == "__main__":
    main()
