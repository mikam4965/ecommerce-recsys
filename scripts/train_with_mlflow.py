"""Train recommendation models with MLflow tracking.

This script trains multiple model configurations and logs
all parameters, metrics, and artifacts to MLflow for comparison.
"""

import sys
from pathlib import Path
import time

import polars as pl
from loguru import logger
import mlflow

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.utils.mlflow_utils import (
    setup_mlflow,
    log_model_params,
    log_metrics,
    log_artifact,
)
from src.evaluation.evaluator import RecommenderEvaluator
from src.models.collaborative import ALSRecommender
from src.models.hybrid import HybridRecommender


def load_data():
    """Load train and test data."""
    train = pl.read_parquet(project_root / "data/processed/train.parquet")
    test = pl.read_parquet(project_root / "data/processed/test.parquet")
    logger.info(f"Train: {len(train):,} events, Test: {len(test):,} events")
    return train, test


def train_and_evaluate(
    model,
    train: pl.DataFrame,
    test: pl.DataFrame,
    model_name: str,
    n_users: int = 1000,
    use_item_features: bool = False,
    use_user_features: bool = False,
):
    """Train model and log to MLflow.

    Args:
        model: Recommender model instance
        train: Training data
        test: Test data
        model_name: Name for the MLflow run
        n_users: Number of users for evaluation
        use_item_features: Whether to use item features (for Hybrid)
        use_user_features: Whether to use user features (for Hybrid)
    """
    with mlflow.start_run(run_name=model_name):
        # Log model parameters
        log_model_params(model)
        mlflow.set_tag("model_name", model_name)
        mlflow.set_tag("use_item_features", str(use_item_features))
        mlflow.set_tag("use_user_features", str(use_user_features))

        # Train
        logger.info(f"Training {model_name}...")
        start_time = time.time()

        if hasattr(model, "fit") and "use_item_features" in model.fit.__code__.co_varnames:
            model.fit(
                train,
                use_item_features=use_item_features,
                use_user_features=use_user_features,
            )
        else:
            model.fit(train)

        train_time = time.time() - start_time
        mlflow.log_metric("train_time_sec", train_time)
        logger.info(f"Training completed in {train_time:.1f}s")

        # Evaluate
        logger.info(f"Evaluating {model_name}...")
        evaluator = RecommenderEvaluator(k_values=[5, 10, 20])

        start_time = time.time()
        results = evaluator.evaluate(model, test, n_users=n_users)
        eval_time = time.time() - start_time
        mlflow.log_metric("eval_time_sec", eval_time)

        # Log all metrics (evaluate returns dict directly)
        metrics = results.copy()
        metrics.pop("model", None)
        metrics.pop("n_users", None)
        metrics.pop("eval_time_sec", None)
        log_metrics(metrics)

        # Print key metrics
        p10 = metrics.get("Precision@10", 0)
        r10 = metrics.get("Recall@10", 0)
        ndcg10 = metrics.get("NDCG@10", 0)
        logger.info(f"{model_name} - P@10: {p10:.4f}, R@10: {r10:.4f}, NDCG@10: {ndcg10:.4f}")

        return metrics


def main():
    """Train multiple model configurations with MLflow tracking."""
    logger.info("=" * 70)
    logger.info("MLFLOW EXPERIMENT TRACKING")
    logger.info("=" * 70)

    # Initialize MLflow
    setup_mlflow()

    # Load data
    train, test = load_data()

    # 1. ALS Baseline
    logger.info("\n[1/4] Training ALS Baseline...")
    als = ALSRecommender(factors=64, iterations=15)
    train_and_evaluate(als, train, test, "ALS-baseline")

    # 2. Hybrid (CF+Items) - Same config as Stage 3
    # Key: use_item_features=True, use_user_features=False
    logger.info("\n[2/4] Training Hybrid (CF+Items)...")
    hybrid_cf_items = HybridRecommender(
        factors=64,
        iterations=15,
        cf_weight=0.7,
        feature_weight=0.3,
    )
    train_and_evaluate(
        hybrid_cf_items, train, test,
        "Hybrid-CF+Items",
        use_item_features=True,
        use_user_features=False,  # Critical: same as Stage 3
    )

    # 3. Hybrid variant with different cf_weight
    logger.info("\n[3/4] Training Hybrid (CF+Items) cf=0.8...")
    hybrid_v2 = HybridRecommender(
        factors=64,
        iterations=15,
        cf_weight=0.8,
        feature_weight=0.2,
    )
    train_and_evaluate(
        hybrid_v2, train, test,
        "Hybrid-CF+Items-cf0.8",
        use_item_features=True,
        use_user_features=False,
    )

    # 4. Hybrid variant with more factors
    logger.info("\n[4/4] Training Hybrid (CF+Items) f=128...")
    hybrid_v3 = HybridRecommender(
        factors=128,
        iterations=20,
        cf_weight=0.7,
        feature_weight=0.3,
    )
    train_and_evaluate(
        hybrid_v3, train, test,
        "Hybrid-CF+Items-f128",
        use_item_features=True,
        use_user_features=False,
    )

    # Summary
    logger.info("\n" + "=" * 70)
    logger.info("TRAINING COMPLETE")
    logger.info("=" * 70)
    logger.info("\nTo view results, run:")
    logger.info("  mlflow ui --port 5000")
    logger.info("\nThen open http://localhost:5000 in your browser")


if __name__ == "__main__":
    main()
