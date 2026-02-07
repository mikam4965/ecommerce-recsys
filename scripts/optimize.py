"""Hyperparameter optimization with Optuna for HybridRecommender.

This script uses Optuna to find optimal hyperparameters for the
HybridRecommender model, with MLflow integration for tracking.
"""

import gc
import sys
from pathlib import Path

import optuna
import polars as pl
import mlflow
import yaml
from loguru import logger

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.utils.mlflow_utils import setup_mlflow
from src.evaluation.evaluator import RecommenderEvaluator
from src.models.hybrid import HybridRecommender

# Global data (loaded once to avoid repeated I/O)
train_data = None
valid_data = None
evaluator = None


def load_data():
    """Load train and validation data."""
    global train_data, valid_data, evaluator

    train_data = pl.read_parquet(project_root / "data/processed/train.parquet")
    valid_data = pl.read_parquet(project_root / "data/processed/valid.parquet")
    evaluator = RecommenderEvaluator(k_values=[10])

    logger.info(f"Train: {len(train_data):,} events")
    logger.info(f"Valid: {len(valid_data):,} events")


def objective(trial: optuna.Trial) -> float:
    """Optuna objective function for HybridRecommender.

    Optimizes NDCG@10 on the validation set.

    Args:
        trial: Optuna trial object.

    Returns:
        NDCG@10 score (higher is better).
    """
    # Hyperparameters to optimize
    params = {
        "factors": trial.suggest_int("factors", 16, 128, step=16),
        "iterations": trial.suggest_int("iterations", 10, 50, step=5),
        "regularization": trial.suggest_float("regularization", 1e-4, 1e-1, log=True),
        "cf_weight": trial.suggest_float("cf_weight", 0.5, 1.0),
        "feature_weight": trial.suggest_float("feature_weight", 0.0, 0.5),
    }

    # Feature flags
    use_item_features = trial.suggest_categorical("use_item_features", [True, False])
    use_user_features = trial.suggest_categorical("use_user_features", [True, False])

    # Ensure at least one feature type is enabled if feature_weight > 0
    if params["feature_weight"] > 0.1 and not use_item_features and not use_user_features:
        use_item_features = True

    logger.info(f"Trial {trial.number}: factors={params['factors']}, "
                f"iter={params['iterations']}, reg={params['regularization']:.4f}")

    # Create and train model
    model = HybridRecommender(**params)

    try:
        model.fit(
            train_data,
            use_item_features=use_item_features,
            use_user_features=use_user_features,
        )

        # Evaluate on validation set (use fewer users for speed)
        results = evaluator.evaluate(model, valid_data, n_users=300, show_progress=False)
        ndcg = results["NDCG@10"]
        precision = results["Precision@10"]
        recall = results["Recall@10"]

        logger.info(f"Trial {trial.number}: NDCG@10={ndcg:.4f}, P@10={precision:.4f}")

        # Log to MLflow (nested run)
        with mlflow.start_run(run_name=f"trial-{trial.number}", nested=True):
            mlflow.log_params(params)
            mlflow.log_param("use_item_features", use_item_features)
            mlflow.log_param("use_user_features", use_user_features)
            mlflow.log_metrics({
                "NDCG_at_10": ndcg,
                "Precision_at_10": precision,
                "Recall_at_10": recall,
            })

        # Cleanup to free memory
        del model
        gc.collect()

        return ndcg

    except Exception as e:
        logger.error(f"Trial {trial.number} failed: {e}")
        return 0.0


def run_optimization(n_trials: int = 50) -> optuna.Study:
    """Run Optuna optimization study.

    Args:
        n_trials: Number of trials to run.

    Returns:
        Completed Optuna study.
    """
    # Setup MLflow
    setup_mlflow()

    with mlflow.start_run(run_name="optuna-optimization"):
        mlflow.set_tag("optimization_type", "optuna")
        mlflow.log_param("n_trials", n_trials)

        # Create study with pruning
        study = optuna.create_study(
            direction="maximize",
            study_name="hybrid-recommender-optimization",
            pruner=optuna.pruners.MedianPruner(
                n_startup_trials=5,
                n_warmup_steps=3,
            ),
        )

        # Run optimization
        logger.info(f"Starting optimization with {n_trials} trials...")
        study.optimize(
            objective,
            n_trials=n_trials,
            show_progress_bar=True,
            gc_after_trial=True,
        )

        # Log best results to parent run
        mlflow.log_params({f"best_{k}": v for k, v in study.best_params.items()})
        mlflow.log_metric("best_ndcg_at_10", study.best_value)

        return study


def save_best_params(study: optuna.Study, output_path: str = "configs/best_params.yaml"):
    """Save best parameters to YAML file.

    Args:
        study: Completed Optuna study.
        output_path: Path for output YAML file.
    """
    best_params = {
        "hybrid_recommender": {
            **study.best_params,
            "best_ndcg_at_10": float(study.best_value),
            "n_trials": len(study.trials),
        }
    }

    output_path = project_root / output_path
    with open(output_path, "w") as f:
        yaml.dump(best_params, f, default_flow_style=False)

    logger.info(f"Best params saved to {output_path}")


def print_top_trials(study: optuna.Study, n: int = 5):
    """Print top N trial configurations.

    Args:
        study: Completed Optuna study.
        n: Number of top trials to show.
    """
    print(f"\n{'='*80}")
    print(f"TOP {n} CONFIGURATIONS")
    print(f"{'='*80}")

    # Sort trials by value (NDCG), handle None values
    valid_trials = [t for t in study.trials if t.value is not None]
    trials = sorted(valid_trials, key=lambda t: t.value, reverse=True)[:n]

    for i, trial in enumerate(trials, 1):
        print(f"\n#{i} Trial {trial.number} - NDCG@10: {trial.value:.4f}")
        print("-" * 40)
        for key, value in trial.params.items():
            if isinstance(value, float):
                print(f"  {key}: {value:.6f}")
            else:
                print(f"  {key}: {value}")


def compare_with_default(study: optuna.Study):
    """Compare best configuration with default.

    Args:
        study: Completed Optuna study.
    """
    print(f"\n{'='*80}")
    print("COMPARISON: BEST vs DEFAULT")
    print(f"{'='*80}")

    # Default configuration
    default_params = {
        "factors": 64,
        "iterations": 15,
        "regularization": 0.01,
        "cf_weight": 0.7,
        "feature_weight": 0.3,
        "use_item_features": True,
        "use_user_features": False,
    }

    print("\nDefault configuration:")
    for key, value in default_params.items():
        print(f"  {key}: {value}")

    print("\nBest configuration:")
    for key, value in study.best_params.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.6f}")
        else:
            print(f"  {key}: {value}")

    # Train and evaluate both
    logger.info("\nTraining default model...")
    default_model = HybridRecommender(
        factors=default_params["factors"],
        iterations=default_params["iterations"],
        regularization=default_params["regularization"],
        cf_weight=default_params["cf_weight"],
        feature_weight=default_params["feature_weight"],
    )
    default_model.fit(
        train_data,
        use_item_features=default_params["use_item_features"],
        use_user_features=default_params["use_user_features"],
    )
    default_results = evaluator.evaluate(default_model, valid_data, n_users=300, show_progress=False)

    logger.info("Training best model...")
    best_model = HybridRecommender(
        factors=study.best_params["factors"],
        iterations=study.best_params["iterations"],
        regularization=study.best_params["regularization"],
        cf_weight=study.best_params["cf_weight"],
        feature_weight=study.best_params["feature_weight"],
    )
    best_model.fit(
        train_data,
        use_item_features=study.best_params["use_item_features"],
        use_user_features=study.best_params["use_user_features"],
    )
    best_results = evaluator.evaluate(best_model, valid_data, n_users=300, show_progress=False)

    print(f"\n{'Metric':<15} {'Default':>12} {'Best':>12} {'Improvement':>15}")
    print("-" * 55)

    for metric in ["NDCG@10", "Precision@10", "Recall@10"]:
        default_val = default_results[metric]
        best_val = best_results[metric]
        if default_val > 0:
            improvement = (best_val - default_val) / default_val * 100
        else:
            improvement = 0
        print(f"{metric:<15} {default_val:>12.4f} {best_val:>12.4f} {improvement:>+14.1f}%")


def main():
    """Main optimization workflow."""
    logger.info("=" * 70)
    logger.info("OPTUNA HYPERPARAMETER OPTIMIZATION")
    logger.info("=" * 70)

    # Load data
    load_data()

    # Run optimization (reduced trials for faster execution)
    n_trials = 20  # Can increase to 50 for more thorough search
    study = run_optimization(n_trials=n_trials)

    # Print results
    print_top_trials(study, n=5)

    # Save best params
    save_best_params(study)

    # Compare with default
    compare_with_default(study)

    # Summary
    print(f"\n{'='*80}")
    print("OPTIMIZATION COMPLETE")
    print(f"{'='*80}")
    print(f"\nBest NDCG@10: {study.best_value:.4f}")
    print(f"Total trials: {len(study.trials)}")
    print(f"\nBest params saved to: configs/best_params.yaml")
    print(f"Run 'mlflow ui' to view all trials")


if __name__ == "__main__":
    main()
