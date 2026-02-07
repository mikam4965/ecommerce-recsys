"""MLflow tracking utilities for recommendation experiments."""

import functools
import time
from pathlib import Path
from typing import Any, Callable

import mlflow
import yaml
from loguru import logger


def setup_mlflow(config_path: str | Path = "configs/mlflow_config.yaml") -> str:
    """Initialize MLflow from config file.

    Args:
        config_path: Path to MLflow config YAML

    Returns:
        Experiment ID
    """
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"MLflow config not found: {config_path}")

    with open(config_path) as f:
        config = yaml.safe_load(f)

    mlflow_config = config.get("mlflow", {})

    # Set tracking URI
    tracking_uri = mlflow_config.get("tracking_uri", "./mlruns")
    mlflow.set_tracking_uri(tracking_uri)
    logger.info(f"MLflow tracking URI: {tracking_uri}")

    # Set/create experiment
    experiment_name = mlflow_config.get("experiment_name", "default")
    experiment = mlflow.set_experiment(experiment_name)
    logger.info(f"MLflow experiment: {experiment_name} (ID: {experiment.experiment_id})")

    return experiment.experiment_id


def track_experiment(
    experiment_name: str | None = None,
    run_name: str | None = None,
):
    """Decorator to track function execution as MLflow run.

    Args:
        experiment_name: Override experiment name
        run_name: Name for the run (defaults to function name)
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            # Set experiment if specified
            if experiment_name:
                mlflow.set_experiment(experiment_name)

            # Start run
            name = run_name or func.__name__
            with mlflow.start_run(run_name=name):
                start_time = time.time()

                # Log function name as tag
                mlflow.set_tag("function", func.__name__)

                try:
                    result = func(*args, **kwargs)

                    # Log training time
                    elapsed = time.time() - start_time
                    mlflow.log_metric("training_time_sec", elapsed)

                    return result

                except Exception as e:
                    mlflow.set_tag("error", str(e))
                    raise

        return wrapper
    return decorator


def log_model_params(model: Any, prefix: str = "") -> None:
    """Log model hyperparameters to MLflow.

    Args:
        model: Model instance with attributes to log
        prefix: Optional prefix for parameter names
    """
    params = {}

    # Common parameters for recommendation models
    param_names = [
        "factors", "iterations", "regularization",
        "cf_weight", "feature_weight",
        "n_candidates", "ranker_iterations",
        "learning_rate", "num_threads",
    ]

    for name in param_names:
        if hasattr(model, name):
            key = f"{prefix}{name}" if prefix else name
            params[key] = getattr(model, name)

    # Log model class name
    params[f"{prefix}model_class" if prefix else "model_class"] = model.__class__.__name__

    if params:
        mlflow.log_params(params)
        logger.debug(f"Logged {len(params)} parameters")


def log_metrics(metrics: dict[str, float], step: int | None = None) -> None:
    """Log metrics dictionary to MLflow.

    Args:
        metrics: Dictionary of metric_name -> value
        step: Optional step number for tracking over time
    """
    if not metrics:
        return

    # Sanitize metric names (MLflow doesn't allow "@" in names)
    sanitized = {
        k.replace("@", "_at_"): v
        for k, v in metrics.items()
        if isinstance(v, (int, float))
    }

    mlflow.log_metrics(sanitized, step=step)
    logger.debug(f"Logged {len(sanitized)} metrics")


def log_artifact(file_path: str | Path, artifact_path: str | None = None) -> None:
    """Log artifact file to MLflow.

    Args:
        file_path: Path to file to log
        artifact_path: Optional subdirectory in artifacts
    """
    file_path = Path(file_path)
    if not file_path.exists():
        logger.warning(f"Artifact not found: {file_path}")
        return

    mlflow.log_artifact(str(file_path), artifact_path)
    logger.debug(f"Logged artifact: {file_path}")


def log_confusion_matrix(y_true, y_pred, labels=None) -> None:
    """Log confusion matrix as artifact.

    Args:
        y_true: True labels
        y_pred: Predicted labels
        labels: Optional label names
    """
    try:
        import matplotlib.pyplot as plt
        from sklearn.metrics import ConfusionMatrixDisplay

        fig, ax = plt.subplots(figsize=(8, 6))
        ConfusionMatrixDisplay.from_predictions(
            y_true, y_pred, labels=labels, ax=ax
        )

        # Save and log
        path = Path("confusion_matrix.png")
        fig.savefig(path, bbox_inches="tight", dpi=100)
        plt.close(fig)

        mlflow.log_artifact(str(path))
        path.unlink()  # Remove temp file

    except ImportError:
        logger.warning("matplotlib or sklearn not available for confusion matrix")
