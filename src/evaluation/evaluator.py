"""Recommender evaluation module."""

import time
from typing import Any

import numpy as np
import polars as pl
from loguru import logger
from tqdm import tqdm

from src.evaluation.metrics import (
    ap_at_k,
    coverage,
    hit_rate_at_k,
    mrr_at_k,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
)
from src.models.base import BaseRecommender


class RecommenderEvaluator:
    """Evaluator for recommendation models."""

    def __init__(self, k_values: list[int] | None = None):
        """Initialize evaluator.

        Args:
            k_values: List of K values for metrics. Default [5, 10, 20].
        """
        self.k_values = k_values or [5, 10, 20]

    def _get_user_relevant_items(
        self,
        test_data: pl.DataFrame,
        user_col: str = "user_id",
        item_col: str = "item_id",
    ) -> dict[int, set[int]]:
        """Extract relevant items per user from test data.

        Args:
            test_data: Test DataFrame with user-item interactions.
            user_col: User column name.
            item_col: Item column name.

        Returns:
            Dictionary mapping user_id to set of relevant item_ids.
        """
        user_items = (
            test_data.group_by(user_col)
            .agg(pl.col(item_col).alias("items"))
        )

        return {
            row[user_col]: set(row["items"])
            for row in user_items.iter_rows(named=True)
        }

    def evaluate(
        self,
        model: BaseRecommender,
        test_data: pl.DataFrame,
        k_values: list[int] | None = None,
        user_col: str = "user_id",
        item_col: str = "item_id",
        n_users: int | None = None,
        show_progress: bool = True,
    ) -> dict[str, Any]:
        """Evaluate model on test data.

        Args:
            model: Fitted recommender model.
            test_data: Test DataFrame with ground truth interactions.
            k_values: List of K values. Uses instance default if None.
            user_col: User column name.
            item_col: Item column name.
            n_users: Number of users to evaluate (None = all).
            show_progress: Whether to show progress bar.

        Returns:
            Dictionary with metrics for each K value.
        """
        k_values = k_values or self.k_values
        max_k = max(k_values)

        logger.info(f"Evaluating {model.name} on {len(test_data):,} test interactions")

        # Get relevant items per user
        user_relevant = self._get_user_relevant_items(test_data, user_col, item_col)

        # Filter to users that exist in model
        if hasattr(model, "user_to_idx"):
            valid_users = [u for u in user_relevant if u in model.user_to_idx]
        else:
            valid_users = list(user_relevant.keys())

        if n_users is not None and n_users < len(valid_users):
            np.random.seed(42)
            valid_users = list(np.random.choice(valid_users, size=n_users, replace=False))

        logger.info(f"Evaluating on {len(valid_users):,} users")

        # Collect metrics per user
        user_metrics = {k: {
            "precision": [],
            "recall": [],
            "ndcg": [],
            "map": [],
            "hit_rate": [],
            "mrr": [],
        } for k in k_values}

        all_recommendations = []

        start_time = time.time()

        iterator = tqdm(valid_users, desc="Evaluating", disable=not show_progress)
        for user_id in iterator:
            relevant = user_relevant[user_id]

            # Get recommendations
            try:
                recs = model.recommend(user_id, n=max_k, filter_already_liked=True)
            except Exception:
                continue

            all_recommendations.append(recs)

            # Calculate metrics for each K
            for k in k_values:
                user_metrics[k]["precision"].append(precision_at_k(recs, relevant, k))
                user_metrics[k]["recall"].append(recall_at_k(recs, relevant, k))
                user_metrics[k]["ndcg"].append(ndcg_at_k(recs, relevant, k))
                user_metrics[k]["map"].append(ap_at_k(recs, relevant, k))
                user_metrics[k]["hit_rate"].append(hit_rate_at_k(recs, relevant, k))
                user_metrics[k]["mrr"].append(mrr_at_k(recs, relevant, k))

        elapsed = time.time() - start_time

        # Aggregate results
        results = {
            "model": model.name,
            "n_users": len(valid_users),
            "eval_time_sec": round(elapsed, 2),
        }

        # Get catalog size
        catalog_size = test_data[item_col].n_unique()

        for k in k_values:
            metrics = user_metrics[k]
            results[f"Precision@{k}"] = np.mean(metrics["precision"])
            results[f"Recall@{k}"] = np.mean(metrics["recall"])
            results[f"NDCG@{k}"] = np.mean(metrics["ndcg"])
            results[f"MAP@{k}"] = np.mean(metrics["map"])
            results[f"HitRate@{k}"] = np.mean(metrics["hit_rate"])
            results[f"MRR@{k}"] = np.mean(metrics["mrr"])

        # Coverage (using max_k recommendations)
        results[f"Coverage@{max_k}"] = coverage(all_recommendations, catalog_size)

        logger.info(f"Evaluation completed in {elapsed:.2f}s")

        return results

    def evaluate_with_ci(
        self,
        model: BaseRecommender,
        test_data: pl.DataFrame,
        k_values: list[int] | None = None,
        n_bootstrap: int = 100,
        confidence: float = 0.95,
        **kwargs,
    ) -> dict[str, Any]:
        """Evaluate model with bootstrap confidence intervals.

        Args:
            model: Fitted recommender model.
            test_data: Test DataFrame.
            k_values: List of K values.
            n_bootstrap: Number of bootstrap samples.
            confidence: Confidence level (e.g., 0.95 for 95% CI).
            **kwargs: Additional args for evaluate().

        Returns:
            Dictionary with metrics and confidence intervals.
        """
        k_values = k_values or self.k_values
        max_k = max(k_values)

        logger.info(f"Evaluating {model.name} with {n_bootstrap} bootstrap samples")

        # Get user-level metrics first
        user_relevant = self._get_user_relevant_items(test_data)

        if hasattr(model, "user_to_idx"):
            valid_users = [u for u in user_relevant if u in model.user_to_idx]
        else:
            valid_users = list(user_relevant.keys())

        # Collect per-user metrics
        user_scores = {k: {
            "precision": [],
            "recall": [],
            "ndcg": [],
            "hit_rate": [],
        } for k in k_values}

        for user_id in tqdm(valid_users, desc="Collecting scores"):
            relevant = user_relevant[user_id]
            try:
                recs = model.recommend(user_id, n=max_k, filter_already_liked=True)
            except Exception:
                continue

            for k in k_values:
                user_scores[k]["precision"].append(precision_at_k(recs, relevant, k))
                user_scores[k]["recall"].append(recall_at_k(recs, relevant, k))
                user_scores[k]["ndcg"].append(ndcg_at_k(recs, relevant, k))
                user_scores[k]["hit_rate"].append(hit_rate_at_k(recs, relevant, k))

        # Bootstrap
        results = {"model": model.name}
        alpha = 1 - confidence

        for k in k_values:
            for metric_name in ["precision", "recall", "ndcg", "hit_rate"]:
                scores = np.array(user_scores[k][metric_name])
                n = len(scores)

                # Bootstrap means
                bootstrap_means = []
                np.random.seed(42)
                for _ in range(n_bootstrap):
                    sample = np.random.choice(scores, size=n, replace=True)
                    bootstrap_means.append(np.mean(sample))

                mean_val = np.mean(scores)
                ci_low = np.percentile(bootstrap_means, alpha / 2 * 100)
                ci_high = np.percentile(bootstrap_means, (1 - alpha / 2) * 100)

                metric_key = f"{metric_name.capitalize()}@{k}"
                results[metric_key] = mean_val
                results[f"{metric_key}_CI"] = (ci_low, ci_high)

        return results

    def compare_models(
        self,
        models: dict[str, BaseRecommender],
        test_data: pl.DataFrame,
        k_values: list[int] | None = None,
        **kwargs,
    ) -> pl.DataFrame:
        """Compare multiple models and return results as DataFrame.

        Args:
            models: Dictionary mapping model name to fitted model.
            test_data: Test DataFrame.
            k_values: List of K values.
            **kwargs: Additional args for evaluate().

        Returns:
            Polars DataFrame with comparison results.
        """
        k_values = k_values or self.k_values

        logger.info(f"Comparing {len(models)} models")

        results = []
        for name, model in models.items():
            logger.info(f"Evaluating {name}...")
            model_results = self.evaluate(
                model, test_data, k_values=k_values, **kwargs
            )
            results.append(model_results)

        # Create DataFrame
        df = pl.DataFrame(results)

        return df

    def format_results(
        self,
        results: dict[str, Any],
        k: int = 10,
    ) -> str:
        """Format evaluation results as readable string.

        Args:
            results: Evaluation results dictionary.
            k: K value to display.

        Returns:
            Formatted string.
        """
        lines = [
            f"Model: {results['model']}",
            f"Users: {results.get('n_users', 'N/A'):,}",
            f"Time: {results.get('eval_time_sec', 'N/A')}s",
            "",
            f"Metrics @{k}:",
            f"  Precision: {results.get(f'Precision@{k}', 0):.4f}",
            f"  Recall:    {results.get(f'Recall@{k}', 0):.4f}",
            f"  NDCG:      {results.get(f'NDCG@{k}', 0):.4f}",
            f"  MAP:       {results.get(f'MAP@{k}', 0):.4f}",
            f"  Hit Rate:  {results.get(f'HitRate@{k}', 0):.4f}",
            f"  MRR:       {results.get(f'MRR@{k}', 0):.4f}",
        ]

        return "\n".join(lines)
