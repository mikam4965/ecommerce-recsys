"""Offline A/B testing module for recommendation system evaluation.

Simulates A/B testing by splitting test users into control and treatment
groups, then comparing recommendation quality metrics between models.
Includes statistical significance testing.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import polars as pl
from loguru import logger
from scipy import stats

from src.evaluation.metrics import (
    hit_rate_at_k,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
)
from src.models.base import BaseRecommender


@dataclass
class ABTestResult:
    """Results from an A/B test comparison.

    Attributes:
        control_name: Name of the control model.
        treatment_name: Name of the treatment model.
        n_control: Number of users in control group.
        n_treatment: Number of users in treatment group.
        control_metrics: Per-user metrics for control group.
        treatment_metrics: Per-user metrics for treatment group.
    """

    control_name: str
    treatment_name: str
    n_control: int
    n_treatment: int
    control_metrics: dict[str, list[float]] = field(default_factory=dict)
    treatment_metrics: dict[str, list[float]] = field(default_factory=dict)

    def summary(self) -> dict[str, dict]:
        """Compute summary statistics with statistical significance.

        Returns:
            Dictionary with metric -> {control_mean, treatment_mean,
            lift_pct, p_value, significant} mappings.
        """
        results = {}
        for metric in self.control_metrics:
            control = np.array(self.control_metrics[metric])
            treatment = np.array(self.treatment_metrics[metric])

            control_mean = float(np.mean(control))
            treatment_mean = float(np.mean(treatment))

            # Lift percentage
            lift = (
                (treatment_mean - control_mean) / control_mean * 100
                if control_mean > 0 else 0.0
            )

            # Welch's t-test (unequal variance)
            if len(control) > 1 and len(treatment) > 1:
                t_stat, p_value = stats.ttest_ind(
                    treatment, control, equal_var=False
                )
            else:
                t_stat, p_value = 0.0, 1.0

            results[metric] = {
                "control_mean": round(control_mean, 4),
                "treatment_mean": round(treatment_mean, 4),
                "lift_pct": round(lift, 2),
                "p_value": round(float(p_value), 4),
                "significant": float(p_value) < 0.05,
                "control_std": round(float(np.std(control)), 4),
                "treatment_std": round(float(np.std(treatment)), 4),
            }

        return results

    def conversion_comparison(self) -> dict:
        """Compare conversion-like metrics (Hit Rate as CTR proxy).

        Returns:
            Dictionary with CTR and conversion comparison.
        """
        result = {}

        # Hit Rate@10 as CTR proxy
        if "HitRate@10" in self.control_metrics:
            ctrl_hr = np.mean(self.control_metrics["HitRate@10"])
            treat_hr = np.mean(self.treatment_metrics["HitRate@10"])
            result["ctr_control"] = round(ctrl_hr * 100, 2)
            result["ctr_treatment"] = round(treat_hr * 100, 2)
            result["ctr_lift_pct"] = round(
                (treat_hr - ctrl_hr) / max(ctrl_hr, 1e-8) * 100, 2
            )

        # Precision@10 as conversion proxy
        if "Precision@10" in self.control_metrics:
            ctrl_p = np.mean(self.control_metrics["Precision@10"])
            treat_p = np.mean(self.treatment_metrics["Precision@10"])
            result["conversion_control"] = round(ctrl_p * 100, 2)
            result["conversion_treatment"] = round(treat_p * 100, 2)
            result["conversion_lift_pct"] = round(
                (treat_p - ctrl_p) / max(ctrl_p, 1e-8) * 100, 2
            )

        return result


class ABTestRunner:
    """Runs offline A/B tests between recommendation models.

    Splits users into control and treatment groups, generates
    recommendations from each model, and computes per-user metrics
    for statistical comparison.
    """

    def __init__(
        self,
        k_values: list[int] | None = None,
        random_seed: int = 42,
    ):
        """Initialize A/B test runner.

        Args:
            k_values: List of K values for metrics.
            random_seed: Random seed for user group assignment.
        """
        self.k_values = k_values or [5, 10, 20]
        self.random_seed = random_seed

    def run_test(
        self,
        control_model: BaseRecommender,
        treatment_model: BaseRecommender,
        test_data: pl.DataFrame,
        n_users: int = 1000,
        split_ratio: float = 0.5,
        control_name: str | None = None,
        treatment_name: str | None = None,
    ) -> ABTestResult:
        """Run A/B test between two models.

        Args:
            control_model: Baseline model (control group A).
            treatment_model: New model to test (treatment group B).
            test_data: Test interactions DataFrame.
            n_users: Number of test users to evaluate.
            split_ratio: Fraction of users assigned to control (default 50/50).
            control_name: Name for control model.
            treatment_name: Name for treatment model.

        Returns:
            ABTestResult with per-user metrics for both groups.
        """
        rng = np.random.default_rng(self.random_seed)

        ctrl_name = control_name or control_model.name
        treat_name = treatment_name or treatment_model.name

        logger.info(f"A/B Test: {ctrl_name} (control) vs {treat_name} (treatment)")

        # Get test users with relevant items
        user_relevant = self._get_user_relevant_items(test_data)
        available_users = list(user_relevant.keys())

        # Sample and split users
        n_users = min(n_users, len(available_users))
        selected_users = rng.choice(available_users, size=n_users, replace=False)
        rng.shuffle(selected_users)

        split_idx = int(n_users * split_ratio)
        control_users = selected_users[:split_idx]
        treatment_users = selected_users[split_idx:]

        logger.info(f"  Control group: {len(control_users)} users")
        logger.info(f"  Treatment group: {len(treatment_users)} users")

        # Evaluate control group
        logger.info(f"  Evaluating {ctrl_name} on control group...")
        control_metrics = self._evaluate_group(
            control_model, control_users, user_relevant
        )

        # Evaluate treatment group
        logger.info(f"  Evaluating {treat_name} on treatment group...")
        treatment_metrics = self._evaluate_group(
            treatment_model, treatment_users, user_relevant
        )

        result = ABTestResult(
            control_name=ctrl_name,
            treatment_name=treat_name,
            n_control=len(control_users),
            n_treatment=len(treatment_users),
            control_metrics=control_metrics,
            treatment_metrics=treatment_metrics,
        )

        return result

    def _evaluate_group(
        self,
        model: BaseRecommender,
        users: np.ndarray,
        user_relevant: dict[int, set[int]],
    ) -> dict[str, list[float]]:
        """Evaluate a model on a group of users.

        Returns per-user metric values for statistical testing.
        """
        metrics: dict[str, list[float]] = {}

        for k in self.k_values:
            metrics[f"Precision@{k}"] = []
            metrics[f"Recall@{k}"] = []
            metrics[f"NDCG@{k}"] = []
            metrics[f"HitRate@{k}"] = []

        for user_id in users:
            relevant = user_relevant.get(int(user_id), set())
            if not relevant:
                continue

            recs = model.recommend(int(user_id), n=max(self.k_values))

            for k in self.k_values:
                top_k = recs[:k]

                metrics[f"Precision@{k}"].append(precision_at_k(top_k, relevant, k))
                metrics[f"Recall@{k}"].append(recall_at_k(top_k, relevant, k))
                metrics[f"NDCG@{k}"].append(ndcg_at_k(top_k, relevant, k))
                metrics[f"HitRate@{k}"].append(hit_rate_at_k(top_k, relevant, k))

        return metrics

    def _get_user_relevant_items(
        self, test_data: pl.DataFrame
    ) -> dict[int, set[int]]:
        """Extract relevant items per user from test data."""
        user_items = test_data.group_by("user_id").agg(
            pl.col("item_id").alias("items")
        )
        return {
            row["user_id"]: set(row["items"])
            for row in user_items.iter_rows(named=True)
        }


def print_ab_results(result: ABTestResult) -> None:
    """Print formatted A/B test results.

    Args:
        result: ABTestResult to display.
    """
    summary = result.summary()
    conversion = result.conversion_comparison()

    print("\n" + "=" * 75)
    print("A/B TEST RESULTS")
    print("=" * 75)
    print(f"Control (A):   {result.control_name} ({result.n_control} users)")
    print(f"Treatment (B): {result.treatment_name} ({result.n_treatment} users)")

    print(f"\n{'Metric':<18} {'Control':>10} {'Treatment':>10} "
          f"{'Lift':>10} {'p-value':>10} {'Sig.':>6}")
    print("-" * 66)

    for metric, values in summary.items():
        sig_marker = "*" if values["significant"] else ""
        print(f"{metric:<18} {values['control_mean']:>10.4f} "
              f"{values['treatment_mean']:>10.4f} "
              f"{values['lift_pct']:>+9.1f}% "
              f"{values['p_value']:>10.4f} {sig_marker:>6}")

    # Conversion comparison (CTR proxy)
    if conversion:
        print(f"\n{'='*75}")
        print("CONVERSION METRICS (offline proxy)")
        print(f"{'='*75}")

        if "ctr_control" in conversion:
            print(f"  CTR (Hit Rate@10):")
            print(f"    Control:   {conversion['ctr_control']:.1f}%")
            print(f"    Treatment: {conversion['ctr_treatment']:.1f}%")
            print(f"    Lift:      {conversion['ctr_lift_pct']:+.1f}%")

        if "conversion_control" in conversion:
            print(f"  Conversion (Precision@10):")
            print(f"    Control:   {conversion['conversion_control']:.2f}%")
            print(f"    Treatment: {conversion['conversion_treatment']:.2f}%")
            print(f"    Lift:      {conversion['conversion_lift_pct']:+.1f}%")

    print(f"\n* p < 0.05 (statistically significant)")
    print("=" * 75)
