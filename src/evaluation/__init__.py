"""Metrics and evaluation."""

from src.evaluation.cold_start import (
    cold_item_analysis,
    compare_cold_start_models,
    evaluate_cold_items,
    evaluate_cold_start,
    get_user_interaction_stats,
    split_by_interaction_count,
)
from src.evaluation.evaluator import RecommenderEvaluator
from src.evaluation.metrics import (
    ap_at_k,
    coverage,
    hit_rate_at_k,
    map_at_k,
    mrr_at_k,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
)

__all__ = [
    # Metrics
    "precision_at_k",
    "recall_at_k",
    "ndcg_at_k",
    "map_at_k",
    "ap_at_k",
    "hit_rate_at_k",
    "mrr_at_k",
    "coverage",
    # Evaluator
    "RecommenderEvaluator",
    # Cold start
    "split_by_interaction_count",
    "get_user_interaction_stats",
    "evaluate_cold_start",
    "cold_item_analysis",
    "evaluate_cold_items",
    "compare_cold_start_models",
]
