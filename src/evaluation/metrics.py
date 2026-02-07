"""Recommendation quality metrics."""

import numpy as np


def precision_at_k(recommended: list[int], relevant: set[int], k: int) -> float:
    """Calculate Precision@K.

    Precision = |recommended ∩ relevant| / k

    Args:
        recommended: List of recommended item IDs (ordered by relevance).
        relevant: Set of relevant (ground truth) item IDs.
        k: Number of top recommendations to consider.

    Returns:
        Precision score in [0, 1].
    """
    if k <= 0:
        return 0.0

    recommended_k = recommended[:k]
    if not recommended_k:
        return 0.0

    n_relevant = len(set(recommended_k) & relevant)
    return n_relevant / k


def recall_at_k(recommended: list[int], relevant: set[int], k: int) -> float:
    """Calculate Recall@K.

    Recall = |recommended ∩ relevant| / |relevant|

    Args:
        recommended: List of recommended item IDs (ordered by relevance).
        relevant: Set of relevant (ground truth) item IDs.
        k: Number of top recommendations to consider.

    Returns:
        Recall score in [0, 1].
    """
    if not relevant or k <= 0:
        return 0.0

    recommended_k = recommended[:k]
    if not recommended_k:
        return 0.0

    n_relevant = len(set(recommended_k) & relevant)
    return n_relevant / len(relevant)


def ndcg_at_k(recommended: list[int], relevant: set[int], k: int) -> float:
    """Calculate Normalized Discounted Cumulative Gain@K.

    NDCG = DCG / IDCG, where DCG = Σ rel_i / log2(i + 1)

    Args:
        recommended: List of recommended item IDs (ordered by relevance).
        relevant: Set of relevant (ground truth) item IDs.
        k: Number of top recommendations to consider.

    Returns:
        NDCG score in [0, 1].
    """
    if not relevant or k <= 0:
        return 0.0

    recommended_k = recommended[:k]
    if not recommended_k:
        return 0.0

    # DCG: sum of 1/log2(i+2) for relevant items at position i
    dcg = 0.0
    for i, item in enumerate(recommended_k):
        if item in relevant:
            dcg += 1.0 / np.log2(i + 2)  # i+2 because i starts at 0

    # IDCG: ideal DCG if all relevant items were at top
    n_relevant = min(len(relevant), k)
    idcg = sum(1.0 / np.log2(i + 2) for i in range(n_relevant))

    if idcg == 0:
        return 0.0

    return dcg / idcg


def ap_at_k(recommended: list[int], relevant: set[int], k: int) -> float:
    """Calculate Average Precision@K.

    AP = (1/|relevant|) * Σ Precision@i * rel(i)

    Args:
        recommended: List of recommended item IDs (ordered by relevance).
        relevant: Set of relevant (ground truth) item IDs.
        k: Number of top recommendations to consider.

    Returns:
        Average Precision score in [0, 1].
    """
    if not relevant or k <= 0:
        return 0.0

    recommended_k = recommended[:k]
    if not recommended_k:
        return 0.0

    score = 0.0
    n_hits = 0

    for i, item in enumerate(recommended_k):
        if item in relevant:
            n_hits += 1
            precision_at_i = n_hits / (i + 1)
            score += precision_at_i

    if n_hits == 0:
        return 0.0

    # Normalize by min(k, |relevant|) for MAP@K
    return score / min(k, len(relevant))


def map_at_k(recommended: list[int], relevant: set[int], k: int) -> float:
    """Calculate Mean Average Precision@K (single user version).

    For multiple users, average ap_at_k across users.

    Args:
        recommended: List of recommended item IDs.
        relevant: Set of relevant item IDs.
        k: Number of top recommendations.

    Returns:
        MAP score in [0, 1].
    """
    return ap_at_k(recommended, relevant, k)


def hit_rate_at_k(recommended: list[int], relevant: set[int], k: int) -> float:
    """Calculate Hit Rate@K.

    Hit Rate = 1 if at least one relevant item in top-k, else 0.

    Args:
        recommended: List of recommended item IDs.
        relevant: Set of relevant item IDs.
        k: Number of top recommendations.

    Returns:
        1.0 if hit, 0.0 otherwise.
    """
    if not relevant or k <= 0:
        return 0.0

    recommended_k = set(recommended[:k])
    return 1.0 if recommended_k & relevant else 0.0


def coverage(all_recommendations: list[list[int]], catalog_size: int) -> float:
    """Calculate catalog coverage.

    Coverage = |unique recommended items| / |catalog|

    Args:
        all_recommendations: List of recommendation lists for all users.
        catalog_size: Total number of items in catalog.

    Returns:
        Coverage score in [0, 1].
    """
    if catalog_size <= 0:
        return 0.0

    unique_items = set()
    for recs in all_recommendations:
        unique_items.update(recs)

    return len(unique_items) / catalog_size


def mrr_at_k(recommended: list[int], relevant: set[int], k: int) -> float:
    """Calculate Mean Reciprocal Rank@K.

    MRR = 1/rank of first relevant item (0 if none in top-k).

    Args:
        recommended: List of recommended item IDs.
        relevant: Set of relevant item IDs.
        k: Number of top recommendations.

    Returns:
        MRR score in [0, 1].
    """
    if not relevant or k <= 0:
        return 0.0

    for i, item in enumerate(recommended[:k]):
        if item in relevant:
            return 1.0 / (i + 1)

    return 0.0
