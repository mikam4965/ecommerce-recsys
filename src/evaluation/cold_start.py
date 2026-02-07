"""Cold start analysis and evaluation module."""

from typing import Any

import numpy as np
import polars as pl
from loguru import logger
from tqdm import tqdm

from src.evaluation.metrics import (
    hit_rate_at_k,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
)
from src.models.base import BaseRecommender


def split_by_interaction_count(
    events: pl.DataFrame,
    cold_threshold: int = 5,
    warm_threshold: int = 20,
    user_col: str = "user_id",
) -> dict[str, list[int]]:
    """Разделение пользователей по количеству взаимодействий.

    Args:
        events: DataFrame с событиями.
        cold_threshold: Порог для cold users (< threshold).
        warm_threshold: Порог для hot users (> threshold).
        user_col: Название колонки с user_id.

    Returns:
        Словарь с группами пользователей:
        - "cold": < cold_threshold взаимодействий
        - "warm": cold_threshold <= x <= warm_threshold
        - "hot": > warm_threshold взаимодействий
    """
    user_counts = events.group_by(user_col).agg(
        pl.len().alias("n_interactions")
    )

    cold_users = user_counts.filter(
        pl.col("n_interactions") < cold_threshold
    )[user_col].to_list()

    warm_users = user_counts.filter(
        (pl.col("n_interactions") >= cold_threshold) &
        (pl.col("n_interactions") <= warm_threshold)
    )[user_col].to_list()

    hot_users = user_counts.filter(
        pl.col("n_interactions") > warm_threshold
    )[user_col].to_list()

    logger.info(f"User split: cold={len(cold_users)}, warm={len(warm_users)}, hot={len(hot_users)}")

    return {
        "cold": cold_users,
        "warm": warm_users,
        "hot": hot_users,
    }


def get_user_interaction_stats(
    events: pl.DataFrame,
    user_col: str = "user_id",
) -> pl.DataFrame:
    """Статистика взаимодействий по пользователям.

    Args:
        events: DataFrame с событиями.
        user_col: Название колонки с user_id.

    Returns:
        DataFrame со статистикой: user_id, n_interactions, n_items, n_categories.
    """
    stats = events.group_by(user_col).agg([
        pl.len().alias("n_interactions"),
        pl.col("item_id").n_unique().alias("n_unique_items"),
        pl.col("category_id").n_unique().alias("n_unique_categories"),
        pl.col("event_type").filter(pl.col("event_type") == "transaction").len().alias("n_transactions"),
    ])

    return stats


def evaluate_cold_start(
    model: BaseRecommender,
    test_data: pl.DataFrame,
    train_data: pl.DataFrame,
    k_values: list[int] | None = None,
    cold_threshold: int = 5,
    warm_threshold: int = 20,
    n_users_per_group: int | None = None,
    show_progress: bool = True,
) -> pl.DataFrame:
    """Оценка модели отдельно для cold/warm/hot групп пользователей.

    Args:
        model: Обученная модель рекомендаций.
        test_data: Тестовые данные.
        train_data: Обучающие данные (для разделения пользователей).
        k_values: Значения K для метрик.
        cold_threshold: Порог для cold users.
        warm_threshold: Порог для hot users.
        n_users_per_group: Максимум пользователей на группу (None = все).
        show_progress: Показывать прогресс.

    Returns:
        DataFrame с метриками для каждой группы.
    """
    k_values = k_values or [5, 10, 20]
    max_k = max(k_values)

    # Разделить пользователей по train_data
    user_groups = split_by_interaction_count(
        train_data,
        cold_threshold=cold_threshold,
        warm_threshold=warm_threshold,
    )

    # Получить ground truth из test_data
    user_relevant = (
        test_data.group_by("user_id")
        .agg(pl.col("item_id").alias("items"))
    )
    user_relevant_dict = {
        row["user_id"]: set(row["items"])
        for row in user_relevant.iter_rows(named=True)
    }

    results = []

    for group_name, user_ids in user_groups.items():
        # Фильтрация пользователей
        if hasattr(model, "user_to_idx"):
            valid_users = [u for u in user_ids if u in model.user_to_idx and u in user_relevant_dict]
        else:
            valid_users = [u for u in user_ids if u in user_relevant_dict]

        if n_users_per_group and len(valid_users) > n_users_per_group:
            np.random.seed(42)
            valid_users = list(np.random.choice(valid_users, size=n_users_per_group, replace=False))

        if len(valid_users) == 0:
            logger.warning(f"No valid users in {group_name} group")
            continue

        logger.info(f"Evaluating {group_name} group: {len(valid_users)} users")

        # Собрать метрики
        group_metrics = {k: {
            "precision": [],
            "recall": [],
            "ndcg": [],
            "hit_rate": [],
        } for k in k_values}

        iterator = tqdm(valid_users, desc=f"{group_name}", disable=not show_progress)
        for user_id in iterator:
            relevant = user_relevant_dict[user_id]

            try:
                recs = model.recommend(user_id, n=max_k, filter_already_liked=True)
            except Exception:
                continue

            for k in k_values:
                group_metrics[k]["precision"].append(precision_at_k(recs, relevant, k))
                group_metrics[k]["recall"].append(recall_at_k(recs, relevant, k))
                group_metrics[k]["ndcg"].append(ndcg_at_k(recs, relevant, k))
                group_metrics[k]["hit_rate"].append(hit_rate_at_k(recs, relevant, k))

        # Агрегировать результаты
        result = {
            "group": group_name,
            "n_users": len(valid_users),
        }

        for k in k_values:
            metrics = group_metrics[k]
            if metrics["precision"]:
                result[f"Precision@{k}"] = np.mean(metrics["precision"])
                result[f"Recall@{k}"] = np.mean(metrics["recall"])
                result[f"NDCG@{k}"] = np.mean(metrics["ndcg"])
                result[f"HitRate@{k}"] = np.mean(metrics["hit_rate"])

        results.append(result)

    return pl.DataFrame(results)


def cold_item_analysis(
    events: pl.DataFrame,
    cold_threshold: int = 10,
    popular_threshold: int = 100,
    item_col: str = "item_id",
) -> dict[str, list[int]]:
    """Анализ товаров по количеству взаимодействий.

    Args:
        events: DataFrame с событиями.
        cold_threshold: Порог для cold items (< threshold).
        popular_threshold: Порог для popular items (> threshold).
        item_col: Название колонки с item_id.

    Returns:
        Словарь с группами товаров:
        - "cold_items": < cold_threshold взаимодействий
        - "warm_items": cold_threshold <= x <= popular_threshold
        - "popular_items": > popular_threshold взаимодействий
    """
    item_counts = events.group_by(item_col).agg(
        pl.len().alias("n_interactions")
    )

    cold_items = item_counts.filter(
        pl.col("n_interactions") < cold_threshold
    )[item_col].to_list()

    warm_items = item_counts.filter(
        (pl.col("n_interactions") >= cold_threshold) &
        (pl.col("n_interactions") <= popular_threshold)
    )[item_col].to_list()

    popular_items = item_counts.filter(
        pl.col("n_interactions") > popular_threshold
    )[item_col].to_list()

    logger.info(f"Item split: cold={len(cold_items)}, warm={len(warm_items)}, popular={len(popular_items)}")

    return {
        "cold_items": cold_items,
        "warm_items": warm_items,
        "popular_items": popular_items,
    }


def evaluate_cold_items(
    model: BaseRecommender,
    test_data: pl.DataFrame,
    train_data: pl.DataFrame,
    cold_items: list[int],
    n_users: int = 1000,
    n_recs: int = 20,
) -> dict[str, Any]:
    """Оценка покрытия холодных товаров в рекомендациях.

    Args:
        model: Обученная модель рекомендаций.
        test_data: Тестовые данные.
        train_data: Обучающие данные.
        cold_items: Список холодных товаров.
        n_users: Количество пользователей для оценки.
        n_recs: Количество рекомендаций на пользователя.

    Returns:
        Словарь с метриками покрытия холодных товаров.
    """
    cold_items_set = set(cold_items)

    # Выбрать пользователей
    if hasattr(model, "user_to_idx"):
        valid_users = list(model.user_to_idx.keys())
    else:
        valid_users = train_data["user_id"].unique().to_list()

    if len(valid_users) > n_users:
        np.random.seed(42)
        valid_users = list(np.random.choice(valid_users, size=n_users, replace=False))

    # Собрать все рекомендации
    all_recs = []
    cold_recs_count = 0
    total_recs_count = 0

    for user_id in tqdm(valid_users, desc="Checking cold item coverage"):
        try:
            recs = model.recommend(user_id, n=n_recs, filter_already_liked=True)
            all_recs.extend(recs)
            total_recs_count += len(recs)
            cold_recs_count += sum(1 for r in recs if r in cold_items_set)
        except Exception:
            continue

    unique_recommended = set(all_recs)
    cold_recommended = unique_recommended & cold_items_set

    return {
        "total_cold_items": len(cold_items_set),
        "cold_items_recommended": len(cold_recommended),
        "cold_item_coverage": len(cold_recommended) / len(cold_items_set) if cold_items_set else 0,
        "cold_items_in_recs_ratio": cold_recs_count / total_recs_count if total_recs_count else 0,
        "unique_items_recommended": len(unique_recommended),
    }


def compare_cold_start_models(
    models: dict[str, BaseRecommender],
    test_data: pl.DataFrame,
    train_data: pl.DataFrame,
    k_values: list[int] | None = None,
    n_users_per_group: int = 500,
) -> pl.DataFrame:
    """Сравнение моделей по группам пользователей.

    Args:
        models: Словарь {название: модель}.
        test_data: Тестовые данные.
        train_data: Обучающие данные.
        k_values: Значения K для метрик.
        n_users_per_group: Максимум пользователей на группу.

    Returns:
        DataFrame со сравнением моделей по группам.
    """
    k_values = k_values or [10]

    all_results = []

    for model_name, model in models.items():
        logger.info(f"\nEvaluating {model_name}...")

        results = evaluate_cold_start(
            model,
            test_data,
            train_data,
            k_values=k_values,
            n_users_per_group=n_users_per_group,
        )

        # Добавить название модели
        results = results.with_columns(pl.lit(model_name).alias("model"))
        all_results.append(results)

    return pl.concat(all_results)
