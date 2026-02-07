"""Hybrid recommender using implicit library with feature-based scoring."""

import pickle
import time
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
from implicit.als import AlternatingLeastSquares
from loguru import logger
from scipy.sparse import coo_matrix, csr_matrix

from src.models.base import BaseRecommender


class HybridRecommender(BaseRecommender):
    """Hybrid recommender combining ALS collaborative filtering with feature-based scoring.

    This model:
    1. Uses implicit's ALS for user-item collaborative filtering
    2. Adds feature-based scoring for items (category similarity)
    3. Combines both scores with configurable weights
    4. Supports cold start recommendations based on user features
    """

    # Event type weights for interactions
    DEFAULT_EVENT_WEIGHTS = {
        "view": 1.0,
        "addtocart": 2.0,
        "transaction": 3.0,
    }

    def __init__(
        self,
        factors: int = 64,
        regularization: float = 0.01,
        iterations: int = 15,
        random_state: int = 42,
        event_weights: dict[str, float] | None = None,
        cf_weight: float = 0.7,
        feature_weight: float = 0.3,
    ):
        """Initialize Hybrid recommender.

        Args:
            factors: Number of latent factors.
            regularization: Regularization parameter.
            iterations: Number of ALS iterations.
            random_state: Random seed for reproducibility.
            event_weights: Custom weights for event types.
            cf_weight: Weight for collaborative filtering scores.
            feature_weight: Weight for feature-based scores.
        """
        super().__init__(name="HybridRecommender")
        self.factors = factors
        self.regularization = regularization
        self.iterations = iterations
        self.random_state = random_state
        self.event_weights = event_weights or self.DEFAULT_EVENT_WEIGHTS
        self.cf_weight = cf_weight
        self.feature_weight = feature_weight

        # Model
        self.model: AlternatingLeastSquares | None = None

        # Mappings
        self.user_to_idx: dict[int, int] = {}
        self.idx_to_user: dict[int, int] = {}
        self.item_to_idx: dict[int, int] = {}
        self.idx_to_item: dict[int, int] = {}

        # Category mappings and features
        self.category_to_idx: dict[str, int] = {}
        self.segment_to_idx: dict[str, int] = {}
        self.item_categories: dict[int, str] = {}  # item_id -> category_id
        self.category_popularity: dict[str, float] = {}  # category -> popularity score

        # User features
        self.user_segments: dict[int, str] = {}  # user_id -> segment
        self.user_favorite_category: dict[int, str] = {}  # user_id -> favorite category
        self.user_view_count: dict[int, int] = {}  # user_id -> view count
        self.user_unique_items: dict[int, int] = {}  # user_id -> unique items viewed
        self.user_recency_days: dict[int, float] = {}  # user_id -> days since last view

        # Segment-category affinity matrix
        self.segment_category_affinity: dict[str, dict[str, float]] = {}

        # Matrices
        self.interactions: csr_matrix | None = None

        # Feature flags
        self.use_item_features = False
        self.use_user_features = False

    def _prepare_interactions(
        self,
        events: pl.DataFrame,
    ) -> csr_matrix:
        """Prepare weighted interaction matrix.

        Args:
            events: Events DataFrame with user_id, item_id, event_type.

        Returns:
            CSR sparse matrix (n_users x n_items) with weighted interactions.
        """
        logger.info("Preparing weighted interaction matrix...")

        # Create user and item mappings
        unique_users = events["user_id"].unique().sort().to_list()
        unique_items = events["item_id"].unique().sort().to_list()

        self.user_to_idx = {user: idx for idx, user in enumerate(unique_users)}
        self.idx_to_user = {idx: user for user, idx in self.user_to_idx.items()}
        self.item_to_idx = {item: idx for idx, item in enumerate(unique_items)}
        self.idx_to_item = {idx: item for item, idx in self.item_to_idx.items()}

        n_users = len(unique_users)
        n_items = len(unique_items)

        logger.info(f"Users: {n_users:,}, Items: {n_items:,}")

        # Calculate weighted interactions
        weighted_events = events.with_columns(
            pl.col("event_type")
            .replace(self.event_weights)
            .cast(pl.Float64)
            .alias("weight")
        )

        # Aggregate by user-item pair (sum of weights)
        interaction_weights = (
            weighted_events.group_by(["user_id", "item_id"])
            .agg(pl.col("weight").sum().alias("weight"))
        )

        # Build COO matrix
        row_indices = [
            self.user_to_idx[u] for u in interaction_weights["user_id"].to_list()
        ]
        col_indices = [
            self.item_to_idx[i] for i in interaction_weights["item_id"].to_list()
        ]
        weights = interaction_weights["weight"].to_list()

        matrix = coo_matrix(
            (weights, (row_indices, col_indices)),
            shape=(n_users, n_items),
            dtype=np.float32,
        )

        logger.info(f"Interactions matrix: {matrix.shape}, nnz={matrix.nnz:,}")

        return matrix.tocsr()

    def _prepare_item_features(
        self,
        events: pl.DataFrame,
        top_n_categories: int = 50,
    ) -> None:
        """Prepare item feature data.

        Args:
            events: Events DataFrame.
            top_n_categories: Number of top categories to track.
        """
        logger.info("Preparing item features...")

        # Get top categories
        category_counts = (
            events.filter(pl.col("category_id") != "unknown")
            .group_by("category_id")
            .agg(pl.len().alias("count"))
            .sort("count", descending=True)
            .head(top_n_categories)
        )

        top_categories = category_counts["category_id"].to_list()
        self.category_to_idx = {cat: idx for idx, cat in enumerate(top_categories)}

        # Calculate category popularity (normalized)
        total_count = category_counts["count"].sum()
        for row in category_counts.iter_rows(named=True):
            self.category_popularity[row["category_id"]] = row["count"] / total_count

        logger.info(f"Tracking {len(top_categories)} categories")

        # Map items to categories
        item_category = (
            events.filter(pl.col("category_id").is_in(top_categories))
            .group_by("item_id")
            .agg(pl.col("category_id").first().alias("category_id"))
        )

        for row in item_category.iter_rows(named=True):
            if row["item_id"] in self.item_to_idx:
                self.item_categories[row["item_id"]] = row["category_id"]

        logger.info(f"Mapped {len(self.item_categories)} items to categories")

    def _prepare_user_features(
        self,
        events: pl.DataFrame,
        rfm_data: pl.DataFrame | None = None,
    ) -> None:
        """Prepare user features from ALL events (views).

        Calculates view-based features for all users:
        - view_count: number of view events
        - unique_items: number of unique items viewed
        - recency_days: days since last activity
        - favorite_category: most viewed category

        Args:
            events: Events DataFrame.
            rfm_data: Optional RFM segmentation DataFrame.
        """
        logger.info("Preparing user features from ALL events...")

        # 1. Store RFM segments if available
        if rfm_data is not None and "segment" in rfm_data.columns:
            segments = rfm_data["segment"].unique().to_list()
            self.segment_to_idx = {seg: idx for idx, seg in enumerate(segments)}

            for row in rfm_data.iter_rows(named=True):
                if row["user_id"] in self.user_to_idx:
                    self.user_segments[row["user_id"]] = row["segment"]

            logger.info(f"Loaded {len(self.user_segments)} user segments")

            # Calculate segment-category affinity
            self._calculate_segment_category_affinity(events, rfm_data)

        # 2. Calculate view count for all users
        view_counts = (
            events.filter(pl.col("event_type") == "view")
            .group_by("user_id")
            .agg(pl.len().alias("view_count"))
        )
        for row in view_counts.iter_rows(named=True):
            if row["user_id"] in self.user_to_idx:
                self.user_view_count[row["user_id"]] = row["view_count"]
        logger.info(f"View counts: {len(self.user_view_count)} users")

        # 3. Calculate unique items viewed
        unique_items = (
            events.group_by("user_id")
            .agg(pl.col("item_id").n_unique().alias("unique_items"))
        )
        for row in unique_items.iter_rows(named=True):
            if row["user_id"] in self.user_to_idx:
                self.user_unique_items[row["user_id"]] = row["unique_items"]
        logger.info(f"Unique items: {len(self.user_unique_items)} users")

        # 4. Calculate recency (days since last activity)
        max_timestamp = events["timestamp"].max()
        last_activity = (
            events.group_by("user_id")
            .agg(pl.col("timestamp").max().alias("last_ts"))
        )
        for row in last_activity.iter_rows(named=True):
            if row["user_id"] in self.user_to_idx:
                # timestamp is in milliseconds
                days = (max_timestamp - row["last_ts"]) / (24 * 60 * 60 * 1000)
                self.user_recency_days[row["user_id"]] = days
        logger.info(f"Recency: {len(self.user_recency_days)} users")

        # 5. Calculate favorite category based on add-to-cart events (stronger signal)
        # Fallback to views if user has no cart events
        cart_events = events.filter(
            (pl.col("event_type") == "addtocart") &
            pl.col("category_id").is_in(list(self.category_to_idx.keys()))
        )

        # First, get favorite category from cart events
        cart_category = (
            cart_events
            .group_by(["user_id", "category_id"])
            .agg(pl.len().alias("count"))
            .sort(["user_id", "count"], descending=[False, True])
            .group_by("user_id")
            .agg(pl.col("category_id").first().alias("fav_category"))
        )

        cart_users = set()
        for row in cart_category.iter_rows(named=True):
            if row["user_id"] in self.user_to_idx:
                self.user_favorite_category[row["user_id"]] = row["fav_category"]
                cart_users.add(row["user_id"])

        logger.info(f"Favorite categories from cart: {len(cart_users)} users")

        # Fallback: for users without cart events, use view events
        view_category = (
            events.filter(
                (pl.col("event_type") == "view") &
                pl.col("category_id").is_in(list(self.category_to_idx.keys()))
            )
            .group_by(["user_id", "category_id"])
            .agg(pl.len().alias("count"))
            .sort(["user_id", "count"], descending=[False, True])
            .group_by("user_id")
            .agg(pl.col("category_id").first().alias("fav_category"))
        )

        view_fallback_count = 0
        for row in view_category.iter_rows(named=True):
            if row["user_id"] in self.user_to_idx and row["user_id"] not in cart_users:
                self.user_favorite_category[row["user_id"]] = row["fav_category"]
                view_fallback_count += 1

        logger.info(f"Favorite categories from views (fallback): {view_fallback_count} users")
        logger.info(f"Total favorite categories: {len(self.user_favorite_category)} users")

    def _calculate_segment_category_affinity(
        self,
        events: pl.DataFrame,
        rfm_data: pl.DataFrame,
    ) -> None:
        """Calculate affinity between segments and categories.

        Args:
            events: Events DataFrame.
            rfm_data: RFM segmentation DataFrame.
        """
        # Join events with user segments
        events_with_segment = events.join(
            rfm_data.select(["user_id", "segment"]),
            on="user_id",
            how="inner",
        )

        # Calculate purchases per segment-category
        segment_category_counts = (
            events_with_segment.filter(
                (pl.col("event_type") == "transaction")
                & (pl.col("category_id").is_in(list(self.category_to_idx.keys())))
            )
            .group_by(["segment", "category_id"])
            .agg(pl.len().alias("count"))
        )

        # Normalize by segment
        segment_totals = (
            segment_category_counts.group_by("segment")
            .agg(pl.col("count").sum().alias("total"))
        )

        segment_category_norm = segment_category_counts.join(
            segment_totals, on="segment", how="left"
        ).with_columns(
            (pl.col("count") / pl.col("total")).alias("affinity")
        )

        # Build affinity dictionary
        for row in segment_category_norm.iter_rows(named=True):
            segment = row["segment"]
            category = row["category_id"]
            affinity = row["affinity"]

            if segment not in self.segment_category_affinity:
                self.segment_category_affinity[segment] = {}
            self.segment_category_affinity[segment][category] = affinity

        logger.info(f"Calculated affinity for {len(self.segment_category_affinity)} segments")

    def fit(
        self,
        events: pl.DataFrame,
        rfm_data: pl.DataFrame | None = None,
        use_item_features: bool = True,
        use_user_features: bool = True,
        **kwargs,
    ) -> "HybridRecommender":
        """Fit hybrid model on interaction data.

        Args:
            events: Events DataFrame with user_id, item_id, event_type, category_id.
            rfm_data: Optional RFM segmentation DataFrame for segment features.
            use_item_features: Whether to use item features.
            use_user_features: Whether to use user features (view-based features
                are calculated for all users; RFM segments are optional).

        Returns:
            Self for method chaining.
        """
        logger.info(f"Fitting {self.name}...")
        logger.info(
            f"Parameters: factors={self.factors}, reg={self.regularization}, "
            f"iterations={self.iterations}"
        )
        logger.info(f"Event weights: {self.event_weights}")
        logger.info(f"Score weights: CF={self.cf_weight}, Features={self.feature_weight}")

        start_time = time.time()

        # Prepare interactions
        self.interactions = self._prepare_interactions(events)

        # Prepare features
        self.use_item_features = use_item_features
        self.use_user_features = use_user_features  # Now works without RFM data

        if self.use_item_features:
            self._prepare_item_features(events)

        if self.use_user_features:
            self._prepare_user_features(events, rfm_data)

        # Initialize and train ALS model
        logger.info(f"Training ALS for {self.iterations} iterations...")

        self.model = AlternatingLeastSquares(
            factors=self.factors,
            regularization=self.regularization,
            iterations=self.iterations,
            random_state=self.random_state,
        )

        self.model.fit(self.interactions)

        elapsed = time.time() - start_time
        self._is_fitted = True

        mode = "CF only"
        if self.use_item_features and self.use_user_features:
            mode = "full hybrid"
        elif self.use_item_features:
            mode = "CF + item features"

        logger.info(f"Fitted {self.name} ({mode}) in {elapsed:.2f}s")

        return self

    def _get_feature_scores(
        self,
        user_id: int,
        item_indices: np.ndarray,
    ) -> np.ndarray:
        """Calculate feature-based scores using view-based features.

        Uses activity, diversity, and recency weights calculated from user's
        browsing history to adjust feature scores.

        Args:
            user_id: User identifier.
            item_indices: Array of item indices.

        Returns:
            Array of feature scores.
        """
        scores = np.zeros(len(item_indices), dtype=np.float32)

        # If user features are disabled, we can't compute meaningful scores
        # Return zeros to let CF scores dominate
        if not self.use_user_features:
            return scores

        user_segment = self.user_segments.get(user_id)
        user_fav_cat = self.user_favorite_category.get(user_id)

        # View-based weights
        user_views = self.user_view_count.get(user_id, 0)
        user_unique = self.user_unique_items.get(user_id, 0)
        user_recency = self.user_recency_days.get(user_id, 365)

        # Normalized weights (saturating functions)
        activity_weight = min(1.0, user_views / 50.0)  # saturates at 50 views
        diversity_weight = min(1.0, user_unique / 20.0)  # saturates at 20 unique items
        # Softer recency decay: 90 days half-life instead of 30
        recency_weight = max(0.3, 1.0 - user_recency / 90.0)

        # Check if user has any meaningful features
        has_features = user_fav_cat is not None or user_views > 0

        for i, item_idx in enumerate(item_indices):
            item_id = self.idx_to_item[item_idx]
            item_cat = self.item_categories.get(item_id)

            if item_cat is None:
                continue

            # Strong boost for favorite category match
            if user_fav_cat and item_cat == user_fav_cat:
                scores[i] += 1.0  # Base category match score

            # Boost based on segment-category affinity
            if user_segment and user_segment in self.segment_category_affinity:
                affinity = self.segment_category_affinity[user_segment].get(item_cat, 0)
                scores[i] += affinity * 0.3

            # Small boost for popular categories (exploration)
            popularity = self.category_popularity.get(item_cat, 0)
            scores[i] += popularity * 0.1

        # Apply activity and recency as global multipliers only if user has features
        if has_features:
            # Combine activity and recency into engagement score
            engagement = (0.5 + 0.5 * activity_weight) * recency_weight
            scores *= engagement

        return scores

    def recommend(
        self,
        user_id: int,
        n: int = 10,
        filter_already_liked: bool = True,
    ) -> list[int]:
        """Generate recommendations for a known user.

        Args:
            user_id: User identifier.
            n: Number of recommendations.
            filter_already_liked: Whether to filter already interacted items.

        Returns:
            List of recommended item IDs.
        """
        self._check_is_fitted()

        if user_id not in self.user_to_idx:
            logger.warning(f"User {user_id} not in training data")
            return self.recommend_cold_start({}, n)

        user_idx = self.user_to_idx[user_id]
        n_items = len(self.item_to_idx)

        # Get CF scores from ALS
        item_indices, cf_scores = self.model.recommend(
            user_idx,
            self.interactions[user_idx],
            N=n_items,
            filter_already_liked_items=filter_already_liked,
        )

        # Combine with feature scores if user features are enabled
        # (item features alone don't provide personalization without user context)
        if self.use_user_features:
            feature_scores = self._get_feature_scores(user_id, item_indices)

            # Only apply feature scoring if we have non-zero scores
            if feature_scores.max() > 0:
                # Normalize scores
                cf_norm = (cf_scores - cf_scores.min()) / (cf_scores.max() - cf_scores.min() + 1e-10)
                feat_norm = (feature_scores - feature_scores.min()) / (feature_scores.max() - feature_scores.min() + 1e-10)

                # Combine scores
                combined_scores = self.cf_weight * cf_norm + self.feature_weight * feat_norm

                # Re-rank by combined scores
                top_indices = np.argsort(-combined_scores)[:n]
                item_indices = item_indices[top_indices]
            else:
                # No feature signal for this user, use pure CF
                item_indices = item_indices[:n]
        else:
            item_indices = item_indices[:n]

        # Convert to item IDs
        recommendations = [self.idx_to_item[idx] for idx in item_indices]

        return recommendations

    def recommend_with_scores(
        self,
        user_id: int,
        n: int = 100,
        filter_already_liked: bool = True,
    ) -> list[tuple[int, float]]:
        """Generate recommendations with scores for ranking.

        Args:
            user_id: User identifier.
            n: Number of recommendations.
            filter_already_liked: Whether to filter already interacted items.

        Returns:
            List of (item_id, score) tuples ordered by score descending.
        """
        self._check_is_fitted()

        if user_id not in self.user_to_idx:
            return []

        user_idx = self.user_to_idx[user_id]
        n_items = len(self.item_to_idx)

        # Get CF scores from ALS
        item_indices, cf_scores = self.model.recommend(
            user_idx,
            self.interactions[user_idx],
            N=n_items,
            filter_already_liked_items=filter_already_liked,
        )

        # Combine with feature scores if user features are enabled
        if self.use_user_features:
            feature_scores = self._get_feature_scores(user_id, item_indices)

            # Only apply feature scoring if we have non-zero scores
            if feature_scores.max() > 0:
                # Normalize scores
                cf_norm = (cf_scores - cf_scores.min()) / (cf_scores.max() - cf_scores.min() + 1e-10)
                feat_norm = (feature_scores - feature_scores.min()) / (feature_scores.max() - feature_scores.min() + 1e-10)

                # Combine scores
                combined_scores = self.cf_weight * cf_norm + self.feature_weight * feat_norm
            else:
                combined_scores = cf_scores
        else:
            combined_scores = cf_scores

        # Get top N indices
        top_indices = np.argsort(-combined_scores)[:n]

        # Return (item_id, score) tuples
        return [
            (self.idx_to_item[item_indices[i]], float(combined_scores[i]))
            for i in top_indices
        ]

    def recommend_cold_start(
        self,
        user_features_dict: dict[str, Any],
        n: int = 10,
    ) -> list[int]:
        """Generate recommendations for a new user based on features.

        Args:
            user_features_dict: Dictionary with user features:
                - 'segment': RFM segment name
                - 'favorite_category': Category ID
            n: Number of recommendations.

        Returns:
            List of recommended item IDs.
        """
        self._check_is_fitted()

        # Get item popularity as base scores
        item_popularity = np.array(self.interactions.sum(axis=0)).flatten()

        # Boost based on user features
        segment = user_features_dict.get("segment")
        fav_category = user_features_dict.get("favorite_category")

        scores = item_popularity.copy()

        for item_id, item_idx in self.item_to_idx.items():
            item_cat = self.item_categories.get(item_id)

            if item_cat is None:
                continue

            # Boost for favorite category
            if fav_category and item_cat == fav_category:
                scores[item_idx] *= 2.0

            # Boost based on segment affinity
            if segment and segment in self.segment_category_affinity:
                affinity = self.segment_category_affinity[segment].get(item_cat, 0)
                scores[item_idx] *= (1 + affinity)

        # Get top N items
        top_indices = np.argsort(-scores)[:n]
        recommendations = [self.idx_to_item[idx] for idx in top_indices]

        return recommendations

    def similar_items(
        self,
        item_id: int,
        n: int = 10,
    ) -> list[int]:
        """Find similar items based on item embeddings.

        Args:
            item_id: Item identifier.
            n: Number of similar items to return.

        Returns:
            List of similar item IDs.
        """
        self._check_is_fitted()

        if item_id not in self.item_to_idx:
            logger.warning(f"Item {item_id} not in training data")
            return []

        item_idx = self.item_to_idx[item_id]

        # Get similar items from ALS model
        similar_indices, scores = self.model.similar_items(
            item_idx,
            N=n + 1,  # +1 because the item itself might be included
        )

        # Filter out the item itself
        similar = [
            self.idx_to_item[idx]
            for idx in similar_indices
            if idx != item_idx
        ][:n]

        return similar

    def get_user_embedding(self, user_id: int) -> np.ndarray | None:
        """Get user embedding vector.

        Args:
            user_id: User identifier.

        Returns:
            User embedding vector or None if not found.
        """
        self._check_is_fitted()

        if user_id not in self.user_to_idx:
            return None

        user_idx = self.user_to_idx[user_id]
        return self.model.user_factors[user_idx]

    def get_item_embedding(self, item_id: int) -> np.ndarray | None:
        """Get item embedding vector.

        Args:
            item_id: Item identifier.

        Returns:
            Item embedding vector or None if not found.
        """
        self._check_is_fitted()

        if item_id not in self.item_to_idx:
            return None

        item_idx = self.item_to_idx[item_id]
        return self.model.item_factors[item_idx]

    def recommend_for_new_user(
        self,
        session_items: list[int],
        n: int = 10,
    ) -> list[int]:
        """Рекомендации для нового пользователя на основе текущей сессии.

        Использует item embeddings для поиска похожих товаров.

        Args:
            session_items: Список item_id из текущей сессии (просмотры/корзина).
            n: Количество рекомендаций.

        Returns:
            Список рекомендуемых item_id.
        """
        self._check_is_fitted()

        # Фильтрация известных товаров
        known_items = [i for i in session_items if i in self.item_to_idx]

        if not known_items:
            logger.warning("No known items in session, falling back to cold start")
            return self.recommend_cold_start({}, n)

        # Получение embeddings товаров сессии
        embeddings = []
        for item_id in known_items:
            emb = self.get_item_embedding(item_id)
            if emb is not None:
                embeddings.append(emb)

        if not embeddings:
            return self.recommend_cold_start({}, n)

        # Средний embedding сессии
        session_embedding = np.mean(embeddings, axis=0)

        # Получение всех item embeddings
        all_embeddings = self.model.item_factors  # (n_items, factors)

        # Косинусное сходство
        norms = np.linalg.norm(all_embeddings, axis=1)
        session_norm = np.linalg.norm(session_embedding)
        similarities = np.dot(all_embeddings, session_embedding) / (norms * session_norm + 1e-10)

        # Исключить товары из сессии
        for item_id in known_items:
            similarities[self.item_to_idx[item_id]] = -np.inf

        # Топ-N
        top_indices = np.argsort(-similarities)[:n]
        return [self.idx_to_item[idx] for idx in top_indices]

    def recommend_new_item(
        self,
        item_features: dict,
        n: int = 10,
    ) -> list[int]:
        """Найти пользователей для нового товара.

        Args:
            item_features: Характеристики товара {"category_id": str}.
            n: Количество пользователей.

        Returns:
            Список user_id, которым подходит товар.
        """
        self._check_is_fitted()

        category_id = item_features.get("category_id")

        if not category_id:
            logger.warning("No category_id provided")
            return []

        # Найти товары той же категории
        similar_items = [
            item_id for item_id, cat in self.item_categories.items()
            if cat == category_id
        ][:50]

        if not similar_items:
            logger.warning(f"No items found in category {category_id}")
            return []

        # Найти пользователей, взаимодействовавших с этими товарами
        user_scores: dict[int, float] = {}

        for item_id in similar_items:
            if item_id in self.item_to_idx:
                item_idx = self.item_to_idx[item_id]
                # Получить пользователей из interaction matrix
                user_indices = self.interactions[:, item_idx].nonzero()[0]
                for user_idx in user_indices:
                    user_id = self.idx_to_user[user_idx]
                    user_scores[user_id] = user_scores.get(user_id, 0) + 1

        if not user_scores:
            return []

        # Усилить оценку через segment-category affinity
        for user_id in list(user_scores.keys()):
            score = user_scores[user_id]
            segment = self.user_segments.get(user_id)

            if segment and segment in self.segment_category_affinity:
                affinity = self.segment_category_affinity[segment].get(category_id, 0)
                user_scores[user_id] = score * (1 + affinity)

        # Топ-N пользователей
        sorted_users = sorted(user_scores.items(), key=lambda x: -x[1])
        return [u[0] for u in sorted_users[:n]]

    def save(self, path: str | Path) -> None:
        """Save model to disk.

        Args:
            path: Path to save the model.
        """
        self._check_is_fitted()
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        data = {
            "model": self.model,
            "interactions": self.interactions,
            "user_to_idx": self.user_to_idx,
            "idx_to_user": self.idx_to_user,
            "item_to_idx": self.item_to_idx,
            "idx_to_item": self.idx_to_item,
            "category_to_idx": self.category_to_idx,
            "segment_to_idx": self.segment_to_idx,
            "item_categories": self.item_categories,
            "category_popularity": self.category_popularity,
            "user_segments": self.user_segments,
            "user_favorite_category": self.user_favorite_category,
            "user_view_count": self.user_view_count,
            "user_unique_items": self.user_unique_items,
            "user_recency_days": self.user_recency_days,
            "segment_category_affinity": self.segment_category_affinity,
            "use_item_features": self.use_item_features,
            "use_user_features": self.use_user_features,
            "factors": self.factors,
            "regularization": self.regularization,
            "iterations": self.iterations,
            "event_weights": self.event_weights,
            "cf_weight": self.cf_weight,
            "feature_weight": self.feature_weight,
        }

        with open(path, "wb") as f:
            pickle.dump(data, f)

        logger.info(f"Saved {self.name} to {path}")

    def load(self, path: str | Path) -> "HybridRecommender":
        """Load model from disk.

        Args:
            path: Path to load the model from.

        Returns:
            Self for method chaining.
        """
        path = Path(path)

        with open(path, "rb") as f:
            data = pickle.load(f)

        self.model = data["model"]
        self.interactions = data["interactions"]
        self.user_to_idx = data["user_to_idx"]
        self.idx_to_user = data["idx_to_user"]
        self.item_to_idx = data["item_to_idx"]
        self.idx_to_item = data["idx_to_item"]
        self.category_to_idx = data["category_to_idx"]
        self.segment_to_idx = data["segment_to_idx"]
        self.item_categories = data["item_categories"]
        self.category_popularity = data["category_popularity"]
        self.user_segments = data["user_segments"]
        self.user_favorite_category = data["user_favorite_category"]
        self.user_view_count = data.get("user_view_count", {})
        self.user_unique_items = data.get("user_unique_items", {})
        self.user_recency_days = data.get("user_recency_days", {})
        self.segment_category_affinity = data["segment_category_affinity"]
        self.use_item_features = data["use_item_features"]
        self.use_user_features = data["use_user_features"]
        self.factors = data["factors"]
        self.regularization = data["regularization"]
        self.iterations = data["iterations"]
        self.event_weights = data["event_weights"]
        self.cf_weight = data["cf_weight"]
        self.feature_weight = data["feature_weight"]
        self._is_fitted = True

        logger.info(f"Loaded {self.name} from {path}")

        return self

    def __repr__(self) -> str:
        mode = "CF only"
        if self.use_item_features and self.use_user_features:
            mode = "full hybrid"
        elif self.use_item_features:
            mode = "CF + item features"

        return (
            f"HybridRecommender(factors={self.factors}, mode={mode}, "
            f"cf_weight={self.cf_weight}, feature_weight={self.feature_weight}, "
            f"fitted={self._is_fitted})"
        )


class TrueLightFMRecommender(BaseRecommender):
    """True hybrid recommender using LightFM library.

    This model trains features directly in the matrix factorization,
    providing true hybrid recommendations with cold start support.

    Note: Requires LightFM library, which works on Linux/WSL but not Windows.
    """

    DEFAULT_EVENT_WEIGHTS = {
        "view": 1.0,
        "addtocart": 2.0,
        "transaction": 3.0,
    }

    def __init__(
        self,
        no_components: int = 64,
        loss: str = "warp",
        epochs: int = 30,
        learning_rate: float = 0.05,
        random_state: int = 42,
        event_weights: dict[str, float] | None = None,
    ):
        """Initialize TrueLightFMRecommender.

        Args:
            no_components: Number of latent factors.
            loss: Loss function ('warp', 'bpr', 'logistic', 'warp-kos').
            epochs: Number of training epochs.
            learning_rate: Learning rate.
            random_state: Random seed.
            event_weights: Custom weights for event types.
        """
        super().__init__(name="TrueLightFMRecommender")
        self.no_components = no_components
        self.loss = loss
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.random_state = random_state
        self.event_weights = event_weights or self.DEFAULT_EVENT_WEIGHTS

        self.model = None
        self.interactions = None
        self.item_features = None
        self.user_features = None

        self.user_to_idx: dict[int, int] = {}
        self.idx_to_user: dict[int, int] = {}
        self.item_to_idx: dict[int, int] = {}
        self.idx_to_item: dict[int, int] = {}

        self.n_user_features = 0
        self.n_item_features = 0
        self.user_feature_names: list[str] = []
        self.item_feature_names: list[str] = []

    def _prepare_interactions(self, events: pl.DataFrame) -> csr_matrix:
        """Prepare weighted interaction matrix."""
        logger.info("Preparing interactions for LightFM...")

        unique_users = events["user_id"].unique().sort().to_list()
        unique_items = events["item_id"].unique().sort().to_list()

        self.user_to_idx = {user: idx for idx, user in enumerate(unique_users)}
        self.idx_to_user = {idx: user for user, idx in self.user_to_idx.items()}
        self.item_to_idx = {item: idx for idx, item in enumerate(unique_items)}
        self.idx_to_item = {idx: item for item, idx in self.item_to_idx.items()}

        n_users = len(unique_users)
        n_items = len(unique_items)
        logger.info(f"Users: {n_users:,}, Items: {n_items:,}")

        weighted_events = events.with_columns(
            pl.col("event_type")
            .replace(self.event_weights)
            .cast(pl.Float64)
            .alias("weight")
        )

        interaction_weights = (
            weighted_events.group_by(["user_id", "item_id"])
            .agg(pl.col("weight").sum().alias("weight"))
        )

        row_indices = [self.user_to_idx[u] for u in interaction_weights["user_id"].to_list()]
        col_indices = [self.item_to_idx[i] for i in interaction_weights["item_id"].to_list()]
        weights = interaction_weights["weight"].to_list()

        matrix = coo_matrix(
            (weights, (row_indices, col_indices)),
            shape=(n_users, n_items),
            dtype=np.float32,
        )

        logger.info(f"Interactions: {matrix.shape}, nnz={matrix.nnz:,}")
        return matrix.tocsr()

    def _prepare_item_features(
        self,
        events: pl.DataFrame,
        top_n_categories: int = 50,
    ) -> csr_matrix:
        """Prepare item features matrix for LightFM."""
        from scipy.sparse import identity, hstack

        logger.info("Preparing item features for LightFM...")
        n_items = len(self.item_to_idx)

        # Get top categories
        category_counts = (
            events.filter(pl.col("category_id") != "unknown")
            .group_by("category_id")
            .agg(pl.len().alias("count"))
            .sort("count", descending=True)
            .head(top_n_categories)
        )

        categories = category_counts["category_id"].to_list()
        cat_to_idx = {cat: idx for idx, cat in enumerate(categories)}

        # Item to category mapping
        item_cat = (
            events.filter(pl.col("category_id").is_in(categories))
            .group_by("item_id")
            .agg(pl.col("category_id").first())
        )

        item_to_cat = {row["item_id"]: row["category_id"] for row in item_cat.iter_rows(named=True)}

        # Build category feature matrix
        row_idx, col_idx = [], []
        for item_id, item_idx in self.item_to_idx.items():
            cat = item_to_cat.get(item_id)
            if cat and cat in cat_to_idx:
                row_idx.append(item_idx)
                col_idx.append(cat_to_idx[cat])

        cat_matrix = coo_matrix(
            (np.ones(len(row_idx)), (row_idx, col_idx)),
            shape=(n_items, len(categories)),
            dtype=np.float32,
        ).tocsr()

        # Combine: [identity | categories]
        item_identity = identity(n_items, dtype=np.float32, format="csr")
        item_features = hstack([item_identity, cat_matrix], format="csr")

        self.n_item_features = item_features.shape[1]
        self.item_feature_names = [f"item_{i}" for i in range(n_items)] + categories

        logger.info(f"Item features shape: {item_features.shape}")
        return item_features

    def _prepare_user_features(
        self,
        events: pl.DataFrame,
        rfm_data: pl.DataFrame | None = None,
    ) -> csr_matrix:
        """Prepare user features matrix for LightFM."""
        from scipy.sparse import identity, hstack

        logger.info("Preparing user features for LightFM...")
        n_users = len(self.user_to_idx)

        feature_matrices = []
        feature_names = [f"user_{i}" for i in range(n_users)]

        # User identity
        user_identity = identity(n_users, dtype=np.float32, format="csr")
        feature_matrices.append(user_identity)

        # Segment features
        if rfm_data is not None and "segment" in rfm_data.columns:
            segments = rfm_data["segment"].unique().to_list()
            seg_to_idx = {seg: idx for idx, seg in enumerate(segments)}

            user_to_seg = {
                row["user_id"]: row["segment"]
                for row in rfm_data.iter_rows(named=True)
            }

            row_idx, col_idx = [], []
            for user_id, user_idx in self.user_to_idx.items():
                seg = user_to_seg.get(user_id)
                if seg and seg in seg_to_idx:
                    row_idx.append(user_idx)
                    col_idx.append(seg_to_idx[seg])

            seg_matrix = coo_matrix(
                (np.ones(len(row_idx)), (row_idx, col_idx)),
                shape=(n_users, len(segments)),
                dtype=np.float32,
            ).tocsr()

            feature_matrices.append(seg_matrix)
            feature_names.extend([f"seg_{s}" for s in segments])
            logger.info(f"Added {len(segments)} segment features")

        # Favorite category features
        user_fav_cat = (
            events.filter(
                pl.col("event_type").is_in(["transaction", "addtocart"])
                & (pl.col("category_id") != "unknown")
            )
            .group_by(["user_id", "category_id"])
            .agg(pl.len().alias("count"))
            .sort(["user_id", "count"], descending=[False, True])
            .group_by("user_id")
            .agg(pl.col("category_id").first().alias("fav_cat"))
        )

        fav_cats = user_fav_cat["fav_cat"].unique().to_list()[:30]
        cat_to_idx = {cat: idx for idx, cat in enumerate(fav_cats)}

        user_to_fav = {
            row["user_id"]: row["fav_cat"]
            for row in user_fav_cat.iter_rows(named=True)
        }

        row_idx, col_idx = [], []
        for user_id, user_idx in self.user_to_idx.items():
            cat = user_to_fav.get(user_id)
            if cat and cat in cat_to_idx:
                row_idx.append(user_idx)
                col_idx.append(cat_to_idx[cat])

        fav_matrix = coo_matrix(
            (np.ones(len(row_idx)), (row_idx, col_idx)),
            shape=(n_users, len(fav_cats)),
            dtype=np.float32,
        ).tocsr()

        feature_matrices.append(fav_matrix)
        feature_names.extend([f"fav_{c}" for c in fav_cats])

        # Combine all
        user_features = hstack(feature_matrices, format="csr")
        self.n_user_features = user_features.shape[1]
        self.user_feature_names = feature_names

        logger.info(f"User features shape: {user_features.shape}")
        return user_features

    def fit(
        self,
        events: pl.DataFrame,
        rfm_data: pl.DataFrame | None = None,
        use_item_features: bool = True,
        use_user_features: bool = True,
        **kwargs,
    ) -> "TrueLightFMRecommender":
        """Fit LightFM model on interaction data.

        Args:
            events: Events DataFrame.
            rfm_data: Optional RFM segmentation DataFrame.
            use_item_features: Whether to use item features.
            use_user_features: Whether to use user features.

        Returns:
            Self for method chaining.
        """
        try:
            from lightfm import LightFM
        except ImportError:
            raise ImportError(
                "LightFM is not installed. On Windows, use WSL or HybridRecommender instead."
            )

        logger.info(f"Fitting {self.name}...")
        logger.info(f"Parameters: components={self.no_components}, loss={self.loss}, epochs={self.epochs}")

        start_time = time.time()

        # Prepare interactions
        self.interactions = self._prepare_interactions(events)

        # Prepare features
        self.item_features = None
        self.user_features = None

        if use_item_features:
            self.item_features = self._prepare_item_features(events)

        if use_user_features:
            self.user_features = self._prepare_user_features(events, rfm_data)

        # Initialize model
        self.model = LightFM(
            no_components=self.no_components,
            loss=self.loss,
            learning_rate=self.learning_rate,
            random_state=self.random_state,
        )

        # Train
        logger.info(f"Training LightFM for {self.epochs} epochs...")
        self.model.fit(
            self.interactions,
            item_features=self.item_features,
            user_features=self.user_features,
            epochs=self.epochs,
            num_threads=4,
            verbose=True,
        )

        elapsed = time.time() - start_time
        self._is_fitted = True

        logger.info(f"Fitted {self.name} in {elapsed:.2f}s")
        return self

    def recommend(
        self,
        user_id: int,
        n: int = 10,
        filter_already_liked: bool = True,
    ) -> list[int]:
        """Generate recommendations for a known user."""
        self._check_is_fitted()

        if user_id not in self.user_to_idx:
            logger.warning(f"User {user_id} not in training data")
            return self.recommend_cold_start({}, n)

        user_idx = self.user_to_idx[user_id]
        n_items = len(self.item_to_idx)

        # Get scores for all items
        scores = self.model.predict(
            user_ids=user_idx,
            item_ids=np.arange(n_items),
            item_features=self.item_features,
            user_features=self.user_features,
        )

        # Filter already liked
        if filter_already_liked:
            liked_items = self.interactions[user_idx].indices
            scores[liked_items] = -np.inf

        # Get top N
        top_indices = np.argsort(-scores)[:n]
        return [self.idx_to_item[idx] for idx in top_indices]

    def recommend_cold_start(
        self,
        user_features_dict: dict[str, Any],
        n: int = 10,
    ) -> list[int]:
        """Generate recommendations for a new user based on features."""
        self._check_is_fitted()

        n_items = len(self.item_to_idx)

        # Build user feature vector
        if self.user_features is not None:
            feature_vec = np.zeros(self.n_user_features, dtype=np.float32)

            segment = user_features_dict.get("segment")
            if segment:
                seg_feature = f"seg_{segment}"
                if seg_feature in self.user_feature_names:
                    idx = self.user_feature_names.index(seg_feature)
                    feature_vec[idx] = 1.0

            fav_cat = user_features_dict.get("favorite_category")
            if fav_cat:
                fav_feature = f"fav_{fav_cat}"
                if fav_feature in self.user_feature_names:
                    idx = self.user_feature_names.index(fav_feature)
                    feature_vec[idx] = 1.0

            user_features_sparse = csr_matrix(feature_vec.reshape(1, -1))
        else:
            user_features_sparse = None

        # Get scores using a virtual user (index 0, but with custom features)
        scores = self.model.predict(
            user_ids=0,
            item_ids=np.arange(n_items),
            item_features=self.item_features,
            user_features=user_features_sparse,
        )

        top_indices = np.argsort(-scores)[:n]
        return [self.idx_to_item[idx] for idx in top_indices]

    def similar_items(self, item_id: int, n: int = 10) -> list[int]:
        """Find similar items based on item embeddings."""
        self._check_is_fitted()

        if item_id not in self.item_to_idx:
            logger.warning(f"Item {item_id} not in training data")
            return []

        item_idx = self.item_to_idx[item_id]

        # Get item embeddings
        if self.item_features is not None:
            # Use biases and embeddings with features
            item_biases, item_embeddings = self.model.get_item_representations(self.item_features)
        else:
            item_biases, item_embeddings = self.model.get_item_representations()

        # Compute cosine similarity
        target_embedding = item_embeddings[item_idx]
        norms = np.linalg.norm(item_embeddings, axis=1)
        target_norm = np.linalg.norm(target_embedding)

        similarities = np.dot(item_embeddings, target_embedding) / (norms * target_norm + 1e-10)
        similarities[item_idx] = -np.inf  # Exclude self

        top_indices = np.argsort(-similarities)[:n]
        return [self.idx_to_item[idx] for idx in top_indices]

    def recommend_for_new_user(
        self,
        session_items: list[int],
        n: int = 10,
    ) -> list[int]:
        """Рекомендации для нового пользователя на основе текущей сессии.

        Использует item embeddings LightFM для поиска похожих товаров.

        Args:
            session_items: Список item_id из текущей сессии.
            n: Количество рекомендаций.

        Returns:
            Список рекомендуемых item_id.
        """
        self._check_is_fitted()

        # Фильтрация известных товаров
        known_items = [i for i in session_items if i in self.item_to_idx]

        if not known_items:
            logger.warning("No known items in session, falling back to cold start")
            return self.recommend_cold_start({}, n)

        # Получение item embeddings
        if self.item_features is not None:
            _, item_embeddings = self.model.get_item_representations(self.item_features)
        else:
            _, item_embeddings = self.model.get_item_representations()

        # Средний embedding товаров сессии
        session_indices = [self.item_to_idx[i] for i in known_items]
        session_embedding = np.mean(item_embeddings[session_indices], axis=0)

        # Косинусное сходство
        norms = np.linalg.norm(item_embeddings, axis=1)
        session_norm = np.linalg.norm(session_embedding)
        similarities = np.dot(item_embeddings, session_embedding) / (norms * session_norm + 1e-10)

        # Исключить товары из сессии
        for item_id in known_items:
            similarities[self.item_to_idx[item_id]] = -np.inf

        # Топ-N
        top_indices = np.argsort(-similarities)[:n]
        return [self.idx_to_item[idx] for idx in top_indices]

    def recommend_new_item(
        self,
        item_features: dict,
        n: int = 10,
    ) -> list[int]:
        """Найти пользователей для нового товара.

        Args:
            item_features: Характеристики товара {"category_id": str}.
            n: Количество пользователей.

        Returns:
            Список user_id, которым подходит товар.
        """
        self._check_is_fitted()

        category_id = item_features.get("category_id")

        if not category_id:
            logger.warning("No category_id provided")
            return []

        # Найти товары с такой категорией в feature names
        category_items = []
        for item_id, item_idx in self.item_to_idx.items():
            # Проверить в item_feature_names
            if category_id in self.item_feature_names:
                category_items.append(item_id)

        # Если категории нет в features, ищем по взаимодействиям
        if not category_items:
            # Fallback: найти товары с высоким сходством к любому товару этой категории
            logger.warning(f"Category {category_id} not in item features, using interaction-based search")
            return []

        # Получить пользователей, взаимодействовавших с этими товарами
        user_scores: dict[int, float] = {}

        for item_id in category_items[:50]:
            if item_id in self.item_to_idx:
                item_idx = self.item_to_idx[item_id]
                user_indices = self.interactions[:, item_idx].nonzero()[0]
                for user_idx in user_indices:
                    user_id = self.idx_to_user[user_idx]
                    user_scores[user_id] = user_scores.get(user_id, 0) + 1

        if not user_scores:
            return []

        # Топ-N пользователей
        sorted_users = sorted(user_scores.items(), key=lambda x: -x[1])
        return [u[0] for u in sorted_users[:n]]

    def save(self, path: str | Path) -> None:
        """Save model to disk."""
        self._check_is_fitted()
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        data = {
            "model": self.model,
            "interactions": self.interactions,
            "item_features": self.item_features,
            "user_features": self.user_features,
            "user_to_idx": self.user_to_idx,
            "idx_to_user": self.idx_to_user,
            "item_to_idx": self.item_to_idx,
            "idx_to_item": self.idx_to_item,
            "n_user_features": self.n_user_features,
            "n_item_features": self.n_item_features,
            "user_feature_names": self.user_feature_names,
            "item_feature_names": self.item_feature_names,
            "no_components": self.no_components,
            "loss": self.loss,
            "epochs": self.epochs,
            "learning_rate": self.learning_rate,
            "event_weights": self.event_weights,
        }

        with open(path, "wb") as f:
            pickle.dump(data, f)

        logger.info(f"Saved {self.name} to {path}")

    def load(self, path: str | Path) -> "TrueLightFMRecommender":
        """Load model from disk."""
        path = Path(path)

        with open(path, "rb") as f:
            data = pickle.load(f)

        self.model = data["model"]
        self.interactions = data["interactions"]
        self.item_features = data["item_features"]
        self.user_features = data["user_features"]
        self.user_to_idx = data["user_to_idx"]
        self.idx_to_user = data["idx_to_user"]
        self.item_to_idx = data["item_to_idx"]
        self.idx_to_item = data["idx_to_item"]
        self.n_user_features = data["n_user_features"]
        self.n_item_features = data["n_item_features"]
        self.user_feature_names = data["user_feature_names"]
        self.item_feature_names = data["item_feature_names"]
        self.no_components = data["no_components"]
        self.loss = data["loss"]
        self.epochs = data["epochs"]
        self.learning_rate = data["learning_rate"]
        self.event_weights = data["event_weights"]
        self._is_fitted = True

        logger.info(f"Loaded {self.name} from {path}")
        return self


# Alias for backwards compatibility
LightFMRecommender = HybridRecommender
