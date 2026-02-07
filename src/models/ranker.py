"""CatBoost-based re-ranker for two-stage recommendations."""

import pickle
import random
import time
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import polars as pl
from loguru import logger

if TYPE_CHECKING:
    from src.models.hybrid import HybridRecommender


class CatBoostRanker:
    """Re-ranker using CatBoost for learning-to-rank.

    Features used:
    - candidate_score: Score from retriever model
    - item_popularity: Normalized interaction count
    - category_match: 1 if item category matches user's favorite
    - user_segment_encoded: Ordinal encoding of RFM segment
    - days_since_category_interaction: Recency of user's category interaction
    """

    # RFM segment ordering (lower index = worse customer)
    SEGMENT_ORDER = [
        "Lost",
        "Hibernating",
        "About To Sleep",
        "At Risk",
        "Can't Lose Them",
        "Need Attention",
        "New Customers",
        "Promising",
        "Potential Loyalists",
        "Loyal Customers",
        "Champions",
    ]

    FEATURE_NAMES = [
        "candidate_score",
        "item_popularity",
        "category_match",
        "user_segment_encoded",
        "days_since_category_interaction",
    ]

    def __init__(
        self,
        iterations: int = 500,
        learning_rate: float = 0.03,
        depth: int = 6,
        loss_function: str = "YetiRank",
        random_state: int = 42,
    ):
        """Initialize CatBoostRanker.

        Args:
            iterations: Number of boosting iterations.
            learning_rate: Learning rate.
            depth: Tree depth.
            loss_function: Ranking loss ('YetiRank', 'PairLogit', 'QueryRMSE').
            random_state: Random seed for reproducibility.
        """
        self.iterations = iterations
        self.learning_rate = learning_rate
        self.depth = depth
        self.loss_function = loss_function
        self.random_state = random_state

        self.model = None
        self._is_fitted = False

        # Feature data (populated during fit)
        self.item_popularity: dict[int, float] = {}
        self.item_categories: dict[int, str] = {}
        self.user_segments: dict[int, str] = {}
        self.user_favorite_category: dict[int, str] = {}
        self.user_category_last_ts: dict[int, dict[str, int]] = {}

        # Reference timestamp for recency calculation
        self.reference_timestamp: int = 0

    def _compute_item_popularity(self, events: pl.DataFrame) -> dict[int, float]:
        """Compute normalized item popularity scores.

        Args:
            events: Training events DataFrame.

        Returns:
            Dictionary mapping item_id to popularity score [0, 1].
        """
        item_counts = (
            events.group_by("item_id")
            .agg(pl.len().alias("count"))
        )

        max_count = item_counts["count"].max()

        popularity = {}
        for row in item_counts.iter_rows(named=True):
            popularity[row["item_id"]] = row["count"] / max_count

        return popularity

    def _compute_user_category_last_ts(
        self,
        events: pl.DataFrame,
    ) -> dict[int, dict[str, int]]:
        """Compute last interaction timestamp per user-category pair.

        Args:
            events: Training events DataFrame.

        Returns:
            Dict[user_id, Dict[category_id, last_timestamp]].
        """
        user_cat_ts = (
            events.group_by(["user_id", "category_id"])
            .agg(pl.col("timestamp").max().alias("last_ts"))
        )

        result: dict[int, dict[str, int]] = {}
        for row in user_cat_ts.iter_rows(named=True):
            user_id = row["user_id"]
            if user_id not in result:
                result[user_id] = {}
            result[user_id][row["category_id"]] = row["last_ts"]

        return result

    def _encode_segment(self, segment: str | None) -> int:
        """Encode RFM segment as ordinal value.

        Args:
            segment: RFM segment name.

        Returns:
            Ordinal value (0-11).
        """
        if segment is None:
            return len(self.SEGMENT_ORDER)  # 11 for unknown

        if segment in self.SEGMENT_ORDER:
            return self.SEGMENT_ORDER.index(segment)

        return len(self.SEGMENT_ORDER)

    def _build_features(
        self,
        user_id: int,
        item_id: int,
        candidate_score: float,
        reference_ts: int,
    ) -> list[float]:
        """Build feature vector for a user-item pair.

        Features:
        1. candidate_score: Score from retriever
        2. item_popularity: Normalized popularity
        3. category_match: 1 if matches user's favorite category
        4. user_segment_encoded: Ordinal RFM segment
        5. days_since_category_interaction: Days since user interacted with category

        Args:
            user_id: User identifier.
            item_id: Item identifier.
            candidate_score: Score from retriever model.
            reference_ts: Reference timestamp for recency calculation.

        Returns:
            Feature vector [5 features].
        """
        # Feature 1: candidate score
        # Feature 2: item popularity
        item_pop = self.item_popularity.get(item_id, 0.0)

        # Feature 3: category match
        item_cat = self.item_categories.get(item_id, "unknown")
        user_fav = self.user_favorite_category.get(user_id, "")
        cat_match = 1.0 if item_cat == user_fav and item_cat != "unknown" else 0.0

        # Feature 4: segment encoding
        segment = self.user_segments.get(user_id)
        seg_encoded = float(self._encode_segment(segment))

        # Feature 5: days since category interaction
        user_cat_ts = self.user_category_last_ts.get(user_id, {})
        last_cat_ts = user_cat_ts.get(item_cat, 0)

        if last_cat_ts > 0:
            # Convert milliseconds to days
            days_since = (reference_ts - last_cat_ts) / (86400 * 1000)
        else:
            days_since = 365.0  # Default for no interaction

        return [candidate_score, item_pop, cat_match, seg_encoded, days_since]

    def _prepare_training_data(
        self,
        events: pl.DataFrame,
        retriever: "HybridRecommender",
        n_negatives: int = 4,
        n_candidates: int = 100,
        max_users: int = 5000,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Generate training data with positive and negative samples.

        Args:
            events: Training events DataFrame.
            retriever: Fitted HybridRecommender for candidate generation.
            n_negatives: Number of negative samples per positive.
            n_candidates: Number of candidates to retrieve per user.
            max_users: Maximum number of users to sample.

        Returns:
            Tuple of (features, labels, group_ids) for CatBoost.
        """
        logger.info("Preparing training data for ranker...")

        features = []
        labels = []
        group_ids = []

        group_id = 0
        ref_ts = self.reference_timestamp

        # Get users with transactions
        users_with_transactions = (
            events.filter(pl.col("event_type") == "transaction")
            ["user_id"].unique().to_list()
        )

        # Filter to users known by retriever
        valid_users = [
            u for u in users_with_transactions
            if u in retriever.user_to_idx
        ]

        # Sample users if too many
        if len(valid_users) > max_users:
            random.seed(self.random_state)
            valid_users = random.sample(valid_users, max_users)

        logger.info(f"Processing {len(valid_users)} users for training data...")

        # Build user -> interacted items mapping
        user_items_map: dict[int, set[int]] = {}
        for row in events.select(["user_id", "item_id"]).unique().iter_rows(named=True):
            user_id = row["user_id"]
            if user_id not in user_items_map:
                user_items_map[user_id] = set()
            user_items_map[user_id].add(row["item_id"])

        processed = 0
        for user_id in valid_users:
            user_items = user_items_map.get(user_id, set())

            if not user_items:
                continue

            # Get candidates with scores from retriever
            candidates = retriever.recommend_with_scores(
                user_id, n=n_candidates, filter_already_liked=False
            )

            if not candidates:
                continue

            # Build candidate lookup
            cand_items = {item_id for item_id, _ in candidates}
            cand_scores = {item_id: score for item_id, score in candidates}

            # Find positives (items user interacted with that are in candidates)
            positives = user_items & cand_items
            negatives = list(cand_items - user_items)

            if not positives or not negatives:
                continue

            # Sample positives if too many
            positives_list = list(positives)
            if len(positives_list) > 10:
                positives_list = random.sample(positives_list, 10)

            for pos_item in positives_list:
                pos_score = cand_scores[pos_item]

                # Add positive sample
                feat = self._build_features(user_id, pos_item, pos_score, ref_ts)
                features.append(feat)
                labels.append(1)
                group_ids.append(group_id)

                # Add negative samples
                n_neg = min(n_negatives, len(negatives))
                neg_sample = random.sample(negatives, n_neg)

                for neg_item in neg_sample:
                    neg_score = cand_scores[neg_item]
                    feat = self._build_features(user_id, neg_item, neg_score, ref_ts)
                    features.append(feat)
                    labels.append(0)
                    group_ids.append(group_id)

                group_id += 1

            processed += 1
            if processed % 1000 == 0:
                logger.info(f"Processed {processed}/{len(valid_users)} users")

        logger.info(f"Generated {len(features)} training samples in {group_id} groups")

        return np.array(features), np.array(labels), np.array(group_ids)

    def fit(
        self,
        events: pl.DataFrame,
        retriever: "HybridRecommender",
        rfm_data: pl.DataFrame | None = None,
        n_negatives: int = 4,
        **kwargs,
    ) -> "CatBoostRanker":
        """Fit the ranker on training data.

        Args:
            events: Training events DataFrame.
            retriever: Fitted HybridRecommender.
            rfm_data: RFM segmentation DataFrame.
            n_negatives: Number of negative samples per positive.

        Returns:
            Self for method chaining.
        """
        try:
            from catboost import CatBoostRanker as CatBoostRankerModel
        except ImportError:
            raise ImportError(
                "CatBoost is not installed. Install with: pip install catboost"
            )

        logger.info("Fitting CatBoostRanker...")
        start_time = time.time()

        # Store reference timestamp
        self.reference_timestamp = events["timestamp"].max()

        # Copy feature data from retriever
        self.item_categories = retriever.item_categories.copy()
        self.user_segments = retriever.user_segments.copy()
        self.user_favorite_category = retriever.user_favorite_category.copy()

        # Compute additional features
        logger.info("Computing item popularity...")
        self.item_popularity = self._compute_item_popularity(events)

        logger.info("Computing user-category timestamps...")
        self.user_category_last_ts = self._compute_user_category_last_ts(events)

        # Load RFM segments if provided and not already loaded
        if rfm_data is not None and not self.user_segments:
            for row in rfm_data.iter_rows(named=True):
                self.user_segments[row["user_id"]] = row["segment"]

        # Prepare training data
        X, y, qid = self._prepare_training_data(
            events, retriever, n_negatives=n_negatives
        )

        if len(X) == 0:
            logger.warning("No training data generated, ranker will pass through")
            self._is_fitted = True
            return self

        # Initialize and train CatBoost
        logger.info(f"Training CatBoost with {len(X)} samples...")

        self.model = CatBoostRankerModel(
            iterations=self.iterations,
            learning_rate=self.learning_rate,
            depth=self.depth,
            loss_function=self.loss_function,
            random_seed=self.random_state,
            verbose=100,
        )

        self.model.fit(
            X, y,
            group_id=qid,
            verbose=100,
        )

        elapsed = time.time() - start_time
        self._is_fitted = True

        logger.info(f"Fitted CatBoostRanker in {elapsed:.2f}s")

        return self

    def rank(
        self,
        user_id: int,
        candidates: list[tuple[int, float]],
        n: int = 10,
        reference_timestamp: int | None = None,
    ) -> list[int]:
        """Re-rank candidates by predicted score.

        Args:
            user_id: User identifier.
            candidates: List of (item_id, score) tuples from retriever.
            n: Number of items to return.
            reference_timestamp: Timestamp for recency features.

        Returns:
            List of top-n item IDs ordered by predicted relevance.
        """
        if not self._is_fitted:
            raise RuntimeError("Ranker is not fitted. Call fit() first.")

        if not candidates:
            return []

        # If no model trained (no data), fall back to retriever order
        if self.model is None:
            return [item_id for item_id, _ in candidates[:n]]

        ref_ts = reference_timestamp or self.reference_timestamp

        # Build features for all candidates
        features = []
        item_ids = []

        for item_id, score in candidates:
            feat = self._build_features(user_id, item_id, score, ref_ts)
            features.append(feat)
            item_ids.append(item_id)

        X = np.array(features)

        # Predict relevance scores
        scores = self.model.predict(X)

        # Sort by predicted score descending
        sorted_indices = np.argsort(-scores)[:n]

        return [item_ids[i] for i in sorted_indices]

    def save(self, path: str | Path) -> None:
        """Save ranker to disk.

        Args:
            path: Path to save the model.
        """
        if not self._is_fitted:
            raise RuntimeError("Ranker is not fitted. Call fit() first.")

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        data = {
            "model": self.model,
            "item_popularity": self.item_popularity,
            "item_categories": self.item_categories,
            "user_segments": self.user_segments,
            "user_favorite_category": self.user_favorite_category,
            "user_category_last_ts": self.user_category_last_ts,
            "reference_timestamp": self.reference_timestamp,
            "iterations": self.iterations,
            "learning_rate": self.learning_rate,
            "depth": self.depth,
            "loss_function": self.loss_function,
        }

        with open(path, "wb") as f:
            pickle.dump(data, f)

        logger.info(f"Saved CatBoostRanker to {path}")

    def load(self, path: str | Path) -> "CatBoostRanker":
        """Load ranker from disk.

        Args:
            path: Path to load the model from.

        Returns:
            Self for method chaining.
        """
        path = Path(path)

        with open(path, "rb") as f:
            data = pickle.load(f)

        self.model = data["model"]
        self.item_popularity = data["item_popularity"]
        self.item_categories = data["item_categories"]
        self.user_segments = data["user_segments"]
        self.user_favorite_category = data["user_favorite_category"]
        self.user_category_last_ts = data["user_category_last_ts"]
        self.reference_timestamp = data["reference_timestamp"]
        self.iterations = data["iterations"]
        self.learning_rate = data["learning_rate"]
        self.depth = data["depth"]
        self.loss_function = data["loss_function"]
        self._is_fitted = True

        logger.info(f"Loaded CatBoostRanker from {path}")

        return self

    def __repr__(self) -> str:
        return (
            f"CatBoostRanker(iterations={self.iterations}, "
            f"depth={self.depth}, fitted={self._is_fitted})"
        )
