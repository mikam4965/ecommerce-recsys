"""Content-Based Filtering Recommender.

Uses item category features to build user preference profiles
and recommend items from preferred categories.

Algorithm:
    1. Build user profile = weighted vector of category interactions
       (transaction=5, addtocart=3, view=1)
    2. Build item vectors = one-hot category encoding
    3. Score items by cosine similarity between user profile and item vector
    4. Within same category, rank by item popularity
"""

import pickle
from collections import defaultdict
from pathlib import Path

import numpy as np
import polars as pl
from loguru import logger

from src.models.base import BaseRecommender

# Event type weights for user profile construction
EVENT_WEIGHTS = {
    "transaction": 5.0,
    "addtocart": 3.0,
    "view": 1.0,
}


class ContentBasedRecommender(BaseRecommender):
    """Content-based recommender using item category features.

    Builds user preference profiles from interaction history weighted by
    event type, then recommends items from preferred categories ranked
    by popularity within each category.
    """

    def __init__(self, name: str = "ContentBased"):
        """Initialize ContentBasedRecommender.

        Args:
            name: Model name for logging.
        """
        super().__init__(name=name)

        # Mappings
        self.user_to_idx: dict[int, int] = {}
        self.idx_to_user: dict[int, int] = {}
        self.item_to_idx: dict[int, int] = {}
        self.idx_to_item: dict[int, int] = {}
        self.category_to_idx: dict[int, int] = {}

        # Item features
        self.item_categories: dict[int, int] = {}       # item_id → category_id
        self.category_items: dict[int, list[int]] = {}   # category_id → [item_ids sorted by popularity]
        self.item_popularity: dict[int, int] = {}        # item_id → interaction count

        # User profiles (numpy vectors of length n_categories)
        self.user_profiles: dict[int, np.ndarray] = {}

        # User interaction history (for filtering already liked)
        self.user_items: dict[int, set[int]] = {}

        self.n_categories = 0

    def fit(self, interactions: pl.DataFrame, **kwargs) -> "ContentBasedRecommender":
        """Fit content-based model on interaction data.

        Args:
            interactions: DataFrame with columns: user_id, item_id, event_type, category_id.

        Returns:
            Self for method chaining.
        """
        logger.info(f"Fitting {self.name}...")
        logger.info(f"  Events: {len(interactions):,}")
        logger.info(f"  Users: {interactions['user_id'].n_unique():,}")
        logger.info(f"  Items: {interactions['item_id'].n_unique():,}")

        # 1. Build ID mappings
        unique_users = interactions["user_id"].unique().sort().to_list()
        unique_items = interactions["item_id"].unique().sort().to_list()

        self.user_to_idx = {uid: i for i, uid in enumerate(unique_users)}
        self.idx_to_user = {i: uid for uid, i in self.user_to_idx.items()}
        self.item_to_idx = {iid: i for i, iid in enumerate(unique_items)}
        self.idx_to_item = {i: iid for iid, i in self.item_to_idx.items()}

        # 2. Build category mapping
        if "category_id" not in interactions.columns:
            logger.warning("No category_id column found, using dummy category")
            interactions = interactions.with_columns(pl.lit(0).alias("category_id"))

        item_cat = (
            interactions.group_by("item_id")
            .agg(pl.col("category_id").first())
        )
        self.item_categories = dict(zip(
            item_cat["item_id"].to_list(),
            item_cat["category_id"].to_list(),
        ))

        unique_categories = sorted(set(self.item_categories.values()))
        self.category_to_idx = {cat: i for i, cat in enumerate(unique_categories)}
        self.n_categories = len(unique_categories)
        logger.info(f"  Categories: {self.n_categories:,}")

        # 3. Compute item popularity
        item_pop = (
            interactions.group_by("item_id")
            .agg(pl.len().alias("count"))
        )
        self.item_popularity = dict(zip(
            item_pop["item_id"].to_list(),
            item_pop["count"].to_list(),
        ))

        # 4. Build category → items (sorted by popularity descending)
        cat_items_dict: dict[int, list[tuple[int, int]]] = defaultdict(list)
        for item_id, cat_id in self.item_categories.items():
            pop = self.item_popularity.get(item_id, 0)
            cat_items_dict[cat_id].append((item_id, pop))

        self.category_items = {}
        for cat_id, items_with_pop in cat_items_dict.items():
            items_with_pop.sort(key=lambda x: x[1], reverse=True)
            self.category_items[cat_id] = [item_id for item_id, _ in items_with_pop]

        # 5. Build user profiles (weighted category vectors)
        self.user_profiles = {}
        self.user_items = defaultdict(set)

        # Process interactions grouped by user
        user_groups = interactions.group_by("user_id").agg([
            pl.col("item_id"),
            pl.col("event_type"),
            pl.col("category_id"),
        ])

        for row in user_groups.iter_rows(named=True):
            user_id = row["user_id"]
            item_ids = row["item_id"]
            event_types = row["event_type"]
            category_ids = row["category_id"]

            # Build category weight vector
            profile = np.zeros(self.n_categories, dtype=np.float32)

            for item_id, event_type, cat_id in zip(item_ids, event_types, category_ids):
                self.user_items[user_id].add(item_id)

                cat_idx = self.category_to_idx.get(cat_id)
                if cat_idx is not None:
                    weight = EVENT_WEIGHTS.get(event_type, 1.0)
                    profile[cat_idx] += weight

            # L2 normalize for cosine similarity
            norm = np.linalg.norm(profile)
            if norm > 0:
                profile /= norm

            self.user_profiles[user_id] = profile

        self._is_fitted = True
        logger.info(f"  {self.name} fitted: {len(self.user_profiles):,} user profiles, "
                     f"{self.n_categories:,} categories")
        return self

    def recommend(
        self,
        user_id: int,
        n: int = 10,
        filter_already_liked: bool = True,
    ) -> list[int]:
        """Recommend items for a user based on category preferences.

        Args:
            user_id: User identifier.
            n: Number of recommendations.
            filter_already_liked: Whether to filter already interacted items.

        Returns:
            List of recommended item IDs.
        """
        self._check_is_fitted()

        profile = self.user_profiles.get(user_id)
        if profile is None:
            return []

        liked_items = self.user_items.get(user_id, set()) if filter_already_liked else set()

        # Score each category by user affinity
        # category_scores[cat_idx] = dot(user_profile, one_hot(cat_idx)) = profile[cat_idx]
        # Since item vectors are one-hot, the score for items in category c = profile[c]

        # Build list of (score, popularity, item_id) for ranking
        candidates = []
        idx_to_cat = {i: cat for cat, i in self.category_to_idx.items()}

        for cat_idx in range(self.n_categories):
            score = float(profile[cat_idx])
            if score <= 0:
                continue

            cat_id = idx_to_cat[cat_idx]
            items_in_cat = self.category_items.get(cat_id, [])

            for item_id in items_in_cat:
                if item_id in liked_items:
                    continue
                pop = self.item_popularity.get(item_id, 0)
                # Primary sort by category affinity, secondary by popularity
                candidates.append((score, pop, item_id))

        # Sort: highest score first, then highest popularity
        candidates.sort(key=lambda x: (x[0], x[1]), reverse=True)

        return [item_id for _, _, item_id in candidates[:n]]

    def save(self, path: str | Path) -> None:
        """Save model to disk.

        Args:
            path: Path to save the model.
        """
        self._check_is_fitted()
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        state = {
            "name": self.name,
            "user_to_idx": self.user_to_idx,
            "idx_to_user": self.idx_to_user,
            "item_to_idx": self.item_to_idx,
            "idx_to_item": self.idx_to_item,
            "category_to_idx": self.category_to_idx,
            "n_categories": self.n_categories,
            "item_categories": self.item_categories,
            "category_items": self.category_items,
            "item_popularity": self.item_popularity,
            "user_profiles": self.user_profiles,
            "user_items": {k: v for k, v in self.user_items.items()},
        }

        with open(path, "wb") as f:
            pickle.dump(state, f, protocol=pickle.HIGHEST_PROTOCOL)

        logger.info(f"  {self.name} saved to {path}")

    def load(self, path: str | Path) -> "ContentBasedRecommender":
        """Load model from disk.

        Args:
            path: Path to load the model from.

        Returns:
            Self for method chaining.
        """
        path = Path(path)

        with open(path, "rb") as f:
            state = pickle.load(f)

        self.name = state["name"]
        self.user_to_idx = state["user_to_idx"]
        self.idx_to_user = state["idx_to_user"]
        self.item_to_idx = state["item_to_idx"]
        self.idx_to_item = state["idx_to_item"]
        self.category_to_idx = state["category_to_idx"]
        self.n_categories = state["n_categories"]
        self.item_categories = state["item_categories"]
        self.category_items = state["category_items"]
        self.item_popularity = state["item_popularity"]
        self.user_profiles = state["user_profiles"]
        self.user_items = state["user_items"]

        self._is_fitted = True
        logger.info(f"  {self.name} loaded from {path}")
        return self
