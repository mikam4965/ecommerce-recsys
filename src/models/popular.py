"""Most Popular recommender."""

import pickle
import time
from pathlib import Path

import numpy as np
import polars as pl
from loguru import logger

from src.models.base import BaseRecommender


class MostPopularRecommender(BaseRecommender):
    """Recommends most popular items to all users.

    Popularity is measured by the number of interactions.
    """

    def __init__(self):
        """Initialize MostPopularRecommender."""
        super().__init__(name="MostPopularRecommender")
        self.popular_items: list[int] = []
        self.user_items: dict[int, set[int]] = {}

    def fit(
        self,
        interactions: pl.DataFrame,
        user_col: str = "user_id",
        item_col: str = "item_id",
        **kwargs,
    ) -> "MostPopularRecommender":
        """Fit the model by calculating item popularity.

        Args:
            interactions: DataFrame with user-item interactions.
            user_col: Column name for user IDs.
            item_col: Column name for item IDs.

        Returns:
            Self for method chaining.
        """
        logger.info(f"Fitting {self.name}...")
        start_time = time.time()

        # Calculate item popularity (interaction count)
        popularity = (
            interactions.group_by(item_col)
            .agg(pl.len().alias("count"))
            .sort("count", descending=True)
        )

        self.popular_items = popularity[item_col].to_list()

        # Store user-item interactions for filtering
        user_items_df = interactions.group_by(user_col).agg(
            pl.col(item_col).alias("items")
        )
        self.user_items = {
            row[user_col]: set(row["items"])
            for row in user_items_df.iter_rows(named=True)
        }

        elapsed = time.time() - start_time
        self._is_fitted = True

        logger.info(f"Fitted {self.name} in {elapsed:.2f}s")
        logger.info(f"Total items: {len(self.popular_items):,}")
        logger.info(f"Total users: {len(self.user_items):,}")

        return self

    def recommend(
        self,
        user_id: int,
        n: int = 10,
        filter_already_liked: bool = True,
    ) -> list[int]:
        """Recommend most popular items.

        Args:
            user_id: User identifier.
            n: Number of recommendations.
            filter_already_liked: Whether to filter items user already interacted with.

        Returns:
            List of recommended item IDs.
        """
        self._check_is_fitted()

        if filter_already_liked and user_id in self.user_items:
            user_liked = self.user_items[user_id]
            recommendations = [
                item for item in self.popular_items
                if item not in user_liked
            ][:n]
        else:
            recommendations = self.popular_items[:n]

        return recommendations

    def save(self, path: str | Path) -> None:
        """Save model to disk.

        Args:
            path: Path to save the model.
        """
        self._check_is_fitted()
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        data = {
            "popular_items": self.popular_items,
            "user_items": self.user_items,
        }

        with open(path, "wb") as f:
            pickle.dump(data, f)

        logger.info(f"Saved {self.name} to {path}")

    def load(self, path: str | Path) -> "MostPopularRecommender":
        """Load model from disk.

        Args:
            path: Path to load the model from.

        Returns:
            Self for method chaining.
        """
        path = Path(path)

        with open(path, "rb") as f:
            data = pickle.load(f)

        self.popular_items = data["popular_items"]
        self.user_items = data["user_items"]
        self._is_fitted = True

        logger.info(f"Loaded {self.name} from {path}")

        return self
