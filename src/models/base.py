"""Base recommender class."""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import numpy as np


class BaseRecommender(ABC):
    """Abstract base class for all recommenders."""

    def __init__(self, name: str = "BaseRecommender"):
        """Initialize recommender.

        Args:
            name: Model name for logging.
        """
        self.name = name
        self._is_fitted = False

    @abstractmethod
    def fit(self, interactions: Any, **kwargs) -> "BaseRecommender":
        """Fit the model on interaction data.

        Args:
            interactions: Interaction data (format depends on implementation).
            **kwargs: Additional parameters.

        Returns:
            Self for method chaining.
        """
        pass

    @abstractmethod
    def recommend(
        self,
        user_id: int,
        n: int = 10,
        filter_already_liked: bool = True,
    ) -> list[int]:
        """Generate recommendations for a user.

        Args:
            user_id: User identifier.
            n: Number of recommendations to generate.
            filter_already_liked: Whether to filter items user already interacted with.

        Returns:
            List of recommended item IDs.
        """
        pass

    def recommend_batch(
        self,
        user_ids: list[int],
        n: int = 10,
        filter_already_liked: bool = True,
    ) -> dict[int, list[int]]:
        """Generate recommendations for multiple users.

        Args:
            user_ids: List of user identifiers.
            n: Number of recommendations per user.
            filter_already_liked: Whether to filter already liked items.

        Returns:
            Dictionary mapping user_id to list of recommended item IDs.
        """
        return {
            user_id: self.recommend(user_id, n, filter_already_liked)
            for user_id in user_ids
        }

    @abstractmethod
    def save(self, path: str | Path) -> None:
        """Save model to disk.

        Args:
            path: Path to save the model.
        """
        pass

    @abstractmethod
    def load(self, path: str | Path) -> "BaseRecommender":
        """Load model from disk.

        Args:
            path: Path to load the model from.

        Returns:
            Self for method chaining.
        """
        pass

    @property
    def is_fitted(self) -> bool:
        """Check if model is fitted."""
        return self._is_fitted

    def _check_is_fitted(self) -> None:
        """Raise error if model is not fitted."""
        if not self._is_fitted:
            raise RuntimeError(f"{self.name} is not fitted. Call fit() first.")

    def __repr__(self) -> str:
        return f"{self.name}(fitted={self._is_fitted})"
