"""Collaborative filtering recommenders using implicit library."""

import pickle
import time
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
from loguru import logger
from scipy.sparse import csr_matrix

from implicit.als import AlternatingLeastSquares
from implicit.bpr import BayesianPersonalizedRanking

from src.models.base import BaseRecommender


def prepare_sparse_matrix(
    interactions: pl.DataFrame,
    user_col: str = "user_id",
    item_col: str = "item_id",
) -> tuple[csr_matrix, dict[int, int], dict[int, int], dict[int, int], dict[int, int]]:
    """Create sparse user-item matrix from interactions.

    Args:
        interactions: DataFrame with user-item interactions.
        user_col: Column name for user IDs.
        item_col: Column name for item IDs.

    Returns:
        Tuple of:
        - CSR sparse matrix (users x items)
        - user_id to matrix index mapping
        - matrix index to user_id mapping
        - item_id to matrix index mapping
        - matrix index to item_id mapping
    """
    logger.info("Preparing sparse matrix...")

    # Create mappings
    unique_users = interactions[user_col].unique().sort().to_list()
    unique_items = interactions[item_col].unique().sort().to_list()

    user_to_idx = {user: idx for idx, user in enumerate(unique_users)}
    idx_to_user = {idx: user for user, idx in user_to_idx.items()}
    item_to_idx = {item: idx for idx, item in enumerate(unique_items)}
    idx_to_item = {idx: item for item, idx in item_to_idx.items()}

    # Count interactions per user-item pair
    interaction_counts = (
        interactions.group_by([user_col, item_col])
        .agg(pl.len().alias("count"))
    )

    # Create sparse matrix
    row_indices = [user_to_idx[u] for u in interaction_counts[user_col].to_list()]
    col_indices = [item_to_idx[i] for i in interaction_counts[item_col].to_list()]
    values = interaction_counts["count"].to_list()

    matrix = csr_matrix(
        (values, (row_indices, col_indices)),
        shape=(len(unique_users), len(unique_items)),
        dtype=np.float32,
    )

    logger.info(f"Matrix shape: {matrix.shape}")
    logger.info(f"Non-zero elements: {matrix.nnz:,}")
    logger.info(f"Sparsity: {1 - matrix.nnz / (matrix.shape[0] * matrix.shape[1]):.4%}")

    return matrix, user_to_idx, idx_to_user, item_to_idx, idx_to_item


class ALSRecommender(BaseRecommender):
    """Alternating Least Squares recommender using implicit library."""

    def __init__(
        self,
        factors: int = 64,
        regularization: float = 0.01,
        iterations: int = 15,
        random_state: int = 42,
    ):
        """Initialize ALS recommender.

        Args:
            factors: Number of latent factors.
            regularization: Regularization parameter.
            iterations: Number of ALS iterations.
            random_state: Random seed for reproducibility.
        """
        super().__init__(name="ALSRecommender")
        self.factors = factors
        self.regularization = regularization
        self.iterations = iterations
        self.random_state = random_state

        self.model: AlternatingLeastSquares | None = None
        self.user_item_matrix: csr_matrix | None = None
        self.user_to_idx: dict[int, int] = {}
        self.idx_to_user: dict[int, int] = {}
        self.item_to_idx: dict[int, int] = {}
        self.idx_to_item: dict[int, int] = {}

    def fit(
        self,
        interactions: pl.DataFrame,
        user_col: str = "user_id",
        item_col: str = "item_id",
        **kwargs,
    ) -> "ALSRecommender":
        """Fit ALS model on interaction data.

        Args:
            interactions: DataFrame with user-item interactions.
            user_col: Column name for user IDs.
            item_col: Column name for item IDs.

        Returns:
            Self for method chaining.
        """
        logger.info(f"Fitting {self.name}...")
        logger.info(f"Parameters: factors={self.factors}, reg={self.regularization}, iterations={self.iterations}")
        start_time = time.time()

        # Prepare sparse matrix
        (
            self.user_item_matrix,
            self.user_to_idx,
            self.idx_to_user,
            self.item_to_idx,
            self.idx_to_item,
        ) = prepare_sparse_matrix(interactions, user_col, item_col)

        # Initialize and train model
        self.model = AlternatingLeastSquares(
            factors=self.factors,
            regularization=self.regularization,
            iterations=self.iterations,
            random_state=self.random_state,
        )

        # implicit expects user-item matrix
        self.model.fit(self.user_item_matrix)

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
        """Generate recommendations for a user.

        Args:
            user_id: User identifier.
            n: Number of recommendations.
            filter_already_liked: Whether to filter already interacted items.

        Returns:
            List of recommended item IDs.
        """
        self._check_is_fitted()

        if user_id not in self.user_to_idx:
            logger.warning(f"User {user_id} not in training data, returning empty list")
            return []

        user_idx = self.user_to_idx[user_id]

        # Get recommendations
        item_indices, scores = self.model.recommend(
            user_idx,
            self.user_item_matrix[user_idx],
            N=n,
            filter_already_liked_items=filter_already_liked,
        )

        # Convert indices back to item IDs
        recommendations = [self.idx_to_item[idx] for idx in item_indices]

        return recommendations

    def save(self, path: str | Path) -> None:
        """Save model to disk."""
        self._check_is_fitted()
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        data = {
            "model": self.model,
            "user_item_matrix": self.user_item_matrix,
            "user_to_idx": self.user_to_idx,
            "idx_to_user": self.idx_to_user,
            "item_to_idx": self.item_to_idx,
            "idx_to_item": self.idx_to_item,
            "factors": self.factors,
            "regularization": self.regularization,
            "iterations": self.iterations,
        }

        with open(path, "wb") as f:
            pickle.dump(data, f)

        logger.info(f"Saved {self.name} to {path}")

    def load(self, path: str | Path) -> "ALSRecommender":
        """Load model from disk."""
        path = Path(path)

        with open(path, "rb") as f:
            data = pickle.load(f)

        self.model = data["model"]
        self.user_item_matrix = data["user_item_matrix"]
        self.user_to_idx = data["user_to_idx"]
        self.idx_to_user = data["idx_to_user"]
        self.item_to_idx = data["item_to_idx"]
        self.idx_to_item = data["idx_to_item"]
        self.factors = data["factors"]
        self.regularization = data["regularization"]
        self.iterations = data["iterations"]
        self._is_fitted = True

        logger.info(f"Loaded {self.name} from {path}")

        return self


class BPRRecommender(BaseRecommender):
    """Bayesian Personalized Ranking recommender using implicit library."""

    def __init__(
        self,
        factors: int = 64,
        learning_rate: float = 0.01,
        regularization: float = 0.01,
        iterations: int = 100,
        random_state: int = 42,
    ):
        """Initialize BPR recommender.

        Args:
            factors: Number of latent factors.
            learning_rate: Learning rate for SGD.
            regularization: Regularization parameter.
            iterations: Number of training iterations.
            random_state: Random seed for reproducibility.
        """
        super().__init__(name="BPRRecommender")
        self.factors = factors
        self.learning_rate = learning_rate
        self.regularization = regularization
        self.iterations = iterations
        self.random_state = random_state

        self.model: BayesianPersonalizedRanking | None = None
        self.user_item_matrix: csr_matrix | None = None
        self.user_to_idx: dict[int, int] = {}
        self.idx_to_user: dict[int, int] = {}
        self.item_to_idx: dict[int, int] = {}
        self.idx_to_item: dict[int, int] = {}

    def fit(
        self,
        interactions: pl.DataFrame,
        user_col: str = "user_id",
        item_col: str = "item_id",
        **kwargs,
    ) -> "BPRRecommender":
        """Fit BPR model on interaction data.

        Args:
            interactions: DataFrame with user-item interactions.
            user_col: Column name for user IDs.
            item_col: Column name for item IDs.

        Returns:
            Self for method chaining.
        """
        logger.info(f"Fitting {self.name}...")
        logger.info(f"Parameters: factors={self.factors}, lr={self.learning_rate}, iterations={self.iterations}")
        start_time = time.time()

        # Prepare sparse matrix
        (
            self.user_item_matrix,
            self.user_to_idx,
            self.idx_to_user,
            self.item_to_idx,
            self.idx_to_item,
        ) = prepare_sparse_matrix(interactions, user_col, item_col)

        # Initialize and train model
        self.model = BayesianPersonalizedRanking(
            factors=self.factors,
            learning_rate=self.learning_rate,
            regularization=self.regularization,
            iterations=self.iterations,
            random_state=self.random_state,
        )

        # implicit expects user-item matrix
        self.model.fit(self.user_item_matrix)

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
        """Generate recommendations for a user."""
        self._check_is_fitted()

        if user_id not in self.user_to_idx:
            logger.warning(f"User {user_id} not in training data, returning empty list")
            return []

        user_idx = self.user_to_idx[user_id]

        item_indices, scores = self.model.recommend(
            user_idx,
            self.user_item_matrix[user_idx],
            N=n,
            filter_already_liked_items=filter_already_liked,
        )

        recommendations = [self.idx_to_item[idx] for idx in item_indices]

        return recommendations

    def save(self, path: str | Path) -> None:
        """Save model to disk."""
        self._check_is_fitted()
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        data = {
            "model": self.model,
            "user_item_matrix": self.user_item_matrix,
            "user_to_idx": self.user_to_idx,
            "idx_to_user": self.idx_to_user,
            "item_to_idx": self.item_to_idx,
            "idx_to_item": self.idx_to_item,
            "factors": self.factors,
            "learning_rate": self.learning_rate,
            "regularization": self.regularization,
            "iterations": self.iterations,
        }

        with open(path, "wb") as f:
            pickle.dump(data, f)

        logger.info(f"Saved {self.name} to {path}")

    def load(self, path: str | Path) -> "BPRRecommender":
        """Load model from disk."""
        path = Path(path)

        with open(path, "rb") as f:
            data = pickle.load(f)

        self.model = data["model"]
        self.user_item_matrix = data["user_item_matrix"]
        self.user_to_idx = data["user_to_idx"]
        self.idx_to_user = data["idx_to_user"]
        self.item_to_idx = data["item_to_idx"]
        self.idx_to_item = data["idx_to_item"]
        self.factors = data["factors"]
        self.learning_rate = data["learning_rate"]
        self.regularization = data["regularization"]
        self.iterations = data["iterations"]
        self._is_fitted = True

        logger.info(f"Loaded {self.name} from {path}")

        return self
