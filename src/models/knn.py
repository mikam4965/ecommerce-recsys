"""Item-based K-Nearest Neighbors Recommender.

Uses cosine similarity between items based on user-item interaction patterns.
Classic memory-based collaborative filtering approach.

Algorithm:
    1. Build sparse user-item interaction matrix
    2. Compute item-item cosine similarity
    3. For each user, score items by aggregated similarity to their history
    4. Return top-N items sorted by score
"""

import pickle
from pathlib import Path

import numpy as np
import polars as pl
from loguru import logger
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import cosine_similarity

from src.models.base import BaseRecommender
from src.models.collaborative import prepare_sparse_matrix


class ItemKNNRecommender(BaseRecommender):
    """Item-based k-NN recommender using cosine similarity.

    Computes item-item similarity from the user-item interaction matrix,
    then recommends items most similar to the user's interaction history.
    """

    def __init__(
        self,
        k: int = 50,
        name: str = "ItemKNN",
    ):
        """Initialize ItemKNNRecommender.

        Args:
            k: Number of nearest neighbors per item.
            name: Model name for logging.
        """
        super().__init__(name=name)
        self.k = k

        self.user_item_matrix: csr_matrix | None = None
        self.user_to_idx: dict[int, int] = {}
        self.idx_to_user: dict[int, int] = {}
        self.item_to_idx: dict[int, int] = {}
        self.idx_to_item: dict[int, int] = {}

        # Item-item similarity matrix (sparse, only top-k neighbors kept)
        self.item_similarity: np.ndarray | None = None

    def fit(
        self,
        interactions: pl.DataFrame,
        user_col: str = "user_id",
        item_col: str = "item_id",
        **kwargs,
    ) -> "ItemKNNRecommender":
        """Fit k-NN model by computing item-item similarity.

        Args:
            interactions: DataFrame with user-item interactions.
            user_col: Column name for user IDs.
            item_col: Column name for item IDs.

        Returns:
            Self for method chaining.
        """
        logger.info(f"Fitting {self.name} (k={self.k})...")

        # Build sparse matrix
        (
            self.user_item_matrix,
            self.user_to_idx,
            self.idx_to_user,
            self.item_to_idx,
            self.idx_to_item,
        ) = prepare_sparse_matrix(interactions, user_col, item_col)

        n_items = self.user_item_matrix.shape[1]
        logger.info(f"Computing item-item cosine similarity for {n_items:,} items...")

        # Compute item-item cosine similarity
        # Transpose to get item-user matrix, then compute cosine
        item_user_matrix = self.user_item_matrix.T.tocsr()

        # Process in chunks to avoid memory issues
        chunk_size = 1000
        n_chunks = (n_items + chunk_size - 1) // chunk_size

        # Keep only top-k similarities per item (memory efficient)
        # Store as dense matrix of shape (n_items, k) for indices and values
        self.sim_indices = np.zeros((n_items, self.k), dtype=np.int32)
        self.sim_values = np.zeros((n_items, self.k), dtype=np.float32)

        for chunk_idx in range(n_chunks):
            start = chunk_idx * chunk_size
            end = min(start + chunk_size, n_items)

            # Compute similarity for this chunk of items
            chunk_sim = cosine_similarity(
                item_user_matrix[start:end],
                item_user_matrix,
            )

            # For each item in chunk, find top-k neighbors (excluding self)
            for i in range(end - start):
                item_idx = start + i
                sims = chunk_sim[i]
                sims[item_idx] = -1  # exclude self

                # Get top-k indices
                top_k_idx = np.argpartition(sims, -self.k)[-self.k:]
                top_k_idx = top_k_idx[np.argsort(sims[top_k_idx])[::-1]]

                self.sim_indices[item_idx] = top_k_idx
                self.sim_values[item_idx] = sims[top_k_idx]

            if (chunk_idx + 1) % 10 == 0:
                logger.info(f"  Processed {end:,}/{n_items:,} items")

        self._is_fitted = True
        logger.info(f"{self.name} fitted: {n_items:,} items, k={self.k} neighbors")
        return self

    def recommend(
        self,
        user_id: int,
        n: int = 10,
        filter_already_liked: bool = True,
    ) -> list[int]:
        """Recommend items for a user based on item similarity.

        Args:
            user_id: User identifier.
            n: Number of recommendations.
            filter_already_liked: Whether to filter already interacted items.

        Returns:
            List of recommended item IDs.
        """
        self._check_is_fitted()

        if user_id not in self.user_to_idx:
            return []

        user_idx = self.user_to_idx[user_id]

        # Get user's interacted items
        user_row = self.user_item_matrix[user_idx]
        interacted_indices = user_row.indices
        interacted_values = user_row.data

        if len(interacted_indices) == 0:
            return []

        # Score all items: aggregate similarity to user's history
        n_items = self.user_item_matrix.shape[1]
        scores = np.zeros(n_items, dtype=np.float64)

        for item_idx, interaction_weight in zip(interacted_indices, interacted_values):
            # Add similarity scores from this item's neighbors
            neighbor_indices = self.sim_indices[item_idx]
            neighbor_sims = self.sim_values[item_idx]

            for nb_idx, nb_sim in zip(neighbor_indices, neighbor_sims):
                if nb_sim > 0:
                    scores[nb_idx] += nb_sim * interaction_weight

        # Filter already liked items
        if filter_already_liked:
            scores[interacted_indices] = -1

        # Get top-n items
        top_indices = np.argpartition(scores, -n)[-n:]
        top_indices = top_indices[np.argsort(scores[top_indices])[::-1]]

        # Filter items with zero score
        result = []
        for idx in top_indices:
            if scores[idx] > 0:
                result.append(self.idx_to_item[idx])

        return result[:n]

    def save(self, path: str | Path) -> None:
        """Save model to disk."""
        self._check_is_fitted()
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        state = {
            "name": self.name,
            "k": self.k,
            "user_item_matrix": self.user_item_matrix,
            "user_to_idx": self.user_to_idx,
            "idx_to_user": self.idx_to_user,
            "item_to_idx": self.item_to_idx,
            "idx_to_item": self.idx_to_item,
            "sim_indices": self.sim_indices,
            "sim_values": self.sim_values,
        }

        with open(path, "wb") as f:
            pickle.dump(state, f, protocol=pickle.HIGHEST_PROTOCOL)

        logger.info(f"  {self.name} saved to {path}")

    def load(self, path: str | Path) -> "ItemKNNRecommender":
        """Load model from disk."""
        path = Path(path)

        with open(path, "rb") as f:
            state = pickle.load(f)

        self.name = state["name"]
        self.k = state["k"]
        self.user_item_matrix = state["user_item_matrix"]
        self.user_to_idx = state["user_to_idx"]
        self.idx_to_user = state["idx_to_user"]
        self.item_to_idx = state["item_to_idx"]
        self.idx_to_item = state["idx_to_item"]
        self.sim_indices = state["sim_indices"]
        self.sim_values = state["sim_values"]

        self._is_fitted = True
        logger.info(f"  {self.name} loaded from {path}")
        return self
