"""Two-stage recommendation pipeline: retrieval + ranking."""

import pickle
import time
from pathlib import Path
from typing import Any

import polars as pl
from loguru import logger

from src.models.base import BaseRecommender
from src.models.hybrid import HybridRecommender
from src.models.ranker import CatBoostRanker


class TwoStageRecommender(BaseRecommender):
    """Two-stage recommender: retrieval (HybridRecommender) + ranking (CatBoostRanker).

    Stage 1: Generate n_candidates (default 100) using HybridRecommender
    Stage 2: Re-rank candidates using CatBoostRanker to select top-n

    This approach allows using:
    - Fast ALS-based retrieval for candidate generation
    - Gradient boosting for precise ranking with rich features
    """

    def __init__(
        self,
        # Retriever parameters
        factors: int = 64,
        regularization: float = 0.01,
        iterations: int = 15,
        cf_weight: float = 0.7,
        feature_weight: float = 0.3,
        # Ranker parameters
        ranker_iterations: int = 500,
        ranker_learning_rate: float = 0.03,
        ranker_depth: int = 6,
        ranker_loss: str = "YetiRank",
        # Pipeline parameters
        n_candidates: int = 100,
        n_negatives: int = 4,
        random_state: int = 42,
    ):
        """Initialize two-stage recommender.

        Args:
            factors: Number of latent factors for retriever.
            regularization: ALS regularization parameter.
            iterations: ALS training iterations.
            cf_weight: Weight for CF scores in retriever.
            feature_weight: Weight for feature scores in retriever.
            ranker_iterations: CatBoost iterations.
            ranker_learning_rate: CatBoost learning rate.
            ranker_depth: CatBoost tree depth.
            ranker_loss: CatBoost loss function.
            n_candidates: Number of candidates from Stage 1.
            n_negatives: Negative samples per positive for ranker training.
            random_state: Random seed.
        """
        super().__init__(name="TwoStageRecommender")

        # Initialize retriever
        self.retriever = HybridRecommender(
            factors=factors,
            regularization=regularization,
            iterations=iterations,
            cf_weight=cf_weight,
            feature_weight=feature_weight,
            random_state=random_state,
        )

        # Initialize ranker
        self.ranker = CatBoostRanker(
            iterations=ranker_iterations,
            learning_rate=ranker_learning_rate,
            depth=ranker_depth,
            loss_function=ranker_loss,
            random_state=random_state,
        )

        # Store parameters
        self.n_candidates = n_candidates
        self.n_negatives = n_negatives
        self.random_state = random_state

        # Reference data
        self.reference_timestamp: int | None = None

    @property
    def user_to_idx(self) -> dict[int, int]:
        """Expose retriever's user mapping for evaluation compatibility."""
        return self.retriever.user_to_idx

    @property
    def idx_to_user(self) -> dict[int, int]:
        """Expose retriever's reverse user mapping."""
        return self.retriever.idx_to_user

    @property
    def item_to_idx(self) -> dict[int, int]:
        """Expose retriever's item mapping for evaluation compatibility."""
        return self.retriever.item_to_idx

    @property
    def idx_to_item(self) -> dict[int, int]:
        """Expose retriever's reverse item mapping."""
        return self.retriever.idx_to_item

    def fit(
        self,
        events: pl.DataFrame,
        rfm_data: pl.DataFrame | None = None,
        use_item_features: bool = True,
        use_user_features: bool = True,
        **kwargs,
    ) -> "TwoStageRecommender":
        """Fit both stages of the pipeline.

        Args:
            events: Training events DataFrame.
            rfm_data: RFM segmentation data (required for full features).
            use_item_features: Whether to use item features in retriever.
            use_user_features: Whether to use user features in retriever.

        Returns:
            Self for method chaining.
        """
        logger.info(f"Fitting {self.name}...")
        start_time = time.time()

        # Store reference timestamp for ranking features
        self.reference_timestamp = events["timestamp"].max()

        # Stage 1: Fit retriever
        logger.info("=" * 50)
        logger.info("Stage 1: Training retriever (HybridRecommender)...")
        logger.info("=" * 50)

        self.retriever.fit(
            events,
            rfm_data=rfm_data,
            use_item_features=use_item_features,
            use_user_features=use_user_features and rfm_data is not None,
        )

        # Stage 2: Fit ranker
        logger.info("=" * 50)
        logger.info("Stage 2: Training ranker (CatBoostRanker)...")
        logger.info("=" * 50)

        self.ranker.fit(
            events,
            retriever=self.retriever,
            rfm_data=rfm_data,
            n_negatives=self.n_negatives,
        )

        elapsed = time.time() - start_time
        self._is_fitted = True

        logger.info("=" * 50)
        logger.info(f"Fitted {self.name} in {elapsed:.2f}s")
        logger.info("=" * 50)

        return self

    def recommend(
        self,
        user_id: int,
        n: int = 10,
        filter_already_liked: bool = True,
    ) -> list[int]:
        """Generate recommendations using two-stage pipeline.

        Args:
            user_id: User identifier.
            n: Number of final recommendations.
            filter_already_liked: Whether to filter interacted items.

        Returns:
            List of recommended item IDs.
        """
        self._check_is_fitted()

        # Stage 1: Get candidates with scores
        candidates = self.retriever.recommend_with_scores(
            user_id,
            n=self.n_candidates,
            filter_already_liked=filter_already_liked,
        )

        if not candidates:
            # Fall back to cold start
            return self.retriever.recommend_cold_start({}, n)

        # Stage 2: Re-rank
        recommendations = self.ranker.rank(
            user_id,
            candidates,
            n=n,
            reference_timestamp=self.reference_timestamp,
        )

        return recommendations

    def recommend_cold_start(
        self,
        user_features_dict: dict[str, Any],
        n: int = 10,
    ) -> list[int]:
        """Cold start recommendations (falls back to retriever only).

        For new users without history, we can't use the ranker effectively
        since it depends on user-specific features.

        Args:
            user_features_dict: Dictionary with user features.
            n: Number of recommendations.

        Returns:
            List of recommended item IDs.
        """
        self._check_is_fitted()
        return self.retriever.recommend_cold_start(user_features_dict, n)

    def recommend_for_new_user(
        self,
        session_items: list[int],
        n: int = 10,
    ) -> list[int]:
        """Recommendations for new user based on session items.

        Falls back to retriever as ranker needs user history.

        Args:
            session_items: List of item IDs from current session.
            n: Number of recommendations.

        Returns:
            List of recommended item IDs.
        """
        self._check_is_fitted()
        return self.retriever.recommend_for_new_user(session_items, n)

    def similar_items(
        self,
        item_id: int,
        n: int = 10,
    ) -> list[int]:
        """Find similar items based on item embeddings.

        Delegates to retriever.

        Args:
            item_id: Item identifier.
            n: Number of similar items.

        Returns:
            List of similar item IDs.
        """
        self._check_is_fitted()
        return self.retriever.similar_items(item_id, n)

    def save(self, path: str | Path) -> None:
        """Save pipeline to disk.

        Args:
            path: Path to save the model.
        """
        self._check_is_fitted()
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        # Save retriever and ranker separately
        retriever_path = path.with_suffix(".retriever.pkl")
        ranker_path = path.with_suffix(".ranker.pkl")

        self.retriever.save(retriever_path)
        self.ranker.save(ranker_path)

        # Save pipeline metadata
        data = {
            "n_candidates": self.n_candidates,
            "n_negatives": self.n_negatives,
            "random_state": self.random_state,
            "reference_timestamp": self.reference_timestamp,
            "retriever_path": str(retriever_path),
            "ranker_path": str(ranker_path),
        }

        with open(path, "wb") as f:
            pickle.dump(data, f)

        logger.info(f"Saved {self.name} to {path}")

    def load(self, path: str | Path) -> "TwoStageRecommender":
        """Load pipeline from disk.

        Args:
            path: Path to load the model from.

        Returns:
            Self for method chaining.
        """
        path = Path(path)

        with open(path, "rb") as f:
            data = pickle.load(f)

        self.n_candidates = data["n_candidates"]
        self.n_negatives = data["n_negatives"]
        self.random_state = data["random_state"]
        self.reference_timestamp = data["reference_timestamp"]

        # Load retriever and ranker
        retriever_path = Path(data["retriever_path"])
        ranker_path = Path(data["ranker_path"])

        self.retriever.load(retriever_path)
        self.ranker.load(ranker_path)

        self._is_fitted = True

        logger.info(f"Loaded {self.name} from {path}")

        return self

    def __repr__(self) -> str:
        return (
            f"TwoStageRecommender("
            f"n_candidates={self.n_candidates}, "
            f"retriever={self.retriever}, "
            f"ranker={self.ranker}, "
            f"fitted={self._is_fitted})"
        )
