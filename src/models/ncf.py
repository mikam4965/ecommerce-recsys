"""Neural Collaborative Filtering (NCF) recommender.

Implements the NCF architecture from:
He et al. "Neural Collaborative Filtering" (WWW 2017)

Combines Generalized Matrix Factorization (GMF) and Multi-Layer Perceptron (MLP)
for learning user-item interactions from implicit feedback data.
"""

import pickle
import time
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
import torch
import torch.nn as nn
import torch.optim as optim
from loguru import logger
from scipy.sparse import csr_matrix
from torch.utils.data import DataLoader, Dataset

from src.models.base import BaseRecommender
from src.models.collaborative import prepare_sparse_matrix


class InteractionDataset(Dataset):
    """PyTorch dataset for implicit feedback interactions.

    Creates positive pairs from the interaction matrix and samples
    negative items (items the user has not interacted with).
    """

    def __init__(
        self,
        user_item_matrix: csr_matrix,
        n_negatives: int = 4,
    ):
        """Initialize dataset.

        Args:
            user_item_matrix: Sparse user-item interaction matrix.
            n_negatives: Number of negative samples per positive pair.
        """
        self.matrix = user_item_matrix
        self.n_users, self.n_items = user_item_matrix.shape
        self.n_negatives = n_negatives

        # Extract positive pairs
        coo = user_item_matrix.tocoo()
        self.user_indices = coo.row.astype(np.int64)
        self.item_indices = coo.col.astype(np.int64)
        self.n_positives = len(self.user_indices)

        # Build user interaction sets for fast negative sampling
        self._user_items: dict[int, set[int]] = {}
        for u, i in zip(self.user_indices, self.item_indices):
            self._user_items.setdefault(int(u), set()).add(int(i))

    def __len__(self) -> int:
        return self.n_positives * (1 + self.n_negatives)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if idx < self.n_positives:
            # Positive sample
            user = self.user_indices[idx]
            item = self.item_indices[idx]
            label = 1.0
        else:
            # Negative sample
            pos_idx = (idx - self.n_positives) % self.n_positives
            user = self.user_indices[pos_idx]
            # Sample random item that user hasn't interacted with
            user_items = self._user_items.get(int(user), set())
            item = np.random.randint(0, self.n_items)
            while item in user_items:
                item = np.random.randint(0, self.n_items)
            label = 0.0

        return (
            torch.tensor(user, dtype=torch.long),
            torch.tensor(item, dtype=torch.long),
            torch.tensor(label, dtype=torch.float32),
        )


class NCFModel(nn.Module):
    """Neural Collaborative Filtering network.

    Architecture:
    - GMF path: element-wise product of user/item embeddings
    - MLP path: concatenation of user/item embeddings through dense layers
    - NeuMF: concatenation of GMF and MLP outputs through prediction layer
    """

    def __init__(
        self,
        n_users: int,
        n_items: int,
        embedding_dim: int = 32,
        mlp_layers: list[int] | None = None,
        dropout: float = 0.2,
    ):
        """Initialize NCF model.

        Args:
            n_users: Number of users.
            n_items: Number of items.
            embedding_dim: Dimension of GMF embeddings.
            mlp_layers: Hidden layer sizes for MLP path.
            dropout: Dropout rate.
        """
        super().__init__()

        if mlp_layers is None:
            mlp_layers = [64, 32, 16]

        # GMF embeddings
        self.gmf_user_embedding = nn.Embedding(n_users, embedding_dim)
        self.gmf_item_embedding = nn.Embedding(n_items, embedding_dim)

        # MLP embeddings (separate from GMF)
        mlp_input_dim = mlp_layers[0]
        self.mlp_user_embedding = nn.Embedding(n_users, mlp_input_dim // 2)
        self.mlp_item_embedding = nn.Embedding(n_items, mlp_input_dim // 2)

        # MLP layers
        layers = []
        for i in range(len(mlp_layers) - 1):
            layers.append(nn.Linear(mlp_layers[i], mlp_layers[i + 1]))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
        self.mlp = nn.Sequential(*layers)

        # Final prediction layer (NeuMF)
        # GMF output (embedding_dim) + MLP output (last mlp_layer)
        self.predict = nn.Linear(embedding_dim + mlp_layers[-1], 1)
        self.sigmoid = nn.Sigmoid()

        # Initialize weights
        self._init_weights()

    def _init_weights(self) -> None:
        """Initialize model weights with Xavier uniform."""
        for module in self.modules():
            if isinstance(module, nn.Embedding):
                nn.init.xavier_uniform_(module.weight)
            elif isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(
        self, user_ids: torch.Tensor, item_ids: torch.Tensor
    ) -> torch.Tensor:
        """Forward pass.

        Args:
            user_ids: User index tensor.
            item_ids: Item index tensor.

        Returns:
            Predicted interaction probabilities.
        """
        # GMF path
        gmf_user = self.gmf_user_embedding(user_ids)
        gmf_item = self.gmf_item_embedding(item_ids)
        gmf_out = gmf_user * gmf_item  # element-wise product

        # MLP path
        mlp_user = self.mlp_user_embedding(user_ids)
        mlp_item = self.mlp_item_embedding(item_ids)
        mlp_input = torch.cat([mlp_user, mlp_item], dim=-1)
        mlp_out = self.mlp(mlp_input)

        # NeuMF: concatenate GMF + MLP outputs
        concat = torch.cat([gmf_out, mlp_out], dim=-1)
        output = self.sigmoid(self.predict(concat))

        return output.squeeze(-1)


class NCFRecommender(BaseRecommender):
    """Neural Collaborative Filtering recommender.

    Wraps the NCF PyTorch model with the BaseRecommender interface
    for compatibility with the evaluation pipeline.

    Attributes:
        embedding_dim: Dimension of GMF embeddings.
        mlp_layers: Hidden layer sizes for MLP path.
        learning_rate: Adam optimizer learning rate.
        batch_size: Training batch size.
        epochs: Number of training epochs.
        n_negatives: Negative samples per positive pair.
        dropout: Dropout rate.
    """

    def __init__(
        self,
        embedding_dim: int = 32,
        mlp_layers: list[int] | None = None,
        learning_rate: float = 0.001,
        batch_size: int = 1024,
        epochs: int = 10,
        n_negatives: int = 4,
        dropout: float = 0.2,
        random_state: int = 42,
    ):
        super().__init__(name="NCFRecommender")
        self.embedding_dim = embedding_dim
        self.mlp_layers = mlp_layers or [64, 32, 16]
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs
        self.n_negatives = n_negatives
        self.dropout = dropout
        self.random_state = random_state

        self.model: NCFModel | None = None
        self.device = torch.device("cpu")
        self.user_item_matrix: csr_matrix | None = None
        self.user_to_idx: dict[int, int] = {}
        self.idx_to_user: dict[int, int] = {}
        self.item_to_idx: dict[int, int] = {}
        self.idx_to_item: dict[int, int] = {}
        self.train_history: list[float] = []

    def fit(
        self,
        interactions: pl.DataFrame,
        user_col: str = "user_id",
        item_col: str = "item_id",
        **kwargs,
    ) -> "NCFRecommender":
        """Fit NCF model on interaction data.

        Args:
            interactions: DataFrame with user-item interactions.
            user_col: Column name for user IDs.
            item_col: Column name for item IDs.

        Returns:
            Self for method chaining.
        """
        torch.manual_seed(self.random_state)
        np.random.seed(self.random_state)

        logger.info(f"Fitting {self.name}...")
        logger.info(f"  embedding_dim={self.embedding_dim}, mlp_layers={self.mlp_layers}")
        logger.info(f"  lr={self.learning_rate}, epochs={self.epochs}, "
                     f"batch_size={self.batch_size}, negatives={self.n_negatives}")
        start_time = time.time()

        # Prepare sparse matrix
        (
            self.user_item_matrix,
            self.user_to_idx,
            self.idx_to_user,
            self.item_to_idx,
            self.idx_to_item,
        ) = prepare_sparse_matrix(interactions, user_col, item_col)

        n_users, n_items = self.user_item_matrix.shape
        logger.info(f"  Users: {n_users:,}, Items: {n_items:,}")

        # Create dataset and dataloader
        dataset = InteractionDataset(
            self.user_item_matrix,
            n_negatives=self.n_negatives,
        )
        dataloader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=0,
        )

        # Initialize model
        self.model = NCFModel(
            n_users=n_users,
            n_items=n_items,
            embedding_dim=self.embedding_dim,
            mlp_layers=self.mlp_layers,
            dropout=self.dropout,
        ).to(self.device)

        # Loss and optimizer (Adam as required by thesis)
        criterion = nn.BCELoss()
        optimizer = optim.Adam(
            self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=1e-5,
        )

        # Training loop
        self.model.train()
        self.train_history = []

        for epoch in range(self.epochs):
            epoch_loss = 0.0
            n_batches = 0

            for users, items, labels in dataloader:
                users = users.to(self.device)
                items = items.to(self.device)
                labels = labels.to(self.device)

                optimizer.zero_grad()
                predictions = self.model(users, items)
                loss = criterion(predictions, labels)
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
                n_batches += 1

            avg_loss = epoch_loss / max(n_batches, 1)
            self.train_history.append(avg_loss)
            logger.info(f"  Epoch {epoch + 1}/{self.epochs} - Loss: {avg_loss:.4f}")

        elapsed = time.time() - start_time
        self._is_fitted = True
        logger.info(f"Fitted {self.name} in {elapsed:.1f}s")

        return self

    @torch.no_grad()
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
            return []

        user_idx = self.user_to_idx[user_id]
        n_items = len(self.item_to_idx)

        self.model.eval()

        # Score all items for this user
        user_tensor = torch.full((n_items,), user_idx, dtype=torch.long, device=self.device)
        item_tensor = torch.arange(n_items, dtype=torch.long, device=self.device)

        scores = self.model(user_tensor, item_tensor).cpu().numpy()

        # Filter already liked items
        if filter_already_liked:
            liked = set(self.user_item_matrix[user_idx].indices)
            for item_idx in liked:
                scores[item_idx] = -1.0

        # Get top-N
        top_indices = np.argsort(scores)[::-1][:n]

        return [self.idx_to_item[int(idx)] for idx in top_indices]

    @torch.no_grad()
    def recommend_with_scores(
        self,
        user_id: int,
        n: int = 10,
        filter_already_liked: bool = True,
    ) -> list[tuple[int, float]]:
        """Generate recommendations with scores.

        Args:
            user_id: User identifier.
            n: Number of recommendations.
            filter_already_liked: Whether to filter already interacted items.

        Returns:
            List of (item_id, score) tuples.
        """
        self._check_is_fitted()

        if user_id not in self.user_to_idx:
            return []

        user_idx = self.user_to_idx[user_id]
        n_items = len(self.item_to_idx)

        self.model.eval()

        user_tensor = torch.full((n_items,), user_idx, dtype=torch.long, device=self.device)
        item_tensor = torch.arange(n_items, dtype=torch.long, device=self.device)

        scores = self.model(user_tensor, item_tensor).cpu().numpy()

        if filter_already_liked:
            liked = set(self.user_item_matrix[user_idx].indices)
            for item_idx in liked:
                scores[item_idx] = -1.0

        top_indices = np.argsort(scores)[::-1][:n]

        return [
            (self.idx_to_item[int(idx)], float(scores[idx]))
            for idx in top_indices
        ]

    def save(self, path: str | Path) -> None:
        """Save model to disk."""
        path = Path(path)
        state = {
            "model_state_dict": self.model.state_dict() if self.model else None,
            "config": {
                "embedding_dim": self.embedding_dim,
                "mlp_layers": self.mlp_layers,
                "learning_rate": self.learning_rate,
                "epochs": self.epochs,
                "n_negatives": self.n_negatives,
                "dropout": self.dropout,
            },
            "mappings": {
                "user_to_idx": self.user_to_idx,
                "idx_to_user": self.idx_to_user,
                "item_to_idx": self.item_to_idx,
                "idx_to_item": self.idx_to_item,
            },
            "n_users": len(self.user_to_idx),
            "n_items": len(self.item_to_idx),
            "train_history": self.train_history,
            "user_item_matrix": self.user_item_matrix,
        }
        with open(path, "wb") as f:
            pickle.dump(state, f)
        logger.info(f"Saved {self.name} to {path}")

    def load(self, path: str | Path) -> "NCFRecommender":
        """Load model from disk."""
        path = Path(path)
        with open(path, "rb") as f:
            state = pickle.load(f)

        self.user_to_idx = state["mappings"]["user_to_idx"]
        self.idx_to_user = state["mappings"]["idx_to_user"]
        self.item_to_idx = state["mappings"]["item_to_idx"]
        self.idx_to_item = state["mappings"]["idx_to_item"]
        self.train_history = state.get("train_history", [])
        self.user_item_matrix = state.get("user_item_matrix")

        config = state["config"]
        self.model = NCFModel(
            n_users=state["n_users"],
            n_items=state["n_items"],
            embedding_dim=config["embedding_dim"],
            mlp_layers=config["mlp_layers"],
            dropout=config["dropout"],
        ).to(self.device)

        if state["model_state_dict"]:
            self.model.load_state_dict(state["model_state_dict"])

        self._is_fitted = True
        logger.info(f"Loaded {self.name} from {path}")
        return self
