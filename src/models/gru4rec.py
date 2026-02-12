"""GRU4Rec — Session-based RNN recommender.

Implements the GRU4Rec architecture from:
Hidasi et al. "Session-based Recommendations with Recurrent Neural Networks" (ICLR 2016)

Uses GRU to model sequential item interactions within sessions
and predict the next item a user is likely to interact with.

Training uses BPR loss with negative sampling for efficient CPU training.
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
from torch.nn.utils.rnn import pack_padded_sequence
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from src.models.base import BaseRecommender
from src.models.collaborative import prepare_sparse_matrix


class SessionDataset(Dataset):
    """PyTorch dataset for session-based interactions.

    Groups interactions by session_id, sorts by timestamp,
    and creates (input_sequence, target_item) training pairs.
    """

    def __init__(
        self,
        sessions: list[list[int]],
        max_seq_len: int = 50,
    ):
        """Initialize dataset.

        Args:
            sessions: List of session item sequences (already mapped to indices).
            max_seq_len: Maximum sequence length (truncate from left).
        """
        self.max_seq_len = max_seq_len
        self.samples: list[tuple[list[int], int]] = []

        for session in sessions:
            if len(session) < 2:
                continue
            # Create (prefix, next_item) pairs for each position
            for i in range(1, len(session)):
                prefix = session[max(0, i - max_seq_len):i]
                target = session[i]
                self.samples.append((prefix, target))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int, int]:
        prefix, target = self.samples[idx]
        seq_tensor = torch.tensor(prefix, dtype=torch.long)
        length = len(prefix)
        return seq_tensor, target, length


def collate_sessions(
    batch: list[tuple[torch.Tensor, int, int]],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Collate variable-length sessions with padding.

    Args:
        batch: List of (sequence, target, length) tuples.

    Returns:
        Padded sequences, targets, and lengths tensors.
    """
    sequences, targets, lengths = zip(*batch)

    # Pad sequences to max length in batch
    max_len = max(lengths)
    padded = torch.zeros(len(batch), max_len, dtype=torch.long)
    for i, seq in enumerate(sequences):
        padded[i, :len(seq)] = seq

    targets = torch.tensor(targets, dtype=torch.long)
    lengths = torch.tensor(lengths, dtype=torch.long)

    return padded, targets, lengths


class GRU4RecModel(nn.Module):
    """GRU-based session recommender network.

    Architecture:
    - Item embedding layer (shared for input and scoring)
    - Dropout on embeddings
    - GRU layer(s)
    - Linear projection from hidden to embedding space
    - BPR scoring via dot product with item embeddings
    """

    def __init__(
        self,
        n_items: int,
        embedding_dim: int = 64,
        hidden_dim: int = 128,
        num_layers: int = 1,
        dropout: float = 0.2,
    ):
        """Initialize GRU4Rec model.

        Args:
            n_items: Number of items (including padding index 0).
            embedding_dim: Item embedding dimension.
            hidden_dim: GRU hidden state dimension.
            num_layers: Number of GRU layers.
            dropout: Dropout rate.
        """
        super().__init__()

        self.n_items = n_items
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.item_embedding = nn.Embedding(
            n_items, embedding_dim, padding_idx=0
        )
        self.emb_dropout = nn.Dropout(dropout)

        self.gru = nn.GRU(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        # Project GRU hidden state to embedding space for dot-product scoring
        self.output_proj = nn.Linear(hidden_dim, embedding_dim)

        self._init_weights()

    def _init_weights(self) -> None:
        """Initialize weights."""
        nn.init.xavier_uniform_(self.item_embedding.weight[1:])
        nn.init.xavier_uniform_(self.output_proj.weight)
        nn.init.zeros_(self.output_proj.bias)

    def _get_session_embedding(
        self,
        item_seq: torch.Tensor,
        lengths: torch.Tensor,
    ) -> torch.Tensor:
        """Get session embedding from GRU hidden state.

        Args:
            item_seq: Padded item index sequences (batch, max_seq_len).
            lengths: Actual sequence lengths (batch,).

        Returns:
            Session embeddings (batch, embedding_dim).
        """
        # Embed items
        embedded = self.item_embedding(item_seq)  # (batch, seq_len, emb_dim)
        embedded = self.emb_dropout(embedded)

        # Sort by length (required by pack_padded_sequence)
        lengths_cpu = lengths.cpu()
        sorted_lengths, sort_idx = lengths_cpu.sort(descending=True)
        embedded = embedded[sort_idx]

        # Pack and run GRU — h_n gives last hidden state at actual seq end
        packed = pack_padded_sequence(
            embedded, sorted_lengths.clamp(min=1), batch_first=True
        )
        _, h_n = self.gru(packed)  # h_n: (num_layers, batch, hidden_dim)
        last_hidden = h_n[-1]  # (batch, hidden_dim) — last GRU layer

        # Unsort to original batch order
        _, unsort_idx = sort_idx.sort()
        last_hidden = last_hidden[unsort_idx]

        # Project to embedding space
        session_emb = self.output_proj(last_hidden)  # (batch, emb_dim)
        return session_emb

    def forward(
        self,
        item_seq: torch.Tensor,
        lengths: torch.Tensor,
        pos_items: torch.Tensor,
        neg_items: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass for training with BPR loss.

        Args:
            item_seq: Padded item sequences (batch, max_seq_len).
            lengths: Actual sequence lengths (batch,).
            pos_items: Positive (target) item indices (batch,).
            neg_items: Negative item indices (batch, n_neg).

        Returns:
            Tuple of (positive_scores, negative_scores).
        """
        session_emb = self._get_session_embedding(item_seq, lengths)

        # Score positive items via dot product
        pos_emb = self.item_embedding(pos_items)  # (batch, emb_dim)
        pos_scores = (session_emb * pos_emb).sum(dim=-1)  # (batch,)

        # Score negative items
        neg_emb = self.item_embedding(neg_items)  # (batch, n_neg, emb_dim)
        neg_scores = torch.bmm(
            neg_emb, session_emb.unsqueeze(-1)
        ).squeeze(-1)  # (batch, n_neg)

        return pos_scores, neg_scores

    @torch.no_grad()
    def predict_all(
        self,
        item_seq: torch.Tensor,
        lengths: torch.Tensor,
    ) -> torch.Tensor:
        """Predict scores for all items (used at inference time).

        Args:
            item_seq: Padded item sequences (batch, max_seq_len).
            lengths: Actual sequence lengths (batch,).

        Returns:
            Scores for all items (batch, n_items).
        """
        session_emb = self._get_session_embedding(item_seq, lengths)
        # Dot product with all item embeddings
        all_emb = self.item_embedding.weight  # (n_items, emb_dim)
        scores = session_emb @ all_emb.T  # (batch, n_items)
        return scores


class GRU4RecRecommender(BaseRecommender):
    """Session-based GRU recommender.

    Wraps GRU4RecModel with the BaseRecommender interface.
    Uses BPR loss with negative sampling for efficient CPU training.

    Attributes:
        embedding_dim: Item embedding dimension.
        hidden_dim: GRU hidden state dimension.
        num_layers: Number of GRU layers.
        dropout: Dropout rate.
        max_seq_len: Maximum session sequence length.
        learning_rate: Adam optimizer learning rate.
        batch_size: Training batch size.
        epochs: Number of training epochs.
        n_negatives: Number of negative samples per positive.
        top_k_items: Limit vocabulary to top-K popular items (0 = no limit).
    """

    def __init__(
        self,
        embedding_dim: int = 64,
        hidden_dim: int = 128,
        num_layers: int = 1,
        dropout: float = 0.2,
        max_seq_len: int = 50,
        learning_rate: float = 0.001,
        batch_size: int = 512,
        epochs: int = 10,
        n_negatives: int = 50,
        top_k_items: int = 20000,
        random_state: int = 42,
    ):
        super().__init__(name="GRU4RecRecommender")
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = dropout
        self.max_seq_len = max_seq_len
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs
        self.n_negatives = n_negatives
        self.top_k_items = top_k_items
        self.random_state = random_state

        self.model: GRU4RecModel | None = None
        self.device = torch.device("cpu")
        self.item_to_idx: dict[int, int] = {}
        self.idx_to_item: dict[int, int] = {}
        self.user_to_idx: dict[int, int] = {}
        self.idx_to_user: dict[int, int] = {}
        self.user_sessions: dict[int, list[list[int]]] = {}
        self.user_item_matrix: csr_matrix | None = None
        self.n_items_model: int = 0
        self.train_history: list[float] = []

    def _build_sessions(
        self,
        interactions: pl.DataFrame,
        user_col: str,
        item_col: str,
    ) -> list[list[int]]:
        """Build session item sequences from interactions.

        Args:
            interactions: DataFrame with session_id, timestamp, item_id, user_id.
            user_col: User column name.
            item_col: Item column name.

        Returns:
            List of session sequences (item indices).
        """
        # Determine session column
        has_session = "session_id" in interactions.columns

        if has_session:
            # Group by session_id, sort by timestamp
            session_df = (
                interactions
                .sort("timestamp")
                .group_by("session_id")
                .agg([
                    pl.col(item_col).alias("items"),
                    pl.col(user_col).first().alias("user_id"),
                ])
            )
        else:
            # Fall back: treat each user as one session
            session_df = (
                interactions
                .sort("timestamp")
                .group_by(user_col)
                .agg([
                    pl.col(item_col).alias("items"),
                    pl.col(user_col).first().alias("user_id"),
                ])
            )

        all_sessions: list[list[int]] = []
        self.user_sessions = {}

        for row in session_df.iter_rows(named=True):
            user_id = row["user_id"]
            items = row["items"]

            # Map to indices (+1 to reserve 0 for padding)
            session_indices = [
                self.item_to_idx[item] + 1
                for item in items
                if item in self.item_to_idx
            ]

            if len(session_indices) >= 2:
                all_sessions.append(session_indices)
                self.user_sessions.setdefault(user_id, []).append(
                    session_indices
                )

        return all_sessions

    def fit(
        self,
        interactions: pl.DataFrame,
        user_col: str = "user_id",
        item_col: str = "item_id",
        **kwargs,
    ) -> "GRU4RecRecommender":
        """Fit GRU4Rec model on interaction data.

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
        logger.info(
            f"  embedding_dim={self.embedding_dim}, hidden_dim={self.hidden_dim}, "
            f"num_layers={self.num_layers}"
        )
        logger.info(
            f"  lr={self.learning_rate}, epochs={self.epochs}, "
            f"batch_size={self.batch_size}, n_negatives={self.n_negatives}"
        )
        start_time = time.time()

        # Filter to top-K most popular items for practical training
        gru_interactions = interactions
        if self.top_k_items and interactions[item_col].n_unique() > self.top_k_items:
            item_counts = (
                interactions.group_by(item_col)
                .len()
                .sort("len", descending=True)
            )
            top_items = item_counts.head(self.top_k_items)[item_col].to_list()
            gru_interactions = interactions.filter(
                pl.col(item_col).is_in(top_items)
            )
            logger.info(
                f"  Vocabulary limited to top {self.top_k_items:,} items "
                f"({len(gru_interactions):,} events)"
            )

        # Build user-item matrix for filtering
        (
            self.user_item_matrix,
            self.user_to_idx,
            self.idx_to_user,
            self.item_to_idx,
            self.idx_to_item,
        ) = prepare_sparse_matrix(gru_interactions, user_col, item_col)

        n_users = len(self.user_to_idx)
        n_items = len(self.item_to_idx)
        self.n_items_model = n_items + 1  # +1 for padding idx 0
        logger.info(f"  Users: {n_users:,}, Items: {n_items:,}")

        # Build session sequences
        all_sessions = self._build_sessions(gru_interactions, user_col, item_col)
        logger.info(f"  Sessions: {len(all_sessions):,}")

        # Create dataset and dataloader
        dataset = SessionDataset(all_sessions, max_seq_len=self.max_seq_len)
        logger.info(f"  Training samples: {len(dataset):,}")

        dataloader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=0,
            collate_fn=collate_sessions,
        )

        # Initialize model
        self.model = GRU4RecModel(
            n_items=self.n_items_model,
            embedding_dim=self.embedding_dim,
            hidden_dim=self.hidden_dim,
            num_layers=self.num_layers,
            dropout=self.dropout,
        ).to(self.device)

        # Optimizer
        optimizer = optim.Adam(
            self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=1e-5,
        )

        # Training loop with BPR loss + negative sampling
        self.model.train()
        self.train_history = []

        for epoch in range(self.epochs):
            epoch_loss = 0.0
            n_batches = 0

            pbar = tqdm(
                dataloader,
                desc=f"Epoch {epoch + 1}/{self.epochs}",
                leave=True,
            )
            for sequences, targets, lengths in pbar:
                sequences = sequences.to(self.device)
                targets = targets.to(self.device)
                lengths = lengths.to(self.device)

                # Sample negative items (avoid padding index 0)
                neg_items = torch.randint(
                    1, self.n_items_model,
                    (len(targets), self.n_negatives),
                    device=self.device,
                )

                optimizer.zero_grad()
                pos_scores, neg_scores = self.model(
                    sequences, lengths, targets, neg_items
                )

                # BPR loss: -log(sigmoid(pos - neg))
                diff = pos_scores.unsqueeze(1) - neg_scores  # (batch, n_neg)
                loss = -torch.log(torch.sigmoid(diff) + 1e-8).mean()

                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5.0)
                optimizer.step()

                epoch_loss += loss.item()
                n_batches += 1

                if n_batches % 50 == 0:
                    pbar.set_postfix(loss=f"{epoch_loss / n_batches:.4f}")

            avg_loss = epoch_loss / max(n_batches, 1)
            self.train_history.append(avg_loss)
            logger.info(
                f"  Epoch {epoch + 1}/{self.epochs} - Loss: {avg_loss:.4f}"
            )

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

        Uses the user's most recent session as GRU context
        to predict the next items.

        Args:
            user_id: User identifier.
            n: Number of recommendations.
            filter_already_liked: Whether to filter already interacted items.

        Returns:
            List of recommended item IDs.
        """
        self._check_is_fitted()

        if user_id not in self.user_sessions:
            return []

        self.model.eval()

        # Get user's last session
        last_session = self.user_sessions[user_id][-1]
        # Truncate to max_seq_len
        seq = last_session[-self.max_seq_len:]

        seq_tensor = torch.tensor([seq], dtype=torch.long, device=self.device)
        length_tensor = torch.tensor([len(seq)], dtype=torch.long)

        scores = self.model.predict_all(seq_tensor, length_tensor)
        scores = scores[0].cpu().numpy()

        # Zero out padding index
        scores[0] = -np.inf

        # Filter already liked items
        if filter_already_liked and user_id in self.user_to_idx:
            user_idx = self.user_to_idx[user_id]
            liked = set(self.user_item_matrix[user_idx].indices)
            for item_idx in liked:
                if item_idx + 1 < len(scores):
                    scores[item_idx + 1] = -np.inf

        # Get top-N indices (offset by 1 for padding)
        top_indices = np.argsort(scores)[::-1][:n * 2]

        results = []
        for idx in top_indices:
            actual_idx = int(idx) - 1
            if actual_idx in self.idx_to_item:
                results.append(self.idx_to_item[actual_idx])
            if len(results) >= n:
                break

        return results

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

        if user_id not in self.user_sessions:
            return []

        self.model.eval()

        last_session = self.user_sessions[user_id][-1]
        seq = last_session[-self.max_seq_len:]

        seq_tensor = torch.tensor([seq], dtype=torch.long, device=self.device)
        length_tensor = torch.tensor([len(seq)], dtype=torch.long)

        scores = self.model.predict_all(seq_tensor, length_tensor)
        scores = scores[0].cpu().numpy()

        scores[0] = -np.inf

        if filter_already_liked and user_id in self.user_to_idx:
            user_idx = self.user_to_idx[user_id]
            liked = set(self.user_item_matrix[user_idx].indices)
            for item_idx in liked:
                if item_idx + 1 < len(scores):
                    scores[item_idx + 1] = -np.inf

        top_indices = np.argsort(scores)[::-1][:n * 2]

        results = []
        for idx in top_indices:
            actual_idx = int(idx) - 1
            if actual_idx in self.idx_to_item:
                results.append(
                    (self.idx_to_item[actual_idx], float(scores[idx]))
                )
            if len(results) >= n:
                break

        return results

    def save(self, path: str | Path) -> None:
        """Save model to disk."""
        path = Path(path)
        state = {
            "model_state_dict": (
                self.model.state_dict() if self.model else None
            ),
            "config": {
                "embedding_dim": self.embedding_dim,
                "hidden_dim": self.hidden_dim,
                "num_layers": self.num_layers,
                "dropout": self.dropout,
                "max_seq_len": self.max_seq_len,
                "learning_rate": self.learning_rate,
                "epochs": self.epochs,
                "n_negatives": self.n_negatives,
                "top_k_items": self.top_k_items,
            },
            "mappings": {
                "user_to_idx": self.user_to_idx,
                "idx_to_user": self.idx_to_user,
                "item_to_idx": self.item_to_idx,
                "idx_to_item": self.idx_to_item,
            },
            "n_items_model": self.n_items_model,
            "user_sessions": self.user_sessions,
            "user_item_matrix": self.user_item_matrix,
            "train_history": self.train_history,
        }
        with open(path, "wb") as f:
            pickle.dump(state, f)
        logger.info(f"Saved {self.name} to {path}")

    def load(self, path: str | Path) -> "GRU4RecRecommender":
        """Load model from disk."""
        path = Path(path)
        with open(path, "rb") as f:
            state = pickle.load(f)

        self.user_to_idx = state["mappings"]["user_to_idx"]
        self.idx_to_user = state["mappings"]["idx_to_user"]
        self.item_to_idx = state["mappings"]["item_to_idx"]
        self.idx_to_item = state["mappings"]["idx_to_item"]
        self.user_sessions = state.get("user_sessions", {})
        self.user_item_matrix = state.get("user_item_matrix")
        self.train_history = state.get("train_history", [])
        self.n_items_model = state.get("n_items_model", len(self.item_to_idx) + 1)

        config = state["config"]

        self.model = GRU4RecModel(
            n_items=self.n_items_model,
            embedding_dim=config["embedding_dim"],
            hidden_dim=config["hidden_dim"],
            num_layers=config["num_layers"],
            dropout=config["dropout"],
        ).to(self.device)

        if state["model_state_dict"]:
            self.model.load_state_dict(state["model_state_dict"])

        self._is_fitted = True
        logger.info(f"Loaded {self.name} from {path}")
        return self
