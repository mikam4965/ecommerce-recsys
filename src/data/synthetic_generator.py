"""Synthetic e-commerce user behavior data generator.

Generates realistic event data (view, addtocart, transaction) with:
- Power-law item popularity distribution
- User profiles (browser, buyer, power_user)
- Session-based transition probabilities
- Temporal patterns (peak hours, weekends)
- Category preferences per user
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import polars as pl
import yaml
from loguru import logger


@dataclass
class UserProfile:
    """User behavior profile configuration."""

    name: str
    fraction: float
    avg_sessions: int
    avg_events_per_session: int
    purchase_probability: float


@dataclass
class UserBehaviorGenerator:
    """Generates synthetic e-commerce user behavior data.

    Attributes:
        seed: Random seed for reproducibility.
        n_users: Number of users to generate.
        n_items: Number of items in catalog.
        n_main_categories: Number of top-level categories.
        n_sub_categories: Number of sub-categories.
        profiles: List of user profile configurations.
        transitions: Session event transition probabilities.
        favorite_category_ratio: Fraction of events in user's favorite categories.
        n_favorite_categories: Number of favorite categories per user.
        time_range_days: Duration of generated events in days.
        peak_hours: List of peak traffic hours.
        weekend_factor: Traffic multiplier for weekends.
        item_popularity_alpha: Zipf exponent for item popularity.
        hot_items_fraction: Fraction of items considered "hot".
        session_gap_minutes: Gap between sessions in minutes.
        max_session_events: Maximum events per session.
    """

    seed: int = 42
    n_users: int = 50_000
    n_items: int = 30_000
    n_main_categories: int = 20
    n_sub_categories: int = 200
    profiles: list[UserProfile] = field(default_factory=list)
    transitions: dict[str, float] = field(default_factory=dict)
    favorite_category_ratio: float = 0.80
    n_favorite_categories: int = 3
    time_range_days: int = 140
    peak_hours: list[int] = field(default_factory=lambda: [10, 11, 12, 13, 14, 18, 19, 20, 21])
    weekend_factor: float = 0.75
    item_popularity_alpha: float = 1.5
    hot_items_fraction: float = 0.02
    session_gap_minutes: int = 30
    max_session_events: int = 50

    # Internal state
    _rng: np.random.Generator = field(init=False, repr=False)
    _items_df: pl.DataFrame | None = field(init=False, default=None, repr=False)
    _item_popularity: np.ndarray | None = field(init=False, default=None, repr=False)
    _category_items: dict[int, list[int]] = field(init=False, default_factory=dict, repr=False)

    def __post_init__(self) -> None:
        self._rng = np.random.default_rng(self.seed)
        if not self.profiles:
            self.profiles = [
                UserProfile("browser", 0.70, 3, 5, 0.05),
                UserProfile("buyer", 0.20, 8, 10, 0.25),
                UserProfile("power_user", 0.10, 20, 15, 0.45),
            ]
        if not self.transitions:
            self.transitions = {
                "view_to_view": 0.60,
                "view_to_cart": 0.15,
                "view_to_end": 0.25,
                "cart_to_purchase": 0.50,
                "cart_to_view": 0.30,
                "cart_to_end": 0.20,
            }

    @classmethod
    def from_config(cls, config_path: str | Path) -> UserBehaviorGenerator:
        """Create generator from YAML config file.

        Args:
            config_path: Path to synthetic.yaml config.

        Returns:
            Configured UserBehaviorGenerator instance.
        """
        config_path = Path(config_path)
        with open(config_path) as f:
            config = yaml.safe_load(f)

        cfg = config.get("synthetic", config)

        profiles = []
        for name, params in cfg.get("user_profiles", {}).items():
            profiles.append(UserProfile(name=name, **params))

        return cls(
            seed=cfg.get("random_seed", 42),
            n_users=cfg.get("n_users", 50_000),
            n_items=cfg.get("n_items", 30_000),
            n_main_categories=cfg.get("n_main_categories", 20),
            n_sub_categories=cfg.get("n_sub_categories", 200),
            profiles=profiles or None,
            transitions=cfg.get("transitions", {}),
            favorite_category_ratio=cfg.get("favorite_category_ratio", 0.80),
            n_favorite_categories=cfg.get("n_favorite_categories", 3),
            time_range_days=cfg.get("time_range_days", 140),
            peak_hours=cfg.get("peak_hours", [10, 11, 12, 13, 14, 18, 19, 20, 21]),
            weekend_factor=cfg.get("weekend_factor", 0.75),
            item_popularity_alpha=cfg.get("item_popularity_alpha", 1.5),
            hot_items_fraction=cfg.get("hot_items_fraction", 0.02),
            session_gap_minutes=cfg.get("session_gap_minutes", 30),
            max_session_events=cfg.get("max_session_events", 50),
        )

    def generate_items(self) -> pl.DataFrame:
        """Generate item catalog with category hierarchy.

        Creates items distributed across main categories and subcategories,
        with a power-law popularity distribution.

        Returns:
            DataFrame with columns: item_id, category_id, parent_category_id, popularity.
        """
        logger.info(f"Generating {self.n_items:,} items with {self.n_main_categories} "
                     f"main + {self.n_sub_categories} sub categories...")

        # Create category hierarchy
        sub_per_main = self.n_sub_categories // self.n_main_categories
        sub_category_ids = []
        parent_category_ids = []

        for main_cat in range(1, self.n_main_categories + 1):
            for sub in range(sub_per_main):
                sub_id = (main_cat - 1) * sub_per_main + sub + 1
                sub_category_ids.append(sub_id)
                parent_category_ids.append(main_cat)

        # Distribute items across subcategories (uneven - some categories are bigger)
        category_sizes = self._rng.dirichlet(np.ones(len(sub_category_ids)) * 0.5)
        items_per_category = np.maximum(
            (category_sizes * self.n_items).astype(int), 1
        )
        # Adjust to match exact item count
        diff = self.n_items - items_per_category.sum()
        if diff > 0:
            indices = self._rng.choice(len(items_per_category), size=diff)
            for idx in indices:
                items_per_category[idx] += 1
        elif diff < 0:
            indices = np.where(items_per_category > 1)[0]
            chosen = self._rng.choice(indices, size=abs(diff), replace=False)
            for idx in chosen:
                items_per_category[idx] -= 1

        # Build item records
        item_ids = []
        item_categories = []
        item_parent_categories = []

        item_id = 1
        for cat_idx, count in enumerate(items_per_category):
            for _ in range(count):
                item_ids.append(item_id)
                item_categories.append(sub_category_ids[cat_idx])
                item_parent_categories.append(parent_category_ids[cat_idx])
                item_id += 1

        # Power-law popularity (Zipf)
        ranks = np.arange(1, self.n_items + 1)
        popularity = 1.0 / np.power(ranks, self.item_popularity_alpha)
        # Shuffle so popularity isn't correlated with item_id
        self._rng.shuffle(popularity)
        popularity = popularity / popularity.sum()

        self._items_df = pl.DataFrame({
            "item_id": item_ids,
            "category_id": item_categories,
            "parent_category_id": item_parent_categories,
            "popularity": popularity.tolist(),
        })

        self._item_popularity = popularity

        # Build category -> items mapping
        self._category_items = {}
        for iid, cat in zip(item_ids, item_categories):
            self._category_items.setdefault(cat, []).append(iid)

        logger.info(f"  Items: {len(item_ids):,}, Categories: {len(sub_category_ids)}")
        return self._items_df

    def generate_user_profiles(self) -> pl.DataFrame:
        """Generate user profiles with behavioral types.

        Each user is assigned a profile (browser/buyer/power_user) and
        a set of favorite categories.

        Returns:
            DataFrame with columns: user_id, profile, n_favorite_categories,
            favorite_categories.
        """
        logger.info(f"Generating {self.n_users:,} user profiles...")

        user_ids = list(range(1, self.n_users + 1))
        profile_names = []
        all_favorites = []

        # Assign profiles
        all_sub_categories = list(self._category_items.keys())

        for _ in range(self.n_users):
            # Pick profile by fraction
            r = self._rng.random()
            cumulative = 0.0
            assigned = self.profiles[-1]
            for profile in self.profiles:
                cumulative += profile.fraction
                if r < cumulative:
                    assigned = profile
                    break

            profile_names.append(assigned.name)

            # Assign favorite categories (weighted by category size)
            cat_weights = np.array([
                len(self._category_items.get(c, [])) for c in all_sub_categories
            ], dtype=float)
            cat_weights /= cat_weights.sum()

            n_favs = min(self.n_favorite_categories, len(all_sub_categories))
            favs = self._rng.choice(
                all_sub_categories, size=n_favs, replace=False, p=cat_weights
            ).tolist()
            all_favorites.append(favs)

        users_df = pl.DataFrame({
            "user_id": user_ids,
            "profile": profile_names,
            "favorite_categories": all_favorites,
        })

        logger.info(f"  Profiles: { {p.name: profile_names.count(p.name) for p in self.profiles} }")
        return users_df

    def _get_profile(self, profile_name: str) -> UserProfile:
        """Get profile config by name."""
        for p in self.profiles:
            if p.name == profile_name:
                return p
        return self.profiles[0]

    def _pick_item(
        self, favorite_categories: list[int], in_cart: bool = False
    ) -> int:
        """Pick an item based on category preferences and popularity.

        Args:
            favorite_categories: User's favorite category IDs.
            in_cart: If True, pick from favorite categories more often.

        Returns:
            Selected item_id.
        """
        use_favorite = self._rng.random() < self.favorite_category_ratio

        if use_favorite and favorite_categories:
            cat = self._rng.choice(favorite_categories)
            candidates = self._category_items.get(cat, [])
            if candidates:
                # Use popularity within category
                pop = np.array([
                    self._item_popularity[iid - 1] for iid in candidates
                ])
                pop = pop / pop.sum()
                return int(self._rng.choice(candidates, p=pop))

        # Global popularity sampling
        return int(self._rng.choice(self.n_items, p=self._item_popularity)) + 1

    def _generate_timestamp(self, base_ts: int, offset_seconds: int) -> int:
        """Generate a realistic timestamp with temporal patterns.

        Args:
            base_ts: Start timestamp (Unix epoch).
            offset_seconds: Maximum offset from base.

        Returns:
            Unix timestamp with applied temporal patterns.
        """
        ts = base_ts + self._rng.integers(0, max(offset_seconds, 1))

        # Apply peak hours bias
        from datetime import datetime, timezone
        dt = datetime.fromtimestamp(ts, tz=timezone.utc)
        hour = dt.hour

        # Reject non-peak hours with some probability
        if hour not in self.peak_hours:
            if self._rng.random() < 0.5:
                # Shift to a peak hour
                peak = int(self._rng.choice(self.peak_hours))
                ts = ts - (hour - peak) * 3600

        # Weekend reduction
        if dt.weekday() >= 5:  # Saturday=5, Sunday=6
            if self._rng.random() > self.weekend_factor:
                # Shift to a weekday
                days_to_shift = dt.weekday() - 4  # Move to Friday
                ts -= days_to_shift * 86400

        return int(ts)

    def generate_events(self, n_events: int = 1_000_000) -> pl.DataFrame:
        """Generate synthetic events with realistic patterns.

        Generates view/addtocart/transaction events following session-based
        transition probabilities and user behavior profiles.

        Args:
            n_events: Target number of events to generate.

        Returns:
            DataFrame with columns: user_id, timestamp, event_type, item_id,
            category_id, session_id.
        """
        if self._items_df is None:
            self.generate_items()

        users_df = self.generate_user_profiles()
        user_profiles = dict(zip(
            users_df["user_id"].to_list(),
            users_df["profile"].to_list(),
        ))
        user_favorites = dict(zip(
            users_df["user_id"].to_list(),
            users_df["favorite_categories"].to_list(),
        ))

        logger.info(f"Generating {n_events:,} events...")

        # Base timestamp: Jan 1 2015 (matches RetailRocket epoch)
        base_ts = 1420070400
        total_seconds = self.time_range_days * 86400

        # Pre-calculate events per user based on profile
        user_ids = list(range(1, self.n_users + 1))
        events_per_user = np.zeros(self.n_users, dtype=int)

        for i, uid in enumerate(user_ids):
            profile = self._get_profile(user_profiles[uid])
            expected = profile.avg_sessions * profile.avg_events_per_session
            events_per_user[i] = max(1, int(self._rng.poisson(expected)))

        # Scale to match target n_events
        scale = n_events / events_per_user.sum()
        events_per_user = np.maximum((events_per_user * scale).astype(int), 1)

        # Adjust for exact count
        diff = n_events - events_per_user.sum()
        if diff > 0:
            indices = self._rng.choice(self.n_users, size=diff)
            for idx in indices:
                events_per_user[idx] += 1
        elif diff < 0:
            indices = np.where(events_per_user > 1)[0]
            chosen = self._rng.choice(indices, size=min(abs(diff), len(indices)), replace=True)
            for idx in chosen:
                if events_per_user[idx] > 1:
                    events_per_user[idx] -= 1

        # Generate events
        all_user_ids = []
        all_timestamps = []
        all_event_types = []
        all_item_ids = []
        all_session_ids = []

        session_counter = 0
        progress_step = self.n_users // 10

        for user_idx, uid in enumerate(user_ids):
            if progress_step > 0 and user_idx % progress_step == 0:
                pct = (user_idx / self.n_users) * 100
                logger.info(f"  Progress: {pct:.0f}% ({user_idx:,}/{self.n_users:,} users)")

            profile = self._get_profile(user_profiles[uid])
            favorites = user_favorites[uid]
            n_user_events = events_per_user[user_idx]

            # Generate sessions until user's event quota is filled
            events_remaining = n_user_events
            max_attempts = n_user_events * 3  # safety limit

            while events_remaining > 0 and max_attempts > 0:
                max_attempts -= 1

                session_counter += 1
                session_id = session_counter

                # Session length target
                session_target = min(
                    max(1, int(self._rng.poisson(profile.avg_events_per_session))),
                    events_remaining,
                    self.max_session_events,
                )

                # Session start time
                session_ts = self._generate_timestamp(base_ts, total_seconds)
                current_ts = session_ts

                # Generate session events using transition model
                state = "view"
                cart_items = []
                session_generated = 0

                for _ in range(session_target):
                    event_added = False

                    if state == "view":
                        item_id = self._pick_item(favorites)
                        all_user_ids.append(uid)
                        all_timestamps.append(current_ts)
                        all_event_types.append("view")
                        all_item_ids.append(item_id)
                        all_session_ids.append(session_id)
                        event_added = True

                        # Transition
                        r = self._rng.random()
                        if r < self.transitions["view_to_cart"]:
                            state = "cart"
                            cart_items.append(item_id)
                        elif r < self.transitions["view_to_cart"] + self.transitions["view_to_end"]:
                            state = "end"
                        # else: stay in view

                    elif state == "cart":
                        item_id = cart_items[-1] if cart_items else self._pick_item(favorites)
                        all_user_ids.append(uid)
                        all_timestamps.append(current_ts)
                        all_event_types.append("addtocart")
                        all_item_ids.append(item_id)
                        all_session_ids.append(session_id)
                        event_added = True

                        # Transition
                        r = self._rng.random()
                        if r < self.transitions["cart_to_purchase"]:
                            state = "purchase"
                        elif r < self.transitions["cart_to_purchase"] + self.transitions["cart_to_view"]:
                            state = "view"
                        else:
                            state = "end"

                    elif state == "purchase":
                        if self._rng.random() < profile.purchase_probability:
                            item_id = cart_items[-1] if cart_items else self._pick_item(favorites)
                            all_user_ids.append(uid)
                            all_timestamps.append(current_ts)
                            all_event_types.append("transaction")
                            all_item_ids.append(item_id)
                            all_session_ids.append(session_id)
                            event_added = True
                        state = "view"
                        cart_items = []

                    elif state == "end":
                        break

                    if event_added:
                        session_generated += 1

                    # Increment time within session (5-60 seconds between events)
                    current_ts += int(self._rng.integers(5, 61))

                events_remaining -= max(session_generated, 1)

        # Build DataFrame
        events = pl.DataFrame({
            "user_id": all_user_ids,
            "timestamp": all_timestamps,
            "event_type": all_event_types,
            "item_id": all_item_ids,
            "session_id": all_session_ids,
        })

        # Add category_id from items
        item_to_category = dict(zip(
            self._items_df["item_id"].to_list(),
            self._items_df["category_id"].to_list(),
        ))
        events = events.with_columns(
            pl.col("item_id").replace_strict(
                item_to_category, default=0
            ).alias("category_id")
        )

        # Sort by timestamp
        events = events.sort("timestamp")

        logger.info(f"  Generated {len(events):,} events")
        logger.info(f"  Event distribution: {events['event_type'].value_counts().sort('event_type').to_dict()}")
        logger.info(f"  Unique users: {events['user_id'].n_unique():,}")
        logger.info(f"  Unique items: {events['item_id'].n_unique():,}")
        logger.info(f"  Sessions: {events['session_id'].n_unique():,}")

        return events

    def add_realistic_patterns(self, events: pl.DataFrame) -> pl.DataFrame:
        """Add realistic patterns to generated events.

        Applies:
        - Hot items boost (top items get extra views)
        - Seasonal trends (gradual changes over time)

        Args:
            events: Generated events DataFrame.

        Returns:
            Events with additional realistic patterns applied.
        """
        logger.info("Adding realistic patterns...")
        n_original = len(events)

        # 1. Hot items: add extra view events for top popular items
        n_hot = max(1, int(self.n_items * self.hot_items_fraction))
        hot_item_ids = (
            self._items_df
            .sort("popularity", descending=True)
            .head(n_hot)["item_id"]
            .to_list()
        )

        # Add 5-20% extra views for hot items
        hot_events = events.filter(
            pl.col("item_id").is_in(hot_item_ids) & (pl.col("event_type") == "view")
        )
        n_extra = int(len(hot_events) * 0.15)
        if n_extra > 0:
            extra_indices = self._rng.choice(len(hot_events), size=n_extra, replace=True)
            extra = hot_events[extra_indices.tolist()]
            # Slightly shift timestamps
            noise = self._rng.integers(-300, 300, size=n_extra)
            extra = extra.with_columns(
                (pl.col("timestamp") + pl.Series("ts_noise", noise.tolist())).alias("timestamp")
            )
            events = pl.concat([events, extra])

        events = events.sort("timestamp")

        logger.info(f"  Added {len(events) - n_original:,} hot item events "
                     f"({n_hot} hot items)")

        return events

    def get_distribution_stats(self, events: pl.DataFrame) -> dict:
        """Calculate distribution statistics for validation.

        Args:
            events: Events DataFrame to analyze.

        Returns:
            Dictionary with distribution statistics.
        """
        stats = {}

        # Event type distribution
        event_counts = events["event_type"].value_counts().sort("event_type")
        total = len(events)
        stats["event_distribution"] = {
            row["event_type"]: {
                "count": row["count"],
                "percentage": round(row["count"] / total * 100, 2),
            }
            for row in event_counts.iter_rows(named=True)
        }

        # User activity distribution
        user_events = events.group_by("user_id").len()
        stats["user_activity"] = {
            "mean": round(user_events["len"].mean(), 2),
            "median": round(user_events["len"].median(), 2),
            "std": round(user_events["len"].std(), 2),
            "min": int(user_events["len"].min()),
            "max": int(user_events["len"].max()),
        }

        # Item popularity distribution
        item_events = events.group_by("item_id").len()
        stats["item_popularity"] = {
            "mean": round(item_events["len"].mean(), 2),
            "median": round(item_events["len"].median(), 2),
            "std": round(item_events["len"].std(), 2),
            "min": int(item_events["len"].min()),
            "max": int(item_events["len"].max()),
            "gini": round(self._gini_coefficient(item_events["len"].to_numpy()), 4),
        }

        # Conversion rates
        n_users = events["user_id"].n_unique()
        cart_users = events.filter(pl.col("event_type") == "addtocart")["user_id"].n_unique()
        purchase_users = events.filter(pl.col("event_type") == "transaction")["user_id"].n_unique()
        stats["conversion"] = {
            "view_to_cart": round(cart_users / n_users * 100, 2),
            "cart_to_purchase": round(
                purchase_users / cart_users * 100, 2
            ) if cart_users > 0 else 0,
            "overall": round(purchase_users / n_users * 100, 2),
        }

        # Sessions
        session_lens = events.group_by("session_id").len()
        stats["sessions"] = {
            "total": events["session_id"].n_unique(),
            "avg_length": round(session_lens["len"].mean(), 2),
            "median_length": round(session_lens["len"].median(), 2),
        }

        return stats

    @staticmethod
    def _gini_coefficient(values: np.ndarray) -> float:
        """Calculate Gini coefficient for distribution inequality."""
        sorted_vals = np.sort(values)
        n = len(sorted_vals)
        cumulative = np.cumsum(sorted_vals)
        return float(
            (2 * np.sum((np.arange(1, n + 1) * sorted_vals)) / (n * cumulative[-1])) - (n + 1) / n
        )

    def save(
        self,
        events: pl.DataFrame,
        output_dir: str | Path,
        items: bool = True,
    ) -> None:
        """Save generated data to parquet files.

        Args:
            events: Events DataFrame.
            output_dir: Output directory path.
            items: Whether to also save item catalog.
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        events_path = output_dir / "synthetic_events.parquet"
        events.write_parquet(events_path)
        logger.info(f"Events saved to {events_path} ({len(events):,} rows)")

        if items and self._items_df is not None:
            items_path = output_dir / "synthetic_items.parquet"
            self._items_df.write_parquet(items_path)
            logger.info(f"Items saved to {items_path} ({len(self._items_df):,} rows)")
