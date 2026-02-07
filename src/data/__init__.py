"""Data loading and preprocessing."""

from src.data.loader import load_categories, load_events, load_item_properties
from src.data.preprocessing import (
    create_final_dataset,
    extract_item_metadata,
    handle_missing_values,
    reconstruct_sessions,
)
from src.data.splitter import split_by_users, temporal_split
from src.data.database import DatabaseManager, init_database, get_connection
from src.data.synthetic_generator import UserBehaviorGenerator

__all__ = [
    # Loader
    "load_events",
    "load_item_properties",
    "load_categories",
    # Preprocessing
    "reconstruct_sessions",
    "extract_item_metadata",
    "handle_missing_values",
    "create_final_dataset",
    # Splitter
    "temporal_split",
    "split_by_users",
    # Database
    "DatabaseManager",
    "init_database",
    "get_connection",
    # Synthetic data
    "UserBehaviorGenerator",
]
