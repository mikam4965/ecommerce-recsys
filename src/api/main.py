"""FastAPI application for recommendation service.

Run with: uvicorn src.api.main:app --reload --port 8000
"""

import os
import sys
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import yaml

from src.api.cache import RecommendationCache
from src.api.routes import router
from src.models.hybrid import HybridRecommender
from src.data.database import DatabaseManager, init_database

# API version
API_VERSION = "1.0.0"

# Configuration from environment
MODEL_PATH = os.getenv("MODEL_PATH", str(PROJECT_ROOT / "models" / "hybrid_best.pkl"))
REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("REDIS_PORT", "6379"))
REDIS_ENABLED = os.getenv("REDIS_ENABLED", "false").lower() == "true"

# Database configuration
DB_CONFIG_PATH = PROJECT_ROOT / "configs" / "database.yaml"
DB_ENABLED = os.getenv("DB_ENABLED", "true").lower() == "true"


def load_db_config() -> dict:
    """Load database configuration from YAML."""
    if DB_CONFIG_PATH.exists():
        with open(DB_CONFIG_PATH) as f:
            config = yaml.safe_load(f)
        return config.get("database", {})
    return {"db_path": "data/database.sqlite", "cache_ttl": 3600}


class AppState:
    """Application state holder."""

    def __init__(self):
        self.model: HybridRecommender | None = None
        self.cache: RecommendationCache | None = None
        self.db: DatabaseManager | None = None


# Global app state
app_state = AppState()


def load_model(path: str | Path) -> HybridRecommender:
    """Load recommendation model from disk.

    Args:
        path: Path to the model file.

    Returns:
        Loaded HybridRecommender.

    Raises:
        FileNotFoundError: If model file not found.
    """
    path = Path(path)

    if not path.exists():
        # Try alternative paths
        alternatives = [
            PROJECT_ROOT / "models" / "hybrid_model.pkl",
            PROJECT_ROOT / "models" / "hybrid_best.pkl",
        ]
        for alt_path in alternatives:
            if alt_path.exists():
                path = alt_path
                logger.info(f"Using alternative model path: {path}")
                break
        else:
            raise FileNotFoundError(
                f"Model not found at {path} or alternatives: {alternatives}"
            )

    logger.info(f"Loading model from {path}...")
    model = HybridRecommender()
    model.load(path)

    logger.info(f"Model loaded: {len(model.user_to_idx):,} users, {len(model.item_to_idx):,} items")

    return model


def init_cache() -> RecommendationCache | None:
    """Initialize Redis cache if enabled.

    Returns:
        RecommendationCache instance or None if disabled.
    """
    if not REDIS_ENABLED:
        logger.info("Redis caching disabled")
        return None

    logger.info(f"Initializing Redis cache: {REDIS_HOST}:{REDIS_PORT}")
    cache = RecommendationCache(
        host=REDIS_HOST,
        port=REDIS_PORT,
        enabled=True,
    )

    if cache.is_connected():
        logger.info("Redis cache connected successfully")
    else:
        logger.warning("Redis cache connection failed, running without cache")
        return None

    return cache


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler.

    Loads model on startup, cleans up on shutdown.
    """
    # Startup
    logger.info("=" * 50)
    logger.info("Starting Recommendation API Service")
    logger.info("=" * 50)

    try:
        app_state.model = load_model(MODEL_PATH)
    except FileNotFoundError as e:
        logger.error(f"Failed to load model: {e}")
        logger.warning("API will start without model - health endpoint will report 'starting'")
    except Exception as e:
        logger.error(f"Unexpected error loading model: {e}")
        raise

    # Initialize cache (Redis - optional)
    app_state.cache = init_cache()

    # Initialize SQLite database
    if DB_ENABLED:
        try:
            db_config = load_db_config()
            db_path = PROJECT_ROOT / db_config.get("db_path", "data/database.sqlite")
            cache_ttl = db_config.get("cache_ttl", 3600)

            # Ensure database is initialized
            init_database(db_path)

            app_state.db = DatabaseManager(
                db_path=db_path,
                cache_ttl=cache_ttl,
            )
            logger.info(f"SQLite database initialized: {db_path}")
        except Exception as e:
            logger.warning(f"Failed to initialize SQLite: {e}")
            app_state.db = None
    else:
        logger.info("SQLite caching disabled")

    logger.info("Recommendation API ready!")
    logger.info(f"Swagger docs available at: http://localhost:8000/docs")

    yield

    # Shutdown
    logger.info("Shutting down Recommendation API...")
    app_state.model = None
    app_state.cache = None
    app_state.db = None
    logger.info("Recommendation API stopped")


# Create FastAPI app
app = FastAPI(
    title="E-commerce Recommendation API",
    description="""
## Recommendation Service for E-commerce Platform

This API provides personalized product recommendations using a hybrid
collaborative filtering model (ALS + content-based features).

### Features:
- **Personalized recommendations** for known users
- **Cold start recommendations** based on session items
- **Similar items** based on item embeddings
- **Popular items** as fallback

### Models:
- HybridRecommender: ALS-based collaborative filtering with item/user features

### Caching:
- Optional Redis caching for popular items and active user recommendations
    """,
    version=API_VERSION,
    contact={
        "name": "Recommendation System",
        "email": "admin@example.com",
    },
    license_info={
        "name": "MIT",
    },
    lifespan=lifespan,
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify actual origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routes
app.include_router(router, prefix="/api/v1")


# Root endpoint
@app.get("/", tags=["Root"])
async def root() -> dict[str, Any]:
    """Root endpoint with API information."""
    return {
        "service": "E-commerce Recommendation API",
        "version": API_VERSION,
        "docs": "/docs",
        "health": "/api/v1/health",
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "src.api.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info",
    )
