"""REST API for recommendation service.

Run with: uvicorn src.api.main:app --reload --port 8000
"""

from src.api.main import app, API_VERSION
from src.api.cache import RecommendationCache
from src.api.schemas import (
    RecommendationRequest,
    RecommendationResponse,
    ColdStartRequest,
    ColdStartResponse,
    SimilarItemsRequest,
    SimilarItemsResponse,
    PopularItemsResponse,
    HealthResponse,
)

__all__ = [
    "app",
    "API_VERSION",
    "RecommendationCache",
    "RecommendationRequest",
    "RecommendationResponse",
    "ColdStartRequest",
    "ColdStartResponse",
    "SimilarItemsRequest",
    "SimilarItemsResponse",
    "PopularItemsResponse",
    "HealthResponse",
]
