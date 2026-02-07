"""Pydantic models for API request/response validation."""

from pydantic import BaseModel, Field


class RecommendationItem(BaseModel):
    """Single recommendation item with score."""

    item_id: int = Field(..., description="Item identifier")
    score: float = Field(..., description="Recommendation score")


class RecommendationRequest(BaseModel):
    """Request for personalized recommendations."""

    user_id: int = Field(..., description="User identifier")
    n_recommendations: int = Field(
        default=10,
        ge=1,
        le=100,
        description="Number of recommendations to return",
    )
    filter_seen: bool = Field(
        default=True,
        description="Whether to filter out items user already interacted with",
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "user_id": 123456,
                    "n_recommendations": 10,
                    "filter_seen": True,
                }
            ]
        }
    }


class RecommendationResponse(BaseModel):
    """Response with personalized recommendations."""

    user_id: int = Field(..., description="User identifier")
    recommendations: list[RecommendationItem] = Field(
        ..., description="List of recommended items with scores"
    )
    model_type: str = Field(
        default="hybrid", description="Type of model used for recommendations"
    )
    is_fallback: bool = Field(
        default=False, description="Whether fallback (popular) recommendations were used"
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "user_id": 123456,
                    "recommendations": [
                        {"item_id": 100, "score": 0.95},
                        {"item_id": 200, "score": 0.87},
                        {"item_id": 300, "score": 0.82},
                    ],
                    "model_type": "hybrid",
                    "is_fallback": False,
                }
            ]
        }
    }


class ColdStartRequest(BaseModel):
    """Request for cold start recommendations based on session."""

    session_items: list[int] = Field(
        ...,
        min_length=1,
        description="List of item IDs from current session (views/cart)",
    )
    n_recommendations: int = Field(
        default=10,
        ge=1,
        le=100,
        description="Number of recommendations to return",
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "session_items": [100, 200, 300],
                    "n_recommendations": 10,
                }
            ]
        }
    }


class ColdStartResponse(BaseModel):
    """Response with session-based recommendations for new users."""

    session_items: list[int] = Field(..., description="Input session items")
    recommendations: list[RecommendationItem] = Field(
        ..., description="List of recommended items with scores"
    )
    known_items_count: int = Field(
        ..., description="Number of session items found in model"
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "session_items": [100, 200, 300],
                    "recommendations": [
                        {"item_id": 400, "score": 0.91},
                        {"item_id": 500, "score": 0.85},
                    ],
                    "known_items_count": 2,
                }
            ]
        }
    }


class SimilarItemsRequest(BaseModel):
    """Request for similar items."""

    item_id: int = Field(..., description="Item identifier to find similar items for")
    n_similar: int = Field(
        default=10,
        ge=1,
        le=100,
        description="Number of similar items to return",
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "item_id": 100,
                    "n_similar": 10,
                }
            ]
        }
    }


class SimilarItemsResponse(BaseModel):
    """Response with similar items."""

    item_id: int = Field(..., description="Source item identifier")
    similar_items: list[RecommendationItem] = Field(
        ..., description="List of similar items with scores"
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "item_id": 100,
                    "similar_items": [
                        {"item_id": 101, "score": 0.92},
                        {"item_id": 102, "score": 0.88},
                    ],
                }
            ]
        }
    }


class PopularItemsResponse(BaseModel):
    """Response with popular items."""

    items: list[RecommendationItem] = Field(
        ..., description="List of popular items with popularity scores"
    )
    cached: bool = Field(default=False, description="Whether result was cached")

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "items": [
                        {"item_id": 100, "score": 1.0},
                        {"item_id": 200, "score": 0.95},
                    ],
                    "cached": True,
                }
            ]
        }
    }


class HealthResponse(BaseModel):
    """Health check response."""

    status: str = Field(..., description="Service status (healthy/unhealthy)")
    model_loaded: bool = Field(..., description="Whether recommendation model is loaded")
    version: str = Field(..., description="API version")
    n_users: int | None = Field(None, description="Number of users in model")
    n_items: int | None = Field(None, description="Number of items in model")

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "status": "healthy",
                    "model_loaded": True,
                    "version": "1.0.0",
                    "n_users": 50000,
                    "n_items": 30000,
                }
            ]
        }
    }


class ErrorResponse(BaseModel):
    """Error response."""

    detail: str = Field(..., description="Error message")
    error_code: str = Field(..., description="Error code for programmatic handling")

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "detail": "User not found in model",
                    "error_code": "USER_NOT_FOUND",
                }
            ]
        }
    }
