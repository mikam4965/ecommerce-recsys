"""API routes for recommendation service."""

from fastapi import APIRouter, Depends, HTTPException, Query, status
from loguru import logger

from src.api.schemas import (
    ColdStartRequest,
    ColdStartResponse,
    ErrorResponse,
    HealthResponse,
    PopularItemsResponse,
    RecommendationItem,
    RecommendationRequest,
    RecommendationResponse,
    SimilarItemsResponse,
)

# Router instance
router = APIRouter()


def get_model():
    """Dependency to get loaded model."""
    from src.api.main import app_state

    if app_state.model is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not loaded. Service is starting up.",
        )
    return app_state.model


def get_cache():
    """Dependency to get cache instance (Redis)."""
    from src.api.main import app_state

    return app_state.cache


def get_db():
    """Dependency to get database instance (SQLite)."""
    from src.api.main import app_state

    return app_state.db


@router.get(
    "/health",
    response_model=HealthResponse,
    tags=["System"],
    summary="Health check",
    description="Check service health and model status.",
)
async def health_check():
    """Health check endpoint."""
    from src.api.main import API_VERSION, app_state

    model = app_state.model
    model_loaded = model is not None

    n_users = None
    n_items = None

    if model_loaded:
        try:
            n_users = len(model.user_to_idx)
            n_items = len(model.item_to_idx)
        except Exception:
            pass

    return HealthResponse(
        status="healthy" if model_loaded else "starting",
        model_loaded=model_loaded,
        version=API_VERSION,
        n_users=n_users,
        n_items=n_items,
    )


@router.post(
    "/recommend",
    response_model=RecommendationResponse,
    responses={
        404: {"model": ErrorResponse, "description": "User not found"},
        503: {"model": ErrorResponse, "description": "Model not loaded"},
    },
    tags=["Recommendations"],
    summary="Get personalized recommendations",
    description="Generate personalized recommendations for a known user.",
)
async def get_recommendations(
    request: RecommendationRequest,
    model=Depends(get_model),
    cache=Depends(get_cache),
    db=Depends(get_db),
):
    """Get personalized recommendations for a user.

    Uses collaborative filtering with optional content-based features.
    Falls back to popular items if user not found.
    """
    user_id = request.user_id
    n = request.n_recommendations
    filter_seen = request.filter_seen

    logger.info(f"Recommendation request: user={user_id}, n={n}")

    # Check SQLite cache first (primary cache)
    if db is not None:
        cached = db.get_cached_recommendations(user_id)
        if cached and len(cached) >= n:
            logger.info(f"SQLite cache HIT for user {user_id}")
            return RecommendationResponse(
                user_id=user_id,
                recommendations=[RecommendationItem(**item) for item in cached[:n]],
                model_type="hybrid",
                is_fallback=False,
            )

    # Check Redis cache (fallback)
    if cache and cache.is_connected():
        cached = cache.get_user_recommendations(user_id, n)
        if cached:
            logger.info(f"Redis cache HIT for user {user_id}")
            return RecommendationResponse(
                user_id=user_id,
                recommendations=[RecommendationItem(**item) for item in cached],
                model_type="hybrid",
                is_fallback=False,
            )

    # Check if user exists in model
    is_known_user = user_id in model.user_to_idx

    if is_known_user:
        # Get recommendations with scores
        try:
            recs_with_scores = model.recommend_with_scores(
                user_id=user_id,
                n=n,
                filter_already_liked=filter_seen,
            )

            # Use rank-based scores: 1.0 for best, decreasing by position
            # This shows relative ranking clearly to API consumers
            recommendations = [
                RecommendationItem(
                    item_id=item_id,
                    score=round(1.0 - (i * 0.05), 4)  # 1.0, 0.95, 0.90, 0.85...
                )
                for i, (item_id, _) in enumerate(recs_with_scores)
            ]

            # Cache result in SQLite (primary)
            if db is not None:
                db.cache_recommendations(
                    user_id,
                    [{"item_id": r.item_id, "score": r.score} for r in recommendations],
                )

            # Cache result in Redis (secondary)
            if cache and cache.is_connected():
                cache.set_user_recommendations(
                    user_id,
                    [{"item_id": r.item_id, "score": r.score} for r in recommendations],
                    n,
                )

            logger.info(f"Generated {len(recommendations)} recommendations for user {user_id}")

            return RecommendationResponse(
                user_id=user_id,
                recommendations=recommendations,
                model_type="hybrid",
                is_fallback=False,
            )

        except Exception as e:
            logger.error(f"Error generating recommendations: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Error generating recommendations: {str(e)}",
            )
    else:
        # Fallback to popular items
        logger.warning(f"User {user_id} not found, using popular items fallback")

        # Get popular items from model
        popular_items = model.recommend_cold_start({}, n)

        recommendations = [
            RecommendationItem(item_id=item_id, score=1.0 - i * 0.01)
            for i, item_id in enumerate(popular_items)
        ]

        return RecommendationResponse(
            user_id=user_id,
            recommendations=recommendations,
            model_type="popular_fallback",
            is_fallback=True,
        )


@router.post(
    "/recommend/cold",
    response_model=ColdStartResponse,
    responses={
        400: {"model": ErrorResponse, "description": "Invalid request"},
        503: {"model": ErrorResponse, "description": "Model not loaded"},
    },
    tags=["Recommendations"],
    summary="Cold start recommendations",
    description="Get recommendations for a new user based on current session items.",
)
async def get_cold_start_recommendations(
    request: ColdStartRequest,
    model=Depends(get_model),
):
    """Get recommendations for a new user based on session.

    Uses item embeddings to find items similar to the session items.
    """
    session_items = request.session_items
    n = request.n_recommendations

    logger.info(f"Cold start request: session_items={len(session_items)}, n={n}")

    # Filter to known items
    known_items = [item for item in session_items if item in model.item_to_idx]

    if not known_items:
        logger.warning("No known items in session, using general cold start")
        # Fall back to popular items
        popular = model.recommend_cold_start({}, n)
        recommendations = [
            RecommendationItem(item_id=item_id, score=1.0 - i * 0.01)
            for i, item_id in enumerate(popular)
        ]
        return ColdStartResponse(
            session_items=session_items,
            recommendations=recommendations,
            known_items_count=0,
        )

    try:
        # Use session-based recommendations
        recommended_items = model.recommend_for_new_user(
            session_items=known_items,
            n=n,
        )

        # Create scores based on ranking
        recommendations = [
            RecommendationItem(item_id=item_id, score=1.0 - i * 0.05)
            for i, item_id in enumerate(recommended_items)
        ]

        logger.info(f"Generated {len(recommendations)} cold start recommendations")

        return ColdStartResponse(
            session_items=session_items,
            recommendations=recommendations,
            known_items_count=len(known_items),
        )

    except Exception as e:
        logger.error(f"Error generating cold start recommendations: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error generating recommendations: {str(e)}",
        )


@router.post(
    "/similar/{item_id}",
    response_model=SimilarItemsResponse,
    responses={
        404: {"model": ErrorResponse, "description": "Item not found"},
        503: {"model": ErrorResponse, "description": "Model not loaded"},
    },
    tags=["Recommendations"],
    summary="Get similar items",
    description="Find items similar to a given item based on embeddings.",
)
async def get_similar_items(
    item_id: int,
    n_similar: int = Query(default=10, ge=1, le=100, description="Number of similar items"),
    model=Depends(get_model),
    cache=Depends(get_cache),
):
    """Get items similar to a given item.

    Uses item embeddings from the collaborative filtering model.
    """
    logger.info(f"Similar items request: item={item_id}, n={n_similar}")

    # Check cache
    if cache and cache.is_connected():
        cached = cache.get_similar_items(item_id, n_similar)
        if cached:
            return SimilarItemsResponse(
                item_id=item_id,
                similar_items=[RecommendationItem(**item) for item in cached],
            )

    # Check if item exists
    if item_id not in model.item_to_idx:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Item {item_id} not found in model",
        )

    try:
        # Get similar items
        similar_ids = model.similar_items(item_id=item_id, n=n_similar)

        # Create response with scores
        similar_items = [
            RecommendationItem(item_id=sid, score=1.0 - i * 0.05)
            for i, sid in enumerate(similar_ids)
        ]

        # Cache result
        if cache and cache.is_connected():
            cache.set_similar_items(
                item_id,
                [{"item_id": s.item_id, "score": s.score} for s in similar_items],
                n_similar,
            )

        logger.info(f"Found {len(similar_items)} similar items for item {item_id}")

        return SimilarItemsResponse(
            item_id=item_id,
            similar_items=similar_items,
        )

    except Exception as e:
        logger.error(f"Error finding similar items: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error finding similar items: {str(e)}",
        )


@router.get(
    "/popular",
    response_model=PopularItemsResponse,
    responses={
        503: {"model": ErrorResponse, "description": "Model not loaded"},
    },
    tags=["Recommendations"],
    summary="Get popular items",
    description="Get globally popular items (fallback recommendations).",
)
async def get_popular_items(
    n: int = Query(default=10, ge=1, le=100, description="Number of popular items"),
    model=Depends(get_model),
    cache=Depends(get_cache),
):
    """Get globally popular items.

    Returns items sorted by popularity (interaction count).
    Useful as a fallback when personalization is not possible.
    """
    logger.info(f"Popular items request: n={n}")

    # Check cache
    from_cache = False
    if cache and cache.is_connected():
        cached = cache.get_popular_items(n)
        if cached:
            from_cache = True
            return PopularItemsResponse(
                items=[RecommendationItem(**item) for item in cached],
                cached=True,
            )

    try:
        # Get popular items using cold start (no user features)
        popular_ids = model.recommend_cold_start({}, n)

        items = [
            RecommendationItem(item_id=item_id, score=1.0 - i * 0.01)
            for i, item_id in enumerate(popular_ids)
        ]

        # Cache result
        if cache and cache.is_connected():
            cache.set_popular_items(
                [{"item_id": i.item_id, "score": i.score} for i in items],
                n,
            )

        logger.info(f"Returned {len(items)} popular items")

        return PopularItemsResponse(items=items, cached=from_cache)

    except Exception as e:
        logger.error(f"Error getting popular items: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error getting popular items: {str(e)}",
        )


@router.get(
    "/cache/stats",
    tags=["System"],
    summary="Cache statistics",
    description="Get cache statistics (SQLite and Redis).",
)
async def cache_stats(cache=Depends(get_cache), db=Depends(get_db)):
    """Get cache statistics."""
    stats = {}

    # SQLite stats
    if db is not None:
        stats["sqlite"] = db.get_stats()
        stats["sqlite"]["enabled"] = True
    else:
        stats["sqlite"] = {"enabled": False}

    # Redis stats
    if cache is not None and cache.is_connected():
        stats["redis"] = cache.get_stats()
    else:
        stats["redis"] = {"enabled": False}

    return stats


@router.post(
    "/cache/flush",
    tags=["System"],
    summary="Flush cache",
    description="Clear all cached recommendations (SQLite and Redis).",
)
async def flush_cache(cache=Depends(get_cache), db=Depends(get_db)):
    """Flush all cached recommendations."""
    results = {}

    # Flush SQLite
    if db is not None:
        count = db.flush_cache()
        results["sqlite"] = {"success": True, "entries_removed": count}
    else:
        results["sqlite"] = {"success": False, "message": "SQLite not configured"}

    # Flush Redis
    if cache is not None and cache.is_connected():
        success = cache.flush_all()
        results["redis"] = {"success": success}
    else:
        results["redis"] = {"success": False, "message": "Redis not configured"}

    return results
