"""Conversion funnel analysis for e-commerce user behavior."""

from pathlib import Path

import plotly.graph_objects as go
import polars as pl
from loguru import logger


def calculate_funnel(events: pl.DataFrame) -> pl.DataFrame:
    """Calculate conversion funnel metrics.

    Args:
        events: Events DataFrame with user interactions.

    Returns:
        DataFrame with stage, users, percentage columns.
    """
    logger.info("Calculating conversion funnel")

    # Count unique users at each stage
    view_users = events.filter(pl.col("event_type") == "view")["user_id"].n_unique()
    cart_users = events.filter(pl.col("event_type") == "addtocart")["user_id"].n_unique()
    purchase_users = events.filter(pl.col("event_type") == "transaction")["user_id"].n_unique()

    logger.info(f"Views: {view_users:,} users")
    logger.info(f"Cart: {cart_users:,} users")
    logger.info(f"Purchase: {purchase_users:,} users")

    # Create funnel dataframe
    funnel = pl.DataFrame({
        "stage": ["Қарау", "Себет", "Сатып алу"],
        "users": [view_users, cart_users, purchase_users],
    })

    # Calculate percentage from views
    funnel = funnel.with_columns(
        (pl.col("users") / view_users * 100).alias("percentage")
    )

    return funnel


def conversion_rates(funnel_df: pl.DataFrame) -> dict:
    """Calculate conversion rates between funnel stages.

    Args:
        funnel_df: DataFrame from calculate_funnel().

    Returns:
        Dictionary with conversion rates.
    """
    users = funnel_df["users"].to_list()
    views, carts, purchases = users[0], users[1], users[2]

    rates = {
        "view_to_cart": carts / views * 100 if views > 0 else 0,
        "cart_to_purchase": purchases / carts * 100 if carts > 0 else 0,
        "view_to_purchase": purchases / views * 100 if views > 0 else 0,
    }

    logger.info(f"View → Cart: {rates['view_to_cart']:.2f}%")
    logger.info(f"Cart → Purchase: {rates['cart_to_purchase']:.2f}%")
    logger.info(f"View → Purchase: {rates['view_to_purchase']:.2f}%")

    return rates


def funnel_by_category(
    events: pl.DataFrame,
    top_n: int = 10,
    min_views: int = 100,
) -> pl.DataFrame:
    """Calculate funnel metrics by category.

    Args:
        events: Events DataFrame.
        top_n: Number of top categories to return.
        min_views: Minimum views to include category.

    Returns:
        DataFrame with category funnel metrics.
    """
    logger.info("Calculating funnel by category")

    # Filter out unknown categories
    events_filtered = events.filter(pl.col("category_id") != "unknown")

    # Count unique users per category and event type
    category_events = (
        events_filtered
        .group_by(["category_id", "event_type"])
        .agg(pl.col("user_id").n_unique().alias("users"))
    )

    # Pivot to get views, carts, purchases as columns
    category_funnel = category_events.pivot(
        index="category_id",
        columns="event_type",
        values="users",
    )

    # Rename columns and handle missing
    category_funnel = category_funnel.rename({
        "view": "views",
        "addtocart": "carts",
        "transaction": "purchases",
    })

    # Fill nulls with 0
    for col in ["views", "carts", "purchases"]:
        if col not in category_funnel.columns:
            category_funnel = category_funnel.with_columns(pl.lit(0).alias(col))
        else:
            category_funnel = category_funnel.with_columns(
                pl.col(col).fill_null(0)
            )

    # Filter by minimum views
    category_funnel = category_funnel.filter(pl.col("views") >= min_views)

    # Calculate conversion rates
    category_funnel = category_funnel.with_columns([
        (pl.col("carts") / pl.col("views") * 100).alias("view_to_cart"),
        (pl.col("purchases") / pl.col("carts") * 100).fill_null(0).alias("cart_to_purchase"),
        (pl.col("purchases") / pl.col("views") * 100).alias("view_to_purchase"),
    ])

    # Sort by view_to_purchase and get top N
    result = (
        category_funnel
        .sort("view_to_purchase", descending=True)
        .head(top_n)
    )

    logger.info(f"Found {len(category_funnel)} categories, returning top {top_n}")

    return result


def funnel_by_segment(
    events: pl.DataFrame,
    rfm_df: pl.DataFrame,
) -> pl.DataFrame:
    """Calculate funnel metrics by RFM segment.

    Args:
        events: Events DataFrame.
        rfm_df: RFM segmentation DataFrame with user_id and segment.

    Returns:
        DataFrame with segment funnel metrics.
    """
    logger.info("Calculating funnel by RFM segment")

    # Join events with RFM segments
    events_with_segment = events.join(
        rfm_df.select(["user_id", "segment"]),
        on="user_id",
        how="left",
    )

    # Fill null segments (users without transactions)
    events_with_segment = events_with_segment.with_columns(
        pl.col("segment").fill_null("No Segment")
    )

    # Count unique users per segment and event type
    segment_events = (
        events_with_segment
        .group_by(["segment", "event_type"])
        .agg(pl.col("user_id").n_unique().alias("users"))
    )

    # Pivot to get views, carts, purchases as columns
    segment_funnel = segment_events.pivot(
        index="segment",
        columns="event_type",
        values="users",
    )

    # Rename columns
    rename_map = {}
    if "view" in segment_funnel.columns:
        rename_map["view"] = "views"
    if "addtocart" in segment_funnel.columns:
        rename_map["addtocart"] = "carts"
    if "transaction" in segment_funnel.columns:
        rename_map["transaction"] = "purchases"

    segment_funnel = segment_funnel.rename(rename_map)

    # Fill nulls with 0
    for col in ["views", "carts", "purchases"]:
        if col not in segment_funnel.columns:
            segment_funnel = segment_funnel.with_columns(pl.lit(0).alias(col))
        else:
            segment_funnel = segment_funnel.with_columns(
                pl.col(col).fill_null(0)
            )

    # Calculate conversion rates
    segment_funnel = segment_funnel.with_columns([
        (pl.col("carts") / pl.col("views") * 100).alias("view_to_cart"),
        (pl.col("purchases") / pl.col("carts") * 100).fill_null(0).alias("cart_to_purchase"),
        (pl.col("purchases") / pl.col("views") * 100).alias("view_to_purchase"),
    ])

    # Sort by view_to_purchase
    result = segment_funnel.sort("view_to_purchase", descending=True)

    logger.info(f"Calculated funnel for {len(result)} segments")

    return result


def compare_segments(
    segment_funnel: pl.DataFrame,
    segments: list[str],
) -> pl.DataFrame:
    """Compare funnel metrics for selected segments.

    Args:
        segment_funnel: DataFrame from funnel_by_segment().
        segments: List of segment names to compare.

    Returns:
        DataFrame with compared segments.
    """
    return segment_funnel.filter(pl.col("segment").is_in(segments))


def visualize_funnel(
    funnel_df: pl.DataFrame,
    save_path: str | Path | None = None,
    title: str = "Конверсия воронкасы",
) -> None:
    """Visualize conversion funnel using Plotly.

    Args:
        funnel_df: DataFrame with stage, users columns.
        save_path: Path to save HTML file.
        title: Chart title.
    """
    stages = funnel_df["stage"].to_list()
    users = funnel_df["users"].to_list()

    fig = go.Figure(go.Funnel(
        y=stages,
        x=users,
        textposition="inside",
        textinfo="value+percent initial",
        marker=dict(
            color=["royalblue", "orange", "green"],
        ),
    ))

    fig.update_layout(
        title=dict(text=title, x=0.5),
        font=dict(size=14),
    )

    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.write_html(str(save_path))
        logger.info(f"Saved funnel visualization to {save_path}")


def visualize_segment_comparison(
    segment_funnel: pl.DataFrame,
    segments: list[str],
    save_path: str | Path | None = None,
) -> None:
    """Visualize funnel comparison for multiple segments.

    Args:
        segment_funnel: DataFrame from funnel_by_segment().
        segments: List of segments to compare.
        save_path: Path to save HTML file.
    """
    comparison = compare_segments(segment_funnel, segments)

    fig = go.Figure()

    colors = ["royalblue", "crimson", "green", "orange", "purple"]

    for i, row in enumerate(comparison.iter_rows(named=True)):
        segment = row["segment"]
        fig.add_trace(go.Funnel(
            name=segment,
            y=["Қарау", "Себет", "Сатып алу"],
            x=[row["views"], row["carts"], row["purchases"]],
            textposition="inside",
            textinfo="value+percent initial",
            marker=dict(color=colors[i % len(colors)]),
        ))

    fig.update_layout(
        title=dict(text=f"Воронка салыстыруы: {' vs '.join(segments)}", x=0.5),
        font=dict(size=12),
    )

    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.write_html(str(save_path))
        logger.info(f"Saved segment comparison to {save_path}")


def run_funnel_analysis(
    events: pl.DataFrame,
    rfm_df: pl.DataFrame | None = None,
) -> dict:
    """Run complete funnel analysis pipeline.

    Args:
        events: Events DataFrame.
        rfm_df: Optional RFM segmentation DataFrame.

    Returns:
        Dictionary with analysis results.
    """
    logger.info("Starting funnel analysis")

    # Calculate overall funnel
    funnel = calculate_funnel(events)
    rates = conversion_rates(funnel)

    # Calculate category funnel
    category_funnel = funnel_by_category(events, top_n=20, min_views=100)

    # Calculate segment funnel if RFM data provided
    segment_funnel = None
    if rfm_df is not None:
        segment_funnel = funnel_by_segment(events, rfm_df)

    results = {
        "funnel": funnel,
        "rates": rates,
        "category_funnel": category_funnel,
        "segment_funnel": segment_funnel,
    }

    logger.info("Funnel analysis complete")

    return results
