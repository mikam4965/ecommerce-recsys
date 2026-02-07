"""Category co-occurrence heatmap analysis."""

from collections import Counter
from pathlib import Path

import plotly.graph_objects as go
import polars as pl
from loguru import logger


def get_top_categories(
    events: pl.DataFrame,
    top_n: int = 20,
    event_type: str = "transaction",
) -> list[str]:
    """Get top N categories by purchase frequency.

    Args:
        events: Events DataFrame.
        top_n: Number of top categories to return.
        event_type: Event type to filter (transaction, addtocart).

    Returns:
        List of top category IDs.
    """
    logger.info(f"Finding top {top_n} categories by {event_type} count")

    # Filter by event type and exclude unknown
    filtered = events.filter(
        (pl.col("event_type") == event_type) &
        (pl.col("category_id") != "unknown")
    )

    # Count by category
    category_counts = (
        filtered
        .group_by("category_id")
        .agg(pl.len().alias("count"))
        .sort("count", descending=True)
        .head(top_n)
    )

    categories = category_counts["category_id"].to_list()
    logger.info(f"Top {len(categories)} categories found")

    return categories


def category_cooccurrence_matrix(
    events: pl.DataFrame,
    categories: list[str] | None = None,
    event_type: str = "transaction",
    top_n: int = 20,
) -> tuple[list[list[int]], list[str]]:
    """Build category co-occurrence matrix.

    Args:
        events: Events DataFrame.
        categories: List of categories to include. If None, uses top_n.
        event_type: Event type to filter.
        top_n: Number of top categories if categories is None.

    Returns:
        Tuple of (matrix as 2D list, list of category labels).
    """
    logger.info("Building category co-occurrence matrix")

    # Get categories if not provided
    if categories is None:
        categories = get_top_categories(events, top_n, event_type)

    n_categories = len(categories)
    cat_to_idx = {cat: i for i, cat in enumerate(categories)}

    # Filter events
    filtered = events.filter(
        (pl.col("event_type") == event_type) &
        (pl.col("category_id").is_in(categories))
    )

    logger.info(f"Filtered events: {len(filtered):,}")

    # Group by session, get unique categories per session
    session_categories = (
        filtered
        .group_by("session_id")
        .agg(pl.col("category_id").unique().alias("categories"))
        .filter(pl.col("categories").list.len() >= 2)
    )

    logger.info(f"Sessions with 2+ categories: {len(session_categories):,}")

    # Count co-occurrences
    pair_counts: Counter = Counter()

    for row in session_categories.iter_rows(named=True):
        cats = sorted(row["categories"])
        for i, cat1 in enumerate(cats):
            for cat2 in cats[i + 1:]:
                if cat1 in cat_to_idx and cat2 in cat_to_idx:
                    pair_counts[(cat1, cat2)] += 1

    # Build symmetric matrix
    matrix = [[0] * n_categories for _ in range(n_categories)]

    for (cat1, cat2), count in pair_counts.items():
        i, j = cat_to_idx[cat1], cat_to_idx[cat2]
        matrix[i][j] = count
        matrix[j][i] = count

    # Add diagonal (self co-occurrence = total sessions with category)
    category_session_counts = (
        filtered
        .group_by("category_id")
        .agg(pl.col("session_id").n_unique().alias("sessions"))
    )

    for row in category_session_counts.iter_rows(named=True):
        cat = row["category_id"]
        if cat in cat_to_idx:
            idx = cat_to_idx[cat]
            matrix[idx][idx] = row["sessions"]

    logger.info(f"Matrix built: {n_categories}x{n_categories}")

    return matrix, categories


def normalize_matrix(
    matrix: list[list[int]],
    method: str = "max",
) -> list[list[float]]:
    """Normalize co-occurrence matrix.

    Args:
        matrix: Raw co-occurrence matrix.
        method: Normalization method (max, total, row).

    Returns:
        Normalized matrix.
    """
    n = len(matrix)

    if method == "max":
        max_val = max(max(row) for row in matrix)
        if max_val == 0:
            return [[0.0] * n for _ in range(n)]
        return [[val / max_val for val in row] for row in matrix]

    elif method == "total":
        total = sum(sum(row) for row in matrix)
        if total == 0:
            return [[0.0] * n for _ in range(n)]
        return [[val / total for val in row] for row in matrix]

    elif method == "row":
        result = []
        for row in matrix:
            row_sum = sum(row)
            if row_sum == 0:
                result.append([0.0] * n)
            else:
                result.append([val / row_sum for val in row])
        return result

    else:
        raise ValueError(f"Unknown method: {method}")


def get_top_pairs(
    matrix: list[list[int]],
    categories: list[str],
    top_n: int = 10,
) -> pl.DataFrame:
    """Extract top N category pairs by co-occurrence.

    Args:
        matrix: Co-occurrence matrix.
        categories: List of category labels.
        top_n: Number of top pairs to return.

    Returns:
        DataFrame with category pairs and counts.
    """
    pairs = []
    n = len(categories)

    for i in range(n):
        for j in range(i + 1, n):
            if matrix[i][j] > 0:
                pairs.append({
                    "category_1": categories[i],
                    "category_2": categories[j],
                    "count": matrix[i][j],
                })

    # Sort and get top N
    pairs.sort(key=lambda x: x["count"], reverse=True)
    top_pairs = pairs[:top_n]

    return pl.DataFrame(top_pairs)


def plot_heatmap(
    matrix: list[list[int]],
    categories: list[str],
    save_path: str | Path | None = None,
    title: str = "Санаттар бірлесу жылу картасы",
    show_values: bool = False,
    colorscale: str = "Blues",
) -> None:
    """Create interactive Plotly heatmap visualization.

    Args:
        matrix: Co-occurrence matrix.
        categories: List of category labels.
        save_path: Path to save HTML file.
        title: Chart title.
        show_values: Whether to show values on cells.
        colorscale: Plotly colorscale name.
    """
    logger.info("Creating heatmap visualization")

    # Create heatmap
    heatmap_kwargs = dict(
        z=matrix,
        x=categories,
        y=categories,
        colorscale=colorscale,
        hovertemplate=(
            "<b>Санат %{x}</b> + <b>Санат %{y}</b><br>"
            "Бірлескен сатып алулар: %{z}<extra></extra>"
        ),
    )

    if show_values:
        heatmap_kwargs["text"] = matrix
        heatmap_kwargs["texttemplate"] = "%{text}"
        heatmap_kwargs["textfont"] = {"size": 8}

    fig = go.Figure(data=go.Heatmap(**heatmap_kwargs))

    fig.update_layout(
        title=dict(text=title, x=0.5),
        xaxis=dict(
            title="Санат",
            tickangle=45,
            tickfont=dict(size=10),
        ),
        yaxis=dict(
            title="Санат",
            tickfont=dict(size=10),
        ),
        height=700,
        width=800,
    )

    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.write_html(str(save_path))
        logger.info(f"Saved heatmap to {save_path}")


def run_cooccurrence_analysis(
    events: pl.DataFrame,
    top_n_categories: int = 20,
    event_type: str = "transaction",
) -> dict:
    """Run complete co-occurrence analysis pipeline.

    Args:
        events: Events DataFrame.
        top_n_categories: Number of top categories to analyze.
        event_type: Event type to filter.

    Returns:
        Dictionary with analysis results.
    """
    logger.info("Starting co-occurrence analysis")

    # Get top categories
    categories = get_top_categories(events, top_n_categories, event_type)

    # Build matrix
    matrix, categories = category_cooccurrence_matrix(
        events,
        categories=categories,
        event_type=event_type,
    )

    # Get top pairs
    top_pairs = get_top_pairs(matrix, categories, top_n=10)

    # Normalize matrix
    normalized_matrix = normalize_matrix(matrix, method="max")

    results = {
        "categories": categories,
        "matrix": matrix,
        "normalized_matrix": normalized_matrix,
        "top_pairs": top_pairs,
        "n_categories": len(categories),
    }

    logger.info(f"Analysis complete: {len(categories)} categories analyzed")

    return results
