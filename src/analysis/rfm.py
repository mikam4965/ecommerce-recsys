"""RFM segmentation for user value analysis."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from loguru import logger


# Segment definitions based on RFM scores
SEGMENT_MAP = {
    # Champions - Best customers
    "555": "Champions", "554": "Champions", "545": "Champions",
    "544": "Champions", "455": "Champions", "454": "Champions",
    "445": "Champions", "535": "Champions", "534": "Champions",

    # Loyal Customers
    "543": "Loyal Customers", "533": "Loyal Customers", "443": "Loyal Customers",
    "444": "Loyal Customers", "435": "Loyal Customers", "434": "Loyal Customers",
    "453": "Loyal Customers", "433": "Loyal Customers", "442": "Loyal Customers",

    # Potential Loyalists
    "553": "Potential Loyalists", "551": "Potential Loyalists", "552": "Potential Loyalists",
    "541": "Potential Loyalists", "542": "Potential Loyalists", "532": "Potential Loyalists",
    "531": "Potential Loyalists", "452": "Potential Loyalists", "451": "Potential Loyalists",

    # New Customers
    "512": "New Customers", "511": "New Customers", "422": "New Customers",
    "421": "New Customers", "412": "New Customers", "411": "New Customers",
    "311": "New Customers", "312": "New Customers",

    # Promising
    "525": "Promising", "524": "Promising", "523": "Promising",
    "522": "Promising", "521": "Promising", "515": "Promising",
    "514": "Promising", "513": "Promising",

    # Need Attention
    "535": "Need Attention", "443": "Need Attention", "434": "Need Attention",
    "343": "Need Attention", "334": "Need Attention", "325": "Need Attention",
    "324": "Need Attention", "333": "Need Attention",

    # About To Sleep
    "331": "About To Sleep", "321": "About To Sleep", "313": "About To Sleep",
    "221": "About To Sleep", "213": "About To Sleep", "212": "About To Sleep",

    # At Risk
    "255": "At Risk", "254": "At Risk", "245": "At Risk",
    "244": "At Risk", "235": "At Risk", "234": "At Risk",
    "225": "At Risk", "224": "At Risk", "153": "At Risk",
    "152": "At Risk", "145": "At Risk", "143": "At Risk",

    # Can't Lose Them
    "155": "Can't Lose Them", "154": "Can't Lose Them", "144": "Can't Lose Them",
    "135": "Can't Lose Them", "134": "Can't Lose Them", "125": "Can't Lose Them",
    "124": "Can't Lose Them", "253": "Can't Lose Them", "252": "Can't Lose Them",
    "243": "Can't Lose Them", "242": "Can't Lose Them", "233": "Can't Lose Them",
    "232": "Can't Lose Them",

    # Hibernating
    "332": "Hibernating", "322": "Hibernating", "231": "Hibernating",
    "222": "Hibernating", "223": "Hibernating", "132": "Hibernating",
    "123": "Hibernating", "122": "Hibernating", "113": "Hibernating",
    "214": "Hibernating", "215": "Hibernating",

    # Lost
    "111": "Lost", "112": "Lost", "121": "Lost",
    "131": "Lost", "141": "Lost", "151": "Lost",
    "211": "Lost", "114": "Lost", "115": "Lost",
}


def calculate_rfm(
    events: pl.DataFrame,
    reference_date: int | None = None,
) -> pl.DataFrame:
    """Calculate RFM metrics for each user.

    Args:
        events: Events DataFrame with transactions.
        reference_date: Reference timestamp for recency calculation.
            If None, uses max timestamp from data.

    Returns:
        DataFrame with user_id, recency, frequency, monetary.
    """
    logger.info("Calculating RFM metrics")

    # Filter to transactions only
    transactions = events.filter(pl.col("event_type") == "transaction")
    logger.info(f"Transaction events: {len(transactions):,}")

    if len(transactions) == 0:
        logger.warning("No transaction events found")
        return pl.DataFrame({
            "user_id": [],
            "recency": [],
            "frequency": [],
            "monetary": [],
        })

    # Get reference date (max timestamp)
    if reference_date is None:
        reference_date = transactions["timestamp"].max()
    logger.info(f"Reference date (timestamp): {reference_date}")

    # Calculate RFM metrics per user
    rfm = transactions.group_by("user_id").agg([
        # Recency: days since last purchase
        ((reference_date - pl.col("timestamp").max()) / (1000 * 60 * 60 * 24))
        .cast(pl.Int64)
        .alias("recency"),

        # Frequency: number of unique purchase sessions
        pl.col("session_id").n_unique().alias("frequency"),

        # Monetary: total items purchased (since no price data)
        pl.len().alias("monetary"),
    ])

    logger.info(f"Users with transactions: {len(rfm):,}")
    logger.info(f"Recency range: {rfm['recency'].min()} - {rfm['recency'].max()} days")
    logger.info(f"Frequency range: {rfm['frequency'].min()} - {rfm['frequency'].max()}")
    logger.info(f"Monetary range: {rfm['monetary'].min()} - {rfm['monetary'].max()}")

    return rfm


def score_rfm(
    rfm_df: pl.DataFrame,
    n_bins: int = 5,
) -> pl.DataFrame:
    """Assign RFM scores (1-5) based on quantiles.

    Args:
        rfm_df: DataFrame with recency, frequency, monetary.
        n_bins: Number of score bins (default 5).

    Returns:
        DataFrame with added r_score, f_score, m_score, rfm_score columns.
    """
    logger.info(f"Scoring RFM metrics with {n_bins} bins")

    # Calculate quantile boundaries for each metric
    r_quantiles = [rfm_df["recency"].quantile(i / n_bins) for i in range(1, n_bins)]
    f_quantiles = [rfm_df["frequency"].quantile(i / n_bins) for i in range(1, n_bins)]
    m_quantiles = [rfm_df["monetary"].quantile(i / n_bins) for i in range(1, n_bins)]

    logger.info(f"Recency quantiles: {r_quantiles}")
    logger.info(f"Frequency quantiles: {f_quantiles}")
    logger.info(f"Monetary quantiles: {m_quantiles}")

    def assign_score(value: int, quantiles: list, reverse: bool = False) -> int:
        """Assign score 1-5 based on quantiles."""
        score = 1
        for q in quantiles:
            if q is not None and value > q:
                score += 1
        if reverse:
            score = n_bins + 1 - score
        return score

    # Apply scoring
    r_scores = []
    f_scores = []
    m_scores = []

    for row in rfm_df.iter_rows(named=True):
        # Recency: lower is better (reverse scoring)
        r_scores.append(assign_score(row["recency"], r_quantiles, reverse=True))
        # Frequency: higher is better
        f_scores.append(assign_score(row["frequency"], f_quantiles, reverse=False))
        # Monetary: higher is better
        m_scores.append(assign_score(row["monetary"], m_quantiles, reverse=False))

    # Add scores to dataframe
    result = rfm_df.with_columns([
        pl.Series("r_score", r_scores).cast(pl.Int64),
        pl.Series("f_score", f_scores).cast(pl.Int64),
        pl.Series("m_score", m_scores).cast(pl.Int64),
    ])

    # Create combined RFM score string
    result = result.with_columns(
        (pl.col("r_score").cast(pl.Utf8) +
         pl.col("f_score").cast(pl.Utf8) +
         pl.col("m_score").cast(pl.Utf8)).alias("rfm_score")
    )

    logger.info("RFM scoring complete")

    return result


def segment_users(rfm_df: pl.DataFrame) -> pl.DataFrame:
    """Assign segment labels based on RFM scores.

    Args:
        rfm_df: DataFrame with rfm_score column.

    Returns:
        DataFrame with added segment column.
    """
    logger.info("Assigning user segments")

    # Map RFM scores to segments
    segments = []
    for row in rfm_df.iter_rows(named=True):
        rfm_score = row["rfm_score"]
        segment = SEGMENT_MAP.get(rfm_score, "Other")
        segments.append(segment)

    result = rfm_df.with_columns(
        pl.Series("segment", segments)
    )

    # Log segment distribution
    segment_counts = result.group_by("segment").agg(pl.len().alias("count"))
    logger.info("Segment distribution:")
    for row in segment_counts.sort("count", descending=True).iter_rows(named=True):
        logger.info(f"  {row['segment']}: {row['count']:,}")

    return result


def get_segment_stats(rfm_df: pl.DataFrame) -> pl.DataFrame:
    """Calculate statistics for each segment.

    Args:
        rfm_df: DataFrame with segment column.

    Returns:
        DataFrame with segment statistics.
    """
    total_users = len(rfm_df)

    stats = rfm_df.group_by("segment").agg([
        pl.len().alias("count"),
        (pl.len() / total_users * 100).alias("percentage"),
        pl.col("recency").mean().alias("avg_recency"),
        pl.col("frequency").mean().alias("avg_frequency"),
        pl.col("monetary").mean().alias("avg_monetary"),
        pl.col("r_score").mean().alias("avg_r_score"),
        pl.col("f_score").mean().alias("avg_f_score"),
        pl.col("m_score").mean().alias("avg_m_score"),
    ]).sort("count", descending=True)

    return stats


def visualize_segments(
    rfm_df: pl.DataFrame,
    save_path: str | Path | None = None,
    figsize: tuple[int, int] = (14, 10),
) -> None:
    """Visualize RFM segment distribution.

    Args:
        rfm_df: DataFrame with segment column.
        save_path: Path to save figure (None = display only).
        figsize: Figure size.
    """
    if len(rfm_df) == 0:
        logger.warning("No data to visualize")
        return

    # Get segment counts
    segment_counts = (
        rfm_df.group_by("segment")
        .agg(pl.len().alias("count"))
        .sort("count", descending=True)
    )

    segments = segment_counts["segment"].to_list()
    counts = segment_counts["count"].to_list()

    # Get segment stats for bar chart
    stats = get_segment_stats(rfm_df)

    # Create figure with 2 subplots
    fig, axes = plt.subplots(1, 2, figsize=figsize)

    # Pie chart
    colors = plt.cm.Set3(np.linspace(0, 1, len(segments)))
    wedges, texts, autotexts = axes[0].pie(
        counts,
        labels=segments,
        autopct=lambda pct: f'{pct:.1f}%' if pct > 3 else '',
        colors=colors,
        startangle=90,
    )
    axes[0].set_title("Сегмент бойынша пайдаланушылар таралуы", fontsize=14)

    # Bar chart of average metrics by segment
    x = np.arange(len(stats))
    width = 0.25

    seg_names = stats["segment"].to_list()
    avg_r = stats["avg_recency"].to_list()
    avg_f = stats["avg_frequency"].to_list()
    avg_m = stats["avg_monetary"].to_list()

    # Normalize for visualization
    max_r = max(avg_r) if avg_r else 1
    max_f = max(avg_f) if avg_f else 1
    max_m = max(avg_m) if avg_m else 1

    axes[1].barh(x - width, [r / max_r for r in avg_r], width, label='Жақындық (норм.)', color='coral')
    axes[1].barh(x, [f / max_f for f in avg_f], width, label='Жиілік (норм.)', color='steelblue')
    axes[1].barh(x + width, [m / max_m for m in avg_m], width, label='Ақшалай (норм.)', color='forestgreen')

    axes[1].set_yticks(x)
    axes[1].set_yticklabels(seg_names, fontsize=9)
    axes[1].set_xlabel("Нормаланған мән")
    axes[1].set_title("Сегмент бойынша орташа RFM көрсеткіштері", fontsize=14)
    axes[1].legend(loc='lower right')
    axes[1].invert_yaxis()

    plt.tight_layout()

    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info(f"Saved visualization to {save_path}")

    plt.close()


def run_rfm_analysis(
    events: pl.DataFrame,
    reference_date: int | None = None,
    n_bins: int = 5,
) -> dict:
    """Run full RFM analysis pipeline.

    Args:
        events: Events DataFrame.
        reference_date: Reference timestamp for recency.
        n_bins: Number of score bins.

    Returns:
        Dictionary with analysis results.
    """
    logger.info("Starting RFM analysis")

    # Calculate RFM metrics
    rfm = calculate_rfm(events, reference_date)

    if len(rfm) == 0:
        return {
            "n_users": 0,
            "rfm_data": pl.DataFrame(),
            "segment_stats": pl.DataFrame(),
        }

    # Score RFM
    rfm_scored = score_rfm(rfm, n_bins)

    # Segment users
    rfm_segmented = segment_users(rfm_scored)

    # Get statistics
    segment_stats = get_segment_stats(rfm_segmented)

    results = {
        "n_users": len(rfm_segmented),
        "rfm_data": rfm_segmented,
        "segment_stats": segment_stats,
    }

    logger.info(f"RFM analysis complete: {results['n_users']} users segmented")

    return results
