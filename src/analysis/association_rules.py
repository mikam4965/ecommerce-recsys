"""Association rules analysis for finding co-purchase patterns."""

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import polars as pl
from loguru import logger
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder


def prepare_transactions(
    events: pl.DataFrame,
    event_type: str = "transaction",
    user_col: str = "user_id",
    item_col: str = "item_id",
    session_col: str = "session_id",
) -> list[list[int]]:
    """Prepare transactions from events data.

    Args:
        events: Events DataFrame with user-item interactions.
        event_type: Event type to filter (transaction, addtocart, view).
        user_col: User column name.
        item_col: Item column name.
        session_col: Session column name for grouping.

    Returns:
        List of transactions (lists of item IDs).
    """
    logger.info(f"Preparing transactions from {len(events):,} events")
    logger.info(f"Filtering by event_type='{event_type}'")

    # Filter by event type
    filtered = events.filter(pl.col("event_type") == event_type)
    logger.info(f"Events after filtering: {len(filtered):,}")

    if len(filtered) == 0:
        logger.warning(f"No events with type '{event_type}' found")
        return []

    # Group items by session
    transactions_df = (
        filtered.group_by(session_col)
        .agg(pl.col(item_col).alias("items"))
    )

    # Filter to sessions with at least 2 items
    transactions_df = transactions_df.filter(pl.col("items").list.len() >= 2)
    logger.info(f"Sessions with 2+ items: {len(transactions_df):,}")

    # Convert to list of lists
    transactions = [row["items"] for row in transactions_df.iter_rows(named=True)]

    logger.info(f"Prepared {len(transactions):,} transactions")

    return transactions


def find_frequent_itemsets(
    transactions: list[list[int]],
    min_support: float = 0.01,
    max_len: int | None = None,
) -> pl.DataFrame:
    """Find frequent itemsets using Apriori algorithm.

    Args:
        transactions: List of transactions (lists of item IDs).
        min_support: Minimum support threshold.
        max_len: Maximum length of itemsets (None = unlimited).

    Returns:
        Polars DataFrame with itemsets and support.
    """
    if not transactions:
        logger.warning("No transactions provided")
        return pl.DataFrame({"itemsets": [], "support": []})

    logger.info(f"Finding frequent itemsets with min_support={min_support}")
    logger.info(f"Processing {len(transactions):,} transactions")

    # Convert item IDs to strings for mlxtend
    transactions_str = [[str(item) for item in t] for t in transactions]

    # Encode transactions
    te = TransactionEncoder()
    te_ary = te.fit(transactions_str).transform(transactions_str)
    df_encoded = pd.DataFrame(te_ary, columns=te.columns_)

    logger.info(f"Encoded matrix shape: {df_encoded.shape}")

    # Run Apriori
    frequent_itemsets = apriori(
        df_encoded,
        min_support=min_support,
        use_colnames=True,
        max_len=max_len,
    )

    logger.info(f"Found {len(frequent_itemsets):,} frequent itemsets")

    if len(frequent_itemsets) == 0:
        return pl.DataFrame({"itemsets": [], "support": []})

    # Convert frozenset to list of ints
    frequent_itemsets["itemsets"] = frequent_itemsets["itemsets"].apply(
        lambda x: [int(i) for i in x]
    )

    # Convert to Polars
    result = pl.DataFrame({
        "itemsets": frequent_itemsets["itemsets"].tolist(),
        "support": frequent_itemsets["support"].tolist(),
    })

    return result


def generate_rules(
    frequent_itemsets: pl.DataFrame,
    min_confidence: float = 0.1,
    metric: str = "confidence",
) -> pl.DataFrame:
    """Generate association rules from frequent itemsets.

    Args:
        frequent_itemsets: DataFrame with itemsets and support.
        min_confidence: Minimum confidence threshold.
        metric: Metric for rule generation (confidence, lift, etc.).

    Returns:
        Polars DataFrame with association rules.
    """
    if len(frequent_itemsets) == 0:
        logger.warning("No frequent itemsets provided")
        return pl.DataFrame({
            "antecedents": [],
            "consequents": [],
            "support": [],
            "confidence": [],
            "lift": [],
        })

    logger.info(f"Generating rules with min_{metric}={min_confidence}")

    # Convert Polars to pandas for mlxtend
    itemsets_list = frequent_itemsets["itemsets"].to_list()
    support_list = frequent_itemsets["support"].to_list()

    # Convert lists back to frozensets for mlxtend
    df_itemsets = pd.DataFrame({
        "itemsets": [frozenset(str(i) for i in items) for items in itemsets_list],
        "support": support_list,
    })

    # Generate rules
    try:
        rules = association_rules(
            df_itemsets,
            metric=metric,
            min_threshold=min_confidence,
        )
    except Exception as e:
        logger.warning(f"Could not generate rules: {e}")
        return pl.DataFrame({
            "antecedents": [],
            "consequents": [],
            "support": [],
            "confidence": [],
            "lift": [],
        })

    logger.info(f"Generated {len(rules):,} association rules")

    if len(rules) == 0:
        return pl.DataFrame({
            "antecedents": [],
            "consequents": [],
            "support": [],
            "confidence": [],
            "lift": [],
        })

    # Convert frozensets to lists of ints
    rules["antecedents"] = rules["antecedents"].apply(
        lambda x: [int(i) for i in x]
    )
    rules["consequents"] = rules["consequents"].apply(
        lambda x: [int(i) for i in x]
    )

    # Convert to Polars
    result = pl.DataFrame({
        "antecedents": rules["antecedents"].tolist(),
        "consequents": rules["consequents"].tolist(),
        "support": rules["support"].tolist(),
        "confidence": rules["confidence"].tolist(),
        "lift": rules["lift"].tolist(),
    })

    return result


def filter_rules(
    rules: pl.DataFrame,
    min_support: float = 0.01,
    min_confidence: float = 0.1,
    min_lift: float = 1.0,
) -> pl.DataFrame:
    """Filter association rules by thresholds.

    Args:
        rules: DataFrame with association rules.
        min_support: Minimum support threshold.
        min_confidence: Minimum confidence threshold.
        min_lift: Minimum lift threshold.

    Returns:
        Filtered and sorted DataFrame.
    """
    if len(rules) == 0:
        return rules

    logger.info(
        f"Filtering rules: support>={min_support}, "
        f"confidence>={min_confidence}, lift>={min_lift}"
    )

    filtered = rules.filter(
        (pl.col("support") >= min_support) &
        (pl.col("confidence") >= min_confidence) &
        (pl.col("lift") >= min_lift)
    ).sort("lift", descending=True)

    logger.info(f"Rules after filtering: {len(filtered):,}")

    return filtered


def get_top_rules_by_lift(
    rules: pl.DataFrame,
    top_n: int = 10,
) -> pl.DataFrame:
    """Get top N rules by lift.

    Args:
        rules: DataFrame with association rules.
        top_n: Number of top rules to return.

    Returns:
        Top N rules sorted by lift.
    """
    if len(rules) == 0:
        return rules

    return rules.sort("lift", descending=True).head(top_n)


def visualize_top_rules(
    rules: pl.DataFrame,
    top_n: int = 20,
    save_path: str | Path | None = None,
    figsize: tuple[int, int] = (12, 8),
) -> None:
    """Visualize top association rules.

    Args:
        rules: DataFrame with association rules.
        top_n: Number of top rules to visualize.
        save_path: Path to save figure (None = display only).
        figsize: Figure size.
    """
    if len(rules) == 0:
        logger.warning("No rules to visualize")
        return

    top_rules = rules.sort("lift", descending=True).head(top_n)

    # Create rule labels
    labels = []
    for row in top_rules.iter_rows(named=True):
        ant = ", ".join(map(str, row["antecedents"]))
        cons = ", ".join(map(str, row["consequents"]))
        labels.append(f"{ant} -> {cons}")

    lifts = top_rules["lift"].to_list()
    confidences = top_rules["confidence"].to_list()
    supports = top_rules["support"].to_list()

    # Create figure with 3 subplots
    fig, axes = plt.subplots(1, 3, figsize=figsize)

    # Lift bar chart
    y_pos = np.arange(len(labels))

    axes[0].barh(y_pos, lifts, color="steelblue")
    axes[0].set_yticks(y_pos)
    axes[0].set_yticklabels(labels, fontsize=8)
    axes[0].set_xlabel("Lift")
    axes[0].set_title("Top Rules by Lift")
    axes[0].invert_yaxis()

    # Confidence bar chart
    axes[1].barh(y_pos, confidences, color="forestgreen")
    axes[1].set_yticks(y_pos)
    axes[1].set_yticklabels(labels, fontsize=8)
    axes[1].set_xlabel("Confidence")
    axes[1].set_title("Confidence")
    axes[1].invert_yaxis()

    # Support bar chart
    axes[2].barh(y_pos, supports, color="coral")
    axes[2].set_yticks(y_pos)
    axes[2].set_yticklabels(labels, fontsize=8)
    axes[2].set_xlabel("Support")
    axes[2].set_title("Support")
    axes[2].invert_yaxis()

    plt.tight_layout()

    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info(f"Saved visualization to {save_path}")

    plt.close()


def run_association_analysis(
    events: pl.DataFrame,
    event_type: str = "transaction",
    min_support: float = 0.01,
    min_confidence: float = 0.1,
    min_lift: float = 1.0,
    max_itemset_len: int | None = 3,
) -> dict[str, Any]:
    """Run full association rules analysis pipeline.

    Args:
        events: Events DataFrame.
        event_type: Event type to analyze.
        min_support: Minimum support threshold.
        min_confidence: Minimum confidence threshold.
        min_lift: Minimum lift threshold.
        max_itemset_len: Maximum itemset length.

    Returns:
        Dictionary with analysis results.
    """
    logger.info("Starting association rules analysis")

    # Prepare transactions
    transactions = prepare_transactions(events, event_type=event_type)

    if not transactions:
        logger.warning("No transactions to analyze")
        return {
            "n_transactions": 0,
            "n_itemsets": 0,
            "n_rules": 0,
            "frequent_itemsets": pl.DataFrame(),
            "rules": pl.DataFrame(),
        }

    # Find frequent itemsets
    itemsets = find_frequent_itemsets(
        transactions,
        min_support=min_support,
        max_len=max_itemset_len,
    )

    # Generate rules
    rules = generate_rules(itemsets, min_confidence=min_confidence)

    # Filter rules
    filtered_rules = filter_rules(
        rules,
        min_support=min_support,
        min_confidence=min_confidence,
        min_lift=min_lift,
    )

    results = {
        "n_transactions": len(transactions),
        "n_itemsets": len(itemsets),
        "n_rules": len(filtered_rules),
        "frequent_itemsets": itemsets,
        "rules": filtered_rules,
    }

    logger.info(f"Analysis complete: {results['n_rules']} rules found")

    return results
