"""Display top association rules by lift."""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import polars as pl

from src.analysis.association_rules import (
    find_frequent_itemsets,
    generate_rules,
    filter_rules,
    get_top_rules_by_lift,
)


def main():
    # Load data
    train = pl.read_parquet(project_root / "data/processed/train.parquet")

    # Prepare category-level transactions
    filtered = train.filter(pl.col("event_type").is_in(["transaction", "addtocart"]))
    filtered = filtered.filter(pl.col("category_id") != "unknown")

    transactions_df = (
        filtered
        .group_by("session_id")
        .agg(pl.col("category_id").unique().alias("categories"))
        .filter(pl.col("categories").list.len() >= 2)
    )

    transactions = []
    for row in transactions_df.iter_rows(named=True):
        cats = [int(c) for c in row["categories"] if c.isdigit()]
        if len(cats) >= 2:
            transactions.append(cats)

    # Find rules
    itemsets = find_frequent_itemsets(transactions, min_support=0.01, max_len=3)
    rules = generate_rules(itemsets, min_confidence=0.1)
    filtered_rules = filter_rules(rules, min_support=0.01, min_confidence=0.1, min_lift=1.0)
    top_rules = get_top_rules_by_lift(filtered_rules, top_n=10)

    print()
    print("=" * 80)
    print("TOP-10 ASSOCIATION RULES BY LIFT (category level)")
    print("=" * 80)
    print()

    for i, row in enumerate(top_rules.iter_rows(named=True), 1):
        ant_cats = row["antecedents"]
        cons_cats = row["consequents"]

        ant_str = ", ".join(str(c) for c in ant_cats)
        cons_str = ", ".join(str(c) for c in cons_cats)

        print(f"{i:2}. Users who bought [category {ant_str}] often buy [category {cons_str}] (lift = {row['lift']:.2f})")
        print(f"    Support: {row['support']:.2%} | Confidence: {row['confidence']:.2%}")
        print()

    print("=" * 80)
    print(f"Total rules found: {len(filtered_rules)}")
    print(f"Sessions analyzed: {len(transactions):,}")
    print("=" * 80)


if __name__ == "__main__":
    main()
