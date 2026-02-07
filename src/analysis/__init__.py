"""User behavior analysis."""

from src.analysis.association_rules import (
    filter_rules,
    find_frequent_itemsets,
    generate_rules,
    get_top_rules_by_lift,
    prepare_transactions,
    run_association_analysis,
    visualize_top_rules,
)
from src.analysis.funnel import (
    calculate_funnel,
    compare_segments,
    conversion_rates,
    funnel_by_category,
    funnel_by_segment,
    run_funnel_analysis,
    visualize_funnel,
    visualize_segment_comparison,
)
from src.analysis.heatmaps import (
    category_cooccurrence_matrix,
    get_top_categories,
    get_top_pairs,
    normalize_matrix,
    plot_heatmap,
    run_cooccurrence_analysis,
)
from src.analysis.rfm import (
    calculate_rfm,
    get_segment_stats,
    run_rfm_analysis,
    score_rfm,
    segment_users,
    visualize_segments,
)

__all__ = [
    # Association rules
    "prepare_transactions",
    "find_frequent_itemsets",
    "generate_rules",
    "filter_rules",
    "get_top_rules_by_lift",
    "visualize_top_rules",
    "run_association_analysis",
    # RFM segmentation
    "calculate_rfm",
    "score_rfm",
    "segment_users",
    "get_segment_stats",
    "visualize_segments",
    "run_rfm_analysis",
    # Conversion funnel
    "calculate_funnel",
    "conversion_rates",
    "funnel_by_category",
    "funnel_by_segment",
    "compare_segments",
    "visualize_funnel",
    "visualize_segment_comparison",
    "run_funnel_analysis",
    # Category heatmaps
    "get_top_categories",
    "category_cooccurrence_matrix",
    "normalize_matrix",
    "get_top_pairs",
    "plot_heatmap",
    "run_cooccurrence_analysis",
]
