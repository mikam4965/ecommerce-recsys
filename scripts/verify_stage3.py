"""Verify Stage 3 (Hybrid Core) implementation.

This script:
1. Trains 5 model variants (ALS, Hybrid CF/Items/Full, TwoStage)
2. Evaluates on all users and by cold/warm/hot groups
3. Creates comparison tables and visualizations
4. Checks 10% improvement requirement over ALS baseline
"""

import sys
from pathlib import Path

import numpy as np
import polars as pl
from loguru import logger

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.evaluation.evaluator import RecommenderEvaluator
from src.evaluation.cold_start import (
    split_by_interaction_count,
    compare_cold_start_models,
)
from src.models.collaborative import ALSRecommender
from src.models.hybrid import HybridRecommender
from src.models.pipeline import TwoStageRecommender


def load_data():
    """Load train, test, and RFM data."""
    logger.info("Loading data...")

    train = pl.read_parquet(project_root / "data/processed/train.parquet")
    test = pl.read_parquet(project_root / "data/processed/test.parquet")

    logger.info(f"Train events: {len(train):,}")
    logger.info(f"Test events: {len(test):,}")

    # Load RFM data
    rfm_path = project_root / "data/processed/user_segments.parquet"
    if not rfm_path.exists():
        rfm_path = project_root / "data/processed/rfm_segmentation.parquet"

    if rfm_path.exists():
        rfm = pl.read_parquet(rfm_path)
        logger.info(f"RFM data: {len(rfm):,} users")
    else:
        rfm = None
        logger.warning("No RFM data found")

    return train, test, rfm


def analyze_user_distribution(train_events):
    """Analyze cold/warm/hot user distribution."""
    logger.info("\n--- User Distribution Analysis ---")

    groups = split_by_interaction_count(train_events)

    print("\nUser Distribution by Interaction Count:")
    print("-" * 50)
    print(f"Cold users (< 5 interactions):    {len(groups['cold']):>10,}")
    print(f"Warm users (5-20 interactions):   {len(groups['warm']):>10,}")
    print(f"Hot users (> 20 interactions):    {len(groups['hot']):>10,}")
    print("-" * 50)
    print(f"Total users:                      {sum(len(v) for v in groups.values()):>10,}")

    return groups


def train_models(train_events, rfm_data):
    """Train all model variants."""
    models = {}

    # 1. ALS (baseline)
    logger.info("\n[1/5] Training ALS (baseline)...")
    als = ALSRecommender(factors=64, iterations=15)
    als.fit(train_events)
    models["ALS"] = als

    # 2. Hybrid (CF only)
    logger.info("\n[2/5] Training Hybrid (CF only)...")
    hybrid_cf = HybridRecommender(
        factors=64, iterations=15,
        cf_weight=1.0, feature_weight=0.0
    )
    hybrid_cf.fit(train_events, use_item_features=False, use_user_features=False)
    models["Hybrid (CF)"] = hybrid_cf

    # 3. Hybrid (CF + Item features)
    logger.info("\n[3/5] Training Hybrid (CF+Items)...")
    hybrid_items = HybridRecommender(
        factors=64, iterations=15,
        cf_weight=0.7, feature_weight=0.3
    )
    hybrid_items.fit(train_events, use_item_features=True, use_user_features=False)
    models["Hybrid (CF+Items)"] = hybrid_items

    # 4. Hybrid (Full) - with reduced feature_weight for stability
    logger.info("\n[4/5] Training Hybrid (Full)...")
    hybrid_full = HybridRecommender(
        factors=64, iterations=15,
        cf_weight=0.95, feature_weight=0.05  # Reduced feature weight
    )
    hybrid_full.fit(
        train_events,
        rfm_data=rfm_data,
        use_item_features=True,
        use_user_features=True  # Use view-based features for all users
    )
    models["Hybrid (Full)"] = hybrid_full

    # 5. TwoStage
    logger.info("\n[5/5] Training TwoStage...")
    two_stage = TwoStageRecommender(
        factors=64, iterations=15,
        cf_weight=0.7, feature_weight=0.3,
        ranker_iterations=300, n_candidates=100
    )
    two_stage.fit(train_events, rfm_data=rfm_data)
    models["TwoStage"] = two_stage

    return models


def evaluate_all_users(models, test_events, n_users=1000):
    """Evaluate all models on sampled users."""
    logger.info(f"\n--- Evaluating on All Users (n={n_users}) ---")

    evaluator = RecommenderEvaluator(k_values=[10])
    results = evaluator.compare_models(
        models, test_events,
        n_users=n_users,
        show_progress=True
    )

    return results


def evaluate_by_group(models, test_events, train_events, n_users_per_group=500):
    """Evaluate models separately for cold/warm/hot users."""
    logger.info(f"\n--- Evaluating by User Group (n={n_users_per_group} per group) ---")

    results = compare_cold_start_models(
        models, test_events, train_events,
        k_values=[10],
        n_users_per_group=n_users_per_group
    )

    return results


def print_all_users_results(results):
    """Print formatted results for all users."""
    print("\n" + "=" * 80)
    print("ALL USERS COMPARISON")
    print("=" * 80)

    print(f"\n{'Model':<20} {'P@10':>10} {'R@10':>10} {'NDCG@10':>10} {'HitRate@10':>12}")
    print("-" * 65)

    for row in results.sort("model").iter_rows(named=True):
        print(
            f"{row['model']:<20} "
            f"{row['Precision@10']:>10.6f} "
            f"{row['Recall@10']:>10.6f} "
            f"{row['NDCG@10']:>10.6f} "
            f"{row['HitRate@10']:>12.4f}"
        )


def print_group_results(results):
    """Print formatted results by user group."""
    print("\n" + "=" * 80)
    print("BY USER GROUP COMPARISON (Precision@10)")
    print("=" * 80)

    # Pivot results by model and group
    models = results["model"].unique().sort().to_list()
    groups = ["cold", "warm", "hot"]

    print(f"\n{'Model':<20} {'Cold':>12} {'Warm':>12} {'Hot':>12}")
    print("-" * 60)

    for model in models:
        model_data = results.filter(pl.col("model") == model)

        cold_p = model_data.filter(pl.col("group") == "cold")["Precision@10"].to_list()
        warm_p = model_data.filter(pl.col("group") == "warm")["Precision@10"].to_list()
        hot_p = model_data.filter(pl.col("group") == "hot")["Precision@10"].to_list()

        cold_val = cold_p[0] if cold_p else 0.0
        warm_val = warm_p[0] if warm_p else 0.0
        hot_val = hot_p[0] if hot_p else 0.0

        print(f"{model:<20} {cold_val:>12.6f} {warm_val:>12.6f} {hot_val:>12.6f}")


def calculate_improvements(all_results, group_results):
    """Calculate improvements over ALS baseline."""
    print("\n" + "=" * 80)
    print("IMPROVEMENT OVER ALS BASELINE")
    print("=" * 80)

    # Get ALS baseline values
    als_all = all_results.filter(pl.col("model") == "ALS")
    if len(als_all) == 0:
        logger.warning("ALS baseline not found in results")
        return

    als_p10 = als_all["Precision@10"][0]

    # Get ALS group values
    als_group = group_results.filter(pl.col("model") == "ALS")
    als_cold = als_group.filter(pl.col("group") == "cold")["Precision@10"].to_list()
    als_warm = als_group.filter(pl.col("group") == "warm")["Precision@10"].to_list()
    als_hot = als_group.filter(pl.col("group") == "hot")["Precision@10"].to_list()

    als_cold = als_cold[0] if als_cold else 0.0
    als_warm = als_warm[0] if als_warm else 0.0
    als_hot = als_hot[0] if als_hot else 0.0

    print(f"\n{'Model':<20} {'All Users':>12} {'Cold':>12} {'Warm':>12} {'Hot':>12}")
    print("-" * 72)

    models = all_results["model"].unique().sort().to_list()

    improvements = []
    for model in models:
        if model == "ALS":
            continue

        # All users improvement
        model_p10 = all_results.filter(pl.col("model") == model)["Precision@10"][0]
        all_impr = (model_p10 - als_p10) / max(als_p10, 1e-10) * 100

        # Group improvements
        model_group = group_results.filter(pl.col("model") == model)

        cold_p = model_group.filter(pl.col("group") == "cold")["Precision@10"].to_list()
        warm_p = model_group.filter(pl.col("group") == "warm")["Precision@10"].to_list()
        hot_p = model_group.filter(pl.col("group") == "hot")["Precision@10"].to_list()

        cold_val = cold_p[0] if cold_p else 0.0
        warm_val = warm_p[0] if warm_p else 0.0
        hot_val = hot_p[0] if hot_p else 0.0

        cold_impr = (cold_val - als_cold) / max(als_cold, 1e-10) * 100
        warm_impr = (warm_val - als_warm) / max(als_warm, 1e-10) * 100
        hot_impr = (hot_val - als_hot) / max(als_hot, 1e-10) * 100

        print(
            f"{model:<20} "
            f"{all_impr:>+11.1f}% "
            f"{cold_impr:>+11.1f}% "
            f"{warm_impr:>+11.1f}% "
            f"{hot_impr:>+11.1f}%"
        )

        improvements.append({
            "model": model,
            "all_users_improvement": all_impr,
            "cold_improvement": cold_impr,
            "warm_improvement": warm_impr,
            "hot_improvement": hot_impr,
        })

    return improvements


def check_success_criteria(improvements):
    """Check if 10% improvement requirement is met."""
    print("\n" + "=" * 80)
    print("SUCCESS CRITERIA CHECK")
    print("=" * 80)

    # Find Hybrid (Full) improvement
    hybrid_full = next((i for i in improvements if i["model"] == "Hybrid (Full)"), None)

    if hybrid_full:
        all_users_met = hybrid_full["all_users_improvement"] >= 10.0
        cold_positive = hybrid_full["cold_improvement"] > 0

        print(f"\nHybrid (Full) All Users Improvement: {hybrid_full['all_users_improvement']:+.1f}%")
        print(f"Requirement (>= 10%): {'PASSED' if all_users_met else 'FAILED'}")

        print(f"\nHybrid (Full) Cold Users Improvement: {hybrid_full['cold_improvement']:+.1f}%")
        print(f"Requirement (> 0%): {'PASSED' if cold_positive else 'FAILED'}")

        if all_users_met and cold_positive:
            print("\n" + "=" * 80)
            print("STAGE 3 VERIFICATION: PASSED")
            print("=" * 80)
            return True
        else:
            print("\n" + "=" * 80)
            print("STAGE 3 VERIFICATION: NEEDS ATTENTION")
            print("=" * 80)
            return False

    return False


def create_visualizations(all_results, group_results, output_dir):
    """Create Plotly visualizations."""
    try:
        import plotly.express as px
        import plotly.graph_objects as go
    except ImportError:
        logger.warning("Plotly not installed, skipping visualizations")
        return

    output_dir = Path(output_dir)

    # 1. All Users Bar Chart
    fig1 = px.bar(
        all_results.to_pandas(),
        x="model",
        y="Precision@10",
        title="All Users: Precision@10 by Model",
        color="model"
    )
    fig1.update_layout(showlegend=False)
    fig1.write_html(output_dir / "stage3_all_users.html")
    logger.info(f"Saved: {output_dir / 'stage3_all_users.html'}")

    # 2. By Group Bar Chart
    fig2 = px.bar(
        group_results.to_pandas(),
        x="group",
        y="Precision@10",
        color="model",
        barmode="group",
        title="Precision@10 by User Group and Model",
        category_orders={"group": ["cold", "warm", "hot"]}
    )
    fig2.write_html(output_dir / "stage3_by_group.html")
    logger.info(f"Saved: {output_dir / 'stage3_by_group.html'}")

    # 3. NDCG comparison
    fig3 = px.bar(
        all_results.to_pandas(),
        x="model",
        y="NDCG@10",
        title="All Users: NDCG@10 by Model",
        color="model"
    )
    fig3.update_layout(showlegend=False)
    fig3.write_html(output_dir / "stage3_ndcg.html")
    logger.info(f"Saved: {output_dir / 'stage3_ndcg.html'}")


def save_results(all_results, group_results, output_dir):
    """Save results to CSV files."""
    output_dir = Path(output_dir)

    # Save all users results
    all_results.write_csv(output_dir / "stage3_all_users.csv")
    logger.info(f"Saved: {output_dir / 'stage3_all_users.csv'}")

    # Save group results
    group_results.write_csv(output_dir / "stage3_by_group.csv")
    logger.info(f"Saved: {output_dir / 'stage3_by_group.csv'}")


def main():
    """Main verification workflow."""
    logger.info("=" * 70)
    logger.info("STAGE 3 (HYBRID CORE) VERIFICATION")
    logger.info("=" * 70)

    # Load data
    train, test, rfm = load_data()

    # Analyze user distribution
    analyze_user_distribution(train)

    # Train all models
    logger.info("\n" + "=" * 70)
    logger.info("TRAINING MODELS")
    logger.info("=" * 70)
    models = train_models(train, rfm)

    # Evaluate on all users
    all_results = evaluate_all_users(models, test, n_users=1000)

    # Evaluate by group
    group_results = evaluate_by_group(models, test, train, n_users_per_group=500)

    # Print results
    print_all_users_results(all_results)
    print_group_results(group_results)

    # Calculate and print improvements
    improvements = calculate_improvements(all_results, group_results)

    # Check success criteria
    if improvements:
        check_success_criteria(improvements)

    # Save results
    output_dir = project_root / "data/processed"
    save_results(all_results, group_results, output_dir)

    # Create visualizations
    create_visualizations(all_results, group_results, output_dir)

    logger.info("\nVerification complete!")

    return all_results, group_results


if __name__ == "__main__":
    main()
