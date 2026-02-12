"""Train all models (ContentBased, NCF, GRU4Rec, ALS, Hybrid, ItemKNN) and run A/B tests."""

import gc
import sys
from pathlib import Path

# Fix Windows console encoding
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import time

import polars as pl
from loguru import logger

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.evaluation.ab_testing import ABTestRunner, print_ab_results
from src.evaluation.evaluator import RecommenderEvaluator
from src.models.collaborative import ALSRecommender
from src.models.hybrid import HybridRecommender
from src.models.ncf import NCFRecommender
from src.models.gru4rec import GRU4RecRecommender
from src.models.content_based import ContentBasedRecommender
from src.models.knn import ItemKNNRecommender


def main():
    """Train NCF and compare with existing models via A/B test."""
    logger.info("=" * 70)
    logger.info("NCF TRAINING & A/B TESTING")
    logger.info("=" * 70)

    # Load data
    train = pl.read_parquet(project_root / "data/processed/train.parquet")
    test = pl.read_parquet(project_root / "data/processed/test.parquet")
    logger.info(f"Train: {len(train):,} events, Test: {len(test):,} events")

    # Filter data to active users/items (reduces memory for all models)
    # This ensures fair comparison: all models trained on same data
    min_user_events = 3
    min_item_events = 5

    user_counts = train.group_by("user_id").len()
    active_users = user_counts.filter(pl.col("len") >= min_user_events)["user_id"]
    item_counts = train.group_by("item_id").len()
    active_items = item_counts.filter(pl.col("len") >= min_item_events)["item_id"]

    train_filtered = train.filter(
        pl.col("user_id").is_in(active_users) & pl.col("item_id").is_in(active_items)
    )
    logger.info(f"Filtered train: {len(train_filtered):,} events "
                f"({train_filtered['user_id'].n_unique():,} users, "
                f"{train_filtered['item_id'].n_unique():,} items)")

    # Free full train data
    del train, user_counts, active_users, item_counts, active_items
    gc.collect()

    # ========================
    # 1. Train NCF model
    # ========================
    ncf_path = project_root / "models" / "ncf_model.pkl"

    if ncf_path.exists():
        logger.info("\n" + "=" * 70)
        logger.info("LOADING: Pre-trained NCF model")
        logger.info("=" * 70)
        ncf = NCFRecommender()
        ncf.load(ncf_path)
        ncf_time = 0.0
        logger.info("NCF model loaded from disk (skipping training)")
    else:
        logger.info("\n" + "=" * 70)
        logger.info("TRAINING: Neural Collaborative Filtering (NCF)")
        logger.info("=" * 70)

        ncf = NCFRecommender(
            embedding_dim=32,
            mlp_layers=[64, 32, 16],
            learning_rate=0.001,
            batch_size=2048,
            epochs=5,
            n_negatives=4,
            dropout=0.2,
        )

        start = time.time()
        ncf.fit(train_filtered)
        ncf_time = time.time() - start
        logger.info(f"NCF training time: {ncf_time:.1f}s")

        ncf.save(ncf_path)

    gc.collect()

    # ========================
    # 2. Train baseline models
    # ========================
    logger.info("\n" + "=" * 70)
    logger.info("TRAINING: ALS Baseline")
    logger.info("=" * 70)

    als = ALSRecommender(factors=64, iterations=15)
    start = time.time()
    als.fit(train_filtered)
    als_time = time.time() - start
    logger.info(f"ALS training time: {als_time:.1f}s")
    gc.collect()

    logger.info("\n" + "=" * 70)
    logger.info("TRAINING: Hybrid Recommender")
    logger.info("=" * 70)

    hybrid = HybridRecommender(
        factors=64,
        iterations=20,
        cf_weight=0.845,
        feature_weight=0.221,
        regularization=0.00693,
    )
    start = time.time()
    hybrid.fit(train_filtered)
    hybrid_time = time.time() - start
    logger.info(f"Hybrid training time: {hybrid_time:.1f}s")

    gc.collect()

    # ========================
    # 3. Train Content-Based model
    # ========================
    logger.info("\n" + "=" * 70)
    logger.info("TRAINING: Content-Based Filtering")
    logger.info("=" * 70)

    content_based = ContentBasedRecommender()
    start = time.time()
    content_based.fit(train_filtered)
    cb_time = time.time() - start
    logger.info(f"Content-Based training time: {cb_time:.1f}s")

    content_based.save(project_root / "models" / "content_based_model.pkl")
    gc.collect()

    # ========================
    # 4. Train Item k-NN model
    # ========================
    logger.info("\n" + "=" * 70)
    logger.info("TRAINING: Item-based k-NN")
    logger.info("=" * 70)

    knn = ItemKNNRecommender(k=50)
    start = time.time()
    knn.fit(train_filtered)
    knn_time = time.time() - start
    logger.info(f"Item k-NN training time: {knn_time:.1f}s")

    knn.save(project_root / "models" / "knn_model.pkl")
    gc.collect()

    # ========================
    # 5. Train GRU4Rec model
    # ========================
    gru_path = project_root / "models" / "gru4rec_model.pkl"

    if gru_path.exists():
        logger.info("\n" + "=" * 70)
        logger.info("LOADING: Pre-trained GRU4Rec model")
        logger.info("=" * 70)
        gru4rec = GRU4RecRecommender()
        gru4rec.load(gru_path)
        gru_time = 0.0
        logger.info("GRU4Rec model loaded from disk (skipping training)")
    else:
        logger.info("\n" + "=" * 70)
        logger.info("TRAINING: GRU4Rec (Session-based RNN)")
        logger.info("=" * 70)

        gru4rec = GRU4RecRecommender(
            embedding_dim=32,
            hidden_dim=64,
            num_layers=1,
            learning_rate=0.001,
            batch_size=256,
            epochs=5,
            dropout=0.2,
            max_seq_len=20,
            n_negatives=50,
            top_k_items=20000,
        )

        start = time.time()
        gru4rec.fit(train_filtered)
        gru_time = time.time() - start
        logger.info(f"GRU4Rec training time: {gru_time:.1f}s")

        gru4rec.save(gru_path)

    # Free filtered train data
    del train_filtered
    gc.collect()

    # ========================
    # 4. Evaluate all models
    # ========================
    logger.info("\n" + "=" * 70)
    logger.info("EVALUATION: All Models")
    logger.info("=" * 70)

    evaluator = RecommenderEvaluator(k_values=[5, 10, 20])

    models = {
        "ALS": als,
        "Hybrid": hybrid,
        "ContentBased": content_based,
        "ItemKNN": knn,
        "NCF": ncf,
        "GRU4Rec": gru4rec,
    }

    for name, model in models.items():
        logger.info(f"\nEvaluating {name}...")
        results = evaluator.evaluate(model, test, n_users=1000)
        print(f"\n{name} Results:")
        print(results)

    # ========================
    # 5. A/B Tests
    # ========================
    logger.info("\n" + "=" * 70)
    logger.info("A/B TESTING")
    logger.info("=" * 70)

    ab_runner = ABTestRunner(k_values=[5, 10, 20])

    # Test 1: ALS (control) vs NCF (treatment)
    logger.info("\n--- Test 1: ALS vs NCF ---")
    result1 = ab_runner.run_test(
        control_model=als,
        treatment_model=ncf,
        test_data=test,
        n_users=1000,
        control_name="ALS (baseline)",
        treatment_name="NCF (deep learning)",
    )
    print_ab_results(result1)

    # Test 2: ALS (control) vs Hybrid (treatment)
    logger.info("\n--- Test 2: ALS vs Hybrid ---")
    result2 = ab_runner.run_test(
        control_model=als,
        treatment_model=hybrid,
        test_data=test,
        n_users=1000,
        control_name="ALS (baseline)",
        treatment_name="Hybrid (CF + features)",
    )
    print_ab_results(result2)

    # Test 3: Hybrid (control) vs NCF (treatment)
    logger.info("\n--- Test 3: Hybrid vs NCF ---")
    result3 = ab_runner.run_test(
        control_model=hybrid,
        treatment_model=ncf,
        test_data=test,
        n_users=1000,
        control_name="Hybrid (CF + features)",
        treatment_name="NCF (deep learning)",
    )
    print_ab_results(result3)

    # Test 4: ALS (control) vs ContentBased (treatment)
    logger.info("\n--- Test 4: ALS vs ContentBased ---")
    result4 = ab_runner.run_test(
        control_model=als,
        treatment_model=content_based,
        test_data=test,
        n_users=1000,
        control_name="ALS (baseline)",
        treatment_name="ContentBased (category filtering)",
    )
    print_ab_results(result4)

    # Test 5: Hybrid (control) vs GRU4Rec (treatment)
    logger.info("\n--- Test 5: Hybrid vs GRU4Rec ---")
    result5 = ab_runner.run_test(
        control_model=hybrid,
        treatment_model=gru4rec,
        test_data=test,
        n_users=1000,
        control_name="Hybrid (CF + features)",
        treatment_name="GRU4Rec (session RNN)",
    )
    print_ab_results(result5)

    # Test 6: ALS (control) vs ItemKNN (treatment)
    logger.info("\n--- Test 6: ALS vs ItemKNN ---")
    result6 = ab_runner.run_test(
        control_model=als,
        treatment_model=knn,
        test_data=test,
        n_users=1000,
        control_name="ALS (matrix factorization)",
        treatment_name="ItemKNN (k-nearest neighbors)",
    )
    print_ab_results(result6)

    # ========================
    # 7. Summary
    # ========================
    print("\n" + "=" * 70)
    print("TRAINING TIME COMPARISON")
    print("=" * 70)
    print(f"  ALS:          {als_time:>8.1f}s")
    print(f"  Hybrid:       {hybrid_time:>8.1f}s")
    print(f"  ContentBased: {cb_time:>8.1f}s")
    print(f"  ItemKNN:      {knn_time:>8.1f}s")
    print(f"  NCF:          {ncf_time:>8.1f}s")
    print(f"  GRU4Rec:      {gru_time:>8.1f}s")

    logger.info("\n" + "=" * 70)
    logger.info("DONE! All models trained, A/B tests complete.")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
