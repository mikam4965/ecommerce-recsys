"""Script to run conversion funnel analysis."""

import sys
from pathlib import Path

import polars as pl
from loguru import logger

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.analysis.funnel import (
    compare_segments,
    run_funnel_analysis,
    visualize_funnel,
    visualize_segment_comparison,
)


def main():
    """Run funnel analysis."""
    logger.info("=" * 70)
    logger.info("Конверсия воронкасын талдау")
    logger.info("=" * 70)

    # Load training data
    train_path = project_root / "data" / "processed" / "train.parquet"
    rfm_path = project_root / "data" / "processed" / "rfm_segmentation.parquet"

    if not train_path.exists():
        logger.error(f"Training data not found: {train_path}")
        logger.info("Run 'python scripts/preprocess.py' first")
        return

    logger.info(f"Loading data from {train_path}")
    events = pl.read_parquet(train_path)
    logger.info(f"Loaded {len(events):,} events")

    # Load RFM data if exists
    rfm_df = None
    if rfm_path.exists():
        rfm_df = pl.read_parquet(rfm_path)
        logger.info(f"Loaded RFM data: {len(rfm_df):,} users")
    else:
        logger.warning("RFM data not found, segment analysis will be skipped")

    # Run analysis
    results = run_funnel_analysis(events, rfm_df)

    # Print overall funnel
    logger.info("\n" + "=" * 70)
    logger.info("ЖАЛПЫ КОНВЕРСИЯ ВОРОНКАСЫ")
    logger.info("=" * 70)

    funnel = results["funnel"]
    rates = results["rates"]

    print("\n{:<15} {:>15} {:>15}".format("Кезең", "Пайдаланушылар", "Қараудан %"))
    print("-" * 47)
    for row in funnel.iter_rows(named=True):
        print("{:<15} {:>15,} {:>14.2f}%".format(
            row["stage"],
            row["users"],
            row["percentage"],
        ))

    print("-" * 47)
    print(f"\nКонверсия көрсеткіштері:")
    print(f"  Қарау -> Себет:      {rates['view_to_cart']:.2f}%")
    print(f"  Себет -> Сатып алу:  {rates['cart_to_purchase']:.2f}%")
    print(f"  Қарау -> Сатып алу:  {rates['view_to_purchase']:.2f}% (толық цикл)")

    # Save funnel visualization
    funnel_viz_path = project_root / "data" / "processed" / "conversion_funnel.html"
    visualize_funnel(funnel, save_path=funnel_viz_path)

    # Print top categories
    logger.info("\n" + "=" * 70)
    logger.info("ҚАРАУ->САТЫП АЛУ КОНВЕРСИЯСЫ БОЙЫНША ТОП-5 САНАТ")
    logger.info("=" * 70)

    category_funnel = results["category_funnel"]

    print("\n{:<12} {:>10} {:>10} {:>10} {:>12} {:>12} {:>12}".format(
        "Санат", "Қарау", "Себет", "Сатып алу", "Қ->С %", "С->СА %", "Қ->СА %"
    ))
    print("-" * 82)

    for row in category_funnel.head(5).iter_rows(named=True):
        print("{:<12} {:>10,} {:>10,} {:>10,} {:>11.2f}% {:>11.2f}% {:>11.2f}%".format(
            str(row["category_id"])[:12],
            row["views"],
            row["carts"],
            row["purchases"],
            row["view_to_cart"],
            row["cart_to_purchase"],
            row["view_to_purchase"],
        ))

    # Print segment comparison
    if results["segment_funnel"] is not None:
        logger.info("\n" + "=" * 70)
        logger.info("RFM СЕГМЕНТІ БОЙЫНША ВОРОНКА")
        logger.info("=" * 70)

        segment_funnel = results["segment_funnel"]

        print("\n{:<20} {:>10} {:>10} {:>10} {:>10} {:>10} {:>10}".format(
            "Сегмент", "Қарау", "Себет", "Сатып алу", "Қ->С %", "С->СА %", "Қ->СА %"
        ))
        print("-" * 84)

        for row in segment_funnel.iter_rows(named=True):
            print("{:<20} {:>10,} {:>10,} {:>10,} {:>9.2f}% {:>9.2f}% {:>9.2f}%".format(
                row["segment"][:20],
                row["views"],
                row["carts"],
                row["purchases"],
                row["view_to_cart"],
                row["cart_to_purchase"],
                row["view_to_purchase"],
            ))

        # Compare Champions vs Lost
        logger.info("\n" + "=" * 70)
        logger.info("САЛЫСТЫРУ: ЧЕМПИОНДАР мен ЖОҒАЛҒАНДАР")
        logger.info("=" * 70)

        segments_to_compare = ["Champions", "Lost"]
        comparison = compare_segments(segment_funnel, segments_to_compare)

        if len(comparison) == 2:
            print("\n{:<15} {:>15} {:>15}".format("Көрсеткіш", "Чемпиондар", "Жоғалғандар"))
            print("-" * 47)

            champ = comparison.filter(pl.col("segment") == "Champions").to_dicts()[0]
            lost = comparison.filter(pl.col("segment") == "Lost").to_dicts()[0]

            print("{:<15} {:>15,} {:>15,}".format("Қарау", champ["views"], lost["views"]))
            print("{:<15} {:>15,} {:>15,}".format("Себет", champ["carts"], lost["carts"]))
            print("{:<15} {:>15,} {:>15,}".format("Сатып алу", champ["purchases"], lost["purchases"]))
            print("{:<15} {:>14.2f}% {:>14.2f}%".format("Қ->С %", champ["view_to_cart"], lost["view_to_cart"]))
            print("{:<15} {:>14.2f}% {:>14.2f}%".format("С->СА %", champ["cart_to_purchase"], lost["cart_to_purchase"]))
            print("{:<15} {:>14.2f}% {:>14.2f}%".format("Қ->СА %", champ["view_to_purchase"], lost["view_to_purchase"]))

            # Save comparison visualization
            comparison_viz_path = project_root / "data" / "processed" / "funnel_champions_vs_lost.html"
            visualize_segment_comparison(segment_funnel, segments_to_compare, save_path=comparison_viz_path)

    logger.info("\n" + "=" * 70)
    logger.info("Воронка талдауы аяқталды!")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
