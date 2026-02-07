"""Generate synthetic e-commerce user behavior data."""

import sys
from pathlib import Path

# Fix Windows console encoding for Kazakh text
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import polars as pl
from loguru import logger

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.data.synthetic_generator import UserBehaviorGenerator


def compare_with_retailrocket(synthetic_stats: dict) -> None:
    """Compare synthetic data distributions with RetailRocket dataset.

    Prints a comparison table showing both distributions side by side.
    """
    # RetailRocket reference statistics
    # Source: actual RetailRocket dataset analysis
    rr_stats = {
        "event_distribution": {
            "view": {"percentage": 94.4},
            "addtocart": {"percentage": 3.4},
            "transaction": {"percentage": 2.2},
        },
        "user_activity": {
            "mean": 3.2,
            "median": 1.0,
        },
        "item_popularity": {
            "gini": 0.85,
        },
        "conversion": {
            "view_to_cart": 8.5,
            "overall": 5.0,
        },
    }

    print("\n" + "=" * 70)
    print("RETAILROCKET ДЕРЕКТЕРІМЕН САЛЫСТЫРУ")
    print("=" * 70)

    # Event distribution comparison
    print(f"\n{'Оқиға түрі':<15} {'RetailRocket':>15} {'Синтетикалық':>15} {'Айырмашылық':>15}")
    print("-" * 62)

    for event_type in ["view", "addtocart", "transaction"]:
        rr_pct = rr_stats["event_distribution"].get(event_type, {}).get("percentage", 0)
        syn_pct = synthetic_stats["event_distribution"].get(event_type, {}).get("percentage", 0)
        diff = syn_pct - rr_pct
        print(f"{event_type:<15} {rr_pct:>14.1f}% {syn_pct:>14.1f}% {diff:>+14.1f}%")

    # User activity comparison
    print(f"\n{'Метрика':<25} {'RetailRocket':>15} {'Синтетикалық':>15}")
    print("-" * 57)

    rr_user = rr_stats["user_activity"]
    syn_user = synthetic_stats["user_activity"]
    print(f"{'Орт. оқиғалар/пайдалан.':<25} {rr_user['mean']:>15.1f} {syn_user['mean']:>15.1f}")
    print(f"{'Медиана оқиғалар':<25} {rr_user['median']:>15.1f} {syn_user['median']:>15.1f}")

    # Item popularity Gini
    rr_gini = rr_stats["item_popularity"]["gini"]
    syn_gini = synthetic_stats["item_popularity"]["gini"]
    print(f"{'Тауар Gini коэффициенті':<25} {rr_gini:>15.4f} {syn_gini:>15.4f}")

    # Conversion
    print(f"\n{'Конверсия':<25} {'RetailRocket':>15} {'Синтетикалық':>15}")
    print("-" * 57)

    rr_conv = rr_stats["conversion"]
    syn_conv = synthetic_stats["conversion"]
    print(f"{'Қарау → Себет':<25} {rr_conv['view_to_cart']:>14.1f}% {syn_conv['view_to_cart']:>14.1f}%")
    print(f"{'Жалпы конверсия':<25} {rr_conv['overall']:>14.1f}% {syn_conv['overall']:>14.1f}%")


def main():
    """Generate synthetic data and compare with RetailRocket."""
    logger.info("=" * 70)
    logger.info("СИНТЕТИКАЛЫҚ ДЕРЕКТЕРДІ ГЕНЕРАЦИЯЛАУ")
    logger.info("=" * 70)

    # Load config
    config_path = project_root / "configs" / "synthetic.yaml"

    if config_path.exists():
        logger.info(f"Конфигурация: {config_path}")
        generator = UserBehaviorGenerator.from_config(config_path)
    else:
        logger.info("Конфигурация табылмады, әдепкі параметрлер қолданылады")
        generator = UserBehaviorGenerator()

    # Generate items
    items = generator.generate_items()

    # Generate events
    n_events = 1_000_000
    logger.info(f"\nМақсатты оқиғалар саны: {n_events:,}")
    events = generator.generate_events(n_events=n_events)

    # Add realistic patterns
    events = generator.add_realistic_patterns(events)

    # Calculate statistics
    stats = generator.get_distribution_stats(events)

    # Print statistics
    print("\n" + "=" * 70)
    print("СИНТЕТИКАЛЫҚ ДЕРЕКТЕР СТАТИСТИКАСЫ")
    print("=" * 70)

    print(f"\nЖалпы оқиғалар: {len(events):,}")
    print(f"Пайдаланушылар: {events['user_id'].n_unique():,}")
    print(f"Тауарлар:       {events['item_id'].n_unique():,}")
    print(f"Сессиялар:      {events['session_id'].n_unique():,}")

    print(f"\n{'Оқиға түрі':<15} {'Саны':>12} {'Проценті':>10}")
    print("-" * 39)
    for event_type, info in stats["event_distribution"].items():
        print(f"{event_type:<15} {info['count']:>12,} {info['percentage']:>9.1f}%")

    print(f"\n{'Пайдаланушы белсенділігі':}")
    print(f"  Орташа:  {stats['user_activity']['mean']:.1f} оқиға/пайдаланушы")
    print(f"  Медиана: {stats['user_activity']['median']:.1f}")
    print(f"  Мин:     {stats['user_activity']['min']}")
    print(f"  Макс:    {stats['user_activity']['max']}")

    print(f"\n{'Тауар танымалдығы':}")
    print(f"  Gini коэффициенті: {stats['item_popularity']['gini']:.4f}")
    print(f"  Орташа:  {stats['item_popularity']['mean']:.1f} оқиға/тауар")
    print(f"  Медиана: {stats['item_popularity']['median']:.1f}")

    print(f"\n{'Конверсия':}")
    print(f"  Қарау → Себет:       {stats['conversion']['view_to_cart']:.1f}%")
    print(f"  Себет → Сатып алу:   {stats['conversion']['cart_to_purchase']:.1f}%")
    print(f"  Жалпы конверсия:     {stats['conversion']['overall']:.1f}%")

    # Compare with RetailRocket
    compare_with_retailrocket(stats)

    # Save data
    output_dir = project_root / "data" / "synthetic"
    generator.save(events, output_dir)

    # Also save a CSV sample for quick inspection
    sample = events.head(1000)
    sample_path = output_dir / "sample_events.csv"
    sample.write_csv(sample_path)
    logger.info(f"Sample (1000 rows) saved to {sample_path}")

    logger.info("\n" + "=" * 70)
    logger.info("Синтетикалық деректер генерациясы аяқталды!")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
