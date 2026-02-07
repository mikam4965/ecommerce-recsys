"""Analytics Dashboard - User Behavior Analysis.

This page provides:
- Conversion funnel visualization
- RFM segmentation analysis
- Category heatmap
- Association rules explorer
"""

import sys
from datetime import datetime, timedelta
from pathlib import Path

import plotly.express as px
import plotly.graph_objects as go
import polars as pl
import streamlit as st

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Page configuration
st.set_page_config(
    page_title="Аналитика - Ұсыныс жүйесі",
    page_icon="📊",
    layout="wide",
)

# Constants
DATA_PATH = PROJECT_ROOT / "data" / "processed"


# =============================================================================
# Data Loading Functions (cached)
# =============================================================================


@st.cache_data(ttl=3600)
def load_events_data() -> pl.DataFrame:
    """Load and cache events data."""
    path = DATA_PATH / "train.parquet"
    if not path.exists():
        return pl.DataFrame()

    df = pl.read_parquet(path)

    # Ensure timestamp is datetime
    if df["timestamp"].dtype != pl.Datetime:
        df = df.with_columns(
            pl.col("timestamp").cast(pl.Datetime("ms"))
        )

    return df


@st.cache_data(ttl=3600)
def load_rfm_data() -> pl.DataFrame:
    """Load and cache RFM segmentation data."""
    path = DATA_PATH / "rfm_segmentation.parquet"
    if not path.exists():
        return pl.DataFrame()

    return pl.read_parquet(path)


@st.cache_data(ttl=3600)
def load_association_rules() -> pl.DataFrame:
    """Load and cache association rules."""
    path = DATA_PATH / "association_rules.parquet"
    if not path.exists():
        return pl.DataFrame()

    return pl.read_parquet(path)


@st.cache_data
def get_date_range(df: pl.DataFrame) -> tuple[datetime, datetime]:
    """Get min and max dates from events data."""
    if df.is_empty():
        today = datetime.now()
        return today - timedelta(days=30), today

    min_date = df["timestamp"].min()
    max_date = df["timestamp"].max()

    # Convert to Python datetime if needed
    if hasattr(min_date, "to_pydatetime"):
        min_date = min_date.to_pydatetime()
    if hasattr(max_date, "to_pydatetime"):
        max_date = max_date.to_pydatetime()

    # Handle None values
    if min_date is None:
        min_date = datetime.now() - timedelta(days=30)
    if max_date is None:
        max_date = datetime.now()

    return min_date, max_date


@st.cache_data
def get_categories(df: pl.DataFrame) -> list[str]:
    """Get unique categories from events data."""
    if df.is_empty():
        return []

    categories = df["category_id"].unique().sort().to_list()
    # Filter out None/null values
    return [c for c in categories if c is not None and c != "unknown"][:50]


@st.cache_data
def get_rfm_segments(df: pl.DataFrame) -> list[str]:
    """Get unique RFM segments."""
    if df.is_empty():
        return []

    return df["segment"].unique().sort().to_list()


# =============================================================================
# Analysis Functions
# =============================================================================


def calculate_funnel_data(
    df: pl.DataFrame,
    start_date: datetime,
    end_date: datetime,
    categories: list[str] | None = None,
) -> dict:
    """Calculate conversion funnel metrics."""
    # Filter by date
    filtered = df.filter(
        (pl.col("timestamp") >= start_date) &
        (pl.col("timestamp") <= end_date)
    )

    # Filter by categories if specified
    if categories:
        filtered = filtered.filter(pl.col("category_id").is_in(categories))

    if filtered.is_empty():
        return {
            "views": 0,
            "carts": 0,
            "transactions": 0,
            "view_to_cart": 0,
            "cart_to_purchase": 0,
            "overall_conversion": 0,
        }

    # Count events by type
    views = len(filtered.filter(pl.col("event_type") == "view"))
    carts = len(filtered.filter(pl.col("event_type") == "addtocart"))
    transactions = len(filtered.filter(pl.col("event_type") == "transaction"))

    # Calculate conversion rates
    view_to_cart = (carts / views * 100) if views > 0 else 0
    cart_to_purchase = (transactions / carts * 100) if carts > 0 else 0
    overall_conversion = (transactions / views * 100) if views > 0 else 0

    return {
        "views": views,
        "carts": carts,
        "transactions": transactions,
        "view_to_cart": view_to_cart,
        "cart_to_purchase": cart_to_purchase,
        "overall_conversion": overall_conversion,
    }


def calculate_category_heatmap(df: pl.DataFrame) -> pl.DataFrame:
    """Calculate category interaction matrix."""
    if df.is_empty():
        return pl.DataFrame()

    # Get top 20 categories by interaction count
    top_categories = (
        df.group_by("category_id")
        .agg(pl.len().alias("count"))
        .sort("count", descending=True)
        .head(20)
        ["category_id"]
        .to_list()
    )

    # Filter to top categories
    filtered = df.filter(pl.col("category_id").is_in(top_categories))

    # Create pivot: event_type x category
    pivot = (
        filtered.group_by(["event_type", "category_id"])
        .agg(pl.len().alias("count"))
        .pivot(
            on="category_id",
            index="event_type",
            values="count",
        )
        .fill_null(0)
    )

    return pivot


# =============================================================================
# Visualization Functions
# =============================================================================


def create_funnel_chart(funnel_data: dict) -> go.Figure:
    """Create conversion funnel visualization."""
    stages = ["Қарау", "Себетке қосу", "Сатып алу"]
    values = [
        funnel_data["views"],
        funnel_data["carts"],
        funnel_data["transactions"],
    ]

    fig = go.Figure(go.Funnel(
        y=stages,
        x=values,
        textposition="inside",
        textinfo="value+percent initial",
        marker=dict(
            color=["#1f77b4", "#ff7f0e", "#2ca02c"],
        ),
        connector=dict(line=dict(color="royalblue", dash="dot", width=3)),
    ))

    fig.update_layout(
        title="Конверсия воронкасы",
        height=400,
        margin=dict(l=20, r=20, t=60, b=20),
    )

    return fig


def create_rfm_pie_chart(rfm_df: pl.DataFrame) -> go.Figure:
    """Create RFM segment distribution pie chart."""
    if rfm_df.is_empty():
        return go.Figure()

    segment_counts = (
        rfm_df.group_by("segment")
        .agg(pl.len().alias("count"))
        .sort("count", descending=True)
    )

    fig = px.pie(
        segment_counts.to_pandas(),
        values="count",
        names="segment",
        title="RFM сегменттерінің таралуы",
        color_discrete_sequence=px.colors.qualitative.Set3,
    )

    fig.update_traces(
        textposition="inside",
        textinfo="percent+label",
    )

    fig.update_layout(
        height=400,
        margin=dict(l=20, r=20, t=60, b=20),
    )

    return fig


def create_category_heatmap(
    df: pl.DataFrame,
    categories: list[str] | None = None,
) -> go.Figure:
    """Create category interaction heatmap."""
    if df.is_empty():
        return go.Figure()

    # Filter by categories if specified
    if categories:
        df = df.filter(pl.col("category_id").is_in(categories))

    # Get top 15 categories
    top_cats = (
        df.group_by("category_id")
        .agg(pl.len().alias("total"))
        .sort("total", descending=True)
        .head(15)
        ["category_id"]
        .to_list()
    )

    df = df.filter(pl.col("category_id").is_in(top_cats))

    # Pivot data for heatmap
    pivot = (
        df.group_by(["event_type", "category_id"])
        .agg(pl.len().alias("count"))
    )

    # Convert to wide format for heatmap
    event_types = ["view", "addtocart", "transaction"]
    heatmap_data = []

    for event_type in event_types:
        row = []
        for cat in top_cats:
            count = pivot.filter(
                (pl.col("event_type") == event_type) &
                (pl.col("category_id") == cat)
            )
            val = count["count"].sum() if not count.is_empty() else 0
            row.append(val)
        heatmap_data.append(row)

    fig = go.Figure(data=go.Heatmap(
        z=heatmap_data,
        x=[str(c)[:15] for c in top_cats],  # Truncate long category names
        y=["Қарау", "Себетке қосу", "Сатып алу"],
        colorscale="Blues",
        hoverongaps=False,
        hovertemplate="Санат: %{x}<br>Оқиға: %{y}<br>Саны: %{z}<extra></extra>",
    ))

    fig.update_layout(
        title="Санат әрекеттесу жылу картасы",
        xaxis_title="Санат",
        yaxis_title="Оқиға түрі",
        height=350,
        margin=dict(l=20, r=20, t=60, b=80),
        xaxis=dict(tickangle=45),
    )

    return fig


# =============================================================================
# Main Page
# =============================================================================


def main():
    """Main analytics page."""
    st.title("📊 Аналитика тақтасы")
    st.markdown("Пайдаланушы мінез-құлқын, конверсия воронкаларын және тұтынушы сегменттерін талдаңыз.")

    # Load data
    events_df = load_events_data()
    rfm_df = load_rfm_data()
    rules_df = load_association_rules()

    if events_df.is_empty():
        st.error("Оқиғалар деректері табылмады. Алдымен алдын ала өңдеуді іске қосыңыз.")
        st.code("python scripts/preprocess.py", language="bash")
        return

    # ==========================================================================
    # Sidebar Filters
    # ==========================================================================

    st.sidebar.header("🔧 Сүзгілер")

    # Date range selector
    min_date, max_date = get_date_range(events_df)

    st.sidebar.subheader("Күн аралығы")
    date_range = st.sidebar.date_input(
        "Кезеңді таңдаңыз",
        value=(min_date.date(), max_date.date()),
        min_value=min_date.date(),
        max_value=max_date.date(),
        key="date_range",
    )

    # Handle single date selection
    if isinstance(date_range, tuple) and len(date_range) == 2:
        start_date = datetime.combine(date_range[0], datetime.min.time())
        end_date = datetime.combine(date_range[1], datetime.max.time())
    else:
        start_date = datetime.combine(date_range, datetime.min.time())
        end_date = datetime.combine(date_range, datetime.max.time())

    # Category multi-select
    st.sidebar.subheader("Санаттар")
    all_categories = get_categories(events_df)

    selected_categories = st.sidebar.multiselect(
        "Санаттар бойынша сүзу",
        options=all_categories,
        default=None,
        help="Барлық санаттарды қосу үшін бос қалдырыңыз",
    )

    # RFM segment selector
    st.sidebar.subheader("RFM сегменті")
    all_segments = get_rfm_segments(rfm_df)

    selected_segment = st.sidebar.selectbox(
        "Сегмент бойынша сүзу",
        options=["Барлығы"] + all_segments,
        index=0,
    )

    # Compare with previous period
    st.sidebar.subheader("Салыстыру")
    compare_previous = st.sidebar.checkbox(
        "Алдыңғы кезеңмен салыстыру",
        value=False,
    )

    # ==========================================================================
    # Main Content
    # ==========================================================================

    # Filter RFM data by segment if selected
    filtered_rfm = rfm_df
    if selected_segment != "Барлығы" and not rfm_df.is_empty():
        filtered_rfm = rfm_df.filter(pl.col("segment") == selected_segment)

    # --------------------------------------------------------------------------
    # Block 1: Conversion Funnel
    # --------------------------------------------------------------------------

    st.header("🔄 Конверсия воронкасы")

    funnel_data = calculate_funnel_data(
        events_df,
        start_date,
        end_date,
        selected_categories if selected_categories else None,
    )

    # Previous period data for comparison
    prev_funnel_data = None
    if compare_previous:
        period_days = (end_date - start_date).days
        prev_start = start_date - timedelta(days=period_days)
        prev_end = start_date - timedelta(seconds=1)

        prev_funnel_data = calculate_funnel_data(
            events_df,
            prev_start,
            prev_end,
            selected_categories if selected_categories else None,
        )

    col1, col2 = st.columns([2, 1])

    with col1:
        fig_funnel = create_funnel_chart(funnel_data)
        st.plotly_chart(fig_funnel, use_container_width=True)

    with col2:
        st.subheader("Конверсия көрсеткіштері")

        # Current period metrics
        st.metric(
            "Қарау → Себет",
            f"{funnel_data['view_to_cart']:.2f}%",
            delta=f"{funnel_data['view_to_cart'] - prev_funnel_data['view_to_cart']:.2f}%"
            if prev_funnel_data else None,
        )

        st.metric(
            "Себет → Сатып алу",
            f"{funnel_data['cart_to_purchase']:.2f}%",
            delta=f"{funnel_data['cart_to_purchase'] - prev_funnel_data['cart_to_purchase']:.2f}%"
            if prev_funnel_data else None,
        )

        st.metric(
            "Жалпы конверсия",
            f"{funnel_data['overall_conversion']:.2f}%",
            delta=f"{funnel_data['overall_conversion'] - prev_funnel_data['overall_conversion']:.2f}%"
            if prev_funnel_data else None,
        )

        with st.expander("📋 Толық сандар"):
            st.write(f"**Қараулар:** {funnel_data['views']:,}")
            st.write(f"**Себетке қосу:** {funnel_data['carts']:,}")
            st.write(f"**Транзакциялар:** {funnel_data['transactions']:,}")

            if prev_funnel_data:
                st.markdown("---")
                st.write("**Алдыңғы кезең:**")
                st.write(f"Қараулар: {prev_funnel_data['views']:,}")
                st.write(f"Себетке қосу: {prev_funnel_data['carts']:,}")
                st.write(f"Транзакциялар: {prev_funnel_data['transactions']:,}")

    # --------------------------------------------------------------------------
    # Block 2: RFM Segmentation
    # --------------------------------------------------------------------------

    st.header("👥 RFM сегментациясы")

    if rfm_df.is_empty():
        st.warning("RFM сегментация деректері табылмады. Алдымен RFM талдауын іске қосыңыз.")
    else:
        col1, col2 = st.columns([1, 1])

        with col1:
            fig_rfm = create_rfm_pie_chart(rfm_df)
            st.plotly_chart(fig_rfm, use_container_width=True)

        with col2:
            st.subheader("Сегмент статистикасы")

            # Calculate segment statistics
            segment_stats = (
                rfm_df.group_by("segment")
                .agg([
                    pl.len().alias("count"),
                    pl.col("recency").mean().alias("avg_recency"),
                    pl.col("frequency").mean().alias("avg_frequency"),
                    pl.col("monetary").mean().alias("avg_monetary"),
                ])
                .sort("count", descending=True)
            )

            # Display as table
            st.dataframe(
                segment_stats.to_pandas().style.format({
                    "count": "{:,}",
                    "avg_recency": "{:.1f}",
                    "avg_frequency": "{:.1f}",
                    "avg_monetary": "{:.2f}",
                }),
                use_container_width=True,
                hide_index=True,
            )

            with st.expander("📊 Сегмент сипаттамалары"):
                st.markdown("""
                | Сегмент | Сипаттама |
                |---------|-----------|
                | Champions | Ең жақсы тұтынушылар: жиі, жақында сатып алады, ең көп жұмсайды |
                | Loyal | Жақсы жиіліктегі тұрақты тұтынушылар |
                | Potential | Өсу әлеуеті бар жақында келген тұтынушылар |
                | At Risk | Бұрын жақсы тұтынушылар, жақында сатып алмаған |
                | Hibernating | Барлық көрсеткіштер бойынша белсенділік төмен |
                """)

    # --------------------------------------------------------------------------
    # Block 3: Category Heatmap
    # --------------------------------------------------------------------------

    st.header("🗺️ Санат жылу картасы")

    fig_heatmap = create_category_heatmap(
        events_df,
        selected_categories if selected_categories else None,
    )
    st.plotly_chart(fig_heatmap, use_container_width=True)

    with st.expander("ℹ️ Бұл жылу картасын қалай оқуға болады"):
        st.markdown("""
        - **Жолдар**: Оқиға түрлері (Қарау, Себетке қосу, Сатып алу)
        - **Бағандар**: Әрекеттесу саны бойынша үздік 15 санат
        - **Түс қанықтығы**: Жоғары мәндер = көп әрекеттесулер
        - **Меңзер**: Әр ұяшық үшін нақты сандарды көру

        Бұл визуализация анықтауға көмектеседі:
        - Жоғары трафикті санаттар (көп қараулар)
        - Жоғары конверсиялы санаттар (қарауларға қатысты жоғары сатып алу)
        - Түсу нүктелері (көп қараулар, бірақ аз сатып алулар)
        """)

    # --------------------------------------------------------------------------
    # Block 4: Association Rules
    # --------------------------------------------------------------------------

    st.header("🔗 Ассоциативтік ережелер")

    if rules_df.is_empty():
        st.warning("Ассоциативтік ережелер табылмады. Алдымен ассоциативтік талдауды іске қосыңыз.")
    else:
        # Sorting options
        col1, col2, col3 = st.columns(3)

        with col1:
            sort_by = st.selectbox(
                "Сұрыптау",
                options=["lift", "confidence", "support"],
                index=0,
            )

        with col2:
            min_lift = st.slider(
                "Мин Lift",
                min_value=1.0,
                max_value=float(rules_df["lift"].max()) if not rules_df.is_empty() else 10.0,
                value=1.0,
                step=0.5,
            )

        with col3:
            min_confidence = st.slider(
                "Мин Confidence",
                min_value=0.0,
                max_value=1.0,
                value=0.0,
                step=0.05,
            )

        # Filter and sort rules
        filtered_rules = rules_df.filter(
            (pl.col("lift") >= min_lift) &
            (pl.col("confidence") >= min_confidence)
        ).sort(sort_by, descending=True)

        # Display top rules
        st.subheader(f"Үздік ережелер ({sort_by} бойынша сұрыпталған)")

        # Format for display
        display_rules = filtered_rules.head(20).to_pandas()

        # Convert frozensets to strings if needed
        if "antecedents" in display_rules.columns:
            display_rules["antecedents"] = display_rules["antecedents"].apply(
                lambda x: ", ".join(map(str, x)) if isinstance(x, (set, frozenset)) else str(x)
            )
        if "consequents" in display_rules.columns:
            display_rules["consequents"] = display_rules["consequents"].apply(
                lambda x: ", ".join(map(str, x)) if isinstance(x, (set, frozenset)) else str(x)
            )

        st.dataframe(
            display_rules.style.format({
                "support": "{:.4f}",
                "confidence": "{:.4f}",
                "lift": "{:.2f}",
            }),
            use_container_width=True,
            hide_index=True,
        )

        with st.expander("📖 Ассоциативтік ережелерді түсіну"):
            st.markdown("""
            **Ассоциативтік ережелер** жиі бірге сатып алынатын тауарларды анықтайды.

            | Метрика | Сипаттама | Түсіндіру |
            |---------|-----------|-----------|
            | **Support** | Барлық транзакцияларда жиынтықтың жиілігі | Жоғары = жиі кездесетін үлгі |
            | **Confidence** | P(салдар \| себеп) | Жоғары = күшті байланыс |
            | **Lift** | Кездейсоқтыққа қатынасы | >1 = оң байланыс |

            **Мысал**: Егер `{нан} → {май}` lift=3.5 болса, нан сатып алушылар
            кездейсоқ тұтынушыларға қарағанда май сатып алу ықтималдығы 3.5 есе жоғары.
            """)

        # Rule statistics
        st.subheader("Ережелер қорытындысы")
        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("Барлық ережелер", f"{len(rules_df):,}")
        with col2:
            st.metric("Орташа Lift", f"{rules_df['lift'].mean():.2f}")
        with col3:
            st.metric("Макс Confidence", f"{rules_df['confidence'].max():.2f}")


if __name__ == "__main__":
    main()
