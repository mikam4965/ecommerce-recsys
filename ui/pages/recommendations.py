"""Recommendations Demo - Interactive Recommendation Explorer.

This page provides:
- User selection and recommendation generation
- User history visualization
- Recommendation cards with scores
- Explainability for recommendations
- Similar items exploration
"""

import sys
from pathlib import Path
from datetime import datetime

import polars as pl
import requests
import streamlit as st

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Page configuration
st.set_page_config(
    page_title="Ұсыныстар - Ұсыныс жүйесі",
    page_icon="🎯",
    layout="wide",
)

# Constants
DATA_PATH = PROJECT_ROOT / "data" / "processed"
DB_PATH = PROJECT_ROOT / "data" / "database.sqlite"
API_BASE_URL = "http://127.0.0.1:8000/api/v1"

# Event type colors
EVENT_COLORS = {
    "view": "#3498db",      # Blue
    "addtocart": "#f39c12",  # Orange
    "transaction": "#27ae60",  # Green
}

EVENT_ICONS = {
    "view": "👁️",
    "addtocart": "🛒",
    "transaction": "💳",
}


# =============================================================================
# Data Loading Functions
# =============================================================================


@st.cache_data(ttl=3600)
def load_events_data() -> pl.DataFrame:
    """Load events data."""
    path = DATA_PATH / "train.parquet"
    if not path.exists():
        return pl.DataFrame()
    return pl.read_parquet(path)


@st.cache_data(ttl=3600)
def load_rfm_data() -> pl.DataFrame:
    """Load RFM segmentation data."""
    path = DATA_PATH / "rfm_segmentation.parquet"
    if not path.exists():
        return pl.DataFrame()
    return pl.read_parquet(path)


@st.cache_data(ttl=3600)
def get_sample_users_by_segment() -> dict[str, list[int]]:
    """Get sample users from each RFM segment."""
    rfm = load_rfm_data()
    if rfm.is_empty():
        return {}

    result = {}
    segments = rfm["segment"].unique().to_list()

    for segment in segments:
        users = (
            rfm.filter(pl.col("segment") == segment)
            .head(5)
            ["user_id"]
            .to_list()
        )
        result[segment] = users

    return result


@st.cache_data(ttl=3600)
def get_items_info() -> dict[int, dict]:
    """Load item information from database."""
    import sqlite3

    if not DB_PATH.exists():
        return {}

    conn = sqlite3.connect(str(DB_PATH))
    cursor = conn.cursor()

    cursor.execute("SELECT item_id, category_id, popularity_score FROM items")
    items = {}
    for row in cursor.fetchall():
        items[row[0]] = {
            "item_id": row[0],
            "category_id": row[1] or "unknown",
            "popularity_score": row[2] or 0.0,
        }

    conn.close()
    return items


@st.cache_data(ttl=60)
def get_user_history(user_id: int, limit: int = 20) -> pl.DataFrame:
    """Get user's recent event history."""
    events = load_events_data()
    if events.is_empty():
        return pl.DataFrame()

    user_events = (
        events.filter(pl.col("user_id") == user_id)
        .sort("timestamp", descending=True)
        .head(limit)
    )

    return user_events


@st.cache_data(ttl=60)
def get_user_segment(user_id: int) -> str | None:
    """Get user's RFM segment."""
    rfm = load_rfm_data()
    if rfm.is_empty():
        return None

    user_data = rfm.filter(pl.col("user_id") == user_id)
    if user_data.is_empty():
        return None

    return user_data["segment"][0]


@st.cache_data(ttl=60)
def get_user_favorite_category(user_id: int) -> str | None:
    """Get user's most frequent category."""
    events = load_events_data()
    if events.is_empty():
        return None

    user_events = events.filter(pl.col("user_id") == user_id)
    if user_events.is_empty():
        return None

    top_category = (
        user_events.filter(pl.col("category_id") != "unknown")
        .group_by("category_id")
        .agg(pl.len().alias("count"))
        .sort("count", descending=True)
        .head(1)
    )

    if top_category.is_empty():
        return None

    return top_category["category_id"][0]


# =============================================================================
# API Functions
# =============================================================================


def check_api_health() -> bool:
    """Check if API is available."""
    try:
        response = requests.get(f"{API_BASE_URL}/health", timeout=5)
        if response.status_code == 200:
            data = response.json()
            return data.get("model_loaded", False)
    except Exception:
        pass
    return False


def get_recommendations(user_id: int, n: int = 10) -> dict:
    """Get recommendations from API."""
    try:
        response = requests.post(
            f"{API_BASE_URL}/recommend",
            json={"user_id": user_id, "n_recommendations": n},
            timeout=30,
        )
        if response.status_code == 200:
            return response.json()
    except Exception as e:
        st.error(f"API қатесі: {e}")
    return {}


def get_similar_items(item_id: int, n: int = 5) -> list[dict]:
    """Get similar items from API."""
    try:
        response = requests.post(
            f"{API_BASE_URL}/similar/{item_id}?n_similar={n}",
            timeout=30,
        )
        if response.status_code == 200:
            data = response.json()
            return data.get("similar_items", [])
    except Exception:
        pass
    return []


# =============================================================================
# Explainability Functions
# =============================================================================


def generate_explanation(
    item_id: int,
    user_id: int,
    user_history: pl.DataFrame,
    favorite_category: str | None,
    items_info: dict,
    is_fallback: bool,
) -> str:
    """Generate explanation for why an item was recommended."""
    item_info = items_info.get(item_id, {})
    item_category = item_info.get("category_id", "unknown")
    popularity = item_info.get("popularity_score", 0)

    # Check if item is from user's favorite category
    if favorite_category and item_category == favorite_category:
        return f"Сіздің таңдаулы санатыңызда танымал ({item_category[:20]}...)"

    # Check if user has interacted with this category before
    if not user_history.is_empty():
        user_categories = user_history["category_id"].unique().to_list()
        if item_category in user_categories:
            return f"Сіздің {item_category[:20]}... санатына қызығушылығыңызға негізделген"

    # Check if it's a highly popular item
    if popularity > 0.8:
        return "Трендте: Барлық пайдаланушылар арасында танымал"
    elif popularity > 0.5:
        return "Жиі сатып алынатын тауар"

    # Check if fallback
    if is_fallback:
        return "Жаңа пайдаланушыларға ұсынылатын танымал тауар"

    # Default collaborative filtering explanation
    return "Ұқсас талғамы бар пайдаланушыларға да бұл ұнады"


# =============================================================================
# UI Components
# =============================================================================


def render_event_row(event_type: str, category: str, item_id: int, timestamp) -> str:
    """Render a styled event row."""
    color = EVENT_COLORS.get(event_type, "#95a5a6")
    icon = EVENT_ICONS.get(event_type, "📌")

    return f"""
    <div style="
        display: flex;
        align-items: center;
        padding: 8px;
        margin: 4px 0;
        background-color: {color}22;
        border-left: 4px solid {color};
        border-radius: 4px;
    ">
        <span style="font-size: 1.2em; margin-right: 10px;">{icon}</span>
        <div style="flex-grow: 1;">
            <strong>{event_type.upper()}</strong>
            <span style="color: #666;"> | Тауар: {item_id} | Санат: {category[:20]}...</span>
        </div>
        <span style="color: #888; font-size: 0.9em;">{timestamp}</span>
    </div>
    """


def render_recommendation_card(
    item: dict,
    items_info: dict,
    explanation: str,
    on_click_key: str,
):
    """Render a recommendation card."""
    item_id = item["item_id"]
    score = item["score"]

    info = items_info.get(item_id, {})
    category = info.get("category_id", "unknown")
    popularity = info.get("popularity_score", 0)

    # Card container
    st.markdown(f"""
    <div style="
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 12px;
        padding: 2px;
        margin-bottom: 10px;
    ">
        <div style="
            background: white;
            border-radius: 10px;
            padding: 15px;
        ">
            <h4 style="margin: 0 0 10px 0; color: #333;">Тауар #{item_id}</h4>
            <p style="color: #666; margin: 5px 0; font-size: 0.9em;">
                Санат: {category[:25]}{'...' if len(str(category)) > 25 else ''}
            </p>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Score metric
    st.metric(
        label="Сәйкестік ұпайы",
        value=f"{score:.2f}",
        delta=f"Танымалдылық: {popularity:.0%}" if popularity else None,
    )

    # Explanation
    st.caption(f"💡 {explanation}")

    # Similar items button
    if st.button(f"🔍 Ұқсас тауарлар", key=on_click_key):
        st.session_state[f"show_similar_{item_id}"] = True
        st.session_state["selected_item"] = item_id


# =============================================================================
# Main Page
# =============================================================================


def main():
    """Main recommendations page."""
    st.title("🎯 Ұсыныстар зерттеушісі")
    st.markdown("Жеке ұсыныстар алыңыз және тауарлар неге ұсынылғанын зерттеңіз.")

    # Check API health
    api_available = check_api_health()

    if not api_available:
        st.error("""
        ⚠️ **API қолжетімсіз**

        Алдымен API серверін іске қосыңыз:
        ```bash
        uvicorn src.api.main:app --port 8000
        ```
        """)
        return

    st.success("✅ API қосылған және модель жүктелген")

    # Load data
    sample_users = get_sample_users_by_segment()
    items_info = get_items_info()

    # ==========================================================================
    # Input Panel (Sidebar)
    # ==========================================================================

    st.sidebar.header("🔧 Пайдаланушыны таңдау")

    # Input method selection
    input_method = st.sidebar.radio(
        "Енгізу әдісін таңдаңыз",
        options=["Пайдаланушы ID енгізу", "Мысалдардан таңдау"],
        index=1,
    )

    user_id = None

    if input_method == "Пайдаланушы ID енгізу":
        user_id_input = st.sidebar.text_input(
            "Пайдаланушы ID",
            placeholder="Пайдаланушы ID енгізіңіз (мысалы, 11248)",
            help="Сандық пайдаланушы ID енгізіңіз",
        )
        if user_id_input:
            try:
                user_id = int(user_id_input)
            except ValueError:
                st.sidebar.error("Жарамды сандық ID енгізіңіз")

    else:
        # Dropdown with examples from each segment
        if sample_users:
            # Create options with segment labels
            options = ["Пайдаланушыны таңдаңыз..."]
            user_map = {}

            for segment, users in sample_users.items():
                for u in users[:2]:  # 2 users per segment
                    label = f"{segment}: Пайдаланушы {u}"
                    options.append(label)
                    user_map[label] = u

            selected = st.sidebar.selectbox(
                "Мысал пайдаланушылар (RFM сегменті бойынша)",
                options=options,
            )

            if selected != "Пайдаланушыны таңдаңыз...":
                user_id = user_map[selected]
        else:
            st.sidebar.warning("Мысал пайдаланушылар жоқ. ID қолмен енгізіңіз.")

    # Recommendations count slider
    n_recommendations = st.sidebar.slider(
        "Ұсыныстар саны",
        min_value=5,
        max_value=20,
        value=10,
        step=1,
    )

    # Get recommendations button
    get_recs_button = st.sidebar.button(
        "🎯 Ұсыныстар алу",
        type="primary",
        use_container_width=True,
        disabled=user_id is None,
    )

    # ==========================================================================
    # Main Content
    # ==========================================================================

    if user_id is None:
        st.info("👈 Бастау үшін бүйір тақтадан пайдаланушыны таңдаңыз")
        return

    # Show user info
    st.header(f"👤 Пайдаланушы {user_id}")

    # User segment and stats
    segment = get_user_segment(user_id)
    favorite_category = get_user_favorite_category(user_id)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Пайдаланушы ID", user_id)
    with col2:
        st.metric("Сегмент", segment or "Белгісіз")
    with col3:
        st.metric("Таңдаулы санат", str(favorite_category)[:15] + "..." if favorite_category else "Жоқ")

    st.divider()

    # --------------------------------------------------------------------------
    # Block 1: User History
    # --------------------------------------------------------------------------

    st.subheader("📜 Пайдаланушы тарихы (Соңғы 20 оқиға)")

    user_history = get_user_history(user_id, limit=20)

    if user_history.is_empty():
        st.warning("Бұл пайдаланушы үшін тарих табылмады")
    else:
        # Style the dataframe
        history_display = user_history.select([
            "timestamp",
            "event_type",
            "category_id",
            "item_id",
        ]).to_pandas()

        # Format timestamp
        history_display["timestamp"] = history_display["timestamp"].apply(
            lambda x: x.strftime("%Y-%m-%d %H:%M") if hasattr(x, "strftime") else str(x)[:16]
        )

        # Add event icons
        history_display["Оқиға"] = history_display["event_type"].apply(
            lambda x: f"{EVENT_ICONS.get(x, '📌')} {x}"
        )

        # Reorder columns
        history_display = history_display[[
            "timestamp", "Оқиға", "category_id", "item_id"
        ]].rename(columns={
            "timestamp": "Уақыт",
            "category_id": "Санат",
            "item_id": "Тауар ID",
        })

        # Color by event type
        def highlight_event(row):
            event_type = row["Оқиға"].split()[-1]  # Extract type from "icon type"
            color = EVENT_COLORS.get(event_type, "#ffffff")
            return [f"background-color: {color}22"] * len(row)

        st.dataframe(
            history_display.style.apply(highlight_event, axis=1),
            use_container_width=True,
            hide_index=True,
            height=300,
        )

        # Quick stats
        with st.expander("📊 Тарих статистикасы"):
            event_counts = user_history.group_by("event_type").agg(pl.len().alias("count"))

            col1, col2, col3 = st.columns(3)
            for event_type, col in zip(["view", "addtocart", "transaction"], [col1, col2, col3]):
                count_df = event_counts.filter(pl.col("event_type") == event_type)
                count = count_df["count"][0] if not count_df.is_empty() else 0
                col.metric(
                    f"{EVENT_ICONS.get(event_type, '')} {event_type.title()}s",
                    count,
                )

    st.divider()

    # --------------------------------------------------------------------------
    # Block 2: Recommendations
    # --------------------------------------------------------------------------

    st.subheader("🎁 Жеке ұсыныстар")

    # Get recommendations when button is clicked or on first load
    if get_recs_button or "recommendations" not in st.session_state:
        with st.spinner("Ұсыныстар жасалуда..."):
            rec_response = get_recommendations(user_id, n_recommendations)

            if rec_response:
                st.session_state["recommendations"] = rec_response
                st.session_state["current_user"] = user_id
            else:
                st.error("Ұсыныстар алу сәтсіз аяқталды")
                return

    # Check if we have recommendations for current user
    if (
        "recommendations" in st.session_state and
        st.session_state.get("current_user") == user_id
    ):
        rec_data = st.session_state["recommendations"]
        recommendations = rec_data.get("recommendations", [])
        is_fallback = rec_data.get("is_fallback", False)
        model_type = rec_data.get("model_type", "unknown")

        # Info banner
        if is_fallback:
            st.warning("📌 Пайдаланушы оқу деректерінде жоқ. Оның орнына танымал тауарлар көрсетілуде.")
        else:
            st.info(f"🤖 Модель: {model_type} | Ұсыныстар: {len(recommendations)}")

        # Display recommendations in cards
        if recommendations:
            # Create columns for cards (3 per row)
            cols_per_row = 3
            rows = (len(recommendations) + cols_per_row - 1) // cols_per_row

            for row_idx in range(rows):
                cols = st.columns(cols_per_row)
                for col_idx in range(cols_per_row):
                    item_idx = row_idx * cols_per_row + col_idx
                    if item_idx < len(recommendations):
                        item = recommendations[item_idx]

                        explanation = generate_explanation(
                            item["item_id"],
                            user_id,
                            user_history,
                            favorite_category,
                            items_info,
                            is_fallback,
                        )

                        with cols[col_idx]:
                            render_recommendation_card(
                                item,
                                items_info,
                                explanation,
                                f"similar_btn_{item_idx}",
                            )
        else:
            st.warning("Ұсыныстар қолжетімсіз")

    st.divider()

    # --------------------------------------------------------------------------
    # Block 3: Explainability Summary
    # --------------------------------------------------------------------------

    st.subheader("🧠 Неліктен бұл ұсыныстар?")

    with st.expander("Сіздің ұсыныстарыңызды түсіну", expanded=False):
        st.markdown("""
        Біздің гибридтік ұсыныс жүйесі тауарларды ұсыну үшін бірнеше сигналдарды біріктіреді:

        | Сигнал | Сипаттама | Салмағы |
        |--------|-----------|--------|
        | **Коллаборативтік сүзгілеу** | Ұқсас сатып алу үлгілері бар пайдаланушылар | Жоғары |
        | **Санат қалауы** | Жиі қарайтын санаттарыңыз | Орташа |
        | **Танымалдылық** | Барлық пайдаланушылар арасында трендтегі тауарлар | Төмен |
        | **Жаңалық** | Соңғы әрекеттер маңыздырақ | Ауыспалы |

        **Түсіндіру түрлері:**
        - 💡 *"Ұқсас талғамы бар пайдаланушыларға..."* - Коллаборативтік сүзгілеуге негізделген
        - 💡 *"Сіздің таңдаулы санатыңызда танымал..."* - Санатқа негізделген сигнал
        - 💡 *"Сіздің қызығушылығыңызға негізделген..."* - Қарау тарихыңыз
        - 💡 *"Трендте..."* - Жоғары танымалдылық ұпайы
        """)

        # User profile summary
        if segment:
            st.markdown(f"""
            **Сіздің профиліңіз:**
            - Сегмент: **{segment}**
            - Таңдаулы санат: **{favorite_category or 'Анықталмаған'}**
            - Тарих оқиғалары: **{len(user_history)}**
            """)

    # --------------------------------------------------------------------------
    # Block 4: Similar Items (Interactive)
    # --------------------------------------------------------------------------

    if "selected_item" in st.session_state:
        selected_item = st.session_state["selected_item"]

        st.subheader(f"🔍 #{selected_item} тауарына ұқсас тауарлар")

        with st.spinner("Ұқсас тауарлар ізделуде..."):
            similar = get_similar_items(selected_item, n=5)

        if similar:
            cols = st.columns(len(similar))
            for i, item in enumerate(similar):
                with cols[i]:
                    item_info = items_info.get(item["item_id"], {})
                    st.markdown(f"""
                    <div style="
                        background: #f8f9fa;
                        border-radius: 8px;
                        padding: 12px;
                        text-align: center;
                    ">
                        <h5>Тауар #{item['item_id']}</h5>
                        <p style="font-size: 0.8em; color: #666;">
                            {item_info.get('category_id', 'unknown')[:15]}...
                        </p>
                        <span style="
                            background: #667eea;
                            color: white;
                            padding: 2px 8px;
                            border-radius: 12px;
                            font-size: 0.8em;
                        ">Ұпай: {item['score']:.2f}</span>
                    </div>
                    """, unsafe_allow_html=True)
        else:
            st.info("Ұқсас тауарлар табылмады (тауар модельде болмауы мүмкін)")

        # Clear selection button
        if st.button("Таңдауды тазалау"):
            del st.session_state["selected_item"]
            st.rerun()


if __name__ == "__main__":
    main()
