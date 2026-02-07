"""E-commerce Recommendation System - Streamlit Dashboard.

Main entry point for the Streamlit application.
Run: streamlit run ui/app.py

Pages are automatically loaded from ui/pages/ directory.
"""

import sys
from pathlib import Path

import streamlit as st

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Page configuration
st.set_page_config(
    page_title="Электрондық сауда ұсыныс жүйесі",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        background: linear-gradient(90deg, #1f77b4, #ff7f0e);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 1rem;
    }
    .page-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 12px;
        padding: 20px;
        color: white;
        margin: 10px 0;
    }
    .page-card h3 {
        color: white;
        margin: 0 0 10px 0;
    }
    .page-card p {
        color: rgba(255,255,255,0.9);
        margin: 0;
    }
    .stat-box {
        background: #f8f9fa;
        border-radius: 8px;
        padding: 15px;
        text-align: center;
        border-left: 4px solid #1f77b4;
    }
</style>
""", unsafe_allow_html=True)


def main():
    """Main application entry point (Home page)."""
    # Header
    st.markdown('<p class="main-header">Электрондық сауда ұсыныс жүйесі</p>', unsafe_allow_html=True)

    st.markdown("""
    **Магистрлік диссертация жобасы**: Ұсыныс алгоритмдерін қолдана отырып,
    интернет-дүкен пайдаланушыларының мінез-құлқын талдау жүйесін жобалау.
    """)

    st.divider()

    # Navigation cards
    st.subheader("📚 Қолжетімді беттер")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("""
        <div class="page-card">
            <h3>📊 Аналитика</h3>
            <p>Пайдаланушы мінез-құлқын талдау, конверсия воронкасы, RFM сегментациясы, ассоциативтік ережелер</p>
        </div>
        """, unsafe_allow_html=True)
        st.caption("Навигация: Бүйір тақта → Аналитика")

    with col2:
        st.markdown("""
        <div class="page-card">
            <h3>🎯 Ұсыныстар</h3>
            <p>Интерактивті демо: кез келген пайдаланушы үшін жеке ұсыныстар алу</p>
        </div>
        """, unsafe_allow_html=True)
        st.caption("Навигация: Бүйір тақта → Ұсыныстар")

    with col3:
        st.markdown("""
        <div class="page-card">
            <h3>🔬 Эксперименттер</h3>
            <p>Модельдерді салыстыру, абляция зерттеуі, оптимизация нәтижелері, API өнімділігі</p>
        </div>
        """, unsafe_allow_html=True)
        st.caption("Навигация: Бүйір тақта → Эксперименттер")

    st.divider()

    # Quick stats
    st.subheader("📈 Деректер жиынының шолуы")

    try:
        import polars as pl

        train_path = PROJECT_ROOT / "data" / "processed" / "train.parquet"
        rfm_path = PROJECT_ROOT / "data" / "processed" / "rfm_segmentation.parquet"

        if train_path.exists():
            train = pl.read_parquet(train_path)

            col1, col2, col3, col4 = st.columns(4)

            with col1:
                st.metric("Барлық оқиғалар", f"{len(train):,}")

            with col2:
                st.metric("Бірегей пайдаланушылар", f"{train['user_id'].n_unique():,}")

            with col3:
                st.metric("Бірегей тауарлар", f"{train['item_id'].n_unique():,}")

            with col4:
                transactions = len(train.filter(pl.col("event_type") == "transaction"))
                st.metric("Транзакциялар", f"{transactions:,}")

            # Event distribution
            st.subheader("Оқиғалар бөлінісі")
            event_counts = train.group_by("event_type").agg(pl.len().alias("count")).sort("count", descending=True)

            col1, col2 = st.columns([2, 1])

            with col1:
                import plotly.express as px
                fig = px.pie(
                    event_counts.to_pandas(),
                    values="count",
                    names="event_type",
                    title="Оқиғалар түрі бойынша",
                    color_discrete_sequence=["#3498db", "#f39c12", "#27ae60"],
                )
                st.plotly_chart(fig, use_container_width=True)

            with col2:
                st.dataframe(
                    event_counts.to_pandas(),
                    use_container_width=True,
                    hide_index=True,
                )

        else:
            st.warning("Оқу деректері табылмады. Алдымен алдын ала өңдеуді іске қосыңыз:")
            st.code("python scripts/preprocess.py", language="bash")

    except Exception as e:
        st.error(f"Деректерді жүктеу қатесі: {e}")

    # System status
    st.divider()
    st.subheader("🔧 Жүйе күйі")

    col1, col2, col3 = st.columns(3)

    # Check API
    with col1:
        try:
            import requests
            response = requests.get("http://127.0.0.1:8000/api/v1/health", timeout=2)
            if response.status_code == 200 and response.json().get("model_loaded"):
                st.success("✅ API: Жұмыс істеп тұр")
            else:
                st.warning("⚠️ API: Іске қосылуда...")
        except Exception:
            st.error("❌ API: Жұмыс істемейді")
            st.caption("Іске қосу: `uvicorn src.api.main:app --port 8000`")

    # Check model
    with col2:
        model_path = PROJECT_ROOT / "models" / "hybrid_best.pkl"
        if model_path.exists():
            size_mb = model_path.stat().st_size / (1024 * 1024)
            st.success(f"✅ Модель: Жүктелді ({size_mb:.1f} MB)")
        else:
            st.error("❌ Модель: Табылмады")
            st.caption("Оқыту: `python scripts/train.py`")

    # Check database
    with col3:
        db_path = PROJECT_ROOT / "data" / "database.sqlite"
        if db_path.exists():
            size_mb = db_path.stat().st_size / (1024 * 1024)
            st.success(f"✅ Дерекқор: Дайын ({size_mb:.1f} MB)")
        else:
            st.warning("⚠️ Дерекқор: Инициализацияланбаған")
            st.caption("Іске қосу: `python scripts/init_database.py`")

    # Footer
    st.divider()
    st.markdown("""
    <div style="text-align: center; color: #666;">
        <p>Streamlit, FastAPI және Polars көмегімен жасалды</p>
        <p>Магистрлік диссертация жобасы | 2025</p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
