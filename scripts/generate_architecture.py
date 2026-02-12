"""Generate system architecture diagram in draw.io (.drawio) and PNG formats.

Usage:
    python scripts/generate_architecture.py

Output:
    docs/architecture.drawio  — editable in draw.io / VS Code
    docs/architecture.png     — ready PNG for thesis
"""

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

PROJECT_ROOT = Path(__file__).parent.parent
DOCS_DIR = PROJECT_ROOT / "docs"
DOCS_DIR.mkdir(exist_ok=True)


# =============================================================================
# 1. Generate draw.io XML
# =============================================================================

def generate_drawio():
    """Generate .drawio XML file with system architecture."""

    # Helper to create mxCell XML
    def cell(id, value, x, y, w, h, style, parent="1", vertex="1", edge="0",
             source=None, target=None):
        attrs = f'id="{id}" value="{value}" style="{style}"'
        if vertex == "1":
            attrs += f' vertex="1" parent="{parent}"'
        if edge == "1":
            attrs += f' edge="1" parent="{parent}"'
            if source:
                attrs += f' source="{source}"'
            if target:
                attrs += f' target="{target}"'
        if vertex == "1":
            return f'        <mxCell {attrs}>\n          <mxGeometry x="{x}" y="{y}" width="{w}" height="{h}" as="geometry"/>\n        </mxCell>'
        else:
            return f'        <mxCell {attrs}>\n          <mxGeometry relative="1" as="geometry"/>\n        </mxCell>'

    def container(id, value, x, y, w, h, fill, stroke):
        style = (f"swimlane;startSize=35;fillColor={fill};strokeColor={stroke};"
                 f"fontStyle=1;fontSize=13;rounded=1;arcSize=8;whiteSpace=wrap;html=1;collapsible=0;")
        return cell(id, value, x, y, w, h, style)

    def box(id, value, x, y, w, h, fill, stroke, parent="1", font_size=11):
        style = (f"rounded=1;whiteSpace=wrap;html=1;fillColor={fill};strokeColor={stroke};"
                 f"fontSize={font_size};fontStyle=0;shadow=1;")
        return cell(id, value, x, y, w, h, style, parent=parent)

    def arrow(id, source, target, label=""):
        style = "edgeStyle=orthogonalEdgeStyle;rounded=1;orthogonalLoop=1;jettySize=auto;html=1;strokeWidth=2;strokeColor=#666666;"
        attrs = f'id="{id}" value="{label}" style="{style}" edge="1" parent="1" source="{source}" target="{target}"'
        return f'        <mxCell {attrs}>\n          <mxGeometry relative="1" as="geometry"/>\n        </mxCell>'

    def dashed_arrow(id, source, target, label=""):
        style = "edgeStyle=orthogonalEdgeStyle;rounded=1;orthogonalLoop=1;jettySize=auto;html=1;strokeWidth=2;strokeColor=#999999;dashed=1;"
        attrs = f'id="{id}" value="{label}" style="{style}" edge="1" parent="1" source="{source}" target="{target}"'
        return f'        <mxCell {attrs}>\n          <mxGeometry relative="1" as="geometry"/>\n        </mxCell>'

    # Colors
    BLUE = ("#dae8fc", "#6c8ebf")      # Data layer
    GREEN = ("#d5e8d4", "#82b366")     # Analysis layer
    ORANGE = ("#ffe6cc", "#d6b656")    # Models layer
    PURPLE = ("#e1d5e7", "#9673a6")    # API layer
    PINK = ("#f8cecc", "#b85450")      # UI layer
    GRAY = ("#f5f5f5", "#666666")      # MLOps
    WHITE = ("#ffffff", "#333333")     # Inner components

    cells = []

    # ─── Title ───
    title_style = "text;html=1;align=center;verticalAlign=middle;resizable=0;points=[];autosize=1;fontSize=18;fontStyle=1;"
    cells.append(cell("2", "Рекомендациялық жүйенің архитектурасы", 250, 0, 500, 30, title_style))

    # ─── Layer 1: Data ───
    cells.append(container("10", "1. Деректер қабаты (Data Layer)", 30, 40, 860, 195, *BLUE))
    cells.append(box("20", "RetailRocket&lt;br&gt;Dataset", 30, 50, 180, 50, "#bbdefb", "#1565c0", parent="10"))
    cells.append(box("21", "Polars&lt;br&gt;Алдын ала өңдеу", 280, 50, 240, 50, "#bbdefb", "#1565c0", parent="10"))
    cells.append(box("22", "train / val / test&lt;br&gt;(.parquet)", 590, 50, 220, 50, "#bbdefb", "#1565c0", parent="10"))
    cells.append(box("23", "events.csv", 20, 130, 140, 35, *WHITE, parent="10", font_size=10))
    cells.append(box("24", "item_properties.csv", 175, 130, 160, 35, *WHITE, parent="10", font_size=10))
    cells.append(box("25", "category_tree.csv", 350, 130, 155, 35, *WHITE, parent="10", font_size=10))

    # ─── Layer 2: Analysis ───
    cells.append(container("11", "2. Талдау модулі (Analysis)", 30, 270, 860, 115, *GREEN))
    cells.append(box("30", "RFM&lt;br&gt;сегментация", 30, 45, 200, 50, "#c8e6c9", "#2e7d32", parent="11"))
    cells.append(box("31", "Ассоциативтік&lt;br&gt;ережелер (Apriori)", 270, 45, 230, 50, "#c8e6c9", "#2e7d32", parent="11"))
    cells.append(box("32", "Конверсия&lt;br&gt;воронкасы", 540, 45, 220, 50, "#c8e6c9", "#2e7d32", parent="11"))

    # ─── Layer 3: Models ───
    cells.append(container("12", "3. Ұсыныс модельдері (Recommendation Models)", 30, 420, 860, 195, *ORANGE))
    cells.append(box("40", "ALS&lt;br&gt;(implicit)", 30, 45, 160, 50, "#fff9c4", "#f9a825", parent="12"))
    cells.append(box("41", "Hybrid&lt;br&gt;(LightFM)", 220, 45, 160, 50, "#fff9c4", "#f9a825", parent="12"))
    cells.append(box("42", "NCF&lt;br&gt;(PyTorch)", 410, 45, 160, 50, "#ffccbc", "#e65100", parent="12"))
    cells.append(box("43", "GRU4Rec&lt;br&gt;(PyTorch RNN)", 600, 45, 180, 50, "#ffccbc", "#e65100", parent="12"))
    cells.append(box("44", "Evaluator&lt;br&gt;Precision, Recall, NDCG, MAP", 100, 125, 310, 45, "#fff9c4", "#f9a825", parent="12", font_size=10))
    cells.append(box("45", "A/B тестілеу&lt;br&gt;(Welch t-test)", 470, 125, 250, 45, "#fff9c4", "#f9a825", parent="12", font_size=10))

    # ─── Layer 4: API ───
    cells.append(container("13", "4. API қабаты (Service Layer)", 30, 650, 860, 105, *PURPLE))
    cells.append(box("50", "FastAPI&lt;br&gt;REST API", 80, 40, 230, 45, "#ce93d8", "#6a1b9a", parent="13"))
    cells.append(box("51", "SQLite&lt;br&gt;Cache", 380, 40, 180, 45, "#ce93d8", "#6a1b9a", parent="13"))
    cells.append(box("52", "Redis&lt;br&gt;Cache", 620, 40, 160, 45, "#ce93d8", "#6a1b9a", parent="13"))

    # ─── Layer 5: UI ───
    cells.append(container("14", "5. Пайдаланушы интерфейсі (Streamlit UI)", 30, 790, 860, 105, *PINK))
    cells.append(box("60", "Басты бет&lt;br&gt;(Dashboard)", 20, 40, 170, 45, "#ef9a9a", "#c62828", parent="14"))
    cells.append(box("61", "Аналитика&lt;br&gt;(Analytics)", 220, 40, 170, 45, "#ef9a9a", "#c62828", parent="14"))
    cells.append(box("62", "Ұсыныстар&lt;br&gt;(Recommendations)", 420, 40, 200, 45, "#ef9a9a", "#c62828", parent="14"))
    cells.append(box("63", "Эксперименттер&lt;br&gt;(Experiments)", 650, 40, 185, 45, "#ef9a9a", "#c62828", parent="14"))

    # ─── MLOps (side panel) ───
    cells.append(container("15", "MLOps", 930, 270, 190, 485, *GRAY))
    cells.append(box("70", "MLflow&lt;br&gt;Tracking", 20, 55, 150, 55, "#e0e0e0", "#424242", parent="15"))
    cells.append(box("71", "Optuna&lt;br&gt;Optimization", 20, 145, 150, 55, "#e0e0e0", "#424242", parent="15"))
    cells.append(box("72", "Random&lt;br&gt;Seed = 42", 20, 235, 150, 55, "#e0e0e0", "#424242", parent="15"))
    cells.append(box("73", "GitHub&lt;br&gt;Version Control", 20, 325, 150, 55, "#e0e0e0", "#424242", parent="15"))

    # ─── Arrows: within Data layer ───
    cells.append(arrow("100", "20", "21"))
    cells.append(arrow("101", "21", "22"))

    # ─── Arrows: between layers ───
    cells.append(arrow("110", "10", "11"))
    cells.append(arrow("111", "11", "12"))
    cells.append(arrow("112", "12", "13"))
    cells.append(arrow("113", "13", "14"))

    # ─── Arrows: MLOps connections ───
    cells.append(dashed_arrow("120", "70", "12", "логирование"))
    cells.append(dashed_arrow("121", "71", "12", "оптимизация"))

    cells_xml = "\n".join(cells)

    xml = f"""<?xml version="1.0" encoding="UTF-8"?>
<mxfile host="app.diagrams.net" type="device">
  <diagram id="arch" name="System Architecture">
    <mxGraphModel dx="1422" dy="920" grid="1" gridSize="10" guides="1" tooltips="1" connect="1" arrows="1" fold="1" page="1" pageScale="1" pageWidth="1200" pageHeight="950" math="0" shadow="0">
      <root>
        <mxCell id="0"/>
        <mxCell id="1" parent="0"/>
{cells_xml}
      </root>
    </mxGraphModel>
  </diagram>
</mxfile>"""

    out_path = DOCS_DIR / "architecture.drawio"
    out_path.write_text(xml, encoding="utf-8")
    print(f"Created: {out_path}")
    return out_path


# =============================================================================
# 2. Generate PNG with matplotlib
# =============================================================================

def generate_png():
    """Generate architecture diagram as PNG using matplotlib."""

    fig, ax = plt.subplots(1, 1, figsize=(16, 12))
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 12)
    ax.set_aspect("equal")
    ax.axis("off")

    # Title
    ax.text(7, 11.6, "Рекомендациялық жүйенің архитектурасы",
            ha="center", va="center", fontsize=20, fontweight="bold",
            fontfamily="sans-serif")
    ax.text(7, 11.25, "Архитектура рекомендательной системы",
            ha="center", va="center", fontsize=12, fontstyle="italic",
            color="#666666", fontfamily="sans-serif")

    def draw_layer(ax, x, y, w, h, label, color, border_color, alpha=0.25):
        """Draw a layer container."""
        rect = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.05",
                               facecolor=color, edgecolor=border_color,
                               linewidth=2, alpha=alpha)
        ax.add_patch(rect)
        ax.text(x + 0.15, y + h - 0.15, label,
                ha="left", va="top", fontsize=11, fontweight="bold",
                color=border_color, fontfamily="sans-serif")

    def draw_box(ax, x, y, w, h, label, color, border_color, fontsize=9):
        """Draw a component box."""
        rect = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.05",
                               facecolor=color, edgecolor=border_color,
                               linewidth=1.5, alpha=0.9)
        ax.add_patch(rect)
        ax.text(x + w / 2, y + h / 2, label,
                ha="center", va="center", fontsize=fontsize,
                fontfamily="sans-serif", linespacing=1.4)

    def draw_arrow(ax, x1, y1, x2, y2, color="#555555", style="-|>", lw=2):
        """Draw an arrow between points."""
        arrow = FancyArrowPatch((x1, y1), (x2, y2),
                                 arrowstyle=style, mutation_scale=15,
                                 linewidth=lw, color=color)
        ax.add_patch(arrow)

    # ─── Layer 1: Data (y=9.1 to 10.9) ───
    draw_layer(ax, 0.3, 9.1, 12.2, 1.8, "1. Деректер қабаты (Data Layer)",
               "#dae8fc", "#1565c0")

    draw_box(ax, 0.6, 9.85, 2.4, 0.8, "RetailRocket\nDataset",
             "#bbdefb", "#1565c0", fontsize=10)
    draw_box(ax, 3.8, 9.85, 3.2, 0.8, "Polars\nАлдын ала өңдеу",
             "#bbdefb", "#1565c0", fontsize=10)
    draw_box(ax, 7.8, 9.85, 3.2, 0.8, "train / val / test\n(.parquet)",
             "#bbdefb", "#1565c0", fontsize=10)

    # Small source file boxes
    draw_box(ax, 0.5, 9.25, 1.8, 0.45, "events.csv", "#ffffff", "#90caf9", fontsize=8)
    draw_box(ax, 2.5, 9.25, 2.3, 0.45, "item_properties.csv", "#ffffff", "#90caf9", fontsize=8)
    draw_box(ax, 5.0, 9.25, 2.1, 0.45, "category_tree.csv", "#ffffff", "#90caf9", fontsize=8)

    # Arrows within data layer
    draw_arrow(ax, 3.0, 10.25, 3.8, 10.25)
    draw_arrow(ax, 7.0, 10.25, 7.8, 10.25)

    # ─── Layer 2: Analysis (y=7.2 to 8.7) ───
    draw_layer(ax, 0.3, 7.2, 12.2, 1.5, "2. Талдау модулі (Analysis)",
               "#d5e8d4", "#2e7d32")

    draw_box(ax, 0.6, 7.4, 3.0, 0.8, "RFM\nсегментация",
             "#c8e6c9", "#2e7d32", fontsize=10)
    draw_box(ax, 4.2, 7.4, 3.6, 0.8, "Ассоциативтік ережелер\n(Apriori)",
             "#c8e6c9", "#2e7d32", fontsize=10)
    draw_box(ax, 8.4, 7.4, 3.6, 0.8, "Конверсия\nворонкасы",
             "#c8e6c9", "#2e7d32", fontsize=10)

    # ─── Layer 3: Models (y=4.6 to 6.8) ───
    draw_layer(ax, 0.3, 4.6, 12.2, 2.2, "3. Ұсыныс модельдері (Recommendation Models)",
               "#ffe6cc", "#e65100")

    # ML models
    draw_box(ax, 0.6, 6.0, 2.4, 0.7, "ALS\n(implicit)", "#fff9c4", "#f9a825", fontsize=10)
    draw_box(ax, 3.3, 6.0, 2.6, 0.7, "Hybrid\n(LightFM)", "#fff9c4", "#f9a825", fontsize=10)
    # DL models (slightly different color)
    draw_box(ax, 6.2, 6.0, 2.4, 0.7, "NCF\n(PyTorch)", "#ffccbc", "#e65100", fontsize=10)
    draw_box(ax, 8.9, 6.0, 3.2, 0.7, "GRU4Rec\n(PyTorch RNN)", "#ffccbc", "#e65100", fontsize=10)

    # Evaluation row
    draw_box(ax, 1.0, 4.85, 4.5, 0.65, "Evaluator\nPrecision, Recall, NDCG, MAP, HitRate",
             "#fff3e0", "#ef6c00", fontsize=9)
    draw_box(ax, 6.2, 4.85, 4.5, 0.65, "A/B тестілеу\n(Welch's t-test, p-value, lift)",
             "#fff3e0", "#ef6c00", fontsize=9)

    # ─── Layer 4: API (y=3.0 to 4.2) ───
    draw_layer(ax, 0.3, 3.0, 12.2, 1.2, "4. API қабаты (Service Layer)",
               "#e1d5e7", "#6a1b9a")

    draw_box(ax, 0.8, 3.15, 3.5, 0.7, "FastAPI\nREST API", "#ce93d8", "#6a1b9a", fontsize=10)
    draw_box(ax, 5.0, 3.15, 2.8, 0.7, "SQLite\nCache", "#ce93d8", "#6a1b9a", fontsize=10)
    draw_box(ax, 8.5, 3.15, 2.8, 0.7, "Redis\nCache", "#ce93d8", "#6a1b9a", fontsize=10)

    # ─── Layer 5: UI (y=1.2 to 2.6) ───
    draw_layer(ax, 0.3, 1.2, 12.2, 1.4, "5. Пайдаланушы интерфейсі (Streamlit UI)",
               "#f8cecc", "#c62828")

    draw_box(ax, 0.6, 1.35, 2.4, 0.75, "Басты бет\n(Dashboard)", "#ef9a9a", "#c62828", fontsize=9)
    draw_box(ax, 3.3, 1.35, 2.6, 0.75, "Аналитика\n(Analytics)", "#ef9a9a", "#c62828", fontsize=9)
    draw_box(ax, 6.2, 1.35, 2.8, 0.75, "Ұсыныстар\n(Recommendations)", "#ef9a9a", "#c62828", fontsize=9)
    draw_box(ax, 9.3, 1.35, 2.8, 0.75, "Эксперименттер\n(Experiments)", "#ef9a9a", "#c62828", fontsize=9)

    # ─── MLOps side panel (x=13 to 15.5) ───
    draw_layer(ax, 13, 3.0, 2.8, 7.9, "MLOps",
               "#f5f5f5", "#424242")

    draw_box(ax, 13.3, 9.15, 2.2, 0.7, "MLflow\nTracking", "#e0e0e0", "#424242", fontsize=9)
    draw_box(ax, 13.3, 8.0, 2.2, 0.7, "Optuna\nOptimization", "#e0e0e0", "#424242", fontsize=9)
    draw_box(ax, 13.3, 6.85, 2.2, 0.7, "Random\nSeed = 42", "#e0e0e0", "#424242", fontsize=9)
    draw_box(ax, 13.3, 5.7, 2.2, 0.7, "GitHub\nVersion Control", "#e0e0e0", "#424242", fontsize=9)

    # ─── Vertical arrows between layers ───
    draw_arrow(ax, 6.4, 9.1, 6.4, 8.7)    # Data → Analysis
    draw_arrow(ax, 6.4, 7.2, 6.4, 6.8)    # Analysis → Models
    draw_arrow(ax, 6.4, 4.6, 6.4, 4.2)    # Models → API
    draw_arrow(ax, 6.4, 3.0, 6.4, 2.6)    # API → UI

    # ─── MLOps dashed arrows ───
    draw_arrow(ax, 13.0, 9.5, 12.5, 9.5, color="#999999", style="-|>", lw=1.5)
    draw_arrow(ax, 13.0, 8.35, 12.5, 6.5, color="#999999", style="-|>", lw=1.5)

    # ─── Legend at bottom ───
    ax.text(0.5, 0.65, "Белгілеулер:", fontsize=10, fontweight="bold",
            fontfamily="sans-serif")

    legend_items = [
        ("#dae8fc", "Деректер қабаты"),
        ("#d5e8d4", "Талдау модулі"),
        ("#ffe6cc", "ML модельдері"),
        ("#ffccbc", "DL модельдері"),
        ("#e1d5e7", "API қабаты"),
        ("#f8cecc", "UI қабаты"),
        ("#e0e0e0", "MLOps"),
    ]

    for i, (color, label) in enumerate(legend_items):
        x_pos = 0.5 + i * 2.1
        rect = FancyBboxPatch((x_pos, 0.15), 0.4, 0.3, boxstyle="round,pad=0.02",
                               facecolor=color, edgecolor="#666666", linewidth=1)
        ax.add_patch(rect)
        ax.text(x_pos + 0.55, 0.3, label, fontsize=8, va="center",
                fontfamily="sans-serif")

    plt.tight_layout(pad=0.5)

    out_path = DOCS_DIR / "architecture.png"
    fig.savefig(out_path, dpi=200, bbox_inches="tight", facecolor="white",
                edgecolor="none")
    plt.close(fig)
    print(f"Created: {out_path}")
    return out_path


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    print("Generating system architecture diagrams...")
    print()

    drawio_path = generate_drawio()
    png_path = generate_png()

    print()
    print("Done!")
    print(f"  draw.io: {drawio_path}")
    print(f"  PNG:     {png_path}")
    print()
    print("To edit: open .drawio in VS Code (Draw.io extension) or app.diagrams.net")
