"""
RiskCharts: Visualizations for component risk distribution, top risky components, and risk gauges.
"""
from typing import Dict, List, Optional
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

DARK_LAYOUT = {
    "paper_bgcolor": "rgba(0,0,0,0)",
    "plot_bgcolor": "rgba(0,0,0,0)",
    "font": {"color": "#c9d1d9", "family": "Inter, -apple-system, sans-serif"},
    "margin": dict(l=20, r=20, t=35, b=20),
}


def create_risk_distribution_chart(high_count: int, medium_count: int, low_count: int) -> go.Figure:
    """Creates a sleek donut chart of the risk category breakdown."""
    labels = ["High Risk", "Medium Risk", "Low Risk"]
    values = [high_count, medium_count, low_count]
    colors = ["#f85149", "#d29922", "#2ea043"]

    fig = go.Figure(
        data=[
            go.Pie(
                labels=labels,
                values=values,
                hole=0.65,
                marker=dict(colors=colors, line=dict(color="#161b22", width=2)),
                textinfo="label+value",
                hoverinfo="label+percent+value",
                textfont=dict(size=12, color="#c9d1d9"),
            )
        ]
    )
    fig.update_layout(
        **DARK_LAYOUT,
        showlegend=False,
        height=220,
        title=dict(text="Risk Category Distribution", font=dict(size=14, color="#8b949e"), x=0.0)
    )
    return fig


def create_top_risky_components_chart(df_predictions: pd.DataFrame, top_n: int = 8) -> go.Figure:
    """Creates a horizontal bar chart displaying top risky components."""
    if df_predictions.empty:
        return go.Figure()

    df_top = df_predictions.sort_values(by="risk_probability", ascending=True).tail(top_n).copy()
    
    # Color map by category
    color_map = {"HIGH": "#f85149", "MEDIUM": "#d29922", "LOW": "#2ea043"}
    bar_colors = [color_map.get(c, "#58a6ff") for c in df_top["risk_category"]]

    # Truncate long paths for clean display
    labels = [p.split("/")[-1] if len(p) > 25 else p for p in df_top["filepath"]]

    fig = go.Figure(
        go.Bar(
            x=df_top["risk_probability"] * 100,
            y=labels,
            orientation="h",
            marker=dict(color=bar_colors, line=dict(color="rgba(255,255,255,0.1)", width=1)),
            customdata=df_top[["filepath", "risk_category", "complexity", "churn", "dependency_count"]],
            hovertemplate=(
                "<b>%{customdata[0]}</b><br>" +
                "Predicted Risk: %{x:.1f}% (%{customdata[1]})<br>" +
                "Complexity: %{customdata[2]}<br>" +
                "Churn: %{customdata[3]}<br>" +
                "Dependencies: %{customdata[4]}<extra></extra>"
            ),
        )
    )
    fig.update_layout(
        **DARK_LAYOUT,
        height=260,
        title=dict(text=f"Top {len(df_top)} High-Risk Components", font=dict(size=14, color="#8b949e"), x=0.0),
        xaxis=dict(
            title="Predicted Risk (%)",
            range=[0, 100],
            gridcolor="#21262d",
            zerolinecolor="#30363d"
        ),
        yaxis=dict(gridcolor="rgba(0,0,0,0)"),
    )
    return fig


def create_risk_gauge_chart(risk_percentage: int, risk_category: str) -> go.Figure:
    """Creates a compact gauge indicator for an individual component's risk score."""
    color_map = {"HIGH": "#f85149", "MEDIUM": "#d29922", "LOW": "#2ea043"}
    bar_color = color_map.get(risk_category, "#58a6ff")

    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=risk_percentage,
            number={"suffix": "%", "font": {"size": 28, "color": "#f0f6fc", "family": "Inter, sans-serif"}},
            gauge={
                "axis": {"range": [0, 100], "tickwidth": 1, "tickcolor": "#30363d"},
                "bar": {"color": bar_color, "thickness": 0.3},
                "bgcolor": "#21262d",
                "borderwidth": 1,
                "bordercolor": "#30363d",
                "steps": [
                    {"range": [0, 33], "color": "rgba(46, 160, 67, 0.15)"},
                    {"range": [33, 66], "color": "rgba(210, 153, 34, 0.15)"},
                    {"range": [66, 100], "color": "rgba(248, 81, 73, 0.15)"},
                ],
            },
        )
    )
    layout_args = dict(DARK_LAYOUT)
    layout_args["margin"] = dict(l=15, r=15, t=20, b=10)
    layout_args["height"] = 140
    fig.update_layout(**layout_args)
    return fig

