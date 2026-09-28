"""
ComponentViewVisualizer: Generates feature contribution signal plots and component comparison charts.
"""
from typing import Dict, List, Optional
import pandas as pd
import plotly.graph_objects as go

from explainability.explainer import FeatureSignal

DARK_LAYOUT = {
    "paper_bgcolor": "rgba(0,0,0,0)",
    "plot_bgcolor": "rgba(0,0,0,0)",
    "font": {"color": "#c9d1d9", "family": "Inter, -apple-system, sans-serif"},
    "margin": dict(l=20, r=20, t=30, b=20),
}


def create_signal_bar_chart(signals: List[FeatureSignal]) -> go.Figure:
    """Creates a horizontal bar chart displaying top contributing risk signals for a component."""
    signals_sorted = sorted(signals[:6], key=lambda s: s.importance_weight, reverse=False)


    names = [s.display_name for s in signals_sorted]
    weights = [s.importance_weight * 100 for s in signals_sorted]
    impact_levels = [s.impact_level for s in signals_sorted]
    values = [s.feature_value for s in signals_sorted]

    color_map = {"HIGH": "#f85149", "MEDIUM": "#d29922", "LOW": "#58a6ff"}
    colors = [color_map.get(lvl, "#58a6ff") for lvl in impact_levels]

    fig = go.Figure(
        go.Bar(
            x=weights,
            y=names,
            orientation="h",
            marker=dict(color=colors, line=dict(color="rgba(255,255,255,0.1)", width=1)),
            customdata=list(zip(impact_levels, values)),
            hovertemplate=(
                "<b>%{y}</b><br>" +
                "Model Impact: <b>%{customdata[0]}</b><br>" +
                "Actual Feature Value: <b>%{customdata[1]}</b><extra></extra>"
            )
        )
    )

    fig.update_layout(
        **DARK_LAYOUT,
        height=220,
        title=dict(text="Contributing Model Signals", font=dict(size=13, color="#8b949e"), x=0.0),
        xaxis=dict(title="Relative Contribution Weight", gridcolor="#21262d"),
        yaxis=dict(gridcolor="rgba(0,0,0,0)"),
    )
    return fig


def create_feature_comparison_chart(component_features: Dict[str, float], repo_avg_features: Dict[str, float]) -> go.Figure:
    """Creates a radar/polar chart comparing component metrics against repository averages."""
    categories = ["LOC", "Complexity", "Churn", "Commits", "Dependencies"]
    
    # Scale each metric normalized to [0, 100] for visual comparison
    comp_vals = [
        min(100, component_features.get("loc", 0) / 5),
        min(100, component_features.get("complexity", 1) * 4),
        min(100, component_features.get("churn", 0) / 10),
        min(100, component_features.get("commit_count", 0) * 5),
        min(100, component_features.get("dependency_count", 0) * 15),
    ]

    avg_vals = [
        min(100, repo_avg_features.get("loc", 0) / 5),
        min(100, repo_avg_features.get("complexity", 1) * 4),
        min(100, repo_avg_features.get("churn", 0) / 10),
        min(100, repo_avg_features.get("commit_count", 0) * 5),
        min(100, repo_avg_features.get("dependency_count", 0) * 15),
    ]

    fig = go.Figure()

    fig.add_trace(go.Scatterpolar(
        r=comp_vals + [comp_vals[0]],
        theta=categories + [categories[0]],
        fill="toself",
        name="Selected Component",
        line=dict(color="#58a6ff", width=2),
        fillcolor="rgba(88, 166, 255, 0.2)"
    ))

    fig.add_trace(go.Scatterpolar(
        r=avg_vals + [avg_vals[0]],
        theta=categories + [categories[0]],
        fill="toself",
        name="Repository Average",
        line=dict(color="#8b949e", width=1.5, dash="dash"),
        fillcolor="rgba(139, 148, 158, 0.1)"
    ))

    fig.update_layout(
        **DARK_LAYOUT,
        polar=dict(
            radialaxis=dict(visible=False, range=[0, 100]),
            bgcolor="rgba(0,0,0,0)"
        ),
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=-0.2, xanchor="center", x=0.5, font=dict(size=11)),
        height=260
    )
    return fig
