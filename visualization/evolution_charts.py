"""
EvolutionCharts: Plots historical metric trajectories (complexity, churn, commit activity) across snapshots.
"""
from typing import Dict, List, Optional
import pandas as pd
import plotly.graph_objects as go
from analysis.historical_analyzer import SnapshotRecord

DARK_LAYOUT = {
    "paper_bgcolor": "rgba(0,0,0,0)",
    "plot_bgcolor": "rgba(0,0,0,0)",
    "font": {"color": "#c9d1d9", "family": "Inter, -apple-system, sans-serif"},
    "margin": dict(l=30, r=20, t=35, b=30),
}


def create_component_evolution_chart(
    snapshots: List[SnapshotRecord],
    target_filepath: str,
    metric_key: str = "complexity",
    metric_label: str = "McCabe Cyclomatic Complexity"
) -> go.Figure:
    """Plots the historical progression of a specific metric for a chosen component."""
    if not snapshots:
        return go.Figure()

    labels = []
    values = []
    hover_texts = []

    for s in snapshots:
        c_dict = s.component_metrics.get(target_filepath, {})
        val = c_dict.get(metric_key, 0)
        date_str = s.timestamp.strftime("%Y-%m-%d")
        lbl = f"S{s.snapshot_index} ({s.commit_hash})"
        labels.append(lbl)
        values.append(val)
        hover_texts.append(
            f"<b>Snapshot {s.snapshot_index}</b> ({s.commit_hash})<br>"
            f"Date: {date_str}<br>"
            f"Commit: {s.commit_message[:40]}...<br>"
            f"{metric_label}: {val}"
        )

    fig = go.Figure()

    # Add line and area fill
    fig.add_trace(
        go.Scatter(
            x=labels,
            y=values,
            mode="lines+markers",
            line=dict(color="#58a6ff", width=2.5),
            marker=dict(size=7, color="#58a6ff", line=dict(color="#ffffff", width=1.5)),
            fill="tozeroy",
            fillcolor="rgba(88, 166, 255, 0.08)",
            hoverinfo="text",
            hovertext=hover_texts,
            name=metric_label
        )
    )

    fig.update_layout(
        **DARK_LAYOUT,
        height=280,
        title=dict(text=f"Historical Evolution: {metric_label}", font=dict(size=14, color="#8b949e"), x=0.0),
        xaxis=dict(gridcolor="#21262d", zerolinecolor="#30363d"),
        yaxis=dict(gridcolor="#21262d", zerolinecolor="#30363d", title=metric_label),
        showlegend=False
    )
    return fig


def create_snapshot_metric_timeline(snapshots: List[SnapshotRecord], target_filepath: str) -> go.Figure:
    """Creates a multi-metric dual-axis timeline chart comparing Churn and Complexity."""
    if not snapshots:
        return go.Figure()

    labels = [f"S{s.snapshot_index} ({s.commit_hash})" for s in snapshots]
    churns = [s.component_metrics.get(target_filepath, {}).get("churn", 0) for s in snapshots]
    complexities = [s.component_metrics.get(target_filepath, {}).get("complexity", 1) for s in snapshots]

    fig = go.Figure()

    # Churn bars
    fig.add_trace(
        go.Bar(
            x=labels,
            y=churns,
            name="Cumulative Churn",
            marker=dict(color="rgba(210, 153, 34, 0.6)"),
            yaxis="y1"
        )
    )

    # Complexity line
    fig.add_trace(
        go.Scatter(
            x=labels,
            y=complexities,
            name="Complexity",
            mode="lines+markers",
            line=dict(color="#f85149", width=2.5),
            marker=dict(size=6),
            yaxis="y2"
        )
    )

    fig.update_layout(
        **DARK_LAYOUT,
        height=280,
        title=dict(text="Evolution Dynamics: Churn vs Complexity", font=dict(size=14, color="#8b949e"), x=0.0),
        xaxis=dict(gridcolor="#21262d"),
        yaxis=dict(title="Churn (Lines Added/Deleted)", gridcolor="#21262d"),
        yaxis2=dict(title="Complexity", overlaying="y", side="right", gridcolor="rgba(0,0,0,0)"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, font=dict(size=11))
    )
    return fig
