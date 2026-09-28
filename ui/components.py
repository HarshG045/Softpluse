"""
UI Components: Reusable UI elements for SoftwarePulse developer dashboard.
"""
import json
from typing import Dict, List, Optional
import pandas as pd
import streamlit as st

from ingestion.repository_metadata import RepositoryMetadata


def render_metric_card(label: str, value: str, subtitle: Optional[str] = None):
    """Renders a sleek developer-tool compact metric block."""
    sub_html = f'<div class="metric-card-sub">{subtitle}</div>' if subtitle else ''
    st.markdown(
        f"""
        <div class="metric-card-compact">
            <div class="metric-card-label">{label}</div>
            <div class="metric-card-value">{value}</div>
            {sub_html}
        </div>
        """,
        unsafe_allow_html=True
    )


def render_risk_pill(risk_percentage: int, category: str) -> str:
    """Returns an HTML snippet for a color-coded risk badge."""
    cat_lower = category.lower()
    return f'<span class="risk-badge risk-badge-{cat_lower}">{risk_percentage}% {category}</span>'


def render_scientific_disclaimer(custom_text: Optional[str] = None):
    """Renders a subtle, non-intrusive scientific methodology note."""
    text = custom_text or (
        "This simulation modifies the model's feature representation to estimate how the "
        "prediction changes. It should be interpreted as a what-if analysis, not proof of causal impact."
    )
    st.markdown(
        f"""
        <div class="methodology-note">
            <span style="font-weight: 600; color: #58a6ff;">Methodology Note:</span> {text}
        </div>
        """,
        unsafe_allow_html=True
    )


def render_export_section(df_predictions: pd.DataFrame, repo_meta: RepositoryMetadata):
    """Provides CSV and JSON export options for analysis results."""
    st.markdown("### Export Analysis Report")
    st.caption("Download structured component risk scores, metrics, and repository analysis.")
    col1, col2 = st.columns(2)

    with col1:
        csv_data = df_predictions.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="Download Risk Report (CSV)",
            data=csv_data,
            file_name=f"{repo_meta.name}_risk_report.csv",
            mime="text/csv",
            use_container_width=True
        )

    with col2:
        report_dict = {
            "repository": repo_meta.to_dict(),
            "total_components": len(df_predictions),
            "predictions": df_predictions.to_dict(orient="records")
        }
        json_data = json.dumps(report_dict, indent=2).encode("utf-8")
        st.download_button(
            label="Download Full Analysis (JSON)",
            data=json_data,
            file_name=f"{repo_meta.name}_analysis_report.json",
            mime="application/json",
            use_container_width=True
        )
