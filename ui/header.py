"""
UI Header: Top context bar displaying repository, branch, analysis state, model architecture, and timestamps.
"""
from typing import Optional
import streamlit as st

from ingestion.repository_metadata import RepositoryMetadata
from ml.train import TrainedModelBundle


def render_header(
    repo_meta: Optional[RepositoryMetadata] = None,
    model_bundle: Optional[TrainedModelBundle] = None,
    is_analyzed: bool = False
):
    """Renders the top developer-tool context bar."""
    if is_analyzed and repo_meta:
        repo_display = f"◇ {repo_meta.name} / {repo_meta.default_branch}"
        status_html = '<span class="context-status-pill pill-analyzed">● Analyzed</span>'
        model_display = f"Model: {model_bundle.model_name}" if model_bundle else "Model: Ready"
        analyzed_time = repo_meta.analyzed_at.strftime("%H:%M:%S UTC")
    else:
        repo_display = "◇ No Repository Loaded"
        status_html = '<span class="context-status-pill pill-idle">● Idle</span>'
        model_display = "Model: Uninitialized"
        analyzed_time = "-"

    st.markdown(
        f"""
        <div class="pulse-context-bar">
            <div class="context-repo-info">
                <span class="mono-font">{repo_display}</span>
                {status_html}
            </div>
            <div class="context-meta-info">
                <span class="mono-font">{model_display}</span>
                <span>Last analyzed: {analyzed_time}</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )
