"""
UI Sidebar: Navigation controls, repository management, and active model inspection.
Redesigned with hierarchical developer-tool aesthetics, section grouping, and clean styling.
"""
from enum import Enum
from pathlib import Path
from typing import Dict, Optional, Tuple
import streamlit as st

from config import DEMO_DATA_DIR
from ingestion.repository_metadata import RepositoryMetadata
from ml.predict import PredictionResult
from ml.train import ClassifierAlgorithm, TrainedModelBundle


class NavigationPage(str, Enum):
    OVERVIEW = "Overview"
    RISK_EXPLORER = "Risk Explorer"
    ARCHITECTURE = "Architecture"
    EVOLUTION = "Evolution"
    DEPENDENCIES = "Dependencies"
    EXPLAINABILITY = "Explainability"
    WHAT_IF_LAB = "What-If Lab"
    MODEL_EVALUATION = "Model Evaluation"
    SETTINGS = "Settings"


NAV_SECTIONS = [
    {
        "header": "WORKSPACE",
        "items": [
            {"id": NavigationPage.OVERVIEW, "label": "Overview", "icon": "▣"}
        ]
    },
    {
        "header": "ANALYSIS",
        "items": [
            {"id": NavigationPage.RISK_EXPLORER, "label": "Risk Explorer", "icon": "◉"},
            {"id": NavigationPage.ARCHITECTURE, "label": "Architecture", "icon": "◈"},
            {"id": NavigationPage.EVOLUTION, "label": "Evolution", "icon": "↗"},
            {"id": NavigationPage.DEPENDENCIES, "label": "Dependencies", "icon": "⌘"},
        ]
    },
    {
        "header": "INTELLIGENCE",
        "items": [
            {"id": NavigationPage.EXPLAINABILITY, "label": "Explainability", "icon": "✦"},
            {"id": NavigationPage.WHAT_IF_LAB, "label": "What-If Lab", "icon": "◇"},
        ]
    },
    {
        "header": "RESEARCH",
        "items": [
            {"id": NavigationPage.MODEL_EVALUATION, "label": "Model Evaluation", "icon": "▤"}
        ]
    },
    {
        "header": "SYSTEM",
        "items": [
            {"id": NavigationPage.SETTINGS, "label": "Settings", "icon": "⚙"}
        ]
    }
]


def render_sidebar(
    repo_meta: Optional[RepositoryMetadata] = None,
    pred_res: Optional[PredictionResult] = None,
    model_bundle: Optional[TrainedModelBundle] = None
) -> Tuple[NavigationPage, str, str, ClassifierAlgorithm, bool]:
    """
    Renders the professional application sidebar.
    Returns (selected_page, repo_source_type, repo_input, selected_algorithm, analyze_clicked).
    """
    # Track active navigation page in session_state
    if "active_nav_page" not in st.session_state:
        st.session_state.active_nav_page = NavigationPage.OVERVIEW

    with st.sidebar:
        # 1. Product Brand Header
        st.markdown(
            """
            <div class="sidebar-brand-container">
                <div>
                    <div class="sidebar-brand-title">SOFTWAREPULSE</div>
                    <div class="sidebar-brand-subtitle">Evolution-aware intelligence</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

        # 2. Sectioned Navigation Menu
        for sec in NAV_SECTIONS:
            st.markdown(f'<div class="nav-section-header">{sec["header"]}</div>', unsafe_allow_html=True)
            for item in sec["items"]:
                page_enum = item["id"]
                is_active = (st.session_state.active_nav_page == page_enum)
                btn_label = f"{item['icon']}  {item['label']}"
                
                # Render clean navigation button
                if st.button(
                    btn_label,
                    key=f"nav_btn_{page_enum.value.lower().replace(' ', '_')}",
                    type="secondary" if not is_active else "primary",
                    use_container_width=True
                ):
                    st.session_state.active_nav_page = page_enum
                    st.rerun()

        st.markdown("<hr style='border-top: 1px solid #21262d; margin: 14px 0 10px 0;'>", unsafe_allow_html=True)

        # 3. Workspace / Repository Section
        st.markdown('<div class="nav-section-header">REPOSITORY</div>', unsafe_allow_html=True)

        source_type = st.selectbox(
            "Source Type",
            options=["Demo Microservices Repo", "Local Repository Path", "Remote GitHub URL"],
            index=0,
            label_visibility="collapsed"
        )

        repo_input = ""
        if source_type == "Demo Microservices Repo":
            demo_path = DEMO_DATA_DIR / "softwarepulse_demo_repo"
            repo_input = str(demo_path)
        elif source_type == "Local Repository Path":
            repo_input = st.text_input("Local Path", value=str(Path.cwd()), help="Path to a local Git repository")
        else:
            repo_input = st.text_input("GitHub Clone URL", value="", placeholder="https://github.com/pallets/flask.git")

        # Repository Metadata Card
        if repo_meta:
            high_c = pred_res.high_risk_count if pred_res else 0
            st.markdown(
                f"""
                <div class="sidebar-workspace-card">
                    <div class="workspace-repo-title">
                        <span>◇</span>
                        <span class="mono-font">{repo_meta.name}</span>
                    </div>
                    <div class="workspace-branch-badge">branch: {repo_meta.default_branch}</div>
                    <div class="workspace-stats-row">
                        <span>{repo_meta.file_count} files</span> • 
                        <span>{repo_meta.commit_count} commits</span> • 
                        <span style="color: {'#ff7b72' if high_c > 0 else '#3fb950'}; font-weight: 600;">{high_c} high risk</span>
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )

        # 4. Active Model Section
        st.markdown('<div class="nav-section-header">ACTIVE MODEL</div>', unsafe_allow_html=True)

        selected_algorithm = st.selectbox(
            "Model Algorithm",
            options=[ClassifierAlgorithm.RANDOM_FOREST, ClassifierAlgorithm.LOGISTIC_REGRESSION],
            format_func=lambda x: x.value,
            index=0,
            label_visibility="collapsed"
        )

        st.markdown(
            f"""
            <div class="sidebar-model-card">
                <div class="sidebar-model-label">PIPELINE ARCHITECTURE</div>
                <div class="sidebar-model-value">{selected_algorithm.value}</div>
                <div class="sidebar-model-desc">Full • Static + Evolution + Dependency</div>
            </div>
            """,
            unsafe_allow_html=True
        )

        # 5. Primary Action Button
        analyze_clicked = st.button("⚡ Run Repository Analysis", type="primary", use_container_width=True)

        # 6. Sidebar Footer
        st.markdown(
            """
            <div style="font-size: 11px; color: #484f58; text-align: center; margin-top: 18px; padding-top: 10px; border-top: 1px solid #161b22;">
                SoftwarePulse v0.1.0<br>
                Research Build • Evolution-Aware
            </div>
            """,
            unsafe_allow_html=True
        )

    return st.session_state.active_nav_page, source_type, repo_input, selected_algorithm, analyze_clicked
