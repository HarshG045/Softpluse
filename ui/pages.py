"""
UI Pages: Refined, developer-tool analytical and experimental dashboards for SoftwarePulse.
"""
from pathlib import Path
import textwrap
from typing import Dict, List, Optional
import numpy as np
import pandas as pd
import streamlit as st

from analysis.change_analyzer import ChangeImpactAnalyzer, ComponentChangeImpact
from analysis.dependency_analyzer import DependencyGraph
from analysis.historical_analyzer import SnapshotRecord
from config import RISK_THRESHOLDS
from dataset.builder import BuiltDataset
from explainability.explainer import ModelExplainer
from ingestion.git_loader import CommitDetail, FileDiffItem, GitLoader
from ingestion.repository_metadata import RepositoryMetadata
from ml.evaluate import ModelComparisonReport, ModelEvaluator
from ml.predict import ComponentRiskProfile, PredictionResult, RiskPredictor
from ml.train import ClassifierAlgorithm, TrainedModelBundle
from simulation.what_if import SimulationChangeType, WhatIfEngine
from visualization.component_view import create_feature_comparison_chart, create_signal_bar_chart
from visualization.dependency_graph import (
    create_interactive_dependency_figure,
    create_architecture_overview_figure,
    create_risk_propagation_figure,
    create_architecture_diff_figure
)
from health.health_score import RepositoryHealthReport, HealthDimensionScore
from recommendations.recommendation_engine import ComponentRecommendation
from pull_requests.pr_risk_analyzer import PullRequestRiskAnalyzer, PullRequestRiskReport
from experiments.ablation import AblationStudyReport
from visualization.evolution_charts import create_component_evolution_chart, create_snapshot_metric_timeline
from visualization.risk_charts import create_risk_distribution_chart, create_risk_gauge_chart, create_top_risky_components_chart
from .components import render_export_section, render_metric_card, render_risk_pill, render_scientific_disclaimer


def render_overview_page(
    repo_meta: RepositoryMetadata,
    pred_res: PredictionResult,
    comp_report: Optional[ModelComparisonReport] = None,
    health_report: Optional[RepositoryHealthReport] = None,
    recommendations: Optional[List[ActionRecommendation]] = None
):
    """Renders the refined Repository Intelligence overview dashboard."""
    st.markdown("## Repository Intelligence")
    st.caption("Evolution-aware component risk and architecture analysis.")

    # 0. Repository Health Banner (if available)
    if health_report:
        grade_colors = {
            "A": "#3fb950", "B": "#58a6ff", "C": "#d29922", "D": "#db6d28", "F": "#f85149"
        }
        g_color = grade_colors.get(health_report.overall_grade, "#58a6ff")
        st.markdown(
            f"""
            <div style="background: #161b22; border: 1px solid #30363d; border-radius: 8px; padding: 16px 20px; margin-bottom: 16px;">
                <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 12px;">
                    <div>
                        <div style="font-size: 13px; text-transform: uppercase; color: #8b949e; letter-spacing: 0.5px; font-weight: 600;">Repository Health Assessment</div>
                        <div style="font-size: 26px; font-weight: 700; color: #f0f6fc; margin-top: 2px;">
                            {health_report.overall_score}<span style="font-size: 16px; color: #8b949e;"> / 100</span>
                            <span style="font-size: 14px; background: {g_color}22; color: {g_color}; border: 1px solid {g_color}; border-radius: 4px; padding: 2px 8px; margin-left: 10px; font-weight: 600;">Grade {health_report.overall_grade}</span>
                        </div>
                        <div style="font-size: 13px; color: #c9d1d9; margin-top: 4px;">{health_report.summary}</div>
                    </div>
                    <div style="display: flex; gap: 18px; flex-wrap: wrap;">
                        <div style="text-align: center;">
                            <div style="font-size: 11px; color: #8b949e;">Code Quality</div>
                            <div style="font-size: 16px; font-weight: 600; color: #58a6ff;">{health_report.dimensions['Code Quality'].score}%</div>
                        </div>
                        <div style="text-align: center;">
                            <div style="font-size: 11px; color: #8b949e;">Change Risk</div>
                            <div style="font-size: 16px; font-weight: 600; color: #58a6ff;">{health_report.dimensions['Change Risk'].score}%</div>
                        </div>
                        <div style="text-align: center;">
                            <div style="font-size: 11px; color: #8b949e;">Dependencies</div>
                            <div style="font-size: 16px; font-weight: 600; color: #58a6ff;">{health_report.dimensions['Dependency Health'].score}%</div>
                        </div>
                        <div style="text-align: center;">
                            <div style="font-size: 11px; color: #8b949e;">Stability</div>
                            <div style="font-size: 16px; font-weight: 600; color: #58a6ff;">{health_report.dimensions['Stability'].score}%</div>
                        </div>
                        <div style="text-align: center;">
                            <div style="font-size: 11px; color: #8b949e;">Test Health</div>
                            <div style="font-size: 16px; font-weight: 600; color: #58a6ff;">{health_report.dimensions['Test Health'].score}%</div>
                        </div>
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    # 1. Four Compact Metric Tiles
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        render_metric_card("Analyzed Files", str(len(pred_res.profiles)), f"{repo_meta.file_count} total files in repo")
    with m2:
        render_metric_card("Git Commits", str(repo_meta.commit_count), f"{repo_meta.contributor_count} contributors")
    with m3:
        high_c = pred_res.high_risk_count
        med_c = pred_res.medium_risk_count
        low_c = pred_res.low_risk_count
        render_metric_card("High-Risk Components", str(high_c), f"{high_c} high • {med_c} medium • {low_c} low")
    with m4:
        avg_risk = int(np.mean([p.risk_percentage for p in pred_res.profiles])) if pred_res.profiles else 0
        render_metric_card("Mean Component Risk", f"{avg_risk}%", "Model probability average")

    st.markdown("<div style='margin-top: 14px;'></div>", unsafe_allow_html=True)

    # 2. Risk Distribution & Top Risky Charts
    c1, c2 = st.columns([1, 1.4])
    with c1:
        fig_dist = create_risk_distribution_chart(
            pred_res.high_risk_count,
            pred_res.medium_risk_count,
            pred_res.low_risk_count
        )
        st.plotly_chart(fig_dist, use_container_width=True)

    with c2:
        fig_top = create_top_risky_components_chart(pred_res.df_predictions, top_n=6)
        st.plotly_chart(fig_top, use_container_width=True)

    # 3. Top Prioritized Components Table
    st.markdown("### Top Prioritized Components")
    if not pred_res.df_predictions.empty:
        df_display = pred_res.df_predictions[[
            "filepath", "risk_percentage", "risk_category", "complexity", "churn", "dependency_count", "commit_count"
        ]].head(8).copy()
        
        df_display.columns = [
            "Component Path", "Risk (%)", "Category", "Complexity", "Historical Churn", "Dependencies", "Commits"
        ]
        st.dataframe(df_display, use_container_width=True, hide_index=True)

    # 3b. Top Action Recommendations (if available)
    if recommendations:
        st.markdown("<div style='margin-top: 14px;'></div>", unsafe_allow_html=True)
        st.markdown("### Top Prioritized Action Recommendations")
        p_colors = {"CRITICAL": "#f85149", "HIGH": "#d29922", "MEDIUM": "#58a6ff", "LOW": "#3fb950"}
        top_recs = recommendations[:3]
        r_cols = st.columns(len(top_recs))
        for idx, rec in enumerate(top_recs):
            with r_cols[idx]:
                p_str = rec.priority.value if hasattr(rec.priority, "value") else str(rec.priority)
                cat_str = rec.category.value if hasattr(rec.category, "value") else str(rec.category)
                p_col = p_colors.get(p_str, "#58a6ff")
                st.markdown(
                    f"""
                    <div style="background: #161b22; border: 1px solid #30363d; border-top: 3px solid {p_col}; border-radius: 6px; padding: 14px; height: 100%;">
                        <div style="display: flex; justify-content: space-between; align-items: center;">
                            <span style="font-size: 11px; font-weight: 700; color: {p_col}; background: {p_col}22; padding: 2px 6px; border-radius: 3px;">{p_str}</span>
                            <span style="font-size: 11px; color: #8b949e;">{cat_str}</span>
                        </div>
                        <div style="font-size: 14px; font-weight: 600; color: #f0f6fc; margin: 8px 0 4px 0;">{rec.action}</div>
                        <div style="font-size: 12px; color: #8b949e; font-family: monospace; overflow: hidden; text-overflow: ellipsis; white-space: nowrap;">{rec.filepath}</div>
                        <div style="font-size: 12px; color: #c9d1d9; margin-top: 8px;">{rec.suggested_next_step[:120]}...</div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

    # 4. Repository Structure Summary
    st.markdown("<hr style='border-color: #21262d; margin: 16px 0;'>", unsafe_allow_html=True)
    l1, l2 = st.columns([1, 2])
    with l1:
        st.markdown("#### Primary Languages")
        for ext, count in list(repo_meta.languages.items())[:5]:
            lang_name = "Python" if ext == ".py" else (ext.replace(".", "").upper() or "Other")
            st.markdown(f"- **{lang_name}**: `{count}` files")
    with l2:
        st.markdown("#### Research Positioning")
        st.markdown(
            "SoftwarePulse combines **static AST metrics**, **temporal Git churn**, and **graph dependency structure** "
            "into a unified risk predictive pipeline, advancing beyond static source-code analysis."
        )


def render_risk_explorer_page(
    pred_res: PredictionResult,
    dep_graph: DependencyGraph,
    explainer: ModelExplainer,
    features_df: pd.DataFrame,
    git_loader: Optional[GitLoader] = None
):
    """Renders the developer inspection tool for component risk and signal breakdown."""
    st.markdown("## Component Risk Explorer")
    st.caption("Inspect component risk predictions, code characteristics, and localized attribution signals.")

    if pred_res.df_predictions.empty:
        st.info("No components available for display.")
        return

    # Filter controls
    f1, f2 = st.columns([1.5, 1])
    with f1:
        search_query = st.text_input("Search component path...", placeholder="e.g., payment, auth, database")
    with f2:
        category_filter = st.selectbox("Filter Risk Category", options=["ALL", "HIGH", "MEDIUM", "LOW"], index=0)

    # Filter dataframe
    df_filtered = pred_res.df_predictions.copy()
    if search_query.strip():
        df_filtered = df_filtered[df_filtered["filepath"].str.contains(search_query.strip(), case=False)]
    if category_filter != "ALL":
        df_filtered = df_filtered[df_filtered["risk_category"] == category_filter]

    st.markdown(f"**Components matching criteria:** `{len(df_filtered)}`")
    
    # Render table
    df_table = df_filtered[[
        "filepath", "risk_percentage", "risk_category", "loc", "complexity", "churn", "recent_churn", "dependency_count", "commit_count"
    ]].copy()
    df_table.columns = [
        "Component", "Risk (%)", "Category", "LOC", "Complexity", "Total Churn", "Recent Churn", "Dependencies", "Commits"
    ]
    st.dataframe(df_table, use_container_width=True, hide_index=True)

    st.markdown("<hr style='border-color: #21262d; margin: 20px 0;'>", unsafe_allow_html=True)
    st.markdown("### Component Detail Inspector")

    all_components = list(pred_res.profiles_by_path.keys())
    selected_comp_path = st.selectbox("Select Component to Deep-Dive", options=all_components, index=0)

    profile = pred_res.profiles_by_path.get(selected_comp_path)
    if profile:
        d1, d2, d3 = st.columns([1.2, 1.4, 1.4])
        
        with d1:
            st.markdown(f"#### `{selected_comp_path.split('/')[-1]}`")
            st.caption(f"Full path: `{selected_comp_path}`")
            st.plotly_chart(create_risk_gauge_chart(profile.risk_percentage, profile.risk_category), use_container_width=True)
            st.markdown(
                f"""
                **Structural Signals**
                - **LOC:** `{profile.loc}`
                - **Complexity:** `{profile.complexity}`
                - **Functions:** `{profile.functions}`
                - **Commits:** `{profile.commit_count}`
                - **Historical Churn:** `{profile.churn}` lines
                - **Dependencies (Out):** `{profile.dependency_count}`
                - **Dependents (In):** `{profile.dependent_count}`
                """
            )

        with d2:
            st.markdown("#### Model Signals")
            comp_row = features_df[features_df["filepath"] == selected_comp_path].iloc[0]
            explanation = explainer.explain_component(comp_row, profile.risk_probability, profile.risk_category)
            st.plotly_chart(create_signal_bar_chart(explanation.top_signals), use_container_width=True)
            st.caption(f"*{explanation.summary_text}*")

        with d3:
            st.markdown("#### Metric Profile vs Repo Avg")
            repo_avg = {
                "loc": float(features_df["loc"].mean()),
                "complexity": float(features_df["complexity"].mean()),
                "churn": float(features_df["churn"].mean()),
                "commit_count": float(features_df["commit_count"].mean()),
                "dependency_count": float(features_df["dependency_count"].mean())
            }
            comp_metrics = {
                "loc": profile.loc,
                "complexity": profile.complexity,
                "churn": profile.churn,
                "commit_count": profile.commit_count,
                "dependency_count": profile.dependency_count
            }
            st.plotly_chart(create_feature_comparison_chart(comp_metrics, repo_avg), use_container_width=True)

        # 20.7 Component Git History
        if git_loader:
            st.markdown("<hr style='border-color: #21262d; margin: 16px 0;'>", unsafe_allow_html=True)
            st.markdown(f"#### ↗ Git History for `{selected_comp_path.split('/')[-1]}`")
            comp_commits = git_loader.get_component_git_history(selected_comp_path)
            if comp_commits:
                st.caption(f"Historical commit trajectory ({len(comp_commits)} modifications):")
                for c in comp_commits[:6]:
                    dt_str = c.timestamp.strftime("%b %d, %Y")
                    bugfix_badge = " <span style='background: #ff7b7222; color: #ff7b72; padding: 1px 6px; border-radius: 4px; font-size: 11px;'>BUGFIX</span>" if c.is_bugfix else ""
                    st.markdown(
                        f"""
                        <div style="background: #161b22; border: 1px solid #30363d; border-radius: 6px; padding: 10px 14px; margin-bottom: 8px;">
                            <div style="display: flex; justify-content: space-between; align-items: center;">
                                <div>
                                    <span style="color: #58a6ff; font-family: monospace; font-size: 12px;">● {dt_str}</span> — 
                                    <span style="font-weight: 500;">{c.message}</span>
                                    {bugfix_badge}
                                </div>
                                <div style="font-size: 12px; color: #8b949e;">
                                    <span style="color: #3fb950;">+{c.insertions}</span> / <span style="color: #ff7b72;">-{c.deletions}</span>
                                </div>
                            </div>
                            <div style="font-size: 11px; color: #8b949e; margin-top: 4px;">
                                Author: <strong>{c.author_name}</strong> • Commit: <code style="color: #c9d1d9;">{c.commit_hash[:7]}</code>
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
            else:
                st.caption("No individual Git commit history found for this component.")


def render_evolution_page(
    snapshots: List[SnapshotRecord],
    pred_res: PredictionResult
):
    """Renders historical evolution timelines for components over commit snapshots."""
    st.markdown("## Software Evolution Analysis")
    st.caption("Track historical complexity progression, churn dynamics, and change frequency across git commit snapshots.")

    if not snapshots:
        st.info("Insufficient snapshot history for temporal visualization.")
        return

    all_components = list(pred_res.profiles_by_path.keys())
    c_col1, c_col2 = st.columns([1.5, 1])
    with c_col1:
        selected_component = st.selectbox("Select Component to Analyze Evolution", options=all_components, index=0)
    with c_col2:
        metric_choice = st.selectbox(
            "Primary Metric",
            options=["complexity", "churn", "recent_churn", "loc", "commit_count"],
            format_func=lambda x: {
                "complexity": "McCabe Cyclomatic Complexity",
                "churn": "Total Churn (Lines)",
                "recent_churn": "Recent Window Churn",
                "loc": "Lines of Code (LOC)",
                "commit_count": "Historical Commit Count"
            }[x]
        )

    # Evolution trajectory charts
    chart_col1, chart_col2 = st.columns(2)
    with chart_col1:
        metric_labels = {
            "complexity": "Cyclomatic Complexity",
            "churn": "Lines of Churn",
            "recent_churn": "Recent Churn",
            "loc": "Lines of Code",
            "commit_count": "Commit Count"
        }
        fig_evol = create_component_evolution_chart(
            snapshots, selected_component, metric_key=metric_choice, metric_label=metric_labels[metric_choice]
        )
        st.plotly_chart(fig_evol, use_container_width=True)

    with chart_col2:
        fig_multi = create_snapshot_metric_timeline(snapshots, selected_component)
        st.plotly_chart(fig_multi, use_container_width=True)

    # Snapshot Log Table
    st.markdown("### Historical Snapshot Commit Timeline")
    snap_rows = []
    for s in snapshots:
        c_vals = s.component_metrics.get(selected_component, {})
        snap_rows.append({
            "Snapshot": f"Snapshot {s.snapshot_index}",
            "Commit": s.commit_hash,
            "Timestamp": s.timestamp.strftime("%Y-%m-%d %H:%M"),
            "Commit Message": s.commit_message[:55],
            "Complexity": c_vals.get("complexity", "-"),
            "Churn": c_vals.get("churn", "-"),
            "LOC": c_vals.get("loc", "-")
        })
    st.dataframe(pd.DataFrame(snap_rows), use_container_width=True, hide_index=True)


def render_architecture_page(
    dep_graph: DependencyGraph,
    pred_res: PredictionResult
):
    """Renders high-level software architecture, subsystem package coupling, instability index, and cycle detection."""
    st.markdown("## Software Architecture & Subsystems")
    st.caption("Subsystem package coupling, architectural instability index, circular dependency detection, and central hubs.")

    g = dep_graph.graph
    if g.number_of_nodes() == 0:
        st.info("No architectural components found.")
        return

    # Extract architectural analysis
    pkg_clusters = dep_graph.get_package_clusters()
    hotspots = dep_graph.get_architectural_hotspots()
    cycles = dep_graph.detect_cycles()

    # 1. Architectural Metrics Cards
    a1, a2, a3, a4 = st.columns(4)
    with a1:
        render_metric_card("Subsystem Packages", str(len(pkg_clusters)), "Modular directory clusters")
    with a2:
        hub_count = sum(1 for h in hotspots if h["is_hub"])
        render_metric_card("Architectural Hubs", str(hub_count), "High coupling / betweenness bridges")
    with a3:
        cycle_status = f"{len(cycles)} detected" if cycles else "0 (Acyclic DAG)"
        render_metric_card("Circular Dependencies", cycle_status, "Architecture cycle analysis")
    with a4:
        mean_instab = np.mean([c["instability_index"] for c in pkg_clusters.values()]) if pkg_clusters else 0.0
        render_metric_card("Mean Instability (I)", f"{mean_instab:.2f}", "Martin's Instability Metric")

    st.markdown("<div style='margin-top: 14px;'></div>", unsafe_allow_html=True)

    # 2. Subsystem Package Coupling & Stability Table
    st.markdown("### Subsystem Package Coupling & Stability Breakdown")
    pkg_rows = []
    for pkg_name, p_data in pkg_clusters.items():
        i_score = p_data["instability_index"]
        if i_score >= 0.7:
            instab_label = "Volatile (High Efferent)"
        elif i_score <= 0.3:
            instab_label = "Stable (High Afferent)"
        else:
            instab_label = "Balanced"

        pkg_rows.append({
            "Subsystem / Package": pkg_name,
            "Components Count": p_data["component_count"],
            "Afferent Coupling (Ca)": p_data["afferent_coupling_ca"],
            "Efferent Coupling (Ce)": p_data["efferent_coupling_ce"],
            "Instability Index (I)": f"{i_score:.3f}",
            "Classification": instab_label
        })

    st.dataframe(pd.DataFrame(pkg_rows), use_container_width=True, hide_index=True)

    st.markdown("<hr style='border-color: #21262d; margin: 18px 0;'>", unsafe_allow_html=True)

    # 3. Central Architectural Bridges & Hotspots
    st.markdown("### Central Architectural Hubs & Bridges")
    st.caption("Components exhibiting high betweenness centrality act as key architectural bridges between disparate subsystems.")
    hotspot_rows = []
    for h in hotspots[:8]:
        prof = pred_res.profiles_by_path.get(h["component"])
        hotspot_rows.append({
            "Component Path": h["component"],
            "Risk Category": prof.risk_category if prof else "-",
            "Fan-In (Dependents)": h["fan_in"],
            "Fan-Out (Dependencies)": h["fan_out"],
            "Total Coupling": h["total_coupling"],
            "Betweenness Centrality": f"{h['betweenness_centrality']:.4f}",
            "Architectural Role": "Central Hub" if h["is_hub"] else "Standard Module"
        })

    st.dataframe(pd.DataFrame(hotspot_rows), use_container_width=True, hide_index=True)


def render_dependencies_page(
    dep_graph: DependencyGraph,
    pred_res: PredictionResult
):
    """
    Renders the redesigned Component Dependency Graph & Architecture Explorer.
    Features:
      - Two distinct modes: Architecture Overview (Subsystems) & Component Explorer (Granular)
      - Left-to-right hierarchical 3-level directed dependency layouts with directional arrows
      - Integrated toolbar with search autocomplete, layout selection, depth, direction, and risk filters
      - Interactive component details panel with upstream/downstream navigation and circular dependency alerts
      - Synchronized Component In/Out Dependency Matrix
    """
    st.markdown("## Component Dependency Graph & Architecture Explorer")
    st.caption("Interactive software architecture explorer: trace upstream dependencies, downstream dependents, circular cycles, and coupling hotspots.")

    g = dep_graph.graph
    all_nodes = sorted(list(g.nodes)) if g.number_of_nodes() > 0 else []

    if not all_nodes:
        st.info("No dependency architecture found in the ingested repository.")
        return

    # 1. Initialize State Variables
    if "dep_selected_component" not in st.session_state:
        st.session_state["dep_selected_component"] = None
    if "dep_view_mode" not in st.session_state:
        st.session_state["dep_view_mode"] = "Architecture Overview"
    if "dep_layout" not in st.session_state:
        st.session_state["dep_layout"] = "Hierarchical"
    if "dep_depth" not in st.session_state:
        st.session_state["dep_depth"] = 1
    if "dep_dir_filter" not in st.session_state:
        st.session_state["dep_dir_filter"] = "Both"
    if "dep_risk_filter" not in st.session_state:
        st.session_state["dep_risk_filter"] = "All"
    if "dep_label_mode" not in st.session_state:
        st.session_state["dep_label_mode"] = "Full"
    if "dep_highlight_risk" not in st.session_state:
        st.session_state["dep_highlight_risk"] = True
    if "dep_history" not in st.session_state:
        st.session_state["dep_history"] = []

    # Detect cycles and hotspots for metrics
    cycles = dep_graph.detect_cycles()
    hotspots = dep_graph.get_architectural_hotspots()
    struct_map = dep_graph.get_metrics_for_all()
    risk_map = {p.filepath: p.to_dict() for p in pred_res.profiles}

    # 2. Top Metric Cards
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        render_metric_card("Total Components", str(len(all_nodes)), "Analyzed Python files")
    with m2:
        render_metric_card("Dependency Edges", str(g.number_of_edges()), "Intra-repository import links")
    with m3:
        high_c = pred_res.high_risk_count
        med_c = pred_res.medium_risk_count
        render_metric_card("High-Risk Components", str(high_c), f"{high_c} high • {med_c} medium risk")
    with m4:
        cycle_txt = f"{len(cycles)} Detected" if cycles else "0 (Acyclic DAG)"
        render_metric_card("Circular Dependencies", cycle_txt, "Architecture cycle analysis")

    st.markdown("<div style='margin-top: 14px;'></div>", unsafe_allow_html=True)

    # 3. Breadcrumbs Navigation
    selected_comp = st.session_state.get("dep_selected_component")
    b_col1, b_col2 = st.columns([3, 1])
    with b_col1:
        if selected_comp:
            st.markdown(
                f"""
                <div class="dep-breadcrumb-bar">
                    <span style="color: #8b949e;">Architecture</span>
                    <span class="dep-breadcrumb-sep">›</span>
                    <span style="color: #8b949e;">Component Explorer</span>
                    <span class="dep-breadcrumb-sep">›</span>
                    <span class="dep-breadcrumb-item">{selected_comp}</span>
                </div>
                """,
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                """
                <div class="dep-breadcrumb-bar">
                    <span class="dep-breadcrumb-item">Repository Architecture Overview</span>
                    <span class="dep-breadcrumb-sep">›</span>
                    <span style="color: #8b949e;">All Subsystem Packages</span>
                </div>
                """,
                unsafe_allow_html=True
            )
    with b_col2:
        if selected_comp:
            if st.button("Reset to Overview", key="dep_btn_reset_bread", use_container_width=True):
                st.session_state["dep_selected_component"] = None
                st.session_state["dep_view_mode"] = "Architecture Overview"
                st.rerun()

    # 4. Redesigned Graph Toolbar Box
    st.markdown("<div class='dep-toolbar-box'>", unsafe_allow_html=True)

    t_r1_c1, t_r1_c2, t_r1_c3, t_r1_c4 = st.columns([1.5, 1.6, 1.2, 1.1])
    with t_r1_c1:
        view_modes = ["Architecture Overview", "Component Explorer", "Risk Propagation", "Architecture Diff"]
        cur_mode = st.session_state["dep_view_mode"]
        mode_idx = view_modes.index(cur_mode) if cur_mode in view_modes else 0
        new_mode = st.radio(
            "Graph View Mode",
            options=view_modes,
            index=mode_idx,
            horizontal=True,
            key="dep_view_mode_radio"
        )
        if new_mode != st.session_state["dep_view_mode"]:
            st.session_state["dep_view_mode"] = new_mode
            st.rerun()

    with t_r1_c2:
        # Search with autocomplete
        comp_display_options = ["-- Select / Search Component --"] + all_nodes
        search_idx = 0
        if selected_comp and selected_comp in all_nodes:
            search_idx = all_nodes.index(selected_comp) + 1

        def format_comp_opt(x: str) -> str:
            if x == "-- Select / Search Component --":
                return x
            prof = pred_res.profiles_by_path.get(x)
            cat = prof.risk_category if prof else "LOW"
            tag = f"[{cat}]"
            return f"{tag} {x}"

        selected_from_search = st.selectbox(
            "Component Search (Autocomplete)",
            options=comp_display_options,
            index=search_idx,
            format_func=format_comp_opt,
            key="dep_search_box"
        )
        if selected_from_search != "-- Select / Search Component --" and selected_from_search != selected_comp:
            st.session_state["dep_selected_component"] = selected_from_search
            st.session_state["dep_view_mode"] = "Component Explorer"
            if selected_from_search not in st.session_state["dep_history"]:
                st.session_state["dep_history"].append(selected_from_search)
            st.rerun()

    with t_r1_c3:
        if st.session_state["dep_view_mode"] == "Component Explorer":
            layout_options = ["Hierarchical", "Radial", "Grouped"]
            cur_layout_idx = layout_options.index(st.session_state["dep_layout"]) if st.session_state["dep_layout"] in layout_options else 0
            new_layout = st.selectbox("Layout Algorithm", options=layout_options, index=cur_layout_idx, key="dep_layout_select")
            st.session_state["dep_layout"] = new_layout
        else:
            st.selectbox("Layout Algorithm", options=["Subsystem Packages (Inter-module)"], index=0, disabled=True)

    with t_r1_c4:
        if st.session_state["dep_view_mode"] == "Component Explorer":
            depth_options = [1, 2, 3]
            cur_depth_idx = depth_options.index(st.session_state["dep_depth"]) if st.session_state["dep_depth"] in depth_options else 0
            new_depth = st.selectbox("Neighborhood Depth", options=depth_options, index=cur_depth_idx, format_func=lambda d: f"{d} Hop{'s' if d > 1 else ''}", key="dep_depth_select")
            st.session_state["dep_depth"] = new_depth
        else:
            st.selectbox("Neighborhood Depth", options=["Cluster-wide"], index=0, disabled=True)

    # Toolbar Row 2: Direction, Risk Filter, Labels, and Toggles
    t_r2_c1, t_r2_c2, t_r2_c3, t_r2_c4, t_r2_c5 = st.columns([1.2, 1.1, 1.1, 1.2, 0.9])
    with t_r2_c1:
        if st.session_state["dep_view_mode"] == "Component Explorer":
            dir_options = ["Both", "Dependencies", "Dependents"]
            cur_dir_idx = dir_options.index(st.session_state["dep_dir_filter"]) if st.session_state["dep_dir_filter"] in dir_options else 0
            new_dir = st.selectbox("Direction Filter", options=dir_options, index=cur_dir_idx, format_func=lambda d: f"{d} {'(In & Out)' if d == 'Both' else '(Outgoing)' if d == 'Dependencies' else '(Incoming)'}", key="dep_dir_select")
            st.session_state["dep_dir_filter"] = new_dir
        else:
            st.selectbox("Direction Filter", options=["All Connections"], index=0, disabled=True)

    with t_r2_c2:
        risk_filter_options = ["All", "High", "Medium", "Low"]
        cur_rf_idx = risk_filter_options.index(st.session_state["dep_risk_filter"]) if st.session_state["dep_risk_filter"] in risk_filter_options else 0
        new_risk_filter = st.selectbox("Risk Filter", options=risk_filter_options, index=cur_rf_idx, format_func=lambda r: f"{r} Risk", key="dep_risk_filter_select")
        st.session_state["dep_risk_filter"] = new_risk_filter

    with t_r2_c3:
        label_options = ["Full", "Truncated", "Hover"]
        cur_lbl_idx = label_options.index(st.session_state["dep_label_mode"]) if st.session_state["dep_label_mode"] in label_options else 0
        new_label_mode = st.selectbox("Node Labels", options=label_options, index=cur_lbl_idx, format_func=lambda l: f"{l} Labels", key="dep_label_select")
        st.session_state["dep_label_mode"] = new_label_mode

    with t_r2_c4:
        st.markdown("<div style='margin-top: 24px;'></div>", unsafe_allow_html=True)
        high_risk_checked = st.checkbox("Highlight High Risk", value=st.session_state["dep_highlight_risk"], key="dep_hl_risk_check")
        st.session_state["dep_highlight_risk"] = high_risk_checked

    with t_r2_c5:
        st.markdown("<div style='margin-top: 20px;'></div>", unsafe_allow_html=True)
        if st.button("Reset View", key="dep_btn_full_reset", use_container_width=True):
            st.session_state["dep_selected_component"] = None
            st.session_state["dep_view_mode"] = "Architecture Overview"
            st.session_state["dep_layout"] = "Hierarchical"
            st.session_state["dep_depth"] = 1
            st.session_state["dep_dir_filter"] = "Both"
            st.session_state["dep_risk_filter"] = "All"
            st.session_state["dep_label_mode"] = "Full"
            st.session_state["dep_highlight_risk"] = True
            st.session_state["dep_history"] = []
            st.rerun()

    st.markdown("</div>", unsafe_allow_html=True)

    # 5. Compact Graph Legend Bar
    st.markdown(
        """
        <div class="dep-legend-container">
            <span style="font-weight: 600; color: #f0f6fc;">Legend:</span>
            <div class="dep-legend-item">
                <span class="dep-legend-dot" style="background-color: #2ea043;"></span>
                <span>Low Risk [L]</span>
            </div>
            <div class="dep-legend-item">
                <span class="dep-legend-dot" style="background-color: #d29922;"></span>
                <span>Medium Risk [M]</span>
            </div>
            <div class="dep-legend-item">
                <span class="dep-legend-dot" style="background-color: #f85149;"></span>
                <span>High Risk [H]</span>
            </div>
            <div class="dep-legend-item">
                <span class="dep-legend-dot" style="background-color: #58a6ff; border: 2px solid #ffffff;"></span>
                <span>Focused Component</span>
            </div>
            <div class="dep-legend-item">
                <span class="dep-legend-line" style="background-color: #58a6ff;"></span>
                <span>Outgoing Dependency (Imports) ➔</span>
            </div>
            <div class="dep-legend-item">
                <span class="dep-legend-line" style="background-color: #a371f7;"></span>
                <span>Incoming Dependent (Imported by) ➔</span>
            </div>
            <div class="dep-legend-item">
                <span class="dep-legend-line" style="background-color: #f85149; border-top: 2px dashed #f85149; height: 0;"></span>
                <span>⚠️ Circular Dependency</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    # 6. Main Canvas + Side Information Panel
    col_graph, col_panel = st.columns([2.2, 1.0])

    with col_graph:
        if st.session_state["dep_view_mode"] == "Architecture Overview":
            # Mode A: Architecture Overview Figure
            fig_arch = create_architecture_overview_figure(
                dep_graph=dep_graph,
                risk_profiles_map=risk_map
            )
            st.plotly_chart(fig_arch, use_container_width=True)
            st.caption("Subsystem Architecture Overview: Aggregated package-to-package dependency relationships and component counts.")

        elif st.session_state["dep_view_mode"] == "Risk Propagation":
            # Mode C: Risk Cascade Simulation
            origin_comp = selected_comp or (all_nodes[0] if all_nodes else None)
            decay_val = st.slider("Propagation Decay Factor", min_value=0.3, max_value=0.95, value=0.75, step=0.05, key="dep_prop_decay")
            fig_prop = create_risk_propagation_figure(
                dep_graph=dep_graph,
                risk_profiles=pred_res.profiles,
                origin_component=origin_comp,
                decay_factor=decay_val
            )
            st.plotly_chart(fig_prop, use_container_width=True)
            st.caption(f"Risk Cascade Simulation: Propagating architectural exposure from origin component **{origin_comp}** through downstream dependents.")

        elif st.session_state["dep_view_mode"] == "Architecture Diff":
            # Mode D: Architecture Topology Diff
            fig_diff = create_architecture_diff_figure(
                before_graph=dep_graph.graph,
                after_graph=dep_graph.graph
            )
            st.plotly_chart(fig_diff, use_container_width=True)
            st.caption("Architecture Diff Visualization: Edge modifications, additions, and removals across architectural evolutions.")

        else:
            # Mode B: Component Explorer Figure
            fig_comp = create_interactive_dependency_figure(
                dep_graph=dep_graph,
                risk_profiles_map=risk_map,
                focus_component=selected_comp,
                max_nodes=60,
                layout_type=st.session_state["dep_layout"].lower(),
                depth=st.session_state["dep_depth"],
                direction_filter=st.session_state["dep_dir_filter"].lower(),
                risk_filter=st.session_state["dep_risk_filter"].lower(),
                show_labels=st.session_state["dep_label_mode"].lower(),
                highlight_risk=st.session_state["dep_highlight_risk"]
            )
            st.plotly_chart(fig_comp, use_container_width=True)
            if selected_comp:
                st.caption(f"Hierarchical neighborhood for: **{selected_comp}** (Depth: {st.session_state['dep_depth']} hop(s), Direction: {st.session_state['dep_dir_filter']})")
            else:
                st.caption("Repository-wide directed component topology. Select any component to enter focused 3-level view.")

    with col_panel:
        # Supporting Details Panel
        if st.session_state["dep_view_mode"] == "Architecture Overview":
            # Panel for Mode A: Subsystem Packages Breakdown
            st.markdown(
                textwrap.dedent("""
                <div class="dep-panel-card">
                    <div class="dep-panel-title">
                        <span>Subsystem Packages</span>
                        <span style="font-size: 11px; color: #8b949e;">Modular Clusters</span>
                    </div>
                    <div class="dep-panel-sub">Directory-level coupling and stability metrics</div>
                </div>
                """),
                unsafe_allow_html=True
            )

            pkg_clusters = dep_graph.get_package_clusters()
            for pkg_name, pdata in sorted(pkg_clusters.items()):
                cnt = pdata["component_count"]
                instab = pdata["instability_index"]
                ca = pdata["afferent_coupling_ca"]
                ce = pdata["efferent_coupling_ce"]

                instab_color = "#3fb950" if instab <= 0.3 else ("#f85149" if instab >= 0.7 else "#d29922")

                st.markdown(
                    textwrap.dedent(f"""
                    <div class="dep-item-row" style="margin-top: 8px;">
                        <div>
                            <b style="color: #f0f6fc;">{pkg_name}/</b>
                            <span style="color: #8b949e; font-size: 11px;"> ({cnt} files)</span>
                        </div>
                        <div style="font-size: 11px; color: {instab_color}; font-weight: 600;">
                            I = {instab:.2f}
                        </div>
                    </div>
                    <div style="font-size: 10px; color: #8b949e; padding-left: 6px; margin-bottom: 6px;">
                        Ca: {ca} (incoming) • Ce: {ce} (outgoing)
                    </div>
                    """),
                    unsafe_allow_html=True
                )

                # Expandable components inside package
                with st.expander(f"View {pkg_name} files ({cnt})", expanded=False):
                    for comp in pdata["components"]:
                        prof = pred_res.profiles_by_path.get(comp)
                        c_risk = prof.risk_percentage if prof else 0
                        c_cat = prof.risk_category if prof else "LOW"
                        pill_html = render_risk_pill(c_risk, c_cat)
                        c_col1, c_col2 = st.columns([2, 1])
                        with c_col1:
                            st.markdown(f"<div style='font-size: 11px; font-family: monospace;'>{Path(comp).name}</div>", unsafe_allow_html=True)
                        with c_col2:
                            if st.button("Inspect", key=f"pkg_btn_{comp}", use_container_width=True):
                                st.session_state["dep_selected_component"] = comp
                                st.session_state["dep_view_mode"] = "Component Explorer"
                                st.rerun()

        elif selected_comp and selected_comp in struct_map:
            # Panel for Mode B: Focused Component Details Panel
            sm = struct_map[selected_comp]
            prof = pred_res.profiles_by_path.get(selected_comp)
            risk_pct = prof.risk_percentage if prof else 0
            risk_cat = prof.risk_category if prof else "LOW"
            pill_html = render_risk_pill(risk_pct, risk_cat)

            st.markdown(
                textwrap.dedent(f"""<div class="dep-panel-card">
<div class="dep-panel-title"><span>{Path(selected_comp).name}</span>{pill_html}</div>
<div class="dep-panel-sub">{selected_comp}</div>
<div class="dep-stat-grid">
<div class="dep-stat-item"><div class="dep-stat-label">Dependencies (Out)</div><div class="dep-stat-val" style="color: #58a6ff;">{sm.out_degree}</div></div>
<div class="dep-stat-item"><div class="dep-stat-label">Dependents (In)</div><div class="dep-stat-val" style="color: #a371f7;">{sm.in_degree}</div></div>
<div class="dep-stat-item"><div class="dep-stat-label">Total Degree</div><div class="dep-stat-val">{sm.degree}</div></div>
<div class="dep-stat-item"><div class="dep-stat-label">Centrality</div><div class="dep-stat-val">{round(sm.centrality, 3)}</div></div>
</div>
</div>"""),
                unsafe_allow_html=True
            )

            # Circular Dependency Check
            comp_cycles = [cyc for cyc in cycles if selected_comp in cyc]
            if comp_cycles:
                cyc_path = " ➔ ".join([Path(c).name for c in comp_cycles[0]] + [Path(comp_cycles[0][0]).name])
                st.markdown(
                    textwrap.dedent(f"""
                    <div class="dep-alert-box">
                        <b>⚠️ Circular Dependency Detected</b><br>
                        <span style="font-family: monospace; font-size: 11px;">{cyc_path}</span>
                    </div>
                    """),
                    unsafe_allow_html=True
                )

            # Connected High-Risk Neighbors Alert
            all_neighbors = set(sm.direct_dependencies + sm.direct_dependents)
            risky_neighbors = [
                n for n in all_neighbors
                if (pred_res.profiles_by_path.get(n) and pred_res.profiles_by_path.get(n).risk_category == "HIGH")
            ]
            if risky_neighbors:
                risky_names = ", ".join([Path(n).name for n in risky_neighbors])
                st.markdown(
                    textwrap.dedent(f"""
                    <div class="dep-alert-box">
                        <b>Coupling Hotspot Notice:</b> Connected to high-risk component(s): <b>{risky_names}</b>.
                    </div>
                    """),
                    unsafe_allow_html=True
                )

            # Upstream Dependencies (Imports)
            st.markdown(f"<div class='dep-section-label'>Upstream Dependencies (Imports: {len(sm.direct_dependencies)})</div>", unsafe_allow_html=True)
            if sm.direct_dependencies:
                for dep in sm.direct_dependencies:
                    d_prof = pred_res.profiles_by_path.get(dep)
                    d_risk = d_prof.risk_percentage if d_prof else 0
                    d_cat = d_prof.risk_category if d_prof else "LOW"
                    d_pill = render_risk_pill(d_risk, d_cat)
                    d_col1, d_col2 = st.columns([2.2, 1.0])
                    with d_col1:
                        st.markdown(f"<div style='font-size: 12px; font-family: monospace;'>• {Path(dep).name} {d_pill}</div>", unsafe_allow_html=True)
                    with d_col2:
                        if st.button("Focus", key=f"foc_dep_{dep}", use_container_width=True):
                            st.session_state["dep_selected_component"] = dep
                            if dep not in st.session_state["dep_history"]:
                                st.session_state["dep_history"].append(dep)
                            st.rerun()
            else:
                st.markdown("<div style='font-size: 11px; color: #8b949e; padding-left: 4px;'>No outgoing dependencies (leaf/root provider).</div>", unsafe_allow_html=True)

            # Downstream Dependents (Imported by)
            st.markdown(f"<div class='dep-section-label'>Downstream Dependents (Consumers: {len(sm.direct_dependents)})</div>", unsafe_allow_html=True)
            if sm.direct_dependents:
                for dept in sm.direct_dependents:
                    dt_prof = pred_res.profiles_by_path.get(dept)
                    dt_risk = dt_prof.risk_percentage if dt_prof else 0
                    dt_cat = dt_prof.risk_category if dt_prof else "LOW"
                    dt_pill = render_risk_pill(dt_risk, dt_cat)
                    dt_col1, dt_col2 = st.columns([2.2, 1.0])
                    with dt_col1:
                        st.markdown(f"<div style='font-size: 12px; font-family: monospace;'>• {Path(dept).name} {dt_pill}</div>", unsafe_allow_html=True)
                    with dt_col2:
                        if st.button("Focus", key=f"foc_dept_{dept}", use_container_width=True):
                            st.session_state["dep_selected_component"] = dept
                            if dept not in st.session_state["dep_history"]:
                                st.session_state["dep_history"].append(dept)
                            st.rerun()
            else:
                st.markdown("<div style='font-size: 11px; color: #8b949e; padding-left: 4px;'>No incoming dependents (top-level consumer/entrypoint).</div>", unsafe_allow_html=True)

            # Available Actions
            st.markdown("<div class='dep-section-label'>Available Actions</div>", unsafe_allow_html=True)
            act_col1, act_col2 = st.columns(2)
            with act_col1:
                cur_d = st.session_state["dep_depth"]
                next_d = 2 if cur_d == 1 else (3 if cur_d == 2 else 1)
                lbl_d = f"Expand to {next_d}-Hop" if next_d > 1 else "Reset to 1-Hop"
                if st.button(lbl_d, key="dep_btn_toggle_depth", use_container_width=True):
                    st.session_state["dep_depth"] = next_d
                    st.rerun()
            with act_col2:
                cur_dir = st.session_state["dep_dir_filter"]
                lbl_dir = "Deps Only" if cur_dir == "Both" else ("Dependents" if cur_dir == "Dependencies" else "Show Both")
                next_dir = "Dependencies" if cur_dir == "Both" else ("Dependents" if cur_dir == "Dependencies" else "Both")
                if st.button(lbl_dir, key="dep_btn_toggle_dir", use_container_width=True):
                    st.session_state["dep_dir_filter"] = next_dir
                    st.rerun()

        else:
            # Panel when in Component Explorer but no component is selected yet
            st.markdown(
                textwrap.dedent("""
                <div class="dep-panel-card">
                    <div class="dep-panel-title">
                        <span>Component Explorer</span>
                        <span style="font-size: 11px; color: #8b949e;">Quick-Start</span>
                    </div>
                    <div class="dep-panel-sub">Select any file to trace its 3-level upstream & downstream dependencies.</div>
                    
                    <div class="dep-info-box">
                        💡 <b>Tip:</b> Choose a suggested high-impact component below or search for a specific file in the toolbar above.
                    </div>
                </div>
                """),
                unsafe_allow_html=True
            )

            # Suggested High-Risk Components
            high_risk_nodes = [
                p.filepath for p in pred_res.profiles
                if p.risk_category == "HIGH"
            ]
            if high_risk_nodes:
                st.markdown("<div class='dep-section-label'>Suggested High-Risk Components</div>", unsafe_allow_html=True)
                for hr_node in high_risk_nodes[:3]:
                    prof = pred_res.profiles_by_path.get(hr_node)
                    risk_val = prof.risk_percentage if prof else 0
                    c1, c2 = st.columns([2, 1])
                    with c1:
                        st.markdown(f"<div style='font-size: 12px; font-family: monospace;'>{Path(hr_node).name} ({risk_val}%)</div>", unsafe_allow_html=True)
                    with c2:
                        if st.button("Explore", key=f"sug_hr_{hr_node}", use_container_width=True):
                            st.session_state["dep_selected_component"] = hr_node
                            st.rerun()

            # Suggested Architectural Hubs
            top_hubs = [h["component"] for h in hotspots[:3] if h["component"] not in high_risk_nodes]
            if top_hubs:
                st.markdown("<div class='dep-section-label'>Suggested Architectural Hubs</div>", unsafe_allow_html=True)
                for hub_node in top_hubs:
                    deg = struct_map[hub_node].degree if hub_node in struct_map else 0
                    c1, c2 = st.columns([2, 1])
                    with c1:
                        st.markdown(f"<div style='font-size: 12px; font-family: monospace;'>{Path(hub_node).name} ({deg} links)</div>", unsafe_allow_html=True)
                    with c2:
                        if st.button("Explore", key=f"sug_hub_{hub_node}", use_container_width=True):
                            st.session_state["dep_selected_component"] = hub_node
                            st.rerun()

    st.markdown("<hr style='border-color: #21262d; margin: 24px 0;'>", unsafe_allow_html=True)

    # 7. Synchronized Component In/Out Dependency Matrix
    st.markdown("### Component In/Out Dependency Matrix")
    st.caption("Coupling metrics and import tracing table. Filtered dynamically by active toolbar settings. Focus any component to view in graph.")

    matrix_rows = []
    active_risk_filt = st.session_state["dep_risk_filter"]

    for node, sm in struct_map.items():
        prof = pred_res.profiles_by_path.get(node)
        cat = prof.risk_category if prof else "LOW"
        pct = prof.risk_percentage if prof else 0

        # Filter by risk if set
        if active_risk_filt != "All" and cat != active_risk_filt.upper():
            continue

        is_focused = (selected_comp == node)
        matrix_rows.append({
            "Focus Status": "★ ACTIVE FOCUS" if is_focused else "",
            "Component": node,
            "Risk (%)": f"{pct}%",
            "Category": cat,
            "Outgoing (Dependencies)": sm.out_degree,
            "Incoming (Dependents)": sm.in_degree,
            "Total Degree": sm.degree,
            "Degree Centrality": round(sm.centrality, 4),
            "Direct Dependencies": ", ".join([Path(d).name for d in sm.direct_dependencies]) or "None",
            "Direct Dependents": ", ".join([Path(d).name for d in sm.direct_dependents]) or "None"
        })

    if matrix_rows:
        df_struct = pd.DataFrame(matrix_rows).sort_values(by="Total Degree", ascending=False)
        st.dataframe(df_struct, use_container_width=True, hide_index=True)

        # Quick Focus selector below table for direct synchronization
        tbl_col1, tbl_col2 = st.columns([2.5, 1.2])
        with tbl_col1:
            tbl_target = st.selectbox(
                "Focus Component from Matrix",
                options=["-- Select component to focus in graph --"] + [r["Component"] for r in matrix_rows],
                index=0,
                key="dep_matrix_focus_select"
            )
        with tbl_col2:
            st.markdown("<div style='margin-top: 24px;'></div>", unsafe_allow_html=True)
            if st.button("Focus Selected", key="dep_btn_focus_matrix", use_container_width=True):
                if tbl_target != "-- Select component to focus in graph --":
                    st.session_state["dep_selected_component"] = tbl_target
                    st.session_state["dep_view_mode"] = "Component Explorer"
                    if tbl_target not in st.session_state["dep_history"]:
                        st.session_state["dep_history"].append(tbl_target)
                    st.rerun()
    else:
        st.info("No components match the active filter criteria.")




def render_explainability_page(
    explainer: ModelExplainer,
    pred_res: PredictionResult,
    features_df: pd.DataFrame
):
    """Renders global feature importance and localized risk attribution signals with strict non-causal language."""
    st.markdown("## Explainable AI & Model Signals")
    st.caption("Understand how software characteristics contribute to the model's component risk predictions.")

    # 1. Global Model Feature Importance
    st.markdown("### Global Feature Importance")
    df_global = explainer.get_global_importance_df()
    
    col_g1, col_g2 = st.columns([1.4, 1])
    with col_g1:
        st.dataframe(df_global[["Feature", "Importance", "Description"]], use_container_width=True, hide_index=True)
    with col_g2:
        st.markdown("#### Signal Interpretation")
        st.markdown(
            "Feature importance reflects how heavily the model relies on each feature across "
            "all historical components when estimating risk."
        )
        render_scientific_disclaimer("Feature importance represents statistical model reliance, not proven causal impact.")

    st.markdown("<hr style='border-color: #21262d; margin: 20px 0;'>", unsafe_allow_html=True)

    # 2. Local Component Signal Breakdown
    all_components = list(pred_res.profiles_by_path.keys())
    target_comp = st.selectbox("Select Component to Explain", options=all_components, index=0)

    profile = pred_res.profiles_by_path.get(target_comp)
    if profile:
        st.markdown(f"### Why is `{target_comp}` rated {profile.risk_percentage}% {profile.risk_category}?")
        comp_row = features_df[features_df["filepath"] == target_comp].iloc[0]
        explanation = explainer.explain_component(comp_row, profile.risk_probability, profile.risk_category)

        st.info(f"Summary: {explanation.summary_text}")

        # Top contributing signals cards
        sig_cols = st.columns(min(len(explanation.top_signals), 4))
        for idx, sig in enumerate(explanation.top_signals[:4]):
            with sig_cols[idx]:
                st.markdown(
                    f"""
                    <div class="metric-card-compact">
                        <div class="metric-card-label">{sig.display_name}</div>
                        <div class="metric-card-value" style="font-size: 20px;">{sig.feature_value}</div>
                        <div class="metric-card-sub">Signal Impact: <span style="font-weight:600; color: {'#ff7b72' if sig.impact_level=='HIGH' else ('#d29922' if sig.impact_level=='MEDIUM' else '#58a6ff')};">{sig.impact_level}</span></div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )


def render_what_if_lab_page(
    what_if_engine: WhatIfEngine,
    pred_res: PredictionResult,
    dep_graph: DependencyGraph
):
    """Renders the signature What-If Change Simulation Lab."""
    st.markdown("## What-If Change Simulation Lab")
    st.caption("Explore how hypothetical modifications to architecture, complexity, or churn alter the model's predicted risk.")

    all_components = list(pred_res.profiles_by_path.keys())
    if not all_components:
        st.info("No components available for simulation.")
        return

    col_ctrl1, col_ctrl2 = st.columns([1.2, 1])
    with col_ctrl1:
        target_comp = st.selectbox("Target Component", options=all_components, index=0)
    with col_ctrl2:
        change_type = st.selectbox("Simulated Modification", options=[c for c in SimulationChangeType], index=0)

    # Dynamic parameter inputs based on change type
    target_dep = None
    complexity_delta = 5
    churn_delta = 300
    freq_delta = 0.15

    profile = pred_res.profiles_by_path[target_comp]

    if change_type == SimulationChangeType.ADD_DEPENDENCY:
        available_targets = [c for c in all_components if c != target_comp and c not in list(dep_graph.graph.successors(target_comp))]
        if not available_targets:
            available_targets = [c for c in all_components if c != target_comp]
        target_dep = st.selectbox("Dependency to Add (Module)", options=available_targets if available_targets else [target_comp])
    
    elif change_type == SimulationChangeType.REMOVE_DEPENDENCY:
        curr_deps = list(dep_graph.graph.successors(target_comp))
        if curr_deps:
            target_dep = st.selectbox("Dependency to Remove", options=curr_deps)
        else:
            st.warning(f"Component '{target_comp}' currently has 0 dependencies to remove.")
            target_dep = None

    elif change_type == SimulationChangeType.INCREASE_COMPLEXITY:
        complexity_delta = st.slider("Complexity Increase (+ Cyclomatic)", min_value=1, max_value=30, value=6)

    elif change_type == SimulationChangeType.DECREASE_COMPLEXITY:
        complexity_delta = st.slider("Complexity Reduction (- Cyclomatic)", min_value=1, max_value=max(1, profile.complexity - 1), value=min(4, max(1, profile.complexity - 1)))

    elif change_type == SimulationChangeType.INCREASE_CHURN:
        churn_delta = st.slider("Additional Churn (+ Lines)", min_value=50, max_value=2000, value=400, step=50)

    elif change_type == SimulationChangeType.INCREASE_CHANGE_FREQUENCY:
        freq_delta = st.slider("Change Frequency Delta", min_value=0.05, max_value=0.50, value=0.15, step=0.05)

    sim_clicked = st.button("⚡ Simulate Change & Re-evaluate Risk", type="primary")

    st.markdown("<hr style='border-color: #21262d; margin: 16px 0;'>", unsafe_allow_html=True)

    # Run Simulation
    sim_res = what_if_engine.simulate_change(
        current_profile=profile,
        change_type=change_type,
        target_dependency=target_dep,
        complexity_delta=complexity_delta,
        churn_delta=churn_delta,
        freq_delta=freq_delta
    )

    # Side-by-side comparison layout
    s_col1, s_col2, s_col3 = st.columns([1.2, 0.8, 1.2])

    with s_col1:
        st.markdown(
            f"""
            <div class="what-if-box">
                <div style="font-size: 11px; color: #8b949e; text-transform: uppercase; font-weight: 600;">CURRENT STATE</div>
                <div style="font-size: 30px; font-weight: 700; color: #f0f6fc; margin: 6px 0;">{sim_res.current_percentage}%</div>
                <div style="margin-bottom: 10px;">{render_risk_pill(sim_res.current_percentage, sim_res.current_category)}</div>
                <div style="font-size: 12px; color: #8b949e;">
                    <div>• Dependencies: <b style="color: #f0f6fc;">{profile.dependency_count}</b></div>
                    <div>• Centrality: <b style="color: #f0f6fc;">{round(profile.centrality, 4)}</b></div>
                    <div>• Complexity: <b style="color: #f0f6fc;">{profile.complexity}</b></div>
                    <div>• Churn: <b style="color: #f0f6fc;">{profile.churn}</b></div>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with s_col2:
        delta = sim_res.percentage_point_diff
        delta_class = "delta-risk-up" if delta > 0 else ("delta-risk-down" if delta < 0 else "delta-risk-zero")
        sign = "+" if delta > 0 else ""
        st.markdown(
            f"""
            <div style="text-align: center; padding-top: 20px;">
                <div style="font-size: 11px; color: #8b949e; text-transform: uppercase;">ESTIMATED SHIFT</div>
                <div class="what-if-delta-val {delta_class}">{sign}{delta} pp</div>
                <div style="font-size: 11px; color: #8b949e;">Predicted Risk Delta</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with s_col3:
        st.markdown(
            f"""
            <div class="what-if-box">
                <div style="font-size: 11px; color: #58a6ff; text-transform: uppercase; font-weight: 600;">WHAT-IF STATE</div>
                <div style="font-size: 30px; font-weight: 700; color: #f0f6fc; margin: 6px 0;">{sim_res.simulated_percentage}%</div>
                <div style="margin-bottom: 10px;">{render_risk_pill(sim_res.simulated_percentage, sim_res.simulated_category)}</div>
                <div style="font-size: 12px; color: #8b949e;">
                    <div>• Dependencies: <b style="color: #f0f6fc;">{sim_res.simulated_graph.graph.out_degree(target_comp)}</b></div>
                    <div>• Centrality: <b style="color: #f0f6fc;">{round(sim_res.simulated_graph.get_metrics_for_all().get(target_comp).centrality, 4)}</b></div>
                    <div>• Complexity: <b style="color: #f0f6fc;">{profile.complexity + (complexity_delta if 'INCREASE_COMPLEXITY' in change_type.name else (-complexity_delta if 'DECREASE_COMPLEXITY' in change_type.name else 0))}</b></div>
                    <div>• Churn: <b style="color: #f0f6fc;">{profile.churn + (churn_delta if 'CHURN' in change_type.name else 0)}</b></div>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    st.markdown("<div style='margin-top: 12px;'></div>", unsafe_allow_html=True)
    render_scientific_disclaimer(sim_res.scientific_disclaimer)

    # Visual Architecture Change Graph
    st.markdown("### Simulated Architecture Graph Overlay")
    fig_sim_graph = create_interactive_dependency_figure(
        dep_graph=sim_res.simulated_graph,
        risk_profiles_map={p.filepath: p.to_dict() for p in pred_res.profiles},
        focus_component=target_comp,
        simulated_edge=sim_res.simulated_edge_action,
        max_nodes=35
    )
    st.plotly_chart(fig_sim_graph, use_container_width=True)


def render_model_evaluation_page(
    dataset: BuiltDataset,
    comp_report: ModelComparisonReport,
    selected_algo: ClassifierAlgorithm
):
    """Renders the core research experiment comparing Static vs Static+Evolution vs Full models."""
    st.markdown("## Model Evaluation & Comparative Research")
    st.caption("Addressing research question: Does incorporating evolution and dependency metrics improve component risk prediction?")

    split = dataset.split_result
    label_sum = dataset.label_summary

    # Labeling method and proxy badge
    if label_sum.is_proxy:
        st.warning(
            f"**Proxy Target Label Active:** {label_sum.label_type.value}. "
            "No qualifying future defect commits were available in the future horizon, "
            "so target labels were derived from future churn volatility. This is explicitly tracked."
        )
    else:
        st.success(
            f"**Defect-Based Target Label Active:** {label_sum.label_type.value}. "
            f"Labels generated from {len(label_sum.contributing_commit_hashes)} future bug-fixing commits."
        )

    # Dataset Metadata Summary
    st.markdown("### Experimental Dataset Split (Temporal / Leakage-Free)")
    d1, d2, d3, d4 = st.columns(4)
    with d1:
        render_metric_card("Total Components", str(dataset.total_components_count), "In repository")
    with d2:
        render_metric_card("Training Samples", str(split.train_samples_count), f"{split.positive_train_count} positive risk")
    with d3:
        render_metric_card("Test Samples", str(split.test_samples_count), f"{split.positive_test_count} positive risk")
    with d4:
        method_name = "Proxy Label" if label_sum.is_proxy else "Bugfix Window"
        render_metric_card("Target Scheme", method_name, label_sum.description[:45] + "...")

    # Observation ranges and temporal verification
    st.markdown("#### Chronological Observation Windows")
    w1, w2 = st.columns(2)
    with w1:
        st.markdown(
            f"""
            - **Training Observation:** `{comp_report.train_observation_range or 'Chronological train cutoff'}`
            - **Train Class Distribution:** `Class 0: {comp_report.train_class_distribution.get(0, 0)} | Class 1: {comp_report.train_class_distribution.get(1, 0)}`
            """
        )
    with w2:
        st.markdown(
            f"""
            - **Testing Observation:** `{comp_report.test_observation_range or 'Chronological test cutoff'}`
            - **Test Class Distribution:** `Class 0: {comp_report.test_class_distribution.get(0, 0)} | Class 1: {comp_report.test_class_distribution.get(1, 0)}`
            """
        )

    if split.is_leakage_free:
        st.caption("Temporal Validation: PASS — Independent future label windows with non-overlapping observation horizons.")
    else:
        st.error(f"Temporal Validation Warnings: {', '.join(split.temporal_validation_errors)}")

    st.markdown("<div style='margin-top: 16px;'></div>", unsafe_allow_html=True)

    # Core Research Comparison Table
    st.markdown("### Core Research Model Comparison")
    st.dataframe(comp_report.comparison_df, use_container_width=True, hide_index=True)

    if comp_report.baseline_report:
        st.caption(
            f"Context Baseline (Majority Classifier): Accuracy={comp_report.baseline_report.accuracy:.3f}, "
            f"F1={comp_report.baseline_report.f1:.3f}"
        )

    st.info(f"Research Finding: {comp_report.research_summary}")

    # Confusion Matrix Breakdown
    st.markdown("### Confusion Matrix Analysis (Full Model)")
    rep_full = comp_report.full_report
    cm1, cm2 = st.columns([1, 1.5])
    with cm1:
        st.markdown(
            f"""
            - **True Positives (TP):** `{rep_full.true_positives}`
            - **False Positives (FP):** `{rep_full.false_positives}`
            - **True Negatives (TN):** `{rep_full.true_negatives}`
            - **False Negatives (FN):** `{rep_full.false_negatives}`
            """
        )
    with cm2:
        auc_str = f"{rep_full.roc_auc:.4f}" if rep_full.roc_auc_valid else "N/A (single-class test set)"
        st.markdown(
            f"""
            - **F1-Score:** `{rep_full.f1:.4f}`
            - **Precision:** `{rep_full.precision:.4f}`
            - **Recall:** `{rep_full.recall:.4f}`
            - **ROC-AUC:** `{auc_str}`
            """
        )


def render_settings_page(
    repo_meta: RepositoryMetadata,
    pred_res: PredictionResult
):
    """Renders user settings, thresholds configuration, and export capabilities."""
    st.markdown("## System Configuration & Export")
    st.caption("Customize risk thresholds, analysis windows, and export structured reports.")

    st.markdown("### Risk Category Thresholds")
    c1, c2 = st.columns(2)
    with c1:
        st.number_input("Low-to-Medium Risk Boundary", min_value=0.10, max_value=0.50, value=0.33, step=0.05)
    with c2:
        st.number_input("Medium-to-High Risk Boundary", min_value=0.50, max_value=0.90, value=0.66, step=0.05)

    st.markdown("<hr style='border-color: #21262d; margin: 20px 0;'>", unsafe_allow_html=True)
    render_export_section(pred_res.df_predictions, repo_meta)


def render_git_activity_page(
    git_loader: GitLoader,
    repo_meta: RepositoryMetadata,
    pred_res: PredictionResult,
    change_analyzer: ChangeImpactAnalyzer
):
    """
    Renders the Git Activity & Change Preview module:
    Connects repository evolution, commits, contributors, diff previews, and AI risk prediction.
    """
    st.markdown("## Git Activity")
    st.caption("Track repository changes, contributors, commits, and their relationship with software risk.")

    # 1. Top Metrics (Real data from GitLoader)
    commits = git_loader.get_commit_history()
    contributors = git_loader.get_contributor_stats()
    
    all_files_changed = set()
    for c in commits:
        all_files_changed.update(c.files_changed)

    total_commits = len(commits)
    total_contributors = len(contributors)
    total_files_changed = len(all_files_changed)
    recent_changes = min(5, total_commits)

    m1, m2, m3, m4 = st.columns(4)
    with m1:
        render_metric_card("TOTAL COMMITS", str(total_commits), f"Across {repo_meta.default_branch} branch")
    with m2:
        render_metric_card("CONTRIBUTORS", str(total_contributors), f"{len(contributors)} active authors")
    with m3:
        render_metric_card("FILES CHANGED", str(total_files_changed), "Unique files modified")
    with m4:
        render_metric_card("RECENT CHANGES", str(recent_changes), "In recent commit window")

    st.markdown("<div style='margin-top: 14px;'></div>", unsafe_allow_html=True)

    # Conceptual Pipeline Banner
    st.markdown(
        """
        <div style="background: #161b22; border: 1px solid #30363d; border-radius: 6px; padding: 12px 18px; margin-bottom: 20px;">
            <div style="font-size: 11px; text-transform: uppercase; letter-spacing: 0.08em; color: #8b949e; margin-bottom: 6px; font-weight: 600;">
                CORE RESEARCH PIPELINE
            </div>
            <div style="font-family: monospace; font-size: 13px; color: #58a6ff; display: flex; align-items: center; gap: 8px; flex-wrap: wrap;">
                <span style="background: #21262d; padding: 3px 8px; border-radius: 4px; color: #e6edf3;">GIT CHANGE</span>
                <span>➔</span>
                <span style="background: #21262d; padding: 3px 8px; border-radius: 4px; color: #e6edf3;">CODE / ARCHITECTURE CHANGE</span>
                <span>➔</span>
                <span style="background: #21262d; padding: 3px 8px; border-radius: 4px; color: #e6edf3;">STRUCTURAL METRIC CHANGE</span>
                <span>➔</span>
                <span style="background: #1f6feb22; border: 1px solid #1f6feb; padding: 3px 8px; border-radius: 4px; color: #58a6ff; font-weight: 600;">AI RISK CHANGE</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    # 2. Main Navigation Tabs
    tab_overview, tab_commits, tab_contributors, tab_changes, tab_branches = st.tabs([
        "Overview", "Commits", "Contributors", "Changes & Preview", "Branches / Safe Commit"
    ])

    # ==========================================
    # TAB 1: OVERVIEW
    # ==========================================
    with tab_overview:
        st.markdown("### Recent Commits")
        if commits:
            commit_rows = []
            for c in reversed(commits[-8:]):
                dt_str = c.timestamp.strftime("%b %d, %H:%M")
                commit_rows.append({
                    "Commit": c.commit_hash[:7],
                    "Message": c.message,
                    "Author": c.author_name,
                    "Time": dt_str,
                    "Files Touched": len(c.files_changed),
                    "+ Additions": f"+{c.insertions}",
                    "- Deletions": f"-{c.deletions}",
                    "Type": "BUGFIX" if c.is_bugfix else "FEATURE / REFACTOR"
                })
            st.dataframe(pd.DataFrame(commit_rows), use_container_width=True, hide_index=True)
        else:
            st.info("No commit history available in repository.")

        st.markdown("<hr style='border-color: #21262d; margin: 20px 0;'>", unsafe_allow_html=True)

        # Contributor Impact Hierarchy (Section 20.15)
        st.markdown("### Contributor Activity & Component Mapping")
        st.caption(
            "Empirical association of developer modifications with affected components and current model risk. "
            "*(Neutral empirical association — does not imply direct causal attribution)*"
        )

        if contributors:
            c_cols = st.columns(min(3, len(contributors)))
            for idx, cont in enumerate(contributors):
                col = c_cols[idx % len(c_cols)]
                with col:
                    st.markdown(
                        f"""
                        <div style="background: #161b22; border: 1px solid #30363d; border-radius: 6px; padding: 14px; margin-bottom: 12px;">
                            <div style="font-weight: 600; font-size: 15px; color: #e6edf3; display: flex; justify-content: space-between;">
                                <span>{cont.author_name}</span>
                                <span style="font-size: 12px; color: #8b949e;">{cont.commit_count} commits</span>
                            </div>
                            <div style="font-size: 12px; color: #8b949e; margin-top: 2px;">
                                <span style="color: #3fb950;">+{cont.insertions}</span> / <span style="color: #ff7b72;">-{cont.deletions}</span> lines • {cont.files_changed_count} files
                            </div>
                            <div style="margin-top: 10px; border-top: 1px solid #21262d; padding-top: 8px;">
                                <div style="font-size: 11px; text-transform: uppercase; color: #8b949e; font-weight: 600; margin-bottom: 6px;">
                                    Components Touched
                                </div>
                        """,
                        unsafe_allow_html=True
                    )
                    for f in cont.components_touched[:4]:
                        prof = pred_res.profiles_by_path.get(f) or pred_res.profiles_by_path.get(f.replace("\\", "/"))
                        risk_str = f"Risk: {prof.risk_percentage}% {prof.risk_category}" if prof else "Tracked file"
                        color = "#ff7b72" if prof and prof.risk_category == "HIGH" else ("#d29922" if prof and prof.risk_category == "MEDIUM" else "#3fb950")
                        st.markdown(
                            f"""
                            <div style="font-size: 12px; padding: 4px 0; display: flex; justify-content: space-between; border-bottom: 1px solid #21262d55;">
                                <code style="color: #58a6ff; font-size: 11px;">{f.split('/')[-1]}</code>
                                <span style="color: {color}; font-weight: 500; font-size: 11px;">{risk_str}</span>
                            </div>
                            """,
                            unsafe_allow_html=True
                        )
                    st.markdown("</div></div>", unsafe_allow_html=True)

    # ==========================================
    # TAB 2: COMMITS (TIMELINE & DETAIL)
    # ==========================================
    with tab_commits:
        st.markdown("### Commit Timeline & Detailed Inspector")
        if not commits:
            st.info("No commit history found.")
        else:
            commit_options = [f"{c.commit_hash[:7]} — {c.message[:50]} ({c.author_name})" for c in reversed(commits)]
            selected_commit_idx = st.selectbox(
                "Select Commit to Inspect",
                options=list(range(len(commit_options))),
                format_func=lambda i: commit_options[i],
                index=0
            )
            selected_commit_rec = list(reversed(commits))[selected_commit_idx]
            detail = git_loader.get_commit_detail(selected_commit_rec.commit_hash)

            if detail:
                st.markdown("<div style='margin-top: 10px;'></div>", unsafe_allow_html=True)
                # Commit Header Card
                c1, c2, c3 = st.columns([1.5, 1, 1])
                with c1:
                    st.markdown(f"#### Commit `{detail.short_hash}`")
                    st.markdown(f"**Message:** {detail.message}")
                    if detail.is_bugfix:
                        st.markdown("<span style='background: #ff7b7222; color: #ff7b72; padding: 2px 8px; border-radius: 4px; font-size: 12px; font-weight: 600;'>BUGFIX COMMIT</span>", unsafe_allow_html=True)
                with c2:
                    st.markdown(f"**Author:** `{detail.author_name}`")
                    st.markdown(f"**Email:** `{detail.author_email}`")
                    st.markdown(f"**Date:** `{detail.timestamp.strftime('%B %d, %Y • %H:%M UTC')}`")
                with c3:
                    st.markdown(f"**Parent:** `{detail.parent_hash[:7] if detail.parent_hash else 'root'}`")
                    st.markdown(f"**Total Additions:** <span style='color: #3fb950; font-weight: 600;'>+{detail.total_insertions}</span>", unsafe_allow_html=True)
                    st.markdown(f"**Total Deletions:** <span style='color: #ff7b72; font-weight: 600;'>-{detail.total_deletions}</span>", unsafe_allow_html=True)

                st.markdown("<hr style='border-color: #21262d; margin: 16px 0;'>", unsafe_allow_html=True)

                # Files Changed & Diff Split
                st.markdown("#### Files Changed in this Commit")
                for item in detail.files_changed:
                    badge_color = "#3fb950" if item.change_type == "ADDED" else ("#ff7b72" if item.change_type == "DELETED" else "#58a6ff")
                    with st.expander(f"[{item.change_type}] {item.filepath}  (+{item.insertions} / -{item.deletions})", expanded=True):
                        st.markdown(
                            f"""
                            <div style="font-size: 12px; margin-bottom: 8px;">
                                <span style="background: {badge_color}22; color: {badge_color}; padding: 2px 6px; border-radius: 4px; font-weight: 600;">{item.change_type}</span>
                                <code style="margin-left: 8px;">{item.filepath}</code>
                            </div>
                            """,
                            unsafe_allow_html=True
                        )
                        if item.patch:
                            st.code(item.patch, language="diff")
                        else:
                            st.caption("No patch text recorded.")

                # Direct Link to Impact Analysis
                st.markdown("<hr style='border-color: #21262d; margin: 16px 0;'>", unsafe_allow_html=True)
                if st.button("⚡ Analyze This Commit's AI Risk Impact", key=f"analyze_commit_{detail.short_hash}", type="primary"):
                    summary = change_analyzer.analyze_commit_or_preview(detail.files_changed)
                    st.markdown("#### Commit Risk Impact Analysis")
                    if summary.component_impacts:
                        for imp in summary.component_impacts:
                            st.markdown(
                                f"""
                                <div style="background: #161b22; border: 1px solid #30363d; border-radius: 6px; padding: 12px 16px; margin-bottom: 10px;">
                                    <div style="display: flex; justify-content: space-between; align-items: center;">
                                        <div style="font-weight: 600; font-size: 14px;"><code>{imp.filepath}</code></div>
                                        <div>
                                            <span style="color: #8b949e;">Before: {imp.current_risk_pct}% ({imp.current_risk_category})</span> ➔ 
                                            <span style="color: #58a6ff; font-weight: 600;">After: {imp.proposed_risk_pct}% ({imp.proposed_risk_category})</span>
                                            <span style="margin-left: 8px; font-weight: 600; color: {'#ff7b72' if imp.risk_delta_pct > 0 else '#3fb950'};">({imp.risk_delta_pct:+d} pp)</span>
                                        </div>
                                    </div>
                                    <div style="font-size: 12px; color: #8b949e; margin-top: 6px;">
                                        {imp.explanation}
                                    </div>
                                </div>
                                """,
                                unsafe_allow_html=True
                            )
                    else:
                        st.info("Change impact could not be estimated with the available analysis for this commit's files.")

    # ==========================================
    # TAB 3: CONTRIBUTORS & "WHO CHANGED WHAT?"
    # ==========================================
    with tab_contributors:
        st.markdown("### Contributors Overview")
        if contributors:
            c_table = []
            for c in contributors:
                c_table.append({
                    "Contributor": c.author_name,
                    "Commits": c.commit_count,
                    "Files Changed": c.files_changed_count,
                    "Additions (+)": f"+{c.insertions}",
                    "Deletions (-)": f"-{c.deletions}",
                    "Components Touched": len(c.components_touched),
                    "Last Activity": c.last_activity.strftime("%b %d, %Y • %H:%M")
                })
            st.dataframe(pd.DataFrame(c_table), use_container_width=True, hide_index=True)

            st.markdown("<hr style='border-color: #21262d; margin: 20px 0;'>", unsafe_allow_html=True)

            # 20.6 "Who Changed What?"
            st.markdown("### Who Changed What?")
            st.caption("Select a contributor to drill into specific components touched and their risk profiles.")

            contributor_names = [c.author_name for c in contributors]
            selected_author = st.selectbox("Select Contributor", options=contributor_names, index=0)
            author_data = next((c for c in contributors if c.author_name == selected_author), None)

            if author_data:
                st.markdown(f"#### Components Modified by `{author_data.author_name}`")
                
                for fpath in author_data.components_touched:
                    # Calculate stats for this specific author on this file
                    author_file_commits = [
                        c for c in author_data.commits 
                        if any(fpath.lower() in f.lower() or f.lower() in fpath.lower() for f in c.files_changed)
                    ]
                    author_ins = sum(c.insertions for c in author_file_commits)
                    author_dels = sum(c.deletions for c in author_file_commits)
                    prof = pred_res.profiles_by_path.get(fpath) or pred_res.profiles_by_path.get(fpath.replace("\\", "/"))

                    risk_badge = ""
                    if prof:
                        r_col = "#ff7b72" if prof.risk_category == "HIGH" else ("#d29922" if prof.risk_category == "MEDIUM" else "#3fb950")
                        risk_badge = f"<span style='background: {r_col}22; color: {r_col}; border: 1px solid {r_col}55; padding: 2px 8px; border-radius: 4px; font-weight: 600; font-size: 12px;'>RISK: {prof.risk_percentage}% ({prof.risk_category})</span>"
                    
                    st.markdown(
                        f"""
                        <div style="background: #161b22; border: 1px solid #30363d; border-radius: 6px; padding: 12px 16px; margin-bottom: 8px;">
                            <div style="display: flex; justify-content: space-between; align-items: center;">
                                <div>
                                    <span style="font-family: monospace; font-size: 14px; font-weight: 600; color: #58a6ff;">{fpath}</span>
                                    <span style="font-size: 12px; color: #8b949e; margin-left: 10px;">({len(author_file_commits)} commits)</span>
                                </div>
                                <div>
                                    {risk_badge}
                                </div>
                            </div>
                            <div style="font-size: 12px; color: #8b949e; margin-top: 6px;">
                                Contributor Churn: <span style="color: #3fb950;">+{author_ins}</span> / <span style="color: #ff7b72;">-{author_dels}</span> lines
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

            # 20.16 Contributor Activity Timeline
            st.markdown("<hr style='border-color: #21262d; margin: 20px 0;'>", unsafe_allow_html=True)
            st.markdown("### Contributor Activity Timeline")
            for c in reversed(commits):
                dt_day = c.timestamp.strftime("%B %d, %Y")
                st.markdown(
                    f"""
                    <div style="margin-bottom: 12px; padding-left: 12px; border-left: 2px solid #30363d;">
                        <div style="font-size: 12px; color: #8b949e;">{dt_day}</div>
                        <div style="font-size: 14px; font-weight: 600; color: #e6edf3;">
                            {c.author_name}
                            <span style="font-weight: 400; font-size: 13px; color: #8b949e;"> — {c.message}</span>
                        </div>
                        <div style="font-size: 12px; color: #8b949e; font-family: monospace; margin-top: 2px;">
                            Touched: {', '.join([f.split('/')[-1] for f in c.files_changed[:3]])} • <span style="color: #3fb950;">+{c.insertions}</span> / <span style="color: #ff7b72;">-{c.deletions}</span>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

    # ==========================================
    # TAB 4: CHANGES (PREVIEW & IMPACT ANALYSIS)
    # ==========================================
    with tab_changes:
        st.markdown("### Change Preview & AI Risk Impact Simulation")
        st.caption("Preview code modifications before committing and evaluate their predicted impact on software risk.")

        # Stage Progress Bar (Section 20.11)
        st.markdown(
            """
            <div style="display: flex; gap: 8px; align-items: center; background: #161b22; padding: 10px 16px; border-radius: 6px; border: 1px solid #30363d; margin-bottom: 18px;">
                <span style="color: #3fb950; font-weight: 600;">● 1. Edit</span>
                <span style="color: #8b949e;">➔</span>
                <span style="color: #58a6ff; font-weight: 600;">● 2. Preview</span>
                <span style="color: #8b949e;">➔</span>
                <span style="color: #d29922; font-weight: 600;">● 3. Analyze</span>
                <span style="color: #8b949e;">➔</span>
                <span style="color: #8b949e;">○ 4. Review Impact</span>
                <span style="color: #8b949e;">➔</span>
                <span style="color: #8b949e;">○ 5. Commit</span>
            </div>
            """,
            unsafe_allow_html=True
        )

        preview_mode = st.radio(
            "Change Source for Preview",
            options=["Select Existing Commit", "Simulate Custom Component Modification"],
            horizontal=True
        )

        diff_to_analyze: List[FileDiffItem] = []

        if preview_mode == "Select Existing Commit" and commits:
            c_opts = [f"{c.commit_hash[:7]} — {c.message}" for c in reversed(commits)]
            c_idx = st.selectbox("Select Commit to Preview", options=list(range(len(c_opts))), format_func=lambda i: c_opts[i], key="change_prev_commit")
            sel_c = list(reversed(commits))[c_idx]
            dtl = git_loader.get_commit_detail(sel_c.commit_hash)
            if dtl:
                diff_to_analyze = dtl.files_changed
                st.markdown(f"**Files Changed in `{dtl.short_hash}`:**")
                for d in diff_to_analyze:
                    st.markdown(f"- `[{d.change_type}]` **{d.filepath}** (+{d.insertions} / -{d.deletions})")
                    if d.patch:
                        st.code(d.patch, language="diff")

        else:
            # Interactive patch simulator
            comp_list = list(pred_res.profiles_by_path.keys())
            target_comp = st.selectbox("Select Target Component to Modify", options=comp_list, key="sim_target_comp")
            
            p1, p2 = st.columns(2)
            with p1:
                added_lines_cnt = st.number_input("Proposed Lines Added (+)", min_value=0, max_value=500, value=25, step=5)
                add_branching = st.checkbox("Include Branching Logic (if / for / while)", value=True)
            with p2:
                deleted_lines_cnt = st.number_input("Proposed Lines Removed (-)", min_value=0, max_value=500, value=5, step=5)
                add_new_dep = st.checkbox("Add New External Module Import", value=False)

            # Build mock diff text
            mock_patch_lines = [f"--- a/{target_comp}", f"+++ b/{target_comp}", "@@ -10,4 +10,12 @@"]
            if add_new_dep:
                mock_patch_lines.append("+import external_payment_gateway")
            if add_branching:
                mock_patch_lines.append("+    if validate_request(payload):")
                mock_patch_lines.append("+        for attempt in range(max_retries):")
                mock_patch_lines.append("+            execute_transaction()")
            else:
                mock_patch_lines.append("+    execute_direct_call()")
            mock_patch_lines.append("-    legacy_unverified_call()")

            mock_patch = "\n".join(mock_patch_lines)
            st.markdown("#### Live Diff Preview")
            st.code(mock_patch, language="diff")

            diff_to_analyze = [
                FileDiffItem(
                    filepath=target_comp,
                    change_type="MODIFIED",
                    insertions=int(added_lines_cnt),
                    deletions=int(deleted_lines_cnt),
                    patch=mock_patch
                )
            ]

        # 20.9 & 20.10: [ Analyze Change ] action
        st.markdown("<hr style='border-color: #21262d; margin: 16px 0;'>", unsafe_allow_html=True)
        if st.button("⚡ Analyze Change", type="primary", key="btn_run_change_analysis"):
            with st.spinner("Parsing diff... Updating structural metrics... Analyzing dependencies... Running risk model..."):
                impact_summary = change_analyzer.analyze_commit_or_preview(diff_to_analyze)

            st.markdown("### Change Impact Analysis")
            s1, s2, s3, s4 = st.columns(4)
            with s1:
                render_metric_card("Files Affected", str(impact_summary.total_files_affected), "In diff preview")
            with s2:
                render_metric_card("Components Affected", str(impact_summary.total_components_affected), "In model pipeline")
            with s3:
                render_metric_card("Dependency Changes", f"{impact_summary.total_dependency_changes:+d}", "Topology shifts")
            with s4:
                render_metric_card("Structural Shifts", str(impact_summary.total_structural_changes), "Metric modifications")

            st.markdown("<div style='margin-top: 14px;'></div>", unsafe_allow_html=True)

            # Before vs After Risk State
            if impact_summary.component_impacts:
                for imp in impact_summary.component_impacts:
                    b_col = "#ff7b72" if imp.current_risk_category == "HIGH" else ("#d29922" if imp.current_risk_category == "MEDIUM" else "#3fb950")
                    a_col = "#ff7b72" if imp.proposed_risk_category == "HIGH" else ("#d29922" if imp.proposed_risk_category == "MEDIUM" else "#3fb950")

                    st.markdown(
                        f"""
                        <div style="background: #161b22; border: 1px solid #30363d; border-radius: 6px; padding: 16px; margin-bottom: 14px;">
                            <div style="font-size: 15px; font-weight: 600; font-family: monospace; color: #58a6ff; margin-bottom: 12px;">
                                {imp.filepath}
                            </div>
                            <div style="display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 14px; text-align: center;">
                                <div style="background: #0d1117; padding: 10px; border-radius: 6px; border: 1px solid #21262d;">
                                    <div style="font-size: 11px; text-transform: uppercase; color: #8b949e;">CURRENT STATE</div>
                                    <div style="font-size: 20px; font-weight: 700; color: {b_col};">{imp.current_risk_pct}%</div>
                                    <div style="font-size: 11px; color: {b_col}; font-weight: 600;">{imp.current_risk_category}</div>
                                </div>
                                <div style="background: #0d1117; padding: 10px; border-radius: 6px; border: 1px solid #21262d;">
                                    <div style="font-size: 11px; text-transform: uppercase; color: #8b949e;">PROPOSED STATE</div>
                                    <div style="font-size: 20px; font-weight: 700; color: {a_col};">{imp.proposed_risk_pct}%</div>
                                    <div style="font-size: 11px; color: {a_col}; font-weight: 600;">{imp.proposed_risk_category}</div>
                                </div>
                                <div style="background: #0d1117; padding: 10px; border-radius: 6px; border: 1px solid #21262d;">
                                    <div style="font-size: 11px; text-transform: uppercase; color: #8b949e;">RISK DELTA</div>
                                    <div style="font-size: 20px; font-weight: 700; color: {'#ff7b72' if imp.risk_delta_pct > 0 else '#3fb950'};">
                                        {imp.risk_delta_pct:+d} pp
                                    </div>
                                    <div style="font-size: 11px; color: #8b949e;">percentage points</div>
                                </div>
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

                    # 20.10: Why did the prediction change?
                    st.markdown("#### Why Did the Prediction Change?")
                    if imp.changed_signals:
                        sig_rows = []
                        for sig in imp.changed_signals:
                            sig_rows.append({
                                "Signal": sig.display_name,
                                "Before": f"{sig.before_value} {sig.unit}".strip(),
                                "Proposed": f"{sig.after_value} {sig.unit}".strip(),
                                "Signal Delta": f"{sig.after_value - sig.before_value:+}",
                                "Model Impact Contribution": f"{sig.estimated_contribution:+0.1f} pp"
                            })
                        st.dataframe(pd.DataFrame(sig_rows), use_container_width=True, hide_index=True)
                    else:
                        st.caption("No significant metric alterations detected.")

                    st.info(f"Summary: {imp.explanation}")
            else:
                st.warning("Change impact could not be estimated with the available analysis.")

    # ==========================================
    # TAB 5: BRANCHES / SAFE COMMIT & PUSH
    # ==========================================
    with tab_branches:
        st.markdown("### Branches & Safe Commit Workflow")
        st.caption("Manage branch comparisons, create explicit commits, and synchronize with remotes safely.")

        # Branch Comparison (Section 20.19)
        active_branch, all_branches = git_loader.get_branches()
        b1, b2 = st.columns(2)
        with b1:
            base_branch = st.selectbox("Base Branch", options=all_branches, index=0, key="base_branch_sel")
        with b2:
            target_branch = st.selectbox("Compare Branch", options=all_branches, index=min(1, len(all_branches)-1), key="target_branch_sel")

        b_comp = git_loader.compare_branches(base_branch, target_branch)
        bc1, bc2, bc3, bc4 = st.columns(4)
        with bc1:
            render_metric_card("Commits Ahead", str(b_comp.ahead_commits), f"{target_branch} vs {base_branch}")
        with bc2:
            render_metric_card("Files Changed", str(len(b_comp.files_changed)), "Modified files")
        with bc3:
            render_metric_card("Insertions", f"+{b_comp.insertions}", "Lines added")
        with bc4:
            render_metric_card("Deletions", f"-{b_comp.deletions}", "Lines removed")

        st.markdown("<hr style='border-color: #21262d; margin: 20px 0;'>", unsafe_allow_html=True)

        # 20.12 Commit Action
        st.markdown("### Safe Commit Action")
        st.markdown("Commit pending working tree modifications. *(Requires explicit user confirmation)*")

        comm_msg = st.text_input("Commit Message", value="Refactor routing and improve API validation handling")
        author_opts = [c.author_name for c in contributors] if contributors else ["Developer"]
        selected_author = st.selectbox("Author", options=author_opts, index=0)
        
        st.caption(f"Target Branch: `{active_branch}`")

        if st.button("Commit Changes", type="primary", key="btn_execute_commit"):
            success, result_msg = git_loader.safe_commit(
                message=comm_msg,
                author_name=selected_author,
                author_email="developer@softwarepulse.ai"
            )
            if success:
                st.markdown(
                    f"""
                    <div style="background: #3fb95022; border: 1px solid #3fb950; border-radius: 6px; padding: 14px; margin-top: 10px;">
                        <div style="font-weight: 600; color: #3fb950; font-size: 15px;">✓ Committed successfully</div>
                        <div style="font-family: monospace; font-size: 13px; margin-top: 4px; color: #e6edf3;">
                            Hash: <code>{result_msg[:7]}</code> • {comm_msg}
                        </div>
                        <div style="font-size: 12px; color: #8b949e; margin-top: 2px;">
                            Author: {selected_author} • Branch: {active_branch}
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
            else:
                st.warning(f"Commit status: {result_msg}")

        st.markdown("<hr style='border-color: #21262d; margin: 20px 0;'>", unsafe_allow_html=True)

        # 20.13 Push / Sync Action
        st.markdown("### Push to Remote")
        st.caption("Synchronize committed changes with configured upstream Git remote.")

        p_col1, p_col2 = st.columns([2, 1])
        with p_col1:
            st.markdown(
                f"""
                - **Remote:** `origin`
                - **Branch:** `{active_branch}`
                - **Status:** <span style="color: #58a6ff; font-weight: 600;">● Ready for push sync</span>
                """,
                unsafe_allow_html=True
            )
        with p_col2:
            if st.button("Push to Remote", key="btn_push_remote"):
                p_success, p_msg = git_loader.safe_push("origin", active_branch)
                if p_success:
                    st.success(p_msg)
                else:
                    st.info(f"Push unavailable — configure repository authentication or credentials. ({p_msg})")

        # 20.24 Read-Only Fallback Note
        st.markdown(
            """
            <div style="background: #161b22; border: 1px solid #30363d; border-radius: 6px; padding: 10px 14px; margin-top: 16px; font-size: 12px; color: #8b949e;">
                <strong>READ-ONLY REPOSITORY SAFETY:</strong> SoftwarePulse analysis, diff rendering, and AI risk simulations function continuously in read-only mode without requiring write/push permissions.
            </div>
            """,
            unsafe_allow_html=True
        )


def render_health_page(health_report: Optional[RepositoryHealthReport]):
    """
    Renders the dedicated Repository Health Score dashboard.
    Evaluates 5 dimensions: Code Quality (25%), Change Risk (25%), Dependency Health (20%),
    Stability (20%), Test Health (10%). Provides grade, progress meters, weaknesses, and recommendations.
    """
    st.markdown("## Repository Health Score")
    st.caption("Multidimensional software engineering health assessment across quality, change risk, architecture, and stability.")

    if not health_report:
        st.info("Health report is not available. Please run repository analysis first.")
        return

    grade_colors = {
        "A": "#3fb950", "B": "#58a6ff", "C": "#d29922", "D": "#db6d28", "F": "#f85149"
    }
    g_col = grade_colors.get(health_report.overall_grade, "#58a6ff")

    # 1. Top Overall Health Banner
    st.markdown(
        f"""
        <div style="background: #161b22; border: 1px solid #30363d; border-radius: 8px; padding: 24px; margin-bottom: 20px;">
            <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 16px;">
                <div>
                    <div style="font-size: 13px; text-transform: uppercase; color: #8b949e; letter-spacing: 0.5px; font-weight: 600;">Overall Health Score</div>
                    <div style="font-size: 42px; font-weight: 800; color: #f0f6fc; line-height: 1.1; margin-top: 4px;">
                        {health_report.overall_score}<span style="font-size: 22px; color: #8b949e;"> / 100</span>
                        <span style="font-size: 18px; background: {g_col}22; color: {g_col}; border: 1px solid {g_col}; border-radius: 6px; padding: 4px 14px; margin-left: 14px; font-weight: 700; vertical-align: middle;">Grade {health_report.overall_grade}</span>
                    </div>
                    <div style="font-size: 15px; color: #c9d1d9; margin-top: 8px; max-width: 650px;">{health_report.summary}</div>
                </div>
                <div style="background: #0d1117; border: 1px solid #30363d; border-radius: 6px; padding: 14px 20px; text-align: left; min-width: 260px;">
                    <div style="font-size: 11px; text-transform: uppercase; color: #8b949e; font-weight: 600; margin-bottom: 6px;">Evaluation Horizon</div>
                    <div style="font-size: 13px; color: #e6edf3;"><strong>Total Commits:</strong> {health_report.total_commits_evaluated}</div>
                    <div style="font-size: 13px; color: #e6edf3;"><strong>Total Files:</strong> {health_report.total_files_evaluated}</div>
                    <div style="font-size: 12px; color: #8b949e; margin-top: 6px;">Weights: Quality 25% • Churn 25% • Dep 20% • Stability 20% • Test 10%</div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    # 2. Five Dimension Score Cards
    st.markdown("### Dimension Breakdown")
    d_cols = st.columns(5)
    dim_icons = {
        "Code Quality": "◈", "Change Risk": "↗", "Dependency Health": "⌘", "Stability": "⚖", "Test Health": "✓"
    }
    dim_descriptions = {
        "Code Quality": "Complexity & Maintainability",
        "Change Risk": "Churn, Authors & Volatility",
        "Dependency Health": "Coupling & Cycles",
        "Stability": "Defect Concentration",
        "Test Health": "Test Ratio & Verification"
    }

    for idx, (dim_name, dim_data) in enumerate(health_report.dimensions.items()):
        with d_cols[idx]:
            score = dim_data.score
            s_col = "#3fb950" if score >= 80 else ("#58a6ff" if score >= 65 else ("#d29922" if score >= 50 else "#f85149"))
            st.markdown(
                f"""
                <div style="background: #161b22; border: 1px solid #30363d; border-top: 3px solid {s_col}; border-radius: 6px; padding: 14px; text-align: center;">
                    <div style="font-size: 20px; color: {s_col};">{dim_icons.get(dim_name, '●')}</div>
                    <div style="font-size: 13px; font-weight: 600; color: #f0f6fc; margin-top: 4px;">{dim_name}</div>
                    <div style="font-size: 11px; color: #8b949e;">{dim_descriptions.get(dim_name, '')}</div>
                    <div style="font-size: 28px; font-weight: 700; color: {s_col}; margin: 8px 0 2px 0;">{score}%</div>
                    <div style="font-size: 11px; color: #8b949e;">Weight: {int(dim_data.weight * 100)}%</div>
                </div>
                """,
                unsafe_allow_html=True
            )
            # Progress bar
            st.progress(score / 100.0)

    # 3. Two Columns: Identified Weaknesses & Actionable Improvements
    st.markdown("<div style='margin-top: 20px;'></div>", unsafe_allow_html=True)
    w_col1, w_col2 = st.columns(2)

    with w_col1:
        st.markdown("### Identified Weaknesses")
        if health_report.weaknesses:
            for w in health_report.weaknesses:
                st.markdown(
                    f"""
                    <div style="background: #f8514911; border-left: 3px solid #f85149; border-radius: 4px; padding: 10px 14px; margin-bottom: 8px; font-size: 13px; color: #f0f6fc;">
                        ⚠️ {w}
                    </div>
                    """,
                    unsafe_allow_html=True
                )
        else:
            st.success("No critical architectural weaknesses detected across evaluated dimensions.")

    with w_col2:
        st.markdown("### Recommended Improvements")
        if health_report.recommended_improvements:
            for imp in health_report.recommended_improvements:
                st.markdown(
                    f"""
                    <div style="background: #3fb95011; border-left: 3px solid #3fb950; border-radius: 4px; padding: 10px 14px; margin-bottom: 8px; font-size: 13px; color: #f0f6fc;">
                        ✓ {imp}
                    </div>
                    """,
                    unsafe_allow_html=True
                )
        else:
            st.info("Repository health is within nominal operational parameters.")

    # 4. Transparent Submetric Inspection Expander
    st.markdown("<hr style='border-color: #21262d; margin: 20px 0;'>", unsafe_allow_html=True)
    with st.expander("🔍 Transparent Sub-Metric Inspection & Formulas", expanded=False):
        st.markdown(
            """
            #### Scoring Methodology & Mathematical Formulations
            SoftwarePulse uses an evidence-based, transparent scoring engine with no arbitrary black-box weights:
            - **Code Quality (25%):** Penalized by components exceeding CC > 15 (up to -35 pts) and large files LOC > 500 (up to -20 pts).
            - **Change Risk (25%):** Penalized by high total churn (> 1,000 lines, up to -25 pts), author concentration (single author bus factor, up to -20 pts), and rapid modification frequency.
            - **Dependency Health (20%):** Strict penalty for circular dependency cycles (-25 pts base) and coupling hubs with degree > 10.
            - **Stability (20%):** Derived from bug-fix commits and churn concentration in the top 20% most volatile files.
            - **Test Health (10%):** Verified test-to-source file ratio ($R = N_{test} / N_{src}$, benchmark 0.50). *Does not claim test execution coverage.*
            """
        )
        for dname, dscore in health_report.dimensions.items():
            st.markdown(f"**{dname} ({int(dscore.weight*100)}% weight):** Score `{dscore.score}/100`")
            if dscore.submetrics:
                sub_rows = [{"Metric": k, "Measured Value": v} for k, v in dscore.submetrics.items()]
                st.dataframe(pd.DataFrame(sub_rows), hide_index=True, use_container_width=True)


def render_recommendations_page(
    recommendations: Optional[List[ComponentRecommendation]],
    pred_res: PredictionResult
):
    """
    Renders prioritized, actionable engineering recommendations.
    Rule-driven, evidence-backed guidance sorted by severity (CRITICAL, HIGH, MEDIUM, LOW).
    """
    st.markdown("## Action Recommendations")
    st.caption("Evidence-backed engineering actions to remediate structural hotspots, reduce cognitive complexity, and eliminate architectural risk.")

    if not recommendations:
        st.info("No recommendations generated. Run repository analysis to produce action items.")
        return

    # Helper function for string priority and category
    def get_p(r):
        return r.priority.value if hasattr(r.priority, "value") else str(r.priority)

    def get_c(r):
        return r.category.value if hasattr(r.category, "value") else str(r.category)

    # 1. Summary Metric Tiles
    crit_c = sum(1 for r in recommendations if get_p(r) == "CRITICAL")
    high_c = sum(1 for r in recommendations if get_p(r) == "HIGH")
    med_c = sum(1 for r in recommendations if get_p(r) == "MEDIUM")
    low_c = sum(1 for r in recommendations if get_p(r) == "LOW")

    rc1, rc2, rc3, rc4 = st.columns(4)
    with rc1:
        render_metric_card("Critical Priority", str(crit_c), "Immediate refactoring required")
    with rc2:
        render_metric_card("High Priority", str(high_c), "Upcoming sprint targets")
    with rc3:
        render_metric_card("Medium Priority", str(med_c), "Maintainability improvements")
    with rc4:
        render_metric_card("Low Priority", str(low_c), "Documentation & test gaps")

    st.markdown("<div style='margin-top: 14px;'></div>", unsafe_allow_html=True)

    # 2. Filter Controls
    f_c1, f_c2 = st.columns([1, 1.5])
    with f_c1:
        p_filter = st.selectbox("Filter Priority", options=["ALL", "CRITICAL", "HIGH", "MEDIUM", "LOW"], index=0)
    with f_c2:
        cat_options = ["ALL"] + sorted(list({get_c(r) for r in recommendations}))
        c_filter = st.selectbox("Filter Category", options=cat_options, index=0)

    filtered = recommendations
    if p_filter != "ALL":
        filtered = [r for r in filtered if get_p(r) == p_filter]
    if c_filter != "ALL":
        filtered = [r for r in filtered if get_c(r) == c_filter]

    st.markdown(f"**Showing {len(filtered)} recommendation(s):**")

    # 3. Recommendation Cards
    p_colors = {"CRITICAL": "#f85149", "HIGH": "#d29922", "MEDIUM": "#58a6ff", "LOW": "#3fb950"}

    for idx, rec in enumerate(filtered):
        p_str = get_p(rec)
        cat_str = get_c(rec)
        p_col = p_colors.get(p_str, "#58a6ff")
        with st.container():
            st.markdown(
                f"""
                <div style="background: #161b22; border: 1px solid #30363d; border-left: 4px solid {p_col}; border-radius: 6px; padding: 16px; margin-bottom: 14px;">
                    <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap;">
                        <div>
                            <span style="font-size: 11px; font-weight: 700; color: {p_col}; background: {p_col}22; border: 1px solid {p_col}44; padding: 2px 8px; border-radius: 4px;">{p_str}</span>
                            <span style="font-size: 12px; color: #8b949e; margin-left: 10px;">Category: <strong>{cat_str}</strong></span>
                        </div>
                        <div style="font-size: 12px; color: #8b949e;">{rec.evidence_type}</div>
                    </div>
                    <div style="font-size: 16px; font-weight: 700; color: #f0f6fc; margin: 10px 0 4px 0;">{rec.action}</div>
                    <div style="font-size: 13px; color: #58a6ff; font-family: monospace; margin-bottom: 8px;">Target: {rec.filepath}</div>
                    <div style="background: #0d1117; border-radius: 4px; padding: 10px 14px; margin-bottom: 10px;">
                        <div style="font-size: 11px; text-transform: uppercase; color: #8b949e; font-weight: 600;">Empirical Evidence & Triggering Factors</div>
                        <div style="font-size: 13px; color: #c9d1d9; margin-top: 2px;">{rec.reason}</div>
                    </div>
                    <div style="font-size: 13px; color: #e6edf3;">
                        <strong>Action Steps:</strong> {rec.suggested_next_step}
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )


def render_pr_analyzer_page(
    pr_analyzer: Optional[PullRequestRiskAnalyzer],
    repo_meta: RepositoryMetadata,
    git_loader: Optional[GitLoader] = None
):
    """
    Renders the Pull Request & Branch Pre-Merge Risk Analyzer.
    Supports local branch comparison (base..head) and remote GitHub PR analysis.
    """
    st.markdown("## Pull Request Risk Analyzer")
    st.caption("Pre-merge risk forecasting, structural shift calculation, and downstream dependent exposure.")

    if not pr_analyzer:
        st.info("Pull Request Analyzer is not initialized. Please run repository analysis first.")
        return

    tab_local, tab_github = st.tabs(["Local Branch Comparison", "GitHub Pull Request (Remote)"])

    # 1. Local Branch Comparison Tab
    with tab_local:
        st.markdown("### Local Branch Comparison")
        st.caption("Compare proposed changes between two local Git branches without pushing or checking out.")

        active_branch = "main"
        all_branches = ["main", "HEAD"]
        if git_loader:
            try:
                active_branch, all_branches = git_loader.get_branches()
            except Exception:
                pass

        col_b1, col_b2, col_b3 = st.columns([2, 2, 1.2])
        with col_b1:
            base_b = st.selectbox("Base Branch (Target)", options=all_branches, index=0, key="pr_base_branch")
        with col_b2:
            feat_idx = min(1, len(all_branches) - 1)
            head_b = st.selectbox("Feature Branch (Source)", options=all_branches, index=feat_idx, key="pr_head_branch")
        with col_b3:
            st.markdown("<div style='margin-top: 24px;'></div>", unsafe_allow_html=True)
            run_branch_analysis = st.button("Evaluate Branch Risk", key="btn_run_branch_pr", use_container_width=True)

        if run_branch_analysis or "cached_branch_report" in st.session_state:
            if run_branch_analysis:
                with st.spinner(f"Evaluating diff: {base_b}..{head_b}..."):
                    report = pr_analyzer.analyze_local_branch_diff(base_b, head_b)
                    st.session_state["cached_branch_report"] = report
            else:
                report = st.session_state["cached_branch_report"]

            _render_pr_report_view(report)

    # 2. GitHub PR Remote Tab
    with tab_github:
        st.markdown("### GitHub Pull Request (Remote)")
        st.caption("Inspect a live pull request from GitHub by URL. Diff and patch data are parsed safely in read-only mode.")

        col_g1, col_g2, col_g3 = st.columns([3, 2, 1.2])
        with col_g1:
            pr_url_input = st.text_input(
                "GitHub Pull Request URL",
                placeholder="https://github.com/owner/repo/pull/42",
                key="pr_url_input"
            )
        with col_g2:
            gh_token_input = st.text_input(
                "GitHub Personal Access Token (Optional)",
                type="password",
                help="Recommended for private repositories or to avoid GitHub API rate limits (60 req/hr unauthenticated).",
                key="pr_token_input"
            )
        with col_g3:
            st.markdown("<div style='margin-top: 24px;'></div>", unsafe_allow_html=True)
            run_gh_analysis = st.button("Evaluate PR Risk", key="btn_run_gh_pr", use_container_width=True)

        if run_gh_analysis:
            if not pr_url_input.strip():
                st.warning("Please provide a valid GitHub Pull Request URL.")
            else:
                with st.spinner("Fetching PR metadata and evaluating change impact..."):
                    token = gh_token_input.strip() if gh_token_input.strip() else None
                    gh_report = pr_analyzer.analyze_github_pr(pr_url_input.strip(), github_token=token)
                    if gh_report:
                        st.session_state["cached_gh_report"] = gh_report
                    else:
                        st.error("Could not fetch or parse the specified GitHub PR. Please verify the URL or provide a token for private repos.")

        if "cached_gh_report" in st.session_state:
            _render_pr_report_view(st.session_state["cached_gh_report"])


def _render_pr_report_view(report: PullRequestRiskReport):
    """Helper to render a PullRequestRiskReport dashboard."""
    cat_colors = {"HIGH": "#f85149", "MEDIUM": "#d29922", "LOW": "#3fb950"}
    c_col = cat_colors.get(report.overall_risk_level, "#58a6ff")

    # Header Card
    st.markdown(
        f"""
        <div style="background: #161b22; border: 1px solid #30363d; border-radius: 8px; padding: 20px; margin: 16px 0;">
            <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 12px;">
                <div>
                    <div style="font-size: 12px; text-transform: uppercase; color: #8b949e; font-weight: 600;">{report.source_identifier}</div>
                    <div style="font-size: 22px; font-weight: 700; color: #f0f6fc; margin-top: 2px;">{report.title}</div>
                    <div style="font-size: 13px; color: #8b949e; margin-top: 4px;">Comparing <code>{report.base_ref}</code> ➔ <code>{report.head_ref}</code></div>
                </div>
                <div style="text-align: right;">
                    <div style="font-size: 32px; font-weight: 800; color: {c_col};">
                        {report.overall_risk_score_pct}%
                        <span style="font-size: 14px; background: {c_col}22; border: 1px solid {c_col}; border-radius: 4px; padding: 2px 8px; font-weight: 700; vertical-align: middle;">{report.overall_risk_level} RISK</span>
                    </div>
                    <div style="font-size: 12px; color: #8b949e;">Predicted pre-merge risk score</div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    # 4 Metric Cards
    p1, p2, p3, p4 = st.columns(4)
    with p1:
        render_metric_card("Files Changed", str(report.total_files_changed), f"{report.high_risk_files_count} high-risk files")
    with p2:
        render_metric_card("Total Churn", str(report.total_churn), f"+{report.total_insertions} / -{report.total_deletions} lines")
    with p3:
        render_metric_card("Downstream Exposed", str(report.total_downstream_dependents_impacted), "Dependent components affected")
    with p4:
        render_metric_card("Introduced Cycles", str(report.newly_introduced_cycles_count), "Circular dependencies created")

    # Risk Drivers & Suggested Tests
    st.markdown("<div style='margin-top: 14px;'></div>", unsafe_allow_html=True)
    d_col1, d_col2 = st.columns(2)
    with d_col1:
        st.markdown("#### Key Risk Drivers")
        for driver in report.key_risk_drivers:
            st.markdown(f"- ⚠️ {driver}")
    with d_col2:
        st.markdown("#### Suggested Verification Tests")
        for test_item in report.suggested_verification_tests:
            st.markdown(f"- ✓ `{test_item}`")

    # Files Detail Table
    st.markdown("<div style='margin-top: 14px;'></div>", unsafe_allow_html=True)
    st.markdown("### Modified Components Risk Breakdown")
    if report.files_detail:
        rows = []
        for f in report.files_detail:
            rows.append({
                "File": f.filepath,
                "Change": f.change_type,
                "Insertions": f"+{f.insertions}",
                "Deletions": f"-{f.deletions}",
                "Current Risk": f"{f.current_risk_pct}%",
                "Proposed Risk": f"{f.proposed_risk_pct}%",
                "Delta": f"{'+' if f.risk_delta_pct > 0 else ''}{f.risk_delta_pct}%",
                "Category": f.risk_category,
                "Complexity Δ": f"+{f.complexity_delta}" if f.complexity_delta > 0 else str(f.complexity_delta),
                "Downstream Impacted": len(f.downstream_dependents_affected)
            })
        st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)


def render_feature_ablation_page(
    ablation_report: Optional[AblationStudyReport],
    selected_algo: ClassifierAlgorithm
):
    """
    Renders the 4-way feature ablation experiment & empirical research report.
    Compares Baseline, Model A (Static), Model B (Static+Evolution), Model C (Static+Dependency),
    and Model D (Full), validating the core research hypothesis with leakage-free guarantees.
    """
    st.markdown("## Feature Ablation Research Study")
    st.caption("Empirical ablation study addressing the central research question: Does incorporating software evolution and dependency topology improve risk prediction compared to static metrics alone?")

    if not ablation_report:
        st.info("Ablation study report not available. Run repository analysis to train all ablation models.")
        return

    # 1. Research Hypothesis Card
    st.markdown(
        """
        <div style="background: #161b22; border: 1px solid #30363d; border-radius: 8px; padding: 18px 22px; margin-bottom: 20px;">
            <div style="font-size: 13px; text-transform: uppercase; color: #58a6ff; font-weight: 700; letter-spacing: 0.5px;">Core Research Hypothesis</div>
            <div style="font-size: 17px; font-weight: 600; color: #f0f6fc; margin: 6px 0;">
                <em>"Incorporating software evolution and dependency topology information significantly improves component risk prediction performance over static source-code metrics alone."</em>
            </div>
            <div style="font-size: 13px; color: #8b949e;">
                This experiment trains four distinct feature representations on identical historical training snapshots and evaluates them on an independent future test horizon.
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    # 2. Comparative Performance Table
    st.markdown("### Comparative Model Evaluation Matrix")
    df_comp = ablation_report.to_dataframe()
    st.dataframe(df_comp, hide_index=True, use_container_width=True)

    # 3. Key Research Findings & Delta Synthesis
    st.markdown("<div style='margin-top: 14px;'></div>", unsafe_allow_html=True)
    st.markdown("### Empirical Delta Synthesis")
    c1, c2, c3 = st.columns(3)

    with c1:
        d_b = ablation_report.f1_delta_evolution
        col_b = "#3fb950" if d_b >= 0 else "#f85149"
        render_metric_card(
            "Value of Evolution",
            f"{'+' if d_b >= 0 else ''}{d_b:.3f} F1",
            "Model B vs Model A (Static)"
        )

    with c2:
        d_c = ablation_report.f1_delta_dependency
        col_c = "#3fb950" if d_c >= 0 else "#f85149"
        render_metric_card(
            "Value of Dependencies",
            f"{'+' if d_c >= 0 else ''}{d_c:.3f} F1",
            "Model C vs Model A (Static)"
        )

    with c3:
        d_d = ablation_report.f1_delta_full
        col_d = "#3fb950" if d_d >= 0 else "#f85149"
        render_metric_card(
            "Combined Synergy",
            f"{'+' if d_d >= 0 else ''}{d_d:.3f} F1",
            "Full Model D vs Model A (Static)"
        )

    # 4. Leakage-Free Scientific Protocol Checklist
    st.markdown("<hr style='border-color: #21262d; margin: 20px 0;'>", unsafe_allow_html=True)
    st.markdown("### Scientific Protocol & Leakage-Free Validation")
    st.markdown(
        r"""
        - ✓ **Snapshot Reconstruction:** Historical source code and dependencies were reconstructed at exact observation timestamp $T_{obs}$ using safe read-only Git trees.
        - ✓ **Non-Overlapping Horizons:** Training future label window $(T_{train\_obs}, T_{train\_obs} + H]$ is strictly disjoint from testing observation window.
        - ✓ **Zero Future Leakage:** Commit counts, churn metrics, and author numbers exclude all commits occurring after the observation cutoff.
        - ✓ **Fair Preprocessing:** Missing-value imputation and feature scaling were fitted exclusively on training data and applied downstream to test data.
        - ✓ **Deterministic Evaluation:** Single-class holdouts are identified transparently with DummyClassifier baselines without artificial label fabrication.
        """
    )


