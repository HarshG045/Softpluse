"""
UI Pages: Refined, developer-tool analytical and experimental dashboards for SoftwarePulse.
"""
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
from visualization.dependency_graph import create_interactive_dependency_figure
from visualization.evolution_charts import create_component_evolution_chart, create_snapshot_metric_timeline
from visualization.risk_charts import create_risk_distribution_chart, create_risk_gauge_chart, create_top_risky_components_chart
from .components import render_export_section, render_metric_card, render_risk_pill, render_scientific_disclaimer


def render_overview_page(
    repo_meta: RepositoryMetadata,
    pred_res: PredictionResult,
    comp_report: Optional[ModelComparisonReport] = None
):
    """Renders the refined Repository Intelligence overview dashboard."""
    st.markdown("## Repository Intelligence")
    st.caption("Evolution-aware component risk and architecture analysis.")

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
    """Renders the granular component-level dependency network graph and direct import tracing."""
    st.markdown("## Component Dependency Graph")
    st.caption("Explore granular component-level coupling, incoming/outgoing directed imports, and neighborhood topologies.")

    g = dep_graph.graph
    all_nodes = sorted(list(g.nodes)) if g.number_of_nodes() > 0 else []

    if not all_nodes:
        st.info("No dependency architecture found.")
        return

    # Controls placed clearly above the graph
    g1, g2 = st.columns([1.5, 1])
    with g1:
        selected_focus = st.selectbox(
            "Focus Neighborhood on Component",
            options=["ALL (Repository-wide)"] + all_nodes,
            index=0
        )
    with g2:
        filter_risky_only = st.checkbox("Highlight High-Risk Components", value=True)

    focus_target = None if selected_focus == "ALL (Repository-wide)" else selected_focus

    # Prominent Interactive Network Graph
    fig_graph = create_interactive_dependency_figure(
        dep_graph=dep_graph,
        risk_profiles_map={p.filepath: p.to_dict() for p in pred_res.profiles},
        focus_component=focus_target,
        max_nodes=60
    )
    st.plotly_chart(fig_graph, use_container_width=True)

    # Component-level Import Tracing Table
    st.markdown("### Component In/Out Dependency Matrix")
    struct_map = dep_graph.get_metrics_for_all()
    rows = []
    for node, sm in struct_map.items():
        prof = pred_res.profiles_by_path.get(node)
        rows.append({
            "Component": node,
            "Risk (%)": f"{prof.risk_percentage}%" if prof else "-",
            "Category": prof.risk_category if prof else "-",
            "Outgoing (Dependencies)": sm.out_degree,
            "Incoming (Dependents)": sm.in_degree,
            "Total Degree": sm.degree,
            "Degree Centrality": round(sm.centrality, 4),
            "Direct Dependencies": ", ".join(sm.direct_dependencies) or "None"
        })

    df_struct = pd.DataFrame(rows).sort_values(by="Total Degree", ascending=False)
    st.dataframe(df_struct, use_container_width=True, hide_index=True)



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

    # Dataset Metadata Summary
    split = dataset.split_result
    st.markdown("### Experimental Dataset Split (Temporal / Leakage-Free)")
    d1, d2, d3, d4 = st.columns(4)
    with d1:
        render_metric_card("Total Components", str(dataset.total_components_count), "In repository")
    with d2:
        render_metric_card("Training Samples", str(split.train_samples_count), f"{split.positive_train_count} positive risk")
    with d3:
        render_metric_card("Test Samples", str(split.test_samples_count), f"{split.positive_test_count} positive risk")
    with d4:
        render_metric_card("Target Scheme", "Bugfix Window", dataset.label_summary.label_type.value)

    st.markdown("<div style='margin-top: 16px;'></div>", unsafe_allow_html=True)

    # Core Research Comparison Table
    st.markdown("### Core Research Model Comparison")
    st.dataframe(comp_report.comparison_df, use_container_width=True, hide_index=True)

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
        st.markdown(
            f"""
            - **F1-Score:** `{rep_full.f1:.4f}`
            - **Precision:** `{rep_full.precision:.4f}`
            - **Recall:** `{rep_full.recall:.4f}`
            - **ROC-AUC:** `{rep_full.roc_auc:.4f}`
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

