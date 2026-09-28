"""
UI Pages: Refined, developer-tool analytical and experimental dashboards for SoftwarePulse.
"""
from typing import Dict, List, Optional
import numpy as np
import pandas as pd
import streamlit as st

from analysis.dependency_analyzer import DependencyGraph
from analysis.historical_analyzer import SnapshotRecord
from config import RISK_THRESHOLDS
from dataset.builder import BuiltDataset
from explainability.explainer import ModelExplainer
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
    features_df: pd.DataFrame
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
