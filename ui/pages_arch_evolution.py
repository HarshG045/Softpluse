"""
Architecture Evolution Replay & Change Impact Simulation UI Page.

This module implements the "Architecture Evolution" view as a self-contained
Streamlit page. It reuses existing SoftwarePulse services and visualization
components without modifying them.

The user flow is:
  Select Commit → Reconstruct Snapshot → Analyze Architecture
  → Compare Changes → Inspect Affected Components → Review Risk Differences
"""
import logging
import textwrap
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
import streamlit as st

from analysis.dependency_analyzer import DependencyGraph
from analysis.impact_propagator import AffectedComponent, ImpactPropagator
from analysis.snapshot_engine import (
    CommitSnapshot,
    SnapshotDiff,
    build_snapshot,
    diff_snapshots,
    get_cached_snapshot,
)
from features.feature_schema import FULL_FEATURES, FEATURE_DESCRIPTIONS
from ingestion.git_loader import CommitRecord, GitLoader
from ml.predict import ComponentRiskProfile, PredictionResult, RiskPredictor
from ml.train import TrainedModelBundle
from ui.components import render_metric_card, render_risk_pill, render_scientific_disclaimer
from visualization.dependency_graph import create_interactive_dependency_figure

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Risk colour helpers (reuse existing design tokens from styles.py)
# ─────────────────────────────────────────────────────────────────────────────
_RISK_COLORS = {"HIGH": "#f85149", "MEDIUM": "#d29922", "LOW": "#2ea043"}
_NODE_ADDITION_COLOR = "#3fb950"   # green — new component
_NODE_REMOVAL_COLOR = "#f85149"    # red — removed component
_NODE_CHANGED_COLOR = "#d29922"    # amber — metric-changed component


def _risk_color(category: str) -> str:
    return _RISK_COLORS.get(category.upper(), "#8b949e")


def _delta_color(delta: int) -> str:
    if delta > 0:
        return "#f85149"
    if delta < 0:
        return "#3fb950"
    return "#8b949e"


# ─────────────────────────────────────────────────────────────────────────────
# Session-state key constants
# ─────────────────────────────────────────────────────────────────────────────
_SK_TAB = "ae_active_tab"
_SK_BASELINE = "ae_baseline_hash"
_SK_COMPARISON = "ae_comparison_hash"
_SK_TIMELINE_SEARCH = "ae_timeline_search"
_SK_IMPACT_DEPTH = "ae_impact_depth"
_SK_SNAP_A = "ae_snapshot_a"
_SK_SNAP_B = "ae_snapshot_b"
_SK_DIFF = "ae_diff"
_SK_IMPACT = "ae_impact"


def _init_session_state():
    defaults = {
        _SK_TAB: "Timeline",
        _SK_BASELINE: None,
        _SK_COMPARISON: None,
        _SK_TIMELINE_SEARCH: "",
        _SK_IMPACT_DEPTH: 2,
        _SK_SNAP_A: None,
        _SK_SNAP_B: None,
        _SK_DIFF: None,
        _SK_IMPACT: None,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


# ─────────────────────────────────────────────────────────────────────────────
# Risk computation helper
# ─────────────────────────────────────────────────────────────────────────────
def _try_compute_risk(
    snapshot: CommitSnapshot,
    model_bundle: TrainedModelBundle
) -> Optional[PredictionResult]:
    """
    Attempts to run risk prediction on a snapshot's reconstructed features.
    Returns a PredictionResult if the model's feature schema is compatible,
    otherwise returns None and annotates the snapshot with the limitation.
    """
    if not snapshot.code_map or not snapshot.dep_graph:
        snapshot.risk_unavailable_reason = "Snapshot has no code metrics."
        return None

    required_features = model_bundle.feature_names
    struct_map = snapshot.dep_graph.get_metrics_for_all()

    rows = []
    for fp, code_m in snapshot.code_map.items():
        git_m = snapshot.git_metrics_map.get(fp)
        struct_m = struct_map.get(fp)

        row = {
            "filepath": fp,
            "loc": code_m.loc,
            "functions": code_m.functions,
            "classes": code_m.classes,
            "complexity": code_m.complexity,
            "imports": code_m.imports,
            "methods": code_m.methods,
            "commit_count": git_m.commit_count if git_m else 0,
            "recent_commit_count": git_m.recent_commit_count if git_m else 0,
            "churn": git_m.churn if git_m else 0,
            "recent_churn": git_m.recent_churn if git_m else 0,
            "change_frequency": git_m.change_frequency if git_m else 0.0,
            "number_of_authors": git_m.number_of_authors if git_m else 0,
            "time_since_last_change": git_m.time_since_last_change if git_m else 999.0,
            "dependency_count": struct_m.dependency_count if struct_m else 0,
            "dependent_count": struct_m.dependent_count if struct_m else 0,
            "degree": struct_m.degree if struct_m else 0,
            "centrality": struct_m.centrality if struct_m else 0.0,
            "in_degree": struct_m.in_degree if struct_m else 0,
            "out_degree": struct_m.out_degree if struct_m else 0,
        }
        rows.append(row)

    if not rows:
        snapshot.risk_unavailable_reason = "No feature rows could be assembled."
        return None

    features_df = pd.DataFrame(rows)

    # Check schema compatibility
    missing = [f for f in required_features if f not in features_df.columns]
    if missing:
        snapshot.risk_unavailable_reason = (
            f"Feature schema mismatch: {len(missing)} feature(s) missing from snapshot "
            f"representation ({', '.join(missing[:5])}{'...' if len(missing) > 5 else ''}). "
            "Historical risk is unavailable for this snapshot."
        )
        return None

    try:
        predictor = RiskPredictor()
        pred_result = predictor.predict(model_bundle, features_df)
        snapshot.risk_available = True
        snapshot.risk_unavailable_reason = ""
        return pred_result
    except Exception as e:
        snapshot.risk_unavailable_reason = f"Model inference error: {e}"
        return None


# ─────────────────────────────────────────────────────────────────────────────
# Sub-renders
# ─────────────────────────────────────────────────────────────────────────────
def _render_commit_timeline(
    commits: List[CommitRecord],
    repo_path: str,
    current_pred_res: PredictionResult
):
    """
    Renders the commit timeline browser with search, date filter,
    and prev/next navigation. Returns the currently selected commit hash.
    """
    st.markdown("### Commit Timeline")
    st.caption(
        "Browse historical commits. Select any commit to reconstruct the repository "
        "architecture at that point in time."
    )

    if not commits:
        st.warning(
            "No commit history found. Ensure the analyzed repository has Git history "
            "and that GitPython or the Git CLI is available."
        )
        return None

    # ── Controls ────────────────────────────────────────────────────────────
    ctl1, ctl2, ctl3 = st.columns([2.0, 1.2, 1.2])
    with ctl1:
        search_query = st.text_input(
            "Search commits",
            value=st.session_state.get(_SK_TIMELINE_SEARCH, ""),
            placeholder="message, author, hash…",
            key="ae_tl_search_input"
        )
        st.session_state[_SK_TIMELINE_SEARCH] = search_query
    with ctl2:
        date_from = st.date_input("From date", value=None, key="ae_tl_date_from")
    with ctl3:
        date_to = st.date_input("To date", value=None, key="ae_tl_date_to")

    # ── Filter commits ───────────────────────────────────────────────────────
    # commits: oldest → newest; display newest first
    display_commits = list(reversed(commits))

    if search_query.strip():
        q = search_query.strip().lower()
        display_commits = [
            c for c in display_commits
            if q in c.message.lower()
            or q in c.author_name.lower()
            or q in c.commit_hash.lower()
        ]

    if date_from:
        from_dt = datetime(date_from.year, date_from.month, date_from.day, tzinfo=timezone.utc)
        display_commits = [c for c in display_commits if c.timestamp >= from_dt]
    if date_to:
        to_dt = datetime(date_to.year, date_to.month, date_to.day, 23, 59, 59, tzinfo=timezone.utc)
        display_commits = [c for c in display_commits if c.timestamp <= to_dt]

    st.caption(f"Showing **{len(display_commits)}** of **{len(commits)}** commits.")

    if not display_commits:
        st.info("No commits match the current filter.")
        return st.session_state.get(_SK_BASELINE)

    # ── Prev / Next navigation ───────────────────────────────────────────────
    current_hash = st.session_state.get(_SK_BASELINE)
    current_idx = next(
        (i for i, c in enumerate(display_commits) if c.commit_hash == current_hash), 0
    )

    nav_col1, nav_col2, nav_col3 = st.columns([1, 2, 1])
    with nav_col1:
        if st.button("◀ Previous", key="ae_tl_prev", use_container_width=True):
            current_idx = max(0, current_idx - 1)
            st.session_state[_SK_BASELINE] = display_commits[current_idx].commit_hash
            st.session_state[_SK_SNAP_A] = None
            st.rerun()
    with nav_col2:
        if display_commits:
            st.markdown(
                f"<div style='text-align:center; font-family:monospace; color:#8b949e; font-size:12px;'>"
                f"Showing commit {current_idx + 1} of {len(display_commits)}"
                f"</div>",
                unsafe_allow_html=True
            )
    with nav_col3:
        if st.button("Next ▶", key="ae_tl_next", use_container_width=True):
            current_idx = min(len(display_commits) - 1, current_idx + 1)
            st.session_state[_SK_BASELINE] = display_commits[current_idx].commit_hash
            st.session_state[_SK_SNAP_A] = None
            st.rerun()

    st.markdown("<div style='margin-top: 10px;'></div>", unsafe_allow_html=True)

    # ── Commit List ──────────────────────────────────────────────────────────
    for i, commit in enumerate(display_commits[:50]):  # Cap display at 50
        is_selected = (commit.commit_hash == st.session_state.get(_SK_BASELINE))
        is_comparison = (commit.commit_hash == st.session_state.get(_SK_COMPARISON))

        border_color = "#58a6ff" if is_selected else ("#a371f7" if is_comparison else "#30363d")
        bg_color = "rgba(56,139,253,0.07)" if is_selected else (
            "rgba(163,113,247,0.07)" if is_comparison else "#161b22"
        )
        indicator = " ◀ BASELINE" if is_selected else (" ◀ COMPARISON" if is_comparison else "")
        indicator_color = "#58a6ff" if is_selected else "#a371f7"

        date_str = commit.timestamp.strftime("%Y-%m-%d %H:%M")
        short_hash = commit.commit_hash[:7]
        msg = commit.message[:72] + ("…" if len(commit.message) > 72 else "")
        files_changed = len(commit.files_changed)
        bugfix_badge = (
            "<span style='background:#ff7b7222;color:#ff7b72;padding:1px 5px;"
            "border-radius:3px;font-size:10px;margin-left:6px'>BUGFIX</span>"
            if commit.is_bugfix else ""
        )

        st.markdown(
            f"""
            <div style="background:{bg_color};border:1px solid {border_color};
                        border-radius:6px;padding:10px 14px;margin-bottom:6px;">
              <div style="display:flex;justify-content:space-between;align-items:center;">
                <div>
                  <span style="color:#58a6ff;font-family:monospace;font-size:12px;">
                    {short_hash}</span>
                  <span style="color:#8b949e;font-size:11px;margin-left:8px;">{date_str}</span>
                  {bugfix_badge}
                  {f'<span style="color:{indicator_color};font-size:10px;margin-left:8px;">{indicator}</span>' if indicator else ''}
                </div>
                <div style="font-size:11px;color:#8b949e;">
                  <span style="color:#3fb950">+{commit.insertions}</span>
                  &nbsp;/&nbsp;
                  <span style="color:#f85149">-{commit.deletions}</span>
                  &nbsp;• {files_changed} file(s)
                </div>
              </div>
              <div style="margin-top:4px;font-size:13px;color:#c9d1d9;">{msg}</div>
              <div style="font-size:11px;color:#8b949e;margin-top:3px;">
                Author: <b>{commit.author_name}</b>
              </div>
            </div>
            """,
            unsafe_allow_html=True
        )

        # Per-commit action buttons
        btn_col1, btn_col2, btn_col3 = st.columns([1, 1, 1])
        with btn_col1:
            if st.button(
                "Set as Baseline",
                key=f"ae_set_base_{commit.commit_hash}",
                use_container_width=True,
                type="primary" if is_selected else "secondary"
            ):
                st.session_state[_SK_BASELINE] = commit.commit_hash
                st.session_state[_SK_SNAP_A] = None
                st.session_state[_SK_DIFF] = None
                st.session_state[_SK_IMPACT] = None
                st.rerun()
        with btn_col2:
            if st.button(
                "Set as Comparison",
                key=f"ae_set_comp_{commit.commit_hash}",
                use_container_width=True,
                type="primary" if is_comparison else "secondary"
            ):
                st.session_state[_SK_COMPARISON] = commit.commit_hash
                st.session_state[_SK_SNAP_B] = None
                st.session_state[_SK_DIFF] = None
                st.session_state[_SK_IMPACT] = None
                st.rerun()
        with btn_col3:
            if st.button(
                "Inspect Snapshot",
                key=f"ae_inspect_{commit.commit_hash}",
                use_container_width=True
            ):
                st.session_state[_SK_BASELINE] = commit.commit_hash
                st.session_state[_SK_SNAP_A] = None
                st.session_state[_SK_TAB] = "Snapshot"
                st.rerun()

    return st.session_state.get(_SK_BASELINE)


def _render_snapshot_view(
    snapshot: CommitSnapshot,
    pred_result: Optional[PredictionResult],
    label: str = "Snapshot"
):
    """Renders a single historical snapshot's graph, metrics and risk view."""
    if snapshot is None:
        st.info("No snapshot selected. Choose a commit from the Timeline tab.")
        return

    # Header
    st.markdown(
        f"""
        <div style="background:#161b22;border:1px solid #30363d;border-radius:6px;
                    padding:12px 16px;margin-bottom:16px;">
          <div style="font-size:13px;color:#8b949e;margin-bottom:2px;">{label}</div>
          <div style="font-size:16px;font-weight:700;color:#f0f6fc;">
            <code style="color:#58a6ff">{snapshot.short_hash}</code>
            &nbsp;— {snapshot.message[:60]}{"…" if len(snapshot.message) > 60 else ""}
          </div>
          <div style="font-size:11px;color:#8b949e;margin-top:4px;">
            {snapshot.author_name} · {snapshot.timestamp.strftime("%Y-%m-%d %H:%M UTC")}
            · {snapshot.files_available} Python files
            {f"· <span style='color:#f85149'>{snapshot.files_parse_errors} parse errors</span>"
              if snapshot.files_parse_errors > 0 else ""}
          </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    # Risk availability notice
    if not snapshot.risk_available:
        st.info(
            f"**Risk estimation unavailable for this snapshot:** "
            f"{snapshot.risk_unavailable_reason} "
            "Structural metrics and dependency graph are still shown below.",
            icon="ℹ️"
        )
    else:
        render_scientific_disclaimer(
            "Historical risk scores are produced by running the existing trained model "
            "against reconstructed features. The model was not retrained on historical data. "
            "Scores reflect model estimates, not proven defect probabilities."
        )

    # Metric cards
    dep_graph = snapshot.dep_graph
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        render_metric_card("Python Files", str(snapshot.files_available), "At this commit")
    with m2:
        edges = dep_graph.graph.number_of_edges() if dep_graph else 0
        render_metric_card("Dependency Edges", str(edges), "Intra-repo imports")
    with m3:
        if pred_result:
            render_metric_card("High-Risk Cmpts", str(pred_result.high_risk_count), "Model estimate")
        else:
            render_metric_card("High-Risk Cmpts", "N/A", "Risk unavailable")
    with m4:
        total_churn = sum(m.churn for m in snapshot.git_metrics_map.values()) if snapshot.git_metrics_map else 0
        render_metric_card("Total Churn", str(total_churn), "Lines up to this commit")

    st.markdown("<div style='margin-top:14px;'></div>", unsafe_allow_html=True)

    # Dependency graph
    if dep_graph and dep_graph.graph.number_of_nodes() > 0:
        st.markdown("#### Dependency Graph at this Snapshot")
        risk_map: Dict = {}
        if pred_result:
            risk_map = {p.filepath: p.to_dict() for p in pred_result.profiles}

        fig = create_interactive_dependency_figure(
            dep_graph=dep_graph,
            risk_profiles_map=risk_map,
            focus_component=None,
            max_nodes=50,
            layout_type="hierarchical",
            depth=1,
            direction_filter="both",
            risk_filter="all",
            show_labels="truncated",
            highlight_risk=bool(risk_map)
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Dependency graph could not be reconstructed for this snapshot.")

    # Component table
    st.markdown("#### Components at this Snapshot")
    rows = []
    for fp, code_m in snapshot.code_map.items():
        git_m = snapshot.git_metrics_map.get(fp)
        risk_pct = "-"
        risk_cat = "-"
        if pred_result:
            prof = pred_result.profiles_by_path.get(fp)
            if prof:
                risk_pct = str(prof.risk_percentage)
                risk_cat = prof.risk_category

        rows.append({
            "Component": fp,
            "LOC": code_m.loc,
            "Complexity": code_m.complexity,
            "Functions": code_m.functions,
            "Imports": code_m.imports,
            "Commits (up to)": git_m.commit_count if git_m else 0,
            "Churn (up to)": git_m.churn if git_m else 0,
            "Risk (%)": risk_pct,
            "Risk Category": risk_cat,
        })

    if rows:
        df = pd.DataFrame(rows)
        st.dataframe(df, use_container_width=True, hide_index=True)


def _render_evolution_comparison(
    snap_a: CommitSnapshot,
    snap_b: CommitSnapshot,
    diff: SnapshotDiff,
    pred_a: Optional[PredictionResult],
    pred_b: Optional[PredictionResult],
    model_bundle: TrainedModelBundle,
    impact: List[AffectedComponent]
):
    """Renders the side-by-side Change Impact Simulation panel."""
    st.markdown("### Change Impact Simulation")
    render_scientific_disclaimer(
        "This comparison shows observed metric and structural differences between two repository "
        "snapshots. Risk score differences are model estimates from the same trained model — "
        "they represent the model's output given different historical feature representations, "
        "not proof of causal impact."
    )

    # ── Summary header cards ─────────────────────────────────────────────────
    col_base, col_vs, col_comp = st.columns([3, 1, 3])
    with col_base:
        st.markdown(
            f"""
            <div style="background:#161b22;border:1px solid #30363d;border-radius:6px;
                        padding:12px 16px;text-align:center;">
              <div style="font-size:11px;color:#8b949e;">BASELINE</div>
              <div style="font-size:18px;font-weight:700;color:#f0f6fc;font-family:monospace;">
                {diff.baseline_hash}</div>
              <div style="font-size:11px;color:#8b949e;">
                {diff.baseline_timestamp.strftime("%Y-%m-%d %H:%M")}</div>
            </div>
            """,
            unsafe_allow_html=True
        )
    with col_vs:
        st.markdown(
            "<div style='text-align:center;padding-top:20px;font-size:20px;color:#8b949e;'>→</div>",
            unsafe_allow_html=True
        )
    with col_comp:
        st.markdown(
            f"""
            <div style="background:#161b22;border:1px solid #a371f7;border-radius:6px;
                        padding:12px 16px;text-align:center;">
              <div style="font-size:11px;color:#a371f7;">COMPARISON</div>
              <div style="font-size:18px;font-weight:700;color:#f0f6fc;font-family:monospace;">
                {diff.comparison_hash}</div>
              <div style="font-size:11px;color:#8b949e;">
                {diff.comparison_timestamp.strftime("%Y-%m-%d %H:%M")}</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    st.markdown("<div style='margin-top:16px;'></div>", unsafe_allow_html=True)

    # ── File-level change summary ────────────────────────────────────────────
    st.markdown("#### File Changes  *(observed)*")
    fc1, fc2, fc3, fc4 = st.columns(4)
    with fc1:
        render_metric_card("Added Files", str(len(diff.added_files)), "New components")
    with fc2:
        render_metric_card("Removed Files", str(len(diff.removed_files)), "Deleted components")
    with fc3:
        render_metric_card("Modified Files", str(len(diff.modified_files)), "Metric-changed")
    with fc4:
        render_metric_card("Dep Edge Δ",
                           f"+{len(diff.added_edges)} / -{len(diff.removed_edges)}",
                           "Dependency graph topology")

    if diff.added_files or diff.removed_files or diff.modified_files:
        with st.expander("View file-level change details", expanded=False):
            if diff.added_files:
                st.markdown("**Added files:**")
                for f in diff.added_files:
                    st.markdown(
                        f"<span style='color:#3fb950;font-family:monospace;font-size:12px;'>"
                        f"＋ {f}</span>", unsafe_allow_html=True)
            if diff.removed_files:
                st.markdown("**Removed files:**")
                for f in diff.removed_files:
                    st.markdown(
                        f"<span style='color:#f85149;font-family:monospace;font-size:12px;'>"
                        f"－ {f}</span>", unsafe_allow_html=True)
            if diff.modified_files:
                st.markdown("**Modified files (metric changes):**")
                for f in diff.modified_files:
                    st.markdown(
                        f"<span style='color:#d29922;font-family:monospace;font-size:12px;'>"
                        f"Δ {f}</span>", unsafe_allow_html=True)

    st.markdown("<hr style='border-color:#21262d;margin:16px 0;'>", unsafe_allow_html=True)

    # ── Structural metric diffs ──────────────────────────────────────────────
    if diff.metric_deltas:
        st.markdown("#### Structural Metric Changes  *(observed)*")
        delta_rows = []
        for fp, deltas in sorted(diff.metric_deltas.items()):
            base_code = snap_a.code_map.get(fp)
            comp_code = snap_b.code_map.get(fp)
            for metric, delta in sorted(deltas.items()):
                base_val = getattr(base_code, metric, 0) if base_code else 0
                comp_val = getattr(comp_code, metric, 0) if comp_code else 0
                delta_rows.append({
                    "Component": Path(fp).name,
                    "Path": fp,
                    "Metric": metric,
                    "Baseline": base_val,
                    "Comparison": comp_val,
                    "Delta": f"{delta:+.0f}",
                    "Description": FEATURE_DESCRIPTIONS.get(metric, "")
                })
        if delta_rows:
            st.dataframe(pd.DataFrame(delta_rows), use_container_width=True, hide_index=True)

    # ── Risk comparison ──────────────────────────────────────────────────────
    st.markdown("<hr style='border-color:#21262d;margin:16px 0;'>", unsafe_allow_html=True)
    st.markdown("#### Risk Score Comparison  *(model estimates)*")

    if pred_a is None and pred_b is None:
        st.warning(
            "Risk scores are unavailable for both snapshots. "
            "This can happen when the model was trained on a different feature schema "
            "or when the historical snapshot lacks sufficient data."
        )
    else:
        risk_rows = []
        # Build union of all component paths
        all_paths = set()
        if pred_a:
            all_paths.update(pred_a.profiles_by_path.keys())
        if pred_b:
            all_paths.update(pred_b.profiles_by_path.keys())

        for fp in sorted(all_paths):
            prof_a = pred_a.profiles_by_path.get(fp) if pred_a else None
            prof_b = pred_b.profiles_by_path.get(fp) if pred_b else None
            risk_a = prof_a.risk_percentage if prof_a else None
            risk_b = prof_b.risk_percentage if prof_b else None
            cat_a = prof_a.risk_category if prof_a else "N/A"
            cat_b = prof_b.risk_category if prof_b else "N/A"
            delta = (risk_b - risk_a) if (risk_a is not None and risk_b is not None) else None
            status = (
                "Added" if prof_a is None else
                "Removed" if prof_b is None else
                "Changed" if delta != 0 else "Unchanged"
            )
            risk_rows.append({
                "Component": Path(fp).name,
                "Path": fp,
                "Baseline Risk (%)": risk_a if risk_a is not None else "N/A",
                "Baseline Category": cat_a,
                "Comparison Risk (%)": risk_b if risk_b is not None else "N/A",
                "Comparison Category": cat_b,
                "Risk Δ (pp)": f"{delta:+d}" if delta is not None else "N/A",
                "Status": status
            })
            if risk_a is not None and risk_b is not None and delta is not None:
                diff.risk_deltas[fp] = (risk_a, risk_b, delta)

        if risk_rows:
            st.dataframe(pd.DataFrame(risk_rows), use_container_width=True, hide_index=True)

        # Aggregate risk change
        if diff.risk_deltas:
            deltas = [v[2] for v in diff.risk_deltas.values()]
            avg_delta = sum(deltas) / len(deltas)
            delta_color = _delta_color(int(avg_delta))
            max_increase = max(deltas)
            max_decrease = min(deltas)

            agg1, agg2, agg3 = st.columns(3)
            with agg1:
                render_metric_card(
                    "Avg Risk Δ",
                    f"{avg_delta:+.1f}pp",
                    "Percentage points across compared components"
                )
            with agg2:
                render_metric_card(
                    "Max Risk Increase",
                    f"+{max_increase}pp" if max_increase > 0 else "None",
                    "Largest upward shift"
                )
            with agg3:
                render_metric_card(
                    "Max Risk Decrease",
                    f"{max_decrease}pp" if max_decrease < 0 else "None",
                    "Largest downward shift"
                )

    # ── Affected component analysis ──────────────────────────────────────────
    st.markdown("<hr style='border-color:#21262d;margin:16px 0;'>", unsafe_allow_html=True)
    st.markdown("#### Affected Components  *(structural dependency analysis)*")
    st.caption(
        "Starting from modified files, components are traced through the dependency graph. "
        "Inclusion indicates a structural dependency relationship — not a proven defect risk."
    )

    if not impact:
        st.info(
            "No affected components identified. "
            "This may occur when modified files have no known dependents in the graph."
        )
    else:
        impact_rows = []
        for ac in impact:
            pred_res_b = pred_b
            risk_pct = "-"
            risk_cat = "-"
            if pred_res_b:
                prof = pred_res_b.profiles_by_path.get(ac.filepath)
                if prof:
                    risk_pct = str(prof.risk_percentage)
                    risk_cat = prof.risk_category

            impact_rows.append({
                "Component": ac.component_name,
                "Path": ac.filepath,
                "Inclusion Reason": ac.reason,
                "Hop Distance": ac.hop_distance,
                "Risk (%) — Comparison": risk_pct,
                "Risk Category": risk_cat,
                "Dependency Chain": " → ".join([Path(p).name for p in ac.dependency_chain])
            })

        df_impact = pd.DataFrame(impact_rows)
        st.dataframe(df_impact, use_container_width=True, hide_index=True)

        imp_summary_rows = [
            ("Directly Modified", len([a for a in impact if a.hop_distance == 0])),
            ("Direct Dependents (Hop 1)", len([a for a in impact if a.hop_distance == 1])),
            ("Transitive Dependents", len([a for a in impact if a.hop_distance > 1])),
        ]
        col_i1, col_i2, col_i3 = st.columns(3)
        for col, (label, val) in zip([col_i1, col_i2, col_i3], imp_summary_rows):
            with col:
                render_metric_card(label, str(val), "Components in impact set")


def _render_graph_evolution(
    snap_a: CommitSnapshot,
    snap_b: CommitSnapshot,
    diff: SnapshotDiff,
    pred_a: Optional[PredictionResult],
    pred_b: Optional[PredictionResult]
):
    """Renders the graph-evolution comparison panel with added/removed/changed highlights."""
    st.markdown("### Architecture Graph Evolution")
    st.caption(
        "Compare the dependency graph structure between two commits. "
        "Node colours indicate structural role in the change."
    )

    # Change-type overlays
    added_set = set(diff.added_files)
    removed_set = set(diff.removed_files)
    modified_set = set(diff.modified_files)

    col_base_g, col_comp_g = st.columns(2)

    for col, label, snapshot, pred_result, highlight_new, highlight_old in [
        (col_base_g, f"Baseline: `{diff.baseline_hash}`", snap_a, pred_a, False, True),
        (col_comp_g, f"Comparison: `{diff.comparison_hash}`", snap_b, pred_b, True, False)
    ]:
        with col:
            st.markdown(f"#### {label}")
            dep_graph = snapshot.dep_graph if snapshot else None
            if dep_graph and dep_graph.graph.number_of_nodes() > 0:
                risk_map: Dict = {}
                if pred_result:
                    risk_map = {p.filepath: p.to_dict() for p in pred_result.profiles}

                fig = create_interactive_dependency_figure(
                    dep_graph=dep_graph,
                    risk_profiles_map=risk_map,
                    focus_component=None,
                    max_nodes=40,
                    layout_type="hierarchical",
                    depth=1,
                    direction_filter="both",
                    risk_filter="all",
                    show_labels="truncated",
                    highlight_risk=bool(risk_map)
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Graph unavailable for this snapshot.")

    # Change legend
    st.markdown(
        """
        <div style="display:flex;gap:16px;font-size:12px;color:#8b949e;
                    background:#0d1117;border:1px solid #21262d;border-radius:6px;
                    padding:8px 14px;margin:8px 0;">
          <span style="font-weight:600;color:#f0f6fc;">Graph Change Legend:</span>
          <span><span style="color:#3fb950;font-weight:600;">＋</span> Added components</span>
          <span><span style="color:#f85149;font-weight:600;">－</span> Removed components</span>
          <span><span style="color:#d29922;font-weight:600;">Δ</span> Metric-changed components</span>
          <span><span style="color:#58a6ff;font-weight:600;">—</span> Unchanged</span>
        </div>
        """,
        unsafe_allow_html=True
    )

    # Summary table of added/removed edges
    if diff.added_edges or diff.removed_edges:
        st.markdown("#### Dependency Edge Changes")
        edge_rows = []
        for src, tgt in diff.added_edges[:20]:
            edge_rows.append({"Type": "ADDED ＋", "From": src, "To": tgt})
        for src, tgt in diff.removed_edges[:20]:
            edge_rows.append({"Type": "REMOVED －", "From": src, "To": tgt})
        if edge_rows:
            st.dataframe(pd.DataFrame(edge_rows), use_container_width=True, hide_index=True)


# ─────────────────────────────────────────────────────────────────────────────
# Main page entry-point
# ─────────────────────────────────────────────────────────────────────────────
def render_arch_evolution_page(
    git_loader: GitLoader,
    commits: List[CommitRecord],
    model_bundle: TrainedModelBundle,
    current_pred_res: PredictionResult,
    current_dep_graph: DependencyGraph
):
    """
    Main entry-point for the Architecture Evolution Replay & Change Impact Simulation page.

    Args:
        git_loader:        Existing GitLoader for the analyzed repository.
        commits:           Full commit history (oldest → newest), already extracted.
        model_bundle:      Trained model bundle for risk inference.
        current_pred_res:  Current (HEAD) prediction result for reference.
        current_dep_graph: Current (HEAD) dependency graph for reference.
    """
    _init_session_state()

    st.markdown("## Architecture Evolution Replay")
    st.caption(
        "Explore how the software architecture evolved across commits. "
        "Select any two historical snapshots to compare structure, metrics, and model risk estimates."
    )

    # ── Repo availability check ──────────────────────────────────────────────
    if not git_loader.is_git_repo():
        st.error(
            "The analyzed repository does not appear to be a valid Git repository. "
            "Architecture Evolution Replay requires Git history. "
            "Please ensure the repository has been properly initialized with `git init` and has commits."
        )
        return

    if not commits:
        st.warning(
            "No commit history found in this repository. "
            "Architecture Evolution requires at least one commit. "
            "Run **⚡ Run Repository Analysis** to refresh the commit log."
        )
        return

    repo_path = str(git_loader.repo_path)

    # ── Quick-access status bar ──────────────────────────────────────────────
    baseline_hash = st.session_state.get(_SK_BASELINE)
    comparison_hash = st.session_state.get(_SK_COMPARISON)

    status_parts = []
    if baseline_hash:
        status_parts.append(
            f"<span style='color:#8b949e;'>Baseline:</span> "
            f"<code style='color:#58a6ff;'>{baseline_hash[:7]}</code>"
        )
    else:
        status_parts.append("<span style='color:#8b949e;'>Baseline: <i>not selected</i></span>")
    if comparison_hash:
        status_parts.append(
            f"<span style='color:#8b949e;'>Comparison:</span> "
            f"<code style='color:#a371f7;'>{comparison_hash[:7]}</code>"
        )
    else:
        status_parts.append("<span style='color:#8b949e;'>Comparison: <i>not selected</i></span>")

    st.markdown(
        f"<div style='background:#161b22;border:1px solid #30363d;border-radius:6px;"
        f"padding:8px 14px;font-size:12px;margin-bottom:14px;display:flex;gap:20px;'>"
        f"{'&nbsp;&nbsp;•&nbsp;&nbsp;'.join(status_parts)}</div>",
        unsafe_allow_html=True
    )

    # ── Tab navigation ───────────────────────────────────────────────────────
    tab_labels = ["Timeline", "Snapshot", "Graph Evolution", "Change Impact"]
    tabs = st.tabs(tab_labels)

    # ── TAB 1: Commit Timeline ───────────────────────────────────────────────
    with tabs[0]:
        _render_commit_timeline(commits, repo_path, current_pred_res)

    # ── TAB 2: Historical Snapshot Inspector ─────────────────────────────────
    with tabs[1]:
        st.markdown("### Historical Snapshot Inspector")
        st.caption(
            "Reconstructs the repository state at the selected baseline commit using "
            "`git show` (read-only). Your working tree is never modified."
        )

        if not baseline_hash:
            st.info("Select a commit as **Baseline** from the Timeline tab to inspect its snapshot.")
        else:
            snap_a: Optional[CommitSnapshot] = st.session_state.get(_SK_SNAP_A)

            if snap_a is None or snap_a.commit_hash[:7] != baseline_hash[:7]:
                with st.spinner(f"Reconstructing snapshot for `{baseline_hash[:7]}`…"):
                    snap_a = build_snapshot(
                        repo_path=repo_path,
                        commit_hash=baseline_hash,
                        all_commits=commits
                    )
                if snap_a is None:
                    st.error(
                        f"Failed to reconstruct snapshot for commit `{baseline_hash[:7]}`. "
                        "The commit may be invalid or the repository history may be incomplete."
                    )
                else:
                    st.session_state[_SK_SNAP_A] = snap_a

            snap_a = st.session_state.get(_SK_SNAP_A)
            if snap_a:
                # Try risk inference
                pred_a = _try_compute_risk(snap_a, model_bundle)
                _render_snapshot_view(snap_a, pred_a, label="Baseline Snapshot")

    # ── TAB 3: Graph Evolution ───────────────────────────────────────────────
    with tabs[2]:
        if not baseline_hash or not comparison_hash:
            st.info(
                "Select both a **Baseline** and a **Comparison** commit from the Timeline tab "
                "to view the graph evolution between them."
            )
        elif baseline_hash == comparison_hash:
            st.warning("Baseline and Comparison commits are the same. Please select different commits.")
        else:
            # Build both snapshots
            snap_a = st.session_state.get(_SK_SNAP_A)
            snap_b = st.session_state.get(_SK_SNAP_B)

            need_rebuild = False
            if snap_a is None or snap_a.commit_hash[:7] != baseline_hash[:7]:
                need_rebuild = True
            if snap_b is None or snap_b.commit_hash[:7] != comparison_hash[:7]:
                need_rebuild = True

            if need_rebuild:
                with st.spinner("Reconstructing both snapshots…"):
                    snap_a = build_snapshot(repo_path, baseline_hash, commits)
                    snap_b = build_snapshot(repo_path, comparison_hash, commits)
                    st.session_state[_SK_SNAP_A] = snap_a
                    st.session_state[_SK_SNAP_B] = snap_b
                    st.session_state[_SK_DIFF] = None

            snap_a = st.session_state.get(_SK_SNAP_A)
            snap_b = st.session_state.get(_SK_SNAP_B)

            if snap_a and snap_b:
                diff = st.session_state.get(_SK_DIFF)
                if diff is None:
                    diff = diff_snapshots(snap_a, snap_b)
                    st.session_state[_SK_DIFF] = diff

                pred_a = _try_compute_risk(snap_a, model_bundle)
                pred_b = _try_compute_risk(snap_b, model_bundle)
                _render_graph_evolution(snap_a, snap_b, diff, pred_a, pred_b)
            else:
                st.error("Failed to reconstruct one or both snapshots. Check commit validity.")

    # ── TAB 4: Change Impact Simulation ─────────────────────────────────────
    with tabs[3]:
        if not baseline_hash or not comparison_hash:
            st.info(
                "Select both a **Baseline** and a **Comparison** commit from the Timeline tab "
                "to run the Change Impact Simulation."
            )
        elif baseline_hash == comparison_hash:
            st.warning("Baseline and Comparison commits are the same. Please select different commits.")
        else:
            # Impact depth selector
            impact_depth = st.slider(
                "Transitive dependency traversal depth",
                min_value=1,
                max_value=4,
                value=st.session_state.get(_SK_IMPACT_DEPTH, 2),
                help="1 = direct dependents only; higher values include transitive dependents.",
                key="ae_impact_depth_slider"
            )
            st.session_state[_SK_IMPACT_DEPTH] = impact_depth

            snap_a = st.session_state.get(_SK_SNAP_A)
            snap_b = st.session_state.get(_SK_SNAP_B)

            need_rebuild = False
            if snap_a is None or snap_a.commit_hash[:7] != baseline_hash[:7]:
                need_rebuild = True
            if snap_b is None or snap_b.commit_hash[:7] != comparison_hash[:7]:
                need_rebuild = True

            if need_rebuild or st.button("▶ Run Change Impact Analysis", type="primary"):
                with st.spinner("Reconstructing snapshots & computing impact…"):
                    snap_a = build_snapshot(repo_path, baseline_hash, commits)
                    snap_b = build_snapshot(repo_path, comparison_hash, commits)
                    st.session_state[_SK_SNAP_A] = snap_a
                    st.session_state[_SK_SNAP_B] = snap_b
                    st.session_state[_SK_DIFF] = None
                    st.session_state[_SK_IMPACT] = None

            snap_a = st.session_state.get(_SK_SNAP_A)
            snap_b = st.session_state.get(_SK_SNAP_B)

            if snap_a and snap_b:
                diff = st.session_state.get(_SK_DIFF)
                if diff is None:
                    diff = diff_snapshots(snap_a, snap_b)
                    st.session_state[_SK_DIFF] = diff

                # Compute affected components
                cached_impact = st.session_state.get(_SK_IMPACT)
                if cached_impact is None:
                    changed_files = diff.added_files + diff.modified_files + diff.removed_files
                    propagator = ImpactPropagator(
                        dep_graph=snap_b.dep_graph if snap_b.dep_graph else current_dep_graph,
                        max_depth=impact_depth
                    )
                    impact = propagator.find_affected(changed_files)
                    st.session_state[_SK_IMPACT] = impact
                else:
                    impact = cached_impact

                pred_a = _try_compute_risk(snap_a, model_bundle)
                pred_b = _try_compute_risk(snap_b, model_bundle)

                _render_evolution_comparison(
                    snap_a=snap_a,
                    snap_b=snap_b,
                    diff=diff,
                    pred_a=pred_a,
                    pred_b=pred_b,
                    model_bundle=model_bundle,
                    impact=impact
                )
            else:
                st.error("Failed to build snapshots. Check commit references and repository access.")
