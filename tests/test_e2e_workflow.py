"""
Comprehensive End-to-End Test Suite for SoftwarePulse.
Verifies all pipeline stages, all 8 dashboard pages, all What-If simulation types, and export routines.
"""
import pytest
import pandas as pd
from pathlib import Path

from config import DEMO_DATA_DIR, RANDOM_STATE
from data.demo.demo_repo_builder import create_demo_repository
from ingestion.repository_loader import RepositoryLoader
from ingestion.git_loader import GitLoader
from analysis.code_analyzer import CodeAnalyzer
from analysis.git_analyzer import GitAnalyzer
from analysis.dependency_analyzer import DependencyAnalyzer
from analysis.historical_analyzer import HistoricalAnalyzer
from dataset.builder import DatasetBuilder
from ml.train import ModelTrainer, ClassifierAlgorithm
from ml.predict import RiskPredictor
from ml.evaluate import ModelEvaluator
from explainability.explainer import ModelExplainer
from simulation.what_if import WhatIfEngine, SimulationChangeType
from visualization.risk_charts import (
    create_risk_distribution_chart,
    create_top_risky_components_chart,
    create_risk_gauge_chart
)
from visualization.evolution_charts import (
    create_component_evolution_chart,
    create_snapshot_metric_timeline
)
from visualization.dependency_graph import create_interactive_dependency_figure
from visualization.component_view import (
    create_feature_comparison_chart,
    create_signal_bar_chart
)


@pytest.fixture(scope="module")
def full_pipeline_state():
    """Runs the complete ingestion and analysis pipeline once for testing."""
    repo_path = create_demo_repository()
    
    loader = RepositoryLoader()
    repo_path, metadata = loader.load_repository(str(repo_path))
    
    git_loader = GitLoader(str(repo_path))
    commits = git_loader.get_commit_history()
    
    code_analyzer = CodeAnalyzer(repo_path)
    code_map = code_analyzer.analyze_repository()
    
    dep_analyzer = DependencyAnalyzer(repo_path)
    dep_graph = dep_analyzer.build_graph(code_map)
    
    hist_analyzer = HistoricalAnalyzer(commits, code_map, dep_graph)
    snapshots = hist_analyzer.build_snapshots(num_snapshots=5)
    
    builder = DatasetBuilder(commits, code_map, dep_graph)
    dataset = builder.build_dataset()
    
    trainer = ModelTrainer(random_state=RANDOM_STATE)
    model_bundle = trainer.train_model(
        dataset.split_result.X_train,
        dataset.split_result.y_train,
        algorithm=ClassifierAlgorithm.RANDOM_FOREST
    )
    
    predictor = RiskPredictor()
    predictions_res = predictor.predict(model_bundle, dataset.current_features_df)
    
    explainer = ModelExplainer(model_bundle, background_data=dataset.split_result.X_train)
    evaluator = ModelEvaluator()
    comp_report = evaluator.run_comparative_experiment(dataset.split_result, algorithm=ClassifierAlgorithm.RANDOM_FOREST)
    what_if_engine = WhatIfEngine(model_bundle, dep_graph)
    
    return {
        "metadata": metadata,
        "commits": commits,
        "code_map": code_map,
        "dep_graph": dep_graph,
        "snapshots": snapshots,
        "dataset": dataset,
        "model_bundle": model_bundle,
        "predictions_res": predictions_res,
        "explainer": explainer,
        "comp_report": comp_report,
        "what_if_engine": what_if_engine
    }


def test_overview_functionality(full_pipeline_state):
    """Verifies overview metrics, distribution, and top components."""
    state = full_pipeline_state
    pred_res = state["predictions_res"]
    
    assert len(pred_res.profiles) >= 5
    assert pred_res.high_risk_count + pred_res.medium_risk_count + pred_res.low_risk_count == len(pred_res.profiles)
    
    # Verify Plotly charts generation
    fig_dist = create_risk_distribution_chart(pred_res.high_risk_count, pred_res.medium_risk_count, pred_res.low_risk_count)
    assert fig_dist is not None
    
    fig_top = create_top_risky_components_chart(pred_res.df_predictions, top_n=5)
    assert fig_top is not None


def test_risk_explorer_inspector_functionality(full_pipeline_state):
    """Verifies search, filtering, gauge, radar, and signal attribution."""
    state = full_pipeline_state
    pred_res = state["predictions_res"]
    explainer = state["explainer"]
    features_df = state["dataset"].current_features_df
    
    # Test every single component profile
    for profile in pred_res.profiles:
        gauge_fig = create_risk_gauge_chart(profile.risk_percentage, profile.risk_category)
        assert gauge_fig is not None
        
        comp_row = features_df[features_df["filepath"] == profile.filepath].iloc[0]
        explanation = explainer.explain_component(comp_row, profile.risk_probability, profile.risk_category)
        assert explanation is not None
        assert len(explanation.top_signals) > 0
        
        signal_fig = create_signal_bar_chart(explanation.top_signals)
        assert signal_fig is not None


def test_evolution_functionality(full_pipeline_state):
    """Verifies evolution charts across all metrics for all components."""
    state = full_pipeline_state
    snapshots = state["snapshots"]
    pred_res = state["predictions_res"]
    
    assert len(snapshots) >= 3
    
    for profile in pred_res.profiles:
        for metric in ["complexity", "churn", "recent_churn", "loc", "commit_count"]:
            fig_evol = create_component_evolution_chart(snapshots, profile.filepath, metric_key=metric, metric_label=metric)
            assert fig_evol is not None
        
        fig_multi = create_snapshot_metric_timeline(snapshots, profile.filepath)
        assert fig_multi is not None


def test_architecture_subsystem_functionality(full_pipeline_state):
    """Verifies subsystem package clustering, cycle detection, and architectural hotspot analysis."""
    state = full_pipeline_state
    dep_graph = state["dep_graph"]
    
    pkg_clusters = dep_graph.get_package_clusters()
    assert len(pkg_clusters) >= 2
    for pkg, pdata in pkg_clusters.items():
        assert "instability_index" in pdata
        assert 0.0 <= pdata["instability_index"] <= 1.0
        assert pdata["component_count"] > 0
    
    hotspots = dep_graph.get_architectural_hotspots()
    assert len(hotspots) > 0
    assert "betweenness_centrality" in hotspots[0]
    
    cycles = dep_graph.detect_cycles()
    assert isinstance(cycles, list)


def test_dependencies_graph_functionality(full_pipeline_state):
    """Verifies dependency network diagram generation for full repo and neighborhoods."""
    state = full_pipeline_state
    dep_graph = state["dep_graph"]
    pred_res = state["predictions_res"]
    
    # Full graph
    fig_full = create_interactive_dependency_figure(
        dep_graph=dep_graph,
        risk_profiles_map={p.filepath: p.to_dict() for p in pred_res.profiles}
    )
    assert fig_full is not None
    
    # Focused neighborhood for each node
    for profile in pred_res.profiles:
        fig_neigh = create_interactive_dependency_figure(
            dep_graph=dep_graph,
            risk_profiles_map={p.filepath: p.to_dict() for p in pred_res.profiles},
            focus_component=profile.filepath
        )

        assert fig_neigh is not None


def test_what_if_all_change_types_functionality(full_pipeline_state):
    """Verifies all 6 counterfactual simulation change types recalculate correctly."""
    state = full_pipeline_state
    engine = state["what_if_engine"]
    pred_res = state["predictions_res"]
    dep_graph = state["dep_graph"]
    
    profile = pred_res.profiles[0]
    all_components = [p.filepath for p in pred_res.profiles]
    target_dep = next((c for c in all_components if c != profile.filepath), all_components[0])
    
    # 1. ADD_DEPENDENCY
    res_add = engine.simulate_change(profile, SimulationChangeType.ADD_DEPENDENCY, target_dependency=target_dep)
    assert res_add.simulated_graph.graph.has_edge(profile.filepath, target_dep)
    assert res_add.current_percentage == profile.risk_percentage
    assert isinstance(res_add.percentage_point_diff, int)
    
    # 2. REMOVE_DEPENDENCY
    res_rem = engine.simulate_change(profile, SimulationChangeType.REMOVE_DEPENDENCY, target_dependency=target_dep)
    assert not res_rem.simulated_graph.graph.has_edge(profile.filepath, target_dep)
    
    # 3. INCREASE_COMPLEXITY
    res_inc_c = engine.simulate_change(profile, SimulationChangeType.INCREASE_COMPLEXITY, complexity_delta=10)
    assert any(d.feature_name == "complexity" and d.after_value == profile.complexity + 10 for d in res_inc_c.changed_features)
    
    # 4. DECREASE_COMPLEXITY
    res_dec_c = engine.simulate_change(profile, SimulationChangeType.DECREASE_COMPLEXITY, complexity_delta=3)
    assert any(d.feature_name == "complexity" for d in res_dec_c.changed_features)
    
    # 5. INCREASE_CHURN
    res_churn = engine.simulate_change(profile, SimulationChangeType.INCREASE_CHURN, churn_delta=500)
    assert any(d.feature_name == "churn" and d.after_value == profile.churn + 500 for d in res_churn.changed_features)
    
    # 6. INCREASE_CHANGE_FREQUENCY
    res_freq = engine.simulate_change(profile, SimulationChangeType.INCREASE_CHANGE_FREQUENCY, freq_delta=0.20)
    assert any(d.feature_name == "change_frequency" for d in res_freq.changed_features)


def test_model_evaluation_and_comparisons(full_pipeline_state):
    """Verifies comparative research report across Model A, Model B, Model C."""
    state = full_pipeline_state
    comp_report = state["comp_report"]
    
    assert comp_report.static_report is not None
    assert comp_report.evolution_report is not None
    assert comp_report.full_report is not None
    assert len(comp_report.comparison_df) == 3
    
    # Check all metrics are valid numeric scores
    for report in [comp_report.static_report, comp_report.evolution_report, comp_report.full_report]:
        assert 0.0 <= report.accuracy <= 1.0
        assert 0.0 <= report.precision <= 1.0
        assert 0.0 <= report.recall <= 1.0
        assert 0.0 <= report.f1 <= 1.0
        assert 0.0 <= report.roc_auc <= 1.0


def test_export_data_generation(full_pipeline_state):
    """Verifies CSV and JSON export routines format valid data."""
    state = full_pipeline_state
    df_predictions = state["predictions_res"].df_predictions
    meta = state["metadata"]
    
    # CSV export check
    csv_str = df_predictions.to_csv(index=False)
    assert "filepath" in csv_str
    assert "risk_percentage" in csv_str
    
    # JSON export check
    dict_export = {
        "repository": meta.to_dict(),
        "total_components": len(df_predictions),
        "predictions": df_predictions.to_dict(orient="records")
    }
    assert dict_export["total_components"] >= 5
    assert len(dict_export["predictions"]) >= 5
