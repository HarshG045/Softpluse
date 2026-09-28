"""
SoftwarePulse: Evolution-Aware Software Risk Prediction and What-If Change Analysis.
Main Streamlit Application Entrypoint.
"""
import sys
import logging
from pathlib import Path
import streamlit as st

# Ensure project root is on sys.path
BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from analysis.change_analyzer import ChangeImpactAnalyzer
from analysis.code_analyzer import CodeAnalyzer
from analysis.dependency_analyzer import DependencyAnalyzer
from analysis.git_analyzer import GitAnalyzer
from analysis.historical_analyzer import HistoricalAnalyzer
from config import DEMO_DATA_DIR, RANDOM_STATE
from data.demo.demo_repo_builder import create_demo_repository
from dataset.builder import DatasetBuilder
from explainability.explainer import ModelExplainer
from ingestion.git_loader import GitLoader
from ingestion.repository_loader import RepositoryLoader
from ml.evaluate import ModelEvaluator
from ml.predict import RiskPredictor
from ml.train import ClassifierAlgorithm, ModelTrainer
from simulation.what_if import WhatIfEngine
from ui.header import render_header
from ui.pages import (
    render_architecture_page,
    render_dependencies_page,
    render_evolution_page,
    render_explainability_page,
    render_git_activity_page,
    render_model_evaluation_page,
    render_overview_page,
    render_risk_explorer_page,
    render_settings_page,
    render_what_if_lab_page,
)
from ui.sidebar import NavigationPage, render_sidebar
from ui.styles import get_custom_css


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("SoftwarePulse")

# Streamlit Configuration
st.set_page_config(
    page_title="SoftwarePulse | Software Evolution Intelligence",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Apply Developer Theme CSS
st.markdown(get_custom_css(), unsafe_allow_html=True)


def execute_repository_analysis(repo_input: str, is_demo: bool, algorithm: ClassifierAlgorithm):
    """Executes the complete analysis, feature engineering, and model training pipeline."""
    with st.status("Analyzing repository...", expanded=True) as status:
        # Step 1: Ensure repository exists
        st.write("Ingesting repository...")
        if is_demo:
            repo_path = create_demo_repository()
            repo_loader = RepositoryLoader()
            repo_path, metadata = repo_loader.load_repository(str(repo_path))
        else:
            repo_loader = RepositoryLoader()
            repo_path, metadata = repo_loader.load_repository(repo_input)

        # Step 2: Git History Extraction
        st.write("Extracting Git commit history & churn logs...")
        git_loader = GitLoader(str(repo_path))
        commits = git_loader.get_commit_history()

        # Step 3: AST Code Metrics
        st.write("Parsing Python AST metrics (LOC, Cyclomatic Complexity, Imports)...")
        code_analyzer = CodeAnalyzer(repo_path)
        code_map = code_analyzer.analyze_repository()

        # Step 4: Dependency Graph
        st.write("Constructing component dependency graph & centrality metrics...")
        dep_analyzer = DependencyAnalyzer(repo_path)
        dep_graph = dep_analyzer.build_graph(code_map)

        # Step 5: Historical Snapshots
        st.write("Reconstructing historical evolution snapshots...")
        hist_analyzer = HistoricalAnalyzer(commits, code_map, dep_graph)
        snapshots = hist_analyzer.build_snapshots(num_snapshots=5)

        # Step 6: Feature Engineering & Leakage-Free Dataset Building
        st.write("Building temporal dataset and computing risk labels...")
        builder = DatasetBuilder(commits, code_map, dep_graph)
        dataset = builder.build_dataset()

        # Step 7: Train ML Risk Prediction Model
        st.write(f"Training {algorithm.value} model & running comparative evaluation...")
        trainer = ModelTrainer(random_state=RANDOM_STATE)
        model_bundle = trainer.train_model(
            dataset.split_result.X_train,
            dataset.split_result.y_train,
            algorithm=algorithm
        )

        # Step 8: Risk Predictions & Explainability
        st.write("Generating component risk predictions and attribution signals...")
        predictor = RiskPredictor()
        predictions_res = predictor.predict(model_bundle, dataset.current_features_df)

        explainer = ModelExplainer(model_bundle, background_data=dataset.split_result.X_train)
        evaluator = ModelEvaluator()
        comp_report = evaluator.run_comparative_experiment(dataset.split_result, algorithm=algorithm)
        what_if_engine = WhatIfEngine(model_bundle, dep_graph)
        change_analyzer = ChangeImpactAnalyzer(model_bundle, dep_graph, predictions_res.profiles)

        status.update(label="Repository Analysis Complete ✓", state="complete", expanded=False)

    return {
        "metadata": metadata,
        "commits": commits,
        "git_loader": git_loader,
        "code_map": code_map,
        "dep_graph": dep_graph,
        "snapshots": snapshots,
        "dataset": dataset,
        "model_bundle": model_bundle,
        "predictions_res": predictions_res,
        "explainer": explainer,
        "comp_report": comp_report,
        "what_if_engine": what_if_engine,
        "change_analyzer": change_analyzer,
        "is_analyzed": True
    }


def main():
    # Retrieve existing state if available for sidebar metadata
    cached_data = st.session_state.get("analysis_data")
    repo_meta = cached_data.get("metadata") if cached_data else None
    pred_res = cached_data.get("predictions_res") if cached_data else None
    model_bundle = cached_data.get("model_bundle") if cached_data else None

    selected_page, source_type, repo_input, selected_algo, analyze_clicked = render_sidebar(
        repo_meta=repo_meta,
        pred_res=pred_res,
        model_bundle=model_bundle
    )
    is_demo = (source_type == "Demo Microservices Repo")

    # Initialize analysis on first run if not already stored or if analyze button clicked
    if analyze_clicked or st.session_state.get("analysis_data") is None:
        try:
            target_input = repo_input if repo_input else str(DEMO_DATA_DIR / "softwarepulse_demo_repo")
            st.session_state.analysis_data = execute_repository_analysis(target_input, is_demo, selected_algo)
            st.rerun()
        except Exception as err:
            st.error(f"Repository analysis failed: {err}")
            with st.expander("Technical details"):
                st.exception(err)
            return

    data = st.session_state.analysis_data

    # Render Top Header
    render_header(
        repo_meta=data["metadata"],
        model_bundle=data["model_bundle"],
        is_analyzed=data["is_analyzed"]
    )

    # Route Pages
    if selected_page == NavigationPage.OVERVIEW:
        render_overview_page(data["metadata"], data["predictions_res"], data["comp_report"])

    elif selected_page == NavigationPage.RISK_EXPLORER:
        render_risk_explorer_page(
            data["predictions_res"],
            data["dep_graph"],
            data["explainer"],
            data["dataset"].current_features_df,
            git_loader=data.get("git_loader")
        )

    elif selected_page == NavigationPage.ARCHITECTURE:
        render_architecture_page(data["dep_graph"], data["predictions_res"])

    elif selected_page == NavigationPage.DEPENDENCIES:
        render_dependencies_page(data["dep_graph"], data["predictions_res"])

    elif selected_page == NavigationPage.EVOLUTION:
        render_evolution_page(data["snapshots"], data["predictions_res"])

    elif selected_page == NavigationPage.GIT_ACTIVITY:
        render_git_activity_page(
            data["git_loader"],
            data["metadata"],
            data["predictions_res"],
            data["change_analyzer"]
        )

    elif selected_page == NavigationPage.EXPLAINABILITY:
        render_explainability_page(
            data["explainer"],
            data["predictions_res"],
            data["dataset"].current_features_df
        )

    elif selected_page == NavigationPage.WHAT_IF_LAB:
        render_what_if_lab_page(data["what_if_engine"], data["predictions_res"], data["dep_graph"])

    elif selected_page == NavigationPage.MODEL_EVALUATION:
        render_model_evaluation_page(data["dataset"], data["comp_report"], selected_algo)

    elif selected_page == NavigationPage.SETTINGS:
        render_settings_page(data["metadata"], data["predictions_res"])


if __name__ == "__main__":
    main()
