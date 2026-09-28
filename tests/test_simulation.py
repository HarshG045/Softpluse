"""
Unit tests for WhatIfEngine simulation.
"""
import pytest
from pathlib import Path
from ingestion.git_loader import GitLoader
from analysis.code_analyzer import CodeAnalyzer
from analysis.dependency_analyzer import DependencyAnalyzer
from dataset.builder import DatasetBuilder
from ml.train import ModelTrainer, ClassifierAlgorithm
from ml.predict import RiskPredictor
from simulation.what_if import WhatIfEngine, SimulationChangeType
from features.feature_schema import FeatureSetType
from data.demo.demo_repo_builder import create_demo_repository

def test_what_if_simulation():
    repo_path = create_demo_repository()
    commits = GitLoader(str(repo_path)).get_commit_history()
    code_map = CodeAnalyzer(repo_path).analyze_repository()
    dep_analyzer = DependencyAnalyzer(repo_path)
    dep_graph = dep_analyzer.build_graph(code_map)
    
    builder = DatasetBuilder(commits, code_map, dep_graph)
    dataset = builder.build_dataset()
    
    trainer = ModelTrainer()
    bundle = trainer.train_model(
        dataset.split_result.X_train,
        dataset.split_result.y_train,
        algorithm=ClassifierAlgorithm.RANDOM_FOREST,
        feature_set_type=FeatureSetType.FULL
    )
    
    predictor = RiskPredictor()
    pred_res = predictor.predict(bundle, dataset.current_features_df)
    
    target_profile = pred_res.profiles[0]
    engine = WhatIfEngine(bundle, dep_graph)
    
    # Test increase complexity
    sim_res = engine.simulate_change(
        current_profile=target_profile,
        change_type=SimulationChangeType.INCREASE_COMPLEXITY,
        complexity_delta=10
    )
    assert sim_res.current_percentage == target_profile.risk_percentage
    assert len(sim_res.changed_features) >= 1
    assert sim_res.scientific_disclaimer != ""
    
    # Test add dependency
    all_nodes = list(dep_graph.graph.nodes)
    target_dep = next((n for n in all_nodes if n != target_profile.filepath), all_nodes[0])
    
    sim_dep_res = engine.simulate_change(
        current_profile=target_profile,
        change_type=SimulationChangeType.ADD_DEPENDENCY,
        target_dependency=target_dep
    )
    assert sim_dep_res.simulated_graph.graph.has_edge(target_profile.filepath, target_dep)
