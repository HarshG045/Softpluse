"""
Unit tests for ModelTrainer, RiskPredictor, and ModelEvaluator.
"""
import pytest
from pathlib import Path
from ingestion.git_loader import GitLoader
from analysis.code_analyzer import CodeAnalyzer
from analysis.dependency_analyzer import DependencyAnalyzer
from dataset.builder import DatasetBuilder
from ml.train import ModelTrainer, ClassifierAlgorithm
from ml.predict import RiskPredictor
from ml.evaluate import ModelEvaluator
from features.feature_schema import FeatureSetType
from data.demo.demo_repo_builder import create_demo_repository

@pytest.fixture(scope="module")
def demo_dataset():
    repo_path = create_demo_repository()
    commits = GitLoader(str(repo_path)).get_commit_history()
    code_map = CodeAnalyzer(repo_path).analyze_repository()
    dep_graph = DependencyAnalyzer(repo_path).build_graph(code_map)
    builder = DatasetBuilder(commits, code_map, dep_graph)
    return builder.build_dataset(), dep_graph

def test_model_training_and_prediction(demo_dataset):
    dataset, dep_graph = demo_dataset
    trainer = ModelTrainer()
    
    bundle = trainer.train_model(
        dataset.split_result.X_train,
        dataset.split_result.y_train,
        algorithm=ClassifierAlgorithm.RANDOM_FOREST,
        feature_set_type=FeatureSetType.FULL
    )
    assert bundle.pipeline is not None
    assert len(bundle.feature_names) > 0
    
    predictor = RiskPredictor()
    pred_res = predictor.predict(bundle, dataset.current_features_df)
    
    assert len(pred_res.profiles) >= 5
    assert pred_res.high_risk_count + pred_res.medium_risk_count + pred_res.low_risk_count == len(pred_res.profiles)
    for p in pred_res.profiles:
        assert 0.0 <= p.risk_probability <= 1.0
        assert p.risk_category in ("LOW", "MEDIUM", "HIGH")

def test_comparative_evaluation(demo_dataset):
    dataset, _ = demo_dataset
    evaluator = ModelEvaluator()
    comp_report = evaluator.run_comparative_experiment(dataset.split_result)
    
    assert comp_report.static_report is not None
    assert comp_report.evolution_report is not None
    assert comp_report.full_report is not None
    assert len(comp_report.comparison_df) == 3
