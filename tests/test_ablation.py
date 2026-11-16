"""
Tests for Feature Ablation Study and Research Experiment engine.
"""
import pytest
from data.demo.demo_repo_builder import create_demo_repository
from ingestion.git_loader import GitLoader
from analysis.code_analyzer import CodeAnalyzer
from analysis.dependency_analyzer import DependencyAnalyzer
from dataset.builder import DatasetBuilder
from experiments.ablation import FeatureAblationExperiment
from ml.train import ClassifierAlgorithm


@pytest.fixture(scope="module")
def demo_split():
    repo_path = create_demo_repository()
    commits = GitLoader(str(repo_path)).get_commit_history()
    code_map = CodeAnalyzer(repo_path).analyze_repository()
    dep_graph = DependencyAnalyzer(repo_path).build_graph(code_map)
    builder = DatasetBuilder(commits, code_map, dep_graph, repo_path=repo_path)
    dataset = builder.build_dataset()
    return dataset.split_result


def test_4_way_feature_ablation_execution(demo_split):
    """Verifies that the ablation study executes all 4 model configurations plus baseline."""
    experiment = FeatureAblationExperiment()
    report = experiment.run_ablation(demo_split, algorithm=ClassifierAlgorithm.RANDOM_FOREST)

    assert report is not None
    assert len(report.comparison_df) == 5  # Baseline + 4 configs
    assert report.static_result.feature_count == 6
    assert report.evolution_result.feature_count == 13
    assert report.dependency_result.feature_count == 12
    assert report.full_result.feature_count == 19

    # All models must evaluate on identical sample counts
    n_test = len(demo_split.y_test)
    assert report.static_result.evaluation_report.test_samples == n_test
    assert report.evolution_result.evaluation_report.test_samples == n_test
    assert report.dependency_result.evaluation_report.test_samples == n_test
    assert report.full_result.evaluation_report.test_samples == n_test

    # Verification of delta vs static
    assert report.static_result.f1_delta_vs_static == 0.0
    assert report.insights_summary != ""


def test_ablation_with_logistic_regression(demo_split):
    """Verifies feature ablation runs with Logistic Regression."""
    experiment = FeatureAblationExperiment()
    report = experiment.run_ablation(demo_split, algorithm=ClassifierAlgorithm.LOGISTIC_REGRESSION)

    assert report.algorithm == ClassifierAlgorithm.LOGISTIC_REGRESSION
    assert len(report.comparison_df) == 5
    for _, row in report.comparison_df.iterrows():
        assert 0.0 <= row["Accuracy"] <= 1.0
        assert 0.0 <= row["F1-Score"] <= 1.0
