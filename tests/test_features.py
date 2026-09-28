"""
Unit tests for FeatureEngineer and DatasetBuilder.
"""
import pytest
from pathlib import Path
from ingestion.git_loader import GitLoader
from analysis.code_analyzer import CodeAnalyzer
from analysis.dependency_analyzer import DependencyAnalyzer
from dataset.builder import DatasetBuilder
from features.feature_schema import FULL_FEATURES
from data.demo.demo_repo_builder import create_demo_repository

@pytest.fixture(scope="module")
def demo_repo():
    return create_demo_repository()

def test_dataset_building_and_features(demo_repo):
    git_loader = GitLoader(str(demo_repo))
    commits = git_loader.get_commit_history()
    
    code_analyzer = CodeAnalyzer(demo_repo)
    code_map = code_analyzer.analyze_repository()
    
    dep_analyzer = DependencyAnalyzer(demo_repo)
    dep_graph = dep_analyzer.build_graph(code_map)
    
    builder = DatasetBuilder(commits, code_map, dep_graph)
    dataset = builder.build_dataset()
    
    assert dataset.total_components_count >= 5
    assert not dataset.current_features_df.empty
    for feat in FULL_FEATURES:
        assert feat in dataset.current_features_df.columns
    
    assert dataset.split_result.train_samples_count > 0
    assert dataset.split_result.test_samples_count > 0
