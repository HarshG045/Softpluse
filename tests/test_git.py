"""
Unit tests for GitLoader and GitAnalyzer.
"""
import pytest
from pathlib import Path
from ingestion.git_loader import GitLoader
from analysis.git_analyzer import GitAnalyzer
from data.demo.demo_repo_builder import create_demo_repository

@pytest.fixture(scope="module")
def demo_repo():
    return create_demo_repository()

def test_git_loader_extraction(demo_repo):
    loader = GitLoader(str(demo_repo))
    assert loader.is_git_repo() is True
    commits = loader.get_commit_history()
    assert len(commits) >= 5
    assert any(c.is_bugfix for c in commits)
    assert commits[0].commit_hash != ""

def test_git_analyzer_metrics(demo_repo):
    loader = GitLoader(str(demo_repo))
    commits = loader.get_commit_history()
    analyzer = GitAnalyzer(commits)
    metrics_map = analyzer.analyze_components()
    
    assert len(metrics_map) > 0
    # Payment service should have high churn and commits
    payment_key = next((k for k in metrics_map if "payment_service.py" in k), None)
    assert payment_key is not None
    assert metrics_map[payment_key].commit_count >= 2
    assert metrics_map[payment_key].churn > 0
    assert metrics_map[payment_key].number_of_authors >= 1
