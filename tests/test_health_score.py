"""
Tests for Repository Health Score calculation, dimension metrics, and weighting.
"""
import pytest
from analysis.code_analyzer import CodeMetrics
from analysis.dependency_analyzer import DependencyGraph
from analysis.git_analyzer import ComponentGitMetrics
from health.health_score import RepositoryHealthScore, HealthDimension
from ingestion.git_loader import CommitRecord
from datetime import datetime, timezone


def test_health_score_with_clean_codebase():
    """Verifies that a well-structured repository receives an A grade."""
    code_map = {
        "service.py": CodeMetrics(filepath="service.py", loc=50, complexity=3, functions=3, classes=1, comment_ratio=0.15),
        "tests/test_service.py": CodeMetrics(filepath="tests/test_service.py", loc=40, complexity=1, functions=2)
    }
    dep_graph = DependencyGraph()
    dep_graph.add_component("service.py")
    dep_graph.add_component("tests/test_service.py")
    dep_graph.add_dependency("tests/test_service.py", "service.py")

    commits = [
        CommitRecord("c1", "Alice", "alice@dev.com", datetime.now(timezone.utc), "feat: init", ["service.py"], 50, 0),
        CommitRecord("c2", "Bob", "bob@dev.com", datetime.now(timezone.utc), "test: add tests", ["tests/test_service.py"], 40, 0)
    ]
    git_map = {
        "service.py": ComponentGitMetrics(filepath="service.py", commit_count=1, churn=50, recent_churn=0, number_of_authors=1),
        "tests/test_service.py": ComponentGitMetrics(filepath="tests/test_service.py", commit_count=1, churn=40, recent_churn=0, number_of_authors=1)
    }

    scorer = RepositoryHealthScore()
    report = scorer.evaluate_repository(
        code_map=code_map,
        git_map=git_map,
        dep_graph=dep_graph,
        commits=commits,
        all_repo_files=["service.py", "tests/test_service.py"]
    )

    assert report.overall_score >= 80
    assert report.overall_grade in ("A", "B")
    assert report.dimensions[HealthDimension.CODE_QUALITY].score >= 80
    assert report.dimensions[HealthDimension.DEPENDENCY_HEALTH].score == 100


def test_health_score_circular_dependency_penalty():
    """Verifies that circular dependencies penalize Dependency Health."""
    dep_graph = DependencyGraph()
    dep_graph.add_dependency("a.py", "b.py")
    dep_graph.add_dependency("b.py", "a.py")  # Circular cycle!

    code_map = {
        "a.py": CodeMetrics(filepath="a.py", loc=30, complexity=2),
        "b.py": CodeMetrics(filepath="b.py", loc=30, complexity=2)
    }
    commits = [CommitRecord("c1", "A", "a@a.com", datetime.now(timezone.utc), "init", ["a.py", "b.py"], 60, 0)]
    git_map = {
        "a.py": ComponentGitMetrics(filepath="a.py", commit_count=1, churn=30),
        "b.py": ComponentGitMetrics(filepath="b.py", commit_count=1, churn=30)
    }

    scorer = RepositoryHealthScore()
    report = scorer.evaluate_repository(
        code_map=code_map,
        git_map=git_map,
        dep_graph=dep_graph,
        commits=commits
    )

    dep_score = report.dimensions[HealthDimension.DEPENDENCY_HEALTH]
    assert dep_score.score <= 75
    assert any("circular" in w.lower() for w in dep_score.weaknesses)


def test_empty_repository_health_handling():
    """Verifies that empty repositories are handled gracefully without exceptions."""
    scorer = RepositoryHealthScore()
    report = scorer.evaluate_repository(
        code_map={},
        git_map={},
        dep_graph=DependencyGraph(),
        commits=[]
    )
    assert report is not None
    assert 0 <= report.overall_score <= 100
