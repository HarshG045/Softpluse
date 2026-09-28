"""
Unit tests for DependencyAnalyzer and DependencyGraph construction.
"""
import pytest
from pathlib import Path
from analysis.code_analyzer import CodeAnalyzer
from analysis.dependency_analyzer import DependencyAnalyzer
from data.demo.demo_repo_builder import create_demo_repository

@pytest.fixture(scope="module")
def demo_repo():
    return create_demo_repository()

def test_dependency_graph_building(demo_repo):
    code_analyzer = CodeAnalyzer(demo_repo)
    code_map = code_analyzer.analyze_repository()
    
    dep_analyzer = DependencyAnalyzer(demo_repo)
    graph = dep_analyzer.build_graph(code_map)
    
    struct_metrics = graph.get_metrics_for_all()
    assert len(struct_metrics) >= 5
    
    # payment_service imports database and auth_service
    payment_key = next((k for k in struct_metrics if "payment_service.py" in k), None)
    assert payment_key is not None
    assert struct_metrics[payment_key].out_degree >= 1
    assert struct_metrics[payment_key].degree >= 1
