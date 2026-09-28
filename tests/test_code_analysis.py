"""
Unit tests for CodeAnalyzer AST parsing and complexity extraction.
"""
import pytest
from pathlib import Path
from analysis.code_analyzer import CodeAnalyzer
from data.demo.demo_repo_builder import create_demo_repository

@pytest.fixture(scope="module")
def demo_repo():
    return create_demo_repository()

def test_ast_code_analyzer(demo_repo):
    analyzer = CodeAnalyzer(demo_repo)
    results = analyzer.analyze_repository()
    
    assert len(results) >= 5
    payment_key = next((k for k in results if "payment_service.py" in k), None)
    assert payment_key is not None
    m = results[payment_key]
    assert m.loc > 20
    assert m.complexity >= 5
    assert m.classes >= 1
    assert m.functions >= 1
    assert len(m.imported_modules) >= 2
    assert m.has_parse_error is False
