"""
Tests for Intelligent Recommendation Engine.
"""
import pytest
from analysis.dependency_analyzer import DependencyGraph
from ml.predict import ComponentRiskProfile
from recommendations.recommendation_engine import (
    IntelligentRecommendationEngine,
    RecommendationPriority,
    RecommendationCategory
)


def test_high_complexity_recommendation_trigger():
    """Verifies that high complexity triggers a refactoring recommendation."""
    engine = IntelligentRecommendationEngine()
    profile = ComponentRiskProfile(
        filepath="services/payment.py",
        risk_probability=0.85,
        risk_percentage=85,
        risk_category="HIGH",
        loc=200,
        complexity=15,  # High!
        functions=4,
        classes=1,
        imports=3,
        methods=4,
        commit_count=5,
        recent_commit_count=2,
        churn=300,
        recent_churn=100,
        change_frequency=0.20,
        number_of_authors=2,
        time_since_last_change=2.0,
        dependency_count=2,
        dependent_count=1,
        degree=3,
        centrality=0.10
    )

    recs = engine.generate_recommendations(profile)
    assert len(recs) > 0
    comp_rec = next((r for r in recs if r.category == RecommendationCategory.REFACTORING), None)
    assert comp_rec is not None
    assert comp_rec.priority in (RecommendationPriority.HIGH, RecommendationPriority.CRITICAL)
    assert "complexity" in comp_rec.triggering_metrics
    assert comp_rec.triggering_metrics["complexity"] == 15.0


def test_circular_dependency_recommendation():
    """Verifies that circular dependency participation triggers CRITICAL priority recommendation."""
    dep_graph = DependencyGraph()
    dep_graph.add_dependency("mod_a.py", "mod_b.py")
    dep_graph.add_dependency("mod_b.py", "mod_a.py")

    engine = IntelligentRecommendationEngine(dep_graph=dep_graph)
    profile = ComponentRiskProfile(
        filepath="mod_a.py",
        risk_probability=0.60,
        risk_percentage=60,
        risk_category="MEDIUM",
        loc=50,
        complexity=4,
        functions=2,
        classes=1,
        imports=1,
        methods=2,
        commit_count=2,
        recent_commit_count=1,
        churn=50,
        recent_churn=20,
        change_frequency=0.10,
        number_of_authors=1,
        time_since_last_change=5.0,
        dependency_count=1,
        dependent_count=1,
        degree=2,
        centrality=0.20
    )

    recs = engine.generate_recommendations(profile)
    cycle_rec = next((r for r in recs if "circular" in r.action.lower() or "cycle" in r.action.lower()), None)
    assert cycle_rec is not None
    assert cycle_rec.priority == RecommendationPriority.CRITICAL
    assert cycle_rec.category == RecommendationCategory.ARCHITECTURE


def test_repository_wide_recommendation_ranking():
    """Verifies that repository-wide recommendations are prioritized from CRITICAL to LOW."""
    engine = IntelligentRecommendationEngine()
    profiles = [
        ComponentRiskProfile(
            filepath="low.py", risk_probability=0.10, risk_percentage=10, risk_category="LOW",
            loc=30, complexity=2, functions=1, classes=1, imports=1, methods=1,
            commit_count=1, recent_commit_count=0, churn=30, recent_churn=0,
            change_frequency=0.05, number_of_authors=2, time_since_last_change=10.0,
            dependency_count=1, dependent_count=0, degree=1, centrality=0.05
        ),
        ComponentRiskProfile(
            filepath="high.py", risk_probability=0.90, risk_percentage=90, risk_category="HIGH",
            loc=350, complexity=22, functions=8, classes=2, imports=5, methods=8,
            commit_count=12, recent_commit_count=6, churn=800, recent_churn=400,
            change_frequency=0.40, number_of_authors=1, time_since_last_change=1.0,
            dependency_count=4, dependent_count=4, degree=8, centrality=0.50
        )
    ]

    recs = engine.generate_repository_recommendations(profiles)
    assert len(recs) >= 3
    # Top recommendation should be CRITICAL or HIGH priority
    assert recs[0].priority in (RecommendationPriority.CRITICAL, RecommendationPriority.HIGH)
