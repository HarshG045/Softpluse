"""
Feature Schema: Definitions of static, evolution, and structural feature sets.
"""
from enum import Enum
from typing import Dict, List

class FeatureSetType(str, Enum):
    STATIC = "Static Only"
    STATIC_PLUS_EVOLUTION = "Static + Evolution"
    FULL = "Full (Static + Evolution + Dependency)"

STATIC_FEATURES = [
    "loc",
    "functions",
    "classes",
    "complexity",
    "imports",
    "methods"
]

EVOLUTION_FEATURES = [
    "commit_count",
    "recent_commit_count",
    "churn",
    "recent_churn",
    "change_frequency",
    "number_of_authors",
    "time_since_last_change"
]

STRUCTURAL_FEATURES = [
    "dependency_count",
    "dependent_count",
    "degree",
    "centrality",
    "in_degree",
    "out_degree"
]

STATIC_PLUS_EVOLUTION_FEATURES = STATIC_FEATURES + EVOLUTION_FEATURES
FULL_FEATURES = STATIC_FEATURES + EVOLUTION_FEATURES + STRUCTURAL_FEATURES

FEATURE_SET_MAP = {
    FeatureSetType.STATIC: STATIC_FEATURES,
    FeatureSetType.STATIC_PLUS_EVOLUTION: STATIC_PLUS_EVOLUTION_FEATURES,
    FeatureSetType.FULL: FULL_FEATURES
}

FEATURE_DESCRIPTIONS: Dict[str, str] = {
    "loc": "Lines of Code (excluding blank/comment lines)",
    "functions": "Total number of functions defined in module",
    "classes": "Number of classes declared in component",
    "complexity": "McCabe Cyclomatic Complexity (branch paths)",
    "imports": "Number of external/internal import statements",
    "methods": "Class-bound methods defined in component",
    "commit_count": "Total historical commits touching this component",
    "recent_commit_count": "Recent commits touching this component in current window",
    "churn": "Total lines added and deleted historically",
    "recent_churn": "Lines added and deleted in recent window",
    "change_frequency": "Ratio of repository commits affecting this file",
    "number_of_authors": "Distinct authors modifying this component",
    "time_since_last_change": "Days elapsed since the most recent commit on this file",
    "dependency_count": "Number of other repository modules this component imports (out-degree)",
    "dependent_count": "Number of modules importing this component (in-degree)",
    "degree": "Total graph connectivity (in-degree + out-degree)",
    "centrality": "Degree centrality within repository dependency graph",
    "in_degree": "Incoming dependency edges",
    "out_degree": "Outgoing dependency edges"
}
