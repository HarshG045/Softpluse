"""
Tests for Pull Request and Branch Risk Analysis.
"""
import pytest
from pathlib import Path
from pull_requests.pr_diff_parser import PRDiffParser
from pull_requests.pr_risk_analyzer import PullRequestRiskAnalyzer, PRFileRiskDetail, PullRequestRiskReport
from ingestion.git_loader import FileDiffItem
from analysis.dependency_analyzer import DependencyGraph, ComponentStructuralMetrics
from ml.predict import ComponentRiskProfile
from ml.train import TrainedModelBundle
import networkx as nx
import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from features.feature_schema import FeatureSetType, STATIC_FEATURES
from ml.train import ClassifierAlgorithm, TrainedModelBundle


SAMPLE_DIFF = """diff --git a/src/auth.py b/src/auth.py
index 1234567..89abcdef 100644
--- a/src/auth.py
+++ b/src/auth.py
@@ -10,3 +10,6 @@ def authenticate(user, password):
+    if not user:
+        raise ValueError("empty user")
+    return True
diff --git a/src/new_helper.py b/src/new_helper.py
new file mode 100644
index 0000000..1234567
--- /dev/null
+++ b/src/new_helper.py
@@ -0,0 +1,5 @@
+def helper():
+    pass
diff --git a/legacy.py b/legacy.py
deleted file mode 100644
index 1234567..0000000
--- a/legacy.py
+++ /dev/null
@@ -1,5 +0,0 @@
-def old():
-    pass
"""


def test_pr_diff_parser_extracts_items():
    items = PRDiffParser.parse_unified_diff(SAMPLE_DIFF)
    assert len(items) == 3

    auth_item = next(i for i in items if "auth.py" in i.filepath)
    assert auth_item.change_type == "MODIFIED"
    assert auth_item.insertions >= 3

    helper_item = next(i for i in items if "new_helper.py" in i.filepath)
    assert helper_item.change_type == "ADDED"
    assert helper_item.insertions >= 2

    legacy_item = next(i for i in items if "legacy.py" in i.filepath)
    assert legacy_item.change_type == "DELETED"
    assert legacy_item.deletions >= 2


def test_pr_diff_parser_empty_input():
    assert PRDiffParser.parse_unified_diff("") == []
    assert PRDiffParser.parse_unified_diff("   ") == []


def test_pr_risk_analyzer_diff_items_generation(tmp_path):
    # Setup minimal mock model bundle and dependency graph
    feature_names = STATIC_FEATURES
    pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("classifier", DummyClassifier(strategy="constant", constant=0))
    ])
    pipe.fit(np.zeros((5, len(feature_names))), np.zeros(5))

    bundle = TrainedModelBundle(
        model_name="dummy_test",
        algorithm=ClassifierAlgorithm.RANDOM_FOREST,
        feature_set_type=FeatureSetType.STATIC,
        feature_names=feature_names,
        pipeline=pipe,
        training_samples=5,
        positive_samples=0,
        negative_samples=5,
        is_single_class=True,
        single_class_label=0
    )

    # Setup dependency graph: app.py depends on auth.py (predecessor = app.py)
    g = nx.DiGraph()
    g.add_edge("src/app.py", "src/auth.py")
    dep_graph = DependencyGraph(graph=g)

    # Setup component profiles
    profiles = [
        ComponentRiskProfile(
            filepath="src/auth.py",
            risk_probability=0.45,
            risk_percentage=45,
            risk_category="MEDIUM",
            loc=80,
            complexity=8,
            functions=3,
            classes=1,
            imports=2,
            methods=2,
            commit_count=5,
            recent_commit_count=2,
            churn=50,
            recent_churn=20,
            change_frequency=0.5,
            number_of_authors=2,
            time_since_last_change=1.0,
            dependency_count=0,
            dependent_count=1,
            degree=1,
            centrality=0.2
        )
    ]

    analyzer = PullRequestRiskAnalyzer(
        repo_path=tmp_path,
        model_bundle=bundle,
        dep_graph=dep_graph,
        component_profiles=profiles
    )

    diff_items = PRDiffParser.parse_unified_diff(SAMPLE_DIFF)
    report = analyzer._generate_report(
        title="Test PR",
        source_identifier="PR #1",
        base_ref="main",
        head_ref="feat/auth",
        diff_items=diff_items
    )

    assert isinstance(report, PullRequestRiskReport)
    assert report.total_files_changed == 3
    assert report.total_churn > 0
    assert report.overall_risk_level in ["LOW", "MEDIUM", "HIGH"]
    assert report.high_risk_files_count + report.medium_risk_files_count + report.low_risk_files_count == 3
    # Verify downstream dependent was captured for auth.py (src/app.py depends on src/auth.py)
    assert report.total_downstream_dependents_impacted >= 1
    auth_detail = next(f for f in report.files_detail if "auth.py" in f.filepath)
    assert "src/app.py" in auth_detail.downstream_dependents_affected
