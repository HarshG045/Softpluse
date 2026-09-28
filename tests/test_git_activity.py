"""
Unit and integration tests for the Git Activity & Change Impact Analysis module.
"""
from pathlib import Path
import pytest

from analysis.change_analyzer import ChangeImpactAnalyzer
from analysis.code_analyzer import CodeAnalyzer
from analysis.dependency_analyzer import DependencyAnalyzer
from data.demo.demo_repo_builder import create_demo_repository
from dataset.builder import DatasetBuilder
from ingestion.git_loader import FileDiffItem, GitLoader
from ml.predict import RiskPredictor
from ml.train import ClassifierAlgorithm, ModelTrainer


@pytest.fixture
def initialized_repo_and_model():
    """Builds demo repository and trains a model bundle with dependency graph and profiles."""
    repo_path = create_demo_repository(force_recreate=True)
    git_loader = GitLoader(str(repo_path))
    commits = git_loader.get_commit_history()

    code_analyzer = CodeAnalyzer(repo_path)
    code_map = code_analyzer.analyze_repository()

    dep_analyzer = DependencyAnalyzer(repo_path)
    dep_graph = dep_analyzer.build_graph(code_map)

    builder = DatasetBuilder(commits, code_map, dep_graph)
    dataset = builder.build_dataset()

    trainer = ModelTrainer(random_state=42)
    model_bundle = trainer.train_model(
        dataset.split_result.X_train,
        dataset.split_result.y_train,
        algorithm=ClassifierAlgorithm.RANDOM_FOREST
    )

    predictor = RiskPredictor()
    pred_res = predictor.predict(model_bundle, dataset.current_features_df)

    change_analyzer = ChangeImpactAnalyzer(model_bundle, dep_graph, pred_res.profiles)

    return {
        "repo_path": repo_path,
        "git_loader": git_loader,
        "commits": commits,
        "dep_graph": dep_graph,
        "model_bundle": model_bundle,
        "pred_res": pred_res,
        "change_analyzer": change_analyzer
    }


def test_git_loader_commits_and_details(initialized_repo_and_model):
    """Verifies commit history, commit detail extraction, diffs, and parents."""
    git_loader: GitLoader = initialized_repo_and_model["git_loader"]
    commits = git_loader.get_commit_history()

    assert len(commits) >= 6, "Expected at least 6 commits in demo repository"
    
    # Test commit detail
    latest = commits[-1]
    detail = git_loader.get_commit_detail(latest.commit_hash)
    assert detail is not None
    assert detail.commit_hash == latest.commit_hash
    assert detail.short_hash == latest.commit_hash[:7]
    assert detail.author_name != ""
    assert len(detail.files_changed) > 0


def test_git_loader_contributors(initialized_repo_and_model):
    """Verifies contributor statistics and 'who changed what' mappings."""
    git_loader: GitLoader = initialized_repo_and_model["git_loader"]
    contributors = git_loader.get_contributor_stats()

    assert len(contributors) >= 2, "Expected multiple contributors in demo repo"
    for c in contributors:
        assert c.author_name != ""
        assert c.commit_count > 0
        assert len(c.components_touched) > 0
        assert c.insertions >= 0
        assert c.deletions >= 0


def test_git_loader_component_history(initialized_repo_and_model):
    """Verifies component-specific Git history lookup."""
    git_loader: GitLoader = initialized_repo_and_model["git_loader"]
    commits = git_loader.get_commit_history()
    first_file = commits[0].files_changed[0] if commits[0].files_changed else "services/payment_service.py"

    history = git_loader.get_component_git_history(first_file)
    assert len(history) >= 1
    assert any(first_file.replace("\\", "/").lower() in f.replace("\\", "/").lower() for c in history for f in c.files_changed)


def test_git_loader_branches(initialized_repo_and_model):
    """Verifies branch listing and branch comparison."""
    git_loader: GitLoader = initialized_repo_and_model["git_loader"]
    active_branch, branches = git_loader.get_branches()
    assert active_branch != ""
    assert len(branches) >= 1

    comparison = git_loader.compare_branches(active_branch, active_branch)
    assert comparison.ahead_commits == 0


def test_change_impact_analyzer_file_diff(initialized_repo_and_model):
    """Verifies that ChangeImpactAnalyzer recalculates structural metrics and ML risk."""
    change_analyzer: ChangeImpactAnalyzer = initialized_repo_and_model["change_analyzer"]
    pred_res = initialized_repo_and_model["pred_res"]

    target_profile = pred_res.profiles[0]
    target_path = target_profile.filepath

    # Construct a diff that adds complexity and dependencies
    diff_text = f"""
--- a/{target_path}
+++ b/{target_path}
@@ -10,4 +10,12 @@
+import external_heavy_lib
+def heavy_processing_routine():
+    if check_condition():
+        for i in range(100):
+            while is_running():
+                execute_step()
-simple_step()
"""
    impact = change_analyzer.analyze_file_diff(
        filepath=target_path,
        diff_text=diff_text,
        change_type="MODIFIED",
        insertions=10,
        deletions=1
    )

    assert impact is not None
    assert impact.filepath == target_profile.filepath
    assert impact.current_risk_pct == target_profile.risk_percentage
    assert isinstance(impact.proposed_risk_pct, int)
    assert len(impact.changed_signals) >= 1
    assert impact.explanation != ""


def test_change_impact_analyzer_commit_summary(initialized_repo_and_model):
    """Verifies that analyze_commit_or_preview aggregates multiple file diffs correctly."""
    change_analyzer: ChangeImpactAnalyzer = initialized_repo_and_model["change_analyzer"]
    pred_res = initialized_repo_and_model["pred_res"]
    target_path = pred_res.profiles[0].filepath

    items = [
        FileDiffItem(
            filepath=target_path,
            change_type="MODIFIED",
            insertions=15,
            deletions=3,
            patch=f"+if test:\n+    pass\n-old_pass()"
        )
    ]

    summary = change_analyzer.analyze_commit_or_preview(items)
    assert summary.total_files_affected == 1
    assert summary.total_components_affected >= 1
    assert len(summary.component_impacts) >= 1
