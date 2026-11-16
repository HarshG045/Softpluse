"""
Comprehensive tests for temporal integrity, leakage prevention, and chronological evaluation.
"""
import subprocess
import pytest
from datetime import datetime, timezone, timedelta
from pathlib import Path
import pandas as pd
import numpy as np

from analysis.code_analyzer import CodeAnalyzer
from analysis.dependency_analyzer import DependencyAnalyzer
from analysis.git_analyzer import GitAnalyzer
from dataset.builder import DatasetBuilder
from dataset.temporal_split import (
    TemporalSplitter,
    TemporalLeakageError,
    InsufficientDataError,
    SplitResult
)
from dataset.labeling import LabelGenerator, TargetLabelType, LabelMethod
from ingestion.git_loader import GitLoader, CommitRecord
from ml.train import ModelTrainer, ClassifierAlgorithm
from ml.evaluate import ModelEvaluator
from features.feature_schema import (
    STATIC_FEATURES,
    STATIC_PLUS_EVOLUTION_FEATURES,
    FULL_FEATURES,
    FeatureSetType
)


@pytest.fixture
def evolution_repo(tmp_path):
    """
    Creates a controlled repository with 8 sequential commits covering:
    - Initial feature addition
    - Churn updates
    - Bug fixes in future periods
    """
    repo_dir = tmp_path / "evolution_repo"
    repo_dir.mkdir()

    subprocess.run(["git", "init"], cwd=str(repo_dir), check=True, capture_output=True)
    subprocess.run(["git", "config", "user.name", "Researcher"], cwd=str(repo_dir), check=True, capture_output=True)
    subprocess.run(["git", "config", "user.email", "researcher@lab.edu"], cwd=str(repo_dir), check=True, capture_output=True)

    # File definitions
    files = {
        "auth.py": "def login():\n    return True\n",
        "payment.py": "def pay():\n    return False\n",
        "api.py": "import auth\nimport payment\ndef route():\n    return 200\n"
    }

    def commit_with_date(msg, hour_offset):
        date_str = f"2025-01-01 {10 + hour_offset:02d}:00:00 +0000"
        env = {
            **subprocess.os.environ,
            "GIT_AUTHOR_DATE": date_str,
            "GIT_COMMITTER_DATE": date_str
        }
        subprocess.run(["git", "commit", "-m", msg], cwd=str(repo_dir), env=env, check=True, capture_output=True)

    # C1: initial
    for name, code in files.items():
        (repo_dir / name).write_text(code, encoding="utf-8")
    subprocess.run(["git", "add", "-A"], cwd=str(repo_dir), check=True, capture_output=True)
    commit_with_date("feat: C1 initial components", 1)

    # C2: expand auth
    (repo_dir / "auth.py").write_text("def login():\n    if True:\n        return True\n    return False\n", encoding="utf-8")
    subprocess.run(["git", "add", "auth.py"], cwd=str(repo_dir), check=True, capture_output=True)
    commit_with_date("feat: C2 expand auth logic", 2)

    # C3: modify payment
    (repo_dir / "payment.py").write_text("def pay():\n    for i in range(3):\n        pass\n    return True\n", encoding="utf-8")
    subprocess.run(["git", "add", "payment.py"], cwd=str(repo_dir), check=True, capture_output=True)
    commit_with_date("feat: C3 update payment logic", 3)

    # C4: fix bug in auth (Train future horizon event)
    (repo_dir / "auth.py").write_text("def login():\n    # fix: security bypass bug\n    return True\n", encoding="utf-8")
    subprocess.run(["git", "add", "auth.py"], cwd=str(repo_dir), check=True, capture_output=True)
    commit_with_date("fix: security bypass bug in auth", 4)

    # C5: update api
    (repo_dir / "api.py").write_text("import auth\nimport payment\ndef route():\n    auth.login()\n    return 200\n", encoding="utf-8")
    subprocess.run(["git", "add", "api.py"], cwd=str(repo_dir), check=True, capture_output=True)
    commit_with_date("feat: C5 update api routing", 5)

    # C6: modify payment
    (repo_dir / "payment.py").write_text("def pay():\n    # add stripe integration\n    return True\n", encoding="utf-8")
    subprocess.run(["git", "add", "payment.py"], cwd=str(repo_dir), check=True, capture_output=True)
    commit_with_date("feat: C6 add stripe payment logic", 6)

    # C7: fix bug in payment (Test future horizon event)
    (repo_dir / "payment.py").write_text("def pay():\n    # fix: crash on invalid currency\n    return True\n", encoding="utf-8")
    subprocess.run(["git", "add", "payment.py"], cwd=str(repo_dir), check=True, capture_output=True)
    commit_with_date("fix: resolve crash on invalid currency in payment", 7)

    # C8: chore update
    (repo_dir / "api.py").write_text("import auth\nimport payment\ndef route():\n    # v2\n    return 200\n", encoding="utf-8")
    subprocess.run(["git", "add", "api.py"], cwd=str(repo_dir), check=True, capture_output=True)
    commit_with_date("chore: C8 bump api version", 8)

    return repo_dir


def test_independent_train_and_test_label_windows(evolution_repo):
    """Verifies training labels and testing labels derive from separate non-overlapping intervals."""
    loader = GitLoader(str(evolution_repo))
    commits = loader.get_commit_history()
    code_map = CodeAnalyzer(evolution_repo).analyze_repository()
    dep_graph = DependencyAnalyzer(evolution_repo).build_graph(code_map)

    builder = DatasetBuilder(commits, code_map, dep_graph, repo_path=evolution_repo)
    dataset = builder.build_dataset()
    split = dataset.split_result

    # 1. Assert boundaries exist
    assert split.train_observation_time is not None
    assert split.train_label_window is not None
    assert split.test_observation_time is not None
    assert split.test_label_window is not None

    train_lbl_start, train_lbl_end = split.train_label_window
    test_lbl_start, test_lbl_end = split.test_label_window

    # 2. Strict non-overlap: train label window ends before or at test label window start
    assert train_lbl_end <= test_lbl_start, (
        f"Train label window end ({train_lbl_end}) must not overlap with test label window start ({test_lbl_start})"
    )

    # 3. Test observation is at or after train label end
    assert split.test_observation_time >= train_lbl_end, (
        f"Test observation ({split.test_observation_time}) must be at or after train label window end ({train_lbl_end})"
    )

    # 4. Leakage-free flag is True
    assert split.is_leakage_free is True
    assert len(split.temporal_validation_errors) == 0


def test_temporal_leakage_detector_catches_violations():
    """Verifies that TemporalSplitter catches intentional future information leakage."""
    now = datetime.now(timezone.utc)
    t_train_obs = now - timedelta(days=50)
    t_train_lbl_start = now - timedelta(days=50)
    t_train_lbl_end = now - timedelta(days=20)

    # Intentional violation 1: test observation BEFORE train label end
    t_test_obs_leaked = now - timedelta(days=30)  # Leaked! 10 days before train label window ended
    t_test_lbl_start = now - timedelta(days=20)
    t_test_lbl_end = now

    with pytest.raises(TemporalLeakageError, match="temporal leakage"):
        TemporalSplitter.validate_temporal_boundaries(
            train_obs_time=t_train_obs,
            train_label_start=t_train_lbl_start,
            train_label_end=t_train_lbl_end,
            test_obs_time=t_test_obs_leaked,
            test_label_start=t_test_lbl_start,
            test_label_end=t_test_lbl_end,
            strict=True
        )

    # Intentional violation 2: overlapping train and test label windows
    t_test_obs_valid = now - timedelta(days=20)
    t_test_lbl_start_overlapping = now - timedelta(days=25)  # Overlaps with train_lbl_end (days=20)
    with pytest.raises(TemporalLeakageError, match="overlaps"):
        TemporalSplitter.validate_temporal_boundaries(
            train_obs_time=t_train_obs,
            train_label_start=t_train_lbl_start,
            train_label_end=t_train_lbl_end,
            test_obs_time=t_test_obs_valid,
            test_label_start=t_test_lbl_start_overlapping,
            test_label_end=t_test_lbl_end,
            strict=True
        )


def test_no_target_in_features_assertion():
    """Verifies that target/ground-truth columns cannot be accidentally passed into features."""
    df_clean = pd.DataFrame({"loc": [10, 20], "complexity": [2, 3]})
    TemporalSplitter.validate_no_target_in_features(df_clean)

    df_contaminated = pd.DataFrame({"loc": [10, 20], "target": [0, 1]})
    with pytest.raises(TemporalLeakageError, match="leaked into feature matrix"):
        TemporalSplitter.validate_no_target_in_features(df_contaminated)


def test_git_metrics_restricted_to_observation_timestamp(evolution_repo):
    """Verifies that GitAnalyzer computes metrics strictly using commits up to observation timestamp."""
    loader = GitLoader(str(evolution_repo))
    all_commits = loader.get_commit_history()
    assert len(all_commits) == 8

    # Cutoff at commit 3 (index 2)
    cutoff_time = all_commits[2].timestamp
    analyzer = GitAnalyzer(all_commits)
    metrics_at_c3 = analyzer.analyze_components(as_of_time=cutoff_time)

    # At C3, auth.py had 2 commits (C1, C2), payment.py had 2 commits (C1, C3), api.py had 1 commit (C1)
    assert metrics_at_c3["auth.py"].commit_count == 2
    assert metrics_at_c3["payment.py"].commit_count == 2
    assert metrics_at_c3["api.py"].commit_count == 1

    # At HEAD, commit counts are higher
    metrics_at_head = analyzer.analyze_components()
    assert metrics_at_head["auth.py"].commit_count > 2
    assert metrics_at_head["payment.py"].commit_count > 2


def test_model_configurations_use_identical_evaluation_samples(evolution_repo):
    """
    Verifies that Model A, Model B, and Model C in the comparative experiment
    evaluate on the exact same test instances and target labels.
    """
    loader = GitLoader(str(evolution_repo))
    commits = loader.get_commit_history()
    code_map = CodeAnalyzer(evolution_repo).analyze_repository()
    dep_graph = DependencyAnalyzer(evolution_repo).build_graph(code_map)

    builder = DatasetBuilder(commits, code_map, dep_graph, repo_path=evolution_repo)
    dataset = builder.build_dataset()

    evaluator = ModelEvaluator()
    comp_report = evaluator.run_comparative_experiment(dataset.split_result)

    rep_a = comp_report.static_report
    rep_b = comp_report.evolution_report
    rep_c = comp_report.full_report

    # Test sample counts must be strictly identical across all 3 models
    assert rep_a.test_samples == rep_b.test_samples == rep_c.test_samples == len(dataset.split_result.y_test)

    # Class distributions must be identical
    assert rep_a.class_distribution == rep_b.class_distribution == rep_c.class_distribution


def test_single_class_test_dataset_handled_transparently():
    """Verifies that single-class test datasets do not report fabricated ROC-AUC scores."""
    X_train = pd.DataFrame({
        "loc": [10.0, 20.0, 30.0, 40.0],
        "complexity": [1.0, 2.0, 3.0, 4.0],
        "functions": [1.0, 2.0, 2.0, 3.0],
        "classes": [1.0, 1.0, 1.0, 1.0],
        "imports": [1.0, 2.0, 1.0, 2.0],
        "methods": [1.0, 2.0, 1.0, 2.0]
    })
    y_train = pd.Series([0, 1, 0, 1])

    trainer = ModelTrainer()
    bundle = trainer.train_model(X_train, y_train, feature_set_type=FeatureSetType.STATIC)

    # Test set where all labels are 0 (single class)
    X_test = pd.DataFrame({
        "loc": [12.0, 15.0],
        "complexity": [1.0, 2.0],
        "functions": [1.0, 1.0],
        "classes": [1.0, 1.0],
        "imports": [1.0, 1.0],
        "methods": [1.0, 1.0]
    })
    y_test_single = pd.Series([0, 0])

    report = ModelEvaluator.evaluate_bundle(bundle, X_test, y_test_single)

    assert report.is_single_class_test is True
    assert report.roc_auc_valid is False
    # Report dictionary shows N/A for ROC-AUC
    dict_rep = report.to_dict()
    assert dict_rep["ROC-AUC"] == "N/A"
    assert dict_rep["Single Class Test"] is True


def test_insufficient_history_small_repository():
    """Verifies that repositories with minimal commits (< 4) handle splitting safely."""
    # Mock commit history with only 2 commits
    dt1 = datetime(2025, 1, 1, tzinfo=timezone.utc)
    dt2 = datetime(2025, 1, 2, tzinfo=timezone.utc)
    commits = [
        CommitRecord("h1", "A", "a@a.com", dt1, "C1", ["main.py"], 10, 0),
        CommitRecord("h2", "A", "a@a.com", dt2, "C2", ["main.py"], 5, 2)
    ]
    code_map = {"main.py": CodeAnalyzer.analyze_source_content("main.py", "print('hi')\n")}
    dep_graph = DependencyAnalyzer.build_graph_from_metrics(code_map)

    builder = DatasetBuilder(commits, code_map, dep_graph)
    dataset = builder.build_dataset()

    assert dataset.total_components_count == 1
    assert dataset.split_result.train_samples_count >= 1
    assert dataset.label_summary is not None
