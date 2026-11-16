"""
Tests for SnapshotReconstructor: Safe, read-only extraction of historical code metrics,
dependency graphs, and git evolution states without working tree checkout.
"""
import subprocess
import pytest
from pathlib import Path

from analysis.snapshot_reconstructor import SnapshotReconstructor, HistoricalSnapshot
from analysis.code_analyzer import CodeAnalyzer
from analysis.dependency_analyzer import DependencyAnalyzer
from ingestion.git_loader import GitLoader


@pytest.fixture
def temp_git_repo(tmp_path):
    """Creates a temporary Git repository with controlled evolutionary changes across 4 commits."""
    repo_dir = tmp_path / "test_repo"
    repo_dir.mkdir()

    # git init
    subprocess.run(["git", "init"], cwd=str(repo_dir), check=True, capture_output=True)
    subprocess.run(["git", "config", "user.name", "Test User"], cwd=str(repo_dir), check=True, capture_output=True)
    subprocess.run(["git", "config", "user.email", "test@test.com"], cwd=str(repo_dir), check=True, capture_output=True)

    # Commit 1: Add a.py (simple)
    a_v1 = "def func_a():\n    return 1\n"
    (repo_dir / "a.py").write_text(a_v1, encoding="utf-8")
    subprocess.run(["git", "add", "a.py"], cwd=str(repo_dir), check=True, capture_output=True)
    subprocess.run(["git", "commit", "-m", "Commit 1: add a.py"], cwd=str(repo_dir), check=True, capture_output=True)
    c1 = subprocess.run(["git", "rev-parse", "HEAD"], cwd=str(repo_dir), capture_output=True, text=True).stdout.strip()

    # Commit 2: Add b.py which imports a.py, and modify a.py
    a_v2 = "def func_a():\n    if True:\n        return 1\n    return 0\n"
    b_v1 = "import a\nclass B:\n    def method_b(self):\n        return a.func_a()\n"
    (repo_dir / "a.py").write_text(a_v2, encoding="utf-8")
    (repo_dir / "b.py").write_text(b_v1, encoding="utf-8")
    subprocess.run(["git", "add", "a.py", "b.py"], cwd=str(repo_dir), check=True, capture_output=True)
    subprocess.run(["git", "commit", "-m", "Commit 2: add b.py, update a.py"], cwd=str(repo_dir), check=True, capture_output=True)
    c2 = subprocess.run(["git", "rev-parse", "HEAD"], cwd=str(repo_dir), capture_output=True, text=True).stdout.strip()

    # Commit 3: Add c.py, delete b.py
    c_v1 = "import a\ndef func_c():\n    return a.func_a()\n"
    (repo_dir / "c.py").write_text(c_v1, encoding="utf-8")
    (repo_dir / "b.py").unlink()
    subprocess.run(["git", "add", "-A"], cwd=str(repo_dir), check=True, capture_output=True)
    subprocess.run(["git", "commit", "-m", "Commit 3: add c.py, delete b.py"], cwd=str(repo_dir), check=True, capture_output=True)
    c3 = subprocess.run(["git", "rev-parse", "HEAD"], cwd=str(repo_dir), capture_output=True, text=True).stdout.strip()

    # Commit 4: Rename c.py to helper.py, fix bug
    helper_code = "import a\ndef func_c():\n    # fix: resolve null return\n    return a.func_a() or 0\n"
    (repo_dir / "c.py").unlink()
    (repo_dir / "helper.py").write_text(helper_code, encoding="utf-8")
    subprocess.run(["git", "add", "-A"], cwd=str(repo_dir), check=True, capture_output=True)
    subprocess.run(["git", "commit", "-m", "fix: rename c to helper and resolve null bug"], cwd=str(repo_dir), check=True, capture_output=True)
    c4 = subprocess.run(["git", "rev-parse", "HEAD"], cwd=str(repo_dir), capture_output=True, text=True).stdout.strip()

    return {
        "repo_dir": repo_dir,
        "commits": [c1, c2, c3, c4]
    }


def test_historical_source_reconstruction_without_checkout(temp_git_repo):
    """Verifies that file contents are accurately read from past commits without changing HEAD."""
    repo_dir = temp_git_repo["repo_dir"]
    commits = temp_git_repo["commits"]
    reconstructor = SnapshotReconstructor(repo_dir)

    head_before = subprocess.run(["git", "rev-parse", "HEAD"], cwd=str(repo_dir), capture_output=True, text=True).stdout.strip()

    # Commit 1 has only a.py
    files_c1 = reconstructor.get_python_files_at_commit(commits[0])
    assert files_c1 == ["a.py"]
    content_a_c1 = reconstructor.get_file_content_at_commit(commits[0], "a.py")
    assert "return 1" in content_a_c1
    assert "if True:" not in content_a_c1

    # Commit 2 has a.py and b.py
    files_c2 = reconstructor.get_python_files_at_commit(commits[1])
    assert set(files_c2) == {"a.py", "b.py"}
    content_a_c2 = reconstructor.get_file_content_at_commit(commits[1], "a.py")
    assert "if True:" in content_a_c2

    # Commit 3: b.py was deleted, c.py added
    files_c3 = reconstructor.get_python_files_at_commit(commits[2])
    assert set(files_c3) == {"a.py", "c.py"}
    assert reconstructor.get_file_content_at_commit(commits[2], "b.py") is None

    # Commit 4: c.py was renamed to helper.py
    files_c4 = reconstructor.get_python_files_at_commit(commits[3])
    assert set(files_c4) == {"a.py", "helper.py"}

    # Verify working tree was NEVER checked out to older commits
    head_after = subprocess.run(["git", "rev-parse", "HEAD"], cwd=str(repo_dir), capture_output=True, text=True).stdout.strip()
    assert head_before == head_after == commits[3]


def test_historical_ast_metrics_extraction(temp_git_repo):
    """Verifies AST complexity changes across commits are accurately measured at past states."""
    repo_dir = temp_git_repo["repo_dir"]
    commits = temp_git_repo["commits"]
    reconstructor = SnapshotReconstructor(repo_dir)

    # At Commit 1, a.py had 1 function, complexity 1
    metrics_c1 = reconstructor.extract_code_metrics_at_commit(commits[0])
    assert "a.py" in metrics_c1
    assert metrics_c1["a.py"].complexity == 1
    assert metrics_c1["a.py"].functions == 1
    assert metrics_c1["a.py"].classes == 0

    # At Commit 2, a.py added an 'if True' branch -> complexity should be 2
    metrics_c2 = reconstructor.extract_code_metrics_at_commit(commits[1])
    assert metrics_c2["a.py"].complexity == 2
    assert "b.py" in metrics_c2
    assert metrics_c2["b.py"].classes == 1
    assert metrics_c2["b.py"].methods == 1


def test_historical_dependency_graph_reconstruction(temp_git_repo):
    """Verifies dependency graph topology reflects historical imports rather than HEAD state."""
    repo_dir = temp_git_repo["repo_dir"]
    commits = temp_git_repo["commits"]
    reconstructor = SnapshotReconstructor(repo_dir)

    # Commit 1: only a.py, 0 dependencies
    g1 = reconstructor.build_dependency_graph_at_commit(commits[0])
    sm1 = g1.get_metrics_for_all()
    assert "a.py" in sm1
    assert sm1["a.py"].degree == 0

    # Commit 2: b.py imports a.py (b -> a)
    g2 = reconstructor.build_dependency_graph_at_commit(commits[1])
    sm2 = g2.get_metrics_for_all()
    assert sm2["b.py"].out_degree == 1  # depends on a
    assert sm2["a.py"].in_degree == 1   # imported by b

    # Commit 3: b.py deleted, c.py imports a.py
    g3 = reconstructor.build_dependency_graph_at_commit(commits[2])
    sm3 = g3.get_metrics_for_all()
    assert "b.py" not in sm3
    assert "c.py" in sm3
    assert sm3["c.py"].out_degree == 1
    assert sm3["a.py"].in_degree == 1


def test_full_snapshot_extraction(temp_git_repo):
    """Verifies extract_full_snapshot bundles code, dependency, and git metrics at timestamp."""
    repo_dir = temp_git_repo["repo_dir"]
    loader = GitLoader(str(repo_dir))
    commits = loader.get_commit_history()

    reconstructor = SnapshotReconstructor(repo_dir)
    snapshot = reconstructor.extract_full_snapshot(
        commit_hash=commits[1].commit_hash,
        timestamp=commits[1].timestamp,
        commits_up_to=commits[:2]
    )

    assert snapshot.commit_hash == commits[1].commit_hash
    assert set(snapshot.python_files) == {"a.py", "b.py"}
    assert "a.py" in snapshot.code_metrics
    assert "b.py" in snapshot.code_metrics
    assert "a.py" in snapshot.git_metrics
    assert snapshot.git_metrics["a.py"].commit_count == 2
    assert snapshot.git_metrics["b.py"].commit_count == 1
