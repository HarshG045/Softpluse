"""
SnapshotReconstructor: Extracts historical source files from Git commits and
reconstructs code metrics, dependency graphs, and git metrics at any past point
in time — without checking out commits in the working directory.

Uses `git ls-tree` and `git show` (read-only operations) to safely extract
historical state. This prevents temporal leakage by ensuring that feature
extraction at observation timestamp T uses only information available at T.
"""
import logging
import re
import subprocess
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from ingestion.git_loader import CommitRecord
from .code_analyzer import CodeAnalyzer, CodeMetrics
from .git_analyzer import GitAnalyzer, ComponentGitMetrics
from .dependency_analyzer import DependencyAnalyzer, DependencyGraph

logger = logging.getLogger(__name__)

# Safety: reject commit hashes containing shell metacharacters
_SAFE_HASH_RE = re.compile(r"^[0-9a-fA-F]{4,40}$")


def _validate_commit_hash(commit_hash: str) -> bool:
    """Returns True only if commit_hash looks like a safe hex SHA."""
    return bool(_SAFE_HASH_RE.match(commit_hash))


@dataclass
class HistoricalSnapshot:
    """Complete feature snapshot reconstructed at a specific historical commit.

    Every metric in this snapshot represents information that was available
    at or before the specified commit. No future data is included.
    """
    commit_hash: str
    timestamp: datetime
    code_metrics: Dict[str, CodeMetrics]
    git_metrics: Dict[str, ComponentGitMetrics]
    dep_graph: DependencyGraph
    python_files: List[str]


class SnapshotReconstructor:
    """Reconstructs repository state at historical commits using read-only Git operations.

    This class never uses `git checkout` or modifies the working directory.
    All extraction is done via `git ls-tree` (file listing) and `git show`
    (file content retrieval), which are safe read-only operations.
    """

    def __init__(self, repo_path: Path):
        self.repo_path = repo_path.resolve()
        if not (self.repo_path / ".git").exists():
            raise ValueError(f"Not a Git repository: {self.repo_path}")

    def get_file_list_at_commit(self, commit_hash: str) -> List[str]:
        """Returns the list of all files that existed at a given commit.

        Uses `git ls-tree -r --name-only <commit>` which is a read-only operation
        that does not affect the working directory.
        """
        if not _validate_commit_hash(commit_hash):
            logger.warning(f"Invalid commit hash rejected: {commit_hash!r}")
            return []

        try:
            result = subprocess.run(
                ["git", "ls-tree", "-r", "--name-only", commit_hash],
                cwd=str(self.repo_path),
                capture_output=True,
                text=True,
                check=True,
                timeout=30
            )
            files = [
                line.strip()
                for line in result.stdout.strip().split("\n")
                if line.strip()
            ]
            return files
        except subprocess.CalledProcessError as e:
            logger.warning(f"git ls-tree failed for {commit_hash}: {e.stderr}")
            return []
        except subprocess.TimeoutExpired:
            logger.warning(f"git ls-tree timed out for {commit_hash}")
            return []

    def get_python_files_at_commit(self, commit_hash: str) -> List[str]:
        """Returns only Python files (.py) that existed at a given commit.

        Excludes common non-source directories (venv, __pycache__, etc.)
        following the same conventions as CodeAnalyzer.analyze_repository.
        """
        all_files = self.get_file_list_at_commit(commit_hash)
        excluded_dirs = {
            ".git", "venv", ".venv", "env", "node_modules",
            "__pycache__", "build", "dist"
        }

        python_files = []
        for f in all_files:
            if not f.endswith(".py"):
                continue
            parts = Path(f).parts
            if any(p.startswith(".") or p in excluded_dirs for p in parts):
                continue
            python_files.append(f.replace("\\", "/"))

        return python_files

    def get_file_content_at_commit(
        self, commit_hash: str, filepath: str
    ) -> Optional[str]:
        """Retrieves the content of a specific file at a given commit.

        Uses `git show <commit>:<filepath>` which is a read-only operation
        that does not affect the working directory.

        Returns None if the file doesn't exist at that commit or on error.
        """
        if not _validate_commit_hash(commit_hash):
            logger.warning(f"Invalid commit hash rejected: {commit_hash!r}")
            return None

        try:
            result = subprocess.run(
                ["git", "show", f"{commit_hash}:{filepath}"],
                cwd=str(self.repo_path),
                capture_output=True,
                text=True,
                timeout=15
            )
            if result.returncode != 0:
                return None
            return result.stdout
        except subprocess.TimeoutExpired:
            logger.warning(f"git show timed out for {commit_hash}:{filepath}")
            return None
        except Exception as e:
            logger.debug(f"Failed to get content for {filepath} at {commit_hash}: {e}")
            return None

    def extract_code_metrics_at_commit(
        self, commit_hash: str
    ) -> Dict[str, CodeMetrics]:
        """Extracts AST code metrics for all Python files at a given commit.

        Uses git show to retrieve file contents and CodeAnalyzer.analyze_source_content
        to parse them. Never reads from the current working directory.
        """
        python_files = self.get_python_files_at_commit(commit_hash)
        metrics_map: Dict[str, CodeMetrics] = {}

        for fp in python_files:
            content = self.get_file_content_at_commit(commit_hash, fp)
            if content is not None:
                metrics = CodeAnalyzer.analyze_source_content(fp, content)
                metrics_map[fp] = metrics
            else:
                # File listed but content unavailable — record with zero metrics
                metrics_map[fp] = CodeMetrics(filepath=fp)

        return metrics_map

    def build_dependency_graph_at_commit(
        self, commit_hash: str,
        code_metrics_map: Optional[Dict[str, CodeMetrics]] = None
    ) -> DependencyGraph:
        """Reconstructs the dependency graph at a specific historical commit.

        If code_metrics_map is not provided, it is extracted from the commit.
        Uses DependencyAnalyzer.build_graph_from_metrics to construct the graph
        from the historical metrics without requiring the current working directory.
        """
        if code_metrics_map is None:
            code_metrics_map = self.extract_code_metrics_at_commit(commit_hash)

        return DependencyAnalyzer.build_graph_from_metrics(code_metrics_map)

    def extract_full_snapshot(
        self,
        commit_hash: str,
        timestamp: datetime,
        commits_up_to: List[CommitRecord],
        target_filepaths: Optional[List[str]] = None
    ) -> HistoricalSnapshot:
        """Reconstructs a complete historical snapshot at a specific commit.

        This is the primary entry point for temporal feature extraction.
        All returned metrics represent information available at or before
        the specified commit. No future data is included.

        Args:
            commit_hash: The Git commit SHA to reconstruct at.
            timestamp: The datetime of the commit (for reference/validation).
            commits_up_to: All CommitRecord objects up to and including this commit
                          (oldest to newest). Used for git evolution metrics.
            target_filepaths: If provided, compute git metrics only for these files.
                            If None, uses the files discovered at the commit.

        Returns:
            HistoricalSnapshot with code metrics, git metrics, and dependency graph
            all extracted from the historical state at the given commit.
        """
        if not _validate_commit_hash(commit_hash):
            raise ValueError(f"Invalid commit hash: {commit_hash!r}")

        # 1. Extract code metrics from historical source files
        code_metrics = self.extract_code_metrics_at_commit(commit_hash)
        python_files = sorted(code_metrics.keys())

        # 2. Build dependency graph from historical import statements
        dep_graph = DependencyAnalyzer.build_graph_from_metrics(code_metrics)

        # 3. Compute git evolution metrics using only commits up to this point
        filepaths_for_git = target_filepaths or python_files
        git_analyzer = GitAnalyzer(commits_up_to)
        git_metrics = git_analyzer.analyze_components(
            target_filepaths=filepaths_for_git,
            as_of_time=timestamp
        )

        return HistoricalSnapshot(
            commit_hash=commit_hash,
            timestamp=timestamp,
            code_metrics=code_metrics,
            git_metrics=git_metrics,
            dep_graph=dep_graph,
            python_files=python_files
        )

