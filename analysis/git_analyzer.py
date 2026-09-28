"""
GitAnalyzer: Computes file-level evolution metrics, churn dynamics, commit frequency, and author counts.
"""
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Dict, List, Optional, Set

from ingestion.git_loader import CommitRecord

logger = logging.getLogger(__name__)


@dataclass
class ComponentGitMetrics:
    """Git evolution metrics for a specific file/component."""
    filepath: str
    commit_count: int = 0
    recent_commit_count: int = 0
    total_insertions: int = 0
    total_deletions: int = 0
    churn: int = 0
    recent_churn: int = 0
    number_of_authors: int = 0
    change_frequency: float = 0.0
    average_changes_per_commit: float = 0.0
    time_since_last_change: float = 0.0  # in days
    time_since_first_change: float = 0.0  # in days
    bugfix_commit_count: int = 0

    def to_dict(self) -> Dict:
        return {
            "filepath": self.filepath,
            "commit_count": self.commit_count,
            "recent_commit_count": self.recent_commit_count,
            "total_insertions": self.total_insertions,
            "total_deletions": self.total_deletions,
            "churn": self.churn,
            "recent_churn": self.recent_churn,
            "number_of_authors": self.number_of_authors,
            "change_frequency": round(self.change_frequency, 4),
            "average_changes_per_commit": round(self.average_changes_per_commit, 2),
            "time_since_last_change": round(self.time_since_last_change, 2),
            "time_since_first_change": round(self.time_since_first_change, 2),
            "bugfix_commit_count": self.bugfix_commit_count
        }


class GitAnalyzer:
    """Analyzes a chronological list of CommitRecord objects to extract per-component evolution metrics."""

    def __init__(self, commits: List[CommitRecord], recent_commits_window: int = 20, recent_days_window: int = 30):
        self.commits = commits  # Chronologically ordered oldest -> newest
        self.recent_commits_window = recent_commits_window
        self.recent_days_window = recent_days_window

    def analyze_components(self, target_filepaths: Optional[List[str]] = None, as_of_time: Optional[datetime] = None) -> Dict[str, ComponentGitMetrics]:
        """
        Calculates evolution metrics for all files appearing in commits or target_filepaths.
        If as_of_time is given, only commits up to as_of_time are included (preventing temporal leakage).
        """
        # Filter commits chronologically up to as_of_time if specified
        active_commits = [c for c in self.commits if as_of_time is None or c.timestamp <= as_of_time]
        total_active_commits = len(active_commits)

        # Set reference timestamp
        if as_of_time:
            ref_time = as_of_time
        elif active_commits:
            ref_time = active_commits[-1].timestamp
        else:
            ref_time = datetime.now(timezone.utc)

        # Recent commit window split
        recent_commit_cutoff_idx = max(0, total_active_commits - self.recent_commits_window)
        recent_commit_hashes = set(c.commit_hash for c in active_commits[recent_commit_cutoff_idx:])

        # Accumulators per filepath
        file_commits: Dict[str, List[CommitRecord]] = {}
        file_insertions: Dict[str, int] = {}
        file_deletions: Dict[str, int] = {}
        file_recent_churn: Dict[str, int] = {}
        file_recent_commits: Dict[str, int] = {}
        file_authors: Dict[str, Set[str]] = {}
        file_bugfix_commits: Dict[str, int] = {}
        file_timestamps: Dict[str, List[datetime]] = {}

        # Initialize targets
        all_known_files = set(target_filepaths or [])

        for commit in active_commits:
            is_recent_by_index = commit.commit_hash in recent_commit_hashes

            days_diff = (ref_time - commit.timestamp).total_seconds() / 86400.0
            is_recent_by_time = days_diff <= self.recent_days_window

            # In commit, file changes
            files_in_this_commit = commit.files_changed
            num_files = max(len(files_in_this_commit), 1)
            # Estimate per-file churn if multiple files in commit
            ins_per_file = max(1, commit.insertions // num_files) if commit.insertions > 0 else 0
            dels_per_file = max(1, commit.deletions // num_files) if commit.deletions > 0 else 0

            for fp in files_in_this_commit:
                norm_fp = fp.replace("\\", "/")
                all_known_files.add(norm_fp)

                file_commits.setdefault(norm_fp, []).append(commit)
                file_insertions[norm_fp] = file_insertions.get(norm_fp, 0) + ins_per_file
                file_deletions[norm_fp] = file_deletions.get(norm_fp, 0) + dels_per_file
                file_authors.setdefault(norm_fp, set()).add(commit.author_email or commit.author_name)
                file_timestamps.setdefault(norm_fp, []).append(commit.timestamp)

                if commit.is_bugfix:
                    file_bugfix_commits[norm_fp] = file_bugfix_commits.get(norm_fp, 0) + 1

                if is_recent_by_index or is_recent_by_time:
                    file_recent_commits[norm_fp] = file_recent_commits.get(norm_fp, 0) + 1
                    file_recent_churn[norm_fp] = file_recent_churn.get(norm_fp, 0) + (ins_per_file + dels_per_file)

        metrics_map: Dict[str, ComponentGitMetrics] = {}

        for fp in all_known_files:
            commits_for_fp = file_commits.get(fp, [])
            c_count = len(commits_for_fp)
            ins = file_insertions.get(fp, 0)
            dels = file_deletions.get(fp, 0)
            churn = ins + dels
            r_churn = file_recent_churn.get(fp, 0)
            r_commits = file_recent_commits.get(fp, 0)
            authors = len(file_authors.get(fp, set()))
            bugfixes = file_bugfix_commits.get(fp, 0)

            # Change frequency: ratio of commits modifying this file
            change_freq = (c_count / total_active_commits) if total_active_commits > 0 else 0.0
            avg_changes = (churn / c_count) if c_count > 0 else 0.0

            # Time since last and first change
            ts_list = file_timestamps.get(fp, [])
            if ts_list:
                time_since_last = max(0.0, (ref_time - ts_list[-1]).total_seconds() / 86400.0)
                time_since_first = max(0.0, (ref_time - ts_list[0]).total_seconds() / 86400.0)
            else:
                time_since_last = 999.0
                time_since_first = 999.0

            metrics_map[fp] = ComponentGitMetrics(
                filepath=fp,
                commit_count=c_count,
                recent_commit_count=r_commits,
                total_insertions=ins,
                total_deletions=dels,
                churn=churn,
                recent_churn=r_churn,
                number_of_authors=authors,
                change_frequency=change_freq,
                average_changes_per_commit=avg_changes,
                time_since_last_change=time_since_last,
                time_since_first_change=time_since_first,
                bugfix_commit_count=bugfixes
            )

        return metrics_map
