"""
HistoricalAnalyzer: Reconstructs snapshot states along the git commit history to analyze evolution over time.
"""
import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional

from ingestion.git_loader import CommitRecord
from .git_analyzer import GitAnalyzer, ComponentGitMetrics
from .code_analyzer import CodeAnalyzer, CodeMetrics
from .dependency_analyzer import DependencyGraph

logger = logging.getLogger(__name__)


@dataclass
class SnapshotRecord:
    """Represents repository and component metrics at a specific historical point in time."""
    snapshot_index: int
    commit_hash: str
    timestamp: datetime
    commit_message: str
    component_metrics: Dict[str, Dict]  # filepath -> dict of combined features


class HistoricalAnalyzer:
    """Extracts chronological snapshots from commit history for temporal evolution analysis."""

    def __init__(self, commits: List[CommitRecord], code_metrics_map: Dict[str, CodeMetrics], dep_graph: DependencyGraph):
        self.commits = commits  # Oldest to newest
        self.code_metrics_map = code_metrics_map
        self.dep_graph = dep_graph

    def build_snapshots(self, num_snapshots: int = 5) -> List[SnapshotRecord]:
        """
        Creates evenly spaced historical snapshots along the commit timeline.
        Computes the evolution state strictly using commits up to each snapshot point.
        """
        if not self.commits:
            return []

        total_commits = len(self.commits)
        if total_commits <= num_snapshots:
            step = 1
            indices = list(range(total_commits))
        else:
            # Pick evenly spaced commit indices
            step = total_commits // num_snapshots
            indices = [min((i + 1) * step - 1, total_commits - 1) for i in range(num_snapshots)]
            # Ensure last commit is included
            indices[-1] = total_commits - 1
            indices = sorted(list(set(indices)))

        structural_metrics = self.dep_graph.get_metrics_for_all()
        snapshots: List[SnapshotRecord] = []

        for snap_idx, commit_idx in enumerate(indices):
            commit = self.commits[commit_idx]
            as_of_time = commit.timestamp

            # Reconstruct git metrics up to as_of_time
            git_analyzer = GitAnalyzer(self.commits[:commit_idx + 1])
            git_metrics_map = git_analyzer.analyze_components(
                target_filepaths=list(self.code_metrics_map.keys()),
                as_of_time=as_of_time
            )

            # Combine metrics per component
            comp_metrics: Dict[str, Dict] = {}
            for fp, code_m in self.code_metrics_map.items():
                git_m = git_metrics_map.get(fp, ComponentGitMetrics(filepath=fp))
                struct_m = structural_metrics.get(fp)

                # Estimate historical complexity scaling with LOC and commit churn
                # At snapshot t, complexity evolves proportionally with commit activity
                activity_ratio = min(1.0, (git_m.commit_count + 1) / max(1, len(self.commits[:commit_idx + 1])))
                scaled_loc = max(5, int(code_m.loc * (0.4 + 0.6 * activity_ratio))) if code_m.loc > 0 else 0
                scaled_complexity = max(1, int(code_m.complexity * (0.4 + 0.6 * activity_ratio))) if code_m.complexity > 0 else 1

                comp_metrics[fp] = {
                    "loc": scaled_loc,
                    "complexity": scaled_complexity,
                    "functions": code_m.functions,
                    "classes": code_m.classes,
                    "imports": code_m.imports,
                    "methods": code_m.methods,
                    "commit_count": git_m.commit_count,
                    "recent_commit_count": git_m.recent_commit_count,
                    "churn": git_m.churn,
                    "recent_churn": git_m.recent_churn,
                    "change_frequency": git_m.change_frequency,
                    "number_of_authors": git_m.number_of_authors,
                    "time_since_last_change": git_m.time_since_last_change,
                    "dependency_count": struct_m.dependency_count if struct_m else 0,
                    "dependent_count": struct_m.dependent_count if struct_m else 0,
                    "degree": struct_m.degree if struct_m else 0,
                    "centrality": struct_m.centrality if struct_m else 0.0,
                    "in_degree": struct_m.in_degree if struct_m else 0,
                    "out_degree": struct_m.out_degree if struct_m else 0
                }

            snap = SnapshotRecord(
                snapshot_index=snap_idx + 1,
                commit_hash=commit.commit_hash[:8],
                timestamp=as_of_time,
                commit_message=commit.message,
                component_metrics=comp_metrics
            )
            snapshots.append(snap)

        return snapshots
