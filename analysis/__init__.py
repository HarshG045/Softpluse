"""
SoftwarePulse Analysis Package.
Includes AST-based code analysis, Git history analysis, dependency graph construction,
historical snapshots, and the new Architecture Evolution snapshot engine.
"""
from .code_analyzer import CodeAnalyzer, CodeMetrics
from .git_analyzer import GitAnalyzer, ComponentGitMetrics
from .dependency_analyzer import DependencyAnalyzer, DependencyGraph
from .historical_analyzer import HistoricalAnalyzer, SnapshotRecord
from .snapshot_engine import CommitSnapshot, SnapshotDiff, build_snapshot, diff_snapshots
from .impact_propagator import ImpactPropagator, AffectedComponent

__all__ = [
    "CodeAnalyzer", "CodeMetrics",
    "GitAnalyzer", "ComponentGitMetrics",
    "DependencyAnalyzer", "DependencyGraph",
    "HistoricalAnalyzer", "SnapshotRecord",
    "CommitSnapshot", "SnapshotDiff", "build_snapshot", "diff_snapshots",
    "ImpactPropagator", "AffectedComponent",
]
