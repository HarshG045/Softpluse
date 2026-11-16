"""
SnapshotEngine: Safe, read-only historical repository snapshot analysis.

Uses `git show <commit>:<path>` to read file content at a specific commit
WITHOUT modifying the working tree, resetting HEAD, or affecting the user's checkout.

Results are cached by (repo_path, commit_hash, analyzer_version) to avoid
re-parsing unchanged historical snapshots.
"""
import hashlib
import logging
import re
import subprocess
import tempfile
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from analysis.code_analyzer import CodeAnalyzer, CodeMetrics
from analysis.dependency_analyzer import DependencyAnalyzer, DependencyGraph
from analysis.git_analyzer import GitAnalyzer, ComponentGitMetrics
from ingestion.git_loader import CommitRecord, GitLoader

logger = logging.getLogger(__name__)

# Bump this version when the analysis logic changes to invalidate old caches.
ANALYZER_VERSION = "1.0.0"

# Simple in-process LRU-style cache keyed by (repo_path, commit_hash, version).
# Stored per Python process — no persistence across restarts.
_SNAPSHOT_CACHE: Dict[str, "CommitSnapshot"] = {}
_MAX_CACHE_SIZE = 20  # evict oldest entries beyond this limit


@dataclass
class CommitSnapshot:
    """Repository state reconstructed at a specific historical commit."""
    commit_hash: str                          # Full 40-char SHA
    short_hash: str                           # 7-char abbreviation
    timestamp: datetime
    author_name: str
    author_email: str
    message: str
    parent_hash: Optional[str]
    # Maps relative file path -> CodeMetrics (AST-extracted at this commit)
    code_map: Dict[str, CodeMetrics] = field(default_factory=dict)
    # Dependency graph reconstructed from code_map imports
    dep_graph: Optional[DependencyGraph] = None
    # Git evolution metrics up to this commit
    git_metrics_map: Dict[str, ComponentGitMetrics] = field(default_factory=dict)
    # Analysis metadata
    files_available: int = 0
    files_parse_errors: int = 0
    analyzer_version: str = ANALYZER_VERSION
    # Whether ML risk could be calculated (requires compatible feature schema)
    risk_available: bool = False
    risk_unavailable_reason: str = ""


@dataclass
class SnapshotDiff:
    """Structural differences between two CommitSnapshots (baseline → comparison)."""
    baseline_hash: str
    comparison_hash: str
    baseline_timestamp: datetime
    comparison_timestamp: datetime
    # File-level changes
    added_files: List[str] = field(default_factory=list)
    removed_files: List[str] = field(default_factory=list)
    modified_files: List[str] = field(default_factory=list)
    # Component-level structural diffs: filepath -> dict of metric deltas
    metric_deltas: Dict[str, Dict[str, float]] = field(default_factory=dict)
    # Dependency graph topology diffs
    added_edges: List[Tuple[str, str]] = field(default_factory=list)
    removed_edges: List[Tuple[str, str]] = field(default_factory=list)
    # Risk score diffs per component: filepath -> (baseline_pct, comparison_pct, delta)
    risk_deltas: Dict[str, Tuple[int, int, int]] = field(default_factory=dict)


def _validate_commit_hash(commit_hash: str) -> bool:
    """
    Validates that a string is a plausible git commit reference.
    Accepts full SHA (40 hex chars), short SHA (7+ hex chars), or safe branch/tag names.
    Rejects path traversal sequences, null bytes, and shell metacharacters.
    """
    # Reject path traversal and dangerous characters first
    if ".." in commit_hash:
        return False
    if "\x00" in commit_hash:
        return False
    if any(c in commit_hash for c in (";", "&", "|", "$", "(", ")", "`", " ", "\t", "\n", "\r", "!")):
        return False
    # Allow hex SHAs (short or full)
    if re.fullmatch(r"[0-9a-fA-F]{7,40}", commit_hash):
        return True
    # Allow safe ref names: letters, digits, -, _, ., and single / (not leading/trailing/consecutive)
    if re.fullmatch(r"[A-Za-z0-9_][A-Za-z0-9._\-]*(/[A-Za-z0-9._\-]+)*", commit_hash):
        return True
    return False



def _cache_key(repo_path: str, commit_hash: str) -> str:
    h = hashlib.sha256(f"{repo_path}|{commit_hash}|{ANALYZER_VERSION}".encode()).hexdigest()[:16]
    return h


def _evict_cache_if_needed():
    """Evicts oldest entries when cache exceeds size limit."""
    global _SNAPSHOT_CACHE
    if len(_SNAPSHOT_CACHE) > _MAX_CACHE_SIZE:
        # Remove oldest half
        keys = list(_SNAPSHOT_CACHE.keys())
        for k in keys[:len(keys) // 2]:
            del _SNAPSHOT_CACHE[k]


def get_cached_snapshot(repo_path: str, commit_hash: str) -> Optional[CommitSnapshot]:
    """Returns a cached snapshot if available."""
    key = _cache_key(repo_path, commit_hash)
    return _SNAPSHOT_CACHE.get(key)


def _store_snapshot(repo_path: str, commit_hash: str, snapshot: CommitSnapshot):
    """Stores a snapshot in the in-process cache."""
    _evict_cache_if_needed()
    key = _cache_key(repo_path, commit_hash)
    _SNAPSHOT_CACHE[key] = snapshot


def _git_list_files_at_commit(repo_path: Path, commit_hash: str) -> List[str]:
    """
    Lists all Python files in the repository at a given commit
    using `git ls-tree -r --name-only <commit>`.
    Returns relative paths in POSIX format.
    """
    try:
        result = subprocess.run(
            ["git", "ls-tree", "-r", "--name-only", commit_hash],
            cwd=str(repo_path),
            capture_output=True,
            text=True,
            timeout=30
        )
        if result.returncode != 0:
            logger.warning(f"git ls-tree failed for {commit_hash}: {result.stderr.strip()}")
            return []

        files = []
        for line in result.stdout.splitlines():
            line = line.strip()
            if line.endswith(".py"):
                # Exclude virtual environments, pycache, etc.
                parts = Path(line).parts
                skip_dirs = {"venv", ".venv", "env", "__pycache__", "build", "dist", "node_modules", ".git"}
                if not any(p in skip_dirs or p.startswith(".") for p in parts):
                    files.append(line.replace("\\", "/"))
        return files
    except subprocess.TimeoutExpired:
        logger.error(f"git ls-tree timed out for {commit_hash}")
        return []
    except Exception as e:
        logger.error(f"git ls-tree error: {e}")
        return []


def _git_show_file(repo_path: Path, commit_hash: str, filepath: str) -> Optional[str]:
    """
    Reads file content at a specific commit using `git show <commit>:<path>`.
    This is a safe, read-only git object inspection — does NOT modify working tree.
    Returns file content as string, or None if unavailable.
    """
    try:
        result = subprocess.run(
            ["git", "show", f"{commit_hash}:{filepath}"],
            cwd=str(repo_path),
            capture_output=True,
            timeout=15
        )
        if result.returncode != 0:
            return None
        # Decode with error replacement to handle binary/non-UTF-8 files gracefully
        return result.stdout.decode("utf-8", errors="replace")
    except subprocess.TimeoutExpired:
        logger.debug(f"git show timed out for {filepath} at {commit_hash}")
        return None
    except Exception as e:
        logger.debug(f"git show error for {filepath}: {e}")
        return None


def _analyze_code_at_commit(
    repo_path: Path,
    commit_hash: str,
    filepaths: List[str]
) -> Tuple[Dict[str, CodeMetrics], int]:
    """
    Extracts AST code metrics for all files at a given commit using git show.
    Returns (code_map, parse_error_count).
    """
    # Use a temporary directory to write files for AST parsing
    parse_errors = 0
    code_map: Dict[str, CodeMetrics] = {}

    with tempfile.TemporaryDirectory(prefix="sp_snapshot_") as tmpdir:
        tmp_path = Path(tmpdir)

        for fp in filepaths:
            content = _git_show_file(repo_path, commit_hash, fp)
            if content is None:
                # File may have been deleted or renamed — skip gracefully
                continue

            # Write content to temp file preserving directory structure
            tmp_file = tmp_path / fp
            tmp_file.parent.mkdir(parents=True, exist_ok=True)
            try:
                tmp_file.write_text(content, encoding="utf-8")
            except Exception:
                continue

        # Run CodeAnalyzer against the temp directory
        if any(tmp_path.rglob("*.py")):
            analyzer = CodeAnalyzer(tmp_path)
            raw_map = analyzer.analyze_repository()
            for fp, metrics in raw_map.items():
                # Re-key to the original (repo-relative) path
                code_map[fp] = metrics
                code_map[fp].filepath = fp  # normalize
                if metrics.has_parse_error:
                    parse_errors += 1

    return code_map, parse_errors


def build_snapshot(
    repo_path: str,
    commit_hash: str,
    all_commits: List[CommitRecord],
    force_refresh: bool = False
) -> Optional[CommitSnapshot]:
    """
    Builds a CommitSnapshot for the given commit by:
      1. Listing Python files at that commit (git ls-tree).
      2. Reading each file's content (git show — read-only).
      3. Extracting AST code metrics in a temp directory.
      4. Reconstructing the dependency graph from imports.
      5. Computing git evolution metrics up to that commit.

    Results are cached. Pass force_refresh=True to bypass the cache.

    SAFETY: No git checkout, reset, or working-tree mutation is performed.
    """
    if not _validate_commit_hash(commit_hash):
        logger.error(f"Invalid commit hash: {commit_hash!r} — aborting snapshot")
        return None

    repo = Path(repo_path).resolve()
    if not (repo / ".git").exists():
        logger.error(f"Path {repo} is not a git repository")
        return None

    # Check cache
    if not force_refresh:
        cached = get_cached_snapshot(str(repo), commit_hash)
        if cached is not None:
            return cached

    # Resolve commit metadata
    try:
        commit_record: Optional[CommitRecord] = None
        for c in all_commits:
            if c.commit_hash.startswith(commit_hash) or commit_hash.startswith(c.commit_hash[:7]):
                commit_record = c
                break

        if commit_record is None:
            # Try git log for the commit
            res = subprocess.run(
                ["git", "log", "-1", "--format=%H|%an|%ae|%ct|%P|%s", commit_hash],
                cwd=str(repo),
                capture_output=True,
                text=True,
                timeout=10
            )
            if res.returncode != 0 or not res.stdout.strip():
                logger.error(f"Cannot resolve commit {commit_hash}")
                return None

            parts = res.stdout.strip().split("|")
            full_hash = parts[0] if parts else commit_hash
            author_name = parts[1] if len(parts) > 1 else "Unknown"
            author_email = parts[2] if len(parts) > 2 else ""
            ts = int(parts[3]) if len(parts) > 3 and parts[3].isdigit() else 0
            parents = parts[4].split() if len(parts) > 4 else []
            message = parts[5] if len(parts) > 5 else ""
            parent_hash = parents[0] if parents else None
            timestamp = datetime.fromtimestamp(ts, tz=timezone.utc) if ts else datetime.now(timezone.utc)
        else:
            full_hash = commit_record.commit_hash
            author_name = commit_record.author_name
            author_email = commit_record.author_email
            timestamp = commit_record.timestamp
            message = commit_record.message
            parent_hash = None  # CommitRecord doesn't store parent
            # Try to get parent
            try:
                res = subprocess.run(
                    ["git", "log", "-1", "--format=%P", full_hash],
                    cwd=str(repo),
                    capture_output=True,
                    text=True,
                    timeout=10
                )
                parents_line = res.stdout.strip().split()
                parent_hash = parents_line[0] if parents_line else None
            except Exception:
                pass

    except Exception as e:
        logger.error(f"Error resolving commit metadata: {e}")
        return None

    # Step 1: List Python files at this commit
    py_files = _git_list_files_at_commit(repo, full_hash)
    if not py_files:
        logger.warning(f"No Python files found at commit {full_hash[:7]}")

    # Step 2 & 3: Extract code metrics from historical file contents
    code_map, parse_errors = _analyze_code_at_commit(repo, full_hash, py_files)

    # Step 4: Reconstruct dependency graph
    dep_graph: Optional[DependencyGraph] = None
    if code_map:
        try:
            dep_analyzer = DependencyAnalyzer(repo)
            dep_graph = dep_analyzer.build_graph(code_map)
        except Exception as e:
            logger.warning(f"Dependency graph reconstruction failed: {e}")
            dep_graph = DependencyGraph()

    # Step 5: Compute git evolution metrics up to this commit
    commits_up_to = [c for c in all_commits if c.timestamp <= timestamp]
    git_metrics_map: Dict[str, ComponentGitMetrics] = {}
    if commits_up_to and code_map:
        try:
            git_analyzer = GitAnalyzer(commits_up_to)
            git_metrics_map = git_analyzer.analyze_components(
                target_filepaths=list(code_map.keys()),
                as_of_time=timestamp
            )
        except Exception as e:
            logger.warning(f"Git metrics extraction failed: {e}")

    snapshot = CommitSnapshot(
        commit_hash=full_hash,
        short_hash=full_hash[:7],
        timestamp=timestamp,
        author_name=author_name,
        author_email=author_email,
        message=message,
        parent_hash=parent_hash,
        code_map=code_map,
        dep_graph=dep_graph,
        git_metrics_map=git_metrics_map,
        files_available=len(code_map),
        files_parse_errors=parse_errors,
        analyzer_version=ANALYZER_VERSION,
        risk_available=False,  # Will be set by caller if model is compatible
        risk_unavailable_reason="Risk calculation deferred to caller."
    )

    _store_snapshot(str(repo), full_hash, snapshot)
    return snapshot


def diff_snapshots(
    baseline: CommitSnapshot,
    comparison: CommitSnapshot
) -> SnapshotDiff:
    """
    Computes structural differences between two CommitSnapshots.
    Compares file sets, code metrics, and dependency graph topology.
    """
    baseline_files = set(baseline.code_map.keys())
    comparison_files = set(comparison.code_map.keys())

    added_files = sorted(comparison_files - baseline_files)
    removed_files = sorted(baseline_files - comparison_files)
    common_files = baseline_files & comparison_files

    # Identify modified files (any metric changed)
    modified_files = []
    metric_deltas: Dict[str, Dict[str, float]] = {}
    COMPARABLE_METRICS = ["loc", "complexity", "functions", "classes", "imports", "methods"]

    for fp in sorted(common_files):
        base_m = baseline.code_map[fp]
        comp_m = comparison.code_map[fp]
        deltas = {}
        for attr in COMPARABLE_METRICS:
            base_val = float(getattr(base_m, attr, 0) or 0)
            comp_val = float(getattr(comp_m, attr, 0) or 0)
            delta = comp_val - base_val
            if delta != 0:
                deltas[attr] = delta
        if deltas:
            modified_files.append(fp)
            metric_deltas[fp] = deltas

    # Dependency graph diffs
    added_edges: List[Tuple[str, str]] = []
    removed_edges: List[Tuple[str, str]] = []
    if baseline.dep_graph and comparison.dep_graph:
        base_edges = set(baseline.dep_graph.graph.edges())
        comp_edges = set(comparison.dep_graph.graph.edges())
        added_edges = sorted(comp_edges - base_edges)
        removed_edges = sorted(base_edges - comp_edges)

    return SnapshotDiff(
        baseline_hash=baseline.short_hash,
        comparison_hash=comparison.short_hash,
        baseline_timestamp=baseline.timestamp,
        comparison_timestamp=comparison.timestamp,
        added_files=added_files,
        removed_files=removed_files,
        modified_files=modified_files,
        metric_deltas=metric_deltas,
        added_edges=added_edges,
        removed_edges=removed_edges,
    )


def clear_snapshot_cache():
    """Clears the in-process snapshot cache. Useful for testing."""
    global _SNAPSHOT_CACHE
    _SNAPSHOT_CACHE.clear()
