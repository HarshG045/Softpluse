"""
LabelGenerator: Computes future bugfix association labels or future churn proxy labels for software components.
Provides rigorous temporal tracking of label windows, contributing commits, and labeling provenance.
"""
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Dict, List, Optional, Set, Tuple
import numpy as np

from config import BUG_KEYWORDS
from ingestion.git_loader import CommitRecord

logger = logging.getLogger(__name__)


def _to_utc(dt: datetime) -> datetime:
    """Ensures datetime object is timezone-aware in UTC."""
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


class TargetLabelType(str, Enum):
    BUGFIX_FUTURE = "Future Bugfix Commit Association (Heuristic)"
    CHURN_PROXY = "High Future Churn / Volatility (Proxy Label)"
    INSUFFICIENT_DATA = "Insufficient Future Data (No Qualifying Events)"


class LabelMethod(str, Enum):
    HEURISTIC_BUGFIX = "heuristic_bugfix"
    CHURN_PROXY = "churn_proxy"
    INSUFFICIENT_DATA = "insufficient_data"


@dataclass
class LabelSummary:
    """Metadata summary describing how ground-truth labels were generated."""
    label_type: TargetLabelType
    positive_count: int
    negative_count: int
    positive_ratio: float
    description: str
    label_method: LabelMethod = LabelMethod.HEURISTIC_BUGFIX
    is_proxy: bool = False
    window_start: Optional[datetime] = None
    window_end: Optional[datetime] = None
    contributing_commit_hashes: List[str] = field(default_factory=list)
    component_provenance: Dict[str, List[str]] = field(default_factory=dict)
    is_single_class: bool = False


class LabelGenerator:
    """Generates ground-truth or proxy risk target labels strictly based on future commit windows."""

    def __init__(
        self,
        commits: List[CommitRecord],
        bug_keywords: Optional[List[str]] = None
    ):
        self.commits = commits  # Oldest to newest
        self.bug_keywords = [kw.lower() for kw in (bug_keywords or BUG_KEYWORDS)]

    def _is_bugfix_commit(self, commit: CommitRecord) -> bool:
        """Determines if a commit is a bug-fixing commit based on configurable keywords."""
        msg = (commit.message or "").lower()
        return any(kw in msg for kw in self.bug_keywords) or commit.is_bugfix

    def generate_labels(
        self,
        filepaths: List[str],
        window_start: Optional[datetime] = None,
        window_end: Optional[datetime] = None,
        split_time: Optional[datetime] = None,
        future_window_commits: Optional[int] = None,
        allow_proxy: bool = True
    ) -> Tuple[Dict[str, int], LabelSummary]:
        """
        Calculates whether each file in `filepaths` is modified by a bugfix commit in the future window
        defined by (window_start, window_end].

        Args:
            filepaths: Components to label.
            window_start: Start of the future label window (exclusive). If None, defaults to split_time.
            window_end: End of the future label window (inclusive). If None, unbounded up to latest commit.
            split_time: Deprecated alias for window_start (backward compatibility).
            future_window_commits: Optional maximum count of future commits to consider.
            allow_proxy: If True, falls back to churn-proxy when bugfix data is insufficient,
                         explicitly documenting the proxy label in LabelSummary.

        Returns:
            Tuple of (labels_map, label_summary).
        """
        effective_start = window_start if window_start is not None else split_time
        if effective_start is None:
            effective_start = datetime.min.replace(tzinfo=timezone.utc)

        start_utc = _to_utc(effective_start)
        end_utc = _to_utc(window_end) if window_end is not None else None

        # Filter commits strictly within the future window: (start_utc, end_utc]
        future_commits = []
        for c in self.commits:
            c_ts = _to_utc(c.timestamp)
            if c_ts > start_utc:
                if end_utc is None or c_ts <= end_utc:
                    future_commits.append(c)

        if future_window_commits and len(future_commits) > future_window_commits:
            future_commits = future_commits[:future_window_commits]

        # Check if bugfix commits exist in future window
        future_bugfix_commits = [c for c in future_commits if self._is_bugfix_commit(c)]
        labels: Dict[str, int] = {fp: 0 for fp in filepaths}
        provenance: Dict[str, List[str]] = {fp: [] for fp in filepaths}
        contributing_commits: List[str] = []

        if len(future_bugfix_commits) > 0:
            for c in future_bugfix_commits:
                commit_used = False
                for f in c.files_changed:
                    norm = f.replace("\\", "/")
                    if norm in labels:
                        labels[norm] = 1
                        if c.commit_hash not in provenance[norm]:
                            provenance[norm].append(c.commit_hash)
                        commit_used = True
                    else:
                        # Partial match for path endings
                        for fp in filepaths:
                            if fp.endswith(norm) or norm.endswith(fp):
                                labels[fp] = 1
                                if c.commit_hash not in provenance[fp]:
                                    provenance[fp].append(c.commit_hash)
                                commit_used = True
                if commit_used and c.commit_hash not in contributing_commits:
                    contributing_commits.append(c.commit_hash)

            pos_count = sum(labels.values())

            # If bugfix labels have both classes, return them directly
            if 0 < pos_count < len(filepaths):
                summary = LabelSummary(
                    label_type=TargetLabelType.BUGFIX_FUTURE,
                    label_method=LabelMethod.HEURISTIC_BUGFIX,
                    positive_count=pos_count,
                    negative_count=len(filepaths) - pos_count,
                    positive_ratio=round(pos_count / max(1, len(filepaths)), 3),
                    description="Target is 1 if component was modified in a future bug-fixing commit, 0 otherwise.",
                    is_proxy=False,
                    window_start=start_utc,
                    window_end=end_utc,
                    contributing_commit_hashes=contributing_commits,
                    component_provenance=provenance,
                    is_single_class=False
                )
                return labels, summary

            # If single class (all 0s or all 1s):
            if allow_proxy:
                logger.info(
                    f"Future bugfix labeling produced single-class outcome ({pos_count}/{len(filepaths)} positive). "
                    "Explicitly falling back to high-churn proxy label."
                )
                return self._generate_churn_proxy(
                    filepaths, future_commits, window_start=start_utc, window_end=end_utc
                )
            else:
                summary = LabelSummary(
                    label_type=TargetLabelType.BUGFIX_FUTURE,
                    label_method=LabelMethod.HEURISTIC_BUGFIX,
                    positive_count=pos_count,
                    negative_count=len(filepaths) - pos_count,
                    positive_ratio=round(pos_count / max(1, len(filepaths)), 3),
                    description="Target is 1 if component was modified in a future bug-fixing commit (single class detected).",
                    is_proxy=False,
                    window_start=start_utc,
                    window_end=end_utc,
                    contributing_commit_hashes=contributing_commits,
                    component_provenance=provenance,
                    is_single_class=True
                )
                return labels, summary

        else:
            # No bugfix commits in future window
            if allow_proxy and len(future_commits) > 0:
                logger.info("No bugfix commits found in future window. Explicitly using high-churn proxy label.")
                return self._generate_churn_proxy(
                    filepaths, future_commits, window_start=start_utc, window_end=end_utc
                )
            else:
                # Insufficient future data
                summary = LabelSummary(
                    label_type=TargetLabelType.INSUFFICIENT_DATA,
                    label_method=LabelMethod.INSUFFICIENT_DATA,
                    positive_count=0,
                    negative_count=len(filepaths),
                    positive_ratio=0.0,
                    description="No future defect or commit events observed in the specified prediction horizon.",
                    is_proxy=False,
                    window_start=start_utc,
                    window_end=end_utc,
                    contributing_commit_hashes=[],
                    component_provenance=provenance,
                    is_single_class=True
                )
                return labels, summary

    def _generate_churn_proxy(
        self,
        filepaths: List[str],
        future_commits: List[CommitRecord],
        window_start: Optional[datetime] = None,
        window_end: Optional[datetime] = None
    ) -> Tuple[Dict[str, int], LabelSummary]:
        """
        Proxy labeling based on future change activity / volatility.
        Explicitly identifies itself as a proxy label with provenance.
        Never fabricates labels when there is no activity at all.
        """
        future_churn: Dict[str, int] = {fp: 0 for fp in filepaths}
        provenance: Dict[str, List[str]] = {fp: [] for fp in filepaths}
        contributing_commits: List[str] = []

        for c in future_commits:
            num_files = max(1, len(c.files_changed))
            per_file_c = (c.insertions + c.deletions) // num_files if (c.insertions + c.deletions) > 0 else 1
            for f in c.files_changed:
                norm = f.replace("\\", "/")
                target_fp = None
                if norm in future_churn:
                    target_fp = norm
                else:
                    for fp in filepaths:
                        if fp.endswith(norm) or norm.endswith(fp):
                            target_fp = fp
                            break

                if target_fp:
                    future_churn[target_fp] += per_file_c
                    if c.commit_hash not in provenance[target_fp]:
                        provenance[target_fp].append(c.commit_hash)
                    if c.commit_hash not in contributing_commits:
                        contributing_commits.append(c.commit_hash)

        churn_values = [v for v in future_churn.values() if v > 0]
        if not churn_values:
            # Completely zero future activity — report insufficient data honestly
            labels = {fp: 0 for fp in filepaths}
            summary = LabelSummary(
                label_type=TargetLabelType.INSUFFICIENT_DATA,
                label_method=LabelMethod.INSUFFICIENT_DATA,
                positive_count=0,
                negative_count=len(filepaths),
                positive_ratio=0.0,
                description="Zero future change activity detected in future observation window.",
                is_proxy=True,
                window_start=window_start,
                window_end=window_end,
                contributing_commit_hashes=[],
                component_provenance=provenance,
                is_single_class=True
            )
            return labels, summary

        median_val = float(np.median(churn_values))
        labels: Dict[str, int] = {}
        for fp in filepaths:
            labels[fp] = 1 if future_churn.get(fp, 0) >= median_val else 0

        pos_count = sum(labels.values())
        summary = LabelSummary(
            label_type=TargetLabelType.CHURN_PROXY,
            label_method=LabelMethod.CHURN_PROXY,
            positive_count=pos_count,
            negative_count=len(filepaths) - pos_count,
            positive_ratio=round(pos_count / max(1, len(filepaths)), 3),
            description="Proxy label: 1 if component has above-median future churn in prediction horizon, 0 otherwise.",
            is_proxy=True,
            window_start=window_start,
            window_end=window_end,
            contributing_commit_hashes=contributing_commits,
            component_provenance=provenance,
            is_single_class=(pos_count == 0 or pos_count == len(filepaths))
        )
        return labels, summary
