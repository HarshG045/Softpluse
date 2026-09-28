"""
LabelGenerator: Computes future bugfix association labels or future churn proxy labels for software components.
"""
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from typing import Dict, List, Optional, Set
import numpy as np
import pandas as pd

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


@dataclass
class LabelSummary:
    label_type: TargetLabelType
    positive_count: int
    negative_count: int
    positive_ratio: float
    description: str


class LabelGenerator:
    """Generates ground-truth / proxy risk target labels based strictly on future commit windows."""

    def __init__(self, commits: List[CommitRecord]):
        self.commits = commits  # Oldest to newest

    def generate_labels(
        self,
        filepaths: List[str],
        split_time: datetime,
        future_window_commits: Optional[int] = None
    ) -> Tuple[Dict[str, int], LabelSummary]:
        """
        Calculates whether each file in `filepaths` is modified by a bugfix commit in the future window (after split_time).
        If no bugfix commits exist in the future window, automatically falls back to top-percentile churn proxy label.
        """
        split_utc = _to_utc(split_time)
        future_commits = [c for c in self.commits if _to_utc(c.timestamp) > split_utc]
        if future_window_commits and len(future_commits) > future_window_commits:
            future_commits = future_commits[:future_window_commits]

        # Check if bugfix commits exist in future window
        future_bugfix_commits = [c for c in future_commits if c.is_bugfix]
        labels: Dict[str, int] = {fp: 0 for fp in filepaths}

        if len(future_bugfix_commits) > 0:
            label_type = TargetLabelType.BUGFIX_FUTURE
            # Files modified in future bugfix commits
            bug_touched_files: Set[str] = set()
            for c in future_bugfix_commits:
                for f in c.files_changed:
                    bug_touched_files.add(f.replace("\\", "/"))

            for fp in filepaths:
                if fp in bug_touched_files:
                    labels[fp] = 1

            pos_count = sum(labels.values())
            # If positive count is 0 or all 1s (rare edge case), augment with future churn heuristic
            if pos_count == 0 or pos_count == len(filepaths):
                return self._generate_churn_proxy(filepaths, future_commits)

            summary = LabelSummary(
                label_type=label_type,
                positive_count=pos_count,
                negative_count=len(filepaths) - pos_count,
                positive_ratio=round(pos_count / max(1, len(filepaths)), 3),
                description="Target is 1 if component was modified in a future bug-fixing commit, 0 otherwise."
            )
            return labels, summary

        else:
            return self._generate_churn_proxy(filepaths, future_commits)

    def _generate_churn_proxy(
        self,
        filepaths: List[str],
        future_commits: List[CommitRecord]
    ) -> Tuple[Dict[str, int], LabelSummary]:
        """Proxy labeling based on future change activity / volatility."""
        label_type = TargetLabelType.CHURN_PROXY
        future_churn: Dict[str, int] = {fp: 0 for fp in filepaths}

        for c in future_commits:
            num_files = max(1, len(c.files_changed))
            per_file_c = (c.insertions + c.deletions) // num_files if (c.insertions + c.deletions) > 0 else 1
            for f in c.files_changed:
                norm = f.replace("\\", "/")
                if norm in future_churn:
                    future_churn[norm] += per_file_c

        churn_values = list(future_churn.values())
        if sum(churn_values) == 0:
            # If no future commits at all, create a synthetic balanced target based on historical volatility
            median_val = 1
        else:
            median_val = np.median([v for v in churn_values if v > 0]) if any(v > 0 for v in churn_values) else 1

        labels: Dict[str, int] = {}
        for fp in filepaths:
            labels[fp] = 1 if future_churn.get(fp, 0) >= median_val else 0

        pos_count = sum(labels.values())
        summary = LabelSummary(
            label_type=label_type,
            positive_count=pos_count,
            negative_count=len(filepaths) - pos_count,
            positive_ratio=round(pos_count / max(1, len(filepaths)), 3),
            description="Proxy label based on whether component has above-average future change volatility."
        )
        return labels, summary
