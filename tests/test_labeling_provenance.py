"""
Tests for Ground-Truth and Proxy Labeling with strict window boundaries and provenance tracking.
"""
from datetime import datetime, timezone, timedelta
import pytest

from ingestion.git_loader import CommitRecord
from dataset.labeling import LabelGenerator, TargetLabelType, LabelMethod


def _make_commit(chash, msg, files, hours_offset=0, is_bugfix=False):
    base = datetime(2025, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
    return CommitRecord(
        commit_hash=chash,
        author_name="Dev",
        author_email="dev@code.org",
        timestamp=base + timedelta(hours=hours_offset),
        message=msg,
        files_changed=files,
        insertions=10,
        deletions=2,
        is_bugfix=is_bugfix
    )


def test_bounded_future_label_window():
    """Verifies that commits outside the (window_start, window_end] interval do not affect labels."""
    commits = [
        _make_commit("c1", "feat: init", ["auth.py"], hours_offset=1),
        _make_commit("c2", "fix: bug in auth", ["auth.py"], hours_offset=5, is_bugfix=True),
        _make_commit("c3", "feat: payment", ["payment.py"], hours_offset=10),
        _make_commit("c4", "fix: bug in payment", ["payment.py"], hours_offset=20, is_bugfix=True),
        _make_commit("c5", "fix: bug in auth", ["auth.py"], hours_offset=30, is_bugfix=True),
    ]

    base = datetime(2025, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
    start_time = base + timedelta(hours=6)   # After c2
    end_time = base + timedelta(hours=25)    # Covers c3, c4 (not c5)

    generator = LabelGenerator(commits)
    labels, summary = generator.generate_labels(
        filepaths=["auth.py", "payment.py"],
        window_start=start_time,
        window_end=end_time,
        allow_proxy=False
    )

    # Within (6h, 25h], only c4 is a bugfix commit, which touched payment.py
    # c2 was before start_time; c5 was after end_time
    assert labels["payment.py"] == 1
    assert labels["auth.py"] == 0
    assert summary.label_type == TargetLabelType.BUGFIX_FUTURE
    assert summary.is_proxy is False
    assert "c4" in summary.contributing_commit_hashes
    assert "c2" not in summary.contributing_commit_hashes
    assert "c5" not in summary.contributing_commit_hashes


def test_label_provenance_tracking():
    """Verifies that contributing commit hashes are tracked per component."""
    commits = [
        _make_commit("c101", "fix: payment null error", ["payment.py"], hours_offset=2, is_bugfix=True),
        _make_commit("c102", "fix: payment zero division", ["payment.py"], hours_offset=4, is_bugfix=True),
        _make_commit("c103", "feat: logger update", ["logger.py"], hours_offset=5)
    ]
    base = datetime(2025, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
    generator = LabelGenerator(commits)

    labels, summary = generator.generate_labels(
        filepaths=["payment.py", "logger.py"],
        window_start=base,
        allow_proxy=False
    )

    assert labels["payment.py"] == 1
    assert labels["logger.py"] == 0
    assert set(summary.component_provenance["payment.py"]) == {"c101", "c102"}
    assert summary.component_provenance["logger.py"] == []


def test_explicit_proxy_labeling_fallback():
    """Verifies that when no bugfix commits exist, proxy labeling is explicitly documented."""
    commits = [
        _make_commit("c1", "feat: heavy churn in payment", ["payment.py"], hours_offset=2),
        _make_commit("c2", "chore: minor doc in auth", ["auth.py"], hours_offset=4)
    ]
    base = datetime(2025, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
    generator = LabelGenerator(commits)

    labels, summary = generator.generate_labels(
        filepaths=["payment.py", "auth.py"],
        window_start=base,
        allow_proxy=True
    )

    # Must be explicitly identified as proxy
    assert summary.is_proxy is True
    assert summary.label_type == TargetLabelType.CHURN_PROXY
    assert summary.label_method == LabelMethod.CHURN_PROXY
    assert "Proxy label" in summary.description


def test_insufficient_data_honestly_reported_when_no_proxy_allowed():
    """Verifies that when proxy fallback is disabled, insufficient bug data is reported honestly without fabrication."""
    commits = [
        _make_commit("c1", "feat: regular commit", ["payment.py"], hours_offset=2),
    ]
    base = datetime(2025, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
    generator = LabelGenerator(commits)

    labels, summary = generator.generate_labels(
        filepaths=["payment.py"],
        window_start=base,
        allow_proxy=False
    )

    assert summary.label_type == TargetLabelType.INSUFFICIENT_DATA
    assert summary.label_method == LabelMethod.INSUFFICIENT_DATA
    assert labels["payment.py"] == 0
    assert summary.positive_count == 0
