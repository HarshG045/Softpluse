"""
TemporalSplitter: Implements temporal / chronological train-test splitting to prevent data leakage.
Enforces strict separation between training observation periods, training future label horizons,
testing observation periods, and testing future label horizons.
"""
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple
import pandas as pd
import numpy as np

from features.feature_schema import FULL_FEATURES
from dataset.labeling import LabelSummary

logger = logging.getLogger(__name__)


def _to_utc(dt: Optional[datetime]) -> Optional[datetime]:
    if dt is None:
        return None
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


class TemporalLeakageError(ValueError):
    """Raised when temporal leakage is detected across observation or label windows."""
    pass


class InsufficientDataError(RuntimeError):
    """Raised when repository history or class distribution is insufficient for valid evaluation."""
    pass


@dataclass
class TemporalConfig:
    """Configuration for chronological observation timestamps and prediction horizons."""
    train_ratio: float = 0.70
    prediction_horizon_commits: Optional[int] = None
    prediction_horizon_days: Optional[float] = None
    allow_proxy_labels: bool = True
    strict_validation: bool = True


@dataclass
class SplitResult:
    """Stores train and test sets along with split point and horizon information."""
    X_train: pd.DataFrame
    y_train: pd.Series
    X_test: pd.DataFrame
    y_test: pd.Series
    train_components: List[str]
    test_components: List[str]
    split_timestamp: Optional[datetime] = None
    train_observation_time: Optional[datetime] = None
    train_label_window: Optional[Tuple[Optional[datetime], Optional[datetime]]] = None
    test_observation_time: Optional[datetime] = None
    test_label_window: Optional[Tuple[Optional[datetime], Optional[datetime]]] = None
    train_label_summary: Optional[LabelSummary] = None
    test_label_summary: Optional[LabelSummary] = None
    train_samples_count: int = 0
    test_samples_count: int = 0
    positive_train_count: int = 0
    positive_test_count: int = 0
    is_leakage_free: bool = True
    temporal_validation_errors: List[str] = field(default_factory=list)


class TemporalSplitter:
    """Performs chronological dataset splitting and enforces temporal non-overlap guarantees."""

    @staticmethod
    def validate_temporal_boundaries(
        train_obs_time: Optional[datetime],
        train_label_start: Optional[datetime],
        train_label_end: Optional[datetime],
        test_obs_time: Optional[datetime],
        test_label_start: Optional[datetime],
        test_label_end: Optional[datetime],
        strict: bool = True
    ) -> List[str]:
        """
        Validates the temporal ordering of observation and label windows:
        1. train_obs_time <= train_label_start
        2. train_label_start < train_label_end (if both present)
        3. train_label_end <= test_obs_time (no train label info in test features)
        4. train_label_end <= test_label_start (no overlap between train & test label windows)
        5. test_obs_time <= test_label_start
        6. test_label_start < test_label_end (if both present)

        Returns list of violation strings. Raises TemporalLeakageError if strict=True and errors exist.
        """
        errors = []
        t_obs_train = _to_utc(train_obs_time)
        t_lbl_train_start = _to_utc(train_label_start)
        t_lbl_train_end = _to_utc(train_label_end)
        t_obs_test = _to_utc(test_obs_time)
        t_lbl_test_start = _to_utc(test_label_start)
        t_lbl_test_end = _to_utc(test_label_end)

        if t_obs_train and t_lbl_train_start and t_obs_train > t_lbl_train_start:
            errors.append(f"Training observation time ({t_obs_train}) occurs after training label start ({t_lbl_train_start}).")

        if t_lbl_train_start and t_lbl_train_end and t_lbl_train_start >= t_lbl_train_end:
            errors.append(f"Training label start ({t_lbl_train_start}) is not strictly before training label end ({t_lbl_train_end}).")

        if t_lbl_train_end and t_obs_test and t_lbl_train_end > t_obs_test:
            errors.append(f"Training label window ends ({t_lbl_train_end}) after test observation time ({t_obs_test}), causing temporal leakage.")

        if t_lbl_train_end and t_lbl_test_start and t_lbl_train_end > t_lbl_test_start:
            errors.append(f"Training label window ({t_lbl_train_start} to {t_lbl_train_end}) overlaps with test label window start ({t_lbl_test_start}).")

        if t_obs_test and t_lbl_test_start and t_obs_test > t_lbl_test_start:
            errors.append(f"Test observation time ({t_obs_test}) occurs after test label start ({t_lbl_test_start}).")

        if t_lbl_test_start and t_lbl_test_end and t_lbl_test_start >= t_lbl_test_end:
            errors.append(f"Test label start ({t_lbl_test_start}) is not strictly before test label end ({t_lbl_test_end}).")

        if strict and errors:
            raise TemporalLeakageError("Temporal boundary validation failed:\n" + "\n".join(f"- {e}" for e in errors))

        return errors

    @staticmethod
    def validate_no_target_in_features(X: pd.DataFrame, target_col_names: Optional[List[str]] = None):
        """Ensures target labels or bugfix indicators are not present in feature columns."""
        forbidden = set(target_col_names or ["target", "label", "is_bugfix", "bugfix_future", "churn_proxy", "risk"])
        found = forbidden.intersection(set(X.columns))
        if found:
            raise TemporalLeakageError(f"Target/ground-truth columns leaked into feature matrix: {found}")

    @staticmethod
    def split_by_time_cutoff(
        df_train_features: pd.DataFrame,
        train_labels: Dict[str, int],
        df_test_features: pd.DataFrame,
        test_labels: Dict[str, int],
        split_time: Optional[datetime] = None,
        train_obs_time: Optional[datetime] = None,
        train_label_window: Optional[Tuple[Optional[datetime], Optional[datetime]]] = None,
        test_obs_time: Optional[datetime] = None,
        test_label_window: Optional[Tuple[Optional[datetime], Optional[datetime]]] = None,
        train_label_summary: Optional[LabelSummary] = None,
        test_label_summary: Optional[LabelSummary] = None,
        strict: bool = False
    ) -> SplitResult:
        """
        Creates split matrices from historical train features and evaluated test features.
        Preserves chronological ordering without random shuffling.
        """
        train_fps = [fp for fp in df_train_features["filepath"] if fp in train_labels]
        test_fps = [fp for fp in df_test_features["filepath"] if fp in test_labels]

        train_sub = df_train_features[df_train_features["filepath"].isin(train_fps)].copy()
        test_sub = df_test_features[df_test_features["filepath"].isin(test_fps)].copy()

        y_train = pd.Series([train_labels[fp] for fp in train_sub["filepath"]], index=train_sub.index)
        y_test = pd.Series([test_labels[fp] for fp in test_sub["filepath"]], index=test_sub.index)

        feature_cols = [c for c in FULL_FEATURES if c in df_train_features.columns]
        X_train = train_sub[feature_cols].copy()
        X_test = test_sub[feature_cols].copy()

        # Check for target leakage
        TemporalSplitter.validate_no_target_in_features(X_train)
        TemporalSplitter.validate_no_target_in_features(X_test)

        # Validate temporal boundaries if window timestamps are provided
        validation_errors = []
        if train_label_window and test_label_window:
            validation_errors = TemporalSplitter.validate_temporal_boundaries(
                train_obs_time=train_obs_time or split_time,
                train_label_start=train_label_window[0],
                train_label_end=train_label_window[1],
                test_obs_time=test_obs_time,
                test_label_start=test_label_window[0],
                test_label_end=test_label_window[1],
                strict=strict
            )

        return SplitResult(
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test,
            train_components=list(train_sub["filepath"]),
            test_components=list(test_sub["filepath"]),
            split_timestamp=split_time,
            train_observation_time=train_obs_time or split_time,
            train_label_window=train_label_window,
            test_observation_time=test_obs_time,
            test_label_window=test_label_window,
            train_label_summary=train_label_summary,
            test_label_summary=test_label_summary,
            train_samples_count=len(X_train),
            test_samples_count=len(X_test),
            positive_train_count=int(y_train.sum()),
            positive_test_count=int(y_test.sum()),
            is_leakage_free=(len(validation_errors) == 0),
            temporal_validation_errors=validation_errors
        )
