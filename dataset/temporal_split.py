"""
TemporalSplitter: Implements temporal / chronological train-test splitting to prevent data leakage.
"""
import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import pandas as pd
import numpy as np

from features.feature_schema import FULL_FEATURES

logger = logging.getLogger(__name__)


@dataclass
class SplitResult:
    """Stores train and test sets along with split point information."""
    X_train: pd.DataFrame
    y_train: pd.Series
    X_test: pd.DataFrame
    y_test: pd.Series
    train_components: List[str]
    test_components: List[str]
    split_timestamp: Optional[datetime] = None
    train_samples_count: int = 0
    test_samples_count: int = 0
    positive_train_count: int = 0
    positive_test_count: int = 0


class TemporalSplitter:
    """Performs chronological dataset splitting."""

    @staticmethod
    def split_by_time_cutoff(
        df_train_features: pd.DataFrame,
        train_labels: Dict[str, int],
        df_test_features: pd.DataFrame,
        test_labels: Dict[str, int],
        split_time: Optional[datetime] = None
    ) -> SplitResult:
        """
        Creates split matrices from historical train features and evaluated test features.
        """
        train_fps = [fp for fp in df_train_features["filepath"] if fp in train_labels]
        test_fps = [fp for fp in df_test_features["filepath"] if fp in test_labels]

        train_sub = df_train_features[df_train_features["filepath"].isin(train_fps)].copy()
        test_sub = df_test_features[df_test_features["filepath"].isin(test_fps)].copy()

        y_train = pd.Series([train_labels[fp] for fp in train_sub["filepath"]], index=train_sub.index)
        y_test = pd.Series([test_labels[fp] for fp in test_sub["filepath"]], index=test_sub.index)

        X_train = train_sub[FULL_FEATURES].copy()
        X_test = test_sub[FULL_FEATURES].copy()

        return SplitResult(
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test,
            train_components=list(train_sub["filepath"]),
            test_components=list(test_sub["filepath"]),
            split_timestamp=split_time,
            train_samples_count=len(X_train),
            test_samples_count=len(X_test),
            positive_train_count=int(y_train.sum()),
            positive_test_count=int(y_test.sum())
        )
