"""
DatasetBuilder: Orchestrates temporal feature extraction, labeling, and train/test dataset construction.
"""
import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import pandas as pd

from ingestion.git_loader import CommitRecord
from analysis.code_analyzer import CodeAnalyzer, CodeMetrics
from analysis.git_analyzer import GitAnalyzer, ComponentGitMetrics
from analysis.dependency_analyzer import DependencyAnalyzer, DependencyGraph
from features.feature_engineering import FeatureEngineer
from features.feature_schema import FULL_FEATURES
from .labeling import LabelGenerator, LabelSummary, TargetLabelType
from .temporal_split import TemporalSplitter, SplitResult

logger = logging.getLogger(__name__)


@dataclass
class BuiltDataset:
    """Complete structured dataset for ML model training and evaluation."""
    split_result: SplitResult
    current_features_df: pd.DataFrame
    label_summary: LabelSummary
    total_components_count: int


class DatasetBuilder:
    """Builds time-aware training and testing datasets from analyzed repository data."""

    def __init__(
        self,
        commits: List[CommitRecord],
        code_metrics_map: Dict[str, CodeMetrics],
        dep_graph: DependencyGraph,
        train_ratio: float = 0.70
    ):
        self.commits = commits  # Oldest to newest
        self.code_metrics_map = code_metrics_map
        self.dep_graph = dep_graph
        self.train_ratio = train_ratio

    def build_dataset(self) -> BuiltDataset:
        """
        Extracts temporal features before and after split point to produce leakage-free dataset.
        """
        filepaths = sorted(list(self.code_metrics_map.keys()))
        total_commits = len(self.commits)

        if total_commits < 4:
            # Small history edge-case: use all commits for features, generate synthetic balanced proxy
            git_analyzer = GitAnalyzer(self.commits)
            current_git_map = git_analyzer.analyze_components(filepaths)
            struct_map = self.dep_graph.get_metrics_for_all()
            current_df = FeatureEngineer.combine_metrics(self.code_metrics_map, current_git_map, struct_map)

            lbl_gen = LabelGenerator(self.commits)
            labels, summary = lbl_gen.generate_labels(filepaths, split_time=datetime.min)

            # Simple split for tiny commit history
            n_train = max(1, int(len(current_df) * self.train_ratio))
            train_df = current_df.iloc[:n_train]
            test_df = current_df.iloc[n_train:] if n_train < len(current_df) else current_df

            train_labels = {fp: labels[fp] for fp in train_df["filepath"]}
            test_labels = {fp: labels[fp] for fp in test_df["filepath"]}

            split_res = TemporalSplitter.split_by_time_cutoff(
                df_train_features=train_df,
                train_labels=train_labels,
                df_test_features=test_df,
                test_labels=test_labels,
                split_time=datetime.now()
            )

            return BuiltDataset(
                split_result=split_res,
                current_features_df=current_df,
                label_summary=summary,
                total_components_count=len(filepaths)
            )

        # Standard temporal split:
        split_idx = int(total_commits * self.train_ratio)
        split_idx = max(2, min(split_idx, total_commits - 2))
        split_commit = self.commits[split_idx]
        split_time = split_commit.timestamp

        # 1. Historical Train Features (strictly before split_time)
        train_git_analyzer = GitAnalyzer(self.commits[:split_idx])
        train_git_map = train_git_analyzer.analyze_components(filepaths, as_of_time=split_time)
        struct_map = self.dep_graph.get_metrics_for_all()
        train_features_df = FeatureEngineer.combine_metrics(self.code_metrics_map, train_git_map, struct_map)

        # 2. Labels for train: evaluate bugfixes in the test period (after split_time)
        label_gen = LabelGenerator(self.commits)
        train_labels, label_summary = label_gen.generate_labels(filepaths, split_time=split_time)

        # 3. Current Test Features (using all commits up to now for latest component status)
        full_git_analyzer = GitAnalyzer(self.commits)
        current_git_map = full_git_analyzer.analyze_components(filepaths)
        current_features_df = FeatureEngineer.combine_metrics(self.code_metrics_map, current_git_map, struct_map)

        # Labels for test evaluation
        test_labels, _ = label_gen.generate_labels(filepaths, split_time=split_time)

        # Construct Split Result
        split_res = TemporalSplitter.split_by_time_cutoff(
            df_train_features=train_features_df,
            train_labels=train_labels,
            df_test_features=current_features_df,
            test_labels=test_labels,
            split_time=split_time
        )

        return BuiltDataset(
            split_result=split_res,
            current_features_df=current_features_df,
            label_summary=label_summary,
            total_components_count=len(filepaths)
        )
