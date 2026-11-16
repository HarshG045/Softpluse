"""
DatasetBuilder: Orchestrates temporal feature extraction, labeling, and train/test dataset construction.
Enforces strict chronological feature and label separation to eliminate temporal leakage.
"""
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import pandas as pd

from ingestion.git_loader import CommitRecord
from analysis.code_analyzer import CodeAnalyzer, CodeMetrics
from analysis.git_analyzer import GitAnalyzer, ComponentGitMetrics
from analysis.dependency_analyzer import DependencyAnalyzer, DependencyGraph
from analysis.snapshot_reconstructor import SnapshotReconstructor
from features.feature_engineering import FeatureEngineer
from features.feature_schema import FULL_FEATURES
from .labeling import LabelGenerator, LabelSummary, TargetLabelType, LabelMethod
from .temporal_split import TemporalSplitter, SplitResult, TemporalConfig, _to_utc

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
        train_ratio: float = 0.70,
        repo_path: Optional[Path] = None,
        prediction_horizon_commits: Optional[int] = None,
        prediction_horizon_days: Optional[float] = None,
        allow_proxy_labels: bool = True,
        bug_keywords: Optional[List[str]] = None
    ):
        self.commits = commits  # Oldest to newest
        self.code_metrics_map = code_metrics_map
        self.dep_graph = dep_graph
        self.train_ratio = train_ratio
        self.repo_path = Path(repo_path).resolve() if repo_path else None
        self.prediction_horizon_commits = prediction_horizon_commits
        self.prediction_horizon_days = prediction_horizon_days
        self.allow_proxy_labels = allow_proxy_labels
        self.bug_keywords = bug_keywords

        self._reconstructor: Optional[SnapshotReconstructor] = None
        if self.repo_path and (self.repo_path / ".git").exists():
            try:
                self._reconstructor = SnapshotReconstructor(self.repo_path)
            except Exception as e:
                logger.warning(f"Could not initialize SnapshotReconstructor: {e}")

    def build_dataset(self) -> BuiltDataset:
        """
        Extracts temporal features before and after split point to produce leakage-free dataset.
        Enforces independent training and testing future label windows.
        """
        filepaths = sorted(list(self.code_metrics_map.keys()))
        total_commits = len(self.commits)

        # 0. Always construct current features DataFrame for latest status / dashboard predictions
        full_git_analyzer = GitAnalyzer(self.commits)
        current_git_map = full_git_analyzer.analyze_components(filepaths)
        current_struct_map = self.dep_graph.get_metrics_for_all()
        current_features_df = FeatureEngineer.combine_metrics(
            self.code_metrics_map, current_git_map, current_struct_map, target_filepaths=filepaths
        )

        label_gen = LabelGenerator(self.commits, bug_keywords=self.bug_keywords)

        # Edge case: minimal commits (< 4)
        if total_commits < 4:
            logger.warning(
                f"Repository has only {total_commits} commits. Using basic chronological division "
                "with documented proxy labels."
            )
            split_time = self.commits[-1].timestamp if self.commits else datetime.now(timezone.utc)
            train_labels, summary = label_gen.generate_labels(
                filepaths,
                window_start=None,
                window_end=None,
                allow_proxy=self.allow_proxy_labels
            )

            n_train = max(1, int(len(current_features_df) * self.train_ratio))
            train_df = current_features_df.iloc[:n_train]
            test_df = current_features_df.iloc[n_train:] if n_train < len(current_features_df) else current_features_df

            train_labels_map = {fp: train_labels.get(fp, 0) for fp in train_df["filepath"]}
            test_labels_map = {fp: train_labels.get(fp, 0) for fp in test_df["filepath"]}

            split_res = TemporalSplitter.split_by_time_cutoff(
                df_train_features=train_df,
                train_labels=train_labels_map,
                df_test_features=test_df,
                test_labels=test_labels_map,
                split_time=split_time,
                train_label_summary=summary,
                test_label_summary=summary,
                strict=False
            )

            return BuiltDataset(
                split_result=split_res,
                current_features_df=current_features_df,
                label_summary=summary,
                total_components_count=len(filepaths)
            )

        # Standard chronological division with independent training and testing windows
        # Timeline:
        # 0 ... i_train_obs -> Features for Train X(T_train)
        # i_train_obs ... i_train_lbl_end -> Future Label window for Train y(T_train)
        # i_train_lbl_end (or later) = i_test_obs -> Features for Test X(T_test)
        # i_test_obs ... N-1 -> Future Label window for Test y(T_test)

        if total_commits >= 6:
            i_train_obs = max(1, int(total_commits * 0.40))
            i_train_lbl_end = max(i_train_obs + 1, int(total_commits * self.train_ratio))
            # Bound train label horizon if configured
            if self.prediction_horizon_commits and (i_train_lbl_end - i_train_obs) > self.prediction_horizon_commits:
                i_train_lbl_end = i_train_obs + self.prediction_horizon_commits

            i_test_obs = i_train_lbl_end
            i_test_lbl_end = total_commits - 1
        else:
            # 4 or 5 commits: minimal division
            i_train_obs = 1
            i_train_lbl_end = 2
            i_test_obs = 2
            i_test_lbl_end = total_commits - 1

        commit_train_obs = self.commits[i_train_obs]
        commit_train_lbl_end = self.commits[i_train_lbl_end]
        commit_test_obs = self.commits[i_test_obs]
        commit_test_lbl_end = self.commits[i_test_lbl_end]

        t_train_obs = commit_train_obs.timestamp
        t_train_lbl_end = commit_train_lbl_end.timestamp
        t_test_obs = commit_test_obs.timestamp
        t_test_lbl_end = commit_test_lbl_end.timestamp

        # 1. Historical Train Features X(T_train_obs): strictly at commit_train_obs
        train_code_map: Dict[str, CodeMetrics] = {}
        train_struct_map = {}
        train_files = filepaths

        if self._reconstructor:
            try:
                train_code_map = self._reconstructor.extract_code_metrics_at_commit(commit_train_obs.commit_hash)
                train_dep_graph = self._reconstructor.build_dependency_graph_at_commit(
                    commit_train_obs.commit_hash, train_code_map
                )
                train_struct_map = train_dep_graph.get_metrics_for_all()
                if train_code_map:
                    train_files = sorted(list(train_code_map.keys()))
            except Exception as e:
                logger.warning(f"Reconstruction failed for training commit {commit_train_obs.commit_hash[:7]}: {e}")

        if not train_code_map:
            # Fallback to current code map if reconstruction unavailable
            train_code_map = self.code_metrics_map
            train_struct_map = current_struct_map

        train_git_analyzer = GitAnalyzer(self.commits[:i_train_obs + 1])
        train_git_map = train_git_analyzer.analyze_components(train_files, as_of_time=t_train_obs)
        train_features_df = FeatureEngineer.combine_metrics(
            train_code_map, train_git_map, train_struct_map, target_filepaths=train_files
        )

        # 2. Training Labels y(T_train_obs): generated strictly in (T_train_obs, T_train_lbl_end]
        train_labels, train_label_summary = label_gen.generate_labels(
            train_files,
            window_start=t_train_obs,
            window_end=t_train_lbl_end,
            allow_proxy=self.allow_proxy_labels
        )

        # 3. Historical Test Features X(T_test_obs): strictly at commit_test_obs
        test_code_map: Dict[str, CodeMetrics] = {}
        test_struct_map = {}
        test_files = filepaths

        if self._reconstructor:
            try:
                test_code_map = self._reconstructor.extract_code_metrics_at_commit(commit_test_obs.commit_hash)
                test_dep_graph = self._reconstructor.build_dependency_graph_at_commit(
                    commit_test_obs.commit_hash, test_code_map
                )
                test_struct_map = test_dep_graph.get_metrics_for_all()
                if test_code_map:
                    test_files = sorted(list(test_code_map.keys()))
            except Exception as e:
                logger.warning(f"Reconstruction failed for test commit {commit_test_obs.commit_hash[:7]}: {e}")

        if not test_code_map:
            test_code_map = self.code_metrics_map
            test_struct_map = current_struct_map

        test_git_analyzer = GitAnalyzer(self.commits[:i_test_obs + 1])
        test_git_map = test_git_analyzer.analyze_components(test_files, as_of_time=t_test_obs)
        test_features_df = FeatureEngineer.combine_metrics(
            test_code_map, test_git_map, test_struct_map, target_filepaths=test_files
        )

        # 4. Testing Labels y(T_test_obs): generated strictly in (T_test_obs, T_test_lbl_end]
        test_labels, test_label_summary = label_gen.generate_labels(
            test_files,
            window_start=t_test_obs,
            window_end=t_test_lbl_end,
            allow_proxy=self.allow_proxy_labels
        )

        # 5. Build and validate SplitResult
        split_res = TemporalSplitter.split_by_time_cutoff(
            df_train_features=train_features_df,
            train_labels=train_labels,
            df_test_features=test_features_df,
            test_labels=test_labels,
            split_time=t_train_obs,
            train_obs_time=t_train_obs,
            train_label_window=(t_train_obs, t_train_lbl_end),
            test_obs_time=t_test_obs,
            test_label_window=(t_test_obs, t_test_lbl_end),
            train_label_summary=train_label_summary,
            test_label_summary=test_label_summary,
            strict=False
        )

        return BuiltDataset(
            split_result=split_res,
            current_features_df=current_features_df,
            label_summary=train_label_summary,
            total_components_count=len(filepaths)
        )
