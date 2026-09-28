"""
FeatureEngineering: Constructs and validates feature matrices across static, evolution, and structural groups.
"""
import logging
from dataclasses import dataclass
from typing import Dict, List, Optional
import pandas as pd
import numpy as np

from analysis.code_analyzer import CodeMetrics
from analysis.git_analyzer import ComponentGitMetrics
from analysis.dependency_analyzer import ComponentStructuralMetrics
from .feature_schema import (
    STATIC_FEATURES,
    EVOLUTION_FEATURES,
    STRUCTURAL_FEATURES,
    FULL_FEATURES,
    FeatureSetType,
    FEATURE_SET_MAP
)

logger = logging.getLogger(__name__)


@dataclass
class ComponentFeatureRow:
    """Consolidated representation of all features for a single component."""
    filepath: str
    loc: int = 0
    functions: int = 0
    classes: int = 0
    complexity: int = 1
    imports: int = 0
    methods: int = 0
    commit_count: int = 0
    recent_commit_count: int = 0
    churn: int = 0
    recent_churn: int = 0
    change_frequency: float = 0.0
    number_of_authors: int = 0
    time_since_last_change: float = 0.0
    dependency_count: int = 0
    dependent_count: int = 0
    degree: int = 0
    centrality: float = 0.0
    in_degree: int = 0
    out_degree: int = 0

    def to_dict(self) -> Dict:
        return {
            "filepath": self.filepath,
            "loc": self.loc,
            "functions": self.functions,
            "classes": self.classes,
            "complexity": self.complexity,
            "imports": self.imports,
            "methods": self.methods,
            "commit_count": self.commit_count,
            "recent_commit_count": self.recent_commit_count,
            "churn": self.churn,
            "recent_churn": self.recent_churn,
            "change_frequency": self.change_frequency,
            "number_of_authors": self.number_of_authors,
            "time_since_last_change": self.time_since_last_change,
            "dependency_count": self.dependency_count,
            "dependent_count": self.dependent_count,
            "degree": self.degree,
            "centrality": self.centrality,
            "in_degree": self.in_degree,
            "out_degree": self.out_degree
        }


class FeatureEngineer:
    """Builds consolidated feature matrices for components."""

    @staticmethod
    def combine_metrics(
        code_map: Dict[str, CodeMetrics],
        git_map: Dict[str, ComponentGitMetrics],
        struct_map: Dict[str, ComponentStructuralMetrics],
        target_filepaths: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Merges code, git, and structural metrics for all components into a pandas DataFrame.
        """
        if target_filepaths is not None:
            all_filepaths = set(target_filepaths)
        elif code_map:
            all_filepaths = set(code_map.keys())
        else:
            all_filepaths = set(code_map.keys()) | set(git_map.keys()) | set(struct_map.keys())

        rows: List[Dict] = []

        for fp in sorted(all_filepaths):
            c_m = code_map.get(fp, CodeMetrics(filepath=fp))
            g_m = git_map.get(fp, ComponentGitMetrics(filepath=fp))
            s_m = struct_map.get(fp, ComponentStructuralMetrics(filepath=fp))

            row = {
                "filepath": fp,
                # Static
                "loc": float(c_m.loc),
                "functions": float(c_m.functions),
                "classes": float(c_m.classes),
                "complexity": float(c_m.complexity),
                "imports": float(c_m.imports),
                "methods": float(c_m.methods),
                # Evolution
                "commit_count": float(g_m.commit_count),
                "recent_commit_count": float(g_m.recent_commit_count),
                "churn": float(g_m.churn),
                "recent_churn": float(g_m.recent_churn),
                "change_frequency": float(g_m.change_frequency),
                "number_of_authors": float(g_m.number_of_authors),
                "time_since_last_change": float(g_m.time_since_last_change),
                # Structural
                "dependency_count": float(s_m.dependency_count),
                "dependent_count": float(s_m.dependent_count),
                "degree": float(s_m.degree),
                "centrality": float(s_m.centrality),
                "in_degree": float(s_m.in_degree),
                "out_degree": float(s_m.out_degree)
            }
            rows.append(row)

        df = pd.DataFrame(rows)
        if df.empty:
            df = pd.DataFrame(columns=["filepath"] + FULL_FEATURES)
        
        # Fill missing values with reasonable defaults
        df[STATIC_FEATURES] = df[STATIC_FEATURES].fillna(0.0)
        df[EVOLUTION_FEATURES] = df[EVOLUTION_FEATURES].fillna(0.0)
        df[STRUCTURAL_FEATURES] = df[STRUCTURAL_FEATURES].fillna(0.0)

        return df

    @staticmethod
    def get_feature_subset(df: pd.DataFrame, feature_set_type: FeatureSetType) -> pd.DataFrame:
        """Returns only the columns corresponding to the specified feature set."""
        cols = FEATURE_SET_MAP[feature_set_type]
        available_cols = [c for c in cols if c in df.columns]
        return df[available_cols].copy()
