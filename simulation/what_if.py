"""
WhatIfEngine: Simulates hypothetical structural, complexity, and churn modifications on component risk.
"""
import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple
import networkx as nx
import numpy as np
import pandas as pd

from analysis.dependency_analyzer import DependencyGraph
from ml.predict import RiskPredictor, ComponentRiskProfile
from ml.train import TrainedModelBundle

logger = logging.getLogger(__name__)


class SimulationChangeType(str, Enum):
    ADD_DEPENDENCY = "Add Dependency"
    REMOVE_DEPENDENCY = "Remove Dependency"
    INCREASE_COMPLEXITY = "Increase Complexity"
    DECREASE_COMPLEXITY = "Decrease Complexity"
    INCREASE_CHURN = "Increase Churn"
    INCREASE_CHANGE_FREQUENCY = "Increase Change Frequency"


@dataclass
class ChangedFeatureDiff:
    """Represents the before/after delta of a modified feature."""
    feature_name: str
    display_name: str
    before_value: float
    after_value: float
    unit: str = ""


@dataclass
class SimulationResult:
    """Encapsulates the complete before/after analysis of a simulated what-if change."""
    target_component: str
    change_type: SimulationChangeType
    target_dependency: Optional[str]
    current_probability: float
    current_percentage: int
    current_category: str
    simulated_probability: float
    simulated_percentage: int
    simulated_category: str
    percentage_point_diff: int
    changed_features: List[ChangedFeatureDiff]
    affected_neighbors: List[str]
    simulated_graph: DependencyGraph
    simulated_edge_action: Optional[Tuple[str, str, str]]  # (source, target, "add"|"remove")
    scientific_disclaimer: str = (
        "This simulation modifies the model's feature representation and estimates how the "
        "prediction changes. It should be interpreted as a what-if analysis, not proof of causal impact."
    )


class WhatIfEngine:
    """Executes counterfactual what-if simulations on components using trained risk models."""

    def __init__(self, model_bundle: TrainedModelBundle, base_graph: DependencyGraph):
        self.bundle = model_bundle
        self.base_graph = base_graph
        self.predictor = RiskPredictor()

    def simulate_change(
        self,
        current_profile: ComponentRiskProfile,
        change_type: SimulationChangeType,
        target_dependency: Optional[str] = None,
        complexity_delta: int = 5,
        churn_delta: int = 300,
        freq_delta: float = 0.15
    ) -> SimulationResult:
        """
        Executes hypothetical change on the component, recomputes network graph metrics,
        and evaluates the new feature representation with the trained ML pipeline.
        """
        sim_graph = self.base_graph.clone()
        src = current_profile.filepath
        changed_diffs: List[ChangedFeatureDiff] = []
        affected_neighbors: List[str] = []
        sim_edge_action = None

        # Build modified feature dictionary
        mod_features = current_profile.to_dict()

        # Handle Dependency Mutations
        if change_type == SimulationChangeType.ADD_DEPENDENCY and target_dependency:
            sim_graph.add_dependency(src, target_dependency)
            sim_edge_action = (src, target_dependency, "add")
            affected_neighbors = [target_dependency]

            # Recalculate graph metrics
            struct_metrics = sim_graph.get_metrics_for_all()
            new_s = struct_metrics.get(src)

            if new_s:
                old_deps = current_profile.dependency_count
                old_cent = current_profile.centrality
                old_deg = current_profile.degree

                mod_features["dependency_count"] = new_s.dependency_count
                mod_features["out_degree"] = new_s.out_degree
                mod_features["degree"] = new_s.degree
                mod_features["centrality"] = new_s.centrality

                changed_diffs.append(ChangedFeatureDiff(
                    feature_name="dependency_count",
                    display_name="Dependency Count",
                    before_value=old_deps,
                    after_value=new_s.dependency_count
                ))
                changed_diffs.append(ChangedFeatureDiff(
                    feature_name="centrality",
                    display_name="Degree Centrality",
                    before_value=round(old_cent, 4),
                    after_value=round(new_s.centrality, 4)
                ))
                changed_diffs.append(ChangedFeatureDiff(
                    feature_name="degree",
                    display_name="Total Degree Connectivity",
                    before_value=old_deg,
                    after_value=new_s.degree
                ))

        elif change_type == SimulationChangeType.REMOVE_DEPENDENCY and target_dependency:
            sim_graph.remove_dependency(src, target_dependency)
            sim_edge_action = (src, target_dependency, "remove")
            affected_neighbors = [target_dependency]

            struct_metrics = sim_graph.get_metrics_for_all()
            new_s = struct_metrics.get(src)

            if new_s:
                old_deps = current_profile.dependency_count
                old_cent = current_profile.centrality
                old_deg = current_profile.degree

                mod_features["dependency_count"] = new_s.dependency_count
                mod_features["out_degree"] = new_s.out_degree
                mod_features["degree"] = new_s.degree
                mod_features["centrality"] = new_s.centrality

                changed_diffs.append(ChangedFeatureDiff(
                    feature_name="dependency_count",
                    display_name="Dependency Count",
                    before_value=old_deps,
                    after_value=new_s.dependency_count
                ))
                changed_diffs.append(ChangedFeatureDiff(
                    feature_name="centrality",
                    display_name="Degree Centrality",
                    before_value=round(old_cent, 4),
                    after_value=round(new_s.centrality, 4)
                ))

        elif change_type == SimulationChangeType.INCREASE_COMPLEXITY:
            old_c = current_profile.complexity
            new_c = old_c + complexity_delta
            mod_features["complexity"] = new_c
            changed_diffs.append(ChangedFeatureDiff(
                feature_name="complexity",
                display_name="Cyclomatic Complexity",
                before_value=old_c,
                after_value=new_c
            ))

        elif change_type == SimulationChangeType.DECREASE_COMPLEXITY:
            old_c = current_profile.complexity
            new_c = max(1, old_c - complexity_delta)
            mod_features["complexity"] = new_c
            changed_diffs.append(ChangedFeatureDiff(
                feature_name="complexity",
                display_name="Cyclomatic Complexity",
                before_value=old_c,
                after_value=new_c
            ))

        elif change_type == SimulationChangeType.INCREASE_CHURN:
            old_churn = current_profile.churn
            old_r_churn = current_profile.recent_churn
            new_churn = old_churn + churn_delta
            new_r_churn = old_r_churn + churn_delta
            mod_features["churn"] = new_churn
            mod_features["recent_churn"] = new_r_churn
            changed_diffs.append(ChangedFeatureDiff(
                feature_name="churn",
                display_name="Total Churn",
                before_value=old_churn,
                after_value=new_churn,
                unit="lines"
            ))
            changed_diffs.append(ChangedFeatureDiff(
                feature_name="recent_churn",
                display_name="Recent Churn",
                before_value=old_r_churn,
                after_value=new_r_churn,
                unit="lines"
            ))

        elif change_type == SimulationChangeType.INCREASE_CHANGE_FREQUENCY:
            old_f = current_profile.change_frequency
            new_f = min(1.0, old_f + freq_delta)
            mod_features["change_frequency"] = new_f
            changed_diffs.append(ChangedFeatureDiff(
                feature_name="change_frequency",
                display_name="Commit Change Frequency",
                before_value=round(old_f, 3),
                after_value=round(new_f, 3)
            ))

        # Re-run ML Prediction on Modified Vector
        df_mod = pd.DataFrame([mod_features])
        pred_res = self.predictor.predict(self.bundle, df_mod)
        sim_profile = pred_res.profiles[0]

        curr_prob = current_profile.risk_probability
        curr_pct = current_profile.risk_percentage
        sim_prob = sim_profile.risk_probability
        sim_pct = sim_profile.risk_percentage
        pct_diff = sim_pct - curr_pct

        return SimulationResult(
            target_component=src,
            change_type=change_type,
            target_dependency=target_dependency,
            current_probability=curr_prob,
            current_percentage=curr_pct,
            current_category=current_profile.risk_category,
            simulated_probability=sim_prob,
            simulated_percentage=sim_pct,
            simulated_category=sim_profile.risk_category,
            percentage_point_diff=pct_diff,
            changed_features=changed_diffs,
            affected_neighbors=affected_neighbors,
            simulated_graph=sim_graph,
            simulated_edge_action=sim_edge_action
        )
