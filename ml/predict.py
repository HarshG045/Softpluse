"""
RiskPredictor: Infers risk probability and assigns risk category (LOW / MEDIUM / HIGH) for components.
"""
import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd

from config import RISK_THRESHOLDS
from .train import TrainedModelBundle

logger = logging.getLogger(__name__)


@dataclass
class ComponentRiskProfile:
    """Complete prediction profile for a software component."""
    filepath: str
    risk_probability: float
    risk_percentage: int
    risk_category: str  # LOW / MEDIUM / HIGH
    loc: int
    complexity: int
    functions: int
    classes: int
    imports: int
    methods: int
    commit_count: int
    recent_commit_count: int
    churn: int
    recent_churn: int
    change_frequency: float
    number_of_authors: int
    time_since_last_change: float
    dependency_count: int
    dependent_count: int
    degree: int
    centrality: float
    in_degree: int = 0
    out_degree: int = 0

    @property
    def component_name(self) -> str:
        return self.filepath.replace("\\", "/").split("/")[-1]

    def to_dict(self) -> Dict:
        return {
            "filepath": self.filepath,
            "risk_probability": round(self.risk_probability, 4),
            "risk_percentage": self.risk_percentage,
            "risk_category": self.risk_category,
            "loc": self.loc,
            "complexity": self.complexity,
            "functions": self.functions,
            "classes": self.classes,
            "imports": self.imports,
            "methods": self.methods,
            "commit_count": self.commit_count,
            "recent_commit_count": self.recent_commit_count,
            "churn": self.churn,
            "recent_churn": self.recent_churn,
            "change_frequency": round(self.change_frequency, 4),
            "number_of_authors": self.number_of_authors,
            "time_since_last_change": round(self.time_since_last_change, 1),
            "dependency_count": self.dependency_count,
            "dependent_count": self.dependent_count,
            "degree": self.degree,
            "centrality": round(self.centrality, 4),
            "in_degree": self.in_degree,
            "out_degree": self.out_degree
        }



@dataclass
class PredictionResult:
    """Encompasses risk predictions across all components."""
    profiles: List[ComponentRiskProfile]
    profiles_by_path: Dict[str, ComponentRiskProfile]
    df_predictions: pd.DataFrame
    high_risk_count: int
    medium_risk_count: int
    low_risk_count: int


class RiskPredictor:
    """Predicts component-level risk from feature representations using a trained model bundle."""

    def __init__(self, thresholds: Optional[Dict[str, Tuple[float, float]]] = None):
        self.thresholds = thresholds or RISK_THRESHOLDS

    def classify_risk(self, probability: float) -> str:
        """Assigns LOW / MEDIUM / HIGH risk category based on probability."""
        prob = max(0.0, min(1.0, probability))
        if prob >= self.thresholds["HIGH"][0]:
            return "HIGH"
        elif prob >= self.thresholds["MEDIUM"][0]:
            return "MEDIUM"
        else:
            return "LOW"

    def predict(self, model_bundle: TrainedModelBundle, features_df: pd.DataFrame) -> PredictionResult:
        """
        Runs model inference on the provided feature DataFrame.
        """
        if features_df.empty:
            return PredictionResult(
                profiles=[],
                profiles_by_path={},
                df_predictions=pd.DataFrame(),
                high_risk_count=0,
                medium_risk_count=0,
                low_risk_count=0
            )

        X = features_df[model_bundle.feature_names].copy()
        
        # Predict probabilities
        try:
            proba = model_bundle.pipeline.predict_proba(X)
            # Probability of positive class (risk)
            if proba.shape[1] > 1:
                risk_probs = proba[:, 1]
            else:
                risk_probs = proba[:, 0]
        except Exception as e:
            logger.warning(f"predict_proba failed ({e}), using decision function fallback")
            try:
                scores = model_bundle.pipeline.decision_function(X)
                # Sigmoid transform
                risk_probs = 1.0 / (1.0 + np.exp(-scores))
            except Exception:
                preds = model_bundle.pipeline.predict(X)
                risk_probs = np.array(preds, dtype=float)

        profiles: List[ComponentRiskProfile] = []
        profiles_by_path: Dict[str, ComponentRiskProfile] = {}
        high_c = 0
        med_c = 0
        low_c = 0

        for idx, row in features_df.iterrows():
            prob = float(risk_probs[idx])
            cat = self.classify_risk(prob)
            pct = int(round(prob * 100))

            if cat == "HIGH":
                high_c += 1
            elif cat == "MEDIUM":
                med_c += 1
            else:
                low_c += 1

            profile = ComponentRiskProfile(
                filepath=str(row["filepath"]),
                risk_probability=prob,
                risk_percentage=pct,
                risk_category=cat,
                loc=int(row.get("loc", 0)),
                complexity=int(row.get("complexity", 1)),
                functions=int(row.get("functions", 0)),
                classes=int(row.get("classes", 0)),
                imports=int(row.get("imports", 0)),
                methods=int(row.get("methods", 0)),
                commit_count=int(row.get("commit_count", 0)),
                recent_commit_count=int(row.get("recent_commit_count", 0)),
                churn=int(row.get("churn", 0)),
                recent_churn=int(row.get("recent_churn", 0)),
                change_frequency=float(row.get("change_frequency", 0.0)),
                number_of_authors=int(row.get("number_of_authors", 0)),
                time_since_last_change=float(row.get("time_since_last_change", 0.0)),
                dependency_count=int(row.get("dependency_count", 0)),
                dependent_count=int(row.get("dependent_count", 0)),
                degree=int(row.get("degree", 0)),
                centrality=float(row.get("centrality", 0.0)),
                in_degree=int(row.get("in_degree", 0)),
                out_degree=int(row.get("out_degree", 0))
            )
            profiles.append(profile)

            profiles_by_path[profile.filepath] = profile

        # Sort profiles by risk probability descending
        profiles.sort(key=lambda p: p.risk_probability, reverse=True)

        # Build DataFrame representation
        df_out = pd.DataFrame([p.to_dict() for p in profiles])

        return PredictionResult(
            profiles=profiles,
            profiles_by_path=profiles_by_path,
            df_predictions=df_out,
            high_risk_count=high_c,
            medium_risk_count=med_c,
            low_risk_count=low_c
        )
