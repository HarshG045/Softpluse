"""
ModelExplainer: Computes global feature importances and local component-level risk attribution signals.
"""
import logging
from dataclasses import dataclass
from typing import Dict, List, Optional
import numpy as np
import pandas as pd

from features.feature_schema import FEATURE_DESCRIPTIONS
from ml.train import ClassifierAlgorithm, TrainedModelBundle

logger = logging.getLogger(__name__)


@dataclass
class FeatureSignal:
    """Represents the contribution signal of a specific feature for a component."""
    feature_name: str
    display_name: str
    feature_value: float
    importance_weight: float  # Normalized 0.0 to 1.0
    impact_level: str         # HIGH / MEDIUM / LOW
    description: str


@dataclass
class ComponentExplanation:
    """Explainability breakdown for a single component's risk prediction."""
    filepath: str
    predicted_probability: float
    risk_category: str
    top_signals: List[FeatureSignal]
    all_signals: List[FeatureSignal]
    summary_text: str


class ModelExplainer:
    """Extracts interpretable risk explanations from trained models."""

    def __init__(self, model_bundle: TrainedModelBundle, background_data: Optional[pd.DataFrame] = None):
        self.bundle = model_bundle
        self.background_data = background_data
        self.global_importances = self._compute_global_importances()

    def _compute_global_importances(self) -> Dict[str, float]:
        """Extracts feature importances or linear coefficients from the trained pipeline."""
        pipeline = self.bundle.pipeline
        feature_names = self.bundle.feature_names

        try:
            clf = pipeline.named_steps.get("classifier", pipeline)
            if hasattr(clf, "feature_importances_"):
                raw_imp = clf.feature_importances_
            elif hasattr(clf, "coef_"):
                raw_imp = np.abs(clf.coef_[0])
            else:
                raw_imp = np.ones(len(feature_names)) / max(1, len(feature_names))

            # Normalize to sum = 1.0
            total = np.sum(raw_imp)
            norm_imp = (raw_imp / total) if total > 0 else np.ones(len(raw_imp)) / len(raw_imp)
            return dict(zip(feature_names, [float(x) for x in norm_imp]))
        except Exception as e:
            logger.warning(f"Failed to extract model importances: {e}")
            equal_val = 1.0 / max(1, len(feature_names))
            return {f: equal_val for f in feature_names}

    def get_global_importance_df(self) -> pd.DataFrame:
        """Returns global feature importances formatted as a sorted DataFrame."""
        rows = []
        for feat, imp in sorted(self.global_importances.items(), key=lambda x: x[1], reverse=True):
            clean_name = feat.replace("_", " ").title()
            rows.append({
                "Feature": clean_name,
                "FeatureKey": feat,
                "Importance": round(imp, 4),
                "Description": FEATURE_DESCRIPTIONS.get(feat, "")
            })
        return pd.DataFrame(rows)

    def explain_component(
        self,
        component_row: pd.Series,
        predicted_prob: float,
        risk_category: str
    ) -> ComponentExplanation:
        """
        Computes localized risk signals for an individual component.
        Combines model feature importance with component's feature magnitude.
        """
        signals: List[FeatureSignal] = []
        filepath = str(component_row.get("filepath", "unknown"))

        for feat in self.bundle.feature_names:
            val = float(component_row.get(feat, 0.0))
            global_imp = self.global_importances.get(feat, 0.0)

            # Local signal is product of importance and relative value magnitude
            # High values with high importance create high impact
            if self.background_data is not None and feat in self.background_data.columns:
                col_max = self.background_data[feat].max()
                norm_val = (val / col_max) if col_max > 0 else 0.5
            else:
                norm_val = min(1.0, val / 50.0) if val > 0 else 0.1

            impact_score = global_imp * (0.3 + 0.7 * norm_val)

            if impact_score >= 0.08:
                impact_level = "HIGH"
            elif impact_score >= 0.04:
                impact_level = "MEDIUM"
            else:
                impact_level = "LOW"

            clean_name = feat.replace("_", " ").title()
            signal = FeatureSignal(
                feature_name=feat,
                display_name=clean_name,
                feature_value=val,
                importance_weight=float(impact_score),
                impact_level=impact_level,
                description=FEATURE_DESCRIPTIONS.get(feat, "")
            )
            signals.append(signal)

        # Sort signals by impact score
        signals.sort(key=lambda s: s.importance_weight, reverse=True)
        top_signals = signals[:5]

        # Generate readable explanation summary
        top_names = [s.display_name for s in top_signals if s.impact_level in ("HIGH", "MEDIUM")][:3]
        if top_names:
            summary = (
                f"The model identifies {', '.join(top_names)} as the primary signals "
                f"contributing to this component's {risk_category} predicted risk profile."
            )
        else:
            summary = f"Risk prediction is distributed evenly across baseline characteristics."

        return ComponentExplanation(
            filepath=filepath,
            predicted_probability=predicted_prob,
            risk_category=risk_category,
            top_signals=top_signals,
            all_signals=signals,
            summary_text=summary
        )
