"""
SoftwarePulse Explainability Package.
Provides global model feature importance and component-level prediction explanations.
"""
from .explainer import ModelExplainer, FeatureSignal, ComponentExplanation

__all__ = ["ModelExplainer", "FeatureSignal", "ComponentExplanation"]
