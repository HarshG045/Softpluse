"""
SoftwarePulse Features Package.
Defines feature schemas, feature groups, and builds feature matrices for ML prediction.
"""
from .feature_schema import (
    STATIC_FEATURES,
    EVOLUTION_FEATURES,
    STRUCTURAL_FEATURES,
    STATIC_PLUS_EVOLUTION_FEATURES,
    FULL_FEATURES,
    FeatureSetType
)
from .feature_engineering import FeatureEngineer, ComponentFeatureRow

__all__ = [
    "STATIC_FEATURES",
    "EVOLUTION_FEATURES",
    "STRUCTURAL_FEATURES",
    "STATIC_PLUS_EVOLUTION_FEATURES",
    "FULL_FEATURES",
    "FeatureSetType",
    "FeatureEngineer",
    "ComponentFeatureRow"
]
