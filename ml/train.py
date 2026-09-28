"""
ModelTrainer: Trains classifiers (Random Forest, Logistic Regression) on different feature configurations.
"""
import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

from config import RANDOM_STATE
from features.feature_schema import (
    FeatureSetType,
    FEATURE_SET_MAP,
    STATIC_FEATURES,
    STATIC_PLUS_EVOLUTION_FEATURES,
    FULL_FEATURES
)

logger = logging.getLogger(__name__)


class ClassifierAlgorithm(str, Enum):
    RANDOM_FOREST = "Random Forest"
    LOGISTIC_REGRESSION = "Logistic Regression"


@dataclass
class TrainedModelBundle:
    """Encapsulates a trained pipeline, feature list, hyperparams, and metadata."""
    model_name: str
    algorithm: ClassifierAlgorithm
    feature_set_type: FeatureSetType
    feature_names: List[str]
    pipeline: Any
    training_timestamp: datetime = field(default_factory=datetime.now)
    training_samples: int = 0
    positive_samples: int = 0
    negative_samples: int = 0


class ModelTrainer:
    """Trains machine learning risk prediction models."""

    def __init__(self, random_state: int = RANDOM_STATE):
        self.random_state = random_state

    def train_model(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        algorithm: ClassifierAlgorithm = ClassifierAlgorithm.RANDOM_FOREST,
        feature_set_type: FeatureSetType = FeatureSetType.FULL
    ) -> TrainedModelBundle:
        """
        Trains a pipeline for the requested algorithm and feature subset.
        """
        target_features = FEATURE_SET_MAP[feature_set_type]
        valid_features = [f for f in target_features if f in X_train.columns]
        
        X_sub = X_train[valid_features].copy()
        y_sub = y_train.copy()

        # Fallback if y_sub only has 1 class
        unique_classes = np.unique(y_sub)
        if len(unique_classes) < 2:
            # Add synthetic balanced variation if dataset is degenerate
            logger.warning("Training set has only 1 class. Applying smoothing for stability.")
            y_sub.iloc[0] = 1 if unique_classes[0] == 0 else 0

        # Construct Scikit-Learn Pipeline
        if algorithm == ClassifierAlgorithm.RANDOM_FOREST:
            clf = RandomForestClassifier(
                n_estimators=100,
                max_depth=6,
                min_samples_split=2,
                class_weight="balanced",
                random_state=self.random_state
            )
            pipeline = Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("classifier", clf)
            ])
        else:
            clf = LogisticRegression(
                max_iter=1000,
                class_weight="balanced",
                random_state=self.random_state
            )
            pipeline = Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("classifier", clf)
            ])

        pipeline.fit(X_sub, y_sub)

        return TrainedModelBundle(
            model_name=f"{algorithm.value} ({feature_set_type.value})",
            algorithm=algorithm,
            feature_set_type=feature_set_type,
            feature_names=valid_features,
            pipeline=pipeline,
            training_timestamp=datetime.now(),
            training_samples=len(X_sub),
            positive_samples=int((y_sub == 1).sum()),
            negative_samples=int((y_sub == 0).sum())
        )
