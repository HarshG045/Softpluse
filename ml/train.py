"""
ModelTrainer: Trains classifiers (Random Forest, Logistic Regression) on different feature configurations.
Prevents data leakage by encapsulating feature scaling and imputing inside pipeline objects.
Handles single-class edge cases transparently without fabricating target labels.
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
from sklearn.dummy import DummyClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

from config import RANDOM_STATE
from dataset.temporal_split import InsufficientDataError
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
    is_single_class: bool = False
    single_class_label: Optional[int] = None


class ModelTrainer:
    """Trains machine learning risk prediction models without temporal leakage or fabricated labels."""

    def __init__(self, random_state: int = RANDOM_STATE):
        self.random_state = random_state

    def train_model(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        algorithm: ClassifierAlgorithm = ClassifierAlgorithm.RANDOM_FOREST,
        feature_set_type: FeatureSetType = FeatureSetType.FULL,
        raise_on_single_class: bool = False
    ) -> TrainedModelBundle:
        """
        Trains a pipeline for the requested algorithm and feature subset.
        Scaling and preprocessing are fitted strictly on X_train.
        Single-class datasets are handled transparently without fabricating labels.
        """
        target_features = FEATURE_SET_MAP[feature_set_type]
        valid_features = [f for f in target_features if f in X_train.columns]

        X_sub = X_train[valid_features].copy()
        y_sub = y_train.copy()

        unique_classes = np.unique(y_sub)
        is_single_class = len(unique_classes) < 2
        single_class_label = int(unique_classes[0]) if is_single_class and len(unique_classes) > 0 else None

        if is_single_class:
            if raise_on_single_class:
                raise InsufficientDataError(
                    f"Training set has only 1 class ({single_class_label}). "
                    "Cannot train binary classifier without examples of both classes."
                )
            logger.warning(
                f"Training set has only 1 class ({single_class_label}). "
                "Fitting DummyClassifier baseline without fabricating false labels."
            )
            pipeline = Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("classifier", DummyClassifier(strategy="most_frequent"))
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
                negative_samples=int((y_sub == 0).sum()),
                is_single_class=True,
                single_class_label=single_class_label
            )

        # Standard multi-class training with leakage-safe pipeline
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
            negative_samples=int((y_sub == 0).sum()),
            is_single_class=False,
            single_class_label=None
        )
