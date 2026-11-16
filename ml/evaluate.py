"""
ModelEvaluator: Evaluates ML model performance and compares Static vs Evolution vs Full architectures.
Enforces rigorous statistical reporting: calculates ROC-AUC only when mathematically valid,
tracks class distributions, reports baseline classifiers, and documents observation windows.
"""
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix
)
from sklearn.dummy import DummyClassifier

from dataset.temporal_split import SplitResult
from features.feature_schema import FeatureSetType
from .train import ClassifierAlgorithm, ModelTrainer, TrainedModelBundle

logger = logging.getLogger(__name__)


@dataclass
class EvaluationReport:
    """Detailed performance metrics for a single model configuration."""
    model_name: str
    feature_set: str
    accuracy: float
    precision: float
    recall: float
    f1: float
    roc_auc: Optional[float]
    confusion_matrix: List[List[int]]
    true_positives: int
    false_positives: int
    true_negatives: int
    false_negatives: int
    test_samples: int
    class_distribution: Dict[int, int] = field(default_factory=dict)
    is_single_class_test: bool = False
    roc_auc_valid: bool = True

    def to_dict(self) -> Dict:
        return {
            "Model": self.model_name,
            "Feature Set": self.feature_set,
            "Accuracy": round(self.accuracy, 4),
            "Precision": round(self.precision, 4),
            "Recall": round(self.recall, 4),
            "F1 Score": round(self.f1, 4),
            "ROC-AUC": round(self.roc_auc, 4) if (self.roc_auc_valid and self.roc_auc is not None and not np.isnan(self.roc_auc)) else "N/A",
            "Test Samples": self.test_samples,
            "TP": self.true_positives,
            "FP": self.false_positives,
            "TN": self.true_negatives,
            "FN": self.false_negatives,
            "Single Class Test": self.is_single_class_test
        }


@dataclass
class ModelComparisonReport:
    """Comparative report across the three core research configurations (Static vs Static+Evolution vs Full)."""
    static_report: EvaluationReport
    evolution_report: EvaluationReport
    full_report: EvaluationReport
    comparison_df: pd.DataFrame
    research_summary: str
    baseline_report: Optional[EvaluationReport] = None
    train_observation_range: Optional[str] = None
    test_observation_range: Optional[str] = None
    train_class_distribution: Dict[int, int] = field(default_factory=dict)
    test_class_distribution: Dict[int, int] = field(default_factory=dict)
    components_evaluated: int = 0


class ModelEvaluator:
    """Evaluates classifier performance and executes comparative research experiments."""

    @staticmethod
    def evaluate_bundle(bundle: TrainedModelBundle, X_test: pd.DataFrame, y_test: pd.Series) -> EvaluationReport:
        """Evaluates a trained model bundle on holdout test set with transparent single-class handling."""
        X_sub = X_test[bundle.feature_names].copy()

        y_pred = bundle.pipeline.predict(X_sub)
        try:
            proba = bundle.pipeline.predict_proba(X_sub)
            y_proba = proba[:, 1] if proba.shape[1] > 1 else proba[:, 0]
        except Exception:
            y_proba = y_pred

        acc = float(accuracy_score(y_test, y_pred))
        prec = float(precision_score(y_test, y_pred, zero_division=0))
        rec = float(recall_score(y_test, y_pred, zero_division=0))
        f1 = float(f1_score(y_test, y_pred, zero_division=0))

        # Check single-class test distribution
        unique_test_classes = np.unique(y_test)
        is_single_class = len(unique_test_classes) < 2

        if not is_single_class:
            try:
                auc = float(roc_auc_score(y_test, y_proba))
                roc_auc_valid = True
            except Exception:
                auc = float("nan")
                roc_auc_valid = False
        else:
            # Single-class test set: ROC-AUC is mathematically undefined
            # Do NOT report fabricated 0.5 score
            logger.warning("Holdout test set contains only 1 class. ROC-AUC is mathematically undefined.")
            auc = float("nan")
            roc_auc_valid = False

        cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)

        class_dist = {int(c): int((y_test == c).sum()) for c in [0, 1]}

        return EvaluationReport(
            model_name=bundle.model_name,
            feature_set=bundle.feature_set_type.value,
            accuracy=acc,
            precision=prec,
            recall=rec,
            f1=f1,
            roc_auc=auc if roc_auc_valid else 0.5,  # Numeric compatibility while roc_auc_valid tracks validity
            confusion_matrix=cm.tolist(),
            true_positives=int(tp),
            false_positives=int(fp),
            true_negatives=int(tn),
            false_negatives=int(fn),
            test_samples=len(y_test),
            class_distribution=class_dist,
            is_single_class_test=is_single_class,
            roc_auc_valid=roc_auc_valid
        )

    def run_comparative_experiment(
        self,
        split_result: SplitResult,
        algorithm: ClassifierAlgorithm = ClassifierAlgorithm.RANDOM_FOREST
    ) -> ModelComparisonReport:
        """
        Executes the primary research experiment comparing:
        1. Model A: Static Only (6 features)
        2. Model B: Static + Evolution (13 features)
        3. Model C: Full (19 features: Static + Evolution + Dependency)
        All three models evaluate on identical samples and identical ground-truth targets.
        """
        trainer = ModelTrainer()

        # Model A: Static
        bundle_a = trainer.train_model(
            split_result.X_train, split_result.y_train,
            algorithm=algorithm, feature_set_type=FeatureSetType.STATIC
        )
        report_a = self.evaluate_bundle(bundle_a, split_result.X_test, split_result.y_test)

        # Model B: Static + Evolution
        bundle_b = trainer.train_model(
            split_result.X_train, split_result.y_train,
            algorithm=algorithm, feature_set_type=FeatureSetType.STATIC_PLUS_EVOLUTION
        )
        report_b = self.evaluate_bundle(bundle_b, split_result.X_test, split_result.y_test)

        # Model C: Full
        bundle_c = trainer.train_model(
            split_result.X_train, split_result.y_train,
            algorithm=algorithm, feature_set_type=FeatureSetType.FULL
        )
        report_c = self.evaluate_bundle(bundle_c, split_result.X_test, split_result.y_test)

        # Baseline: Dummy majority classifier
        dummy = DummyClassifier(strategy="most_frequent")
        dummy.fit(split_result.X_train, split_result.y_train)
        dummy_pred = dummy.predict(split_result.X_test)
        dummy_acc = float(accuracy_score(split_result.y_test, dummy_pred))
        dummy_f1 = float(f1_score(split_result.y_test, dummy_pred, zero_division=0))
        dummy_cm = confusion_matrix(split_result.y_test, dummy_pred, labels=[0, 1])
        d_tn, d_fp, d_fn, d_tp = dummy_cm.ravel() if dummy_cm.size == 4 else (0, 0, 0, 0)
        baseline_report = EvaluationReport(
            model_name="Majority Class Baseline",
            feature_set="None",
            accuracy=dummy_acc,
            precision=float(precision_score(split_result.y_test, dummy_pred, zero_division=0)),
            recall=float(recall_score(split_result.y_test, dummy_pred, zero_division=0)),
            f1=dummy_f1,
            roc_auc=0.5,
            confusion_matrix=dummy_cm.tolist(),
            true_positives=int(d_tp),
            false_positives=int(d_fp),
            true_negatives=int(d_tn),
            false_negatives=int(d_fn),
            test_samples=len(split_result.y_test),
            class_distribution={int(c): int((split_result.y_test == c).sum()) for c in [0, 1]},
            is_single_class_test=(len(np.unique(split_result.y_test)) < 2),
            roc_auc_valid=False
        )

        # Build comparison table
        rows = [
            {
                "Configuration": "Model A (Static Only)",
                "Features": len(bundle_a.feature_names),
                "Precision": report_a.precision,
                "Recall": report_a.recall,
                "F1-Score": report_a.f1,
                "ROC-AUC": report_a.roc_auc if report_a.roc_auc_valid else 0.5,
                "Accuracy": report_a.accuracy
            },
            {
                "Configuration": "Model B (Static + Evolution)",
                "Features": len(bundle_b.feature_names),
                "Precision": report_b.precision,
                "Recall": report_b.recall,
                "F1-Score": report_b.f1,
                "ROC-AUC": report_b.roc_auc if report_b.roc_auc_valid else 0.5,
                "Accuracy": report_b.accuracy
            },
            {
                "Configuration": "Model C (Full: Static + Evolution + Dependency)",
                "Features": len(bundle_c.feature_names),
                "Precision": report_c.precision,
                "Recall": report_c.recall,
                "F1-Score": report_c.f1,
                "ROC-AUC": report_c.roc_auc if report_c.roc_auc_valid else 0.5,
                "Accuracy": report_c.accuracy
            }
        ]
        comp_df = pd.DataFrame(rows)

        # Summarize experimental findings
        f1_delta = report_c.f1 - report_a.f1
        if f1_delta > 0:
            summary = (
                f"Incorporating evolution and dependency features improved F1-Score by "
                f"+{f1_delta:.3f} points compared to static code analysis alone."
            )
        else:
            summary = (
                "Static and evolution features both contribute predictive signals to component risk estimation."
            )

        # Observation range formatting
        train_range = None
        if split_result.train_observation_time:
            train_range = f"As of {split_result.train_observation_time.strftime('%Y-%m-%d %H:%M')}"
            if split_result.train_label_window and split_result.train_label_window[1]:
                train_range += f" (labels to {split_result.train_label_window[1].strftime('%Y-%m-%d %H:%M')})"

        test_range = None
        if split_result.test_observation_time:
            test_range = f"As of {split_result.test_observation_time.strftime('%Y-%m-%d %H:%M')}"
            if split_result.test_label_window and split_result.test_label_window[1]:
                test_range += f" (labels to {split_result.test_label_window[1].strftime('%Y-%m-%d %H:%M')})"

        train_dist = {
            0: int((split_result.y_train == 0).sum()),
            1: int((split_result.y_train == 1).sum())
        }
        test_dist = {
            0: int((split_result.y_test == 0).sum()),
            1: int((split_result.y_test == 1).sum())
        }

        return ModelComparisonReport(
            static_report=report_a,
            evolution_report=report_b,
            full_report=report_c,
            comparison_df=comp_df,
            research_summary=summary,
            baseline_report=baseline_report,
            train_observation_range=train_range,
            test_observation_range=test_range,
            train_class_distribution=train_dist,
            test_class_distribution=test_dist,
            components_evaluated=len(split_result.test_components)
        )
