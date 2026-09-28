"""
SoftwarePulse Machine Learning Package.
Implements Model A (Static), Model B (Static+Evolution), Model C (Full), Random Forest, Logistic Regression, evaluation, and predictions.
"""
from .train import ModelTrainer, TrainedModelBundle
from .predict import RiskPredictor, PredictionResult, ComponentRiskProfile
from .evaluate import ModelEvaluator, EvaluationReport, ModelComparisonReport
from .model_manager import ModelManager

__all__ = [
    "ModelTrainer",
    "TrainedModelBundle",
    "RiskPredictor",
    "PredictionResult",
    "ComponentRiskProfile",
    "ModelEvaluator",
    "EvaluationReport",
    "ModelComparisonReport",
    "ModelManager"
]
