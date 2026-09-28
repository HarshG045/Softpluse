"""
ModelManager: Handles persistence, caching, and loading of trained models and metadata.
"""
import logging
import pickle
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

from config import MODELS_DIR
from .train import TrainedModelBundle

logger = logging.getLogger(__name__)


class ModelManager:
    """Manages serialization and caching of trained ML models."""

    def __init__(self, models_dir: Path = MODELS_DIR):
        self.models_dir = models_dir
        self.models_dir.mkdir(parents=True, exist_ok=True)

    def save_model(self, bundle: TrainedModelBundle, filename: Optional[str] = None) -> Path:
        """Saves a TrainedModelBundle to disk."""
        if not filename:
            safe_name = bundle.model_name.replace(" ", "_").replace("(", "").replace(")", "").replace("+", "plus").lower()
            filename = f"{safe_name}.pkl"
        
        path = self.models_dir / filename
        with open(path, "wb") as f:
            pickle.dump(bundle, f)
        logger.info(f"Model successfully saved to {path}")
        return path

    def load_model(self, filename: str) -> Optional[TrainedModelBundle]:
        """Loads a TrainedModelBundle from disk."""
        path = self.models_dir / filename
        if not path.exists():
            return None
        with open(path, "rb") as f:
            bundle = pickle.load(f)
        return bundle
