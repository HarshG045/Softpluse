"""
SoftwarePulse Dataset Package.
Handles temporal labeling (bugfix heuristic / proxy label), chronological train-test splitting, and dataset building.
"""
from .labeling import LabelGenerator, TargetLabelType
from .temporal_split import TemporalSplitter, SplitResult
from .builder import DatasetBuilder, BuiltDataset

__all__ = [
    "LabelGenerator",
    "TargetLabelType",
    "TemporalSplitter",
    "SplitResult",
    "DatasetBuilder",
    "BuiltDataset"
]
