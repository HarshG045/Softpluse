"""
SoftwarePulse Simulation Package.
Provides What-If Change simulation engine across structural dependencies, complexity, and churn parameters.
"""
from .what_if import WhatIfEngine, SimulationChangeType, SimulationResult, ChangedFeatureDiff

__all__ = ["WhatIfEngine", "SimulationChangeType", "SimulationResult", "ChangedFeatureDiff"]
