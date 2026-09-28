"""
Configuration settings for SoftwarePulse.
Provides centralized defaults for risk prediction, repository analysis, and UI parameters.
"""
from pathlib import Path

# Base Paths
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
DEMO_DATA_DIR = DATA_DIR / "demo"
MODELS_DIR = BASE_DIR / "models"

# Ensure directories exist
for p in [RAW_DATA_DIR, PROCESSED_DATA_DIR, DEMO_DATA_DIR, MODELS_DIR]:
    p.mkdir(parents=True, exist_ok=True)

# Risk Category Thresholds (User-Configurable defaults)
RISK_THRESHOLDS = {
    "LOW": (0.0, 0.33),
    "MEDIUM": (0.33, 0.66),
    "HIGH": (0.66, 1.0)
}

# Heuristic Bug-Fix Commit Keywords
BUG_KEYWORDS = [
    "fix", "bug", "bugfix", "defect", "patch", "error", 
    "crash", "failure", "issue", "resolve", "close", "resolves",
    "regression", "fault", "vulnerability", "leak"
]

# Analysis Parameters
RECENT_WINDOW_DAYS = 30
RECENT_WINDOW_COMMITS = 20
MIN_COMMITS_FOR_ANALYSIS = 3
MAX_GRAPH_NODES_DEFAULT = 60
RANDOM_STATE = 42

# Supported Extensions for parsing
SUPPORTED_PYTHON_EXTENSIONS = {".py"}

# Feature Sets definition
STATIC_FEATURES = [
    "loc",
    "functions",
    "classes",
    "complexity",
    "imports",
    "methods"
]

EVOLUTION_FEATURES = [
    "commit_count",
    "recent_commit_count",
    "churn",
    "recent_churn",
    "change_frequency",
    "number_of_authors",
    "time_since_last_change"
]

STRUCTURAL_FEATURES = [
    "dependency_count",
    "dependent_count",
    "degree",
    "centrality",
    "in_degree",
    "out_degree"
]

ALL_FEATURES = STATIC_FEATURES + EVOLUTION_FEATURES + STRUCTURAL_FEATURES
