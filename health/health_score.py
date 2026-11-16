"""
Repository Health Score Engine.
Evaluates repository condition across five dimensions:
  1. Code Quality (Complexity, LOC, Maintainability)
  2. Change Risk (Churn, Frequency, Authors, Recency)
  3. Dependency Health (Coupling, Centrality, Cycles)
  4. Stability (Defect Density, Volatility, Risk Distribution)
  5. Test Health (Test-to-Source Ratio, Test Structure)

Produces a transparent 0-100 overall score, dimension breakdowns,
identified weaknesses, and recommended improvements.
"""
import logging
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np

from analysis.code_analyzer import CodeMetrics
from analysis.dependency_analyzer import DependencyGraph
from analysis.git_analyzer import ComponentGitMetrics
from ingestion.git_loader import CommitRecord
from ml.predict import PredictionResult

logger = logging.getLogger(__name__)


class HealthDimension(str, Enum):
    CODE_QUALITY = "Code Quality"
    CHANGE_RISK = "Change Risk"
    DEPENDENCY_HEALTH = "Dependency Health"
    STABILITY = "Stability"
    TEST_HEALTH = "Test Health"


# Configurable dimension weights (sum to 1.0)
DEFAULT_HEALTH_WEIGHTS: Dict[HealthDimension, float] = {
    HealthDimension.CODE_QUALITY: 0.25,
    HealthDimension.CHANGE_RISK: 0.25,
    HealthDimension.DEPENDENCY_HEALTH: 0.20,
    HealthDimension.STABILITY: 0.20,
    HealthDimension.TEST_HEALTH: 0.10,
}


@dataclass
class HealthDimensionScore:
    """Detailed score and metric breakdown for a single health dimension."""
    dimension: HealthDimension
    score: int  # 0 to 100
    weight: float
    grade: str  # A, B, C, D, F
    status: str  # EXCELLENT, HEALTHY, MODERATE, ELEVATED_RISK, CRITICAL
    metrics: Dict[str, float] = field(default_factory=dict)
    weaknesses: List[str] = field(default_factory=list)
    improvements: List[str] = field(default_factory=list)


@dataclass
class RepositoryHealthReport:
    """Consolidated health report across all 5 dimensions."""
    overall_score: int  # 0 to 100
    overall_grade: str  # A, B, C, D, F
    overall_status: str
    dimensions: Dict[HealthDimension, HealthDimensionScore]
    top_weaknesses: List[str]
    recommended_improvements: List[str]
    total_components_evaluated: int
    test_coverage_available: bool = False
    test_coverage_note: str = "Test coverage data requires coverage instrumentation (pytest-cov). Ratio proxy reported."


def _score_to_grade(score: float) -> Tuple[str, str]:
    if score >= 85:
        return "A", "EXCELLENT"
    elif score >= 70:
        return "B", "HEALTHY"
    elif score >= 55:
        return "C", "MODERATE"
    elif score >= 40:
        return "D", "ELEVATED_RISK"
    else:
        return "F", "CRITICAL"


class RepositoryHealthScore:
    """Calculates multidimensional repository health from actual analysis data."""

    def __init__(self, weights: Optional[Dict[HealthDimension, float]] = None):
        self.weights = weights or DEFAULT_HEALTH_WEIGHTS

    def evaluate_repository(
        self,
        code_map: Dict[str, CodeMetrics],
        git_map: Dict[str, ComponentGitMetrics],
        dep_graph: DependencyGraph,
        commits: List[CommitRecord],
        prediction_res: Optional[PredictionResult] = None,
        all_repo_files: Optional[List[str]] = None
    ) -> RepositoryHealthReport:
        """Computes the comprehensive repository health assessment."""
        dim_scores: Dict[HealthDimension, HealthDimensionScore] = {}

        dim_scores[HealthDimension.CODE_QUALITY] = self._evaluate_code_quality(code_map)
        dim_scores[HealthDimension.CHANGE_RISK] = self._evaluate_change_risk(git_map, commits)
        dim_scores[HealthDimension.DEPENDENCY_HEALTH] = self._evaluate_dependency_health(dep_graph)
        dim_scores[HealthDimension.STABILITY] = self._evaluate_stability(git_map, commits, prediction_res)
        dim_scores[HealthDimension.TEST_HEALTH] = self._evaluate_test_health(code_map, all_repo_files)

        # Weighted aggregate
        overall_val = sum(
            dim_scores[dim].score * self.weights.get(dim, 0.20)
            for dim in dim_scores
        )
        overall_score = max(0, min(100, int(round(overall_val))))
        overall_grade, overall_status = _score_to_grade(overall_score)

        # Compile top weaknesses and recommendations across all dimensions
        all_weaknesses = []
        all_improvements = []
        for d in dim_scores.values():
            all_weaknesses.extend(d.weaknesses)
            all_improvements.extend(d.improvements)

        return RepositoryHealthReport(
            overall_score=overall_score,
            overall_grade=overall_grade,
            overall_status=overall_status,
            dimensions=dim_scores,
            top_weaknesses=all_weaknesses[:5],
            recommended_improvements=all_improvements[:5],
            total_components_evaluated=len(code_map)
        )

    def _evaluate_code_quality(self, code_map: Dict[str, CodeMetrics]) -> HealthDimensionScore:
        dim = HealthDimension.CODE_QUALITY
        weight = self.weights[dim]
        if not code_map:
            return HealthDimensionScore(dim, 100, weight, "A", "EXCELLENT")

        complexities = [m.complexity for m in code_map.values()]
        locs = [m.loc for m in code_map.values()]
        comment_ratios = [m.comment_ratio for m in code_map.values()]

        avg_comp = float(np.mean(complexities))
        max_comp = max(complexities)
        high_comp_count = sum(1 for c in complexities if c >= 10)
        very_high_comp = sum(1 for c in complexities if c >= 20)
        large_files = sum(1 for loc in locs if loc >= 300)

        # Base score 100 with deduction penalties
        score = 100.0
        weaknesses = []
        improvements = []

        if avg_comp > 10:
            score -= 25.0
            weaknesses.append(f"High repository-average cyclomatic complexity ({avg_comp:.1f})")
            improvements.append("Refactor large conditional blocks and nested loops into dedicated methods.")
        elif avg_comp > 6:
            score -= 10.0

        if very_high_comp > 0:
            score -= min(25.0, very_high_comp * 8.0)
            weaknesses.append(f"{very_high_comp} component(s) have critical complexity (>= 20)")
            improvements.append("Prioritize modular decomposition on components with complexity >= 20.")

        if large_files > 0:
            score -= min(20.0, large_files * 5.0)
            weaknesses.append(f"{large_files} large source file(s) exceed 300 lines of code")
            improvements.append("Split monolithic files into cohesive single-responsibility modules.")

        avg_comments = float(np.mean(comment_ratios)) if comment_ratios else 0.0
        if avg_comments < 0.05:
            score -= 5.0

        final_score = max(0, min(100, int(round(score))))
        grade, status = _score_to_grade(final_score)

        return HealthDimensionScore(
            dimension=dim,
            score=final_score,
            weight=weight,
            grade=grade,
            status=status,
            metrics={
                "average_complexity": round(avg_comp, 2),
                "max_complexity": float(max_comp),
                "high_complexity_files": float(high_comp_count),
                "critical_complexity_files": float(very_high_comp),
                "average_loc": round(float(np.mean(locs)), 1),
                "large_files_count": float(large_files),
                "average_comment_ratio": round(avg_comments, 3)
            },
            weaknesses=weaknesses,
            improvements=improvements
        )

    def _evaluate_change_risk(
        self,
        git_map: Dict[str, ComponentGitMetrics],
        commits: List[CommitRecord]
    ) -> HealthDimensionScore:
        dim = HealthDimension.CHANGE_RISK
        weight = self.weights[dim]
        if not git_map or not commits:
            return HealthDimensionScore(dim, 100, weight, "A", "EXCELLENT")

        recent_churns = [m.recent_churn for m in git_map.values()]
        total_churns = [m.churn for m in git_map.values()]
        author_counts = [m.number_of_authors for m in git_map.values()]

        tot_recent_churn = sum(recent_churns)
        tot_churn = sum(total_churns)
        single_author_files = sum(1 for a in author_counts if a == 1)

        score = 100.0
        weaknesses = []
        improvements = []

        # High recent churn volatility
        if tot_churn > 0 and (tot_recent_churn / tot_churn) > 0.40:
            score -= 20.0
            weaknesses.append(f"Elevated recent churn volatility ({tot_recent_churn} lines recently modified)")
            improvements.append("Increase code review depth for components undergoing rapid recent modifications.")

        # Bus factor / author concentration
        if len(git_map) > 0 and (single_author_files / len(git_map)) > 0.60:
            score -= 15.0
            weaknesses.append(f"{single_author_files} components modified by only a single contributor (bus factor risk)")
            improvements.append("Promote cross-team code reviews and knowledge sharing on solo-authored components.")

        final_score = max(0, min(100, int(round(score))))
        grade, status = _score_to_grade(final_score)

        return HealthDimensionScore(
            dimension=dim,
            score=final_score,
            weight=weight,
            grade=grade,
            status=status,
            metrics={
                "total_recent_churn": float(tot_recent_churn),
                "total_historical_churn": float(tot_churn),
                "single_contributor_files": float(single_author_files),
                "total_commits": float(len(commits))
            },
            weaknesses=weaknesses,
            improvements=improvements
        )

    def _evaluate_dependency_health(self, dep_graph: DependencyGraph) -> HealthDimensionScore:
        dim = HealthDimension.DEPENDENCY_HEALTH
        weight = self.weights[dim]
        g = dep_graph.graph
        if g.number_of_nodes() == 0:
            return HealthDimensionScore(dim, 100, weight, "A", "EXCELLENT")

        struct_map = dep_graph.get_metrics_for_all()
        cycles = dep_graph.detect_cycles()
        degrees = [sm.degree for sm in struct_map.values()]
        centralities = [sm.centrality for sm in struct_map.values()]

        avg_degree = float(np.mean(degrees)) if degrees else 0.0
        max_degree = max(degrees) if degrees else 0
        hubs = sum(1 for d in degrees if d >= 4)

        score = 100.0
        weaknesses = []
        improvements = []

        # Heavy penalty for circular dependencies
        if len(cycles) > 0:
            deduction = min(50.0, 25.0 + (len(cycles) - 1) * 15.0)
            score -= deduction
            weaknesses.append(f"{len(cycles)} circular dependency cycle(s) detected in component architecture")
            improvements.append("Break import cycles using dependency inversion or an event-driven architecture.")

        # Architectural coupling hotspots
        if hubs > 0:
            score -= min(20.0, hubs * 5.0)
            weaknesses.append(f"{hubs} architectural hub(s) have high coupling (degree >= 4)")
            improvements.append("Introduce interface facades to decouple heavily connected hub components.")

        final_score = max(0, min(100, int(round(score))))
        grade, status = _score_to_grade(final_score)

        return HealthDimensionScore(
            dimension=dim,
            score=final_score,
            weight=weight,
            grade=grade,
            status=status,
            metrics={
                "circular_cycles_count": float(len(cycles)),
                "average_degree": round(avg_degree, 2),
                "max_degree": float(max_degree),
                "architectural_hubs": float(hubs),
                "total_components": float(len(struct_map))
            },
            weaknesses=weaknesses,
            improvements=improvements
        )

    def _evaluate_stability(
        self,
        git_map: Dict[str, ComponentGitMetrics],
        commits: List[CommitRecord],
        prediction_res: Optional[PredictionResult]
    ) -> HealthDimensionScore:
        dim = HealthDimension.STABILITY
        weight = self.weights[dim]
        if not git_map or not commits:
            return HealthDimensionScore(dim, 100, weight, "A", "EXCELLENT")

        bugfix_commits = sum(1 for c in commits if c.is_bugfix)
        bugfix_ratio = bugfix_commits / max(1, len(commits))

        high_risk_comps = prediction_res.high_risk_count if prediction_res else 0
        tot_comps = len(prediction_res.profiles) if prediction_res else len(git_map)
        high_risk_ratio = high_risk_comps / max(1, tot_comps)

        score = 100.0
        weaknesses = []
        improvements = []

        if bugfix_ratio > 0.35:
            score -= 20.0
            weaknesses.append(f"High historical bug-fix density ({bugfix_ratio*100:.1f}% of commits are defect fixes)")
            improvements.append("Incorporate automated regression test suites to reduce defect recurrence.")
        elif bugfix_ratio > 0.20:
            score -= 10.0

        if high_risk_ratio > 0.40:
            score -= 25.0
            weaknesses.append(f"{high_risk_comps} component(s) ({high_risk_ratio*100:.1f}%) classified as high risk")
            improvements.append("Address top risky components identified by the ML risk explorer.")
        elif high_risk_ratio > 0.20:
            score -= 10.0

        final_score = max(0, min(100, int(round(score))))
        grade, status = _score_to_grade(final_score)

        return HealthDimensionScore(
            dimension=dim,
            score=final_score,
            weight=weight,
            grade=grade,
            status=status,
            metrics={
                "bugfix_commits_count": float(bugfix_commits),
                "bugfix_commit_ratio": round(bugfix_ratio, 3),
                "high_risk_components_count": float(high_risk_comps),
                "high_risk_ratio": round(high_risk_ratio, 3)
            },
            weaknesses=weaknesses,
            improvements=improvements
        )

    def _evaluate_test_health(
        self,
        code_map: Dict[str, CodeMetrics],
        all_repo_files: Optional[List[str]]
    ) -> HealthDimensionScore:
        dim = HealthDimension.TEST_HEALTH
        weight = self.weights[dim]

        files = list(code_map.keys())
        if all_repo_files:
            files = list(set(files + all_repo_files))

        test_files = [
            f for f in files
            if Path(f).name.startswith("test_") or Path(f).name.endswith("_test.py") or "tests/" in f.replace("\\", "/")
        ]
        source_files = [f for f in files if f not in test_files and f.endswith(".py")]

        test_count = len(test_files)
        src_count = len(source_files)
        ratio = test_count / max(1, src_count)

        score = 100.0
        weaknesses = []
        improvements = []

        if test_count == 0:
            score = 30.0
            weaknesses.append("No automated test files detected (zero test_*.py or tests/ files)")
            improvements.append("Introduce automated test suites with pytest for core business components.")
        elif ratio < 0.30:
            score = 65.0
            weaknesses.append(f"Low test-to-source ratio ({test_count} tests for {src_count} source files: {ratio:.2f})")
            improvements.append("Increase test coverage by adding test suites for untested modules.")
        elif ratio < 0.60:
            score = 80.0

        final_score = max(0, min(100, int(round(score))))
        grade, status = _score_to_grade(final_score)

        return HealthDimensionScore(
            dimension=dim,
            score=final_score,
            weight=weight,
            grade=grade,
            status=status,
            metrics={
                "test_files_count": float(test_count),
                "source_files_count": float(src_count),
                "test_to_source_ratio": round(ratio, 3),
                "has_dedicated_test_dir": 1.0 if any("tests" in f for f in files) else 0.0
            },
            weaknesses=weaknesses,
            improvements=improvements
        )
