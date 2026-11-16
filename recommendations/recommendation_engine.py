"""
Intelligent Recommendation Engine for SoftwarePulse.
Translates multi-modal repository metrics, explainability signals, and dependency topology
into evidence-based, prioritized, actionable developer recommendations.
"""
import logging
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Set

from analysis.code_analyzer import CodeMetrics
from analysis.dependency_analyzer import DependencyGraph
from analysis.git_analyzer import ComponentGitMetrics
from ml.predict import ComponentRiskProfile

logger = logging.getLogger(__name__)


class RecommendationPriority(str, Enum):
    CRITICAL = "CRITICAL"
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"


class RecommendationCategory(str, Enum):
    REFACTORING = "Code Refactoring"
    ARCHITECTURE = "Architectural Decoupling"
    TESTING = "Testing & Verification"
    REVIEW_PROCESS = "Review & Governance"
    KNOWLEDGE_SHARING = "Knowledge Transfer"


@dataclass
class ComponentRecommendation:
    """An evidence-based, actionable recommendation for a software component."""
    component_name: str
    filepath: str
    action: str
    reason: str
    priority: RecommendationPriority
    category: RecommendationCategory
    triggering_metrics: Dict[str, float]
    suggested_next_step: str
    evidence_type: str = "Quantitative Repository Metric"

    def to_dict(self) -> Dict:
        return {
            "Component": self.component_name,
            "Filepath": self.filepath,
            "Priority": self.priority.value,
            "Category": self.category.value,
            "Action": self.action,
            "Reason": self.reason,
            "Supporting Metrics": ", ".join(f"{k}={v}" for k, v in self.triggering_metrics.items()),
            "Next Step": self.suggested_next_step
        }


class IntelligentRecommendationEngine:
    """Generates concrete, actionable recommendations based on empirical metrics and graph topology."""

    def __init__(self, dep_graph: Optional[DependencyGraph] = None):
        self.dep_graph = dep_graph
        self.cycles = dep_graph.detect_cycles() if dep_graph else []
        self.cycle_members: Set[str] = set()
        for cyc in self.cycles:
            self.cycle_members.update(cyc)

    def generate_recommendations(
        self,
        profile: ComponentRiskProfile,
        code_metric: Optional[CodeMetrics] = None,
        git_metric: Optional[ComponentGitMetrics] = None,
        has_tests: bool = True
    ) -> List[ComponentRecommendation]:
        """Generates deduplicated, evidence-based recommendations for a specific component."""
        recs: List[ComponentRecommendation] = []
        fp = profile.filepath
        c_name = profile.component_name

        # Rule 1: High Cyclomatic Complexity
        if profile.complexity >= 10:
            is_critical = profile.complexity >= 20
            prio = RecommendationPriority.CRITICAL if is_critical else RecommendationPriority.HIGH
            recs.append(ComponentRecommendation(
                component_name=c_name,
                filepath=fp,
                action="Decompose complex branching logic into smaller, testable functions",
                reason=f"McCabe cyclomatic complexity is {profile.complexity} (>= 10 threshold), indicating nested branching paths that increase defect risk.",
                priority=prio,
                category=RecommendationCategory.REFACTORING,
                triggering_metrics={"complexity": float(profile.complexity), "functions": float(profile.functions)},
                suggested_next_step="Extract nested if/else conditionals and loops into helper functions with single responsibilities."
            ))

        # Rule 2: High Recent Churn / Volatility
        if profile.recent_churn >= 150 or profile.change_frequency >= 0.25:
            recs.append(ComponentRecommendation(
                component_name=c_name,
                filepath=fp,
                action="Enforce strict peer review and regression testing for upcoming pull requests",
                reason=f"High change volatility ({profile.recent_churn} lines recently churned; touched in {profile.change_frequency*100:.1f}% of commits).",
                priority=RecommendationPriority.HIGH,
                category=RecommendationCategory.REVIEW_PROCESS,
                triggering_metrics={"recent_churn": float(profile.recent_churn), "change_frequency": round(profile.change_frequency, 3)},
                suggested_next_step="Inspect recent commits touching this file to verify whether modifications address recurring regression defects."
            ))

        # Rule 3: Circular Dependency Participation
        if fp in self.cycle_members:
            cycle_path = next((cyc for cyc in self.cycles if fp in cyc), [])
            cyc_str = " ➔ ".join(p.split("/")[-1] for p in cycle_path)
            recs.append(ComponentRecommendation(
                component_name=c_name,
                filepath=fp,
                action="Break circular dependency cycle through interface abstraction or event emission",
                reason=f"Component participates in an architectural import cycle: ({cyc_str}), preventing independent testing and modular deployment.",
                priority=RecommendationPriority.CRITICAL,
                category=RecommendationCategory.ARCHITECTURE,
                triggering_metrics={"cycle_length": float(len(cycle_path)), "degree": float(profile.degree)},
                suggested_next_step="Apply Dependency Inversion: extract shared types/interfaces to a common base module."
            ))

        # Rule 4: High Architectural Centrality / Hub Risk
        if (profile.dependent_count >= 3 or profile.degree >= 5 or profile.centrality >= 0.25) and profile.risk_percentage >= 40:
            recs.append(ComponentRecommendation(
                component_name=c_name,
                filepath=fp,
                action="Protect downstream dependents with rigorous contract and integration tests",
                reason=f"Component is an architectural hub imported by {profile.dependent_count} other modules (degree={profile.degree}). Changes pose high blast-radius risk.",
                priority=RecommendationPriority.HIGH,
                category=RecommendationCategory.ARCHITECTURE,
                triggering_metrics={"dependent_count": float(profile.dependent_count), "centrality": round(profile.centrality, 3)},
                suggested_next_step=f"Add integration contract tests verifying compatibility for all {profile.dependent_count} dependent modules."
            ))

        # Rule 5: Monolithic File Size
        if profile.loc >= 250:
            recs.append(ComponentRecommendation(
                component_name=c_name,
                filepath=fp,
                action="Split large monolithic file into cohesive submodules",
                reason=f"File contains {profile.loc} lines of code, exceeding the 250 LOC modularity guideline.",
                priority=RecommendationPriority.MEDIUM,
                category=RecommendationCategory.REFACTORING,
                triggering_metrics={"loc": float(profile.loc), "classes": float(profile.classes), "methods": float(profile.methods)},
                suggested_next_step="Group related class definitions and helper functions into separate dedicated files within a package folder."
            ))

        # Rule 6: Single-Contributor Exposure (Bus Factor)
        if profile.number_of_authors == 1 and profile.commit_count >= 2 and profile.risk_percentage >= 50:
            recs.append(ComponentRecommendation(
                component_name=c_name,
                filepath=fp,
                action="Schedule peer walkthrough to reduce single-contributor knowledge silo",
                reason=f"High-risk component ({profile.risk_percentage}%) has been modified by only 1 developer across {profile.commit_count} commits.",
                priority=RecommendationPriority.MEDIUM,
                category=RecommendationCategory.KNOWLEDGE_SHARING,
                triggering_metrics={"number_of_authors": 1.0, "commit_count": float(profile.commit_count)},
                suggested_next_step="Assign a secondary developer for code reviews and documentation walkthroughs."
            ))

        # Rule 7: Missing Test File Affinity for Risky Component
        stem = Path(fp).stem
        if not has_tests and profile.risk_percentage >= 40:
            recs.append(ComponentRecommendation(
                component_name=c_name,
                filepath=fp,
                action=f"Create automated test suite: test_{stem}.py",
                reason=f"Component has elevated predicted risk ({profile.risk_percentage}%) but no dedicated automated test module was detected.",
                priority=RecommendationPriority.HIGH,
                category=RecommendationCategory.TESTING,
                triggering_metrics={"risk_percentage": float(profile.risk_percentage)},
                suggested_next_step=f"Create tests/test_{stem}.py with pytest covering happy paths and edge cases."
            ))

        # Sort recommendations: CRITICAL -> HIGH -> MEDIUM -> LOW
        priority_order = {
            RecommendationPriority.CRITICAL: 0,
            RecommendationPriority.HIGH: 1,
            RecommendationPriority.MEDIUM: 2,
            RecommendationPriority.LOW: 3
        }
        recs.sort(key=lambda r: priority_order.get(r.priority, 4))

        return recs

    def generate_repository_recommendations(
        self,
        profiles: List[ComponentRiskProfile],
        all_repo_files: Optional[List[str]] = None
    ) -> List[ComponentRecommendation]:
        """Collects and ranks recommendations across all components in the repository."""
        all_recs: List[ComponentRecommendation] = []
        test_file_stems = set()
        if all_repo_files:
            for f in all_repo_files:
                name = Path(f).name
                if name.startswith("test_") or name.endswith("_test.py"):
                    clean = name.replace("test_", "").replace("_test.py", "").replace(".py", "")
                    test_file_stems.add(clean.lower())

        for p in profiles:
            stem = Path(p.filepath).stem.lower()
            has_tests = stem in test_file_stems or any(stem in s for s in test_file_stems)
            comp_recs = self.generate_recommendations(p, has_tests=has_tests)
            all_recs.extend(comp_recs)

        # Sort by priority
        prio_order = {
            RecommendationPriority.CRITICAL: 0,
            RecommendationPriority.HIGH: 1,
            RecommendationPriority.MEDIUM: 2,
            RecommendationPriority.LOW: 3
        }
        all_recs.sort(key=lambda r: (prio_order.get(r.priority, 4), -r.triggering_metrics.get("complexity", 0)))
        return all_recs
