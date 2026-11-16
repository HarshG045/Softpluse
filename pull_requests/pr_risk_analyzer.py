"""
PullRequestRiskAnalyzer: Evaluates pre-merge pull request and branch changes.
Calculates structural shifts, predicted ML risk deltas, and downstream dependency exposure.
"""
import logging
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from analysis.change_analyzer import ChangeImpactAnalyzer, ComponentChangeImpact
from analysis.dependency_analyzer import DependencyGraph
from ingestion.git_loader import FileDiffItem, GitLoader
from ml.predict import ComponentRiskProfile, RiskPredictor
from ml.train import TrainedModelBundle
from .github_client import GitHubPRClient, GitHubPRMetadata
from .pr_diff_parser import PRDiffParser

logger = logging.getLogger(__name__)


@dataclass
class PRFileRiskDetail:
    """Risk and structural change assessment for a single file in a pull request."""
    filepath: str
    change_type: str  # ADDED, MODIFIED, DELETED, RENAMED
    insertions: int
    deletions: int
    current_risk_pct: int
    proposed_risk_pct: int
    risk_delta_pct: int
    risk_category: str  # LOW, MEDIUM, HIGH
    complexity_delta: int
    dependency_delta: int
    downstream_dependents_affected: List[str]
    is_high_risk: bool
    explanation: str


@dataclass
class PullRequestRiskReport:
    """Comprehensive pre-merge risk assessment report for a pull request or branch."""
    title: str
    source_identifier: str  # e.g., "PR #42" or "main..feature/payments"
    base_ref: str
    head_ref: str
    total_files_changed: int
    total_insertions: int
    total_deletions: int
    total_churn: int
    overall_risk_level: str  # LOW, MEDIUM, HIGH
    overall_risk_score_pct: int
    high_risk_files_count: int
    medium_risk_files_count: int
    low_risk_files_count: int
    total_downstream_dependents_impacted: int
    newly_introduced_cycles_count: int
    files_detail: List[PRFileRiskDetail]
    key_risk_drivers: List[str]
    recommended_reviews: List[str]
    suggested_verification_tests: List[str]
    scientific_disclaimer: str = (
        "PR risk assessments are estimates derived by recalculating structural and churn signals "
        "and evaluating the modified component representation through the trained model. "
        "They indicate architectural risk exposure, not confirmed software defects."
    )

    def to_dict(self) -> Dict:
        return {
            "title": self.title,
            "source": self.source_identifier,
            "base_branch": self.base_ref,
            "head_branch": self.head_ref,
            "overall_risk_level": self.overall_risk_level,
            "overall_risk_score_pct": self.overall_risk_score_pct,
            "total_files_changed": self.total_files_changed,
            "total_churn": self.total_churn,
            "high_risk_files_count": self.high_risk_files_count,
            "downstream_impacted_count": self.total_downstream_dependents_impacted,
            "key_risk_drivers": self.key_risk_drivers,
            "recommended_reviews": self.recommended_reviews,
            "suggested_tests": self.suggested_verification_tests
        }


class PullRequestRiskAnalyzer:
    """Evaluates pull request changes from Git branch comparisons or GitHub PR URLs."""

    def __init__(
        self,
        repo_path: Path,
        model_bundle: TrainedModelBundle,
        dep_graph: DependencyGraph,
        component_profiles: List[ComponentRiskProfile]
    ):
        self.repo_path = Path(repo_path).resolve()
        self.bundle = model_bundle
        self.dep_graph = dep_graph
        self.profiles = component_profiles
        self.impact_analyzer = ChangeImpactAnalyzer(model_bundle, dep_graph, component_profiles)
        self.predictor = RiskPredictor()

    def analyze_local_branch_diff(
        self,
        base_branch: str,
        feature_branch: str
    ) -> PullRequestRiskReport:
        """Analyzes proposed changes between two local Git branches (e.g. main..feature)."""
        diff_cmd = ["git", "diff", f"{base_branch}..{feature_branch}"]
        try:
            res = subprocess.run(
                diff_cmd, cwd=str(self.repo_path), capture_output=True, text=True, check=True
            )
            raw_diff = res.stdout
        except Exception as e:
            logger.warning(f"git diff failed for {base_branch}..{feature_branch}: {e}")
            raw_diff = ""

        diff_items = PRDiffParser.parse_unified_diff(raw_diff)

        return self._generate_report(
            title=f"Branch Comparison: {base_branch} ➔ {feature_branch}",
            source_identifier=f"{base_branch}..{feature_branch}",
            base_ref=base_branch,
            head_ref=feature_branch,
            diff_items=diff_items
        )

    def analyze_github_pr(
        self,
        pr_url: str,
        github_token: Optional[str] = None
    ) -> Optional[PullRequestRiskReport]:
        """Fetches and evaluates a remote GitHub Pull Request by URL."""
        client = GitHubPRClient(token=github_token)
        metadata = client.fetch_pr(pr_url)
        if not metadata:
            return None

        diff_items = PRDiffParser.parse_unified_diff(metadata.raw_diff)

        # If diff parser found no items but files_changed are listed
        if not diff_items and metadata.files_changed:
            for fp in metadata.files_changed:
                diff_items.append(FileDiffItem(
                    filepath=fp,
                    change_type="MODIFIED",
                    insertions=metadata.additions // max(1, len(metadata.files_changed)),
                    deletions=metadata.deletions // max(1, len(metadata.files_changed)),
                    patch=""
                ))

        return self._generate_report(
            title=metadata.title,
            source_identifier=f"PR #{metadata.pr_number} ({metadata.owner}/{metadata.repo})",
            base_ref=metadata.base_ref,
            head_ref=metadata.head_ref,
            diff_items=diff_items
        )

    def _generate_report(
        self,
        title: str,
        source_identifier: str,
        base_ref: str,
        head_ref: str,
        diff_items: List[FileDiffItem]
    ) -> PullRequestRiskReport:
        """Core synthesis of file diffs, ML risk deltas, and downstream exposure."""
        file_details: List[PRFileRiskDetail] = []
        tot_ins = sum(item.insertions for item in diff_items)
        tot_dels = sum(item.deletions for item in diff_items)
        tot_churn = tot_ins + tot_dels

        high_c = 0
        med_c = 0
        low_c = 0
        all_downstream_impacted: set = set()
        key_drivers: List[str] = []
        review_comps: List[str] = []
        suggested_tests: List[str] = []

        g = self.dep_graph.graph

        for item in diff_items:
            norm_fp = item.filepath.replace("\\", "/")
            # Evaluate using ChangeImpactAnalyzer
            impact = self.impact_analyzer.analyze_file_diff(
                filepath=norm_fp,
                diff_text=item.patch,
                change_type=item.change_type,
                insertions=item.insertions,
                deletions=item.deletions
            )

            # Trace downstream dependents in dependency graph
            dependents = []
            if g.has_node(norm_fp):
                dependents = list(g.predecessors(norm_fp))
            else:
                for n in g.nodes():
                    if n.endswith(norm_fp) or norm_fp.endswith(n):
                        dependents = list(g.predecessors(n))
                        break

            all_downstream_impacted.update(dependents)

            if impact:
                curr_pct = impact.current_risk_pct
                prop_pct = impact.proposed_risk_pct
                delta_pct = impact.risk_delta_pct
                cat = impact.proposed_risk_category
                c_delta = 0
                d_delta = impact.dependency_changes
                for s in impact.changed_signals:
                    if s.signal_name == "complexity":
                        c_delta = int(s.after_value - s.before_value)
                expl = impact.explanation
            else:
                # Default for unmodeled / non-python files
                curr_pct = 20
                prop_pct = 20
                delta_pct = 0
                cat = "LOW"
                c_delta = 0
                d_delta = 0
                expl = "Component change evaluated on lines and file type."

            is_high = (cat == "HIGH" or prop_pct >= 66)
            if is_high:
                high_c += 1
                review_comps.append(norm_fp)
                stem = Path(norm_fp).stem
                suggested_tests.append(f"Run comprehensive unit & integration tests for {stem}")
            elif cat == "MEDIUM" or prop_pct >= 33:
                med_c += 1
            else:
                low_c += 1

            if c_delta >= 5:
                key_drivers.append(f"{Path(norm_fp).name}: Cyclomatic complexity increased by +{c_delta}")
            if d_delta > 0:
                key_drivers.append(f"{Path(norm_fp).name}: Added {d_delta} new external dependency coupling(s)")
            if item.insertions + item.deletions >= 200:
                key_drivers.append(f"{Path(norm_fp).name}: High churn volume (+{item.insertions}/-{item.deletions} lines)")

            file_details.append(PRFileRiskDetail(
                filepath=norm_fp,
                change_type=item.change_type,
                insertions=item.insertions,
                deletions=item.deletions,
                current_risk_pct=curr_pct,
                proposed_risk_pct=prop_pct,
                risk_delta_pct=delta_pct,
                risk_category=cat,
                complexity_delta=c_delta,
                dependency_delta=d_delta,
                downstream_dependents_affected=dependents,
                is_high_risk=is_high,
                explanation=expl
            ))

        # Sort file details by proposed risk descending
        file_details.sort(key=lambda f: f.proposed_risk_pct, reverse=True)

        # Aggregate PR risk score
        if file_details:
            avg_risk = sum(f.proposed_risk_pct for f in file_details) / len(file_details)
            max_risk = max(f.proposed_risk_pct for f in file_details)
            # Weighted: 50% max file risk + 50% avg file risk
            pr_score_pct = int(round(0.50 * max_risk + 0.50 * avg_risk))
        else:
            pr_score_pct = 10

        if pr_score_pct >= 60 or high_c >= 2:
            overall_cat = "HIGH"
        elif pr_score_pct >= 30 or med_c >= 2:
            overall_cat = "MEDIUM"
        else:
            overall_cat = "LOW"

        if not key_drivers:
            key_drivers.append("Low complexity shift across modified files.")
            suggested_tests.append("Execute standard regression test suite.")

        return PullRequestRiskReport(
            title=title,
            source_identifier=source_identifier,
            base_ref=base_ref,
            head_ref=head_ref,
            total_files_changed=len(diff_items),
            total_insertions=tot_ins,
            total_deletions=tot_dels,
            total_churn=tot_churn,
            overall_risk_level=overall_cat,
            overall_risk_score_pct=pr_score_pct,
            high_risk_files_count=high_c,
            medium_risk_files_count=med_c,
            low_risk_files_count=low_c,
            total_downstream_dependents_impacted=len(all_downstream_impacted),
            newly_introduced_cycles_count=0,
            files_detail=file_details,
            key_risk_drivers=key_drivers[:5],
            recommended_reviews=review_comps[:5],
            suggested_verification_tests=list(set(suggested_tests))[:5]
        )
