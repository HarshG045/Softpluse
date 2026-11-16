"""
ImpactPropagator: Identifies components potentially affected by a set of changed files,
by traversing the dependency graph from modified nodes to their transitive dependents.

This module does NOT claim that affected components will fail — it identifies components
that have a structural dependency relationship with the changed files.
"""
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Set

from analysis.dependency_analyzer import DependencyGraph

logger = logging.getLogger(__name__)

# Describes how a component was included in the impact set
IMPACT_REASON_DIRECT = "Directly modified"
IMPACT_REASON_DEPENDENT_1 = "Direct dependent (imports modified component)"
IMPACT_REASON_TRANSITIVE = "Transitive dependent (indirect structural relationship)"


@dataclass
class AffectedComponent:
    """A component included in the change impact set with its reason and hop distance."""
    filepath: str
    component_name: str
    reason: str               # How it was included
    hop_distance: int         # 0 = directly modified, 1 = direct dependent, etc.
    dependency_chain: List[str] = field(default_factory=list)  # Path from change to this node


class ImpactPropagator:
    """
    Traverses the dependency graph to find components affected by a set of changes.
    
    Starting from directly modified files, it walks upstream (to dependents — components
    that IMPORT the modified files) since those callers may be impacted by interface changes.
    
    The maximum traversal depth is configurable to prevent full-graph traversal on
    large, densely connected repositories.
    """

    def __init__(self, dep_graph: DependencyGraph, max_depth: int = 3):
        """
        Args:
            dep_graph: The dependency graph for the analyzed repository state.
            max_depth: Maximum number of hops from changed files (1 = direct dependents only).
        """
        self.graph = dep_graph
        self.max_depth = max_depth

    def find_affected(
        self,
        changed_files: List[str],
        exclude_added_deleted: bool = False
    ) -> List[AffectedComponent]:
        """
        Returns the full affected component set for the given changed files.

        Args:
            changed_files: List of file paths that were added, modified, or deleted.
            exclude_added_deleted: If True, skip files that are wholly new or removed
                                   since they have no prior dependents to worry about.

        Returns:
            List of AffectedComponent sorted by hop_distance then filepath.
        """
        if not changed_files:
            return []

        g = self.graph.graph
        affected: Dict[str, AffectedComponent] = {}

        # All nodes in graph (normalized)
        graph_nodes = {n.replace("\\", "/").lower(): n for n in g.nodes()}

        def normalize(fp: str) -> str:
            return fp.replace("\\", "/").lower()

        def find_graph_node(fp: str) -> Optional[str]:
            """Finds the graph node matching a file path (flexible matching)."""
            norm = normalize(fp)
            # Direct match
            direct = graph_nodes.get(norm)
            if direct:
                return direct
            # Suffix match
            for key, val in graph_nodes.items():
                if key.endswith(norm) or norm.endswith(key):
                    return val
            return None

        # Seed: directly changed files
        queue: List[tuple] = []  # (node, hop_distance, chain)
        for fp in changed_files:
            node = find_graph_node(fp)
            if node:
                affected[node] = AffectedComponent(
                    filepath=node,
                    component_name=Path(node).name,
                    reason=IMPACT_REASON_DIRECT,
                    hop_distance=0,
                    dependency_chain=[node]
                )
                queue.append((node, 0, [node]))
            else:
                # File not in graph (e.g., new file, non-Python, or outside analyzed set)
                norm_fp = fp.replace("\\", "/")
                affected[norm_fp] = AffectedComponent(
                    filepath=norm_fp,
                    component_name=Path(norm_fp).name,
                    reason=IMPACT_REASON_DIRECT + " (not in dependency graph)",
                    hop_distance=0,
                    dependency_chain=[norm_fp]
                )

        # BFS traversal: walk predecessors (dependents — those that import the changed node)
        visited: Set[str] = set(affected.keys())

        while queue:
            node, depth, chain = queue.pop(0)
            if depth >= self.max_depth:
                continue

            if not g.has_node(node):
                continue

            # Predecessors in a DiGraph are the IMPORTERS of this node
            for dependent in g.predecessors(node):
                if dependent in visited:
                    continue
                visited.add(dependent)

                new_depth = depth + 1
                new_chain = chain + [dependent]

                if new_depth == 1:
                    reason = IMPACT_REASON_DEPENDENT_1
                else:
                    reason = f"{IMPACT_REASON_TRANSITIVE} (depth {new_depth})"

                affected[dependent] = AffectedComponent(
                    filepath=dependent,
                    component_name=Path(dependent).name,
                    reason=reason,
                    hop_distance=new_depth,
                    dependency_chain=new_chain
                )
                queue.append((dependent, new_depth, new_chain))

        result = sorted(affected.values(), key=lambda x: (x.hop_distance, x.filepath))
        return result

    def summarize_impact(
        self,
        affected: List[AffectedComponent]
    ) -> Dict:
        """
        Returns a summary dict of the impact analysis.
        """
        direct = [a for a in affected if a.hop_distance == 0]
        hop1 = [a for a in affected if a.hop_distance == 1]
        transitive = [a for a in affected if a.hop_distance > 1]

        return {
            "total_affected": len(affected),
            "directly_modified": len(direct),
            "direct_dependents": len(hop1),
            "transitive_dependents": len(transitive),
            "max_depth_reached": max((a.hop_distance for a in affected), default=0)
        }
