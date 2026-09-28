"""
DependencyAnalyzer: Builds component dependency graph using NetworkX and calculates graph structural metrics.
"""
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple
import networkx as nx

from analysis.code_analyzer import CodeMetrics

logger = logging.getLogger(__name__)


@dataclass
class ComponentStructuralMetrics:
    """Graph structural metrics for a specific component."""
    filepath: str
    dependency_count: int = 0      # Out-degree (modules this component depends on)
    dependent_count: int = 0       # In-degree (modules that depend on this component)
    degree: int = 0                # Total degree
    centrality: float = 0.0        # Degree or Betweenness centrality
    in_degree: int = 0
    out_degree: int = 0
    direct_dependencies: List[str] = field(default_factory=list)
    direct_dependents: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict:
        return {
            "filepath": self.filepath,
            "dependency_count": self.dependency_count,
            "dependent_count": self.dependent_count,
            "degree": self.degree,
            "centrality": round(self.centrality, 4),
            "in_degree": self.in_degree,
            "out_degree": self.out_degree,
            "direct_dependencies": self.direct_dependencies,
            "direct_dependents": self.direct_dependents
        }


class DependencyGraph:
    """Encapsulates a NetworkX DiGraph representing component dependency architecture."""

    def __init__(self, graph: Optional[nx.DiGraph] = None):
        self.graph = graph if graph is not None else nx.DiGraph()

    def add_component(self, filepath: str, **attrs):
        """Adds a component node."""
        self.graph.add_node(filepath, **attrs)

    def add_dependency(self, source: str, target: str):
        """Adds a directed dependency edge: source depends on target (source -> target)."""
        if not self.graph.has_node(source):
            self.graph.add_node(source)
        if not self.graph.has_node(target):
            self.graph.add_node(target)
        self.graph.add_edge(source, target)

    def remove_dependency(self, source: str, target: str):
        """Removes a dependency edge if it exists."""
        if self.graph.has_edge(source, target):
            self.graph.remove_edge(source, target)

    def get_metrics_for_all(self) -> Dict[str, ComponentStructuralMetrics]:
        """Calculates structural metrics for all nodes in the graph."""
        n_nodes = self.graph.number_of_nodes()
        if n_nodes == 0:
            return {}

        # Degree centrality
        try:
            centrality_map = nx.degree_centrality(self.graph)
        except Exception:
            centrality_map = {n: 0.0 for n in self.graph.nodes}

        results: Dict[str, ComponentStructuralMetrics] = {}
        for node in self.graph.nodes:
            # Out-edges = dependencies (outgoing links)
            deps = list(self.graph.successors(node))
            # In-edges = dependents (incoming links)
            dependents = list(self.graph.predecessors(node))
            
            out_deg = len(deps)
            in_deg = len(dependents)
            deg = in_deg + out_deg
            cent = centrality_map.get(node, 0.0)

            results[node] = ComponentStructuralMetrics(
                filepath=node,
                dependency_count=out_deg,
                dependent_count=in_deg,
                degree=deg,
                centrality=cent,
                in_degree=in_deg,
                out_degree=out_deg,
                direct_dependencies=deps,
                direct_dependents=dependents
            )

        return results

    def get_neighborhood(self, filepath: str, radius: int = 1) -> nx.DiGraph:
        """Extracts the immediate neighborhood subgraph around a specific component."""
        if not self.graph.has_node(filepath):
            g = nx.DiGraph()
            g.add_node(filepath)
            return g

        # Get direct successors and predecessors
        nodes = {filepath}
        nodes.update(self.graph.successors(filepath))
        nodes.update(self.graph.predecessors(filepath))

        return self.graph.subgraph(nodes).copy()

    def detect_cycles(self) -> List[List[str]]:
        """Detects circular dependency cycles in the architecture."""
        try:
            cycles = list(nx.simple_cycles(self.graph))
            return cycles
        except Exception:
            return []

    def get_architectural_hotspots(self) -> List[Dict]:
        """Identifies central architectural bridge components based on betweenness & degree."""
        try:
            betweenness = nx.betweenness_centrality(self.graph)
        except Exception:
            betweenness = {n: 0.0 for n in self.graph.nodes}

        struct_map = self.get_metrics_for_all()
        hotspots = []
        for node, sm in struct_map.items():
            b_score = betweenness.get(node, 0.0)
            # Architectural fan-in (dependents) vs fan-out (dependencies)
            hotspots.append({
                "component": node,
                "fan_in": sm.dependent_count,
                "fan_out": sm.dependency_count,
                "total_coupling": sm.degree,
                "betweenness_centrality": round(b_score, 4),
                "degree_centrality": round(sm.centrality, 4),
                "is_hub": (sm.degree >= 3 or b_score > 0.1)
            })

        hotspots.sort(key=lambda x: (x["betweenness_centrality"], x["total_coupling"]), reverse=True)
        return hotspots

    def get_package_clusters(self) -> Dict[str, Dict]:
        """Groups components into subsystem packages and calculates package-level coupling & instability."""
        pkg_nodes: Dict[str, List[str]] = {}
        for node in self.graph.nodes:
            parts = Path(node).parts
            pkg = parts[0] if len(parts) > 1 else "root"
            pkg_nodes.setdefault(pkg, []).append(node)

        clusters = {}
        for pkg, nodes in pkg_nodes.items():
            # Efferent coupling (Ce): edges from nodes in this pkg to nodes outside
            ce = 0
            # Afferent coupling (Ca): edges from nodes outside to nodes in this pkg
            ca = 0
            
            for n in nodes:
                for target in self.graph.successors(n):
                    if target not in nodes:
                        ce += 1
                for source in self.graph.predecessors(n):
                    if source not in nodes:
                        ca += 1

            # Martin's Instability Metric: I = Ce / (Ca + Ce)
            instability = round(ce / (ca + ce), 3) if (ca + ce) > 0 else 0.0

            clusters[pkg] = {
                "package": pkg,
                "component_count": len(nodes),
                "components": nodes,
                "afferent_coupling_ca": ca,
                "efferent_coupling_ce": ce,
                "instability_index": instability
            }

        return clusters

    def clone(self) -> "DependencyGraph":
        """Returns a deep copy of the dependency graph."""
        return DependencyGraph(self.graph.copy())



class DependencyAnalyzer:
    """Resolves file-level imports into an intra-repository DependencyGraph."""

    def __init__(self, repo_path: Path):
        self.repo_path = repo_path.resolve()

    def build_graph(self, code_metrics_map: Dict[str, CodeMetrics]) -> DependencyGraph:
        """
        Builds a dependency graph from AST code metrics.
        Matches imported module names with actual repository components.
        """
        dep_graph = DependencyGraph()

        # Add all known components
        for fp in code_metrics_map.keys():
            dep_graph.add_component(fp)

        # Build lookup table: module name / stem / dotted path -> filepath
        module_lookup: Dict[str, str] = {}
        for fp in code_metrics_map.keys():
            stem = Path(fp).stem
            module_lookup[stem] = fp
            # dotted path (e.g. services.payment_service)
            dotted = fp.replace("/", ".").replace("\\", ".").replace(".py", "")
            module_lookup[dotted] = fp
            # relative sub-paths
            parts = Path(fp).parts
            if len(parts) >= 2:
                sub_dot = ".".join(parts).replace(".py", "")
                module_lookup[sub_dot] = fp

        # Connect dependencies based on imported modules
        for source_fp, metrics in code_metrics_map.items():
            for imp in metrics.imported_modules:
                # Direct match or partial match
                target_fp = module_lookup.get(imp)
                if not target_fp:
                    # check if any key in module_lookup ends with imp or starts with imp
                    for k, v in module_lookup.items():
                        if k.endswith(f".{imp}") or k == imp or imp.endswith(f".{k}"):
                            target_fp = v
                            break

                if target_fp and target_fp != source_fp:
                    dep_graph.add_dependency(source_fp, target_fp)


        return dep_graph
