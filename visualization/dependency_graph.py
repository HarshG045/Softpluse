"""
DependencyGraphVisualizer: Interactive Plotly network graph rendering for component architectures and What-If edge changes.
"""
from typing import Dict, List, Optional, Set, Tuple
import networkx as nx
import numpy as np
import plotly.graph_objects as go

from analysis.dependency_analyzer import DependencyGraph

DARK_LAYOUT = {
    "paper_bgcolor": "rgba(0,0,0,0)",
    "plot_bgcolor": "rgba(0,0,0,0)",
    "font": {"color": "#c9d1d9", "family": "Inter, -apple-system, sans-serif"},
    "margin": dict(l=10, r=10, t=25, b=10),
}


def create_interactive_dependency_figure(
    dep_graph: DependencyGraph,
    risk_profiles_map: Optional[Dict[str, Dict]] = None,
    focus_component: Optional[str] = None,
    simulated_edge: Optional[Tuple[str, str, str]] = None,  # (source, target, action: "add"|"remove")
    max_nodes: int = 50
) -> go.Figure:
    """
    Renders an interactive Plotly network diagram of components and dependency edges.
    Supports focus neighborhood subgraphs and What-If hypothetical edge styling.
    """
    g = dep_graph.graph

    if g.number_of_nodes() == 0:
        fig = go.Figure()
        fig.update_layout(**DARK_LAYOUT, title="No components found in dependency graph.")
        return fig

    # If focus_component is given, extract neighborhood
    if focus_component and g.has_node(focus_component):
        nodes_to_keep = {focus_component}
        nodes_to_keep.update(g.successors(focus_component))
        nodes_to_keep.update(g.predecessors(focus_component))
        
        # If simulated target is involved, add it
        if simulated_edge:
            nodes_to_keep.add(simulated_edge[0])
            nodes_to_keep.add(simulated_edge[1])

        sub_g = g.subgraph(nodes_to_keep).copy()
    else:
        # Limit to top connected nodes if too large
        if g.number_of_nodes() > max_nodes:
            degrees = dict(g.degree())
            sorted_nodes = sorted(degrees.keys(), key=lambda k: degrees[k], reverse=True)[:max_nodes]
            sub_g = g.subgraph(sorted_nodes).copy()
        else:
            sub_g = g.copy()

    # If simulated edge is "add", add it to subgraph
    if simulated_edge and simulated_edge[2] == "add":
        sub_g.add_node(simulated_edge[0])
        sub_g.add_node(simulated_edge[1])
        sub_g.add_edge(simulated_edge[0], simulated_edge[1])

    # Compute 2D node layout using spring_layout with deterministic seed
    pos = nx.spring_layout(sub_g, seed=42, k=1.2 / np.sqrt(max(1, sub_g.number_of_nodes())), iterations=50)

    # 1. Base Edges
    edge_x = []
    edge_y = []
    sim_edge_x = []
    sim_edge_y = []

    for edge in sub_g.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]

        is_this_sim = (
            simulated_edge and 
            simulated_edge[0] == edge[0] and 
            simulated_edge[1] == edge[1]
        )

        if is_this_sim:
            sim_edge_x.extend([x0, x1, None])
            sim_edge_y.extend([y0, y1, None])
        else:
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])

    fig = go.Figure()

    # Standard dependency edges
    fig.add_trace(
        go.Scatter(
            x=edge_x, y=edge_y,
            line=dict(width=1.2, color="#30363d"),
            hoverinfo="none",
            mode="lines",
            name="Dependency Edge"
        )
    )

    # What-If Hypothetical Edge
    if sim_edge_x:
        fig.add_trace(
            go.Scatter(
                x=sim_edge_x, y=sim_edge_y,
                line=dict(width=2.5, color="#a371f7", dash="dash"),
                hoverinfo="text",
                hovertext="Simulated What-If Dependency Edge",
                mode="lines",
                name="Simulated Dependency"
            )
        )

    # 2. Nodes
    node_x = []
    node_y = []
    node_colors = []
    node_sizes = []
    node_texts = []
    node_borders = []

    risk_colors = {"HIGH": "#f85149", "MEDIUM": "#d29922", "LOW": "#2ea043"}

    for node in sub_g.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)

        # Profile details
        prof = (risk_profiles_map or {}).get(node, {})
        cat = prof.get("risk_category", "LOW")
        prob = prof.get("risk_percentage", 20)
        dep_c = prof.get("dependency_count", len(list(sub_g.successors(node))))
        dept_c = prof.get("dependent_count", len(list(sub_g.predecessors(node))))

        # Highlighting focused component
        if focus_component and node == focus_component:
            node_borders.append("#58a6ff")
            node_sizes.append(22)
        else:
            node_borders.append("#161b22")
            node_sizes.append(15)

        node_colors.append(risk_colors.get(cat, "#58a6ff"))

        short_name = node.split("/")[-1]
        node_texts.append(
            f"<b>{node}</b><br>"
            f"Predicted Risk: <b>{prob}% ({cat})</b><br>"
            f"Dependencies (Out): {dep_c}<br>"
            f"Dependents (In): {dept_c}"
        )

    fig.add_trace(
        go.Scatter(
            x=node_x, y=node_y,
            mode="markers+text",
            marker=dict(
                size=node_sizes,
                color=node_colors,
                line=dict(width=2, color=node_borders)
            ),
            text=[n.split("/")[-1] for n in sub_g.nodes()],
            textposition="top center",
            textfont=dict(size=10, color="#8b949e"),
            hoverinfo="text",
            hovertext=node_texts,
            name="Components"
        )
    )

    fig.update_layout(
        **DARK_LAYOUT,
        showlegend=False,
        height=380,
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
    )

    return fig
