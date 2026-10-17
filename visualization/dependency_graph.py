"""
DependencyGraphVisualizer: Professional, evolution-aware interactive Plotly network
graph rendering for software architecture exploration, hierarchical component dependencies,
and counterfactual What-If change analysis.
"""
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple
import networkx as nx
import numpy as np
import plotly.graph_objects as go

from analysis.dependency_analyzer import DependencyGraph

DARK_LAYOUT = {
    "paper_bgcolor": "rgba(0,0,0,0)",
    "plot_bgcolor": "rgba(0,0,0,0)",
    "font": {"color": "#c9d1d9", "family": "Inter, -apple-system, sans-serif"},
    "margin": dict(l=15, r=15, t=35, b=20),
}

RISK_COLORS = {
    "HIGH": "#f85149",
    "MEDIUM": "#d29922",
    "LOW": "#2ea043"
}

RISK_ICONS = {
    "HIGH": "[H]",
    "MEDIUM": "[M]",
    "LOW": "[L]"
}


def _get_package_for_node(node: str) -> str:
    """Extracts top-level directory/package name for a file path."""
    parts = Path(node).parts
    return parts[0] if len(parts) > 1 else "root"


def compute_hierarchical_neighborhood_layout(
    sub_g: nx.DiGraph,
    focus_component: Optional[str] = None,
    depth: int = 1,
    direction_filter: str = "both"
) -> Tuple[Dict[str, Tuple[float, float]], Dict[str, int]]:
    """
    Computes a clean, left-to-right hierarchical layered layout:
      - Left columns (X < 0): Upstream dependencies (modules the focus component imports).
      - Center column (X = 0): Selected focus component.
      - Right columns (X > 0): Downstream dependents (modules that import the focus component).
    
    Returns:
      (pos_map, layer_map)
    """
    pos: Dict[str, Tuple[float, float]] = {}
    node_layers: Dict[str, int] = {}

    if sub_g.number_of_nodes() == 0:
        return pos, node_layers

    # If focus component is active
    if focus_component and sub_g.has_node(focus_component):
        node_layers[focus_component] = 0

        # Discover upstream dependencies (successors: what focus imports)
        curr_deps = {focus_component}
        for d in range(1, depth + 1):
            next_deps = set()
            for n in curr_deps:
                for target in sub_g.successors(n):
                    if target not in node_layers:
                        node_layers[target] = -d
                        next_deps.add(target)
            curr_deps = next_deps

        # Discover downstream dependents (predecessors: what imports focus)
        curr_dependents = {focus_component}
        for d in range(1, depth + 1):
            next_dependents = set()
            for n in curr_dependents:
                for source in sub_g.predecessors(n):
                    if source not in node_layers:
                        node_layers[source] = d
                        next_dependents.add(source)
            curr_dependents = next_dependents

        # Any remaining nodes in sub_g placed at Level 0 or adjacent
        for n in sub_g.nodes():
            if n not in node_layers:
                node_layers[n] = 0

    else:
        # Full repository layout without focus component:
        # Use DAG condensation generations to handle cyclic graphs gracefully
        try:
            condensation = nx.condensation(sub_g)
            topo_gen = list(nx.topological_generations(condensation))
            mapping = condensation.graph.get("mapping", {})
            for gen_idx, scc_nodes in enumerate(topo_gen):
                for scc_id in scc_nodes:
                    members = condensation.nodes[scc_id].get("members", set())
                    for m in members:
                        if m in sub_g:
                            node_layers[m] = gen_idx
        except Exception:
            # Fallback simple degree ordering
            degrees = dict(sub_g.degree())
            sorted_nodes = sorted(sub_g.nodes(), key=lambda n: degrees.get(n, 0), reverse=True)
            for idx, n in enumerate(sorted_nodes):
                node_layers[n] = idx % 4

    # Group nodes by layer
    layers: Dict[int, List[str]] = {}
    for n, l in node_layers.items():
        layers.setdefault(l, []).append(n)

    # Position nodes: X proportional to layer, Y spaced evenly
    x_spacing = 2.4
    y_spacing = 1.35

    sorted_layer_keys = sorted(layers.keys())
    for layer_key in sorted_layer_keys:
        layer_nodes = layers[layer_key]
        x_val = layer_key * x_spacing
        k = len(layer_nodes)

        # Sort layer nodes alphabetically or by degree for consistency
        layer_nodes.sort()

        for idx, node in enumerate(layer_nodes):
            if k == 1:
                y_val = 0.0
            else:
                y_val = (idx - (k - 1) / 2.0) * y_spacing
            pos[node] = (x_val, y_val)

    return pos, node_layers


def compute_radial_layout(
    sub_g: nx.DiGraph,
    focus_component: Optional[str] = None
) -> Dict[str, Tuple[float, float]]:
    """Computes a radial layout with focus node at center and neighbors on concentric arcs."""
    pos: Dict[str, Tuple[float, float]] = {}
    if sub_g.number_of_nodes() == 0:
        return pos

    if focus_component and sub_g.has_node(focus_component):
        pos[focus_component] = (0.0, 0.0)

        # Successors (dependencies) on left arc [2*pi/3, 4*pi/3]
        succs = sorted([n for n in sub_g.successors(focus_component) if n != focus_component])
        # Predecessors (dependents) on right arc [-pi/3, pi/3]
        preds = sorted([n for n in sub_g.predecessors(focus_component) if n != focus_component and n not in succs])
        # Mutual / cyclic nodes
        mutual = sorted([n for n in sub_g.successors(focus_component) if n in sub_g.predecessors(focus_component) and n != focus_component])

        # Radius
        r1 = 2.2
        if succs:
            angles = np.linspace(2 * np.pi / 3, 4 * np.pi / 3, len(succs))
            for n, th in zip(succs, angles):
                pos[n] = (r1 * np.cos(th), r1 * np.sin(th))

        if preds:
            angles = np.linspace(-np.pi / 3, np.pi / 3, len(preds))
            for n, th in zip(preds, angles):
                pos[n] = (r1 * np.cos(th), r1 * np.sin(th))

        # Remaining nodes on outer ring
        remainder = [n for n in sub_g.nodes() if n not in pos]
        if remainder:
            r2 = 3.8
            angles = np.linspace(0, 2 * np.pi, len(remainder), endpoint=False)
            for n, th in zip(remainder, angles):
                pos[n] = (r2 * np.cos(th), r2 * np.sin(th))

    else:
        # Radial shell layout based on degree
        try:
            degrees = dict(sub_g.degree())
            hubs = [n for n, d in degrees.items() if d >= 3]
            others = [n for n in sub_g.nodes() if n not in hubs]
            shells = [hubs, others] if hubs and others else [list(sub_g.nodes())]
            pos = nx.shell_layout(sub_g, nlist=shells)
        except Exception:
            pos = nx.spring_layout(sub_g, seed=42)

    return pos


def compute_grouped_layout(sub_g: nx.DiGraph) -> Dict[str, Tuple[float, float]]:
    """Groups components by directory/package clusters arranged cleanly."""
    pos: Dict[str, Tuple[float, float]] = {}
    if sub_g.number_of_nodes() == 0:
        return pos

    pkg_map: Dict[str, List[str]] = {}
    for n in sub_g.nodes():
        pkg = _get_package_for_node(n)
        pkg_map.setdefault(pkg, []).append(n)

    pkgs = sorted(pkg_map.keys())
    n_pkgs = len(pkgs)
    cluster_radius = 3.6

    for p_idx, pkg in enumerate(pkgs):
        theta_p = 2 * np.pi * p_idx / max(1, n_pkgs)
        cx = cluster_radius * np.cos(theta_p)
        cy = cluster_radius * np.sin(theta_p)

        members = sorted(pkg_map[pkg])
        m_count = len(members)

        if m_count == 1:
            pos[members[0]] = (cx, cy)
        else:
            local_r = 0.9 + 0.15 * m_count
            for m_idx, member in enumerate(members):
                theta_m = 2 * np.pi * m_idx / m_count
                pos[member] = (cx + local_r * np.cos(theta_m), cy + local_r * np.sin(theta_m))

    return pos


def create_architecture_overview_figure(
    dep_graph: DependencyGraph,
    risk_profiles_map: Optional[Dict[str, Dict]] = None,
    selected_package: Optional[str] = None
) -> go.Figure:
    """
    Renders the Architecture Overview Mode:
    A simplified repository map grouped by logical folders/packages, showing
    package-to-package dependency relationships, component counts, instability,
    and architectural coupling hotspots.
    """
    g = dep_graph.graph
    if g.number_of_nodes() == 0:
        fig = go.Figure()
        fig.update_layout(**DARK_LAYOUT, title="No components available in repository.")
        return fig

    # Build package clusters
    pkg_clusters = dep_graph.get_package_clusters()
    if not pkg_clusters:
        fig = go.Figure()
        fig.update_layout(**DARK_LAYOUT, title="No package clusters found.")
        return fig

    # Create inter-package directed graph
    pkg_g = nx.DiGraph()
    for pkg, pdata in pkg_clusters.items():
        # Determine highest risk category within package
        comp_risks = [
            (risk_profiles_map or {}).get(comp, {}).get("risk_category", "LOW")
            for comp in pdata["components"]
        ]
        max_risk = "HIGH" if "HIGH" in comp_risks else ("MEDIUM" if "MEDIUM" in comp_risks else "LOW")
        avg_risk_pct = int(np.mean([
            (risk_profiles_map or {}).get(comp, {}).get("risk_percentage", 20)
            for comp in pdata["components"]
        ])) if pdata["components"] else 0

        pkg_g.add_node(
            pkg,
            component_count=pdata["component_count"],
            afferent_ca=pdata["afferent_coupling_ca"],
            efferent_ce=pdata["efferent_coupling_ce"],
            instability=pdata["instability_index"],
            max_risk=max_risk,
            avg_risk_pct=avg_risk_pct,
            components=pdata["components"]
        )

    # Calculate cross-package edge weights
    for u, v in g.edges():
        pkg_u = _get_package_for_node(u)
        pkg_v = _get_package_for_node(v)
        if pkg_u != pkg_v and pkg_u in pkg_g and pkg_v in pkg_g:
            if pkg_g.has_edge(pkg_u, pkg_v):
                pkg_g[pkg_u][pkg_v]["weight"] += 1
            else:
                pkg_g.add_edge(pkg_u, pkg_v, weight=1)

    # Position package nodes in a clean circular layout
    pkgs = sorted(pkg_g.nodes())
    n_pkgs = len(pkgs)
    r = 2.8
    pos: Dict[str, Tuple[float, float]] = {}
    for idx, p in enumerate(pkgs):
        th = 2 * np.pi * idx / max(1, n_pkgs)
        pos[p] = (r * np.cos(th), r * np.sin(th))

    fig = go.Figure()

    # Draw inter-package dependency edges with directional arrows
    for u, v, data in pkg_g.edges(data=True):
        weight = data.get("weight", 1)
        x0, y0 = pos[u]
        x1, y1 = pos[v]

        dx = x1 - x0
        dy = y1 - y0
        dist = np.sqrt(dx * dx + dy * dy)
        if dist > 0.001:
            # Offset so arrow tip touches package node boundary cleanly
            offset_ratio = 0.22
            sx = x0 + dx * offset_ratio
            sy = y0 + dy * offset_ratio
            tx = x1 - dx * offset_ratio
            ty = y1 - dy * offset_ratio
        else:
            sx, sy, tx, ty = x0, y0, x1, y1

        is_selected = selected_package and (u == selected_package or v == selected_package)
        edge_color = "#58a6ff" if is_selected else "#484f58"
        edge_width = min(6.0, 1.5 + weight * 0.8)

        # Arrow annotation
        fig.add_annotation(
            x=tx, y=ty,
            ax=sx, ay=sy,
            xref="x", yref="y",
            axref="x", ayref="y",
            showarrow=True,
            arrowhead=3,
            arrowsize=1.3,
            arrowwidth=edge_width,
            arrowcolor=edge_color,
            opacity=0.9
        )

        # Midpoint hover badge for edge
        mid_x = (x0 + x1) / 2.0
        mid_y = (y0 + y1) / 2.0
        fig.add_trace(
            go.Scatter(
                x=[mid_x], y=[mid_y],
                mode="text",
                text=[f"{weight} imports"],
                textposition="middle center",
                textfont=dict(size=9, color="#8b949e"),
                hoverinfo="text",
                hovertext=f"<b>{u} ➔ {v}</b><br>Cross-module imports: <b>{weight}</b>",
                showlegend=False
            )
        )

    # Draw package nodes
    node_x, node_y = [], []
    node_sizes = []
    node_colors = []
    node_borders = []
    node_labels = []
    hover_texts = []

    for p in pkgs:
        x, y = pos[p]
        node_x.append(x)
        node_y.append(y)

        attrs = pkg_g.nodes[p]
        cnt = attrs["component_count"]
        m_risk = attrs["max_risk"]
        avg_pct = attrs["avg_risk_pct"]
        instab = attrs["instability"]
        ca = attrs["afferent_ca"]
        ce = attrs["efferent_ce"]

        # Size based on component count
        size = min(56, 32 + cnt * 6)
        node_sizes.append(size)

        # Border color based on selection or risk
        if selected_package and p == selected_package:
            node_borders.append("#58a6ff")
        else:
            node_borders.append("#30363d")

        node_colors.append(RISK_COLORS.get(m_risk, "#2ea043"))
        node_labels.append(f"<b>{p}/</b><br>({cnt} files)")

        comps_list_html = "<br>• " + "<br>• ".join(attrs["components"][:5])
        if len(attrs["components"]) > 5:
            comps_list_html += f"<br>... (+{len(attrs['components']) - 5} more)"

        hover_texts.append(
            f"<b>Package: {p}/</b><br>"
            f"Components: <b>{cnt}</b><br>"
            f"Highest Risk: <b>{m_risk}</b> (Avg: {avg_pct}%)<br>"
            f"Afferent Coupling (Ca): <b>{ca}</b> (Incoming from other packages)<br>"
            f"Efferent Coupling (Ce): <b>{ce}</b> (Outgoing to other packages)<br>"
            f"Instability Index (I): <b>{instab:.2f}</b><br>"
            f"Included files:{comps_list_html}"
        )

    fig.add_trace(
        go.Scatter(
            x=node_x, y=node_y,
            mode="markers+text",
            marker=dict(
                size=node_sizes,
                color=node_colors,
                line=dict(width=3, color=node_borders),
                opacity=0.95
            ),
            text=node_labels,
            textposition="bottom center",
            textfont=dict(size=11, color="#f0f6fc", family="JetBrains Mono, monospace"),
            hoverinfo="text",
            hovertext=hover_texts,
            showlegend=False
        )
    )

    fig.update_layout(
        **DARK_LAYOUT,
        height=480,
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
    )

    return fig


def create_interactive_dependency_figure(
    dep_graph: DependencyGraph,
    risk_profiles_map: Optional[Dict[str, Dict]] = None,
    focus_component: Optional[str] = None,
    simulated_edge: Optional[Tuple[str, str, str]] = None,  # (source, target, action: "add"|"remove")
    max_nodes: int = 60,
    layout_type: str = "hierarchical",  # "hierarchical", "radial", "grouped"
    depth: int = 1,
    direction_filter: str = "both",  # "both", "dependencies", "dependents"
    risk_filter: str = "all",  # "all", "high", "medium", "low"
    show_labels: str = "full",  # "full", "truncated", "hover"
    highlight_risk: bool = True,
    show_edges: str = "all"  # "all", "direct"
) -> go.Figure:
    """
    Renders an interactive, evolution-aware dependency architecture explorer figure.
    Features:
      - Hierarchical 3-level directed dependency layout (Upstream -> Focus -> Downstream)
      - Clear directional arrowheads with node boundary offsets
      - Visual distinction for incoming vs outgoing vs circular dependency edges
      - Risk color-coding with accessible [H], [M], [L] badges
      - What-If simulation hypothetical edge overlay
      - Full collision-avoiding node positioning
    """
    g = dep_graph.graph

    if g.number_of_nodes() == 0:
        fig = go.Figure()
        fig.update_layout(**DARK_LAYOUT, title="No components found in dependency graph.")
        return fig

    # Detect cycles in the overall graph to flag circular dependencies
    all_cycles = dep_graph.detect_cycles()
    cycle_edges: Set[Tuple[str, str]] = set()
    for cyc in all_cycles:
        for idx in range(len(cyc)):
            cycle_edges.add((cyc[idx], cyc[(idx + 1) % len(cyc)]))

    # 1. Extract Active Subgraph based on focus_component and depth
    if focus_component and g.has_node(focus_component):
        nodes_to_keep: Set[str] = {focus_component}

        # Upstream dependencies (modules focus imports)
        if direction_filter in ("both", "dependencies"):
            curr = {focus_component}
            for _ in range(depth):
                nxt = set()
                for n in curr:
                    for target in g.successors(n):
                        nxt.add(target)
                nodes_to_keep.update(nxt)
                curr = nxt

        # Downstream dependents (modules that import focus)
        if direction_filter in ("both", "dependents"):
            curr = {focus_component}
            for _ in range(depth):
                nxt = set()
                for n in curr:
                    for source in g.predecessors(n):
                        nxt.add(source)
                nodes_to_keep.update(nxt)
                curr = nxt

        # Include simulated edge nodes if applicable
        if simulated_edge:
            nodes_to_keep.add(simulated_edge[0])
            nodes_to_keep.add(simulated_edge[1])

        sub_g = g.subgraph(nodes_to_keep).copy()

    else:
        # Full graph or top connected nodes
        if g.number_of_nodes() > max_nodes:
            degrees = dict(g.degree())
            sorted_nodes = sorted(degrees.keys(), key=lambda k: degrees[k], reverse=True)[:max_nodes]
            sub_g = g.subgraph(sorted_nodes).copy()
        else:
            sub_g = g.copy()

    # Apply Risk Filter if active (always keep focus_component)
    if risk_filter != "all" and risk_profiles_map:
        target_cat = risk_filter.upper()
        matching_nodes = set()
        for n in sub_g.nodes():
            prof = risk_profiles_map.get(n, {})
            cat = prof.get("risk_category", "LOW")
            if cat == target_cat or (focus_component and n == focus_component):
                matching_nodes.add(n)
        if matching_nodes:
            sub_g = sub_g.subgraph(matching_nodes).copy()

    # Handle What-If Simulated Edge
    if simulated_edge:
        src, tgt, action = simulated_edge
        if action == "add":
            sub_g.add_node(src)
            sub_g.add_node(tgt)
            sub_g.add_edge(src, tgt)
        elif action == "remove":
            if sub_g.has_edge(src, tgt):
                sub_g.remove_edge(src, tgt)

    # 2. Compute Layout Coordinates
    node_layers: Dict[str, int] = {}
    if layout_type == "hierarchical":
        pos, node_layers = compute_hierarchical_neighborhood_layout(
            sub_g=sub_g,
            focus_component=focus_component,
            depth=depth,
            direction_filter=direction_filter
        )
    elif layout_type == "radial":
        pos = compute_radial_layout(sub_g=sub_g, focus_component=focus_component)
    elif layout_type == "grouped":
        pos = compute_grouped_layout(sub_g=sub_g)
    else:
        pos = nx.spring_layout(sub_g, seed=42)

    fig = go.Figure()

    # 3. Render Directed Edges & Arrow Annotations
    sim_edge_x, sim_edge_y = [], []
    hover_edge_x, hover_edge_y, hover_edge_text = [], [], []

    for edge in sub_g.edges():
        u, v = edge[0], edge[1]
        if u not in pos or v not in pos:
            continue

        x0, y0 = pos[u]
        x1, y1 = pos[v]

        is_this_sim = (
            simulated_edge and
            simulated_edge[0] == u and
            simulated_edge[1] == v
        )

        is_circular = (u, v) in cycle_edges or (v, u) in sub_g.edges()

        # Determine edge styling based on direction and relationship
        if is_this_sim:
            edge_color = "#39d353" if simulated_edge[2] == "add" else "#f85149"
            edge_width = 3.0
            dash_style = "dash"
            sim_edge_x.extend([x0, x1, None])
            sim_edge_y.extend([y0, y1, None])
        elif is_circular:
            edge_color = "#f85149"  # Circular warning
            edge_width = 2.2
            dash_style = "dot"
        elif focus_component:
            if u == focus_component:
                # Outgoing dependency (focus imports v)
                edge_color = "#58a6ff"
                edge_width = 2.4
                dash_style = "solid"
            elif v == focus_component:
                # Incoming dependent (u imports focus)
                edge_color = "#a371f7"
                edge_width = 2.4
                dash_style = "solid"
            else:
                edge_color = "#21262d"  # Muted background edge
                edge_width = 1.0
                dash_style = "solid"
        else:
            edge_color = "#30363d"
            edge_width = 1.4
            dash_style = "solid"

        # Calculate offset so arrow tip touches node marker boundary cleanly
        dx = x1 - x0
        dy = y1 - y0
        dist = np.sqrt(dx * dx + dy * dy)
        if dist > 0.001:
            offset_start = 0.12
            offset_end = 0.14
            sx = x0 + dx * offset_start
            sy = y0 + dy * offset_start
            tx = x1 - dx * offset_end
            ty = y1 - dy * offset_end
        else:
            sx, sy, tx, ty = x0, y0, x1, y1

        # Add directed arrow annotation
        fig.add_annotation(
            x=tx, y=ty,
            ax=sx, ay=sy,
            xref="x", yref="y",
            axref="x", ayref="y",
            showarrow=True,
            arrowhead=3,
            arrowsize=1.2,
            arrowwidth=edge_width,
            arrowcolor=edge_color,
            opacity=0.88
        )

        # Edge hover mid-point marker
        mid_x = (x0 + x1) / 2.0
        mid_y = (y0 + y1) / 2.0
        rel_type = "Circular Dependency" if is_circular else "Direct Import Dependency"
        hover_edge_x.append(mid_x)
        hover_edge_y.append(mid_y)
        hover_edge_text.append(
            f"<b>Dependency Link</b><br>"
            f"<b>{u}</b> ➔ <b>{v}</b><br>"
            f"Type: {rel_type}<br>"
            f"<i>({u.split('/')[-1]} imports {v.split('/')[-1]})</i>"
        )

    # Simulated edge line trace
    if sim_edge_x:
        fig.add_trace(
            go.Scatter(
                x=sim_edge_x, y=sim_edge_y,
                line=dict(width=2.8, color="#39d353", dash="dash"),
                hoverinfo="text",
                hovertext="Simulated Hypothetical Edge",
                mode="lines",
                name="Simulated Dependency"
            )
        )

    # Invisible scatter trace for rich edge hover tooltips
    if hover_edge_x:
        fig.add_trace(
            go.Scatter(
                x=hover_edge_x, y=hover_edge_y,
                mode="markers",
                marker=dict(size=7, color="rgba(0,0,0,0)", symbol="circle"),
                hoverinfo="text",
                hovertext=hover_edge_text,
                showlegend=False
            )
        )

    # 4. Render Nodes
    node_x, node_y = [], []
    node_colors = []
    node_sizes = []
    node_borders = []
    node_border_widths = []
    node_display_labels = []
    node_hover_texts = []
    node_text_positions = []

    for node in sub_g.nodes():
        if node not in pos:
            continue
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)

        prof = (risk_profiles_map or {}).get(node, {})
        cat = prof.get("risk_category", "LOW")
        prob = prof.get("risk_percentage", 20)
        dep_c = len(list(sub_g.successors(node)))
        dept_c = len(list(sub_g.predecessors(node)))

        is_focused = (focus_component and node == focus_component)
        is_cyclic_node = any(node in cyc for cyc in all_cycles)

        # Node sizes and border highlights
        if is_focused:
            node_sizes.append(28)
            node_borders.append("#58a6ff")
            node_border_widths.append(3.5)
        elif is_cyclic_node:
            node_sizes.append(20)
            node_borders.append("#f85149")
            node_border_widths.append(2.5)
        elif highlight_risk and cat == "HIGH":
            node_sizes.append(22)
            node_borders.append("#ff7b72")
            node_border_widths.append(2.5)
        else:
            node_sizes.append(17)
            node_borders.append("#161b22")
            node_border_widths.append(1.8)

        node_colors.append(RISK_COLORS.get(cat, "#2ea043"))

        # Label formatting based on show_labels setting
        stem_name = Path(node).name
        pkg = _get_package_for_node(node)
        risk_tag = RISK_ICONS.get(cat, "[L]")

        if show_labels == "hover":
            node_display_labels.append("")
        elif show_labels == "truncated":
            node_display_labels.append(f"{risk_tag} {stem_name}")
        else:  # full
            node_display_labels.append(f"{risk_tag} [{pkg}] {stem_name}")

        # Intelligent text positioning to prevent edge collisions
        layer = node_layers.get(node, 0)
        if layout_type == "hierarchical" and focus_component:
            if layer < 0:
                node_text_positions.append("middle left")
            elif layer > 0:
                node_text_positions.append("middle right")
            else:
                node_text_positions.append("top center")
        else:
            node_text_positions.append("top center")

        # Rich Tooltip
        role_label = "Central Bridge / Hub" if (dep_c + dept_c) >= 3 else "Standard Component"
        cycle_note = "<br><span style='color: #f85149;'>⚠️ Part of Circular Dependency</span>" if is_cyclic_node else ""
        node_hover_texts.append(
            f"<b>{node}</b>{cycle_note}<br>"
            f"Package: <b>{pkg}/</b><br>"
            f"Predicted Risk: <b>{prob}% ({cat})</b><br>"
            f"Outgoing Dependencies (Imports): <b>{dep_c}</b><br>"
            f"Incoming Dependents (Imported by): <b>{dept_c}</b><br>"
            f"Role: <b>{role_label}</b>"
        )

    # Add Nodes Trace
    fig.add_trace(
        go.Scatter(
            x=node_x, y=node_y,
            mode="markers+text" if show_labels != "hover" else "markers",
            marker=dict(
                size=node_sizes,
                color=node_colors,
                line=dict(width=node_border_widths, color=node_borders),
                opacity=0.95
            ),
            text=node_display_labels,
            textposition=node_text_positions,
            textfont=dict(size=11, color="#f0f6fc", family="JetBrains Mono, monospace"),
            hoverinfo="text",
            hovertext=node_hover_texts,
            name="Components"
        )
    )

    # 5. Calculate bounds for X and Y with generous padding to prevent text clipping
    all_x = [pos[n][0] for n in sub_g.nodes() if n in pos]
    all_y = [pos[n][1] for n in sub_g.nodes() if n in pos]

    # Add Column Banners for Hierarchical View
    if layout_type == "hierarchical" and focus_component and sub_g.has_node(focus_component):
        # Determine min and max Y to place headers comfortably above
        max_y = max(all_y) if all_y else 0.0
        header_y = max_y + 1.25

        # Check if left column has nodes
        has_left = any(node_layers.get(n, 0) < 0 for n in sub_g.nodes())
        has_right = any(node_layers.get(n, 0) > 0 for n in sub_g.nodes())

        if has_left:
            fig.add_annotation(
                x=-2.4, y=header_y,
                text="<b>◀ UPSTREAM DEPENDENCIES</b><br><span style='font-size: 10px; color: #8b949e;'>Modules this component imports</span>",
                showarrow=False,
                font=dict(size=11, color="#58a6ff"),
                align="center"
            )

        fig.add_annotation(
            x=0.0, y=header_y,
            text="<b>★ SELECTED COMPONENT ★</b><br><span style='font-size: 10px; color: #8b949e;'>Focus Node</span>",
            showarrow=False,
            font=dict(size=11, color="#f0f6fc"),
            align="center"
        )

        if has_right:
            fig.add_annotation(
                x=2.4, y=header_y,
                text="<b>DOWNSTREAM DEPENDENTS ▶</b><br><span style='font-size: 10px; color: #8b949e;'>Modules that import this component</span>",
                showarrow=False,
                font=dict(size=11, color="#a371f7"),
                align="center"
            )

    # Calculate bounds for X and Y with generous padding to prevent text clipping
    if all_x:
        x_min, x_max = min(all_x), max(all_x)
        x_pad = max(1.5, (x_max - x_min) * 0.28)
        x_range = [x_min - x_pad, x_max + x_pad]
    else:
        x_range = [-3.5, 3.5]

    if all_y:
        y_min, y_max = min(all_y), max(all_y)
        y_pad = max(0.9, (y_max - y_min) * 0.2)
        extra_top = 1.4 if (layout_type == "hierarchical" and focus_component and sub_g.has_node(focus_component)) else 0.4
        y_range = [y_min - y_pad, y_max + y_pad + extra_top]
    else:
        y_range = [-2, 2]

    fig.update_layout(
        **DARK_LAYOUT,
        showlegend=False,
        height=500,
        xaxis=dict(range=x_range, showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(range=y_range, showgrid=False, zeroline=False, showticklabels=False),
    )

    return fig
