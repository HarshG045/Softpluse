"""
SoftwarePulse Visualization Package.
Generates Plotly-based risk charts, evolution timelines, interactive dependency graphs, and component signal plots.
"""
from .risk_charts import create_risk_distribution_chart, create_top_risky_components_chart, create_risk_gauge_chart
from .evolution_charts import create_component_evolution_chart, create_snapshot_metric_timeline
from .dependency_graph import create_interactive_dependency_figure
from .component_view import create_feature_comparison_chart, create_signal_bar_chart

__all__ = [
    "create_risk_distribution_chart",
    "create_top_risky_components_chart",
    "create_risk_gauge_chart",
    "create_component_evolution_chart",
    "create_snapshot_metric_timeline",
    "create_interactive_dependency_figure",
    "create_feature_comparison_chart",
    "create_signal_bar_chart"
]
