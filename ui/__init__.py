"""
SoftwarePulse UI Package.
Provides dark-theme styles, navigation sidebar, header, shared components, and dedicated dashboard pages.
"""
from .styles import get_custom_css
from .sidebar import render_sidebar, NavigationPage
from .header import render_header
from .components import render_metric_card, render_risk_pill, render_scientific_disclaimer
from .pages import (
    render_overview_page,
    render_risk_explorer_page,
    render_architecture_page,
    render_evolution_page,
    render_dependencies_page,
    render_explainability_page,
    render_what_if_lab_page,
    render_model_evaluation_page,
    render_settings_page
)

__all__ = [
    "get_custom_css",
    "render_sidebar",
    "NavigationPage",
    "render_header",
    "render_metric_card",
    "render_risk_pill",
    "render_scientific_disclaimer",
    "render_overview_page",
    "render_risk_explorer_page",
    "render_architecture_page",
    "render_evolution_page",
    "render_dependencies_page",
    "render_explainability_page",
    "render_what_if_lab_page",
    "render_model_evaluation_page",
    "render_settings_page"
]

