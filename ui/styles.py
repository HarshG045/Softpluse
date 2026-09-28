"""
UI Styles: Professional Dark Theme Design System for SoftwarePulse Developer Tool.
Inspired by GitHub, Linear, Vercel, and Raycast developer tools.
"""

def get_custom_css() -> str:
    """Returns custom CSS for developer-tool aesthetics."""
    return """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500;600&display=swap');

    /* Global Base */
    html, body, [class*="css"], .stApp {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
        background-color: #0d1117 !important;
        color: #c9d1d9;
    }

    /* Monospace strictly for code/paths/hashes */
    code, pre, .mono-font, .stCode {
        font-family: 'JetBrains Mono', ui-monospace, monospace !important;
    }

    /* Streamlit Default Header Adjustment (Prevent Content Clipping) */
    header[data-testid="stHeader"] {
        background-color: transparent !important;
        height: 2.5rem !important;
        z-index: 10 !important;
    }

    [data-testid="stDecoration"] {
        display: none !important;
    }

    /* Main Container Padding - Generous top padding to prevent header cutoff */
    .block-container {
        padding-top: 4.5rem !important;
        padding-bottom: 3rem !important;
        padding-left: 2.2rem !important;
        padding-right: 2.2rem !important;
        max-width: 1440px;
    }

    /* Sidebar Base */
    section[data-testid="stSidebar"] {
        background-color: #090d13 !important;
        border-right: 1px solid #30363d !important;
    }

    section[data-testid="stSidebar"] > div {
        padding-top: 3.2rem !important;
        padding-left: 0.8rem !important;
        padding-right: 0.8rem !important;
    }


    /* Sidebar Brand Header */
    .sidebar-brand-container {
        display: flex;
        align-items: center;
        justify-content: space-between;
        padding: 6px 8px 14px 8px;
        border-bottom: 1px solid #21262d;
        margin-bottom: 14px;
    }

    .sidebar-brand-title {
        font-size: 15px;
        font-weight: 700;
        color: #f0f6fc;
        letter-spacing: -0.2px;
        line-height: 1.2;
    }

    .sidebar-brand-subtitle {
        font-size: 11px;
        color: #8b949e;
        margin-top: 2px;
        font-weight: 400;
    }

    /* Sidebar Navigation Section Headers */
    .nav-section-header {
        font-size: 11px;
        font-weight: 600;
        color: #6e7681;
        text-transform: uppercase;
        letter-spacing: 0.6px;
        padding: 10px 10px 4px 10px;
        margin-top: 4px;
    }

    /* Sidebar Nav Buttons Styling */
    div[data-testid="stSidebar"] div.stButton > button {
        width: 100% !important;
        display: flex !important;
        justify-content: flex-start !important;
        align-items: center !important;
        text-align: left !important;
        background-color: transparent !important;
        color: #8b949e !important;
        border: 1px solid transparent !important;
        border-radius: 6px !important;
        padding: 7px 12px !important;
        font-size: 13px !important;
        font-weight: 500 !important;
        margin-bottom: 2px !important;
        transition: all 0.15s ease-in-out !important;
    }

    div[data-testid="stSidebar"] div.stButton > button:hover {
        background-color: #161b22 !important;
        border-color: #30363d !important;
        color: #f0f6fc !important;
    }

    /* Active Nav Button */
    div[data-testid="stSidebar"] div.stButton > button.nav-active-btn {
        background-color: rgba(56, 139, 253, 0.12) !important;
        border-left: 3px solid #58a6ff !important;
        border-top: 1px solid rgba(56, 139, 253, 0.2) !important;
        border-right: 1px solid rgba(56, 139, 253, 0.2) !important;
        border-bottom: 1px solid rgba(56, 139, 253, 0.2) !important;
        color: #58a6ff !important;
        font-weight: 600 !important;
    }

    /* Sidebar Workspace / Repository Card */
    .sidebar-workspace-card {
        background-color: #161b22;
        border: 1px solid #30363d;
        border-radius: 6px;
        padding: 10px 12px;
        margin: 8px 0 12px 0;
    }

    .workspace-repo-title {
        font-size: 13px;
        font-weight: 600;
        color: #f0f6fc;
        display: flex;
        align-items: center;
        gap: 6px;
    }

    .workspace-branch-badge {
        display: inline-block;
        font-size: 11px;
        color: #8b949e;
        font-family: 'JetBrains Mono', monospace;
        margin-top: 2px;
    }

    .workspace-stats-row {
        font-size: 11px;
        color: #8b949e;
        margin-top: 6px;
        padding-top: 6px;
        border-top: 1px solid #21262d;
    }

    /* Sidebar Model Card */
    .sidebar-model-card {
        background-color: #161b22;
        border: 1px solid #30363d;
        border-radius: 6px;
        padding: 10px 12px;
        margin: 8px 0 12px 0;
    }

    .sidebar-model-label {
        font-size: 10px;
        font-weight: 600;
        color: #8b949e;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }

    .sidebar-model-value {
        font-size: 13px;
        font-weight: 600;
        color: #f0f6fc;
        margin-top: 2px;
    }

    .sidebar-model-desc {
        font-size: 11px;
        color: #8b949e;
        margin-top: 2px;
    }

    /* Top Context / Status Header Bar */
    .pulse-context-bar {
        display: flex;
        align-items: center;
        justify-content: space-between;
        padding: 10px 16px;
        background-color: #161b22;
        border: 1px solid #30363d;
        border-radius: 6px;
        margin-bottom: 20px;
    }

    .context-repo-info {
        display: flex;
        align-items: center;
        gap: 10px;
        font-size: 14px;
        font-weight: 600;
        color: #f0f6fc;
    }

    .context-status-pill {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        font-size: 12px;
        padding: 2px 8px;
        border-radius: 12px;
        font-weight: 500;
    }

    .pill-analyzed {
        background-color: rgba(46, 160, 67, 0.15);
        color: #3fb950;
        border: 1px solid rgba(46, 160, 67, 0.4);
    }

    .pill-idle {
        background-color: rgba(139, 148, 158, 0.15);
        color: #8b949e;
        border: 1px solid rgba(139, 148, 158, 0.4);
    }

    .context-meta-info {
        display: flex;
        align-items: center;
        gap: 16px;
        font-size: 12px;
        color: #8b949e;
    }

    /* Compact Metric Cards */
    .metric-card-compact {
        background-color: #161b22;
        border: 1px solid #30363d;
        border-radius: 6px;
        padding: 14px 16px;
        transition: border-color 0.2s ease, transform 0.15s ease;
    }

    .metric-card-compact:hover {
        border-color: #58a6ff;
    }

    .metric-card-label {
        font-size: 11px;
        color: #8b949e;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        font-weight: 600;
        margin-bottom: 4px;
    }

    .metric-card-value {
        font-size: 26px;
        font-weight: 700;
        color: #f0f6fc;
        line-height: 1.1;
    }

    .metric-card-sub {
        font-size: 12px;
        color: #8b949e;
        margin-top: 4px;
    }

    /* Risk Badges */
    .risk-badge {
        display: inline-block;
        padding: 2px 7px;
        border-radius: 4px;
        font-weight: 600;
        font-size: 11px;
        text-transform: uppercase;
        letter-spacing: 0.4px;
    }

    .risk-badge-high {
        background-color: rgba(248, 81, 73, 0.18);
        color: #ff7b72;
        border: 1px solid rgba(248, 81, 73, 0.4);
    }

    .risk-badge-medium {
        background-color: rgba(210, 153, 34, 0.18);
        color: #d29922;
        border: 1px solid rgba(210, 153, 34, 0.4);
    }

    .risk-badge-low {
        background-color: rgba(46, 160, 67, 0.18);
        color: #3fb950;
        border: 1px solid rgba(46, 160, 67, 0.4);
    }

    /* What-If Split Comparison Panel */
    .what-if-box {
        background-color: #161b22;
        border: 1px solid #30363d;
        border-radius: 6px;
        padding: 16px;
    }

    .what-if-delta-val {
        font-size: 30px;
        font-weight: 700;
        margin: 6px 0;
    }

    .delta-risk-up {
        color: #ff7b72;
    }

    .delta-risk-down {
        color: #3fb950;
    }

    .delta-risk-zero {
        color: #8b949e;
    }

    /* Non-intrusive Methodology Note */
    .methodology-note {
        padding: 8px 12px;
        background-color: rgba(56, 139, 253, 0.08);
        border-left: 3px solid #58a6ff;
        border-radius: 0 4px 4px 0;
        font-size: 12px;
        color: #8b949e;
        margin: 10px 0;
    }

    /* Dataframe / Table styling */
    .stDataFrame {
        border: 1px solid #30363d !important;
        border-radius: 6px !important;
        background-color: #161b22 !important;
    }

    /* Buttons */
    .stButton > button {
        background-color: #21262d !important;
        color: #c9d1d9 !important;
        border: 1px solid #30363d !important;
        border-radius: 6px !important;
        font-weight: 500 !important;
        font-size: 13px !important;
        transition: all 0.15s ease !important;
    }

    .stButton > button:hover {
        background-color: #30363d !important;
        border-color: #8b949e !important;
        color: #f0f6fc !important;
    }

    .stButton > button[kind="primary"] {
        background-color: #238636 !important;
        color: #ffffff !important;
        border-color: rgba(240, 246, 252, 0.1) !important;
        font-weight: 600 !important;
    }

    .stButton > button[kind="primary"]:hover {
        background-color: #2ea043 !important;
        border-color: #3fb950 !important;
    }

    /* Subtle Dividers */
    hr {
        border: none !important;
        border-top: 1px solid #21262d !important;
        margin: 18px 0 !important;
    }
    </style>
    """
