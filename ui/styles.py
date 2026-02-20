"""Global Streamlit CSS styles."""

from __future__ import annotations

from ui.theme import (
    ACCENT,
    NEGATIVE,
    NEON_BLUE,
    NEON_PURPLE,
    POSITIVE,
    PRIMARY_BG,
    TEXT_LIGHT,
    TEXT_MUTED,
    WARNING,
)


def app_css() -> str:
    return f"""
<style>
/* Global styles */
.stApp {{
    background-color: {PRIMARY_BG};
    color: {TEXT_LIGHT};
    font-family: 'Inter', 'Segoe UI', sans-serif;
}}

/* Custom scrollbar */
::-webkit-scrollbar {{ width: 6px; height: 6px; }}
::-webkit-scrollbar-track {{ background: {PRIMARY_BG}; }}
::-webkit-scrollbar-thumb {{ background: linear-gradient(180deg, {NEON_BLUE}, {NEON_PURPLE}); border-radius: 3px; }}
::-webkit-scrollbar-thumb:hover {{ background: linear-gradient(180deg, {POSITIVE}, {NEON_BLUE}); }}

/* Metric delta colors */
.metric-delta-positive {{
    color: {POSITIVE};
    font-weight: 600;
    font-size: 0.85rem;
    text-shadow: 0 0 10px rgba(0, 255, 136, 0.5);
}}
.metric-delta-negative {{
    color: {NEGATIVE};
    font-weight: 600;
    font-size: 0.85rem;
    text-shadow: 0 0 10px rgba(255, 51, 102, 0.5);
}}

/* Glassmorphism titles */
h1.title {{
    font-size: 2.8rem;
    font-weight: 800;
    background: linear-gradient(135deg, {ACCENT}, {NEON_BLUE}, {NEON_PURPLE});
    background-size: 200% 200%;
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    animation: titleGlow 3s ease infinite;
    margin-bottom: 0.4rem;
    letter-spacing: -0.5px;
}}

@keyframes titleGlow {{
    0% {{ background-position: 0% 50%; }}
    50% {{ background-position: 100% 50%; }}
    100% {{ background-position: 0% 50%; }}
}}

p.subtitle {{
    font-size: 1.05rem;
    color: {TEXT_MUTED};
    margin-top: 0;
    margin-bottom: 2rem;
}}

/* Glassmorphism metric cards */
.metric-card {{
    background: rgba(0, 0, 0, 0.85);
    backdrop-filter: blur(20px);
    -webkit-backdrop-filter: blur(20px);
    border: 1px solid rgba(0, 212, 255, 0.15);
    border-radius: 16px;
    padding: 24px 20px;
    text-align: center;
    box-shadow: 0 8px 32px rgba(0, 0, 0, 0.4), inset 0 1px 0 rgba(255, 255, 255, 0.05);
    margin-bottom: 20px;
    position: relative;
    overflow: hidden;
    transition: all 0.3s ease;
}}
.metric-card:hover {{
    border-color: rgba(0, 212, 255, 0.4);
    box-shadow: 0 8px 32px rgba(0, 212, 255, 0.15), inset 0 1px 0 rgba(255, 255, 255, 0.1);
    transform: translateY(-2px);
}}
.metric-card::before {{
    content: '';
    position: absolute;
    top: 0;
    left: -100%;
    width: 200%;
    height: 1px;
    background: linear-gradient(90deg, transparent, {NEON_BLUE}, transparent);
    animation: shimmer 3s ease infinite;
}}

@keyframes shimmer {{
    0% {{ left: -100%; }}
    100% {{ left: 100%; }}
}}

.metric-label {{
    font-size: 0.8rem;
    color: {TEXT_MUTED};
    margin-bottom: 8px;
    letter-spacing: 1.5px;
    text-transform: uppercase;
    font-weight: 500;
}}
.metric-value {{
    font-size: 1.9rem;
    font-weight: 700;
    color: {ACCENT};
    text-shadow: 0 0 20px rgba(255, 255, 255, 0.1);
}}

/* Glassmorphism panel boxes */
.panel-box {{
    background: rgba(0, 0, 0, 0.75);
    backdrop-filter: blur(16px);
    -webkit-backdrop-filter: blur(16px);
    border-radius: 18px;
    padding: 28px;
    margin-bottom: 32px;
    border: 1px solid rgba(0, 212, 255, 0.1);
    box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
    position: relative;
    overflow: hidden;
}}
.panel-box::after {{
    content: '';
    position: absolute;
    bottom: 0;
    left: 0;
    width: 100%;
    height: 1px;
    background: linear-gradient(90deg, transparent, rgba(0, 212, 255, 0.3), transparent);
}}

/* Neon glow signal cards */
.signal-long {{
    background: rgba(0, 255, 136, 0.08);
    border: 1px solid rgba(0, 255, 136, 0.3);
    border-radius: 12px;
    padding: 12px;
    text-align: center;
    box-shadow: 0 0 20px rgba(0, 255, 136, 0.1);
}}
.signal-short {{
    background: rgba(255, 51, 102, 0.08);
    border: 1px solid rgba(255, 51, 102, 0.3);
    border-radius: 12px;
    padding: 12px;
    text-align: center;
    box-shadow: 0 0 20px rgba(255, 51, 102, 0.1);
}}
.signal-wait {{
    background: rgba(255, 209, 102, 0.08);
    border: 1px solid rgba(255, 209, 102, 0.3);
    border-radius: 12px;
    padding: 12px;
    text-align: center;
    box-shadow: 0 0 20px rgba(255, 209, 102, 0.1);
}}

/* Enhanced table styling */
.table-container {{ overflow-x: auto; }}
table.dataframe {{
    width: 100% !important;
    border-collapse: collapse;
    background-color: rgba(0, 0, 0, 0.9);
    backdrop-filter: blur(10px);
}}
table.dataframe thead tr {{
    background: linear-gradient(90deg, rgba(0, 212, 255, 0.1), rgba(178, 75, 243, 0.1));
}}
table.dataframe th {{
    color: {NEON_BLUE};
    padding: 12px;
    text-align: left;
    font-size: 0.85rem;
    border-bottom: 1px solid rgba(0, 212, 255, 0.2);
    letter-spacing: 0.5px;
    text-transform: uppercase;
    font-weight: 600;
}}
table.dataframe td {{
    padding: 10px 12px;
    font-size: 0.9rem;
    border-bottom: 1px solid rgba(255, 255, 255, 0.04);
    color: {TEXT_LIGHT};
    transition: background 0.2s;
}}
table.dataframe tbody tr:hover {{
    background-color: rgba(0, 212, 255, 0.05);
}}

/* Streamlit tab styling */
.stTabs [data-baseweb="tab-list"] {{
    gap: 2px;
    background: rgba(0, 0, 0, 0.8);
    border-radius: 10px;
    padding: 3px;
    flex-wrap: nowrap;
    overflow-x: auto;
}}
.stTabs [data-baseweb="tab"] {{
    border-radius: 6px;
    color: {TEXT_MUTED};
    font-weight: 500;
    padding: 4px 8px;
    font-size: 0.75rem;
    transition: all 0.2s ease;
    white-space: nowrap;
}}
.stTabs [aria-selected="true"] {{
    background: linear-gradient(135deg, rgba(0, 212, 255, 0.15), rgba(178, 75, 243, 0.15)) !important;
    color: {ACCENT} !important;
    border-bottom-color: {NEON_BLUE} !important;
}}

/* Streamlit button styling */
.stButton > button {{
    background: linear-gradient(135deg, rgba(0, 212, 255, 0.2), rgba(178, 75, 243, 0.2));
    border: 1px solid rgba(0, 212, 255, 0.3);
    color: {ACCENT};
    border-radius: 10px;
    font-weight: 600;
    letter-spacing: 0.5px;
    transition: all 0.3s ease;
}}
.stButton > button:hover {{
    background: linear-gradient(135deg, rgba(0, 212, 255, 0.35), rgba(178, 75, 243, 0.35));
    border-color: {NEON_BLUE};
    box-shadow: 0 0 20px rgba(0, 212, 255, 0.2);
    transform: translateY(-1px);
}}
.stButton > button[kind="primary"] {{
    background: linear-gradient(135deg, {NEON_BLUE}, {NEON_PURPLE});
    border: none;
    color: white;
    font-weight: 700;
    text-shadow: 0 1px 2px rgba(0,0,0,0.3);
}}
.stButton > button[kind="primary"]:hover {{
    box-shadow: 0 0 30px rgba(0, 212, 255, 0.4);
}}

/* Sidebar styling */
section[data-testid="stSidebar"] {{
    background-color: {PRIMARY_BG};
    border-right: 1px solid rgba(255, 255, 255, 0.06);
}}

/* Input fields */
.stTextInput > div > div > input,
.stSelectbox > div > div,
.stMultiSelect > div > div {{
    background-color: rgba(0, 0, 0, 0.9) !important;
    border-color: rgba(0, 212, 255, 0.2) !important;
    color: {TEXT_LIGHT} !important;
    border-radius: 8px !important;
}}

/* Fix dark blue backgrounds on Streamlit widgets */
.stSlider > div,
.stCheckbox > label,
.stRadio > div,
.stNumberInput > div > div > input {{
    background-color: transparent !important;
}}
[data-testid="stMetric"],
[data-testid="stMetricValue"],
[data-testid="column"] {{
    background-color: transparent !important;
}}
div[data-testid="stVerticalBlock"] > div {{
    background-color: transparent !important;
}}
.stSelectbox label,
.stTextInput label,
.stMultiSelect label,
.stSlider label,
.stCheckbox label,
.stNumberInput label {{
    color: {TEXT_MUTED} !important;
}}

/* Expander styling */
.streamlit-expanderHeader {{
    background: rgba(0, 0, 0, 0.7);
    border-radius: 8px;
}}

/* Live ticker bar */
.ticker-bar {{
    background: linear-gradient(90deg, rgba(0, 212, 255, 0.05), rgba(178, 75, 243, 0.05), rgba(0, 212, 255, 0.05));
    border: 1px solid rgba(0, 212, 255, 0.1);
    border-radius: 10px;
    padding: 8px 16px;
    margin-bottom: 16px;
    overflow: hidden;
    position: relative;
}}
.ticker-content {{
    display: flex;
    animation: tickerScroll 30s linear infinite;
    white-space: nowrap;
    gap: 40px;
}}
@keyframes tickerScroll {{
    0% {{ transform: translateX(0); }}
    100% {{ transform: translateX(-50%); }}
}}
.ticker-item {{
    display: inline-flex;
    align-items: center;
    gap: 8px;
    font-size: 0.82rem;
    font-weight: 500;
    color: {TEXT_LIGHT};
}}

/* Heatmap cell */
.heatmap-cell {{
    border-radius: 6px;
    padding: 8px;
    text-align: center;
    transition: transform 0.2s;
    cursor: default;
}}
.heatmap-cell:hover {{
    transform: scale(1.05);
    z-index: 10;
}}

/* Section header */
.god-header {{
    background: linear-gradient(90deg, transparent, rgba(0, 212, 255, 0.1), transparent);
    border-left: 3px solid {NEON_BLUE};
    padding: 8px 16px;
    margin: 16px 0;
    border-radius: 0 8px 8px 0;
}}

/* Pulse animation for live data */
.pulse {{
    animation: pulse 2s ease infinite;
}}
@keyframes pulse {{
    0%, 100% {{ opacity: 1; }}
    50% {{ opacity: 0.6; }}
}}

/* Risk level indicators */
.risk-low {{ color: {POSITIVE}; text-shadow: 0 0 8px rgba(0, 255, 136, 0.4); }}
.risk-medium {{ color: {WARNING}; text-shadow: 0 0 8px rgba(255, 209, 102, 0.4); }}
.risk-high {{ color: {NEGATIVE}; text-shadow: 0 0 8px rgba(255, 51, 102, 0.4); }}
.risk-extreme {{ color: #FF0000; text-shadow: 0 0 12px rgba(255, 0, 0, 0.6); animation: pulse 1s ease infinite; }}

/* Fibonacci level bars */
.fib-level {{
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 6px 12px;
    margin: 2px 0;
    border-radius: 6px;
    font-size: 0.85rem;
    font-weight: 500;
    transition: all 0.2s;
}}
.fib-level:hover {{ transform: translateX(4px); }}

/* Monte Carlo confidence band */
.mc-stat {{
    text-align: center;
    padding: 12px;
    background: rgba(0, 0, 0, 0.7);
    border-radius: 10px;
    border: 1px solid rgba(255, 255, 255, 0.05);
}}

/* Whale tracker entry */
.whale-entry {{
    display: flex;
    align-items: center;
    gap: 12px;
    padding: 10px 14px;
    background: rgba(0, 0, 0, 0.7);
    border-radius: 10px;
    border-left: 3px solid {NEON_BLUE};
    margin: 6px 0;
    transition: all 0.2s;
}}
.whale-entry:hover {{
    background: rgba(0, 0, 0, 0.9);
    border-left-color: {NEON_PURPLE};
}}

/* Advanced screener row highlight */
.screener-match {{
    background: rgba(0, 255, 136, 0.05);
    border: 1px solid rgba(0, 255, 136, 0.2);
    border-radius: 8px;
    padding: 8px 12px;
    margin: 4px 0;
}}

/* Tooltip question mark */
.tt {{
    display: inline-flex;
    align-items: center;
    justify-content: center;
    width: 16px; height: 16px;
    border-radius: 50%;
    background: rgba(0, 212, 255, 0.15);
    color: {NEON_BLUE};
    font-size: 0.65rem;
    font-weight: 700;
    cursor: help;
    position: relative;
    vertical-align: middle;
    margin-left: 4px;
    border: 1px solid rgba(0, 212, 255, 0.3);
}}
.tt .ttt {{
    visibility: hidden;
    opacity: 0;
    position: absolute;
    bottom: 125%;
    left: 50%;
    transform: translateX(-50%);
    background: #000000;
    color: {TEXT_LIGHT};
    padding: 8px 12px;
    border-radius: 8px;
    font-size: 0.78rem;
    font-weight: 400;
    line-height: 1.4;
    width: max-content;
    max-width: 280px;
    white-space: normal;
    z-index: 999;
    border: 1px solid rgba(0, 212, 255, 0.2);
    box-shadow: 0 4px 16px rgba(0,0,0,0.4);
    transition: opacity 0.2s;
    pointer-events: none;
}}
.tt:hover .ttt {{
    visibility: visible;
    opacity: 1;
}}

/* Mobile-first polish */
@media (max-width: 900px) {{
    h1.title {{
        font-size: 2rem;
        line-height: 1.15;
    }}
    .panel-box {{
        padding: 16px;
        margin-bottom: 18px;
        border-radius: 14px;
    }}
    .metric-card {{
        padding: 14px 12px;
        border-radius: 12px;
        margin-bottom: 10px;
    }}
    .metric-label {{
        font-size: 0.68rem;
        letter-spacing: 1px;
    }}
    .metric-value {{
        font-size: 1.25rem;
    }}
    .stTabs [data-baseweb="tab"] {{
        font-size: 0.72rem;
        padding: 4px 7px;
    }}
}}

@media (max-width: 640px) {{
    .stDataFrame, div[data-testid="stDataFrame"] {{
        font-size: 0.78rem !important;
    }}
    .stButton > button {{
        font-size: 0.82rem;
    }}
}}
</style>
"""
