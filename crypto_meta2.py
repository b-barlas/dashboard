import datetime
import time
import math
import json
import hashlib
import streamlit as st
import pandas as pd
import numpy as np
import requests
import plotly.graph_objs as go
import plotly.express as px
from plotly.subplots import make_subplots
import ccxt
import ta
from typing import Tuple, List, Dict, Optional
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor, as_completed
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


def _wma(series: pd.Series, length: int) -> pd.Series:
    """Weighted Moving Average – gives more weight to recent prices."""
    weights = np.arange(1, length + 1, dtype=float)
    return series.rolling(window=length).apply(
        lambda x: np.dot(x, weights) / weights.sum(), raw=True
    )


def _supertrend(high: pd.Series, low: pd.Series, close: pd.Series,
                length: int = 10, multiplier: float = 3.0) -> pd.DataFrame:
    """SuperTrend indicator using ATR bands."""
    atr = ta.volatility.average_true_range(high, low, close, window=length)
    hl2 = (high + low) / 2
    upper_band = hl2 + multiplier * atr
    lower_band = hl2 - multiplier * atr

    supertrend = pd.Series(np.nan, index=close.index)
    direction = pd.Series(1, index=close.index)

    for i in range(length, len(close)):
        if close.iloc[i] > upper_band.iloc[i - 1]:
            direction.iloc[i] = 1
        elif close.iloc[i] < lower_band.iloc[i - 1]:
            direction.iloc[i] = -1
        else:
            direction.iloc[i] = direction.iloc[i - 1]
            if direction.iloc[i] == 1 and lower_band.iloc[i] < lower_band.iloc[i - 1]:
                lower_band.iloc[i] = lower_band.iloc[i - 1]
            if direction.iloc[i] == -1 and upper_band.iloc[i] > upper_band.iloc[i - 1]:
                upper_band.iloc[i] = upper_band.iloc[i - 1]

        supertrend.iloc[i] = lower_band.iloc[i] if direction.iloc[i] == 1 else upper_band.iloc[i]

    result = pd.DataFrame({
        f'SUPERT_{length}_{multiplier}': supertrend,
        f'SUPERTd_{length}_{multiplier}': direction,
    }, index=close.index)
    return result


# Set up page title, icon and wide layout
st.set_page_config(
    page_title="Crypto Command Center",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# === Debug / diagnostics ===
if 'debug_mode' not in st.session_state:
    st.session_state['debug_mode'] = False
with st.sidebar:
    with st.expander("Developer Tools", expanded=False):
        st.session_state['debug_mode'] = st.toggle('Debug mode', value=st.session_state['debug_mode'])

def _debug(msg: str) -> None:
    """Emit a debug message only when Debug mode is enabled."""
    if st.session_state.get('debug_mode', False):
        st.sidebar.write(msg)



PRIMARY_BG = "#0D1117"        # overall app background – pure dark
CARD_BG    = "#161B22"        # cards and panel backgrounds – dark grey
ACCENT     = "#FFFFFF"        # white color
POSITIVE   = "#00FF88"        # neon green for positive change
NEGATIVE   = "#FF3366"        # neon red for negative change
WARNING    = "#FFD166"        # yellow for neutral or caution
TEXT_LIGHT = "#E5E7EB"        # light text color
TEXT_MUTED = "#8CA1B6"        # muted grey for secondary text
NEON_BLUE  = "#00D4FF"        # neon blue for accents
NEON_PURPLE = "#B24BF3"       # neon purple for gradients
GOLD       = "#FFD700"        # gold for premium features

st.markdown(f"""
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
    background: rgba(15, 22, 41, 0.7);
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
    background: rgba(15, 22, 41, 0.6);
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
    background-color: rgba(10, 14, 26, 0.8);
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
    background: rgba(15, 22, 41, 0.5);
    border-radius: 10px;
    padding: 3px;
    flex-wrap: wrap;
}}
.stTabs [data-baseweb="tab"] {{
    border-radius: 6px;
    color: {TEXT_MUTED};
    font-weight: 500;
    padding: 6px 10px;
    font-size: 0.82rem;
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
    background-color: rgba(15, 22, 41, 0.8) !important;
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
    background: rgba(15, 22, 41, 0.5);
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
    background: rgba(15, 22, 41, 0.5);
    border-radius: 10px;
    border: 1px solid rgba(255, 255, 255, 0.05);
}}

/* Whale tracker entry */
.whale-entry {{
    display: flex;
    align-items: center;
    gap: 12px;
    padding: 10px 14px;
    background: rgba(15, 22, 41, 0.5);
    border-radius: 10px;
    border-left: 3px solid {NEON_BLUE};
    margin: 6px 0;
    transition: all 0.2s;
}}
.whale-entry:hover {{
    background: rgba(15, 22, 41, 0.8);
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
    background: #1C2333;
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
</style>
""", unsafe_allow_html=True)


def _tip(label: str, tooltip: str) -> str:
    """Return HTML for a label with a hover tooltip question mark."""
    return (f"{label}<span class='tt'>?<span class='ttt'>{tooltip}</span></span>")


# Exchange set up with caching – Kraken (primary) with Coinbase & Bitstamp fallbacks
# All three exchanges are FCA‑regulated and legally available in the UK.
_EXCHANGE_CONFIGS = [
    ("kraken", {}),
    ("coinbase", {}),
    ("bitstamp", {}),
]

@st.cache_resource(show_spinner=False)
def get_exchange():
    for name, extra in _EXCHANGE_CONFIGS:
        try:
            ex = getattr(ccxt, name)({"enableRateLimit": True, **extra})
            ex.load_markets()  # verify connectivity
            return ex
        except Exception as e:
            print(f"Exchange {name} unavailable: {e}")
    # last resort – return Kraken instance even if offline so the rest of the
    # code doesn't crash; individual API calls will still fail gracefully.
    return ccxt.kraken({"enableRateLimit": True})

EXCHANGE = get_exchange()

# Fetch BTC and ETH prices in USD from CoinGecko
@st.cache_data(ttl=120, show_spinner=False)
def get_btc_eth_prices():
    try:
        url = "https://api.coingecko.com/api/v3/simple/price"
        params = {"ids": "bitcoin,ethereum", "vs_currencies": "usd"}
        response = requests.get(url, params=params, timeout=10).json()
        btc = response.get("bitcoin", {}).get("usd")
        eth = response.get("ethereum", {}).get("usd")
        return (btc if btc else None), (eth if eth else None)
    except Exception as e:
        _debug(f"get_btc_eth_prices error: {e}")
        return None, None

# Fetch market dominance and total/alt market cap from CoinGecko
@st.cache_data(ttl=300, show_spinner=False)
def get_market_indices():
    """Fetch global market indices and dominance values for major assets.

    Returns BTC and ETH dominance percentages, total and alt market cap values,
    the 24h percentage change in total market cap, and dominance values for
    several leading altcoins (BNB, SOL, ADA, XRP).  All dominance values are
    returned as integers representing percentage points (e.g. 42 for 42%).
    
    If the API call fails, zeros are returned for all fields.
    """
    try:
        data = requests.get("https://api.coingecko.com/api/v3/global", timeout=10).json().get("data", {})
        mcap_pct = data.get("market_cap_percentage", {}) or {}
        # Fetch dominance values for BTC, ETH and major altcoins.  The Coingecko
        # API returns keys in lower‑case symbol format.  If a coin is absent,
        # default to 0.0 dominance.  Dominance values are percentages (0–100).
        btc_dom = float(mcap_pct.get("btc", 0.0))
        eth_dom = float(mcap_pct.get("eth", 0.0))
        bnb_dom = float(mcap_pct.get("bnb", 0.0))
        # Coingecko uses 'sol' for Solana dominance and 'ada' for Cardano.
        sol_dom = float(mcap_pct.get("sol", 0.0))
        ada_dom = float(mcap_pct.get("ada", 0.0))
        xrp_dom = float(mcap_pct.get("xrp", 0.0))
        total_mcap = float(data.get("total_market_cap", {}).get("usd", 0.0))
        # Alt market cap excludes BTC only; this value is retained for display but
        # not used in the AI market outlook calculation.
        alt_mcap = total_mcap * (1 - btc_dom / 100.0)
        mcap_24h_pct = float(data.get("market_cap_change_percentage_24h_usd", 0.0))
        # Return all dominance values rounded to the nearest integer.  The altcoin
        # dominance values enable weighting in the AI market outlook calculation.
        return (
            int(round(btc_dom, 0)),
            int(round(eth_dom, 0)),
            int(total_mcap),
            int(alt_mcap),
            mcap_24h_pct,
            int(round(bnb_dom, 0)),
            int(round(sol_dom, 0)),
            int(round(ada_dom, 0)),
            int(round(xrp_dom, 0)),
        )
    except Exception as e:
        # Log the error and return zeros for all values to avoid breaking the
        # dashboard.  Using ints ensures consistent return types across the
        # success and failure paths.
        print(f"get_market_indices error: {e}")
        return 0, 0, 0, 0, 0.0, 0, 0, 0, 0


# Fetch fear and greed index from alternative.me
def get_fear_greed():
    try:
        data = requests.get("https://api.alternative.me/fng/?limit=1", timeout=10).json()
        value = int(data.get("data", [{}])[0].get("value", 0))
        label = data.get("data", [{}])[0].get("value_classification", "Unknown")
        return value, label
    except Exception as e:
        _debug(f"get_fear_greed error: {e}")
        return None, "Unavailable"

def get_social_sentiment(symbol: str) -> tuple[int, str]:
    """Return a naive sentiment score (0–100) and label based on 24h price change.

    The score is centred at 50 with each percentage point of change shifting
    the score by one point.  For example, a +5% move yields a score of 55,
    while a −10% move yields 40.  The score is clipped between 0 and 100.
    """
    try:
        change = get_price_change(symbol) or 0.0
    except Exception:
        change = 0.0
    # Map change to a 0–100 scale around 50
    score = int(max(0, min(100, 50 + change)))
    # Determine sentiment category
    if score >= 75:
        label = "Strongly Bullish"
    elif score >= 55:
        label = "Bullish"
    elif score >= 45:
        label = "Neutral"
    elif score >= 25:
        label = "Bearish"
    else:
        label = "Strongly Bearish"
    return score, label

@st.cache_resource(show_spinner=False)
def get_markets() -> dict:
    try:
        # Markets may already be loaded by get_exchange(); reload to be safe.
        return EXCHANGE.load_markets()
    except Exception as e:
        st.warning(f"Failed to load markets ({EXCHANGE.id}): {e}")
        return {}

MARKETS = get_markets()


def _symbol_variants(symbol: str) -> list[str]:
    """Return symbol variants to try on the exchange (e.g. USDT→USD)."""
    variants = [symbol]
    if "/" in symbol:
        base, quote = symbol.split("/", 1)
        if quote == "USDT":
            variants.append(f"{base}/USD")
        elif quote == "USD":
            variants.append(f"{base}/USDT")
    return variants


def _normalize_coin_input(raw: str) -> str:
    """Normalize coin input: 'BTC' → 'BTC/USDT', 'btc' → 'BTC/USDT', keeps 'BTC/USD' as-is."""
    raw = raw.strip().upper()
    if not raw:
        return raw
    # Strip common user mistakes (dollar signs, commas, extra spaces)
    raw = raw.replace("$", "").replace(",", "").strip()
    if "/" not in raw:
        return f"{raw}/USDT"
    return raw


def _validate_coin_symbol(symbol: str) -> str | None:
    """Return an error message if the symbol looks invalid, else None."""
    if not symbol:
        return "Please enter a coin symbol."
    base = symbol.split("/")[0] if "/" in symbol else symbol
    if len(base) < 2 or len(base) > 10:
        return f"'{base}' doesn't look like a valid coin ticker (2-10 characters expected)."
    if not base.isalpha():
        return f"'{base}' contains invalid characters. Use letters only (e.g. BTC, ETH)."
    return None


def _sr_lookback(timeframe: str | None = None) -> int:
    """Return support/resistance lookback bars adapted to timeframe.

    Lower timeframes have more noise so we look at more bars.
    """
    mapping = {"1m": 60, "3m": 50, "5m": 50, "15m": 40, "1h": 30, "4h": 20, "1d": 20}
    return mapping.get(timeframe or "", 30)


@dataclass
class AnalysisResult:
    """Structured result from analyse() — replaces fragile 16-element tuple."""
    signal: str = "NO DATA"
    leverage: int = 1
    comment: str = ""
    volume_spike: bool = False
    atr_comment: str = ""
    candle_pattern: str = ""
    confidence: float = 0.0
    adx: float = 0.0
    supertrend: str = ""
    ichimoku: str = ""
    stochrsi_k: float = 0.0
    bollinger: str = ""
    vwap: str = ""
    psar: str = ""
    williams: str = ""
    cci: str = ""


def _build_indicator_grid(supertrend_trend: str, ichimoku_trend: str, vwap_label: str,
                          adx_val: float, bollinger_bias: str, stochrsi_k_val: float,
                          psar_trend: str, williams_label: str, cci_label: str,
                          volume_spike: bool, atr_comment: str, candle_pattern: str) -> str:
    """Build the professional indicator grid HTML used in Spot, Position, and AI tabs."""
    indicators = []
    if supertrend_trend:
        indicators.append(("SuperTrend", format_trend(supertrend_trend), _ind_color(supertrend_trend)))
    if ichimoku_trend:
        indicators.append(("Ichimoku", format_trend(ichimoku_trend), _ind_color(ichimoku_trend)))
    if vwap_label:
        indicators.append(("VWAP", vwap_label, _ind_color(vwap_label)))
    if not np.isnan(adx_val):
        indicators.append(("ADX", format_adx(adx_val), WARNING))
    if bollinger_bias:
        indicators.append(("Bollinger", bollinger_bias, _ind_color(bollinger_bias)))
    if not np.isnan(stochrsi_k_val):
        srsi_c = POSITIVE if stochrsi_k_val < 0.2 else (NEGATIVE if stochrsi_k_val > 0.8 else WARNING)
        indicators.append(("StochRSI", f"{stochrsi_k_val:.2f}", srsi_c))
    if "Bullish" in psar_trend or "Bearish" in psar_trend:
        indicators.append(("PSAR", psar_trend, _ind_color(psar_trend)))
    if williams_label:
        indicators.append(("Williams %R", williams_label.replace("🟢 ", "").replace("🔴 ", "").replace("🟡 ", ""), _ind_color(williams_label)))
    if cci_label:
        indicators.append(("CCI", cci_label.replace("🟢 ", "").replace("🔴 ", "").replace("🟡 ", ""), _ind_color(cci_label)))
    if volume_spike:
        indicators.append(("Volume", "Spike ▲", POSITIVE))
    atr_clean = atr_comment.replace("▲", "").replace("▼", "").replace("–", "").strip()
    if atr_clean:
        indicators.append(("Volatility", atr_clean, _ind_color(atr_clean)))
    if candle_pattern:
        indicators.append(("Pattern", candle_pattern.split(" (")[0], WARNING))
    if not indicators:
        return ""
    grid_items = "".join(
        f"<div style='text-align:center; padding:6px;'>"
        f"<div style='color:{TEXT_MUTED}; font-size:0.7rem; text-transform:uppercase;'>{name}</div>"
        f"<div style='color:{color}; font-size:0.85rem; font-weight:600;'>{val}</div>"
        f"</div>"
        for name, val, color in indicators
    )
    return (
        f"<div style='display:grid; grid-template-columns:repeat(auto-fill, minmax(90px, 1fr)); "
        f"gap:4px; background:{CARD_BG}; border-radius:8px; padding:10px; margin:8px 0;'>"
        f"{grid_items}</div>"
    )


def _calc_conviction(signal_dir: str, ai_dir: str, confidence: float) -> tuple[str, str]:
    """Return (label, color) for conviction based on signal/AI alignment and confidence.

    signal_dir: 'LONG', 'SHORT', 'WAIT'
    ai_dir:     'LONG', 'SHORT', 'NEUTRAL'
    """
    if signal_dir in ("LONG", "SHORT") and signal_dir == ai_dir:
        if confidence >= 75:
            return "HIGH", POSITIVE
        elif confidence >= 60:
            return "MEDIUM", WARNING
        return "LOW", TEXT_MUTED
    if signal_dir in ("LONG", "SHORT") and ai_dir not in ("NEUTRAL", "WAIT", "") and signal_dir != ai_dir:
        return "CONFLICT", NEGATIVE
    return "LOW", TEXT_MUTED


# Fetch price change percentage for a given symbol via ccxt
@st.cache_data(ttl=60, show_spinner=False)
def get_price_change(symbol: str) -> float | None:
    for variant in _symbol_variants(symbol):
        try:
            ticker = EXCHANGE.fetch_ticker(variant)
            percent = ticker.get('percentage')
            return round(percent, 2) if percent is not None else None
        except Exception as e:
            _debug(f"get_price_change failed: {variant} → {e}")
    return None

# Fetch OHLCV data for a symbol and timeframe
@st.cache_data(ttl=60, show_spinner=False)
def fetch_ohlcv_cached(symbol: str, timeframe: str, limit: int = 120) -> pd.DataFrame:
    """Fetch OHLCV data via ccxt and return a DataFrame. Raises on error."""
    data = EXCHANGE.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
    df = pd.DataFrame(data, columns=["timestamp", "open", "high", "low", "close", "volume"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
    return df

# === CoinGecko fallback for coins not listed on the primary exchange ===
_TF_TO_CG_DAYS = {
    "1m": 2, "3m": 2, "5m": 2, "15m": 7,
    "1h": 90, "4h": 180, "1d": 365,
}


@st.cache_data(ttl=3600, show_spinner=False)
def _coingecko_coin_id(symbol: str) -> str | None:
    """Resolve a ticker symbol (e.g. 'WLFI') to a CoinGecko coin ID."""
    try:
        resp = requests.get(
            "https://api.coingecko.com/api/v3/search",
            params={"query": symbol},
            timeout=10,
        )
        if resp.status_code == 200:
            for coin in resp.json().get("coins", []):
                if coin.get("symbol", "").upper() == symbol.upper():
                    return coin["id"]
    except Exception as e:
        _debug(f"CoinGecko coin ID lookup failed for '{symbol}': {e}")
    return None


@st.cache_data(ttl=120, show_spinner=False)
def _coingecko_market_chart(coin_id: str, days: int) -> pd.DataFrame | None:
    """Fetch OHLC data from CoinGecko /ohlc endpoint, with volume from /market_chart."""
    try:
        # 1) Real OHLC candles from the dedicated OHLC endpoint
        ohlc_resp = requests.get(
            f"https://api.coingecko.com/api/v3/coins/{coin_id}/ohlc",
            params={"vs_currency": "usd", "days": days},
            timeout=15,
        )
        if ohlc_resp.status_code == 200:
            ohlc_data = ohlc_resp.json()
            if ohlc_data and isinstance(ohlc_data, list) and len(ohlc_data) > 5:
                df = pd.DataFrame(ohlc_data, columns=["timestamp", "open", "high", "low", "close"])
                # 2) Get volume from market_chart
                try:
                    vol_resp = requests.get(
                        f"https://api.coingecko.com/api/v3/coins/{coin_id}/market_chart",
                        params={"vs_currency": "usd", "days": days},
                        timeout=15,
                    )
                    if vol_resp.status_code == 200:
                        volumes = vol_resp.json().get("total_volumes", [])
                        if volumes:
                            df_v = pd.DataFrame(volumes, columns=["ts_v", "volume"])
                            # Merge volume by nearest timestamp
                            df["ts_ms"] = df["timestamp"]
                            df_v["ts_ms"] = df_v["ts_v"]
                            df = pd.merge_asof(
                                df.sort_values("ts_ms"),
                                df_v[["ts_ms", "volume"]].sort_values("ts_ms"),
                                on="ts_ms", direction="nearest",
                            )
                            df.drop(columns=["ts_ms"], inplace=True)
                        else:
                            df["volume"] = 0
                    else:
                        df["volume"] = 0
                except Exception:
                    df["volume"] = 0
                df["volume"] = df["volume"].fillna(0)
                df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
                df = df[["timestamp", "open", "high", "low", "close", "volume"]]
                return df
        # 3) Fallback: market_chart only (price-only OHLC)
        resp = requests.get(
            f"https://api.coingecko.com/api/v3/coins/{coin_id}/market_chart",
            params={"vs_currency": "usd", "days": days},
            timeout=15,
        )
        if resp.status_code != 200:
            return None
        data = resp.json()
        prices = data.get("prices", [])
        volumes = data.get("total_volumes", [])
        if not prices:
            return None
        df_p = pd.DataFrame(prices, columns=["timestamp", "close"])
        df_v = pd.DataFrame(volumes, columns=["timestamp", "volume"])
        df = df_p.merge(df_v, on="timestamp", how="left")
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
        df["volume"] = df["volume"].fillna(0)
        df["open"] = df["close"]
        df["high"] = df["close"]
        df["low"] = df["close"]
        df = df[["timestamp", "open", "high", "low", "close", "volume"]]
        return df
    except Exception:
        return None


def _fetch_coingecko_ohlcv(symbol: str, timeframe: str, limit: int) -> pd.DataFrame | None:
    """CoinGecko fallback: resolve symbol and fetch market chart data."""
    base = symbol.split("/")[0].strip()
    coin_id = _coingecko_coin_id(base)
    if not coin_id:
        return None
    days = _TF_TO_CG_DAYS.get(timeframe, 30)
    df = _coingecko_market_chart(coin_id, days)
    if df is not None and len(df) > limit:
        df = df.tail(limit).reset_index(drop=True)
    return df


def fetch_ohlcv(symbol: str, timeframe: str, limit: int = 120) -> pd.DataFrame | None:
    """Safe OHLCV fetch. Tries symbol variants on exchange, falls back to CoinGecko."""
    # Try symbol and its variants on the primary exchange
    for variant in _symbol_variants(symbol):
        try:
            return fetch_ohlcv_cached(variant, timeframe, limit)
        except Exception as e:
            _debug(f"fetch_ohlcv primary failed: {variant} {timeframe} (limit={limit}) → {e}")
    # Fallback: CoinGecko
    try:
        cg_df = _fetch_coingecko_ohlcv(symbol, timeframe, limit)
        if cg_df is not None and not cg_df.empty:
            _debug(f"fetch_ohlcv CoinGecko fallback OK: {symbol} {timeframe} → {len(cg_df)} rows")
            return cg_df
    except Exception as e:
        _debug(f"fetch_ohlcv CoinGecko fallback failed: {symbol} {timeframe} → {e}")
    return None


@st.cache_data(ttl=120, show_spinner=False)
def get_major_ohlcv_bundle(timeframe: str, limit: int = 500) -> dict[str, pd.DataFrame | None]:
    """Fetch a bundle of major market OHLCV frames for a timeframe.
    Keys are always BASE/USDT for consistency (even if exchange uses /USD)."""
    majors = ["BTC/USDT", "ETH/USDT", "BNB/USDT", "SOL/USDT", "ADA/USDT", "XRP/USDT"]
    out: dict[str, pd.DataFrame | None] = {}
    for sym in majors:
        # fetch_ohlcv already tries USDT→USD variants via _symbol_variants
        out[sym] = fetch_ohlcv(sym, timeframe, limit=limit)
    return out

def _ind_color(val: str) -> str:
    """Return colour for an indicator value (Bullish=green, Bearish=red, else=yellow)."""
    if any(k in val for k in ["Bullish", "Above", "Oversold", "Low"]):
        return POSITIVE
    if any(k in val for k in ["Bearish", "Below", "Overbought", "High"]):
        return NEGATIVE
    return WARNING


def signal_badge(signal: str) -> str:
    """Return a simplified badge for the given signal."""
    if signal in ("STRONG BUY", "BUY"):
        return "🟢 LONG"
    elif signal in ("STRONG SELL", "SELL"):
        return "🔴 SHORT"
    else:
        return "⚪ WAIT"


def leverage_badge(lev: int) -> str:
    """Display leverage as a formatted badge (e.g. x5)."""
    return f"x{lev}"


def confidence_score_badge(confidence: float) -> str:
    score = round(confidence)
    if score >= 80:
        label = "STRONG BUY"
    elif score >= 60:
        label = "BUY"
    elif score >= 40:
        label = "WAIT"
    elif score >= 20:
        label = "SELL"
    else:
        label = "STRONG SELL"
    return f"{score} ({label})"


def signal_plain(signal: str) -> str:
    """Map detailed signals to a plain LONG/SHORT/WAIT label."""
    if signal in ("STRONG BUY", "BUY"):
        return "LONG"
    elif signal in ("STRONG SELL", "SELL"):
        return "SHORT"
    else:
        return "WAIT"

def format_delta(delta):
     if delta is None:
         return ''
     if delta > 0:
         triangle = "▲"
     elif delta < 0:
         triangle = "▼"
     else:
         triangle = "→"
     return f"{triangle} {abs(delta):.2f}%"

def format_trend(trend: str) -> str:
    if trend == "Bullish":
        return "▲ Bullish"
    elif trend == "Bearish":
        return "▼ Bearish"
    else:
        return "–"

def format_adx(adx: float) -> str:
    if adx < 20:
        return f"▼ {adx:.1f} (Weak)"
    elif adx < 25:
        return f"→ {adx:.1f} (Starting)"
    elif adx < 50:
        return f"▲ {adx:.1f} (Strong)"
    elif adx < 75:
        return f"▲▲ {adx:.1f} (Very Strong)"
    else:
        return f"🔥 {adx:.1f} (Extreme)"

def format_stochrsi(value):
    if value < 0.2:
        return "🟢 Low"
    elif value > 0.8:
        return "🔴 High"
    else:
        return "→ Neutral"


def style_delta(val: str) -> str:
    if val.startswith("▲"):
        return f'color: {POSITIVE}; font-weight: 600;'
    elif val.startswith("▼"):
        return f'color: {NEGATIVE}; font-weight: 600;'
    return ''

def style_signal(val: str) -> str:
    if 'LONG' in val:
        return f'color: {POSITIVE}; font-weight: 600;'
    if 'SHORT' in val:
        return f'color: {NEGATIVE}; font-weight: 600;'
    return f'color: {WARNING}; font-weight: 600;'

def style_confidence(val: str) -> str:
    if "STRONG BUY" in val or "BUY" in val:
        return f"color: {POSITIVE}; font-weight: 600;"
    elif "WAIT" in val:
        return f"color: {WARNING}; font-weight: 600;"
    else:
        return f"color: {NEGATIVE}; font-weight: 600;"

            
def style_scalp_opp(val: str) -> str:
    if val == "LONG":
        return f'color: {POSITIVE}; font-weight: 600;'
    elif val == "SHORT":
        return f'color: {NEGATIVE}; font-weight: 600;'
    return ''

def readable_market_cap(value):
    trillion = 1_000_000_000_000
    billion = 1_000_000_000
    million = 1_000_000

    if value >= trillion:
        return f"{value / trillion:.2f}T"
    elif value >= billion:
        return f"{value / billion:.2f}B"
    elif value >= million:
        return f"{value / million:.2f}M"
    else:
        return f"{value:,}"

def detect_volume_spike(df: pd.DataFrame, window: int = 20, multiplier: float = 2.0) -> bool:
    """
    Detects if the last volume bar is a significant spike compared to recent average.

    Parameters:
    - df: DataFrame containing 'volume' column.
    - window: Number of past bars to average.
    - multiplier: Threshold multiplier for detecting spike.

    Returns:
    - True if spike detected, otherwise False.
    """

    if 'volume' not in df.columns or len(df) < window + 1:
        return False

    recent_volumes = df['volume'].iloc[-(window+1):-1]
    avg_volume = recent_volumes.mean()
    last_volume = df['volume'].iloc[-1]

    return last_volume > avg_volume * multiplier

def detect_candle_pattern(df: pd.DataFrame) -> str:
    if df is None or len(df) < 5:
        return ""

    last = df.iloc[-1]
    prev = df.iloc[-2]
    prev2 = df.iloc[-3]

    body_last = abs(last['close'] - last['open'])
    body_prev = abs(prev['close'] - prev['open'])
    body_prev2 = abs(prev2['close'] - prev2['open'])

    # Bullish Engulfing
    if prev['close'] < prev['open'] and last['close'] > last['open'] and \
       last['close'] > prev['open'] and last['open'] < prev['close']:
        return "▲ Bullish Engulfing (strong reversal up)"

    # Bearish Engulfing
    if prev['close'] > prev['open'] and last['close'] < last['open'] and \
       last['open'] > prev['close'] and last['close'] < prev['open']:
        return "▼ Bearish Engulfing (strong reversal down)"

    # Hammer
    lower_shadow = min(last['open'], last['close']) - last['low']
    upper_shadow = last['high'] - max(last['open'], last['close'])
    if body_last < lower_shadow and upper_shadow < lower_shadow * 0.5:
        return "▲ Hammer (bullish bottom wick)"

    # Inverted Hammer
    if upper_shadow > 2 * body_last and lower_shadow < body_last:
        return "▲ Inverted Hammer (potential bottom reversal)"

    # Hanging Man
    if lower_shadow > 2 * body_last and upper_shadow < body_last:
        return "▼ Hanging Man (possible top reversal)"

    # Shooting Star
    if upper_shadow > 2 * body_last and lower_shadow < body_last and last['close'] < last['open']:
        return "▼ Shooting Star (bearish top wick)"

    # Doji
    if body_last / (last['high'] - last['low'] + 1e-9) < 0.1:
        return "- Doji (market indecision)"

    if prev2['close'] < prev2['open'] and \
       body_prev < min(body_prev2, body_last) and \
       last['close'] > last['open'] and last['close'] > ((prev2['open'] + prev2['close']) / 2):
        return "▲ Morning Star (3-bar bullish reversal)"

    if prev2['close'] > prev2['open'] and \
       body_prev < min(body_prev2, body_last) and \
       last['close'] < last['open'] and last['close'] < ((prev2['open'] + prev2['close']) / 2):
        return "▼ Evening Star (3-bar bearish reversal)"

    # Piercing Line
    if prev['close'] < prev['open'] and last['open'] < prev['close'] and \
       last['close'] > ((prev['open'] + prev['close']) / 2) and last['close'] < prev['open']:
        return "▲ Piercing Line (mid-level reversal)"

    # Dark Cloud Cover
    if prev['close'] > prev['open'] and last['open'] > prev['close'] and \
       last['close'] < ((prev['open'] + prev['close']) / 2) and last['close'] > prev['open']:
        return "▼ Dark Cloud Cover (mid-level reversal)"

    # Three White Soldiers
    if all(df.iloc[-i]['close'] > df.iloc[-i]['open'] for i in range(1, 4)):
        return "▲ Three White Soldiers (strong bullish confirmation)"

    # Three Black Crows
    if all(df.iloc[-i]['close'] < df.iloc[-i]['open'] for i in range(1, 4)):
        return "▼ Three Black Crows (strong bearish confirmation)"

    return ""


def explain_candle_pattern(pattern: str) -> str:
    explanations = {
        "Hammer": "bullish bottom wick",
        "Bullish Engulfing": "strong reversal up",
        "Morning Star": "3-bar bullish reversal",
        "Piercing Line": "mid-level reversal",
        "Inverted Hammer": "potential bottom reversal",
        "Three White Soldiers": "strong bullish confirmation",

        "Shooting Star": "bearish top wick",
        "Bearish Engulfing": "strong reversal down",
        "Evening Star": "3-bar bearish reversal",
        "Dark Cloud Cover": "mid-level reversal",
        "Hanging Man": "possible top reversal",
        "Three Black Crows": "strong bearish confirmation",
        "Doji": "market indecision"
    }
    return explanations.get(pattern, "")

def get_signal_from_confidence(confidence: float) -> Tuple[str, str]:
    score = round(confidence)
    if score >= 80:
        return "STRONG BUY", "🚀 Strong bullish bias. High confidence to go LONG."
    elif score >= 60:
        return "BUY", "📈 Bullish leaning. Consider LONG entry."
    elif score >= 40:
        return "WAIT", "⏳ No clear direction. Market indecision."
    elif score >= 20:
        return "SELL", "📉 Bearish leaning. SHORT may be considered."
    else:
        return "STRONG SELL", "⚠️ Strong bearish bias. SHORT with high confidence."

def analyse(df: pd.DataFrame) -> AnalysisResult:

    if df is None or len(df) < 55:
        return AnalysisResult(comment="Insufficient data")

    # Infer timeframe from candle spacing for adaptive S/R lookback
    _inferred_tf = None
    if "timestamp" in df.columns and len(df) >= 2:
        _delta_mins = (df["timestamp"].iloc[-1] - df["timestamp"].iloc[-2]).total_seconds() / 60
        if _delta_mins <= 1.5:
            _inferred_tf = "1m"
        elif _delta_mins <= 4:
            _inferred_tf = "3m"
        elif _delta_mins <= 7:
            _inferred_tf = "5m"
        elif _delta_mins <= 20:
            _inferred_tf = "15m"
        elif _delta_mins <= 90:
            _inferred_tf = "1h"
        elif _delta_mins <= 300:
            _inferred_tf = "4h"
        else:
            _inferred_tf = "1d"

    df["ema5"] = ta.trend.ema_indicator(df["close"], window=5)
    df["ema9"] = ta.trend.ema_indicator(df["close"], window=9)
    df["ema21"] = ta.trend.ema_indicator(df["close"], window=21)
    df["ema50"] = ta.trend.ema_indicator(df["close"], window=50)
    df["rsi"] = ta.momentum.rsi(df["close"], window=14)
    macd_ind = ta.trend.MACD(df["close"])
    df["macd"] = macd_ind.macd()
    df["macd_signal"] = macd_ind.macd_signal()
    df["macd_diff"] = macd_ind.macd_diff()
    df["obv"] = ta.volume.on_balance_volume(df["close"], df["volume"])
    df["atr"] = ta.volatility.average_true_range(df["high"], df["low"], df["close"], window=14)

    # VWAP Calculation
    df["typical_price"] = (df["high"] + df["low"] + df["close"]) / 3
    _vwap_den = df["volume"].cumsum().replace(0, np.nan)
    df["vwap"] = (df["typical_price"] * df["volume"]).cumsum() / _vwap_den

    # Parabolic SAR
    try:
        psar_ind = ta.trend.PSARIndicator(high=df['high'], low=df['low'], close=df['close'])
        psar_up = psar_ind.psar_up()
        psar_down = psar_ind.psar_down()
        df["psar"] = psar_up.fillna(psar_down)
    except Exception as e:
        _debug(f"PSAR Error: {e}")
        df["psar"] = np.nan

    # Williams %R
    df["williams_r"] = ta.momentum.williams_r(df["high"], df["low"], df["close"], lbp=14)

    # CCI calculation
    df["cci"] = ta.trend.cci(df["high"], df["low"], df["close"], window=20)
    cci_val = df["cci"].iloc[-1]
            
    # Ichimoku Cloud
    ichimoku = ta.trend.IchimokuIndicator(
        high=df["high"],
        low=df["low"],
        window1=9,
        window2=26,
        window3=52,
        visual=False 
    )
    df["tenkan"] = ichimoku.ichimoku_conversion_line()
    df["kijun"] = ichimoku.ichimoku_base_line()
    df["senkou_a"] = ichimoku.ichimoku_a()
    df["senkou_b"] = ichimoku.ichimoku_b()
    
    # Use the last *closed* candle to avoid repainting on live/updating charts
    latest = df.iloc[-2] if len(df) >= 2 else df.iloc[-1]

    vwap_val = latest.get("vwap", np.nan)
    if pd.isna(vwap_val):
        vwap_label = "Unavailable"
    elif latest["close"] > vwap_val:
        vwap_label = "🟢 Above"
    elif latest["close"] < vwap_val:
        vwap_label = "🔴 Below"
    else:
        vwap_label = "→ Near VWAP"


    volume_spike = detect_volume_spike(df)
    candle_pattern = detect_candle_pattern(df)
    candle_label = candle_pattern.split(" (")[0] if candle_pattern else ""

    atr_latest = latest["atr"]
    if atr_latest > latest["close"] * 0.05:
        atr_comment = "▲ High"
    elif atr_latest < latest["close"] * 0.02:
        atr_comment = "▼ Low"
    else:
        atr_comment = "– Moderate"

    # ADX calculation
    adx_series = ta.trend.adx(df['high'], df['low'], df['close'], window=14)
    df['adx'] = adx_series
    adx_val = df['adx'].iloc[-1]

    # === BIAS SCORE CALCULATION ===
    # Categories: Trend, Momentum, Volume, Volatility
    
    # SuperTrend calculation
    try:
        st_data = _supertrend(df['high'], df['low'], df['close'], length=10, multiplier=3.0)
        df['supertrend'] = st_data[st_data.columns[0]]
    except Exception as e:
        _debug(f"SuperTrend Error: {e}")
        df['supertrend'] = np.nan

    # Stochastic RSI
    stoch_rsi = ta.momentum.StochRSIIndicator(close=df["close"], window=14, smooth1=3, smooth2=3)
    df["stochrsi_k"] = stoch_rsi.stochrsi_k()
    
    # Bollinger Bands (ddof=0 for population std, matching standard BB definition)
    df['bb_mid'] = df['close'].rolling(window=20).mean()
    df['bb_std'] = df['close'].rolling(window=20).std(ddof=0)
    df['bb_upper'] = df['bb_mid'] + 2 * df['bb_std']
    df['bb_lower'] = df['bb_mid'] - 2 * df['bb_std']
    
    latest = df.iloc[-1]
    stochrsi_k_val = latest.get("stochrsi_k", np.nan)
    williams_val = latest["williams_r"]
    psar_val = latest.get("psar", np.nan)
    bb_upper = latest['bb_upper']
    bb_lower = latest['bb_lower']
    bb_range = bb_upper - bb_lower
    bb_buffer = bb_range * 0.01
    close_price = latest['close']
    
    # === CATEGORY 1: TREND SCORE (-1 to +1) ===
    trend_signals = []
    
    # EMA Alignment (strongest trend indicator)
    if latest["ema5"] > latest["ema21"] > latest["ema50"]:
        trend_signals.append(1.0)  # Strong uptrend
    elif latest["ema5"] < latest["ema21"] < latest["ema50"]:
        trend_signals.append(-1.0)  # Strong downtrend
    elif latest["ema5"] > latest["ema21"]:
        trend_signals.append(0.5)  # Moderate uptrend
    elif latest["ema5"] < latest["ema21"]:
        trend_signals.append(-0.5)  # Moderate downtrend
    else:
        trend_signals.append(0.0)
    
    # SuperTrend
    supertrend_trend = "Unavailable"
    st_val = latest.get("supertrend", np.nan)
    if pd.notna(st_val):
        if latest["close"] > st_val:
            trend_signals.append(0.5)
            supertrend_trend = "Bullish"
        elif latest["close"] < st_val:
            trend_signals.append(-0.5)
            supertrend_trend = "Bearish"
    
    # Ichimoku Cloud
    try:
        sa = latest.get("senkou_a", np.nan)
        sb = latest.get("senkou_b", np.nan)
        if pd.isna(sa) or pd.isna(sb):
            ichimoku_trend = "Unavailable"
        else:
            if latest["close"] > max(sa, sb):
                trend_signals.append(0.5)
                ichimoku_trend = "Bullish"
            elif latest["close"] < min(sa, sb):
                trend_signals.append(-0.5)
                ichimoku_trend = "Bearish"
            else:
                ichimoku_trend = "Neutral"
    except Exception:
        ichimoku_trend = "Unavailable"
    
    # Parabolic SAR
    psar_trend = ""
    if pd.notna(psar_val):
        if latest["close"] > psar_val:
            trend_signals.append(0.3)
            psar_trend = "▲ Bullish"
        elif latest["close"] < psar_val:
            trend_signals.append(-0.3)
            psar_trend = "▼ Bearish"
    
    # ADX - Trend strength (not direction)
    if adx_val >= 25:
        # Strong trend exists, amplify current trend signals
        trend_strength = min(adx_val / 50, 1.0)  # Normalize to 0-1
        trend_signals = [s * (1 + trend_strength * 0.5) for s in trend_signals]
    
    trend_score = np.mean(trend_signals) if trend_signals else 0.0
    trend_score = np.clip(trend_score, -1, 1)
    
    # === CATEGORY 2: MOMENTUM SCORE (-1 to +1) ===
    momentum_signals = []
    
    # RSI
    rsi_val = latest["rsi"]
    if rsi_val > 70:
        momentum_signals.append(-0.5)  # Overbought
    elif rsi_val > 55:
        momentum_signals.append(0.5)  # Bullish momentum
    elif rsi_val < 30:
        momentum_signals.append(0.5)  # Oversold (reversal potential)
    elif rsi_val < 45:
        momentum_signals.append(-0.5)  # Bearish momentum
    else:
        momentum_signals.append(0.0)
    
    # MACD
    if latest["macd"] > latest["macd_signal"] and latest["macd_diff"] > 0:
        momentum_signals.append(1.0)  # Strong bullish
    elif latest["macd"] < latest["macd_signal"] and latest["macd_diff"] < 0:
        momentum_signals.append(-1.0)  # Strong bearish
    elif latest["macd"] > latest["macd_signal"]:
        momentum_signals.append(0.5)
    elif latest["macd"] < latest["macd_signal"]:
        momentum_signals.append(-0.5)
    
    # Stochastic RSI
    if stochrsi_k_val >= 0.9:
        momentum_signals.append(-0.5)  # Overbought
    elif stochrsi_k_val <= 0.1:
        momentum_signals.append(0.5)  # Oversold
    elif stochrsi_k_val >= 0.8:
        momentum_signals.append(-0.3)
    elif stochrsi_k_val <= 0.2:
        momentum_signals.append(0.3)
    
    # Williams %R
    if williams_val < -80:
        momentum_signals.append(0.5)  # Oversold
    elif williams_val > -20:
        momentum_signals.append(-0.5)  # Overbought
    
    # CCI
    if cci_val > 100:
        momentum_signals.append(-0.5)  # Overbought
    elif cci_val < -100:
        momentum_signals.append(0.5)  # Oversold
    
    momentum_score = np.mean(momentum_signals) if momentum_signals else 0.0
    momentum_score = np.clip(momentum_score, -1, 1)
    
    # === CATEGORY 3: VOLUME SCORE (-1 to +1) ===
    volume_signals = []
    
    # OBV Trend (guard against short data)
    _obv_back = min(5, len(df) - 1)
    if _obv_back > 0:
        if df["obv"].iloc[-1] > df["obv"].iloc[-_obv_back]:
            volume_signals.append(0.5)
        elif df["obv"].iloc[-1] < df["obv"].iloc[-_obv_back]:
            volume_signals.append(-0.5)
    
    # Volume Spike
    if volume_spike:
        if latest["close"] > latest["open"]:
            volume_signals.append(0.5)  # Bullish volume
        else:
            volume_signals.append(-0.5)  # Bearish volume
    
    # VWAP Position
    if latest["close"] > latest["vwap"]:
        volume_signals.append(0.5)
    elif latest["close"] < latest["vwap"]:
        volume_signals.append(-0.5)
    
    volume_score = np.mean(volume_signals) if volume_signals else 0.0
    volume_score = np.clip(volume_score, -1, 1)
    
    # === CATEGORY 4: VOLATILITY SCORE (-1 to +1) ===
    volatility_signals = []
    
    # ATR - Lower is better for entry
    atr_ratio = atr_latest / latest["close"]
    if atr_ratio < 0.015:
        volatility_signals.append(0.5)  # Low volatility - good
    elif atr_ratio > 0.05:
        volatility_signals.append(-0.5)  # High volatility - risky
    
    # Bollinger Band Width - Narrower is better
    bb_width_pct = bb_range / latest['close']
    if bb_width_pct < 0.05:
        volatility_signals.append(0.5)  # Tight bands
    elif bb_width_pct > 0.15:
        volatility_signals.append(-0.5)  # Wide bands
    
    volatility_score = np.mean(volatility_signals) if volatility_signals else 0.0
    volatility_score = np.clip(volatility_score, -1, 1)
    
    # === WEIGHTED COMBINATION ===
    weights = {
        'trend': 0.40,       # Most important
        'momentum': 0.30,
        'volume': 0.20,
        'volatility': 0.10
    }
    
    final_score = (
        trend_score * weights['trend'] +
        momentum_score * weights['momentum'] +
        volume_score * weights['volume'] +
        volatility_score * weights['volatility']
    )
    
    # Convert to 0-100 confidence
    confidence_score = round((final_score + 1) / 2 * 100, 1)
    confidence_score = float(np.clip(confidence_score, 0, 100))

    # Regime-weighted confidence: discount when market is ranging (ADX < 20)
    if not pd.isna(adx_val) and adx_val < 20:
        _regime_discount = np.interp(adx_val, [0, 20], [0.70, 1.0])  # up to 30% discount
        confidence_score = round(confidence_score * _regime_discount, 1)
        confidence_score = float(np.clip(confidence_score, 0, 100))

    # === SIGNAL GENERATION WITH QUALITY FILTERS ===
    
    # Adaptive thresholds based on volatility and trend strength
    if volatility_score < -0.3:  # High volatility
        buy_threshold = 70
        sell_threshold = 30
    elif adx_val < 20:  # Weak trend
        buy_threshold = 75
        sell_threshold = 25
    else:  # Normal conditions
        buy_threshold = 65
        sell_threshold = 35
    
    # Initial signal based on confidence
    if confidence_score >= buy_threshold:
        base_signal = "BUY"
    elif confidence_score <= sell_threshold:
        base_signal = "SELL"
    else:
        base_signal = "WAIT"
    
    # Quality filters - must pass these for BUY/SELL signals
    if base_signal == "BUY":
        # Check trend alignment
        if trend_score < 0.2:  # Not bullish enough
            signal = "WAIT"
            comment = "⏳ Bullish setup incomplete. Trend not confirmed."
        # Check volume support
        elif volume_score < 0:  # Negative volume
            signal = "WAIT"
            comment = "⏳ Bullish setup needs volume confirmation."
        # Check momentum
        elif momentum_score < -0.3:  # Bearish momentum
            signal = "WAIT"
            comment = "⏳ Bullish setup but momentum divergence detected."
        # Check volatility
        elif volatility_score < -0.5:  # Too volatile
            signal = "WAIT"
            comment = "⚠️ High volatility detected. Wait for calmer conditions."
        # All filters passed
        else:
            if confidence_score >= 80:
                signal = "STRONG BUY"
                comment = "🚀 Strong bullish bias. High confidence to go LONG."
            else:
                signal = "BUY"
                comment = "📈 Bullish leaning. Consider LONG entry."
    
    elif base_signal == "SELL":
        # Check trend alignment
        if trend_score > -0.2:  # Not bearish enough
            signal = "WAIT"
            comment = "⏳ Bearish setup incomplete. Trend not confirmed."
        # Check volume support
        elif volume_score > 0:  # Positive volume
            signal = "WAIT"
            comment = "⏳ Bearish setup needs volume confirmation."
        # Check momentum
        elif momentum_score > 0.3:  # Bullish momentum
            signal = "WAIT"
            comment = "⏳ Bearish setup but momentum divergence detected."
        # Check volatility
        elif volatility_score < -0.5:  # Too volatile
            signal = "WAIT"
            comment = "⚠️ High volatility detected. Wait for calmer conditions."
        # All filters passed
        else:
            if confidence_score <= 20:
                signal = "STRONG SELL"
                comment = "⚠️ Strong bearish bias. SHORT with high confidence."
            else:
                signal = "SELL"
                comment = "📉 Bearish leaning. SHORT may be considered."
    
    else:  # WAIT signal
        signal = "WAIT"
        # Provide helpful context
        if abs(trend_score) < 0.1:
            comment = "⏳ No clear trend direction. Market ranging."
        elif abs(momentum_score) < 0.1:
            comment = "⏳ Weak momentum. Wait for stronger signals."
        elif volatility_score < -0.5:
            comment = "⚠️ High volatility. Risky conditions."
        else:
            comment = "⏳ Mixed signals. No clear direction."

    # Williams %R display label
    williams_label = ""
    if not np.isnan(williams_val):
        if williams_val < -80:
            williams_label = "🟢 Oversold"
        elif williams_val > -20:
            williams_label = "🔴 Overbought"
        else:
            williams_label = "🟡 Neutral"

    # CCI label (for display)
    cci_label = "🟡 Neutral"
    if cci_val > 100:
        cci_label = "🔴 Overbought"
    elif cci_val < -100:
        cci_label = "🟢 Oversold"
        
    # Bollinger Label (for display)
    bollinger_bias = "→ Neutral"
    if close_price > bb_upper + bb_buffer:
        bollinger_bias = "🔴 Overbought"
    elif close_price > bb_upper:
        bollinger_bias = "→ Near Top"
    elif close_price < bb_lower - bb_buffer:
        bollinger_bias = "🟢 Oversold"
    elif close_price < bb_lower:
        bollinger_bias = "→ Near Bottom"

    
    # === Leverage calculation (IMPROVED) ===
    risk_score = 0.0
    bollinger_width = (df['bb_upper'].iloc[-1] - df['bb_lower'].iloc[-1])
    volatility_factor = min(bollinger_width / max(latest["close"], 1e-9), 0.1)
    rsi_factor = 0.1 if latest["rsi"] > 70 or latest["rsi"] < 30 else 0
    _obv_back_lev = min(5, len(df) - 1)
    obv_factor = 0.1 if (_obv_back_lev > 0 and df["obv"].iloc[-1] > df["obv"].iloc[-_obv_back_lev] and latest["close"] > latest["ema21"]) else 0
    _sr_bars = _sr_lookback(_inferred_tf)
    recent = df.tail(_sr_bars)
    support = recent["low"].min()
    resistance = recent["high"].max()
    current_price = latest["close"]
    sr_factor = 0.1 if abs(current_price - support) / current_price < 0.02 or abs(current_price - resistance) / current_price < 0.02 else 0
    risk_score = volatility_factor + rsi_factor + obv_factor + sr_factor

    # More conservative leverage recommendations
    if risk_score <= 0.15:
        lev_base = int(round(np.interp(risk_score, [0.00, 0.15], [10, 6])))
    elif risk_score <= 0.25:
        lev_base = int(round(np.interp(risk_score, [0.15, 0.25], [6, 4])))
    else:
        rs = min(risk_score, 0.40)
        lev_base = int(round(np.interp(rs, [0.25, 0.40], [4, 2])))

    # Even more conservative based on confidence
    if confidence_score < 50:
        lev_base = min(lev_base, 2)  # Very low leverage for low confidence
    elif confidence_score < 65:
        lev_base = min(lev_base, 4)
    elif confidence_score < 75:
        lev_base = min(lev_base, 6)

    return AnalysisResult(
        signal=signal, leverage=lev_base, comment=comment, volume_spike=volume_spike,
        atr_comment=atr_comment, candle_pattern=candle_pattern, confidence=confidence_score,
        adx=adx_val, supertrend=supertrend_trend, ichimoku=ichimoku_trend,
        stochrsi_k=stochrsi_k_val, bollinger=bollinger_bias, vwap=vwap_label,
        psar=psar_trend, williams=williams_label, cci=cci_label,
    )


def get_scalping_entry_target(
    df: pd.DataFrame,
    confidence_score: float,
    supertrend_trend: str,
    ichimoku_trend: str,
    vwap_label: str,
    volume_spike: bool,
    strict_mode: bool = True
):
    if df is None or len(df) <= 30:
        return None, 0.0, 0.0, 0.0, 0.0, ""

    # === Indicators ===
    df['ema5'] = df['close'].ewm(span=5).mean()
    df['ema13'] = df['close'].ewm(span=13).mean()
    df['ema21'] = df['close'].ewm(span=21).mean()
    df['atr'] = ta.volatility.average_true_range(df['high'], df['low'], df['close'], window=14)
    df['rsi'] = ta.momentum.rsi(df['close'], window=14)

    # MACD
    macd_ind = ta.trend.MACD(df['close'])
    df['macd'] = macd_ind.macd()
    df['macd_signal'] = macd_ind.macd_signal()
    df['macd_diff'] = macd_ind.macd_diff()

    # ADX
    adx_val = float(ta.trend.adx(df['high'], df['low'], df['close'], window=14).iloc[-1])

    # StochRSI
    stoch = ta.momentum.StochRSIIndicator(close=df['close'], window=14, smooth1=3, smooth2=3)
    stochrsi_k_val = float(stoch.stochrsi_k().iloc[-1])

    # Bollinger Bands bias
    bb = ta.volatility.BollingerBands(close=df['close'], window=20, window_dev=2)
    last_close = df['close'].iloc[-1]
    hband = float(bb.bollinger_hband().iloc[-1])
    lband = float(bb.bollinger_lband().iloc[-1])
    if last_close > hband:
        bollinger_bias = "Overbought"
    elif last_close < lband:
        bollinger_bias = "Oversold"
    else:
        bollinger_bias = "Neutral"

    latest = df.iloc[-1]
    close_price = latest['close']
    atr = latest['atr']

    # === Trend Confirmation ===
    ema_trend_up = latest['ema5'] > latest['ema13'] > latest['ema21']
    ema_trend_down = latest['ema5'] < latest['ema13'] < latest['ema21']
    macd_confirm = latest['macd'] > latest['macd_signal'] and latest['macd_diff'] > 0
    rsi_confirm_long = latest['rsi'] > 55
    rsi_confirm_short = latest['rsi'] < 45

    # === Direction ===
    scalp_direction = None
    if confidence_score >= 65 and ema_trend_up and macd_confirm and rsi_confirm_long:
        scalp_direction = "LONG"
    elif confidence_score <= 35 and ema_trend_down and not macd_confirm and rsi_confirm_short:
        scalp_direction = "SHORT"

    # === Strict Mode Filters ===
    if strict_mode and scalp_direction is not None:
        # 1) Trend strength: ADX < 20 volume spike is mandatory
        if (pd.isna(adx_val) or adx_val < 20) and not volume_spike:
            return None, None, None, None, None, "No trend strength / no volume"
    
        # 2) Confirmation Supertrend / Ichimoku / VWAP (min 2)
        if scalp_direction == "LONG":
            confirms = 0
            confirms += 1 if supertrend_trend == "Bullish" else 0
            confirms += 1 if ichimoku_trend == "Bullish" else 0
            confirms += 1 if vwap_label == "🟢 Above" else 0
            if confirms < 2:
                return None, None, None, None, None, "Regime filters not aligned (need 2/3)"
    
            # 3) Over and momentum
            if bollinger_bias == "Overbought":
                return None, None, None, None, None, "Overbought"
            if not (0.20 <= stochrsi_k_val <= 0.85 and rsi_confirm_long and macd_confirm):
                return None, None, None, None, None, "Momentum fail"
    
        elif scalp_direction == "SHORT":
            confirms = 0
            confirms += 1 if supertrend_trend == "Bearish" else 0
            confirms += 1 if ichimoku_trend == "Bearish" else 0
            confirms += 1 if vwap_label == "🔴 Below" else 0
            if confirms < 2:
                return None, None, None, None, None, "Regime filters not aligned (need 2/3)"
    
            if bollinger_bias == "Oversold":
                return None, None, None, None, None, "Oversold"
            if not (0.15 <= stochrsi_k_val <= 0.80 and rsi_confirm_short and not macd_confirm):
                return None, None, None, None, None, "Momentum fail"
    
        # 4) Volatility
        if (atr / close_price) < 0.0015:  # 0.15%
            return None, None, None, None, None, "Low Volatility"

    # === Support / Resistance (timeframe-aware lookback) ===
    _scalp_sr_bars = _sr_lookback()  # default 30 for scalping context
    recent = df.tail(_scalp_sr_bars)
    support = recent['low'].min()
    resistance = recent['high'].max()

    # === Entry, Stop, Target ===
    entry_s = stop_s = target_s = 0.0
    breakout_note = ""
    rr_min = 1.5

    if scalp_direction == "LONG":
        entry_s  = max(close_price, latest['ema5']) + 0.25 * atr
        stop_s   = close_price - 0.75 * atr
        target_s = entry_s + 1.5 * atr
        if target_s > resistance:
            breakout_note = f"⚠️ Target (${target_s:.4f}) is above resistance (${resistance:.4f}). Breakout needed."
    elif scalp_direction == "SHORT":
        entry_s  = min(close_price, latest['ema5']) - 0.25 * atr
        stop_s   = close_price + 0.75 * atr
        target_s = entry_s - 1.5 * atr
        if target_s < support:
            breakout_note = f"⚠️ Target (${target_s:.4f}) is below support (${support:.4f}). Breakout needed."

    rr_ratio = abs(target_s - entry_s) / abs(entry_s - stop_s) if entry_s != stop_s else 0.0

    return scalp_direction, entry_s, target_s, stop_s, rr_ratio, breakout_note

# === Machine Learning Prediction ===
def ml_predict_direction(df: pd.DataFrame) -> tuple[float, str]:
    """
    Train an advanced machine‑learning classifier (Gradient Boosting) on recent candles to estimate whether
    the next candle's close will be higher (bullish) or lower (bearish).  This function generates
    a range of technical indicators and statistical features, fits a Gradient Boosting classifier to the
    historical data, and returns the probability of an upward move along with a directional label.
    """
    if df is None or len(df) < 60:
        return 0.5, "NEUTRAL"

    df = df.copy().reset_index(drop=True)

    # Calculate features - IMPROVED: Added more features
    df['ema5'] = ta.trend.ema_indicator(df['close'], window=5)
    df['ema9'] = ta.trend.ema_indicator(df['close'], window=9)
    df['ema21'] = ta.trend.ema_indicator(df['close'], window=21)
    df['rsi'] = ta.momentum.rsi(df['close'], window=14)
    macd_ind = ta.trend.MACD(df['close'])
    df['macd'] = macd_ind.macd()
    df['macd_signal'] = macd_ind.macd_signal()
    df['macd_diff'] = macd_ind.macd_diff()
    df['obv'] = ta.volume.on_balance_volume(df['close'], df['volume'])
    df['atr'] = ta.volatility.average_true_range(df['high'], df['low'], df['close'], window=14)
    
    # Additional features for better prediction
    df['returns'] = df['close'].pct_change()
    df['volume_ratio'] = df['volume'] / df['volume'].rolling(20).mean()
    bb = ta.volatility.BollingerBands(df['close'])
    df['bb_width'] = (bb.bollinger_hband() - bb.bollinger_lband()) / bb.bollinger_mavg()

    # Target: 1 if next close > current close, else 0
    df['target'] = (df['close'].shift(-1) > df['close']).astype(int)

    # Features: select numeric columns and shift by 1 to avoid look-ahead
    feature_cols = ['ema5','ema9','ema21','rsi','macd','macd_signal','macd_diff','obv','atr','returns','volume_ratio','bb_width']
    df_features = df[feature_cols].shift(1)
    df_model = pd.concat([df_features, df['target']], axis=1).dropna()

    if len(df_model) < 50:
        return 0.5, "NEUTRAL"

    X = df_model[feature_cols].astype(float).values
    y = df_model['target'].astype(int).values

    try:
        from sklearn.ensemble import GradientBoostingClassifier
        from sklearn.preprocessing import StandardScaler

        # Time-series aware split: train on first 80%, validate, predict last
        split_idx = int(len(X) * 0.8)
        X_train, y_train = X[:split_idx], y[:split_idx]
        X_pred = X[-1:]

        # Fit scaler only on training data to prevent data leakage
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_pred_scaled = scaler.transform(X_pred)

        model = GradientBoostingClassifier(
            n_estimators=100,
            max_depth=3,
            learning_rate=0.1,
            subsample=0.8,
            random_state=42
        )
        model.fit(X_train_scaled, y_train)

        prob_up = float(model.predict_proba(X_pred_scaled)[0][1])
    except Exception as e:
        _debug(f"GradientBoosting failed ({e}), falling back to LogisticRegression")
        try:
            split_idx = int(len(X) * 0.8)
            model = LogisticRegression(max_iter=1000)
            model.fit(X[:split_idx], y[:split_idx])
            prob_up = float(model.predict_proba(X[-1:].reshape(1, -1))[0][1])
        except Exception:
            return 0.5, "NEUTRAL"

    direction = "LONG" if prob_up >= 0.6 else ("SHORT" if prob_up <= 0.4 else "NEUTRAL")
    return prob_up, direction


# ╔══════════════════════════════════════════════════════════════╗
# ║                  ADVANCED ANALYSIS FUNCTIONS                  ║
# ╚══════════════════════════════════════════════════════════════╝

def ml_ensemble_predict(df: pd.DataFrame) -> tuple[float, str, dict]:
    """Ensemble ML prediction combining 3 models with voting.

    Returns (probability, direction, model_details_dict).
    """
    if df is None or len(df) < 60:
        return 0.5, "NEUTRAL", {}

    df = df.copy().reset_index(drop=True)
    df['ema5'] = ta.trend.ema_indicator(df['close'], window=5)
    df['ema9'] = ta.trend.ema_indicator(df['close'], window=9)
    df['ema21'] = ta.trend.ema_indicator(df['close'], window=21)
    df['rsi'] = ta.momentum.rsi(df['close'], window=14)
    macd_ind = ta.trend.MACD(df['close'])
    df['macd'] = macd_ind.macd()
    df['macd_signal'] = macd_ind.macd_signal()
    df['macd_diff'] = macd_ind.macd_diff()
    df['obv'] = ta.volume.on_balance_volume(df['close'], df['volume'])
    df['atr'] = ta.volatility.average_true_range(df['high'], df['low'], df['close'], window=14)
    df['returns'] = df['close'].pct_change()
    df['volume_ratio'] = df['volume'] / df['volume'].rolling(20).mean()
    bb = ta.volatility.BollingerBands(df['close'])
    df['bb_width'] = (bb.bollinger_hband() - bb.bollinger_lband()) / bb.bollinger_mavg()
    # Extra features for ensemble
    df['ema_spread'] = (df['ema5'] - df['ema21']) / df['close']
    df['rsi_slope'] = df['rsi'].diff(3)
    df['vol_trend'] = df['volume'].rolling(5).mean() / df['volume'].rolling(20).mean()

    df['target'] = (df['close'].shift(-1) > df['close']).astype(int)
    feature_cols = ['ema5','ema9','ema21','rsi','macd','macd_signal','macd_diff',
                    'obv','atr','returns','volume_ratio','bb_width',
                    'ema_spread','rsi_slope','vol_trend']
    df_features = df[feature_cols].shift(1)
    df_model = pd.concat([df_features, df['target']], axis=1).dropna()

    if len(df_model) < 50:
        return 0.5, "NEUTRAL", {}

    X = df_model[feature_cols].astype(float).values
    y = df_model['target'].astype(int).values

    try:
        split_idx = int(len(X) * 0.8)
        X_train, y_train = X[:split_idx], y[:split_idx]
        X_pred = X[-1:]

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_pred_scaled = scaler.transform(X_pred)

        gb = GradientBoostingClassifier(n_estimators=150, max_depth=4, learning_rate=0.08,
                                         subsample=0.85, random_state=42)
        rf = RandomForestClassifier(n_estimators=200, max_depth=5, random_state=42)
        lr = LogisticRegression(max_iter=1000, random_state=42)

        gb.fit(X_train_scaled, y_train)
        rf.fit(X_train_scaled, y_train)
        lr.fit(X_train_scaled, y_train)

        gb_prob = float(gb.predict_proba(X_pred_scaled)[0][1])
        rf_prob = float(rf.predict_proba(X_pred_scaled)[0][1])
        lr_prob = float(lr.predict_proba(X_pred_scaled)[0][1])

        # Weighted ensemble: GB has highest weight
        weights = [0.45, 0.35, 0.20]
        prob_up = gb_prob * weights[0] + rf_prob * weights[1] + lr_prob * weights[2]

        details = {
            'gradient_boosting': gb_prob,
            'random_forest': rf_prob,
            'logistic_regression': lr_prob,
            'ensemble': prob_up,
            'agreement': sum(1 for p in [gb_prob, rf_prob, lr_prob] if p > 0.5) / 3.0,
        }
    except Exception:
        return 0.5, "NEUTRAL", {}

    direction = "LONG" if prob_up >= 0.58 else ("SHORT" if prob_up <= 0.42 else "NEUTRAL")
    return prob_up, direction, details


def calculate_fibonacci_levels(df: pd.DataFrame, lookback: int = 100) -> dict:
    """Calculate Fibonacci retracement and extension levels from swing high/low."""
    if df is None or len(df) < 20:
        return {}

    data = df.tail(min(lookback, len(df)))
    swing_high = data['high'].max()
    swing_low = data['low'].min()
    swing_range = swing_high - swing_low

    if swing_range <= 0:
        return {}

    high_idx = data['high'].idxmax()
    low_idx = data['low'].idxmin()
    is_uptrend = low_idx < high_idx

    levels = {}
    fib_ratios = [0, 0.236, 0.382, 0.5, 0.618, 0.786, 1.0]
    fib_names = ['0%', '23.6%', '38.2%', '50%', '61.8%', '78.6%', '100%']

    for ratio, name in zip(fib_ratios, fib_names):
        if is_uptrend:
            price = swing_high - swing_range * ratio
        else:
            price = swing_low + swing_range * ratio
        levels[name] = price

    # Extension levels
    ext_ratios = [1.272, 1.618, 2.0, 2.618]
    ext_names = ['127.2%', '161.8%', '200%', '261.8%']
    for ratio, name in zip(ext_ratios, ext_names):
        if is_uptrend:
            price = swing_high - swing_range * ratio
        else:
            price = swing_low + swing_range * ratio
        levels[name] = price

    levels['_swing_high'] = swing_high
    levels['_swing_low'] = swing_low
    levels['_is_uptrend'] = is_uptrend
    return levels


def monte_carlo_simulation(df: pd.DataFrame, num_simulations: int = 500,
                           num_days: int = 30) -> dict:
    """Run Monte Carlo simulation for price prediction with confidence intervals."""
    if df is None or len(df) < 30:
        return {}

    returns = df['close'].pct_change().dropna()
    mu = returns.mean()
    sigma = returns.std()
    last_price = float(df['close'].iloc[-1])

    simulations = np.zeros((num_simulations, num_days))
    for i in range(num_simulations):
        daily_returns = np.random.normal(mu, sigma, num_days)
        price_path = last_price * np.cumprod(1 + daily_returns)
        simulations[i] = price_path

    final_prices = simulations[:, -1]
    return {
        'simulations': simulations,
        'last_price': last_price,
        'mean_price': float(np.mean(final_prices)),
        'median_price': float(np.median(final_prices)),
        'p5': float(np.percentile(final_prices, 5)),
        'p25': float(np.percentile(final_prices, 25)),
        'p75': float(np.percentile(final_prices, 75)),
        'p95': float(np.percentile(final_prices, 95)),
        'min_price': float(np.min(final_prices)),
        'max_price': float(np.max(final_prices)),
        'prob_profit': float(np.mean(final_prices > last_price)),
        'expected_return': float((np.mean(final_prices) - last_price) / last_price * 100),
        'var_95': float(np.percentile(final_prices / last_price - 1, 5) * 100),
    }


def detect_divergence(df: pd.DataFrame) -> list[dict]:
    """Detect RSI and MACD divergences — bullish and bearish."""
    if df is None or len(df) < 30:
        return []

    divergences = []
    df = df.copy()
    df['rsi'] = ta.momentum.rsi(df['close'], window=14)
    macd_ind = ta.trend.MACD(df['close'])
    df['macd'] = macd_ind.macd()

    lookback = min(20, len(df) - 5)
    recent = df.tail(lookback)

    # Find local minima and maxima in price
    for i in range(2, len(recent) - 2):
        idx = recent.index[i]
        prev2 = recent.iloc[i - 2]
        prev1 = recent.iloc[i - 1]
        curr = recent.iloc[i]
        next1 = recent.iloc[i + 1] if i + 1 < len(recent) else curr
        next2 = recent.iloc[i + 2] if i + 2 < len(recent) else curr

        # Check for local low in price but higher low in RSI (bullish divergence)
        if curr['low'] < prev1['low'] and curr['low'] < next1['low']:
            # Look for previous low
            for j in range(max(0, i - 10), i - 2):
                prev_low = recent.iloc[j]
                if prev_low['low'] < recent.iloc[max(0, j-1)]['low']:
                    if curr['close'] < prev_low['close'] and curr['rsi'] > prev_low['rsi']:
                        divergences.append({
                            'type': 'BULLISH RSI',
                            'description': 'Price making lower lows but RSI making higher lows',
                            'strength': 'STRONG',
                            'color': POSITIVE,
                        })
                        break

        # Check for local high in price but lower high in RSI (bearish divergence)
        if curr['high'] > prev1['high'] and curr['high'] > next1['high']:
            for j in range(max(0, i - 10), i - 2):
                prev_high = recent.iloc[j]
                if prev_high['high'] > recent.iloc[max(0, j-1)]['high']:
                    if curr['close'] > prev_high['close'] and curr['rsi'] < prev_high['rsi']:
                        divergences.append({
                            'type': 'BEARISH RSI',
                            'description': 'Price making higher highs but RSI making lower highs',
                            'strength': 'STRONG',
                            'color': NEGATIVE,
                        })
                        break

    # MACD divergence check
    last_5 = df.tail(5)
    if len(last_5) >= 5:
        price_trend = last_5['close'].iloc[-1] - last_5['close'].iloc[0]
        macd_trend = last_5['macd'].iloc[-1] - last_5['macd'].iloc[0]
        if price_trend > 0 and macd_trend < 0:
            divergences.append({
                'type': 'BEARISH MACD',
                'description': 'Price rising but MACD declining — momentum weakening',
                'strength': 'MODERATE',
                'color': WARNING,
            })
        elif price_trend < 0 and macd_trend > 0:
            divergences.append({
                'type': 'BULLISH MACD',
                'description': 'Price falling but MACD rising — selling pressure weakening',
                'strength': 'MODERATE',
                'color': WARNING,
            })

    return divergences


def calculate_risk_metrics(df: pd.DataFrame, risk_free_rate: float = 0.02) -> dict:
    """Calculate comprehensive risk metrics: VaR, Sharpe, Sortino, Calmar, Max Drawdown."""
    if df is None or len(df) < 20:
        return {}

    returns = df['close'].pct_change().dropna()
    if len(returns) < 10:
        return {}

    # Basic statistics
    mean_return = float(returns.mean())
    std_return = float(returns.std())
    total_return = float((df['close'].iloc[-1] / df['close'].iloc[0]) - 1)

    # Annualization factor (assume ~365 trading days for crypto)
    ann_factor = 365

    # Value at Risk (Historical)
    var_95 = float(np.percentile(returns, 5))
    var_99 = float(np.percentile(returns, 1))

    # Conditional VaR (Expected Shortfall)
    cvar_95 = float(returns[returns <= var_95].mean()) if len(returns[returns <= var_95]) > 0 else var_95

    # Sharpe Ratio (annualized)
    daily_rf = risk_free_rate / ann_factor
    excess_returns = returns - daily_rf
    sharpe = float(excess_returns.mean() / (excess_returns.std() + 1e-9) * np.sqrt(ann_factor))

    # Sortino Ratio (only penalizes downside volatility)
    downside_returns = returns[returns < 0]
    downside_std = float(downside_returns.std()) if len(downside_returns) > 0 else 1e-9
    sortino = float((mean_return - daily_rf) / (downside_std + 1e-9) * np.sqrt(ann_factor))

    # Maximum Drawdown
    cumulative = (1 + returns).cumprod()
    peak = cumulative.cummax()
    drawdown = (cumulative - peak) / peak
    max_drawdown = float(drawdown.min())
    max_dd_duration = 0
    dd_count = 0
    for dd in drawdown:
        if dd < 0:
            dd_count += 1
            max_dd_duration = max(max_dd_duration, dd_count)
        else:
            dd_count = 0

    # Calmar Ratio
    ann_return = mean_return * ann_factor
    calmar = float(ann_return / (abs(max_drawdown) + 1e-9))

    # Win rate
    win_rate = float((returns > 0).sum() / len(returns) * 100)

    # Skewness and Kurtosis
    skewness = float(returns.skew())
    kurtosis = float(returns.kurtosis())

    # Best/Worst day
    best_day = float(returns.max() * 100)
    worst_day = float(returns.min() * 100)

    return {
        'total_return': total_return * 100,
        'ann_return': ann_return * 100,
        'ann_volatility': std_return * np.sqrt(ann_factor) * 100,
        'sharpe': sharpe,
        'sortino': sortino,
        'calmar': calmar,
        'max_drawdown': max_drawdown * 100,
        'max_dd_duration': max_dd_duration,
        'var_95': var_95 * 100,
        'var_99': var_99 * 100,
        'cvar_95': cvar_95 * 100,
        'win_rate': win_rate,
        'skewness': skewness,
        'kurtosis': kurtosis,
        'best_day': best_day,
        'worst_day': worst_day,
        'mean_daily': mean_return * 100,
        'drawdown_series': drawdown,
        'cumulative_returns': cumulative,
    }


def calculate_volume_profile(df: pd.DataFrame, num_bins: int = 30) -> dict:
    """Calculate volume at price levels for volume profile visualization."""
    if df is None or len(df) < 10:
        return {}

    price_min = df['low'].min()
    price_max = df['high'].max()
    bin_edges = np.linspace(price_min, price_max, num_bins + 1)
    volumes = np.zeros(num_bins)

    for _, row in df.iterrows():
        for b in range(num_bins):
            if row['low'] <= bin_edges[b + 1] and row['high'] >= bin_edges[b]:
                overlap = min(row['high'], bin_edges[b + 1]) - max(row['low'], bin_edges[b])
                total_range = row['high'] - row['low']
                if total_range > 0:
                    volumes[b] += row['volume'] * (overlap / total_range)

    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    poc_idx = int(np.argmax(volumes))

    return {
        'bin_centers': bin_centers,
        'volumes': volumes,
        'poc_price': float(bin_centers[poc_idx]),
        'poc_volume': float(volumes[poc_idx]),
        'value_area_high': float(bin_centers[min(poc_idx + int(num_bins * 0.35), num_bins - 1)]),
        'value_area_low': float(bin_centers[max(poc_idx - int(num_bins * 0.35), 0)]),
    }


def detect_market_regime(df: pd.DataFrame) -> dict:
    """Detect current market regime: Trending, Ranging, Volatile, or Breakout."""
    if df is None or len(df) < 30:
        return {'regime': 'UNKNOWN', 'color': TEXT_MUTED}

    adx = ta.trend.adx(df['high'], df['low'], df['close'], window=14)
    adx_val = float(adx.iloc[-1]) if not adx.empty else 0

    bb = ta.volatility.BollingerBands(df['close'], window=20)
    bb_width = float(((bb.bollinger_hband() - bb.bollinger_lband()) / bb.bollinger_mavg()).iloc[-1])

    atr = ta.volatility.average_true_range(df['high'], df['low'], df['close'], window=14)
    atr_pct = float(atr.iloc[-1] / df['close'].iloc[-1])

    # Detect regime
    if adx_val > 40:
        regime = 'STRONG TREND'
        color = POSITIVE
        desc = 'Powerful directional move. Trend-following strategies optimal.'
    elif adx_val > 25:
        regime = 'TRENDING'
        color = NEON_BLUE
        desc = 'Clear directional bias. EMAs and MACD reliable.'
    elif bb_width < 0.03:
        regime = 'COMPRESSION'
        color = NEON_PURPLE
        desc = 'Extreme low volatility. Breakout imminent. Watch for explosive move.'
    elif atr_pct > 0.05:
        regime = 'HIGH VOLATILITY'
        color = NEGATIVE
        desc = 'Choppy conditions. Reduce position size. Wide stops needed.'
    elif adx_val < 15:
        regime = 'RANGING'
        color = WARNING
        desc = 'No trend. Mean-reversion strategies may work. Avoid breakout trades.'
    else:
        regime = 'TRANSITIONING'
        color = TEXT_MUTED
        desc = 'Market shifting between regimes. Wait for confirmation.'

    return {
        'regime': regime,
        'color': color,
        'description': desc,
        'adx': adx_val,
        'bb_width': bb_width,
        'atr_pct': atr_pct * 100,
    }


@st.cache_data(ttl=300, show_spinner=False)
def fetch_trending_coins() -> list[dict]:
    """Fetch trending coins from CoinGecko for whale/momentum tracking."""
    try:
        resp = requests.get("https://api.coingecko.com/api/v3/search/trending", timeout=10)
        if resp.status_code == 200:
            data = resp.json()
            coins = []
            for item in data.get('coins', [])[:15]:
                c = item.get('item', {})
                coins.append({
                    'name': c.get('name', ''),
                    'symbol': c.get('symbol', '').upper(),
                    'market_cap_rank': c.get('market_cap_rank', 0),
                    'price_btc': c.get('price_btc', 0),
                    'score': c.get('score', 0),
                })
            return coins
    except Exception:
        pass
    return []


@st.cache_data(ttl=120, show_spinner=False)
def fetch_top_gainers_losers(limit: int = 20) -> tuple[list, list]:
    """Fetch top gainers and losers from CoinGecko."""
    try:
        url = "https://api.coingecko.com/api/v3/coins/markets"
        params = {
            "vs_currency": "usd", "order": "market_cap_desc",
            "per_page": 250, "page": 1, "sparkline": False,
            "price_change_percentage": "24h",
        }
        resp = requests.get(url, params=params, timeout=15)
        if resp.status_code == 200:
            data = resp.json()
            valid = [c for c in data if c.get('price_change_percentage_24h') is not None]
            sorted_coins = sorted(valid, key=lambda x: x.get('price_change_percentage_24h', 0), reverse=True)
            gainers = sorted_coins[:limit]
            losers = sorted_coins[-limit:][::-1]
            return gainers, losers
    except Exception:
        pass
    return [], []


def get_top_volume_usdt_symbols(top_n: int = 100, vs_currency: str = "usd"):
    """Obtain top USDT trading pairs from CoinGecko and filter by exchange markets."""
    try:
        url = "https://api.coingecko.com/api/v3/coins/markets"
        params = {
            "vs_currency": vs_currency,
            "order": "volume_desc",
            "per_page": min(top_n, 250),
            "page": 1,
            "sparkline": False,
        }
        resp = requests.get(url, params=params, timeout=10)
        if resp.status_code != 200:
            _debug(f"CoinGecko error: {resp.status_code} {resp.text}")
            return [], []
        data = resp.json()
        if not isinstance(data, list):
            _debug(f"CoinGecko invalid data type: {type(data)}")
            return [], []

        markets = MARKETS
        valid = []
        seen = set()

        for coin in data:
            symbol = (coin.get("symbol") or "").upper()
            if not symbol or symbol in seen:
                continue
            seen.add(symbol)
            for quote in ("USDT", "USD"):
                pair = f"{symbol}/{quote}"
                if pair in markets:
                    valid.append(pair)
                    break

        return valid, data
    except Exception as e:
        _debug(f"get_top_volume_usdt_symbols error: {e}")
        return [], []


def render_market_tab():
    """Render the Market Dashboard tab containing top‑level crypto metrics and scanning."""

    # Fetch global market data
    # Unpack market indices.  The function returns BTC/ETH dominance, market caps,
    # 24h change and dominance values for BNB, SOL, ADA and XRP.  We keep the
    # additional dominance values for use in the AI market outlook calculation.
    btc_dom, eth_dom, total_mcap, alt_mcap, mcap_24h_pct, bnb_dom, sol_dom, ada_dom, xrp_dom = get_market_indices()
    fg_value, fg_label = get_fear_greed()
    fg_value = fg_value if fg_value is not None else 0
    btc_price, eth_price = get_btc_eth_prices()
    btc_price = btc_price or 0
    eth_price = eth_price or 0

    # Compute percentage change for market cap
    delta_mcap = mcap_24h_pct

    # Compute price change percentages using ccxt
    btc_change = get_price_change("BTC/USDT")
    eth_change = get_price_change("ETH/USDT")

    # Display headline and subtitle
    st.markdown("<h1 class='title'>Crypto Command Center</h1>", unsafe_allow_html=True)
    st.markdown(
        f"<p style='color:{TEXT_MUTED}; font-size:0.94rem;'>"
        "Live metrics for BTC, ETH and the broader market. "
        "Top coins are dynamically selected based on 24h volume rankings from CoinGecko, "
        "and filtered to include only USDT pairs actively traded on the exchange. "
        "Each coin is scored based on real-time technical signals."
        "</p>",
        unsafe_allow_html=True,
    )

    # Determine which timeframe to use for the market prediction.  We rely on
    # Streamlit session state to persist the selected timeframe from the
    # scanner controls.  On the first render, default to 1h.  This allows
    # the market prediction card to update automatically when the user
    # changes the timeframe in the scanner below.  We fetch BTC/USDT as a
    # proxy for the overall crypto market and compute a prediction using
    # 500 candles of history.
    selected_timeframe = st.session_state.get("market_timeframe", "1h")
    # Top row: Price and market cap metrics.
    m1, m2, m3, m4 = st.columns(4, gap="medium")
    # Bitcoin price
    with m1:
        delta_class = "metric-delta-positive" if (btc_change or 0) >= 0 else "metric-delta-negative"
        delta_text = f"({btc_change:+.2f}%)" if btc_change is not None else ""
        st.markdown(
            f"<div class='metric-card'>"
            f"  <div class='metric-label'>Bitcoin Price</div>"
            f"  <div class='metric-value'>${btc_price:,.2f}</div>"
            f"  <div class='{delta_class}'>{delta_text}</div>"
            f"</div>",
            unsafe_allow_html=True,
        )
    # Ethereum price
    with m2:
        delta_class = "metric-delta-positive" if (eth_change or 0) >= 0 else "metric-delta-negative"
        delta_text = f"({eth_change:+.2f}%)" if eth_change is not None else ""
        st.markdown(
            f"<div class='metric-card'>"
            f"  <div class='metric-label'>Ethereum Price</div>"
            f"  <div class='metric-value'>${eth_price:,.2f}</div>"
            f"  <div class='{delta_class}'>{delta_text}</div>"
            f"</div>",
            unsafe_allow_html=True,
        )
    # Total market cap
    with m3:
        delta_class = "metric-delta-positive" if delta_mcap >= 0 else "metric-delta-negative"
        st.markdown(
            f"<div class='metric-card'>"
            f"  <div class='metric-label'>Total Market Cap</div>"
            f"  <div class='metric-value'>${total_mcap / 1e12:.2f}T</div>"
            f"  <div class='{delta_class}'>({delta_mcap:+.2f}%)</div>"
            f"</div>",
            unsafe_allow_html=True,
        )
    # Fear & Greed index
    with m4:
        sentiment_color = POSITIVE if "Greed" in fg_label else (NEGATIVE if "Fear" in fg_label else WARNING)
        st.markdown(
            f"<div class='metric-card'>"
            f"  <div class='metric-label'>Fear &amp; Greed "
            f"<span title='Crypto Fear &amp; Greed Index (0-100). "
            f"0-25 = Extreme Fear (potential buy zone), 75-100 = Extreme Greed (potential sell zone).' "
            f"style='cursor:help; font-size:0.7rem;'>ℹ️</span></div>"
            f"  <div class='metric-value'>{fg_value}</div>"
            f"  <div style='color:{sentiment_color};font-size:0.9rem;'>{fg_label}</div>"
            f"</div>",
            unsafe_allow_html=True,
        )
    # Second row: Dominance gauges and AI market outlook
    # Compute AI market outlook using a dominance-weighted ML prediction across
    # BTC, ETH and major altcoins (BNB, SOL, ADA, XRP) on the selected timeframe.
    try:
        bundle_behav = get_major_ohlcv_bundle(selected_timeframe, limit=500)
        btc_df_behav = bundle_behav.get('BTC/USDT')
        eth_df_behav = bundle_behav.get('ETH/USDT')
        bnb_df_behav = bundle_behav.get('BNB/USDT')
        sol_df_behav = bundle_behav.get('SOL/USDT')
        ada_df_behav = bundle_behav.get('ADA/USDT')
        xrp_df_behav = bundle_behav.get('XRP/USDT')
        # Initialise probabilities at a neutral value of 0.5.  If data
        # retrieval or training fails for an asset, the neutral prior will
        # prevent it from skewing the combined outlook.
        btc_prob = eth_prob = bnb_prob = sol_prob = ada_prob = xrp_prob = 0.5
        if btc_df_behav is not None and not btc_df_behav.empty:
            btc_prob, _ = ml_predict_direction(btc_df_behav)
        if eth_df_behav is not None and not eth_df_behav.empty:
            eth_prob, _ = ml_predict_direction(eth_df_behav)
        if bnb_df_behav is not None and not bnb_df_behav.empty:
            bnb_prob, _ = ml_predict_direction(bnb_df_behav)
        if sol_df_behav is not None and not sol_df_behav.empty:
            sol_prob, _ = ml_predict_direction(sol_df_behav)
        if ada_df_behav is not None and not ada_df_behav.empty:
            ada_prob, _ = ml_predict_direction(ada_df_behav)
        if xrp_df_behav is not None and not xrp_df_behav.empty:
            xrp_prob, _ = ml_predict_direction(xrp_df_behav)
        # Compute a weighted probability across all assets.  Dominance values
        # reflect each coin's share of the total crypto market.  If the sum of
        # dominances is zero (unlikely), default to 1 to avoid division by zero.
        dom_sum = btc_dom + eth_dom + bnb_dom + sol_dom + ada_dom + xrp_dom
        dom_sum = dom_sum if dom_sum > 0 else 1.0
        behaviour_prob = (
            btc_prob * btc_dom
            + eth_prob * eth_dom
            + bnb_prob * bnb_dom
            + sol_prob * sol_dom
            + ada_prob * ada_dom
            + xrp_prob * xrp_dom
        ) / dom_sum
    except Exception:
        behaviour_prob = 0.5
    # Determine behaviour direction from the combined probability
    if behaviour_prob >= 0.6:
        behaviour_dir = "LONG"
    elif behaviour_prob <= 0.4:
        behaviour_dir = "SHORT"
    else:
        behaviour_dir = "NEUTRAL"
    # Map behaviour direction to a label for display and choose colour.  We
    # reuse the POSITIVE/NEGATIVE/WARNING colours defined above.
    if behaviour_dir == "LONG":
        behaviour_label = "Up"
        behaviour_color = POSITIVE
    elif behaviour_dir == "SHORT":
        behaviour_label = "Down"
        behaviour_color = NEGATIVE
    else:
        behaviour_label = "Neutral"
        behaviour_color = WARNING

    g1, g2, g3 = st.columns(3, gap="medium")
    # BTC dominance gauge
    with g1:
        fig_btc = go.Figure(go.Indicator(
            mode="gauge+number",
            value=btc_dom,
            gauge={
                'axis': {'range': [0, 100], 'tickwidth': 1},
                'bar': {'color': ACCENT},
                'bgcolor': CARD_BG,
                'steps': [
                    {'range': [0, 40], 'color': NEGATIVE},
                    {'range': [40, 60], 'color': WARNING},
                    {'range': [60, 100], 'color': POSITIVE},
                ],
            },
            title={'text': 'BTC Dominance (%)', 'font': {'size': 16, 'color': ACCENT}},
            number={'font': {'color': ACCENT, 'size': 38}},
        ))
        fig_btc.update_layout(
            height=170,
            margin=dict(l=10, r=10, t=40, b=15),
            plot_bgcolor="#0e1117",
            paper_bgcolor="#0e1117",
        )
        st.plotly_chart(fig_btc, width="stretch")

    # ETH dominance gauge
    with g2:
        fig_eth = go.Figure(go.Indicator(
            mode="gauge+number",
            value=eth_dom,
            gauge={
                'axis': {'range': [0, 100], 'tickwidth': 1},
                'bar': {'color': ACCENT},
                'bgcolor': CARD_BG,
                'steps': [
                    {'range': [0, 15], 'color': NEGATIVE},
                    {'range': [15, 25], 'color': WARNING},
                    {'range': [25, 100], 'color': POSITIVE},
                ],
            },
            title={'text': 'ETH Dominance (%)', 'font': {'size': 16, 'color': ACCENT}},
            number={'font': {'color': ACCENT, 'size': 38}},
        ))
        fig_eth.update_layout(
            height=170,
            margin=dict(l=10, r=10, t=40, b=15),
            plot_bgcolor="#0e1117",
            paper_bgcolor="#0e1117",
        )
        st.plotly_chart(fig_eth, width="stretch")

    # AI market outlook gauge
    with g3:
        fig_behaviour = go.Figure(go.Indicator(
            mode="gauge+number",
            value=int(round(behaviour_prob * 100)),
            gauge={
                'axis': {'range': [0, 100], 'tickwidth': 1},
                'bar': {'color': ACCENT},
                'bgcolor': CARD_BG,
                'steps': [
                    {'range': [0, 40], 'color': NEGATIVE},
                    {'range': [40, 60], 'color': WARNING},
                    {'range': [60, 100], 'color': POSITIVE},
                ],
            },
            title={'text': 'AI Market Outlook (%)', 'font': {'size': 16, 'color': ACCENT}},
            number={'font': {'color': ACCENT, 'size': 38}},
        ))
        fig_behaviour.update_layout(
            height=170,
            margin=dict(l=10, r=10, t=40, b=15),
            plot_bgcolor="#0e1117",
            paper_bgcolor="#0e1117",
        )
        st.plotly_chart(fig_behaviour, width="stretch")
        # Direction label + tooltip explanation
        st.markdown(
            f"<div style='text-align:center; margin-top:-12px;'>"
            f"<span style='color:{behaviour_color}; font-size:0.9rem;'>{behaviour_label}</span>"
            f"<span title='Dominance-weighted ML prediction across BTC, ETH, BNB, SOL, ADA, XRP "
            f"on the selected timeframe. Each coin&apos;s prediction is weighted by its market dominance. "
            f"&gt;60% = Bullish, &lt;40% = Bearish, 40-60% = Neutral.' "
            f"style='cursor:help; color:{TEXT_MUTED}; font-size:0.8rem; margin-left:6px;'>ℹ️</span>"
            f"</div>",
            unsafe_allow_html=True,
        )


    # Divider
    st.markdown("\n\n")

    # Top coin scanner controls
    st.markdown(
        f"<h2 style='color:{ACCENT};margin-bottom:0.5rem;'>Coin Signal Scanner</h2>",
        unsafe_allow_html=True,
    )

    controls = st.columns([1.5, 1.5, 1, 1], gap="medium")
    with controls[0]:
        # Persist the selected timeframe in session state so the market
        # prediction card updates when this value changes.  The key ensures
        # the selection is stored under 'market_timeframe'.
        timeframe = st.selectbox(
            "Select timeframe",
            ['5m', '15m', '1h', '4h', '1d'],
            index=2,
            key="market_timeframe"
        )
    with controls[1]:
        signal_filter = st.selectbox("Signal", ['LONG', 'SHORT', 'BOTH'], index=2)
    with controls[2]:
        top_n = st.slider("Top N", min_value=3, max_value=50, value=50)
    # Strict scalp mode is always enabled (non-strict path removed).


    # Fetch top coins
    with st.spinner(f"Scanning {top_n} coins ({signal_filter}) [{timeframe}] ..."):
        usdt_symbols, market_data = get_top_volume_usdt_symbols(max(top_n, 50))
    
        # skip "wrapped"
        seen_symbols = set()
        unique_market_data = []
        for coin in market_data:
            coin_id = (coin.get("id") or "").lower()
            symbol = (coin.get("symbol") or "").upper()
            if not symbol:
                continue
            if "wrapped" in coin_id:
                continue
            if symbol in seen_symbols:
                continue
            seen_symbols.add(symbol)
            unique_market_data.append(coin)
    
        # Market cap map
        mcap_map = {}
        for coin in unique_market_data:
            symbol = (coin.get("symbol") or "").upper()
            name = (coin.get("name") or "").upper()
            mcap = int(coin.get("market_cap") or 0)
            if symbol:
                if symbol not in mcap_map or mcap > mcap_map[symbol]:
                    mcap_map[symbol] = mcap
            if name:
                if name not in mcap_map or mcap > mcap_map[name]:
                    mcap_map[name] = mcap
    
        # USDT match
        valid_bases = {(c.get("symbol") or "").upper() for c in unique_market_data}
        working_symbols = [s for s in usdt_symbols if s.split("/")[0].upper() in valid_bases]
        working_symbols = working_symbols[:top_n]
    
        # Analysis — parallelised data fetching for speed
        def _scan_one(sym: str) -> dict | None:
            """Analyse a single symbol for the scanner. Returns a row dict or None."""
            df = fetch_ohlcv(sym, timeframe, limit=500)
            if df is None or len(df) <= 30:
                return None

            df['ema9'] = ta.trend.ema_indicator(df['close'], window=9)
            df['ema21'] = ta.trend.ema_indicator(df['close'], window=21)
            df['rsi'] = ta.momentum.rsi(df['close'], window=14)

            prob_up, ai_direction = ml_predict_direction(df)
            latest = df.iloc[-1]

            base = sym.split('/')[0].upper()
            mcap_val = mcap_map.get(base, 0)
            price = float(latest['close'])
            price_change = get_price_change(sym)

            a = analyse(df)
            signal, lev, volume_spike = a.signal, a.leverage, a.volume_spike
            atr_comment_v, candle_pattern_v, confidence_score_v = a.atr_comment, a.candle_pattern, a.confidence
            adx_val_v, supertrend_trend_v, ichimoku_trend_v = a.adx, a.supertrend, a.ichimoku
            stochrsi_k_val_v, bollinger_bias_v, vwap_label_v = a.stochrsi_k, a.bollinger, a.vwap
            psar_trend_v = a.psar

            scalp_direction, entry_s, target_s, stop_s, rr_ratio, breakout_note = get_scalping_entry_target(
                df, confidence_score_v, supertrend_trend_v, ichimoku_trend_v, vwap_label_v,
                volume_spike, strict_mode=True,
            )
            entry_price = entry_s if scalp_direction else 0.0
            target_price = target_s if scalp_direction else 0.0

            include = (
                (signal_filter == 'BOTH') or
                (signal_filter == 'LONG' and signal in ['STRONG BUY', 'BUY']) or
                (signal_filter == 'SHORT' and signal in ['STRONG SELL', 'SELL'])
            )
            if not include:
                return None

            signal_direction = "LONG" if signal in ['STRONG BUY', 'BUY'] else ("SHORT" if signal in ['STRONG SELL', 'SELL'] else "NEUTRAL")

            if signal_direction == "LONG" and ai_direction == "SHORT":
                ai_display = "⚠️ SHORT (Divergence)"
            elif signal_direction == "SHORT" and ai_direction == "LONG":
                ai_display = "⚠️ LONG (Divergence)"
            elif signal_direction != "NEUTRAL" and ai_direction == "NEUTRAL":
                ai_display = f"{ai_direction} (Weak)"
            else:
                ai_display = ai_direction

            _conv_lbl, _conv_clr = _calc_conviction(signal_direction, ai_direction, confidence_score_v)
            _emoji_map = {"HIGH": "🟢", "MEDIUM": "🟡", "LOW": "⚪", "CONFLICT": "🔴"}
            conviction = f"{_emoji_map.get(_conv_lbl, '')} {_conv_lbl}" if _conv_lbl else ""

            return {
                'Coin': base,
                'Price ($)': f"{price:,.2f}",
                'Signal': signal_plain(signal),
                'Confidence': confidence_score_badge(confidence_score_v),
                'AI Prediction': ai_display,
                'Conviction': conviction,
                'Market Cap ($)': readable_market_cap(mcap_val),
                'Leverage': leverage_badge(lev) if scalp_direction else '',
                'Scalp Opportunity': scalp_direction or "",
                'Entry Price': f"${entry_price:,.2f}" if entry_price else '',
                'Target Price': f"${target_price:,.2f}" if target_price else '',
                'Δ (%)': format_delta(price_change) if price_change is not None else '',
                'Spike Alert': '▲ Spike' if volume_spike else '',
                'ADX': round(adx_val_v, 1),
                'SuperTrend': supertrend_trend_v,
                'Volatility': atr_comment_v,
                'Stochastic RSI': round(stochrsi_k_val_v, 2),
                'Candle Pattern': candle_pattern_v,
                'Ichimoku': ichimoku_trend_v,
                'Bollinger': bollinger_bias_v,
                'VWAP': vwap_label_v,
                'PSAR': psar_trend_v if psar_trend_v != "Unavailable" else '',
                'Williams %R': a.williams,
                'CCI': a.cci,
                '__confidence_val': confidence_score_v,
            }

        # Parallel scan using ThreadPoolExecutor (5-10x faster than sequential)
        results: list[dict] = []
        with ThreadPoolExecutor(max_workers=6) as executor:
            futures = {executor.submit(_scan_one, sym): sym for sym in working_symbols}
            for future in as_completed(futures):
                try:
                    row = future.result()
                    if row is not None:
                        results.append(row)
                except Exception as e:
                    _debug(f"Scanner error for {futures[future]}: {e}")

        # Sort results by confidence score (descending)
        results = sorted(results, key=lambda x: x['__confidence_val'], reverse=True)
    
        # Limit to top_n results
        results = results[:top_n]
    
        # Prepare DataFrame for display
        if results:
            # Column legend
            st.markdown(
                f"<details style='margin-bottom:0.8rem;'>"
                f"<summary style='color:{ACCENT}; cursor:pointer; font-size:0.9rem;'>"
                f"ℹ️ Column Guide (click to expand)</summary>"
                f"<div style='color:{TEXT_MUTED}; font-size:0.84rem; line-height:1.7; padding:0.5rem;'>"
                "<b>Signal</b> — Technical analysis direction (LONG/SHORT/WAIT) based on EMA, RSI, MACD, "
                "SuperTrend, Ichimoku, Bollinger, ADX, and more. LONG = all indicators agree bullish.<br>"
                "<b>Confidence</b> — How strongly the indicators agree (0-100%). Higher = more indicators "
                "are aligned in the same direction. &ge;75% is strong.<br>"
                "<b>AI Prediction</b> — Machine learning model (Gradient Boosting) trained on recent candles "
                "to predict the next candle's direction. Independent from the Signal.<br>"
                "<b>Conviction</b> — Combined alignment check: "
                "🟢 HIGH = Signal + AI agree &amp; confidence &ge;75%, "
                "🟡 MEDIUM = agree &amp; confidence 60-75%, "
                "⚪ LOW = weak agreement, "
                "🔴 CONFLICT = Signal and AI disagree. "
                "<b>Only trade 🟢 HIGH conviction setups with confidence.</b>"
                "</div></details>",
                unsafe_allow_html=True,
            )

            df_results = pd.DataFrame(results)

            # Trend and ADX column visual
            df_results["SuperTrend"] = df_results["SuperTrend"].apply(format_trend)
            df_results["ADX"] = df_results["ADX"].apply(format_adx)
            df_results["Ichimoku"] = df_results["Ichimoku"].apply(format_trend)
            df_results["Stochastic RSI"] = df_results["Stochastic RSI"].apply(format_stochrsi)

            # '__confidence_val' hide
            df_display = df_results.drop(columns=['__confidence_val'])

            # Style
            styled = (
                df_display.style
                .map(style_signal, subset=['Signal', 'AI Prediction'])
                .map(style_confidence, subset=['Confidence'])
                .map(style_scalp_opp, subset=['Scalp Opportunity'])
                .map(style_delta, subset=['Δ (%)'])
            )
            st.dataframe(styled, width="stretch")

            # Provide option to download the scanning results as a CSV.  Users
            # can further analyse the output or archive it.  We drop the
            # formatting columns and use a simple CSV export.
            csv_market = df_results.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="Download Scan Results (CSV)",
                data=csv_market,
                file_name="scan_results.csv",
                mime="text/csv"
            )
        else:
            st.info("No coins matched the criteria.")



def render_spot_tab():
    """Render the Spot Trading tab which allows instant analysis of a selected coin."""
    st.markdown(
        f"<h2 style='color:{ACCENT};margin-bottom:0.5rem;'>Spot Trading</h2>",
        unsafe_allow_html=True,
    )
    coin = _normalize_coin_input(st.text_input(
        "Coin (e.g. BTC, ETH, TAO)",
        value="BTC",
        key="spot_coin_input",
    ))
    timeframe = st.selectbox("Timeframe", ['1m', '3m', '5m', '15m', '1h', '4h', '1d'], index=4)
    if st.button("Analyse", type="primary"):
        _val_err = _validate_coin_symbol(coin)
        if _val_err:
            st.error(_val_err)
            return
        df = fetch_ohlcv(coin, timeframe)
        if df is None or len(df) < 30:
            st.error(f"Could not fetch data for **{coin}** on {timeframe}. The coin may not be listed on supported exchanges. Try a major pair (BTC, ETH) or check the symbol.")
            return
        a = analyse(df)
        signal, lev, comment, volume_spike = a.signal, a.leverage, a.comment, a.volume_spike
        atr_comment, candle_pattern, confidence_score = a.atr_comment, a.candle_pattern, a.confidence
        adx_val, supertrend_trend, ichimoku_trend = a.adx, a.supertrend, a.ichimoku
        stochrsi_k_val, bollinger_bias, vwap_label = a.stochrsi_k, a.bollinger, a.vwap
        psar_trend, williams_label, cci_label = a.psar, a.williams, a.cci

        current_price = df['close'].iloc[-1]

        # Display summary grid
        signal_clean = signal_plain(signal)
        try:
            _ai_prob_s, ai_dir_s = ml_predict_direction(df)
        except Exception:
            ai_dir_s = "NEUTRAL"

        sig_dir_s = "LONG" if signal in ['STRONG BUY', 'BUY'] else ("SHORT" if signal in ['STRONG SELL', 'SELL'] else "WAIT")
        conv_lbl_s, conv_c_s = _calc_conviction(sig_dir_s, ai_dir_s, confidence_score)

        sig_c_s = POSITIVE if "LONG" in signal_clean else (NEGATIVE if "SHORT" in signal_clean else WARNING)
        ai_c_s = POSITIVE if ai_dir_s == "LONG" else (NEGATIVE if ai_dir_s == "SHORT" else WARNING)
        conf_c_s = POSITIVE if confidence_score >= 70 else (WARNING if confidence_score >= 50 else NEGATIVE)
        st.markdown(
            f"<div style='display:grid; grid-template-columns:repeat(4, 1fr); "
            f"gap:4px; background:{CARD_BG}; border-radius:8px; padding:10px; margin:8px 0;'>"
            f"<div style='text-align:center; padding:6px;'>"
            f"<div style='color:{TEXT_MUTED}; font-size:0.7rem; text-transform:uppercase;'>Signal</div>"
            f"<div style='color:{sig_c_s}; font-size:0.85rem; font-weight:600;'>{signal_clean}</div></div>"
            f"<div style='text-align:center; padding:6px;'>"
            f"<div style='color:{TEXT_MUTED}; font-size:0.7rem; text-transform:uppercase;'>Confidence</div>"
            f"<div style='color:{conf_c_s}; font-size:0.85rem; font-weight:600;'>{confidence_score:.0f}%</div></div>"
            f"<div style='text-align:center; padding:6px;'>"
            f"<div style='color:{TEXT_MUTED}; font-size:0.7rem; text-transform:uppercase;'>AI Prediction</div>"
            f"<div style='color:{ai_c_s}; font-size:0.85rem; font-weight:600;'>{ai_dir_s}</div></div>"
            f"<div style='text-align:center; padding:6px;'>"
            f"<div style='color:{TEXT_MUTED}; font-size:0.7rem; text-transform:uppercase;'>Conviction</div>"
            f"<div style='color:{conv_c_s}; font-size:0.85rem; font-weight:600;'>{conv_lbl_s}</div></div>"
            f"</div>",
            unsafe_allow_html=True,
        )

        st.markdown(f"<p style='color:{TEXT_MUTED}; font-size:0.88rem;'>{comment}</p>", unsafe_allow_html=True)

        # Market regime warning (ADX < 20 = ranging market)
        if not np.isnan(adx_val) and adx_val < 20:
            st.markdown(
                f"<div style='background:#2D1B00; border-left:4px solid {WARNING}; "
                f"padding:8px 12px; border-radius:4px; margin:6px 0;'>"
                f"<span style='color:{WARNING}; font-weight:600;'>Market Ranging (ADX {adx_val:.0f})</span>"
                f"<span style='color:{TEXT_MUTED}; font-size:0.82rem;'> — Trend signals are less reliable. "
                f"Confidence discounted. Consider smaller positions or waiting for a clear trend.</span></div>",
                unsafe_allow_html=True,
            )

        # Risk alert: high leverage suggestion + low confidence
        if lev >= 6 and confidence_score < 55:
            st.markdown(
                f"<div style='background:#2D0A0A; border-left:4px solid {NEGATIVE}; "
                f"padding:8px 12px; border-radius:4px; margin:6px 0;'>"
                f"<span style='color:{NEGATIVE}; font-weight:600;'>Risk Warning</span>"
                f"<span style='color:{TEXT_MUTED}; font-size:0.82rem;'> — Leverage x{lev} with only "
                f"{confidence_score:.0f}% confidence is dangerous. Reduce position size or wait for higher conviction.</span></div>",
                unsafe_allow_html=True,
            )

        # Indicator grid (professional card layout)
        _grid_html = _build_indicator_grid(
            supertrend_trend, ichimoku_trend, vwap_label, adx_val, bollinger_bias,
            stochrsi_k_val, psar_trend, williams_label, cci_label,
            volume_spike, atr_comment, candle_pattern,
        )
        if _grid_html:
            st.markdown(_grid_html, unsafe_allow_html=True,
            )

        # Price box
        st.markdown(f"<div class='metric-card'><div class='metric-label'>Current Price</div><div class='metric-value'>${current_price:,.2f}</div></div>", unsafe_allow_html=True)


        sent_score, sent_label = get_social_sentiment(coin)
        gauge_sent = go.Figure(go.Indicator(
            mode="gauge+number",
            value=sent_score,
            gauge={
                'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': TEXT_MUTED},
                'bar': {'color': ACCENT},
                'bgcolor': PRIMARY_BG,
                'steps': [
                    {'range': [0, 25], 'color': NEGATIVE},
                    {'range': [25, 45], 'color': WARNING},
                    {'range': [45, 55], 'color': TEXT_MUTED},
                    {'range': [55, 75], 'color': POSITIVE},
                    {'range': [75, 100], 'color': POSITIVE},
                ],
            },
            title={'text': f"Sentiment ({sent_label})", 'font': {'size': 16, 'color': ACCENT}},
            number={'font': {'color': TEXT_LIGHT, 'size': 36}}
        ))
        gauge_sent.update_layout(
            height=170,
            margin=dict(l=10, r=10, t=40, b=15),
            template='plotly_dark',
            paper_bgcolor=PRIMARY_BG
        )
        st.plotly_chart(gauge_sent, width="stretch")
        # Plot candlestick with EMAs
        fig = go.Figure()
        fig.add_trace(go.Candlestick(
            x=df['timestamp'], open=df['open'], high=df['high'], low=df['low'], close=df['close'],
            increasing_line_color=POSITIVE, decreasing_line_color=NEGATIVE, name="Price"
        ))
        # Plot EMAs
        for window, color in [(5, '#F472B6'), (9, '#60A5FA'), (21, '#FBBF24'), (50, '#FCD34D')]:
            ema_series = ta.trend.ema_indicator(df['close'], window=window)
            fig.add_trace(go.Scatter(x=df['timestamp'], y=ema_series, mode='lines',
                                     name=f"EMA{window}", line=dict(color=color, width=1.5)))
        # Plot weighted moving averages (WMA) for additional insight.  The WMA gives
        # more weight to recent prices and can help identify trend shifts earlier.
        try:
            wma20 = _wma(df['close'], length=20)
            wma50 = _wma(df['close'], length=50)
            fig.add_trace(go.Scatter(x=df['timestamp'], y=wma20, mode='lines',
                                     name="WMA20", line=dict(color='#34D399', width=1, dash='dot')))
            fig.add_trace(go.Scatter(x=df['timestamp'], y=wma50, mode='lines',
                                     name="WMA50", line=dict(color='#10B981', width=1, dash='dash')))
        except Exception as e:
            _debug(f"WMA chart overlay error: {e}")
        # Place legend at top left for candlestick chart
        fig.update_layout(
            height=380,
            template='plotly_dark',
            margin=dict(l=20, r=20, t=30, b=30),
            xaxis_rangeslider_visible=False,
            showlegend=True,
            legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='left', x=0)
        )
        st.plotly_chart(fig, width="stretch")
        # RSI chart
        rsi_fig = go.Figure()
        for period, color in [(6, '#D8B4FE'), (14, '#A78BFA'), (24, '#818CF8')]:
            rsi_series = ta.momentum.rsi(df['close'], window=period)
            rsi_fig.add_trace(go.Scatter(
                x=df['timestamp'], y=rsi_series, mode='lines', name=f"RSI {period}",
                line=dict(color=color, width=2)
            ))
        # Add overbought/oversold bands
        rsi_fig.add_hline(y=70, line=dict(color=NEGATIVE, dash='dot', width=1), name="Overbought")
        rsi_fig.add_hline(y=30, line=dict(color=POSITIVE, dash='dot', width=1), name="Oversold")
        rsi_fig.update_layout(
            height=180,
            template='plotly_dark',
            margin=dict(l=20, r=20, t=20, b=30),
            yaxis=dict(title="RSI"),
            showlegend=True,
            legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='left', x=0)
        )
        st.plotly_chart(rsi_fig, width="stretch")

        # MACD chart
        macd_ind = ta.trend.MACD(df['close'])
        df['macd'] = macd_ind.macd()
        df['macd_signal'] = macd_ind.macd_signal()
        df['macd_diff'] = macd_ind.macd_diff()
        macd_fig = go.Figure()
        macd_fig.add_trace(go.Scatter(
            x=df['timestamp'], y=df['macd'], name="MACD",
            line=dict(color=ACCENT, width=2)
        ))
        macd_fig.add_trace(go.Scatter(
            x=df['timestamp'], y=df['macd_signal'], name="Signal",
            line=dict(color=WARNING, width=2, dash='dot')
        ))
        macd_fig.add_trace(go.Bar(
            x=df['timestamp'], y=df['macd_diff'], name="Histogram",
            marker_color=CARD_BG
        ))
        macd_fig.update_layout(
            height=200,
            template='plotly_dark',
            margin=dict(l=20, r=20, t=20, b=30),
            yaxis=dict(title="MACD"),
            showlegend=True,
            legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='left', x=0)
        )
        st.plotly_chart(macd_fig, width="stretch")

        # Volume & OBV chart
        df['obv'] = ta.volume.on_balance_volume(df['close'], df['volume'])
        volume_fig = go.Figure()
        volume_fig.add_trace(go.Bar(
            x=df['timestamp'], y=df['volume'], name="Volume", marker_color="#6B7280"
        ))
        volume_fig.add_trace(go.Scatter(
            x=df['timestamp'], y=df['obv'], name="OBV",
            line=dict(color=WARNING, width=1.5, dash='dot'),
            yaxis='y2'
        ))
        volume_fig.update_layout(
            height=180,
            template='plotly_dark',
            margin=dict(l=20, r=20, t=20, b=30),
            yaxis=dict(title="Volume"),
            yaxis2=dict(overlaying='y', side='right', title='OBV', showgrid=False),
            showlegend=True,
            legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='left', x=0)
        )
        st.plotly_chart(volume_fig, width="stretch")

        # Technical snapshot
        # Compute indicators for snapshot
        df['ema9'] = ta.trend.ema_indicator(df['close'], window=9)
        df['ema21'] = ta.trend.ema_indicator(df['close'], window=21)
        df['rsi14'] = ta.momentum.rsi(df['close'], window=14)
        latest = df.iloc[-1]
        ema9 = latest['ema9']
        ema21 = latest['ema21']
        macd_val = df['macd'].iloc[-1]
        rsi_val = latest['rsi14']
        _obv_back_pos = min(5, len(df) - 1)
        obv_change = ((df['obv'].iloc[-1] - df['obv'].iloc[-_obv_back_pos]) / abs(df['obv'].iloc[-_obv_back_pos]) * 100) if (_obv_back_pos > 0 and df['obv'].iloc[-_obv_back_pos] != 0) else 0
        recent = df.tail(_sr_lookback())
        support = recent['low'].min()
        resistance = recent['high'].max()
        current_price = latest['close']
        support_dist = abs(current_price - support) / current_price * 100
        resistance_dist = abs(current_price - resistance) / current_price * 100
        # Build snapshot HTML
        snapshot_html = f"""
        <div class='panel-box'>
          <b style='color:{ACCENT}; font-size:1.05rem;'>📊 Technical Snapshot</b><br>
          <ul style='color:{TEXT_MUTED}; font-size:0.9rem; line-height:1.5; list-style-position:inside; margin-top:6px;'>
            <li>EMA Trend (9 vs 21): <b>{ema9:.2f}</b> vs <b>{ema21:.2f}</b> {('🟢' if ema9 > ema21 else '🔴')} — When EMA9 is above EMA21 the short‑term trend is bullish; otherwise bearish.</li>
            <li>MACD: <b>{macd_val:.2f}</b> {('🟢' if macd_val > 0 else '🔴')} — Positive MACD indicates upward momentum; negative values suggest downward pressure.</li>
            <li>RSI (14): <b>{rsi_val:.2f}</b> {('🟢' if rsi_val > 55 else ('🟠' if 45 <= rsi_val <= 55 else '🔴'))} — Above 70 may signal overbought, below 30 oversold. Values above 50 favour bulls.</li>
            <li>OBV change (last 5 candles): <b>{obv_change:+.2f}%</b> {('🟢' if obv_change > 0 else '🔴')} — Rising OBV supports the price move; falling OBV warns against continuation.</li>
            <li>Support / Resistance: support at <b>${support:,.2f}</b> ({support_dist:.2f}% away), resistance at <b>${resistance:,.2f}</b> ({resistance_dist:.2f}% away).</li>
          </ul>
        </div>
        """
        st.markdown(snapshot_html, unsafe_allow_html=True)


def render_position_tab():
    """Render the Position Analyser tab for evaluating open positions."""
    st.markdown(
        f"<h2 style='color:{ACCENT};margin-bottom:0.5rem;'>Position Analyser</h2>",
        unsafe_allow_html=True,
    )
    # Assign a unique key to avoid StreamlitDuplicateElementId errors
    coin = _normalize_coin_input(st.text_input(
        "Coin (e.g. BTC, ETH, TAO)",
        value="BTC",
        key="position_coin_input",
    ))
    selected_timeframes = st.multiselect("Select up to 3 Timeframes", ['1m', '3m', '5m', '15m', '1h', '4h', '1d'], default=['3m'], max_selections=3)

    default_entry_price: float = 0.0
    for _v in _symbol_variants(coin):
        try:
            ticker = EXCHANGE.fetch_ticker(_v)
            default_entry_price = float(ticker.get('last', 0) or 0)
            break
        except Exception:
            continue

    entry_price = st.number_input("Entry Price", min_value=0.0, format="%.4f", value=default_entry_price)
    direction = st.selectbox("Position Direction", ["LONG", "SHORT"])

    # Strict scalp mode is always enabled (non-strict path removed).
   
    if st.button("Analyse Position", type="primary"):
        _val_err = _validate_coin_symbol(coin)
        if _val_err:
            st.error(_val_err)
            return
        tf_order = {'1m': 1, '3m': 2, '5m': 3, '15m': 4, '1h': 5, '4h': 6, '1d': 7}
        largest_tf = max(selected_timeframes, key=lambda tf: tf_order[tf])

        cols = st.columns(len(selected_timeframes))

        for idx, tf in enumerate(selected_timeframes):
            with cols[idx]:
                df = fetch_ohlcv(coin, tf, limit=200)
                if df is None or len(df) < 55:
                    st.error(f"Not enough data to analyse position for {tf}.")
                    continue

                a = analyse(df)
                signal, lev, comment, volume_spike = a.signal, a.leverage, a.comment, a.volume_spike
                atr_comment, candle_pattern, confidence_score = a.atr_comment, a.candle_pattern, a.confidence
                adx_val, supertrend_trend, ichimoku_trend = a.adx, a.supertrend, a.ichimoku
                stochrsi_k_val, bollinger_bias, vwap_label = a.stochrsi_k, a.bollinger, a.vwap
                psar_trend, williams_label, cci_label = a.psar, a.williams, a.cci

                current_price = df['close'].iloc[-1]
                pnl = entry_price - current_price if direction == "SHORT" else current_price - entry_price
                pnl_percent = (pnl / entry_price * 100) if entry_price else 0

                col = POSITIVE if pnl_percent > 0 else (WARNING if abs(pnl_percent) < 1 else NEGATIVE)
                icon = '🟢' if pnl_percent > 0 else ('🟠' if abs(pnl_percent) < 1 else '🔴')

                st.markdown(
                    f"<div class='panel-box' style='background-color:{col};color:{PRIMARY_BG};'>"
                    f"  {icon} <strong>{direction} Position ({tf})</strong><br>"
                    f"  Entry: ${entry_price:,.4f} | Current: ${current_price:,.4f} ({pnl_percent:+.2f}%)"
                    f"</div>",
                    unsafe_allow_html=True,
                )

                signal_clean = signal_plain(signal)

                # -- AI Prediction for this coin/timeframe --
                try:
                    _ai_prob, ai_dir = ml_predict_direction(df)
                except Exception:
                    ai_dir = "NEUTRAL"

                # Conviction: alignment of Signal + AI + Confidence
                sig_direction = "LONG" if signal in ['STRONG BUY', 'BUY'] else ("SHORT" if signal in ['STRONG SELL', 'SELL'] else "WAIT")
                conviction_lbl, conviction_c = _calc_conviction(sig_direction, ai_dir, confidence_score)

                # Signal / Confidence / AI / Conviction summary grid
                sig_color = POSITIVE if "LONG" in signal_clean else (NEGATIVE if "SHORT" in signal_clean else WARNING)
                ai_color = POSITIVE if ai_dir == "LONG" else (NEGATIVE if ai_dir == "SHORT" else WARNING)
                conf_color = POSITIVE if confidence_score >= 70 else (WARNING if confidence_score >= 50 else NEGATIVE)
                summary_row = (
                    f"<div style='text-align:center; padding:6px;'>"
                    f"<div style='color:{TEXT_MUTED}; font-size:0.7rem; text-transform:uppercase;'>Signal</div>"
                    f"<div style='color:{sig_color}; font-size:0.85rem; font-weight:600;'>{signal_clean}</div></div>"
                    f"<div style='text-align:center; padding:6px;'>"
                    f"<div style='color:{TEXT_MUTED}; font-size:0.7rem; text-transform:uppercase;'>Confidence</div>"
                    f"<div style='color:{conf_color}; font-size:0.85rem; font-weight:600;'>{confidence_score:.0f}%</div></div>"
                    f"<div style='text-align:center; padding:6px;'>"
                    f"<div style='color:{TEXT_MUTED}; font-size:0.7rem; text-transform:uppercase;'>AI Prediction</div>"
                    f"<div style='color:{ai_color}; font-size:0.85rem; font-weight:600;'>{ai_dir}</div></div>"
                    f"<div style='text-align:center; padding:6px;'>"
                    f"<div style='color:{TEXT_MUTED}; font-size:0.7rem; text-transform:uppercase;'>Conviction</div>"
                    f"<div style='color:{conviction_c}; font-size:0.85rem; font-weight:600;'>{conviction_lbl}</div></div>"
                )
                st.markdown(
                    f"<div style='display:grid; grid-template-columns:repeat(4, 1fr); "
                    f"gap:4px; background:{CARD_BG}; border-radius:8px; padding:10px; margin:8px 0;'>"
                    f"{summary_row}</div>",
                    unsafe_allow_html=True,
                )

                st.markdown(f"<p style='color:{TEXT_MUTED}; font-size:0.88rem;'>{comment}</p>", unsafe_allow_html=True)

                # Market regime warning (ADX < 20)
                if not np.isnan(adx_val) and adx_val < 20:
                    st.markdown(
                        f"<div style='background:#2D1B00; border-left:4px solid {WARNING}; "
                        f"padding:6px 10px; border-radius:4px; margin:4px 0; font-size:0.82rem;'>"
                        f"<span style='color:{WARNING}; font-weight:600;'>Ranging (ADX {adx_val:.0f})</span>"
                        f"<span style='color:{TEXT_MUTED};'> — Signals less reliable.</span></div>",
                        unsafe_allow_html=True,
                    )

                # Risk alert for position
                if direction == sig_direction and confidence_score < 50:
                    st.markdown(
                        f"<div style='background:#2D0A0A; border-left:4px solid {NEGATIVE}; "
                        f"padding:6px 10px; border-radius:4px; margin:4px 0; font-size:0.82rem;'>"
                        f"<span style='color:{NEGATIVE}; font-weight:600;'>Low Confidence</span>"
                        f"<span style='color:{TEXT_MUTED};'> — Position direction matches signal but confidence "
                        f"is only {confidence_score:.0f}%. Consider tightening stop-loss.</span></div>",
                        unsafe_allow_html=True,
                    )
                elif direction != sig_direction and sig_direction != "WAIT":
                    st.markdown(
                        f"<div style='background:#2D0A0A; border-left:4px solid {NEGATIVE}; "
                        f"padding:6px 10px; border-radius:4px; margin:4px 0; font-size:0.82rem;'>"
                        f"<span style='color:{NEGATIVE}; font-weight:600;'>Signal Conflict</span>"
                        f"<span style='color:{TEXT_MUTED};'> — Your {direction} position conflicts with "
                        f"the current {sig_direction} signal. Review position validity.</span></div>",
                        unsafe_allow_html=True,
                    )

                # -- Indicator grid (professional card layout) --
                _grid_html = _build_indicator_grid(
                    supertrend_trend, ichimoku_trend, vwap_label, adx_val, bollinger_bias,
                    stochrsi_k_val, psar_trend, williams_label, cci_label,
                    volume_spike, atr_comment, candle_pattern,
                )
                if _grid_html:
                    st.markdown(_grid_html, unsafe_allow_html=True)

                df['ema5'] = ta.trend.ema_indicator(df['close'], window=5)
                df['ema13'] = ta.trend.ema_indicator(df['close'], window=13)
                df['ema9'] = ta.trend.ema_indicator(df['close'], window=9)
                df['ema21'] = ta.trend.ema_indicator(df['close'], window=21)
                macd_ind = ta.trend.MACD(df['close'])
                df['macd'] = macd_ind.macd()
                df['macd_signal'] = macd_ind.macd_signal()
                df['macd_diff'] = macd_ind.macd_diff()
                df['rsi6'] = ta.momentum.rsi(df['close'], window=6)
                df['rsi14'] = ta.momentum.rsi(df['close'], window=14)
                df['rsi24'] = ta.momentum.rsi(df['close'], window=24)
                df['obv'] = ta.volume.on_balance_volume(df['close'], df['volume'])

                recent_sr = df.tail(_sr_lookback())
                support_sr = recent_sr['low'].min()
                resistance_sr = recent_sr['high'].max()
                suggestion = ""

                if direction == "LONG":
                    if current_price < support_sr:
                        suggestion = (
                            f"🔻 Price has broken below the key support at <b>${support_sr:,.4f}</b>.<br>"
                            f"This invalidates the bullish setup. <b>Consider closing the position (stop-out).</b>"
                        )
                    elif current_price < entry_price:
                        suggestion = (
                            f"⚠️ Price is trading below the entry level.<br>"
                            f"Monitor support at <b>${support_sr:,.4f}</b>. If it fails, risk increases significantly.<br>"
                            f"<i>Maintain caution unless support holds and momentum returns.</i>"
                        )
                    elif current_price < resistance_sr:
                        suggestion = (
                            f"📈 Price is above entry but below resistance at <b>${resistance_sr:,.4f}</b>.<br>"
                            f"<i>Consider holding the position. A breakout may offer further upside.</i>"
                        )
                    else:
                        suggestion = (
                            f"🟢 Price has broken above resistance at <b>${resistance_sr:,.4f}</b>.<br>"
                            f"<b>Consider taking partial profits or trailing your stop.</b>"
                        )
                else:
                    if current_price > resistance_sr:
                        suggestion = (
                            f"🔺 Price has broken above key resistance at <b>${resistance_sr:,.4f}</b>.<br>"
                            f"This invalidates the bearish case. <b>Consider closing the position (stop-out).</b>"
                        )
                    elif current_price > entry_price:
                        suggestion = (
                            f"⚠️ Price is above the short entry level.<br>"
                            f"Watch resistance at <b>${resistance_sr:,.4f}</b>. If it holds, the trade may still be valid.<br>"
                            f"<i>Remain cautious—trend may be reversing.</i>"
                        )
                    elif current_price > support_sr:
                        suggestion = (
                            f"📉 Price is below entry, approaching support at <b>${support_sr:,.4f}</b>.<br>"
                            f"<i>Consider holding. Breakdown of support could validate the short setup further.</i>"
                        )
                    else:
                        suggestion = (
                            f"🟢 Price has broken below support at <b>${support_sr:,.4f}</b>.<br>"
                            f"<b>Consider taking partial profits or holding to maximise gain.</b>"
                        )

                st.markdown(
                    f"<div class='panel-box'>"
                    f"  <b style='color:{ACCENT}; font-size:1.05rem;'>🧠 Strategy Suggestion ({tf})</b><br>"
                    f"  <p style='color:{TEXT_MUTED}; font-size:0.9rem; margin-top:6px;'>{suggestion}</p>"
                    f"</div>",
                    unsafe_allow_html=True
                )

                # === Scalping Setup ===
                df_scalp = fetch_ohlcv(coin, tf, limit=100)
                
                if df_scalp is not None and len(df_scalp) > 30:
                
                    # Technical calculations
                    df_scalp['ema5'] = df_scalp['close'].ewm(span=5).mean()
                    df_scalp['ema13'] = df_scalp['close'].ewm(span=13).mean()
                    df_scalp['ema21'] = df_scalp['close'].ewm(span=21).mean()
                    df_scalp['atr'] = ta.volatility.average_true_range(df_scalp['high'], df_scalp['low'], df_scalp['close'], window=14)
                    df_scalp['rsi'] = ta.momentum.rsi(df_scalp['close'], window=14)
                    _macd_scalp = ta.trend.MACD(df_scalp['close'])
                    df_scalp['macd'] = _macd_scalp.macd()
                    df_scalp['macd_signal'] = _macd_scalp.macd_signal()
                    df_scalp['macd_diff'] = _macd_scalp.macd_diff()
                    df_scalp['obv'] = ta.volume.on_balance_volume(df_scalp['close'], df_scalp['volume'])
                
                    latest = df_scalp.iloc[-1]
                    close_price = latest['close']
                    ema5_val = latest['ema5']
                    ema13_val = latest['ema13']
                    macd_hist_s = latest['macd_diff']
                    rsi14_val = latest['rsi']
                    _obv_back_scalp = min(5, len(df_scalp) - 1)
                    obv5 = df_scalp['obv'].iloc[-_obv_back_scalp] if _obv_back_scalp > 0 else df_scalp['obv'].iloc[-1]
                    obv_change_s = ((latest['obv'] - obv5) / abs(obv5) * 100) if obv5 != 0 else 0
                    _sr_scalp = _sr_lookback()
                    support_s = df_scalp['low'].tail(_sr_scalp).min()
                    resistance_s = df_scalp['high'].tail(_sr_scalp).max()
                    support_dist_s = abs(close_price - support_s) / close_price * 100
                    resistance_dist_s = abs(close_price - resistance_s) / close_price * 100
                
                    scalping_snapshot_html = f"""
                    <div class='panel-box'>
                      <b style='color:{ACCENT}; font-size:1.05rem;'>📊 Technical Snapshot (Scalping)</b><br>
                      <ul style='color:{TEXT_MUTED}; font-size:0.9rem; line-height:1.5; list-style-position:inside; margin-top:6px;'>
                        <li>EMA Trend (5 vs 13): <b>${ema5_val:,.2f}</b> vs <b>${ema13_val:,.2f}</b> {('🟢' if ema5_val > ema13_val else '🔴')}</li>
                        <li>MACD Histogram: <b>{macd_hist_s:.2f}</b> {('🟢' if macd_hist_s > 0 else '🔴')}</li>
                        <li>RSI (14): <b>{rsi14_val:.2f}</b> {('🟢' if rsi14_val > 50 else '🔴')}</li>
                        <li>OBV Change (last 5 candles): <b>{obv_change_s:+.2f}%</b> {('🟢' if obv_change_s > 0 else '🔴')}</li>
                        <li>Support / Resistance: support at <b>${support_s:,.4f}</b> ({support_dist_s:.2f}% away),
                             resistance at <b>${resistance_s:,.4f}</b> ({resistance_dist_s:.2f}% away)</li>
                      </ul>
                    </div>
                    """
                
                
                    # === Scalping Strategy Call ===
                    scalp_direction, entry_s, target_s, stop_s, rr_ratio, breakout_note = get_scalping_entry_target(
                        df_scalp,
                        confidence_score,
                        supertrend_trend,
                        ichimoku_trend,
                        vwap_label,
                        volume_spike,
                        strict_mode=True
                    )
                
                    # === Display Scalping Result ===
                    if scalp_direction:
                        color = POSITIVE if scalp_direction == "LONG" else NEGATIVE
                        icon = "🟢" if scalp_direction == "LONG" else "🔴"
                        st.markdown(
                            f"""
                            <div class='panel-box' style='background-color:{color};color:{PRIMARY_BG};'>
                              {icon} <b>Scalping {scalp_direction}</b><br>
                              Entry: <b>${entry_s:,.4f}</b><br>
                              Stop Loss: <b>${stop_s:,.4f}</b><br>
                              Target: <b>${target_s:,.4f}</b><br>
                              Risk/Reward: <b>{rr_ratio:.2f}</b> — {'✅ Good' if rr_ratio >= 1.5 else '⚠️ Too low (ideal ≥ 1.5)'}
                            </div>
                            """,
                            unsafe_allow_html=True
                        )
                    else:
                        msg = breakout_note or "No valid scalping setup with current filters."
                        st.info(msg)
                
                    st.markdown(scalping_snapshot_html, unsafe_allow_html=True)

        df_candle = fetch_ohlcv(coin, largest_tf, limit=100)
        if df_candle is not None and len(df_candle) >= 30:
            fig_candle = go.Figure()
            fig_candle.add_trace(go.Candlestick(
                x=df_candle['timestamp'], open=df_candle['open'], high=df_candle['high'],
                low=df_candle['low'], close=df_candle['close'],
                increasing_line_color=POSITIVE, decreasing_line_color=NEGATIVE, name="Price"
            ))
            # Plot EMAs
            for window, color in [(5, '#F472B6'), (9, '#60A5FA'), (13, '#A78BFA'), (21, '#FBBF24'), (50, '#FCD34D')]:
                ema_series = ta.trend.ema_indicator(df_candle['close'], window=window)
                fig_candle.add_trace(go.Scatter(x=df_candle['timestamp'], y=ema_series, mode='lines',
                                                name=f"EMA{window}", line=dict(color=color, width=1.5)))
            # Plot weighted moving averages (WMA) for deeper trend insight
            try:
                wma20_c = _wma(df_candle['close'], length=20)
                wma50_c = _wma(df_candle['close'], length=50)
                fig_candle.add_trace(go.Scatter(x=df_candle['timestamp'], y=wma20_c, mode='lines',
                                                name="WMA20", line=dict(color='#34D399', width=1, dash='dot')))
                fig_candle.add_trace(go.Scatter(x=df_candle['timestamp'], y=wma50_c, mode='lines',
                                                name="WMA50", line=dict(color='#10B981', width=1, dash='dash')))
            except Exception as e:
                _debug(f"WMA candlestick overlay error: {e}")
            fig_candle.update_layout(
                height=380,
                template='plotly_dark',
                margin=dict(l=20, r=20, t=30, b=30),
                xaxis_rangeslider_visible=False,
                showlegend=True,
                legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='left', x=0)
            )
            st.markdown(f"<h4 style='color:{ACCENT};'>📈 Candlestick Chart – {largest_tf}</h4>", unsafe_allow_html=True)
            st.plotly_chart(fig_candle, width="stretch")
        else:
            st.warning(f"Not enough data to display candlestick chart for {largest_tf}.")


def render_guide_tab():
    """Render a comprehensive Analysis Guide for beginners and advanced users."""

    st.markdown(
        f"<h2 style='color:#E5E7EB; font-size:1.8rem; margin-bottom:0.5rem;'>📚 Analysis Guide</h2>",
        unsafe_allow_html=True,
    )
    st.markdown(
        f"<p style='color:#8CA1B6; font-size:1rem; margin-bottom:2rem;'>Complete guide to understanding how the dashboard analyzes cryptocurrency markets</p>",
        unsafe_allow_html=True,
    )

    # Introduction
    intro_html = f"""
    <div class='panel-box'>
      <b style='color:#06D6A0; font-size:1.3rem;'>👋 How This Dashboard Works</b>
      <p style='color:#E5E7EB; font-size:0.95rem; line-height:1.7; margin-top:1rem;'>
        This dashboard analyzes cryptocurrency markets using technical indicators - mathematical calculations based on price, volume, and time data. 
        Instead of relying on just one indicator (which can give false signals), this system combines multiple indicators across four categories:
        <b>Trend</b>, <b>Momentum</b>, <b>Volume</b>, and <b>Volatility</b>.
      </p>
      <p style='color:#E5E7EB; font-size:0.95rem; line-height:1.7; margin-top:0.5rem;'>
        By grouping indicators into categories and weighting them properly, the system avoids counting the same information multiple times.
        The result is a <b>Confidence Score (0-100%)</b> that tells you how strong the buy or sell signal is.
      </p>
    </div>
    """

    # Signal Generation
    signal_html = f"""
    <div class='panel-box'>
      <b style='color:#06D6A0; font-size:1.3rem;'>🎯 How Signals Are Generated</b>
      
      <p style='color:#E5E7EB; font-size:0.95rem; margin-top:1rem; line-height:1.7;'>
        The dashboard calculates four separate scores, each measuring a different aspect of market behavior. 
        Each score ranges from <b>-1 (very bearish)</b> to <b>+1 (very bullish)</b>:
      </p>
      
      <div style='margin-top:1.5rem; padding:15px; background-color:rgba(6,214,160,0.05); border-left:4px solid #06D6A0; border-radius:6px;'>
        <b style='color:#06D6A0; font-size:1.1rem;'>1️⃣ TREND Score (40% weight)</b>
        <p style='color:#8CA1B6; font-size:0.9rem; margin:0.8rem 0 0.5rem 0; line-height:1.6;'>
          <b>What it measures:</b> Is the price going up, down, or sideways over time?
        </p>
        <p style='color:#8CA1B6; font-size:0.88rem; margin:0.3rem 0; line-height:1.6;'>
          <b>How it works:</b>
        </p>
        <ul style='color:#8CA1B6; font-size:0.88rem; line-height:1.7; margin-left:1.2rem;'>
          <li><b>EMA (Exponential Moving Average):</b> Shows average price over time. If shorter EMAs (5, 9, 21 periods) are above longer EMAs (50 periods), it's an uptrend.</li>
          <li><b>SuperTrend:</b> Draws a line above/below price. If price is above the line = uptrend, below = downtrend. Simple visual confirmation.</li>
          <li><b>Ichimoku Cloud:</b> Shows a "cloud" on the chart. Price above cloud = bullish, below cloud = bearish. The cloud acts like support/resistance.</li>
          <li><b>Parabolic SAR:</b> Shows dots above/below candles. Dots below = uptrend, dots above = downtrend. When dots flip, trend may be reversing.</li>
          <li><b>ADX (Average Directional Index):</b> Doesn't show direction, only trend <i>strength</i>. Above 25 = strong trend (whichever direction), below 20 = weak/no trend.</li>
        </ul>
        <p style='color:#8CA1B6; font-size:0.85rem; margin-top:0.8rem; padding:8px; background-color:rgba(255,255,255,0.03); border-radius:4px;'>
          💡 <b>Why 40% weight?</b> Trend is the most important factor. Trading with the trend gives higher probability of success. "The trend is your friend."
        </p>
      </div>
      
      <div style='margin-top:1.5rem; padding:15px; background-color:rgba(255,209,102,0.05); border-left:4px solid #FFD166; border-radius:6px;'>
        <b style='color:#06D6A0; font-size:1.1rem;'>2️⃣ MOMENTUM Score (30% weight)</b>
        <p style='color:#8CA1B6; font-size:0.9rem; margin:0.8rem 0 0.5rem 0; line-height:1.6;'>
          <b>What it measures:</b> How fast and strong is the price movement? Is it accelerating or slowing down?
        </p>
        <p style='color:#8CA1B6; font-size:0.88rem; margin:0.3rem 0; line-height:1.6;'>
          <b>How it works:</b>
        </p>
        <ul style='color:#8CA1B6; font-size:0.88rem; line-height:1.7; margin-left:1.2rem;'>
          <li><b>RSI (Relative Strength Index):</b> Ranges from 0-100. Above 70 = overbought (may reverse down), below 30 = oversold (may reverse up). 
              Between 45-55 = neutral.</li>
          <li><b>MACD (Moving Average Convergence Divergence):</b> Shows momentum with a line and histogram. When MACD line crosses above signal line and histogram is positive = bullish momentum building.</li>
          <li><b>Stochastic RSI:</b> More sensitive version of RSI. Ranges 0-1. Above 0.8 = overbought, below 0.2 = oversold. Better for timing entries/exits.</li>
          <li><b>Williams %R:</b> Similar to Stochastic. Below -80 = oversold, above -20 = overbought. Confirms momentum signals.</li>
          <li><b>CCI (Commodity Channel Index):</b> Measures deviation from average price. Above +100 = overbought, below -100 = oversold.</li>
        </ul>
        <p style='color:#8CA1B6; font-size:0.85rem; margin-top:0.8rem; padding:8px; background-color:rgba(255,255,255,0.03); border-radius:4px;'>
          💡 <b>Why 30% weight?</b> Momentum shows if the move has "power" behind it. Strong momentum = move likely continues. Weak momentum = move may fade.
        </p>
      </div>
      
      <div style='margin-top:1.5rem; padding:15px; background-color:rgba(6,214,160,0.05); border-left:4px solid #06D6A0; border-radius:6px;'>
        <b style='color:#06D6A0; font-size:1.1rem;'>3️⃣ VOLUME Score (20% weight)</b>
        <p style='color:#8CA1B6; font-size:0.9rem; margin:0.8rem 0 0.5rem 0; line-height:1.6;'>
          <b>What it measures:</b> Are buyers/sellers actually participating? Is there real money behind the move?
        </p>
        <p style='color:#8CA1B6; font-size:0.88rem; margin:0.3rem 0; line-height:1.6;'>
          <b>How it works:</b>
        </p>
        <ul style='color:#8CA1B6; font-size:0.88rem; line-height:1.7; margin-left:1.2rem;'>
          <li><b>OBV (On-Balance Volume):</b> Adds volume on up days, subtracts on down days. Rising OBV = accumulation (buying), falling = distribution (selling).</li>
          <li><b>Volume Spikes:</b> Sudden 2x+ increase in volume. If price goes up on high volume = strong buy signal. If price drops on high volume = strong sell signal. 
              Volume confirms the move is real, not just noise.</li>
          <li><b>VWAP (Volume Weighted Average Price):</b> Average price weighted by volume. Price above VWAP = bullish, below = bearish.</li>
        </ul>
        <p style='color:#8CA1B6; font-size:0.85rem; margin-top:0.8rem; padding:8px; background-color:rgba(255,255,255,0.03); border-radius:4px;'>
          💡 <b>Why 20% weight?</b> Volume confirms price moves. "Volume precedes price" - smart money shows up in volume before price moves.
        </p>
      </div>
      
      <div style='margin-top:1.5rem; padding:15px; background-color:rgba(255,209,102,0.05); border-left:4px solid #FFD166; border-radius:6px;'>
        <b style='color:#06D6A0; font-size:1.1rem;'>4️⃣ VOLATILITY Score (10% weight)</b>
        <p style='color:#8CA1B6; font-size:0.9rem; margin:0.8rem 0 0.5rem 0; line-height:1.6;'>
          <b>What it measures:</b> How risky is the market right now? Is price stable or whipping around?
        </p>
        <p style='color:#8CA1B6; font-size:0.88rem; margin:0.3rem 0; line-height:1.6;'>
          <b>How it works:</b>
        </p>
        <ul style='color:#8CA1B6; font-size:0.88rem; line-height:1.7; margin-left:1.2rem;'>
          <li><b>ATR (Average True Range):</b> Measures how much price typically moves. Low ATR = stable market (easier to trade), high ATR = volatile market (riskier). 
              Expressed as % of price. Under 1.5% = very stable, over 5% = very volatile.</li>
          <li><b>Bollinger Band Width:</b> Bands expand in volatile markets, contract in calm markets. Narrow bands = low risk, wide bands = high risk. 
              When bands are tight, a big move is often coming.</li>
        </ul>
        <p style='color:#8CA1B6; font-size:0.85rem; margin-top:0.8rem; padding:8px; background-color:rgba(255,255,255,0.03); border-radius:4px;'>
          💡 <b>Why only 10% weight?</b> Volatility doesn't predict direction, only risk level. It's important but shouldn't override trend/momentum/volume.
        </p>
      </div>
      
      <div style='margin-top:1.5rem; padding:15px; background-color:rgba(6,214,160,0.1); border-radius:8px;'>
        <b style='color:#06D6A0; font-size:1.1rem;'>📐 FINAL CALCULATION</b>
        <p style='color:#E5E7EB; font-size:0.9rem; margin:0.8rem 0; line-height:1.7;'>
          Each category gets a score from -1 to +1. These are combined with weights:
        </p>
        <div style='font-family:monospace; background-color:rgba(0,0,0,0.3); padding:12px; border-radius:6px; color:#06D6A0; font-size:0.85rem;'>
          Final Score = (Trend × 0.40) + (Momentum × 0.30) + (Volume × 0.20) + (Volatility × 0.10)
        </div>
        <p style='color:#E5E7EB; font-size:0.9rem; margin:0.8rem 0; line-height:1.7;'>
          This gives a number from -1 to +1, which is then converted to a percentage (0-100%):
        </p>
        <div style='font-family:monospace; background-color:rgba(0,0,0,0.3); padding:12px; border-radius:6px; color:#06D6A0; font-size:0.85rem;'>
          Confidence = (Final Score + 1) ÷ 2 × 100
        </div>
        <p style='color:#8CA1B6; font-size:0.88rem; margin-top:0.8rem; line-height:1.6;'>
          <b>Example:</b> If Trend=+0.8, Momentum=+0.6, Volume=+0.4, Volatility=+0.5<br>
          Final = (0.8×0.4) + (0.6×0.3) + (0.4×0.2) + (0.5×0.1) = 0.65<br>
          Confidence = (0.65 + 1) ÷ 2 × 100 = <b style='color:#06D6A0;'>82.5%</b> → <b>STRONG BUY</b>
        </p>
      </div>
    </div>
    """

    # Quality Filters
    filters_html = f"""
    <div class='panel-box'>
      <b style='color:#06D6A0; font-size:1.3rem;'>🔒 Quality Filters (Signal Validation)</b>
      <p style='color:#E5E7EB; font-size:0.95rem; margin-top:1rem; line-height:1.7;'>
        Even if the confidence score is high, signals must pass additional quality checks to avoid false signals. 
        Think of these as "safety gates" - the signal needs a green light from multiple gates before showing BUY or SELL.
      </p>
      
      <div style='margin-top:1.5rem; padding:12px; background-color:rgba(6,214,160,0.08); border-radius:6px;'>
        <b style='color:#06D6A0; font-size:1.05rem;'>✅ For BUY Signals - Must Pass ALL:</b>
        <ul style='color:#8CA1B6; font-size:0.9rem; line-height:1.7; margin:0.8rem 0 0.5rem 1.2rem;'>
          <li><b>Trend Check:</b> Trend Score must be above +0.2 (confirmed uptrend). Won't show BUY in a downtrend even if other indicators say buy.</li>
          <li><b>Volume Check:</b> Volume Score must be 0 or positive (volume supporting the move). No buying without volume confirmation.</li>
          <li><b>Momentum Check:</b> Momentum Score must be above -0.3 (not strongly bearish). Avoids buying when momentum is clearly against you.</li>
          <li><b>Volatility Check:</b> Volatility Score must be above -0.5 (not too risky). Won't signal in extremely choppy/dangerous conditions.</li>
        </ul>
        <p style='color:#8CA1B6; font-size:0.85rem; margin-top:0.5rem; padding:8px; background-color:rgba(255,255,255,0.03); border-radius:4px;'>
          ⚠️ If ANY filter fails, signal changes to WAIT even if confidence is 80%+. Better safe than sorry.
        </p>
      </div>
      
      <div style='margin-top:1rem; padding:12px; background-color:rgba(239,71,111,0.08); border-radius:6px;'>
        <b style='color:#EF476F; font-size:1.05rem;'>✅ For SELL Signals - Must Pass ALL:</b>
        <ul style='color:#8CA1B6; font-size:0.9rem; line-height:1.7; margin:0.8rem 0 0.5rem 1.2rem;'>
          <li><b>Trend Check:</b> Trend Score must be below -0.2 (confirmed downtrend). Won't show SELL in an uptrend.</li>
          <li><b>Volume Check:</b> Volume Score must be 0 or negative (volume supporting down move).</li>
          <li><b>Momentum Check:</b> Momentum Score must be below +0.3 (not strongly bullish).</li>
          <li><b>Volatility Check:</b> Volatility Score must be above -0.5 (not too risky).</li>
        </ul>
      </div>
      
      <div style='margin-top:1.5rem; padding:12px; background-color:rgba(255,209,102,0.1); border-radius:6px;'>
        <b style='color:#FFD166; font-size:1.05rem;'>🎚️ Adaptive Thresholds (Smart Adjustment)</b>
        <p style='color:#8CA1B6; font-size:0.88rem; margin:0.8rem 0; line-height:1.7;'>
          The confidence thresholds automatically adjust based on market conditions:
        </p>
        <ul style='color:#8CA1B6; font-size:0.88rem; line-height:1.7; margin-left:1.2rem;'>
          <li><b>Normal Market:</b> BUY if ≥65%, SELL if ≤35% (standard)</li>
          <li><b>High Volatility:</b> BUY if ≥70%, SELL if ≤30% (more strict - harder to get signals in choppy markets)</li>
          <li><b>Weak Trend (ADX < 20):</b> BUY if ≥75%, SELL if ≤25% (very strict - ranging markets are hard to trade)</li>
        </ul>
        <p style='color:#8CA1B6; font-size:0.85rem; margin-top:0.8rem; padding:8px; background-color:rgba(255,255,255,0.03); border-radius:4px;'>
          💡 This prevents the dashboard from giving signals in bad market conditions. When the market is unclear, it's better to wait.
        </p>
      </div>
    </div>
    """

    # Confidence Levels

    # Continue with more sections... (character limit, will continue in next response)

    st.markdown(intro_html, unsafe_allow_html=True)
    st.markdown(signal_html, unsafe_allow_html=True)
    st.markdown(filters_html, unsafe_allow_html=True)

    # Leverage section

    # Backtest section

    # ML section

    # Leverage section - detailed
    # Leverage section - in styled panel
    leverage_panel = """
    <div class='panel-box'>
      <b style='color:#06D6A0; font-size:1.3rem;'>⚖️ Leverage Recommendations</b>
      <p style='color:#E5E7EB; font-size:0.95rem; margin-top:1rem; line-height:1.7;'>
        <b>Conservative Approach:</b> Dashboard suggests maximum 10x leverage.
      </p>
      <p style='color:#E5E7EB; font-size:0.95rem; line-height:1.7; margin-top:1rem;'>
        <b>How It Works:</b>
      </p>
      <ul style='color:#8CA1B6; font-size:0.9rem; line-height:1.7; margin-left:1.2rem;'>
        <li><b>Confidence-Based:</b> Higher confidence score allows higher leverage
          <ul style='margin-left:1rem; margin-top:0.3rem;'>
            <li>Below 50%: Max 2x (very low confidence)</li>
            <li>50-65%: Max 4x</li>
            <li>65-75%: Max 6x</li>
            <li>Above 75%: Max 10x</li>
          </ul>
        </li>
        <li style='margin-top:0.5rem;'><b>Risk Score Adjustment:</b> Combines 4 factors:
          <ul style='margin-left:1rem; margin-top:0.3rem;'>
            <li>Bollinger Band width (volatility)</li>
            <li>RSI extremes (overbought/oversold)</li>
            <li>Volume patterns</li>
            <li>Support/resistance proximity</li>
          </ul>
        </li>
      </ul>
      <p style='color:#E5E7EB; font-size:0.9rem; margin-top:1rem; padding:10px; background-color:rgba(239,71,111,0.1); border-radius:6px;'>
        <b style='color:#EF476F;'>Best Practice:</b> Use 2-3x on best setups (80%+ confidence), 1-2x on good setups. Never use maximum suggested leverage.
      </p>
    </div>
    """
    st.markdown(leverage_panel, unsafe_allow_html=True)
    
    # Backtest section - in styled panel
    backtest_panel = """
    <div class='panel-box'>
      <b style='color:#06D6A0; font-size:1.3rem;'>📊 Backtest Metrics Explained</b>
      <p style='color:#E5E7EB; font-size:0.95rem; margin-top:1rem; line-height:1.7;'>
        Test your strategy on historical data before risking real money. Key metrics:
      </p>
      <div style='margin-top:1.5rem;'>
        <p style='color:#E5E7EB; font-size:0.9rem; line-height:1.7;'>
          <b>Win Rate:</b> Percentage of profitable trades. Above 55% is good, above 60% is excellent. 
          Remember: You can have 40% win rate and still profit if wins are bigger than losses.
        </p>
        <p style='color:#E5E7EB; font-size:0.9rem; line-height:1.7; margin-top:0.8rem;'>
          <b>Profit Factor:</b> Gross profit ÷ gross loss. Above 1.5 is good, above 2.0 is excellent. 
          Below 1.0 means losing money overall.
        </p>
        <p style='color:#E5E7EB; font-size:0.9rem; line-height:1.7; margin-top:0.8rem;'>
          <b>Max Drawdown:</b> Largest peak-to-trough loss. Shows your worst losing streak. 
          Below 15% is excellent, 15-25% is acceptable.
        </p>
        <p style='color:#E5E7EB; font-size:0.9rem; line-height:1.7; margin-top:0.8rem;'>
          <b>Sharpe Ratio:</b> Risk-adjusted returns (annualised). Above 1.0 is good, above 2.0 is excellent. 
          Measures return per unit of risk.
        </p>
        <p style='color:#E5E7EB; font-size:0.9rem; line-height:1.7; margin-top:0.8rem;'>
          <b>Commission Impact:</b> Default 0.1% per trade (0.2% round trip). On short timeframes (1m-5m), 
          commission can eat most profits. Use 15m+ for better results.
        </p>
      </div>
    </div>
    """
    st.markdown(backtest_panel, unsafe_allow_html=True)
    
    # ML section - in styled panel
    ml_panel = """
    <div class='panel-box'>
      <b style='color:#06D6A0; font-size:1.3rem;'>🤖 Machine Learning Predictions</b>
      <p style='color:#E5E7EB; font-size:0.95rem; margin-top:1rem; line-height:1.7;'>
        Uses Gradient Boosting algorithm to predict next candle direction.
      </p>
      <div style='margin-top:1.5rem;'>
        <p style='color:#E5E7EB; font-size:0.9rem;'><b>How It Works:</b></p>
        <ol style='color:#8CA1B6; font-size:0.9rem; line-height:1.7; margin-left:1.2rem; margin-top:0.5rem;'>
          <li>Fetches recent OHLCV data</li>
          <li>Calculates 12 technical indicators (EMAs, RSI, MACD, OBV, ATR, Bollinger width, etc.)</li>
          <li>Trains on historical patterns</li>
          <li>Predicts: LONG if ≥60% probability up, SHORT if ≤40%, NEUTRAL otherwise</li>
        </ol>
      </div>
      <p style='color:#E5E7EB; font-size:0.9rem; margin-top:1rem; line-height:1.7;'>
        <b>Expected Performance:</b> 60-65% accuracy on well-trained data.
      </p>
      <div style='margin-top:1rem;'>
        <p style='color:#E5E7EB; font-size:0.9rem;'><b>When to Use:</b></p>
        <ul style='color:#8CA1B6; font-size:0.9rem; line-height:1.7; margin-left:1.2rem; margin-top:0.5rem;'>
          <li>✅ You have 500+ candles of data</li>
          <li>✅ You want additional confirmation</li>
          <li>❌ Don't use as your only decision factor</li>
        </ul>
      </div>
      <p style='color:#E5E7EB; font-size:0.85rem; margin-top:1rem; padding:10px; background-color:rgba(255,209,102,0.1); border-radius:6px;'>
        <b style='color:#FFD166;'>Limitations:</b> Only sees past data. Cannot predict news events, black swans, or regime changes. 
        Always combine with other analysis.
      </p>
    </div>
    """
    st.markdown(ml_panel, unsafe_allow_html=True)
    
    # Tips section - in styled panel
    tips_panel = """
    <div class='panel-box'>
      <b style='color:#06D6A0; font-size:1.3rem;'>💡 Pro Trading Tips</b>
      <ul style='color:#E5E7EB; font-size:0.9rem; line-height:1.8; margin-top:1rem; margin-left:1.2rem;'>
        <li><b>Multi-Timeframe Confirmation:</b> Check 3 timeframes (e.g., 15m, 1h, 4h). If all agree, signal is stronger.</li>
        <li><b>Volume is King:</b> Never trade without volume confirmation. Volume Score must be positive for buys.</li>
        <li><b>High Confidence ≠ Guaranteed:</b> Even 90% confidence can lose. Always use stop-losses.</li>
        <li><b>Backtest First:</b> Test your timeframe and settings for 500+ candles before risking real money.</li>
        <li><b>Start Small:</b> Begin with 2x leverage max. Only increase after proving consistency over 50+ trades.</li>
        <li><b>Market Conditions Matter:</b> Dashboard works best in trending markets (ADX > 25). In choppy markets, expect more WAIT signals.</li>
        <li><b>Patience Wins:</b> Wait for 70%+ confidence. Better to miss a trade than force a marginal 60-65% one.</li>
        <li><b>Risk Management:</b> Risk only 1-2% of capital per trade. With $10,000, that's $100-200 max loss.</li>
      </ul>
      <p style='color:#E5E7EB; font-size:0.85rem; margin-top:1.5rem; padding:12px; background-color:rgba(239,71,111,0.12); border-left:4px solid #EF476F; border-radius:6px;'>
        <b style='color:#EF476F;'>⚠️ Important:</b> This is a tool, not magic. Provides probability-based signals. 
        YOU make final decisions. YOU are responsible for your trading results.
      </p>
    </div>
    """
    st.markdown(tips_panel, unsafe_allow_html=True)

    # === New Features Guide Sections ===

    # Multi-Timeframe Confluence
    mtf_panel = """
    <div class='panel-box'>
      <b style='color:#06D6A0; font-size:1.3rem;'>🔀 Multi-Timeframe Confluence</b>
      <p style='color:#E5E7EB; font-size:0.95rem; margin-top:1rem; line-height:1.7;'>
        Analyses your chosen coin across 5 timeframes (5m, 15m, 1h, 4h, 1d) simultaneously and combines their signals into a single <b>Confluence Score</b>.
      </p>
      <div style='margin-top:1rem;'>
        <p style='color:#E5E7EB; font-size:0.9rem;'><b>How It Works:</b></p>
        <ul style='color:#8CA1B6; font-size:0.9rem; line-height:1.7; margin-left:1.2rem; margin-top:0.5rem;'>
          <li>Runs the full technical analysis on each of the 5 timeframes.</li>
          <li>Counts how many timeframes agree on the same direction (LONG or SHORT).</li>
          <li>Confluence % = (agreeing timeframes / total valid timeframes) × 100.</li>
          <li>Also shows average confidence across all timeframes.</li>
        </ul>
      </div>
      <div style='margin-top:1rem;'>
        <p style='color:#E5E7EB; font-size:0.9rem;'><b>How to Read It:</b></p>
        <ul style='color:#8CA1B6; font-size:0.9rem; line-height:1.7; margin-left:1.2rem; margin-top:0.5rem;'>
          <li><b>80-100% Confluence:</b> Very strong setup — almost all timeframes agree. Best entries.</li>
          <li><b>60-80% Confluence:</b> Moderate agreement — trade with caution, use smaller position.</li>
          <li><b>Below 60%:</b> Weak confluence — timeframes are mixed, better to wait.</li>
        </ul>
      </div>
      <p style='color:#E5E7EB; font-size:0.85rem; margin-top:1rem; padding:10px; background-color:rgba(255,209,102,0.1); border-radius:6px;'>
        <b style='color:#FFD166;'>Pro Tip:</b> The best trades happen when the 1h, 4h, and 1d all agree on the same direction.
        Short timeframes (5m, 15m) are useful for fine-tuning entries, but the higher timeframes set the trend.
      </p>
    </div>
    """
    st.markdown(mtf_panel, unsafe_allow_html=True)

    # Correlation Matrix
    corr_panel = """
    <div class='panel-box'>
      <b style='color:#06D6A0; font-size:1.3rem;'>📊 Correlation Matrix</b>
      <p style='color:#E5E7EB; font-size:0.95rem; margin-top:1rem; line-height:1.7;'>
        Shows how major crypto assets (BTC, ETH, BNB, SOL, ADA, XRP) move relative to each other.
        You can also add your own coin to see how it correlates with the majors.
      </p>
      <div style='margin-top:1rem;'>
        <p style='color:#E5E7EB; font-size:0.9rem;'><b>How to Read the Heatmap:</b></p>
        <ul style='color:#8CA1B6; font-size:0.9rem; line-height:1.7; margin-left:1.2rem; margin-top:0.5rem;'>
          <li><b>+1.0 (bright green):</b> Perfectly correlated — they move in the same direction at the same time.</li>
          <li><b>0.0 (yellow):</b> No correlation — their movements are independent of each other.</li>
          <li><b>-1.0 (red):</b> Negatively correlated — when one goes up, the other tends to go down.</li>
        </ul>
      </div>
      <div style='margin-top:1rem;'>
        <p style='color:#E5E7EB; font-size:0.9rem;'><b>Why It Matters:</b></p>
        <ul style='color:#8CA1B6; font-size:0.9rem; line-height:1.7; margin-left:1.2rem; margin-top:0.5rem;'>
          <li><b>Diversification:</b> If two assets are highly correlated (>0.8), holding both doubles your risk instead of spreading it.</li>
          <li><b>Hedging:</b> Low or negative correlation pairs can be used to hedge positions.</li>
          <li><b>Market Regime:</b> When all correlations spike to 1.0, it usually means panic selling or a macro event — everything moves together.</li>
        </ul>
      </div>
    </div>
    """
    st.markdown(corr_panel, unsafe_allow_html=True)

    # Session Analysis
    session_panel = """
    <div class='panel-box'>
      <b style='color:#06D6A0; font-size:1.3rem;'>🌍 Session Analysis</b>
      <p style='color:#E5E7EB; font-size:0.95rem; margin-top:1rem; line-height:1.7;'>
        Crypto trades 24/7, but activity varies by global trading session. This tool breaks down market behaviour across three sessions:
      </p>
      <ul style='color:#8CA1B6; font-size:0.9rem; line-height:1.7; margin-left:1.2rem; margin-top:0.5rem;'>
        <li><b>Asian Session (00:00-08:00 UTC):</b> Tokyo, Hong Kong, Singapore. Generally lower volume, tighter ranges.</li>
        <li><b>European Session (08:00-16:00 UTC):</b> London, Frankfurt. Usually high volume, strong moves.</li>
        <li><b>US Session (16:00-00:00 UTC):</b> New York, Chicago. Highest volatility, major news releases.</li>
      </ul>
      <div style='margin-top:1rem;'>
        <p style='color:#E5E7EB; font-size:0.9rem;'><b>Metrics Explained:</b></p>
        <ul style='color:#8CA1B6; font-size:0.9rem; line-height:1.7; margin-left:1.2rem; margin-top:0.5rem;'>
          <li><b>Volume:</b> Total amount traded during the session. Higher volume means more liquidity and easier trade execution.</li>
          <li><b>Avg Range (%):</b> Average high-to-low price swing per candle. Higher range = more volatility = bigger potential moves.</li>
          <li><b>Avg Return (%):</b> Average open-to-close change per candle. Positive = price tends to rise, negative = tends to fall.</li>
        </ul>
      </div>
      <p style='color:#E5E7EB; font-size:0.85rem; margin-top:1rem; padding:10px; background-color:rgba(255,209,102,0.1); border-radius:6px;'>
        <b style='color:#FFD166;'>Pro Tip:</b> Scalpers should target sessions with high volume AND moderate volatility.
        Swing traders can use this data to time entries at the start of active sessions.
      </p>
    </div>
    """
    st.markdown(session_panel, unsafe_allow_html=True)

    # R:R Calculator
    rr_panel = """
    <div class='panel-box'>
      <b style='color:#06D6A0; font-size:1.3rem;'>⚖️ Risk/Reward Calculator</b>
      <p style='color:#E5E7EB; font-size:0.95rem; margin-top:1rem; line-height:1.7;'>
        An interactive tool to plan your trades before entering. Enter your entry, stop loss, and take profit to see the risk/reward ratio and optimal position size.
      </p>
      <div style='margin-top:1rem;'>
        <p style='color:#E5E7EB; font-size:0.9rem;'><b>Key Outputs:</b></p>
        <ul style='color:#8CA1B6; font-size:0.9rem; line-height:1.7; margin-left:1.2rem; margin-top:0.5rem;'>
          <li><b>R:R Ratio:</b> Reward divided by risk. A ratio of 1:2 means you gain $2 for every $1 risked. Aim for at least 1:1.5.</li>
          <li><b>Position Size:</b> How much of the asset to buy based on your account size and risk tolerance (e.g. 2% risk per trade).</li>
          <li><b>PnL Table:</b> Shows potential profit and loss at different leverage levels (1x to 20x), so you can visualise the impact of leverage.</li>
        </ul>
      </div>
      <p style='color:#E5E7EB; font-size:0.85rem; margin-top:1rem; padding:10px; background-color:rgba(239,71,111,0.1); border-radius:6px;'>
        <b style='color:#EF476F;'>Important:</b> Always plan your risk BEFORE entering a trade.
        The calculator helps you determine the right position size so a single losing trade does not wipe out your account.
      </p>
    </div>
    """
    st.markdown(rr_panel, unsafe_allow_html=True)

    # Liquidation Levels
    liq_panel = """
    <div class='panel-box'>
      <b style='color:#06D6A0; font-size:1.3rem;'>💀 Liquidation Level Estimator</b>
      <p style='color:#E5E7EB; font-size:0.95rem; margin-top:1rem; line-height:1.7;'>
        Shows the price at which your position would be liquidated (forced closed) for different leverage levels.
        Uses the simplified formula for isolated margin.
      </p>
      <div style='margin-top:1rem;'>
        <p style='color:#E5E7EB; font-size:0.9rem;'><b>How to Read It:</b></p>
        <ul style='color:#8CA1B6; font-size:0.9rem; line-height:1.7; margin-left:1.2rem; margin-top:0.5rem;'>
          <li><b>Distance from Entry:</b> How far price needs to move against you before liquidation. At 10x, that's only ~10%.</li>
          <li><b>Risk Level:</b> LOW (>10% distance), Medium (3-10%), HIGH (<3%). Higher leverage = closer liquidation = more danger.</li>
        </ul>
      </div>
      <div style='margin-top:1rem;'>
        <p style='color:#E5E7EB; font-size:0.9rem;'><b>Key Takeaways:</b></p>
        <ul style='color:#8CA1B6; font-size:0.9rem; line-height:1.7; margin-left:1.2rem; margin-top:0.5rem;'>
          <li>At <b>2x leverage</b>, price must move ~50% against you for liquidation — relatively safe.</li>
          <li>At <b>10x leverage</b>, only ~10% move needed — a normal daily swing can wipe you out.</li>
          <li>At <b>50x+ leverage</b>, even a 2% move can liquidate — extremely dangerous.</li>
        </ul>
      </div>
      <p style='color:#E5E7EB; font-size:0.85rem; margin-top:1rem; padding:10px; background-color:rgba(239,71,111,0.1); border-radius:6px;'>
        <b style='color:#EF476F;'>Warning:</b> These are estimates for isolated margin. Actual liquidation prices vary by exchange
        due to maintenance margin rates and funding fees. Always check your exchange for exact values.
      </p>
    </div>
    """
    st.markdown(liq_panel, unsafe_allow_html=True)

        # Confidence Levels - rendered in parts
    st.markdown('<div class="panel-box"><b style="color:#06D6A0; font-size:1.3rem;">📊 Understanding Confidence Levels</b><p style="color:#E5E7EB; font-size:0.95rem; margin-top:1rem;">The confidence score tells you how likely the signal is to work out. Higher confidence = higher probability of success.</p></div>', unsafe_allow_html=True)
    
    st.markdown('<div style="padding:12px; margin:10px 0; background: linear-gradient(90deg, rgba(6,214,160,0.25) 0%, transparent 100%); border-left:4px solid #06D6A0; border-radius:6px;"><b style="color:#06D6A0;">80-100%: STRONG BUY</b><p style="color:#E5E7EB; margin:0.5rem 0 0 0;"><b>What it means:</b> All indicators aligned. Best setup!</p><p style="color:#8CA1B6; margin:0.5rem 0 0 0;"><b>Action:</b> Strong buy signal. Best entries.</p></div>', unsafe_allow_html=True)
    
    st.markdown('<div style="padding:12px; margin:10px 0; background: linear-gradient(90deg, rgba(6,214,160,0.12) 0%, transparent 100%); border-left:4px solid #06D6A0; border-radius:6px;"><b style="color:#06D6A0;">65-80%: BUY</b><p style="color:#E5E7EB; margin:0.5rem 0 0 0;"><b>What it means:</b> Good setup. High probability.</p><p style="color:#8CA1B6; margin:0.5rem 0 0 0;"><b>Action:</b> Good entry. Smaller position than STRONG BUY.</p></div>', unsafe_allow_html=True)
    
    st.markdown('<div style="padding:12px; margin:10px 0; background: linear-gradient(90deg, rgba(255,209,102,0.15) 0%, transparent 100%); border-left:4px solid #FFD166; border-radius:6px;"><b style="color:#FFD166;">35-65%: WAIT</b><p style="color:#E5E7EB; margin:0.5rem 0 0 0;"><b>What it means:</b> Mixed signals. No edge.</p><p style="color:#8CA1B6; margin:0.5rem 0 0 0;"><b>Action:</b> DO NOT TRADE. Wait for better conditions.</p></div>', unsafe_allow_html=True)
    
    st.markdown('<div style="padding:12px; margin:10px 0; background: linear-gradient(90deg, rgba(239,71,111,0.2) 0%, transparent 100%); border-left:4px solid #EF476F; border-radius:6px;"><b style="color:#EF476F;">20-35%: SELL | 0-20%: STRONG SELL</b><p style="color:#E5E7EB; margin:0.5rem 0 0 0;"><b>What it means:</b> Bearish setup.</p><p style="color:#8CA1B6; margin:0.5rem 0 0 0;"><b>Action:</b> Short opportunity.</p></div>', unsafe_allow_html=True)

    # ── Spot Tab ──
    spot_panel = """
    <div class='panel-box'>
      <b style='color:#06D6A0; font-size:1.3rem;'>Spot Analysis Tab</b>
      <p style='color:#E5E7EB; font-size:0.95rem; margin-top:1rem; line-height:1.7;'>
        Spot market analysis performs an in-depth technical review of your selected cryptocurrency.
        Designed for direct buy/sell decisions without leverage.
      </p>
      <div style='margin-top:1rem;'>
        <p style='color:#E5E7EB; font-size:0.9rem;'><b>What It Shows:</b></p>
        <ul style='color:#8CA1B6; font-size:0.9rem; line-height:1.7; margin-left:1.2rem; margin-top:0.5rem;'>
          <li><b>Price & Change:</b> Current price, 24-hour change percentage, and trading volume.</li>
          <li><b>Technical Signal:</b> The composite signal (BUY/SELL/WAIT) from Trend, Momentum, Volume, and Volatility scores.</li>
          <li><b>Confidence Score:</b> 0-100%, indicates how strong the signal is.</li>
          <li><b>Category Details:</b> Each of the four categories (Trend, Momentum, Volume, Volatility) is visualized separately with supporting indicators.</li>
          <li><b>Fear & Greed Index:</b> Overall market fear/greed level for Bitcoin (0=Extreme Fear, 100=Extreme Greed).</li>
          <li><b>Leverage Suggestion:</b> Recommended maximum leverage based on confidence score and risk level.</li>
        </ul>
      </div>
      <p style='color:#E5E7EB; font-size:0.85rem; margin-top:1rem; padding:10px; background-color:rgba(255,209,102,0.1); border-radius:6px;'>
        <b style='color:#FFD166;'>Tip:</b> Spot market is safer as it uses no leverage.
        Entry with 65%+ confidence score is recommended.
      </p>
    </div>
    """
    st.markdown(spot_panel, unsafe_allow_html=True)

    # ── Position Tab ──
    position_panel = """
    <div class='panel-box'>
      <b style='color:#06D6A0; font-size:1.3rem;'>Position Management Tab</b>
      <p style='color:#E5E7EB; font-size:0.95rem; margin-top:1rem; line-height:1.7;'>
        Designed to track and manage your open positions.
        Visually presents entry price, stop-loss, take-profit and other details for futures/margin trades.
      </p>
      <div style='margin-top:1rem;'>
        <p style='color:#E5E7EB; font-size:0.9rem;'><b>What It Shows:</b></p>
        <ul style='color:#8CA1B6; font-size:0.9rem; line-height:1.7; margin-left:1.2rem; margin-top:0.5rem;'>
          <li><b>Position Details:</b> Entry price, current price, profit/loss percentage.</li>
          <li><b>Stop-Loss & Take-Profit:</b> ATR-based automatically calculated stop and target levels.</li>
          <li><b>Leverage Analysis:</b> Liquidation distance and risk level at the selected leverage.</li>
          <li><b>Technical Update:</b> Shows the current state of technical indicators even while a position is open.</li>
        </ul>
      </div>
    </div>
    """
    st.markdown(position_panel, unsafe_allow_html=True)

    # ── Ensemble AI Tab ──
    ensemble_panel = """
    <div class='panel-box'>
      <b style='color:#06D6A0; font-size:1.3rem;'>Ensemble AI Tab</b>
      <p style='color:#E5E7EB; font-size:0.95rem; margin-top:1rem; line-height:1.7;'>
        Runs <b>3 different machine learning models</b> instead of a single ML model and combines their results via weighted voting.
        This approach produces more reliable predictions than any single model.
      </p>
      <div style='margin-top:1.5rem; padding:15px; background-color:rgba(6,214,160,0.05); border-left:4px solid #06D6A0; border-radius:6px;'>
        <b style='color:#06D6A0; font-size:1.1rem;'>The 3 Models:</b>
        <ul style='color:#8CA1B6; font-size:0.9rem; line-height:1.7; margin-left:1.2rem; margin-top:0.5rem;'>
          <li><b>Gradient Boosting (45% weight):</b> The most powerful model. Learns complex relationships in the data by sequentially building decision trees that correct previous errors. Best at capturing nonlinear patterns.</li>
          <li><b>Random Forest (35% weight):</b> Builds hundreds of independent decision trees and decides by majority vote. Resistant to overfitting.</li>
          <li><b>Logistic Regression (20% weight):</b> The simplest model. Captures linear relationships. Acts as a balancing force when the other two are wrong.</li>
        </ul>
      </div>
      <div style='margin-top:1.5rem; padding:15px; background-color:rgba(255,209,102,0.05); border-left:4px solid #FFD166; border-radius:6px;'>
        <b style='color:#FFD166; font-size:1.1rem;'>Key Metrics:</b>
        <ul style='color:#8CA1B6; font-size:0.9rem; line-height:1.7; margin-left:1.2rem; margin-top:0.5rem;'>
          <li><b>Model Agreement:</b> Shows how many of the 3 models vote in the same direction. 100% = all three agree, 66% = two agree one differs, 33% = no consensus.</li>
          <li><b>Ensemble Probability:</b> Weighted average probability. Above 60% = LONG (bullish), below 40% = SHORT (bearish), in between = NEUTRAL.</li>
          <li><b>Gauge Chart:</b> Green zone = buy signal, red zone = sell signal. The further the needle is from center, the stronger the signal.</li>
        </ul>
      </div>
      <div style='margin-top:1rem;'>
        <p style='color:#E5E7EB; font-size:0.9rem;'><b>How It Works:</b></p>
        <ol style='color:#8CA1B6; font-size:0.9rem; line-height:1.7; margin-left:1.2rem; margin-top:0.5rem;'>
          <li>OHLCV data is fetched for the selected coin.</li>
          <li>12 technical indicators are computed (EMAs, RSI, MACD, OBV, ATR, Bollinger width, etc.).</li>
          <li>All 3 models are trained on these indicators.</li>
          <li>Each model independently predicts (LONG/SHORT probability).</li>
          <li>Predictions are combined with weights: (GB x 0.45) + (RF x 0.35) + (LR x 0.20).</li>
        </ol>
      </div>
      <p style='color:#E5E7EB; font-size:0.85rem; margin-top:1rem; padding:10px; background-color:rgba(255,209,102,0.1); border-radius:6px;'>
        <b style='color:#FFD166;'>Warning:</b> ML models only look at historical data. They cannot predict news, regulations, or unexpected events.
        Use Ensemble AI alongside other technical analysis tools, not in isolation.
      </p>
    </div>
    """
    st.markdown(ensemble_panel, unsafe_allow_html=True)

    # ── Heatmap Tab ──
    heatmap_panel = """
    <div class='panel-box'>
      <b style='color:#06D6A0; font-size:1.3rem;'>Market Heatmap Tab</b>
      <p style='color:#E5E7EB; font-size:0.95rem; margin-top:1rem; line-height:1.7;'>
        Displays the top 100 cryptocurrencies in a single <b>treemap</b> visualization.
        Understand the overall market at a glance.
      </p>
      <div style='margin-top:1rem;'>
        <p style='color:#E5E7EB; font-size:0.9rem;'><b>How to Read:</b></p>
        <ul style='color:#8CA1B6; font-size:0.9rem; line-height:1.7; margin-left:1.2rem; margin-top:0.5rem;'>
          <li><b>Tile Size:</b> The higher a coin's market cap, the larger its tile. BTC and ETH are usually the largest.</li>
          <li><b>Tile Color:</b> Determined by the 24-hour price change:
            <ul style='margin-left:1rem; margin-top:0.3rem;'>
              <li><span style='color:#00CC00;'>Green</span> = Price increased (darker green = bigger increase)</li>
              <li><span style='color:#FF0000;'>Red</span> = Price decreased (darker red = bigger decrease)</li>
              <li><span style='color:#888;'>Gray</span> = No change or very little change</li>
            </ul>
          </li>
        </ul>
      </div>
      <div style='margin-top:1rem;'>
        <p style='color:#E5E7EB; font-size:0.9rem;'><b>Additional Info:</b></p>
        <ul style='color:#8CA1B6; font-size:0.9rem; line-height:1.7; margin-left:1.2rem; margin-top:0.5rem;'>
          <li><b>Top Gainers:</b> The 10 coins with the highest 24h price increase.</li>
          <li><b>Top Losers:</b> The 10 coins with the largest 24h price decrease.</li>
        </ul>
      </div>
      <p style='color:#E5E7EB; font-size:0.85rem; margin-top:1rem; padding:10px; background-color:rgba(255,209,102,0.1); border-radius:6px;'>
        <b style='color:#FFD166;'>Tip:</b> If most of the map is green, the market is generally bullish;
        if mostly red, it's bearish. Watch for sector-level trends by observing tile clusters.
      </p>
    </div>
    """
    st.markdown(heatmap_panel, unsafe_allow_html=True)

    # ── Monte Carlo Simulation Tab ──
    mc_panel = """
    <div class='panel-box'>
      <b style='color:#06D6A0; font-size:1.3rem;'>Monte Carlo Simulation Tab</b>
      <p style='color:#E5E7EB; font-size:0.95rem; margin-top:1rem; line-height:1.7;'>
        Simulates possible future price paths based on historical price movements.
        Instead of a single prediction, it produces hundreds or thousands of probability scenarios.
      </p>
      <div style='margin-top:1.5rem; padding:15px; background-color:rgba(6,214,160,0.05); border-left:4px solid #06D6A0; border-radius:6px;'>
        <b style='color:#06D6A0; font-size:1.1rem;'>How It Works:</b>
        <ol style='color:#8CA1B6; font-size:0.9rem; line-height:1.7; margin-left:1.2rem; margin-top:0.5rem;'>
          <li>Daily/hourly returns are calculated from historical price data.</li>
          <li>The mean (mu) and standard deviation (sigma) of returns are computed.</li>
          <li>Each simulation: New_Price = Old_Price x e^(mu - sigma^2/2 + sigma x random_number)</li>
          <li>This is repeated for each day to form a price path.</li>
          <li>The process is repeated N times (e.g. 500 or 1000 simulations).</li>
        </ol>
      </div>
      <div style='margin-top:1.5rem; padding:15px; background-color:rgba(255,209,102,0.05); border-left:4px solid #FFD166; border-radius:6px;'>
        <b style='color:#FFD166; font-size:1.1rem;'>Displayed Metrics:</b>
        <ul style='color:#8CA1B6; font-size:0.9rem; line-height:1.7; margin-left:1.2rem; margin-top:0.5rem;'>
          <li><b>Profit Probability:</b> Percentage of simulations that ended above the current price. E.g. 65% means 650 out of 1000 simulations ended profitably.</li>
          <li><b>Expected Return:</b> Average return percentage across all simulations. Positive = profit expected on average.</li>
          <li><b>VaR 95% (Value at Risk):</b> Worst-case loss at 95% confidence. E.g. -15% means 95% of the time you won't lose more than this. The remaining 5% could be worse.</li>
          <li><b>Median Target:</b> The median (middle) ending price of all simulations. Not affected by extreme outliers, more reliable than the mean.</li>
        </ul>
      </div>
      <div style='margin-top:1rem;'>
        <p style='color:#E5E7EB; font-size:0.9rem;'><b>Reading the Charts:</b></p>
        <ul style='color:#8CA1B6; font-size:0.9rem; line-height:1.7; margin-left:1.2rem; margin-top:0.5rem;'>
          <li><b>Simulation Paths:</b> Each thin line is a possible price scenario. Areas where lines cluster are the most likely price ranges.</li>
          <li><b>90% Confidence Band (light blue):</b> Price has a 90% chance of staying within this band.</li>
          <li><b>50% Confidence Band (purple):</b> Price has a 50% chance of staying within this narrower band.</li>
          <li><b>Median line (blue):</b> The most likely middle-ground scenario.</li>
          <li><b>Histogram:</b> Distribution of simulation ending prices. The peak of the bell curve is the most likely ending price.</li>
        </ul>
      </div>
      <p style='color:#E5E7EB; font-size:0.85rem; margin-top:1rem; padding:10px; background-color:rgba(239,71,111,0.1); border-radius:6px;'>
        <b style='color:#EF476F;'>Limitation:</b> Monte Carlo is based on historical volatility. Sudden news, regulations, or market structure changes
        fall outside the model. Treat results as a probability distribution, not an exact price prediction.
      </p>
    </div>
    """
    st.markdown(mc_panel, unsafe_allow_html=True)

    # ── Fibonacci Analysis Tab ──
    fib_panel = """
    <div class='panel-box'>
      <b style='color:#06D6A0; font-size:1.3rem;'>Fibonacci Analysis Tab</b>
      <p style='color:#E5E7EB; font-size:0.95rem; margin-top:1rem; line-height:1.7;'>
        Calculates support/resistance levels using Fibonacci-derived ratios.
        Also performs divergence detection, volume profile analysis, and market regime classification.
      </p>
      <div style='margin-top:1.5rem; padding:15px; background-color:rgba(6,214,160,0.05); border-left:4px solid #06D6A0; border-radius:6px;'>
        <b style='color:#06D6A0; font-size:1.1rem;'>Fibonacci Levels:</b>
        <p style='color:#8CA1B6; font-size:0.9rem; margin-top:0.5rem; line-height:1.7;'>
          Calculated by applying Fibonacci ratios to the range between the highest and lowest price in a given period:
        </p>
        <ul style='color:#8CA1B6; font-size:0.9rem; line-height:1.7; margin-left:1.2rem; margin-top:0.5rem;'>
          <li><b>Retracement Levels:</b>
            <ul style='margin-left:1rem; margin-top:0.3rem;'>
              <li><b>23.6%:</b> Shallow retracement — first support in strong trends</li>
              <li><b>38.2%:</b> Moderate retracement — frequently used support level</li>
              <li><b>50.0%:</b> Halfway point — psychologically significant level</li>
              <li><b>61.8%:</b> "Golden ratio" — strongest Fibonacci level; if price holds here, the trend continues</li>
              <li><b>78.6%:</b> Deep retracement — if broken, the trend may be over</li>
            </ul>
          </li>
          <li style='margin-top:0.5rem;'><b>Extension Levels:</b>
            <ul style='margin-left:1rem; margin-top:0.3rem;'>
              <li><b>127.2%:</b> First target — conservative</li>
              <li><b>161.8%:</b> Second target — golden ratio extension, common take-profit point</li>
              <li><b>200.0%:</b> Third target — reachable in strong trends</li>
              <li><b>261.8%:</b> Aggressive target — rarely reached, requires extreme momentum</li>
            </ul>
          </li>
        </ul>
      </div>
      <div style='margin-top:1.5rem; padding:15px; background-color:rgba(255,209,102,0.05); border-left:4px solid #FFD166; border-radius:6px;'>
        <b style='color:#FFD166; font-size:1.1rem;'>Divergence Detection:</b>
        <p style='color:#8CA1B6; font-size:0.9rem; margin-top:0.5rem; line-height:1.7;'>
          Detects discrepancies between price and technical indicators (RSI, MACD):
        </p>
        <ul style='color:#8CA1B6; font-size:0.9rem; line-height:1.7; margin-left:1.2rem; margin-top:0.5rem;'>
          <li><b>Bullish Divergence:</b> Price makes a lower low while RSI/MACD makes a higher low — bullish signal. The downtrend may be weakening.</li>
          <li><b>Bearish Divergence:</b> Price makes a higher high while RSI/MACD makes a lower high — bearish signal. The uptrend may be weakening.</li>
        </ul>
        <p style='color:#8CA1B6; font-size:0.85rem; margin-top:0.5rem; padding:8px; background-color:rgba(255,255,255,0.03); border-radius:4px;'>
          Divergence is a strong early warning signal but not sufficient for opening a trade on its own. More reliable when combined with Fibonacci levels.
        </p>
      </div>
      <div style='margin-top:1.5rem; padding:15px; background-color:rgba(0,212,255,0.05); border-left:4px solid #00D4FF; border-radius:6px;'>
        <b style='color:#00D4FF; font-size:1.1rem;'>Volume Profile:</b>
        <ul style='color:#8CA1B6; font-size:0.9rem; line-height:1.7; margin-left:1.2rem; margin-top:0.5rem;'>
          <li><b>Definition:</b> A horizontal histogram showing how much trading volume occurred at each price level.</li>
          <li><b>POC (Point of Control):</b> The price level with the highest volume. Acts as strong support/resistance because the most trading occurred here.</li>
          <li><b>Value Area:</b> The price range where 70% of total volume occurred. A breakout from this range suggests a strong move.</li>
        </ul>
      </div>
      <div style='margin-top:1.5rem; padding:15px; background-color:rgba(178,75,243,0.05); border-left:4px solid #B24BF3; border-radius:6px;'>
        <b style='color:#B24BF3; font-size:1.1rem;'>Market Regime:</b>
        <p style='color:#8CA1B6; font-size:0.9rem; margin-top:0.5rem; line-height:1.7;'>
          Automatically classifies the current market state:
        </p>
        <ul style='color:#8CA1B6; font-size:0.9rem; line-height:1.7; margin-left:1.2rem; margin-top:0.5rem;'>
          <li><b>Trending:</b> ADX > 25. Strong directional movement. Trend-following strategies work best here.</li>
          <li><b>Ranging:</b> ADX < 20 and narrow Bollinger bands. Price is confined to a range. Support/resistance strategies are suitable.</li>
          <li><b>High Volatility:</b> ATR above normal. Large price swings, high risk.</li>
          <li><b>Compression:</b> Very narrow Bollinger bands. A big move may be imminent (direction unknown).</li>
        </ul>
      </div>
    </div>
    """
    st.markdown(fib_panel, unsafe_allow_html=True)

    # ── Risk Analytics Tab ──
    risk_panel = """
    <div class='panel-box'>
      <b style='color:#06D6A0; font-size:1.3rem;'>Risk Analytics Tab</b>
      <p style='color:#E5E7EB; font-size:0.95rem; margin-top:1rem; line-height:1.7;'>
        Generates a detailed risk profile for your selected cryptocurrency.
        Calculates the same risk metrics used by professional fund managers.
      </p>
      <div style='margin-top:1.5rem; padding:15px; background-color:rgba(6,214,160,0.05); border-left:4px solid #06D6A0; border-radius:6px;'>
        <b style='color:#06D6A0; font-size:1.1rem;'>Risk Metrics Explained:</b>
        <ul style='color:#8CA1B6; font-size:0.9rem; line-height:1.7; margin-left:1.2rem; margin-top:0.5rem;'>
          <li><b>Sharpe Ratio:</b> Return per unit of risk. Annualized return minus the risk-free rate, divided by volatility.
            <ul style='margin-left:1rem; margin-top:0.3rem;'>
              <li>< 0: Losing money</li>
              <li>0 - 1.0: Low performance</li>
              <li>1.0 - 2.0: Good</li>
              <li>> 2.0: Excellent</li>
            </ul>
          </li>
          <li style='margin-top:0.5rem;'><b>Sortino Ratio:</b> Similar to Sharpe but only considers downside volatility (loss risk).
            Upward fluctuations are not penalized. > 2.0 is considered very good.</li>
          <li style='margin-top:0.5rem;'><b>Calmar Ratio:</b> Annualized return divided by max drawdown.
            Shows how much return is generated per unit of risk. > 3.0 is excellent.</li>
          <li style='margin-top:0.5rem;'><b>Max Drawdown:</b> Largest peak-to-trough loss percentage.
            E.g. -30% means the portfolio lost 30% at its worst. Below 20% is considered good for crypto.</li>
          <li style='margin-top:0.5rem;'><b>VaR 95% (Value at Risk):</b> Worst-case daily loss at 95% confidence.
            E.g. -5% means on 95% of days you won't lose more than this.</li>
          <li style='margin-top:0.5rem;'><b>CVaR 95% (Conditional VaR / Expected Shortfall):</b> Average loss beyond VaR.
            The average loss in the worst 5% of cases. Always worse than VaR and measures "tail risk".</li>
          <li style='margin-top:0.5rem;'><b>Skewness:</b> Whether the return distribution is symmetric.
            <ul style='margin-left:1rem; margin-top:0.3rem;'>
              <li>Negative skew: Long left tail, higher risk of large losses</li>
              <li>Near zero: Symmetric distribution</li>
              <li>Positive skew: Long right tail, potential for large gains</li>
            </ul>
          </li>
          <li style='margin-top:0.5rem;'><b>Kurtosis:</b> Measures the frequency of extreme moves.
            <ul style='margin-left:1rem; margin-top:0.3rem;'>
              <li>> 3 (leptokurtic): More extreme values than normal distribution, higher "black swan" risk</li>
              <li>Near 3 (mesokurtic): Similar to normal distribution</li>
              <li>< 3 (platykurtic): Fewer extreme values, more predictable</li>
            </ul>
          </li>
        </ul>
      </div>
      <div style='margin-top:1rem;'>
        <p style='color:#E5E7EB; font-size:0.9rem;'><b>Reading the Charts:</b></p>
        <ul style='color:#8CA1B6; font-size:0.9rem; line-height:1.7; margin-left:1.2rem; margin-top:0.5rem;'>
          <li><b>Drawdown Chart:</b> Shows how far the portfolio dropped from its peak over time. Deep troughs represent major loss periods.</li>
          <li><b>Cumulative Return Chart:</b> Shows total return since inception. A consistently rising line indicates a healthy investment.</li>
          <li><b>Return Distribution Histogram:</b> Shows how often daily returns fall in each range. Ideally a narrow, symmetric bell curve.</li>
        </ul>
      </div>
    </div>
    """
    st.markdown(risk_panel, unsafe_allow_html=True)

    # ── Whale Tracker Tab ──
    whale_panel = """
    <div class='panel-box'>
      <b style='color:#06D6A0; font-size:1.3rem;'>Whale Tracker Tab</b>
      <p style='color:#E5E7EB; font-size:0.95rem; margin-top:1rem; line-height:1.7;'>
        Tracks major market movements and trends. Uses CoinGecko data to identify
        which coins are popular and which show abnormal volume increases.
      </p>
      <div style='margin-top:1.5rem; padding:15px; background-color:rgba(6,214,160,0.05); border-left:4px solid #06D6A0; border-radius:6px;'>
        <b style='color:#06D6A0; font-size:1.1rem;'>Data Displayed:</b>
        <ul style='color:#8CA1B6; font-size:0.9rem; line-height:1.7; margin-left:1.2rem; margin-top:0.5rem;'>
          <li><b>Trending Coins:</b> Pulled from CoinGecko search trends. Shows the most searched and popular coins. Rising popularity often precedes price action.</li>
          <li><b>Top Gainers:</b> Coins with the highest percentage increase in the last 24 hours. Price, market cap, and change percentage are shown together.</li>
          <li><b>Top Losers:</b> Coins with the largest percentage loss in the last 24 hours. Evaluate alongside volume to distinguish panic selling from healthy correction.</li>
          <li><b>Volume Surge Scanner:</b> Detects coins with above-normal trading volume. Sudden volume spikes may indicate that large players (whales, institutions) are building positions.</li>
        </ul>
      </div>
      <div style='margin-top:1rem;'>
        <p style='color:#E5E7EB; font-size:0.9rem;'><b>How to Use:</b></p>
        <ul style='color:#8CA1B6; font-size:0.9rem; line-height:1.7; margin-left:1.2rem; margin-top:0.5rem;'>
          <li>Watch for coins that appear in Trending consistently but haven't pumped yet — could be an early entry opportunity.</li>
          <li>Don't chase coins in Top Gainers with very high increases (avoid FOMO) — pullbacks usually follow.</li>
          <li>If Volume Surge coincides with price increase, it's a strong signal; if only volume rises, it may be accumulation.</li>
        </ul>
      </div>
      <p style='color:#E5E7EB; font-size:0.85rem; margin-top:1rem; padding:10px; background-color:rgba(255,209,102,0.1); border-radius:6px;'>
        <b style='color:#FFD166;'>Tip:</b> Use Whale Tracker alongside the Market Heatmap.
        The Heatmap shows the big picture, while Whale Tracker highlights specific opportunities.
      </p>
    </div>
    """
    st.markdown(whale_panel, unsafe_allow_html=True)

    # ── Advanced Screener Tab ──
    screener_panel = """
    <div class='panel-box'>
      <b style='color:#06D6A0; font-size:1.3rem;'>Advanced Screener Tab</b>
      <p style='color:#E5E7EB; font-size:0.95rem; margin-top:1rem; line-height:1.7;'>
        Automatically scans multiple cryptocurrencies based on your technical criteria and filters
        to help you find trading opportunities.
      </p>
      <div style='margin-top:1.5rem; padding:15px; background-color:rgba(6,214,160,0.05); border-left:4px solid #06D6A0; border-radius:6px;'>
        <b style='color:#06D6A0; font-size:1.1rem;'>Filter Options:</b>
        <ul style='color:#8CA1B6; font-size:0.9rem; line-height:1.7; margin-left:1.2rem; margin-top:0.5rem;'>
          <li><b>Min Confidence:</b> Only shows coins above this confidence score. E.g. selecting 70% lists only coins with strong signals.</li>
          <li><b>Signal Type:</b> Choose which signal types you are looking for:
            <ul style='margin-left:1rem; margin-top:0.3rem;'>
              <li><b>ALL:</b> All signal types</li>
              <li><b>STRONG BUY:</b> Only very strong buy signals</li>
              <li><b>BUY:</b> Normal buy signals</li>
              <li><b>SELL / STRONG SELL:</b> Sell signals (short opportunities)</li>
            </ul>
          </li>
          <li><b>Timeframe:</b> The period for analysis (5m, 15m, 1h, 4h, 1d).</li>
          <li><b>Coin List:</b> Select coins to scan. By default, the top 20+ most popular coins are scanned.</li>
        </ul>
      </div>
      <div style='margin-top:1rem;'>
        <p style='color:#E5E7EB; font-size:0.9rem;'><b>Reading Results:</b></p>
        <ul style='color:#8CA1B6; font-size:0.9rem; line-height:1.7; margin-left:1.2rem; margin-top:0.5rem;'>
          <li>For each coin: signal direction (BUY/SELL/WAIT), confidence score, price, and 4 category scores are shown.</li>
          <li>Results are sorted by confidence score — strongest opportunities appear at the top.</li>
          <li>A progress bar is shown during scanning. Data is fetched from the API for each coin, so it may take a few minutes.</li>
        </ul>
      </div>
      <p style='color:#E5E7EB; font-size:0.85rem; margin-top:1rem; padding:10px; background-color:rgba(255,209,102,0.1); border-radius:6px;'>
        <b style='color:#FFD166;'>Tip:</b> Running the Screener on 4h or 1d timeframes gives more reliable results.
        Short periods (5m, 15m) tend to produce more noise (false signals).
        Don't forget to inspect found opportunities in the Spot or Fibonacci tabs for a detailed analysis.
      </p>
    </div>
    """
    st.markdown(screener_panel, unsafe_allow_html=True)

    # ── Tab Overview Summary ──
    overview_panel = """
    <div class='panel-box'>
      <b style='color:#06D6A0; font-size:1.3rem;'>All Tabs Overview</b>
      <p style='color:#E5E7EB; font-size:0.95rem; margin-top:1rem; line-height:1.7;'>
        Quick summary of all 17 dashboard tabs:
      </p>
      <div style='margin-top:1rem; display:grid; grid-template-columns:1fr 1fr; gap:10px;'>
        <div style='padding:10px; background:rgba(6,214,160,0.05); border-radius:6px; border-left:3px solid #06D6A0;'>
          <b style='color:#06D6A0; font-size:0.85rem;'>Market</b>
          <p style='color:#8CA1B6; font-size:0.8rem; margin-top:4px;'>Overall market status, Fear & Greed, BTC dominance</p>
        </div>
        <div style='padding:10px; background:rgba(6,214,160,0.05); border-radius:6px; border-left:3px solid #06D6A0;'>
          <b style='color:#06D6A0; font-size:0.85rem;'>Spot</b>
          <p style='color:#8CA1B6; font-size:0.8rem; margin-top:4px;'>Technical analysis, signal generation, confidence score</p>
        </div>
        <div style='padding:10px; background:rgba(6,214,160,0.05); border-radius:6px; border-left:3px solid #06D6A0;'>
          <b style='color:#06D6A0; font-size:0.85rem;'>Position</b>
          <p style='color:#8CA1B6; font-size:0.8rem; margin-top:4px;'>Position management, PnL tracking</p>
        </div>
        <div style='padding:10px; background:rgba(6,214,160,0.05); border-radius:6px; border-left:3px solid #06D6A0;'>
          <b style='color:#06D6A0; font-size:0.85rem;'>AI Prediction</b>
          <p style='color:#8CA1B6; font-size:0.8rem; margin-top:4px;'>Single-model ML prediction (Gradient Boosting)</p>
        </div>
        <div style='padding:10px; background:rgba(0,212,255,0.05); border-radius:6px; border-left:3px solid #00D4FF;'>
          <b style='color:#00D4FF; font-size:0.85rem;'>Multi-TF</b>
          <p style='color:#8CA1B6; font-size:0.8rem; margin-top:4px;'>Simultaneous analysis across 5 timeframes</p>
        </div>
        <div style='padding:10px; background:rgba(0,212,255,0.05); border-radius:6px; border-left:3px solid #00D4FF;'>
          <b style='color:#00D4FF; font-size:0.85rem;'>Correlation</b>
          <p style='color:#8CA1B6; font-size:0.8rem; margin-top:4px;'>Cross-coin correlation analysis</p>
        </div>
        <div style='padding:10px; background:rgba(0,212,255,0.05); border-radius:6px; border-left:3px solid #00D4FF;'>
          <b style='color:#00D4FF; font-size:0.85rem;'>Sessions</b>
          <p style='color:#8CA1B6; font-size:0.8rem; margin-top:4px;'>Asia/Europe/US session analysis</p>
        </div>
        <div style='padding:10px; background:rgba(0,212,255,0.05); border-radius:6px; border-left:3px solid #00D4FF;'>
          <b style='color:#00D4FF; font-size:0.85rem;'>Tools</b>
          <p style='color:#8CA1B6; font-size:0.8rem; margin-top:4px;'>R:R calculator, liquidation levels</p>
        </div>
        <div style='padding:10px; background:rgba(0,212,255,0.05); border-radius:6px; border-left:3px solid #00D4FF;'>
          <b style='color:#00D4FF; font-size:0.85rem;'>Backtest</b>
          <p style='color:#8CA1B6; font-size:0.8rem; margin-top:4px;'>Strategy testing on historical data</p>
        </div>
        <div style='padding:10px; background:rgba(0,212,255,0.05); border-radius:6px; border-left:3px solid #00D4FF;'>
          <b style='color:#00D4FF; font-size:0.85rem;'>Analysis Guide</b>
          <p style='color:#8CA1B6; font-size:0.8rem; margin-top:4px;'>This page — all detailed explanations</p>
        </div>
        <div style='padding:10px; background:rgba(178,75,243,0.05); border-radius:6px; border-left:3px solid #B24BF3;'>
          <b style='color:#B24BF3; font-size:0.85rem;'>Ensemble AI</b>
          <p style='color:#8CA1B6; font-size:0.8rem; margin-top:4px;'>3-model combined ML prediction</p>
        </div>
        <div style='padding:10px; background:rgba(178,75,243,0.05); border-radius:6px; border-left:3px solid #B24BF3;'>
          <b style='color:#B24BF3; font-size:0.85rem;'>Heatmap</b>
          <p style='color:#8CA1B6; font-size:0.8rem; margin-top:4px;'>Top 100 coin treemap view</p>
        </div>
        <div style='padding:10px; background:rgba(178,75,243,0.05); border-radius:6px; border-left:3px solid #B24BF3;'>
          <b style='color:#B24BF3; font-size:0.85rem;'>Monte Carlo</b>
          <p style='color:#8CA1B6; font-size:0.8rem; margin-top:4px;'>Price simulation, probability distribution</p>
        </div>
        <div style='padding:10px; background:rgba(178,75,243,0.05); border-radius:6px; border-left:3px solid #B24BF3;'>
          <b style='color:#B24BF3; font-size:0.85rem;'>Fibonacci</b>
          <p style='color:#8CA1B6; font-size:0.8rem; margin-top:4px;'>Fib levels, divergence, volume profile</p>
        </div>
        <div style='padding:10px; background:rgba(178,75,243,0.05); border-radius:6px; border-left:3px solid #B24BF3;'>
          <b style='color:#B24BF3; font-size:0.85rem;'>Risk Analytics</b>
          <p style='color:#8CA1B6; font-size:0.8rem; margin-top:4px;'>Sharpe, Sortino, VaR, drawdown analysis</p>
        </div>
        <div style='padding:10px; background:rgba(178,75,243,0.05); border-radius:6px; border-left:3px solid #B24BF3;'>
          <b style='color:#B24BF3; font-size:0.85rem;'>Whale Tracker</b>
          <p style='color:#8CA1B6; font-size:0.8rem; margin-top:4px;'>Trending coins, volume surges</p>
        </div>
        <div style='padding:10px; background:rgba(178,75,243,0.05); border-radius:6px; border-left:3px solid #B24BF3;'>
          <b style='color:#B24BF3; font-size:0.85rem;'>Screener</b>
          <p style='color:#8CA1B6; font-size:0.8rem; margin-top:4px;'>Multi-coin scanning, filter-based search</p>
        </div>
      </div>
    </div>
    """
    st.markdown(overview_panel, unsafe_allow_html=True)

def render_ml_tab():
    """
    Render a tab that uses a simple machine‑learning model to predict the
    probability of the next candle closing higher or lower.
    """

    st.markdown(
        f"<h2 style='color:{ACCENT};'>AI Prediction</h2>",
        unsafe_allow_html=True,
    )
    st.markdown(
        f"<p style='color:#8CA1B6;font-size:0.9rem;'>"
        "This tool trains an advanced Gradient Boosting model on recent candles to estimate whether the next candle will close higher or lower. "
        "The output is a probability and a suggested direction (LONG/SHORT/NEUTRAL). "
        "Use this information in conjunction with other analysis; past performance does not guarantee future results.</p>",
        unsafe_allow_html=True
    )
    # Assign a unique key to avoid StreamlitDuplicateElementId errors on AI tab
    coin = _normalize_coin_input(st.text_input(
        "Coin (e.g. BTC, ETH, TAO)",
        value="BTC",
        key="ai_coin_input",
    ))
    # Allow the user to select up to three timeframes to evaluate.  This works
    # similarly to the Position Analyser tab, enabling a multi‑timeframe view
    # of the AI prediction.  By default, we select '1h'.
    selected_timeframes = st.multiselect(
        "Select up to 3 Timeframes", ['1m','3m','5m','15m','1h','4h','1d'], default=['5m'], max_selections=3
    )
    if st.button("Predict", type="primary"):
        _val_err = _validate_coin_symbol(coin)
        if _val_err:
            st.error(_val_err)
            return
        # Use columns to display each timeframe side by side.  One column per
        # selected timeframe.  The panel includes the probability, direction,
        # entry/exit prices, leverage (if applicable) and a detailed list of
        # technical parameters similar to the Position Analyser.
        cols = st.columns(len(selected_timeframes)) if selected_timeframes else [st.empty()]
        for idx, tf in enumerate(selected_timeframes):
            with cols[idx]:
                with st.spinner(f"Fetching data and training model for {tf}..."):
                    df = fetch_ohlcv(coin, tf, limit=500)
                    if df is None or len(df) < 60:
                        st.error(f"Not enough data to train the model for {tf}. Try a different symbol or timeframe.")
                        continue
                    # Prediction using the advanced ML model
                    prob, direction = ml_predict_direction(df)
                    # Determine entry and exit prices using the scalping logic rather than
                    # a fixed ATR formula.  This leverages the analysis results and
                    # confidence metrics to suggest an entry and target if a scalp
                    # opportunity exists.  If no opportunity is detected or the
                    # prediction is neutral, these prices remain unset.
                    entry_price = None
                    exit_price = None

                    # Analysis details
                    try:
                        ar = analyse(df)
                        suggested_lev = ar.leverage if direction != "NEUTRAL" else None
                        volume_spike = ar.volume_spike
                        atr_comment = ar.atr_comment
                        candle_pattern = ar.candle_pattern
                        confidence_score = ar.confidence
                        adx_val = ar.adx
                        supertrend_trend = ar.supertrend
                        ichimoku_trend = ar.ichimoku
                        stochrsi_k_val = ar.stochrsi_k
                        bollinger_bias = ar.bollinger
                        vwap_label = ar.vwap
                        psar_trend = ar.psar
                        williams_label = ar.williams
                        cci_label = ar.cci
                    except Exception:
                        suggested_lev = None
                        volume_spike = False
                        atr_comment = ""
                        candle_pattern = ""
                        confidence_score = 0.0
                        adx_val = np.nan
                        supertrend_trend = ""
                        ichimoku_trend = ""
                        stochrsi_k_val = np.nan
                        bollinger_bias = ""
                        vwap_label = ""
                        psar_trend = ""
                        williams_label = ""
                        cci_label = ""

                    # Compute scalping entry/exit using the built‑in logic.  Only
                    # attempt to compute if the direction is not neutral and a
                    # valid analysis result exists.  This approach lets the ML
                    # model decide whether there is a scalp setup rather than
                    # applying a generic ATR‑based offset.
                    entry_source = None  # track whether entry/exit come from AI or fallback
                    try:
                        scalp_direction, entry_s, target_s, stop_s, rr_ratio, breakout_note = get_scalping_entry_target(
                            df,
                            confidence_score,
                            supertrend_trend,
                            ichimoku_trend,
                            vwap_label,
                            volume_spike,
                            strict_mode=True
                        )
                        # Only use the scalping result if it matches the ML direction.  This avoids
                        # showing long entries for short predictions or vice versa.  Also skip
                        # entries when the prediction is neutral.
                        if (scalp_direction and scalp_direction == direction and direction != "NEUTRAL"):
                            entry_price = entry_s
                            exit_price = target_s
                            entry_source = 'ai'
                    except Exception as e:
                        _debug(f"AI tab scalping entry error: {e}")

                    # Fallback: if no scalping entry was found but the ML direction is
                    # LONG or SHORT, use an ATR‑based offset as a generic entry/exit.
                    if entry_price is None and direction in ["LONG", "SHORT"]:
                        try:
                            prev_close = df['close'].shift(1)
                            tr_fb = pd.concat([
                                (df['high'] - df['low']).abs(),
                                (df['high'] - prev_close).abs(),
                                (df['low'] - prev_close).abs()
                            ], axis=1).max(axis=1)
                            atr_series_fb = tr_fb.rolling(window=14, min_periods=1).mean()
                            atr_val_fb = float(atr_series_fb.iloc[-1]) if len(atr_series_fb) > 0 else 0.0
                            close_price_fb = float(df['close'].iloc[-1])
                            if direction == "LONG":
                                entry_price = close_price_fb + 0.25 * atr_val_fb
                                exit_price = entry_price + 1.5 * atr_val_fb
                            elif direction == "SHORT":
                                entry_price = close_price_fb - 0.25 * atr_val_fb
                                exit_price = entry_price - 1.5 * atr_val_fb
                            entry_source = 'fallback'
                        except Exception:
                            entry_price = None
                            exit_price = None

                    # If the entry is unverified (fallback), do not recommend leverage.  A
                    # fallback entry is considered less trustworthy, so we avoid
                    # suggesting an aggressive leverage value.
                    if entry_source == 'fallback':
                        suggested_lev = None

                    # Compute a combined market‑wide probability for the same timeframe.
                    # We evaluate BTC/USDT, ETH/USDT and major alt pairs (BNB, SOL, ADA, XRP)
                    # over 500 candles and weight their probabilities by current dominance
                    # values.  This captures capital rotation across a wider set of top
                    # assets and improves sensitivity during alt seasons.  If data
                    # retrieval or training fails for any asset, a neutral probability
                    # of 0.5 is used for that asset.
                    try:
                        bundle = get_major_ohlcv_bundle(tf, limit=500)
                        btc_df_tf = bundle.get('BTC/USDT')
                        eth_df_tf = bundle.get('ETH/USDT')
                        bnb_df_tf = bundle.get('BNB/USDT')
                        sol_df_tf = bundle.get('SOL/USDT')
                        ada_df_tf = bundle.get('ADA/USDT')
                        xrp_df_tf = bundle.get('XRP/USDT')
                        # Initialise probabilities to neutral
                        btc_prob_tf = eth_prob_tf = bnb_prob_tf = sol_prob_tf = ada_prob_tf = xrp_prob_tf = 0.5
                        if btc_df_tf is not None and not btc_df_tf.empty:
                            btc_prob_tf, _ = ml_predict_direction(btc_df_tf)
                        if eth_df_tf is not None and not eth_df_tf.empty:
                            eth_prob_tf, _ = ml_predict_direction(eth_df_tf)
                        if bnb_df_tf is not None and not bnb_df_tf.empty:
                            bnb_prob_tf, _ = ml_predict_direction(bnb_df_tf)
                        if sol_df_tf is not None and not sol_df_tf.empty:
                            sol_prob_tf, _ = ml_predict_direction(sol_df_tf)
                        if ada_df_tf is not None and not ada_df_tf.empty:
                            ada_prob_tf, _ = ml_predict_direction(ada_df_tf)
                        if xrp_df_tf is not None and not xrp_df_tf.empty:
                            xrp_prob_tf, _ = ml_predict_direction(xrp_df_tf)
                        # Fetch dominance percentages to weight the probabilities
                        btc_dom_tf, eth_dom_tf, _, _, _, bnb_dom_tf, sol_dom_tf, ada_dom_tf, xrp_dom_tf = get_market_indices()
                        # Compute the sum of all dominances and prevent division by zero
                        dom_sum_tf = btc_dom_tf + eth_dom_tf + bnb_dom_tf + sol_dom_tf + ada_dom_tf + xrp_dom_tf
                        dom_sum_tf = dom_sum_tf if dom_sum_tf > 0 else 1.0
                        mkt_prob_tf = (
                            btc_prob_tf * btc_dom_tf
                            + eth_prob_tf * eth_dom_tf
                            + bnb_prob_tf * bnb_dom_tf
                            + sol_prob_tf * sol_dom_tf
                            + ada_prob_tf * ada_dom_tf
                            + xrp_prob_tf * xrp_dom_tf
                        ) / dom_sum_tf
                    except Exception:
                        mkt_prob_tf = 0.5
                    # Determine combined market direction based on probability
                    if mkt_prob_tf >= 0.6:
                        mkt_dir_tf = "LONG"
                    elif mkt_prob_tf <= 0.4:
                        mkt_dir_tf = "SHORT"
                    else:
                        mkt_dir_tf = "NEUTRAL"

                    # Prediction panel
                    if direction == "LONG":
                        dir_color = POSITIVE
                    elif direction == "SHORT":
                        dir_color = NEGATIVE
                    else:
                        dir_color = WARNING
                    # Determine the latest closing price for display
                    current_price = float(df['close'].iloc[-1]) if not df.empty else 0.0
                    panel_html = (
                        f"<div class='panel-box'>"
                        f"<b style='color:{ACCENT}; font-size:1.2rem;'>AI Prediction ({tf})</b><br>"
                        f"<p style='color:{TEXT_LIGHT}; font-size:1rem; margin-top:6px;'>"
                        f"Current Price: <b>${current_price:,.4f}</b><br>"
                        f"AI Market Outlook: <b>{mkt_prob_tf*100:.1f}%</b> ({mkt_dir_tf})<br>"
                        f"Suggested Direction: <b><span style='color:{dir_color};'>{direction}</span></b>"
                    )
                    # Only show entry/exit suggestions for non‑neutral signals.  If
                    # the ML model signals neutral, entry_price will be None.
                    if entry_price is not None and exit_price is not None:
                        if entry_source == 'ai':
                            # AI‑generated entry/exit are coloured green
                            panel_html += (
                                f"<br><span style='color:{POSITIVE};'>AI Entry (≈):</span> <b>${entry_price:,.4f}</b>"
                                f"<br><span style='color:{POSITIVE};'>AI Exit (≈):</span> <b>${exit_price:,.4f}</b>"
                            )
                        elif entry_source == 'fallback':
                            # Fallback (unverified) entry/exit are coloured orange
                            panel_html += (
                                f"<br><span style='color:{WARNING};'>Unverified Entry (≈):</span> <b>${entry_price:,.4f}</b>"
                                f"<br><span style='color:{WARNING};'>Unverified Exit (≈):</span> <b>${exit_price:,.4f}</b>"
                            )
                    if suggested_lev is not None:
                        panel_html += (
                            f"<br>Suggested Leverage: <b>{suggested_lev}X</b>"
                        )
                    panel_html += ("</p></div>")
                    st.markdown(panel_html, unsafe_allow_html=True)

                    # Indicator grid (same professional layout as other tabs)
                    _grid_html = _build_indicator_grid(
                        supertrend_trend, ichimoku_trend, vwap_label, adx_val, bollinger_bias,
                        stochrsi_k_val, psar_trend, williams_label, cci_label,
                        volume_spike, atr_comment, candle_pattern,
                    )
                    if _grid_html:
                        st.markdown(_grid_html, unsafe_allow_html=True)

                    # === Charts Section ===
                    # Candlestick with EMAs and WMAs
                    try:
                        fig = go.Figure()
                        # Candlestick trace
                        fig.add_trace(go.Candlestick(
                            x=df['timestamp'],
                            open=df['open'], high=df['high'], low=df['low'], close=df['close'],
                            increasing_line_color=POSITIVE, decreasing_line_color=NEGATIVE,
                            name="Price"
                        ))
                        # Exponential moving averages
                        for window, color in [(5, '#F472B6'), (9, '#60A5FA'), (21, '#FBBF24'), (50, '#FCD34D')]:
                            ema_series = ta.trend.ema_indicator(df['close'], window=window)
                            fig.add_trace(go.Scatter(
                                x=df['timestamp'], y=ema_series, mode='lines',
                                name=f"EMA{window}", line=dict(color=color, width=1.5)
                            ))
                        # Weighted moving averages (WMA)
                        try:
                            wma20 = _wma(df['close'], length=20)
                            wma50 = _wma(df['close'], length=50)
                            fig.add_trace(go.Scatter(
                                x=df['timestamp'], y=wma20, mode='lines',
                                name="WMA20", line=dict(color='#34D399', width=1, dash='dot')
                            ))
                            fig.add_trace(go.Scatter(
                                x=df['timestamp'], y=wma50, mode='lines',
                                name="WMA50", line=dict(color='#10B981', width=1, dash='dash')
                            ))
                        except Exception as e:
                            _debug(f"AI tab WMA overlay error: {e}")
                        # Layout settings
                        fig.update_layout(
                            height=380,
                            template='plotly_dark',
                            margin=dict(l=20, r=20, t=30, b=30),
                            xaxis_rangeslider_visible=False,
                            showlegend=True,
                            legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='left', x=0)
                        )
                        st.plotly_chart(fig, width="stretch")
                    except Exception as e:
                        st.warning(f"Could not render price chart: {e}")

                    # RSI chart
                    try:
                        rsi_fig = go.Figure()
                        for period, color in [(6, '#D8B4FE'), (14, '#A78BFA'), (24, '#818CF8')]:
                            rsi_series = ta.momentum.rsi(df['close'], window=period)
                            rsi_fig.add_trace(go.Scatter(
                                x=df['timestamp'], y=rsi_series, mode='lines', name=f"RSI {period}",
                                line=dict(color=color, width=2)
                            ))
                        # Overbought/oversold lines
                        rsi_fig.add_hline(y=70, line=dict(color=NEGATIVE, dash='dot', width=1), name="Overbought")
                        rsi_fig.add_hline(y=30, line=dict(color=POSITIVE, dash='dot', width=1), name="Oversold")
                        rsi_fig.update_layout(
                            height=180,
                            template='plotly_dark',
                            margin=dict(l=20, r=20, t=20, b=30),
                            yaxis=dict(title="RSI"),
                            showlegend=True,
                            legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='left', x=0)
                        )
                        st.plotly_chart(rsi_fig, width="stretch")
                    except Exception as e:
                        _debug(f"RSI chart error: {e}")

                    # MACD chart
                    try:
                        macd_ind = ta.trend.MACD(df['close'])
                        df['macd'] = macd_ind.macd()
                        df['macd_signal'] = macd_ind.macd_signal()
                        df['macd_diff'] = macd_ind.macd_diff()
                        macd_fig = go.Figure()
                        macd_fig.add_trace(go.Scatter(
                            x=df['timestamp'], y=df['macd'], name="MACD",
                            line=dict(color=ACCENT, width=2)
                        ))
                        macd_fig.add_trace(go.Scatter(
                            x=df['timestamp'], y=df['macd_signal'], name="Signal",
                            line=dict(color=WARNING, width=2, dash='dot')
                        ))
                        macd_fig.add_trace(go.Bar(
                            x=df['timestamp'], y=df['macd_diff'], name="Histogram",
                            marker_color=CARD_BG
                        ))
                        macd_fig.update_layout(
                            height=200,
                            template='plotly_dark',
                            margin=dict(l=20, r=20, t=20, b=30),
                            yaxis=dict(title="MACD"),
                            showlegend=True,
                            legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='left', x=0)
                        )
                        st.plotly_chart(macd_fig, width="stretch")
                    except Exception as e:
                        _debug(f"MACD chart error: {e}")

                    # Volume & OBV chart
                    try:
                        df['obv'] = ta.volume.on_balance_volume(df['close'], df['volume'])
                        volume_fig = go.Figure()
                        volume_fig.add_trace(go.Bar(
                            x=df['timestamp'], y=df['volume'], name="Volume", marker_color="#6B7280"
                        ))
                        volume_fig.add_trace(go.Scatter(
                            x=df['timestamp'], y=df['obv'], name="OBV",
                            line=dict(color=WARNING, width=1.5, dash='dot'),
                            yaxis='y2'
                        ))
                        volume_fig.update_layout(
                            height=180,
                            template='plotly_dark',
                            margin=dict(l=20, r=20, t=20, b=30),
                            yaxis=dict(title="Volume"),
                            yaxis2=dict(overlaying='y', side='right', title='OBV', showgrid=False),
                            showlegend=True,
                            legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='left', x=0)
                        )
                        st.plotly_chart(volume_fig, width="stretch")
                    except Exception as e:
                        _debug(f"Volume/OBV chart error: {e}")

                    # Technical Snapshot
                    try:
                        # Compute indicators for snapshot
                        df['ema9'] = ta.trend.ema_indicator(df['close'], window=9)
                        df['ema21'] = ta.trend.ema_indicator(df['close'], window=21)
                        df['rsi14'] = ta.momentum.rsi(df['close'], window=14)
                        latest_snap = df.iloc[-1]
                        ema9_val = latest_snap['ema9']
                        ema21_val = latest_snap['ema21']
                        macd_val = df['macd'].iloc[-1] if 'macd' in df else 0.0
                        rsi_val = latest_snap['rsi14']
                        # OBV change over last 5 candles
                        obv_change = 0.0
                        if 'obv' in df and len(df['obv']) > 5 and df['obv'].iloc[-5] != 0:
                            obv_change = (df['obv'].iloc[-1] - df['obv'].iloc[-5]) / abs(df['obv'].iloc[-5]) * 100
                        # Support/resistance (adaptive lookback)
                        recent = df.tail(_sr_lookback())
                        support = recent['low'].min()
                        resistance = recent['high'].max()
                        current_price_snap = latest_snap['close']
                        support_dist = abs(current_price_snap - support) / current_price_snap * 100 if current_price_snap else 0.0
                        resistance_dist = abs(current_price_snap - resistance) / current_price_snap * 100 if current_price_snap else 0.0
                        snapshot_html = f"""
                        <div class='panel-box'>
                          <b style='color:{ACCENT}; font-size:1.05rem;'>📊 Technical Snapshot</b><br>
                          <ul style='color:{TEXT_MUTED}; font-size:0.9rem; line-height:1.5; list-style-position:inside; margin-top:6px;'>
                            <li>EMA Trend (9 vs 21): <b>{ema9_val:.2f}</b> vs <b>{ema21_val:.2f}</b> {('🟢' if ema9_val > ema21_val else '🔴')} — When EMA9 is above EMA21 the short‑term trend is bullish; otherwise bearish.</li>
                            <li>MACD: <b>{macd_val:.2f}</b> {('🟢' if macd_val > 0 else '🔴')} — Positive MACD indicates upward momentum; negative values suggest downward pressure.</li>
                            <li>RSI (14): <b>{rsi_val:.2f}</b> {('🟢' if rsi_val > 55 else ('🟠' if 45 <= rsi_val <= 55 else '🔴'))} — Above 70 may signal overbought, below 30 oversold. Values above 50 favour bulls.</li>
                            <li>OBV change (last 5 candles): <b>{obv_change:+.2f}%</b> {('🟢' if obv_change > 0 else '🔴')} — Rising OBV supports the price move; falling OBV warns against continuation.</li>
                            <li>Support / Resistance: support at <b>${support:,.2f}</b> ({support_dist:.2f}% away), resistance at <b>${resistance:,.2f}</b> ({resistance_dist:.2f}% away).</li>
                          </ul>
                        </div>
                        """
                        st.markdown(snapshot_html, unsafe_allow_html=True)
                    except Exception as e:
                        st.warning(f"Could not render technical snapshot: {e}")

def run_backtest(df: pd.DataFrame, threshold: float = 70, exit_after: int = 5,
                 commission: float = 0.001, slippage: float = 0.0005) -> tuple[pd.DataFrame, str]:
    """
    BACKTEST ENGINE with comprehensive metrics.

    Args:
        df: OHLCV dataframe
        threshold: Minimum confidence score to take trades
        exit_after: Number of candles to hold position
        commission: Trading commission per side (default 0.1% = 0.001)
        slippage: Simulated slippage per trade (default 0.05% = 0.0005)

    Returns:
        results_df: DataFrame with all trades
        summary_html: HTML summary with metrics
    """
    results = []
    equity_curve = [10000]  # Starting capital
    peak = 10000
    max_drawdown = 0
    consecutive_wins = 0
    consecutive_losses = 0
    max_consecutive_losses = 0
    # Use rolling window instead of full copy for performance
    window_size = 200

    for i in range(55, len(df) - exit_after):
        start_idx = max(0, i - window_size)
        df_slice = df.iloc[start_idx:i + 1]
        if len(df_slice) < 55:
            continue
        try:
            result = analyse(df_slice)
            raw_signal = result.signal
            conf_score = result.confidence
        except Exception:
            continue

        sig_plain = "LONG" if raw_signal in ["STRONG BUY", "BUY"] else \
                    "SHORT" if raw_signal in ["STRONG SELL", "SELL"] else "WAIT"

        if conf_score >= threshold and sig_plain in ["LONG", "SHORT"]:
            entry_price = df['close'].iloc[i]
            future_price = df['close'].iloc[i + exit_after]
            # Total cost = commission (entry + exit) + slippage (entry + exit)
            total_cost = 2 * commission + 2 * slippage

            if sig_plain == "LONG":
                pnl = ((future_price - entry_price) / entry_price - total_cost) * 100
            else:  # SHORT
                pnl = ((entry_price - future_price) / entry_price - total_cost) * 100
            
            # Update equity curve
            equity = equity_curve[-1] * (1 + pnl / 100)
            equity_curve.append(equity)
            
            # Track drawdown
            peak = max(peak, equity)
            drawdown = (peak - equity) / peak * 100
            max_drawdown = max(max_drawdown, drawdown)
            
            # Track consecutive wins/losses
            if pnl > 0:
                consecutive_wins += 1
                consecutive_losses = 0
            else:
                consecutive_losses += 1
                consecutive_wins = 0
                max_consecutive_losses = max(max_consecutive_losses, consecutive_losses)

            results.append({
                "Date": df['timestamp'].iloc[i],
                "Confidence": round(conf_score, 1),
                "Signal": sig_plain,          # LONG / SHORT
                "Entry": entry_price,
                "Exit": future_price,
                "PnL (%)": round(pnl, 2),
                "Equity": round(equity, 2)
            })

    df_results = pd.DataFrame(results)

    if df_results.empty:
        summary_html = (
            "<div style='color:#FFB000;margin-top:1rem;'>"
            "<p><b>⚠️ No Signals:</b> No trades met the threshold criteria</p>"
            "<p>Try lowering the confidence threshold or using more data</p>"
            "</div>"
        )
        return df_results, summary_html

    # Calculate comprehensive metrics
    wins = (df_results["PnL (%)"] > 0).sum()
    losses = (df_results["PnL (%)"] <= 0).sum()
    total_trades = wins + losses
    winrate = (wins / total_trades) * 100 if total_trades > 0 else 0.0
    
    # Profit factor (gross profit / gross loss)
    gross_profit = df_results[df_results["PnL (%)"] > 0]["PnL (%)"].sum()
    gross_loss = abs(df_results[df_results["PnL (%)"] <= 0]["PnL (%)"].sum())
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
    
    # Average win/loss
    avg_win = df_results[df_results["PnL (%)"] > 0]["PnL (%)"].mean() if wins > 0 else 0
    avg_loss = df_results[df_results["PnL (%)"] <= 0]["PnL (%)"].mean() if losses > 0 else 0
    
    # Total return
    total_return = (equity_curve[-1] - equity_curve[0]) / equity_curve[0] * 100
    
    # Sharpe ratio (annualized - assuming 365 trading periods per year)
    returns = df_results["PnL (%)"].astype(float) / 100.0
    mean_return = returns.mean()
    std_return = returns.std()
    sharpe_ratio = (mean_return / (std_return + 1e-9)) * np.sqrt(365 / exit_after) if std_return > 0 else 0

    # Color coding for metrics
    wr_color = POSITIVE if winrate >= 55 else WARNING if winrate >= 45 else NEGATIVE
    pf_color = POSITIVE if profit_factor >= 1.5 else WARNING if profit_factor >= 1.0 else NEGATIVE
    dd_color = POSITIVE if max_drawdown < 15 else WARNING if max_drawdown < 25 else NEGATIVE
    ret_color = POSITIVE if total_return > 0 else NEGATIVE
    sr_color = POSITIVE if sharpe_ratio > 1.0 else WARNING if sharpe_ratio > 0.5 else NEGATIVE

    # Use simpler HTML without f-string color variables - use inline colors
    summary_html = f"""
    <div style='margin-top:1rem; background-color:#16213E; padding:20px; border-radius:10px;'>
        <h3 style='color:#06D6A0; margin-top:0;'>📊 Backtest Results</h3>
        
        <div style='display: grid; grid-template-columns: repeat(3, 1fr); gap: 15px; margin-top:15px;'>
            <div style='text-align:center;'>
                <p style='color:#8CA1B6; margin:0; font-size:0.85rem;'>TOTAL TRADES</p>
                <p style='color:#06D6A0; margin:5px 0; font-size:1.8rem; font-weight:600;'>{total_trades}</p>
            </div>
            
            <div style='text-align:center;'>
                <p style='color:#8CA1B6; margin:0; font-size:0.85rem;'>WIN RATE</p>
                <p style='color:{wr_color}; margin:5px 0; font-size:1.8rem; font-weight:600;'>{winrate:.1f}%</p>
                <p style='color:#8CA1B6; margin:0; font-size:0.75rem;'>({wins}W / {losses}L)</p>
            </div>
            
            <div style='text-align:center;'>
                <p style='color:#8CA1B6; margin:0; font-size:0.85rem;'>PROFIT FACTOR</p>
                <p style='color:{pf_color}; margin:5px 0; font-size:1.8rem; font-weight:600;'>{profit_factor:.2f}</p>
                <p style='color:#8CA1B6; margin:0; font-size:0.75rem;'>Target: ≥1.5</p>
            </div>
            
            <div style='text-align:center;'>
                <p style='color:#8CA1B6; margin:0; font-size:0.85rem;'>TOTAL RETURN</p>
                <p style='color:{ret_color}; margin:5px 0; font-size:1.8rem; font-weight:600;'>{total_return:+.2f}%</p>
                <p style='color:#8CA1B6; margin:0; font-size:0.75rem;'>${equity_curve[0]:.0f} → ${equity_curve[-1]:.0f}</p>
            </div>
            
            <div style='text-align:center;'>
                <p style='color:#8CA1B6; margin:0; font-size:0.85rem;'>MAX DRAWDOWN</p>
                <p style='color:{dd_color}; margin:5px 0; font-size:1.8rem; font-weight:600;'>{max_drawdown:.2f}%</p>
                <p style='color:#8CA1B6; margin:0; font-size:0.75rem;'>Target: &lt;15%</p>
            </div>
            
            <div style='text-align:center;'>
                <p style='color:#8CA1B6; margin:0; font-size:0.85rem;'>SHARPE RATIO</p>
                <p style='color:{sr_color}; margin:5px 0; font-size:1.8rem; font-weight:600;'>{sharpe_ratio:.2f}</p>
                <p style='color:#8CA1B6; margin:0; font-size:0.75rem;'>Target: &gt;1.0</p>
            </div>
        </div>
        
        <div style='margin-top:20px; padding-top:20px; border-top:1px solid rgba(255,255,255,0.1);'>
            <div style='display: grid; grid-template-columns: repeat(3, 1fr); gap:10px;'>
                <div>
                    <p style='color:#8CA1B6; margin:0; font-size:0.8rem;'>Avg Win</p>
                    <p style='color:#06D6A0; margin:5px 0; font-weight:600;'>{avg_win:+.2f}%</p>
                </div>
                <div>
                    <p style='color:#8CA1B6; margin:0; font-size:0.8rem;'>Avg Loss</p>
                    <p style='color:#EF476F; margin:5px 0; font-weight:600;'>{avg_loss:.2f}%</p>
                </div>
                <div>
                    <p style='color:#8CA1B6; margin:0; font-size:0.8rem;'>Max Consecutive Losses</p>
                    <p style='color:#FFD166; margin:5px 0; font-weight:600;'>{max_consecutive_losses}</p>
                </div>
            </div>
        </div>
        
        <div style='margin-top:15px; padding:10px; background-color:rgba(255,255,255,0.05); border-radius:5px;'>
            <p style='color:#8CA1B6; margin:0; font-size:0.75rem;'>
                ℹ️ Commission included: {commission*100:.2f}% per trade (entry + exit)
            </p>
        </div>
    </div>
    """

    return df_results, summary_html


def render_backtest_tab():
    """Render the Backtest tab to simulate past signals."""
    st.markdown(f"<h2 style='color:{ACCENT};'>Backtest Simulator</h2>", unsafe_allow_html=True)
    st.markdown(
        f"<div class='panel-box' style='margin-bottom:1rem;'>"
        f"<b style='color:{ACCENT};'>How the backtest works:</b>"
        f"<ul style='color:{TEXT_MUTED}; font-size:0.88rem; line-height:1.7; margin-top:0.5rem;'>"
        "<li>The engine slides a window through historical candles and runs the <b>full technical analysis</b> "
        "(EMA, RSI, MACD, SuperTrend, Ichimoku, Bollinger, ADX, etc.) at each step.</li>"
        "<li>When the <b>Signal</b> is LONG or SHORT <b>and</b> the <b>Confidence Score</b> exceeds your threshold, "
        "a simulated trade is opened at the closing price.</li>"
        "<li>The trade is automatically closed after <b>N candles</b> (your exit setting). "
        "Commission is deducted on both entry and exit.</li>"
        "<li>This tests whether the dashboard's signal + confidence system would have been profitable "
        "on the chosen coin and timeframe. Use it to calibrate your confidence threshold and hold duration.</li>"
        "</ul></div>",
        unsafe_allow_html=True,
    )

    col1, col2 = st.columns(2)
    
    with col1:
        coin = _normalize_coin_input(st.text_input(
            "Coin (e.g. BTC, ETH, TAO)",
            value="BTC",
            key="backtest_coin_input",
        ))
        timeframe = st.selectbox("Timeframe", ["3m", "5m", "15m", "1h", "4h", "1d"], index=2)
        limit = st.slider("Number of Candles", 100, 1000, step=100, value=500)
    
    with col2:
        threshold = st.slider("Confidence Threshold (%)", 50, 90, step=5, value=65,
                             help="Only take trades with confidence above this level")
        exit_after = st.slider("Exit After N Candles", 1, 20, step=1, value=5)
        commission = st.slider("Commission (%)", 0.0, 1.0, step=0.05, value=0.1,
                              help="Trading fee per trade (typical spot: 0.1%)") / 100
        slippage = st.slider("Slippage (%)", 0.0, 0.5, step=0.01, value=0.05,
                            help="Simulated slippage per trade (market impact + spread)") / 100

    if st.button("🚀 Run Backtest", type="primary"):
        _val_err = _validate_coin_symbol(coin)
        if _val_err:
            st.error(_val_err)
            return
        st.info("Fetching data and running comprehensive analysis...")

        df = fetch_ohlcv(coin, timeframe, limit)

        if df is not None and not df.empty:
            st.success(f"✅ Fetched {len(df)} candles. Running backtest...")

            try:
                result_df, summary_html = run_backtest(df, threshold, exit_after, commission, slippage)

                if not result_df.empty:
                    # Display metrics using Streamlit native components instead of HTML
                    st.markdown("### 📊 Backtest Results")
                    
                    # Calculate metrics from result_df
                    total_trades = len(result_df)
                    wins = len(result_df[result_df["PnL (%)"] > 0])
                    losses = total_trades - wins
                    winrate = (wins / total_trades * 100) if total_trades > 0 else 0
                    
                    gross_profit = result_df[result_df["PnL (%)"] > 0]["PnL (%)"].sum()
                    gross_loss = abs(result_df[result_df["PnL (%)"] <= 0]["PnL (%)"].sum())
                    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
                    
                    # Max drawdown from equity curve
                    if 'Equity' in result_df.columns:
                        equity = result_df['Equity'].values
                        peak = equity[0]
                        max_dd = 0
                        for val in equity:
                            if val > peak:
                                peak = val
                            dd = (peak - val) / peak * 100
                            if dd > max_dd:
                                max_dd = dd
                        max_drawdown = max_dd
                        total_return = (equity[-1] - equity[0]) / equity[0] * 100
                    else:
                        max_drawdown = 0
                        total_return = 0
                    
                    # Sharpe ratio
                    returns = result_df["PnL (%)"].astype(float) / 100.0
                    mean_return = returns.mean()
                    std_return = returns.std()
                    sharpe_ratio = (mean_return / (std_return + 1e-9)) * np.sqrt(365 / exit_after) if std_return > 0 else 0
                    
                    avg_win = result_df[result_df["PnL (%)"] > 0]["PnL (%)"].mean() if wins > 0 else 0
                    avg_loss = result_df[result_df["PnL (%)"] <= 0]["PnL (%)"].mean() if losses > 0 else 0
                    
                    # Display in columns
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Total Trades", total_trades)
                        st.metric("Avg Win", f"{avg_win:+.2f}%")
                    with col2:
                        st.metric("Win Rate", f"{winrate:.1f}%", f"{wins}W / {losses}L")
                        st.metric("Avg Loss", f"{avg_loss:.2f}%")
                    with col3:
                        st.metric("Profit Factor", f"{profit_factor:.2f}", "Target: ≥1.5")
                        st.metric("Costs", f"{commission*100:.2f}% + {slippage*100:.2f}%", "comm + slip")
                    
                    col4, col5, col6 = st.columns(3)
                    with col4:
                        st.metric("Total Return", f"{total_return:+.2f}%")
                    with col5:
                        st.metric("Max Drawdown", f"{max_drawdown:.2f}%", "Target: <15%")
                    with col6:
                        st.metric("Sharpe Ratio", f"{sharpe_ratio:.2f}", "Target: >1.0")
                    
                    
                    # Equity Curve Chart
                    if 'Equity' in result_df.columns:
                        st.markdown(f"<h3 style='color:{ACCENT}; margin-top:2rem;'>💰 Equity Curve</h3>", unsafe_allow_html=True)
                        equity_fig = go.Figure()
                        equity_fig.add_trace(go.Scatter(
                            x=result_df['Date'],
                            y=result_df['Equity'],
                            mode='lines',
                            name='Equity',
                            line=dict(color=POSITIVE, width=2),
                            fill='tozeroy',
                            fillcolor='rgba(6, 214, 160, 0.1)'
                        ))
                        equity_fig.add_hline(y=10000, line=dict(color=WARNING, dash='dash', width=1), 
                                           annotation_text="Starting Capital")
                        equity_fig.update_layout(
                            template='plotly_dark',
                            height=300,
                            xaxis_title='Date',
                            yaxis_title='Equity ($)',
                            showlegend=False,
                            hovermode='x unified'
                        )
                        st.plotly_chart(equity_fig, width="stretch")

                    # Trade History Table
                    st.markdown(f"<h3 style='color:{ACCENT}; margin-top:2rem;'>📜 Trade History</h3>", unsafe_allow_html=True)
                    
                    # Format display
                    styled_df = result_df.copy()
                    styled_df['Date'] = styled_df['Date'].dt.strftime('%Y-%m-%d %H:%M')
                    styled_df["Entry"] = styled_df["Entry"].apply(lambda x: f"${x:,.4f}")
                    styled_df["Exit"] = styled_df["Exit"].apply(lambda x: f"${x:,.4f}")
                    styled_df["PnL (%)"] = styled_df["PnL (%)"].apply(lambda x: f"{x:+.2f}%")
                    styled_df["Confidence"] = styled_df["Confidence"].apply(lambda x: f"{x:.1f}%")
                    if 'Equity' in styled_df.columns:
                        styled_df["Equity"] = styled_df["Equity"].apply(lambda x: f"${x:,.2f}")
                    

                    st.dataframe(styled_df, width="stretch")

                    # Offer a download button for the raw results
                    csv_bytes = result_df.to_csv(index=False).encode('utf-8')
                    filename = f"{coin.replace('/', '_')}_{timeframe}_backtest.csv"
                    st.download_button(
                        label="Download Results (CSV)",
                        data=csv_bytes,
                        file_name=filename,
                        mime="text/csv"
                    )

                else:
                    st.warning("No signals generated for the given threshold.")

            except Exception as e:
                st.error(f"Error during backtest: {e}")

        else:
            st.error("Failed to fetch historical data. Please check the symbol or connection.")


def render_multitf_tab():
    """Multi-timeframe confluence analysis."""
    st.markdown(f"<h2 style='color:{ACCENT};'>Multi-Timeframe Confluence</h2>", unsafe_allow_html=True)
    st.markdown(
        f"<p style='color:{TEXT_MUTED}; font-size:0.9rem;'>"
        "Analyse signals across 5m, 15m, 1h, 4h, 1d and combine into a single confluence score. "
        "Higher confluence = more timeframes agree on direction."
        "</p>",
        unsafe_allow_html=True,
    )
    coin = _normalize_coin_input(st.text_input("Coin (e.g. BTC, ETH, TAO)", value="BTC", key="mtf_coin_input"))
    if st.button("Run Multi-TF Analysis", type="primary"):
        _val_err = _validate_coin_symbol(coin)
        if _val_err:
            st.error(_val_err)
            return
        timeframes = ["5m", "15m", "1h", "4h", "1d"]
        with st.spinner("Analysing across all timeframes..."):
            rows = []
            for tf in timeframes:
                df = fetch_ohlcv(coin, tf, limit=200)
                if df is None or len(df) < 55:
                    rows.append({"Timeframe": tf, "Signal": "NO DATA", "Confidence": 0.0,
                                 "SuperTrend": "", "Ichimoku": "", "VWAP": "", "ADX": 0.0})
                    continue
                ar = analyse(df)
                rows.append({
                    "Timeframe": tf,
                    "Signal": signal_plain(ar.signal),
                    "Confidence": round(ar.confidence, 1),
                    "SuperTrend": format_trend(ar.supertrend),
                    "Ichimoku": format_trend(ar.ichimoku),
                    "VWAP": ar.vwap,
                    "ADX": round(ar.adx, 1),
                })

            # Confluence calculation
            valid = [r for r in rows if r["Signal"] != "NO DATA"]
            long_c = sum(1 for r in valid if r["Signal"] == "LONG")
            short_c = sum(1 for r in valid if r["Signal"] == "SHORT")
            total_valid = len(valid)
            avg_conf = np.mean([r["Confidence"] for r in valid]) if valid else 0.0

            if total_valid > 0:
                confluence_pct = max(long_c, short_c) / total_valid * 100
                dominant = "LONG" if long_c > short_c else ("SHORT" if short_c > long_c else "NEUTRAL")
            else:
                confluence_pct = 0
                dominant = "NEUTRAL"

            # Confluence gauge
            conf_color = POSITIVE if dominant == "LONG" else (NEGATIVE if dominant == "SHORT" else WARNING)
            fig_conf = go.Figure(go.Indicator(
                mode="gauge+number",
                value=round(confluence_pct),
                gauge={
                    "axis": {"range": [0, 100], "tickwidth": 1},
                    "bar": {"color": conf_color},
                    "bgcolor": CARD_BG,
                    "steps": [
                        {"range": [0, 40], "color": NEGATIVE},
                        {"range": [40, 60], "color": WARNING},
                        {"range": [60, 100], "color": POSITIVE},
                    ],
                },
                title={"text": f"Confluence ({dominant})", "font": {"size": 16, "color": ACCENT}},
                number={"font": {"color": ACCENT, "size": 38}, "suffix": "%"},
            ))
            fig_conf.update_layout(
                height=200, margin=dict(l=10, r=10, t=50, b=15),
                plot_bgcolor="#0e1117", paper_bgcolor="#0e1117",
            )
            st.plotly_chart(fig_conf, width="stretch")

            # Summary cards
            c1, c2, c3 = st.columns(3)
            with c1:
                st.markdown(
                    f"<div class='metric-card'><div class='metric-label'>Dominant Direction</div>"
                    f"<div class='metric-value' style='color:{conf_color};'>{dominant}</div></div>",
                    unsafe_allow_html=True,
                )
            with c2:
                st.markdown(
                    f"<div class='metric-card'><div class='metric-label'>TFs Agreeing</div>"
                    f"<div class='metric-value'>{max(long_c, short_c)}/{total_valid}</div></div>",
                    unsafe_allow_html=True,
                )
            with c3:
                st.markdown(
                    f"<div class='metric-card'><div class='metric-label'>Avg Confidence</div>"
                    f"<div class='metric-value'>{avg_conf:.1f}%</div></div>",
                    unsafe_allow_html=True,
                )

            # Detail table
            df_mtf = pd.DataFrame(rows)
            styled_mtf = (
                df_mtf.style
                .map(style_signal, subset=["Signal"])
            )
            st.dataframe(styled_mtf, width="stretch")

            # Recommendation
            if confluence_pct >= 80 and dominant != "NEUTRAL":
                st.success(f"Strong confluence: {int(confluence_pct)}% of timeframes agree on {dominant}. High-probability setup.")
            elif confluence_pct >= 60 and dominant != "NEUTRAL":
                st.info(f"Moderate confluence: {int(confluence_pct)}% agree on {dominant}. Consider with caution.")
            else:
                st.warning("Weak confluence. Timeframes are mixed — wait for better alignment.")


def render_correlation_tab():
    """Render a correlation matrix for major crypto assets."""
    st.markdown(f"<h2 style='color:{ACCENT};'>Correlation Matrix</h2>", unsafe_allow_html=True)
    st.markdown(
        f"<p style='color:{TEXT_MUTED}; font-size:0.9rem;'>"
        "Shows how major crypto assets move relative to each other. "
        "High correlation (near 1.0) = they move together. Low/negative = they diverge."
        "</p>",
        unsafe_allow_html=True,
    )
    corr_c1, corr_c2 = st.columns(2)
    with corr_c1:
        tf_corr = st.selectbox("Timeframe", ["15m", "1h", "4h", "1d"], index=3, key="corr_tf")
    with corr_c2:
        custom_coins_raw = st.text_input(
            "Add coins (up to 4, comma-separated)",
            value="",
            placeholder="e.g. DOGE, TAO, LINK, FET",
            key="corr_custom_coin",
        ).upper().strip()
    if st.button("Generate Correlation Matrix", type="primary"):
        symbols = ["BTC/USDT", "ETH/USDT", "BNB/USDT", "SOL/USDT", "ADA/USDT", "XRP/USDT"]
        # Parse comma-separated custom coins (up to 4)
        if custom_coins_raw:
            custom_list = [c.strip() for c in custom_coins_raw.split(",") if c.strip()][:4]
            for cc in custom_list:
                normalized = _normalize_coin_input(cc)
                if normalized and normalized not in symbols:
                    symbols.append(normalized)
        labels = [s.split("/")[0] for s in symbols]
        with st.spinner("Fetching data for correlation analysis..."):
            returns_dict = {}
            failed_coins = []
            for sym, label in zip(symbols, labels):
                df = fetch_ohlcv(sym, tf_corr, limit=200)
                if df is not None and len(df) > 10:
                    returns_dict[label] = df["close"].pct_change().dropna().values
                else:
                    failed_coins.append(sym)
            if failed_coins:
                st.warning(
                    f"Could not fetch data for: **{', '.join(failed_coins)}**. "
                    f"These coins were not found on {EXCHANGE.id.title()} or CoinGecko. "
                    f"Please check the symbol is correct."
                )
            if len(returns_dict) < 2:
                st.error("Not enough data to compute correlations.")
                return
            # Align lengths
            min_len = min(len(v) for v in returns_dict.values())
            aligned = {k: v[-min_len:] for k, v in returns_dict.items()}
            df_ret = pd.DataFrame(aligned)
            corr = df_ret.corr()

            # Heatmap
            fig_corr = go.Figure(data=go.Heatmap(
                z=corr.values,
                x=corr.columns.tolist(),
                y=corr.index.tolist(),
                colorscale="RdYlGn",
                zmin=-1, zmax=1,
                text=np.round(corr.values, 2),
                texttemplate="%{text}",
                textfont={"size": 14},
            ))
            fig_corr.update_layout(
                height=450,
                template="plotly_dark",
                margin=dict(l=60, r=20, t=30, b=60),
                xaxis=dict(side="bottom"),
            )
            st.plotly_chart(fig_corr, width="stretch")

            # Insights
            pairs = []
            cols_list = list(corr.columns)
            for i in range(len(cols_list)):
                for j in range(i + 1, len(cols_list)):
                    pairs.append((cols_list[i], cols_list[j], corr.iloc[i, j]))
            pairs.sort(key=lambda x: x[2])
            st.markdown(f"**Most correlated:** {pairs[-1][0]}-{pairs[-1][1]} ({pairs[-1][2]:.2f})")
            st.markdown(f"**Least correlated:** {pairs[0][0]}-{pairs[0][1]} ({pairs[0][2]:.2f})")
            if any(p[2] < 0.3 for p in pairs):
                low_pairs = [f"{p[0]}-{p[1]}" for p in pairs if p[2] < 0.3]
                st.info(f"Diversification opportunity: {', '.join(low_pairs[:3])} have low correlation.")


def render_sessions_tab():
    """Session-based analysis: Asian, European, US sessions."""
    st.markdown(f"<h2 style='color:{ACCENT};'>Session Analysis</h2>", unsafe_allow_html=True)
    st.markdown(
        f"<p style='color:{TEXT_MUTED}; font-size:0.9rem;'>"
        "Analyse trading activity across global sessions: "
        "Asian (00:00-08:00 UTC), European (08:00-16:00 UTC), US (16:00-00:00 UTC). "
        "Find which session has the most volume, volatility, and best conditions."
        "</p>",
        unsafe_allow_html=True,
    )
    # Explain key metrics so users understand the results
    st.markdown(
        f"<div class='panel-box' style='margin-bottom:1rem;'>"
        f"<b style='color:{ACCENT};'>What the metrics mean:</b>"
        f"<ul style='color:{TEXT_MUTED}; font-size:0.88rem; line-height:1.7; margin-top:0.5rem;'>"
        "<li><b>Volume:</b> The total amount of the asset traded during that session. "
        "Higher volume = more active market, tighter spreads, and easier to enter/exit trades.</li>"
        "<li><b>Avg Range (%):</b> The average difference between the highest and lowest price within each candle, "
        "expressed as a percentage. Higher range = more volatility = bigger potential profits but also bigger risk.</li>"
        "<li><b>Avg Return (%):</b> The average percentage change from open to close per candle. "
        "Positive = price tends to go up during this session. Negative = price tends to go down.</li>"
        "</ul></div>",
        unsafe_allow_html=True,
    )
    coin_s = _normalize_coin_input(st.text_input("Coin (e.g. BTC, ETH, TAO)", value="BTC", key="session_coin_input"))
    if st.button("Analyse Sessions", type="primary"):
        _val_err = _validate_coin_symbol(coin_s)
        if _val_err:
            st.error(_val_err)
            return
        with st.spinner("Fetching hourly data for session analysis..."):
            df = fetch_ohlcv(coin_s, "1h", limit=500)
            if df is None:
                st.error(
                    f"**{coin_s}** could not be found on **{EXCHANGE.id.title()}** or CoinGecko. "
                    f"Please check the symbol and try again."
                )
                return
            if len(df) < 48:
                st.error(
                    f"Only {len(df)} hourly candles available for **{coin_s}** (need at least 48). "
                    f"This coin may have limited history."
                )
                return

            df["hour"] = df["timestamp"].dt.hour
            df["range_pct"] = (df["high"] - df["low"]) / df["low"] * 100
            df["return_pct"] = df["close"].pct_change() * 100

            def _session(h: int) -> str:
                if 0 <= h < 8:
                    return "Asian (00-08 UTC)"
                elif 8 <= h < 16:
                    return "European (08-16 UTC)"
                else:
                    return "US (16-00 UTC)"

            df["session"] = df["hour"].apply(_session)
            session_order = ["Asian (00-08 UTC)", "European (08-16 UTC)", "US (16-00 UTC)"]

            grouped = df.groupby("session").agg(
                avg_volume=("volume", "mean"),
                total_volume=("volume", "sum"),
                avg_range=("range_pct", "mean"),
                avg_return=("return_pct", "mean"),
                candle_count=("close", "count"),
            ).reindex(session_order)

            # Session summary cards
            session_colors = [WARNING, POSITIVE, NEGATIVE]
            cols = st.columns(3)
            for idx, (sess, row) in enumerate(grouped.iterrows()):
                with cols[idx]:
                    st.markdown(
                        f"<div class='metric-card'>"
                        f"<div class='metric-label'>{sess}</div>"
                        f"<div class='metric-value' style='font-size:1.2rem;'>"
                        f"Vol: {readable_market_cap(int(row['total_volume']))}</div>"
                        f"<div style='color:{TEXT_MUTED};font-size:0.85rem;'>"
                        f"Avg Range: {row['avg_range']:.3f}% | Avg Return: {row['avg_return']:+.3f}%"
                        f"</div></div>",
                        unsafe_allow_html=True,
                    )

            # Volume by session bar chart
            fig_vol = go.Figure()
            for idx, sess in enumerate(session_order):
                if sess in grouped.index:
                    fig_vol.add_trace(go.Bar(
                        x=[sess], y=[grouped.loc[sess, "avg_volume"]],
                        name=sess, marker_color=session_colors[idx],
                    ))
            fig_vol.update_layout(
                title="Average Hourly Volume by Session",
                height=300, template="plotly_dark",
                margin=dict(l=20, r=20, t=40, b=30),
                showlegend=False,
            )
            st.plotly_chart(fig_vol, width="stretch")

            # Volatility by session bar chart
            fig_range = go.Figure()
            for idx, sess in enumerate(session_order):
                if sess in grouped.index:
                    fig_range.add_trace(go.Bar(
                        x=[sess], y=[grouped.loc[sess, "avg_range"]],
                        name=sess, marker_color=session_colors[idx],
                    ))
            fig_range.update_layout(
                title="Average Price Range (%) by Session",
                height=300, template="plotly_dark",
                margin=dict(l=20, r=20, t=40, b=30),
                showlegend=False,
            )
            st.plotly_chart(fig_range, width="stretch")

            # Best session recommendation
            best_vol = grouped["avg_volume"].idxmax()
            best_range = grouped["avg_range"].idxmax()
            st.info(
                f"**Highest volume:** {best_vol} | "
                f"**Most volatile:** {best_range} | "
                f"Best for scalping: choose sessions with high volume AND moderate volatility."
            )


def render_tools_tab():
    """R:R Calculator and Liquidation Levels."""
    st.markdown(f"<h2 style='color:{ACCENT};'>Trading Tools</h2>", unsafe_allow_html=True)

    # === R:R Calculator ===
    st.markdown(f"<h3 style='color:{ACCENT};'>Risk/Reward Calculator</h3>", unsafe_allow_html=True)
    st.markdown(
        f"<p style='color:{TEXT_MUTED}; font-size:0.9rem;'>"
        "Calculate position size, risk/reward ratio, and PnL for different leverage levels."
        "</p>",
        unsafe_allow_html=True,
    )
    rc1, rc2 = st.columns(2)
    with rc1:
        rr_entry = st.number_input("Entry Price ($)", min_value=0.0, value=100000.0, format="%.2f", key="rr_entry")
        rr_stop = st.number_input("Stop Loss ($)", min_value=0.0, value=98000.0, format="%.2f", key="rr_stop")
        rr_target = st.number_input("Take Profit ($)", min_value=0.0, value=105000.0, format="%.2f", key="rr_target")
    with rc2:
        rr_account = st.number_input("Account Size ($)", min_value=0.0, value=10000.0, format="%.2f", key="rr_account")
        rr_risk_pct = st.slider("Risk per Trade (%)", min_value=0.5, max_value=10.0, value=2.0, step=0.5, key="rr_risk")
        rr_direction = st.selectbox("Direction", ["LONG", "SHORT"], key="rr_dir")

    if st.button("Calculate R:R", type="primary"):
        if rr_entry <= 0:
            st.error("Entry price must be > 0")
        else:
            if rr_direction == "LONG":
                risk_per_unit = abs(rr_entry - rr_stop)
                reward_per_unit = abs(rr_target - rr_entry)
            else:
                risk_per_unit = abs(rr_stop - rr_entry)
                reward_per_unit = abs(rr_entry - rr_target)

            rr_ratio = reward_per_unit / risk_per_unit if risk_per_unit > 0 else 0
            risk_amount = rr_account * (rr_risk_pct / 100)
            position_size = risk_amount / risk_per_unit if risk_per_unit > 0 else 0
            position_value = position_size * rr_entry

            # Summary
            rr_color = POSITIVE if rr_ratio >= 1.5 else (WARNING if rr_ratio >= 1.0 else NEGATIVE)
            c1, c2, c3, c4 = st.columns(4)
            with c1:
                st.markdown(
                    f"<div class='metric-card'><div class='metric-label'>R:R Ratio</div>"
                    f"<div class='metric-value' style='color:{rr_color};'>1:{rr_ratio:.2f}</div></div>",
                    unsafe_allow_html=True,
                )
            with c2:
                st.markdown(
                    f"<div class='metric-card'><div class='metric-label'>Risk Amount</div>"
                    f"<div class='metric-value'>${risk_amount:,.2f}</div></div>",
                    unsafe_allow_html=True,
                )
            with c3:
                st.markdown(
                    f"<div class='metric-card'><div class='metric-label'>Position Size</div>"
                    f"<div class='metric-value'>{position_size:,.6f}</div></div>",
                    unsafe_allow_html=True,
                )
            with c4:
                st.markdown(
                    f"<div class='metric-card'><div class='metric-label'>Position Value</div>"
                    f"<div class='metric-value'>${position_value:,.2f}</div></div>",
                    unsafe_allow_html=True,
                )

            # PnL at various leverage levels
            st.markdown(f"<h4 style='color:{ACCENT};'>PnL at Different Leverage Levels</h4>", unsafe_allow_html=True)
            lev_rows = []
            for lev in [1, 2, 3, 5, 10, 15, 20]:
                pnl_win = reward_per_unit * position_size * lev
                pnl_loss = risk_per_unit * position_size * lev
                pnl_win_pct = (pnl_win / rr_account) * 100 if rr_account > 0 else 0
                pnl_loss_pct = (pnl_loss / rr_account) * 100 if rr_account > 0 else 0
                lev_rows.append({
                    "Leverage": f"x{lev}",
                    "Win ($)": f"+${pnl_win:,.2f}",
                    "Win (%)": f"+{pnl_win_pct:.2f}%",
                    "Loss ($)": f"-${pnl_loss:,.2f}",
                    "Loss (%)": f"-{pnl_loss_pct:.2f}%",
                    "Effective Size ($)": f"${position_value * lev:,.2f}",
                })
            st.dataframe(pd.DataFrame(lev_rows), width="stretch")

            if rr_ratio < 1.0:
                st.error("R:R ratio is below 1.0 — risk exceeds reward. Not recommended.")
            elif rr_ratio < 1.5:
                st.warning("R:R ratio is below 1.5. Consider adjusting targets or stops.")
            else:
                st.success(f"R:R ratio of 1:{rr_ratio:.2f} is healthy.")

    # === Liquidation Levels ===
    st.markdown("---")
    st.markdown(f"<h3 style='color:{ACCENT};'>Liquidation Level Estimator</h3>", unsafe_allow_html=True)
    st.markdown(
        f"<p style='color:{TEXT_MUTED}; font-size:0.9rem;'>"
        "Estimate liquidation prices for various leverage levels. "
        "Assumes isolated margin with 100% maintenance margin rate for simplicity."
        "</p>",
        unsafe_allow_html=True,
    )
    lq1, lq2 = st.columns(2)
    with lq1:
        lq_entry = st.number_input("Entry Price ($)", min_value=0.0, value=100000.0, format="%.2f", key="lq_entry")
    with lq2:
        lq_direction = st.selectbox("Direction", ["LONG", "SHORT"], key="lq_dir")

    if st.button("Show Liquidation Levels", type="primary"):
        if lq_entry <= 0:
            st.error("Entry price must be > 0")
        else:
            liq_rows = []
            leverages = [2, 3, 5, 10, 15, 20, 25, 50, 75, 100]
            for lev in leverages:
                if lq_direction == "LONG":
                    liq_price = lq_entry * (1 - 1 / lev)
                    distance_pct = (lq_entry - liq_price) / lq_entry * 100
                else:
                    liq_price = lq_entry * (1 + 1 / lev)
                    distance_pct = (liq_price - lq_entry) / lq_entry * 100
                liq_rows.append({
                    "Leverage": f"x{lev}",
                    "Liquidation Price": f"${liq_price:,.2f}",
                    "Distance from Entry": f"{distance_pct:.2f}%",
                    "Risk Level": "Low" if distance_pct > 10 else ("Medium" if distance_pct > 3 else "HIGH"),
                })
            df_liq = pd.DataFrame(liq_rows)

            def _style_risk(val: str) -> str:
                v = val.upper()
                if v == "LOW":
                    return f"color: {POSITIVE}; font-weight: 600;"
                elif v == "HIGH":
                    return f"color: {NEGATIVE}; font-weight: 600;"
                return f"color: {WARNING}; font-weight: 600;"

            styled_liq = df_liq.style.map(_style_risk, subset=["Risk Level"])
            st.dataframe(styled_liq, width="stretch")

            # Visual chart
            fig_liq = go.Figure()
            liq_prices = []
            for lev in leverages:
                if lq_direction == "LONG":
                    liq_prices.append(lq_entry * (1 - 1 / lev))
                else:
                    liq_prices.append(lq_entry * (1 + 1 / lev))

            fig_liq.add_trace(go.Scatter(
                x=[f"x{l}" for l in leverages], y=liq_prices,
                mode="lines+markers", name="Liquidation Price",
                line=dict(color=NEGATIVE, width=2),
                marker=dict(size=8),
            ))
            fig_liq.add_hline(
                y=lq_entry,
                line=dict(color=POSITIVE, dash="dash", width=1),
                annotation_text=f"Entry: ${lq_entry:,.2f}",
            )
            fig_liq.update_layout(
                height=350, template="plotly_dark",
                margin=dict(l=20, r=20, t=30, b=30),
                xaxis_title="Leverage", yaxis_title="Liquidation Price ($)",
            )
            st.plotly_chart(fig_liq, width="stretch")


# ╔══════════════════════════════════════════════════════════════╗
# ║                    ADVANCED TABS                              ║
# ╚══════════════════════════════════════════════════════════════╝

def render_heatmap_tab():
    """Market heatmap with treemap visualization."""
    st.markdown(
        f"<h2 style='color:{ACCENT};'>Market Heatmap</h2>",
        unsafe_allow_html=True,
    )
    st.markdown(
        f"<div class='panel-box'>"
        f"<b style='color:{ACCENT}; font-size:1rem;'>What does this tab show?</b>"
        f"<p style='color:{TEXT_MUTED}; font-size:0.9rem; margin-top:6px; line-height:1.6;'>"
        f"Displays the top 100 cryptocurrencies in a single view. "
        f"Each tile's {_tip('size', 'The larger the tile, the higher the coin\\'s market capitalization.')} "
        f"represents market cap, while its {_tip('color', 'Green = price increased in the last 24h. Red = price decreased. Darker color = larger change.')} "
        f"reflects the 24-hour price change. "
        f"See which coins are rising and which are falling at a glance.</p>"
        f"</div>",
        unsafe_allow_html=True,
    )

    with st.spinner("Loading market data for heatmap..."):
        try:
            url = "https://api.coingecko.com/api/v3/coins/markets"
            params = {
                "vs_currency": "usd", "order": "market_cap_desc",
                "per_page": 100, "page": 1, "sparkline": False,
                "price_change_percentage": "24h",
            }
            resp = requests.get(url, params=params, timeout=15)
            if resp.status_code != 200:
                st.error("Could not fetch market data for heatmap.")
                return
            coins = resp.json()
        except Exception as e:
            st.error(f"Heatmap data error: {e}")
            return

    if not coins:
        st.warning("No data available.")
        return

    hm_data = []
    for c in coins:
        mcap = c.get('market_cap', 0) or 0
        change = c.get('price_change_percentage_24h', 0) or 0
        price = c.get('current_price', 0) or 0
        symbol = (c.get('symbol', '') or '').upper()
        name = c.get('name', '')
        if mcap > 0:
            hm_data.append({
                'Symbol': symbol,
                'Name': name,
                'Market Cap': mcap,
                'Change 24h (%)': round(change, 2),
                'Price': price,
                'Sector': 'Crypto',
            })

    if not hm_data:
        st.warning("No valid data for heatmap.")
        return

    df_hm = pd.DataFrame(hm_data)

    fig = px.treemap(
        df_hm, path=['Sector', 'Symbol'], values='Market Cap',
        color='Change 24h (%)',
        color_continuous_scale=['#FF0000', '#CC0000', '#660000', '#333333', '#003300', '#006600', '#00CC00'],
        color_continuous_midpoint=0,
        custom_data=['Name', 'Price', 'Change 24h (%)', 'Market Cap'],
    )
    fig.update_traces(
        hovertemplate=(
            "<b>%{customdata[0]} (%{label})</b><br>"
            "Price: $%{customdata[1]:,.2f}<br>"
            "24h Change: %{customdata[2]:+.2f}%<br>"
            "Market Cap: $%{customdata[3]:,.0f}<extra></extra>"
        ),
        textinfo="label+text+percent root",
        texttemplate="<b>%{label}</b><br>%{customdata[2]:+.2f}%",
    )
    fig.update_layout(
        height=650, template='plotly_dark',
        margin=dict(l=5, r=5, t=30, b=5),
        paper_bgcolor=PRIMARY_BG,
        font=dict(size=12),
    )
    st.plotly_chart(fig, use_container_width=True)

    # Top gainers & losers
    st.markdown(f"<div class='god-header'><b style='color:{NEON_BLUE};'>Top Movers (24h)</b></div>",
                unsafe_allow_html=True)
    col_g, col_l = st.columns(2)
    df_sorted = df_hm.sort_values('Change 24h (%)', ascending=False)
    with col_g:
        st.markdown(f"<b style='color:{POSITIVE};'>Top Gainers</b>", unsafe_allow_html=True)
        for _, row in df_sorted.head(10).iterrows():
            st.markdown(
                f"<div style='display:flex; justify-content:space-between; padding:4px 8px; "
                f"border-left:2px solid {POSITIVE}; margin:2px 0; border-radius:0 4px 4px 0; "
                f"background:rgba(0,255,136,0.05);'>"
                f"<span style='color:{TEXT_LIGHT};'>{row['Symbol']}</span>"
                f"<span style='color:{POSITIVE}; font-weight:600;'>+{row['Change 24h (%)']:.2f}%</span></div>",
                unsafe_allow_html=True,
            )
    with col_l:
        st.markdown(f"<b style='color:{NEGATIVE};'>Top Losers</b>", unsafe_allow_html=True)
        for _, row in df_sorted.tail(10).iloc[::-1].iterrows():
            st.markdown(
                f"<div style='display:flex; justify-content:space-between; padding:4px 8px; "
                f"border-left:2px solid {NEGATIVE}; margin:2px 0; border-radius:0 4px 4px 0; "
                f"background:rgba(255,51,102,0.05);'>"
                f"<span style='color:{TEXT_LIGHT};'>{row['Symbol']}</span>"
                f"<span style='color:{NEGATIVE}; font-weight:600;'>{row['Change 24h (%)']:.2f}%</span></div>",
                unsafe_allow_html=True,
            )


def render_monte_carlo_tab():
    """Monte Carlo simulation for price prediction."""
    st.markdown(
        f"<h2 style='color:{ACCENT};'>Monte Carlo Simulation</h2>",
        unsafe_allow_html=True,
    )
    st.markdown(
        f"<div class='panel-box'>"
        f"<b style='color:{ACCENT}; font-size:1rem;'>What does this tab show?</b>"
        f"<p style='color:{TEXT_MUTED}; font-size:0.9rem; margin-top:6px; line-height:1.6;'>"
        f"{_tip('Monte Carlo Simulation', 'A statistical method that calculates mean and standard deviation from historical price movements, then generates thousands of random price paths based on that distribution.')} "
        f"analyzes past price movements to estimate where the price could go in the future. "
        f"Instead of a single prediction, it produces thousands of probability paths.</p>"
        f"<p style='color:{TEXT_MUTED}; font-size:0.85rem; margin-top:6px; line-height:1.6;'>"
        f"<b>Key metrics:</b> "
        f"{_tip('Profit Probability', 'Percentage of simulations that ended above the current price. If 60%, then 600 out of 1000 simulations resulted in profit.')} — Chance of profit, "
        f"{_tip('Expected Return', 'Average return across all simulations. E.g. +5% means profit is expected on average.')} — Average expected gain, "
        f"{_tip('VaR 95%', 'Value at Risk. The worst-case loss covering 95% of scenarios. E.g. -12% means you won\\'t lose more than this 95% of the time.')} — Risk value, "
        f"{_tip('Median Target', 'The median (middle) final price across all simulations. Unlike the mean, it is not affected by extreme outliers.')} — Median target price.</p>"
        f"</div>",
        unsafe_allow_html=True,
    )

    col1, col2, col3 = st.columns(3)
    with col1:
        mc_coin = _normalize_coin_input(st.text_input("Coin", value="BTC", key="mc_coin"))
    with col2:
        mc_sims = st.slider("Simulations", 100, 2000, 500, step=100, key="mc_sims")
    with col3:
        mc_days = st.slider("Forecast Days", 7, 90, 30, key="mc_days")

    mc_tf = st.selectbox("Base Timeframe", ['1h', '4h', '1d'], index=2, key="mc_tf")

    if st.button("Run Simulation", type="primary", key="mc_run"):
        _val_err = _validate_coin_symbol(mc_coin)
        if _val_err:
            st.error(_val_err)
            return

        with st.spinner(f"Running {mc_sims} simulations for {mc_days} days..."):
            df = fetch_ohlcv(mc_coin, mc_tf, limit=500)
            if df is None or len(df) < 30:
                st.error("Not enough data. Try a different coin or timeframe.")
                return

            result = monte_carlo_simulation(df, num_simulations=mc_sims, num_days=mc_days)
            if not result:
                st.error("Simulation failed.")
                return

        # Display stats
        st.markdown(f"<div class='god-header'><b style='color:{NEON_BLUE};'>Simulation Results</b></div>",
                    unsafe_allow_html=True)

        s1, s2, s3, s4 = st.columns(4)
        with s1:
            prob_color = POSITIVE if result['prob_profit'] >= 0.5 else NEGATIVE
            st.markdown(
                f"<div class='mc-stat'>"
                f"<div style='color:{TEXT_MUTED}; font-size:0.75rem;'>PROFIT PROBABILITY</div>"
                f"<div style='color:{prob_color}; font-size:1.6rem; font-weight:700;'>{result['prob_profit']*100:.1f}%</div>"
                f"</div>", unsafe_allow_html=True)
        with s2:
            ret_color = POSITIVE if result['expected_return'] > 0 else NEGATIVE
            st.markdown(
                f"<div class='mc-stat'>"
                f"<div style='color:{TEXT_MUTED}; font-size:0.75rem;'>EXPECTED RETURN</div>"
                f"<div style='color:{ret_color}; font-size:1.6rem; font-weight:700;'>{result['expected_return']:+.2f}%</div>"
                f"</div>", unsafe_allow_html=True)
        with s3:
            st.markdown(
                f"<div class='mc-stat'>"
                f"<div style='color:{TEXT_MUTED}; font-size:0.75rem;'>VALUE AT RISK (95%)</div>"
                f"<div style='color:{NEGATIVE}; font-size:1.6rem; font-weight:700;'>{result['var_95']:.2f}%</div>"
                f"</div>", unsafe_allow_html=True)
        with s4:
            st.markdown(
                f"<div class='mc-stat'>"
                f"<div style='color:{TEXT_MUTED}; font-size:0.75rem;'>MEDIAN TARGET</div>"
                f"<div style='color:{ACCENT}; font-size:1.6rem; font-weight:700;'>${result['median_price']:,.2f}</div>"
                f"</div>", unsafe_allow_html=True)

        # Price range panel
        st.markdown(
            f"<div class='panel-box'>"
            f"<b style='color:{ACCENT};'>Price Distribution ({mc_days}-day forecast)</b>"
            f"<div style='display:grid; grid-template-columns:repeat(5,1fr); gap:10px; margin-top:12px;'>"
            f"<div style='text-align:center;'><div style='color:{NEGATIVE}; font-size:0.7rem;'>WORST CASE</div>"
            f"<div style='color:{NEGATIVE}; font-weight:600;'>${result['min_price']:,.2f}</div></div>"
            f"<div style='text-align:center;'><div style='color:{WARNING}; font-size:0.7rem;'>5th PCTL</div>"
            f"<div style='color:{WARNING}; font-weight:600;'>${result['p5']:,.2f}</div></div>"
            f"<div style='text-align:center;'><div style='color:{NEON_BLUE}; font-size:0.7rem;'>MEDIAN</div>"
            f"<div style='color:{NEON_BLUE}; font-weight:600;'>${result['median_price']:,.2f}</div></div>"
            f"<div style='text-align:center;'><div style='color:{WARNING}; font-size:0.7rem;'>95th PCTL</div>"
            f"<div style='color:{WARNING}; font-weight:600;'>${result['p95']:,.2f}</div></div>"
            f"<div style='text-align:center;'><div style='color:{POSITIVE}; font-size:0.7rem;'>BEST CASE</div>"
            f"<div style='color:{POSITIVE}; font-weight:600;'>${result['max_price']:,.2f}</div></div>"
            f"</div></div>",
            unsafe_allow_html=True,
        )

        # Plot simulation paths
        fig = go.Figure()
        sims = result['simulations']
        days_range = list(range(1, mc_days + 1))

        sample_count = min(100, mc_sims)
        indices = np.random.choice(mc_sims, sample_count, replace=False)
        for idx in indices:
            fig.add_trace(go.Scatter(
                x=days_range, y=sims[idx], mode='lines',
                line=dict(width=0.5, color='rgba(0, 212, 255, 0.08)'),
                showlegend=False, hoverinfo='skip',
            ))

        p5 = np.percentile(sims, 5, axis=0)
        p25 = np.percentile(sims, 25, axis=0)
        p50 = np.percentile(sims, 50, axis=0)
        p75 = np.percentile(sims, 75, axis=0)
        p95 = np.percentile(sims, 95, axis=0)

        fig.add_trace(go.Scatter(x=days_range, y=p95, mode='lines', line=dict(width=0), showlegend=False))
        fig.add_trace(go.Scatter(x=days_range, y=p5, mode='lines', fill='tonexty',
                                  fillcolor='rgba(0, 212, 255, 0.1)', line=dict(width=0), name='90% CI'))
        fig.add_trace(go.Scatter(x=days_range, y=p75, mode='lines', line=dict(width=0), showlegend=False))
        fig.add_trace(go.Scatter(x=days_range, y=p25, mode='lines', fill='tonexty',
                                  fillcolor='rgba(178, 75, 243, 0.15)', line=dict(width=0), name='50% CI'))
        fig.add_trace(go.Scatter(x=days_range, y=p50, mode='lines',
                                  line=dict(color=NEON_BLUE, width=2), name='Median'))
        fig.add_hline(y=result['last_price'], line=dict(color=WARNING, dash='dash', width=1),
                      annotation_text=f"Current: ${result['last_price']:,.2f}")

        fig.update_layout(
            height=500, template='plotly_dark',
            title=f"Monte Carlo — {mc_coin} ({mc_sims} paths, {mc_days} days)",
            xaxis_title="Days", yaxis_title="Price ($)",
            margin=dict(l=20, r=20, t=50, b=30),
            paper_bgcolor=PRIMARY_BG,
        )
        st.plotly_chart(fig, use_container_width=True)

        # Final price distribution histogram
        final_prices = sims[:, -1]
        fig_hist = go.Figure()
        fig_hist.add_trace(go.Histogram(x=final_prices, nbinsx=50, marker_color=NEON_BLUE, opacity=0.7))
        fig_hist.add_vline(x=result['last_price'], line=dict(color=WARNING, dash='dash', width=2),
                           annotation_text="Current Price")
        fig_hist.update_layout(height=300, template='plotly_dark', title="Final Price Distribution",
                                xaxis_title="Price ($)", yaxis_title="Frequency",
                                margin=dict(l=20, r=20, t=50, b=30), paper_bgcolor=PRIMARY_BG)
        st.plotly_chart(fig_hist, use_container_width=True)


def render_fibonacci_tab():
    """Fibonacci retracement and extension analysis."""
    st.markdown(
        f"<h2 style='color:{ACCENT};'>Fibonacci Analysis</h2>",
        unsafe_allow_html=True,
    )
    st.markdown(
        f"<div class='panel-box'>"
        f"<b style='color:{ACCENT}; font-size:1rem;'>What does this tab show?</b>"
        f"<p style='color:{TEXT_MUTED}; font-size:0.9rem; margin-top:6px; line-height:1.6;'>"
        f"{_tip('Fibonacci levels', 'Fibonacci numbers are a ratio sequence found in nature. In financial markets they are used as support/resistance levels for price retracements and extensions. Key levels: 38.2%, 50%, 61.8%.')} "
        f"show how far the price may retrace or extend. "
        f"Also includes {_tip('divergence', 'When price makes a new low/high but RSI or MACD does not confirm it, a reversal becomes more likely.')} detection "
        f"and {_tip('volume profile', 'Shows how much trading volume occurred at each price level. The highest-volume level is marked as POC (Point of Control) and acts as strong support/resistance.')} analysis.</p>"
        f"<p style='color:{TEXT_MUTED}; font-size:0.85rem; margin-top:6px; line-height:1.6;'>"
        f"<b>Additional info:</b> "
        f"{_tip('Regime', 'Current market state: Trending, Ranging, Compression (breakout expected), or High Volatility.')} — Market regime, "
        f"{_tip('EXT', 'Extension levels. When price moves beyond 100%, these show the next targets: 127.2%, 161.8%, 200%, 261.8%.')} — Extension levels.</p>"
        f"</div>",
        unsafe_allow_html=True,
    )

    col1, col2, col3 = st.columns(3)
    with col1:
        fib_coin = _normalize_coin_input(st.text_input("Coin", value="BTC", key="fib_coin"))
    with col2:
        fib_tf = st.selectbox("Timeframe", ['5m', '15m', '1h', '4h', '1d'], index=2, key="fib_tf")
    with col3:
        fib_lookback = st.slider("Lookback Bars", 30, 500, 120, key="fib_lookback")

    if st.button("Calculate Fibonacci", type="primary", key="fib_run"):
        _val_err = _validate_coin_symbol(fib_coin)
        if _val_err:
            st.error(_val_err)
            return

        with st.spinner("Calculating Fibonacci levels..."):
            df = fetch_ohlcv(fib_coin, fib_tf, limit=fib_lookback)
            if df is None or len(df) < 20:
                st.error("Not enough data.")
                return

            levels = calculate_fibonacci_levels(df, lookback=fib_lookback)
            if not levels:
                st.error("Could not calculate levels.")
                return

            divergences = detect_divergence(df)
            vp = calculate_volume_profile(df)
            regime = detect_market_regime(df)

        current_price = float(df['close'].iloc[-1])
        is_uptrend = levels.get('_is_uptrend', True)

        # Market regime badge
        st.markdown(
            f"<div style='display:flex; gap:16px; align-items:center; margin:10px 0;'>"
            f"<div style='background:rgba(15,22,41,0.7); border:1px solid {regime['color']}; "
            f"border-radius:10px; padding:8px 16px;'>"
            f"<span style='color:{TEXT_MUTED}; font-size:0.7rem;'>REGIME</span> "
            f"<span style='color:{regime['color']}; font-weight:700;'>{regime['regime']}</span></div>"
            f"<div style='color:{TEXT_MUTED}; font-size:0.85rem;'>{regime['description']}</div>"
            f"</div>",
            unsafe_allow_html=True,
        )

        # Divergence alerts
        if divergences:
            for div in divergences:
                st.markdown(
                    f"<div style='background:rgba(15,22,41,0.5); border-left:3px solid {div['color']}; "
                    f"padding:8px 14px; border-radius:0 8px 8px 0; margin:4px 0;'>"
                    f"<b style='color:{div['color']};'>{div['type']} DIVERGENCE</b> "
                    f"<span style='color:{TEXT_MUTED}; font-size:0.85rem;'>({div['strength']}) — "
                    f"{div['description']}</span></div>",
                    unsafe_allow_html=True,
                )

        # Fibonacci levels display
        st.markdown(f"<div class='god-header'><b style='color:{NEON_BLUE};'>"
                    f"Fibonacci Levels ({'Uptrend' if is_uptrend else 'Downtrend'})</b></div>",
                    unsafe_allow_html=True)

        fib_colors = {
            '0%': '#FF3366', '23.6%': '#FF6B6B', '38.2%': '#FFD166',
            '50%': '#00D4FF', '61.8%': '#B24BF3', '78.6%': '#00FF88',
            '100%': '#FFFFFF', '127.2%': '#FFD700', '161.8%': '#FF8C00',
            '200%': '#FF4500', '261.8%': '#DC143C',
        }
        fib_display = ['0%', '23.6%', '38.2%', '50%', '61.8%', '78.6%', '100%',
                        '127.2%', '161.8%', '200%', '261.8%']

        for name in fib_display:
            price = levels.get(name, 0)
            if price <= 0:
                continue
            dist_pct = (current_price - price) / current_price * 100
            if abs(dist_pct) < 1:
                bg, border_c, label_extra = 'rgba(0, 212, 255, 0.15)', NEON_BLUE, " CURRENT ZONE"
            elif price > current_price:
                bg, border_c, label_extra = 'rgba(0, 255, 136, 0.05)', POSITIVE, ""
            else:
                bg, border_c, label_extra = 'rgba(255, 51, 102, 0.05)', NEGATIVE, ""

            is_ext = name in ['127.2%', '161.8%', '200%', '261.8%']
            tag = " EXT" if is_ext else ""
            st.markdown(
                f"<div class='fib-level' style='background:{bg}; border-left:3px solid {border_c};'>"
                f"<span style='color:{fib_colors.get(name, NEON_BLUE)}; font-weight:600;'>"
                f"{name}{tag}{label_extra}</span>"
                f"<span style='color:{ACCENT}; font-weight:600;'>${price:,.2f}</span>"
                f"<span style='color:{TEXT_MUTED}; font-size:0.8rem;'>{dist_pct:+.2f}%</span>"
                f"</div>",
                unsafe_allow_html=True,
            )

        # Candlestick chart with Fibonacci levels
        fig = go.Figure()
        fig.add_trace(go.Candlestick(
            x=df['timestamp'], open=df['open'], high=df['high'],
            low=df['low'], close=df['close'],
            increasing_line_color=POSITIVE, decreasing_line_color=NEGATIVE, name="Price",
        ))
        for name in fib_display:
            price = levels.get(name, 0)
            if price <= 0:
                continue
            fig.add_hline(y=price, line=dict(color=fib_colors.get(name, TEXT_MUTED), dash='dot', width=1),
                          annotation_text=f"Fib {name}: ${price:,.2f}",
                          annotation_font=dict(size=9, color=fib_colors.get(name, TEXT_MUTED)))

        if vp and 'poc_price' in vp:
            fig.add_hline(y=vp['poc_price'], line=dict(color=GOLD, dash='dash', width=1.5),
                          annotation_text=f"POC: ${vp['poc_price']:,.2f}")

        fig.update_layout(height=550, template='plotly_dark',
                          title=f"Fibonacci — {fib_coin} ({fib_tf})",
                          margin=dict(l=20, r=20, t=50, b=30),
                          xaxis_rangeslider_visible=False, paper_bgcolor=PRIMARY_BG)
        st.plotly_chart(fig, use_container_width=True)


def render_risk_analytics_tab():
    """Advanced risk analytics."""
    st.markdown(
        f"<h2 style='color:{ACCENT};'>Risk Analytics</h2>",
        unsafe_allow_html=True,
    )
    st.markdown(
        f"<div class='panel-box'>"
        f"<b style='color:{ACCENT}; font-size:1rem;'>What does this tab show?</b>"
        f"<p style='color:{TEXT_MUTED}; font-size:0.9rem; margin-top:6px; line-height:1.6;'>"
        f"Calculates professional risk metrics for your selected coin. "
        f"Quantifies how risky an asset is with hard numbers.</p>"
        f"<p style='color:{TEXT_MUTED}; font-size:0.85rem; margin-top:6px; line-height:1.6;'>"
        f"<b>Metrics:</b> "
        f"{_tip('Sharpe Ratio', 'Return per unit of risk. Above 1.0 is good, above 2.0 is excellent. Negative means you are losing money relative to the risk taken.')} | "
        f"{_tip('Sortino Ratio', 'Like Sharpe but only penalizes downside volatility. Does not penalize upward price swings. A fairer measure.')} | "
        f"{_tip('Max Drawdown', 'Largest peak-to-trough loss percentage. Below 15% is good, above 30% is dangerous.')} | "
        f"{_tip('VaR 95%', 'Value at Risk. On any given day you won\\'t lose more than this 95% of the time. E.g. -3% means on 95 out of 100 days the loss is under 3%.')} | "
        f"{_tip('Calmar Ratio', 'Annual return divided by max drawdown. Shows return relative to risk. Above 1.0 is good.')} | "
        f"{_tip('Skewness', 'Asymmetry of the return distribution. Negative = large losses more frequent, positive = large gains more frequent.')} | "
        f"{_tip('Kurtosis', 'Peakedness of the return distribution. Higher values = more extreme moves (tail risk) than normal.')} </p>"
        f"</div>",
        unsafe_allow_html=True,
    )

    col1, col2 = st.columns(2)
    with col1:
        risk_coin = _normalize_coin_input(st.text_input("Coin", value="BTC", key="risk_coin"))
    with col2:
        risk_tf = st.selectbox("Timeframe", ['1h', '4h', '1d'], index=2, key="risk_tf")

    if st.button("Analyze Risk", type="primary", key="risk_run"):
        _val_err = _validate_coin_symbol(risk_coin)
        if _val_err:
            st.error(_val_err)
            return

        with st.spinner("Calculating risk metrics..."):
            df = fetch_ohlcv(risk_coin, risk_tf, limit=500)
            if df is None or len(df) < 30:
                st.error("Not enough data.")
                return
            metrics = calculate_risk_metrics(df)
            if not metrics:
                st.error("Could not calculate metrics.")
                return

        # Key metrics grid
        st.markdown(f"<div class='god-header'><b style='color:{NEON_BLUE};'>Key Risk Metrics</b></div>",
                    unsafe_allow_html=True)

        r1, r2, r3, r4 = st.columns(4)
        with r1:
            sr_c = POSITIVE if metrics['sharpe'] > 1 else (WARNING if metrics['sharpe'] > 0 else NEGATIVE)
            st.markdown(f"<div class='mc-stat'>"
                        f"<div style='color:{TEXT_MUTED}; font-size:0.7rem;'>SHARPE RATIO</div>"
                        f"<div style='color:{sr_c}; font-size:1.5rem; font-weight:700;'>{metrics['sharpe']:.2f}</div>"
                        f"</div>", unsafe_allow_html=True)
        with r2:
            so_c = POSITIVE if metrics['sortino'] > 1.5 else (WARNING if metrics['sortino'] > 0 else NEGATIVE)
            st.markdown(f"<div class='mc-stat'>"
                        f"<div style='color:{TEXT_MUTED}; font-size:0.7rem;'>SORTINO RATIO</div>"
                        f"<div style='color:{so_c}; font-size:1.5rem; font-weight:700;'>{metrics['sortino']:.2f}</div>"
                        f"</div>", unsafe_allow_html=True)
        with r3:
            dd_c = POSITIVE if abs(metrics['max_drawdown']) < 15 else (WARNING if abs(metrics['max_drawdown']) < 30 else NEGATIVE)
            st.markdown(f"<div class='mc-stat'>"
                        f"<div style='color:{TEXT_MUTED}; font-size:0.7rem;'>MAX DRAWDOWN</div>"
                        f"<div style='color:{dd_c}; font-size:1.5rem; font-weight:700;'>{metrics['max_drawdown']:.2f}%</div>"
                        f"</div>", unsafe_allow_html=True)
        with r4:
            st.markdown(f"<div class='mc-stat'>"
                        f"<div style='color:{TEXT_MUTED}; font-size:0.7rem;'>VALUE AT RISK (95%)</div>"
                        f"<div style='color:{NEGATIVE}; font-size:1.5rem; font-weight:700;'>{metrics['var_95']:.2f}%</div>"
                        f"</div>", unsafe_allow_html=True)

        r5, r6, r7, r8 = st.columns(4)
        with r5:
            ret_c = POSITIVE if metrics['total_return'] > 0 else NEGATIVE
            st.markdown(f"<div class='mc-stat'>"
                        f"<div style='color:{TEXT_MUTED}; font-size:0.7rem;'>TOTAL RETURN</div>"
                        f"<div style='color:{ret_c}; font-size:1.5rem; font-weight:700;'>{metrics['total_return']:+.2f}%</div>"
                        f"</div>", unsafe_allow_html=True)
        with r6:
            cal_c = POSITIVE if metrics['calmar'] > 1 else WARNING
            st.markdown(f"<div class='mc-stat'>"
                        f"<div style='color:{TEXT_MUTED}; font-size:0.7rem;'>CALMAR RATIO</div>"
                        f"<div style='color:{cal_c}; font-size:1.5rem; font-weight:700;'>{metrics['calmar']:.2f}</div>"
                        f"</div>", unsafe_allow_html=True)
        with r7:
            wr_c = POSITIVE if metrics['win_rate'] > 50 else NEGATIVE
            st.markdown(f"<div class='mc-stat'>"
                        f"<div style='color:{TEXT_MUTED}; font-size:0.7rem;'>WIN RATE</div>"
                        f"<div style='color:{wr_c}; font-size:1.5rem; font-weight:700;'>{metrics['win_rate']:.1f}%</div>"
                        f"</div>", unsafe_allow_html=True)
        with r8:
            st.markdown(f"<div class='mc-stat'>"
                        f"<div style='color:{TEXT_MUTED}; font-size:0.7rem;'>ANN. VOLATILITY</div>"
                        f"<div style='color:{WARNING}; font-size:1.5rem; font-weight:700;'>{metrics['ann_volatility']:.1f}%</div>"
                        f"</div>", unsafe_allow_html=True)

        # Performance details
        st.markdown(
            f"<div class='panel-box'><b style='color:{ACCENT};'>Performance Details</b>"
            f"<div style='display:grid; grid-template-columns:repeat(4,1fr); gap:12px; margin-top:12px;'>"
            f"<div><span style='color:{TEXT_MUTED}; font-size:0.75rem;'>Best Day</span><br>"
            f"<span style='color:{POSITIVE}; font-weight:600;'>{metrics['best_day']:+.2f}%</span></div>"
            f"<div><span style='color:{TEXT_MUTED}; font-size:0.75rem;'>Worst Day</span><br>"
            f"<span style='color:{NEGATIVE}; font-weight:600;'>{metrics['worst_day']:.2f}%</span></div>"
            f"<div><span style='color:{TEXT_MUTED}; font-size:0.75rem;'>Skewness</span><br>"
            f"<span style='color:{ACCENT}; font-weight:600;'>{metrics['skewness']:.3f}</span></div>"
            f"<div><span style='color:{TEXT_MUTED}; font-size:0.75rem;'>Kurtosis</span><br>"
            f"<span style='color:{ACCENT}; font-weight:600;'>{metrics['kurtosis']:.3f}</span></div>"
            f"</div></div>",
            unsafe_allow_html=True,
        )

        # Drawdown chart
        dd_series = metrics['drawdown_series']
        ts_vals = df['timestamp'].iloc[1:len(dd_series)+1] if 'timestamp' in df.columns else list(range(len(dd_series)))
        fig_dd = go.Figure()
        fig_dd.add_trace(go.Scatter(x=ts_vals, y=dd_series.values * 100, fill='tozeroy',
                                     fillcolor='rgba(255, 51, 102, 0.2)',
                                     line=dict(color=NEGATIVE, width=1.5), name='Drawdown %'))
        fig_dd.update_layout(height=250, template='plotly_dark', title="Drawdown Analysis",
                              yaxis_title="Drawdown (%)", margin=dict(l=20, r=20, t=50, b=30),
                              paper_bgcolor=PRIMARY_BG)
        st.plotly_chart(fig_dd, use_container_width=True)

        # Cumulative returns
        cum_ret = metrics['cumulative_returns']
        ts_vals2 = df['timestamp'].iloc[1:len(cum_ret)+1] if 'timestamp' in df.columns else list(range(len(cum_ret)))
        fig_cum = go.Figure()
        fig_cum.add_trace(go.Scatter(x=ts_vals2, y=(cum_ret.values - 1) * 100, fill='tozeroy',
                                      fillcolor='rgba(0, 212, 255, 0.1)',
                                      line=dict(color=NEON_BLUE, width=2), name='Return %'))
        fig_cum.add_hline(y=0, line=dict(color=TEXT_MUTED, dash='dash', width=1))
        fig_cum.update_layout(height=300, template='plotly_dark', title="Cumulative Returns",
                               yaxis_title="Return (%)", margin=dict(l=20, r=20, t=50, b=30),
                               paper_bgcolor=PRIMARY_BG)
        st.plotly_chart(fig_cum, use_container_width=True)

        # Return distribution
        returns = df['close'].pct_change().dropna() * 100
        fig_dist = go.Figure()
        fig_dist.add_trace(go.Histogram(x=returns, nbinsx=50, marker_color=NEON_PURPLE, opacity=0.7))
        fig_dist.add_vline(x=0, line=dict(color=ACCENT, dash='dash', width=1))
        fig_dist.add_vline(x=metrics['var_95'], line=dict(color=NEGATIVE, dash='dash', width=2),
                           annotation_text=f"VaR 95%: {metrics['var_95']:.2f}%")
        fig_dist.update_layout(height=250, template='plotly_dark', title="Return Distribution",
                                xaxis_title="Return (%)", yaxis_title="Frequency",
                                margin=dict(l=20, r=20, t=50, b=30), paper_bgcolor=PRIMARY_BG)
        st.plotly_chart(fig_dist, use_container_width=True)


def render_whale_tab():
    """Whale tracking and momentum signals."""
    st.markdown(
        f"<h2 style='color:{ACCENT};'>Whale Tracker</h2>",
        unsafe_allow_html=True,
    )
    st.markdown(
        f"<div class='panel-box'>"
        f"<b style='color:{ACCENT}; font-size:1rem;'>What does this tab show?</b>"
        f"<p style='color:{TEXT_MUTED}; font-size:0.9rem; margin-top:6px; line-height:1.6;'>"
        f"Tracks major market movements and {_tip('trends', 'CoinGecko Trending list is based on sudden spikes in user searches. If a coin is trending, interest is rapidly growing.')}. "
        f"Consists of three main sections:</p>"
        f"<p style='color:{TEXT_MUTED}; font-size:0.85rem; margin-top:6px; line-height:1.6;'>"
        f"<b>1. Trending Coins</b> — Most searched and popular coins on CoinGecko.<br>"
        f"<b>2. Top Gainers / Losers</b> — Biggest 24-hour price winners and losers.<br>"
        f"<b>3. {_tip('Volume Surge Scanner', 'Compares the latest candle volume to the 20-candle average. Above 1.5x is flagged as a Volume Surge. Can be an early signal of large players accumulating.')}:</b> "
        f"Detects abnormal volume spikes — may indicate that whales (large players) are making moves.</p>"
        f"</div>",
        unsafe_allow_html=True,
    )

    with st.spinner("Fetching whale & momentum data..."):
        trending = fetch_trending_coins()
        gainers, losers = fetch_top_gainers_losers(15)

    # Trending coins
    st.markdown(f"<div class='god-header'><b style='color:{NEON_BLUE};'>Trending Coins</b></div>",
                unsafe_allow_html=True)
    if trending:
        for i, coin in enumerate(trending[:10]):
            rank_color = GOLD if i < 3 else NEON_BLUE if i < 6 else TEXT_MUTED
            st.markdown(
                f"<div class='whale-entry'>"
                f"<span style='color:{rank_color}; font-weight:700; font-size:1.1rem; min-width:30px;'>#{i+1}</span>"
                f"<span style='color:{ACCENT}; font-weight:600;'>{coin['symbol']}</span>"
                f"<span style='color:{TEXT_MUTED}; font-size:0.8rem;'>{coin['name']}</span>"
                f"<span style='color:{TEXT_MUTED}; font-size:0.8rem; margin-left:auto;'>"
                f"Rank #{coin['market_cap_rank'] or 'N/A'}</span></div>",
                unsafe_allow_html=True,
            )
    else:
        st.info("Trending data unavailable.")

    # Gainers / losers
    st.markdown(f"<div class='god-header'><b style='color:{NEON_BLUE};'>Market Momentum (24h)</b></div>",
                unsafe_allow_html=True)
    col_g, col_l = st.columns(2)
    with col_g:
        st.markdown(f"<b style='color:{POSITIVE};'>TOP GAINERS</b>", unsafe_allow_html=True)
        for c in (gainers or [])[:12]:
            change = c.get('price_change_percentage_24h', 0)
            symbol = (c.get('symbol', '') or '').upper()
            price = c.get('current_price', 0)
            st.markdown(
                f"<div style='display:flex; justify-content:space-between; padding:4px 8px; "
                f"border-left:2px solid {POSITIVE}; margin:2px 0; background:rgba(0,255,136,0.04); "
                f"border-radius:0 4px 4px 0;'>"
                f"<span style='color:{ACCENT};'>{symbol} <span style='color:{TEXT_MUTED}; font-size:0.75rem;'>"
                f"${price:,.4f}</span></span>"
                f"<span style='color:{POSITIVE}; font-weight:600;'>+{change:.2f}%</span></div>",
                unsafe_allow_html=True,
            )
    with col_l:
        st.markdown(f"<b style='color:{NEGATIVE};'>TOP LOSERS</b>", unsafe_allow_html=True)
        for c in (losers or [])[:12]:
            change = c.get('price_change_percentage_24h', 0)
            symbol = (c.get('symbol', '') or '').upper()
            price = c.get('current_price', 0)
            st.markdown(
                f"<div style='display:flex; justify-content:space-between; padding:4px 8px; "
                f"border-left:2px solid {NEGATIVE}; margin:2px 0; background:rgba(255,51,102,0.04); "
                f"border-radius:0 4px 4px 0;'>"
                f"<span style='color:{ACCENT};'>{symbol} <span style='color:{TEXT_MUTED}; font-size:0.75rem;'>"
                f"${price:,.4f}</span></span>"
                f"<span style='color:{NEGATIVE}; font-weight:600;'>{change:.2f}%</span></div>",
                unsafe_allow_html=True,
            )

    # Volume surge scanner
    st.markdown(f"<div class='god-header'><b style='color:{NEON_BLUE};'>Volume Surge Scanner</b></div>",
                unsafe_allow_html=True)
    scan_tf = st.selectbox("Scan Timeframe", ['5m', '15m', '1h', '4h'], index=2, key="whale_scan_tf")
    if st.button("Scan for Volume Surges", key="whale_scan"):
        with st.spinner("Scanning..."):
            symbols = ["BTC/USDT", "ETH/USDT", "SOL/USDT", "BNB/USDT", "XRP/USDT",
                        "ADA/USDT", "DOGE/USDT", "AVAX/USDT", "DOT/USDT", "LINK/USDT"]
            surges = []
            for sym in symbols:
                try:
                    df_s = fetch_ohlcv(sym, scan_tf, limit=50)
                    if df_s is not None and len(df_s) > 21:
                        avg_vol = df_s['volume'].iloc[-21:-1].mean()
                        last_vol = df_s['volume'].iloc[-1]
                        if avg_vol > 0:
                            ratio = last_vol / avg_vol
                            if ratio > 1.5:
                                surges.append({
                                    'symbol': sym.split('/')[0], 'ratio': ratio,
                                    'change': ((df_s['close'].iloc[-1] / df_s['close'].iloc[-2]) - 1) * 100,
                                })
                except Exception:
                    continue
            if surges:
                surges.sort(key=lambda x: x['ratio'], reverse=True)
                for s in surges:
                    ic = NEGATIVE if s['ratio'] > 3 else (WARNING if s['ratio'] > 2 else NEON_BLUE)
                    pc = POSITIVE if s['change'] > 0 else NEGATIVE
                    st.markdown(
                        f"<div class='whale-entry' style='border-left-color:{ic};'>"
                        f"<span style='color:{ACCENT}; font-weight:700;'>{s['symbol']}</span>"
                        f"<span style='color:{ic}; font-weight:600;'>{s['ratio']:.1f}x Vol</span>"
                        f"<span style='color:{pc};'>{s['change']:+.2f}%</span></div>",
                        unsafe_allow_html=True,
                    )
            else:
                st.info("No significant volume surges detected.")


def render_screener_tab():
    """Advanced multi-condition screener."""
    st.markdown(
        f"<h2 style='color:{ACCENT};'>Advanced Screener</h2>",
        unsafe_allow_html=True,
    )
    st.markdown(
        f"<div class='panel-box'>"
        f"<b style='color:{ACCENT}; font-size:1rem;'>What does this tab show?</b>"
        f"<p style='color:{TEXT_MUTED}; font-size:0.9rem; margin-top:6px; line-height:1.6;'>"
        f"Scans the entire market for coins matching your criteria. You can apply multiple filters simultaneously.</p>"
        f"<p style='color:{TEXT_MUTED}; font-size:0.85rem; margin-top:6px; line-height:1.6;'>"
        f"<b>Filters:</b> "
        f"{_tip('Min Confidence', 'Minimum confidence score. E.g. setting 60% shows only coins with 60%+ confidence.')} | "
        f"{_tip('Signal Filter', 'Choose which signal types to display: STRONG BUY, BUY, SELL, etc.')} | "
        f"{_tip('Min ADX', 'Minimum trend strength. ADX above 20 means a trending market. 25+ is a strong trend.')} | "
        f"{_tip('RSI Range', 'Sets the RSI range. 30-70 is neutral, below 30 is oversold, above 70 is overbought.')} | "
        f"{_tip('Volume Spike Only', 'When checked, only coins showing abnormal volume increases are listed.')} </p>"
        f"</div>",
        unsafe_allow_html=True,
    )

    col1, col2, col3 = st.columns(3)
    with col1:
        scr_tf = st.selectbox("Timeframe", ['5m', '15m', '1h', '4h', '1d'], index=2, key="scr_tf")
        min_confidence = st.slider("Min Confidence %", 0, 100, 60, key="scr_conf")
    with col2:
        signal_filter = st.multiselect("Signal Filter", ['STRONG BUY', 'BUY', 'WAIT', 'SELL', 'STRONG SELL'],
                                        default=['STRONG BUY', 'BUY'], key="scr_signal")
        min_adx = st.slider("Min ADX", 0, 80, 20, key="scr_adx")
    with col3:
        rsi_range = st.slider("RSI Range", 0, 100, (20, 80), key="scr_rsi")
        volume_spike_only = st.checkbox("Volume Spike Only", value=False, key="scr_volspike")

    if st.button("Run Screener", type="primary", key="scr_run"):
        with st.spinner("Scanning markets..."):
            symbols_to_scan = ["BTC/USDT", "ETH/USDT", "SOL/USDT", "BNB/USDT",
                                "XRP/USDT", "ADA/USDT", "DOGE/USDT", "AVAX/USDT",
                                "DOT/USDT", "LINK/USDT", "MATIC/USDT", "UNI/USDT",
                                "ATOM/USDT", "NEAR/USDT", "LTC/USDT", "BCH/USDT"]
            results = []
            progress = st.progress(0)
            for i, sym in enumerate(symbols_to_scan):
                progress.progress((i + 1) / len(symbols_to_scan))
                try:
                    df = fetch_ohlcv(sym, scr_tf, limit=120)
                    if df is None or len(df) < 55:
                        continue
                    a = analyse(df)
                    if a.confidence < min_confidence:
                        continue
                    if signal_filter and a.signal not in signal_filter:
                        continue
                    if not np.isnan(a.adx) and a.adx < min_adx:
                        continue
                    rsi_val = ta.momentum.rsi(df['close'], window=14).iloc[-1]
                    if rsi_val < rsi_range[0] or rsi_val > rsi_range[1]:
                        continue
                    if volume_spike_only and not a.volume_spike:
                        continue
                    try:
                        ai_prob, ai_dir = ml_predict_direction(df)
                    except Exception:
                        ai_prob, ai_dir = 0.5, "NEUTRAL"
                    results.append({
                        'Symbol': sym.split('/')[0],
                        'Price': df['close'].iloc[-1],
                        'Signal': a.signal,
                        'Confidence': a.confidence,
                        'AI': ai_dir,
                        'RSI': round(rsi_val, 1),
                        'ADX': round(a.adx, 1) if not np.isnan(a.adx) else 0,
                        'Leverage': a.leverage,
                    })
                except Exception:
                    continue
            progress.empty()

        if results:
            st.markdown(
                f"<div style='background:rgba(0,255,136,0.08); border:1px solid rgba(0,255,136,0.3); "
                f"border-radius:10px; padding:12px; margin:10px 0; text-align:center;'>"
                f"<span style='color:{POSITIVE}; font-weight:700; font-size:1.2rem;'>"
                f"{len(results)} coins match</span></div>",
                unsafe_allow_html=True,
            )
            results.sort(key=lambda x: x['Confidence'], reverse=True)
            for r in results:
                sig_c = POSITIVE if r['Signal'] in ['STRONG BUY', 'BUY'] else (NEGATIVE if r['Signal'] in ['STRONG SELL', 'SELL'] else WARNING)
                ai_c = POSITIVE if r['AI'] == 'LONG' else (NEGATIVE if r['AI'] == 'SHORT' else WARNING)
                conf_c = POSITIVE if r['Confidence'] >= 70 else (WARNING if r['Confidence'] >= 50 else NEGATIVE)
                st.markdown(
                    f"<div style='display:grid; grid-template-columns:70px 100px 100px 80px 70px 60px 60px; "
                    f"gap:8px; align-items:center; padding:10px 14px; "
                    f"background:rgba(15,22,41,0.5); border-radius:10px; margin:4px 0; "
                    f"border:1px solid rgba(0,212,255,0.08);'>"
                    f"<span style='color:{ACCENT}; font-weight:700;'>{r['Symbol']}</span>"
                    f"<span style='color:{ACCENT}; font-size:0.85rem;'>${r['Price']:,.4f}</span>"
                    f"<span style='color:{sig_c}; font-weight:700;'>{r['Signal']}</span>"
                    f"<span style='color:{conf_c}; font-weight:600;'>{r['Confidence']:.0f}%</span>"
                    f"<span style='color:{ai_c}; font-weight:600;'>{r['AI']}</span>"
                    f"<span style='color:{TEXT_MUTED};'>RSI {r['RSI']}</span>"
                    f"<span style='color:{WARNING};'>x{r['Leverage']}</span></div>",
                    unsafe_allow_html=True,
                )
        else:
            st.warning("No coins matched. Try relaxing filters.")


def render_ensemble_ml_tab():
    """Enhanced ensemble ML prediction."""
    st.markdown(
        f"<h2 style='color:{ACCENT};'>Ensemble AI</h2>",
        unsafe_allow_html=True,
    )
    st.markdown(
        f"<div class='panel-box'>"
        f"<b style='color:{ACCENT}; font-size:1rem;'>What does this tab show?</b>"
        f"<p style='color:{TEXT_MUTED}; font-size:0.9rem; margin-top:6px; line-height:1.6;'>"
        f"Instead of a single model, trains three different {_tip('machine learning models', 'Methods where the computer learns patterns from historical data to make predictions about the future.')} "
        f"and produces a combined prediction via weighted voting. "
        f"More reliable than a single model because the models compensate for each other's errors.</p>"
        f"<p style='color:{TEXT_MUTED}; font-size:0.85rem; margin-top:6px; line-height:1.6;'>"
        f"<b>Models:</b> "
        f"{_tip('Gradient Boosting', 'A tree-based model that iteratively corrects errors. Highest weight (45%). Captures fine details well.')} (45% weight) | "
        f"{_tip('Random Forest', 'Averages the output of hundreds of decision trees. Resistant to overfitting. 35% weight.')} (35% weight) | "
        f"{_tip('Logistic Regression', 'The simplest model. Draws a linear boundary. Acts as a stabilizer. 20% weight.')} (20% weight)</p>"
        f"<p style='color:{TEXT_MUTED}; font-size:0.85rem; margin-top:6px;'>"
        f"{_tip('Model Agreement', 'How many of the three models predict the same direction (LONG or SHORT). 100% = all three agree. 33% = only one differs.')} "
        f"shows how much the models agree with each other.</p>"
        f"</div>",
        unsafe_allow_html=True,
    )

    col1, col2 = st.columns(2)
    with col1:
        ens_coin = _normalize_coin_input(st.text_input("Coin", value="BTC", key="ens_coin"))
    with col2:
        ens_tf = st.selectbox("Timeframe", ['5m', '15m', '1h', '4h', '1d'], index=2, key="ens_tf")

    if st.button("Run Ensemble Prediction", type="primary", key="ens_run"):
        _val_err = _validate_coin_symbol(ens_coin)
        if _val_err:
            st.error(_val_err)
            return

        with st.spinner("Training 3 ML models..."):
            df = fetch_ohlcv(ens_coin, ens_tf, limit=500)
            if df is None or len(df) < 60:
                st.error("Not enough data.")
                return
            prob, direction, details = ml_ensemble_predict(df)
            if not details:
                st.error("Ensemble prediction failed.")
                return

        dir_color = POSITIVE if direction == "LONG" else (NEGATIVE if direction == "SHORT" else WARNING)
        agreement_pct = details.get('agreement', 0) * 100
        agreement_color = POSITIVE if agreement_pct >= 66 else (WARNING if agreement_pct >= 33 else NEGATIVE)

        # Big direction display
        st.markdown(
            f"<div style='text-align:center; padding:24px; background:rgba(15,22,41,0.7); "
            f"border:2px solid {dir_color}; border-radius:16px; margin:16px 0;'>"
            f"<div style='color:{TEXT_MUTED}; font-size:0.8rem; letter-spacing:2px;'>ENSEMBLE PREDICTION</div>"
            f"<div style='color:{dir_color}; font-size:3rem; font-weight:800; margin:8px 0; "
            f"text-shadow:0 0 20px {dir_color};'>{direction}</div>"
            f"<div style='color:{ACCENT}; font-size:1.3rem;'>Probability: {prob*100:.1f}%</div>"
            f"<div style='color:{agreement_color}; font-size:0.9rem; margin-top:6px;'>"
            f"Model Agreement: {agreement_pct:.0f}%</div></div>",
            unsafe_allow_html=True,
        )

        # Individual models
        st.markdown(f"<div class='god-header'><b style='color:{NEON_BLUE};'>Individual Models</b></div>",
                    unsafe_allow_html=True)
        m1, m2, m3 = st.columns(3)
        models = [
            ("Gradient Boosting", details.get('gradient_boosting', 0.5), "45%", NEON_BLUE),
            ("Random Forest", details.get('random_forest', 0.5), "35%", NEON_PURPLE),
            ("Logistic Regression", details.get('logistic_regression', 0.5), "20%", WARNING),
        ]
        for col, (name, pv, weight, color) in zip([m1, m2, m3], models):
            with col:
                md = "LONG" if pv >= 0.55 else ("SHORT" if pv <= 0.45 else "NEUTRAL")
                mc = POSITIVE if md == "LONG" else (NEGATIVE if md == "SHORT" else WARNING)
                st.markdown(
                    f"<div style='background:rgba(15,22,41,0.7); border:1px solid {color}; "
                    f"border-radius:12px; padding:16px; text-align:center;'>"
                    f"<div style='color:{color}; font-weight:700;'>{name}</div>"
                    f"<div style='color:{TEXT_MUTED}; font-size:0.7rem;'>Weight: {weight}</div>"
                    f"<div style='color:{mc}; font-size:1.8rem; font-weight:700; margin:8px 0;'>{md}</div>"
                    f"<div style='color:{ACCENT};'>{pv*100:.1f}%</div></div>",
                    unsafe_allow_html=True,
                )

        # Gauge chart
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=prob * 100,
            gauge={
                'axis': {'range': [0, 100]},
                'bar': {'color': dir_color},
                'bgcolor': PRIMARY_BG,
                'steps': [
                    {'range': [0, 20], 'color': 'rgba(255, 51, 102, 0.3)'},
                    {'range': [20, 40], 'color': 'rgba(255, 51, 102, 0.15)'},
                    {'range': [40, 60], 'color': 'rgba(255, 209, 102, 0.15)'},
                    {'range': [60, 80], 'color': 'rgba(0, 255, 136, 0.15)'},
                    {'range': [80, 100], 'color': 'rgba(0, 255, 136, 0.3)'},
                ],
            },
            title={'text': "Ensemble Bullish Probability", 'font': {'size': 16, 'color': ACCENT}},
            number={'font': {'color': TEXT_LIGHT, 'size': 40}, 'suffix': '%'},
        ))
        fig_gauge.update_layout(height=280, margin=dict(l=30, r=30, t=60, b=20),
                                 template='plotly_dark', paper_bgcolor=PRIMARY_BG)
        st.plotly_chart(fig_gauge, use_container_width=True)


def main():
    """Entry point for the Streamlit app."""

    with st.sidebar:
        st.markdown(
            f"<div style='text-align:center; margin:8px 0;'>"
            f"<span style='color:{ACCENT}; font-size:1.1rem; font-weight:700;'>"
            f"Crypto Command Center</span></div>",
            unsafe_allow_html=True,
        )

        # Auto-refresh control
        auto_refresh = st.checkbox("Auto-Refresh", value=False, key="auto_refresh")
        if auto_refresh:
            refresh_interval = st.slider("Refresh Interval (sec)", 30, 300, 60, key="refresh_interval")
            st.markdown(
                f"<div class='pulse' style='text-align:center; color:{POSITIVE}; font-size:0.8rem;'>"
                f"LIVE — Refreshing every {refresh_interval}s</div>",
                unsafe_allow_html=True,
            )
            time.sleep(refresh_interval)
            st.rerun()

    # All tabs
    tabs = st.tabs([
        "Market", "Spot", "Position", "AI Prediction",
        "Ensemble AI", "Heatmap", "Monte Carlo",
        "Fibonacci", "Risk Analytics", "Whale Tracker",
        "Screener", "Multi-TF", "Correlation",
        "Sessions", "Tools", "Backtest", "Analysis Guide",
    ])

    with tabs[0]:
        render_market_tab()
    with tabs[1]:
        render_spot_tab()
    with tabs[2]:
        render_position_tab()
    with tabs[3]:
        render_ml_tab()
    with tabs[4]:
        render_ensemble_ml_tab()
    with tabs[5]:
        render_heatmap_tab()
    with tabs[6]:
        render_monte_carlo_tab()
    with tabs[7]:
        render_fibonacci_tab()
    with tabs[8]:
        render_risk_analytics_tab()
    with tabs[9]:
        render_whale_tab()
    with tabs[10]:
        render_screener_tab()
    with tabs[11]:
        render_multitf_tab()
    with tabs[12]:
        render_correlation_tab()
    with tabs[13]:
        render_sessions_tab()
    with tabs[14]:
        render_tools_tab()
    with tabs[15]:
        render_backtest_tab()
    with tabs[16]:
        render_guide_tab()

main()
