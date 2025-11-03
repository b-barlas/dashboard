import datetime
import streamlit as st
import pandas as pd
import numpy as np
import requests
import plotly.graph_objs as go
import ccxt
import ta
from streamlit_autorefresh import st_autorefresh
from typing import Tuple
import pandas_ta as pta
from sklearn.linear_model import LogisticRegression


# Set up page title, icon and wide layout
st.set_page_config(
    page_title="Crypto Market Dashboard",
    page_icon="📊",
    layout="wide",
)


PRIMARY_BG = "#0D1117"        # overall app background – near‑black
CARD_BG    = "#16213E"        # cards and panel backgrounds – dark blue
ACCENT     = "#FFFFFF"        # white color
POSITIVE   = "#06D6A0"        # green for positive change
NEGATIVE   = "#EF476F"        # red for negative change
WARNING    = "#FFD166"        # yellow for neutral or caution
TEXT_LIGHT = "#E5E7EB"        # light text color
TEXT_MUTED = "#8CA1B6"        # muted grey for secondary text


st.markdown(f"""
    <style>
    .metric-delta-positive {{
        color: {POSITIVE};
        font-weight: 600;
        font-size: 0.85rem;
    }}
    .metric-delta-negative {{
        color: {NEGATIVE};
        font-weight: 600;
        font-size: 0.85rem;
    }}
    </style>
""", unsafe_allow_html=True)



st.markdown(
    f"""
    <style>
    /* Global styles */
    .stApp {{
        background-color: {PRIMARY_BG};
        color: {TEXT_LIGHT};
        font-family: 'Segoe UI', sans-serif;
    }}

    /* Titles and subtitles */
    h1.title {{
        font-size: 2.4rem;
        font-weight: 700;
        color: {ACCENT};
        margin-bottom: 0.4rem;
    }}
    p.subtitle {{
        font-size: 1.05rem;
        color: {TEXT_MUTED};
        margin-top: 0;
        margin-bottom: 2rem;
    }}

    /* Card styling */
    .metric-card {{
        background-color: {PRIMARY_BG};
        border: 1px solid rgba(255, 255, 255, 0.06);
        border-radius: 14px;
        padding: 24px 20px;
        text-align: center;
        box-shadow: 0 8px 16px rgba(0, 0, 0, 0.5);
        margin-bottom: 20px;
    }}
    .metric-label {{
        font-size: 0.9rem;
        color: {TEXT_MUTED};
        margin-bottom: 8px;
        letter-spacing: 0.5px;
    }}
    .metric-value {{
        font-size: 1.8rem;
        font-weight: 600;
        color: {ACCENT};
    }}
    .metric-delta-positive, .metric-delta-negative {{
        font-size: 0.9rem;
        font-weight: 500;
    }}
    .metric-delta-positive {{ color: {POSITIVE}; }}
    .metric-delta-negative {{ color: {NEGATIVE}; }}

    /* Panel boxes for larger sections */
    .panel-box {{
        background-color: {PRIMARY_BG};
        border-radius: 16px;
        padding: 28px;
        margin-bottom: 32px;
        border: 1px solid rgba(255, 255, 255, 0.06);
        box-shadow: 0 8px 16px rgba(0, 0, 0, 0.5);
    }}

    /* Table styling */
    .table-container {{ overflow-x: auto; }}
    table.dataframe {{
        width: 100% !important;
        border-collapse: collapse;
        background-color: {PRIMARY_BG};
    }}
    table.dataframe thead tr {{
        background-color: {PRIMARY_BG};
    }}
    table.dataframe th {{
        color: {ACCENT};
        padding: 10px;
        text-align: left;
        font-size: 0.9rem;
        border-bottom: 1px solid rgba(255, 255, 255, 0.08);
    }}
    table.dataframe td {{
        padding: 10px;
        font-size: 0.9rem;
        border-bottom: 1px solid rgba(255, 255, 255, 0.04);
        color: {TEXT_LIGHT};
    }}

    /* Remove row hover highlight from Streamlit table */
    table.dataframe tbody tr:hover {{
        background-color: rgba(255, 255, 255, 0.03);
    }}

    /* Force dark theme on Streamlit's interactive dataframes (st.dataframe) */
    [data-testid="stDataFrame"] div[role="grid"] {{
        background-color: {PRIMARY_BG} !important;
        color: {TEXT_LIGHT} !important;
    }}
    [data-testid="stDataFrame"] .row-header,
    [data-testid="stDataFrame"] .cell,
    [data-testid="stDataFrame"] th,
    [data-testid="stDataFrame"] td {{
        background-color: {PRIMARY_BG} !important;
        color: {TEXT_LIGHT} !important;
        border-color: rgba(255, 255, 255, 0.08) !important;
    }}
    /* Override default table (st.table) background */
    [data-testid="stTable"] table {{
        background-color: {PRIMARY_BG} !important;
        color: {TEXT_LIGHT} !important;
        border-color: rgba(255, 255, 255, 0.08) !important;
    }}
    </style>
    """,
    unsafe_allow_html=True
)



# Exchange set up with caching
#
# In the UK, direct access to Binance can be restricted.  To ensure the
# dashboard operates smoothly on Streamlit Cloud or in regions where
# Binance isn’t available, the default exchange provider is set to
# Bybit.  The function can be extended to support additional
# providers (e.g. 'okx', 'kucoin', 'bitget', etc.) if necessary.  A
# fallback mechanism tries each provider in sequence until one
# successfully loads markets.  This avoids hard dependency on any
# single exchange and keeps the app functional when a particular
# provider is blocked.
@st.cache_resource(show_spinner=False)
def get_exchange(provider: str = "bybit"):
    """Return a ccxt exchange instance using the specified provider.

    If the requested provider fails to initialise (e.g. not
    available), a list of fallback providers is tried in order.  The
    returned exchange uses rate limiting and time adjustment options
    similar to the previous Binance setup.  If no providers can be
    initialised, an exception is raised.

    Parameters
    ----------
    provider: str
        The primary exchange identifier (e.g. 'bybit', 'okx').  This can
        be customised via an environment variable or secrets if
        desired.  Defaults to "bybit" for UK compliance.

    Returns
    -------
    ccxt.Exchange
        A configured exchange instance ready to fetch tickers and OHLCV.
    """
    # Determine the list of providers to try.  Always include the
    # requested provider first, followed by sensible defaults.  Order
    # matters: providers at the front are tried before those at the end.
    fallback_providers = [provider.lower()] + [
        p for p in ["bybit", "okx", "kucoin", "bitget", "kraken", "coinbasepro"]
        if p.lower() != provider.lower()
    ]
    last_error = None
    for exch_id in fallback_providers:
        try:
            # Dynamically get the exchange class from ccxt
            exchange_class = getattr(ccxt, exch_id)
            # Instantiate with rate limit and time adjustment options
            exchange = exchange_class({
                "enableRateLimit": True,
                "options": {
                    "adjustForTimeDifference": True
                }
            })
            # Try loading markets to verify connectivity
            exchange.load_markets()
            return exchange
        except Exception as e:
            last_error = e
            continue
    # If all providers failed, raise the last error for debugging
    raise RuntimeError(f"No available exchanges could be initialised. Last error: {last_error}")

# Initialise the global exchange using the default provider.  If you wish to
# change the provider without modifying code, set an environment
# variable STREAMLIT_EXCHANGE_PROVIDER to one of the supported
# identifiers (e.g. 'bybit', 'okx').  The call below reads that
# variable; if unset, it defaults to "bybit".  This makes deployment to
# Streamlit Cloud in regions with Binance restrictions straightforward.
import os
_provider_env = os.environ.get("STREAMLIT_EXCHANGE_PROVIDER", "bybit")
try:
    EXCHANGE = get_exchange(_provider_env)
except Exception as e:
    # Log and fall back to a safe value of None so that downstream
    # functions handle missing exchange gracefully.  In practice this
    # should seldom occur because the fallback list is robust.
    st.warning(f"Exchange provider initialisation failed: {e}")
    EXCHANGE = None

# Fetch BTC and ETH prices in USD from CoinGecko
@st.cache_data(ttl=120, show_spinner=False)
def get_btc_eth_prices():
    try:
        url = "https://api.coingecko.com/api/v3/simple/price"
        params = {"ids": "bitcoin,ethereum", "vs_currencies": "usd"}
        response = requests.get(url, params=params, timeout=10).json()
        return response.get("bitcoin", {}).get("usd", 0), response.get("ethereum", {}).get("usd", 0)
    except Exception as e:
        print(f"get_btc_eth_prices error: {e}")
        return 0, 0

# Fetch market dominance and total/alt market cap from CoinGecko
@st.cache_data(ttl=1800, show_spinner=False)
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
@st.cache_data(ttl=300, show_spinner=False)
def get_fear_greed():
    try:
        data = requests.get("https://api.alternative.me/fng/?limit=1", timeout=10).json()
        value = int(data.get("data", [{}])[0].get("value", 0))
        label = data.get("data", [{}])[0].get("value_classification", "Unknown")
        return value, label
    except Exception as e:
        print(f"get_fear_greed error: {e}")
        return 0, "Unknown"

@st.cache_data(ttl=300, show_spinner=False)
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
    """
    Load and return the market symbols from the configured exchange.

    Returns an empty dictionary if the exchange is not initialised.
    """
    if EXCHANGE is None:
        return {}
    try:
        return EXCHANGE.load_markets()
    except Exception as e:
        st.warning(f"Markets yüklenemedi: {e}")
        return {}

MARKETS = get_markets()


# Fetch price change percentage for a given symbol via ccxt
@st.cache_data(ttl=60, show_spinner=False)
def get_price_change(symbol: str) -> float | None:
    """
    Return the 24h percentage change for the given trading pair.

    If no exchange is initialised or the ticker cannot be fetched,
    returns None.  The percentage is rounded to two decimal places.
    """
    if EXCHANGE is None:
        return None
    try:
        ticker = EXCHANGE.fetch_ticker(symbol)
        percent = ticker.get("percentage")
        return round(percent, 2) if percent is not None else None
    except Exception:
        return None

# Fetch OHLCV data for a symbol and timeframe
@st.cache_data(ttl=300, show_spinner=False)
def fetch_ohlcv(symbol: str, timeframe: str, limit: int = 120) -> pd.DataFrame | None:
    """
    Fetch OHLCV data for a trading pair and timeframe.

    If the configured exchange cannot retrieve data for the symbol or
    timeframe, None is returned.  Data is returned as a DataFrame with
    timestamp converted to pandas datetime.  The function gracefully
    handles the case where no exchange is configured.
    """
    if EXCHANGE is None:
        return None
    try:
        data = EXCHANGE.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
        df = pd.DataFrame(data, columns=["timestamp", "open", "high", "low", "close", "volume"])
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
        return df
    except Exception:
        return None

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
     triangle = "▲" if delta > 0 else "▼"
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
    if df is None or len(df) < 3:
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

    # Morning Star (prev2 down, prev small body, last up & closes > midpoint of prev2)
    if prev2['close'] < prev2['open'] and \
       body_prev < min(body_prev2, body_last) and \
       last['close'] > last['open'] and last['close'] > ((prev2['open'] + prev2['close']) / 2):
        return "▲ Morning Star (3-bar bullish reversal)"

    # Evening Star (prev2 up, prev small body, last down & closes < midpoint of prev2)
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

# Analyse a dataframe of price data and signal, lev_base, comment, volume_spike, atr_comment, candle_pattern, confidence_score, adx, supertrend, ichimoku trend, stochRSI, bollinger, vwap_label, psar_trend, williams_val, cci_label
def analyse(df: pd.DataFrame) -> tuple[str, int, str, bool, str, str, float, float, str, str, float, str, str, str, str, str]:

    if df is None or len(df) < 30:
        return (
            "NO DATA", 1, "Insufficient data", False, "", "", 
            0.0, 0.0, "", "", 0.0, "", "", "", "", ""
        )


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
    df["vwap"] = (df["typical_price"] * df["volume"]).cumsum() / df["volume"].cumsum()

    # Parabolic SAR
    try:
        psar_df = pta.psar(df['high'], df['low'], df['close'])

        if 'PSARl' in psar_df.columns and 'PSARs' in psar_df.columns:
            df["psar"] = psar_df['PSARl'].fillna(psar_df['PSARs'])
        else:
            df["psar"] = psar_df.iloc[:, 0]
    except Exception as e:
        print("PSAR Error:", e)
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
    
    latest = df.iloc[-1]

    vwap_label = "→ Near VWAP"
    if latest["close"] > latest["vwap"]:
        vwap_label = "🟢 Above"
    elif latest["close"] < latest["vwap"]:
        vwap_label = "🔴 Below"


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

    # === BIAS SCORE ===
    bias_score = 0.0    
    
    # SuperTrend calculation
    try:
        st_data = pta.supertrend(df['high'], df['low'], df['close'], length=10, multiplier=3.0)
        df['supertrend'] = st_data[st_data.columns[0]]
    except Exception as e:
        print("SuperTrend Error:", e)
        df['supertrend'] = np.nan

    # Parabolic SAR bias
    psar_val = latest.get("psar", np.nan)
    if pd.notna(psar_val):
        if latest["close"] > psar_val:
            bias_score += 0.5
        elif latest["close"] < psar_val:
            bias_score -= 0.5

    # Williams %R bias
    williams_val = latest["williams_r"]
    if williams_val < -80:
        bias_score += 1.0  # oversold
    elif williams_val > -20:
        bias_score -= 1.0  # overbought

    # CCI bias impact
    if cci_val > 100:
        bias_score -= 1.0
    elif cci_val < -100:
        bias_score += 1.0

    # Stochastic RSI
    stoch_rsi = ta.momentum.StochRSIIndicator(close=df["close"], window=14, smooth1=3, smooth2=3)
    df["stochrsi_k"] = stoch_rsi.stochrsi_k()
    
    # Bollinger Bands
    df['bb_mid'] = df['close'].rolling(window=20).mean()
    df['bb_std'] = df['close'].rolling(window=20).std()
    df['bb_upper'] = df['bb_mid'] + 2 * df['bb_std']
    df['bb_lower'] = df['bb_mid'] - 2 * df['bb_std']
    
    latest = df.iloc[-1]
    stochrsi_k_val = latest.get("stochrsi_k", np.nan)
    

    # SuperTrend direction
    supertrend_trend = "Unavailable"
    st_val = latest.get("supertrend", np.nan)
    
    if pd.notna(st_val):
        if latest["close"] > st_val:
            bias_score += 1.0
            supertrend_trend = "Bullish"
        elif latest["close"] < st_val:
            bias_score -= 1.0
            supertrend_trend = "Bearish"
    
    # EMA trend
    if latest["ema5"] > latest["ema9"] and latest["ema9"] > latest["ema21"] and latest["ema21"] > latest["ema50"]:
        bias_score += 2.0
    elif latest["ema50"] > latest["ema21"] > latest["ema9"] > latest["ema5"]:
        bias_score -= 2.0
    
    # RSI
    try:
        if latest["rsi"] >= 60:
            bias_score += 1.0   # Trend up
        elif latest["rsi"] <= 40:
            bias_score -= 1.0   # Trend down
    except Exception:
        pass

    
    # MACD
    if latest["macd"] > latest["macd_signal"] and latest["macd_diff"] > 0:
        bias_score += 2.0
    elif latest["macd"] < latest["macd_signal"] and latest["macd_diff"] < 0:
        bias_score -= 2.0
    
    # OBV
    if df["obv"].iloc[-1] > df["obv"].iloc[-5]:
        bias_score += 1.0
    elif df["obv"].iloc[-1] < df["obv"].iloc[-5]:
        bias_score -= 1.0

    # VWAP
    if latest["close"] > latest["vwap"]:
        bias_score += 1.0
    elif latest["close"] < latest["vwap"]:
        bias_score -= 1.0

    # Volume Spike
    if volume_spike:

        if latest["close"] > latest["ema21"]:
            bias_score += 0.5
        elif latest["close"] < latest["ema21"]:
            bias_score -= 0.5
    
    # ATR
    atr_ratio = atr_latest / latest["close"]
    
    if atr_ratio > 0.05:
        bias_score -= 1.0 
    elif atr_ratio > 0.03:
        bias_score -= 0.5
    elif atr_ratio >= 0.015:
        bias_score += 0.5 
    else:
        bias_score -= 0.5 

    
    # Candle + EMA
    bullish_patterns = ["Bullish Engulfing", "Hammer"]
    bearish_patterns = ["Bearish Engulfing", "Shooting Star"]
    if candle_label in bullish_patterns and latest["ema5"] > latest["ema21"]:
        bias_score += 2.0
    elif candle_label in bearish_patterns and latest["ema5"] < latest["ema21"]:
        bias_score -= 2.0

    # ADX bias score
    if adx_val >= 40:
        bias_score += 1.5  # very strong trend
    elif adx_val >= 25:
        bias_score += 1.0  # strong trend
    elif adx_val < 15:
        bias_score -= 1.0  # very weak trend
    elif adx_val < 20:
        bias_score -= 0.5  ## weak trend


    # Ichimoku Trend (Cloud position)
    try:
        sa = latest.get("senkou_a", np.nan)
        sb = latest.get("senkou_b", np.nan)
    
        if pd.isna(sa) or pd.isna(sb):
            ichimoku_trend = "Unavailable"
        else:
            if latest["close"] > sa and latest["close"] > sb:
                bias_score += 0.5
                ichimoku_trend = "Bullish"
            elif latest["close"] < sa and latest["close"] < sb:
                bias_score -= 0.5
                ichimoku_trend = "Bearish"
            else:
                ichimoku_trend = "Neutral"
    except Exception:
        ichimoku_trend = "Unavailable"

    
    # Tenkan > Kijun check (momentum direction)
    try:
        tenkan = latest.get("tenkan", np.nan)
        kijun  = latest.get("kijun",  np.nan)
        if pd.notna(tenkan) and pd.notna(kijun):
            if tenkan > kijun:
                bias_score += 0.5
            elif tenkan < kijun:
                bias_score -= 0.5
    except Exception:
        pass

    
    # Senkou A direction check (cloud angle)
    try:
        sa_now  = df["senkou_a"].iloc[-1]
        sa_prev = df["senkou_a"].iloc[-2]
        if pd.notna(sa_now) and pd.notna(sa_prev):
            if sa_now > sa_prev:
                bias_score += 0.5
            elif sa_now < sa_prev:
                bias_score -= 0.5
    except Exception:
        pass


    # Stochastic RSI bias
    stoch_k = latest["stochrsi_k"]
    
    if stoch_k >= 0.9:
        bias_score -= 1.0
    elif stoch_k >= 0.8:
        bias_score -= 0.5
    
    elif stoch_k <= 0.1:
        bias_score += 1.0
    elif stoch_k <= 0.2:
        bias_score += 0.5

    # Bollinger Band bias
    bb_upper = latest['bb_upper']
    bb_lower = latest['bb_lower']
    bb_range = bb_upper - bb_lower
    bb_buffer = bb_range * 0.01  # %1 buffer
    close_price = latest['close']
    
    if close_price > bb_upper + bb_buffer:
        bias_score -= 1.5  
    elif close_price < bb_lower - bb_buffer:
        bias_score += 1.5  
    elif close_price < bb_lower:
        bias_score += 0.5  
    
        
    
        
    confidence_score = round((bias_score + 20.0) / 40 * 100, 1)
    confidence_score = float(np.clip(confidence_score, 0, 100))

    signal, comment = get_signal_from_confidence(confidence_score)

    # Parabolic SAR (for display)
    psar_trend = ""
    if pd.notna(psar_val):
        if latest["close"] > psar_val:
            psar_trend = "▲ Bullish"
        elif latest["close"] < psar_val:
            psar_trend = "▼ Bearish"

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

    
    # === Leverage points ===
    risk_score = 0.0
    bollinger_width = (df['bb_upper'].iloc[-1] - df['bb_lower'].iloc[-1])
    volatility_factor = min(bollinger_width / latest["close"], 0.1)
    rsi_factor = 0.1 if latest["rsi"] > 70 or latest["rsi"] < 30 else 0
    obv_factor = 0.1 if df["obv"].iloc[-1] > df["obv"].iloc[-5] and latest["close"] > latest["ema21"] else 0
    recent = df.tail(20)
    support = recent["low"].min()
    resistance = recent["high"].max()
    current_price = latest["close"]
    sr_factor = 0.1 if abs(current_price - support) / current_price < 0.02 or abs(current_price - resistance) / current_price < 0.02 else 0
    risk_score = volatility_factor + rsi_factor + obv_factor + sr_factor

    if risk_score <= 0.15:
        # 0.00 → 12x, 0.15 → 8x
        lev_base = int(round(np.interp(risk_score, [0.00, 0.15], [12, 8])))
    elif risk_score <= 0.25:
        # 0.15 → 8x, 0.25 → 6x
        lev_base = int(round(np.interp(risk_score, [0.15, 0.25], [8, 6])))
    else:
        # 0.25 → 6x, 0.40+ → 4x (cap at 0.40)
        rs = min(risk_score, 0.40)
        lev_base = int(round(np.interp(rs, [0.25, 0.40], [6, 4])))

    if confidence_score < 40:
        lev_base = min(lev_base, 4)
    elif confidence_score < 70:
        lev_base = min(lev_base, 8)

    return signal, lev_base, comment, volume_spike, atr_comment, candle_pattern, confidence_score, adx_val, supertrend_trend, ichimoku_trend, stochrsi_k_val, bollinger_bias, vwap_label, psar_trend, williams_label, cci_label



def get_scalping_entry_target(
    df: pd.DataFrame,
    confidence_score: float,
    supertrend_trend: str,
    ichimoku_trend: str,
    vwap_label: str,
    volume_spike: bool,
    strict_mode: bool = False
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

    # === Support / Resistance ===
    recent = df.tail(20)
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
    Train a lightweight machine‑learning classifier on recent candles to estimate the
    probability that the next candle's close will be higher (bullish).  A suite of
    technical indicators is computed and shifted to avoid look‑ahead bias.  A
    deterministic LogisticRegression model is then fit.  The function returns
    the probability of an up move and a directional label (LONG/SHORT/NEUTRAL)
    using fixed thresholds of 60%/40% respectively.

    Parameters
    ----------
    df : pd.DataFrame
        A DataFrame containing columns ['open','high','low','close','volume'].

    Returns
    -------
    tuple[float, str]
        (probability_up, direction), where probability_up is in [0, 1].
    """
    # Require a minimum number of candles to build a sensible feature set
    if df is None or len(df) < 60:
        return 0.5, "NEUTRAL"

    # Copy the input to avoid modifying external data and reset index for consistency
    df = df.copy().reset_index(drop=True)

    # Compute a set of technical indicators.  We avoid look‑ahead by shifting
    # features later.  Each indicator is computed on the 'close' series or
    # appropriate OHLC fields.
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

    # Define the prediction target: 1 if the next close is greater than the current close
    df['target'] = (df['close'].shift(-1) > df['close']).astype(int)

    # Specify the feature columns and shift by one row to ensure only historical
    # information is used to predict the next candle.  Concatenate with the target
    # and drop any rows with missing values introduced by shifting/indicators.
    feature_cols = ['ema5', 'ema9', 'ema21', 'rsi', 'macd', 'macd_signal', 'macd_diff', 'obv', 'atr']
    df_features = df[feature_cols].shift(1)
    df_model = pd.concat([df_features, df['target']], axis=1).dropna()

    # If there are not enough rows after dropping NA, return a neutral signal
    if len(df_model) < 50:
        return 0.5, "NEUTRAL"

    X = df_model[feature_cols].astype(float).values
    y = df_model['target'].astype(int).values

    try:
        # Instantiate a deterministic Logistic Regression model.  Setting
        # random_state ensures the coefficients are reproducible across runs and
        # environments (e.g. local vs Streamlit Cloud).
        model = LogisticRegression(max_iter=1000, random_state=42)
        # Fit on all but the last observation, reserving the final row for prediction
        model.fit(X[:-1], y[:-1])
        # Predict probability of class 1 (next candle up) on the most recent row
        prob_up = float(model.predict_proba(X[-1].reshape(1, -1))[0][1])
    except Exception:
        # In case of any unexpected training error, fall back to a neutral probability
        return 0.5, "NEUTRAL"

    # Translate probability into a directional label using fixed thresholds
    if prob_up >= 0.6:
        direction = "LONG"
    elif prob_up <= 0.4:
        direction = "SHORT"
    else:
        direction = "NEUTRAL"
    return prob_up, direction


def render_market_tab():
    """Render the Market Dashboard tab containing top‑level crypto metrics and scanning."""

    # Fetch global market data
    # Unpack market indices.  The function returns BTC/ETH dominance, market caps,
    # 24h change and dominance values for BNB, SOL, ADA and XRP.  We keep the
    # additional dominance values for use in the AI market outlook calculation.
    btc_dom, eth_dom, total_mcap, alt_mcap, mcap_24h_pct, bnb_dom, sol_dom, ada_dom, xrp_dom = get_market_indices()
    fg_value, fg_label = get_fear_greed()
    btc_price, eth_price = get_btc_eth_prices()
    

    # Compute percentage change for market cap
    delta_mcap = mcap_24h_pct
    

    # Compute price change percentages using ccxt
    btc_change = get_price_change("BTC/USDT")
    eth_change = get_price_change("ETH/USDT")

    # Display headline and subtitle
    st.markdown("<h1 class='title'>Crypto Market Dashboard</h1>", unsafe_allow_html=True)
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
    try:
        mkt_df = fetch_ohlcv("BTC/USDT", selected_timeframe, limit=500)
        if mkt_df is not None and not mkt_df.empty:
            mkt_prob, mkt_dir = ml_predict_direction(mkt_df)
        else:
            mkt_prob, mkt_dir = 0.5, "NEUTRAL"
    except Exception:
        mkt_prob, mkt_dir = 0.5, "NEUTRAL"
    # Determine arrow and color for the market prediction card.  The text
    # component will be used below when rendering the metric card, and
    # mkt_color drives the text colour.  Market prediction values are
    # displayed to one decimal place below.
    if mkt_dir == "LONG":
        mkt_arrow = "▲"
        mkt_label = "Up"
        mkt_color = POSITIVE
    elif mkt_dir == "SHORT":
        mkt_arrow = "▼"
        mkt_label = "Down"
        mkt_color = NEGATIVE
    else:
        mkt_arrow = "➖"
        mkt_label = "Neutral"
        mkt_color = WARNING

    # Top row: Price and market cap metrics, including a market prediction card.
    m1, m2, m3, m4, m5 = st.columns(5, gap="medium")
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
            f"  <div class='metric-label'>Fear &amp; Greed</div>"
            f"  <div class='metric-value'>{fg_value}</div>"
            f"  <div style='color:{sentiment_color};font-size:0.9rem;'>{fg_label}</div>"
            f"</div>",
            unsafe_allow_html=True,
        )
    # Market prediction card
    with m5:
        st.markdown(
            f"<div class='metric-card'>"
            f"  <div class='metric-label'>Market Prediction</div>"
            f"  <div class='metric-value'>{mkt_prob*100:.1f}%</div>"
            f"  <div style='color:{mkt_color};font-size:0.9rem;'>{mkt_label}</div>"
            f"</div>",
            unsafe_allow_html=True,
        )

    # Second row: Dominance gauges and AI market outlook
    # Compute an advanced AI market outlook using a longer timeframe (4h) on
    # BTC, ETH and major altcoins (BNB, SOL, ADA, XRP).  We fetch 500 candles
    # for each asset and weight their predicted probabilities by their current
    # dominance values.  This makes the indicator sensitive to shifts in
    # capital between BTC and leading altcoins rather than BTC/ETH alone.
    try:
        btc_df_behav = fetch_ohlcv("BTC/USDT", "4h", limit=500)
        eth_df_behav = fetch_ohlcv("ETH/USDT", "4h", limit=500)
        bnb_df_behav = fetch_ohlcv("BNB/USDT", "4h", limit=500)
        sol_df_behav = fetch_ohlcv("SOL/USDT", "4h", limit=500)
        ada_df_behav = fetch_ohlcv("ADA/USDT", "4h", limit=500)
        xrp_df_behav = fetch_ohlcv("XRP/USDT", "4h", limit=500)
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
        # Compute a weighted probability across all assets.  Each asset's
        # predicted probability is weighted by its current dominance.  Only
        # include assets that have a non‑zero dominance and a defined
        # probability; this avoids skewing the result with neutral priors
        # associated with coins that are absent or extremely small.  If the
        # aggregate weight is zero (very unlikely), fall back to a neutral value.
        probs_dict = {
            'BTC': (btc_prob, btc_dom),
            'ETH': (eth_prob, eth_dom),
            'BNB': (bnb_prob, bnb_dom),
            'SOL': (sol_prob, sol_dom),
            'ADA': (ada_prob, ada_dom),
            'XRP': (xrp_prob, xrp_dom),
        }
        weighted_sum = 0.0
        weight_total = 0.0
        for name, (prob, dom) in probs_dict.items():
            # Skip entries with zero dominance; treat missing probabilities as neutral (0.5)
            if dom and dom > 0:
                p = prob if prob is not None else 0.5
                weighted_sum += p * dom
                weight_total += dom
        behaviour_prob = weighted_sum / weight_total if weight_total > 0 else 0.5
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
        st.plotly_chart(fig_btc, use_container_width=True)

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
        st.plotly_chart(fig_eth, use_container_width=True)

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
        st.plotly_chart(fig_behaviour, use_container_width=True)
        # Show the directional label below the gauge to indicate trend
        st.markdown(
            f"<div style='text-align:center; color:{behaviour_color}; font-size:0.9rem; margin-top:-12px;'>"
            f"{behaviour_label}"
            f"</div>",
            unsafe_allow_html=True
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
    with controls[3]:
        strict_toggle_market = st.toggle(
            "Strict scalp",
            value=True,
            key="strict_scalp_market",
        )



    # Fetch top coins
    with st.spinner(f"Scanning {top_n} coins ({signal_filter}) [{timeframe}] ..."):
        # Obtain top USDT trading pairs from CoinGecko and filter by exchange markets
        def get_top_volume_usdt_symbols(top_n: int = 100, vs_currency: str = "usd"):
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
                    print("Coingecko error:", resp.status_code, resp.text)
                    return [], []
                data = resp.json()
                if not isinstance(data, list):
                    print("Coingecko invalid data type:", type(data), data)
                    return [], []
    
                markets = MARKETS
                valid = []
                seen = set()
    
                for coin in data:
                    symbol = (coin.get("symbol") or "").upper()
                    if not symbol or symbol in seen:
                        continue
                    seen.add(symbol)
    
                    pair = f"{symbol}/USDT"
                    if pair in markets:
                        valid.append(pair)
    
                return valid, data
            except Exception as e:
                print(f"get_top_volume_usdt_symbols error: {e}")
                return [], []
    

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
    
        # Analysis
        results: list[dict] = []
        for sym in working_symbols:
            # Fetch a larger window of historical data for the AI prediction.  A
            # limit of 500 ensures there are enough candles to compute
            # technical features and train the gradient boosting model similar to the AI tab.
            df = fetch_ohlcv(sym, timeframe, limit=500)
            if df is None or len(df) <= 30:
                continue
    
            df['ema9'] = ta.trend.ema_indicator(df['close'], window=9)
            df['ema21'] = ta.trend.ema_indicator(df['close'], window=21)
            df['rsi'] = ta.momentum.rsi(df['close'], window=14)

            # Compute an AI prediction based on recent candles.  The
            # ml_predict_direction function fits an XGBoost classifier to
            # technical features derived from the recent OHLCV data and
            # returns both a probability and a categorical direction
            # (LONG/SHORT/NEUTRAL).  If the data set is too small, it
            # gracefully defaults to a neutral prediction.
            prob_up, ai_direction = ml_predict_direction(df)

            latest = df.iloc[-1]
    
            base = sym.split('/')[0].upper()
            mcap_val = mcap_map.get(base, 0)
    
            price = float(latest['close'])
            price_change = get_price_change(sym)
    
            # main analysis
            signal, lev, comment, volume_spike, atr_comment, candle_pattern, confidence_score, adx_val, \
            supertrend_trend, ichimoku_trend, stochrsi_k_val, bollinger_bias, vwap_label, psar_trend, \
            williams_label, cci_label = analyse(df)
    
            # scalping set-up
            scalp_direction, entry_s, target_s, stop_s, rr_ratio, breakout_note = get_scalping_entry_target(
                df,
                confidence_score,
                supertrend_trend,
                ichimoku_trend,
                vwap_label,
                volume_spike,
                strict_mode=strict_toggle_market
            )
            entry_price = entry_s if scalp_direction else 0.0
            target_price = target_s if scalp_direction else 0.0

            # leverage
            lev_base = lev 
            conservative_lev = max(1, int(round(lev_base * 0.85)))
            medium_risk_lev  = min(max(1, int(round(lev_base * 1.10))), 14)
            high_risk_lev    = min(max(1, int(round(lev_base * 1.35))), 20)
            
                        
                
            # filter (LONG/SHORT/BOTH)
            include = (
                (signal_filter == 'BOTH') or
                (signal_filter == 'LONG' and signal in ['STRONG BUY', 'BUY']) or
                (signal_filter == 'SHORT' and signal in ['STRONG SELL', 'SELL'])
            )
            if not include:
                continue
    
            results.append({
                'Coin': base,
                'Price ($)': f"{price:,.2f}",
                'Signal': signal_plain(signal),
                'Confidence': confidence_score_badge(confidence_score),
                # Include AI prediction (LONG/SHORT/NEUTRAL) based on recent candles.
                'AI Prediction': ai_direction,
                'Market Cap ($)': readable_market_cap(mcap_val),
                'Low Risk (X)': leverage_badge(conservative_lev),
                'Medium Risk (X)': leverage_badge(medium_risk_lev),
                'High Risk (X)': leverage_badge(high_risk_lev),
                'Scalp Opportunity': scalp_direction or "",
                'Entry Price': f"${entry_price:,.2f}" if entry_price else '',
                'Target Price': f"${target_price:,.2f}" if target_price else '',
                'Δ (%)': format_delta(price_change) if price_change is not None else '',
                'Spike Alert': '▲ Spike' if volume_spike else '',
                'ADX': round(adx_val, 1),
                'SuperTrend': supertrend_trend,
                'Volatility': atr_comment,
                'Stochastic RSI': round(stochrsi_k_val, 2),
                'Candle Pattern': candle_pattern,
                'Ichimoku': ichimoku_trend,
                'Bollinger': bollinger_bias,
                'VWAP': vwap_label,
                'PSAR': psar_trend if psar_trend != "Unavailable" else '',
                'Williams %R': williams_label,
                'CCI': cci_label,
                '__confidence_val': confidence_score,
            })
    
        # Sort results by confidence score (descending)
        results = sorted(results, key=lambda x: x['__confidence_val'], reverse=True)
    
        # Limit to top_n results
        results = results[:top_n]
    
        # Prepare DataFrame for display
        if results:
            df_results = pd.DataFrame(results)
    
            # Trend and ADX column visual
            df_results["SuperTrend"] = df_results["SuperTrend"].apply(format_trend)
            df_results["ADX"] = df_results["ADX"].apply(format_adx)
            df_results["Ichimoku"] = df_results["Ichimoku"].apply(format_trend)
            df_results["Stochastic RSI"] = df_results["Stochastic RSI"].apply(format_stochrsi)
    
            # '__confidence_val' hide
            df_display = df_results.drop(columns=['__confidence_val'])
    
            # Style
            # Apply styling to several columns.  Include the AI Prediction column in
            # the signal styling to highlight LONG/SHORT/NEUTRAL outcomes.
            styled = (
                df_display.style
                .map(style_signal, subset=['Signal', 'AI Prediction'])
                .map(style_confidence, subset=['Confidence'])
                .map(style_scalp_opp, subset=['Scalp Opportunity'])
                .map(style_delta, subset=['Δ (%)'])
            )
            st.dataframe(styled, use_container_width=True)

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
    # Assign a unique key to avoid StreamlitDuplicateElementId errors when multiple text_input calls share the same label
    coin = st.text_input(
        "Enter coin symbol (e.g., BTC/USDT)",
        value="BTC/USDT",
        key="spot_coin_input",
    ).upper()
    timeframe = st.selectbox("Timeframe", ['1m', '3m', '5m', '15m', '1h', '4h', '1d'], index=4)
    if st.button("Analyse", type="primary"):
        df = fetch_ohlcv(coin, timeframe)
        if df is None or len(df) < 30:
            st.error("Could not fetch data or not enough candles. Try another symbol/timeframe.")
            return
        signal, lev, comment, volume_spike, atr_comment, candle_pattern, confidence_score, adx_val, supertrend_trend, ichimoku_trend, stochrsi_k_val, bollinger_bias, vwap_label, psar_trend, williams_label, cci_label = analyse(df)

        # --- Leverage tiers (proportional to analyse() -> lev) ---
        lev_base = lev
        conservative_lev = max(1, int(round(lev_base * 0.85)))
        medium_risk_lev  = min(max(1, int(round(lev_base * 1.10))), 14)
        high_risk_lev    = min(max(1, int(round(lev_base * 1.35))), 20)

        current_price = df['close'].iloc[-1]

        # Display summary
        signal_clean = signal_plain(signal)
        
        st.markdown(
            f"**Signal:** {signal_clean} | "
            f"**Leverage:** LOW x{conservative_lev} • MED x{medium_risk_lev} • HIGH x{high_risk_lev} | "
            f"**Confidence:** {confidence_score_badge(confidence_score)}",
            unsafe_allow_html=True
        )

        st.markdown(f"<p style='color:{TEXT_MUTED};'>{comment}</p>", unsafe_allow_html=True)

        # Volume / Volatility / Pattern explanations
        explanations = []

        if volume_spike:
            explanations.append("📈 <b>Volume Spike detected</b> – sudden increase in trading activity.")

        # Clean ATR comment (strip emoji/symbols)
        atr_clean = atr_comment.replace("▲", "").replace("▼", "").replace("–", "").strip()
        if atr_clean == "Moderate":
            explanations.append("🔄 <b>Volatility is moderate</b> – stable price conditions.")
        elif atr_clean == "High":
            explanations.append("⚠️ <b>Volatility is high</b> – expect sharp moves.")
        elif atr_clean == "Low":
            explanations.append("🟢 <b>Volatility is low</b> – steady market behaviour.")

        if candle_pattern:
            explanations.append(f"🕯️ <b>Candle pattern:</b> {candle_pattern}")

        if not np.isnan(adx_val):
            explanations.append(f"📊 <b>ADX:</b> {format_adx(adx_val)}")


        if supertrend_trend:
            explanations.append(f"📈 <b>SuperTrend:</b> {format_trend(supertrend_trend)}")

        if ichimoku_trend == "Bullish":
            explanations.append("☁️ <b>Ichimoku Cloud:</b> Price is above the cloud → <span style='color:limegreen;'>Bullish</span> signal.")
        elif ichimoku_trend == "Bearish":
            explanations.append("☁️ <b>Ichimoku Cloud:</b> Price is below the cloud → <span style='color:red;'>Bearish</span> signal.")
        elif ichimoku_trend == "Neutral":
            explanations.append("☁️ <b>Ichimoku Cloud:</b> Price is inside the cloud → <span style='color:orange;'>Neutral</span> state.")
        
        if not np.isnan(stochrsi_k_val):
            if stochrsi_k_val < 0.2:
                explanations.append("🟢 <b>Stochastic RSI:</b> Oversold (< 0.2) – possible rebound.")
            elif stochrsi_k_val > 0.8:
                explanations.append("🔴 <b>Stochastic RSI:</b> Overbought (> 0.8) – possible pullback.")
            else:
                explanations.append("🟡 <b>Stochastic RSI:</b> Neutral zone.")

        if "Overbought" in bollinger_bias:
            explanations.append("🔴 <b>Bollinger:</b> Price is strongly above upper band — <span style='color:red;'>overbought zone</span>.")
        elif "Oversold" in bollinger_bias:
            explanations.append("🟢 <b>Bollinger:</b> Price is strongly below lower band — <span style='color:limegreen;'>oversold zone</span>.")
        elif "Near" in bollinger_bias:
            explanations.append(f"🟡 <b>Bollinger:</b> {bollinger_bias} — price is near edge of band.")
        else:
            explanations.append("➖ <b>Bollinger:</b> Price is inside bands — <span style='color:gray;'>neutral zone</span>.")


        if vwap_label == "🟢 Above":
            explanations.append("🟢 <b>VWAP:</b> Price is above VWAP — <span style='color:limegreen;'>bullish bias</span>.")
        elif vwap_label == "🔴 Below":
            explanations.append("🔴 <b>VWAP:</b> Price is below VWAP — <span style='color:red;'>bearish bias</span>.")
        elif "Near" in vwap_label:
            explanations.append("🟡 <b>VWAP:</b> Price is near VWAP — <span style='color:gray;'>neutral zone</span>.")

        if "Bullish" in psar_trend:
            explanations.append("🟢 <b>Parabolic SAR:</b> Price is above PSAR — <span style='color:limegreen;'>bullish continuation</span>.")
        elif "Bearish" in psar_trend:
            explanations.append("🔴 <b>Parabolic SAR:</b> Price is below PSAR — <span style='color:red;'>bearish reversal</span>.")
        elif psar_trend == "":
            explanations.append("➖ <b>Parabolic SAR:</b> Not available — insufficient data or calculation error.")

        if williams_label == "🟢 Oversold":
            explanations.append("🟢 <b>Williams %R:</b> Below −80 — <span style='color:limegreen;'>oversold</span> zone.")
        elif williams_label == "🔴 Overbought":
            explanations.append("🔴 <b>Williams %R:</b> Above −20 — <span style='color:red;'>overbought</span> zone.")
        elif williams_label == "🟡 Neutral":
            explanations.append("🟡 <b>Williams %R:</b> Neutral range.")

        if "Oversold" in cci_label:
            explanations.append("🟢 <b>CCI:</b> Oversold (< -100) — potential bullish reversal.")
        elif "Overbought" in cci_label:
            explanations.append("🔴 <b>CCI:</b> Overbought (> 100) — potential bearish reversal.")
        else:
            explanations.append("🟡 <b>CCI:</b> Neutral range.")
        
                
                                
        if explanations:
            st.markdown("<br/>".join(explanations), unsafe_allow_html=True)

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
        st.plotly_chart(gauge_sent, use_container_width=True)
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
            wma20 = pta.wma(df['close'], length=20)
            wma50 = pta.wma(df['close'], length=50)
            fig.add_trace(go.Scatter(x=df['timestamp'], y=wma20, mode='lines',
                                     name="WMA20", line=dict(color='#34D399', width=1, dash='dot')))
            fig.add_trace(go.Scatter(x=df['timestamp'], y=wma50, mode='lines',
                                     name="WMA50", line=dict(color='#10B981', width=1, dash='dash')))
        except Exception:
            pass
        # Place legend at top left for candlestick chart
        fig.update_layout(
            height=380,
            template='plotly_dark',
            margin=dict(l=20, r=20, t=30, b=30),
            xaxis_rangeslider_visible=False,
            showlegend=True,
            legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='left', x=0)
        )
        st.plotly_chart(fig, use_container_width=True)
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
        st.plotly_chart(rsi_fig, use_container_width=True)

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
        st.plotly_chart(macd_fig, use_container_width=True)

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
        st.plotly_chart(volume_fig, use_container_width=True)

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
        obv_change = ((df['obv'].iloc[-1] - df['obv'].iloc[-5]) / abs(df['obv'].iloc[-5]) * 100) if df['obv'].iloc[-5] != 0 else 0
        recent = df.tail(20)
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
    coin = st.text_input(
        "Coin Symbol (e.g. BTC/USDT)",
        value="BTC/USDT",
        key="position_coin_input",
    ).upper()
    selected_timeframes = st.multiselect("Select up to 3 Timeframes", ['1m', '3m', '5m', '15m', '1h', '4h', '1d'], default=['3m'], max_selections=3)

    default_entry_price: float = 0.0
    try:
        df_current = fetch_ohlcv(coin, selected_timeframes[0], limit=1)
        if df_current is not None and len(df_current) > 0:
            default_entry_price = float(df_current['close'].iloc[-1])
    except Exception:
        default_entry_price = 0.0

    entry_price = st.number_input("Entry Price", min_value=0.0, format="%.4f", value=default_entry_price)
    direction = st.selectbox("Position Direction", ["LONG", "SHORT"])

    strict_toggle_position = st.toggle(
        "Strict scalp",
        value=True,
        key="strict_scalp_position",
    )
   
    if st.button("Analyse Position", type="primary"):
        tf_order = {'1m': 1, '3m': 2, '5m': 3, '15m': 4, '1h': 5, '4h': 6, '1d': 7}
        largest_tf = max(selected_timeframes, key=lambda tf: tf_order[tf])

        cols = st.columns(len(selected_timeframes))

        for idx, tf in enumerate(selected_timeframes):
            with cols[idx]:
                df = fetch_ohlcv(coin, tf, limit=100)
                if df is None or len(df) < 30:
                    st.error(f"Not enough data to analyse position for {tf}.")
                    continue

                signal, lev, comment, volume_spike, atr_comment, candle_pattern, confidence_score, adx_val, supertrend_trend, ichimoku_trend, stochrsi_k_val, bollinger_bias, vwap_label, psar_trend, williams_label, cci_label = analyse(df)
                
                # --- Leverage tiers (proportional to analyse() -> lev) ---
                lev_base = lev
                conservative_lev = max(1, int(round(lev_base * 0.85)))
                medium_risk_lev  = min(max(1, int(round(lev_base * 1.10))), 14)
                high_risk_lev    = min(max(1, int(round(lev_base * 1.35))), 20)
                
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

                st.markdown(
                    f"**Signal:** {signal_clean} | "
                    f"**Leverage:** LOW X{conservative_lev} • MED X{medium_risk_lev} • HIGH X{high_risk_lev} | "
                    f"**Confidence:** {confidence_score_badge(confidence_score)}",
                    unsafe_allow_html=True
                )

                st.markdown(f"<p style='color:{TEXT_MUTED};'>{comment}</p>", unsafe_allow_html=True)
                

                explanations = []
                if volume_spike:
                    explanations.append("📈 <b>Volume Spike detected</b> – sudden increase in trading activity.")

                atr_clean = atr_comment.replace("▲", "").replace("▼", "").replace("–", "").strip()
                if atr_clean == "Moderate":
                    explanations.append("🔄 <b>Volatility is moderate</b> – stable price conditions.")
                elif atr_clean == "High":
                    explanations.append("⚠️ <b>Volatility is high</b> – expect sharp moves.")
                elif atr_clean == "Low":
                    explanations.append("🟢 <b>Volatility is low</b> – steady market behaviour.")

                if candle_pattern:
                    explanations.append(f"🕯️ <b>Candle pattern:</b> {candle_pattern}")

                if not np.isnan(adx_val):
                    explanations.append(f"📊 <b>ADX:</b> {format_adx(adx_val)}")
                
                if supertrend_trend in ["Bullish", "Bearish", "Neutral"]:
                    trend_icon = "▲" if supertrend_trend == "Bullish" else "▼" if supertrend_trend == "Bearish" else "–"
                    explanations.append(f"📈 <b>SuperTrend:</b> {trend_icon} {supertrend_trend}")

                if ichimoku_trend == "Bullish":
                    explanations.append("☁️ <b>Ichimoku Cloud:</b> Price is above the cloud → <span style='color:limegreen;'>Bullish</span> signal.")
                elif ichimoku_trend == "Bearish":
                    explanations.append("☁️ <b>Ichimoku Cloud:</b> Price is below the cloud → <span style='color:red;'>Bearish</span> signal.")
                elif ichimoku_trend == "Neutral":
                    explanations.append("☁️ <b>Ichimoku Cloud:</b> Price is inside the cloud → <span style='color:orange;'>Neutral</span> state.")
                
                if not np.isnan(stochrsi_k_val):
                    if stochrsi_k_val < 0.2:
                        explanations.append("🟢 <b>Stochastic RSI:</b> Oversold (< 0.2) – possible rebound.")
                    elif stochrsi_k_val > 0.8:
                        explanations.append("🔴 <b>Stochastic RSI:</b> Overbought (> 0.8) – possible pullback.")
                    else:
                        explanations.append("🟡 <b>Stochastic RSI:</b> Neutral zone.")

                if "Overbought" in bollinger_bias:
                    explanations.append("🔴 <b>Bollinger:</b> Price is strongly above upper band — <span style='color:red;'>overbought zone</span>.")
                elif "Oversold" in bollinger_bias:
                    explanations.append("🟢 <b>Bollinger:</b> Price is strongly below lower band — <span style='color:limegreen;'>oversold zone</span>.")
                elif "Near" in bollinger_bias:
                    explanations.append(f"🟡 <b>Bollinger:</b> {bollinger_bias} — price is near edge of band.")
                else:
                    explanations.append("➖ <b>Bollinger:</b> Price is inside bands — <span style='color:gray;'>neutral zone</span>.")
                               
                if vwap_label == "🟢 Above":
                    explanations.append("🟢 <b>VWAP:</b> Price is above VWAP — <span style='color:limegreen;'>bullish bias</span>.")
                elif vwap_label == "🔴 Below":
                    explanations.append("🔴 <b>VWAP:</b> Price is below VWAP — <span style='color:red;'>bearish bias</span>.")
                elif "Near" in vwap_label:
                    explanations.append("🟡 <b>VWAP:</b> Price is near VWAP — <span style='color:gray;'>neutral zone</span>.")

                if "Bullish" in psar_trend:
                    explanations.append("🟢 <b>Parabolic SAR:</b> Price is above PSAR — <span style='color:limegreen;'>bullish continuation</span>.")
                elif "Bearish" in psar_trend:
                    explanations.append("🔴 <b>Parabolic SAR:</b> Price is below PSAR — <span style='color:red;'>bearish reversal</span>.")
                elif psar_trend == "":
                    explanations.append("➖ <b>Parabolic SAR:</b> Not available — insufficient data or calculation error.")

                if williams_label == "🟢 Oversold":
                    explanations.append("🟢 <b>Williams %R:</b> Below −80 — <span style='color:limegreen;'>oversold</span> zone.")
                elif williams_label == "🔴 Overbought":
                    explanations.append("🔴 <b>Williams %R:</b> Above −20 — <span style='color:red;'>overbought</span> zone.")
                elif williams_label == "🟡 Neutral":
                    explanations.append("🟡 <b>Williams %R:</b> Neutral range.")
                
                if "Oversold" in cci_label:
                    explanations.append("🟢 <b>CCI:</b> Oversold (< -100) — potential bullish reversal.")
                elif "Overbought" in cci_label:
                    explanations.append("🔴 <b>CCI:</b> Overbought (> 100) — potential bearish reversal.")
                else:
                    explanations.append("🟡 <b>CCI:</b> Neutral range.")
                
                                                                
                                                
                if explanations:
                    st.markdown("<br/>".join(explanations), unsafe_allow_html=True)

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

                recent_sr = df.tail(20)
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
                    df_scalp['macd'] = ta.trend.macd(df_scalp['close'])
                    df_scalp['macd_signal'] = ta.trend.macd_signal(df_scalp['close'])
                    df_scalp['macd_diff'] = df_scalp['macd'] - df_scalp['macd_signal']
                    df_scalp['obv'] = ta.volume.on_balance_volume(df_scalp['close'], df_scalp['volume'])
                
                    latest = df_scalp.iloc[-1]
                    close_price = latest['close']
                    ema5_val = latest['ema5']
                    ema13_val = latest['ema13']
                    macd_hist_s = latest['macd_diff']
                    rsi14_val = latest['rsi']
                    obv5 = df_scalp['obv'].iloc[-5]
                    obv_change_s = ((latest['obv'] - obv5) / abs(obv5) * 100) if obv5 != 0 else 0
                    support_s = df_scalp['low'].tail(20).min()
                    resistance_s = df_scalp['high'].tail(20).max()
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
                        strict_mode=strict_toggle_position
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
                        msg = breakout_note or ("No valid scalping setup with current filters." if strict_toggle_position else "No valid scalping setup (soft mode).")
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
                wma20_c = pta.wma(df_candle['close'], length=20)
                wma50_c = pta.wma(df_candle['close'], length=50)
                fig_candle.add_trace(go.Scatter(x=df_candle['timestamp'], y=wma20_c, mode='lines',
                                                name="WMA20", line=dict(color='#34D399', width=1, dash='dot')))
                fig_candle.add_trace(go.Scatter(x=df_candle['timestamp'], y=wma50_c, mode='lines',
                                                name="WMA50", line=dict(color='#10B981', width=1, dash='dash')))
            except Exception:
                pass
            fig_candle.update_layout(
                height=380,
                template='plotly_dark',
                margin=dict(l=20, r=20, t=30, b=30),
                xaxis_rangeslider_visible=False,
                showlegend=True,
                legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='left', x=0)
            )
            st.markdown(f"<h4 style='color:{ACCENT};'>📈 Candlestick Chart – {largest_tf}</h4>", unsafe_allow_html=True)
            st.plotly_chart(fig_candle, use_container_width=True)
        else:
            st.warning(f"Not enough data to display candlestick chart for {largest_tf}.")





def render_guide_tab():
    """Render an Analysis Guide explaining the calculations used in the dashboard."""

    st.markdown(
        f"<h2 style='color:{ACCENT}; font-size:1.6rem; margin-bottom:1rem;'>Analysis Guide</h2>",
        unsafe_allow_html=True,
    )

    signal_html = f"""
    <div class='panel-box'>
      <b style='color:{ACCENT}; font-size:1.2rem;'>Signal &amp; Bias Score</b>
      <ul style='color:{TEXT_MUTED}; font-size:0.92rem; line-height:1.6; margin-top:0.5rem;'>
        <li><span style='color:{ACCENT};'>EMA Trend:</span> Measures trend strength using Exponential Moving Averages. EMA5 &gt; EMA9 &gt; EMA21 &gt; EMA50 indicates strong bullish alignment → <b>+2 points</b>. Reverse order signals bearish → <b>−2 points</b>.</li> 
        <li><span style='color:{ACCENT};'>RSI (Relative Strength Index):</span> Uses a trend-oriented pivot approach. RSI ≥ 60 → <b>+1</b> (upward bias); RSI ≤ 40 → <b>−1</b> (downward bias). Overbought/oversold timing is primarily managed via StochRSI and Bollinger Bands.</li>  
        <li><span style='color:{ACCENT};'>MACD:</span> Tracks trend direction and momentum. Bullish crossover (MACD &gt; Signal and positive histogram) → <b>+2</b>, Bearish crossover → <b>−2</b>.</li>
        <li><span style='color:{ACCENT};'>OBV (On-Balance Volume):</span> Volume-based trend indicator. OBV rising over the last 5 candles → <b>+1</b>, falling → <b>−1</b>.</li>
        <li><span style='color:{ACCENT};'>VWAP:</span> Volume-weighted average price. Price above VWAP → <b>+1</b>, below VWAP → <b>−1</b>.</li>
        <li><span style='color:{ACCENT};'>Volume Spike:</span> Detects sudden volume surges. Volume ≥ ~2× recent average and price above EMA21 → <b>+0.5</b>; if below EMA21 → <b>−0.5</b>.</li>
        <li><span style='color:{ACCENT};'>Volatility (ATR):</span> Measures price volatility (ATR/Close). 1.5–3% → <b>+0.5</b>, 3–5% → <b>−0.5</b>, &gt;5% → <b>−1</b>, &lt;1.5% → <b>−0.5</b>.</li>
        <li><span style='color:{ACCENT};'>Candle Pattern + EMA:</span> Candlestick pattern combined with trend context. Bullish pattern + above EMA21 → <b>+2</b>. Bearish pattern + below EMA21 → <b>−2</b>.</li>
        <li><span style='color:{ACCENT};'>ADX:</span> Measures trend strength (directionless). ADX ≥ 40 → <b>+1.5</b>; ADX ≥ 25 → <b>+1.0</b>; ADX &lt; 15 → <b>−1.0</b>; 15 ≤ ADX &lt; 20 → <b>−0.5</b>.</li>
        <li><span style='color:{ACCENT};'>SuperTrend:</span> Trend direction filter. Price above SuperTrend → <b>+1</b>, below → <b>−1</b>.</li>
        <li><span style='color:{ACCENT};'>Ichimoku Cloud:</span> With safety checks: Price above both Senkou A &amp; B → <b>+0.5</b>, below both → <b>−0.5</b>, inside cloud or missing data → <i>neutral</i>. Tenkan &gt; Kijun → <b>+0.5</b> / Tenkan &lt; Kijun → <b>−0.5</b>. Senkou A sloping upward → <b>+0.5</b> / downward → <b>−0.5</b>.</li>
        <li><span style='color:{ACCENT};'>Stochastic RSI:</span> Advanced momentum/timing. %K ≤ 0.1 → <b>+1</b>, ≤ 0.2 → <b>+0.5</b>, ≥ 0.9 → <b>−1</b>, ≥ 0.8 → <b>−0.5</b>.</li>
        <li><span style='color:{ACCENT};'>Bollinger Bands:</span> Price extremes. Far above upper band → <b>−1.5</b>, slightly above → <b>−0.5</b>; far below lower band → <b>+1.5</b>, slightly below → <b>+0.5</b>.</li>
        <li><span style='color:{ACCENT};'>Parabolic SAR:</span> Trend reversal indicator. Price above PSAR → <b>+0.5</b>, below → <b>−0.5</b>.</li>
        <li><span style='color:{ACCENT};'>Williams %R:</span> Momentum oscillator. Value &lt; −80 → <b>+1</b>, value &gt; −20 → <b>−1</b>.</li>
        <li><span style='color:{ACCENT};'>CCI (Commodity Channel Index):</span> Measures deviation from mean trend. CCI &lt; −100 → <b>+1</b>, CCI &gt; 100 → <b>−1</b>, neutral → <b>0</b>.</li>
      </ul>
      <p style='color:{TEXT_MUTED}; font-size:0.9rem; margin-top:0.5rem;'>
        The total bias score ranges from −20 to +20; the Confidence Score is derived from this value.
      </p>
    </div>
    """




    confidence_html = f"""
    <div class='panel-box'>
      <b style='color:{ACCENT}; font-size:1.2rem;'>Confidence Score</b>
      <p style='color:{TEXT_MUTED}; font-size:0.9rem; margin-top:0.5rem;'>
        Confidence Score is a normalised metric derived from the Bias Score, scaled to a range of 0–100.
      </p>
      <ul style='color:{TEXT_MUTED}; font-size:0.92rem; line-height:1.6;'>
        <li><span style='color:{ACCENT};'>STRONG BUY:</span> Score ≥ 80 → strong bullish bias</li>
        <li><span style='color:{ACCENT};'>BUY:</span> Score 60–79 → moderate bullish bias</li>
        <li><span style='color:{ACCENT};'>WAIT:</span> Score 40–59 → neutral / indecisive</li>
        <li><span style='color:{ACCENT};'>SELL:</span> Score 20–39 → moderate bearish bias</li>
        <li><span style='color:{ACCENT};'>STRONG SELL:</span> Score &lt; 20 → strong bearish bias</li>
      </ul>
    </div>
    """
    

    leverage_html = f"""
    <div class='panel-box'>
      <b style='color:{ACCENT}; font-size:1.2rem;'>Leverage Calculation</b>
      <p style='color:{TEXT_MUTED}; font-size:0.9rem; margin-top:0.5rem;'>
        Suggested leverage is based on both confidence level and calculated risk score:
      </p>
      
      <p style='color:{TEXT_MUTED}; font-size:0.92rem; line-height:1.6;'>
        <span style='color:{ACCENT};'>Confidence Limits:</span><br/>
        Confidence &lt; 40 → <b>max x4</b><br/>
        Confidence 40–69 → <b>max x8</b><br/>
        Confidence ≥ 70 → <b>full risk-based leverage allowed</b>
      </p>
    
    <p style='color:{ACCENT}; margin-top:1rem;'>Risk Score Calculation:</p>
    <div style='color:{TEXT_MUTED}; font-size:0.92rem; line-height:1.8; margin-left:0;'>
      <div><span style='color:{ACCENT};'>Bollinger Band Width:</span> Wider bands = more volatility (adds up to +0.1 risk)</div>
      <div><span style='color:{ACCENT};'>RSI Extremes:</span> RSI above 70 or below 30 adds +0.1 risk</div>
      <div><span style='color:{ACCENT};'>OBV Strength:</span> OBV increasing while price is above EMA21 adds +0.1 risk</div>
      <div><span style='color:{ACCENT};'>Support/Resistance Proximity:</span> If price is within 2% of key levels, adds +0.1 risk</div>
    </div>


    
      <p style='color:{TEXT_MUTED}; font-size:0.9rem; margin-top:1rem;'>
        Final leverage tiers (if confidence permits):<br/>
        • Risk &lt; 0.15 → <b>3–7×</b> leverage<br/>
        • 0.15–0.25 → <b>8–12×</b> leverage<br/>
        • ≥ 0.25 → <b>13–20×</b> leverage
      </p>
    
      <p style='color:{TEXT_MUTED}; font-size:0.9rem;'>
        <b>Tiered Output:</b><br/>
        • <b>Low Risk:</b> base<br/>
        • <b>Medium Risk:</b> base + 3 (max 14)<br/>
        • <b>High Risk:</b> base + 6 (max 20)
      </p>
    </div>
    """


    candle_html = f"""
    <div class='panel-box'>
      <b style='color:{ACCENT}; font-size:1.2rem;'>Recognised Candle Patterns</b>
      <p style='color:{TEXT_MUTED}; font-size:0.9rem;'>Used to detect potential market reversals:</p>
      <ul style='color:{TEXT_MUTED}; font-size:0.92rem; line-height:1.6;'>
        <li><b style='color:{ACCENT};'>Bullish:</b> Hammer, Bullish Engulfing, Morning Star, Piercing Line, Inverted Hammer, Three White Soldiers</li>
        <li><b style='color:{ACCENT};'>Bearish:</b> Shooting Star, Bearish Engulfing, Evening Star, Dark Cloud Cover, Hanging Man, Three Black Crows</li>
        <li><b style='color:{ACCENT};'>Neutral:</b> Doji → indicates market indecision</li>
      </ul>
    </div>
    """

    position_html = f"""
    <div class='panel-box'>
      <b style='color:{ACCENT}; font-size:1.2rem;'>Position &amp; Scalping Analysis</b>
      <p style='color:{TEXT_MUTED}; font-size:0.92rem; margin-top:0.5rem;'>
        In the <b style='color:{ACCENT};'>Position Analyser</b>, enter a symbol, timeframe and your entry price.
        The dashboard recalculates the signal and strength, shows your unrealised PnL, and provides strategy suggestions
        based on breakout or breakdown behaviour around support/resistance levels.
      </p>
    
      <p style='color:{TEXT_MUTED}; font-size:0.92rem;'>
        The scalping module has two operating modes:
        <b style='color:{ACCENT};'>Regular</b> and <b style='color:{ACCENT};'>Strict</b>.
      </p>
    
      <b style='color:{ACCENT}; font-size:1rem;'>Regular Scalping:</b>
      <p style='color:{TEXT_MUTED}; font-size:0.92rem; margin-top:0.3rem;'>
        In regular mode, the system uses a simpler 3-step confirmation to determine LONG or SHORT direction:
      </p>
      <ol style='margin-left:1.2rem; color:{TEXT_MUTED}; font-size:0.92rem;'>
        <li><b style='color:{ACCENT};'>EMA Trend:</b> 
          Bullish if EMA5 &gt; EMA13 &gt; EMA21, bearish if EMA5 &lt; EMA13 &lt; EMA21.
        </li>
        <li><b style='color:{ACCENT};'>MACD Confirmation:</b>
          Above signal line for LONG, below for SHORT (positive histogram preferred).
        </li>
        <li><b style='color:{ACCENT};'>RSI Filter:</b>
          LONG if RSI14 &gt; 55, SHORT if RSI14 &lt; 45.
        </li>
      </ol>
      <p style='color:{TEXT_MUTED}; font-size:0.92rem;'>
        Once confirmed, ATR14 is used to set entry, target, and stop-loss prices.
      </p>
    
      <b style='color:{ACCENT}; font-size:1rem;'>Strict Scalping:</b>
      <p style='color:{TEXT_MUTED}; font-size:0.92rem; margin-top:0.3rem;'>
        In strict mode, multiple layers of filtering are applied before confirming a trade direction:
      </p>
      <ol style='margin-left:1.2rem; color:{TEXT_MUTED}; font-size:0.92rem;'>
        <li><b style='color:{ACCENT};'>Trend Strength &amp; Flow:</b> 
          If ADX &lt; 20, a valid <i>volume spike</i> must be present.
        </li>
        <li><b style='color:{ACCENT};'>2/3 Regime Alignment:</b>
          Direction must align with at least two of: Supertrend, Ichimoku trend bias, VWAP.
        </li>
        <li><b style='color:{ACCENT};'>Momentum Window:</b>
          LONG: StochRSI 0.20–0.85, bullish RSI bias, MACD bullish.<br/>
          SHORT: StochRSI 0.15–0.80, bearish RSI bias, MACD bearish.
        </li>
        <li><b style='color:{ACCENT};'>Overbought / Oversold Filter:</b>
          Blocks LONG if Bollinger bias is "Overbought", SHORT if "Oversold".
        </li>
        <li><b style='color:{ACCENT};'>Volatility Floor:</b>
          ATR/price ≥ 0.15% to ensure enough market movement.
        </li>
      </ol>
    
      <p style='color:{TEXT_MUTED}; font-size:0.92rem;'>
        In both modes, ATR-based price levels are calculated as follows:
      </p>
      <ul style='margin-left:1.2rem; color:{TEXT_MUTED}; font-size:0.92rem;'>
        <li><b style='color:{ACCENT};'>LONG Setup:</b><br/>
          Entry = Close + 0.25 × ATR<br/>
          Target = Close + 1.5 × ATR<br/>
          Stop Loss = Close − 0.75 × ATR
        </li>
        <br/>
        <li><b style='color:{ACCENT};'>SHORT Setup:</b><br/>
          Entry = Close − 0.25 × ATR<br/>
          Target = Close − 1.5 × ATR<br/>
          Stop Loss = Close + 0.75 × ATR
        </li>
      </ul>
    
      <p style='color:{TEXT_MUTED}; font-size:0.92rem;'>
        Support and resistance levels are displayed for context but are not directly used in the ATR formula.
      </p>
    </div>
    """

    ml_guide_html = f"""
    <div class='panel-box'>
      <b style='color:{ACCENT}; font-size:1.2rem;'>AI Prediction</b>
      <p style='color:{TEXT_MUTED}; font-size:0.92rem; margin-top:0.5rem;'>
        The AI Prediction tool employs an advanced gradient boosting classifier (XGBoost) to analyse recent candles and estimate whether the next candle will close higher or lower.  Below is a summary of how it works.  This model is data‑driven and should be used alongside other analysis; it is not a guarantee of future results.
      </p>
      <ol style='margin-left:1.2rem; color:{TEXT_MUTED}; font-size:0.92rem; line-height:1.6;'>
        <li><span style='color:{ACCENT};'>Data Collection:</span> The model retrieves up to 500 of the most recent OHLCV bars (open, high, low, close and volume) for the selected symbol and timeframe.</li>
        <li><span style='color:{ACCENT};'>Feature Engineering:</span> For each bar it computes several technical indicators including EMA5, EMA9, EMA21, RSI14, MACD (line, signal &amp; histogram), On‑Balance Volume (OBV), Average True Range (ATR), a 10‑period momentum value (close minus the close 10 bars ago), and a 14‑period volatility measure (standard deviation of returns).</li>
        <li><span style='color:{ACCENT};'>Target Definition:</span> The training target is set to 1 if the next candle’s close is higher than the current close and 0 otherwise.</li>
        <li><span style='color:{ACCENT};'>Model Training:</span> An XGBoost classifier with a modest tree depth and 200 boosting rounds is fitted on all but the most recent row.  The algorithm learns nonlinear relationships among the indicators to estimate the probability of a bullish outcome.</li>
        <li><span style='color:{ACCENT};'>Prediction &amp; Interpretation:</span> The latest indicator values are fed into the trained model to generate a probability for an up move.  The result is categorised into <b>LONG</b> (≥ 60% probability), <b>SHORT</b> (≤ 40%), or <b>NEUTRAL</b> (between these thresholds).</li>
        <li><span style='color:{ACCENT};'>Entry &amp; Exit Levels:</span> Once a direction has been predicted, the system attempts to locate a matching scalp setup.  If such a setup is found, the corresponding entry and exit prices are displayed as <b>AI Entry</b> and <b>AI Exit</b>.  These are derived from a proprietary algorithm that combines confidence, SuperTrend, Ichimoku Cloud, VWAP and volume‑spike analysis.  If no matching setup exists, a fallback mechanism uses ATR14 to compute provisional levels; these are labelled <i>Unverified Entry</i> and <i>Unverified Exit</i> to denote lower confidence.</li>
        <li><span style='color:{ACCENT};'>Caveats:</span> Despite the sophisticated modelling technique, the prediction remains based solely on past data, which can be noisy and non‑stationary.  Use these signals as part of a broader trading plan rather than definitive advice.</li>
      </ol>
    </div>
    """




    st.markdown(signal_html, unsafe_allow_html=True)
    st.markdown(confidence_html, unsafe_allow_html=True)
    st.markdown(leverage_html, unsafe_allow_html=True)
    st.markdown(candle_html, unsafe_allow_html=True)
    st.markdown(position_html, unsafe_allow_html=True)

    st.markdown(ml_guide_html, unsafe_allow_html=True)

    market_guide_html = f"""
    <div class='panel-box'>
      <b style='color:{ACCENT}; font-size:1.2rem;'>AI Market Outlook</b>
      <p style='color:{TEXT_MUTED}; font-size:0.92rem; margin-top:0.5rem;'>
        The AI Market Outlook extends the gradient boosting approach to a broader time horizon <em>and</em> a broader set of assets.  It trains XGBoost models on 500 four‑hour candles (about 83&nbsp;days) for BTC/USDT, ETH/USDT, BNB/USDT, SOL/USDT, ADA/USDT and XRP/USDT.  Each model uses the same technical features as the AI Prediction tool (EMA5/9/21, RSI14, MACD components, OBV, ATR, momentum and volatility).
      </p>
      <p style='color:{TEXT_MUTED}; font-size:0.92rem;'>
        To gauge the overall market tone, the individual probabilities for these six assets are weighted by their current dominance percentages.  For example, if BTC dominance is 50&nbsp;%, ETH is 20&nbsp;%, BNB is 5&nbsp;%, SOL is 3&nbsp;%, ADA is 2&nbsp;% and XRP is 2&nbsp;%, the combined probability is (BTC<sub>prob</sub> × 50&nbsp;+ ETH<sub>prob</sub> × 20&nbsp;+ BNB<sub>prob</sub> × 5&nbsp;+ SOL<sub>prob</sub> × 3&nbsp;+ ADA<sub>prob</sub> × 2&nbsp;+ XRP<sub>prob</sub> × 2&nbsp;) ÷ 82.  Weighting in this way makes the indicator sensitive to capital rotation: when money flows from BTC into altcoins during an <i>alt season</i>, the probabilities of assets with higher dominance carry more influence.
      </p>
      <p style='color:{TEXT_MUTED}; font-size:0.92rem;'>
        The resulting probability is displayed as a percentage on the dashboard and colour‑coded: values above 60&nbsp;% are labelled <span style='color:{POSITIVE};'><b>Up</b></span>, below 40&nbsp;% are labelled <span style='color:{NEGATIVE};'><b>Down</b></span>, and intermediate values are considered <span style='color:{WARNING};'><b>Neutral</b></span>.  As with all model‑based signals, the AI Market Outlook should be used alongside other indicators and sound risk management.
      </p>
    </div>
    """

    st.markdown(market_guide_html, unsafe_allow_html=True)


def render_ml_tab():
    """
    Render a tab that uses a simple machine‑learning model to predict the
    probability of the next candle closing higher or lower.
    """

    st.markdown(
        f"<h2 style='color:{ACCENT};margin-bottom:0.5rem;'>AI Prediction</h2>",
        unsafe_allow_html=True,
    )
    st.markdown(
        f"<p style='color:{TEXT_MUTED};font-size:0.9rem;'>"
        "This tool trains an advanced gradient boosting (XGBoost) model on recent candles to estimate whether the next candle will close higher or lower. "
        "The output is a probability and a suggested direction (LONG/SHORT/NEUTRAL). "
        "Use this information in conjunction with other analysis; past performance does not guarantee future results.</p>",
        unsafe_allow_html=True
    )
    # Assign a unique key to avoid StreamlitDuplicateElementId errors on AI tab
    coin = st.text_input(
        "Coin Symbol (e.g. BTC/USDT)",
        value="BTC/USDT",
        key="ai_coin_input",
    ).upper()
    # Allow the user to select up to three timeframes to evaluate.  This works
    # similarly to the Position Analyser tab, enabling a multi‑timeframe view
    # of the AI prediction.  By default, we select '1h'.
    selected_timeframes = st.multiselect(
        "Select up to 3 Timeframes", ['1m','3m','5m','15m','1h','4h','1d'], default=['5m'], max_selections=3
    )
    if st.button("Predict", type="primary"):
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
                        analysis_result = analyse(df)
                        lev_base = analysis_result[1]
                        suggested_lev = lev_base if direction != "NEUTRAL" else None
                        volume_spike = analysis_result[3]
                        atr_comment = analysis_result[4]
                        candle_pattern = analysis_result[5]
                        confidence_score = analysis_result[6]
                        adx_val = analysis_result[7]
                        supertrend_trend = analysis_result[8]
                        ichimoku_trend = analysis_result[9]
                        stochrsi_k_val = analysis_result[10]
                        bollinger_bias = analysis_result[11]
                        vwap_label = analysis_result[12]
                        psar_trend = analysis_result[13]
                        williams_label = analysis_result[14]
                        cci_label = analysis_result[15]
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
                            strict_mode=False
                        )
                        # Only use the scalping result if it matches the ML direction.  This avoids
                        # showing long entries for short predictions or vice versa.  Also skip
                        # entries when the prediction is neutral.
                        if (scalp_direction and scalp_direction == direction and direction != "NEUTRAL"):
                            entry_price = entry_s
                            exit_price = target_s
                            entry_source = 'ai'
                    except Exception:
                        pass

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
                        btc_df_tf = fetch_ohlcv("BTC/USDT", tf, limit=500)
                        eth_df_tf = fetch_ohlcv("ETH/USDT", tf, limit=500)
                        bnb_df_tf = fetch_ohlcv("BNB/USDT", tf, limit=500)
                        sol_df_tf = fetch_ohlcv("SOL/USDT", tf, limit=500)
                        ada_df_tf = fetch_ohlcv("ADA/USDT", tf, limit=500)
                        xrp_df_tf = fetch_ohlcv("XRP/USDT", tf, limit=500)
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
                        # Compute a dominance‑weighted probability.  Only coins with
                        # positive dominance contribute to the combined market
                        # outlook.  Probabilities default to 0.5 when data is
                        # missing.  If the total weight is zero, fall back to 0.5.
                        probs_tf = {
                            'BTC': (btc_prob_tf, btc_dom_tf),
                            'ETH': (eth_prob_tf, eth_dom_tf),
                            'BNB': (bnb_prob_tf, bnb_dom_tf),
                            'SOL': (sol_prob_tf, sol_dom_tf),
                            'ADA': (ada_prob_tf, ada_dom_tf),
                            'XRP': (xrp_prob_tf, xrp_dom_tf),
                        }
                        weighted_sum_tf = 0.0
                        weight_total_tf = 0.0
                        for name, (p, dom) in probs_tf.items():
                            if dom and dom > 0:
                                prob_val = p if p is not None else 0.5
                                weighted_sum_tf += prob_val * dom
                                weight_total_tf += dom
                        mkt_prob_tf = weighted_sum_tf / weight_total_tf if weight_total_tf > 0 else 0.5
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
                        f"Probability of Up Move: <b>{prob*100:.1f}%</b><br>"
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

                    # Explanation bullets
                    explanations = []
                    if volume_spike:
                        explanations.append("📈 <b>Volume Spike detected</b> – sudden increase in trading activity.")
                    # ATR comment
                    atr_clean = atr_comment.replace("▲", "").replace("▼", "").replace("–", "").strip()
                    if atr_clean == "Moderate":
                        explanations.append("🔄 <b>Volatility is moderate</b> – stable price conditions.")
                    elif atr_clean == "High":
                        explanations.append("⚠️ <b>Volatility is high</b> – expect sharp moves.")
                    elif atr_clean == "Low":
                        explanations.append("🟢 <b>Volatility is low</b> – steady market behaviour.")
                    # Candle pattern
                    if candle_pattern:
                        explanations.append(f"🕯️ <b>Candle pattern:</b> {candle_pattern}")
                    # ADX
                    if not np.isnan(adx_val):
                        explanations.append(f"📊 <b>ADX:</b> {format_adx(adx_val)}")
                    # SuperTrend
                    if supertrend_trend in ["Bullish", "Bearish", "Neutral"]:
                        trend_icon = "▲" if supertrend_trend == "Bullish" else "▼" if supertrend_trend == "Bearish" else "–"
                        explanations.append(f"📈 <b>SuperTrend:</b> {trend_icon} {supertrend_trend}")
                    # Ichimoku Cloud
                    if ichimoku_trend == "Bullish":
                        explanations.append("☁️ <b>Ichimoku Cloud:</b> Price is above the cloud → <span style='color:limegreen;'>Bullish</span> signal.")
                    elif ichimoku_trend == "Bearish":
                        explanations.append("☁️ <b>Ichimoku Cloud:</b> Price is below the cloud → <span style='color:red;'>Bearish</span> signal.")
                    elif ichimoku_trend == "Neutral":
                        explanations.append("☁️ <b>Ichimoku Cloud:</b> Price is inside the cloud → <span style='color:orange;'>Neutral</span> state.")
                    # Stochastic RSI
                    if not np.isnan(stochrsi_k_val):
                        if stochrsi_k_val < 0.2:
                            explanations.append("🟢 <b>Stochastic RSI:</b> Oversold (< 0.2) – possible rebound.")
                        elif stochrsi_k_val > 0.8:
                            explanations.append("🔴 <b>Stochastic RSI:</b> Overbought (> 0.8) – possible pullback.")
                        else:
                            explanations.append("🟡 <b>Stochastic RSI:</b> Neutral zone.")
                    # Bollinger Bands
                    if "Overbought" in bollinger_bias:
                        explanations.append("🔴 <b>Bollinger:</b> Price is strongly above upper band — <span style='color:red;'>overbought zone</span>.")
                    elif "Oversold" in bollinger_bias:
                        explanations.append("🟢 <b>Bollinger:</b> Price is strongly below lower band — <span style='color:limegreen;'>oversold zone</span>.")
                    elif "Near" in bollinger_bias:
                        explanations.append(f"🟡 <b>Bollinger:</b> {bollinger_bias} — price is near edge of band.")
                    else:
                        explanations.append("➖ <b>Bollinger:</b> Price is inside bands — <span style='color:gray;'>neutral zone</span>.")
                    # VWAP
                    if vwap_label == "🟢 Above":
                        explanations.append("🟢 <b>VWAP:</b> Price is above VWAP — <span style='color:limegreen;'>bullish bias</span>.")
                    elif vwap_label == "🔴 Below":
                        explanations.append("🔴 <b>VWAP:</b> Price is below VWAP — <span style='color:red;'>bearish bias</span>.")
                    elif "Near" in vwap_label:
                        explanations.append("🟡 <b>VWAP:</b> Price is near VWAP — <span style='color:gray;'>neutral zone</span>.")
                    # Parabolic SAR
                    if "Bullish" in psar_trend:
                        explanations.append("🟢 <b>Parabolic SAR:</b> Price is above PSAR — <span style='color:limegreen;'>bullish continuation</span>.")
                    elif "Bearish" in psar_trend:
                        explanations.append("🔴 <b>Parabolic SAR:</b> Price is below PSAR — <span style='color:red;'>bearish reversal</span>.")
                    elif psar_trend == "":
                        explanations.append("➖ <b>Parabolic SAR:</b> Not available — insufficient data or calculation error.")
                    # Williams %R
                    if williams_label == "🟢 Oversold":
                        explanations.append("🟢 <b>Williams %R:</b> Below −80 — <span style='color:limegreen;'>oversold</span> zone.")
                    elif williams_label == "🔴 Overbought":
                        explanations.append("🔴 <b>Williams %R:</b> Above −20 — <span style='color:red;'>overbought</span> zone.")
                    elif williams_label == "🟡 Neutral":
                        explanations.append("🟡 <b>Williams %R:</b> Neutral range.")
                    # CCI
                    if "Oversold" in cci_label:
                        explanations.append("🟢 <b>CCI:</b> Oversold (< -100) — potential bullish reversal.")
                    elif "Overbought" in cci_label:
                        explanations.append("🔴 <b>CCI:</b> Overbought (> 100) — potential bearish reversal.")
                    elif "Neutral" in cci_label:
                        explanations.append("🟡 <b>CCI:</b> Neutral range.")
                    # Display explanations as markdown list
                    if explanations:
                        st.markdown("<ul style='padding-left: 1.2rem;'>" + "".join([f"<li style='margin-bottom:4px;'>{exp}</li>" for exp in explanations]) + "</ul>", unsafe_allow_html=True)

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
                            wma20 = pta.wma(df['close'], length=20)
                            wma50 = pta.wma(df['close'], length=50)
                            fig.add_trace(go.Scatter(
                                x=df['timestamp'], y=wma20, mode='lines',
                                name="WMA20", line=dict(color='#34D399', width=1, dash='dot')
                            ))
                            fig.add_trace(go.Scatter(
                                x=df['timestamp'], y=wma50, mode='lines',
                                name="WMA50", line=dict(color='#10B981', width=1, dash='dash')
                            ))
                        except Exception:
                            pass
                        # Layout settings
                        fig.update_layout(
                            height=380,
                            template='plotly_dark',
                            margin=dict(l=20, r=20, t=30, b=30),
                            xaxis_rangeslider_visible=False,
                            showlegend=True,
                            legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='left', x=0)
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    except Exception:
                        pass

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
                        st.plotly_chart(rsi_fig, use_container_width=True)
                    except Exception:
                        pass

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
                        st.plotly_chart(macd_fig, use_container_width=True)
                    except Exception:
                        pass

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
                        st.plotly_chart(volume_fig, use_container_width=True)
                    except Exception:
                        pass

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
                        # Support/resistance from last 20 bars
                        recent = df.tail(20)
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
                    except Exception:
                        pass

def run_backtest(df: pd.DataFrame, threshold: float = 70, exit_after: int = 5) -> tuple[pd.DataFrame, str]:
    results = []

    for i in range(30, len(df) - exit_after):
        df_slice = df.iloc[:i + 1].copy()
        try:
            result = analyse(df_slice)
            raw_signal = result[0]           # "STRONG BUY", "BUY", "WAIT", "SELL", "STRONG SELL"
            conf_score = result[6]
        except Exception:
            continue

        # Normalize to LONG/SHORT/WAIT
        sig_plain = "LONG" if raw_signal in ["STRONG BUY", "BUY"] else \
                    "SHORT" if raw_signal in ["STRONG SELL", "SELL"] else "WAIT"

        if conf_score >= threshold and sig_plain in ["LONG", "SHORT"]:
            entry_price = df['close'].iloc[i]
            future_price = df['close'].iloc[i + exit_after]
            pnl = (future_price - entry_price) / entry_price * 100
            if sig_plain == "SHORT":
                pnl *= -1

            results.append({
                "Date": df['timestamp'].iloc[i],
                "Confidence": round(conf_score, 1),
                "Signal": sig_plain,          # LONG / SHORT
                "Entry": entry_price,
                "Exit": future_price,
                "PnL (%)": round(pnl, 2)
            })

    df_results = pd.DataFrame(results)

    if df_results.empty:
        summary_html = (
            "<div style='color:#FFB000;margin-top:1rem;'>"
            "<p><b>Total Signals:</b> 0</p>"
            "<p><b>Win Rate:</b> 0.0%</p>"
            "<p><b>Average PnL:</b> 0.00%</p>"
            "</div>"
        )
        return df_results, summary_html

    wins = (df_results["PnL (%)"] > 0).sum()
    losses = (df_results["PnL (%)"] <= 0).sum()
    winrate = (wins / (wins + losses)) * 100 if (wins + losses) > 0 else 0.0

    # Compute simple returns and Sharpe ratio (mean/standard deviation).  A
    # small epsilon is added to the denominator to avoid division by zero.
    returns = df_results["PnL (%)"].astype(float) / 100.0
    mean_return = returns.mean()
    std_return = returns.std()
    sharpe_ratio = mean_return / (std_return + 1e-9)

    summary_html = f"""
    <div style='color:#FFB000;margin-top:1rem;'>
        <p><b>Total Signals:</b> {wins + losses}</p>
        <p><b>Win Rate:</b> {winrate:.1f}%</p>
        <p><b>Average PnL:</b> {df_results['PnL (%)'].mean():.2f}%</p>
        <p><b>Sharpe Ratio:</b> {sharpe_ratio:.2f}</p>
    </div>
    """

    return df_results, summary_html



def render_backtest_tab():
    """Render the Backtest tab to simulate past signals."""
    st.markdown(f"<h2 style='color:{ACCENT};'>Backtest Simulator</h2>", unsafe_allow_html=True)

    # Assign a unique key to avoid StreamlitDuplicateElementId errors on Backtest tab
    coin = st.text_input(
        "Coin Symbol",
        value="BTC/USDT",
        key="backtest_coin_input",
    ).upper()
    timeframe = st.selectbox("Timeframe", ["3m", "5m", "15m", "1h", "4h", "1d"], index=2)
    limit = st.slider("Number of Candles", 100, 1000, step=100, value=500)
    threshold = st.slider("Confidence Score Threshold", 0, 100, step=5, value=70)
    exit_after = st.slider("Exit After N Candles", 1, 20, step=1, value=5)

    if st.button("Run Backtest"):
        st.info("Fetching data and analysing, please wait...")

        df = fetch_ohlcv(coin, timeframe, limit)

        if df is not None and not df.empty:
            st.success(f"Fetched {len(df)} candles. Running analysis...")

            try:
                result_df, summary_html = run_backtest(df, threshold, exit_after)

                if not result_df.empty:
                    st.markdown(summary_html, unsafe_allow_html=True)

                    # Format display
                    styled_df = result_df.copy()
                    styled_df["Entry"] = styled_df["Entry"].apply(lambda x: f"${x:,.4f}")
                    styled_df["Exit"] = styled_df["Exit"].apply(lambda x: f"${x:,.4f}")
                    styled_df["PnL (%)"] = styled_df["PnL (%)"].apply(lambda x: f"{x:.2f}%")
                    styled_df["Confidence"] = styled_df["Confidence"].apply(lambda x: f"{x:,.1f}")
                    

                    st.dataframe(styled_df)

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


def main():
    """Entry point for the Streamlit app."""
    tabs = st.tabs(["Market", "Spot", "Position", "AI Prediction", "Backtest", "Analysis Guide"])
    with tabs[0]:
        render_market_tab()
    with tabs[1]:
        render_spot_tab()
    with tabs[2]:
        render_position_tab()
    with tabs[3]:
        render_ml_tab()
    with tabs[4]:
        render_backtest_tab()
    with tabs[5]:
        render_guide_tab()

main()
