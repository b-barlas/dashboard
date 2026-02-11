import datetime
import streamlit as st
import pandas as pd
import numpy as np
import requests
import plotly.graph_objs as go
import ccxt
import ta
from typing import Tuple
from sklearn.linear_model import LogisticRegression


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
    page_title="Crypto Market Dashboard",
    page_icon="📊",
    layout="wide",
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
    </style>
    """,
    unsafe_allow_html=True
)


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
def get_fear_greed():
    try:
        data = requests.get("https://api.alternative.me/fng/?limit=1", timeout=10).json()
        value = int(data.get("data", [{}])[0].get("value", 0))
        label = data.get("data", [{}])[0].get("value_classification", "Unknown")
        return value, label
    except Exception as e:
        print(f"get_fear_greed error: {e}")
        return 0, "Unknown"

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
        st.warning(f"Markets yüklenemedi ({EXCHANGE.id}): {e}")
        return {}

MARKETS = get_markets()


# Fetch price change percentage for a given symbol via ccxt
@st.cache_data(ttl=60, show_spinner=False)
def get_price_change(symbol: str) -> float | None:
    try:
        ticker = EXCHANGE.fetch_ticker(symbol)
        percent = ticker.get('percentage')
        return round(percent, 2) if percent is not None else None
    except Exception as e:
        _debug(f"get_price_change failed: {symbol} → {e}")
        return None

# Fetch OHLCV data for a symbol and timeframe
@st.cache_data(ttl=60, show_spinner=False)
def fetch_ohlcv_cached(symbol: str, timeframe: str, limit: int = 120) -> pd.DataFrame:
    """Fetch OHLCV data via ccxt and return a DataFrame. Raises on error."""
    data = EXCHANGE.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
    df = pd.DataFrame(data, columns=["timestamp", "open", "high", "low", "close", "volume"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
    return df

def fetch_ohlcv(symbol: str, timeframe: str, limit: int = 120) -> pd.DataFrame | None:
    """Safe OHLCV fetch. Returns None on failure; shows details in Debug mode."""
    try:
        return fetch_ohlcv_cached(symbol, timeframe, limit)
    except Exception as e:
        _debug(f"fetch_ohlcv failed: {symbol} {timeframe} (limit={limit}) → {e}")
        return None


@st.cache_data(ttl=120, show_spinner=False)
def get_major_ohlcv_bundle(timeframe: str, limit: int = 500) -> dict[str, pd.DataFrame | None]:
    """Fetch a bundle of major market OHLCV frames for a timeframe."""
    majors = ["BTC/USDT", "ETH/USDT", "BNB/USDT", "SOL/USDT", "ADA/USDT", "XRP/USDT"]
    out: dict[str, pd.DataFrame | None] = {}
    for sym in majors:
        out[sym] = fetch_ohlcv(sym, timeframe, limit=limit)
    return out
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

def analyse(df: pd.DataFrame) -> tuple[str, int, str, bool, str, str, float, float, str, str, float, str, str, str, str, str]:

    if df is None or len(df) < 120:
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
    _vwap_den = df["volume"].cumsum().replace(0, np.nan)
    df["vwap"] = (df["typical_price"] * df["volume"]).cumsum() / _vwap_den

    # Parabolic SAR
    try:
        psar_ind = ta.trend.PSARIndicator(high=df['high'], low=df['low'], close=df['close'])
        psar_up = psar_ind.psar_up()
        psar_down = psar_ind.psar_down()
        df["psar"] = psar_up.fillna(psar_down)
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
    
    # Use the last *closed* candle to avoid repainting on live/updating charts
    latest = df.iloc[-2] if len(df) >= 2 else df.iloc[-1]

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

    # === BIAS SCORE CALCULATION ===
    # Categories: Trend, Momentum, Volume, Volatility
    
    # SuperTrend calculation
    try:
        st_data = _supertrend(df['high'], df['low'], df['close'], length=10, multiplier=3.0)
        df['supertrend'] = st_data[st_data.columns[0]]
    except Exception as e:
        print("SuperTrend Error:", e)
        df['supertrend'] = np.nan

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
    
    # OBV Trend
    if df["obv"].iloc[-1] > df["obv"].iloc[-5]:
        volume_signals.append(0.5)
    elif df["obv"].iloc[-1] < df["obv"].iloc[-5]:
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
    volatility_factor = min(bollinger_width / latest["close"], 0.1)
    rsi_factor = 0.1 if latest["rsi"] > 70 or latest["rsi"] < 30 else 0
    obv_factor = 0.1 if df["obv"].iloc[-1] > df["obv"].iloc[-5] and latest["close"] > latest["ema21"] else 0
    recent = df.tail(20)
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

    return signal, lev_base, comment, volume_spike, atr_comment, candle_pattern, confidence_score, adx_val, supertrend_trend, ichimoku_trend, stochrsi_k_val, bollinger_bias, vwap_label, psar_trend, williams_label, cci_label


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

        # Scale features for better performance
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X[:-1])  # Exclude last point (no target)
        X_last_scaled = scaler.transform(X[-1:])

        # Train Gradient Boosting model
        model = GradientBoostingClassifier(
            n_estimators=100,
            max_depth=4,
            learning_rate=0.1,
            random_state=42
        )
        model.fit(X_scaled, y[:-1])

        # Predict probability
        prob_up = float(model.predict_proba(X_last_scaled)[0][1])
    except Exception as e:
        # Fallback to LogisticRegression if Gradient Boosting fails
        print(f"GradientBoosting failed ({e}), falling back to LogisticRegression")
        try:
            model = LogisticRegression(max_iter=1000)
            model.fit(X[:-1], y[:-1])
            prob_up = float(model.predict_proba(X[-1:].reshape(1, -1))[0][1])
        except Exception:
            return 0.5, "NEUTRAL"

    direction = "LONG" if prob_up >= 0.6 else ("SHORT" if prob_up <= 0.4 else "NEUTRAL")
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
        mkt_df = get_major_ohlcv_bundle(selected_timeframe, limit=500).get('BTC/USDT')
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
        bundle_behav = get_major_ohlcv_bundle('4h', limit=500)
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
    # Strict scalp mode is always enabled (non-strict path removed).


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
            # ml_predict_direction function fits a Gradient Boosting classifier to
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
                strict_mode=True
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
    
            # Check for Signal-AI divergence
            signal_direction = "LONG" if signal in ['STRONG BUY', 'BUY'] else ("SHORT" if signal in ['STRONG SELL', 'SELL'] else "NEUTRAL")
            
            # Format AI prediction with divergence warning if needed
            if signal_direction == "LONG" and ai_direction == "SHORT":
                ai_display = "⚠️ SHORT (Divergence)"
            elif signal_direction == "SHORT" and ai_direction == "LONG":
                ai_display = "⚠️ LONG (Divergence)"
            elif signal_direction != "NEUTRAL" and ai_direction == "NEUTRAL":
                ai_display = f"{ai_direction} (Weak)"
            else:
                ai_display = ai_direction  # Aligned or both neutral
            
            results.append({
                'Coin': base,
                'Price ($)': f"{price:,.2f}",
                'Signal': signal_plain(signal),
                'Confidence': confidence_score_badge(confidence_score),
                # Include AI prediction with divergence warning
                'AI Prediction': ai_display,
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
            wma20 = _wma(df['close'], length=20)
            wma50 = _wma(df['close'], length=50)
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
        ticker = EXCHANGE.fetch_ticker(coin)
        default_entry_price = float(ticker.get('last', 0) or 0)
    except Exception:
        default_entry_price = 0.0

    entry_price = st.number_input("Entry Price", min_value=0.0, format="%.4f", value=default_entry_price)
    direction = st.selectbox("Position Direction", ["LONG", "SHORT"])

    # Strict scalp mode is always enabled (non-strict path removed).
   
    if st.button("Analyse Position", type="primary"):
        tf_order = {'1m': 1, '3m': 2, '5m': 3, '15m': 4, '1h': 5, '4h': 6, '1d': 7}
        largest_tf = max(selected_timeframes, key=lambda tf: tf_order[tf])

        cols = st.columns(len(selected_timeframes))

        for idx, tf in enumerate(selected_timeframes):
            with cols[idx]:
                df = fetch_ohlcv(coin, tf, limit=200)
                if df is None or len(df) < 120:
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


        # Confidence Levels - rendered in parts
    st.markdown('<div class="panel-box"><b style="color:#06D6A0; font-size:1.3rem;">📊 Understanding Confidence Levels</b><p style="color:#E5E7EB; font-size:0.95rem; margin-top:1rem;">The confidence score tells you how likely the signal is to work out. Higher confidence = higher probability of success.</p></div>', unsafe_allow_html=True)
    
    st.markdown('<div style="padding:12px; margin:10px 0; background: linear-gradient(90deg, rgba(6,214,160,0.25) 0%, transparent 100%); border-left:4px solid #06D6A0; border-radius:6px;"><b style="color:#06D6A0;">80-100%: STRONG BUY</b><p style="color:#E5E7EB; margin:0.5rem 0 0 0;"><b>What it means:</b> All indicators aligned. Best setup!</p><p style="color:#8CA1B6; margin:0.5rem 0 0 0;"><b>Action:</b> Strong buy signal. Best entries.</p></div>', unsafe_allow_html=True)
    
    st.markdown('<div style="padding:12px; margin:10px 0; background: linear-gradient(90deg, rgba(6,214,160,0.12) 0%, transparent 100%); border-left:4px solid #06D6A0; border-radius:6px;"><b style="color:#06D6A0;">65-80%: BUY</b><p style="color:#E5E7EB; margin:0.5rem 0 0 0;"><b>What it means:</b> Good setup. High probability.</p><p style="color:#8CA1B6; margin:0.5rem 0 0 0;"><b>Action:</b> Good entry. Smaller position than STRONG BUY.</p></div>', unsafe_allow_html=True)
    
    st.markdown('<div style="padding:12px; margin:10px 0; background: linear-gradient(90deg, rgba(255,209,102,0.15) 0%, transparent 100%); border-left:4px solid #FFD166; border-radius:6px;"><b style="color:#FFD166;">35-65%: WAIT</b><p style="color:#E5E7EB; margin:0.5rem 0 0 0;"><b>What it means:</b> Mixed signals. No edge.</p><p style="color:#8CA1B6; margin:0.5rem 0 0 0;"><b>Action:</b> DO NOT TRADE. Wait for better conditions.</p></div>', unsafe_allow_html=True)
    
    st.markdown('<div style="padding:12px; margin:10px 0; background: linear-gradient(90deg, rgba(239,71,111,0.2) 0%, transparent 100%); border-left:4px solid #EF476F; border-radius:6px;"><b style="color:#EF476F;">20-35%: SELL | 0-20%: STRONG SELL</b><p style="color:#E5E7EB; margin:0.5rem 0 0 0;"><b>What it means:</b> Bearish setup.</p><p style="color:#8CA1B6; margin:0.5rem 0 0 0;"><b>Action:</b> Short opportunity.</p></div>', unsafe_allow_html=True)

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
                            strict_mode=True
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

def run_backtest(df: pd.DataFrame, threshold: float = 70, exit_after: int = 5, commission: float = 0.001) -> tuple[pd.DataFrame, str]:
    """
    BACKTEST ENGINE with comprehensive metrics
    
    Args:
        df: OHLCV dataframe
        threshold: Minimum confidence score to take trades
        exit_after: Number of candles to hold position
        commission: Trading commission (default 0.1% = 0.001)
    
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

    for i in range(120, len(df) - exit_after):
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
            
            # Calculate PnL with commission (entry + exit)
            if sig_plain == "LONG":
                pnl = ((future_price - entry_price) / entry_price - 2 * commission) * 100
            else:  # SHORT
                pnl = ((entry_price - future_price) / entry_price - 2 * commission) * 100
            
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
    st.markdown(f"<p style='color:{TEXT_MUTED};'>Test your strategy with realistic trading conditions</p>", unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    
    with col1:
        coin = st.text_input(
            "Coin Symbol",
            value="BTC/USDT",
            key="backtest_coin_input",
        ).upper()
        timeframe = st.selectbox("Timeframe", ["3m", "5m", "15m", "1h", "4h", "1d"], index=2)
        limit = st.slider("Number of Candles", 100, 1000, step=100, value=500)
    
    with col2:
        threshold = st.slider("Confidence Threshold (%)", 50, 90, step=5, value=65,
                             help="Only take trades with confidence above this level")
        exit_after = st.slider("Exit After N Candles", 1, 20, step=1, value=5)
        commission = st.slider("Commission (%)", 0.0, 1.0, step=0.05, value=0.1,
                              help="Trading fee per trade (typical spot: 0.1%)") / 100

    if st.button("🚀 Run Backtest", type="primary"):
        st.info("Fetching data and running comprehensive analysis...")

        df = fetch_ohlcv(coin, timeframe, limit)

        if df is not None and not df.empty:
            st.success(f"✅ Fetched {len(df)} candles. Running backtest...")

            try:
                result_df, summary_html = run_backtest(df, threshold, exit_after, commission)

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
                        st.metric("Commission", f"{commission*100:.2f}%", "per trade")
                    
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
                        st.plotly_chart(equity_fig, use_container_width=True)

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
                    

                    st.dataframe(styled_df, use_container_width=True)

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
