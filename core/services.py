"""Application service layer for data access and analysis wrappers."""

from __future__ import annotations

import time
from typing import Tuple

import pandas as pd
import requests
import streamlit as st

from core.data import (
    fetch_ohlcv as fetch_ohlcv_core,
    fetch_ohlcv_cached as fetch_ohlcv_cached_core,
    get_btc_eth_prices as get_btc_eth_prices_core,
    get_exchange as get_exchange_core,
    get_fear_greed as get_fear_greed_core,
    get_major_ohlcv_bundle as get_major_ohlcv_bundle_core,
    get_market_indices as get_market_indices_core,
    get_markets as get_markets_core,
    get_price_change as get_price_change_core,
    normalize_coin_input as normalize_coin_input_core,
    symbol_variants as symbol_variants_core,
    validate_coin_symbol as validate_coin_symbol_core,
)
from core.advanced_analysis import (
    calculate_fibonacci_levels as calculate_fibonacci_levels_core,
    calculate_volume_profile as calculate_volume_profile_core,
    detect_divergence as detect_divergence_core,
    detect_market_regime as detect_market_regime_core,
    monte_carlo_simulation as monte_carlo_simulation_core,
)
from core.market_data import (
    fetch_top_gainers_losers as fetch_top_gainers_losers_core,
    fetch_trending_coins as fetch_trending_coins_core,
    get_top_volume_usdt_symbols as get_top_volume_usdt_symbols_core,
)
from core.ml import ml_ensemble_predict as ml_ensemble_predict_core
from core.ml import ml_predict_direction as ml_predict_direction_core
from core.policy import UK_SAFE_EXCHANGE_FALLBACKS
from core.risk import calculate_risk_metrics as calculate_risk_metrics_core
from core.scalping import get_scalping_entry_target as get_scalping_entry_target_core
from core.signals import (
    AnalysisResult,
    analyse as analyse_core,
    detect_candle_pattern as detect_candle_pattern_core,
    detect_volume_spike as detect_volume_spike_core,
    sr_lookback as sr_lookback_core,
)
from ui.theme import NEGATIVE, NEON_BLUE, NEON_PURPLE, POSITIVE, TEXT_MUTED, WARNING


def _debug(msg: str) -> None:
    if st.session_state.get("debug_mode", False):
        st.sidebar.write(msg)


def _http_get_json(url: str, params: dict | None = None, timeout: int = 10,
                   retries: int = 3, backoff_sec: float = 0.7):
    """GET JSON with small retry/backoff for transient API failures."""
    last_exc = None
    for attempt in range(1, retries + 1):
        try:
            resp = requests.get(url, params=params, timeout=timeout)
            if resp.status_code == 200:
                return resp.json()
            _debug(f"HTTP {resp.status_code} for {url} (attempt {attempt}/{retries})")
        except Exception as exc:
            last_exc = exc
            _debug(f"HTTP error for {url} (attempt {attempt}/{retries}): {exc}")
        if attempt < retries:
            time.sleep(backoff_sec * attempt)
    if last_exc:
        _debug(f"Request failed after {retries} attempts: {url} ({last_exc})")
    return None

_EXCHANGE_CONFIGS = list(UK_SAFE_EXCHANGE_FALLBACKS)

@st.cache_resource(show_spinner=False)
def get_exchange():
    return get_exchange_core(_EXCHANGE_CONFIGS)

EXCHANGE = get_exchange()

# Fetch BTC and ETH prices in USD from CoinGecko
@st.cache_data(ttl=120, show_spinner=False)
def get_btc_eth_prices():
    try:
        return get_btc_eth_prices_core()
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
        return get_market_indices_core()
    except Exception as e:
        # Log the error and return zeros for all values to avoid breaking the
        # dashboard.  Using ints ensures consistent return types across the
        # success and failure paths.
        print(f"get_market_indices error: {e}")
        return 0, 0, 0, 0, 0.0, 0, 0, 0, 0


# Fetch fear and greed index from alternative.me
def get_fear_greed():
    try:
        return get_fear_greed_core()
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
        return get_markets_core(EXCHANGE)
    except Exception as e:
        st.warning(f"Failed to load markets ({EXCHANGE.id}): {e}")
        return {}

MARKETS = get_markets()


def _symbol_variants(symbol: str) -> list[str]:
    return symbol_variants_core(symbol)


def _normalize_coin_input(raw: str) -> str:
    return normalize_coin_input_core(raw)


def _validate_coin_symbol(symbol: str) -> str | None:
    return validate_coin_symbol_core(symbol)


def _sr_lookback(timeframe: str | None = None) -> int:
    return sr_lookback_core(timeframe)


# Fetch price change percentage for a given symbol via ccxt
@st.cache_data(ttl=60, show_spinner=False)
def get_price_change(symbol: str) -> float | None:
    return get_price_change_core(EXCHANGE, symbol)

# Fetch OHLCV data for a symbol and timeframe
@st.cache_data(ttl=60, show_spinner=False)
def fetch_ohlcv_cached(symbol: str, timeframe: str, limit: int = 120) -> pd.DataFrame:
    """Fetch OHLCV data via ccxt and return a DataFrame. Raises on error."""
    return fetch_ohlcv_cached_core(EXCHANGE, symbol, timeframe, limit)

def fetch_ohlcv(symbol: str, timeframe: str, limit: int = 120) -> pd.DataFrame | None:
    """Safe OHLCV fetch. Tries symbol variants on exchange, falls back to CoinGecko."""
    return fetch_ohlcv_core(EXCHANGE, symbol, timeframe, limit, fetch_ohlcv_cached)


@st.cache_data(ttl=120, show_spinner=False)
def get_major_ohlcv_bundle(timeframe: str, limit: int = 500) -> dict[str, pd.DataFrame | None]:
    """Fetch a bundle of major market OHLCV frames for a timeframe."""
    return get_major_ohlcv_bundle_core(fetch_ohlcv, timeframe, limit=limit)


def detect_volume_spike(df: pd.DataFrame, window: int = 20, multiplier: float = 2.0) -> bool:
    return detect_volume_spike_core(df, window=window, multiplier=multiplier)

def detect_candle_pattern(df: pd.DataFrame) -> str:
    return detect_candle_pattern_core(df)


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
    return analyse_core(df, debug_fn=_debug)


def get_scalping_entry_target(
    df: pd.DataFrame,
    confidence_score: float,
    supertrend_trend: str,
    ichimoku_trend: str,
    vwap_label: str,
    volume_spike: bool,
    strict_mode: bool = True
):
    return get_scalping_entry_target_core(
        df,
        confidence_score,
        supertrend_trend,
        ichimoku_trend,
        vwap_label,
        volume_spike,
        strict_mode=strict_mode,
        sr_lookback_fn=_sr_lookback,
    )

# === Machine Learning Prediction ===
def ml_predict_direction(df: pd.DataFrame) -> tuple[float, str]:
    return ml_predict_direction_core(df, debug_fn=_debug)


# ╔══════════════════════════════════════════════════════════════╗
# ║                  ADVANCED ANALYSIS FUNCTIONS                  ║
# ╚══════════════════════════════════════════════════════════════╝

def ml_ensemble_predict(df: pd.DataFrame) -> tuple[float, str, dict]:
    return ml_ensemble_predict_core(df)


def calculate_fibonacci_levels(df: pd.DataFrame, lookback: int = 100) -> dict:
    return calculate_fibonacci_levels_core(df, lookback=lookback)


def monte_carlo_simulation(df: pd.DataFrame, num_simulations: int = 500,
                           num_days: int = 30) -> dict:
    return monte_carlo_simulation_core(df, num_simulations=num_simulations, num_days=num_days)


def detect_divergence(df: pd.DataFrame) -> list[dict]:
    return detect_divergence_core(
        df,
        positive_color=POSITIVE,
        negative_color=NEGATIVE,
        warning_color=WARNING,
    )


def calculate_risk_metrics(df: pd.DataFrame, risk_free_rate: float = 0.02,
                           timeframe: str = "1d") -> dict:
    """Backward-compatible wrapper to the core risk engine."""
    return calculate_risk_metrics_core(df, risk_free_rate=risk_free_rate, timeframe=timeframe)


def calculate_volume_profile(df: pd.DataFrame, num_bins: int = 30) -> dict:
    return calculate_volume_profile_core(df, num_bins=num_bins)


def detect_market_regime(df: pd.DataFrame) -> dict:
    return detect_market_regime_core(
        df,
        positive_color=POSITIVE,
        neon_blue_color=NEON_BLUE,
        neon_purple_color=NEON_PURPLE,
        negative_color=NEGATIVE,
        warning_color=WARNING,
        text_muted_color=TEXT_MUTED,
    )


@st.cache_data(ttl=300, show_spinner=False)
def fetch_trending_coins() -> list[dict]:
    return fetch_trending_coins_core(_http_get_json)


@st.cache_data(ttl=120, show_spinner=False)
def _fetch_top_gainers_losers_cached(limit: int = 20) -> tuple[list, list]:
    return fetch_top_gainers_losers_core(_http_get_json, limit=limit)


def fetch_top_gainers_losers(limit: int = 20) -> tuple[list, list]:
    gainers, losers = _fetch_top_gainers_losers_cached(limit=limit)
    if gainers or losers:
        return gainers, losers
    # Avoid serving cached empty payloads for the full TTL when CoinGecko has
    # transient 429/timeout responses.
    _debug("fetch_top_gainers_losers cache returned empty; retrying uncached fetch.")
    return fetch_top_gainers_losers_core(_http_get_json, limit=limit)


def get_top_volume_usdt_symbols(top_n: int = 100, vs_currency: str = "usd"):
    return get_top_volume_usdt_symbols_core(
        _http_get_json,
        MARKETS,
        _debug,
        top_n=top_n,
        vs_currency=vs_currency,
    )
