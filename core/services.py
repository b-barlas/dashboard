"""Application service layer for data access and analysis wrappers."""

from __future__ import annotations

import time

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
from core.scalping import (
    get_scalping_entry_target as get_scalping_entry_target_core,
    scalp_quality_gate as scalp_quality_gate_core,
)
from core.telemetry import record_event
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


def _safe_float(v) -> float | None:
    try:
        f = float(v)
        return f if pd.notna(f) else None
    except Exception:
        return None


def _is_finite(v) -> bool:
    return _safe_float(v) is not None


def _is_positive(v) -> bool:
    f = _safe_float(v)
    return f is not None and f > 0


def _http_get_json(url: str, params: dict | None = None, timeout: int = 10,
                   retries: int = 3, backoff_sec: float = 0.7):
    """GET JSON with small retry/backoff for transient API failures."""
    last_exc = None
    for attempt in range(1, retries + 1):
        try:
            resp = requests.get(url, params=params, timeout=timeout)
            if resp.status_code == 200:
                record_event(st, "http_success", status="ok", source=url)
                return resp.json()
            _debug(f"HTTP {resp.status_code} for {url} (attempt {attempt}/{retries})")
            record_event(st, "http_failure", status="error", source=url, detail=f"status={resp.status_code}")
        except Exception as exc:
            last_exc = exc
            _debug(f"HTTP error for {url} (attempt {attempt}/{retries}): {exc}")
            record_event(st, "http_failure", status="error", source=url, detail=str(exc))
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
    several leading altcoins (BNB, SOL, ADA, XRP). Dominance values are
    returned as percentage points with 1 decimal precision.
    
    If the API call fails, zero-like defaults are returned to keep the app alive.
    Callers should treat all-zero dominance payloads as unavailable data.
    """
    try:
        return get_market_indices_core()
    except Exception as e:
        # Log the error and return zeros for all values to avoid breaking the
        # dashboard.  Using numeric defaults ensures consistent return types across the
        # success and failure paths.
        print(f"get_market_indices error: {e}")
        return 0.0, 0.0, 0, 0, 0.0, 0.0, 0.0, 0.0, 0.0


# Fetch fear and greed index from alternative.me
def get_fear_greed():
    try:
        return get_fear_greed_core()
    except Exception as e:
        _debug(f"get_fear_greed error: {e}")
        return None, "Unavailable"


@st.cache_data(ttl=90, show_spinner=False)
def _fetch_btc_eth_from_coingecko_with_change() -> dict[str, float | None]:
    data = _http_get_json(
        "https://api.coingecko.com/api/v3/simple/price",
        params={
            "ids": "bitcoin,ethereum",
            "vs_currencies": "usd",
            "include_24hr_change": "true",
        },
        timeout=10,
        retries=3,
    ) or {}
    btc = data.get("bitcoin", {}) if isinstance(data, dict) else {}
    eth = data.get("ethereum", {}) if isinstance(data, dict) else {}
    return {
        "btc_price": _safe_float(btc.get("usd")),
        "btc_change": _safe_float(btc.get("usd_24h_change")),
        "eth_price": _safe_float(eth.get("usd")),
        "eth_change": _safe_float(eth.get("usd_24h_change")),
    }


def _fetch_pair_price_change_from_exchange(symbol: str) -> tuple[float | None, float | None]:
    for variant in symbol_variants_core(symbol):
        try:
            ticker = EXCHANGE.fetch_ticker(variant)
            price = _safe_float(ticker.get("last"))
            if price is None:
                price = _safe_float(ticker.get("close"))
            change = _safe_float(ticker.get("percentage"))
            if price is not None:
                return price, change
        except Exception:
            continue
    return None, None


def _fg_label_from_value(value: float | None) -> str:
    if value is None:
        return "Unavailable"
    v = int(max(0, min(100, round(value))))
    if v <= 24:
        return "Extreme Fear"
    if v <= 44:
        return "Fear"
    if v <= 55:
        return "Neutral"
    if v <= 74:
        return "Greed"
    return "Extreme Greed"


def _valid_top_metric_field(field: str, value) -> bool:
    if field in {
        "btc_price", "eth_price", "total_mcap", "alt_mcap",
        "btc_dom", "eth_dom", "bnb_dom", "sol_dom", "ada_dom", "xrp_dom",
    }:
        return _is_positive(value)
    if field in {"btc_change", "eth_change", "mcap_24h_pct"}:
        return _is_finite(value)
    if field == "fg_value":
        f = _safe_float(value)
        return f is not None and 0 <= f <= 100
    if field == "fg_label":
        return isinstance(value, str) and value.strip() != ""
    return value is not None


def get_market_top_snapshot() -> dict[str, float | int | str | None]:
    """Unified top-of-market snapshot with provider fallback and last-good cache.

    Rules:
    - BTC/ETH price and 24h change come from the same provider in a given fetch.
      Provider order: Exchange ticker -> CoinGecko simple price.
    - Market indices are fetched from CoinGecko global.
    - Fear & Greed is fetched from alternative.me.
    - Any missing field falls back to the last valid snapshot in session state.
    """
    key = "market_top_snapshot_v1"
    previous = st.session_state.get(key, {})
    live: dict[str, float | int | str | None] = {}

    # 1) BTC/ETH price+change: same-provider consistency
    ex_btc_price, ex_btc_change = _fetch_pair_price_change_from_exchange("BTC/USDT")
    ex_eth_price, ex_eth_change = _fetch_pair_price_change_from_exchange("ETH/USDT")
    exchange_complete = all(
        _valid_top_metric_field(name, value)
        for name, value in {
            "btc_price": ex_btc_price,
            "btc_change": ex_btc_change,
            "eth_price": ex_eth_price,
            "eth_change": ex_eth_change,
        }.items()
    )
    if exchange_complete:
        live.update(
            {
                "btc_price": ex_btc_price,
                "btc_change": ex_btc_change,
                "eth_price": ex_eth_price,
                "eth_change": ex_eth_change,
            }
        )
    else:
        cg = _fetch_btc_eth_from_coingecko_with_change()
        cg_complete = all(
            _valid_top_metric_field(name, value)
            for name, value in cg.items()
        )
        if cg_complete:
            live.update(cg)

    # 2) Market indices
    try:
        (
            btc_dom,
            eth_dom,
            total_mcap,
            alt_mcap,
            mcap_24h_pct,
            bnb_dom,
            sol_dom,
            ada_dom,
            xrp_dom,
        ) = get_market_indices_core()
        live.update(
            {
                "btc_dom": _safe_float(btc_dom),
                "eth_dom": _safe_float(eth_dom),
                "total_mcap": _safe_float(total_mcap),
                "alt_mcap": _safe_float(alt_mcap),
                "mcap_24h_pct": _safe_float(mcap_24h_pct),
                "bnb_dom": _safe_float(bnb_dom),
                "sol_dom": _safe_float(sol_dom),
                "ada_dom": _safe_float(ada_dom),
                "xrp_dom": _safe_float(xrp_dom),
            }
        )
    except Exception as e:
        _debug(f"get_market_top_snapshot indices fallback: {e}")

    # 3) Fear & Greed
    try:
        fg_value, fg_label = get_fear_greed_core()
        fg_value_f = _safe_float(fg_value)
        live["fg_value"] = fg_value_f
        live["fg_label"] = fg_label if _valid_top_metric_field("fg_label", fg_label) else _fg_label_from_value(fg_value_f)
    except Exception as e:
        _debug(f"get_market_top_snapshot fear_greed fallback: {e}")

    # 4) Merge with last-good snapshot field-by-field
    fields = [
        "btc_price", "btc_change", "eth_price", "eth_change",
        "btc_dom", "eth_dom", "total_mcap", "alt_mcap", "mcap_24h_pct",
        "bnb_dom", "sol_dom", "ada_dom", "xrp_dom", "fg_value", "fg_label",
    ]
    merged: dict[str, float | int | str | None] = {}
    updated_snapshot = dict(previous) if isinstance(previous, dict) else {}
    for field in fields:
        live_value = live.get(field)
        if _valid_top_metric_field(field, live_value):
            merged[field] = live_value
            updated_snapshot[field] = live_value
        else:
            prev_value = previous.get(field) if isinstance(previous, dict) else None
            if _valid_top_metric_field(field, prev_value):
                merged[field] = prev_value
            else:
                merged[field] = None

    if not _valid_top_metric_field("fg_label", merged.get("fg_label")):
        merged["fg_label"] = _fg_label_from_value(_safe_float(merged.get("fg_value")))

    # Persist only when we have at least one valid live field.
    if any(_valid_top_metric_field(f, live.get(f)) for f in fields):
        st.session_state[key] = updated_snapshot

    return merged

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


def analyse(df: pd.DataFrame) -> AnalysisResult:
    return analyse_core(df, debug_fn=_debug)


def get_scalping_entry_target(
    df: pd.DataFrame,
    bias_score: float,
    supertrend_trend: str,
    ichimoku_trend: str,
    vwap_label: str,
):
    return get_scalping_entry_target_core(
        df,
        bias_score,
        supertrend_trend,
        ichimoku_trend,
        vwap_label,
        sr_lookback_fn=_sr_lookback,
    )


def scalp_quality_gate(
    *,
    scalp_direction: str | None,
    signal_direction: str | None,
    rr_ratio: float | None,
    adx_val: float | None,
    strength: float | None,
    conviction_label: str | None,
    entry: float | None,
    stop: float | None,
    target: float | None,
    min_rr: float = 1.50,
    min_adx: float = 20.0,
    min_strength: float = 55.0,
) -> tuple[bool, str]:
    return scalp_quality_gate_core(
        scalp_direction=scalp_direction,
        signal_direction=signal_direction,
        rr_ratio=rr_ratio,
        adx_val=adx_val,
        strength=strength,
        conviction_label=conviction_label,
        entry=entry,
        stop=stop,
        target=target,
        min_rr=min_rr,
        min_adx=min_adx,
        min_strength=min_strength,
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
def _fetch_trending_coins_cached() -> list[dict]:
    return fetch_trending_coins_core(_http_get_json)


def fetch_trending_coins() -> list[dict]:
    rows = _fetch_trending_coins_cached()
    if rows:
        return rows
    # Avoid serving cached empty payloads for the full TTL when providers
    # are temporarily rate-limited.
    _debug("fetch_trending_coins cache returned empty; retrying uncached fetch.")
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
