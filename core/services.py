"""Application service layer for data access and analysis wrappers."""

from __future__ import annotations

import threading
import time
from pathlib import Path

import pandas as pd
import requests
import streamlit as st

from core.data import (
    coingecko_coin_id as coingecko_coin_id_core,
    fetch_coingecko_ohlcv_by_coin_id as fetch_coingecko_ohlcv_by_coin_id_core,
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
from core.catalyst_engine import load_manual_catalyst_events
from core.market_data import (
    fetch_market_cap_rows_for_symbols as fetch_market_cap_rows_for_symbols_core,
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
from core.symbols import canonical_base_symbol, is_stable_base_symbol
from core.signals import (
    AnalysisResult,
    analyse as analyse_core,
    detect_candle_pattern as detect_candle_pattern_core,
    detect_volume_spike as detect_volume_spike_core,
    sr_lookback as sr_lookback_core,
)
from ui.theme import NEGATIVE, NEON_BLUE, NEON_PURPLE, POSITIVE, TEXT_MUTED, WARNING

type MarketIndicesPayload = tuple[float, float, int, int, float, float, float, float, float]

_TOP_SNAPSHOT_KEY = "market_top_snapshot_v1"
_TOP_SNAPSHOT_META_KEY = f"{_TOP_SNAPSHOT_KEY}__meta"
_TOP_SNAPSHOT_FIELDS = [
    "btc_price", "btc_change", "eth_price", "eth_change",
    "btc_dom", "eth_dom", "total_mcap", "alt_mcap", "mcap_24h_pct",
    "bnb_dom", "sol_dom", "ada_dom", "xrp_dom", "fg_value", "fg_label",
]
_TOP_SNAPSHOT_FIELD_MAX_AGE_SEC = {
    "btc_price": 15 * 60,
    "btc_change": 15 * 60,
    "eth_price": 15 * 60,
    "eth_change": 15 * 60,
    "btc_dom": 30 * 60,
    "eth_dom": 30 * 60,
    "total_mcap": 30 * 60,
    "alt_mcap": 30 * 60,
    "mcap_24h_pct": 30 * 60,
    "bnb_dom": 30 * 60,
    "sol_dom": 30 * 60,
    "ada_dom": 30 * 60,
    "xrp_dom": 30 * 60,
    "fg_value": 6 * 60 * 60,
    "fg_label": 6 * 60 * 60,
}


def _debug(msg: str) -> None:
    if not st.session_state.get("debug_mode", False):
        return
    if threading.current_thread() is threading.main_thread():
        st.sidebar.write(msg)
        return
    print(f"[debug] {msg}")


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


def _read_top_snapshot_state(session_state: dict) -> tuple[dict, dict]:
    previous = session_state.get(_TOP_SNAPSHOT_KEY, {})
    previous_meta = session_state.get(_TOP_SNAPSHOT_META_KEY, {})
    return (
        dict(previous) if isinstance(previous, dict) else {},
        dict(previous_meta) if isinstance(previous_meta, dict) else {},
    )


def _merge_top_snapshot_fields(
    live: dict[str, float | int | str | None],
    previous: dict,
    previous_meta: dict,
    *,
    now_epoch: float,
) -> tuple[dict[str, float | int | str | None], dict, dict]:
    merged: dict[str, float | int | str | None] = {}
    updated_snapshot = dict(previous)
    updated_meta = dict(previous_meta)
    for field in _TOP_SNAPSHOT_FIELDS:
        live_value = live.get(field)
        if _valid_top_metric_field(field, live_value):
            merged[field] = live_value
            updated_snapshot[field] = live_value
            updated_meta[field] = now_epoch
            continue

        prev_value = previous.get(field)
        if _snapshot_field_is_fresh(field, prev_value, previous_meta, now_epoch=now_epoch):
            merged[field] = prev_value
            updated_snapshot[field] = prev_value
            continue

        merged[field] = None
        updated_snapshot.pop(field, None)
        updated_meta.pop(field, None)

    if not _valid_top_metric_field("fg_label", merged.get("fg_label")):
        merged["fg_label"] = _fg_label_from_value(_safe_float(merged.get("fg_value")))

    return merged, updated_snapshot, updated_meta


def _should_persist_top_snapshot_state(
    live: dict[str, float | int | str | None],
    updated_snapshot: dict,
    previous: dict,
    updated_meta: dict,
    previous_meta: dict,
) -> bool:
    return (
        any(_valid_top_metric_field(field, live.get(field)) for field in _TOP_SNAPSHOT_FIELDS)
        or updated_snapshot != previous
        or updated_meta != previous_meta
    )


def _persist_top_snapshot_state(session_state: dict, snapshot: dict, meta: dict) -> None:
    session_state[_TOP_SNAPSHOT_KEY] = snapshot
    session_state[_TOP_SNAPSHOT_META_KEY] = meta


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
@st.cache_data(ttl=300, show_spinner=False)
def _fetch_fear_greed_cached() -> tuple[int | None, str]:
    return get_fear_greed_core()


def get_fear_greed():
    try:
        fg_value, fg_label = _fetch_fear_greed_cached()
        fg_value_f = _safe_float(fg_value)
        fg_label_s = fg_label if _valid_top_metric_field("fg_label", fg_label) else _fg_label_from_value(fg_value_f)
        if _valid_top_metric_field("fg_value", fg_value_f) or _valid_top_metric_field("fg_label", fg_label_s):
            return fg_value_f, fg_label_s
    except Exception as e:
        _debug(f"get_fear_greed cached fetch error: {e}")

    try:
        fg_value, fg_label = get_fear_greed_core()
        fg_value_f = _safe_float(fg_value)
        fg_label_s = fg_label if _valid_top_metric_field("fg_label", fg_label) else _fg_label_from_value(fg_value_f)
        return fg_value_f, fg_label_s
    except Exception as e:
        _debug(f"get_fear_greed error: {e}")
        return None, "Unavailable"


def _safe_secret_value(key: str) -> str | None:
    try:
        secrets = getattr(st, "secrets", None)
        if secrets is None:
            return None
        if hasattr(secrets, "get"):
            value = secrets.get(key)
        elif key in secrets:
            value = secrets[key]
        else:
            value = None
        return str(value).strip() if value else None
    except Exception:
        return None


def _manual_catalyst_path() -> Path:
    return Path(__file__).resolve().parents[1] / "data" / "market_catalysts.json"


@st.cache_data(ttl=300, show_spinner=False)
def _load_manual_market_catalysts_cached() -> list[dict]:
    return load_manual_catalyst_events(_manual_catalyst_path())


@st.cache_data(ttl=900, show_spinner=False)
def _fetch_fmp_economic_calendar_cached(api_key: str, start_date: str, end_date: str) -> list[dict]:
    payload = _http_get_json(
        "https://financialmodelingprep.com/stable/economic-calendar",
        params={
            "from": start_date,
            "to": end_date,
            "apikey": api_key,
        },
        timeout=12,
        retries=2,
    )
    if not isinstance(payload, list):
        return []
    rows: list[dict] = []
    for item in payload:
        if not isinstance(item, dict):
            continue
        title = str(item.get("event") or item.get("name") or "").strip()
        when_raw = item.get("date") or item.get("datetime")
        if not title or when_raw is None:
            continue
        impact = str(item.get("impact") or item.get("importance") or "medium").strip().lower()
        if impact not in {"high", "medium", "low"}:
            impact = "medium"
        rows.append(
            {
                "title": title,
                "event_time": when_raw,
                "severity": impact,
                "category": "macro",
                "scope": "market",
                "source": "FMP",
                "tag": str(item.get("country") or "").strip(),
            }
        )
    return rows


def get_market_catalyst_events(now: object | None = None) -> list[dict]:
    ts_now = pd.Timestamp.utcnow() if now is None else pd.to_datetime(now, utc=True, errors="coerce")
    if pd.isna(ts_now):
        ts_now = pd.Timestamp.utcnow()
    start_date = ts_now.strftime("%Y-%m-%d")
    end_date = (ts_now + pd.Timedelta(days=4)).strftime("%Y-%m-%d")

    events: list[dict] = []
    try:
        events.extend(_load_manual_market_catalysts_cached())
    except Exception as e:
        _debug(f"manual catalyst load failed: {e}")

    api_key = _safe_secret_value("FMP_API_KEY")
    if api_key:
        try:
            events.extend(_fetch_fmp_economic_calendar_cached(api_key, start_date, end_date))
        except Exception as e:
            _debug(f"FMP economic calendar fetch failed: {e}")

    return events


@st.cache_data(ttl=180, show_spinner=False)
def _fetch_binance_funding_history_cached(symbol: str) -> list[dict]:
    payload = _http_get_json(
        "https://fapi.binance.com/fapi/v1/fundingRate",
        params={"symbol": str(symbol).upper(), "limit": 2},
        timeout=10,
        retries=2,
    )
    return payload if isinstance(payload, list) else []


@st.cache_data(ttl=180, show_spinner=False)
def _fetch_binance_open_interest_hist_cached(symbol: str) -> list[dict]:
    payload = _http_get_json(
        "https://fapi.binance.com/futures/data/openInterestHist",
        params={"symbol": str(symbol).upper(), "period": "5m", "limit": 2},
        timeout=10,
        retries=2,
    )
    return payload if isinstance(payload, list) else []


@st.cache_data(ttl=180, show_spinner=False)
def _fetch_binance_long_short_ratio_cached(symbol: str) -> list[dict]:
    payload = _http_get_json(
        "https://fapi.binance.com/futures/data/globalLongShortAccountRatio",
        params={"symbol": str(symbol).upper(), "period": "5m", "limit": 2},
        timeout=10,
        retries=2,
    )
    return payload if isinstance(payload, list) else []


def get_market_flow_proxy_rows(symbols: list[str] | None = None) -> list[dict]:
    rows: list[dict] = []
    for symbol in list(symbols or ["BTCUSDT", "ETHUSDT"]):
        try:
            funding_rows = _fetch_binance_funding_history_cached(symbol)
            oi_rows = _fetch_binance_open_interest_hist_cached(symbol)
            ratio_rows = _fetch_binance_long_short_ratio_cached(symbol)
        except Exception as e:
            _debug(f"flow proxy fetch failed for {symbol}: {e}")
            continue

        if not funding_rows or len(oi_rows) < 2 or not ratio_rows:
            continue
        try:
            funding_rate = float(funding_rows[-1].get("fundingRate"))
            oi_now = float(oi_rows[-1].get("sumOpenInterest"))
            oi_prev = float(oi_rows[-2].get("sumOpenInterest"))
            long_short_ratio = float(ratio_rows[-1].get("longShortRatio"))
        except Exception:
            continue
        if oi_prev <= 0:
            continue
        oi_change_pct = ((oi_now / oi_prev) - 1.0) * 100.0
        rows.append(
            {
                "symbol": str(symbol).upper(),
                "funding_rate": funding_rate,
                "oi_change_pct": oi_change_pct,
                "long_short_ratio": long_short_ratio,
            }
        )
    return rows


def _fetch_coingecko_heatmap_rows(limit: int = 180) -> list[dict]:
    payload = _http_get_json(
        "https://api.coingecko.com/api/v3/coins/markets",
        params={
            "vs_currency": "usd",
            "order": "market_cap_desc",
            "per_page": min(max(limit, 100), 250),
            "page": 1,
            "sparkline": False,
            "price_change_percentage": "24h",
        },
        timeout=8,
        retries=2,
    )
    if not isinstance(payload, list):
        return []

    rows: list[dict] = []
    for coin in payload:
        symbol = str(coin.get("symbol") or "").upper().strip()
        if not symbol:
            continue
        mcap = _safe_float(coin.get("market_cap"))
        if mcap is None or mcap <= 0:
            continue
        pct = coin.get("price_change_percentage_24h")
        if pct is None:
            pct = coin.get("price_change_percentage_24h_in_currency")
        cid = str(coin.get("id") or "").strip()
        key = f"cg:{cid}" if cid else f"cg:{symbol}:{str(coin.get('name') or symbol).strip()}"
        rows.append(
            {
                "Symbol": symbol,
                "Name": str(coin.get("name") or symbol),
                "TreemapKey": key,
                "Market Cap": float(mcap),
                "Change 24h (%)": _safe_float(pct) or 0.0,
                "Price": _safe_float(coin.get("current_price")) or 0.0,
                "Sector": "Crypto",
                "Stablecoin": is_stable_base_symbol(symbol),
                "Provider": "CoinGecko",
            }
        )

    rows.sort(key=lambda row: row["Market Cap"], reverse=True)
    return rows[:limit]


def _fetch_coinpaprika_heatmap_rows(limit: int = 180) -> list[dict]:
    payload = _http_get_json(
        "https://api.coinpaprika.com/v1/tickers",
        timeout=8,
        retries=2,
    )
    if not isinstance(payload, list):
        return []

    rows: list[dict] = []
    for coin in payload:
        symbol = str(coin.get("symbol") or "").upper().strip()
        if not symbol:
            continue
        quotes = coin.get("quotes") if isinstance(coin, dict) else {}
        usd_q = quotes.get("USD") if isinstance(quotes, dict) else {}
        mcap = _safe_float(usd_q.get("market_cap"))
        if mcap is None or mcap <= 0:
            continue
        pid = str(coin.get("id") or "").strip()
        key = f"cp:{pid}" if pid else f"cp:{symbol}:{str(coin.get('name') or symbol).strip()}"
        rows.append(
            {
                "Symbol": symbol,
                "Name": str(coin.get("name") or symbol),
                "TreemapKey": key,
                "Market Cap": float(mcap),
                "Change 24h (%)": _safe_float(usd_q.get("percent_change_24h")) or 0.0,
                "Price": _safe_float(usd_q.get("price")) or 0.0,
                "Sector": "Crypto",
                "Stablecoin": is_stable_base_symbol(symbol),
                "Provider": "CoinPaprika",
            }
        )

    rows.sort(key=lambda row: row["Market Cap"], reverse=True)
    return rows[:limit]


def get_heatmap_rows(limit: int = 180, live_ttl_sec: int = 90) -> tuple[list[dict], str, str, str | None]:
    cache_key = "heatmap_last_good_v3"
    live_key = "heatmap_live_cache_v1"
    now_epoch = time.time()
    now_utc = pd.Timestamp.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")

    live_cached = st.session_state.get(live_key, {})
    if isinstance(live_cached, dict):
        live_rows = live_cached.get("rows")
        live_at = _safe_float(live_cached.get("fetched_at")) or 0.0
        if isinstance(live_rows, list) and live_rows and (now_epoch - live_at) <= live_ttl_sec:
            return (
                live_rows,
                str(live_cached.get("source") or "Live cache"),
                "LIVE",
                str(live_cached.get("ts") or now_utc),
            )

    rows = _fetch_coingecko_heatmap_rows(limit=limit)
    if rows:
        payload = {"rows": rows, "source": "CoinGecko", "ts": now_utc, "fetched_at": now_epoch}
        st.session_state[live_key] = payload
        st.session_state[cache_key] = payload
        return rows, "CoinGecko", "LIVE", now_utc

    rows = _fetch_coinpaprika_heatmap_rows(limit=limit)
    if rows:
        payload = {"rows": rows, "source": "CoinPaprika", "ts": now_utc, "fetched_at": now_epoch}
        st.session_state[live_key] = payload
        st.session_state[cache_key] = payload
        return rows, "CoinPaprika", "LIVE", now_utc

    cached = st.session_state.get(cache_key, {})
    cached_rows = cached.get("rows") if isinstance(cached, dict) else None
    if isinstance(cached_rows, list) and cached_rows:
        return (
            cached_rows,
            str(cached.get("source") or "Cached"),
            "CACHED",
            str(cached.get("ts") or ""),
        )

    return [], "Unavailable", "EMPTY", None


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


def _pair_price_change_from_ticker(ticker: object) -> tuple[float | None, float | None]:
    if not isinstance(ticker, dict):
        return None, None
    price = _safe_float(ticker.get("last"))
    if price is None:
        price = _safe_float(ticker.get("close"))
    change = _safe_float(ticker.get("percentage"))
    if price is None:
        return None, None
    return price, change


@st.cache_data(ttl=60, show_spinner=False)
def _fetch_pair_price_change_from_exchange_direct(symbol: str) -> tuple[float | None, float | None]:
    for variant in symbol_variants_core(symbol):
        try:
            price, change = _pair_price_change_from_ticker(EXCHANGE.fetch_ticker(variant))
            if price is not None:
                return price, change
        except Exception:
            continue
    return None, None


def _fetch_pair_price_change_from_exchange(symbol: str) -> tuple[float | None, float | None]:
    tickers = fetch_exchange_tickers_snapshot()
    if tickers:
        for variant in symbol_variants_core(symbol):
            price, change = _pair_price_change_from_ticker(tickers.get(variant))
            if price is not None:
                return price, change
    return _fetch_pair_price_change_from_exchange_direct(symbol)


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
        label = str(value or "").strip()
        return label not in {"", "Unavailable", "Unknown", "N/A"}
    return value is not None


def _indices_payload_ok(payload) -> bool:
    """Minimal validity gate for market indices tuple."""
    try:
        if not isinstance(payload, (list, tuple)) or len(payload) != 9:
            return False
        total_mcap = _safe_float(payload[2])
        return total_mcap is not None and total_mcap > 0
    except Exception:
        return False


def _snapshot_field_is_fresh(field: str, value, meta: dict[str, object] | None, *, now_epoch: float) -> bool:
    if not _valid_top_metric_field(field, value):
        return False
    if not isinstance(meta, dict):
        return False
    try:
        field_epoch = float(meta.get(field))
    except Exception:
        return False
    max_age = float(_TOP_SNAPSHOT_FIELD_MAX_AGE_SEC.get(field, 0))
    if max_age <= 0:
        return False
    age = max(0.0, float(now_epoch) - field_epoch)
    return age <= max_age


@st.cache_data(ttl=300, show_spinner=False)
def _fetch_market_indices_coinpaprika() -> MarketIndicesPayload | None:
    """Fallback global metrics from CoinPaprika tickers endpoint.

    Derives total market cap, selected asset dominance and weighted 24h cap change.
    """
    data = _http_get_json("https://api.coinpaprika.com/v1/tickers", timeout=12, retries=2)
    if not isinstance(data, list):
        return None

    symbol_caps = {"BTC": 0.0, "ETH": 0.0, "BNB": 0.0, "SOL": 0.0, "ADA": 0.0, "XRP": 0.0}
    total_mcap = 0.0
    weighted_change_num = 0.0
    weighted_change_den = 0.0

    for row in data:
        try:
            symbol = str((row or {}).get("symbol") or "").upper()
            quotes = (row or {}).get("quotes") if isinstance(row, dict) else {}
            usd = quotes.get("USD") if isinstance(quotes, dict) else {}
            mcap = _safe_float(usd.get("market_cap")) if isinstance(usd, dict) else None
            chg24 = _safe_float(usd.get("percent_change_24h")) if isinstance(usd, dict) else None
            if mcap is None or mcap <= 0:
                continue
            total_mcap += float(mcap)
            if chg24 is not None:
                weighted_change_num += float(mcap) * float(chg24)
                weighted_change_den += float(mcap)
            if symbol in symbol_caps:
                symbol_caps[symbol] += float(mcap)
        except Exception:
            continue

    if total_mcap <= 0:
        return None

    btc_cap = symbol_caps["BTC"]
    eth_cap = symbol_caps["ETH"]
    bnb_cap = symbol_caps["BNB"]
    sol_cap = symbol_caps["SOL"]
    ada_cap = symbol_caps["ADA"]
    xrp_cap = symbol_caps["XRP"]
    alt_mcap = max(total_mcap - btc_cap, 0.0)
    mcap_24h_pct = (weighted_change_num / weighted_change_den) if weighted_change_den > 0 else 0.0

    return (
        round((btc_cap / total_mcap) * 100.0, 1),
        round((eth_cap / total_mcap) * 100.0, 1),
        int(total_mcap),
        int(alt_mcap),
        float(mcap_24h_pct),
        round((bnb_cap / total_mcap) * 100.0, 1),
        round((sol_cap / total_mcap) * 100.0, 1),
        round((ada_cap / total_mcap) * 100.0, 1),
        round((xrp_cap / total_mcap) * 100.0, 1),
    )


def get_market_top_snapshot() -> dict[str, float | int | str | None]:
    """Unified top-of-market snapshot with provider fallback and last-good cache.

    Rules:
    - BTC/ETH price and 24h change come from the same provider in a given fetch.
      Provider order: Exchange ticker -> CoinGecko simple price.
    - Market indices are fetched from CoinGecko global.
    - Fear & Greed is fetched from alternative.me.
    - Any missing field falls back to the last valid snapshot only while that
      specific field remains inside its freshness window.
    """
    previous, previous_meta = _read_top_snapshot_state(st.session_state)
    live: dict[str, float | int | str | None] = {}
    now_epoch = time.time()

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

    # 2) Market indices (CoinGecko global -> CoinPaprika fallback)
    try:
        indices_payload: MarketIndicesPayload | None = get_market_indices()
        if not _indices_payload_ok(indices_payload):
            cp_payload = _fetch_market_indices_coinpaprika()
            if _indices_payload_ok(cp_payload):
                indices_payload = cp_payload
                _debug("Market indices fallback active: CoinPaprika.")
        if _indices_payload_ok(indices_payload):
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
            ) = indices_payload
        else:
            raise RuntimeError("No valid market indices payload from providers.")
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
        fg_value, fg_label = get_fear_greed()
        fg_value_f = _safe_float(fg_value)
        live["fg_value"] = fg_value_f
        live["fg_label"] = fg_label if _valid_top_metric_field("fg_label", fg_label) else _fg_label_from_value(fg_value_f)
    except Exception as e:
        _debug(f"get_market_top_snapshot fear_greed fallback: {e}")

    # 4) Merge with last-good snapshot field-by-field
    merged, updated_snapshot, updated_meta = _merge_top_snapshot_fields(
        live,
        previous,
        previous_meta,
        now_epoch=now_epoch,
    )
    should_persist = _should_persist_top_snapshot_state(
        live,
        updated_snapshot,
        previous,
        updated_meta,
        previous_meta,
    )
    if should_persist:
        _persist_top_snapshot_state(st.session_state, updated_snapshot, updated_meta)

    return merged

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


def fetch_coingecko_ohlcv_by_coin_id(coin_id: str, timeframe: str, limit: int = 120) -> pd.DataFrame | None:
    """Direct CoinGecko OHLCV fetch by coin id for provider-resolved custom watchlist symbols."""
    df = fetch_coingecko_ohlcv_by_coin_id_core(coin_id, timeframe, limit)
    if df is not None and not df.empty:
        df.attrs["source_symbol"] = str(coin_id or "").strip()
        df.attrs["source_provider"] = "coingecko"
    return df


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
    confidence: float | None = None,
    conviction_label: str | None,
    entry: float | None,
    stop: float | None,
    target: float | None,
    min_rr: float = 1.50,
    min_adx: float = 20.0,
    min_confidence: float = 55.0,
) -> tuple[bool, str]:
    return scalp_quality_gate_core(
        scalp_direction=scalp_direction,
        signal_direction=signal_direction,
        rr_ratio=rr_ratio,
        adx_val=adx_val,
        confidence=confidence,
        conviction_label=conviction_label,
        entry=entry,
        stop=stop,
        target=target,
        min_rr=min_rr,
        min_adx=min_adx,
        min_confidence=min_confidence,
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


def monte_carlo_simulation(
    df: pd.DataFrame,
    num_simulations: int = 500,
    num_days: int = 30,
    seed: int | None = None,
) -> dict:
    return monte_carlo_simulation_core(
        df,
        num_simulations=num_simulations,
        num_days=num_days,
        seed=seed,
    )


def detect_divergence(df: pd.DataFrame) -> list[dict]:
    return detect_divergence_core(df)


def calculate_risk_metrics(
    df: pd.DataFrame,
    risk_free_rate: float = 0.02,
    timeframe: str = "1d",
    close_series: pd.Series | None = None,
) -> dict:
    """Backward-compatible wrapper to the core risk engine."""
    return calculate_risk_metrics_core(
        df,
        risk_free_rate=risk_free_rate,
        timeframe=timeframe,
        close_series=close_series,
    )


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

_TOP_GAINERS_LOSERS_CACHE_VERSION = 2


@st.cache_data(ttl=120, show_spinner=False)
def _fetch_top_gainers_losers_cached(
    limit: int = 20,
    cache_version: int = _TOP_GAINERS_LOSERS_CACHE_VERSION,
) -> tuple[list, list]:
    _ = cache_version
    return fetch_top_gainers_losers_core(_http_get_json, limit=limit)


def fetch_top_gainers_losers(limit: int = 20) -> tuple[list, list]:
    gainers, losers = _fetch_top_gainers_losers_cached(
        limit=limit,
        cache_version=_TOP_GAINERS_LOSERS_CACHE_VERSION,
    )
    if gainers or losers:
        return gainers, losers
    # Avoid serving cached empty payloads for the full TTL when CoinGecko has
    # transient 429/timeout responses.
    _debug("fetch_top_gainers_losers cache returned empty; retrying uncached fetch.")
    gainers, losers = fetch_top_gainers_losers_core(_http_get_json, limit=limit)
    if gainers or losers:
        return gainers, losers
    _debug("broad gainers/losers fetch returned empty; retrying with narrower single-page fallback.")
    return fetch_top_gainers_losers_core(_http_get_json, limit=limit, max_pages=1, per_page=100)


def _fetch_exchange_tickers_snapshot_uncached() -> dict:
    try:
        has = getattr(EXCHANGE, "has", {}) or {}
        if not has.get("fetchTickers"):
            return {}
        data = EXCHANGE.fetch_tickers()
        return data if isinstance(data, dict) else {}
    except Exception as e:
        _debug(f"exchange ticker snapshot fallback unavailable: {e}")
        return {}


@st.cache_data(ttl=60, show_spinner=False)
def _fetch_exchange_tickers_snapshot_cached() -> dict:
    return _fetch_exchange_tickers_snapshot_uncached()


def fetch_exchange_tickers_snapshot() -> dict:
    data = _fetch_exchange_tickers_snapshot_cached()
    if data:
        return data
    # Avoid pinning an empty/error snapshot for the full TTL during provider wobble.
    return _fetch_exchange_tickers_snapshot_uncached()


def get_top_volume_usdt_symbols(top_n: int = 100, vs_currency: str = "usd"):
    return get_top_volume_usdt_symbols_core(
        _http_get_json,
        MARKETS,
        _debug,
        top_n=top_n,
        vs_currency=vs_currency,
        exchange_tickers_fetcher=fetch_exchange_tickers_snapshot,
    )


@st.cache_data(ttl=300, show_spinner=False)
def _fetch_market_cap_rows_for_symbols_cached(
    symbols: tuple[str, ...],
    vs_currency: str = "usd",
) -> list[dict]:
    return fetch_market_cap_rows_for_symbols_core(
        _http_get_json,
        coingecko_coin_id_core,
        list(symbols),
        _debug,
        vs_currency=vs_currency,
    )


def get_market_cap_rows_for_symbols(
    symbols: list[str] | tuple[str, ...],
    vs_currency: str = "usd",
) -> list[dict]:
    normalized: list[str] = []
    seen: set[str] = set()
    for raw_symbol in symbols:
        raw = str(raw_symbol or "").strip()
        base = raw.split("/", 1)[0] if "/" in raw else raw
        symbol = canonical_base_symbol(base)
        if not symbol or symbol in seen:
            continue
        seen.add(symbol)
        normalized.append(symbol)
    if not normalized:
        return []

    cache_key = tuple(sorted(normalized))
    rows = _fetch_market_cap_rows_for_symbols_cached(cache_key, vs_currency=vs_currency)
    if rows:
        return rows

    _debug("get_market_cap_rows_for_symbols cache returned empty; retrying uncached fetch.")
    return fetch_market_cap_rows_for_symbols_core(
        _http_get_json,
        coingecko_coin_id_core,
        normalized,
        _debug,
        vs_currency=vs_currency,
    )
