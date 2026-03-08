from __future__ import annotations

import math
from typing import Any, Callable

import ccxt
import pandas as pd
import requests

from core.symbols import base_symbol_candidates, canonical_base_symbol


TF_TO_CG_DAYS = {
    "1m": 1,
    "3m": 1,
    "5m": 1,
    "15m": 7,
    "1h": 90,
    "4h": 90,
    "1d": 365,
}

TF_TO_RESAMPLE_RULE = {
    "1m": "1min",
    "3m": "3min",
    "5m": "5min",
    "15m": "15min",
    "1h": "1h",
    "4h": "4h",
    "1d": "1d",
}

TF_TO_SECONDS = {
    "1m": 60,
    "3m": 180,
    "5m": 300,
    "15m": 900,
    "1h": 3600,
    "4h": 14400,
    "1d": 86400,
}

COINGECKO_ID_OVERRIDES: dict[str, str] = {
    "BTC": "bitcoin",
    "ETH": "ethereum",
    "BNB": "binancecoin",
    "SOL": "solana",
    "ADA": "cardano",
    "XRP": "ripple",
    "DOGE": "dogecoin",
    "AVAX": "avalanche-2",
    "DOT": "polkadot",
    "LINK": "chainlink",
    "TON": "the-open-network",
    "TAO": "bittensor",
}


def get_exchange(exchange_configs: list[tuple[str, dict[str, Any]]]) -> Any:
    for name, extra in exchange_configs:
        try:
            ex = getattr(ccxt, name)({"enableRateLimit": True, **extra})
            ex.load_markets()
            return ex
        except Exception:
            continue
    return ccxt.kraken({"enableRateLimit": True})


def get_markets(exchange: Any) -> dict:
    return exchange.load_markets()


def _positive_rank(value: object) -> int | None:
    try:
        rank = int(value)
    except Exception:
        return None
    return rank if rank > 0 else None


def _market_chart_frame(points: object, value_col: str) -> pd.DataFrame:
    if not isinstance(points, list) or not points:
        return pd.DataFrame(columns=["timestamp", value_col])

    df = pd.DataFrame(points, columns=["timestamp", value_col])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True, errors="coerce")
    df[value_col] = pd.to_numeric(df[value_col], errors="coerce")
    df = df.dropna(subset=["timestamp", value_col]).sort_values("timestamp")
    return df.drop_duplicates(subset=["timestamp"], keep="last")


def _resample_market_chart_ohlcv(prices: object, volumes: object, timeframe: str) -> pd.DataFrame | None:
    tf_key = str(timeframe or "").strip().lower()
    rule = TF_TO_RESAMPLE_RULE.get(tf_key)
    target_seconds = TF_TO_SECONDS.get(tf_key)
    if rule is None or target_seconds is None:
        return None

    price_df = _market_chart_frame(prices, "price")
    if len(price_df) < 3:
        return None

    price_series = price_df.set_index("timestamp")["price"].astype(float)
    price_deltas = price_series.index.to_series().diff().dt.total_seconds().dropna()
    price_deltas = price_deltas[price_deltas > 0]
    if price_deltas.empty:
        return None
    source_seconds = float(price_deltas.median())
    if source_seconds > float(target_seconds):
        return None

    sample_count = max(1, int(round(float(target_seconds) / source_seconds)))
    sample_counts = price_series.resample(rule).count()
    ohlc = price_series.resample(rule).ohlc()
    ohlc = ohlc[sample_counts >= sample_count].dropna()
    if ohlc.empty:
        return None

    volume_df = _market_chart_frame(volumes, "volume")
    if volume_df.empty:
        volume_series = pd.Series(0.0, index=ohlc.index, dtype=float)
    else:
        volume_series = (
            volume_df.set_index("timestamp")["volume"]
            .astype(float)
            .resample(rule)
            .last()
            .reindex(ohlc.index)
            .ffill()
            .fillna(0.0)
        )

    out = ohlc.reset_index()
    out["volume"] = volume_series.to_numpy(dtype=float)
    out["timestamp"] = out["timestamp"].dt.tz_convert(None)
    out = out[["timestamp", "open", "high", "low", "close", "volume"]]
    out.attrs["volume_is_24h_aggregate"] = True
    return out


def _attach_market_chart_volume(df: pd.DataFrame, volumes: object) -> pd.DataFrame:
    out = df.copy()
    volume_df = _market_chart_frame(volumes, "volume")
    if volume_df.empty:
        out["volume"] = 0.0
        out.attrs["volume_is_24h_aggregate"] = True
        return out

    merged = pd.merge_asof(
        out.sort_values("timestamp"),
        volume_df.sort_values("timestamp"),
        on="timestamp",
        direction="nearest",
    )
    merged["volume"] = pd.to_numeric(merged["volume"], errors="coerce").fillna(0.0)
    merged.attrs["volume_is_24h_aggregate"] = True
    return merged


def _normalize_ohlc_timeframe(df: pd.DataFrame, timeframe: str) -> pd.DataFrame | None:
    tf_key = str(timeframe or "").strip().lower()
    rule = TF_TO_RESAMPLE_RULE.get(tf_key)
    target_seconds = TF_TO_SECONDS.get(tf_key)
    if rule is None or target_seconds is None:
        return None
    if df is None or df.empty or "timestamp" not in df.columns:
        return None

    out = df.copy()
    out["timestamp"] = pd.to_datetime(out["timestamp"], utc=True, errors="coerce")
    out = out.dropna(subset=["timestamp"]).sort_values("timestamp")
    if len(out) < 3:
        return None

    deltas = out["timestamp"].diff().dt.total_seconds().dropna()
    deltas = deltas[deltas > 0]
    if deltas.empty:
        return None
    source_seconds = float(deltas.median())
    if not math.isfinite(source_seconds) or source_seconds <= 0:
        return None
    if source_seconds > float(target_seconds):
        return None

    sample_count = max(1, int(round(float(target_seconds) / source_seconds)))
    frame = out.set_index("timestamp")
    agg_spec: dict[str, str] = {
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
    }
    if "volume" in frame.columns:
        agg_spec["volume"] = "last"
    counts = frame["close"].resample(rule).count()
    resampled = frame.resample(rule).agg(agg_spec)
    resampled = resampled[counts >= sample_count].dropna(subset=["open", "high", "low", "close"])
    if resampled.empty:
        return None
    if "volume" not in resampled.columns:
        resampled["volume"] = 0.0
    resampled["volume"] = pd.to_numeric(resampled["volume"], errors="coerce").fillna(0.0)
    resampled = resampled.reset_index()
    resampled["timestamp"] = resampled["timestamp"].dt.tz_convert(None)
    resampled.attrs["volume_is_24h_aggregate"] = True
    return resampled[["timestamp", "open", "high", "low", "close", "volume"]]


def _coingecko_days_for_request(timeframe: str, limit: int) -> int:
    tf_key = str(timeframe or "").strip().lower()
    max_days = int(TF_TO_CG_DAYS.get(tf_key, 30))
    tf_seconds = TF_TO_SECONDS.get(tf_key)
    if tf_seconds is None:
        return max_days
    requested_rows = max(int(limit), 60)
    needed_days = math.ceil((requested_rows * int(tf_seconds)) / 86400.0) + 1
    return max(1, min(max_days, int(needed_days)))


def get_btc_eth_prices() -> tuple[float | None, float | None]:
    url = "https://api.coingecko.com/api/v3/simple/price"
    params = {"ids": "bitcoin,ethereum", "vs_currencies": "usd"}
    response = requests.get(url, params=params, timeout=10).json()
    btc = response.get("bitcoin", {}).get("usd")
    eth = response.get("ethereum", {}).get("usd")
    return (btc if btc else None), (eth if eth else None)


def get_market_indices() -> tuple[float, float, int, int, float, float, float, float, float]:
    data = requests.get("https://api.coingecko.com/api/v3/global", timeout=10).json().get("data", {})
    mcap_pct = data.get("market_cap_percentage", {}) or {}
    btc_dom = float(mcap_pct.get("btc", 0.0))
    eth_dom = float(mcap_pct.get("eth", 0.0))
    bnb_dom = float(mcap_pct.get("bnb", 0.0))
    sol_dom = float(mcap_pct.get("sol", 0.0))
    ada_dom = float(mcap_pct.get("ada", 0.0))
    xrp_dom = float(mcap_pct.get("xrp", 0.0))
    total_mcap = float(data.get("total_market_cap", {}).get("usd", 0.0))
    alt_mcap = total_mcap * (1 - btc_dom / 100.0)
    mcap_24h_pct = float(data.get("market_cap_change_percentage_24h_usd", 0.0))
    return (
        round(btc_dom, 1),
        round(eth_dom, 1),
        int(total_mcap),
        int(alt_mcap),
        mcap_24h_pct,
        round(bnb_dom, 1),
        round(sol_dom, 1),
        round(ada_dom, 1),
        round(xrp_dom, 1),
    )


def get_fear_greed() -> tuple[int, str]:
    data = requests.get("https://api.alternative.me/fng/?limit=1", timeout=10).json()
    value = int(data.get("data", [{}])[0].get("value", 0))
    label = data.get("data", [{}])[0].get("value_classification", "Unknown")
    return value, label


def symbol_variants(symbol: str) -> list[str]:
    raw_symbol = str(symbol or "").strip().upper()
    if not raw_symbol:
        return []
    variants: list[str] = [raw_symbol]
    if "/" in symbol:
        base, quote = raw_symbol.split("/", 1)
        base_candidates = base_symbol_candidates(base)
        quote_candidates = [quote]
        if quote == "USDT":
            quote_candidates.append("USD")
        elif quote == "USD":
            quote_candidates.append("USDT")
        for base_candidate in base_candidates:
            for quote_candidate in quote_candidates:
                pair = f"{base_candidate}/{quote_candidate}"
                if pair not in variants:
                    variants.append(pair)
    return variants


def normalize_coin_input(raw: str) -> str:
    raw = raw.strip().upper()
    if not raw:
        return raw
    raw = raw.replace("$", "").replace(",", "").strip()
    if "/" not in raw:
        return f"{raw}/USDT"
    return raw


def validate_coin_symbol(symbol: str) -> str | None:
    if not symbol:
        return "Please enter a coin symbol."
    base = symbol.split("/")[0] if "/" in symbol else symbol
    if len(base) < 2 or len(base) > 10:
        return f"'{base}' doesn't look like a valid coin ticker (2-10 characters expected)."
    if not base.isalpha():
        return f"'{base}' contains invalid characters. Use letters only (e.g. BTC, ETH)."
    return None


def get_price_change(exchange: Any, symbol: str) -> float | None:
    for variant in symbol_variants(symbol):
        try:
            ticker = exchange.fetch_ticker(variant)
            percent = ticker.get("percentage")
            return round(percent, 2) if percent is not None else None
        except Exception:
            continue
    return None


def fetch_ohlcv_cached(exchange: Any, symbol: str, timeframe: str, limit: int = 120) -> pd.DataFrame:
    data = exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
    df = pd.DataFrame(data, columns=["timestamp", "open", "high", "low", "close", "volume"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
    return df


def coingecko_coin_id(symbol: str) -> str | None:
    symbol_key = canonical_base_symbol(symbol)
    if not symbol_key:
        return None
    if symbol_key in COINGECKO_ID_OVERRIDES:
        return COINGECKO_ID_OVERRIDES[symbol_key]

    resp = requests.get(
        "https://api.coingecko.com/api/v3/search",
        params={"query": symbol_key},
        timeout=10,
    )
    if resp.status_code == 200:
        exact_matches = [
            coin
            for coin in resp.json().get("coins", [])
            if coin.get("id") and coin.get("symbol", "").upper() == symbol_key
        ]
        if len(exact_matches) == 1:
            return str(exact_matches[0]["id"])

        ranked_matches = [coin for coin in exact_matches if _positive_rank(coin.get("market_cap_rank")) is not None]
        if len(ranked_matches) == 1:
            return str(ranked_matches[0]["id"])
    return None


def coingecko_market_chart(coin_id: str, days: int, timeframe: str) -> pd.DataFrame | None:
    market_chart_payload: dict | None = None
    ohlc_resp = requests.get(
        f"https://api.coingecko.com/api/v3/coins/{coin_id}/ohlc",
        params={"vs_currency": "usd", "days": days},
        timeout=15,
    )
    if ohlc_resp.status_code == 200:
        ohlc_data = ohlc_resp.json()
        if ohlc_data and isinstance(ohlc_data, list) and len(ohlc_data) > 5:
            df = pd.DataFrame(ohlc_data, columns=["timestamp", "open", "high", "low", "close"])
            df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
            try:
                vol_resp = requests.get(
                    f"https://api.coingecko.com/api/v3/coins/{coin_id}/market_chart",
                    params={"vs_currency": "usd", "days": days},
                    timeout=15,
                )
                if vol_resp.status_code == 200:
                    payload = vol_resp.json()
                    market_chart_payload = payload if isinstance(payload, dict) else {}
                    df = _attach_market_chart_volume(df, market_chart_payload.get("total_volumes", []))
                else:
                    df["volume"] = 0.0
                    df.attrs["volume_is_24h_aggregate"] = True
            except Exception:
                df["volume"] = 0.0
                df.attrs["volume_is_24h_aggregate"] = True
            normalized = _normalize_ohlc_timeframe(df, timeframe)
            if normalized is not None:
                return normalized

    if market_chart_payload is None:
        resp = requests.get(
            f"https://api.coingecko.com/api/v3/coins/{coin_id}/market_chart",
            params={"vs_currency": "usd", "days": days},
            timeout=15,
        )
        if resp.status_code != 200:
            return None
        payload = resp.json()
        if not isinstance(payload, dict):
            return None
        market_chart_payload = payload
    return _resample_market_chart_ohlcv(
        market_chart_payload.get("prices", []),
        market_chart_payload.get("total_volumes", []),
        timeframe,
    )


def fetch_coingecko_ohlcv(symbol: str, timeframe: str, limit: int) -> pd.DataFrame | None:
    base = symbol.split("/")[0].strip()
    coin_id = coingecko_coin_id(base)
    if not coin_id:
        return None
    days = _coingecko_days_for_request(timeframe, limit)
    df = coingecko_market_chart(coin_id, days, timeframe)
    if df is not None and len(df) > limit:
        df = df.tail(limit).reset_index(drop=True)
    return df


def fetch_ohlcv(
    exchange: Any,
    symbol: str,
    timeframe: str,
    limit: int,
    cached_fetcher: Callable[[str, str, int], pd.DataFrame],
) -> pd.DataFrame | None:
    best_exchange_df: pd.DataFrame | None = None
    best_exchange_variant: str | None = None
    min_preferred_rows = max(1, min(int(limit), 60))
    for variant in symbol_variants(symbol):
        try:
            df = cached_fetcher(variant, timeframe, limit)
            if df is None or df.empty:
                continue
            if best_exchange_df is None or len(df) > len(best_exchange_df):
                best_exchange_df = df
                best_exchange_variant = variant
            if len(df) >= int(limit):
                break
        except Exception:
            continue
    if best_exchange_df is not None:
        if len(best_exchange_df) >= min_preferred_rows:
            best_exchange_df.attrs["source_symbol"] = str(best_exchange_variant or symbol)
            best_exchange_df.attrs["source_provider"] = "exchange"
            return best_exchange_df
    try:
        cg_df = fetch_coingecko_ohlcv(symbol, timeframe, limit)
        if cg_df is not None and not cg_df.empty:
            cg_df.attrs["source_symbol"] = symbol
            cg_df.attrs["source_provider"] = "coingecko"
            return cg_df
    except Exception:
        pass
    if best_exchange_df is not None:
        best_exchange_df.attrs["source_symbol"] = str(best_exchange_variant or symbol)
        best_exchange_df.attrs["source_provider"] = "exchange"
        return best_exchange_df
    return None


def get_major_ohlcv_bundle(
    fetcher: Callable[[str, str, int], pd.DataFrame | None], timeframe: str, limit: int = 500
) -> dict[str, pd.DataFrame | None]:
    majors = ["BTC/USDT", "ETH/USDT", "BNB/USDT", "SOL/USDT", "ADA/USDT", "XRP/USDT"]
    out: dict[str, pd.DataFrame | None] = {}
    for sym in majors:
        out[sym] = fetcher(sym, timeframe, limit)
    return out
