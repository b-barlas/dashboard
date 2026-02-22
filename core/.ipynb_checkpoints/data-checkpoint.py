from __future__ import annotations

from typing import Any, Callable

import ccxt
import pandas as pd
import requests


TF_TO_CG_DAYS = {
    "1m": 2,
    "3m": 2,
    "5m": 2,
    "15m": 7,
    "1h": 90,
    "4h": 180,
    "1d": 365,
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


def get_btc_eth_prices() -> tuple[float | None, float | None]:
    url = "https://api.coingecko.com/api/v3/simple/price"
    params = {"ids": "bitcoin,ethereum", "vs_currencies": "usd"}
    response = requests.get(url, params=params, timeout=10).json()
    btc = response.get("bitcoin", {}).get("usd")
    eth = response.get("ethereum", {}).get("usd")
    return (btc if btc else None), (eth if eth else None)


def get_market_indices() -> tuple[int, int, int, int, float, int, int, int, int]:
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


def get_fear_greed() -> tuple[int, str]:
    data = requests.get("https://api.alternative.me/fng/?limit=1", timeout=10).json()
    value = int(data.get("data", [{}])[0].get("value", 0))
    label = data.get("data", [{}])[0].get("value_classification", "Unknown")
    return value, label


def symbol_variants(symbol: str) -> list[str]:
    variants = [symbol]
    if "/" in symbol:
        base, quote = symbol.split("/", 1)
        if quote == "USDT":
            variants.append(f"{base}/USD")
        elif quote == "USD":
            variants.append(f"{base}/USDT")
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
    resp = requests.get(
        "https://api.coingecko.com/api/v3/search",
        params={"query": symbol},
        timeout=10,
    )
    if resp.status_code == 200:
        for coin in resp.json().get("coins", []):
            if coin.get("symbol", "").upper() == symbol.upper():
                return coin["id"]
    return None


def coingecko_market_chart(coin_id: str, days: int) -> pd.DataFrame | None:
    ohlc_resp = requests.get(
        f"https://api.coingecko.com/api/v3/coins/{coin_id}/ohlc",
        params={"vs_currency": "usd", "days": days},
        timeout=15,
    )
    if ohlc_resp.status_code == 200:
        ohlc_data = ohlc_resp.json()
        if ohlc_data and isinstance(ohlc_data, list) and len(ohlc_data) > 5:
            df = pd.DataFrame(ohlc_data, columns=["timestamp", "open", "high", "low", "close"])
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
                        df["ts_ms"] = df["timestamp"]
                        df_v["ts_ms"] = df_v["ts_v"]
                        df = pd.merge_asof(
                            df.sort_values("ts_ms"),
                            df_v[["ts_ms", "volume"]].sort_values("ts_ms"),
                            on="ts_ms",
                            direction="nearest",
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
            return df[["timestamp", "open", "high", "low", "close", "volume"]]

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
    return df[["timestamp", "open", "high", "low", "close", "volume"]]


def fetch_coingecko_ohlcv(symbol: str, timeframe: str, limit: int) -> pd.DataFrame | None:
    base = symbol.split("/")[0].strip()
    coin_id = coingecko_coin_id(base)
    if not coin_id:
        return None
    days = TF_TO_CG_DAYS.get(timeframe, 30)
    df = coingecko_market_chart(coin_id, days)
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
    for variant in symbol_variants(symbol):
        try:
            return cached_fetcher(variant, timeframe, limit)
        except Exception:
            continue
    try:
        cg_df = fetch_coingecko_ohlcv(symbol, timeframe, limit)
        if cg_df is not None and not cg_df.empty:
            return cg_df
    except Exception:
        pass
    return None


def get_major_ohlcv_bundle(
    fetcher: Callable[[str, str, int], pd.DataFrame | None], timeframe: str, limit: int = 500
) -> dict[str, pd.DataFrame | None]:
    majors = ["BTC/USDT", "ETH/USDT", "BNB/USDT", "SOL/USDT", "ADA/USDT", "XRP/USDT"]
    out: dict[str, pd.DataFrame | None] = {}
    for sym in majors:
        out[sym] = fetcher(sym, timeframe, limit)
    return out

