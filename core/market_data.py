"""Market-wide enrichment data helpers (multi-provider)."""

from __future__ import annotations

from typing import Callable


_BASE_ALIASES = {
    "BTC": ("BTC", "XBT"),
}


def _base_candidates(symbol: str) -> tuple[str, ...]:
    s = (symbol or "").upper()
    if not s:
        return tuple()
    return _BASE_ALIASES.get(s, (s,))


def fetch_trending_coins(http_get_json: Callable[..., object]) -> list[dict]:
    data = http_get_json("https://api.coingecko.com/api/v3/search/trending", timeout=10, retries=3)
    if not isinstance(data, dict):
        return []

    coins = []
    for item in data.get("coins", [])[:15]:
        c = item.get("item", {})
        coins.append(
            {
                "name": c.get("name", ""),
                "symbol": c.get("symbol", "").upper(),
                "market_cap_rank": c.get("market_cap_rank", 0),
                "price_btc": c.get("price_btc", 0),
                "score": c.get("score", 0),
            }
        )
    return coins


def fetch_top_gainers_losers(http_get_json: Callable[..., object], limit: int = 20) -> tuple[list, list]:
    url = "https://api.coingecko.com/api/v3/coins/markets"
    params = {
        "vs_currency": "usd",
        "order": "market_cap_desc",
        "per_page": 250,
        "page": 1,
        "sparkline": False,
        "price_change_percentage": "24h",
    }
    data = http_get_json(url, params=params, timeout=15, retries=3)
    if not isinstance(data, list):
        return [], []

    valid = []
    for coin in data:
        pct = coin.get("price_change_percentage_24h")
        if pct is None:
            pct = coin.get("price_change_percentage_24h_in_currency")
        if pct is None:
            change_abs = coin.get("price_change_24h")
            current_price = coin.get("current_price")
            # Fallback when API omits percentage field.
            if change_abs is not None and current_price not in (None, 0):
                try:
                    prev_price = float(current_price) - float(change_abs)
                    if prev_price > 0:
                        pct = (float(change_abs) / prev_price) * 100.0
                        coin["price_change_percentage_24h"] = pct
                except Exception:
                    pct = None
        if pct is not None:
            # Canonical key used by tabs.
            coin["price_change_percentage_24h"] = float(pct)
            valid.append(coin)

    if not valid:
        return [], []
    sorted_coins = sorted(valid, key=lambda x: x.get("price_change_percentage_24h", 0), reverse=True)
    gainers = sorted_coins[:limit]
    losers = sorted_coins[-limit:][::-1]
    return gainers, losers


def get_top_volume_usdt_symbols(
    http_get_json: Callable[..., object],
    markets: dict,
    debug_fn: Callable[[str], None],
    top_n: int = 100,
    vs_currency: str = "usd",
) -> tuple[list[str], list]:
    def _pairs_from_symbols(symbols: list[str]) -> list[str]:
        valid: list[str] = []
        seen: set[str] = set()
        for sym in symbols:
            symbol = (sym or "").upper()
            if not symbol or symbol in seen:
                continue
            seen.add(symbol)
            matched = False
            for base in _base_candidates(symbol):
                for quote in ("USDT", "USD"):
                    pair = f"{base}/{quote}"
                    if pair in markets:
                        valid.append(pair)
                        matched = True
                        break
                if matched:
                    break
        return valid

    def _coingecko_symbols() -> tuple[list[str], list]:
        url = "https://api.coingecko.com/api/v3/coins/markets"
        params = {
            "vs_currency": vs_currency,
            "order": "volume_desc",
            "per_page": min(top_n, 250),
            "page": 1,
            "sparkline": False,
        }
        data = http_get_json(url, params=params, timeout=10, retries=3)
        if not isinstance(data, list):
            debug_fn(f"CoinGecko invalid data type: {type(data)}")
            return [], []
        symbols = [(coin.get("symbol") or "").upper() for coin in data]
        return symbols, data

    def _coinpaprika_symbols() -> tuple[list[str], list]:
        # Public fallback provider when CoinGecko is rate-limited.
        url = "https://api.coinpaprika.com/v1/tickers"
        data = http_get_json(url, timeout=12, retries=2)
        if not isinstance(data, list):
            debug_fn(f"CoinPaprika invalid data type: {type(data)}")
            return [], []
        rows = []
        for coin in data:
            symbol = (coin.get("symbol") or "").upper()
            if not symbol:
                continue
            quotes = coin.get("quotes") if isinstance(coin, dict) else {}
            usd_q = quotes.get("USD") if isinstance(quotes, dict) else {}
            vol24 = usd_q.get("volume_24h") if isinstance(usd_q, dict) else None
            mcap = usd_q.get("market_cap") if isinstance(usd_q, dict) else None
            try:
                vol24_f = float(vol24) if vol24 is not None else 0.0
            except Exception:
                vol24_f = 0.0
            rows.append(
                {
                    "id": coin.get("id"),
                    "symbol": symbol.lower(),
                    "market_cap": mcap if isinstance(mcap, (int, float)) else 0,
                    "_volume_24h": vol24_f,
                }
            )
        rows = sorted(rows, key=lambda x: float(x.get("_volume_24h", 0.0)), reverse=True)
        rows = rows[: min(max(top_n * 2, top_n), 300)]
        symbols = [(r.get("symbol") or "").upper() for r in rows]
        return symbols, rows

    def _exchange_fallback_pairs() -> list[str]:
        # Fallback to exchange-available USD/USDT markets when CoinGecko symbols
        # cannot be mapped (rate-limit / payload drift / symbol mismatch).
        pairs: list[str] = []
        for pair in sorted(markets.keys()):
            if not isinstance(pair, str) or "/" not in pair:
                continue
            base, quote = pair.split("/", 1)
            if not base or quote not in {"USDT", "USD"}:
                continue
            pairs.append(pair)
            if len(pairs) >= top_n:
                break
        return pairs

    try:
        cg_symbols, cg_data = _coingecko_symbols()
        valid = _pairs_from_symbols(cg_symbols)
        if valid:
            return valid, cg_data

        cp_symbols, cp_data = _coinpaprika_symbols()
        valid_cp = _pairs_from_symbols(cp_symbols)
        if valid_cp:
            debug_fn("CoinGecko empty/unmapped; using CoinPaprika fallback.")
            return valid_cp, cp_data

        fallback = _exchange_fallback_pairs()
        if fallback:
            debug_fn("Provider mapping empty; using exchange market fallback.")
            # Prefer whichever provider data is available for enrichment.
            data_out = cg_data if cg_data else cp_data
            return fallback, data_out
        return [], (cg_data if cg_data else cp_data)
    except Exception as exc:
        debug_fn(f"get_top_volume_usdt_symbols error: {exc}")
        fallback = _exchange_fallback_pairs()
        if fallback:
            debug_fn("Using exchange market fallback after error.")
            return fallback, []
        return [], []
