"""Market-wide and CoinGecko-derived data helpers."""

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
    try:
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

        valid = []
        seen = set()
        for coin in data:
            symbol = (coin.get("symbol") or "").upper()
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

        # Safety fallback: if CoinGecko->exchange matching is sparse (e.g. alias
        # differences), fill from available exchange USD/USDT spot pairs so the
        # scanner can still analyse multiple assets.
        if len(valid) < max(5, min(20, top_n)):
            for pair in sorted(markets.keys()):
                if not isinstance(pair, str):
                    continue
                if "/" not in pair or ":" in pair:
                    continue
                base, quote = pair.split("/", 1)
                if quote not in {"USDT", "USD"}:
                    continue
                base_u = base.upper()
                if base_u in {"USD", "USDT", "EUR", "GBP"}:
                    continue
                if pair not in valid:
                    valid.append(pair)
                if len(valid) >= top_n:
                    break

        return valid, data
    except Exception as exc:
        debug_fn(f"get_top_volume_usdt_symbols error: {exc}")
        return [], []
