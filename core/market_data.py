"""Market-wide enrichment data helpers (multi-provider)."""

from __future__ import annotations

import math
from typing import Callable

from core.symbols import base_symbol_candidates, canonical_base_symbol


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
    exchange_tickers_fetcher: Callable[[], object] | None = None,
) -> tuple[list[str], list]:
    provider_fetch_n = min(250, max(int(top_n) * 3, int(top_n) + 25))
    exchange_tickers_snapshot: dict | None = None

    def _filter_ambiguous_provider_rows(rows: object, provider_name: str) -> list[dict]:
        if not isinstance(rows, list):
            return []

        ids_by_symbol: dict[str, set[str]] = {}
        for row in rows:
            if not isinstance(row, dict):
                continue
            symbol = canonical_base_symbol((row.get("symbol") or "").strip())
            if not symbol:
                continue
            coin_id = str(row.get("id") or "").strip().lower()
            if coin_id:
                ids_by_symbol.setdefault(symbol, set()).add(coin_id)

        ambiguous = {symbol for symbol, ids in ids_by_symbol.items() if len(ids) > 1}
        if ambiguous:
            sample = ", ".join(sorted(ambiguous)[:3])
            more = "" if len(ambiguous) <= 3 else f" +{len(ambiguous) - 3} more"
            debug_fn(
                f"{provider_name} duplicate ticker ambiguity; skipped canonical symbols: {sample}{more}."
            )

        filtered: list[dict] = []
        for row in rows:
            if not isinstance(row, dict):
                continue
            symbol = canonical_base_symbol((row.get("symbol") or "").strip())
            if not symbol or symbol in ambiguous:
                continue
            filtered.append(row)
        return filtered

    def _safe_float(value: object) -> float | None:
        try:
            out = float(value)
        except Exception:
            return None
        if not math.isfinite(out):
            return None
        return out

    def _ticker_quote_volume(ticker: object) -> float:
        if not isinstance(ticker, dict):
            return 0.0
        direct = _safe_float(
            ticker.get("quoteVolume")
            if "quoteVolume" in ticker
            else ticker.get("quote_volume")
        )
        if direct is not None and direct > 0:
            return float(direct)

        base_vol = _safe_float(
            ticker.get("baseVolume")
            if "baseVolume" in ticker
            else ticker.get("base_volume")
        )
        ref_price = (
            _safe_float(ticker.get("vwap"))
            or _safe_float(ticker.get("last"))
            or _safe_float(ticker.get("close"))
        )
        if base_vol is not None and base_vol > 0 and ref_price is not None and ref_price > 0:
            return float(base_vol) * float(ref_price)
        return 0.0

    def _exchange_tickers() -> dict:
        nonlocal exchange_tickers_snapshot
        if exchange_tickers_snapshot is not None:
            return exchange_tickers_snapshot
        tickers: dict = {}
        if exchange_tickers_fetcher is not None:
            try:
                fetched = exchange_tickers_fetcher()
                if isinstance(fetched, dict):
                    tickers = fetched
            except Exception as exc:
                debug_fn(f"Exchange ticker snapshot unavailable for pair ranking: {exc}")
        exchange_tickers_snapshot = tickers
        return exchange_tickers_snapshot

    def _best_pair_for_symbol(symbol: str) -> tuple[str | None, bool]:
        candidates: list[tuple[int, str, str]] = []
        seen_pairs: set[str] = set()
        order = 0
        for base in base_symbol_candidates(symbol):
            for quote in ("USDT", "USD"):
                pair = f"{base}/{quote}"
                if pair not in markets or pair in seen_pairs:
                    continue
                seen_pairs.add(pair)
                candidates.append((order, pair, quote))
                order += 1
        if not candidates:
            return None, False

        tickers = _exchange_tickers()
        ranked: list[tuple[float, int, str, str]] = []
        for idx, pair, quote in candidates:
            volume_rank = _ticker_quote_volume(tickers.get(pair))
            if volume_rank > 0:
                ranked.append((float(volume_rank), 1 if quote == "USDT" else 0, pair, str(idx)))
        if ranked:
            ranked.sort(key=lambda item: (-item[0], -item[1], item[3]))
            return ranked[0][2], True

        if len(candidates) == 1:
            return candidates[0][1], True

        debug_fn(
            f"Strict pair ranking skipped {symbol}: multiple exchange pairs exist but ticker liquidity is unavailable."
        )
        return None, True

    def _pairs_from_symbols(symbols: list[str]) -> tuple[list[str], bool]:
        valid: list[str] = []
        seen: set[str] = set()
        strict_unresolved = False
        for sym in symbols:
            symbol = (sym or "").upper()
            if not symbol or symbol in seen:
                continue
            seen.add(symbol)
            pair, had_candidates = _best_pair_for_symbol(symbol)
            if pair:
                valid.append(pair)
            elif had_candidates:
                strict_unresolved = True
        return valid, strict_unresolved

    def _coingecko_symbols() -> tuple[list[str], list]:
        url = "https://api.coingecko.com/api/v3/coins/markets"
        params = {
            "vs_currency": vs_currency,
            "order": "volume_desc",
            "per_page": provider_fetch_n,
            "page": 1,
            "sparkline": False,
        }
        data = http_get_json(url, params=params, timeout=10, retries=3)
        if not isinstance(data, list):
            debug_fn(f"CoinGecko invalid data type: {type(data)}")
            return [], []
        safe_rows = _filter_ambiguous_provider_rows(data, "CoinGecko")
        symbols = [(coin.get("symbol") or "").upper() for coin in safe_rows]
        return symbols, safe_rows

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
        rows = _filter_ambiguous_provider_rows(rows, "CoinPaprika")
        rows = rows[: min(max(provider_fetch_n, int(top_n)), 400)]
        symbols = [(r.get("symbol") or "").upper() for r in rows]
        return symbols, rows

    def _exchange_fallback_pairs() -> list[str]:
        # Fallback to exchange-available USD/USDT markets when CoinGecko symbols
        # cannot be mapped (rate-limit / payload drift / symbol mismatch).
        tickers = _exchange_tickers()

        per_base: dict[str, tuple[float, str]] = {}
        for pair in sorted(markets.keys()):
            if not isinstance(pair, str) or "/" not in pair:
                continue
            base, quote = pair.split("/", 1)
            if not base or quote not in {"USDT", "USD"}:
                continue
            canonical = canonical_base_symbol(base)
            volume_rank = _ticker_quote_volume(tickers.get(pair))
            if volume_rank <= 0:
                continue
            current = per_base.get(canonical)
            current_pair = current[1] if current else ""
            current_volume = float(current[0]) if current else -1.0
            should_replace = volume_rank > current_volume
            if not should_replace and current and volume_rank == current_volume:
                should_replace = quote == "USDT" and not str(current_pair).endswith("/USDT")
            if should_replace or current is None:
                per_base[canonical] = (volume_rank, pair)

        if not per_base:
            debug_fn("Exchange market fallback skipped: ranked ticker volumes unavailable.")
            return []

        pairs: list[str] = []
        ranked = sorted(
            per_base.items(),
            key=lambda item: (-float(item[1][0]), str(item[1][1])),
        )
        for _base, (_volume_rank, pair) in ranked:
            pairs.append(pair)
            if len(pairs) >= top_n:
                break
        return pairs

    try:
        cg_symbols, cg_data = _coingecko_symbols()
        valid, cg_strict_unresolved = _pairs_from_symbols(cg_symbols)
        if valid:
            return valid, cg_data
        if cg_data and cg_strict_unresolved:
            debug_fn("Provider liquidity universe available, but strict pair ranking could not resolve exchange feeds.")
            return [], cg_data

        cp_symbols, cp_data = _coinpaprika_symbols()
        valid_cp, cp_strict_unresolved = _pairs_from_symbols(cp_symbols)
        if valid_cp:
            debug_fn("CoinGecko empty/unmapped; using CoinPaprika fallback.")
            return valid_cp, cp_data
        if cp_data and cp_strict_unresolved:
            debug_fn("CoinPaprika liquidity universe available, but strict pair ranking could not resolve exchange feeds.")
            return [], cp_data

        fallback = _exchange_fallback_pairs()
        if fallback:
            debug_fn("Provider mapping empty; using exchange market fallback.")
            # Provider rows are intentionally dropped here: if symbols cannot be
            # mapped to the exchange universe, enrichment is no longer reliable.
            return fallback, []
        return [], []
    except Exception as exc:
        debug_fn(f"get_top_volume_usdt_symbols error: {exc}")
        fallback = _exchange_fallback_pairs()
        if fallback:
            debug_fn("Using exchange market fallback after error.")
            return fallback, []
        return [], []


def fetch_market_cap_rows_for_symbols(
    http_get_json: Callable[..., object],
    resolve_coin_id: Callable[[str], str | None],
    symbols: list[str] | tuple[str, ...],
    debug_fn: Callable[[str], None],
    vs_currency: str = "usd",
) -> list[dict]:
    requested_symbols: list[str] = []
    seen_requested: set[str] = set()
    for raw_symbol in symbols:
        raw = str(raw_symbol or "").strip()
        base = raw.split("/", 1)[0] if "/" in raw else raw
        symbol = canonical_base_symbol(base)
        if not symbol or symbol in seen_requested:
            continue
        seen_requested.add(symbol)
        requested_symbols.append(symbol)

    if not requested_symbols:
        return []

    symbol_to_id: dict[str, str] = {}
    for symbol in requested_symbols:
        try:
            coin_id = resolve_coin_id(symbol)
        except Exception as exc:
            debug_fn(f"CoinGecko coin id resolution failed for {symbol}: {exc}")
            coin_id = None
        if coin_id:
            symbol_to_id[symbol] = str(coin_id)

    rows_by_symbol: dict[str, dict] = {}
    if symbol_to_id:
        coin_ids = list(dict.fromkeys(symbol_to_id.values()))
        data = http_get_json(
            "https://api.coingecko.com/api/v3/coins/markets",
            params={
                "vs_currency": vs_currency,
                "ids": ",".join(coin_ids),
                "order": "market_cap_desc",
                "per_page": max(1, len(coin_ids)),
                "page": 1,
                "sparkline": False,
            },
            timeout=10,
            retries=3,
        )
        if isinstance(data, list):
            id_to_symbol = {coin_id: symbol for symbol, coin_id in symbol_to_id.items()}
            for coin in data:
                coin_id = str((coin or {}).get("id") or "").strip()
                if not coin_id:
                    continue
                symbol = id_to_symbol.get(coin_id)
                if not symbol:
                    continue
                row = dict(coin)
                row["symbol"] = symbol.lower()
                rows_by_symbol[symbol] = row
        else:
            debug_fn(f"CoinGecko market-cap payload invalid for targeted lookup: {type(data)}")

    missing_symbols = [symbol for symbol in requested_symbols if symbol not in rows_by_symbol]
    if missing_symbols:
        data = http_get_json("https://api.coinpaprika.com/v1/tickers", timeout=12, retries=2)
        if isinstance(data, list):
            missing_set = set(missing_symbols)
            paprika_candidates: dict[str, list[dict]] = {symbol: [] for symbol in missing_symbols}
            for coin in data:
                symbol = canonical_base_symbol((coin or {}).get("symbol") or "")
                if symbol not in missing_set:
                    continue
                quotes = (coin or {}).get("quotes") if isinstance(coin, dict) else {}
                usd_q = quotes.get("USD") if isinstance(quotes, dict) else {}
                mcap = usd_q.get("market_cap") if isinstance(usd_q, dict) else None
                try:
                    mcap_f = float(mcap) if mcap is not None else 0.0
                except Exception:
                    mcap_f = 0.0
                paprika_candidates.setdefault(symbol, []).append(
                    {
                        "id": coin.get("id"),
                        "symbol": symbol.lower(),
                        "market_cap": mcap_f,
                    }
                )
            for symbol in missing_symbols:
                candidates = list(paprika_candidates.get(symbol, []))
                if not candidates:
                    continue
                if len(candidates) > 1:
                    debug_fn(f"CoinPaprika ambiguous ticker match for {symbol}; skipping fallback enrichment.")
                    continue
                candidate = candidates[0]
                current = rows_by_symbol.get(symbol)
                current_mcap = 0.0
                if isinstance(current, dict):
                    try:
                        current_mcap = float(current.get("market_cap") or 0.0)
                    except Exception:
                        current_mcap = 0.0
                if float(candidate.get("market_cap") or 0.0) <= current_mcap:
                    continue
                rows_by_symbol[symbol] = candidate
        else:
            debug_fn(f"CoinPaprika market-cap payload invalid for targeted lookup: {type(data)}")

    order = {symbol: idx for idx, symbol in enumerate(requested_symbols)}
    rows = list(rows_by_symbol.values())
    rows.sort(
        key=lambda row: order.get(
            canonical_base_symbol((row or {}).get("symbol") or ""),
            len(order),
        )
    )
    return rows
