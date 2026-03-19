"""Shared symbol canonicalization and alias helpers."""

from __future__ import annotations


STABLE_BASE_SYMBOLS: frozenset[str] = frozenset(
    {
        "USDT",
        "USDC",
        "BUSD",
        "DAI",
        "TUSD",
        "USDE",
        "USDD",
        "FDUSD",
        "PYUSD",
        "RLUSD",
        "USDP",
        "GUSD",
        "LUSD",
        "FRAX",
        "EURS",
        "EURC",
        "SUSDE",
        "USD1",
        "USDG",
    }
)


BASE_ALIASES: dict[str, tuple[str, ...]] = {
    "BTC": ("BTC", "XBT"),
}

_ALIAS_TO_CANONICAL = {
    alias.upper(): canonical
    for canonical, aliases in BASE_ALIASES.items()
    for alias in aliases
}


def canonical_base_symbol(symbol: str) -> str:
    s = str(symbol or "").strip().upper()
    if not s:
        return ""
    return _ALIAS_TO_CANONICAL.get(s, s)


def is_stable_base_symbol(symbol: str) -> bool:
    s = str(symbol or "").strip().upper()
    if not s:
        return False
    return s in STABLE_BASE_SYMBOLS


def base_symbol_candidates(symbol: str) -> tuple[str, ...]:
    raw = str(symbol or "").strip().upper()
    if not raw:
        return tuple()
    canonical = canonical_base_symbol(raw)
    aliases = BASE_ALIASES.get(canonical, (canonical,))
    ordered: list[str] = []
    for item in (raw, canonical, *aliases):
        token = str(item or "").strip().upper()
        if token and token not in ordered:
            ordered.append(token)
    return tuple(ordered)
