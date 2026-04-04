"""Dependency factory and bootstrap-safe fallback helpers."""

from __future__ import annotations

from collections.abc import Mapping


def direction_key_fallback(direction: str) -> str:
    d = str(direction or "").strip().upper()
    if d in {"UPSIDE", "LONG", "BUY", "BULLISH", "STRONG BUY"}:
        return "UPSIDE"
    if d in {"DOWNSIDE", "SHORT", "SELL", "BEARISH", "STRONG SELL"}:
        return "DOWNSIDE"
    return "NEUTRAL"


def direction_label_fallback(direction: str) -> str:
    d = direction_key_fallback(direction)
    if d == "UPSIDE":
        return "Upside"
    if d == "DOWNSIDE":
        return "Downside"
    return "Neutral"


def signal_plain_fallback(signal: str) -> str:
    s = str(signal or "").strip().upper()
    if s in {"STRONG BUY", "BUY"}:
        return "LONG"
    if s in {"STRONG SELL", "SELL"}:
        return "SHORT"
    return "WAIT"


def missing_fetch_coingecko_ohlcv_by_coin_id(*_args, **_kwargs):
    return None


missing_fetch_coingecko_ohlcv_by_coin_id._codex_missing_dep = True
missing_fetch_coingecko_ohlcv_by_coin_id._codex_missing_dep_reason = (
    "dependency injection missing at app boot"
)


def sanitize_trading_terms_fallback(text):
    return "" if text is None else str(text)


def style_delta_fallback(*_args, **_kwargs) -> str:
    return ""


def style_scalp_opp_fallback(*_args, **_kwargs) -> str:
    return ""


def style_signal_fallback(*_args, **_kwargs) -> str:
    return ""


OPTIONAL_DEP_DEFAULTS: dict[str, object] = {
    # UI/helper fallbacks: safe to inject if a helper import changed.
    "direction_key": direction_key_fallback,
    "direction_label": direction_label_fallback,
    "signal_plain": signal_plain_fallback,
    "fetch_coingecko_ohlcv_by_coin_id": missing_fetch_coingecko_ohlcv_by_coin_id,
    "sanitize_trading_terms": sanitize_trading_terms_fallback,
    "style_delta": style_delta_fallback,
    "style_scalp_opp": style_scalp_opp_fallback,
    "style_signal": style_signal_fallback,
}


def build_app_deps(source: Mapping[str, object], **overrides: object) -> dict:
    """Build the dependency dictionary expected by `ui.app_shell.render_app`."""
    from ui.tab_registry import required_dep_keys

    required = {"st", "ACCENT", "POSITIVE"} | required_dep_keys()
    merged = dict(source)
    merged.update(overrides)

    # Soft-fill optional helper dependencies to keep app boot resilient across
    # helper refactors.
    for key, value in OPTIONAL_DEP_DEFAULTS.items():
        if key in required and key not in merged:
            merged[key] = value

    missing = sorted(k for k in required if k not in merged)
    if missing:
        # KeyError messages can be redacted in some Streamlit environments.
        # Keep a plain log line so the exact missing keys are visible in logs.
        print(f"[deps_factory] Missing app dependencies: {', '.join(missing)}")
        raise KeyError(f"Missing app dependencies: {', '.join(missing)}")

    return {k: merged[k] for k in required}
