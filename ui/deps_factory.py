"""Dependency factory for app-shell injection."""

from __future__ import annotations

from collections.abc import Mapping

from ui.tab_registry import required_dep_keys


def _direction_key_fallback(direction: str) -> str:
    d = str(direction or "").strip().upper()
    if d in {"UPSIDE", "LONG", "BUY", "BULLISH", "STRONG BUY"}:
        return "UPSIDE"
    if d in {"DOWNSIDE", "SHORT", "SELL", "BEARISH", "STRONG SELL"}:
        return "DOWNSIDE"
    return "NEUTRAL"


def _direction_label_fallback(direction: str) -> str:
    d = _direction_key_fallback(direction)
    if d == "UPSIDE":
        return "Upside"
    if d == "DOWNSIDE":
        return "Downside"
    return "Neutral"


def _signal_plain_fallback(signal: str) -> str:
    s = str(signal or "").strip().upper()
    if s in {"STRONG BUY", "BUY"}:
        return "LONG"
    if s in {"STRONG SELL", "SELL"}:
        return "SHORT"
    return "WAIT"


_OPTIONAL_DEP_DEFAULTS: dict[str, object] = {
    # UI/helper fallbacks: safe to inject if a helper import changed.
    "direction_key": _direction_key_fallback,
    "direction_label": _direction_label_fallback,
    "signal_plain": _signal_plain_fallback,
    "sanitize_trading_terms": lambda t: "" if t is None else str(t),
    "style_delta": lambda *_args, **_kwargs: "",
    "style_scalp_opp": lambda *_args, **_kwargs: "",
    "style_signal": lambda *_args, **_kwargs: "",
}


def build_app_deps(source: Mapping[str, object], **overrides: object) -> dict:
    """Build the dependency dictionary expected by `ui.app_shell.render_app`."""
    required = {"st", "ACCENT", "POSITIVE"} | required_dep_keys()
    merged = dict(source)
    merged.update(overrides)

    # Soft-fill optional helper dependencies to keep app boot resilient across
    # helper refactors.
    for key, value in _OPTIONAL_DEP_DEFAULTS.items():
        if key in required and key not in merged:
            merged[key] = value

    missing = sorted(k for k in required if k not in merged)
    if missing:
        # KeyError messages can be redacted in some Streamlit environments.
        # Keep a plain log line so the exact missing keys are visible in logs.
        print(f"[deps_factory] Missing app dependencies: {', '.join(missing)}")
        raise KeyError(f"Missing app dependencies: {', '.join(missing)}")

    return {k: merged[k] for k in required}
