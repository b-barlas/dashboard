"""UI formatting and badge helpers."""

from __future__ import annotations

import math

POSITIVE = "#00FF88"
NEGATIVE = "#FF3366"
WARNING = "#FFD166"


def signal_badge(signal: str) -> str:
    """Return a simplified badge for the given signal."""
    if signal in ("STRONG BUY", "BUY"):
        return "🟢 Upside"
    if signal in ("STRONG SELL", "SELL"):
        return "🔴 Downside"
    return "⚪ Neutral"


def leverage_badge(lev: int) -> str:
    """Display leverage as a formatted badge (e.g. x5)."""
    return f"x{lev}"


def bias_score_badge(bias_score: float) -> str:
    score = round(bias_score)
    if score >= 80:
        label = "STRONG BUY"
    elif score >= 60:
        label = "BUY"
    elif score >= 40:
        label = "WAIT"
    elif score >= 20:
        label = "SELL"
    else:
        label = "STRONG SELL"
    return f"{score} ({label})"


def confidence_score_badge(confidence: float) -> str:
    # Backward-compatible alias; prefer bias_score_badge.
    return bias_score_badge(confidence)


def signal_plain(signal: str) -> str:
    """Map detailed signals to a plain LONG/SHORT/WAIT label."""
    if signal in ("STRONG BUY", "BUY"):
        return "LONG"
    if signal in ("STRONG SELL", "SELL"):
        return "SHORT"
    return "WAIT"


def direction_label(direction: str) -> str:
    d = str(direction or "").upper()
    if d == "LONG":
        return "Upside"
    if d == "SHORT":
        return "Downside"
    return "Neutral"


def format_delta(delta):
    if delta is None:
        return ""
    if delta > 0:
        triangle = "▲"
    elif delta < 0:
        triangle = "▼"
    else:
        triangle = "→"
    return f"{triangle} {abs(delta):.2f}%"


def format_trend(trend: str) -> str:
    if trend == "Bullish":
        return "▲ Bullish"
    if trend == "Bearish":
        return "▼ Bearish"
    return "–"


def format_adx(adx: float) -> str:
    try:
        adx_f = float(adx)
    except Exception:
        return "N/A"
    if math.isnan(adx_f):
        return "N/A"
    if adx_f < 20:
        return f"▼ {adx_f:.1f} (Weak)"
    if adx_f < 25:
        return f"→ {adx_f:.1f} (Starting)"
    if adx_f < 50:
        return f"▲ {adx_f:.1f} (Strong)"
    if adx_f < 75:
        return f"▲▲ {adx_f:.1f} (Very Strong)"
    return f"🔥 {adx_f:.1f} (Extreme)"


def format_stochrsi(value):
    try:
        v = float(value)
    except Exception:
        return "N/A"
    if math.isnan(v):
        return "N/A"
    if v < 0.2:
        return "🟢 Low"
    if v > 0.8:
        return "🔴 High"
    return "→ Neutral"


def style_delta(val: str, positive: str = POSITIVE, negative: str = NEGATIVE) -> str:
    if val.startswith("▲"):
        return f"color: {positive}; font-weight: 600;"
    if val.startswith("▼"):
        return f"color: {negative}; font-weight: 600;"
    return ""


def style_signal(val: str, positive: str = POSITIVE, negative: str = NEGATIVE, warning: str = WARNING) -> str:
    if "LONG" in val or "UPSIDE" in val:
        return f"color: {positive}; font-weight: 600;"
    if "SHORT" in val or "DOWNSIDE" in val:
        return f"color: {negative}; font-weight: 600;"
    return f"color: {warning}; font-weight: 600;"


def style_confidence(val: str, positive: str = POSITIVE, negative: str = NEGATIVE, warning: str = WARNING) -> str:
    if "STRONG BUY" in val or "BUY" in val:
        return f"color: {positive}; font-weight: 600;"
    if "WAIT" in val:
        return f"color: {warning}; font-weight: 600;"
    return f"color: {negative}; font-weight: 600;"


def style_scalp_opp(val: str, positive: str = POSITIVE, negative: str = NEGATIVE) -> str:
    if val in {"LONG", "Upside"}:
        return f"color: {positive}; font-weight: 600;"
    if val in {"SHORT", "Downside"}:
        return f"color: {negative}; font-weight: 600;"
    return ""


def readable_market_cap(value):
    trillion = 1_000_000_000_000
    billion = 1_000_000_000
    million = 1_000_000

    if value >= trillion:
        return f"{value / trillion:.2f}T"
    if value >= billion:
        return f"{value / billion:.2f}B"
    if value >= million:
        return f"{value / million:.2f}M"
    return f"{value:,}"
