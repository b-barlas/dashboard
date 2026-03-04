"""UI formatting and badge helpers."""

from __future__ import annotations

import math
import re

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
        label = "Strong Bullish"
    elif score >= 60:
        label = "Bullish"
    elif score >= 40:
        label = "Neutral"
    elif score >= 20:
        label = "Bearish"
    else:
        label = "Strong Bearish"
    return f"{score} ({label})"


def signal_plain(signal: str) -> str:
    """Map detailed signals to an internal LONG/SHORT/WAIT label."""
    s = str(signal or "").strip().upper()
    if not s:
        return "WAIT"
    if s in {
        "STRONG BUY",
        "BUY",
        "LONG",
        "UPSIDE",
        "STRONG UPSIDE",
        "BULLISH",
        "STRONG BULLISH",
    }:
        return "LONG"
    if s in {
        "STRONG SELL",
        "SELL",
        "SHORT",
        "DOWNSIDE",
        "STRONG DOWNSIDE",
        "BEARISH",
        "STRONG BEARISH",
    }:
        return "SHORT"
    return "WAIT"


def direction_label(direction: str) -> str:
    d = direction_key(direction)
    if d == "UPSIDE":
        return "Upside"
    if d == "DOWNSIDE":
        return "Downside"
    return "Neutral"


def direction_key(direction: str) -> str:
    """Normalize any direction-like text into UPSIDE/DOWNSIDE/NEUTRAL."""
    d = str(direction or "").strip().upper()
    if not d:
        return "NEUTRAL"
    if d in {"UPSIDE", "LONG", "BUY", "BULLISH", "STRONG BUY"}:
        return "UPSIDE"
    if d in {"DOWNSIDE", "SHORT", "SELL", "BEARISH", "STRONG SELL"}:
        return "DOWNSIDE"
    return "NEUTRAL"


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
    t = str(trend or "").strip().lower()
    if t == "bullish":
        return "▲ Bullish"
    if t == "bearish":
        return "▼ Bearish"
    if t == "neutral":
        return "→ Neutral"
    return ""


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


def _stochrsi_soft_thresholds(timeframe: str | None = None) -> tuple[float, float]:
    """Return timeframe-adaptive soft thresholds as (low, high)."""
    tf = str(timeframe or "").strip().lower()
    if tf in {"1m", "3m", "5m"}:
        return 0.15, 0.85
    if tf in {"4h", "1d"}:
        return 0.22, 0.78
    return 0.20, 0.80


def format_stochrsi(value, timeframe: str | None = None):
    try:
        v = float(value)
    except Exception:
        return "N/A"
    if math.isnan(v):
        return "N/A"
    low_thr, high_thr = _stochrsi_soft_thresholds(timeframe)
    if v <= low_thr:
        return "🟢 Low"
    if v >= high_thr:
        return "🔴 High"
    return "→ Neutral"


def style_delta(val: str, positive: str = POSITIVE, negative: str = NEGATIVE) -> str:
    if val.startswith("▲"):
        return f"color: {positive}; font-weight: 600;"
    if val.startswith("▼"):
        return f"color: {negative}; font-weight: 600;"
    return ""


def style_signal(val: str, positive: str = POSITIVE, negative: str = NEGATIVE, warning: str = WARNING) -> str:
    if "UPSIDE" in val or "BULLISH" in val:
        return f"color: {positive}; font-weight: 600;"
    if "DOWNSIDE" in val or "BEARISH" in val:
        return f"color: {negative}; font-weight: 600;"
    return f"color: {warning}; font-weight: 600;"


def style_scalp_opp(val: str, positive: str = POSITIVE, negative: str = NEGATIVE) -> str:
    k = direction_key(val)
    if k == "UPSIDE":
        return f"color: {positive}; font-weight: 600;"
    if k == "DOWNSIDE":
        return f"color: {negative}; font-weight: 600;"
    return ""


def sanitize_trading_terms(text: object) -> str:
    """Normalize legacy long/short and buy/sell wording for user-facing UI text."""
    s = "" if text is None else str(text)
    if not s:
        return ""
    replacements = [
        (r"\bSTRONG BUY\b", "Strong Bullish"),
        (r"\bSTRONG SELL\b", "Strong Bearish"),
        (r"\bBUY\b", "Bullish"),
        (r"\bSELL\b", "Bearish"),
        (r"\bLONG\b", "Upside"),
        (r"\bSHORT\b", "Downside"),
    ]
    for pattern, repl in replacements:
        s = re.sub(pattern, repl, s, flags=re.IGNORECASE)
    return s


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
