"""Theme constants and UI composition helpers."""

from __future__ import annotations

import math

from ui.helpers import format_adx, format_trend

PRIMARY_BG = "#000000"
CARD_BG = "#000000"
ACCENT = "#FFFFFF"
POSITIVE = "#00FF88"
NEGATIVE = "#FF3366"
WARNING = "#FFD166"
TEXT_LIGHT = "#E5E7EB"
TEXT_MUTED = "#8CA1B6"
NEON_BLUE = "#00D4FF"
NEON_PURPLE = "#B24BF3"
GOLD = "#FFD700"


def tip(label: str, tooltip: str) -> str:
    """Return HTML for a label with a hover tooltip question mark."""
    return f"{label}<span class='tt'>?<span class='ttt'>{tooltip}</span></span>"


def ind_color(val: str) -> str:
    """Return color for an indicator value."""
    if any(k in val for k in ["Bullish", "Above", "Oversold", "Low"]):
        return POSITIVE
    if any(k in val for k in ["Bearish", "Below", "Overbought", "High"]):
        return NEGATIVE
    return WARNING


def build_indicator_grid(
    supertrend_trend: str,
    ichimoku_trend: str,
    vwap_label: str,
    adx_val: float,
    bollinger_bias: str,
    stochrsi_k_val: float,
    psar_trend: str,
    williams_label: str,
    cci_label: str,
    volume_spike: bool,
    atr_comment: str,
    candle_pattern: str,
) -> str:
    """Build indicator grid HTML used across Spot/Position/AI tabs."""
    indicators = []
    if supertrend_trend:
        indicators.append(("SuperTrend", format_trend(supertrend_trend), ind_color(supertrend_trend)))
    if ichimoku_trend:
        indicators.append(("Ichimoku", format_trend(ichimoku_trend), ind_color(ichimoku_trend)))
    if vwap_label:
        indicators.append(("VWAP", vwap_label, ind_color(vwap_label)))
    if not _is_nan(adx_val):
        indicators.append(("ADX", format_adx(adx_val), WARNING))
    if bollinger_bias:
        indicators.append(("Bollinger", bollinger_bias, ind_color(bollinger_bias)))
    if not _is_nan(stochrsi_k_val):
        srsi_c = POSITIVE if stochrsi_k_val < 0.2 else (NEGATIVE if stochrsi_k_val > 0.8 else WARNING)
        indicators.append(("StochRSI", f"{stochrsi_k_val:.2f}", srsi_c))
    if "Bullish" in psar_trend or "Bearish" in psar_trend:
        indicators.append(("PSAR", psar_trend, ind_color(psar_trend)))
    if williams_label:
        indicators.append(("Williams %R", williams_label.replace("🟢 ", "").replace("🔴 ", "").replace("🟡 ", ""), ind_color(williams_label)))
    if cci_label:
        indicators.append(("CCI", cci_label.replace("🟢 ", "").replace("🔴 ", "").replace("🟡 ", ""), ind_color(cci_label)))
    if volume_spike:
        indicators.append(("Volume", "Spike ▲", POSITIVE))
    atr_clean = atr_comment.replace("▲", "").replace("▼", "").replace("–", "").strip()
    if atr_clean:
        indicators.append(("Volatility", atr_clean, ind_color(atr_clean)))
    if candle_pattern:
        indicators.append(("Pattern", candle_pattern.split(" (")[0], WARNING))
    if not indicators:
        return ""
    grid_items = "".join(
        f"<div style='text-align:center; padding:6px;'>"
        f"<div style='color:{TEXT_MUTED}; font-size:0.7rem; text-transform:uppercase;'>{name}</div>"
        f"<div style='color:{color}; font-size:0.85rem; font-weight:600;'>{val}</div>"
        f"</div>"
        for name, val, color in indicators
    )
    return (
        f"<div style='display:grid; grid-template-columns:repeat(auto-fill, minmax(90px, 1fr)); "
        f"gap:4px; background:{CARD_BG}; border-radius:8px; padding:10px; margin:8px 0;'>"
        f"{grid_items}</div>"
    )


def calc_conviction(signal_dir: str, ai_dir: str, confidence: float) -> tuple[str, str]:
    """Return conviction label and color based on signal/AI alignment."""
    if signal_dir in ("LONG", "SHORT") and signal_dir == ai_dir:
        if confidence >= 75:
            return "HIGH", POSITIVE
        if confidence >= 60:
            return "MEDIUM", WARNING
        return "LOW", TEXT_MUTED
    if signal_dir in ("LONG", "SHORT") and ai_dir not in ("NEUTRAL", "WAIT", "") and signal_dir != ai_dir:
        return "CONFLICT", NEGATIVE
    return "LOW", TEXT_MUTED


def _is_nan(value: float) -> bool:
    try:
        return math.isnan(value)
    except Exception:
        return False
