"""Theme constants and UI composition helpers."""

from __future__ import annotations

import html
import math

from ui.helpers import format_adx, format_stochrsi, format_trend

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
    v = str(val or "")
    u = v.upper()
    if "VERY STRONG" in u or "EXTREME" in u:
        return POSITIVE
    if "WEAK" in u:
        return NEGATIVE
    if "STARTING" in u:
        return WARNING
    if "STRONG" in u:
        return POSITIVE
    if "UP SPIKE" in u:
        return POSITIVE
    if "DOWN SPIKE" in u:
        return NEGATIVE
    if "SPIKE" in u:
        return WARNING
    if any(k in v for k in ["Bullish", "Above", "Oversold", "Low", "Near Bottom"]):
        return POSITIVE
    if any(k in v for k in ["Bearish", "Below", "Overbought", "High", "Near Top"]):
        return NEGATIVE
    return WARNING


def _clean_indicator_label(val: str) -> str:
    """Strip icon glyphs/prefix markers so grid stays text-first."""
    v = str(val or "").strip()
    for token in ["🟢", "🔴", "🟡", "⚪", "🔥", "▲▲", "▲", "▼", "→", "–"]:
        v = v.replace(token, "")
    return " ".join(v.split()).strip()


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
    atr_comment: str = "",
    candle_pattern: str = "",
    spike_label: str = "",
    spike_hover: str = "",
    timeframe: str | None = None,
    ichimoku_hover: str | None = None,
) -> str:
    """Build indicator grid HTML used across Spot/Position/AI tabs."""
    indicators: list[tuple[str, str, str, str]] = []
    if supertrend_trend:
        supertrend_txt = _clean_indicator_label(format_trend(supertrend_trend))
        indicators.append(("SuperTrend", supertrend_txt, ind_color(supertrend_txt), ""))
    if ichimoku_trend:
        ichimoku_txt = _clean_indicator_label(format_trend(ichimoku_trend))
        indicators.append(("Ichimoku", ichimoku_txt, ind_color(ichimoku_txt), str(ichimoku_hover or "")))
    if vwap_label:
        vwap_txt = _clean_indicator_label(vwap_label)
        indicators.append(("VWAP", vwap_txt, ind_color(vwap_txt), ""))
    if not _is_nan(adx_val):
        adx_txt = _clean_indicator_label(format_adx(adx_val))
        indicators.append(("ADX", adx_txt, ind_color(adx_txt), ""))
    if bollinger_bias:
        boll_txt = _clean_indicator_label(bollinger_bias)
        indicators.append(("Bollinger", boll_txt, ind_color(boll_txt), ""))
    if not _is_nan(stochrsi_k_val):
        srsi_txt = _clean_indicator_label(format_stochrsi(stochrsi_k_val, timeframe=timeframe))
        indicators.append(("StochRSI", srsi_txt, ind_color(srsi_txt), ""))
    if "Bullish" in psar_trend or "Bearish" in psar_trend:
        psar_txt = _clean_indicator_label(psar_trend)
        indicators.append(("PSAR", psar_txt, ind_color(psar_txt), ""))
    if williams_label:
        will_txt = _clean_indicator_label(williams_label)
        indicators.append(("Williams %R", will_txt, ind_color(will_txt), ""))
    if cci_label:
        cci_txt = _clean_indicator_label(cci_label)
        indicators.append(("CCI", cci_txt, ind_color(cci_txt), ""))
    if volume_spike:
        spike_txt = _clean_indicator_label(spike_label) if str(spike_label or "").strip() else "Spike"
        indicators.append(("Volume", spike_txt, ind_color(spike_txt), str(spike_hover or "")))
    atr_clean = atr_comment.replace("▲", "").replace("▼", "").replace("–", "").strip()
    if atr_clean:
        indicators.append(("Volatility", atr_clean, ind_color(atr_clean), ""))
    if candle_pattern:
        pattern_txt = _clean_indicator_label(candle_pattern.split(" (")[0])
        indicators.append(("Pattern", pattern_txt, ind_color(pattern_txt), ""))
    if not indicators:
        return ""
    grid_items = ""
    for name, val, color, tooltip in indicators:
        tt_attr = f" title='{html.escape(tooltip, quote=True)}'" if tooltip else ""
        grid_items += (
            f"<div style='text-align:center; padding:6px;'>"
            f"<div style='color:{TEXT_MUTED}; font-size:0.7rem; text-transform:uppercase;'>{name}</div>"
            f"<div style='color:{color}; font-size:0.85rem; font-weight:600;'{tt_attr}>{val}</div>"
            f"</div>"
        )
    return (
        f"<div style='display:grid; grid-template-columns:repeat(auto-fill, minmax(90px, 1fr)); "
        f"gap:4px; background:{CARD_BG}; border-radius:8px; padding:10px; margin:8px 0;'>"
        f"{grid_items}</div>"
    )


def calc_conviction(
    signal_dir: str,
    ai_dir: str,
    strength: float,
    ai_agreement: float = 0.0,
) -> tuple[str, str]:
    """Return alignment quality using Direction + AI agreement + strength."""
    def _dir_key(value: str) -> str:
        s = str(value or "").strip().upper()
        if s in {"UPSIDE", "LONG", "BUY", "BULLISH"}:
            return "UPSIDE"
        if s in {"DOWNSIDE", "SHORT", "SELL", "BEARISH"}:
            return "DOWNSIDE"
        return "NEUTRAL"

    sdir = _dir_key(signal_dir)
    adir = _dir_key(ai_dir)
    agree = max(0.0, min(1.0, float(ai_agreement)))
    s = float(strength)

    if sdir == "NEUTRAL":
        return "WEAK", TEXT_MUTED
    if adir != "NEUTRAL" and sdir != adir:
        return "CONFLICT", NEGATIVE
    if adir == "NEUTRAL":
        if s >= 70:
            return "TREND", WARNING
        return "WEAK", TEXT_MUTED
    if sdir == adir:
        if s >= 72 and agree >= 0.67:
            return "HIGH", POSITIVE
        if s >= 60 and agree >= 0.50:
            return "MEDIUM", WARNING
        return "WEAK", TEXT_MUTED
    return "WEAK", TEXT_MUTED


def _is_nan(value: float) -> bool:
    try:
        return math.isnan(value)
    except Exception:
        return False
