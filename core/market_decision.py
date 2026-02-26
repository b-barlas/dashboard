"""Market scanner decision policy helpers."""

from __future__ import annotations

from math import isnan


def setup_badge(scalp_dir: str, signal_dir: str, ai_dir: str) -> str:
    if scalp_dir and signal_dir in {"LONG", "SHORT"} and signal_dir == ai_dir == scalp_dir:
        return "🟢 Aligned"
    if scalp_dir and signal_dir in {"LONG", "SHORT"} and signal_dir == scalp_dir and ai_dir == "NEUTRAL":
        return "🟡 Tech-Only"
    if scalp_dir:
        return "⚪ Draft"
    return "🔴 No Setup"


def action_decision(
    signal_dir: str,
    strength: float,
    setup_badge_val: str,
    conviction_label: str,
    agreement: float,
    adx_val: float,
    rr_ratio: float,
    has_plan: bool,
    *,
    min_enter_rr: float = 1.30,
) -> str:
    if signal_dir not in {"LONG", "SHORT"}:
        return "⛔ SKIP"
    if conviction_label == "CONFLICT" or strength < 35:
        return "⛔ SKIP"

    adx_ok = isnan(adx_val) or adx_val >= 16
    enter_ok = (
        strength >= 60
        and conviction_label in {"HIGH", "MEDIUM"}
        and agreement >= 0.60
        and adx_ok
    )
    if enter_ok:
        return "✅ ENTER"
    return "👀 WATCH"


def trade_quality(
    action: str,
    setup_badge_val: str,
    conviction_label: str,
    strength: float,
    agreement: float,
    rr_ratio: float,
) -> str:
    if conviction_label == "CONFLICT":
        return "🔴 C"
    if "ENTER" in action and "Aligned" in setup_badge_val and strength >= 60 and agreement >= 0.65 and rr_ratio >= 1.5:
        return "🟢 A"
    if (
        ("ENTER" in action or "WATCH" in action)
        and "No Setup" not in setup_badge_val
        and conviction_label != "CONFLICT"
        and strength >= 45
        and agreement >= 0.55
        and rr_ratio >= 1.3
    ):
        return "🟡 B"
    return "🔴 C"
