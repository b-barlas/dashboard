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
    # Require known trend-strength context for ENTER. Unknown ADX can still be WATCH.
    if isnan(adx_val):
        return "👀 WATCH"

    adx_f = float(adx_val)
    # In very weak trend, skip only when strength is also weak-mid.
    if adx_f < 12 and strength < 70:
        return "⛔ SKIP"

    # Dynamic gate by trend regime: in stronger trends, allow earlier confirmation.
    if adx_f >= 25:
        min_strength = 55.0
        min_agreement = 0.55
    else:
        min_strength = 60.0
        min_agreement = 0.60

    enter_main = (
        adx_f >= 16
        and strength >= min_strength
        and conviction_label in {"HIGH", "MEDIUM"}
        and agreement >= min_agreement
    )
    # Controlled technical-only path to avoid neutral lock when trend and strength are exceptional.
    enter_tech_only = (
        conviction_label == "TECH-ONLY"
        and adx_f >= 24
        and strength >= 72
    )

    if enter_main or enter_tech_only:
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
