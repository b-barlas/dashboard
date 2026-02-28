"""Market scanner decision policy helpers."""

from __future__ import annotations

from math import isnan


def structure_state(
    signal_dir: str,
    ai_dir: str,
    strength: float,
    agreement: float,
) -> str:
    """Classify market structure quality without depending on scalp planning.

    This state is intentionally direction/confirmation-centric:
    - FULL: strong direction + AI alignment
    - TREND: tradable trend context, but not strongest alignment
    - EARLY: early structure, requires tighter risk controls
    - NONE: no actionable structure
    """
    if signal_dir not in {"LONG", "SHORT"}:
        return "NONE"

    s = float(strength)
    a = float(agreement)
    ad = str(ai_dir or "").upper()

    if ad in {"LONG", "SHORT"} and ad != signal_dir:
        return "NONE"

    if ad == signal_dir:
        if s >= 60 and a >= 0.67:
            return "FULL"
        if s >= 55:
            return "TREND"
        return "EARLY"

    if ad == "NEUTRAL":
        if s >= 70:
            return "TREND"
        if s >= 55:
            return "EARLY"
        return "NONE"

    return "NONE"


def action_decision(
    signal_dir: str,
    strength: float,
    structure_state_val: str,
    conviction_label: str,
    agreement: float,
    adx_val: float,
) -> str:
    action, _reason = _action_decision_core(
        signal_dir,
        strength,
        structure_state_val,
        conviction_label,
        agreement,
        adx_val,
    )
    return action


def action_reason(
    signal_dir: str,
    strength: float,
    structure_state_val: str,
    conviction_label: str,
    agreement: float,
    adx_val: float,
) -> str:
    _action, reason = _action_decision_core(
        signal_dir,
        strength,
        structure_state_val,
        conviction_label,
        agreement,
        adx_val,
    )
    return reason


def action_decision_with_reason(
    signal_dir: str,
    strength: float,
    structure_state_val: str,
    conviction_label: str,
    agreement: float,
    adx_val: float,
) -> tuple[str, str]:
    """Return action class and compact reason code in one call."""
    return _action_decision_core(
        signal_dir,
        strength,
        structure_state_val,
        conviction_label,
        agreement,
        adx_val,
    )


def _action_decision_core(
    signal_dir: str,
    strength: float,
    structure_state_val: str,
    conviction_label: str,
    agreement: float,
    adx_val: float,
) -> tuple[str, str]:
    if signal_dir not in {"LONG", "SHORT"}:
        return "⛔ SKIP", "NO_DIRECTION"
    if conviction_label == "CONFLICT" or strength < 35:
        if conviction_label == "CONFLICT":
            return "⛔ SKIP", "TECH_AI_CONFLICT"
        return "⛔ SKIP", "LOW_STRENGTH"
    if structure_state_val == "NONE":
        return "⛔ SKIP", "NO_STRUCTURE"
    # Require known trend-strength context for ENTER. Unknown ADX can still be WATCH.
    if isnan(adx_val):
        return "👀 WATCH", "ADX_UNKNOWN"

    adx_f = float(adx_val)
    # In very weak trend, skip only when strength is also weak-mid.
    if adx_f < 12 and strength < 70:
        return "⛔ SKIP", "ADX_TOO_LOW"

    # ENTER classes require non-weak trend context (ADX >= 20) so Action
    # semantics stay consistent with the ADX label model.
    # Primary class: both trend and AI confirmations are aligned.
    enter_trend_ai = (
        structure_state_val == "FULL"
        and adx_f >= 20
        and strength >= 60
        and agreement >= 0.67
        and conviction_label in {"HIGH", "MEDIUM"}
    )

    # Leader + veto model:
    # - Trend-Led: trend drives the decision; AI only acts as guardrail/veto.
    # - AI-Led: AI drives the decision; trend only acts as guardrail/veto.
    enter_trend_led = (
        structure_state_val in {"FULL", "TREND", "EARLY"}
        and adx_f >= 20
        and strength >= 56
        and agreement < 0.78
        and conviction_label != "CONFLICT"
    )
    enter_ai_led = (
        structure_state_val in {"EARLY", "TREND"}
        and adx_f >= 20
        and strength >= 45
        and agreement >= 0.78
        and conviction_label != "CONFLICT"
        and not (adx_f < 14 and strength < 55)
    )

    if enter_trend_ai:
        return "✅ ENTER (Trend+AI)", "ENTER_TREND_AI"
    if enter_trend_led:
        return "🟡 ENTER (Trend-Led)", "ENTER_TREND_LED"
    if enter_ai_led:
        return "🟡 ENTER (AI-Led)", "ENTER_AI_LED"
    return "👀 WATCH", "NEEDS_CONFIRMATION"
