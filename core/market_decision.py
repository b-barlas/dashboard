"""Market scanner decision policy helpers."""

from __future__ import annotations

from math import isnan

ACTION_SKIP = "⛔ SKIP"
ACTION_WATCH = "WATCH"
ACTION_ENTER_TREND_AI = "✅ ENTER (Trend+AI)"
ACTION_ENTER_TREND_LED = "🟡 ENTER (Trend-Led)"
ACTION_ENTER_AI_LED = "🟡 ENTER (AI-Led)"

ACTION_REASON_TEXT = {
    "NO_DIRECTION": "No clear direction (neutral signal).",
    "TECH_AI_CONFLICT": "Technical and AI directions are in conflict.",
    "LOW_STRENGTH": "Strength is below minimum threshold.",
    "NO_STRUCTURE": "Structure quality is not actionable.",
    "ADX_UNKNOWN": "Trend strength (ADX) is unavailable; waiting.",
    "ADX_TOO_LOW": "Trend strength is too low for execution.",
    "ENTER_TREND_AI": "Trend and AI confirmations align with quality gates.",
    "ENTER_TREND_LED": "Trend leads and passes quality gates; AI is used as veto only.",
    "ENTER_AI_LED": "AI leads with strong agreement; trend is used as veto only.",
    "NEEDS_CONFIRMATION": "Direction exists, but confirmation is incomplete.",
    "AI_UNAVAILABLE": "AI ensemble is unavailable; forcing WATCH until model context recovers.",
}


def ai_vote_metrics(
    ai_dir: str,
    directional_agreement: float,
    consensus_agreement: float,
) -> tuple[int, float, float]:
    """Return (display_votes, display_ratio, decision_agreement).

    - display_ratio/votes: for UI display. If AI direction is neutral, use consensus agreement.
    - decision_agreement: for decision engine. Always directional agreement (market/spot parity).
    """
    ai_key = _dir_key(ai_dir)
    dir_agree = max(0.0, min(1.0, float(directional_agreement)))
    cons_agree = max(0.0, min(1.0, float(consensus_agreement)))
    display_ratio = dir_agree if ai_key in {"UPSIDE", "DOWNSIDE"} else cons_agree
    display_votes = max(0, min(3, int(round(display_ratio * 3.0))))
    return display_votes, display_ratio, dir_agree


def normalize_action_class(action: str) -> str:
    """Normalize action text to a stable class key.

    Returns one of: ENTER_TREND_AI, ENTER_TREND_LED, ENTER_AI_LED, WATCH, SKIP, UNKNOWN.
    """
    s = str(action or "").strip().upper()
    if not s:
        return "UNKNOWN"
    if "ENTER (TREND+AI)" in s:
        return "ENTER_TREND_AI"
    if "ENTER (TREND-LED)" in s:
        return "ENTER_TREND_LED"
    if "ENTER (AI-LED)" in s:
        return "ENTER_AI_LED"
    if "WATCH" in s:
        return "WATCH"
    if "SKIP" in s:
        return "SKIP"
    return "UNKNOWN"


def action_rank(action: str) -> int:
    cls = normalize_action_class(action)
    if cls.startswith("ENTER_"):
        return 3
    if cls == "WATCH":
        return 2
    if cls == "SKIP":
        return 1
    return 0


def compact_action_label(action: str) -> str:
    """Compact, UI-safe action label for tables/kpi."""
    cls = normalize_action_class(action)
    if cls == "WATCH":
        return "WATCH"
    if cls == "ENTER_TREND_AI":
        return "ENTER T+AI"
    if cls == "ENTER_TREND_LED":
        return "ENTER Trend"
    if cls == "ENTER_AI_LED":
        return "ENTER AI"
    if cls == "SKIP":
        return "SKIP"
    return str(action or "").strip()


def action_reason_text(code: str) -> str:
    return ACTION_REASON_TEXT.get(str(code or "").upper(), "")


def _dir_key(value: str) -> str:
    s = str(value or "").strip().upper()
    if s in {"UPSIDE", "LONG", "BUY", "BULLISH"}:
        return "UPSIDE"
    if s in {"DOWNSIDE", "SHORT", "SELL", "BEARISH"}:
        return "DOWNSIDE"
    return "NEUTRAL"


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
    sdir = _dir_key(signal_dir)
    ad = _dir_key(ai_dir)
    if sdir == "NEUTRAL":
        return "NONE"

    s = float(strength)
    a = float(agreement)

    if ad != "NEUTRAL" and ad != sdir:
        return "NONE"

    if ad == sdir:
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
    if _dir_key(signal_dir) == "NEUTRAL":
        return ACTION_SKIP, "NO_DIRECTION"
    if conviction_label == "CONFLICT" or strength < 35:
        if conviction_label == "CONFLICT":
            return ACTION_SKIP, "TECH_AI_CONFLICT"
        return ACTION_SKIP, "LOW_STRENGTH"
    if structure_state_val == "NONE":
        return ACTION_SKIP, "NO_STRUCTURE"
    # Require known trend-strength context for ENTER. Unknown ADX can still be WATCH.
    if isnan(adx_val):
        return ACTION_WATCH, "ADX_UNKNOWN"

    adx_f = float(adx_val)
    # In very weak trend, skip only when strength is also weak-mid.
    if adx_f < 12 and strength < 70:
        return ACTION_SKIP, "ADX_TOO_LOW"

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
        return ACTION_ENTER_TREND_AI, "ENTER_TREND_AI"
    if enter_trend_led:
        return ACTION_ENTER_TREND_LED, "ENTER_TREND_LED"
    if enter_ai_led:
        return ACTION_ENTER_AI_LED, "ENTER_AI_LED"
    return ACTION_WATCH, "NEEDS_CONFIRMATION"
