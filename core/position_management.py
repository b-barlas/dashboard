from __future__ import annotations

from dataclasses import dataclass

from core.trading_copy import trade_gate_key

from core.trading_copy import copy_text, playbook_key


@dataclass(frozen=True)
class PositionManagementSnapshot:
    action_key: str
    label: str
    score: int
    tone: str
    size_guidance: str
    adds_guidance: str
    risk_guidance: str
    note: str


def _position_bias(direction: str) -> str:
    return "UPSIDE" if str(direction or "").strip().upper() == "LONG" else "DOWNSIDE"


def _is_aligned(position_bias: str, signal_bias: str) -> bool:
    normalized = str(signal_bias or "").strip().upper()
    if normalized in {"", "UNKNOWN", "NEUTRAL", "WAIT"}:
        return False
    return normalized == position_bias


def _playbook_is_hostile(playbook: str) -> bool:
    key = playbook_key(playbook)
    return key in {"MEAN_REVERSION_OR_STAND_ASIDE", "DEFENSIVE_DOWNSIDE_ONLY"}


def _flow_is_supportive(position_bias: str, flow_proxy: str) -> bool:
    key = str(flow_proxy or "").strip().upper()
    return (position_bias == "UPSIDE" and "SHORTS CROWDED" in key) or (
        position_bias == "DOWNSIDE" and "LONGS CROWDED" in key
    )


def _flow_is_crowded_against(position_bias: str, flow_proxy: str) -> bool:
    key = str(flow_proxy or "").strip().upper()
    return (position_bias == "UPSIDE" and "LONGS CROWDED" in key) or (
        position_bias == "DOWNSIDE" and "SHORTS CROWDED" in key
    )


def _volatility_bucket(volatility_regime: str) -> str:
    key = str(volatility_regime or "").strip().upper()
    if "HIGH" in key:
        return "HOT"
    if "LOW" in key:
        return "CALM"
    return "NORMAL"


def _leverage_bucket(leverage: float) -> str:
    lev = float(leverage or 0.0)
    if lev >= 10.0:
        return "HIGH"
    if lev >= 6.0:
        return "ELEVATED"
    return "NORMAL"


def _speed_bucket(short_term_move_pct: float | None) -> str:
    if short_term_move_pct is None:
        return "NORMAL"
    move = abs(float(short_term_move_pct))
    if move >= 3.0:
        return "FAST"
    if move >= 1.5:
        return "ACTIVE"
    return "NORMAL"


def _spike_support(position_bias: str, volume_spike_label: str) -> tuple[bool, bool]:
    key = str(volume_spike_label or "").strip().upper()
    supportive = (position_bias == "UPSIDE" and "UP SPIKE" in key) or (
        position_bias == "DOWNSIDE" and "DOWN SPIKE" in key
    )
    adverse = (position_bias == "UPSIDE" and "DOWN SPIKE" in key) or (
        position_bias == "DOWNSIDE" and "UP SPIKE" in key
    )
    return supportive, adverse


def build_position_management_snapshot(
    *,
    direction: str,
    health_label: str,
    health_score: float,
    health_notes: list[str] | None,
    levered_pnl_pct: float,
    liq_distance_pct: float | None,
    leverage: float,
    invalidated: bool,
    invalidation_distance_pct: float | None,
    spot_direction: str,
    tactical_direction: str,
    ai_direction: str,
    selected_confidence: float,
    context_fit_label: str,
    context_fit_aggression: str,
    adaptive_label: str,
    execution_fit_label: str,
    session_fit_label: str,
    archive_guardrail_label: str,
    catalyst_window: str,
    trade_gate: str,
    playbook: str,
    flow_proxy: str,
    volatility_regime: str = "",
    short_term_move_pct: float | None = None,
    volume_spike_label: str = "",
    hold_profile_label: str = "",
    hold_profile_note: str = "",
    exit_quality_label: str = "",
    exit_quality_note: str = "",
) -> PositionManagementSnapshot:
    note_set = {str(item or "").strip().lower() for item in list(health_notes or []) if str(item or "").strip()}
    score = int(max(0, min(100, round(float(health_score)))))
    position_bias = _position_bias(direction)
    spot_aligned = _is_aligned(position_bias, spot_direction)
    tactical_aligned = _is_aligned(position_bias, tactical_direction)
    ai_aligned = _is_aligned(position_bias, ai_direction)

    context_label = str(context_fit_label or "").strip()
    context_key = trade_gate_key(context_label)
    adaptive = str(adaptive_label or "").strip()
    execution = str(execution_fit_label or "").strip()
    session = str(session_fit_label or "").strip()
    guardrail = str(archive_guardrail_label or "").strip().upper()
    catalyst = str(catalyst_window or "").strip()
    gate = str(trade_gate or "").strip()
    gate_key = trade_gate_key(gate)
    playbook_value = str(playbook or "").strip()
    flow_value = str(flow_proxy or "").strip()
    hold_profile = str(hold_profile_label or "").strip()
    hold_profile_text = str(hold_profile_note or "").strip()
    exit_quality = str(exit_quality_label or "").strip()
    exit_quality_text = str(exit_quality_note or "").strip()

    blocking_catalyst = catalyst.startswith("Blocking")
    high_impact_catalyst = catalyst.startswith("High Impact")
    stand_aside_context = context_key == "NO_TRADE" or gate_key == "NO_TRADE"
    defensive_context = context_key == "DEFENSIVE_ONLY"
    guardrail_severe = "GUARDRAIL" in guardrail
    guardrail_caution = "CAUTION" in guardrail
    guardrailed = guardrail_severe or guardrail_caution
    session_fragile = session == "Session Fragile"
    hostile_playbook = _playbook_is_hostile(playbook_value)
    supportive_flow = _flow_is_supportive(position_bias, flow_value)
    crowded_against = _flow_is_crowded_against(position_bias, flow_value)
    volatility_bucket = _volatility_bucket(volatility_regime)
    leverage_bucket = _leverage_bucket(leverage)
    speed_bucket = _speed_bucket(short_term_move_pct)
    supportive_spike, adverse_spike = _spike_support(position_bias, volume_spike_label)
    high_stress_exposure = volatility_bucket == "HOT" and leverage_bucket in {"ELEVATED", "HIGH"}
    fast_hostile_tape = speed_bucket == "FAST" and leverage_bucket in {"ELEVATED", "HIGH"}
    supportive_stack = (
        adaptive == "Historically Favored"
        and execution == "Execution Proven"
        and session == "Session Supportive"
        and context_key == "TRADEABLE"
        and spot_aligned
        and tactical_aligned
        and not guardrailed
        and not blocking_catalyst
        and not high_impact_catalyst
        and not hostile_playbook
        and not crowded_against
        and not high_stress_exposure
    )

    if not spot_aligned and not tactical_aligned:
        score -= 16
    elif not spot_aligned or not tactical_aligned:
        score -= 8
    else:
        score += 4

    if not ai_aligned and str(ai_direction or "").strip().upper() not in {"", "UNKNOWN", "NEUTRAL"}:
        score -= 5
    if float(selected_confidence or 0.0) >= 68.0:
        score += 4
    elif float(selected_confidence or 0.0) < 45.0:
        score -= 6

    if stand_aside_context:
        score -= 18
    elif defensive_context:
        score -= 8
    if hostile_playbook:
        score -= 12

    if guardrail_severe:
        score -= 10
    elif guardrail_caution:
        score -= 5
    if session_fragile:
        score -= 4
    if blocking_catalyst:
        score -= 18
    elif high_impact_catalyst:
        score -= 5

    if liq_distance_pct is not None and float(liq_distance_pct) < 4.0:
        score -= 16
    elif liq_distance_pct is not None and float(liq_distance_pct) < 8.0:
        score -= 7

    if volatility_bucket == "HOT":
        score -= 4
        if leverage_bucket == "HIGH":
            score -= 8
        elif leverage_bucket == "ELEVATED":
            score -= 4
    elif volatility_bucket == "CALM" and leverage_bucket == "NORMAL":
        score += 2
    if speed_bucket == "FAST":
        score -= 3
        if leverage_bucket in {"ELEVATED", "HIGH"}:
            score -= 3
    elif speed_bucket == "ACTIVE" and leverage_bucket == "HIGH":
        score -= 2
    if adverse_spike:
        score -= 4
    elif supportive_spike and float(levered_pnl_pct) >= 0.0 and not high_stress_exposure:
        score += 2

    if invalidation_distance_pct is not None and float(invalidation_distance_pct) < 1.0:
        score -= 12
    elif invalidation_distance_pct is not None and float(invalidation_distance_pct) < 2.0:
        score -= 6

    if float(levered_pnl_pct) >= 8.0:
        score += 5
    elif float(levered_pnl_pct) <= -8.0:
        score -= 8
    elif float(levered_pnl_pct) < 0.0:
        score -= 3

    if supportive_stack:
        score += 8
    elif supportive_flow:
        score += 3
    if crowded_against:
        score -= 3
    if exit_quality == "Healthy Exit Discipline":
        score += 2
    elif exit_quality == "Late Loss Risk" and float(levered_pnl_pct) <= 0.0:
        score -= 4
    if hold_profile == "Needs Room":
        score += 3
    elif hold_profile == "Quick Follow-Through":
        score -= 2

    score = int(max(0, min(100, round(score))))

    if invalidated or health_label == "EXIT" or score < 20:
        return PositionManagementSnapshot(
            action_key="EXIT",
            label=copy_text("position.mgmt.exit.label"),
            score=score,
            tone="negative",
            size_guidance=copy_text("position.mgmt.exit.size"),
            adds_guidance=copy_text("position.mgmt.exit.adds"),
            risk_guidance=copy_text("position.mgmt.exit.risk"),
            note=copy_text("position.mgmt.exit.note"),
        )

    if (
        health_label == "REDUCE"
        or stand_aside_context
        or defensive_context
        or guardrailed
        or blocking_catalyst
        or high_stress_exposure
        or fast_hostile_tape
        or adverse_spike
        or (liq_distance_pct is not None and float(liq_distance_pct) < 6.0 and float(leverage) >= 8.0)
        or score < 55
    ):
        reason = "Edge is weakening"
        if blocking_catalyst:
            reason = "A near-term catalyst window is too close to stay aggressive"
        elif adverse_spike:
            reason = "Volume spike is leaning against the position, so protection matters more than upside"
        elif high_stress_exposure:
            reason = "High volatility and leverage together make this position too fragile to press"
        elif fast_hostile_tape:
            reason = "Short-term move speed is too aggressive for the current leverage profile"
        elif guardrailed:
            reason = "Matched archive history is warning against pressing this window"
        elif stand_aside_context or defensive_context:
            reason = f"Current market stance is {context_label.lower()}"
        elif hostile_playbook:
            reason = "Current playbook is not a regime to press open positions"
        elif crowded_against:
            reason = "Crowding is leaning against the position, so protection matters more than upside"
        elif "signal conflict" in note_set:
            reason = "Live directional structure is now conflicting with the position"
        elif "liquidation too close" in note_set:
            reason = "Liquidation is too close to justify normal size"
        elif not spot_aligned and not tactical_aligned:
            reason = "Spot and tactical bias are no longer supporting the position"
        reduce_note_extra = f" {exit_quality_text.strip()}" if exit_quality_text else ""
        return PositionManagementSnapshot(
            action_key="REDUCE",
            label=copy_text("position.mgmt.reduce.label"),
            score=score,
            tone="warning",
            size_guidance=copy_text("position.mgmt.reduce.size"),
            adds_guidance=copy_text("position.mgmt.reduce.adds"),
            risk_guidance=copy_text("position.mgmt.reduce.risk"),
            note=copy_text("position.mgmt.reduce.note", reason=reason, extra=reduce_note_extra),
        )

    if supportive_stack and float(levered_pnl_pct) >= 0.0 and score >= 78:
        return PositionManagementSnapshot(
            action_key="PRESS",
            label=copy_text("position.mgmt.press.label"),
            score=score,
            tone="positive",
            size_guidance=copy_text("position.mgmt.press.size"),
            adds_guidance=copy_text("position.mgmt.press.adds"),
            risk_guidance=copy_text("position.mgmt.press.risk"),
            note=(
                "Archive, session, and live structure are aligned. "
                + ("Crowding is supportive. " if supportive_flow else "")
                + ("Volume spike is confirming the move. " if supportive_spike else "")
                + ("Volatility is calm enough to let the winner breathe. " if volatility_bucket == "CALM" else "")
                + ("Give this winner room. " if hold_profile == "Needs Room" else "")
                + ("Archive says winners are often cut too early here. " if exit_quality == "Winner Cut Risk" else "")
                + "This is a window to manage like a winner, not a rescue trade."
            ).strip(),
        )

    hold_note = "The position is still manageable, but this is a discipline hold rather than an aggressive add window."
    if high_stress_exposure:
        hold_note = "The position can still work, but high volatility and leverage together argue for discipline over aggression."
    if spot_aligned and tactical_aligned and session == "Session Supportive":
        hold_note = "The position still has structural support. Hold cleanly, but wait for fresh confirmation before pressing."
    if adverse_spike:
        hold_note = "The position still has a path, but the latest volume spike is leaning against it. Manage tighter than usual."
    if hold_profile_text:
        hold_note = f"{hold_note} {hold_profile_text}".strip()
    if exit_quality_text:
        hold_note = f"{hold_note} {exit_quality_text}".strip()
    return PositionManagementSnapshot(
        action_key="HOLD",
        label=copy_text("position.mgmt.hold.label"),
        score=score,
        tone="info",
        size_guidance=copy_text("position.mgmt.hold.size"),
        adds_guidance=context_fit_aggression or "Selective adds only",
        risk_guidance=copy_text("position.mgmt.hold.risk"),
        note=hold_note,
    )
