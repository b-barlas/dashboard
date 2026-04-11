"""Shared signal text/label formatters used across trader-facing tabs."""

from __future__ import annotations

import math

from core.ai_spot_bias import ai_spot_bias_display_votes
from core.confidence import ai_confidence_bucket, confidence_bucket
from core.market_decision import normalize_action_class
from core.trading_copy import copy_text, get_copy_audience, playbook_display, playbook_key, trade_gate_display, trade_gate_key

_ACTION_PRESENTATION = {
    "ENTER_TREND_AI": {
        "family": "enter",
        "trader": "TREND+AI",
        "neutral": "High-Quality Setup",
    },
    "ENTER_TREND_LED": {
        "family": "enter",
        "trader": "TREND-led",
        "neutral": "High-Quality Setup",
    },
    "ENTER_AI_LED": {
        "family": "enter",
        "trader": "AI-led",
        "neutral": "High-Quality Setup",
    },
    "PROBE": {
        "family": "probe",
        "trader": "EARLY",
        "neutral": "Early Setup",
    },
    "WATCH": {
        "family": "watch",
        "trader": "WATCH",
        "neutral": "Developing",
    },
    "SKIP": {
        "family": "skip",
        "trader": "SKIP",
        "neutral": "Not Aligned",
    },
}


def _normalized_action_key(raw_action: str) -> str:
    raw = str(raw_action or "").strip().upper()
    if raw in _ACTION_PRESENTATION:
        return raw
    return normalize_action_class(raw_action)


def _probe_display_text(
    *,
    audience: str,
    action_reason: str | None = None,
) -> str:
    reason = str(action_reason or "").strip().upper()
    if audience == "neutral":
        if reason == "PROBE_TREND":
            return "Early Trend-Led Setup"
        if reason == "PROBE_AI":
            return "Early Model-Led Setup"
        if reason == "PROBE_DUAL":
            return "Early Trend + Model Setup"
        if reason in {"ARCHIVE_UPGRADE_TO_PROBE", "ARCHIVE_DOWNGRADE_TO_PROBE"}:
            return "Early Archive-Calibrated Setup"
        return "Early Setup"
    if reason == "PROBE_TREND":
        return "EARLY (Trend-Led)"
    if reason == "PROBE_AI":
        return "EARLY (AI-Led)"
    if reason == "PROBE_DUAL":
        return "EARLY (Trend+AI)"
    if reason in {"ARCHIVE_UPGRADE_TO_PROBE", "ARCHIVE_DOWNGRADE_TO_PROBE"}:
        return "EARLY (Archive)"
    return "EARLY"


def spot_bias_label(direction: str) -> str:
    raw = str(direction or "").strip().upper()
    if raw == "UPSIDE":
        return "Upside"
    if raw == "DOWNSIDE":
        return "Downside"
    return "Neutral"


def spot_confidence_display(score: float) -> str:
    score_f = max(0.0, min(100.0, float(score)))
    return f"{score_f:.0f}% ({confidence_bucket(score_f).title()})"


def ai_confidence_display(snapshot, score: float) -> str:
    score_f = max(0.0, min(100.0, float(score)))
    label = ai_confidence_bucket(
        score_f,
        direction=str(snapshot.direction or ""),
        support_votes=int(ai_spot_bias_display_votes(snapshot)),
        timeframe_conflict=bool(snapshot.timeframe_conflict),
        degraded_data=bool(snapshot.degraded_data),
    )
    return f"{score_f:.0f}% ({label.title()})"


def ai_spot_tf_note(snapshot) -> str:
    status = str(getattr(snapshot, "status", "") or "").strip()
    note = str(getattr(snapshot, "note", "") or "").strip()
    suffix_parts = []
    if status:
        suffix_parts.append(f"Status {status}")
    if note:
        suffix_parts.append(note)
    suffix = f" | {' | '.join(suffix_parts)}" if suffix_parts else ""
    return (
        f"{str(snapshot.timeframe).upper()}: {spot_bias_label(snapshot.direction)} | "
        f"Score {float(snapshot.score):.1f} | "
        f"Prob Up {float(snapshot.probability_up) * 100:.0f}% | "
        f"Directional agreement {float(snapshot.directional_agreement) * 100:.0f}% | "
        f"Consensus {float(snapshot.consensus_agreement) * 100:.0f}%{suffix}"
    )


def ai_spot_note(snapshot) -> str:
    dots = ai_spot_bias_display_votes(snapshot)
    pair_label = str(getattr(snapshot, "anchor_pair_label", "") or "").strip()
    if not pair_label:
        lead = getattr(snapshot, "lead_snapshot", getattr(snapshot, "one_day", None))
        confirm = getattr(snapshot, "confirm_snapshot", getattr(snapshot, "four_hour", None))
        if lead is not None and confirm is not None:
            pair_label = f"{str(lead.timeframe).upper()} + {str(confirm.timeframe).upper()}"
        else:
            pair_label = "HTF"
    return (
        f"AI spot bias ({pair_label}): {spot_bias_label(snapshot.direction)} | "
        f"Combined score {float(snapshot.score):.1f} | "
        f"Conviction quality {float(snapshot.conviction_quality):.0f} | "
        f"Timeframe alignment {float(snapshot.timeframe_alignment):.0f} | "
        f"Displayed model-support dots {dots}/3 | "
        f"{str(snapshot.note or '').strip()} | "
        f"{ai_spot_tf_note(getattr(snapshot, 'lead_snapshot', getattr(snapshot, 'one_day', None)))} | "
        f"{ai_spot_tf_note(getattr(snapshot, 'confirm_snapshot', getattr(snapshot, 'four_hour', None)))}"
    )


def ai_confidence_note(snapshot, score: float, confidence_snapshot=None) -> str:
    dots = ai_spot_bias_display_votes(snapshot)
    caps: list[str] = []
    direction_key = str(snapshot.direction or "").strip().upper()
    if direction_key == "NEUTRAL":
        caps.append("neutral-verdict cap <=58")
    if bool(snapshot.timeframe_conflict):
        caps.append("timeframe-conflict cap <=30")
    if bool(snapshot.degraded_data):
        caps.append("degraded-data cap <=35")
    if direction_key != "NEUTRAL" and int(dots) <= 1:
        caps.append("low-model-support cap <=59")
    cap_text = f" | Active caps: {', '.join(caps)}" if caps else ""
    note = (
        f"AI confidence: {float(score):.1f}% "
        f"({ai_confidence_bucket(float(score), direction=str(snapshot.direction or ''), support_votes=int(dots), timeframe_conflict=bool(snapshot.timeframe_conflict), degraded_data=bool(snapshot.degraded_data)).title()}) | "
        f"HTF AI verdict {spot_bias_label(snapshot.direction)} | "
        f"Combined score {float(snapshot.score):.1f} | "
        f"Conviction quality {float(snapshot.conviction_quality):.0f} | "
        f"Timeframe alignment {float(snapshot.timeframe_alignment):.0f} | "
        f"Consensus quality {float(snapshot.consensus_quality):.0f} | "
        f"Model support {int(dots)}/3{cap_text}"
    )
    calibration_note = str(getattr(confidence_snapshot, "note", "") or "").strip()
    if calibration_note:
        note = f"{note} | {calibration_note}".strip()
    return note


def adx_bucket_only(adx_value: float) -> str:
    try:
        adx_f = float(adx_value)
    except Exception:
        return ""
    if not math.isfinite(adx_f):
        return ""
    if adx_f < 20:
        return "Weak"
    if adx_f < 25:
        return "Starting"
    if adx_f < 50:
        return "Strong"
    if adx_f < 75:
        return "Very Strong"
    return "Extreme"


def action_family(raw_action: str) -> str:
    cls = _normalized_action_key(raw_action)
    if cls in _ACTION_PRESENTATION:
        return str(_ACTION_PRESENTATION[cls]["family"])
    return "unknown"


def setup_confirm_display(
    raw_action: str,
    *,
    audience: str | None = None,
    action_reason: str | None = None,
) -> str:
    cls = _normalized_action_key(raw_action)
    profile = _ACTION_PRESENTATION.get(cls)
    if profile:
        selected_audience = str(audience or get_copy_audience())
        if cls == "PROBE":
            return _probe_display_text(
                audience=selected_audience,
                action_reason=action_reason,
            )
        return str(profile.get(selected_audience) or profile.get("trader") or cls)
    return str(raw_action or "").strip() or "SKIP"


def trade_gate_display_label(label: str, *, audience: str | None = None) -> str:
    key = trade_gate_key(label)
    if key == "NO_TRADE":
        return copy_text("stance.display.stand_aside", audience=audience)
    if key == "DEFENSIVE_ONLY":
        return copy_text("stance.display.defensive_only", audience=audience)
    if key == "SELECTIVE_ONLY":
        return copy_text("stance.display.selective_only", audience=audience)
    if key == "TRADEABLE":
        return copy_text("stance.display.tradeable", audience=audience)
    normalized = str(label or "").strip()
    return normalized or "Unknown"


def _clean_note_part(value: object) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    text = text.replace("Recent market archive: ", "").replace("Recent Market scanner read: ", "").strip()
    text = " ".join(text.split())
    return text.strip(" |")


def compact_note_parts(parts: list[object], *, limit: int = 4) -> str:
    seen: set[str] = set()
    ordered: list[str] = []
    for part in list(parts or []):
        clean = _clean_note_part(part)
        key = clean.rstrip(".").strip().lower()
        if not key or key in seen:
            continue
        seen.add(key)
        ordered.append(clean)
        if len(ordered) >= max(1, int(limit or 0)):
            break
    return " • ".join(ordered)


def archived_execution_stance_label(
    *,
    trade_gate: str,
    adaptive_edge: str = "",
    archive_guardrail_severity: str = "",
) -> str:
    gate = trade_gate_key(trade_gate)
    adaptive = str(adaptive_edge or "").strip()
    guardrail = str(archive_guardrail_severity or "").strip()

    if gate == "NO_TRADE" or guardrail == "Guardrail":
        return trade_gate_display_label("NO_TRADE")
    if gate == "DEFENSIVE_ONLY" or adaptive == "Historically Weak":
        return trade_gate_display_label("DEFENSIVE_ONLY")
    if gate == "TRADEABLE" and adaptive == "Historically Favored" and guardrail == "Clear":
        return trade_gate_display_label("TRADEABLE")
    if gate in {"TRADEABLE", "SELECTIVE_ONLY"}:
        return trade_gate_display_label("SELECTIVE_ONLY")
    return trade_gate_display_label(gate)


def context_fit_snapshot(
    adaptive_snapshot,
    *,
    market_context: dict[str, str] | None = None,
    recent_symbol_market_signal: dict[str, str] | None = None,
) -> dict[str, str]:
    market_context = dict(market_context or {})
    recent_symbol_market_signal = dict(recent_symbol_market_signal or {})

    adaptive_label = str(getattr(adaptive_snapshot, "label", "") or "").strip()
    execution_label = str(getattr(adaptive_snapshot, "execution_fit_label", "") or "").strip()
    session_label = str(getattr(adaptive_snapshot, "session_fit_label", "") or "").strip()
    archive_label = str(getattr(adaptive_snapshot, "archive_guardrail_label", "") or "").strip()

    playbook = playbook_display(
        market_context.get("Playbook Key") or market_context.get("Playbook") or "",
    ).strip()
    trade_gate = trade_gate_display(
        market_context.get("Trade Gate Key") or market_context.get("Trade Gate") or "",
    ).strip()
    trade_gate_value_key = trade_gate_key(market_context.get("Trade Gate Key") or trade_gate)
    playbook_value_key = playbook_key(market_context.get("Playbook Key") or playbook)
    catalyst_window = str(market_context.get("Catalyst Window") or "").strip()
    flow_proxy = str(market_context.get("Flow Proxy") or "").strip()
    symbol_lead = str(recent_symbol_market_signal.get("Lead") or "").strip()
    signal_note = str(recent_symbol_market_signal.get("Signal Note") or "").strip()

    if "GUARDRAIL" in archive_label.upper() or trade_gate_value_key == "NO_TRADE" or catalyst_window.startswith("Blocking"):
        label = trade_gate_display_label("NO_TRADE")
        tone = "negative"
        aggression = copy_text("context_fit.aggression.no_fresh_risk")
    elif adaptive_label == "Historically Favored" and execution_label == "Execution Proven" and session_label == "Session Supportive":
        label = trade_gate_display_label("TRADEABLE")
        tone = "positive"
        aggression = copy_text("context_fit.aggression.normal_aggression")
    elif adaptive_label == "Historically Favored" or execution_label == "Execution Proven" or symbol_lead == "LEAD":
        label = trade_gate_display_label("SELECTIVE_ONLY")
        tone = "info"
        aggression = copy_text("context_fit.aggression.selective_adds_only")
    elif execution_label == "Execution Fragile" or session_label == "Session Fragile" or adaptive_label == "Historically Weak":
        label = trade_gate_display_label("DEFENSIVE_ONLY")
        tone = "warning"
        aggression = copy_text("context_fit.aggression.reduced_aggression")
    else:
        label = trade_gate_display_label("SELECTIVE_ONLY")
        tone = "neutral"
        aggression = copy_text("context_fit.aggression.probe_only")

    parts: list[str] = []
    if playbook and playbook != "Unknown":
        parts.append(f"Playbook: {playbook}")
    if trade_gate and trade_gate != "Unknown":
        parts.append(f"Gate: {trade_gate}")
    if catalyst_window and catalyst_window != "Unknown":
        parts.append(f"Catalyst: {catalyst_window}")
    if flow_proxy and flow_proxy != "Unknown":
        parts.append(f"Flow: {flow_proxy}")
    if signal_note:
        parts.append(signal_note.replace("Recent Market scanner read: ", "").strip())
    note = " | ".join(parts[:5]) if parts else "Context archive is still building."

    return {
        "gate_key": trade_gate_key(label),
        "playbook_key": playbook_value_key,
        "label": label,
        "tone": tone,
        "aggression": aggression,
        "note": note,
    }


def execution_read_note(
    adaptive_snapshot,
    *,
    context_fit: dict[str, str],
    market_context_note: str = "",
    scanner_signal_note: str = "",
) -> str:
    base_note = _clean_note_part(getattr(adaptive_snapshot, "note", ""))
    stance = trade_gate_display_label(str((context_fit or {}).get("label") or ""))
    aggression = str((context_fit or {}).get("aggression") or "").strip()
    context_note = _clean_note_part(str((context_fit or {}).get("note") or "").strip())
    if base_note and context_note.lower().startswith(f"playbook: {base_note.lower()}"):
        trimmed = context_note[len(f"Playbook: {base_note}") :].lstrip(" |")
        context_note = trimmed.strip()
    stance_line = f"Stance: {stance} — {aggression}." if stance and aggression else ""
    return compact_note_parts(
        [
            base_note,
            getattr(adaptive_snapshot, "execution_fit_note", ""),
            getattr(adaptive_snapshot, "session_fit_note", ""),
            stance_line,
            context_note,
            market_context_note,
            scanner_signal_note,
        ],
        limit=5,
    )
