"""Shared market risk-budget and signal sizing helpers."""

from __future__ import annotations

from dataclasses import dataclass

from core.catalyst_engine import catalyst_signal_note, catalyst_signal_size_cap
from core.market_decision import normalize_action_class


@dataclass(frozen=True)
class RiskSizingSnapshot:
    tier_key: str
    label: str
    unit_fraction: float
    note: str


def _direction_key(value: object) -> str:
    d = str(value or "").strip().upper()
    if d in {"UPSIDE", "LONG", "BUY", "BULLISH"}:
        return "UPSIDE"
    if d in {"DOWNSIDE", "SHORT", "SELL", "BEARISH"}:
        return "DOWNSIDE"
    return "NEUTRAL"


def _tier_from_fraction(unit_fraction: float, *, capped_note: str | None = None) -> RiskSizingSnapshot:
    fraction = float(max(0.0, min(1.0, unit_fraction)))
    if fraction >= 0.99:
        label = "Full Unit"
        key = "FULL"
        note = "Top-tier setup. Normal size is justified."
    elif fraction >= 0.49:
        label = "Half Unit"
        key = "HALF"
        note = "Good setup, but keep size controlled."
    elif fraction >= 0.24:
        label = "Probe Only"
        key = "PROBE"
        note = "Early or selective setup. Use only probe-size risk."
    else:
        label = "Flat"
        key = "FLAT"
        note = "Do not allocate fresh risk to this setup."
    if capped_note:
        note = f"{note} {capped_note}".strip()
    return RiskSizingSnapshot(
        tier_key=key,
        label=label,
        unit_fraction=fraction,
        note=note,
    )


def market_default_risk_budget(market_trade_gate_snapshot, market_catalyst_snapshot=None) -> RiskSizingSnapshot:
    gate_key = str(getattr(market_trade_gate_snapshot, "gate_key", "") or "").strip().upper()
    if gate_key == "TRADEABLE":
        base = 1.0
    elif gate_key == "SELECTIVE_ONLY":
        base = 0.5
    elif gate_key == "DEFENSIVE_ONLY":
        base = 0.25
    else:
        base = 0.0

    catalyst_cap_raw = getattr(market_catalyst_snapshot, "size_cap_fraction", None)
    catalyst_cap = 1.0 if catalyst_cap_raw is None else float(catalyst_cap_raw)
    capped_note = None
    if catalyst_cap < base:
        capped_note = "Catalyst risk is capping size."
    return _tier_from_fraction(min(base, catalyst_cap), capped_note=capped_note)


def build_signal_risk_sizing(
    *,
    market_trade_gate_snapshot,
    market_catalyst_snapshot=None,
    direction: str,
    setup_confirm: str,
    confidence: float | None,
    ai_confidence: float | None,
    ai_aligned: bool,
    market_lead_aligned: bool,
    lead_active: bool,
    rr_ratio: float | None,
    adaptive_edge_score: float | None = None,
    session_fit_score: float | None = None,
    archive_guardrail_penalty: float | None = None,
    archive_guardrail_label: str | None = None,
    archive_guardrail_note: str | None = None,
    symbol: str | None = None,
    sector_tag: str | None = None,
) -> RiskSizingSnapshot:
    action_class = normalize_action_class(setup_confirm)
    direction_key = _direction_key(direction)
    confidence_value = float(max(0.0, min(100.0, float(confidence or 0.0))))
    ai_conf_value = float(max(0.0, min(100.0, float(ai_confidence or 0.0))))
    rr_value = float(rr_ratio or 0.0)
    adaptive_score = float(max(0.0, min(100.0, float(adaptive_edge_score or 50.0))))
    session_score = float(session_fit_score or 0.0)
    guardrail_penalty = float(max(0.0, float(archive_guardrail_penalty or 0.0)))
    guardrail_label_key = str(archive_guardrail_label or "").strip().upper()
    guardrail_note_text = str(archive_guardrail_note or "").strip()

    gate_key = str(getattr(market_trade_gate_snapshot, "gate_key", "") or "").strip().upper()
    gate_reason = str(getattr(market_trade_gate_snapshot, "reason_code", "") or "").strip().upper()
    if getattr(market_trade_gate_snapshot, "no_trade", False) or direction_key == "NEUTRAL" or action_class == "SKIP":
        return _tier_from_fraction(0.0)

    gate_budget_snapshot = market_default_risk_budget(
        market_trade_gate_snapshot,
        market_catalyst_snapshot,
    )
    gate_cap = gate_budget_snapshot.unit_fraction
    catalyst_cap = catalyst_signal_size_cap(
        market_catalyst_snapshot,
        symbol=str(symbol or ""),
        sector_tag=str(sector_tag or ""),
    )
    catalyst_note = catalyst_signal_note(
        market_catalyst_snapshot,
        symbol=str(symbol or ""),
        sector_tag=str(sector_tag or ""),
    )
    gate_cap = min(gate_cap, catalyst_cap)
    if gate_key == "DEFENSIVE_ONLY":
        gate_cap = 0.5 if direction_key == "DOWNSIDE" else 0.25
        gate_cap = min(gate_cap, catalyst_cap)
    elif gate_key == "SELECTIVE_ONLY" and not market_lead_aligned:
        gate_cap = min(gate_cap, 0.25)

    if action_class == "PROBE":
        desired = 0.25
        if (
            lead_active
            and confidence_value >= 66.0
            and rr_value >= 1.7
            and (market_lead_aligned or ai_aligned)
        ):
            desired = 0.5
    elif action_class == "WATCH":
        desired = 0.0
    else:
        score = 0
        if action_class == "ENTER_TREND_AI":
            score += 3
        elif action_class in {"ENTER_TREND_LED", "ENTER_AI_LED"}:
            score += 2
        if confidence_value >= 82.0:
            score += 2
        elif confidence_value >= 70.0:
            score += 1
        if ai_aligned and ai_conf_value >= 60.0:
            score += 1
        if market_lead_aligned:
            score += 1
        if lead_active:
            score += 1
        if rr_value >= 2.2:
            score += 2
        elif rr_value >= 1.7:
            score += 1

        if score >= 8:
            desired = 1.0
        elif score >= 5:
            desired = 0.5
        else:
            desired = 0.25

        if adaptive_score >= 64.0:
            desired = min(1.0, desired + 0.25)
        elif adaptive_score <= 36.0:
            desired = max(0.0, desired - 0.25)
        if session_score >= 2.5:
            desired = min(1.0, desired + 0.25)
        elif session_score <= -2.5:
            desired = max(0.0, desired - 0.25)

    severe_archive_cluster = (
        guardrail_label_key == "ARCHIVE GUARDRAIL"
        and guardrail_penalty >= 6.5
        and (
            gate_reason in {"ARCHIVE_GUARDRAIL", "FILTER_HARDER_ARCHIVE", "SELECTIVE_ARCHIVE_WEAK", "ARCHIVE_CLUSTER_NO_TRADE"}
            or gate_key in {"SELECTIVE_ONLY", "DEFENSIVE_ONLY"}
            or session_score <= -2.0
        )
    )
    weak_cluster_probe_cap = False
    if severe_archive_cluster:
        desired = min(desired, 0.25)
        weak_cluster_probe_cap = True
    elif guardrail_penalty >= 6.0:
        desired = max(0.0, desired - 0.5)
    elif guardrail_penalty >= 3.0:
        desired = max(0.0, desired - 0.25)

    actual = min(desired, gate_cap)
    note_parts: list[str] = []
    if actual < desired and gate_key in {"SELECTIVE_ONLY", "DEFENSIVE_ONLY"}:
        note_parts.append("Current market gate is capping size.")
    if actual < desired and catalyst_cap < desired and "Catalyst risk is capping size." not in note_parts:
        note_parts.append("Catalyst risk is capping size.")
    if catalyst_note and catalyst_cap < 1.0:
        note_parts.append(catalyst_note)
    if action_class == "PROBE":
        note_parts.append("Probe setup: starter size only until full confirmation appears.")
    if adaptive_score >= 64.0 and action_class not in {"WATCH", "PROBE"}:
        note_parts.append("Learned execution history is supporting size.")
    elif adaptive_score <= 36.0 and action_class not in {"WATCH", "PROBE"}:
        note_parts.append("Learned execution history is trimming size.")
    if session_score >= 2.5 and action_class not in {"WATCH", "PROBE"}:
        note_parts.append("Current session archive is supporting size.")
    elif session_score <= -2.5 and action_class not in {"WATCH", "PROBE"}:
        note_parts.append("Current session archive is trimming size.")
    if weak_cluster_probe_cap:
        note_parts.append("Historically weak alert/playbook timing cluster is forcing probe-only size.")
    if guardrail_penalty >= 3.0:
        note_parts.append(
            guardrail_note_text
            or (
                "Archive guardrail is trimming size."
                if guardrail_label_key == "ARCHIVE GUARDRAIL"
                else "Archive caution is keeping size smaller."
            )
        )
    if action_class == "PROBE":
        actual = min(actual, 0.5)
    capped_note = " ".join(note_parts) or None
    return _tier_from_fraction(actual, capped_note=capped_note)
