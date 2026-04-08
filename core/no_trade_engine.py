"""Shared market trade-gate / no-trade engine."""

from __future__ import annotations

from dataclasses import dataclass

from core.trading_copy import copy_text, trade_gate_display


@dataclass(frozen=True)
class MarketTradeGateSnapshot:
    gate_key: str
    label: str
    reason_code: str
    note: str
    tone: str
    no_trade: bool
    tradable: bool
    session_fit_label: str = ""
    session_fit_note: str = ""
    archive_guardrail_label: str = ""
    archive_guardrail_note: str = ""


def _gate_label(gate_key: str) -> str:
    return trade_gate_display(gate_key)


def build_market_trade_gate(
    *,
    market_regime_snapshot,
    market_catalyst_snapshot=None,
    scan_degraded: bool,
    setup_quality_score: float,
    setup_quality_mode: str,
    market_lead_score: float,
    market_lead_state: str,
    direction_score: float,
    breadth_score: float,
    trust_score: float,
    ready_count: int,
    watch_count: int,
    skip_count: int,
    probe_count: int = 0,
    session_fit_score: float = 0.0,
    session_fit_label: str = "",
    session_fit_note: str = "",
    archive_guardrail_penalty: float = 0.0,
    archive_guardrail_label: str = "",
    archive_guardrail_note: str = "",
) -> MarketTradeGateSnapshot:
    mode = str(setup_quality_mode or "").strip().upper()
    lead_state = str(market_lead_state or "").strip().upper()
    regime_key = str(getattr(market_regime_snapshot, "regime_key", "") or "").strip().upper()
    regime_note = str(getattr(market_regime_snapshot, "note", "") or "").strip()
    regime_playbook = str(getattr(market_regime_snapshot, "playbook", "") or "").strip()
    regime_no_trade = bool(getattr(market_regime_snapshot, "no_trade", False))

    sq = float(max(0.0, min(100.0, setup_quality_score)))
    lead_score = float(max(0.0, min(100.0, market_lead_score)))
    direction_strength = float(max(0.0, min(100.0, direction_score)))
    breadth_strength = float(max(0.0, min(100.0, breadth_score)))
    trust = float(max(0.0, min(100.0, trust_score)))
    session_score = float(session_fit_score or 0.0)
    session_label = str(session_fit_label or "").strip()
    session_note = str(session_fit_note or "").strip()
    archive_penalty = float(archive_guardrail_penalty or 0.0)
    archive_label = str(archive_guardrail_label or "").strip()
    archive_note = str(archive_guardrail_note or "").strip()
    catalyst_gate_bias = str(getattr(market_catalyst_snapshot, "gate_bias", "NONE") or "NONE").strip().upper()
    catalyst_note = str(getattr(market_catalyst_snapshot, "note", "") or "").strip()

    ready = max(0, int(ready_count))
    probe = max(0, int(probe_count))
    watch = max(0, int(watch_count))
    skip = max(0, int(skip_count))
    total = max(1, ready + probe + watch + skip)
    ready_ratio = ready / float(total)
    probe_ratio = probe / float(total)
    watch_ratio = watch / float(total)
    skip_ratio = skip / float(total)

    if scan_degraded:
        return MarketTradeGateSnapshot(
            gate_key="NO_TRADE",
            label=_gate_label("NO_TRADE"),
            reason_code="DEGRADED_SCAN",
            note=copy_text("no_trade.note.degraded_scan"),
            tone="negative",
            no_trade=True,
            tradable=False,
            session_fit_label=session_label,
            session_fit_note=session_note,
            archive_guardrail_label=archive_label,
            archive_guardrail_note=archive_note,
        )

    if regime_no_trade:
        return MarketTradeGateSnapshot(
            gate_key="NO_TRADE",
            label=_gate_label("NO_TRADE"),
            reason_code="REGIME_NO_TRADE",
            note=(regime_note or "The current market regime does not justify forcing new trades.")
            + (f" Current playbook: {regime_playbook}." if regime_playbook else ""),
            tone="negative",
            no_trade=True,
            tradable=False,
            session_fit_label=session_label,
            session_fit_note=session_note,
            archive_guardrail_label=archive_label,
            archive_guardrail_note=archive_note,
        )

    if catalyst_gate_bias == "NO_TRADE":
        return MarketTradeGateSnapshot(
            gate_key="NO_TRADE",
            label=_gate_label("NO_TRADE"),
            reason_code="CATALYST_BLOCK",
            note=catalyst_note or copy_text("no_trade.note.catalyst_block_default"),
            tone="negative",
            no_trade=True,
            tradable=False,
            session_fit_label=session_label,
            session_fit_note=session_note,
            archive_guardrail_label=archive_label,
            archive_guardrail_note=archive_note,
        )

    if ready <= 0 and (skip_ratio >= 0.65 or watch_ratio >= 0.75):
        if probe > 0:
            return MarketTradeGateSnapshot(
                gate_key="SELECTIVE_ONLY",
                label=_gate_label("SELECTIVE_ONLY"),
                reason_code="PROBE_ONLY_SETUPS",
                note=copy_text("no_trade.note.probe_only"),
                tone="warning",
                no_trade=False,
                tradable=True,
                session_fit_label=session_label,
                session_fit_note=session_note,
                archive_guardrail_label=archive_label,
                archive_guardrail_note=archive_note,
            )
        return MarketTradeGateSnapshot(
            gate_key="NO_TRADE",
            label=_gate_label("NO_TRADE"),
            reason_code="NO_READY_SETUPS",
            note=copy_text("no_trade.note.no_ready_setups"),
            tone="negative",
            no_trade=True,
            tradable=False,
            session_fit_label=session_label,
            session_fit_note=session_note,
            archive_guardrail_label=archive_label,
            archive_guardrail_note=archive_note,
        )

    if lead_state in {"BALANCED", "NONE", "NEUTRAL"} and breadth_strength < 40.0 and trust < 46.0:
        return MarketTradeGateSnapshot(
            gate_key="NO_TRADE",
            label=_gate_label("NO_TRADE"),
            reason_code="WEAK_PARTICIPATION",
            note=copy_text("no_trade.note.weak_participation"),
            tone="negative",
            no_trade=True,
            tradable=False,
            session_fit_label=session_label,
            session_fit_note=session_note,
        )

    if archive_penalty >= 6.5 and (
        session_score <= -2.0
        or catalyst_gate_bias == "SELECTIVE_ONLY"
        or mode == "SELECTIVE"
        or regime_key in {"ALT_ROTATION", "SELECTIVE_BREAKOUT", "SELECTIVE_BALANCE"}
    ):
        archive_cluster_note = (
            archive_note
            or "Matched archive history is weak across the current alert, playbook, and timing window."
        )
        if session_note:
            archive_cluster_note = f"{archive_cluster_note} {session_note}".strip()
        return MarketTradeGateSnapshot(
            gate_key="NO_TRADE",
            label=_gate_label("NO_TRADE"),
            reason_code="ARCHIVE_CLUSTER_NO_TRADE",
            note=(
                f"{copy_text('no_trade.note.archive_cluster_prefix')} "
                f"{archive_cluster_note}"
            ).strip(),
            tone="negative",
            no_trade=True,
            tradable=False,
            session_fit_label=session_label,
            session_fit_note=session_note,
            archive_guardrail_label=archive_label,
            archive_guardrail_note=archive_note,
        )

    if mode == "RISK-OFF":
        if lead_state == "DOWNSIDE" and direction_strength >= 45.0 and lead_score <= 45.0:
            return MarketTradeGateSnapshot(
                gate_key="DEFENSIVE_ONLY",
                label=_gate_label("DEFENSIVE_ONLY"),
                reason_code="RISK_OFF_DEFENSIVE",
                note=copy_text("no_trade.note.risk_off_defensive"),
                tone="warning",
                no_trade=False,
                tradable=True,
                session_fit_label=session_label,
                session_fit_note=session_note,
                archive_guardrail_label=archive_label,
                archive_guardrail_note=archive_note,
            )
        return MarketTradeGateSnapshot(
            gate_key="NO_TRADE",
            label=_gate_label("NO_TRADE"),
            reason_code="RISK_OFF_WEAKNESS",
            note=copy_text("no_trade.note.risk_off_weakness"),
            tone="negative",
            no_trade=True,
            tradable=False,
            session_fit_label=session_label,
            session_fit_note=session_note,
            archive_guardrail_label=archive_label,
            archive_guardrail_note=archive_note,
        )

    if mode == "SELECTIVE" or regime_key in {"ALT_ROTATION", "SELECTIVE_BREAKOUT", "SELECTIVE_BALANCE"}:
        selective_note = copy_text("no_trade.note.selective_clean")
        reason_code = "SELECTIVE_FILTER"
        if probe_ratio >= 0.25 and ready <= 0:
            selective_note = copy_text("no_trade.note.selective_probe")
            reason_code = "SELECTIVE_PROBE_WINDOW"
        if catalyst_gate_bias == "SELECTIVE_ONLY":
            selective_note = (
                f"{selective_note} {catalyst_note}".strip()
                if catalyst_note
                else copy_text("no_trade.note.selective_catalyst_default")
            )
            reason_code = "CATALYST_SELECTIVE"
        if session_score <= -3.5:
            selective_note = (
                f"{selective_note} {session_note}"
                if session_note
                else copy_text("no_trade.note.selective_session_default")
            )
            reason_code = "SELECTIVE_SESSION_WEAK"
        if archive_penalty >= 3.0:
            selective_note = (
                f"{selective_note} {archive_note}".strip()
                if archive_note
                else copy_text("no_trade.note.selective_archive_default")
            )
            reason_code = "SELECTIVE_ARCHIVE_WEAK"
        elif session_score >= 2.5 and session_note:
            selective_note = f"{selective_note} {session_note}"
        return MarketTradeGateSnapshot(
            gate_key="SELECTIVE_ONLY",
            label=_gate_label("SELECTIVE_ONLY"),
            reason_code=reason_code,
            note=selective_note,
            tone="warning",
            no_trade=False,
            tradable=True,
            session_fit_label=session_label,
            session_fit_note=session_note,
            archive_guardrail_label=archive_label,
            archive_guardrail_note=archive_note,
        )

    if (
        mode == "RISK-ON"
        and lead_state == "UPSIDE"
        and ready_ratio >= 0.10
        and sq >= 68.0
        and breadth_strength >= 50.0
        and trust >= 50.0
    ):
        if catalyst_gate_bias == "SELECTIVE_ONLY":
            return MarketTradeGateSnapshot(
                gate_key="SELECTIVE_ONLY",
                label=_gate_label("SELECTIVE_ONLY"),
                reason_code="CATALYST_SELECTIVE",
                note=(
                    catalyst_note
                    or copy_text("no_trade.note.tradeable_catalyst_default")
                ),
                tone="warning",
                no_trade=False,
                tradable=True,
                session_fit_label=session_label,
                session_fit_note=session_note,
                archive_guardrail_label=archive_label,
                archive_guardrail_note=archive_note,
            )
        if session_score <= -3.5:
            return MarketTradeGateSnapshot(
                gate_key="SELECTIVE_ONLY",
                label=_gate_label("SELECTIVE_ONLY"),
                reason_code="SESSION_ARCHIVE_WEAK",
                note=(
                    f"{copy_text('no_trade.note.tradeable_session_prefix')} "
                    f"{session_note}".strip()
                ),
                tone="warning",
                no_trade=False,
                tradable=True,
                session_fit_label=session_label,
                session_fit_note=session_note,
                archive_guardrail_label=archive_label,
                archive_guardrail_note=archive_note,
            )
        if archive_penalty >= 5.0:
            return MarketTradeGateSnapshot(
                gate_key="SELECTIVE_ONLY",
                label=_gate_label("SELECTIVE_ONLY"),
                reason_code="ARCHIVE_GUARDRAIL",
                note=(
                    archive_note
                    or copy_text("no_trade.note.tradeable_archive_default")
                ),
                tone="warning",
                no_trade=False,
                tradable=True,
                session_fit_label=session_label,
                session_fit_note=session_note,
                archive_guardrail_label=archive_label,
                archive_guardrail_note=archive_note,
            )
        tradeable_note = copy_text("no_trade.note.tradeable_base")
        if session_score >= 2.5 and session_note:
            tradeable_note = f"{tradeable_note} {session_note}"
        return MarketTradeGateSnapshot(
            gate_key="TRADEABLE",
            label=_gate_label("TRADEABLE"),
            reason_code="RISK_ON_CLEAR",
            note=tradeable_note,
            tone="positive",
            no_trade=False,
            tradable=True,
            session_fit_label=session_label,
            session_fit_note=session_note,
            archive_guardrail_label=archive_label,
            archive_guardrail_note=archive_note,
        )

    fallback_note = copy_text("no_trade.note.filter_harder_base")
    fallback_reason = "FILTER_HARDER"
    if catalyst_gate_bias == "SELECTIVE_ONLY":
        fallback_note = (
            f"{fallback_note} {catalyst_note}".strip()
            if catalyst_note
            else copy_text("no_trade.note.filter_harder_catalyst_default")
        )
        fallback_reason = "CATALYST_SELECTIVE"
    if session_score <= -3.5:
        fallback_note = f"{fallback_note} {session_note}".strip()
        fallback_reason = "FILTER_HARDER_SESSION_WEAK"
    if archive_penalty >= 3.0:
        fallback_note = f"{fallback_note} {archive_note}".strip() if archive_note else (
            copy_text("no_trade.note.filter_harder_archive_default")
        )
        fallback_reason = "FILTER_HARDER_ARCHIVE"
    elif session_score >= 2.5 and session_note:
        fallback_note = f"{fallback_note} {session_note}"
    return MarketTradeGateSnapshot(
        gate_key="SELECTIVE_ONLY",
        label=_gate_label("SELECTIVE_ONLY"),
        reason_code=fallback_reason,
        note=fallback_note,
        tone="warning",
        no_trade=False,
        tradable=True,
        session_fit_label=session_label,
        session_fit_note=session_note,
        archive_guardrail_label=archive_label,
        archive_guardrail_note=archive_note,
    )
