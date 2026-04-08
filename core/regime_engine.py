"""Shared market regime engine for live dashboard playbook gating."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class MarketRegimeSnapshot:
    regime_key: str
    label: str
    playbook: str
    note: str
    tradable: bool
    no_trade: bool
    score: float


def build_market_regime_snapshot(
    *,
    setup_quality_score: float,
    setup_quality_mode: str,
    market_lead_score: float,
    market_lead_state: str,
    lead_breadth_component: float,
    lead_rotation_component: float,
    lead_flow_component: float,
    lead_dominance_component: float,
    direction_score: float,
    breadth_score: float,
    trust_score: float,
) -> MarketRegimeSnapshot:
    mode = str(setup_quality_mode or "").strip().upper()
    lead_state = str(market_lead_state or "").strip().upper()
    sq = float(max(0.0, min(100.0, setup_quality_score)))
    lead_score = float(max(0.0, min(100.0, market_lead_score)))
    breadth_component = float(lead_breadth_component)
    rotation_component = float(lead_rotation_component)
    flow_component = float(lead_flow_component)
    dominance_component = float(lead_dominance_component)
    direction_strength = float(max(0.0, min(100.0, direction_score)))
    breadth_strength = float(max(0.0, min(100.0, breadth_score)))
    trust = float(max(0.0, min(100.0, trust_score)))

    if (
        mode == "RISK-OFF"
        and (
            lead_state == "DOWNSIDE"
            or lead_score <= 42.0
            or flow_component <= -18.0
        )
    ):
        return MarketRegimeSnapshot(
            regime_key="RISK_OFF_PRESSURE",
            label="Risk-Off Pressure",
            playbook="Defensive / downside only",
            note="Broad conditions are weak or fragmented. Protect capital first and keep upside attempts selective.",
            tradable=False,
            no_trade=True,
            score=min(sq, 45.0),
        )

    if (
        mode == "RISK-ON"
        and lead_state == "UPSIDE"
        and direction_strength >= 45.0
        and breadth_strength >= 45.0
        and trust >= 45.0
    ):
        return MarketRegimeSnapshot(
            regime_key="RISK_ON_TREND",
            label="Risk-On Trend",
            playbook="Trend continuation",
            note="Conditions support continuation hunting. Let confirmed strength lead and avoid overthinking clean setups.",
            tradable=True,
            no_trade=False,
            score=max(sq, lead_score),
        )

    if (
        lead_state == "UPSIDE"
        and sq >= 58.0
        and rotation_component >= 8.0
        and dominance_component >= -5.0
        and breadth_component >= 10.0
    ):
        return MarketRegimeSnapshot(
            regime_key="ALT_ROTATION",
            label="Alt Rotation",
            playbook="Selective upside rotation",
            note="Early upside pressure is spreading beyond the majors. Focus on leaders and require clean confirmation.",
            tradable=True,
            no_trade=False,
            score=max(sq, lead_score),
        )

    if (
        lead_state in {"UPSIDE", "DOWNSIDE"}
        and sq >= 50.0
        and abs(breadth_component) < 18.0
    ):
        return MarketRegimeSnapshot(
            regime_key="SELECTIVE_BREAKOUT",
            label="Selective Breakout",
            playbook="Wait for confirmation",
            note="Pressure is building, but breadth is not fully broad yet. Prioritize clean leaders and avoid forcing second-tier names.",
            tradable=True,
            no_trade=False,
            score=float((sq + lead_score) * 0.5),
        )

    if sq < 52.0 or (abs(breadth_component) < 10.0 and abs(rotation_component) < 10.0):
        return MarketRegimeSnapshot(
            regime_key="RANGE_CHOP",
            label="Range / Chop",
            playbook="Mean reversion or stand aside",
            note="The market is not offering broad directional edge yet. Reduce aggression and treat breakouts with skepticism.",
            tradable=False,
            no_trade=True,
            score=min(sq, 55.0),
        )

    return MarketRegimeSnapshot(
        regime_key="SELECTIVE_BALANCE",
        label="Selective Balance",
        playbook="Selective only",
        note="Some setups can work, but the market still wants filtering. Take only aligned ideas with clear structure.",
        tradable=True,
        no_trade=False,
        score=float((sq * 0.65) + (lead_score * 0.35)),
    )
