from __future__ import annotations

import unittest
from types import SimpleNamespace

from core.no_trade_engine import build_market_trade_gate
from core.regime_engine import build_market_regime_snapshot


class NoTradeEngineTests(unittest.TestCase):
    def test_degraded_scan_forces_no_trade(self) -> None:
        regime = build_market_regime_snapshot(
            setup_quality_score=71.0,
            setup_quality_mode="Risk-On",
            market_lead_score=66.0,
            market_lead_state="Upside",
            lead_breadth_component=18.0,
            lead_rotation_component=12.0,
            lead_flow_component=14.0,
            lead_dominance_component=8.0,
            direction_score=62.0,
            breadth_score=66.0,
            trust_score=61.0,
        )
        snap = build_market_trade_gate(
            market_regime_snapshot=regime,
            scan_degraded=True,
            setup_quality_score=71.0,
            setup_quality_mode="Risk-On",
            market_lead_score=66.0,
            market_lead_state="Upside",
            direction_score=62.0,
            breadth_score=66.0,
            trust_score=61.0,
            ready_count=3,
            watch_count=4,
            skip_count=2,
        )
        self.assertEqual(snap.gate_key, "NO_TRADE")
        self.assertTrue(snap.no_trade)

    def test_risk_off_without_clean_edge_blocks_new_trades(self) -> None:
        regime = build_market_regime_snapshot(
            setup_quality_score=44.0,
            setup_quality_mode="Risk-Off",
            market_lead_score=35.0,
            market_lead_state="Balanced",
            lead_breadth_component=-8.0,
            lead_rotation_component=-10.0,
            lead_flow_component=-18.0,
            lead_dominance_component=-6.0,
            direction_score=28.0,
            breadth_score=30.0,
            trust_score=42.0,
        )
        snap = build_market_trade_gate(
            market_regime_snapshot=regime,
            scan_degraded=False,
            setup_quality_score=44.0,
            setup_quality_mode="Risk-Off",
            market_lead_score=35.0,
            market_lead_state="Balanced",
            direction_score=28.0,
            breadth_score=30.0,
            trust_score=42.0,
            ready_count=0,
            watch_count=3,
            skip_count=7,
        )
        self.assertEqual(snap.gate_key, "NO_TRADE")
        self.assertEqual(snap.reason_code, "REGIME_NO_TRADE")

    def test_selective_market_returns_selective_gate(self) -> None:
        regime = build_market_regime_snapshot(
            setup_quality_score=61.0,
            setup_quality_mode="Selective",
            market_lead_score=69.0,
            market_lead_state="Upside",
            lead_breadth_component=18.0,
            lead_rotation_component=20.0,
            lead_flow_component=14.0,
            lead_dominance_component=2.0,
            direction_score=38.0,
            breadth_score=58.0,
            trust_score=52.0,
        )
        snap = build_market_trade_gate(
            market_regime_snapshot=regime,
            scan_degraded=False,
            setup_quality_score=61.0,
            setup_quality_mode="Selective",
            market_lead_score=69.0,
            market_lead_state="Upside",
            direction_score=38.0,
            breadth_score=58.0,
            trust_score=52.0,
            ready_count=1,
            watch_count=5,
            skip_count=3,
        )
        self.assertEqual(snap.gate_key, "SELECTIVE_ONLY")
        self.assertFalse(snap.no_trade)

    def test_risk_on_clear_environment_becomes_tradeable(self) -> None:
        regime = build_market_regime_snapshot(
            setup_quality_score=76.0,
            setup_quality_mode="Risk-On",
            market_lead_score=73.0,
            market_lead_state="Upside",
            lead_breadth_component=24.0,
            lead_rotation_component=12.0,
            lead_flow_component=16.0,
            lead_dominance_component=6.0,
            direction_score=61.0,
            breadth_score=67.0,
            trust_score=63.0,
        )
        snap = build_market_trade_gate(
            market_regime_snapshot=regime,
            scan_degraded=False,
            setup_quality_score=76.0,
            setup_quality_mode="Risk-On",
            market_lead_score=73.0,
            market_lead_state="Upside",
            direction_score=61.0,
            breadth_score=67.0,
            trust_score=63.0,
            ready_count=3,
            watch_count=4,
            skip_count=1,
        )
        self.assertEqual(snap.gate_key, "TRADEABLE")
        self.assertTrue(snap.tradable)

    def test_probe_only_market_stays_selective_not_no_trade(self) -> None:
        regime = build_market_regime_snapshot(
            setup_quality_score=68.0,
            setup_quality_mode="Selective",
            market_lead_score=66.0,
            market_lead_state="Upside",
            lead_breadth_component=15.0,
            lead_rotation_component=8.0,
            lead_flow_component=10.0,
            lead_dominance_component=2.0,
            direction_score=46.0,
            breadth_score=55.0,
            trust_score=54.0,
        )
        snap = build_market_trade_gate(
            market_regime_snapshot=regime,
            scan_degraded=False,
            setup_quality_score=68.0,
            setup_quality_mode="Selective",
            market_lead_score=66.0,
            market_lead_state="Upside",
            direction_score=46.0,
            breadth_score=55.0,
            trust_score=54.0,
            ready_count=0,
            watch_count=2,
            skip_count=10,
            probe_count=3,
        )
        self.assertEqual(snap.gate_key, "SELECTIVE_ONLY")
        self.assertEqual(snap.reason_code, "PROBE_ONLY_SETUPS")

    def test_risk_on_clear_environment_downgrades_when_session_archive_is_weak(self) -> None:
        regime = build_market_regime_snapshot(
            setup_quality_score=76.0,
            setup_quality_mode="Risk-On",
            market_lead_score=73.0,
            market_lead_state="Upside",
            lead_breadth_component=24.0,
            lead_rotation_component=12.0,
            lead_flow_component=16.0,
            lead_dominance_component=6.0,
            direction_score=61.0,
            breadth_score=67.0,
            trust_score=63.0,
        )
        snap = build_market_trade_gate(
            market_regime_snapshot=regime,
            scan_degraded=False,
            setup_quality_score=76.0,
            setup_quality_mode="Risk-On",
            market_lead_score=73.0,
            market_lead_state="Upside",
            direction_score=61.0,
            breadth_score=67.0,
            trust_score=63.0,
            ready_count=3,
            watch_count=4,
            skip_count=1,
            session_fit_score=-4.0,
            session_fit_label="Session Fragile",
            session_fit_note="European session has been a weak conversion window lately.",
        )
        self.assertEqual(snap.gate_key, "SELECTIVE_ONLY")
        self.assertEqual(snap.reason_code, "SESSION_ARCHIVE_WEAK")

    def test_market_wide_catalyst_downgrades_tradeable_to_selective(self) -> None:
        regime = build_market_regime_snapshot(
            setup_quality_score=76.0,
            setup_quality_mode="Risk-On",
            market_lead_score=73.0,
            market_lead_state="Upside",
            lead_breadth_component=24.0,
            lead_rotation_component=12.0,
            lead_flow_component=16.0,
            lead_dominance_component=6.0,
            direction_score=61.0,
            breadth_score=67.0,
            trust_score=63.0,
        )
        snap = build_market_trade_gate(
            market_regime_snapshot=regime,
            market_catalyst_snapshot=SimpleNamespace(
                gate_bias="SELECTIVE_ONLY",
                note="US CPI is within the next day. Keep size smaller and stay selective.",
            ),
            scan_degraded=False,
            setup_quality_score=76.0,
            setup_quality_mode="Risk-On",
            market_lead_score=73.0,
            market_lead_state="Upside",
            direction_score=61.0,
            breadth_score=67.0,
            trust_score=63.0,
            ready_count=3,
            watch_count=4,
            skip_count=1,
        )
        self.assertEqual(snap.gate_key, "SELECTIVE_ONLY")
        self.assertEqual(snap.reason_code, "CATALYST_SELECTIVE")

    def test_archive_guardrail_can_downgrade_tradeable_market(self) -> None:
        regime = build_market_regime_snapshot(
            setup_quality_score=76.0,
            setup_quality_mode="Risk-On",
            market_lead_score=73.0,
            market_lead_state="Upside",
            lead_breadth_component=24.0,
            lead_rotation_component=12.0,
            lead_flow_component=16.0,
            lead_dominance_component=6.0,
            direction_score=61.0,
            breadth_score=67.0,
            trust_score=63.0,
        )
        snap = build_market_trade_gate(
            market_regime_snapshot=regime,
            scan_degraded=False,
            setup_quality_score=76.0,
            setup_quality_mode="Risk-On",
            market_lead_score=73.0,
            market_lead_state="Upside",
            direction_score=61.0,
            breadth_score=67.0,
            trust_score=63.0,
            ready_count=3,
            watch_count=4,
            skip_count=1,
            archive_guardrail_penalty=5.8,
            archive_guardrail_label="Archive Guardrail",
            archive_guardrail_note="The tape looks open, but matched playbook history is weak enough to avoid full aggression.",
        )
        self.assertEqual(snap.gate_key, "SELECTIVE_ONLY")
        self.assertEqual(snap.reason_code, "ARCHIVE_GUARDRAIL")

    def test_severe_archive_cluster_can_force_no_trade(self) -> None:
        regime = build_market_regime_snapshot(
            setup_quality_score=63.0,
            setup_quality_mode="Selective",
            market_lead_score=67.0,
            market_lead_state="Upside",
            lead_breadth_component=16.0,
            lead_rotation_component=12.0,
            lead_flow_component=10.0,
            lead_dominance_component=2.0,
            direction_score=44.0,
            breadth_score=58.0,
            trust_score=54.0,
        )
        snap = build_market_trade_gate(
            market_regime_snapshot=regime,
            scan_degraded=False,
            setup_quality_score=63.0,
            setup_quality_mode="Selective",
            market_lead_score=67.0,
            market_lead_state="Upside",
            direction_score=44.0,
            breadth_score=58.0,
            trust_score=54.0,
            ready_count=1,
            watch_count=5,
            skip_count=2,
            session_fit_score=-2.8,
            session_fit_label="Session Fragile",
            session_fit_note="European session has been a weaker conversion window lately.",
            archive_guardrail_penalty=6.9,
            archive_guardrail_label="Archive Guardrail",
            archive_guardrail_note="Matched archive history is weak across Primary Alert: Trade Gate and Primary Alert x Session: Trade Gate | European (08-16 UTC).",
        )
        self.assertEqual(snap.gate_key, "NO_TRADE")
        self.assertEqual(snap.reason_code, "ARCHIVE_CLUSTER_NO_TRADE")


if __name__ == "__main__":
    unittest.main()
