from __future__ import annotations

import unittest

from core.catalyst_engine import build_market_catalyst_snapshot
from core.no_trade_engine import build_market_trade_gate
from core.regime_engine import build_market_regime_snapshot
from core.risk_sizing_engine import build_signal_risk_sizing, market_default_risk_budget


class RiskSizingEngineTests(unittest.TestCase):
    def test_market_default_budget_matches_gate(self) -> None:
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
        gate = build_market_trade_gate(
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
        sizing = market_default_risk_budget(gate)
        self.assertEqual(sizing.tier_key, "HALF")

    def test_catalyst_caps_default_budget(self) -> None:
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
        gate = build_market_trade_gate(
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
        catalyst = build_market_catalyst_snapshot(
            [{"title": "US CPI", "event_time": "2026-04-05T12:30:00Z", "severity": "high"}],
            now="2026-04-05T08:00:00Z",
        )
        sizing = market_default_risk_budget(gate, catalyst)
        self.assertEqual(sizing.tier_key, "FLAT")

    def test_no_trade_gate_forces_flat(self) -> None:
        regime = build_market_regime_snapshot(
            setup_quality_score=42.0,
            setup_quality_mode="Risk-Off",
            market_lead_score=34.0,
            market_lead_state="Downside",
            lead_breadth_component=-35.0,
            lead_rotation_component=-18.0,
            lead_flow_component=-24.0,
            lead_dominance_component=-12.0,
            direction_score=18.0,
            breadth_score=22.0,
            trust_score=40.0,
        )
        gate = build_market_trade_gate(
            market_regime_snapshot=regime,
            scan_degraded=False,
            setup_quality_score=42.0,
            setup_quality_mode="Risk-Off",
            market_lead_score=34.0,
            market_lead_state="Downside",
            direction_score=18.0,
            breadth_score=22.0,
            trust_score=40.0,
            ready_count=0,
            watch_count=2,
            skip_count=7,
        )
        sizing = build_signal_risk_sizing(
            market_trade_gate_snapshot=gate,
            market_catalyst_snapshot=None,
            direction="Upside",
            setup_confirm="✅ ENTER (Trend+AI)",
            confidence=84.0,
            ai_confidence=72.0,
            ai_aligned=True,
            market_lead_aligned=True,
            lead_active=True,
            rr_ratio=2.4,
        )
        self.assertEqual(sizing.tier_key, "FLAT")

    def test_tradeable_market_allows_full_unit_for_best_setup(self) -> None:
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
        gate = build_market_trade_gate(
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
        sizing = build_signal_risk_sizing(
            market_trade_gate_snapshot=gate,
            market_catalyst_snapshot=None,
            direction="Upside",
            setup_confirm="✅ ENTER (Trend+AI)",
            confidence=88.0,
            ai_confidence=72.0,
            ai_aligned=True,
            market_lead_aligned=True,
            lead_active=True,
            rr_ratio=2.3,
        )
        self.assertEqual(sizing.tier_key, "FULL")
        self.assertAlmostEqual(sizing.unit_fraction, 1.0, places=4)

    def test_selective_market_caps_size(self) -> None:
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
        gate = build_market_trade_gate(
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
        sizing = build_signal_risk_sizing(
            market_trade_gate_snapshot=gate,
            market_catalyst_snapshot=None,
            direction="Upside",
            setup_confirm="✅ ENTER (Trend+AI)",
            confidence=88.0,
            ai_confidence=72.0,
            ai_aligned=True,
            market_lead_aligned=False,
            lead_active=True,
            rr_ratio=2.3,
        )
        self.assertEqual(sizing.tier_key, "PROBE")
        self.assertAlmostEqual(sizing.unit_fraction, 0.25, places=4)

    def test_adaptive_edge_can_trim_or_boost_size_within_gate(self) -> None:
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
        gate = build_market_trade_gate(
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
        weak = build_signal_risk_sizing(
            market_trade_gate_snapshot=gate,
            market_catalyst_snapshot=None,
            direction="Upside",
            setup_confirm="✅ ENTER (Trend+AI)",
            confidence=72.0,
            ai_confidence=65.0,
            ai_aligned=True,
            market_lead_aligned=True,
            lead_active=True,
            rr_ratio=1.8,
            adaptive_edge_score=32.0,
        )
        strong = build_signal_risk_sizing(
            market_trade_gate_snapshot=gate,
            market_catalyst_snapshot=None,
            direction="Upside",
            setup_confirm="✅ ENTER (Trend+AI)",
            confidence=72.0,
            ai_confidence=65.0,
            ai_aligned=True,
            market_lead_aligned=True,
            lead_active=True,
            rr_ratio=1.8,
            adaptive_edge_score=68.0,
        )
        self.assertLess(weak.unit_fraction, strong.unit_fraction)

    def test_watch_setup_remains_flat_even_with_lead_and_rr(self) -> None:
        regime = build_market_regime_snapshot(
            setup_quality_score=68.0,
            setup_quality_mode="Selective",
            market_lead_score=71.0,
            market_lead_state="Upside",
            lead_breadth_component=18.0,
            lead_rotation_component=9.0,
            lead_flow_component=11.0,
            lead_dominance_component=3.0,
            direction_score=44.0,
            breadth_score=55.0,
            trust_score=57.0,
        )
        gate = build_market_trade_gate(
            market_regime_snapshot=regime,
            scan_degraded=False,
            setup_quality_score=68.0,
            setup_quality_mode="Selective",
            market_lead_score=71.0,
            market_lead_state="Upside",
            direction_score=44.0,
            breadth_score=55.0,
            trust_score=57.0,
            ready_count=0,
            probe_count=2,
            watch_count=4,
            skip_count=2,
        )
        sizing = build_signal_risk_sizing(
            market_trade_gate_snapshot=gate,
            market_catalyst_snapshot=None,
            direction="Upside",
            setup_confirm="WATCH",
            confidence=72.0,
            ai_confidence=61.0,
            ai_aligned=True,
            market_lead_aligned=True,
            lead_active=True,
            rr_ratio=1.9,
        )
        self.assertEqual(sizing.tier_key, "FLAT")
        self.assertAlmostEqual(sizing.unit_fraction, 0.0, places=4)

    def test_severe_archive_cluster_forces_probe_only_size(self) -> None:
        regime = build_market_regime_snapshot(
            setup_quality_score=76.0,
            setup_quality_mode="Risk-On",
            market_lead_score=69.0,
            market_lead_state="Upside",
            lead_breadth_component=18.0,
            lead_rotation_component=20.0,
            lead_flow_component=14.0,
            lead_dominance_component=2.0,
            direction_score=61.0,
            breadth_score=67.0,
            trust_score=63.0,
        )
        gate = build_market_trade_gate(
            market_regime_snapshot=regime,
            scan_degraded=False,
            setup_quality_score=76.0,
            setup_quality_mode="Risk-On",
            market_lead_score=69.0,
            market_lead_state="Upside",
            direction_score=61.0,
            breadth_score=67.0,
            trust_score=63.0,
            ready_count=3,
            watch_count=4,
            skip_count=1,
            session_fit_score=-1.5,
            session_fit_label="Session Mixed",
            session_fit_note="The session archive is mixed rather than decisively supportive.",
            archive_guardrail_penalty=6.9,
            archive_guardrail_label="Archive Guardrail",
            archive_guardrail_note="Matched archive history is weak across current alert and timing buckets.",
        )
        sizing = build_signal_risk_sizing(
            market_trade_gate_snapshot=gate,
            market_catalyst_snapshot=None,
            direction="Upside",
            setup_confirm="✅ ENTER (Trend+AI)",
            confidence=86.0,
            ai_confidence=70.0,
            ai_aligned=True,
            market_lead_aligned=True,
            lead_active=True,
            rr_ratio=2.2,
            adaptive_edge_score=66.0,
            session_fit_score=-1.5,
            archive_guardrail_penalty=6.9,
            archive_guardrail_label="Archive Guardrail",
            archive_guardrail_note="Matched archive history is weak across current alert and timing buckets.",
        )
        self.assertEqual(sizing.tier_key, "PROBE")
        self.assertAlmostEqual(sizing.unit_fraction, 0.25, places=4)
        self.assertIn("probe-only size", sizing.note.lower())

    def test_session_fit_can_trim_or_support_size(self) -> None:
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
        gate = build_market_trade_gate(
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
        weak_session = build_signal_risk_sizing(
            market_trade_gate_snapshot=gate,
            market_catalyst_snapshot=None,
            direction="Upside",
            setup_confirm="✅ ENTER (Trend+AI)",
            confidence=72.0,
            ai_confidence=65.0,
            ai_aligned=True,
            market_lead_aligned=True,
            lead_active=True,
            rr_ratio=1.8,
            adaptive_edge_score=50.0,
            session_fit_score=-3.5,
        )
        strong_session = build_signal_risk_sizing(
            market_trade_gate_snapshot=gate,
            market_catalyst_snapshot=None,
            direction="Upside",
            setup_confirm="✅ ENTER (Trend+AI)",
            confidence=72.0,
            ai_confidence=65.0,
            ai_aligned=True,
            market_lead_aligned=True,
            lead_active=True,
            rr_ratio=1.8,
            adaptive_edge_score=50.0,
            session_fit_score=3.5,
        )
        self.assertLess(weak_session.unit_fraction, strong_session.unit_fraction)
        self.assertIn("trimming size", weak_session.note.lower())
        self.assertIn("supporting size", strong_session.note.lower())

    def test_archive_guardrail_can_trim_size(self) -> None:
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
        gate = build_market_trade_gate(
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
        baseline = build_signal_risk_sizing(
            market_trade_gate_snapshot=gate,
            market_catalyst_snapshot=None,
            direction="Upside",
            setup_confirm="✅ ENTER (Trend+AI)",
            confidence=88.0,
            ai_confidence=72.0,
            ai_aligned=True,
            market_lead_aligned=True,
            lead_active=True,
            rr_ratio=2.3,
        )
        guarded = build_signal_risk_sizing(
            market_trade_gate_snapshot=gate,
            market_catalyst_snapshot=None,
            direction="Upside",
            setup_confirm="✅ ENTER (Trend+AI)",
            confidence=88.0,
            ai_confidence=72.0,
            ai_aligned=True,
            market_lead_aligned=True,
            lead_active=True,
            rr_ratio=2.3,
            archive_guardrail_penalty=6.5,
            archive_guardrail_label="Archive Guardrail",
            archive_guardrail_note="Matched archive history is weak enough here to actively trim aggression.",
        )
        self.assertLess(guarded.unit_fraction, baseline.unit_fraction)
        self.assertIn("trim aggression", guarded.note)

    def test_targeted_catalyst_caps_matching_sector_only(self) -> None:
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
        gate = build_market_trade_gate(
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
        catalyst = build_market_catalyst_snapshot(
            [
                {
                    "title": "AI Token Unlock Wave",
                    "event_time": "2026-04-05T18:00:00Z",
                    "severity": "high",
                    "category": "unlock",
                    "scope": "sector",
                    "tag": "AI",
                }
            ],
            now="2026-04-05T10:00:00Z",
        )
        affected = build_signal_risk_sizing(
            market_trade_gate_snapshot=gate,
            market_catalyst_snapshot=catalyst,
            direction="Upside",
            setup_confirm="✅ ENTER (Trend+AI)",
            confidence=88.0,
            ai_confidence=72.0,
            ai_aligned=True,
            market_lead_aligned=True,
            lead_active=True,
            rr_ratio=2.3,
            symbol="FET",
            sector_tag="AI",
        )
        unaffected = build_signal_risk_sizing(
            market_trade_gate_snapshot=gate,
            market_catalyst_snapshot=catalyst,
            direction="Upside",
            setup_confirm="✅ ENTER (Trend+AI)",
            confidence=88.0,
            ai_confidence=72.0,
            ai_aligned=True,
            market_lead_aligned=True,
            lead_active=True,
            rr_ratio=2.3,
            symbol="ETH",
            sector_tag="L1",
        )
        self.assertLess(affected.unit_fraction, unaffected.unit_fraction)
        self.assertIn("unlock wave", affected.note.lower())


if __name__ == "__main__":
    unittest.main()
