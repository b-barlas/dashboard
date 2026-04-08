from __future__ import annotations

import unittest

from core.regime_engine import build_market_regime_snapshot


class RegimeEngineTests(unittest.TestCase):
    def test_classifies_risk_off_pressure(self) -> None:
        snap = build_market_regime_snapshot(
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
        self.assertEqual(snap.regime_key, "RISK_OFF_PRESSURE")
        self.assertTrue(snap.no_trade)

    def test_classifies_alt_rotation(self) -> None:
        snap = build_market_regime_snapshot(
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
        self.assertEqual(snap.regime_key, "ALT_ROTATION")
        self.assertEqual(snap.playbook, "Selective upside rotation")

    def test_classifies_risk_on_trend(self) -> None:
        snap = build_market_regime_snapshot(
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
        self.assertEqual(snap.regime_key, "RISK_ON_TREND")
        self.assertTrue(snap.tradable)

    def test_classifies_range_chop_when_edge_is_thin(self) -> None:
        snap = build_market_regime_snapshot(
            setup_quality_score=49.0,
            setup_quality_mode="Selective",
            market_lead_score=50.0,
            market_lead_state="Balanced",
            lead_breadth_component=4.0,
            lead_rotation_component=2.0,
            lead_flow_component=1.0,
            lead_dominance_component=0.0,
            direction_score=20.0,
            breadth_score=34.0,
            trust_score=44.0,
        )
        self.assertEqual(snap.regime_key, "RANGE_CHOP")
        self.assertTrue(snap.no_trade)


if __name__ == "__main__":
    unittest.main()
