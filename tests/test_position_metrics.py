from __future__ import annotations

import unittest

from core.position_metrics import (
    compute_hard_invalidation,
    compute_health_decision,
    compute_position_pnl,
    estimate_liquidation,
)


class PositionMetricsTests(unittest.TestCase):
    def test_compute_position_pnl_long(self) -> None:
        out = compute_position_pnl(
            entry_price=100.0,
            current_price=105.0,
            direction="LONG",
            leverage=10.0,
            margin_used=1000.0,
            funding_impact_pct=0.10,
        )
        self.assertAlmostEqual(out["raw_pct"], 5.0, places=6)
        self.assertAlmostEqual(out["levered_pct"], 50.0, places=6)
        self.assertAlmostEqual(out["notional"], 10000.0, places=6)
        self.assertAlmostEqual(out["gross_usd"], 500.0, places=6)
        self.assertAlmostEqual(out["funding_usd"], 10.0, places=6)
        self.assertAlmostEqual(out["net_usd"], 510.0, places=6)

    def test_estimate_liquidation_short(self) -> None:
        out = estimate_liquidation(entry_price=100.0, current_price=98.0, direction="SHORT", leverage=10.0)
        self.assertIsNotNone(out["liq_price"])
        self.assertAlmostEqual(float(out["liq_price"]), 110.0, places=6)
        self.assertGreater(float(out["distance_pct"]), 0.0)

    def test_hard_invalidation_flags_break(self) -> None:
        out = compute_hard_invalidation(
            direction="LONG",
            support=95.0,
            resistance=110.0,
            atr14=2.0,
            buffer_mult=0.5,
            current_price=93.5,
        )
        self.assertTrue(out["invalidated"])
        self.assertAlmostEqual(float(out["level"]), 94.0, places=6)

    def test_health_decision_exit_on_invalidated(self) -> None:
        out = compute_health_decision(
            direction="LONG",
            signal_direction="SHORT",
            confidence=42.0,
            conviction_label="CONFLICT",
            liq_distance_pct=3.0,
            invalidated=True,
            levered_pnl_pct=-12.0,
        )
        self.assertEqual(out["label"], "EXIT")
        self.assertLess(int(out["score"]), 40)


if __name__ == "__main__":
    unittest.main()
