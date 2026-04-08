from __future__ import annotations

import unittest

from core.flow_proxy_engine import build_market_flow_proxy_snapshot


class FlowProxyEngineTests(unittest.TestCase):
    def test_short_crowding_detected(self) -> None:
        snap = build_market_flow_proxy_snapshot(
            [
                {"symbol": "BTCUSDT", "funding_rate": -0.0002, "oi_change_pct": 3.2, "long_short_ratio": 0.78},
                {"symbol": "ETHUSDT", "funding_rate": -0.0001, "oi_change_pct": 2.5, "long_short_ratio": 0.82},
            ]
        )
        self.assertEqual(snap.state, "SHORT_CROWDING")

    def test_long_crowding_detected(self) -> None:
        snap = build_market_flow_proxy_snapshot(
            [
                {"symbol": "BTCUSDT", "funding_rate": 0.0002, "oi_change_pct": 3.0, "long_short_ratio": 1.30},
                {"symbol": "ETHUSDT", "funding_rate": 0.00015, "oi_change_pct": 2.6, "long_short_ratio": 1.18},
            ]
        )
        self.assertEqual(snap.state, "LONG_CROWDING")

    def test_balanced_when_not_stretched(self) -> None:
        snap = build_market_flow_proxy_snapshot(
            [
                {"symbol": "BTCUSDT", "funding_rate": 0.00002, "oi_change_pct": 0.8, "long_short_ratio": 1.01},
                {"symbol": "ETHUSDT", "funding_rate": -0.00001, "oi_change_pct": 0.5, "long_short_ratio": 0.99},
            ]
        )
        self.assertEqual(snap.state, "BALANCED")


if __name__ == "__main__":
    unittest.main()
