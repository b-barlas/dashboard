import unittest

import pandas as pd

from core.archive_intelligence import (
    archive_policy_for_signal,
    build_archive_intelligence_snapshot,
    build_archive_policy_map,
    filter_archive_events_by_setup,
)


class ArchiveIntelligenceTests(unittest.TestCase):
    def test_auto_best_excludes_skip_and_selects_best_pocket(self) -> None:
        rows = []
        for idx in range(10):
            rows.append(
                {
                    "symbol": "TRX",
                    "timeframe": "1h",
                    "direction": "Upside",
                    "status": "RESOLVED",
                    "setup_confirm": "WATCH",
                    "directional_return_pct": 0.4,
                    "adverse_excursion_pct": 0.1,
                }
            )
        for idx in range(20):
            rows.append(
                {
                    "symbol": "TRX",
                    "timeframe": "1h",
                    "direction": "Upside",
                    "status": "RESOLVED",
                    "setup_confirm": "SKIP",
                    "directional_return_pct": 2.0,
                    "adverse_excursion_pct": 0.1,
                }
            )

        snapshot = build_archive_intelligence_snapshot(pd.DataFrame(rows), setup_filter_value="AUTO_BEST")

        self.assertTrue(snapshot.available)
        self.assertEqual(snapshot.symbol, "TRX")
        self.assertEqual(snapshot.setup_class, "WATCH")
        self.assertEqual(snapshot.direction, "UPSIDE")
        self.assertEqual(snapshot.completed, 10)
        self.assertGreater(snapshot.policy_delta, 0.0)

        filtered = filter_archive_events_by_setup(pd.DataFrame(rows), "AUTO_BEST")
        self.assertEqual(set(filtered["__setup_class"].unique()), {"WATCH"})

    def test_archive_policy_map_returns_exact_signal_policy(self) -> None:
        rows = []
        for idx in range(12):
            rows.append(
                {
                    "symbol": "RAVE",
                    "timeframe": "15m",
                    "direction": "Downside",
                    "status": "RESOLVED",
                    "setup_confirm": "ENTER_TREND_AI",
                    "directional_return_pct": 0.7,
                    "adverse_excursion_pct": 0.2,
                }
            )

        policy_map = build_archive_policy_map(pd.DataFrame(rows), min_completed=8)
        policy = archive_policy_for_signal(
            policy_map,
            symbol="RAVE",
            timeframe="15m",
            setup_confirm="ENTER_TREND_AI",
            direction="Downside",
        )
        miss = archive_policy_for_signal(
            policy_map,
            symbol="RAVE",
            timeframe="1h",
            setup_confirm="ENTER_TREND_AI",
            direction="Downside",
        )

        self.assertTrue(policy.available)
        self.assertEqual(policy.completed, 12)
        self.assertGreater(policy.policy_delta, 0.0)
        self.assertGreater(policy.coverage_factor, 0.0)
        self.assertLess(policy.coverage_factor, 1.0)
        self.assertFalse(miss.available)
        self.assertEqual(miss.coverage_factor, 0.0)

    def test_archive_policy_coverage_reaches_full_strength_with_deeper_history(self) -> None:
        rows = []
        for idx in range(36):
            rows.append(
                {
                    "symbol": "SOL",
                    "timeframe": "1h",
                    "direction": "Upside",
                    "status": "RESOLVED",
                    "setup_confirm": "WATCH",
                    "directional_return_pct": 0.5,
                    "adverse_excursion_pct": 0.1,
                }
            )

        policy_map = build_archive_policy_map(pd.DataFrame(rows), min_completed=8)
        policy = archive_policy_for_signal(
            policy_map,
            symbol="SOL",
            timeframe="1h",
            setup_confirm="WATCH",
            direction="Upside",
        )

        self.assertTrue(policy.available)
        self.assertEqual(policy.quality_label, "Strong")
        self.assertEqual(policy.coverage_factor, 1.0)

    def test_archive_snapshot_requires_completed_outcome_rows(self) -> None:
        rows = []
        for idx in range(12):
            rows.append(
                {
                    "symbol": "TRX",
                    "timeframe": "1h",
                    "direction": "Upside",
                    "status": "OPEN",
                    "setup_confirm": "WATCH",
                    "directional_return_pct": None,
                    "adverse_excursion_pct": None,
                }
            )
        for idx in range(7):
            rows.append(
                {
                    "symbol": "TRX",
                    "timeframe": "1h",
                    "direction": "Upside",
                    "status": "RESOLVED",
                    "setup_confirm": "WATCH",
                    "directional_return_pct": 0.4,
                    "adverse_excursion_pct": 0.1,
                }
            )

        snapshot = build_archive_intelligence_snapshot(pd.DataFrame(rows), setup_filter_value="AUTO_BEST")

        self.assertFalse(snapshot.available)


if __name__ == "__main__":
    unittest.main()
