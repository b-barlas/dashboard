import unittest

import pandas as pd

from tabs.sessions_tab import (
    DRIFT_BIAS_DEADBAND_PCT,
    SESSION_ORDER,
    _compute_session_metrics,
    _drift_bias_label,
    _format_volume_compact,
    _liquidity_label,
    _range_profile_label,
    _relative_quality_label,
)


class SessionLogicTests(unittest.TestCase):
    def test_helper_labels_are_relative_and_user_facing(self):
        self.assertEqual(_relative_quality_label(80), "Leading")
        self.assertEqual(_relative_quality_label(55), "Balanced")
        self.assertEqual(_relative_quality_label(20), "Lagging")
        self.assertEqual(_liquidity_label(0.8), "Deep")
        self.assertEqual(_liquidity_label(0.5), "Average")
        self.assertEqual(_liquidity_label(0.1), "Thin")
        self.assertEqual(_range_profile_label(0.8), "Controlled")
        self.assertEqual(_range_profile_label(0.5), "Tradable")
        self.assertEqual(_range_profile_label(0.1), "Stretched")
        self.assertEqual(_drift_bias_label(0.1), "Up Tilt")
        self.assertEqual(_drift_bias_label(-0.1), "Down Tilt")
        self.assertEqual(_drift_bias_label(0.0), "Flat")
        self.assertEqual(_drift_bias_label(DRIFT_BIAS_DEADBAND_PCT / 2), "Flat")
        self.assertEqual(_drift_bias_label(-(DRIFT_BIAS_DEADBAND_PCT / 2)), "Flat")

    def test_compact_volume_formatter(self):
        self.assertEqual(_format_volume_compact(2_500_000_000), "2.50B")
        self.assertEqual(_format_volume_compact(12_300_000), "12.30M")
        self.assertEqual(_format_volume_compact(45_600), "45.60K")

    def test_compute_session_metrics_returns_all_sessions_and_relative_fields(self):
        ts = pd.to_datetime(
            [
                "2026-03-01 01:00:00+00:00",
                "2026-03-01 02:00:00+00:00",
                "2026-03-01 09:00:00+00:00",
                "2026-03-01 10:00:00+00:00",
                "2026-03-01 17:00:00+00:00",
                "2026-03-01 18:00:00+00:00",
            ],
            utc=True,
        )
        df = pd.DataFrame(
            {
                "timestamp": ts,
                "open": [100, 101, 102, 103, 104, 105],
                "high": [102, 103, 106, 107, 108, 109],
                "low": [99, 100, 101, 102, 103, 104],
                "close": [101, 102, 105, 104, 107, 106],
                "volume": [1000, 1100, 2000, 2200, 1400, 1500],
            }
        )

        metrics = _compute_session_metrics(df)

        self.assertEqual(list(metrics.index), SESSION_ORDER)
        for column in [
            "relative_quality",
            "relative_label",
            "liquidity_label",
            "range_profile_label",
            "drift_bias_label",
        ]:
            self.assertIn(column, metrics.columns)

        self.assertTrue(((metrics["relative_quality"] >= 0) & (metrics["relative_quality"] <= 100)).all())
        self.assertEqual(int(metrics["candle_count"].sum()), 6)


if __name__ == "__main__":
    unittest.main()
