import unittest

import pandas as pd

from tabs.risk_tab import (
    _closed_close_series,
    _closed_timestamp_series,
    _metric_status,
    _risk_profile_summary,
    _tf_thresholds,
)


class RiskTabLogicTests(unittest.TestCase):
    def test_closed_close_series_drops_invalid_values_and_open_candle(self) -> None:
        frame = pd.DataFrame(
            {
                "close": [100.0, 101.0, float("nan"), 103.0, 104.0],
                "timestamp": pd.date_range("2026-01-01", periods=5, freq="h", tz="UTC"),
            }
        )
        out = _closed_close_series(frame)
        self.assertEqual(out.tolist(), [100.0, 101.0, 103.0])

    def test_closed_timestamp_series_follows_filtered_close_index(self) -> None:
        frame = pd.DataFrame(
            {
                "close": [100.0, 101.0, 102.0, 103.0],
                "timestamp": pd.date_range("2026-01-01", periods=4, freq="h", tz="UTC"),
            }
        )
        close_series = _closed_close_series(frame)
        out = _closed_timestamp_series(frame, close_series)
        self.assertEqual(len(out), len(close_series))
        self.assertEqual(str(out.iloc[-1]), "2026-01-01 02:00:00+00:00")

    def test_tf_thresholds_change_by_timeframe(self) -> None:
        self.assertGreater(_tf_thresholds("1h")["dd_good"], _tf_thresholds("1d")["dd_good"])
        self.assertLess(_tf_thresholds("1h")["var_good"], _tf_thresholds("1d")["var_good"])

    def test_metric_status_supports_lower_better_metrics(self) -> None:
        out = _metric_status(2.0, good=3.0, neutral=5.0, lower_better=True, positive="g", warning="w", negative="n")
        self.assertEqual(out, ("Healthy", "g"))

    def test_risk_profile_summary_returns_defensive_on_high_risk_tail(self) -> None:
        text, color = _risk_profile_summary(
            regime="High Risk",
            sharpe=0.2,
            cvar95=-9.0,
            positive="p",
            warning="w",
            negative="n",
        )
        self.assertIn("Defensive profile", text)
        self.assertEqual(color, "n")


if __name__ == "__main__":
    unittest.main()
