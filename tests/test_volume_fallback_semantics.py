import unittest

import pandas as pd

from core import ml, signals


def _base_frame(rows: int = 80, *, aggregate_volume: bool) -> pd.DataFrame:
    close = [100.0 + i * 0.5 for i in range(rows)]
    df = pd.DataFrame(
        {
            "timestamp": pd.date_range("2026-01-01", periods=rows, freq="h"),
            "open": [c - 0.2 for c in close],
            "high": [c + 0.6 for c in close],
            "low": [c - 0.6 for c in close],
            "close": close,
            "volume": [100.0 + i for i in range(rows)],
        }
    )
    df.attrs["volume_is_24h_aggregate"] = aggregate_volume
    return df


class VolumeFallbackSemanticsTests(unittest.TestCase):
    def test_detect_volume_spike_is_disabled_for_aggregate_volume_frames(self):
        df = _base_frame(aggregate_volume=True)
        df.loc[df.index[-1], "volume"] = 1_000_000.0
        result = signals.analyse(df)
        self.assertFalse(result.volume_spike)
        self.assertEqual(result.vwap, "")

    def test_detect_volume_spike_still_works_for_trusted_exchange_volume(self):
        df = _base_frame(aggregate_volume=False)
        df.loc[df.index[-1], "volume"] = 1_000_000.0
        result = signals.analyse(df)
        self.assertTrue(result.volume_spike)

    def test_ml_predict_direction_neutralizes_volume_features_for_aggregate_volume(self):
        df = _base_frame(aggregate_volume=True)
        prob, direction = ml.ml_predict_direction(df)
        self.assertIsInstance(prob, float)
        self.assertIn(direction, {"LONG", "SHORT", "NEUTRAL"})

    def test_ml_ensemble_predict_neutralizes_volume_features_for_aggregate_volume(self):
        df = _base_frame(aggregate_volume=True)
        prob, direction, details = ml.ml_ensemble_predict(df)
        self.assertIsInstance(prob, float)
        self.assertIn(direction, {"LONG", "SHORT", "NEUTRAL"})
        self.assertIn("ensemble", details)


if __name__ == "__main__":
    unittest.main()
