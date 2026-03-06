import unittest

from core.multitf import compute_multitf_alignment


class MultiTFLogicTests(unittest.TestCase):
    def test_neutral_weights_dilute_weighted_alignment(self):
        rows = [
            {"timeframe": "1d", "direction": "UPSIDE", "strength": 70, "weight": 2.6},
            {"timeframe": "4h", "direction": "NEUTRAL", "strength": 40, "weight": 2.1},
        ]
        metrics = compute_multitf_alignment(rows)
        self.assertAlmostEqual(metrics["weighted_alignment_pct"], 2.6 / (2.6 + 2.1) * 100.0, places=4)
        self.assertEqual(metrics["dominant_bias"], "NEUTRAL")

    def test_higher_timeframe_bias_uses_only_structure_rows(self):
        rows = [
            {"timeframe": "5m", "direction": "UPSIDE", "strength": 60, "weight": 1.0},
            {"timeframe": "15m", "direction": "UPSIDE", "strength": 62, "weight": 1.2},
            {"timeframe": "1h", "direction": "DOWNSIDE", "strength": 58, "weight": 1.6},
            {"timeframe": "4h", "direction": "DOWNSIDE", "strength": 65, "weight": 2.1},
            {"timeframe": "1d", "direction": "DOWNSIDE", "strength": 67, "weight": 2.6},
        ]
        metrics = compute_multitf_alignment(rows)
        self.assertEqual(metrics["dominant_bias"], "DOWNSIDE")
        self.assertEqual(metrics["higher_tf_bias"], "DOWNSIDE")
        self.assertEqual(metrics["tactical_bias"], "UPSIDE")

    def test_all_neutral_rows_return_neutral_bias(self):
        rows = [
            {"timeframe": "5m", "direction": "NEUTRAL", "strength": 22, "weight": 1.0},
            {"timeframe": "15m", "direction": "NEUTRAL", "strength": 25, "weight": 1.2},
            {"timeframe": "1h", "direction": "NEUTRAL", "strength": 30, "weight": 1.6},
        ]
        metrics = compute_multitf_alignment(rows)
        self.assertEqual(metrics["dominant_bias"], "NEUTRAL")
        self.assertEqual(metrics["higher_tf_bias"], "NEUTRAL")
        self.assertEqual(metrics["tactical_bias"], "NEUTRAL")
        self.assertEqual(metrics["weighted_alignment_pct"], 0.0)
        self.assertEqual(metrics["raw_alignment_pct"], 0.0)

    def test_equal_upside_and_downside_weights_return_neutral_bias(self):
        rows = [
            {"timeframe": "5m", "direction": "UPSIDE", "strength": 60, "weight": 1.0},
            {"timeframe": "15m", "direction": "DOWNSIDE", "strength": 60, "weight": 1.0},
        ]
        metrics = compute_multitf_alignment(rows)
        self.assertEqual(metrics["dominant_bias"], "NEUTRAL")
        self.assertEqual(metrics["weighted_alignment_pct"], 50.0)


if __name__ == "__main__":
    unittest.main()
