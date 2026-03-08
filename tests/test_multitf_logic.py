import unittest

from core.multitf import HIGHER_TFS, compute_multitf_alignment, summarize_scope_bias


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

    def test_scope_summary_excludes_no_data_slots(self):
        rows = [
            {"timeframe": "1h", "direction": "UPSIDE", "strength": 58, "weight": 1.6},
            {"timeframe": "4h", "direction": "", "strength": 0, "weight": 2.1},
            {"timeframe": "1d", "direction": "", "strength": 0, "weight": 2.6},
        ]
        summary = summarize_scope_bias(rows, HIGHER_TFS, "higher timeframes", "UPSIDE")
        self.assertIn("1 directional", summary)
        self.assertIn("2 unavailable", summary)
        self.assertNotIn("All 3", summary)

    def test_scope_summary_explains_weight_neutral_case(self):
        rows = [
            {"timeframe": "1h", "direction": "DOWNSIDE", "strength": 58, "weight": 1.6},
            {"timeframe": "4h", "direction": "NEUTRAL", "strength": 40, "weight": 2.1},
            {"timeframe": "1d", "direction": "UPSIDE", "strength": 67, "weight": 2.6},
        ]
        summary = summarize_scope_bias(rows, HIGHER_TFS, "higher timeframes", "NEUTRAL")
        self.assertIn("neutral", summary.lower())
        self.assertIn("largest directional share", summary.lower())
        self.assertIn("directional", summary.lower())


if __name__ == "__main__":
    unittest.main()
