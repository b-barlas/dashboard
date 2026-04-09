import unittest

from tabs.ml_tab import _empty_matrix_row, _reference_plan_fields, _summarize_workspace_rows


class MlTabLogicTests(unittest.TestCase):
    def test_empty_matrix_row_uses_reference_columns(self) -> None:
        row = _empty_matrix_row("1h")
        self.assertEqual(row["Timeframe"], "1h")
        self.assertEqual(row["Reference Entry"], "N/A")
        self.assertEqual(row["Reference Source"], "N/A")
        self.assertEqual(row["Technical Reference Target"], "N/A")

    def test_reference_plan_fields_prefers_ai_aligned_levels(self) -> None:
        out = _reference_plan_fields("$100.00", "$110.00", "$99.00", "$108.00")
        self.assertEqual(out["Reference Entry"], "$100.00")
        self.assertEqual(out["Reference Target"], "$110.00")
        self.assertEqual(out["Reference Source"], "AI-Aligned")

    def test_reference_plan_fields_falls_back_to_technical_context(self) -> None:
        out = _reference_plan_fields("N/A", "N/A", "$99.00", "$108.00")
        self.assertEqual(out["Reference Entry"], "$99.00")
        self.assertEqual(out["Reference Target"], "$108.00")
        self.assertEqual(out["Reference Source"], "Technical Context")

    def test_workspace_summary_handles_mixed_rows(self) -> None:
        rows = [
            {"Timeframe": "1h", "DirectionKey": "LONG", "Selected Model Prob %": 72.0, "Ensemble Agree %": 66.0},
            {"Timeframe": "4h", "DirectionKey": "LONG", "Selected Model Prob %": 68.0, "Ensemble Agree %": 70.0},
            {"Timeframe": "1d", "DirectionKey": "SHORT", "Selected Model Prob %": 28.0, "Ensemble Agree %": 62.0},
            {"Timeframe": "5m", "DirectionKey": "NO_DATA", "Selected Model Prob %": 0.0, "Ensemble Agree %": 0.0},
        ]
        out = _summarize_workspace_rows(rows)
        self.assertEqual(out["dominant"], "LONG")
        self.assertAlmostEqual(float(out["avg_prob"]), 56.0, places=4)
        self.assertAlmostEqual(float(out["avg_agree"]), 66.0, places=4)
        self.assertAlmostEqual(float(out["consistency"]), 2 / 3 * 100.0, places=4)
        self.assertEqual(int(out["tf_valid_count"]), 3)


if __name__ == "__main__":
    unittest.main()
