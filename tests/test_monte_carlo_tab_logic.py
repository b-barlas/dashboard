import unittest

from tabs.monte_carlo_tab import _band_from_floor, _history_limit, _mc_decision_profile, _mc_thresholds


class MonteCarloTabLogicTests(unittest.TestCase):
    def test_mc_thresholds_widen_with_horizon(self) -> None:
        short = _mc_thresholds(30)
        long = _mc_thresholds(120)
        self.assertLess(long["prob_healthy"], short["prob_healthy"])
        self.assertLess(long["var_healthy"], short["var_healthy"])

    def test_band_from_floor_marks_lower_value_risky(self) -> None:
        out = _band_from_floor(-12.0, -8.0, -15.0, positive_color="p", warning_color="w", negative_color="n")
        self.assertEqual(out, ("Watch", "w"))

    def test_history_limit_scales_with_horizon(self) -> None:
        self.assertGreater(_history_limit(800), _history_limit(30))

    def test_decision_profile_turns_defensive_on_heavy_tail_risk(self) -> None:
        tone, color, note = _mc_decision_profile(
            [("Watch", "w"), ("Watch", "w"), ("Risky", "n"), ("Risky", "n"), ("Watch", "w")],
            positive_color="p",
            warning_color="w",
            negative_color="n",
        )
        self.assertEqual((tone, color), ("Defensive", "n"))
        self.assertIn("Tail-risk profile", note)


if __name__ == "__main__":
    unittest.main()
