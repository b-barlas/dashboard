import unittest

from tabs.fibonacci_tab import _distance_points, _divergence_impact, _poc_points, _trend_bias_row


class FibonacciTabLogicTests(unittest.TestCase):
    def test_distance_points_reward_closer_levels(self) -> None:
        self.assertGreater(_distance_points(0.5), _distance_points(3.5))

    def test_poc_points_reward_nearby_poc(self) -> None:
        self.assertGreater(_poc_points(0.8), _poc_points(4.5))

    def test_divergence_impact_is_direction_aware(self) -> None:
        rows = [
            {"type": "BULLISH RSI", "strength": "STRONG"},
            {"type": "BEARISH MACD", "strength": "MODERATE"},
        ]
        uptrend_penalty, uptrend_conflict, uptrend_support = _divergence_impact(rows, is_uptrend=True)
        downtrend_penalty, downtrend_conflict, downtrend_support = _divergence_impact(rows, is_uptrend=False)
        self.assertEqual((uptrend_conflict, uptrend_support), (1, 1))
        self.assertEqual((downtrend_conflict, downtrend_support), (1, 1))
        self.assertGreaterEqual(uptrend_penalty, 0)
        self.assertGreaterEqual(downtrend_penalty, 0)

    def test_trend_bias_row_marks_bearish_context_as_supportive(self) -> None:
        row = _trend_bias_row(False)
        self.assertEqual(row["Value"], "▼ Bearish")
        self.assertEqual(row["Status"], "● Supportive")


if __name__ == "__main__":
    unittest.main()
