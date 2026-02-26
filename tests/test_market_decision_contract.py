import unittest

from core.market_decision import action_decision, setup_badge, trade_quality


class MarketDecisionContractTests(unittest.TestCase):
    def test_setup_badge_aligned(self):
        self.assertEqual(setup_badge("LONG", "LONG", "LONG"), "🟢 Aligned")

    def test_setup_badge_tech_only(self):
        self.assertEqual(setup_badge("SHORT", "SHORT", "NEUTRAL"), "🟡 Tech-Only")

    def test_action_requires_valid_direction(self):
        out = action_decision("NEUTRAL", 70, "🟢 Aligned", "HIGH", 0.8, 30.0, 2.0, True)
        self.assertEqual(out, "⛔ SKIP")

    def test_action_enter_independent_from_rr_and_plan(self):
        out = action_decision("LONG", 70, "🟢 Aligned", "HIGH", 0.8, 30.0, 0.2, False)
        self.assertEqual(out, "✅ ENTER")

    def test_action_skip_on_conflict(self):
        out = action_decision("LONG", 70, "🟢 Aligned", "CONFLICT", 0.8, 30.0, 1.8, True)
        self.assertEqual(out, "⛔ SKIP")

    def test_action_skip_on_weak_strength(self):
        out = action_decision("SHORT", 34, "🟡 Tech-Only", "MEDIUM", 0.75, 29.0, 2.2, True)
        self.assertEqual(out, "⛔ SKIP")

    def test_action_enter_on_strict_gate(self):
        out = action_decision("SHORT", 62, "🟢 Aligned", "HIGH", 0.7, 26.0, 1.6, True)
        self.assertEqual(out, "✅ ENTER")

    def test_action_enter_on_trend_momentum_path(self):
        out = action_decision("LONG", 56, "⚪ Draft", "LOW", 0.52, 28.0, 1.1, False)
        self.assertEqual(out, "✅ ENTER")

    def test_action_enter_tech_only_when_trend_and_strength_are_exceptional(self):
        out = action_decision("LONG", 74, "🟡 Tech-Only", "TECH-ONLY", 0.0, 27.0, 1.2, False)
        self.assertEqual(out, "✅ ENTER")

    def test_action_watch_when_adx_unknown(self):
        out = action_decision("LONG", 74, "🟢 Aligned", "HIGH", 0.8, float("nan"), 1.8, True)
        self.assertEqual(out, "👀 WATCH")

    def test_trade_quality_a_grade(self):
        out = trade_quality("✅ ENTER", "🟢 Aligned", "HIGH", 68, 0.8, 1.7)
        self.assertEqual(out, "🟢 A")

    def test_trade_quality_b_grade(self):
        out = trade_quality("👀 WATCH", "🟡 Tech-Only", "MEDIUM", 50, 0.6, 1.35)
        self.assertEqual(out, "🟡 B")

    def test_trade_quality_c_grade(self):
        out = trade_quality("👀 WATCH", "🔴 No Setup", "LOW", 42, 0.4, 1.1)
        self.assertEqual(out, "🔴 C")


if __name__ == "__main__":
    unittest.main()
