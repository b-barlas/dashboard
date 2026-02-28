import unittest

from core.market_decision import (
    action_decision,
    action_decision_with_reason,
    action_reason,
    structure_state,
)


class MarketDecisionContractTests(unittest.TestCase):
    def test_structure_state_full(self):
        self.assertEqual(structure_state("LONG", "LONG", 68, 0.8), "FULL")

    def test_structure_state_trend(self):
        self.assertEqual(structure_state("SHORT", "NEUTRAL", 72, 0.0), "TREND")

    def test_structure_state_early(self):
        self.assertEqual(structure_state("LONG", "NEUTRAL", 56, 0.0), "EARLY")

    def test_action_requires_valid_direction(self):
        out = action_decision("NEUTRAL", 70, "FULL", "HIGH", 0.8, 30.0)
        self.assertEqual(out, "⛔ SKIP")

    def test_action_enter_from_core_quality_inputs(self):
        out = action_decision("LONG", 70, "FULL", "HIGH", 0.8, 30.0)
        self.assertEqual(out, "✅ ENTER (Trend+AI)")

    def test_action_skip_on_conflict(self):
        out = action_decision("LONG", 70, "FULL", "CONFLICT", 0.8, 30.0)
        self.assertEqual(out, "⛔ SKIP")

    def test_action_skip_on_weak_strength(self):
        out = action_decision("SHORT", 34, "TREND", "MEDIUM", 0.75, 29.0)
        self.assertEqual(out, "⛔ SKIP")

    def test_action_enter_on_strict_gate(self):
        out = action_decision("SHORT", 62, "FULL", "HIGH", 0.7, 26.0)
        self.assertEqual(out, "✅ ENTER (Trend+AI)")

    def test_action_enter_on_trend_led_path(self):
        out = action_decision("LONG", 58, "TREND", "WEAK", 0.40, 23.0)
        self.assertEqual(out, "🟡 ENTER (Trend-Led)")

    def test_action_enter_trend_led_when_trend_is_exceptional(self):
        out = action_decision("LONG", 74, "TREND", "TREND", 0.0, 27.0)
        self.assertEqual(out, "🟡 ENTER (Trend-Led)")

    def test_action_enter_trend_led_when_structure_is_early(self):
        out = action_decision("LONG", 58, "EARLY", "WEAK", 0.20, 22.0)
        self.assertEqual(out, "🟡 ENTER (Trend-Led)")

    def test_action_enter_ai_led_when_agreement_is_exceptional(self):
        out = action_decision("LONG", 46, "EARLY", "WEAK", 0.85, 20.0)
        self.assertEqual(out, "🟡 ENTER (AI-Led)")

    def test_action_prefers_ai_led_when_agreement_is_exceptional(self):
        out = action_decision("LONG", 58, "EARLY", "WEAK", 0.85, 22.0)
        self.assertEqual(out, "🟡 ENTER (AI-Led)")

    def test_action_watch_when_adx_unknown(self):
        out = action_decision("LONG", 74, "FULL", "HIGH", 0.8, float("nan"))
        self.assertEqual(out, "WATCH")

    def test_action_watch_when_adx_is_weak_even_if_quality_is_high(self):
        out = action_decision("LONG", 74, "FULL", "HIGH", 0.8, 19.9)
        self.assertEqual(out, "WATCH")

    def test_action_reason_conflict(self):
        out = action_reason("LONG", 70, "FULL", "CONFLICT", 0.8, 30.0)
        self.assertEqual(out, "TECH_AI_CONFLICT")

    def test_action_reason_enter_trend_ai(self):
        out = action_reason("LONG", 70, "FULL", "HIGH", 0.8, 30.0)
        self.assertEqual(out, "ENTER_TREND_AI")

    def test_action_decision_with_reason_contract(self):
        action, reason = action_decision_with_reason("LONG", 70, "FULL", "HIGH", 0.8, 30.0)
        self.assertEqual(action, "✅ ENTER (Trend+AI)")
        self.assertEqual(reason, "ENTER_TREND_AI")


if __name__ == "__main__":
    unittest.main()
