import unittest

from ui.helpers import (
    confidence_score_badge,
    format_adx,
    format_delta,
    format_stochrsi,
    format_trend,
    leverage_badge,
    readable_market_cap,
    signal_badge,
    signal_plain,
    style_confidence,
    style_delta,
    style_scalp_opp,
    style_signal,
)


class UiHelpersTests(unittest.TestCase):
    def test_signal_mappings(self):
        self.assertEqual(signal_badge("BUY"), "🟢 LONG")
        self.assertEqual(signal_badge("SELL"), "🔴 SHORT")
        self.assertEqual(signal_plain("STRONG BUY"), "LONG")
        self.assertEqual(signal_plain("STRONG SELL"), "SHORT")
        self.assertEqual(signal_plain("WAIT"), "WAIT")

    def test_formatters(self):
        self.assertEqual(leverage_badge(5), "x5")
        self.assertIn("BUY", confidence_score_badge(72.4))
        self.assertTrue(format_delta(1.23).startswith("▲"))
        self.assertTrue(format_delta(-0.5).startswith("▼"))
        self.assertEqual(format_trend("Bullish"), "▲ Bullish")
        self.assertIn("Strong", format_adx(35.0))
        self.assertEqual(format_stochrsi(0.1), "🟢 Low")

    def test_styles_and_market_cap(self):
        self.assertIn("color", style_delta("▲ 1.00%"))
        self.assertIn("font-weight", style_signal("LONG"))
        self.assertIn("font-weight", style_confidence("80 (STRONG BUY)"))
        self.assertIn("font-weight", style_scalp_opp("SHORT"))
        self.assertEqual(readable_market_cap(1_500_000_000), "1.50B")


if __name__ == "__main__":
    unittest.main()
