import unittest

from ui.helpers import (
    bias_score_badge,
    format_adx,
    format_delta,
    format_stochrsi,
    format_trend,
    leverage_badge,
    readable_market_cap,
    sanitize_trading_terms,
    signal_badge,
    signal_plain,
    style_delta,
    style_scalp_opp,
    style_signal,
)


class UiHelpersTests(unittest.TestCase):
    def test_signal_mappings(self):
        self.assertEqual(signal_badge("BUY"), "🟢 Upside")
        self.assertEqual(signal_badge("SELL"), "🔴 Downside")
        self.assertEqual(signal_plain("STRONG BUY"), "LONG")
        self.assertEqual(signal_plain("STRONG SELL"), "SHORT")
        self.assertEqual(signal_plain("WAIT"), "WAIT")

    def test_formatters(self):
        self.assertEqual(leverage_badge(5), "x5")
        self.assertIn("Bullish", bias_score_badge(72.4))
        self.assertTrue(format_delta(1.23).startswith("▲"))
        self.assertTrue(format_delta(-0.5).startswith("▼"))
        self.assertEqual(format_trend("Bullish"), "▲ Bullish")
        self.assertEqual(format_trend("Neutral"), "→ Neutral")
        self.assertEqual(format_trend("Unavailable"), "")
        self.assertIn("Strong", format_adx(35.0))
        self.assertIn("Weak", format_adx(19.9))
        self.assertIn("Starting", format_adx(20.0))
        self.assertIn("Starting", format_adx(24.9))
        self.assertIn("Strong", format_adx(25.0))
        self.assertIn("Very Strong", format_adx(50.0))
        self.assertIn("Extreme", format_adx(75.0))
        self.assertEqual(format_stochrsi(0.1), "🟢 Low")
        self.assertEqual(format_stochrsi(0.79, timeframe="4h"), "🔴 High")
        self.assertEqual(format_stochrsi(0.21, timeframe="4h"), "🟢 Low")
        self.assertEqual(format_stochrsi(0.84, timeframe="1m"), "→ Neutral")

    def test_styles_and_market_cap(self):
        self.assertIn("color", style_delta("▲ 1.00%"))
        self.assertIn("font-weight", style_signal("Upside"))
        self.assertIn("font-weight", style_scalp_opp("Downside"))
        self.assertEqual(readable_market_cap(1_500_000_000), "1.50B")

    def test_sanitize_trading_terms(self):
        raw = "STRONG BUY with LONG bias, otherwise SELL / SHORT fallback"
        clean = sanitize_trading_terms(raw)
        self.assertNotIn("BUY", clean.upper())
        self.assertNotIn("SELL", clean.upper())
        self.assertNotIn("LONG", clean.upper())
        self.assertNotIn("SHORT", clean.upper())
        self.assertIn("BULLISH", clean.upper())
        self.assertIn("DOWNSIDE", clean.upper())


if __name__ == "__main__":
    unittest.main()
