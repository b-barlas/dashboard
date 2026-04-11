import unittest

from ui.signal_panels import normalize_indicator_label, normalize_indicator_name


class SignalPanelsTests(unittest.TestCase):
    def test_indicator_names_match_market_semantics(self):
        self.assertEqual(normalize_indicator_name("StochRSI"), "Stochastic RSI")
        self.assertEqual(normalize_indicator_name("Pattern"), "Candle Pattern")
        self.assertEqual(normalize_indicator_name("Volume"), "Spike Alert")

    def test_indicator_label_preserves_meaningful_hyphens(self):
        self.assertEqual(
            normalize_indicator_label(
                "Morning Star (3-bar bullish reversal)",
                name="Candle Pattern",
            ),
            "▲ Morning Star (3-bar bullish reversal)",
        )
        self.assertEqual(
            normalize_indicator_label(
                "Spinning Top (neutral indecision)",
                name="Candle Pattern",
            ),
            "- Spinning Top (neutral indecision)",
        )

