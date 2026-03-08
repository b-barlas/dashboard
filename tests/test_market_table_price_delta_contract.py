import pathlib
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[1]
MARKET_TAB_PATH = ROOT / "tabs" / "market_tab.py"


class MarketTablePriceDeltaContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.text = MARKET_TAB_PATH.read_text(encoding="utf-8")

    def test_price_column_uses_latest_candle_close(self):
        self.assertRegex(
            self.text,
            r"price\s*=\s*float\(latest_closed\[['\"]close['\"]\]\)",
            msg="Price ($) must come from latest candle close.",
        )
        self.assertIn(
            "'Price ($)': _fmt_price(price)",
            self.text,
            msg="Price ($) table cell must render the computed price.",
        )

    def test_delta_uses_selected_timeframe_closed_candles(self):
        self.assertRegex(
            self.text,
            r"prev_close\s*=\s*float\(df_eval\[['\"]close['\"]\]\.iloc\[-2\]\)",
            msg="Delta must reference previous closed candle close on selected timeframe.",
        )
        self.assertRegex(
            self.text,
            r"last_closed\s*=\s*float\(df_eval\[['\"]close['\"]\]\.iloc\[-1\]\)",
            msg="Delta must reference latest closed candle close on selected timeframe.",
        )
        self.assertRegex(
            self.text,
            r"price_change\s*=\s*\(\(last_closed\s*/\s*prev_close\)\s*-\s*1\.0\)\s*\*\s*100\.0",
            msg="Delta formula must be candle-over-candle percentage change.",
        )
        self.assertIn(
            "'Δ (%)': format_delta(price_change) if price_change is not None else ''",
            self.text,
            msg="Delta table cell must format computed price_change.",
        )

    def test_delta_has_explicit_fallback(self):
        self.assertIn(
            "if price_change is None:",
            self.text,
            msg="Delta flow must keep an explicit fallback guard.",
        )
        self.assertIn(
            "price_change = _fetch_ticker_delta_once(",
            self.text,
            msg="Delta fallback must use ticker percentage only when candle delta is unavailable.",
        )

    def test_ui_copy_matches_price_delta_semantics(self):
        self.assertIn(
            "Price ($) shows the latest candle close.",
            self.text,
            msg="Price caption must state latest candle close.",
        )
        self.assertIn(
            "change from previous closed candle to latest closed candle on selected timeframe",
            self.text,
            msg="Column guide must explain selected-timeframe closed-candle delta semantics.",
        )


if __name__ == "__main__":
    unittest.main()
