import pathlib
import re
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[1]
MARKET_TAB_PATH = ROOT / "tabs" / "market_tab.py"


class MarketCoinColumnContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.text = MARKET_TAB_PATH.read_text(encoding="utf-8")

    def test_coin_cell_uses_pair_tooltip_metadata(self):
        self.assertIn("pair = str(row.get(\"__pair\", \"\")).strip()", self.text)
        self.assertIn("'__pair': sym", self.text)
        self.assertRegex(
            self.text,
            r"(?s)hidden_meta_cols\s*=\s*\[.*?\"__pair\"",
        )

    def test_stablecoin_filter_is_part_of_scan_signature_and_universe_filter(self):
        self.assertIn("scan_sig = (timeframe, direction_filter, int(top_n), bool(exclude_stables))", self.text)
        self.assertIn("if exclude_stables:", self.text)
        self.assertIn(
            "working_symbols = [s for s in working_symbols if not _is_stable_base(s.split(\"/\")[0].upper())]",
            self.text,
        )

    def test_sorting_has_deterministic_tie_breakers(self):
        self.assertIn('-float(x.get("__mcap_val", 0))', self.text)
        self.assertIn('str(x.get("Coin", ""))', self.text)

    def test_csv_export_drops_internal_coin_metadata(self):
        self.assertIn(
            'csv_df = csv_df[[c for c in csv_df.columns if not str(c).startswith("__")]]',
            self.text,
        )


if __name__ == "__main__":
    unittest.main()
