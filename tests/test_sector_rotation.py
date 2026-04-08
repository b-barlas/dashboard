from __future__ import annotations

import unittest

from core.sector_rotation import build_sector_rotation_snapshot, classify_symbol_sector


class SectorRotationTests(unittest.TestCase):
    def test_classifies_known_symbols(self) -> None:
        self.assertEqual(classify_symbol_sector("DOGE"), "Meme")
        self.assertEqual(classify_symbol_sector("UNI"), "DeFi")
        self.assertEqual(classify_symbol_sector("FET"), "AI")

    def test_detects_upside_sector_rotation(self) -> None:
        rows = [
            {"Coin": "DOGE", "Direction": "Upside", "__confidence_val": 82.0, "__emerging_label": "Emerging Upside"},
            {"Coin": "SHIB", "Direction": "Upside", "__confidence_val": 76.0, "__emerging_label": "Emerging Upside"},
            {"Coin": "PEPE", "Direction": "Upside", "__confidence_val": 68.0, "__emerging_label": ""},
            {"Coin": "BTC", "Direction": "Neutral", "__confidence_val": 35.0, "__emerging_label": ""},
        ]
        snap = build_sector_rotation_snapshot(rows)
        self.assertEqual(snap.state, "UPSIDE")
        self.assertEqual(snap.leader_sector, "Meme")

    def test_detects_downside_sector_pressure(self) -> None:
        rows = [
            {"Coin": "UNI", "Direction": "Downside", "__confidence_val": 81.0, "__emerging_label": "Emerging Downside"},
            {"Coin": "AAVE", "Direction": "Downside", "__confidence_val": 74.0, "__emerging_label": "Emerging Downside"},
            {"Coin": "MKR", "Direction": "Downside", "__confidence_val": 66.0, "__emerging_label": ""},
        ]
        snap = build_sector_rotation_snapshot(rows)
        self.assertEqual(snap.state, "DOWNSIDE")
        self.assertEqual(snap.leader_sector, "DeFi")

    def test_returns_none_when_no_sector_clusters(self) -> None:
        rows = [
            {"Coin": "BTC", "Direction": "Neutral", "__confidence_val": 30.0, "__emerging_label": ""},
            {"Coin": "ETH", "Direction": "Neutral", "__confidence_val": 34.0, "__emerging_label": ""},
        ]
        snap = build_sector_rotation_snapshot(rows)
        self.assertEqual(snap.state, "NONE")


if __name__ == "__main__":
    unittest.main()
