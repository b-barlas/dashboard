from __future__ import annotations

import unittest

import pandas as pd

from tabs.position_tab import _position_hold_archive_slice, _position_hold_window_note


class PositionTabLogicTests(unittest.TestCase):
    def test_position_hold_archive_slice_normalizes_pair_symbol_to_archive_base(self) -> None:
        df = pd.DataFrame(
            [
                {"symbol": "BTC", "timeframe": "1h", "direction": "UPSIDE", "status": "RESOLVED", "signal_key": f"a{i}"}
                for i in range(8)
            ]
            + [
                {"symbol": "ETH", "timeframe": "1h", "direction": "UPSIDE", "status": "RESOLVED", "signal_key": f"b{i}"}
                for i in range(10)
            ]
        )
        out, label = _position_hold_archive_slice(
            df,
            symbol="BTC/USDT",
            timeframe="1h",
            direction="LONG",
        )
        self.assertEqual(len(out), 8)
        self.assertEqual(label, "BTC 1H Upside")

    def test_position_hold_archive_slice_prefers_symbol_timeframe_direction_match(self) -> None:
        df = pd.DataFrame(
            [
                {"symbol": "BTC", "timeframe": "1h", "direction": "UPSIDE", "status": "RESOLVED", "signal_key": f"a{i}"}
                for i in range(8)
            ]
            + [
                {"symbol": "ETH", "timeframe": "1h", "direction": "UPSIDE", "status": "RESOLVED", "signal_key": f"b{i}"}
                for i in range(10)
            ]
        )
        out, label = _position_hold_archive_slice(
            df,
            symbol="BTC",
            timeframe="1h",
            direction="LONG",
        )
        self.assertEqual(len(out), 8)
        self.assertEqual(label, "BTC 1H Upside")

    def test_position_hold_archive_slice_falls_back_to_broader_direction_archive(self) -> None:
        df = pd.DataFrame(
            [
                {"symbol": "BTC", "timeframe": "15m", "direction": "UPSIDE", "status": "RESOLVED", "signal_key": f"a{i}"}
                for i in range(3)
            ]
            + [
                {"symbol": f"ALT{i}", "timeframe": "4h", "direction": "UPSIDE", "status": "RESOLVED", "signal_key": f"b{i}"}
                for i in range(8)
            ]
        )
        out, label = _position_hold_archive_slice(
            df,
            symbol="BTC",
            timeframe="1h",
            direction="LONG",
        )
        self.assertEqual(len(out), 11)
        self.assertEqual(label, "broader Upside archive")

    def test_position_hold_window_note_describes_available_snapshot(self) -> None:
        note, tone = _position_hold_window_note(
            {
                "available": True,
                "resolved_signals": 14,
                "best_bar": 4,
                "best_label": "around 4 bars",
                "best_style": "Quick Follow-Through",
                "sample": 10,
                "avg_dir_return_pct": 1.35,
                "follow_through_pct": 60.0,
                "fade_after_bar": 8,
            },
            scope_label="BTC 1H Upside",
        )
        self.assertIn("Best at: 4 bars", note)
        self.assertIn("fades after roughly <b>8</b> bars", note)
        self.assertIn("BTC 1H Upside", note)
        self.assertEqual(tone, "positive")


if __name__ == "__main__":
    unittest.main()
