import unittest

import pandas as pd

from core.archive_expected_path import build_archive_expected_path_projection, with_expected_path_reference_price


class ArchiveExpectedPathTests(unittest.TestCase):
    def test_expected_path_tracks_alternate_and_caution_without_blocking_primary(self) -> None:
        events = []
        windows = []
        for idx in range(8):
            up_key = f"up-{idx}"
            down_key = f"down-{idx}"
            events.extend(
                [
                    {
                        "signal_key": up_key,
                        "symbol": "TRX",
                        "timeframe": "1h",
                        "direction": "Upside",
                        "status": "RESOLVED",
                        "price": 0.32,
                        "event_time": f"2026-04-28T0{idx}:00:00Z",
                    },
                    {
                        "signal_key": down_key,
                        "symbol": "TRX",
                        "timeframe": "1h",
                        "direction": "Downside",
                        "status": "RESOLVED",
                        "price": 0.32,
                        "event_time": f"2026-04-28T0{idx}:30:00Z",
                    },
                ]
            )
            windows.extend(
                [
                    {
                        "signal_key": up_key,
                        "bars_ahead": 4,
                        "directional_return_pct": 0.9,
                        "adverse_excursion_pct": 0.2 if idx < 6 else 0.7,
                    },
                    {
                        "signal_key": down_key,
                        "bars_ahead": 4,
                        "directional_return_pct": 0.9,
                        "adverse_excursion_pct": 0.2 if idx < 6 else 0.7,
                    },
                ]
            )

        snapshot = build_archive_expected_path_projection(
            df_events=pd.DataFrame(events),
            df_forward_windows=pd.DataFrame(windows),
            symbol_filter="TRX",
            timeframe_filter="1h",
            min_samples=8,
            now="2026-04-28T12:00:00Z",
        )
        priced = with_expected_path_reference_price(snapshot, 0.32, "latest close")

        self.assertTrue(priced["available"])
        self.assertIn(priced["direction"], {"UPSIDE", "DOWNSIDE"})
        self.assertTrue(priced["alternate_path"])
        self.assertTrue(priced["path_conflict"])
        self.assertGreaterEqual(priced["caution_pullback_pct"], priced["normal_pullback_pct"])
        self.assertGreater(priced["caution_price"], 0.0)


if __name__ == "__main__":
    unittest.main()
