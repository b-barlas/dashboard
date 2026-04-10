import unittest

import numpy as np
import pandas as pd

from core.spot_direction import (
    TimeframeDirectionSnapshot,
    analyze_timeframe_direction,
    build_spot_direction_snapshot,
)


def _frame(
    *,
    rows: int = 260,
    start: float = 100.0,
    end: float = 130.0,
    wave_amp: float = 2.0,
) -> pd.DataFrame:
    base = np.linspace(start, end, rows)
    wave = np.sin(np.linspace(0, 12 * np.pi, rows)) * wave_amp
    close = base + wave
    open_ = close - np.sign(end - start or 1.0) * 0.15
    high = close + 0.65
    low = close - 0.65
    volume = np.linspace(100.0, 200.0, rows)
    return pd.DataFrame(
        {
            "timestamp": pd.date_range("2026-01-01", periods=rows, freq="h"),
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
        }
    )


def _breakout_frame(*, rows: int = 260, base: float = 100.0, breakout_end: float = 118.0) -> pd.DataFrame:
    close = np.full(rows, base, dtype=float)
    close += np.sin(np.linspace(0, 10 * np.pi, rows)) * 0.8
    close[-18:] = np.linspace(close[-19], breakout_end, 18)
    open_ = close - 0.15
    high = close + 0.65
    low = close - 0.65
    volume = np.linspace(100.0, 220.0, rows)
    return pd.DataFrame(
        {
            "timestamp": pd.date_range("2026-01-01", periods=rows, freq="h"),
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
        }
    )


class SpotDirectionContractTests(unittest.TestCase):
    def test_analyze_timeframe_direction_reads_uptrend(self) -> None:
        snap = analyze_timeframe_direction(_frame(start=100.0, end=140.0), timeframe="4h")
        self.assertIsInstance(snap, TimeframeDirectionSnapshot)
        self.assertEqual(snap.direction, "UPSIDE")
        self.assertGreater(snap.score, 20.0)
        self.assertGreater(snap.structure_score, 0.0)

    def test_analyze_timeframe_direction_reads_downtrend(self) -> None:
        snap = analyze_timeframe_direction(_frame(start=140.0, end=100.0), timeframe="1d")
        self.assertEqual(snap.direction, "DOWNSIDE")
        self.assertLess(snap.score, -20.0)
        self.assertLess(snap.structure_score, 0.0)

    def test_analyze_timeframe_direction_reads_range_as_neutral(self) -> None:
        snap = analyze_timeframe_direction(_frame(start=100.0, end=101.0, wave_amp=3.0), timeframe="4h")
        self.assertEqual(snap.direction, "NEUTRAL")
        self.assertEqual(snap.structure_label, "RANGE")

    def test_analyze_timeframe_direction_detects_breakout_structure(self) -> None:
        snap = analyze_timeframe_direction(_breakout_frame(), timeframe="1d")
        self.assertEqual(snap.structure_label, "BREAKOUT_UP")
        self.assertGreater(snap.structure_score, 0.0)

    def test_build_spot_direction_snapshot_prefers_aligned_higher_timeframes(self) -> None:
        out = build_spot_direction_snapshot(
            df_4h=_frame(start=100.0, end=130.0),
            df_1d=_frame(start=90.0, end=150.0),
        )
        self.assertEqual(out.direction, "UPSIDE")
        self.assertEqual(out.timeframe_alignment, 100.0)
        self.assertFalse(out.timeframe_conflict)
        self.assertIn("1D structure", out.note)

    def test_build_spot_direction_snapshot_returns_neutral_on_conflict(self) -> None:
        out = build_spot_direction_snapshot(
            df_4h=_frame(start=130.0, end=100.0),
            df_1d=_frame(start=90.0, end=150.0),
        )
        self.assertEqual(out.direction, "NEUTRAL")
        self.assertTrue(out.timeframe_conflict)
        self.assertEqual(out.timeframe_alignment, 0.0)

    def test_build_spot_direction_snapshot_returns_neutral_when_daily_bias_is_neutral(self) -> None:
        out = build_spot_direction_snapshot(
            df_4h=_frame(start=100.0, end=130.0),
            df_1d=_frame(start=100.0, end=101.0, wave_amp=3.0),
        )
        self.assertEqual(out.direction, "NEUTRAL")
        self.assertFalse(out.degraded_data)

    def test_build_spot_direction_snapshot_can_promote_early_daily_bias_when_4h_confirms(self) -> None:
        out = build_spot_direction_snapshot(
            df_4h=_frame(start=100.0, end=130.0),
            df_1d=_frame(start=100.0, end=102.0, wave_amp=3.0),
        )
        self.assertEqual(out.direction, "UPSIDE")
        self.assertFalse(out.timeframe_conflict)
        self.assertEqual(out.timeframe_alignment, 100.0)
        self.assertIn("does not oppose", out.note)

    def test_build_spot_direction_snapshot_can_promote_emerging_daily_bias_from_4h_breakout(self) -> None:
        out = build_spot_direction_snapshot(
            df_4h=_breakout_frame(base=100.0, breakout_end=122.0),
            df_1d=_frame(start=100.0, end=106.0, wave_amp=1.2),
        )
        self.assertEqual(out.direction, "UPSIDE")
        self.assertFalse(out.timeframe_conflict)

    def test_build_spot_direction_snapshot_marks_missing_context_as_degraded(self) -> None:
        out = build_spot_direction_snapshot(
            df_4h=_frame(rows=40, start=100.0, end=130.0),
            df_1d=_frame(start=90.0, end=150.0),
        )
        self.assertEqual(out.direction, "NEUTRAL")
        self.assertTrue(out.degraded_data)
        self.assertEqual(out.note, "Higher-timeframe context is incomplete.")

    def test_build_spot_direction_snapshot_supports_custom_anchor_pair(self) -> None:
        out = build_spot_direction_snapshot(
            df_4h=None,
            df_1d=None,
            confirm_df=_frame(start=100.0, end=125.0),
            lead_df=_frame(start=92.0, end=150.0),
            confirm_timeframe="1h",
            lead_timeframe="4h",
        )
        self.assertEqual(out.direction, "UPSIDE")
        self.assertEqual(out.lead_timeframe, "4h")
        self.assertEqual(out.confirm_timeframe, "1h")
        self.assertEqual(out.anchor_pair_label, "4H + 1H")
        self.assertIn("4H", out.note)


if __name__ == "__main__":
    unittest.main()
