from __future__ import annotations

import unittest

import pandas as pd

from tabs.signal_review_tab import (
    _annotate_actual_exit_quality,
    _annotate_actual_hold_style,
    _execution_vs_system_note,
    _prefer_known_summary_rows,
    _qualified_summary_rows,
    _review_scope_note,
)


class SignalReviewLogicTests(unittest.TestCase):
    def test_annotate_actual_hold_style_buckets_closed_trades(self) -> None:
        df = pd.DataFrame(
            [
                {
                    "actual_trade_status": "CLOSED",
                    "actual_entry_at": "2026-04-04T08:00:00Z",
                    "actual_exit_at": "2026-04-04T11:00:00Z",
                },
                {
                    "actual_trade_status": "CLOSED",
                    "actual_entry_at": "2026-04-04T08:00:00Z",
                    "actual_exit_at": "2026-04-05T08:30:00Z",
                },
                {
                    "actual_trade_status": "OPEN",
                    "actual_entry_at": "2026-04-04T08:00:00Z",
                    "actual_exit_at": "",
                },
            ]
        )
        out = _annotate_actual_hold_style(df)
        self.assertEqual(list(out["Hold Style"]), ["Quick Follow-Through", "Needs Room", "Open / Unjournaled"])
        self.assertIn("Actual Hold Hours", out.columns)

    def test_annotate_actual_exit_quality_buckets_trade_exits(self) -> None:
        df = pd.DataFrame(
            [
                {
                    "actual_trade_status": "CLOSED",
                    "actual_exit_reason": "Target",
                    "actual_pnl_pct": 3.2,
                },
                {
                    "actual_trade_status": "CLOSED",
                    "actual_exit_reason": "Manual Exit",
                    "actual_pnl_pct": 1.1,
                },
                {
                    "actual_trade_status": "CLOSED",
                    "actual_exit_reason": "Stop",
                    "actual_pnl_pct": -2.0,
                },
                {
                    "actual_trade_status": "CLOSED",
                    "actual_exit_reason": "Time Exit",
                    "actual_pnl_pct": -0.9,
                },
                {
                    "actual_trade_status": "OPEN",
                    "actual_exit_reason": "",
                    "actual_pnl_pct": None,
                },
            ]
        )
        out = _annotate_actual_exit_quality(df)
        self.assertEqual(
            list(out["Exit Quality"]),
            [
                "Target Winner",
                "Manual Winner Exit",
                "Protected Loss Exit",
                "Late Manual Loss",
                "Open / Unjournaled",
            ],
        )

    def test_review_scope_note_mentions_filtered_slice_not_full_archive(self) -> None:
        note = _review_scope_note("Resolved", "4h", 200, 87)
        self.assertIn("87 rows shown", note)
        self.assertIn("latest up to 200", note)
        self.assertIn("resolved signals", note)
        self.assertIn("4H only", note)
        self.assertIn("not the full tracker history", note)

    def test_execution_vs_system_note_says_archive_building_when_no_taken_trades(self) -> None:
        note, tone = _execution_vs_system_note(
            {
                "taken": 0.0,
                "taken_resolved": 0.0,
                "actual_closed": 0.0,
                "taken_follow_through_rate": 0.0,
                "actual_win_rate": 0.0,
                "execution_gap_pct": 0.0,
                "skipped_winners": 0.0,
            }
        )
        self.assertIn("journal is still building", note)
        self.assertEqual(tone, "neutral")

    def test_qualified_summary_rows_filters_on_minimum_count(self) -> None:
        df = pd.DataFrame(
            [
                {"Bucket": "A", "Resolved": 12},
                {"Bucket": "B", "Resolved": 7},
                {"Bucket": "C", "Resolved": 8},
            ]
        )
        out = _qualified_summary_rows(df, count_field="Resolved", min_count=8)
        self.assertEqual(list(out["Bucket"]), ["A", "C"])

    def test_prefer_known_summary_rows_drops_unknown_when_known_exists(self) -> None:
        df = pd.DataFrame(
            [
                {"Session": "Unknown", "Resolved": 20},
                {"Session": "US (16-00 UTC)", "Resolved": 9},
                {"Session": "Asia (00-08 UTC)", "Resolved": 8},
            ]
        )
        out = _prefer_known_summary_rows(df, label_field="Session")
        self.assertEqual(list(out["Session"]), ["US (16-00 UTC)", "Asia (00-08 UTC)"])


if __name__ == "__main__":
    unittest.main()
