from __future__ import annotations

import unittest

import pandas as pd

from tabs.signal_review_tab import (
    _annotate_actual_exit_quality,
    _annotate_actual_hold_style,
    _archive_building_card,
    _build_coin_hold_guidance_rows,
    _build_execution_signal_cards,
    _build_execution_review_cards,
    _build_hold_signal_cards,
    _display_trade_direction,
    _execution_vs_system_note,
    _format_review_metric,
    _follow_through_horizon_note,
    _hold_guidance_cell,
    _hold_archive_progress_snapshot,
    _hold_scope_label,
    _missing_hold_backfill_count,
    _refresh_scope_badge,
    _review_scope_summary,
    _learning_readiness_summary,
    _hold_window_note,
    _prepare_section_cards,
    _prefer_known_summary_rows,
    _qualified_summary_rows,
    _same_hold_progress,
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

    def test_review_scope_summary_is_compact(self) -> None:
        summary = _review_scope_summary(
            status_filter="All",
            timeframe_filter="All",
            limit=200,
            rows_in_view=73,
            symbol_filter="eth",
        )
        self.assertIn("ETH • All TF • All Status", summary)
        self.assertIn("73 of 200 rows shown", summary)
        self.assertIn("KPIs and deep dives use this view", summary)

    def test_learning_readiness_summary_is_compact(self) -> None:
        body, tone = _learning_readiness_summary(
            mode="current_only",
            current_rows=905,
            total_rows=1677,
        )
        self.assertIn("Current-only learning active", body)
        self.assertIn("905 current resolved", body)
        self.assertIn("1677 loaded", body)
        self.assertEqual(tone, "positive")

    def test_hold_scope_label_formats_coin_and_optional_timeframe(self) -> None:
        self.assertEqual(_hold_scope_label(symbol_filter="btc", timeframe_filter="All"), "BTC")
        self.assertEqual(_hold_scope_label(symbol_filter="btc", timeframe_filter="1h"), "BTC 1H")

    def test_refresh_scope_badge_uses_selected_scope(self) -> None:
        self.assertEqual(
            _refresh_scope_badge(symbol_filter="btc", timeframe_filter="1h", resolved_now=3),
            "BTC 1H +3 refreshed",
        )
        self.assertEqual(
            _refresh_scope_badge(symbol_filter="", timeframe_filter="All", resolved_now=0),
            "Market up to date",
        )

    def test_follow_through_horizon_note_mentions_selected_timeframe_horizon(self) -> None:
        note = _follow_through_horizon_note("4h")
        self.assertIn("4H", note)
        self.assertIn("12 bars", note)

    def test_follow_through_horizon_note_mentions_core_scanner_timeframes_when_unscoped(self) -> None:
        note = _follow_through_horizon_note("All")
        self.assertIn("5m = 12 bars", note)
        self.assertIn("15m = 16 bars", note)
        self.assertIn("1h = 12 bars", note)
        self.assertIn("4h = 12 bars", note)
        self.assertIn("1d = 10 bars", note)

    def test_format_review_metric_returns_na_when_unavailable(self) -> None:
        self.assertEqual(_format_review_metric(12.345, available=False, pct=True), "N/A")

    def test_format_review_metric_formats_signed_percent_when_available(self) -> None:
        self.assertEqual(
            _format_review_metric(0.493, available=True, pct=True, signed=True, decimals=2),
            "+0.49%",
        )

    def test_hold_window_note_describes_available_snapshot(self) -> None:
        note, tone = _hold_window_note(
            {
                "available": True,
                "resolved_signals": 18,
                "best_bar": 4,
                "best_label": "around 4 bars",
                "best_style": "Quick Follow-Through",
                "sample": 12,
                "avg_dir_return_pct": 1.42,
                "follow_through_pct": 62.5,
                "fade_after_bar": 8,
            }
        )
        self.assertIn("Best at: 4 bars", note)
        self.assertIn("fade after roughly <b>8</b> bars", note)
        self.assertIn("Quick Follow-Through", note)
        self.assertIn("62.5%", note)
        self.assertEqual(tone, "positive")

    def test_hold_guidance_cell_formats_available_and_building_states(self) -> None:
        self.assertEqual(
            _hold_guidance_cell(
                {
                    "available": True,
                    "best_bar": 4,
                    "best_label": "around 4 bars",
                    "sample": 12,
                }
            ),
            "Best at 4 bars (n=12)",
        )
        self.assertEqual(
            _hold_guidance_cell(
                {
                    "available": True,
                    "best_bar": 4,
                    "fade_after_bar": 6,
                    "best_label": "around 4 bars",
                    "sample": 10,
                }
            ),
            "Best at 4 bars, fades after 6 (n=10)",
        )
        self.assertEqual(
            _hold_guidance_cell(
                {
                    "available": False,
                    "resolved_signals": 5,
                }
            ),
            "Building (5 resolved)",
        )
        self.assertEqual(_hold_guidance_cell({"available": False, "resolved_signals": 0}), "—")
        self.assertEqual(_hold_guidance_cell({"available": False, "resolved_signals": 0}, direction_label="Downside"), "—")

    def test_build_coin_hold_guidance_rows_returns_timeframe_breakdown(self) -> None:
        df_events = pd.DataFrame(
            [
                {"signal_key": "a1", "timeframe": "5m", "direction": "Upside", "status": "RESOLVED"},
                {"signal_key": "a2", "timeframe": "5m", "direction": "Downside", "status": "RESOLVED"},
                {"signal_key": "b1", "timeframe": "1h", "direction": "Upside", "status": "RESOLVED"},
            ]
        )
        df_forward_windows = pd.DataFrame(
            [
                {"signal_key": "a1", "bars_ahead": 2, "directional_return_pct": 1.2, "adverse_excursion_pct": 0.2},
                {"signal_key": "a2", "bars_ahead": 4, "directional_return_pct": 0.8, "adverse_excursion_pct": 0.3},
                {"signal_key": "b1", "bars_ahead": 6, "directional_return_pct": 1.5, "adverse_excursion_pct": 0.4},
            ]
        )

        def _fake_hold_window_intelligence(df_scope: pd.DataFrame, _df_windows: pd.DataFrame) -> dict[str, object]:
            resolved = int(len(df_scope))
            if df_scope.empty:
                return {"available": False, "resolved_signals": 0}
            timeframe = str(df_scope.iloc[0]["timeframe"])
            direction = str(df_scope.iloc[0]["direction"])
            best_bar_map = {
                ("5m", "Upside"): 2,
                ("5m", "Downside"): 4,
                ("1h", "Upside"): 6,
            }
            if (timeframe, direction) not in best_bar_map:
                return {"available": False, "resolved_signals": resolved}
            best_bar = int(best_bar_map[(timeframe, direction)])
            return {
                "available": True,
                "resolved_signals": resolved,
                "best_bar": best_bar,
                "best_label": f"around {best_bar} bars",
                "sample": resolved,
            }

        rows = _build_coin_hold_guidance_rows(
            df_events,
            df_forward_windows,
            _fake_hold_window_intelligence,
            timeframe_filter="All",
        )
        self.assertEqual([row["Timeframe"] for row in rows], ["5M", "1H"])
        self.assertEqual(rows[0]["Upside Hold"], "Best at 2 bars (n=1)")
        self.assertEqual(rows[0]["Downside Hold"], "Best at 4 bars (n=1)")
        self.assertEqual(rows[1]["Upside Hold"], "Best at 6 bars (n=1)")
        self.assertEqual(rows[1]["Downside Hold"], "—")

    def test_missing_hold_backfill_count_counts_resolved_without_checkpoints(self) -> None:
        df_events = pd.DataFrame(
            [
                {"signal_key": "a1", "status": "RESOLVED"},
                {"signal_key": "a2", "status": "RESOLVED"},
                {"signal_key": "a3", "status": "OPEN"},
            ]
        )
        df_forward_windows = pd.DataFrame(
            [
                {"signal_key": "a1", "bars_ahead": 4},
            ]
        )
        self.assertEqual(_missing_hold_backfill_count(df_events, df_forward_windows), 1)

    def test_hold_archive_progress_snapshot_reports_coverage(self) -> None:
        df_events = pd.DataFrame(
            [
                {"signal_key": "a1", "status": "RESOLVED"},
                {"signal_key": "a2", "status": "RESOLVED"},
                {"signal_key": "a3", "status": "RESOLVED"},
                {"signal_key": "a4", "status": "OPEN"},
            ]
        )
        df_forward_windows = pd.DataFrame(
            [
                {"signal_key": "a1", "bars_ahead": 4},
                {"signal_key": "a3", "bars_ahead": 6},
            ]
        )
        snapshot = _hold_archive_progress_snapshot(df_events, df_forward_windows)
        self.assertEqual(snapshot["resolved"], 3.0)
        self.assertEqual(snapshot["ready"], 2.0)
        self.assertEqual(snapshot["missing"], 1.0)
        self.assertAlmostEqual(snapshot["coverage_pct"], 66.6666667, places=3)

    def test_same_hold_progress_detects_identical_scope_snapshots(self) -> None:
        left = {"resolved": 65.0, "ready": 7.0, "missing": 58.0, "coverage_pct": 10.7692308}
        right = {"resolved": 65.0, "ready": 7.0, "missing": 58.0, "coverage_pct": 10.7692308}
        other = {"resolved": 65.0, "ready": 8.0, "missing": 57.0, "coverage_pct": 12.3076923}
        self.assertTrue(_same_hold_progress(left, right))
        self.assertFalse(_same_hold_progress(left, other))

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

    def test_build_execution_review_cards_flags_thin_overlay_and_journal(self) -> None:
        cards = _build_execution_review_cards(
            {
                "total": 12.0,
                "overlay_marked": 3.0,
                "overlay_coverage_pct": 25.0,
                "taken": 1.0,
                "taken_resolved": 0.0,
                "actual_closed": 0.0,
                "journal_coverage_pct": 0.0,
                "execution_gap_pct": 0.0,
                "skipped_winners": 0.0,
                "skipped_resolved": 0.0,
                "skipped_winner_rate": 0.0,
            }
        )
        self.assertEqual(cards[0]["title"], "Manual Marking")
        self.assertIn("25.0% coverage", cards[0]["body_html"])
        self.assertEqual(cards[1]["title"], "Still Building")
        self.assertIn("Trade Journal", cards[1]["body_html"])
        self.assertIn("Execution Edge", cards[1]["body_html"])

    def test_build_execution_review_cards_reads_positive_execution_edge_when_archive_is_mature(self) -> None:
        cards = _build_execution_review_cards(
            {
                "total": 20.0,
                "overlay_marked": 18.0,
                "overlay_coverage_pct": 90.0,
                "taken": 8.0,
                "taken_resolved": 6.0,
                "actual_closed": 6.0,
                "journal_coverage_pct": 75.0,
                "execution_gap_pct": 0.82,
                "skipped_winners": 2.0,
                "skipped_resolved": 5.0,
                "skipped_winner_rate": 40.0,
            }
        )
        self.assertEqual(cards[0]["title"], "Manual Marking")
        self.assertEqual(cards[1]["title"], "Trade Journal")
        self.assertEqual(cards[2]["title"], "Execution Edge")
        self.assertIn("+0.82%", cards[2]["body_html"])
        self.assertIn("Skipped winners", cards[2]["body_html"])
        self.assertEqual(cards[2]["tone"], "positive")

    def test_build_execution_signal_cards_surface_execution_drag_and_missed_winners(self) -> None:
        works_cards, fail_cards = _build_execution_signal_cards(
            {
                "taken": 8.0,
                "taken_resolved": 6.0,
                "actual_closed": 4.0,
                "journal_coverage_pct": 50.0,
                "execution_gap_pct": -0.9,
                "skipped_winners": 3.0,
                "skipped_resolved": 5.0,
                "skipped_winner_rate": 60.0,
                "actual_win_rate": 50.0,
            }
        )
        self.assertEqual(works_cards, [])
        self.assertEqual(fail_cards[0]["title"], "Thin Journal Coverage")
        self.assertEqual(fail_cards[1]["title"], "Execution Drag")
        self.assertEqual(fail_cards[2]["title"], "Missed Winners")

    def test_build_hold_signal_cards_surface_best_and_weakest_profiles(self) -> None:
        hold_guidance_rows = [
            {
                "Timeframe": "5M",
                "Upside Snapshot": {
                    "available": True,
                    "best_label": "around 2 bars",
                    "best_style": "Explosive",
                    "sample": 12,
                    "follow_through_pct": 66.0,
                    "avg_dir_return_pct": 1.4,
                    "edge_score": 1.1,
                },
                "Downside Snapshot": {
                    "available": True,
                    "best_label": "around 6 bars",
                    "best_style": "Needs Room",
                    "sample": 9,
                    "follow_through_pct": 38.0,
                    "avg_dir_return_pct": -0.5,
                    "edge_score": -0.7,
                },
            }
        ]
        works_cards, fail_cards = _build_hold_signal_cards(hold_guidance_rows, symbol_filter="btc")
        self.assertEqual(works_cards[0]["title"], "Best Hold Edge")
        self.assertIn("BTC 5M Upside", works_cards[0]["body_html"])
        self.assertEqual(fail_cards[0]["title"], "Weakest Hold Edge")
        self.assertIn("BTC 5M Downside", fail_cards[0]["body_html"])
        self.assertEqual(fail_cards[0]["tone"], "warning")

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

    def test_prepare_section_cards_condenses_building_cards_into_archive_status(self) -> None:
        cards = [
            {
                "title": "Best Session",
                "body_html": "Session looks healthy.",
                "tone": "positive",
            },
            _archive_building_card("Hold Profile Archive", "Still building."),
            _archive_building_card("Primary Alert Archive", "Still building."),
        ]
        out = _prepare_section_cards(cards, max_actionable=3)
        self.assertEqual(out[0]["title"], "Best Session")
        self.assertEqual(out[1]["title"], "Still Building")
        self.assertIn("Hold Profile Archive", out[1]["body_html"])
        self.assertIn("Primary Alert Archive", out[1]["body_html"])

    def test_display_trade_direction_uses_upside_downside_labels(self) -> None:
        self.assertEqual(_display_trade_direction("LONG"), "Upside")
        self.assertEqual(_display_trade_direction("Short"), "Downside")
        self.assertEqual(_display_trade_direction("Upside"), "Upside")
        self.assertEqual(_display_trade_direction(""), "")


if __name__ == "__main__":
    unittest.main()
