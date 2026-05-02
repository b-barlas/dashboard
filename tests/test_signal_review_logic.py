from __future__ import annotations

import unittest

import pandas as pd

from tabs.signal_review_tab import (
    _BEST_SIGNAL_LEADERBOARD_LIMIT,
    _best_signal_summary,
    _build_best_signal_leaderboard,
    _build_coin_hold_guidance_rows,
    _build_coin_timeframe_intelligence_bundle,
    _build_expected_path_projection,
    _display_trade_direction,
    _format_review_metric,
    _expected_path_building_body_html,
    _expected_path_body_html,
    _expected_path_kpi_items,
    _fetch_expected_path_reference_price,
    _filter_events_by_setup,
    _filter_events_to_setup_direction,
    _filter_events_to_setup_pocket,
    _follow_through_horizon_note,
    _hold_guidance_cell,
    _load_coin_timeframe_frames,
    _missing_hold_backfill_count,
    _refresh_scope_badge,
    _best_signal_learning_summary,
    _learning_readiness_summary,
    _ordered_timeframe_scope,
    _prefer_known_summary_rows,
    _selected_best_signal_coin,
    _selected_dataframe_row_index,
    _select_best_signal_coin,
    _select_setup_pocket,
    _setup_filter_value,
    _setup_pocket_label,
)


class SignalReviewLogicTests(unittest.TestCase):
    def test_learning_readiness_summary_is_compact(self) -> None:
        body, tone = _learning_readiness_summary(
            mode="current_only",
            current_rows=905,
            total_rows=1677,
        )
        self.assertIn("Learning active", body)
        self.assertIn("905 signals", body)
        self.assertIn("history window", body)
        self.assertEqual(tone, "positive")

    def test_best_signal_learning_summary_uses_selected_best_setup_count(self) -> None:
        body, tone = _best_signal_learning_summary({"available": True, "resolved": 19, "best_resolved": 16})
        self.assertIn("Learning active", body)
        self.assertIn("16 completed", body)
        self.assertNotIn("19", body)
        self.assertEqual(tone, "positive")

    def test_ordered_timeframe_scope_keeps_canonical_order_and_extras(self) -> None:
        self.assertEqual(
            _ordered_timeframe_scope(timeframe_filter="All", available_timeframes={"1h", "5m", "30m"}),
            ["5m", "15m", "1h", "4h", "1d", "30m"],
        )
        self.assertEqual(_ordered_timeframe_scope(timeframe_filter="4h", available_timeframes={"1h"}), ["4h"])

    def test_load_coin_timeframe_frames_filters_base_events_by_status_for_specific_tf(self) -> None:
        base_events = pd.DataFrame(
            [
                {"signal_key": "open-1", "timeframe": "1h", "status": "OPEN"},
                {"signal_key": "resolved-1", "timeframe": "1h", "status": "RESOLVED"},
                {"signal_key": "resolved-5m", "timeframe": "5m", "status": "RESOLVED"},
            ]
        )

        def fake_fetch_events(**_kwargs):
            raise AssertionError("specific timeframe should reuse filtered base_events")

        def fake_fetch_windows(**kwargs):
            return pd.DataFrame({"signal_key": list(kwargs["signal_keys"]), "bars_ahead": [1] * len(kwargs["signal_keys"])})

        frames = _load_coin_timeframe_frames(
            fetch_signal_events_df=fake_fetch_events,
            fetch_signal_forward_windows_df=fake_fetch_windows,
            symbol_filter="TRX",
            timeframe_filter="1h",
            status_filter="Resolved",
            current_market_version="test",
            analysis_limit=100,
            db_path=":memory:",
            base_events=base_events,
        )

        self.assertEqual(len(frames), 1)
        self.assertEqual(list(frames[0]["events"]["signal_key"]), ["resolved-1"])
        self.assertEqual(list(frames[0]["windows"]["signal_key"]), ["resolved-1"])

    def test_load_coin_timeframe_frames_reuses_selected_pocket_scope_for_all_tf(self) -> None:
        base_events = pd.DataFrame(
            [
                {
                    "signal_key": "watch-up-1",
                    "timeframe": "1h",
                    "direction": "UPSIDE",
                    "setup_confirm": "WATCH",
                    "status": "RESOLVED",
                },
                {
                    "signal_key": "watch-up-2",
                    "timeframe": "1h",
                    "direction": "UPSIDE",
                    "setup_confirm": "WATCH",
                    "status": "RESOLVED",
                },
                {
                    "signal_key": "watch-down-1",
                    "timeframe": "1h",
                    "direction": "DOWNSIDE",
                    "setup_confirm": "WATCH",
                    "status": "RESOLVED",
                },
                {
                    "signal_key": "enter-up-1",
                    "timeframe": "5m",
                    "direction": "UPSIDE",
                    "setup_confirm": "ENTER_TREND_AI",
                    "status": "RESOLVED",
                },
            ]
        )
        pocket = {
            "available": True,
            "setup_class": "WATCH",
            "direction": "UPSIDE",
        }
        selected_scope = _filter_events_to_setup_direction(base_events, pocket)

        def fake_fetch_events(**_kwargs):
            raise AssertionError("All timeframe read should reuse the selected setup scope")

        def fake_fetch_windows(**kwargs):
            return pd.DataFrame({"signal_key": list(kwargs["signal_keys"]), "bars_ahead": [1] * len(kwargs["signal_keys"])})

        frames = _load_coin_timeframe_frames(
            fetch_signal_events_df=fake_fetch_events,
            fetch_signal_forward_windows_df=fake_fetch_windows,
            symbol_filter="TRX",
            timeframe_filter="All",
            status_filter="Resolved",
            current_market_version="test",
            analysis_limit=100,
            db_path=":memory:",
            base_events=selected_scope,
        )

        frame_map = {str(frame["timeframe"]): frame["events"] for frame in frames}
        self.assertEqual(list(frame_map["1h"]["signal_key"]), ["watch-up-1", "watch-up-2"])
        self.assertEqual(frame_map["5m"].to_dict("records"), [])

    def test_select_best_signal_coin_balances_across_timeframes(self) -> None:
        df = pd.DataFrame(
            [
                {"symbol": "AAA", "timeframe": "5m", "status": "RESOLVED", "directional_return_pct": 1.0},
                {"symbol": "AAA", "timeframe": "5m", "status": "RESOLVED", "directional_return_pct": 1.1},
                {"symbol": "AAA", "timeframe": "5m", "status": "RESOLVED", "directional_return_pct": 0.9},
                {"symbol": "AAA", "timeframe": "5m", "status": "RESOLVED", "directional_return_pct": 1.2},
                {"symbol": "AAA", "timeframe": "1h", "status": "RESOLVED", "directional_return_pct": 0.8},
                {"symbol": "AAA", "timeframe": "1h", "status": "RESOLVED", "directional_return_pct": 0.7},
                {"symbol": "AAA", "timeframe": "1h", "status": "RESOLVED", "directional_return_pct": 0.9},
                {"symbol": "AAA", "timeframe": "1h", "status": "RESOLVED", "directional_return_pct": 0.6},
                {"symbol": "BBB", "timeframe": "5m", "status": "RESOLVED", "directional_return_pct": 1.0},
                {"symbol": "BBB", "timeframe": "5m", "status": "RESOLVED", "directional_return_pct": 1.1},
                {"symbol": "BBB", "timeframe": "5m", "status": "RESOLVED", "directional_return_pct": 0.9},
                {"symbol": "BBB", "timeframe": "5m", "status": "RESOLVED", "directional_return_pct": 1.2},
                {"symbol": "BBB", "timeframe": "15m", "status": "RESOLVED", "directional_return_pct": -0.2},
                {"symbol": "BBB", "timeframe": "15m", "status": "RESOLVED", "directional_return_pct": -0.1},
                {"symbol": "BBB", "timeframe": "15m", "status": "RESOLVED", "directional_return_pct": 0.0},
                {"symbol": "BBB", "timeframe": "15m", "status": "RESOLVED", "directional_return_pct": -0.3},
            ]
        )
        df["direction"] = "UPSIDE"
        selection = _select_best_signal_coin(df_events=df, timeframe_filter="All", min_resolved=4, min_timeframes=2)
        self.assertTrue(selection["available"])
        self.assertEqual(selection["symbol"], "AAA")
        self.assertEqual(selection["mode"], "best_available")
        self.assertEqual(selection["qualified_timeframes"], 2)

    def test_select_best_signal_coin_marks_cross_timeframe_when_thresholds_are_met(self) -> None:
        df = pd.DataFrame(
            [
                {"symbol": "AAA", "timeframe": "5m", "status": "RESOLVED", "directional_return_pct": 1.0},
                {"symbol": "AAA", "timeframe": "5m", "status": "RESOLVED", "directional_return_pct": 1.1},
                {"symbol": "AAA", "timeframe": "5m", "status": "RESOLVED", "directional_return_pct": 0.9},
                {"symbol": "AAA", "timeframe": "5m", "status": "RESOLVED", "directional_return_pct": 1.2},
                {"symbol": "AAA", "timeframe": "1h", "status": "RESOLVED", "directional_return_pct": 0.8},
                {"symbol": "AAA", "timeframe": "1h", "status": "RESOLVED", "directional_return_pct": 0.7},
                {"symbol": "AAA", "timeframe": "1h", "status": "RESOLVED", "directional_return_pct": 0.9},
                {"symbol": "AAA", "timeframe": "1h", "status": "RESOLVED", "directional_return_pct": 0.6},
                {"symbol": "AAA", "timeframe": "4h", "status": "RESOLVED", "directional_return_pct": 0.5},
                {"symbol": "AAA", "timeframe": "4h", "status": "RESOLVED", "directional_return_pct": 0.4},
                {"symbol": "AAA", "timeframe": "4h", "status": "RESOLVED", "directional_return_pct": 0.6},
                {"symbol": "AAA", "timeframe": "4h", "status": "RESOLVED", "directional_return_pct": 0.3},
            ]
        )
        df["direction"] = "UPSIDE"
        selection = _select_best_signal_coin(
            df_events=df,
            timeframe_filter="All",
            min_resolved=4,
            min_timeframes=2,
            min_total_resolved=8,
        )
        self.assertTrue(selection["available"])
        self.assertEqual(selection["symbol"], "AAA")
        self.assertEqual(selection["mode"], "cross_timeframe")
        self.assertEqual(selection["qualified_timeframes"], 3)

    def test_select_best_signal_coin_uses_selected_timeframe(self) -> None:
        df = pd.DataFrame(
            [
                {"symbol": "AAA", "timeframe": "1h", "status": "RESOLVED", "directional_return_pct": 0.2},
                {"symbol": "AAA", "timeframe": "1h", "status": "RESOLVED", "directional_return_pct": 0.3},
                {"symbol": "AAA", "timeframe": "1h", "status": "RESOLVED", "directional_return_pct": 0.4},
                {"symbol": "AAA", "timeframe": "1h", "status": "RESOLVED", "directional_return_pct": 0.5},
                {"symbol": "BBB", "timeframe": "1h", "status": "RESOLVED", "directional_return_pct": 1.2},
                {"symbol": "BBB", "timeframe": "1h", "status": "RESOLVED", "directional_return_pct": 1.1},
                {"symbol": "BBB", "timeframe": "1h", "status": "RESOLVED", "directional_return_pct": 1.0},
                {"symbol": "BBB", "timeframe": "1h", "status": "RESOLVED", "directional_return_pct": 0.9},
            ]
        )
        df["direction"] = "UPSIDE"
        selection = _select_best_signal_coin(df_events=df, timeframe_filter="1h", min_resolved=4)
        self.assertTrue(selection["available"])
        self.assertEqual(selection["symbol"], "BBB")
        self.assertEqual(selection["mode"], "timeframe")
        self.assertEqual(selection["best_timeframe"], "1h")

    def test_select_best_signal_coin_requires_path_samples_when_windows_are_supplied(self) -> None:
        rows = []
        for idx in range(16):
            rows.append(
                {
                    "signal_key": f"bnb-{idx}",
                    "symbol": "BNB",
                    "timeframe": "1h",
                    "status": "RESOLVED",
                    "direction": "DOWNSIDE",
                    "directional_return_pct": 2.0,
                }
            )
        for idx in range(12):
            rows.append(
                {
                    "signal_key": f"trx-{idx}",
                    "symbol": "TRX",
                    "timeframe": "1h",
                    "status": "RESOLVED",
                    "direction": "UPSIDE",
                    "directional_return_pct": 1.0,
                }
            )
        windows = pd.DataFrame({"signal_key": [f"trx-{idx}" for idx in range(8)]})

        selection = _select_best_signal_coin(
            df_events=pd.DataFrame(rows),
            df_forward_windows=windows,
            timeframe_filter="1h",
            min_resolved=8,
            min_path_samples=8,
        )

        self.assertTrue(selection["available"])
        self.assertEqual(selection["symbol"], "TRX")

    def test_select_best_signal_coin_uses_same_setup_pockets_as_leaderboard(self) -> None:
        rows = []
        for idx in range(6):
            rows.append(
                {
                    "symbol": "PENGU",
                    "timeframe": "5m",
                    "status": "RESOLVED",
                    "direction": "UPSIDE",
                    "setup_confirm": "WATCH",
                    "directional_return_pct": 1.0,
                }
            )
            rows.append(
                {
                    "symbol": "PENGU",
                    "timeframe": "5m",
                    "status": "RESOLVED",
                    "direction": "UPSIDE",
                    "setup_confirm": "PROBE",
                    "directional_return_pct": 1.0,
                }
            )
        for idx in range(8):
            rows.append(
                {
                    "symbol": "BNB",
                    "timeframe": "5m",
                    "status": "RESOLVED",
                    "direction": "DOWNSIDE",
                    "setup_confirm": "PROBE",
                    "directional_return_pct": 0.8,
                }
            )

        df = pd.DataFrame(rows)
        selection = _select_best_signal_coin(
            df_events=df,
            timeframe_filter="All",
            min_resolved=8,
            min_timeframes=1,
            min_total_resolved=8,
        )
        board = _build_best_signal_leaderboard(
            df_events=df,
            timeframe_filter="All",
            min_resolved=8,
            min_timeframes=1,
            min_total_resolved=8,
        )

        self.assertEqual(selection["symbol"], "BNB")
        self.assertEqual(list(board["Coin"]), ["BNB"])
        self.assertNotIn("PENGU", set(board["Coin"]))

    def test_best_signal_summary_mentions_cross_timeframe_mode(self) -> None:
        summary = _best_signal_summary(
            selection={
                "symbol": "ETH",
                "mode": "cross_timeframe",
                "qualified_timeframes": 3,
                "follow_through_pct": 62.5,
                "avg_dir_return_pct": 1.42,
                "resolved": 34,
                "best_timeframe": "1h",
            },
            timeframe_filter="All",
            analysis_limit=10000,
        )
        self.assertIn("ETH", summary)
        self.assertIn("enough history", summary)
        self.assertIn("1H", summary)

    def test_best_signal_summary_softens_best_available_copy(self) -> None:
        summary = _best_signal_summary(
            selection={
                "symbol": "LDO",
                "mode": "best_available",
                "qualified_timeframes": 1,
                "follow_through_pct": 90.9,
                "avg_dir_return_pct": 3.25,
                "resolved": 11,
                "best_timeframe": "1h",
            },
            timeframe_filter="All",
            analysis_limit=10000,
        )
        self.assertIn("Best available read with enough history", summary)
        self.assertIn("Only <b>1</b> timeframe", summary)

    def test_build_best_signal_leaderboard_labels_modes(self) -> None:
        df = pd.DataFrame(
            [
                {"symbol": "AAA", "timeframe": "5m", "status": "RESOLVED", "directional_return_pct": 1.0},
                {"symbol": "AAA", "timeframe": "5m", "status": "RESOLVED", "directional_return_pct": 1.1},
                {"symbol": "AAA", "timeframe": "5m", "status": "RESOLVED", "directional_return_pct": 0.9},
                {"symbol": "AAA", "timeframe": "5m", "status": "RESOLVED", "directional_return_pct": 1.2},
                {"symbol": "AAA", "timeframe": "1h", "status": "RESOLVED", "directional_return_pct": 0.8},
                {"symbol": "AAA", "timeframe": "1h", "status": "RESOLVED", "directional_return_pct": 0.7},
                {"symbol": "AAA", "timeframe": "1h", "status": "RESOLVED", "directional_return_pct": 0.9},
                {"symbol": "AAA", "timeframe": "1h", "status": "RESOLVED", "directional_return_pct": 0.6},
                {"symbol": "AAA", "timeframe": "4h", "status": "RESOLVED", "directional_return_pct": 0.5},
                {"symbol": "AAA", "timeframe": "4h", "status": "RESOLVED", "directional_return_pct": 0.4},
                {"symbol": "AAA", "timeframe": "4h", "status": "RESOLVED", "directional_return_pct": 0.6},
                {"symbol": "AAA", "timeframe": "4h", "status": "RESOLVED", "directional_return_pct": 0.3},
                {"symbol": "BBB", "timeframe": "1h", "status": "RESOLVED", "directional_return_pct": 1.2},
                {"symbol": "BBB", "timeframe": "1h", "status": "RESOLVED", "directional_return_pct": 1.1},
                {"symbol": "BBB", "timeframe": "1h", "status": "RESOLVED", "directional_return_pct": 1.0},
                {"symbol": "BBB", "timeframe": "1h", "status": "RESOLVED", "directional_return_pct": 0.9},
            ]
        )
        df["direction"] = "UPSIDE"
        board = _build_best_signal_leaderboard(
            df_events=df,
            timeframe_filter="All",
            limit=5,
            min_resolved=4,
            min_timeframes=2,
            min_total_resolved=8,
        )
        self.assertEqual(list(board.columns), ["Coin", "Mode", "Best Setup", "Follow-Through", "Resolved", "Best TF", "Avg Move"])
        self.assertEqual(str(board.iloc[0]["Coin"]), "AAA")
        self.assertEqual(str(board.iloc[0]["Mode"]), "Best Signal")
        self.assertEqual(str(board.iloc[1]["Coin"]), "BBB")
        self.assertEqual(str(board.iloc[1]["Mode"]), "Best Read")

    def test_build_best_signal_leaderboard_shows_best_setup_direction(self) -> None:
        rows = []
        for idx in range(12):
            rows.append(
                {
                    "symbol": "TRX",
                    "timeframe": "1h",
                    "status": "RESOLVED",
                    "direction": "UPSIDE",
                    "setup_confirm": "WATCH",
                    "directional_return_pct": 0.6,
                }
            )
        for idx in range(12):
            rows.append(
                {
                    "symbol": "ENJ",
                    "timeframe": "1h",
                    "status": "RESOLVED",
                    "direction": "DOWNSIDE",
                    "setup_confirm": "ENTER_TREND_AI",
                    "directional_return_pct": 0.8,
                }
            )

        board = _build_best_signal_leaderboard(
            df_events=pd.DataFrame(rows),
            timeframe_filter="1h",
            min_resolved=8,
        )

        self.assertEqual(str(board.iloc[0]["Best Setup"]), "ENTER ↓ T+AI")
        self.assertEqual(str(board.iloc[1]["Best Setup"]), "WATCH ↑")

    def test_build_best_signal_leaderboard_defaults_to_top_10(self) -> None:
        rows = []
        for idx in range(12):
            symbol = f"C{idx:02d}"
            rows.append(
                {
                    "symbol": symbol,
                    "timeframe": "1h",
                    "status": "RESOLVED",
                    "direction": "UPSIDE",
                    "directional_return_pct": 1.0 + (idx / 100.0),
                }
            )
        board = _build_best_signal_leaderboard(
            df_events=pd.DataFrame(rows),
            timeframe_filter="1h",
            min_resolved=1,
        )
        self.assertEqual(len(board), _BEST_SIGNAL_LEADERBOARD_LIMIT)

    def test_build_best_signal_leaderboard_excludes_neutral_only_rows(self) -> None:
        rows = [
            {"symbol": "ALGO", "timeframe": "1h", "status": "RESOLVED", "direction": "NEUTRAL", "directional_return_pct": 1.0}
            for _ in range(20)
        ]
        rows.extend(
            {
                "symbol": "TRX",
                "timeframe": "1h",
                "status": "RESOLVED",
                "direction": "UPSIDE",
                "directional_return_pct": 1.0,
            }
            for _ in range(12)
        )

        board = _build_best_signal_leaderboard(
            df_events=pd.DataFrame(rows),
            timeframe_filter="1h",
            min_resolved=8,
        )

        self.assertEqual(list(board["Coin"]), ["TRX"])

    def test_build_best_signal_leaderboard_requires_path_ready_rows_when_windows_are_supplied(self) -> None:
        rows = []
        for idx in range(16):
            rows.append(
                {
                    "signal_key": f"bnb-{idx}",
                    "symbol": "BNB",
                    "timeframe": "1h",
                    "status": "RESOLVED",
                    "direction": "DOWNSIDE",
                    "directional_return_pct": 2.0,
                }
            )
        for idx in range(12):
            rows.append(
                {
                    "signal_key": f"trx-{idx}",
                    "symbol": "TRX",
                    "timeframe": "1h",
                    "status": "RESOLVED",
                    "direction": "UPSIDE",
                    "directional_return_pct": 1.0,
                }
            )
        windows = pd.DataFrame({"signal_key": [f"trx-{idx}" for idx in range(8)]})

        board = _build_best_signal_leaderboard(
            df_events=pd.DataFrame(rows),
            df_forward_windows=windows,
            timeframe_filter="1h",
            min_resolved=8,
            min_path_samples=8,
        )

        self.assertEqual(list(board["Coin"]), ["TRX"])

    def test_selected_dataframe_row_index_reads_mapping_state(self) -> None:
        self.assertEqual(_selected_dataframe_row_index({"selection": {"rows": [2]}}), 2)
        self.assertIsNone(_selected_dataframe_row_index({"selection": {"rows": []}}))

    def test_selected_best_signal_coin_returns_selected_coin(self) -> None:
        board = pd.DataFrame(
            [
                {"Coin": "TRX", "Mode": "Best Read"},
                {"Coin": "ETH", "Mode": "Best Signal"},
            ]
        )
        selected = _selected_best_signal_coin(board, {"selection": {"rows": [1]}})
        self.assertEqual(selected, "ETH")

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
        self.assertIn("12 candles", note)

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
            "Best at 4 bars",
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
            "Best at 4 bars, fades after 6 bars",
        )
        self.assertEqual(
            _hold_guidance_cell(
                {
                    "available": False,
                    "resolved_signals": 5,
                }
            ),
            "Building (5 completed)",
        )
        self.assertEqual(_hold_guidance_cell({"available": False, "resolved_signals": 0}), "—")
        self.assertEqual(_hold_guidance_cell({"available": False, "resolved_signals": 0}, direction_label="Downside"), "—")

    def test_setup_pocket_auto_best_prefers_actionable_history_over_skip(self) -> None:
        rows = []
        for idx in range(8):
            rows.append(
                {
                    "signal_key": f"watch-{idx}",
                    "symbol": "TRX",
                    "timeframe": "1h",
                    "direction": "Upside",
                    "setup_confirm": "WATCH",
                    "status": "RESOLVED",
                    "directional_return_pct": 0.85,
                    "adverse_excursion_pct": 0.10,
                }
            )
        for idx in range(8):
            rows.append(
                {
                    "signal_key": f"skip-{idx}",
                    "symbol": "TRX",
                    "timeframe": "1h",
                    "direction": "Upside",
                    "setup_confirm": "SKIP",
                    "status": "RESOLVED",
                    "directional_return_pct": 2.00,
                    "adverse_excursion_pct": 0.05,
                }
            )

        pocket = _select_setup_pocket(pd.DataFrame(rows), setup_filter_value="AUTO_BEST")

        self.assertTrue(pocket["available"])
        self.assertEqual(pocket["setup_class"], "WATCH")
        self.assertEqual(pocket["setup_label"], "WATCH")
        self.assertEqual(pocket["timeframe"], "1h")
        self.assertEqual(pocket["direction"], "UPSIDE")
        self.assertEqual(_setup_pocket_label(pocket), "WATCH ↑ • 1H")

    def test_setup_pocket_specific_filter_restricts_best_read(self) -> None:
        rows = []
        for idx in range(8):
            rows.append(
                {
                    "signal_key": f"watch-{idx}",
                    "symbol": "ETH",
                    "timeframe": "1h",
                    "direction": "Upside",
                    "setup_confirm": "WATCH",
                    "status": "RESOLVED",
                    "directional_return_pct": 1.20,
                    "adverse_excursion_pct": 0.10,
                }
            )
        for idx in range(10):
            rows.append(
                {
                    "signal_key": f"enter-{idx}",
                    "symbol": "ETH",
                    "timeframe": "15m",
                    "direction": "Downside",
                    "setup_confirm": "ENTER_TREND_AI",
                    "status": "RESOLVED",
                    "directional_return_pct": 0.25,
                    "adverse_excursion_pct": 0.20,
                }
            )

        pocket = _select_setup_pocket(pd.DataFrame(rows), setup_filter_value=_setup_filter_value("ENTER T+AI"))

        self.assertTrue(pocket["available"])
        self.assertEqual(pocket["setup_class"], "ENTER_TREND_AI")
        self.assertEqual(pocket["timeframe"], "15m")
        self.assertEqual(pocket["direction"], "DOWNSIDE")
        self.assertEqual(pocket["completed"], 10)

    def test_setup_pocket_filter_keeps_open_rows_for_same_pocket_metrics(self) -> None:
        rows = []
        for idx in range(8):
            rows.append(
                {
                    "signal_key": f"probe-{idx}",
                    "symbol": "SOL",
                    "timeframe": "4h",
                    "direction": "Upside",
                    "setup_confirm": "PROBE",
                    "status": "RESOLVED",
                    "directional_return_pct": 0.55,
                    "adverse_excursion_pct": 0.15,
                }
            )
        rows.append(
            {
                "signal_key": "probe-open",
                "symbol": "SOL",
                "timeframe": "4h",
                "direction": "Upside",
                "setup_confirm": "PROBE",
                "status": "OPEN",
                "directional_return_pct": None,
                "adverse_excursion_pct": None,
            }
        )
        rows.append(
            {
                "signal_key": "watch-other",
                "symbol": "SOL",
                "timeframe": "4h",
                "direction": "Upside",
                "setup_confirm": "WATCH",
                "status": "OPEN",
                "directional_return_pct": None,
                "adverse_excursion_pct": None,
            }
        )
        df = pd.DataFrame(rows)

        pocket = _select_setup_pocket(df, setup_filter_value="AUTO_BEST")
        filtered = _filter_events_to_setup_pocket(df, pocket)

        self.assertTrue(pocket["available"])
        self.assertEqual(pocket["setup_label"], "EARLY")
        self.assertEqual(len(filtered), 9)
        self.assertEqual(set(filtered["signal_key"]), {f"probe-{idx}" for idx in range(8)} | {"probe-open"})

    def test_filter_events_by_setup_maps_user_friendly_setup_labels(self) -> None:
        df = pd.DataFrame(
            [
                {"setup_confirm": "PROBE", "symbol": "BTC"},
                {"setup_confirm": "WATCH", "symbol": "ETH"},
            ]
        )

        filtered = _filter_events_by_setup(df, _setup_filter_value("EARLY"))

        self.assertEqual(filtered["symbol"].tolist(), ["BTC"])
        self.assertEqual(filtered["__setup_label"].tolist(), ["EARLY"])

    def test_filter_events_by_setup_auto_best_excludes_skip_and_unknown(self) -> None:
        df = pd.DataFrame(
            [
                {"setup_confirm": "SKIP", "symbol": "SKIPPER"},
                {"setup_confirm": "legacy value", "symbol": "OLD"},
                {"setup_confirm": "WATCH", "symbol": "WATCHER"},
            ]
        )

        filtered = _filter_events_by_setup(df, "AUTO_BEST")

        self.assertEqual(filtered["symbol"].tolist(), ["WATCHER"])
        self.assertEqual(filtered["__setup_class"].tolist(), ["WATCH"])

    def test_expected_path_projection_builds_plain_scenario(self) -> None:
        events = []
        windows = []
        for idx in range(10):
            key = f"sig-{idx}"
            events.append(
                {
                    "signal_key": key,
                    "symbol": "ETH",
                    "timeframe": "1h",
                    "direction": "Upside",
                    "status": "RESOLVED",
                    "price": 100.0 + idx,
                    "event_time": f"2026-04-28T0{idx % 9}:00:00Z",
                }
            )
            windows.extend(
                [
                    {
                        "signal_key": key,
                        "bars_ahead": 4,
                        "directional_return_pct": 0.9 + idx * 0.02,
                        "adverse_excursion_pct": 0.25,
                    },
                    {
                        "signal_key": key,
                        "bars_ahead": 8,
                        "directional_return_pct": 2.0 + idx * 0.04,
                        "adverse_excursion_pct": 0.55,
                    },
                    {
                        "signal_key": key,
                        "bars_ahead": 12,
                        "directional_return_pct": 0.45 + idx * 0.01,
                        "adverse_excursion_pct": 0.75,
                    },
                ]
            )

        snapshot = _build_expected_path_projection(
            df_events=pd.DataFrame(events),
            df_forward_windows=pd.DataFrame(windows),
            symbol_filter="ETH",
            timeframe_filter="1h",
            min_samples=8,
            now="2026-04-28T12:00:00Z",
        )

        self.assertTrue(snapshot["available"])
        self.assertEqual(snapshot["timeframe"], "1h")
        self.assertEqual(snapshot["direction"], "UPSIDE")
        self.assertEqual(snapshot["best_bar"], 8)
        self.assertEqual(snapshot["fade_after_bar"], 12)
        self.assertEqual(snapshot["read_quality"], "Thin")
        self.assertEqual(snapshot["archive_check_sample"], 10)
        self.assertAlmostEqual(snapshot["zone_hit_rate_pct"], 60.0, places=4)
        self.assertAlmostEqual(snapshot["clean_path_rate_pct"], 60.0, places=4)
        body = _expected_path_body_html(snapshot)
        self.assertIn("Reference price", body)
        self.assertIn("Expected zone", body)
        self.assertIn("Best path window", body)
        self.assertIn("next 8 candles on 1H", body)
        self.assertIn("Move from that price", body)
        self.assertIn("usual upside window", body)
        self.assertIn("Normal pullback", body)
        self.assertIn("can still be normal", body)
        self.assertIn("Path weakens after", body)
        self.assertIn("Archive check", body)
        self.assertIn("60% reached this zone", body)
        self.assertIn("Price path", body)
        self.assertIn("$", body)

    def test_expected_path_keeps_price_path_timing_separate_from_hold_efficiency(self) -> None:
        events = []
        windows = []
        for idx in range(8):
            key = f"trx-{idx}"
            events.append(
                {
                    "signal_key": key,
                    "symbol": "TRX",
                    "timeframe": "1h",
                    "direction": "Upside",
                    "status": "RESOLVED",
                    "price": 0.32,
                    "event_time": f"2026-04-28T0{idx}:00:00Z",
                }
            )
            windows.extend(
                [
                    {
                        "signal_key": key,
                        "bars_ahead": 1,
                        "directional_return_pct": 0.40,
                        "adverse_excursion_pct": 0.05,
                    },
                    {
                        "signal_key": key,
                        "bars_ahead": 2,
                        "directional_return_pct": 0.12,
                        "adverse_excursion_pct": 0.05,
                    },
                    {
                        "signal_key": key,
                        "bars_ahead": 4,
                        "directional_return_pct": 0.70,
                        "adverse_excursion_pct": 0.85,
                    },
                    {
                        "signal_key": key,
                        "bars_ahead": 8,
                        "directional_return_pct": 0.25,
                        "adverse_excursion_pct": 0.10,
                    },
                ]
            )

        snapshot = _build_expected_path_projection(
            df_events=pd.DataFrame(events),
            df_forward_windows=pd.DataFrame(windows),
            symbol_filter="TRX",
            timeframe_filter="1h",
            min_samples=8,
            now="2026-04-28T12:00:00Z",
        )

        self.assertTrue(snapshot["available"])
        self.assertEqual(snapshot["best_bar"], 4)
        self.assertEqual(snapshot["fade_after_bar"], 8)
        body = _expected_path_body_html(snapshot)
        self.assertIn("next 4 candles on 1H", body)
        self.assertIn("Path weakens after: <b>8 bars</b>", body)
        self.assertIn("Hold window above is the efficiency read", body)

    def test_expected_path_archive_check_uses_favorable_excursion_when_available(self) -> None:
        events = []
        windows = []
        for idx in range(8):
            key = f"zone-{idx}"
            events.append(
                {
                    "signal_key": key,
                    "symbol": "SOL",
                    "timeframe": "1h",
                    "direction": "Upside",
                    "status": "RESOLVED",
                    "price": 100.0,
                    "event_time": f"2026-04-28T0{idx}:00:00Z",
                }
            )
            windows.append(
                {
                    "signal_key": key,
                    "bars_ahead": 4,
                    "directional_return_pct": 0.5,
                    "favorable_excursion_pct": 0.7 if idx < 6 else 0.2,
                    "adverse_excursion_pct": 0.2,
                }
            )

        snapshot = _build_expected_path_projection(
            df_events=pd.DataFrame(events),
            df_forward_windows=pd.DataFrame(windows),
            symbol_filter="SOL",
            timeframe_filter="1h",
            min_samples=8,
            now="2026-04-28T12:00:00Z",
        )

        self.assertTrue(snapshot["available"])
        self.assertEqual(snapshot["archive_check_sample"], 8)
        self.assertAlmostEqual(snapshot["zone_hit_rate_pct"], 75.0, places=4)
        body = _expected_path_body_html(snapshot)
        self.assertIn("75% reached this zone", body)

    def test_expected_path_kpi_items_explain_caution_without_extra_card(self) -> None:
        snapshot = {
            "available": True,
            "symbol": "TRX",
            "timeframe": "1h",
            "direction": "UPSIDE",
            "sample": 18,
            "read_quality": "Good",
            "best_bar": 8,
            "fade_after_bar": 0,
            "best_zone_low_pct": 0.2,
            "best_zone_high_pct": 0.6,
            "normal_pullback_pct": 0.25,
            "caution_pullback_pct": 0.6,
            "reference_price": 0.32,
            "reference_price_label": "latest close",
            "price_zone_label": "$0.320640 - $0.321920",
            "pullback_price_label": "$0.319200",
            "caution_price_label": "$0.318080",
            "archive_check_sample": 18,
            "zone_hit_rate_pct": 72.0,
            "clean_path_rate_pct": 61.0,
            "path_conflict": True,
        }

        items = _expected_path_kpi_items(snapshot)
        shakeout = next(item for item in items if item["label"] == "Normal Shakeout")
        history = next(item for item in items if item["label"] == "History Depth")

        self.assertIn("caution below $0.318080", shakeout["subtext"])
        self.assertEqual(history["badge_text"], "72% reached zone")
        self.assertIn("61% clean route", history["subtext"])
        self.assertIn("alternate read is close", history["subtext"])

    def test_expected_path_reference_price_tries_usdt_pair_for_base_symbol(self) -> None:
        calls: list[str] = []

        def fake_fetch(symbol: str, timeframe: str, limit: int) -> pd.DataFrame:
            calls.append(symbol)
            if symbol == "H/USDT":
                return pd.DataFrame({"close": [0.151, 0.154]})
            return pd.DataFrame()

        price, label = _fetch_expected_path_reference_price(fake_fetch, "H", "1h")

        self.assertEqual(price, 0.154)
        self.assertEqual(label, "latest close")
        self.assertIn("H", calls)
        self.assertIn("H/USDT", calls)

    def test_expected_path_projection_hides_when_sample_is_too_thin(self) -> None:
        events = pd.DataFrame(
            [
                {
                    "signal_key": "sig-1",
                    "symbol": "ETH",
                    "timeframe": "1h",
                    "direction": "Upside",
                    "status": "RESOLVED",
                    "price": 100.0,
                    "event_time": "2026-04-28T10:00:00Z",
                }
            ]
        )
        windows = pd.DataFrame(
            [
                {
                    "signal_key": "sig-1",
                    "bars_ahead": 4,
                    "directional_return_pct": 1.0,
                    "adverse_excursion_pct": 0.2,
                }
            ]
        )

        snapshot = _build_expected_path_projection(
            df_events=events,
            df_forward_windows=windows,
            symbol_filter="ETH",
            timeframe_filter="1h",
            min_samples=8,
            now="2026-04-28T12:00:00Z",
        )

        self.assertFalse(snapshot["available"])

    def test_expected_path_building_body_reports_checkpoint_gap(self) -> None:
        events = pd.DataFrame(
            [
                {
                    "signal_key": f"algo-{idx}",
                    "symbol": "ALGO",
                    "timeframe": "1h",
                    "direction": "Upside",
                    "status": "RESOLVED",
                }
                for idx in range(4)
            ]
        )
        windows = pd.DataFrame({"signal_key": ["algo-0", "algo-1"]})

        body = _expected_path_building_body_html(
            symbol_filter="ALGO",
            timeframe_filter="1h",
            df_events=events,
            df_forward_windows=windows,
            min_samples=8,
        )

        self.assertIn("ALGO path is still building", body)
        self.assertIn("2/8", body)
        self.assertIn("4</b> completed signals", body)

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

    def test_build_coin_timeframe_intelligence_bundle_uses_per_timeframe_frames(self) -> None:
        timeframe_frames = [
            {
                "timeframe": "5m",
                "events": pd.DataFrame(
                    [
                        {
                            "signal_key": "a1",
                            "symbol": "BTC",
                            "timeframe": "5m",
                            "direction": "Upside",
                            "status": "RESOLVED",
                            "directional_return_pct": 0.4,
                        }
                    ]
                ),
                "windows": pd.DataFrame(
                    [
                        {"signal_key": "a1", "bars_ahead": 2, "directional_return_pct": 0.4, "adverse_excursion_pct": 0.1}
                    ]
                ),
            },
            {
                "timeframe": "1h",
                "events": pd.DataFrame(
                    [
                        {
                            "signal_key": "b1",
                            "symbol": "BTC",
                            "timeframe": "1h",
                            "direction": "Downside",
                            "status": "RESOLVED",
                            "directional_return_pct": 1.2,
                        }
                    ]
                ),
                "windows": pd.DataFrame(
                    [
                        {"signal_key": "b1", "bars_ahead": 6, "directional_return_pct": 1.2, "adverse_excursion_pct": 0.2}
                    ]
                ),
            },
        ]

        def _fake_summary(df_scope: pd.DataFrame, _group_field: str) -> pd.DataFrame:
            timeframe = str(df_scope.iloc[0]["timeframe"]).strip().lower()
            if timeframe == "5m":
                return pd.DataFrame(
                    [
                        {
                            "timeframe": "5m",
                            "Resolved": 1,
                            "FollowThroughPct": 40.0,
                            "AvgDirReturnPct": 0.4,
                        }
                    ]
                )
            return pd.DataFrame(
                [
                    {
                        "timeframe": "1h",
                        "Resolved": 1,
                        "FollowThroughPct": 70.0,
                        "AvgDirReturnPct": 1.2,
                    }
                ]
            )

        def _fake_hold_window_intelligence(df_scope: pd.DataFrame, _df_windows: pd.DataFrame) -> dict[str, object]:
            if df_scope.empty:
                return {"available": False, "resolved_signals": 0}
            timeframe = str(df_scope.iloc[0]["timeframe"]).strip().lower()
            direction = str(df_scope.iloc[0]["direction"]).strip()
            best_bar = 2 if timeframe == "5m" else 6
            if direction == "Upside" and timeframe == "5m":
                return {
                    "available": True,
                    "resolved_signals": 1,
                    "best_bar": best_bar,
                    "best_label": f"around {best_bar} bars",
                    "sample": 1,
                }
            if direction == "Downside" and timeframe == "1h":
                return {
                    "available": True,
                    "resolved_signals": 1,
                    "best_bar": best_bar,
                    "best_label": f"around {best_bar} bars",
                    "sample": 1,
                }
            return {"available": False, "resolved_signals": int(len(df_scope))}

        display_df, summary_df, hold_rows = _build_coin_timeframe_intelligence_bundle(
            timeframe_frames=timeframe_frames,
            build_signal_cohort_summary=_fake_summary,
            build_hold_window_intelligence=_fake_hold_window_intelligence,
        )
        self.assertEqual(list(display_df["Timeframe"]), ["5M", "1H"])
        self.assertIn("Completed", display_df.columns)
        self.assertNotIn("Resolved", display_df.columns)
        self.assertEqual(list(summary_df["timeframe"]), ["5m", "1h"])
        self.assertEqual(display_df.iloc[0]["Upside Hold"], "Best at 2 bars")
        self.assertEqual(display_df.iloc[1]["Downside Hold"], "Best at 6 bars")
        self.assertEqual([row["Timeframe"] for row in hold_rows], ["5M", "1H"])
        self.assertEqual(hold_rows[0]["Upside Hold"], "Best at 2 bars")
        self.assertEqual(hold_rows[0]["Downside Hold"], "—")
        self.assertEqual(hold_rows[1]["Upside Hold"], "—")
        self.assertEqual(hold_rows[1]["Downside Hold"], "Best at 6 bars")

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

    def test_display_trade_direction_uses_upside_downside_labels(self) -> None:
        self.assertEqual(_display_trade_direction("LONG"), "Upside")
        self.assertEqual(_display_trade_direction("Short"), "Downside")
        self.assertEqual(_display_trade_direction("Upside"), "Upside")
        self.assertEqual(_display_trade_direction(""), "")


if __name__ == "__main__":
    unittest.main()
