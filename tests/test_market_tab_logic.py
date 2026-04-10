import unittest
from unittest.mock import patch
from types import SimpleNamespace

import pandas as pd

from tabs.market_scan_helpers import (
    _actionable_analysis_batch_size,
    _actionable_direction_include,
    _actionable_frame_hunt_score,
    _actionable_setup_score,
    _actionable_tactical_candidate_score,
    _actionable_universe_movement_score,
    _candidate_scan_symbols,
    _initial_scan_batch_size,
    _initial_scan_symbols,
    _next_refill_candidate_batch,
    _next_scan_pool_target,
    _scan_candidate_pool_size,
)
from tabs.market_tab import (
    _alert_lane_label,
    _alert_archive_label,
    _actionable_market_result_priority_key,
    _compress_market_alerts_for_display,
    _rank_market_alerts_by_archive,
    _audit_scan_summary_lines,
    _ai_fallback_note,
    _build_custom_scan_universe,
    _build_market_cap_map,
    _canonical_pair_base,
    _coingecko_coin_id_fallback_available,
    _coingecko_coin_id_fallback_reason,
    _confidence_badge,
    _coingecko_coin_id_unavailable_message,
    _custom_watchlist_fallback_coin_id,
    _custom_watchlist_enrichment_coverage,
    _custom_watchlist_missing_status,
    _delta_fallback_symbol,
    _direction_fetch_symbol,
    _filter_scan_symbols,
    _fetch_market_scan_ohlcv,
    _healthy_empty_seen_for_sig,
    _healthy_empty_registry,
    _last_good_registry,
    _last_good_snapshot_for_sig,
    _normalize_custom_bases,
    _consume_market_custom_clear,
    _market_scan_signature,
    _market_data_mode,
    _next_universe_fetch_n,
    _pair_provenance_label,
    _remember_display_scan_state,
    _resolve_notice_scan_state,
    _resolve_display_scan_state,
    _queue_market_custom_clear,
    _merge_market_cap_maps,
    _market_lead_breadth_component,
    _market_lead_snapshot,
    _market_result_priority_key,
    _market_result_priority_key_for_mode,
    _market_hidden_meta_cols,
    _pick_best_scalp_opportunity,
    _scan_universe_notice,
    _pick_confidence_leader,
    _prepare_scan_market_enrichment,
    _prepare_closed_frame,
    _remember_last_good_snapshot,
    _remember_healthy_empty_sig,
    _scan_attempt_is_stale,
    _should_rescan_market,
    _should_use_major_fallback,
    _should_use_cached_scan,
    _fetch_ticker_delta_once,
    _extract_ai_verdict,
    _extract_confidence_label,
    _setup_confirm_priority,
    _setup_status_summary,
    _share_line,
    _share_line_against_total,
    _sync_market_cap_cells,
    _underfilled_universe_message,
)
from core.symbols import is_stable_base_symbol
from threading import Lock


class MarketTabLogicTests(unittest.TestCase):
    def test_alert_lane_label_uses_trader_facing_terms(self):
        self.assertEqual(
            _alert_lane_label(SimpleNamespace(severity="HIGH", tone="negative")),
            "Stand Aside",
        )
        self.assertEqual(
            _alert_lane_label(SimpleNamespace(severity="MEDIUM", tone="positive")),
            "Action",
        )
        self.assertEqual(
            _alert_lane_label(SimpleNamespace(severity="INFO", tone="warning")),
            "Context",
        )

    def test_compress_market_alerts_for_display_keeps_primary_then_context_summary(self):
        alerts = [
            SimpleNamespace(alert_key="TRADE_GATE", severity="HIGH", tone="negative", title="No-Trade gate active", note=""),
            SimpleNamespace(alert_key="MARKET_LEAD", severity="MEDIUM", tone="positive", title="Upside pressure is building", note=""),
            SimpleNamespace(alert_key="SESSION_FIT", severity="INFO", tone="positive", title="Current session has been supportive", note=""),
            SimpleNamespace(alert_key="FLOW_PROXY", severity="INFO", tone="positive", title="Shorts Crowded in BTC", note=""),
        ]
        display = _compress_market_alerts_for_display(alerts, max_items=2)
        self.assertEqual(len(display), 2)
        self.assertEqual(display[0].alert_key, "TRADE_GATE")
        self.assertEqual(display[1].alert_key, "MARKET_LEAD")

        display = _compress_market_alerts_for_display(alerts[1:], max_items=2)
        self.assertEqual(len(display), 2)
        self.assertEqual(display[0].alert_key, "MARKET_LEAD")
        self.assertEqual(display[1].alert_key, "CONTEXT_STACK")

    def test_rank_market_alerts_by_archive_keeps_protected_alerts_first_and_reorders_info_alerts(self):
        alerts = [
            SimpleNamespace(alert_key="TRADE_GATE"),
            SimpleNamespace(alert_key="SESSION_FIT"),
            SimpleNamespace(alert_key="MARKET_LEAD"),
        ]
        summary_df = pd.DataFrame(
            [
                {
                    "Primary Alert": "Trade Gate",
                    "Resolved": 8,
                    "FollowThroughPct": 48.0,
                    "ClosedTradeCount": 3,
                    "ActualWinRatePct": 50.0,
                },
                {
                    "Primary Alert": "Session Fit",
                    "Resolved": 8,
                    "FollowThroughPct": 41.0,
                    "ClosedTradeCount": 3,
                    "ActualWinRatePct": 42.0,
                },
                {
                    "Primary Alert": "Market Lead",
                    "Resolved": 8,
                    "FollowThroughPct": 66.0,
                    "ClosedTradeCount": 3,
                    "ActualWinRatePct": 61.0,
                },
            ]
        )
        with patch("tabs.market_tab.build_alert_effectiveness_summary", return_value=summary_df):
            ranked = _rank_market_alerts_by_archive(alerts, pd.DataFrame({"symbol": ["BTC"]}))
        self.assertEqual([alert.alert_key for alert in ranked], ["TRADE_GATE", "MARKET_LEAD", "SESSION_FIT"])

    def test_alert_archive_label_maps_known_keys(self):
        self.assertEqual(_alert_archive_label("MARKET_LEAD"), "Market Lead")
        self.assertEqual(_alert_archive_label("SESSION_FIT"), "Session Fit")

    def test_market_lead_breadth_component_needs_real_participation(self):
        rows = [
            {"__emerging_label": "Emerging Upside"},
            {"__emerging_label": "Emerging Upside"},
        ]
        score, upside, downside = _market_lead_breadth_component(rows)
        self.assertEqual((score, upside, downside), (0.0, 2, 0))

        rows.append({"__emerging_label": "Emerging Downside"})
        score, upside, downside = _market_lead_breadth_component(rows)
        self.assertGreater(score, 0.0)
        self.assertEqual((upside, downside), (2, 1))

    def test_market_lead_snapshot_detects_upside_pressure_building(self):
        snapshot = _market_lead_snapshot(
            produced_rows=[
                {"__emerging_label": "Emerging Upside"},
                {"__emerging_label": "Emerging Upside"},
                {"__emerging_label": "Emerging Upside"},
                {"__emerging_label": "Emerging Downside"},
            ],
            delta_mcap=2.2,
            btc_change=-0.3,
            eth_change=0.1,
            btc_dom=53.0,
            eth_dom=12.2,
            custom_mode_active=False,
        )
        self.assertEqual(snapshot.state, "UPSIDE")
        self.assertEqual(snapshot.label, "Upside")
        self.assertGreater(snapshot.score, 62.0)

    def test_market_result_priority_prefers_executable_supported_setup(self):
        stronger = {
            "Coin": "SOL",
            "__action_raw": "✅ ENTER (Trend+AI)",
            "__confidence_val": 82.0,
            "__adaptive_edge_score": 67.0,
            "__archive_guardrail_penalty": 0.0,
            "__rr_val": 2.1,
            "__ai_confidence_val": 71.0,
            "__risk_unit_fraction": 1.0,
            "__mcap_val": 50_000_000_000,
        }
        weaker = {
            "Coin": "ADA",
            "__action_raw": "✅ ENTER (Trend+AI)",
            "__confidence_val": 84.0,
            "__adaptive_edge_score": 69.0,
            "__archive_guardrail_penalty": 5.8,
            "__rr_val": 2.2,
            "__ai_confidence_val": 74.0,
            "__risk_unit_fraction": 0.25,
            "__mcap_val": 20_000_000_000,
        }
        ranked = sorted([weaker, stronger], key=_market_result_priority_key)
        self.assertEqual(ranked[0]["Coin"], "SOL")

    def test_market_hidden_meta_cols_include_render_contract_fields(self):
        df_columns = [
            "Coin",
            "Scalp Opportunity",
            "__scalp_reason_short",
            "__scalp_reason_text",
            "__scalp_display_state",
            "__adaptive_edge_note",
            "__setup_calibration_note",
            "__delta_note",
        ]
        hidden_cols = _market_hidden_meta_cols(df_columns, ["Coin", "Scalp Opportunity"])
        self.assertEqual(
            hidden_cols,
            [
                "__adaptive_edge_note",
                "__delta_note",
                "__scalp_display_state",
                "__scalp_reason_short",
                "__scalp_reason_text",
                "__setup_calibration_note",
            ],
        )

    def test_pick_confidence_leader_prefers_live_setup_over_skip(self):
        df = pd.DataFrame(
            [
                {
                    "Coin": "BTC",
                    "__confidence_val": 92.0,
                    "__action_raw": "SKIP",
                    "Setup Confirm": "SKIP",
                    "Direction": "Upside",
                    "__ai_confidence_val": 65.0,
                },
                {
                    "Coin": "ETH",
                    "__confidence_val": 81.0,
                    "__action_raw": "PROBE",
                    "Setup Confirm": "PROBE",
                    "Direction": "Upside",
                    "__ai_confidence_val": 52.0,
                },
            ]
        )
        coin, score, sub = _pick_confidence_leader(df)
        self.assertEqual(coin, "ETH")
        self.assertEqual(score, 81.0)
        self.assertIn("Setup: PROBE", sub)

    def test_pick_confidence_leader_falls_back_to_skip_when_only_skip_exists(self):
        df = pd.DataFrame(
            [
                {
                    "Coin": "BTC",
                    "__confidence_val": 92.0,
                    "__action_raw": "SKIP",
                    "Setup Confirm": "SKIP",
                    "Direction": "Upside",
                    "__ai_confidence_val": 65.0,
                }
            ]
        )
        coin, score, sub = _pick_confidence_leader(df)
        self.assertEqual(coin, "BTC")
        self.assertEqual(score, 92.0)
        self.assertIn("Setup: SKIP", sub)

    def test_pick_best_scalp_opportunity_prefers_live_over_conditional(self):
        df = pd.DataFrame(
            [
                {
                    "Coin": "BTC",
                    "Scalp Opportunity": "Upside",
                    "__scalp_display_state": "CONDITIONAL",
                    "__scalp_reason_short": "No-Trade",
                    "R:R": "2.50*",
                    "__action_raw": "PROBE",
                    "Setup Confirm": "PROBE",
                    "__confidence_val": 70.0,
                },
                {
                    "Coin": "ETH",
                    "Scalp Opportunity": "Downside",
                    "__scalp_display_state": "LIVE",
                    "__scalp_reason_short": "",
                    "R:R": "1.40",
                    "__action_raw": "PROBE",
                    "Setup Confirm": "PROBE",
                    "__confidence_val": 68.0,
                },
            ]
        )
        head, sub = _pick_best_scalp_opportunity(df)
        self.assertEqual(head, "ETH (1.40)")
        self.assertIn("Live", sub)
        self.assertIn("Live: 1", sub)
        self.assertIn("Conditional: 1", sub)

    def test_pick_best_scalp_opportunity_falls_back_to_best_conditional(self):
        df = pd.DataFrame(
            [
                {
                    "Coin": "BTC",
                    "Scalp Opportunity": "Upside",
                    "__scalp_display_state": "CONDITIONAL",
                    "__scalp_reason_short": "No-Trade",
                    "R:R": "1.10*",
                    "__action_raw": "WATCH",
                    "Setup Confirm": "WATCH",
                    "__confidence_val": 66.0,
                },
                {
                    "Coin": "SOL",
                    "Scalp Opportunity": "Downside",
                    "__scalp_display_state": "CONDITIONAL",
                    "__scalp_reason_short": "Archive",
                    "R:R": "1.80*",
                    "__action_raw": "PROBE",
                    "Setup Confirm": "PROBE",
                    "__confidence_val": 71.0,
                },
            ]
        )
        head, sub = _pick_best_scalp_opportunity(df)
        self.assertEqual(head, "SOL (1.80)")
        self.assertIn("Conditional", sub)
        self.assertIn("Archive", sub)
        self.assertIn("Live: 0", sub)
        self.assertIn("Conditional: 2", sub)

    def test_audit_scan_summary_lines_show_attempted_vs_displayed(self):
        lines = _audit_scan_summary_lines(
            displayed_rows=50,
            attempted_count=87,
            produced_count=58,
            skipped_count=29,
            ranked_out_count=8,
            source_label="LIVE",
        )
        self.assertEqual(lines[0], "**Rows shown:** `50`")
        self.assertIn("attempted `87`", lines[1])
        self.assertIn("produced `58`", lines[1])
        self.assertIn("skipped `29`", lines[1])
        self.assertIn("ranked out `8`", lines[1])

    def test_coin_id_fallback_availability_detects_missing_dependency_marker(self):
        def _missing(*_args, **_kwargs):
            return None

        _missing._codex_missing_dep = True
        _missing._codex_missing_dep_reason = "dependency injection missing at app boot"

        self.assertFalse(_coingecko_coin_id_fallback_available(_missing))
        self.assertTrue(_coingecko_coin_id_fallback_available(lambda *_args, **_kwargs: None))
        self.assertEqual(
            _coingecko_coin_id_fallback_reason(_missing),
            "dependency injection missing at app boot",
        )
        self.assertEqual(_coingecko_coin_id_fallback_reason(lambda *_args, **_kwargs: None), "")

    def test_coin_id_unavailable_message_includes_reason_when_present(self):
        self.assertEqual(
            _coingecko_coin_id_unavailable_message("dependency injection missing at app boot"),
            "no exchange OHLCV data; CoinGecko fallback unavailable (dependency injection missing at app boot)",
        )
        self.assertEqual(
            _coingecko_coin_id_unavailable_message(""),
            "no exchange OHLCV data; CoinGecko fallback unavailable",
        )

    def test_extract_ai_verdict_strips_votes_and_degraded_marker(self):
        self.assertEqual(_extract_ai_verdict("Upside (2/3)"), "Upside")
        self.assertEqual(_extract_ai_verdict("Neutral (0/3) *"), "Neutral")

    def test_extract_confidence_label_reads_badge_suffix(self):
        self.assertEqual(_extract_confidence_label("81% (Medium)"), "Medium")
        self.assertEqual(_extract_confidence_label("44% (Very Low)"), "Very Low")

    def test_share_line_formats_counts_in_requested_order(self):
        line = _share_line({"Watch": 6, "Ready": 3, "Skip": 1}, ["Ready", "Watch", "Skip"])
        self.assertEqual(line, "Ready: 3 (30%) • Watch: 6 (60%) • Skip: 1 (10%)")

    def test_share_line_against_total_handles_sparse_counts(self):
        line = _share_line_against_total({"Emerging Upside": 2}, ["Emerging Upside", "Emerging Downside"], 10)
        self.assertEqual(line, "Emerging Upside: 2 (20%) • Emerging Downside: 0 (0%)")

    def test_shared_stable_base_helper_recognizes_usd1(self):
        self.assertTrue(is_stable_base_symbol("USD1"))
        self.assertTrue(is_stable_base_symbol("usd1"))
        self.assertTrue(is_stable_base_symbol("USDG"))
        self.assertTrue(is_stable_base_symbol("usdg"))

    def test_queue_market_custom_clear_marks_pending_and_clears_applied_watchlist(self):
        state = {
            "market_custom_coin_input": "BTC,ETH",
            "market_custom_bases_applied": ["BTC", "ETH"],
        }
        _queue_market_custom_clear(state)
        self.assertTrue(state["market_clear_custom_pending"])
        self.assertEqual(state["market_custom_bases_applied"], [])
        self.assertEqual(state["market_custom_coin_input"], "BTC,ETH")

    def test_consume_market_custom_clear_removes_input_before_widget_creation(self):
        state = {
            "market_clear_custom_pending": True,
            "market_custom_coin_input": "BTC,ETH",
            "market_custom_bases_applied": ["BTC", "ETH"],
        }
        _consume_market_custom_clear(state)
        self.assertNotIn("market_clear_custom_pending", state)
        self.assertNotIn("market_custom_coin_input", state)
        self.assertEqual(state["market_custom_bases_applied"], [])

    def test_candidate_scan_symbols_excludes_usd1_when_stable_filter_enabled(self):
        out = _candidate_scan_symbols(
            usdt_symbols=["BTC/USDT", "USD1/USDT", "USDG/USDT", "ETH/USDT"],
            market_rows=[],
            exclude_stables=True,
            custom_bases_applied=[],
        )
        self.assertEqual(out, ["BTC/USDT", "ETH/USDT"])

    def test_custom_watchlist_missing_status_reports_skipped_symbols(self):
        out = _custom_watchlist_missing_status(
            ["COS", "BANANAS31", "TOWNS"],
            [{"Coin": "BANANAS31"}],
            [("COS/USDT", "no OHLCV data"), ("TOWNS/USDT", "insufficient candles (23)")],
        )
        self.assertEqual(
            out,
            [("COS", "no OHLCV data"), ("TOWNS", "insufficient candles (23)")],
        )

    def test_custom_watchlist_missing_status_marks_unresolved_coin_id_fallback(self):
        out = _custom_watchlist_missing_status(
            ["COS", "BANANAS31"],
            [{"Coin": "BANANAS31"}],
            [],
        )
        self.assertEqual(out, [("COS", "no exchange pair; coin-id unresolved for fallback")])

    def test_custom_watchlist_missing_status_marks_unavailable_coin_id_fallback(self):
        out = _custom_watchlist_missing_status(
            ["COS", "BANANAS31"],
            [{"Coin": "BANANAS31"}],
            [],
            coin_id_map={"COS": "contentos"},
            coingecko_coin_id_fallback_available=False,
            coingecko_coin_id_fallback_reason="dependency injection missing at app boot",
        )
        self.assertEqual(
            out,
            [
                (
                    "COS",
                    "no exchange OHLCV data; CoinGecko fallback unavailable "
                    "(dependency injection missing at app boot)",
                )
            ],
        )

    def test_custom_watchlist_missing_status_marks_empty_coin_id_fallback(self):
        out = _custom_watchlist_missing_status(
            ["COS", "BANANAS31"],
            [{"Coin": "BANANAS31"}],
            [],
            coin_id_map={"COS": "contentos"},
            coingecko_coin_id_fallback_available=True,
        )
        self.assertEqual(out, [("COS", "no exchange OHLCV data; CoinGecko fallback returned empty")])

    def test_custom_watchlist_fallback_coin_id_only_applies_in_custom_mode(self):
        coin_id_map = {"COS": "contentos", "BANANAS31": "banana-gun"}
        self.assertEqual(
            _custom_watchlist_fallback_coin_id(
                "COS/USDT",
                custom_mode_active=True,
                coin_id_map=coin_id_map,
            ),
            "contentos",
        )
        self.assertIsNone(
            _custom_watchlist_fallback_coin_id(
                "COS/USDT",
                custom_mode_active=False,
                coin_id_map=coin_id_map,
            )
        )

    def test_fetch_market_scan_ohlcv_prefers_exchange_before_coin_id_fallback(self):
        frame = pd.DataFrame({"close": [1.0]})

        def _exchange_fetch(symbol, timeframe, limit=0):
            return frame

        def _coin_id_fetch(_coin_id, _timeframe, limit=0):
            raise AssertionError("coin-id fallback should not run when exchange data exists")

        out = _fetch_market_scan_ohlcv(
            fetch_ohlcv=_exchange_fetch,
            fetch_coingecko_ohlcv_by_coin_id=_coin_id_fetch,
            fetch_lock=Lock(),
            symbol="COS/USDT",
            timeframe="1h",
            limit=120,
            fallback_coin_id="contentos",
        )
        self.assertIs(out, frame)

    def test_build_custom_scan_universe_builds_usdt_pairs_from_custom_bases(self):
        rows = [{"symbol": "COS", "market_cap": 100, "id": "contentos"}]

        def _market_rows_fetch(symbols, vs_currency="usd"):
            self.assertEqual(tuple(symbols), ("COS", "USD1"))
            self.assertEqual(vs_currency, "usd")
            return rows

        unique_market_data, mcap_map, usdt_symbols, candidate_symbol_pool = _build_custom_scan_universe(
            custom_bases_applied=["COS", "USD1"],
            get_market_cap_rows_for_symbols=_market_rows_fetch,
            exclude_stables=True,
            scan_pool_n=10,
        )
        self.assertEqual(usdt_symbols, ["COS/USDT", "USD1/USDT"])
        self.assertEqual(candidate_symbol_pool, ["COS/USDT"])
        self.assertEqual(len(unique_market_data), 1)
        self.assertEqual(mcap_map["COS"], 100)

    def test_alias_aware_symbol_filter_keeps_xbt_pair_for_btc_market_row(self):
        symbols = ["XBT/USD", "ETH/USD"]
        market_rows = [{"symbol": "btc"}, {"symbol": "eth"}]
        out = _filter_scan_symbols(symbols, market_rows)
        self.assertEqual(out, ["XBT/USD", "ETH/USD"])

    def test_canonical_pair_base_maps_xbt_to_btc(self):
        self.assertEqual(_canonical_pair_base("XBT/USD"), "BTC")
        self.assertEqual(_canonical_pair_base("BTC/USDT"), "BTC")

    def test_market_cap_map_uses_canonical_base(self):
        out = _build_market_cap_map(
            [
                {"symbol": "btc", "market_cap": 100},
                {"symbol": "XBT", "market_cap": 200},
            ]
        )
        self.assertEqual(out["BTC"], 200)

    def test_scan_market_enrichment_keeps_highest_market_cap_across_duplicate_symbols(self):
        unique_rows, mcap_map = _prepare_scan_market_enrichment(
            [
                {"id": "foo-small", "symbol": "foo", "market_cap": 100},
                {"id": "foo-large", "symbol": "foo", "market_cap": 200},
            ]
        )
        self.assertEqual(len(unique_rows), 1)
        self.assertEqual(mcap_map["FOO"], 200)

    def test_pair_provenance_label_shows_actual_exchange_pair(self):
        self.assertEqual(
            _pair_provenance_label("BTC/USDT", "XBT/USD", "exchange"),
            "XBT/USD",
        )

    def test_pair_provenance_label_marks_coingecko_fallback(self):
        self.assertEqual(
            _pair_provenance_label("BTC/USDT", "BTC/USDT", "coingecko"),
            "BTC/USDT (CoinGecko fallback)",
        )

    def test_delta_fallback_symbol_uses_actual_exchange_pair_only(self):
        self.assertEqual(
            _delta_fallback_symbol("BTC/USDT", "XBT/USD", "exchange"),
            "XBT/USD",
        )
        self.assertIsNone(_delta_fallback_symbol("BTC/USDT", "BTC/USDT", "coingecko"))

    def test_market_data_mode_marks_major_fallback_as_degraded(self):
        self.assertEqual(
            _market_data_mode(has_market_rows=True, used_major_fallback=True),
            "MAJOR FALLBACK MODE",
        )
        self.assertEqual(
            _market_data_mode(has_market_rows=False, used_major_fallback=False),
            "EXCHANGE-ONLY MODE",
        )
        self.assertEqual(
            _market_data_mode(
                has_market_rows=False,
                used_major_fallback=False,
                custom_mode_active=True,
            ),
            "CUSTOM WATCHLIST MODE (EXCHANGE-ONLY)",
        )
        self.assertEqual(
            _market_data_mode(
                has_market_rows=True,
                used_major_fallback=False,
                custom_mode_active=True,
            ),
            "CUSTOM WATCHLIST MODE",
        )
        self.assertEqual(
            _market_data_mode(
                has_market_rows=True,
                used_major_fallback=False,
                custom_mode_active=True,
                custom_watchlist_enriched_count=1,
                custom_watchlist_total_count=2,
            ),
            "CUSTOM WATCHLIST MODE (PARTIAL ENRICHMENT)",
        )

    def test_setup_confirm_priority_orders_enter_classes_strictly(self):
        self.assertGreater(_setup_confirm_priority("TREND+AI"), _setup_confirm_priority("TREND-led"))
        self.assertGreater(_setup_confirm_priority("TREND-led"), _setup_confirm_priority("AI-led"))
        self.assertGreater(_setup_confirm_priority("AI-led"), _setup_confirm_priority("PROBE"))
        self.assertGreater(_setup_confirm_priority("PROBE"), _setup_confirm_priority("WATCH"))
        self.assertGreater(_setup_confirm_priority("WATCH"), _setup_confirm_priority("SKIP"))

    def test_market_result_priority_key_prefers_trend_plus_ai_before_other_enter_classes(self):
        rows = [
            {
                "Coin": "SOL",
                "__action_raw": "🟡 ENTER (Trend-Led)",
                "__confidence_val": 90.0,
                "__ai_confidence_val": 40.0,
                "__mcap_val": 500,
            },
            {
                "Coin": "BTC",
                "__action_raw": "✅ ENTER (Trend+AI)",
                "__confidence_val": 70.0,
                "__ai_confidence_val": 80.0,
                "__mcap_val": 100,
            },
            {
                "Coin": "ETH",
                "__action_raw": "🟡 ENTER (AI-Led)",
                "__confidence_val": 95.0,
                "__ai_confidence_val": 95.0,
                "__mcap_val": 1000,
            },
        ]
        ordered = sorted(rows, key=_market_result_priority_key)
        self.assertEqual([row["Coin"] for row in ordered], ["BTC", "SOL", "ETH"])

    def test_market_result_priority_key_prefers_confidence_when_available(self):
        rows = [
            {
                "Coin": "BTC",
                "__action_raw": "TREND+AI",
                "__confidence_val": 62.0,
                "__ai_confidence_val": 90.0,
                "__mcap_val": 100,
            },
            {
                "Coin": "ETH",
                "__action_raw": "TREND+AI",
                "__confidence_val": 88.0,
                "__ai_confidence_val": 40.0,
                "__mcap_val": 100,
            },
        ]
        ordered = sorted(rows, key=_market_result_priority_key)
        self.assertEqual([row["Coin"] for row in ordered], ["ETH", "BTC"])

    def test_market_result_priority_key_uses_ai_confidence_as_visible_tiebreaker(self):
        rows = [
            {
                "Coin": "BTC",
                "__action_raw": "TREND+AI",
                "__confidence_val": 82.0,
                "__ai_confidence_val": 58.0,
                "__mcap_val": 100,
            },
            {
                "Coin": "ETH",
                "__action_raw": "TREND+AI",
                "__confidence_val": 82.0,
                "__ai_confidence_val": 76.0,
                "__mcap_val": 100,
            },
        ]
        ordered = sorted(rows, key=_market_result_priority_key)
        self.assertEqual([row["Coin"] for row in ordered], ["ETH", "BTC"])

    def test_actionable_setup_score_rewards_clean_execution_structure(self):
        stronger = _actionable_setup_score(
            timeframe="1h",
            execution_structure_quality=84.0,
            execution_trend_quality=80.0,
            execution_regime_quality=72.0,
            execution_location_quality=78.0,
            trend_led_score=76.0,
            ai_led_score=66.0,
            rr_ratio=1.9,
            adx_val=23.0,
            delta_pct=1.4,
            volatility_label="– Moderate",
            vwap_label="🟢 Above",
            bollinger_bias="→ Near Bottom",
            signal_direction="UPSIDE",
            volume_spike=True,
            spike_dir="UP",
        )
        weaker = _actionable_setup_score(
            timeframe="1h",
            execution_structure_quality=60.0,
            execution_trend_quality=58.0,
            execution_regime_quality=55.0,
            execution_location_quality=52.0,
            trend_led_score=44.0,
            ai_led_score=40.0,
            rr_ratio=1.2,
            adx_val=14.0,
            delta_pct=4.9,
            volatility_label="▲ High",
            vwap_label="🔴 Below",
            bollinger_bias="🔴 Overbought",
            signal_direction="UPSIDE",
            volume_spike=False,
            spike_dir="",
        )
        self.assertGreater(stronger, weaker)

    def test_actionable_frame_hunt_score_prefers_fresh_aligned_pressure(self):
        def _frame(closes, volumes):
            opens = [closes[0], *closes[:-1]]
            highs = [max(o, c) * 1.003 for o, c in zip(opens, closes)]
            lows = [min(o, c) * 0.997 for o, c in zip(opens, closes)]
            return pd.DataFrame(
                {
                    "open": opens,
                    "high": highs,
                    "low": lows,
                    "close": closes,
                    "volume": volumes,
                }
            )

        strong_df = _frame(
            [
                100.0, 101.6, 99.9, 102.2, 100.8, 103.0, 101.2, 103.4,
                102.1, 104.0, 103.0, 104.1, 103.2, 104.0, 103.5, 104.2,
                103.8, 104.4, 104.0, 104.6, 104.2, 104.5, 104.4, 104.7,
                104.6, 104.8, 104.7, 104.85, 104.9, 105.0, 105.05, 105.3,
            ],
            [100.0] * 31 + [245.0],
        )
        weak_df = _frame(
            [
                100.0, 100.2, 100.1, 100.3, 100.0, 100.2, 100.1, 100.0,
                99.8, 99.7, 99.6, 99.5, 99.4, 99.3, 99.2, 99.0,
                98.9, 98.8, 98.7, 98.6, 98.55, 98.5, 98.45, 98.4,
                98.35, 98.3, 98.25, 98.2, 98.15, 98.1, 98.0, 97.9,
            ],
            [100.0] * 31 + [68.0],
        )
        self.assertGreater(
            _actionable_frame_hunt_score(
                df_eval=strong_df,
                timeframe="1h",
                direction_filter="Upside",
            ),
            _actionable_frame_hunt_score(
                df_eval=weak_df,
                timeframe="1h",
                direction_filter="Upside",
            ),
        )

    def test_actionable_tactical_candidate_score_surfaces_clean_early_neutral_htf_setup(self):
        score = _actionable_tactical_candidate_score(
            spot_direction="Neutral",
            signal_direction="Upside",
            ai_direction="Neutral",
            ai_agreement=0.42,
            frame_hunt_score=86.0,
            execution_structure_quality=84.0,
            execution_trend_quality=78.0,
            execution_location_quality=76.0,
            rr_ratio=1.9,
            adx_val=21.0,
        )
        self.assertGreaterEqual(score, 72.0)

    def test_actionable_direction_include_allows_strong_tactical_candidate(self):
        self.assertTrue(
            _actionable_direction_include(
                direction_filter="Upside",
                scan_mode="Actionable Setups",
                spot_direction="Neutral",
                signal_direction="Upside",
                tactical_candidate_score=77.0,
            )
        )
        self.assertFalse(
            _actionable_direction_include(
                direction_filter="Upside",
                scan_mode="Broad Market",
                spot_direction="Neutral",
                signal_direction="Upside",
                tactical_candidate_score=77.0,
            )
        )

    def test_actionable_market_result_priority_prefers_context_and_setup_quality(self):
        stronger = {
            "Coin": "DOGE",
            "__action_raw": "PROBE",
            "__actionable_context_score": 74.0,
            "__actionable_setup_score": 81.0,
            "__actionable_archive_score": 0.0,
            "__risk_unit_fraction": 0.25,
            "__confidence_val": 70.0,
            "__adaptive_edge_score": 58.0,
            "__archive_guardrail_penalty": 8.0,
            "__rr_val": 1.8,
            "__ai_confidence_val": 62.0,
        }
        weaker = {
            "Coin": "XRP",
            "__action_raw": "PROBE",
            "__actionable_context_score": 52.0,
            "__actionable_setup_score": 63.0,
            "__actionable_archive_score": 0.0,
            "__risk_unit_fraction": 0.25,
            "__confidence_val": 72.0,
            "__adaptive_edge_score": 60.0,
            "__archive_guardrail_penalty": 8.0,
            "__rr_val": 1.9,
            "__ai_confidence_val": 65.0,
        }
        ordered = sorted([weaker, stronger], key=_actionable_market_result_priority_key)
        self.assertEqual(ordered[0]["Coin"], "DOGE")
        mode_ordered = sorted(
            [weaker, stronger],
            key=lambda row: _market_result_priority_key_for_mode(row, "Actionable Setups"),
        )
        self.assertEqual(mode_ordered[0]["Coin"], "DOGE")

    def test_actionable_market_result_priority_uses_archive_bias_as_tiebreaker(self):
        supportive = {
            "Coin": "SOL",
            "__action_raw": "PROBE",
            "__actionable_context_score": 68.0,
            "__actionable_tactical_score": 76.0,
            "__actionable_setup_score": 77.0,
            "__actionable_archive_score": 4.2,
            "__risk_unit_fraction": 0.25,
            "__confidence_val": 69.0,
            "__adaptive_edge_score": 56.0,
            "__archive_guardrail_penalty": 2.0,
            "__rr_val": 1.8,
            "__ai_confidence_val": 61.0,
        }
        cautious = {
            "Coin": "APT",
            "__action_raw": "PROBE",
            "__actionable_context_score": 68.0,
            "__actionable_tactical_score": 76.0,
            "__actionable_setup_score": 77.0,
            "__actionable_archive_score": -3.8,
            "__risk_unit_fraction": 0.25,
            "__confidence_val": 69.0,
            "__adaptive_edge_score": 56.0,
            "__archive_guardrail_penalty": 2.0,
            "__rr_val": 1.8,
            "__ai_confidence_val": 61.0,
        }
        ordered = sorted([cautious, supportive], key=_actionable_market_result_priority_key)
        self.assertEqual(ordered[0]["Coin"], "SOL")

    def test_direction_fetch_symbol_keeps_canonical_requested_symbol_for_htf_context(self):
        self.assertEqual(_direction_fetch_symbol("BTC/USDT", "XBT/USD", "exchange"), "BTC/USDT")
        self.assertEqual(_direction_fetch_symbol("BTC/USDT", "BTC/USDT", "coingecko"), "BTC/USDT")

    def test_confidence_badge_formats_bucket(self):
        self.assertEqual(_confidence_badge(84.0), "84% (High)")
        self.assertEqual(_confidence_badge(41.0), "41% (Low)")

    def test_ai_fallback_note_surfaces_ml_safety_fallback(self):
        note = _ai_fallback_note({"status": "insufficient_features"})
        self.assertIn("AI fallback active", note)
        self.assertIn("neutral safety output", note)

    def test_setup_status_summary_downgrades_cached_and_degraded_sources(self):
        label, head, sub = _setup_status_summary(
            enter_count=2,
            watch_count=1,
            skip_count=0,
            source_label="CACHED (2026-03-07 10:00:00 UTC)",
        )
        self.assertEqual(label, "Setup Status")
        self.assertEqual(head, "CACHED SETUPS")
        self.assertIn("CACHED READY: 2", sub)
        self.assertIn("PROBE: 0", sub)

        _label, degraded_head, degraded_sub = _setup_status_summary(
            enter_count=1,
            watch_count=2,
            skip_count=3,
            source_label="LIVE (DEGRADED)",
        )
        self.assertEqual(degraded_head, "DEGRADED SETUPS")
        self.assertIn("DEGRADED READY: 1", degraded_sub)
        self.assertIn("PROBE: 0", degraded_sub)

    def test_market_scan_signature_ignores_top_n_in_custom_mode(self):
        first = _market_scan_signature(
            timeframe="1h",
            direction_filter="Both",
            top_n=50,
            exclude_stables=True,
            custom_bases_applied=["DOGE"],
            scan_mode="Actionable Setups",
        )
        second = _market_scan_signature(
            timeframe="1h",
            direction_filter="Both",
            top_n=10,
            exclude_stables=True,
            custom_bases_applied=["DOGE"],
            scan_mode="Broad Market",
        )
        self.assertEqual(first, second)
        self.assertEqual(first, ("1h", "Both", 0, True, ("DOGE",), "Broad Market"))

    def test_market_scan_signature_is_order_and_alias_insensitive_in_custom_mode(self):
        first = _market_scan_signature(
            timeframe="1h",
            direction_filter="Both",
            top_n=50,
            exclude_stables=True,
            custom_bases_applied=["BTC", "ETH"],
            scan_mode="Actionable Setups",
        )
        second = _market_scan_signature(
            timeframe="1h",
            direction_filter="Both",
            top_n=50,
            exclude_stables=True,
            custom_bases_applied=["ETH", "XBT"],
            scan_mode="Broad Market",
        )
        self.assertEqual(first, second)
        self.assertEqual(first, ("1h", "Both", 0, True, ("BTC", "ETH"), "Broad Market"))

    def test_market_scan_signature_tracks_scan_mode_outside_custom_mode(self):
        broad = _market_scan_signature(
            timeframe="1h",
            direction_filter="Both",
            top_n=10,
            exclude_stables=True,
            custom_bases_applied=[],
            scan_mode="Broad Market",
        )
        hunter = _market_scan_signature(
            timeframe="1h",
            direction_filter="Both",
            top_n=10,
            exclude_stables=True,
            custom_bases_applied=[],
            scan_mode="Actionable Setups",
        )
        self.assertNotEqual(broad, hunter)
        self.assertEqual(hunter, ("1h", "Both", 10, True, (), "Actionable Setups"))

    def test_normalize_custom_bases_dedupes_aliases(self):
        out = _normalize_custom_bases(["BTC", "XBT", "eth/usdt", "ETH"])
        self.assertEqual(out, ["BTC", "ETH"])

    def test_custom_watchlist_enrichment_coverage_counts_per_base(self):
        enriched, total = _custom_watchlist_enrichment_coverage(
            ["BTC/USDT", "ETH/USDT", "XBT/USD"],
            {"BTC": 100},
        )
        self.assertEqual((enriched, total), (1, 2))

    def test_prepare_closed_frame_drops_last_candle(self):
        df = pd.DataFrame({"close": range(70), "open": range(70)})
        out = _prepare_closed_frame(df, min_rows=55)
        self.assertIsNotNone(out)
        self.assertEqual(len(out), 69)
        self.assertEqual(int(out["close"].iloc[-1]), 68)

    def test_prepare_closed_frame_preserves_attrs(self):
        df = pd.DataFrame({"close": range(70), "open": range(70)})
        df.attrs["volume_is_24h_aggregate"] = True
        out = _prepare_closed_frame(df, min_rows=55)
        self.assertIsNotNone(out)
        self.assertTrue(bool(out.attrs.get("volume_is_24h_aggregate")))

    def test_scan_attempt_is_stale_after_ttl(self):
        with patch("tabs.market_tab.pd.Timestamp.now", return_value=pd.Timestamp("2026-03-07 10:10:00", tz="UTC")):
            self.assertTrue(_scan_attempt_is_stale("2026-03-07 10:00:00 UTC", 5))
            self.assertFalse(_scan_attempt_is_stale("2026-03-07 10:08:00 UTC", 5))

    def test_should_rescan_market_when_same_signature_scan_is_stale(self):
        with patch("tabs.market_tab.pd.Timestamp.now", return_value=pd.Timestamp("2026-03-07 10:10:00", tz="UTC")):
            self.assertTrue(
                _should_rescan_market(
                    run_scan=False,
                    last_sig=("1h", "Both", 50, True, ()),
                    scan_sig=("1h", "Both", 50, True, ()),
                    has_results_state=True,
                    last_attempt_ts="2026-03-07 10:00:00 UTC",
                    refresh_ttl_minutes=5,
                )
            )
            self.assertFalse(
                _should_rescan_market(
                    run_scan=False,
                    last_sig=("1h", "Both", 50, True, ()),
                    scan_sig=("1h", "Both", 50, True, ()),
                    has_results_state=True,
                    last_attempt_ts="2026-03-07 10:08:00 UTC",
                    refresh_ttl_minutes=5,
                )
            )

    def test_should_rescan_market_uses_short_backoff_for_cached_or_degraded_source(self):
        with patch("tabs.market_tab.pd.Timestamp.now", return_value=pd.Timestamp("2026-03-07 10:10:00", tz="UTC")):
            self.assertFalse(
                _should_rescan_market(
                    run_scan=False,
                    last_sig=("1h", "Both", 50, True, ()),
                    scan_sig=("1h", "Both", 50, True, ()),
                    has_results_state=True,
                    last_attempt_ts="2026-03-07 10:09:45 UTC",
                    refresh_ttl_minutes=5,
                    current_source_label="CACHED (2026-03-07 10:00:00 UTC)",
                )
            )
            self.assertTrue(
                _should_rescan_market(
                    run_scan=False,
                    last_sig=("1h", "Both", 50, True, ()),
                    scan_sig=("1h", "Both", 50, True, ()),
                    has_results_state=True,
                    last_attempt_ts="2026-03-07 10:09:20 UTC",
                    refresh_ttl_minutes=5,
                    current_source_label="CACHED (2026-03-07 10:00:00 UTC)",
                )
            )
            self.assertTrue(
                _should_rescan_market(
                    run_scan=False,
                    last_sig=("1h", "Both", 50, True, ()),
                    scan_sig=("1h", "Both", 50, True, ()),
                    has_results_state=True,
                    last_attempt_ts="2026-03-07 10:09:20 UTC",
                    refresh_ttl_minutes=5,
                    current_source_label="LIVE (DEGRADED)",
                )
            )

    def test_should_use_major_fallback_only_when_liquidity_universe_is_missing(self):
        self.assertTrue(
            _should_use_major_fallback(
                working_symbols=[],
                custom_mode_active=False,
                source_pair_count=0,
                market_row_count=0,
            )
        )
        self.assertFalse(
            _should_use_major_fallback(
                working_symbols=[],
                custom_mode_active=False,
                source_pair_count=12,
                market_row_count=0,
            )
        )
        self.assertFalse(
            _should_use_major_fallback(
                working_symbols=[],
                custom_mode_active=False,
                source_pair_count=0,
                market_row_count=8,
            )
        )

    def test_cache_fallback_is_disabled_for_healthy_empty_scan(self):
        self.assertFalse(
            _should_use_cached_scan(
                prev_results=[{"Coin": "BTC"}],
                cache_sig=("1h", "Both", 50, True, ()),
                scan_sig=("1h", "Both", 50, True, ()),
                cache_ts="2026-03-07 10:00:00 UTC",
                ttl_minutes=15,
                scan_degraded=False,
            )
        )

    def test_cache_fallback_requires_degraded_scan_and_fresh_timestamp(self):
        with patch("tabs.market_tab.pd.Timestamp.now", return_value=pd.Timestamp("2026-03-07 10:10:00", tz="UTC")):
            self.assertTrue(
                _should_use_cached_scan(
                    prev_results=[{"Coin": "BTC"}],
                    cache_sig=("1h", "Both", 50, True, ()),
                    scan_sig=("1h", "Both", 50, True, ()),
                    cache_ts="2026-03-07 10:00:00 UTC",
                    ttl_minutes=15,
                    scan_degraded=True,
                )
            )

    def test_cache_fallback_is_blocked_after_newer_healthy_empty_scan_for_same_signature(self):
        with patch("tabs.market_tab.pd.Timestamp.now", return_value=pd.Timestamp("2026-03-07 10:10:00", tz="UTC")):
            self.assertFalse(
                _should_use_cached_scan(
                    prev_results=[{"Coin": "BTC"}],
                    cache_sig=("1h", "Both", 50, True, ()),
                    scan_sig=("1h", "Both", 50, True, ()),
                    cache_ts="2026-03-07 10:00:00 UTC",
                    ttl_minutes=15,
                    scan_degraded=True,
                    healthy_empty_seen=True,
                )
            )

    def test_healthy_empty_registry_tracks_multiple_signatures(self):
        registry = _healthy_empty_registry(None)
        registry = _remember_healthy_empty_sig(registry, ("1h", "Both", 50, True, ()))
        registry = _remember_healthy_empty_sig(registry, ("4h", "Upside", 20, True, ()))
        self.assertTrue(_healthy_empty_seen_for_sig(registry, ("1h", "Both", 50, True, ())))
        self.assertTrue(_healthy_empty_seen_for_sig(registry, ("4h", "Upside", 20, True, ())))
        self.assertFalse(_healthy_empty_seen_for_sig(registry, ("1d", "Both", 50, True, ())))

    def test_last_good_registry_tracks_multiple_signatures(self):
        registry = _last_good_registry(None)
        registry = _remember_last_good_snapshot(
            registry,
            ("1h", "Both", 50, True, ()),
            [{"Coin": "BTC"}],
            "2026-03-07 10:00:00 UTC",
            "FULL MARKET MODE",
        )
        registry = _remember_last_good_snapshot(
            registry,
            ("4h", "Both", 20, True, ()),
            [{"Coin": "ETH"}],
            "2026-03-07 10:05:00 UTC",
            "EXCHANGE-ONLY MODE",
        )
        one_h = _last_good_snapshot_for_sig(registry, ("1h", "Both", 50, True, ()))
        four_h = _last_good_snapshot_for_sig(registry, ("4h", "Both", 20, True, ()))
        self.assertEqual(one_h["results"], [{"Coin": "BTC"}])
        self.assertEqual(four_h["results"], [{"Coin": "ETH"}])
        self.assertEqual(one_h["mode"], "FULL MARKET MODE")
        self.assertEqual(four_h["mode"], "EXCHANGE-ONLY MODE")

    def test_last_good_registry_seeds_from_legacy_single_snapshot(self):
        registry = _last_good_registry(
            None,
            legacy_sig=("1h", "Both", 50, True, ()),
            legacy_results=[{"Coin": "BTC"}],
            legacy_ts="2026-03-07 10:00:00 UTC",
            legacy_mode="FULL MARKET MODE",
        )
        snap = _last_good_snapshot_for_sig(registry, ("1h", "Both", 50, True, ()))
        self.assertEqual(snap["results"], [{"Coin": "BTC"}])
        self.assertEqual(snap["ts"], "2026-03-07 10:00:00 UTC")

    def test_next_universe_fetch_n_grows_when_filters_underfill_top_n(self):
        self.assertEqual(
            _next_universe_fetch_n(
                50,
                custom_mode_active=False,
                eligible_count=28,
                requested_n=50,
            ),
            100,
        )
        self.assertEqual(
            _next_universe_fetch_n(
                100,
                custom_mode_active=False,
                eligible_count=100,
                requested_n=50,
            ),
            100,
        )

    def test_next_universe_fetch_n_grows_in_exchange_only_mode_when_underfilled(self):
        self.assertEqual(
            _next_universe_fetch_n(
                50,
                custom_mode_active=False,
                eligible_count=35,
                requested_n=50,
            ),
            100,
        )

    def test_scan_candidate_pool_size_adds_non_custom_headroom(self):
        self.assertEqual(
            _scan_candidate_pool_size(50, custom_mode_active=False),
            75,
        )
        self.assertEqual(
            _scan_candidate_pool_size(10, custom_mode_active=False),
            20,
        )
        self.assertEqual(
            _scan_candidate_pool_size(10, custom_mode_active=True),
            10,
        )
        self.assertEqual(
            _scan_candidate_pool_size(10, custom_mode_active=False, scan_mode="Actionable Setups"),
            45,
        )

    def test_initial_scan_batch_size_expands_for_actionable_focus(self):
        self.assertEqual(
            _initial_scan_batch_size(
                10,
                scan_pool_n=35,
                custom_mode_active=False,
                scan_mode="Actionable Setups",
            ),
            20,
        )
        self.assertEqual(
            _initial_scan_batch_size(
                10,
                scan_pool_n=35,
                custom_mode_active=False,
                scan_mode="Broad Market",
            ),
            10,
        )

    def test_initial_scan_symbols_actionable_focus_mixes_ranked_and_exploratory_candidates(self):
        out = _initial_scan_symbols(
            candidate_pool=[f"COIN{i}/USDT" for i in range(1, 41)],
            requested_n=10,
            scan_pool_n=40,
            custom_mode_active=False,
            scan_mode="Actionable Setups",
            timeframe="5m",
        )
        self.assertEqual(len(out), 20)
        self.assertEqual(out[:10], [f"COIN{i}/USDT" for i in range(1, 11)])
        self.assertTrue(any(symbol not in [f"COIN{i}/USDT" for i in range(11, 21)] for symbol in out[10:]))

    def test_actionable_analysis_batch_size_caps_deep_scan(self):
        self.assertEqual(
            _actionable_analysis_batch_size(
                10,
                fetched_n=35,
                scan_mode="Actionable Setups",
            ),
            28,
        )
        self.assertEqual(
            _actionable_analysis_batch_size(
                10,
                fetched_n=35,
                scan_mode="Broad Market",
            ),
            35,
        )

    def test_next_refill_candidate_batch_uses_remaining_pool_after_attrition(self):
        out = _next_refill_candidate_batch(
            candidate_pool=["BTC/USDT", "ETH/USDT", "SOL/USDT", "XRP/USDT"],
            attempted_symbols={"BTC/USDT", "ETH/USDT"},
            requested_n=3,
            produced_n=1,
            custom_mode_active=False,
            used_major_fallback=False,
        )
        self.assertEqual(out, ["SOL/USDT", "XRP/USDT"])

    def test_candidate_scan_symbols_actionable_focus_prefers_active_liquid_candidates(self):
        out = _candidate_scan_symbols(
            usdt_symbols=["BTC/USDT", "DOGE/USDT", "XRP/USDT"],
            market_rows=[
                {"symbol": "btc", "market_cap": 2_000_000_000_000, "price_change_percentage_24h": 0.4},
                {"symbol": "doge", "market_cap": 25_000_000_000, "price_change_percentage_24h": 4.2},
                {"symbol": "xrp", "market_cap": 120_000_000_000, "price_change_percentage_24h": 1.5},
            ],
            exclude_stables=False,
            custom_bases_applied=[],
            timeframe="1h",
            direction_filter="Upside",
            scan_mode="Actionable Setups",
        )
        self.assertEqual(out[0], "DOGE/USDT")

    def test_candidate_scan_symbols_actionable_focus_biases_leading_sector(self):
        out = _candidate_scan_symbols(
            usdt_symbols=["FET/USDT", "UNI/USDT", "AAVE/USDT"],
            market_rows=[
                {"symbol": "fet", "market_cap": 4_000_000_000, "price_change_percentage_24h": 2.8},
                {"symbol": "uni", "market_cap": 5_000_000_000, "price_change_percentage_24h": 2.7},
                {"symbol": "aave", "market_cap": 4_800_000_000, "price_change_percentage_24h": 0.4},
            ],
            exclude_stables=False,
            custom_bases_applied=[],
            timeframe="1h",
            direction_filter="Upside",
            scan_mode="Actionable Setups",
            classify_symbol_sector=lambda symbol: {"FET": "AI", "UNI": "DeFi", "AAVE": "DeFi"}.get(str(symbol).upper(), "Other"),
        )
        self.assertEqual(out[0], "FET/USDT")

    def test_actionable_universe_movement_score_keeps_quiet_candidate_alive(self):
        self.assertGreater(
            _actionable_universe_movement_score(
                0.35,
                timeframe="1h",
                direction_filter="Upside",
            ),
            0.0,
        )

    def test_candidate_scan_symbols_actionable_setups_respects_downside_direction(self):
        out = _candidate_scan_symbols(
            usdt_symbols=["BTC/USDT", "DOGE/USDT", "XRP/USDT"],
            market_rows=[
                {"symbol": "btc", "market_cap": 2_000_000_000_000, "price_change_percentage_24h": 0.8},
                {"symbol": "doge", "market_cap": 25_000_000_000, "price_change_percentage_24h": -4.2},
                {"symbol": "xrp", "market_cap": 120_000_000_000, "price_change_percentage_24h": -1.5},
            ],
            exclude_stables=False,
            custom_bases_applied=[],
            timeframe="1h",
            direction_filter="Downside",
            scan_mode="Actionable Setups",
        )
        self.assertEqual(out[0], "DOGE/USDT")

    def test_next_scan_pool_target_grows_after_pool_is_exhausted(self):
        self.assertEqual(
            _next_scan_pool_target(
                75,
                requested_n=50,
                produced_n=20,
                custom_mode_active=False,
                used_major_fallback=False,
            ),
            112,
        )
        self.assertEqual(
            _next_scan_pool_target(
                250,
                requested_n=50,
                produced_n=20,
                custom_mode_active=False,
                used_major_fallback=False,
            ),
            250,
        )
        self.assertEqual(
            _next_scan_pool_target(
                75,
                requested_n=50,
                produced_n=50,
                custom_mode_active=False,
                used_major_fallback=False,
            ),
            75,
        )

    def test_fetch_ticker_delta_once_hits_selected_symbol_only_once(self):
        calls: list[str] = []

        def _fake_get_price_change(symbol: str):
            calls.append(symbol)
            return 1.25

        self.assertEqual(_fetch_ticker_delta_once(_fake_get_price_change, "XBT/USD"), 1.25)
        self.assertEqual(calls, ["XBT/USD"])

    def test_scan_universe_notice_suppresses_underfill_once_pool_is_filled(self):
        self.assertIsNone(
            _scan_universe_notice(
                candidate_count=3,
                requested_n=3,
                custom_mode_active=False,
                used_major_fallback=False,
                has_market_rows=True,
                source_pair_count=10,
                market_row_count=10,
                top_n=3,
            )
        )
        level, message = _scan_universe_notice(
            candidate_count=2,
            requested_n=3,
            custom_mode_active=False,
            used_major_fallback=False,
            has_market_rows=True,
            source_pair_count=10,
            market_row_count=10,
            top_n=3,
        )
        self.assertEqual(level, "info")
        self.assertIn("returned 2 eligible symbols", message)

    def test_scan_universe_notice_explains_pair_ranking_unresolved_without_major_fallback(self):
        level, message = _scan_universe_notice(
            candidate_count=0,
            requested_n=3,
            custom_mode_active=False,
            used_major_fallback=False,
            has_market_rows=True,
            source_pair_count=0,
            market_row_count=25,
            top_n=3,
        )
        self.assertEqual(level, "warning")
        self.assertIn("strict exchange pair ranking could not resolve", message)

    def test_sync_market_cap_cells_rewrites_rows_to_latest_enrichment_map(self):
        rows = [
            {"Coin": "BTC", "Market Cap ($)": "—", "__mcap_val": 0},
            {"Coin": "XBT", "Market Cap ($)": "old", "__mcap_val": 1},
            {"Coin": "DOGE", "Market Cap ($)": "old", "__mcap_val": 1},
        ]
        out = _sync_market_cap_cells(
            rows,
            {"BTC": 1000},
            lambda value: f"mcap:{value}",
        )
        self.assertEqual(out[0]["Market Cap ($)"], "mcap:1000")
        self.assertEqual(out[0]["__mcap_val"], 1000)
        self.assertEqual(out[1]["Market Cap ($)"], "mcap:1000")
        self.assertEqual(out[1]["__mcap_val"], 1000)
        self.assertEqual(out[2]["Market Cap ($)"], "—")
        self.assertEqual(out[2]["__mcap_val"], 0)

    def test_merge_market_cap_maps_keeps_highest_values(self):
        out = _merge_market_cap_maps({"BTC": 100, "ETH": 50}, {"BTC": 80, "ETH": 120, "DOGE": 10})
        self.assertEqual(out, {"BTC": 100, "ETH": 120, "DOGE": 10})

    def test_resolve_display_scan_state_prefers_contributing_batches_for_live_rows(self):
        state = None
        state = _remember_display_scan_state(
            state,
            batch_results=[{"Coin": "BTC"}],
            candidate_count=50,
            mcap_map={"BTC": 1000},
            has_market_rows=True,
            source_pair_count=50,
            market_row_count=50,
        )
        state = _remember_display_scan_state(
            state,
            batch_results=[],
            candidate_count=0,
            mcap_map={},
            has_market_rows=False,
            source_pair_count=0,
            market_row_count=0,
        )
        resolved = _resolve_display_scan_state(
            fresh_results=[{"Coin": "BTC"}],
            current_candidate_count=0,
            current_mcap_map={},
            current_has_market_rows=False,
            current_source_pair_count=0,
            current_market_row_count=0,
            display_state=state,
        )
        self.assertEqual(resolved["candidate_count"], 50)
        self.assertEqual(resolved["mcap_map"], {"BTC": 1000})
        self.assertTrue(bool(resolved["has_market_rows"]))
        self.assertEqual(resolved["source_pair_count"], 50)
        self.assertEqual(resolved["market_row_count"], 50)

    def test_resolve_notice_scan_state_keeps_widest_universe_seen_during_scan(self):
        notice = _resolve_notice_scan_state(
            current_candidate_count=75,
            current_has_market_rows=False,
            current_source_pair_count=0,
            current_market_row_count=0,
            display_state={
                "candidate_count": 50,
                "has_market_rows": True,
                "source_pair_count": 50,
                "market_row_count": 50,
            },
        )
        self.assertEqual(notice["candidate_count"], 75)
        self.assertTrue(bool(notice["has_market_rows"]))
        self.assertEqual(notice["source_pair_count"], 50)
        self.assertEqual(notice["market_row_count"], 50)

    def test_resolve_display_scan_state_falls_back_to_latest_state_without_live_rows(self):
        resolved = _resolve_display_scan_state(
            fresh_results=[],
            current_candidate_count=0,
            current_mcap_map={},
            current_has_market_rows=False,
            current_source_pair_count=12,
            current_market_row_count=0,
            display_state={"candidate_count": 50, "mcap_map": {"BTC": 1000}, "has_market_rows": True},
        )
        self.assertEqual(resolved["candidate_count"], 0)
        self.assertEqual(resolved["mcap_map"], {})
        self.assertFalse(bool(resolved["has_market_rows"]))
        self.assertEqual(resolved["source_pair_count"], 12)

    def test_underfilled_universe_message_prefers_major_fallback_wording(self):
        out = _underfilled_universe_message(
            custom_mode_active=False,
            used_major_fallback=True,
            has_market_rows=False,
            working_count=10,
            requested_n=50,
        )
        self.assertIn("Hardcoded major fallback universe", out)
        self.assertNotIn("exchange-ranked pairs", out)


if __name__ == "__main__":
    unittest.main()
