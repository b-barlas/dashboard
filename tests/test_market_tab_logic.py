import unittest
from unittest.mock import patch

import pandas as pd

from tabs.market_tab import (
    _ai_fallback_note,
    _build_market_cap_map,
    _canonical_pair_base,
    _custom_watchlist_enrichment_coverage,
    _delta_fallback_symbol,
    _filter_scan_symbols,
    _healthy_empty_seen_for_sig,
    _healthy_empty_registry,
    _last_good_registry,
    _last_good_snapshot_for_sig,
    _normalize_custom_bases,
    _market_scan_signature,
    _market_data_mode,
    _next_refill_candidate_batch,
    _next_scan_pool_target,
    _next_universe_fetch_n,
    _pair_provenance_label,
    _remember_display_scan_state,
    _resolve_notice_scan_state,
    _resolve_display_scan_state,
    _scan_candidate_pool_size,
    _merge_market_cap_maps,
    _market_result_priority_key,
    _scan_universe_notice,
    _prepare_scan_market_enrichment,
    _prepare_closed_frame,
    _remember_last_good_snapshot,
    _remember_healthy_empty_sig,
    _scan_attempt_is_stale,
    _should_rescan_market,
    _should_use_major_fallback,
    _should_use_cached_scan,
    _fetch_ticker_delta_once,
    _setup_confirm_priority,
    _setup_status_summary,
    _sync_market_cap_cells,
    _underfilled_universe_message,
)


class MarketTabLogicTests(unittest.TestCase):
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
        self.assertGreater(_setup_confirm_priority("AI-led"), _setup_confirm_priority("WATCH"))
        self.assertGreater(_setup_confirm_priority("WATCH"), _setup_confirm_priority("SKIP"))

    def test_market_result_priority_key_prefers_trend_plus_ai_before_other_enter_classes(self):
        rows = [
            {
                "Coin": "SOL",
                "__action_raw": "🟡 ENTER (Trend-Led)",
                "__structure_state": "FULL",
                "__strength_val": 90.0,
                "__mcap_val": 500,
            },
            {
                "Coin": "BTC",
                "__action_raw": "✅ ENTER (Trend+AI)",
                "__structure_state": "FULL",
                "__strength_val": 70.0,
                "__mcap_val": 100,
            },
            {
                "Coin": "ETH",
                "__action_raw": "🟡 ENTER (AI-Led)",
                "__structure_state": "FULL",
                "__strength_val": 95.0,
                "__mcap_val": 1000,
            },
        ]
        ordered = sorted(rows, key=_market_result_priority_key)
        self.assertEqual([row["Coin"] for row in ordered], ["BTC", "SOL", "ETH"])

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
        self.assertIn("CACHED ENTER: 2", sub)

        _label, degraded_head, degraded_sub = _setup_status_summary(
            enter_count=1,
            watch_count=2,
            skip_count=3,
            source_label="LIVE (DEGRADED)",
        )
        self.assertEqual(degraded_head, "DEGRADED SETUPS")
        self.assertIn("DEGRADED ENTER: 1", degraded_sub)

    def test_market_scan_signature_ignores_top_n_in_custom_mode(self):
        first = _market_scan_signature(
            timeframe="1h",
            direction_filter="Both",
            top_n=50,
            exclude_stables=True,
            custom_bases_applied=["DOGE"],
        )
        second = _market_scan_signature(
            timeframe="1h",
            direction_filter="Both",
            top_n=10,
            exclude_stables=True,
            custom_bases_applied=["DOGE"],
        )
        self.assertEqual(first, second)
        self.assertEqual(first, ("1h", "Both", 0, True, ("DOGE",)))

    def test_market_scan_signature_is_order_and_alias_insensitive_in_custom_mode(self):
        first = _market_scan_signature(
            timeframe="1h",
            direction_filter="Both",
            top_n=50,
            exclude_stables=True,
            custom_bases_applied=["BTC", "ETH"],
        )
        second = _market_scan_signature(
            timeframe="1h",
            direction_filter="Both",
            top_n=50,
            exclude_stables=True,
            custom_bases_applied=["ETH", "XBT"],
        )
        self.assertEqual(first, second)
        self.assertEqual(first, ("1h", "Both", 0, True, ("BTC", "ETH")))

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
