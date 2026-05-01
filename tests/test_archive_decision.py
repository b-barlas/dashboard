import unittest
from types import SimpleNamespace

import pandas as pd

from core.archive_decision import (
    apply_archive_confidence_guardrail,
    apply_archive_invalidation_guardrail,
    archive_decision_observability,
    archive_decision_score_adjustment,
    archive_decision_confidence_factor,
    archive_decision_feedback_for_signal,
    build_archive_decision_snapshot,
    build_archive_decision_feedback_map,
    build_archive_decision_feedback_model,
    build_archive_signal_decision_snapshot,
    calibrate_archive_decision_scores,
)


class ArchiveDecisionTests(unittest.TestCase):
    def test_decision_snapshot_uses_one_setup_scope_for_timing_and_path(self) -> None:
        events = []
        windows = []
        for idx in range(8):
            watch_key = f"watch-up-{idx}"
            enter_key = f"enter-down-{idx}"
            events.extend(
                [
                    {
                        "signal_key": watch_key,
                        "symbol": "TRX",
                        "timeframe": "1h",
                        "direction": "Upside",
                        "status": "RESOLVED",
                        "setup_confirm": "WATCH",
                        "directional_return_pct": 0.25,
                        "adverse_excursion_pct": 0.10,
                        "price": 0.32,
                        "event_time": f"2026-04-28T0{idx}:00:00Z",
                    },
                    {
                        "signal_key": enter_key,
                        "symbol": "TRX",
                        "timeframe": "1h",
                        "direction": "Downside",
                        "status": "RESOLVED",
                        "setup_confirm": "ENTER_TREND_AI",
                        "directional_return_pct": 1.10,
                        "adverse_excursion_pct": 0.20,
                        "price": 0.32,
                        "event_time": f"2026-04-28T0{idx}:30:00Z",
                    },
                ]
            )
            windows.append(
                {
                    "signal_key": enter_key,
                    "bars_ahead": 4,
                    "directional_return_pct": 1.05,
                    "adverse_excursion_pct": 0.15,
                    "favorable_excursion_pct": 1.20,
                }
            )

        for idx in range(2):
            events.append(
                {
                    "signal_key": f"enter-down-15m-{idx}",
                    "symbol": "TRX",
                    "timeframe": "15m",
                    "direction": "Downside",
                    "status": "RESOLVED",
                    "setup_confirm": "ENTER_TREND_AI",
                    "directional_return_pct": 0.8,
                    "adverse_excursion_pct": 0.15,
                    "price": 0.32,
                    "event_time": f"2026-04-28T1{idx}:30:00Z",
                }
            )

        snapshot = build_archive_decision_snapshot(
            df_events=pd.DataFrame(events),
            df_resolved_events=pd.DataFrame(events),
            df_forward_windows=pd.DataFrame(windows),
            symbol_filter="TRX",
            timeframe_filter="All",
            setup_filter_value="AUTO_BEST",
            min_completed=8,
        )

        self.assertTrue(snapshot.available)
        self.assertTrue(snapshot.setup.available)
        self.assertEqual(snapshot.setup.setup_class, "ENTER_TREND_AI")
        self.assertEqual(snapshot.setup.direction, "DOWNSIDE")
        self.assertEqual(snapshot.setup.timeframe, "1h")
        self.assertGreater(snapshot.confidence_factor, 0.0)
        self.assertIn(snapshot.confidence_tier, {"Thin", "Good", "Strong"})
        self.assertEqual(set(snapshot.metric_events["signal_key"]), {f"enter-down-{idx}" for idx in range(8)})
        self.assertIn("15m", set(snapshot.direction_events["timeframe"]))
        self.assertEqual(set(snapshot.hold_events["signal_key"]), {f"enter-down-{idx}" for idx in range(8)})
        self.assertEqual(int(snapshot.hold_window["best_bar"]), 4)
        self.assertTrue(snapshot.expected_path["available"])
        self.assertEqual(snapshot.expected_path["direction"], "DOWNSIDE")

    def test_signal_decision_snapshot_keeps_actionable_setup_strict(self) -> None:
        events = []
        windows = []
        for idx in range(3):
            key = f"watch-{idx}"
            events.append(
                {
                    "signal_key": key,
                    "symbol": "BTC",
                    "timeframe": "1h",
                    "direction": "Upside",
                    "status": "RESOLVED",
                    "setup_confirm": "WATCH",
                    "directional_return_pct": 0.2,
                    "adverse_excursion_pct": 0.1,
                }
            )
            windows.append(
                {
                    "signal_key": key,
                    "bars_ahead": 4,
                    "directional_return_pct": 0.2,
                    "adverse_excursion_pct": 0.1,
                }
            )
        for idx in range(10):
            events.append(
                {
                    "signal_key": f"enter-{idx}",
                    "symbol": "BTC",
                    "timeframe": "1h",
                    "direction": "Upside",
                    "status": "RESOLVED",
                    "setup_confirm": "ENTER_TREND_AI",
                    "directional_return_pct": 1.5,
                    "adverse_excursion_pct": 0.2,
                }
            )

        snapshot = build_archive_signal_decision_snapshot(
            df_events=pd.DataFrame(events),
            df_forward_windows=pd.DataFrame(windows),
            symbol="BTC/USDT",
            timeframe="1h",
            direction="LONG",
            setup_confirm="WATCH",
        )

        self.assertEqual(snapshot.scope_label, "BTC 1H WATCH ↑")
        self.assertEqual(set(snapshot.hold_events["signal_key"]), {"watch-0", "watch-1", "watch-2"})
        self.assertFalse(snapshot.setup.available)
        self.assertEqual(int(snapshot.hold_window["resolved_signals"]), 3)

    def test_signal_decision_snapshot_does_not_borrow_other_setup_when_missing(self) -> None:
        events = [
            {
                "signal_key": f"enter-{idx}",
                "symbol": "BTC",
                "timeframe": "1h",
                "direction": "Upside",
                "status": "RESOLVED",
                "setup_confirm": "ENTER_TREND_AI",
                "directional_return_pct": 1.5,
                "adverse_excursion_pct": 0.2,
            }
            for idx in range(10)
        ]

        snapshot = build_archive_signal_decision_snapshot(
            df_events=pd.DataFrame(events),
            df_forward_windows=pd.DataFrame(),
            symbol="BTC",
            timeframe="1h",
            direction="LONG",
            setup_confirm="WATCH",
        )

        self.assertFalse(snapshot.available)
        self.assertTrue(snapshot.hold_events.empty)
        self.assertEqual(snapshot.scope_label, "No BTC 1H WATCH ↑ archive pocket yet")

    def test_archive_decision_score_adjustment_rewards_clean_timing_path(self) -> None:
        clean = archive_decision_score_adjustment(
            SimpleNamespace(
                hold_window={
                    "available": True,
                    "sample": 32,
                    "follow_through_pct": 68.0,
                    "avg_dir_return_pct": 1.2,
                    "avg_adverse_excursion_pct": 0.2,
                },
                expected_path={
                    "available": True,
                    "sample": 32,
                    "follow_through_pct": 66.0,
                    "zone_hit_rate_pct": 78.0,
                    "clean_path_rate_pct": 70.0,
                    "caution_break_rate_pct": 8.0,
                    "path_conflict": False,
                },
            )
        )
        messy = archive_decision_score_adjustment(
            SimpleNamespace(
                hold_window={
                    "available": True,
                    "sample": 32,
                    "follow_through_pct": 42.0,
                    "avg_dir_return_pct": -0.4,
                    "avg_adverse_excursion_pct": 1.4,
                },
                expected_path={
                    "available": True,
                    "sample": 32,
                    "follow_through_pct": 44.0,
                    "zone_hit_rate_pct": 35.0,
                    "clean_path_rate_pct": 32.0,
                    "caution_break_rate_pct": 48.0,
                    "path_conflict": True,
                },
            )
        )
        thin = archive_decision_score_adjustment(
            SimpleNamespace(
                hold_window={
                    "available": True,
                    "sample": 4,
                    "follow_through_pct": 68.0,
                    "avg_dir_return_pct": 1.2,
                    "avg_adverse_excursion_pct": 0.2,
                },
                expected_path={},
            )
        )

        self.assertGreater(clean[0], 0.0)
        self.assertGreater(clean[1], 0.0)
        self.assertLess(messy[0], 0.0)
        self.assertLess(messy[1], 0.0)
        self.assertLess(thin[0], clean[0])

    def test_archive_invalidation_guardrail_dampens_risky_positive_boosts(self) -> None:
        risky = SimpleNamespace(
            expected_path={
                "available": True,
                "sample": 32,
                "archive_check_sample": 32,
                "path_conflict": True,
                "zone_hit_rate_pct": 32.0,
                "clean_path_rate_pct": 24.0,
                "caution_break_rate_pct": 58.0,
                "follow_through_pct": 46.0,
            }
        )
        clean = SimpleNamespace(
            expected_path={
                "available": True,
                "sample": 32,
                "archive_check_sample": 32,
                "path_conflict": False,
                "zone_hit_rate_pct": 82.0,
                "clean_path_rate_pct": 74.0,
                "caution_break_rate_pct": 4.0,
                "follow_through_pct": 68.0,
            }
        )

        risky_delta, risky_expectancy = apply_archive_invalidation_guardrail(8.0, 6.0, risky)
        clean_delta, clean_expectancy = apply_archive_invalidation_guardrail(8.0, 6.0, clean)
        negative_delta, negative_expectancy = apply_archive_invalidation_guardrail(-4.0, -2.0, risky)

        self.assertLess(risky_delta, 8.0)
        self.assertLess(risky_expectancy, 6.0)
        self.assertEqual((clean_delta, clean_expectancy), (8.0, 6.0))
        self.assertLess(negative_delta, -4.0)
        self.assertLess(negative_expectancy, -2.0)

    def test_archive_confidence_guardrail_scales_thin_history_more_than_strong_history(self) -> None:
        thin = SimpleNamespace(
            setup=SimpleNamespace(completed=4, coverage_factor=0.125),
            hold_window={"available": True, "sample": 4, "resolved_signals": 4},
            expected_path={"available": False},
            confidence_factor=0.0,
        )
        strong = SimpleNamespace(
            setup=SimpleNamespace(completed=40, coverage_factor=1.0),
            hold_window={"available": True, "sample": 40, "resolved_signals": 40},
            expected_path={
                "available": True,
                "sample": 40,
                "archive_check_sample": 40,
                "read_quality": "Strong",
                "path_conflict": False,
                "zone_hit_rate_pct": 82.0,
                "clean_path_rate_pct": 74.0,
                "caution_break_rate_pct": 4.0,
                "follow_through_pct": 68.0,
            },
            confidence_factor=1.0,
        )

        thin_auto = SimpleNamespace(**{**thin.__dict__, "confidence_factor": archive_decision_confidence_factor(thin)})
        strong_auto = SimpleNamespace(**{**strong.__dict__, "confidence_factor": archive_decision_confidence_factor(strong)})
        thin_delta, thin_expectancy = apply_archive_confidence_guardrail(8.0, 4.0, thin_auto)
        strong_delta, strong_expectancy = apply_archive_confidence_guardrail(8.0, 4.0, strong_auto)

        self.assertLess(thin_auto.confidence_factor, strong_auto.confidence_factor)
        self.assertLess(thin_delta, strong_delta)
        self.assertLess(thin_expectancy, strong_expectancy)
        self.assertEqual((strong_delta, strong_expectancy), (8.0, 4.0))

    def test_archive_decision_observability_exposes_hidden_audit_fields(self) -> None:
        snapshot = SimpleNamespace(
            confidence_factor=0.67,
            confidence_tier="Good",
            expected_path={
                "available": True,
                "sample": 32,
                "archive_check_sample": 32,
                "path_conflict": True,
                "zone_hit_rate_pct": 36.0,
                "clean_path_rate_pct": 30.0,
                "caution_break_rate_pct": 44.0,
                "follow_through_pct": 48.0,
            },
        )
        feedback = SimpleNamespace(active=True, multiplier=0.82)

        meta = archive_decision_observability(snapshot, feedback)

        self.assertAlmostEqual(float(meta["archive_confidence_factor"]), 0.67)
        self.assertEqual(meta["archive_confidence_tier"], "Good")
        self.assertGreater(float(meta["archive_invalidation_risk"]), 0.0)
        self.assertAlmostEqual(float(meta["archive_feedback_multiplier"]), 0.82)

    def test_archive_decision_feedback_model_scales_trusted_and_weak_nudges(self) -> None:
        trusted_rows = []
        weak_rows = []
        for idx in range(32):
            trusted_rows.append(
                {
                    "status": "RESOLVED",
                    "archive_decision_delta": 2.0 if idx < 24 else -2.0,
                    "directional_return_pct": 1.0 if idx < 24 else -0.7,
                }
            )
            weak_rows.append(
                {
                    "status": "RESOLVED",
                    "archive_decision_delta": 2.0 if idx < 24 else -2.0,
                    "directional_return_pct": -0.8 if idx < 24 else 0.6,
                }
            )

        trusted = build_archive_decision_feedback_model(pd.DataFrame(trusted_rows), min_samples=24)
        weak = build_archive_decision_feedback_model(pd.DataFrame(weak_rows), min_samples=24)
        thin = build_archive_decision_feedback_model(pd.DataFrame(trusted_rows[:8]), min_samples=24)

        self.assertTrue(trusted.active)
        self.assertGreater(trusted.multiplier, 1.0)
        self.assertTrue(weak.active)
        self.assertLess(weak.multiplier, 1.0)
        self.assertFalse(thin.active)
        self.assertEqual(thin.multiplier, 1.0)

        boosted = calibrate_archive_decision_scores(4.0, 2.0, trusted)
        damped = calibrate_archive_decision_scores(4.0, 2.0, weak)
        unchanged = calibrate_archive_decision_scores(4.0, 2.0, thin)
        self.assertGreater(boosted[0], 4.0)
        self.assertLess(damped[0], 4.0)
        self.assertEqual(unchanged, (4.0, 2.0))

    def test_archive_feedback_prefers_total_archive_delta_when_available(self) -> None:
        rows = [
            {
                "status": "RESOLVED",
                "archive_decision_delta": -2.0,
                "archive_total_delta": 3.0,
                "directional_return_pct": 0.8,
            }
            for _ in range(28)
        ]

        feedback = build_archive_decision_feedback_model(pd.DataFrame(rows), min_samples=24)

        self.assertTrue(feedback.active)
        self.assertGreater(feedback.multiplier, 1.0)

    def test_archive_feedback_weights_recent_outcomes_more_than_old_ones(self) -> None:
        rows = []
        for idx in range(30):
            rows.append(
                {
                    "status": "RESOLVED",
                    "archive_total_delta": 3.0,
                    "directional_return_pct": 0.8,
                    "event_time": f"2026-01-{(idx % 28) + 1:02d}T00:00:00Z",
                }
            )
        for idx in range(18):
            rows.append(
                {
                    "status": "RESOLVED",
                    "archive_total_delta": 3.0,
                    "directional_return_pct": -0.8,
                    "event_time": f"2026-04-{13 + idx:02d}T00:00:00Z",
                }
            )

        feedback = build_archive_decision_feedback_model(pd.DataFrame(rows), min_samples=24)

        self.assertTrue(feedback.active)
        self.assertLess(feedback.hit_rate_pct, 50.0)
        self.assertLess(feedback.multiplier, 1.0)

    def test_archive_feedback_map_prefers_exact_signal_scope_over_global(self) -> None:
        rows = []
        for idx in range(30):
            rows.append(
                {
                    "status": "RESOLVED",
                    "timeframe": "1h",
                    "setup_confirm": "WATCH",
                    "direction": "Upside",
                    "archive_total_delta": 3.0,
                    "directional_return_pct": 0.8,
                }
            )
            rows.append(
                {
                    "status": "RESOLVED",
                    "timeframe": "15m",
                    "setup_confirm": "ENTER_TREND_AI",
                    "direction": "Downside",
                    "archive_total_delta": 3.0,
                    "directional_return_pct": -0.8,
                }
            )

        feedback_map = build_archive_decision_feedback_map(pd.DataFrame(rows), min_samples=24)
        global_feedback = build_archive_decision_feedback_model(pd.DataFrame(rows), min_samples=24)
        exact = archive_decision_feedback_for_signal(
            feedback_map,
            global_feedback,
            timeframe="1h",
            setup_confirm="WATCH_UP",
            direction="LONG",
        )
        weak_exact = archive_decision_feedback_for_signal(
            feedback_map,
            global_feedback,
            timeframe="15m",
            setup_confirm="ENTER T+AI",
            direction="SHORT",
        )

        self.assertTrue(global_feedback.active)
        self.assertAlmostEqual(global_feedback.multiplier, 1.0, places=2)
        self.assertGreater(exact.multiplier, global_feedback.multiplier)
        self.assertLess(weak_exact.multiplier, global_feedback.multiplier)

    def test_archive_feedback_map_prefers_coin_scope_when_available(self) -> None:
        rows = []
        for idx in range(30):
            rows.append(
                {
                    "status": "RESOLVED",
                    "symbol": "RAVE",
                    "timeframe": "1h",
                    "setup_confirm": "WATCH",
                    "direction": "Upside",
                    "archive_total_delta": 3.0,
                    "directional_return_pct": 0.9,
                }
            )
            rows.append(
                {
                    "status": "RESOLVED",
                    "symbol": "TRX",
                    "timeframe": "1h",
                    "setup_confirm": "WATCH",
                    "direction": "Upside",
                    "archive_total_delta": 3.0,
                    "directional_return_pct": -0.7,
                }
            )

        feedback_map = build_archive_decision_feedback_map(pd.DataFrame(rows), min_samples=24)
        global_feedback = build_archive_decision_feedback_model(pd.DataFrame(rows), min_samples=24)
        rave_feedback = archive_decision_feedback_for_signal(
            feedback_map,
            global_feedback,
            symbol="RAVE/USDT",
            timeframe="1h",
            setup_confirm="WATCH_UP",
            direction="LONG",
        )
        trx_feedback = archive_decision_feedback_for_signal(
            feedback_map,
            global_feedback,
            symbol="TRX",
            timeframe="1h",
            setup_confirm="WATCH_UP",
            direction="LONG",
        )

        self.assertAlmostEqual(global_feedback.multiplier, 1.0, places=2)
        self.assertGreater(rave_feedback.multiplier, global_feedback.multiplier)
        self.assertLess(trx_feedback.multiplier, global_feedback.multiplier)

    def test_archive_feedback_map_prefers_matching_market_context_when_available(self) -> None:
        rows = []
        for idx in range(30):
            rows.append(
                {
                    "status": "RESOLVED",
                    "timeframe": "1h",
                    "setup_confirm": "WATCH",
                    "direction": "Upside",
                    "market_playbook_key": "TREND_CONTINUATION",
                    "market_trade_gate_key": "TRADEABLE",
                    "archive_total_delta": 3.0,
                    "directional_return_pct": 0.9,
                }
            )
            rows.append(
                {
                    "status": "RESOLVED",
                    "timeframe": "1h",
                    "setup_confirm": "WATCH",
                    "direction": "Upside",
                    "market_playbook_key": "STAND_ASIDE",
                    "market_trade_gate_key": "NO_TRADE",
                    "archive_total_delta": 3.0,
                    "directional_return_pct": -0.8,
                }
            )

        feedback_map = build_archive_decision_feedback_map(pd.DataFrame(rows), min_samples=24)
        global_feedback = build_archive_decision_feedback_model(pd.DataFrame(rows), min_samples=24)
        tradeable_feedback = archive_decision_feedback_for_signal(
            feedback_map,
            global_feedback,
            timeframe="1h",
            setup_confirm="WATCH_UP",
            direction="LONG",
            playbook_key="TREND_CONTINUATION",
            trade_gate_key="TRADEABLE",
        )
        no_trade_feedback = archive_decision_feedback_for_signal(
            feedback_map,
            global_feedback,
            timeframe="1h",
            setup_confirm="WATCH_UP",
            direction="LONG",
            playbook_key="STAND_ASIDE",
            trade_gate_key="NO_TRADE",
        )

        self.assertAlmostEqual(global_feedback.multiplier, 1.0, places=2)
        self.assertGreater(tradeable_feedback.multiplier, global_feedback.multiplier)
        self.assertLess(no_trade_feedback.multiplier, global_feedback.multiplier)


if __name__ == "__main__":
    unittest.main()
