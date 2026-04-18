from __future__ import annotations

import os
import sqlite3
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from core.signal_tracker import (
    annotate_alert_footprint,
    backfill_signal_forward_windows_for_frame,
    backfill_signal_forward_windows_via_fetch,
    build_alert_effectiveness_summary,
    build_actual_exit_quality_profile,
    build_actual_trade_hold_profile,
    build_hold_window_cohort_summary,
    build_hold_window_intelligence,
    build_recent_market_context_snapshot,
    build_recent_symbol_market_signal_snapshot,
    build_signal_cohort_summary,
    build_execution_overlay_snapshot,
    build_signal_review_snapshot,
    count_market_alerts,
    fetch_market_alerts_df,
    fetch_signal_forward_windows_df,
    fetch_signal_events_df,
    init_signal_tracker_db,
    log_market_alerts,
    log_signal_events,
    prefer_current_decision_version_slice,
    resolve_open_signal_events_via_fetch,
    resolve_open_signal_events_for_frame,
    save_signal_trade_journal,
    save_signal_trade_overlay,
)


class SignalTrackerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.db_path = str(Path(self.temp_dir.name) / "signals.sqlite3")
        init_signal_tracker_db(self.db_path)

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def test_log_signal_events_upserts_duplicate_candle(self) -> None:
        event = {
            "source": "Market",
            "symbol": "BTC",
            "timeframe": "1h",
            "event_time": "2026-04-04T12:00:00Z",
            "decision_version": "market-scanner-legacy-v0",
            "direction": "Upside",
            "setup_confirm": "WATCH",
            "lead_label": "Emerging Upside",
            "lead_direction": "Upside",
            "confidence": 71.0,
            "ai_ensemble": "Upside (2/3)",
            "ai_direction": "Upside",
            "ai_confidence": 64.0,
            "market_lead_label": "Upside",
            "market_lead_score": 63.0,
            "market_lead_upside": 4,
            "market_lead_downside": 1,
            "price": 100.0,
            "delta_pct": 0.5,
        }
        self.assertEqual(log_signal_events([event], self.db_path), 1)
        updated_event = dict(event)
        updated_event["decision_version"] = "market-scanner-2026-04-10-v1"
        self.assertEqual(log_signal_events([updated_event], self.db_path), 1)
        df = fetch_signal_events_df(limit=20, db_path=self.db_path)
        self.assertEqual(len(df), 1)
        self.assertEqual(str(df.iloc[0]["symbol"]), "BTC")
        self.assertEqual(str(df.iloc[0]["session_bucket"]), "European (08-16 UTC)")
        self.assertEqual(
            str(df.iloc[0]["created_decision_version"]),
            "market-scanner-legacy-v0",
        )
        self.assertEqual(
            str(df.iloc[0]["decision_version"]),
            "market-scanner-2026-04-10-v1",
        )

    def test_log_signal_events_persists_market_alert_footprint(self) -> None:
        event = {
            "source": "Market",
            "symbol": "BTC",
            "timeframe": "1h",
            "event_time": "2026-04-04T12:00:00Z",
            "scan_focus": "Actionable Setups",
            "direction": "Upside",
            "setup_confirm": "WATCH",
            "market_trade_gate": "Tradeable",
            "market_alert_keys": "MARKET_LEAD|EXECUTION_STANCE",
            "market_primary_alert": "MARKET_LEAD",
            "actionable_frame_score": 72.5,
            "actionable_setup_score": 81.0,
            "actionable_context_score": 67.5,
            "actionable_tactical_score": 74.0,
            "price": 100.0,
        }
        self.assertEqual(log_signal_events([event], self.db_path), 1)
        df = fetch_signal_events_df(limit=20, db_path=self.db_path)
        self.assertEqual(str(df.iloc[0]["market_alert_keys"]), "MARKET_LEAD|EXECUTION_STANCE")
        self.assertEqual(str(df.iloc[0]["market_primary_alert"]), "MARKET_LEAD")
        self.assertEqual(str(df.iloc[0]["scan_focus"]), "Actionable Setups")
        self.assertAlmostEqual(float(df.iloc[0]["actionable_frame_score"]), 72.5, places=4)
        self.assertAlmostEqual(float(df.iloc[0]["actionable_setup_score"]), 81.0, places=4)
        self.assertAlmostEqual(float(df.iloc[0]["actionable_context_score"]), 67.5, places=4)
        self.assertAlmostEqual(float(df.iloc[0]["actionable_tactical_score"]), 74.0, places=4)

    def test_fetch_signal_events_df_applies_timeframe_filter_before_limit(self) -> None:
        events = []
        for idx in range(4):
            events.append(
                {
                    "source": "Market",
                    "symbol": f"FAST{idx}",
                    "timeframe": "15m",
                    "event_time": f"2026-04-04T12:0{idx}:00Z",
                    "direction": "Upside",
                    "setup_confirm": "WATCH",
                    "price": 100.0 + idx,
                }
            )
        for idx in range(2):
            events.append(
                {
                    "source": "Market",
                    "symbol": f"SWING{idx}",
                    "timeframe": "1d",
                    "event_time": f"2026-04-03T12:0{idx}:00Z",
                    "direction": "Upside",
                    "setup_confirm": "WATCH",
                    "price": 200.0 + idx,
                }
            )
        self.assertEqual(log_signal_events(events, self.db_path), 6)
        df = fetch_signal_events_df(limit=2, source="Market", timeframe="1d", db_path=self.db_path)
        self.assertEqual(len(df), 2)
        self.assertEqual(set(df["timeframe"].astype(str)), {"1d"})

    def test_fetch_signal_events_df_applies_symbol_filter_before_limit(self) -> None:
        events = []
        for idx in range(4):
            events.append(
                {
                    "source": "Market",
                    "symbol": "BTC",
                    "timeframe": "1h",
                    "event_time": f"2026-04-04T12:0{idx}:00Z",
                    "direction": "Upside",
                    "setup_confirm": "WATCH",
                    "price": 100.0 + idx,
                }
            )
        for idx in range(3):
            events.append(
                {
                    "source": "Market",
                    "symbol": "ETH",
                    "timeframe": "1h",
                    "event_time": f"2026-04-03T12:0{idx}:00Z",
                    "direction": "Upside",
                    "setup_confirm": "WATCH",
                    "price": 200.0 + idx,
                }
            )
        self.assertEqual(log_signal_events(events, self.db_path), 7)
        df = fetch_signal_events_df(limit=2, source="Market", symbol="ETH", db_path=self.db_path)
        self.assertEqual(len(df), 2)
        self.assertEqual(set(df["symbol"].astype(str)), {"ETH"})

    def test_fetch_signal_events_df_applies_decision_version_filter_before_limit(self) -> None:
        events = []
        for idx in range(4):
            events.append(
                {
                    "source": "Market",
                    "symbol": f"CUR{idx}",
                    "timeframe": "1h",
                    "event_time": f"2026-04-04T13:0{idx}:00Z",
                    "decision_version": "market-scanner-2026-04-10-v1",
                    "direction": "Upside",
                    "setup_confirm": "WATCH",
                    "price": 100.0 + idx,
                }
            )
        for idx in range(3):
            events.append(
                {
                    "source": "Market",
                    "symbol": f"LEG{idx}",
                    "timeframe": "1h",
                    "event_time": f"2026-04-03T13:0{idx}:00Z",
                    "decision_version": "",
                    "direction": "Upside",
                    "setup_confirm": "WATCH",
                    "price": 200.0 + idx,
                }
            )
        self.assertEqual(log_signal_events(events, self.db_path), 7)
        current_df = fetch_signal_events_df(
            limit=2,
            source="Market",
            decision_version="market-scanner-2026-04-10-v1",
            db_path=self.db_path,
        )
        self.assertEqual(len(current_df), 2)
        self.assertTrue(current_df["decision_version"].fillna("").eq("market-scanner-2026-04-10-v1").all())
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "UPDATE signal_events SET decision_version = '' WHERE symbol LIKE 'LEG%'"
            )
            conn.commit()
        legacy_df = fetch_signal_events_df(
            limit=2,
            source="Market",
            decision_version="Legacy / Unversioned",
            db_path=self.db_path,
        )
        self.assertEqual(len(legacy_df), 2)
        self.assertTrue(legacy_df["decision_version"].fillna("").eq("").all())

    def test_log_signal_events_creates_mirror_snapshot_when_configured(self) -> None:
        mirror_dir = str(Path(self.temp_dir.name) / "mirror")
        event = {
            "source": "Market",
            "symbol": "BTC",
            "timeframe": "1h",
            "event_time": "2026-04-04T12:00:00Z",
            "direction": "Upside",
            "setup_confirm": "WATCH",
            "market_trade_gate": "Tradeable",
        }
        with patch.dict(
            os.environ,
            {
                "SIGNAL_TRACKER_MIRROR_DIR": mirror_dir,
                "SIGNAL_TRACKER_MIRROR_MINUTES": "1",
                "SIGNAL_TRACKER_MIRROR_KEEP": "4",
            },
            clear=False,
        ):
            self.assertEqual(log_signal_events([event], self.db_path), 1)
        mirror_files = list(Path(mirror_dir).glob("signals.mirror-*.sqlite3"))
        self.assertEqual(len(mirror_files), 1)

    def test_resolve_open_signal_events_for_frame_computes_outcome(self) -> None:
        event = {
            "source": "Market",
            "symbol": "ETH",
            "timeframe": "1h",
            "event_time": "2026-04-04T10:00:00Z",
            "direction": "Upside",
            "setup_confirm": "READY",
            "lead_label": "Emerging Upside",
            "lead_direction": "Upside",
            "confidence": 80.0,
            "ai_ensemble": "Upside (3/3)",
            "ai_direction": "Upside",
            "ai_confidence": 75.0,
            "market_lead_label": "Upside",
            "market_lead_score": 68.0,
            "market_lead_upside": 5,
            "market_lead_downside": 1,
            "market_regime": "Alt Rotation",
            "market_playbook": "Selective upside rotation",
            "market_no_trade": False,
            "market_trade_gate": "Selective Only",
            "market_no_trade_reason": "SELECTIVE_FILTER",
            "risk_tier": "Probe Only",
            "risk_unit_fraction": 0.25,
            "sector_tag": "DeFi",
            "market_sector_rotation": "DeFi Rotation",
            "market_catalyst_state": "Catalyst Caution",
            "market_catalyst_event": "US CPI",
            "market_catalyst_blocking": False,
            "market_catalyst_category": "macro",
            "market_catalyst_scope": "market",
            "market_catalyst_tag": "US",
            "market_catalyst_targeted": False,
            "market_catalyst_window": "High Impact (6-24h)",
            "market_flow_state": "Longs Crowded",
            "market_flow_bias": "LONG_CROWDING",
            "adaptive_edge_label": "Historically Favored",
            "adaptive_edge_score": 61.0,
            "archive_guardrail_label": "Archive Guardrail",
            "archive_guardrail_penalty": 6.2,
            "archive_guardrail_note": "Matched archive history is weak enough here to actively trim aggression.",
            "price": 100.0,
            "horizon_bars": 8,
            "entry_price": 100.0,
            "stop_loss": 95.0,
            "target_price": 110.0,
            "rr_ratio": 2.0,
        }
        log_signal_events([event], self.db_path)
        df = pd.DataFrame(
            {
                "timestamp": pd.to_datetime(
                    [
                        "2026-04-04T08:00:00Z",
                        "2026-04-04T09:00:00Z",
                        "2026-04-04T10:00:00Z",
                        "2026-04-04T11:00:00Z",
                        "2026-04-04T12:00:00Z",
                        "2026-04-04T13:00:00Z",
                        "2026-04-04T14:00:00Z",
                        "2026-04-04T15:00:00Z",
                        "2026-04-04T16:00:00Z",
                        "2026-04-04T17:00:00Z",
                        "2026-04-04T18:00:00Z",
                        "2026-04-04T19:00:00Z",
                        "2026-04-04T20:00:00Z",
                    ],
                    utc=True,
                ),
                "open": [99, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110],
                "high": [100, 101, 101, 103, 104, 106, 108, 109, 111, 112, 113, 114, 115],
                "low": [98, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109],
                "close": [99, 100, 100, 102, 103, 104, 106, 107, 108, 109, 110, 111, 112],
            }
        )
        resolved = resolve_open_signal_events_for_frame(
            symbol="ETH",
            timeframe="1h",
            df_ohlcv=df,
            source="Market",
            db_path=self.db_path,
        )
        self.assertEqual(resolved, 1)
        out = fetch_signal_events_df(limit=20, db_path=self.db_path)
        row = out.iloc[0]
        self.assertEqual(str(row["status"]), "RESOLVED")
        self.assertEqual(str(row["plan_outcome"]), "TP")
        self.assertEqual(str(row["market_regime"]), "Alt Rotation")
        self.assertEqual(str(row["market_trade_gate"]), "Selective Only")
        self.assertEqual(str(row["market_no_trade_reason"]), "SELECTIVE_FILTER")
        self.assertEqual(str(row["risk_tier"]), "Probe Only")
        self.assertAlmostEqual(float(row["risk_unit_fraction"]), 0.25, places=4)
        self.assertEqual(str(row["sector_tag"]), "DeFi")
        self.assertEqual(str(row["market_sector_rotation"]), "DeFi Rotation")
        self.assertEqual(str(row["market_catalyst_state"]), "Catalyst Caution")
        self.assertEqual(str(row["market_catalyst_event"]), "US CPI")
        self.assertEqual(int(row["market_catalyst_blocking"]), 0)
        self.assertEqual(str(row["market_catalyst_category"]), "macro")
        self.assertEqual(str(row["market_catalyst_scope"]), "market")
        self.assertEqual(str(row["market_catalyst_tag"]), "US")
        self.assertEqual(int(row["market_catalyst_targeted"]), 0)
        self.assertEqual(str(row["market_catalyst_window"]), "High Impact (6-24h)")
        self.assertEqual(str(row["market_flow_state"]), "Longs Crowded")
        self.assertEqual(str(row["market_flow_bias"]), "LONG_CROWDING")
        self.assertEqual(str(row["adaptive_edge_label"]), "Historically Favored")
        self.assertAlmostEqual(float(row["adaptive_edge_score"]), 61.0, places=4)
        self.assertEqual(str(row["archive_guardrail_label"]), "Archive Guardrail")
        self.assertAlmostEqual(float(row["archive_guardrail_penalty"]), 6.2, places=4)
        self.assertIn("trim aggression", str(row["archive_guardrail_note"]))
        self.assertGreater(float(row["directional_return_pct"]), 0.0)
        forward_windows = fetch_signal_forward_windows_df(
            signal_keys=[str(row["signal_key"])],
            db_path=self.db_path,
        )
        self.assertEqual(set(forward_windows["bars_ahead"].astype(int)), {1, 2, 4, 6, 8})
        checkpoint_4 = forward_windows[forward_windows["bars_ahead"].astype(int) == 4].iloc[0]
        self.assertGreater(float(checkpoint_4["directional_return_pct"]), 0.0)

    def test_build_hold_window_intelligence_prefers_best_pain_adjusted_bar(self) -> None:
        events = []
        base_time = pd.Timestamp("2026-04-04T10:00:00Z")
        for idx in range(8):
            events.append(
                {
                    "source": "Market",
                    "symbol": f"ETH{idx}",
                    "timeframe": "1h",
                    "event_time": (base_time + pd.Timedelta(hours=idx)).strftime("%Y-%m-%dT%H:%M:%SZ"),
                    "direction": "Upside",
                    "setup_confirm": "WATCH",
                    "price": 100.0,
                    "horizon_bars": 8,
                }
            )
        self.assertEqual(log_signal_events(events, self.db_path), 8)
        df = fetch_signal_events_df(limit=20, source="Market", db_path=self.db_path)
        with sqlite3.connect(self.db_path) as conn:
            for _, row in df.iterrows():
                conn.execute(
                    """
                    UPDATE signal_events
                    SET status = 'RESOLVED',
                        directional_return_pct = 2.0,
                        favorable_excursion_pct = 3.0,
                        adverse_excursion_pct = 1.0
                    WHERE signal_key = ?
                    """,
                    (str(row["signal_key"]),),
                )
                for bars, dir_ret, adv in [
                    (1, 0.2, 0.1),
                    (2, 0.5, 0.15),
                    (4, 1.4, 0.25),
                    (6, 1.1, 0.45),
                    (8, 0.7, 0.7),
                ]:
                    conn.execute(
                        """
                        INSERT INTO signal_forward_windows (
                            signal_key, bars_ahead, directional_return_pct,
                            favorable_excursion_pct, adverse_excursion_pct, window_end_time, created_at
                        ) VALUES (?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            str(row["signal_key"]),
                            bars,
                            dir_ret,
                            dir_ret + 0.2,
                            adv,
                            "2026-04-04T12:00:00Z",
                            "2026-04-04T12:00:00Z",
                        ),
                    )
            conn.commit()
        updated = fetch_signal_events_df(limit=20, source="Market", db_path=self.db_path)
        windows = fetch_signal_forward_windows_df(signal_keys=updated["signal_key"].astype(str).tolist(), db_path=self.db_path)
        snapshot = build_hold_window_intelligence(updated, windows)
        self.assertTrue(bool(snapshot["available"]))
        self.assertEqual(int(snapshot["best_bar"]), 4)
        self.assertEqual(str(snapshot["best_style"]), "Quick Follow-Through")
        self.assertEqual(int(snapshot["fade_after_bar"]), 6)

    def test_build_hold_window_intelligence_reports_building_when_resolved_rows_exist_without_windows(self) -> None:
        events = [
            {
                "source": "Market",
                "symbol": "ETH",
                "timeframe": "15m",
                "event_time": "2026-04-04T10:00:00Z",
                "direction": "Upside",
                "setup_confirm": "WATCH",
                "price": 100.0,
                "status": "RESOLVED",
                "signal_key": "eth-15m-1",
            }
        ]
        df_events = pd.DataFrame(events)
        snapshot = build_hold_window_intelligence(df_events, pd.DataFrame())
        self.assertFalse(bool(snapshot["available"]))
        self.assertEqual(int(snapshot["resolved_signals"]), 1)

    def test_build_hold_window_cohort_summary_suggests_hold_by_group(self) -> None:
        events = []
        base_time = pd.Timestamp("2026-04-04T10:00:00Z")
        setup_configs = [
            ("ENTER ↑ T+AI", [(1, 0.4, 0.10), (2, 0.9, 0.15), (4, 1.7, 0.20), (6, 1.0, 0.45)]),
            ("EARLY ↑ Trend", [(1, 0.2, 0.08), (2, 0.8, 0.12), (4, 0.6, 0.25), (6, 0.3, 0.40)]),
        ]
        for group_idx, (setup_confirm, checkpoints) in enumerate(setup_configs):
            for idx in range(4):
                events.append(
                    {
                        "source": "Market",
                        "symbol": f"G{group_idx}{idx}",
                        "timeframe": "1h",
                        "event_time": (base_time + pd.Timedelta(hours=(group_idx * 8) + idx)).strftime("%Y-%m-%dT%H:%M:%SZ"),
                        "direction": "Upside",
                        "setup_confirm": setup_confirm,
                        "price": 100.0,
                        "horizon_bars": 6,
                    }
                )
        self.assertEqual(log_signal_events(events, self.db_path), 8)
        df = fetch_signal_events_df(limit=20, source="Market", db_path=self.db_path)
        with sqlite3.connect(self.db_path) as conn:
            for _, row in df.iterrows():
                setup_confirm = str(row["setup_confirm"])
                conn.execute(
                    """
                    UPDATE signal_events
                    SET status = 'RESOLVED',
                        directional_return_pct = 1.5,
                        favorable_excursion_pct = 2.0,
                        adverse_excursion_pct = 0.5
                    WHERE signal_key = ?
                    """,
                    (str(row["signal_key"]),),
                )
                checkpoints = setup_configs[0][1] if setup_confirm == "ENTER ↑ T+AI" else setup_configs[1][1]
                for bars, dir_ret, adv in checkpoints:
                    conn.execute(
                        """
                        INSERT INTO signal_forward_windows (
                            signal_key, bars_ahead, directional_return_pct,
                            favorable_excursion_pct, adverse_excursion_pct, window_end_time, created_at
                        ) VALUES (?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            str(row["signal_key"]),
                            bars,
                            dir_ret,
                            dir_ret + 0.2,
                            adv,
                            "2026-04-04T12:00:00Z",
                            "2026-04-04T12:00:00Z",
                        ),
                    )
            conn.commit()
        updated = fetch_signal_events_df(limit=20, source="Market", db_path=self.db_path)
        windows = fetch_signal_forward_windows_df(signal_keys=updated["signal_key"].astype(str).tolist(), db_path=self.db_path)
        summary = build_hold_window_cohort_summary(updated, windows, "setup_confirm")
        self.assertEqual(set(summary["setup_confirm"].astype(str)), {"ENTER ↑ T+AI", "EARLY ↑ Trend"})
        enter_row = summary[summary["setup_confirm"].astype(str) == "ENTER ↑ T+AI"].iloc[0]
        early_row = summary[summary["setup_confirm"].astype(str) == "EARLY ↑ Trend"].iloc[0]
        self.assertEqual(str(enter_row["Suggested Hold"]), "around 4 bars")
        self.assertEqual(str(enter_row["Hold Style"]), "Quick Follow-Through")
        self.assertEqual(str(early_row["Suggested Hold"]), "around 2 bars")
        self.assertEqual(str(early_row["Hold Style"]), "Explosive")

    def test_backfill_signal_forward_windows_for_frame_populates_missing_resolved_rows(self) -> None:
        event = {
            "source": "Market",
            "symbol": "BTC",
            "timeframe": "1h",
            "event_time": "2026-04-04T10:00:00Z",
            "direction": "Upside",
            "setup_confirm": "WATCH",
            "price": 100.0,
            "horizon_bars": 6,
        }
        self.assertEqual(log_signal_events([event], self.db_path), 1)
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                UPDATE signal_events
                SET status = 'RESOLVED',
                    directional_return_pct = 1.4,
                    favorable_excursion_pct = 2.0,
                    adverse_excursion_pct = 0.5,
                    resolved_at = '2026-04-04T16:00:00Z'
                WHERE symbol = 'BTC'
                """
            )
            conn.commit()
        df = pd.DataFrame(
            {
                "timestamp": pd.date_range("2026-04-04 09:00:00", periods=8, freq="1h", tz="UTC"),
                "open": [99, 100, 100, 101, 102, 103, 104, 105],
                "high": [100, 101, 102, 103, 105, 106, 107, 108],
                "low": [98, 99, 99, 100, 101, 102, 103, 104],
                "close": [99, 100, 101, 102, 103, 104, 105, 106],
            }
        )
        backfilled = backfill_signal_forward_windows_for_frame(
            symbol="BTC",
            timeframe="1h",
            df_ohlcv=df,
            source="Market",
            db_path=self.db_path,
        )
        self.assertEqual(backfilled, 1)
        updated = fetch_signal_events_df(limit=5, source="Market", db_path=self.db_path)
        windows = fetch_signal_forward_windows_df(signal_keys=updated["signal_key"].astype(str).tolist(), db_path=self.db_path)
        self.assertEqual(set(windows["bars_ahead"].astype(int)), {1, 2, 4, 6})

    def test_backfill_signal_forward_windows_via_fetch_targets_missing_pairs(self) -> None:
        events = [
            {
                "source": "Market",
                "symbol": "BTC",
                "timeframe": "1h",
                "event_time": "2026-04-04T10:00:00Z",
                "direction": "Upside",
                "setup_confirm": "WATCH",
                "price": 100.0,
                "horizon_bars": 6,
            },
            {
                "source": "Market",
                "symbol": "ETH",
                "timeframe": "15m",
                "event_time": "2026-04-04T10:15:00Z",
                "direction": "Downside",
                "setup_confirm": "WATCH",
                "price": 50.0,
                "horizon_bars": 4,
            },
        ]
        self.assertEqual(log_signal_events(events, self.db_path), 2)
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("UPDATE signal_events SET status = 'RESOLVED', resolved_at = '2026-04-04T12:00:00Z'")
            conn.execute(
                """
                UPDATE signal_events
                SET decision_version = '', created_decision_version = ''
                WHERE event_time = '2026-04-03T10:00:00Z'
                """
            )
            conn.commit()

        def _fake_fetch(symbol: str, timeframe: str, limit: int = 260) -> pd.DataFrame:
            if symbol == "BTC":
                return pd.DataFrame(
                    {
                        "timestamp": pd.date_range("2026-04-04 09:00:00", periods=9, freq="1h", tz="UTC"),
                        "open": [99, 100, 100, 101, 102, 103, 104, 105, 106],
                        "high": [100, 101, 102, 103, 105, 106, 107, 108, 109],
                        "low": [98, 99, 99, 100, 101, 102, 103, 104, 105],
                        "close": [99, 100, 101, 102, 103, 104, 105, 106, 107],
                    }
                )
            return pd.DataFrame(
                {
                    "timestamp": pd.date_range("2026-04-04 10:00:00", periods=7, freq="15min", tz="UTC"),
                    "open": [50, 50, 49.8, 49.5, 49.2, 49.0, 48.8],
                    "high": [50.2, 50.1, 49.9, 49.7, 49.4, 49.1, 48.9],
                    "low": [49.9, 49.7, 49.4, 49.1, 48.8, 48.6, 48.4],
                    "close": [50.0, 49.8, 49.5, 49.2, 49.0, 48.8, 48.6],
                }
            )

        backfilled = backfill_signal_forward_windows_via_fetch(
            fetch_ohlcv=_fake_fetch,
            source="Market",
            db_path=self.db_path,
            limit_pairs=10,
            rows_per_pair=10,
            candle_limit=50,
        )
        self.assertEqual(backfilled, 2)
        df = fetch_signal_events_df(limit=10, source="Market", db_path=self.db_path)
        windows = fetch_signal_forward_windows_df(signal_keys=df["signal_key"].astype(str).tolist(), db_path=self.db_path)
        self.assertEqual(set(df["symbol"].astype(str)), {"BTC", "ETH"})
        self.assertTrue((windows.groupby("signal_key")["bars_ahead"].nunique() >= 3).all())

    def test_backfill_signal_forward_windows_via_fetch_respects_decision_version_scope(self) -> None:
        events = [
            {
                "source": "Market",
                "symbol": "BTC",
                "timeframe": "1h",
                "event_time": "2026-04-04T10:00:00Z",
                "direction": "Upside",
                "setup_confirm": "WATCH",
                "price": 100.0,
                "horizon_bars": 6,
                "decision_version": "market-scanner-2026-04-10-v1",
            },
            {
                "source": "Market",
                "symbol": "BTC",
                "timeframe": "1h",
                "event_time": "2026-04-03T10:00:00Z",
                "direction": "Upside",
                "setup_confirm": "WATCH",
                "price": 100.0,
                "horizon_bars": 6,
                "decision_version": "",
            },
        ]
        self.assertEqual(log_signal_events(events, self.db_path), 2)
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("UPDATE signal_events SET status = 'RESOLVED', resolved_at = '2026-04-04T12:00:00Z'")
            conn.execute(
                """
                UPDATE signal_events
                SET decision_version = '', created_decision_version = ''
                WHERE event_time = '2026-04-03T10:00:00Z'
                """
            )
            conn.commit()

        def _fake_fetch(_symbol: str, _timeframe: str, limit: int = 260) -> pd.DataFrame:
            return pd.DataFrame(
                {
                    "timestamp": pd.date_range("2026-04-03 09:00:00", periods=34, freq="1h", tz="UTC"),
                    "open": list(range(99, 133)),
                    "high": list(range(100, 134)),
                    "low": list(range(98, 132)),
                    "close": list(range(99, 133)),
                }
            )

        backfilled = backfill_signal_forward_windows_via_fetch(
            fetch_ohlcv=_fake_fetch,
            source="Market",
            db_path=self.db_path,
            limit_pairs=10,
            rows_per_pair=10,
            candle_limit=50,
            symbol="BTC",
            timeframe="1h",
            decision_version="market-scanner-2026-04-10-v1",
        )
        self.assertEqual(backfilled, 1)
        df = fetch_signal_events_df(limit=10, source="Market", db_path=self.db_path)
        current_key = str(df[df["decision_version"].fillna("").eq("market-scanner-2026-04-10-v1")].iloc[0]["signal_key"])
        legacy_key = str(df[df["decision_version"].fillna("").eq("")].iloc[0]["signal_key"])
        windows = fetch_signal_forward_windows_df(signal_keys=[current_key, legacy_key], db_path=self.db_path)
        self.assertIn(current_key, set(windows["signal_key"].astype(str)))
        self.assertNotIn(legacy_key, set(windows["signal_key"].astype(str)))

    def test_backfill_signal_forward_windows_for_frame_prefers_recent_missing_rows(self) -> None:
        events = [
            {
                "source": "Market",
                "symbol": "BTC",
                "timeframe": "1h",
                "event_time": "2026-04-01T10:00:00Z",
                "direction": "Upside",
                "setup_confirm": "WATCH",
                "price": 100.0,
                "horizon_bars": 4,
            },
            {
                "source": "Market",
                "symbol": "BTC",
                "timeframe": "1h",
                "event_time": "2026-04-04T10:00:00Z",
                "direction": "Upside",
                "setup_confirm": "WATCH",
                "price": 100.0,
                "horizon_bars": 4,
            },
        ]
        self.assertEqual(log_signal_events(events, self.db_path), 2)
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("UPDATE signal_events SET status = 'RESOLVED', resolved_at = '2026-04-04T14:00:00Z'")
            conn.commit()
        df = pd.DataFrame(
            {
                "timestamp": pd.date_range("2026-04-04 09:00:00", periods=8, freq="1h", tz="UTC"),
                "open": [99, 100, 100, 101, 102, 103, 104, 105],
                "high": [100, 101, 102, 103, 104, 105, 106, 107],
                "low": [98, 99, 99, 100, 101, 102, 103, 104],
                "close": [99, 100, 101, 102, 103, 104, 105, 106],
            }
        )
        backfilled = backfill_signal_forward_windows_for_frame(
            symbol="BTC",
            timeframe="1h",
            df_ohlcv=df,
            source="Market",
            db_path=self.db_path,
            limit_rows=1,
        )
        self.assertEqual(backfilled, 1)
        all_events = fetch_signal_events_df(limit=10, source="Market", db_path=self.db_path)
        recent_key = str(all_events[all_events["event_time"].astype(str).eq("2026-04-04T10:00:00Z")].iloc[0]["signal_key"])
        old_key = str(all_events[all_events["event_time"].astype(str).eq("2026-04-01T10:00:00Z")].iloc[0]["signal_key"])
        windows = fetch_signal_forward_windows_df(signal_keys=[recent_key, old_key], db_path=self.db_path)
        self.assertIn(recent_key, set(windows["signal_key"].astype(str)))
        self.assertNotIn(old_key, set(windows["signal_key"].astype(str)))

    def test_resolve_open_signal_events_via_fetch_respects_symbol_and_timeframe_scope(self) -> None:
        events = [
            {
                "source": "Market",
                "symbol": "BTC",
                "timeframe": "1h",
                "event_time": "2026-04-04T10:00:00Z",
                "direction": "Upside",
                "setup_confirm": "WATCH",
                "price": 100.0,
                "horizon_bars": 4,
            },
            {
                "source": "Market",
                "symbol": "ETH",
                "timeframe": "15m",
                "event_time": "2026-04-04T10:15:00Z",
                "direction": "Downside",
                "setup_confirm": "WATCH",
                "price": 50.0,
                "horizon_bars": 4,
            },
        ]
        self.assertEqual(log_signal_events(events, self.db_path), 2)
        fetch_calls: list[tuple[str, str]] = []

        def _fake_fetch(symbol: str, timeframe: str, limit: int = 260) -> pd.DataFrame:
            fetch_calls.append((symbol, timeframe))
            return pd.DataFrame(
                {
                    "timestamp": pd.date_range("2026-04-04 09:00:00", periods=8, freq="1h", tz="UTC")
                    if timeframe == "1h"
                    else pd.date_range("2026-04-04 10:00:00", periods=8, freq="15min", tz="UTC"),
                    "open": [99, 100, 100, 101, 102, 103, 104, 105],
                    "high": [100, 101, 102, 103, 104, 105, 106, 107],
                    "low": [98, 99, 99, 100, 101, 102, 103, 104],
                    "close": [99, 100, 101, 102, 103, 104, 105, 106],
                }
            )

        resolved = resolve_open_signal_events_via_fetch(
            fetch_ohlcv=_fake_fetch,
            source="Market",
            db_path=self.db_path,
            limit_pairs=None,
            symbol="BTC",
            timeframe="1h",
        )
        self.assertEqual(resolved, 1)
        self.assertEqual(fetch_calls, [("BTC", "1h")])
        df = fetch_signal_events_df(limit=10, source="Market", db_path=self.db_path)
        btc_status = str(df[df["symbol"].astype(str).eq("BTC")].iloc[0]["status"])
        eth_status = str(df[df["symbol"].astype(str).eq("ETH")].iloc[0]["status"])
        self.assertEqual(btc_status, "RESOLVED")
        self.assertEqual(eth_status, "OPEN")

    def test_review_snapshot_and_cohort_summary(self) -> None:
        events = [
            {
                "source": "Market",
                "symbol": "SOL",
                "timeframe": "1h",
                "event_time": "2026-04-04T10:00:00Z",
                "direction": "Upside",
                "setup_confirm": "WATCH",
                "lead_label": "Emerging Upside",
                "lead_direction": "Upside",
                "market_lead_label": "Upside",
                "price": 50.0,
            },
            {
                "source": "Market",
                "symbol": "ADA",
                "timeframe": "1h",
                "event_time": "2026-04-04T11:00:00Z",
                "direction": "Downside",
                "setup_confirm": "SKIP",
                "lead_label": "",
                "lead_direction": "",
                "market_lead_label": "Balanced",
                "price": 40.0,
            },
        ]
        log_signal_events(events, self.db_path)
        df = fetch_signal_events_df(limit=20, db_path=self.db_path)
        snapshot = build_signal_review_snapshot(df)
        self.assertEqual(snapshot["total"], 2.0)
        cohort = build_signal_cohort_summary(
            df.assign(
                Lead=["LEAD", "No LEAD"],
                Session=["European (08-16 UTC)", "European (08-16 UTC)"],
                **{"Scan Focus": ["Broad Market", "Actionable Setups"]},
            ),
            "Lead",
        )
        self.assertFalse(cohort.empty)
        self.assertIn("Signals", cohort.columns)
        self.assertIn("ActualWinRatePct", cohort.columns)
        scan_focus_cohort = build_signal_cohort_summary(
            df.assign(**{"Scan Focus": ["Broad Market", "Actionable Setups"]}),
            "Scan Focus",
        )
        self.assertFalse(scan_focus_cohort.empty)
        self.assertIn("Scan Focus", scan_focus_cohort.columns)

    def test_alert_footprint_annotation_and_summary(self) -> None:
        df = pd.DataFrame(
            [
                {
                    "symbol": "FET",
                    "status": "RESOLVED",
                    "directional_return_pct": 4.0,
                    "market_alert_keys": "MARKET_LEAD|EXECUTION_STANCE",
                    "market_primary_alert": "MARKET_LEAD",
                    "trade_decision": "Taken",
                    "actual_trade_status": "CLOSED",
                    "actual_pnl_pct": 3.0,
                },
                {
                    "symbol": "ARB",
                    "status": "RESOLVED",
                    "directional_return_pct": -2.0,
                    "market_trade_gate": "Defensive Only",
                    "adaptive_edge_label": "Historically Weak",
                    "archive_guardrail_label": "Archive Guardrail",
                    "trade_decision": "Skipped",
                    "actual_trade_status": "",
                    "actual_pnl_pct": None,
                },
            ]
        )
        annotated = annotate_alert_footprint(df)
        self.assertIn("Primary Alert", annotated.columns)
        self.assertIn("Alert Footprint", annotated.columns)
        self.assertEqual(str(annotated.iloc[0]["Primary Alert"]), "Market Lead")

        exploded = build_alert_effectiveness_summary(annotated, primary_only=False)
        self.assertFalse(exploded.empty)
        self.assertIn("Alert Key", exploded.columns)
        self.assertIn("Execution Stance", set(exploded["Alert Key"]))

        primary = build_alert_effectiveness_summary(annotated, primary_only=True)
        self.assertFalse(primary.empty)
        self.assertIn("Primary Alert", primary.columns)

    def test_prefer_current_decision_version_slice_uses_current_when_sample_is_healthy(self) -> None:
        df = pd.DataFrame(
            {
                "symbol": [f"C{i}" for i in range(6)],
                "decision_version": [
                    "market-scanner-2026-04-10-v1",
                    "market-scanner-2026-04-10-v1",
                    "market-scanner-2026-04-10-v1",
                    "market-scanner-2026-04-10-v1",
                    "legacy-v0",
                    "legacy-v0",
                ],
            }
        )
        filtered = prefer_current_decision_version_slice(df, source="Market", min_rows=4)
        self.assertEqual(filtered.attrs.get("decision_version_mode"), "current_only")
        self.assertEqual(len(filtered), 4)
        self.assertTrue((filtered["decision_version"] == "market-scanner-2026-04-10-v1").all())

    def test_prefer_current_decision_version_slice_falls_back_when_current_sample_is_thin(self) -> None:
        df = pd.DataFrame(
            {
                "symbol": [f"C{i}" for i in range(5)],
                "decision_version": [
                    "market-scanner-2026-04-10-v1",
                    "market-scanner-2026-04-10-v1",
                    "legacy-v0",
                    "legacy-v0",
                    "legacy-v0",
                ],
            }
        )
        filtered = prefer_current_decision_version_slice(df, source="Market", min_rows=4)
        self.assertEqual(filtered.attrs.get("decision_version_mode"), "mixed_fallback")
        self.assertEqual(len(filtered), 5)

    def test_alert_footprint_annotation_handles_nan_fields_safely(self) -> None:
        df = pd.DataFrame(
            [
                {
                    "symbol": "ETH",
                    "status": "RESOLVED",
                    "directional_return_pct": 1.5,
                    "market_catalyst_blocking": float("nan"),
                    "market_catalyst_state": float("nan"),
                    "market_trade_gate": float("nan"),
                    "market_lead_label": float("nan"),
                    "market_flow_bias": float("nan"),
                    "market_sector_rotation": float("nan"),
                    "adaptive_edge_label": float("nan"),
                    "archive_guardrail_label": float("nan"),
                }
            ]
        )
        annotated = annotate_alert_footprint(df)
        self.assertEqual(str(annotated.iloc[0]["Primary Alert"]), "No Alert Footprint")
        self.assertEqual(str(annotated.iloc[0]["Alert Footprint"]), "No Alert Footprint")

    def test_actual_trade_hold_profile_identifies_quick_follow_through_cluster(self) -> None:
        df = pd.DataFrame(
            [
                {
                    "market_playbook": "Trend continuation",
                    "session_bucket": "European (08-16 UTC)",
                    "market_trade_gate": "Tradeable",
                    "market_catalyst_window": "Far / Clear",
                    "actual_trade_status": "CLOSED",
                    "actual_entry_at": "2026-04-04T08:00:00Z",
                    "actual_exit_at": "2026-04-04T11:00:00Z",
                    "actual_pnl_pct": 2.4,
                },
                {
                    "market_playbook": "Trend continuation",
                    "session_bucket": "European (08-16 UTC)",
                    "market_trade_gate": "Tradeable",
                    "market_catalyst_window": "Far / Clear",
                    "actual_trade_status": "CLOSED",
                    "actual_entry_at": "2026-04-05T08:00:00Z",
                    "actual_exit_at": "2026-04-05T12:00:00Z",
                    "actual_pnl_pct": 1.8,
                },
                {
                    "market_playbook": "Trend continuation",
                    "session_bucket": "European (08-16 UTC)",
                    "market_trade_gate": "Tradeable",
                    "market_catalyst_window": "Far / Clear",
                    "actual_trade_status": "CLOSED",
                    "actual_entry_at": "2026-04-06T08:00:00Z",
                    "actual_exit_at": "2026-04-06T13:00:00Z",
                    "actual_pnl_pct": -0.6,
                },
            ]
        )
        profile = build_actual_trade_hold_profile(
            df,
            playbook="Trend continuation",
            session_bucket="European (08-16 UTC)",
            trade_gate="Tradeable",
            catalyst_window="Far / Clear",
        )
        self.assertEqual(profile["label"], "Quick Follow-Through")
        self.assertGreater(float(profile["median_hold_hours"]), 0.0)

    def test_actual_trade_hold_profile_prefers_timeframe_direction_and_sector_slice(self) -> None:
        df = pd.DataFrame(
            [
                {
                    "symbol": "SOL",
                    "timeframe": "4h",
                    "direction": "Upside",
                    "sector_tag": "DeFi",
                    "market_playbook": "Trend continuation",
                    "session_bucket": "European (08-16 UTC)",
                    "market_trade_gate": "Tradeable",
                    "market_catalyst_window": "Far / Clear",
                    "actual_trade_status": "CLOSED",
                    "actual_entry_at": "2026-04-01T08:00:00Z",
                    "actual_exit_at": "2026-04-02T06:00:00Z",
                    "actual_pnl_pct": 4.2,
                },
                {
                    "symbol": "SOL",
                    "timeframe": "4h",
                    "direction": "Upside",
                    "sector_tag": "DeFi",
                    "market_playbook": "Trend continuation",
                    "session_bucket": "European (08-16 UTC)",
                    "market_trade_gate": "Tradeable",
                    "market_catalyst_window": "Far / Clear",
                    "actual_trade_status": "CLOSED",
                    "actual_entry_at": "2026-04-03T08:00:00Z",
                    "actual_exit_at": "2026-04-04T04:00:00Z",
                    "actual_pnl_pct": 3.1,
                },
                {
                    "symbol": "SOL",
                    "timeframe": "4h",
                    "direction": "Upside",
                    "sector_tag": "DeFi",
                    "market_playbook": "Trend continuation",
                    "session_bucket": "European (08-16 UTC)",
                    "market_trade_gate": "Tradeable",
                    "market_catalyst_window": "Far / Clear",
                    "actual_trade_status": "CLOSED",
                    "actual_entry_at": "2026-04-05T08:00:00Z",
                    "actual_exit_at": "2026-04-06T08:00:00Z",
                    "actual_pnl_pct": -0.8,
                },
                {
                    "symbol": "SOL",
                    "timeframe": "4h",
                    "direction": "Upside",
                    "sector_tag": "DeFi",
                    "market_playbook": "Trend continuation",
                    "session_bucket": "European (08-16 UTC)",
                    "market_trade_gate": "Tradeable",
                    "market_catalyst_window": "Far / Clear",
                    "actual_trade_status": "CLOSED",
                    "actual_entry_at": "2026-04-07T08:00:00Z",
                    "actual_exit_at": "2026-04-08T06:00:00Z",
                    "actual_pnl_pct": 2.5,
                },
                {
                    "symbol": "AAVE",
                    "timeframe": "1h",
                    "direction": "Upside",
                    "sector_tag": "DeFi",
                    "market_playbook": "Trend continuation",
                    "session_bucket": "European (08-16 UTC)",
                    "market_trade_gate": "Tradeable",
                    "market_catalyst_window": "Far / Clear",
                    "actual_trade_status": "CLOSED",
                    "actual_entry_at": "2026-04-01T08:00:00Z",
                    "actual_exit_at": "2026-04-01T11:00:00Z",
                    "actual_pnl_pct": 2.0,
                },
                {
                    "symbol": "AAVE",
                    "timeframe": "1h",
                    "direction": "Upside",
                    "sector_tag": "DeFi",
                    "market_playbook": "Trend continuation",
                    "session_bucket": "European (08-16 UTC)",
                    "market_trade_gate": "Tradeable",
                    "market_catalyst_window": "Far / Clear",
                    "actual_trade_status": "CLOSED",
                    "actual_entry_at": "2026-04-02T08:00:00Z",
                    "actual_exit_at": "2026-04-02T12:00:00Z",
                    "actual_pnl_pct": 1.9,
                },
                {
                    "symbol": "AAVE",
                    "timeframe": "1h",
                    "direction": "Upside",
                    "sector_tag": "DeFi",
                    "market_playbook": "Trend continuation",
                    "session_bucket": "European (08-16 UTC)",
                    "market_trade_gate": "Tradeable",
                    "market_catalyst_window": "Far / Clear",
                    "actual_trade_status": "CLOSED",
                    "actual_entry_at": "2026-04-03T08:00:00Z",
                    "actual_exit_at": "2026-04-03T13:00:00Z",
                    "actual_pnl_pct": 2.2,
                },
                {
                    "symbol": "AAVE",
                    "timeframe": "1h",
                    "direction": "Upside",
                    "sector_tag": "DeFi",
                    "market_playbook": "Trend continuation",
                    "session_bucket": "European (08-16 UTC)",
                    "market_trade_gate": "Tradeable",
                    "market_catalyst_window": "Far / Clear",
                    "actual_trade_status": "CLOSED",
                    "actual_entry_at": "2026-04-04T08:00:00Z",
                    "actual_exit_at": "2026-04-04T12:00:00Z",
                    "actual_pnl_pct": 1.7,
                },
            ]
        )
        profile = build_actual_trade_hold_profile(
            df,
            symbol="SOL",
            timeframe="4h",
            direction="LONG",
            sector_tag="DeFi",
            playbook="Trend continuation",
            session_bucket="European (08-16 UTC)",
            trade_gate="Tradeable",
            catalyst_window="Far / Clear",
        )
        self.assertEqual(profile["label"], "Needs Room")
        self.assertIn("SOL symbol", str(profile["note"]))
        self.assertIn("4H timeframe", str(profile["note"]))
        self.assertIn("Upside direction", str(profile["note"]))
        self.assertIn("DeFi sector", str(profile["note"]))

    def test_actual_trade_hold_profile_normalizes_pair_symbol_to_archive_base(self) -> None:
        df = pd.DataFrame(
            [
                {
                    "symbol": "SOL",
                    "timeframe": "4h",
                    "direction": "Upside",
                    "actual_trade_status": "CLOSED",
                    "actual_entry_at": "2026-04-01T08:00:00Z",
                    "actual_exit_at": "2026-04-02T06:00:00Z",
                    "actual_pnl_pct": 4.2,
                },
                {
                    "symbol": "SOL",
                    "timeframe": "4h",
                    "direction": "Upside",
                    "actual_trade_status": "CLOSED",
                    "actual_entry_at": "2026-04-03T08:00:00Z",
                    "actual_exit_at": "2026-04-04T04:00:00Z",
                    "actual_pnl_pct": 3.1,
                },
                {
                    "symbol": "SOL",
                    "timeframe": "4h",
                    "direction": "Upside",
                    "actual_trade_status": "CLOSED",
                    "actual_entry_at": "2026-04-05T08:00:00Z",
                    "actual_exit_at": "2026-04-06T02:00:00Z",
                    "actual_pnl_pct": 2.3,
                },
            ]
        )
        profile = build_actual_trade_hold_profile(
            df,
            symbol="SOL/USDT",
            timeframe="4h",
            direction="LONG",
        )
        self.assertEqual(profile["label"], "Needs Room")
        self.assertIn("SOL/USDT symbol", str(profile["note"]))

    def test_actual_trade_hold_profile_falls_back_when_specific_slice_is_too_thin(self) -> None:
        df = pd.DataFrame(
            [
                {
                    "symbol": "SOL",
                    "timeframe": "4h",
                    "direction": "Upside",
                    "sector_tag": "DeFi",
                    "market_playbook": "Trend continuation",
                    "session_bucket": "European (08-16 UTC)",
                    "market_trade_gate": "Tradeable",
                    "market_catalyst_window": "Far / Clear",
                    "actual_trade_status": "CLOSED",
                    "actual_entry_at": "2026-04-01T08:00:00Z",
                    "actual_exit_at": "2026-04-02T00:00:00Z",
                    "actual_pnl_pct": -0.4,
                },
                {
                    "symbol": "SOL",
                    "timeframe": "4h",
                    "direction": "Upside",
                    "sector_tag": "DeFi",
                    "market_playbook": "Trend continuation",
                    "session_bucket": "European (08-16 UTC)",
                    "market_trade_gate": "Tradeable",
                    "market_catalyst_window": "Far / Clear",
                    "actual_trade_status": "CLOSED",
                    "actual_entry_at": "2026-04-03T08:00:00Z",
                    "actual_exit_at": "2026-04-03T23:00:00Z",
                    "actual_pnl_pct": -0.2,
                },
                {
                    "symbol": "AAVE",
                    "timeframe": "1h",
                    "direction": "Upside",
                    "sector_tag": "DeFi",
                    "market_playbook": "Trend continuation",
                    "session_bucket": "European (08-16 UTC)",
                    "market_trade_gate": "Tradeable",
                    "market_catalyst_window": "Far / Clear",
                    "actual_trade_status": "CLOSED",
                    "actual_entry_at": "2026-04-04T08:00:00Z",
                    "actual_exit_at": "2026-04-04T11:00:00Z",
                    "actual_pnl_pct": 1.6,
                },
                {
                    "symbol": "AAVE",
                    "timeframe": "1h",
                    "direction": "Upside",
                    "sector_tag": "DeFi",
                    "market_playbook": "Trend continuation",
                    "session_bucket": "European (08-16 UTC)",
                    "market_trade_gate": "Tradeable",
                    "market_catalyst_window": "Far / Clear",
                    "actual_trade_status": "CLOSED",
                    "actual_entry_at": "2026-04-05T08:00:00Z",
                    "actual_exit_at": "2026-04-05T12:00:00Z",
                    "actual_pnl_pct": 1.8,
                },
                {
                    "symbol": "AAVE",
                    "timeframe": "1h",
                    "direction": "Upside",
                    "sector_tag": "DeFi",
                    "market_playbook": "Trend continuation",
                    "session_bucket": "European (08-16 UTC)",
                    "market_trade_gate": "Tradeable",
                    "market_catalyst_window": "Far / Clear",
                    "actual_trade_status": "CLOSED",
                    "actual_entry_at": "2026-04-06T08:00:00Z",
                    "actual_exit_at": "2026-04-06T13:00:00Z",
                    "actual_pnl_pct": -0.5,
                },
            ]
        )
        profile = build_actual_trade_hold_profile(
            df,
            symbol="SOL",
            timeframe="4h",
            direction="LONG",
            sector_tag="DeFi",
            playbook="Trend continuation",
            session_bucket="European (08-16 UTC)",
            trade_gate="Tradeable",
            catalyst_window="Far / Clear",
        )
        self.assertEqual(profile["label"], "Quick Follow-Through")
        self.assertIn("broader cluster", str(profile["note"]))
        self.assertIn("4H timeframe", str(profile["note"]))
        self.assertIn("SOL symbol", str(profile["note"]))

    def test_actual_exit_quality_profile_flags_winner_cut_risk(self) -> None:
        df = pd.DataFrame(
            [
                {
                    "symbol": "SOL",
                    "timeframe": "4h",
                    "direction": "Upside",
                    "sector_tag": "DeFi",
                    "market_playbook": "Trend continuation",
                    "session_bucket": "European (08-16 UTC)",
                    "market_trade_gate": "Tradeable",
                    "market_catalyst_window": "Far / Clear",
                    "actual_trade_status": "CLOSED",
                    "actual_exit_reason": "Manual Exit",
                    "actual_pnl_pct": 2.8,
                },
                {
                    "symbol": "SOL",
                    "timeframe": "4h",
                    "direction": "Upside",
                    "sector_tag": "DeFi",
                    "market_playbook": "Trend continuation",
                    "session_bucket": "European (08-16 UTC)",
                    "market_trade_gate": "Tradeable",
                    "market_catalyst_window": "Far / Clear",
                    "actual_trade_status": "CLOSED",
                    "actual_exit_reason": "Time Exit",
                    "actual_pnl_pct": 1.7,
                },
                {
                    "symbol": "SOL",
                    "timeframe": "4h",
                    "direction": "Upside",
                    "sector_tag": "DeFi",
                    "market_playbook": "Trend continuation",
                    "session_bucket": "European (08-16 UTC)",
                    "market_trade_gate": "Tradeable",
                    "market_catalyst_window": "Far / Clear",
                    "actual_trade_status": "CLOSED",
                    "actual_exit_reason": "Target",
                    "actual_pnl_pct": -0.4,
                },
                {
                    "symbol": "SOL",
                    "timeframe": "4h",
                    "direction": "Upside",
                    "sector_tag": "DeFi",
                    "market_playbook": "Trend continuation",
                    "session_bucket": "European (08-16 UTC)",
                    "market_trade_gate": "Tradeable",
                    "market_catalyst_window": "Far / Clear",
                    "actual_trade_status": "CLOSED",
                    "actual_exit_reason": "Stop",
                    "actual_pnl_pct": -0.8,
                },
            ]
        )
        profile = build_actual_exit_quality_profile(
            df,
            symbol="SOL",
            timeframe="4h",
            direction="LONG",
            sector_tag="DeFi",
            playbook="Trend continuation",
            session_bucket="European (08-16 UTC)",
            trade_gate="Tradeable",
            catalyst_window="Far / Clear",
        )
        self.assertEqual(profile["label"], "Winner Cut Risk")
        self.assertIn("SOL symbol", str(profile["note"]))
        self.assertIn("4H timeframe", str(profile["note"]))
        self.assertGreater(float(profile["winner_manual_rate"]), 60.0)

    def test_recent_market_context_snapshot_prefers_latest_market_archive(self) -> None:
        df = pd.DataFrame(
            [
                {
                    "source": "Market",
                    "event_time": "2026-04-04T04:00:00Z",
                    "market_lead_label": "Balanced",
                    "market_regime": "Range / Chop",
                    "market_playbook": "Stand aside / mean reversion only",
                    "market_trade_gate": "No-Trade",
                    "market_sector_rotation": "Mixed Sector Rotation",
                    "market_catalyst_state": "Catalyst Caution",
                    "market_catalyst_window": "Blocking (<6h)",
                    "market_catalyst_scope": "Market",
                    "market_catalyst_targeted": 0,
                    "market_flow_state": "Longs Crowded",
                },
                {
                    "source": "Market",
                    "event_time": "2026-04-04T14:00:00Z",
                    "market_lead_label": "Upside",
                    "market_regime": "Alt Rotation",
                    "market_playbook": "Selective upside rotation",
                    "market_trade_gate": "Selective Only",
                    "market_sector_rotation": "AI Rotation",
                    "market_catalyst_state": "Catalyst Clear",
                    "market_catalyst_window": "Far / Clear",
                    "market_catalyst_scope": "Market",
                    "market_catalyst_targeted": 0,
                    "market_flow_state": "Shorts Crowded",
                },
                {
                    "source": "Market",
                    "event_time": "2026-04-04T14:10:00Z",
                    "market_lead_label": "Upside",
                    "market_regime": "Alt Rotation",
                    "market_playbook": "Selective upside rotation",
                    "market_trade_gate": "Selective Only",
                    "market_sector_rotation": "AI Rotation",
                    "market_catalyst_state": "Catalyst Clear",
                    "market_catalyst_window": "Far / Clear",
                    "market_catalyst_scope": "Market",
                    "market_catalyst_targeted": 0,
                    "market_flow_state": "Shorts Crowded",
                },
                {
                    "source": "Spot",
                    "event_time": "2026-04-04T14:12:00Z",
                    "market_regime": "Risk-Off Trend",
                    "market_playbook": "Ignore Me",
                },
            ]
        )
        context = build_recent_market_context_snapshot(df, lookback_hours=6.0, max_rows=50)
        self.assertEqual(context["Market Lead"], "Upside")
        self.assertEqual(context["Market Regime"], "Alt Rotation")
        self.assertEqual(context["Playbook"], "Selective upside rotation")
        self.assertEqual(context["Trade Gate"], "Selective Only")
        self.assertEqual(context["Catalyst Window"], "Far / Clear")
        self.assertEqual(context["Catalyst Targeting"], "Market-Wide")
        self.assertIn("Selective upside rotation", context["Context Note"])

    def test_recent_symbol_market_signal_snapshot_reads_latest_matching_symbol_and_timeframe(self) -> None:
        df = pd.DataFrame(
            [
                {
                    "source": "Market",
                    "symbol": "SOL",
                    "timeframe": "15m",
                    "event_time": "2026-04-04T13:00:00Z",
                    "lead_label": "",
                    "lead_active": 0,
                    "setup_confirm": "WATCH",
                    "risk_tier": "Watch Only",
                    "direction": "Upside",
                    "adaptive_edge_label": "Historically Neutral",
                },
                {
                    "source": "Market",
                    "symbol": "SOL",
                    "timeframe": "1h",
                    "event_time": "2026-04-04T14:30:00Z",
                    "lead_label": "Emerging Upside",
                    "lead_active": 1,
                    "setup_confirm": "✅ ENTER (Trend+AI)",
                    "risk_tier": "Tier 1",
                    "direction": "Upside",
                    "adaptive_edge_label": "Historically Favored",
                },
                {
                    "source": "Market",
                    "symbol": "BTC",
                    "timeframe": "1h",
                    "event_time": "2026-04-04T14:35:00Z",
                    "lead_label": "Emerging Downside",
                    "lead_active": 1,
                },
            ]
        )
        signal = build_recent_symbol_market_signal_snapshot(df, symbol="SOL", timeframe="1h")
        self.assertEqual(signal["Lead"], "LEAD")
        self.assertEqual(signal["Lead Label"], "Emerging Upside")
        self.assertEqual(signal["Setup Confirm"], "✅ ENTER (Trend+AI)")
        self.assertEqual(signal["Risk Tier"], "Tier 1")
        self.assertIn("Emerging Upside", signal["Signal Note"])

    def test_recent_symbol_market_signal_snapshot_normalizes_pair_symbol_to_archive_base(self) -> None:
        df = pd.DataFrame(
            [
                {
                    "source": "Market",
                    "symbol": "SOL",
                    "timeframe": "1h",
                    "event_time": "2026-04-04T14:30:00Z",
                    "lead_label": "Emerging Upside",
                    "lead_active": 1,
                    "setup_confirm": "✅ ENTER (Trend+AI)",
                }
            ]
        )
        signal = build_recent_symbol_market_signal_snapshot(df, symbol="SOL/USDT", timeframe="1h")
        self.assertEqual(signal["Lead"], "LEAD")
        self.assertEqual(signal["Lead Label"], "Emerging Upside")

    def test_market_alerts_upsert_and_deactivate(self) -> None:
        first = [
            {
                "alert_key": "MARKET_LEAD",
                "state_signature": "UPSIDE|68",
                "severity": "MEDIUM",
                "title": "Upside pressure is building",
                "note": "Early internals are improving.",
            }
        ]
        second = [
            {
                "alert_key": "CATALYST_CAUTION",
                "state_signature": "CAUTION|US CPI",
                "severity": "MEDIUM",
                "title": "Catalyst window active: US CPI",
                "note": "Keep size smaller.",
            }
        ]
        self.assertEqual(log_market_alerts(first, db_path=self.db_path), 1)
        active = fetch_market_alerts_df(active_only=True, db_path=self.db_path)
        self.assertEqual(len(active), 1)
        self.assertEqual(str(active.iloc[0]["alert_key"]), "MARKET_LEAD")

        self.assertEqual(log_market_alerts(second, db_path=self.db_path), 1)
        active = fetch_market_alerts_df(active_only=True, db_path=self.db_path)
        self.assertEqual(len(active), 1)
        self.assertEqual(str(active.iloc[0]["alert_key"]), "CATALYST_CAUTION")

        all_alerts = fetch_market_alerts_df(active_only=False, db_path=self.db_path)
        self.assertEqual(len(all_alerts), 2)
        prior = all_alerts[all_alerts["alert_key"] == "MARKET_LEAD"].iloc[0]
        self.assertEqual(int(prior["active"]), 0)
        self.assertEqual(count_market_alerts(active_only=True, db_path=self.db_path), 1)
        self.assertEqual(count_market_alerts(active_only=False, db_path=self.db_path), 2)

    def test_save_signal_trade_overlay_updates_manual_execution_fields(self) -> None:
        event = {
            "source": "Market",
            "symbol": "BTC",
            "timeframe": "1h",
            "event_time": "2026-04-04T12:00:00Z",
            "direction": "Upside",
            "setup_confirm": "WATCH",
            "lead_label": "Emerging Upside",
            "lead_direction": "Upside",
            "price": 100.0,
        }
        log_signal_events([event], self.db_path)
        df = fetch_signal_events_df(limit=10, db_path=self.db_path)
        signal_key = str(df.iloc[0]["signal_key"])
        self.assertTrue(
            save_signal_trade_overlay(
                signal_key,
                trade_decision="Taken",
                trade_note="Followed the breakout.",
                db_path=self.db_path,
            )
        )
        updated = fetch_signal_events_df(limit=10, db_path=self.db_path)
        row = updated.iloc[0]
        self.assertEqual(str(row["trade_decision"]), "Taken")
        self.assertEqual(str(row["trade_note"]), "Followed the breakout.")
        self.assertTrue(str(row["trade_marked_at"]))

        self.assertTrue(save_signal_trade_overlay(signal_key, trade_decision="", trade_note="", db_path=self.db_path))
        cleared = fetch_signal_events_df(limit=10, db_path=self.db_path)
        row = cleared.iloc[0]
        self.assertTrue(pd.isna(row["trade_decision"]))

    def test_save_signal_trade_journal_computes_long_trade_pnl_and_marks_taken(self) -> None:
        event = {
            "source": "Market",
            "symbol": "ETH",
            "timeframe": "4h",
            "event_time": "2026-04-04T12:00:00Z",
            "direction": "Upside",
            "setup_confirm": "WATCH",
            "price": 100.0,
        }
        log_signal_events([event], self.db_path)
        df = fetch_signal_events_df(limit=10, db_path=self.db_path)
        signal_key = str(df.iloc[0]["signal_key"])

        self.assertTrue(
            save_signal_trade_journal(
                signal_key,
                actual_trade_side="Long",
                actual_entry_price=100,
                actual_entry_at="2026-04-04T12:05:00Z",
                actual_exit_price=110,
                actual_exit_at="2026-04-04T18:00:00Z",
                actual_exit_reason="Target",
                db_path=self.db_path,
            )
        )
        updated = fetch_signal_events_df(limit=10, db_path=self.db_path)
        row = updated.iloc[0]
        self.assertEqual(str(row["actual_trade_side"]), "LONG")
        self.assertEqual(str(row["actual_trade_status"]), "CLOSED")
        self.assertEqual(str(row["actual_exit_reason"]), "Target")
        self.assertAlmostEqual(float(row["actual_pnl_pct"]), 10.0, places=4)
        self.assertEqual(str(row["trade_decision"]), "Taken")

        snapshot = build_signal_review_snapshot(updated)
        self.assertEqual(snapshot["taken"], 1.0)
        self.assertEqual(snapshot["actual_closed"], 1.0)
        self.assertEqual(snapshot["actual_win_rate"], 100.0)
        self.assertAlmostEqual(snapshot["avg_actual_pnl"], 10.0, places=4)

    def test_save_signal_trade_journal_computes_short_trade_pnl_and_can_clear(self) -> None:
        event = {
            "source": "Market",
            "symbol": "BTC",
            "timeframe": "1h",
            "event_time": "2026-04-04T12:00:00Z",
            "direction": "Downside",
            "setup_confirm": "WATCH",
            "price": 100.0,
        }
        log_signal_events([event], self.db_path)
        df = fetch_signal_events_df(limit=10, db_path=self.db_path)
        signal_key = str(df.iloc[0]["signal_key"])

        self.assertTrue(
            save_signal_trade_journal(
                signal_key,
                actual_trade_side="Short",
                actual_entry_price=100,
                actual_entry_at="2026-04-04T12:10:00Z",
                actual_exit_price=90,
                actual_exit_at="2026-04-04T16:00:00Z",
                actual_exit_reason="Manual Exit",
                db_path=self.db_path,
            )
        )
        updated = fetch_signal_events_df(limit=10, db_path=self.db_path)
        row = updated.iloc[0]
        self.assertAlmostEqual(float(row["actual_pnl_pct"]), 10.0, places=4)

        self.assertTrue(save_signal_trade_journal(signal_key, db_path=self.db_path))
        cleared = fetch_signal_events_df(limit=10, db_path=self.db_path)
        row = cleared.iloc[0]
        self.assertTrue(pd.isna(row["actual_trade_status"]))
        self.assertTrue(pd.isna(row["actual_pnl_pct"]))

    def test_build_execution_overlay_snapshot_highlights_execution_gap_and_skipped_winners(self) -> None:
        df = pd.DataFrame(
            [
                {
                    "trade_decision": "Taken",
                    "status": "RESOLVED",
                    "directional_return_pct": 8.0,
                    "actual_trade_status": "CLOSED",
                    "actual_pnl_pct": 5.0,
                },
                {
                    "trade_decision": "Taken",
                    "status": "RESOLVED",
                    "directional_return_pct": -3.0,
                    "actual_trade_status": "CLOSED",
                    "actual_pnl_pct": -4.0,
                },
                {
                    "trade_decision": "Skipped",
                    "status": "RESOLVED",
                    "directional_return_pct": 6.0,
                    "actual_trade_status": "",
                    "actual_pnl_pct": None,
                },
            ]
        )
        snapshot = build_execution_overlay_snapshot(df)
        self.assertEqual(snapshot["total"], 3.0)
        self.assertEqual(snapshot["overlay_marked"], 3.0)
        self.assertEqual(snapshot["overlay_coverage_pct"], 100.0)
        self.assertEqual(snapshot["taken"], 2.0)
        self.assertEqual(snapshot["taken_resolved"], 2.0)
        self.assertEqual(snapshot["actual_closed"], 2.0)
        self.assertEqual(snapshot["journal_coverage_pct"], 100.0)
        self.assertEqual(snapshot["actual_win_rate"], 50.0)
        self.assertAlmostEqual(snapshot["avg_actual_pnl"], 0.5, places=4)
        self.assertAlmostEqual(snapshot["avg_signal_dir_return_on_taken"], 2.5, places=4)
        self.assertAlmostEqual(snapshot["execution_gap_pct"], -2.0, places=4)
        self.assertEqual(snapshot["skipped_winners"], 1.0)
        self.assertEqual(snapshot["skipped_winner_rate"], 100.0)


if __name__ == "__main__":
    unittest.main()
