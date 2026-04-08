from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from core.signal_tracker import (
    annotate_alert_footprint,
    build_alert_effectiveness_summary,
    build_actual_exit_quality_profile,
    build_actual_trade_hold_profile,
    build_recent_market_context_snapshot,
    build_recent_symbol_market_signal_snapshot,
    build_signal_cohort_summary,
    build_execution_overlay_snapshot,
    build_signal_review_snapshot,
    fetch_market_alerts_df,
    fetch_signal_events_df,
    init_signal_tracker_db,
    log_market_alerts,
    log_signal_events,
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
        self.assertEqual(log_signal_events([event], self.db_path), 1)
        df = fetch_signal_events_df(limit=20, db_path=self.db_path)
        self.assertEqual(len(df), 1)
        self.assertEqual(str(df.iloc[0]["symbol"]), "BTC")
        self.assertEqual(str(df.iloc[0]["session_bucket"]), "European (08-16 UTC)")

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
        self.assertEqual(snapshot["taken"], 2.0)
        self.assertEqual(snapshot["taken_resolved"], 2.0)
        self.assertEqual(snapshot["actual_closed"], 2.0)
        self.assertEqual(snapshot["actual_win_rate"], 50.0)
        self.assertAlmostEqual(snapshot["avg_actual_pnl"], 0.5, places=4)
        self.assertAlmostEqual(snapshot["avg_signal_dir_return_on_taken"], 2.5, places=4)
        self.assertAlmostEqual(snapshot["execution_gap_pct"], -2.0, places=4)
        self.assertEqual(snapshot["skipped_winners"], 1.0)
        self.assertEqual(snapshot["skipped_winner_rate"], 100.0)


if __name__ == "__main__":
    unittest.main()
