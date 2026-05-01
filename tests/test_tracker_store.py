from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from core.signal_tracker import (
    fetch_market_alerts_df,
    fetch_signal_events_df,
    init_signal_tracker_db,
    log_market_alerts,
    log_signal_events,
)
from core.tracker_store import (
    backup_signal_tracker_db,
    build_tracker_storage_snapshot,
    create_signal_tracker_mirror_snapshot,
    latest_signal_tracker_mirror_path,
    latest_signal_tracker_backup_path,
    mirror_signal_tracker_db_if_due,
    quarantine_signal_tracker_db,
    read_signal_tracker_db_bytes,
    recover_signal_tracker_db_from_latest_mirror,
    resolve_signal_tracker_mirror_dir,
    resolve_signal_tracker_db_path,
    restore_signal_tracker_db_bytes,
)


class TrackerStoreTests(unittest.TestCase):
    def test_resolve_signal_tracker_db_path_uses_explicit_path(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = resolve_signal_tracker_db_path(str(Path(tmp) / "custom.sqlite3"))
            self.assertTrue(path.endswith("custom.sqlite3"))
            self.assertTrue(Path(path).parent.exists())

    def test_resolve_signal_tracker_db_path_honors_env_override(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            custom = str(Path(tmp) / "env-tracker.sqlite3")
            with patch.dict(os.environ, {"SIGNAL_TRACKER_DB_PATH": custom}, clear=False):
                path = resolve_signal_tracker_db_path()
            self.assertEqual(path, custom)

    def test_resolve_signal_tracker_mirror_dir_honors_env_override(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            custom = str(Path(tmp) / "mirror")
            with patch.dict(os.environ, {"SIGNAL_TRACKER_MIRROR_DIR": custom}, clear=False):
                path = resolve_signal_tracker_mirror_dir()
            self.assertEqual(path, custom)
            self.assertTrue(Path(path).exists())

    def test_build_tracker_storage_snapshot_flags_workspace_default_as_not_deploy_safe(self) -> None:
        snap = build_tracker_storage_snapshot()
        self.assertEqual(snap.label, "Workspace Storage")
        self.assertIn("not deploy-safe", snap.note)
        self.assertEqual(snap.durability_label, "Local Only")

    def test_build_tracker_storage_snapshot_marks_env_override_as_custom_storage(self) -> None:
        custom = "tracker_backups/env-tracker.sqlite3"
        with patch.dict(os.environ, {"SIGNAL_TRACKER_DB_PATH": custom}, clear=False):
            snap = build_tracker_storage_snapshot()
        self.assertEqual(snap.label, "Custom Storage Override")
        self.assertIn("custom DB path override", snap.note)
        self.assertIn("Mirror", snap.durability_label)

    def test_build_tracker_storage_snapshot_reports_mirror_rail_when_configured(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            db_path = str(Path(tmp) / "tracker.sqlite3")
            mirror_dir = str(Path(tmp) / "mirror")
            init_signal_tracker_db(db_path)
            with patch.dict(os.environ, {"SIGNAL_TRACKER_MIRROR_DIR": mirror_dir}, clear=False):
                snap = build_tracker_storage_snapshot(db_path)
            self.assertTrue(snap.mirror_enabled)
            self.assertEqual(snap.mirror_dir, mirror_dir)
            self.assertIn("no snapshot has been written yet", snap.mirror_note)
            self.assertEqual(snap.recovery_status, "Healthy")

    def test_backup_signal_tracker_db_creates_restorable_snapshot(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            db_path = str(Path(tmp) / "tracker.sqlite3")
            init_signal_tracker_db(db_path)
            log_signal_events(
                [
                    {
                        "source": "Market",
                        "symbol": "BTC",
                        "timeframe": "1h",
                        "event_time": "2026-04-05T10:00:00Z",
                        "direction": "UPSIDE",
                        "setup_confirm": "READY",
                        "market_trade_gate": "TRADEABLE",
                    }
                ],
                db_path=db_path,
            )
            backup_path = backup_signal_tracker_db(db_path)
            self.assertTrue(Path(backup_path).exists())
            restored_bytes = read_signal_tracker_db_bytes(backup_path)
            self.assertTrue(restored_bytes.startswith(b"SQLite format 3\x00"))
            self.assertEqual(latest_signal_tracker_backup_path(db_path), backup_path)

    def test_log_signal_events_persists_archive_decision_calibration(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            db_path = str(Path(tmp) / "tracker.sqlite3")
            init_signal_tracker_db(db_path)

            log_signal_events(
                [
                    {
                        "source": "Market",
                        "symbol": "TRX",
                        "timeframe": "1h",
                        "event_time": "2026-04-05T10:00:00Z",
                        "direction": "UPSIDE",
                        "setup_confirm": "WATCH_UP",
                        "archive_policy_delta": 1.25,
                        "archive_policy_completed": 36,
                        "archive_policy_quality": "Good",
                        "archive_policy_coverage": 0.72,
                        "archive_decision_delta": 2.5,
                        "archive_expectancy_delta": 0.7,
                        "archive_total_delta": 4.25,
                        "archive_total_expectancy_delta": 55.7,
                        "archive_decision_scope": "WATCH_UP 1H Upside",
                    }
                ],
                db_path=db_path,
            )

            row = fetch_signal_events_df(limit=20, db_path=db_path).iloc[0]
            self.assertAlmostEqual(float(row["archive_policy_delta"]), 1.25)
            self.assertEqual(int(row["archive_policy_completed"]), 36)
            self.assertEqual(row["archive_policy_quality"], "Good")
            self.assertAlmostEqual(float(row["archive_policy_coverage"]), 0.72)
            self.assertAlmostEqual(float(row["archive_decision_delta"]), 2.5)
            self.assertAlmostEqual(float(row["archive_expectancy_delta"]), 0.7)
            self.assertAlmostEqual(float(row["archive_total_delta"]), 4.25)
            self.assertAlmostEqual(float(row["archive_total_expectancy_delta"]), 55.7)
            self.assertEqual(row["archive_decision_scope"], "WATCH_UP 1H Upside")

    def test_restore_signal_tracker_db_bytes_replaces_current_db_and_keeps_backup(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            db_path = str(Path(tmp) / "tracker.sqlite3")
            init_signal_tracker_db(db_path)
            log_signal_events(
                [
                    {
                        "source": "Market",
                        "symbol": "BTC",
                        "timeframe": "1h",
                        "event_time": "2026-04-05T10:00:00Z",
                        "direction": "UPSIDE",
                        "setup_confirm": "READY",
                        "market_trade_gate": "TRADEABLE",
                    }
                ],
                db_path=db_path,
            )
            log_market_alerts(
                [
                    {
                        "alert_key": "LEAD_CLUSTER",
                        "state_signature": "btc-ready",
                        "severity": "INFO",
                        "title": "Lead cluster forming",
                        "note": "Test alert",
                    }
                ],
                db_path=db_path,
            )
            snapshot_bytes = read_signal_tracker_db_bytes(db_path)

            log_signal_events(
                [
                    {
                        "source": "Market",
                        "symbol": "ETH",
                        "timeframe": "4h",
                        "event_time": "2026-04-05T12:00:00Z",
                        "direction": "DOWNSIDE",
                        "setup_confirm": "WATCH",
                        "market_trade_gate": "SELECTIVE_ONLY",
                    }
                ],
                db_path=db_path,
            )
            result = restore_signal_tracker_db_bytes(snapshot_bytes, db_path=db_path, backup_existing=True)

            self.assertEqual(fetch_signal_events_df(limit=20, db_path=db_path)["symbol"].tolist(), ["BTC"])
            self.assertEqual(fetch_market_alerts_df(limit=20, db_path=db_path)["alert_key"].tolist(), ["LEAD_CLUSTER"])
            self.assertTrue(Path(result.path).exists())
            self.assertTrue(Path(result.backup_path).exists())
            self.assertGreater(result.restored_size, 0)

    def test_restore_signal_tracker_db_bytes_rejects_invalid_payload(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            db_path = str(Path(tmp) / "tracker.sqlite3")
            init_signal_tracker_db(db_path)
            with self.assertRaisesRegex(ValueError, "valid SQLite tracker snapshot"):
                restore_signal_tracker_db_bytes(b"not-a-sqlite-file", db_path=db_path)

    def test_create_signal_tracker_mirror_snapshot_and_if_due_keep_history_trimmed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            db_path = str(Path(tmp) / "tracker.sqlite3")
            mirror_dir = str(Path(tmp) / "mirror")
            init_signal_tracker_db(db_path)
            log_signal_events(
                [
                    {
                        "source": "Market",
                        "symbol": "BTC",
                        "timeframe": "1h",
                        "event_time": "2026-04-05T10:00:00Z",
                        "direction": "UPSIDE",
                    }
                ],
                db_path=db_path,
            )
            first_path = create_signal_tracker_mirror_snapshot(db_path, mirror_dir=mirror_dir, keep=2)
            second_path = create_signal_tracker_mirror_snapshot(db_path, mirror_dir=mirror_dir, keep=2)
            third_path = create_signal_tracker_mirror_snapshot(db_path, mirror_dir=mirror_dir, keep=2)
            mirror_files = sorted(Path(mirror_dir).glob("tracker.mirror-*.sqlite3"))
            self.assertTrue(bool(first_path))
            self.assertTrue(bool(second_path))
            self.assertTrue(Path(third_path).exists())
            self.assertEqual(len(mirror_files), 2)
            self.assertEqual(mirror_signal_tracker_db_if_due(db_path, mirror_dir=mirror_dir, min_minutes=60, keep=2), "")
            self.assertEqual(latest_signal_tracker_mirror_path(db_path, mirror_dir=mirror_dir), third_path)

    def test_recover_signal_tracker_db_from_latest_mirror_restores_missing_primary(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            db_path = str(Path(tmp) / "tracker.sqlite3")
            mirror_dir = str(Path(tmp) / "mirror")
            init_signal_tracker_db(db_path)
            log_signal_events(
                [
                    {
                        "source": "Market",
                        "symbol": "BTC",
                        "timeframe": "1h",
                        "event_time": "2026-04-05T10:00:00Z",
                        "direction": "UPSIDE",
                        "setup_confirm": "READY",
                    }
                ],
                db_path=db_path,
            )
            create_signal_tracker_mirror_snapshot(db_path, mirror_dir=mirror_dir, keep=4)
            Path(db_path).unlink()

            result = recover_signal_tracker_db_from_latest_mirror(
                db_path,
                mirror_dir=mirror_dir,
                auto_restore=True,
            )
            self.assertIsNotNone(result)
            self.assertTrue(Path(db_path).exists())
            self.assertEqual(fetch_signal_events_df(limit=20, db_path=db_path)["symbol"].tolist(), ["BTC"])

    def test_recover_signal_tracker_db_from_latest_mirror_quarantines_invalid_primary(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            db_path = str(Path(tmp) / "tracker.sqlite3")
            mirror_dir = str(Path(tmp) / "mirror")
            init_signal_tracker_db(db_path)
            log_signal_events(
                [
                    {
                        "source": "Market",
                        "symbol": "BTC",
                        "timeframe": "1h",
                        "event_time": "2026-04-05T10:00:00Z",
                        "direction": "UPSIDE",
                    }
                ],
                db_path=db_path,
            )
            create_signal_tracker_mirror_snapshot(db_path, mirror_dir=mirror_dir, keep=4)
            Path(db_path).write_bytes(b"corrupt-db")
            wal_path = Path(f"{db_path}-wal")
            shm_path = Path(f"{db_path}-shm")
            if wal_path.exists():
                wal_path.unlink()
            if shm_path.exists():
                shm_path.unlink()

            result = recover_signal_tracker_db_from_latest_mirror(
                db_path,
                mirror_dir=mirror_dir,
                auto_restore=True,
            )
            self.assertIsNotNone(result)
            self.assertTrue(Path(db_path).exists())
            self.assertTrue(bool(result.backup_path))
            self.assertTrue(Path(result.backup_path).exists())
            self.assertEqual(fetch_signal_events_df(limit=20, db_path=db_path)["symbol"].tolist(), ["BTC"])

    def test_recover_signal_tracker_db_from_latest_backup_when_mirror_is_missing(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            db_path = str(Path(tmp) / "tracker.sqlite3")
            init_signal_tracker_db(db_path)
            log_signal_events(
                [
                    {
                        "source": "Market",
                        "symbol": "BTC",
                        "timeframe": "1h",
                        "event_time": "2026-04-05T10:00:00Z",
                        "direction": "UPSIDE",
                    }
                ],
                db_path=db_path,
            )
            backup_path = backup_signal_tracker_db(db_path)
            Path(db_path).write_bytes(b"corrupt-db")
            Path(f"{db_path}-wal").unlink(missing_ok=True)
            Path(f"{db_path}-shm").unlink(missing_ok=True)

            result = recover_signal_tracker_db_from_latest_mirror(db_path, auto_restore=True)

            self.assertIsNotNone(result)
            self.assertEqual(fetch_signal_events_df(limit=20, db_path=db_path)["symbol"].tolist(), ["BTC"])
            self.assertEqual(latest_signal_tracker_backup_path(db_path), backup_path)

    def test_init_signal_tracker_db_moves_invalid_primary_aside_without_backup(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            db_path = str(Path(tmp) / "tracker.sqlite3")
            Path(db_path).write_bytes(b"corrupt-db")

            init_signal_tracker_db(db_path)

            self.assertEqual(fetch_signal_events_df(limit=20, db_path=db_path).to_dict("records"), [])
            invalid_paths = list(Path(tmp).glob("tracker.invalid-*.sqlite3"))
            self.assertEqual(len(invalid_paths), 1)

    def test_quarantine_signal_tracker_db_copies_invalid_primary(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            db_path = str(Path(tmp) / "tracker.sqlite3")
            Path(db_path).write_bytes(b"corrupt-db")
            quarantine_path = quarantine_signal_tracker_db(db_path)
            self.assertTrue(Path(quarantine_path).exists())
            self.assertEqual(Path(quarantine_path).read_bytes(), b"corrupt-db")


if __name__ == "__main__":
    unittest.main()
