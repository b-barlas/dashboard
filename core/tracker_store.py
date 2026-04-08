"""Storage resolution helpers for signal/alert tracking persistence."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import os
import shutil
import sqlite3
import tempfile
from pathlib import Path


_DEFAULT_DB_NAME = "signal_tracker.sqlite3"
_ENV_DB_PATH_KEYS = ("SIGNAL_TRACKER_DB_PATH", "TRACKER_DB_PATH")
_ENV_MIRROR_DIR_KEYS = (
    "SIGNAL_TRACKER_MIRROR_DIR",
    "TRACKER_MIRROR_DIR",
    "SIGNAL_TRACKER_BACKUP_DIR",
    "TRACKER_BACKUP_DIR",
)
_ENV_MIRROR_KEEP_KEYS = ("SIGNAL_TRACKER_MIRROR_KEEP", "TRACKER_MIRROR_KEEP")
_ENV_MIRROR_MINUTES_KEYS = ("SIGNAL_TRACKER_MIRROR_MINUTES", "TRACKER_MIRROR_MINUTES")
_ENV_AUTO_RESTORE_KEYS = ("SIGNAL_TRACKER_AUTO_RESTORE", "TRACKER_AUTO_RESTORE")


@dataclass(frozen=True)
class TrackerStorageSnapshot:
    path: str
    source: str
    label: str
    tone: str
    note: str
    exists: bool
    size_bytes: int
    filename: str
    mirror_enabled: bool
    mirror_dir: str
    mirror_count: int
    mirror_latest_path: str
    mirror_latest_age_minutes: int
    mirror_note: str
    durability_label: str
    durability_tone: str
    durability_note: str
    primary_valid: bool
    auto_restore_enabled: bool
    recovery_status: str
    recovery_note: str


@dataclass(frozen=True)
class TrackerRestoreResult:
    path: str
    backup_path: str
    restored_size: int


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _default_data_dir() -> Path:
    path = _repo_root() / "data"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _env_db_path() -> str | None:
    for key in _ENV_DB_PATH_KEYS:
        value = str(os.environ.get(key) or "").strip()
        if value:
            return value
    return None


def _env_mirror_dir() -> str | None:
    for key in _ENV_MIRROR_DIR_KEYS:
        value = str(os.environ.get(key) or "").strip()
        if value:
            return value
    return None


def _env_int(keys: tuple[str, ...], default: int) -> int:
    for key in keys:
        raw = str(os.environ.get(key) or "").strip()
        if not raw:
            continue
        try:
            value = int(raw)
        except Exception:
            continue
        if value > 0:
            return value
    return int(default)


def _env_bool(keys: tuple[str, ...], default: bool) -> bool:
    for key in keys:
        raw = str(os.environ.get(key) or "").strip().lower()
        if not raw:
            continue
        if raw in {"1", "true", "yes", "on"}:
            return True
        if raw in {"0", "false", "no", "off"}:
            return False
    return bool(default)


def _resolve_path(raw: str) -> Path:
    path = Path(str(raw).strip())
    if not path.is_absolute():
        path = (_repo_root() / path).resolve()
    return path


def resolve_signal_tracker_db_path(db_path: str | None = None) -> str:
    raw = str(db_path or _env_db_path() or (_default_data_dir() / _DEFAULT_DB_NAME)).strip()
    path = _resolve_path(raw)
    path.parent.mkdir(parents=True, exist_ok=True)
    return str(path)


def resolve_signal_tracker_mirror_dir(mirror_dir: str | None = None) -> str:
    raw = str(mirror_dir or _env_mirror_dir() or "").strip()
    if not raw:
        return ""
    path = _resolve_path(raw)
    path.mkdir(parents=True, exist_ok=True)
    return str(path)


def _list_tracker_mirror_paths(resolved: Path, *, mirror_dir: str | None = None) -> list[Path]:
    mirror_root = str(resolve_signal_tracker_mirror_dir(mirror_dir) or "").strip()
    if not mirror_root:
        return []
    root = Path(mirror_root)
    pattern = f"{resolved.stem}.mirror-*{resolved.suffix or '.sqlite3'}"
    return sorted(
        [path for path in root.glob(pattern) if path.is_file()],
        key=lambda item: item.stat().st_mtime,
        reverse=True,
    )


def latest_signal_tracker_mirror_path(db_path: str | None = None, *, mirror_dir: str | None = None) -> str:
    resolved = Path(resolve_signal_tracker_db_path(db_path))
    latest = next(iter(_list_tracker_mirror_paths(resolved, mirror_dir=mirror_dir)), None)
    return str(latest or "")


def _tracker_file_is_valid(path: Path) -> bool:
    if not path.exists():
        return False
    try:
        _validate_tracker_db_file(path)
    except Exception:
        return False
    return True


def build_tracker_storage_snapshot(db_path: str | None = None) -> TrackerStorageSnapshot:
    raw_env = _env_db_path()
    resolved = Path(resolve_signal_tracker_db_path(db_path))
    path_text = str(resolved)
    exists = resolved.exists()
    size_bytes = int(resolved.stat().st_size) if exists else 0
    mirror_dir = str(resolve_signal_tracker_mirror_dir() or "").strip()
    mirror_paths = _list_tracker_mirror_paths(resolved, mirror_dir=mirror_dir)
    latest_mirror = mirror_paths[0] if mirror_paths else None
    primary_valid = _tracker_file_is_valid(resolved)
    auto_restore_enabled = _env_bool(_ENV_AUTO_RESTORE_KEYS, True)
    mirror_age_minutes = -1
    if latest_mirror is not None:
        try:
            age_seconds = max(
                0.0,
                datetime.now(timezone.utc).timestamp() - float(latest_mirror.stat().st_mtime),
            )
            mirror_age_minutes = int(round(age_seconds / 60.0))
        except Exception:
            mirror_age_minutes = -1

    repo_root = _repo_root()
    repo_data_dir = repo_root / "data"
    source = "explicit" if db_path else ("env_override" if raw_env else "default")
    in_repo_data = False
    in_tmp = path_text.startswith("/tmp/") or path_text.startswith("/var/folders/")
    try:
        in_repo_data = resolved.is_relative_to(repo_data_dir)
    except Exception:
        in_repo_data = False

    if in_tmp:
        label = "Ephemeral Storage"
        tone = "warning"
        note = (
            "Tracker history is pointing at a temporary filesystem path. This is fine for debugging, "
            "but archive memory can disappear on restart or cleanup."
        )
    elif source == "default" and in_repo_data:
        label = "Workspace Storage"
        tone = "warning"
        note = (
            "Tracker history is stored in the workspace data folder. This is durable on this machine, "
            "but not deploy-safe across cloud redeploys. Use SIGNAL_TRACKER_DB_PATH or TRACKER_DB_PATH "
            "to point at a persistent mount or synced path when needed."
        )
    elif source in {"explicit", "env_override"}:
        label = "Custom Storage Override"
        tone = "positive"
        note = (
            "Tracker history is using a custom DB path override. That is the right direction for durable setups, "
            "but the true durability still depends on the backing volume or mount behind that path."
        )
    else:
        label = "Custom Local Storage"
        tone = "neutral"
        note = "Tracker history is stored outside the default workspace path. Verify that this location is backed up as expected."

    if mirror_dir:
        if latest_mirror is not None and mirror_age_minutes >= 0:
            mirror_note = (
                f"Rolling mirror snapshots are enabled at {mirror_dir}. "
                f"Latest mirror is {mirror_age_minutes} minutes old."
            )
        else:
            mirror_note = (
                f"Rolling mirror snapshots are enabled at {mirror_dir}, but no snapshot has been written yet."
            )
    else:
        mirror_note = (
            "Rolling mirror snapshots are not configured. Set SIGNAL_TRACKER_MIRROR_DIR or TRACKER_MIRROR_DIR "
            "to keep durable backup copies on a persistent mount."
        )

    if in_tmp:
        durability_label = "At Risk"
        durability_tone = "negative"
        durability_note = "Temporary storage path. Archive memory can disappear on restart or cleanup."
    elif source == "default" and in_repo_data:
        durability_label = "Local Only"
        durability_tone = "warning"
        durability_note = (
            "Good for this machine, but not deploy-safe. Use SIGNAL_TRACKER_DB_PATH plus a persistent mount, "
            "and ideally configure a mirror dir as well."
        )
    elif source in {"explicit", "env_override"} and mirror_dir and primary_valid:
        durability_label = "Deploy-Ready Path"
        durability_tone = "positive"
        durability_note = (
            "Custom storage override is active and rolling mirror snapshots are configured. "
            "This is the right shape for durable deploy memory, assuming the backing mount is persistent."
        )
    elif source in {"explicit", "env_override"}:
        durability_label = "Override, Mirror Missing"
        durability_tone = "warning"
        durability_note = (
            "Custom storage override is active, but there is no mirror rail. This can still work, "
            "but recovery is weaker than it should be."
        )
    else:
        durability_label = "Needs Verification"
        durability_tone = "neutral"
        durability_note = "Storage path is custom, but you should still verify that the backing location is truly persistent."

    if primary_valid:
        recovery_status = "Healthy"
        recovery_note = (
            "Primary tracker DB is valid."
            + (
                " Auto-restore from mirror is armed if this file disappears later."
                if mirror_dir and auto_restore_enabled
                else ""
            )
        )
    elif latest_mirror is not None and auto_restore_enabled:
        recovery_status = "Recoverable from Mirror"
        recovery_note = (
            "Primary tracker DB is missing or invalid, but the latest mirror snapshot can restore it automatically on startup."
        )
    elif latest_mirror is not None:
        recovery_status = "Mirror Available"
        recovery_note = (
            "A mirror snapshot exists, but auto-restore is disabled. Manual restore is still available from Signal Review."
        )
    elif exists:
        recovery_status = "At Risk"
        recovery_note = "Primary tracker DB exists but does not validate cleanly, and there is no mirror snapshot to recover from."
    else:
        recovery_status = "Fresh / Empty"
        recovery_note = "No tracker archive exists yet. The first live scan will create it."

    return TrackerStorageSnapshot(
        path=path_text,
        source=source,
        label=label,
        tone=tone,
        note=note,
        exists=exists,
        size_bytes=size_bytes,
        filename=resolved.name,
        mirror_enabled=bool(mirror_dir),
        mirror_dir=mirror_dir,
        mirror_count=int(len(mirror_paths)),
        mirror_latest_path=str(latest_mirror or ""),
        mirror_latest_age_minutes=int(mirror_age_minutes),
        mirror_note=mirror_note,
        durability_label=durability_label,
        durability_tone=durability_tone,
        durability_note=durability_note,
        primary_valid=bool(primary_valid),
        auto_restore_enabled=bool(auto_restore_enabled),
        recovery_status=recovery_status,
        recovery_note=recovery_note,
    )


def read_signal_tracker_db_bytes(db_path: str | None = None) -> bytes:
    resolved = Path(resolve_signal_tracker_db_path(db_path))
    if not resolved.exists():
        return b""
    with tempfile.TemporaryDirectory() as tmp:
        snapshot_path = Path(tmp) / resolved.name
        _sqlite_backup_file(resolved, snapshot_path)
        return snapshot_path.read_bytes()


def _sqlite_backup_file(source_path: Path, target_path: Path) -> None:
    if not source_path.exists():
        raise FileNotFoundError(str(source_path))
    target_path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(str(source_path)) as src, sqlite3.connect(str(target_path)) as dest:
        src.backup(dest)


def _tracker_backup_path(resolved: Path) -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    suffix = resolved.suffix or ".sqlite3"
    return resolved.with_name(f"{resolved.stem}.backup-{stamp}{suffix}")


def _tracker_quarantine_path(resolved: Path) -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    suffix = resolved.suffix or ".sqlite3"
    return resolved.with_name(f"{resolved.stem}.invalid-{stamp}{suffix}")


def _tracker_mirror_path(resolved: Path, mirror_dir: Path) -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    suffix = resolved.suffix or ".sqlite3"
    return mirror_dir / f"{resolved.stem}.mirror-{stamp}{suffix}"


def backup_signal_tracker_db(db_path: str | None = None) -> str:
    resolved = Path(resolve_signal_tracker_db_path(db_path))
    if not resolved.exists():
        return ""
    backup_path = _tracker_backup_path(resolved)
    _sqlite_backup_file(resolved, backup_path)
    return str(backup_path)


def quarantine_signal_tracker_db(db_path: str | None = None) -> str:
    resolved = Path(resolve_signal_tracker_db_path(db_path))
    if not resolved.exists():
        return ""
    quarantine_path = _tracker_quarantine_path(resolved)
    shutil.copy2(resolved, quarantine_path)
    return str(quarantine_path)


def create_signal_tracker_mirror_snapshot(
    db_path: str | None = None,
    *,
    mirror_dir: str | None = None,
    keep: int | None = None,
) -> str:
    resolved = Path(resolve_signal_tracker_db_path(db_path))
    mirror_root = str(resolve_signal_tracker_mirror_dir(mirror_dir) or "").strip()
    if not resolved.exists() or not mirror_root:
        return ""
    root = Path(mirror_root)
    snapshot_path = _tracker_mirror_path(resolved, root)
    _sqlite_backup_file(resolved, snapshot_path)
    keep_count = max(1, int(keep or _env_int(_ENV_MIRROR_KEEP_KEYS, 24)))
    for stale_path in _list_tracker_mirror_paths(resolved, mirror_dir=mirror_root)[keep_count:]:
        try:
            stale_path.unlink()
        except Exception:
            continue
    return str(snapshot_path)


def mirror_signal_tracker_db_if_due(
    db_path: str | None = None,
    *,
    mirror_dir: str | None = None,
    min_minutes: int | None = None,
    keep: int | None = None,
) -> str:
    resolved = Path(resolve_signal_tracker_db_path(db_path))
    mirror_root = str(resolve_signal_tracker_mirror_dir(mirror_dir) or "").strip()
    if not resolved.exists() or not mirror_root:
        return ""
    latest = next(iter(_list_tracker_mirror_paths(resolved, mirror_dir=mirror_root)), None)
    threshold_minutes = max(1, int(min_minutes or _env_int(_ENV_MIRROR_MINUTES_KEYS, 60)))
    if latest is not None:
        try:
            age_seconds = max(0.0, datetime.now(timezone.utc).timestamp() - float(latest.stat().st_mtime))
            if age_seconds < (float(threshold_minutes) * 60.0):
                return ""
        except Exception:
            pass
    return create_signal_tracker_mirror_snapshot(
        str(resolved),
        mirror_dir=mirror_root,
        keep=keep,
    )


def recover_signal_tracker_db_from_latest_mirror(
    db_path: str | None = None,
    *,
    mirror_dir: str | None = None,
    auto_restore: bool | None = None,
) -> TrackerRestoreResult | None:
    resolved = Path(resolve_signal_tracker_db_path(db_path))
    primary_valid = _tracker_file_is_valid(resolved)
    if primary_valid:
        return None
    latest_path = latest_signal_tracker_mirror_path(str(resolved), mirror_dir=mirror_dir)
    if not latest_path:
        return None
    if not bool(_env_bool(_ENV_AUTO_RESTORE_KEYS, True) if auto_restore is None else auto_restore):
        return None

    backup_path = ""
    if resolved.exists() and not primary_valid:
        backup_path = quarantine_signal_tracker_db(str(resolved))
    restore_result = restore_signal_tracker_db_bytes(
        Path(latest_path).read_bytes(),
        db_path=str(resolved),
        backup_existing=False,
    )
    if backup_path and not str(restore_result.backup_path or "").strip():
        return TrackerRestoreResult(
            path=restore_result.path,
            backup_path=backup_path,
            restored_size=restore_result.restored_size,
        )
    return restore_result


def _validate_tracker_db_file(path: Path) -> None:
    with sqlite3.connect(str(path)) as conn:
        names = {
            str(row[0])
            for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
        }
    required = {"signal_events", "market_alerts"}
    missing = sorted(required - names)
    if missing:
        raise ValueError(f"Missing tracker tables: {', '.join(missing)}")


def restore_signal_tracker_db_bytes(
    data: bytes,
    db_path: str | None = None,
    *,
    backup_existing: bool = True,
) -> TrackerRestoreResult:
    payload = bytes(data or b"")
    if len(payload) < 32 or not payload.startswith(b"SQLite format 3\x00"):
        raise ValueError("Uploaded file is not a valid SQLite tracker snapshot.")

    resolved = Path(resolve_signal_tracker_db_path(db_path))
    resolved.parent.mkdir(parents=True, exist_ok=True)
    backup_path = ""
    if backup_existing and resolved.exists():
        backup_path = backup_signal_tracker_db(str(resolved))

    wal_path = resolved.with_name(f"{resolved.name}-wal")
    shm_path = resolved.with_name(f"{resolved.name}-shm")
    temp_restore = resolved.with_name(f"{resolved.stem}.restore_tmp{resolved.suffix or '.sqlite3'}")
    temp_restore.write_bytes(payload)
    try:
        _validate_tracker_db_file(temp_restore)
        shutil.move(str(temp_restore), str(resolved))
        if wal_path.exists():
            wal_path.unlink()
        if shm_path.exists():
            shm_path.unlink()
    except Exception:
        if temp_restore.exists():
            temp_restore.unlink()
        if backup_path and Path(backup_path).exists():
            shutil.copy2(backup_path, resolved)
        raise

    return TrackerRestoreResult(
        path=str(resolved),
        backup_path=str(backup_path or ""),
        restored_size=int(resolved.stat().st_size) if resolved.exists() else 0,
    )


def connect_signal_tracker_db(db_path: str | None = None) -> sqlite3.Connection:
    path = resolve_signal_tracker_db_path(db_path)
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA synchronous=NORMAL;")
    return conn
