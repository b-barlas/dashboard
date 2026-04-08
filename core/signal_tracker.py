"""Persistent signal outcome tracking for live dashboard signals."""

from __future__ import annotations

import hashlib
import math
import sqlite3
from collections.abc import Mapping, Sequence

import pandas as pd
from core.session_utils import session_bucket_for_timestamp
from core.tracker_store import (
    connect_signal_tracker_db,
    mirror_signal_tracker_db_if_due,
    recover_signal_tracker_db_from_latest_mirror,
    resolve_signal_tracker_db_path,
)

_OPEN_STATUS = "OPEN"
_RESOLVED_STATUS = "RESOLVED"


def get_signal_tracker_db_path() -> str:
    return resolve_signal_tracker_db_path()


def _sync_tracker_mirror(db_path: str) -> None:
    try:
        mirror_signal_tracker_db_if_due(db_path)
    except Exception:
        return


def _utc_iso(value: object | None = None) -> str:
    ts = pd.Timestamp.utcnow() if value is None else pd.to_datetime(value, utc=True, errors="coerce")
    if pd.isna(ts):
        ts = pd.Timestamp.utcnow()
    return ts.strftime("%Y-%m-%dT%H:%M:%SZ")


def _signal_key(source: str, symbol: str, timeframe: str, event_time: object) -> str:
    raw = f"{str(source).strip().lower()}|{str(symbol).strip().upper()}|{str(timeframe).strip().lower()}|{_utc_iso(event_time)}"
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()


def _timeframe_horizon_bars(timeframe: str) -> int:
    key = str(timeframe or "").strip().lower()
    if key in {"15m", "30m"}:
        return 16
    if key in {"1h", "2h"}:
        return 12
    if key == "4h":
        return 12
    if key == "1d":
        return 10
    return 12


def _direction_key(direction: object) -> str:
    d = str(direction or "").strip().upper()
    if d in {"UPSIDE", "LONG", "BUY", "BULLISH"}:
        return "UPSIDE"
    if d in {"DOWNSIDE", "SHORT", "SELL", "BEARISH"}:
        return "DOWNSIDE"
    return "NEUTRAL"


def _trade_side_key(side: object) -> str:
    s = str(side or "").strip().upper()
    if s in {"LONG", "BUY", "UPSIDE"}:
        return "LONG"
    if s in {"SHORT", "SELL", "DOWNSIDE"}:
        return "SHORT"
    return ""


def _float_or_none(value: object) -> float | None:
    try:
        out = float(value)
    except Exception:
        return None
    if not math.isfinite(out):
        return None
    return out


def _text_or_empty(value: object) -> str:
    if value is None or pd.isna(value):
        return ""
    return str(value).strip()


_ALERT_KEY_DISPLAY = {
    "CATALYST_BLOCK": "Catalyst Block",
    "TRADE_GATE": "Trade Gate",
    "MARKET_LEAD": "Market Lead",
    "LEARNED_EDGE": "Learned Edge",
    "ACTIONABLE_CLUSTER": "Actionable Cluster",
    "ARCHIVE_GUARDRAIL": "Archive Guardrail",
    "EXECUTION_STANCE": "Execution Stance",
    "PLAYBOOK_WINDOW": "Playbook Window",
    "SECTOR_ROTATION": "Sector Rotation",
    "SESSION_FIT": "Session Fit",
    "FLOW_PROXY": "Flow Proxy",
    "CATALYST_CAUTION": "Catalyst Caution",
}


def _alert_key_display(value: object) -> str:
    key = str(value or "").strip().upper()
    return _ALERT_KEY_DISPLAY.get(key, key.replace("_", " ").title() if key else "No Alert Footprint")


def _split_alert_keys(value: object) -> list[str]:
    text = _text_or_empty(value)
    if not text:
        return []
    return [chunk.strip().upper() for chunk in text.split("|") if str(chunk).strip()]


def _infer_alert_keys_from_event_row(row: Mapping[str, object]) -> list[str]:
    stored = _split_alert_keys(row.get("market_alert_keys"))
    if stored:
        return stored

    keys: list[str] = []
    market_catalyst_blocking = bool(_float_or_none(row.get("market_catalyst_blocking")) or 0.0)
    catalyst_state = _text_or_empty(row.get("market_catalyst_state")).upper()
    trade_gate = _text_or_empty(row.get("market_trade_gate"))
    market_lead_label = _text_or_empty(row.get("market_lead_label")).upper()
    market_lead_score = _float_or_none(row.get("market_lead_score")) or 50.0
    flow_bias = _text_or_empty(row.get("market_flow_bias")).upper()
    sector_rotation = _text_or_empty(row.get("market_sector_rotation"))
    adaptive_edge = _text_or_empty(row.get("adaptive_edge_label"))
    archive_guardrail = _text_or_empty(row.get("archive_guardrail_label"))

    if market_catalyst_blocking:
        keys.append("CATALYST_BLOCK")
    elif "CAUTION" in catalyst_state:
        keys.append("CATALYST_CAUTION")

    if trade_gate in {"No-Trade", "Defensive Only"}:
        keys.append("TRADE_GATE")

    if (market_lead_label == "UPSIDE" and market_lead_score >= 65.0) or (
        market_lead_label == "DOWNSIDE" and market_lead_score <= 35.0
    ):
        keys.append("MARKET_LEAD")

    if adaptive_edge in {"Historically Favored", "Historically Weak"}:
        keys.append("LEARNED_EDGE")

    if archive_guardrail in {"Archive Guardrail", "Archive Caution"}:
        keys.append("ARCHIVE_GUARDRAIL")

    if flow_bias in {"SHORT_CROWDING", "LONG_CROWDING"}:
        keys.append("FLOW_PROXY")

    if sector_rotation not in {"", "Unknown", "Mixed Sector Rotation", "None"}:
        keys.append("SECTOR_ROTATION")

    if trade_gate or adaptive_edge or archive_guardrail:
        if trade_gate == "No-Trade" or archive_guardrail == "Archive Guardrail":
            keys.append("EXECUTION_STANCE")
        elif trade_gate == "Defensive Only" or adaptive_edge == "Historically Weak":
            keys.append("EXECUTION_STANCE")
        elif trade_gate in {"Tradeable", "Selective Only"} and adaptive_edge == "Historically Favored":
            keys.append("EXECUTION_STANCE")

    deduped: list[str] = []
    for key in keys:
        normalized = str(key).strip().upper()
        if normalized and normalized not in deduped:
            deduped.append(normalized)
    return deduped


def _connect(db_path: str | None = None):
    return connect_signal_tracker_db(db_path)


def _ensure_signal_tracker_columns(conn: sqlite3.Connection) -> None:
    existing = {
        str(row["name"])
        for row in conn.execute("PRAGMA table_info(signal_events)").fetchall()
    }
    expected = {
        "session_bucket": "TEXT",
        "market_regime": "TEXT",
        "market_playbook": "TEXT",
        "market_no_trade": "INTEGER",
        "market_trade_gate": "TEXT",
        "market_alert_keys": "TEXT",
        "market_primary_alert": "TEXT",
        "market_no_trade_reason": "TEXT",
        "risk_tier": "TEXT",
        "risk_unit_fraction": "REAL",
        "sector_tag": "TEXT",
        "market_sector_rotation": "TEXT",
        "market_catalyst_state": "TEXT",
        "market_catalyst_event": "TEXT",
        "market_catalyst_blocking": "INTEGER",
        "market_catalyst_category": "TEXT",
        "market_catalyst_scope": "TEXT",
        "market_catalyst_tag": "TEXT",
        "market_catalyst_targeted": "INTEGER",
        "market_catalyst_window": "TEXT",
        "market_flow_state": "TEXT",
        "market_flow_bias": "TEXT",
        "adaptive_edge_label": "TEXT",
        "adaptive_edge_score": "REAL",
        "archive_guardrail_label": "TEXT",
        "archive_guardrail_penalty": "REAL",
        "archive_guardrail_note": "TEXT",
        "trade_decision": "TEXT",
        "trade_note": "TEXT",
        "trade_marked_at": "TEXT",
        "actual_trade_side": "TEXT",
        "actual_entry_price": "REAL",
        "actual_entry_at": "TEXT",
        "actual_exit_price": "REAL",
        "actual_exit_at": "TEXT",
        "actual_exit_reason": "TEXT",
        "actual_pnl_pct": "REAL",
        "actual_trade_status": "TEXT",
    }
    for column, column_type in expected.items():
        if column in existing:
            continue
        conn.execute(f"ALTER TABLE signal_events ADD COLUMN {column} {column_type}")


def init_signal_tracker_db(db_path: str | None = None) -> str:
    path = resolve_signal_tracker_db_path(db_path or get_signal_tracker_db_path())
    try:
        recover_signal_tracker_db_from_latest_mirror(path)
    except Exception:
        pass
    with _connect(path) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS signal_events (
                signal_key TEXT PRIMARY KEY,
                source TEXT NOT NULL,
                symbol TEXT NOT NULL,
                timeframe TEXT NOT NULL,
                event_time TEXT NOT NULL,
                session_bucket TEXT,
                horizon_bars INTEGER NOT NULL,
                direction TEXT NOT NULL,
                setup_confirm TEXT,
                action_reason TEXT,
                lead_label TEXT,
                lead_direction TEXT,
                lead_active INTEGER NOT NULL DEFAULT 0,
                confidence REAL,
                ai_ensemble TEXT,
                ai_direction TEXT,
                ai_confidence REAL,
                ai_aligned INTEGER,
                market_lead_label TEXT,
                market_lead_score REAL,
                market_lead_upside INTEGER,
                market_lead_downside INTEGER,
                market_lead_aligned INTEGER,
                market_regime TEXT,
                market_playbook TEXT,
                market_no_trade INTEGER,
                market_trade_gate TEXT,
                market_alert_keys TEXT,
                market_primary_alert TEXT,
                market_no_trade_reason TEXT,
                risk_tier TEXT,
                risk_unit_fraction REAL,
                sector_tag TEXT,
                market_sector_rotation TEXT,
                market_catalyst_state TEXT,
                market_catalyst_event TEXT,
                market_catalyst_blocking INTEGER,
                market_catalyst_category TEXT,
                market_catalyst_scope TEXT,
                market_catalyst_tag TEXT,
                market_catalyst_targeted INTEGER,
                market_catalyst_window TEXT,
                market_flow_state TEXT,
                market_flow_bias TEXT,
                adaptive_edge_label TEXT,
                adaptive_edge_score REAL,
                archive_guardrail_label TEXT,
                archive_guardrail_penalty REAL,
                archive_guardrail_note TEXT,
                trade_decision TEXT,
                trade_note TEXT,
                trade_marked_at TEXT,
                actual_trade_side TEXT,
                actual_entry_price REAL,
                actual_entry_at TEXT,
                actual_exit_price REAL,
                actual_exit_at TEXT,
                actual_exit_reason TEXT,
                actual_pnl_pct REAL,
                actual_trade_status TEXT,
                price REAL,
                delta_pct REAL,
                entry_price REAL,
                stop_loss REAL,
                target_price REAL,
                rr_ratio REAL,
                has_plan INTEGER NOT NULL DEFAULT 0,
                status TEXT NOT NULL DEFAULT 'OPEN',
                plan_outcome TEXT,
                directional_return_pct REAL,
                favorable_excursion_pct REAL,
                adverse_excursion_pct REAL,
                resolved_at TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
            """
        )
        _ensure_signal_tracker_columns(conn)
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_signal_events_open ON signal_events(status, source, symbol, timeframe, event_time)"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_signal_events_source_time ON signal_events(source, event_time)"
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS market_alerts (
                source TEXT NOT NULL,
                alert_key TEXT NOT NULL,
                state_signature TEXT NOT NULL,
                severity TEXT NOT NULL,
                title TEXT NOT NULL,
                note TEXT NOT NULL,
                active INTEGER NOT NULL DEFAULT 1,
                first_seen_at TEXT NOT NULL,
                last_seen_at TEXT NOT NULL,
                times_seen INTEGER NOT NULL DEFAULT 1,
                PRIMARY KEY (source, alert_key, state_signature)
            )
            """
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_market_alerts_source_active ON market_alerts(source, active, last_seen_at)"
        )
        conn.commit()
    return path


def log_signal_events(events: Sequence[Mapping[str, object]], db_path: str | None = None) -> int:
    if not events:
        return 0
    path = init_signal_tracker_db(db_path)
    written = 0
    with _connect(path) as conn:
        now_iso = _utc_iso()
        for event in events:
            source = str(event.get("source") or "Market").strip() or "Market"
            symbol = str(event.get("symbol") or "").strip().upper()
            timeframe = str(event.get("timeframe") or "").strip().lower()
            event_time = event.get("event_time")
            if not symbol or not timeframe or event_time is None:
                continue
            signal_key = _signal_key(source, symbol, timeframe, event_time)
            direction = _direction_key(event.get("direction"))
            lead_direction = _direction_key(event.get("lead_direction"))
            ai_direction = _direction_key(event.get("ai_direction"))
            market_lead_label = str(event.get("market_lead_label") or "").strip()
            market_lead_direction = _direction_key(market_lead_label)
            horizon_bars = int(event.get("horizon_bars") or _timeframe_horizon_bars(timeframe))
            session_bucket = str(event.get("session_bucket") or session_bucket_for_timestamp(event_time)).strip()
            payload = {
                "signal_key": signal_key,
                "source": source,
                "symbol": symbol,
                "timeframe": timeframe,
                "event_time": _utc_iso(event_time),
                "session_bucket": session_bucket,
                "horizon_bars": horizon_bars,
                "direction": direction,
                "setup_confirm": str(event.get("setup_confirm") or "").strip(),
                "action_reason": str(event.get("action_reason") or "").strip(),
                "lead_label": str(event.get("lead_label") or "").strip(),
                "lead_direction": lead_direction,
                "lead_active": int(lead_direction in {"UPSIDE", "DOWNSIDE"}),
                "confidence": _float_or_none(event.get("confidence")),
                "ai_ensemble": str(event.get("ai_ensemble") or "").strip(),
                "ai_direction": ai_direction,
                "ai_confidence": _float_or_none(event.get("ai_confidence")),
                "ai_aligned": int(direction in {"UPSIDE", "DOWNSIDE"} and ai_direction == direction),
                "market_lead_label": market_lead_label,
                "market_lead_score": _float_or_none(event.get("market_lead_score")),
                "market_lead_upside": int(event.get("market_lead_upside") or 0),
                "market_lead_downside": int(event.get("market_lead_downside") or 0),
                "market_lead_aligned": int(
                    direction in {"UPSIDE", "DOWNSIDE"} and market_lead_direction == direction
                ),
                "market_regime": str(event.get("market_regime") or "").strip(),
                "market_playbook": str(event.get("market_playbook") or "").strip(),
                "market_no_trade": int(bool(event.get("market_no_trade"))),
                "market_trade_gate": str(event.get("market_trade_gate") or "").strip(),
                "market_alert_keys": str(event.get("market_alert_keys") or "").strip(),
                "market_primary_alert": str(event.get("market_primary_alert") or "").strip(),
                "market_no_trade_reason": str(event.get("market_no_trade_reason") or "").strip(),
                "risk_tier": str(event.get("risk_tier") or "").strip(),
                "risk_unit_fraction": _float_or_none(event.get("risk_unit_fraction")),
                "sector_tag": str(event.get("sector_tag") or "").strip(),
                "market_sector_rotation": str(event.get("market_sector_rotation") or "").strip(),
                "market_catalyst_state": str(event.get("market_catalyst_state") or "").strip(),
                "market_catalyst_event": str(event.get("market_catalyst_event") or "").strip(),
                "market_catalyst_blocking": int(bool(event.get("market_catalyst_blocking"))),
                "market_catalyst_category": str(event.get("market_catalyst_category") or "").strip(),
                "market_catalyst_scope": str(event.get("market_catalyst_scope") or "").strip(),
                "market_catalyst_tag": str(event.get("market_catalyst_tag") or "").strip(),
                "market_catalyst_targeted": int(bool(event.get("market_catalyst_targeted"))),
                "market_catalyst_window": str(event.get("market_catalyst_window") or "").strip(),
                "market_flow_state": str(event.get("market_flow_state") or "").strip(),
                "market_flow_bias": str(event.get("market_flow_bias") or "").strip(),
                "adaptive_edge_label": str(event.get("adaptive_edge_label") or "").strip(),
                "adaptive_edge_score": _float_or_none(event.get("adaptive_edge_score")),
                "archive_guardrail_label": str(event.get("archive_guardrail_label") or "").strip(),
                "archive_guardrail_penalty": _float_or_none(event.get("archive_guardrail_penalty")),
                "archive_guardrail_note": str(event.get("archive_guardrail_note") or "").strip(),
                "price": _float_or_none(event.get("price")),
                "delta_pct": _float_or_none(event.get("delta_pct")),
                "entry_price": _float_or_none(event.get("entry_price")),
                "stop_loss": _float_or_none(event.get("stop_loss")),
                "target_price": _float_or_none(event.get("target_price")),
                "rr_ratio": _float_or_none(event.get("rr_ratio")),
                "has_plan": int(
                    _float_or_none(event.get("entry_price")) is not None
                    and _float_or_none(event.get("stop_loss")) is not None
                    and _float_or_none(event.get("target_price")) is not None
                ),
                "updated_at": now_iso,
                "created_at": now_iso,
            }
            conn.execute(
                """
                INSERT INTO signal_events (
                    signal_key, source, symbol, timeframe, event_time, session_bucket, horizon_bars, direction,
                    setup_confirm, action_reason, lead_label, lead_direction, lead_active,
                    confidence, ai_ensemble, ai_direction, ai_confidence, ai_aligned,
                    market_lead_label, market_lead_score, market_lead_upside, market_lead_downside,
                    market_lead_aligned, market_regime, market_playbook, market_no_trade,
                    market_trade_gate, market_alert_keys, market_primary_alert, market_no_trade_reason,
                    risk_tier, risk_unit_fraction,
                    sector_tag, market_sector_rotation,
                    market_catalyst_state, market_catalyst_event, market_catalyst_blocking,
                    market_catalyst_category, market_catalyst_scope, market_catalyst_tag, market_catalyst_targeted, market_catalyst_window,
                    market_flow_state, market_flow_bias,
                    adaptive_edge_label, adaptive_edge_score, archive_guardrail_label, archive_guardrail_penalty, archive_guardrail_note,
                    price, delta_pct, entry_price, stop_loss, target_price,
                    rr_ratio, has_plan, created_at, updated_at
                ) VALUES (
                    :signal_key, :source, :symbol, :timeframe, :event_time, :session_bucket, :horizon_bars, :direction,
                    :setup_confirm, :action_reason, :lead_label, :lead_direction, :lead_active,
                    :confidence, :ai_ensemble, :ai_direction, :ai_confidence, :ai_aligned,
                    :market_lead_label, :market_lead_score, :market_lead_upside, :market_lead_downside,
                    :market_lead_aligned, :market_regime, :market_playbook, :market_no_trade,
                    :market_trade_gate, :market_alert_keys, :market_primary_alert, :market_no_trade_reason,
                    :risk_tier, :risk_unit_fraction,
                    :sector_tag, :market_sector_rotation,
                    :market_catalyst_state, :market_catalyst_event, :market_catalyst_blocking,
                    :market_catalyst_category, :market_catalyst_scope, :market_catalyst_tag, :market_catalyst_targeted, :market_catalyst_window,
                    :market_flow_state, :market_flow_bias,
                    :adaptive_edge_label, :adaptive_edge_score, :archive_guardrail_label, :archive_guardrail_penalty, :archive_guardrail_note,
                    :price, :delta_pct, :entry_price, :stop_loss, :target_price,
                    :rr_ratio, :has_plan, :created_at, :updated_at
                )
                ON CONFLICT(signal_key) DO UPDATE SET
                    session_bucket=excluded.session_bucket,
                    setup_confirm=excluded.setup_confirm,
                    action_reason=excluded.action_reason,
                    lead_label=excluded.lead_label,
                    lead_direction=excluded.lead_direction,
                    lead_active=excluded.lead_active,
                    confidence=excluded.confidence,
                    ai_ensemble=excluded.ai_ensemble,
                    ai_direction=excluded.ai_direction,
                    ai_confidence=excluded.ai_confidence,
                    ai_aligned=excluded.ai_aligned,
                    market_lead_label=excluded.market_lead_label,
                    market_lead_score=excluded.market_lead_score,
                    market_lead_upside=excluded.market_lead_upside,
                    market_lead_downside=excluded.market_lead_downside,
                    market_lead_aligned=excluded.market_lead_aligned,
                    market_regime=excluded.market_regime,
                    market_playbook=excluded.market_playbook,
                    market_no_trade=excluded.market_no_trade,
                    market_trade_gate=excluded.market_trade_gate,
                    market_alert_keys=excluded.market_alert_keys,
                    market_primary_alert=excluded.market_primary_alert,
                    market_no_trade_reason=excluded.market_no_trade_reason,
                    risk_tier=excluded.risk_tier,
                    risk_unit_fraction=excluded.risk_unit_fraction,
                    sector_tag=excluded.sector_tag,
                    market_sector_rotation=excluded.market_sector_rotation,
                    market_catalyst_state=excluded.market_catalyst_state,
                    market_catalyst_event=excluded.market_catalyst_event,
                    market_catalyst_blocking=excluded.market_catalyst_blocking,
                    market_catalyst_category=excluded.market_catalyst_category,
                    market_catalyst_scope=excluded.market_catalyst_scope,
                    market_catalyst_tag=excluded.market_catalyst_tag,
                    market_catalyst_targeted=excluded.market_catalyst_targeted,
                    market_catalyst_window=excluded.market_catalyst_window,
                    market_flow_state=excluded.market_flow_state,
                    market_flow_bias=excluded.market_flow_bias,
                    adaptive_edge_label=excluded.adaptive_edge_label,
                    adaptive_edge_score=excluded.adaptive_edge_score,
                    archive_guardrail_label=excluded.archive_guardrail_label,
                    archive_guardrail_penalty=excluded.archive_guardrail_penalty,
                    archive_guardrail_note=excluded.archive_guardrail_note,
                    price=excluded.price,
                    delta_pct=excluded.delta_pct,
                    entry_price=excluded.entry_price,
                    stop_loss=excluded.stop_loss,
                    target_price=excluded.target_price,
                    rr_ratio=excluded.rr_ratio,
                    has_plan=excluded.has_plan,
                    updated_at=excluded.updated_at
                """,
                payload,
            )
            written += 1
        conn.commit()
    if written > 0:
        _sync_tracker_mirror(path)
    return written


def _resolve_event_from_frame(row: Mapping[str, object], df_ohlcv: pd.DataFrame) -> dict[str, object] | None:
    if df_ohlcv is None or df_ohlcv.empty or "close" not in df_ohlcv.columns:
        return None
    df_eval = df_ohlcv.reset_index(drop=True).copy()
    ts = pd.to_datetime(df_eval.get("timestamp"), utc=True, errors="coerce")
    if ts.isna().all():
        return None
    event_time = pd.to_datetime(row["event_time"], utc=True, errors="coerce")
    if pd.isna(event_time):
        return None
    match = ts[ts == event_time]
    if not match.empty:
        event_pos = int(match.index[-1])
    else:
        prior = ts[ts <= event_time]
        if prior.empty:
            return None
        event_pos = int(prior.index[-1])

    horizon_bars = max(1, int(row["horizon_bars"]))
    future_end = event_pos + horizon_bars
    if future_end >= len(df_eval):
        return None

    event_price = _float_or_none(row["price"])
    if event_price is None or event_price <= 0:
        try:
            event_price = float(df_eval["close"].iloc[event_pos])
        except Exception:
            return None

    high_col = "high" if "high" in df_eval.columns else "close"
    low_col = "low" if "low" in df_eval.columns else "close"
    future = df_eval.iloc[event_pos + 1 : future_end + 1].copy()
    if future.empty:
        return None
    future_close = pd.to_numeric(future["close"], errors="coerce").dropna()
    future_high = pd.to_numeric(future[high_col], errors="coerce").dropna()
    future_low = pd.to_numeric(future[low_col], errors="coerce").dropna()
    if future_close.empty:
        return None

    direction = _direction_key(row["direction"])
    end_price = float(future_close.iloc[-1])
    raw_return_pct = ((end_price / event_price) - 1.0) * 100.0
    max_high = float(future_high.max()) if not future_high.empty else end_price
    min_low = float(future_low.min()) if not future_low.empty else end_price
    max_up_pct = ((max_high / event_price) - 1.0) * 100.0
    max_down_pct = ((min_low / event_price) - 1.0) * 100.0

    if direction == "UPSIDE":
        directional_return_pct = raw_return_pct
        favorable_excursion_pct = max(0.0, max_up_pct)
        adverse_excursion_pct = max(0.0, -max_down_pct)
    elif direction == "DOWNSIDE":
        directional_return_pct = -raw_return_pct
        favorable_excursion_pct = max(0.0, -max_down_pct)
        adverse_excursion_pct = max(0.0, max_up_pct)
    else:
        directional_return_pct = raw_return_pct
        favorable_excursion_pct = max(0.0, abs(max_up_pct))
        adverse_excursion_pct = max(0.0, abs(max_down_pct))

    plan_outcome = ""
    entry_price = _float_or_none(row["entry_price"])
    stop_loss = _float_or_none(row["stop_loss"])
    target_price = _float_or_none(row["target_price"])
    if entry_price is not None and stop_loss is not None and target_price is not None and direction in {"UPSIDE", "DOWNSIDE"}:
        if direction == "UPSIDE":
            hit_tp = bool((pd.to_numeric(future[high_col], errors="coerce") >= target_price).fillna(False).any())
            hit_sl = bool((pd.to_numeric(future[low_col], errors="coerce") <= stop_loss).fillna(False).any())
        else:
            hit_tp = bool((pd.to_numeric(future[low_col], errors="coerce") <= target_price).fillna(False).any())
            hit_sl = bool((pd.to_numeric(future[high_col], errors="coerce") >= stop_loss).fillna(False).any())
        if hit_tp and hit_sl:
            plan_outcome = "BOTH"
        elif hit_tp:
            plan_outcome = "TP"
        elif hit_sl:
            plan_outcome = "SL"
        else:
            plan_outcome = "TIMEOUT"

    return {
        "status": _RESOLVED_STATUS,
        "plan_outcome": plan_outcome or None,
        "directional_return_pct": directional_return_pct,
        "favorable_excursion_pct": favorable_excursion_pct,
        "adverse_excursion_pct": adverse_excursion_pct,
        "resolved_at": _utc_iso(future.get("timestamp", pd.Series(dtype=object)).iloc[-1] if "timestamp" in future.columns else None),
        "updated_at": _utc_iso(),
    }


def resolve_open_signal_events_for_frame(
    *,
    symbol: str,
    timeframe: str,
    df_ohlcv: pd.DataFrame,
    source: str = "Market",
    db_path: str | None = None,
) -> int:
    path = init_signal_tracker_db(db_path)
    resolved = 0
    symbol_key = str(symbol or "").strip().upper()
    timeframe_key = str(timeframe or "").strip().lower()
    with _connect(path) as conn:
        rows = conn.execute(
            """
            SELECT * FROM signal_events
            WHERE status = ? AND source = ? AND symbol = ? AND timeframe = ?
            ORDER BY event_time ASC
            """,
            (_OPEN_STATUS, str(source or "Market"), symbol_key, timeframe_key),
        ).fetchall()
        for row in rows:
            outcome = _resolve_event_from_frame(row, df_ohlcv)
            if not outcome:
                continue
            conn.execute(
                """
                UPDATE signal_events
                SET status = :status,
                    plan_outcome = :plan_outcome,
                    directional_return_pct = :directional_return_pct,
                    favorable_excursion_pct = :favorable_excursion_pct,
                    adverse_excursion_pct = :adverse_excursion_pct,
                    resolved_at = :resolved_at,
                    updated_at = :updated_at
                WHERE signal_key = :signal_key
                """,
                {"signal_key": row["signal_key"], **outcome},
            )
            resolved += 1
        conn.commit()
    if resolved > 0:
        _sync_tracker_mirror(path)
    return resolved


def resolve_open_signal_events_via_fetch(
    *,
    fetch_ohlcv,
    source: str = "Market",
    db_path: str | None = None,
    limit_pairs: int = 12,
    candle_limit: int = 260,
) -> int:
    path = init_signal_tracker_db(db_path)
    total_resolved = 0
    with _connect(path) as conn:
        pairs = conn.execute(
            """
            SELECT symbol, timeframe, MAX(event_time) AS latest_event
            FROM signal_events
            WHERE status = ? AND source = ?
            GROUP BY symbol, timeframe
            ORDER BY latest_event DESC
            LIMIT ?
            """,
            (_OPEN_STATUS, str(source or "Market"), int(limit_pairs)),
        ).fetchall()
    for row in pairs:
        symbol = str(row["symbol"])
        timeframe = str(row["timeframe"])
        try:
            df = fetch_ohlcv(symbol, timeframe, limit=int(candle_limit))
        except Exception:
            continue
        if df is None or len(df) <= 3:
            continue
        df_eval = df.iloc[:-1].copy() if len(df) > 1 else df
        total_resolved += resolve_open_signal_events_for_frame(
            symbol=symbol,
            timeframe=timeframe,
            df_ohlcv=df_eval,
            source=source,
            db_path=path,
        )
    return total_resolved


def log_market_alerts(
    alerts: Sequence[Mapping[str, object]],
    *,
    source: str = "Market",
    db_path: str | None = None,
) -> int:
    path = init_signal_tracker_db(db_path)
    source_key = str(source or "Market").strip() or "Market"
    run_now = pd.Timestamp.utcnow().strftime("%Y-%m-%dT%H:%M:%S.%fZ")
    written = 0
    with _connect(path) as conn:
        for alert in list(alerts or []):
            alert_key = str(alert.get("alert_key") or "").strip().upper()
            state_signature = str(alert.get("state_signature") or "").strip()
            title = str(alert.get("title") or "").strip()
            note = str(alert.get("note") or "").strip()
            severity = str(alert.get("severity") or "INFO").strip().upper() or "INFO"
            if not alert_key or not state_signature or not title:
                continue
            conn.execute(
                """
                INSERT INTO market_alerts (
                    source, alert_key, state_signature, severity, title, note,
                    active, first_seen_at, last_seen_at, times_seen
                ) VALUES (
                    :source, :alert_key, :state_signature, :severity, :title, :note,
                    1, :first_seen_at, :last_seen_at, 1
                )
                ON CONFLICT(source, alert_key, state_signature) DO UPDATE SET
                    severity=excluded.severity,
                    title=excluded.title,
                    note=excluded.note,
                    active=1,
                    last_seen_at=excluded.last_seen_at,
                    times_seen=market_alerts.times_seen + 1
                """,
                {
                    "source": source_key,
                    "alert_key": alert_key,
                    "state_signature": state_signature,
                    "severity": severity,
                    "title": title,
                    "note": note,
                    "first_seen_at": run_now,
                    "last_seen_at": run_now,
                },
            )
            written += 1
        conn.execute(
            """
            UPDATE market_alerts
            SET active = 0
            WHERE source = ? AND last_seen_at <> ?
            """,
            (source_key, run_now),
        )
        conn.commit()
    if alerts:
        _sync_tracker_mirror(path)
    return written


def fetch_market_alerts_df(
    *,
    limit: int = 100,
    active_only: bool = False,
    source: str | None = None,
    db_path: str | None = None,
) -> pd.DataFrame:
    path = init_signal_tracker_db(db_path)
    query = "SELECT * FROM market_alerts WHERE 1=1"
    params: list[object] = []
    if source:
        query += " AND source = ?"
        params.append(str(source))
    if active_only:
        query += " AND active = 1"
    query += " ORDER BY active DESC, last_seen_at DESC LIMIT ?"
    params.append(int(limit))
    with _connect(path) as conn:
        return pd.read_sql_query(query, conn, params=params)


def fetch_signal_events_df(
    *,
    limit: int = 250,
    status: str | None = None,
    source: str | None = None,
    db_path: str | None = None,
) -> pd.DataFrame:
    path = init_signal_tracker_db(db_path)
    query = "SELECT * FROM signal_events WHERE 1=1"
    params: list[object] = []
    if status:
        query += " AND status = ?"
        params.append(str(status).strip().upper())
    if source:
        query += " AND source = ?"
        params.append(str(source))
    query += " ORDER BY event_time DESC LIMIT ?"
    params.append(int(limit))
    with _connect(path) as conn:
        return pd.read_sql_query(query, conn, params=params)


def _dominant_text_value(series: pd.Series | None, *, default: str) -> str:
    if not isinstance(series, pd.Series):
        return str(default)
    cleaned = series.fillna("").astype(str).str.strip()
    cleaned = cleaned[cleaned.ne("")]
    if cleaned.empty:
        return str(default)
    counts = cleaned.value_counts(dropna=False)
    return str(counts.index[0]).strip() or str(default)


def build_recent_market_context_snapshot(
    df_events: pd.DataFrame,
    *,
    lookback_hours: float = 12.0,
    max_rows: int = 120,
) -> dict[str, str]:
    snapshot = {
        "Market Lead": "No Clear Lead",
        "Market Regime": "Unknown",
        "Playbook": "Unknown",
        "Trade Gate": "Unknown",
        "Sector Rotation": "Unknown",
        "Catalyst State": "Unknown",
        "Catalyst Window": "Unknown",
        "Catalyst Scope": "Unknown",
        "Catalyst Targeting": "Unknown",
        "Flow Proxy": "Unknown",
        "Context Summary": "",
        "Context Note": "",
    }
    if df_events is None or df_events.empty:
        return snapshot

    d = df_events.copy()
    if "source" in d.columns:
        source_series = d["source"].fillna("").astype(str).str.strip()
        d = d[source_series.eq("Market")].copy()
    if d.empty:
        return snapshot

    event_ts = pd.to_datetime(d.get("event_time", pd.Series(index=d.index, dtype=object)), utc=True, errors="coerce")
    d["_event_ts"] = event_ts
    recent = d[d["_event_ts"].notna()].copy()
    if not recent.empty:
        latest_ts = recent["_event_ts"].max()
        cutoff = latest_ts - pd.Timedelta(hours=max(1.0, float(lookback_hours or 0.0)))
        recent = recent[recent["_event_ts"] >= cutoff].copy()
    if recent.empty:
        recent = d.copy()
    recent = recent.sort_values(by="_event_ts", ascending=False, na_position="last").head(int(max_rows))

    catalyst_targeting = (
        recent.get("market_catalyst_targeted", pd.Series(index=recent.index, dtype=int))
        .fillna(0)
        .astype(int)
        .map({1: "Targeted", 0: "Market-Wide"})
    )

    snapshot.update(
        {
            "Market Lead": _dominant_text_value(recent.get("market_lead_label"), default="No Clear Lead"),
            "Market Regime": _dominant_text_value(recent.get("market_regime"), default="Unknown"),
            "Playbook": _dominant_text_value(recent.get("market_playbook"), default="Unknown"),
            "Trade Gate": _dominant_text_value(recent.get("market_trade_gate"), default="Unknown"),
            "Sector Rotation": _dominant_text_value(recent.get("market_sector_rotation"), default="Unknown"),
            "Catalyst State": _dominant_text_value(recent.get("market_catalyst_state"), default="Unknown"),
            "Catalyst Window": _dominant_text_value(recent.get("market_catalyst_window"), default="Unknown"),
            "Catalyst Scope": _dominant_text_value(recent.get("market_catalyst_scope"), default="Unknown"),
            "Catalyst Targeting": _dominant_text_value(catalyst_targeting, default="Unknown"),
            "Flow Proxy": _dominant_text_value(recent.get("market_flow_state"), default="Unknown"),
        }
    )

    summary_parts: list[str] = []
    for value in (
        snapshot["Market Regime"],
        snapshot["Playbook"],
        snapshot["Trade Gate"],
        snapshot["Catalyst Window"],
    ):
        if value not in {"", "Unknown", "No Clear Lead"}:
            summary_parts.append(value)
    snapshot["Context Summary"] = " • ".join(summary_parts[:4])

    note_parts: list[str] = []
    for value in (
        snapshot["Market Regime"],
        snapshot["Playbook"],
        snapshot["Trade Gate"],
        snapshot["Catalyst Window"],
        snapshot["Sector Rotation"],
        snapshot["Flow Proxy"],
    ):
        if value not in {"", "Unknown", "No Clear Lead"}:
            note_parts.append(value)
    if note_parts:
        snapshot["Context Note"] = f"Recent market archive: {' | '.join(note_parts[:6])}."

    return snapshot


def build_recent_symbol_market_signal_snapshot(
    df_events: pd.DataFrame,
    *,
    symbol: str,
    timeframe: str | None = None,
) -> dict[str, str]:
    snapshot = {
        "Lead": "No LEAD",
        "Lead Label": "",
        "Timeframe": "",
        "Setup Confirm": "Unknown",
        "Risk Tier": "Unknown",
        "Direction": "Unknown",
        "Adaptive Edge": "Unknown",
        "Signal Note": "",
    }
    normalized_symbol = str(symbol or "").strip().upper()
    if not normalized_symbol or df_events is None or df_events.empty:
        return snapshot

    d = df_events.copy()
    if "source" in d.columns:
        source_series = d["source"].fillna("").astype(str).str.strip()
        d = d[source_series.eq("Market")].copy()
    if "symbol" in d.columns:
        symbol_series = d["symbol"].fillna("").astype(str).str.strip().str.upper()
        d = d[symbol_series.eq(normalized_symbol)].copy()
    if d.empty:
        return snapshot

    timeframe_key = str(timeframe or "").strip().lower()
    if timeframe_key and "timeframe" in d.columns:
        tf_series = d["timeframe"].fillna("").astype(str).str.strip().str.lower()
        if bool(tf_series.eq(timeframe_key).any()):
            d = d[tf_series.eq(timeframe_key)].copy()

    event_ts = pd.to_datetime(d.get("event_time", pd.Series(index=d.index, dtype=object)), utc=True, errors="coerce")
    d["_event_ts"] = event_ts
    d = d.sort_values(by="_event_ts", ascending=False, na_position="last")
    if d.empty:
        return snapshot
    row = d.iloc[0]

    lead_label = str(row.get("lead_label") or "").strip()
    lead_active_raw = row.get("lead_active")
    try:
        lead_active = bool(int(float(lead_active_raw))) if pd.notna(lead_active_raw) else bool(lead_label)
    except Exception:
        lead_active = bool(lead_label)
    setup_confirm = str(row.get("setup_confirm") or "").strip() or "Unknown"
    risk_tier = str(row.get("risk_tier") or "").strip() or "Unknown"
    direction = str(row.get("direction") or "").strip() or "Unknown"
    adaptive_edge = str(row.get("adaptive_edge_label") or "").strip() or "Unknown"
    signal_timeframe = str(row.get("timeframe") or "").strip()

    note_parts: list[str] = []
    if signal_timeframe:
        note_parts.append(signal_timeframe.upper())
    if lead_label:
        note_parts.append(lead_label)
    if setup_confirm not in {"", "Unknown"}:
        note_parts.append(setup_confirm)
    if risk_tier not in {"", "Unknown"}:
        note_parts.append(risk_tier)

    snapshot.update(
        {
            "Lead": "LEAD" if lead_active else "No LEAD",
            "Lead Label": lead_label,
            "Timeframe": signal_timeframe,
            "Setup Confirm": setup_confirm,
            "Risk Tier": risk_tier,
            "Direction": direction,
            "Adaptive Edge": adaptive_edge,
            "Signal Note": (
                f"Recent Market scanner read: {' | '.join(note_parts)}."
                if note_parts
                else ""
            ),
        }
    )
    return snapshot


def save_signal_trade_overlay(
    signal_key: str,
    *,
    trade_decision: str | None,
    trade_note: str | None = None,
    db_path: str | None = None,
) -> bool:
    key = str(signal_key or "").strip()
    if not key:
        return False
    decision = str(trade_decision or "").strip()
    note = str(trade_note or "").strip()
    marked_at = _utc_iso() if decision else None
    path = init_signal_tracker_db(db_path)
    with _connect(path) as conn:
        cur = conn.execute(
            """
            UPDATE signal_events
            SET trade_decision = ?,
                trade_note = ?,
                trade_marked_at = ?,
                updated_at = ?
            WHERE signal_key = ?
            """,
            (
                decision or None,
                note or None,
                marked_at,
                _utc_iso(),
                key,
            ),
        )
        conn.commit()
        updated = int(cur.rowcount) > 0
    if updated:
        _sync_tracker_mirror(path)
    return updated


def save_signal_trade_journal(
    signal_key: str,
    *,
    actual_trade_side: str | None = None,
    actual_entry_price: object = None,
    actual_entry_at: object = None,
    actual_exit_price: object = None,
    actual_exit_at: object = None,
    actual_exit_reason: str | None = None,
    db_path: str | None = None,
) -> bool:
    key = str(signal_key or "").strip()
    if not key:
        return False

    side = _trade_side_key(actual_trade_side)
    entry_price = _float_or_none(actual_entry_price)
    exit_price = _float_or_none(actual_exit_price)
    exit_reason = str(actual_exit_reason or "").strip()

    journal_is_blank = (
        not side
        and entry_price is None
        and exit_price is None
        and not str(actual_entry_at or "").strip()
        and not str(actual_exit_at or "").strip()
        and not exit_reason
    )
    if journal_is_blank:
        path = init_signal_tracker_db(db_path)
        with _connect(path) as conn:
            cur = conn.execute(
                """
                UPDATE signal_events
                SET actual_trade_side = NULL,
                    actual_entry_price = NULL,
                    actual_entry_at = NULL,
                    actual_exit_price = NULL,
                    actual_exit_at = NULL,
                    actual_exit_reason = NULL,
                    actual_pnl_pct = NULL,
                    actual_trade_status = NULL,
                    updated_at = ?
                WHERE signal_key = ?
                """,
                (_utc_iso(), key),
            )
            conn.commit()
            updated = int(cur.rowcount) > 0
        if updated:
            _sync_tracker_mirror(path)
        return updated

    if not side or entry_price is None or entry_price <= 0:
        return False
    if exit_price is not None and exit_price <= 0:
        return False
    if exit_price is None:
        exit_reason = ""
    elif exit_reason == "":
        exit_reason = "Manual Exit"

    entry_at_iso = _utc_iso(actual_entry_at) if str(actual_entry_at or "").strip() else _utc_iso()
    exit_at_iso = None
    if exit_price is not None:
        exit_at_iso = _utc_iso(actual_exit_at) if str(actual_exit_at or "").strip() else _utc_iso()

    actual_trade_status = "CLOSED" if exit_price is not None else "OPEN"
    actual_pnl_pct = None
    if exit_price is not None:
        if side == "LONG":
            actual_pnl_pct = ((exit_price / entry_price) - 1.0) * 100.0
        else:
            actual_pnl_pct = ((entry_price - exit_price) / entry_price) * 100.0

    path = init_signal_tracker_db(db_path)
    with _connect(path) as conn:
        cur = conn.execute(
            """
            UPDATE signal_events
            SET actual_trade_side = ?,
                actual_entry_price = ?,
                actual_entry_at = ?,
                actual_exit_price = ?,
                actual_exit_at = ?,
                actual_exit_reason = ?,
                actual_pnl_pct = ?,
                actual_trade_status = ?,
                trade_decision = CASE
                    WHEN ? THEN 'Taken'
                    ELSE trade_decision
                END,
                trade_marked_at = CASE
                    WHEN ? AND trade_marked_at IS NULL THEN ?
                    ELSE trade_marked_at
                END,
                updated_at = ?
            WHERE signal_key = ?
            """,
            (
                side,
                entry_price,
                entry_at_iso,
                exit_price,
                exit_at_iso,
                exit_reason or None,
                actual_pnl_pct,
                actual_trade_status,
                1,
                1,
                _utc_iso(),
                _utc_iso(),
                key,
            ),
        )
        conn.commit()
        updated = int(cur.rowcount) > 0
    if updated:
        _sync_tracker_mirror(path)
    return updated


def build_signal_review_snapshot(df_events: pd.DataFrame) -> dict[str, float]:
    if df_events is None or df_events.empty:
        return {
            "total": 0.0,
            "resolved": 0.0,
            "open": 0.0,
            "follow_through_rate": 0.0,
            "planned_tp_rate": 0.0,
            "avg_dir_return": 0.0,
            "avg_favorable_excursion": 0.0,
            "avg_adverse_excursion": 0.0,
            "taken": 0.0,
            "actual_open": 0.0,
            "actual_closed": 0.0,
            "actual_win_rate": 0.0,
            "avg_actual_pnl": 0.0,
        }
    d = df_events.copy()
    d["directional_return_pct"] = pd.to_numeric(d.get("directional_return_pct"), errors="coerce")
    d["favorable_excursion_pct"] = pd.to_numeric(d.get("favorable_excursion_pct"), errors="coerce")
    d["adverse_excursion_pct"] = pd.to_numeric(d.get("adverse_excursion_pct"), errors="coerce")
    d["actual_pnl_pct"] = pd.to_numeric(d.get("actual_pnl_pct"), errors="coerce")
    resolved = d[d["status"].astype(str).str.upper() == _RESOLVED_STATUS]
    planned = resolved[resolved.get("has_plan", pd.Series(dtype=int)).fillna(0).astype(int) == 1]
    taken = d[d.get("trade_decision", pd.Series(dtype=object)).fillna("").astype(str).str.upper() == "TAKEN"]
    actual_open = d[d.get("actual_trade_status", pd.Series(dtype=object)).fillna("").astype(str).str.upper() == "OPEN"]
    actual_closed = d[d.get("actual_trade_status", pd.Series(dtype=object)).fillna("").astype(str).str.upper() == "CLOSED"]
    total = float(len(d))
    resolved_n = float(len(resolved))
    open_n = float(len(d[d["status"].astype(str).str.upper() == _OPEN_STATUS]))
    follow_through_rate = float((resolved["directional_return_pct"] > 0).mean() * 100.0) if len(resolved) else 0.0
    planned_tp_rate = float((planned["plan_outcome"].astype(str).str.upper() == "TP").mean() * 100.0) if len(planned) else 0.0
    return {
        "total": total,
        "resolved": resolved_n,
        "open": open_n,
        "follow_through_rate": follow_through_rate,
        "planned_tp_rate": planned_tp_rate,
        "avg_dir_return": float(resolved["directional_return_pct"].mean()) if len(resolved) else 0.0,
        "avg_favorable_excursion": float(resolved["favorable_excursion_pct"].mean()) if len(resolved) else 0.0,
        "avg_adverse_excursion": float(resolved["adverse_excursion_pct"].mean()) if len(resolved) else 0.0,
        "taken": float(len(taken)),
        "actual_open": float(len(actual_open)),
        "actual_closed": float(len(actual_closed)),
        "actual_win_rate": float((actual_closed["actual_pnl_pct"] > 0).mean() * 100.0) if len(actual_closed) else 0.0,
        "avg_actual_pnl": float(actual_closed["actual_pnl_pct"].mean()) if len(actual_closed) else 0.0,
    }


def build_execution_overlay_snapshot(df_events: pd.DataFrame) -> dict[str, float]:
    if df_events is None or df_events.empty:
        return {
            "taken": 0.0,
            "taken_resolved": 0.0,
            "taken_follow_through_rate": 0.0,
            "actual_closed": 0.0,
            "actual_win_rate": 0.0,
            "avg_actual_pnl": 0.0,
            "avg_signal_dir_return_on_taken": 0.0,
            "execution_gap_pct": 0.0,
            "skipped_winners": 0.0,
            "skipped_resolved": 0.0,
            "skipped_winner_rate": 0.0,
        }
    d = df_events.copy()
    d["directional_return_pct"] = pd.to_numeric(d.get("directional_return_pct"), errors="coerce")
    d["actual_pnl_pct"] = pd.to_numeric(d.get("actual_pnl_pct"), errors="coerce")
    d["trade_decision"] = d.get("trade_decision", pd.Series(dtype=object)).fillna("").astype(str).str.upper()
    d["status"] = d.get("status", pd.Series(dtype=object)).fillna("").astype(str).str.upper()
    d["actual_trade_status"] = d.get("actual_trade_status", pd.Series(dtype=object)).fillna("").astype(str).str.upper()

    taken = d[d["trade_decision"] == "TAKEN"]
    taken_resolved = taken[taken["status"] == _RESOLVED_STATUS]
    actual_closed = taken[taken["actual_trade_status"] == "CLOSED"]
    skipped = d[d["trade_decision"] == "SKIPPED"]
    skipped_resolved = skipped[skipped["status"] == _RESOLVED_STATUS]

    taken_follow_through_rate = (
        float((taken_resolved["directional_return_pct"] > 0).mean() * 100.0) if len(taken_resolved) else 0.0
    )
    actual_win_rate = float((actual_closed["actual_pnl_pct"] > 0).mean() * 100.0) if len(actual_closed) else 0.0
    avg_actual_pnl = float(actual_closed["actual_pnl_pct"].mean()) if len(actual_closed) else 0.0
    avg_signal_dir_return_on_taken = (
        float(actual_closed["directional_return_pct"].mean()) if len(actual_closed) else 0.0
    )
    skipped_winners = float((skipped_resolved["directional_return_pct"] > 0).sum()) if len(skipped_resolved) else 0.0
    skipped_winner_rate = (
        float((skipped_resolved["directional_return_pct"] > 0).mean() * 100.0) if len(skipped_resolved) else 0.0
    )
    return {
        "taken": float(len(taken)),
        "taken_resolved": float(len(taken_resolved)),
        "taken_follow_through_rate": taken_follow_through_rate,
        "actual_closed": float(len(actual_closed)),
        "actual_win_rate": actual_win_rate,
        "avg_actual_pnl": avg_actual_pnl,
        "avg_signal_dir_return_on_taken": avg_signal_dir_return_on_taken,
        "execution_gap_pct": avg_actual_pnl - avg_signal_dir_return_on_taken,
        "skipped_winners": skipped_winners,
        "skipped_resolved": float(len(skipped_resolved)),
        "skipped_winner_rate": skipped_winner_rate,
    }


def _narrow_actual_trade_archive(
    d: pd.DataFrame,
    *,
    symbol: str = "",
    timeframe: str = "",
    direction: str = "",
    sector_tag: str = "",
    playbook: str = "",
    session_bucket: str = "",
    trade_gate: str = "",
    catalyst_window: str = "",
) -> tuple[pd.DataFrame, list[str], list[str]]:
    applied_filters: list[str] = []
    skipped_filters: list[str] = []

    def _apply_archive_filter(
        frame: pd.DataFrame,
        *,
        column: str,
        value: str,
        label: str,
        min_rows: int,
        normalizer=None,
    ) -> pd.DataFrame:
        selected = str(value or "").strip()
        if not selected or selected == "Unknown" or column not in frame.columns:
            return frame
        series = frame[column]
        if normalizer is None:
            normalized_selected = selected
            normalized_series = series.fillna("").astype(str).str.strip()
        else:
            normalized_selected = str(normalizer(selected) or "").strip()
            if not normalized_selected or normalized_selected == "NEUTRAL":
                return frame
            normalized_series = series.map(normalizer).fillna("").astype(str).str.strip()
        narrowed = frame[normalized_series.eq(normalized_selected)].copy()
        if len(narrowed) >= int(min_rows):
            applied_filters.append(label)
            return narrowed
        if len(narrowed) > 0:
            skipped_filters.append(label)
        return frame

    d = _apply_archive_filter(
        d,
        column="timeframe",
        value=timeframe,
        label=f"{str(timeframe or '').upper()} timeframe",
        min_rows=4,
        normalizer=lambda value: str(value or "").strip().lower(),
    )
    d = _apply_archive_filter(
        d,
        column="direction",
        value=direction,
        label=f"{_direction_key(direction).title()} direction",
        min_rows=4,
        normalizer=_direction_key,
    )
    d = _apply_archive_filter(
        d,
        column="symbol",
        value=symbol,
        label=f"{str(symbol or '').strip().upper()} symbol",
        min_rows=3,
        normalizer=lambda value: str(value or "").strip().upper(),
    )
    d = _apply_archive_filter(
        d,
        column="sector_tag",
        value=sector_tag,
        label=f"{str(sector_tag or '').strip()} sector",
        min_rows=3,
    )
    d = _apply_archive_filter(d, column="market_playbook", value=playbook, label="current playbook", min_rows=3)
    d = _apply_archive_filter(d, column="session_bucket", value=session_bucket, label="current session", min_rows=3)
    d = _apply_archive_filter(d, column="market_trade_gate", value=trade_gate, label="current gate", min_rows=3)
    d = _apply_archive_filter(
        d, column="market_catalyst_window", value=catalyst_window, label="current catalyst window", min_rows=3
    )
    return d, applied_filters, skipped_filters


def _archive_scope_note(applied_filters: list[str], skipped_filters: list[str]) -> str:
    scope_bits: list[str] = []
    if applied_filters:
        scope_bits.append(f"Using {' + '.join(applied_filters)} archive.")
    if skipped_filters:
        skipped_text = ", ".join(skipped_filters[:3])
        if len(skipped_filters) > 3:
            skipped_text = f"{skipped_text}, and other thinner slices"
        verb = "were" if len(skipped_filters) > 1 else "was"
        scope_bits.append(
            f"{skipped_text} {verb} too thin, so this falls back to the broader cluster."
        )
    return " ".join(scope_bits).strip()


def build_actual_trade_hold_profile(
    df_events: pd.DataFrame,
    *,
    symbol: str = "",
    timeframe: str = "",
    direction: str = "",
    sector_tag: str = "",
    playbook: str = "",
    session_bucket: str = "",
    trade_gate: str = "",
    catalyst_window: str = "",
) -> dict[str, object]:
    if df_events is None or df_events.empty:
        return {
            "label": "Archive Building",
            "note": "There is not enough closed trade history yet to judge hold style.",
            "sample_size": 0,
            "median_hold_hours": 0.0,
        }

    d = df_events.copy()
    d["actual_trade_status"] = d.get("actual_trade_status", pd.Series(dtype=object)).fillna("").astype(str).str.upper()
    d = d[d["actual_trade_status"] == "CLOSED"].copy()
    if d.empty:
        return {
            "label": "Archive Building",
            "note": "Closed trade history is still building for hold-style guidance.",
            "sample_size": 0,
            "median_hold_hours": 0.0,
        }

    d, applied_filters, skipped_filters = _narrow_actual_trade_archive(
        d,
        symbol=symbol,
        timeframe=timeframe,
        direction=direction,
        sector_tag=sector_tag,
        playbook=playbook,
        session_bucket=session_bucket,
        trade_gate=trade_gate,
        catalyst_window=catalyst_window,
    )

    d["actual_entry_at"] = pd.to_datetime(d.get("actual_entry_at"), utc=True, errors="coerce")
    d["actual_exit_at"] = pd.to_datetime(d.get("actual_exit_at"), utc=True, errors="coerce")
    d["actual_pnl_pct"] = pd.to_numeric(d.get("actual_pnl_pct"), errors="coerce")
    d = d.dropna(subset=["actual_entry_at", "actual_exit_at"]).copy()
    if d.empty:
        return {
            "label": "Archive Building",
            "note": "Closed trades exist, but hold-time timestamps are incomplete.",
            "sample_size": 0,
            "median_hold_hours": 0.0,
        }

    d["hold_hours"] = (d["actual_exit_at"] - d["actual_entry_at"]).dt.total_seconds() / 3600.0
    d = d[d["hold_hours"] >= 0.0].copy()
    if d.empty:
        return {
            "label": "Archive Building",
            "note": "Closed trades exist, but hold-time measurements are not usable yet.",
            "sample_size": 0,
            "median_hold_hours": 0.0,
        }

    winners = d[d["actual_pnl_pct"] > 0].copy()
    sample = winners if len(winners) >= 2 else d
    median_hold_hours = float(sample["hold_hours"].median()) if len(sample) else 0.0
    win_rate = float((d["actual_pnl_pct"] > 0).mean() * 100.0) if len(d) else 0.0
    sample_size = int(len(d))

    if sample_size < 3:
        label = "Archive Building"
        note = "Hold-style archive is still thin, so stay flexible rather than over-optimizing exits."
    elif median_hold_hours <= 6.0:
        label = "Quick Follow-Through"
        note = (
            f"This cluster usually pays quickly ({median_hold_hours:.1f}h median hold, "
            f"{win_rate:.0f}% win rate). If follow-through stalls, trim faster."
        )
    elif median_hold_hours >= 18.0:
        label = "Needs Room"
        note = (
            f"This cluster tends to reward patience ({median_hold_hours:.1f}h median hold, "
            f"{win_rate:.0f}% win rate). Avoid choking winners too early."
        )
    else:
        label = "Standard Hold"
        note = (
            f"This cluster is usually a medium-hold trade ({median_hold_hours:.1f}h median hold, "
            f"{win_rate:.0f}% win rate). Manage actively, but do not rush exits."
        )

    scope_note = _archive_scope_note(applied_filters, skipped_filters)
    if scope_note:
        note = f"{note} {scope_note}".strip()

    return {
        "label": label,
        "note": note,
        "sample_size": sample_size,
        "median_hold_hours": median_hold_hours,
        "applied_filters": list(applied_filters),
        "skipped_filters": list(skipped_filters),
    }


def build_actual_exit_quality_profile(
    df_events: pd.DataFrame,
    *,
    symbol: str = "",
    timeframe: str = "",
    direction: str = "",
    sector_tag: str = "",
    playbook: str = "",
    session_bucket: str = "",
    trade_gate: str = "",
    catalyst_window: str = "",
) -> dict[str, object]:
    if df_events is None or df_events.empty:
        return {
            "label": "Archive Building",
            "note": "There is not enough closed trade history yet to judge exit quality.",
            "sample_size": 0,
            "winner_manual_rate": 0.0,
            "loser_late_rate": 0.0,
            "loser_protected_rate": 0.0,
        }

    d = df_events.copy()
    d["actual_trade_status"] = d.get("actual_trade_status", pd.Series(dtype=object)).fillna("").astype(str).str.upper()
    d = d[d["actual_trade_status"] == "CLOSED"].copy()
    if d.empty:
        return {
            "label": "Archive Building",
            "note": "Closed trade history is still building for exit-discipline guidance.",
            "sample_size": 0,
            "winner_manual_rate": 0.0,
            "loser_late_rate": 0.0,
            "loser_protected_rate": 0.0,
        }

    d, applied_filters, skipped_filters = _narrow_actual_trade_archive(
        d,
        symbol=symbol,
        timeframe=timeframe,
        direction=direction,
        sector_tag=sector_tag,
        playbook=playbook,
        session_bucket=session_bucket,
        trade_gate=trade_gate,
        catalyst_window=catalyst_window,
    )
    d["actual_pnl_pct"] = pd.to_numeric(d.get("actual_pnl_pct"), errors="coerce")
    d["actual_exit_reason"] = d.get("actual_exit_reason", pd.Series(dtype=object)).fillna("").astype(str).str.strip().str.upper()
    d = d.dropna(subset=["actual_pnl_pct"]).copy()
    if d.empty:
        return {
            "label": "Archive Building",
            "note": "Closed trades exist, but realized PnL is incomplete for exit-quality guidance.",
            "sample_size": 0,
            "winner_manual_rate": 0.0,
            "loser_late_rate": 0.0,
            "loser_protected_rate": 0.0,
        }

    winners = d[d["actual_pnl_pct"] > 0].copy()
    losers = d[d["actual_pnl_pct"] <= 0].copy()
    manual_reasons = {"MANUAL EXIT", "TIME EXIT"}
    protected_reasons = {"STOP", "INVALIDATION"}

    winner_manual_rate = float(winners["actual_exit_reason"].isin(manual_reasons).mean() * 100.0) if len(winners) else 0.0
    winner_target_rate = float(winners["actual_exit_reason"].eq("TARGET").mean() * 100.0) if len(winners) else 0.0
    loser_late_rate = float(losers["actual_exit_reason"].isin(manual_reasons).mean() * 100.0) if len(losers) else 0.0
    loser_protected_rate = (
        float(losers["actual_exit_reason"].isin(protected_reasons).mean() * 100.0) if len(losers) else 0.0
    )
    sample_size = int(len(d))

    if sample_size < 3:
        label = "Archive Building"
        note = "Exit-discipline archive is still thin, so use this as a light coaching note rather than a hard rule."
    elif len(winners) >= 2 and winner_manual_rate >= 60.0:
        label = "Winner Cut Risk"
        note = (
            f"This cluster often banks winners manually ({winner_manual_rate:.0f}% of winners) instead of letting them fully play out. "
            "Be careful not to choke good trades too early."
        )
    elif len(losers) >= 2 and loser_late_rate >= 60.0:
        label = "Late Loss Risk"
        note = (
            f"This cluster often exits losers late ({loser_late_rate:.0f}% manual/time exits on losing trades). "
            "Cut faster when structure breaks."
        )
    elif len(winners) >= 2 and len(losers) >= 2 and winner_target_rate >= 50.0 and loser_protected_rate >= 50.0:
        label = "Healthy Exit Discipline"
        note = (
            f"Winners are reaching target often ({winner_target_rate:.0f}%) and losers are usually protected "
            f"with stop or invalidation exits ({loser_protected_rate:.0f}%)."
        )
    else:
        label = "Mixed Exit Discipline"
        note = (
            f"Winner exits are {winner_target_rate:.0f}% target / {winner_manual_rate:.0f}% manual, while loser protection is "
            f"{loser_protected_rate:.0f}% stop-or-invalidation."
        )

    scope_note = _archive_scope_note(applied_filters, skipped_filters)
    if scope_note:
        note = f"{note} {scope_note}".strip()

    return {
        "label": label,
        "note": note,
        "sample_size": sample_size,
        "winner_manual_rate": winner_manual_rate,
        "loser_late_rate": loser_late_rate,
        "loser_protected_rate": loser_protected_rate,
        "applied_filters": list(applied_filters),
        "skipped_filters": list(skipped_filters),
    }


def build_signal_cohort_summary(df_events: pd.DataFrame, group_field: str) -> pd.DataFrame:
    if df_events is None or df_events.empty or group_field not in df_events.columns:
        return pd.DataFrame()
    d = df_events.copy()
    d["directional_return_pct"] = pd.to_numeric(d.get("directional_return_pct"), errors="coerce")
    d["favorable_excursion_pct"] = pd.to_numeric(d.get("favorable_excursion_pct"), errors="coerce")
    d["adverse_excursion_pct"] = pd.to_numeric(d.get("adverse_excursion_pct"), errors="coerce")
    d["actual_pnl_pct"] = pd.to_numeric(d.get("actual_pnl_pct"), errors="coerce")
    d["is_resolved"] = (d["status"].astype(str).str.upper() == _RESOLVED_STATUS).astype(int)
    d["is_follow_through"] = ((d["is_resolved"] == 1) & (d["directional_return_pct"] > 0)).astype(int)
    d["is_tp"] = (d.get("plan_outcome", pd.Series(dtype=object)).astype(str).str.upper() == "TP").astype(int)
    d["is_sl"] = (d.get("plan_outcome", pd.Series(dtype=object)).astype(str).str.upper() == "SL").astype(int)
    d["is_taken"] = (d.get("trade_decision", pd.Series(dtype=object)).fillna("").astype(str).str.upper() == "TAKEN").astype(int)
    d["is_trade_closed"] = (
        d.get("actual_trade_status", pd.Series(dtype=object)).fillna("").astype(str).str.upper() == "CLOSED"
    ).astype(int)
    d["is_trade_win"] = ((d["is_trade_closed"] == 1) & (d["actual_pnl_pct"] > 0)).astype(int)
    grouped = (
        d.groupby(group_field, dropna=False)
        .agg(
            Signals=("symbol", "count"),
            Resolved=("is_resolved", "sum"),
            FollowThroughCount=("is_follow_through", "sum"),
            AvgDirReturnPct=("directional_return_pct", "mean"),
            AvgFavExcPct=("favorable_excursion_pct", "mean"),
            AvgAdvExcPct=("adverse_excursion_pct", "mean"),
            TpCount=("is_tp", "sum"),
            SlCount=("is_sl", "sum"),
            TakenCount=("is_taken", "sum"),
            ClosedTradeCount=("is_trade_closed", "sum"),
            TradeWinCount=("is_trade_win", "sum"),
            AvgActualPnlPct=("actual_pnl_pct", "mean"),
        )
        .reset_index()
    )
    grouped["FollowThroughPct"] = grouped.apply(
        lambda r: (float(r["FollowThroughCount"]) / float(r["Resolved"]) * 100.0) if float(r["Resolved"]) > 0 else 0.0,
        axis=1,
    )
    grouped["TpRatePct"] = grouped.apply(
        lambda r: (float(r["TpCount"]) / float(r["Resolved"]) * 100.0) if float(r["Resolved"]) > 0 else 0.0,
        axis=1,
    )
    grouped["SlRatePct"] = grouped.apply(
        lambda r: (float(r["SlCount"]) / float(r["Resolved"]) * 100.0) if float(r["Resolved"]) > 0 else 0.0,
        axis=1,
    )
    grouped["TakenPct"] = grouped.apply(
        lambda r: (float(r["TakenCount"]) / float(r["Signals"]) * 100.0) if float(r["Signals"]) > 0 else 0.0,
        axis=1,
    )
    grouped["ActualWinRatePct"] = grouped.apply(
        lambda r: (float(r["TradeWinCount"]) / float(r["ClosedTradeCount"]) * 100.0) if float(r["ClosedTradeCount"]) > 0 else 0.0,
        axis=1,
    )
    grouped = grouped.drop(columns=["FollowThroughCount", "TpCount", "SlCount", "TradeWinCount"])
    grouped[group_field] = grouped[group_field].replace("", "Unknown").fillna("Unknown")
    return grouped.sort_values(by=["Signals", "FollowThroughPct"], ascending=[False, False]).reset_index(drop=True)


def annotate_alert_footprint(df_events: pd.DataFrame) -> pd.DataFrame:
    if df_events is None or df_events.empty:
        return pd.DataFrame()
    d = df_events.copy()
    alert_keys_series: list[list[str]] = []
    primary_alerts: list[str] = []
    alert_footprints: list[str] = []
    for _, row in d.iterrows():
        keys = _infer_alert_keys_from_event_row(row.to_dict())
        displays = [_alert_key_display(key) for key in keys]
        alert_keys_series.append(displays)
        primary_alerts.append(displays[0] if displays else "No Alert Footprint")
        alert_footprints.append(" | ".join(displays) if displays else "No Alert Footprint")
    d["Primary Alert"] = primary_alerts
    d["Alert Footprint"] = alert_footprints
    d["__alert_keys_display"] = alert_keys_series
    return d


def build_alert_effectiveness_summary(df_events: pd.DataFrame, *, primary_only: bool = False) -> pd.DataFrame:
    annotated = annotate_alert_footprint(df_events)
    if annotated.empty:
        return pd.DataFrame()
    if primary_only:
        return build_signal_cohort_summary(annotated, "Primary Alert")

    exploded = annotated.copy()
    exploded["Alert Key"] = exploded["__alert_keys_display"]
    exploded = exploded.explode("Alert Key")
    exploded["Alert Key"] = exploded["Alert Key"].fillna("No Alert Footprint").replace("", "No Alert Footprint")
    return build_signal_cohort_summary(exploded, "Alert Key")
