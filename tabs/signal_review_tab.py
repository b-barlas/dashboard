from __future__ import annotations

from collections.abc import Mapping

import pandas as pd
from core.decision_version import current_decision_version
from core.session_utils import session_bucket_for_timestamp
from core.signal_tracker import prefer_current_decision_version_slice
from core.trading_copy import copy_text, playbook_display, playbook_key, trade_gate_display, trade_gate_key

from ui.ctx import get_ctx
from ui.signal_formatters import archived_execution_stance_label, setup_confirm_display
from ui.primitives import render_insight_card, render_kpi_grid, render_page_header


_MIN_SIGNAL_ARCHIVE_ROWS = 8
_MIN_EXECUTION_ARCHIVE_ROWS = 3

_FOLLOW_THROUGH_HORIZONS: dict[str, int] = {
    "5m": 12,
    "15m": 16,
    "30m": 16,
    "1h": 12,
    "2h": 12,
    "4h": 12,
    "1d": 10,
}

_HOLD_GUIDANCE_TIMEFRAME_ORDER = ("5m", "15m", "1h", "4h", "1d")


def _display_trade_direction(value: object) -> str:
    side = str(value or "").strip().upper()
    if side in {"LONG", "UPSIDE", "BUY"}:
        return "Upside"
    if side in {"SHORT", "DOWNSIDE", "SELL"}:
        return "Downside"
    return ""


def _annotate_actual_hold_style(df_events: pd.DataFrame) -> pd.DataFrame:
    if df_events is None or df_events.empty:
        return pd.DataFrame()
    d = df_events.copy()
    if "actual_trade_status" not in d.columns:
        return d
    d["actual_trade_status"] = d["actual_trade_status"].fillna("").astype(str).str.upper()
    d["actual_entry_at"] = pd.to_datetime(d.get("actual_entry_at"), utc=True, errors="coerce")
    d["actual_exit_at"] = pd.to_datetime(d.get("actual_exit_at"), utc=True, errors="coerce")
    d["Actual Hold Hours"] = (
        (d["actual_exit_at"] - d["actual_entry_at"]).dt.total_seconds() / 3600.0
    )
    d["Actual Hold Hours"] = pd.to_numeric(d["Actual Hold Hours"], errors="coerce")

    def _style_for_row(row: pd.Series) -> str:
        status = str(row.get("actual_trade_status") or "").strip().upper()
        hold_hours = row.get("Actual Hold Hours")
        if status != "CLOSED":
            return "Open / Unjournaled"
        if pd.isna(hold_hours):
            return "Unknown Hold"
        if float(hold_hours) <= 6.0:
            return "Quick Follow-Through"
        if float(hold_hours) >= 18.0:
            return "Needs Room"
        return "Standard Hold"

    d["Hold Style"] = d.apply(_style_for_row, axis=1)
    return d


def _annotate_actual_exit_quality(df_events: pd.DataFrame) -> pd.DataFrame:
    if df_events is None or df_events.empty:
        return pd.DataFrame()
    d = df_events.copy()
    if "actual_trade_status" not in d.columns:
        return d
    d["actual_trade_status"] = d["actual_trade_status"].fillna("").astype(str).str.upper()
    d["actual_pnl_pct"] = pd.to_numeric(d.get("actual_pnl_pct"), errors="coerce")
    d["actual_exit_reason"] = d.get("actual_exit_reason", pd.Series(dtype=object)).fillna("").astype(str).str.strip().str.upper()

    def _quality_for_row(row: pd.Series) -> str:
        status = str(row.get("actual_trade_status") or "").strip().upper()
        if status != "CLOSED":
            return "Open / Unjournaled"
        pnl = row.get("actual_pnl_pct")
        if pd.isna(pnl):
            return "Unknown Exit"
        reason = str(row.get("actual_exit_reason") or "").strip().upper()
        if float(pnl) > 0.0:
            if reason == "TARGET":
                return "Target Winner"
            if reason in {"MANUAL EXIT", "TIME EXIT"}:
                return "Manual Winner Exit"
            return "Winner Other Exit"
        if reason in {"STOP", "INVALIDATION"}:
            return "Protected Loss Exit"
        if reason in {"MANUAL EXIT", "TIME EXIT"}:
            return "Late Manual Loss"
        return "Loss Other Exit"

    d["Exit Quality"] = d.apply(_quality_for_row, axis=1)
    return d


def _normalize_symbol_filter(value: object) -> str:
    text = str(value or "").strip().upper()
    if not text:
        return ""
    for separator in ("/", "-", " "):
        if separator in text:
            text = text.split(separator, 1)[0].strip()
    return text


def _hold_scope_label(*, symbol_filter: str, timeframe_filter: str) -> str:
    symbol_label = str(symbol_filter or "").strip().upper()
    timeframe_label = str(timeframe_filter or "").strip()
    if symbol_label and timeframe_label and timeframe_label != "All":
        return f"{symbol_label} {timeframe_label.upper()}"
    if symbol_label:
        return symbol_label
    return "current coin scope"


def _refresh_scope_badge(*, symbol_filter: str, timeframe_filter: str, resolved_now: int) -> str:
    scope_label = "Market"
    symbol_label = str(symbol_filter or "").strip().upper()
    timeframe_label = str(timeframe_filter or "").strip()
    if symbol_label and timeframe_label and timeframe_label != "All":
        scope_label = f"{symbol_label} {timeframe_label.upper()}"
    elif symbol_label:
        scope_label = symbol_label
    elif timeframe_label and timeframe_label != "All":
        scope_label = timeframe_label.upper()
    if resolved_now > 0:
        return f"{scope_label} +{resolved_now} refreshed"
    return f"{scope_label} up to date"


def _follow_through_horizon_note(timeframe_filter: str) -> str:
    tf_key = str(timeframe_filter or "").strip().lower()
    if tf_key in _FOLLOW_THROUGH_HORIZONS:
        return (
            f"Follow-through here resolves on the archive horizon for {tf_key.upper()}: "
            f"{_FOLLOW_THROUGH_HORIZONS[tf_key]} bars after the signal."
        )
    return (
        "Follow-through uses the archive horizon after each signal: "
        "5m = 12 bars, 15m = 16 bars, 1h = 12 bars, 4h = 12 bars, 1d = 10 bars."
    )


def _review_scope_summary(
    *,
    status_filter: str,
    timeframe_filter: str,
    limit: int,
    rows_in_view: int,
    symbol_filter: str = "",
) -> str:
    scope_parts: list[str] = []
    if str(symbol_filter or "").strip():
        scope_parts.append(str(symbol_filter).strip().upper())
    else:
        scope_parts.append("Market")
    scope_parts.append("All TF" if str(timeframe_filter) == "All" else str(timeframe_filter).upper())
    scope_parts.append("All Status" if str(status_filter) == "All" else str(status_filter).title())
    return (
        f"<b>{' • '.join(scope_parts)}</b><br>"
        f"{int(rows_in_view)} of {int(limit)} rows shown<br>"
        "KPIs and deep dives use this view"
    )


def _learning_readiness_summary(*, mode: str, current_rows: int, total_rows: int) -> tuple[str, str]:
    if mode == "current_only":
        return (
            f"<b>Current-only learning active</b><br>{int(current_rows)} current resolved • {int(total_rows)} loaded",
            "positive",
        )
    if mode == "mixed_fallback":
        return (
            f"<b>Mixed fallback</b><br>{int(current_rows)} current resolved • {int(total_rows)} loaded",
            "warning",
        )
    if mode == "unversioned_fallback":
        return (
            "<b>Legacy fallback active</b><br>Current scanner history is not isolated yet",
            "warning",
        )
    if mode == "empty":
        return (
            "<b>No resolved history yet</b><br>Learning turns on after the first resolved archive rows",
            "neutral",
        )
    return (
        f"<b>Learning building</b><br>{int(current_rows)} current resolved • {int(total_rows)} loaded",
        "neutral",
    )


def _archive_health_summary(storage_snapshot) -> tuple[str, str]:
    headline = str(storage_snapshot.durability_label or "Archive OK").strip() or "Archive OK"
    recovery = str(storage_snapshot.recovery_status or "Unknown").strip() or "Unknown"
    body = f"<b>{headline}</b><br>Recovery: {recovery}"
    tone = str(storage_snapshot.durability_tone or "neutral")
    return body, tone


def _hold_window_note(snapshot: Mapping[str, object]) -> tuple[str, str]:
    resolved_signals = int(snapshot.get("resolved_signals") or 0)
    if resolved_signals <= 0:
        return (
            "No resolved signals are available in this view yet, so hold-window learning cannot start.",
            "neutral",
        )
    if not bool(snapshot.get("available")):
        return (
            (
                f"This view is still building hold-window intelligence. We have <b>{resolved_signals}</b> resolved signals, "
                "but need a deeper checkpoint sample before the suggestion becomes trustworthy."
            ),
            "neutral",
        )

    best_bar = int(snapshot.get("best_bar") or 0)
    best_label = str(snapshot.get("best_label") or "").strip() or "around 0 bars"
    best_style = str(snapshot.get("best_style") or "Standard Hold").strip()
    sample = int(snapshot.get("sample") or 0)
    avg_dir_return_pct = float(snapshot.get("avg_dir_return_pct") or 0.0)
    follow_through_pct = float(snapshot.get("follow_through_pct") or 0.0)
    fade_after_bar = int(snapshot.get("fade_after_bar") or 0)
    fade_text = (
        f"Edge starts to fade after roughly <b>{fade_after_bar}</b> bars."
        if fade_after_bar > 0
        else "Edge has not clearly faded inside the measured checkpoint ladder yet."
    )
    tone = "positive" if avg_dir_return_pct > 0.0 else "warning"
    lead_text = (
        f"<b>Best at: {best_bar} bars</b><br>"
        if best_bar > 0
        else f"<b>Suggested hold: {best_label}</b><br>"
    )
    return (
        (
            lead_text
            + (
            f"Style: <b>{best_style}</b>.<br>"
            f"Historical sweet spot in this view, based on <b>{sample}</b> resolved signals at that checkpoint. "
            f"Avg directional return: <b>{avg_dir_return_pct:+.2f}%</b>, follow-through: <b>{follow_through_pct:.1f}%</b>. "
            f"{fade_text}"
            )
        ),
        tone,
    )


def _hold_guidance_cell(snapshot: Mapping[str, object], *, direction_label: str | None = None) -> str:
    resolved_signals = int(snapshot.get("resolved_signals") or 0)
    if bool(snapshot.get("available")):
        best_bar = int(snapshot.get("best_bar") or 0)
        fade_after_bar = int(snapshot.get("fade_after_bar") or 0)
        if best_bar > 0 and fade_after_bar > best_bar:
            best_label = f"Best at {best_bar} bars, fades after {fade_after_bar}"
        elif best_bar > 0:
            best_label = f"Best at {best_bar} bars"
        else:
            best_label = str(snapshot.get("best_label") or "").strip() or "around 0 bars"
        return best_label
    if resolved_signals > 0:
        return f"Building ({resolved_signals} resolved)"
    return "—"


def _missing_hold_backfill_count(df_events: pd.DataFrame, df_forward_windows: pd.DataFrame) -> int:
    if df_events is None or df_events.empty or "signal_key" not in df_events.columns:
        return 0
    resolved = df_events.copy()
    status_series = resolved.get("status", pd.Series(index=resolved.index, dtype=object)).fillna("").astype(str).str.upper()
    resolved = resolved[status_series.eq("RESOLVED")].copy()
    if resolved.empty:
        return 0
    resolved_keys = set(resolved["signal_key"].fillna("").astype(str).str.strip())
    resolved_keys.discard("")
    if not resolved_keys:
        return 0
    if df_forward_windows is None or df_forward_windows.empty or "signal_key" not in df_forward_windows.columns:
        return int(len(resolved_keys))
    forward_keys = set(df_forward_windows["signal_key"].fillna("").astype(str).str.strip())
    forward_keys.discard("")
    return int(len(resolved_keys.difference(forward_keys)))


def _hold_archive_progress_snapshot(df_events: pd.DataFrame, df_forward_windows: pd.DataFrame) -> dict[str, float]:
    if df_events is None or df_events.empty or "signal_key" not in df_events.columns:
        return {"resolved": 0.0, "ready": 0.0, "missing": 0.0, "coverage_pct": 0.0}
    resolved = df_events.copy()
    status_series = resolved.get("status", pd.Series(index=resolved.index, dtype=object)).fillna("").astype(str).str.upper()
    resolved = resolved[status_series.eq("RESOLVED")].copy()
    if resolved.empty:
        return {"resolved": 0.0, "ready": 0.0, "missing": 0.0, "coverage_pct": 0.0}
    resolved_keys = set(resolved["signal_key"].fillna("").astype(str).str.strip())
    resolved_keys.discard("")
    if not resolved_keys:
        return {"resolved": 0.0, "ready": 0.0, "missing": 0.0, "coverage_pct": 0.0}
    forward_keys: set[str] = set()
    if df_forward_windows is not None and not df_forward_windows.empty and "signal_key" in df_forward_windows.columns:
        forward_keys = set(df_forward_windows["signal_key"].fillna("").astype(str).str.strip())
        forward_keys.discard("")
    ready = int(len(resolved_keys.intersection(forward_keys)))
    missing = int(len(resolved_keys.difference(forward_keys)))
    resolved_count = int(len(resolved_keys))
    coverage_pct = (ready / resolved_count * 100.0) if resolved_count > 0 else 0.0
    return {
        "resolved": float(resolved_count),
        "ready": float(ready),
        "missing": float(missing),
        "coverage_pct": float(coverage_pct),
    }


def _same_hold_progress(left: Mapping[str, object], right: Mapping[str, object]) -> bool:
    return (
        int(float(left.get("resolved") or 0.0)) == int(float(right.get("resolved") or 0.0))
        and int(float(left.get("ready") or 0.0)) == int(float(right.get("ready") or 0.0))
        and int(float(left.get("missing") or 0.0)) == int(float(right.get("missing") or 0.0))
        and round(float(left.get("coverage_pct") or 0.0), 4) == round(float(right.get("coverage_pct") or 0.0), 4)
    )


def _hold_guidance_direction_note(
    snapshot: Mapping[str, object],
    *,
    symbol: str,
    timeframe: str,
    direction_label: str,
) -> tuple[str, str]:
    resolved_signals = int(snapshot.get("resolved_signals") or 0)
    scope_label = f"{symbol.upper()} {timeframe.upper()} {direction_label}"
    if resolved_signals <= 0:
        return (
            f"No resolved <b>{direction_label.lower()}</b> signals for <b>{scope_label}</b> are available in this view yet.",
            "neutral",
        )
    if not bool(snapshot.get("available")):
        return (
            (
                f"<b>{scope_label}</b> is still building hold guidance. "
                f"We have <b>{resolved_signals}</b> resolved signals in this view, but not enough checkpoint depth yet."
            ),
            "neutral",
        )
    best_bar = int(snapshot.get("best_bar") or 0)
    best_label = str(snapshot.get("best_label") or "").strip() or "around 0 bars"
    best_style = str(snapshot.get("best_style") or "Standard Hold").strip()
    sample = int(snapshot.get("sample") or 0)
    avg_dir_return_pct = float(snapshot.get("avg_dir_return_pct") or 0.0)
    follow_through_pct = float(snapshot.get("follow_through_pct") or 0.0)
    fade_after_bar = int(snapshot.get("fade_after_bar") or 0)
    fade_text = (
        f"Edge starts to fade after roughly <b>{fade_after_bar}</b> bars."
        if fade_after_bar > 0
        else "Edge has not clearly faded inside the measured checkpoint ladder yet."
    )
    tone = "positive" if avg_dir_return_pct > 0.0 else "warning"
    lead_text = (
        f"<b>{direction_label}: Best at {best_bar} bars</b><br>"
        if best_bar > 0
        else f"<b>{direction_label}: {best_label}</b><br>"
    )
    return (
        (
            lead_text
            +
            f"Style: <b>{best_style}</b>.<br>"
            f"Historical read for <b>{scope_label}</b>, based on <b>{sample}</b> resolved signals at that checkpoint. "
            f"Avg directional return: <b>{avg_dir_return_pct:+.2f}%</b>, follow-through: <b>{follow_through_pct:.1f}%</b>. "
            f"{fade_text}"
        ),
        tone,
    )


def _build_coin_hold_guidance_rows(
    df_events: pd.DataFrame,
    df_forward_windows: pd.DataFrame,
    build_hold_window_intelligence,
    *,
    timeframe_filter: str,
) -> list[dict[str, object]]:
    if df_events is None or df_events.empty:
        return []
    d = df_events.copy()
    if "signal_key" not in d.columns or "timeframe" not in d.columns:
        return []
    d["signal_key"] = d["signal_key"].fillna("").astype(str).str.strip()
    d = d[d["signal_key"].ne("")].copy()
    if d.empty:
        return []
    d["_hold_direction"] = d.get("direction", pd.Series(index=d.index, dtype=object)).map(_display_trade_direction)
    d["timeframe"] = d["timeframe"].fillna("").astype(str).str.strip().str.lower()
    available_timeframes = set(d["timeframe"].tolist())
    if str(timeframe_filter or "").strip() == "All":
        ordered = [tf for tf in _HOLD_GUIDANCE_TIMEFRAME_ORDER if tf in available_timeframes]
        extras = sorted(available_timeframes.difference(_HOLD_GUIDANCE_TIMEFRAME_ORDER))
        timeframes = [*ordered, *extras]
    else:
        timeframes = [str(timeframe_filter).strip().lower()]
    if df_forward_windows is None or df_forward_windows.empty:
        windows = pd.DataFrame(columns=["signal_key"])
    else:
        windows = df_forward_windows.copy()
        windows["signal_key"] = windows["signal_key"].fillna("").astype(str).str.strip()
    rows: list[dict[str, object]] = []
    for timeframe in timeframes:
        tf_df = d[d["timeframe"].eq(timeframe)].copy()
        if tf_df.empty:
            continue
        row: dict[str, object] = {"Timeframe": timeframe.upper()}
        for direction_label in ("Upside", "Downside"):
            direction_df = tf_df[tf_df["_hold_direction"].eq(direction_label)].copy()
            direction_keys = direction_df["signal_key"].tolist()
            direction_windows = (
                windows[windows["signal_key"].isin(direction_keys)].copy()
                if direction_keys and not windows.empty
                else pd.DataFrame(columns=windows.columns)
            )
            snapshot = build_hold_window_intelligence(direction_df, direction_windows)
            row[f"{direction_label} Snapshot"] = snapshot
            row[f"{direction_label} Hold"] = _hold_guidance_cell(snapshot, direction_label=direction_label)
        rows.append(row)
    return rows


def _format_review_metric(
    value: float,
    *,
    available: bool,
    decimals: int = 1,
    pct: bool = False,
    signed: bool = False,
) -> str:
    if not available:
        return "N/A"
    number = float(value)
    if pct:
        return f"{number:+.{decimals}f}%" if signed else f"{number:.{decimals}f}%"
    return f"{number:+.{decimals}f}" if signed else f"{number:.{decimals}f}"


def _qualified_summary_rows(
    summary_df: pd.DataFrame,
    *,
    count_field: str,
    min_count: int,
) -> pd.DataFrame:
    if summary_df is None or summary_df.empty or count_field not in summary_df.columns:
        return pd.DataFrame()
    d = summary_df.copy()
    d[count_field] = pd.to_numeric(d[count_field], errors="coerce").fillna(0.0)
    return d[d[count_field] >= float(min_count)].copy()


def _prefer_known_summary_rows(summary_df: pd.DataFrame, *, label_field: str) -> pd.DataFrame:
    if summary_df is None or summary_df.empty or label_field not in summary_df.columns:
        return pd.DataFrame()
    d = summary_df.copy()
    labels = d[label_field].fillna("").astype(str).str.strip()
    known_mask = ~labels.str.contains(r"\bUnknown\b", case=False, na=False)
    if bool(known_mask.any()):
        return d.loc[known_mask].copy()
    return pd.DataFrame()


def _archive_building_card(title: str, body_html: str) -> dict[str, str]:
    return {
        "title": title,
        "body_html": body_html,
        "tone": "neutral",
        "kind": "building",
    }


def _prepare_section_cards(
    cards: list[dict[str, str]],
    *,
    max_actionable: int = 3,
) -> list[dict[str, str]]:
    visible_cards = [card for card in list(cards or []) if card]
    if not visible_cards:
        return []
    actionable = [card for card in visible_cards if str(card.get("kind") or "").strip().lower() != "building"]
    building = [card for card in visible_cards if str(card.get("kind") or "").strip().lower() == "building"]
    prepared = actionable[: max(1, int(max_actionable))]
    if building:
        titles = [str(card.get("title") or "").strip() for card in building if str(card.get("title") or "").strip()]
        preview = ", ".join(titles[:3])
        extra = "" if len(titles) <= 3 else f" +{len(titles) - 3} more"
        prepared.append(
            {
                "title": "Still Building",
                "body_html": (
                    f"Still building: <b>{preview}</b>{extra}. "
                    "These need more resolved signals or journaled closed trades before they become trustworthy."
                ),
                "tone": "neutral",
            }
        )
    return prepared if prepared else visible_cards[:1]


def _execution_vs_system_note(execution_snapshot: dict[str, float]) -> tuple[str, str]:
    taken = float(execution_snapshot.get("taken", 0.0) or 0.0)
    taken_resolved = float(execution_snapshot.get("taken_resolved", 0.0) or 0.0)
    actual_closed = float(execution_snapshot.get("actual_closed", 0.0) or 0.0)
    if taken <= 0.0 and actual_closed <= 0.0:
        return (
            "Execution journal is still building. The system archive is live, but you have not journaled any taken trades yet.",
            "neutral",
        )
    if taken_resolved < _MIN_EXECUTION_ARCHIVE_ROWS or actual_closed < _MIN_EXECUTION_ARCHIVE_ROWS:
        return (
            "Execution archive is still thin. Use this section directionally for now, but wait for more journaled trades before trusting hard conclusions.",
            "neutral",
        )
    return (
        (
            f"Taken setups had <b>{execution_snapshot['taken_follow_through_rate']:.1f}%</b> signal follow-through, "
            f"while closed real trades finished with <b>{execution_snapshot['actual_win_rate']:.1f}%</b> win rate. "
            f"Execution gap is <b>{execution_snapshot['execution_gap_pct']:+.2f}%</b>. "
            f"Skipped winners: <b>{int(execution_snapshot['skipped_winners'])}</b>."
        ),
        "positive" if float(execution_snapshot.get("execution_gap_pct", 0.0) or 0.0) >= 0.0 else "warning",
    )


def _build_execution_review_cards(execution_snapshot: dict[str, float]) -> list[dict[str, str]]:
    total = int(execution_snapshot.get("total", 0.0) or 0.0)
    overlay_marked = int(execution_snapshot.get("overlay_marked", 0.0) or 0.0)
    overlay_coverage_pct = float(execution_snapshot.get("overlay_coverage_pct", 0.0) or 0.0)
    taken = int(execution_snapshot.get("taken", 0.0) or 0.0)
    taken_resolved = int(execution_snapshot.get("taken_resolved", 0.0) or 0.0)
    actual_closed = int(execution_snapshot.get("actual_closed", 0.0) or 0.0)
    journal_coverage_pct = float(execution_snapshot.get("journal_coverage_pct", 0.0) or 0.0)
    execution_gap_pct = float(execution_snapshot.get("execution_gap_pct", 0.0) or 0.0)
    skipped_winners = int(execution_snapshot.get("skipped_winners", 0.0) or 0.0)
    skipped_resolved = int(execution_snapshot.get("skipped_resolved", 0.0) or 0.0)
    skipped_winner_rate = float(execution_snapshot.get("skipped_winner_rate", 0.0) or 0.0)

    cards: list[dict[str, str]] = []

    if total <= 0:
        return [
            _archive_building_card(
                "Manual Marking",
                "No signals are in this view yet, so execution review is still waiting for archive data.",
            )
        ]

    if overlay_marked <= 0:
        cards.append(
            _archive_building_card(
                "Manual Marking",
                "No setups in this view have been tagged as <b>Taken</b>, <b>Skipped</b>, or <b>Observed</b> yet. Start there before reading execution conclusions.",
            )
        )
    elif overlay_coverage_pct < 50.0:
        cards.append(
            {
                "title": "Manual Marking",
                "body_html": (
                    f"Only <b>{overlay_marked}/{total}</b> setups in this view are tagged "
                    f"(<b>{overlay_coverage_pct:.1f}% coverage</b>). "
                    "Tag more setups first so skipped-winner and execution-gap reads become trustworthy."
                ),
                "tone": "warning",
            }
        )
    else:
        cards.append(
            {
                "title": "Manual Marking",
                "body_html": (
                    f"Manual marking is healthy in this view with <b>{overlay_marked}/{total}</b> setups marked "
                    f"(<b>{overlay_coverage_pct:.1f}% coverage</b>)."
                ),
                "tone": "positive",
            }
        )

    if taken <= 0:
        cards.append(
            _archive_building_card(
                "Trade Journal",
                "You have not marked any setups as <b>Taken</b> in this view yet, so the real execution archive is still building.",
            )
        )
    elif actual_closed < _MIN_EXECUTION_ARCHIVE_ROWS:
        cards.append(
            _archive_building_card(
                "Trade Journal",
                (
                    f"<b>{taken}</b> taken setup(s) are marked, but only <b>{actual_closed}</b> closed trade(s) are journaled. "
                    "Add real exits before trusting realized PnL or win-rate conclusions."
                ),
            )
        )
    elif journal_coverage_pct < 60.0:
        cards.append(
            {
                "title": "Trade Journal",
                "body_html": (
                    f"Closed-trade journaling is improving, but only <b>{journal_coverage_pct:.1f}% of taken setups</b> in this view have a recorded exit. "
                    "A few more real exits would make the execution read materially stronger."
                ),
                "tone": "warning",
            }
        )
    else:
        cards.append(
            {
                "title": "Trade Journal",
                "body_html": (
                    f"Closed-trade journaling is usable here: <b>{actual_closed}</b> closed trade(s) across <b>{taken}</b> taken setup(s) "
                    f"(<b>{journal_coverage_pct:.1f}% coverage</b>)."
                ),
                "tone": "positive",
            }
        )

    if taken_resolved < _MIN_EXECUTION_ARCHIVE_ROWS or actual_closed < _MIN_EXECUTION_ARCHIVE_ROWS:
        cards.append(
            _archive_building_card(
                "Execution Edge",
                "Execution edge is still building. We need more taken setups to resolve and more closed real trades before comparing system quality to realized execution cleanly.",
            )
        )
    else:
        if execution_gap_pct >= 0.5:
            tone = "positive"
            edge_read = (
                f"Real execution is running <b>{execution_gap_pct:+.2f}% ahead</b> of the signal's directional return on taken trades."
            )
        elif execution_gap_pct <= -0.5:
            tone = "warning"
            edge_read = (
                f"Real execution is trailing the signal edge by <b>{execution_gap_pct:+.2f}%</b> on taken trades."
            )
        else:
            tone = "neutral"
            edge_read = (
                f"Real execution is broadly aligned with the signal edge at <b>{execution_gap_pct:+.2f}%</b>."
            )
        skipped_read = (
            f" Skipped winners: <b>{skipped_winners}</b> out of <b>{skipped_resolved}</b> skipped resolved setups "
            f"(<b>{skipped_winner_rate:.1f}%</b>)."
            if skipped_resolved > 0
            else ""
        )
        cards.append(
            {
                "title": "Execution Edge",
                "body_html": edge_read + skipped_read,
                "tone": tone,
            }
        )

    return _prepare_section_cards(cards, max_actionable=3)


def _build_execution_signal_cards(execution_snapshot: Mapping[str, object]) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    works_cards: list[dict[str, str]] = []
    fail_cards: list[dict[str, str]] = []

    taken = int(execution_snapshot.get("taken", 0.0) or 0.0)
    taken_resolved = int(execution_snapshot.get("taken_resolved", 0.0) or 0.0)
    actual_closed = int(execution_snapshot.get("actual_closed", 0.0) or 0.0)
    journal_coverage_pct = float(execution_snapshot.get("journal_coverage_pct", 0.0) or 0.0)
    execution_gap_pct = float(execution_snapshot.get("execution_gap_pct", 0.0) or 0.0)
    skipped_winners = int(execution_snapshot.get("skipped_winners", 0.0) or 0.0)
    skipped_resolved = int(execution_snapshot.get("skipped_resolved", 0.0) or 0.0)
    skipped_winner_rate = float(execution_snapshot.get("skipped_winner_rate", 0.0) or 0.0)
    actual_win_rate = float(execution_snapshot.get("actual_win_rate", 0.0) or 0.0)

    if taken >= _MIN_EXECUTION_ARCHIVE_ROWS:
        if journal_coverage_pct >= 60.0 and actual_closed >= _MIN_EXECUTION_ARCHIVE_ROWS:
            works_cards.append(
                {
                    "title": "Journal Coverage",
                    "body_html": (
                        f"Real execution archive is usable here: <b>{actual_closed}</b> closed trade(s) across "
                        f"<b>{taken}</b> taken setups (<b>{journal_coverage_pct:.1f}% coverage</b>)."
                    ),
                    "tone": "positive",
                }
            )
        elif journal_coverage_pct > 0.0:
            fail_cards.append(
                {
                    "title": "Thin Journal Coverage",
                    "body_html": (
                        f"Only <b>{journal_coverage_pct:.1f}%</b> of taken setups have a recorded exit "
                        f"(<b>{actual_closed}</b> closed trade(s) across <b>{taken}</b> taken setups). "
                        "Journal more exits before trusting realized execution conclusions."
                    ),
                    "tone": "warning",
                }
            )

    if taken_resolved >= _MIN_EXECUTION_ARCHIVE_ROWS and actual_closed >= _MIN_EXECUTION_ARCHIVE_ROWS:
        if execution_gap_pct >= 0.5:
            works_cards.append(
                {
                    "title": "Execution Edge",
                    "body_html": (
                        f"Taken trades are running <b>{execution_gap_pct:+.2f}%</b> ahead of the signal edge, "
                        f"with <b>{actual_win_rate:.1f}%</b> realized win rate across closed trades."
                    ),
                    "tone": "positive",
                }
            )
        elif execution_gap_pct <= -0.5:
            fail_cards.append(
                {
                    "title": "Execution Drag",
                    "body_html": (
                        f"Taken trades are trailing the signal edge by <b>{execution_gap_pct:+.2f}%</b>. "
                        "The system is finding edge, but realized execution is giving some of it back."
                    ),
                    "tone": "warning",
                }
            )

    if skipped_resolved >= _MIN_EXECUTION_ARCHIVE_ROWS and skipped_winner_rate >= 40.0:
        fail_cards.append(
            {
                "title": "Missed Winners",
                "body_html": (
                    f"<b>{skipped_winners}</b> of <b>{skipped_resolved}</b> skipped resolved setups were winners "
                    f"(<b>{skipped_winner_rate:.1f}%</b>). "
                    "Selection discipline may be leaving too much valid edge on the table."
                ),
                "tone": "warning",
            }
        )

    return works_cards, fail_cards


def _build_hold_signal_cards(
    hold_guidance_rows: list[Mapping[str, object]],
    *,
    symbol_filter: str,
) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    records: list[dict[str, object]] = []
    for row in list(hold_guidance_rows or []):
        timeframe_label = str(row.get("Timeframe") or "").strip().upper()
        for direction_label in ("Upside", "Downside"):
            snapshot = row.get(f"{direction_label} Snapshot", {})
            if not isinstance(snapshot, Mapping):
                continue
            records.append(
                {
                    "timeframe": timeframe_label or "UNKNOWN",
                    "direction": direction_label,
                    "available": bool(snapshot.get("available")),
                    "best_label": str(snapshot.get("best_label") or "").strip(),
                    "best_style": str(snapshot.get("best_style") or "").strip() or "Standard Hold",
                    "sample": int(snapshot.get("sample") or 0),
                    "resolved_signals": int(snapshot.get("resolved_signals") or 0),
                    "follow_through_pct": float(snapshot.get("follow_through_pct") or 0.0),
                    "avg_dir_return_pct": float(snapshot.get("avg_dir_return_pct") or 0.0),
                    "edge_score": float(snapshot.get("edge_score") or 0.0),
                }
            )

    available_records = [record for record in records if bool(record.get("available"))]
    if not available_records:
        return [], []

    symbol_label = str(symbol_filter or "").strip().upper() or "Coin"
    works_cards: list[dict[str, str]] = []
    fail_cards: list[dict[str, str]] = []

    best_record = sorted(
        available_records,
        key=lambda record: (
            float(record.get("edge_score", 0.0) or 0.0),
            float(record.get("avg_dir_return_pct", 0.0) or 0.0),
            float(record.get("follow_through_pct", 0.0) or 0.0),
            int(record.get("sample", 0) or 0),
        ),
        reverse=True,
    )[0]
    works_cards.append(
        {
            "title": "Best Hold Edge",
            "body_html": (
                f"<b>{symbol_label} {best_record['timeframe']} {best_record['direction']}</b> has the cleanest hold profile so far "
                f"({best_record['best_label']}, {best_record['best_style']}, "
                f"<b>{best_record['follow_through_pct']:.1f}%</b> follow-through, "
                f"<b>{best_record['avg_dir_return_pct']:+.2f}%</b> avg directional return across "
                f"<b>{int(best_record['sample'])}</b> checkpoint-matched resolved signals)."
            ),
            "tone": "positive" if float(best_record["avg_dir_return_pct"]) > 0.0 else "neutral",
        }
    )

    weakest_record = sorted(
        available_records,
        key=lambda record: (
            float(record.get("edge_score", 0.0) or 0.0),
            float(record.get("avg_dir_return_pct", 0.0) or 0.0),
            float(record.get("follow_through_pct", 0.0) or 0.0),
            int(record.get("sample", 0) or 0),
        ),
    )[0]
    if (
        len(available_records) >= 2
        or float(weakest_record.get("avg_dir_return_pct", 0.0) or 0.0) < 0.0
        or float(weakest_record.get("follow_through_pct", 0.0) or 0.0) < 45.0
    ):
        fail_cards.append(
            {
                "title": "Weakest Hold Edge",
                "body_html": (
                    f"<b>{symbol_label} {weakest_record['timeframe']} {weakest_record['direction']}</b> is the weakest current hold profile "
                    f"({weakest_record['best_label']}, {weakest_record['best_style']}, "
                    f"<b>{weakest_record['follow_through_pct']:.1f}%</b> follow-through, "
                    f"<b>{weakest_record['avg_dir_return_pct']:+.2f}%</b> avg directional return across "
                    f"<b>{int(weakest_record['sample'])}</b> checkpoint-matched resolved signals)."
                ),
                "tone": "warning" if float(weakest_record["avg_dir_return_pct"]) < 0.0 or float(weakest_record["follow_through_pct"]) < 45.0 else "neutral",
            }
        )

    return works_cards, fail_cards


def _render_insight_card_grid(st, cards: list[dict[str, str]], *, columns: int = 3) -> None:
    visible_cards = [card for card in cards if card]
    if not visible_cards:
        return
    cols = st.columns(columns, gap="medium")
    for idx, card in enumerate(visible_cards):
        with cols[idx % columns]:
            render_insight_card(
                st,
                title=str(card.get("title") or ""),
                body_html=str(card.get("body_html") or ""),
                tone=str(card.get("tone") or "neutral"),
            )


def _render_compact_cohort_tables(
    st,
    *,
    df_events: pd.DataFrame,
    build_signal_cohort_summary,
    specs: list[tuple[str, str] | tuple[str, str, str]],
) -> None:
    visible_specs: list[tuple[str, str, pd.DataFrame]] = []
    for spec in specs:
        if len(spec) == 2:
            group_field, title = spec
            display_label = group_field
        else:
            group_field, title, display_label = spec
        if group_field not in df_events.columns:
            continue
        summary_df = build_signal_cohort_summary(df_events, group_field)
        if summary_df is None or summary_df.empty:
            continue
        known_summary_df = _prefer_known_summary_rows(summary_df, label_field=group_field)
        if known_summary_df is not None and not known_summary_df.empty:
            summary_df = known_summary_df
        if display_label != group_field and group_field in summary_df.columns:
            summary_df = summary_df.rename(columns={group_field: display_label})
        visible_specs.append((group_field, title, summary_df))
    if not visible_specs:
        st.caption("No cohort data is available in this view yet.")
        return
    cols = st.columns(2, gap="medium")
    for idx, (_, title, summary_df) in enumerate(visible_specs):
        with cols[idx % 2]:
            st.markdown(f"##### {title}")
            st.dataframe(summary_df.round(2), hide_index=True, width="stretch")


def _render_hold_window_cohort_tables(
    st,
    *,
    df_events: pd.DataFrame,
    df_forward_windows: pd.DataFrame,
    build_hold_window_cohort_summary,
    specs: list[tuple[str, str]],
) -> None:
    visible_specs: list[tuple[str, str, pd.DataFrame]] = []
    for group_field, title in specs:
        if group_field not in df_events.columns:
            continue
        summary_df = build_hold_window_cohort_summary(
            df_events,
            df_forward_windows,
            group_field,
        )
        if summary_df is None or summary_df.empty:
            continue
        known_summary_df = _prefer_known_summary_rows(summary_df, label_field=group_field)
        if known_summary_df is not None and not known_summary_df.empty:
            summary_df = known_summary_df
        visible_specs.append((group_field, title, summary_df))
    if not visible_specs:
        st.caption("Hold history is still building in this view. We need more resolved signals with checkpoint history.")
        return
    cols = st.columns(2, gap="medium")
    for idx, (_, title, summary_df) in enumerate(visible_specs):
        with cols[idx % 2]:
            st.markdown(f"##### {title}")
            st.dataframe(summary_df.round(2), hide_index=True, width="stretch")


def _render_execution_review_section(
    *,
    st,
    df_events: pd.DataFrame,
    execution_snapshot: dict[str, float],
    db_path: str,
    save_signal_trade_overlay,
    save_signal_trade_journal,
    positive_color: str,
    warning_color: str,
    negative_color: str,
    muted_color: str,
) -> None:
    st.markdown("### Execution Review")
    st.caption("Use this section after the top-level read. It separates system quality from your own execution decisions.")
    review_cards = _build_execution_review_cards(execution_snapshot)
    _render_insight_card_grid(st, review_cards, columns=3)
    trade_overlay_count = int(execution_snapshot.get("overlay_marked", 0.0) or 0.0)
    total_signals = int(execution_snapshot.get("total", float(len(df_events))) or 0.0)
    taken_count = int(execution_snapshot.get("taken", 0.0) or 0.0)
    closed_trade_count = int(execution_snapshot.get("actual_closed", 0.0) or 0.0)
    overlay_coverage_pct = float(execution_snapshot.get("overlay_coverage_pct", 0.0) or 0.0)
    journal_coverage_pct = float(execution_snapshot.get("journal_coverage_pct", 0.0) or 0.0)
    execution_gap_pct = float(execution_snapshot.get("execution_gap_pct", 0.0) or 0.0)
    skipped_winners = int(execution_snapshot.get("skipped_winners", 0.0) or 0.0)
    skipped_resolved = int(execution_snapshot.get("skipped_resolved", 0.0) or 0.0)
    skipped_winner_rate = float(execution_snapshot.get("skipped_winner_rate", 0.0) or 0.0)
    avg_actual_pnl = float(execution_snapshot.get("avg_actual_pnl", 0.0) or 0.0)
    st.markdown(
        (
            f"<div class='market-note-box' style='border:1px solid rgba(0,212,255,0.26); border-left:4px solid {positive_color};"
            " background:rgba(0,212,255,0.04); margin:0.35rem 0 1rem 0;'>"
            f"<b style='color:{positive_color};'>Workflow:</b> "
            "First tag the setup as Taken, Skipped, or Observed. Then, if you actually took it, add the real entry and exit. "
            "That keeps system signal quality separate from your own execution."
            "</div>"
        ),
        unsafe_allow_html=True,
    )
    review_cols = st.columns([1.2, 1.0], gap="medium")
    render_kpi_grid(
        st,
        items=[
            {
                "label": "Overlay Coverage",
                "value": f"{trade_overlay_count}/{total_signals}",
                "value_color": positive_color if trade_overlay_count > 0 else muted_color,
                "subtext": f"{overlay_coverage_pct:.1f}% of this view has a manual execution tag",
            },
            {
                "label": "Journal Coverage",
                "value": f"{closed_trade_count}/{taken_count}",
                "value_color": positive_color if closed_trade_count > 0 else muted_color,
                "subtext": (
                    f"{journal_coverage_pct:.1f}% of taken setups now have a recorded exit"
                    if taken_count > 0
                    else "No taken setups in this view yet"
                ),
            },
            {
                "label": "Execution Gap",
                "value": _format_review_metric(
                    execution_gap_pct,
                    available=closed_trade_count > 0,
                    pct=True,
                    signed=True,
                    decimals=2,
                ),
                "value_color": (
                    positive_color
                    if closed_trade_count > 0 and execution_gap_pct >= 0.0
                    else (negative_color if closed_trade_count > 0 else muted_color)
                ),
                "subtext": (
                    "Avg realized PnL vs signal directional return on taken trades"
                    if closed_trade_count > 0
                    else "Execution gap appears after you journal closed trades"
                ),
            },
            {
                "label": "Skipped Winners",
                "value": skipped_winners,
                "value_color": warning_color if skipped_winners > 0 else muted_color,
                "subtext": (
                    f"{skipped_winner_rate:.1f}% of skipped resolved setups were winners"
                    if skipped_resolved > 0
                    else "No skipped resolved setups in this view yet"
                ),
            },
        ],
        columns=4,
    )
    render_kpi_grid(
        st,
        items=[
            {
                "label": "Avg Realized PnL",
                "value": _format_review_metric(
                    avg_actual_pnl,
                    available=closed_trade_count > 0,
                    pct=True,
                    signed=True,
                    decimals=2,
                ),
                "value_color": (
                    positive_color
                    if closed_trade_count > 0 and avg_actual_pnl >= 0.0
                    else (negative_color if closed_trade_count > 0 else muted_color)
                ),
                "subtext": (
                    "Average realized result across journaled closed trades"
                    if closed_trade_count > 0
                    else "Realized PnL appears after you journal closed trades"
                ),
            },
            {
                "label": "Taken Resolved",
                "value": int(execution_snapshot.get("taken_resolved", 0.0) or 0.0),
                "value_color": positive_color if int(execution_snapshot.get("taken_resolved", 0.0) or 0.0) > 0 else muted_color,
                "subtext": "Taken setups that have already resolved on the archive horizon",
            },
            {
                "label": "Taken Follow-Through",
                "value": _format_review_metric(
                    float(execution_snapshot.get("taken_follow_through_rate", 0.0) or 0.0),
                    available=int(execution_snapshot.get("taken_resolved", 0.0) or 0.0) > 0,
                    pct=True,
                    decimals=1,
                ),
                "value_color": (
                    positive_color
                    if int(execution_snapshot.get("taken_resolved", 0.0) or 0.0) > 0
                    and float(execution_snapshot.get("taken_follow_through_rate", 0.0) or 0.0) >= 55.0
                    else (warning_color if int(execution_snapshot.get("taken_resolved", 0.0) or 0.0) > 0 else muted_color)
                ),
                "subtext": (
                    "Signal follow-through rate across taken resolved setups"
                    if int(execution_snapshot.get("taken_resolved", 0.0) or 0.0) > 0
                    else "Follow-through appears after taken setups resolve"
                ),
            },
            {
                "label": "Closed Trade Win Rate",
                "value": _format_review_metric(
                    float(execution_snapshot.get("actual_win_rate", 0.0) or 0.0),
                    available=closed_trade_count > 0,
                    pct=True,
                    decimals=1,
                ),
                "value_color": (
                    positive_color
                    if closed_trade_count > 0 and float(execution_snapshot.get("actual_win_rate", 0.0) or 0.0) >= 55.0
                    else (warning_color if closed_trade_count > 0 else muted_color)
                ),
                "subtext": (
                    "Win rate across journaled closed trades"
                    if closed_trade_count > 0
                    else "Win rate appears after you journal closed trades"
                ),
            },
        ],
        columns=4,
    )
    signal_options: dict[str, str] = {}
    for _, row in df_events.iterrows():
        signal_key = str(row.get("signal_key") or "").strip()
        if not signal_key:
            continue
        event_time = pd.to_datetime(row.get("event_time"), errors="coerce")
        ts_label = event_time.strftime("%Y-%m-%d %H:%M") if pd.notna(event_time) else "Unknown time"
        setup_label = setup_confirm_display(
            str(row.get("setup_confirm") or ""),
            action_reason=str(row.get("action_reason") or ""),
            direction=str(row.get("direction") or ""),
        )
        label = (
            f"{ts_label} • {str(row.get('symbol') or '')} • "
            f"{str(row.get('timeframe') or '')} • {setup_label}"
        )
        signal_options[label] = signal_key

    with review_cols[0]:
        if signal_options:
            with st.expander("Mark or Journal a Setup", expanded=False):
                selected_label = st.selectbox(
                    "Tracked setup",
                    list(signal_options.keys()),
                    index=0,
                    key="signal_review_trade_overlay_pick",
                    help="Use this to mark whether you actually took a tracked setup, skipped it, or just observed it.",
                )
                selected_key = signal_options[selected_label]
                selected_row = df_events[df_events["signal_key"].astype(str) == selected_key].iloc[0]
                current_decision = str(selected_row.get("trade_decision") or "").strip()
                current_note = str(selected_row.get("trade_note") or "").strip()
                current_side = str(selected_row.get("actual_trade_side") or "").strip().upper()
                current_entry_price = selected_row.get("actual_entry_price")
                current_entry_at = str(selected_row.get("actual_entry_at") or "").strip()
                current_exit_price = selected_row.get("actual_exit_price")
                current_exit_at = str(selected_row.get("actual_exit_at") or "").strip()
                current_exit_reason = str(selected_row.get("actual_exit_reason") or "").strip()
                current_trade_status = str(selected_row.get("actual_trade_status") or "").strip().upper()
                decision_options = ["Taken", "Skipped", "Observed", "Clear"]
                default_idx = decision_options.index(current_decision) if current_decision in decision_options else 0
                with st.form("signal_trade_overlay_form", clear_on_submit=False):
                    chosen_decision = st.selectbox(
                        "Your action",
                        decision_options,
                        index=default_idx,
                        key="signal_trade_overlay_decision",
                    )
                    trade_note = st.text_input(
                        "Note",
                        value=current_note,
                        key="signal_trade_overlay_note",
                        placeholder="Optional execution note",
                    )
                    submitted = st.form_submit_button("Save tag", use_container_width=False)
                if submitted:
                    decision_value = "" if chosen_decision == "Clear" else chosen_decision
                    saved = save_signal_trade_overlay(
                        selected_key,
                        trade_decision=decision_value,
                        trade_note=trade_note,
                        db_path=db_path,
                    )
                    if saved:
                        st.success("Setup tag saved.")
                        st.rerun()
                    else:
                        st.error("Setup tag could not be saved for that signal.")

                has_journal = any(
                    [
                        current_side,
                        current_entry_at,
                        current_exit_at,
                        pd.notna(current_entry_price),
                        pd.notna(current_exit_price),
                        current_exit_reason,
                    ]
                )
                if current_decision == "Taken" or has_journal:
                    signal_direction = str(selected_row.get("direction") or "").strip().upper()
                    suggested_side = "Upside" if signal_direction == "UPSIDE" else "Downside"
                    side_options = ["Upside", "Downside"]
                    default_side = _display_trade_direction(current_side) or suggested_side
                    default_side_idx = side_options.index(default_side)
                    exit_reason_options = ["Open", "Target", "Stop", "Manual Exit", "Time Exit", "Invalidation", "Clear"]
                    mapped_exit_reason = current_exit_reason if current_exit_reason in exit_reason_options else ("Open" if current_trade_status != "CLOSED" else "Manual Exit")
                    with st.form("signal_trade_journal_form", clear_on_submit=False):
                        st.markdown("#### Real Trade Journal")
                        st.caption("Use this only for trades you really took. It keeps system signal quality separate from your actual execution.")
                        chosen_side = st.selectbox(
                            "Trade direction",
                            side_options,
                            index=default_side_idx,
                            key="signal_trade_journal_side",
                        )
                        entry_price_text = st.text_input(
                            "Entry price",
                            value="" if pd.isna(current_entry_price) else f"{float(current_entry_price):.8f}".rstrip("0").rstrip("."),
                            key="signal_trade_journal_entry_price",
                            placeholder="Example: 102.45",
                        )
                        entry_time_text = st.text_input(
                            "Entry time (UTC)",
                            value=current_entry_at,
                            key="signal_trade_journal_entry_at",
                            placeholder="2026-04-04T12:00:00Z",
                        )
                        exit_price_text = st.text_input(
                            "Exit price",
                            value="" if pd.isna(current_exit_price) else f"{float(current_exit_price):.8f}".rstrip("0").rstrip("."),
                            key="signal_trade_journal_exit_price",
                            placeholder="Leave blank if still open",
                        )
                        exit_time_text = st.text_input(
                            "Exit time (UTC)",
                            value=current_exit_at,
                            key="signal_trade_journal_exit_at",
                            placeholder="Leave blank if still open",
                        )
                        chosen_exit_reason = st.selectbox(
                            "Exit reason",
                            exit_reason_options,
                            index=exit_reason_options.index(mapped_exit_reason),
                            key="signal_trade_journal_exit_reason",
                        )
                        journal_submitted = st.form_submit_button("Save journal", use_container_width=False)
                    if journal_submitted:
                        if chosen_exit_reason == "Clear":
                            journal_saved = save_signal_trade_journal(selected_key, db_path=db_path)
                        else:
                            journal_saved = save_signal_trade_journal(
                                selected_key,
                                actual_trade_side=chosen_side,
                                actual_entry_price=entry_price_text,
                                actual_entry_at=entry_time_text,
                                actual_exit_price="" if chosen_exit_reason == "Open" else exit_price_text,
                                actual_exit_at="" if chosen_exit_reason == "Open" else exit_time_text,
                                actual_exit_reason="" if chosen_exit_reason == "Open" else chosen_exit_reason,
                                db_path=db_path,
                            )
                        if journal_saved:
                            st.success("Actual trade journal saved.")
                            st.rerun()
                        else:
                            st.error("Trade journal could not be saved. Entry price and trade direction are required.")
        else:
            st.caption("No tracked signals are available in this view yet for journaling.")


def _render_tracker_backup_restore(
    *,
    st,
    db_path: str,
    storage_snapshot,
    read_signal_tracker_db_bytes,
    backup_signal_tracker_db,
    restore_signal_tracker_db_bytes,
    fetch_signal_events_df,
    fetch_market_alerts_df,
) -> None:
    with st.expander("Archive Tools", expanded=False):
        render_insight_card(
            st,
            title="Archive Storage",
            body_html=(
                f"<b>{storage_snapshot.label}</b><br>"
                f"{storage_snapshot.note}<br><br>"
                f"<span style='color:#8B949E;'>Durability:</span> {storage_snapshot.durability_label}<br>"
                f"<span style='color:#8B949E;'>Durability note:</span> {storage_snapshot.durability_note}<br>"
                f"<span style='color:#8B949E;'>Archive:</span> {int(storage_snapshot.size_bytes):,} bytes<br>"
                f"<span style='color:#8B949E;'>Recovery:</span> {storage_snapshot.recovery_status}<br>"
                f"<span style='color:#8B949E;'>Recovery note:</span> {storage_snapshot.recovery_note}<br>"
                f"<span style='color:#8B949E;'>Backup mirror:</span> "
                f"{storage_snapshot.mirror_dir if storage_snapshot.mirror_enabled else 'Not configured'}<br>"
                f"<span style='color:#8B949E;'>Backup snapshots:</span> {int(storage_snapshot.mirror_count):,}<br>"
                f"<span style='color:#8B949E;'>Path:</span> {storage_snapshot.path}"
            ),
            tone=str(storage_snapshot.tone or "neutral"),
        )
        st.caption(
            "Use this area to protect the learning archive before risky changes, or move it cleanly between machines."
        )
        if st.button("Create Local Restore Point", key="signal_review_local_restore_point"):
            backup_path = str(backup_signal_tracker_db(db_path) or "").strip()
            if backup_path:
                st.session_state["signal_review_tracker_notice"] = f"Local restore point created at {backup_path}"
                st.session_state["signal_review_tracker_notice_tone"] = "success"
                st.rerun()
            st.session_state["signal_review_tracker_notice"] = "Archive database does not exist yet, so there is nothing to back up."
            st.session_state["signal_review_tracker_notice_tone"] = "warning"
            st.rerun()
        db_bytes = read_signal_tracker_db_bytes(db_path)
        st.download_button(
            "Download Archive DB",
            data=db_bytes,
            file_name=str(storage_snapshot.filename or "signal_tracker.sqlite3"),
            mime="application/x-sqlite3",
            disabled=not bool(db_bytes),
            on_click="ignore",
        )
        full_events_csv = fetch_signal_events_df(limit=100000, source="Market", db_path=db_path)
        full_alerts_csv = fetch_market_alerts_df(limit=5000, source="Market", db_path=db_path)
        st.download_button(
            "Download Signal Events CSV",
            data=full_events_csv.to_csv(index=False).encode("utf-8"),
            file_name="signal_events_backup.csv",
            mime="text/csv",
            on_click="ignore",
        )
        st.download_button(
            "Download Market Alerts CSV",
            data=full_alerts_csv.to_csv(index=False).encode("utf-8"),
            file_name="market_alerts_backup.csv",
            mime="text/csv",
            on_click="ignore",
        )
        st.markdown("---")
        st.caption(
            "Restore replaces the current archive database after first creating a local backup copy. Only upload archive snapshots created from this dashboard."
        )
        uploaded_tracker_db = st.file_uploader(
            "Upload archive snapshot",
            type=["sqlite3", "db", "sqlite"],
            key="signal_review_tracker_restore_upload",
            help="Accepted files are SQLite archive snapshots previously exported from Signal Archive.",
        )
        if st.button("Restore Uploaded Archive DB", key="signal_review_restore_uploaded_db"):
            if uploaded_tracker_db is None:
                st.session_state["signal_review_tracker_notice"] = "Choose an archive snapshot file before running restore."
                st.session_state["signal_review_tracker_notice_tone"] = "warning"
                st.rerun()
            try:
                restore_result = restore_signal_tracker_db_bytes(
                    uploaded_tracker_db.getvalue(),
                    db_path=db_path,
                    backup_existing=True,
                )
            except Exception as exc:
                st.session_state["signal_review_tracker_notice"] = f"Restore failed: {exc}"
                st.session_state["signal_review_tracker_notice_tone"] = "warning"
                st.rerun()
            backup_msg = (
                f" Previous DB backed up to {restore_result.backup_path}."
                if str(restore_result.backup_path or "").strip()
                else ""
            )
            st.session_state["signal_review_tracker_notice"] = (
                f"Archive DB restored to {restore_result.path} ({int(restore_result.restored_size):,} bytes).{backup_msg}"
            )
            st.session_state["signal_review_tracker_notice_tone"] = "success"
            st.rerun()


def render(ctx: dict) -> None:
    st = get_ctx(ctx, "st")
    POSITIVE = get_ctx(ctx, "POSITIVE")
    NEGATIVE = get_ctx(ctx, "NEGATIVE")
    WARNING = get_ctx(ctx, "WARNING")
    TEXT_MUTED = get_ctx(ctx, "TEXT_MUTED")
    fetch_ohlcv = get_ctx(ctx, "fetch_ohlcv")
    get_signal_tracker_db_path = get_ctx(ctx, "get_signal_tracker_db_path")
    init_signal_tracker_db = get_ctx(ctx, "init_signal_tracker_db")
    fetch_market_alerts_df = get_ctx(ctx, "fetch_market_alerts_df")
    count_market_alerts = get_ctx(ctx, "count_market_alerts")
    resolve_open_signal_events_via_fetch = get_ctx(ctx, "resolve_open_signal_events_via_fetch")
    backfill_signal_forward_windows_via_fetch = get_ctx(ctx, "backfill_signal_forward_windows_via_fetch")
    fetch_signal_events_df = get_ctx(ctx, "fetch_signal_events_df")
    fetch_signal_forward_windows_df = get_ctx(ctx, "fetch_signal_forward_windows_df")
    save_signal_trade_overlay = get_ctx(ctx, "save_signal_trade_overlay")
    save_signal_trade_journal = get_ctx(ctx, "save_signal_trade_journal")
    build_signal_review_snapshot = get_ctx(ctx, "build_signal_review_snapshot")
    build_execution_overlay_snapshot = get_ctx(ctx, "build_execution_overlay_snapshot")
    build_signal_cohort_summary = get_ctx(ctx, "build_signal_cohort_summary")
    build_hold_window_intelligence = get_ctx(ctx, "build_hold_window_intelligence")
    build_hold_window_cohort_summary = get_ctx(ctx, "build_hold_window_cohort_summary")
    annotate_alert_footprint = get_ctx(ctx, "annotate_alert_footprint")
    build_alert_effectiveness_summary = get_ctx(ctx, "build_alert_effectiveness_summary")
    build_adaptive_context_model = get_ctx(ctx, "build_adaptive_context_model")
    build_learning_edge_table = get_ctx(ctx, "build_learning_edge_table")
    build_tracker_storage_snapshot = get_ctx(ctx, "build_tracker_storage_snapshot")
    read_signal_tracker_db_bytes = get_ctx(ctx, "read_signal_tracker_db_bytes")
    backup_signal_tracker_db = get_ctx(ctx, "backup_signal_tracker_db")
    restore_signal_tracker_db_bytes = get_ctx(ctx, "restore_signal_tracker_db_bytes")

    db_path = init_signal_tracker_db(get_signal_tracker_db_path())
    storage_snapshot = build_tracker_storage_snapshot(db_path)

    render_page_header(
        st,
        title="Signal Archive",
        intro_html=(
            "Live signal and execution archive. Use this page to review what the dashboard actually logged, "
            "what you actually took, and whether the gap is in the system or in execution, not as a live entry screen."
        ),
    )

    status_filter = st.selectbox("Status", ["All", "Open", "Resolved"], index=0, key="signal_review_status")
    timeframe_filter = st.selectbox("Timeframe", ["All", "5m", "15m", "1h", "4h", "1d"], index=0, key="signal_review_tf")
    coin_filter_input = st.text_input(
        "Coin (optional)",
        value="",
        key="signal_review_symbol",
        placeholder="BTC",
        help="Leave blank to review the full archive, or type one coin symbol to focus on that coin only.",
    )
    symbol_filter = _normalize_symbol_filter(coin_filter_input)
    limit = st.slider("Rows in View", 50, 1000, 200, 25, key="signal_review_limit")
    current_market_version = current_decision_version("Market")
    refresh_limit_pairs = None if (symbol_filter or timeframe_filter != "All") else 48

    with st.spinner("Refreshing recent signal outcomes..."):
        resolved_now = int(
            resolve_open_signal_events_via_fetch(
                fetch_ohlcv=fetch_ohlcv,
                source="Market",
                db_path=db_path,
                limit_pairs=refresh_limit_pairs,
                candle_limit=260,
                symbol=symbol_filter or None,
                timeframe=None if timeframe_filter == "All" else timeframe_filter,
            )
        )

    df_events = fetch_signal_events_df(
        limit=int(limit),
        status=None if status_filter == "All" else status_filter.upper(),
        source="Market",
        timeframe=None if timeframe_filter == "All" else timeframe_filter,
        symbol=symbol_filter or None,
        decision_version=current_market_version,
        db_path=db_path,
    )
    df_alerts = fetch_market_alerts_df(limit=100, source="Market", db_path=db_path)
    active_alerts_count = int(count_market_alerts(active_only=True, source="Market", db_path=db_path))
    adaptive_archive_df = fetch_signal_events_df(
        limit=2000,
        status="RESOLVED",
        source="Market",
        db_path=db_path,
    )
    adaptive_archive_df = prefer_current_decision_version_slice(
        adaptive_archive_df,
        source="Market",
    )
    adaptive_mode = str(adaptive_archive_df.attrs.get("decision_version_mode") or "mixed_fallback")
    adaptive_rows = int(adaptive_archive_df.attrs.get("decision_version_rows") or 0)
    adaptive_total_rows = int(adaptive_archive_df.attrs.get("decision_version_total_rows") or len(adaptive_archive_df))
    learning_summary, learning_tone = _learning_readiness_summary(
        mode=adaptive_mode,
        current_rows=adaptive_rows,
        total_rows=adaptive_total_rows,
    )
    archive_health_body, archive_health_tone = _archive_health_summary(storage_snapshot)
    show_archive_health = archive_health_tone not in {"neutral", "positive"}

    top_card_specs: list[tuple[str, str, str]] = [
        (
            "Current View",
            _review_scope_summary(
                status_filter=status_filter,
                timeframe_filter=timeframe_filter,
                limit=int(limit),
                rows_in_view=int(len(df_events)),
                symbol_filter=symbol_filter,
            ),
            "neutral",
        ),
        ("Learning Readiness", learning_summary, learning_tone),
    ]
    if show_archive_health:
        top_card_specs.append(("Archive Alert", archive_health_body, archive_health_tone))

    top_insight_cols = st.columns(len(top_card_specs), gap="medium")
    for col, (title, body_html, tone) in zip(top_insight_cols, top_card_specs):
        with col:
            render_insight_card(
                st,
                title=title,
                body_html=body_html,
                tone=tone,
            )
    tracker_notice = st.session_state.pop("signal_review_tracker_notice", None)
    tracker_notice_tone = str(st.session_state.pop("signal_review_tracker_notice_tone", "info") or "info")
    if tracker_notice:
        if tracker_notice_tone == "success":
            st.success(str(tracker_notice))
        elif tracker_notice_tone == "warning":
            st.warning(str(tracker_notice))
        else:
            st.info(str(tracker_notice))

    if df_events.empty:
        st.info("No tracked signals match this view yet. Change the filters or let Market log more signals first.")
        _render_tracker_backup_restore(
            st=st,
            db_path=db_path,
            storage_snapshot=storage_snapshot,
            read_signal_tracker_db_bytes=read_signal_tracker_db_bytes,
            backup_signal_tracker_db=backup_signal_tracker_db,
            restore_signal_tracker_db_bytes=restore_signal_tracker_db_bytes,
            fetch_signal_events_df=fetch_signal_events_df,
            fetch_market_alerts_df=fetch_market_alerts_df,
        )
        return

    snapshot = build_signal_review_snapshot(df_events)
    execution_snapshot = build_execution_overlay_snapshot(df_events)
    signal_keys = (
        df_events["signal_key"].fillna("").astype(str).str.strip().tolist()
        if "signal_key" in df_events.columns
        else []
    )
    hold_guidance_enabled = bool(symbol_filter)
    df_forward_windows = (
        fetch_signal_forward_windows_df(
            signal_keys=signal_keys,
            db_path=db_path,
        )
        if hold_guidance_enabled and signal_keys
        else pd.DataFrame()
    )
    df_hold_archive_events = (
        fetch_signal_events_df(
            limit=2000,
            source="Market",
            timeframe=None if timeframe_filter == "All" else timeframe_filter,
            symbol=symbol_filter or None,
            decision_version=current_market_version,
            db_path=db_path,
        )
        if hold_guidance_enabled
        else pd.DataFrame()
    )
    hold_archive_signal_keys = (
        df_hold_archive_events["signal_key"].fillna("").astype(str).str.strip().tolist()
        if hold_guidance_enabled and "signal_key" in df_hold_archive_events.columns
        else []
    )
    df_hold_archive_windows = (
        fetch_signal_forward_windows_df(
            signal_keys=hold_archive_signal_keys,
            db_path=db_path,
        )
        if hold_guidance_enabled and hold_archive_signal_keys
        else pd.DataFrame()
    )
    hold_guidance_rows = (
        _build_coin_hold_guidance_rows(
            df_events,
            df_forward_windows,
            build_hold_window_intelligence,
            timeframe_filter=timeframe_filter,
        )
        if hold_guidance_enabled
        else []
    )
    hold_slice_progress = (
        _hold_archive_progress_snapshot(df_events, df_forward_windows)
        if hold_guidance_enabled
        else {"resolved": 0.0, "ready": 0.0, "missing": 0.0, "coverage_pct": 0.0}
    )
    hold_archive_progress = (
        _hold_archive_progress_snapshot(df_hold_archive_events, df_hold_archive_windows)
        if hold_guidance_enabled
        else {"resolved": 0.0, "ready": 0.0, "missing": 0.0, "coverage_pct": 0.0}
    )
    missing_hold_backfill = (
        _missing_hold_backfill_count(df_hold_archive_events, df_hold_archive_windows)
        if hold_guidance_enabled
        else 0
    )
    autofilled_hold_now = 0
    if hold_guidance_enabled and missing_hold_backfill > 0:
        autofilled_hold_now = int(
            backfill_signal_forward_windows_via_fetch(
                fetch_ohlcv=fetch_ohlcv,
                source="Market",
                db_path=db_path,
                limit_pairs=4,
                rows_per_pair=150,
                candle_limit=260,
                symbol=symbol_filter,
                timeframe=None if timeframe_filter == "All" else timeframe_filter,
                decision_version=current_market_version,
            )
        )
        if autofilled_hold_now > 0:
            df_forward_windows = (
                fetch_signal_forward_windows_df(
                    signal_keys=signal_keys,
                    db_path=db_path,
                )
                if signal_keys
                else pd.DataFrame()
            )
            df_hold_archive_windows = (
                fetch_signal_forward_windows_df(
                    signal_keys=hold_archive_signal_keys,
                    db_path=db_path,
                )
                if hold_archive_signal_keys
                else pd.DataFrame()
            )
            hold_guidance_rows = _build_coin_hold_guidance_rows(
                df_events,
                df_forward_windows,
                build_hold_window_intelligence,
                timeframe_filter=timeframe_filter,
            )
            hold_slice_progress = _hold_archive_progress_snapshot(df_events, df_forward_windows)
            hold_archive_progress = _hold_archive_progress_snapshot(df_hold_archive_events, df_hold_archive_windows)
            missing_hold_backfill = _missing_hold_backfill_count(df_hold_archive_events, df_hold_archive_windows)
    taken_count = int(snapshot["taken"])
    actual_closed = int(snapshot["actual_closed"])
    resolved_count = int(snapshot["resolved"])
    resolved_metrics_available = resolved_count > 0
    follow_value = _format_review_metric(
        float(snapshot["follow_through_rate"]),
        available=resolved_metrics_available,
        decimals=1,
        pct=True,
    )
    avg_dir_value = _format_review_metric(
        float(snapshot["avg_dir_return"]),
        available=resolved_metrics_available,
        decimals=2,
        pct=True,
        signed=True,
    )
    avg_mae_value = _format_review_metric(
        float(snapshot["avg_adverse_excursion"]),
        available=resolved_metrics_available,
        decimals=2,
        pct=True,
    )
    tone_follow = (
        POSITIVE
        if resolved_metrics_available and float(snapshot["follow_through_rate"]) >= 55.0
        else (WARNING if resolved_metrics_available and float(snapshot["follow_through_rate"]) >= 45.0 else TEXT_MUTED)
    )
    tone_dir = POSITIVE if resolved_metrics_available and float(snapshot["avg_dir_return"]) >= 0.0 else (NEGATIVE if resolved_metrics_available else TEXT_MUTED)
    tone_mae = NEGATIVE if resolved_metrics_available else TEXT_MUTED
    resolved_metrics_note = (
        "Current view, resolved signals finishing in their intended direction"
        if resolved_metrics_available
        else "No resolved signals in the current view yet"
    )
    avg_dir_note = (
        "Current view directional move after the signal horizon"
        if resolved_metrics_available
        else "Directional return appears after signals in this view resolve"
    )
    avg_mae_note = (
        "Average adverse excursion in the current view"
        if resolved_metrics_available
        else "Adverse excursion appears after signals in this view resolve"
    )
    st.markdown("### Overview")
    st.caption("Quick archive + execution read.")
    st.caption(_follow_through_horizon_note(timeframe_filter))
    render_kpi_grid(
        st,
        items=[
            {
                "label": "Follow-Through",
                "value": follow_value,
                "value_color": tone_follow,
                "subtext": resolved_metrics_note,
            },
            {
                "label": "Avg Dir Return",
                "value": avg_dir_value,
                "value_color": tone_dir,
                "subtext": avg_dir_note,
            },
            {
                "label": "Signals in View",
                "value": int(snapshot["total"]),
                "subtext": "Current scanner version archive window",
            },
            {
                "label": "Resolved in View",
                "value": int(snapshot["resolved"]),
                "subtext": f"Open in view: {int(snapshot['open'])}",
                "badge_text": _refresh_scope_badge(
                    symbol_filter=symbol_filter,
                    timeframe_filter=timeframe_filter,
                    resolved_now=resolved_now,
                ),
                "badge_color": POSITIVE if resolved_now > 0 else TEXT_MUTED,
                "badge_tone": "positive" if resolved_now > 0 else "neutral",
            },
        ],
        columns=4,
    )
    render_kpi_grid(
        st,
        items=[
            {
                "label": "Taken Setups",
                "value": taken_count,
                "value_color": POSITIVE if taken_count > 0 else TEXT_MUTED,
                "subtext": "Tracked signals you marked as Taken inside this view",
            },
            {
                "label": "Closed Journaled Trades",
                "value": actual_closed,
                "value_color": POSITIVE if actual_closed > 0 else TEXT_MUTED,
                "subtext": "Taken trades in this view with a recorded exit",
            },
            {
                "label": "Avg MAE",
                "value": avg_mae_value,
                "value_color": tone_mae,
                "subtext": avg_mae_note,
            },
            {
                "label": "Live Alerts",
                "value": active_alerts_count,
                "value_color": WARNING if active_alerts_count else TEXT_MUTED,
                "subtext": "Live market alerts active now, outside this view",
            },
        ],
        columns=4,
    )
    execution_vs_system_note, execution_vs_system_tone = _execution_vs_system_note(execution_snapshot)
    render_insight_card(
        st,
        title="Execution vs System",
        body_html=execution_vs_system_note,
        tone=execution_vs_system_tone,
    )
    if hold_guidance_enabled:
        st.markdown("### Hold Guidance")
        hold_scope = _hold_scope_label(symbol_filter=symbol_filter, timeframe_filter=timeframe_filter)
        slice_coverage_pct = float(hold_slice_progress.get("coverage_pct") or 0.0)
        slice_coverage_tone = POSITIVE if slice_coverage_pct >= 70.0 else (WARNING if slice_coverage_pct >= 35.0 else TEXT_MUTED)
        archive_coverage_pct = float(hold_archive_progress.get("coverage_pct") or 0.0)
        archive_coverage_tone = POSITIVE if archive_coverage_pct >= 70.0 else (WARNING if archive_coverage_pct >= 35.0 else TEXT_MUTED)
        same_hold_scope = _same_hold_progress(hold_slice_progress, hold_archive_progress)
        if same_hold_scope:
            st.caption(
                f"{hold_scope}: current view already spans the available coin history."
            )
            render_kpi_grid(
                st,
                items=[
                    {
                        "label": "Resolved",
                        "value": int(hold_archive_progress.get("resolved") or 0),
                        "value_color": POSITIVE if int(hold_archive_progress.get("resolved") or 0) > 0 else TEXT_MUTED,
                        "subtext": "Resolved signals in the current-version coin history for this scope",
                    },
                    {
                        "label": "Hold Ready",
                        "value": int(hold_archive_progress.get("ready") or 0),
                        "value_color": POSITIVE if int(hold_archive_progress.get("ready") or 0) > 0 else TEXT_MUTED,
                        "subtext": "Resolved signals already carrying hold checkpoints",
                    },
                    {
                        "label": "Missing",
                        "value": int(hold_archive_progress.get("missing") or 0),
                        "value_color": WARNING if int(hold_archive_progress.get("missing") or 0) > 0 else TEXT_MUTED,
                        "subtext": "Resolved signals still waiting for a hold-history fill pass",
                    },
                    {
                        "label": "Coverage",
                        "value": f"{archive_coverage_pct:.1f}%",
                        "value_color": archive_coverage_tone,
                        "subtext": "Checkpoint coverage for the current-version coin history in this scope",
                    },
                ],
                columns=4,
            )
        else:
            st.caption(
                f"{hold_scope}: guidance uses the current view; coverage uses broader coin history."
            )
            render_kpi_grid(
                st,
                items=[
                    {
                        "label": "View Resolved",
                        "value": int(hold_slice_progress.get("resolved") or 0),
                        "value_color": POSITIVE if int(hold_slice_progress.get("resolved") or 0) > 0 else TEXT_MUTED,
                        "subtext": "Resolved signals in the current view",
                    },
                    {
                        "label": "View Hold Ready",
                        "value": int(hold_slice_progress.get("ready") or 0),
                        "value_color": POSITIVE if int(hold_slice_progress.get("ready") or 0) > 0 else TEXT_MUTED,
                        "subtext": "Resolved signals in this view already carrying hold checkpoints",
                    },
                    {
                        "label": "View Missing",
                        "value": int(hold_slice_progress.get("missing") or 0),
                        "value_color": WARNING if int(hold_slice_progress.get("missing") or 0) > 0 else TEXT_MUTED,
                        "subtext": "Resolved signals in this view still missing checkpoints",
                    },
                    {
                        "label": "View Coverage",
                        "value": f"{slice_coverage_pct:.1f}%",
                        "value_color": slice_coverage_tone,
                        "subtext": "Checkpoint coverage inside the current view",
                    },
                ],
                columns=4,
            )
            render_kpi_grid(
                st,
                items=[
                    {
                        "label": "History Resolved",
                        "value": int(hold_archive_progress.get("resolved") or 0),
                        "value_color": POSITIVE if int(hold_archive_progress.get("resolved") or 0) > 0 else TEXT_MUTED,
                        "subtext": "Resolved signals in the broader current-version coin history",
                    },
                    {
                        "label": "History Hold Ready",
                        "value": int(hold_archive_progress.get("ready") or 0),
                        "value_color": POSITIVE if int(hold_archive_progress.get("ready") or 0) > 0 else TEXT_MUTED,
                        "subtext": "Resolved signals in that broader history already carrying hold checkpoints",
                    },
                    {
                        "label": "History Missing",
                        "value": int(hold_archive_progress.get("missing") or 0),
                        "value_color": WARNING if int(hold_archive_progress.get("missing") or 0) > 0 else TEXT_MUTED,
                        "subtext": "Resolved signals in that broader history still waiting for a fill pass",
                    },
                    {
                        "label": "History Coverage",
                        "value": f"{archive_coverage_pct:.1f}%",
                        "value_color": archive_coverage_tone,
                        "subtext": "Checkpoint coverage across the broader current-version coin history",
                    },
                ],
                columns=4,
            )
        if autofilled_hold_now > 0:
            st.caption(
                f"{hold_scope}: filled {autofilled_hold_now} hold-history row(s) in the background. {missing_hold_backfill} still pending."
            )
        elif missing_hold_backfill > 0:
            st.caption(
                f"{hold_scope}: hold history is still filling in the background. {missing_hold_backfill} resolved row(s) still pending."
            )
        else:
            st.caption(f"Hold history is up to date for {hold_scope}.")
        if str(timeframe_filter) == "All":
            st.caption(
                f"{symbol_filter}: timeframe breakdown."
            )
            hold_guidance_table = pd.DataFrame(
                [
                    {
                        "Timeframe": str(row.get("Timeframe") or ""),
                        "Upside Hold": str(row.get("Upside Hold") or "—"),
                        "Downside Hold": str(row.get("Downside Hold") or "—"),
                    }
                    for row in hold_guidance_rows
                ]
            )
            if hold_guidance_table.empty:
                st.caption("Hold guidance is still building for this coin in the current view.")
            else:
                st.dataframe(hold_guidance_table, hide_index=True, width="stretch")
        else:
            st.caption(
                f"{symbol_filter} {str(timeframe_filter).upper()}: upside and downside are learned separately."
            )
            single_scope = hold_guidance_rows[0] if hold_guidance_rows else {}
            upside_note, upside_tone = _hold_guidance_direction_note(
                single_scope.get("Upside Snapshot", {}),
                symbol=symbol_filter,
                timeframe=str(timeframe_filter),
                direction_label="Upside",
            )
            downside_note, downside_tone = _hold_guidance_direction_note(
                single_scope.get("Downside Snapshot", {}),
                symbol=symbol_filter,
                timeframe=str(timeframe_filter),
                direction_label="Downside",
            )
            hold_cols = st.columns(2, gap="medium")
            with hold_cols[0]:
                render_insight_card(
                    st,
                    title="Upside Hold",
                    body_html=upside_note,
                    tone=upside_tone,
                )
            with hold_cols[1]:
                render_insight_card(
                    st,
                    title="Downside Hold",
                    body_html=downside_note,
                    tone=downside_tone,
                )
    else:
        render_insight_card(
            st,
            title="Hold Guidance",
            body_html=(
                "Add a <b>coin</b> above to unlock hold guidance. "
                "Signal Archive only gives hold advice when the scope is tight enough to stay operational."
            ),
            tone="neutral",
        )

    adaptive_model = build_adaptive_context_model(df_events)
    learned_edges_df = build_learning_edge_table(adaptive_model, limit=12)

    if "lead_active" in df_events.columns:
        df_events["Lead Status"] = df_events["lead_active"].fillna(0).astype(int).map({1: "Lead", 0: "No Lead"})
    if "ai_aligned" in df_events.columns:
        df_events["AI Alignment"] = df_events["ai_aligned"].fillna(0).astype(int).map({1: "Aligned", 0: "Not aligned"})
    if "market_lead_label" in df_events.columns:
        df_events["Market Lead"] = df_events["market_lead_label"].replace("", "No Clear Lead").fillna("No Clear Lead")
    if "market_regime" in df_events.columns:
        df_events["Market Regime"] = df_events["market_regime"].replace("", "Unknown").fillna("Unknown")
    if "scan_focus" in df_events.columns:
        df_events["Scan Focus"] = df_events["scan_focus"].replace("", "Unknown").fillna("Unknown")
    if "setup_confirm" in df_events.columns:
        df_events["Setup Confirm"] = df_events.apply(
            lambda row: setup_confirm_display(
                str(row.get("setup_confirm") or ""),
                action_reason=str(row.get("action_reason") or ""),
                direction=str(row.get("direction") or ""),
            ),
            axis=1,
        )
    if "market_playbook_key" in df_events.columns or "market_playbook" in df_events.columns:
        playbook_keys = df_events.get("market_playbook_key", pd.Series(index=df_events.index, dtype=object))
        playbook_display_values = df_events.get("market_playbook", pd.Series(index=df_events.index, dtype=object))
        df_events["Playbook"] = pd.Series(playbook_keys, index=df_events.index).fillna("").astype(str).str.strip().map(
            lambda value: playbook_display(value) if value else ""
        )
        fallback_playbook = pd.Series(playbook_display_values, index=df_events.index).fillna("").astype(str).str.strip()
        df_events["Playbook"] = df_events["Playbook"].where(df_events["Playbook"].ne(""), fallback_playbook)
        df_events["Playbook"] = df_events["Playbook"].replace("", "Unknown").fillna("Unknown")
        df_events["Playbook Key"] = pd.Series(playbook_keys, index=df_events.index).fillna("").astype(str).str.strip()
        df_events["Playbook Key"] = df_events["Playbook Key"].where(
            df_events["Playbook Key"].ne(""),
            fallback_playbook.map(playbook_key),
        )
        df_events["Playbook Key"] = df_events["Playbook Key"].replace("", "Unknown").fillna("Unknown")
    if "market_trade_gate_key" in df_events.columns or "market_trade_gate" in df_events.columns:
        trade_gate_keys = df_events.get("market_trade_gate_key", pd.Series(index=df_events.index, dtype=object))
        trade_gate_display_values = df_events.get("market_trade_gate", pd.Series(index=df_events.index, dtype=object))
        df_events["Trade Gate"] = pd.Series(trade_gate_keys, index=df_events.index).fillna("").astype(str).str.strip().map(
            lambda value: trade_gate_display(value) if value else ""
        )
        fallback_trade_gate = pd.Series(trade_gate_display_values, index=df_events.index).fillna("").astype(str).str.strip()
        df_events["Trade Gate"] = df_events["Trade Gate"].where(df_events["Trade Gate"].ne(""), fallback_trade_gate)
        df_events["Trade Gate"] = df_events["Trade Gate"].replace("", "Unknown").fillna("Unknown")
        df_events["Trade Gate Key"] = pd.Series(trade_gate_keys, index=df_events.index).fillna("").astype(str).str.strip()
        df_events["Trade Gate Key"] = df_events["Trade Gate Key"].where(
            df_events["Trade Gate Key"].ne(""),
            fallback_trade_gate.map(trade_gate_key),
        )
        df_events["Trade Gate Key"] = df_events["Trade Gate Key"].replace("", "Unknown").fillna("Unknown")
    if "market_no_trade_reason" in df_events.columns:
        df_events[copy_text("review.label.no_trade_reason")] = (
            df_events["market_no_trade_reason"]
            .replace("", "None")
            .fillna("None")
            .astype(str)
            .str.replace("_", " ", regex=False)
            .str.title()
        )
    if "risk_tier" in df_events.columns:
        df_events["Risk Tier"] = df_events["risk_tier"].replace("", "Unknown").fillna("Unknown")
    if "sector_tag" in df_events.columns:
        df_events["Sector"] = df_events["sector_tag"].replace("", "Other").fillna("Other")
    if "market_sector_rotation" in df_events.columns:
        df_events["Sector Rotation"] = (
            df_events["market_sector_rotation"].replace("", "Unknown").fillna("Unknown")
        )
    if "market_catalyst_state" in df_events.columns:
        df_events["Catalyst State"] = (
            df_events["market_catalyst_state"].replace("", "Unknown").fillna("Unknown")
        )
    if "market_catalyst_event" in df_events.columns:
        df_events["Catalyst Event"] = (
            df_events["market_catalyst_event"].replace("", "None").fillna("None")
        )
    if "market_catalyst_category" in df_events.columns:
        df_events["Catalyst Category"] = (
            df_events["market_catalyst_category"].replace("", "Unknown").fillna("Unknown")
        )
    if "market_catalyst_scope" in df_events.columns:
        df_events["Catalyst Scope"] = (
            df_events["market_catalyst_scope"].replace("", "Unknown").fillna("Unknown")
        )
    if "market_catalyst_tag" in df_events.columns:
        df_events["Catalyst Tag"] = (
            df_events["market_catalyst_tag"].replace("", "None").fillna("None")
        )
    if "market_catalyst_targeted" in df_events.columns:
        df_events["Catalyst Targeting"] = (
            df_events["market_catalyst_targeted"].fillna(0).astype(int).map({1: "Targeted", 0: "Market-Wide"})
        )
    if "market_catalyst_window" in df_events.columns:
        df_events["Catalyst Window"] = (
            df_events["market_catalyst_window"].replace("", "Unknown").fillna("Unknown")
        )
    if "market_flow_state" in df_events.columns:
        df_events["Flow Read"] = (
            df_events["market_flow_state"].replace("", "Unknown").fillna("Unknown")
        )
    if "session_bucket" in df_events.columns:
        df_events["Session"] = df_events["session_bucket"].replace("", "Unknown").fillna("Unknown")
    elif "event_time" in df_events.columns:
        df_events["Session"] = pd.to_datetime(df_events["event_time"], utc=True, errors="coerce").map(
            lambda ts: session_bucket_for_timestamp(ts) if pd.notna(ts) else "Unknown"
        )
    if "Playbook" in df_events.columns and "Session" in df_events.columns:
        df_events["Playbook x Session"] = (
            df_events["Playbook"].astype(str).str.strip().replace("", "Unknown")
            + " | "
            + df_events["Session"].astype(str).str.strip().replace("", "Unknown")
        )
    if "Playbook" in df_events.columns and "Catalyst Window" in df_events.columns:
        df_events["Playbook x Catalyst Window"] = (
            df_events["Playbook"].astype(str).str.strip().replace("", "Unknown")
            + " | "
            + df_events["Catalyst Window"].astype(str).str.strip().replace("", "Unknown")
        )
    if "adaptive_edge_label" in df_events.columns:
        df_events["Learned Edge"] = (
            df_events["adaptive_edge_label"].replace("", "Unknown").fillna("Unknown")
        )
    if "archive_guardrail_label" in df_events.columns:
        df_events["History Guardrail"] = (
            df_events["archive_guardrail_label"].replace("", "Archive Clear").fillna("Archive Clear")
        )
    if "archive_guardrail_penalty" in df_events.columns:
        penalty_series = pd.to_numeric(df_events["archive_guardrail_penalty"], errors="coerce").fillna(0.0)
        df_events["History Guardrail Level"] = penalty_series.map(
            lambda value: (
                "Guardrail"
                if float(value) >= 5.0
                else ("Caution" if float(value) >= 3.0 else "Clear")
            )
        )
    if "Trade Gate" in df_events.columns:
        df_events["Execution Readiness"] = df_events.apply(
            lambda row: archived_execution_stance_label(
                trade_gate=str(row.get("Trade Gate") or ""),
                adaptive_edge=str(row.get("Learned Edge") or ""),
                archive_guardrail_severity=str(row.get("History Guardrail Level") or ""),
            ),
            axis=1,
        )
    if "trade_decision" in df_events.columns:
        df_events["Trade Decision"] = (
            df_events["trade_decision"].replace("", "Unmarked").fillna("Unmarked")
        )
    if "actual_trade_status" in df_events.columns:
        df_events["Actual Trade Status"] = (
            df_events["actual_trade_status"].replace("", "Unjournaled").fillna("Unjournaled")
        )
    if "actual_exit_reason" in df_events.columns:
        df_events["Actual Exit Reason"] = (
            df_events["actual_exit_reason"].replace("", "Open / Unset").fillna("Open / Unset")
        )
    df_events = annotate_alert_footprint(df_events)
    df_events = _annotate_actual_hold_style(df_events)
    df_events = _annotate_actual_exit_quality(df_events)

    session_summary_df = build_signal_cohort_summary(df_events, "Session") if "Session" in df_events.columns else pd.DataFrame()
    catalyst_window_summary_df = (
        build_signal_cohort_summary(df_events, "Catalyst Window") if "Catalyst Window" in df_events.columns else pd.DataFrame()
    )
    archive_guardrail_summary_df = (
        build_signal_cohort_summary(df_events, "History Guardrail") if "History Guardrail" in df_events.columns else pd.DataFrame()
    )
    execution_stance_summary_df = (
        build_signal_cohort_summary(df_events, "Execution Readiness") if "Execution Readiness" in df_events.columns else pd.DataFrame()
    )
    scan_focus_summary_df = (
        build_signal_cohort_summary(df_events, "Scan Focus") if "Scan Focus" in df_events.columns else pd.DataFrame()
    )
    hold_style_summary_df = build_signal_cohort_summary(df_events, "Hold Style") if "Hold Style" in df_events.columns else pd.DataFrame()
    exit_quality_summary_df = (
        build_signal_cohort_summary(df_events, "Exit Quality") if "Exit Quality" in df_events.columns else pd.DataFrame()
    )
    playbook_session_summary_df = (
        build_signal_cohort_summary(df_events, "Playbook x Session") if "Playbook x Session" in df_events.columns else pd.DataFrame()
    )
    playbook_catalyst_summary_df = (
        build_signal_cohort_summary(df_events, "Playbook x Catalyst Window")
        if "Playbook x Catalyst Window" in df_events.columns
        else pd.DataFrame()
    )
    primary_alert_summary_df = build_alert_effectiveness_summary(df_events, primary_only=True)
    works_cards: list[dict[str, str]] = []
    fail_cards: list[dict[str, str]] = []

    execution_work_cards, execution_fail_cards = _build_execution_signal_cards(execution_snapshot)
    works_cards.extend(execution_work_cards)
    fail_cards.extend(execution_fail_cards)
    if hold_guidance_enabled:
        hold_work_cards, hold_fail_cards = _build_hold_signal_cards(
            hold_guidance_rows,
            symbol_filter=symbol_filter,
        )
        works_cards.extend(hold_work_cards)
        fail_cards.extend(hold_fail_cards)

    qualified_session_execution_df = _qualified_summary_rows(
        session_summary_df,
        count_field="ClosedTradeCount",
        min_count=_MIN_EXECUTION_ARCHIVE_ROWS,
    )
    qualified_session_signal_df = _qualified_summary_rows(
        session_summary_df,
        count_field="Resolved",
        min_count=_MIN_SIGNAL_ARCHIVE_ROWS,
    )
    qualified_known_session_execution_df = _prefer_known_summary_rows(
        qualified_session_execution_df,
        label_field="Session",
    )
    qualified_known_session_signal_df = _prefer_known_summary_rows(
        qualified_session_signal_df,
        label_field="Session",
    )
    if not qualified_known_session_execution_df.empty:
        best_execution_row = qualified_known_session_execution_df.sort_values(
            ["ActualWinRatePct", "ClosedTradeCount", "Signals"], ascending=[False, False, False]
        ).iloc[0]
        works_cards.append(
            {
                "title": "Best Session",
                "body_html": (
                    f"<b>{best_execution_row['Session']}</b> is currently converting best in real execution "
                    f"({float(best_execution_row['ActualWinRatePct']):.1f}% across "
                    f"{int(best_execution_row['ClosedTradeCount'])} closed trades)."
                ),
                "tone": "positive" if float(best_execution_row["ActualWinRatePct"]) >= 55.0 else "neutral",
            }
        )
    elif not qualified_known_session_signal_df.empty:
        best_follow_row = qualified_known_session_signal_df.sort_values(
            ["FollowThroughPct", "Resolved", "Signals"], ascending=[False, False, False]
        ).iloc[0]
        works_cards.append(
            {
                "title": "Best Session",
                "body_html": (
                    f"Execution archive is still building. On the signal side, <b>{best_follow_row['Session']}</b> "
                    f"is currently the cleanest session ({float(best_follow_row['FollowThroughPct']):.1f}% follow-through "
                    f"across {int(best_follow_row['Resolved'])} resolved signals)."
                ),
                "tone": "neutral",
            }
        )
    else:
        works_cards.append(
            _archive_building_card(
                "Best Session",
                "Session archive is still building. We need more resolved signals or journaled trades before trusting timing rankings.",
            )
        )

    qualified_execution_stance_df = _qualified_summary_rows(
        execution_stance_summary_df,
        count_field="Resolved",
        min_count=_MIN_SIGNAL_ARCHIVE_ROWS,
    )
    if len(qualified_execution_stance_df) >= 2:
        strongest_execution_stance = qualified_execution_stance_df.sort_values(
            ["FollowThroughPct", "Resolved", "Signals"], ascending=[False, False, False]
        ).iloc[0]
        weakest_execution_stance = qualified_execution_stance_df.sort_values(
            ["FollowThroughPct", "Resolved", "Signals"], ascending=[True, False, False]
        ).iloc[0]
        works_cards.append(
            {
                "title": "Best Tradeability",
                "body_html": (
                    f"<b>{strongest_execution_stance['Execution Readiness']}</b> is the cleanest current tradeability state "
                    f"({float(strongest_execution_stance['FollowThroughPct']):.1f}% follow-through across "
                    f"{int(strongest_execution_stance['Resolved'])} resolved signals)."
                ),
                "tone": "positive"
                if trade_gate_key(strongest_execution_stance["Execution Readiness"]) == "TRADEABLE"
                else "neutral",
            }
        )
        fail_cards.append(
            {
                "title": "Weakest Tradeability",
                "body_html": (
                    f"<b>{weakest_execution_stance['Execution Readiness']}</b> is the weakest current tradeability state "
                    f"({float(weakest_execution_stance['FollowThroughPct']):.1f}% follow-through across "
                    f"{int(weakest_execution_stance['Resolved'])} resolved signals)."
                ),
                "tone": "warning",
            }
        )
    elif len(qualified_execution_stance_df) == 1:
        only_stance = qualified_execution_stance_df.iloc[0]
        works_cards.append(
            _archive_building_card(
                "Tradeability History",
                (
                    f"Only <b>{only_stance['Execution Readiness']}</b> has enough resolved history in this view so far "
                    f"({int(only_stance['Resolved'])} resolved signals). We need more tradeability variety before ranking strongest vs weakest."
                ),
            )
        )
    else:
        works_cards.append(
            _archive_building_card(
                "Tradeability History",
                "Tradeability history is still building. We need more resolved signals before trusting readiness rankings.",
            )
        )

    qualified_primary_alert_df = _qualified_summary_rows(
        primary_alert_summary_df,
        count_field="Resolved",
        min_count=_MIN_SIGNAL_ARCHIVE_ROWS,
    )
    if not qualified_primary_alert_df.empty:
        strongest_primary_alert = qualified_primary_alert_df.sort_values(
            ["FollowThroughPct", "Resolved", "Signals"], ascending=[False, False, False]
        ).iloc[0]
        works_cards.append(
            {
                "title": "Best Lead Alert",
                "body_html": (
                    f"<b>{strongest_primary_alert['Primary Alert']}</b> is converting best "
                    f"({float(strongest_primary_alert['FollowThroughPct']):.1f}% follow-through across "
                    f"{int(strongest_primary_alert['Resolved'])} resolved signals)."
                ),
                "tone": "positive" if float(strongest_primary_alert["FollowThroughPct"]) >= 55.0 else "neutral",
            }
        )
    else:
        works_cards.append(
            _archive_building_card(
                "Lead Alert History",
                "Lead-alert history is still building. We need more resolved signals before trusting alert rankings.",
            )
        )

    qualified_scan_focus_df = _qualified_summary_rows(
        scan_focus_summary_df,
        count_field="Resolved",
        min_count=_MIN_SIGNAL_ARCHIVE_ROWS,
    )
    qualified_known_scan_focus_df = _prefer_known_summary_rows(
        qualified_scan_focus_df,
        label_field="Scan Focus",
    )
    if len(qualified_known_scan_focus_df) >= 2:
        strongest_scan_focus = qualified_known_scan_focus_df.sort_values(
            ["FollowThroughPct", "Resolved", "Signals"], ascending=[False, False, False]
        ).iloc[0]
        weakest_scan_focus = qualified_known_scan_focus_df.sort_values(
            ["FollowThroughPct", "Resolved", "Signals"], ascending=[True, False, False]
        ).iloc[0]
        works_cards.append(
            {
                "title": "Best Scanner Focus",
                "body_html": (
                    f"<b>{strongest_scan_focus['Scan Focus']}</b> is converting best in this view "
                    f"({float(strongest_scan_focus['FollowThroughPct']):.1f}% follow-through across "
                    f"{int(strongest_scan_focus['Resolved'])} resolved signals)."
                ),
                "tone": "positive" if float(strongest_scan_focus["FollowThroughPct"]) >= 55.0 else "neutral",
            }
        )
        fail_cards.append(
            {
                "title": "Weakest Scanner Focus",
                "body_html": (
                    f"<b>{weakest_scan_focus['Scan Focus']}</b> is converting weakest in this view "
                    f"({float(weakest_scan_focus['FollowThroughPct']):.1f}% follow-through across "
                    f"{int(weakest_scan_focus['Resolved'])} resolved signals)."
                ),
                "tone": "warning" if float(weakest_scan_focus["FollowThroughPct"]) < 45.0 else "neutral",
            }
        )
    elif len(qualified_known_scan_focus_df) == 1:
        only_focus = qualified_known_scan_focus_df.iloc[0]
        works_cards.append(
            _archive_building_card(
                "Scanner Focus History",
                (
                    f"Only <b>{only_focus['Scan Focus']}</b> has enough resolved history in this view so far "
                    f"({int(only_focus['Resolved'])} resolved signals). We need both Broad Market and Breakout Radar "
                    "in the archive before comparing focus quality."
                ),
            )
        )
    else:
        works_cards.append(
            _archive_building_card(
                "Scanner Focus History",
                "Scanner-focus archive is still building. We need more resolved signals before comparing Broad Market vs Breakout Radar.",
            )
        )

    qualified_playbook_session_df = _qualified_summary_rows(
        playbook_session_summary_df,
        count_field="Resolved",
        min_count=_MIN_SIGNAL_ARCHIVE_ROWS,
    )
    qualified_known_playbook_session_df = _prefer_known_summary_rows(
        qualified_playbook_session_df,
        label_field="Playbook x Session",
    )
    if not qualified_known_playbook_session_df.empty:
        best_playbook_session = qualified_known_playbook_session_df.sort_values(
            ["FollowThroughPct", "Resolved", "Signals"], ascending=[False, False, False]
        ).iloc[0]
        works_cards.append(
            {
                "title": "Best Playbook Timing",
                "body_html": (
                    f"<b>{best_playbook_session['Playbook x Session']}</b> is the strongest timing combo "
                    f"({float(best_playbook_session['FollowThroughPct']):.1f}% follow-through across "
                    f"{int(best_playbook_session['Resolved'])} resolved signals)."
                ),
                "tone": "positive" if float(best_playbook_session["FollowThroughPct"]) >= 55.0 else "neutral",
            }
        )
    else:
        works_cards.append(
            _archive_building_card(
                "Playbook Timing",
                "Playbook timing archive is still building. We need more resolved signals before trusting timing combos.",
            )
        )

    qualified_hold_style_df = _qualified_summary_rows(
        hold_style_summary_df,
        count_field="ClosedTradeCount",
        min_count=_MIN_EXECUTION_ARCHIVE_ROWS,
    )
    if not qualified_hold_style_df.empty:
        best_hold_style = qualified_hold_style_df.sort_values(
            ["ActualWinRatePct", "ClosedTradeCount", "Signals"], ascending=[False, False, False]
        ).iloc[0]
        works_cards.append(
            {
                "title": "Best Hold Profile",
                "body_html": (
                    f"<b>{best_hold_style['Hold Style']}</b> is the healthiest hold profile so far "
                    f"({float(best_hold_style['ActualWinRatePct']):.1f}% closed-trade win rate across "
                    f"{int(best_hold_style['ClosedTradeCount'])} journaled trades)."
                ),
                "tone": "positive" if float(best_hold_style["ActualWinRatePct"]) >= 55.0 else "neutral",
            }
        )
    else:
        works_cards.append(
            _archive_building_card(
                "Hold Profile History",
                "Hold-profile coaching is still building. We need more journaled closed trades before trusting this read.",
            )
        )

    qualified_catalyst_window_df = _qualified_summary_rows(
        catalyst_window_summary_df,
        count_field="Resolved",
        min_count=_MIN_SIGNAL_ARCHIVE_ROWS,
    )
    qualified_known_catalyst_window_df = _prefer_known_summary_rows(
        qualified_catalyst_window_df,
        label_field="Catalyst Window",
    )
    if len(qualified_known_catalyst_window_df) >= 2:
        weakest_catalyst_row = qualified_known_catalyst_window_df.sort_values(
            ["FollowThroughPct", "Resolved", "Signals"], ascending=[True, False, False]
        ).iloc[0]
        fail_cards.append(
            {
                "title": "Weakest Event Window",
                "body_html": (
                    f"<b>{weakest_catalyst_row['Catalyst Window']}</b> is the weakest event window "
                    f"({float(weakest_catalyst_row['FollowThroughPct']):.1f}% follow-through across "
                    f"{int(weakest_catalyst_row['Resolved'])} resolved signals)."
                ),
                "tone": "warning" if float(weakest_catalyst_row["FollowThroughPct"]) < 45.0 else "neutral",
            }
        )
    else:
        fail_cards.append(
            _archive_building_card(
                "Event Window History",
                "Event-window archive is still too thin or too one-sided to trust weakest-window rankings.",
            )
        )

    qualified_guardrail_df = _qualified_summary_rows(
        archive_guardrail_summary_df,
        count_field="Resolved",
        min_count=_MIN_SIGNAL_ARCHIVE_ROWS,
    )
    qualified_known_guardrail_df = _prefer_known_summary_rows(
        qualified_guardrail_df,
        label_field="History Guardrail",
    )
    if len(qualified_known_guardrail_df) >= 2:
        strongest_guardrail_row = qualified_known_guardrail_df.sort_values(
            ["FollowThroughPct", "Resolved", "Signals"], ascending=[True, False, False]
        ).iloc[0]
        fail_cards.append(
            {
                "title": "Weakest Guardrail Cluster",
                "body_html": (
                    f"<b>{strongest_guardrail_row['History Guardrail']}</b> is the weakest matched history cluster "
                    f"({float(strongest_guardrail_row['FollowThroughPct']):.1f}% across "
                    f"{int(strongest_guardrail_row['Resolved'])} resolved signals)."
                ),
                "tone": "warning",
            }
        )
    else:
        fail_cards.append(
            _archive_building_card(
                "History Guardrail",
                "History-guardrail data is still too thin to rank weak clusters confidently.",
            )
        )

    qualified_playbook_catalyst_df = _qualified_summary_rows(
        playbook_catalyst_summary_df,
        count_field="Resolved",
        min_count=_MIN_SIGNAL_ARCHIVE_ROWS,
    )
    qualified_known_playbook_catalyst_df = _prefer_known_summary_rows(
        qualified_playbook_catalyst_df,
        label_field="Playbook x Catalyst Window",
    )
    if len(qualified_known_playbook_catalyst_df) >= 2:
        weakest_playbook_catalyst = qualified_known_playbook_catalyst_df.sort_values(
            ["FollowThroughPct", "Resolved", "Signals"], ascending=[True, False, False]
        ).iloc[0]
        fail_cards.append(
            {
                "title": "Weakest Playbook/Event Fit",
                "body_html": (
                    f"<b>{weakest_playbook_catalyst['Playbook x Catalyst Window']}</b> is the weakest combo "
                    f"({float(weakest_playbook_catalyst['FollowThroughPct']):.1f}% follow-through across "
                    f"{int(weakest_playbook_catalyst['Resolved'])} resolved signals)."
                ),
                "tone": "warning" if float(weakest_playbook_catalyst["FollowThroughPct"]) < 45.0 else "neutral",
            }
        )
    else:
        fail_cards.append(
            _archive_building_card(
                "Playbook/Event History",
                "Playbook/event archive is still too thin to rank weak combinations confidently.",
            )
        )

    qualified_exit_quality_df = _qualified_summary_rows(
        exit_quality_summary_df,
        count_field="ClosedTradeCount",
        min_count=_MIN_EXECUTION_ARCHIVE_ROWS,
    )
    if len(qualified_exit_quality_df) >= 2:
        weakest_exit_quality = qualified_exit_quality_df.sort_values(
            ["ActualWinRatePct", "ClosedTradeCount", "Signals"], ascending=[True, False, False]
        ).iloc[0]
        fail_cards.append(
            {
                "title": "Weakest Exit Discipline",
                "body_html": (
                    f"<b>{weakest_exit_quality['Exit Quality']}</b> is the weakest realized exit pattern "
                    f"({float(weakest_exit_quality['ActualWinRatePct']):.1f}% across "
                    f"{int(weakest_exit_quality['ClosedTradeCount'])} closed trades)."
                ),
                "tone": "warning",
            }
        )
    else:
        fail_cards.append(
            _archive_building_card(
                "Exit Discipline History",
                "Exit-discipline archive is still building. We need more journaled closed trades before trusting weakest-exit rankings.",
            )
        )

    works_cards = _prepare_section_cards(works_cards, max_actionable=3)
    fail_cards = _prepare_section_cards(fail_cards, max_actionable=3)

    st.markdown("### What Works")
    st.caption("Current trust list.")
    _render_insight_card_grid(st, works_cards, columns=3)

    st.markdown("### What Needs Care")
    st.caption("Current caution list.")
    _render_insight_card_grid(st, fail_cards, columns=3)

    _render_execution_review_section(
        st=st,
        df_events=df_events,
        execution_snapshot=execution_snapshot,
        db_path=db_path,
        save_signal_trade_overlay=save_signal_trade_overlay,
        save_signal_trade_journal=save_signal_trade_journal,
        positive_color=POSITIVE,
        warning_color=WARNING,
        negative_color=NEGATIVE,
        muted_color=TEXT_MUTED,
    )

    st.markdown("### Deep Dives")
    st.caption("Optional detail.")
    with st.expander("Setup & Market", expanded=False):
        _render_compact_cohort_tables(
            st,
            df_events=df_events,
            build_signal_cohort_summary=build_signal_cohort_summary,
            specs=[
                ("Setup Confirm", "By Setup Confirm"),
                ("direction", "By Direction", "Direction"),
                ("timeframe", "By Timeframe", "Timeframe"),
                ("Session", "By Session"),
                ("Market Regime", "By Market Regime"),
                ("Scan Focus", "By Scanner Focus", "Scanner Focus"),
                ("Playbook", "By Playbook"),
                ("AI Alignment", "By AI Alignment"),
            ],
        )

    with st.expander("Context & Catalysts", expanded=False):
        _render_compact_cohort_tables(
            st,
            df_events=df_events,
            build_signal_cohort_summary=build_signal_cohort_summary,
            specs=[
                ("Primary Alert", "By Lead Alert"),
                ("Catalyst State", "By Catalyst State"),
                ("Catalyst Window", "By Catalyst Window"),
                ("Flow Read", "By Tape Read", "Tape Read"),
                ("Sector Rotation", "By Sector Rotation"),
                ("Sector", "By Sector"),
            ],
        )

    with st.expander("Execution & Journal", expanded=False):
        _render_compact_cohort_tables(
            st,
            df_events=df_events,
            build_signal_cohort_summary=build_signal_cohort_summary,
            specs=[
                ("Execution Readiness", "By Tradeability", "Tradeability"),
                ("Risk Tier", "By Risk Tier"),
                ("Trade Decision", "By Action Taken", "Action Taken"),
                ("Actual Trade Status", "By Journal Status", "Journal Status"),
                (copy_text("review.label.no_trade_reason"), "By Skip Reason", "Skip Reason"),
                ("Hold Style", "By Hold Style"),
                ("Exit Quality", "By Exit Quality"),
                ("Actual Exit Reason", "By Exit Reason", "Exit Reason"),
            ],
        )

    if hold_guidance_enabled:
        with st.expander("Hold Guidance Detail", expanded=False):
            st.caption(
                "These stay coin-specific. They use resolved signals in the current view and read sparse forward checkpoints to suggest a historical hold window."
            )
            _render_hold_window_cohort_tables(
                st,
                df_events=df_events,
                df_forward_windows=df_forward_windows,
                build_hold_window_cohort_summary=build_hold_window_cohort_summary,
                specs=[
                    ("Setup Confirm", "By Setup Confirm"),
                    ("timeframe", "By Timeframe"),
                    ("direction", "By Direction"),
                    ("Playbook", "By Playbook"),
                ],
            )

    recent_cols = [
        "event_time",
        "symbol",
        "timeframe",
        "Primary Alert",
        "Alert Footprint",
        "Scan Focus",
        "Setup Confirm",
        "direction",
        "Lead Status",
        "Market Lead",
        "Sector",
        "Sector Rotation",
        "Catalyst State",
        "Catalyst Window",
        "Catalyst Scope",
        "Catalyst Targeting",
        "Catalyst Category",
        "Catalyst Tag",
        "Catalyst Event",
        "Flow Read",
        "Session",
        "Learned Edge",
        "Execution Readiness",
        "Market Regime",
        "Trade Gate",
        "Trade Decision",
        "Actual Trade Status",
        "Hold Style",
        "Actual Hold Hours",
        "Exit Quality",
        "Actual Exit Reason",
        "Risk Tier",
        copy_text("review.label.no_trade_reason"),
        "Playbook",
        "Playbook x Session",
        "Playbook x Catalyst Window",
        "adaptive_edge_score",
        "actionable_frame_score",
        "actionable_setup_score",
        "actionable_context_score",
        "actionable_tactical_score",
        "trade_note",
        "actual_trade_side",
        "actual_entry_price",
        "actual_entry_at",
        "actual_exit_price",
        "actual_exit_at",
        "actual_pnl_pct",
        "confidence",
        "ai_confidence",
        "plan_outcome",
        "directional_return_pct",
        "favorable_excursion_pct",
        "adverse_excursion_pct",
        "status",
    ]
    recent_df = df_events[[c for c in recent_cols if c in df_events.columns]].copy()
    if "actual_trade_side" in recent_df.columns:
        recent_df["actual_trade_side"] = recent_df["actual_trade_side"].map(_display_trade_direction).replace("", "—")
    if "event_time" in recent_df.columns:
        recent_df["event_time"] = pd.to_datetime(recent_df["event_time"], errors="coerce").dt.strftime("%Y-%m-%d %H:%M")
    rename_map = {
        "event_time": "Signal Time",
        "symbol": "Coin",
        "timeframe": "TF",
        "Primary Alert": "Lead Alert",
        "Scan Focus": "Scanner Focus",
        "Lead Status": "Lead Signal",
        "Market Lead": "AI View",
        "Flow Read": "Tape Read",
        "Learned Edge": "Archive Read",
        "Execution Readiness": "Tradeability",
        "direction": "Direction",
        "confidence": "Confidence",
        "ai_confidence": "AI Confidence",
        "plan_outcome": "Plan Outcome",
        "directional_return_pct": "Dir Return %",
        "favorable_excursion_pct": "MFE %",
        "adverse_excursion_pct": "MAE %",
        "status": "Status",
        "adaptive_edge_score": "Archive Score",
        "actionable_frame_score": "Hunt Score",
        "actionable_setup_score": "Setup Score",
        "actionable_context_score": "Context Score",
        "actionable_tactical_score": "Tactical Score",
        "trade_note": "Trade Note",
        "actual_trade_side": "Trade Direction",
        "actual_entry_price": "Actual Entry",
        "actual_entry_at": "Entry Time",
        "actual_exit_price": "Actual Exit",
        "actual_exit_at": "Exit Time",
        "actual_pnl_pct": "Actual PnL %",
    }
    recent_df = recent_df.rename(columns=rename_map)
    with st.expander("Data Tables", expanded=False):
        st.markdown("#### Recent Signals")
        st.dataframe(recent_df.round(2), hide_index=True, width="stretch")

        if not df_alerts.empty:
            alerts_df = df_alerts.copy()
            if "last_seen_at" in alerts_df.columns:
                alerts_df["last_seen_at"] = pd.to_datetime(alerts_df["last_seen_at"], errors="coerce").dt.strftime("%Y-%m-%d %H:%M")
            if "first_seen_at" in alerts_df.columns:
                alerts_df["first_seen_at"] = pd.to_datetime(alerts_df["first_seen_at"], errors="coerce").dt.strftime("%Y-%m-%d %H:%M")
            if "active" in alerts_df.columns:
                alerts_df["Active"] = alerts_df["active"].fillna(0).astype(int).map({1: "Yes", 0: "No"})
            alerts_df = alerts_df.rename(
                columns={
                    "severity": "Severity",
                    "title": "Title",
                    "note": "Note",
                    "last_seen_at": "Last Seen",
                    "first_seen_at": "First Seen",
                    "times_seen": "Times Seen",
                    "alert_key": "Alert Key",
                }
            )
            st.markdown("#### Recent Market Alerts")
            st.dataframe(
                alerts_df[
                    [
                        c
                        for c in ["Last Seen", "Active", "Severity", "Title", "Note", "Times Seen", "Alert Key"]
                        if c in alerts_df.columns
                    ]
                ],
                hide_index=True,
                width="stretch",
            )

        if not learned_edges_df.empty:
            st.markdown("#### Archive Learning")
            st.dataframe(learned_edges_df.round(2), hide_index=True, width="stretch")

    _render_tracker_backup_restore(
        st=st,
        db_path=db_path,
        storage_snapshot=storage_snapshot,
        read_signal_tracker_db_bytes=read_signal_tracker_db_bytes,
        backup_signal_tracker_db=backup_signal_tracker_db,
        restore_signal_tracker_db_bytes=restore_signal_tracker_db_bytes,
        fetch_signal_events_df=fetch_signal_events_df,
        fetch_market_alerts_df=fetch_market_alerts_df,
    )
