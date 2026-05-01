from __future__ import annotations

from collections.abc import Mapping
import html
import time

import pandas as pd
from core.archive_decision import build_archive_decision_snapshot
from core.archive_intelligence import (
    archive_direction_key,
    archive_setup_class_key,
    archive_setup_class_label,
    build_archive_intelligence_snapshot,
    filter_archive_events_by_setup,
)
from core.archive_expected_path import (
    build_archive_expected_path_projection,
    projection_float,
    with_expected_path_reference_price,
)
from core.archive_policy import ARCHIVE_LEARNING_WINDOW_ROWS
from core.decision_version import current_decision_version
from core.session_utils import session_bucket_for_timestamp
from core.trading_copy import copy_text, playbook_display, playbook_key, trade_gate_display, trade_gate_key

from ui.ctx import get_ctx
from ui.signal_formatters import archived_execution_stance_label, setup_confirm_display
from ui.primitives import render_insight_card, render_kpi_grid, render_page_header


_BEST_SIGNAL_MIN_RESOLVED = 12
_BEST_SIGNAL_MIN_TIMEFRAMES = 2
_BEST_SIGNAL_MIN_TOTAL_RESOLVED = 24
_BEST_SIGNAL_LEADERBOARD_LIMIT = 10
_MIN_SIGNAL_ARCHIVE_ROWS = _BEST_SIGNAL_MIN_RESOLVED
_MIN_SETUP_POCKET_ROWS = 8

_SETUP_FILTER_OPTIONS: tuple[tuple[str, str], ...] = (
    ("Auto Best", "AUTO_BEST"),
    ("ENTER T+AI", "ENTER_TREND_AI"),
    ("ENTER Trend", "ENTER_TREND_LED"),
    ("ENTER AI", "ENTER_AI_LED"),
    ("EARLY", "PROBE"),
    ("WATCH", "WATCH"),
)

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


def _best_signal_quality_score(
    *,
    follow_through_pct: float,
    avg_dir_return_pct: float,
    resolved: int,
    resolved_baseline: int,
    qualified_timeframes: int = 1,
    timeframe_baseline: int = 1,
) -> float:
    resolved_baseline = max(1, int(resolved_baseline))
    timeframe_baseline = max(1, int(timeframe_baseline))
    sample_ratio = min(1.0, max(0.0, float(resolved) / float(resolved_baseline * 2)))
    timeframe_ratio = min(1.0, max(0.0, float(qualified_timeframes) / float(timeframe_baseline)))
    return (
        float(follow_through_pct)
        + (float(avg_dir_return_pct) * 5.0)
        + (sample_ratio * 6.0)
        + (timeframe_ratio * 6.0)
    )


def _display_trade_direction(value: object) -> str:
    side = str(value or "").strip().upper()
    if side in {"LONG", "UPSIDE", "BUY"}:
        return "Upside"
    if side in {"SHORT", "DOWNSIDE", "SELL"}:
        return "Downside"
    return ""


def _normalize_symbol_filter(value: object) -> str:
    text = str(value or "").strip().upper()
    if not text:
        return ""
    for separator in ("/", "-", " "):
        if separator in text:
            text = text.split(separator, 1)[0].strip()
    return text


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
        return f"Timing window: {tf_key.upper()} signals are measured after {_FOLLOW_THROUGH_HORIZONS[tf_key]} candles."
    return (
        "Timing windows: "
        "5m = 12 bars, 15m = 16 bars, 1h = 12 bars, 4h = 12 bars, 1d = 10 bars."
    )


def _coin_view_summary(
    *,
    symbol_filter: str,
    timeframe_filter: str,
    analysis_limit: int,
    rows_loaded: int,
) -> str:
    scope_parts = [str(symbol_filter or "").strip().upper()]
    if str(timeframe_filter or "").strip() and str(timeframe_filter) != "All":
        scope_parts.append(str(timeframe_filter).upper())
    return (
        f"<b>{' • '.join(scope_parts)}</b><br>"
        f"{int(rows_loaded)} signals loaded from the latest {int(analysis_limit)}<br>"
        "Shows timing, expected path, and the strongest historical setup pocket"
    )


def _archive_direction_key(value: object) -> str:
    return archive_direction_key(value)


def _setup_filter_value(label: str) -> str:
    option_map = {display: value for display, value in _SETUP_FILTER_OPTIONS}
    return option_map.get(str(label or "").strip(), "AUTO_BEST")


def _setup_class_key(value: object) -> str:
    return archive_setup_class_key(value)


def _setup_class_label(value: object) -> str:
    return archive_setup_class_label(value)


def _setup_display_label(value: object, direction: object) -> str:
    key = _setup_class_key(value)
    if key == "UNKNOWN":
        return _setup_class_label(value)
    return setup_confirm_display(key, audience="trader", direction=str(direction or ""))


def _annotate_setup_class(df_events: pd.DataFrame) -> pd.DataFrame:
    if df_events is None or df_events.empty:
        return pd.DataFrame()
    out = df_events.copy()
    setup_series = out.get("setup_confirm", pd.Series(index=out.index, dtype=object))
    out["__setup_class"] = setup_series.fillna("").astype(str).map(_setup_class_key)
    out["__setup_label"] = out["__setup_class"].map(_setup_class_label)
    return out


def _filter_events_by_setup(df_events: pd.DataFrame, setup_filter_value: str) -> pd.DataFrame:
    return filter_archive_events_by_setup(df_events, setup_filter_value)


def _select_setup_pocket(
    df_events: pd.DataFrame,
    *,
    setup_filter_value: str,
    min_completed: int = _MIN_SETUP_POCKET_ROWS,
) -> dict[str, object]:
    if df_events is None or df_events.empty:
        return _setup_pocket_from_snapshot(None)
    snapshot = build_archive_intelligence_snapshot(
        df_events,
        setup_filter_value=setup_filter_value,
        min_completed=min_completed,
    )
    return _setup_pocket_from_snapshot(snapshot)


def _setup_pocket_from_snapshot(snapshot: object | None) -> dict[str, object]:
    empty = {
        "available": False,
        "setup_class": "",
        "setup_label": "Building",
        "setup_display": "Building",
        "timeframe": "",
        "direction": "",
        "signals": 0,
        "completed": 0,
        "follow_through_pct": 0.0,
        "avg_dir_return_pct": 0.0,
        "avg_adverse_excursion_pct": 0.0,
        "score": 0.0,
    }
    if snapshot is None or not bool(getattr(snapshot, "available", False)):
        return empty
    return {
        "available": True,
        "setup_class": str(snapshot.setup_class).strip().upper(),
        "setup_label": str(snapshot.setup_label).strip() or "Other",
        "setup_display": _setup_display_label(snapshot.setup_class, snapshot.direction),
        "timeframe": str(snapshot.timeframe).strip().lower(),
        "direction": str(snapshot.direction).strip().upper(),
        "signals": int(snapshot.signals or 0),
        "completed": int(snapshot.completed or 0),
        "follow_through_pct": float(snapshot.follow_through_pct or 0.0),
        "avg_dir_return_pct": float(snapshot.avg_dir_return_pct or 0.0),
        "avg_adverse_excursion_pct": float(snapshot.avg_adverse_excursion_pct or 0.0),
        "score": float(snapshot.score or 0.0),
    }


def _filter_events_to_setup_pocket(df_events: pd.DataFrame, pocket: Mapping[str, object]) -> pd.DataFrame:
    if df_events is None or df_events.empty or not bool(pocket.get("available")):
        return pd.DataFrame()
    d = _annotate_setup_class(df_events)
    d["timeframe"] = d["timeframe"].fillna("").astype(str).str.strip().str.lower()
    d["__direction_key"] = d["direction"].map(_archive_direction_key)
    return d[
        d["timeframe"].eq(str(pocket.get("timeframe") or "").strip().lower())
        & d["__setup_class"].eq(str(pocket.get("setup_class") or "").strip().upper())
        & d["__direction_key"].eq(str(pocket.get("direction") or "").strip().upper())
    ].copy()


def _filter_events_to_setup_direction(df_events: pd.DataFrame, pocket: Mapping[str, object]) -> pd.DataFrame:
    if df_events is None or df_events.empty or not bool(pocket.get("available")):
        return pd.DataFrame()
    d = _annotate_setup_class(df_events)
    d["__direction_key"] = d["direction"].map(_archive_direction_key)
    return d[
        d["__setup_class"].eq(str(pocket.get("setup_class") or "").strip().upper())
        & d["__direction_key"].eq(str(pocket.get("direction") or "").strip().upper())
    ].copy()


def _setup_pocket_label(pocket: Mapping[str, object]) -> str:
    if not bool(pocket.get("available")):
        return "Building"
    setup = str(pocket.get("setup_display") or pocket.get("setup_label") or "Setup").strip()
    timeframe = str(pocket.get("timeframe") or "").strip().upper()
    parts = [part for part in (setup, timeframe) if part]
    return " • ".join(parts)


def _filter_directional_signal_rows(df_events: pd.DataFrame) -> pd.DataFrame:
    if df_events is None or df_events.empty or "direction" not in df_events.columns:
        return pd.DataFrame()
    d = df_events.copy()
    d["__direction_key"] = d["direction"].map(_archive_direction_key)
    d = d[d["__direction_key"].isin({"UPSIDE", "DOWNSIDE"})].copy()
    return d


def _merge_path_sample_counts(
    grouped: pd.DataFrame,
    df_events: pd.DataFrame,
    df_forward_windows: pd.DataFrame | None,
) -> tuple[pd.DataFrame, bool]:
    if df_forward_windows is None:
        return grouped.copy(), False
    out = grouped.copy()
    out["PathSamples"] = 0
    required_group_cols = {"symbol", "timeframe", "__direction_key"}
    if (
        out.empty
        or df_events is None
        or df_events.empty
        or df_forward_windows.empty
        or not required_group_cols.issubset(df_events.columns)
        or "signal_key" not in df_events.columns
        or "signal_key" not in df_forward_windows.columns
    ):
        return out, True
    window_keys = set(df_forward_windows["signal_key"].fillna("").astype(str).str.strip())
    window_keys.discard("")
    if not window_keys:
        return out, True
    event_scope = df_events.copy()
    event_scope["signal_key"] = event_scope["signal_key"].fillna("").astype(str).str.strip()
    event_scope = event_scope[event_scope["signal_key"].isin(window_keys)].copy()
    if event_scope.empty:
        return out, True
    group_cols = ["symbol", "timeframe", "__direction_key"]
    if "__setup_class" in out.columns and "__setup_class" in event_scope.columns:
        group_cols.append("__setup_class")
    path_counts = (
        event_scope.groupby(group_cols, dropna=False)["signal_key"]
        .nunique()
        .reset_index(name="PathSamples")
    )
    out = out.drop(columns=["PathSamples"], errors="ignore").merge(
        path_counts,
        on=group_cols,
        how="left",
    )
    out["PathSamples"] = pd.to_numeric(out["PathSamples"], errors="coerce").fillna(0).astype(int)
    return out, True


def _select_best_signal_coin(
    *,
    df_events: pd.DataFrame,
    df_forward_windows: pd.DataFrame | None = None,
    timeframe_filter: str,
    min_resolved: int = _BEST_SIGNAL_MIN_RESOLVED,
    min_path_samples: int = 8,
    min_timeframes: int = _BEST_SIGNAL_MIN_TIMEFRAMES,
    min_total_resolved: int = _BEST_SIGNAL_MIN_TOTAL_RESOLVED,
) -> dict[str, object]:
    empty = {
        "available": False,
        "symbol": "",
        "mode": "empty",
        "qualified_timeframes": 0,
        "follow_through_pct": 0.0,
        "avg_dir_return_pct": 0.0,
        "resolved": 0,
        "best_timeframe": "",
    }
    if df_events is None or df_events.empty:
        return empty
    required = {"symbol", "timeframe", "status"}
    if not required.issubset(df_events.columns):
        return empty
    d = df_events.copy()
    d["status"] = d["status"].fillna("").astype(str).str.upper()
    d = d[d["status"].eq("RESOLVED")].copy()
    if d.empty:
        return empty
    d = _filter_directional_signal_rows(d)
    if d.empty:
        return empty
    d["symbol"] = d["symbol"].fillna("").astype(str).str.strip().str.upper()
    d["timeframe"] = d["timeframe"].fillna("").astype(str).str.strip().str.lower()
    d = d[d["symbol"].ne("") & d["timeframe"].ne("")].copy()
    if d.empty:
        return empty
    if "signal_key" in d.columns:
        d["signal_key"] = d["signal_key"].fillna("").astype(str).str.strip()
    d["directional_return_pct"] = pd.to_numeric(d.get("directional_return_pct"), errors="coerce")
    grouped = (
        d.groupby(["symbol", "timeframe", "__direction_key"], dropna=False)
        .agg(
            Resolved=("symbol", "count"),
            FollowThroughPct=(
                "directional_return_pct",
                lambda s: float((pd.to_numeric(s, errors="coerce") > 0).mean() * 100.0) if len(pd.Series(s).dropna()) else 0.0,
            ),
            AvgDirReturnPct=("directional_return_pct", "mean"),
        )
        .reset_index()
    )
    grouped["QualityScore"] = grouped.apply(
        lambda row: _best_signal_quality_score(
            follow_through_pct=float(row.get("FollowThroughPct") or 0.0),
            avg_dir_return_pct=float(row.get("AvgDirReturnPct") or 0.0),
            resolved=int(row.get("Resolved") or 0),
            resolved_baseline=min_resolved,
        ),
        axis=1,
    )
    grouped, path_filter_active = _merge_path_sample_counts(grouped, d, df_forward_windows)
    if not path_filter_active and "PathSamples" not in grouped.columns:
        grouped["PathSamples"] = grouped["Resolved"]
    qualified = grouped[grouped["Resolved"] >= int(max(1, min_resolved))].copy()
    if path_filter_active:
        qualified = qualified[qualified["PathSamples"] >= int(max(1, min_path_samples))].copy()
    if qualified.empty:
        return empty

    timeframe_text = str(timeframe_filter or "").strip().lower()
    if timeframe_text and timeframe_text != "all":
        qualified = qualified[qualified["timeframe"].eq(timeframe_text)].copy()
        if qualified.empty:
            return empty
        best = qualified.sort_values(
            ["QualityScore", "FollowThroughPct", "AvgDirReturnPct", "PathSamples", "Resolved"],
            ascending=[False, False, False, False, False],
        ).iloc[0]
        return {
            "available": True,
            "symbol": str(best["symbol"]).strip().upper(),
            "mode": "timeframe",
            "qualified_timeframes": 1,
            "follow_through_pct": float(best["FollowThroughPct"] or 0.0),
            "avg_dir_return_pct": float(best["AvgDirReturnPct"] or 0.0),
            "resolved": int(best["Resolved"] or 0),
            "best_timeframe": str(best["timeframe"]).strip().lower(),
        }

    symbol_scores = (
        qualified.groupby("symbol", dropna=False)
        .agg(
            QualifiedTimeframes=("timeframe", "nunique"),
            AvgFollowThroughPct=("FollowThroughPct", "mean"),
            AvgDirReturnPct=("AvgDirReturnPct", "mean"),
            Resolved=("Resolved", "sum"),
            PathSamples=("PathSamples", "sum"),
        )
        .reset_index()
    )
    symbol_scores["QualityScore"] = symbol_scores.apply(
        lambda row: _best_signal_quality_score(
            follow_through_pct=float(row.get("AvgFollowThroughPct") or 0.0),
            avg_dir_return_pct=float(row.get("AvgDirReturnPct") or 0.0),
            resolved=int(row.get("Resolved") or 0),
            resolved_baseline=min_total_resolved,
            qualified_timeframes=int(row.get("QualifiedTimeframes") or 0),
            timeframe_baseline=min_timeframes,
        ),
        axis=1,
    )
    preferred = symbol_scores[
        (symbol_scores["QualifiedTimeframes"] >= int(max(1, min_timeframes)))
        & (symbol_scores["Resolved"] >= int(max(1, min_total_resolved)))
    ].copy()
    mode = "cross_timeframe"
    if preferred.empty:
        preferred = symbol_scores.copy()
        mode = "best_available"
    if preferred.empty:
        return empty
    best_symbol_row = preferred.sort_values(
        ["QualityScore", "AvgFollowThroughPct", "AvgDirReturnPct", "QualifiedTimeframes", "PathSamples", "Resolved"],
        ascending=[False, False, False, False, False, False],
    ).iloc[0]
    best_symbol = str(best_symbol_row["symbol"]).strip().upper()
    best_timeframe_row = qualified[qualified["symbol"].eq(best_symbol)].sort_values(
        ["QualityScore", "FollowThroughPct", "AvgDirReturnPct", "PathSamples", "Resolved"],
        ascending=[False, False, False, False, False],
    ).iloc[0]
    return {
        "available": True,
        "symbol": best_symbol,
        "mode": mode,
        "qualified_timeframes": int(best_symbol_row["QualifiedTimeframes"] or 0),
        "follow_through_pct": float(best_symbol_row["AvgFollowThroughPct"] or 0.0),
        "avg_dir_return_pct": float(best_symbol_row["AvgDirReturnPct"] or 0.0),
        "resolved": int(best_symbol_row["Resolved"] or 0),
        "best_timeframe": str(best_timeframe_row["timeframe"]).strip().lower(),
    }


def _best_signal_summary(
    *,
    selection: Mapping[str, object],
    timeframe_filter: str,
    analysis_limit: int,
) -> str:
    symbol = str(selection.get("symbol") or "").strip().upper()
    best_timeframe = str(selection.get("best_timeframe") or "").strip().lower()
    follow = float(selection.get("follow_through_pct") or 0.0)
    avg_dir = float(selection.get("avg_dir_return_pct") or 0.0)
    resolved = int(selection.get("resolved") or 0)
    qualified_timeframes = int(selection.get("qualified_timeframes") or 0)
    mode = str(selection.get("mode") or "").strip()
    if str(timeframe_filter or "").strip() and str(timeframe_filter) != "All":
        return (
            f"<b>{symbol}</b><br>"
            f"Best leader with enough history for <b>{str(timeframe_filter).upper()}</b><br>"
            f"{follow:.1f}% follow-through • {avg_dir:+.2f}% avg move • {resolved} completed"
        )
    if mode == "best_available":
        timeframe_note = (
            f"Only <b>{qualified_timeframes}</b> timeframe"
            if qualified_timeframes == 1
            else f"Only <b>{qualified_timeframes}</b> timeframes"
        )
        return (
            f"<b>{symbol}</b><br>"
            "Best available read with enough history<br>"
            f"{follow:.1f}% follow-through • {avg_dir:+.2f}% avg move • best TF {best_timeframe.upper() if best_timeframe else 'N/A'} • "
            f"{timeframe_note} ready"
        )
    return (
        f"<b>{symbol}</b><br>"
        "Best leader with enough history<br>"
        f"{follow:.1f}% follow-through • {avg_dir:+.2f}% avg move • best TF {best_timeframe.upper() if best_timeframe else 'N/A'} • "
        f"{qualified_timeframes} timeframe pockets ready"
    )


def _build_best_signal_leaderboard(
    *,
    df_events: pd.DataFrame,
    df_forward_windows: pd.DataFrame | None = None,
    timeframe_filter: str,
    limit: int = _BEST_SIGNAL_LEADERBOARD_LIMIT,
    min_resolved: int = _BEST_SIGNAL_MIN_RESOLVED,
    min_path_samples: int = 8,
    min_timeframes: int = _BEST_SIGNAL_MIN_TIMEFRAMES,
    min_total_resolved: int = _BEST_SIGNAL_MIN_TOTAL_RESOLVED,
) -> pd.DataFrame:
    if df_events is None or df_events.empty:
        return pd.DataFrame()
    required = {"symbol", "timeframe", "status"}
    if not required.issubset(df_events.columns):
        return pd.DataFrame()
    d = df_events.copy()
    d["status"] = d["status"].fillna("").astype(str).str.upper()
    d = d[d["status"].eq("RESOLVED")].copy()
    if d.empty:
        return pd.DataFrame()
    d = _filter_directional_signal_rows(d)
    if d.empty:
        return pd.DataFrame()
    d["symbol"] = d["symbol"].fillna("").astype(str).str.strip().str.upper()
    d["timeframe"] = d["timeframe"].fillna("").astype(str).str.strip().str.lower()
    d = d[d["symbol"].ne("") & d["timeframe"].ne("")].copy()
    if d.empty:
        return pd.DataFrame()
    d = _annotate_setup_class(d)
    if "signal_key" in d.columns:
        d["signal_key"] = d["signal_key"].fillna("").astype(str).str.strip()
    d["directional_return_pct"] = pd.to_numeric(d.get("directional_return_pct"), errors="coerce")
    grouped = (
        d.groupby(["symbol", "timeframe", "__setup_class", "__setup_label", "__direction_key"], dropna=False)
        .agg(
            Resolved=("symbol", "count"),
            FollowThroughPct=(
                "directional_return_pct",
                lambda s: float((pd.to_numeric(s, errors="coerce") > 0).mean() * 100.0) if len(pd.Series(s).dropna()) else 0.0,
            ),
            AvgDirReturnPct=("directional_return_pct", "mean"),
        )
        .reset_index()
    )
    grouped["QualityScore"] = grouped.apply(
        lambda row: _best_signal_quality_score(
            follow_through_pct=float(row.get("FollowThroughPct") or 0.0),
            avg_dir_return_pct=float(row.get("AvgDirReturnPct") or 0.0),
            resolved=int(row.get("Resolved") or 0),
            resolved_baseline=min_resolved,
        ),
        axis=1,
    )
    grouped, path_filter_active = _merge_path_sample_counts(grouped, d, df_forward_windows)
    if not path_filter_active and "PathSamples" not in grouped.columns:
        grouped["PathSamples"] = grouped["Resolved"]
    qualified = grouped[grouped["Resolved"] >= int(max(1, min_resolved))].copy()
    if path_filter_active:
        qualified = qualified[qualified["PathSamples"] >= int(max(1, min_path_samples))].copy()
    if qualified.empty:
        return pd.DataFrame()
    timeframe_text = str(timeframe_filter or "").strip().lower()
    if timeframe_text and timeframe_text != "all":
        qualified = qualified[qualified["timeframe"].eq(timeframe_text)].copy()
        if qualified.empty:
            return pd.DataFrame()
        board = (
            qualified.sort_values(
                ["QualityScore", "FollowThroughPct", "AvgDirReturnPct", "PathSamples", "Resolved"],
                ascending=[False, False, False, False, False],
            )
            .groupby("symbol", dropna=False)
            .head(1)
        )
        board = board.sort_values(
            ["QualityScore", "FollowThroughPct", "AvgDirReturnPct", "PathSamples", "Resolved"],
            ascending=[False, False, False, False, False],
        ).head(int(limit)).copy()
        board["Mode"] = "Best Signal"
        board["Best TF"] = board["timeframe"].astype(str).str.upper()
        board["Best Setup"] = board.apply(
            lambda row: _setup_display_label(row.get("__setup_class"), row.get("__direction_key")),
            axis=1,
        )
        board["Follow-Through"] = board["FollowThroughPct"].map(lambda value: f"{float(value):.1f}%")
        board["Avg Move"] = board["AvgDirReturnPct"].map(lambda value: f"{float(value):+.2f}%")
        board = board.rename(columns={"symbol": "Coin"})
        return board[["Coin", "Mode", "Best Setup", "Follow-Through", "Resolved", "Best TF", "Avg Move"]].reset_index(drop=True)

    symbol_scores = (
        qualified.groupby("symbol", dropna=False)
        .agg(
            QualifiedTimeframes=("timeframe", "nunique"),
            AvgFollowThroughPct=("FollowThroughPct", "mean"),
            AvgDirReturnPct=("AvgDirReturnPct", "mean"),
            Resolved=("Resolved", "sum"),
            PathSamples=("PathSamples", "sum"),
        )
        .reset_index()
    )
    symbol_scores["QualityScore"] = symbol_scores.apply(
        lambda row: _best_signal_quality_score(
            follow_through_pct=float(row.get("AvgFollowThroughPct") or 0.0),
            avg_dir_return_pct=float(row.get("AvgDirReturnPct") or 0.0),
            resolved=int(row.get("Resolved") or 0),
            resolved_baseline=min_total_resolved,
            qualified_timeframes=int(row.get("QualifiedTimeframes") or 0),
            timeframe_baseline=min_timeframes,
        ),
        axis=1,
    )
    best_timeframes = (
        qualified.sort_values(
            ["QualityScore", "FollowThroughPct", "AvgDirReturnPct", "PathSamples", "Resolved"],
            ascending=[False, False, False, False, False],
        )
        .groupby("symbol", dropna=False)
        .head(1)[
            [
                "symbol",
                "timeframe",
                "__setup_class",
                "__direction_key",
                "FollowThroughPct",
                "AvgDirReturnPct",
                "Resolved",
                "PathSamples",
            ]
        ]
        .rename(
            columns={
                "timeframe": "best_timeframe",
                "__setup_class": "best_setup_class",
                "__direction_key": "best_direction",
                "FollowThroughPct": "BestFollowThroughPct",
                "AvgDirReturnPct": "BestAvgDirReturnPct",
                "Resolved": "BestResolved",
                "PathSamples": "BestPathSamples",
            }
        )
    )
    board = symbol_scores.merge(best_timeframes, on="symbol", how="left")
    board["Mode"] = board.apply(
        lambda row: (
            "Best Signal"
            if int(row.get("QualifiedTimeframes") or 0) >= int(max(1, min_timeframes))
            and int(row.get("Resolved") or 0) >= int(max(1, min_total_resolved))
            else "Best Read"
        ),
        axis=1,
    )
    board = board.sort_values(
        ["QualityScore", "AvgFollowThroughPct", "AvgDirReturnPct", "QualifiedTimeframes", "PathSamples", "Resolved"],
        ascending=[False, False, False, False, False, False],
    ).head(int(limit)).copy()
    board["Best TF"] = board["best_timeframe"].fillna("").astype(str).str.upper()
    board["Best Setup"] = board.apply(
        lambda row: _setup_display_label(row.get("best_setup_class"), row.get("best_direction")),
        axis=1,
    )
    board["Follow-Through"] = board["BestFollowThroughPct"].map(lambda value: f"{float(value):.1f}%")
    board["Avg Move"] = board["BestAvgDirReturnPct"].map(lambda value: f"{float(value):+.2f}%")
    board["Resolved"] = pd.to_numeric(board["BestResolved"], errors="coerce").fillna(0).astype(int)
    board = board.rename(columns={"symbol": "Coin"})
    return board[["Coin", "Mode", "Best Setup", "Follow-Through", "Resolved", "Best TF", "Avg Move"]].reset_index(drop=True)


def _selected_dataframe_row_index(selection_state: object) -> int | None:
    if selection_state is None:
        return None
    selection_obj = None
    if isinstance(selection_state, Mapping):
        selection_obj = selection_state.get("selection")
    else:
        selection_obj = getattr(selection_state, "selection", None)
    if selection_obj is None:
        return None
    if isinstance(selection_obj, Mapping):
        rows = selection_obj.get("rows")
    else:
        rows = getattr(selection_obj, "rows", None)
    if not rows:
        return None
    try:
        selected_idx = int(rows[0])
    except Exception:
        return None
    return selected_idx if selected_idx >= 0 else None


def _selected_best_signal_coin(leaderboard_df: pd.DataFrame, selection_state: object) -> str:
    if leaderboard_df is None or leaderboard_df.empty or "Coin" not in leaderboard_df.columns:
        return ""
    selected_idx = _selected_dataframe_row_index(selection_state)
    if selected_idx is None or selected_idx >= len(leaderboard_df):
        return ""
    return _normalize_symbol_filter(leaderboard_df.iloc[int(selected_idx)]["Coin"])


def _learning_readiness_summary(*, mode: str, current_rows: int, total_rows: int) -> tuple[str, str]:
    if mode == "current_only":
        return (
            f"<b>Learning active</b><br>{int(current_rows)} signals in this history window",
            "positive",
        )
    if mode == "mixed_fallback":
        return (
            f"<b>Broader archive read</b><br>{int(current_rows)} matching signals inside {int(total_rows)} recent completed signals",
            "warning",
        )
    if mode == "unversioned_fallback":
        return (
            "<b>Older archive read</b><br>Current-version history is still separating",
            "warning",
        )
    if mode == "empty":
        return (
            "<b>No completed history yet</b><br>Learning turns on after the first completed signals",
            "neutral",
        )
    return (
        f"<b>Learning building</b><br>{int(current_rows)} signals in this history window",
        "neutral",
    )


def _archive_health_summary(storage_snapshot) -> tuple[str, str]:
    raw_headline = str(storage_snapshot.durability_label or "Archive OK").strip() or "Archive OK"
    headline_map = {
        "Override, Mirror Missing": "Backup Mirror Missing",
        "Custom Storage Override": "Custom Archive Path",
        "Deploy-Ready Path": "Archive Protected",
        "Workspace Storage": "Workspace Archive",
        "Ephemeral Storage": "Temporary Archive",
    }
    headline = headline_map.get(raw_headline, raw_headline)
    recovery = str(storage_snapshot.recovery_status or "Unknown").strip() or "Unknown"
    body = f"<b>{headline}</b><br>Recovery status: {recovery}"
    tone = str(storage_snapshot.durability_tone or "neutral")
    return body, tone


def _hold_guidance_cell(snapshot: Mapping[str, object], *, direction_label: str | None = None) -> str:
    resolved_signals = int(snapshot.get("resolved_signals") or 0)
    if bool(snapshot.get("available")):
        best_bar = int(snapshot.get("best_bar") or 0)
        fade_after_bar = int(snapshot.get("fade_after_bar") or 0)
        if best_bar > 0 and fade_after_bar > best_bar:
            best_label = f"Best at {_format_bar_count(best_bar)}, fades after {_format_bar_count(fade_after_bar)}"
        elif best_bar > 0:
            best_label = f"Best at {_format_bar_count(best_bar)}"
        else:
            best_label = str(snapshot.get("best_label") or "").strip() or "around 0 bars"
        return best_label
    if resolved_signals > 0:
        return f"Building ({resolved_signals} completed)"
    return "—"


def _projection_float(value: object, default: float = 0.0) -> float:
    return projection_float(value, default)


def _format_projection_price(value: object) -> str:
    price = _projection_float(value, 0.0)
    if price <= 0:
        return ""
    if price >= 1000:
        return f"${price:,.2f}"
    if price >= 1:
        return f"${price:,.4f}"
    if price >= 0.01:
        return f"${price:,.6f}"
    if price >= 0.0001:
        return f"${price:,.8f}"
    return f"${price:,.10f}"


def _format_projection_pct(value: object, *, signed: bool = True) -> str:
    number = _projection_float(value, 0.0)
    return f"{number:+.2f}%" if signed else f"{number:.2f}%"


def _format_bar_count(value: object) -> str:
    bars = int(_projection_float(value, 0.0))
    return f"{bars} bar" if bars == 1 else f"{bars} bars"


def _format_directional_projection_range(low_pct: object, high_pct: object, direction_key: str) -> str:
    low = abs(_projection_float(low_pct, 0.0))
    high = abs(_projection_float(high_pct, 0.0))
    low, high = sorted([low, high])
    if str(direction_key or "").strip().upper() == "DOWNSIDE":
        return f"-{low:.2f}% to -{high:.2f}%"
    return f"+{low:.2f}% to +{high:.2f}%"


def _expected_path_archive_check_html(snapshot: Mapping[str, object]) -> str:
    sample = int(snapshot.get("archive_check_sample") or 0)
    zone_hit = snapshot.get("zone_hit_rate_pct")
    clean_path = snapshot.get("clean_path_rate_pct")
    if sample <= 0 or zone_hit is None or clean_path is None:
        return ""
    try:
        zone_hit_pct = float(zone_hit)
        clean_path_pct = float(clean_path)
    except Exception:
        return ""
    if pd.isna(zone_hit_pct) or pd.isna(clean_path_pct):
        return ""
    return (
        f"Archive check: <b>{zone_hit_pct:.0f}% reached this zone</b>; "
        f"{clean_path_pct:.0f}% stayed within normal pullback<br>"
    )


def _with_expected_path_reference_price(
    snapshot: Mapping[str, object],
    reference_price: object,
    reference_label: str = "latest close",
) -> dict[str, object]:
    out = with_expected_path_reference_price(snapshot, reference_price, reference_label)
    price_low = _projection_float(out.get("price_zone_low"), 0.0)
    price_high = _projection_float(out.get("price_zone_high"), 0.0)
    pullback_price = _projection_float(out.get("pullback_price"), 0.0)
    caution_price = _projection_float(out.get("caution_price"), 0.0)
    if price_low > 0 and price_high > 0:
        out["price_zone_label"] = f"{_format_projection_price(price_low)} - {_format_projection_price(price_high)}"
    if pullback_price > 0:
        out["pullback_price_label"] = _format_projection_price(pullback_price)
    if caution_price > 0:
        out["caution_price_label"] = _format_projection_price(caution_price)
    return out


def _fetch_expected_path_reference_price(fetch_ohlcv, symbol: str, timeframe: str) -> tuple[float | None, str]:
    if not callable(fetch_ohlcv):
        return None, ""
    symbol_text = str(symbol or "").strip().upper()
    timeframe_text = str(timeframe or "").strip().lower()
    if not symbol_text or not timeframe_text or timeframe_text == "all":
        return None, ""
    symbol_candidates = [symbol_text]
    if "/" not in symbol_text:
        symbol_candidates.extend([f"{symbol_text}/USDT", f"{symbol_text}/USD"])
    seen: set[str] = set()
    for candidate in symbol_candidates:
        candidate = str(candidate or "").strip().upper()
        if not candidate or candidate in seen:
            continue
        seen.add(candidate)
        try:
            df = fetch_ohlcv(candidate, timeframe_text, limit=6)
        except Exception:
            continue
        if df is None or getattr(df, "empty", True) or "close" not in getattr(df, "columns", []):
            continue
        closes = pd.to_numeric(df["close"], errors="coerce").dropna()
        if closes.empty:
            continue
        return float(closes.iloc[-1]), "latest close"
    return None, ""


def _expected_path_timing_label(snapshot: Mapping[str, object]) -> str:
    bars = int(snapshot.get("best_bar") or 0)
    timeframe = str(snapshot.get("timeframe") or "").strip().upper()
    if bars <= 0 or not timeframe:
        return ""
    candle_label = "candle" if bars == 1 else "candles"
    return f"next {bars} {candle_label} on {timeframe}"


def _build_expected_path_projection(
    *,
    df_events: pd.DataFrame,
    df_forward_windows: pd.DataFrame,
    symbol_filter: str,
    timeframe_filter: str,
    min_samples: int = 8,
    now: object | None = None,
) -> dict[str, object]:
    snapshot = build_archive_expected_path_projection(
        df_events=df_events,
        df_forward_windows=df_forward_windows,
        symbol_filter=symbol_filter,
        timeframe_filter=timeframe_filter,
        timeframe_order=_HOLD_GUIDANCE_TIMEFRAME_ORDER,
        min_samples=min_samples,
        now=now,
    )
    if bool(snapshot.get("available")) and _projection_float(snapshot.get("reference_price"), 0.0) > 0:
        return _with_expected_path_reference_price(
            snapshot,
            snapshot.get("reference_price"),
            str(snapshot.get("reference_price_label") or "latest archived price"),
        )
    return snapshot


def _expected_path_body_html(snapshot: Mapping[str, object]) -> str:
    if not bool(snapshot.get("available")):
        return ""
    symbol = html.escape(str(snapshot.get("symbol") or "").strip().upper() or "Coin")
    timeframe = html.escape(str(snapshot.get("timeframe") or "").strip().upper())
    direction_key = str(snapshot.get("direction") or "").strip().upper()
    direction = "Upside" if direction_key == "UPSIDE" else ("Downside" if direction_key == "DOWNSIDE" else "Direction")
    direction_escaped = html.escape(direction)
    move_hint = "usual upside window" if direction_key == "UPSIDE" else "usual downside window"
    move_pct = _format_directional_projection_range(
        snapshot.get("best_zone_low_pct"),
        snapshot.get("best_zone_high_pct"),
        direction_key,
    )
    price_zone = str(snapshot.get("price_zone_label") or "").strip()
    pullback_pct = _projection_float(snapshot.get("normal_pullback_pct"), 0.0)
    if direction_key == "DOWNSIDE":
        pullback_pct_label = f"+{pullback_pct:.2f}% against the setup"
    else:
        pullback_pct_label = f"-{pullback_pct:.2f}%"
    pullback = pullback_pct_label
    pullback_price = str(snapshot.get("pullback_price_label") or "").strip()
    if pullback_price:
        pullback = f"{pullback} ({html.escape(pullback_price)})"
    fade_after = int(snapshot.get("fade_after_bar") or 0)
    fade_text = _format_bar_count(fade_after) if fade_after > 0 else "not clear yet"
    quality = html.escape(str(snapshot.get("read_quality") or "Thin"))
    sample = int(snapshot.get("sample") or 0)
    reference_price = _projection_float(snapshot.get("reference_price"), 0.0)
    reference_label = html.escape(str(snapshot.get("reference_price_label") or "latest close"))
    timing_label = html.escape(_expected_path_timing_label(snapshot))
    timing_html = f"Timing: <b>{timing_label}</b><br>" if timing_label else ""
    archive_check_html = _expected_path_archive_check_html(snapshot)
    if price_zone and reference_price > 0:
        return (
            f"<b>{symbol} {timeframe} {direction_escaped}</b><br>"
            f"Reference price: <b>{_format_projection_price(reference_price)}</b> "
            f"<span style='opacity:0.70;'>({reference_label})</span><br>"
            f"Expected zone: <b>{html.escape(price_zone)}</b><br>"
            f"{timing_html.replace('Timing:', 'Best path window:', 1)}"
            f"Move from that price: <b>{html.escape(move_pct)}</b> "
            f"<span style='opacity:0.70;'>({html.escape(move_hint)})</span><br>"
            f"Normal pullback: <b>{html.escape(pullback_price or pullback)}</b> "
            f"<span style='opacity:0.70;'>({html.escape(pullback_pct_label)}, can still be normal)</span><br>"
            f"Path weakens after: <b>{html.escape(fade_text)}</b><br>"
            f"{archive_check_html}"
            f"History depth: <b>{quality}</b> ({sample} similar signals)<br>"
            "<span style='opacity:0.72;'>Price path from similar past signals, not a price target. Hold window above is the efficiency read.</span>"
        )
    return (
        f"<b>{symbol} {timeframe} {direction_escaped}</b><br>"
        f"Expected move: <b>{html.escape(move_pct)}</b> "
        f"<span style='opacity:0.70;'>({html.escape(move_hint)})</span><br>"
        f"{timing_html.replace('Timing:', 'Best path window:', 1)}"
        f"Normal pullback: <b>{pullback}</b> "
        "<span style='opacity:0.70;'>(can still be normal)</span><br>"
        f"Path weakens after: <b>{html.escape(fade_text)}</b><br>"
        f"{archive_check_html}"
        f"History depth: <b>{quality}</b> ({sample} similar signals)<br>"
        "<span style='opacity:0.72;'>Price path from similar past signals, not a price target. Hold window above is the efficiency read.</span>"
    )


def _expected_path_scope_label(snapshot: Mapping[str, object]) -> str:
    symbol = str(snapshot.get("symbol") or "").strip().upper() or "Coin"
    timeframe = str(snapshot.get("timeframe") or "").strip().upper()
    direction_key = str(snapshot.get("direction") or "").strip().upper()
    direction = "Upside" if direction_key == "UPSIDE" else ("Downside" if direction_key == "DOWNSIDE" else "Direction")
    parts = [symbol]
    if timeframe:
        parts.append(timeframe)
    parts.append(direction)
    return " ".join(parts)


def _expected_path_kpi_items(snapshot: Mapping[str, object]) -> list[dict[str, object]]:
    if not bool(snapshot.get("available")):
        return []

    direction_key = str(snapshot.get("direction") or "").strip().upper()
    move_pct = _format_directional_projection_range(
        snapshot.get("best_zone_low_pct"),
        snapshot.get("best_zone_high_pct"),
        direction_key,
    )
    price_zone = str(snapshot.get("price_zone_label") or "").strip()
    reference_price = _projection_float(snapshot.get("reference_price"), 0.0)
    reference_label = str(snapshot.get("reference_price_label") or "latest close").strip() or "latest close"
    zone_value = price_zone if price_zone and reference_price > 0 else move_pct
    zone_subtext = (
        f"From {_format_projection_price(reference_price)} {reference_label}; move {move_pct}"
        if price_zone and reference_price > 0
        else "Usual move window from similar signals"
    )

    bars = int(snapshot.get("best_bar") or 0)
    timeframe = str(snapshot.get("timeframe") or "").strip().upper()
    timing_value = f"Next {bars}" if bars > 0 else "Building"
    candle_label = "candle" if bars == 1 else "candles"
    timing_subtext = f"{candle_label} on {timeframe}" if bars > 0 and timeframe else "Waiting for timing history"

    pullback_pct = max(0.0, abs(_projection_float(snapshot.get("normal_pullback_pct"), 0.0)))
    if direction_key == "DOWNSIDE":
        pullback_pct_label = f"+{pullback_pct:.2f}% against setup"
        caution_side = "above"
    else:
        pullback_pct_label = f"-{pullback_pct:.2f}%"
        caution_side = "below"
    pullback_price = str(snapshot.get("pullback_price_label") or "").strip()
    caution_price = str(snapshot.get("caution_price_label") or "").strip()
    caution_pct = max(0.0, abs(_projection_float(snapshot.get("caution_pullback_pct"), pullback_pct)))
    caution_subtext = f"{pullback_pct_label} can still be normal"
    if caution_price:
        caution_subtext = f"Normal up to {pullback_pct_label}; caution {caution_side} {caution_price}"
    elif caution_pct > pullback_pct:
        caution_subtext = f"Normal up to {pullback_pct_label}; caution past {caution_pct:.2f}%"

    fade_after = int(snapshot.get("fade_after_bar") or 0)
    fade_value = _format_bar_count(fade_after) if fade_after > 0 else "Not clear"
    fade_subtext = "Archive edge usually weakens here" if fade_after > 0 else "No clean fade point yet"

    quality = str(snapshot.get("read_quality") or "Thin")
    sample = int(snapshot.get("sample") or 0)
    archive_sample = int(snapshot.get("archive_check_sample") or 0)
    zone_hit = snapshot.get("zone_hit_rate_pct")
    clean_path = snapshot.get("clean_path_rate_pct")
    badge_text = ""
    if archive_sample > 0 and zone_hit is not None:
        try:
            zone_hit_pct = float(zone_hit)
            if not pd.isna(zone_hit_pct):
                badge_text = f"{zone_hit_pct:.0f}% reached zone"
        except Exception:
            badge_text = ""
    history_subtext = f"{sample} similar signals"
    if clean_path is not None:
        try:
            clean_path_pct = float(clean_path)
            if not pd.isna(clean_path_pct):
                history_subtext = f"{history_subtext}; {clean_path_pct:.0f}% clean route"
        except Exception:
            pass
    if bool(snapshot.get("path_conflict")):
        history_subtext = f"{history_subtext}; alternate read is close"

    quality_tone = "positive" if quality == "Strong" else ("warning" if quality == "Good" else "neutral")
    quality_color = "#00FF88" if quality == "Strong" else ("#FFD166" if quality == "Good" else "#8AA0B8")

    return [
        {
            "label": "Expected Zone",
            "value": zone_value,
            "value_color": "#00FF88" if direction_key == "UPSIDE" else "#FF3366" if direction_key == "DOWNSIDE" else "#F7FBFF",
            "subtext": zone_subtext,
        },
        {
            "label": "Timing",
            "value": timing_value,
            "subtext": timing_subtext,
        },
        {
            "label": "Normal Shakeout",
            "value": pullback_price or pullback_pct_label,
            "subtext": caution_subtext,
        },
        {
            "label": "Fades After",
            "value": fade_value,
            "subtext": fade_subtext,
        },
        {
            "label": "History Depth",
            "value": quality,
            "value_color": quality_color,
            "subtext": history_subtext,
            "badge_text": badge_text,
            "badge_tone": quality_tone,
        },
    ]


def _expected_path_building_body_html(
    *,
    symbol_filter: str,
    timeframe_filter: str,
    df_events: pd.DataFrame,
    df_forward_windows: pd.DataFrame,
    min_samples: int = 8,
) -> str:
    symbol = html.escape(str(symbol_filter or "").strip().upper() or "Coin")
    timeframe_text = str(timeframe_filter or "").strip()
    scope_label = timeframe_text.upper() if timeframe_text and timeframe_text != "All" else "this scope"
    if df_events is None or df_events.empty:
        return (
            f"<b>{symbol} path is still building.</b><br>"
            f"No completed signals are loaded for <b>{html.escape(scope_label)}</b> yet."
        )
    if "timeframe" not in df_events.columns or "direction" not in df_events.columns:
        return (
            f"<b>{symbol} path is still building.</b><br>"
            "This view does not have enough timing data yet."
        )
    d = df_events.copy()
    d["timeframe"] = d["timeframe"].fillna("").astype(str).str.strip().str.lower()
    d["__direction_key"] = d["direction"].map(_archive_direction_key)
    status = d.get("status", pd.Series(index=d.index, dtype=object)).fillna("").astype(str).str.upper()
    d = d[status.eq("RESOLVED") & d["__direction_key"].isin({"UPSIDE", "DOWNSIDE"})].copy()
    timeframe_key = timeframe_text.lower()
    if timeframe_key and timeframe_key != "all":
        d = d[d["timeframe"].eq(timeframe_key)].copy()
    if d.empty:
        return (
            f"<b>{symbol} path is still building.</b><br>"
            f"No upside/downside history is ready for <b>{html.escape(scope_label)}</b> yet."
        )
    if "signal_key" not in d.columns:
        return (
            f"<b>{symbol} path is still building.</b><br>"
            "History exists, but timing is not linked yet."
        )
    directional_count = int(d["signal_key"].fillna("").astype(str).str.strip().replace("", pd.NA).dropna().nunique())
    window_keys: set[str] = set()
    if df_forward_windows is not None and not df_forward_windows.empty and "signal_key" in df_forward_windows.columns:
        window_keys = set(df_forward_windows["signal_key"].fillna("").astype(str).str.strip())
        window_keys.discard("")
    checkpoint_count = int(
        d[d["signal_key"].fillna("").astype(str).str.strip().isin(window_keys)]["signal_key"].nunique()
        if window_keys
        else 0
    )
    need = int(max(1, min_samples))
    if checkpoint_count < need:
        return (
            f"<b>{symbol} path is still building.</b><br>"
            f"<b>{html.escape(scope_label)}</b> has <b>{checkpoint_count}/{need}</b> timed examples ready "
            f"(<b>{directional_count}</b> completed signals found). "
            "Expected Path appears once enough similar signals finish their timing window."
        )
    return (
        f"<b>{symbol} path is still building.</b><br>"
        f"<b>{html.escape(scope_label)}</b> has enough timed examples, but no clean price zone passed the quality filter yet."
    )


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


def _hold_autofill_scope_key(*, symbol_filter: str, timeframe_filter: str, decision_version: str) -> str:
    symbol_label = str(symbol_filter or "").strip().upper() or "MARKET"
    timeframe_label = str(timeframe_filter or "").strip().upper() or "ALL"
    version_label = str(decision_version or "").strip() or "UNVERSIONED"
    return f"{symbol_label}|{timeframe_label}|{version_label}"


def _should_run_hold_autofill(
    st,
    *,
    scope_key: str,
    cooldown_seconds: int = 600,
) -> bool:
    registry = st.session_state.get("signal_review_hold_autofill_registry", {})
    if not isinstance(registry, dict):
        return True
    last_run = float(registry.get(scope_key) or 0.0)
    return (time.time() - last_run) >= float(cooldown_seconds)


def _record_hold_autofill_attempt(st, *, scope_key: str) -> None:
    registry = st.session_state.get("signal_review_hold_autofill_registry", {})
    if not isinstance(registry, dict):
        registry = {}
    registry[scope_key] = float(time.time())
    st.session_state["signal_review_hold_autofill_registry"] = registry


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


def _ordered_timeframe_scope(*, timeframe_filter: str, available_timeframes: set[str] | None = None) -> list[str]:
    timeframe_text = str(timeframe_filter or "").strip().lower()
    if timeframe_text and timeframe_text != "all":
        return [timeframe_text]
    extras = sorted(set(available_timeframes or set()).difference(_HOLD_GUIDANCE_TIMEFRAME_ORDER))
    return [*_HOLD_GUIDANCE_TIMEFRAME_ORDER, *extras]


def _build_coin_timeframe_intelligence_bundle(
    *,
    timeframe_frames: list[dict[str, object]],
    build_signal_cohort_summary,
    build_hold_window_intelligence,
) -> tuple[pd.DataFrame, pd.DataFrame, list[dict[str, object]]]:
    display_rows: list[dict[str, object]] = []
    summary_rows: list[dict[str, object]] = []
    hold_rows: list[dict[str, object]] = []
    timeframe_order = {tf: idx for idx, tf in enumerate(_HOLD_GUIDANCE_TIMEFRAME_ORDER)}
    for frame in list(timeframe_frames or []):
        timeframe_key = str(frame.get("timeframe") or "").strip().lower()
        if not timeframe_key:
            continue
        timeframe_label = timeframe_key.upper()
        tf_df = frame.get("events")
        tf_windows = frame.get("windows")
        if not isinstance(tf_df, pd.DataFrame):
            tf_df = pd.DataFrame()
        if not isinstance(tf_windows, pd.DataFrame):
            tf_windows = pd.DataFrame()

        resolved = 0
        follow_through_pct = 0.0
        avg_dir_return_pct = 0.0
        if not tf_df.empty:
            tf_summary_df = build_signal_cohort_summary(tf_df, "timeframe")
            if tf_summary_df is not None and not tf_summary_df.empty:
                tf_summary = tf_summary_df.iloc[0]
                resolved = int(tf_summary.get("Resolved") or 0)
                follow_through_pct = float(tf_summary.get("FollowThroughPct") or 0.0)
                avg_dir_return_pct = float(tf_summary.get("AvgDirReturnPct") or 0.0)

        hold_row = {
            "Timeframe": timeframe_label,
            "Upside Snapshot": {"available": False, "resolved_signals": 0},
            "Upside Hold": "—",
            "Downside Snapshot": {"available": False, "resolved_signals": 0},
            "Downside Hold": "—",
        }
        tf_hold_rows = _build_coin_hold_guidance_rows(
            tf_df,
            tf_windows,
            build_hold_window_intelligence,
            timeframe_filter=timeframe_key,
        )
        if tf_hold_rows:
            hold_row = tf_hold_rows[0]
        hold_rows.append(hold_row)
        summary_rows.append(
            {
                "_order": timeframe_order.get(timeframe_key, 999),
                "timeframe": timeframe_key,
                "Resolved": resolved,
                "FollowThroughPct": follow_through_pct,
                "AvgDirReturnPct": avg_dir_return_pct,
            }
        )
        display_rows.append(
            {
                "_order": timeframe_order.get(timeframe_key, 999),
                "Timeframe": timeframe_label,
                "Completed": resolved,
                "Follow-Through": f"{follow_through_pct:.1f}%" if resolved > 0 else "N/A",
                "Avg Move": f"{avg_dir_return_pct:+.2f}%" if resolved > 0 else "N/A",
                "Upside Hold": str(hold_row.get("Upside Hold") or "—"),
                "Downside Hold": str(hold_row.get("Downside Hold") or "—"),
            }
        )

    display_df = pd.DataFrame(display_rows)
    if not display_df.empty:
        display_df = (
            display_df.sort_values(["_order", "Completed"], ascending=[True, False])
            .drop(columns=["_order"])
            .reset_index(drop=True)
        )
    summary_df = pd.DataFrame(summary_rows)
    if not summary_df.empty:
        summary_df = (
            summary_df.sort_values(["_order", "Resolved"], ascending=[True, False])
            .drop(columns=["_order"])
            .reset_index(drop=True)
    )
    return display_df, summary_df, hold_rows


def _load_coin_timeframe_frames(
    *,
    fetch_signal_events_df,
    fetch_signal_forward_windows_df,
    symbol_filter: str,
    timeframe_filter: str,
    status_filter: str,
    current_market_version: str,
    analysis_limit: int,
    db_path: str,
    base_events: pd.DataFrame,
) -> list[dict[str, object]]:
    available_timeframes: set[str] = set()
    if isinstance(base_events, pd.DataFrame) and not base_events.empty and "timeframe" in base_events.columns:
        available_timeframes = set(
            base_events["timeframe"].fillna("").astype(str).str.strip().str.lower().tolist()
        )
        available_timeframes.discard("")
    timeframe_scope = _ordered_timeframe_scope(
        timeframe_filter=timeframe_filter,
        available_timeframes=available_timeframes,
    )
    frames: list[dict[str, object]] = []
    has_base_events = isinstance(base_events, pd.DataFrame) and not base_events.empty and "timeframe" in base_events.columns
    for timeframe_key in timeframe_scope:
        if has_base_events:
            tf_events = base_events.copy()
            if not tf_events.empty and "timeframe" in tf_events.columns:
                tf_events = tf_events[
                    tf_events["timeframe"].fillna("").astype(str).str.strip().str.lower().eq(timeframe_key)
                ].copy()
            if str(status_filter or "").strip() != "All" and not tf_events.empty and "status" in tf_events.columns:
                tf_events = tf_events[
                    tf_events["status"].fillna("").astype(str).str.upper().eq(str(status_filter).strip().upper())
                ].copy()
        else:
            tf_events = fetch_signal_events_df(
                limit=int(analysis_limit),
                status=None if status_filter == "All" else status_filter.upper(),
                source="Market",
                timeframe=timeframe_key,
                symbol=symbol_filter,
                decision_version=current_market_version,
                db_path=db_path,
            )
        tf_signal_keys = (
            tf_events["signal_key"].fillna("").astype(str).str.strip().tolist()
            if not tf_events.empty and "signal_key" in tf_events.columns
            else []
        )
        tf_windows = (
            fetch_signal_forward_windows_df(
                signal_keys=tf_signal_keys,
                db_path=db_path,
            )
            if tf_signal_keys
            else pd.DataFrame()
        )
        frames.append(
            {
                "timeframe": timeframe_key,
                "events": tf_events,
                "windows": tf_windows,
            }
        )
    return frames


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


def _prefer_known_summary_rows(summary_df: pd.DataFrame, *, label_field: str) -> pd.DataFrame:
    if summary_df is None or summary_df.empty or label_field not in summary_df.columns:
        return pd.DataFrame()
    d = summary_df.copy()
    labels = d[label_field].fillna("").astype(str).str.strip()
    known_mask = ~labels.str.contains(r"\bUnknown\b", case=False, na=False)
    if bool(known_mask.any()):
        return d.loc[known_mask].copy()
    return pd.DataFrame()


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
        st.caption("No breakdown data is available in this view yet.")
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
        st.caption("Timing history is still building in this view.")
        return
    cols = st.columns(2, gap="medium")
    for idx, (_, title, summary_df) in enumerate(visible_specs):
        with cols[idx % 2]:
            st.markdown(f"##### {title}")
            st.dataframe(summary_df.round(2), hide_index=True, width="stretch")


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
        st.caption("CSV exports prepare on demand so ordinary archive reads stay fast.")
        export_cache = st.session_state.get("signal_review_archive_export_cache", {})
        if not isinstance(export_cache, dict):
            export_cache = {}
        if st.button("Prepare CSV Exports", key="signal_review_prepare_exports"):
            full_events_csv = fetch_signal_events_df(limit=100000, source="Market", db_path=db_path)
            full_alerts_csv = fetch_market_alerts_df(limit=5000, source="Market", db_path=db_path)
            export_cache = {
                "signal_events_csv": full_events_csv.to_csv(index=False).encode("utf-8"),
                "market_alerts_csv": full_alerts_csv.to_csv(index=False).encode("utf-8"),
                "prepared_at": pd.Timestamp.utcnow().strftime("%Y-%m-%d %H:%M UTC"),
            }
            st.session_state["signal_review_archive_export_cache"] = export_cache
            st.rerun()
        signal_events_csv = export_cache.get("signal_events_csv")
        market_alerts_csv = export_cache.get("market_alerts_csv")
        if signal_events_csv and market_alerts_csv:
            st.caption(f"CSV exports prepared from the current archive snapshot at {export_cache.get('prepared_at')}.")
            st.download_button(
                "Download Signal Events CSV",
                data=signal_events_csv,
                file_name="signal_events_backup.csv",
                mime="text/csv",
                on_click="ignore",
            )
            st.download_button(
                "Download Market Alerts CSV",
                data=market_alerts_csv,
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
    resolve_open_signal_events_via_fetch = get_ctx(ctx, "resolve_open_signal_events_via_fetch")
    backfill_signal_forward_windows_via_fetch = get_ctx(ctx, "backfill_signal_forward_windows_via_fetch")
    fetch_signal_events_df = get_ctx(ctx, "fetch_signal_events_df")
    fetch_signal_forward_windows_df = get_ctx(ctx, "fetch_signal_forward_windows_df")
    build_signal_review_snapshot = get_ctx(ctx, "build_signal_review_snapshot")
    build_signal_cohort_summary = get_ctx(ctx, "build_signal_cohort_summary")
    build_hold_window_intelligence = get_ctx(ctx, "build_hold_window_intelligence")
    build_hold_window_cohort_summary = get_ctx(ctx, "build_hold_window_cohort_summary")
    annotate_alert_footprint = get_ctx(ctx, "annotate_alert_footprint")
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
            "Review logged market signals, their strongest historical setup pockets, and expected path. "
            "This is an archive read, not a live entry screen."
        ),
    )

    pending_view_mode = str(st.session_state.pop("signal_review_pending_view_mode", "") or "").strip()
    pending_symbol = _normalize_symbol_filter(st.session_state.pop("signal_review_pending_symbol", ""))
    pending_timeframe = str(st.session_state.pop("signal_review_pending_timeframe", "") or "").strip()
    pending_notice = str(st.session_state.pop("signal_review_pending_notice", "") or "").strip()
    pending_notice_tone = str(st.session_state.pop("signal_review_pending_notice_tone", "info") or "info")
    if pending_view_mode:
        st.session_state["signal_review_view_mode"] = pending_view_mode
        st.session_state.pop("signal_review_best_signal_leaderboard", None)
    if pending_symbol:
        st.session_state["signal_review_symbol"] = pending_symbol
    if pending_timeframe in {"5m", "15m", "1h", "4h", "1d"}:
        st.session_state["signal_review_tf"] = pending_timeframe
    if pending_notice:
        st.session_state["signal_review_tracker_notice"] = pending_notice
        st.session_state["signal_review_tracker_notice_tone"] = pending_notice_tone

    analysis_limit = ARCHIVE_LEARNING_WINDOW_ROWS
    recent_table_limit = 250

    view_mode = st.radio(
        "View Mode",
        ["Coin", "Best Signal"],
        horizontal=True,
        key="signal_review_view_mode",
        help="Pick a coin yourself or let the archive choose the strongest coin with enough history.",
    )
    coin_filter_input = ""
    if view_mode == "Coin":
        coin_filter_input = st.text_input(
            "Coin",
            value="",
            key="signal_review_symbol",
            placeholder="BTC",
            help="Type one coin symbol to load its history.",
        )
    symbol_filter = _normalize_symbol_filter(coin_filter_input)
    timeframe_options = ["All", "5m", "15m", "1h", "4h", "1d"]
    timeframe_filter = st.selectbox(
        "Timeframe",
        timeframe_options,
        index=0,
        key="signal_review_tf",
        help=(
            "All searches every timeframe. A specific timeframe limits Best Signal to that view."
            if view_mode == "Best Signal"
            else "All gives the broadest read. A specific timeframe narrows the coin view."
        ),
    )
    setup_filter_options = [label for label, _ in _SETUP_FILTER_OPTIONS]
    if str(st.session_state.get("signal_review_setup_type") or "") not in setup_filter_options:
        st.session_state["signal_review_setup_type"] = "Auto Best"
    setup_filter_label = st.selectbox(
        "Setup Type",
        setup_filter_options,
        index=0,
        key="signal_review_setup_type",
        help=(
            "Auto Best finds the strongest actionable setup pocket in this scope. "
            "Pick a specific setup type only when you want to narrow the archive read."
        ),
    )
    setup_filter_value = _setup_filter_value(setup_filter_label)
    status_filter = "Resolved" if view_mode == "Best Signal" else "All"

    current_market_version = current_decision_version("Market")
    best_signal_selection: dict[str, object] = {"available": False}
    best_signal_scope_df = pd.DataFrame()
    best_signal_windows_df = pd.DataFrame()
    refresh_limit_pairs = None

    if view_mode == "Coin" and not symbol_filter:
        render_insight_card(
            st,
            title="Enter a Coin",
            body_html=(
                "Type a <b>coin</b> above. Signal Archive will load its strongest learned setup pocket, timing, and expected path."
            ),
            tone="neutral",
        )
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

    with st.spinner("Refreshing recent signal outcomes..."):
        if view_mode == "Best Signal":
            resolved_now = int(
                resolve_open_signal_events_via_fetch(
                    fetch_ohlcv=fetch_ohlcv,
                    source="Market",
                    db_path=db_path,
                    limit_pairs=24,
                    candle_limit=260,
                    symbol=None,
                    timeframe=None if timeframe_filter == "All" else timeframe_filter,
                )
            )
            best_signal_scope_df = fetch_signal_events_df(
                limit=int(analysis_limit),
                status="RESOLVED",
                source="Market",
                timeframe=None if timeframe_filter == "All" else timeframe_filter,
                decision_version=current_market_version,
                db_path=db_path,
            )
            if setup_filter_value != "ALL":
                best_signal_scope_df = _filter_events_by_setup(best_signal_scope_df, setup_filter_value)
            best_signal_scope_keys = (
                best_signal_scope_df["signal_key"].fillna("").astype(str).str.strip().drop_duplicates().tolist()
                if not best_signal_scope_df.empty and "signal_key" in best_signal_scope_df.columns
                else []
            )
            best_signal_windows_df = (
                fetch_signal_forward_windows_df(
                    signal_keys=best_signal_scope_keys,
                    db_path=db_path,
                )
                if best_signal_scope_keys
                else pd.DataFrame()
            )
            best_signal_selection = _select_best_signal_coin(
                df_events=best_signal_scope_df,
                df_forward_windows=best_signal_windows_df,
                timeframe_filter=timeframe_filter,
            )
            symbol_filter = _normalize_symbol_filter(best_signal_selection.get("symbol"))
        else:
            resolved_now = int(
                resolve_open_signal_events_via_fetch(
                    fetch_ohlcv=fetch_ohlcv,
                    source="Market",
                    db_path=db_path,
                    limit_pairs=refresh_limit_pairs,
                    candle_limit=260,
                    symbol=symbol_filter,
                    timeframe=None if timeframe_filter == "All" else timeframe_filter,
                )
            )

    if not symbol_filter:
        render_insight_card(
            st,
            title="Best Signal Building",
            body_html=(
                "No leader yet. The archive needs more completed signals with timing history."
            ),
            tone="neutral",
        )
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

    df_events = fetch_signal_events_df(
        limit=int(analysis_limit),
        status=None if status_filter == "All" else status_filter.upper(),
        source="Market",
        timeframe=None if timeframe_filter == "All" else timeframe_filter,
        symbol=symbol_filter,
        decision_version=current_market_version,
        db_path=db_path,
    )
    adaptive_archive_df = fetch_signal_events_df(
        limit=analysis_limit,
        status="RESOLVED",
        source="Market",
        timeframe=None if timeframe_filter == "All" else timeframe_filter,
        symbol=symbol_filter,
        decision_version=current_market_version,
        db_path=db_path,
    )
    if setup_filter_value != "ALL":
        df_events = _filter_events_by_setup(df_events, setup_filter_value)
        adaptive_archive_df = _filter_events_by_setup(adaptive_archive_df, setup_filter_value)
    adaptive_rows = int(len(adaptive_archive_df))
    adaptive_total_rows = int(len(adaptive_archive_df))
    leaderboard_source_df = best_signal_scope_df if view_mode == "Best Signal" else adaptive_archive_df
    best_signal_leaderboard_df = (
        _build_best_signal_leaderboard(
            df_events=leaderboard_source_df,
            df_forward_windows=best_signal_windows_df if view_mode == "Best Signal" else None,
            timeframe_filter=timeframe_filter,
            limit=_BEST_SIGNAL_LEADERBOARD_LIMIT,
        )
        if view_mode == "Best Signal"
        else pd.DataFrame()
    )
    if adaptive_rows >= _MIN_SIGNAL_ARCHIVE_ROWS:
        adaptive_mode = "current_only"
    elif adaptive_rows > 0:
        adaptive_mode = "building"
    else:
        adaptive_mode = "empty"
    learning_summary, learning_tone = _learning_readiness_summary(
        mode=adaptive_mode,
        current_rows=adaptive_rows,
        total_rows=adaptive_total_rows,
    )
    archive_health_body, archive_health_tone = _archive_health_summary(storage_snapshot)
    show_archive_health = archive_health_tone not in {"neutral", "positive"}

    top_card_specs: list[tuple[str, str, str]] = [
        (
            (
                "Coin View"
                if view_mode == "Coin"
                else ("Best Read" if str(best_signal_selection.get("mode") or "").strip() == "best_available" else "Best Signal")
            ),
            (
                _coin_view_summary(
                    symbol_filter=symbol_filter,
                    timeframe_filter=timeframe_filter,
                    analysis_limit=int(analysis_limit),
                    rows_loaded=int(len(df_events)),
                )
                if view_mode == "Coin"
                else _best_signal_summary(
                    selection=best_signal_selection,
                    timeframe_filter=timeframe_filter,
                    analysis_limit=int(analysis_limit),
                )
            ),
            "neutral",
        ),
        ("Learning Readiness", learning_summary, learning_tone),
    ]
    if show_archive_health:
        top_card_specs.append(("Archive Storage", archive_health_body, archive_health_tone))

    top_insight_cols = st.columns(len(top_card_specs), gap="medium")
    for col, (title, body_html, tone) in zip(top_insight_cols, top_card_specs):
        with col:
            render_insight_card(
                st,
                title=title,
                body_html=body_html,
                tone=tone,
            )

    if view_mode == "Best Signal" and not best_signal_leaderboard_df.empty:
        qualified_count = int(len(best_signal_leaderboard_df))
        st.markdown("### Best Signal Leaderboard")
        st.caption(
            f"Showing {qualified_count} coin(s), up to {_BEST_SIGNAL_LEADERBOARD_LIMIT}. "
            "Select a row to open the full coin read."
        )
        leaderboard_state = st.dataframe(
            best_signal_leaderboard_df,
            hide_index=True,
            width="stretch",
            key="signal_review_best_signal_leaderboard",
            on_select="rerun",
            selection_mode="single-row",
        )
        selected_leaderboard_coin = _selected_best_signal_coin(best_signal_leaderboard_df, leaderboard_state)
        if selected_leaderboard_coin:
            selected_best_tf = ""
            selected_rows = best_signal_leaderboard_df[
                best_signal_leaderboard_df["Coin"].astype(str).str.upper().eq(selected_leaderboard_coin)
            ]
            if not selected_rows.empty:
                selected_best_tf = str(selected_rows.iloc[0].get("Best TF") or "").strip().upper()
            st.session_state["signal_review_pending_view_mode"] = "Coin"
            st.session_state["signal_review_pending_symbol"] = selected_leaderboard_coin
            if selected_best_tf:
                st.session_state["signal_review_pending_timeframe"] = selected_best_tf.lower()
                st.session_state["signal_review_pending_notice"] = (
                    f"Opened {selected_leaderboard_coin} from Best Signal Leaderboard. Best TF: {selected_best_tf}."
                )
                st.session_state["signal_review_pending_notice_tone"] = "info"
            st.rerun()
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
        empty_scope = "actionable setup" if setup_filter_value == "AUTO_BEST" else "selected setup"
        st.info(f"No {empty_scope} signals match this coin/timeframe yet. Let Market log more history.")
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

    hold_guidance_enabled = bool(symbol_filter)
    df_hold_archive_events_raw = (
        fetch_signal_events_df(
            limit=int(analysis_limit),
            status="RESOLVED",
            source="Market",
            timeframe=None if timeframe_filter == "All" else timeframe_filter,
            symbol=symbol_filter,
            decision_version=current_market_version,
            db_path=db_path,
        )
        if hold_guidance_enabled
        else pd.DataFrame()
    )
    if not df_hold_archive_events_raw.empty:
        if setup_filter_value != "ALL":
            df_hold_archive_events_raw = _filter_events_by_setup(df_hold_archive_events_raw, setup_filter_value)
    hold_archive_signal_keys = (
        df_hold_archive_events_raw["signal_key"].fillna("").astype(str).str.strip().tolist()
        if not df_hold_archive_events_raw.empty and "signal_key" in df_hold_archive_events_raw.columns
        else []
    )
    df_hold_archive_windows_raw = (
        fetch_signal_forward_windows_df(
            signal_keys=hold_archive_signal_keys,
            db_path=db_path,
        )
        if hold_guidance_enabled and hold_archive_signal_keys
        else pd.DataFrame()
    )
    archive_decision = build_archive_decision_snapshot(
        df_events=df_events,
        df_resolved_events=df_hold_archive_events_raw,
        df_forward_windows=df_hold_archive_windows_raw,
        symbol_filter=symbol_filter,
        timeframe_filter=timeframe_filter,
        setup_filter_value=setup_filter_value,
        min_completed=_MIN_SETUP_POCKET_ROWS,
        build_hold_window_intelligence_fn=build_hold_window_intelligence,
    )
    setup_pocket = _setup_pocket_from_snapshot(archive_decision.setup)
    metric_df_events = archive_decision.metric_events if not archive_decision.metric_events.empty else df_events.copy()
    timeframe_read_events = (
        archive_decision.direction_events if not archive_decision.direction_events.empty else metric_df_events.copy()
    )
    snapshot = build_signal_review_snapshot(metric_df_events)
    df_forward_windows = archive_decision.metric_windows
    df_hold_archive_events = archive_decision.hold_events
    df_hold_archive_windows = archive_decision.hold_windows
    timeframe_frames: list[dict[str, object]] = []
    timeframe_intelligence_df = pd.DataFrame()
    missing_hold_backfill = _missing_hold_backfill_count(df_hold_archive_events, df_hold_archive_windows)
    hold_autofill_scope_key = _hold_autofill_scope_key(
        symbol_filter=symbol_filter,
        timeframe_filter=timeframe_filter,
        decision_version=current_market_version,
    )
    if hold_guidance_enabled:
        timeframe_frames = _load_coin_timeframe_frames(
            fetch_signal_events_df=fetch_signal_events_df,
            fetch_signal_forward_windows_df=fetch_signal_forward_windows_df,
            symbol_filter=symbol_filter,
            timeframe_filter=timeframe_filter,
            status_filter="Resolved",
            current_market_version=current_market_version,
            analysis_limit=int(analysis_limit),
            db_path=db_path,
            base_events=timeframe_read_events,
        )
        timeframe_intelligence_df, _, _ = _build_coin_timeframe_intelligence_bundle(
            timeframe_frames=timeframe_frames,
            build_signal_cohort_summary=build_signal_cohort_summary,
            build_hold_window_intelligence=build_hold_window_intelligence,
        )
    autofilled_hold_now = 0
    if (
        hold_guidance_enabled
        and missing_hold_backfill > 0
        and _should_run_hold_autofill(st, scope_key=hold_autofill_scope_key)
    ):
        _record_hold_autofill_attempt(st, scope_key=hold_autofill_scope_key)
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
            df_hold_archive_windows_raw = (
                fetch_signal_forward_windows_df(
                    signal_keys=hold_archive_signal_keys,
                    db_path=db_path,
                )
                if hold_archive_signal_keys
                else pd.DataFrame()
            )
            archive_decision = build_archive_decision_snapshot(
                df_events=df_events,
                df_resolved_events=df_hold_archive_events_raw,
                df_forward_windows=df_hold_archive_windows_raw,
                symbol_filter=symbol_filter,
                timeframe_filter=timeframe_filter,
                setup_filter_value=setup_filter_value,
                min_completed=_MIN_SETUP_POCKET_ROWS,
                build_hold_window_intelligence_fn=build_hold_window_intelligence,
            )
            setup_pocket = _setup_pocket_from_snapshot(archive_decision.setup)
            metric_df_events = archive_decision.metric_events if not archive_decision.metric_events.empty else df_events.copy()
            timeframe_read_events = (
                archive_decision.direction_events if not archive_decision.direction_events.empty else metric_df_events.copy()
            )
            snapshot = build_signal_review_snapshot(metric_df_events)
            df_forward_windows = archive_decision.metric_windows
            df_hold_archive_events = archive_decision.hold_events
            df_hold_archive_windows = archive_decision.hold_windows
            timeframe_frames = _load_coin_timeframe_frames(
                fetch_signal_events_df=fetch_signal_events_df,
                fetch_signal_forward_windows_df=fetch_signal_forward_windows_df,
                symbol_filter=symbol_filter,
                timeframe_filter=timeframe_filter,
                status_filter="Resolved",
                current_market_version=current_market_version,
                analysis_limit=int(analysis_limit),
                db_path=db_path,
                base_events=timeframe_read_events,
            )
            timeframe_intelligence_df, _, _ = _build_coin_timeframe_intelligence_bundle(
                timeframe_frames=timeframe_frames,
                build_signal_cohort_summary=build_signal_cohort_summary,
                build_hold_window_intelligence=build_hold_window_intelligence,
            )
            missing_hold_backfill = _missing_hold_backfill_count(df_hold_archive_events, df_hold_archive_windows)
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
    tone_follow = (
        POSITIVE
        if resolved_metrics_available and float(snapshot["follow_through_rate"]) >= 55.0
        else (WARNING if resolved_metrics_available and float(snapshot["follow_through_rate"]) >= 45.0 else TEXT_MUTED)
    )
    tone_dir = POSITIVE if resolved_metrics_available and float(snapshot["avg_dir_return"]) >= 0.0 else (NEGATIVE if resolved_metrics_available else TEXT_MUTED)
    best_setup_label = _setup_pocket_label(setup_pocket)
    best_setup_subtext = (
        f"{int(setup_pocket.get('completed') or 0)} completed in selected pocket"
        if bool(setup_pocket.get("available"))
        else f"Needs at least {_MIN_SETUP_POCKET_ROWS} completed signals"
    )
    selected_scope_text = "selected pocket" if bool(setup_pocket.get("available")) else "current scope"
    resolved_metrics_note = (
        f"Signals in the {selected_scope_text} that moved in the expected direction"
        if resolved_metrics_available
        else f"Appears after the {selected_scope_text} completes"
    )
    avg_dir_note = (
        f"Average move inside the {selected_scope_text}"
        if resolved_metrics_available
        else f"Appears after the {selected_scope_text} completes"
    )
    st.markdown("### Overview")
    if bool(setup_pocket.get("available")):
        st.caption(f"Best pocket: {best_setup_label}. {_follow_through_horizon_note(str(setup_pocket.get('timeframe') or timeframe_filter))}")
    else:
        st.caption(f"Best pocket is still building. {_follow_through_horizon_note(timeframe_filter)}")
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
                "label": "Avg Move",
                "value": avg_dir_value,
                "value_color": tone_dir,
                "subtext": avg_dir_note,
            },
            {
                "label": "Signals",
                "value": int(snapshot["total"]),
                "subtext": f"Signals in the {selected_scope_text}",
            },
            {
                "label": "Completed",
                "value": int(snapshot["resolved"]),
                "subtext": f"Open: {int(snapshot['open'])}",
                "badge_text": _refresh_scope_badge(
                    symbol_filter=symbol_filter,
                    timeframe_filter=timeframe_filter,
                    resolved_now=resolved_now,
                ),
                "badge_color": POSITIVE if resolved_now > 0 else TEXT_MUTED,
                "badge_tone": "positive" if resolved_now > 0 else "neutral",
            },
            {
                "label": "Best Setup",
                "value": str(setup_pocket.get("setup_display") or setup_pocket.get("setup_label") or "Building"),
                "value_color": POSITIVE if bool(setup_pocket.get("available")) else TEXT_MUTED,
                "subtext": best_setup_subtext,
            },
        ],
        columns=5,
    )
    st.markdown("### Timeframe Read")
    st.caption("Hold columns show efficient timing. Expected Path below shows the historical price route.")
    if autofilled_hold_now > 0:
        st.caption(
            f"{symbol_filter}: filled {autofilled_hold_now} timing row(s). {missing_hold_backfill} still pending."
        )
    elif missing_hold_backfill > 0:
        st.caption(
            f"{symbol_filter}: timing history is still filling. {missing_hold_backfill} pending."
        )
    else:
        st.caption(f"Timing history is up to date for {symbol_filter}.")
    if timeframe_intelligence_df.empty:
        st.caption("Timeframe read is still building.")
    else:
        st.dataframe(timeframe_intelligence_df, hide_index=True, width="stretch")
    expected_path_projection = dict(archive_decision.expected_path)
    if bool(expected_path_projection.get("available")):
        live_ref_price, live_ref_label = _fetch_expected_path_reference_price(
            fetch_ohlcv,
            str(expected_path_projection.get("symbol") or symbol_filter),
            str(expected_path_projection.get("timeframe") or timeframe_filter),
        )
        if live_ref_price is not None:
            expected_path_projection = _with_expected_path_reference_price(
                expected_path_projection,
                live_ref_price,
                live_ref_label,
            )
        st.markdown("### Expected Path")
        st.caption(
            f"{_expected_path_scope_label(expected_path_projection)}. "
            "Archive scenario from similar past signals, not a price target."
        )
        render_kpi_grid(
            st,
            items=_expected_path_kpi_items(expected_path_projection),
            columns=5,
        )
    else:
        render_insight_card(
            st,
            title="Expected Path Building",
            body_html=_expected_path_building_body_html(
                symbol_filter=symbol_filter,
                timeframe_filter=timeframe_filter,
                df_events=df_hold_archive_events,
                df_forward_windows=df_hold_archive_windows,
            ),
            tone="neutral",
        )

    if "session_bucket" in df_events.columns:
        df_events["Session"] = df_events["session_bucket"].replace("", "Unknown").fillna("Unknown")
    elif "event_time" in df_events.columns:
        df_events["Session"] = pd.to_datetime(df_events["event_time"], utc=True, errors="coerce").map(
            lambda ts: session_bucket_for_timestamp(ts) if pd.notna(ts) else "Unknown"
        )

    detail_toggle_cols = st.columns(2, gap="medium")
    with detail_toggle_cols[0]:
        show_deep_dives = st.toggle(
            "Show Details",
            value=False,
            key="signal_review_show_deep_dives",
            help="Show breakdowns by setup, session, context, and timing.",
        )
    with detail_toggle_cols[1]:
        show_data_tables = st.toggle(
            "Show Raw Tables",
            value=False,
            key="signal_review_show_data_tables",
            help="Show source rows and archive support tables.",
        )

    need_advanced_detail = bool(show_deep_dives or show_data_tables)
    if not need_advanced_detail:
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

    advanced_df_events = df_events.copy()
    learned_edges_df = pd.DataFrame()
    if show_data_tables:
        adaptive_model = build_adaptive_context_model(df_events)
        learned_edges_df = build_learning_edge_table(adaptive_model, limit=12)

    if "lead_active" in advanced_df_events.columns:
        advanced_df_events["Lead Status"] = advanced_df_events["lead_active"].fillna(0).astype(int).map({1: "Lead", 0: "No Lead"})
    if "ai_aligned" in advanced_df_events.columns:
        advanced_df_events["AI Alignment"] = advanced_df_events["ai_aligned"].fillna(0).astype(int).map({1: "Aligned", 0: "Not aligned"})
    if "market_lead_label" in advanced_df_events.columns:
        advanced_df_events["Market Lead"] = advanced_df_events["market_lead_label"].replace("", "No Clear Lead").fillna("No Clear Lead")
    if "market_regime" in advanced_df_events.columns:
        advanced_df_events["Market Regime"] = advanced_df_events["market_regime"].replace("", "Unknown").fillna("Unknown")
    if "scan_focus" in advanced_df_events.columns:
        advanced_df_events["Scan Mode"] = advanced_df_events["scan_focus"].replace("", "Unknown").fillna("Unknown")
    if "setup_confirm" in advanced_df_events.columns:
        advanced_df_events["Setup Confirm"] = advanced_df_events.apply(
            lambda row: setup_confirm_display(
                str(row.get("setup_confirm") or ""),
                action_reason=str(row.get("action_reason") or ""),
                direction=str(row.get("direction") or ""),
            ),
            axis=1,
        )
    if "market_playbook_key" in advanced_df_events.columns or "market_playbook" in advanced_df_events.columns:
        playbook_keys = advanced_df_events.get("market_playbook_key", pd.Series(index=advanced_df_events.index, dtype=object))
        playbook_display_values = advanced_df_events.get("market_playbook", pd.Series(index=advanced_df_events.index, dtype=object))
        advanced_df_events["Playbook"] = pd.Series(playbook_keys, index=advanced_df_events.index).fillna("").astype(str).str.strip().map(
            lambda value: playbook_display(value) if value else ""
        )
        fallback_playbook = pd.Series(playbook_display_values, index=advanced_df_events.index).fillna("").astype(str).str.strip()
        advanced_df_events["Playbook"] = advanced_df_events["Playbook"].where(advanced_df_events["Playbook"].ne(""), fallback_playbook)
        advanced_df_events["Playbook"] = advanced_df_events["Playbook"].replace("", "Unknown").fillna("Unknown")
        advanced_df_events["Playbook Key"] = pd.Series(playbook_keys, index=advanced_df_events.index).fillna("").astype(str).str.strip()
        advanced_df_events["Playbook Key"] = advanced_df_events["Playbook Key"].where(
            advanced_df_events["Playbook Key"].ne(""),
            fallback_playbook.map(playbook_key),
        )
        advanced_df_events["Playbook Key"] = advanced_df_events["Playbook Key"].replace("", "Unknown").fillna("Unknown")
    if "market_trade_gate_key" in advanced_df_events.columns or "market_trade_gate" in advanced_df_events.columns:
        trade_gate_keys = advanced_df_events.get("market_trade_gate_key", pd.Series(index=advanced_df_events.index, dtype=object))
        trade_gate_display_values = advanced_df_events.get("market_trade_gate", pd.Series(index=advanced_df_events.index, dtype=object))
        advanced_df_events["Market Stance"] = pd.Series(trade_gate_keys, index=advanced_df_events.index).fillna("").astype(str).str.strip().map(
            lambda value: trade_gate_display(value) if value else ""
        )
        fallback_trade_gate = pd.Series(trade_gate_display_values, index=advanced_df_events.index).fillna("").astype(str).str.strip()
        advanced_df_events["Market Stance"] = advanced_df_events["Market Stance"].where(advanced_df_events["Market Stance"].ne(""), fallback_trade_gate)
        advanced_df_events["Market Stance"] = advanced_df_events["Market Stance"].replace("", "Unknown").fillna("Unknown")
        advanced_df_events["Market Stance Key"] = pd.Series(trade_gate_keys, index=advanced_df_events.index).fillna("").astype(str).str.strip()
        advanced_df_events["Market Stance Key"] = advanced_df_events["Market Stance Key"].where(
            advanced_df_events["Market Stance Key"].ne(""),
            fallback_trade_gate.map(trade_gate_key),
        )
        advanced_df_events["Market Stance Key"] = advanced_df_events["Market Stance Key"].replace("", "Unknown").fillna("Unknown")
    if "market_no_trade_reason" in advanced_df_events.columns:
        advanced_df_events[copy_text("review.label.no_trade_reason")] = (
            advanced_df_events["market_no_trade_reason"]
            .replace("", "None")
            .fillna("None")
            .astype(str)
            .str.replace("_", " ", regex=False)
            .str.title()
        )
    if "risk_tier" in advanced_df_events.columns:
        advanced_df_events["Risk Tier"] = advanced_df_events["risk_tier"].replace("", "Unknown").fillna("Unknown")
    if "sector_tag" in advanced_df_events.columns:
        advanced_df_events["Sector"] = advanced_df_events["sector_tag"].replace("", "Other").fillna("Other")
    if "market_sector_rotation" in advanced_df_events.columns:
        advanced_df_events["Sector Rotation"] = (
            advanced_df_events["market_sector_rotation"].replace("", "Unknown").fillna("Unknown")
        )
    if "market_catalyst_state" in advanced_df_events.columns:
        advanced_df_events["Catalyst State"] = (
            advanced_df_events["market_catalyst_state"].replace("", "Unknown").fillna("Unknown")
        )
    if "market_catalyst_event" in advanced_df_events.columns:
        advanced_df_events["Catalyst Event"] = (
            advanced_df_events["market_catalyst_event"].replace("", "None").fillna("None")
        )
    if "market_catalyst_category" in advanced_df_events.columns:
        advanced_df_events["Catalyst Category"] = (
            advanced_df_events["market_catalyst_category"].replace("", "Unknown").fillna("Unknown")
        )
    if "market_catalyst_scope" in advanced_df_events.columns:
        advanced_df_events["Catalyst Scope"] = (
            advanced_df_events["market_catalyst_scope"].replace("", "Unknown").fillna("Unknown")
        )
    if "market_catalyst_tag" in advanced_df_events.columns:
        advanced_df_events["Catalyst Tag"] = (
            advanced_df_events["market_catalyst_tag"].replace("", "None").fillna("None")
        )
    if "market_catalyst_targeted" in advanced_df_events.columns:
        advanced_df_events["Catalyst Targeting"] = (
            advanced_df_events["market_catalyst_targeted"].fillna(0).astype(int).map({1: "Targeted", 0: "Market-Wide"})
        )
    if "market_catalyst_window" in advanced_df_events.columns:
        advanced_df_events["Catalyst Window"] = (
            advanced_df_events["market_catalyst_window"].replace("", "Unknown").fillna("Unknown")
        )
    if "market_flow_state" in advanced_df_events.columns:
        advanced_df_events["Flow Read"] = (
            advanced_df_events["market_flow_state"].replace("", "Unknown").fillna("Unknown")
        )
    if "Playbook" in advanced_df_events.columns and "Session" in advanced_df_events.columns:
        advanced_df_events["Playbook x Session"] = (
            advanced_df_events["Playbook"].astype(str).str.strip().replace("", "Unknown")
            + " | "
            + advanced_df_events["Session"].astype(str).str.strip().replace("", "Unknown")
        )
    if "Playbook" in advanced_df_events.columns and "Catalyst Window" in advanced_df_events.columns:
        advanced_df_events["Playbook x Catalyst Window"] = (
            advanced_df_events["Playbook"].astype(str).str.strip().replace("", "Unknown")
            + " | "
            + advanced_df_events["Catalyst Window"].astype(str).str.strip().replace("", "Unknown")
        )
    if "adaptive_edge_label" in advanced_df_events.columns:
        advanced_df_events["Learned Edge"] = (
            advanced_df_events["adaptive_edge_label"].replace("", "Unknown").fillna("Unknown")
        )
    if "archive_guardrail_label" in advanced_df_events.columns:
        advanced_df_events["History Guardrail"] = (
            advanced_df_events["archive_guardrail_label"].replace("", "Archive Clear").fillna("Archive Clear")
        )
    if "archive_guardrail_penalty" in advanced_df_events.columns:
        penalty_series = pd.to_numeric(advanced_df_events["archive_guardrail_penalty"], errors="coerce").fillna(0.0)
        advanced_df_events["History Guardrail Level"] = penalty_series.map(
            lambda value: (
                "Guardrail"
                if float(value) >= 5.0
                else ("Caution" if float(value) >= 3.0 else "Clear")
            )
        )
    if "Market Stance" in advanced_df_events.columns:
        advanced_df_events["Readiness"] = advanced_df_events.apply(
            lambda row: archived_execution_stance_label(
                trade_gate=str(row.get("Market Stance") or ""),
                adaptive_edge=str(row.get("Learned Edge") or ""),
                archive_guardrail_severity=str(row.get("History Guardrail Level") or ""),
            ),
            axis=1,
        )
    advanced_df_events = annotate_alert_footprint(advanced_df_events)

    if show_deep_dives:
        st.markdown("### Details")
        st.caption("Optional breakdowns.")
        with st.expander("Core", expanded=False):
            _render_compact_cohort_tables(
                st,
                df_events=advanced_df_events,
                build_signal_cohort_summary=build_signal_cohort_summary,
                specs=[
                    ("Setup Confirm", "By Setup Confirm"),
                    ("direction", "By Direction", "Direction"),
                    ("timeframe", "By Timeframe", "Timeframe"),
                    ("Session", "By Session"),
                    ("Playbook", "By Playbook"),
                ],
            )

        with st.expander("Context", expanded=False):
            _render_compact_cohort_tables(
                st,
                df_events=advanced_df_events,
                build_signal_cohort_summary=build_signal_cohort_summary,
                specs=[
                    ("Readiness", "By Readiness", "Readiness"),
                    ("Primary Alert", "By Lead Alert"),
                    ("Catalyst State", "By Catalyst State"),
                    ("Catalyst Window", "By Catalyst Window"),
                    ("Flow Read", "By Tape Read", "Tape Read"),
                    ("Sector", "By Sector"),
                ],
            )

        if hold_guidance_enabled:
            with st.expander("Timing Detail", expanded=False):
                st.caption(
                    "Coin-specific timing reads from completed signals."
                )
                _render_hold_window_cohort_tables(
                    st,
                    df_events=advanced_df_events,
                    df_forward_windows=df_forward_windows,
                    build_hold_window_cohort_summary=build_hold_window_cohort_summary,
                    specs=[
                        ("Setup Confirm", "By Setup Confirm"),
                        ("direction", "By Direction"),
                        ("Playbook", "By Playbook"),
                    ],
                )

    if show_data_tables:
        recent_cols = [
            "event_time",
            "symbol",
            "timeframe",
            "Primary Alert",
            "Alert Footprint",
            "Scan Mode",
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
            "Readiness",
            "Market Regime",
            "Market Stance",
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
            "confidence",
            "ai_confidence",
            "plan_outcome",
            "directional_return_pct",
            "favorable_excursion_pct",
            "adverse_excursion_pct",
            "status",
        ]
        recent_df = advanced_df_events[[c for c in recent_cols if c in advanced_df_events.columns]].copy().head(int(recent_table_limit))
        if "event_time" in recent_df.columns:
            recent_df["event_time"] = pd.to_datetime(recent_df["event_time"], errors="coerce").dt.strftime("%Y-%m-%d %H:%M")
        rename_map = {
            "event_time": "Signal Time",
            "symbol": "Coin",
            "timeframe": "TF",
            "Primary Alert": "Lead Alert",
            "Lead Status": "Lead Signal",
            "Market Lead": "AI View",
            "Flow Read": "Tape Read",
            "Learned Edge": "Archive Read",
            "direction": "Direction",
            "confidence": "Confidence",
            "ai_confidence": "AI Confidence",
            "plan_outcome": "Plan Outcome",
            "directional_return_pct": "Move %",
            "favorable_excursion_pct": "Best Move %",
            "adverse_excursion_pct": "Drawdown %",
            "status": "Status",
            "adaptive_edge_score": "Archive Score",
            "actionable_frame_score": "Hunt Score",
            "actionable_setup_score": "Setup Score",
            "actionable_context_score": "Context Score",
            "actionable_tactical_score": "Tactical Score",
        }
        recent_df = recent_df.rename(columns=rename_map)
        st.markdown("### Raw Tables")
        st.caption("Source rows for auditing.")
        st.markdown("#### Recent Signals")
        st.caption(f"Showing latest {min(int(recent_table_limit), int(len(df_events)))} rows in this scope.")
        st.dataframe(recent_df.round(2), hide_index=True, width="stretch")

        df_alerts = fetch_market_alerts_df(limit=100, source="Market", db_path=db_path)
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
