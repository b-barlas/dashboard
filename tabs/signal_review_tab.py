from __future__ import annotations

from collections.abc import Mapping
import html
import time

import pandas as pd
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
            return "Open / Not Closed"
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
            return "Open / Not Closed"
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
        "Shows timing, expected path, and execution quality for this coin"
    )


def _archive_direction_key(value: object) -> str:
    side = str(value or "").strip().upper()
    if side in {"UPSIDE", "LONG", "BUY", "BULLISH", "STRONG BUY"}:
        return "UPSIDE"
    if side in {"DOWNSIDE", "SHORT", "SELL", "BEARISH", "STRONG SELL"}:
        return "DOWNSIDE"
    return ""


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
    path_counts = (
        event_scope.groupby(["symbol", "timeframe", "__direction_key"], dropna=False)["signal_key"]
        .nunique()
        .reset_index(name="PathSamples")
    )
    out = out.drop(columns=["PathSamples"], errors="ignore").merge(
        path_counts,
        on=["symbol", "timeframe", "__direction_key"],
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
        board["Follow-Through"] = board["FollowThroughPct"].map(lambda value: f"{float(value):.1f}%")
        board["Avg Move"] = board["AvgDirReturnPct"].map(lambda value: f"{float(value):+.2f}%")
        board = board.rename(columns={"symbol": "Coin"})
        return board[["Coin", "Mode", "Follow-Through", "Resolved", "Best TF", "Avg Move"]].reset_index(drop=True)

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
        .head(1)[["symbol", "timeframe"]]
        .rename(columns={"timeframe": "best_timeframe"})
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
    board["Follow-Through"] = board["AvgFollowThroughPct"].map(lambda value: f"{float(value):.1f}%")
    board["Avg Move"] = board["AvgDirReturnPct"].map(lambda value: f"{float(value):+.2f}%")
    board = board.rename(columns={"symbol": "Coin"})
    return board[["Coin", "Mode", "Follow-Through", "Resolved", "Best TF", "Avg Move"]].reset_index(drop=True)


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
    try:
        out = float(value)
    except Exception:
        return float(default)
    if pd.isna(out):
        return float(default)
    return float(out)


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


def _projection_direction_key(value: object) -> str:
    return _archive_direction_key(value)


def _expected_path_read_quality(sample: int) -> str:
    if int(sample) >= 32:
        return "Strong"
    if int(sample) >= 16:
        return "Good"
    return "Thin"


def _with_expected_path_reference_price(
    snapshot: Mapping[str, object],
    reference_price: object,
    reference_label: str = "latest close",
) -> dict[str, object]:
    out = dict(snapshot)
    ref_price = _projection_float(reference_price, 0.0)
    if ref_price <= 0:
        return out

    direction_key = str(out.get("direction") or "").strip().upper()
    lower_return = abs(_projection_float(out.get("best_zone_low_pct"), 0.0))
    upper_return = abs(_projection_float(out.get("best_zone_high_pct"), 0.0))
    lower_return, upper_return = sorted([lower_return, upper_return])
    normal_pullback = max(0.0, abs(_projection_float(out.get("normal_pullback_pct"), 0.0)))

    if direction_key == "DOWNSIDE":
        p1 = ref_price * (1.0 - lower_return / 100.0)
        p2 = ref_price * (1.0 - upper_return / 100.0)
        pullback_price = ref_price * (1.0 + normal_pullback / 100.0)
    else:
        p1 = ref_price * (1.0 + lower_return / 100.0)
        p2 = ref_price * (1.0 + upper_return / 100.0)
        pullback_price = ref_price * (1.0 - normal_pullback / 100.0)

    price_low, price_high = sorted([p1, p2])
    out.update(
        {
            "reference_price": ref_price,
            "reference_price_label": str(reference_label or "latest close").strip() or "latest close",
            "price_zone_label": f"{_format_projection_price(price_low)} - {_format_projection_price(price_high)}",
            "pullback_price_label": _format_projection_price(pullback_price),
        }
    )
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


def _format_archive_reference_label(ts: object, *, stale: bool) -> str:
    prefix = "last archived price" if stale else "latest archived price"
    parsed = pd.to_datetime(ts, utc=True, errors="coerce")
    if pd.isna(parsed):
        return prefix
    return f"{prefix}, {pd.Timestamp(parsed).strftime('%b %d')}"


def _expected_path_timing_label(snapshot: Mapping[str, object]) -> str:
    bars = int(snapshot.get("best_bar") or 0)
    timeframe = str(snapshot.get("timeframe") or "").strip().upper()
    if bars <= 0 or not timeframe:
        return ""
    candle_label = "candle" if bars == 1 else "candles"
    return f"next {bars} {candle_label} on {timeframe}"


def _expected_path_reference_price(
    df_events: pd.DataFrame,
    *,
    timeframe: str,
    direction: str,
    now: object | None = None,
    max_age_hours: int = 48,
) -> tuple[float | None, str]:
    if df_events is None or df_events.empty or "price" not in df_events.columns:
        return None, ""
    d = df_events.copy()
    if "timeframe" in d.columns:
        d = d[d["timeframe"].fillna("").astype(str).str.strip().str.lower().eq(str(timeframe).lower())].copy()
    if "direction" in d.columns:
        d["__direction"] = d["direction"].map(_projection_direction_key)
        d = d[d["__direction"].eq(str(direction).upper())].copy()
    d["__price"] = pd.to_numeric(d.get("price"), errors="coerce")
    d = d[d["__price"].notna() & (d["__price"] > 0)].copy()
    if d.empty:
        return None, ""
    if "event_time" in d.columns:
        d["__event_ts"] = pd.to_datetime(d["event_time"], utc=True, errors="coerce")
        d = d.sort_values("__event_ts", ascending=False)
        latest = d.iloc[0]
        ts = latest.get("__event_ts")
        if pd.notna(ts):
            now_ts = pd.to_datetime(now, utc=True, errors="coerce") if now is not None else pd.Timestamp.now(tz="UTC")
            if pd.notna(now_ts):
                age_hours = max(0.0, float((pd.Timestamp(now_ts) - pd.Timestamp(ts)).total_seconds()) / 3600.0)
                return (
                    float(latest["__price"]),
                    _format_archive_reference_label(ts, stale=age_hours > float(max_age_hours)),
                )
    latest_price = float(d.iloc[0]["__price"])
    return latest_price, "latest archived price"


def _expected_path_snapshot_for_group(
    *,
    events_df: pd.DataFrame,
    windows_df: pd.DataFrame,
    symbol_filter: str,
    timeframe: str,
    direction: str,
    min_samples: int,
    now: object | None = None,
) -> dict[str, object]:
    empty = {
        "available": False,
        "symbol": str(symbol_filter or "").strip().upper(),
        "timeframe": str(timeframe or "").strip().lower(),
        "direction": str(direction or "").strip().upper(),
        "sample": 0,
    }
    if events_df is None or events_df.empty or windows_df is None or windows_df.empty:
        return empty
    required_event_cols = {"signal_key", "timeframe", "direction"}
    required_window_cols = {"signal_key", "bars_ahead", "directional_return_pct", "adverse_excursion_pct"}
    if not required_event_cols.issubset(events_df.columns) or not required_window_cols.issubset(windows_df.columns):
        return empty

    e = events_df.copy()
    e["signal_key"] = e["signal_key"].fillna("").astype(str).str.strip()
    e["timeframe"] = e["timeframe"].fillna("").astype(str).str.strip().str.lower()
    e["__direction"] = e["direction"].map(_projection_direction_key)
    status = e.get("status", pd.Series(index=e.index, dtype=object)).fillna("").astype(str).str.upper()
    e = e[
        e["signal_key"].ne("")
        & e["timeframe"].eq(str(timeframe).strip().lower())
        & e["__direction"].eq(str(direction).strip().upper())
        & status.eq("RESOLVED")
    ].copy()
    if e.empty:
        return empty

    w = windows_df.copy()
    w["signal_key"] = w["signal_key"].fillna("").astype(str).str.strip()
    w["bars_ahead"] = pd.to_numeric(w["bars_ahead"], errors="coerce")
    w["directional_return_pct"] = pd.to_numeric(w["directional_return_pct"], errors="coerce")
    w["adverse_excursion_pct"] = pd.to_numeric(w["adverse_excursion_pct"], errors="coerce")
    w = w[
        w["signal_key"].ne("")
        & w["bars_ahead"].notna()
        & w["directional_return_pct"].notna()
        & w["adverse_excursion_pct"].notna()
    ].copy()
    if w.empty:
        return empty

    merged = w.merge(e[["signal_key", "timeframe", "__direction", "price", "event_time"] if {"price", "event_time"}.issubset(e.columns) else ["signal_key", "timeframe", "__direction"]], on="signal_key", how="inner")
    if merged.empty:
        return empty
    sample = int(merged["signal_key"].nunique())
    if sample < int(min_samples):
        out = dict(empty)
        out.update({"sample": sample, "read_quality": "Building"})
        return out

    by_bar_rows: list[dict[str, object]] = []
    for bar, group in merged.groupby("bars_ahead", dropna=True):
        bar_sample = int(group["signal_key"].nunique())
        if bar_sample < max(3, min(int(min_samples), 8)):
            continue
        returns = pd.to_numeric(group["directional_return_pct"], errors="coerce").dropna()
        adverse = pd.to_numeric(group["adverse_excursion_pct"], errors="coerce").dropna()
        if returns.empty:
            continue
        median_return = float(returns.median())
        if median_return <= 0.0:
            continue
        adverse_median = float(adverse.median()) if not adverse.empty else 0.0
        follow_through = float((returns > 0.0).mean() * 100.0)
        path_score = median_return - (0.30 * adverse_median) + ((follow_through - 50.0) / 100.0)
        by_bar_rows.append(
            {
                "bars_ahead": int(bar),
                "sample": bar_sample,
                "median_return": median_return,
                "lower_return": float(returns.quantile(0.40)),
                "upper_return": float(returns.quantile(0.75)),
                "normal_pullback": max(0.0, float(adverse.quantile(0.65)) if not adverse.empty else 0.0),
                "follow_through": follow_through,
                "path_score": path_score,
            }
        )
    if not by_bar_rows:
        out = dict(empty)
        out.update({"sample": sample, "read_quality": _expected_path_read_quality(sample)})
        return out

    by_bar_rows.sort(key=lambda row: (float(row["path_score"]), float(row["median_return"]), int(row["sample"])), reverse=True)
    best = by_bar_rows[0]
    best_bar = int(best["bars_ahead"])
    best_median = float(best["median_return"])
    fade_after_bar = 0
    for row in sorted([r for r in by_bar_rows if int(r["bars_ahead"]) > best_bar], key=lambda r: int(r["bars_ahead"])):
        if float(row["median_return"]) <= max(0.05, best_median * 0.65) or float(row["follow_through"]) < 50.0:
            fade_after_bar = int(row["bars_ahead"])
            break

    ref_price, ref_label = _expected_path_reference_price(
        merged,
        timeframe=str(timeframe).strip().lower(),
        direction=str(direction).strip().upper(),
        now=now,
    )
    lower_return = float(min(best["lower_return"], best["upper_return"]))
    upper_return = float(max(best["lower_return"], best["upper_return"]))
    normal_pullback = float(best["normal_pullback"])

    score = (
        float(best["path_score"])
        + min(1.5, sample / 24.0)
        + (0.25 if _expected_path_read_quality(sample) == "Strong" else 0.0)
    )
    snapshot = {
        "available": True,
        "symbol": str(symbol_filter or "").strip().upper(),
        "timeframe": str(timeframe or "").strip().lower(),
        "direction": str(direction or "").strip().upper(),
        "sample": sample,
        "read_quality": _expected_path_read_quality(sample),
        "best_bar": best_bar,
        "fade_after_bar": fade_after_bar,
        "best_zone_low_pct": lower_return,
        "best_zone_high_pct": upper_return,
        "normal_pullback_pct": normal_pullback,
        "follow_through_pct": float(best["follow_through"]),
        "score": float(score),
        "reference_price": ref_price,
        "reference_price_label": ref_label,
        "price_zone_label": "",
        "pullback_price_label": "",
    }
    return _with_expected_path_reference_price(snapshot, ref_price, ref_label)


def _build_expected_path_projection(
    *,
    df_events: pd.DataFrame,
    df_forward_windows: pd.DataFrame,
    symbol_filter: str,
    timeframe_filter: str,
    min_samples: int = 8,
    now: object | None = None,
) -> dict[str, object]:
    empty = {"available": False, "symbol": str(symbol_filter or "").strip().upper(), "sample": 0}
    if df_events is None or df_events.empty or df_forward_windows is None or df_forward_windows.empty:
        return empty
    if "timeframe" not in df_events.columns or "direction" not in df_events.columns:
        return empty
    timeframe_text = str(timeframe_filter or "").strip().lower()
    available_timeframes = set(df_events["timeframe"].fillna("").astype(str).str.strip().str.lower().tolist())
    available_timeframes.discard("")
    timeframes = _ordered_timeframe_scope(
        timeframe_filter=timeframe_filter,
        available_timeframes=available_timeframes,
    )
    if timeframe_text and timeframe_text != "all":
        timeframes = [timeframe_text]
    snapshots: list[dict[str, object]] = []
    for timeframe in timeframes:
        for direction in ("UPSIDE", "DOWNSIDE"):
            snap = _expected_path_snapshot_for_group(
                events_df=df_events,
                windows_df=df_forward_windows,
                symbol_filter=symbol_filter,
                timeframe=timeframe,
                direction=direction,
                min_samples=min_samples,
                now=now,
            )
            if bool(snap.get("available")):
                snapshots.append(snap)
    if not snapshots:
        return empty
    snapshots.sort(
        key=lambda snap: (
            float(snap.get("score") or 0.0),
            float(snap.get("follow_through_pct") or 0.0),
            int(snap.get("sample") or 0),
        ),
        reverse=True,
    )
    return snapshots[0]


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
    if price_zone and reference_price > 0:
        return (
            f"<b>{symbol} {timeframe} {direction_escaped}</b><br>"
            f"Reference price: <b>{_format_projection_price(reference_price)}</b> "
            f"<span style='opacity:0.70;'>({reference_label})</span><br>"
            f"Expected zone: <b>{html.escape(price_zone)}</b><br>"
            f"{timing_html.replace('Timing:', 'Best path window:', 1)}"
            f"Move from that price: <b>{html.escape(move_pct)}</b> "
            f"<span style='opacity:0.70;'>({html.escape(move_hint)})</span><br>"
            f"Normal shakeout: <b>{html.escape(pullback_price or pullback)}</b> "
            f"<span style='opacity:0.70;'>({html.escape(pullback_pct_label)}, can still be normal)</span><br>"
            f"Path weakens after: <b>{html.escape(fade_text)}</b><br>"
            f"History depth: <b>{quality}</b> ({sample} similar signals)<br>"
            "<span style='opacity:0.72;'>Price path from similar past signals, not a price target. Hold window above is the efficiency read.</span>"
        )
    return (
        f"<b>{symbol} {timeframe} {direction_escaped}</b><br>"
        f"Expected move: <b>{html.escape(move_pct)}</b> "
        f"<span style='opacity:0.70;'>({html.escape(move_hint)})</span><br>"
        f"{timing_html.replace('Timing:', 'Best path window:', 1)}"
        f"Normal shakeout: <b>{pullback}</b> "
        "<span style='opacity:0.70;'>(can still be normal)</span><br>"
        f"Path weakens after: <b>{html.escape(fade_text)}</b><br>"
        f"History depth: <b>{quality}</b> ({sample} similar signals)<br>"
        "<span style='opacity:0.72;'>Price path from similar past signals, not a price target. Hold window above is the efficiency read.</span>"
    )


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
    for timeframe_key in timeframe_scope:
        if str(timeframe_filter or "").strip() != "All" and isinstance(base_events, pd.DataFrame):
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
                    f"Still building: <b>{preview}</b>{extra}. More completed signals or closed trades needed."
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
            "Execution journal is empty. Mark taken trades to compare your execution with the system.",
            "neutral",
        )
    if taken_resolved < _MIN_EXECUTION_ARCHIVE_ROWS or actual_closed < _MIN_EXECUTION_ARCHIVE_ROWS:
        return (
            "Execution sample is still thin. Use this as a rough read for now.",
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
                "No signals are in this view yet, so execution review is still waiting for history.",
            )
        ]

    if overlay_marked <= 0:
        cards.append(
            _archive_building_card(
                "Manual Marking",
                "No setups are tagged yet. Mark them as <b>Taken</b>, <b>Skipped</b>, or <b>Observed</b> before reading execution quality.",
            )
        )
    elif overlay_coverage_pct < 50.0:
        cards.append(
            {
                "title": "Manual Marking",
                "body_html": (
                    f"Only <b>{overlay_marked}/{total}</b> setups in this view are tagged "
                    f"(<b>{overlay_coverage_pct:.1f}% coverage</b>). "
                    "Tag more setups first so the execution read becomes reliable."
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
                "No setups are marked as <b>Taken</b> yet. Add taken trades before reading real execution quality.",
            )
        )
    elif actual_closed < _MIN_EXECUTION_ARCHIVE_ROWS:
        cards.append(
            _archive_building_card(
                "Trade Journal",
                (
                    f"<b>{taken}</b> taken setup(s) are marked, but only <b>{actual_closed}</b> closed trade(s) have an exit. "
                    "Add real exits before reading PnL or win rate."
                ),
            )
        )
    elif journal_coverage_pct < 60.0:
        cards.append(
            {
                "title": "Trade Journal",
                "body_html": (
                    f"Closed-trade journaling is improving, but only <b>{journal_coverage_pct:.1f}% of taken setups</b> in this view have a recorded exit. "
                    "A few more exits will make this read stronger."
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
                "Execution Quality",
                "Execution quality is still building. Add more completed taken setups and closed trades before comparing your results with the signal.",
            )
        )
    else:
        if execution_gap_pct >= 0.5:
            tone = "positive"
            edge_read = (
                f"Your closed trades are running <b>{execution_gap_pct:+.2f}% ahead</b> of the matching signal move."
            )
        elif execution_gap_pct <= -0.5:
            tone = "warning"
            edge_read = (
                f"Your closed trades are trailing the matching signal move by <b>{execution_gap_pct:+.2f}%</b>."
            )
        else:
            tone = "neutral"
            edge_read = (
                f"Your closed trades are broadly aligned with the matching signal move at <b>{execution_gap_pct:+.2f}%</b>."
            )
        skipped_read = (
            f" Skipped setups that worked: <b>{skipped_winners}</b> out of <b>{skipped_resolved}</b> "
            f"(<b>{skipped_winner_rate:.1f}%</b>)."
            if skipped_resolved > 0
            else ""
        )
        cards.append(
            {
                "title": "Execution Quality",
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
                    "title": "Journal Complete",
                    "body_html": (
                        f"Trade journal is usable here: <b>{actual_closed}</b> closed trade(s) across "
                        f"<b>{taken}</b> taken setups (<b>{journal_coverage_pct:.1f}% coverage</b>)."
                    ),
                    "tone": "positive",
                }
            )
        elif journal_coverage_pct > 0.0:
            fail_cards.append(
                {
                    "title": "Thin Journal",
                    "body_html": (
                        f"Only <b>{journal_coverage_pct:.1f}%</b> of taken setups have a recorded exit "
                        f"(<b>{actual_closed}</b> closed trade(s) across <b>{taken}</b> taken setups). "
                        "Add more exits before reading real execution quality."
                    ),
                    "tone": "warning",
                }
            )

    if taken_resolved >= _MIN_EXECUTION_ARCHIVE_ROWS and actual_closed >= _MIN_EXECUTION_ARCHIVE_ROWS:
        if execution_gap_pct >= 0.5:
            works_cards.append(
                {
                    "title": "Execution Quality",
                    "body_html": (
                        f"Taken trades are running <b>{execution_gap_pct:+.2f}%</b> ahead of the matching signal move, "
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
                        f"Taken trades are trailing the matching signal move by <b>{execution_gap_pct:+.2f}%</b>. "
                        "The signal worked better than the real trade result."
                    ),
                    "tone": "warning",
                }
            )

    if skipped_resolved >= _MIN_EXECUTION_ARCHIVE_ROWS and skipped_winner_rate >= 40.0:
        fail_cards.append(
            {
                "title": "Missed Winners",
                "body_html": (
                    f"<b>{skipped_winners}</b> of <b>{skipped_resolved}</b> skipped setups ended up working "
                    f"(<b>{skipped_winner_rate:.1f}%</b>). "
                    "Your selection filter may be too strict in this pocket."
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
            "title": "Best Hold Window",
            "body_html": (
                f"<b>{symbol_label} {best_record['timeframe']} {best_record['direction']}</b> has the cleanest hold profile so far "
                f"({best_record['best_label']}, {best_record['best_style']}, "
                f"<b>{best_record['follow_through_pct']:.1f}%</b> follow-through, "
                f"<b>{best_record['avg_dir_return_pct']:+.2f}%</b> avg move across "
                    f"<b>{int(best_record['sample'])}</b> similar signals)."
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
                "title": "Weakest Hold Window",
                "body_html": (
                    f"<b>{symbol_label} {weakest_record['timeframe']} {weakest_record['direction']}</b> is the weakest current hold profile "
                    f"({weakest_record['best_label']}, {weakest_record['best_style']}, "
                    f"<b>{weakest_record['follow_through_pct']:.1f}%</b> follow-through, "
                    f"<b>{weakest_record['avg_dir_return_pct']:+.2f}%</b> avg move across "
                    f"<b>{int(weakest_record['sample'])}</b> similar signals)."
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
    st.caption("Mark what you did and compare it with the system signal.")
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
            "Tag the setup first. If you took it, add the real entry and exit."
            "</div>"
        ),
        unsafe_allow_html=True,
    )
    review_cols = st.columns([1.2, 1.0], gap="medium")
    render_kpi_grid(
        st,
        items=[
            {
                "label": "Tagged Setups",
                "value": f"{trade_overlay_count}/{total_signals}",
                "value_color": positive_color if trade_overlay_count > 0 else muted_color,
                "subtext": f"{overlay_coverage_pct:.1f}% of this view is tagged",
            },
            {
                "label": "Journal Complete",
                "value": f"{closed_trade_count}/{taken_count}",
                "value_color": positive_color if closed_trade_count > 0 else muted_color,
                "subtext": (
                    f"{journal_coverage_pct:.1f}% of taken setups have an exit"
                    if taken_count > 0
                    else "No taken setups in this view yet"
                ),
            },
            {
                "label": "Real vs Signal",
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
                    "Closed trade result vs matching signal move"
                    if closed_trade_count > 0
                    else "Appears after closed trades"
                ),
            },
            {
                "label": "Skipped Winners",
                "value": skipped_winners,
                "value_color": warning_color if skipped_winners > 0 else muted_color,
                "subtext": (
                    f"{skipped_winner_rate:.1f}% of skipped setups worked"
                    if skipped_resolved > 0
                    else "Appears after skipped setups resolve"
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
                    "Average result across closed trades"
                    if closed_trade_count > 0
                    else "Appears after closed trades"
                ),
            },
            {
                "label": "Taken Completed",
                "value": int(execution_snapshot.get("taken_resolved", 0.0) or 0.0),
                "value_color": positive_color if int(execution_snapshot.get("taken_resolved", 0.0) or 0.0) > 0 else muted_color,
                "subtext": "Taken setups that reached the timing window",
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
                    "Follow-through rate across completed taken setups"
                    if int(execution_snapshot.get("taken_resolved", 0.0) or 0.0) > 0
                    else "Appears after taken setups complete"
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
                    "Win rate across closed trades"
                    if closed_trade_count > 0
                    else "Appears after closed trades"
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
            "Review logged market signals, expected path, and your execution quality. "
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
                "Type a <b>coin</b> above. Signal Archive will load its timing, expected path, and execution read."
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
    active_alerts_count = int(count_market_alerts(active_only=True, source="Market", db_path=db_path))
    adaptive_archive_df = fetch_signal_events_df(
        limit=analysis_limit,
        status="RESOLVED",
        source="Market",
        timeframe=None if timeframe_filter == "All" else timeframe_filter,
        symbol=symbol_filter,
        decision_version=current_market_version,
        db_path=db_path,
    )
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
        st.markdown("### Best Signal Leaderboard")
        st.caption(
            "Top 10 coins with enough history. Select a row to open the full coin read."
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
        st.info("No signals match this coin/timeframe yet. Try All or let Market log more history.")
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
    hold_archive_signal_keys = []
    df_hold_archive_windows = pd.DataFrame()
    timeframe_frames: list[dict[str, object]] = []
    timeframe_summary_df = pd.DataFrame()
    hold_guidance_rows: list[dict[str, object]] = []
    timeframe_intelligence_df = pd.DataFrame()
    missing_hold_backfill = 0
    hold_autofill_scope_key = _hold_autofill_scope_key(
        symbol_filter=symbol_filter,
        timeframe_filter=timeframe_filter,
        decision_version=current_market_version,
    )
    if hold_guidance_enabled:
        hold_archive_signal_keys = (
            df_hold_archive_events["signal_key"].fillna("").astype(str).str.strip().tolist()
            if "signal_key" in df_hold_archive_events.columns
            else []
        )
        df_hold_archive_windows = (
            fetch_signal_forward_windows_df(
                signal_keys=hold_archive_signal_keys,
                db_path=db_path,
            )
            if hold_archive_signal_keys
            else pd.DataFrame()
        )
        missing_hold_backfill = _missing_hold_backfill_count(df_hold_archive_events, df_hold_archive_windows)
        timeframe_frames = _load_coin_timeframe_frames(
            fetch_signal_events_df=fetch_signal_events_df,
            fetch_signal_forward_windows_df=fetch_signal_forward_windows_df,
            symbol_filter=symbol_filter,
            timeframe_filter=timeframe_filter,
            status_filter="Resolved",
            current_market_version=current_market_version,
            analysis_limit=int(analysis_limit),
            db_path=db_path,
            base_events=df_events,
        )
        timeframe_intelligence_df, timeframe_summary_df, hold_guidance_rows = _build_coin_timeframe_intelligence_bundle(
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
            timeframe_frames = _load_coin_timeframe_frames(
                fetch_signal_events_df=fetch_signal_events_df,
                fetch_signal_forward_windows_df=fetch_signal_forward_windows_df,
                symbol_filter=symbol_filter,
                timeframe_filter=timeframe_filter,
                status_filter="Resolved",
                current_market_version=current_market_version,
                analysis_limit=int(analysis_limit),
                db_path=db_path,
                base_events=df_events,
            )
            timeframe_intelligence_df, timeframe_summary_df, hold_guidance_rows = _build_coin_timeframe_intelligence_bundle(
                timeframe_frames=timeframe_frames,
                build_signal_cohort_summary=build_signal_cohort_summary,
                build_hold_window_intelligence=build_hold_window_intelligence,
            )
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
        "Completed signals that moved in the expected direction"
        if resolved_metrics_available
        else "Appears after signals complete"
    )
    avg_dir_note = (
        "Average move after the timing window"
        if resolved_metrics_available
        else "Appears after signals complete"
    )
    avg_mae_note = (
        "Average worst move against the signal"
        if resolved_metrics_available
        else "Appears after signals complete"
    )
    st.markdown("### Overview")
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
                "label": "Avg Move",
                "value": avg_dir_value,
                "value_color": tone_dir,
                "subtext": avg_dir_note,
            },
            {
                "label": "Signals",
                "value": int(snapshot["total"]),
                "subtext": "Signals loaded here",
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
        ],
        columns=4,
    )
    render_kpi_grid(
        st,
        items=[
            {
                "label": "Taken",
                "value": taken_count,
                "value_color": POSITIVE if taken_count > 0 else TEXT_MUTED,
                "subtext": "Setups marked Taken",
            },
            {
                "label": "Closed Trades",
                "value": actual_closed,
                "value_color": POSITIVE if actual_closed > 0 else TEXT_MUTED,
                "subtext": "Taken trades with exits",
            },
            {
                "label": "Avg Drawdown",
                "value": avg_mae_value,
                "value_color": tone_mae,
                "subtext": avg_mae_note,
            },
            {
                "label": "Live Alerts",
                "value": active_alerts_count,
                "value_color": WARNING if active_alerts_count else TEXT_MUTED,
                "subtext": "Active now, outside this archive view",
            },
        ],
        columns=4,
    )
    execution_vs_system_note, execution_vs_system_tone = _execution_vs_system_note(execution_snapshot)
    render_insight_card(
        st,
        title="Execution Check",
        body_html=execution_vs_system_note,
        tone=execution_vs_system_tone,
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
    expected_path_projection = _build_expected_path_projection(
        df_events=df_hold_archive_events,
        df_forward_windows=df_hold_archive_windows,
        symbol_filter=symbol_filter,
        timeframe_filter=timeframe_filter,
    )
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
        render_insight_card(
            st,
            title="Expected Path",
            body_html=_expected_path_body_html(expected_path_projection),
            tone=(
                "positive"
                if float(expected_path_projection.get("follow_through_pct") or 0.0) >= 55.0
                else "neutral"
            ),
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

    session_summary_df = build_signal_cohort_summary(df_events, "Session") if "Session" in df_events.columns else pd.DataFrame()
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

    qualified_timeframe_df = _qualified_summary_rows(
        timeframe_summary_df,
        count_field="Resolved",
        min_count=_MIN_SIGNAL_ARCHIVE_ROWS,
    )
    if len(qualified_timeframe_df) >= 2:
        strongest_timeframe = qualified_timeframe_df.sort_values(
            ["FollowThroughPct", "AvgDirReturnPct", "Resolved"], ascending=[False, False, False]
        ).iloc[0]
        weakest_timeframe = qualified_timeframe_df.sort_values(
            ["FollowThroughPct", "AvgDirReturnPct", "Resolved"], ascending=[True, True, False]
        ).iloc[0]
        works_cards.append(
            {
                "title": "Best Timeframe",
                "body_html": (
                    f"<b>{str(strongest_timeframe['timeframe']).upper()}</b> is the cleanest current pocket for <b>{symbol_filter}</b> "
                    f"({float(strongest_timeframe['FollowThroughPct']):.1f}% follow-through, "
                    f"<b>{float(strongest_timeframe['AvgDirReturnPct']):+.2f}%</b> avg move across "
                    f"{int(strongest_timeframe['Resolved'])} completed signals)."
                ),
                "tone": "positive" if float(strongest_timeframe["AvgDirReturnPct"]) >= 0.0 else "neutral",
            }
        )
        fail_cards.append(
            {
                "title": "Weakest Timeframe",
                "body_html": (
                    f"<b>{str(weakest_timeframe['timeframe']).upper()}</b> is currently the weakest pocket for <b>{symbol_filter}</b> "
                    f"({float(weakest_timeframe['FollowThroughPct']):.1f}% follow-through, "
                    f"<b>{float(weakest_timeframe['AvgDirReturnPct']):+.2f}%</b> avg move across "
                    f"{int(weakest_timeframe['Resolved'])} completed signals)."
                ),
                "tone": "warning" if float(weakest_timeframe["FollowThroughPct"]) < 45.0 or float(weakest_timeframe["AvgDirReturnPct"]) < 0.0 else "neutral",
            }
        )
    elif len(qualified_timeframe_df) == 1:
        only_timeframe = qualified_timeframe_df.iloc[0]
        works_cards.append(
            _archive_building_card(
                "Timeframe Spread",
                (
                    f"Only <b>{str(only_timeframe['timeframe']).upper()}</b> has enough history for <b>{symbol_filter}</b> so far "
                    f"({int(only_timeframe['Resolved'])} completed signals). We need more timeframe variety before ranking strongest vs weakest."
                ),
            )
        )
    else:
        works_cards.append(
            _archive_building_card(
                "Timeframe Spread",
                "Timeframe history is still building for this coin. We need more completed signals before ranking the cleanest pockets.",
            )
        )

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
                    f"<b>{best_execution_row['Session']}</b> is converting best in real execution for <b>{symbol_filter}</b> "
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
                    f"Trade journal is still building. On the signal side, <b>{best_follow_row['Session']}</b> "
                    f"is currently the cleanest session for <b>{symbol_filter}</b> "
                    f"({float(best_follow_row['FollowThroughPct']):.1f}% follow-through across "
                    f"{int(best_follow_row['Resolved'])} completed signals)."
                ),
                "tone": "neutral",
            }
        )

    works_cards = _prepare_section_cards(works_cards, max_actionable=3)
    fail_cards = _prepare_section_cards(fail_cards, max_actionable=3)

    st.markdown("### What Works")
    st.caption("Strongest reads for this coin.")
    _render_insight_card_grid(st, works_cards, columns=3)

    st.markdown("### What Needs Care")
    st.caption("Areas to treat carefully.")
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
        advanced_df_events["Execution Readiness"] = advanced_df_events.apply(
            lambda row: archived_execution_stance_label(
                trade_gate=str(row.get("Market Stance") or ""),
                adaptive_edge=str(row.get("Learned Edge") or ""),
                archive_guardrail_severity=str(row.get("History Guardrail Level") or ""),
            ),
            axis=1,
        )
    if "trade_decision" in advanced_df_events.columns:
        advanced_df_events["Trade Decision"] = (
            advanced_df_events["trade_decision"].replace("", "Unmarked").fillna("Unmarked")
        )
    if "actual_trade_status" in advanced_df_events.columns:
        advanced_df_events["Actual Trade Status"] = (
            advanced_df_events["actual_trade_status"].replace("", "Not Journaled").fillna("Not Journaled")
        )
    if "actual_exit_reason" in advanced_df_events.columns:
        advanced_df_events["Actual Exit Reason"] = (
            advanced_df_events["actual_exit_reason"].replace("", "Open / Unset").fillna("Open / Unset")
        )
    advanced_df_events = annotate_alert_footprint(advanced_df_events)
    advanced_df_events = _annotate_actual_hold_style(advanced_df_events)
    advanced_df_events = _annotate_actual_exit_quality(advanced_df_events)

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
                    ("Primary Alert", "By Lead Alert"),
                    ("Catalyst State", "By Catalyst State"),
                    ("Catalyst Window", "By Catalyst Window"),
                    ("Flow Read", "By Tape Read", "Tape Read"),
                    ("Sector", "By Sector"),
                ],
            )

        with st.expander("Execution", expanded=False):
            _render_compact_cohort_tables(
                st,
                df_events=advanced_df_events,
                build_signal_cohort_summary=build_signal_cohort_summary,
                specs=[
                    ("Execution Readiness", "By Tradeability", "Tradeability"),
                    ("Trade Decision", "By Action Taken", "Action Taken"),
                    ("Actual Trade Status", "By Journal Status", "Journal Status"),
                    ("Hold Style", "By Hold Style"),
                    ("Exit Quality", "By Exit Quality"),
                    ("Actual Exit Reason", "By Exit Reason", "Exit Reason"),
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
            "Execution Readiness",
            "Market Regime",
            "Market Stance",
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
        recent_df = advanced_df_events[[c for c in recent_cols if c in advanced_df_events.columns]].copy().head(int(recent_table_limit))
        if "actual_trade_side" in recent_df.columns:
            recent_df["actual_trade_side"] = recent_df["actual_trade_side"].map(_display_trade_direction).replace("", "—")
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
            "Execution Readiness": "Tradeability",
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
            "trade_note": "Trade Note",
            "actual_trade_side": "Trade Direction",
            "actual_entry_price": "Actual Entry",
            "actual_entry_at": "Entry Time",
            "actual_exit_price": "Actual Exit",
            "actual_exit_at": "Exit Time",
            "actual_pnl_pct": "Actual PnL %",
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
