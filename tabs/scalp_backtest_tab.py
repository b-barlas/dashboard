from __future__ import annotations

from collections import Counter
import html
import re
import time

import numpy as np
import pandas as pd
import plotly.graph_objs as go

from core.archive_policy import SCALP_ARCHIVE_WINDOW_ROWS
from core.backtest import (
    build_scalp_outcome_study,
    summarize_scalp_outcome_study,
)
from core.confidence import ai_confidence_bucket, confidence_bucket
from core.decision_version import current_decision_version
from core.symbols import is_stable_base_symbol
from ui.ctx import get_ctx
from ui.primitives import render_badge_row, render_help_details, render_insight_card, render_kpi_grid, render_page_header
from ui.snapshot_cache import live_or_snapshot


def _safe_float(v: object, default: float = float("nan")) -> float:
    try:
        n = float(v)
        return n if np.isfinite(n) else default
    except Exception:
        return default


def _avg_hit_bar(df_events: pd.DataFrame) -> float:
    if df_events is None or df_events.empty:
        return float("nan")
    hit = pd.to_numeric(df_events.get("Hit Bar", pd.Series(dtype=float)), errors="coerce")
    hit = hit.replace([np.inf, -np.inf], np.nan).dropna()
    return float(hit.mean()) if not hit.empty else float("nan")


def _parse_custom_bases(raw: str, limit: int = 10) -> list[str]:
    tokens = re.split(r"[\s,;\n]+", str(raw or "").upper().strip())
    out: list[str] = []
    seen: set[str] = set()
    for tok in tokens:
        t = tok.strip()
        if not t:
            continue
        if "/" in t:
            t = t.split("/", 1)[0].strip()
        for suf in ("-USDT", "_USDT", "USDT", "-USD", "_USD", "USD"):
            if t.endswith(suf) and len(t) > len(suf):
                t = t[: -len(suf)]
                break
        t = re.sub(r"[^A-Z0-9]", "", t)
        if len(t) < 2 or len(t) > 15:
            continue
        if t in seen:
            continue
        seen.add(t)
        out.append(t)
        if len(out) >= int(limit):
            break
    return out


def _filter_scalp_archive_scope(
    df_events: pd.DataFrame,
    *,
    timeframe: str,
    custom_bases: list[str],
    exclude_stables: bool,
) -> pd.DataFrame:
    if df_events is None or df_events.empty:
        return pd.DataFrame()
    d = df_events.copy()
    if "timeframe" in d.columns:
        d["timeframe"] = d["timeframe"].fillna("").astype(str).str.strip().str.lower()
        d = d[d["timeframe"].eq(str(timeframe or "").strip().lower())].copy()
    if "symbol" in d.columns:
        d["symbol"] = d["symbol"].fillna("").astype(str).str.strip().str.upper()
        if custom_bases:
            allowed = {str(value).strip().upper() for value in list(custom_bases or []) if str(value).strip()}
            d = d[d["symbol"].isin(allowed)].copy()
        if exclude_stables:
            d = d[~d["symbol"].map(is_stable_base_symbol)].copy()
    return d.reset_index(drop=True)


def _display_trade_direction(value: object) -> str:
    side = str(value or "").strip().upper()
    if side in {"LONG", "UPSIDE", "BUY"}:
        return "Upside"
    if side in {"SHORT", "DOWNSIDE", "SELL"}:
        return "Downside"
    return ""


def _render_insight_card_grid(st, cards: list[dict[str, str]], *, columns: int = 2) -> None:
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


def _build_scalp_simulation_cohort_summary(
    df_events: pd.DataFrame,
    group_field: str,
    *,
    forward_bars: int,
) -> pd.DataFrame:
    if df_events is None or df_events.empty or group_field not in df_events.columns:
        return pd.DataFrame()
    ret_col = f"Realized Return @+{int(forward_bars)} (%)"
    if ret_col not in df_events.columns:
        ret_col = f"Return @+{int(forward_bars)} (%)"
    d = df_events.copy()
    d[group_field] = d[group_field].fillna("").astype(str).str.strip().replace("", "Unknown")
    outcome = d.get("Outcome", pd.Series(index=d.index, dtype=object)).fillna("").astype(str).str.upper()
    d["_is_tp"] = outcome.eq("TP").astype(int)
    d["_is_sl"] = outcome.isin({"SL", "BOTH"}).astype(int)
    d["_is_timeout"] = outcome.eq("TIMEOUT").astype(int)
    d["_rr"] = pd.to_numeric(d.get("R:R"), errors="coerce")
    d["_ret"] = pd.to_numeric(d.get(ret_col), errors="coerce")
    d["_hit_bar"] = pd.to_numeric(d.get("Hit Bar"), errors="coerce")
    grouped = (
        d.groupby(group_field, dropna=False)
        .agg(
            Signals=(group_field, "count"),
            TPCount=("_is_tp", "sum"),
            SLCount=("_is_sl", "sum"),
            TimeoutCount=("_is_timeout", "sum"),
            AvgRR=("_rr", "mean"),
            AvgOutcomePct=("_ret", "mean"),
            MedianOutcomePct=("_ret", "median"),
            AvgHitBar=("_hit_bar", "mean"),
        )
        .reset_index()
    )
    grouped["TP First %"] = grouped.apply(
        lambda r: (float(r["TPCount"]) / float(r["Signals"]) * 100.0) if float(r["Signals"]) > 0 else 0.0,
        axis=1,
    )
    grouped["SL First %"] = grouped.apply(
        lambda r: (float(r["SLCount"]) / float(r["Signals"]) * 100.0) if float(r["Signals"]) > 0 else 0.0,
        axis=1,
    )
    grouped["Timeout %"] = grouped.apply(
        lambda r: (float(r["TimeoutCount"]) / float(r["Signals"]) * 100.0) if float(r["Signals"]) > 0 else 0.0,
        axis=1,
    )
    grouped = grouped.drop(columns=["TPCount", "SLCount", "TimeoutCount"])
    return grouped.sort_values(
        by=["Signals", "TP First %", "AvgOutcomePct"],
        ascending=[False, False, False],
    ).reset_index(drop=True)


def _build_scalp_study_signal_cards(df_events: pd.DataFrame, *, forward_bars: int) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    works_cards: list[dict[str, str]] = []
    fail_cards: list[dict[str, str]] = []
    setup_summary = _build_scalp_simulation_cohort_summary(df_events, "Setup Confirm", forward_bars=forward_bars)
    direction_summary = _build_scalp_simulation_cohort_summary(df_events, "Direction", forward_bars=forward_bars)
    primary_summary = setup_summary if not setup_summary.empty else direction_summary
    primary_label = "Setup Confirm" if not setup_summary.empty else "Direction"
    if primary_summary.empty:
        return works_cards, fail_cards

    qualified = primary_summary[primary_summary["Signals"] >= 3].copy()
    if qualified.empty:
        return works_cards, fail_cards

    best = qualified.sort_values(
        by=["TP First %", "AvgOutcomePct", "MedianOutcomePct", "Signals"],
        ascending=[False, False, False, False],
    ).iloc[0]
    works_cards.append(
        {
            "title": "What Works",
            "body_html": (
                f"<b>{str(best[primary_label])}</b> is the cleanest simulated scalp cohort in this study "
                f"({float(best['TP First %']):.1f}% TP-first, "
                f"{float(best['AvgOutcomePct']):+.2f}% avg outcome, "
                f"{float(best['AvgHitBar']):.1f} average hit bar across "
                f"{int(best['Signals'])} events)."
            ),
            "tone": "positive" if float(best["AvgOutcomePct"]) >= 0.0 else "neutral",
        }
    )

    worst = qualified.sort_values(
        by=["TP First %", "AvgOutcomePct", "MedianOutcomePct", "Signals"],
        ascending=[True, True, True, False],
    ).iloc[0]
    if str(worst[primary_label]) != str(best[primary_label]) or len(qualified) > 1:
        fail_cards.append(
            {
                "title": "What Needs Care",
                "body_html": (
                    f"<b>{str(worst[primary_label])}</b> is the weakest simulated scalp cohort in this study "
                    f"({float(worst['TP First %']):.1f}% TP-first, "
                    f"{float(worst['AvgOutcomePct']):+.2f}% avg outcome, "
                    f"{float(worst['Timeout %']):.1f}% timeout rate across "
                    f"{int(worst['Signals'])} events)."
                ),
                "tone": (
                    "warning"
                    if float(worst["AvgOutcomePct"]) < 0.0 or float(worst["TP First %"]) < 45.0
                    else "neutral"
                ),
            }
        )

    return works_cards, fail_cards


def _build_live_scalp_kpi_rows(
    *,
    archive_snapshot: dict,
    execution_snapshot: dict,
    positive: str,
    warning: str,
    negative: str,
    muted: str,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    resolved = int(archive_snapshot.get("resolved") or 0)
    taken = int(execution_snapshot.get("taken") or 0)
    actual_closed = int(execution_snapshot.get("actual_closed") or 0)
    follow_value = f"{float(archive_snapshot.get('follow_through_rate') or 0.0):.1f}%" if resolved > 0 else "N/A"
    avg_dir_raw = float(archive_snapshot.get("avg_dir_return") or 0.0)
    avg_dir_value = f"{avg_dir_raw:+.2f}%" if resolved > 0 else "N/A"
    actual_win_rate = (
        f"{float(execution_snapshot.get('actual_win_rate') or 0.0):.1f}%"
        if actual_closed > 0
        else "N/A"
    )
    actual_pnl_raw = float(execution_snapshot.get("avg_actual_pnl") or 0.0)
    actual_pnl_value = f"{actual_pnl_raw:+.2f}%" if actual_closed > 0 else "N/A"
    overview_items = [
        {
            "label": "Follow-Through",
            "value": follow_value,
            "value_color": (positive if resolved > 0 and float(archive_snapshot.get("follow_through_rate") or 0.0) >= 55.0 else (warning if resolved > 0 else muted)),
            "subtext": (
                "Resolved live scalp signals finishing in the intended direction"
                if resolved > 0
                else "No resolved live scalp signals in this view yet"
            ),
        },
        {
            "label": "Avg Dir Return",
            "value": avg_dir_value,
            "value_color": (positive if resolved > 0 and avg_dir_raw >= 0.0 else (negative if resolved > 0 else muted)),
            "subtext": (
                "Average directional move after the scalp archive horizon"
                if resolved > 0
                else "Directional return appears after live scalp signals resolve"
            ),
        },
        {
            "label": "Signals in View",
            "value": int(archive_snapshot.get("total") or 0),
            "subtext": "Latest live scalp archive window",
        },
        {
            "label": "Resolved in View",
            "value": resolved,
            "subtext": f"Open in view: {int(archive_snapshot.get('open') or 0)}",
        },
    ]
    execution_items = [
        {
            "label": "Taken Setups",
            "value": taken,
            "value_color": positive if taken > 0 else muted,
            "subtext": "Tracked scalp signals you marked as Taken in this view",
        },
        {
            "label": "Closed Journaled Trades",
            "value": actual_closed,
            "value_color": positive if actual_closed > 0 else muted,
            "subtext": (
                f"{float(execution_snapshot.get('journal_coverage_pct') or 0.0):.1f}% of taken setups now have an exit"
                if taken > 0
                else "No taken scalp setups journaled yet"
            ),
        },
        {
            "label": "Actual Win Rate",
            "value": actual_win_rate,
            "value_color": positive if actual_closed > 0 and float(execution_snapshot.get("actual_win_rate") or 0.0) >= 50.0 else (warning if actual_closed > 0 else muted),
            "subtext": "Win rate across journaled closed scalp trades",
        },
        {
            "label": "Avg Realized PnL",
            "value": actual_pnl_value,
            "value_color": positive if actual_closed > 0 and actual_pnl_raw >= 0.0 else (negative if actual_closed > 0 else muted),
            "subtext": "Average realized result across journaled closed scalp trades",
        },
    ]
    if taken > 0:
        execution_items[1]["badge_text"] = f"{int(execution_snapshot.get('overlay_marked') or 0)}/{int(execution_snapshot.get('total') or 0)} tagged"
        execution_items[1]["badge_tone"] = "positive" if float(execution_snapshot.get("overlay_coverage_pct") or 0.0) >= 60.0 else "neutral"
        execution_items[1]["badge_color"] = positive if float(execution_snapshot.get("overlay_coverage_pct") or 0.0) >= 60.0 else muted
    return overview_items, execution_items


def _build_historical_scalp_kpi_rows(
    *,
    summary: dict,
    events_df: pd.DataFrame,
    forward_bars: int,
    symbols_with_events: int,
    universe_scanned: int,
    positive: str,
    warning: str,
    negative: str,
    muted: str,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    occurrences = int(summary.get("occurrences") or 0)
    tp_rate = _safe_float(summary.get("tp_rate"), 0.0)
    sl_rate = _safe_float(summary.get("sl_rate"), 0.0)
    timeout_rate = _safe_float(summary.get("timeout_rate"), 0.0)
    avg_outcome = _safe_float(summary.get("avg_outcome"), 0.0)
    median_outcome = _safe_float(summary.get("median_outcome"), 0.0)
    avg_hit = _avg_hit_bar(events_df)
    rr_series = pd.to_numeric(events_df.get("R:R", pd.Series(dtype=float)), errors="coerce")
    avg_rr = float(rr_series.mean()) if not rr_series.dropna().empty else float("nan")
    overview_items = [
        {
            "label": "Events in Study",
            "value": occurrences,
            "subtext": "Scalp-matched historical events in this simulation window",
        },
        {
            "label": "Symbols with Events",
            "value": symbols_with_events,
            "subtext": f"Out of {int(universe_scanned)} scanned symbols",
        },
        {
            "label": "TP First Rate",
            "value": f"{tp_rate:.1f}%",
            "value_color": positive if tp_rate >= 50.0 else warning,
            "subtext": f"Target was hit before stop inside the next {int(forward_bars)} bars",
        },
        {
            "label": f"Avg Outcome @+{int(forward_bars)}",
            "value": f"{avg_outcome:+.2f}%",
            "value_color": positive if avg_outcome >= 0.0 else negative,
            "subtext": "Average realized outcome across scalp events that passed quality checks",
        },
    ]
    diagnostics_items = [
        {
            "label": "Avg Hit Bar",
            "value": (f"+{avg_hit:.1f}" if np.isfinite(avg_hit) else "N/A"),
            "value_color": positive if np.isfinite(avg_hit) and avg_hit <= max(4.0, float(forward_bars) / 2.0) else muted,
            "subtext": "Average bar where TP or SL resolved first",
        },
        {
            "label": "Avg R:R",
            "value": (f"{avg_rr:.2f}" if np.isfinite(avg_rr) else "N/A"),
            "value_color": positive if np.isfinite(avg_rr) and avg_rr >= 1.5 else (warning if np.isfinite(avg_rr) else muted),
            "subtext": "Average planned risk-reward for simulated scalp entries",
        },
        {
            "label": "SL First Rate",
            "value": f"{sl_rate:.1f}%",
            "value_color": negative if sl_rate >= 40.0 else warning,
            "subtext": "Stop was hit before target inside the study window",
        },
        {
            "label": "Timeout Rate",
            "value": f"{timeout_rate:.1f}%",
            "value_color": warning if timeout_rate >= 35.0 else muted,
            "subtext": f"No TP or SL reached inside {int(forward_bars)} bars",
            "badge_text": (f"Median {median_outcome:+.2f}%" if occurrences > 0 else ""),
            "badge_tone": "neutral",
            "badge_color": muted,
        },
    ]
    return overview_items, diagnostics_items


def _build_scalp_study_timing_inputs(
    df_events: pd.DataFrame,
    *,
    forward_bars: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if df_events is None or df_events.empty:
        return pd.DataFrame(), pd.DataFrame()

    events = df_events.copy().reset_index(drop=True)
    events["signal_key"] = [f"study-{idx + 1}" for idx in range(len(events))]
    events["status"] = "RESOLVED"

    window_rows: list[dict[str, object]] = []
    for idx, row in events.iterrows():
        signal_key = str(events.at[idx, "signal_key"])
        running_worst = 0.0
        for step in range(1, int(forward_bars) + 1):
            return_col = f"Directional Return +{step} (%)"
            directional_return_pct = _safe_float(row.get(return_col), float("nan"))
            if not np.isfinite(directional_return_pct):
                continue
            running_worst = min(running_worst, float(directional_return_pct))
            window_rows.append(
                {
                    "signal_key": signal_key,
                    "bars_ahead": int(step),
                    "directional_return_pct": float(directional_return_pct),
                    "adverse_excursion_pct": abs(running_worst) if running_worst < 0.0 else 0.0,
                }
            )

    return events, pd.DataFrame(window_rows)


def _build_historical_timing_cards(
    df_events: pd.DataFrame,
    *,
    timeframe: str,
    custom_bases: list[str],
    forward_bars: int,
    build_hold_window_intelligence,
) -> list[dict[str, str]]:
    events, windows = _build_scalp_study_timing_inputs(df_events, forward_bars=forward_bars)
    if events.empty:
        return []

    scope_label = (
        f"{', '.join(custom_bases[:4])}" + (f" +{len(custom_bases) - 4} more" if len(custom_bases) > 4 else "")
        if custom_bases
        else f"{str(timeframe).upper()} study"
    )
    events["_timing_direction"] = (
        events.get("Direction", pd.Series(index=events.index, dtype=object)).map(_display_trade_direction)
    )

    cards: list[dict[str, str]] = []
    for direction_label in ("Upside", "Downside"):
        direction_events = events[events["_timing_direction"].eq(direction_label)].copy()
        direction_keys = (
            direction_events["signal_key"].fillna("").astype(str).str.strip().tolist()
            if "signal_key" in direction_events.columns
            else []
        )
        direction_windows = (
            windows[windows["signal_key"].fillna("").astype(str).str.strip().isin(direction_keys)].copy()
            if direction_keys and not windows.empty
            else pd.DataFrame()
        )
        snapshot = build_hold_window_intelligence(direction_events, direction_windows)
        body_html, tone = _scalp_timing_note(
            snapshot,
            direction_label=direction_label,
            scope_label=scope_label,
        )
        cards.append(
            {
                "title": f"{direction_label} Timing",
                "body_html": body_html,
                "tone": tone,
            }
        )
    return cards


def _forward_window_progress_snapshot(df_events: pd.DataFrame, df_forward_windows: pd.DataFrame) -> dict[str, float]:
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


def _scalp_timing_note(
    snapshot: dict[str, object],
    *,
    direction_label: str,
    scope_label: str,
) -> tuple[str, str]:
    resolved_signals = int(snapshot.get("resolved_signals") or 0)
    if resolved_signals <= 0:
        return (
            f"No resolved <b>{direction_label.lower()}</b> scalp signals are available yet for <b>{scope_label}</b>.",
            "neutral",
        )
    if not bool(snapshot.get("available")):
        return (
            (
                f"<b>{direction_label}</b> timing is still building for <b>{scope_label}</b>. "
                f"We have <b>{resolved_signals}</b> resolved signals, but the checkpoint depth is not ready yet."
            ),
            "neutral",
        )
    best_bar = int(snapshot.get("best_bar") or 0)
    best_style = str(snapshot.get("best_style") or "Standard Hold").strip()
    fade_after_bar = int(snapshot.get("fade_after_bar") or 0)
    avg_dir_return_pct = float(snapshot.get("avg_dir_return_pct") or 0.0)
    follow_through_pct = float(snapshot.get("follow_through_pct") or 0.0)
    fade_text = (
        f"Edge fades after roughly <b>{fade_after_bar}</b> bars."
        if fade_after_bar > 0
        else "Edge has not clearly faded inside the measured checkpoint ladder yet."
    )
    tone = "positive" if avg_dir_return_pct > 0.0 else "warning"
    return (
        (
            f"<b>{direction_label}: Best at {best_bar} bars</b><br>"
            f"Style: <b>{best_style}</b>.<br>"
            f"Avg directional return: <b>{avg_dir_return_pct:+.2f}%</b>, "
            f"follow-through: <b>{follow_through_pct:.1f}%</b>. "
            f"{fade_text}"
        ),
        tone,
    )


def render(ctx: dict) -> None:
    st = get_ctx(ctx, "st")
    ACCENT = get_ctx(ctx, "ACCENT")
    TEXT_MUTED = get_ctx(ctx, "TEXT_MUTED")
    POSITIVE = get_ctx(ctx, "POSITIVE")
    WARNING = get_ctx(ctx, "WARNING")
    NEGATIVE = "#FF3366"
    fetch_ohlcv = get_ctx(ctx, "fetch_ohlcv")
    get_top_volume_usdt_symbols = get_ctx(ctx, "get_top_volume_usdt_symbols")
    analyse = get_ctx(ctx, "analyse")
    ml_ensemble_predict = get_ctx(ctx, "ml_ensemble_predict")
    signal_plain = get_ctx(ctx, "signal_plain")
    direction_key = get_ctx(ctx, "direction_key")
    _calc_conviction = get_ctx(ctx, "_calc_conviction")
    get_scalping_entry_target = get_ctx(ctx, "get_scalping_entry_target")
    scalp_quality_gate = get_ctx(ctx, "scalp_quality_gate")
    _sr_lookback = get_ctx(ctx, "_sr_lookback")
    get_signal_tracker_db_path = get_ctx(ctx, "get_signal_tracker_db_path")
    init_signal_tracker_db = get_ctx(ctx, "init_signal_tracker_db")
    fetch_signal_events_df = get_ctx(ctx, "fetch_signal_events_df")
    fetch_signal_forward_windows_df = get_ctx(ctx, "fetch_signal_forward_windows_df")
    build_signal_review_snapshot = get_ctx(ctx, "build_signal_review_snapshot")
    build_execution_overlay_snapshot = get_ctx(ctx, "build_execution_overlay_snapshot")
    build_signal_cohort_summary = get_ctx(ctx, "build_signal_cohort_summary")
    build_hold_window_intelligence = get_ctx(ctx, "build_hold_window_intelligence")
    save_signal_trade_overlay = get_ctx(ctx, "save_signal_trade_overlay")
    save_signal_trade_journal = get_ctx(ctx, "save_signal_trade_journal")

    db_path = init_signal_tracker_db(get_signal_tracker_db_path())

    def _chip(text: str, tone: str, extra_class: str = "", title: str | None = None) -> str:
        if not str(text or "").strip():
            return ""
        tone_class = {
            "pos": "sb-pos",
            "neg": "sb-neg",
            "warn": "sb-warn",
            "muted": "sb-muted",
            "info": "sb-info",
        }.get(tone, "sb-muted")
        title_attr = f" title='{html.escape(title)}'" if title else ""
        return (
            f"<span class='sb-chip {tone_class} {extra_class}'{title_attr}>"
            f"{html.escape(str(text))}</span>"
        )

    def _direction_chip(value: str) -> str:
        v = str(value or "").strip().upper()
        if "UPSIDE" in v:
            return _chip("Upside", "pos")
        if "DOWNSIDE" in v:
            return _chip("Downside", "neg")
        return _chip("Neutral", "warn")

    def _parse_support_votes(votes_text: object) -> int:
        m = re.search(r"(\d)\s*/\s*3", str(votes_text or ""))
        return max(0, min(3, int(m.group(1)))) if m else 0

    def _confidence_chip(value: object) -> str:
        try:
            s = float(value)
        except Exception:
            return ""
        label = str(confidence_bucket(s) or "LOW").upper()
        text = f"{s:.0f}% ({label.title()})"
        if label == "HIGH":
            tone = "pos"
        elif label == "MEDIUM":
            tone = "warn"
        else:
            tone = "neg"
        return _chip(text, tone)

    def _ai_confidence_chip(
        value: object,
        *,
        direction: object,
        votes_text: object,
        timeframe_conflict: object = False,
        degraded_data: object = False,
    ) -> str:
        try:
            s = float(value)
        except Exception:
            return ""
        if not np.isfinite(s):
            return ""
        label = ai_confidence_bucket(
            s,
            direction=str(direction or ""),
            support_votes=_parse_support_votes(votes_text),
            timeframe_conflict=bool(timeframe_conflict),
            degraded_data=bool(degraded_data),
        ).title()
        text = f"{s:.0f}% ({label})"
        if label.upper() == "HIGH":
            tone = "pos"
        elif label.upper() == "MEDIUM":
            tone = "warn"
        else:
            tone = "neg"
        return _chip(text, tone)

    def _ai_ensemble_chip(direction: str, votes_text: str) -> str:
        d = str(direction or "").strip()
        tone = "warn"
        if str(direction or "").strip().upper().startswith("UPSIDE"):
            tone = "pos"
        elif str(direction or "").strip().upper().startswith("DOWNSIDE"):
            tone = "neg"
        m = re.search(r"(\d)\s*/\s*3", str(votes_text or ""))
        votes_n = int(m.group(1)) if m else 0
        votes_n = max(0, min(3, votes_n))
        dots = "".join(
            f"<span class='sb-ai-dot{' is-filled' if i < votes_n else ''}'></span>"
            for i in range(3)
        )
        tone_class = {
            "pos": "sb-pos",
            "neg": "sb-neg",
            "warn": "sb-warn",
            "muted": "sb-muted",
            "info": "sb-info",
        }.get(tone, "sb-muted")
        return (
            f"<span class='sb-chip {tone_class} sb-chip-ai'>"
            f"<span class='sb-ai-text'>{html.escape(d or 'Neutral')}</span>"
            f"<span class='sb-ai-dots'>{dots}</span>"
            f"</span>"
        )

    def _outcome_chip(value: str) -> str:
        v = str(value or "").strip().upper()
        if v == "TP":
            return _chip("TP", "pos")
        if v in {"SL", "BOTH"}:
            return _chip(v, "neg")
        if v == "TIMEOUT":
            return _chip("TIMEOUT", "warn")
        return _chip(v or "N/A", "muted")

    def _gate_reason_label(code: str) -> str:
        key = str(code or "").strip().upper()
        labels = {
            "NO_SCALP_DIRECTION": "No scalp direction",
            "SIGNAL_DIRECTION_NEUTRAL": "Signal neutral",
            "DIRECTION_MISMATCH": "Direction mismatch",
            "CONFLICT": "Tech/AI conflict",
            "RR_TOO_LOW": "R:R too low",
            "ADX_TOO_LOW": "ADX too low",
            "CONFIDENCE_TOO_LOW": "Confidence too low",
            "INVALID_LEVELS": "Invalid levels",
            "UNKNOWN_GATE_REJECT": "Unknown reject",
        }
        return labels.get(key, key.title())

    def _price_step_html(price_val: object, event_price: float) -> str:
        try:
            p = float(price_val)
            e = float(event_price)
            if not np.isfinite(p) or not np.isfinite(e) or e <= 0:
                return ""
            pct = ((p / e) - 1.0) * 100.0
            if pct > 0:
                pct_html = f"<span class='sb-step-pos'>(+{pct:.2f}%)</span>"
            elif pct < 0:
                pct_html = f"<span class='sb-step-neg'>({pct:.2f}%)</span>"
            else:
                pct_html = "<span class='sb-step-neu'>(0.00%)</span>"
            return f"<span class='sb-step-price'>${p:,.6f}</span> {pct_html}"
        except Exception:
            return ""

    def _render_hover_table(
        df_table: pd.DataFrame,
        header_tips: dict[str, str],
        table_id: str,
    ) -> None:
        if df_table.empty:
            return
        cols = list(df_table.columns)
        header_html = "".join(
            "<th>"
            f"<span class='sb-hdr' title='{html.escape(header_tips.get(c, ''), quote=True)}'>"
            f"{html.escape(str(c))} <span class='sb-help'>?</span>"
            "</span>"
            "</th>"
            for c in cols
        )
        row_html: list[str] = []
        for _, row in df_table.iterrows():
            cells = []
            for c in cols:
                v = row.get(c, "")
                txt = "" if pd.isna(v) else str(v)
                cells.append(f"<td>{html.escape(txt)}</td>")
            row_html.append("<tr>" + "".join(cells) + "</tr>")
        st.markdown(
            f"""
            <style>
              .sb-wrap-{table_id} {{
                width:100%;
                overflow-x:auto;
                border:1px solid rgba(0,212,255,0.20);
                border-radius:12px;
                background:linear-gradient(180deg, rgba(10,16,26,0.94), rgba(6,11,20,0.94));
              }}
              .sb-table-{table_id} {{
                width:max-content;
                min-width:100%;
                border-collapse:separate;
                border-spacing:0;
                font-size:0.83rem;
                font-family:'Manrope','Segoe UI',sans-serif;
              }}
              .sb-table-{table_id} th {{
                text-align:left;
                padding:10px 10px;
                color:{TEXT_MUTED};
                font-weight:700;
                border-bottom:1px solid rgba(148,163,184,0.22);
                border-right:1px solid rgba(148,163,184,0.08);
                white-space:nowrap;
                background:linear-gradient(180deg, rgba(19,26,38,0.98), rgba(12,19,31,0.98));
              }}
              .sb-table-{table_id} td {{
                padding:9px 10px;
                color:#E5E7EB;
                border-bottom:1px solid rgba(148,163,184,0.12);
                border-right:1px solid rgba(148,163,184,0.07);
                white-space:nowrap;
              }}
              .sb-table-{table_id} tr:hover td {{
                background-color:rgba(0,212,255,0.05);
              }}
              .sb-hdr {{
                display:inline-flex;
                align-items:center;
                gap:6px;
                cursor:help;
              }}
            </style>
            <div class='sb-wrap-{table_id}'>
              <table class='sb-table-{table_id}'>
                <thead><tr>{header_html}</tr></thead>
                <tbody>{''.join(row_html)}</tbody>
              </table>
            </div>
            """,
            unsafe_allow_html=True,
        )

    def _render_event_table_html(df_events: pd.DataFrame, n_forward: int) -> None:
        cols = [
            "Event Time",
            "Coin",
            "Pair",
            "Setup Confirm",
            "Direction",
            "Confidence",
            "AI Ensemble",
            "AI Confidence",
            "Event Price",
            "Target",
            "Stop",
            "R:R",
            "Outcome",
            "Hit Bar",
        ] + [f"Price +{i}" for i in range(1, n_forward + 1)] + [
            f"End Price (+{n_forward})",
            f"Realized Return @+{n_forward} (%)",
            f"Close Directional Return @+{n_forward} (%)",
        ]

        header_html = "".join(f"<th>{html.escape(c)}</th>" for c in cols)
        row_html = []
        for _, r in df_events.iterrows():
            event_price = pd.to_numeric(r.get("Event Price"), errors="coerce")
            cells = []
            for c in cols:
                if c == "Event Time":
                    ts = pd.to_datetime(r.get(c), errors="coerce")
                    txt = ts.strftime("%Y-%m-%d %H:%M") if pd.notna(ts) else ""
                    cells.append(f"<td><span class='sb-plain'>{html.escape(txt)}</span></td>")
                elif c in {"Coin", "Pair", "Setup Confirm"}:
                    cells.append(f"<td><span class='sb-plain'>{html.escape(str(r.get(c, '')))}</span></td>")
                elif c == "Direction":
                    cells.append(f"<td>{_direction_chip(str(r.get(c, '')))}</td>")
                elif c == "Confidence":
                    cells.append(f"<td>{_confidence_chip(r.get(c, r.get('Strength')))}</td>")
                elif c == "AI Ensemble":
                    cells.append(
                        f"<td>{_ai_ensemble_chip(str(r.get('AI Direction', 'Neutral')), str(r.get('AI Votes', '0/3')))}</td>"
                    )
                elif c == "AI Confidence":
                    ai_data_partial = r.get("AI Data Partial", r.get("AI Degraded", False))
                    cells.append(
                        f"<td>{_ai_confidence_chip(r.get(c, r.get('AI Confidence')), direction=r.get('AI Direction'), votes_text=r.get('AI Votes'), timeframe_conflict=r.get('AI Timeframe Conflict', False), degraded_data=ai_data_partial)}</td>"
                    )
                elif c in {"Event Price", "Target", "Stop"}:
                    v = pd.to_numeric(r.get(c), errors="coerce")
                    txt = f"${float(v):,.6f}" if pd.notna(v) else ""
                    cells.append(f"<td><span class='sb-plain'>{html.escape(txt)}</span></td>")
                elif c == "R:R":
                    v = pd.to_numeric(r.get(c), errors="coerce")
                    if pd.notna(v):
                        tone = "pos" if float(v) >= 1.8 else ("warn" if float(v) >= 1.4 else "neg")
                        cells.append(f"<td>{_chip(f'{float(v):.2f}', tone)}</td>")
                    else:
                        cells.append("<td></td>")
                elif c == "Outcome":
                    cells.append(f"<td>{_outcome_chip(str(r.get(c, '')))}</td>")
                elif c == "Hit Bar":
                    v = pd.to_numeric(r.get(c), errors="coerce")
                    txt = f"+{int(v)}" if pd.notna(v) else ""
                    cells.append(f"<td><span class='sb-plain'>{html.escape(txt)}</span></td>")
                elif c.startswith("Price +"):
                    cells.append(f"<td>{_price_step_html(r.get(c), float(event_price))}</td>")
                elif c.startswith("End Price (+"):
                    v = pd.to_numeric(r.get(c), errors="coerce")
                    txt = f"${float(v):,.6f}" if pd.notna(v) else ""
                    cells.append(f"<td><span class='sb-plain'>{html.escape(txt)}</span></td>")
                elif c.startswith("Realized Return @+") or c.startswith("Return @+"):
                    v = pd.to_numeric(r.get(c), errors="coerce")
                    if pd.isna(v) and c.startswith("Realized Return @+"):
                        legacy_col = c.replace("Realized Return @+", "Return @+")
                        v = pd.to_numeric(r.get(legacy_col), errors="coerce")
                    if pd.notna(v):
                        cells.append(
                            f"<td>{_chip(f'{float(v):+.2f}%', 'pos' if float(v) > 0 else ('neg' if float(v) < 0 else 'warn'))}</td>"
                        )
                    else:
                        cells.append("<td></td>")
                elif c.startswith("Close Directional Return @+"):
                    v = pd.to_numeric(r.get(c), errors="coerce")
                    if pd.notna(v):
                        cells.append(
                            f"<td>{_chip(f'{float(v):+.2f}%', 'pos' if float(v) > 0 else ('neg' if float(v) < 0 else 'warn'))}</td>"
                        )
                    else:
                        cells.append("<td></td>")
                else:
                    cells.append(f"<td><span class='sb-plain'>{html.escape(str(r.get(c, '')))}</span></td>")
            row_html.append("<tr>" + "".join(cells) + "</tr>")

        st.markdown(
            f"""
            <style>
              .sb-wrap {{
                width:100%;
                overflow-x:auto;
                border:1px solid rgba(0,212,255,0.20);
                border-radius:12px;
                background:linear-gradient(180deg, rgba(6,10,18,0.96), rgba(4,8,14,0.96));
              }}
              .sb-table {{
                width:max-content;
                min-width:100%;
                border-collapse:separate;
                border-spacing:0;
                font-size:0.81rem;
                font-family:'Manrope','Segoe UI',sans-serif;
              }}
              .sb-table th {{
                text-align:left;
                padding:10px 10px;
                color:{TEXT_MUTED};
                font-weight:700;
                border-bottom:1px solid rgba(148,163,184,0.22);
                border-right:1px solid rgba(148,163,184,0.08);
                white-space:nowrap;
                position:sticky;
                top:0;
                z-index:2;
                background:linear-gradient(180deg, rgba(18,24,36,0.98), rgba(12,18,30,0.98));
              }}
              .sb-table td {{
                padding:8px 10px;
                color:#E5E7EB;
                border-bottom:1px solid rgba(148,163,184,0.12);
                border-right:1px solid rgba(148,163,184,0.07);
                white-space:nowrap;
              }}
              .sb-table tr:hover td {{ background-color:rgba(0,212,255,0.06); }}
              .sb-chip {{
                display:inline-flex;
                align-items:center;
                gap:6px;
                padding:2px 8px;
                border-radius:999px;
                border:1px solid rgba(148,163,184,0.34);
                background:rgba(148,163,184,0.10);
                font-size:0.74rem;
                font-weight:700;
                white-space:nowrap;
              }}
              .sb-chip-ai {{ gap:8px; min-height:26px; padding:2px 8px; }}
              .sb-ai-dots {{ display:inline-flex; align-items:center; gap:3px; }}
              .sb-ai-dot {{
                width:8px; height:8px; border-radius:999px; border:1px solid currentColor;
                background:transparent; opacity:0.55; flex:0 0 8px;
              }}
              .sb-ai-dot.is-filled {{ background:currentColor; opacity:1; }}
              .sb-pos {{ color:{POSITIVE}; border-color:rgba(0,255,136,0.42); background:rgba(0,255,136,0.10); }}
              .sb-neg {{ color:#FF3366; border-color:rgba(255,51,102,0.44); background:rgba(255,51,102,0.10); }}
              .sb-warn {{ color:{WARNING}; border-color:rgba(255,209,102,0.46); background:rgba(255,209,102,0.10); }}
              .sb-info {{ color:{ACCENT}; border-color:rgba(0,212,255,0.46); background:rgba(0,212,255,0.10); }}
              .sb-muted {{ color:{TEXT_MUTED}; border-color:rgba(140,161,182,0.35); background:rgba(140,161,182,0.08); }}
              .sb-step-price {{ color:#E5E7EB; }}
              .sb-step-pos {{ color:{POSITIVE}; font-weight:700; }}
              .sb-step-neg {{ color:#FF3366; font-weight:700; }}
              .sb-step-neu {{ color:{TEXT_MUTED}; font-weight:700; }}
              .sb-plain {{ color:#E5E7EB; }}
            </style>
            <div class="sb-wrap">
              <table class="sb-table">
                <thead><tr>{header_html}</tr></thead>
                <tbody>{''.join(row_html)}</tbody>
              </table>
            </div>
            """,
            unsafe_allow_html=True,
        )

    render_page_header(
        st,
        title="Scalp Lab",
        intro_html=(
            "Live scalp archive on top, historical study below. "
            "Use this page to compare real tracked scalp behavior with closed-candle planner study before changing policy."
        ),
    )
    intro_cols = st.columns(2)
    with intro_cols[0]:
        render_insight_card(
            st,
            title="How To Use It",
            body_html=(
                "Start with <b>Live Scalp Archive</b> for current truth, then use <b>Historical Study</b> to stress-test the scalp planner."
            ),
            tone="neutral",
        )
    with intro_cols[1]:
        render_insight_card(
            st,
            title="Shared Engine",
            body_html=(
                "Both sections use the same <b>entry / stop / target planner</b> and <b>scalp quality checks</b> as the Market tab."
            ),
            tone="positive",
        )
    render_help_details(
        st,
        summary="Quick read",
        body_html=(
            "1. The top block is <b>live archive truth</b>; the lower block is <b>historical study</b>.<br>"
            "2. If <b>Custom Coins</b> is empty, study runs on the top-volume universe.<br>"
            "3. If <b>Custom Coins</b> has symbols, study runs only on those symbols.<br>"
            "4. Each historical event is labeled as <b>TP-first</b>, <b>SL-first</b>, or <b>TIMEOUT</b> inside the chosen forward window."
        ),
    )

    c1, c2 = st.columns(2)
    with c1:
        timeframe = st.selectbox(
            "Timeframe",
            ["5m", "15m", "1h", "4h", "1d"],
            index=1,
            key="scalp_bt_timeframe",
        )
        lookback_candles = st.slider("Lookback Candles", 200, 3000, step=100, value=1000, key="scalp_bt_lookback")
    with c2:
        universe_size = st.slider(
            "Universe Size (Top Volume Pairs)",
            10,
            120,
            step=5,
            value=40,
            key="scalp_bt_universe",
        )
        forward_bars = st.slider(
            "Tracking Window (bars after event)",
            3,
            30,
            step=1,
            value=10,
            key="scalp_bt_forward",
        )
        st.caption("Example: 10 means each scalp event is tracked for 10 bars after trigger.")

    cc1, cc2 = st.columns([0.86, 0.14])
    with cc1:
        custom_coin_input = st.text_input(
            "Custom Coins (max 10)",
            value=st.session_state.get("scalp_bt_custom_coin_input", ""),
            key="scalp_bt_custom_coin_input",
            placeholder="BTC, ETH, SOL",
            help="Optional. If filled, study runs only on these symbols.",
        )
    with cc2:
        # Align clear button with custom input field row.
        st.markdown("<div style='height:1.9rem;'></div>", unsafe_allow_html=True)
        clear_custom = st.button("Clear", width="stretch", key="scalp_bt_clear_custom")
    if clear_custom:
        st.session_state["scalp_bt_custom_coin_input"] = ""
        st.rerun()

    custom_bases = _parse_custom_bases(custom_coin_input, limit=10)
    custom_mode_active = bool(custom_bases)
    if custom_mode_active:
        preview = ", ".join(custom_bases[:6])
        more = "" if len(custom_bases) <= 6 else f" +{len(custom_bases) - 6}"
        st.caption(
            f"Custom mode active: scanning {len(custom_bases)} symbol(s): {preview}{more}. "
            "Universe Size is ignored while custom mode is active."
        )

    if hasattr(st, "checkbox"):
        exclude_stables = st.checkbox(
            "Exclude stablecoins",
            value=True,
            key="scalp_bt_exclude_stables",
            help="Enabled by default to keep scalp study focused on directional assets.",
        )
    else:
        # Contract-test dummy Streamlit object fallback.
        exclude_stables = True
    est_symbols = len(custom_bases) if custom_mode_active else int(universe_size)
    est_workload = int(lookback_candles) * int(est_symbols)
    st.caption(
        f"Estimated workload: {int(est_symbols)} symbols x {int(lookback_candles)} candles = {est_workload:,} rows to scan."
    )

    archive_history_df = fetch_signal_events_df(
        limit=SCALP_ARCHIVE_WINDOW_ROWS,
        source="Scalp",
        timeframe=timeframe,
        decision_version=current_decision_version("Market"),
        db_path=db_path,
    )
    archive_view_df = _filter_scalp_archive_scope(
        archive_history_df,
        timeframe=timeframe,
        custom_bases=custom_bases,
        exclude_stables=bool(exclude_stables),
    )

    st.markdown("### Live Scalp Archive")
    if archive_view_df.empty:
        scope_text = (
            f" for {', '.join(custom_bases[:6])}" + (f" +{len(custom_bases) - 6} more" if len(custom_bases) > 6 else "")
            if custom_mode_active
            else ""
        )
        st.caption(
            f"No learned scalp archive rows yet on {timeframe.upper()}{scope_text}. "
            "The tracker will start filling this view from live Market scans."
        )
    else:
        archive_snapshot = build_signal_review_snapshot(archive_view_df)
        execution_snapshot = build_execution_overlay_snapshot(archive_view_df)
        archive_signal_keys = (
            archive_view_df["signal_key"].fillna("").astype(str).str.strip().tolist()
            if "signal_key" in archive_view_df.columns
            else []
        )
        archive_forward_windows_df = (
            fetch_signal_forward_windows_df(signal_keys=archive_signal_keys, db_path=db_path)
            if archive_signal_keys
            else pd.DataFrame()
        )
        timing_progress = _forward_window_progress_snapshot(archive_view_df, archive_forward_windows_df)
        live_overview_items, live_execution_items = _build_live_scalp_kpi_rows(
            archive_snapshot=archive_snapshot,
            execution_snapshot=execution_snapshot,
            positive=POSITIVE,
            warning=WARNING,
            negative=NEGATIVE,
            muted=TEXT_MUTED,
        )
        st.markdown("#### Overview")
        st.caption("Quick live archive + execution read.")
        render_kpi_grid(
            st,
            columns=4,
            items=live_overview_items,
        )
        render_kpi_grid(st, columns=4, items=live_execution_items)
        scope_text = (
            f"{', '.join(custom_bases[:6])}" + (f" +{len(custom_bases) - 6} more" if len(custom_bases) > 6 else "")
            if custom_mode_active
            else "current Market universe"
        )
        st.caption(
            f"Live tracker truth for {timeframe.upper()} scalp signals. Scope: {scope_text}. "
            f"Open {int(archive_snapshot['open'])} • "
            f"Avg MAE {float(archive_snapshot['avg_adverse_excursion']):.2f}% • "
            f"Avg MFE {float(archive_snapshot['avg_favorable_excursion']):.2f}% • "
            f"Planned TP {float(archive_snapshot['planned_tp_rate']):.1f}%."
        )
        direction_archive = (
            archive_view_df.assign(
                Direction=archive_view_df.get("direction", pd.Series(dtype=object)).fillna("").astype(str).str.title()
            )
            .groupby("Direction", dropna=False)
            .agg(
                Signals=("signal_key", "count"),
                Resolved=("status", lambda s: int(s.fillna("").astype(str).str.upper().eq("RESOLVED").sum())),
                FollowThroughPct=(
                    "directional_return_pct",
                    lambda s: float((pd.to_numeric(s, errors="coerce") > 0).mean() * 100.0),
                ),
                AvgDirReturnPct=("directional_return_pct", lambda s: float(pd.to_numeric(s, errors="coerce").mean())),
            )
            .reset_index()
        )
        if not direction_archive.empty:
            direction_archive = direction_archive.rename(
                columns={
                    "FollowThroughPct": "Follow-Through %",
                    "AvgDirReturnPct": "Avg Dir Return %",
                }
            )
        coin_archive = (
            archive_view_df.assign(Coin=archive_view_df.get("symbol", pd.Series(dtype=object)).fillna("").astype(str))
            .groupby("Coin", dropna=False)
            .agg(
                Signals=("signal_key", "count"),
                Resolved=("status", lambda s: int(s.fillna("").astype(str).str.upper().eq("RESOLVED").sum())),
                FollowThroughPct=(
                    "directional_return_pct",
                    lambda s: float((pd.to_numeric(s, errors="coerce") > 0).mean() * 100.0),
                ),
                AvgDirReturnPct=("directional_return_pct", lambda s: float(pd.to_numeric(s, errors="coerce").mean())),
            )
            .reset_index()
            .sort_values(["Signals", "FollowThroughPct"], ascending=[False, False])
            .head(12)
        )
        if not coin_archive.empty:
            coin_archive = coin_archive.rename(
                columns={
                    "FollowThroughPct": "Follow-Through %",
                    "AvgDirReturnPct": "Avg Dir Return %",
                }
            )
        archive_analysis_df = archive_view_df.copy()
        archive_analysis_df["Setup Confirm"] = (
            archive_analysis_df.get("setup_confirm", pd.Series(dtype=object))
            .fillna("")
            .astype(str)
            .str.strip()
            .replace("", "Unknown")
        )
        archive_analysis_df["Scalp State"] = (
            archive_analysis_df.get("action_reason", pd.Series(dtype=object))
            .fillna("")
            .astype(str)
            .str.strip()
            .replace("", "Unknown")
        )
        setup_summary_df = build_signal_cohort_summary(archive_analysis_df, "Setup Confirm")
        state_summary_df = build_signal_cohort_summary(archive_analysis_df, "Scalp State")

        works_cards: list[dict[str, str]] = []
        fail_cards: list[dict[str, str]] = []
        qualified_setup_summary = setup_summary_df[setup_summary_df["Resolved"] >= 3].copy() if not setup_summary_df.empty else pd.DataFrame()
        if not qualified_setup_summary.empty:
            best_setup = qualified_setup_summary.sort_values(
                by=["FollowThroughPct", "TpRatePct", "AvgDirReturnPct", "Signals"],
                ascending=[False, False, False, False],
            ).iloc[0]
            worst_setup = qualified_setup_summary.sort_values(
                by=["FollowThroughPct", "TpRatePct", "AvgDirReturnPct", "Signals"],
                ascending=[True, True, True, False],
            ).iloc[0]
            works_cards.append(
                {
                    "title": "Setup Edge",
                    "body_html": (
                        f"<b>{str(best_setup['Setup Confirm'])}</b> is the cleanest live scalp cohort so far "
                        f"({float(best_setup['FollowThroughPct']):.1f}% follow-through, "
                        f"{float(best_setup['TpRatePct']):.1f}% planned TP rate, "
                        f"{float(best_setup['AvgDirReturnPct']):+.2f}% avg directional return across "
                        f"{int(best_setup['Resolved'])} resolved signals)."
                    ),
                    "tone": "positive" if float(best_setup["AvgDirReturnPct"]) >= 0.0 else "neutral",
                }
            )
            if str(worst_setup["Setup Confirm"]) != str(best_setup["Setup Confirm"]) or len(qualified_setup_summary) > 1:
                fail_cards.append(
                    {
                        "title": "Weak Setup Edge",
                        "body_html": (
                            f"<b>{str(worst_setup['Setup Confirm'])}</b> is the weakest live scalp cohort in this scope "
                            f"({float(worst_setup['FollowThroughPct']):.1f}% follow-through, "
                            f"{float(worst_setup['TpRatePct']):.1f}% planned TP rate, "
                            f"{float(worst_setup['AvgDirReturnPct']):+.2f}% avg directional return across "
                            f"{int(worst_setup['Resolved'])} resolved signals)."
                        ),
                        "tone": (
                            "warning"
                            if float(worst_setup["AvgDirReturnPct"]) < 0.0 or float(worst_setup["FollowThroughPct"]) < 50.0
                            else "neutral"
                        ),
                    }
                )
        else:
            works_cards.append(
                {
                    "title": "Setup Edge",
                    "body_html": (
                        "Live scalp cohorts are still thin. We need at least <b>3 resolved signals</b> in a setup family "
                        "before calling out what works."
                    ),
                    "tone": "neutral",
                }
            )

        st.markdown("#### What Works")
        st.caption("Current trust list.")
        _render_insight_card_grid(st, works_cards, columns=2)

        st.markdown("#### What Needs Care")
        st.caption("Current caution list.")
        if not fail_cards:
            fail_cards.append(
                {
                    "title": "Weak Setup Edge",
                    "body_html": "No clear weak live scalp cohort stands out yet in this view. Keep building resolved history.",
                    "tone": "neutral",
                }
            )
        _render_insight_card_grid(st, fail_cards, columns=2)

        st.markdown("#### Execution Review")
        st.caption(
            "Tag whether you actually took a tracked scalp setup. If you took it, add the real entry and exit so we can separate planner quality from your own execution."
        )

        overlay_breakdown = (
            archive_view_df.assign(
                **{
                    "Action Taken": archive_view_df.get("trade_decision", pd.Series(dtype=object))
                    .fillna("")
                    .astype(str)
                    .str.strip()
                    .replace("", "Unmarked")
                }
            )
            .groupby("Action Taken", dropna=False)
            .agg(Signals=("signal_key", "count"))
            .reset_index()
            .sort_values(["Signals", "Action Taken"], ascending=[False, True])
        )
        journal_breakdown = (
            archive_view_df.assign(
                **{
                    "Journal Status": archive_view_df.get("actual_trade_status", pd.Series(dtype=object))
                    .fillna("")
                    .astype(str)
                    .str.strip()
                    .replace("", "Unjournaled")
                }
            )
            .groupby("Journal Status", dropna=False)
            .agg(Signals=("signal_key", "count"))
            .reset_index()
            .sort_values(["Signals", "Journal Status"], ascending=[False, True])
        )
        with st.expander("Action Taken & Journal", expanded=False):
            overlay_cols = st.columns(2, gap="medium")
            with overlay_cols[0]:
                st.markdown("##### By Action Taken")
                st.dataframe(overlay_breakdown, hide_index=True, width="stretch")
            with overlay_cols[1]:
                st.markdown("##### By Journal Status")
                st.dataframe(journal_breakdown, hide_index=True, width="stretch")

        signal_options: dict[str, str] = {}
        for _, row in archive_view_df.iterrows():
            signal_key = str(row.get("signal_key") or "").strip()
            if not signal_key:
                continue
            event_time = pd.to_datetime(row.get("event_time"), errors="coerce")
            ts_label = event_time.strftime("%Y-%m-%d %H:%M") if pd.notna(event_time) else "Unknown time"
            scalp_reason = str(row.get("lead_label") or "").strip()
            setup_label = str(row.get("setup_confirm") or "").strip() or "Unknown setup"
            label = f"{ts_label} • {str(row.get('symbol') or '')} • {setup_label}"
            if scalp_reason:
                label = f"{label} • {scalp_reason}"
            signal_options[label] = signal_key

        with st.expander("Mark or Journal a Live Scalp Signal", expanded=False):
            if not signal_options:
                st.caption("No tracked scalp signals are available in this view yet.")
            else:
                selected_label = st.selectbox(
                    "Tracked scalp setup",
                    list(signal_options.keys()),
                    index=0,
                    key="scalp_lab_trade_overlay_pick",
                    help="Tag whether you took it, skipped it, or only observed it. If you took it, add the real trade journal below.",
                )
                selected_key = signal_options[selected_label]
                selected_row = archive_view_df[archive_view_df["signal_key"].astype(str) == selected_key].iloc[0]
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
                with st.form("scalp_lab_trade_overlay_form", clear_on_submit=False):
                    chosen_decision = st.selectbox(
                        "Your action",
                        decision_options,
                        index=default_idx,
                        key="scalp_lab_trade_overlay_decision",
                    )
                    trade_note = st.text_input(
                        "Execution note",
                        value=current_note,
                        key="scalp_lab_trade_overlay_note",
                        placeholder="Optional note on why you took, skipped, or observed it",
                    )
                    overlay_submitted = st.form_submit_button("Save tag", use_container_width=False)
                if overlay_submitted:
                    decision_value = "" if chosen_decision == "Clear" else chosen_decision
                    saved = save_signal_trade_overlay(
                        selected_key,
                        trade_decision=decision_value,
                        trade_note=trade_note,
                        db_path=db_path,
                    )
                    if saved:
                        st.success("Scalp setup tag saved.")
                        st.rerun()
                    else:
                        st.error("Scalp setup tag could not be saved for that signal.")

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
                    mapped_exit_reason = (
                        current_exit_reason
                        if current_exit_reason in exit_reason_options
                        else ("Open" if current_trade_status != "CLOSED" else "Manual Exit")
                    )
                    with st.form("scalp_lab_trade_journal_form", clear_on_submit=False):
                        st.markdown("##### Real Trade Journal")
                        st.caption("Use this only for trades you really took. It keeps scalp planner quality separate from your actual execution.")
                        chosen_side = st.selectbox(
                            "Trade direction",
                            side_options,
                            index=default_side_idx,
                            key="scalp_lab_trade_journal_side",
                        )
                        entry_price_text = st.text_input(
                            "Entry price",
                            value="" if pd.isna(current_entry_price) else f"{float(current_entry_price):.8f}".rstrip("0").rstrip("."),
                            key="scalp_lab_trade_journal_entry_price",
                            placeholder="Example: 0.12345",
                        )
                        entry_time_text = st.text_input(
                            "Entry time (UTC)",
                            value=current_entry_at,
                            key="scalp_lab_trade_journal_entry_at",
                            placeholder="2026-04-20T12:00:00Z",
                        )
                        exit_price_text = st.text_input(
                            "Exit price",
                            value="" if pd.isna(current_exit_price) else f"{float(current_exit_price):.8f}".rstrip("0").rstrip("."),
                            key="scalp_lab_trade_journal_exit_price",
                            placeholder="Leave blank if still open",
                        )
                        exit_time_text = st.text_input(
                            "Exit time (UTC)",
                            value=current_exit_at,
                            key="scalp_lab_trade_journal_exit_at",
                            placeholder="Leave blank if still open",
                        )
                        chosen_exit_reason = st.selectbox(
                            "Exit reason",
                            exit_reason_options,
                            index=exit_reason_options.index(mapped_exit_reason),
                            key="scalp_lab_trade_journal_exit_reason",
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
                            st.success("Scalp trade journal saved.")
                            st.rerun()
                        else:
                            st.error("Scalp trade journal could not be saved. Entry price and trade direction are required.")

        st.markdown("#### Timing Intelligence")
        render_kpi_grid(
            st,
            columns=4,
            items=[
                {
                    "label": "Resolved Timing Sample",
                    "value": int(timing_progress["resolved"]),
                    "subtext": "Resolved live scalp signals in this current view",
                },
                {
                    "label": "Timing Ready",
                    "value": int(timing_progress["ready"]),
                    "value_color": POSITIVE if int(timing_progress["ready"]) > 0 else TEXT_MUTED,
                    "subtext": "Resolved signals already carrying checkpoint history",
                },
                {
                    "label": "Timing Missing",
                    "value": int(timing_progress["missing"]),
                    "value_color": WARNING if int(timing_progress["missing"]) > 0 else TEXT_MUTED,
                    "subtext": "Resolved signals still waiting for checkpoint depth",
                },
                {
                    "label": "Timing Coverage",
                    "value": f"{float(timing_progress['coverage_pct']):.1f}%",
                    "value_color": POSITIVE if float(timing_progress["coverage_pct"]) >= 50.0 else (WARNING if float(timing_progress["coverage_pct"]) > 0 else TEXT_MUTED),
                    "subtext": "Checkpoint coverage inside the current live scalp view",
                },
            ],
        )
        scope_label = (
            f"{', '.join(custom_bases[:4])}" + (f" +{len(custom_bases) - 4} more" if len(custom_bases) > 4 else "")
            if custom_mode_active
            else f"{timeframe.upper()} live scalp view"
        )
        archive_view_timing_df = archive_view_df.copy()
        archive_view_timing_df["_timing_direction"] = (
            archive_view_timing_df.get("direction", pd.Series(index=archive_view_timing_df.index, dtype=object)).map(_display_trade_direction)
        )
        timing_cards = []
        for direction_label in ("Upside", "Downside"):
            direction_df = archive_view_timing_df[archive_view_timing_df["_timing_direction"].eq(direction_label)].copy()
            direction_keys = (
                direction_df["signal_key"].fillna("").astype(str).str.strip().tolist()
                if "signal_key" in direction_df.columns
                else []
            )
            direction_windows_df = (
                archive_forward_windows_df[
                    archive_forward_windows_df["signal_key"].fillna("").astype(str).str.strip().isin(direction_keys)
                ].copy()
                if direction_keys and not archive_forward_windows_df.empty
                else pd.DataFrame()
            )
            snapshot = build_hold_window_intelligence(direction_df, direction_windows_df)
            body_html, tone = _scalp_timing_note(
                snapshot,
                direction_label=direction_label,
                scope_label=scope_label,
            )
            timing_cards.append(
                {
                    "title": f"{direction_label} Timing",
                    "body_html": body_html,
                    "tone": tone,
                }
            )
        _render_insight_card_grid(st, timing_cards, columns=2)

        st.markdown("#### Deep Dives")
        st.caption("Optional detail.")
        with st.expander("Setup & State", expanded=False):
            deep_dive_cols = st.columns(2, gap="medium")
            with deep_dive_cols[0]:
                st.markdown("##### By Setup Confirm")
                if setup_summary_df.empty:
                    st.caption("No setup-level live archive breakdown yet.")
                else:
                    st.dataframe(
                        setup_summary_df[
                            [
                                "Setup Confirm",
                                "Signals",
                                "Resolved",
                                "FollowThroughPct",
                                "TpRatePct",
                                "AvgDirReturnPct",
                            ]
                        ].rename(
                            columns={
                                "FollowThroughPct": "Follow-Through %",
                                "TpRatePct": "Planned TP %",
                                "AvgDirReturnPct": "Avg Dir Return %",
                            }
                        ).round(2),
                        hide_index=True,
                        width="stretch",
                    )
            with deep_dive_cols[1]:
                st.markdown("##### By Scalp State")
                if state_summary_df.empty:
                    st.caption("No scalp-state live archive breakdown yet.")
                else:
                    st.dataframe(
                        state_summary_df[
                            [
                                "Scalp State",
                                "Signals",
                                "Resolved",
                                "FollowThroughPct",
                                "TpRatePct",
                                "AvgDirReturnPct",
                            ]
                        ].rename(
                            columns={
                                "FollowThroughPct": "Follow-Through %",
                                "TpRatePct": "Planned TP %",
                                "AvgDirReturnPct": "Avg Dir Return %",
                            }
                        ).round(2),
                        hide_index=True,
                        width="stretch",
                    )
        with st.expander("Direction & Coin", expanded=False):
            deep_dive_cols = st.columns(2, gap="medium")
            with deep_dive_cols[0]:
                st.markdown("##### By Direction")
                if direction_archive.empty:
                    st.caption("No direction-level live archive breakdown yet.")
                else:
                    st.dataframe(direction_archive.round(2), hide_index=True, width="stretch")
            with deep_dive_cols[1]:
                st.markdown("##### By Coin")
                if coin_archive.empty:
                    st.caption("No coin-level live archive breakdown yet.")
                else:
                    st.dataframe(coin_archive.round(2), hide_index=True, width="stretch")

        with st.expander("Data Table", expanded=False):
            archive_recent = archive_view_df.copy()
            if "event_time" in archive_recent.columns:
                archive_recent["event_time"] = (
                    pd.to_datetime(archive_recent["event_time"], errors="coerce")
                    .dt.strftime("%Y-%m-%d %H:%M")
                )
            if "direction" in archive_recent.columns:
                archive_recent["direction"] = archive_recent["direction"].fillna("").astype(str).str.title()
            archive_recent = archive_recent.rename(
                columns={
                    "event_time": "Signal Time",
                    "symbol": "Coin",
                    "setup_confirm": "Setup Confirm",
                    "action_reason": "Scalp State",
                    "lead_label": "Scalp Reason",
                    "direction": "Direction",
                    "confidence": "Confidence",
                    "ai_confidence": "AI Confidence",
                    "entry_price": "Entry",
                    "stop_loss": "Stop",
                    "target_price": "Target",
                    "rr_ratio": "R:R",
                    "status": "Status",
                    "plan_outcome": "Plan Outcome",
                    "directional_return_pct": "Dir Return %",
                    "trade_decision": "Action Taken",
                    "actual_trade_status": "Journal Status",
                    "actual_pnl_pct": "Actual PnL %",
                }
            )
            if "trade_decision" in archive_recent.columns:
                archive_recent["trade_decision"] = (
                    archive_recent["trade_decision"].fillna("").astype(str).str.strip().replace("", "—")
                )
            if "actual_trade_status" in archive_recent.columns:
                archive_recent["actual_trade_status"] = (
                    archive_recent["actual_trade_status"].fillna("").astype(str).str.strip().replace("", "—")
                )
            if "actual_pnl_pct" in archive_recent.columns:
                archive_recent["actual_pnl_pct"] = pd.to_numeric(
                    archive_recent["actual_pnl_pct"],
                    errors="coerce",
                )
            visible_cols = [
                "Signal Time",
                "Coin",
                "Setup Confirm",
                "Scalp State",
                "Scalp Reason",
                "Direction",
                "Confidence",
                "AI Confidence",
                "Entry",
                "Stop",
                "Target",
                "R:R",
                "Status",
                "Plan Outcome",
                "Dir Return %",
                "Action Taken",
                "Journal Status",
                "Actual PnL %",
            ]
            st.dataframe(
                archive_recent[[c for c in visible_cols if c in archive_recent.columns]].round(2),
                hide_index=True,
                width="stretch",
            )

    state_key = "scalp_bt_last_result"
    run_now = st.button("Run Scalp Outcome Study", type="primary", key="scalp_bt_run")
    current_inputs_signature = {
        "timeframe": str(timeframe),
        "lookback_candles": int(lookback_candles),
        "universe_size": int(universe_size),
        "forward_bars": int(forward_bars),
        "custom_coin_input": str(custom_coin_input or "").strip().upper(),
        "exclude_stables": bool(exclude_stables),
    }

    events_df: pd.DataFrame
    summary: dict
    pairs: list[str]
    fetch_limit: int
    symbols_with_events = 0
    no_data_symbols = 0
    failed_symbols = 0
    cache_hits = 0
    diag_bars_evaluated = 0
    diag_analysis_fail = 0
    diag_signal_side_reject = 0
    diag_plan_fail = 0
    diag_gate_pass_candidates = 0
    diag_side_key_reject = 0
    diag_price_level_reject = 0
    diag_forward_window_reject = 0
    diag_gate_reject_counts: Counter[str] = Counter()
    diag_plan_fail_counts: Counter[str] = Counter()
    processed_symbols = 0
    truncated_by_time = False
    runtime_budget_sec = 75.0
    showing_saved_result = False

    if run_now:
        fetch_limit = int(lookback_candles)
        pairs, _provider_rows = get_top_volume_usdt_symbols(top_n=int(universe_size), vs_currency="usd")
        pairs = [str(p).upper() for p in pairs if isinstance(p, str) and "/" in p]
        pairs = list(dict.fromkeys(pairs))[: int(universe_size)]
        if custom_mode_active:
            pairs = [f"{b}/USDT" for b in custom_bases]
        if exclude_stables:
            before_n = len(pairs)
            pairs = [p for p in pairs if not is_stable_base_symbol(p.split("/", 1)[0])]
            removed_n = max(0, before_n - len(pairs))
            if removed_n:
                st.caption(f"Stablecoin filter removed {removed_n} pair(s) from study universe.")
        if not pairs:
            st.error("Universe is empty. No exchange pairs available for scalp study.")
            return

        # Guardrail to keep UI responsive on large universe*lookback combinations.
        max_rows_budget = 90000
        estimated_rows = len(pairs) * int(fetch_limit)
        if estimated_rows > max_rows_budget and fetch_limit > 0:
            capped_n = max(10, int(max_rows_budget // int(fetch_limit)))
            capped_n = min(capped_n, len(pairs))
            if capped_n < len(pairs):
                st.warning(
                    f"Workload capped for responsiveness: reading {capped_n}/{len(pairs)} symbols "
                    f"(budget {max_rows_budget:,} rows)."
                )
                pairs = pairs[:capped_n]

        st.info(
            f"Fetching candles and finding scalp-matched events across {len(pairs)} symbols..."
        )
        all_events: list[pd.DataFrame] = []
        started_at = time.time()
        progress_bar = st.progress(0, text="Starting scalp scan...")
        for pair in pairs:
            elapsed = time.time() - started_at
            if elapsed > runtime_budget_sec:
                truncated_by_time = True
                break
            try:
                df_live = fetch_ohlcv(pair, timeframe, limit=fetch_limit)
                df, used_cache, _cache_ts = live_or_snapshot(
                    st,
                    f"scalp_outcome::{pair}::{timeframe}::{fetch_limit}::{forward_bars}",
                    df_live,
                    max_age_sec=1800,
                    current_sig=(pair, timeframe, fetch_limit, forward_bars),
                )
                if used_cache:
                    cache_hits += 1
                if df is None or df.empty:
                    no_data_symbols += 1
                    continue

                # Keep parity with Market tab: compute decisions on closed-candle context.
                df_eval = df.iloc[:-1].copy() if len(df) > 1 else df.copy()
                if df_eval is None or len(df_eval) <= 55:
                    no_data_symbols += 1
                    continue

                events_sym = build_scalp_outcome_study(
                    df=df_eval,
                    analyzer=analyse,
                    ml_predictor=ml_ensemble_predict,
                    conviction_fn=_calc_conviction,
                    signal_plain_fn=signal_plain,
                    direction_key_fn=direction_key,
                    get_scalping_entry_target_fn=get_scalping_entry_target,
                    scalp_quality_gate_fn=scalp_quality_gate,
                    sr_lookback_fn=_sr_lookback,
                    timeframe=timeframe,
                    df_4h=(df if timeframe == "4h" else fetch_ohlcv(pair, "4h", limit=260)),
                    df_1d=(df if timeframe == "1d" else fetch_ohlcv(pair, "1d", limit=260)),
                    forward_bars=forward_bars,
                )
                diag = {}
                try:
                    diag = dict(getattr(events_sym, "attrs", {}).get("diagnostics", {}) or {})
                except Exception:
                    diag = {}
                diag_bars_evaluated += int(diag.get("bars_evaluated", 0) or 0)
                diag_analysis_fail += int(diag.get("analysis_fail", 0) or 0)
                diag_signal_side_reject += int(diag.get("signal_side_reject", 0) or 0)
                diag_plan_fail += int(diag.get("plan_fail", 0) or 0)
                diag_gate_pass_candidates += int(diag.get("gate_pass_candidates", 0) or 0)
                diag_side_key_reject += int(diag.get("side_key_reject", 0) or 0)
                diag_price_level_reject += int(diag.get("price_level_reject", 0) or 0)
                diag_forward_window_reject += int(diag.get("forward_window_reject", 0) or 0)
                diag_gate_reject_counts.update(
                    {
                        str(k): int(v)
                        for k, v in dict(diag.get("gate_reject_counts", {}) or {}).items()
                        if str(k).strip()
                    }
                )
                diag_plan_fail_counts.update(
                    {
                        str(k): int(v)
                        for k, v in dict(diag.get("plan_fail_counts", {}) or {}).items()
                        if str(k).strip()
                    }
                )
                if events_sym is None or events_sym.empty:
                    continue
                events_sym = events_sym.copy()
                events_sym["Pair"] = pair
                events_sym["Coin"] = pair.split("/", 1)[0]
                all_events.append(events_sym)
                symbols_with_events += 1
            except Exception:
                failed_symbols += 1
            finally:
                processed_symbols += 1
                progress_bar.progress(
                    min(100, int((processed_symbols / max(1, len(pairs))) * 100)),
                    text=f"Scanning {processed_symbols}/{len(pairs)} symbols...",
                )

        progress_bar.empty()

        if not all_events:
            st.warning(
                "No scalp-matched events were found in this window. "
                "Try increasing lookback or using a faster timeframe."
            )
            if diag_gate_reject_counts:
                reject_parts = []
                for reason_code, count in diag_gate_reject_counts.most_common(4):
                    reject_parts.append(f"{_gate_reason_label(reason_code)} {int(count)}")
                if reject_parts:
                    st.caption(f"Top quality rejects: {' • '.join(reject_parts)}")
            if diag_plan_fail_counts:
                plan_parts = [f"{reason} {int(count)}" for reason, count in diag_plan_fail_counts.most_common(3)]
                if plan_parts:
                    st.caption(f"Top plan failures: {' • '.join(plan_parts)}")
            if (
                no_data_symbols
                or failed_symbols
                or diag_bars_evaluated
                or diag_signal_side_reject
                or diag_plan_fail
                or diag_analysis_fail
            ):
                st.caption(
                    f"Support data: no-data symbols={no_data_symbols}, failed symbols={failed_symbols}, "
                    f"cache hits={cache_hits}, bars evaluated={diag_bars_evaluated}, "
                    f"signal-neutral rejects={diag_signal_side_reject}, plan-fail={diag_plan_fail}, "
                    f"analysis-fail={diag_analysis_fail}, quality-pass candidates={diag_gate_pass_candidates}."
                )
            return

        events_df = pd.concat(all_events, ignore_index=True)
        events_df = events_df.sort_values("Event Time", ascending=False).reset_index(drop=True)
        summary = summarize_scalp_outcome_study(events_df, forward_bars)

        st.session_state[state_key] = {
            "events_df": events_df,
            "summary": summary,
            "pairs": list(pairs),
            "fetch_limit": int(fetch_limit),
            "symbols_with_events": int(symbols_with_events),
            "no_data_symbols": int(no_data_symbols),
            "failed_symbols": int(failed_symbols),
            "cache_hits": int(cache_hits),
            "diag_bars_evaluated": int(diag_bars_evaluated),
            "diag_analysis_fail": int(diag_analysis_fail),
            "diag_signal_side_reject": int(diag_signal_side_reject),
            "diag_plan_fail": int(diag_plan_fail),
            "diag_gate_pass_candidates": int(diag_gate_pass_candidates),
            "diag_side_key_reject": int(diag_side_key_reject),
            "diag_price_level_reject": int(diag_price_level_reject),
            "diag_forward_window_reject": int(diag_forward_window_reject),
            "diag_gate_reject_counts": dict(diag_gate_reject_counts),
            "diag_plan_fail_counts": dict(diag_plan_fail_counts),
            "processed_symbols": int(processed_symbols),
            "truncated_by_time": bool(truncated_by_time),
            "runtime_budget_sec": float(runtime_budget_sec),
            "inputs_signature": current_inputs_signature,
        }
    else:
        saved = st.session_state.get(state_key)
        if not isinstance(saved, dict):
            return
        saved_inputs_signature = saved.get("inputs_signature", {})
        if isinstance(saved_inputs_signature, dict) and saved_inputs_signature and saved_inputs_signature != current_inputs_signature:
            st.warning(
                "Inputs changed since the last run. Click 'Run Scalp Outcome Study' to refresh results for the current scope."
            )
            return
        events_df = saved.get("events_df", pd.DataFrame())
        if not isinstance(events_df, pd.DataFrame) or events_df.empty:
            return
        summary = saved.get("summary", summarize_scalp_outcome_study(events_df, forward_bars))
        pairs = list(saved.get("pairs", []))
        fetch_limit = int(saved.get("fetch_limit", lookback_candles))
        symbols_with_events = int(saved.get("symbols_with_events", 0))
        no_data_symbols = int(saved.get("no_data_symbols", 0))
        failed_symbols = int(saved.get("failed_symbols", 0))
        cache_hits = int(saved.get("cache_hits", 0))
        diag_bars_evaluated = int(saved.get("diag_bars_evaluated", 0))
        diag_analysis_fail = int(saved.get("diag_analysis_fail", 0))
        diag_signal_side_reject = int(saved.get("diag_signal_side_reject", 0))
        diag_plan_fail = int(saved.get("diag_plan_fail", 0))
        diag_gate_pass_candidates = int(saved.get("diag_gate_pass_candidates", 0))
        diag_side_key_reject = int(saved.get("diag_side_key_reject", 0))
        diag_price_level_reject = int(saved.get("diag_price_level_reject", 0))
        diag_forward_window_reject = int(saved.get("diag_forward_window_reject", 0))
        diag_gate_reject_counts = Counter(dict(saved.get("diag_gate_reject_counts", {}) or {}))
        diag_plan_fail_counts = Counter(dict(saved.get("diag_plan_fail_counts", {}) or {}))
        processed_symbols = int(saved.get("processed_symbols", len(pairs)))
        truncated_by_time = bool(saved.get("truncated_by_time", False))
        runtime_budget_sec = float(saved.get("runtime_budget_sec", runtime_budget_sec))
        showing_saved_result = True

    tp_rate = _safe_float(summary.get("tp_rate"), 0.0)
    sl_rate = _safe_float(summary.get("sl_rate"), 0.0)
    timeout_rate = _safe_float(summary.get("timeout_rate"), 0.0)
    median_outcome = _safe_float(summary.get("median_outcome"), 0.0)
    avg_hit = _avg_hit_bar(events_df)
    historical_overview_items, _historical_diagnostic_items = _build_historical_scalp_kpi_rows(
        summary=summary,
        events_df=events_df,
        forward_bars=forward_bars,
        symbols_with_events=symbols_with_events,
        universe_scanned=len(pairs),
        positive=POSITIVE,
        warning=WARNING,
        negative=NEGATIVE,
        muted=TEXT_MUTED,
    )

    avg_hit_text = f"+{avg_hit:.1f} bars" if np.isfinite(avg_hit) else "N/A"
    median_text = f"{median_outcome:+.2f}%"
    st.markdown("### Historical Study")
    st.caption("Simulation-only read using the current scalp planner and quality checks on historical closed candles.")
    study_badges: list[dict[str, object]] = [
        {"text": "Study complete", "tone": "positive"},
        {"text": f"{int(summary['occurrences'])} events • {symbols_with_events}/{len(pairs)} symbols", "tone": "accent"},
    ]
    if showing_saved_result:
        study_badges.append({"text": "Showing last run", "tone": "neutral"})
    if truncated_by_time:
        study_badges.append({"text": f"Partial run ({processed_symbols}/{len(pairs)})", "tone": "warning"})
    if int(summary["occurrences"]) < 30:
        study_badges.append({"text": "Small sample", "tone": "warning"})
    render_badge_row(st, badges=study_badges)
    st.markdown("#### Overview")
    render_kpi_grid(st, columns=4, items=historical_overview_items)
    historical_timing_cards = _build_historical_timing_cards(
        events_df,
        timeframe=timeframe,
        custom_bases=custom_bases,
        forward_bars=forward_bars,
        build_hold_window_intelligence=build_hold_window_intelligence,
    )
    st.markdown("#### Timing Read")
    st.caption("Study timing by direction inside the current timeframe scope.")
    _render_insight_card_grid(st, historical_timing_cards, columns=2)
    st.caption(
        f"Scalp read: TP {tp_rate:.1f}% • SL {sl_rate:.1f}% • Timeout {timeout_rate:.1f}% • "
        f"Avg hit speed {avg_hit_text} • Median outcome {median_text}."
    )
    if no_data_symbols or failed_symbols or cache_hits:
        st.caption(
            f"Study support data: cache hits={cache_hits}, no-data symbols={no_data_symbols}, "
            f"failed symbols={failed_symbols}, bars evaluated={diag_bars_evaluated}, "
            f"quality-pass candidates={diag_gate_pass_candidates}."
        )
    if truncated_by_time:
        st.caption(
            f"Runtime cap reached at about {int(runtime_budget_sec)}s, so this study shows a partial scan "
            f"from {processed_symbols}/{len(pairs)} symbols."
        )

    works_cards, fail_cards = _build_scalp_study_signal_cards(events_df, forward_bars=forward_bars)
    if works_cards:
        st.markdown("### What Works")
        _render_insight_card_grid(st, works_cards, columns=2)
    if fail_cards:
        st.markdown("### What Needs Care")
        _render_insight_card_grid(st, fail_cards, columns=2)

    direction_summary = _build_scalp_simulation_cohort_summary(events_df, "Direction", forward_bars=forward_bars)
    coin_summary = _build_scalp_simulation_cohort_summary(events_df, "Coin", forward_bars=forward_bars)
    setup_summary = _build_scalp_simulation_cohort_summary(events_df, "Setup Confirm", forward_bars=forward_bars)
    ai_direction_summary = _build_scalp_simulation_cohort_summary(events_df, "AI Direction", forward_bars=forward_bars)

    st.markdown("### Deep Dives")
    st.caption("Optional detail.")
    with st.expander("Direction & Coin", expanded=False):
        deep_dive_cols = st.columns(2, gap="medium")
        with deep_dive_cols[0]:
            st.markdown("##### By Direction")
            if direction_summary.empty:
                st.caption("No direction-level simulation breakdown available.")
            else:
                st.dataframe(
                    direction_summary[
                        [
                            "Direction",
                            "Signals",
                            "TP First %",
                            "SL First %",
                            "Timeout %",
                            "AvgRR",
                            "AvgOutcomePct",
                            "AvgHitBar",
                        ]
                    ].rename(
                        columns={
                            "AvgRR": "Avg R:R",
                            "AvgOutcomePct": "Avg Outcome %",
                            "AvgHitBar": "Avg Hit Bar",
                        }
                    ).round(2),
                    hide_index=True,
                    width="stretch",
                )
        with deep_dive_cols[1]:
            st.markdown("##### By Coin")
            if coin_summary.empty:
                st.caption("No coin-level simulation breakdown available.")
            else:
                st.dataframe(
                    coin_summary[
                        [
                            "Coin",
                            "Signals",
                            "TP First %",
                            "SL First %",
                            "Timeout %",
                            "AvgRR",
                            "AvgOutcomePct",
                            "MedianOutcomePct",
                        ]
                    ].rename(
                        columns={
                            "AvgRR": "Avg R:R",
                            "AvgOutcomePct": "Avg Outcome %",
                            "MedianOutcomePct": "Median Outcome %",
                        }
                    ).round(2),
                    hide_index=True,
                    width="stretch",
                )

    with st.expander("Setup & AI", expanded=False):
        deep_dive_cols_2 = st.columns(2, gap="medium")
        with deep_dive_cols_2[0]:
            st.markdown("##### By Setup Confirm")
            if setup_summary.empty:
                st.caption("No setup-family simulation breakdown available.")
            else:
                st.dataframe(
                    setup_summary[
                        [
                            "Setup Confirm",
                            "Signals",
                            "TP First %",
                            "SL First %",
                            "Timeout %",
                            "AvgOutcomePct",
                            "MedianOutcomePct",
                        ]
                    ].rename(
                        columns={
                            "AvgOutcomePct": "Avg Outcome %",
                            "MedianOutcomePct": "Median Outcome %",
                        }
                    ).round(2),
                    hide_index=True,
                    width="stretch",
                )
        with deep_dive_cols_2[1]:
            st.markdown("##### By AI Direction")
            if ai_direction_summary.empty:
                st.caption("No AI-direction simulation breakdown available.")
            else:
                st.dataframe(
                    ai_direction_summary[
                        [
                            "AI Direction",
                            "Signals",
                            "TP First %",
                            "SL First %",
                            "Timeout %",
                            "AvgOutcomePct",
                            "AvgHitBar",
                        ]
                    ].rename(
                        columns={
                            "AvgOutcomePct": "Avg Outcome %",
                            "AvgHitBar": "Avg Hit Bar",
                        }
                    ).round(2),
                    hide_index=True,
                    width="stretch",
                )

    with st.expander("Advanced Study Detail", expanded=False):
        st.markdown("#### Outcome mix")
        outcome = events_df.get("Outcome", pd.Series(dtype=object)).astype(str).str.upper()
        total_n = max(1, len(events_df))
        outcome_df = pd.DataFrame(
            {
                "Outcome": ["TP", "SL/BOTH", "TIMEOUT"],
                "Count": [
                    int((outcome == "TP").sum()),
                    int(outcome.isin({"SL", "BOTH"}).sum()),
                    int((outcome == "TIMEOUT").sum()),
                ],
            }
        )
        outcome_df["Share"] = outcome_df["Count"].map(lambda n: f"{(float(n) / total_n) * 100.0:.1f}%")
        _render_hover_table(
            outcome_df,
            {
                "Outcome": "Scalp exit type.",
                "Count": "How many events ended with this outcome.",
                "Share": "Outcome share in current filtered sample.",
            },
            table_id="scalp-outcome-mix",
        )

        st.markdown(f"#### TP/SL Hit Timing (next {forward_bars} bars)")

        def _cum_rate(df_slice: pd.DataFrame, outcome_keys: set[str]) -> list[float]:
            n = len(df_slice)
            if n <= 0:
                return [float("nan")] * forward_bars
            hit = pd.to_numeric(df_slice.get("Hit Bar", pd.Series(dtype=float)), errors="coerce")
            out = df_slice.get("Outcome", pd.Series(dtype=object)).astype(str).str.upper()
            out_mask = out.isin(outcome_keys)
            vals: list[float] = []
            for step in range(1, forward_bars + 1):
                vals.append(float(((out_mask) & (hit <= step)).mean() * 100.0))
            return vals

        fig = go.Figure()
        x_axis = list(range(1, forward_bars + 1))
        st.caption("TP-first timing by scalp direction across the scanned universe.")
        dir_specs = [("Upside", POSITIVE, "solid"), ("Downside", "#FF3366", "dot")]
        plotted = 0
        for dir_name, dir_color, dir_dash in dir_specs:
            dir_df = events_df[events_df["Direction"].astype(str) == dir_name]
            if dir_df.empty:
                continue
            tp_path = _cum_rate(dir_df, {"TP"})
            tp_rate_now = float((dir_df["Outcome"].astype(str).str.upper() == "TP").mean() * 100.0)
            fig.add_trace(
                go.Scatter(
                    x=x_axis,
                    y=tp_path,
                    mode="lines+markers",
                    name=f"{dir_name} TP-first (n={len(dir_df)}, total {tp_rate_now:.1f}%)",
                    line=dict(color=dir_color, width=2, dash=dir_dash),
                    marker=dict(size=5),
                )
            )
            plotted += 1

        sl_path = _cum_rate(events_df, {"SL", "BOTH"})
        timeout_path = _cum_rate(events_df, {"TIMEOUT"})
        fig.add_trace(
            go.Scatter(
                x=x_axis,
                y=sl_path,
                mode="lines",
                name="SL-first (all)",
                line=dict(color="#FF3366", width=1.6, dash="dash"),
            )
        )
        fig.add_trace(
            go.Scatter(
                x=x_axis,
                y=timeout_path,
                mode="lines",
                name="Timeout (all)",
                line=dict(color=WARNING, width=1.4, dash="dot"),
            )
        )
        if plotted == 0:
            st.info("No direction path available for the selected sample.")

        fig.update_layout(
            template="plotly_dark",
            height=320,
            xaxis_title="Bars After Event",
            yaxis_title="Cumulative Event Share (%)",
            showlegend=True,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0.0),
            yaxis=dict(range=[0, 100]),
        )
        st.plotly_chart(fig, width="stretch")

    with st.expander("Study Event Table", expanded=False):
        _render_event_table_html(events_df, forward_bars)

    export_df = events_df.rename(columns={"Strength": "Confidence"}) if "Strength" in events_df.columns else events_df
    csv_data = export_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download Scalp Outcome Study (CSV)",
        data=csv_data,
        file_name=f"scalp_universe_{timeframe}_n{len(pairs)}_lb{fetch_limit}_fwd{forward_bars}.csv",
        mime="text/csv",
    )
