"""Reusable archive intelligence built from resolved signal history."""

from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Mapping

import pandas as pd

from core.market_decision import compact_action_label, normalize_action_class


ACTIONABLE_SETUP_CLASSES = {"ENTER_TREND_AI", "ENTER_TREND_LED", "ENTER_AI_LED", "PROBE", "WATCH"}
MIN_SETUP_POCKET_ROWS = 8


@dataclass(frozen=True)
class ArchiveIntelligenceSnapshot:
    available: bool
    symbol: str = ""
    timeframe: str = ""
    setup_class: str = ""
    setup_label: str = "Building"
    direction: str = ""
    signals: int = 0
    completed: int = 0
    follow_through_pct: float = 0.0
    avg_dir_return_pct: float = 0.0
    avg_adverse_excursion_pct: float = 0.0
    score: float = 0.0
    policy_delta: float = 0.0
    coverage_factor: float = 0.0
    quality_label: str = "Building"


def archive_direction_key(value: object) -> str:
    side = str(value or "").strip().upper()
    if side in {"UPSIDE", "LONG", "BUY", "BULLISH", "STRONG BUY"}:
        return "UPSIDE"
    if side in {"DOWNSIDE", "SHORT", "SELL", "BEARISH", "STRONG SELL"}:
        return "DOWNSIDE"
    return ""


def archive_setup_class_key(value: object) -> str:
    key = normalize_action_class(str(value or ""))
    return key if key else "UNKNOWN"


def archive_setup_class_label(value: object) -> str:
    key = archive_setup_class_key(value)
    if key == "PROBE":
        return "EARLY"
    if key == "UNKNOWN":
        return "Other"
    return compact_action_label(key)


def annotate_archive_setup_class(df_events: pd.DataFrame) -> pd.DataFrame:
    if df_events is None or df_events.empty:
        return pd.DataFrame()
    out = df_events.copy()
    setup_series = out.get("setup_confirm", pd.Series(index=out.index, dtype=object))
    out["__setup_class"] = setup_series.fillna("").astype(str).map(archive_setup_class_key)
    out["__setup_label"] = out["__setup_class"].map(archive_setup_class_label)
    return out


def filter_archive_events_by_setup(df_events: pd.DataFrame, setup_filter_value: str) -> pd.DataFrame:
    if df_events is None or df_events.empty:
        return pd.DataFrame()
    d = annotate_archive_setup_class(df_events)
    filter_value = str(setup_filter_value or "").strip().upper()
    if filter_value in {"", "AUTO_BEST"}:
        return d[d["__setup_class"].isin(ACTIONABLE_SETUP_CLASSES)].copy()
    if filter_value == "ALL":
        return d
    return d[d["__setup_class"].eq(filter_value)].copy()


def archive_setup_pocket_score(
    *,
    follow_through_pct: float,
    avg_dir_return_pct: float,
    avg_adverse_excursion_pct: float,
    completed: int,
    completed_baseline: int = MIN_SETUP_POCKET_ROWS,
) -> float:
    sample_ratio = min(1.0, max(0.0, float(completed) / float(max(1, completed_baseline) * 2)))
    return (
        float(follow_through_pct)
        + (float(avg_dir_return_pct) * 5.0)
        - (max(0.0, float(avg_adverse_excursion_pct)) * 1.5)
        + (sample_ratio * 8.0)
    )


def _archive_policy_delta(
    *,
    follow_through_pct: float,
    avg_dir_return_pct: float,
    avg_adverse_excursion_pct: float,
    completed: int,
) -> float:
    sample_strength = min(1.0, max(0.0, float(completed) / 36.0))
    edge = ((float(follow_through_pct) - 50.0) / 8.0) + (float(avg_dir_return_pct) * 1.25)
    edge -= max(0.0, float(avg_adverse_excursion_pct)) * 0.35
    return max(-8.0, min(8.0, edge * sample_strength))


def _archive_quality_label(completed: int) -> str:
    if int(completed) >= 32:
        return "Strong"
    if int(completed) >= 16:
        return "Good"
    if int(completed) >= MIN_SETUP_POCKET_ROWS:
        return "Thin"
    return "Building"


def _archive_coverage_factor(completed: int) -> float:
    if int(completed) <= 0:
        return 0.0
    return min(1.0, max(0.0, float(completed) / 32.0))


def _prepared_archive_groups(
    df_events: pd.DataFrame,
    *,
    setup_filter_value: str,
    min_completed: int,
    symbol: str | None = None,
    timeframe: str | None = None,
    direction: str | None = None,
    setup_class: str | None = None,
) -> pd.DataFrame:
    if df_events is None or df_events.empty:
        return pd.DataFrame()
    required = {"status", "timeframe", "direction"}
    if not required.issubset(df_events.columns):
        return pd.DataFrame()
    d = filter_archive_events_by_setup(df_events, setup_filter_value)
    if d.empty:
        return pd.DataFrame()
    d["status"] = d["status"].fillna("").astype(str).str.upper()
    d["symbol"] = d.get("symbol", pd.Series(index=d.index, dtype=object)).fillna("").astype(str).str.strip().str.upper()
    d["timeframe"] = d["timeframe"].fillna("").astype(str).str.strip().str.lower()
    d["__direction_key"] = d["direction"].map(archive_direction_key)
    d["directional_return_pct"] = pd.to_numeric(d.get("directional_return_pct"), errors="coerce")
    d["adverse_excursion_pct"] = pd.to_numeric(d.get("adverse_excursion_pct"), errors="coerce")
    d = d[d["timeframe"].ne("") & d["__direction_key"].isin({"UPSIDE", "DOWNSIDE"})].copy()
    d = d[d["status"].eq("RESOLVED") & d["directional_return_pct"].notna()].copy()
    if symbol:
        d = d[d["symbol"].eq(str(symbol).strip().upper())].copy()
    if timeframe and str(timeframe).strip().lower() != "all":
        d = d[d["timeframe"].eq(str(timeframe).strip().lower())].copy()
    if direction:
        d = d[d["__direction_key"].eq(archive_direction_key(direction))].copy()
    if setup_class:
        d = d[d["__setup_class"].eq(archive_setup_class_key(setup_class))].copy()
    if d.empty:
        return pd.DataFrame()

    grouped = (
        d.groupby(["symbol", "timeframe", "__setup_class", "__setup_label", "__direction_key"], dropna=False)
        .agg(
            Signals=("status", "count"),
            Completed=("status", "count"),
            FollowThroughPct=(
                "directional_return_pct",
                lambda s: float((pd.to_numeric(s, errors="coerce").dropna() > 0).mean() * 100.0)
                if len(pd.to_numeric(s, errors="coerce").dropna())
                else 0.0,
            ),
            AvgDirReturnPct=("directional_return_pct", "mean"),
            AvgAdverseExcursionPct=("adverse_excursion_pct", "mean"),
        )
        .reset_index()
    )
    for metric_col in ("FollowThroughPct", "AvgDirReturnPct", "AvgAdverseExcursionPct"):
        grouped[metric_col] = pd.to_numeric(grouped[metric_col], errors="coerce").fillna(0.0)
    grouped = grouped[grouped["Completed"] >= int(max(1, min_completed))].copy()
    if grouped.empty:
        return grouped
    grouped["QualityScore"] = grouped.apply(
        lambda row: archive_setup_pocket_score(
            follow_through_pct=float(row.get("FollowThroughPct") or 0.0),
            avg_dir_return_pct=float(row.get("AvgDirReturnPct") or 0.0),
            avg_adverse_excursion_pct=float(row.get("AvgAdverseExcursionPct") or 0.0),
            completed=int(row.get("Completed") or 0),
            completed_baseline=min_completed,
        ),
        axis=1,
    )
    grouped["PolicyDelta"] = grouped.apply(
        lambda row: _archive_policy_delta(
            follow_through_pct=float(row.get("FollowThroughPct") or 0.0),
            avg_dir_return_pct=float(row.get("AvgDirReturnPct") or 0.0),
            avg_adverse_excursion_pct=float(row.get("AvgAdverseExcursionPct") or 0.0),
            completed=int(row.get("Completed") or 0),
        ),
        axis=1,
    )
    return grouped


def _snapshot_from_row(row: Mapping[str, object]) -> ArchiveIntelligenceSnapshot:
    completed = int(row.get("Completed") or 0)
    return ArchiveIntelligenceSnapshot(
        available=True,
        symbol=str(row.get("symbol") or "").strip().upper(),
        timeframe=str(row.get("timeframe") or "").strip().lower(),
        setup_class=str(row.get("__setup_class") or "").strip().upper(),
        setup_label=str(row.get("__setup_label") or "Other").strip() or "Other",
        direction=str(row.get("__direction_key") or "").strip().upper(),
        signals=int(row.get("Signals") or 0),
        completed=completed,
        follow_through_pct=float(row.get("FollowThroughPct") or 0.0),
        avg_dir_return_pct=float(row.get("AvgDirReturnPct") or 0.0),
        avg_adverse_excursion_pct=float(row.get("AvgAdverseExcursionPct") or 0.0),
        score=float(row.get("QualityScore") or 0.0),
        policy_delta=float(row.get("PolicyDelta") or 0.0),
        coverage_factor=_archive_coverage_factor(completed),
        quality_label=_archive_quality_label(completed),
    )


def build_archive_intelligence_snapshot(
    df_events: pd.DataFrame,
    *,
    setup_filter_value: str = "AUTO_BEST",
    min_completed: int = MIN_SETUP_POCKET_ROWS,
    symbol: str | None = None,
    timeframe: str | None = None,
    direction: str | None = None,
    setup_class: str | None = None,
) -> ArchiveIntelligenceSnapshot:
    grouped = _prepared_archive_groups(
        df_events,
        setup_filter_value=setup_filter_value,
        min_completed=min_completed,
        symbol=symbol,
        timeframe=timeframe,
        direction=direction,
        setup_class=setup_class,
    )
    if grouped.empty:
        return ArchiveIntelligenceSnapshot(available=False)
    best = grouped.sort_values(
        ["QualityScore", "FollowThroughPct", "AvgDirReturnPct", "Completed", "Signals"],
        ascending=[False, False, False, False, False],
    ).iloc[0]
    return _snapshot_from_row(best)


def build_archive_policy_map(
    df_events: pd.DataFrame,
    *,
    setup_filter_value: str = "AUTO_BEST",
    min_completed: int = MIN_SETUP_POCKET_ROWS,
) -> dict[tuple[str, str, str, str], ArchiveIntelligenceSnapshot]:
    grouped = _prepared_archive_groups(
        df_events,
        setup_filter_value=setup_filter_value,
        min_completed=min_completed,
    )
    if grouped.empty:
        return {}
    policy: dict[tuple[str, str, str, str], ArchiveIntelligenceSnapshot] = {}
    for row in grouped.to_dict("records"):
        snap = _snapshot_from_row(row)
        key = (snap.symbol, snap.timeframe, snap.setup_class, snap.direction)
        policy[key] = snap
    return policy


def archive_policy_for_signal(
    policy_map: Mapping[tuple[str, str, str, str], ArchiveIntelligenceSnapshot],
    *,
    symbol: object,
    timeframe: object,
    setup_confirm: object,
    direction: object,
) -> ArchiveIntelligenceSnapshot:
    key = (
        str(symbol or "").strip().upper(),
        str(timeframe or "").strip().lower(),
        archive_setup_class_key(setup_confirm),
        archive_direction_key(direction),
    )
    return policy_map.get(key, ArchiveIntelligenceSnapshot(available=False))
