"""Composable archive decision snapshot for learned signal reads."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import pandas as pd

from core.archive_expected_path import build_archive_expected_path_projection
from core.archive_intelligence import (
    ACTIONABLE_SETUP_CLASSES,
    MIN_SETUP_POCKET_ROWS,
    ArchiveIntelligenceSnapshot,
    annotate_archive_setup_class,
    archive_direction_key,
    archive_setup_class_key,
    archive_setup_class_label,
    build_archive_intelligence_snapshot,
)
from core.signal_tracker import build_hold_window_intelligence as default_build_hold_window_intelligence


_ARCHIVE_FEEDBACK_HALF_LIFE_DAYS = 14.0
_ARCHIVE_FEEDBACK_OLD_WEIGHT_FLOOR = 0.35


@dataclass(frozen=True)
class ArchiveDecisionSnapshot:
    """Single reusable read assembled from archive history."""

    available: bool
    setup: ArchiveIntelligenceSnapshot
    metric_events: pd.DataFrame
    direction_events: pd.DataFrame
    metric_windows: pd.DataFrame
    hold_events: pd.DataFrame
    hold_windows: pd.DataFrame
    hold_window: dict[str, object]
    expected_path: dict[str, object]
    scope_label: str = ""
    confidence_factor: float = 0.0
    confidence_tier: str = "Building"


@dataclass(frozen=True)
class ArchiveDecisionFeedback:
    """How much the system should trust archive ranking nudges right now."""

    active: bool = False
    sample: int = 0
    hit_rate_pct: float = 0.0
    avg_signed_return_pct: float = 0.0
    multiplier: float = 1.0
    expectancy_multiplier: float = 1.0


def archive_symbol_key(value: object) -> str:
    text = str(value or "").strip().upper()
    if not text:
        return ""
    for separator in ("/", "-", "_", " "):
        if separator in text:
            text = text.split(separator, 1)[0].strip()
            break
    for quote_suffix in ("USDT", "USDC", "FDUSD", "BUSD", "USD"):
        if text.endswith(quote_suffix) and len(text) > len(quote_suffix) + 1:
            candidate = text[: -len(quote_suffix)].strip()
            if candidate.isalpha():
                text = candidate
                break
    return text


def _archive_float(value: object, default: float = 0.0) -> float:
    try:
        out = float(value)
    except Exception:
        return float(default)
    if pd.isna(out):
        return float(default)
    return float(out)


def _archive_feedback_delta_series(df_events: pd.DataFrame) -> pd.Series:
    total_delta = (
        pd.to_numeric(df_events["archive_total_delta"], errors="coerce")
        if "archive_total_delta" in df_events.columns
        else pd.Series(index=df_events.index, dtype=float)
    )
    decision_delta = (
        pd.to_numeric(df_events["archive_decision_delta"], errors="coerce")
        if "archive_decision_delta" in df_events.columns
        else pd.Series(index=df_events.index, dtype=float)
    )
    return total_delta.where(total_delta.notna(), decision_delta)


def _archive_feedback_recency_weights(df_events: pd.DataFrame) -> pd.Series:
    if df_events is None or df_events.empty:
        return pd.Series(dtype=float)

    timestamp_series: pd.Series | None = None
    for column in ("resolved_at", "event_time"):
        if column not in df_events.columns:
            continue
        parsed = pd.to_datetime(df_events[column], errors="coerce", utc=True)
        if parsed.notna().any():
            timestamp_series = pd.Series(parsed, index=df_events.index)
            break

    if timestamp_series is None:
        return pd.Series(1.0, index=df_events.index, dtype=float)

    valid_timestamps = timestamp_series.notna()
    if not bool(valid_timestamps.any()):
        return pd.Series(1.0, index=df_events.index, dtype=float)

    anchor = timestamp_series[valid_timestamps].max()
    age_days = (anchor - timestamp_series).dt.total_seconds().div(86400.0).clip(lower=0.0)
    decay = 0.5 ** (age_days / _ARCHIVE_FEEDBACK_HALF_LIFE_DAYS)
    floor = float(_ARCHIVE_FEEDBACK_OLD_WEIGHT_FLOOR)
    weights = floor + ((1.0 - floor) * decay)
    return pd.to_numeric(weights, errors="coerce").fillna(floor).clip(lower=floor, upper=1.0)


def _archive_weighted_average(values: pd.Series, weights: pd.Series, default: float = 0.0) -> float:
    numeric_values = pd.to_numeric(values, errors="coerce")
    numeric_weights = pd.to_numeric(weights.reindex(numeric_values.index), errors="coerce").fillna(1.0)
    valid = numeric_values.notna() & numeric_weights.gt(0)
    if not bool(valid.any()):
        return float(default)
    denominator = float(numeric_weights[valid].sum())
    if denominator <= 0:
        return float(default)
    return float((numeric_values[valid] * numeric_weights[valid]).sum() / denominator)


def _feedback_from_prepared_frame(
    df_events: pd.DataFrame,
    *,
    min_samples: int,
) -> ArchiveDecisionFeedback:
    if df_events is None or df_events.empty:
        return ArchiveDecisionFeedback()
    d = df_events[df_events["__archive_delta"].abs().ge(0.05) & df_events["__return"].notna()].copy()
    if d.empty:
        return ArchiveDecisionFeedback()

    archive_supported = d["__archive_delta"].gt(0)
    outcome_worked = d["__return"].gt(0)
    archive_hit = archive_supported.eq(outcome_worked)
    sample = int(len(d))
    signed_return = d["__return"] * d["__archive_delta"].map(lambda value: 1.0 if float(value) >= 0 else -1.0)
    recency_weights = _archive_feedback_recency_weights(d)
    hit_rate = float(_archive_weighted_average(archive_hit.astype(float), recency_weights) * 100.0)
    avg_signed_return = _archive_float(_archive_weighted_average(signed_return, recency_weights))
    if sample < int(max(1, min_samples)):
        return ArchiveDecisionFeedback(
            active=False,
            sample=sample,
            hit_rate_pct=hit_rate,
            avg_signed_return_pct=avg_signed_return,
            multiplier=1.0,
            expectancy_multiplier=1.0,
        )

    sample_strength = min(1.0, max(0.0, sample / 96.0))
    hit_edge = (hit_rate - 50.0) / 50.0
    return_edge = max(-1.0, min(1.0, avg_signed_return / 2.0))
    trust_edge = (hit_edge * 0.32) + (return_edge * 0.12)
    multiplier = 1.0 + (trust_edge * sample_strength)
    expectancy_multiplier = 1.0 + (((hit_edge * 0.24) + (return_edge * 0.09)) * sample_strength)

    return ArchiveDecisionFeedback(
        active=True,
        sample=sample,
        hit_rate_pct=hit_rate,
        avg_signed_return_pct=avg_signed_return,
        multiplier=max(0.55, min(1.15, multiplier)),
        expectancy_multiplier=max(0.65, min(1.10, expectancy_multiplier)),
    )


def archive_decision_score_adjustment(snapshot: object) -> tuple[float, float]:
    """Return quiet ranking deltas from exact archive timing/path quality."""

    hold = getattr(snapshot, "hold_window", {}) or {}
    expected_path = getattr(snapshot, "expected_path", {}) or {}
    timing_delta = 0.0
    if isinstance(hold, dict) and bool(hold.get("available")):
        sample = max(_archive_float(hold.get("sample")), _archive_float(hold.get("resolved_signals")))
        sample_strength = min(1.0, max(0.0, sample / 32.0))
        follow_edge = (_archive_float(hold.get("follow_through_pct")) - 50.0) / 13.0
        move_edge = _archive_float(hold.get("avg_dir_return_pct")) * 0.85
        adverse_drag = max(0.0, _archive_float(hold.get("avg_adverse_excursion_pct"))) * 0.35
        timing_delta = max(-4.0, min(4.0, (follow_edge + move_edge - adverse_drag) * sample_strength))

    path_delta = 0.0
    if isinstance(expected_path, dict) and bool(expected_path.get("available")):
        sample = max(
            _archive_float(expected_path.get("sample")),
            _archive_float(expected_path.get("archive_check_sample")),
        )
        sample_strength = min(1.0, max(0.0, sample / 32.0))
        follow_edge = (_archive_float(expected_path.get("follow_through_pct")) - 50.0) / 16.0
        zone_edge = (_archive_float(expected_path.get("zone_hit_rate_pct")) - 50.0) / 24.0
        clean_edge = (_archive_float(expected_path.get("clean_path_rate_pct")) - 50.0) / 34.0
        caution_drag = max(0.0, _archive_float(expected_path.get("caution_break_rate_pct"))) / 38.0
        path_delta = max(-3.0, min(3.0, (follow_edge + zone_edge + clean_edge - caution_drag) * sample_strength))
        if bool(expected_path.get("path_conflict")):
            path_delta *= 0.55

    archive_delta = max(-5.0, min(5.0, timing_delta + path_delta))
    expectancy_delta = max(-3.5, min(3.5, (timing_delta * 0.55) + (path_delta * 0.75)))
    return archive_delta, expectancy_delta


def archive_invalidation_risk(snapshot: object) -> float:
    """Estimate how much the learned path is at risk of breaking down."""

    expected_path = getattr(snapshot, "expected_path", {}) or {}
    if not isinstance(expected_path, dict) or not bool(expected_path.get("available")):
        return 0.0

    sample = max(
        _archive_float(expected_path.get("sample")),
        _archive_float(expected_path.get("archive_check_sample")),
    )
    if sample <= 0:
        return 0.0
    sample_strength = min(1.0, max(0.0, sample / 24.0))
    zone_hit = _archive_float(expected_path.get("zone_hit_rate_pct"), 50.0)
    clean_path = _archive_float(expected_path.get("clean_path_rate_pct"), 50.0)
    caution_break = _archive_float(expected_path.get("caution_break_rate_pct"), 0.0)
    follow_through = _archive_float(expected_path.get("follow_through_pct"), 50.0)

    risk = 0.0
    if bool(expected_path.get("path_conflict")):
        risk += 0.35
    if caution_break > 20.0:
        risk += min(0.35, (caution_break - 20.0) / 80.0)
    if clean_path < 45.0:
        risk += min(0.25, (45.0 - clean_path) / 80.0)
    if zone_hit < 45.0:
        risk += min(0.20, (45.0 - zone_hit) / 100.0)
    if follow_through < 50.0:
        risk += min(0.20, (50.0 - follow_through) / 100.0)
    return max(0.0, min(0.75, risk * sample_strength))


def archive_confidence_tier(factor: float | int | None) -> str:
    confidence = max(0.0, min(1.0, _archive_float(factor)))
    if confidence >= 0.82:
        return "Strong"
    if confidence >= 0.58:
        return "Good"
    if confidence >= 0.34:
        return "Thin"
    return "Building"


def archive_decision_confidence_factor(snapshot: object) -> float:
    setup = getattr(snapshot, "setup", None)
    hold = getattr(snapshot, "hold_window", {}) or {}
    expected_path = getattr(snapshot, "expected_path", {}) or {}

    setup_completed = _archive_float(getattr(setup, "completed", 0.0) if setup is not None else 0.0)
    setup_coverage = _archive_float(getattr(setup, "coverage_factor", 0.0) if setup is not None else 0.0)
    hold_sample = max(_archive_float(hold.get("sample")), _archive_float(hold.get("resolved_signals")))
    path_sample = 0.0
    path_quality_score = 0.0
    if isinstance(expected_path, dict) and bool(expected_path.get("available")):
        path_sample = max(
            _archive_float(expected_path.get("sample")),
            _archive_float(expected_path.get("archive_check_sample")),
        )
        quality = str(expected_path.get("read_quality") or "").strip().lower()
        if quality == "strong":
            path_quality_score = 1.0
        elif quality == "good":
            path_quality_score = 0.72
        elif quality == "thin":
            path_quality_score = 0.42
        else:
            path_quality_score = min(0.35, path_sample / 32.0)

    depth_factor = max(
        min(1.0, setup_completed / 32.0),
        min(1.0, hold_sample / 32.0),
        min(1.0, path_sample / 32.0),
    )
    checkpoint_factor = 0.0
    if hold_sample > 0 and path_sample > 0:
        checkpoint_factor = min(1.0, path_sample / max(1.0, hold_sample))
    elif path_sample > 0:
        checkpoint_factor = min(1.0, path_sample / 32.0)
    elif hold_sample > 0:
        checkpoint_factor = min(0.55, hold_sample / 32.0)
    setup_factor = max(setup_coverage, min(1.0, setup_completed / 32.0))
    risk_discount = 1.0 - (archive_invalidation_risk(snapshot) * 0.45)

    confidence = (
        (depth_factor * 0.40)
        + (checkpoint_factor * 0.24)
        + (setup_factor * 0.20)
        + (path_quality_score * 0.16)
    ) * max(0.55, min(1.0, risk_discount))
    return max(0.0, min(1.0, confidence))


def apply_archive_invalidation_guardrail(
    archive_delta: float | int | None,
    expectancy_delta: float | int | None,
    snapshot: object,
) -> tuple[float, float]:
    """Dampen archive boosts when the expected path has learned invalidation risk."""

    raw_archive = _archive_float(archive_delta)
    raw_expectancy = _archive_float(expectancy_delta)
    risk = archive_invalidation_risk(snapshot)
    if risk <= 0:
        return raw_archive, raw_expectancy

    positive_multiplier = max(0.25, 1.0 - risk)
    negative_multiplier = min(1.35, 1.0 + (risk * 0.45))

    guarded_archive = raw_archive * (positive_multiplier if raw_archive >= 0 else negative_multiplier)
    guarded_expectancy = raw_expectancy * (positive_multiplier if raw_expectancy >= 0 else negative_multiplier)
    return (
        max(-20.0, min(20.0, guarded_archive)),
        max(-20.0, min(20.0, guarded_expectancy)),
    )


def apply_archive_confidence_guardrail(
    archive_delta: float | int | None,
    expectancy_delta: float | int | None,
    snapshot: object,
) -> tuple[float, float]:
    """Scale archive ranking impact by central archive confidence."""

    raw_archive = _archive_float(archive_delta)
    raw_expectancy = _archive_float(expectancy_delta)
    confidence = max(0.0, min(1.0, _archive_float(getattr(snapshot, "confidence_factor", None), -1.0)))
    if confidence < 0:
        confidence = archive_decision_confidence_factor(snapshot)
    multiplier = 0.25 + (0.75 * confidence)
    return (
        max(-20.0, min(20.0, raw_archive * multiplier)),
        max(-20.0, min(20.0, raw_expectancy * multiplier)),
    )


def archive_decision_observability(
    snapshot: object,
    feedback: ArchiveDecisionFeedback | object | None = None,
) -> dict[str, object]:
    """Return hidden audit fields explaining archive ranking impact."""

    confidence_factor = max(
        0.0,
        min(1.0, _archive_float(getattr(snapshot, "confidence_factor", None), -1.0)),
    )
    if confidence_factor < 0:
        confidence_factor = archive_decision_confidence_factor(snapshot)
    confidence_tier = str(getattr(snapshot, "confidence_tier", "") or "").strip()
    if not confidence_tier:
        confidence_tier = archive_confidence_tier(confidence_factor)
    return {
        "archive_confidence_factor": confidence_factor,
        "archive_confidence_tier": confidence_tier,
        "archive_invalidation_risk": archive_invalidation_risk(snapshot),
        "archive_feedback_multiplier": _archive_float(getattr(feedback, "multiplier", 1.0), 1.0),
    }


def build_archive_decision_feedback_model(
    df_events: pd.DataFrame | None,
    *,
    min_samples: int = 24,
) -> ArchiveDecisionFeedback:
    """Learn whether previous archive nudges improved resolved signal outcomes."""

    if df_events is None or df_events.empty:
        return ArchiveDecisionFeedback()
    if "directional_return_pct" not in df_events.columns or not (
        "archive_total_delta" in df_events.columns or "archive_decision_delta" in df_events.columns
    ):
        return ArchiveDecisionFeedback()

    d = df_events.copy()
    if "status" in d.columns:
        d = d[d["status"].fillna("").astype(str).str.strip().str.upper().eq("RESOLVED")].copy()
    d["__archive_delta"] = _archive_feedback_delta_series(d)
    d["__return"] = pd.to_numeric(d["directional_return_pct"], errors="coerce")
    return _feedback_from_prepared_frame(d, min_samples=min_samples)


def archive_decision_feedback_key(
    *,
    symbol: object = "",
    timeframe: object = "",
    setup_confirm: object = "",
    direction: object = "",
) -> tuple[str, str, str, str]:
    return (
        archive_symbol_key(symbol),
        str(timeframe or "").strip().lower(),
        archive_setup_class_key(setup_confirm),
        archive_direction_key(direction),
    )


def archive_decision_context_key(value: object) -> str:
    text = str(value or "").strip().upper()
    if not text or text in {"UNKNOWN", "N/A", "NONE", "NULL", "NAN"}:
        return ""
    compact = "".join(ch if ch.isalnum() else "_" for ch in text)
    while "__" in compact:
        compact = compact.replace("__", "_")
    return compact.strip("_")


def _context_series(d: pd.DataFrame, key_col: str, label_col: str) -> pd.Series:
    key_values = (
        d[key_col].fillna("").astype(str).str.strip()
        if key_col in d.columns
        else pd.Series("", index=d.index, dtype=object)
    )
    label_values = (
        d[label_col].fillna("").astype(str).str.strip()
        if label_col in d.columns
        else pd.Series("", index=d.index, dtype=object)
    )
    return key_values.where(key_values.ne(""), label_values).map(archive_decision_context_key)


def build_archive_decision_feedback_map(
    df_events: pd.DataFrame | None,
    *,
    min_samples: int = 18,
) -> dict[tuple[str, ...], ArchiveDecisionFeedback]:
    """Build scoped archive-feedback models with broad fallbacks."""

    if df_events is None or df_events.empty:
        return {}
    if "directional_return_pct" not in df_events.columns or not (
        "archive_total_delta" in df_events.columns or "archive_decision_delta" in df_events.columns
    ):
        return {}
    d = _prepare_events(df_events)
    d = _resolved_events(d)
    if d.empty:
        return {}
    d["__archive_delta"] = _archive_feedback_delta_series(d)
    d["__return"] = pd.to_numeric(d["directional_return_pct"], errors="coerce")
    d["__playbook_key"] = _context_series(d, "market_playbook_key", "market_playbook")
    d["__trade_gate_key"] = _context_series(d, "market_trade_gate_key", "market_trade_gate")
    d = d[
        d["__archive_delta"].abs().ge(0.05)
        & d["__return"].notna()
        & d["timeframe"].ne("")
        & d["__setup_class"].ne("")
        & d["__direction_key"].isin({"UPSIDE", "DOWNSIDE"})
    ].copy()
    if d.empty:
        return {}

    feedback_map: dict[tuple[str, ...], ArchiveDecisionFeedback] = {}
    context_group_specs = [
        (
            ["symbol", "timeframe", "__setup_class", "__direction_key", "__playbook_key", "__trade_gate_key"],
            lambda row: (row[0], row[1], row[2], row[3], row[4], row[5]),
        ),
        (
            ["symbol", "__setup_class", "__direction_key", "__playbook_key", "__trade_gate_key"],
            lambda row: (row[0], "", row[1], row[2], row[3], row[4]),
        ),
        (
            ["symbol", "timeframe", "__direction_key", "__playbook_key", "__trade_gate_key"],
            lambda row: (row[0], row[1], "", row[2], row[3], row[4]),
        ),
        (
            ["symbol", "__direction_key", "__playbook_key", "__trade_gate_key"],
            lambda row: (row[0], "", "", row[1], row[2], row[3]),
        ),
        (
            ["timeframe", "__setup_class", "__direction_key", "__playbook_key", "__trade_gate_key"],
            lambda row: ("", row[0], row[1], row[2], row[3], row[4]),
        ),
        (
            ["__setup_class", "__direction_key", "__playbook_key", "__trade_gate_key"],
            lambda row: ("", "", row[0], row[1], row[2], row[3]),
        ),
        (
            ["timeframe", "__direction_key", "__playbook_key", "__trade_gate_key"],
            lambda row: ("", row[0], "", row[1], row[2], row[3]),
        ),
        (
            ["__direction_key", "__playbook_key", "__trade_gate_key"],
            lambda row: ("", "", "", row[0], row[1], row[2]),
        ),
    ]
    context_rows = d[d["__playbook_key"].ne("") | d["__trade_gate_key"].ne("")].copy()
    if not context_rows.empty:
        for group_cols, key_builder in context_group_specs:
            for values, group in context_rows.groupby(group_cols, dropna=False):
                value_tuple = values if isinstance(values, tuple) else (values,)
                normalized_values = tuple(str(item or "").strip() for item in value_tuple)
                feedback = _feedback_from_prepared_frame(group, min_samples=min_samples)
                if bool(feedback.active):
                    feedback_map[key_builder(normalized_values)] = feedback

    group_specs = [
        (["symbol", "timeframe", "__setup_class", "__direction_key"], lambda row: (row[0], row[1], row[2], row[3])),
        (["symbol", "__setup_class", "__direction_key"], lambda row: (row[0], "", row[1], row[2])),
        (["symbol", "timeframe", "__direction_key"], lambda row: (row[0], row[1], "", row[2])),
        (["symbol", "__direction_key"], lambda row: (row[0], "", "", row[1])),
        (["timeframe", "__setup_class", "__direction_key"], lambda row: ("", row[0], row[1], row[2])),
        (["__setup_class", "__direction_key"], lambda row: ("", "", row[0], row[1])),
        (["timeframe", "__direction_key"], lambda row: ("", row[0], "", row[1])),
        (["__direction_key"], lambda row: ("", "", "", row[0])),
    ]
    for group_cols, key_builder in group_specs:
        for values, group in d.groupby(group_cols, dropna=False):
            value_tuple = values if isinstance(values, tuple) else (values,)
            feedback = _feedback_from_prepared_frame(group, min_samples=min_samples)
            if bool(feedback.active):
                feedback_map[key_builder(tuple(str(item or "").strip() for item in value_tuple))] = feedback
    return feedback_map


def archive_decision_feedback_for_signal(
    feedback_map: dict[tuple[str, ...], ArchiveDecisionFeedback] | None,
    fallback_feedback: ArchiveDecisionFeedback | object | None = None,
    *,
    symbol: object = "",
    timeframe: object = "",
    setup_confirm: object = "",
    direction: object = "",
    playbook_key: object = "",
    trade_gate_key: object = "",
) -> ArchiveDecisionFeedback | object | None:
    key = archive_decision_feedback_key(
        symbol=symbol,
        timeframe=timeframe,
        setup_confirm=setup_confirm,
        direction=direction,
    )
    symbol_key, tf, setup, side = key
    playbook = archive_decision_context_key(playbook_key)
    trade_gate = archive_decision_context_key(trade_gate_key)
    exact_context_candidates = []
    symbol_context_candidates = []
    broad_context_candidates = []
    if playbook or trade_gate:
        exact_context_candidates = [
            (symbol_key, tf, setup, side, playbook, trade_gate),
            (symbol_key, tf, setup, side, playbook, ""),
            (symbol_key, tf, setup, side, "", trade_gate),
        ]
        symbol_context_candidates = [
            (symbol_key, "", setup, side, playbook, trade_gate),
            (symbol_key, tf, "", side, playbook, trade_gate),
            (symbol_key, "", "", side, playbook, trade_gate),
        ]
        broad_context_candidates = [
            ("", tf, setup, side, playbook, trade_gate),
            ("", "", setup, side, playbook, trade_gate),
            ("", tf, "", side, playbook, trade_gate),
            ("", "", "", side, playbook, trade_gate),
        ]

    exact_signal_candidates = [
        (symbol_key, tf, setup, side),
    ]
    symbol_signal_candidates = [
        (symbol_key, "", setup, side),
        (symbol_key, tf, "", side),
        (symbol_key, "", "", side),
    ]
    broad_signal_candidates = [
        ("", tf, setup, side),
        ("", "", setup, side),
        ("", tf, "", side),
        ("", "", "", side),
    ]
    candidates = (
        exact_context_candidates
        + exact_signal_candidates
        + symbol_context_candidates
        + symbol_signal_candidates
        + broad_context_candidates
        + broad_signal_candidates
    )
    for candidate in candidates:
        feedback = (feedback_map or {}).get(candidate)
        if feedback is not None and bool(getattr(feedback, "active", False)):
            return feedback
    return fallback_feedback


def calibrate_archive_decision_scores(
    archive_delta: float | int | None,
    expectancy_delta: float | int | None,
    feedback: ArchiveDecisionFeedback | object | None,
) -> tuple[float, float]:
    """Apply the closed-loop feedback model without changing visible scanner rules."""

    raw_archive = _archive_float(archive_delta)
    raw_expectancy = _archive_float(expectancy_delta)
    if feedback is None or not bool(getattr(feedback, "active", False)):
        return raw_archive, raw_expectancy
    return (
        raw_archive * _archive_float(getattr(feedback, "multiplier", 1.0), 1.0),
        raw_expectancy * _archive_float(getattr(feedback, "expectancy_multiplier", 1.0), 1.0),
    )


def _empty_frame_like(df: pd.DataFrame | None = None) -> pd.DataFrame:
    if isinstance(df, pd.DataFrame):
        return pd.DataFrame(columns=df.columns)
    return pd.DataFrame()


def _prepare_events(df_events: pd.DataFrame | None) -> pd.DataFrame:
    if df_events is None or df_events.empty:
        return pd.DataFrame()
    out = annotate_archive_setup_class(df_events)
    for col in ("symbol", "timeframe", "direction", "signal_key", "status"):
        if col not in out.columns:
            out[col] = ""
    out["symbol"] = out["symbol"].fillna("").astype(str).str.strip().str.upper()
    out["timeframe"] = out["timeframe"].fillna("").astype(str).str.strip().str.lower()
    out["signal_key"] = out["signal_key"].fillna("").astype(str).str.strip()
    out["status"] = out["status"].fillna("").astype(str).str.strip().str.upper()
    out["__direction_key"] = out["direction"].map(archive_direction_key)
    return out


def _resolved_events(df_events: pd.DataFrame) -> pd.DataFrame:
    if df_events is None or df_events.empty or "status" not in df_events.columns:
        return pd.DataFrame()
    return df_events[df_events["status"].eq("RESOLVED")].copy()


def _filter_events_to_setup_direction(
    df_events: pd.DataFrame,
    setup: ArchiveIntelligenceSnapshot,
) -> pd.DataFrame:
    if df_events is None or df_events.empty or not bool(setup.available):
        return pd.DataFrame()
    return df_events[
        df_events["__setup_class"].eq(str(setup.setup_class or "").strip().upper())
        & df_events["__direction_key"].eq(str(setup.direction or "").strip().upper())
    ].copy()


def _filter_events_to_setup_pocket(
    df_events: pd.DataFrame,
    setup: ArchiveIntelligenceSnapshot,
) -> pd.DataFrame:
    if df_events is None or df_events.empty or not bool(setup.available):
        return pd.DataFrame()
    return df_events[
        df_events["timeframe"].eq(str(setup.timeframe or "").strip().lower())
        & df_events["__setup_class"].eq(str(setup.setup_class or "").strip().upper())
        & df_events["__direction_key"].eq(str(setup.direction or "").strip().upper())
    ].copy()


def _prepare_windows(df_forward_windows: pd.DataFrame | None) -> pd.DataFrame:
    if df_forward_windows is None or df_forward_windows.empty:
        return pd.DataFrame()
    out = df_forward_windows.copy()
    if "signal_key" not in out.columns:
        out["signal_key"] = ""
    out["signal_key"] = out["signal_key"].fillna("").astype(str).str.strip()
    return out


def _filter_windows_to_events(df_windows: pd.DataFrame, df_events: pd.DataFrame) -> pd.DataFrame:
    if (
        df_windows is None
        or df_windows.empty
        or df_events is None
        or df_events.empty
        or "signal_key" not in df_events.columns
        or "signal_key" not in df_windows.columns
    ):
        return _empty_frame_like(df_windows)
    keys = set(df_events["signal_key"].fillna("").astype(str).str.strip())
    keys.discard("")
    if not keys:
        return _empty_frame_like(df_windows)
    return df_windows[df_windows["signal_key"].isin(keys)].copy()


def select_archive_signal_scope_events(
    df_events: pd.DataFrame,
    *,
    symbol: str,
    timeframe: str,
    direction: str,
    setup_confirm: str = "",
    min_rows: int = MIN_SETUP_POCKET_ROWS,
) -> tuple[pd.DataFrame, str]:
    """Select the archive scope for a live signal without borrowing wrong setup history."""

    if df_events is None or df_events.empty:
        return pd.DataFrame(), "No archive sample"
    d = _prepare_events(df_events)
    d = _resolved_events(d)
    if d.empty:
        return pd.DataFrame(), "No resolved archive sample"

    symbol_key = archive_symbol_key(symbol)
    timeframe_key = str(timeframe or "").strip().lower()
    direction_key = archive_direction_key(direction)
    setup_class = archive_setup_class_key(setup_confirm)
    setup_label = archive_setup_class_label(setup_class)
    setup_is_actionable = setup_class in ACTIONABLE_SETUP_CLASSES
    direction_mark = "↑" if direction_key == "UPSIDE" else "↓" if direction_key == "DOWNSIDE" else ""
    setup_scope = f"{setup_label} {direction_mark}".strip()

    def _match(
        frame: pd.DataFrame,
        *,
        require_symbol: bool,
        require_timeframe: bool,
        require_direction: bool,
        require_setup: bool = False,
    ) -> pd.DataFrame:
        out = frame.copy()
        if require_symbol:
            out = out[out["symbol"].map(archive_symbol_key).fillna("").astype(str).str.upper().eq(symbol_key)].copy()
        if require_timeframe:
            out = out[out["timeframe"].eq(timeframe_key)].copy()
        if require_direction:
            out = out[out["__direction_key"].eq(direction_key)].copy()
        if require_setup:
            out = out[out["__setup_class"].eq(setup_class)].copy()
        return out.copy()

    if setup_is_actionable:
        setup_exact = _match(
            d,
            require_symbol=True,
            require_timeframe=True,
            require_direction=True,
            require_setup=True,
        )
        label = f"{symbol_key} {str(timeframe).upper()} {setup_scope}".strip()
        if not setup_exact.empty:
            return setup_exact, label
        return pd.DataFrame(), f"No {label} archive pocket yet"

    direction_label = str(direction_key or "Archive").title()
    candidates = [
        (_match(d, require_symbol=True, require_timeframe=True, require_direction=True), f"{symbol_key} {str(timeframe).upper()} {direction_label}"),
        (_match(d, require_symbol=True, require_timeframe=False, require_direction=True), f"{symbol_key} {direction_label}"),
        (_match(d, require_symbol=False, require_timeframe=True, require_direction=True), f"{str(timeframe).upper()} {direction_label} market"),
        (_match(d, require_symbol=False, require_timeframe=False, require_direction=True), f"broader {direction_label} archive"),
    ]
    for frame, label in candidates:
        if len(frame) >= int(max(1, min_rows)):
            return frame, label
    for frame, label in candidates:
        if not frame.empty:
            return frame, label
    return pd.DataFrame(), "No matching archive sample"


def build_archive_signal_decision_snapshot(
    *,
    df_events: pd.DataFrame,
    df_forward_windows: pd.DataFrame | None = None,
    symbol: str,
    timeframe: str,
    direction: str,
    setup_confirm: str = "",
    min_completed: int = MIN_SETUP_POCKET_ROWS,
    build_hold_window_intelligence_fn: Callable[..., dict[str, object]] | None = None,
) -> ArchiveDecisionSnapshot:
    """Build the exact archive read for a live Market/Position signal."""

    scope_events, scope_label = select_archive_signal_scope_events(
        df_events,
        symbol=symbol,
        timeframe=timeframe,
        direction=direction,
        setup_confirm=setup_confirm,
        min_rows=min_completed,
    )
    windows = _prepare_windows(df_forward_windows)
    scope_windows = _filter_windows_to_events(windows, scope_events)
    hold_builder = build_hold_window_intelligence_fn or default_build_hold_window_intelligence
    hold_window = hold_builder(scope_events, scope_windows)
    symbol_key = archive_symbol_key(symbol)
    setup_class = archive_setup_class_key(setup_confirm)
    setup_filter_value = setup_class if setup_class in ACTIONABLE_SETUP_CLASSES else "AUTO_BEST"
    setup = build_archive_intelligence_snapshot(
        scope_events,
        setup_filter_value=setup_filter_value,
        min_completed=min_completed,
        symbol=symbol_key or None,
        timeframe=timeframe,
        direction=direction,
        setup_class=setup_class if setup_class in ACTIONABLE_SETUP_CLASSES else None,
    )
    expected_path = build_archive_expected_path_projection(
        df_events=scope_events,
        df_forward_windows=scope_windows,
        symbol_filter=symbol_key,
        timeframe_filter=timeframe,
        min_samples=min_completed,
    )
    confidence_snapshot = ArchiveDecisionSnapshot(
        available=not scope_events.empty or bool(setup.available) or bool(expected_path.get("available")),
        setup=setup,
        metric_events=scope_events,
        direction_events=scope_events,
        metric_windows=scope_windows,
        hold_events=scope_events,
        hold_windows=scope_windows,
        hold_window=hold_window,
        expected_path=expected_path,
        scope_label=scope_label,
    )
    confidence_factor = archive_decision_confidence_factor(confidence_snapshot)
    return ArchiveDecisionSnapshot(
        available=not scope_events.empty or bool(setup.available) or bool(expected_path.get("available")),
        setup=setup,
        metric_events=scope_events,
        direction_events=scope_events,
        metric_windows=scope_windows,
        hold_events=scope_events,
        hold_windows=scope_windows,
        hold_window=hold_window,
        expected_path=expected_path,
        scope_label=scope_label,
        confidence_factor=confidence_factor,
        confidence_tier=archive_confidence_tier(confidence_factor),
    )


def build_archive_decision_snapshot(
    *,
    df_events: pd.DataFrame,
    df_resolved_events: pd.DataFrame | None = None,
    df_forward_windows: pd.DataFrame | None = None,
    symbol_filter: str = "",
    timeframe_filter: str = "All",
    setup_filter_value: str = "AUTO_BEST",
    min_completed: int = MIN_SETUP_POCKET_ROWS,
    timeframe_order: tuple[str, ...] = ("5m", "15m", "1h", "4h", "1d"),
    build_hold_window_intelligence_fn: Callable[..., dict[str, object]] | None = None,
) -> ArchiveDecisionSnapshot:
    """Build the archive read once so UI, Market, and Position can share it."""

    events = _prepare_events(df_events)
    resolved_events = (
        _prepare_events(df_resolved_events)
        if df_resolved_events is not None and not df_resolved_events.empty
        else _resolved_events(events)
    )
    windows = _prepare_windows(df_forward_windows)
    hold_builder = build_hold_window_intelligence_fn or default_build_hold_window_intelligence

    setup_source = resolved_events if not resolved_events.empty else events
    setup = build_archive_intelligence_snapshot(
        setup_source,
        setup_filter_value=setup_filter_value,
        min_completed=min_completed,
        symbol=symbol_filter or None,
        timeframe=None if str(timeframe_filter or "").strip().lower() == "all" else timeframe_filter,
    )

    if bool(setup.available):
        metric_events = _filter_events_to_setup_pocket(events, setup)
        direction_events = _filter_events_to_setup_direction(events, setup)
        hold_events = _filter_events_to_setup_pocket(resolved_events, setup)
    else:
        metric_events = events.copy()
        direction_events = events.copy()
        hold_events = resolved_events.copy()

    if metric_events.empty:
        metric_events = events.copy()
    if direction_events.empty:
        direction_events = metric_events.copy()
    if hold_events.empty and not resolved_events.empty:
        hold_events = resolved_events.copy()

    metric_windows = _filter_windows_to_events(windows, metric_events)
    hold_windows = _filter_windows_to_events(windows, hold_events)
    hold_window = hold_builder(hold_events, hold_windows)
    expected_path = build_archive_expected_path_projection(
        df_events=hold_events,
        df_forward_windows=hold_windows,
        symbol_filter=symbol_filter,
        timeframe_filter=timeframe_filter,
        timeframe_order=timeframe_order,
        min_samples=min_completed,
    )
    confidence_snapshot = ArchiveDecisionSnapshot(
        available=bool(setup.available) or bool(expected_path.get("available")) or not hold_events.empty,
        setup=setup,
        metric_events=metric_events,
        direction_events=direction_events,
        metric_windows=metric_windows,
        hold_events=hold_events,
        hold_windows=hold_windows,
        hold_window=hold_window,
        expected_path=expected_path,
        scope_label="",
    )
    confidence_factor = archive_decision_confidence_factor(confidence_snapshot)

    return ArchiveDecisionSnapshot(
        available=bool(setup.available) or bool(expected_path.get("available")) or not hold_events.empty,
        setup=setup,
        metric_events=metric_events,
        direction_events=direction_events,
        metric_windows=metric_windows,
        hold_events=hold_events,
        hold_windows=hold_windows,
        hold_window=hold_window,
        expected_path=expected_path,
        scope_label="",
        confidence_factor=confidence_factor,
        confidence_tier=archive_confidence_tier(confidence_factor),
    )
