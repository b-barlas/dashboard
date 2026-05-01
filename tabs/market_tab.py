from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from ui.ctx import get_ctx

from concurrent.futures import ThreadPoolExecutor, as_completed
import html
import math
import re
import time
from threading import Lock
from types import SimpleNamespace

import pandas as pd
import streamlit as st
from core.ai_spot_bias import (
    ai_spot_bias_consensus_agreement,
    ai_spot_bias_directional_agreement,
    ai_spot_bias_display_votes,
    ai_spot_bias_probability_up,
    ai_spot_bias_status,
    build_ai_spot_bias_snapshot,
)
from core.archive_decision import (
    apply_archive_confidence_guardrail,
    apply_archive_invalidation_guardrail,
    archive_decision_observability,
    archive_decision_score_adjustment,
    archive_decision_feedback_for_signal,
    build_archive_decision_feedback_map,
    build_archive_decision_feedback_model,
    build_archive_signal_decision_snapshot,
    calibrate_archive_decision_scores,
)
from core.archive_intelligence import archive_policy_for_signal, build_archive_policy_map
from core.archive_policy import ARCHIVE_LEARNING_WINDOW_ROWS
from core.confidence import (
    ai_confidence_bucket,
    build_ai_confidence_snapshot,
    build_confidence_snapshot,
    build_execution_confidence_snapshot,
    confidence_bucket,
)
from core.decision_version import current_decision_version
from core.market_decision import (
    apply_setup_archive_calibration,
    ai_led_confirmation_snapshot,
    ai_vote_metrics,
    action_reason_text,
    emerging_bias_snapshot,
    normalize_action_class,
    selected_timeframe_execution_snapshot,
    selected_timeframe_rr_ratio,
    spot_action_decision_with_reason,
    spot_structure_state,
    trend_led_confirmation_snapshot,
)
from core.scalping import scalp_gate_thresholds, scalp_reason_short_label, scalp_reason_text
from core.scalping import apply_scalp_archive_calibration
from core.signal_contract import bias_confidence_from_bias
from core.spot_direction import build_spot_direction_snapshot
from core.timeframe_anchors import choose_anchor_context
from core.metric_catalog import (
    AI_LONG_THRESHOLD,
    AI_SHORT_THRESHOLD,
    direction_from_prob,
)
from core.adaptive_weighting import (
    build_adaptive_context_model,
    build_actionable_ranking_model,
    build_actionable_ranking_snapshot,
    build_ai_confidence_calibration_model,
    build_ai_confidence_calibration_snapshot,
    build_archive_guardrail_snapshot,
    build_confidence_calibration_model,
    build_confidence_calibration_snapshot,
    build_risk_sizing_calibration_model,
    build_risk_sizing_calibration_snapshot,
    build_scalp_calibration_model,
    build_scalp_calibration_snapshot,
    build_setup_calibration_model,
    build_setup_calibration_snapshot,
    build_session_fit_snapshot,
    build_trade_gate_calibration_model,
    build_trade_gate_calibration_snapshot,
)
from core.catalyst_engine import catalyst_signal_note, catalyst_window_label
from core.no_trade_engine import apply_market_trade_gate_archive_calibration
from core.session_utils import session_bucket_for_timestamp
from core.signal_tracker import build_alert_effectiveness_summary, prefer_current_decision_version_slice
from core.symbols import canonical_base_symbol, is_stable_base_symbol
from core.trading_copy import copy_text
from tabs.market_scan_helpers import (
    SCAN_MODE_ACTIONABLE as _SCAN_MODE_ACTIONABLE,
    SCAN_MODE_BROAD as _SCAN_MODE_BROAD,
    SCAN_MODE_EMERGING as _SCAN_MODE_EMERGING,
    SCAN_MODE_TRENDING as _SCAN_MODE_TRENDING,
    _actionable_analysis_batch_size,
    _actionable_context_score,
    _actionable_direction_include,
    _actionable_frame_hunt_score,
    _actionable_setup_score,
    _actionable_tactical_candidate_score,
    _apply_breakout_archive_feedback_to_market_rows,
    _apply_breakout_memory_to_market_rows,
    _apply_scanner_trace_feedback_to_market_rows,
    _build_breakout_archive_feedback_map,
    _build_scanner_trace_feedback_map,
    _emerging_candidate_score,
    _candidate_scan_symbols as _candidate_scan_symbols_impl,
    _initial_scan_symbols,
    _next_refill_candidate_batch,
    _next_scan_pool_target,
    _normalize_scan_mode,
    _scan_candidate_pool_size,
)
from tabs.whale_tab import _compute_scan_thresholds, _run_volume_anomaly_scan
from ui.primitives import render_help_details, render_kpi_grid, render_page_header
from ui.signal_formatters import (
    setup_confirm_display as _shared_setup_confirm_display,
    spot_bias_label as _shared_spot_bias_label,
    trade_gate_display_label,
)

_LAST_GOOD_RESULTS_KEY = "market_scan_last_good_results"
_LAST_GOOD_SIG_KEY = "market_scan_last_good_sig"
_LAST_GOOD_TS_KEY = "market_scan_last_good_ts"
_LAST_GOOD_MODE_KEY = "market_scan_last_good_mode"
_LAST_GOOD_REGISTRY_KEY = "market_scan_last_good_by_sig"
_LAST_HEALTHY_EMPTY_SIG_KEY = "market_scan_last_healthy_empty_sig"
_LAST_SCAN_ATTEMPT_TS_KEY = "market_scan_last_attempt_ts"
_DATA_HEALTH_ITEMS_KEY = "market_data_health_items"
_TRENDING_VOLUME_CACHE_KEY = "market_trending_volume_anomaly_cache"
_HEALTHY_EMPTY_HISTORY_LIMIT = 32
_LAST_GOOD_HISTORY_LIMIT = 32
_TRENDING_VOLUME_CACHE_LIMIT = 8
_TRENDING_VOLUME_CACHE_TTL_SECONDS = 300
_SCAN_REFRESH_TTL_MINUTES = 3
_RECOVERY_RETRY_BACKOFF_SECONDS = 30
_AUTO_TIMEFRAME_LEARNING_STATE_KEY = "market_auto_timeframe_learning_state"
_AUTO_TIMEFRAME_LEARNING_TIMEFRAMES = ("5m", "15m", "1h", "4h", "1d")
_AUTO_TIMEFRAME_LEARNING_SCAN_FOCUS = "Auto Timeframe Sweep"
_AUTO_TIMEFRAME_LEARNING_GLOBAL_COOLDOWN_SECONDS = 45
_AUTO_TIMEFRAME_LEARNING_FETCH_N = 80
_AUTO_TIMEFRAME_LEARNING_MAX_TIMEFRAMES_PER_PASS = 2
_AUTO_TIMEFRAME_LEARNING_MIN_INTERVAL_SECONDS = {
    "5m": 8 * 60,
    "15m": 15 * 60,
    "1h": 30 * 60,
    "4h": 2 * 60 * 60,
    "1d": 6 * 60 * 60,
}
_AUTO_TIMEFRAME_LEARNING_SYMBOL_LIMITS = {
    "5m": 6,
    "15m": 8,
    "1h": 8,
    "4h": 8,
    "1d": 6,
}
_AUTO_TIMEFRAME_LEARNING_OPEN_SYMBOL_LIMITS = {
    "5m": 8,
    "15m": 10,
    "1h": 8,
    "4h": 8,
    "1d": 6,
}
_AUTO_TIMEFRAME_LEARNING_CANDLE_LIMITS = {
    "5m": 360,
    "15m": 320,
    "1h": 300,
    "4h": 220,
    "1d": 180,
}
_AUTO_TIMEFRAME_LEARNING_BACKFILL_PAIR_LIMITS = {
    "5m": 2,
    "15m": 3,
    "1h": 3,
    "4h": 2,
    "1d": 2,
}
_AUTO_TIMEFRAME_LEARNING_USABLE_TARGETS = {
    "5m": 24,
    "15m": 32,
    "1h": 48,
    "4h": 24,
    "1d": 16,
}
_SETUP_CONFIRM_PRIORITY = {
    "ENTER_TREND_AI": 6,
    "ENTER_TREND_LED": 5,
    "ENTER_AI_LED": 4,
    "PROBE": 3,
    "WATCH": 2,
    "SKIP": 1,
}
_AI_FALLBACK_STATUS_TEXT = {
    "insufficient_candles": "AI safety read: not enough candles for reliable ML; neutral output is shown.",
    "insufficient_features": "AI safety read: indicators produced too few clean ML rows; neutral output is shown.",
    "single_class_window": "AI safety read: one-sided training window forced neutral output.",
    "model_exception": "AI safety read: model instability forced neutral output.",
}
_ALERT_ARCHIVE_DISPLAY = {
    "CATALYST_BLOCK": "Catalyst Block",
    "TRADE_GATE": "Market Stance",
    "MARKET_LEAD": "Market Lead",
    "LEARNED_EDGE": "Learned Edge",
    "ACTIONABLE_CLUSTER": "Actionable Cluster",
    "ARCHIVE_GUARDRAIL": "History Caution",
    "EXECUTION_STANCE": "Market Stance",
    "PLAYBOOK_WINDOW": "Playbook Window",
    "SECTOR_ROTATION": "Sector Rotation",
    "SESSION_FIT": "Session Fit",
    "FLOW_PROXY": "Flow Proxy",
    "CATALYST_CAUTION": "Catalyst Caution",
}
_PROTECTED_ALERT_KEYS = {"CATALYST_BLOCK", "TRADE_GATE", "CATALYST_CAUTION", "ARCHIVE_GUARDRAIL"}
_CLEAREST_TREND_COLS = ("ADX", "SuperTrend", "Ichimoku", "VWAP", "PSAR")
_CLEAREST_MOMENTUM_COLS = ("Stochastic RSI", "Williams %R", "CCI", "Candle Pattern")
_CLEAREST_ACTIVITY_COLS = ("Bollinger", "Volatility", "Spike Alert")


def _canonical_pair_base(symbol: str) -> str:
    raw = str(symbol or "").strip()
    base = raw.split("/", 1)[0] if "/" in raw else raw
    return canonical_base_symbol(base)


def _setup_confirm_class_key(value: str) -> str:
    s = str(value or "").strip().upper()
    if "TREND+AI" in s or "T+AI" in s:
        return "ENTER_TREND_AI"
    if s == "TREND" or s == "TREND-LED" or "TREND-LED" in s or s.endswith(" TREND"):
        return "ENTER_TREND_LED"
    if s == "AI" or s == "AI-LED" or "AI-LED" in s or s.endswith(" AI"):
        return "ENTER_AI_LED"
    return normalize_action_class(s)


def _setup_confirm_priority(value: str) -> int:
    return int(_SETUP_CONFIRM_PRIORITY.get(_setup_confirm_class_key(value), 0))


def _sortable_float(value: object) -> float:
    try:
        return float(value or 0.0)
    except Exception:
        return 0.0


def _execution_friction_score(
    *,
    mcap_val: float | int | None,
    volatility_label: str | None,
    delta_pct: float | None,
    spike_present: bool,
    execution_confidence: float | None = None,
) -> float:
    score = 58.0
    mcap = float(max(0.0, float(mcap_val or 0.0)))
    if mcap >= 200_000_000_000:
        score += 16.0
    elif mcap >= 50_000_000_000:
        score += 12.0
    elif mcap >= 10_000_000_000:
        score += 7.0
    elif mcap >= 2_000_000_000:
        score += 2.0
    elif mcap >= 500_000_000:
        score -= 8.0
    elif mcap > 0.0:
        score -= 14.0

    vol_key = str(volatility_label or "").strip().upper()
    if "LOW" in vol_key:
        score += 6.0
    elif "MODERATE" in vol_key:
        score += 1.0
    elif "HIGH" in vol_key:
        score -= 8.0
    elif "EXTREME" in vol_key:
        score -= 12.0

    delta_abs = abs(float(delta_pct or 0.0))
    if delta_abs >= 5.0:
        score -= 10.0
    elif delta_abs >= 3.0:
        score -= 6.0
    elif delta_abs >= 1.5:
        score -= 3.0

    if spike_present:
        score -= 6.0

    exec_conf = _sortable_float(execution_confidence)
    if exec_conf >= 82.0:
        score += 2.0
    elif 0.0 < exec_conf <= 45.0:
        score -= 2.0

    return max(0.0, min(100.0, score))


def _expectancy_bias_score(
    *,
    archive_delta: float | None,
    bucket_resolved: float | None,
    matched_factors: float | int | None,
) -> float:
    delta = max(-8.0, min(8.0, _sortable_float(archive_delta)))
    if abs(delta) < 0.01:
        return 50.0
    bucket_strength = min(1.0, _sortable_float(bucket_resolved) / 36.0)
    factor_strength = min(1.0, _sortable_float(matched_factors) / 4.0)
    strength = 0.75 * bucket_strength + 0.25 * factor_strength
    multiplier = 1.20 + 1.40 * strength
    score = 50.0 + delta * multiplier
    return max(30.0, min(70.0, score))


def _coverage_adjusted_archive_scores(
    *,
    base_archive_score: float | None,
    base_expectancy_score: float | None,
    policy_delta: float | None,
    policy_coverage: float | None,
) -> tuple[float, float]:
    coverage = max(0.0, min(1.0, _sortable_float(policy_coverage)))
    base_weight = 0.35 + (0.65 * coverage)
    archive_score = (_sortable_float(base_archive_score) * base_weight) + _sortable_float(policy_delta)
    expectancy_score = max(
        30.0,
        min(
            70.0,
            50.0
            + ((_sortable_float(base_expectancy_score) - 50.0) * base_weight)
            + (_sortable_float(policy_delta) * 1.5),
        ),
    )
    return archive_score, expectancy_score


@st.cache_data(ttl=60, show_spinner=False)
def _market_archive_bundle(
    *,
    _fetch_signal_events_df,
    _fetch_signal_forward_windows_df,
    db_path: str,
    decision_version_target: str,
) -> dict[str, object]:
    adaptive_history_raw_df = _fetch_signal_events_df(
        limit=ARCHIVE_LEARNING_WINDOW_ROWS,
        status="RESOLVED",
        source="Market",
        db_path=db_path,
    )
    adaptive_history_df = prefer_current_decision_version_slice(
        adaptive_history_raw_df,
        source="Market",
    )
    adaptive_decision_rows = int(adaptive_history_df.attrs.get("decision_version_rows") or 0)
    adaptive_decision_total_rows = int(
        adaptive_history_df.attrs.get("decision_version_total_rows") or len(adaptive_history_df)
    )
    adaptive_version_series = (
        adaptive_history_raw_df.get("decision_version", pd.Series(index=adaptive_history_raw_df.index, dtype=object))
        .fillna("")
        .astype(str)
        .str.strip()
    )
    adaptive_current_mask = adaptive_version_series.eq(decision_version_target)
    adaptive_signal_keys = (
        adaptive_history_df["signal_key"].fillna("").astype(str).str.strip().drop_duplicates().tolist()
        if isinstance(adaptive_history_df, pd.DataFrame)
        and not adaptive_history_df.empty
        and "signal_key" in adaptive_history_df.columns
        else []
    )
    adaptive_forward_windows_df = (
        _fetch_signal_forward_windows_df(
            signal_keys=adaptive_signal_keys,
            db_path=db_path,
        )
        if callable(_fetch_signal_forward_windows_df) and adaptive_signal_keys
        else pd.DataFrame()
    )
    adaptive_has_plan = pd.to_numeric(
        adaptive_history_raw_df.get("has_plan", pd.Series(index=adaptive_history_raw_df.index, dtype=float)),
        errors="coerce",
    ).fillna(0).astype(int)
    adaptive_plan_outcome = (
        adaptive_history_raw_df.get("plan_outcome", pd.Series(index=adaptive_history_raw_df.index, dtype=object))
        .fillna("")
        .astype(str)
        .str.strip()
        .str.upper()
    )
    adaptive_current_scalp_planned_rows = int((_adaptive_current_mask := adaptive_current_mask & adaptive_has_plan.eq(1)).sum())
    adaptive_current_scalp_resolved_rows = int((_adaptive_current_mask & adaptive_plan_outcome.ne("")).sum())

    return {
        "raw_df": adaptive_history_raw_df,
        "df": adaptive_history_df,
        "forward_windows_df": adaptive_forward_windows_df,
        "decision_mode": str(adaptive_history_df.attrs.get("decision_version_mode") or "mixed_fallback"),
        "decision_target": str(adaptive_history_df.attrs.get("decision_version_target") or decision_version_target),
        "decision_rows": adaptive_decision_rows,
        "decision_total_rows": adaptive_decision_total_rows,
        "current_scalp_planned_rows": adaptive_current_scalp_planned_rows,
        "current_scalp_resolved_rows": adaptive_current_scalp_resolved_rows,
        "adaptive_model": build_adaptive_context_model(adaptive_history_df),
        "ai_confidence_calibration_model": build_ai_confidence_calibration_model(adaptive_history_df),
        "confidence_calibration_model": build_confidence_calibration_model(adaptive_history_df),
        "setup_calibration_model": build_setup_calibration_model(adaptive_history_df),
        "actionable_ranking_model": build_actionable_ranking_model(adaptive_history_df),
        "risk_sizing_calibration_model": build_risk_sizing_calibration_model(adaptive_history_df),
        "trade_gate_calibration_model": build_trade_gate_calibration_model(adaptive_history_df),
        "scalp_calibration_model": build_scalp_calibration_model(adaptive_history_df),
        "archive_policy_map": build_archive_policy_map(adaptive_history_df),
        "archive_decision_feedback_model": build_archive_decision_feedback_model(adaptive_history_df),
        "archive_decision_feedback_map": build_archive_decision_feedback_map(adaptive_history_df),
    }


def _market_result_priority_key(row: dict) -> tuple[float, float, float, float, float, float, float, float, float, float, str]:
    return (
        -float(_setup_confirm_priority(str(row.get("__action_raw", row.get("Setup Confirm", ""))))),
        -_sortable_float(row.get("__risk_unit_fraction", 0.0)),
        -_sortable_float(row.get("__execution_friction_score", 50.0)),
        -_sortable_float(row.get("__expectancy_bias_score", 50.0)),
        -_sortable_float(row.get("__confidence_val", 0.0)),
        -_sortable_float(row.get("__adaptive_edge_score", 50.0)),
        -_sortable_float(row.get("__actionable_archive_score", 0.0)),
        _sortable_float(row.get("__archive_guardrail_penalty", 0.0)),
        -_sortable_float(row.get("__ai_confidence_val", 0.0)),
        -_sortable_float(row.get("__mcap_val", 0)),
        str(row.get("Coin", "")),
    )


def _actionable_market_result_priority_key(
    row: dict,
) -> tuple[float, float, float, float, float, float, float, float, float, float, float, float, str]:
    return (
        -float(_setup_confirm_priority(str(row.get("__action_raw", row.get("Setup Confirm", ""))))),
        -_sortable_float(row.get("__actionable_context_score", 0.0)),
        -_sortable_float(row.get("__actionable_tactical_score", 0.0)),
        -_sortable_float(row.get("__actionable_setup_score", 0.0)),
        -_sortable_float(row.get("__expectancy_bias_score", 50.0)),
        -_sortable_float(row.get("__execution_friction_score", 50.0)),
        -_sortable_float(row.get("__actionable_archive_score", 0.0)),
        -_sortable_float(row.get("__risk_unit_fraction", 0.0)),
        -_sortable_float(row.get("__confidence_val", 0.0)),
        -_sortable_float(row.get("__adaptive_edge_score", 50.0)),
        _sortable_float(row.get("__archive_guardrail_penalty", 0.0)),
        -_sortable_float(row.get("__ai_confidence_val", 0.0)),
        str(row.get("Coin", "")),
    )


def _emerging_market_result_priority_key(
    row: dict,
) -> tuple[float, float, float, float, float, float, float, float, float, float, str]:
    lead_active = 1.0 if _signal_tracker_direction_key(row.get("__emerging_direction")) in {"UPSIDE", "DOWNSIDE"} else 0.0
    return (
        -float(_setup_confirm_priority(str(row.get("__action_raw", row.get("Setup Confirm", ""))))),
        -_sortable_float(row.get("__emerging_rank_score", 0.0)),
        -lead_active,
        -_sortable_float(row.get("__actionable_frame_score", 0.0)),
        -_sortable_float(row.get("__actionable_tactical_score", 0.0)),
        -_sortable_float(row.get("__confidence_val", 0.0)),
        -_sortable_float(row.get("__execution_friction_score", 50.0)),
        -_sortable_float(row.get("__expectancy_bias_score", 50.0)),
        -_sortable_float(row.get("__ai_confidence_val", 0.0)),
        _sortable_float(row.get("__archive_guardrail_penalty", 0.0)),
        str(row.get("Coin", "")),
    )


def _market_result_priority_key_for_mode(row: dict, scan_mode: str) -> tuple:
    normalized_mode = _normalize_scan_mode(scan_mode)
    if normalized_mode == _SCAN_MODE_ACTIONABLE:
        return _actionable_market_result_priority_key(row)
    if normalized_mode in {_SCAN_MODE_EMERGING, _SCAN_MODE_TRENDING}:
        return _emerging_market_result_priority_key(row)
    return _market_result_priority_key(row)


_MARKET_RENDER_META_COLS = [
    "__action_reason",
    "__action_raw",
    "__adaptive_edge_note",
    "__ai_confidence_note",
    "__ai_note",
    "__catalyst_fit_note",
    "__confidence_note",
    "__delta_note",
    "__direction_note",
    "__emerging_direction",
    "__emerging_label",
    "__emerging_note",
    "__entry_note",
    "__execution_fit_note",
    "__ichimoku_detail",
    "__pair",
    "__rr_note",
    "__scalp_display_state",
    "__scalp_reason_short",
    "__scalp_reason_text",
    "__session_fit_note",
    "__setup_calibration_note",
    "__spike_candle_pct",
    "__spike_dir",
    "__spike_vol_ratio",
    "__spike_vwap_ctx",
    "__target_note",
    "__adx_raw",
]


def _market_hidden_meta_cols(df_columns, display_cols) -> list[str]:
    available = set(df_columns)
    display_set = set(display_cols)
    return [col for col in _MARKET_RENDER_META_COLS if col in available and col not in display_set]


def _decision_version_mode_label(mode: str) -> str:
    key = str(mode or "").strip().lower()
    if key == "current_only":
        return "Current Learning"
    if key == "mixed_fallback":
        return "Broader Archive Backup"
    if key == "unversioned_fallback":
        return "Legacy Archive Backup"
    if key == "empty":
        return "Empty"
    return "Mixed Archive"


def _market_calibration_diagnostic_lines(
    *,
    mode: str,
    target_version: str,
    current_rows: int,
    total_rows: int,
    scalp_planned_rows: int,
    scalp_resolved_rows: int,
    min_current_rows: int = 80,
) -> list[str]:
    target_label = str(target_version or "Current Learning").strip() or "Current Learning"
    lines = [
        f"- Calibration mode: **{_decision_version_mode_label(mode)}**",
        f"- Current learning version: `{target_label}`",
    ]
    if mode == "current_only":
        lines.append(
            f"- Current learning archive is active: **{int(current_rows)}** resolved rows are driving adaptive calibration directly "
            f"(recent resolved pool loaded: **{int(total_rows)}**)."
        )
    elif mode == "mixed_fallback":
        lines.append(
            f"- Current learning archive is still building: **{int(current_rows)}** of **{int(total_rows)}** recent resolved rows "
            f"match the new market logic, so adaptive calibration still leans on broader archive history until it reaches **{int(min_current_rows)}**."
        )
    elif mode == "unversioned_fallback":
        lines.append(
            f"- The loaded resolved pool has **{int(total_rows)}** rows, but they do not isolate the current learning version yet, "
            "so adaptive calibration is still reading older archive history."
        )
    elif mode == "empty":
        lines.append("- No resolved archive rows are available yet, so adaptive calibration has nothing to learn from.")
    else:
        lines.append(
            f"- Resolved learned rows: **{int(current_rows)}** of **{int(total_rows)}** loaded. Adaptive calibration is reading a broader archive slice."
        )

    if int(scalp_planned_rows) > 0 or int(scalp_resolved_rows) > 0:
        lines.append(
            f"- Scalp learning sample for the current system: **{int(scalp_planned_rows)}** planned rows, "
            f"**{int(scalp_resolved_rows)}** resolved outcomes."
        )
    else:
        lines.append(
            "- Scalp learning sample for the current system is still empty, so scalp archive calibration is in build-up mode."
        )
    return lines


def _emerging_badge_tone(direction: str) -> str:
    key = str(direction or "").strip().upper()
    if key == "UPSIDE":
        return "pos"
    if key == "DOWNSIDE":
        return "neg"
    return "muted"


def _emerging_badge_symbol(direction: str) -> str:
    key = str(direction or "").strip().upper()
    if key == "UPSIDE":
        return "↗"
    if key == "DOWNSIDE":
        return "↘"
    return "•"


def _emerging_badge_text(direction: str) -> str:
    key = str(direction or "").strip().upper()
    if key in {"UPSIDE", "DOWNSIDE"}:
        return "PUSH"
    return "INFO"


@dataclass(frozen=True)
class MarketLeadSnapshot:
    score: float
    state: str
    label: str
    note: str
    breadth_component: float
    rotation_component: float
    flow_component: float
    dominance_component: float
    upside_leads: int
    downside_leads: int


def _signal_tracker_direction_key(value: object) -> str:
    d = str(value or "").strip().upper()
    if d in {"UPSIDE", "LONG", "BUY", "BULLISH"}:
        return "UPSIDE"
    if d in {"DOWNSIDE", "SHORT", "SELL", "BEARISH"}:
        return "DOWNSIDE"
    return "NEUTRAL"


def _confidence_value_from_badge(text: object) -> float | None:
    match = re.search(r"(-?\d+(?:\.\d+)?)\s*%", str(text or ""))
    if not match:
        return None
    try:
        return float(match.group(1))
    except Exception:
        return None


def _adaptive_execution_summary(model: dict[str, object]) -> str:
    actual_closed = int(model.get("overall_actual_closed") or 0)
    if actual_closed < 8:
        return "Execution archive is still building."
    win_pct = float(model.get("overall_actual_win_pct") or 0.0)
    avg_pnl = float(model.get("overall_avg_actual_pnl") or 0.0)
    if win_pct >= 58.0 and avg_pnl >= 0.5:
        tone = "Execution archive is strong"
    elif win_pct <= 42.0 or avg_pnl <= -0.5:
        tone = "Execution archive is weak"
    else:
        tone = "Execution archive is mixed"
    return f"{tone}: {win_pct:.0f}% win rate across {actual_closed} closed trades, avg {avg_pnl:+.2f}%."


def _adaptive_execution_brief(model: dict[str, object]) -> str:
    actual_closed = int(model.get("overall_actual_closed") or 0)
    if actual_closed < 8:
        return ""
    win_pct = float(model.get("overall_actual_win_pct") or 0.0)
    avg_pnl = float(model.get("overall_avg_actual_pnl") or 0.0)
    if win_pct >= 58.0 and avg_pnl >= 0.5:
        return "Execution archive: strong."
    if win_pct <= 42.0 or avg_pnl <= -0.5:
        return "Execution archive: weak."
    return ""


def _session_archive_summary(session_fit_snapshot) -> str:
    label = str(getattr(session_fit_snapshot, "label", "") or "").strip()
    note = str(getattr(session_fit_snapshot, "note", "") or "").strip()
    if not label:
        return ""
    return f"Session archive: {label}. {note}".strip()


def _compact_trade_gate_note(
    *,
    market_trade_gate_snapshot,
    market_catalyst_snapshot,
    market_flow_snapshot,
    market_default_budget_snapshot,
    session_fit_snapshot,
    adaptive_model: dict[str, object],
    enter_count: int,
    probe_count: int,
) -> str:
    gate_key = str(getattr(market_trade_gate_snapshot, "gate_key", "") or "").strip().upper()
    catalyst_label = str(getattr(market_catalyst_snapshot, "label", "") or "").strip()
    flow_label = str(getattr(market_flow_snapshot, "label", "") or "").strip()
    session_label = str(getattr(session_fit_snapshot, "label", "") or "").strip()
    size_label = str(getattr(market_default_budget_snapshot, "label", "") or "").strip()

    parts: list[str] = []
    if gate_key == "TRADEABLE":
        parts.append(copy_text("market.trade_gate.summary.tradeable"))
    elif gate_key == "SELECTIVE_ONLY":
        if int(enter_count) <= 0 and int(probe_count) > 0:
            parts.append(copy_text("market.trade_gate.summary.selective_probe"))
        else:
            parts.append(copy_text("market.trade_gate.summary.selective_clean"))
    elif gate_key == "DEFENSIVE_ONLY":
        parts.append(copy_text("market.trade_gate.summary.defensive"))
    else:
        parts.append(copy_text("market.trade_gate.summary.stand_aside"))

    if size_label:
        parts.append(copy_text("market.trade_gate.summary.size_cap", size_label=size_label))
    if catalyst_label not in {"", "No Near Catalyst", "Catalyst Clear"}:
        parts.append(copy_text("market.trade_gate.summary.catalyst", catalyst_label=catalyst_label))
    if flow_label not in {"", "Flow Balanced"}:
        parts.append(copy_text("market.trade_gate.summary.flow", flow_label=flow_label))
    if session_label in {"Session Supportive", "Session Fragile"}:
        parts.append(
            copy_text(
                "market.trade_gate.summary.session",
                session_label=session_label.replace("Session ", ""),
            )
        )

    archive_brief = _adaptive_execution_brief(adaptive_model)
    if archive_brief:
        parts.append(archive_brief)
    return " ".join(parts).strip()


def _market_signal_log_events(
    *,
    rows: list[dict],
    timeframe: str,
    scan_mode: str,
    market_lead_snapshot: MarketLeadSnapshot,
    market_regime_snapshot,
    market_trade_gate_snapshot,
    build_signal_risk_sizing,
    sector_rotation_snapshot,
    classify_symbol_sector,
    market_catalyst_snapshot,
    market_flow_snapshot,
    session_fit_snapshot,
    market_alerts,
) -> list[dict[str, object]]:
    events: list[dict[str, object]] = []
    decision_version = current_decision_version("Market")
    alert_keys = [
        str(getattr(alert, "alert_key", "") or "").strip().upper()
        for alert in list(market_alerts or [])
        if str(getattr(alert, "alert_key", "") or "").strip()
    ]
    alert_keys_text = "|".join(alert_keys)
    primary_alert = alert_keys[0] if alert_keys else ""
    for row in rows:
        event_time = row.get("__event_time")
        symbol = str(row.get("Coin") or "").strip().upper()
        if not symbol or event_time is None:
            continue
        ai_ensemble = str(row.get("AI Ensemble") or "").strip()
        sector_tag = classify_symbol_sector(symbol)
        direction_raw = str(row.get("Direction") or "")
        confidence_val = row.get("__confidence_val", _confidence_value_from_badge(row.get("Confidence")))
        ai_conf_val = row.get("__ai_confidence_val", _confidence_value_from_badge(row.get("AI Confidence")))
        lead_direction = str(row.get("__emerging_direction") or "")
        market_lead_aligned = (
            _signal_tracker_direction_key(direction_raw) in {"UPSIDE", "DOWNSIDE"}
            and _signal_tracker_direction_key(direction_raw) == _signal_tracker_direction_key(market_lead_snapshot.label)
        )
        risk_sizing_snapshot = build_signal_risk_sizing(
            market_trade_gate_snapshot=market_trade_gate_snapshot,
            market_catalyst_snapshot=market_catalyst_snapshot,
            direction=direction_raw,
            setup_confirm=str(row.get("__action_raw", row.get("Setup Confirm", "")) or ""),
            confidence=confidence_val,
            ai_confidence=ai_conf_val,
            ai_aligned=(
                _signal_tracker_direction_key(direction_raw)
                == _signal_tracker_direction_key(ai_ensemble.split("(", 1)[0].strip())
            ),
            market_lead_aligned=market_lead_aligned,
            lead_active=_signal_tracker_direction_key(lead_direction) in {"UPSIDE", "DOWNSIDE"},
            rr_ratio=row.get("__rr_val"),
            adaptive_edge_score=row.get("__adaptive_edge_score"),
            session_fit_score=float(getattr(session_fit_snapshot, "score", 0.0) or 0.0),
            archive_guardrail_penalty=row.get("__archive_guardrail_penalty"),
            archive_guardrail_label=row.get("__archive_guardrail_label"),
            archive_guardrail_note=row.get("__archive_guardrail_note"),
            archive_risk_delta=row.get("__risk_archive_delta"),
            archive_risk_note=row.get("__risk_archive_note"),
            symbol=symbol,
            sector_tag=str(sector_tag or ""),
        )
        risk_label = str(row.get("__risk_tier_label") or getattr(risk_sizing_snapshot, "label", "") or "")
        risk_fraction = row.get("__risk_unit_fraction")
        risk_fraction_value = (
            _sortable_float(risk_fraction)
            if risk_fraction is not None
            else float(getattr(risk_sizing_snapshot, "unit_fraction", 0.0) or 0.0)
        )
        events.append(
            {
                "source": "Market",
                "decision_version": decision_version,
                "symbol": symbol,
                "timeframe": str(row.get("__timeframe") or timeframe),
                "event_time": event_time,
                "session_bucket": session_bucket_for_timestamp(event_time),
                "scan_focus": _normalize_scan_mode(scan_mode),
                "direction": direction_raw,
                "setup_confirm": str(row.get("__action_raw", row.get("Setup Confirm", "")) or ""),
                "action_reason": str(row.get("__action_reason") or ""),
                "lead_label": str(row.get("__emerging_label") or ""),
                "lead_direction": lead_direction,
                "confidence": confidence_val,
                "ai_ensemble": ai_ensemble,
                "ai_direction": ai_ensemble.split("(", 1)[0].strip(),
                "ai_confidence": ai_conf_val,
                "market_lead_label": market_lead_snapshot.label,
                "market_lead_score": market_lead_snapshot.score,
                "market_lead_upside": market_lead_snapshot.upside_leads,
                "market_lead_downside": market_lead_snapshot.downside_leads,
                "market_regime": str(getattr(market_regime_snapshot, "label", "") or ""),
                "market_playbook_key": str(getattr(market_regime_snapshot, "playbook_key", "") or ""),
                "market_playbook": str(getattr(market_regime_snapshot, "playbook", "") or ""),
                "market_no_trade": bool(getattr(market_trade_gate_snapshot, "no_trade", False)),
                "market_trade_gate_key": str(getattr(market_trade_gate_snapshot, "gate_key", "") or ""),
                "market_trade_gate": str(getattr(market_trade_gate_snapshot, "label", "") or ""),
                "market_alert_keys": alert_keys_text,
                "market_primary_alert": primary_alert,
                "market_no_trade_reason": str(getattr(market_trade_gate_snapshot, "reason_code", "") or ""),
                "risk_tier": risk_label,
                "risk_unit_fraction": risk_fraction_value,
                "sector_tag": str(sector_tag or "").strip(),
                "market_sector_rotation": str(getattr(sector_rotation_snapshot, "label", "") or ""),
                "market_catalyst_state": str(getattr(market_catalyst_snapshot, "label", "") or ""),
                "market_catalyst_event": str(getattr(market_catalyst_snapshot, "next_event", "") or ""),
                "market_catalyst_blocking": bool(getattr(market_catalyst_snapshot, "blocking", False)),
                "market_catalyst_category": str(getattr(market_catalyst_snapshot, "category", "") or ""),
                "market_catalyst_scope": str(getattr(market_catalyst_snapshot, "scope", "") or ""),
                "market_catalyst_tag": str(getattr(market_catalyst_snapshot, "tag", "") or ""),
                "market_catalyst_targeted": bool(getattr(market_catalyst_snapshot, "targeted_only", False)),
                "market_catalyst_window": catalyst_window_label(market_catalyst_snapshot),
                "market_flow_state": str(getattr(market_flow_snapshot, "label", "") or ""),
                "market_flow_bias": str(getattr(market_flow_snapshot, "state", "") or ""),
                "adaptive_edge_label": str(row.get("__adaptive_edge_label") or ""),
                "adaptive_edge_score": row.get("__adaptive_edge_score"),
                "actionable_frame_score": row.get("__actionable_frame_score"),
                "actionable_setup_score": row.get("__actionable_setup_score"),
                "actionable_context_score": row.get("__actionable_context_score"),
                "actionable_tactical_score": row.get("__actionable_tactical_score"),
                "archive_guardrail_label": str(row.get("__archive_guardrail_label") or ""),
                "archive_guardrail_penalty": row.get("__archive_guardrail_penalty"),
                "archive_guardrail_note": str(row.get("__archive_guardrail_note") or ""),
                "archive_policy_delta": row.get("__archive_policy_delta"),
                "archive_policy_completed": row.get("__archive_policy_completed"),
                "archive_policy_quality": str(row.get("__archive_policy_quality") or ""),
                "archive_policy_coverage": row.get("__archive_policy_coverage"),
                "archive_decision_delta": row.get("__archive_decision_delta"),
                "archive_expectancy_delta": row.get("__archive_expectancy_delta"),
                "archive_total_delta": row.get("__archive_total_delta"),
                "archive_total_expectancy_delta": row.get("__archive_total_expectancy_delta"),
                "archive_decision_scope": str(row.get("__archive_decision_scope") or ""),
                "price": row.get("__price_val"),
                "delta_pct": row.get("__delta_pct"),
                "entry_price": row.get("__entry_val"),
                "stop_loss": row.get("__stop_val"),
                "target_price": row.get("__target_val"),
                "rr_ratio": row.get("__rr_val"),
            }
        )
    return events


def _scalp_signal_log_events(
    *,
    rows: list[dict],
    timeframe: str,
    scan_mode: str,
    market_lead_snapshot: MarketLeadSnapshot,
    market_regime_snapshot,
    market_trade_gate_snapshot,
    build_signal_risk_sizing,
    sector_rotation_snapshot,
    classify_symbol_sector,
    market_catalyst_snapshot,
    market_flow_snapshot,
    session_fit_snapshot,
    market_alerts,
) -> list[dict[str, object]]:
    events: list[dict[str, object]] = []
    decision_version = current_decision_version("Market")
    alert_keys = [
        str(getattr(alert, "alert_key", "") or "").strip().upper()
        for alert in list(market_alerts or [])
        if str(getattr(alert, "alert_key", "") or "").strip()
    ]
    alert_keys_text = "|".join(alert_keys)
    primary_alert = alert_keys[0] if alert_keys else ""
    for row in rows:
        scalp_direction = str(row.get("__scalp_direction_raw") or "").strip()
        scalp_state = str(row.get("__scalp_display_state") or "").strip().upper()
        if _signal_tracker_direction_key(scalp_direction) not in {"UPSIDE", "DOWNSIDE"}:
            continue
        if scalp_state not in {"LIVE", "CONDITIONAL"}:
            continue
        event_time = row.get("__event_time")
        symbol = str(row.get("Coin") or "").strip().upper()
        if not symbol or event_time is None:
            continue
        ai_ensemble = str(row.get("AI Ensemble") or "").strip()
        sector_tag = classify_symbol_sector(symbol)
        direction_raw = scalp_direction
        confidence_val = row.get("__confidence_val", _confidence_value_from_badge(row.get("Confidence")))
        ai_conf_val = row.get("__ai_confidence_val", _confidence_value_from_badge(row.get("AI Confidence")))
        lead_direction = str(row.get("__emerging_direction") or "")
        market_lead_aligned = (
            _signal_tracker_direction_key(direction_raw) in {"UPSIDE", "DOWNSIDE"}
            and _signal_tracker_direction_key(direction_raw) == _signal_tracker_direction_key(market_lead_snapshot.label)
        )
        risk_sizing_snapshot = build_signal_risk_sizing(
            market_trade_gate_snapshot=market_trade_gate_snapshot,
            market_catalyst_snapshot=market_catalyst_snapshot,
            direction=direction_raw,
            setup_confirm=str(row.get("__action_raw", row.get("Setup Confirm", "")) or ""),
            confidence=confidence_val,
            ai_confidence=ai_conf_val,
            ai_aligned=(
                _signal_tracker_direction_key(direction_raw)
                == _signal_tracker_direction_key(ai_ensemble.split("(", 1)[0].strip())
            ),
            market_lead_aligned=market_lead_aligned,
            lead_active=_signal_tracker_direction_key(lead_direction) in {"UPSIDE", "DOWNSIDE"},
            rr_ratio=row.get("__scalp_rr_val_raw", row.get("__rr_val")),
            adaptive_edge_score=row.get("__adaptive_edge_score"),
            session_fit_score=float(getattr(session_fit_snapshot, "score", 0.0) or 0.0),
            archive_guardrail_penalty=row.get("__archive_guardrail_penalty"),
            archive_guardrail_label=row.get("__archive_guardrail_label"),
            archive_guardrail_note=row.get("__archive_guardrail_note"),
            archive_risk_delta=row.get("__risk_archive_delta"),
            archive_risk_note=row.get("__risk_archive_note"),
            symbol=symbol,
            sector_tag=str(sector_tag or ""),
        )
        risk_label = str(row.get("__risk_tier_label") or getattr(risk_sizing_snapshot, "label", "") or "")
        risk_fraction = row.get("__risk_unit_fraction")
        risk_fraction_value = (
            _sortable_float(risk_fraction)
            if risk_fraction is not None
            else float(getattr(risk_sizing_snapshot, "unit_fraction", 0.0) or 0.0)
        )
        events.append(
            {
                "source": "Scalp",
                "decision_version": decision_version,
                "symbol": symbol,
                "timeframe": str(row.get("__timeframe") or timeframe),
                "event_time": event_time,
                "session_bucket": session_bucket_for_timestamp(event_time),
                "scan_focus": _normalize_scan_mode(scan_mode),
                "direction": direction_raw,
                "setup_confirm": str(row.get("__action_raw", row.get("Setup Confirm", "")) or ""),
                "action_reason": scalp_state,
                "lead_label": str(row.get("__scalp_reason_short") or ""),
                "lead_direction": lead_direction,
                "confidence": confidence_val,
                "ai_ensemble": ai_ensemble,
                "ai_direction": ai_ensemble.split("(", 1)[0].strip(),
                "ai_confidence": ai_conf_val,
                "market_lead_label": market_lead_snapshot.label,
                "market_lead_score": market_lead_snapshot.score,
                "market_lead_upside": market_lead_snapshot.upside_leads,
                "market_lead_downside": market_lead_snapshot.downside_leads,
                "market_regime": str(getattr(market_regime_snapshot, "label", "") or ""),
                "market_playbook_key": str(getattr(market_regime_snapshot, "playbook_key", "") or ""),
                "market_playbook": str(getattr(market_regime_snapshot, "playbook", "") or ""),
                "market_no_trade": bool(getattr(market_trade_gate_snapshot, "no_trade", False)),
                "market_trade_gate_key": str(getattr(market_trade_gate_snapshot, "gate_key", "") or ""),
                "market_trade_gate": str(getattr(market_trade_gate_snapshot, "label", "") or ""),
                "market_alert_keys": alert_keys_text,
                "market_primary_alert": primary_alert,
                "market_no_trade_reason": str(getattr(market_trade_gate_snapshot, "reason_code", "") or ""),
                "risk_tier": risk_label,
                "risk_unit_fraction": risk_fraction_value,
                "sector_tag": str(sector_tag or "").strip(),
                "market_sector_rotation": str(getattr(sector_rotation_snapshot, "label", "") or ""),
                "market_catalyst_state": str(getattr(market_catalyst_snapshot, "label", "") or ""),
                "market_catalyst_event": str(getattr(market_catalyst_snapshot, "next_event", "") or ""),
                "market_catalyst_blocking": bool(getattr(market_catalyst_snapshot, "blocking", False)),
                "market_catalyst_category": str(getattr(market_catalyst_snapshot, "category", "") or ""),
                "market_catalyst_scope": str(getattr(market_catalyst_snapshot, "scope", "") or ""),
                "market_catalyst_tag": str(getattr(market_catalyst_snapshot, "tag", "") or ""),
                "market_catalyst_targeted": bool(getattr(market_catalyst_snapshot, "targeted_only", False)),
                "market_catalyst_window": catalyst_window_label(market_catalyst_snapshot),
                "market_flow_state": str(getattr(market_flow_snapshot, "label", "") or ""),
                "market_flow_bias": str(getattr(market_flow_snapshot, "state", "") or ""),
                "adaptive_edge_label": str(row.get("__adaptive_edge_label") or ""),
                "adaptive_edge_score": row.get("__adaptive_edge_score"),
                "actionable_frame_score": row.get("__actionable_frame_score"),
                "actionable_setup_score": row.get("__actionable_setup_score"),
                "actionable_context_score": row.get("__actionable_context_score"),
                "actionable_tactical_score": row.get("__actionable_tactical_score"),
                "archive_guardrail_label": str(row.get("__archive_guardrail_label") or ""),
                "archive_guardrail_penalty": row.get("__archive_guardrail_penalty"),
                "archive_guardrail_note": str(row.get("__archive_guardrail_note") or ""),
                "price": row.get("__price_val"),
                "delta_pct": row.get("__delta_pct"),
                "entry_price": row.get("__scalp_entry_val_raw", row.get("__entry_val")),
                "stop_loss": row.get("__scalp_stop_val_raw", row.get("__stop_val")),
                "target_price": row.get("__scalp_target_val_raw", row.get("__target_val")),
                "rr_ratio": row.get("__scalp_rr_val_raw", row.get("__rr_val")),
            }
        )
    return events


def _market_lead_breadth_component(rows: list[dict]) -> tuple[float, int, int]:
    upside = 0
    downside = 0
    for row in rows:
        label = str((row or {}).get("__emerging_label", "")).strip()
        if label == "Emerging Upside":
            upside += 1
        elif label == "Emerging Downside":
            downside += 1
    total = upside + downside
    if total < 3:
        return 0.0, upside, downside
    score = ((upside - downside) / float(total)) * 100.0
    return float(max(-100.0, min(100.0, score))), upside, downside


def _market_lead_snapshot(
    *,
    produced_rows: list[dict],
    delta_mcap: float | None,
    btc_change: float | None,
    eth_change: float | None,
    btc_dom: float | None,
    eth_dom: float | None,
    custom_mode_active: bool,
) -> MarketLeadSnapshot:
    def _clip_signed(value: float | None, scale: float) -> float:
        if value is None or pd.isna(value):
            return 0.0
        return float(max(-100.0, min(100.0, float(value) * scale)))

    breadth_component, upside_leads, downside_leads = _market_lead_breadth_component(
        [] if custom_mode_active else list(produced_rows or [])
    )

    majors = [
        float(v)
        for v in (btc_change, eth_change)
        if v is not None and not pd.isna(v)
    ]
    major_avg = float(sum(majors) / len(majors)) if majors else 0.0
    rotation_component = _clip_signed(
        (float(delta_mcap) if delta_mcap is not None and not pd.isna(delta_mcap) else 0.0) - major_avg,
        18.0,
    )
    flow_component = _clip_signed(delta_mcap, 16.0)

    dom_parts: list[float] = []
    if btc_dom is not None and not pd.isna(btc_dom):
        dom_parts.append((55.0 - float(btc_dom)) * 8.0)
    if eth_dom is not None and not pd.isna(eth_dom):
        dom_parts.append((float(eth_dom) - 11.0) * 14.0)
    dominance_component = float(
        max(-100.0, min(100.0, (sum(dom_parts) / len(dom_parts)) if dom_parts else 0.0))
    )

    signed_score = (
        0.35 * breadth_component
        + 0.30 * rotation_component
        + 0.20 * flow_component
        + 0.15 * dominance_component
    )
    score = float(max(0.0, min(100.0, 50.0 + signed_score * 0.5)))

    if score >= 62.0:
        state = "UPSIDE"
        label = "Upside"
        note = "Early upside pressure is building before full confirmation."
    elif score <= 38.0:
        state = "DOWNSIDE"
        label = "Downside"
        note = "Early downside pressure is building before full confirmation."
    elif abs(signed_score) < 10.0 and (upside_leads + downside_leads) < 3:
        state = "NONE"
        label = "No Clear Pressure"
        note = "No meaningful early pressure is building yet."
    else:
        state = "BALANCED"
        label = "Balanced"
        note = "Early pressure is mixed across breadth and market internals."

    return MarketLeadSnapshot(
        score=score,
        state=state,
        label=label,
        note=note,
        breadth_component=breadth_component,
        rotation_component=rotation_component,
        flow_component=flow_component,
        dominance_component=dominance_component,
        upside_leads=upside_leads,
        downside_leads=downside_leads,
    )


def _coin_pair_meta(coin: str, pair: str) -> str:
    coin_txt = str(coin or "").strip()
    pair_txt = str(pair or "").strip()
    if not coin_txt or not pair_txt:
        return ""
    coin_up = coin_txt.upper()
    pair_up = pair_txt.upper()
    if pair_up in {
        coin_up,
        f"{coin_up}/USDT",
        f"{coin_up}/USD",
        f"{coin_up}/USDC",
    }:
        return ""
    return pair_txt


def _ai_fallback_note(ai_details: dict | None) -> str:
    if not isinstance(ai_details, dict):
        return ""
    status = str(ai_details.get("status") or "").strip()
    if not status:
        return ""
    note = _AI_FALLBACK_STATUS_TEXT.get(status, "")
    if not note:
        return ""
    err_detail = str(ai_details.get("error") or "").strip()
    if err_detail and status == "model_exception":
        return f"{note} Reason: {err_detail}"
    return note


def _setup_status_summary(
    *,
    enter_count: int,
    probe_count: int = 0,
    watch_count: int,
    skip_count: int,
    source_label: str | None,
) -> tuple[str, str, str]:
    source = str(source_label or "").strip().upper()
    label = "Setup Readiness"
    if source.startswith("CACHED"):
        head = (
            copy_text("market.status.head.cached_ready")
            if int(enter_count) > 0
            else (
                copy_text("market.status.head.cached_probe")
                if int(probe_count) > 0
                else copy_text("market.status.head.cached_none")
            )
        )
        sub = copy_text(
            "market.status.sub.cached",
            enter_count=enter_count,
            probe_count=probe_count,
            watch_count=watch_count,
            skip_count=skip_count,
        )
        return label, head, sub
    if "DEGRADED" in source:
        head = (
            copy_text("market.status.head.degraded_ready")
            if int(enter_count) > 0
            else (
                copy_text("market.status.head.degraded_probe")
                if int(probe_count) > 0
                else copy_text("market.status.head.degraded_none")
            )
        )
        sub = copy_text(
            "market.status.sub.degraded",
            enter_count=enter_count,
            probe_count=probe_count,
            watch_count=watch_count,
            skip_count=skip_count,
        )
        return label, head, sub
    head = (
        copy_text("market.status.head.live_ready")
        if int(enter_count) > 0
        else (
            copy_text("market.status.head.live_probe")
            if int(probe_count) > 0
            else copy_text("market.status.head.live_none")
        )
    )
    sub = copy_text(
        "market.status.sub.live",
        enter_count=enter_count,
        probe_count=probe_count,
        watch_count=watch_count,
        skip_count=skip_count,
    )
    return label, head, sub


def _alert_archive_label(alert_key: str) -> str:
    key = str(alert_key or "").strip().upper()
    return _ALERT_ARCHIVE_DISPLAY.get(key, key.replace("_", " ").title() if key else "No Alert Footprint")


def _alert_lane_label(alert: object) -> str:
    severity = str(getattr(alert, "severity", "INFO") or "INFO").strip().upper()
    tone = str(getattr(alert, "tone", "") or "").strip().lower()
    if severity == "HIGH":
        return trade_gate_display_label("NO_TRADE")
    if severity == "MEDIUM":
        if tone == "positive":
            return "Action"
        return "Caution"
    if tone == "positive":
        return "Context+"
    return "Context"


def _alert_is_primary(alert: object) -> bool:
    severity = str(getattr(alert, "severity", "INFO") or "INFO").strip().upper()
    alert_key = str(getattr(alert, "alert_key", "") or "").strip().upper()
    if severity in {"HIGH", "MEDIUM"}:
        return True
    return alert_key in (_PROTECTED_ALERT_KEYS | {"MARKET_LEAD", "ACTIONABLE_CLUSTER", "LEARNED_EDGE"})


def _compress_market_alerts_for_display(alerts: list[object], *, max_items: int = 2) -> list[object]:
    ordered = list(alerts or [])
    limit = max(1, int(max_items or 0))
    if len(ordered) <= limit:
        return ordered

    primary = [alert for alert in ordered if _alert_is_primary(alert)]
    context = [alert for alert in ordered if not _alert_is_primary(alert)]
    display: list[object] = primary[:limit]

    if len(display) >= limit:
        return display[:limit]

    slots_left = limit - len(display)
    if not context:
        return (display + primary[len(display):])[:limit]

    if slots_left > 1:
        display.extend(context[:slots_left])
        return display[:limit]

    positive_titles = [
        str(getattr(alert, "title", "") or "").strip()
        for alert in context
        if str(getattr(alert, "tone", "") or "").strip().lower() == "positive"
    ]
    caution_titles = [
        str(getattr(alert, "title", "") or "").strip()
        for alert in context
        if str(getattr(alert, "tone", "") or "").strip().lower() != "positive"
    ]
    if caution_titles:
        summary_titles = [title for title in caution_titles[:2] if title]
        summary = SimpleNamespace(
            alert_key="CONTEXT_STACK",
            severity="INFO",
            tone="warning",
            title="Context stack needs caution",
            note=(
                f"Also watching {', '.join(summary_titles)}."
                if summary_titles
                else "Several secondary context reads are weakening the market."
            ),
        )
    else:
        summary_titles = [title for title in positive_titles[:3] if title]
        summary = SimpleNamespace(
            alert_key="CONTEXT_STACK",
            severity="INFO",
            tone="positive",
            title="Supportive context stack is lining up",
            note=(
                f"Also watching {', '.join(summary_titles)}."
                if summary_titles
                else "Several secondary context reads are supporting the current market."
            ),
        )
    display.append(summary)
    return display[:limit]


def _rank_market_alerts_by_archive(alerts: list[object], df_events: pd.DataFrame) -> list[object]:
    ordered_alerts = list(alerts or [])
    if not ordered_alerts or df_events is None or df_events.empty:
        return ordered_alerts
    summary_df = build_alert_effectiveness_summary(df_events, primary_only=True)
    if summary_df.empty or "Primary Alert" not in summary_df.columns:
        return ordered_alerts

    archive_rows = {
        str(row.get("Primary Alert") or "").strip(): row
        for _, row in summary_df.iterrows()
    }
    ranked: list[tuple[int, float, int, object]] = []
    for idx, alert in enumerate(ordered_alerts):
        alert_key = str(getattr(alert, "alert_key", "") or "").strip().upper()
        archive_label = _alert_archive_label(alert_key)
        row = archive_rows.get(archive_label)
        follow_score = 50.0
        actual_score = 50.0
        if row is not None:
            resolved = float(row.get("Resolved", 0.0) or 0.0)
            closed_count = float(row.get("ClosedTradeCount", 0.0) or 0.0)
            if resolved >= 4.0:
                follow_score = float(row.get("FollowThroughPct", 50.0) or 50.0)
            if closed_count >= 2.0:
                actual_score = float(row.get("ActualWinRatePct", 50.0) or 50.0)
        archive_score = (follow_score * 0.7) + (actual_score * 0.3)
        protected_rank = 0 if alert_key in _PROTECTED_ALERT_KEYS else 1
        ranked.append((protected_rank, -archive_score, idx, alert))
    ranked.sort(key=lambda item: (item[0], item[1], item[2]))
    return [alert for _, _, _, alert in ranked]


def _trade_gate_banner_html(label: str, note: str, tone: str, reason_code: str) -> str:
    tone_key = str(tone or "warning").strip().lower()
    accent = {
        "positive": "var(--positive, #3CF2A4)",
        "negative": "var(--negative, #FF4D7A)",
        "warning": "var(--warning, #FFD166)",
    }.get(tone_key, "var(--warning, #FFD166)")
    reason = _market_stance_reason_label(reason_code)
    meta_html = (
        f"<div style='font-size:0.72rem; letter-spacing:0.12em; text-transform:uppercase; color:rgba(255,255,255,0.56);'>{html.escape(reason)}</div>"
        if reason
        else ""
    )
    return (
        "<div class='app-insight-card app-insight-card--neutral' "
        f"style='border-color:{accent}; box-shadow:0 0 0 1px color-mix(in srgb, {accent} 20%, transparent) inset;'>"
        "<div style='display:flex; align-items:flex-start; justify-content:space-between; gap:14px;'>"
        "<div>"
        "<div class='app-insight-title'>Market Stance</div>"
        f"<div class='app-insight-body'>{html.escape(str(note or '').strip())}</div>"
        "</div>"
        "<div style='display:flex; flex-direction:column; align-items:flex-end; gap:6px;'>"
        f"<span class='app-chip app-chip--neutral' style='color:{accent}; border-color:{accent}; background:rgba(0,0,0,0.28); white-space:nowrap;'>{html.escape(str(label or '').strip())}</span>"
        f"{meta_html}"
        "</div>"
        "</div>"
        "</div>"
    )


def _market_stance_reason_label(reason_code: str) -> str:
    key = str(reason_code or "").strip().upper()
    labels = {
        "DEGRADED_SCAN": "Partial Data",
        "REGIME_NO_TRADE": "Market Not Ready",
        "CATALYST_BLOCK": "Catalyst Nearby",
        "PROBE_ONLY_SETUPS": "Early Only",
        "NO_READY_SETUPS": "No Ready Setups",
        "WEAK_PARTICIPATION": "Weak Breadth",
        "ARCHIVE_CLUSTER_NO_TRADE": "History Caution",
        "RISK_OFF_DEFENSIVE": "Risk-Off",
        "RISK_OFF_WEAKNESS": "Risk-Off Weakness",
        "CATALYST_SELECTIVE": "Catalyst Caution",
        "SESSION_ARCHIVE_WEAK": "Weak Session",
        "ARCHIVE_GUARDRAIL": "History Caution",
        "SELECTIVE_FILTER": "Selective",
        "SELECTIVE_PROBE_WINDOW": "Early Window",
        "SELECTIVE_SESSION_WEAK": "Weak Session",
        "SELECTIVE_ARCHIVE_WEAK": "History Caution",
        "FILTER_HARDER_SESSION_WEAK": "Filter Harder",
        "FILTER_HARDER_ARCHIVE": "History Caution",
        "ARCHIVE_GATE_CAUTION": "History Caution",
        "ARCHIVE_GATE_SUPPORT": "History Support",
        "RISK_ON_CLEAR": "Risk-On",
    }
    if key in labels:
        return labels[key]
    return key.replace("_", " ").title()


def _compact_alert_note(note: str) -> str:
    text = re.sub(r"\s+", " ", str(note or "").strip())
    if not text:
        return ""
    sentence_match = re.match(r"^(.+?[.!?])(?:\s|$)", text)
    if sentence_match:
        first = sentence_match.group(1).strip()
        if len(first) <= 120:
            return first
    if len(text) <= 120:
        return text
    clipped = text[:117].rstrip(" ,.;:")
    return f"{clipped}..."


def _compact_hover_note(note: str, *, limit: int = 140) -> str:
    text = re.sub(r"\s+", " ", str(note or "").strip())
    if not text:
        return ""
    sentence_match = re.match(r"^(.+?[.!?])(?:\s|$)", text)
    if sentence_match:
        first = sentence_match.group(1).strip()
        if len(first) <= limit:
            return first
    if len(text) <= limit:
        return text
    clipped = text[: max(0, limit - 3)].rstrip(" ,.;:")
    return f"{clipped}..."


def _market_alert_strip_html(alerts: list[object], *, total_active: int | None = None) -> str:
    rows_html: list[str] = []
    tone_map = {
        "positive": "var(--positive, #3CF2A4)",
        "negative": "var(--negative, #FF4D7A)",
        "warning": "var(--warning, #FFD166)",
    }
    severity_tone = {
        "HIGH": "negative",
        "MEDIUM": "warning",
        "INFO": "positive",
    }
    for alert in list(alerts or []):
        severity = str(getattr(alert, "severity", "INFO") or "INFO").strip().upper()
        tone_key = str(getattr(alert, "tone", "") or severity_tone.get(severity, "warning")).strip().lower()
        accent = tone_map.get(tone_key, tone_map["warning"])
        lane_label = _alert_lane_label(alert)
        title = str(getattr(alert, "title", "") or "").strip()
        note = _compact_alert_note(str(getattr(alert, "note", "") or "").strip())
        rows_html.append(
            "<div style='flex:1 1 340px; min-width:280px; border:1px solid rgba(255,255,255,0.08); "
            "border-radius:14px; background:rgba(255,255,255,0.018); padding:10px 12px;'>"
            "<div style='display:flex; align-items:flex-start; gap:10px;'>"
            f"<span class='app-chip app-chip--neutral' style='white-space:nowrap; color:{accent}; border-color:{accent};"
            " background:rgba(0,0,0,0.24); min-width:64px; justify-content:center;'>"
            f"{html.escape(lane_label)}</span>"
            "<div style='min-width:0;'>"
            f"<div style='font-weight:700; color:#F5F7FB; font-size:0.90rem; line-height:1.25;'>{html.escape(title)}</div>"
            f"<div style='color:rgba(255,255,255,0.68); font-size:0.82rem; line-height:1.38; margin-top:4px;'>{html.escape(note)}</div>"
            "</div>"
            "</div>"
            "</div>"
        )
    if not rows_html:
        return ""
    total_count = int(total_active if total_active is not None else len(rows_html))
    return (
        "<div class='app-insight-card app-insight-card--neutral' "
        "style='border-color:rgba(255,255,255,0.06); background:rgba(255,255,255,0.01);'>"
        "<div style='display:flex; align-items:center; justify-content:space-between; gap:12px;'>"
        "<div class='app-insight-title'>Market Notes</div>"
        f"<div style='font-size:0.72rem; letter-spacing:0.12em; text-transform:uppercase; color:rgba(255,255,255,0.56);'>{len(rows_html)} shown • {total_count} current</div>"
        "</div>"
        "<div style='display:flex; flex-wrap:wrap; gap:10px; margin-top:12px;'>"
        f"{''.join(rows_html)}"
        "</div>"
        "</div>"
    )


def _queue_market_custom_clear(session_state: dict) -> None:
    session_state["market_clear_custom_pending"] = True
    session_state["market_custom_bases_applied"] = []


def _consume_market_custom_clear(session_state: dict) -> None:
    if not bool(session_state.pop("market_clear_custom_pending", False)):
        return
    session_state.pop("market_custom_coin_input", None)
    session_state["market_custom_bases_applied"] = []


def _share_line(counts: dict[str, int], order: list[str]) -> str:
    total = int(sum(max(0, int(v)) for v in counts.values()))
    if total <= 0:
        return "No rows."
    parts: list[str] = []
    seen: set[str] = set()
    for key in order:
        if key in counts:
            seen.add(key)
            count = int(counts.get(key, 0) or 0)
            pct = count / total * 100.0
            parts.append(f"{key}: {count} ({pct:.0f}%)")
    for key in sorted(counts.keys()):
        if key in seen:
            continue
        count = int(counts.get(key, 0) or 0)
        pct = count / total * 100.0
        parts.append(f"{key}: {count} ({pct:.0f}%)")
    return " • ".join(parts)


def _share_line_against_total(counts: dict[str, int], order: list[str], total: int) -> str:
    total_n = int(max(0, total))
    if total_n <= 0:
        return "No rows."
    parts: list[str] = []
    for key in order:
        count = int(counts.get(key, 0) or 0)
        pct = count / total_n * 100.0
        parts.append(f"{key}: {count} ({pct:.0f}%)")
    return " • ".join(parts)


def _extract_ai_verdict(value: object) -> str:
    text = str(value or "").strip()
    if not text:
        return "Unknown"
    verdict = re.sub(r"\s*\(\s*\d+\s*/\s*3\s*\)\s*\*?\s*$", "", text).strip()
    return verdict or "Unknown"


def _extract_confidence_label(value: object) -> str:
    text = str(value or "").strip()
    match = re.search(r"\(([^()]+)\)\s*$", text)
    if match:
        return match.group(1).strip() or "Unknown"
    return text or "Unknown"


def _row_direction_vote(row: dict) -> str:
    direction = _signal_tracker_direction_key(row.get("Direction", row.get("__direction", "")))
    if direction in {"UPSIDE", "DOWNSIDE"}:
        return direction
    return ""


def _is_numeric_indicator_active(value: object, *, minimum: float) -> bool:
    try:
        numeric = float(value)
    except Exception:
        match = re.search(r"(-?\d+(?:\.\d+)?)", str(value or ""))
        if not match:
            return False
        try:
            numeric = float(match.group(1))
        except Exception:
            return False
    return bool(pd.notna(numeric) and numeric >= minimum)


def _indicator_direction_vote(value: object, *, column: str, row: dict | None = None) -> str:
    row = row or {}
    column_key = str(column or "").strip()
    text = str(value or "").strip()
    upper = text.upper()
    if not text or upper in {"NAN", "NONE", "N/A", "UNAVAILABLE", "NEUTRAL", "MIXED"}:
        return ""

    if column_key == "ADX":
        if not _is_numeric_indicator_active(value, minimum=25.0) and not any(
            marker in upper for marker in ("STRONG", "VERY STRONG", "EXTREME", "🔥")
        ):
            return ""
        return _row_direction_vote(row)

    if column_key == "Volatility":
        if "HIGH" not in upper and "EXTREME" not in upper:
            return ""
        return _row_direction_vote(row)

    if column_key == "Spike Alert":
        spike_dir = str(row.get("__spike_dir", "") or "").strip().upper()
        if spike_dir in {"UP", "UPSIDE", "BULLISH"}:
            return "UPSIDE"
        if spike_dir in {"DOWN", "DOWNSIDE", "BEARISH"}:
            return "DOWNSIDE"
        if "SPIKE" not in upper:
            return ""
        return _row_direction_vote(row)

    if column_key == "Stochastic RSI":
        try:
            numeric = float(value)
        except Exception:
            numeric = float("nan")
        if pd.notna(numeric):
            if numeric <= 25.0:
                return "UPSIDE"
            if numeric >= 75.0:
                return "DOWNSIDE"
            return ""

    upside_markers = (
        "▲",
        "🟢",
        "BULLISH",
        "ABOVE",
        "OVERSOLD",
        "NEAR BOTTOM",
        "LOW",
        "SUPPORT",
        "UP SPIKE",
    )
    downside_markers = (
        "▼",
        "🔴",
        "BEARISH",
        "BELOW",
        "OVERBOUGHT",
        "NEAR TOP",
        "HIGH",
        "RESISTANCE",
        "DOWN SPIKE",
    )
    neutral_markers = ("NEUTRAL", "MIXED", "BALANCED", "NEAR VWAP", "MID", "MODERATE", "STARTING", "WEAK")
    if any(marker in upper for marker in neutral_markers):
        return ""
    if any(marker in upper for marker in downside_markers):
        return "DOWNSIDE"
    if any(marker in upper for marker in upside_markers):
        return "UPSIDE"
    return ""


def _clearest_direction_group_count(row: dict, columns: tuple[str, ...], target_direction: str) -> int:
    target = str(target_direction or "").strip().upper()
    return sum(
        1
        for column in columns
        if _indicator_direction_vote(row.get(column, ""), column=column, row=row) == target
    )


def _pick_clearest_direction(df_results: pd.DataFrame) -> tuple[str, str]:
    if len(df_results) <= 0:
        return "No direction read", "Current table has no advanced indicator data yet."

    ranked: list[tuple[float, float, float, float, float, float, str, str, str]] = []
    for _, row_series in df_results.iterrows():
        row = row_series.to_dict()
        votes = [
            _indicator_direction_vote(row.get(column, ""), column=column, row=row)
            for column in (*_CLEAREST_TREND_COLS, *_CLEAREST_MOMENTUM_COLS, *_CLEAREST_ACTIVITY_COLS)
        ]
        up_votes = sum(1 for vote in votes if vote == "UPSIDE")
        down_votes = sum(1 for vote in votes if vote == "DOWNSIDE")
        if up_votes == down_votes or max(up_votes, down_votes) <= 0:
            continue

        direction = "UPSIDE" if up_votes > down_votes else "DOWNSIDE"
        dominant_votes = max(up_votes, down_votes)
        opposing_votes = min(up_votes, down_votes)
        trend_count = _clearest_direction_group_count(row, _CLEAREST_TREND_COLS, direction)
        momentum_count = _clearest_direction_group_count(row, _CLEAREST_MOMENTUM_COLS, direction)
        activity_count = _clearest_direction_group_count(row, _CLEAREST_ACTIVITY_COLS, direction)
        confidence = _sortable_float(row.get("__confidence_val", 0.0))
        setup_priority = float(_setup_confirm_priority(str(row.get("__action_raw", row.get("Setup Confirm", "")))))
        coin = str(row.get("Coin", "—") or "—").strip() or "—"
        subtext = (
            f"Trend {trend_count}/{len(_CLEAREST_TREND_COLS)} • "
            f"Momentum {momentum_count}/{len(_CLEAREST_MOMENTUM_COLS)} • "
            f"Activity {activity_count}/{len(_CLEAREST_ACTIVITY_COLS)}"
        )
        ranked.append(
            (
                float(dominant_votes),
                float(dominant_votes - opposing_votes),
                float(trend_count),
                float(momentum_count),
                float(activity_count),
                confidence + setup_priority,
                coin,
                direction.title(),
                subtext,
            )
        )

    if not ranked:
        return "No clear direction", "No clear advanced indicator alignment in the current table."

    ranked.sort(key=lambda item: (-item[0], -item[1], -item[2], -item[3], -item[4], -item[5], item[6]))
    _, _, _, _, _, _, coin, direction, subtext = ranked[0]
    return f"{coin} • {direction}", subtext


def _pick_best_scalp_opportunity(df_results: pd.DataFrame) -> tuple[str, str]:
    if "Scalp Opportunity" not in df_results.columns or len(df_results) <= 0:
        return "No scalp read", "Current table has no scalp timing data yet."

    working = df_results.copy()
    working["__scalp_state"] = (
        working.get("__scalp_display_state", pd.Series(index=working.index, dtype=object))
        .astype(str)
        .str.upper()
    )
    working["__scalp_reason_short"] = (
        working.get("__scalp_reason_short", pd.Series(index=working.index, dtype=object))
        .astype(str)
        .str.strip()
    )
    working = working[working["Scalp Opportunity"].astype(str).isin(["Upside", "Downside"])]
    if working.empty:
        return "No clean scalp", "No live or conditional scalp candidate in the current table."

    live_count = int(working["__scalp_state"].eq("LIVE").sum())
    conditional_count = int(working["__scalp_state"].eq("CONDITIONAL").sum())
    working["__rr"] = pd.to_numeric(
        working["R:R"]
        .astype(str)
        .str.replace("🟢", "", regex=False)
        .str.replace("🟡", "", regex=False)
        .str.replace("🔴", "", regex=False)
        .str.replace("*", "", regex=False)
        .str.strip(),
        errors="coerce",
    )
    working["__action_rank"] = (
        working.get("__action_raw", working.get("Setup Confirm", pd.Series(dtype=str)))
        .astype(str)
        .apply(_setup_confirm_priority)
    )
    working["__state_rank"] = working["__scalp_state"].map({"LIVE": 2, "CONDITIONAL": 1}).fillna(0)
    working["__confidence_num"] = pd.to_numeric(
        working.get("__confidence_val", pd.Series(index=working.index, dtype=float)),
        errors="coerce",
    ).fillna(0.0)
    working = working.dropna(subset=["__rr"])
    working = working[working["__rr"] > 0]
    if working.empty:
        count_sub = f"Live: {live_count} • Conditional: {conditional_count}"
        return "No valid R:R", count_sub

    scoped = working[working["__scalp_state"] == "LIVE"].copy()
    if scoped.empty:
        scoped = working[working["__scalp_state"] == "CONDITIONAL"].copy()
    if scoped.empty:
        scoped = working

    best_row = scoped.sort_values(
        ["__rr", "__action_rank", "__confidence_num", "Coin"],
        ascending=[False, False, False, True],
    ).iloc[0]
    best_coin = str(best_row.get("Coin", "—"))
    best_rr = float(best_row.get("__rr", 0.0) or 0.0)
    best_head = f"{best_coin} ({best_rr:.2f})"
    best_state = str(best_row.get("__scalp_state", "")).strip().title() or "Scalp"
    best_direction = str(best_row.get("Scalp Opportunity", "")).strip()
    best_reason = str(best_row.get("__scalp_reason_short", "")).strip()
    best_action = str(best_row.get("__action_raw", best_row.get("Setup Confirm", ""))).strip()
    best_action_compact = _shared_setup_confirm_display(
        best_action,
        action_reason=str(best_row.get("__action_reason", "")).strip(),
        direction=str(best_row.get("Direction", "")).strip(),
    )

    sub_parts = [best_state, best_direction]
    if best_state.upper() == "CONDITIONAL" and best_reason:
        sub_parts.append(best_reason)
    sub_parts.append(f"Setup: {best_action_compact}")
    sub_parts.append(f"Live: {live_count}")
    sub_parts.append(f"Conditional: {conditional_count}")
    return best_head, " • ".join(sub_parts)


def _valid_market_bases(market_rows: list[dict]) -> set[str]:
    out: set[str] = set()
    for row in market_rows:
        base = canonical_base_symbol((row or {}).get("symbol") or "")
        if base:
            out.add(base)
    return out


def _filter_scan_symbols(usdt_symbols: list[str], market_rows: list[dict]) -> list[str]:
    valid_bases = _valid_market_bases(market_rows)
    if not valid_bases:
        return list(usdt_symbols)
    return [pair for pair in usdt_symbols if _canonical_pair_base(pair) in valid_bases]


def _build_market_cap_map(market_rows: list[dict]) -> dict[str, int]:
    out: dict[str, int] = {}
    for coin in market_rows:
        symbol = canonical_base_symbol((coin or {}).get("symbol") or "")
        raw_mcap = (coin or {}).get("market_cap")
        try:
            if isinstance(raw_mcap, str):
                raw_mcap = raw_mcap.replace(",", "").strip()
            mcap_f = float(raw_mcap) if raw_mcap is not None and pd.notna(raw_mcap) else 0.0
            mcap = int(mcap_f) if pd.notna(mcap_f) and mcap_f > 0 else 0
        except Exception:
            mcap = 0
        if symbol and (symbol not in out or mcap > out[symbol]):
            out[symbol] = mcap
    return out


def _build_market_coin_id_map(market_rows: list[dict]) -> dict[str, str]:
    out: dict[str, str] = {}
    for coin in market_rows:
        symbol = canonical_base_symbol((coin or {}).get("symbol") or "")
        coin_id = str((coin or {}).get("id") or "").strip().lower()
        if symbol and coin_id and symbol not in out:
            out[symbol] = coin_id
    return out


def _build_market_row_map(market_rows: list[dict]) -> dict[str, dict]:
    out: dict[str, dict] = {}
    for coin in market_rows:
        symbol = canonical_base_symbol((coin or {}).get("symbol") or "")
        if symbol and symbol not in out:
            out[symbol] = dict(coin)
    return out


def _build_custom_scan_universe(
    *,
    custom_bases_applied: list[str],
    get_market_cap_rows_for_symbols,
    exclude_stables: bool,
    scan_pool_n: int,
) -> tuple[list[dict], dict[str, int], list[str], list[str]]:
    unique_market_data, mcap_map = _prepare_scan_market_enrichment(
        get_market_cap_rows_for_symbols(tuple(custom_bases_applied), vs_currency="usd")
    )
    usdt_symbols = [f"{base}/USDT" for base in custom_bases_applied]
    candidate_symbol_pool = _candidate_scan_symbols(
        usdt_symbols=usdt_symbols,
        market_rows=[],
        exclude_stables=bool(exclude_stables),
        custom_bases_applied=custom_bases_applied,
    )[:scan_pool_n]
    return unique_market_data, mcap_map, usdt_symbols, candidate_symbol_pool


def _custom_watchlist_fallback_coin_id(
    symbol: str,
    *,
    custom_mode_active: bool,
    coin_id_map: dict[str, str],
) -> str | None:
    if not bool(custom_mode_active):
        return None
    return coin_id_map.get(_canonical_pair_base(symbol))


def _fetch_market_scan_ohlcv(
    *,
    fetch_ohlcv,
    fetch_coingecko_ohlcv_by_coin_id,
    fetch_lock,
    symbol: str,
    timeframe: str,
    limit: int,
    fallback_coin_id: str | None = None,
) -> pd.DataFrame | None:
    with fetch_lock:
        df = fetch_ohlcv(symbol, timeframe, limit=limit)
        if df is not None:
            return df
        if not fallback_coin_id:
            return None
        return fetch_coingecko_ohlcv_by_coin_id(fallback_coin_id, timeframe, limit=limit)


def _prepare_closed_frame(df: pd.DataFrame | None, *, min_rows: int = 55) -> pd.DataFrame | None:
    if df is None:
        return None
    if len(df) <= int(min_rows):
        return None
    df_eval = df.iloc[:-1].copy()
    if len(df_eval) < int(min_rows):
        return None
    return df_eval


def _auto_learning_ts(value: object | None = None) -> pd.Timestamp | None:
    if value is None or str(value).strip() == "":
        return None
    ts = pd.to_datetime(value, utc=True, errors="coerce")
    if pd.isna(ts):
        return None
    return pd.Timestamp(ts)


def _auto_learning_state(raw_state: object) -> dict[str, object]:
    state = dict(raw_state) if isinstance(raw_state, dict) else {}
    attempts = state.get("last_attempt_by_timeframe")
    if not isinstance(attempts, dict):
        attempts = {}
    results = state.get("last_result_by_timeframe")
    if not isinstance(results, dict):
        results = {}
    return {
        "running": bool(state.get("running")),
        "last_attempt_at": str(state.get("last_attempt_at") or ""),
        "last_attempt_by_timeframe": {
            str(k).strip().lower(): str(v or "")
            for k, v in attempts.items()
            if str(k).strip().lower() in _AUTO_TIMEFRAME_LEARNING_TIMEFRAMES
        },
        "last_result_by_timeframe": {
            str(k).strip().lower(): dict(v)
            for k, v in results.items()
            if str(k).strip().lower() in _AUTO_TIMEFRAME_LEARNING_TIMEFRAMES and isinstance(v, dict)
        },
    }


def _auto_learning_timeframe_stats(
    df_events: pd.DataFrame | None,
    df_forward_windows: pd.DataFrame | None = None,
) -> dict[str, dict[str, object]]:
    stats: dict[str, dict[str, object]] = {
        tf: {
            "rows": 0,
            "resolved_rows": 0,
            "usable_rows": 0,
            "window_rows": 0,
            "target_rows": int(_AUTO_TIMEFRAME_LEARNING_USABLE_TARGETS.get(tf, 24)),
            "coverage_ratio": 0.0,
            "checkpoint_coverage_ratio": 0.0,
            "needs_checkpoint_backfill": True,
            "last_event": None,
        }
        for tf in _AUTO_TIMEFRAME_LEARNING_TIMEFRAMES
    }
    if df_events is None or df_events.empty or "timeframe" not in df_events.columns:
        return stats
    d = df_events.copy()
    d["__tf"] = d["timeframe"].fillna("").astype(str).str.strip().str.lower()
    d["__status"] = (
        d["status"].fillna("").astype(str).str.strip().str.upper()
        if "status" in d.columns
        else "RESOLVED"
    )
    if "directional_return_pct" in d.columns:
        d["__has_outcome"] = pd.to_numeric(d["directional_return_pct"], errors="coerce").notna()
    else:
        d["__has_outcome"] = True
    window_keys: set[str] = set()
    window_scope_known = (
        isinstance(df_forward_windows, pd.DataFrame)
        and "signal_key" in df_forward_windows.columns
        and "signal_key" in d.columns
    )
    if window_scope_known:
        if "signal_key" in df_forward_windows.columns:
            window_keys = set(df_forward_windows["signal_key"].fillna("").astype(str).str.strip())
            window_keys.discard("")
        d["__signal_key"] = d["signal_key"].fillna("").astype(str).str.strip()
        d["__has_window"] = d["__signal_key"].isin(window_keys)
    else:
        d["__has_window"] = False
    event_col = "event_time" if "event_time" in d.columns else ("updated_at" if "updated_at" in d.columns else "")
    if event_col:
        d["__event_ts"] = pd.to_datetime(d[event_col], utc=True, errors="coerce")
    else:
        d["__event_ts"] = pd.NaT
    for tf in _AUTO_TIMEFRAME_LEARNING_TIMEFRAMES:
        group = d[d["__tf"].eq(tf)]
        if group.empty:
            continue
        last_event = group["__event_ts"].dropna().max()
        resolved_group = group[group["__status"].eq("RESOLVED")]
        resolved_with_outcome = resolved_group[resolved_group["__has_outcome"]]
        windowed_group = resolved_group[resolved_group["__has_window"]]
        usable_rows = int(len(windowed_group)) if window_scope_known else int(len(resolved_with_outcome))
        resolved_count = int(len(resolved_group))
        target_rows = int(_AUTO_TIMEFRAME_LEARNING_USABLE_TARGETS.get(tf, 24))
        checkpoint_ratio = float(len(windowed_group) / resolved_count) if resolved_count > 0 else 0.0
        coverage_ratio = min(1.0, float(usable_rows) / float(max(1, target_rows)))
        stats[tf] = {
            "rows": int(len(group)),
            "resolved_rows": resolved_count,
            "usable_rows": int(usable_rows),
            "window_rows": int(len(windowed_group)),
            "target_rows": target_rows,
            "coverage_ratio": coverage_ratio,
            "checkpoint_coverage_ratio": checkpoint_ratio,
            "needs_checkpoint_backfill": bool(window_scope_known and resolved_count > int(len(windowed_group))),
            "last_event": None if pd.isna(last_event) else pd.Timestamp(last_event),
        }
    return stats


def _select_auto_learning_timeframe(
    df_events: pd.DataFrame | None,
    state: dict[str, object] | None,
    *,
    df_forward_windows: pd.DataFrame | None = None,
    now: object | None = None,
    current_timeframe: str | None = None,
) -> str | None:
    selected = _select_auto_learning_timeframes(
        df_events,
        state,
        df_forward_windows=df_forward_windows,
        now=now,
        current_timeframe=current_timeframe,
        max_count=1,
    )
    return selected[0] if selected else None


def _select_auto_learning_timeframes(
    df_events: pd.DataFrame | None,
    state: dict[str, object] | None,
    *,
    df_forward_windows: pd.DataFrame | None = None,
    now: object | None = None,
    current_timeframe: str | None = None,
    max_count: int = _AUTO_TIMEFRAME_LEARNING_MAX_TIMEFRAMES_PER_PASS,
) -> list[str]:
    state = _auto_learning_state(state)
    now_ts = _auto_learning_ts(now) or pd.Timestamp.now(tz="UTC")
    last_global = _auto_learning_ts(state.get("last_attempt_at"))
    if last_global is not None:
        global_age = (now_ts - last_global).total_seconds()
        if global_age < _AUTO_TIMEFRAME_LEARNING_GLOBAL_COOLDOWN_SECONDS:
            return []

    stats = _auto_learning_timeframe_stats(df_events, df_forward_windows)
    attempts = state.get("last_attempt_by_timeframe")
    attempts = attempts if isinstance(attempts, dict) else {}
    current_tf = str(current_timeframe or "").strip().lower()
    ranked: list[tuple[float, int, str]] = []
    for idx, tf in enumerate(_AUTO_TIMEFRAME_LEARNING_TIMEFRAMES):
        interval = float(_AUTO_TIMEFRAME_LEARNING_MIN_INTERVAL_SECONDS.get(tf, 1800))
        last_attempt = _auto_learning_ts(attempts.get(tf))
        attempt_age = float("inf") if last_attempt is None else (now_ts - last_attempt).total_seconds()
        if attempt_age < interval:
            continue
        frame_stats = stats.get(tf, {})
        row_count = int(frame_stats.get("rows") or 0)
        usable_rows = int(frame_stats.get("usable_rows") or 0)
        resolved_rows = int(frame_stats.get("resolved_rows") or 0)
        target_rows = int(_AUTO_TIMEFRAME_LEARNING_USABLE_TARGETS.get(tf, 24))
        last_event = frame_stats.get("last_event")
        last_event_ts = last_event if isinstance(last_event, pd.Timestamp) else _auto_learning_ts(last_event)
        event_age = float("inf") if last_event_ts is None else max(0.0, (now_ts - last_event_ts).total_seconds())
        coverage_gap = max(0, target_rows - usable_rows)
        missing_bonus = 10_000.0 if usable_rows <= 0 else 0.0
        coverage_bonus = float(coverage_gap) * 180.0
        stale_bonus = min(2_000.0, event_age / max(interval, 1.0) * 120.0)
        scarcity_bonus = min(800.0, 800.0 / max(1.0, float(usable_rows or resolved_rows or row_count)))
        current_penalty = 1_600.0 if tf == current_tf else 0.0
        priority = missing_bonus + coverage_bonus + stale_bonus + scarcity_bonus - current_penalty
        ranked.append((priority, -idx, tf))
    if not ranked:
        return []
    ranked.sort(reverse=True)
    safe_count = max(1, int(max_count or 1))
    return [str(item[2]) for item in ranked[:safe_count]]


def _mark_auto_learning_attempt(
    state: dict[str, object] | None,
    *,
    timeframe: str,
    now: object | None = None,
    written: int = 0,
    resolved: int = 0,
    backfilled: int = 0,
    errors: int = 0,
) -> dict[str, object]:
    next_state = _auto_learning_state(state)
    tf = str(timeframe or "").strip().lower()
    now_ts = _auto_learning_ts(now) or pd.Timestamp.now(tz="UTC")
    now_iso = now_ts.strftime("%Y-%m-%dT%H:%M:%SZ")
    attempts = dict(next_state.get("last_attempt_by_timeframe") or {})
    results = dict(next_state.get("last_result_by_timeframe") or {})
    if tf:
        attempts[tf] = now_iso
        results[tf] = {
            "written": int(written),
            "resolved": int(resolved),
            "backfilled": int(backfilled),
            "errors": int(errors),
            "updated_at": now_iso,
        }
    next_state["last_attempt_at"] = now_iso
    next_state["last_attempt_by_timeframe"] = attempts
    next_state["last_result_by_timeframe"] = results
    return next_state


def _auto_learning_candle_limit(timeframe: str) -> int:
    return int(_AUTO_TIMEFRAME_LEARNING_CANDLE_LIMITS.get(str(timeframe or "").strip().lower(), 260))


def _auto_learning_symbol_limit(
    timeframe: str,
    df_events: pd.DataFrame | None = None,
    df_forward_windows: pd.DataFrame | None = None,
) -> int:
    tf = str(timeframe or "").strip().lower()
    base = int(_AUTO_TIMEFRAME_LEARNING_SYMBOL_LIMITS.get(tf, 8))
    stats = _auto_learning_timeframe_stats(df_events, df_forward_windows).get(tf, {})
    usable_rows = int(stats.get("usable_rows") or 0)
    target_rows = int(_AUTO_TIMEFRAME_LEARNING_USABLE_TARGETS.get(tf, 24))
    if usable_rows <= 0:
        return min(base + 4, 12)
    if usable_rows < max(1, target_rows // 2):
        return min(base + 3, 12)
    if usable_rows < target_rows:
        return min(base + 2, 10)
    return base


def _auto_learning_backfill_pair_limit(
    timeframe: str,
    df_events: pd.DataFrame | None = None,
    df_forward_windows: pd.DataFrame | None = None,
) -> int:
    tf = str(timeframe or "").strip().lower()
    base = int(_AUTO_TIMEFRAME_LEARNING_BACKFILL_PAIR_LIMITS.get(tf, 2))
    stats = _auto_learning_timeframe_stats(df_events, df_forward_windows).get(tf, {})
    usable_rows = int(stats.get("usable_rows") or 0)
    target_rows = int(_AUTO_TIMEFRAME_LEARNING_USABLE_TARGETS.get(tf, 24))
    if usable_rows <= 0:
        return min(base + 3, 6)
    if usable_rows < max(1, target_rows // 2):
        return min(base + 2, 5)
    if usable_rows < target_rows:
        return min(base + 1, 4)
    return base


def _auto_learning_open_symbol_candidates(
    df_events: pd.DataFrame | None,
    *,
    timeframe: str,
    exclude_stables: bool,
    limit: int,
) -> list[str]:
    if df_events is None or df_events.empty:
        return []
    required = {"symbol", "timeframe", "status"}
    if not required.issubset(df_events.columns):
        return []
    d = df_events.copy()
    d["__tf"] = d["timeframe"].fillna("").astype(str).str.strip().str.lower()
    d["__status"] = d["status"].fillna("").astype(str).str.strip().str.upper()
    d["__base"] = d["symbol"].fillna("").astype(str).map(_canonical_pair_base)
    d = d[
        d["__tf"].eq(str(timeframe or "").strip().lower())
        & d["__status"].eq("OPEN")
        & d["__base"].ne("")
    ].copy()
    if d.empty:
        return []
    if bool(exclude_stables):
        d = d[~d["__base"].map(is_stable_base_symbol)].copy()
    if d.empty:
        return []
    if "event_time" in d.columns:
        d["__event_ts"] = pd.to_datetime(d["event_time"], utc=True, errors="coerce")
        d = d.sort_values("__event_ts", ascending=False)
    bases: list[str] = []
    seen: set[str] = set()
    for base in d["__base"].tolist():
        base_text = str(base or "").strip().upper()
        if not base_text or base_text in seen:
            continue
        seen.add(base_text)
        bases.append(f"{base_text}/USDT")
        if len(bases) >= int(max(1, limit)):
            break
    return bases


def _auto_learning_symbol_candidates(
    *,
    df_events: pd.DataFrame | None,
    candidate_pairs: Sequence[str],
    timeframe: str,
    exclude_stables: bool,
    open_limit: int,
    total_limit: int,
) -> list[str]:
    symbols: list[str] = []
    seen: set[str] = set()

    def add_symbol(value: object) -> None:
        base = _canonical_pair_base(str(value or ""))
        if not base or base in seen:
            return
        if bool(exclude_stables) and is_stable_base_symbol(base):
            return
        seen.add(base)
        text = str(value or "").strip().upper()
        symbols.append(text if "/" in text else f"{base}/USDT")

    for symbol in _auto_learning_open_symbol_candidates(
        df_events,
        timeframe=timeframe,
        exclude_stables=exclude_stables,
        limit=open_limit,
    ):
        add_symbol(symbol)

    safe_total = max(int(total_limit or 1), int(open_limit or 1), 1)
    for pair in list(candidate_pairs or []):
        if len(symbols) >= safe_total:
            break
        add_symbol(pair)

    return symbols[:safe_total]


def _auto_timeframe_learning_event_from_frame(
    *,
    symbol: str,
    timeframe: str,
    df_eval: pd.DataFrame,
    analyse,
    ml_ensemble_predict,
    signal_plain,
    direction_key,
) -> dict[str, object] | None:
    if df_eval is None or df_eval.empty or len(df_eval) < 60:
        return None
    if "timestamp" not in df_eval.columns or "close" not in df_eval.columns:
        return None
    base = _canonical_pair_base(symbol)
    if not base or is_stable_base_symbol(base):
        return None
    latest = df_eval.iloc[-1]
    event_time = latest.get("timestamp")
    event_ts = _auto_learning_ts(event_time)
    if event_ts is None:
        return None
    price = _sortable_float(latest.get("close"))
    if price <= 0:
        return None
    prev_close = _sortable_float(df_eval["close"].iloc[-2]) if len(df_eval) >= 2 else 0.0
    delta_pct = ((price / prev_close) - 1.0) * 100.0 if prev_close > 0 else None

    analysis = analyse(df_eval)
    prob_up, ai_direction_raw, ai_details = ml_ensemble_predict(df_eval)
    bias = _sortable_float(getattr(analysis, "bias", 50.0))
    tech_direction = direction_key(signal_plain(getattr(analysis, "signal", "")))
    if tech_direction == "NEUTRAL":
        if bias >= 72.0:
            tech_direction = "UPSIDE"
        elif bias <= 28.0:
            tech_direction = "DOWNSIDE"
    ai_direction = direction_key(ai_direction_raw)
    tech_confidence = float(bias_confidence_from_bias(bias))
    try:
        prob = float(prob_up)
    except Exception:
        prob = 0.5
    ai_confidence = float(max(0.0, min(100.0, 50.0 + abs(prob - 0.5) * 100.0)))
    directional_agreement = 0.0
    if isinstance(ai_details, dict):
        directional_agreement = _sortable_float(ai_details.get("directional_agreement"))

    direction = "NEUTRAL"
    reason = ""
    confidence = 0.0
    if tech_direction in {"UPSIDE", "DOWNSIDE"} and tech_direction == ai_direction:
        direction = tech_direction
        confidence = min(100.0, max(tech_confidence, ai_confidence) + 5.0)
        reason = "AUTO_ALIGNED"
    elif tech_direction in {"UPSIDE", "DOWNSIDE"} and tech_confidence >= 62.0:
        direction = tech_direction
        confidence = tech_confidence
        reason = "AUTO_TECHNICAL"
    elif ai_direction in {"UPSIDE", "DOWNSIDE"} and ai_confidence >= 62.0 and directional_agreement >= 0.50:
        direction = ai_direction
        confidence = ai_confidence
        reason = "AUTO_AI"
    if direction not in {"UPSIDE", "DOWNSIDE"} or confidence < 55.0:
        return None

    adx_val = _sortable_float(getattr(analysis, "adx", 0.0))
    setup_confirm = "WATCH"
    if reason == "AUTO_ALIGNED" and confidence >= 78.0 and adx_val >= 20.0:
        setup_confirm = "ENTER_TREND_AI"
    elif confidence >= 66.0 and (adx_val >= 17.0 or reason == "AUTO_ALIGNED"):
        setup_confirm = "PROBE"

    return {
        "source": "Market",
        "decision_version": current_decision_version("Market"),
        "symbol": base,
        "timeframe": str(timeframe or "").strip().lower(),
        "event_time": event_ts.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "session_bucket": session_bucket_for_timestamp(event_ts),
        "scan_focus": _AUTO_TIMEFRAME_LEARNING_SCAN_FOCUS,
        "direction": direction,
        "setup_confirm": setup_confirm,
        "action_reason": reason,
        "lead_label": "Auto timeframe sweep",
        "lead_direction": direction,
        "confidence": confidence,
        "ai_ensemble": str(ai_direction_raw or "NEUTRAL"),
        "ai_direction": ai_direction,
        "ai_confidence": ai_confidence,
        "market_lead_label": "",
        "market_lead_score": None,
        "market_lead_upside": 0,
        "market_lead_downside": 0,
        "market_regime": "",
        "market_playbook_key": "",
        "market_playbook": "",
        "market_no_trade": False,
        "market_trade_gate_key": "",
        "market_trade_gate": "",
        "market_alert_keys": "",
        "market_primary_alert": "",
        "market_no_trade_reason": "",
        "risk_tier": "",
        "risk_unit_fraction": None,
        "sector_tag": "",
        "market_sector_rotation": "",
        "market_catalyst_state": "",
        "market_catalyst_event": "",
        "market_catalyst_blocking": False,
        "market_catalyst_category": "",
        "market_catalyst_scope": "",
        "market_catalyst_tag": "",
        "market_catalyst_targeted": False,
        "market_catalyst_window": "",
        "market_flow_state": "",
        "market_flow_bias": "",
        "adaptive_edge_label": "",
        "adaptive_edge_score": None,
        "actionable_frame_score": None,
        "actionable_setup_score": None,
        "actionable_context_score": None,
        "actionable_tactical_score": None,
        "archive_guardrail_label": "",
        "archive_guardrail_penalty": None,
        "archive_guardrail_note": "",
        "price": price,
        "delta_pct": delta_pct,
        "entry_price": None,
        "stop_loss": None,
        "target_price": None,
        "rr_ratio": None,
    }


def _run_auto_timeframe_learning_sweep(
    *,
    session_state,
    current_timeframe: str,
    exclude_stables: bool,
    get_top_volume_usdt_symbols,
    fetch_ohlcv,
    analyse,
    ml_ensemble_predict,
    signal_plain,
    direction_key,
    fetch_signal_events_df,
    log_signal_events,
    resolve_open_signal_events_for_frame,
    backfill_signal_forward_windows_via_fetch=None,
    fetch_signal_forward_windows_df=None,
    db_path: str,
    debug,
) -> dict[str, object]:
    state = _auto_learning_state(session_state.get(_AUTO_TIMEFRAME_LEARNING_STATE_KEY))
    if bool(state.get("running")):
        return {"ran": False, "reason": "already_running"}
    now_ts = pd.Timestamp.now(tz="UTC")
    last_global = _auto_learning_ts(state.get("last_attempt_at"))
    if last_global is not None:
        global_age = (now_ts - last_global).total_seconds()
        if global_age < _AUTO_TIMEFRAME_LEARNING_GLOBAL_COOLDOWN_SECONDS:
            session_state[_AUTO_TIMEFRAME_LEARNING_STATE_KEY] = state
            return {"ran": False, "reason": "not_due"}
    try:
        df_recent = fetch_signal_events_df(
            limit=ARCHIVE_LEARNING_WINDOW_ROWS,
            source="Market",
            decision_version=current_decision_version("Market"),
            db_path=db_path,
        )
    except Exception as e:
        if callable(debug):
            debug(f"Auto timeframe learning scope unavailable: {e.__class__.__name__}: {str(e).strip()}")
        return {"ran": False, "reason": "scope_unavailable"}

    df_forward_windows = pd.DataFrame()
    if callable(fetch_signal_forward_windows_df) and isinstance(df_recent, pd.DataFrame) and "signal_key" in df_recent.columns:
        signal_keys = [
            str(value).strip()
            for value in df_recent["signal_key"].dropna().tolist()
            if str(value).strip()
        ]
        if signal_keys:
            df_forward_windows = pd.DataFrame(columns=["signal_key"])
            try:
                fetched_forward_windows = fetch_signal_forward_windows_df(
                    signal_keys=list(dict.fromkeys(signal_keys)),
                    limit=ARCHIVE_LEARNING_WINDOW_ROWS,
                    db_path=db_path,
                )
                if isinstance(fetched_forward_windows, pd.DataFrame):
                    df_forward_windows = fetched_forward_windows.copy()
                    if "signal_key" not in df_forward_windows.columns:
                        df_forward_windows["signal_key"] = pd.Series(dtype=object)
            except Exception as e:
                df_forward_windows = pd.DataFrame()
                if callable(debug):
                    debug(f"Auto timeframe learning checkpoint scope unavailable: {e.__class__.__name__}: {str(e).strip()}")

    timeframes = _select_auto_learning_timeframes(
        df_recent,
        state,
        df_forward_windows=df_forward_windows,
        now=now_ts,
        current_timeframe=current_timeframe,
    )
    if not timeframes:
        session_state[_AUTO_TIMEFRAME_LEARNING_STATE_KEY] = state
        return {"ran": False, "reason": "not_due"}

    state["running"] = True
    session_state[_AUTO_TIMEFRAME_LEARNING_STATE_KEY] = state
    total_written = 0
    total_resolved = 0
    total_backfilled = 0
    total_errors = 0
    per_timeframe: list[dict[str, object]] = []
    try:
        try:
            candidate_pairs, _market_rows = get_top_volume_usdt_symbols(_AUTO_TIMEFRAME_LEARNING_FETCH_N)
            universe_errors = 0
        except Exception as e:
            universe_errors = 1
            if callable(debug):
                debug(f"Auto timeframe learning universe failed: {e.__class__.__name__}: {str(e).strip()}")
            candidate_pairs = []

        for timeframe in timeframes:
            state = _mark_auto_learning_attempt(state, timeframe=timeframe, now=now_ts)
            state["running"] = True
            session_state[_AUTO_TIMEFRAME_LEARNING_STATE_KEY] = state
            written = 0
            resolved = 0
            backfilled = 0
            errors = int(universe_errors)
            candle_limit = _auto_learning_candle_limit(timeframe)
            symbols = _auto_learning_symbol_candidates(
                df_events=df_recent,
                candidate_pairs=list(candidate_pairs or []),
                timeframe=timeframe,
                exclude_stables=bool(exclude_stables),
                open_limit=int(_AUTO_TIMEFRAME_LEARNING_OPEN_SYMBOL_LIMITS.get(timeframe, 6)),
                total_limit=_auto_learning_symbol_limit(timeframe, df_recent, df_forward_windows),
            )

            events: list[dict[str, object]] = []
            fetch_lock = Lock()
            for symbol in symbols:
                try:
                    with fetch_lock:
                        df = fetch_ohlcv(symbol, timeframe, limit=candle_limit)
                except Exception as e:
                    errors += 1
                    if callable(debug):
                        debug(f"Auto timeframe learning OHLCV failed for {symbol} ({timeframe}): {e.__class__.__name__}: {str(e).strip()}")
                    continue
                df_eval = _prepare_closed_frame(df, min_rows=60)
                if df_eval is None:
                    continue
                base = _canonical_pair_base(symbol)
                try:
                    resolved += int(
                        resolve_open_signal_events_for_frame(
                            symbol=base,
                            timeframe=timeframe,
                            df_ohlcv=df_eval,
                            source="Market",
                            db_path=db_path,
                        )
                    )
                    resolved += int(
                        resolve_open_signal_events_for_frame(
                            symbol=base,
                            timeframe=timeframe,
                            df_ohlcv=df_eval,
                            source="Scalp",
                            db_path=db_path,
                        )
                    )
                except Exception as e:
                    errors += 1
                    if callable(debug):
                        debug(f"Auto timeframe learning resolve failed for {base} ({timeframe}): {e.__class__.__name__}: {str(e).strip()}")
                try:
                    event = _auto_timeframe_learning_event_from_frame(
                        symbol=symbol,
                        timeframe=timeframe,
                        df_eval=df_eval,
                        analyse=analyse,
                        ml_ensemble_predict=ml_ensemble_predict,
                        signal_plain=signal_plain,
                        direction_key=direction_key,
                    )
                except Exception as e:
                    errors += 1
                    if callable(debug):
                        debug(f"Auto timeframe learning event failed for {base} ({timeframe}): {e.__class__.__name__}: {str(e).strip()}")
                    event = None
                if event:
                    events.append(event)
            if events:
                written = int(log_signal_events(events, db_path=db_path))
            if callable(backfill_signal_forward_windows_via_fetch):
                try:
                    backfilled = int(
                        backfill_signal_forward_windows_via_fetch(
                            fetch_ohlcv=fetch_ohlcv,
                            source="Market",
                            db_path=db_path,
                            limit_pairs=_auto_learning_backfill_pair_limit(timeframe, df_recent, df_forward_windows),
                            rows_per_pair=40,
                            candle_limit=candle_limit,
                            timeframe=timeframe,
                            decision_version=current_decision_version("Market"),
                        )
                    )
                except Exception as e:
                    errors += 1
                    if callable(debug):
                        debug(f"Auto timeframe learning window backfill failed ({timeframe}): {e.__class__.__name__}: {str(e).strip()}")
            state = _mark_auto_learning_attempt(
                state,
                timeframe=timeframe,
                now=now_ts,
                written=written,
                resolved=resolved,
                backfilled=backfilled,
                errors=errors,
            )
            state["running"] = True
            session_state[_AUTO_TIMEFRAME_LEARNING_STATE_KEY] = state
            total_written += int(written)
            total_resolved += int(resolved)
            total_backfilled += int(backfilled)
            total_errors += int(errors)
            per_timeframe.append(
                {
                    "timeframe": timeframe,
                    "written": int(written),
                    "resolved": int(resolved),
                    "backfilled": int(backfilled),
                    "errors": int(errors),
                    "symbols": len(symbols),
                }
            )
            try:
                if callable(debug):
                    debug(
                        f"Auto timeframe learning sweep {timeframe}: "
                        f"written={written}, resolved={resolved}, backfilled={backfilled}, errors={errors}."
                    )
            except Exception:
                pass
    finally:
        state["running"] = False
        session_state[_AUTO_TIMEFRAME_LEARNING_STATE_KEY] = state
    return {
        "ran": True,
        "timeframe": timeframes[0] if timeframes else "",
        "timeframes": timeframes,
        "written": total_written,
        "resolved": total_resolved,
        "backfilled": total_backfilled,
        "errors": total_errors,
        "results": per_timeframe,
    }


def _direction_fetch_symbol(symbol: str, actual_symbol: str, source_provider: str) -> str:
    # Keep HTF Direction/AI fetches anchored to the canonical requested symbol.
    # If we inherit the selected-timeframe provider/variant here, the visible
    # HTF columns can drift when the user changes timeframe and the selected
    # candle fetch resolves through a different exchange/provider path.
    return str(symbol or actual_symbol or "").strip()


def _confidence_badge(score: float) -> str:
    score_f = _sortable_float(score)
    return f"{score_f:.0f}% ({confidence_bucket(score_f).title()})"


def _ai_confidence_badge(snapshot, score: float) -> str:
    score_f = _sortable_float(score)
    dots = ai_spot_bias_display_votes(snapshot)
    label = ai_confidence_bucket(
        score_f,
        direction=str(snapshot.direction or ""),
        support_votes=int(dots),
        timeframe_conflict=bool(snapshot.timeframe_conflict),
        degraded_data=bool(snapshot.degraded_data),
    )
    return f"{score_f:.0f}% ({label.title()})"


def _coingecko_coin_id_fallback_available(fetcher: object) -> bool:
    return callable(fetcher) and not bool(getattr(fetcher, "_codex_missing_dep", False))


def _coingecko_coin_id_fallback_reason(fetcher: object) -> str:
    if callable(fetcher) and not bool(getattr(fetcher, "_codex_missing_dep", False)):
        return ""
    detail = str(getattr(fetcher, "_codex_missing_dep_reason", "") or "").strip()
    if detail:
        return detail
    if callable(fetcher):
        return "dependency marked unavailable in this session"
    return "backup fetcher is not callable"


def _coingecko_coin_id_unavailable_message(reason: str | None) -> str:
    detail = str(reason or "").strip()
    if detail:
        return f"no exchange OHLCV data; CoinGecko backup unavailable ({detail})"
    return "no exchange OHLCV data; CoinGecko backup unavailable"


def _audit_scan_summary_lines(
    *,
    displayed_rows: int,
    attempted_count: int,
    produced_count: int,
    skipped_count: int,
    ranked_out_count: int,
    source_label: str,
    scan_mode: str = _SCAN_MODE_BROAD,
    timeframe: str = "1h",
    direction_filter: str = "Both",
) -> list[str]:
    lines = [f"**Rows shown:** `{int(max(0, displayed_rows))}`"]
    if int(max(0, attempted_count)) > 0:
        summary = (
            f"**Live read attempt:** attempted `{int(max(0, attempted_count))}`"
            f" • produced `{int(max(0, produced_count))}`"
            f" • skipped `{int(max(0, skipped_count))}`"
        )
        if int(max(0, ranked_out_count)) > 0:
            summary += f" • ranked out `{int(max(0, ranked_out_count))}`"
        lines.append(summary)
    lines.append(
        f"**Scan mode:** `{_normalize_scan_mode(scan_mode)}` • **Timeframe:** `{str(timeframe).upper()}` • **Direction:** `{direction_filter}`"
    )
    if str(source_label or "").strip().upper().startswith("CACHED"):
        lines.append("_Current table is cached. Live attempt stats reflect the latest refresh attempt, not the cached rows._")
    return lines


def _scanner_trace_diagnostic_lines(summary: dict[str, object]) -> list[str]:
    total = int(summary.get("total_rows") or 0)
    if total <= 0:
        return ["- Coverage trace has no rows yet for this scope. Run a fresh scan to start collecting it."]
    scans = int(summary.get("scan_count") or 0)
    symbols = int(summary.get("symbol_count") or 0)
    lines = [
        (
            f"- Trace rows: **{total}** across **{scans}** recent scan(s), "
            f"covering **{symbols}** symbol(s). Shown rate: **{float(summary.get('shown_rate_pct') or 0.0):.1f}%**."
        ),
        (
            f"- Stages: shown **{int(summary.get('shown_count') or 0)}**"
            f" • ranked out **{int(summary.get('ranked_out_count') or 0)}**"
            f" • filtered out **{int(summary.get('filtered_out_count') or 0)}**"
            f" • skipped **{int(summary.get('skipped_count') or 0)}**"
            f" • not scanned this pass **{int(summary.get('candidate_only_count') or 0)}**."
        ),
    ]
    top_ranked_out = list(summary.get("top_ranked_out") or [])
    if top_ranked_out:
        formatted = []
        for row in top_ranked_out[:5]:
            symbol = str(row.get("symbol") or "").strip()
            score = float(row.get("score") or 0.0)
            rank = row.get("candidate_rank")
            rank_text = f"#{int(rank)}" if rank is not None else "unranked"
            formatted.append(f"{symbol} ({score:.1f}, {rank_text})")
        lines.append(f"- Strongest Top N cuts: {', '.join(formatted)}.")
    top_filtered_out = list(summary.get("top_filtered_out") or [])
    if top_filtered_out:
        formatted = []
        for row in top_filtered_out[:5]:
            symbol = str(row.get("symbol") or "").strip()
            score = float(row.get("score") or 0.0)
            formatted.append(f"{symbol} ({score:.1f})")
        lines.append(f"- Strongest gate-filtered candidates: {', '.join(formatted)}.")
    top_skip_reasons = list(summary.get("top_skip_reasons") or [])
    if top_skip_reasons:
        formatted = [
            f"{str(row.get('reason') or 'unknown')} ({int(row.get('count') or 0)})"
            for row in top_skip_reasons[:5]
        ]
        lines.append(f"- Top skip reasons: {', '.join(formatted)}.")
    return lines


def _archive_learning_rows(
    *,
    visible_rows: list[dict],
    produced_rows: list[dict],
) -> list[dict]:
    source_rows = list(produced_rows or []) if produced_rows else list(visible_rows or [])
    out: list[dict] = []
    seen: set[tuple[object, ...]] = set()
    for idx, row in enumerate(source_rows):
        if not isinstance(row, dict):
            continue
        key = (
            str(row.get("Coin") or "").strip().upper(),
            str(row.get("__timeframe") or "").strip(),
            str(row.get("__event_time") or "").strip(),
        )
        if not any(key):
            key = ("__idx__", idx)
        if key in seen:
            continue
        seen.add(key)
        out.append(row)
    return out


def _scanner_trace_base(value: object) -> str:
    text = str(value or "").strip().upper()
    if not text:
        return ""
    if "(" in text:
        text = text.split("(", 1)[0].strip()
    if "/" in text:
        text = text.split("/", 1)[0].strip()
    for separator in ("-", "_", " "):
        if separator in text:
            text = text.split(separator, 1)[0].strip()
            break
    for suffix in ("USDT", "USDC", "FDUSD", "BUSD", "USD"):
        if text.endswith(suffix) and len(text) > len(suffix) + 1:
            text = text[: -len(suffix)].strip()
            break
    text = re.sub(r"[^A-Z0-9]", "", text)
    return canonical_base_symbol(text)


def _scanner_trace_row_base(row: dict) -> str:
    for key in ("Coin", "symbol", "base", "pair"):
        base = _scanner_trace_base(row.get(key))
        if base:
            return base
    return ""


def _scanner_trace_events(
    *,
    candidate_symbols: list[str],
    attempted_symbols: set[str],
    skipped_symbols: list[tuple[str, str]],
    produced_rows: list[dict],
    visible_rows: list[dict],
    market_rows: list[dict],
    timeframe: str,
    scan_mode: str,
    direction_filter: str,
    observed_at: object,
    source_label: str,
    data_mode: str,
) -> list[dict[str, object]]:
    """Build passive scanner-stage audit rows without changing scanner behavior."""

    def rows_by_base(rows: list[dict]) -> dict[str, dict]:
        out: dict[str, dict] = {}
        for row in rows or []:
            if not isinstance(row, dict):
                continue
            base = _scanner_trace_row_base(row)
            if base and base not in out:
                out[base] = row
        return out

    candidate_pairs: dict[str, str] = {}
    candidate_rank: dict[str, int] = {}
    ordered_bases: list[str] = []
    for idx, symbol in enumerate(candidate_symbols or [], start=1):
        base = _scanner_trace_base(symbol)
        if not base:
            continue
        if base not in candidate_rank:
            candidate_rank[base] = idx
            candidate_pairs[base] = str(symbol or "").strip()
            ordered_bases.append(base)

    attempted_bases = {
        base
        for base in (_scanner_trace_base(symbol) for symbol in attempted_symbols or set())
        if base
    }
    skipped_by_base: dict[str, str] = {}
    for symbol, reason in skipped_symbols or []:
        base = _scanner_trace_base(symbol)
        if base and base not in skipped_by_base:
            skipped_by_base[base] = str(reason or "").strip()

    produced_by_base = rows_by_base(produced_rows or [])
    visible_by_base = rows_by_base(visible_rows or [])
    market_by_base = rows_by_base(market_rows or [])
    shown_rank = {
        _scanner_trace_row_base(row): idx
        for idx, row in enumerate(visible_rows or [], start=1)
        if isinstance(row, dict) and _scanner_trace_row_base(row)
    }

    for base in [*produced_by_base.keys(), *visible_by_base.keys(), *skipped_by_base.keys(), *attempted_bases]:
        if base and base not in candidate_rank and base not in ordered_bases:
            ordered_bases.append(base)

    scan_focus = _normalize_scan_mode(scan_mode)
    tf = str(timeframe or "").strip().lower()
    direction_text = str(direction_filter or "Both").strip() or "Both"
    scan_id = f"market|{scan_focus}|{tf}|{direction_text}|{pd.to_datetime(observed_at, utc=True, errors='coerce')}"
    events: list[dict[str, object]] = []
    for base in ordered_bases:
        produced = base in produced_by_base
        shown = base in visible_by_base
        skipped = base in skipped_by_base
        attempted = base in attempted_bases or produced or shown or skipped
        if shown:
            stage = "shown"
            reason = "shown"
        elif produced:
            stage = "ranked_out"
            reason = "top_n_cut"
        elif skipped:
            stage = "skipped"
            reason = skipped_by_base.get(base, "")
        elif attempted:
            stage = "filtered_out"
            reason = "direction_or_quality_gate"
        else:
            stage = "candidate"
            reason = "not_scanned_this_pass"

        row = produced_by_base.get(base) or visible_by_base.get(base) or market_by_base.get(base) or {}
        events.append(
            {
                "scan_id": scan_id,
                "observed_at": observed_at,
                "source": "Market",
                "scan_focus": scan_focus,
                "timeframe": tf,
                "direction_filter": direction_text,
                "symbol": base,
                "pair": candidate_pairs.get(base) or str(row.get("pair") or row.get("symbol") or ""),
                "stage": stage,
                "reason": reason,
                "candidate_rank": candidate_rank.get(base),
                "shown_rank": shown_rank.get(base),
                "attempted": attempted,
                "produced": produced,
                "shown": shown,
                "skipped": skipped,
                "setup_confirm": str(row.get("__action_raw", row.get("Setup Confirm", "")) or ""),
                "direction": str(row.get("Direction") or row.get("__signal_direction_raw") or ""),
                "confidence": row.get("__confidence_val", _confidence_value_from_badge(row.get("Confidence"))),
                "ai_confidence": row.get("__ai_confidence_val", _confidence_value_from_badge(row.get("AI Confidence"))),
                "radar_source_kind": row.get("_radar_source_kind") or row.get("radar_source_kind"),
                "radar_source_score": row.get("_radar_source_score") or row.get("radar_source_score"),
                "radar_freshness_score": row.get("_radar_freshness_score") or row.get("radar_freshness_score"),
                "emerging_rank_score": row.get("__emerging_rank_score") or row.get("_emerging_rank_score"),
                "actionable_frame_score": row.get("__actionable_frame_score"),
                "actionable_tactical_score": row.get("__actionable_tactical_score"),
                "market_cap": row.get("__mcap_val") or row.get("market_cap"),
                "price": row.get("__price_val") or row.get("current_price") or row.get("price"),
                "delta_pct": row.get("__delta_pct") or row.get("price_change_percentage_24h"),
                "data_mode": data_mode,
                "source_label": source_label,
            }
        )
    return events


def _spot_bias_label(direction: str) -> str:
    return _shared_spot_bias_label(direction)


def _spot_tf_summary(snapshot) -> str:
    return (
        f"{str(snapshot.timeframe).upper()}: { _spot_bias_label(snapshot.direction)} | "
        f"Structure {snapshot.structure_label} | "
        f"Trend {float(snapshot.trend_score):.0f} | "
        f"Regime {snapshot.regime_label}"
    )


def _spot_lead_snapshot(snapshot):
    return getattr(snapshot, "lead_snapshot", getattr(snapshot, "one_day", None))


def _spot_confirm_snapshot(snapshot):
    return getattr(snapshot, "confirm_snapshot", getattr(snapshot, "four_hour", None))


def _spot_anchor_pair_label(snapshot) -> str:
    label = str(getattr(snapshot, "anchor_pair_label", "") or "").strip()
    if label:
        return label
    lead = _spot_lead_snapshot(snapshot)
    confirm = _spot_confirm_snapshot(snapshot)
    if lead is None or confirm is None:
        return "Higher-TF"
    return f"{str(lead.timeframe).upper()} + {str(confirm.timeframe).upper()}"


def _spot_direction_note(
    snapshot,
    *,
    selected_timeframe: str,
    tactical_direction: str,
    tactical_signal: str,
    tactical_bias: float,
    tactical_comment: str,
) -> str:
    lead = _spot_lead_snapshot(snapshot)
    confirm = _spot_confirm_snapshot(snapshot)
    lead_summary = _spot_tf_summary(lead)
    confirm_summary = _spot_tf_summary(confirm)
    tactical = (
        f"Selected {str(selected_timeframe).upper()}: "
        f"{_spot_bias_label(tactical_direction)} | {tactical_signal}"
    )
    summary = str(snapshot.note or "").strip()
    note = (
        f"Higher-timeframe direction from {_spot_anchor_pair_label(snapshot)} closed anchors: "
        f"{_spot_bias_label(snapshot.direction)}. "
        f"{summary} "
        f"{lead_summary}. {confirm_summary}. {tactical}."
    )
    comment = str(tactical_comment or "").strip()
    if comment:
        note += f" Local read: {comment}."
    return note.strip()


def _confidence_note(snapshot, score: float, confidence_snapshot=None) -> str:
    caps: list[str] = []
    if str(snapshot.direction or "").strip().upper() == "NEUTRAL":
        caps.append("neutral bias")
    if bool(snapshot.timeframe_conflict):
        caps.append("timeframe conflict")
    if float(snapshot.structure_quality) < 40.0:
        caps.append("weak structure")
    if bool(snapshot.degraded_data):
        caps.append("partial data")
    if bool(snapshot.range_regime):
        caps.append("range regime")
    cap_text = f" Limits active: {', '.join(caps)}." if caps else ""
    note = (
        f"How strong the Direction call is: {float(score):.1f}% ({confidence_bucket(score).title()}). "
        f"Built from alignment, structure, trend, regime, and location quality."
        f"{cap_text}"
    )
    calibration_note = str(getattr(confidence_snapshot, "note", "") or "").strip()
    if calibration_note:
        note = f"{note} {_compact_hover_note(calibration_note)}".strip()
    return note


def _ai_spot_tf_summary(snapshot) -> str:
    status = str(getattr(snapshot, "status", "") or "").strip()
    note = str(getattr(snapshot, "note", "") or "").strip()
    suffix_parts = []
    if status:
        suffix_parts.append(status)
    if note:
        suffix_parts.append(note)
    suffix = f" | {' | '.join(suffix_parts)}" if suffix_parts else ""
    return (
        f"{str(snapshot.timeframe).upper()}: {_spot_bias_label(snapshot.direction)} | "
        f"Up probability {float(snapshot.probability_up) * 100:.0f}%{suffix}"
    )


def _ai_spot_bias_note(snapshot) -> str:
    dots = ai_spot_bias_display_votes(snapshot)
    return (
        f"AI view from {_spot_anchor_pair_label(snapshot)}: {_spot_bias_label(snapshot.direction)}. "
        f"Model support: {dots}/3 dots. "
        f"{str(snapshot.note or '').strip()} "
        f"{_ai_spot_tf_summary(_spot_lead_snapshot(snapshot))}. "
        f"{_ai_spot_tf_summary(_spot_confirm_snapshot(snapshot))}."
    )


def _ai_confidence_note(snapshot, score: float, confidence_snapshot=None) -> str:
    dots = ai_spot_bias_display_votes(snapshot)
    caps: list[str] = []
    if str(snapshot.direction or "").strip().upper() == "NEUTRAL":
        caps.append("neutral AI verdict")
    if bool(snapshot.timeframe_conflict):
        caps.append("timeframe conflict")
    if bool(snapshot.degraded_data):
        caps.append("partial data")
    if str(snapshot.direction or "").strip().upper() != "NEUTRAL" and int(dots) <= 1:
        caps.append("low model support")
    cap_text = f" Limits active: {', '.join(caps)}." if caps else ""
    note = (
        f"How trustworthy the AI verdict is: {float(score):.1f}% "
        f"({ai_confidence_bucket(float(score), direction=str(snapshot.direction or ''), support_votes=int(dots), timeframe_conflict=bool(snapshot.timeframe_conflict), degraded_data=bool(snapshot.degraded_data)).title()}). "
        f"AI side: {_spot_bias_label(snapshot.direction)}. Model support: {int(dots)}/3."
        f"{cap_text}"
    )
    calibration_note = str(getattr(confidence_snapshot, "note", "") or "").strip()
    if calibration_note:
        note = f"{note} {_compact_hover_note(calibration_note)}".strip()
    return note


def _cache_is_fresh(cache_ts: str | None, ttl_minutes: int) -> bool:
    ts_parsed = pd.to_datetime(cache_ts, utc=True, errors="coerce")
    if pd.isna(ts_parsed):
        return False
    try:
        age_minutes = (pd.Timestamp.now(tz="UTC") - ts_parsed).total_seconds() / 60.0
    except Exception:
        return False
    return age_minutes <= float(ttl_minutes)


def _scan_attempt_is_stale(scan_ts: str | None, ttl_minutes: int) -> bool:
    if not scan_ts:
        return True
    return not _cache_is_fresh(scan_ts, ttl_minutes)


def _scan_sig_key(scan_sig) -> tuple:
    if isinstance(scan_sig, tuple):
        return scan_sig
    if isinstance(scan_sig, list):
        return tuple(scan_sig)
    return (scan_sig,)


def _last_good_registry(
    raw,
    *,
    legacy_sig=None,
    legacy_results: list[dict] | None = None,
    legacy_ts: str | None = None,
    legacy_mode: str | None = None,
) -> dict[tuple, dict[str, object]]:
    out: dict[tuple, dict[str, object]] = {}
    if isinstance(raw, dict):
        for key, value in raw.items():
            if not isinstance(value, dict):
                continue
            results = value.get("results")
            if not isinstance(results, list) or not results:
                continue
            out[_scan_sig_key(key)] = {
                "results": list(results),
                "ts": str(value.get("ts") or ""),
                "mode": str(value.get("mode") or ""),
            }

    if legacy_sig and isinstance(legacy_results, list) and legacy_results:
        sig_key = _scan_sig_key(legacy_sig)
        if sig_key not in out:
            out[sig_key] = {
                "results": list(legacy_results),
                "ts": str(legacy_ts or ""),
                "mode": str(legacy_mode or ""),
            }
    return out


def _last_good_snapshot_for_sig(registry: dict[tuple, dict[str, object]], scan_sig):
    return registry.get(_scan_sig_key(scan_sig))


def _remember_last_good_snapshot(
    registry: dict[tuple, dict[str, object]],
    scan_sig,
    results: list[dict],
    ts: str,
    mode: str,
) -> dict[tuple, dict[str, object]]:
    out = dict(registry) if isinstance(registry, dict) else {}
    key = _scan_sig_key(scan_sig)
    out.pop(key, None)
    out[key] = {
        "results": list(results),
        "ts": str(ts or ""),
        "mode": str(mode or ""),
    }
    while len(out) > _LAST_GOOD_HISTORY_LIMIT:
        oldest_key = next(iter(out))
        out.pop(oldest_key, None)
    return out


def _healthy_empty_registry(raw) -> dict[tuple, str]:
    if not isinstance(raw, dict):
        return {}
    out: dict[tuple, str] = {}
    for key, value in raw.items():
        out[_scan_sig_key(key)] = str(value)
    return out


def _healthy_empty_seen_for_sig(registry: dict[tuple, str], scan_sig) -> bool:
    return _scan_sig_key(scan_sig) in registry


def _remember_healthy_empty_sig(registry: dict[tuple, str], scan_sig) -> dict[tuple, str]:
    out = dict(registry) if isinstance(registry, dict) else {}
    key = _scan_sig_key(scan_sig)
    if key in out:
        out.pop(key, None)
    out[key] = pd.Timestamp.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")
    while len(out) > _HEALTHY_EMPTY_HISTORY_LIMIT:
        oldest_key = next(iter(out))
        out.pop(oldest_key, None)
    return out


def _clear_healthy_empty_sig(registry: dict[tuple, str], scan_sig) -> dict[tuple, str]:
    out = dict(registry) if isinstance(registry, dict) else {}
    out.pop(_scan_sig_key(scan_sig), None)
    return out


def _normalize_custom_bases(custom_bases: list[str] | tuple[str, ...], *, sort_output: bool = False) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for raw_symbol in custom_bases:
        raw = str(raw_symbol or "").strip().upper()
        if not raw:
            continue
        base = raw.split("/", 1)[0].strip() if "/" in raw else raw
        symbol = canonical_base_symbol(base)
        if not symbol or symbol in seen:
            continue
        seen.add(symbol)
        out.append(symbol)
    if sort_output:
        out = sorted(out)
    return out


def _parse_market_custom_bases(raw: str, limit: int = 10) -> list[str]:
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
        canonical = canonical_base_symbol(t)
        if not canonical or canonical in seen:
            continue
        seen.add(canonical)
        out.append(canonical)
        if len(out) >= int(limit):
            break
    return out


def _apply_market_custom_input_state(
    session_state: dict,
    *,
    raw_value: str | None = None,
    limit: int = 10,
) -> list[str]:
    raw = session_state.get("market_custom_coin_input", "") if raw_value is None else raw_value
    parsed = _parse_market_custom_bases(str(raw or ""), limit=limit)
    session_state["market_custom_bases_applied"] = parsed
    return parsed


def _market_scan_signature(
    *,
    timeframe: str,
    direction_filter: str,
    top_n: int,
    exclude_stables: bool,
    custom_bases_applied: list[str],
    scan_mode: str = _SCAN_MODE_BROAD,
) -> tuple:
    custom_tuple = tuple(_normalize_custom_bases(custom_bases_applied, sort_output=True))
    effective_top_n = 0 if custom_tuple else int(top_n)
    effective_scan_mode = _SCAN_MODE_BROAD if custom_tuple else _normalize_scan_mode(scan_mode)
    return (
        timeframe,
        direction_filter,
        effective_top_n,
        bool(exclude_stables),
        custom_tuple,
        effective_scan_mode,
    )


def _source_requires_immediate_retry(source_label: str | None) -> bool:
    source = str(source_label or "").strip().upper()
    if not source:
        return False
    return source.startswith("CACHED") or "DEGRADED" in source


def _recovery_retry_ttl_minutes(source_label: str | None) -> float | None:
    if not _source_requires_immediate_retry(source_label):
        return None
    return float(_RECOVERY_RETRY_BACKOFF_SECONDS) / 60.0


def _should_rescan_market(
    *,
    run_scan: bool,
    last_sig,
    scan_sig,
    has_results_state: bool,
    last_attempt_ts: str | None,
    refresh_ttl_minutes: int,
    current_source_label: str | None = None,
) -> bool:
    if run_scan:
        return True
    if not has_results_state:
        return True
    if _scan_sig_key(last_sig) != _scan_sig_key(scan_sig):
        return True
    recovery_ttl = _recovery_retry_ttl_minutes(current_source_label)
    if recovery_ttl is not None:
        return _scan_attempt_is_stale(last_attempt_ts, recovery_ttl)
    return _scan_attempt_is_stale(last_attempt_ts, refresh_ttl_minutes)


def _should_use_cached_scan(
    *,
    prev_results: list[dict],
    cache_sig,
    scan_sig,
    cache_ts: str | None,
    ttl_minutes: int,
    scan_degraded: bool,
    healthy_empty_seen: bool = False,
) -> bool:
    if not scan_degraded:
        return False
    if not prev_results:
        return False
    if healthy_empty_seen:
        return False
    if not cache_sig or tuple(cache_sig) != tuple(scan_sig):
        return False
    return _cache_is_fresh(cache_ts, ttl_minutes)


def _should_use_major_fallback(
    *,
    working_symbols: list[str],
    custom_mode_active: bool,
    source_pair_count: int,
    market_row_count: int,
) -> bool:
    if custom_mode_active:
        return False
    if working_symbols:
        return False
    return int(source_pair_count) == 0 and int(market_row_count) == 0


def _pair_provenance_label(requested_symbol: str, actual_symbol: str | None, provider: str | None) -> str:
    label = str(actual_symbol or requested_symbol or "").strip()
    if not label:
        label = str(requested_symbol or "").strip()
    if str(provider or "").strip().lower() == "coingecko":
        return f"{label} (CoinGecko backup)"
    return label


def _custom_watchlist_enrichment_coverage(
    working_symbols: list[str],
    mcap_map: dict[str, int],
) -> tuple[int, int]:
    working_bases: list[str] = []
    seen: set[str] = set()
    for symbol in working_symbols:
        base = _canonical_pair_base(symbol)
        if not base or base in seen:
            continue
        seen.add(base)
        working_bases.append(base)
    total = len(working_bases)
    enriched = sum(1 for base in working_bases if int(mcap_map.get(base) or 0) > 0)
    return enriched, total


def _market_data_mode(
    *,
    has_market_rows: bool,
    used_major_fallback: bool,
    custom_mode_active: bool = False,
    custom_watchlist_enriched_count: int = 0,
    custom_watchlist_total_count: int = 0,
) -> str:
    if custom_mode_active:
        if custom_watchlist_total_count > 0:
            if custom_watchlist_enriched_count <= 0:
                return "CUSTOM WATCHLIST MODE (EXCHANGE-ONLY)"
            if custom_watchlist_enriched_count < custom_watchlist_total_count:
                return "CUSTOM WATCHLIST MODE (PARTIAL ENRICHMENT)"
        return "CUSTOM WATCHLIST MODE" if has_market_rows else "CUSTOM WATCHLIST MODE (EXCHANGE-ONLY)"
    if used_major_fallback:
        return "MAJOR BACKUP MODE"
    return "FULL MARKET MODE" if has_market_rows else "EXCHANGE-ONLY MODE"


def _market_data_mode_display(mode_label: str) -> str:
    mode_up = str(mode_label or "").strip().upper()
    if mode_up == "FULL MARKET MODE":
        return "Full Market Data"
    if mode_up == "CUSTOM WATCHLIST MODE":
        return "Watchlist Data"
    if mode_up == "CUSTOM WATCHLIST MODE (PARTIAL ENRICHMENT)":
        return "Watchlist Data (Partial)"
    if mode_up == "CUSTOM WATCHLIST MODE (EXCHANGE-ONLY)":
        return "Watchlist Data (Exchange Only)"
    if mode_up == "MAJOR BACKUP MODE":
        return "Major Backup Data"
    if mode_up == "EXCHANGE-ONLY MODE":
        return "Exchange-Only Data"
    return str(mode_label or "").strip() or "Data Mode"


def _underfilled_universe_message(
    *,
    custom_mode_active: bool,
    used_major_fallback: bool,
    has_market_rows: bool,
    working_count: int,
    requested_n: int,
) -> str:
    if custom_mode_active:
        return f"Custom mode active: reading {working_count} / {requested_n} requested symbols."
    if used_major_fallback:
        return (
            f"Hardcoded major backup universe currently returned {working_count} eligible symbols "
            f"(requested {requested_n}). This is not a full live top-volume market sweep."
        )
    if has_market_rows:
        return (
            f"Liquidity universe currently returned {working_count} eligible symbols "
            f"(requested {requested_n}). Market read remains strict to top-volume matched pairs."
        )
    return (
        f"Exchange-only universe currently returned {working_count} eligible symbols "
        f"(requested {requested_n}). Market read is using exchange-ranked pairs because provider enrichment is unavailable."
    )


def _scan_universe_notice(
    *,
    candidate_count: int,
    requested_n: int,
    custom_mode_active: bool,
    used_major_fallback: bool,
    has_market_rows: bool,
    source_pair_count: int = 0,
    market_row_count: int = 0,
    top_n: int = 0,
) -> tuple[str, str] | None:
    if int(candidate_count) > 0:
        if int(candidate_count) < int(requested_n):
            return (
                "info",
                _underfilled_universe_message(
                    custom_mode_active=custom_mode_active,
                    used_major_fallback=used_major_fallback,
                    has_market_rows=has_market_rows,
                    working_count=int(candidate_count),
                    requested_n=requested_n,
                ),
            )
        return None
    if custom_mode_active:
        return (
            "warning",
            "Custom watchlist did not produce eligible symbols after normalization/filtering. "
            "Check symbol spelling (e.g., BTC, ETH, SOL) or disable stablecoin exclusion.",
        )
    if int(market_row_count) > 0 and int(source_pair_count) == 0:
        return (
            "warning",
            "Provider liquidity universe was available, but strict exchange pair ranking could not resolve "
            "usable USD/USDT feeds. Hardcoded major backup was intentionally not used.",
        )
    return (
        "info",
        "Liquidity universe was available, but current filters left no eligible market symbols. "
        "Hardcoded major backup was intentionally not used. "
        f"Source pairs: {int(source_pair_count)}, market rows: {int(market_row_count)}, "
        f"requested top_n: {int(top_n)}.",
    )


def _dedupe_market_rows(market_data: list[dict]) -> list[dict]:
    seen_symbols: set[str] = set()
    unique_market_data: list[dict] = []
    for coin in market_data:
        coin_id = (coin.get("id") or "").lower()
        symbol = (coin.get("symbol") or "").upper()
        if not symbol:
            continue
        if "wrapped" in coin_id:
            continue
        if symbol in seen_symbols:
            continue
        seen_symbols.add(symbol)
        unique_market_data.append(coin)
    return unique_market_data


def _prepare_scan_market_enrichment(market_data: list[dict]) -> tuple[list[dict], dict[str, int]]:
    unique_market_data = _dedupe_market_rows(market_data)
    mcap_map = _build_market_cap_map(market_data)
    return unique_market_data, mcap_map


def _merge_scan_market_rows(*row_groups: list[dict]) -> list[dict]:
    merged: list[dict] = []
    for rows in row_groups:
        if not isinstance(rows, list):
            continue
        for row in rows:
            if isinstance(row, dict):
                merged.append(dict(row))
    return _dedupe_market_rows(merged)


def _merge_breakout_radar_market_rows(*row_groups: list[dict]) -> list[dict]:
    merged_by_symbol: dict[str, dict] = {}
    symbol_order: list[str] = []
    for rows in row_groups:
        if not isinstance(rows, list):
            continue
        for row in rows:
            if not isinstance(row, dict):
                continue
            symbol = canonical_base_symbol((row or {}).get("symbol") or "")
            coin_id = str((row or {}).get("id") or "").strip().lower()
            if not symbol or "wrapped" in coin_id:
                continue
            incoming = dict(row)
            incoming["symbol"] = symbol.lower()
            existing = merged_by_symbol.get(symbol)
            if existing is None:
                merged_by_symbol[symbol] = incoming
                symbol_order.append(symbol)
                continue
            for key in ("market_cap", "_radar_source_score", "_quote_volume_24h", "_volume_24h"):
                if _sortable_float(incoming.get(key)) > _sortable_float(existing.get(key)):
                    existing[key] = incoming.get(key)
            for key in ("price_change_percentage_24h", "price_change_percentage_24h_in_currency"):
                if abs(_sortable_float(incoming.get(key))) > abs(_sortable_float(existing.get(key))):
                    existing[key] = incoming.get(key)
            if not str(existing.get("id") or "").strip() and str(incoming.get("id") or "").strip():
                existing["id"] = incoming.get("id")
            if not str(existing.get("_radar_source_kind") or "").strip() and str(incoming.get("_radar_source_kind") or "").strip():
                existing["_radar_source_kind"] = incoming.get("_radar_source_kind")
    return [merged_by_symbol[symbol] for symbol in symbol_order]


def _ticker_quote_volume_24h(ticker: object) -> float:
    if not isinstance(ticker, dict):
        return 0.0
    direct = _sortable_float(
        ticker.get("quoteVolume")
        if "quoteVolume" in ticker
        else ticker.get("quote_volume")
    )
    if direct > 0:
        return float(direct)
    base_vol = _sortable_float(
        ticker.get("baseVolume")
        if "baseVolume" in ticker
        else ticker.get("base_volume")
    )
    ref_price = (
        _sortable_float(ticker.get("vwap"))
        or _sortable_float(ticker.get("last"))
        or _sortable_float(ticker.get("close"))
    )
    if base_vol > 0 and ref_price > 0:
        return float(base_vol) * float(ref_price)
    return 0.0


def _ticker_pct_change_24h(ticker: object) -> float:
    if not isinstance(ticker, dict):
        return 0.0
    return _sortable_float(ticker.get("percentage"))


def _aligned_breakout_move(pct_change_24h: float, *, direction_filter: str) -> float:
    direction_key = str(direction_filter or "").strip().upper()
    move = _sortable_float(pct_change_24h)
    if direction_key == "UPSIDE":
        return max(0.0, move)
    if direction_key == "DOWNSIDE":
        return max(0.0, -move)
    return abs(move)


def _exchange_breakout_source_score(
    pct_change_24h: float,
    *,
    quote_volume_24h: float,
    direction_filter: str,
) -> float:
    aligned_move = _aligned_breakout_move(pct_change_24h, direction_filter=direction_filter)
    if aligned_move <= 0.35:
        return 0.0
    volume_score = max(0.0, min(1.0, (math.log10(max(quote_volume_24h, 1.0)) - 5.6) / 2.6))
    if aligned_move < 1.0:
        move_score = 0.26 + 0.34 * (aligned_move / 1.0)
    elif aligned_move < 4.0:
        move_score = 0.60 + 0.32 * ((aligned_move - 1.0) / 3.0)
    elif aligned_move <= 12.0:
        move_score = 0.92 - 0.18 * ((aligned_move - 4.0) / 8.0)
    else:
        move_score = 0.70 - min(0.24, 0.012 * (aligned_move - 12.0))
    return max(0.0, min(1.0, 0.62 * move_score + 0.38 * volume_score))


def _provider_breakout_source_score(
    pct_change_24h: float,
    *,
    direction_filter: str,
    market_cap: object = 0.0,
    trending_rank: object = 0.0,
    source_kind: str,
) -> float:
    aligned_move = _aligned_breakout_move(pct_change_24h, direction_filter=direction_filter)
    if source_kind == "trending":
        raw_rank = _sortable_float(trending_rank)
        if raw_rank > 0:
            rank_val = max(1.0, raw_rank)
            rank_score = max(0.0, min(1.0, 1.0 - ((rank_val - 1.0) / 19.0)))
        else:
            rank_score = 0.0
        if aligned_move <= 0.35 and rank_score <= 0.0:
            return 0.0
        move_score = max(0.0, min(1.0, aligned_move / 6.0))
        base_score = 0.58 * rank_score + 0.42 * move_score
        if aligned_move <= 0.35:
            return max(0.0, min(0.38, 0.38 * rank_score))
        return max(0.0, min(1.0, base_score))
    mcap_score = max(0.0, min(1.0, (_sortable_float(market_cap) / 300_000_000.0)))
    if aligned_move < 0.5:
        move_score = 0.18 + 0.24 * (aligned_move / 0.5)
    elif aligned_move < 3.5:
        move_score = 0.42 + 0.46 * ((aligned_move - 0.5) / 3.0)
    elif aligned_move <= 10.0:
        move_score = 0.88 - 0.18 * ((aligned_move - 3.5) / 6.5)
    else:
        move_score = 0.68 - min(0.26, 0.014 * (aligned_move - 10.0))
    return max(0.0, min(1.0, 0.82 * move_score + 0.18 * mcap_score))


def _annotate_breakout_provider_rows(
    rows: list[dict],
    *,
    direction_filter: str,
    source_kind: str,
) -> list[dict]:
    annotated: list[dict] = []
    for row in list(rows or []):
        if not isinstance(row, dict):
            continue
        base = canonical_base_symbol((row or {}).get("symbol") or "")
        if not base:
            continue
        out = dict(row)
        out["symbol"] = base.lower()
        out["_radar_source_kind"] = source_kind
        out["_radar_source_score"] = _provider_breakout_source_score(
            _sortable_float(out.get("price_change_percentage_24h")),
            direction_filter=direction_filter,
            market_cap=out.get("market_cap"),
            trending_rank=out.get("_trending_rank", out.get("score", out.get("market_cap_rank"))),
            source_kind=source_kind,
        )
        annotated.append(out)
    return annotated


def _breakout_prescan_profile(scan_timeframe: str) -> tuple[str, int]:
    tf = str(scan_timeframe or "").strip().lower()
    if tf in {"4h", "1d"}:
        return "1h", 96
    if tf == "1h":
        return "15m", 96
    if tf == "15m":
        return "5m", 96
    return tf or "5m", 84


def _build_breakout_freshness_snapshot(
    df_eval: pd.DataFrame | None,
    *,
    direction_filter: str,
) -> SimpleNamespace:
    if df_eval is None or len(df_eval) < 24:
        return SimpleNamespace(score=0.0, direction="Neutral")
    try:
        close = pd.to_numeric(df_eval.get("close"), errors="coerce").dropna()
        high = pd.to_numeric(df_eval.get("high"), errors="coerce").dropna()
        low = pd.to_numeric(df_eval.get("low"), errors="coerce").dropna()
    except Exception:
        return SimpleNamespace(score=0.0, direction="Neutral")
    if len(close) < 24 or len(high) < 24 or len(low) < 24:
        return SimpleNamespace(score=0.0, direction="Neutral")

    current = _sortable_float(close.iloc[-1])
    prev_3 = _sortable_float(close.iloc[-4]) if len(close) >= 4 else current
    prev_8 = _sortable_float(close.iloc[-9]) if len(close) >= 9 else prev_3
    recent_move_pct = ((current / max(prev_3, 1e-9)) - 1.0) * 100.0 if prev_3 > 0 else 0.0
    prior_move_pct = ((prev_3 / max(prev_8, 1e-9)) - 1.0) * 100.0 if prev_8 > 0 else 0.0

    ema_fast = _sortable_float(close.ewm(span=8, adjust=False).mean().iloc[-1])
    ema_slow = _sortable_float(close.ewm(span=21, adjust=False).mean().iloc[-1])
    upper_break = _sortable_float(high.iloc[-13:-1].max())
    lower_break = _sortable_float(low.iloc[-13:-1].min())
    avg_abs_move_pct = _sortable_float(close.pct_change().abs().tail(14).mean()) * 100.0

    def _directional_score(side: str) -> float:
        is_up = side == "UPSIDE"
        aligned_recent = max(0.0, recent_move_pct) if is_up else max(0.0, -recent_move_pct)
        aligned_prior = max(0.0, prior_move_pct) if is_up else max(0.0, -prior_move_pct)
        accel = max(0.0, aligned_recent - aligned_prior)
        breakout = (
            current >= upper_break * 0.997 if is_up else current <= lower_break * 1.003
        )
        trend_ok = ema_fast >= ema_slow if is_up else ema_fast <= ema_slow
        extension_pct = (
            max(0.0, ((current / max(ema_slow, 1e-9)) - 1.0) * 100.0)
            if is_up
            else max(0.0, ((ema_slow / max(current, 1e-9)) - 1.0) * 100.0)
        )
        extension_cap = max(2.2, avg_abs_move_pct * 5.0)
        extension_penalty = max(0.0, (extension_pct / max(extension_cap, 1e-9)) - 1.0)
        score = (
            0.42 * min(1.0, aligned_recent / 2.8)
            + 0.26 * min(1.0, accel / 2.2)
            + 0.20 * (1.0 if breakout else 0.0)
            + 0.12 * (1.0 if trend_ok else 0.0)
            - min(0.34, 0.20 * extension_penalty)
        )
        return max(0.0, min(1.0, score))

    up_score = _directional_score("UPSIDE")
    down_score = _directional_score("DOWNSIDE")
    direction_key = str(direction_filter or "").strip().upper()
    if direction_key == "UPSIDE":
        return SimpleNamespace(score=float(up_score), direction="Upside")
    if direction_key == "DOWNSIDE":
        return SimpleNamespace(score=float(down_score), direction="Downside")
    if up_score >= down_score:
        return SimpleNamespace(score=float(up_score), direction="Upside")
    return SimpleNamespace(score=float(down_score), direction="Downside")


def _enrich_breakout_radar_freshness(
    *,
    base_pairs: list[str],
    market_rows: list[dict],
    fetch_ohlcv,
    scan_timeframe: str,
    direction_filter: str = "Both",
    max_candidates: int = 18,
    freshness_cache: dict[tuple[str, str, str], float] | None = None,
) -> list[dict]:
    if not callable(fetch_ohlcv) or not market_rows:
        return list(market_rows or [])
    pair_map = {str(pair): _canonical_pair_base(pair) for pair in list(base_pairs or []) if isinstance(pair, str)}
    shortlist: list[tuple[str, dict]] = []
    for row in list(market_rows or []):
        if not isinstance(row, dict):
            continue
        base = canonical_base_symbol((row or {}).get("symbol") or "")
        if not base:
            continue
        pair = next((pair for pair, pair_base in pair_map.items() if pair_base == base), "")
        if not pair:
            continue
        shortlist.append((pair, row))
    shortlist.sort(
        key=lambda item: (
            -_sortable_float((item[1] or {}).get("_radar_source_score", 0.0)),
            -max(
                _sortable_float((item[1] or {}).get("_quote_volume_24h")),
                _sortable_float((item[1] or {}).get("total_volume")),
                _sortable_float((item[1] or {}).get("_volume_24h")),
            ),
            -abs(_sortable_float((item[1] or {}).get("price_change_percentage_24h"))),
            str((item[1] or {}).get("symbol") or ""),
        ),
    )
    prescan_timeframe, prescan_limit = _breakout_prescan_profile(scan_timeframe)
    direction_key = str(direction_filter or "Both").strip().upper()
    freshness_scores: dict[str, float] = {}
    fetch_lock = Lock()
    for pair, row in shortlist[: max(8, int(max_candidates))]:
        cache_key = (str(pair), str(prescan_timeframe), direction_key)
        cached_score = freshness_cache.get(cache_key) if isinstance(freshness_cache, dict) else None
        if cached_score is None:
            try:
                raw = _fetch_market_scan_ohlcv(
                    fetch_ohlcv=fetch_ohlcv,
                    fetch_coingecko_ohlcv_by_coin_id=lambda *_args, **_kwargs: None,
                    fetch_lock=fetch_lock,
                    symbol=pair,
                    timeframe=prescan_timeframe,
                    limit=prescan_limit,
                    fallback_coin_id=None,
                )
            except Exception:
                raw = None
            prepared = _prepare_closed_frame(raw, min_rows=28)
            snapshot = _build_breakout_freshness_snapshot(prepared, direction_filter=direction_filter)
            cached_score = float(snapshot.score or 0.0)
            if isinstance(freshness_cache, dict):
                freshness_cache[cache_key] = cached_score
        base = canonical_base_symbol((row or {}).get("symbol") or "")
        if base and _sortable_float(cached_score) > 0:
            freshness_scores[base] = float(cached_score)

    enriched: list[dict] = []
    for row in list(market_rows or []):
        if not isinstance(row, dict):
            continue
        out = dict(row)
        base = canonical_base_symbol((row or {}).get("symbol") or "")
        out["_radar_freshness_score"] = float(freshness_scores.get(base, 0.0))
        enriched.append(out)
    return enriched


def _build_exchange_breakout_rows(
    *,
    fetch_exchange_tickers_snapshot,
    direction_filter: str,
    limit: int,
) -> list[dict]:
    try:
        tickers = fetch_exchange_tickers_snapshot()
    except Exception:
        tickers = {}
    if not isinstance(tickers, dict) or not tickers:
        return []

    rows_by_base: dict[str, dict] = {}
    for pair, ticker in tickers.items():
        if not isinstance(pair, str) or "/" not in pair:
            continue
        raw_base, raw_quote = pair.split("/", 1)
        quote = str(raw_quote or "").upper()
        if quote not in {"USDT", "USD"}:
            continue
        base = canonical_base_symbol(raw_base)
        if not base or is_stable_base_symbol(base):
            continue
        quote_volume_24h = _ticker_quote_volume_24h(ticker)
        if quote_volume_24h <= 125_000:
            continue
        pct_change_24h = _ticker_pct_change_24h(ticker)
        radar_source_score = _exchange_breakout_source_score(
            pct_change_24h,
            quote_volume_24h=quote_volume_24h,
            direction_filter=direction_filter,
        )
        if radar_source_score <= 0.20:
            continue
        row = {
            "symbol": base.lower(),
            "id": "",
            "market_cap": 0,
            "price_change_percentage_24h": pct_change_24h,
            "_quote_volume_24h": quote_volume_24h,
            "_radar_source_score": radar_source_score,
            "_radar_source_kind": "exchange_breakout",
        }
        existing = rows_by_base.get(base)
        if existing is None or (
            _sortable_float(row.get("_radar_source_score")) > _sortable_float(existing.get("_radar_source_score"))
            or (
                _sortable_float(row.get("_radar_source_score")) == _sortable_float(existing.get("_radar_source_score"))
                and _sortable_float(row.get("_quote_volume_24h")) > _sortable_float(existing.get("_quote_volume_24h"))
            )
        ):
            rows_by_base[base] = row

    ranked_rows = sorted(
        rows_by_base.values(),
        key=lambda row: (
            -_sortable_float(row.get("_radar_source_score")),
            -_sortable_float(row.get("_quote_volume_24h")),
            -abs(_sortable_float(row.get("price_change_percentage_24h"))),
            str(row.get("symbol") or ""),
        ),
    )
    return ranked_rows[: max(10, int(limit))]


def _breakout_memory_seed_bases(memory_rows: object, *, limit: int = 36) -> list[str]:
    if memory_rows is None:
        return []
    if isinstance(memory_rows, pd.DataFrame):
        if memory_rows.empty:
            return []
        records = [dict(row) for row in memory_rows.to_dict("records")]
    elif isinstance(memory_rows, list):
        records = [dict(row) for row in memory_rows if isinstance(row, dict)]
    else:
        return []

    best_by_base: dict[str, float] = {}
    for row in records:
        base = canonical_base_symbol((row or {}).get("symbol") or "")
        if not base:
            continue
        source_score = _sortable_float(row.get("radar_source_score") or row.get("_radar_source_score"))
        freshness_score = _sortable_float(row.get("radar_freshness_score") or row.get("_radar_freshness_score"))
        pct_change = abs(_sortable_float(row.get("pct_change_24h") or row.get("price_change_percentage_24h")))
        quote_volume = _sortable_float(
            row.get("quote_volume_24h")
            or row.get("_quote_volume_24h")
            or row.get("total_volume")
            or row.get("_volume_24h")
        )
        pressure = max(source_score, freshness_score)
        if pressure < 0.34 and pct_change < 1.6:
            continue
        volume_score = 0.0
        if quote_volume > 0.0:
            volume_score = max(0.0, min(0.35, (math.log10(max(quote_volume, 1.0)) - 5.0) / 8.0))
        score = pressure + min(0.45, pct_change / 16.0) + volume_score
        best_by_base[base] = max(score, best_by_base.get(base, 0.0))

    ranked = sorted(best_by_base.items(), key=lambda item: (-item[1], item[0]))
    return [base for base, _score in ranked[: max(0, int(limit))]]


def _build_breakout_radar_universe(
    *,
    base_pairs: list[str],
    base_market_rows: list[dict],
    breakout_memory_rows: object | None = None,
    fetch_top_gainers_losers,
    fetch_trending_coins,
    fetch_exchange_tickers_snapshot,
    get_market_cap_rows_for_symbols,
    direction_filter: str,
    provider_fetch_n: int,
) -> tuple[list[str], list[dict], dict[str, int]]:
    radar_bases: list[str] = []
    seen_bases: set[str] = set()

    def _remember_base(raw_symbol: object) -> None:
        base = canonical_base_symbol(str(raw_symbol or "").strip())
        if not base or base in seen_bases:
            return
        seen_bases.add(base)
        radar_bases.append(base)

    for pair in list(base_pairs or []):
        _remember_base(_canonical_pair_base(pair))

    gainers: list[dict] = []
    losers: list[dict] = []
    trending: list[dict] = []
    exchange_breakouts: list[dict] = []
    try:
        gainers, losers = fetch_top_gainers_losers(limit=max(25, min(int(provider_fetch_n), 80)))
    except Exception:
        gainers, losers = [], []
    try:
        trending = fetch_trending_coins()
    except Exception:
        trending = []
    exchange_breakouts = _build_exchange_breakout_rows(
        fetch_exchange_tickers_snapshot=fetch_exchange_tickers_snapshot,
        direction_filter=direction_filter,
        limit=max(25, min(int(provider_fetch_n), 60)),
    )

    direction_key = str(direction_filter or "").strip().upper()
    directional_rows_raw: list[dict] = []
    if direction_key == "UPSIDE":
        directional_rows_raw = list(gainers or [])
    elif direction_key == "DOWNSIDE":
        directional_rows_raw = list(losers or [])
    else:
        directional_rows_raw = [*(gainers or [])[:50], *(losers or [])[:28]]
    directional_rows = _annotate_breakout_provider_rows(
        directional_rows_raw,
        direction_filter=direction_filter,
        source_kind="gainers_losers",
    )
    trending_rows = _annotate_breakout_provider_rows(
        [
            {
                **dict(row),
                "_trending_rank": idx + 1,
            }
            for idx, row in enumerate(list(trending or []))
        ],
        direction_filter=direction_filter,
        source_kind="trending",
    )

    for row in directional_rows[:80]:
        _remember_base((row or {}).get("symbol"))
    for row in trending_rows[:20]:
        _remember_base((row or {}).get("symbol"))
    for row in list(exchange_breakouts or [])[:60]:
        _remember_base((row or {}).get("symbol"))
    for base in _breakout_memory_seed_bases(breakout_memory_rows, limit=36):
        _remember_base(base)

    targeted_rows = get_market_cap_rows_for_symbols(tuple(radar_bases), vs_currency="usd")
    merged_market_rows = _merge_breakout_radar_market_rows(
        base_market_rows,
        directional_rows,
        trending_rows,
        exchange_breakouts,
        targeted_rows,
    )
    merged_pairs: list[str] = []
    seen_pairs: set[str] = set()
    for base in radar_bases:
        pair = f"{base}/USDT"
        if pair in seen_pairs:
            continue
        seen_pairs.add(pair)
        merged_pairs.append(pair)

    merged_mcap_map = _build_market_cap_map(merged_market_rows)
    return merged_pairs, merged_market_rows, merged_mcap_map


def _trending_scan_timeframe(scan_timeframe: str) -> str:
    tf = str(scan_timeframe or "").strip().lower()
    if tf in {"5m", "15m", "1h", "4h"}:
        return tf
    return "4h"


def _trending_direction_allows(pct_change: object, *, direction_filter: str) -> bool:
    direction_key = str(direction_filter or "Both").strip().upper()
    if direction_key == "BOTH":
        return True
    move = _sortable_float(pct_change)
    if direction_key == "UPSIDE":
        return move >= -0.15
    if direction_key == "DOWNSIDE":
        return move <= 0.15
    return True


def _volume_anomaly_rows_from_surges(
    surges: list[dict],
    *,
    direction_filter: str,
) -> list[dict]:
    rows: list[dict] = []
    for row in list(surges or []):
        if not isinstance(row, dict):
            continue
        base = canonical_base_symbol(row.get("Symbol") or "")
        if not base or is_stable_base_symbol(base):
            continue
        pct_24h = _sortable_float(row.get("24h %"))
        pct_1 = _sortable_float(row.get("1-Candle %"))
        pct_ref = pct_24h if abs(pct_24h) >= 0.01 else pct_1
        if not _trending_direction_allows(pct_ref, direction_filter=direction_filter):
            continue
        anomaly_score = max(0.0, min(1.0, _sortable_float(row.get("Score"))))
        rows.append(
            {
                "symbol": base.lower(),
                "id": "",
                "market_cap": 0,
                "price_change_percentage_24h": pct_ref,
                "_volume_24h": _sortable_float(row.get("Last Vol")),
                "_radar_source_kind": "volume_anomaly",
                "_radar_source_score": max(0.28, min(1.0, anomaly_score)),
            }
        )
    return rows


def _trending_volume_cache_key(
    *,
    symbols: list[str],
    scan_tf: str,
    ratio_gate: float,
    z_gate: float,
    extreme_ratio_gate: float,
    extreme_z_gate: float,
) -> tuple[object, ...]:
    return (
        str(scan_tf or "").strip().lower(),
        tuple(str(symbol or "").strip().upper() for symbol in list(symbols or [])),
        round(float(ratio_gate), 3),
        round(float(z_gate), 3),
        round(float(extreme_ratio_gate), 3),
        round(float(extreme_z_gate), 3),
    )


def _get_cached_trending_volume_surges(
    cache: dict | None,
    cache_key: tuple[object, ...],
    *,
    now_ts: float,
    ttl_seconds: int,
) -> list[dict] | None:
    if not isinstance(cache, dict):
        return None
    item = cache.get(cache_key)
    if not isinstance(item, dict):
        return None
    try:
        item_ts = float(item.get("ts") or 0.0)
    except Exception:
        item_ts = 0.0
    if item_ts <= 0.0 or (float(now_ts) - item_ts) > float(ttl_seconds):
        cache.pop(cache_key, None)
        return None
    surges = item.get("surges")
    if not isinstance(surges, list):
        return None
    return [dict(row) for row in surges if isinstance(row, dict)]


def _remember_trending_volume_surges(
    cache: dict | None,
    cache_key: tuple[object, ...],
    surges: list[dict],
    *,
    now_ts: float,
    max_entries: int = _TRENDING_VOLUME_CACHE_LIMIT,
) -> None:
    if not isinstance(cache, dict):
        return
    cache[cache_key] = {
        "ts": float(now_ts),
        "surges": [dict(row) for row in list(surges or []) if isinstance(row, dict)],
    }
    if len(cache) <= int(max_entries):
        return
    ranked_keys = sorted(
        list(cache.keys()),
        key=lambda key: float(cache.get(key, {}).get("ts") or 0.0) if isinstance(cache.get(key), dict) else 0.0,
    )
    for old_key in ranked_keys[: max(0, len(cache) - int(max_entries))]:
        cache.pop(old_key, None)


def _build_trending_scan_universe(
    *,
    base_market_rows: list[dict],
    fetch_trending_coins,
    fetch_top_gainers_losers,
    get_top_volume_usdt_symbols,
    get_market_cap_rows_for_symbols,
    fetch_ohlcv,
    direction_filter: str,
    scan_timeframe: str,
    provider_fetch_n: int,
    volume_anomaly_cache: dict | None = None,
    cache_now_ts: float | None = None,
) -> tuple[list[str], list[dict], dict[str, int]]:
    trending_bases: list[str] = []
    seen_bases: set[str] = set()

    def _remember_base(raw_symbol: object) -> None:
        base = canonical_base_symbol(str(raw_symbol or "").strip())
        if not base or base in seen_bases or is_stable_base_symbol(base):
            return
        seen_bases.add(base)
        trending_bases.append(base)

    try:
        trending = fetch_trending_coins()
    except Exception:
        trending = []
    try:
        gainers, losers = fetch_top_gainers_losers(limit=max(20, min(int(provider_fetch_n), 60)))
    except Exception:
        gainers, losers = [], []

    direction_key = str(direction_filter or "Both").strip().upper()
    if direction_key == "UPSIDE":
        momentum_raw = list(gainers or [])
    elif direction_key == "DOWNSIDE":
        momentum_raw = list(losers or [])
    else:
        momentum_raw = [*(gainers or [])[:35], *(losers or [])[:20]]

    trending_rows = _annotate_breakout_provider_rows(
        [
            {
                **dict(row),
                "_trending_rank": idx + 1,
            }
            for idx, row in enumerate(list(trending or []))
        ],
        direction_filter=direction_filter,
        source_kind="trending",
    )
    momentum_rows = _annotate_breakout_provider_rows(
        momentum_raw,
        direction_filter=direction_filter,
        source_kind="gainers_losers",
    )

    volume_rows: list[dict] = []
    try:
        anomaly_symbols, _raw = get_top_volume_usdt_symbols(max(24, min(60, int(provider_fetch_n))))
        scan_tf = _trending_scan_timeframe(scan_timeframe)
        ratio_gate, z_gate, extreme_ratio_gate, extreme_z_gate = _compute_scan_thresholds(scan_tf, 1.5, 2.0)
        scan_symbols = list(anomaly_symbols or [])[: max(20, min(50, int(provider_fetch_n)))]
        now_ts = float(cache_now_ts) if cache_now_ts is not None else time.time()
        cache_key = _trending_volume_cache_key(
            symbols=scan_symbols,
            scan_tf=scan_tf,
            ratio_gate=ratio_gate,
            z_gate=z_gate,
            extreme_ratio_gate=extreme_ratio_gate,
            extreme_z_gate=extreme_z_gate,
        )
        surges = _get_cached_trending_volume_surges(
            volume_anomaly_cache,
            cache_key,
            now_ts=now_ts,
            ttl_seconds=_TRENDING_VOLUME_CACHE_TTL_SECONDS,
        )
        if surges is None:
            surges, _diag = _run_volume_anomaly_scan(
                scan_symbols,
                fetch_ohlcv=fetch_ohlcv,
                scan_tf=scan_tf,
                ratio_gate=ratio_gate,
                z_gate=z_gate,
                extreme_ratio_gate=extreme_ratio_gate,
                extreme_z_gate=extreme_z_gate,
            )
            _remember_trending_volume_surges(
                volume_anomaly_cache,
                cache_key,
                surges,
                now_ts=now_ts,
            )
        volume_rows = _volume_anomaly_rows_from_surges(surges, direction_filter=direction_filter)
    except Exception:
        volume_rows = []

    for row in volume_rows[:40]:
        _remember_base((row or {}).get("symbol"))
    for row in trending_rows[:20]:
        _remember_base((row or {}).get("symbol"))
    for row in momentum_rows[:55]:
        _remember_base((row or {}).get("symbol"))

    targeted_rows = get_market_cap_rows_for_symbols(tuple(trending_bases), vs_currency="usd")
    merged_market_rows = _merge_breakout_radar_market_rows(
        base_market_rows,
        volume_rows,
        trending_rows,
        momentum_rows,
        targeted_rows,
    )
    merged_pairs: list[str] = []
    seen_pairs: set[str] = set()
    for base in trending_bases:
        pair = f"{base}/USDT"
        if pair in seen_pairs:
            continue
        seen_pairs.add(pair)
        merged_pairs.append(pair)

    merged_mcap_map = _build_market_cap_map(merged_market_rows)
    return merged_pairs, merged_market_rows, merged_mcap_map


def _sync_market_cap_cells(
    rows: list[dict],
    mcap_map: dict[str, int],
    readable_market_cap,
) -> list[dict]:
    synced_rows: list[dict] = []
    for row in rows:
        out = dict(row)
        base = canonical_base_symbol(str(out.get("Coin") or ""))
        mcap_val = int(mcap_map.get(base) or 0) if base else 0
        out["Market Cap ($)"] = readable_market_cap(mcap_val) if mcap_val else "—"
        out["__mcap_val"] = mcap_val
        synced_rows.append(out)
    return synced_rows


def _merge_market_cap_maps(
    current: dict[str, int],
    incoming: dict[str, int],
) -> dict[str, int]:
    merged = dict(current) if isinstance(current, dict) else {}
    for base, raw_value in (incoming or {}).items():
        try:
            value = int(raw_value or 0)
        except Exception:
            value = 0
        if value <= 0:
            continue
        if value > int(merged.get(base) or 0):
            merged[base] = value
    return merged


def _remember_display_scan_state(
    state: dict[str, object] | None,
    *,
    batch_results: list[dict],
    candidate_count: int,
    mcap_map: dict[str, int],
    has_market_rows: bool,
    source_pair_count: int,
    market_row_count: int,
) -> dict[str, object]:
    out = dict(state) if isinstance(state, dict) else {
        "candidate_count": 0,
        "mcap_map": {},
        "has_market_rows": False,
        "source_pair_count": 0,
        "market_row_count": 0,
    }
    if not batch_results:
        return out
    out["candidate_count"] = max(int(out.get("candidate_count") or 0), int(candidate_count))
    out["mcap_map"] = _merge_market_cap_maps(
        out.get("mcap_map") if isinstance(out.get("mcap_map"), dict) else {},
        mcap_map,
    )
    out["has_market_rows"] = bool(out.get("has_market_rows")) or bool(has_market_rows)
    out["source_pair_count"] = max(int(out.get("source_pair_count") or 0), int(source_pair_count))
    out["market_row_count"] = max(int(out.get("market_row_count") or 0), int(market_row_count))
    return out


def _resolve_display_scan_state(
    *,
    fresh_results: list[dict],
    current_candidate_count: int,
    current_mcap_map: dict[str, int],
    current_has_market_rows: bool,
    current_source_pair_count: int,
    current_market_row_count: int,
    display_state: dict[str, object] | None,
) -> dict[str, object]:
    if fresh_results and isinstance(display_state, dict) and int(display_state.get("candidate_count") or 0) > 0:
        return {
            "candidate_count": int(display_state.get("candidate_count") or 0),
            "mcap_map": dict(display_state.get("mcap_map") or {}),
            "has_market_rows": bool(display_state.get("has_market_rows")),
            "source_pair_count": int(display_state.get("source_pair_count") or 0),
            "market_row_count": int(display_state.get("market_row_count") or 0),
        }
    return {
        "candidate_count": int(current_candidate_count),
        "mcap_map": dict(current_mcap_map or {}),
        "has_market_rows": bool(current_has_market_rows),
        "source_pair_count": int(current_source_pair_count),
        "market_row_count": int(current_market_row_count),
    }


def _resolve_notice_scan_state(
    *,
    current_candidate_count: int,
    current_has_market_rows: bool,
    current_source_pair_count: int,
    current_market_row_count: int,
    display_state: dict[str, object] | None,
) -> dict[str, object]:
    out = {
        "candidate_count": int(current_candidate_count),
        "has_market_rows": bool(current_has_market_rows),
        "source_pair_count": int(current_source_pair_count),
        "market_row_count": int(current_market_row_count),
    }
    if not isinstance(display_state, dict):
        return out
    out["candidate_count"] = max(int(out["candidate_count"]), int(display_state.get("candidate_count") or 0))
    out["has_market_rows"] = bool(out["has_market_rows"]) or bool(display_state.get("has_market_rows"))
    out["source_pair_count"] = max(int(out["source_pair_count"]), int(display_state.get("source_pair_count") or 0))
    out["market_row_count"] = max(int(out["market_row_count"]), int(display_state.get("market_row_count") or 0))
    return out


def _is_stable_base(base: str) -> bool:
    return is_stable_base_symbol(base)


def _candidate_scan_symbols(
    *,
    usdt_symbols: list[str],
    market_rows: list[dict],
    exclude_stables: bool,
    custom_bases_applied: list[str],
    timeframe: str = "1h",
    direction_filter: str = "Both",
    scan_mode: str = _SCAN_MODE_BROAD,
    classify_symbol_sector=None,
) -> list[str]:
    candidates = _candidate_scan_symbols_impl(
        usdt_symbols=usdt_symbols,
        market_rows=market_rows,
        exclude_stables=False,
        custom_bases_applied=custom_bases_applied,
        timeframe=timeframe,
        direction_filter=direction_filter,
        scan_mode=scan_mode,
        classify_symbol_sector=classify_symbol_sector,
    )
    if exclude_stables:
        candidates = [s for s in candidates if "/" in s and not _is_stable_base(s.split("/")[0].upper())]
    return candidates


def _custom_watchlist_missing_status(
    custom_bases_applied: list[str],
    visible_rows: list[dict],
    skipped_symbols: list[tuple[str, str]],
    *,
    coin_id_map: dict[str, str] | None = None,
    coingecko_coin_id_fallback_available: bool = True,
    coingecko_coin_id_fallback_reason: str = "",
) -> list[tuple[str, str]]:
    requested_bases = _normalize_custom_bases(custom_bases_applied)
    if not requested_bases:
        return []
    visible_bases: set[str] = set()
    for row in visible_rows:
        base = canonical_base_symbol((row or {}).get("Coin") or "")
        if base:
            visible_bases.add(base)
    skipped_by_base: dict[str, str] = {}
    for symbol, reason in skipped_symbols:
        base = _canonical_pair_base(symbol)
        if base and base not in skipped_by_base:
            skipped_by_base[base] = str(reason or "").strip() or "scan skipped"
    missing: list[tuple[str, str]] = []
    for base in requested_bases:
        if base in visible_bases:
            continue
        reason = skipped_by_base.get(base)
        if not reason:
            fallback_coin_id = (coin_id_map or {}).get(base)
            if fallback_coin_id and not bool(coingecko_coin_id_fallback_available):
                reason = _coingecko_coin_id_unavailable_message(coingecko_coin_id_fallback_reason)
            elif fallback_coin_id:
                reason = "no exchange OHLCV data; CoinGecko backup returned empty"
            else:
                reason = "no exchange pair; coin-id unresolved for backup"
        missing.append((base, reason))
    return missing


def _next_universe_fetch_n(
    current_fetch_n: int,
    *,
    custom_mode_active: bool,
    eligible_count: int,
    requested_n: int,
    max_fetch_n: int = 250,
) -> int:
    if custom_mode_active:
        return int(current_fetch_n)
    if int(eligible_count) >= int(requested_n):
        return int(current_fetch_n)
    if int(current_fetch_n) >= int(max_fetch_n):
        return int(current_fetch_n)
    next_fetch_n = max(
        int(current_fetch_n * 1.5),
        int(current_fetch_n) + max(int(requested_n), 25),
    )
    return min(int(max_fetch_n), int(next_fetch_n))


def _delta_fallback_symbol(
    requested_symbol: str,
    actual_symbol: str | None,
    source_provider: str | None,
) -> str | None:
    if str(source_provider or "").strip().lower() != "exchange":
        return None
    symbol = str(actual_symbol or requested_symbol or "").strip()
    return symbol or None


def _fetch_ticker_delta_once(get_price_change, fallback_symbol: str | None, fetch_lock: Lock | None = None):
    symbol = str(fallback_symbol or "").strip()
    if not symbol:
        return None
    if fetch_lock is None:
        return get_price_change(symbol)
    with fetch_lock:
        return get_price_change(symbol)


def render(ctx: dict) -> None:
    st = get_ctx(ctx, "st")
    ACCENT = get_ctx(ctx, "ACCENT")
    TEXT_MUTED = get_ctx(ctx, "TEXT_MUTED")
    POSITIVE = get_ctx(ctx, "POSITIVE")
    NEGATIVE = get_ctx(ctx, "NEGATIVE")
    WARNING = get_ctx(ctx, "WARNING")
    get_market_top_snapshot = get_ctx(ctx, "get_market_top_snapshot")
    get_price_change = get_ctx(ctx, "get_price_change")
    _tip = get_ctx(ctx, "_tip")
    get_major_ohlcv_bundle = get_ctx(ctx, "get_major_ohlcv_bundle")
    ml_ensemble_predict = get_ctx(ctx, "ml_ensemble_predict")
    get_top_volume_usdt_symbols = get_ctx(ctx, "get_top_volume_usdt_symbols")
    fetch_top_gainers_losers = get_ctx(ctx, "fetch_top_gainers_losers")
    fetch_trending_coins = get_ctx(ctx, "fetch_trending_coins")
    fetch_exchange_tickers_snapshot = get_ctx(ctx, "fetch_exchange_tickers_snapshot")
    get_market_cap_rows_for_symbols = get_ctx(ctx, "get_market_cap_rows_for_symbols")
    build_market_regime_snapshot = get_ctx(ctx, "build_market_regime_snapshot")
    build_market_trade_gate = get_ctx(ctx, "build_market_trade_gate")
    build_signal_risk_sizing = get_ctx(ctx, "build_signal_risk_sizing")
    market_default_risk_budget = get_ctx(ctx, "market_default_risk_budget")
    build_sector_rotation_snapshot = get_ctx(ctx, "build_sector_rotation_snapshot")
    classify_symbol_sector = get_ctx(ctx, "classify_symbol_sector")
    build_market_flow_proxy_snapshot = get_ctx(ctx, "build_market_flow_proxy_snapshot")
    get_market_flow_proxy_rows = get_ctx(ctx, "get_market_flow_proxy_rows")
    build_market_alerts = get_ctx(ctx, "build_market_alerts")
    build_market_catalyst_snapshot = get_ctx(ctx, "build_market_catalyst_snapshot")
    get_market_catalyst_events = get_ctx(ctx, "get_market_catalyst_events")
    fetch_coingecko_ohlcv_by_coin_id = get_ctx(ctx, "fetch_coingecko_ohlcv_by_coin_id")
    coingecko_coin_id_fallback_available = _coingecko_coin_id_fallback_available(fetch_coingecko_ohlcv_by_coin_id)
    coingecko_coin_id_fallback_reason = _coingecko_coin_id_fallback_reason(fetch_coingecko_ohlcv_by_coin_id)
    fetch_ohlcv = get_ctx(ctx, "fetch_ohlcv")
    analyse = get_ctx(ctx, "analyse")
    get_scalping_entry_target = get_ctx(ctx, "get_scalping_entry_target")
    scalp_quality_gate = get_ctx(ctx, "scalp_quality_gate")
    _calc_conviction = get_ctx(ctx, "_calc_conviction")
    signal_plain = get_ctx(ctx, "signal_plain")
    direction_key = get_ctx(ctx, "direction_key")
    direction_label = get_ctx(ctx, "direction_label")
    readable_market_cap = get_ctx(ctx, "readable_market_cap")
    format_delta = get_ctx(ctx, "format_delta")
    format_trend = get_ctx(ctx, "format_trend")
    format_adx = get_ctx(ctx, "format_adx")
    format_stochrsi = get_ctx(ctx, "format_stochrsi")
    sanitize_trading_terms = get_ctx(ctx, "sanitize_trading_terms")
    get_signal_tracker_db_path = get_ctx(ctx, "get_signal_tracker_db_path")
    init_signal_tracker_db = get_ctx(ctx, "init_signal_tracker_db")
    fetch_signal_events_df = get_ctx(ctx, "fetch_signal_events_df")
    fetch_signal_forward_windows_df = get_ctx(ctx, "fetch_signal_forward_windows_df")
    fetch_breakout_radar_snapshots_df = get_ctx(ctx, "fetch_breakout_radar_snapshots_df")
    fetch_scanner_trace_events_df = get_ctx(ctx, "fetch_scanner_trace_events_df")
    build_scanner_trace_summary = get_ctx(ctx, "build_scanner_trace_summary")
    build_adaptive_context_model = get_ctx(ctx, "build_adaptive_context_model")
    build_live_signal_adaptive_snapshot = get_ctx(ctx, "build_live_signal_adaptive_snapshot")
    log_market_alerts = get_ctx(ctx, "log_market_alerts")
    log_signal_events = get_ctx(ctx, "log_signal_events")
    log_breakout_radar_snapshots = get_ctx(ctx, "log_breakout_radar_snapshots")
    log_scanner_trace_events = get_ctx(ctx, "log_scanner_trace_events")
    resolve_open_signal_events_for_frame = get_ctx(ctx, "resolve_open_signal_events_for_frame")
    backfill_signal_forward_windows_via_fetch = get_ctx(ctx, "backfill_signal_forward_windows_via_fetch")
    _debug = get_ctx(ctx, "_debug")
    signal_tracker_db_path = init_signal_tracker_db(get_signal_tracker_db_path())
    major_fallback_symbols = [
        "BTC/USDT",
        "ETH/USDT",
        "SOL/USDT",
        "XRP/USDT",
        "BNB/USDT",
        "ADA/USDT",
        "DOGE/USDT",
        "AVAX/USDT",
        "LINK/USDT",
        "TON/USDT",
    ]

    # Unified top snapshot (provider-consistent + fallback + last-good cache).
    top_snapshot = get_market_top_snapshot()
    btc_dom_raw = top_snapshot.get("btc_dom")
    eth_dom_raw = top_snapshot.get("eth_dom")
    total_mcap_raw = top_snapshot.get("total_mcap")
    mcap_24h_pct = top_snapshot.get("mcap_24h_pct")
    bnb_dom_raw = top_snapshot.get("bnb_dom")
    sol_dom_raw = top_snapshot.get("sol_dom")
    ada_dom_raw = top_snapshot.get("ada_dom")
    xrp_dom_raw = top_snapshot.get("xrp_dom")

    def _to_num(v: object) -> float:
        try:
            f = float(v)
            return f if pd.notna(f) else 0.0
        except Exception:
            return 0.0

    btc_dom = _to_num(btc_dom_raw)
    eth_dom = _to_num(eth_dom_raw)
    bnb_dom = _to_num(bnb_dom_raw)
    sol_dom = _to_num(sol_dom_raw)
    ada_dom = _to_num(ada_dom_raw)
    xrp_dom = _to_num(xrp_dom_raw)
    total_mcap = _to_num(total_mcap_raw)
    fg_value_raw = top_snapshot.get("fg_value")
    fg_label = str(top_snapshot.get("fg_label") or "Unavailable")
    fg_value = fg_value_raw if isinstance(fg_value_raw, (int, float)) else None
    fg_available = fg_value is not None
    btc_price_raw = top_snapshot.get("btc_price")
    eth_price_raw = top_snapshot.get("eth_price")
    btc_price = float(btc_price_raw) if isinstance(btc_price_raw, (int, float)) else None
    eth_price = float(eth_price_raw) if isinstance(eth_price_raw, (int, float)) else None

    # Treat all-zero dominance payload as unavailable upstream enrichment.
    dominance_sum = (
        max(btc_dom, 0.0)
        + max(eth_dom, 0.0)
        + max(bnb_dom, 0.0)
        + max(sol_dom, 0.0)
        + max(ada_dom, 0.0)
        + max(xrp_dom, 0.0)
    )
    dominance_feed_ok = dominance_sum > 0.01
    mcap_feed_ok = total_mcap > 0

    btc_dom_display = btc_dom if dominance_feed_ok else None
    eth_dom_display = eth_dom if dominance_feed_ok else None

    # Compute percentage change for market cap
    delta_mcap = float(mcap_24h_pct) if pd.notna(mcap_24h_pct) and mcap_feed_ok else float("nan")

    # Price changes come from the same provider as price in top snapshot.
    btc_change = top_snapshot.get("btc_change")
    eth_change = top_snapshot.get("eth_change")

    render_page_header(
        st,
        title="Crypto Market Intelligence Hub",
        hero=True,
        intro_html=(
            f"Your market overview dashboard. Shows live BTC/ETH prices, total market cap, "
            f"{_tip('Fear & Greed Index', copy_text('market.hero.fear_greed_tip'))} "
            f"and {_tip('BTC Dominance', 'Bitcoin’s share of the total crypto market. Rising dominance usually means money is hiding in BTC; falling dominance can support altcoins.')}. "
            f"The Market tab scans Broad Market, Breakout Radar, Trending Coins, or your custom watchlist, then scores each symbol with closed-candle technical and AI context."
        ),
    )

    # Determine which timeframe to use for market bias gauges. We rely on
    # Streamlit session state to persist the selected timeframe from the
    # scanner controls. On first render, default to 1h. Bias is computed
    # from a six-asset major bundle (BTC/ETH/BNB/SOL/ADA/XRP) on 500
    # candles, using dominance weights when available and equal-weight
    # fallback when dominance feed is unavailable.
    selected_timeframe = st.session_state.get("market_timeframe", "1h")
    # Top row: Price and market cap metrics.
    m1, m2, m3, m4 = st.columns(4, gap="medium")
    def _header_band_html(change_pct: float | None, tone_color: str) -> str:
        if change_pct is None or pd.isna(change_pct):
            marker_html = ""
        else:
            clipped = max(-5.0, min(5.0, float(change_pct)))
            marker_left = 50.0 + clipped * 10.0
            marker_html = (
                f"<span class='market-header-band-marker' style='left:{marker_left:.2f}%; "
                f"background:{tone_color}; box-shadow:0 0 0 4px rgba(255,255,255,0.04), 0 0 16px {tone_color};'></span>"
            )
        return (
            "<div class='market-header-band'>"
            f"<span class='market-header-band-seg market-header-band-seg--neg'></span>"
            f"<span class='market-header-band-seg market-header-band-seg--neutral'></span>"
            f"<span class='market-header-band-seg market-header-band-seg--pos'></span>"
            f"{marker_html}"
            "</div>"
            "<div class='market-header-band-guides'><span>-5%</span><span>24H FLOW</span><span>+5%</span></div>"
        )

    def _market_summary_card(
        *,
        title: str,
        value_text: str,
        delta_pct: float | None,
        context_label: str,
        unavailable: bool = False,
    ) -> str:
        accent = TEXT_MUTED if unavailable else (POSITIVE if (delta_pct or 0) >= 0 else NEGATIVE)
        if unavailable:
            return (
                f"<div class='market-header-card market-header-card--muted' style='--header-accent:{accent};'>"
                f"<div class='market-header-title'>{html.escape(title)}</div>"
                "<div class='market-header-main'>"
                "<div class='market-header-value'>N/A</div>"
                "<div class='market-header-pill market-header-pill--muted'>Offline</div>"
                "</div>"
                "<div class='market-header-note'>Live feed unavailable.</div>"
                "</div>"
            )
        delta = float(delta_pct or 0.0)
        tone_color = POSITIVE if delta >= 0 else NEGATIVE
        delta_arrow = "▲" if delta >= 0 else "▼"
        delta_text = f"{delta_arrow} {abs(delta):.2f}%"
        return (
            f"<div class='market-header-card' style='--header-accent:{tone_color};'>"
            f"<div class='market-header-title'>{html.escape(title)}</div>"
            f"<div class='market-header-main'>"
            f"<div class='market-header-value'>{html.escape(value_text)}</div>"
            f"<div class='market-header-pill' style='color:{tone_color}; border-color:{tone_color};'>{delta_text}</div>"
            f"</div>"
            f"{_header_band_html(delta, tone_color)}"
            f"<div class='market-header-note'>{html.escape(context_label)}</div>"
            f"</div>"
        )

    def _fear_greed_card(value: float | None, label: str, available: bool) -> str:
        fg_color = POSITIVE if "Greed" in label else (NEGATIVE if "Fear" in label else WARNING)
        if not available or value is None:
            return (
                f"<div class='market-header-card market-header-card--sentiment market-header-card--muted' style='--header-accent:{TEXT_MUTED};'>"
                "<div class='market-header-title'>Fear &amp; Greed</div>"
                "<div class='market-header-main market-header-main--sentiment'>"
                "<div class='market-header-value'>N/A</div>"
                "<div class='market-header-pill market-header-pill--muted'>Offline</div>"
                "</div>"
                "<div class='market-header-note'>Sentiment feed unavailable.</div>"
                "</div>"
            )
        marker_left = max(0.0, min(100.0, float(value)))
        return (
            f"<div class='market-header-card market-header-card--sentiment' style='--header-accent:{fg_color};'>"
            "<div class='market-header-title'>Fear &amp; Greed "
            "<span title='Quick sentiment gauge from 0 to 100. Lower means fear; higher means greed.' "
            "class='market-header-info'>i</span></div>"
            "<div class='market-header-main market-header-main--sentiment'>"
            f"<div class='market-header-value'>{int(value):d}</div>"
            f"<div class='market-header-pill' style='color:{fg_color}; border-color:{fg_color};'>{html.escape(label)}</div>"
            "</div>"
            "<div class='market-header-sentiment-scale'>"
            "<span class='market-header-sentiment-seg market-header-sentiment-seg--fear'></span>"
            "<span class='market-header-sentiment-seg market-header-sentiment-seg--caution'></span>"
            "<span class='market-header-sentiment-seg market-header-sentiment-seg--neutral'></span>"
            "<span class='market-header-sentiment-seg market-header-sentiment-seg--greed'></span>"
            "<span class='market-header-sentiment-seg market-header-sentiment-seg--extreme'></span>"
            f"<span class='market-header-sentiment-marker' style='left:{marker_left:.2f}%; background:{fg_color}; box-shadow:0 0 0 4px rgba(255,255,255,0.04), 0 0 16px {fg_color};'></span>"
            "</div>"
            "<div class='market-header-band-guides'><span>Fear</span><span>Neutral</span><span>Greed</span></div>"
            "<div class='market-header-note'>Sentiment context, not a standalone trigger.</div>"
            "</div>"
        )

    # Bitcoin price
    with m1:
        st.markdown(
            _market_summary_card(
                title="Bitcoin Price",
                value_text=f"${btc_price:,.2f}" if btc_price is not None and btc_price > 0 else "N/A",
                delta_pct=btc_change,
                context_label="BTC is setting the market tone right now.",
                unavailable=not (btc_price is not None and btc_price > 0),
            ),
            unsafe_allow_html=True,
        )
    # Ethereum price
    with m2:
        st.markdown(
            _market_summary_card(
                title="Ethereum Price",
                value_text=f"${eth_price:,.2f}" if eth_price is not None and eth_price > 0 else "N/A",
                delta_pct=eth_change,
                context_label="ETH participation sets alt tone.",
                unavailable=not (eth_price is not None and eth_price > 0),
            ),
            unsafe_allow_html=True,
        )
    # Total market cap
    with m3:
        st.markdown(
            _market_summary_card(
                title="Total Market Cap",
                value_text=f"${total_mcap / 1e12:.2f}T" if mcap_feed_ok else "N/A",
                delta_pct=delta_mcap if pd.notna(delta_mcap) else None,
                context_label="Broad crypto balance sheet.",
                unavailable=not mcap_feed_ok,
            ),
            unsafe_allow_html=True,
        )
    # Fear & Greed index
    with m4:
        st.markdown(
            _fear_greed_card(fg_value, fg_label, fg_available),
            unsafe_allow_html=True,
        )
    # Second row: Dominance gauges and AI market outlook
    # Compute AI market outlook using a dominance-weighted ML prediction across
    # BTC, ETH and major altcoins (BNB, SOL, ADA, XRP) on the selected timeframe.
    btc_prob = eth_prob = bnb_prob = sol_prob = ada_prob = xrp_prob = 0.5
    try:
        bundle_behav = get_major_ohlcv_bundle(selected_timeframe, limit=500)
        btc_df_behav = bundle_behav.get('BTC/USDT')
        eth_df_behav = bundle_behav.get('ETH/USDT')
        bnb_df_behav = bundle_behav.get('BNB/USDT')
        sol_df_behav = bundle_behav.get('SOL/USDT')
        ada_df_behav = bundle_behav.get('ADA/USDT')
        xrp_df_behav = bundle_behav.get('XRP/USDT')
        # Initialise probabilities at a neutral value of 0.5.  If data
        # retrieval or training fails for an asset, the neutral prior will
        # prevent it from skewing the combined outlook.
        btc_eval = _prepare_closed_frame(btc_df_behav, min_rows=60)
        eth_eval = _prepare_closed_frame(eth_df_behav, min_rows=60)
        bnb_eval = _prepare_closed_frame(bnb_df_behav, min_rows=60)
        sol_eval = _prepare_closed_frame(sol_df_behav, min_rows=60)
        ada_eval = _prepare_closed_frame(ada_df_behav, min_rows=60)
        xrp_eval = _prepare_closed_frame(xrp_df_behav, min_rows=60)
        if btc_eval is not None:
            btc_prob, _, _ = ml_ensemble_predict(btc_eval)
        if eth_eval is not None:
            eth_prob, _, _ = ml_ensemble_predict(eth_eval)
        if bnb_eval is not None:
            bnb_prob, _, _ = ml_ensemble_predict(bnb_eval)
        if sol_eval is not None:
            sol_prob, _, _ = ml_ensemble_predict(sol_eval)
        if ada_eval is not None:
            ada_prob, _, _ = ml_ensemble_predict(ada_eval)
        if xrp_eval is not None:
            xrp_prob, _, _ = ml_ensemble_predict(xrp_eval)
        # Compute a weighted probability across all assets.  Prefer market-cap
        # dominance weights; if dominance enrichment is unavailable, fall back
        # to equal weights so the score remains usable in exchange-only mode.
        dominance_weights = [
            max(float(btc_dom), 0.0),
            max(float(eth_dom), 0.0),
            max(float(bnb_dom), 0.0),
            max(float(sol_dom), 0.0),
            max(float(ada_dom), 0.0),
            max(float(xrp_dom), 0.0),
        ]
        dom_sum = float(sum(dominance_weights))
        if dom_sum > 0.01:
            weights = [w / dom_sum for w in dominance_weights]
            behaviour_weight_mode = "dominance"
        else:
            weights = [1.0 / 6.0] * 6
            behaviour_weight_mode = "equal"
        behaviour_prob = (
            btc_prob * weights[0]
            + eth_prob * weights[1]
            + bnb_prob * weights[2]
            + sol_prob * weights[3]
            + ada_prob * weights[4]
            + xrp_prob * weights[5]
        )
    except Exception as e:
        _debug(f"AI market-bias fallback to neutral: {e.__class__.__name__}: {str(e).strip()}")
        behaviour_prob = 0.5
        behaviour_weight_mode = "equal"
    behaviour_prob = float(max(0.0, min(1.0, behaviour_prob)))
    major_probs = [btc_prob, eth_prob, bnb_prob, sol_prob, ada_prob, xrp_prob]
    # Determine behaviour direction from the combined probability
    behaviour_dir = direction_from_prob(float(behaviour_prob))
    # Map behaviour direction to a label for display and choose colour.  We
    # reuse the POSITIVE/NEGATIVE/WARNING colours defined above.
    behaviour_side = direction_key(behaviour_dir)
    if behaviour_side == "UPSIDE":
        behaviour_label = "Upside"
        behaviour_color = POSITIVE
    elif behaviour_side == "DOWNSIDE":
        behaviour_label = "Downside"
        behaviour_color = NEGATIVE
    else:
        behaviour_label = "Neutral"
        behaviour_color = WARNING

    # Composite market score (0-100): Direction + Regime + Breadth + Trust
    direction_score = float(max(0.0, min(100.0, abs(float(behaviour_prob) - 0.5) * 200.0)))
    major_upsides = sum(1 for p in major_probs if float(p) >= AI_LONG_THRESHOLD)
    major_downsides = sum(1 for p in major_probs if float(p) <= AI_SHORT_THRESHOLD)
    breadth_score = float(max(major_upsides, major_downsides) / max(len(major_probs), 1) * 100.0)

    if mcap_feed_ok and pd.notna(delta_mcap):
        mcap_chg = abs(float(delta_mcap))
        # Continuous regime scoring to avoid jumpy mode transitions.
        if mcap_chg <= 1.5:
            regime_score = 72.0 + (mcap_chg / 1.5) * 10.0
        elif mcap_chg <= 4.0:
            regime_score = 82.0 - ((mcap_chg - 1.5) / 2.5) * 24.0
        else:
            regime_score = 58.0 - (mcap_chg - 4.0) * 4.0
        regime_score = float(max(38.0, min(90.0, regime_score)))
        regime_score_fallback = False
    else:
        # Neutral fallback when market-cap regime input is unavailable.
        regime_score = 50.0
        regime_score_fallback = True

    try:
        spread = float(pd.Series(major_probs).std())
    except Exception as e:
        _debug(f"Trust-score spread fallback used: {e.__class__.__name__}: {str(e).strip()}")
        spread = 0.18
    trust_score = float(max(0.0, min(100.0, 78.0 - spread * 100.0)))
    if direction_score < 25:
        trust_score = min(trust_score, 55.0)

    composite_score = (
        0.35 * direction_score
        + 0.20 * regime_score
        + 0.25 * breadth_score
        + 0.20 * trust_score
    )
    composite_score = float(max(0.0, min(100.0, composite_score)))
    composite_mode = (
        "Risk-On" if composite_score >= 68 else ("Selective" if composite_score >= 52 else "Risk-Off")
    )
    composite_color = POSITIVE if composite_mode == "Risk-On" else (WARNING if composite_mode == "Selective" else NEGATIVE)
    ai_bias_tip = (
        "Dominance-weighted ML direction across BTC/ETH/BNB/SOL/ADA/XRP. "
        "Direction signal only."
    )
    if behaviour_weight_mode == "equal":
        ai_bias_tip += " Dominance feed unavailable: equal-weight backup read is active."

    def _score_tone(v: float) -> tuple[str, str]:
        x = float(v)
        if x >= 68:
            return ("Strong", POSITIVE)
        if x >= 52:
            return ("Moderate", WARNING)
        return ("Weak", NEGATIVE)

    def _dom_state(v: float | None, low_cut: float, high_cut: float) -> tuple[str, str]:
        if v is None or pd.isna(v):
            return ("N/A", TEXT_MUTED)
        x = float(v)
        if x >= high_cut:
            return ("High", POSITIVE)
        if x >= low_cut:
            return ("Balanced", WARNING)
        return ("Low", NEGATIVE)

    def _clip_pct(v: float | None) -> float:
        if v is None or pd.isna(v):
            return 0.0
        return float(max(0.0, min(100.0, float(v))))

    def _polar(cx: float, cy: float, r: float, angle_deg: float) -> tuple[float, float]:
        rad = math.radians(angle_deg)
        return cx + r * math.cos(rad), cy + r * math.sin(rad)

    def _arc_path(cx: float, cy: float, r: float, start_deg: float, end_deg: float) -> str:
        x1, y1 = _polar(cx, cy, r, start_deg)
        x2, y2 = _polar(cx, cy, r, end_deg)
        large_arc = 1 if abs(end_deg - start_deg) > 180 else 0
        sweep = 1 if end_deg > start_deg else 0
        return f"M {x1:.2f} {y1:.2f} A {r:.2f} {r:.2f} 0 {large_arc} {sweep} {x2:.2f} {y2:.2f}"

    def _orbital_svg_html(
        *,
        value: float | None,
        segments: list[tuple[float, float, str]],
        marker_color: str,
        accent_color: str,
    ) -> str:
        cx, cy, r = 120.0, 104.0, 80.0
        start_deg, end_deg = 150.0, 390.0
        base_path = _arc_path(cx, cy, r, start_deg, end_deg)
        seg_paths = "".join(
            f"<path d='{_arc_path(cx, cy, r, start_deg + (seg_start / 100.0) * (end_deg - start_deg), start_deg + (seg_end / 100.0) * (end_deg - start_deg))}' "
            f"stroke='{color}' stroke-width='12.5' stroke-linecap='round' fill='none' />"
            for seg_start, seg_end, color in segments
        )
        inner_glow = f"<path d='{base_path}' stroke='{accent_color}' stroke-width='3' stroke-linecap='round' fill='none' opacity='0.16' />"
        marker_html = ""
        if value is not None and not pd.isna(value):
            angle = start_deg + (_clip_pct(value) / 100.0) * (end_deg - start_deg)
            mx, my = _polar(cx, cy, r, angle)
            marker_html = (
                f"<circle cx='{mx:.2f}' cy='{my:.2f}' r='6.25' fill='{marker_color}' stroke='rgba(3,8,15,0.95)' stroke-width='2' />"
                f"<circle cx='{mx:.2f}' cy='{my:.2f}' r='10.5' fill='none' stroke='{marker_color}' stroke-width='2' opacity='0.16' />"
            )
        return (
            "<svg class='market-orbital-svg' viewBox='0 0 240 184' preserveAspectRatio='xMidYMid meet' aria-hidden='true'>"
            "<defs>"
            "<linearGradient id='orbitalShell' x1='0%' x2='100%' y1='0%' y2='0%'>"
            "<stop offset='0%' stop-color='rgba(255,255,255,0.04)' />"
            "<stop offset='100%' stop-color='rgba(255,255,255,0.01)' />"
            "</linearGradient>"
            "</defs>"
            f"<path d='{base_path}' stroke='rgba(255,255,255,0.06)' stroke-width='14.5' stroke-linecap='round' fill='none' />"
            f"{inner_glow}"
            f"{seg_paths}"
            f"{marker_html}"
            "</svg>"
        )

    def _stat_metric(label: str, value: float, tone_color: str, tip_text: str) -> str:
        return (
            f"<div class='market-statline-item' title='{html.escape(tip_text, quote=True)}'>"
            f"<span class='market-statline-label'>{html.escape(label)}</span>"
            f"<span class='market-statline-value' style='color:{tone_color};'>{float(value):.0f}</span>"
            f"</div>"
        )

    def _render_market_orbital_card(
        *,
        title: str,
        title_hover: str,
        value_text: str,
        unit: str = "",
        chart_html: str,
        guide_labels: tuple[str, str, str],
        note: str,
        footer_html: str = "",
        top_meta_text: str = "",
        top_meta_color: str = "",
        top_meta_hover: str = "",
    ) -> str:
        unit_html = f"<span class='market-top-unit'>{html.escape(unit)}</span>" if unit else ""
        footer = f"<div class='market-top-footer'>{footer_html}</div>" if footer_html else ""
        left_label, mid_label, right_label = [html.escape(lbl) for lbl in guide_labels]
        top_meta_html = ""
        if top_meta_text:
            top_meta_html = (
                f"<span class='market-orbital-topmeta' title='{html.escape(top_meta_hover, quote=True)}' "
                f"style='border:1px solid {top_meta_color}; color:{top_meta_color};'>{html.escape(top_meta_text)}</span>"
            )
        return (
            "<div class='market-orbital-card'>"
            "<div class='market-orbital-title-row'>"
            f"<div class='market-orbital-title' title='{html.escape(title_hover, quote=True)}'>{html.escape(title)}</div>"
            f"{top_meta_html}"
            "</div>"
            "<div class='market-orbital-stage'>"
            f"{chart_html}"
            "<div class='market-orbital-center'>"
            f"<div class='market-orbital-value-row'><div class='market-orbital-value'>{html.escape(value_text)}</div>{unit_html}</div>"
            "</div>"
            "</div>"
            f"<div class='market-orbital-guides'><span>{left_label}</span><span>{mid_label}</span><span>{right_label}</span></div>"
            f"<div class='market-orbital-note'>{note}</div>"
            f"{footer}"
            "</div>"
        )
    market_signal_cards_placeholder = st.container()

    # Divider
    st.markdown("\n\n")

    # Top coin scanner controls
    st.markdown(
        f"<div class='market-section-title' style='color:{ACCENT};'>Market Setup Radar</div>",
        unsafe_allow_html=True,
    )
    st.markdown(
        """
        <style>
        div[data-testid="stButton"] > button {
          white-space: nowrap;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    def _apply_custom_coin_input() -> None:
        _apply_market_custom_input_state(st.session_state, limit=10)

    _consume_market_custom_clear(st.session_state)
    custom_bases_applied = _normalize_custom_bases(list(st.session_state.get("market_custom_bases_applied", [])))
    custom_mode_active = bool(custom_bases_applied)

    controls = st.columns([1.02, 1.08, 1.08, 0.82, 1.42, 0.92], gap="medium")
    with controls[0]:
        # Persist the selected timeframe in session state so the market
        # prediction card updates when this value changes.  The key ensures
        # the selection is stored under 'market_timeframe'.
        timeframe = st.selectbox(
            "Timeframe",
            ['5m', '15m', '1h', '4h', '1d'],
            index=2,
            key="market_timeframe",
            help="Controls candle delta, tactical levels, scalp timing, and selected-timeframe setup checks.",
        )
    with controls[1]:
        direction_filter = st.selectbox(
            "Direction",
            ['Upside', 'Downside', 'Both'],
            index=2,
            format_func=lambda x: "All Directions" if x == "Both" else x,
            help="Filter table candidates by intended direction. All Directions shows both upside and downside reads.",
        )
    with controls[2]:
        current_scan_mode = str(st.session_state.get("market_scan_mode") or "").strip()
        if current_scan_mode not in {_SCAN_MODE_BROAD, _SCAN_MODE_EMERGING, _SCAN_MODE_TRENDING}:
            normalized_scan_mode = _normalize_scan_mode(current_scan_mode)
            if normalized_scan_mode in {_SCAN_MODE_ACTIONABLE, _SCAN_MODE_EMERGING}:
                st.session_state["market_scan_mode"] = _SCAN_MODE_EMERGING
            elif normalized_scan_mode == _SCAN_MODE_TRENDING:
                st.session_state["market_scan_mode"] = _SCAN_MODE_TRENDING
            else:
                st.session_state["market_scan_mode"] = _SCAN_MODE_BROAD
        scan_mode = st.selectbox(
            "Scan Mode",
            [_SCAN_MODE_BROAD, _SCAN_MODE_EMERGING, _SCAN_MODE_TRENDING],
            index=0,
            key="market_scan_mode",
            disabled=custom_mode_active,
            help="Broad Market is the clean liquid-universe read. Breakout Radar hunts early acceleration. Trending Coins starts from attention, momentum, and volume anomaly candidates.",
        )
    with controls[3]:
        top_n_default = int(st.session_state.get("market_top_n", 10))
        top_n = st.slider(
            "Top N",
            min_value=3,
            max_value=50,
            value=top_n_default,
            key="market_top_n",
            disabled=custom_mode_active,
            help="How many ranked rows to show. Disabled in Custom Coins mode because the watchlist size controls the table.",
        )
    with controls[4]:
        custom_coin_input = st.text_input(
            "Custom Coins (max 10)",
            value=st.session_state.get("market_custom_coin_input", ""),
            key="market_custom_coin_input",
            placeholder="BTC, ETH, SOL",
            help="Optional watchlist mode. Enter up to 10 symbols separated by comma, then press Enter or Scan.",
            on_change=_apply_custom_coin_input,
        )
    with controls[5]:
        run_scan = st.button("Scan", type="primary", width="stretch")
        clear_custom = st.button(
            "Clear",
            width="stretch",
            disabled=not bool(st.session_state.get("market_custom_bases_applied", [])),
            key="market_clear_custom",
        )

    custom_bases_draft = _parse_market_custom_bases(custom_coin_input, limit=10)
    if run_scan:
        custom_bases_draft = _apply_market_custom_input_state(
            st.session_state,
            raw_value=custom_coin_input,
            limit=10,
        )
        custom_bases_applied = list(custom_bases_draft)
        custom_mode_active = bool(custom_bases_applied)
    if clear_custom:
        _queue_market_custom_clear(st.session_state)
        st.rerun()

    if custom_mode_active:
        preview = ", ".join(custom_bases_applied[:6])
        more = "" if len(custom_bases_applied) <= 6 else f" +{len(custom_bases_applied) - 6}"
        st.markdown(
            f"<div class='market-note-box' style='border:1px solid rgba(0,212,255,0.34); border-left:4px solid {ACCENT}; "
            f"background:rgba(0,212,255,0.06); color:{TEXT_MUTED}; margin-top:0.25rem;'>"
            f"<b style='color:{ACCENT};'>Watchlist Mode:</b> reading {len(custom_bases_applied)} coin(s): "
            f"{preview}{more}. Top N is disabled while custom mode is active."
            f"</div>",
            unsafe_allow_html=True,
        )
    elif custom_coin_input.strip() and custom_bases_draft:
        st.caption("Custom symbols are ready. Press Enter or Scan to apply watchlist mode.")
    elif _normalize_scan_mode(scan_mode) == _SCAN_MODE_EMERGING:
        st.caption("Breakout Radar uses the same table, reaches deeper into active-liquidity names, and looks for earlier acceleration. Expect more noise than Broad Market.")
    elif _normalize_scan_mode(scan_mode) == _SCAN_MODE_TRENDING:
        st.caption("Trending Coins uses the same table, but starts from search trend, 24h momentum, and volume-anomaly candidates.")

    exclude_stables = st.toggle(
        "Exclude stablecoins",
        value=True,
        key="market_exclude_stables",
        help="Hide stable/synthetic USD-pegged coins from the market universe.",
    )
    CACHE_TTL_MINUTES = 15
    gate_min_rr, gate_min_adx, gate_min_confidence = scalp_gate_thresholds(timeframe)

    def _fmt_price(v: float) -> str:
        try:
            p = float(v)
        except Exception:
            return ""
        if p >= 1000:
            return f"${p:,.2f}"
        if p >= 1:
            return f"${p:,.4f}"
        if p >= 0.01:
            return f"${p:,.6f}"
        if p >= 0.0001:
            return f"${p:,.8f}"
        return f"${p:,.10f}"

    def _build_scalp_display_payload(
        *,
        timeframe_value: str | None,
        scalp_direction: str | None,
        signal_direction: str | None,
        rr_ratio: float | None,
        adx_val: float | None,
        confidence: float | None,
        conviction_label: str | None,
        entry: float | None,
        stop: float | None,
        target: float | None,
        setup_confirm: str | None = None,
        market_trade_gate_key: str | None = None,
        archive_guardrail_penalty: float | None = None,
        archive_guardrail_label: str | None = None,
        direction_value: str | None = None,
        ai_aligned: bool | None = None,
        scan_focus_value: str | None = None,
        breakout_note: str = "",
        close_ref: float | None = None,
        ema_ref: float | None = None,
    ) -> dict[str, object]:
        scalp_calibration_snapshot = build_scalp_calibration_snapshot(
            scalp_calibration_model,
            signal={
                "Setup Confirm": str(setup_confirm or ""),
                "AI Alignment": "Aligned" if bool(ai_aligned) else "Not aligned",
                "Timeframe": str(timeframe_value or "Unknown"),
                "Scan Focus": str(scan_focus_value or "Unknown"),
                "Direction": str(direction_value or signal_direction or ""),
            },
        )
        gate_min_rr_local, gate_min_adx_local, gate_min_confidence_local = scalp_gate_thresholds(timeframe_value)
        gate_pass, gate_reason = scalp_quality_gate(
            scalp_direction=scalp_direction,
            signal_direction=signal_direction,
            rr_ratio=rr_ratio,
            adx_val=adx_val,
            confidence=confidence,
            conviction_label=conviction_label,
            entry=entry,
            stop=stop,
            target=target,
            min_rr=gate_min_rr_local,
            min_adx=gate_min_adx_local,
            min_confidence=gate_min_confidence_local,
            timeframe=timeframe_value,
            setup_confirm=setup_confirm,
            market_trade_gate_key=market_trade_gate_key,
            archive_guardrail_penalty=archive_guardrail_penalty,
            archive_guardrail_label=archive_guardrail_label,
        )
        gate_pass, gate_reason = apply_scalp_archive_calibration(
            gate_pass,
            gate_reason,
            calibration_delta=float(getattr(scalp_calibration_snapshot, "delta", 0.0) or 0.0),
            rr_ratio=rr_ratio,
            adx_val=adx_val,
            confidence=confidence,
            timeframe=timeframe_value,
        )
        reason_text = scalp_reason_text(
            gate_reason,
            timeframe=timeframe_value,
            min_rr=gate_min_rr_local,
            min_adx=gate_min_adx_local,
            min_confidence=gate_min_confidence_local,
        )
        scalp_calibration_note = str(getattr(scalp_calibration_snapshot, "note", "") or "")
        if scalp_calibration_note:
            reason_text = f"{reason_text} {scalp_calibration_note}".strip() if reason_text else scalp_calibration_note
        show_blocked = bool(scalp_direction) and str(gate_reason or "").upper() not in {
            "NO_SCALP_DIRECTION",
            "SIGNAL_DIRECTION_NEUTRAL",
            "UNSUPPORTED_TIMEFRAME",
        }
        blocked_short_reason = scalp_reason_short_label(gate_reason) if show_blocked and not gate_pass else ""
        display_state = "LIVE" if gate_pass else ("CONDITIONAL" if show_blocked else "NONE")
        scalp_label = (
            direction_label(scalp_direction or "")
            if gate_pass
            else (
                direction_label(scalp_direction or "")
                if show_blocked
                else ""
            )
        )
        show_levels = bool(entry and stop and target) and display_state in {"LIVE", "CONDITIONAL"}
        entry_note = ""
        if show_levels and entry and close_ref and ema_ref:
            try:
                base_dir = direction_label(scalp_direction or "")
                entry_note = (
                    f"{base_dir} entry guide built from the latest close, EMA5, and an ATR buffer. "
                    f"Close {_fmt_price(float(close_ref))} • EMA5 {_fmt_price(float(ema_ref))}"
                )
            except Exception:
                entry_note = ""
        target_note = str(breakout_note or "").strip() if show_levels else ""
        rr_note = ""
        if show_levels and target_note:
            rr_note = "This reward-to-risk depends on the target condition shown in the target tooltip."
        if display_state == "CONDITIONAL":
            reference_prefix = "Reference only while scalp veto is active."
            entry_note = f"{reference_prefix} {entry_note}".strip() if entry_note else reference_prefix
            target_note = f"{reference_prefix} {target_note}".strip() if target_note else reference_prefix
            rr_note = f"{reference_prefix} {reason_text}".strip()
        return {
            "pass": bool(gate_pass),
            "display_state": display_state,
            "reason": str(gate_reason or ""),
            "reason_text": reason_text,
            "reason_short": blocked_short_reason,
            "label": scalp_label,
            "entry_display": _fmt_price(entry) if show_levels and entry else "",
            "entry_val": float(entry) if show_levels and entry else None,
            "stop_display": _fmt_price(stop) if show_levels and stop else "",
            "stop_val": float(stop) if show_levels and stop else None,
            "target_display": _fmt_price(target) if show_levels and target else "",
            "target_val": float(target) if show_levels and target else None,
            "rr_badge": _rr_badge(rr_ratio if show_levels else 0.0),
            "rr_val": float(rr_ratio) if show_levels and rr_ratio else None,
            "entry_note": entry_note,
            "target_note": target_note,
            "rr_note": rr_note,
            "calibration_delta": float(getattr(scalp_calibration_snapshot, "delta", 0.0) or 0.0),
            "calibration_note": scalp_calibration_note,
        }

    def _normalize_indicator_label(v: object) -> str:
        raw = str(v or "").strip()
        if not raw or raw in {"Unavailable", "N/A", "nan"}:
            return "N/A"
        clean = (
            raw.replace("🟢 ", "")
            .replace("🔴 ", "")
            .replace("🟡 ", "")
            .replace("▲▲ ", "")
            .replace("▲ ", "")
            .replace("▼ ", "")
            .replace("→ ", "")
            .replace("- ", "")
            .replace("– ", "")
            .strip()
        )
        clean_upper = clean.upper()
        if "NEAR TOP" in clean_upper:
            return "▼ Near Top"
        if "NEAR BOTTOM" in clean_upper:
            return "▲ Near Bottom"
        if "NEAR VWAP" in clean_upper:
            return "- Near VWAP"
        if "BULLISH" in clean_upper or clean_upper in {"ABOVE", "OVERSOLD", "LOW"}:
            return f"▲ {clean}"
        if "BEARISH" in clean_upper or clean_upper in {"BELOW", "OVERBOUGHT", "HIGH"}:
            return f"▼ {clean}"
        if any(k in clean_upper for k in ["NEUTRAL", "MODERATE", "STARTING", "INDECISION", "MIXED"]):
            return f"- {clean}"
        return clean

    def _compact_adx_label(v: object) -> str:
        raw = str(v or "").strip()
        if not raw or raw.upper() in {"N/A", "NA", "NAN", "UNAVAILABLE"}:
            return ""
        # format_adx output examples:
        # "▼ 17.9 (Weak)", "→ 23.4 (Starting)", "▲ 34.8 (Strong)",
        # "▲▲ 58.1 (Very Strong)", "🔥 79.3 (Extreme)"
        m = re.search(r"\(([^)]+)\)", raw)
        if not m:
            return raw
        bucket = m.group(1).strip()
        if raw.startswith("▲▲"):
            return f"▲▲ {bucket}"
        if raw.startswith("▲"):
            return f"▲ {bucket}"
        if raw.startswith("▼"):
            return f"▼ {bucket}"
        if raw.startswith("→"):
            return f"- {bucket}"
        if raw.startswith("🔥"):
            return f"🔥 {bucket}"
        return bucket

    def _rr_badge(rr_val: float) -> str:
        if rr_val <= 0:
            return ""
        return f"{rr_val:.2f}"

    def _setup_confirm_display(raw_action: str, action_reason: str | None = None, direction: str | None = None) -> str:
        return _shared_setup_confirm_display(raw_action, action_reason=action_reason, direction=direction)
    def _setup_confirm_class(value: str) -> str:
        return _setup_confirm_class_key(value)

    def _setup_confirm_rank(value: str) -> int:
        return _setup_confirm_priority(value)

    def _tone_for_text(text: str, *, neutral_tone: str = "muted") -> str:
        s = str(text).strip().upper()
        if not s or s in {"N/A", "NA", "NAN", "UNAVAILABLE", "-"}:
            return "muted"
        s = (
            s.replace("🟢 ", "")
            .replace("🔴 ", "")
            .replace("🟡 ", "")
            .replace("⚪ ", "")
            .replace("✅ ", "")
            .replace("⏳ ", "")
            .replace("⛔ ", "")
        ).strip()
        if s.startswith("▲"):
            return "pos"
        if s.startswith("▼"):
            return "neg"
        if s.startswith("→"):
            return "warn"
        if any(
            k in s
            for k in [
                "ENTER", "UPSIDE", "ALIGNED", "GOOD", "STRONG", "VERY STRONG", "EXTREME",
                "ABOVE", "BULLISH", "OVERSOLD", "NEAR BOTTOM",
            ]
        ):
            return "pos"
        if any(
            k in s
            for k in [
                "SKIP", "DOWNSIDE", "CONFLICT", "WEAK", "BEARISH",
                "OVERBOUGHT", "BELOW", "NEAR TOP",
            ]
        ):
            return "neg"
        if "PROBE" in s:
            return "warn"
        if "WATCH" in s:
            return "info"
        if any(k in s for k in ["WAIT", "MIXED", "EARLY", "TREND", "NEUTRAL", "MEDIUM", "STARTING", "MODERATE", "SPIKE"]):
            return "warn"
        return neutral_tone

    def _tone_for_col(col: str, text: str) -> str:
        s = str(text or "").strip().upper()
        s = (
            s.replace("🟢 ", "")
            .replace("🔴 ", "")
            .replace("🟡 ", "")
            .replace("⚪ ", "")
            .replace("✅ ", "")
            .replace("⏳ ", "")
            .replace("⛔ ", "")
        ).strip()
        if not s or s in {"N/A", "NA", "NAN", "UNAVAILABLE", "-"}:
            return "muted"

        if col == "Setup Confirm":
            cls = _setup_confirm_class(s)
            if cls in {"ENTER_TREND_AI", "ENTER_TREND_LED", "ENTER_AI_LED"}:
                return "pos"
            if cls == "PROBE":
                return "warn"
            if cls == "WATCH":
                return "info"
            if cls == "SKIP":
                return "neg"
            return "warn"

        if col == "Direction":
            if "UPSIDE" in s:
                return "pos"
            if "DOWNSIDE" in s:
                return "neg"
            return "muted"

        if col in {"Confidence", "AI Confidence"}:
            if "HIGH" in s:
                return "pos"
            if "MEDIUM" in s:
                return "warn"
            if "VERY LOW" in s or "LOW" in s:
                return "neg"
            return "warn"

        if col == "R:R":
            try:
                rr = float(
                    s.replace("🟢", "")
                    .replace("🟡", "")
                    .replace("🔴", "")
                    .strip()
                )
                if rr >= 2.0:
                    return "pos"
                if rr >= 1.5:
                    return "warn"
                return "neg"
            except Exception:
                return "muted"

        if col == "Scalp Opportunity":
            if "UPSIDE" in s:
                return "pos"
            if "DOWNSIDE" in s:
                return "neg"
            return "muted"

        if col == "AI Ensemble":
            if s.startswith("UPSIDE"):
                return "pos"
            if s.startswith("DOWNSIDE"):
                return "neg"
            return "muted"

        if col == "Volatility":
            if "LOW" in s:
                return "pos"
            if "MODERATE" in s or "NEUTRAL" in s:
                return "muted"
            if "HIGH" in s or "EXTREME" in s:
                return "neg"
            return "muted"

        if col == "Spike Alert":
            return "warn" if "SPIKE" in s else "muted"

        if col == "ADX":
            if "EXTREME" in s or "VERY STRONG" in s or "STRONG" in s:
                return "pos"
            if "STARTING" in s:
                return "warn"
            if "WEAK" in s:
                return "neg"
            return "muted"

        if col in {"SuperTrend", "Ichimoku", "VWAP", "PSAR"}:
            if "BULLISH" in s or "ABOVE" in s:
                return "pos"
            if "BEARISH" in s or "BELOW" in s:
                return "neg"
            return "muted"

        if col in {"Bollinger", "Stochastic RSI", "Williams %R", "CCI"}:
            if "OVERSOLD" in s or "NEAR BOTTOM" in s or "LOW" in s:
                return "pos"
            if "OVERBOUGHT" in s or "NEAR TOP" in s or "HIGH" in s:
                return "neg"
            return "muted"

        if col == "Candle Pattern":
            if s.startswith("▲"):
                return "pos"
            if s.startswith("▼"):
                return "neg"
            if s.startswith("→") or s.startswith("-"):
                return "muted"
            # Compatibility fallback when arrow prefixes are missing.
            if "BULLISH" in s:
                return "pos"
            if "BEARISH" in s:
                return "neg"
            if "NEUTRAL" in s or "INDECISION" in s:
                return "muted"
            return "muted"

        return _tone_for_text(s)

    def _tone_css_class(tone_key: str) -> str:
        return {
            "pos": "mk-tone-pos",
            "neg": "mk-tone-neg",
            "warn": "mk-tone-warn",
            "info": "mk-tone-info",
            "muted": "mk-tone-muted",
        }.get(tone_key, "mk-tone-muted")

    def _chip(
        text: object,
        tone: str | None = None,
        title: str | None = None,
        extra_class: str = "",
    ) -> str:
        raw = "" if text is None else str(text).strip()
        if not raw or raw.upper() in {"N/A", "NA", "NAN", "UNAVAILABLE", "-"}:
            return ""
        tone_key = tone or _tone_for_text(raw)
        tone_map = {
            "pos": "mk-pos",
            "neg": "mk-neg",
            "warn": "mk-warn",
            "muted": "mk-muted",
            "info": "mk-info",
        }
        cls = tone_map.get(tone_key, "mk-muted")
        title_attr = f" title='{html.escape(title)}'" if title else ""
        cls_full = f"mk-chip {cls} {extra_class}".strip()
        return f"<span class='{cls_full}'{title_attr}>{html.escape(raw)}</span>"

    def _scalp_chip(text: str, row: dict, *, title_override: str | None = None) -> str:
        raw = str(text or "").strip()
        if not raw or raw.upper() in {"N/A", "NA", "NAN", "UNAVAILABLE", "-", "NEUTRAL"}:
            return ""
        tone_key = _tone_for_col("Scalp Opportunity", raw)
        display_state = str(row.get("__scalp_display_state", "")).strip().upper()
        reason_text = str(title_override or row.get("__scalp_reason_text", "")).strip()
        if not reason_text and display_state == "LIVE":
            reason_text = "Live scalp passed the current intraday setup, market stance, and quality checks."
        tone_cls = _tone_css_class(tone_key)
        state_cls = "mk-scalp-live" if display_state == "LIVE" else "mk-scalp-conditional"
        state_symbol = "✓" if display_state == "LIVE" else "!"
        title_attr = f" title='{html.escape(reason_text, quote=True)}'" if reason_text else ""
        return (
            f"<span class='mk-scalp-wrap {tone_cls} {state_cls}'{title_attr}>"
            f"<span class='mk-scalp-main'>"
            f"<span class='mk-scalp-label'>{html.escape(raw)}</span>"
            f"</span>"
            f"<span class='mk-scalp-state'>{html.escape(state_symbol)}</span>"
            f"</span>"
        )

    def _score_metric(
        text: object,
        *,
        tone: str,
        title: str | None = None,
        extra_class: str = "",
    ) -> str:
        raw = "" if text is None else str(text).strip()
        if not raw or raw.upper() in {"N/A", "NA", "NAN", "UNAVAILABLE", "-"}:
            return ""
        score = _confidence_value_from_badge(raw)
        if score is None:
            score = 0.0
        label = _extract_confidence_label(raw)
        meter_pct = max(8.0, min(100.0, float(score)))
        title_attr = f" title='{html.escape(title, quote=True)}'" if title else ""
        cls = f"mk-score-wrap {_tone_css_class(tone)} {extra_class}".strip()
        return (
            f"<span class='{cls}'{title_attr}>"
            f"<span class='mk-score-topline'>"
            f"<span class='mk-score-value'>{score:.0f}</span>"
            f"<span class='mk-score-unit'>%</span>"
            f"<span class='mk-score-label'>{html.escape(label)}</span>"
            f"</span>"
            f"<span class='mk-score-track'><span class='mk-score-fill' style='width:{meter_pct:.0f}%'></span></span>"
            f"</span>"
        )

    def _ensemble_metric(text: object, *, tone: str, votes: int, title: str | None = None) -> str:
        raw = "" if text is None else str(text).strip()
        if not raw or raw.upper() in {"N/A", "NA", "NAN", "UNAVAILABLE", "-"}:
            return ""
        title_attr = f" title='{html.escape(title, quote=True)}'" if title else ""
        cls = f"mk-ensemble-wrap {_tone_css_class(tone)}".strip()
        votes_n = max(0, min(3, int(votes)))
        dots_html = "".join(
            f"<span class='mk-ai-dot{' is-filled' if i < votes_n else ''}'></span>"
            for i in range(3)
        )
        return (
            f"<span class='{cls}'{title_attr}>"
            f"<span class='mk-ensemble-text'>{html.escape(raw)}</span>"
            f"<span class='mk-ai-dots'>{dots_html}</span>"
            f"</span>"
        )

    def _direction_metric(text: object, *, tone: str, title: str | None = None) -> str:
        raw = "" if text is None else str(text).strip()
        if not raw or raw.upper() in {"N/A", "NA", "NAN", "UNAVAILABLE", "-"}:
            return ""
        marker = "-"
        if tone == "pos":
            marker = "▲"
        elif tone == "neg":
            marker = "▼"
        title_attr = f" title='{html.escape(title, quote=True)}'" if title else ""
        cls = f"mk-direction-wrap {_tone_css_class(tone)}".strip()
        return (
            f"<span class='{cls}'{title_attr}>"
            f"<span class='mk-direction-marker'>{html.escape(marker)}</span>"
            f"<span class='mk-direction-text'>{html.escape(raw)}</span>"
            f"</span>"
        )

    def _rr_metric(
        text: object,
        *,
        tone: str,
        title: str | None = None,
        conditional: bool = False,
    ) -> str:
        raw = "" if text is None else str(text).strip()
        if not raw or raw.upper() in {"N/A", "NA", "NAN", "UNAVAILABLE", "-"}:
            return ""
        title_attr = f" title='{html.escape(title, quote=True)}'" if title else ""
        cls = f"mk-rr-wrap {_tone_css_class(tone)}".strip()
        if conditional:
            cls += " mk-rr-conditional"
        suffix = "*" if title else ""
        return (
            f"<span class='{cls}'{title_attr}>"
            f"<span class='mk-rr-value'>{html.escape(raw)}{suffix}</span>"
            f"</span>"
        )

    def _indicator_metric(
        text: object,
        *,
        tone: str,
        title: str | None = None,
        extra_class: str = "",
    ) -> str:
        raw = "" if text is None else str(text).strip()
        if not raw or raw.upper() in {"N/A", "NA", "NAN", "UNAVAILABLE", "-"}:
            return ""
        clean = _strip_indicator_prefix(raw)
        glyph = ""
        if raw.startswith("▲▲"):
            glyph = "▲▲"
        elif raw.startswith("▲"):
            glyph = "▲"
        elif raw.startswith("▼"):
            glyph = "▼"
        elif raw.startswith("→") or raw.startswith("-"):
            glyph = "-"
        elif raw.startswith("🔥"):
            glyph = "✦"
        title_attr = f" title='{html.escape(title, quote=True)}'" if title else ""
        cls = f"mk-indicator-wrap {_tone_css_class(tone)} {extra_class}".strip()
        if not glyph and tone == "muted":
            glyph = "-"
        marker_html = (
            f"<span class='mk-indicator-glyph'>{html.escape(glyph)}</span>"
            if glyph
            else "<span class='mk-indicator-dot'></span>"
        )
        return (
            f"<span class='{cls}'{title_attr}>"
            f"{marker_html}"
            f"<span class='mk-indicator-text'>{html.escape(clean)}</span>"
            f"</span>"
        )

    def _strip_indicator_prefix(value: object) -> str:
        raw = str(value or "").strip()
        for prefix in ("▲▲ ", "▲ ", "▼ ", "→ ", "- ", "🔥 ", "– "):
            if raw.startswith(prefix):
                return raw[len(prefix):].strip()
        return raw

    def _column_header_tooltip(col: str) -> str:
        return {
            "Coin": "Asset ticker. Hover the row value to see the feed pair or backup source when one is used.",
            "Price ($)": "Latest closed-candle price from the active data feed.",
            "Δ (%)": "Move from the previous closed candle to the latest closed candle on your selected timeframe.",
            "Setup Confirm": copy_text("market.tooltip.setup_confirm"),
            "Direction": "Main higher-timeframe technical bias from closed anchor candles.",
            "Confidence": "How strong and trustworthy that Direction call is.",
            "AI Ensemble": "Higher-timeframe AI view. Dots show how many models agree.",
            "AI Confidence": "How trustworthy the AI verdict is.",
            "R:R": "Reward-to-risk ratio for the separate scalp timing lens, not the main Setup Confirm verdict.",
            "Entry Price": copy_text("market.tooltip.entry_price"),
            "Stop Loss": copy_text("market.tooltip.stop_loss"),
            "Target Price": copy_text("market.tooltip.target_price"),
            "Scalp Opportunity": copy_text("market.tooltip.scalp_opportunity"),
            "Market Cap ($)": "Size and liquidity context for the asset.",
            "ADX": "Trend strength indicator. It measures how strong the move is, not whether it is up or down.",
            "SuperTrend": "ATR-based trend follower showing whether price still behaves bullish or bearish.",
            "Ichimoku": "Cloud-based trend context that blends trend, momentum, and support-resistance structure.",
            "VWAP": "Shows whether price is trading above, below, or near its volume-weighted average price.",
            "PSAR": "Parabolic SAR trend indicator. It flips when short-term trend pressure changes.",
            "Stochastic RSI": "Shows whether short-term momentum is near the top, middle, or bottom of its recent range.",
            "Williams %R": "Momentum oscillator showing whether price is near the top or bottom of its recent range.",
            "CCI": "Shows whether price is stretched above or below its recent average.",
            "Candle Pattern": "Most recent candlestick pattern label from the latest candles.",
            "Bollinger": "Shows where price sits inside its volatility bands: near the top, middle, or bottom.",
            "Volatility": "ATR-style volatility regime. It tells you whether price movement is quiet or active.",
            "Spike Alert": "Flags unusual volume activity compared with recent candles.",
        }.get(col, "")

    def _indicator_cell_title(col: str, row: dict, txt: str) -> str:
        clean = _strip_indicator_prefix(txt)
        upper = clean.upper()
        if col == "SuperTrend":
            if "BULLISH" in upper:
                return "SuperTrend is bullish. Price is still holding above its trailing trend line."
            if "BEARISH" in upper:
                return "SuperTrend is bearish. Price is still holding below its trailing trend line."
            return "SuperTrend is neutral. The trend follower is not leaning clearly either way."
        if col == "Ichimoku":
            detail = str(row.get("__ichimoku_detail", "")).strip().replace(" | ", " • ")
            if detail:
                return f"Ichimoku reads {clean.lower()}. {detail}."
            return f"Ichimoku reads {clean.lower()}. This is the cloud-based trend context."
        if col == "VWAP":
            if "ABOVE" in upper:
                return "Price is trading above VWAP, which usually means buyers still have intraday control."
            if "BELOW" in upper:
                return "Price is trading below VWAP, which usually means sellers still have intraday control."
            return "Price is near VWAP, so intraday control looks balanced."
        if col == "PSAR":
            if "BULLISH" in upper:
                return "PSAR is bullish. The trailing stop dots still support upside trend pressure."
            if "BEARISH" in upper:
                return "PSAR is bearish. The trailing stop dots still support downside trend pressure."
            return "PSAR is neutral right now."
        if col == "Bollinger":
            if "TOP" in upper or "OVERBOUGHT" in upper:
                return "Price is near the upper band, so the move may be stretched on the upside."
            if "BOTTOM" in upper or "OVERSOLD" in upper:
                return "Price is near the lower band, so the move may be stretched on the downside."
            return "Price is sitting around the middle of its volatility bands."
        if col == "Stochastic RSI":
            if "HIGH" in upper or "OVERBOUGHT" in upper:
                return "Short-term momentum is near the top of its recent range."
            if "LOW" in upper or "OVERSOLD" in upper:
                return "Short-term momentum is near the bottom of its recent range."
            return "Short-term momentum is balanced in the middle of its recent range."
        if col == "Volatility":
            if "LOW" in upper:
                return "Volatility is low. Price movement is relatively quiet right now."
            if "HIGH" in upper or "EXTREME" in upper:
                return "Volatility is high. Price movement is relatively active and fast right now."
            return f"Volatility reads {clean.lower()}. This describes the current activity level of price movement."
        if col == "Williams %R":
            if "OVERBOUGHT" in upper or "HIGH" in upper:
                return "Williams %R says price is near the top of its recent range and may be stretched."
            if "OVERSOLD" in upper or "LOW" in upper:
                return "Williams %R says price is near the bottom of its recent range and may be stretched."
            return "Williams %R is neutral, so momentum is not stretched at either edge of the range."
        if col == "CCI":
            if "OVERBOUGHT" in upper or "HIGH" in upper:
                return "CCI says price is stretched above its recent average."
            if "OVERSOLD" in upper or "LOW" in upper:
                return "CCI says price is stretched below its recent average."
            return "CCI is neutral, so price is not far from its recent average."
        if col == "Candle Pattern":
            if txt.startswith("▲"):
                return f"Latest candle pattern is bullish: {clean}."
            if txt.startswith("▼"):
                return f"Latest candle pattern is bearish: {clean}."
            return f"Latest candle pattern is neutral or indecisive: {clean}."
        return clean

    def _direction_anchor_label(note: str) -> str:
        match = re.search(r"\(([^)]+)\):", str(note or ""))
        return str(match.group(1)).strip() if match else "the higher-timeframe anchor pair"

    def _setup_confirm_cell_title(row: dict, display_txt: str) -> str:
        raw_action = str(row.get("__action_raw", display_txt) or "").strip()
        cls = normalize_action_class(raw_action)
        reason_code = str(row.get("__action_reason", "") or "").strip()
        reason_text = action_reason_text(reason_code)
        setup_calibration_note = str(row.get("__setup_calibration_note", "") or "").strip()
        meaning_map = {
            "ENTER": "Fully confirmed. This is the strongest setup class in the table.",
            "PROBE": "Promising, but still small-risk only.",
            "WATCH": "Interesting, but timing still needs more proof.",
            "SKIP": "Too weak, conflicted, or poorly located right now.",
        }
        if cls.startswith("ENTER_"):
            meaning = meaning_map["ENTER"]
        else:
            meaning = meaning_map.get(cls, "Current market read for this setup.")
        parts = [f"{display_txt}: {meaning}"]
        if reason_text:
            parts.append(f"Why: {_compact_hover_note(reason_text, limit=100)}")
        if reason_code.startswith("ARCHIVE_") and setup_calibration_note:
            parts.append(f"Archive: {_compact_hover_note(setup_calibration_note, limit=90)}")
        return " ".join(parts)

    def _direction_cell_title(row: dict, txt: str) -> str:
        clean = str(txt or "").strip() or "Neutral"
        anchor_label = _direction_anchor_label(str(row.get("__direction_note", "") or ""))
        meaning = {
            "Upside": "The broader technical trend still leans up.",
            "Downside": "The broader technical trend still leans down.",
            "Neutral": "The broader technical read is mixed, so there is no strong directional read yet.",
        }.get(clean, "This is the broader technical bias from the market scan.")
        return f"Direction from {anchor_label}: {clean}. {meaning}"

    def _confidence_cell_title(row: dict, txt: str) -> str:
        score = _confidence_value_from_badge(txt) or 0.0
        label = _extract_confidence_label(txt)
        archive_note = str(row.get("__setup_calibration_note", "") or "").strip()
        base = f"Trust score for Direction: {score:.0f}% {label}. Higher means the technical bias looks more reliable."
        if archive_note:
            return f"{base} Archive read: {_compact_hover_note(archive_note, limit=95)}"
        return base

    def _ai_ensemble_cell_title(row: dict, txt: str) -> str:
        verdict = str(txt or "").strip() or "Neutral"
        votes = row.get("__ai_votes")
        try:
            votes_n = max(0, min(3, int(votes)))
        except Exception:
            votes_n = 0
        meaning = {
            "Upside": "The higher-timeframe AI stack leans up.",
            "Downside": "The higher-timeframe AI stack leans down.",
            "Neutral": "The higher-timeframe AI stack does not see a clear directional read.",
        }.get(verdict, "This is the higher-timeframe AI verdict.")
        ai_note = str(row.get("__ai_note", "") or "").strip()
        parts = [f"Higher-timeframe AI verdict: {verdict}. {meaning}", f"Model agreement: {votes_n}/3."]
        ai_note_l = ai_note.lower()
        if ai_note and any(token in ai_note_l for token in ("fallback", "safety", "partial", "incomplete")):
            parts.append(_compact_hover_note(ai_note, limit=90))
        return " ".join(parts)

    def _ai_confidence_cell_title(row: dict, txt: str) -> str:
        score = _confidence_value_from_badge(txt) or 0.0
        label = _extract_confidence_label(txt)
        votes = row.get("__ai_votes")
        try:
            votes_n = max(0, min(3, int(votes)))
        except Exception:
            votes_n = 0
        agreement_hint = (
            "Model agreement is strong."
            if votes_n >= 3
            else "Model agreement is mixed." if votes_n == 2 else "Model agreement is weak."
        )
        return (
            f"Trust score for the AI verdict: {score:.0f}% {label}. "
            f"{agreement_hint}"
        )

    def _rr_cell_title(row: dict, txt: str) -> str:
        try:
            rr = float(str(txt or "").replace("*", "").strip())
        except Exception:
            rr = None
        state = str(row.get("__scalp_display_state", "") or "").strip().upper()
        if rr is None:
            base = "Scalp-only reward-to-risk estimate."
        elif rr >= 2.0:
            base = f"Scalp-only reward-to-risk estimate: {rr:.2f}. Potential reward is meaningfully larger than risk."
        elif rr >= 1.0:
            base = f"Scalp-only reward-to-risk estimate: {rr:.2f}. Potential reward is only slightly larger than risk."
        else:
            base = f"Scalp-only reward-to-risk estimate: {rr:.2f}. Potential reward is smaller than risk."
        if state == "CONDITIONAL":
            base = f"{base} Reference only: this scalp idea is not fully approved."
        rr_note = str(row.get("__rr_note", "") or "").strip()
        if rr_note:
            base = f"{base} {_compact_hover_note(rr_note, limit=90)}"
        return base

    def _scalp_cell_title(row: dict, txt: str) -> str:
        direction = str(txt or "").strip() or "Neutral"
        state = str(row.get("__scalp_display_state", "") or "").strip().upper()
        reason_text = str(row.get("__scalp_reason_text", "") or "").strip()
        if state == "LIVE":
            base = f"Short-term scalp view: {direction}. This setup passed the current intraday checks."
        else:
            base = f"Short-term scalp view: {direction}. A scalp idea exists, but a veto is still active."
        if reason_text:
            base = f"{base} Why: {_compact_hover_note(reason_text, limit=95)}"
        return base

    def _price_cell_title(row: dict, txt: str) -> str:
        pair = str(row.get("__pair", "") or "").strip()
        base = "Latest closed price on the selected timeframe."
        if pair:
            return f"{base} Feed pair: {pair}."
        return base

    def _entry_cell_title(row: dict) -> str:
        state = str(row.get("__scalp_display_state", "") or "").strip().upper()
        entry_note = str(row.get("__entry_note", "") or "").strip()
        base = (
            "Suggested scalp entry price."
            if state != "CONDITIONAL"
            else "Reference scalp entry only. The scalp idea is still conditional."
        )
        if entry_note:
            return f"{base} {_compact_hover_note(entry_note, limit=95)}"
        return base

    def _stop_cell_title(row: dict, txt: str) -> str:
        state = str(row.get("__scalp_display_state", "") or "").strip().upper()
        base = (
            f"Protective stop for the scalp lens at {txt}. If price hits this level, the setup is invalidated."
            if txt
            else "Protective stop for the scalp lens."
        )
        if state == "CONDITIONAL":
            return f"{base} Reference only because the scalp idea is not fully approved."
        return base

    def _target_cell_title(row: dict) -> str:
        state = str(row.get("__scalp_display_state", "") or "").strip().upper()
        target_note = str(row.get("__target_note", "") or "").strip()
        base = (
            "Suggested scalp target price."
            if state != "CONDITIONAL"
            else "Reference scalp target only. The scalp idea is still conditional."
        )
        if target_note:
            return f"{base} {_compact_hover_note(target_note, limit=95)}"
        return base

    def _market_cap_cell_title(txt: str) -> str:
        return (
            f"Approximate market size: {txt}. Larger market caps are usually easier to trade cleanly."
            if txt
            else "Approximate market size and liquidity context."
        )

    def _render_cell(col: str, row: dict) -> str:
        val = row.get(col, "")
        txt = "" if val is None else str(val).strip()
        if txt.upper() in {"N/A", "NA", "NAN", "UNAVAILABLE", "-"}:
            txt = ""
        if col == "Coin":
            pair = str(row.get("__pair", "")).strip()
            pair_meta = _coin_pair_meta(txt, pair)
            emerging_label = str(row.get("__emerging_label", "")).strip()
            emerging_note = str(row.get("__emerging_note", "")).strip()
            emerging_direction = str(row.get("__emerging_direction", ""))
            emerging_tone = _emerging_badge_tone(emerging_direction)
            tone_map = {
                "pos": "mk-pos",
                "neg": "mk-neg",
                "warn": "mk-warn",
                "muted": "mk-muted",
                "info": "mk-info",
            }
            badge_html = ""
            if emerging_label:
                badge_cls = tone_map.get(emerging_tone, "mk-muted")
                badge_symbol = _emerging_badge_symbol(emerging_direction)
                badge_text = _emerging_badge_text(emerging_direction)
                badge_title_text = emerging_label if not emerging_note else f"{emerging_label} | {emerging_note}"
                badge_title = f" title='{html.escape(badge_title_text)}'"
                badge_html = (
                    f"<span class='mk-coin-badge {badge_cls}'{badge_title}>"
                    f"<span class='mk-em-key'>{html.escape(badge_text)}</span>"
                    f"<span class='mk-em-arrow'>{html.escape(badge_symbol)}</span>"
                    f"</span>"
                )
            pair_meta_html = (
                f"<span class='mk-coin-meta'>{html.escape(pair_meta)}</span>"
                if pair_meta
                else ""
            )
            if pair:
                return (
                    f"<span class='mk-coin-wrap'>"
                    f"<span class='mk-coin-top'>"
                    f"<span class='mk-coin'>{html.escape(txt)}</span>"
                    f"{badge_html}"
                    f"</span>"
                    f"{pair_meta_html}"
                    f"<span class='mk-coin-tooltip'>{html.escape(pair)}</span>"
                    f"</span>"
                )
            return (
                f"<span class='mk-coin-wrap'>"
                f"<span class='mk-coin-top'>"
                f"<span class='mk-coin'>{html.escape(txt)}</span>"
                f"{badge_html}"
                f"</span>"
                f"</span>"
            )
        if col in {"Setup Confirm", "Direction", "Confidence", "AI Confidence", "R:R", "Scalp Opportunity"}:
            if col == "Setup Confirm":
                reason_code = str(row.get("__action_reason", "")).strip()
                raw_action = str(row.get("__action_raw", txt))
                display_txt = _setup_confirm_display(raw_action, reason_code, direction=str(row.get("Direction", "")).strip())
                sc_cls = _setup_confirm_class(raw_action or txt)
                extra_cls = "mk-chip-action"
                if sc_cls == "ENTER_TREND_LED":
                    extra_cls += " mk-sc-trend-led"
                elif sc_cls == "ENTER_AI_LED":
                    extra_cls += " mk-sc-ai-led"
                elif sc_cls == "PROBE":
                    extra_cls += " mk-sc-probe"
                elif sc_cls == "WATCH":
                    extra_cls += " mk-sc-watch"
                title_txt = _setup_confirm_cell_title(row, display_txt)
                return _chip(
                    display_txt,
                    _tone_for_col(col, raw_action or txt),
                    title=title_txt,
                    extra_class=extra_cls,
                )
            if col == "Direction":
                return _direction_metric(
                    txt,
                    tone=_tone_for_col(col, txt),
                    title=_direction_cell_title(row, txt) if txt else None,
                )
            if col == "Confidence":
                return _score_metric(
                    txt,
                    tone=_tone_for_col(col, txt),
                    title=_confidence_cell_title(row, txt) if txt else None,
                    extra_class="mk-score-confidence",
                )
            if col == "AI Confidence":
                return _score_metric(
                    txt,
                    tone=_tone_for_col(col, txt),
                    title=_ai_confidence_cell_title(row, txt) if txt else None,
                    extra_class="mk-score-ai",
                )
            if col == "R:R":
                return _rr_metric(
                    txt,
                    tone=_tone_for_col(col, txt),
                    title=_rr_cell_title(row, txt) if txt else None,
                    conditional=str(row.get("__scalp_display_state", "")).strip().upper() == "CONDITIONAL",
                )
            if col == "Scalp Opportunity":
                return _scalp_chip(txt, row, title_override=_scalp_cell_title(row, txt))
            extra_cls = ""
            return _chip(
                txt,
                _tone_for_col(col, txt),
                title=None,
                extra_class=extra_cls,
            )
        if col == "AI Ensemble":
            t = _tone_for_col(col, txt)
            base_txt = re.sub(r"\s*\(\s*\d+\s*/\s*3\s*\)\s*$", "", txt).strip() or txt
            votes = row.get("__ai_votes")
            try:
                votes_n = int(votes)
            except Exception:
                m = re.search(r"\((\d)\s*/\s*3\)", txt)
                votes_n = int(m.group(1)) if m else 0
            votes_n = max(0, min(3, votes_n))
            return _ensemble_metric(
                base_txt,
                tone=t,
                votes=votes_n,
                title=_ai_ensemble_cell_title(row, base_txt),
            )
        if col == "Spike Alert":
            if not txt:
                return ""
            spike_dir = str(row.get("__spike_dir", "")).upper()
            if spike_dir == "UP":
                spike_tone = "pos"
                spike_label = "▲ Up Spike"
            elif spike_dir == "DOWN":
                spike_tone = "neg"
                spike_label = "▼ Down Spike"
            else:
                spike_tone = "warn"
                spike_label = "→ Spike"
            detail_parts: list[str] = []
            spike_vol_ratio = row.get("__spike_vol_ratio")
            spike_candle_pct = row.get("__spike_candle_pct")
            spike_vwap_ctx = str(row.get("__spike_vwap_ctx", "")).strip()
            try:
                if pd.notna(spike_vol_ratio):
                    detail_parts.append(f"Volume vs recent average: {float(spike_vol_ratio):.2f}x")
            except Exception:
                pass
            try:
                if pd.notna(spike_candle_pct):
                    detail_parts.append(f"Candle move: {float(spike_candle_pct):+,.2f}%")
            except Exception:
                pass
            if spike_vwap_ctx:
                detail_parts.append(f"VWAP context: {spike_vwap_ctx}")
            detail_title = " • ".join(detail_parts) if detail_parts else "Unusual volume activity detected"
            return _indicator_metric(
                spike_label,
                tone=spike_tone,
                title=detail_title,
                extra_class="mk-indicator-spike",
            )
        if col == "Δ (%)":
            if not txt:
                return ""
            delta_note = str(row.get("__delta_note", "")).strip()
            title_attr = f" title='{html.escape(delta_note, quote=True)}'" if delta_note else ""
            if txt.startswith("▲"):
                return f"<span class='mk-delta mk-pos-t mk-num-strong'{title_attr}>{html.escape(txt)}</span>"
            if txt.startswith("▼"):
                return f"<span class='mk-delta mk-neg-t mk-num-strong'{title_attr}>{html.escape(txt)}</span>"
            return f"<span class='mk-delta mk-muted-t mk-num-strong'{title_attr}>{html.escape(txt)}</span>"
        if col == "Price ($)":
            if not txt:
                return ""
            plain = txt[1:] if txt.startswith("$") else txt
            return (
                f"<span class='mk-plain mk-num mk-price' title='{html.escape(_price_cell_title(row, txt), quote=True)}'>"
                f"{html.escape(plain)}</span>"
            )
        if col == "Entry Price":
            if not txt:
                return ""
            entry_note = _entry_cell_title(row)
            level_cls = "mk-plain mk-num mk-level"
            if str(row.get("__scalp_display_state", "")).strip().upper() == "CONDITIONAL":
                level_cls += " mk-level-conditional"
            if entry_note:
                return (
                    f"<span class='{level_cls}' title='{html.escape(entry_note, quote=True)}'>"
                    f"{html.escape(txt)}</span>"
                )
            return f"<span class='{level_cls}'>{html.escape(txt)}</span>"
        if col == "Stop Loss":
            if not txt:
                return ""
            stop_note = _stop_cell_title(row, txt)
            level_cls = "mk-plain mk-num mk-level"
            if str(row.get("__scalp_display_state", "")).strip().upper() == "CONDITIONAL":
                level_cls += " mk-level-conditional"
            return f"<span class='{level_cls}' title='{html.escape(stop_note, quote=True)}'>{html.escape(txt)}</span>"
        if col == "Target Price":
            if not txt:
                return ""
            target_note = _target_cell_title(row)
            level_cls = "mk-plain mk-num mk-level"
            if str(row.get("__scalp_display_state", "")).strip().upper() == "CONDITIONAL":
                level_cls += " mk-level-conditional"
            if target_note:
                return (
                    f"<span class='{level_cls}' title='{html.escape(target_note, quote=True)}'>"
                    f"{html.escape(txt)}</span>"
                )
            return f"<span class='{level_cls}'>{html.escape(txt)}</span>"
        if col == "Ichimoku":
            ichi_title = str(row.get("__ichimoku_detail", "")).strip()
            ichi_summary = _indicator_cell_title(col, row, txt) if txt else ""
            ichi_full_title = (
                f"{ichi_summary} {ichi_title.replace(' | ', ' • ')}".strip()
                if ichi_title and ichi_summary
                else (ichi_title or ichi_summary)
            )
            return _indicator_metric(
                txt,
                tone=_tone_for_col(col, txt),
                title=ichi_full_title or None,
                extra_class="mk-indicator-ichi",
            ) if txt else ""
        if col == "ADX":
            adx_raw = row.get("__adx_raw")
            adx_title = None
            try:
                if pd.notna(adx_raw):
                    adx_f = float(adx_raw)
                    if adx_f >= float(gate_min_adx):
                        adx_title = f"ADX {adx_f:.1f}. Trend strength looks usable for scalp setups."
                    else:
                        adx_title = f"ADX {adx_f:.1f}. Trend strength still looks weak for scalp setups."
            except Exception:
                adx_title = None
            return _indicator_metric(
                txt,
                tone=_tone_for_col(col, txt),
                title=adx_title,
                extra_class="mk-indicator-adx",
            ) if txt else ""
        if col in {"SuperTrend", "Ichimoku", "VWAP", "Bollinger", "Stochastic RSI", "Volatility", "PSAR", "Williams %R", "CCI", "Candle Pattern"}:
            return _indicator_metric(
                txt,
                tone=_tone_for_col(col, txt),
                title=_compact_hover_note(_indicator_cell_title(col, row, txt), limit=120),
                extra_class="mk-indicator-generic",
            ) if txt else ""
        if col == "Market Cap ($)":
            return (
                f"<span class='mk-plain mk-num' title='{html.escape(_market_cap_cell_title(txt), quote=True)}'>"
                f"{html.escape(txt)}</span>"
            ) if txt else ""
        return f"<span class='mk-plain'>{html.escape(txt)}</span>"

    def _render_pro_table(df: pd.DataFrame, cols: list[str]) -> None:
        sticky_order: list[str] = ["Coin"]
        core_signal_cols = {"Setup Confirm", "Direction", "Confidence", "AI Ensemble", "AI Confidence"}
        trend_cols = {"ADX", "SuperTrend", "Ichimoku", "VWAP", "PSAR"}
        momentum_cols = {"Stochastic RSI", "Williams %R", "CCI", "Candle Pattern"}
        volatility_volume_cols = {"Bollinger", "Volatility", "Spike Alert"}
        col_widths = {
            "Coin": 182,
            "Price ($)": 122,
            "Δ (%)": 92,
            "Setup Confirm": 160,
            "Direction": 130,
            "Confidence": 150,
            "AI Ensemble": 170,
            "AI Confidence": 155,
        }
        left_offsets: dict[str, str] = {}
        running_left = 0
        for c in sticky_order:
            left_offsets[c] = f"{running_left}px"
            running_left += col_widths[c]
        sticky_cols = set(sticky_order)

        group_row_html = ""
        core_row_html = ""
        has_advanced_cols = any(c not in primary_cols for c in cols)
        core_members = [c for c in cols if c in core_signal_cols]
        setup_snapshot_title = (
            "Main decision area: setup verdict, higher-timeframe direction, confidence, and AI view."
        )
        group_titles = {
            "Trend Structure": "Trend-following indicators that describe structure and directional control.",
            "Momentum Signals": "Momentum and reversal-style indicators showing whether the move is stretching or improving.",
            "Volatility & Volume": "Context indicators showing activity level, volume anomalies, and band location.",
        }
        if has_advanced_cols:
            first_core_idx = cols.index(core_members[0]) if core_members else 0
            last_core_idx = cols.index(core_members[-1]) if core_members else -1
            group_cells = []

            if first_core_idx > 0:
                group_cells.append(f"<th colspan='{first_core_idx}' class='mk-core-gap'></th>")
            if core_members:
                core_span = (last_core_idx - first_core_idx) + 1
                group_cells.append(
                    f"<th colspan='{core_span}' class='mk-core-cell' title='{html.escape(setup_snapshot_title, quote=True)}'>"
                    "Setup Snapshot</th>"
                )

            trailing_primary = max(len(primary_cols) - (last_core_idx + 1), 0)
            if trailing_primary > 0:
                group_cells.append(f"<th colspan='{trailing_primary}' class='mk-core-gap'></th>")

            group_defs = [
                ("Trend Structure", [c for c in cols if c in trend_cols], "trend"),
                ("Momentum Signals", [c for c in cols if c in momentum_cols], "momentum"),
                ("Volatility & Volume", [c for c in cols if c in volatility_volume_cols], "context"),
            ]
            for label, members, tone in group_defs:
                if not members:
                    continue
                group_cells.append(
                    f"<th colspan='{len(members)}' class='mk-group-{tone}' "
                    f"title='{html.escape(group_titles.get(label, ''), quote=True)}'>{html.escape(label)}</th>"
                )
            if group_cells:
                group_row_html = f"<tr class='mk-group-row'>{''.join(group_cells)}</tr>"
        elif core_members:
            first_core_idx = cols.index(core_members[0])
            last_core_idx = cols.index(core_members[-1])
            left_span = first_core_idx
            core_span = (last_core_idx - first_core_idx) + 1
            right_span = len(cols) - last_core_idx - 1
            core_cells = []
            if left_span > 0:
                core_cells.append(f"<th colspan='{left_span}' class='mk-core-gap'></th>")
            core_cells.append(
                f"<th colspan='{core_span}' class='mk-core-cell' title='{html.escape(setup_snapshot_title, quote=True)}'>"
                "Setup Snapshot</th>"
            )
            if right_span > 0:
                core_cells.append(f"<th colspan='{right_span}' class='mk-core-gap'></th>")
            core_row_html = f"<tr class='mk-core-row'>{''.join(core_cells)}</tr>"

        header_html = []
        for c in cols:
            sticky = ""
            width_style = ""
            header_title = _column_header_tooltip(c)
            title_attr = f" title='{html.escape(header_title, quote=True)}'" if header_title else ""
            if c in col_widths:
                w = col_widths[c]
                width_style = f"min-width:{w}px; max-width:{w}px; width:{w}px;"
            if c in sticky_cols:
                sticky = (
                    f"position:sticky; left:{left_offsets[c]}; z-index:7; "
                    f"background:linear-gradient(180deg, rgba(18,24,36,0.99), rgba(12,18,30,0.99)); "
                    f"box-shadow: 1px 0 0 rgba(148,163,184,0.16);"
                )
            header_html.append(f"<th{title_attr} style='{width_style}{sticky}'>{html.escape(c)}</th>")

        rows_html = []
        for _, r in df.iterrows():
            row_dict = r.to_dict()
            row_action = str(row_dict.get("__action_raw", row_dict.get("Setup Confirm", "")) or "")
            row_action_cls = _setup_confirm_class(row_action)
            row_class = ""
            if row_action_cls in {"ENTER_TREND_AI", "ENTER_TREND_LED", "ENTER_AI_LED"}:
                row_class = "mk-row-ready"
            elif row_action_cls == "PROBE":
                row_class = "mk-row-probe"
            elif row_action_cls == "WATCH":
                row_class = "mk-row-watch"
            elif row_action_cls == "SKIP":
                row_class = "mk-row-skip"
            cell_html = []
            for c in cols:
                sticky = ""
                width_style = ""
                cell_classes: list[str] = []
                if c in col_widths:
                    w = col_widths[c]
                    width_style = f"min-width:{w}px; max-width:{w}px; width:{w}px;"
                if c in sticky_cols:
                    sticky = (
                        f"position:sticky; left:{left_offsets[c]}; z-index:6; "
                        f"background:rgba(8,12,20,1.0); box-shadow:1px 0 0 rgba(148,163,184,0.22), 2px 0 10px rgba(0,0,0,0.24);"
                    )
                if c == "Coin":
                    cell_classes.append("mk-coin-cell")
                cell_class_attr = f" class='{' '.join(cell_classes)}'" if cell_classes else ""
                cell_html.append(f"<td{cell_class_attr} style='{width_style}{sticky}'>{_render_cell(c, row_dict)}</td>")
            row_class_attr = f" class='{row_class}'" if row_class else ""
            rows_html.append(f"<tr{row_class_attr}>" + "".join(cell_html) + "</tr>")

        st.markdown(
            f"""
            <style>
            .scan-kpi-value {{
              color:#F8FAFC;
              font-family:'Space Grotesk','Manrope',sans-serif;
              font-size:2rem;
              font-weight:700;
              letter-spacing:0.2px;
              margin-top:4px;
              line-height:1.1;
            }}
            .scan-kpi-sub {{
              color:{TEXT_MUTED};
              font-size:0.84rem;
              margin-top:8px;
              letter-spacing:0.15px;
            }}
            .mk-wrap {{
              width:100%;
              overflow-x:auto;
              border:1px solid rgba(148,163,184,0.18);
              border-radius:14px;
              background:linear-gradient(180deg, rgba(7,11,18,0.98), rgba(5,9,15,0.98));
              box-shadow:0 12px 28px rgba(0,0,0,0.30), inset 0 0 0 1px rgba(255,255,255,0.02);
            }}
            .mk-table {{
              width:max-content;
              min-width:100%;
              border-collapse:separate;
              border-spacing:0;
              font-size:0.82rem;
              font-family:'Manrope','Segoe UI',sans-serif;
            }}
            .mk-table th {{
              text-align:left;
              padding:10px 10px;
              color:rgba(191,211,230,0.78);
              font-weight:700;
              letter-spacing:0.18px;
              border-bottom:1px solid rgba(148,163,184,0.22);
              border-right:1px solid rgba(148,163,184,0.08);
              white-space:nowrap;
              top:0;
              position:sticky;
              z-index:4;
              background:linear-gradient(180deg, rgba(18,24,36,0.98), rgba(13,18,28,0.98));
            }}
            .mk-group-row th {{
              position:static;
              top:auto;
              z-index:1;
              padding:7px 10px 6px;
              font-size:0.56rem;
              font-weight:850;
              letter-spacing:0.18em;
              text-transform:uppercase;
              color:rgba(191,211,230,0.58);
              background:linear-gradient(180deg, rgba(10,16,26,0.98), rgba(8,13,22,0.98));
              border-bottom:1px solid rgba(148,163,184,0.14);
              border-right:1px solid rgba(148,163,184,0.06);
            }}
            .mk-group-core {{
              box-shadow: inset 0 -1px 0 rgba(148,163,184,0.18);
            }}
            .mk-group-trend {{
              color:rgba(191,211,230,0.66);
              box-shadow: inset 0 -1px 0 rgba(125,211,252,0.10);
            }}
            .mk-group-momentum {{
              color:rgba(191,211,230,0.66);
              box-shadow: inset 0 -1px 0 rgba(253,224,71,0.10);
            }}
            .mk-group-context {{
              color:rgba(191,211,230,0.66);
              box-shadow: inset 0 -1px 0 rgba(244,114,182,0.10);
            }}
            .mk-core-row th {{
              position:static;
              top:auto;
              z-index:1;
              padding:6px 10px 5px;
              font-size:0.55rem;
              font-weight:860;
              letter-spacing:0.18em;
              text-transform:uppercase;
              background:linear-gradient(180deg, rgba(9,14,23,0.98), rgba(8,12,21,0.98));
              border-bottom:1px solid rgba(148,163,184,0.10);
              border-right:1px solid rgba(148,163,184,0.04);
            }}
            .mk-core-gap {{
              color:transparent;
            }}
            .mk-core-cell {{
              text-align:left;
              color:rgba(191,211,230,0.72);
              box-shadow:
                inset 0 1px 0 rgba(255,255,255,0.03),
                inset 0 -1px 0 rgba(94,234,212,0.14);
            }}
            .mk-header-row th {{
              top:0;
              z-index:4;
            }}
            .mk-table td {{
              padding:8px 10px;
              color:#E5E7EB;
              border-bottom:1px solid rgba(148,163,184,0.12);
              border-right:1px solid rgba(148,163,184,0.07);
              white-space:nowrap;
              vertical-align:middle;
              overflow:hidden;
              text-overflow:ellipsis;
              transition:
                background-color 0.16s ease,
                border-color 0.16s ease,
                box-shadow 0.16s ease;
            }}
            .mk-table tbody tr:nth-child(odd) td {{
              background-color:rgba(255,255,255,0.012);
            }}
            .mk-table tbody tr:nth-child(even) td {{
              background-color:rgba(255,255,255,0.004);
            }}
            .mk-table tr:hover td {{
              background-color:rgba(148,163,184,0.050);
              border-bottom-color:rgba(148,163,184,0.22);
            }}
            .mk-table tr:hover td[style*="position:sticky"] {{
              background:linear-gradient(180deg, rgba(11,18,29,1.0), rgba(8,14,24,1.0)) !important;
            }}
            .mk-table tr:hover td.mk-coin-cell {{
              box-shadow:
                inset 2px 0 0 rgba(0,212,255,0.24),
                1px 0 0 rgba(148,163,184,0.22),
                2px 0 10px rgba(0,0,0,0.24) !important;
            }}
            .mk-table tbody tr.mk-row-ready td.mk-coin-cell {{
              box-shadow:
                inset 3px 0 0 rgba(0,255,136,0.42),
                1px 0 0 rgba(148,163,184,0.22),
                2px 0 10px rgba(0,0,0,0.24) !important;
            }}
            .mk-table tbody tr.mk-row-probe td.mk-coin-cell {{
              box-shadow:
                inset 3px 0 0 rgba(255,209,102,0.36),
                1px 0 0 rgba(148,163,184,0.22),
                2px 0 10px rgba(0,0,0,0.24) !important;
            }}
            .mk-table tbody tr.mk-row-watch td.mk-coin-cell {{
              box-shadow:
                inset 3px 0 0 rgba(125,211,252,0.30),
                1px 0 0 rgba(148,163,184,0.22),
                2px 0 10px rgba(0,0,0,0.24) !important;
            }}
            .mk-table tbody tr.mk-row-skip td.mk-coin-cell {{
              box-shadow:
                inset 3px 0 0 rgba(255,51,102,0.28),
                1px 0 0 rgba(148,163,184,0.22),
                2px 0 10px rgba(0,0,0,0.24) !important;
            }}
            .mk-chip {{
              display:inline-flex;
              align-items:center;
              gap:6px;
              min-height:22px;
              padding:2px 8px;
              max-width:100%;
              border-radius:999px;
              border:1px solid rgba(148,163,184,0.18);
              background:rgba(15,23,36,0.76);
              font-size:0.73rem;
              font-weight:760;
              overflow:hidden;
              text-overflow:ellipsis;
              white-space:nowrap;
              box-sizing:border-box;
              box-shadow:none;
            }}
            .mk-chip-wrap {{
              position:relative;
              display:inline-flex;
              align-items:center;
              max-width:100%;
              z-index:1;
              cursor:help;
            }}
            .mk-chip-wrap:hover {{
              z-index:60;
            }}
            .mk-chip-wrap:hover .mk-chip-tooltip {{
              opacity:1;
              visibility:visible;
              transform:translateY(-50%) translateX(0);
            }}
            .mk-chip-tooltip {{
              position:absolute;
              left:calc(100% + 10px);
              top:50%;
              transform:translateY(-50%) translateX(4px);
              z-index:40;
              max-width:340px;
              padding:8px 10px;
              border-radius:10px;
              border:1px solid rgba(0,212,255,0.30);
              background:rgba(6,12,24,0.96);
              color:#D6E8FF;
              font-size:0.71rem;
              font-weight:700;
              line-height:1.3;
              white-space:normal;
              box-shadow:0 8px 20px rgba(0,0,0,0.35);
              opacity:0;
              visibility:hidden;
              transition:opacity 0.14s ease, visibility 0.14s ease, transform 0.14s ease;
              pointer-events:none;
            }}
            .mk-chip-action {{
              min-height:23px;
              font-size:0.70rem;
              padding:2px 8px;
              font-weight:820;
              letter-spacing:0.02em;
              gap:4px;
              background:rgba(14,20,31,0.92);
            }}
            .mk-scalp-wrap {{
              display:inline-flex;
              align-items:center;
              gap:8px;
              min-height:22px;
              max-width:100%;
            }}
            .mk-scalp-main {{
              display:inline-flex;
              align-items:center;
              min-width:0;
              line-height:1;
            }}
            .mk-scalp-label {{
              font-size:0.81rem;
              font-weight:800;
              letter-spacing:0.01em;
              color:currentColor;
              white-space:nowrap;
            }}
            .mk-scalp-state {{
              display:inline-flex;
              align-items:center;
              justify-content:center;
              width:16px;
              height:16px;
              border-radius:999px;
              font-size:0.58rem;
              font-weight:900;
              line-height:1;
              flex:0 0 16px;
              box-shadow:inset 0 0 0 1px rgba(255,255,255,0.04);
            }}
            .mk-scalp-live .mk-scalp-state {{
              color:#04120A;
              background:rgba(255,255,255,0.82);
            }}
            .mk-scalp-conditional .mk-scalp-state {{
              color:#FFF4CC;
              background:rgba(255,209,102,0.14);
              border:1px solid rgba(255,209,102,0.24);
            }}
            .mk-scalp-conditional .mk-scalp-line {{
              opacity:0.18;
            }}
            .mk-tone-pos {{ color:{POSITIVE}; }}
            .mk-tone-neg {{ color:{NEGATIVE}; }}
            .mk-tone-warn {{ color:{WARNING}; }}
            .mk-tone-info {{ color:{ACCENT}; }}
            .mk-tone-muted {{ color:rgba(191,211,230,0.74); }}
            .mk-score-wrap {{
              display:inline-flex;
              flex-direction:column;
              gap:5px;
              min-width:108px;
              max-width:100%;
              line-height:1;
            }}
            .mk-score-topline {{
              display:inline-flex;
              align-items:flex-end;
              gap:4px;
              min-width:0;
            }}
            .mk-score-value {{
              color:#F8FAFC;
              font-size:0.98rem;
              font-weight:840;
              letter-spacing:-0.01em;
              font-variant-numeric: tabular-nums;
              font-feature-settings:"tnum" 1, "lnum" 1;
            }}
            .mk-score-unit {{
              font-size:0.62rem;
              font-weight:770;
              letter-spacing:0.04em;
              color:rgba(191,211,230,0.58);
              transform:translateY(-1px);
              margin-right:2px;
            }}
            .mk-score-label {{
              font-size:0.57rem;
              font-weight:780;
              letter-spacing:0.14em;
              text-transform:uppercase;
              color:currentColor;
              opacity:0.88;
            }}
            .mk-score-track {{
              position:relative;
              width:100%;
              max-width:116px;
              height:4px;
              border-radius:999px;
              background:
                repeating-linear-gradient(
                  90deg,
                  rgba(15,23,42,0.00) 0 18px,
                  rgba(15,23,42,0.36) 18px 19px
                ),
                linear-gradient(90deg, rgba(148,163,184,0.18), rgba(148,163,184,0.09));
              box-shadow: inset 0 0 0 1px rgba(255,255,255,0.025);
              overflow:hidden;
            }}
            .mk-score-fill {{
              display:block;
              height:100%;
              border-radius:999px;
              background:currentColor;
              opacity:0.96;
            }}
            .mk-score-confidence .mk-score-value {{
              letter-spacing:-0.02em;
            }}
            .mk-score-ai {{
              opacity:0.92;
            }}
            .mk-score-ai .mk-score-value {{
              font-size:0.91rem;
              color:rgba(248,250,252,0.95);
            }}
            .mk-score-ai .mk-score-unit {{
              font-size:0.58rem;
              color:rgba(191,211,230,0.5);
            }}
            .mk-score-ai .mk-score-label {{
              opacity:0.80;
            }}
            .mk-score-ai .mk-score-track {{
              max-width:108px;
              height:3px;
              background:
                repeating-linear-gradient(
                  90deg,
                  rgba(15,23,42,0.00) 0 18px,
                  rgba(15,23,42,0.28) 18px 19px
                ),
                linear-gradient(90deg, rgba(148,163,184,0.15), rgba(148,163,184,0.07));
            }}
            .mk-ensemble-wrap {{
              display:inline-flex;
              align-items:center;
              gap:8px;
              min-height:22px;
              max-width:100%;
              line-height:1.1;
            }}
            .mk-ensemble-text {{
              font-weight:770;
              letter-spacing:0.01em;
            }}
            .mk-direction-wrap {{
              display:inline-flex;
              align-items:center;
              gap:7px;
              min-height:22px;
              max-width:100%;
              line-height:1.1;
            }}
            .mk-direction-marker {{
              font-size:0.68rem;
              font-weight:860;
              line-height:1;
              opacity:0.94;
            }}
            .mk-direction-text {{
              font-weight:800;
              letter-spacing:0.01em;
            }}
            .mk-rr-wrap {{
              display:inline-flex;
              align-items:center;
              min-height:20px;
              font-variant-numeric: tabular-nums;
              font-feature-settings:"tnum" 1, "lnum" 1;
            }}
            .mk-rr-value {{
              font-weight:820;
              letter-spacing:0.01em;
            }}
            .mk-rr-conditional {{
              opacity:0.72;
            }}
            .mk-indicator-wrap {{
              display:inline-flex;
              align-items:center;
              gap:6px;
              min-height:19px;
              max-width:100%;
              padding:1px 0 2px;
              border-bottom:1px solid rgba(148,163,184,0.12);
              line-height:1.05;
            }}
            .mk-indicator-glyph {{
              font-size:0.68rem;
              font-weight:860;
              letter-spacing:0.02em;
              line-height:1;
              flex:0 0 auto;
              opacity:0.95;
            }}
            .mk-indicator-dot {{
              width:6px;
              height:6px;
              border-radius:999px;
              background:currentColor;
              opacity:0.72;
              flex:0 0 6px;
            }}
            .mk-indicator-text {{
              font-size:0.73rem;
              font-weight:750;
              letter-spacing:0.01em;
              color:currentColor;
              white-space:nowrap;
            }}
            .mk-indicator-adx .mk-indicator-text {{
              font-weight:800;
            }}
            .mk-indicator-spike {{
              border-bottom-color:rgba(255,209,102,0.16);
            }}
            .mk-num {{
              display:inline-block;
              font-variant-numeric: tabular-nums;
              font-feature-settings:"tnum" 1, "lnum" 1;
              letter-spacing:0.01em;
            }}
            .mk-num-strong {{
              font-variant-numeric: tabular-nums;
              font-feature-settings:"tnum" 1, "lnum" 1;
            }}
            .mk-price {{
              color:#F8FAFC;
              font-weight:760;
            }}
            .mk-level {{
              color:#DCE7F5;
              font-weight:700;
            }}
            .mk-level-conditional {{
              color:rgba(220,231,245,0.72);
            }}
            .mk-ai-dots {{
              display:inline-flex;
              align-items:center;
              gap:4px;
              min-height:12px;
            }}
            .mk-ai-dot {{
              width:7px;
              height:7px;
              border-radius:999px;
              border:1px solid currentColor;
              background:transparent;
              opacity:0.32;
              flex:0 0 7px;
            }}
            .mk-ai-dot.is-filled {{
              background:currentColor;
              opacity:0.96;
            }}
            .mk-sc-trend-led {{
              color:#58C4F6 !important;
              border-color:rgba(88,196,246,0.34) !important;
              background:rgba(88,196,246,0.08) !important;
            }}
            .mk-sc-ai-led {{
              color:#2EE6D6 !important;
              border-color:rgba(46,230,214,0.34) !important;
              background:rgba(46,230,214,0.08) !important;
            }}
            .mk-sc-probe {{
              color:#E7D6B0 !important;
              border-color:rgba(231,214,176,0.28) !important;
              background:rgba(231,214,176,0.06) !important;
            }}
            .mk-sc-watch {{
              color:#A7B6C8 !important;
              border-color:rgba(167,182,200,0.24) !important;
              background:rgba(167,182,200,0.05) !important;
            }}
            .mk-pos {{ color:{POSITIVE}; border-color:rgba(0,255,136,0.24); background:rgba(0,255,136,0.045); }}
            .mk-neg {{ color:{NEGATIVE}; border-color:rgba(255,51,102,0.26); background:rgba(255,51,102,0.045); }}
            .mk-warn {{ color:{WARNING}; border-color:rgba(255,209,102,0.24); background:rgba(255,209,102,0.05); }}
            .mk-info {{ color:{ACCENT}; border-color:rgba(0,212,255,0.22); background:rgba(0,212,255,0.045); }}
            .mk-muted {{ color:{TEXT_MUTED}; border-color:rgba(140,161,182,0.18); background:rgba(140,161,182,0.05); }}
            .mk-coin {{ font-weight:800; letter-spacing:0.2px; color:#F8FAFC; }}
            .mk-coin-wrap {{
              position:relative;
              display:inline-flex;
              flex-direction:column;
              align-items:flex-start;
              justify-content:center;
              gap:3px;
            }}
            .mk-coin-top {{
              display:inline-flex;
              align-items:center;
              flex-wrap:wrap;
              gap:7px;
            }}
            .mk-coin-meta {{
              color:rgba(191,211,230,0.62);
              font-size:0.54rem;
              font-weight:650;
              letter-spacing:0.10em;
              text-transform:uppercase;
              line-height:1;
              opacity:0.78;
            }}
            .mk-coin-badge {{
              display:inline-flex;
              align-items:center;
              justify-content:center;
              gap:5px;
              min-width:26px;
              height:21px;
              padding:0 8px 0 9px;
              border-radius:999px;
              border:1px solid rgba(148,163,184,0.28);
              background:
                linear-gradient(180deg, rgba(15,23,42,0.94), rgba(15,23,42,0.78)),
                radial-gradient(circle at top, rgba(255,255,255,0.06), rgba(255,255,255,0.00) 58%);
              font-size:0.72rem;
              font-weight:800;
              letter-spacing:0.01em;
              line-height:1;
              box-shadow:
                inset 0 0 0 1px rgba(255,255,255,0.03),
                0 1px 4px rgba(0,0,0,0.18),
                0 0 0 1px rgba(255,255,255,0.02);
              backdrop-filter: blur(6px);
            }}
            .mk-em-key {{
              font-size:0.49rem;
              font-weight:900;
              letter-spacing:0.18em;
              text-transform:uppercase;
              opacity:0.82;
            }}
            .mk-em-arrow {{
              font-size:0.82rem;
              font-weight:900;
              line-height:1;
              transform:translateY(-0.5px);
            }}
            .mk-coin-badge.mk-pos {{
              border-color:rgba(16,185,129,0.38);
              background:
                linear-gradient(180deg, rgba(4,34,28,0.96), rgba(5,28,23,0.82)),
                radial-gradient(circle at top, rgba(16,185,129,0.11), rgba(16,185,129,0.00) 58%);
              box-shadow:
                inset 0 0 0 1px rgba(255,255,255,0.03),
                0 1px 4px rgba(0,0,0,0.18),
                0 0 0 1px rgba(16,185,129,0.08);
            }}
            .mk-coin-badge.mk-neg {{
              border-color:rgba(244,63,94,0.38);
              background:
                linear-gradient(180deg, rgba(42,12,23,0.96), rgba(34,10,19,0.82)),
                radial-gradient(circle at top, rgba(244,63,94,0.11), rgba(244,63,94,0.00) 58%);
              box-shadow:
                inset 0 0 0 1px rgba(255,255,255,0.03),
                0 1px 4px rgba(0,0,0,0.18),
                0 0 0 1px rgba(244,63,94,0.08);
            }}
            .mk-table td.mk-coin-cell {{
              overflow:visible !important;
              position:relative;
            }}
            .mk-coin-tooltip {{
              position:absolute;
              left:calc(100% + 8px);
              top:50%;
              transform:translateY(-50%);
              z-index:40;
              opacity:0;
              visibility:hidden;
              transition:opacity 0.14s ease, visibility 0.14s ease;
              pointer-events:none;
              white-space:nowrap;
              border:1px solid rgba(0,212,255,0.40);
              background:rgba(6,12,24,0.96);
              color:#D6E8FF;
              font-size:0.70rem;
              font-weight:700;
              line-height:1.2;
              border-radius:8px;
              padding:4px 8px;
              box-shadow:0 8px 20px rgba(0,0,0,0.35);
            }}
            .mk-coin-wrap:hover .mk-coin-tooltip {{
              opacity:1;
              visibility:visible;
            }}
            .mk-plain {{ color:#E5E7EB; }}
            .mk-delta {{ font-weight:700; }}
            .mk-pos-t {{ color:{POSITIVE}; }}
            .mk-neg-t {{ color:{NEGATIVE}; }}
            .mk-muted-t {{ color:{TEXT_MUTED}; }}
            </style>
            <div class="mk-wrap">
              <table class="mk-table">
                <thead>{group_row_html}{core_row_html}<tr class="mk-header-row">{''.join(header_html)}</tr></thead>
                <tbody>{''.join(rows_html)}</tbody>
              </table>
            </div>
            """,
            unsafe_allow_html=True,
        )

    scan_sig = _market_scan_signature(
        timeframe=timeframe,
        direction_filter=direction_filter,
        top_n=int(top_n),
        exclude_stables=bool(exclude_stables),
        custom_bases_applied=custom_bases_applied,
        scan_mode=scan_mode,
    )
    last_sig = st.session_state.get("market_scan_sig")
    last_attempt_ts = str(st.session_state.get(_LAST_SCAN_ATTEMPT_TS_KEY) or "")
    current_source_label = str(st.session_state.get("market_scan_source", "LIVE") or "LIVE")
    should_scan = _should_rescan_market(
        run_scan=bool(run_scan),
        last_sig=last_sig,
        scan_sig=scan_sig,
        has_results_state=("market_scan_results" in st.session_state),
        last_attempt_ts=last_attempt_ts,
        refresh_ttl_minutes=_SCAN_REFRESH_TTL_MINUTES,
        current_source_label=current_source_label,
    )

    results: list[dict] = st.session_state.get("market_scan_results", [])
    source_label = current_source_label
    data_mode = st.session_state.get("market_data_mode", "FULL MARKET MODE")
    scan_degraded = "DEGRADED" in current_source_label.upper()
    live_produced_rows: list[dict] = []
    live_result_count_before_limit = 0
    live_ranked_out_count = 0
    attempted_symbols: set[str] = set()
    skipped_symbols: list[tuple[str, str]] = []
    market_decision_version = current_decision_version("Market")
    data_health_items = list(st.session_state.get(_DATA_HEALTH_ITEMS_KEY, []))
    if should_scan:
        data_health_items = []
        st.session_state[_DATA_HEALTH_ITEMS_KEY] = []

    def _add_data_health_item(tone: str, title: str, body: str) -> None:
        item = {
            "tone": str(tone or "info").strip().lower(),
            "title": str(title or "").strip(),
            "body": str(body or "").strip(),
        }
        if not item["title"] and not item["body"]:
            return
        if item not in data_health_items:
            data_health_items.append(item)

    def _data_health_display_items(base_items: list[dict], *, source_label_text: str, data_mode_text: str) -> list[dict]:
        items = [dict(item) for item in base_items if isinstance(item, dict)]

        def add_once(tone: str, title: str, body: str) -> None:
            item = {"tone": tone, "title": title, "body": body}
            if item not in items:
                items.append(item)

        source_up = str(source_label_text or "").upper()
        mode_up = str(data_mode_text or "").upper()
        if "DEGRADED" in source_up or "PARTIAL" in source_up:
            add_once(
                "warning",
                "Partial Live Read",
                "The fresh scan is usable, but coverage is incomplete.",
            )
        if source_up.startswith("CACHED"):
            add_once(
                "warning",
                "Cached Snapshot",
                "Showing the latest matching saved scan because the fresh read was incomplete.",
            )
        if "PARTIAL ENRICHMENT" in mode_up:
            add_once(
                "warning",
                "Partial Enrichment",
                "Some market-cap fields are missing; candle and indicator reads are still live.",
            )
        if "EXCHANGE-ONLY" in mode_up:
            add_once(
                "warning",
                "Exchange-Only Feed",
                "Exchange candles are live; enrichment fields such as market cap may be blank.",
            )
        if mode_up.startswith("MAJOR BACKUP"):
            add_once(
                "warning",
                "Universe Backup",
                "The broad liquidity list failed, so this is a majors-only backup read.",
            )
        return items

    def _render_data_health_band(*, source_chip: str, source_color: str, source_display_label: str, mode_label: str, mode_color: str, items: list[dict]) -> None:
        warning_items = [item for item in items if str(item.get("tone") or "").lower() == "warning"]
        if not warning_items:
            return
        accent_color = WARNING if warning_items else POSITIVE
        status_text = "Needs Care" if warning_items else "Healthy"
        source_detail = str(source_display_label or "").strip()
        if source_detail.upper() in {"LIVE", "LIVE (PARTIAL)", "LIVE PARTIAL"}:
            source_text = str(source_chip or "Live Data")
        elif source_detail:
            source_text = f"{source_chip} • {source_detail}"
        else:
            source_text = str(source_chip or "Data Source")
        visible_items = items[:4]
        lines = []
        for item in visible_items:
            tone = str(item.get("tone") or "info").lower()
            item_color = WARNING if tone == "warning" else (POSITIVE if tone == "positive" else ACCENT)
            title = html.escape(str(item.get("title") or "").strip())
            body = html.escape(str(item.get("body") or "").strip())
            if title and body:
                lines.append(f"<span><b style='color:{item_color};'>{title}:</b> {body}</span>")
            elif title:
                lines.append(f"<span><b style='color:{item_color};'>{title}</b></span>")
            elif body:
                lines.append(f"<span>{body}</span>")
        if len(items) > len(visible_items):
            lines.append(f"<span>+{len(items) - len(visible_items)} more health note(s).</span>")
        detail_html = (
            f"<div style='display:flex; flex-direction:column; gap:4px; margin-top:7px; line-height:1.42;'>"
            f"{''.join(lines)}"
            f"</div>"
        )
        st.markdown(
            f"<div class='market-note-box' style='border:1px solid rgba(148,163,184,0.20); border-left:4px solid {accent_color}; "
            f"background:linear-gradient(180deg, rgba(255,255,255,0.026), rgba(255,255,255,0.010)); color:{TEXT_MUTED}; "
            f"padding:7px 10px; margin:0.18rem 0 0.62rem 0;'>"
            f"<div style='display:flex; align-items:center; justify-content:space-between; gap:12px; flex-wrap:wrap;'>"
            f"<div style='display:flex; align-items:center; gap:9px; flex-wrap:wrap;'>"
            f"<span style='color:{accent_color}; font-weight:900; letter-spacing:0.08em; text-transform:uppercase;'>Data Health</span>"
            f"<span class='market-inline-chip' style='border:1px solid {accent_color}; color:{accent_color}; background:rgba(255,255,255,0.04);'>{status_text}</span>"
            f"</div>"
            f"<div style='display:flex; align-items:center; gap:8px; flex-wrap:wrap;'>"
            f"<span class='market-inline-chip' style='border:1px solid {source_color}; color:{source_color}; background:rgba(255,255,255,0.04);'>{html.escape(source_text)}</span>"
            f"<span class='market-inline-chip' style='border:1px solid {mode_color}; color:{mode_color}; background:rgba(255,255,255,0.04);'>{html.escape(mode_label)}</span>"
            f"</div>"
            f"</div>"
            f"{detail_html}"
            f"</div>",
            unsafe_allow_html=True,
        )

    archive_bundle = _market_archive_bundle(
        _fetch_signal_events_df=fetch_signal_events_df,
        _fetch_signal_forward_windows_df=fetch_signal_forward_windows_df,
        db_path=signal_tracker_db_path,
        decision_version_target=market_decision_version,
    )
    adaptive_history_raw_df = archive_bundle["raw_df"]
    adaptive_history_df = archive_bundle["df"]
    adaptive_forward_windows_df = archive_bundle["forward_windows_df"]
    breakout_archive_feedback_map = _build_breakout_archive_feedback_map(
        adaptive_history_df,
        timeframe=str(timeframe),
        direction_filter=str(direction_filter),
    )
    scanner_trace_feedback_map: dict[str, dict[str, float]] = {}
    if _normalize_scan_mode(scan_mode) in {_SCAN_MODE_EMERGING, _SCAN_MODE_TRENDING}:
        try:
            scanner_trace_rows = fetch_scanner_trace_events_df(
                scan_focus=str(_normalize_scan_mode(scan_mode)),
                timeframe=str(timeframe),
                direction_filter=str(direction_filter),
                lookback_hours=24,
                limit=2400,
                db_path=signal_tracker_db_path,
            )
            scanner_trace_feedback_map = _build_scanner_trace_feedback_map(scanner_trace_rows)
        except Exception as e:
            _debug(
                f"Scanner trace feedback unavailable ({timeframe}): "
                f"{e.__class__.__name__}: {str(e).strip()}"
            )
    adaptive_decision_mode = str(archive_bundle["decision_mode"])
    adaptive_decision_target = str(archive_bundle["decision_target"])
    adaptive_decision_rows = int(archive_bundle["decision_rows"])
    adaptive_decision_total_rows = int(archive_bundle["decision_total_rows"])
    adaptive_current_scalp_planned_rows = int(archive_bundle["current_scalp_planned_rows"])
    adaptive_current_scalp_resolved_rows = int(archive_bundle["current_scalp_resolved_rows"])
    adaptive_model = archive_bundle["adaptive_model"]
    ai_confidence_calibration_model = archive_bundle["ai_confidence_calibration_model"]
    confidence_calibration_model = archive_bundle["confidence_calibration_model"]
    setup_calibration_model = archive_bundle["setup_calibration_model"]
    actionable_ranking_model = archive_bundle["actionable_ranking_model"]
    risk_sizing_calibration_model = archive_bundle["risk_sizing_calibration_model"]
    trade_gate_calibration_model = archive_bundle["trade_gate_calibration_model"]
    scalp_calibration_model = archive_bundle["scalp_calibration_model"]
    archive_policy_map = archive_bundle["archive_policy_map"]
    archive_decision_feedback_model = archive_bundle["archive_decision_feedback_model"]
    archive_decision_feedback_map = archive_bundle["archive_decision_feedback_map"]

    # Fetch top coins
    if should_scan:
        spinner_label = (
            f"Reading custom watchlist ({len(custom_bases_applied)}) ({direction_filter}) [{timeframe}] ..."
            if custom_mode_active
            else (
                f"Reading breakout radar for {top_n} early candidates ({direction_filter}) [{timeframe}] ..."
                if _normalize_scan_mode(scan_mode) == _SCAN_MODE_EMERGING
                else (
                    f"Reading trending coins for {top_n} attention candidates ({direction_filter}) [{timeframe}] ..."
                    if _normalize_scan_mode(scan_mode) == _SCAN_MODE_TRENDING
                    else f"Reading {top_n} coins ({direction_filter}) [{timeframe}] ..."
                )
            )
        )
        with st.spinner(spinner_label):
            requested_n = len(custom_bases_applied) if custom_mode_active else int(top_n)
            scan_pool_n = _scan_candidate_pool_size(
                requested_n,
                custom_mode_active=custom_mode_active,
                scan_mode=scan_mode,
            )
            unique_market_data: list[dict] = []
            mcap_map: dict[str, int] = {}
            candidate_symbol_pool: list[str] = []
            working_symbols: list[str] = []
            usdt_symbols: list[str] = []
            provider_fetch_n: int | None = None
            breakout_freshness_cache: dict[tuple[str, str, str], float] = {}
            breakout_memory_history_df = pd.DataFrame()
            trending_volume_cache = st.session_state.get(_TRENDING_VOLUME_CACHE_KEY)
            if not isinstance(trending_volume_cache, dict):
                trending_volume_cache = {}
                st.session_state[_TRENDING_VOLUME_CACHE_KEY] = trending_volume_cache
            if _normalize_scan_mode(scan_mode) == _SCAN_MODE_EMERGING and not custom_mode_active:
                try:
                    breakout_memory_history_df = fetch_breakout_radar_snapshots_df(
                        timeframe=str(timeframe),
                        direction_filter=str(direction_filter),
                        lookback_hours=72,
                        limit=6000,
                        db_path=signal_tracker_db_path,
                    )
                except Exception as e:
                    _debug(
                        f"Breakout Radar memory unavailable ({timeframe}): "
                        f"{e.__class__.__name__}: {str(e).strip()}"
                    )

            def _load_noncustom_scan_universe(
                target_pool_n: int,
                *,
                current_fetch_n: int | None = None,
            ) -> tuple[int, list[str], list[dict], dict[str, int], list[str]]:
                normalized_scan_mode = _normalize_scan_mode(scan_mode)
                if normalized_scan_mode == _SCAN_MODE_ACTIONABLE:
                    provider_fetch_n_local = min(250, max(int(top_n) * 4, 80))
                elif normalized_scan_mode == _SCAN_MODE_EMERGING:
                    provider_fetch_n_local = min(250, max(int(top_n) * 6, 120))
                elif normalized_scan_mode == _SCAN_MODE_TRENDING:
                    provider_fetch_n_local = min(180, max(int(top_n) * 5, 80))
                else:
                    provider_fetch_n_local = min(250, max(int(top_n), 50))
                if current_fetch_n is not None:
                    provider_fetch_n_local = max(provider_fetch_n_local, int(current_fetch_n))
                while True:
                    usdt_symbols_local, market_data_local = get_top_volume_usdt_symbols(provider_fetch_n_local)
                    unique_market_data_local, mcap_map_local = _prepare_scan_market_enrichment(market_data_local)
                    if normalized_scan_mode == _SCAN_MODE_EMERGING:
                        usdt_symbols_local, unique_market_data_local, radar_mcap_map = _build_breakout_radar_universe(
                            base_pairs=usdt_symbols_local,
                            base_market_rows=unique_market_data_local,
                            breakout_memory_rows=breakout_memory_history_df,
                            fetch_top_gainers_losers=fetch_top_gainers_losers,
                            fetch_trending_coins=fetch_trending_coins,
                            fetch_exchange_tickers_snapshot=fetch_exchange_tickers_snapshot,
                            get_market_cap_rows_for_symbols=get_market_cap_rows_for_symbols,
                            direction_filter=direction_filter,
                            provider_fetch_n=provider_fetch_n_local,
                        )
                        unique_market_data_local = _enrich_breakout_radar_freshness(
                            base_pairs=usdt_symbols_local,
                            market_rows=unique_market_data_local,
                            fetch_ohlcv=fetch_ohlcv,
                            scan_timeframe=timeframe,
                            direction_filter=direction_filter,
                            max_candidates=max(14, min(int(top_n) * 3, 26)),
                            freshness_cache=breakout_freshness_cache,
                        )
                        unique_market_data_local = _apply_breakout_memory_to_market_rows(
                            unique_market_data_local,
                            breakout_memory_history_df,
                            direction_filter=direction_filter,
                        )
                        unique_market_data_local = _apply_breakout_archive_feedback_to_market_rows(
                            unique_market_data_local,
                            breakout_archive_feedback_map,
                        )
                        mcap_map_local = _merge_market_cap_maps(mcap_map_local, radar_mcap_map)
                    elif normalized_scan_mode == _SCAN_MODE_TRENDING:
                        usdt_symbols_local, unique_market_data_local, trending_mcap_map = _build_trending_scan_universe(
                            base_market_rows=unique_market_data_local,
                            fetch_trending_coins=fetch_trending_coins,
                            fetch_top_gainers_losers=fetch_top_gainers_losers,
                            get_top_volume_usdt_symbols=get_top_volume_usdt_symbols,
                            get_market_cap_rows_for_symbols=get_market_cap_rows_for_symbols,
                            fetch_ohlcv=fetch_ohlcv,
                            direction_filter=direction_filter,
                            scan_timeframe=timeframe,
                            provider_fetch_n=provider_fetch_n_local,
                            volume_anomaly_cache=trending_volume_cache,
                        )
                        unique_market_data_local = _enrich_breakout_radar_freshness(
                            base_pairs=usdt_symbols_local,
                            market_rows=unique_market_data_local,
                            fetch_ohlcv=fetch_ohlcv,
                            scan_timeframe=timeframe,
                            direction_filter=direction_filter,
                            max_candidates=max(10, min(int(top_n) * 2, 18)),
                            freshness_cache=breakout_freshness_cache,
                        )
                        mcap_map_local = _merge_market_cap_maps(mcap_map_local, trending_mcap_map)
                    if normalized_scan_mode in {_SCAN_MODE_EMERGING, _SCAN_MODE_TRENDING}:
                        unique_market_data_local = _apply_scanner_trace_feedback_to_market_rows(
                            unique_market_data_local,
                            scanner_trace_feedback_map,
                        )
                    eligible_symbols_local = _candidate_scan_symbols(
                        usdt_symbols=usdt_symbols_local,
                        market_rows=unique_market_data_local,
                        exclude_stables=bool(exclude_stables),
                        custom_bases_applied=custom_bases_applied,
                        timeframe=timeframe,
                        direction_filter=direction_filter,
                        scan_mode=scan_mode,
                        classify_symbol_sector=classify_symbol_sector,
                    )
                    next_fetch_n_local = _next_universe_fetch_n(
                        provider_fetch_n_local,
                        custom_mode_active=False,
                        eligible_count=len(eligible_symbols_local),
                        requested_n=target_pool_n,
                    )
                    candidate_symbol_pool_local = eligible_symbols_local[:target_pool_n]
                    if next_fetch_n_local == provider_fetch_n_local:
                        return (
                            provider_fetch_n_local,
                            usdt_symbols_local,
                            unique_market_data_local,
                            mcap_map_local,
                            candidate_symbol_pool_local,
                        )
                    provider_fetch_n_local = next_fetch_n_local

            if custom_mode_active:
                unique_market_data, mcap_map, usdt_symbols, candidate_symbol_pool = _build_custom_scan_universe(
                    custom_bases_applied=custom_bases_applied,
                    get_market_cap_rows_for_symbols=get_market_cap_rows_for_symbols,
                    exclude_stables=bool(exclude_stables),
                    scan_pool_n=scan_pool_n,
                )
                working_symbols = candidate_symbol_pool[:requested_n]
            else:
                provider_fetch_n, usdt_symbols, unique_market_data, mcap_map, candidate_symbol_pool = (
                    _load_noncustom_scan_universe(scan_pool_n)
                )
                working_symbols = _initial_scan_symbols(
                    candidate_pool=candidate_symbol_pool,
                    market_rows=unique_market_data,
                    requested_n=requested_n,
                    scan_pool_n=scan_pool_n,
                    custom_mode_active=False,
                    scan_mode=scan_mode,
                    timeframe=timeframe,
                )

            has_market_rows = bool(unique_market_data)
            coin_id_map = _build_market_coin_id_map(unique_market_data)
            market_row_map = _build_market_row_map(unique_market_data)
            used_major_fallback = False

            if _should_use_major_fallback(
                working_symbols=working_symbols,
                custom_mode_active=custom_mode_active,
                source_pair_count=len(usdt_symbols),
                market_row_count=len(unique_market_data),
            ):
                # Hard fallback universe for temporary upstream outages.
                used_major_fallback = True
                candidate_symbol_pool = major_fallback_symbols[: min(top_n, len(major_fallback_symbols))]
                working_symbols = list(candidate_symbol_pool)
                if working_symbols:
                    _add_data_health_item(
                        "warning",
                        "Universe Backup",
                        "Primary liquidity universe could not produce usable symbols; Market read switched to a hardcoded major backup universe.",
                    )

            # Two-phase scan:
            # 1) Fetch OHLCV with a narrow lock for shared exchange safety.
            # 2) Run analysis/model pipeline in parallel on fetched frames.
            fetch_lock = Lock()
            def _scan_one(
                sym: str,
                df_eval: pd.DataFrame,
                pair_label: str,
                actual_symbol: str,
                source_provider: str,
                chosen_anchor_plan,
                df_direction_confirm: pd.DataFrame | None,
                df_direction_lead: pd.DataFrame | None,
                frame_hunt_score: float,
            ) -> dict | None:
                """Analyse a single symbol for the scanner. Returns a row dict or None."""

                _ai_prob, ai_direction, ai_details = ml_ensemble_predict(df_eval)
                agreement = float(ai_details.get("agreement", 0.0)) if isinstance(ai_details, dict) else 0.0
                directional_agreement = float(ai_details.get("directional_agreement", agreement)) if isinstance(ai_details, dict) else agreement
                consensus_agreement = float(ai_details.get("consensus_agreement", 0.0)) if isinstance(ai_details, dict) else 0.0
                latest_closed = df_eval.iloc[-1]

                base = _canonical_pair_base(sym)
                try:
                    resolve_open_signal_events_for_frame(
                        symbol=base,
                        timeframe=timeframe,
                        df_ohlcv=df_eval,
                        source="Market",
                        db_path=signal_tracker_db_path,
                    )
                except Exception as e:
                    _debug(
                        f"Signal tracker resolve failed for {base} ({timeframe}): "
                        f"{e.__class__.__name__}: {str(e).strip()}"
                    )
                try:
                    resolve_open_signal_events_for_frame(
                        symbol=base,
                        timeframe=timeframe,
                        df_ohlcv=df_eval,
                        source="Scalp",
                        db_path=signal_tracker_db_path,
                    )
                except Exception as e:
                    _debug(
                        f"Scalp tracker resolve failed for {base} ({timeframe}): "
                        f"{e.__class__.__name__}: {str(e).strip()}"
                    )
                mcap_val = mcap_map.get(base)
                market_row = market_row_map.get(base) or {}
                radar_source_score = _sortable_float((market_row or {}).get("_radar_source_score", 0.0))
                radar_freshness_score = _sortable_float((market_row or {}).get("_radar_freshness_score", 0.0))
                # Keep price semantics aligned with all decision metrics (closed-candle context).
                price = float(latest_closed["close"])
                # Delta source of truth: selected-timeframe closed candles.
                # This keeps table delta aligned with tactical setup and confidence calculations.
                price_change = None
                delta_note = "Move between the last two closed candles on your selected timeframe."
                try:
                    prev_close = float(df_eval["close"].iloc[-2])
                    last_closed = float(df_eval["close"].iloc[-1])
                    if pd.notna(prev_close) and prev_close > 0 and pd.notna(last_closed):
                        price_change = ((last_closed / prev_close) - 1.0) * 100.0
                except Exception as e:
                    _debug(f"Delta candle fallback for {sym} ({timeframe}): {e.__class__.__name__}: {str(e).strip()}")
                    price_change = None
                # Safety fallback (rare): if candle delta is unavailable, use ticker percentage.
                fallback_delta_symbol = _delta_fallback_symbol(sym, actual_symbol, source_provider)
                if price_change is None:
                    if fallback_delta_symbol:
                        try:
                            # Protect shared exchange ticker fallback under the same lock.
                            price_change = _fetch_ticker_delta_once(
                                get_price_change,
                                fallback_delta_symbol,
                                fetch_lock,
                            )
                            if price_change is not None:
                                delta_note = (
                                    f"Using exchange ticker change for {fallback_delta_symbol} "
                                    "because the candle-based move was unavailable."
                                )
                        except Exception as e:
                            _debug(
                                f"Ticker delta fallback failed for {fallback_delta_symbol} ({timeframe}): "
                                f"{e.__class__.__name__}: {str(e).strip()}"
                            )
                            price_change = None

                a = analyse(df_eval)
                signal, volume_spike = a.signal, a.volume_spike
                atr_comment_v, candle_pattern_v, bias_score_v = a.atr_comment, a.candle_pattern, a.bias
                adx_val_v, supertrend_trend_v, ichimoku_trend_v = a.adx, a.supertrend, a.ichimoku
                stochrsi_k_val_v, bollinger_bias_v, vwap_label_v = a.stochrsi_k, a.bollinger, a.vwap
                psar_trend_v = a.psar

                spike_dir = ""
                spike_vol_ratio = float("nan")
                spike_candle_pct = float("nan")
                spike_vwap_ctx = str(vwap_label_v or "").replace("🟢 ", "").replace("🔴 ", "").replace("→ ", "").strip()
                if volume_spike:
                    try:
                        prev_vol_avg = float(df_eval["volume"].iloc[-21:-1].mean()) if len(df_eval) >= 21 else float("nan")
                        last_vol = float(df_eval["volume"].iloc[-1]) if len(df_eval) >= 1 else float("nan")
                        if pd.notna(prev_vol_avg) and prev_vol_avg > 0 and pd.notna(last_vol):
                            spike_vol_ratio = last_vol / prev_vol_avg
                    except Exception:
                        spike_vol_ratio = float("nan")
                    try:
                        o = float(latest_closed["open"])
                        c = float(latest_closed["close"])
                        if pd.notna(o) and pd.notna(c) and o > 0:
                            spike_candle_pct = ((c / o) - 1.0) * 100.0
                        if pd.notna(o) and pd.notna(c):
                            if c > o:
                                spike_dir = "UP"
                            elif c < o:
                                spike_dir = "DOWN"
                            else:
                                spike_dir = "NEUTRAL"
                    except Exception:
                        spike_dir = "NEUTRAL"

                scalp_direction = None
                entry_s = target_s = stop_s = rr_ratio = 0.0
                breakout_note = ""

                signal_direction = direction_key(signal_plain(signal))
                signal_text = sanitize_trading_terms(signal)
                comment_text = sanitize_trading_terms(str(getattr(a, 'comment', '') or '').strip())
                spot_snapshot = build_spot_direction_snapshot(
                    df_4h=None,
                    df_1d=None,
                    lead_df=df_direction_lead,
                    confirm_df=df_direction_confirm,
                    lead_timeframe=chosen_anchor_plan.lead_timeframe,
                    confirm_timeframe=chosen_anchor_plan.confirm_timeframe,
                )
                ai_spot_snapshot = build_ai_spot_bias_snapshot(
                    df_4h=None,
                    df_1d=None,
                    lead_df=df_direction_lead,
                    confirm_df=df_direction_confirm,
                    lead_timeframe=chosen_anchor_plan.lead_timeframe,
                    confirm_timeframe=chosen_anchor_plan.confirm_timeframe,
                    predictor=ml_ensemble_predict,
                )
                confidence_calibration_snapshot = build_confidence_calibration_snapshot(
                    confidence_calibration_model,
                    signal={
                        "Direction": str(spot_snapshot.direction or ""),
                        "AI Alignment": (
                            "Aligned"
                            if _signal_tracker_direction_key(spot_snapshot.direction) in {"UPSIDE", "DOWNSIDE"}
                            and _signal_tracker_direction_key(spot_snapshot.direction)
                            == _signal_tracker_direction_key(ai_spot_snapshot.direction)
                            else "Not aligned"
                        ),
                        "Timeframe": str(timeframe or "Unknown"),
                        "Scan Focus": str(_normalize_scan_mode(scan_mode) or "Unknown"),
                    },
                )
                confidence_snapshot = build_confidence_snapshot(
                    direction=spot_snapshot.direction,
                    timeframe_alignment=spot_snapshot.timeframe_alignment,
                    structure_quality=spot_snapshot.structure_quality,
                    trend_quality=spot_snapshot.trend_quality,
                    regime_quality=spot_snapshot.regime_quality,
                    location_quality=spot_snapshot.location_quality,
                    timeframe_conflict=spot_snapshot.timeframe_conflict,
                    degraded_data=spot_snapshot.degraded_data,
                    range_regime=spot_snapshot.range_regime,
                    archive_calibration_delta=float(getattr(confidence_calibration_snapshot, "delta", 0.0) or 0.0),
                    archive_calibration_note=str(getattr(confidence_calibration_snapshot, "note", "") or ""),
                )

                direction_note = _spot_direction_note(
                    spot_snapshot,
                    selected_timeframe=timeframe,
                    tactical_direction=signal_direction,
                    tactical_signal=signal_text,
                    tactical_bias=float(bias_score_v),
                    tactical_comment=comment_text,
                )
                confidence_note = _confidence_note(
                    spot_snapshot,
                    float(confidence_snapshot.score),
                    confidence_snapshot,
                )

                ai_direction_key = direction_key(ai_direction)
                _, _, decision_agreement = ai_vote_metrics(
                    ai_direction_key,
                    float(directional_agreement),
                    float(consensus_agreement),
                )
                ai_display_votes = ai_spot_bias_display_votes(ai_spot_snapshot)
                ai_display = f"{direction_label(ai_spot_snapshot.direction)} ({ai_display_votes}/3)"
                if bool(ai_spot_snapshot.degraded_data):
                    ai_display += " *"
                ai_note = _ai_spot_bias_note(ai_spot_snapshot)

                ai_spot_direction_key = direction_key(ai_spot_snapshot.direction)
                ai_spot_agreement = float(ai_spot_bias_directional_agreement(ai_spot_snapshot))
                ai_spot_consensus = float(ai_spot_bias_consensus_agreement(ai_spot_snapshot))
                ai_spot_probability_up = float(ai_spot_bias_probability_up(ai_spot_snapshot))
                ai_spot_status = str(ai_spot_bias_status(ai_spot_snapshot) or "")
                directional_confidence = float(bias_confidence_from_bias(float(bias_score_v)))
                structure_val = spot_structure_state(
                    spot_snapshot.direction,
                    signal_direction,
                    ai_direction,
                    float(confidence_snapshot.score),
                    float(decision_agreement),
                )
                _conv_lbl, _ = _calc_conviction(
                    signal_direction,
                    ai_direction,
                    directional_confidence,
                    float(decision_agreement),
                )
                execution_confidence = build_execution_confidence_snapshot(
                    direction=signal_direction,
                    bias_score=float(bias_score_v),
                    adx_val=float(adx_val_v) if pd.notna(adx_val_v) else float("nan"),
                    structure_state=str(structure_val),
                    conviction_label=str(_conv_lbl),
                    ai_agreement=float(decision_agreement),
                )
                execution_snapshot = selected_timeframe_execution_snapshot(
                    df=df_eval,
                    direction=spot_snapshot.direction,
                    bias_score=float(bias_score_v),
                    adx_val=float(adx_val_v) if pd.notna(adx_val_v) else float("nan"),
                    supertrend_trend=str(supertrend_trend_v),
                    ichimoku_trend=str(ichimoku_trend_v),
                    vwap_label=str(vwap_label_v),
                    psar_trend=str(psar_trend_v),
                    bollinger_bias=str(bollinger_bias_v),
                    williams_label=str(a.williams),
                    cci_label=str(a.cci),
                )
                setup_rr_val = float(selected_timeframe_rr_ratio(execution_snapshot, direction=spot_snapshot.direction))
                trend_led_snapshot = trend_led_confirmation_snapshot(
                    spot_dir=spot_snapshot.direction,
                    spot_confidence=float(confidence_snapshot.score),
                    tactical_dir=signal_direction,
                    adx_val=float(adx_val_v) if pd.notna(adx_val_v) else float("nan"),
                    structure_quality=float(execution_snapshot.structure_quality),
                    trend_quality=float(execution_snapshot.trend_quality),
                    regime_quality=float(execution_snapshot.regime_quality),
                    location_quality=float(execution_snapshot.location_quality),
                    rr_ratio=setup_rr_val if math.isfinite(setup_rr_val) and setup_rr_val > 0.0 else None,
                )
                ai_led_snapshot = ai_led_confirmation_snapshot(
                    spot_dir=spot_snapshot.direction,
                    spot_confidence=float(confidence_snapshot.score),
                    ai_dir=ai_spot_direction_key,
                    ai_probability=float(ai_spot_probability_up),
                    directional_agreement=float(ai_spot_agreement),
                    consensus_agreement=float(ai_spot_consensus),
                    adx_val=float(adx_val_v) if pd.notna(adx_val_v) else float("nan"),
                    location_quality=float(execution_snapshot.location_quality),
                    rr_ratio=setup_rr_val if math.isfinite(setup_rr_val) and setup_rr_val > 0.0 else None,
                    ai_status=ai_spot_status,
                )
                actionable_tactical_score = _actionable_tactical_candidate_score(
                    spot_direction=spot_snapshot.direction,
                    signal_direction=signal_direction,
                    ai_direction=ai_spot_snapshot.direction,
                    ai_agreement=float(ai_spot_agreement),
                    frame_hunt_score=float(frame_hunt_score),
                    execution_structure_quality=float(execution_snapshot.structure_quality),
                    execution_trend_quality=float(execution_snapshot.trend_quality),
                    execution_location_quality=float(execution_snapshot.location_quality),
                    rr_ratio=setup_rr_val if math.isfinite(setup_rr_val) and setup_rr_val > 0.0 else None,
                    adx_val=float(adx_val_v) if pd.notna(adx_val_v) else float("nan"),
                )
                emerging_ai_confidence_snapshot = build_ai_confidence_snapshot(
                    direction=ai_spot_snapshot.direction,
                    combined_score=float(ai_spot_snapshot.score),
                    conviction_quality=float(ai_spot_snapshot.conviction_quality),
                    timeframe_alignment=float(ai_spot_snapshot.timeframe_alignment),
                    consensus_quality=float(ai_spot_snapshot.consensus_quality),
                    support_votes=int(ai_display_votes),
                    timeframe_conflict=bool(ai_spot_snapshot.timeframe_conflict),
                    degraded_data=bool(ai_spot_snapshot.degraded_data),
                    archive_calibration_delta=0.0,
                    archive_calibration_note="",
                )
                emerging_precheck_snapshot = emerging_bias_snapshot(
                    spot_snapshot=spot_snapshot,
                    ai_spot_snapshot=ai_spot_snapshot,
                    ai_confidence_score=float(emerging_ai_confidence_snapshot.score),
                    tech_confidence_score=float(confidence_snapshot.score),
                )
                include = _actionable_direction_include(
                    direction_filter=direction_filter,
                    scan_mode=scan_mode,
                    spot_direction=spot_snapshot.direction,
                    signal_direction=signal_direction,
                    tactical_candidate_score=actionable_tactical_score,
                    emerging_direction=getattr(emerging_precheck_snapshot, "direction", ""),
                    frame_hunt_score=frame_hunt_score,
                    radar_source_score=max(radar_source_score, radar_freshness_score),
                )
                if not include:
                    return None
                actionable_tactical_candidate = (
                    _normalize_scan_mode(scan_mode) == _SCAN_MODE_ACTIONABLE
                    and _signal_tracker_direction_key(spot_snapshot.direction) == "NEUTRAL"
                    and _signal_tracker_direction_key(signal_direction) in {"UPSIDE", "DOWNSIDE"}
                    and float(actionable_tactical_score) >= 72.0
                )
                action, action_reason_code = spot_action_decision_with_reason(
                    spot_snapshot.direction,
                    float(confidence_snapshot.score),
                    signal_direction,
                    ai_spot_snapshot.direction,
                    float(ai_spot_agreement),
                    float(adx_val_v) if pd.notna(adx_val_v) else float("nan"),
                    trend_led_snapshot=trend_led_snapshot,
                    ai_led_snapshot=ai_led_snapshot,
                )
                setup_calibration_snapshot = build_setup_calibration_snapshot(
                    setup_calibration_model,
                    signal={
                        "Setup Confirm": str(action or ""),
                        "AI Alignment": (
                            "Aligned"
                            if _signal_tracker_direction_key(spot_snapshot.direction) in {"UPSIDE", "DOWNSIDE"}
                            and _signal_tracker_direction_key(spot_snapshot.direction) == _signal_tracker_direction_key(ai_spot_snapshot.direction)
                            else "Not aligned"
                        ),
                        "Timeframe": str(timeframe or "Unknown"),
                        "Scan Focus": str(_normalize_scan_mode(scan_mode) or "Unknown"),
                        "Direction": str(spot_snapshot.direction or ""),
                    },
                )
                action, action_reason_code = apply_setup_archive_calibration(
                    action,
                    action_reason_code,
                    calibration_delta=float(getattr(setup_calibration_snapshot, "delta", 0.0) or 0.0),
                )
                scalp_direction, entry_s, target_s, stop_s, rr_ratio, breakout_note = get_scalping_entry_target(
                    df_eval,
                    bias_score_v,
                    supertrend_trend_v,
                    ichimoku_trend_v,
                    vwap_label_v,
                    timeframe=timeframe,
                    execution_snapshot=execution_snapshot,
                    trend_led_snapshot=trend_led_snapshot,
                    ai_led_snapshot=ai_led_snapshot,
                    spot_direction=spot_snapshot.direction,
                    ai_direction=ai_spot_snapshot.direction,
                )
                scalp_display = _build_scalp_display_payload(
                    timeframe_value=timeframe,
                    scalp_direction=scalp_direction,
                    signal_direction=signal_direction,
                    rr_ratio=rr_ratio,
                    adx_val=float(adx_val_v) if pd.notna(adx_val_v) else float("nan"),
                    confidence=float(execution_confidence.score),
                    conviction_label=str(_conv_lbl),
                    entry=entry_s,
                    stop=stop_s,
                    target=target_s,
                    setup_confirm=action,
                    direction_value=spot_snapshot.direction,
                    ai_aligned=(
                        _signal_tracker_direction_key(spot_snapshot.direction) in {"UPSIDE", "DOWNSIDE"}
                        and _signal_tracker_direction_key(spot_snapshot.direction) == _signal_tracker_direction_key(ai_spot_snapshot.direction)
                    ),
                    scan_focus_value=str(_normalize_scan_mode(scan_mode) or "Unknown"),
                    breakout_note=str(breakout_note or ""),
                    close_ref=float(latest_closed["close"]) if pd.notna(latest_closed["close"]) else None,
                    ema_ref=float(df_eval["close"].ewm(span=5, adjust=False).mean().iloc[-1]),
                )
                rr_val = float(scalp_display["rr_val"] or 0.0)
                ai_confidence_calibration_snapshot = build_ai_confidence_calibration_snapshot(
                    ai_confidence_calibration_model,
                    signal={
                        "Setup Confirm": str(action or ""),
                        "AI Alignment": (
                            "Aligned"
                            if _signal_tracker_direction_key(spot_snapshot.direction) in {"UPSIDE", "DOWNSIDE"}
                            and _signal_tracker_direction_key(spot_snapshot.direction) == _signal_tracker_direction_key(ai_spot_snapshot.direction)
                            else "Not aligned"
                        ),
                        "Timeframe": str(timeframe or "Unknown"),
                        "Scan Focus": str(_normalize_scan_mode(scan_mode) or "Unknown"),
                        "Direction": str(spot_snapshot.direction or ""),
                    },
                )
                ai_confidence_snapshot = build_ai_confidence_snapshot(
                    direction=ai_spot_snapshot.direction,
                    combined_score=float(ai_spot_snapshot.score),
                    conviction_quality=float(ai_spot_snapshot.conviction_quality),
                    timeframe_alignment=float(ai_spot_snapshot.timeframe_alignment),
                    consensus_quality=float(ai_spot_snapshot.consensus_quality),
                    support_votes=int(ai_display_votes),
                    timeframe_conflict=bool(ai_spot_snapshot.timeframe_conflict),
                    degraded_data=bool(ai_spot_snapshot.degraded_data),
                    archive_calibration_delta=float(getattr(ai_confidence_calibration_snapshot, "delta", 0.0) or 0.0),
                    archive_calibration_note=str(getattr(ai_confidence_calibration_snapshot, "note", "") or ""),
                )
                ai_confidence_note = _ai_confidence_note(
                    ai_spot_snapshot,
                    float(ai_confidence_snapshot.score),
                    ai_confidence_snapshot,
                )
                emerging_snapshot = emerging_bias_snapshot(
                    spot_snapshot=spot_snapshot,
                    ai_spot_snapshot=ai_spot_snapshot,
                    ai_confidence_score=float(emerging_ai_confidence_snapshot.score),
                    tech_confidence_score=float(confidence_snapshot.score),
                )
                emerging_rank_score = _emerging_candidate_score(
                    timeframe=timeframe,
                    direction_filter=direction_filter,
                    spot_direction=spot_snapshot.direction,
                    signal_direction=signal_direction,
                    emerging_direction=str(getattr(emerging_snapshot, "direction", "") or ""),
                    emerging_active=bool(getattr(emerging_snapshot, "active", False)),
                    frame_hunt_score=float(frame_hunt_score),
                    tactical_candidate_score=float(actionable_tactical_score),
                    execution_structure_quality=float(execution_snapshot.structure_quality),
                    execution_trend_quality=float(execution_snapshot.trend_quality),
                    execution_location_quality=float(execution_snapshot.location_quality),
                    tech_confidence_score=float(confidence_snapshot.score),
                    ai_confidence_score=float(ai_confidence_snapshot.score),
                    market_cap=mcap_val,
                    market_pct_change_24h=_sortable_float((market_row or {}).get("price_change_percentage_24h")),
                    volume_spike=bool(volume_spike),
                    spike_dir=spike_dir,
                    radar_source_score=radar_source_score,
                    radar_freshness_score=radar_freshness_score,
                    radar_memory_score=_sortable_float((market_row or {}).get("_radar_memory_score")),
                    radar_archive_edge_score=_sortable_float((market_row or {}).get("_radar_archive_edge_score")),
                    radar_trace_boost_score=_sortable_float((market_row or {}).get("_radar_trace_boost_score")),
                )
                if actionable_tactical_candidate:
                    tactical_note = (
                        "Actionable tactical candidate: selected timeframe is aligned early while higher-timeframe direction is still neutral."
                    )
                    confidence_note = (
                        f"{confidence_note} {tactical_note}".strip()
                        if confidence_note
                        else tactical_note
                    )
                ichimoku_cell = format_trend(ichimoku_trend_v)
                ichi_detail_parts: list[str] = []
                if a.ichimoku_tk_cross:
                    ichi_detail_parts.append(f"TK Cross: {a.ichimoku_tk_cross.replace('▲ ', '').replace('▼ ', '').replace('→ ', '')}")
                if a.ichimoku_future_bias:
                    ichi_detail_parts.append(
                        f"Future Cloud: {a.ichimoku_future_bias.replace('▲ ', '').replace('▼ ', '').replace('→ ', '')}"
                    )
                if a.ichimoku_cloud_strength:
                    ichi_detail_parts.append(
                        f"Cloud Strength: {a.ichimoku_cloud_strength.replace('▲ ', '').replace('▼ ', '').replace('→ ', '')}"
                    )
                ichimoku_detail = " | ".join(ichi_detail_parts)

                return {
                    'Coin': base,
                    '__pair': pair_label,
                    '__event_time': latest_closed.get("timestamp"),
                    '__timeframe': timeframe,
                    '__emerging_label': emerging_snapshot.label,
                    '__emerging_direction': emerging_snapshot.direction,
                    '__emerging_note': emerging_snapshot.note,
                    'Price ($)': _fmt_price(price),
                    '__price_val': float(price),
                    'Δ (%)': format_delta(price_change) if price_change is not None else '',
                    '__delta_pct': float(price_change) if price_change is not None else None,
                    '__delta_note': delta_note if price_change is not None else "",
                    'Setup Confirm': _setup_confirm_display(action, action_reason_code, direction=str(spot_snapshot.direction or "")),
                    '__action_raw': action,
                    '__action_reason': action_reason_code,
                    '__setup_calibrated': True,
                    '__setup_calibration_delta': float(getattr(setup_calibration_snapshot, "delta", 0.0) or 0.0),
                    '__setup_calibration_note': str(getattr(setup_calibration_snapshot, "note", "") or ""),
                    '__signal_direction_raw': signal_direction,
                    'Direction': direction_label(spot_snapshot.direction),
                    '__direction_note': direction_note,
                    'Confidence': _confidence_badge(float(confidence_snapshot.score)),
                    '__confidence_note': confidence_note,
                    'AI Ensemble': ai_display,
                    '__ai_votes': ai_display_votes,
                    '__ai_note': ai_note,
                    '__ai_direction_raw': str(ai_spot_snapshot.direction or ""),
                    '__ai_score_raw': float(ai_spot_snapshot.score),
                    '__ai_conviction_quality_raw': float(ai_spot_snapshot.conviction_quality),
                    '__ai_timeframe_alignment_raw': float(ai_spot_snapshot.timeframe_alignment),
                    '__ai_consensus_quality_raw': float(ai_spot_snapshot.consensus_quality),
                    '__ai_timeframe_conflict_raw': bool(ai_spot_snapshot.timeframe_conflict),
                    '__ai_degraded_data_raw': bool(ai_spot_snapshot.degraded_data),
                    'AI Confidence': _ai_confidence_badge(ai_spot_snapshot, float(ai_confidence_snapshot.score)),
                    '__ai_confidence_note': ai_confidence_note,
                    '__ai_confidence_val': float(ai_confidence_snapshot.score),
                    'Scalp Opportunity': str(scalp_display["label"] or ""),
                    '__scalp_direction_raw': scalp_direction,
                    '__scalp_entry_val_raw': float(entry_s) if entry_s else None,
                    '__scalp_stop_val_raw': float(stop_s) if stop_s else None,
                    '__scalp_target_val_raw': float(target_s) if target_s else None,
                    '__scalp_rr_val_raw': float(rr_ratio) if rr_ratio else None,
                    '__scalp_breakout_note_raw': str(breakout_note or ""),
                    '__scalp_reason_text': str(scalp_display["reason_text"] or ""),
                    '__scalp_reason_short': str(scalp_display["reason_short"] or ""),
                    '__scalp_display_state': str(scalp_display["display_state"] or ""),
                    'Entry Price': str(scalp_display["entry_display"] or ""),
                    '__entry_val': scalp_display["entry_val"],
                    '__entry_note': str(scalp_display["entry_note"] or ""),
                    'Stop Loss': str(scalp_display["stop_display"] or ""),
                    '__stop_val': scalp_display["stop_val"],
                    'Target Price': str(scalp_display["target_display"] or ""),
                    '__target_val': scalp_display["target_val"],
                    '__target_note': str(scalp_display["target_note"] or ""),
                    '__rr_note': str(scalp_display["rr_note"] or ""),
                    'R:R': str(scalp_display["rr_badge"] or ""),
                    '__rr_val': scalp_display["rr_val"],
                    'Market Cap ($)': readable_market_cap(mcap_val) if mcap_val else "—",
                    '__mcap_val': int(mcap_val) if mcap_val else 0,
                    'Spike Alert': '→ Spike' if volume_spike else '',
                    '__spike_dir': spike_dir,
                    '__spike_vol_ratio': spike_vol_ratio,
                    '__spike_candle_pct': spike_candle_pct,
                    '__spike_vwap_ctx': spike_vwap_ctx,
                    'ADX': round(adx_val_v, 1) if pd.notna(adx_val_v) else float("nan"),
                    '__adx_raw': round(adx_val_v, 2) if pd.notna(adx_val_v) else float("nan"),
                    'SuperTrend': supertrend_trend_v,
                    'Volatility': atr_comment_v,
                    'Stochastic RSI': round(stochrsi_k_val_v, 2) if pd.notna(stochrsi_k_val_v) else float("nan"),
                    'Candle Pattern': candle_pattern_v,
                    'Ichimoku': ichimoku_cell,
                    '__ichimoku_detail': ichimoku_detail,
                    'Bollinger': bollinger_bias_v,
                    'VWAP': vwap_label_v,
                    'PSAR': psar_trend_v if psar_trend_v != "Unavailable" else '',
                    'Williams %R': a.williams,
                    'CCI': a.cci,
                    '__confidence_val': float(confidence_snapshot.score),
                    '__execution_confidence_val': float(execution_confidence.score),
                    '__execution_conviction_label': str(_conv_lbl),
                    '__emerging_rank_score': float(emerging_rank_score),
                    '__radar_trace_boost_score': _sortable_float((market_row or {}).get("_radar_trace_boost_score")),
                    '__actionable_frame_score': float(frame_hunt_score),
                    '__actionable_tactical_score': float(actionable_tactical_score),
                    '__actionable_setup_score': _actionable_setup_score(
                        timeframe=timeframe,
                        execution_structure_quality=float(execution_snapshot.structure_quality),
                        execution_trend_quality=float(execution_snapshot.trend_quality),
                        execution_regime_quality=float(execution_snapshot.regime_quality),
                        execution_location_quality=float(execution_snapshot.location_quality),
                        trend_led_score=float(trend_led_snapshot.score),
                        ai_led_score=float(ai_led_snapshot.score),
                        rr_ratio=rr_val,
                        adx_val=float(adx_val_v) if pd.notna(adx_val_v) else float("nan"),
                        delta_pct=price_change,
                        volatility_label=str(atr_comment_v),
                        vwap_label=str(vwap_label_v),
                        bollinger_bias=str(bollinger_bias_v),
                        signal_direction=signal_direction,
                        volume_spike=bool(volume_spike),
                        spike_dir=spike_dir,
                        frame_hunt_score=frame_hunt_score,
                    ),
                }

            def _scan_candidate_batch(symbol_batch: list[str]) -> tuple[list[dict], list[tuple[str, str]], int]:
                priority_hunt_active = (
                    _normalize_scan_mode(scan_mode) in {_SCAN_MODE_ACTIONABLE, _SCAN_MODE_EMERGING, _SCAN_MODE_TRENDING}
                    and not custom_mode_active
                )
                selected_frames: list[
                    tuple[str, pd.DataFrame, str, str, str, object | None, float]
                ] = []
                fetch_failures: list[tuple[str, str]] = []
                for sym in symbol_batch:
                    fallback_coin_id = _custom_watchlist_fallback_coin_id(
                        sym,
                        custom_mode_active=custom_mode_active,
                        coin_id_map=coin_id_map,
                    )
                    df = _fetch_market_scan_ohlcv(
                        fetch_ohlcv=fetch_ohlcv,
                        fetch_coingecko_ohlcv_by_coin_id=fetch_coingecko_ohlcv_by_coin_id,
                        fetch_lock=fetch_lock,
                        symbol=sym,
                        timeframe=timeframe,
                        limit=500,
                        fallback_coin_id=fallback_coin_id,
                    )
                    if df is None:
                        reason = "no OHLCV data"
                        if custom_mode_active:
                            if fallback_coin_id and not coingecko_coin_id_fallback_available:
                                reason = _coingecko_coin_id_unavailable_message(
                                    coingecko_coin_id_fallback_reason
                                )
                            elif fallback_coin_id:
                                reason = "no exchange OHLCV data; CoinGecko backup returned empty"
                            else:
                                reason = "no exchange pair; coin-id unresolved for backup"
                        fetch_failures.append((sym, reason))
                        continue
                    actual_symbol = str(df.attrs.get("source_symbol") or "").strip() or sym
                    source_provider = str(df.attrs.get("source_provider") or "").strip() or "exchange"
                    pair_label = _pair_provenance_label(
                        sym,
                        actual_symbol,
                        source_provider,
                    )
                    if len(df) <= 60:
                        fetch_failures.append((pair_label, f"insufficient candles ({len(df)})"))
                        continue
                    # Align analysis and scalp planning on same closed-candle context.
                    df_eval = _prepare_closed_frame(df, min_rows=56)
                    if df_eval is None:
                        fetch_failures.append((pair_label, "insufficient closed-candle history"))
                        continue
                    frame_hunt_score = _actionable_frame_hunt_score(
                        df_eval=df_eval,
                        timeframe=timeframe,
                        direction_filter=direction_filter,
                    )
                    selected_frames.append(
                        (
                            sym,
                            df_eval,
                            pair_label,
                            actual_symbol,
                            source_provider,
                            fallback_coin_id,
                            frame_hunt_score,
                        )
                    )

                analysis_frames = list(selected_frames)
                if priority_hunt_active and analysis_frames:
                    analysis_batch_n = _actionable_analysis_batch_size(
                        requested_n=requested_n,
                        fetched_n=len(analysis_frames),
                        scan_mode=scan_mode,
                    )
                    analysis_frames = sorted(
                        analysis_frames,
                        key=lambda item: (-_sortable_float(item[6]), str(item[2])),
                    )[:analysis_batch_n]

                fetched_frames: list[
                    tuple[str, pd.DataFrame, str, str, str, object, pd.DataFrame | None, pd.DataFrame | None, float]
                ] = []
                for (
                    sym,
                    df_eval,
                    pair_label,
                    actual_symbol,
                    source_provider,
                    fallback_coin_id,
                    frame_hunt_score,
                ) in analysis_frames:
                    direction_fetch_symbol = _direction_fetch_symbol(sym, actual_symbol, source_provider)
                    frame_cache: dict[str, pd.DataFrame | None] = {}

                    def _fetch_anchor_frame(anchor_timeframe: str) -> pd.DataFrame | None:
                        if anchor_timeframe in frame_cache:
                            return frame_cache[anchor_timeframe]
                        raw = _fetch_market_scan_ohlcv(
                            fetch_ohlcv=fetch_ohlcv,
                            fetch_coingecko_ohlcv_by_coin_id=fetch_coingecko_ohlcv_by_coin_id,
                            fetch_lock=fetch_lock,
                            symbol=direction_fetch_symbol,
                            timeframe=anchor_timeframe,
                            limit=260,
                            fallback_coin_id=fallback_coin_id,
                        )
                        prepared = _prepare_closed_frame(raw, min_rows=81)
                        frame_cache[anchor_timeframe] = prepared
                        return prepared

                    chosen_anchor_plan, df_direction_lead, df_direction_confirm = choose_anchor_context(
                        timeframe,
                        _fetch_anchor_frame,
                    )
                    fetched_frames.append(
                        (
                            sym,
                            df_eval,
                            pair_label,
                            actual_symbol,
                            source_provider,
                            chosen_anchor_plan,
                            df_direction_confirm,
                            df_direction_lead,
                            frame_hunt_score,
                        )
                    )

                batch_results: list[dict] = []
                scan_errors: list[tuple[str, str]] = []
                with ThreadPoolExecutor(max_workers=4) as executor:
                    futures = {
                        executor.submit(
                            _scan_one,
                            sym,
                            df_eval,
                            pair_label,
                            actual_symbol,
                            source_provider,
                            chosen_anchor_plan,
                            df_direction_confirm,
                            df_direction_lead,
                            frame_hunt_score,
                        ): pair_label
                        for (
                            sym,
                            df_eval,
                            pair_label,
                            actual_symbol,
                            source_provider,
                            chosen_anchor_plan,
                            df_direction_confirm,
                            df_direction_lead,
                            frame_hunt_score,
                        ) in fetched_frames
                    }
                    for future in as_completed(futures):
                        try:
                            row = future.result()
                            if row is not None:
                                batch_results.append(row)
                        except Exception as e:
                            pair_label = futures[future]
                            err = f"{e.__class__.__name__}: {str(e).strip()}".strip(": ")
                            scan_errors.append((pair_label, err))
                            _debug(f"Scanner error for {pair_label}: {err}")
                return batch_results, [*fetch_failures, *scan_errors], len(selected_frames)

            fresh_results: list[dict] = []
            live_produced_rows: list[dict] = []
            skipped_symbols: list[tuple[str, str]] = []
            total_fetched_frame_count = 0
            attempted_symbols: set[str] = set()
            live_result_count_before_limit = 0
            live_ranked_out_count = 0
            display_scan_state: dict[str, object] | None = None
            scan_pool_target_n = int(scan_pool_n)
            pending_batch = list(working_symbols)
            while True:
                if pending_batch:
                    batch_results, batch_skipped, batch_fetched = _scan_candidate_batch(pending_batch)
                    fresh_results.extend(batch_results)
                    skipped_symbols.extend(batch_skipped)
                    total_fetched_frame_count += batch_fetched
                    attempted_symbols.update(pending_batch)
                    display_scan_state = _remember_display_scan_state(
                        display_scan_state,
                        batch_results=batch_results,
                        candidate_count=len(candidate_symbol_pool),
                        mcap_map=mcap_map,
                        has_market_rows=bool(unique_market_data),
                        source_pair_count=len(usdt_symbols),
                        market_row_count=len(unique_market_data),
                    )

                if len(fresh_results) >= requested_n:
                    break

                pending_batch = _next_refill_candidate_batch(
                    candidate_pool=candidate_symbol_pool,
                    attempted_symbols=attempted_symbols,
                    requested_n=requested_n,
                    produced_n=len(fresh_results),
                    custom_mode_active=custom_mode_active,
                    used_major_fallback=used_major_fallback,
                    scan_mode=scan_mode,
                )
                if pending_batch:
                    continue

                next_pool_target_n = _next_scan_pool_target(
                    scan_pool_target_n,
                    requested_n=requested_n,
                    produced_n=len(fresh_results),
                    custom_mode_active=custom_mode_active,
                    used_major_fallback=used_major_fallback,
                    scan_mode=scan_mode,
                )
                if next_pool_target_n <= scan_pool_target_n:
                    break
                scan_pool_target_n = next_pool_target_n

                provider_fetch_n, usdt_symbols, unique_market_data, mcap_map, candidate_symbol_pool = (
                    _load_noncustom_scan_universe(
                        scan_pool_target_n,
                        current_fetch_n=provider_fetch_n,
                    )
                )
                pending_batch = _next_refill_candidate_batch(
                    candidate_pool=candidate_symbol_pool,
                    attempted_symbols=attempted_symbols,
                    requested_n=requested_n,
                    produced_n=len(fresh_results),
                    custom_mode_active=custom_mode_active,
                    used_major_fallback=used_major_fallback,
                    scan_mode=scan_mode,
                )
                if not pending_batch and scan_pool_target_n >= 250:
                    break

            if _normalize_scan_mode(scan_mode) == _SCAN_MODE_EMERGING and not custom_mode_active:
                try:
                    log_breakout_radar_snapshots(
                        unique_market_data,
                        timeframe=str(timeframe),
                        direction_filter=str(direction_filter),
                        db_path=signal_tracker_db_path,
                    )
                except Exception as e:
                    _debug(
                        f"Breakout Radar memory log failed ({timeframe}): "
                        f"{e.__class__.__name__}: {str(e).strip()}"
                    )

            display_state = _resolve_display_scan_state(
                fresh_results=fresh_results,
                current_candidate_count=len(candidate_symbol_pool),
                current_mcap_map=mcap_map,
                current_has_market_rows=bool(unique_market_data),
                current_source_pair_count=len(usdt_symbols),
                current_market_row_count=len(unique_market_data),
                display_state=display_scan_state,
            )
            notice_state = _resolve_notice_scan_state(
                current_candidate_count=len(candidate_symbol_pool),
                current_has_market_rows=bool(unique_market_data),
                current_source_pair_count=len(usdt_symbols),
                current_market_row_count=len(unique_market_data),
                display_state=display_scan_state,
            )
            display_mcap_map = dict(display_state.get("mcap_map") or {})
            has_market_rows = bool(display_state.get("has_market_rows"))
            custom_enriched_count, custom_total_count = _custom_watchlist_enrichment_coverage(
                candidate_symbol_pool,
                display_mcap_map,
            ) if custom_mode_active else (0, 0)
            data_mode = _market_data_mode(
                has_market_rows=has_market_rows,
                used_major_fallback=used_major_fallback,
                custom_mode_active=custom_mode_active,
                custom_watchlist_enriched_count=custom_enriched_count,
                custom_watchlist_total_count=custom_total_count,
            )
            st.session_state["market_data_mode"] = data_mode

            universe_notice = _scan_universe_notice(
                candidate_count=int(notice_state.get("candidate_count") or 0),
                requested_n=requested_n,
                custom_mode_active=custom_mode_active,
                used_major_fallback=used_major_fallback,
                has_market_rows=bool(notice_state.get("has_market_rows")),
                source_pair_count=int(notice_state.get("source_pair_count") or 0),
                market_row_count=int(notice_state.get("market_row_count") or 0),
                top_n=int(top_n),
            )
            if universe_notice is not None:
                level, message = universe_notice
                _add_data_health_item(level, "Universe Scope", message)

            if skipped_symbols:
                st.session_state["market_scan_error_count"] = len(skipped_symbols)
                sample = ", ".join(f"{sym} ({err})" for sym, err in skipped_symbols[:3])
                more = "" if len(skipped_symbols) <= 3 else f" +{len(skipped_symbols) - 3} more"
                _add_data_health_item(
                    "warning",
                    "Partial Coverage",
                    f"Skipped {len(skipped_symbols)} / {len(attempted_symbols)} symbols. Sample: {sample}{more}.",
                )
            else:
                st.session_state["market_scan_error_count"] = 0

            if custom_mode_active:
                if not coingecko_coin_id_fallback_available:
                    unavailable_reason = (
                        f" Reason: {coingecko_coin_id_fallback_reason}."
                        if coingecko_coin_id_fallback_reason
                        else ""
                    )
                    _add_data_health_item(
                        "warning",
                        "Watchlist Backup",
                        "CoinGecko custom-watchlist backup is unavailable; exchange-missing custom symbols may stay hidden."
                        f"{unavailable_reason}",
                    )
                custom_missing = _custom_watchlist_missing_status(
                    custom_bases_applied,
                    fresh_results,
                    skipped_symbols,
                    coin_id_map=coin_id_map,
                    coingecko_coin_id_fallback_available=coingecko_coin_id_fallback_available,
                    coingecko_coin_id_fallback_reason=coingecko_coin_id_fallback_reason,
                )
                if custom_missing:
                    detail = " • ".join(f"{base}: {reason}" for base, reason in custom_missing)
                    _add_data_health_item(
                        "warning",
                        "Hidden Watchlist Coins",
                        f"Some custom watchlist coins could not be shown. Hidden: {detail}.",
                    )

            scan_degraded = bool(skipped_symbols) or (bool(attempted_symbols) and total_fetched_frame_count == 0)

            last_good_registry = _last_good_registry(
                st.session_state.get(_LAST_GOOD_REGISTRY_KEY),
                legacy_sig=st.session_state.get(_LAST_GOOD_SIG_KEY),
                legacy_results=st.session_state.get(_LAST_GOOD_RESULTS_KEY),
                legacy_ts=st.session_state.get(_LAST_GOOD_TS_KEY),
                legacy_mode=st.session_state.get(_LAST_GOOD_MODE_KEY),
            )
            if last_good_registry:
                st.session_state[_LAST_GOOD_REGISTRY_KEY] = last_good_registry
            else:
                st.session_state.pop(_LAST_GOOD_REGISTRY_KEY, None)
            last_good_snapshot = _last_good_snapshot_for_sig(last_good_registry, scan_sig)
            last_good_results = (
                list(last_good_snapshot.get("results", []))
                if isinstance(last_good_snapshot, dict)
                else []
            )
            limit_n = len(custom_bases_applied) if custom_mode_active else int(top_n)
            fresh_results = _sync_market_cap_cells(fresh_results, display_mcap_map, readable_market_cap)
            live_result_count_before_limit = len(fresh_results)
            live_produced_rows = list(fresh_results)
            # Sort by setup priority: TREND+AI > TREND-led > AI-led > WATCH > SKIP,
            # then structure, confidence, and market-cap tie-breakers.
            fresh_results = sorted(
                fresh_results,
                key=lambda row: _market_result_priority_key_for_mode(row, scan_mode),
            )[:limit_n]
            live_ranked_out_count = max(0, live_result_count_before_limit - len(fresh_results))
            scan_observed_at = pd.Timestamp.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")
            trace_source_label = "LIVE (DEGRADED)" if scan_degraded else "LIVE"
            try:
                log_scanner_trace_events(
                    _scanner_trace_events(
                        candidate_symbols=list(candidate_symbol_pool or []),
                        attempted_symbols=set(attempted_symbols or set()),
                        skipped_symbols=list(skipped_symbols or []),
                        produced_rows=list(live_produced_rows or []),
                        visible_rows=list(fresh_results or []),
                        market_rows=list(unique_market_data or []),
                        timeframe=str(timeframe),
                        scan_mode=str(scan_mode),
                        direction_filter=str(direction_filter),
                        observed_at=scan_observed_at,
                        source_label=trace_source_label,
                        data_mode=str(data_mode),
                    ),
                    db_path=signal_tracker_db_path,
                )
            except Exception as e:
                _debug(
                    f"Scanner trace log failed for Market scan ({timeframe}): "
                    f"{e.__class__.__name__}: {str(e).strip()}"
                )
            if fresh_results:
                st.session_state["market_scan_results"] = fresh_results
                st.session_state["market_scan_sig"] = scan_sig
                current_source = "LIVE (DEGRADED)" if scan_degraded else "LIVE"
                st.session_state["market_scan_source"] = current_source
                st.session_state["market_data_mode"] = data_mode
                source_label = current_source
                results = fresh_results
                if not scan_degraded:
                    ts_now = pd.Timestamp.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")
                    last_good_registry = _remember_last_good_snapshot(
                        last_good_registry,
                        scan_sig,
                        fresh_results,
                        ts_now,
                        data_mode,
                    )
                    st.session_state[_LAST_GOOD_REGISTRY_KEY] = last_good_registry
                    st.session_state[_LAST_GOOD_RESULTS_KEY] = fresh_results
                    st.session_state[_LAST_GOOD_SIG_KEY] = scan_sig
                    st.session_state[_LAST_GOOD_TS_KEY] = ts_now
                    st.session_state[_LAST_GOOD_MODE_KEY] = data_mode
                    healthy_empty_registry = _clear_healthy_empty_sig(
                        _healthy_empty_registry(st.session_state.get(_LAST_HEALTHY_EMPTY_SIG_KEY)),
                        scan_sig,
                    )
                    if healthy_empty_registry:
                        st.session_state[_LAST_HEALTHY_EMPTY_SIG_KEY] = healthy_empty_registry
                    else:
                        st.session_state.pop(_LAST_HEALTHY_EMPTY_SIG_KEY, None)
            else:
                st.session_state["market_scan_sig"] = scan_sig
                cache_sig = scan_sig if last_good_results else None
                ts = str(last_good_snapshot.get("ts") or "unknown time") if isinstance(last_good_snapshot, dict) else "unknown time"
                last_good_mode = (
                    str(last_good_snapshot.get("mode") or data_mode)
                    if isinstance(last_good_snapshot, dict)
                    else data_mode
                )
                healthy_empty_registry = _healthy_empty_registry(st.session_state.get(_LAST_HEALTHY_EMPTY_SIG_KEY))
                healthy_empty_seen = _healthy_empty_seen_for_sig(healthy_empty_registry, scan_sig)
                if _should_use_cached_scan(
                    prev_results=last_good_results,
                    cache_sig=cache_sig,
                    scan_sig=scan_sig,
                    cache_ts=ts,
                    ttl_minutes=CACHE_TTL_MINUTES,
                    scan_degraded=scan_degraded,
                    healthy_empty_seen=healthy_empty_seen,
                ):
                    results = last_good_results
                    st.session_state["market_scan_results"] = results
                    data_mode = last_good_mode
                    source_label = f"CACHED ({ts})"
                    st.session_state["market_scan_source"] = source_label
                    st.session_state["market_data_mode"] = data_mode
                    _add_data_health_item(
                        "warning",
                        "Cached Snapshot",
                        f"Live scan returned no rows. Showing last successful snapshot from {ts} for the same timeframe/filter.",
                    )
                else:
                    results = []
                    st.session_state["market_scan_results"] = []
                    source_label = "LIVE (DEGRADED)" if scan_degraded else "LIVE"
                    st.session_state["market_scan_source"] = source_label
                    st.session_state["market_data_mode"] = data_mode
                    has_other_cached_snapshot = bool(last_good_registry) and not bool(last_good_results)
                    if has_other_cached_snapshot:
                        _add_data_health_item(
                            "warning",
                            "No Matching Snapshot",
                            "Live scan returned no rows for the current timeframe/filter; stale cache from another setting was not used.",
                        )
                    elif healthy_empty_seen:
                        _add_data_health_item(
                            "info",
                            "Healthy Empty Read",
                            "Latest healthy live scan for this timeframe/filter had no candidates; older cached setups were suppressed.",
                        )
                    elif scan_degraded and last_good_results:
                        _add_data_health_item(
                            "warning",
                            "Stale Snapshot Blocked",
                            f"Live scan returned no rows and cache is older than {CACHE_TTL_MINUTES} minutes; stale snapshot was not used.",
                        )
                    elif scan_degraded:
                        _add_data_health_item(
                            "warning",
                            "Partial Empty Read",
                            "Live market read was partial and produced no usable candidates; no last-good snapshot was available.",
                        )
                    else:
                        _add_data_health_item(
                            "info",
                            "No Candidates",
                            "Live scan completed successfully but found no current candidates for this timeframe/filter.",
                        )
                    if not scan_degraded:
                        st.session_state[_LAST_HEALTHY_EMPTY_SIG_KEY] = _remember_healthy_empty_sig(
                            healthy_empty_registry,
                            scan_sig,
                        )
            st.session_state[_LAST_SCAN_ATTEMPT_TS_KEY] = scan_observed_at
            st.session_state[_DATA_HEALTH_ITEMS_KEY] = list(data_health_items)

    # Prepare DataFrame for display
    if results:
        source_is_degraded = "DEGRADED" in source_label.upper()
        source_color = WARNING if (source_is_degraded or source_label.startswith("CACHED")) else POSITIVE
        source_chip = (
            "Partial Live"
            if source_is_degraded
            else ("Live Data" if source_label.startswith("LIVE") else "Cached Snapshot")
        )
        source_display_label = str(source_label).replace("DEGRADED", "PARTIAL")
        mode_color = ACCENT if data_mode in {"FULL MARKET MODE", "CUSTOM WATCHLIST MODE"} else WARNING
        render_help_details(
            st,
            summary="How to read this table (?)",
            body_html=copy_text("market.help.scanner_guide_html"),
        )
        _render_data_health_band(
            source_chip=source_chip,
            source_color=source_color,
            source_display_label=source_display_label,
            mode_label=_market_data_mode_display(str(data_mode)),
            mode_color=mode_color,
            items=_data_health_display_items(data_health_items, source_label_text=source_label, data_mode_text=str(data_mode)),
        )
        show_advanced = st.toggle("Show advanced columns", value=False, key="market_show_adv_cols")
        show_diagnostics = bool(st.session_state.get("market_show_diagnostics", False))

        df_results = pd.DataFrame(results)
        df_live_produced = pd.DataFrame(live_produced_rows) if live_produced_rows else pd.DataFrame()

        def _distribution_bundle(df: pd.DataFrame) -> dict[str, object]:
            total = int(len(df))
            if total <= 0:
                return {
                    "total": 0,
                    "setup_counts": {"Ready": 0, "Probe": 0, "Watch": 0, "Skip": 0},
                    "direction_counts": {},
                    "ai_ensemble_counts": {},
                    "ai_confidence_counts": {},
                    "emerging_counts": {},
                    "enter_count": 0,
                    "probe_count": 0,
                    "watch_count": 0,
                    "skip_count": 0,
                }
            if "__action_raw" in df.columns:
                action_series = df["__action_raw"].astype(str)
            else:
                action_series = df.get("Setup Confirm", pd.Series(dtype=str)).astype(str)
            action_class_series = action_series.apply(_setup_confirm_class)
            enter_count_local = int(action_class_series.str.startswith("ENTER_").sum())
            probe_count_local = int((action_class_series == "PROBE").sum())
            watch_count_local = int((action_class_series == "WATCH").sum())
            skip_count_local = int((action_class_series == "SKIP").sum())
            direction_counts_local = (
                df["Direction"].astype(str).replace("", "Unknown").value_counts(dropna=False).to_dict()
                if "Direction" in df.columns
                else {}
            )
            ai_ensemble_counts_local = (
                df["AI Ensemble"].apply(_extract_ai_verdict).value_counts(dropna=False).to_dict()
                if "AI Ensemble" in df.columns
                else {}
            )
            ai_confidence_counts_local = (
                df["AI Confidence"].apply(_extract_confidence_label).value_counts(dropna=False).to_dict()
                if "AI Confidence" in df.columns
                else {}
            )
            emerging_counts_local = (
                df["__emerging_label"]
                .astype(str)
                .replace("", pd.NA)
                .dropna()
                .value_counts(dropna=False)
                .to_dict()
                if "__emerging_label" in df.columns
                else {}
            )
            return {
                "total": total,
                "setup_counts": {
                    "Ready": enter_count_local,
                    "Probe": probe_count_local,
                    "Watch": watch_count_local,
                    "Skip": skip_count_local,
                },
                "direction_counts": direction_counts_local,
                "ai_ensemble_counts": ai_ensemble_counts_local,
                "ai_confidence_counts": ai_confidence_counts_local,
                "emerging_counts": emerging_counts_local,
                "enter_count": enter_count_local,
                "probe_count": probe_count_local,
                "watch_count": watch_count_local,
                "skip_count": skip_count_local,
            }

        # Quick scan health summary (visual-first, logic unchanged)
        shown_bundle = _distribution_bundle(df_results)
        produced_bundle = _distribution_bundle(df_live_produced)
        # Keep the decision surface anchored to the same visible table slice.
        # The broader produced universe still appears in diagnostics below.
        gate_bundle = shown_bundle

        def _lead_component_display(value: float) -> tuple[float, str]:
            clipped = max(-100.0, min(100.0, float(value)))
            display = 50.0 + clipped * 0.5
            if clipped >= 15.0:
                return display, POSITIVE
            if clipped <= -15.0:
                return display, NEGATIVE
            return display, WARNING

        market_lead_snapshot = _market_lead_snapshot(
            produced_rows=live_produced_rows,
            delta_mcap=delta_mcap if pd.notna(delta_mcap) else None,
            btc_change=float(btc_change) if btc_change is not None and not pd.isna(btc_change) else None,
            eth_change=float(eth_change) if eth_change is not None and not pd.isna(eth_change) else None,
            btc_dom=btc_dom_display,
            eth_dom=eth_dom_display,
            custom_mode_active=custom_mode_active,
        )
        sector_rotation_snapshot = build_sector_rotation_snapshot(list(live_produced_rows or []))
        catalyst_events = get_market_catalyst_events()
        market_catalyst_snapshot = build_market_catalyst_snapshot(catalyst_events)
        market_flow_rows = get_market_flow_proxy_rows()
        market_flow_snapshot = build_market_flow_proxy_snapshot(market_flow_rows)
        current_session_bucket = session_bucket_for_timestamp()
        session_fit_snapshot = build_session_fit_snapshot(adaptive_model, current_session_bucket)
        market_regime_snapshot = build_market_regime_snapshot(
            setup_quality_score=float(composite_score),
            setup_quality_mode=str(composite_mode),
            market_lead_score=float(market_lead_snapshot.score),
            market_lead_state=str(market_lead_snapshot.state),
            lead_breadth_component=float(market_lead_snapshot.breadth_component),
            lead_rotation_component=float(market_lead_snapshot.rotation_component),
            lead_flow_component=float(market_lead_snapshot.flow_component),
            lead_dominance_component=float(market_lead_snapshot.dominance_component),
            direction_score=float(direction_score),
            breadth_score=float(breadth_score),
            trust_score=float(trust_score),
        )
        market_archive_guardrail_snapshot = build_archive_guardrail_snapshot(
            adaptive_model,
            signal={
                "Market Lead": str(getattr(market_lead_snapshot, "label", "") or "No Clear Pressure"),
                "Market Regime": str(getattr(market_regime_snapshot, "label", "") or "Unknown"),
                "Playbook Key": str(getattr(market_regime_snapshot, "playbook_key", "") or "Unknown"),
                "Playbook": str(getattr(market_regime_snapshot, "playbook", "") or "Unknown"),
                "Trade Gate": "Unknown",
                "Sector Rotation": str(getattr(sector_rotation_snapshot, "label", "") or "Unknown"),
                "Catalyst State": str(getattr(market_catalyst_snapshot, "label", "") or "Unknown"),
                "Catalyst Window": catalyst_window_label(market_catalyst_snapshot),
                "Catalyst Scope": str(getattr(market_catalyst_snapshot, "scope", "") or "Unknown"),
                "Catalyst Targeting": "Targeted" if bool(getattr(market_catalyst_snapshot, "targeted_only", False)) else "Market-Wide",
                "Flow Proxy": str(getattr(market_flow_snapshot, "label", "") or "Unknown"),
                "Session": current_session_bucket,
                "Timeframe": str(timeframe or "Unknown"),
            },
        )
        market_trade_gate_snapshot = build_market_trade_gate(
            market_regime_snapshot=market_regime_snapshot,
            market_catalyst_snapshot=market_catalyst_snapshot,
            scan_degraded=bool(scan_degraded),
            setup_quality_score=float(composite_score),
            setup_quality_mode=str(composite_mode),
            market_lead_score=float(market_lead_snapshot.score),
            market_lead_state=str(market_lead_snapshot.state),
            direction_score=float(direction_score),
            breadth_score=float(breadth_score),
            trust_score=float(trust_score),
            ready_count=int(gate_bundle["enter_count"]),
            probe_count=int(gate_bundle["probe_count"]),
            watch_count=int(gate_bundle["watch_count"]),
            skip_count=int(gate_bundle["skip_count"]),
            session_fit_score=float(getattr(session_fit_snapshot, "score", 0.0) or 0.0),
            session_fit_label=str(getattr(session_fit_snapshot, "label", "") or ""),
            session_fit_note=str(getattr(session_fit_snapshot, "note", "") or ""),
            archive_guardrail_penalty=float(getattr(market_archive_guardrail_snapshot, "penalty", 0.0) or 0.0),
            archive_guardrail_label=str(getattr(market_archive_guardrail_snapshot, "label", "") or ""),
            archive_guardrail_note=str(getattr(market_archive_guardrail_snapshot, "note", "") or ""),
        )
        trade_gate_calibration_snapshot = build_trade_gate_calibration_snapshot(
            trade_gate_calibration_model,
            signal={
                "Trade Gate Key": str(getattr(market_trade_gate_snapshot, "gate_key", "") or "Unknown"),
                "Trade Gate": str(getattr(market_trade_gate_snapshot, "label", "") or "Unknown"),
                "Playbook Key": str(getattr(market_regime_snapshot, "playbook_key", "") or "Unknown"),
                "Playbook": str(getattr(market_regime_snapshot, "playbook", "") or "Unknown"),
                "Market Regime": str(getattr(market_regime_snapshot, "label", "") or "Unknown"),
                "Session": current_session_bucket,
                "Catalyst Window": catalyst_window_label(market_catalyst_snapshot),
            },
        )
        market_trade_gate_snapshot = apply_market_trade_gate_archive_calibration(
            market_trade_gate_snapshot,
            calibration_delta=float(getattr(trade_gate_calibration_snapshot, "delta", 0.0) or 0.0),
            calibration_note=str(getattr(trade_gate_calibration_snapshot, "note", "") or ""),
        )
        market_default_budget_snapshot = market_default_risk_budget(
            market_trade_gate_snapshot,
            market_catalyst_snapshot,
        )
        archive_log_rows = _archive_learning_rows(
            visible_rows=list(results or []),
            produced_rows=list(live_produced_rows or []),
        )
        archive_signal_decision_cache: dict[tuple[str, str, str, str], object] = {}
        for row in archive_log_rows:
            direction_raw = str(row.get("Direction") or "")
            ai_ensemble = str(row.get("AI Ensemble") or "").strip()
            ai_direction = ai_ensemble.split("(", 1)[0].strip()
            catalyst_window = catalyst_window_label(market_catalyst_snapshot)
            if not bool(row.get("__setup_calibrated")):
                setup_calibration_snapshot = build_setup_calibration_snapshot(
                    setup_calibration_model,
                    signal={
                        "Setup Confirm": str(row.get("__action_raw", row.get("Setup Confirm", "")) or ""),
                        "AI Alignment": (
                            "Aligned"
                            if _signal_tracker_direction_key(direction_raw) in {"UPSIDE", "DOWNSIDE"}
                            and _signal_tracker_direction_key(direction_raw) == _signal_tracker_direction_key(ai_direction)
                            else "Not aligned"
                        ),
                        "Timeframe": str(row.get("__timeframe") or timeframe or "Unknown"),
                        "Scan Focus": str(_normalize_scan_mode(scan_mode) or "Unknown"),
                        "Direction": direction_raw,
                    },
                )
                calibrated_action_raw, calibrated_action_reason = apply_setup_archive_calibration(
                    str(row.get("__action_raw", row.get("Setup Confirm", "")) or ""),
                    str(row.get("__action_reason", "") or ""),
                    calibration_delta=float(getattr(setup_calibration_snapshot, "delta", 0.0) or 0.0),
                )
                row["__action_raw"] = calibrated_action_raw
                row["__action_reason"] = calibrated_action_reason
                row["Setup Confirm"] = _setup_confirm_display(
                    calibrated_action_raw,
                    calibrated_action_reason,
                    direction=str(row.get("Direction", "")).strip(),
                )
                row["__setup_calibration_delta"] = float(getattr(setup_calibration_snapshot, "delta", 0.0) or 0.0)
                row["__setup_calibration_note"] = str(getattr(setup_calibration_snapshot, "note", "") or "")
                row["__setup_calibrated"] = True
            if "__ai_score_raw" in row:
                ai_confidence_calibration_snapshot = build_ai_confidence_calibration_snapshot(
                    ai_confidence_calibration_model,
                    signal={
                        "Setup Confirm": str(row.get("__action_raw", row.get("Setup Confirm", "")) or ""),
                        "AI Alignment": (
                            "Aligned"
                            if _signal_tracker_direction_key(direction_raw) in {"UPSIDE", "DOWNSIDE"}
                            and _signal_tracker_direction_key(direction_raw) == _signal_tracker_direction_key(ai_direction)
                            else "Not aligned"
                        ),
                        "Timeframe": str(row.get("__timeframe") or timeframe or "Unknown"),
                        "Scan Focus": str(_normalize_scan_mode(scan_mode) or "Unknown"),
                        "Direction": direction_raw,
                    },
                )
                ai_confidence_snapshot = build_ai_confidence_snapshot(
                    direction=str(row.get("__ai_direction_raw") or ""),
                    combined_score=float(row.get("__ai_score_raw") or 0.0),
                    conviction_quality=float(row.get("__ai_conviction_quality_raw") or 0.0),
                    timeframe_alignment=float(row.get("__ai_timeframe_alignment_raw") or 0.0),
                    consensus_quality=float(row.get("__ai_consensus_quality_raw") or 0.0),
                    support_votes=int(row.get("__ai_votes") or 0),
                    timeframe_conflict=bool(row.get("__ai_timeframe_conflict_raw")),
                    degraded_data=bool(row.get("__ai_degraded_data_raw")),
                    archive_calibration_delta=float(getattr(ai_confidence_calibration_snapshot, "delta", 0.0) or 0.0),
                    archive_calibration_note=str(getattr(ai_confidence_calibration_snapshot, "note", "") or ""),
                )
                ai_snapshot_proxy = SimpleNamespace(
                    direction=str(row.get("__ai_direction_raw") or ""),
                    degraded_data=bool(row.get("__ai_degraded_data_raw")),
                    timeframe_conflict=bool(row.get("__ai_timeframe_conflict_raw")),
                    support_votes=int(row.get("__ai_votes") or 0),
                )
                row["AI Confidence"] = _ai_confidence_badge(
                    ai_snapshot_proxy,
                    float(ai_confidence_snapshot.score),
                )
                row["__ai_confidence_note"] = _ai_confidence_note(
                    ai_snapshot_proxy,
                    float(ai_confidence_snapshot.score),
                    ai_confidence_snapshot,
                )
                row["__ai_confidence_val"] = float(ai_confidence_snapshot.score)
            adaptive_snapshot = build_live_signal_adaptive_snapshot(
                adaptive_model,
                signal={
                    "Setup Confirm": str(row.get("__action_raw", row.get("Setup Confirm", "")) or ""),
                    "Lead": "LEAD" if _signal_tracker_direction_key(row.get("__emerging_direction")) in {"UPSIDE", "DOWNSIDE"} else "No LEAD",
                    "AI Alignment": (
                        "Aligned"
                        if _signal_tracker_direction_key(direction_raw) in {"UPSIDE", "DOWNSIDE"}
                        and _signal_tracker_direction_key(direction_raw) == _signal_tracker_direction_key(ai_direction)
                        else "Not aligned"
                    ),
                    "Market Lead": str(getattr(market_lead_snapshot, "label", "") or "No Clear Pressure"),
                    "Market Regime": str(getattr(market_regime_snapshot, "label", "") or "Unknown"),
                    "Playbook Key": str(getattr(market_regime_snapshot, "playbook_key", "") or "Unknown"),
                    "Playbook": str(getattr(market_regime_snapshot, "playbook", "") or "Unknown"),
                    "Trade Gate Key": str(getattr(market_trade_gate_snapshot, "gate_key", "") or "Unknown"),
                    "Trade Gate": str(getattr(market_trade_gate_snapshot, "label", "") or "Unknown"),
                    "Sector Rotation": str(getattr(sector_rotation_snapshot, "label", "") or "Unknown"),
                    "Catalyst State": str(getattr(market_catalyst_snapshot, "label", "") or "Unknown"),
                    "Catalyst Window": catalyst_window,
                    "Catalyst Scope": str(getattr(market_catalyst_snapshot, "scope", "") or "Unknown"),
                    "Catalyst Targeting": "Targeted" if bool(getattr(market_catalyst_snapshot, "targeted_only", False)) else "Market-Wide",
                    "Flow Proxy": str(getattr(market_flow_snapshot, "label", "") or "Unknown"),
                    "Session": session_bucket_for_timestamp(row.get("__event_time")),
                    "Timeframe": str(row.get("__timeframe") or timeframe or "Unknown"),
                },
            )
            row["__adaptive_edge_score"] = float(getattr(adaptive_snapshot, "score", 50.0) or 50.0)
            row["__adaptive_edge_label"] = str(getattr(adaptive_snapshot, "label", "") or "")
            row["__adaptive_edge_note"] = str(getattr(adaptive_snapshot, "note", "") or "")
            row["__execution_fit_note"] = str(getattr(adaptive_snapshot, "execution_fit_note", "") or "")
            row["__session_fit_score"] = float(getattr(adaptive_snapshot, "session_fit_score", 0.0) or 0.0)
            row["__session_fit_note"] = str(getattr(adaptive_snapshot, "session_fit_note", "") or "")
            row["__archive_guardrail_penalty"] = float(getattr(adaptive_snapshot, "archive_guardrail_penalty", 0.0) or 0.0)
            row["__archive_guardrail_label"] = str(getattr(adaptive_snapshot, "archive_guardrail_label", "") or "")
            row["__archive_guardrail_note"] = str(getattr(adaptive_snapshot, "archive_guardrail_note", "") or "")
            row["__catalyst_fit_note"] = catalyst_signal_note(
                market_catalyst_snapshot,
                symbol=str(row.get("Coin") or ""),
                sector_tag=str(classify_symbol_sector(str(row.get("Coin") or "")) or ""),
            )
            risk_sizing_calibration_snapshot = build_risk_sizing_calibration_snapshot(
                risk_sizing_calibration_model,
                signal={
                    "Setup Confirm": str(row.get("__action_raw", row.get("Setup Confirm", "")) or ""),
                    "AI Alignment": (
                        "Aligned"
                        if _signal_tracker_direction_key(direction_raw) in {"UPSIDE", "DOWNSIDE"}
                        and _signal_tracker_direction_key(direction_raw) == _signal_tracker_direction_key(ai_direction)
                        else "Not aligned"
                    ),
                    "Timeframe": str(row.get("__timeframe") or timeframe or "Unknown"),
                    "Scan Focus": str(_normalize_scan_mode(scan_mode) or "Unknown"),
                    "Direction": direction_raw,
                },
            )
            live_risk_sizing_snapshot = build_signal_risk_sizing(
                market_trade_gate_snapshot=market_trade_gate_snapshot,
                market_catalyst_snapshot=market_catalyst_snapshot,
                direction=direction_raw,
                setup_confirm=str(row.get("__action_raw", row.get("Setup Confirm", "")) or ""),
                confidence=row.get("__confidence_val", _confidence_value_from_badge(row.get("Confidence"))),
                ai_confidence=row.get("__ai_confidence_val", _confidence_value_from_badge(row.get("AI Confidence"))),
                ai_aligned=(
                    _signal_tracker_direction_key(direction_raw) in {"UPSIDE", "DOWNSIDE"}
                    and _signal_tracker_direction_key(direction_raw) == _signal_tracker_direction_key(ai_direction)
                ),
                market_lead_aligned=(
                    _signal_tracker_direction_key(direction_raw) in {"UPSIDE", "DOWNSIDE"}
                    and _signal_tracker_direction_key(direction_raw) == _signal_tracker_direction_key(market_lead_snapshot.label)
                ),
                lead_active=_signal_tracker_direction_key(row.get("__emerging_direction")) in {"UPSIDE", "DOWNSIDE"},
                rr_ratio=row.get("__rr_val"),
                adaptive_edge_score=row.get("__adaptive_edge_score"),
                session_fit_score=row.get("__session_fit_score"),
                archive_guardrail_penalty=row.get("__archive_guardrail_penalty"),
                archive_guardrail_label=row.get("__archive_guardrail_label"),
                archive_guardrail_note=row.get("__archive_guardrail_note"),
                archive_risk_delta=float(getattr(risk_sizing_calibration_snapshot, "delta", 0.0) or 0.0),
                archive_risk_note=str(getattr(risk_sizing_calibration_snapshot, "note", "") or ""),
                symbol=str(row.get("Coin") or ""),
                sector_tag=str(classify_symbol_sector(str(row.get("Coin") or "")) or ""),
            )
            row["__risk_tier_label"] = str(getattr(live_risk_sizing_snapshot, "label", "") or "")
            row["__risk_unit_fraction"] = float(getattr(live_risk_sizing_snapshot, "unit_fraction", 0.0) or 0.0)
            row["__risk_archive_delta"] = float(getattr(risk_sizing_calibration_snapshot, "delta", 0.0) or 0.0)
            row["__risk_archive_note"] = str(getattr(risk_sizing_calibration_snapshot, "note", "") or "")
            row["__actionable_context_score"] = _actionable_context_score(
                adaptive_edge_score=row.get("__adaptive_edge_score"),
                session_fit_score=row.get("__session_fit_score"),
                archive_guardrail_penalty=row.get("__archive_guardrail_penalty"),
                direction=direction_raw,
                market_lead_state=str(getattr(market_lead_snapshot, "state", "") or ""),
                symbol=str(row.get("Coin") or ""),
                classify_symbol_sector=classify_symbol_sector,
                sector_rotation_snapshot=sector_rotation_snapshot,
            )
            actionable_archive_snapshot = build_actionable_ranking_snapshot(
                actionable_ranking_model,
                signal={
                    "Setup Confirm": str(row.get("__action_raw", row.get("Setup Confirm", "")) or ""),
                    "AI Alignment": (
                        "Aligned"
                        if _signal_tracker_direction_key(direction_raw) in {"UPSIDE", "DOWNSIDE"}
                        and _signal_tracker_direction_key(direction_raw) == _signal_tracker_direction_key(ai_direction)
                        else "Not aligned"
                    ),
                    "Timeframe": str(row.get("__timeframe") or timeframe or "Unknown"),
                    "Scan Focus": str(_normalize_scan_mode(scan_mode) or "Unknown"),
                    "Direction": direction_raw,
                },
            )
            archive_policy_snapshot = archive_policy_for_signal(
                archive_policy_map,
                symbol=str(row.get("Coin") or ""),
                timeframe=str(row.get("__timeframe") or timeframe or ""),
                setup_confirm=str(row.get("__action_raw", row.get("Setup Confirm", "")) or ""),
                direction=direction_raw,
            )
            archive_signal_decision_key = (
                str(row.get("Coin") or "").strip().upper(),
                str(row.get("__timeframe") or timeframe or "").strip().lower(),
                str(row.get("__action_raw", row.get("Setup Confirm", "")) or "").strip().upper(),
                str(direction_raw or "").strip().upper(),
            )
            archive_signal_decision = archive_signal_decision_cache.get(archive_signal_decision_key)
            if archive_signal_decision is None:
                try:
                    archive_signal_decision = build_archive_signal_decision_snapshot(
                        df_events=adaptive_history_df,
                        df_forward_windows=adaptive_forward_windows_df,
                        symbol=str(row.get("Coin") or ""),
                        timeframe=str(row.get("__timeframe") or timeframe or ""),
                        setup_confirm=str(row.get("__action_raw", row.get("Setup Confirm", "")) or ""),
                        direction=direction_raw,
                    )
                except Exception as e:
                    if callable(_debug):
                        _debug(
                            "Archive decision snapshot unavailable for "
                            f"{str(row.get('Coin') or '').strip()} "
                            f"({e.__class__.__name__}: {str(e).strip()})"
                        )
                    archive_signal_decision = SimpleNamespace(
                        available=False,
                        scope_label="",
                        hold_window={},
                        expected_path={},
                    )
                archive_signal_decision_cache[archive_signal_decision_key] = archive_signal_decision
            archive_decision_delta, archive_expectancy_delta = archive_decision_score_adjustment(
                archive_signal_decision
            )
            archive_policy_delta = float(getattr(archive_policy_snapshot, "policy_delta", 0.0) or 0.0)
            archive_policy_coverage = float(getattr(archive_policy_snapshot, "coverage_factor", 0.0) or 0.0)
            base_actionable_archive_score = float(getattr(actionable_archive_snapshot, "delta", 0.0) or 0.0)
            base_expectancy_bias_score = _expectancy_bias_score(
                archive_delta=getattr(actionable_archive_snapshot, "delta", 0.0),
                bucket_resolved=getattr(actionable_archive_snapshot, "bucket_resolved", 0.0),
                matched_factors=getattr(actionable_archive_snapshot, "matched_factors", 0),
            )
            archive_score_before_decision, expectancy_score_before_decision = _coverage_adjusted_archive_scores(
                base_archive_score=base_actionable_archive_score,
                base_expectancy_score=base_expectancy_bias_score,
                policy_delta=archive_policy_delta,
                policy_coverage=archive_policy_coverage,
            )
            archive_total_delta = max(
                -20.0,
                min(20.0, _sortable_float(archive_score_before_decision) + archive_decision_delta),
            )
            archive_total_expectancy_delta = max(
                30.0,
                min(70.0, _sortable_float(expectancy_score_before_decision) + archive_expectancy_delta),
            )
            archive_decision_feedback = archive_decision_feedback_for_signal(
                archive_decision_feedback_map,
                archive_decision_feedback_model,
                symbol=str(row.get("Coin") or ""),
                timeframe=str(row.get("__timeframe") or timeframe or ""),
                setup_confirm=str(row.get("__action_raw", row.get("Setup Confirm", "")) or ""),
                direction=direction_raw,
                playbook_key=str(getattr(market_regime_snapshot, "playbook_key", "") or ""),
                trade_gate_key=str(getattr(market_trade_gate_snapshot, "gate_key", "") or ""),
            )
            archive_total_delta, archive_total_expectancy_edge = calibrate_archive_decision_scores(
                archive_total_delta,
                archive_total_expectancy_delta - 50.0,
                archive_decision_feedback,
            )
            archive_total_delta, archive_total_expectancy_edge = apply_archive_invalidation_guardrail(
                archive_total_delta,
                archive_total_expectancy_edge,
                archive_signal_decision,
            )
            archive_total_delta, archive_total_expectancy_edge = apply_archive_confidence_guardrail(
                archive_total_delta,
                archive_total_expectancy_edge,
                archive_signal_decision,
            )
            archive_observability = archive_decision_observability(
                archive_signal_decision,
                archive_decision_feedback,
            )
            archive_total_expectancy_delta = max(30.0, min(70.0, 50.0 + archive_total_expectancy_edge))
            row["__archive_policy_delta"] = archive_policy_delta
            row["__archive_policy_completed"] = int(getattr(archive_policy_snapshot, "completed", 0) or 0)
            row["__archive_policy_quality"] = str(getattr(archive_policy_snapshot, "quality_label", "") or "")
            row["__archive_policy_coverage"] = archive_policy_coverage
            row["__archive_decision_delta"] = archive_decision_delta
            row["__archive_expectancy_delta"] = archive_expectancy_delta
            row["__archive_total_delta"] = archive_total_delta
            row["__archive_total_expectancy_delta"] = archive_total_expectancy_delta
            row["__archive_decision_scope"] = str(getattr(archive_signal_decision, "scope_label", "") or "")
            row["__archive_confidence_factor"] = archive_observability["archive_confidence_factor"]
            row["__archive_confidence_tier"] = archive_observability["archive_confidence_tier"]
            row["__archive_invalidation_risk"] = archive_observability["archive_invalidation_risk"]
            row["__archive_feedback_multiplier"] = archive_observability["archive_feedback_multiplier"]
            row["__actionable_archive_score"] = archive_total_delta
            row["__expectancy_bias_score"] = archive_total_expectancy_delta
            row["__execution_friction_score"] = _execution_friction_score(
                mcap_val=row.get("__mcap_val"),
                volatility_label=str(row.get("Volatility") or ""),
                delta_pct=row.get("__delta_pct"),
                spike_present=bool(str(row.get("Spike Alert") or "").strip() or str(row.get("__spike_dir") or "").strip()),
                execution_confidence=row.get("__execution_confidence_val"),
            )
        results = sorted(
            list(results or []),
            key=lambda row: _market_result_priority_key_for_mode(row, scan_mode),
        )
        df_results = pd.DataFrame(results)
        shown_bundle = _distribution_bundle(df_results)
        produced_bundle = _distribution_bundle(df_live_produced)
        display_gate_bundle = shown_bundle
        display_market_trade_gate_snapshot = build_market_trade_gate(
            market_regime_snapshot=market_regime_snapshot,
            market_catalyst_snapshot=market_catalyst_snapshot,
            scan_degraded=bool(scan_degraded),
            setup_quality_score=float(composite_score),
            setup_quality_mode=str(composite_mode),
            market_lead_score=float(market_lead_snapshot.score),
            market_lead_state=str(market_lead_snapshot.state),
            direction_score=float(direction_score),
            breadth_score=float(breadth_score),
            trust_score=float(trust_score),
            ready_count=int(display_gate_bundle["enter_count"]),
            probe_count=int(display_gate_bundle["probe_count"]),
            watch_count=int(display_gate_bundle["watch_count"]),
            skip_count=int(display_gate_bundle["skip_count"]),
            session_fit_score=float(getattr(session_fit_snapshot, "score", 0.0) or 0.0),
            session_fit_label=str(getattr(session_fit_snapshot, "label", "") or ""),
            session_fit_note=str(getattr(session_fit_snapshot, "note", "") or ""),
            archive_guardrail_penalty=float(getattr(market_archive_guardrail_snapshot, "penalty", 0.0) or 0.0),
            archive_guardrail_label=str(getattr(market_archive_guardrail_snapshot, "label", "") or ""),
            archive_guardrail_note=str(getattr(market_archive_guardrail_snapshot, "note", "") or ""),
        )
        display_trade_gate_calibration_snapshot = build_trade_gate_calibration_snapshot(
            trade_gate_calibration_model,
            signal={
                "Trade Gate Key": str(getattr(display_market_trade_gate_snapshot, "gate_key", "") or "Unknown"),
                "Trade Gate": str(getattr(display_market_trade_gate_snapshot, "label", "") or "Unknown"),
                "Playbook Key": str(getattr(market_regime_snapshot, "playbook_key", "") or "Unknown"),
                "Playbook": str(getattr(market_regime_snapshot, "playbook", "") or "Unknown"),
                "Market Regime": str(getattr(market_regime_snapshot, "label", "") or "Unknown"),
                "Session": current_session_bucket,
                "Catalyst Window": catalyst_window_label(market_catalyst_snapshot),
            },
        )
        display_market_trade_gate_snapshot = apply_market_trade_gate_archive_calibration(
            display_market_trade_gate_snapshot,
            calibration_delta=float(getattr(display_trade_gate_calibration_snapshot, "delta", 0.0) or 0.0),
            calibration_note=str(getattr(display_trade_gate_calibration_snapshot, "note", "") or ""),
        )
        display_market_default_budget_snapshot = market_default_risk_budget(
            display_market_trade_gate_snapshot,
            market_catalyst_snapshot,
        )
        # Rebuild displayed risk tiers against the same final gate shown above.
        for row in list(results or []):
            direction_raw = str(row.get("Direction") or "")
            ai_ensemble = str(row.get("AI Ensemble") or "").strip()
            ai_direction = ai_ensemble.split("(", 1)[0].strip()
            scalp_display = _build_scalp_display_payload(
                timeframe_value=str(row.get("__timeframe") or timeframe or ""),
                scalp_direction=row.get("__scalp_direction_raw"),
                signal_direction=str(row.get("__signal_direction_raw") or ""),
                rr_ratio=row.get("__scalp_rr_val_raw"),
                adx_val=row.get("__adx_raw"),
                confidence=row.get("__execution_confidence_val"),
                conviction_label=str(row.get("__execution_conviction_label") or ""),
                entry=row.get("__scalp_entry_val_raw"),
                stop=row.get("__scalp_stop_val_raw"),
                target=row.get("__scalp_target_val_raw"),
                setup_confirm=str(row.get("__action_raw", row.get("Setup Confirm", "")) or ""),
                market_trade_gate_key=str(getattr(display_market_trade_gate_snapshot, "gate_key", "") or ""),
                archive_guardrail_penalty=row.get("__archive_guardrail_penalty"),
                archive_guardrail_label=str(row.get("__archive_guardrail_label") or ""),
                direction_value=direction_raw,
                ai_aligned=(
                    _signal_tracker_direction_key(direction_raw) in {"UPSIDE", "DOWNSIDE"}
                    and _signal_tracker_direction_key(direction_raw) == _signal_tracker_direction_key(ai_direction)
                ),
                scan_focus_value=str(_normalize_scan_mode(scan_mode) or "Unknown"),
                breakout_note=str(row.get("__scalp_breakout_note_raw") or ""),
            )
            row["Scalp Opportunity"] = str(scalp_display["label"] or "")
            row["__scalp_reason_text"] = str(scalp_display["reason_text"] or "")
            row["__scalp_reason_short"] = str(scalp_display["reason_short"] or "")
            row["__scalp_display_state"] = str(scalp_display["display_state"] or "")
            row["Entry Price"] = str(scalp_display["entry_display"] or "")
            row["__entry_val"] = scalp_display["entry_val"]
            row["__entry_note"] = str(scalp_display["entry_note"] or "") if scalp_display["display_state"] in {"LIVE", "CONDITIONAL"} else ""
            row["Stop Loss"] = str(scalp_display["stop_display"] or "")
            row["__stop_val"] = scalp_display["stop_val"]
            row["Target Price"] = str(scalp_display["target_display"] or "")
            row["__target_val"] = scalp_display["target_val"]
            row["__target_note"] = str(scalp_display["target_note"] or "") if scalp_display["display_state"] in {"LIVE", "CONDITIONAL"} else ""
            row["__rr_note"] = str(scalp_display["rr_note"] or "") if scalp_display["display_state"] in {"LIVE", "CONDITIONAL"} else ""
            row["R:R"] = str(scalp_display["rr_badge"] or "")
            row["__rr_val"] = scalp_display["rr_val"]
            live_risk_sizing_snapshot = build_signal_risk_sizing(
                market_trade_gate_snapshot=display_market_trade_gate_snapshot,
                market_catalyst_snapshot=market_catalyst_snapshot,
                direction=direction_raw,
                setup_confirm=str(row.get("__action_raw", row.get("Setup Confirm", "")) or ""),
                confidence=row.get("__confidence_val", _confidence_value_from_badge(row.get("Confidence"))),
                ai_confidence=row.get("__ai_confidence_val", _confidence_value_from_badge(row.get("AI Confidence"))),
                ai_aligned=(
                    _signal_tracker_direction_key(direction_raw) in {"UPSIDE", "DOWNSIDE"}
                    and _signal_tracker_direction_key(direction_raw) == _signal_tracker_direction_key(ai_direction)
                ),
                market_lead_aligned=(
                    _signal_tracker_direction_key(direction_raw) in {"UPSIDE", "DOWNSIDE"}
                    and _signal_tracker_direction_key(direction_raw) == _signal_tracker_direction_key(market_lead_snapshot.label)
                ),
                lead_active=_signal_tracker_direction_key(row.get("__emerging_direction")) in {"UPSIDE", "DOWNSIDE"},
                rr_ratio=row.get("__rr_val"),
                adaptive_edge_score=row.get("__adaptive_edge_score"),
                session_fit_score=row.get("__session_fit_score"),
                archive_guardrail_penalty=row.get("__archive_guardrail_penalty"),
                archive_guardrail_label=row.get("__archive_guardrail_label"),
                archive_guardrail_note=row.get("__archive_guardrail_note"),
                archive_risk_delta=float(row.get("__risk_archive_delta") or 0.0),
                archive_risk_note=str(row.get("__risk_archive_note") or ""),
                symbol=str(row.get("Coin") or ""),
                sector_tag=str(classify_symbol_sector(str(row.get("Coin") or "")) or ""),
            )
            row["__risk_tier_label"] = str(getattr(live_risk_sizing_snapshot, "label", "") or "")
            row["__risk_unit_fraction"] = float(getattr(live_risk_sizing_snapshot, "unit_fraction", 0.0) or 0.0)
        results = sorted(
            list(results or []),
            key=lambda row: _market_result_priority_key_for_mode(row, scan_mode),
        )
        df_results = pd.DataFrame(results)
        shown_bundle = _distribution_bundle(df_results)
        market_alerts = build_market_alerts(
            market_lead_snapshot=market_lead_snapshot,
            market_regime_snapshot=market_regime_snapshot,
            market_trade_gate_snapshot=display_market_trade_gate_snapshot,
            market_catalyst_snapshot=market_catalyst_snapshot,
            market_flow_snapshot=market_flow_snapshot,
            sector_rotation_snapshot=sector_rotation_snapshot,
            session_fit_snapshot=session_fit_snapshot,
            rows=list(results or []),
            max_alerts=4,
        )
        market_alerts = _rank_market_alerts_by_archive(list(market_alerts or []), adaptive_history_df)
        display_market_alerts = _compress_market_alerts_for_display(list(market_alerts or []), max_items=2)
        try:
            log_signal_events(
                _market_signal_log_events(
                    rows=list(archive_log_rows or []),
                    timeframe=str(timeframe),
                    scan_mode=str(scan_mode),
                    market_lead_snapshot=market_lead_snapshot,
                    market_regime_snapshot=market_regime_snapshot,
                    market_trade_gate_snapshot=display_market_trade_gate_snapshot,
                    build_signal_risk_sizing=build_signal_risk_sizing,
                    sector_rotation_snapshot=sector_rotation_snapshot,
                    classify_symbol_sector=classify_symbol_sector,
                    market_catalyst_snapshot=market_catalyst_snapshot,
                    market_flow_snapshot=market_flow_snapshot,
                    session_fit_snapshot=session_fit_snapshot,
                    market_alerts=market_alerts,
                ),
                db_path=signal_tracker_db_path,
            )
        except Exception as e:
            _debug(
                f"Signal tracker log failed for Market scan ({timeframe}): "
                f"{e.__class__.__name__}: {str(e).strip()}"
            )
        try:
            log_signal_events(
                _scalp_signal_log_events(
                    rows=list(archive_log_rows or []),
                    timeframe=str(timeframe),
                    scan_mode=str(scan_mode),
                    market_lead_snapshot=market_lead_snapshot,
                    market_regime_snapshot=market_regime_snapshot,
                    market_trade_gate_snapshot=display_market_trade_gate_snapshot,
                    build_signal_risk_sizing=build_signal_risk_sizing,
                    sector_rotation_snapshot=sector_rotation_snapshot,
                    classify_symbol_sector=classify_symbol_sector,
                    market_catalyst_snapshot=market_catalyst_snapshot,
                    market_flow_snapshot=market_flow_snapshot,
                    session_fit_snapshot=session_fit_snapshot,
                    market_alerts=market_alerts,
                ),
                db_path=signal_tracker_db_path,
            )
        except Exception as e:
            _debug(
                f"Signal tracker log failed for Scalp scan ({timeframe}): "
                f"{e.__class__.__name__}: {str(e).strip()}"
            )
        try:
            log_market_alerts(
                [
                    {
                        "alert_key": getattr(alert, "alert_key", ""),
                        "state_signature": getattr(alert, "state_signature", ""),
                        "severity": getattr(alert, "severity", "INFO"),
                        "title": getattr(alert, "title", ""),
                        "note": getattr(alert, "note", ""),
                    }
                    for alert in list(market_alerts or [])
                ],
                source="Market",
                db_path=signal_tracker_db_path,
            )
        except Exception as e:
            _debug(
                f"Market alert log failed for Market scan ({timeframe}): "
                f"{e.__class__.__name__}: {str(e).strip()}"
            )
        if market_lead_snapshot.state == "UPSIDE":
            market_lead_color = POSITIVE
        elif market_lead_snapshot.state == "DOWNSIDE":
            market_lead_color = NEGATIVE
        elif market_lead_snapshot.state == "BALANCED":
            market_lead_color = WARNING
        else:
            market_lead_color = TEXT_MUTED

        ai_direction_hover = (
            "AI summary across BTC, ETH, BNB, SOL, ADA, and XRP. "
            "Lower scores lean downside, the middle zone is neutral, and higher scores lean upside."
        )
        if behaviour_weight_mode == "equal":
            ai_direction_hover += " Dominance data is unavailable right now, so all coins are weighted equally."
        ai_direction_score = int(round(behaviour_prob * 100))

        setup_quality_hover = (
            "Overall market regime for finding setups. "
            "It blends direction clarity, market health, participation, and model trust, then maps that into the current playbook."
        )
        setup_mode_hover = {
            "Risk-On": (
                "Risk-On: market conditions are supportive enough to hunt setups more actively."
            ),
            "Selective": (
                "Selective: some setups can work, but confirmation should stay strict."
            ),
            "Risk-Off": (
                "Risk-Off: the market is weak or fragmented, so staying defensive makes more sense."
            ),
        }.get(composite_mode, setup_quality_hover)
        setup_quality_hover = (
            f"{setup_mode_hover} "
            f"Current playbook: {market_regime_snapshot.playbook}. "
            f"Breakdown -> Direction {int(round(direction_score))}, Regime {int(round(regime_score))}, "
            f"Breadth {int(round(breadth_score))}, Trust {int(round(trust_score))}."
        )

        market_lead_hover = (
            "Early market-pressure gauge before full confirmation. "
            "It asks whether the market is starting to lean up or down beneath the surface."
        )
        if custom_mode_active:
            market_lead_hover += " Custom watchlist mode uses less breadth data, so this card leans more on broad market internals."
        market_lead_hover = (
            f"{market_lead_hover} "
            f"Breakdown -> Breadth {int(round(market_lead_snapshot.breadth_component))}, "
            f"Rotation {int(round(market_lead_snapshot.rotation_component))}, "
            f"Flow {int(round(market_lead_snapshot.flow_component))}, "
            f"Dominance {int(round(market_lead_snapshot.dominance_component))}."
        )

        with market_signal_cards_placeholder:
            g1, g2, g3, g4, g5 = st.columns(5, gap="medium")
            with g1:
                btc_state, btc_color = _dom_state(btc_dom_display, AI_SHORT_THRESHOLD * 100, AI_LONG_THRESHOLD * 100)
                st.markdown(
                    _render_market_orbital_card(
                        title="BTC Dominance",
                        title_hover="Bitcoin's share of the crypto market. Rising BTC dominance usually means the market is getting more defensive.",
                        value_text=f"{float(btc_dom_display):.1f}" if btc_dom_display is not None else "N/A",
                        unit="%",
                        chart_html=_orbital_svg_html(
                            value=btc_dom_display,
                            segments=[
                                (0.0, AI_SHORT_THRESHOLD * 100, NEGATIVE),
                                (AI_SHORT_THRESHOLD * 100, AI_LONG_THRESHOLD * 100, WARNING),
                                (AI_LONG_THRESHOLD * 100, 100.0, POSITIVE),
                            ],
                            marker_color=btc_color,
                            accent_color=btc_color,
                        ),
                        guide_labels=("Alt-heavy", "Balanced", "BTC-led"),
                        note="Context only. High BTC share usually means money is hiding in Bitcoin first.",
                        footer_html=(
                            f"<div class='market-top-meta'><span>Current state</span><span><strong style='color:{btc_color};'>{btc_state}</strong></span></div>"
                        ),
                    ),
                    unsafe_allow_html=True,
                )
            with g2:
                eth_state, eth_color = _dom_state(eth_dom_display, 9.0, 13.0)
                st.markdown(
                    _render_market_orbital_card(
                        title="ETH Dominance",
                        title_hover="Ethereum's share of the crypto market. Rising ETH dominance usually means alt participation is improving.",
                        value_text=f"{float(eth_dom_display):.1f}" if eth_dom_display is not None else "N/A",
                        unit="%",
                        chart_html=_orbital_svg_html(
                            value=eth_dom_display,
                            segments=[
                                (0.0, 9.0, NEGATIVE),
                                (9.0, 13.0, WARNING),
                                (13.0, 100.0, POSITIVE),
                            ],
                            marker_color=eth_color,
                            accent_color=eth_color,
                        ),
                        guide_labels=("Muted", "Balanced", "ETH-led"),
                        note="Context only. Rising ETH share usually means broader alt participation is starting to improve.",
                        footer_html=(
                            f"<div class='market-top-meta'><span>Current state</span><span><strong style='color:{eth_color};'>{eth_state}</strong></span></div>"
                        ),
                    ),
                    unsafe_allow_html=True,
                )
            with g3:
                st.markdown(
                    _render_market_orbital_card(
                        title="Early Pressure",
                        title_hover=market_lead_hover,
                        value_text=f"{int(round(market_lead_snapshot.score)):d}",
                        unit="/100",
                        chart_html=_orbital_svg_html(
                            value=float(market_lead_snapshot.score),
                            segments=[
                                (0.0, 38.0, NEGATIVE),
                                (38.0, 62.0, WARNING),
                                (62.0, 100.0, POSITIVE),
                            ],
                            marker_color=market_lead_color,
                            accent_color=market_lead_color,
                        ),
                        guide_labels=("Downside", "Balanced", "Upside"),
                        note=market_lead_snapshot.note,
                        footer_html=(
                            f"<div class='market-top-meta'><span>Current state</span>"
                            f"<span><strong style='color:{market_lead_color};'>{html.escape(market_lead_snapshot.label)}</strong></span></div>"
                        ),
                    ),
                    unsafe_allow_html=True,
                )
            with g4:
                st.markdown(
                    _render_market_orbital_card(
                        title="AI Direction Bias",
                        title_hover=ai_direction_hover,
                        value_text=f"{ai_direction_score:d}",
                        unit="/100",
                        chart_html=_orbital_svg_html(
                            value=float(ai_direction_score),
                            segments=[
                                (0.0, AI_SHORT_THRESHOLD * 100, NEGATIVE),
                                (AI_SHORT_THRESHOLD * 100, AI_LONG_THRESHOLD * 100, WARNING),
                                (AI_LONG_THRESHOLD * 100, 100.0, POSITIVE),
                            ],
                            marker_color=behaviour_color,
                            accent_color=behaviour_color,
                        ),
                        guide_labels=("Downside", "Neutral", "Upside"),
                        note=(
                            "AI summary across BTC/ETH/BNB/SOL/ADA/XRP. "
                            + ("Dominance weighting is active." if behaviour_weight_mode == "dominance" else "Equal weights are active right now.")
                        ),
                        footer_html=(
                            f"<div class='market-top-meta'><span>Bias</span><span><strong style='color:{behaviour_color};'>{html.escape(behaviour_label)} Bias</strong></span></div>"
                        ),
                    ),
                    unsafe_allow_html=True,
                )
            with g5:
                st.markdown(
                    _render_market_orbital_card(
                        title="Market Regime",
                        title_hover=setup_quality_hover,
                        value_text=f"{int(round(composite_score)):d}",
                        unit="/100",
                        chart_html=_orbital_svg_html(
                            value=float(composite_score),
                            segments=[
                                (0.0, 52.0, NEGATIVE),
                                (52.0, 68.0, WARNING),
                                (68.0, 100.0, POSITIVE),
                            ],
                            marker_color=composite_color,
                            accent_color=composite_color,
                        ),
                        guide_labels=("Risk-Off", "Selective", "Risk-On"),
                        note=market_regime_snapshot.note,
                        footer_html=(
                            f"<div class='market-top-meta'><span>Regime</span>"
                            f"<span><strong style='color:{composite_color};'>{html.escape(str(composite_mode or 'Selective'))}</strong></span></div>"
                        ),
                    ),
                    unsafe_allow_html=True,
                )

        enter_count = int(shown_bundle["enter_count"])
        probe_count = int(shown_bundle["probe_count"])
        watch_count = int(shown_bundle["watch_count"])
        skip_count = int(shown_bundle["skip_count"])
        st.markdown(
            _trade_gate_banner_html(
                trade_gate_display_label(display_market_trade_gate_snapshot.label),
                _compact_trade_gate_note(
                    market_trade_gate_snapshot=display_market_trade_gate_snapshot,
                    market_catalyst_snapshot=market_catalyst_snapshot,
                    market_flow_snapshot=market_flow_snapshot,
                    market_default_budget_snapshot=display_market_default_budget_snapshot,
                    session_fit_snapshot=session_fit_snapshot,
                    adaptive_model=adaptive_model,
                    enter_count=enter_count,
                    probe_count=probe_count,
                ),
                display_market_trade_gate_snapshot.tone,
                display_market_trade_gate_snapshot.reason_code,
            ),
            unsafe_allow_html=True,
        )
        if display_market_alerts:
            st.markdown(
                _market_alert_strip_html(display_market_alerts, total_active=len(market_alerts)),
                unsafe_allow_html=True,
            )
        st.markdown("<div style='height:0.55rem;'></div>", unsafe_allow_html=True)
        lead_bundle = shown_bundle
        lead_scope_label = "Current table"
        emerging_counts = dict(lead_bundle["emerging_counts"])
        emerging_up_count = int(emerging_counts.get("Emerging Upside", 0))
        emerging_down_count = int(emerging_counts.get("Emerging Downside", 0))
        emerging_total = int(emerging_up_count + emerging_down_count)

        best_scalp_coin, best_scalp_sub = _pick_best_scalp_opportunity(df_results)

        clearest_direction_head, clearest_direction_sub = _pick_clearest_direction(df_results)

        status_label, status_head, status_sub = _setup_status_summary(
            enter_count=enter_count,
            probe_count=probe_count,
            watch_count=watch_count,
            skip_count=skip_count,
            source_label=source_label,
        )
        if emerging_total <= 0:
            emerging_head = "NO CLEAR PRESSURE"
        elif emerging_up_count > 0 and emerging_down_count <= 0:
            emerging_head = (
                f"{emerging_up_count} UPSIDE PUSH"
                if emerging_up_count == 1
                else f"{emerging_up_count} UPSIDE PUSHES"
            )
        elif emerging_down_count > 0 and emerging_up_count <= 0:
            emerging_head = (
                f"{emerging_down_count} DOWNSIDE PUSH"
                if emerging_down_count == 1
                else f"{emerging_down_count} DOWNSIDE PUSHES"
            )
        else:
            emerging_head = f"{emerging_total} PRESSURE MOVES"
        sector_meta = (
            f" • Sector: {sector_rotation_snapshot.leader_sector}"
            if emerging_total > 0
            and str(getattr(sector_rotation_snapshot, 'leader_sector', '')).strip() not in {"", "None", "Other"}
            else ""
        )
        emerging_sub = (
            f"{lead_scope_label} • Upside: {emerging_up_count} • Downside: {emerging_down_count}{sector_meta}"
        )
        render_kpi_grid(
            st,
            items=[
                {
                    "label": status_label,
                    "value": status_head,
                    "subtext": status_sub,
                    "label_title": copy_text("market.status.label_title"),
                },
                {
                    "label": "Pressure Build",
                    "value": emerging_head,
                    "subtext": emerging_sub,
                    "label_title": "Names in the current table where early upside or downside pressure is building before full confirmation.",
                },
                {
                    "label": "Best Scalp Timing",
                    "value": best_scalp_coin,
                    "subtext": best_scalp_sub,
                    "label_title": "Best separate short-term timing candidate in the current table. This is not the main setup verdict.",
                },
                {
                    "label": "Clearest Direction",
                    "value": clearest_direction_head,
                    "subtext": clearest_direction_sub,
                    "label_title": "Advanced indicators with the clearest upside/downside alignment in the current table.",
                },
            ],
            columns=4,
        )
        st.markdown("<div style='height:0.7rem;'></div>", unsafe_allow_html=True)

        total_rows = int(shown_bundle["total"])
        attempted_count = int(len(attempted_symbols)) if "attempted_symbols" in locals() else 0
        skipped_count_live = int(len(skipped_symbols)) if "skipped_symbols" in locals() else 0
        setup_counts = dict(shown_bundle["setup_counts"])
        direction_counts = dict(shown_bundle["direction_counts"])
        ai_ensemble_counts = dict(shown_bundle["ai_ensemble_counts"])
        ai_confidence_counts = dict(shown_bundle["ai_confidence_counts"])
        audit_bundle = produced_bundle if int(produced_bundle["total"]) > 0 else shown_bundle
        audit_total_rows = int(audit_bundle["total"])
        audit_setup_counts = dict(audit_bundle["setup_counts"])
        audit_direction_counts = dict(audit_bundle["direction_counts"])
        audit_ai_ensemble_counts = dict(audit_bundle["ai_ensemble_counts"])
        probe_ratio = (int(audit_bundle["probe_count"]) / audit_total_rows) if audit_total_rows > 0 else 0.0
        watch_ratio = (int(audit_bundle["watch_count"]) / audit_total_rows) if audit_total_rows > 0 else 0.0
        direction_neutral_ratio = (
            int(audit_direction_counts.get("Neutral", 0)) / audit_total_rows
        ) if audit_total_rows > 0 else 0.0
        ai_neutral_ratio = (
            int(audit_ai_ensemble_counts.get("Neutral", 0)) / audit_total_rows
        ) if audit_total_rows > 0 else 0.0
        skip_ratio = (int(audit_bundle["skip_count"]) / audit_total_rows) if audit_total_rows > 0 else 0.0
        audit_flags: list[str] = []
        if probe_ratio >= 0.35 and int(audit_bundle["enter_count"]) <= 0:
            audit_flags.append(copy_text("market.audit.probe_heavy"))
        if watch_ratio >= 0.65:
            audit_flags.append(copy_text("market.audit.watch_heavy"))
        if ai_neutral_ratio >= 0.65:
            audit_flags.append("AI is mostly Neutral: higher-timeframe AI sees limited clean edge in the current market regime.")
        if skip_ratio >= 0.65:
            if direction_neutral_ratio >= 0.65:
                audit_flags.append(
                    copy_text("market.audit.skip_neutral")
                )
            else:
                audit_flags.append(copy_text("market.audit.skip_risk"))
        if show_diagnostics:
            with st.expander("Distribution Audit", expanded=False):
                st.caption(
                    "Use this live snapshot to judge whether the current thresholds are balanced or too strict for the market regime."
                )
                st.markdown("**Adaptive Calibration**")
                for line in _market_calibration_diagnostic_lines(
                    mode=adaptive_decision_mode,
                    target_version=adaptive_decision_target,
                    current_rows=adaptive_decision_rows,
                    total_rows=adaptive_decision_total_rows,
                    scalp_planned_rows=adaptive_current_scalp_planned_rows,
                    scalp_resolved_rows=adaptive_current_scalp_resolved_rows,
                ):
                    st.markdown(line)
                for line in _audit_scan_summary_lines(
                    displayed_rows=total_rows,
                    attempted_count=attempted_count,
                    produced_count=live_result_count_before_limit,
                    skipped_count=skipped_count_live,
                    ranked_out_count=live_ranked_out_count,
                    source_label=source_label,
                    scan_mode=scan_mode,
                    timeframe=timeframe,
                    direction_filter=direction_filter,
                ):
                    st.markdown(line)
                st.markdown("**Coverage Trace**")
                try:
                    trace_rows = fetch_scanner_trace_events_df(
                        scan_focus=str(_normalize_scan_mode(scan_mode)),
                        timeframe=str(timeframe),
                        direction_filter=str(direction_filter),
                        lookback_hours=24,
                        limit=1200,
                        db_path=signal_tracker_db_path,
                    )
                    trace_summary = build_scanner_trace_summary(trace_rows, top_n=5)
                    for line in _scanner_trace_diagnostic_lines(trace_summary):
                        st.markdown(line)
                    st.caption(
                        "Coverage Trace shows scanner coverage and recent Top N cuts. Breakout Radar and Trending Coins use it as a small capped ranking nudge."
                    )
                except Exception as e:
                    st.markdown(f"- Coverage trace unavailable: `{e.__class__.__name__}`.")
                st.markdown("**Current table**")
                st.markdown(f"Setup Confirm: {_share_line(setup_counts, ['Ready', 'Probe', 'Watch', 'Skip'])}")
                st.markdown(f"Direction: {_share_line(direction_counts, ['Upside', 'Downside', 'Neutral'])}")
                st.markdown(f"AI Ensemble: {_share_line(ai_ensemble_counts, ['Upside', 'Downside', 'Neutral'])}")
                st.markdown(
                    f"AI Confidence: {_share_line(ai_confidence_counts, ['High', 'Medium', 'Low', 'Very Low'])}"
                )
                st.markdown(
                    f"Pressure Build: {_share_line_against_total(dict(shown_bundle['emerging_counts']), ['Emerging Upside', 'Emerging Downside'], total_rows)}"
                )
                if int(produced_bundle["total"]) > 0 and int(produced_bundle["total"]) != total_rows:
                    produced_total = int(produced_bundle["total"])
                    st.markdown("**Analyzed before Top N**")
                    st.markdown(
                        f"Setup Confirm: {_share_line(dict(produced_bundle['setup_counts']), ['Ready', 'Probe', 'Watch', 'Skip'])}"
                    )
                    st.markdown(
                        f"Direction: {_share_line(dict(produced_bundle['direction_counts']), ['Upside', 'Downside', 'Neutral'])}"
                    )
                    st.markdown(
                        f"AI Ensemble: {_share_line(dict(produced_bundle['ai_ensemble_counts']), ['Upside', 'Downside', 'Neutral'])}"
                    )
                    st.markdown(
                        f"AI Confidence: {_share_line(dict(produced_bundle['ai_confidence_counts']), ['High', 'Medium', 'Low', 'Very Low'])}"
                    )
                    st.markdown(
                        f"Pressure Build: {_share_line_against_total(dict(produced_bundle['emerging_counts']), ['Emerging Upside', 'Emerging Downside'], produced_total)}"
                    )
                    if produced_total != total_rows:
                        st.caption("Skew flags below are based on all analyzed rows before the Top N cut.")
                if audit_flags:
                    for flag in audit_flags:
                        st.markdown(f"- {flag}")
                else:
                    st.markdown("- No obvious distribution skew detected in this scan.")

        # indicator visual formatting
        df_results["SuperTrend"] = df_results["SuperTrend"].apply(format_trend)
        df_results["ADX"] = df_results["ADX"].apply(format_adx).apply(_compact_adx_label)
        def _format_ichimoku_cell(v: object) -> str:
            raw = str(v or "").strip()
            if not raw or raw.upper() in {"UNAVAILABLE", "N/A", "NA", "NAN"}:
                return ""
            cleaned = (
                raw.replace("🟢 ", "")
                .replace("🔴 ", "")
                .replace("🟡 ", "")
                .replace("▲▲ ", "")
                .replace("▲ ", "")
                .replace("▼ ", "")
                .replace("→ ", "")
                .replace("– ", "")
                .strip()
            )
            up = cleaned.upper()
            if "BULLISH" in up:
                return "Bullish"
            if "BEARISH" in up:
                return "Bearish"
            if "NEUTRAL" in up:
                return "Neutral"
            return cleaned

        df_results["Ichimoku"] = (
            df_results["Ichimoku"]
            .apply(_format_ichimoku_cell)
            .apply(_normalize_indicator_label)
        )
        df_results["Stochastic RSI"] = (
            df_results["Stochastic RSI"]
            .apply(lambda v: format_stochrsi(v, timeframe=timeframe))
            .apply(_normalize_indicator_label)
        )
        df_results["VWAP"] = df_results["VWAP"].apply(_normalize_indicator_label)
        df_results["Bollinger"] = df_results["Bollinger"].apply(_normalize_indicator_label)
        df_results["Volatility"] = df_results["Volatility"].apply(_normalize_indicator_label)
        df_results["PSAR"] = df_results["PSAR"].apply(_normalize_indicator_label)
        df_results["Williams %R"] = df_results["Williams %R"].apply(_normalize_indicator_label)
        df_results["CCI"] = df_results["CCI"].apply(_normalize_indicator_label)
        df_results["Candle Pattern"] = df_results["Candle Pattern"].apply(_normalize_indicator_label)

        primary_cols = [
            "Coin",
            "Price ($)",
            "Δ (%)",
            "Setup Confirm",
            "Direction",
            "Confidence",
            "AI Ensemble",
            "AI Confidence",
            "R:R",
            "Entry Price",
            "Stop Loss",
            "Target Price",
            "Scalp Opportunity",
            "Market Cap ($)",
        ]
        advanced_extra_cols = [
            "ADX",
            "SuperTrend",
            "Ichimoku",
            "VWAP",
            "PSAR",
            "Stochastic RSI",
            "Williams %R",
            "CCI",
            "Candle Pattern",
            "Bollinger",
            "Volatility",
            "Spike Alert",
        ]
        all_cols = primary_cols + [c for c in advanced_extra_cols if c not in primary_cols]
        display_cols = all_cols if show_advanced else primary_cols
        hidden_meta_cols = _market_hidden_meta_cols(df_results.columns, display_cols)
        df_display = df_results[display_cols + hidden_meta_cols].copy()

        _render_pro_table(df_display, display_cols)

        def _csv_clean_text(v: object) -> str:
            if v is None:
                return ""
            try:
                if pd.isna(v):
                    return ""
            except Exception:
                pass
            s = str(v).strip()
            if not s or s.upper() in {"N/A", "NA", "NAN", "UNAVAILABLE", "-"}:
                return ""
            for token in (
                "✅", "🟡", "⛔", "⌛", "👀", "🟢", "🔴", "⚪", "🔥",
                "⚠️", "⚠", "🚀", "📈", "📉", "•", "*", "★",
            ):
                s = s.replace(token, "")
            s = re.sub(r"[▲▼→–]+", "", s)
            s = re.sub(r"\s{2,}", " ", s).strip()
            return s

        def _csv_clean_price(v: object) -> str:
            s = _csv_clean_text(v)
            if not s:
                return ""
            return s.replace("$", "").replace(",", "").strip()

        def _csv_clean_delta(v: object) -> str:
            s = str(v or "").strip()
            if not s:
                return ""
            sign = "+"
            if s.startswith("▼"):
                sign = "-"
            elif s.startswith("→"):
                sign = ""
            cleaned = re.sub(r"^[\s▲▼→–-]+", "", s).strip()
            if not cleaned:
                return ""
            return f"{sign}{cleaned}" if sign else cleaned

        csv_df = df_results.copy()
        for col in csv_df.columns:
            if col in {"Price ($)", "Entry Price", "Stop Loss", "Target Price"}:
                csv_df[col] = csv_df[col].apply(_csv_clean_price)
            elif col == "Δ (%)":
                csv_df[col] = csv_df[col].apply(_csv_clean_delta)
            elif csv_df[col].dtype == object or pd.api.types.is_string_dtype(csv_df[col]):
                csv_df[col] = csv_df[col].apply(_csv_clean_text)

        # Never export internal/meta columns.
        csv_df = csv_df[[c for c in csv_df.columns if not str(c).startswith("__")]]

        csv_market = csv_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="Download Scan Results (CSV)",
            data=csv_market,
            file_name="scan_results.csv",
            mime="text/csv"
        )
    else:
        source_is_degraded = "DEGRADED" in str(source_label).upper()
        source_color = WARNING if (source_is_degraded or str(source_label).startswith("CACHED")) else POSITIVE
        source_chip = (
            "Partial Live"
            if source_is_degraded
            else ("Live Data" if str(source_label).startswith("LIVE") else "Cached Snapshot")
        )
        mode_color = ACCENT if data_mode in {"FULL MARKET MODE", "CUSTOM WATCHLIST MODE"} else WARNING
        _render_data_health_band(
            source_chip=source_chip,
            source_color=source_color,
            source_display_label=str(source_label).replace("DEGRADED", "PARTIAL"),
            mode_label=_market_data_mode_display(str(data_mode)),
            mode_color=mode_color,
            items=_data_health_display_items(data_health_items, source_label_text=str(source_label), data_mode_text=str(data_mode)),
        )
        if "DEGRADED" in str(source_label).upper():
            st.info(
                "No coins were shown because the latest live market read was incomplete and did not produce a usable table."
            )
        else:
            st.info("No coins matched this scan. Try All Directions, a higher Top N, or a different timeframe.")

    _run_auto_timeframe_learning_sweep(
        session_state=st.session_state,
        current_timeframe=str(timeframe),
        exclude_stables=bool(exclude_stables),
        get_top_volume_usdt_symbols=get_top_volume_usdt_symbols,
        fetch_ohlcv=fetch_ohlcv,
        analyse=analyse,
        ml_ensemble_predict=ml_ensemble_predict,
        signal_plain=signal_plain,
        direction_key=direction_key,
        fetch_signal_events_df=fetch_signal_events_df,
        log_signal_events=log_signal_events,
        resolve_open_signal_events_for_frame=resolve_open_signal_events_for_frame,
        backfill_signal_forward_windows_via_fetch=backfill_signal_forward_windows_via_fetch,
        fetch_signal_forward_windows_df=fetch_signal_forward_windows_df,
        db_path=signal_tracker_db_path,
        debug=_debug,
    )
