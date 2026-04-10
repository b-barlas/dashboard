"""Empirical feedback learning from resolved signal history."""

from __future__ import annotations

from dataclasses import dataclass
import math

import pandas as pd
from core.confidence import normalize_direction
from core.market_decision import normalize_action_class
from core.session_utils import session_bucket_for_timestamp
from core.trading_copy import playbook_display, trade_gate_display, trade_gate_key


_LENS_WEIGHTS = {
    "Setup Confirm": 1.00,
    "Lead": 0.80,
    "AI Alignment": 0.70,
    "Market Lead": 0.80,
    "Market Regime": 0.90,
    "Playbook": 0.75,
    "Trade Gate": 0.90,
    "Execution Stance": 0.60,
    "Primary Alert": 0.45,
    "Primary Alert x Playbook": 0.50,
    "Primary Alert x Session": 0.45,
    "Playbook x Session": 0.65,
    "Playbook x Catalyst Window": 0.70,
    "Sector Rotation": 0.60,
    "Catalyst State": 0.55,
    "Catalyst Window": 0.70,
    "Catalyst Scope": 0.40,
    "Catalyst Targeting": 0.35,
    "Flow Proxy": 0.55,
    "Session": 0.50,
    "Timeframe": 0.45,
}

_ARCHIVE_GUARDRAIL_LENSES = {
    "Trade Gate": 1.00,
    "Playbook": 0.95,
    "Primary Alert": 0.75,
    "Primary Alert x Playbook": 0.80,
    "Primary Alert x Session": 0.75,
    "Playbook x Session": 0.75,
    "Playbook x Catalyst Window": 0.85,
    "Catalyst Window": 0.90,
    "Session": 0.85,
    "Market Regime": 0.80,
    "Catalyst Scope": 0.50,
}

_AI_CONFIDENCE_CALIBRATION_LENSES = {
    "Setup Class": 1.00,
    "AI Alignment": 0.85,
    "Timeframe": 0.55,
    "Scan Focus": 0.40,
    "Setup Class x AI Alignment": 0.90,
    "Timeframe x Setup Class": 0.75,
    "Scan Focus x Setup Class": 0.60,
}

_AI_CONFIDENCE_MIN_RESOLVED = 30
_AI_CONFIDENCE_MIN_BUCKET = 12
_AI_CONFIDENCE_MAX_DELTA = 5.0

_CONFIDENCE_CALIBRATION_LENSES = {
    "Direction": 1.00,
    "AI Alignment": 0.75,
    "Timeframe": 0.70,
    "Scan Focus": 0.40,
    "Direction x Timeframe": 0.95,
    "Timeframe x AI Alignment": 0.75,
    "Direction x AI Alignment": 0.65,
}

_CONFIDENCE_CALIBRATION_MIN_RESOLVED = 30
_CONFIDENCE_CALIBRATION_MIN_BUCKET = 12
_CONFIDENCE_CALIBRATION_MAX_DELTA = 5.0

_SETUP_CALIBRATION_LENSES = {
    "Setup Class": 1.00,
    "AI Alignment": 0.80,
    "Timeframe": 0.60,
    "Direction": 0.50,
    "Scan Focus": 0.35,
    "Setup Class x AI Alignment": 0.90,
    "Timeframe x Setup Class": 0.80,
    "Setup Class x Direction": 0.70,
    "Scan Focus x Setup Class": 0.55,
}

_SETUP_CALIBRATION_MIN_RESOLVED = 40
_SETUP_CALIBRATION_MIN_BUCKET = 14
_SETUP_CALIBRATION_MAX_DELTA = 4.5

_ACTIONABLE_RANKING_LENSES = {
    "Setup Class": 1.00,
    "AI Alignment": 0.70,
    "Timeframe": 0.70,
    "Direction": 0.55,
    "Scan Focus": 0.35,
    "Timeframe x Setup Class": 0.90,
    "Setup Class x Direction": 0.80,
    "Setup Class x AI Alignment": 0.75,
    "Scan Focus x Setup Class": 0.55,
}

_ACTIONABLE_RANKING_MIN_RESOLVED = 35
_ACTIONABLE_RANKING_MIN_BUCKET = 12
_ACTIONABLE_RANKING_MAX_DELTA = 8.0

_RISK_SIZING_CALIBRATION_LENSES = {
    "Setup Class": 1.00,
    "AI Alignment": 0.70,
    "Timeframe": 0.70,
    "Direction": 0.55,
    "Scan Focus": 0.35,
    "Timeframe x Setup Class": 0.90,
    "Setup Class x Direction": 0.80,
    "Setup Class x AI Alignment": 0.75,
    "Scan Focus x Setup Class": 0.55,
}

_RISK_SIZING_MIN_RESOLVED = 40
_RISK_SIZING_MIN_BUCKET = 14
_RISK_SIZING_MAX_DELTA = 0.25

_TRADE_GATE_CALIBRATION_LENSES = {
    "Trade Gate": 1.00,
    "Playbook": 0.90,
    "Market Regime": 0.75,
    "Session": 0.65,
    "Catalyst Window": 0.75,
    "Playbook x Session": 0.85,
    "Playbook x Catalyst Window": 0.90,
}

_TRADE_GATE_MIN_RESOLVED = 40
_TRADE_GATE_MIN_BUCKET = 14
_TRADE_GATE_MAX_DELTA = 1.0

_SCALP_CALIBRATION_LENSES = {
    "Setup Class": 1.00,
    "Timeframe": 0.90,
    "Direction": 0.65,
    "AI Alignment": 0.55,
    "Scan Focus": 0.35,
    "Timeframe x Setup Class": 0.95,
    "Setup Class x Direction": 0.80,
    "Setup Class x AI Alignment": 0.65,
}

_SCALP_SUPPORTED_TIMEFRAMES = {"1m", "3m", "5m", "15m", "1h"}
_SCALP_CALIBRATION_MIN_RESOLVED = 24
_SCALP_CALIBRATION_MIN_BUCKET = 8
_SCALP_CALIBRATION_MAX_DELTA = 1.0


def _text_value(value: object) -> str:
    if value is None or pd.isna(value):
        return ""
    return str(value).strip()


@dataclass(frozen=True)
class AdaptiveEdgeSnapshot:
    score: float
    label: str
    note: str
    matched_factors: int
    resolved_sample: int
    actual_trade_sample: int
    execution_fit_label: str
    execution_fit_note: str
    session_fit_score: float
    session_fit_label: str
    session_fit_note: str
    archive_guardrail_penalty: float
    archive_guardrail_label: str
    archive_guardrail_note: str


@dataclass(frozen=True)
class SessionFitSnapshot:
    score: float
    label: str
    note: str
    resolved_sample: int
    actual_trade_sample: int


@dataclass(frozen=True)
class ArchiveGuardrailSnapshot:
    penalty: float
    label: str
    note: str
    matched_factors: int


@dataclass(frozen=True)
class AICalibrationSnapshot:
    delta: float
    note: str
    matched_factors: int
    resolved_sample: int


@dataclass(frozen=True)
class ConfidenceCalibrationSnapshot:
    delta: float
    note: str
    matched_factors: int
    resolved_sample: int


@dataclass(frozen=True)
class SetupCalibrationSnapshot:
    delta: float
    note: str
    matched_factors: int
    resolved_sample: int


@dataclass(frozen=True)
class ActionableRankingSnapshot:
    delta: float
    note: str
    matched_factors: int
    resolved_sample: int


@dataclass(frozen=True)
class RiskSizingCalibrationSnapshot:
    delta: float
    note: str
    matched_factors: int
    resolved_sample: int


@dataclass(frozen=True)
class TradeGateCalibrationSnapshot:
    delta: float
    note: str
    matched_factors: int
    resolved_sample: int


@dataclass(frozen=True)
class ScalpCalibrationSnapshot:
    delta: float
    note: str
    matched_factors: int
    resolved_sample: int


def _weighted_mean(values: pd.Series, weights: pd.Series) -> float:
    numeric_values = pd.to_numeric(values, errors="coerce")
    numeric_weights = pd.to_numeric(weights, errors="coerce").fillna(0.0)
    mask = numeric_values.notna() & numeric_weights.gt(0.0)
    if not bool(mask.any()):
        return 0.0
    weighted_sum = float((numeric_values[mask] * numeric_weights[mask]).sum())
    total_weight = float(numeric_weights[mask].sum())
    if total_weight <= 0.0:
        return 0.0
    return weighted_sum / total_weight


def _compose_combo_value(left: object, right: object) -> str:
    left_value = str(left or "").strip() or "Unknown"
    right_value = str(right or "").strip() or "Unknown"
    return f"{left_value} | {right_value}"


def _direction_value(value: object) -> str:
    direction = normalize_direction(str(value or ""))
    if direction == "UPSIDE":
        return "Upside"
    if direction == "DOWNSIDE":
        return "Downside"
    return "Neutral"


def _setup_class_value(value: object) -> str:
    action_class = normalize_action_class(str(value or ""))
    return action_class or "UNKNOWN"


def _execution_stance_value(trade_gate: object, adaptive_edge: object, archive_guardrail: object) -> str:
    gate = trade_gate_key(trade_gate)
    adaptive = str(adaptive_edge or "").strip()
    guardrail = str(archive_guardrail or "").strip()

    if gate == "NO_TRADE" or guardrail == "Archive Guardrail":
        return trade_gate_display("NO_TRADE")
    if gate == "DEFENSIVE_ONLY" or adaptive == "Historically Weak":
        return trade_gate_display("DEFENSIVE_ONLY")
    if gate == "TRADEABLE" and adaptive == "Historically Favored" and guardrail == "Archive Clear":
        return trade_gate_display("TRADEABLE")
    if gate in {"TRADEABLE", "SELECTIVE_ONLY"}:
        return trade_gate_display("SELECTIVE_ONLY")
    return trade_gate_display(gate) if gate != "UNKNOWN" else "Unknown"


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


def _primary_alert_value(row_or_signal: dict[str, object]) -> str:
    primary = str(row_or_signal.get("market_primary_alert") or row_or_signal.get("Primary Alert") or "").strip()
    if primary:
        return _alert_key_display(primary)

    catalyst_state = str(row_or_signal.get("market_catalyst_state") or row_or_signal.get("Catalyst State") or "").strip().upper()
    catalyst_window = str(row_or_signal.get("market_catalyst_window") or row_or_signal.get("Catalyst Window") or "").strip()
    trade_gate = _trade_gate_value(row_or_signal)
    market_lead = str(row_or_signal.get("market_lead_label") or row_or_signal.get("Market Lead") or "").strip()
    flow_proxy = str(row_or_signal.get("market_flow_state") or row_or_signal.get("Flow Proxy") or "").strip()
    sector_rotation = str(row_or_signal.get("market_sector_rotation") or row_or_signal.get("Sector Rotation") or "").strip()
    session = str(row_or_signal.get("Session") or row_or_signal.get("session_bucket") or "").strip()

    if catalyst_window.startswith("Blocking") or "BLOCK" in catalyst_state:
        return "Catalyst Block"
    if trade_gate_key(trade_gate) in {"NO_TRADE", "DEFENSIVE_ONLY"}:
        return "Trade Gate"
    if market_lead in {"Upside", "Downside"}:
        return "Market Lead"
    if flow_proxy in {"Shorts Crowded", "Longs Crowded"}:
        return "Flow Proxy"
    if sector_rotation not in {"", "Unknown", "Mixed Sector Rotation", "None"}:
        return "Sector Rotation"
    if session.startswith("European") or session.startswith("US"):
        return "Session Fit"
    if catalyst_state == "Catalyst Caution":
        return "Catalyst Caution"
    return "No Alert Footprint"


def _playbook_value(row_or_signal: dict[str, object]) -> str:
    key = _text_value(row_or_signal.get("market_playbook_key")) or _text_value(row_or_signal.get("Playbook Key"))
    if key and key.upper() not in {"UNKNOWN", "NAN"}:
        return playbook_display(key)
    return _text_value(row_or_signal.get("market_playbook")) or _text_value(row_or_signal.get("Playbook"))


def _trade_gate_value(row_or_signal: dict[str, object]) -> str:
    key = _text_value(row_or_signal.get("market_trade_gate_key")) or _text_value(row_or_signal.get("Trade Gate Key"))
    if key and key.upper() not in {"UNKNOWN", "NAN"}:
        return trade_gate_display(key, audience="trader")
    return _text_value(row_or_signal.get("market_trade_gate")) or _text_value(row_or_signal.get("Trade Gate"))


def _prepare_resolved_events(df_events: pd.DataFrame) -> pd.DataFrame:
    if df_events is None or df_events.empty:
        return pd.DataFrame()
    d = df_events.copy()
    d["status"] = d.get("status", pd.Series(dtype=object)).astype(str).str.upper()
    d = d[d["status"] == "RESOLVED"].copy()
    if d.empty:
        return d
    event_ts = pd.to_datetime(d.get("event_time", pd.Series(index=d.index, dtype=object)), utc=True, errors="coerce")
    latest_event_ts = event_ts.max()
    if pd.notna(latest_event_ts):
        age_hours = ((latest_event_ts - event_ts).dt.total_seconds() / 3600.0).clip(lower=0.0)
        d["recency_weight"] = age_hours.map(lambda hours: max(0.25, math.exp(-float(hours) / (24.0 * 21.0))))
    else:
        d["recency_weight"] = 1.0
    d["directional_return_pct"] = pd.to_numeric(d.get("directional_return_pct"), errors="coerce")
    d["Lead"] = d.get("lead_active", pd.Series(dtype=int)).fillna(0).astype(int).map({1: "LEAD", 0: "No LEAD"})
    d["AI Alignment"] = d.get("ai_aligned", pd.Series(dtype=int)).fillna(0).astype(int).map({1: "Aligned", 0: "Not aligned"})
    d["Market Lead"] = d.get("market_lead_label", pd.Series(dtype=object)).replace("", "No Clear Lead").fillna("No Clear Lead")
    d["Market Regime"] = d.get("market_regime", pd.Series(dtype=object)).replace("", "Unknown").fillna("Unknown")
    d["Playbook Key"] = d.get("market_playbook_key", pd.Series(dtype=object)).replace("", "Unknown").fillna("Unknown")
    d["Trade Gate Key"] = d.get("market_trade_gate_key", pd.Series(dtype=object)).replace("", "Unknown").fillna("Unknown")
    d["Playbook"] = d.apply(lambda row: _playbook_value(row.to_dict()) or "Unknown", axis=1)
    d["Trade Gate"] = d.apply(lambda row: _trade_gate_value(row.to_dict()) or "Unknown", axis=1)
    d["Sector Rotation"] = d.get("market_sector_rotation", pd.Series(dtype=object)).replace("", "Unknown").fillna("Unknown")
    d["Catalyst State"] = d.get("market_catalyst_state", pd.Series(dtype=object)).replace("", "Unknown").fillna("Unknown")
    d["Catalyst Window"] = d.get("market_catalyst_window", pd.Series(dtype=object)).replace("", "Unknown").fillna("Unknown")
    d["Catalyst Scope"] = d.get("market_catalyst_scope", pd.Series(dtype=object)).replace("", "Unknown").fillna("Unknown")
    d["Catalyst Targeting"] = (
        d.get("market_catalyst_targeted", pd.Series(dtype=int))
        .fillna(0)
        .astype(int)
        .map({1: "Targeted", 0: "Market-Wide"})
    )
    d["Flow Proxy"] = d.get("market_flow_state", pd.Series(dtype=object)).replace("", "Unknown").fillna("Unknown")
    session_series = d.get("session_bucket", pd.Series(index=d.index, dtype=object))
    if not isinstance(session_series, pd.Series):
        session_series = pd.Series(session_series, index=d.index, dtype=object)
    session_series = session_series.reindex(d.index).fillna("").astype(str).str.strip()
    event_time_series = d.get("event_time", pd.Series(index=d.index, dtype=object))
    if not isinstance(event_time_series, pd.Series):
        event_time_series = pd.Series(event_time_series, index=d.index, dtype=object)
    derived_session = pd.to_datetime(event_time_series.reindex(d.index), utc=True, errors="coerce").map(
        lambda ts: session_bucket_for_timestamp(ts) if pd.notna(ts) else "Unknown"
    )
    d["Session"] = session_series.where(session_series.ne(""), derived_session).replace("", "Unknown").fillna("Unknown")
    d["Timeframe"] = d.get("timeframe", pd.Series(dtype=object)).replace("", "Unknown").fillna("Unknown")
    d["Setup Confirm"] = d.get("setup_confirm", pd.Series(dtype=object)).replace("", "Unknown").fillna("Unknown")
    d["Setup Class"] = d.get("setup_confirm", pd.Series(dtype=object)).map(_setup_class_value).replace("", "UNKNOWN").fillna("UNKNOWN")
    d["Scan Focus"] = d.get("scan_focus", pd.Series(dtype=object)).replace("", "Unknown").fillna("Unknown")
    d["Direction"] = d.get("direction", pd.Series(dtype=object)).map(_direction_value).replace("", "Neutral").fillna("Neutral")
    d["Timeframe x Setup Class"] = d.apply(
        lambda row: _compose_combo_value(row.get("Timeframe"), row.get("Setup Class")),
        axis=1,
    )
    d["Setup Class x Direction"] = d.apply(
        lambda row: _compose_combo_value(row.get("Setup Class"), row.get("Direction")),
        axis=1,
    )
    d["Setup Class x AI Alignment"] = d.apply(
        lambda row: _compose_combo_value(row.get("Setup Class"), row.get("AI Alignment")),
        axis=1,
    )
    d["Scan Focus x Setup Class"] = d.apply(
        lambda row: _compose_combo_value(row.get("Scan Focus"), row.get("Setup Class")),
        axis=1,
    )
    d["Direction x Timeframe"] = d.apply(
        lambda row: _compose_combo_value(row.get("Direction"), row.get("Timeframe")),
        axis=1,
    )
    d["Timeframe x AI Alignment"] = d.apply(
        lambda row: _compose_combo_value(row.get("Timeframe"), row.get("AI Alignment")),
        axis=1,
    )
    d["Direction x AI Alignment"] = d.apply(
        lambda row: _compose_combo_value(row.get("Direction"), row.get("AI Alignment")),
        axis=1,
    )
    d["Execution Stance"] = d.apply(
        lambda row: _execution_stance_value(
            row.get("market_trade_gate"),
            row.get("adaptive_edge_label"),
            row.get("archive_guardrail_label"),
        ),
        axis=1,
    )
    d["Primary Alert"] = d.apply(lambda row: _primary_alert_value(row.to_dict()), axis=1)
    d["Primary Alert x Playbook"] = d.apply(
        lambda row: _compose_combo_value(row.get("Primary Alert"), row.get("Playbook")),
        axis=1,
    )
    d["Primary Alert x Session"] = d.apply(
        lambda row: _compose_combo_value(row.get("Primary Alert"), row.get("Session")),
        axis=1,
    )
    d["Playbook x Session"] = d.apply(
        lambda row: _compose_combo_value(row.get("Playbook"), row.get("Session")),
        axis=1,
    )
    d["Playbook x Catalyst Window"] = d.apply(
        lambda row: _compose_combo_value(row.get("Playbook"), row.get("Catalyst Window")),
        axis=1,
    )
    d["is_follow"] = (d["directional_return_pct"] > 0).astype(int)
    d["actual_trade_status"] = d.get("actual_trade_status", pd.Series(dtype=object)).fillna("").astype(str).str.upper()
    d["actual_pnl_pct"] = pd.to_numeric(d.get("actual_pnl_pct"), errors="coerce")
    d["is_trade_closed"] = (d["actual_trade_status"] == "CLOSED").astype(int)
    d["is_trade_win"] = ((d["is_trade_closed"] == 1) & (d["actual_pnl_pct"] > 0)).astype(int)
    has_plan_series = d.get("has_plan", pd.Series(index=d.index, dtype=float))
    if not isinstance(has_plan_series, pd.Series):
        has_plan_series = pd.Series(has_plan_series, index=d.index, dtype=float)
    d["has_plan"] = pd.to_numeric(has_plan_series.reindex(d.index), errors="coerce").fillna(0).astype(int)
    rr_ratio_series = d.get("rr_ratio", pd.Series(index=d.index, dtype=float))
    if not isinstance(rr_ratio_series, pd.Series):
        rr_ratio_series = pd.Series(rr_ratio_series, index=d.index, dtype=float)
    d["rr_ratio"] = pd.to_numeric(rr_ratio_series.reindex(d.index), errors="coerce")
    plan_outcome_series = d.get("plan_outcome", pd.Series(index=d.index, dtype=object))
    if not isinstance(plan_outcome_series, pd.Series):
        plan_outcome_series = pd.Series(plan_outcome_series, index=d.index, dtype=object)
    d["plan_outcome"] = plan_outcome_series.reindex(d.index).fillna("").astype(str).str.upper()
    d["is_tp"] = (d["plan_outcome"] == "TP").astype(int)
    d["is_sl"] = (d["plan_outcome"] == "SL").astype(int)
    return d


def build_scalp_calibration_model(df_events: pd.DataFrame, *, min_samples: int = 6) -> dict[str, object]:
    resolved = _prepare_resolved_events(df_events)
    if resolved.empty:
        return {
            "resolved_count": 0,
            "overall_tp_pct": 0.0,
            "overall_sl_pct": 0.0,
            "overall_follow_pct": 0.0,
            "overall_avg_return": 0.0,
            "lenses": {},
        }

    planned = resolved[
        (pd.to_numeric(resolved.get("has_plan"), errors="coerce").fillna(0).astype(int) == 1)
        & resolved.get("Timeframe", pd.Series(index=resolved.index, dtype=object)).isin(_SCALP_SUPPORTED_TIMEFRAMES)
    ].copy()
    if planned.empty:
        return {
            "resolved_count": 0,
            "overall_tp_pct": 0.0,
            "overall_sl_pct": 0.0,
            "overall_follow_pct": 0.0,
            "overall_avg_return": 0.0,
            "lenses": {},
        }

    recency_weights = pd.to_numeric(planned.get("recency_weight"), errors="coerce").fillna(1.0)
    overall_tp_pct = float(_weighted_mean(planned["is_tp"], recency_weights) * 100.0)
    overall_sl_pct = float(_weighted_mean(planned["is_sl"], recency_weights) * 100.0)
    overall_follow_pct = float(_weighted_mean(planned["is_follow"], recency_weights) * 100.0)
    overall_avg_return = float(_weighted_mean(planned["directional_return_pct"], recency_weights))
    lenses: dict[str, dict[str, dict[str, float]]] = {}
    for lens in _SCALP_CALIBRATION_LENSES:
        if lens not in planned.columns:
            continue
        lens_map: dict[str, dict[str, float]] = {}
        for value, bucket_df in planned.groupby(lens, dropna=False):
            sample_n = int(len(bucket_df))
            if sample_n < int(min_samples):
                continue
            bucket_weights = pd.to_numeric(bucket_df.get("recency_weight"), errors="coerce").fillna(1.0)
            tp_pct = float(_weighted_mean(bucket_df["is_tp"], bucket_weights) * 100.0)
            sl_pct = float(_weighted_mean(bucket_df["is_sl"], bucket_weights) * 100.0)
            follow_pct = float(_weighted_mean(bucket_df["is_follow"], bucket_weights) * 100.0)
            avg_dir = float(_weighted_mean(bucket_df["directional_return_pct"], bucket_weights))
            edge_score = (
                ((tp_pct - overall_tp_pct) * 0.60)
                - ((sl_pct - overall_sl_pct) * 0.55)
                + ((follow_pct - overall_follow_pct) * 0.30)
                + ((avg_dir - overall_avg_return) * 4.5)
            )
            lens_map[str(value or "Unknown")] = {
                "resolved": float(sample_n),
                "tp_pct": tp_pct,
                "sl_pct": sl_pct,
                "follow_pct": follow_pct,
                "avg_dir_return": avg_dir,
                "edge_score": edge_score,
            }
        if lens_map:
            lenses[lens] = lens_map
    return {
        "resolved_count": int(len(planned)),
        "overall_tp_pct": overall_tp_pct,
        "overall_sl_pct": overall_sl_pct,
        "overall_follow_pct": overall_follow_pct,
        "overall_avg_return": overall_avg_return,
        "lenses": lenses,
    }


def build_scalp_calibration_snapshot(
    model: dict[str, object],
    *,
    signal: dict[str, object],
) -> ScalpCalibrationSnapshot:
    resolved_count = int(model.get("resolved_count") or 0)
    if resolved_count < _SCALP_CALIBRATION_MIN_RESOLVED:
        return ScalpCalibrationSnapshot(delta=0.0, note="", matched_factors=0, resolved_sample=resolved_count)

    signal_values = dict(signal or {})
    signal_values["Setup Class"] = _setup_class_value(signal_values.get("Setup Confirm"))
    signal_values["AI Alignment"] = (
        "Aligned" if str(signal_values.get("AI Alignment") or "").strip().lower().startswith("aligned") else "Not aligned"
    )
    signal_values["Timeframe"] = str(signal_values.get("Timeframe") or "").strip() or "Unknown"
    signal_values["Direction"] = _direction_value(signal_values.get("Direction"))
    signal_values["Scan Focus"] = str(signal_values.get("Scan Focus") or "").strip() or "Unknown"
    signal_values["Timeframe x Setup Class"] = _compose_combo_value(
        signal_values.get("Timeframe"),
        signal_values.get("Setup Class"),
    )
    signal_values["Setup Class x Direction"] = _compose_combo_value(
        signal_values.get("Setup Class"),
        signal_values.get("Direction"),
    )
    signal_values["Setup Class x AI Alignment"] = _compose_combo_value(
        signal_values.get("Setup Class"),
        signal_values.get("AI Alignment"),
    )

    lenses = dict(model.get("lenses") or {})
    contributions: list[tuple[float, str, float]] = []
    for lens, weight in _SCALP_CALIBRATION_LENSES.items():
        value = str(signal_values.get(lens) or "").strip() or "Unknown"
        lens_map = lenses.get(lens)
        if not isinstance(lens_map, dict):
            continue
        bucket = lens_map.get(value)
        if not isinstance(bucket, dict):
            continue
        bucket_resolved = float(bucket.get("resolved") or 0.0)
        if bucket_resolved < float(_SCALP_CALIBRATION_MIN_BUCKET):
            continue
        edge_score = float(bucket.get("edge_score") or 0.0)
        sample_factor = min(1.0, bucket_resolved / 28.0)
        weighted = edge_score * float(weight) * sample_factor * 0.02
        if abs(weighted) < 0.05:
            continue
        contributions.append((weighted, f"{lens}: {value}", bucket_resolved))

    if not contributions:
        return ScalpCalibrationSnapshot(delta=0.0, note="", matched_factors=0, resolved_sample=resolved_count)

    contributions.sort(key=lambda item: abs(item[0]), reverse=True)
    delta = max(
        -_SCALP_CALIBRATION_MAX_DELTA,
        min(_SCALP_CALIBRATION_MAX_DELTA, sum(value for value, _, _ in contributions)),
    )
    if abs(delta) < 0.12:
        return ScalpCalibrationSnapshot(delta=0.0, note="", matched_factors=0, resolved_sample=resolved_count)

    strongest = [name for value, name, _ in contributions if value > 0][:2]
    weakest = [name for value, name, _ in contributions if value < 0][:2]
    if delta > 0.0 and strongest:
        note = "Archive scalp calibration is modestly supportive here. Strongest cohorts: " + ", ".join(strongest) + "."
    elif delta < 0.0 and weakest:
        note = "Archive scalp calibration is modestly cautious here. Weakest cohorts: " + ", ".join(weakest) + "."
    else:
        note = ""

    return ScalpCalibrationSnapshot(
        delta=float(delta),
        note=note,
        matched_factors=len(contributions),
        resolved_sample=resolved_count,
    )


def build_ai_confidence_calibration_model(df_events: pd.DataFrame, *, min_samples: int = 8) -> dict[str, object]:
    resolved = _prepare_resolved_events(df_events)
    if resolved.empty:
        return {
            "resolved_count": 0,
            "overall_follow_pct": 0.0,
            "overall_avg_return": 0.0,
            "lenses": {},
        }

    recency_weights = pd.to_numeric(resolved.get("recency_weight"), errors="coerce").fillna(1.0)
    overall_follow_pct = float(_weighted_mean(resolved["is_follow"], recency_weights) * 100.0)
    overall_avg_return = float(_weighted_mean(resolved["directional_return_pct"], recency_weights))
    lenses: dict[str, dict[str, dict[str, float]]] = {}
    for lens in _AI_CONFIDENCE_CALIBRATION_LENSES:
        if lens not in resolved.columns:
            continue
        lens_map: dict[str, dict[str, float]] = {}
        for value, bucket_df in resolved.groupby(lens, dropna=False):
            sample_n = int(len(bucket_df))
            if sample_n < int(min_samples):
                continue
            bucket_weights = pd.to_numeric(bucket_df.get("recency_weight"), errors="coerce").fillna(1.0)
            follow_pct = float(_weighted_mean(bucket_df["is_follow"], bucket_weights) * 100.0)
            avg_dir = float(_weighted_mean(bucket_df["directional_return_pct"], bucket_weights))
            edge_score = ((follow_pct - overall_follow_pct) * 0.60) + ((avg_dir - overall_avg_return) * 5.0)
            lens_map[str(value or "Unknown")] = {
                "resolved": float(sample_n),
                "follow_pct": follow_pct,
                "avg_dir_return": avg_dir,
                "edge_score": edge_score,
            }
        if lens_map:
            lenses[lens] = lens_map
    return {
        "resolved_count": int(len(resolved)),
        "overall_follow_pct": overall_follow_pct,
        "overall_avg_return": overall_avg_return,
        "lenses": lenses,
    }


def build_ai_confidence_calibration_snapshot(model: dict[str, object], *, signal: dict[str, object]) -> AICalibrationSnapshot:
    resolved_count = int(model.get("resolved_count") or 0)
    if resolved_count < _AI_CONFIDENCE_MIN_RESOLVED:
        return AICalibrationSnapshot(delta=0.0, note="", matched_factors=0, resolved_sample=resolved_count)

    signal_values = dict(signal or {})
    signal_values["Setup Class"] = _setup_class_value(signal_values.get("Setup Confirm"))
    signal_values["AI Alignment"] = (
        "Aligned" if str(signal_values.get("AI Alignment") or "").strip().lower().startswith("aligned") else "Not aligned"
    )
    signal_values["Timeframe"] = str(signal_values.get("Timeframe") or "").strip() or "Unknown"
    signal_values["Scan Focus"] = str(signal_values.get("Scan Focus") or "").strip() or "Unknown"
    signal_values["Direction"] = _direction_value(signal_values.get("Direction"))
    signal_values["Timeframe x Setup Class"] = _compose_combo_value(
        signal_values.get("Timeframe"),
        signal_values.get("Setup Class"),
    )
    signal_values["Setup Class x AI Alignment"] = _compose_combo_value(
        signal_values.get("Setup Class"),
        signal_values.get("AI Alignment"),
    )
    signal_values["Scan Focus x Setup Class"] = _compose_combo_value(
        signal_values.get("Scan Focus"),
        signal_values.get("Setup Class"),
    )

    lenses = dict(model.get("lenses") or {})
    contributions: list[tuple[float, str, float]] = []
    for lens, weight in _AI_CONFIDENCE_CALIBRATION_LENSES.items():
        value = str(signal_values.get(lens) or "").strip() or "Unknown"
        lens_map = lenses.get(lens)
        if not isinstance(lens_map, dict):
            continue
        bucket = lens_map.get(value)
        if not isinstance(bucket, dict):
            continue
        bucket_resolved = float(bucket.get("resolved") or 0.0)
        if bucket_resolved < float(_AI_CONFIDENCE_MIN_BUCKET):
            continue
        edge_score = float(bucket.get("edge_score") or 0.0)
        sample_factor = min(1.0, bucket_resolved / 40.0)
        weighted = edge_score * float(weight) * sample_factor * 0.08
        if abs(weighted) < 0.18:
            continue
        contributions.append((weighted, f"{lens}: {value}", bucket_resolved))

    if not contributions:
        return AICalibrationSnapshot(delta=0.0, note="", matched_factors=0, resolved_sample=resolved_count)

    contributions.sort(key=lambda item: abs(item[0]), reverse=True)
    delta = max(
        -_AI_CONFIDENCE_MAX_DELTA,
        min(_AI_CONFIDENCE_MAX_DELTA, sum(value for value, _, _ in contributions)),
    )
    if abs(delta) < 0.35:
        return AICalibrationSnapshot(delta=0.0, note="", matched_factors=0, resolved_sample=resolved_count)

    strongest = [name for value, name, _ in contributions if value > 0][:2]
    weakest = [name for value, name, _ in contributions if value < 0][:2]
    if delta > 0.0 and strongest:
        note = "Archive calibration is modestly supportive here. Strongest cohorts: " + ", ".join(strongest) + "."
    elif delta < 0.0 and weakest:
        note = "Archive calibration is modestly cautious here. Weakest cohorts: " + ", ".join(weakest) + "."
    else:
        note = ""

    return AICalibrationSnapshot(
        delta=float(delta),
        note=note,
        matched_factors=len(contributions),
        resolved_sample=resolved_count,
    )


def build_confidence_calibration_model(df_events: pd.DataFrame, *, min_samples: int = 8) -> dict[str, object]:
    resolved = _prepare_resolved_events(df_events)
    if resolved.empty:
        return {
            "resolved_count": 0,
            "overall_follow_pct": 0.0,
            "overall_avg_return": 0.0,
            "lenses": {},
        }

    recency_weights = pd.to_numeric(resolved.get("recency_weight"), errors="coerce").fillna(1.0)
    overall_follow_pct = float(_weighted_mean(resolved["is_follow"], recency_weights) * 100.0)
    overall_avg_return = float(_weighted_mean(resolved["directional_return_pct"], recency_weights))
    lenses: dict[str, dict[str, dict[str, float]]] = {}
    for lens in _CONFIDENCE_CALIBRATION_LENSES:
        if lens not in resolved.columns:
            continue
        lens_map: dict[str, dict[str, float]] = {}
        for value, bucket_df in resolved.groupby(lens, dropna=False):
            sample_n = int(len(bucket_df))
            if sample_n < int(min_samples):
                continue
            bucket_weights = pd.to_numeric(bucket_df.get("recency_weight"), errors="coerce").fillna(1.0)
            follow_pct = float(_weighted_mean(bucket_df["is_follow"], bucket_weights) * 100.0)
            avg_dir = float(_weighted_mean(bucket_df["directional_return_pct"], bucket_weights))
            edge_score = ((follow_pct - overall_follow_pct) * 0.65) + ((avg_dir - overall_avg_return) * 5.5)
            lens_map[str(value or "Unknown")] = {
                "resolved": float(sample_n),
                "follow_pct": follow_pct,
                "avg_dir_return": avg_dir,
                "edge_score": edge_score,
            }
        if lens_map:
            lenses[lens] = lens_map
    return {
        "resolved_count": int(len(resolved)),
        "overall_follow_pct": overall_follow_pct,
        "overall_avg_return": overall_avg_return,
        "lenses": lenses,
    }


def build_confidence_calibration_snapshot(
    model: dict[str, object],
    *,
    signal: dict[str, object],
) -> ConfidenceCalibrationSnapshot:
    resolved_count = int(model.get("resolved_count") or 0)
    if resolved_count < _CONFIDENCE_CALIBRATION_MIN_RESOLVED:
        return ConfidenceCalibrationSnapshot(delta=0.0, note="", matched_factors=0, resolved_sample=resolved_count)

    signal_values = dict(signal or {})
    signal_values["Direction"] = _direction_value(signal_values.get("Direction"))
    signal_values["AI Alignment"] = (
        "Aligned" if str(signal_values.get("AI Alignment") or "").strip().lower().startswith("aligned") else "Not aligned"
    )
    signal_values["Timeframe"] = str(signal_values.get("Timeframe") or "").strip() or "Unknown"
    signal_values["Scan Focus"] = str(signal_values.get("Scan Focus") or "").strip() or "Unknown"
    signal_values["Direction x Timeframe"] = _compose_combo_value(
        signal_values.get("Direction"),
        signal_values.get("Timeframe"),
    )
    signal_values["Timeframe x AI Alignment"] = _compose_combo_value(
        signal_values.get("Timeframe"),
        signal_values.get("AI Alignment"),
    )
    signal_values["Direction x AI Alignment"] = _compose_combo_value(
        signal_values.get("Direction"),
        signal_values.get("AI Alignment"),
    )

    lenses = dict(model.get("lenses") or {})
    contributions: list[tuple[float, str, float]] = []
    for lens, weight in _CONFIDENCE_CALIBRATION_LENSES.items():
        value = str(signal_values.get(lens) or "").strip() or "Unknown"
        lens_map = lenses.get(lens)
        if not isinstance(lens_map, dict):
            continue
        bucket = lens_map.get(value)
        if not isinstance(bucket, dict):
            continue
        bucket_resolved = float(bucket.get("resolved") or 0.0)
        if bucket_resolved < float(_CONFIDENCE_CALIBRATION_MIN_BUCKET):
            continue
        edge_score = float(bucket.get("edge_score") or 0.0)
        sample_factor = min(1.0, bucket_resolved / 40.0)
        weighted = edge_score * float(weight) * sample_factor * 0.08
        if abs(weighted) < 0.18:
            continue
        contributions.append((weighted, f"{lens}: {value}", bucket_resolved))

    if not contributions:
        return ConfidenceCalibrationSnapshot(delta=0.0, note="", matched_factors=0, resolved_sample=resolved_count)

    contributions.sort(key=lambda item: abs(item[0]), reverse=True)
    delta = max(
        -_CONFIDENCE_CALIBRATION_MAX_DELTA,
        min(_CONFIDENCE_CALIBRATION_MAX_DELTA, sum(value for value, _, _ in contributions)),
    )
    if abs(delta) < 0.35:
        return ConfidenceCalibrationSnapshot(delta=0.0, note="", matched_factors=0, resolved_sample=resolved_count)

    strongest = [name for value, name, _ in contributions if value > 0][:2]
    weakest = [name for value, name, _ in contributions if value < 0][:2]
    if delta > 0.0 and strongest:
        note = "Archive confidence calibration is modestly supportive here. Strongest cohorts: " + ", ".join(strongest) + "."
    elif delta < 0.0 and weakest:
        note = "Archive confidence calibration is modestly cautious here. Weakest cohorts: " + ", ".join(weakest) + "."
    else:
        note = ""

    return ConfidenceCalibrationSnapshot(
        delta=float(delta),
        note=note,
        matched_factors=len(contributions),
        resolved_sample=resolved_count,
    )


def build_setup_calibration_model(df_events: pd.DataFrame, *, min_samples: int = 10) -> dict[str, object]:
    resolved = _prepare_resolved_events(df_events)
    if resolved.empty:
        return {
            "resolved_count": 0,
            "overall_follow_pct": 0.0,
            "overall_avg_return": 0.0,
            "lenses": {},
        }

    recency_weights = pd.to_numeric(resolved.get("recency_weight"), errors="coerce").fillna(1.0)
    overall_follow_pct = float(_weighted_mean(resolved["is_follow"], recency_weights) * 100.0)
    overall_avg_return = float(_weighted_mean(resolved["directional_return_pct"], recency_weights))
    lenses: dict[str, dict[str, dict[str, float]]] = {}
    for lens in _SETUP_CALIBRATION_LENSES:
        if lens not in resolved.columns:
            continue
        lens_map: dict[str, dict[str, float]] = {}
        for value, bucket_df in resolved.groupby(lens, dropna=False):
            sample_n = int(len(bucket_df))
            if sample_n < int(min_samples):
                continue
            bucket_weights = pd.to_numeric(bucket_df.get("recency_weight"), errors="coerce").fillna(1.0)
            follow_pct = float(_weighted_mean(bucket_df["is_follow"], bucket_weights) * 100.0)
            avg_dir = float(_weighted_mean(bucket_df["directional_return_pct"], bucket_weights))
            edge_score = ((follow_pct - overall_follow_pct) * 0.70) + ((avg_dir - overall_avg_return) * 6.0)
            lens_map[str(value or "Unknown")] = {
                "resolved": float(sample_n),
                "follow_pct": follow_pct,
                "avg_dir_return": avg_dir,
                "edge_score": edge_score,
            }
        if lens_map:
            lenses[lens] = lens_map
    return {
        "resolved_count": int(len(resolved)),
        "overall_follow_pct": overall_follow_pct,
        "overall_avg_return": overall_avg_return,
        "lenses": lenses,
    }


def build_setup_calibration_snapshot(model: dict[str, object], *, signal: dict[str, object]) -> SetupCalibrationSnapshot:
    resolved_count = int(model.get("resolved_count") or 0)
    if resolved_count < _SETUP_CALIBRATION_MIN_RESOLVED:
        return SetupCalibrationSnapshot(delta=0.0, note="", matched_factors=0, resolved_sample=resolved_count)

    signal_values = dict(signal or {})
    signal_values["Setup Class"] = _setup_class_value(signal_values.get("Setup Confirm"))
    signal_values["AI Alignment"] = (
        "Aligned" if str(signal_values.get("AI Alignment") or "").strip().lower().startswith("aligned") else "Not aligned"
    )
    signal_values["Timeframe"] = str(signal_values.get("Timeframe") or "").strip() or "Unknown"
    signal_values["Direction"] = _direction_value(signal_values.get("Direction"))
    signal_values["Scan Focus"] = str(signal_values.get("Scan Focus") or "").strip() or "Unknown"
    signal_values["Timeframe x Setup Class"] = _compose_combo_value(
        signal_values.get("Timeframe"),
        signal_values.get("Setup Class"),
    )
    signal_values["Setup Class x Direction"] = _compose_combo_value(
        signal_values.get("Setup Class"),
        signal_values.get("Direction"),
    )
    signal_values["Setup Class x AI Alignment"] = _compose_combo_value(
        signal_values.get("Setup Class"),
        signal_values.get("AI Alignment"),
    )
    signal_values["Scan Focus x Setup Class"] = _compose_combo_value(
        signal_values.get("Scan Focus"),
        signal_values.get("Setup Class"),
    )

    lenses = dict(model.get("lenses") or {})
    contributions: list[tuple[float, str, float]] = []
    for lens, weight in _SETUP_CALIBRATION_LENSES.items():
        value = str(signal_values.get(lens) or "").strip() or "Unknown"
        lens_map = lenses.get(lens)
        if not isinstance(lens_map, dict):
            continue
        bucket = lens_map.get(value)
        if not isinstance(bucket, dict):
            continue
        bucket_resolved = float(bucket.get("resolved") or 0.0)
        if bucket_resolved < float(_SETUP_CALIBRATION_MIN_BUCKET):
            continue
        edge_score = float(bucket.get("edge_score") or 0.0)
        sample_factor = min(1.0, bucket_resolved / 45.0)
        weighted = edge_score * float(weight) * sample_factor * 0.07
        if abs(weighted) < 0.16:
            continue
        contributions.append((weighted, f"{lens}: {value}", bucket_resolved))

    if not contributions:
        return SetupCalibrationSnapshot(delta=0.0, note="", matched_factors=0, resolved_sample=resolved_count)

    contributions.sort(key=lambda item: abs(item[0]), reverse=True)
    delta = max(
        -_SETUP_CALIBRATION_MAX_DELTA,
        min(_SETUP_CALIBRATION_MAX_DELTA, sum(value for value, _, _ in contributions)),
    )
    if abs(delta) < 0.35:
        return SetupCalibrationSnapshot(delta=0.0, note="", matched_factors=0, resolved_sample=resolved_count)

    strongest = [name for value, name, _ in contributions if value > 0][:2]
    weakest = [name for value, name, _ in contributions if value < 0][:2]
    if delta > 0.0 and strongest:
        note = "Archive setup calibration is modestly supportive here. Strongest cohorts: " + ", ".join(strongest) + "."
    elif delta < 0.0 and weakest:
        note = "Archive setup calibration is modestly cautious here. Weakest cohorts: " + ", ".join(weakest) + "."
    else:
        note = ""

    return SetupCalibrationSnapshot(
        delta=float(delta),
        note=note,
        matched_factors=len(contributions),
        resolved_sample=resolved_count,
    )


def build_actionable_ranking_model(df_events: pd.DataFrame, *, min_samples: int = 10) -> dict[str, object]:
    resolved = _prepare_resolved_events(df_events)
    if resolved.empty:
        return {
            "resolved_count": 0,
            "overall_follow_pct": 0.0,
            "overall_avg_return": 0.0,
            "lenses": {},
        }

    recency_weights = pd.to_numeric(resolved.get("recency_weight"), errors="coerce").fillna(1.0)
    overall_follow_pct = float(_weighted_mean(resolved["is_follow"], recency_weights) * 100.0)
    overall_avg_return = float(_weighted_mean(resolved["directional_return_pct"], recency_weights))
    lenses: dict[str, dict[str, dict[str, float]]] = {}
    for lens in _ACTIONABLE_RANKING_LENSES:
        if lens not in resolved.columns:
            continue
        lens_map: dict[str, dict[str, float]] = {}
        for value, bucket_df in resolved.groupby(lens, dropna=False):
            sample_n = int(len(bucket_df))
            if sample_n < int(min_samples):
                continue
            bucket_weights = pd.to_numeric(bucket_df.get("recency_weight"), errors="coerce").fillna(1.0)
            follow_pct = float(_weighted_mean(bucket_df["is_follow"], bucket_weights) * 100.0)
            avg_dir = float(_weighted_mean(bucket_df["directional_return_pct"], bucket_weights))
            edge_score = ((follow_pct - overall_follow_pct) * 0.75) + ((avg_dir - overall_avg_return) * 6.0)
            lens_map[str(value or "Unknown")] = {
                "resolved": float(sample_n),
                "follow_pct": follow_pct,
                "avg_dir_return": avg_dir,
                "edge_score": edge_score,
            }
        if lens_map:
            lenses[lens] = lens_map
    return {
        "resolved_count": int(len(resolved)),
        "overall_follow_pct": overall_follow_pct,
        "overall_avg_return": overall_avg_return,
        "lenses": lenses,
    }


def build_actionable_ranking_snapshot(model: dict[str, object], *, signal: dict[str, object]) -> ActionableRankingSnapshot:
    resolved_count = int(model.get("resolved_count") or 0)
    if resolved_count < _ACTIONABLE_RANKING_MIN_RESOLVED:
        return ActionableRankingSnapshot(delta=0.0, note="", matched_factors=0, resolved_sample=resolved_count)

    signal_values = dict(signal or {})
    signal_values["Setup Class"] = _setup_class_value(signal_values.get("Setup Confirm"))
    signal_values["AI Alignment"] = (
        "Aligned" if str(signal_values.get("AI Alignment") or "").strip().lower().startswith("aligned") else "Not aligned"
    )
    signal_values["Timeframe"] = str(signal_values.get("Timeframe") or "").strip() or "Unknown"
    signal_values["Direction"] = _direction_value(signal_values.get("Direction"))
    signal_values["Scan Focus"] = str(signal_values.get("Scan Focus") or "").strip() or "Unknown"
    signal_values["Timeframe x Setup Class"] = _compose_combo_value(
        signal_values.get("Timeframe"),
        signal_values.get("Setup Class"),
    )
    signal_values["Setup Class x Direction"] = _compose_combo_value(
        signal_values.get("Setup Class"),
        signal_values.get("Direction"),
    )
    signal_values["Setup Class x AI Alignment"] = _compose_combo_value(
        signal_values.get("Setup Class"),
        signal_values.get("AI Alignment"),
    )
    signal_values["Scan Focus x Setup Class"] = _compose_combo_value(
        signal_values.get("Scan Focus"),
        signal_values.get("Setup Class"),
    )

    lenses = dict(model.get("lenses") or {})
    contributions: list[tuple[float, str, float]] = []
    for lens, weight in _ACTIONABLE_RANKING_LENSES.items():
        value = str(signal_values.get(lens) or "").strip() or "Unknown"
        lens_map = lenses.get(lens)
        if not isinstance(lens_map, dict):
            continue
        bucket = lens_map.get(value)
        if not isinstance(bucket, dict):
            continue
        bucket_resolved = float(bucket.get("resolved") or 0.0)
        if bucket_resolved < float(_ACTIONABLE_RANKING_MIN_BUCKET):
            continue
        edge_score = float(bucket.get("edge_score") or 0.0)
        sample_factor = min(1.0, bucket_resolved / 36.0)
        weighted = edge_score * float(weight) * sample_factor * 0.10
        if abs(weighted) < 0.20:
            continue
        contributions.append((weighted, f"{lens}: {value}", bucket_resolved))

    if not contributions:
        return ActionableRankingSnapshot(delta=0.0, note="", matched_factors=0, resolved_sample=resolved_count)

    contributions.sort(key=lambda item: abs(item[0]), reverse=True)
    delta = max(
        -_ACTIONABLE_RANKING_MAX_DELTA,
        min(_ACTIONABLE_RANKING_MAX_DELTA, sum(value for value, _, _ in contributions)),
    )
    if abs(delta) < 0.40:
        return ActionableRankingSnapshot(delta=0.0, note="", matched_factors=0, resolved_sample=resolved_count)

    strongest = [name for value, name, _ in contributions if value > 0][:2]
    weakest = [name for value, name, _ in contributions if value < 0][:2]
    if delta > 0.0 and strongest:
        note = "Archive ranking is supportive here. Strongest cohorts: " + ", ".join(strongest) + "."
    elif delta < 0.0 and weakest:
        note = "Archive ranking is cautious here. Weakest cohorts: " + ", ".join(weakest) + "."
    else:
        note = ""

    return ActionableRankingSnapshot(
        delta=float(delta),
        note=note,
        matched_factors=len(contributions),
        resolved_sample=resolved_count,
    )


def build_risk_sizing_calibration_model(df_events: pd.DataFrame, *, min_samples: int = 10) -> dict[str, object]:
    resolved = _prepare_resolved_events(df_events)
    if resolved.empty:
        return {
            "resolved_count": 0,
            "overall_follow_pct": 0.0,
            "overall_avg_return": 0.0,
            "lenses": {},
        }

    recency_weights = pd.to_numeric(resolved.get("recency_weight"), errors="coerce").fillna(1.0)
    overall_follow_pct = float(_weighted_mean(resolved["is_follow"], recency_weights) * 100.0)
    overall_avg_return = float(_weighted_mean(resolved["directional_return_pct"], recency_weights))
    lenses: dict[str, dict[str, dict[str, float]]] = {}
    for lens in _RISK_SIZING_CALIBRATION_LENSES:
        if lens not in resolved.columns:
            continue
        lens_map: dict[str, dict[str, float]] = {}
        for value, bucket_df in resolved.groupby(lens, dropna=False):
            sample_n = int(len(bucket_df))
            if sample_n < int(min_samples):
                continue
            bucket_weights = pd.to_numeric(bucket_df.get("recency_weight"), errors="coerce").fillna(1.0)
            follow_pct = float(_weighted_mean(bucket_df["is_follow"], bucket_weights) * 100.0)
            avg_dir = float(_weighted_mean(bucket_df["directional_return_pct"], bucket_weights))
            edge_score = ((follow_pct - overall_follow_pct) * 0.75) + ((avg_dir - overall_avg_return) * 6.0)
            lens_map[str(value or "Unknown")] = {
                "resolved": float(sample_n),
                "follow_pct": follow_pct,
                "avg_dir_return": avg_dir,
                "edge_score": edge_score,
            }
        if lens_map:
            lenses[lens] = lens_map
    return {
        "resolved_count": int(len(resolved)),
        "overall_follow_pct": overall_follow_pct,
        "overall_avg_return": overall_avg_return,
        "lenses": lenses,
    }


def build_risk_sizing_calibration_snapshot(
    model: dict[str, object],
    *,
    signal: dict[str, object],
) -> RiskSizingCalibrationSnapshot:
    resolved_count = int(model.get("resolved_count") or 0)
    if resolved_count < _RISK_SIZING_MIN_RESOLVED:
        return RiskSizingCalibrationSnapshot(delta=0.0, note="", matched_factors=0, resolved_sample=resolved_count)

    signal_values = dict(signal or {})
    signal_values["Setup Class"] = _setup_class_value(signal_values.get("Setup Confirm"))
    signal_values["AI Alignment"] = (
        "Aligned" if str(signal_values.get("AI Alignment") or "").strip().lower().startswith("aligned") else "Not aligned"
    )
    signal_values["Timeframe"] = str(signal_values.get("Timeframe") or "").strip() or "Unknown"
    signal_values["Direction"] = _direction_value(signal_values.get("Direction"))
    signal_values["Scan Focus"] = str(signal_values.get("Scan Focus") or "").strip() or "Unknown"
    signal_values["Timeframe x Setup Class"] = _compose_combo_value(
        signal_values.get("Timeframe"),
        signal_values.get("Setup Class"),
    )
    signal_values["Setup Class x Direction"] = _compose_combo_value(
        signal_values.get("Setup Class"),
        signal_values.get("Direction"),
    )
    signal_values["Setup Class x AI Alignment"] = _compose_combo_value(
        signal_values.get("Setup Class"),
        signal_values.get("AI Alignment"),
    )
    signal_values["Scan Focus x Setup Class"] = _compose_combo_value(
        signal_values.get("Scan Focus"),
        signal_values.get("Setup Class"),
    )

    lenses = dict(model.get("lenses") or {})
    contributions: list[tuple[float, str, float]] = []
    for lens, weight in _RISK_SIZING_CALIBRATION_LENSES.items():
        value = str(signal_values.get(lens) or "").strip() or "Unknown"
        lens_map = lenses.get(lens)
        if not isinstance(lens_map, dict):
            continue
        bucket = lens_map.get(value)
        if not isinstance(bucket, dict):
            continue
        bucket_resolved = float(bucket.get("resolved") or 0.0)
        if bucket_resolved < float(_RISK_SIZING_MIN_BUCKET):
            continue
        edge_score = float(bucket.get("edge_score") or 0.0)
        sample_factor = min(1.0, bucket_resolved / 45.0)
        weighted = edge_score * float(weight) * sample_factor * 0.004
        if abs(weighted) < 0.015:
            continue
        contributions.append((weighted, f"{lens}: {value}", bucket_resolved))

    if not contributions:
        return RiskSizingCalibrationSnapshot(delta=0.0, note="", matched_factors=0, resolved_sample=resolved_count)

    contributions.sort(key=lambda item: abs(item[0]), reverse=True)
    delta = max(
        -_RISK_SIZING_MAX_DELTA,
        min(_RISK_SIZING_MAX_DELTA, sum(value for value, _, _ in contributions)),
    )
    if abs(delta) < 0.04:
        return RiskSizingCalibrationSnapshot(delta=0.0, note="", matched_factors=0, resolved_sample=resolved_count)

    strongest = [name for value, name, _ in contributions if value > 0][:2]
    weakest = [name for value, name, _ in contributions if value < 0][:2]
    if delta > 0.0 and strongest:
        note = "Archive sizing calibration is modestly supportive here. Strongest cohorts: " + ", ".join(strongest) + "."
    elif delta < 0.0 and weakest:
        note = "Archive sizing calibration is modestly cautious here. Weakest cohorts: " + ", ".join(weakest) + "."
    else:
        note = ""

    return RiskSizingCalibrationSnapshot(
        delta=float(delta),
        note=note,
        matched_factors=len(contributions),
        resolved_sample=resolved_count,
    )


def build_trade_gate_calibration_model(df_events: pd.DataFrame, *, min_samples: int = 10) -> dict[str, object]:
    resolved = _prepare_resolved_events(df_events)
    if resolved.empty:
        return {
            "resolved_count": 0,
            "overall_follow_pct": 0.0,
            "overall_avg_return": 0.0,
            "lenses": {},
        }

    recency_weights = pd.to_numeric(resolved.get("recency_weight"), errors="coerce").fillna(1.0)
    overall_follow_pct = float(_weighted_mean(resolved["is_follow"], recency_weights) * 100.0)
    overall_avg_return = float(_weighted_mean(resolved["directional_return_pct"], recency_weights))
    lenses: dict[str, dict[str, dict[str, float]]] = {}
    for lens in _TRADE_GATE_CALIBRATION_LENSES:
        if lens not in resolved.columns:
            continue
        lens_map: dict[str, dict[str, float]] = {}
        for value, bucket_df in resolved.groupby(lens, dropna=False):
            sample_n = int(len(bucket_df))
            if sample_n < int(min_samples):
                continue
            bucket_weights = pd.to_numeric(bucket_df.get("recency_weight"), errors="coerce").fillna(1.0)
            follow_pct = float(_weighted_mean(bucket_df["is_follow"], bucket_weights) * 100.0)
            avg_dir = float(_weighted_mean(bucket_df["directional_return_pct"], bucket_weights))
            edge_score = ((follow_pct - overall_follow_pct) * 0.70) + ((avg_dir - overall_avg_return) * 5.5)
            lens_map[str(value or "Unknown")] = {
                "resolved": float(sample_n),
                "follow_pct": follow_pct,
                "avg_dir_return": avg_dir,
                "edge_score": edge_score,
            }
        if lens_map:
            lenses[lens] = lens_map
    return {
        "resolved_count": int(len(resolved)),
        "overall_follow_pct": overall_follow_pct,
        "overall_avg_return": overall_avg_return,
        "lenses": lenses,
    }


def build_trade_gate_calibration_snapshot(
    model: dict[str, object],
    *,
    signal: dict[str, object],
) -> TradeGateCalibrationSnapshot:
    resolved_count = int(model.get("resolved_count") or 0)
    if resolved_count < _TRADE_GATE_MIN_RESOLVED:
        return TradeGateCalibrationSnapshot(delta=0.0, note="", matched_factors=0, resolved_sample=resolved_count)

    signal_values = dict(signal or {})
    signal_values["Playbook"] = _playbook_value(signal_values) or str(signal_values.get("Playbook") or "Unknown")
    signal_values["Trade Gate"] = _trade_gate_value(signal_values) or str(signal_values.get("Trade Gate") or "Unknown")
    signal_values["Market Regime"] = str(signal_values.get("Market Regime") or "").strip() or "Unknown"
    signal_values["Session"] = str(signal_values.get("Session") or "").strip() or "Unknown"
    signal_values["Catalyst Window"] = str(signal_values.get("Catalyst Window") or "").strip() or "Unknown"
    signal_values["Playbook x Session"] = _compose_combo_value(
        signal_values.get("Playbook"),
        signal_values.get("Session"),
    )
    signal_values["Playbook x Catalyst Window"] = _compose_combo_value(
        signal_values.get("Playbook"),
        signal_values.get("Catalyst Window"),
    )

    lenses = dict(model.get("lenses") or {})
    contributions: list[tuple[float, str, float]] = []
    for lens, weight in _TRADE_GATE_CALIBRATION_LENSES.items():
        value = str(signal_values.get(lens) or "").strip() or "Unknown"
        lens_map = lenses.get(lens)
        if not isinstance(lens_map, dict):
            continue
        bucket = lens_map.get(value)
        if not isinstance(bucket, dict):
            continue
        bucket_resolved = float(bucket.get("resolved") or 0.0)
        if bucket_resolved < float(_TRADE_GATE_MIN_BUCKET):
            continue
        edge_score = float(bucket.get("edge_score") or 0.0)
        sample_factor = min(1.0, bucket_resolved / 45.0)
        weighted = edge_score * float(weight) * sample_factor * 0.02
        if abs(weighted) < 0.08:
            continue
        contributions.append((weighted, f"{lens}: {value}", bucket_resolved))

    if not contributions:
        return TradeGateCalibrationSnapshot(delta=0.0, note="", matched_factors=0, resolved_sample=resolved_count)

    contributions.sort(key=lambda item: abs(item[0]), reverse=True)
    delta = max(
        -_TRADE_GATE_MAX_DELTA,
        min(_TRADE_GATE_MAX_DELTA, sum(value for value, _, _ in contributions)),
    )
    if abs(delta) < 0.18:
        return TradeGateCalibrationSnapshot(delta=0.0, note="", matched_factors=0, resolved_sample=resolved_count)

    strongest = [name for value, name, _ in contributions if value > 0][:2]
    weakest = [name for value, name, _ in contributions if value < 0][:2]
    if delta > 0.0 and strongest:
        note = "Archive gate calibration is modestly supportive here. Strongest cohorts: " + ", ".join(strongest) + "."
    elif delta < 0.0 and weakest:
        note = "Archive gate calibration is modestly cautious here. Weakest cohorts: " + ", ".join(weakest) + "."
    else:
        note = ""

    return TradeGateCalibrationSnapshot(
        delta=float(delta),
        note=note,
        matched_factors=len(contributions),
        resolved_sample=resolved_count,
    )


def build_adaptive_context_model(df_events: pd.DataFrame, *, min_samples: int = 5) -> dict[str, object]:
    resolved = _prepare_resolved_events(df_events)
    if resolved.empty:
        return {
            "resolved_count": 0,
            "overall_follow_pct": 0.0,
            "overall_avg_return": 0.0,
            "lenses": {},
        }

    recency_weights = pd.to_numeric(resolved.get("recency_weight"), errors="coerce").fillna(1.0)
    overall_follow_pct = float(_weighted_mean(resolved["is_follow"], recency_weights) * 100.0)
    overall_avg_return = float(_weighted_mean(resolved["directional_return_pct"], recency_weights))
    actual_closed = resolved[resolved["is_trade_closed"] == 1].copy()
    overall_actual_closed = int(len(actual_closed))
    if len(actual_closed):
        actual_weights = pd.to_numeric(actual_closed.get("recency_weight"), errors="coerce").fillna(1.0)
        overall_actual_win_pct = float(
            _weighted_mean((pd.to_numeric(actual_closed["actual_pnl_pct"], errors="coerce") > 0).astype(float), actual_weights) * 100.0
        )
        overall_avg_actual_pnl = float(_weighted_mean(actual_closed["actual_pnl_pct"], actual_weights))
    else:
        overall_actual_win_pct = 0.0
        overall_avg_actual_pnl = 0.0
    lenses: dict[str, dict[str, dict[str, float]]] = {}
    for lens in _LENS_WEIGHTS:
        if lens not in resolved.columns:
            continue
        lens_map: dict[str, dict[str, float]] = {}
        for value, bucket_df in resolved.groupby(lens, dropna=False):
            sample_n = int(len(bucket_df))
            if sample_n < int(min_samples):
                continue
            bucket_weights = pd.to_numeric(bucket_df.get("recency_weight"), errors="coerce").fillna(1.0)
            follow_pct = float(_weighted_mean(bucket_df["is_follow"], bucket_weights) * 100.0)
            avg_dir = float(_weighted_mean(bucket_df["directional_return_pct"], bucket_weights))
            signal_edge = ((follow_pct - overall_follow_pct) * 0.55) + ((avg_dir - overall_avg_return) * 4.0)
            actual_bucket_df = bucket_df[bucket_df["is_trade_closed"] == 1].copy()
            actual_closed_n = float(len(actual_bucket_df))
            if actual_closed_n > 0.0:
                actual_weights = pd.to_numeric(actual_bucket_df.get("recency_weight"), errors="coerce").fillna(1.0)
                actual_win_pct = float(
                    _weighted_mean(
                        (pd.to_numeric(actual_bucket_df["actual_pnl_pct"], errors="coerce") > 0).astype(float),
                        actual_weights,
                    )
                    * 100.0
                )
                avg_actual_pnl = float(_weighted_mean(actual_bucket_df["actual_pnl_pct"], actual_weights))
            else:
                actual_win_pct = 0.0
                avg_actual_pnl = 0.0
            actual_edge = 0.0
            if overall_actual_closed >= 8 and actual_closed_n >= 3:
                actual_edge = ((actual_win_pct - overall_actual_win_pct) * 0.45) + (
                    (avg_actual_pnl - overall_avg_actual_pnl) * 5.0
                )
                actual_edge *= min(1.0, actual_closed_n / 10.0)
            edge_score = signal_edge + (actual_edge * 0.65)
            lens_map[str(value or "Unknown")] = {
                "resolved": float(sample_n),
                "follow_pct": follow_pct,
                "avg_dir_return": avg_dir,
                "actual_closed": actual_closed_n,
                "actual_win_pct": actual_win_pct,
                "avg_actual_pnl": avg_actual_pnl,
                "signal_edge": signal_edge,
                "actual_edge": actual_edge,
                "edge_score": edge_score,
            }
        if lens_map:
            lenses[lens] = lens_map
    return {
        "resolved_count": int(len(resolved)),
        "overall_follow_pct": overall_follow_pct,
        "overall_avg_return": overall_avg_return,
        "overall_actual_closed": overall_actual_closed,
        "overall_actual_win_pct": overall_actual_win_pct,
        "overall_avg_actual_pnl": overall_avg_actual_pnl,
        "lenses": lenses,
    }


def build_session_fit_snapshot(model: dict[str, object], session_value: str) -> SessionFitSnapshot:
    resolved_count = int(model.get("resolved_count") or 0)
    actual_trade_sample = int(model.get("overall_actual_closed") or 0)
    normalized_session = str(session_value or "").strip() or "Unknown"
    if resolved_count < 20:
        return SessionFitSnapshot(
            score=0.0,
            label="Session Unproven",
            note="The session archive is still building, so current timing fit is not proven yet.",
            resolved_sample=resolved_count,
            actual_trade_sample=actual_trade_sample,
        )

    lens_map = dict(dict(model.get("lenses") or {}).get("Session") or {})
    bucket = lens_map.get(normalized_session)
    if not bucket:
        return SessionFitSnapshot(
            score=0.0,
            label="Session Mixed",
            note=f"There is not enough matched history yet for {normalized_session}.",
            resolved_sample=resolved_count,
            actual_trade_sample=actual_trade_sample,
        )

    bucket_resolved = float(bucket.get("resolved") or 0.0)
    score = float(bucket.get("edge_score") or 0.0)
    actual_closed = float(bucket.get("actual_closed") or 0.0)
    actual_win_pct = float(bucket.get("actual_win_pct") or 0.0)
    avg_actual_pnl = float(bucket.get("avg_actual_pnl") or 0.0)
    follow_pct = float(bucket.get("follow_pct") or 0.0)

    if actual_closed >= 4 and actual_win_pct >= 55.0 and avg_actual_pnl >= 0.2 and score >= 1.5:
        label = "Session Supportive"
        note = (
            f"{normalized_session} has been one of the cleaner execution windows lately "
            f"({actual_win_pct:.0f}% closed-trade win rate, {avg_actual_pnl:+.2f}% avg realized PnL)."
        )
    elif bucket_resolved >= 5 and score <= -2.5:
        label = "Session Fragile"
        note = (
            f"{normalized_session} has been a weaker conversion window lately "
            f"({follow_pct:.0f}% signal follow-through on matched history)."
        )
    else:
        label = "Session Mixed"
        note = f"{normalized_session} is usable, but the archive is mixed rather than decisively supportive."

    return SessionFitSnapshot(
        score=score,
        label=label,
        note=note,
        resolved_sample=resolved_count,
        actual_trade_sample=actual_trade_sample,
    )


def build_archive_guardrail_snapshot(model: dict[str, object], *, signal: dict[str, str]) -> ArchiveGuardrailSnapshot:
    resolved_count = int(model.get("resolved_count") or 0)
    if resolved_count < 20:
        return ArchiveGuardrailSnapshot(
            penalty=0.0,
            label="Archive Clear",
            note="",
            matched_factors=0,
        )

    lenses = dict(model.get("lenses") or {})
    hits: list[tuple[float, str, str]] = []
    signal_values = dict(signal or {})
    signal_values["Playbook"] = _playbook_value(signal_values) or str(signal_values.get("Playbook") or "Unknown")
    signal_values["Trade Gate"] = _trade_gate_value(signal_values) or str(signal_values.get("Trade Gate") or "Unknown")
    signal_values["Playbook x Session"] = _compose_combo_value(
        signal_values.get("Playbook"),
        signal_values.get("Session"),
    )
    signal_values["Playbook x Catalyst Window"] = _compose_combo_value(
        signal_values.get("Playbook"),
        signal_values.get("Catalyst Window"),
    )
    signal_values["Primary Alert"] = _primary_alert_value(signal_values)
    signal_values["Primary Alert x Playbook"] = _compose_combo_value(
        signal_values.get("Primary Alert"),
        signal_values.get("Playbook"),
    )
    signal_values["Primary Alert x Session"] = _compose_combo_value(
        signal_values.get("Primary Alert"),
        signal_values.get("Session"),
    )
    for lens, weight in _ARCHIVE_GUARDRAIL_LENSES.items():
        value = str(signal_values.get(lens) or "").strip() or "Unknown"
        lens_map = lenses.get(lens)
        if not isinstance(lens_map, dict):
            continue
        bucket = lens_map.get(value)
        if not isinstance(bucket, dict):
            continue
        bucket_resolved = float(bucket.get("resolved") or 0.0)
        if bucket_resolved < 6.0:
            continue
        edge_score = float(bucket.get("edge_score") or 0.0)
        actual_edge = float(bucket.get("actual_edge") or 0.0)
        actual_closed = float(bucket.get("actual_closed") or 0.0)
        if edge_score > -2.25 and actual_edge > -0.75:
            continue
        severity = max(0.0, abs(min(0.0, edge_score)) - 1.5) * float(weight) * min(1.0, bucket_resolved / 16.0)
        if actual_closed >= 3.0 and actual_edge < 0.0:
            severity += min(1.75, abs(actual_edge) * 0.9)
        if severity < 0.6:
            continue
        hits.append((severity, lens, value))

    if not hits:
        return ArchiveGuardrailSnapshot(
            penalty=0.0,
            label="Archive Clear",
            note="",
            matched_factors=0,
        )

    hits.sort(key=lambda item: item[0], reverse=True)
    penalty = float(min(8.0, sum(value for value, _, _ in hits)))
    weakest_hits = list(hits[:2])
    trade_gate_hit = next((item for item in hits if item[1] == "Trade Gate"), None)
    if trade_gate_hit and all(item[1] != "Trade Gate" for item in weakest_hits):
        weakest_hits = [trade_gate_hit, weakest_hits[0]] if weakest_hits else [trade_gate_hit]
    weakest = [f"{lens}: {value}" for _, lens, value in weakest_hits[:2]]
    if penalty >= 5.0:
        label = "Archive Guardrail"
        note = (
            "Matched archive history is weak enough here to actively trim aggression. "
            f"Weakest buckets: {', '.join(weakest)}."
        )
    else:
        label = "Archive Caution"
        note = (
            "Matched archive history is soft in this window. "
            f"Watch the weaker buckets: {', '.join(weakest)}."
        )
    return ArchiveGuardrailSnapshot(
        penalty=penalty,
        label=label,
        note=note,
        matched_factors=len(hits),
    )


def build_live_signal_adaptive_snapshot(model: dict[str, object], *, signal: dict[str, str]) -> AdaptiveEdgeSnapshot:
    resolved_count = int(model.get("resolved_count") or 0)
    actual_trade_sample = int(model.get("overall_actual_closed") or 0)
    session_fit_snapshot = build_session_fit_snapshot(model, str(signal.get("Session") or "Unknown"))
    archive_guardrail_snapshot = build_archive_guardrail_snapshot(model, signal=signal)
    signal_values = dict(signal or {})
    signal_values["Playbook"] = _playbook_value(signal_values) or str(signal_values.get("Playbook") or "Unknown")
    signal_values["Trade Gate"] = _trade_gate_value(signal_values) or str(signal_values.get("Trade Gate") or "Unknown")
    signal_values["Playbook x Session"] = _compose_combo_value(
        signal_values.get("Playbook"),
        signal_values.get("Session"),
    )
    signal_values["Playbook x Catalyst Window"] = _compose_combo_value(
        signal_values.get("Playbook"),
        signal_values.get("Catalyst Window"),
    )
    signal_values["Primary Alert"] = _primary_alert_value(signal_values)
    signal_values["Primary Alert x Playbook"] = _compose_combo_value(
        signal_values.get("Primary Alert"),
        signal_values.get("Playbook"),
    )
    signal_values["Primary Alert x Session"] = _compose_combo_value(
        signal_values.get("Primary Alert"),
        signal_values.get("Session"),
    )
    if resolved_count < 20:
        return AdaptiveEdgeSnapshot(
            score=50.0,
            label="No Learned Edge Yet",
            note="The review database does not have enough resolved history yet to weight live setups confidently.",
            matched_factors=0,
            resolved_sample=resolved_count,
            actual_trade_sample=actual_trade_sample,
            execution_fit_label="Execution Unproven",
            execution_fit_note="There are not enough closed real trades yet to judge execution fit on similar setups.",
            session_fit_score=float(session_fit_snapshot.score),
            session_fit_label=session_fit_snapshot.label,
            session_fit_note=session_fit_snapshot.note,
            archive_guardrail_penalty=float(archive_guardrail_snapshot.penalty),
            archive_guardrail_label=archive_guardrail_snapshot.label,
            archive_guardrail_note=archive_guardrail_snapshot.note,
        )

    lenses = dict(model.get("lenses") or {})
    contributions: list[tuple[float, str, float]] = []
    actual_contributions: list[float] = []
    for lens, weight in _LENS_WEIGHTS.items():
        value = str(signal_values.get(lens) or "").strip() or "Unknown"
        lens_map = lenses.get(lens)
        if not isinstance(lens_map, dict) or value not in lens_map:
            continue
        bucket = lens_map[value]
        bucket_resolved = float(bucket.get("resolved") or 0.0)
        bucket_edge = float(bucket.get("edge_score") or 0.0)
        sample_factor = min(1.0, bucket_resolved / 25.0)
        weighted = bucket_edge * float(weight) * sample_factor * 0.35
        actual_edge = float(bucket.get("actual_edge") or 0.0)
        actual_weighted = actual_edge * float(weight) * sample_factor * 0.35 * 0.65
        if abs(actual_weighted) >= 0.35:
            actual_contributions.append(actual_weighted)
        if abs(weighted) < 0.5:
            continue
        contributions.append((weighted, f"{lens}: {value}", bucket_resolved))

    if not contributions:
        return AdaptiveEdgeSnapshot(
            score=50.0,
            label="No Learned Edge Yet",
            note="History is available, but not enough matching cohorts are strong enough to change the live read yet.",
            matched_factors=0,
            resolved_sample=resolved_count,
            actual_trade_sample=actual_trade_sample,
            execution_fit_label="Execution Unproven" if actual_trade_sample < 8 else "Execution Mixed",
            execution_fit_note=(
                "There are not enough closed real trades yet to judge execution fit on similar setups."
                if actual_trade_sample < 8
                else "Real execution history is mixed for similar setups."
            ),
            session_fit_score=float(session_fit_snapshot.score),
            session_fit_label=session_fit_snapshot.label,
            session_fit_note=session_fit_snapshot.note,
            archive_guardrail_penalty=float(archive_guardrail_snapshot.penalty),
            archive_guardrail_label=archive_guardrail_snapshot.label,
            archive_guardrail_note=archive_guardrail_snapshot.note,
        )

    contributions.sort(key=lambda item: abs(item[0]), reverse=True)
    total = max(-18.0, min(18.0, sum(value for value, _, _ in contributions)))
    score = float(max(20.0, min(80.0, 50.0 + total)))
    positive = [name for value, name, _ in contributions if value > 0][:2]
    negative = [name for value, name, _ in contributions if value < 0][:2]

    preliminary_label = "Historically Neutral"
    if score >= 58.0:
        preliminary_label = "Historically Favored"
    elif score <= 42.0:
        preliminary_label = "Historically Weak"

    execution_stance = _execution_stance_value(
        signal_values.get("Trade Gate"),
        preliminary_label,
        archive_guardrail_snapshot.label,
    )
    stance_note = ""
    execution_stance_bucket = dict(lenses.get("Execution Stance") or {}).get(execution_stance)
    if isinstance(execution_stance_bucket, dict):
        bucket_resolved = float(execution_stance_bucket.get("resolved") or 0.0)
        sample_factor = min(1.0, bucket_resolved / 25.0)
        stance_edge = float(execution_stance_bucket.get("edge_score") or 0.0)
        stance_actual_edge = float(execution_stance_bucket.get("actual_edge") or 0.0)
        stance_adjustment = (stance_edge * 0.18 * sample_factor) + (stance_actual_edge * 0.12 * sample_factor)
        if abs(stance_adjustment) >= 0.2:
            total = max(-18.0, min(18.0, total + stance_adjustment))
            score = float(max(20.0, min(80.0, 50.0 + total)))
        if bucket_resolved >= 6.0:
            if stance_adjustment >= 0.35:
                stance_note = f"Execution stance {execution_stance} has also been supportive in recent archive history."
            elif stance_adjustment <= -0.35:
                stance_note = f"Execution stance {execution_stance} has also been fragile in recent archive history."

    if score >= 58.0:
        label = "Historically Favored"
        if positive:
            note = "History is leaning in your favor here. Strongest matching factors: " + ", ".join(positive) + "."
        else:
            note = "History is leaning in your favor here."
        if actual_trade_sample >= 8:
            note = f"{note} Real trade history is also contributing."
    elif score <= 42.0:
        label = "Historically Weak"
        if negative:
            note = "History has struggled with this mix. Weakest matching factors: " + ", ".join(negative) + "."
        else:
            note = "History has struggled with this mix."
        if actual_trade_sample >= 8:
            note = f"{note} Real trade history is also contributing."
    else:
        label = "Historically Neutral"
        note = "Resolved history is mixed for this setup mix. Treat the learned edge as neutral."

    if stance_note:
        note = f"{note} {stance_note}".strip()
    if archive_guardrail_snapshot.note:
        note = f"{note} {archive_guardrail_snapshot.label}: {archive_guardrail_snapshot.note}".strip()

    actual_total = sum(actual_contributions)
    if actual_trade_sample < 8:
        execution_fit_label = "Execution Unproven"
        execution_fit_note = "There are not enough closed real trades yet to judge execution fit on similar setups."
    elif actual_total >= 2.5:
        execution_fit_label = "Execution Proven"
        execution_fit_note = "Your own closed trades in similar setups have generally converted well."
    elif actual_total <= -2.5:
        execution_fit_label = "Execution Fragile"
        execution_fit_note = "Your own closed trades in similar setups have struggled to convert cleanly."
    else:
        execution_fit_label = "Execution Mixed"
        execution_fit_note = "Your own closed trades in similar setups are mixed so far."

    return AdaptiveEdgeSnapshot(
        score=score,
        label=label,
        note=note,
        matched_factors=len(contributions),
        resolved_sample=resolved_count,
        actual_trade_sample=actual_trade_sample,
        execution_fit_label=execution_fit_label,
        execution_fit_note=execution_fit_note,
        session_fit_score=float(session_fit_snapshot.score),
        session_fit_label=session_fit_snapshot.label,
        session_fit_note=session_fit_snapshot.note,
        archive_guardrail_penalty=float(archive_guardrail_snapshot.penalty),
        archive_guardrail_label=archive_guardrail_snapshot.label,
        archive_guardrail_note=archive_guardrail_snapshot.note,
    )


def build_learning_edge_table(model: dict[str, object], *, limit: int = 12) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for lens, values in dict(model.get("lenses") or {}).items():
        if not isinstance(values, dict):
            continue
        for value, bucket in values.items():
            rows.append(
                {
                    "Lens": lens,
                    "Bucket": value,
                    "Resolved": float(bucket.get("resolved") or 0.0),
                    "ActualClosed": float(bucket.get("actual_closed") or 0.0),
                    "FollowThroughPct": float(bucket.get("follow_pct") or 0.0),
                    "AvgDirReturnPct": float(bucket.get("avg_dir_return") or 0.0),
                    "ActualWinPct": float(bucket.get("actual_win_pct") or 0.0),
                    "AvgActualPnlPct": float(bucket.get("avg_actual_pnl") or 0.0),
                    "SignalEdge": float(bucket.get("signal_edge") or 0.0),
                    "ActualEdge": float(bucket.get("actual_edge") or 0.0),
                    "EdgeScore": float(bucket.get("edge_score") or 0.0),
                }
            )
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    return df.sort_values(["EdgeScore", "Resolved"], ascending=[False, False]).head(int(limit)).reset_index(drop=True)
