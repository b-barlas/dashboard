"""Empirical feedback learning from resolved signal history."""

from __future__ import annotations

from dataclasses import dataclass
import math

import pandas as pd
from core.session_utils import session_bucket_for_timestamp


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


def _execution_stance_value(trade_gate: object, adaptive_edge: object, archive_guardrail: object) -> str:
    gate = str(trade_gate or "").strip()
    adaptive = str(adaptive_edge or "").strip()
    guardrail = str(archive_guardrail or "").strip()

    if gate == "No-Trade" or guardrail == "Archive Guardrail":
        return "Stand Aside"
    if gate == "Defensive Only" or adaptive == "Historically Weak":
        return "Defensive Only"
    if gate == "Tradeable" and adaptive == "Historically Favored" and guardrail == "Archive Clear":
        return "Tradeable"
    if gate in {"Tradeable", "Selective Only"}:
        return "Selective Only"
    return gate or "Unknown"


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
    trade_gate = str(row_or_signal.get("market_trade_gate") or row_or_signal.get("Trade Gate") or "").strip()
    market_lead = str(row_or_signal.get("market_lead_label") or row_or_signal.get("Market Lead") or "").strip()
    flow_proxy = str(row_or_signal.get("market_flow_state") or row_or_signal.get("Flow Proxy") or "").strip()
    sector_rotation = str(row_or_signal.get("market_sector_rotation") or row_or_signal.get("Sector Rotation") or "").strip()
    session = str(row_or_signal.get("Session") or row_or_signal.get("session_bucket") or "").strip()

    if catalyst_window.startswith("Blocking") or "BLOCK" in catalyst_state:
        return "Catalyst Block"
    if trade_gate in {"No-Trade", "Defensive Only"}:
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
    d["Playbook"] = d.get("market_playbook", pd.Series(dtype=object)).replace("", "Unknown").fillna("Unknown")
    d["Trade Gate"] = d.get("market_trade_gate", pd.Series(dtype=object)).replace("", "Unknown").fillna("Unknown")
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
    return d


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
    weakest = [f"{lens}: {value}" for _, lens, value in hits[:2]]
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
