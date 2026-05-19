"""Higher-timeframe AI bias engine for spot direction context."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TypedDict

import numpy as np
import pandas as pd

from core.ml import ml_ensemble_predict

MIN_AI_ROWS = 60
_TIMEFRAME_DIRECTION_THRESHOLD = 55.0
_FINAL_AI_BIAS_THRESHOLD = 52.0


class TraceSample(TypedDict):
    direction: str
    probability_up: float
    directional_agreement: float
    consensus_agreement: float
    probability_edge: float
    directional_quality: float
    consensus_quality: float
    agreement_quality: float
    signed_strength: float
    model_votes: tuple[str, str, str]
    status: str


@dataclass(frozen=True)
class TimeframeAIBiasSnapshot:
    timeframe: str
    direction: str
    score: float
    probability_up: float
    directional_agreement: float
    consensus_agreement: float
    conviction_quality: float
    consensus_quality: float
    degraded: bool = False
    status: str = ""
    note: str = ""
    model_votes: tuple[str, str, str] = ("NEUTRAL", "NEUTRAL", "NEUTRAL")


@dataclass(frozen=True)
class AISpotBiasSnapshot:
    direction: str
    score: float
    timeframe_alignment: float
    conviction_quality: float
    consensus_quality: float
    timeframe_conflict: bool
    degraded_data: bool
    note: str
    four_hour: TimeframeAIBiasSnapshot
    one_day: TimeframeAIBiasSnapshot
    support_votes: int = 0
    lead_timeframe: str = "1d"
    confirm_timeframe: str = "4h"

    @property
    def lead_snapshot(self) -> TimeframeAIBiasSnapshot:
        return self.one_day

    @property
    def confirm_snapshot(self) -> TimeframeAIBiasSnapshot:
        return self.four_hour

    @property
    def anchor_pair_label(self) -> str:
        return f"{self.lead_timeframe.upper()} + {self.confirm_timeframe.upper()}"


def _lead_snapshot(snapshot: AISpotBiasSnapshot | object):
    return getattr(snapshot, "lead_snapshot", getattr(snapshot, "one_day", None))


def _confirm_snapshot(snapshot: AISpotBiasSnapshot | object):
    return getattr(snapshot, "confirm_snapshot", getattr(snapshot, "four_hour", None))


def _empty_tf_snapshot(timeframe: str, *, note: str = "Higher-timeframe AI context is incomplete.") -> TimeframeAIBiasSnapshot:
    return TimeframeAIBiasSnapshot(
        timeframe=timeframe,
        direction="NEUTRAL",
        score=0.0,
        probability_up=0.5,
        directional_agreement=0.0,
        consensus_agreement=0.0,
        conviction_quality=0.0,
        consensus_quality=0.0,
        degraded=True,
        status="insufficient_context",
        note=note,
        model_votes=("NEUTRAL", "NEUTRAL", "NEUTRAL"),
    )


def _safe_frame(df: pd.DataFrame | None) -> pd.DataFrame | None:
    if df is None or len(df) < MIN_AI_ROWS:
        return None
    required = {"open", "high", "low", "close", "volume"}
    if not required.issubset(set(df.columns)):
        return None
    out = df.copy()
    for col in ("open", "high", "low", "close", "volume"):
        out[col] = pd.to_numeric(out[col], errors="coerce")
    out = out.dropna(subset=["open", "high", "low", "close", "volume"])
    if len(out) < MIN_AI_ROWS:
        return None
    return out.reset_index(drop=True)


def _dir_key(direction: str) -> str:
    raw = str(direction or "").strip().upper()
    if raw in {"UPSIDE", "LONG", "BUY", "BULLISH"}:
        return "UPSIDE"
    if raw in {"DOWNSIDE", "SHORT", "SELL", "BEARISH"}:
        return "DOWNSIDE"
    return "NEUTRAL"


def _clamp_100(value: float) -> float:
    return max(0.0, min(100.0, float(value)))


def _opposite_dir(direction: str) -> str:
    key = _dir_key(direction)
    if key == "UPSIDE":
        return "DOWNSIDE"
    if key == "DOWNSIDE":
        return "UPSIDE"
    return "NEUTRAL"


def _model_htf_verdict(lead_vote: str, confirm_vote: str) -> str:
    lead = _dir_key(lead_vote)
    confirm = _dir_key(confirm_vote)
    if lead == "NEUTRAL":
        return "NEUTRAL"
    if confirm == _opposite_dir(lead):
        return "NEUTRAL"
    return lead


def _ai_probability_edge_quality(ai_probability: float, ai_dir: str) -> float:
    if pd.isna(ai_probability):
        return 20.0
    direction = _dir_key(ai_dir)
    if direction == "NEUTRAL":
        return 20.0
    prob_up = max(0.0, min(1.0, float(ai_probability)))
    directional_prob = prob_up if direction == "UPSIDE" else (1.0 - prob_up)
    if directional_prob < 0.53:
        return 10.0
    if directional_prob < 0.55:
        return 25.0
    if directional_prob < 0.58:
        return 42.0
    if directional_prob < 0.62:
        return 60.0
    if directional_prob < 0.66:
        return 78.0
    if directional_prob < 0.72:
        return 88.0
    return 96.0


def _soft_direction_from_probability(ai_probability: float) -> str:
    if pd.isna(ai_probability):
        return "NEUTRAL"
    prob_up = max(0.0, min(1.0, float(ai_probability)))
    if prob_up >= 0.54:
        return "UPSIDE"
    if prob_up <= 0.46:
        return "DOWNSIDE"
    return "NEUTRAL"


def _ai_directional_agreement_quality(directional_agreement: float) -> float:
    agreement = max(0.0, min(1.0, float(directional_agreement)))
    if agreement < 0.34:
        return 10.0
    if agreement < 0.50:
        return 25.0
    if agreement < 0.67:
        return 45.0
    if agreement < 0.78:
        return 65.0
    if agreement < 1.0:
        return 84.0
    return 96.0


def _ai_consensus_quality(consensus_agreement: float) -> float:
    consensus = max(0.0, min(1.0, float(consensus_agreement)))
    if consensus < 0.34:
        return 15.0
    if consensus < 0.67:
        return 45.0
    if consensus < 1.0:
        return 72.0
    return 94.0


def _timeframe_alignment(lead: TimeframeAIBiasSnapshot, confirm: TimeframeAIBiasSnapshot) -> tuple[float, bool]:
    if lead.direction == "NEUTRAL":
        return 0.0, False
    if confirm.direction == lead.direction:
        return 100.0, False
    if confirm.direction == "NEUTRAL":
        return 70.0, False
    return 0.0, True


def _trace_offsets(timeframe: str) -> tuple[int, ...]:
    tf = str(timeframe or "").strip().lower()
    if tf == "1w":
        return (0, 1, 2)
    if tf == "1d":
        return (0, 1, 3)
    return (0, 1, 2, 4)


def _trace_weights(length: int) -> list[float]:
    base = [1.0, 0.85, 0.70, 0.55]
    if length <= len(base):
        return base[:length]
    weights = list(base)
    while len(weights) < length:
        weights.append(max(0.30, weights[-1] - 0.12))
    return weights


def _trace_sample(
    probability_up: float,
    raw_direction: str,
    details: dict,
) -> TraceSample:
    raw_dir = _dir_key(raw_direction)
    candidate_dir = raw_dir if raw_dir != "NEUTRAL" else _soft_direction_from_probability(float(probability_up))
    directional_agreement = float(details.get("directional_agreement", details.get("agreement", 0.0)) or 0.0)
    consensus_agreement = float(details.get("consensus_agreement", 0.0) or 0.0)
    raw_votes = list(details.get("model_votes", []) or [])
    normalized_votes = [_dir_key(vote) for vote in raw_votes[:3]]
    if len(normalized_votes) < 3:
        support_count = int(round(max(0.0, min(1.0, directional_agreement)) * 3.0))
        fallback_dir = candidate_dir if candidate_dir != "NEUTRAL" else _dir_key(details.get("consensus_label", "NEUTRAL"))
        normalized_votes.extend([fallback_dir] * support_count)
        normalized_votes.extend(["NEUTRAL"] * max(0, 3 - len(normalized_votes)))
        normalized_votes = normalized_votes[:3]
    probability_edge = _ai_probability_edge_quality(float(probability_up), candidate_dir)
    directional_quality = _ai_directional_agreement_quality(directional_agreement)
    consensus_quality = _ai_consensus_quality(consensus_agreement)
    agreement_quality = _clamp_100(0.70 * directional_quality + 0.30 * consensus_quality)
    if candidate_dir == "UPSIDE":
        signed_strength = +_clamp_100(0.70 * probability_edge + 0.30 * agreement_quality)
    elif candidate_dir == "DOWNSIDE":
        signed_strength = -_clamp_100(0.70 * probability_edge + 0.30 * agreement_quality)
    else:
        signed_strength = 0.0
    return {
        "direction": candidate_dir,
        "probability_up": float(probability_up),
        "directional_agreement": directional_agreement,
        "consensus_agreement": consensus_agreement,
        "probability_edge": probability_edge,
        "directional_quality": directional_quality,
        "consensus_quality": consensus_quality,
        "agreement_quality": agreement_quality,
        "signed_strength": signed_strength,
        "model_votes": (
            normalized_votes[0],
            normalized_votes[1],
            normalized_votes[2],
        ),
        "status": "",
    }


def _build_recent_trace(
    safe: pd.DataFrame,
    *,
    timeframe: str,
    predictor,
    trace_offsets: tuple[int, ...] | None = None,
) -> list[TraceSample]:
    trace: list[TraceSample] = []
    total_len = len(safe)
    offsets = tuple(trace_offsets) if trace_offsets is not None else _trace_offsets(timeframe)
    for offset in offsets:
        end_idx = total_len - int(offset)
        if end_idx < MIN_AI_ROWS:
            continue
        frame = safe.iloc[:end_idx].copy()
        try:
            probability_up, raw_direction, details = predictor(frame)
        except Exception:
            continue
        details = details if isinstance(details, dict) else {}
        sample = _trace_sample(probability_up, raw_direction, details)
        sample["status"] = str(details.get("status") or "").strip().lower()
        trace.append(sample)
    return trace


def _trace_model_verdict(trace: list[TraceSample], model_idx: int) -> str:
    if not trace:
        return "NEUTRAL"
    weights = _trace_weights(len(trace))
    signed_score = 0.0
    active_weight = 0.0
    directional_sequence: list[str] = []
    for weight, sample in zip(weights, trace):
        votes = sample["model_votes"]
        vote = _dir_key(votes[model_idx] if model_idx < len(votes) else "NEUTRAL")
        if vote == "UPSIDE":
            signed_score += weight
            active_weight += weight
            directional_sequence.append(vote)
        elif vote == "DOWNSIDE":
            signed_score -= weight
            active_weight += weight
            directional_sequence.append(vote)
    if active_weight <= 0.0 or abs(signed_score) <= 1e-9:
        return "NEUTRAL"

    dominant_dir = "UPSIDE" if signed_score > 0.0 else "DOWNSIDE"
    dominance_ratio = abs(signed_score) / active_weight
    aligned_weight = 0.0
    for weight, sample in zip(weights, trace):
        votes = sample["model_votes"]
        vote = _dir_key(votes[model_idx] if model_idx < len(votes) else "NEUTRAL")
        if vote == dominant_dir:
            aligned_weight += weight
    persistence_ratio = aligned_weight / active_weight if active_weight > 0.0 else 0.0
    flips = sum(
        1 for idx in range(1, len(directional_sequence)) if directional_sequence[idx] != directional_sequence[idx - 1]
    )

    if dominance_ratio < 0.30:
        return "NEUTRAL"
    if persistence_ratio < 0.55:
        return "NEUTRAL"
    if flips >= 2 and dominance_ratio < 0.70:
        return "NEUTRAL"
    return dominant_dir


def _trace_model_votes(trace: list[TraceSample]) -> tuple[str, str, str]:
    return (
        _trace_model_verdict(trace, 0),
        _trace_model_verdict(trace, 1),
        _trace_model_verdict(trace, 2),
    )


def _persistence_quality(trace: list[TraceSample], dominant_dir: str) -> float:
    if not trace or dominant_dir == "NEUTRAL":
        return 0.0
    weights = _trace_weights(len(trace))
    aligned = 0.0
    total = 0.0
    for weight, sample in zip(weights, trace):
        sample_dir = str(sample.get("direction") or "NEUTRAL")
        if sample_dir == "NEUTRAL":
            continue
        total += weight
        if sample_dir == dominant_dir:
            aligned += weight
    if total <= 0.0:
        return 0.0
    return _clamp_100((aligned / total) * 100.0)


def _stability_quality(trace: list[TraceSample], dominant_dir: str) -> float:
    dirs = [str(sample.get("direction") or "NEUTRAL") for sample in trace if str(sample.get("direction") or "NEUTRAL") != "NEUTRAL"]
    if len(dirs) <= 1:
        return 55.0 if dominant_dir == "NEUTRAL" else 80.0
    flips = sum(1 for idx in range(1, len(dirs)) if dirs[idx] != dirs[idx - 1])
    if flips == 0:
        return 95.0
    if flips == 1:
        return 72.0
    if flips == 2:
        return 48.0
    return 25.0


def _timeframe_ai_bias_from_trace(
    trace: list[TraceSample],
    *,
    timeframe: str,
    empty_note: str = "AI model could not produce enough recent closed-bar context.",
) -> TimeframeAIBiasSnapshot:
    if not trace:
        return _empty_tf_snapshot(timeframe, note=empty_note)
    latest = trace[0]
    trace_model_votes = _trace_model_votes(trace)
    status = str(latest.get("status") or "").strip().lower()
    if status:
        return TimeframeAIBiasSnapshot(
            timeframe=timeframe,
            direction="NEUTRAL",
            score=0.0,
            probability_up=float(latest.get("probability_up") or 0.5),
            directional_agreement=float(latest.get("directional_agreement") or 0.0),
            consensus_agreement=float(latest.get("consensus_agreement") or 0.0),
            conviction_quality=0.0,
            consensus_quality=float(latest.get("consensus_quality") or 0.0),
            degraded=True,
            status=status,
            note="AI timeframe context is incomplete; neutral safety output is shown.",
            model_votes=trace_model_votes,
        )

    weights = _trace_weights(len(trace))
    total_weight = float(sum(weights)) if weights else 1.0
    weighted_bias = sum(float(sample.get("signed_strength") or 0.0) * weight for sample, weight in zip(trace, weights)) / total_weight
    weighted_prob_up = sum(float(sample.get("probability_up") or 0.5) * weight for sample, weight in zip(trace, weights)) / total_weight
    weighted_directional_agreement = sum(float(sample.get("directional_agreement") or 0.0) * weight for sample, weight in zip(trace, weights)) / total_weight
    weighted_consensus_agreement = sum(
        float(sample.get("consensus_agreement") or 0.0) * weight for sample, weight in zip(trace, weights)
    ) / total_weight
    weighted_consensus_quality = sum(
        float(sample.get("consensus_quality") or 0.0) * weight for sample, weight in zip(trace, weights)
    ) / total_weight
    weighted_agreement_quality = sum(
        float(sample.get("agreement_quality") or 0.0) * weight for sample, weight in zip(trace, weights)
    ) / total_weight
    dominant_dir = _soft_direction_from_probability(weighted_prob_up)
    if dominant_dir == "NEUTRAL" and weighted_bias >= 18.0:
        dominant_dir = "UPSIDE"
    elif dominant_dir == "NEUTRAL" and weighted_bias <= -18.0:
        dominant_dir = "DOWNSIDE"
    persistence_quality = _persistence_quality(trace, dominant_dir)
    stability_quality = _stability_quality(trace, dominant_dir)
    conviction_quality = _clamp_100(
        0.35 * abs(weighted_bias)
        + 0.25 * weighted_agreement_quality
        + 0.25 * persistence_quality
        + 0.15 * stability_quality
    )

    if dominant_dir == "NEUTRAL":
        return TimeframeAIBiasSnapshot(
            timeframe=timeframe,
            direction="NEUTRAL",
            score=0.0,
            probability_up=float(weighted_prob_up),
            directional_agreement=float(weighted_directional_agreement),
            consensus_agreement=float(weighted_consensus_agreement),
            conviction_quality=conviction_quality,
            consensus_quality=float(weighted_consensus_quality),
            degraded=False,
            status="",
            note="Recent AI history is not directional enough on this timeframe.",
            model_votes=trace_model_votes,
        )

    if conviction_quality < _TIMEFRAME_DIRECTION_THRESHOLD:
        return TimeframeAIBiasSnapshot(
            timeframe=timeframe,
            direction="NEUTRAL",
            score=0.0,
            probability_up=float(weighted_prob_up),
            directional_agreement=float(weighted_directional_agreement),
            consensus_agreement=float(weighted_consensus_agreement),
            conviction_quality=conviction_quality,
            consensus_quality=float(weighted_consensus_quality),
            degraded=False,
            status="",
            note="AI persistence/stability is below the higher-timeframe directional threshold.",
            model_votes=trace_model_votes,
        )

    signed_score = conviction_quality if dominant_dir == "UPSIDE" else -conviction_quality
    return TimeframeAIBiasSnapshot(
        timeframe=timeframe,
        direction=dominant_dir,
        score=float(np.clip(signed_score, -100.0, 100.0)),
        probability_up=float(weighted_prob_up),
        directional_agreement=float(weighted_directional_agreement),
        consensus_agreement=float(weighted_consensus_agreement),
        conviction_quality=conviction_quality,
        consensus_quality=float(weighted_consensus_quality),
        degraded=False,
        status="",
        note="AI direction is supported by recent persistence and stability on this timeframe.",
        model_votes=trace_model_votes,
    )


def analyze_timeframe_ai_bias(
    df: pd.DataFrame | None,
    *,
    timeframe: str,
    predictor=ml_ensemble_predict,
    trace_offsets: tuple[int, ...] | None = None,
) -> TimeframeAIBiasSnapshot:
    safe = _safe_frame(df)
    if safe is None:
        return _empty_tf_snapshot(timeframe)

    trace = _build_recent_trace(
        safe,
        timeframe=timeframe,
        predictor=predictor,
        trace_offsets=trace_offsets,
    )
    return _timeframe_ai_bias_from_trace(
        trace,
        timeframe=timeframe,
        empty_note="AI model could not produce enough recent closed-bar context.",
    )


def _htf_support_votes(
    final_direction: str,
    lead: TimeframeAIBiasSnapshot,
    confirm: TimeframeAIBiasSnapshot,
) -> int:
    direction = _dir_key(final_direction)
    support = 0
    for idx in range(3):
        lead_vote = _dir_key(lead.model_votes[idx] if idx < len(lead.model_votes) else "NEUTRAL")
        confirm_vote = _dir_key(confirm.model_votes[idx] if idx < len(confirm.model_votes) else "NEUTRAL")
        model_verdict = _model_htf_verdict(lead_vote, confirm_vote)
        if model_verdict == direction:
            support += 1
    return max(0, min(3, support))


def _build_ai_spot_bias_from_timeframes(
    *,
    lead: TimeframeAIBiasSnapshot,
    confirm: TimeframeAIBiasSnapshot,
    lead_timeframe: str,
    confirm_timeframe: str,
) -> AISpotBiasSnapshot:
    degraded_data = bool(confirm.degraded or lead.degraded)
    timeframe_alignment, timeframe_conflict = _timeframe_alignment(lead, confirm)
    score = float(np.clip(0.60 * lead.score + 0.40 * confirm.score, -100.0, 100.0))
    conviction_quality = float(
        np.clip(0.60 * lead.conviction_quality + 0.40 * confirm.conviction_quality, 0.0, 100.0)
    )
    consensus_quality = float(
        np.clip(0.60 * lead.consensus_quality + 0.40 * confirm.consensus_quality, 0.0, 100.0)
    )

    if degraded_data:
        direction = "NEUTRAL"
        note = "Higher-timeframe AI context is incomplete."
    elif lead.direction == "NEUTRAL":
        direction = "NEUTRAL"
        note = f"{lead_timeframe.upper()} AI bias is not directional enough."
    elif timeframe_conflict:
        direction = "NEUTRAL"
        note = f"{confirm_timeframe.upper()} AI bias conflicts with the {lead_timeframe.upper()} AI bias."
    elif abs(score) < _FINAL_AI_BIAS_THRESHOLD:
        direction = "NEUTRAL"
        note = "Combined higher-timeframe AI score is too weak."
    else:
        direction = lead.direction
        note = (
            f"{lead_timeframe.upper()} AI bias leads and "
            f"{confirm_timeframe.upper()} AI does not oppose it."
        )
    support_votes = _htf_support_votes(direction, lead, confirm)

    return AISpotBiasSnapshot(
        direction=direction,
        score=score,
        timeframe_alignment=timeframe_alignment,
        conviction_quality=conviction_quality,
        consensus_quality=consensus_quality,
        timeframe_conflict=timeframe_conflict,
        degraded_data=degraded_data,
        note=note,
        four_hour=confirm,
        one_day=lead,
        support_votes=support_votes,
        lead_timeframe=lead_timeframe,
        confirm_timeframe=confirm_timeframe,
    )


def build_ai_spot_bias_snapshot(
    *,
    df_4h: pd.DataFrame | None,
    df_1d: pd.DataFrame | None,
    predictor=ml_ensemble_predict,
    lead_df: pd.DataFrame | None = None,
    confirm_df: pd.DataFrame | None = None,
    lead_timeframe: str = "1d",
    confirm_timeframe: str = "4h",
    trace_offsets: tuple[int, ...] | None = None,
) -> AISpotBiasSnapshot:
    if lead_df is None and confirm_df is None:
        lead_df = df_1d
        confirm_df = df_4h
        lead_timeframe = "1d"
        confirm_timeframe = "4h"

    same_anchor_frame = lead_df is confirm_df and str(lead_timeframe).lower() == str(confirm_timeframe).lower()
    confirm = analyze_timeframe_ai_bias(
        confirm_df,
        timeframe=confirm_timeframe,
        predictor=predictor,
        trace_offsets=trace_offsets,
    )
    lead = confirm if same_anchor_frame else analyze_timeframe_ai_bias(
        lead_df,
        timeframe=lead_timeframe,
        predictor=predictor,
        trace_offsets=trace_offsets,
    )

    return _build_ai_spot_bias_from_timeframes(
        lead=lead,
        confirm=confirm,
        lead_timeframe=lead_timeframe,
        confirm_timeframe=confirm_timeframe,
    )


def build_ai_spot_bias_snapshot_from_prediction(
    *,
    probability_up: float,
    raw_direction: str,
    details: dict | None,
    timeframe: str,
) -> AISpotBiasSnapshot:
    """Fast path for callers that already computed the latest AI prediction."""

    safe_details = details if isinstance(details, dict) else {}
    sample = _trace_sample(float(probability_up), str(raw_direction or ""), safe_details)
    sample["status"] = str(safe_details.get("status") or "").strip().lower()
    tf_snapshot = _timeframe_ai_bias_from_trace([sample], timeframe=timeframe)
    return _build_ai_spot_bias_from_timeframes(
        lead=tf_snapshot,
        confirm=tf_snapshot,
        lead_timeframe=timeframe,
        confirm_timeframe=timeframe,
    )


def ai_spot_bias_display_votes(snapshot: AISpotBiasSnapshot) -> int:
    """Return how many ensemble models support the final HTF AI bias."""

    if bool(snapshot.degraded_data):
        return 0
    return max(0, min(3, int(snapshot.support_votes)))


def ai_spot_bias_probability_up(snapshot: AISpotBiasSnapshot) -> float:
    lead = _lead_snapshot(snapshot)
    confirm = _confirm_snapshot(snapshot)
    return max(
        0.0,
        min(
            1.0,
            0.60 * float(getattr(lead, "probability_up", 0.5) or 0.5)
            + 0.40 * float(getattr(confirm, "probability_up", 0.5) or 0.5),
        ),
    )


def ai_spot_bias_directional_agreement(snapshot: AISpotBiasSnapshot) -> float:
    lead = _lead_snapshot(snapshot)
    confirm = _confirm_snapshot(snapshot)
    return max(
        0.0,
        min(
            1.0,
            0.60 * float(getattr(lead, "directional_agreement", 0.0) or 0.0)
            + 0.40 * float(getattr(confirm, "directional_agreement", 0.0) or 0.0),
        ),
    )


def ai_spot_bias_consensus_agreement(snapshot: AISpotBiasSnapshot) -> float:
    lead = _lead_snapshot(snapshot)
    confirm = _confirm_snapshot(snapshot)
    return max(
        0.0,
        min(
            1.0,
            0.60 * float(getattr(lead, "consensus_agreement", 0.0) or 0.0)
            + 0.40 * float(getattr(confirm, "consensus_agreement", 0.0) or 0.0),
        ),
    )


def ai_spot_bias_status(snapshot: AISpotBiasSnapshot) -> str:
    lead = _lead_snapshot(snapshot)
    confirm = _confirm_snapshot(snapshot)
    for status in (
        str(getattr(lead, "status", "") or "").strip(),
        str(getattr(confirm, "status", "") or "").strip(),
    ):
        if status:
            return status
    if bool(snapshot.degraded_data):
        return "insufficient_context"
    return ""
