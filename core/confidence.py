"""Shared confidence scoring helpers for spot direction."""

from __future__ import annotations

from dataclasses import dataclass

from core.signal_contract import bias_confidence_from_bias

CONFIDENCE_VERY_LOW_MAX = 39.0
CONFIDENCE_LOW_MAX = 59.0
CONFIDENCE_MEDIUM_MAX = 79.0

_WEIGHT_TIMEFRAME_ALIGNMENT = 0.30
_WEIGHT_STRUCTURE_QUALITY = 0.25
_WEIGHT_TREND_QUALITY = 0.20
_WEIGHT_REGIME_QUALITY = 0.15
_WEIGHT_LOCATION_QUALITY = 0.10

_AI_WEIGHT_CONVICTION = 0.40
_AI_WEIGHT_COMBINED_SCORE = 0.25
_AI_WEIGHT_TIMEFRAME_ALIGNMENT = 0.15
_AI_WEIGHT_CONSENSUS = 0.10
_AI_WEIGHT_MODEL_SUPPORT = 0.10

_TACTICAL_WEIGHT_BIAS = 0.40
_TACTICAL_WEIGHT_TREND = 0.25
_TACTICAL_WEIGHT_ALIGNMENT = 0.20
_TACTICAL_WEIGHT_STRUCTURE = 0.15


def clamp_100(value: float) -> float:
    return max(0.0, min(100.0, float(value)))


def normalize_direction(direction: str) -> str:
    raw = str(direction or "").strip().upper()
    if raw in {"UPSIDE", "LONG", "BUY", "BULLISH"}:
        return "UPSIDE"
    if raw in {"DOWNSIDE", "SHORT", "SELL", "BEARISH"}:
        return "DOWNSIDE"
    return "NEUTRAL"


def confidence_from_components(
    *,
    direction: str,
    timeframe_alignment: float,
    structure_quality: float,
    trend_quality: float,
    regime_quality: float,
    location_quality: float,
    timeframe_conflict: bool = False,
    degraded_data: bool = False,
    range_regime: bool = False,
) -> float:
    """Return a capped 0-100 confidence score for a direction call.

    Base formula:
    - 30% timeframe alignment
    - 25% structure quality
    - 20% trend quality
    - 15% regime quality
    - 10% location quality

    Hard caps intentionally keep confidence honest in structurally weak states.
    """

    tf_align = clamp_100(timeframe_alignment)
    structure = clamp_100(structure_quality)
    trend = clamp_100(trend_quality)
    regime = clamp_100(regime_quality)
    location = clamp_100(location_quality)

    score = (
        _WEIGHT_TIMEFRAME_ALIGNMENT * tf_align
        + _WEIGHT_STRUCTURE_QUALITY * structure
        + _WEIGHT_TREND_QUALITY * trend
        + _WEIGHT_REGIME_QUALITY * regime
        + _WEIGHT_LOCATION_QUALITY * location
    )

    cap = 100.0
    if normalize_direction(direction) == "NEUTRAL":
        cap = min(cap, 35.0)
    if timeframe_conflict:
        cap = min(cap, 15.0)
    if structure < 40.0:
        cap = min(cap, 45.0)
    if degraded_data:
        cap = min(cap, 55.0)
    if range_regime:
        cap = min(cap, 35.0)

    return clamp_100(min(score, cap))


def confidence_bucket(confidence: float) -> str:
    score = clamp_100(confidence)
    if score <= CONFIDENCE_VERY_LOW_MAX:
        return "VERY LOW"
    if score <= CONFIDENCE_LOW_MAX:
        return "LOW"
    if score <= CONFIDENCE_MEDIUM_MAX:
        return "MEDIUM"
    return "HIGH"


def ai_confidence_bucket(
    score: float,
    *,
    direction: str,
    support_votes: int,
    timeframe_conflict: bool = False,
    degraded_data: bool = False,
) -> str:
    score_f = clamp_100(score)
    direction_key = normalize_direction(direction)
    support = max(0, min(3, int(support_votes)))

    if degraded_data or timeframe_conflict:
        return "VERY LOW"

    if direction_key == "NEUTRAL":
        if score_f >= 52.0 and support >= 2:
            return "MEDIUM"
        if score_f >= 40.0:
            return "LOW"
        return "VERY LOW"

    if support >= 3 and score_f >= 85.0:
        return "HIGH"
    if support >= 2 and score_f >= 68.0:
        return "MEDIUM"
    if score_f >= 45.0:
        return "LOW"
    return "VERY LOW"


@dataclass(frozen=True)
class ConfidenceSnapshot:
    score: float
    label: str


def build_confidence_snapshot(**kwargs) -> ConfidenceSnapshot:
    score = confidence_from_components(**kwargs)
    return ConfidenceSnapshot(score=score, label=confidence_bucket(score))


def ai_confidence_from_components(
    *,
    direction: str,
    combined_score: float,
    conviction_quality: float,
    timeframe_alignment: float,
    consensus_quality: float,
    support_votes: int,
    timeframe_conflict: bool = False,
    degraded_data: bool = False,
) -> float:
    direction_key = normalize_direction(direction)
    score_quality = clamp_100(abs(float(combined_score)))
    conviction = clamp_100(conviction_quality)
    tf_align = clamp_100(timeframe_alignment)
    consensus = clamp_100(consensus_quality)
    support = max(0.0, min(3.0, float(support_votes))) / 3.0 * 100.0

    score = (
        _AI_WEIGHT_CONVICTION * conviction
        + _AI_WEIGHT_COMBINED_SCORE * score_quality
        + _AI_WEIGHT_TIMEFRAME_ALIGNMENT * tf_align
        + _AI_WEIGHT_CONSENSUS * consensus
        + _AI_WEIGHT_MODEL_SUPPORT * support
    )

    cap = 100.0
    if direction_key == "NEUTRAL":
        cap = min(cap, 58.0)
    if timeframe_conflict:
        cap = min(cap, 30.0)
    if degraded_data:
        cap = min(cap, 35.0)
    if direction_key != "NEUTRAL" and support <= (100.0 / 3.0):
        cap = min(cap, 59.0)

    return clamp_100(min(score, cap))


def build_ai_confidence_snapshot(**kwargs) -> ConfidenceSnapshot:
    score = ai_confidence_from_components(**kwargs)
    return ConfidenceSnapshot(
        score=score,
        label=ai_confidence_bucket(
            score,
            direction=str(kwargs.get("direction", "")),
            support_votes=int(kwargs.get("support_votes", 0) or 0),
            timeframe_conflict=bool(kwargs.get("timeframe_conflict", False)),
            degraded_data=bool(kwargs.get("degraded_data", False)),
        ),
    )


def execution_trend_quality(adx_val: float | None) -> float:
    try:
        adx = float(adx_val)
    except Exception:
        return 35.0
    if adx != adx:
        return 35.0
    if adx < 12.0:
        return 10.0
    if adx < 18.0:
        return 25.0
    if adx < 25.0:
        return 55.0
    if adx < 35.0:
        return 75.0
    if adx < 50.0:
        return 88.0
    return 96.0


def execution_structure_quality(structure_state: str) -> float:
    state = str(structure_state or "").strip().upper()
    if state == "FULL":
        return 95.0
    if state == "TREND":
        return 78.0
    if state == "EARLY":
        return 60.0
    return 25.0


def execution_alignment_quality(conviction_label: str, ai_agreement: float = 0.0) -> float:
    label = str(conviction_label or "").strip().upper()
    agree = max(0.0, min(1.0, float(ai_agreement)))
    if label == "CONFLICT":
        return 0.0
    if label == "HIGH":
        return clamp_100(82.0 + agree * 18.0)
    if label == "MEDIUM":
        return clamp_100(62.0 + agree * 18.0)
    if label == "TREND":
        return clamp_100(55.0 + agree * 10.0)
    if label == "WEAK":
        return clamp_100(30.0 + agree * 10.0)
    return clamp_100(35.0 + agree * 10.0)


def execution_confidence_from_components(
    *,
    direction: str,
    bias_score: float,
    adx_val: float | None,
    structure_state: str,
    conviction_label: str,
    ai_agreement: float = 0.0,
) -> float:
    direction_key = normalize_direction(direction)
    bias_quality = clamp_100(bias_confidence_from_bias(float(bias_score)))
    trend_quality = execution_trend_quality(adx_val)
    alignment_quality = execution_alignment_quality(conviction_label, ai_agreement)
    structure_quality = execution_structure_quality(structure_state)

    score = (
        _TACTICAL_WEIGHT_BIAS * bias_quality
        + _TACTICAL_WEIGHT_TREND * trend_quality
        + _TACTICAL_WEIGHT_ALIGNMENT * alignment_quality
        + _TACTICAL_WEIGHT_STRUCTURE * structure_quality
    )

    cap = 100.0
    if direction_key == "NEUTRAL":
        cap = min(cap, 35.0)
    if str(conviction_label or "").strip().upper() == "CONFLICT":
        cap = min(cap, 15.0)
    if str(structure_state or "").strip().upper() == "NONE":
        cap = min(cap, 45.0)
    try:
        adx = float(adx_val)
    except Exception:
        adx = float("nan")
    if adx != adx:
        cap = min(cap, 55.0)
    elif adx < 12.0:
        cap = min(cap, 35.0)

    return clamp_100(min(score, cap))


def build_execution_confidence_snapshot(**kwargs) -> ConfidenceSnapshot:
    score = execution_confidence_from_components(**kwargs)
    return ConfidenceSnapshot(score=score, label=confidence_bucket(score))
