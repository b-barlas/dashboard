"""Core alignment metrics for the Multi-TF tab."""

from __future__ import annotations

from typing import Iterable

TF_SEQUENCE = ["5m", "15m", "1h", "4h", "1d"]
TF_WEIGHTS = {"5m": 1.0, "15m": 1.2, "1h": 1.6, "4h": 2.1, "1d": 2.6}
TACTICAL_TFS = {"5m", "15m"}
HIGHER_TFS = {"1h", "4h", "1d"}
VALID_DIRECTIONS = {"UPSIDE", "DOWNSIDE", "NEUTRAL"}
MIN_BIAS_ALIGNMENT_PCT = 60.0


def _alignment_read(pct: float) -> str:
    if pct >= 72:
        return "Tight"
    if pct >= 48:
        return "Mixed"
    return "Loose"


def _dominant(upside: float, downside: float, neutral: float) -> str:
    candidates = {
        "UPSIDE": float(upside),
        "DOWNSIDE": float(downside),
        "NEUTRAL": float(neutral),
    }
    best = max(candidates.values()) if candidates else 0.0
    winners = [key for key, value in candidates.items() if value == best and value > 0]
    if len(winners) != 1:
        return "NEUTRAL"
    return winners[0]


def _normalize_rows(rows: Iterable[dict], allowed_timeframes: set[str] | None = None) -> list[dict]:
    normalized: list[dict] = []
    for row in rows:
        timeframe = str(row.get("timeframe") or row.get("Timeframe") or "").strip()
        direction = str(row.get("direction") or row.get("Direction") or "").strip().upper()
        if direction not in VALID_DIRECTIONS:
            continue
        if allowed_timeframes is not None and timeframe not in allowed_timeframes:
            continue
        normalized.append(
            {
                "timeframe": timeframe,
                "direction": direction,
                "strength": float(row.get("strength", row.get("Strength", 0.0)) or 0.0),
                "weight": float(row.get("weight", row.get("Weight", TF_WEIGHTS.get(timeframe, 1.0))) or 0.0),
            }
        )
    return normalized


def _subset_metrics(rows: list[dict], allowed_timeframes: set[str] | None = None) -> dict:
    subset = _normalize_rows(rows, allowed_timeframes)
    count = len(subset)
    total_weight = sum(row["weight"] for row in subset)
    upside_count = sum(1 for row in subset if row["direction"] == "UPSIDE")
    downside_count = sum(1 for row in subset if row["direction"] == "DOWNSIDE")
    neutral_count = sum(1 for row in subset if row["direction"] == "NEUTRAL")
    upside_weight = sum(row["weight"] for row in subset if row["direction"] == "UPSIDE")
    downside_weight = sum(row["weight"] for row in subset if row["direction"] == "DOWNSIDE")
    neutral_weight = sum(row["weight"] for row in subset if row["direction"] == "NEUTRAL")
    raw_alignment_pct = (max(upside_count, downside_count) / count * 100.0) if count else 0.0
    weighted_alignment_pct = (max(upside_weight, downside_weight) / total_weight * 100.0) if total_weight else 0.0
    avg_strength = sum(row["strength"] for row in subset) / count if count else 0.0
    dominant_bias = _dominant(upside_weight, downside_weight, neutral_weight)
    # A directional bias should only print when the non-neutral side has broad enough
    # weighted agreement. Otherwise we keep the read neutral and let alignment % tell the story.
    if weighted_alignment_pct < MIN_BIAS_ALIGNMENT_PCT:
        dominant_bias = "NEUTRAL"
    return {
        "rows": subset,
        "count": count,
        "upside_count": upside_count,
        "downside_count": downside_count,
        "neutral_count": neutral_count,
        "upside_weight": upside_weight,
        "downside_weight": downside_weight,
        "neutral_weight": neutral_weight,
        "total_weight": total_weight,
        "raw_alignment_pct": raw_alignment_pct,
        "weighted_alignment_pct": weighted_alignment_pct,
        "dominant_bias": dominant_bias,
        "alignment_read": _alignment_read(weighted_alignment_pct),
        "avg_strength": avg_strength,
    }


def compute_multitf_alignment(rows: list[dict]) -> dict:
    total_slots = max(len(rows), len(TF_SEQUENCE))
    overall = _subset_metrics(rows)
    higher = _subset_metrics(rows, HIGHER_TFS)
    tactical = _subset_metrics(rows, TACTICAL_TFS)
    coverage_count = overall["count"]
    coverage_pct = coverage_count / total_slots * 100.0 if total_slots else 0.0
    if coverage_count >= 5:
        coverage_read = "Full"
    elif coverage_count >= 3:
        coverage_read = "Partial"
    else:
        coverage_read = "Thin"
    return {
        "coverage_count": coverage_count,
        "coverage_total": total_slots,
        "coverage_pct": coverage_pct,
        "coverage_read": coverage_read,
        "dominant_bias": overall["dominant_bias"],
        "weighted_alignment_pct": overall["weighted_alignment_pct"],
        "raw_alignment_pct": overall["raw_alignment_pct"],
        "alignment_read": overall["alignment_read"],
        "avg_strength": overall["avg_strength"],
        "neutral_count": overall["neutral_count"],
        "higher_tf_bias": higher["dominant_bias"],
        "higher_tf_alignment_pct": higher["weighted_alignment_pct"],
        "higher_tf_read": higher["alignment_read"],
        "tactical_bias": tactical["dominant_bias"],
        "tactical_alignment_pct": tactical["weighted_alignment_pct"],
        "tactical_read": tactical["alignment_read"],
        "higher_tf_count": higher["count"],
        "tactical_count": tactical["count"],
    }
