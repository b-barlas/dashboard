"""Shared signal semantics for direction, bias and confidence."""

from __future__ import annotations

from dataclasses import dataclass


def clamp_100(value: float) -> float:
    return max(0.0, min(100.0, float(value)))


def direction_from_bias(bias: float, *, long_threshold: float = 60.0, short_threshold: float = 40.0) -> str:
    b = clamp_100(bias)
    if b >= long_threshold:
        return "LONG"
    if b <= short_threshold:
        return "SHORT"
    return "NEUTRAL"


def bias_confidence_from_bias(bias: float) -> float:
    """Convert directional bias (0-100) to direction-agnostic confidence (0-100).

    Uses a non-linear calibration so real-market mid-range bias values do not
    collapse into low-looking confidence bands. This keeps interpretation closer
    to trader expectations while preserving:
    - 50 bias -> 0 confidence
    - 0/100 bias -> 100 confidence
    """
    b = clamp_100(bias)
    raw = abs(b - 50.0) / 50.0  # 0..1
    gamma = 0.70
    return clamp_100((raw ** gamma) * 100.0)


@dataclass(frozen=True)
class SignalSnapshot:
    direction: str
    bias: float
    confidence: float
    setup: str = ""
    alignment: str = ""
    rr: float = 0.0
    action: str = ""
    trade_quality: str = ""
