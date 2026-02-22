"""Shared signal semantics for direction, bias and strength."""

from __future__ import annotations

from dataclasses import dataclass

STRENGTH_STRONG = 75.0
STRENGTH_GOOD = 60.0
STRENGTH_MIXED = 40.0


def clamp_100(value: float) -> float:
    return max(0.0, min(100.0, float(value)))


def direction_from_bias(bias: float, *, long_threshold: float = 60.0, short_threshold: float = 40.0) -> str:
    b = clamp_100(bias)
    if b >= long_threshold:
        return "LONG"
    if b <= short_threshold:
        return "SHORT"
    return "NEUTRAL"


def strength_from_bias(bias: float) -> float:
    """Convert directional bias (0-100) to direction-agnostic signal strength (0-100).

    Uses a non-linear calibration so real-market mid-range bias values do not
    collapse into low-looking strength bands. This keeps interpretation closer
    to trader expectations while preserving:
    - 50 bias -> 0 strength
    - 0/100 bias -> 100 strength
    """
    b = clamp_100(bias)
    raw = abs(b - 50.0) / 50.0  # 0..1
    gamma = 0.70
    return clamp_100((raw ** gamma) * 100.0)


def strength_bucket(strength: float) -> str:
    s = clamp_100(strength)
    if s >= STRENGTH_STRONG:
        return "STRONG"
    if s >= STRENGTH_GOOD:
        return "GOOD"
    if s >= STRENGTH_MIXED:
        return "MIXED"
    return "WEAK"


@dataclass(frozen=True)
class SignalSnapshot:
    direction: str
    bias: float
    strength: float
    setup: str = ""
    alignment: str = ""
    rr: float = 0.0
    action: str = ""
    trade_quality: str = ""
