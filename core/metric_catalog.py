from __future__ import annotations

AI_LONG_THRESHOLD = 0.58
AI_SHORT_THRESHOLD = 0.42

AI_AGREE_STRONG = 0.75
AI_AGREE_MEDIUM = 0.60

BIAS_STRONG = 80.0
BIAS_GOOD = 65.0
BIAS_MIXED = 50.0


def direction_from_prob(prob: float) -> str:
    p = float(prob)
    if p >= AI_LONG_THRESHOLD:
        return "LONG"
    if p <= AI_SHORT_THRESHOLD:
        return "SHORT"
    return "NEUTRAL"


def ai_stability_bucket(agreement_ratio: float) -> str:
    a = float(agreement_ratio)
    if a >= AI_AGREE_STRONG:
        return "Strong"
    if a >= AI_AGREE_MEDIUM:
        return "Medium"
    return "Weak"


def bias_bucket(bias_score: float) -> str:
    c = float(bias_score)
    if c >= BIAS_STRONG:
        return "Strong"
    if c >= BIAS_GOOD:
        return "Good"
    if c >= BIAS_MIXED:
        return "Mixed"
    return "Weak"
