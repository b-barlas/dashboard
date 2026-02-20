from __future__ import annotations

AI_LONG_THRESHOLD = 0.58
AI_SHORT_THRESHOLD = 0.42

AI_AGREE_STRONG = 0.75
AI_AGREE_MEDIUM = 0.60

CONF_STRONG = 80.0
CONF_GOOD = 65.0
CONF_MEDIUM = 50.0


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


def confidence_bucket(confidence: float) -> str:
    c = float(confidence)
    if c >= CONF_STRONG:
        return "Strong"
    if c >= CONF_GOOD:
        return "Good"
    if c >= CONF_MEDIUM:
        return "Mixed"
    return "Weak"
