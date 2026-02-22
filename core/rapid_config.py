from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RapidConfig:
    score_confidence_weight: float = 0.30
    score_setup_weight: float = 0.20
    score_ai_weight: float = 0.20
    score_trend_weight: float = 0.15
    score_execution_weight: float = 0.15

    score_penalty_conflict: float = -18.0
    score_penalty_low_conviction: float = -8.0

    action_ready_min_confidence: float = 60.0
    action_ready_min_score: float = 76.0
    action_wait_min_score: float = 64.0

    trend_adx_weak: float = 18.0
    trend_adx_starting: float = 25.0
    trend_adx_strong: float = 40.0

    default_universe: int = 20
    default_min_score: int = 68
    history_max_items: int = 50


DEFAULT_RAPID_CONFIG = RapidConfig()
