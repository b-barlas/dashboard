from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd

from core.rapid_config import DEFAULT_RAPID_CONFIG, RapidConfig


def setup_badge(scalp_dir: str, signal_dir: str, ai_dir: str) -> str:
    if scalp_dir and signal_dir in {"LONG", "SHORT"} and signal_dir == ai_dir == scalp_dir:
        return "Aligned"
    if scalp_dir and signal_dir in {"LONG", "SHORT"} and signal_dir == scalp_dir and ai_dir == "NEUTRAL":
        return "Tech-Only"
    if scalp_dir:
        return "Draft"
    return "No Setup"


def grade_from_score(score: float) -> str:
    if score >= 82:
        return "A+"
    if score >= 74:
        return "A"
    if score >= 66:
        return "B"
    return "C"


def compute_rapid_score(
    *,
    signal_dir: str,
    strength: float,
    setup: str,
    conviction_label: str,
    ai_dir: str,
    agreement: float,
    adx: float,
    rr: float,
    has_plan: bool,
    cfg: RapidConfig = DEFAULT_RAPID_CONFIG,
) -> float:
    if signal_dir not in {"LONG", "SHORT"}:
        return 0.0
    if not has_plan:
        return 5.0

    strength_score = max(0.0, min(100.0, float(strength)))
    setup_score = {"Aligned": 100.0, "Tech-Only": 72.0, "Draft": 45.0}.get(setup, 0.0)

    ai_align = 100.0 if ai_dir == signal_dir else (60.0 if ai_dir == "NEUTRAL" else 15.0)
    ai_score = 0.55 * ai_align + 0.45 * max(0.0, min(100.0, agreement * 100.0))

    if pd.isna(adx):
        trend_score = 55.0
    else:
        adx_f = float(adx)
        trend_score = float(
            np.interp(
                adx_f,
                [0.0, cfg.trend_adx_weak, cfg.trend_adx_starting, cfg.trend_adx_strong, 60.0],
                [30.0, 45.0, 65.0, 85.0, 95.0],
            )
        )

    rr_f = max(0.0, float(rr))
    rr_score = float(np.interp(min(rr_f, 3.0), [0.0, 1.0, 1.2, 1.5, 2.0, 3.0], [0.0, 42.0, 62.0, 80.0, 92.0, 100.0]))

    conv_penalty = 0.0
    if conviction_label == "CONFLICT":
        conv_penalty = cfg.score_penalty_conflict
    elif conviction_label in {"LOW", "TECH-ONLY"}:
        conv_penalty = cfg.score_penalty_low_conviction

    # Extra penalty when AI is explicitly opposite and highly certain (3/3-like agreement).
    if ai_dir not in {signal_dir, "NEUTRAL"} and float(agreement) >= 0.67:
        conv_penalty += -6.0

    score = (
        cfg.score_confidence_weight * strength_score
        + cfg.score_setup_weight * setup_score
        + cfg.score_ai_weight * ai_score
        + cfg.score_trend_weight * trend_score
        + cfg.score_execution_weight * rr_score
        + conv_penalty
    )
    return max(0.0, min(100.0, score))


def decide_action(
    *,
    signal_dir: str,
    strength: float,
    setup: str,
    conviction_label: str,
    ai_dir: str,
    score: float,
    has_plan: bool,
    cfg: RapidConfig = DEFAULT_RAPID_CONFIG,
) -> str:
    if signal_dir not in {"LONG", "SHORT"}:
        return "SKIP"
    if not has_plan:
        return "WAIT"
    if float(strength) < 25.0:
        return "SKIP"
    if setup == "No Setup" or conviction_label == "CONFLICT":
        return "SKIP"
    if ai_dir not in {signal_dir, "NEUTRAL"}:
        return "WAIT"
    if conviction_label in {"HIGH", "MEDIUM"} and strength >= cfg.action_ready_min_confidence and score >= cfg.action_ready_min_score:
        return "READY"
    if score >= cfg.action_wait_min_score:
        return "WAIT"
    return "SKIP"


def summarize_quality_history(rows: Iterable[dict]) -> dict[str, float]:
    items = list(rows)
    if not items:
        return {
            "scans": 0.0,
            "ready_rate": 0.0,
            "avg_best_score": 0.0,
            "avg_candidates": 0.0,
            "trend_share": 0.0,
        }

    scans = float(len(items))
    ready_rate = 100.0 * sum(1 for x in items if x.get("best_action") == "READY") / scans
    avg_best_score = sum(float(x.get("best_score", 0.0)) for x in items) / scans
    avg_candidates = sum(float(x.get("qualified_count", 0.0)) for x in items) / scans
    trend_share = 100.0 * sum(1 for x in items if float(x.get("strong_adx_share", 0.0)) >= 0.5) / scans
    return {
        "scans": scans,
        "ready_rate": ready_rate,
        "avg_best_score": avg_best_score,
        "avg_candidates": avg_candidates,
        "trend_share": trend_share,
    }
