"""Shared signal text/label formatters used across trader-facing tabs."""

from __future__ import annotations

import math

from core.ai_spot_bias import ai_spot_bias_display_votes
from core.confidence import ai_confidence_bucket, confidence_bucket
from core.market_decision import normalize_action_class


def spot_bias_label(direction: str) -> str:
    raw = str(direction or "").strip().upper()
    if raw == "UPSIDE":
        return "Upside"
    if raw == "DOWNSIDE":
        return "Downside"
    return "Neutral"


def spot_confidence_display(score: float) -> str:
    score_f = max(0.0, min(100.0, float(score)))
    return f"{score_f:.0f}% ({confidence_bucket(score_f).title()})"


def ai_confidence_display(snapshot, score: float) -> str:
    score_f = max(0.0, min(100.0, float(score)))
    label = ai_confidence_bucket(
        score_f,
        direction=str(snapshot.direction or ""),
        support_votes=int(ai_spot_bias_display_votes(snapshot)),
        timeframe_conflict=bool(snapshot.timeframe_conflict),
        degraded_data=bool(snapshot.degraded_data),
    )
    return f"{score_f:.0f}% ({label.title()})"


def ai_spot_tf_note(snapshot) -> str:
    status = str(getattr(snapshot, "status", "") or "").strip()
    note = str(getattr(snapshot, "note", "") or "").strip()
    suffix_parts = []
    if status:
        suffix_parts.append(f"Status {status}")
    if note:
        suffix_parts.append(note)
    suffix = f" | {' | '.join(suffix_parts)}" if suffix_parts else ""
    return (
        f"{str(snapshot.timeframe).upper()}: {spot_bias_label(snapshot.direction)} | "
        f"Score {float(snapshot.score):.1f} | "
        f"Prob Up {float(snapshot.probability_up) * 100:.0f}% | "
        f"Directional agreement {float(snapshot.directional_agreement) * 100:.0f}% | "
        f"Consensus {float(snapshot.consensus_agreement) * 100:.0f}%{suffix}"
    )


def ai_spot_note(snapshot) -> str:
    dots = ai_spot_bias_display_votes(snapshot)
    return (
        f"AI spot bias (1D + 4H): {spot_bias_label(snapshot.direction)} | "
        f"Combined score {float(snapshot.score):.1f} | "
        f"Conviction quality {float(snapshot.conviction_quality):.0f} | "
        f"Timeframe alignment {float(snapshot.timeframe_alignment):.0f} | "
        f"Displayed model-support dots {dots}/3 | "
        f"{str(snapshot.note or '').strip()} | "
        f"{ai_spot_tf_note(snapshot.one_day)} | "
        f"{ai_spot_tf_note(snapshot.four_hour)}"
    )


def ai_confidence_note(snapshot, score: float) -> str:
    dots = ai_spot_bias_display_votes(snapshot)
    caps: list[str] = []
    direction_key = str(snapshot.direction or "").strip().upper()
    if direction_key == "NEUTRAL":
        caps.append("neutral-verdict cap <=58")
    if bool(snapshot.timeframe_conflict):
        caps.append("timeframe-conflict cap <=30")
    if bool(snapshot.degraded_data):
        caps.append("degraded-data cap <=35")
    if direction_key != "NEUTRAL" and int(dots) <= 1:
        caps.append("low-model-support cap <=59")
    cap_text = f" | Active caps: {', '.join(caps)}" if caps else ""
    return (
        f"AI confidence: {float(score):.1f}% "
        f"({ai_confidence_bucket(float(score), direction=str(snapshot.direction or ''), support_votes=int(dots), timeframe_conflict=bool(snapshot.timeframe_conflict), degraded_data=bool(snapshot.degraded_data)).title()}) | "
        f"HTF AI verdict {spot_bias_label(snapshot.direction)} | "
        f"Combined score {float(snapshot.score):.1f} | "
        f"Conviction quality {float(snapshot.conviction_quality):.0f} | "
        f"Timeframe alignment {float(snapshot.timeframe_alignment):.0f} | "
        f"Consensus quality {float(snapshot.consensus_quality):.0f} | "
        f"Model support {int(dots)}/3{cap_text}"
    )


def adx_bucket_only(adx_value: float) -> str:
    try:
        adx_f = float(adx_value)
    except Exception:
        return ""
    if not math.isfinite(adx_f):
        return ""
    if adx_f < 20:
        return "Weak"
    if adx_f < 25:
        return "Starting"
    if adx_f < 50:
        return "Strong"
    if adx_f < 75:
        return "Very Strong"
    return "Extreme"


def setup_confirm_display(raw_action: str) -> str:
    cls = normalize_action_class(raw_action)
    if cls == "ENTER_TREND_AI":
        return "TREND+AI"
    if cls == "ENTER_TREND_LED":
        return "TREND-led"
    if cls == "ENTER_AI_LED":
        return "AI-led"
    if cls == "WATCH":
        return "WATCH"
    if cls == "SKIP":
        return "SKIP"
    return str(raw_action or "").strip() or "SKIP"

