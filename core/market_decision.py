"""Market scanner decision policy helpers."""

from __future__ import annotations

from dataclasses import dataclass
from math import isnan

import numpy as np
import pandas as pd
import ta

from core.ai_spot_bias import AISpotBiasSnapshot
from core.signal_contract import bias_confidence_from_bias
from core.spot_direction import SpotDirectionSnapshot

ACTION_SKIP = "⛔ SKIP"
ACTION_WATCH = "WATCH"
ACTION_ENTER_TREND_AI = "✅ ENTER (Trend+AI)"
ACTION_ENTER_TREND_LED = "🟡 ENTER (Trend-Led)"
ACTION_ENTER_AI_LED = "🟡 ENTER (AI-Led)"

ACTION_REASON_TEXT = {
    "NO_DIRECTION": "No clear main direction yet.",
    "TECH_AI_CONFLICT": "Technical direction and AI direction are opposite.",
    "LOW_CONFIDENCE": "A direction exists, but confidence is still too low.",
    "NO_STRUCTURE": "The selected timeframe structure is too weak right now.",
    "POOR_LOCATION": "The direction may be right, but the current price area is not a clean entry zone.",
    "RR_TOO_LOW": "The possible reward is too small compared with the risk.",
    "RISK_UNDEFINED": "A clean stop/target plan is still missing.",
    "ADX_UNKNOWN": "Trend strength is unavailable, so it is safer to wait.",
    "ADX_TOO_LOW": "Trend strength is too weak right now.",
    "TACTICAL_NEUTRAL": "The main direction exists, but the selected timeframe has not aligned yet.",
    "TREND_SCORE_TOO_LOW": "The trend setup is forming, but it is not strong enough yet.",
    "AI_NEUTRAL": "AI does not see a clear edge yet.",
    "AI_EDGE_WEAK": "AI has a hint, but the edge is not strong enough yet.",
    "DUAL_NOT_ELITE": "Trend and AI both help, but the combined setup is not strong enough yet.",
    "TREND_SPOT_CONFLICT": "The short-term trend is fighting the main direction.",
    "AI_SPOT_CONFLICT": "AI is fighting the main direction.",
    "SPOT_CONFIRMATION_CONFLICT": "The short-term confirmation layers disagree with the main direction.",
    "ENTER_TREND_AI": "Trend and AI align, and both pass the highest confirmation bar.",
    "ENTER_TREND_LED": "The main direction is supported by a clean technical continuation setup.",
    "ENTER_AI_LED": "AI strongly supports the main direction, and execution conditions are good enough.",
    "NEEDS_CONFIRMATION": "The idea is alive, but it still needs more confirmation.",
    "AI_UNAVAILABLE": "AI is temporarily unavailable, so this stays on watch.",
}

_HARD_SKIP_REASON_ORDER = (
    "NO_DIRECTION",
    "LOW_CONFIDENCE",
    "SPOT_CONFIRMATION_CONFLICT",
    "NO_STRUCTURE",
    "RR_TOO_LOW",
)

_WATCH_REASON_ORDER = (
    "AI_UNAVAILABLE",
    "POOR_LOCATION",
    "TACTICAL_NEUTRAL",
    "AI_NEUTRAL",
    "AI_EDGE_WEAK",
    "TREND_SCORE_TOO_LOW",
    "DUAL_NOT_ELITE",
    "TREND_SPOT_CONFLICT",
    "AI_SPOT_CONFLICT",
    "ADX_TOO_LOW",
    "ADX_UNKNOWN",
    "RISK_UNDEFINED",
    "NEEDS_CONFIRMATION",
)


@dataclass(frozen=True)
class TrendLedConfirmationSnapshot:
    state: str
    score: float
    structure_continuation: float
    location_quality: float
    trend_integrity: float
    regime_quality: float
    risk_quality: float
    reason_code: str


@dataclass(frozen=True)
class SelectedTimeframeExecutionSnapshot:
    structure_quality: float
    trend_quality: float
    regime_quality: float
    location_quality: float
    support: float
    resistance: float
    close: float
    atr: float
    ema21: float


@dataclass(frozen=True)
class AILedConfirmationSnapshot:
    state: str
    score: float
    ai_conviction: float
    probability_edge: float
    consensus_quality: float
    location_quality: float
    risk_quality: float
    reason_code: str


@dataclass(frozen=True)
class TrendAIConfirmationSnapshot:
    state: str
    score: float
    trend_score: float
    ai_score: float
    spot_confidence: float
    reason_code: str


@dataclass(frozen=True)
class EmergingBiasSnapshot:
    active: bool
    direction: str
    label: str
    note: str


def ai_vote_metrics(
    ai_dir: str,
    directional_agreement: float,
    consensus_agreement: float,
) -> tuple[int, float, float]:
    """Return (display_votes, display_ratio, decision_agreement).

    - display_ratio/votes: for UI display. If AI direction is neutral, use consensus agreement.
    - decision_agreement: for decision engine. Always directional agreement (market/spot parity).
    """
    ai_key = _dir_key(ai_dir)
    dir_agree = max(0.0, min(1.0, float(directional_agreement)))
    cons_agree = max(0.0, min(1.0, float(consensus_agreement)))
    display_ratio = dir_agree if ai_key in {"UPSIDE", "DOWNSIDE"} else cons_agree
    display_votes = max(0, min(3, int(round(display_ratio * 3.0))))
    return display_votes, display_ratio, dir_agree


def normalize_action_class(action: str) -> str:
    """Normalize action text to a stable class key.

    Returns one of: ENTER_TREND_AI, ENTER_TREND_LED, ENTER_AI_LED, WATCH, SKIP, UNKNOWN.
    """
    s = str(action or "").strip().upper()
    if not s:
        return "UNKNOWN"
    if "ENTER (TREND+AI)" in s:
        return "ENTER_TREND_AI"
    if "ENTER (TREND-LED)" in s:
        return "ENTER_TREND_LED"
    if "ENTER (AI-LED)" in s:
        return "ENTER_AI_LED"
    if "WATCH" in s:
        return "WATCH"
    if "SKIP" in s:
        return "SKIP"
    return "UNKNOWN"


def action_rank(action: str) -> int:
    cls = normalize_action_class(action)
    if cls.startswith("ENTER_"):
        return 3
    if cls == "WATCH":
        return 2
    if cls == "SKIP":
        return 1
    return 0


def compact_action_label(action: str) -> str:
    """Compact, UI-safe action label for tables/kpi."""
    cls = normalize_action_class(action)
    if cls == "WATCH":
        return "WATCH"
    if cls == "ENTER_TREND_AI":
        return "ENTER T+AI"
    if cls == "ENTER_TREND_LED":
        return "ENTER Trend"
    if cls == "ENTER_AI_LED":
        return "ENTER AI"
    if cls == "SKIP":
        return "SKIP"
    return str(action or "").strip()


def action_reason_text(code: str) -> str:
    return ACTION_REASON_TEXT.get(str(code or "").upper(), "")


def _resolve_watch_skip_outcome(*reason_codes: str) -> tuple[str, str]:
    normalized = [
        str(code or "").strip().upper()
        for code in reason_codes
        if str(code or "").strip()
    ]
    if not normalized:
        return ACTION_WATCH, "NEEDS_CONFIRMATION"
    for code in _HARD_SKIP_REASON_ORDER:
        if code in normalized:
            return ACTION_SKIP, code
    for code in _WATCH_REASON_ORDER:
        if code in normalized:
            return ACTION_WATCH, code
    return ACTION_WATCH, normalized[0]


def _dir_key(value: str) -> str:
    s = str(value or "").strip().upper()
    if s in {"UPSIDE", "LONG", "BUY", "BULLISH"}:
        return "UPSIDE"
    if s in {"DOWNSIDE", "SHORT", "SELL", "BEARISH"}:
        return "DOWNSIDE"
    return "NEUTRAL"


def _clamp_100(value: float) -> float:
    return max(0.0, min(100.0, float(value)))


def emerging_bias_snapshot(
    *,
    spot_snapshot: SpotDirectionSnapshot,
    ai_spot_snapshot: AISpotBiasSnapshot,
    ai_confidence_score: float,
) -> EmergingBiasSnapshot:
    spot_direction = _dir_key(getattr(spot_snapshot, "direction", ""))
    if spot_direction != "NEUTRAL":
        return EmergingBiasSnapshot(
            active=False,
            direction="NEUTRAL",
            label="",
            note="Confirmed HTF direction already exists, so no emerging leader badge is needed.",
        )

    if bool(getattr(spot_snapshot, "degraded_data", False)):
        return EmergingBiasSnapshot(
            active=False,
            direction="NEUTRAL",
            label="",
            note="Higher-timeframe technical context is incomplete.",
        )

    if bool(getattr(ai_spot_snapshot, "degraded_data", False)):
        return EmergingBiasSnapshot(
            active=False,
            direction="NEUTRAL",
            label="",
            note="Higher-timeframe AI context is incomplete.",
        )

    one_day = getattr(spot_snapshot, "one_day", None)
    four_hour = getattr(spot_snapshot, "four_hour", None)
    if one_day is None or four_hour is None:
        return EmergingBiasSnapshot(
            active=False,
            direction="NEUTRAL",
            label="",
            note="Higher-timeframe technical structure is unavailable.",
        )

    daily_structure = str(getattr(one_day, "structure_label", "") or "").strip().upper()
    four_hour_structure = str(getattr(four_hour, "structure_label", "") or "").strip().upper()
    daily_raw = float(getattr(one_day, "raw_score", 0.0) or 0.0)
    daily_trend = float(getattr(one_day, "trend_score", 0.0) or 0.0)
    four_hour_score = float(getattr(four_hour, "score", 0.0) or 0.0)
    four_hour_raw = float(getattr(four_hour, "raw_score", 0.0) or 0.0)
    four_hour_trend = float(getattr(four_hour, "trend_score", 0.0) or 0.0)

    ai_direction = _dir_key(getattr(ai_spot_snapshot, "direction", ""))
    ai_support_votes = int(max(0, min(3, int(getattr(ai_spot_snapshot, "support_votes", 0) or 0))))
    ai_confidence = _clamp_100(ai_confidence_score)

    bullish_four_hour = (
        four_hour_structure in {"BREAKOUT_UP", "EARLY_UP", "HH/HL"}
        and four_hour_score >= 14.0
        and four_hour_raw >= 12.0
        and four_hour_trend >= 8.0
    )
    bearish_four_hour = (
        four_hour_structure in {"BREAKOUT_DOWN", "EARLY_DOWN", "LH/LL"}
        and four_hour_score <= -14.0
        and four_hour_raw <= -12.0
        and four_hour_trend <= -8.0
    )

    daily_guard_up = (
        daily_structure not in {"BREAKOUT_DOWN", "EARLY_DOWN", "LH/LL"}
        and daily_raw > -10.0
        and daily_trend > -8.0
    )
    daily_guard_down = (
        daily_structure not in {"BREAKOUT_UP", "EARLY_UP", "HH/HL"}
        and daily_raw < 10.0
        and daily_trend < 8.0
    )

    ai_confirms_up = ai_direction == "UPSIDE" and ai_confidence >= 68.0 and ai_support_votes >= 2
    ai_confirms_down = ai_direction == "DOWNSIDE" and ai_confidence >= 68.0 and ai_support_votes >= 2

    if bullish_four_hour and daily_guard_up and ai_confirms_up:
        return EmergingBiasSnapshot(
            active=True,
            direction="UPSIDE",
            label="Emerging Upside",
            note=(
                "4H technical structure is leading higher, AI confirms upside, "
                "and 1D is not opposing the move yet."
            ),
        )

    if bearish_four_hour and daily_guard_down and ai_confirms_down:
        return EmergingBiasSnapshot(
            active=True,
            direction="DOWNSIDE",
            label="Emerging Downside",
            note=(
                "4H technical structure is leading lower, AI confirms downside, "
                "and 1D is not opposing the move yet."
            ),
        )

    return EmergingBiasSnapshot(
        active=False,
        direction="NEUTRAL",
        label="",
        note="No clean emerging leader signal is present yet.",
    )


def _infer_timeframe(df: pd.DataFrame | None) -> str | None:
    if df is None or "timestamp" not in df.columns or len(df) < 2:
        return None
    try:
        delta_mins = (df["timestamp"].iloc[-1] - df["timestamp"].iloc[-2]).total_seconds() / 60.0
    except Exception:
        return None
    if delta_mins <= 1.5:
        return "1m"
    if delta_mins <= 4.0:
        return "3m"
    if delta_mins <= 7.0:
        return "5m"
    if delta_mins <= 20.0:
        return "15m"
    if delta_mins <= 90.0:
        return "1h"
    if delta_mins <= 300.0:
        return "4h"
    return "1d"


def _execution_sr_lookback(timeframe: str | None) -> int:
    mapping = {"1m": 60, "3m": 50, "5m": 50, "15m": 40, "1h": 30, "4h": 20, "1d": 20}
    return mapping.get(str(timeframe or "").lower(), 30)


def _trend_indicator_direction(label: str | None) -> str:
    s = str(label or "").strip().upper()
    if not s:
        return "NEUTRAL"
    if "BULL" in s or "ABOVE" in s:
        return "UPSIDE"
    if "BEAR" in s or "BELOW" in s:
        return "DOWNSIDE"
    return "NEUTRAL"


def _location_label_score(direction: str, label: str | None, *, kind: str) -> float:
    dir_key = _dir_key(direction)
    text = str(label or "").strip().upper()
    if not text or dir_key == "NEUTRAL":
        return 50.0

    if kind == "BOLLINGER":
        if dir_key == "UPSIDE":
            if "OVERBOUGHT" in text:
                return 15.0
            if "NEAR TOP" in text:
                return 30.0
            if "NEAR BOTTOM" in text:
                return 78.0
            return 55.0
        if "OVERSOLD" in text:
            return 15.0
        if "NEAR BOTTOM" in text:
            return 30.0
        if "NEAR TOP" in text:
            return 78.0
        return 55.0

    if dir_key == "UPSIDE":
        if "OVERBOUGHT" in text:
            return 20.0
        if "OVERSOLD" in text:
            return 72.0
        return 52.0

    if "OVERSOLD" in text:
        return 20.0
    if "OVERBOUGHT" in text:
        return 72.0
    return 52.0


def _execution_indicator_alignment(direction: str, *labels: str | None) -> float:
    dir_key = _dir_key(direction)
    if dir_key == "NEUTRAL":
        return 25.0

    supportive = 0
    opposing = 0
    neutral = 0
    for label in labels:
        label_dir = _trend_indicator_direction(label)
        if label_dir == dir_key:
            supportive += 1
        elif label_dir == "NEUTRAL":
            neutral += 1
        else:
            opposing += 1

    score = 25.0 + 22.5 * supportive + 5.0 * neutral - 20.0 * opposing
    return _clamp_100(score)


def _execution_regime_quality(df: pd.DataFrame | None, adx_val: float) -> float:
    adx_quality = _trend_strength_quality(adx_val)
    if df is None or len(df) < 20 or "close" not in df.columns:
        return adx_quality

    close = pd.to_numeric(df["close"], errors="coerce").dropna()
    if len(close) < 20:
        return adx_quality

    recent = close.iloc[-30:] if len(close) >= 30 else close
    rets = recent.pct_change().dropna()
    if rets.empty:
        return adx_quality

    drift = abs(float(recent.iloc[-1] / recent.iloc[0] - 1.0))
    noise = float(rets.std()) * np.sqrt(len(rets))
    trend_ratio = drift / (noise + 1e-9)

    if trend_ratio >= 1.2:
        drift_quality = 92.0
    elif trend_ratio >= 0.9:
        drift_quality = 78.0
    elif trend_ratio >= 0.7:
        drift_quality = 62.0
    elif trend_ratio >= 0.5:
        drift_quality = 45.0
    else:
        drift_quality = 28.0

    return _clamp_100(0.60 * adx_quality + 0.40 * drift_quality)


def selected_timeframe_execution_snapshot(
    *,
    df: pd.DataFrame | None,
    direction: str,
    bias_score: float,
    adx_val: float,
    supertrend_trend: str | None = None,
    ichimoku_trend: str | None = None,
    vwap_label: str | None = None,
    psar_trend: str | None = None,
    bollinger_bias: str | None = None,
    williams_label: str | None = None,
    cci_label: str | None = None,
) -> SelectedTimeframeExecutionSnapshot:
    """Build selected-timeframe execution quality for Setup Confirm.

    This intentionally ignores higher-timeframe structure and focuses on
    whether the currently selected timeframe is offering a clean continuation
    path in the already-defined spot direction.
    """

    dir_key = _dir_key(direction)
    bias_quality = _clamp_100(bias_confidence_from_bias(float(bias_score)))
    indicator_alignment = _execution_indicator_alignment(
        dir_key,
        supertrend_trend,
        ichimoku_trend,
        vwap_label,
        psar_trend,
    )

    structure_quality = _clamp_100(0.45 * bias_quality + 0.55 * indicator_alignment)

    safe_df = None
    if df is not None and len(df) >= 25 and {"high", "low", "close"}.issubset(set(df.columns)):
        safe_df = df.copy()
        for col in ("high", "low", "close"):
            safe_df[col] = pd.to_numeric(safe_df[col], errors="coerce")
        safe_df = safe_df.dropna(subset=["high", "low", "close"]).reset_index(drop=True)
        if len(safe_df) < 25:
            safe_df = None

    close = 0.0
    atr = float("nan")
    ema21 = float("nan")
    support = 0.0
    resistance = 0.0
    price_vs_ema_quality = 45.0
    ema_slope_quality = 45.0

    if safe_df is not None:
        close = float(safe_df["close"].iloc[-1])
        ema21_series = safe_df["close"].ewm(span=21, adjust=False).mean()
        ema21 = float(ema21_series.iloc[-1]) if pd.notna(ema21_series.iloc[-1]) else float("nan")
        try:
            atr_series = ta.volatility.average_true_range(
                safe_df["high"],
                safe_df["low"],
                safe_df["close"],
                window=14,
            )
            atr = float(atr_series.iloc[-1]) if pd.notna(atr_series.iloc[-1]) else float("nan")
        except Exception:
            atr = float("nan")

        if dir_key == "UPSIDE":
            price_vs_ema_quality = 90.0 if np.isfinite(ema21) and close >= ema21 else 25.0
        elif dir_key == "DOWNSIDE":
            price_vs_ema_quality = 90.0 if np.isfinite(ema21) and close <= ema21 else 25.0

        slope_window = min(5, len(ema21_series) - 1)
        if slope_window >= 1:
            ema21_prev = float(ema21_series.iloc[-1 - slope_window])
            if np.isfinite(ema21) and np.isfinite(ema21_prev) and abs(ema21_prev) > 1e-12:
                slope = (ema21 / ema21_prev) - 1.0
                if dir_key == "UPSIDE":
                    ema_slope_quality = 88.0 if slope > 0.0 else (50.0 if slope > -0.0005 else 20.0)
                elif dir_key == "DOWNSIDE":
                    ema_slope_quality = 88.0 if slope < 0.0 else (50.0 if slope < 0.0005 else 20.0)

        lookback = _execution_sr_lookback(_infer_timeframe(safe_df))
        recent = safe_df.tail(min(len(safe_df), lookback))
        support = float(recent["low"].min())
        resistance = float(recent["high"].max())

    trend_quality = _clamp_100(
        0.40 * _trend_strength_quality(adx_val)
        + 0.35 * price_vs_ema_quality
        + 0.25 * ema_slope_quality
    )
    regime_quality = _execution_regime_quality(safe_df, adx_val)

    location_score = 55.0
    if safe_df is not None and close > 0.0:
        room_to_resistance = max(0.0, (resistance - close) / close * 100.0)
        room_to_support = max(0.0, (close - support) / close * 100.0)

        if dir_key == "UPSIDE":
            if room_to_resistance >= 4.0:
                location_score += 20.0
            elif room_to_resistance >= 2.0:
                location_score += 8.0
            elif room_to_resistance <= 1.0:
                location_score -= 25.0
            elif room_to_resistance <= 2.0:
                location_score -= 12.0

            if room_to_support <= 3.0:
                location_score += 12.0
            elif room_to_support >= 8.0:
                location_score -= 8.0
        elif dir_key == "DOWNSIDE":
            if room_to_support >= 4.0:
                location_score += 20.0
            elif room_to_support >= 2.0:
                location_score += 8.0
            elif room_to_support <= 1.0:
                location_score -= 25.0
            elif room_to_support <= 2.0:
                location_score -= 12.0

            if room_to_resistance <= 3.0:
                location_score += 12.0
            elif room_to_resistance >= 8.0:
                location_score -= 8.0

        if np.isfinite(atr) and atr > 0.0 and np.isfinite(ema21):
            extension_atr = abs(close - ema21) / atr
            if extension_atr > 1.8:
                location_score -= 22.0
            elif extension_atr > 1.2:
                location_score -= 10.0

    location_score = _clamp_100(
        0.60 * location_score
        + 0.20 * _location_label_score(dir_key, bollinger_bias, kind="BOLLINGER")
        + 0.10 * _location_label_score(dir_key, williams_label, kind="OSCILLATOR")
        + 0.10 * _location_label_score(dir_key, cci_label, kind="OSCILLATOR")
    )

    return SelectedTimeframeExecutionSnapshot(
        structure_quality=structure_quality,
        trend_quality=trend_quality,
        regime_quality=regime_quality,
        location_quality=location_score,
        support=support,
        resistance=resistance,
        close=close,
        atr=atr,
        ema21=ema21,
    )


def selected_timeframe_rr_ratio(
    snapshot: SelectedTimeframeExecutionSnapshot,
    *,
    direction: str,
) -> float:
    """Estimate spot-style reward/risk from local selected-timeframe structure.

    This is intentionally separate from the scalp planner. It uses the local
    support/resistance envelope already derived for execution quality and asks:
    "Is there still enough room in the main direction before the next local
    barrier, relative to nearby invalidation distance?"
    """

    dir_key = _dir_key(direction)
    try:
        close = float(snapshot.close)
        support = float(snapshot.support)
        resistance = float(snapshot.resistance)
        atr = float(snapshot.atr)
        ema21 = float(snapshot.ema21)
    except Exception:
        return float("nan")

    if dir_key == "NEUTRAL" or not np.isfinite(close) or close <= 0.0:
        return float("nan")
    if not np.isfinite(support) or not np.isfinite(resistance):
        return float("nan")

    if dir_key == "UPSIDE":
        reward = resistance - close
        structure_risk = close - support
        ema_risk = (close - ema21) * 0.40 if np.isfinite(ema21) and close >= ema21 else float("nan")
    else:
        reward = close - support
        structure_risk = resistance - close
        ema_risk = (ema21 - close) * 0.40 if np.isfinite(ema21) and close <= ema21 else float("nan")

    candidate_risks = [float(value) for value in (structure_risk, ema_risk) if np.isfinite(value) and float(value) > 0.0]
    if not np.isfinite(reward) or not candidate_risks:
        return float("nan")

    reward = max(0.0, float(reward))
    risk = min(candidate_risks)
    atr_floor = float(atr) * 0.35 if np.isfinite(atr) and atr > 0.0 else 0.0
    price_floor = close * 0.002
    min_risk = max(atr_floor, price_floor, 1e-9)
    if np.isfinite(atr) and atr > 0.0:
        reward += float(atr) * 1.50
    min_reward = max((float(atr) * 0.35 if np.isfinite(atr) and atr > 0.0 else 0.0), close * 0.001, 1e-9)

    if reward < min_reward:
        return 0.0

    return float(np.clip(reward / max(risk, min_risk), 0.0, 10.0))


def _trend_strength_quality(adx_val: float) -> float:
    if isnan(adx_val):
        return 35.0
    if adx_val < 12.0:
        return 10.0
    if adx_val < 18.0:
        return 25.0
    if adx_val < 25.0:
        return 55.0
    if adx_val < 35.0:
        return 75.0
    if adx_val < 50.0:
        return 88.0
    return 96.0


def _risk_reward_quality(rr_ratio: float) -> float:
    if isnan(rr_ratio):
        return 35.0
    if rr_ratio < 1.30:
        return 15.0
    if rr_ratio < 1.50:
        return 30.0
    if rr_ratio < 1.70:
        return 45.0
    if rr_ratio < 2.00:
        return 65.0
    if rr_ratio < 2.40:
        return 80.0
    return 92.0


def _ai_probability_edge_quality(ai_probability: float, ai_dir: str) -> float:
    if isnan(ai_probability):
        return 20.0
    direction = _dir_key(ai_dir)
    if direction == "NEUTRAL":
        return 20.0
    prob_up = max(0.0, min(1.0, float(ai_probability)))
    directional_prob = prob_up if direction == "UPSIDE" else (1.0 - prob_up)
    if directional_prob < 0.55:
        return 10.0
    if directional_prob < 0.58:
        return 25.0
    if directional_prob < 0.61:
        return 42.0
    if directional_prob < 0.65:
        return 60.0
    if directional_prob < 0.70:
        return 78.0
    if directional_prob < 0.76:
        return 88.0
    return 96.0


def _ai_directional_agreement_quality(directional_agreement: float) -> float:
    a = max(0.0, min(1.0, float(directional_agreement)))
    if a < 0.34:
        return 10.0
    if a < 0.50:
        return 25.0
    if a < 0.67:
        return 45.0
    if a < 0.78:
        return 65.0
    if a < 1.0:
        return 84.0
    return 96.0


def _ai_consensus_quality(consensus_agreement: float) -> float:
    c = max(0.0, min(1.0, float(consensus_agreement)))
    if c < 0.34:
        return 15.0
    if c < 0.67:
        return 45.0
    if c < 1.0:
        return 72.0
    return 94.0


def trend_led_confirmation_snapshot(
    *,
    spot_dir: str,
    spot_confidence: float,
    tactical_dir: str,
    adx_val: float,
    structure_quality: float,
    trend_quality: float,
    regime_quality: float,
    location_quality: float,
    rr_ratio: float | None = None,
) -> TrendLedConfirmationSnapshot:
    """Pure technical continuation confirmation for Trend-led entries.

    AI is intentionally excluded. The snapshot answers:
    "Is the current higher-timeframe direction technically ready for a
    continuation-style entry on the selected timeframe?"
    """

    spot = _dir_key(spot_dir)
    tactical = _dir_key(tactical_dir)
    spot_conf = float(spot_confidence)
    adx = float(adx_val)
    structure = _clamp_100(structure_quality)
    location = _clamp_100(location_quality)
    regime = _clamp_100(regime_quality)
    trend_integrity = _clamp_100(0.55 * float(trend_quality) + 0.45 * _trend_strength_quality(adx))
    rr = float(rr_ratio) if rr_ratio is not None else float("nan")
    risk_quality = _risk_reward_quality(rr)
    score = _clamp_100(
        0.35 * structure
        + 0.25 * location
        + 0.20 * trend_integrity
        + 0.10 * regime
        + 0.10 * risk_quality
    )

    if spot == "NEUTRAL":
        return TrendLedConfirmationSnapshot(
            state="SKIP",
            score=score,
            structure_continuation=structure,
            location_quality=location,
            trend_integrity=trend_integrity,
            regime_quality=regime,
            risk_quality=risk_quality,
            reason_code="NO_DIRECTION",
        )
    if spot_conf < 70.0:
        return TrendLedConfirmationSnapshot(
            state="SKIP",
            score=score,
            structure_continuation=structure,
            location_quality=location,
            trend_integrity=trend_integrity,
            regime_quality=regime,
            risk_quality=risk_quality,
            reason_code="LOW_CONFIDENCE",
        )
    if tactical != spot:
        return TrendLedConfirmationSnapshot(
            state="WATCH" if tactical == "NEUTRAL" else "SKIP",
            score=score,
            structure_continuation=structure,
            location_quality=location,
            trend_integrity=trend_integrity,
            regime_quality=regime,
            risk_quality=risk_quality,
            reason_code="TACTICAL_NEUTRAL" if tactical == "NEUTRAL" else "TREND_SPOT_CONFLICT",
        )
    if isnan(adx):
        return TrendLedConfirmationSnapshot(
            state="WATCH",
            score=score,
            structure_continuation=structure,
            location_quality=location,
            trend_integrity=trend_integrity,
            regime_quality=regime,
            risk_quality=risk_quality,
            reason_code="ADX_UNKNOWN",
        )
    if adx < 18.0:
        return TrendLedConfirmationSnapshot(
            state="WATCH",
            score=score,
            structure_continuation=structure,
            location_quality=location,
            trend_integrity=trend_integrity,
            regime_quality=regime,
            risk_quality=risk_quality,
            reason_code="ADX_TOO_LOW",
        )
    if structure < 60.0:
        return TrendLedConfirmationSnapshot(
            state="SKIP",
            score=score,
            structure_continuation=structure,
            location_quality=location,
            trend_integrity=trend_integrity,
            regime_quality=regime,
            risk_quality=risk_quality,
            reason_code="NO_STRUCTURE",
        )
    if location < 55.0:
        return TrendLedConfirmationSnapshot(
            state="WATCH",
            score=score,
            structure_continuation=structure,
            location_quality=location,
            trend_integrity=trend_integrity,
            regime_quality=regime,
            risk_quality=risk_quality,
            reason_code="POOR_LOCATION",
        )
    if isnan(rr):
        return TrendLedConfirmationSnapshot(
            state="WATCH",
            score=score,
            structure_continuation=structure,
            location_quality=location,
            trend_integrity=trend_integrity,
            regime_quality=regime,
            risk_quality=risk_quality,
            reason_code="RISK_UNDEFINED",
        )
    if rr < 1.70:
        return TrendLedConfirmationSnapshot(
            state="SKIP",
            score=score,
            structure_continuation=structure,
            location_quality=location,
            trend_integrity=trend_integrity,
            regime_quality=regime,
            risk_quality=risk_quality,
            reason_code="RR_TOO_LOW",
        )
    if score >= 78.0:
        return TrendLedConfirmationSnapshot(
            state="READY",
            score=score,
            structure_continuation=structure,
            location_quality=location,
            trend_integrity=trend_integrity,
            regime_quality=regime,
            risk_quality=risk_quality,
            reason_code="ENTER_TREND_LED",
        )
    if score >= 62.0:
        return TrendLedConfirmationSnapshot(
            state="WATCH",
            score=score,
            structure_continuation=structure,
            location_quality=location,
            trend_integrity=trend_integrity,
            regime_quality=regime,
            risk_quality=risk_quality,
            reason_code="NEEDS_CONFIRMATION",
        )
    return TrendLedConfirmationSnapshot(
        state="WATCH",
        score=score,
        structure_continuation=structure,
        location_quality=location,
        trend_integrity=trend_integrity,
        regime_quality=regime,
        risk_quality=risk_quality,
        reason_code="TREND_SCORE_TOO_LOW",
    )


def ai_led_confirmation_snapshot(
    *,
    spot_dir: str,
    spot_confidence: float,
    ai_dir: str,
    ai_probability: float,
    directional_agreement: float,
    consensus_agreement: float,
    adx_val: float,
    location_quality: float,
    rr_ratio: float | None = None,
    ai_status: str | None = None,
) -> AILedConfirmationSnapshot:
    """Pure AI confirmation for AI-led entries.

    Trend/tactical direction is intentionally excluded. The snapshot answers:
    "Does the AI layer independently provide a high-quality directional edge
    for the current higher-timeframe spot bias, while the selected timeframe
    still offers acceptable execution conditions?"
    """

    spot = _dir_key(spot_dir)
    ai = _dir_key(ai_dir)
    spot_conf = float(spot_confidence)
    status = str(ai_status or "").strip().lower()
    adx = float(adx_val)
    location = _clamp_100(location_quality)
    rr = float(rr_ratio) if rr_ratio is not None else float("nan")
    risk_quality = _risk_reward_quality(rr)
    execution_quality = _trend_strength_quality(adx)
    probability_edge = _ai_probability_edge_quality(float(ai_probability), ai)
    directional_quality = _ai_directional_agreement_quality(directional_agreement)
    consensus_quality = _ai_consensus_quality(consensus_agreement)
    ai_conviction = _clamp_100(0.60 * directional_quality + 0.40 * probability_edge)
    score = _clamp_100(
        0.40 * ai_conviction
        + 0.20 * consensus_quality
        + 0.15 * probability_edge
        + 0.10 * execution_quality
        + 0.10 * location
        + 0.05 * risk_quality
    )

    if spot == "NEUTRAL":
        return AILedConfirmationSnapshot(
            state="SKIP",
            score=score,
            ai_conviction=ai_conviction,
            probability_edge=probability_edge,
            consensus_quality=consensus_quality,
            location_quality=location,
            risk_quality=risk_quality,
            reason_code="NO_DIRECTION",
        )
    if spot_conf < 65.0:
        return AILedConfirmationSnapshot(
            state="SKIP",
            score=score,
            ai_conviction=ai_conviction,
            probability_edge=probability_edge,
            consensus_quality=consensus_quality,
            location_quality=location,
            risk_quality=risk_quality,
            reason_code="LOW_CONFIDENCE",
        )
    if status:
        return AILedConfirmationSnapshot(
            state="WATCH",
            score=score,
            ai_conviction=ai_conviction,
            probability_edge=probability_edge,
            consensus_quality=consensus_quality,
            location_quality=location,
            risk_quality=risk_quality,
            reason_code="AI_UNAVAILABLE",
        )
    if isnan(adx):
        return AILedConfirmationSnapshot(
            state="WATCH",
            score=score,
            ai_conviction=ai_conviction,
            probability_edge=probability_edge,
            consensus_quality=consensus_quality,
            location_quality=location,
            risk_quality=risk_quality,
            reason_code="ADX_UNKNOWN",
        )
    if ai == "NEUTRAL":
        return AILedConfirmationSnapshot(
            state="WATCH",
            score=score,
            ai_conviction=ai_conviction,
            probability_edge=probability_edge,
            consensus_quality=consensus_quality,
            location_quality=location,
            risk_quality=risk_quality,
            reason_code="AI_NEUTRAL",
        )
    if adx < 18.0:
        return AILedConfirmationSnapshot(
            state="WATCH",
            score=score,
            ai_conviction=ai_conviction,
            probability_edge=probability_edge,
            consensus_quality=consensus_quality,
            location_quality=location,
            risk_quality=risk_quality,
            reason_code="ADX_TOO_LOW",
        )
    if ai != spot:
        return AILedConfirmationSnapshot(
            state="WATCH",
            score=score,
            ai_conviction=ai_conviction,
            probability_edge=probability_edge,
            consensus_quality=consensus_quality,
            location_quality=location,
            risk_quality=risk_quality,
            reason_code="AI_SPOT_CONFLICT",
        )
    if float(directional_agreement) < 0.67:
        return AILedConfirmationSnapshot(
            state="WATCH",
            score=score,
            ai_conviction=ai_conviction,
            probability_edge=probability_edge,
            consensus_quality=consensus_quality,
            location_quality=location,
            risk_quality=risk_quality,
            reason_code="AI_EDGE_WEAK",
        )
    if probability_edge < 60.0:
        return AILedConfirmationSnapshot(
            state="WATCH",
            score=score,
            ai_conviction=ai_conviction,
            probability_edge=probability_edge,
            consensus_quality=consensus_quality,
            location_quality=location,
            risk_quality=risk_quality,
            reason_code="AI_EDGE_WEAK",
        )
    if location < 55.0:
        return AILedConfirmationSnapshot(
            state="WATCH",
            score=score,
            ai_conviction=ai_conviction,
            probability_edge=probability_edge,
            consensus_quality=consensus_quality,
            location_quality=location,
            risk_quality=risk_quality,
            reason_code="POOR_LOCATION",
        )
    if isnan(rr):
        return AILedConfirmationSnapshot(
            state="WATCH",
            score=score,
            ai_conviction=ai_conviction,
            probability_edge=probability_edge,
            consensus_quality=consensus_quality,
            location_quality=location,
            risk_quality=risk_quality,
            reason_code="RISK_UNDEFINED",
        )
    if rr < 1.70:
        return AILedConfirmationSnapshot(
            state="SKIP",
            score=score,
            ai_conviction=ai_conviction,
            probability_edge=probability_edge,
            consensus_quality=consensus_quality,
            location_quality=location,
            risk_quality=risk_quality,
            reason_code="RR_TOO_LOW",
        )
    if score >= 80.0:
        return AILedConfirmationSnapshot(
            state="READY",
            score=score,
            ai_conviction=ai_conviction,
            probability_edge=probability_edge,
            consensus_quality=consensus_quality,
            location_quality=location,
            risk_quality=risk_quality,
            reason_code="ENTER_AI_LED",
        )
    if score >= 65.0:
        return AILedConfirmationSnapshot(
            state="WATCH",
            score=score,
            ai_conviction=ai_conviction,
            probability_edge=probability_edge,
            consensus_quality=consensus_quality,
            location_quality=location,
            risk_quality=risk_quality,
            reason_code="NEEDS_CONFIRMATION",
        )
    return AILedConfirmationSnapshot(
        state="WATCH",
        score=score,
        ai_conviction=ai_conviction,
        probability_edge=probability_edge,
        consensus_quality=consensus_quality,
        location_quality=location,
        risk_quality=risk_quality,
        reason_code="AI_EDGE_WEAK",
    )


def trend_ai_confirmation_snapshot(
    *,
    spot_confidence: float,
    trend_led_snapshot: TrendLedConfirmationSnapshot,
    ai_led_snapshot: AILedConfirmationSnapshot,
) -> TrendAIConfirmationSnapshot:
    """Elite dual-confirmation gate for Trend+AI.

    Both motors must already be independently READY. This helper then asks:
    "Are they strong enough together to deserve the strongest setup class?"
    """

    spot_conf = float(spot_confidence)
    trend_score = _clamp_100(float(trend_led_snapshot.score))
    ai_score = _clamp_100(float(ai_led_snapshot.score))
    score = _clamp_100(0.55 * trend_score + 0.45 * ai_score)

    if spot_conf < 75.0:
        return TrendAIConfirmationSnapshot(
            state="SKIP",
            score=score,
            trend_score=trend_score,
            ai_score=ai_score,
            spot_confidence=spot_conf,
            reason_code="LOW_CONFIDENCE",
        )
    if str(trend_led_snapshot.state).upper() != "READY":
        return TrendAIConfirmationSnapshot(
            state="WATCH",
            score=score,
            trend_score=trend_score,
            ai_score=ai_score,
            spot_confidence=spot_conf,
            reason_code=str(trend_led_snapshot.reason_code or "NEEDS_CONFIRMATION"),
        )
    if str(ai_led_snapshot.state).upper() != "READY":
        return TrendAIConfirmationSnapshot(
            state="WATCH",
            score=score,
            trend_score=trend_score,
            ai_score=ai_score,
            spot_confidence=spot_conf,
            reason_code=str(ai_led_snapshot.reason_code or "NEEDS_CONFIRMATION"),
        )
    if trend_score < 82.0 or ai_score < 84.0:
        return TrendAIConfirmationSnapshot(
            state="WATCH",
            score=score,
            trend_score=trend_score,
            ai_score=ai_score,
            spot_confidence=spot_conf,
            reason_code="DUAL_NOT_ELITE",
        )
    if float(trend_led_snapshot.structure_continuation) < 72.0:
        return TrendAIConfirmationSnapshot(
            state="WATCH",
            score=score,
            trend_score=trend_score,
            ai_score=ai_score,
            spot_confidence=spot_conf,
            reason_code="DUAL_NOT_ELITE",
        )
    if float(trend_led_snapshot.location_quality) < 65.0:
        return TrendAIConfirmationSnapshot(
            state="WATCH",
            score=score,
            trend_score=trend_score,
            ai_score=ai_score,
            spot_confidence=spot_conf,
            reason_code="POOR_LOCATION",
        )
    if float(ai_led_snapshot.ai_conviction) < 84.0 or float(ai_led_snapshot.consensus_quality) < 72.0:
        return TrendAIConfirmationSnapshot(
            state="WATCH",
            score=score,
            trend_score=trend_score,
            ai_score=ai_score,
            spot_confidence=spot_conf,
            reason_code="DUAL_NOT_ELITE",
        )
    if score >= 85.0:
        return TrendAIConfirmationSnapshot(
            state="READY",
            score=score,
            trend_score=trend_score,
            ai_score=ai_score,
            spot_confidence=spot_conf,
            reason_code="ENTER_TREND_AI",
        )
    return TrendAIConfirmationSnapshot(
        state="WATCH",
        score=score,
        trend_score=trend_score,
        ai_score=ai_score,
        spot_confidence=spot_conf,
        reason_code="DUAL_NOT_ELITE",
    )


def spot_structure_state(
    spot_dir: str,
    tactical_dir: str,
    ai_dir: str,
    confidence: float,
    agreement: float,
) -> str:
    """Classify spot setup structure around the higher-timeframe direction.

    - FULL: higher-timeframe spot direction is confirmed by both tactical trend and AI
    - TREND: tactical trend confirms; AI does not veto
    - EARLY: AI strongly confirms; tactical trend does not veto
    - NONE: no actionable confirmation
    """
    spot = _dir_key(spot_dir)
    tactical = _dir_key(tactical_dir)
    ai = _dir_key(ai_dir)
    conf = float(confidence)
    agree = max(0.0, min(1.0, float(agreement)))

    if spot == "NEUTRAL" or conf < 45.0:
        return "NONE"

    trend_confirms = tactical == spot
    trend_opposes = tactical not in {"NEUTRAL", spot}
    ai_confirms = ai == spot and agree >= 0.67
    ai_exceptional = ai == spot and agree >= 0.78
    ai_opposes = ai not in {"NEUTRAL", spot} and agree >= 0.67

    if trend_confirms and ai_confirms and conf >= 75.0:
        return "FULL"
    if trend_confirms and not ai_opposes and conf >= 65.0:
        return "TREND"
    if ai_exceptional and not trend_opposes and conf >= 65.0:
        return "EARLY"
    if conf >= 55.0 and ((trend_confirms and not ai_opposes) or (ai_confirms and not trend_opposes)):
        return "EARLY"
    return "NONE"


def structure_state(
    signal_dir: str,
    ai_dir: str,
    confidence: float,
    agreement: float,
) -> str:
    """Classify market structure quality without depending on scalp planning.

    This state is intentionally direction/confirmation-centric:
    - FULL: strong direction + AI alignment
    - TREND: tradable trend context, but not strongest alignment
    - EARLY: early structure, requires tighter risk controls
    - NONE: no actionable structure
    """
    sdir = _dir_key(signal_dir)
    ad = _dir_key(ai_dir)
    if sdir == "NEUTRAL":
        return "NONE"

    conf = float(confidence)
    a = float(agreement)

    if ad != "NEUTRAL" and ad != sdir:
        return "NONE"

    if ad == sdir:
        if conf >= 60 and a >= 0.67:
            return "FULL"
        if conf >= 55:
            return "TREND"
        return "EARLY"

    if ad == "NEUTRAL":
        if conf >= 70:
            return "TREND"
        if conf >= 55:
            return "EARLY"
        return "NONE"

    return "NONE"


def action_decision_with_reason(
    signal_dir: str,
    confidence: float,
    structure_state_val: str,
    conviction_label: str,
    agreement: float,
    adx_val: float,
) -> tuple[str, str]:
    """Return action class and compact reason code in one call."""
    return _action_decision_core(
        signal_dir,
        confidence,
        structure_state_val,
        conviction_label,
        agreement,
        adx_val,
    )


def spot_action_decision_with_reason(
    spot_dir: str,
    confidence: float,
    tactical_dir: str,
    ai_dir: str,
    agreement: float,
    adx_val: float,
    trend_led_snapshot: TrendLedConfirmationSnapshot | None = None,
    ai_led_snapshot: AILedConfirmationSnapshot | None = None,
) -> tuple[str, str]:
    """Spot-oriented setup decision.

    Uses higher-timeframe spot direction/confidence as the anchor.
    Trend-led and AI-led semantics are intentionally separated:
    - Trend-led: tactical trend confirms spot direction through a pure technical snapshot.
    - AI-led: AI confirms spot direction through a pure AI snapshot.
    """
    spot = _dir_key(spot_dir)
    tactical = _dir_key(tactical_dir)
    ai = _dir_key(ai_dir)
    conf = float(confidence)
    agree = max(0.0, min(1.0, float(agreement)))

    if spot == "NEUTRAL":
        return ACTION_SKIP, "NO_DIRECTION"
    if conf < 45.0:
        return ACTION_SKIP, "LOW_CONFIDENCE"
    trend_confirms = tactical == spot
    trend_opposes = tactical not in {"NEUTRAL", spot}
    ai_confirms = ai == spot and agree >= 0.67
    ai_exceptional = ai == spot and agree >= 0.78
    ai_opposes = ai not in {"NEUTRAL", spot} and agree >= 0.67
    if trend_led_snapshot is not None or ai_led_snapshot is not None:
        trend_ready = trend_led_snapshot is not None and str(trend_led_snapshot.state).upper() == "READY"
        trend_watch = trend_led_snapshot is not None and str(trend_led_snapshot.state).upper() == "WATCH"
        trend_skip = trend_led_snapshot is not None and str(trend_led_snapshot.state).upper() == "SKIP"
        trend_reason = str((trend_led_snapshot.reason_code if trend_led_snapshot is not None else "") or "NEEDS_CONFIRMATION")
        ai_ready = ai_led_snapshot is not None and str(ai_led_snapshot.state).upper() == "READY"
        ai_watch = ai_led_snapshot is not None and str(ai_led_snapshot.state).upper() == "WATCH"
        ai_skip = ai_led_snapshot is not None and str(ai_led_snapshot.state).upper() == "SKIP"
        ai_reason = str((ai_led_snapshot.reason_code if ai_led_snapshot is not None else "") or "NEEDS_CONFIRMATION")
        trend_ai_snapshot = (
            trend_ai_confirmation_snapshot(
                spot_confidence=conf,
                trend_led_snapshot=trend_led_snapshot,
                ai_led_snapshot=ai_led_snapshot,
            )
            if trend_led_snapshot is not None and ai_led_snapshot is not None
            else None
        )
        trend_ai_ready = trend_ai_snapshot is not None and str(trend_ai_snapshot.state).upper() == "READY"

        if trend_ai_ready:
            return ACTION_ENTER_TREND_AI, "ENTER_TREND_AI"
        if trend_ready:
            return ACTION_ENTER_TREND_LED, "ENTER_TREND_LED"
        if ai_ready:
            return ACTION_ENTER_AI_LED, "ENTER_AI_LED"
        if trend_opposes and ai_opposes:
            return ACTION_SKIP, "SPOT_CONFIRMATION_CONFLICT"
        reasons: list[str] = []
        if trend_ai_snapshot is not None and str(trend_ai_snapshot.state).upper() != "READY":
            reasons.append(str(trend_ai_snapshot.reason_code or "NEEDS_CONFIRMATION"))
        if trend_led_snapshot is not None and (trend_watch or trend_skip):
            reasons.append(trend_reason)
        if ai_led_snapshot is not None and (ai_watch or ai_skip):
            reasons.append(ai_reason)
        if trend_opposes:
            reasons.append("TREND_SPOT_CONFLICT")
        if ai_opposes:
            reasons.append("AI_SPOT_CONFLICT")
        return _resolve_watch_skip_outcome(*reasons)

    if isnan(adx_val):
        return ACTION_WATCH, "ADX_UNKNOWN"

    adx_f = float(adx_val)
    structure_state_val = spot_structure_state(spot, tactical, ai, conf, agree)

    if adx_f < 12.0:
        return ACTION_SKIP, "ADX_TOO_LOW"
    if trend_opposes and ai_opposes:
        return ACTION_SKIP, "SPOT_CONFIRMATION_CONFLICT"
    if trend_opposes:
        return ACTION_WATCH, "TREND_SPOT_CONFLICT"
    if ai_opposes:
        return ACTION_WATCH, "AI_SPOT_CONFLICT"

    if structure_state_val == "FULL" and adx_f >= 18.0:
        return ACTION_ENTER_TREND_AI, "ENTER_TREND_AI"
    if structure_state_val == "TREND" and adx_f >= 18.0:
        return ACTION_ENTER_TREND_LED, "ENTER_TREND_LED"
    if structure_state_val == "EARLY" and ai_exceptional and adx_f >= 18.0:
        return ACTION_ENTER_AI_LED, "ENTER_AI_LED"
    return ACTION_WATCH, "NEEDS_CONFIRMATION"


def _action_decision_core(
    signal_dir: str,
    confidence: float,
    structure_state_val: str,
    conviction_label: str,
    agreement: float,
    adx_val: float,
) -> tuple[str, str]:
    if _dir_key(signal_dir) == "NEUTRAL":
        return ACTION_SKIP, "NO_DIRECTION"
    if conviction_label == "CONFLICT" or confidence < 35:
        if conviction_label == "CONFLICT":
            return ACTION_SKIP, "TECH_AI_CONFLICT"
        return ACTION_SKIP, "LOW_CONFIDENCE"
    if structure_state_val == "NONE":
        return ACTION_SKIP, "NO_STRUCTURE"
    # Require known trend-strength context for ENTER. Unknown ADX can still be WATCH.
    if isnan(adx_val):
        return ACTION_WATCH, "ADX_UNKNOWN"

    adx_f = float(adx_val)
    # In very weak trend, skip only when confidence is also weak-mid.
    if adx_f < 12 and confidence < 70:
        return ACTION_SKIP, "ADX_TOO_LOW"

    # ENTER classes require non-weak trend context (ADX >= 20) so Action
    # semantics stay consistent with the ADX label model.
    # Primary class: both trend and AI confirmations are aligned.
    enter_trend_ai = (
        structure_state_val == "FULL"
        and adx_f >= 20
        and confidence >= 60
        and agreement >= 0.67
        and conviction_label in {"HIGH", "MEDIUM"}
    )

    # Leader + veto model:
    # - Trend-Led: trend drives the decision; AI only acts as guardrail/veto.
    # - AI-Led: AI drives the decision; trend only acts as guardrail/veto.
    enter_trend_led = (
        structure_state_val in {"FULL", "TREND", "EARLY"}
        and adx_f >= 20
        and confidence >= 56
        and agreement < 0.78
        and conviction_label != "CONFLICT"
    )
    enter_ai_led = (
        structure_state_val in {"EARLY", "TREND"}
        and adx_f >= 20
        and confidence >= 45
        and agreement >= 0.78
        and conviction_label != "CONFLICT"
        and not (adx_f < 14 and confidence < 55)
    )

    if enter_trend_ai:
        return ACTION_ENTER_TREND_AI, "ENTER_TREND_AI"
    if enter_trend_led:
        return ACTION_ENTER_TREND_LED, "ENTER_TREND_LED"
    if enter_ai_led:
        return ACTION_ENTER_AI_LED, "ENTER_AI_LED"
    return ACTION_WATCH, "NEEDS_CONFIRMATION"
