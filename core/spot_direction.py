"""Higher-timeframe spot direction engine."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import ta

from core.confidence import normalize_direction

MIN_DIRECTION_ROWS = 80
_TIMEFRAME_SCORE_THRESHOLD = 12.0
_FINAL_DIRECTION_SCORE_THRESHOLD = 20.0
_EARLY_LEAD_SCORE_THRESHOLD = 8.0
_EARLY_FINAL_DIRECTION_SCORE_THRESHOLD = 12.0


@dataclass(frozen=True)
class TimeframeDirectionSnapshot:
    timeframe: str
    direction: str
    score: float
    raw_score: float
    structure_score: float
    structure_label: str
    trend_score: float
    momentum_score: float
    regime_quality: float
    regime_label: str
    location_quality: float
    support: float
    resistance: float
    close: float
    degraded: bool = False


@dataclass(frozen=True)
class SpotDirectionSnapshot:
    direction: str
    score: float
    timeframe_alignment: float
    structure_quality: float
    trend_quality: float
    regime_quality: float
    location_quality: float
    timeframe_conflict: bool
    degraded_data: bool
    range_regime: bool
    note: str
    four_hour: TimeframeDirectionSnapshot
    one_day: TimeframeDirectionSnapshot
    lead_timeframe: str = "1d"
    confirm_timeframe: str = "4h"

    @property
    def lead_snapshot(self) -> TimeframeDirectionSnapshot:
        return self.one_day

    @property
    def confirm_snapshot(self) -> TimeframeDirectionSnapshot:
        return self.four_hour

    @property
    def anchor_pair_label(self) -> str:
        return f"{self.lead_timeframe.upper()} + {self.confirm_timeframe.upper()}"


def _empty_tf_snapshot(timeframe: str) -> TimeframeDirectionSnapshot:
    return TimeframeDirectionSnapshot(
        timeframe=timeframe,
        direction="NEUTRAL",
        score=0.0,
        raw_score=0.0,
        structure_score=0.0,
        structure_label="UNAVAILABLE",
        trend_score=0.0,
        momentum_score=0.0,
        regime_quality=0.0,
        regime_label="UNAVAILABLE",
        location_quality=0.0,
        support=0.0,
        resistance=0.0,
        close=0.0,
        degraded=True,
    )


def _safe_frame(df: pd.DataFrame | None) -> pd.DataFrame | None:
    if df is None or len(df) < MIN_DIRECTION_ROWS:
        return None
    required = {"open", "high", "low", "close"}
    if not required.issubset(set(df.columns)):
        return None
    out = df.copy()
    for col in ("open", "high", "low", "close"):
        out[col] = pd.to_numeric(out[col], errors="coerce")
    out = out.dropna(subset=["open", "high", "low", "close"])
    if len(out) < MIN_DIRECTION_ROWS:
        return None
    return out.reset_index(drop=True)


def _pivot_values(series: pd.Series, *, kind: str, window: int = 2) -> list[float]:
    values: list[float] = []
    if len(series) < (window * 2 + 1):
        return values
    for idx in range(window, len(series) - window):
        mid = float(series.iloc[idx])
        left = series.iloc[idx - window : idx]
        right = series.iloc[idx + 1 : idx + window + 1]
        if kind == "high":
            if mid > float(left.max()) and mid >= float(right.max()):
                values.append(mid)
        else:
            if mid < float(left.min()) and mid <= float(right.min()):
                values.append(mid)
    return values


def _structure_threshold(df: pd.DataFrame) -> float:
    try:
        atr = ta.volatility.average_true_range(df["high"], df["low"], df["close"], window=14)
        atr_val = float(atr.iloc[-1]) if pd.notna(atr.iloc[-1]) else 0.0
    except Exception:
        atr_val = 0.0
    close = float(df["close"].iloc[-1]) if len(df) > 0 else 0.0
    return max(atr_val * 0.5, close * 0.003, 1e-6)


def _breakout_structure_signal(df: pd.DataFrame, threshold: float) -> tuple[float, str] | None:
    lookback = min(len(df), 30)
    if lookback < 15:
        return None

    recent = df.tail(lookback).reset_index(drop=True)
    prior = recent.iloc[:-5]
    latest = recent.iloc[-5:]
    latest_close = float(recent["close"].iloc[-1])
    latest_high = float(latest["high"].max())
    latest_low = float(latest["low"].min())
    prior_high = float(prior["high"].max()) if not prior.empty else latest_high
    prior_low = float(prior["low"].min()) if not prior.empty else latest_low

    if latest_close > prior_high + threshold and latest_low >= prior_low:
        return 55.0, "BREAKOUT_UP"
    if latest_close < prior_low - threshold and latest_high <= prior_high:
        return -55.0, "BREAKOUT_DOWN"
    return None


def _structure_components(df: pd.DataFrame) -> tuple[float, str]:
    pivot_highs = _pivot_values(df["high"], kind="high")
    pivot_lows = _pivot_values(df["low"], kind="low")
    threshold = _structure_threshold(df)

    if len(pivot_highs) >= 2 and len(pivot_lows) >= 2:
        last_high, prev_high = float(pivot_highs[-1]), float(pivot_highs[-2])
        last_low, prev_low = float(pivot_lows[-1]), float(pivot_lows[-2])
        hh = last_high > (prev_high + threshold)
        hl = last_low > (prev_low + threshold)
        lh = last_high < (prev_high - threshold)
        ll = last_low < (prev_low - threshold)
        if hh and hl:
            return 100.0, "HH/HL"
        if lh and ll:
            return -100.0, "LH/LL"
        if hh or hl:
            return 45.0, "EARLY_UP"
        if lh or ll:
            return -45.0, "EARLY_DOWN"
        breakout = _breakout_structure_signal(df, threshold)
        if breakout is not None:
            return breakout
        return 0.0, "RANGE"

    recent = df.tail(min(len(df), 60)).reset_index(drop=True)
    split = max(10, len(recent) // 2)
    first = recent.iloc[:split]
    second = recent.iloc[split:]
    if len(first) < 5 or len(second) < 5:
        return 0.0, "RANGE"
    if float(second["high"].max()) > float(first["high"].max()) + threshold and float(second["low"].min()) > float(first["low"].min()) + threshold:
        return 45.0, "EARLY_UP"
    if float(second["high"].max()) < float(first["high"].max()) - threshold and float(second["low"].min()) < float(first["low"].min()) - threshold:
        return -45.0, "EARLY_DOWN"

    breakout = _breakout_structure_signal(df, threshold)
    if breakout is not None:
        return breakout
    return 0.0, "RANGE"


def _trend_components(df: pd.DataFrame) -> tuple[float, float, float, float]:
    close = df["close"]
    ema20 = close.ewm(span=20, adjust=False).mean()
    ema50 = close.ewm(span=50, adjust=False).mean()
    ema200 = close.ewm(span=200, adjust=False).mean()

    latest_close = float(close.iloc[-1])
    latest_ema20 = float(ema20.iloc[-1])
    latest_ema50 = float(ema50.iloc[-1])
    latest_ema200 = float(ema200.iloc[-1])

    score = 0.0
    score += 30.0 if latest_close > latest_ema200 else -30.0
    score += 25.0 if latest_ema50 > latest_ema200 else -25.0
    score += 20.0 if latest_ema20 > latest_ema50 else -20.0

    slope_window = min(5, len(df) - 1)
    ema50_prev = float(ema50.iloc[-1 - slope_window])
    ema200_prev = float(ema200.iloc[-1 - slope_window])
    ema50_slope = ((latest_ema50 / ema50_prev) - 1.0) if abs(ema50_prev) > 1e-12 else 0.0
    ema200_slope = ((latest_ema200 / ema200_prev) - 1.0) if abs(ema200_prev) > 1e-12 else 0.0
    if ema50_slope > 0.0:
        score += 15.0
    elif ema50_slope < 0.0:
        score -= 15.0
    if ema200_slope > 0.0:
        score += 10.0
    elif ema200_slope < 0.0:
        score -= 10.0

    return float(np.clip(score, -100.0, 100.0)), latest_ema20, latest_ema50, latest_ema200


def _momentum_components(df: pd.DataFrame) -> float:
    close = df["close"]
    rsi = ta.momentum.rsi(close, window=14)
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    macd_signal = macd.ewm(span=9, adjust=False).mean()
    macd_hist = macd - macd_signal

    score = 0.0
    latest_rsi = float(rsi.iloc[-1]) if pd.notna(rsi.iloc[-1]) else 50.0
    latest_hist = float(macd_hist.iloc[-1]) if pd.notna(macd_hist.iloc[-1]) else 0.0

    if latest_rsi > 52.0:
        score += 50.0
    elif latest_rsi < 48.0:
        score -= 50.0

    if latest_hist > 0.0:
        score += 50.0
    elif latest_hist < 0.0:
        score -= 50.0

    return float(np.clip(score, -100.0, 100.0))


def _regime_components(df: pd.DataFrame) -> tuple[float, str]:
    close = pd.to_numeric(df["close"], errors="coerce").dropna()
    if len(close) >= 20:
        recent = close.iloc[-30:] if len(close) >= 30 else close
        rets = recent.pct_change().dropna()
        if not rets.empty:
            drift = abs(float(recent.iloc[-1] / recent.iloc[0] - 1.0))
            noise = float(rets.std()) * np.sqrt(len(rets))
            trend_ratio = drift / (noise + 1e-9)
        else:
            trend_ratio = 0.0
    else:
        trend_ratio = 0.0

    try:
        adx_series = ta.trend.adx(df["high"], df["low"], df["close"], window=14)
        adx_val = float(adx_series.iloc[-1]) if pd.notna(adx_series.iloc[-1]) else float("nan")
    except Exception:
        adx_val = float("nan")

    if np.isfinite(adx_val):
        if adx_val >= 30.0 and trend_ratio >= 1.2:
            return 95.0, "TREND"
        if adx_val >= 25.0 and trend_ratio >= 0.9:
            return 85.0, "TREND"
        if trend_ratio <= 0.6:
            return 35.0, "MIXED" if adx_val >= 25.0 else "RANGE"
        if adx_val >= 18.0 or trend_ratio >= 0.8:
            return 60.0, "MIXED"
        return 25.0, "RANGE"

    if len(close) < 20:
        return 0.0, "UNAVAILABLE"
    if trend_ratio <= 0.0:
        return 0.0, "UNAVAILABLE"
    if trend_ratio >= 1.2:
        return 80.0, "TREND"
    if trend_ratio <= 0.6:
        return 25.0, "RANGE"
    return 55.0, "MIXED"


def _direction_from_score(score: float) -> str:
    if score >= _TIMEFRAME_SCORE_THRESHOLD:
        return "UPSIDE"
    if score <= -_TIMEFRAME_SCORE_THRESHOLD:
        return "DOWNSIDE"
    return "NEUTRAL"


def _regime_multiplier(regime_label: str, regime_quality: float) -> float:
    label = str(regime_label or "").strip().upper()
    quality = float(np.clip(regime_quality, 0.0, 100.0))
    if label == "TREND":
        return 1.0
    if label == "MIXED":
        return 0.85
    if label == "RANGE":
        return 0.55
    if quality <= 0.0:
        return 0.0
    return float(np.clip(quality / 100.0, 0.40, 1.0))


def _location_quality(df: pd.DataFrame, direction: str) -> tuple[float, float, float, float]:
    recent = df.tail(min(len(df), 40))
    support = float(recent["low"].min())
    resistance = float(recent["high"].max())
    close = float(df["close"].iloc[-1])

    if close <= 0.0:
        return 0.0, support, resistance, close

    dir_key = normalize_direction(direction)
    score = 50.0
    room_to_resistance = max(0.0, (resistance - close) / close * 100.0)
    room_to_support = max(0.0, (close - support) / close * 100.0)

    if dir_key == "UPSIDE":
        if room_to_resistance >= 6.0:
            score += 25.0
        elif room_to_resistance >= 3.0:
            score += 10.0
        elif room_to_resistance <= 1.5:
            score -= 25.0
        elif room_to_resistance <= 3.0:
            score -= 10.0

        if room_to_support <= 3.0:
            score += 20.0
        elif room_to_support >= 10.0:
            score -= 10.0
    elif dir_key == "DOWNSIDE":
        if room_to_support >= 6.0:
            score += 25.0
        elif room_to_support >= 3.0:
            score += 10.0
        elif room_to_support <= 1.5:
            score -= 25.0
        elif room_to_support <= 3.0:
            score -= 10.0

        if room_to_resistance <= 3.0:
            score += 20.0
        elif room_to_resistance >= 10.0:
            score -= 10.0

    return float(np.clip(score, 0.0, 100.0)), support, resistance, close


def analyze_timeframe_direction(df: pd.DataFrame | None, *, timeframe: str) -> TimeframeDirectionSnapshot:
    safe = _safe_frame(df)
    if safe is None:
        return _empty_tf_snapshot(timeframe)

    structure_score, structure_label = _structure_components(safe)
    trend_score, _ema20, _ema50, _ema200 = _trend_components(safe)
    momentum_score = _momentum_components(safe)
    regime_quality, regime_label = _regime_components(safe)

    base_score = 0.50 * structure_score + 0.35 * trend_score + 0.15 * momentum_score
    raw_score = float(np.clip(base_score, -100.0, 100.0))
    score = raw_score * _regime_multiplier(regime_label, regime_quality)
    score = float(np.clip(score, -100.0, 100.0))
    direction = _direction_from_score(score)
    location_quality, support, resistance, close = _location_quality(safe, direction)

    return TimeframeDirectionSnapshot(
        timeframe=timeframe,
        direction=direction,
        score=score,
        raw_score=raw_score,
        structure_score=structure_score,
        structure_label=structure_label,
        trend_score=trend_score,
        momentum_score=momentum_score,
        regime_quality=regime_quality,
        regime_label=regime_label,
        location_quality=location_quality,
        support=support,
        resistance=resistance,
        close=close,
        degraded=False,
    )


def _early_lead_direction(snapshot: TimeframeDirectionSnapshot) -> tuple[str, bool]:
    if snapshot.direction != "NEUTRAL":
        return snapshot.direction, False

    structure_label = str(snapshot.structure_label or "").strip().upper()
    bullish_structure = structure_label in {"EARLY_UP", "HH/HL", "BREAKOUT_UP"}
    bearish_structure = structure_label in {"EARLY_DOWN", "LH/LL", "BREAKOUT_DOWN"}

    if (
        float(snapshot.raw_score) >= _EARLY_LEAD_SCORE_THRESHOLD
        and float(snapshot.trend_score) >= 10.0
        and (bullish_structure or float(snapshot.momentum_score) > 0.0)
    ):
        return "UPSIDE", True
    if (
        float(snapshot.raw_score) <= -_EARLY_LEAD_SCORE_THRESHOLD
        and float(snapshot.trend_score) <= -10.0
        and (bearish_structure or float(snapshot.momentum_score) < 0.0)
    ):
        return "DOWNSIDE", True
    return "NEUTRAL", False


def _emerging_lead_direction(
    lead: TimeframeDirectionSnapshot,
    confirm: TimeframeDirectionSnapshot,
) -> tuple[str, bool]:
    if lead.direction != "NEUTRAL":
        return lead.direction, False

    lead_structure = str(lead.structure_label or "").strip().upper()
    confirm_structure = str(confirm.structure_label or "").strip().upper()

    if (
        float(lead.raw_score) >= 4.0
        and float(lead.trend_score) >= 5.0
        and lead_structure not in {"EARLY_DOWN", "LH/LL", "BREAKOUT_DOWN"}
        and confirm_structure in {"BREAKOUT_UP", "EARLY_UP", "HH/HL"}
        and float(confirm.score) >= _TIMEFRAME_SCORE_THRESHOLD
    ):
        return "UPSIDE", True

    if (
        float(lead.raw_score) <= -4.0
        and float(lead.trend_score) <= -5.0
        and lead_structure not in {"EARLY_UP", "HH/HL", "BREAKOUT_UP"}
        and confirm_structure in {"BREAKOUT_DOWN", "EARLY_DOWN", "LH/LL"}
        and float(confirm.score) <= -_TIMEFRAME_SCORE_THRESHOLD
    ):
        return "DOWNSIDE", True

    return "NEUTRAL", False


def _timeframe_alignment(
    lead: TimeframeDirectionSnapshot,
    confirm: TimeframeDirectionSnapshot,
    *,
    lead_direction: str | None = None,
    early_lead_bias: bool = False,
) -> tuple[float, bool]:
    resolved_lead_direction = normalize_direction(lead_direction or lead.direction)
    if resolved_lead_direction == "NEUTRAL":
        return 0.0, False
    if confirm.direction == resolved_lead_direction:
        return (80.0 if early_lead_bias else 100.0), False
    if confirm.direction == "NEUTRAL":
        return (50.0 if early_lead_bias else 70.0), False
    return 0.0, True


def build_spot_direction_snapshot(
    *,
    df_4h: pd.DataFrame | None,
    df_1d: pd.DataFrame | None,
    lead_df: pd.DataFrame | None = None,
    confirm_df: pd.DataFrame | None = None,
    lead_timeframe: str = "1d",
    confirm_timeframe: str = "4h",
) -> SpotDirectionSnapshot:
    if lead_df is None and confirm_df is None:
        lead_df = df_1d
        confirm_df = df_4h
        lead_timeframe = "1d"
        confirm_timeframe = "4h"

    confirm = analyze_timeframe_direction(confirm_df, timeframe=confirm_timeframe)
    lead = analyze_timeframe_direction(lead_df, timeframe=lead_timeframe)

    degraded_data = bool(confirm.degraded or lead.degraded)
    lead_direction, early_lead_bias = _early_lead_direction(lead)
    if lead_direction == "NEUTRAL":
        lead_direction, early_lead_bias = _emerging_lead_direction(lead, confirm)
    timeframe_alignment, timeframe_conflict = _timeframe_alignment(
        lead,
        confirm,
        lead_direction=lead_direction,
        early_lead_bias=early_lead_bias,
    )
    lead_effective_score = lead.raw_score if early_lead_bias else lead.score
    score = 0.60 * lead_effective_score + 0.40 * confirm.score
    score = float(np.clip(score, -100.0, 100.0))

    structure_quality = float(np.clip(0.60 * abs(lead.structure_score) + 0.40 * abs(confirm.structure_score), 0.0, 100.0))
    trend_quality = float(np.clip(0.60 * abs(lead.trend_score) + 0.40 * abs(confirm.trend_score), 0.0, 100.0))
    regime_quality = float(np.clip(0.60 * lead.regime_quality + 0.40 * confirm.regime_quality, 0.0, 100.0))
    location_quality = float(np.clip(0.60 * lead.location_quality + 0.40 * confirm.location_quality, 0.0, 100.0))
    range_regime = lead.regime_label == "RANGE"

    if degraded_data:
        direction = "NEUTRAL"
        note = "Higher-timeframe context is incomplete."
    elif lead_direction == "NEUTRAL":
        direction = "NEUTRAL"
        note = f"{lead_timeframe.upper()} structure is not directional enough."
    elif timeframe_conflict:
        direction = "NEUTRAL"
        note = f"{confirm_timeframe.upper()} direction conflicts with {lead_timeframe.upper()} bias."
    elif early_lead_bias:
        if confirm.direction != lead_direction:
            direction = "NEUTRAL"
            note = (
                f"{lead_timeframe.upper()} bias is still early and "
                f"{confirm_timeframe.upper()} has not confirmed it yet."
            )
        elif abs(score) < _EARLY_FINAL_DIRECTION_SCORE_THRESHOLD:
            direction = "NEUTRAL"
            note = (
                f"Early {lead_timeframe.upper()} bias exists, but the combined "
                f"{lead_timeframe.upper()} + {confirm_timeframe.upper()} score is still too weak."
            )
        else:
            direction = lead_direction
            note = (
                f"{lead_timeframe.upper()} bias is still early, but "
                f"{confirm_timeframe.upper()} confirms the same side."
            )
    elif abs(score) < _FINAL_DIRECTION_SCORE_THRESHOLD:
        direction = "NEUTRAL"
        note = "Combined higher-timeframe score is too weak."
    else:
        direction = lead.direction
        note = (
            f"{lead_timeframe.upper()} structure defines the bias and "
            f"{confirm_timeframe.upper()} does not oppose it."
        )

    return SpotDirectionSnapshot(
        direction=direction,
        score=score,
        timeframe_alignment=timeframe_alignment,
        structure_quality=structure_quality,
        trend_quality=trend_quality,
        regime_quality=regime_quality,
        location_quality=location_quality,
        timeframe_conflict=timeframe_conflict,
        degraded_data=degraded_data,
        range_regime=range_regime,
        note=note,
        four_hour=confirm,
        one_day=lead,
        lead_timeframe=lead_timeframe,
        confirm_timeframe=confirm_timeframe,
    )
