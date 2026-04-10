"""Scalping entry/target decision logic."""

from __future__ import annotations

from typing import Callable

import pandas as pd
import ta

_SUPPORTED_SCALP_TIMEFRAMES = {"1m", "3m", "5m", "15m", "1h"}


def _normalize_timeframe(value: str | None) -> str:
    return str(value or "").strip().lower()


def _dir_key(value: str | None) -> str:
    s = str(value or "").strip().upper()
    if s in {"LONG", "UPSIDE", "BUY", "BULLISH"}:
        return "UPSIDE"
    if s in {"SHORT", "DOWNSIDE", "SELL", "BEARISH"}:
        return "DOWNSIDE"
    return "NEUTRAL"


def _setup_key(value: str | None) -> str:
    s = str(value or "").strip().upper()
    if "ENTER" in s:
        return "ENTER"
    if "PROBE" in s:
        return "PROBE"
    if "WATCH" in s:
        return "WATCH"
    if "SKIP" in s:
        return "SKIP"
    return "UNKNOWN"


def scalp_timeframe_supported(timeframe: str | None) -> bool:
    return _normalize_timeframe(timeframe) in _SUPPORTED_SCALP_TIMEFRAMES


def scalp_gate_thresholds(timeframe: str | None) -> tuple[float, float, float]:
    """Timeframe-adaptive scalp gate thresholds: (min_rr, min_adx, min_confidence)."""
    t = _normalize_timeframe(timeframe)
    if t in {"1m", "3m", "5m", "15m"}:
        return 1.30, 18.0, 52.0
    if t == "1h":
        return 1.40, 18.0, 52.0
    return 1.40, 18.0, 52.0


def scalp_reason_text(
    reason_code: str | None,
    *,
    timeframe: str | None = None,
    min_rr: float | None = None,
    min_adx: float | None = None,
    min_confidence: float | None = None,
) -> str:
    code = str(reason_code or "").strip().upper()
    tf = _normalize_timeframe(timeframe)
    if code == "UNSUPPORTED_TIMEFRAME":
        frame = tf or "this timeframe"
        return (
            f"Scalp lens is intraday-only and stays disabled on {frame}. "
            "Use 1m/3m/5m/15m/1h for scalp timing."
        )
    if code == "SETUP_NOT_READY":
        return "Main setup is not ready enough yet for a scalp entry."
    if code == "MARKET_NO_TRADE":
        return "Market-wide trade gate is standing aside, so scalp timing stays blocked."
    if code == "MARKET_DEFENSIVE_ONLY":
        return "Market-wide stance is defensive, so scalp timing stays blocked."
    if code == "ARCHIVE_GUARDRAIL":
        return "Matched archive history is too weak in this window for a scalp plan."
    if code == "ARCHIVE_SCALP_SUPPORT":
        return "Archive scalp history is supportive enough to keep this borderline intraday plan live."
    if code == "ARCHIVE_SCALP_CAUTION":
        return "Archive scalp history is soft here, so this borderline intraday plan stays blocked."
    if code == "NO_SCALP_DIRECTION":
        return "No clean intraday scalp direction is forming on current structure."
    if code == "SIGNAL_DIRECTION_NEUTRAL":
        return "Selected-timeframe direction is neutral, so scalp timing stays blocked."
    if code == "DIRECTION_MISMATCH":
        return "Local scalp side is fighting the main selected-timeframe direction."
    if code == "CONFLICT":
        return "Technical and AI confirmation are still in conflict."
    if code == "RR_TOO_LOW":
        return f"Scalp reward-to-risk is below the required threshold ({float(min_rr or 0.0):.2f})."
    if code == "ADX_TOO_LOW":
        return f"Trend strength is below the required threshold ({float(min_adx or 0.0):.0f})."
    if code == "CONFIDENCE_TOO_LOW":
        return f"Execution confidence is below the required threshold ({float(min_confidence or 0.0):.0f})."
    if code == "INVALID_LEVELS":
        return "Entry, stop, or target levels are not valid on the current candle structure."
    return ""


def scalp_reason_short_label(reason_code: str | None) -> str:
    code = str(reason_code or "").strip().upper()
    short = {
        "SETUP_NOT_READY": "Setup",
        "MARKET_NO_TRADE": "No-Trade",
        "MARKET_DEFENSIVE_ONLY": "Defensive",
        "ARCHIVE_GUARDRAIL": "Archive",
        "ARCHIVE_SCALP_CAUTION": "Archive",
        "DIRECTION_MISMATCH": "Mismatch",
        "CONFLICT": "Conflict",
        "RR_TOO_LOW": "Low R:R",
        "ADX_TOO_LOW": "Low ADX",
        "CONFIDENCE_TOO_LOW": "Low Conf",
        "INVALID_LEVELS": "Levels",
        "SIGNAL_DIRECTION_NEUTRAL": "Neutral Dir",
    }
    return short.get(code, "")


def scalp_quality_gate(
    *,
    scalp_direction: str | None,
    signal_direction: str | None,
    rr_ratio: float | None,
    adx_val: float | None,
    confidence: float | None = None,
    conviction_label: str | None,
    entry: float | None,
    stop: float | None,
    target: float | None,
    min_rr: float = 1.50,
    min_adx: float = 20.0,
    min_confidence: float = 55.0,
    timeframe: str | None = None,
    setup_confirm: str | None = None,
    market_trade_gate_key: str | None = None,
    archive_guardrail_penalty: float | None = None,
    archive_guardrail_label: str | None = None,
) -> tuple[bool, str]:
    """Single source-of-truth gate for execution-ready scalp opportunities.

    Returns:
      (True, "PASS") when all guards pass, otherwise (False, REASON_CODE).
    """
    dir_norm = _dir_key(scalp_direction)
    sig_norm = _dir_key(signal_direction)
    conv = str(conviction_label or "").upper().strip()
    gate_key = str(market_trade_gate_key or "").strip().upper()
    setup_key = _setup_key(setup_confirm)
    tf = _normalize_timeframe(timeframe)
    guardrail_label = str(archive_guardrail_label or "").strip().upper()
    try:
        archive_penalty = float(archive_guardrail_penalty) if archive_guardrail_penalty is not None else 0.0
    except Exception:
        archive_penalty = 0.0

    if tf and not scalp_timeframe_supported(tf):
        return False, "UNSUPPORTED_TIMEFRAME"
    if gate_key == "NO_TRADE":
        return False, "MARKET_NO_TRADE"
    if gate_key == "DEFENSIVE_ONLY":
        return False, "MARKET_DEFENSIVE_ONLY"
    if setup_key in {"SKIP", "WATCH"}:
        return False, "SETUP_NOT_READY"
    if archive_penalty >= 6.0 or guardrail_label == "ARCHIVE GUARDRAIL":
        return False, "ARCHIVE_GUARDRAIL"

    if dir_norm not in {"UPSIDE", "DOWNSIDE"}:
        return False, "NO_SCALP_DIRECTION"
    if sig_norm not in {"UPSIDE", "DOWNSIDE"}:
        return False, "SIGNAL_DIRECTION_NEUTRAL"
    if dir_norm != sig_norm:
        return False, "DIRECTION_MISMATCH"
    if conv == "CONFLICT":
        return False, "CONFLICT"

    try:
        rr = float(rr_ratio) if rr_ratio is not None else float("nan")
    except Exception:
        rr = float("nan")
    if pd.isna(rr) or rr < float(min_rr):
        return False, "RR_TOO_LOW"

    try:
        adx_f = float(adx_val) if adx_val is not None else float("nan")
    except Exception:
        adx_f = float("nan")
    if pd.isna(adx_f) or adx_f < float(min_adx):
        return False, "ADX_TOO_LOW"

    try:
        confidence_f = float(confidence) if confidence is not None else float("nan")
    except Exception:
        confidence_f = float("nan")
    if pd.isna(confidence_f) or confidence_f < float(min_confidence):
        return False, "CONFIDENCE_TOO_LOW"

    try:
        e = float(entry) if entry is not None else 0.0
        s = float(stop) if stop is not None else 0.0
        t = float(target) if target is not None else 0.0
    except Exception:
        return False, "INVALID_LEVELS"
    if e <= 0.0 or s <= 0.0 or t <= 0.0:
        return False, "INVALID_LEVELS"

    return True, "PASS"


def apply_scalp_archive_calibration(
    gate_pass: bool,
    gate_reason: str | None,
    *,
    calibration_delta: float = 0.0,
    rr_ratio: float | None = None,
    adx_val: float | None = None,
    confidence: float | None = None,
    timeframe: str | None = None,
) -> tuple[bool, str]:
    delta = float(calibration_delta or 0.0)
    reason = str(gate_reason or "").strip().upper() or ("PASS" if gate_pass else "")
    if abs(delta) < 0.55:
        return gate_pass, gate_reason or ("PASS" if gate_pass else "")

    hard_blockers = {
        "UNSUPPORTED_TIMEFRAME",
        "SETUP_NOT_READY",
        "MARKET_NO_TRADE",
        "MARKET_DEFENSIVE_ONLY",
        "ARCHIVE_GUARDRAIL",
        "NO_SCALP_DIRECTION",
        "SIGNAL_DIRECTION_NEUTRAL",
        "DIRECTION_MISMATCH",
        "CONFLICT",
        "INVALID_LEVELS",
    }
    if reason in hard_blockers:
        return gate_pass, gate_reason or reason

    min_rr, min_adx, min_confidence = scalp_gate_thresholds(timeframe)
    try:
        rr_val = float(rr_ratio) if rr_ratio is not None else float("nan")
    except Exception:
        rr_val = float("nan")
    try:
        adx_f = float(adx_val) if adx_val is not None else float("nan")
    except Exception:
        adx_f = float("nan")
    try:
        confidence_f = float(confidence) if confidence is not None else float("nan")
    except Exception:
        confidence_f = float("nan")

    if not gate_pass and delta >= 0.55:
        if reason == "RR_TOO_LOW" and pd.notna(rr_val) and rr_val >= (float(min_rr) - 0.12):
            return True, "ARCHIVE_SCALP_SUPPORT"
        if reason == "ADX_TOO_LOW" and pd.notna(adx_f) and adx_f >= (float(min_adx) - 2.0):
            return True, "ARCHIVE_SCALP_SUPPORT"
        if reason == "CONFIDENCE_TOO_LOW" and pd.notna(confidence_f) and confidence_f >= (float(min_confidence) - 4.0):
            return True, "ARCHIVE_SCALP_SUPPORT"

    if gate_pass and delta <= -0.55:
        borderline_rr = pd.notna(rr_val) and rr_val <= (float(min_rr) + 0.12)
        borderline_adx = pd.notna(adx_f) and adx_f <= (float(min_adx) + 2.0)
        borderline_confidence = pd.notna(confidence_f) and confidence_f <= (float(min_confidence) + 4.0)
        if borderline_rr or borderline_adx or borderline_confidence:
            return False, "ARCHIVE_SCALP_CAUTION"

    return gate_pass, gate_reason or ("PASS" if gate_pass else "")


def get_scalping_entry_target(
    df: pd.DataFrame,
    bias_score: float,
    supertrend_trend: str,
    ichimoku_trend: str,
    vwap_label: str,
    *,
    sr_lookback_fn: Callable[[str | None], int],
    timeframe: str | None = None,
    execution_snapshot=None,
    trend_led_snapshot=None,
    ai_led_snapshot=None,
    spot_direction: str | None = None,
    ai_direction: str | None = None,
):
    if df is None or len(df) <= 30:
        return None, 0.0, 0.0, 0.0, 0.0, ""

    df = df.copy()
    inferred_tf = None
    if "timestamp" in df.columns and len(df) >= 2:
        delta_mins = (df["timestamp"].iloc[-1] - df["timestamp"].iloc[-2]).total_seconds() / 60
        if delta_mins <= 1.5:
            inferred_tf = "1m"
        elif delta_mins <= 4:
            inferred_tf = "3m"
        elif delta_mins <= 7:
            inferred_tf = "5m"
        elif delta_mins <= 20:
            inferred_tf = "15m"
        elif delta_mins <= 90:
            inferred_tf = "1h"
        elif delta_mins <= 300:
            inferred_tf = "4h"
        else:
            inferred_tf = "1d"

    scalp_tf = _normalize_timeframe(timeframe) or _normalize_timeframe(inferred_tf) or "15m"
    if not scalp_timeframe_supported(scalp_tf):
        return None, 0.0, 0.0, 0.0, 0.0, "Unsupported timeframe"

    latest_idx = -1
    closed_df = df

    df["ema5"] = df["close"].ewm(span=5).mean()
    df["ema13"] = df["close"].ewm(span=13).mean()
    df["ema21"] = df["close"].ewm(span=21).mean()
    df["atr"] = ta.volatility.average_true_range(df["high"], df["low"], df["close"], window=14)
    df["rsi"] = ta.momentum.rsi(df["close"], window=14)

    macd_ind = ta.trend.MACD(df["close"])
    df["macd"] = macd_ind.macd()
    df["macd_signal"] = macd_ind.macd_signal()
    df["macd_diff"] = macd_ind.macd_diff()
    try:
        adx_series = ta.trend.adx(df["high"], df["low"], df["close"], window=14)
        adx_val = float(adx_series.iloc[latest_idx]) if not pd.isna(adx_series.iloc[latest_idx]) else float("nan")
    except Exception:
        adx_val = float("nan")

    latest = df.iloc[latest_idx]

    close_price = latest["close"]
    atr = latest["atr"]
    execution_support = float("nan")
    execution_resistance = float("nan")
    execution_structure = float("nan")
    execution_trend = float("nan")
    execution_location = float("nan")
    execution_ema21 = float("nan")
    try:
        execution_close = float(getattr(execution_snapshot, "close", float("nan")))
        execution_atr = float(getattr(execution_snapshot, "atr", float("nan")))
        execution_support = float(getattr(execution_snapshot, "support", float("nan")))
        execution_resistance = float(getattr(execution_snapshot, "resistance", float("nan")))
        execution_structure = float(getattr(execution_snapshot, "structure_quality", float("nan")))
        execution_trend = float(getattr(execution_snapshot, "trend_quality", float("nan")))
        execution_location = float(getattr(execution_snapshot, "location_quality", float("nan")))
        execution_ema21 = float(getattr(execution_snapshot, "ema21", float("nan")))
        if pd.notna(execution_close) and execution_close > 0.0:
            close_price = execution_close
        if pd.notna(execution_atr) and execution_atr > 0.0:
            atr = execution_atr
    except Exception:
        pass
    if pd.isna(close_price) or pd.isna(atr) or float(close_price) <= 0.0 or float(atr) <= 0.0:
        return None, None, None, None, None, "Invalid ATR/price"

    ema_trend_up = latest["ema5"] > latest["ema13"] > latest["ema21"]
    ema_trend_down = latest["ema5"] < latest["ema13"] < latest["ema21"]
    macd_confirm_long = latest["macd"] > latest["macd_signal"] and latest["macd_diff"] > 0
    macd_confirm_short = latest["macd"] < latest["macd_signal"] and latest["macd_diff"] < 0

    if not pd.isna(adx_val) and adx_val >= 30:
        rsi_long_min, rsi_short_max = 52.0, 48.0
    elif not pd.isna(adx_val) and adx_val >= 20:
        rsi_long_min, rsi_short_max = 54.0, 46.0
    else:
        rsi_long_min, rsi_short_max = 56.0, 44.0
    rsi_confirm_long = latest["rsi"] >= rsi_long_min
    rsi_confirm_short = latest["rsi"] <= rsi_short_max

    long_regime_confirms = sum(
        [
            1 if supertrend_trend == "Bullish" else 0,
            1 if ichimoku_trend == "Bullish" else 0,
            1 if vwap_label == "🟢 Above" else 0,
        ]
    )
    short_regime_confirms = sum(
        [
            1 if supertrend_trend == "Bearish" else 0,
            1 if ichimoku_trend == "Bearish" else 0,
            1 if vwap_label == "🔴 Below" else 0,
        ]
    )
    long_votes = 0.0
    short_votes = 0.0
    long_votes += 1.0 if bias_score >= 56 else 0.0
    short_votes += 1.0 if bias_score <= 44 else 0.0
    long_votes += 1.0 if ema_trend_up else 0.0
    short_votes += 1.0 if ema_trend_down else 0.0
    long_votes += 1.0 if macd_confirm_long else 0.0
    short_votes += 1.0 if macd_confirm_short else 0.0
    long_votes += 1.0 if rsi_confirm_long else 0.0
    short_votes += 1.0 if rsi_confirm_short else 0.0
    long_votes += 1.0 if long_regime_confirms >= 2 else 0.0
    short_votes += 1.0 if short_regime_confirms >= 2 else 0.0

    spot_key = _dir_key(spot_direction)
    ai_key = _dir_key(ai_direction)
    if pd.notna(execution_structure) and execution_structure >= 60.0 and spot_key in {"UPSIDE", "DOWNSIDE"}:
        long_votes += 0.75 if spot_key == "UPSIDE" else 0.0
        short_votes += 0.75 if spot_key == "DOWNSIDE" else 0.0
    if pd.notna(execution_trend) and execution_trend >= 60.0 and spot_key in {"UPSIDE", "DOWNSIDE"}:
        long_votes += 0.75 if spot_key == "UPSIDE" else 0.0
        short_votes += 0.75 if spot_key == "DOWNSIDE" else 0.0
    if pd.notna(execution_location) and execution_location >= 58.0 and spot_key in {"UPSIDE", "DOWNSIDE"}:
        long_votes += 0.50 if spot_key == "UPSIDE" else 0.0
        short_votes += 0.50 if spot_key == "DOWNSIDE" else 0.0

    trend_led_state = str(getattr(trend_led_snapshot, "state", "") or "").strip().upper()
    trend_led_score = float(getattr(trend_led_snapshot, "score", float("nan")) or float("nan"))
    if spot_key in {"UPSIDE", "DOWNSIDE"}:
        if trend_led_state == "READY":
            long_votes += 1.0 if spot_key == "UPSIDE" else 0.0
            short_votes += 1.0 if spot_key == "DOWNSIDE" else 0.0
        elif trend_led_state == "WATCH" and pd.notna(trend_led_score) and trend_led_score >= 65.0:
            long_votes += 0.40 if spot_key == "UPSIDE" else 0.0
            short_votes += 0.40 if spot_key == "DOWNSIDE" else 0.0

    ai_led_state = str(getattr(ai_led_snapshot, "state", "") or "").strip().upper()
    ai_led_score = float(getattr(ai_led_snapshot, "score", float("nan")) or float("nan"))
    if ai_key in {"UPSIDE", "DOWNSIDE"}:
        if ai_led_state == "READY":
            long_votes += 0.85 if ai_key == "UPSIDE" else 0.0
            short_votes += 0.85 if ai_key == "DOWNSIDE" else 0.0
        elif ai_led_state == "WATCH" and pd.notna(ai_led_score) and ai_led_score >= 68.0:
            long_votes += 0.35 if ai_key == "UPSIDE" else 0.0
            short_votes += 0.35 if ai_key == "DOWNSIDE" else 0.0

    if spot_key in {"UPSIDE", "DOWNSIDE"} and spot_key == ai_key:
        long_votes += 0.40 if spot_key == "UPSIDE" else 0.0
        short_votes += 0.40 if spot_key == "DOWNSIDE" else 0.0

    scalp_direction = None
    if long_votes >= 4.0 and long_votes > short_votes + 0.25:
        scalp_direction = "LONG"
    elif short_votes >= 4.0 and short_votes > long_votes + 0.25:
        scalp_direction = "SHORT"

    recent = closed_df.tail(sr_lookback_fn(scalp_tf or inferred_tf))
    support = recent["low"].min()
    resistance = recent["high"].max()
    if pd.notna(execution_support) and execution_support > 0.0:
        support = execution_support
    if pd.notna(execution_resistance) and execution_resistance > 0.0:
        resistance = execution_resistance

    entry_s = stop_s = target_s = 0.0
    breakout_note = ""
    ema_anchor = float(latest["ema5"])
    if pd.notna(execution_ema21) and execution_ema21 > 0.0:
        if scalp_direction == "LONG":
            ema_anchor = max(ema_anchor, execution_ema21)
        elif scalp_direction == "SHORT":
            ema_anchor = min(ema_anchor, execution_ema21)

    if scalp_direction == "LONG":
        entry_s = max(float(close_price), float(ema_anchor)) + 0.20 * float(atr)
        stop_struct = float(support)
        stop_floor = entry_s - 1.80 * float(atr)
        stop_ceil = entry_s - 0.60 * float(atr)
        stop_s = max(stop_struct, stop_floor)
        stop_s = min(stop_s, stop_ceil)
        target_s = max(float(resistance), entry_s + 1.40 * float(atr))
        if target_s > resistance:
            breakout_note = f"⚠️ Target (${target_s:.4f}) is above resistance (${resistance:.4f}). Breakout needed."
    elif scalp_direction == "SHORT":
        entry_s = min(float(close_price), float(ema_anchor)) - 0.20 * float(atr)
        stop_struct = float(resistance)
        stop_floor = entry_s + 0.60 * float(atr)
        stop_ceil = entry_s + 1.80 * float(atr)
        stop_s = min(stop_struct, stop_ceil)
        stop_s = max(stop_s, stop_floor)
        target_s = min(float(support), entry_s - 1.40 * float(atr))
        if target_s < support:
            breakout_note = f"⚠️ Target (${target_s:.4f}) is below support (${support:.4f}). Breakout needed."

    if stop_s <= 0.0 or target_s <= 0.0 or entry_s <= 0.0:
        return None, None, None, None, None, "Invalid plan levels"
    rr_ratio = abs(target_s - entry_s) / abs(entry_s - stop_s) if entry_s != stop_s else 0.0
    return scalp_direction, entry_s, target_s, stop_s, rr_ratio, breakout_note
