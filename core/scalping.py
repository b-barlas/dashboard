"""Scalping entry/target decision logic."""

from __future__ import annotations

from typing import Callable

import pandas as pd
import ta


def _dir_key(value: str | None) -> str:
    s = str(value or "").strip().upper()
    if s in {"LONG", "UPSIDE", "BUY", "BULLISH"}:
        return "UPSIDE"
    if s in {"SHORT", "DOWNSIDE", "SELL", "BEARISH"}:
        return "DOWNSIDE"
    return "NEUTRAL"


def scalp_quality_gate(
    *,
    scalp_direction: str | None,
    signal_direction: str | None,
    rr_ratio: float | None,
    adx_val: float | None,
    strength: float | None,
    conviction_label: str | None,
    entry: float | None,
    stop: float | None,
    target: float | None,
    min_rr: float = 1.50,
    min_adx: float = 20.0,
    min_strength: float = 55.0,
) -> tuple[bool, str]:
    """Single source-of-truth gate for execution-ready scalp opportunities.

    Returns:
      (True, "PASS") when all guards pass, otherwise (False, REASON_CODE).
    """
    dir_norm = _dir_key(scalp_direction)
    sig_norm = _dir_key(signal_direction)
    conv = str(conviction_label or "").upper().strip()

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
        strength_f = float(strength) if strength is not None else float("nan")
    except Exception:
        strength_f = float("nan")
    if pd.isna(strength_f) or strength_f < float(min_strength):
        return False, "STRENGTH_TOO_LOW"

    try:
        e = float(entry) if entry is not None else 0.0
        s = float(stop) if stop is not None else 0.0
        t = float(target) if target is not None else 0.0
    except Exception:
        return False, "INVALID_LEVELS"
    if e <= 0.0 or s <= 0.0 or t <= 0.0:
        return False, "INVALID_LEVELS"

    return True, "PASS"


def get_scalping_entry_target(
    df: pd.DataFrame,
    bias_score: float,
    supertrend_trend: str,
    ichimoku_trend: str,
    vwap_label: str,
    volume_spike: bool,
    strict_mode: bool = True,
    *,
    sr_lookback_fn: Callable[[str | None], int],
):
    if df is None or len(df) <= 30:
        return None, 0.0, 0.0, 0.0, 0.0, ""

    df = df.copy()
    _inferred_tf = None
    if "timestamp" in df.columns and len(df) >= 2:
        _delta_mins = (df["timestamp"].iloc[-1] - df["timestamp"].iloc[-2]).total_seconds() / 60
        if _delta_mins <= 1.5:
            _inferred_tf = "1m"
        elif _delta_mins <= 4:
            _inferred_tf = "3m"
        elif _delta_mins <= 7:
            _inferred_tf = "5m"
        elif _delta_mins <= 20:
            _inferred_tf = "15m"
        elif _delta_mins <= 90:
            _inferred_tf = "1h"
        elif _delta_mins <= 300:
            _inferred_tf = "4h"
        else:
            _inferred_tf = "1d"

    # Standard policy: callers pass a closed-candle frame.
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

    stoch = ta.momentum.StochRSIIndicator(close=df["close"], window=14, smooth1=3, smooth2=3)
    stoch_k = stoch.stochrsi_k()
    stochrsi_k_val = float(stoch_k.iloc[latest_idx]) if not pd.isna(stoch_k.iloc[latest_idx]) else float("nan")

    bb = ta.volatility.BollingerBands(close=df["close"], window=20, window_dev=2)
    latest = df.iloc[latest_idx]
    last_close = float(latest["close"])
    hband_v = bb.bollinger_hband().iloc[latest_idx]
    lband_v = bb.bollinger_lband().iloc[latest_idx]
    hband = float(hband_v) if not pd.isna(hband_v) else float("nan")
    lband = float(lband_v) if not pd.isna(lband_v) else float("nan")
    if last_close > hband:
        bollinger_bias = "Overbought"
    elif last_close < lband:
        bollinger_bias = "Oversold"
    else:
        bollinger_bias = "Neutral"

    close_price = latest["close"]
    atr = latest["atr"]
    if pd.isna(close_price) or pd.isna(atr) or float(close_price) <= 0.0 or float(atr) <= 0.0:
        return None, None, None, None, None, "Invalid ATR/price"

    ema_trend_up = latest["ema5"] > latest["ema13"] > latest["ema21"]
    ema_trend_down = latest["ema5"] < latest["ema13"] < latest["ema21"]
    macd_confirm_long = latest["macd"] > latest["macd_signal"] and latest["macd_diff"] > 0
    macd_confirm_short = latest["macd"] < latest["macd_signal"] and latest["macd_diff"] < 0

    # Regime-adaptive RSI gates: stronger trend allows earlier entries.
    if not pd.isna(adx_val) and adx_val >= 30:
        rsi_long_min, rsi_short_max = 52.0, 48.0
    elif not pd.isna(adx_val) and adx_val >= 20:
        rsi_long_min, rsi_short_max = 54.0, 46.0
    else:
        rsi_long_min, rsi_short_max = 56.0, 44.0
    rsi_confirm_long = latest["rsi"] >= rsi_long_min
    rsi_confirm_short = latest["rsi"] <= rsi_short_max

    # Build directional votes from independent blocks (bias + trend + momentum + regime).
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
    long_votes = 0
    short_votes = 0
    long_votes += 1 if bias_score >= 56 else 0
    short_votes += 1 if bias_score <= 44 else 0
    long_votes += 1 if ema_trend_up else 0
    short_votes += 1 if ema_trend_down else 0
    long_votes += 1 if macd_confirm_long else 0
    short_votes += 1 if macd_confirm_short else 0
    long_votes += 1 if rsi_confirm_long else 0
    short_votes += 1 if rsi_confirm_short else 0
    long_votes += 1 if long_regime_confirms >= 2 else 0
    short_votes += 1 if short_regime_confirms >= 2 else 0

    scalp_direction = None
    if long_votes >= 4 and long_votes > short_votes:
        scalp_direction = "LONG"
    elif short_votes >= 4 and short_votes > long_votes:
        scalp_direction = "SHORT"

    if strict_mode and scalp_direction is not None:
        # Trend-strength gate.
        if pd.isna(adx_val) or adx_val < 20:
            return None, None, None, None, None, "No trend strength (ADX < 20)"

        if scalp_direction == "LONG":
            if long_regime_confirms < 2:
                return None, None, None, None, None, "Regime filters not aligned (need 2/3)"

            if bollinger_bias == "Overbought" and not volume_spike:
                return None, None, None, None, None, "Overbought"
            # Adaptive stoch range by trend quality.
            if not pd.isna(adx_val) and adx_val >= 25:
                stoch_lo, stoch_hi = 0.18, 0.88
            else:
                stoch_lo, stoch_hi = 0.20, 0.82
            if pd.isna(stochrsi_k_val) or not (stoch_lo <= stochrsi_k_val <= stoch_hi and rsi_confirm_long and macd_confirm_long):
                return None, None, None, None, None, "Momentum fail"

        elif scalp_direction == "SHORT":
            if short_regime_confirms < 2:
                return None, None, None, None, None, "Regime filters not aligned (need 2/3)"

            if bollinger_bias == "Oversold" and not volume_spike:
                return None, None, None, None, None, "Oversold"
            if not pd.isna(adx_val) and adx_val >= 25:
                stoch_lo, stoch_hi = 0.12, 0.82
            else:
                stoch_lo, stoch_hi = 0.15, 0.78
            if pd.isna(stochrsi_k_val) or not (stoch_lo <= stochrsi_k_val <= stoch_hi and rsi_confirm_short and macd_confirm_short):
                return None, None, None, None, None, "Momentum fail"

    recent = closed_df.tail(sr_lookback_fn(_inferred_tf))
    support = recent["low"].min()
    resistance = recent["high"].max()

    entry_s = stop_s = target_s = 0.0
    breakout_note = ""
    min_rr_strict = 1.50

    if scalp_direction == "LONG":
        entry_s = max(float(close_price), float(latest["ema5"])) + 0.20 * float(atr)
        # Keep scalp risk in a practical ATR band (avoid extreme wide/tight stops).
        stop_struct = float(support)
        stop_floor = entry_s - 1.80 * float(atr)  # widest allowed stop
        stop_ceil = entry_s - 0.60 * float(atr)   # tightest allowed stop
        stop_s = max(stop_struct, stop_floor)
        stop_s = min(stop_s, stop_ceil)

        # Require at least a minimum extension even if structure is close.
        target_s = max(float(resistance), entry_s + 1.40 * float(atr))
        if target_s > resistance:
            breakout_note = f"⚠️ Target (${target_s:.4f}) is above resistance (${resistance:.4f}). Breakout needed."
    elif scalp_direction == "SHORT":
        entry_s = min(float(close_price), float(latest["ema5"])) - 0.20 * float(atr)
        stop_struct = float(resistance)
        stop_floor = entry_s + 0.60 * float(atr)  # tightest allowed stop
        stop_ceil = entry_s + 1.80 * float(atr)   # widest allowed stop
        stop_s = min(stop_struct, stop_ceil)
        stop_s = max(stop_s, stop_floor)

        target_s = min(float(support), entry_s - 1.40 * float(atr))
        if target_s < support:
            breakout_note = f"⚠️ Target (${target_s:.4f}) is below support (${support:.4f}). Breakout needed."

    if stop_s <= 0.0 or target_s <= 0.0 or entry_s <= 0.0:
        return None, None, None, None, None, "Invalid plan levels"
    rr_ratio = abs(target_s - entry_s) / abs(entry_s - stop_s) if entry_s != stop_s else 0.0
    if strict_mode and rr_ratio < min_rr_strict:
        return None, None, None, None, None, f"Poor R:R ({rr_ratio:.2f} < {min_rr_strict:.2f})"
    return scalp_direction, entry_s, target_s, stop_s, rr_ratio, breakout_note
