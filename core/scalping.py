"""Scalping entry/target decision logic."""

from __future__ import annotations

from typing import Callable

import pandas as pd
import ta


def get_scalping_entry_target(
    df: pd.DataFrame,
    confidence_score: float,
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
    macd_confirm = latest["macd"] > latest["macd_signal"] and latest["macd_diff"] > 0
    rsi_confirm_long = latest["rsi"] > 55
    rsi_confirm_short = latest["rsi"] < 45

    scalp_direction = None
    if confidence_score >= 65 and ema_trend_up and macd_confirm and rsi_confirm_long:
        scalp_direction = "LONG"
    elif confidence_score <= 35 and ema_trend_down and not macd_confirm and rsi_confirm_short:
        scalp_direction = "SHORT"

    if strict_mode and scalp_direction is not None:
        if (pd.isna(adx_val) or adx_val < 20) and not volume_spike:
            return None, None, None, None, None, "No trend strength / no volume"

        if scalp_direction == "LONG":
            confirms = 0
            confirms += 1 if supertrend_trend == "Bullish" else 0
            confirms += 1 if ichimoku_trend == "Bullish" else 0
            confirms += 1 if vwap_label == "🟢 Above" else 0
            if confirms < 2:
                return None, None, None, None, None, "Regime filters not aligned (need 2/3)"

            if bollinger_bias == "Overbought":
                return None, None, None, None, None, "Overbought"
            if pd.isna(stochrsi_k_val) or not (0.20 <= stochrsi_k_val <= 0.85 and rsi_confirm_long and macd_confirm):
                return None, None, None, None, None, "Momentum fail"

        elif scalp_direction == "SHORT":
            confirms = 0
            confirms += 1 if supertrend_trend == "Bearish" else 0
            confirms += 1 if ichimoku_trend == "Bearish" else 0
            confirms += 1 if vwap_label == "🔴 Below" else 0
            if confirms < 2:
                return None, None, None, None, None, "Regime filters not aligned (need 2/3)"

            if bollinger_bias == "Oversold":
                return None, None, None, None, None, "Oversold"
            if pd.isna(stochrsi_k_val) or not (0.15 <= stochrsi_k_val <= 0.80 and rsi_confirm_short and not macd_confirm):
                return None, None, None, None, None, "Momentum fail"

        if (atr / close_price) < 0.0015:
            return None, None, None, None, None, "Low Volatility"

    recent = closed_df.tail(sr_lookback_fn(_inferred_tf))
    support = recent["low"].min()
    resistance = recent["high"].max()

    entry_s = stop_s = target_s = 0.0
    breakout_note = ""

    if scalp_direction == "LONG":
        entry_s = max(float(close_price), float(latest["ema5"])) + 0.20 * float(atr)
        stop_s = min(float(support), entry_s - 1.00 * float(atr))
        target_s = max(float(resistance), entry_s + 1.50 * float(atr))
        if target_s > resistance:
            breakout_note = f"⚠️ Target (${target_s:.4f}) is above resistance (${resistance:.4f}). Breakout needed."
    elif scalp_direction == "SHORT":
        entry_s = min(float(close_price), float(latest["ema5"])) - 0.20 * float(atr)
        stop_s = max(float(resistance), entry_s + 1.00 * float(atr))
        target_s = min(float(support), entry_s - 1.50 * float(atr))
        if target_s < support:
            breakout_note = f"⚠️ Target (${target_s:.4f}) is below support (${support:.4f}). Breakout needed."

    if stop_s <= 0.0 or target_s <= 0.0 or entry_s <= 0.0:
        return None, None, None, None, None, "Invalid plan levels"
    rr_ratio = abs(target_s - entry_s) / abs(entry_s - stop_s) if entry_s != stop_s else 0.0
    return scalp_direction, entry_s, target_s, stop_s, rr_ratio, breakout_note
