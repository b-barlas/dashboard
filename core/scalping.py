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
    df["ema5"] = df["close"].ewm(span=5).mean()
    df["ema13"] = df["close"].ewm(span=13).mean()
    df["ema21"] = df["close"].ewm(span=21).mean()
    df["atr"] = ta.volatility.average_true_range(df["high"], df["low"], df["close"], window=14)
    df["rsi"] = ta.momentum.rsi(df["close"], window=14)

    macd_ind = ta.trend.MACD(df["close"])
    df["macd"] = macd_ind.macd()
    df["macd_signal"] = macd_ind.macd_signal()
    df["macd_diff"] = macd_ind.macd_diff()
    adx_val = float(ta.trend.adx(df["high"], df["low"], df["close"], window=14).iloc[-1])

    stoch = ta.momentum.StochRSIIndicator(close=df["close"], window=14, smooth1=3, smooth2=3)
    stochrsi_k_val = float(stoch.stochrsi_k().iloc[-1])

    bb = ta.volatility.BollingerBands(close=df["close"], window=20, window_dev=2)
    last_close = df["close"].iloc[-1]
    hband = float(bb.bollinger_hband().iloc[-1])
    lband = float(bb.bollinger_lband().iloc[-1])
    if last_close > hband:
        bollinger_bias = "Overbought"
    elif last_close < lband:
        bollinger_bias = "Oversold"
    else:
        bollinger_bias = "Neutral"

    latest = df.iloc[-1]
    close_price = latest["close"]
    atr = latest["atr"]

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
            if not (0.20 <= stochrsi_k_val <= 0.85 and rsi_confirm_long and macd_confirm):
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
            if not (0.15 <= stochrsi_k_val <= 0.80 and rsi_confirm_short and not macd_confirm):
                return None, None, None, None, None, "Momentum fail"

        if (atr / close_price) < 0.0015:
            return None, None, None, None, None, "Low Volatility"

    recent = df.tail(sr_lookback_fn())
    support = recent["low"].min()
    resistance = recent["high"].max()

    entry_s = stop_s = target_s = 0.0
    breakout_note = ""

    if scalp_direction == "LONG":
        entry_s = max(close_price, latest["ema5"]) + 0.25 * atr
        stop_s = close_price - 0.75 * atr
        target_s = entry_s + 1.5 * atr
        if target_s > resistance:
            breakout_note = f"⚠️ Target (${target_s:.4f}) is above resistance (${resistance:.4f}). Breakout needed."
    elif scalp_direction == "SHORT":
        entry_s = min(close_price, latest["ema5"]) - 0.25 * atr
        stop_s = close_price + 0.75 * atr
        target_s = entry_s - 1.5 * atr
        if target_s < support:
            breakout_note = f"⚠️ Target (${target_s:.4f}) is below support (${support:.4f}). Breakout needed."

    rr_ratio = abs(target_s - entry_s) / abs(entry_s - stop_s) if entry_s != stop_s else 0.0
    return scalp_direction, entry_s, target_s, stop_s, rr_ratio, breakout_note

