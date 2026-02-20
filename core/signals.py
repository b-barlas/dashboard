from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd
import ta


def supertrend(
    high: pd.Series, low: pd.Series, close: pd.Series, length: int = 10, multiplier: float = 3.0
) -> pd.DataFrame:
    """SuperTrend indicator using ATR bands."""
    atr = ta.volatility.average_true_range(high, low, close, window=length)
    hl2 = (high + low) / 2
    upper_band = hl2 + multiplier * atr
    lower_band = hl2 - multiplier * atr

    st = pd.Series(np.nan, index=close.index)
    direction = pd.Series(1, index=close.index)

    for i in range(length, len(close)):
        if close.iloc[i] > upper_band.iloc[i - 1]:
            direction.iloc[i] = 1
        elif close.iloc[i] < lower_band.iloc[i - 1]:
            direction.iloc[i] = -1
        else:
            direction.iloc[i] = direction.iloc[i - 1]
            if direction.iloc[i] == 1 and lower_band.iloc[i] < lower_band.iloc[i - 1]:
                lower_band.iloc[i] = lower_band.iloc[i - 1]
            if direction.iloc[i] == -1 and upper_band.iloc[i] > upper_band.iloc[i - 1]:
                upper_band.iloc[i] = upper_band.iloc[i - 1]

        st.iloc[i] = lower_band.iloc[i] if direction.iloc[i] == 1 else upper_band.iloc[i]

    return pd.DataFrame(
        {
            f"SUPERT_{length}_{multiplier}": st,
            f"SUPERTd_{length}_{multiplier}": direction,
        },
        index=close.index,
    )


def sr_lookback(timeframe: str | None = None) -> int:
    """Return support/resistance lookback bars adapted to timeframe."""
    mapping = {"1m": 60, "3m": 50, "5m": 50, "15m": 40, "1h": 30, "4h": 20, "1d": 20}
    return mapping.get(timeframe or "", 30)


@dataclass
class AnalysisResult:
    signal: str = "NO DATA"
    leverage: int = 1
    comment: str = ""
    volume_spike: bool = False
    atr_comment: str = ""
    candle_pattern: str = ""
    confidence: float = 0.0
    adx: float = 0.0
    supertrend: str = ""
    ichimoku: str = ""
    stochrsi_k: float = 0.0
    bollinger: str = ""
    vwap: str = ""
    psar: str = ""
    williams: str = ""
    cci: str = ""


def detect_volume_spike(df: pd.DataFrame, window: int = 20, multiplier: float = 2.0) -> bool:
    if "volume" not in df.columns or len(df) < window + 1:
        return False
    recent_volumes = df["volume"].iloc[-(window + 1) : -1]
    avg_volume = recent_volumes.mean()
    last_volume = df["volume"].iloc[-1]
    return bool(last_volume > avg_volume * multiplier)


def detect_candle_pattern(df: pd.DataFrame) -> str:
    if df is None or len(df) < 5:
        return ""

    last = df.iloc[-1]
    prev = df.iloc[-2]
    prev2 = df.iloc[-3]

    body_last = abs(last["close"] - last["open"])
    body_prev = abs(prev["close"] - prev["open"])
    body_prev2 = abs(prev2["close"] - prev2["open"])

    if (
        prev["close"] < prev["open"]
        and last["close"] > last["open"]
        and last["close"] > prev["open"]
        and last["open"] < prev["close"]
    ):
        return "▲ Bullish Engulfing (strong reversal up)"

    if (
        prev["close"] > prev["open"]
        and last["close"] < last["open"]
        and last["open"] > prev["close"]
        and last["close"] < prev["open"]
    ):
        return "▼ Bearish Engulfing (strong reversal down)"

    lower_shadow = min(last["open"], last["close"]) - last["low"]
    upper_shadow = last["high"] - max(last["open"], last["close"])
    if body_last < lower_shadow and upper_shadow < lower_shadow * 0.5:
        return "▲ Hammer (bullish bottom wick)"
    if upper_shadow > 2 * body_last and lower_shadow < body_last:
        return "▲ Inverted Hammer (potential bottom reversal)"
    if lower_shadow > 2 * body_last and upper_shadow < body_last:
        return "▼ Hanging Man (possible top reversal)"
    if upper_shadow > 2 * body_last and lower_shadow < body_last and last["close"] < last["open"]:
        return "▼ Shooting Star (bearish top wick)"
    if body_last / (last["high"] - last["low"] + 1e-9) < 0.1:
        return "- Doji (market indecision)"

    if (
        prev2["close"] < prev2["open"]
        and body_prev < min(body_prev2, body_last)
        and last["close"] > last["open"]
        and last["close"] > ((prev2["open"] + prev2["close"]) / 2)
    ):
        return "▲ Morning Star (3-bar bullish reversal)"

    if (
        prev2["close"] > prev2["open"]
        and body_prev < min(body_prev2, body_last)
        and last["close"] < last["open"]
        and last["close"] < ((prev2["open"] + prev2["close"]) / 2)
    ):
        return "▼ Evening Star (3-bar bearish reversal)"

    if (
        prev["close"] < prev["open"]
        and last["open"] < prev["close"]
        and last["close"] > ((prev["open"] + prev["close"]) / 2)
        and last["close"] < prev["open"]
    ):
        return "▲ Piercing Line (mid-level reversal)"

    if (
        prev["close"] > prev["open"]
        and last["open"] > prev["close"]
        and last["close"] < ((prev["open"] + prev["close"]) / 2)
        and last["close"] > prev["open"]
    ):
        return "▼ Dark Cloud Cover (mid-level reversal)"

    if all(df.iloc[-i]["close"] > df.iloc[-i]["open"] for i in range(1, 4)):
        return "▲ Three White Soldiers (strong bullish confirmation)"
    if all(df.iloc[-i]["close"] < df.iloc[-i]["open"] for i in range(1, 4)):
        return "▼ Three Black Crows (strong bearish confirmation)"
    return ""


def analyse(df: pd.DataFrame, debug_fn: Callable[[str], None] | None = None) -> AnalysisResult:
    if df is None or len(df) < 55:
        return AnalysisResult(comment="Insufficient data")
    df = df.copy()
    debug = debug_fn or (lambda _msg: None)

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

    df["ema5"] = ta.trend.ema_indicator(df["close"], window=5)
    df["ema9"] = ta.trend.ema_indicator(df["close"], window=9)
    df["ema21"] = ta.trend.ema_indicator(df["close"], window=21)
    df["ema50"] = ta.trend.ema_indicator(df["close"], window=50)
    df["rsi"] = ta.momentum.rsi(df["close"], window=14)
    macd_ind = ta.trend.MACD(df["close"])
    df["macd"] = macd_ind.macd()
    df["macd_signal"] = macd_ind.macd_signal()
    df["macd_diff"] = macd_ind.macd_diff()
    df["obv"] = ta.volume.on_balance_volume(df["close"], df["volume"])
    df["atr"] = ta.volatility.average_true_range(df["high"], df["low"], df["close"], window=14)

    df["typical_price"] = (df["high"] + df["low"] + df["close"]) / 3
    _vwap_den = df["volume"].cumsum().replace(0, np.nan)
    df["vwap"] = (df["typical_price"] * df["volume"]).cumsum() / _vwap_den

    try:
        psar_ind = ta.trend.PSARIndicator(high=df["high"], low=df["low"], close=df["close"])
        psar_up = psar_ind.psar_up()
        psar_down = psar_ind.psar_down()
        df["psar"] = psar_up.fillna(psar_down)
    except Exception as e:
        debug(f"PSAR Error: {e}")
        df["psar"] = np.nan

    df["williams_r"] = ta.momentum.williams_r(df["high"], df["low"], df["close"], lbp=14)
    df["cci"] = ta.trend.cci(df["high"], df["low"], df["close"], window=20)

    ichimoku = ta.trend.IchimokuIndicator(
        high=df["high"], low=df["low"], window1=9, window2=26, window3=52, visual=False
    )
    df["tenkan"] = ichimoku.ichimoku_conversion_line()
    df["kijun"] = ichimoku.ichimoku_base_line()
    df["senkou_a"] = ichimoku.ichimoku_a()
    df["senkou_b"] = ichimoku.ichimoku_b()

    latest_idx = -2 if len(df) >= 2 else -1
    latest = df.iloc[latest_idx]

    vwap_val = latest.get("vwap", np.nan)
    if pd.isna(vwap_val):
        vwap_label = "Unavailable"
    elif latest["close"] > vwap_val:
        vwap_label = "🟢 Above"
    elif latest["close"] < vwap_val:
        vwap_label = "🔴 Below"
    else:
        vwap_label = "→ Near VWAP"

    volume_spike = detect_volume_spike(df)
    candle_pattern = detect_candle_pattern(df)

    atr_latest = latest["atr"]
    if atr_latest > latest["close"] * 0.05:
        atr_comment = "▲ High"
    elif atr_latest < latest["close"] * 0.02:
        atr_comment = "▼ Low"
    else:
        atr_comment = "– Moderate"

    try:
        df["adx"] = ta.trend.adx(df["high"], df["low"], df["close"], window=14)
    except Exception as e:
        debug(f"ADX Error: {e}")
        df["adx"] = np.nan
    # Read ADX from the updated frame instead of the stale `latest` snapshot.
    adx_val = float(df["adx"].iloc[latest_idx]) if pd.notna(df["adx"].iloc[latest_idx]) else np.nan

    try:
        st_data = supertrend(df["high"], df["low"], df["close"], length=10, multiplier=3.0)
        df["supertrend"] = st_data[st_data.columns[0]]
    except Exception as e:
        debug(f"SuperTrend Error: {e}")
        df["supertrend"] = np.nan

    stoch_rsi = ta.momentum.StochRSIIndicator(close=df["close"], window=14, smooth1=3, smooth2=3)
    df["stochrsi_k"] = stoch_rsi.stochrsi_k()

    df["bb_mid"] = df["close"].rolling(window=20).mean()
    df["bb_std"] = df["close"].rolling(window=20).std(ddof=0)
    df["bb_upper"] = df["bb_mid"] + 2 * df["bb_std"]
    df["bb_lower"] = df["bb_mid"] - 2 * df["bb_std"]

    latest = df.iloc[latest_idx]
    cci_val = latest["cci"]
    stochrsi_k_val = latest.get("stochrsi_k", np.nan)
    williams_val = latest["williams_r"]
    psar_val = latest.get("psar", np.nan)
    bb_upper = latest["bb_upper"]
    bb_lower = latest["bb_lower"]
    bb_range = bb_upper - bb_lower
    bb_buffer = bb_range * 0.01
    close_price = latest["close"]

    trend_signals = []
    if latest["ema5"] > latest["ema21"] > latest["ema50"]:
        trend_signals.append(1.0)
    elif latest["ema5"] < latest["ema21"] < latest["ema50"]:
        trend_signals.append(-1.0)
    elif latest["ema5"] > latest["ema21"]:
        trend_signals.append(0.5)
    elif latest["ema5"] < latest["ema21"]:
        trend_signals.append(-0.5)
    else:
        trend_signals.append(0.0)

    supertrend_trend = "Unavailable"
    st_val = latest.get("supertrend", np.nan)
    if pd.notna(st_val):
        if latest["close"] > st_val:
            trend_signals.append(0.5)
            supertrend_trend = "Bullish"
        elif latest["close"] < st_val:
            trend_signals.append(-0.5)
            supertrend_trend = "Bearish"

    try:
        sa = latest.get("senkou_a", np.nan)
        sb = latest.get("senkou_b", np.nan)
        if pd.isna(sa) or pd.isna(sb):
            ichimoku_trend = "Unavailable"
        else:
            if latest["close"] > max(sa, sb):
                trend_signals.append(0.5)
                ichimoku_trend = "Bullish"
            elif latest["close"] < min(sa, sb):
                trend_signals.append(-0.5)
                ichimoku_trend = "Bearish"
            else:
                ichimoku_trend = "Neutral"
    except Exception:
        ichimoku_trend = "Unavailable"

    psar_trend = ""
    if pd.notna(psar_val):
        if latest["close"] > psar_val:
            trend_signals.append(0.3)
            psar_trend = "▲ Bullish"
        elif latest["close"] < psar_val:
            trend_signals.append(-0.3)
            psar_trend = "▼ Bearish"

    if adx_val >= 25:
        trend_strength = min(adx_val / 50, 1.0)
        trend_signals = [s * (1 + trend_strength * 0.5) for s in trend_signals]

    trend_score = np.clip(np.mean(trend_signals) if trend_signals else 0.0, -1, 1)

    momentum_signals = []
    rsi_val = latest["rsi"]
    if rsi_val > 70:
        momentum_signals.append(-0.5)
    elif rsi_val > 55:
        momentum_signals.append(0.5)
    elif rsi_val < 30:
        momentum_signals.append(0.5)
    elif rsi_val < 45:
        momentum_signals.append(-0.5)
    else:
        momentum_signals.append(0.0)

    if latest["macd"] > latest["macd_signal"] and latest["macd_diff"] > 0:
        momentum_signals.append(1.0)
    elif latest["macd"] < latest["macd_signal"] and latest["macd_diff"] < 0:
        momentum_signals.append(-1.0)
    elif latest["macd"] > latest["macd_signal"]:
        momentum_signals.append(0.5)
    elif latest["macd"] < latest["macd_signal"]:
        momentum_signals.append(-0.5)

    if stochrsi_k_val >= 0.9:
        momentum_signals.append(-0.5)
    elif stochrsi_k_val <= 0.1:
        momentum_signals.append(0.5)
    elif stochrsi_k_val >= 0.8:
        momentum_signals.append(-0.3)
    elif stochrsi_k_val <= 0.2:
        momentum_signals.append(0.3)

    if williams_val < -80:
        momentum_signals.append(0.5)
    elif williams_val > -20:
        momentum_signals.append(-0.5)

    if cci_val > 100:
        momentum_signals.append(-0.5)
    elif cci_val < -100:
        momentum_signals.append(0.5)

    momentum_score = np.clip(np.mean(momentum_signals) if momentum_signals else 0.0, -1, 1)

    volume_signals = []
    _obv_back = min(5, len(df) - 1)
    if _obv_back > 0:
        if df["obv"].iloc[-1] > df["obv"].iloc[-_obv_back]:
            volume_signals.append(0.5)
        elif df["obv"].iloc[-1] < df["obv"].iloc[-_obv_back]:
            volume_signals.append(-0.5)

    if volume_spike:
        volume_signals.append(0.5 if latest["close"] > latest["open"] else -0.5)

    if latest["close"] > latest["vwap"]:
        volume_signals.append(0.5)
    elif latest["close"] < latest["vwap"]:
        volume_signals.append(-0.5)

    volume_score = np.clip(np.mean(volume_signals) if volume_signals else 0.0, -1, 1)

    volatility_signals = []
    atr_ratio = atr_latest / latest["close"]
    if atr_ratio < 0.015:
        volatility_signals.append(0.5)
    elif atr_ratio > 0.05:
        volatility_signals.append(-0.5)
    bb_width_pct = bb_range / latest["close"]
    if bb_width_pct < 0.05:
        volatility_signals.append(0.5)
    elif bb_width_pct > 0.15:
        volatility_signals.append(-0.5)
    volatility_score = np.clip(np.mean(volatility_signals) if volatility_signals else 0.0, -1, 1)

    final_score = trend_score * 0.40 + momentum_score * 0.30 + volume_score * 0.20 + volatility_score * 0.10
    confidence_score = float(np.clip(round((final_score + 1) / 2 * 100, 1), 0, 100))

    if not pd.isna(adx_val) and adx_val < 20:
        _regime_discount = np.interp(adx_val, [0, 20], [0.70, 1.0])
        confidence_score = float(np.clip(round(confidence_score * _regime_discount, 1), 0, 100))

    if volatility_score < -0.3:
        buy_threshold, sell_threshold = 70, 30
    elif adx_val < 20:
        buy_threshold, sell_threshold = 75, 25
    else:
        buy_threshold, sell_threshold = 65, 35

    if confidence_score >= buy_threshold:
        base_signal = "BUY"
    elif confidence_score <= sell_threshold:
        base_signal = "SELL"
    else:
        base_signal = "WAIT"

    if base_signal == "BUY":
        if trend_score < 0.2:
            signal, comment = "WAIT", "⏳ Bullish setup incomplete. Trend not confirmed."
        elif volume_score < 0:
            signal, comment = "WAIT", "⏳ Bullish setup needs volume confirmation."
        elif momentum_score < -0.3:
            signal, comment = "WAIT", "⏳ Bullish setup but momentum divergence detected."
        elif volatility_score < -0.5:
            signal, comment = "WAIT", "⚠️ High volatility detected. Wait for calmer conditions."
        elif confidence_score >= 80:
            signal, comment = "STRONG BUY", "🚀 Strong bullish bias. High confidence to go LONG."
        else:
            signal, comment = "BUY", "📈 Bullish leaning. Consider LONG entry."
    elif base_signal == "SELL":
        if trend_score > -0.2:
            signal, comment = "WAIT", "⏳ Bearish setup incomplete. Trend not confirmed."
        elif volume_score > 0:
            signal, comment = "WAIT", "⏳ Bearish setup needs volume confirmation."
        elif momentum_score > 0.3:
            signal, comment = "WAIT", "⏳ Bearish setup but momentum divergence detected."
        elif volatility_score < -0.5:
            signal, comment = "WAIT", "⚠️ High volatility detected. Wait for calmer conditions."
        elif confidence_score <= 20:
            signal, comment = "STRONG SELL", "⚠️ Strong bearish bias. SHORT with high confidence."
        else:
            signal, comment = "SELL", "📉 Bearish leaning. SHORT may be considered."
    else:
        signal = "WAIT"
        if abs(trend_score) < 0.1:
            comment = "⏳ No clear trend direction. Market ranging."
        elif abs(momentum_score) < 0.1:
            comment = "⏳ Weak momentum. Wait for stronger signals."
        elif volatility_score < -0.5:
            comment = "⚠️ High volatility. Risky conditions."
        else:
            comment = "⏳ Mixed signals. No clear direction."

    if np.isnan(williams_val):
        williams_label = ""
    elif williams_val < -80:
        williams_label = "🟢 Oversold"
    elif williams_val > -20:
        williams_label = "🔴 Overbought"
    else:
        williams_label = "🟡 Neutral"

    cci_label = "🟡 Neutral"
    if cci_val > 100:
        cci_label = "🔴 Overbought"
    elif cci_val < -100:
        cci_label = "🟢 Oversold"

    bollinger_bias = "→ Neutral"
    if close_price > bb_upper + bb_buffer:
        bollinger_bias = "🔴 Overbought"
    elif close_price > bb_upper:
        bollinger_bias = "→ Near Top"
    elif close_price < bb_lower - bb_buffer:
        bollinger_bias = "🟢 Oversold"
    elif close_price < bb_lower:
        bollinger_bias = "→ Near Bottom"

    bollinger_width = df["bb_upper"].iloc[-1] - df["bb_lower"].iloc[-1]
    volatility_factor = min(bollinger_width / max(latest["close"], 1e-9), 0.1)
    rsi_factor = 0.1 if latest["rsi"] > 70 or latest["rsi"] < 30 else 0
    _obv_back_lev = min(5, len(df) - 1)
    obv_factor = (
        0.1
        if (
            _obv_back_lev > 0
            and df["obv"].iloc[-1] > df["obv"].iloc[-_obv_back_lev]
            and latest["close"] > latest["ema21"]
        )
        else 0
    )
    recent = df.tail(sr_lookback(_inferred_tf))
    support = recent["low"].min()
    resistance = recent["high"].max()
    current_price = latest["close"]
    sr_factor = (
        0.1
        if abs(current_price - support) / current_price < 0.02
        or abs(current_price - resistance) / current_price < 0.02
        else 0
    )
    risk_score = volatility_factor + rsi_factor + obv_factor + sr_factor

    if risk_score <= 0.15:
        lev_base = int(round(np.interp(risk_score, [0.00, 0.15], [10, 6])))
    elif risk_score <= 0.25:
        lev_base = int(round(np.interp(risk_score, [0.15, 0.25], [6, 4])))
    else:
        rs = min(risk_score, 0.40)
        lev_base = int(round(np.interp(rs, [0.25, 0.40], [4, 2])))

    if confidence_score < 50:
        lev_base = min(lev_base, 2)
    elif confidence_score < 65:
        lev_base = min(lev_base, 4)
    elif confidence_score < 75:
        lev_base = min(lev_base, 6)

    return AnalysisResult(
        signal=signal,
        leverage=lev_base,
        comment=comment,
        volume_spike=volume_spike,
        atr_comment=atr_comment,
        candle_pattern=candle_pattern,
        confidence=confidence_score,
        adx=adx_val,
        supertrend=supertrend_trend,
        ichimoku=ichimoku_trend,
        stochrsi_k=stochrsi_k_val,
        bollinger=bollinger_bias,
        vwap=vwap_label,
        psar=psar_trend,
        williams=williams_label,
        cci=cci_label,
    )
