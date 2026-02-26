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

    @property
    def bias(self) -> float:
        """Alias kept for dashboard terminology compatibility."""
        return float(self.confidence)


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

    # Timeframe-adaptive oscillator thresholds.
    if _inferred_tf in {"1m", "3m", "5m"}:
        rsi_ob, rsi_os = 75.0, 25.0
        stoch_ob_ext, stoch_ob_soft = 0.92, 0.85
        stoch_os_soft, stoch_os_ext = 0.15, 0.08
        williams_ob, williams_os = -15.0, -85.0
        cci_ob, cci_os = 130.0, -130.0
    elif _inferred_tf in {"4h", "1d"}:
        rsi_ob, rsi_os = 68.0, 32.0
        stoch_ob_ext, stoch_ob_soft = 0.88, 0.78
        stoch_os_soft, stoch_os_ext = 0.22, 0.12
        williams_ob, williams_os = -22.0, -78.0
        cci_ob, cci_os = 90.0, -90.0
    else:
        rsi_ob, rsi_os = 70.0, 30.0
        stoch_ob_ext, stoch_ob_soft = 0.90, 0.80
        stoch_os_soft, stoch_os_ext = 0.20, 0.10
        williams_ob, williams_os = -20.0, -80.0
        cci_ob, cci_os = 100.0, -100.0

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

    # Standard policy: callers pass a closed-candle frame (live candle removed upstream).
    # Therefore latest bar for analysis is always the last row.
    latest_idx = -1
    latest_pos = len(df) - 1
    latest = df.iloc[latest_idx]
    closed_df = df

    vwap_val = latest.get("vwap", np.nan)
    if pd.isna(vwap_val):
        vwap_label = "Unavailable"
    elif latest["close"] > vwap_val:
        vwap_label = "🟢 Above"
    elif latest["close"] < vwap_val:
        vwap_label = "🔴 Below"
    else:
        vwap_label = "→ Near VWAP"

    volume_spike = detect_volume_spike(closed_df)
    candle_pattern = detect_candle_pattern(closed_df)

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

    # Trend slope confirmation helps reduce whipsaw when level-only signals are mixed.
    if latest_pos >= 3:
        ema9_prev = float(df["ema9"].iloc[latest_pos - 3])
        ema9_now = float(df["ema9"].iloc[latest_pos])
        ema21_prev = float(df["ema21"].iloc[latest_pos - 3])
        ema21_now = float(df["ema21"].iloc[latest_pos])
        if np.isfinite(ema9_prev) and np.isfinite(ema9_now) and abs(ema9_prev) > 1e-12:
            ema9_slope = (ema9_now / ema9_prev) - 1.0
            if ema9_slope > 0.001:
                trend_signals.append(0.25)
            elif ema9_slope < -0.001:
                trend_signals.append(-0.25)
        if np.isfinite(ema21_prev) and np.isfinite(ema21_now) and abs(ema21_prev) > 1e-12:
            ema21_slope = (ema21_now / ema21_prev) - 1.0
            if ema21_slope > 0.0007:
                trend_signals.append(0.2)
            elif ema21_slope < -0.0007:
                trend_signals.append(-0.2)

    if adx_val >= 25:
        trend_strength = min(adx_val / 50, 1.0)
        trend_signals = [s * (1 + trend_strength * 0.5) for s in trend_signals]

    trend_score = np.clip(np.mean(trend_signals) if trend_signals else 0.0, -1, 1)

    momentum_signals = []
    trend_hint = 1 if latest["ema5"] > latest["ema21"] else (-1 if latest["ema5"] < latest["ema21"] else 0)
    strong_trend = bool(pd.notna(adx_val) and adx_val >= 25)
    rsi_val = latest["rsi"]
    if rsi_val > rsi_ob:
        momentum_signals.append(-0.2 if strong_trend and trend_hint > 0 else -0.5)
    elif rsi_val > 55:
        momentum_signals.append(0.5)
    elif rsi_val < rsi_os:
        momentum_signals.append(0.2 if strong_trend and trend_hint < 0 else 0.5)
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

    if latest_pos >= 1:
        macd_now = float(df["macd_diff"].iloc[latest_pos])
        macd_prev = float(df["macd_diff"].iloc[latest_pos - 1])
        if np.isfinite(macd_now) and np.isfinite(macd_prev):
            macd_slope = macd_now - macd_prev
            if macd_slope > 0:
                momentum_signals.append(0.2)
            elif macd_slope < 0:
                momentum_signals.append(-0.2)

    if stochrsi_k_val >= stoch_ob_ext:
        momentum_signals.append(-0.5)
    elif stochrsi_k_val <= stoch_os_ext:
        momentum_signals.append(0.5)
    elif stochrsi_k_val >= stoch_ob_soft:
        momentum_signals.append(-0.3)
    elif stochrsi_k_val <= stoch_os_soft:
        momentum_signals.append(0.3)

    if williams_val < williams_os:
        momentum_signals.append(0.5)
    elif williams_val > williams_ob:
        momentum_signals.append(-0.5)

    if cci_val > cci_ob:
        momentum_signals.append(-0.5)
    elif cci_val < cci_os:
        momentum_signals.append(0.5)

    momentum_score = np.clip(np.mean(momentum_signals) if momentum_signals else 0.0, -1, 1)

    volume_signals = []
    _obv_back = min(5, len(closed_df) - 1)
    if _obv_back > 0:
        if closed_df["obv"].iloc[-1] > closed_df["obv"].iloc[-_obv_back]:
            volume_signals.append(0.5)
        elif closed_df["obv"].iloc[-1] < closed_df["obv"].iloc[-_obv_back]:
            volume_signals.append(-0.5)

    if volume_spike:
        volume_signals.append(0.5 if latest["close"] > latest["open"] else -0.5)

    vol_ma = float(closed_df["volume"].tail(20).mean()) if len(closed_df) >= 20 else float(closed_df["volume"].mean())
    vol_ratio = float(latest["volume"] / max(vol_ma, 1e-9))
    if vol_ratio >= 1.5:
        volume_signals.append(0.3 if latest["close"] >= latest["open"] else -0.3)
    elif vol_ratio <= 0.7:
        volume_signals.append(-0.2 if latest["close"] >= latest["open"] else 0.2)

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

    # Regime-aware category weights improve robustness across trend vs range environments.
    if pd.notna(adx_val) and adx_val >= 25:
        w_trend, w_momentum, w_volume, w_volatility = 0.50, 0.25, 0.15, 0.10
    elif pd.notna(adx_val) and adx_val < 18:
        w_trend, w_momentum, w_volume, w_volatility = 0.30, 0.35, 0.20, 0.15
    else:
        w_trend, w_momentum, w_volume, w_volatility = 0.40, 0.30, 0.20, 0.10

    final_score = (
        trend_score * w_trend
        + momentum_score * w_momentum
        + volume_score * w_volume
        + volatility_score * w_volatility
    )

    # Penalize setups where trend and momentum strongly disagree.
    if abs(trend_score) >= 0.45 and abs(momentum_score) >= 0.35 and np.sign(trend_score) != np.sign(momentum_score):
        final_score *= 0.85

    # Small confirmation boost when trend and volume align in direction.
    if abs(trend_score) >= 0.35 and abs(volume_score) >= 0.30 and np.sign(trend_score) == np.sign(volume_score):
        final_score += 0.05 * float(np.sign(trend_score))

    final_score = float(np.clip(final_score, -1.0, 1.0))
    bias_score = float(np.clip(round((final_score + 1) / 2 * 100, 1), 0, 100))

    if not pd.isna(adx_val) and adx_val < 20:
        _regime_discount = np.interp(adx_val, [0, 20], [0.70, 1.0])
        # Preserve neutral center (50) while damping directional extremity in weak trends.
        bias_score = float(np.clip(round(50.0 + (bias_score - 50.0) * _regime_discount, 1), 0, 100))

    if volatility_score < -0.35:
        buy_threshold, sell_threshold = 72, 28
    elif pd.notna(adx_val) and adx_val >= 30:
        buy_threshold, sell_threshold = 62, 38
    elif pd.notna(adx_val) and adx_val < 18:
        buy_threshold, sell_threshold = 76, 24
    else:
        buy_threshold, sell_threshold = 66, 34

    if bias_score >= buy_threshold:
        base_signal = "BUY"
    elif bias_score <= sell_threshold:
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
        elif bias_score >= 80:
            signal, comment = "STRONG BUY", "🚀 Strong bullish bias. Directional edge supports LONG."
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
        elif bias_score <= 20:
            signal, comment = "STRONG SELL", "⚠️ Strong bearish bias. Directional edge supports SHORT."
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
    elif williams_val < williams_os:
        williams_label = "🟢 Oversold"
    elif williams_val > williams_ob:
        williams_label = "🔴 Overbought"
    else:
        williams_label = "🟡 Neutral"

    cci_label = "🟡 Neutral"
    if cci_val > cci_ob:
        cci_label = "🔴 Overbought"
    elif cci_val < cci_os:
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

    bollinger_width = df["bb_upper"].iloc[latest_idx] - df["bb_lower"].iloc[latest_idx]
    volatility_factor = min(bollinger_width / max(latest["close"], 1e-9), 0.1)
    rsi_factor = 0.1 if latest["rsi"] > rsi_ob or latest["rsi"] < rsi_os else 0
    _obv_back_lev = min(5, len(closed_df) - 1)
    obv_factor = (
        0.1
        if (
            _obv_back_lev > 0
            and closed_df["obv"].iloc[-1] > closed_df["obv"].iloc[-_obv_back_lev]
            and latest["close"] > latest["ema21"]
        )
        else 0
    )
    recent = closed_df.tail(sr_lookback(_inferred_tf))
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

    if bias_score < 50:
        lev_base = min(lev_base, 2)
    elif bias_score < 65:
        lev_base = min(lev_base, 4)
    elif bias_score < 75:
        lev_base = min(lev_base, 6)

    return AnalysisResult(
        signal=signal,
        leverage=lev_base,
        comment=comment,
        volume_spike=volume_spike,
        atr_comment=atr_comment,
        candle_pattern=candle_pattern,
        confidence=bias_score,
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
