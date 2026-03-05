"""Advanced analysis primitives extracted from the service layer."""

from __future__ import annotations

import numpy as np
import pandas as pd
import ta


def dedupe_divergences(divergences: list[dict]) -> list[dict]:
    """Keep one strongest row per divergence type to avoid over-counting."""
    if not divergences:
        return []
    strength_rank = {"STRONG": 3, "MODERATE": 2, "WEAK": 1}
    picked: dict[str, dict] = {}
    for row in divergences:
        d_type = str((row or {}).get("type", "")).strip().upper()
        if not d_type:
            d_type = str((row or {}).get("description", "")).strip().upper() or "UNKNOWN"
        strength = str((row or {}).get("strength", "MODERATE")).strip().upper()
        rank = int(strength_rank.get(strength, 0))
        prev = picked.get(d_type)
        if prev is None or rank > int(prev.get("_rank", -1)):
            out = dict(row or {})
            out["_rank"] = rank
            picked[d_type] = out
    ordered = sorted(picked.values(), key=lambda x: (-int(x.get("_rank", 0)), str(x.get("type", ""))))
    for row in ordered:
        row.pop("_rank", None)
    return ordered


def calculate_fibonacci_levels(df: pd.DataFrame, lookback: int = 100) -> dict:
    if df is None or len(df) < 20:
        return {}

    data = df.tail(min(lookback, len(df)))
    swing_high = data["high"].max()
    swing_low = data["low"].min()
    swing_range = swing_high - swing_low
    if swing_range <= 0:
        return {}

    high_idx = data["high"].idxmax()
    low_idx = data["low"].idxmin()
    is_uptrend = low_idx < high_idx

    levels = {}
    fib_ratios = [0, 0.236, 0.382, 0.5, 0.618, 0.786, 1.0]
    fib_names = ["0%", "23.6%", "38.2%", "50%", "61.8%", "78.6%", "100%"]
    for ratio, name in zip(fib_ratios, fib_names):
        if is_uptrend:
            price = swing_high - swing_range * ratio
        else:
            price = swing_low + swing_range * ratio
        levels[name] = price

    ext_ratios = [1.272, 1.618, 2.0, 2.618]
    ext_names = ["127.2%", "161.8%", "200%", "261.8%"]
    for ratio, name in zip(ext_ratios, ext_names):
        ext = ratio - 1.0
        if is_uptrend:
            # Uptrend extensions are projected above swing high.
            price = swing_high + swing_range * ext
        else:
            # Downtrend extensions are projected below swing low.
            price = swing_low - swing_range * ext
        levels[name] = price

    levels["_swing_high"] = swing_high
    levels["_swing_low"] = swing_low
    levels["_is_uptrend"] = is_uptrend
    return levels


def monte_carlo_simulation(df: pd.DataFrame, num_simulations: int = 500, num_days: int = 30) -> dict:
    if df is None or len(df) < 30:
        return {}
    num_simulations = int(max(50, min(20000, num_simulations)))
    num_days = int(max(1, min(3650, num_days)))

    close = pd.to_numeric(df.get("close"), errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    close = close[close > 0]
    if len(close) < 30:
        return {}
    # Use closed-candle anchor when possible to avoid partial-candle noise.
    if len(close) > 40:
        close_eval = close.iloc[:-1].copy()
    else:
        close_eval = close.copy()

    returns = close_eval.pct_change().replace([np.inf, -np.inf], np.nan).dropna()
    if len(returns) < 20:
        return {}
    returns = returns.clip(lower=-0.95, upper=0.95)
    mu = float(returns.mean())
    sigma = float(returns.std(ddof=1))
    last_price = float(close_eval.iloc[-1])
    if last_price <= 0:
        return {}

    rng = np.random.default_rng()
    hist = returns.to_numpy(dtype=float)
    # Robust path generation: blend empirical bootstrap shocks with Gaussian shocks.
    boot = rng.choice(hist, size=(num_simulations, num_days), replace=True)
    gauss = rng.normal(loc=mu, scale=max(sigma, 1e-9), size=(num_simulations, num_days))
    daily_returns = np.clip(0.70 * boot + 0.30 * gauss, -0.95, 0.95)

    simulations = last_price * np.cumprod(1.0 + daily_returns, axis=1)

    final_prices = simulations[:, -1]
    term_rets = final_prices / last_price - 1.0
    var_95 = float(np.percentile(term_rets, 5))
    tail = term_rets[term_rets <= var_95]
    cvar_95 = float(tail.mean()) if len(tail) > 0 else var_95
    return {
        "simulations": simulations,
        "last_price": last_price,
        "mean_price": float(np.mean(final_prices)),
        "median_price": float(np.median(final_prices)),
        "p5": float(np.percentile(final_prices, 5)),
        "p25": float(np.percentile(final_prices, 25)),
        "p75": float(np.percentile(final_prices, 75)),
        "p95": float(np.percentile(final_prices, 95)),
        "min_price": float(np.min(final_prices)),
        "max_price": float(np.max(final_prices)),
        "prob_profit": float(np.mean(final_prices > last_price)),
        "expected_return": float((np.mean(final_prices) - last_price) / last_price * 100),
        "var_95": float(var_95 * 100),
        "cvar_95": float(cvar_95 * 100),
    }


def detect_divergence(df: pd.DataFrame) -> list[dict]:
    if df is None or len(df) < 30:
        return []

    divergences = []
    df = df.copy()
    df["rsi"] = ta.momentum.rsi(df["close"], window=14)
    macd_ind = ta.trend.MACD(df["close"])
    df["macd"] = macd_ind.macd()

    lookback = min(20, len(df) - 5)
    recent = df.tail(lookback)
    for i in range(2, len(recent) - 2):
        prev1 = recent.iloc[i - 1]
        curr = recent.iloc[i]
        next1 = recent.iloc[i + 1] if i + 1 < len(recent) else curr

        if curr["low"] < prev1["low"] and curr["low"] < next1["low"]:
            for j in range(max(0, i - 10), i - 2):
                prev_low = recent.iloc[j]
                if prev_low["low"] < recent.iloc[max(0, j - 1)]["low"]:
                    if curr["close"] < prev_low["close"] and curr["rsi"] > prev_low["rsi"]:
                        divergences.append(
                            {
                                "type": "BULLISH RSI",
                                "description": "Price making lower lows but RSI making higher lows",
                                "strength": "STRONG",
                            }
                        )
                        break

        if curr["high"] > prev1["high"] and curr["high"] > next1["high"]:
            for j in range(max(0, i - 10), i - 2):
                prev_high = recent.iloc[j]
                if prev_high["high"] > recent.iloc[max(0, j - 1)]["high"]:
                    if curr["close"] > prev_high["close"] and curr["rsi"] < prev_high["rsi"]:
                        divergences.append(
                            {
                                "type": "BEARISH RSI",
                                "description": "Price making higher highs but RSI making lower highs",
                                "strength": "STRONG",
                            }
                        )
                        break

    last_5 = df.tail(5)
    if len(last_5) >= 5:
        price_trend = last_5["close"].iloc[-1] - last_5["close"].iloc[0]
        macd_trend = last_5["macd"].iloc[-1] - last_5["macd"].iloc[0]
        if price_trend > 0 and macd_trend < 0:
            divergences.append(
                {
                    "type": "BEARISH MACD",
                    "description": "Price rising but MACD declining — momentum weakening",
                    "strength": "MODERATE",
                }
            )
        elif price_trend < 0 and macd_trend > 0:
            divergences.append(
                {
                    "type": "BULLISH MACD",
                    "description": "Price falling but MACD rising — selling pressure weakening",
                    "strength": "MODERATE",
                }
            )
    return dedupe_divergences(divergences)


def calculate_volume_profile(df: pd.DataFrame, num_bins: int = 30) -> dict:
    if df is None or len(df) < 10:
        return {}

    price_min = df["low"].min()
    price_max = df["high"].max()
    bin_edges = np.linspace(price_min, price_max, num_bins + 1)
    volumes = np.zeros(num_bins)

    for _, row in df.iterrows():
        for b in range(num_bins):
            if row["low"] <= bin_edges[b + 1] and row["high"] >= bin_edges[b]:
                overlap = min(row["high"], bin_edges[b + 1]) - max(row["low"], bin_edges[b])
                total_range = row["high"] - row["low"]
                if total_range > 0:
                    volumes[b] += row["volume"] * (overlap / total_range)

    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    poc_idx = int(np.argmax(volumes))
    return {
        "bin_centers": bin_centers,
        "volumes": volumes,
        "poc_price": float(bin_centers[poc_idx]),
        "poc_volume": float(volumes[poc_idx]),
        "value_area_high": float(bin_centers[min(poc_idx + int(num_bins * 0.35), num_bins - 1)]),
        "value_area_low": float(bin_centers[max(poc_idx - int(num_bins * 0.35), 0)]),
    }


def detect_market_regime(
    df: pd.DataFrame,
    *,
    positive_color: str,
    neon_blue_color: str,
    neon_purple_color: str,
    negative_color: str,
    warning_color: str,
    text_muted_color: str,
) -> dict:
    if df is None or len(df) < 30:
        return {"regime": "UNKNOWN", "color": text_muted_color}

    try:
        adx = ta.trend.adx(df["high"], df["low"], df["close"], window=14)
        adx_val = float(adx.iloc[-1]) if not adx.empty and not pd.isna(adx.iloc[-1]) else 0.0
    except Exception:
        adx_val = 0.0

    try:
        bb = ta.volatility.BollingerBands(df["close"], window=20)
        hband = bb.bollinger_hband()
        lband = bb.bollinger_lband()
        mavg = bb.bollinger_mavg().replace(0, np.nan)
        bb_width_raw = ((hband - lband) / mavg).iloc[-1]
        bb_width = float(bb_width_raw) if not pd.isna(bb_width_raw) else 0.0
    except Exception:
        bb_width = 0.0

    try:
        atr = ta.volatility.average_true_range(df["high"], df["low"], df["close"], window=14)
        close_last = float(df["close"].iloc[-1])
        atr_last = float(atr.iloc[-1]) if not atr.empty and not pd.isna(atr.iloc[-1]) else 0.0
        atr_pct = float(atr_last / close_last) if close_last > 0 else 0.0
    except Exception:
        atr_pct = 0.0

    if adx_val > 40:
        regime = "STRONG TREND"
        color = positive_color
        desc = "Powerful directional move. Trend-following strategies optimal."
    elif adx_val > 25:
        regime = "TRENDING"
        color = neon_blue_color
        desc = "Clear directional bias. EMAs and MACD reliable."
    elif bb_width < 0.03:
        regime = "COMPRESSION"
        color = neon_purple_color
        desc = "Extreme low volatility. Breakout imminent. Watch for explosive move."
    elif atr_pct > 0.05:
        regime = "HIGH VOLATILITY"
        color = negative_color
        desc = "Choppy conditions. Reduce position size. Wide stops needed."
    elif adx_val < 15:
        regime = "RANGING"
        color = warning_color
        desc = "No trend. Mean-reversion strategies may work. Avoid breakout trades."
    else:
        regime = "TRANSITIONING"
        color = text_muted_color
        desc = "Market shifting between regimes. Wait for confirmation."

    return {
        "regime": regime,
        "color": color,
        "description": desc,
        "adx": adx_val,
        "bb_width": bb_width,
        "atr_pct": atr_pct * 100,
    }
