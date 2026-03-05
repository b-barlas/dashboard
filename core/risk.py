from __future__ import annotations

import numpy as np
import pandas as pd


def annualization_factor(timeframe: str) -> int:
    """Return periods-per-year for a candle timeframe."""
    tf = (timeframe or "").strip().lower()
    mapping = {
        "1m": 60 * 24 * 365,
        "3m": 20 * 24 * 365,
        "5m": 12 * 24 * 365,
        "15m": 4 * 24 * 365,
        "1h": 24 * 365,
        "4h": 6 * 365,
        "1d": 365,
    }
    return mapping.get(tf, 365)


def calculate_risk_metrics(
    df: pd.DataFrame,
    risk_free_rate: float = 0.02,
    timeframe: str = "1d",
    close_series: pd.Series | None = None,
) -> dict:
    """Calculate VaR, Sharpe, Sortino, Calmar, drawdown, and distribution stats."""
    if df is None or len(df) < 20:
        return {}

    if close_series is not None:
        close = pd.to_numeric(close_series, errors="coerce")
    else:
        close = pd.to_numeric(df.get("close"), errors="coerce")
    close = close.replace([np.inf, -np.inf], np.nan).dropna()
    close = close[close > 0]
    if len(close) < 20:
        return {}

    returns = close.pct_change().replace([np.inf, -np.inf], np.nan).dropna()
    if len(returns) < 10:
        return {}

    mean_return = float(returns.mean())
    std_return = float(returns.std(ddof=1))
    total_return = float((close.iloc[-1] / close.iloc[0]) - 1)

    ann_factor = annualization_factor(timeframe)
    risk_free_rate = float(max(-0.50, min(1.00, risk_free_rate)))
    var_95 = float(np.percentile(returns, 5))
    var_99 = float(np.percentile(returns, 1))
    tail_95 = returns[returns <= var_95]
    cvar_95 = float(tail_95.mean()) if len(tail_95) > 0 else var_95

    period_rf = risk_free_rate / ann_factor
    excess_returns = returns - period_rf
    sharpe = float(excess_returns.mean() / (float(excess_returns.std(ddof=1)) + 1e-9) * np.sqrt(ann_factor))

    downside = np.minimum(excess_returns.to_numpy(dtype=float), 0.0)
    downside_dev = float(np.sqrt(np.mean(np.square(downside)))) if downside.size > 0 else 0.0
    sortino = float(excess_returns.mean() / (downside_dev + 1e-9) * np.sqrt(ann_factor))

    cumulative = (1 + returns).cumprod()
    peak = cumulative.cummax()
    drawdown = (cumulative - peak) / peak
    max_drawdown = float(drawdown.min())
    max_dd_duration = 0
    dd_count = 0
    for dd in drawdown:
        if dd < 0:
            dd_count += 1
            max_dd_duration = max(max_dd_duration, dd_count)
        else:
            dd_count = 0

    ann_return = float(np.power(1.0 + total_return, ann_factor / max(len(returns), 1)) - 1.0)
    calmar = float(ann_return / (abs(max_drawdown) + 1e-9))
    win_rate = float((returns > 0).sum() / len(returns) * 100)

    return {
        "total_return": total_return * 100,
        "ann_return": ann_return * 100,
        "ann_volatility": std_return * np.sqrt(ann_factor) * 100,
        "sharpe": sharpe,
        "sortino": sortino,
        "calmar": calmar,
        "max_drawdown": max_drawdown * 100,
        "max_dd_duration": max_dd_duration,
        "var_95": var_95 * 100,
        "var_99": var_99 * 100,
        "cvar_95": cvar_95 * 100,
        "win_rate": win_rate,
        "skewness": float(returns.skew()),
        "kurtosis": float(returns.kurtosis()),
        "best_period": float(returns.max() * 100),
        "worst_period": float(returns.min() * 100),
        "mean_period": mean_return * 100,
        # Backward-compatible aliases
        "best_day": float(returns.max() * 100),
        "worst_day": float(returns.min() * 100),
        "mean_daily": mean_return * 100,
        "drawdown_series": drawdown,
        "cumulative_returns": cumulative,
    }
