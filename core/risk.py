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
    df: pd.DataFrame, risk_free_rate: float = 0.02, timeframe: str = "1d"
) -> dict:
    """Calculate VaR, Sharpe, Sortino, Calmar, drawdown, and distribution stats."""
    if df is None or len(df) < 20:
        return {}

    returns = df["close"].pct_change().dropna()
    if len(returns) < 10:
        return {}

    mean_return = float(returns.mean())
    std_return = float(returns.std())
    total_return = float((df["close"].iloc[-1] / df["close"].iloc[0]) - 1)

    ann_factor = annualization_factor(timeframe)
    var_95 = float(np.percentile(returns, 5))
    var_99 = float(np.percentile(returns, 1))
    cvar_95 = float(returns[returns <= var_95].mean()) if len(returns[returns <= var_95]) > 0 else var_95

    period_rf = risk_free_rate / ann_factor
    excess_returns = returns - period_rf
    sharpe = float(excess_returns.mean() / (excess_returns.std() + 1e-9) * np.sqrt(ann_factor))

    downside_returns = returns[returns < 0]
    downside_std = float(downside_returns.std()) if len(downside_returns) > 0 else 1e-9
    sortino = float((mean_return - period_rf) / (downside_std + 1e-9) * np.sqrt(ann_factor))

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

    ann_return = mean_return * ann_factor
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
        "best_day": float(returns.max() * 100),
        "worst_day": float(returns.min() * 100),
        "mean_daily": mean_return * 100,
        "drawdown_series": drawdown,
        "cumulative_returns": cumulative,
    }

