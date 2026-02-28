from __future__ import annotations

from typing import Callable, Protocol, Tuple

import numpy as np
import pandas as pd
from core.signal_contract import strength_from_bias


class AnalysisLike(Protocol):
    signal: str
    bias: float


def _read_bias_like(result: AnalysisLike) -> float:
    """Read bias score from analysis result."""
    try:
        return float(getattr(result, "bias"))
    except Exception:
        return 50.0


def _infer_regime(df_slice: pd.DataFrame, analysis_obj: AnalysisLike) -> tuple[str, float]:
    """Classify local market regime at entry time.

    Priority:
    1) ADX from analysis object (if available)
    2) Price-structure fallback using drift/noise ratio
    """
    adx_raw = getattr(analysis_obj, "adx", np.nan)
    try:
        adx_val = float(adx_raw)
    except Exception:
        adx_val = np.nan

    if np.isfinite(adx_val):
        if adx_val >= 25:
            return "TREND", float(np.clip((adx_val - 20.0) * 4.0 + 60.0, 0.0, 100.0))
        if adx_val <= 18:
            return "RANGE", float(np.clip(70.0 - (adx_val - 10.0) * 3.0, 0.0, 100.0))
        return "MIXED", 50.0

    close = pd.to_numeric(df_slice["close"], errors="coerce").dropna()
    if len(close) < 15:
        return "MIXED", 50.0

    lookback = close.iloc[-30:] if len(close) >= 30 else close
    rets = lookback.pct_change().dropna()
    if rets.empty:
        return "MIXED", 50.0

    drift = abs(float(lookback.iloc[-1] / lookback.iloc[0] - 1.0))
    noise = float(rets.std()) * np.sqrt(len(rets))
    trend_ratio = drift / (noise + 1e-9)

    if trend_ratio >= 1.2:
        return "TREND", float(np.clip(60.0 + (trend_ratio - 1.2) * 35.0, 0.0, 100.0))
    if trend_ratio <= 0.6:
        return "RANGE", float(np.clip(70.0 + (0.6 - trend_ratio) * 40.0, 0.0, 100.0))
    return "MIXED", 50.0


def run_backtest(
    df: pd.DataFrame,
    analyzer: Callable[[pd.DataFrame], AnalysisLike],
    threshold: float = 70,
    exit_after: int = 5,
    commission: float = 0.001,
    slippage: float = 0.0005,
) -> Tuple[pd.DataFrame, str]:
    """Run single-position backtest over a price series.

    Entry filter is direction-agnostic:
    - Compute strength from directional bias (bias score)
    - Enter LONG/SHORT only when strength >= threshold
    """
    exit_after = max(1, int(exit_after))
    commission = max(0.0, float(commission))
    slippage = max(0.0, float(slippage))
    results = []
    equity_curve = [10000.0]
    peak = 10000.0
    max_drawdown = 0.0
    consecutive_losses = 0
    max_consecutive_losses = 0
    window_size = 200

    i = 55
    # Signal is computed on bar i close. Execution starts on next bar open.
    while i < len(df) - exit_after - 1:
        start_idx = max(0, i - window_size)
        df_slice = df.iloc[start_idx : i + 1]
        if len(df_slice) < 55:
            i += 1
            continue

        try:
            result = analyzer(df_slice)
            raw_signal = result.signal
            bias_score = _read_bias_like(result)
            strength_score = float(strength_from_bias(bias_score))
        except Exception:
            i += 1
            continue

        sig_plain = (
            "LONG"
            if raw_signal in ["STRONG BUY", "BUY"]
            else ("SHORT" if raw_signal in ["STRONG SELL", "SELL"] else "WAIT")
        )

        long_ok = sig_plain == "LONG" and strength_score >= threshold
        short_ok = sig_plain == "SHORT" and strength_score >= threshold
        if not (long_ok or short_ok):
            i += 1
            continue

        entry_idx = i + 1
        exit_idx = entry_idx + exit_after
        if exit_idx >= len(df):
            break

        entry_open = float(df["open"].iloc[entry_idx]) if "open" in df.columns else float(df["close"].iloc[entry_idx])
        exit_open = float(df["open"].iloc[exit_idx]) if "open" in df.columns else float(df["close"].iloc[exit_idx])
        if entry_open <= 0 or exit_open <= 0:
            i += 1
            continue

        if sig_plain == "LONG":
            entry_exec = entry_open * (1.0 + slippage)
            exit_exec = exit_open * (1.0 - slippage)
            gross_ret = (exit_exec - entry_exec) / entry_exec
        else:
            entry_exec = entry_open * (1.0 - slippage)
            exit_exec = exit_open * (1.0 + slippage)
            gross_ret = (entry_exec - exit_exec) / entry_exec

        net_ret = gross_ret - 2.0 * commission
        pnl = net_ret * 100.0
        regime, regime_score = _infer_regime(df_slice, result)

        equity = equity_curve[-1] * (1 + pnl / 100)
        equity_curve.append(equity)

        peak = max(peak, equity)
        drawdown = (peak - equity) / peak * 100
        max_drawdown = max(max_drawdown, drawdown)

        if pnl <= 0:
            consecutive_losses += 1
            max_consecutive_losses = max(max_consecutive_losses, consecutive_losses)
        else:
            consecutive_losses = 0

        results.append(
            {
                "Date": df["timestamp"].iloc[i],
                "Signal Time": df["timestamp"].iloc[i],
                "Strength": round(strength_score, 1),
                "Bias": round(bias_score, 1),
                "Signal": sig_plain,
                "Entry": entry_exec,
                "Exit": exit_exec,
                "PnL (%)": round(pnl, 2),
                "Equity": round(equity, 2),
                "Regime": regime,
                "Regime Score": round(regime_score, 1),
                "Holding Bars": int(exit_after),
            }
        )

        # Single-position mode: next signal evaluation starts at exit bar.
        i = exit_idx

    df_results = pd.DataFrame(results)
    if df_results.empty:
        return (
            df_results,
            "<div style='color:#FFB000;margin-top:1rem;'>"
            "<p><b>⚠️ No Signals:</b> No trades met the threshold criteria</p>"
            "<p>Try lowering the strength threshold or using more data</p>"
            "</div>",
        )

    wins = int((df_results["PnL (%)"] > 0).sum())
    losses = int((df_results["PnL (%)"] <= 0).sum())
    total_trades = wins + losses
    winrate = (wins / total_trades) * 100 if total_trades > 0 else 0.0

    gross_profit = float(df_results[df_results["PnL (%)"] > 0]["PnL (%)"].sum())
    gross_loss = abs(float(df_results[df_results["PnL (%)"] <= 0]["PnL (%)"].sum()))
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")

    avg_win = (
        float(df_results[df_results["PnL (%)"] > 0]["PnL (%)"].mean()) if wins > 0 else 0.0
    )
    avg_loss = (
        float(df_results[df_results["PnL (%)"] <= 0]["PnL (%)"].mean()) if losses > 0 else 0.0
    )

    total_return = (equity_curve[-1] - equity_curve[0]) / equity_curve[0] * 100
    returns = df_results["PnL (%)"].astype(float) / 100.0
    mean_return = float(returns.mean())
    std_return = float(returns.std())
    # Approximate annualization by timeframe implied from candle timestamps.
    ann_factor = 365.0
    if len(df) >= 2 and "timestamp" in df.columns:
        try:
            dt_hours = (df["timestamp"].iloc[1] - df["timestamp"].iloc[0]).total_seconds() / 3600.0
            if dt_hours > 0:
                periods_per_day = 24.0 / dt_hours
                ann_factor = periods_per_day * 365.0 / exit_after
        except Exception:
            ann_factor = 365.0 / exit_after
    else:
        ann_factor = 365.0 / exit_after

    sharpe_ratio = (mean_return / (std_return + 1e-9)) * np.sqrt(ann_factor) if std_return > 0 else 0.0

    summary_html = f"""
    <div style='margin-top:1rem; background-color:#16213E; padding:20px; border-radius:10px;'>
        <h3 style='color:#06D6A0; margin-top:0;'>📊 Backtest Results</h3>
        <p style='color:#8CA1B6; margin:0;'>Trades: {total_trades} | Win Rate: {winrate:.1f}% | Profit Factor: {profit_factor:.2f}</p>
        <p style='color:#8CA1B6; margin:6px 0 0 0;'>Return: {total_return:+.2f}% | Max DD: {max_drawdown:.2f}% | Sharpe: {sharpe_ratio:.2f}</p>
        <p style='color:#8CA1B6; margin:6px 0 0 0;'>Avg Win: {avg_win:+.2f}% | Avg Loss: {avg_loss:.2f}% | Max Consecutive Losses: {max_consecutive_losses}</p>
    </div>
    """
    return df_results, summary_html
