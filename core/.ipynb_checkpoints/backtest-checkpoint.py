from __future__ import annotations

from typing import Callable, Protocol, Tuple

import numpy as np
import pandas as pd


class AnalysisLike(Protocol):
    signal: str
    confidence: float


def run_backtest(
    df: pd.DataFrame,
    analyzer: Callable[[pd.DataFrame], AnalysisLike],
    threshold: float = 70,
    exit_after: int = 5,
    commission: float = 0.001,
    slippage: float = 0.0005,
) -> Tuple[pd.DataFrame, str]:
    """Run single-position backtest over a price series.

    Entry filter is direction-aware:
    - LONG if confidence >= threshold
    - SHORT if confidence <= (100 - threshold)
    """
    results = []
    equity_curve = [10000.0]
    peak = 10000.0
    max_drawdown = 0.0
    consecutive_losses = 0
    max_consecutive_losses = 0
    window_size = 200

    i = 55
    while i < len(df) - exit_after:
        start_idx = max(0, i - window_size)
        df_slice = df.iloc[start_idx : i + 1]
        if len(df_slice) < 55:
            i += 1
            continue

        try:
            result = analyzer(df_slice)
            raw_signal = result.signal
            conf_score = float(result.confidence)
        except Exception:
            i += 1
            continue

        sig_plain = (
            "LONG"
            if raw_signal in ["STRONG BUY", "BUY"]
            else ("SHORT" if raw_signal in ["STRONG SELL", "SELL"] else "WAIT")
        )

        long_ok = sig_plain == "LONG" and conf_score >= threshold
        short_ok = sig_plain == "SHORT" and conf_score <= (100 - threshold)
        if not (long_ok or short_ok):
            i += 1
            continue

        entry_price = float(df["close"].iloc[i])
        future_price = float(df["close"].iloc[i + exit_after])
        total_cost = 2 * commission + 2 * slippage

        if sig_plain == "LONG":
            pnl = ((future_price - entry_price) / entry_price - total_cost) * 100
        else:
            pnl = ((entry_price - future_price) / entry_price - total_cost) * 100

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
                "Confidence": round(conf_score, 1),
                "Signal": sig_plain,
                "Entry": entry_price,
                "Exit": future_price,
                "PnL (%)": round(pnl, 2),
                "Equity": round(equity, 2),
            }
        )

        i += exit_after

    df_results = pd.DataFrame(results)
    if df_results.empty:
        return (
            df_results,
            "<div style='color:#FFB000;margin-top:1rem;'>"
            "<p><b>⚠️ No Signals:</b> No trades met the threshold criteria</p>"
            "<p>Try lowering the confidence threshold or using more data</p>"
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
    sharpe_ratio = (
        (mean_return / (std_return + 1e-9)) * np.sqrt(365 / exit_after) if std_return > 0 else 0.0
    )

    summary_html = f"""
    <div style='margin-top:1rem; background-color:#16213E; padding:20px; border-radius:10px;'>
        <h3 style='color:#06D6A0; margin-top:0;'>📊 Backtest Results</h3>
        <p style='color:#8CA1B6; margin:0;'>Trades: {total_trades} | Win Rate: {winrate:.1f}% | Profit Factor: {profit_factor:.2f}</p>
        <p style='color:#8CA1B6; margin:6px 0 0 0;'>Return: {total_return:+.2f}% | Max DD: {max_drawdown:.2f}% | Sharpe: {sharpe_ratio:.2f}</p>
        <p style='color:#8CA1B6; margin:6px 0 0 0;'>Avg Win: {avg_win:+.2f}% | Avg Loss: {avg_loss:.2f}% | Max Consecutive Losses: {max_consecutive_losses}</p>
    </div>
    """
    return df_results, summary_html

