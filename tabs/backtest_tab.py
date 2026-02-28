from __future__ import annotations

from ui.ctx import get_ctx

import numpy as np
import plotly.graph_objs as go
from core.signal_contract import strength_from_bias
from ui.snapshot_cache import live_or_snapshot


def render(ctx: dict) -> None:
    st = get_ctx(ctx, "st")
    ACCENT = get_ctx(ctx, "ACCENT")
    TEXT_MUTED = get_ctx(ctx, "TEXT_MUTED")
    POSITIVE = get_ctx(ctx, "POSITIVE")
    WARNING = get_ctx(ctx, "WARNING")
    _tip = ctx.get("_tip", lambda label, _text: label)
    _normalize_coin_input = get_ctx(ctx, "_normalize_coin_input")
    _validate_coin_symbol = get_ctx(ctx, "_validate_coin_symbol")
    analyse = ctx.get("analyse")
    fetch_ohlcv = get_ctx(ctx, "fetch_ohlcv")
    run_backtest = get_ctx(ctx, "run_backtest")

    st.markdown(f"<h2 style='color:{ACCENT};'>Backtest Simulator</h2>", unsafe_allow_html=True)
    st.markdown(
        f"<div class='panel-box' style='margin-bottom:1rem;'>"
        f"<b style='color:{ACCENT};'>How the backtest works:</b>"
        f"<ul style='color:{TEXT_MUTED}; font-size:0.88rem; line-height:1.7; margin-top:0.5rem;'>"
        "<li>The engine slides a window through historical candles and runs the <b>full technical analysis</b> "
        "(EMA, RSI, MACD, SuperTrend, Ichimoku, Bollinger, ADX, etc.) at each step.</li>"
        "<li>When the <b>Signal</b> is LONG or SHORT <b>and</b> the <b>Strength Score</b> exceeds your threshold, "
        "a simulated trade is opened at the closing price.</li>"
        "<li>The trade is automatically closed after <b>N candles</b> (your exit setting). "
        "Commission is deducted on both entry and exit.</li>"
        "<li>This tests whether the dashboard's signal + strength system would have been profitable "
        "on the chosen coin and timeframe. Use it to calibrate your strength threshold and hold duration.</li>"
        "</ul></div>",
        unsafe_allow_html=True,
    )
    st.markdown(
        f"<details style='margin-bottom:0.7rem;'>"
        f"<summary style='color:{ACCENT}; cursor:pointer;'>How to read quickly (?)</summary>"
        f"<div style='color:{TEXT_MUTED}; font-size:0.85rem; line-height:1.7; margin-top:0.5rem;'>"
        f"<b>1.</b> Start with <b>Profit Factor</b>, <b>Max Drawdown</b>, and <b>Total Return</b> together.<br>"
        f"<b>2.</b> A high win rate with weak profit factor is usually not robust.<br>"
        f"<b>3.</b> Increase threshold until drawdown becomes acceptable without killing trade count."
        f"</div></details>",
        unsafe_allow_html=True,
    )

    col1, col2 = st.columns(2)
    with col1:
        coin = _normalize_coin_input(
            st.text_input("Coin (e.g. BTC, ETH, TAO)", value="BTC", key="backtest_coin_input")
        )
        timeframe = st.selectbox("Timeframe", ["3m", "5m", "15m", "1h", "4h", "1d"], index=2)
        limit = st.slider("Number of Candles", 100, 1000, step=100, value=500)
    with col2:
        threshold = st.slider(
            "Strength Threshold (%)",
            35,
            85,
            step=5,
            value=55,
            help="Only take trades with strength above this level",
            key="backtest_threshold",
        )
        exit_after = st.slider("Exit After N Candles", 1, 20, step=1, value=5)
        commission = (
            st.slider(
                "Commission (%)",
                0.0,
                1.0,
                step=0.05,
                value=0.1,
                help="Trading fee per trade (typical spot: 0.1%)",
            )
            / 100
        )
        slippage = (
            st.slider(
                "Slippage (%)",
                0.0,
                0.5,
                step=0.01,
                value=0.05,
                help="Simulated slippage per trade (market impact + spread)",
            )
            / 100
        )

    run_clicked = st.button("🚀 Run Backtest", type="primary", key="backtest_run_btn")
    force_run = bool(st.session_state.pop("backtest_force_run", False))
    if not (run_clicked or force_run):
        return

    _val_err = _validate_coin_symbol(coin)
    if _val_err:
        st.error(_val_err)
        return

    st.info("Fetching data and running comprehensive analysis...")
    df_live = fetch_ohlcv(coin, timeframe, limit)
    df, used_cache, cache_ts = live_or_snapshot(
        st,
        f"backtest_df::{coin}::{timeframe}::{limit}",
        df_live,
        max_age_sec=1800,
        current_sig=(coin, timeframe, limit),
    )
    if used_cache:
        st.warning(f"Live data unavailable. Using cached snapshot from {cache_ts}.")
    if df is None or df.empty:
        st.error("Failed to fetch historical data. Please check the symbol or connection.")
        return

    st.success(f"✅ Fetched {len(df)} candles. Running backtest...")
    try:
        result_df, _summary_html = run_backtest(
            df,
            threshold=threshold,
            exit_after=exit_after,
            commission=commission,
            slippage=slippage,
        )
    except Exception as e:
        st.error(f"Error during backtest: {e}")
        return

    if result_df.empty:
        st.warning("No signals generated for the given threshold.")
        actionable_strengths: list[float] = []
        long_count = 0
        short_count = 0
        if callable(analyse):
            for i in range(55, len(df) - exit_after - 1):
                s0 = max(0, i - 200)
                df_slice = df.iloc[s0:i + 1]
                if len(df_slice) < 55:
                    continue
                try:
                    a = analyse(df_slice)
                    sig = (
                        "LONG"
                        if a.signal in ["STRONG BUY", "BUY"]
                        else ("SHORT" if a.signal in ["STRONG SELL", "SELL"] else "WAIT")
                    )
                    if sig == "LONG":
                        long_count += 1
                    elif sig == "SHORT":
                        short_count += 1
                    if sig in {"LONG", "SHORT"}:
                        actionable_strengths.append(float(strength_from_bias(float(a.bias))))
                except Exception:
                    continue

        if actionable_strengths:
            arr = np.array(actionable_strengths, dtype=float)
            p70 = float(np.percentile(arr, 70))
            suggested = int(max(35, min(85, round(p70 / 5) * 5)))
            st.info(
                f"Diagnostics: {len(actionable_strengths)} actionable bars found "
                f"(LONG {long_count}, SHORT {short_count}), but very few pass threshold {threshold}%. "
                f"Suggested threshold: {suggested}%."
            )
            if suggested != threshold and st.button(
                f"Re-run with suggested threshold ({suggested}%)",
                key="backtest_rerun_suggested_btn",
            ):
                st.session_state["backtest_threshold"] = suggested
                st.session_state["backtest_force_run"] = True
                st.rerun()
            else:
                return
        else:
            if callable(analyse):
                bias_strengths: list[float] = []
                bias_long = 0
                bias_short = 0
                for i in range(55, len(df) - exit_after - 1):
                    s0 = max(0, i - 200)
                    df_slice = df.iloc[s0:i + 1]
                    if len(df_slice) < 55:
                        continue
                    try:
                        a = analyse(df_slice)
                        bias = float(a.bias)
                        s = float(strength_from_bias(bias))
                        bias_strengths.append(s)
                        if bias >= 60:
                            bias_long += 1
                        elif bias <= 40:
                            bias_short += 1
                    except Exception:
                        continue
                if bias_strengths:
                    avg_s = float(np.mean(bias_strengths))
                    p75_s = float(np.percentile(np.array(bias_strengths), 75))
                    st.info(
                        "No actionable LONG/SHORT bars were produced by the full signal gate in this window. "
                        f"Bias-only diagnostics: LONG candidates={bias_long}, SHORT candidates={bias_short}, "
                        f"avg strength={avg_s:.1f}%, 75th percentile={p75_s:.1f}%."
                    )
                    st.caption(
                        "Interpretation: strength exists, but quality filters can still force WAIT. "
                        "Try lower timeframe, more candles, or a less selective market phase."
                    )
                else:
                    st.info("No actionable LONG/SHORT bars were produced in this window. Try lower timeframe or more candles.")
            else:
                st.info("No actionable LONG/SHORT bars were produced in this window. Try lower timeframe or more candles.")
            return
        return

    st.markdown("### 📊 Backtest Results")
    total_trades = len(result_df)
    wins = len(result_df[result_df["PnL (%)"] > 0])
    losses = total_trades - wins
    winrate = (wins / total_trades * 100) if total_trades > 0 else 0
    gross_profit = result_df[result_df["PnL (%)"] > 0]["PnL (%)"].sum()
    gross_loss = abs(result_df[result_df["PnL (%)"] <= 0]["PnL (%)"].sum())
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")

    if "Equity" in result_df.columns:
        equity = result_df["Equity"].values
        peak = equity[0]
        max_dd = 0
        for val in equity:
            if val > peak:
                peak = val
            dd = (peak - val) / peak * 100
            if dd > max_dd:
                max_dd = dd
        max_drawdown = max_dd
        total_return = (equity[-1] - 10000.0) / 10000.0 * 100
    else:
        max_drawdown = 0
        total_return = 0

    returns = result_df["PnL (%)"].astype(float) / 100.0
    mean_return = returns.mean()
    std_return = returns.std()
    sharpe_ratio = (mean_return / (std_return + 1e-9)) * np.sqrt(365 / exit_after) if std_return > 0 else 0
    avg_win = result_df[result_df["PnL (%)"] > 0]["PnL (%)"].mean() if wins > 0 else 0
    avg_loss = result_df[result_df["PnL (%)"] <= 0]["PnL (%)"].mean() if losses > 0 else 0

    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("Total Trades", total_trades)
        st.metric("Avg Win", f"{avg_win:+.2f}%")
    with c2:
        st.metric("Win Rate", f"{winrate:.1f}%", f"{wins}W / {losses}L")
        st.metric("Avg Loss", f"{avg_loss:.2f}%")
    with c3:
        st.metric("Profit Factor", f"{profit_factor:.2f}", "Target: ≥1.5")
        st.metric("Costs", f"{commission*100:.2f}% + {slippage*100:.2f}%", "comm + slip")

    c4, c5, c6 = st.columns(3)
    with c4:
        st.metric("Total Return", f"{total_return:+.2f}%")
    with c5:
        st.metric("Max Drawdown", f"{max_drawdown:.2f}%", "Target: <15%")
    with c6:
        st.metric("Sharpe Ratio", f"{sharpe_ratio:.2f}", "Target: >1.0")

    if profit_factor >= 1.5 and max_drawdown <= 15 and total_return > 0:
        profile = "Deployable Profile"
        profile_body = "Model quality is strong for this setup. Consider forward-testing with strict risk limits."
        profile_color = POSITIVE
    elif profit_factor >= 1.2 and max_drawdown <= 25:
        profile = "Selective Profile"
        profile_body = "Edge exists but is moderate. Tighten threshold/exit rules before relying on it."
        profile_color = WARNING
    else:
        profile = "Defensive Profile"
        profile_body = "Risk-adjusted quality is weak for this configuration. Re-tune before live use."
        profile_color = "#ff4d6d"

    st.markdown(
        f"<div style='border:1px solid rgba(0,212,255,0.18); border-left:4px solid {profile_color}; "
        f"border-radius:12px; padding:12px 14px; margin:8px 0 14px 0; "
        f"background:linear-gradient(140deg, rgba(0,0,0,0.72), rgba(10,18,30,0.88));'>"
        f"<div style='color:{profile_color}; font-size:0.98rem; font-weight:700;'>Backtest Profile: {profile}</div>"
        f"<div style='color:{TEXT_MUTED}; font-size:0.86rem; line-height:1.55; margin-top:4px;'>{profile_body}</div>"
        f"</div>",
        unsafe_allow_html=True,
    )

    if "Regime" in result_df.columns:
        st.markdown(f"<h3 style='color:{ACCENT}; margin-top:1.4rem;'>🧭 Regime Comparison</h3>", unsafe_allow_html=True)
        st.markdown(
            f"<div style='color:{TEXT_MUTED}; font-size:0.84rem; margin-bottom:0.45rem;'>"
            f"TREND = directional market, RANGE = mean-reversion market, MIXED = transition."
            f"</div>",
            unsafe_allow_html=True,
        )
        reg = result_df.copy()
        reg["is_win"] = (reg["PnL (%)"] > 0).astype(int)
        regime_stats = (
            reg.groupby("Regime", dropna=False)
            .agg(
                Trades=("PnL (%)", "count"),
                WinRate=("is_win", "mean"),
                AvgPnL=("PnL (%)", "mean"),
                TotalPnL=("PnL (%)", "sum"),
                AvgRegimeScore=("Regime Score", "mean"),
            )
            .reset_index()
        )
        regime_stats["WinRate"] = regime_stats["WinRate"] * 100.0
        regime_stats["WinRate"] = regime_stats["WinRate"].map(lambda v: f"{v:.1f}%")
        regime_stats["AvgPnL"] = regime_stats["AvgPnL"].map(lambda v: f"{v:+.2f}%")
        regime_stats["TotalPnL"] = regime_stats["TotalPnL"].map(lambda v: f"{v:+.2f}%")
        regime_stats["AvgRegimeScore"] = regime_stats["AvgRegimeScore"].map(lambda v: f"{v:.1f}")
        st.dataframe(regime_stats, width="stretch", hide_index=True)

    if "Equity" in result_df.columns:
        st.markdown(f"<h3 style='color:{ACCENT}; margin-top:2rem;'>💰 Equity Curve</h3>", unsafe_allow_html=True)
        equity_fig = go.Figure()
        equity_fig.add_trace(
            go.Scatter(
                x=result_df["Date"],
                y=result_df["Equity"],
                mode="lines",
                name="Equity",
                line=dict(color=POSITIVE, width=2),
                fill="tozeroy",
                fillcolor="rgba(6, 214, 160, 0.1)",
            )
        )
        equity_fig.add_hline(
            y=10000, line=dict(color=WARNING, dash="dash", width=1), annotation_text="Starting Capital"
        )
        equity_fig.update_layout(
            template="plotly_dark",
            height=300,
            xaxis_title="Date",
            yaxis_title="Equity ($)",
            showlegend=False,
            hovermode="x unified",
        )
        st.plotly_chart(equity_fig, width="stretch")

    st.markdown(f"<h3 style='color:{ACCENT}; margin-top:2rem;'>📜 Trade History</h3>", unsafe_allow_html=True)
    styled_df = result_df.copy()
    styled_df["Date"] = styled_df["Date"].dt.strftime("%Y-%m-%d %H:%M")
    styled_df["Entry"] = styled_df["Entry"].apply(lambda x: f"${x:,.4f}")
    styled_df["Exit"] = styled_df["Exit"].apply(lambda x: f"${x:,.4f}")
    styled_df["PnL (%)"] = styled_df["PnL (%)"].apply(lambda x: f"{x:+.2f}%")
    if "Strength" in styled_df.columns:
        styled_df["Strength"] = styled_df["Strength"].apply(lambda x: f"{x:.1f}%")
    if "Bias" in styled_df.columns:
        styled_df["Bias"] = styled_df["Bias"].apply(lambda x: f"{x:.1f}%")
    if "Regime Score" in styled_df.columns:
        styled_df["Regime Score"] = styled_df["Regime Score"].apply(lambda x: f"{x:.1f}")
    if "Equity" in styled_df.columns:
        styled_df["Equity"] = styled_df["Equity"].apply(lambda x: f"${x:,.2f}")
    st.markdown(
        f"<details style='margin:0.35rem 0 0.45rem 0;'>"
        f"<summary style='color:{ACCENT}; cursor:pointer;'>Trade History Column Guide (?)</summary>"
        f"<div style='color:{TEXT_MUTED}; font-size:0.84rem; line-height:1.7; margin-top:0.5rem;'>"
        f"{_tip('Signal', 'Direction opened by the model at that bar (LONG/SHORT).')} | "
        f"{_tip('Strength', 'Signal power score at entry (0-100).')} | "
        f"{_tip('PnL (%)', 'Net trade return after commission and slippage assumptions.')} | "
        f"{_tip('Regime', 'Market state around the trade (TREND/RANGE/MIXED).')} | "
        f"{_tip('Regime Score', 'Numeric trend-quality context used by the engine at entry.')}"
        f"</div></details>",
        unsafe_allow_html=True,
    )
    st.dataframe(styled_df, width="stretch")

    csv_bytes = result_df.to_csv(index=False).encode("utf-8")
    filename = f"{coin.replace('/', '_')}_{timeframe}_backtest.csv"
    st.download_button(
        label="Download Results (CSV)",
        data=csv_bytes,
        file_name=filename,
        mime="text/csv",
    )
