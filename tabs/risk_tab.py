from __future__ import annotations

from ui.ctx import get_ctx

import numpy as np
import pandas as pd
import plotly.graph_objs as go


def render(ctx: dict) -> None:
    st = get_ctx(ctx, "st")
    ACCENT = get_ctx(ctx, "ACCENT")
    TEXT_MUTED = get_ctx(ctx, "TEXT_MUTED")
    POSITIVE = get_ctx(ctx, "POSITIVE")
    NEGATIVE = get_ctx(ctx, "NEGATIVE")
    WARNING = get_ctx(ctx, "WARNING")
    NEON_BLUE = get_ctx(ctx, "NEON_BLUE")
    NEON_PURPLE = get_ctx(ctx, "NEON_PURPLE")
    PRIMARY_BG = get_ctx(ctx, "PRIMARY_BG")
    _tip = get_ctx(ctx, "_tip")
    _normalize_coin_input = get_ctx(ctx, "_normalize_coin_input")
    _validate_coin_symbol = get_ctx(ctx, "_validate_coin_symbol")
    fetch_ohlcv = get_ctx(ctx, "fetch_ohlcv")
    calculate_risk_metrics = get_ctx(ctx, "calculate_risk_metrics")

    st.markdown(
        f"""
        <style>
        .risk-kpi-grid {{
            display:grid;
            grid-template-columns:repeat(4,minmax(0,1fr));
            gap:10px;
            margin:8px 0 12px 0;
        }}
        .risk-kpi {{
            border:1px solid rgba(0,212,255,0.16);
            border-radius:12px;
            padding:12px 14px;
            background:linear-gradient(140deg, rgba(0,0,0,0.72), rgba(10,18,30,0.88));
        }}
        .risk-kpi-label {{
            color:{TEXT_MUTED};
            font-size:0.70rem;
            text-transform:uppercase;
            letter-spacing:0.8px;
        }}
        .risk-kpi-value {{
            color:{ACCENT};
            font-size:1.2rem;
            font-weight:700;
            margin-top:4px;
        }}
        .risk-badge {{
            display:inline-flex;
            align-items:center;
            gap:6px;
            margin-top:7px;
            padding:2px 9px;
            border-radius:999px;
            font-size:0.72rem;
            font-weight:700;
            border:1px solid rgba(255,255,255,0.18);
            background:rgba(0,0,0,0.28);
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(f"<h2 style='color:{ACCENT};'>Risk Analytics</h2>", unsafe_allow_html=True)
    st.markdown(
        f"<div class='panel-box'>"
        f"<b style='color:{ACCENT}; font-size:1rem;'>What does this tab show?</b>"
        f"<p style='color:{TEXT_MUTED}; font-size:0.9rem; margin-top:6px; line-height:1.6;'>"
        f"Measures downside risk, volatility, and risk-adjusted performance from historical candles. "
        f"Use this tab to answer: <i>Is this coin worth the risk on this timeframe?</i>"
        f"</p>"
        f"<p style='color:{TEXT_MUTED}; font-size:0.85rem; margin-top:6px; line-height:1.6;'>"
        f"{_tip('Sharpe', 'Risk-adjusted return using total volatility. Higher is better.')} | "
        f"{_tip('Sortino', 'Risk-adjusted return using only downside volatility. Higher is better.')} | "
        f"{_tip('Max Drawdown', 'Worst peak-to-trough historical loss in %. Lower is safer.')} | "
        f"{_tip('VaR 95%', '5th percentile return. Typical bad-day threshold; more negative means higher tail risk.')} | "
        f"{_tip('CVaR 95%', 'Average loss on the worst 5% of returns. Captures crash severity better than VaR.')} | "
        f"{_tip('Calmar', 'Annual return divided by max drawdown. Higher means better reward per drawdown risk.')}"
        f"</p>"
        f"</div>",
        unsafe_allow_html=True,
    )

    c1, c2, c3 = st.columns(3)
    with c1:
        risk_coin = _normalize_coin_input(st.text_input("Coin", value="BTC", key="risk_coin"))
    with c2:
        risk_tf = st.selectbox("Timeframe", ["1h", "4h", "1d"], index=2, key="risk_tf")
    with c3:
        lookback = st.selectbox("Lookback Candles", [300, 500, 800], index=1, key="risk_lookback")

    if not st.button("Analyze Risk", type="primary", key="risk_run"):
        return

    _val_err = _validate_coin_symbol(risk_coin)
    if _val_err:
        st.error(_val_err)
        return

    with st.spinner("Calculating risk metrics..."):
        df = fetch_ohlcv(risk_coin, risk_tf, limit=int(lookback))
        if df is None or len(df) < 30:
            st.error("Not enough data.")
            return
        metrics = calculate_risk_metrics(df, timeframe=risk_tf)
        if not metrics:
            st.error("Could not calculate metrics.")
            return

    returns = df["close"].pct_change().dropna()
    rolling_window = min(30, max(10, len(returns) // 5))
    rolling_vol = (returns.rolling(rolling_window).std() * 100.0).dropna()

    var95 = float(metrics["var_95"])
    cvar95 = float(metrics["cvar_95"])
    dd_abs = abs(float(metrics["max_drawdown"]))
    sharpe = float(metrics["sharpe"])
    sortino = float(metrics["sortino"])
    calmar = float(metrics["calmar"])
    ann_vol = float(metrics["ann_volatility"])

    def _status(value: float, good: float, neutral: float, lower_better: bool = False) -> tuple[str, str]:
        if lower_better:
            if value <= good:
                return "Healthy", POSITIVE
            if value <= neutral:
                return "Watch", WARNING
            return "Risky", NEGATIVE
        if value >= good:
            return "Healthy", POSITIVE
        if value >= neutral:
            return "Watch", WARNING
        return "Risky", NEGATIVE

    # Simple risk regime classifier for beginner readability.
    if dd_abs > 35 or var95 < -5.0 or sharpe < 0:
        regime = "High Risk"
        regime_color = NEGATIVE
    elif dd_abs > 20 or var95 < -3.0 or sharpe < 0.8:
        regime = "Medium Risk"
        regime_color = WARNING
    else:
        regime = "Lower Risk"
        regime_color = POSITIVE

    sharpe_s, sharpe_c = _status(sharpe, good=1.2, neutral=0.4)
    sortino_s, sortino_c = _status(sortino, good=1.6, neutral=0.6)
    dd_s, dd_c = _status(dd_abs, good=15.0, neutral=30.0, lower_better=True)
    var_s, var_c = _status(abs(var95), good=2.5, neutral=4.0, lower_better=True)
    cvar_s, cvar_c = _status(abs(cvar95), good=3.2, neutral=5.5, lower_better=True)
    cal_s, cal_c = _status(calmar, good=1.2, neutral=0.6)
    vol_s, vol_c = _status(ann_vol, good=45.0, neutral=80.0, lower_better=True)

    st.markdown(
        f"<div class='risk-kpi-grid'>"
        f"<div class='risk-kpi'><div class='risk-kpi-label'>Risk Regime</div><div class='risk-kpi-value' style='color:{regime_color};'>{regime}</div>"
        f"<span class='risk-badge' style='color:{regime_color}; border-color:{regime_color};'><span style='color:{regime_color};'>&#9679;</span>{regime}</span></div>"
        f"<div class='risk-kpi'><div class='risk-kpi-label'>Sharpe</div><div class='risk-kpi-value'>{sharpe:.2f}</div>"
        f"<span class='risk-badge' style='color:{sharpe_c}; border-color:{sharpe_c};'><span style='color:{sharpe_c};'>&#9679;</span>{sharpe_s}</span></div>"
        f"<div class='risk-kpi'><div class='risk-kpi-label'>Max Drawdown</div><div class='risk-kpi-value'>{metrics['max_drawdown']:.2f}%</div>"
        f"<span class='risk-badge' style='color:{dd_c}; border-color:{dd_c};'><span style='color:{dd_c};'>&#9679;</span>{dd_s}</span></div>"
        f"<div class='risk-kpi'><div class='risk-kpi-label'>VaR 95%</div><div class='risk-kpi-value'>{var95:.2f}%</div>"
        f"<span class='risk-badge' style='color:{var_c}; border-color:{var_c};'><span style='color:{var_c};'>&#9679;</span>{var_s}</span></div>"
        f"</div>",
        unsafe_allow_html=True,
    )

    st.markdown(
        f"<details style='margin-bottom:0.7rem;'>"
        f"<summary style='color:{ACCENT}; cursor:pointer;'>How to read quickly (?)</summary>"
        f"<div style='color:{TEXT_MUTED}; font-size:0.85rem; line-height:1.7; margin-top:0.5rem;'>"
        f"<b>1.</b> Start with <b>Risk Regime</b> and <b>Max Drawdown</b> to estimate downside stress.<br>"
        f"<b>2.</b> Check <b>VaR/CVaR</b> for tail risk: more negative values mean deeper bad-day losses.<br>"
        f"<b>3.</b> Use <b>Sharpe/Sortino/Calmar</b> to judge if returns justify risk taken.<br>"
        f"<b>4.</b> Confirm if volatility is expanding using the rolling-vol chart before sizing up."
        f"</div></details>",
        unsafe_allow_html=True,
    )

    st.markdown(
        f"<div class='risk-kpi-grid'>"
        f"<div class='risk-kpi'><div class='risk-kpi-label'>Sortino</div><div class='risk-kpi-value'>{sortino:.2f}</div>"
        f"<span class='risk-badge' style='color:{sortino_c}; border-color:{sortino_c};'><span style='color:{sortino_c};'>&#9679;</span>{sortino_s}</span></div>"
        f"<div class='risk-kpi'><div class='risk-kpi-label'>Calmar</div><div class='risk-kpi-value'>{calmar:.2f}</div>"
        f"<span class='risk-badge' style='color:{cal_c}; border-color:{cal_c};'><span style='color:{cal_c};'>&#9679;</span>{cal_s}</span></div>"
        f"<div class='risk-kpi'><div class='risk-kpi-label'>CVaR 95%</div><div class='risk-kpi-value'>{cvar95:.2f}%</div>"
        f"<span class='risk-badge' style='color:{cvar_c}; border-color:{cvar_c};'><span style='color:{cvar_c};'>&#9679;</span>{cvar_s}</span></div>"
        f"<div class='risk-kpi'><div class='risk-kpi-label'>Annualized Volatility</div><div class='risk-kpi-value'>{ann_vol:.1f}%</div>"
        f"<span class='risk-badge' style='color:{vol_c}; border-color:{vol_c};'><span style='color:{vol_c};'>&#9679;</span>{vol_s}</span></div>"
        f"</div>",
        unsafe_allow_html=True,
    )

    summary_df = pd.DataFrame(
        [
            {"Metric": "Total Return", "Value": f"{metrics['total_return']:+.2f}%", "Status": "Info"},
            {"Metric": "Annualized Return", "Value": f"{metrics['ann_return']:+.2f}%", "Status": "Info"},
            {"Metric": "Win Rate", "Value": f"{metrics['win_rate']:.1f}%", "Status": ("Healthy" if metrics["win_rate"] >= 55 else ("Watch" if metrics["win_rate"] >= 45 else "Risky"))},
            {"Metric": "Best Candle", "Value": f"{metrics['best_day']:+.2f}%", "Status": "Info"},
            {"Metric": "Worst Candle", "Value": f"{metrics['worst_day']:+.2f}%", "Status": ("Healthy" if metrics["worst_day"] > -4 else ("Watch" if metrics["worst_day"] > -8 else "Risky"))},
            {"Metric": "Skewness", "Value": f"{metrics['skewness']:.3f}", "Status": ("Healthy" if metrics["skewness"] > -0.3 else ("Watch" if metrics["skewness"] > -1.0 else "Risky"))},
            {"Metric": "Kurtosis", "Value": f"{metrics['kurtosis']:.3f}", "Status": ("Healthy" if metrics["kurtosis"] < 3 else ("Watch" if metrics["kurtosis"] < 6 else "Risky"))},
            {"Metric": "Max Drawdown Duration", "Value": f"{int(metrics['max_dd_duration'])} candles", "Status": ("Healthy" if metrics["max_dd_duration"] < 40 else ("Watch" if metrics["max_dd_duration"] < 100 else "Risky"))},
        ]
    )
    st.markdown(
        f"<div style='color:{TEXT_MUTED}; font-size:0.83rem; margin:2px 0 8px 0; line-height:1.6;'>"
        f"<b style='color:{ACCENT};'>Table Metric Guide:</b> "
        f"{_tip('Total Return', 'Overall % change from first to last candle in selected lookback.')} | "
        f"{_tip('Annualized Return', 'Return projected to 1 year based on selected timeframe statistics.')} | "
        f"{_tip('Win Rate', 'Percent of candles with positive return.')} | "
        f"{_tip('Best/Worst Candle', 'Single best and worst return observations in the dataset.')} | "
        f"{_tip('Skewness', 'Direction of tail asymmetry. Negative means heavier left tail.')} | "
        f"{_tip('Kurtosis', 'Tail thickness vs normal distribution. Higher means more extreme outliers.')} | "
        f"{_tip('Max Drawdown Duration', 'Longest continuous underwater period before a new equity high.')}"
        f"</div>",
        unsafe_allow_html=True,
    )
    st.markdown(f"<b style='color:{ACCENT};'>Risk Summary Table</b>", unsafe_allow_html=True)
    def _status_style(v: str) -> str:
        if v == "Healthy":
            return f"color:{POSITIVE}; font-weight:700;"
        if v == "Watch":
            return f"color:{WARNING}; font-weight:700;"
        if v == "Risky":
            return f"color:{NEGATIVE}; font-weight:700;"
        return f"color:{TEXT_MUTED}; font-weight:600;"

    st.dataframe(
        summary_df.style.map(_status_style, subset=["Status"]),
        width="stretch",
        hide_index=True,
    )

    dd_series = metrics["drawdown_series"]
    ts_vals = df["timestamp"].iloc[1 : len(dd_series) + 1] if "timestamp" in df.columns else list(range(len(dd_series)))
    fig_dd = go.Figure()
    fig_dd.add_trace(
        go.Scatter(
            x=ts_vals,
            y=dd_series.values * 100,
            fill="tozeroy",
            fillcolor="rgba(255, 51, 102, 0.2)",
            line=dict(color=NEGATIVE, width=1.6),
            name="Drawdown %",
        )
    )
    fig_dd.update_layout(
        height=250,
        template="plotly_dark",
        title="Drawdown Curve",
        yaxis_title="Drawdown (%)",
        margin=dict(l=20, r=20, t=45, b=25),
        paper_bgcolor=PRIMARY_BG,
    )
    st.plotly_chart(fig_dd, width="stretch")

    cum_ret = metrics["cumulative_returns"]
    ts_vals2 = df["timestamp"].iloc[1 : len(cum_ret) + 1] if "timestamp" in df.columns else list(range(len(cum_ret)))
    fig_cum = go.Figure()
    fig_cum.add_trace(
        go.Scatter(
            x=ts_vals2,
            y=(cum_ret.values - 1) * 100,
            fill="tozeroy",
            fillcolor="rgba(0, 212, 255, 0.1)",
            line=dict(color=NEON_BLUE, width=2),
            name="Cumulative Return %",
        )
    )
    fig_cum.add_hline(y=0, line=dict(color=TEXT_MUTED, dash="dash", width=1))
    fig_cum.update_layout(
        height=290,
        template="plotly_dark",
        title="Cumulative Returns",
        yaxis_title="Return (%)",
        margin=dict(l=20, r=20, t=45, b=25),
        paper_bgcolor=PRIMARY_BG,
    )
    st.plotly_chart(fig_cum, width="stretch")

    # Distribution + rolling volatility (subplots avoided for simpler Streamlit rendering performance).
    returns_pct = returns * 100.0
    fig_dist = go.Figure()
    fig_dist.add_trace(go.Histogram(x=returns_pct, nbinsx=50, marker_color=NEON_PURPLE, opacity=0.72))
    fig_dist.add_vline(x=0, line=dict(color=ACCENT, dash="dash", width=1))
    fig_dist.add_vline(
        x=var95,
        line=dict(color=NEGATIVE, dash="dot", width=2),
        annotation_text=f"VaR 95%: {var95:.2f}%",
    )
    fig_dist.add_vline(
        x=cvar95,
        line=dict(color=WARNING, dash="dot", width=2),
        annotation_text=f"CVaR 95%: {cvar95:.2f}%",
    )
    fig_dist.update_layout(
        height=250,
        template="plotly_dark",
        title="Return Distribution",
        xaxis_title="Return (%)",
        yaxis_title="Frequency",
        margin=dict(l=20, r=20, t=45, b=25),
        paper_bgcolor=PRIMARY_BG,
    )
    st.plotly_chart(fig_dist, width="stretch")

    fig_rv = go.Figure()
    if not rolling_vol.empty:
        rv_ts = df["timestamp"].iloc[len(df) - len(rolling_vol):] if "timestamp" in df.columns else list(range(len(rolling_vol)))
        fig_rv.add_trace(
            go.Scatter(
                x=rv_ts,
                y=rolling_vol.values,
                mode="lines",
                line=dict(color=WARNING, width=2),
                name=f"Rolling Vol ({rolling_window})",
            )
        )
    fig_rv.update_layout(
        height=250,
        template="plotly_dark",
        title=f"Rolling Volatility ({rolling_window} candles)",
        yaxis_title="Std Dev (%)",
        margin=dict(l=20, r=20, t=45, b=25),
        paper_bgcolor=PRIMARY_BG,
    )
    st.plotly_chart(fig_rv, width="stretch")
