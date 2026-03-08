from __future__ import annotations

from ui.ctx import get_ctx

import numpy as np
import pandas as pd
import plotly.graph_objs as go
from ui.primitives import render_help_details, render_insight_card, render_kpi_grid, render_page_header


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

    render_page_header(
        st,
        title="Risk Analytics",
        intro_html=(
            "Measures downside risk, volatility, and risk-adjusted performance from historical candles. "
            "Use this tab to answer: <i>Is this coin worth the risk on this timeframe?</i>"
            f"<br><br>{_tip('Sharpe', 'Risk-adjusted return using total volatility. Higher is better.')} | "
            f"{_tip('Sortino', 'Risk-adjusted return using only downside volatility. Higher is better.')} | "
            f"{_tip('Max Drawdown', 'Worst peak-to-trough historical loss in %. Lower is safer.')} | "
            f"{_tip('VaR 95%', '5th percentile return. Typical bad-day threshold; more negative means higher tail risk.')} | "
            f"{_tip('CVaR 95%', 'Average loss on the worst 5% of returns. Captures crash severity better than VaR.')} | "
            f"{_tip('Calmar', 'Annual return divided by max drawdown. Higher means better reward per drawdown risk.')}"
        ),
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

    def _closed_close_series(frame: pd.DataFrame) -> pd.Series:
        close = pd.to_numeric(frame.get("close"), errors="coerce")
        close = close.replace([np.inf, -np.inf], np.nan).dropna()
        close = close[close > 0]
        # Risk analytics should run on closed candles only.
        if len(close) >= 2:
            close = close.iloc[:-1]
        return close

    def _closed_timestamp_series(frame: pd.DataFrame, close_series: pd.Series):
        if "timestamp" not in frame.columns:
            return None
        ts = pd.to_datetime(frame["timestamp"], errors="coerce")
        ts = ts.reindex(close_series.index)
        ts = ts.dropna()
        return ts

    def _tf_thresholds(tf: str) -> dict[str, float]:
        profile = (tf or "").lower().strip()
        if profile == "1h":
            return {
                "regime_dd_high": 45.0,
                "regime_dd_med": 28.0,
                "regime_var_high": -3.2,
                "regime_var_med": -2.0,
                "regime_sharpe_high": 0.2,
                "regime_sharpe_med": 0.8,
                "dd_good": 22.0,
                "dd_neutral": 35.0,
                "var_good": 1.8,
                "var_neutral": 3.0,
                "cvar_good": 2.6,
                "cvar_neutral": 4.5,
            }
        if profile == "4h":
            return {
                "regime_dd_high": 38.0,
                "regime_dd_med": 24.0,
                "regime_var_high": -4.2,
                "regime_var_med": -2.8,
                "regime_sharpe_high": 0.1,
                "regime_sharpe_med": 0.8,
                "dd_good": 18.0,
                "dd_neutral": 30.0,
                "var_good": 2.3,
                "var_neutral": 3.8,
                "cvar_good": 3.2,
                "cvar_neutral": 5.0,
            }
        return {
            "regime_dd_high": 32.0,
            "regime_dd_med": 20.0,
            "regime_var_high": -5.0,
            "regime_var_med": -3.5,
            "regime_sharpe_high": 0.0,
            "regime_sharpe_med": 0.8,
            "dd_good": 15.0,
            "dd_neutral": 27.0,
            "var_good": 3.0,
            "var_neutral": 4.8,
            "cvar_good": 4.2,
            "cvar_neutral": 6.2,
        }

    with st.spinner("Calculating risk metrics..."):
        df = fetch_ohlcv(risk_coin, risk_tf, limit=int(lookback))
        if df is None or len(df) < 30:
            st.error("Not enough data.")
            return
        close_eval = _closed_close_series(df)
        if len(close_eval) < 30:
            st.error("Not enough closed-candle data.")
            return
        ts_eval = _closed_timestamp_series(df, close_eval)
        metrics = calculate_risk_metrics(df, timeframe=risk_tf, close_series=close_eval)
        if not metrics:
            st.error("Could not calculate metrics.")
            return

    returns = close_eval.pct_change().replace([np.inf, -np.inf], np.nan).dropna()
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

    tf_thresholds = _tf_thresholds(risk_tf)

    # Timeframe-aware regime classifier for beginner readability.
    if (
        dd_abs > tf_thresholds["regime_dd_high"]
        or var95 < tf_thresholds["regime_var_high"]
        or sharpe < tf_thresholds["regime_sharpe_high"]
    ):
        regime = "High Risk"
        regime_color = NEGATIVE
    elif (
        dd_abs > tf_thresholds["regime_dd_med"]
        or var95 < tf_thresholds["regime_var_med"]
        or sharpe < tf_thresholds["regime_sharpe_med"]
    ):
        regime = "Medium Risk"
        regime_color = WARNING
    else:
        regime = "Lower Risk"
        regime_color = POSITIVE

    sharpe_s, sharpe_c = _status(sharpe, good=1.2, neutral=0.4)
    sortino_s, sortino_c = _status(sortino, good=1.6, neutral=0.6)
    dd_s, dd_c = _status(dd_abs, good=tf_thresholds["dd_good"], neutral=tf_thresholds["dd_neutral"], lower_better=True)
    var_s, var_c = _status(abs(var95), good=tf_thresholds["var_good"], neutral=tf_thresholds["var_neutral"], lower_better=True)
    cvar_s, cvar_c = _status(abs(cvar95), good=tf_thresholds["cvar_good"], neutral=tf_thresholds["cvar_neutral"], lower_better=True)
    cal_s, cal_c = _status(calmar, good=1.2, neutral=0.6)
    vol_s, vol_c = _status(ann_vol, good=45.0, neutral=80.0, lower_better=True)

    if regime == "Lower Risk" and sharpe >= 1.0 and cvar95 > -5.0:
        profile_text = "Favourable risk profile for selective participation."
        profile_color = POSITIVE
    elif regime == "High Risk" or cvar95 <= -8.0:
        profile_text = "Defensive profile: prioritize capital protection."
        profile_color = NEGATIVE
    else:
        profile_text = "Mixed profile: keep position size controlled."
        profile_color = WARNING

    render_kpi_grid(
        st,
        items=[
            {
                "label": "Risk Regime",
                "value": regime,
                "value_color": regime_color,
                "badge_text": regime,
                "badge_color": regime_color,
                "badge_dot": True,
            },
            {
                "label": "Sharpe",
                "value": f"{sharpe:.2f}",
                "badge_text": sharpe_s,
                "badge_color": sharpe_c,
                "badge_dot": True,
            },
            {
                "label": "Max Drawdown",
                "value": f"{metrics['max_drawdown']:.2f}%",
                "badge_text": dd_s,
                "badge_color": dd_c,
                "badge_dot": True,
            },
            {
                "label": "VaR 95%",
                "value": f"{var95:.2f}%",
                "badge_text": var_s,
                "badge_color": var_c,
                "badge_dot": True,
            },
        ],
    )
    render_insight_card(
        st,
        title="Decision Profile",
        body_html=f"<span style='color:{profile_color}; font-weight:700;'>{profile_text}</span>",
        tone=(
            "positive"
            if profile_color == POSITIVE
            else ("negative" if profile_color == NEGATIVE else "warning")
        ),
    )
    st.markdown(
        f"<div style='color:{TEXT_MUTED}; font-size:0.80rem; margin:0 0 8px 0;'>"
        f"Threshold profile adjusts by timeframe ({risk_tf}) so risk labels stay realistic across 1h / 4h / 1d candles."
        f"</div>",
        unsafe_allow_html=True,
    )

    render_help_details(
        st,
        summary="How to read quickly (?)",
        body_html=(
            "<b>1.</b> Start with <b>Risk Regime</b> and <b>Max Drawdown</b> to estimate downside stress.<br>"
            "<b>2.</b> Check <b>VaR/CVaR</b> for tail risk: more negative values mean deeper bad-day losses.<br>"
            "<b>3.</b> Use <b>Sharpe/Sortino/Calmar</b> to judge if returns justify risk taken.<br>"
            "<b>4.</b> Confirm if volatility is expanding using the rolling-vol chart before sizing up.<br>"
            "<b>5.</b> Metrics run on <b>closed candles</b> to avoid live-candle noise."
        ),
    )

    render_kpi_grid(
        st,
        items=[
            {
                "label": "Sortino",
                "value": f"{sortino:.2f}",
                "badge_text": sortino_s,
                "badge_color": sortino_c,
                "badge_dot": True,
            },
            {
                "label": "Calmar",
                "value": f"{calmar:.2f}",
                "badge_text": cal_s,
                "badge_color": cal_c,
                "badge_dot": True,
            },
            {
                "label": "CVaR 95%",
                "value": f"{cvar95:.2f}%",
                "badge_text": cvar_s,
                "badge_color": cvar_c,
                "badge_dot": True,
            },
            {
                "label": "Annualized Volatility",
                "value": f"{ann_vol:.1f}%",
                "badge_text": vol_s,
                "badge_color": vol_c,
                "badge_dot": True,
            },
        ],
    )

    best_period = float(metrics.get("best_period", metrics.get("best_day", 0.0)))
    worst_period = float(metrics.get("worst_period", metrics.get("worst_day", 0.0)))

    summary_df = pd.DataFrame(
        [
            {"Metric": "Total Return", "Value": f"{metrics['total_return']:+.2f}%", "Status": "Info"},
            {"Metric": "Annualized Return", "Value": f"{metrics['ann_return']:+.2f}%", "Status": "Info"},
            {"Metric": "Win Rate", "Value": f"{metrics['win_rate']:.1f}%", "Status": ("Healthy" if metrics["win_rate"] >= 55 else ("Watch" if metrics["win_rate"] >= 45 else "Risky"))},
            {"Metric": "Best Candle", "Value": f"{best_period:+.2f}%", "Status": "Info"},
            {"Metric": "Worst Candle", "Value": f"{worst_period:+.2f}%", "Status": ("Healthy" if worst_period > -4 else ("Watch" if worst_period > -8 else "Risky"))},
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
    if ts_eval is not None:
        ts_vals = ts_eval.reindex(dd_series.index)
        ts_vals = ts_vals if not ts_vals.isna().all() else list(range(len(dd_series)))
    else:
        ts_vals = list(range(len(dd_series)))
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
    if ts_eval is not None:
        ts_vals2 = ts_eval.reindex(cum_ret.index)
        ts_vals2 = ts_vals2 if not ts_vals2.isna().all() else list(range(len(cum_ret)))
    else:
        ts_vals2 = list(range(len(cum_ret)))
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

    with st.expander("Advanced Risk Charts"):
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
            if ts_eval is not None:
                rv_ts = ts_eval.reindex(rolling_vol.index)
                rv_ts = rv_ts if not rv_ts.isna().all() else list(range(len(rolling_vol)))
            else:
                rv_ts = list(range(len(rolling_vol)))
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
