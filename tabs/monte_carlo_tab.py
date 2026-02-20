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
    monte_carlo_simulation = get_ctx(ctx, "monte_carlo_simulation")
    """Monte Carlo scenario engine with timeframe-aware horizon scaling."""

    st.markdown(
        f"""
        <style>
        .mc-kpi-grid {{
            display:grid;
            grid-template-columns:repeat(4,minmax(0,1fr));
            gap:10px;
            margin:8px 0 12px 0;
        }}
        .mc-kpi {{
            border:1px solid rgba(0,212,255,0.16);
            border-radius:12px;
            padding:12px 14px;
            background:linear-gradient(140deg, rgba(0,0,0,0.72), rgba(10,18,30,0.88));
        }}
        .mc-kpi-label {{
            color:{TEXT_MUTED};
            font-size:0.70rem;
            text-transform:uppercase;
            letter-spacing:0.8px;
        }}
        .mc-kpi-value {{
            color:{ACCENT};
            font-size:1.2rem;
            font-weight:700;
            margin-top:4px;
        }}
        .mc-badge {{
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

    st.markdown(f"<h2 style='color:{ACCENT};'>Monte Carlo Simulation</h2>", unsafe_allow_html=True)
    st.markdown(
        f"<div class='panel-box'>"
        f"<b style='color:{ACCENT}; font-size:1rem;'>What does this tab show?</b>"
        f"<p style='color:{TEXT_MUTED}; font-size:0.9rem; margin-top:6px; line-height:1.6;'>"
        f"Projects many possible future price paths from historical return distribution. "
        f"Use it for probability-aware planning, not exact price prediction."
        f"</p>"
        f"<p style='color:{TEXT_MUTED}; font-size:0.85rem; margin-top:6px; line-height:1.6;'>"
        f"{_tip('Profit Probability', 'Share of simulated paths ending above current price at horizon.')} | "
        f"{_tip('Expected Return', 'Average terminal return across all simulations.')} | "
        f"{_tip('VaR 95%', '5th percentile terminal return: downside threshold in bad scenarios.')} | "
        f"{_tip('Median Target', 'Middle terminal price; more robust than mean against outliers.')}"
        f"</p>"
        f"</div>",
        unsafe_allow_html=True,
    )

    c1, c2, c3 = st.columns(3)
    with c1:
        mc_coin = _normalize_coin_input(st.text_input("Coin", value="BTC", key="mc_coin"))
    with c2:
        mc_sims = st.slider("Simulations", 200, 3000, 800, step=100, key="mc_sims")
    with c3:
        mc_days = st.slider("Forecast Days", 7, 120, 30, key="mc_days")
    mc_tf = st.selectbox("Base Timeframe", ["1h", "4h", "1d"], index=2, key="mc_tf")

    if not st.button("Run Simulation", type="primary", key="mc_run"):
        return

    _val_err = _validate_coin_symbol(mc_coin)
    if _val_err:
        st.error(_val_err)
        return

    tf_to_steps_per_day = {"1h": 24, "4h": 6, "1d": 1}
    steps_per_day = tf_to_steps_per_day.get(mc_tf, 1)
    horizon_steps = int(mc_days * steps_per_day)

    with st.spinner(f"Running {mc_sims} simulations ({mc_days} days, {horizon_steps} steps)..."):
        # Need enough data for stable distribution estimate.
        df = fetch_ohlcv(mc_coin, mc_tf, limit=700)
        if df is None or len(df) < 60:
            st.error("Not enough data. Try a different coin/timeframe.")
            return

        result = monte_carlo_simulation(df, num_simulations=mc_sims, num_days=horizon_steps)
        if not result:
            st.error("Simulation failed.")
            return

    prob = float(result["prob_profit"]) * 100.0
    exp_ret = float(result["expected_return"])
    var95 = float(result["var_95"])
    median_price = float(result["median_price"])
    last_price = float(result["last_price"])
    median_ret = ((median_price / last_price) - 1.0) * 100.0 if last_price > 0 else 0.0

    prob_status = ("Healthy", POSITIVE) if prob >= 60 else (("Watch", WARNING) if prob >= 45 else ("Risky", NEGATIVE))
    ret_status = ("Healthy", POSITIVE) if exp_ret > 3 else (("Watch", WARNING) if exp_ret >= -2 else ("Risky", NEGATIVE))
    var_status = ("Healthy", POSITIVE) if var95 >= -8 else (("Watch", WARNING) if var95 >= -15 else ("Risky", NEGATIVE))
    med_status = ("Healthy", POSITIVE) if median_ret > 2 else (("Watch", WARNING) if median_ret >= -2 else ("Risky", NEGATIVE))

    st.markdown(
        f"<div class='mc-kpi-grid'>"
        f"<div class='mc-kpi'><div class='mc-kpi-label'>Profit Probability</div><div class='mc-kpi-value'>{prob:.1f}%</div>"
        f"<span class='mc-badge' style='color:{prob_status[1]}; border-color:{prob_status[1]};'><span>&#9679;</span>{prob_status[0]}</span></div>"
        f"<div class='mc-kpi'><div class='mc-kpi-label'>Expected Return</div><div class='mc-kpi-value'>{exp_ret:+.2f}%</div>"
        f"<span class='mc-badge' style='color:{ret_status[1]}; border-color:{ret_status[1]};'><span>&#9679;</span>{ret_status[0]}</span></div>"
        f"<div class='mc-kpi'><div class='mc-kpi-label'>VaR 95%</div><div class='mc-kpi-value'>{var95:.2f}%</div>"
        f"<span class='mc-badge' style='color:{var_status[1]}; border-color:{var_status[1]};'><span>&#9679;</span>{var_status[0]}</span></div>"
        f"<div class='mc-kpi'><div class='mc-kpi-label'>Median Target</div><div class='mc-kpi-value'>${median_price:,.2f}</div>"
        f"<span class='mc-badge' style='color:{med_status[1]}; border-color:{med_status[1]};'><span>&#9679;</span>{med_status[0]}</span></div>"
        f"</div>",
        unsafe_allow_html=True,
    )

    st.markdown(
        f"<details style='margin-bottom:0.7rem;'>"
        f"<summary style='color:{ACCENT}; cursor:pointer;'>How to read quickly (?)</summary>"
        f"<div style='color:{TEXT_MUTED}; font-size:0.85rem; line-height:1.7; margin-top:0.5rem;'>"
        f"<b>1.</b> Check Profit Probability and VaR together, not separately.<br>"
        f"<b>2.</b> If expected return is positive but VaR is deeply negative, size down.<br>"
        f"<b>3.</b> Median target is usually better than max/best-case for planning.<br>"
        f"<b>4.</b> This model assumes historical volatility structure persists."
        f"</div></details>",
        unsafe_allow_html=True,
    )

    st.caption(
        f"Horizon scaling: {mc_days} calendar days x {steps_per_day} steps/day ({mc_tf}) = {horizon_steps} simulation steps."
    )

    sims = result["simulations"]
    horizon_range = list(range(1, horizon_steps + 1))
    sample_count = min(120, mc_sims)
    sample_idx = np.random.choice(mc_sims, sample_count, replace=False)

    fig = go.Figure()
    for idx in sample_idx:
        fig.add_trace(
            go.Scatter(
                x=horizon_range,
                y=sims[idx],
                mode="lines",
                line=dict(width=0.5, color="rgba(0, 212, 255, 0.08)"),
                showlegend=False,
                hoverinfo="skip",
            )
        )
    p5 = np.percentile(sims, 5, axis=0)
    p25 = np.percentile(sims, 25, axis=0)
    p50 = np.percentile(sims, 50, axis=0)
    p75 = np.percentile(sims, 75, axis=0)
    p95 = np.percentile(sims, 95, axis=0)
    fig.add_trace(go.Scatter(x=horizon_range, y=p95, mode="lines", line=dict(width=0), showlegend=False))
    fig.add_trace(
        go.Scatter(
            x=horizon_range,
            y=p5,
            mode="lines",
            fill="tonexty",
            fillcolor="rgba(0, 212, 255, 0.1)",
            line=dict(width=0),
            name="90% CI",
        )
    )
    fig.add_trace(go.Scatter(x=horizon_range, y=p75, mode="lines", line=dict(width=0), showlegend=False))
    fig.add_trace(
        go.Scatter(
            x=horizon_range,
            y=p25,
            mode="lines",
            fill="tonexty",
            fillcolor="rgba(178, 75, 243, 0.15)",
            line=dict(width=0),
            name="50% CI",
        )
    )
    fig.add_trace(go.Scatter(x=horizon_range, y=p50, mode="lines", line=dict(color=NEON_BLUE, width=2), name="Median Path"))
    fig.add_hline(y=last_price, line=dict(color=WARNING, dash="dash", width=1), annotation_text=f"Current: ${last_price:,.2f}")
    fig.update_layout(
        height=500,
        template="plotly_dark",
        title=f"Monte Carlo Path Cloud — {mc_coin}",
        xaxis_title=f"Steps ({mc_tf})",
        yaxis_title="Price ($)",
        margin=dict(l=20, r=20, t=50, b=30),
        paper_bgcolor=PRIMARY_BG,
    )
    st.plotly_chart(fig, width="stretch")

    final_prices = sims[:, -1]
    dist_df = pd.DataFrame(
        [
            {"Metric": "Worst Case", "Value": f"${result['min_price']:,.2f}", "Status": "Risky"},
            {"Metric": "5th Percentile", "Value": f"${result['p5']:,.2f}", "Status": "Watch"},
            {"Metric": "Median", "Value": f"${result['median_price']:,.2f}", "Status": "Base Case"},
            {"Metric": "95th Percentile", "Value": f"${result['p95']:,.2f}", "Status": "Watch"},
            {"Metric": "Best Case", "Value": f"${result['max_price']:,.2f}", "Status": "Optimistic"},
        ]
    )
    st.markdown(f"<b style='color:{ACCENT};'>Terminal Price Summary</b>", unsafe_allow_html=True)
    st.dataframe(dist_df, width="stretch", hide_index=True)

    fig_hist = go.Figure()
    fig_hist.add_trace(go.Histogram(x=final_prices, nbinsx=50, marker_color=NEON_PURPLE, opacity=0.75))
    fig_hist.add_vline(x=last_price, line=dict(color=WARNING, dash="dash", width=2), annotation_text="Current")
    fig_hist.add_vline(x=median_price, line=dict(color=NEON_BLUE, dash="dot", width=2), annotation_text="Median")
    fig_hist.update_layout(
        height=300,
        template="plotly_dark",
        title="Final Price Distribution",
        xaxis_title="Price ($)",
        yaxis_title="Frequency",
        margin=dict(l=20, r=20, t=45, b=25),
        paper_bgcolor=PRIMARY_BG,
    )
    st.plotly_chart(fig_hist, width="stretch")
