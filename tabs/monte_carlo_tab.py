from __future__ import annotations

import math
import zlib

import numpy as np
import pandas as pd
import plotly.graph_objs as go

from ui.ctx import get_ctx


def _mc_thresholds(days: int) -> dict[str, float]:
    safe_days = max(1, int(days))
    risk_scale = min(2.2, math.sqrt(safe_days / 30.0))
    linear_scale = safe_days / 30.0
    if safe_days <= 30:
        prob_healthy, prob_watch = 60.0, 48.0
    elif safe_days <= 90:
        prob_healthy, prob_watch = 58.0, 47.0
    else:
        prob_healthy, prob_watch = 56.0, 46.0
    return {
        "prob_healthy": prob_healthy,
        "prob_watch": prob_watch,
        "ret_healthy": 3.0 * linear_scale,
        "ret_watch": -2.0 * linear_scale,
        "var_healthy": -8.0 * risk_scale,
        "var_watch": -15.0 * risk_scale,
        "cvar_healthy": -10.0 * risk_scale,
        "cvar_watch": -18.0 * risk_scale,
        "med_healthy": 2.0 * linear_scale,
        "med_watch": -2.0 * linear_scale,
    }


def _band_from_floor(
    value: float,
    healthy_floor: float,
    watch_floor: float,
    *,
    positive_color: str,
    warning_color: str,
    negative_color: str,
) -> tuple[str, str]:
    if value >= healthy_floor:
        return "Healthy", positive_color
    if value >= watch_floor:
        return "Watch", warning_color
    return "Risky", negative_color


def _history_limit(horizon_steps: int) -> int:
    # Keep enough context for robust return distribution on longer horizons.
    return int(min(4000, max(700, round(horizon_steps * 1.8))))


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
        @media (max-width: 1100px) {{
            .mc-kpi-grid {{ grid-template-columns:repeat(2,minmax(0,1fr)); }}
        }}
        @media (max-width: 640px) {{
            .mc-kpi-grid {{ grid-template-columns:1fr; }}
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(f"<h2 style='color:{ACCENT};'>Monte Carlo Scenario Engine</h2>", unsafe_allow_html=True)
    st.markdown(
        f"<div class='panel-box'>"
        f"<b style='color:{ACCENT}; font-size:1rem;'>What does this tab show?</b>"
        f"<p style='color:{TEXT_MUTED}; font-size:0.9rem; margin-top:6px; line-height:1.6;'>"
        f"Projects many possible future paths from historical return behavior. "
        f"Use this tab for probability-aware risk planning, not direct entry timing."
        f"</p>"
        f"<p style='color:{TEXT_MUTED}; font-size:0.85rem; margin-top:6px; line-height:1.6;'>"
        f"{_tip('Profit Probability', 'Share of simulations ending above current price at the selected horizon.')} | "
        f"{_tip('Expected Return', 'Average terminal return across all simulations at horizon.')} | "
        f"{_tip('VaR 95%', '5th percentile terminal return; downside line in bad scenarios.')} | "
        f"{_tip('CVaR 95%', 'Average terminal return in worst 5% paths; deeper tail-risk gauge.')} | "
        f"{_tip('Median Target', 'Middle terminal price; robust planning anchor less sensitive to outliers.')}"
        f"</p>"
        f"</div>",
        unsafe_allow_html=True,
    )

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        mc_coin = _normalize_coin_input(st.text_input("Coin", value="BTC", key="mc_coin"))
    with c2:
        mc_tf = st.selectbox("Base Timeframe", ["1h", "4h", "1d"], index=2, key="mc_tf")
    with c3:
        mc_sims = st.slider("Simulations", 200, 3000, 800, step=100, key="mc_sims")
    with c4:
        mc_days = st.slider("Forecast Days", 7, 120, 30, key="mc_days")

    if not st.button("Run Simulation", type="primary", key="mc_run"):
        return

    validation_error = _validate_coin_symbol(mc_coin)
    if validation_error:
        st.error(validation_error)
        return

    tf_to_steps_per_day = {"1h": 24, "4h": 6, "1d": 1}
    steps_per_day = tf_to_steps_per_day.get(mc_tf, 1)
    horizon_steps = int(mc_days * steps_per_day)
    fetch_limit = _history_limit(horizon_steps)
    min_required_rows = max(90, int(horizon_steps * 0.70))
    seed = zlib.crc32(f"{mc_coin}|{mc_tf}|{mc_days}|{mc_sims}".encode("utf-8")) & 0xFFFFFFFF

    with st.spinner(f"Running {mc_sims} simulations ({mc_days} days, {horizon_steps} steps)..."):
        df = fetch_ohlcv(mc_coin, mc_tf, limit=fetch_limit)
        if df is None or len(df) < 60:
            st.error("Not enough data. Try a different coin/timeframe.")
            return
        if len(df) < min_required_rows:
            st.warning(
                f"History is limited for this horizon ({len(df)} rows vs recommended {min_required_rows}). "
                "Interpret probability and tail metrics as lower-confidence."
            )

        result = monte_carlo_simulation(
            df,
            num_simulations=mc_sims,
            num_days=horizon_steps,
            seed=seed,
        )
        if not result:
            st.error("Simulation failed.")
            return

    thresholds = _mc_thresholds(mc_days)
    prob = float(result["prob_profit"]) * 100.0
    exp_ret = float(result["expected_return"])
    var95 = float(result["var_95"])
    cvar95 = float(result.get("cvar_95", var95))
    median_price = float(result["median_price"])
    last_price = float(result["last_price"])
    median_ret = ((median_price / last_price) - 1.0) * 100.0 if last_price > 0 else 0.0

    prob_status = _band_from_floor(
        prob,
        thresholds["prob_healthy"],
        thresholds["prob_watch"],
        positive_color=POSITIVE,
        warning_color=WARNING,
        negative_color=NEGATIVE,
    )
    ret_status = _band_from_floor(
        exp_ret,
        thresholds["ret_healthy"],
        thresholds["ret_watch"],
        positive_color=POSITIVE,
        warning_color=WARNING,
        negative_color=NEGATIVE,
    )
    var_status = _band_from_floor(
        var95,
        thresholds["var_healthy"],
        thresholds["var_watch"],
        positive_color=POSITIVE,
        warning_color=WARNING,
        negative_color=NEGATIVE,
    )
    cvar_status = _band_from_floor(
        cvar95,
        thresholds["cvar_healthy"],
        thresholds["cvar_watch"],
        positive_color=POSITIVE,
        warning_color=WARNING,
        negative_color=NEGATIVE,
    )
    med_status = _band_from_floor(
        median_ret,
        thresholds["med_healthy"],
        thresholds["med_watch"],
        positive_color=POSITIVE,
        warning_color=WARNING,
        negative_color=NEGATIVE,
    )

    risky_count = sum(x[0] == "Risky" for x in [prob_status, ret_status, var_status, cvar_status, med_status])
    healthy_count = sum(x[0] == "Healthy" for x in [prob_status, ret_status, var_status, cvar_status, med_status])
    if risky_count >= 2 or cvar_status[0] == "Risky":
        decision_tone = "Defensive"
        decision_color = NEGATIVE
        decision_note = "Tail-risk profile is heavy. Reduce size and avoid aggressive exposure."
    elif healthy_count >= 3 and prob_status[0] == "Healthy":
        decision_tone = "Favourable"
        decision_color = POSITIVE
        decision_note = "Probability and tail-risk profile are constructive for selective risk-on plans."
    else:
        decision_tone = "Selective"
        decision_color = WARNING
        decision_note = "Mixed profile. Keep sizing disciplined and require stronger confluence."

    st.markdown(
        f"<div class='mc-kpi-grid'>"
        f"<div class='mc-kpi'><div class='mc-kpi-label'>Profit Probability</div><div class='mc-kpi-value'>{prob:.1f}%</div>"
        f"<span class='mc-badge' style='color:{prob_status[1]}; border-color:{prob_status[1]};'><span>&#9679;</span>{prob_status[0]}</span></div>"
        f"<div class='mc-kpi'><div class='mc-kpi-label'>Expected Return</div><div class='mc-kpi-value'>{exp_ret:+.2f}%</div>"
        f"<span class='mc-badge' style='color:{ret_status[1]}; border-color:{ret_status[1]};'><span>&#9679;</span>{ret_status[0]}</span></div>"
        f"<div class='mc-kpi'><div class='mc-kpi-label'>VaR 95%</div><div class='mc-kpi-value'>{var95:.2f}%</div>"
        f"<span class='mc-badge' style='color:{var_status[1]}; border-color:{var_status[1]};'><span>&#9679;</span>{var_status[0]}</span></div>"
        f"<div class='mc-kpi'><div class='mc-kpi-label'>CVaR 95%</div><div class='mc-kpi-value'>{cvar95:.2f}%</div>"
        f"<span class='mc-badge' style='color:{cvar_status[1]}; border-color:{cvar_status[1]};'><span>&#9679;</span>{cvar_status[0]}</span></div>"
        f"</div>",
        unsafe_allow_html=True,
    )
    st.markdown(
        f"<div class='elite-card' style='margin:2px 0 10px 0; border-color:rgba(0,212,255,0.22);'>"
        f"<div style='display:flex; justify-content:space-between; gap:10px; flex-wrap:wrap;'>"
        f"<span style='color:{TEXT_MUTED}; font-size:0.82rem;'>Risk/Reward Profile: "
        f"<b style='color:{decision_color};'>{decision_tone}</b></span>"
        f"<span style='color:{TEXT_MUTED}; font-size:0.82rem;'>"
        f"<b style='color:{decision_color};'>{decision_note}</b></span>"
        f"</div></div>",
        unsafe_allow_html=True,
    )
    st.markdown(
        f"<div class='mc-kpi-grid' style='grid-template-columns:repeat(2,minmax(0,1fr)); margin-top:-2px;'>"
        f"<div class='mc-kpi'><div class='mc-kpi-label'>Median Target</div><div class='mc-kpi-value'>${median_price:,.2f}</div>"
        f"<span class='mc-badge' style='color:{med_status[1]}; border-color:{med_status[1]};'><span>&#9679;</span>{med_status[0]}</span></div>"
        f"<div class='mc-kpi'><div class='mc-kpi-label'>Current Price</div><div class='mc-kpi-value'>${last_price:,.2f}</div>"
        f"<span class='mc-badge' style='color:{TEXT_MUTED}; border-color:{TEXT_MUTED};'><span>&#9679;</span>Reference</span></div>"
        f"</div>",
        unsafe_allow_html=True,
    )

    st.markdown(
        f"<details style='margin-bottom:0.7rem;'>"
        f"<summary style='color:{ACCENT}; cursor:pointer;'>How to read quickly (?)</summary>"
        f"<div style='color:{TEXT_MUTED}; font-size:0.85rem; line-height:1.7; margin-top:0.5rem;'>"
        f"<b>1.</b> Prioritize Prob + VaR + CVaR together before reading expected return.<br>"
        f"<b>2.</b> If expected return is positive but CVaR is Risky, treat it as fragile edge.<br>"
        f"<b>3.</b> Median target is usually a better planning anchor than best-case tails.<br>"
        f"<b>4.</b> Threshold bands are horizon-aware: longer windows require wider risk tolerance."
        f"</div></details>",
        unsafe_allow_html=True,
    )
    st.caption(
        f"Horizon scaling: {mc_days} days x {steps_per_day} steps/day ({mc_tf}) = {horizon_steps} steps. "
        f"History fetched: {len(df)} rows (recommended >= {min_required_rows})."
    )

    sims = np.asarray(result["simulations"], dtype=float)
    step_count = int(sims.shape[1]) if sims.ndim == 2 else horizon_steps
    horizon_range = np.arange(1, step_count + 1)
    sample_count = min(120, int(sims.shape[0]))
    sample_rng = np.random.default_rng(seed + 17)
    sample_idx = sample_rng.choice(int(sims.shape[0]), sample_count, replace=False)

    st.markdown(f"<b style='color:{ACCENT};'>Path Cloud</b>", unsafe_allow_html=True)
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
            fillcolor="rgba(0, 212, 255, 0.10)",
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
    fig.add_trace(
        go.Scatter(
            x=horizon_range,
            y=p50,
            mode="lines",
            line=dict(color=NEON_BLUE, width=2),
            name="Median Path",
        )
    )
    fig.add_hline(
        y=last_price,
        line=dict(color=WARNING, dash="dash", width=1),
        annotation_text=f"Current: ${last_price:,.2f}",
    )
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
