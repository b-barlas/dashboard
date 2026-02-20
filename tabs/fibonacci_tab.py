from ui.ctx import get_ctx

import pandas as pd
import plotly.graph_objs as go


def render(ctx: dict) -> None:
    st = get_ctx(ctx, "st")
    ACCENT = get_ctx(ctx, "ACCENT")
    TEXT_MUTED = get_ctx(ctx, "TEXT_MUTED")
    POSITIVE = get_ctx(ctx, "POSITIVE")
    NEGATIVE = get_ctx(ctx, "NEGATIVE")
    WARNING = get_ctx(ctx, "WARNING")
    GOLD = get_ctx(ctx, "GOLD")
    PRIMARY_BG = get_ctx(ctx, "PRIMARY_BG")
    _tip = get_ctx(ctx, "_tip")
    _normalize_coin_input = get_ctx(ctx, "_normalize_coin_input")
    _validate_coin_symbol = get_ctx(ctx, "_validate_coin_symbol")
    fetch_ohlcv = get_ctx(ctx, "fetch_ohlcv")
    calculate_fibonacci_levels = get_ctx(ctx, "calculate_fibonacci_levels")
    detect_divergence = get_ctx(ctx, "detect_divergence")
    calculate_volume_profile = get_ctx(ctx, "calculate_volume_profile")
    detect_market_regime = get_ctx(ctx, "detect_market_regime")
    """Decision-oriented Fibonacci tab."""

    st.markdown(f"<h2 style='color:{ACCENT};'>Fibonacci Analysis</h2>", unsafe_allow_html=True)
    st.markdown(
        f"<div class='panel-box'>"
        f"<b style='color:{ACCENT}; font-size:1rem;'>What this page is for</b>"
        f"<p style='color:{TEXT_MUTED}; font-size:0.9rem; margin-top:6px; line-height:1.6;'>"
        f"Converts structure into an actionable map: "
        f"{_tip('Context', 'Trend + regime + POC position.')}, "
        f"{_tip('Execution Levels', 'Most relevant Fib levels for pullback and extension targets.')}, "
        f"{_tip('Warnings', 'Divergence and volatility signs that reduce setup quality.')}."
        f"</p></div>",
        unsafe_allow_html=True,
    )

    c1, c2, c3 = st.columns(3)
    with c1:
        fib_coin = _normalize_coin_input(st.text_input("Coin", value="BTC", key="fib_coin"))
    with c2:
        fib_tf = st.selectbox("Timeframe", ["5m", "15m", "1h", "4h", "1d"], index=2, key="fib_tf")
    with c3:
        fib_lookback = st.slider("Lookback Bars", 30, 500, 120, key="fib_lookback")

    if not st.button("Build Fib Decision Map", type="primary", key="fib_run"):
        return

    _val_err = _validate_coin_symbol(fib_coin)
    if _val_err:
        st.error(_val_err)
        return

    with st.spinner("Building decision map..."):
        df = fetch_ohlcv(fib_coin, fib_tf, limit=fib_lookback)
        if df is None or len(df) < 20:
            st.error("Not enough data.")
            return

        levels = calculate_fibonacci_levels(df, lookback=fib_lookback)
        if not levels:
            st.error("Could not calculate levels.")
            return

        divergences = detect_divergence(df)
        vp = calculate_volume_profile(df)
        regime = detect_market_regime(df)

    current = float(df["close"].iloc[-1])
    is_uptrend = bool(levels.get("_is_uptrend", True))
    regime_name = str(regime.get("regime", "UNKNOWN"))
    regime_desc = str(regime.get("description", ""))
    regime_color = regime.get("color", WARNING)
    div_count = len(divergences or [])
    poc = float(vp.get("poc_price", current)) if vp else current
    poc_dist = ((current - poc) / current) * 100 if current > 0 else 0.0

    key_levels = ["38.2%", "50%", "61.8%", "100%", "161.8%"]
    nearest = []
    for lv in key_levels:
        p = float(levels.get(lv, 0) or 0)
        if p <= 0:
            continue
        d = abs((current - p) / current) * 100 if current > 0 else 0.0
        nearest.append((lv, p, d))
    nearest.sort(key=lambda x: x[2])
    nearest_level = nearest[0][0] if nearest else "N/A"
    nearest_dist = nearest[0][2] if nearest else 0.0

    # Top decision summary
    trend_text = "Bullish pullback context" if is_uptrend else "Bearish pullback context"
    if nearest_dist <= 1.0 and div_count == 0:
        decision = "Setup quality: STRONG"
        decision_color = POSITIVE
        decision_note = f"Price is near {nearest_level} with no strong divergence conflict."
    elif nearest_dist <= 2.5:
        decision = "Setup quality: MODERATE"
        decision_color = WARNING
        decision_note = f"Price is near {nearest_level}, but confirmation is still needed."
    else:
        decision = "Setup quality: WEAK"
        decision_color = NEGATIVE
        decision_note = f"Price is far from key Fib zones ({nearest_level} is {nearest_dist:.2f}% away)."

    st.markdown(
        f"<div class='panel-box' style='border-left:4px solid {decision_color};'>"
        f"<b style='color:{decision_color};'>{decision}</b><br>"
        f"<span style='color:{TEXT_MUTED};'>{trend_text} | Regime: {regime_name} | {decision_note}</span>"
        f"</div>",
        unsafe_allow_html=True,
    )

    st.markdown(
        f"<details style='margin-bottom:0.7rem;'>"
        f"<summary style='color:{ACCENT}; cursor:pointer;'>How to read in 15 seconds (?)</summary>"
        f"<div style='color:{TEXT_MUTED}; font-size:0.85rem; line-height:1.7; margin-top:0.5rem;'>"
        f"<b>1.</b> Read the decision box first.<br>"
        f"<b>2.</b> Check nearest execution levels (38.2/50/61.8).<br>"
        f"<b>3.</b> If divergence warning exists, reduce size or wait confirmation.<br>"
        f"<b>4.</b> Use Spot/Position tab to finalize entry trigger."
        f"</div></details>",
        unsafe_allow_html=True,
    )

    # Block 1: Context
    st.markdown(f"<b style='color:{ACCENT};'>1) Context</b>", unsafe_allow_html=True)
    if abs(poc_dist) <= 1.0:
        poc_status = "● Near POC"
    elif abs(poc_dist) <= 3.0:
        poc_status = "■ Mid Distance"
    else:
        poc_status = "• Far from POC"

    if div_count == 0:
        div_status = "● Clean"
    elif div_count <= 2:
        div_status = "■ Watch"
    else:
        div_status = "▼ High Conflict"

    regime_icon = "▲" if "TREND" in regime_name else ("■" if regime_name in {"RANGING", "COMPRESSION"} else "▼")
    context_df = pd.DataFrame(
        [
            {"Metric": "Trend Bias (?)", "Value": ("▲ Bullish" if is_uptrend else "▼ Bearish"), "Status": ("● Supportive" if is_uptrend else "▼ Caution")},
            {"Metric": "Regime (?)", "Value": f"{regime_icon} {regime_name}", "Status": ("● Trend-Friendly" if "TREND" in regime_name else ("■ Neutral" if regime_name in {"RANGING", "COMPRESSION"} else "▼ Risky"))},
            {"Metric": "POC Distance (?)", "Value": f"{poc_dist:+.2f}%", "Status": poc_status},
            {"Metric": "Divergence Alerts (?)", "Value": str(div_count), "Status": div_status},
        ]
    )
    st.markdown(
        f"<div style='color:{TEXT_MUTED}; font-size:0.83rem; margin:2px 0 8px 0; line-height:1.6;'>"
        f"<b style='color:{ACCENT};'>Metric Guide:</b> "
        f"{_tip('Regime', 'Current market state. Trending regimes support continuation; ranging/compression require patience.')} | "
        f"{_tip('POC Distance', 'Distance to highest-volume price node. Near POC often means price magnet/decision area.')} | "
        f"{_tip('Divergence Alerts', 'Momentum disagreement count. Higher count increases reversal/conflict risk.')}"
        f"</div>",
        unsafe_allow_html=True,
    )

    def _ctx_style(v: str) -> str:
        s = str(v)
        if "Supportive" in s or "Trend-Friendly" in s or "Near POC" in s or "Clean" in s:
            return f"color:{POSITIVE}; font-weight:700;"
        if "Neutral" in s or "Mid Distance" in s or "Watch" in s:
            return f"color:{WARNING}; font-weight:700;"
        if "Caution" in s or "Risky" in s or "Far from POC" in s or "High Conflict" in s:
            return f"color:{NEGATIVE}; font-weight:700;"
        return f"color:{TEXT_MUTED};"

    st.dataframe(
        context_df.style.map(_ctx_style, subset=["Value", "Status"]),
        width="stretch",
        hide_index=True,
    )
    st.caption(f"Regime detail: {regime_desc}")

    # Block 2: Execution levels
    st.markdown(f"<b style='color:{ACCENT};'>2) Execution Levels</b>", unsafe_allow_html=True)
    exec_rows = []
    for lv in key_levels:
        p = float(levels.get(lv, 0) or 0)
        if p <= 0:
            continue
        dist = ((current - p) / current) * 100 if current > 0 else 0.0
        if abs(dist) <= 1.0:
            zone = "● In Zone"
        elif abs(dist) <= 2.5:
            zone = "■ Near"
        else:
            zone = "• Far"
        exec_rows.append({"Level": lv, "Price": p, "Distance %": dist, "Zone": zone})
    exec_df = pd.DataFrame(exec_rows)

    def _zone_style(v: str) -> str:
        if "In Zone" in v:
            return f"color:{POSITIVE}; font-weight:700;"
        if "Near" in v:
            return f"color:{WARNING}; font-weight:700;"
        return f"color:{TEXT_MUTED};"

    st.dataframe(
        exec_df.style.format({"Price": "{:,.2f}", "Distance %": "{:+.2f}"}).map(_zone_style, subset=["Zone"]),
        width="stretch",
        hide_index=True,
    )

    # Block 3: Warnings
    st.markdown(f"<b style='color:{ACCENT};'>3) Warnings</b>", unsafe_allow_html=True)
    if divergences:
        warn_rows = []
        for d in divergences:
            d_type = str(d.get("type", "DIVERGENCE"))
            d_strength = str(d.get("strength", ""))
            icon = "▲" if "BULLISH" in d_type else ("▼" if "BEARISH" in d_type else "■")
            warn_rows.append(
                {
                    "Signal": f"{icon} {d_type}",
                    "Strength": d_strength,
                    "Comment": str(d.get("description", "")),
                }
            )
        st.dataframe(pd.DataFrame(warn_rows), width="stretch", hide_index=True)
    else:
        st.success("No major divergence warning in selected lookback.")

    # Lean chart with only key levels + POC (less visual noise).
    fig = go.Figure()
    fig.add_trace(
        go.Candlestick(
            x=df["timestamp"],
            open=df["open"],
            high=df["high"],
            low=df["low"],
            close=df["close"],
            increasing_line_color=POSITIVE,
            decreasing_line_color=NEGATIVE,
            name="Price",
        )
    )
    level_colors = {"38.2%": WARNING, "50%": ACCENT, "61.8%": "#B24BF3", "100%": "#FFFFFF", "161.8%": GOLD}
    for lv in key_levels:
        p = float(levels.get(lv, 0) or 0)
        if p <= 0:
            continue
        fig.add_hline(
            y=p,
            line=dict(color=level_colors.get(lv, TEXT_MUTED), dash="dot", width=1.3),
            annotation_text=f"{lv}: ${p:,.2f}",
            annotation_font=dict(size=9, color=level_colors.get(lv, TEXT_MUTED)),
        )
    fig.add_hline(
        y=poc,
        line=dict(color=GOLD, dash="dash", width=1.4),
        annotation_text=f"POC: ${poc:,.2f}",
        annotation_font=dict(size=9, color=GOLD),
    )
    fig.update_layout(
        height=520,
        template="plotly_dark",
        title=f"Fib Decision Chart — {fib_coin} ({fib_tf})",
        margin=dict(l=20, r=20, t=50, b=30),
        xaxis_rangeslider_visible=False,
        paper_bgcolor=PRIMARY_BG,
    )
    st.plotly_chart(fig, width="stretch")
