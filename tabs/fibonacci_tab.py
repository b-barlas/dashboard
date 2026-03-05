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

    def _fmt_price(v: float) -> str:
        p = float(v)
        ap = abs(p)
        if ap >= 100:
            d = 2
        elif ap >= 1:
            d = 4
        elif ap >= 0.1:
            d = 5
        elif ap >= 0.01:
            d = 6
        elif ap >= 0.001:
            d = 7
        else:
            d = 8
        return f"${p:,.{d}f}"

    def _distance_points(dist_pct: float) -> int:
        d = abs(float(dist_pct))
        if d <= 0.80:
            return 34
        if d <= 1.50:
            return 26
        if d <= 2.50:
            return 16
        if d <= 4.00:
            return 8
        return 2

    def _poc_points(poc_dist_pct: float) -> int:
        d = abs(float(poc_dist_pct))
        if d <= 1.00:
            return 18
        if d <= 3.00:
            return 10
        if d <= 5.00:
            return 4
        return 0

    def _regime_points(regime_name: str) -> int:
        m = {
            "STRONG TREND": 22,
            "TRENDING": 16,
            "TRANSITIONING": 10,
            "RANGING": 8,
            "COMPRESSION": 8,
            "HIGH VOLATILITY": 4,
            "UNKNOWN": 6,
        }
        return int(m.get(str(regime_name or "UNKNOWN").upper().strip(), 6))

    def _divergence_impact(rows: list[dict], *, is_uptrend: bool) -> tuple[int, int, int]:
        """Return (net_penalty, conflict_count, supportive_count)."""
        if not rows:
            return 0, 0, 0
        conflict_penalty = 0
        supportive_bonus = 0
        conflict_count = 0
        supportive_count = 0
        for row in rows:
            row_type = str((row or {}).get("type", "")).upper().strip()
            strength = str((row or {}).get("strength", "MODERATE")).upper().strip()
            if strength == "STRONG":
                w_pen, w_bonus = 14, 4
            elif strength == "MODERATE":
                w_pen, w_bonus = 8, 2
            else:
                w_pen, w_bonus = 4, 1
            is_bull = "BULLISH" in row_type
            is_bear = "BEARISH" in row_type
            supportive = (is_uptrend and is_bull) or ((not is_uptrend) and is_bear)
            conflict = (is_uptrend and is_bear) or ((not is_uptrend) and is_bull)
            if supportive:
                supportive_count += 1
                supportive_bonus += w_bonus
            elif conflict:
                conflict_count += 1
                conflict_penalty += w_pen
            else:
                conflict_penalty += int(round(w_pen * 0.35))
        net_penalty = max(0, min(28, conflict_penalty - supportive_bonus))
        return net_penalty, conflict_count, supportive_count

    st.markdown(f"<h2 style='color:{ACCENT};'>Fibonacci Analysis</h2>", unsafe_allow_html=True)
    st.markdown(
        f"<div class='panel-box'>"
        f"<b style='color:{ACCENT}; font-size:1rem;'>What does this tab show?</b>"
        f"<p style='color:{TEXT_MUTED}; font-size:0.9rem; margin-top:6px; line-height:1.6;'>"
        f"Builds a decision-oriented Fib map from "
        f"{_tip('Context', 'Trend + regime + POC distance to detect when structure supports continuation.')}, "
        f"{_tip('Execution Levels', 'Most relevant Fib pullback/extension levels and distance to current price.')}, "
        f"and {_tip('Warnings', 'Divergence alerts that reduce setup quality.')}. "
        f"All calculations use closed candles to avoid live-candle noise."
        f"</p></div>",
        unsafe_allow_html=True,
    )
    st.markdown(
        f"<details style='margin-bottom:0.7rem;'>"
        f"<summary style='color:{ACCENT}; cursor:pointer;'>How to read quickly (?)</summary>"
        f"<div style='color:{TEXT_MUTED}; font-size:0.85rem; line-height:1.7; margin-top:0.5rem;'>"
        f"<b>1.</b> Start with Setup Quality and nearest key Fib.<br>"
        f"<b>2.</b> Check Context (Regime + POC Distance + Divergence Alerts).<br>"
        f"<b>3.</b> Use Execution Levels (38.2/50/61.8 first, 100/161.8 as extension map).<br>"
        f"<b>4.</b> Divergence is direction-aware: conflict alerts reduce quality more than supportive alerts."
        f"</div></details>",
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

        # Closed-candle frame for all analytics and chart consistency.
        df_eval = df.iloc[:-1].copy() if len(df) > 30 else df.copy()
        if df_eval is None or len(df_eval) < 20:
            st.error("Not enough closed-candle data.")
            return

        levels = calculate_fibonacci_levels(df_eval, lookback=fib_lookback)
        if not levels:
            st.error("Could not calculate levels.")
            return

        divergences = detect_divergence(df_eval)
        vp = calculate_volume_profile(df_eval)
        regime = detect_market_regime(df_eval)

    current = float(df_eval["close"].iloc[-1])
    is_uptrend = bool(levels.get("_is_uptrend", True))
    regime_name = str(regime.get("regime", "UNKNOWN"))
    regime_desc = str(regime.get("description", ""))
    div_count = len(divergences or [])
    poc = float(vp.get("poc_price", current)) if vp else current
    poc_dist = ((current - poc) / current) * 100 if current > 0 else 0.0

    execution_levels = ["38.2%", "50%", "61.8%"]
    key_levels = execution_levels + ["100%", "161.8%"]
    nearest = []
    for lv in key_levels:
        p = float(levels.get(lv, 0) or 0)
        if p <= 0:
            continue
        d = abs((current - p) / current) * 100 if current > 0 else 0.0
        nearest.append((lv, p, d))
    nearest.sort(key=lambda x: x[2])
    nearest_exec = [row for row in nearest if row[0] in execution_levels]
    nearest_level = nearest_exec[0][0] if nearest_exec else "N/A"
    nearest_price = nearest_exec[0][1] if nearest_exec else current
    nearest_dist = nearest_exec[0][2] if nearest_exec else 0.0

    dist_pts = _distance_points(nearest_dist)
    regime_pts = _regime_points(regime_name)
    poc_pts = _poc_points(poc_dist)
    div_pen, div_conflict_count, div_supportive_count = _divergence_impact(divergences, is_uptrend=is_uptrend)
    setup_quality = max(0, min(100, dist_pts + regime_pts + poc_pts - div_pen))

    trend_text = "Bullish pullback context" if is_uptrend else "Bearish pullback context"
    if setup_quality >= 68:
        decision = "Setup quality: STRONG"
        decision_color = POSITIVE
        action_hint = "Action: execution-ready zone. Wait only for trigger confirmation."
    elif setup_quality >= 45:
        decision = "Setup quality: MODERATE"
        decision_color = WARNING
        action_hint = "Action: confirmation-first, smaller size, strict invalidation."
    else:
        decision = "Setup quality: WEAK"
        decision_color = NEGATIVE
        action_hint = "Action: stand aside until quality improves."
    decision_note = (
        f"Nearest {nearest_level} at {_fmt_price(nearest_price)} ({nearest_dist:.2f}% away) | "
        f"Regime: {regime_name} | POC distance: {poc_dist:+.2f}% | "
        f"Divergence alerts: {div_count} (conflict {div_conflict_count}, supportive {div_supportive_count})"
    )

    k1, k2, k3, k4 = st.columns(4)
    st.markdown(
        f"""
        <style>
        .fib-kpi-card.metric-card {{
            min-height: 168px;
            display: flex;
            flex-direction: column;
            justify-content: center;
            align-items: center;
            padding-top: 18px;
            padding-bottom: 18px;
        }}
        .fib-kpi-card .metric-sub {{
            font-size: 0.95rem;
            color: {TEXT_MUTED};
            margin-top: 6px;
            line-height: 1.2;
            min-height: 1.2em;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )
    with k1:
        st.markdown(
            f"<div class='metric-card fib-kpi-card'>"
            f"<div class='metric-label'>Setup Quality</div>"
            f"<div class='metric-value' style='color:{decision_color};'>{setup_quality:.0f}/100</div>"
            f"<div class='metric-sub'>&nbsp;</div>"
            f"</div>",
            unsafe_allow_html=True,
        )
    with k2:
        st.markdown(
            f"<div class='metric-card fib-kpi-card'>"
            f"<div class='metric-label'>Nearest Key Level</div>"
            f"<div class='metric-value' style='color:{ACCENT};'>{nearest_level}</div>"
            f"<div class='metric-sub'>{nearest_dist:.2f}% away</div>"
            f"</div>",
            unsafe_allow_html=True,
        )
    with k3:
        poc_col = POSITIVE if abs(poc_dist) <= 1 else (WARNING if abs(poc_dist) <= 3 else NEGATIVE)
        st.markdown(
            f"<div class='metric-card fib-kpi-card'>"
            f"<div class='metric-label'>POC Distance</div>"
            f"<div class='metric-value' style='color:{poc_col};'>{poc_dist:+.2f}%</div>"
            f"<div class='metric-sub'>&nbsp;</div>"
            f"</div>",
            unsafe_allow_html=True,
        )
    with k4:
        div_col = POSITIVE if div_count == 0 else (WARNING if div_count <= 2 else NEGATIVE)
        st.markdown(
            f"<div class='metric-card fib-kpi-card'>"
            f"<div class='metric-label'>Divergence Alerts</div>"
            f"<div class='metric-value' style='color:{div_col};'>{div_count}</div>"
            f"<div class='metric-sub'>&nbsp;</div>"
            f"</div>",
            unsafe_allow_html=True,
        )

    st.markdown(
        f"<div class='panel-box' style='border-left:4px solid {decision_color};'>"
        f"<b style='color:{decision_color};'>{decision}</b><br>"
        f"<span style='color:{TEXT_MUTED};'>{trend_text} | {decision_note}</span>"
        f"<br><span style='color:{decision_color}; font-size:0.84rem;'><b>{action_hint}</b></span>"
        f"<br><span style='color:{TEXT_MUTED}; font-size:0.80rem;'>"
        f"Score components: Distance {dist_pts} + Regime {regime_pts} + POC {poc_pts} − Net Divergence {div_pen}"
        f"<br>Bands: STRONG >= 68 | MODERATE 45-67 | WEAK < 45"
        f"</span>"
        f"</div>",
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
            {"Metric": "Trend Bias", "Value": ("▲ Bullish" if is_uptrend else "▼ Bearish"), "Status": ("● Supportive" if is_uptrend else "▼ Caution")},
            {"Metric": "Regime", "Value": f"{regime_icon} {regime_name}", "Status": ("● Trend-Friendly" if "TREND" in regime_name else ("■ Neutral" if regime_name in {"RANGING", "COMPRESSION"} else "▼ Risky"))},
            {"Metric": "POC Distance", "Value": f"{poc_dist:+.2f}%", "Status": poc_status},
            {"Metric": "Divergence Alerts", "Value": str(div_count), "Status": div_status},
        ]
    )
    st.markdown(
        f"<div style='color:{TEXT_MUTED}; font-size:0.83rem; margin:2px 0 8px 0; line-height:1.6;'>"
        f"<b style='color:{ACCENT};'>Metric Guide:</b> "
        f"{_tip('Trend Bias', 'Swing-direction context used for Fib mapping.')} | "
        f"{_tip('Regime', 'Market state. Trending regimes support continuation better than chop/compression.')} | "
        f"{_tip('POC Distance', 'Distance to highest-volume node. Near POC often acts as decision/magnet area.')} | "
        f"{_tip('Divergence Alerts', 'Direction-aware alerts: conflict signals reduce quality more than supportive ones.')}"
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
    st.caption("Execution focus: 38.2 / 50 / 61.8 first. 100 / 161.8 mostly extension map.")
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
        if lv in {"38.2%", "50%", "61.8%"}:
            role = "Pullback map" if is_uptrend else "Rebound map"
        elif lv == "100%":
            role = "Swing edge reference"
        else:
            role = "Extension target"
        exec_rows.append(
            {
                "Level": lv,
                "Price": _fmt_price(p),
                "Distance %": f"{dist:+.2f}%",
                "Zone": zone,
                "Role": role,
            }
        )
    exec_df = pd.DataFrame(exec_rows)

    def _zone_style(v: str) -> str:
        if "In Zone" in v:
            return f"color:{POSITIVE}; font-weight:700;"
        if "Near" in v:
            return f"color:{WARNING}; font-weight:700;"
        return f"color:{TEXT_MUTED};"

    role_help = {
        "Pullback map": "Primary retracement area in uptrend; look for reaction/hold for continuation.",
        "Rebound map": "Primary retracement area in downtrend; look for rejection/loss of level for continuation.",
        "Swing edge reference": "Key swing boundary (100%); often used as structure validation/invalidation reference.",
        "Extension target": "Beyond swing projection area; mainly used as extension target, not first entry zone.",
    }
    st.dataframe(
        exec_df.style.map(_zone_style, subset=["Zone"]),
        width="stretch",
        hide_index=True,
        column_config={
            "Role": st.column_config.TextColumn(
                "Role",
                help="Role meaning: Pullback/Rebound map = primary execution area, Swing edge reference = key structure boundary, Extension target = projection zone.",
            )
        },
    )
    st.markdown(
        f"<div style='color:{TEXT_MUTED}; font-size:0.83rem; margin:2px 0 8px 0; line-height:1.6;'>"
        f"<b style='color:{ACCENT};'>Role Guide:</b> "
        f"{_tip('Pullback map', role_help['Pullback map'])} | "
        f"{_tip('Rebound map', role_help['Rebound map'])} | "
        f"{_tip('Swing edge reference', role_help['Swing edge reference'])} | "
        f"{_tip('Extension target', role_help['Extension target'])}"
        f"</div>",
        unsafe_allow_html=True,
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
                    "Impact": (
                        "Supportive"
                        if (is_uptrend and "BULLISH" in d_type) or ((not is_uptrend) and "BEARISH" in d_type)
                        else ("Conflict" if (is_uptrend and "BEARISH" in d_type) or ((not is_uptrend) and "BULLISH" in d_type) else "Neutral")
                    ),
                    "Comment": str(d.get("description", "")),
                }
            )
        warn_df = pd.DataFrame(warn_rows)

        def _warn_signal_style(v: str) -> str:
            s = str(v).upper()
            if "BULLISH" in s:
                return f"color:{POSITIVE}; font-weight:700;"
            if "BEARISH" in s:
                return f"color:{NEGATIVE}; font-weight:700;"
            return f"color:{WARNING}; font-weight:700;"

        def _warn_strength_style(v: str) -> str:
            s = str(v).upper()
            if "STRONG" in s:
                return "font-weight:700;"
            if "MODERATE" in s:
                return "font-weight:700;"
            return f"color:{TEXT_MUTED};"

        def _warn_impact_style(v: str) -> str:
            s = str(v).upper()
            if "SUPPORTIVE" in s:
                return f"color:{POSITIVE}; font-weight:700;"
            if "CONFLICT" in s:
                return f"color:{NEGATIVE}; font-weight:700;"
            return f"color:{WARNING}; font-weight:700;"

        st.dataframe(
            warn_df.style.map(_warn_signal_style, subset=["Signal"])
            .map(_warn_strength_style, subset=["Strength"])
            .map(_warn_impact_style, subset=["Impact"]),
            width="stretch",
            hide_index=True,
        )
        st.caption("If warnings stack up, reduce size and require stronger confirmation.")
    else:
        st.success("No major divergence warning in selected lookback.")

    # Lean chart with key Fib levels + POC. Uses closed candles (same as analytics).
    fig = go.Figure()
    fig.add_trace(
        go.Candlestick(
            x=df_eval["timestamp"],
            open=df_eval["open"],
            high=df_eval["high"],
            low=df_eval["low"],
            close=df_eval["close"],
            increasing_line_color=POSITIVE,
            decreasing_line_color=NEGATIVE,
            name="Price (closed candles)",
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
            annotation_text=f"{lv}: {_fmt_price(p)}",
            annotation_font=dict(size=9, color=level_colors.get(lv, TEXT_MUTED)),
        )
    fig.add_hline(
        y=poc,
        line=dict(color=GOLD, dash="dash", width=1.4),
        annotation_text=f"POC: {_fmt_price(poc)}",
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
    st.caption("Chart and calculations both use closed candles for consistency.")
