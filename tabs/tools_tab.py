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
    _tip = get_ctx(ctx, "_tip")
    """Simple beginner-friendly position planner."""

    st.markdown(f"<h2 style='color:{ACCENT};'>Trading Tools</h2>", unsafe_allow_html=True)
    st.markdown(
        f"<div class='panel-box'>"
        f"<b style='color:{ACCENT}; font-size:1rem;'>What does this tab show?</b>"
        f"<p style='color:{TEXT_MUTED}; font-size:0.9rem; margin-top:6px; line-height:1.6;'>"
        f"Simple position planner. Enter "
        f"{_tip('Entry', 'Price where you open the trade.')}, "
        f"{_tip('Stop Loss', 'Price where you exit on loss.')}, "
        f"{_tip('Take Profit', 'Price where you exit on profit.')}, "
        f"{_tip('Margin Used', 'Cash allocated to this position.')}, and "
        f"{_tip('Leverage', 'Position multiplier. Notional = margin x leverage.')}. "
        f"PnL includes optional {_tip('Funding Rate (%)', 'Positive funding: LONG pays SHORT. Negative funding: SHORT pays LONG.')}. "
        f"Trading fees are not included in this simplified model."
        f"</p></div>",
        unsafe_allow_html=True,
    )

    c1, c2, c3 = st.columns(3)
    with c1:
        entry = st.number_input("Entry Price ($)", min_value=0.0, value=100000.0, format="%.4f")
        stop = st.number_input("Stop Loss ($)", min_value=0.0, value=98000.0, format="%.4f")
        target = st.number_input("Take Profit ($)", min_value=0.0, value=105000.0, format="%.4f")
    with c2:
        margin = st.number_input("Margin Used ($)", min_value=0.0, value=1000.0, format="%.2f")
        direction = st.selectbox("Direction", ["LONG", "SHORT"])
        leverage = st.selectbox("Main Leverage", [1, 2, 3, 5, 10, 15, 20, 25, 50, 100], index=4)
    with c3:
        funding_rate = st.number_input(
            "Funding Rate (decimal)",
            min_value=-0.05000,
            max_value=0.05000,
            value=0.00000,
            step=0.00001,
            format="%.5f",
            help="Per funding period in decimal form (e.g. 0.00010 = 0.01%). Positive: LONG pays, SHORT receives.",
        )
        funding_periods = st.number_input(
            "Funding Period Count",
            min_value=0,
            max_value=24,
            value=1,
            step=1,
            help="How many funding events you expect while holding the position.",
        )
        scenario_lev = st.multiselect(
            "Compare Leverages",
            [1, 2, 3, 5, 10, 15, 20, 25, 50, 100],
            default=[1, 5, 10, 20],
            help="Optional comparison table.",
        )

    if st.button("Calculate", type="primary"):
        if entry <= 0 or stop <= 0 or target <= 0:
            st.error("Entry, Stop, and Target must be greater than 0.")
            return
        if margin <= 0:
            st.error("Margin must be greater than 0.")
            return
        if direction == "LONG" and not (stop < entry < target):
            st.error("For LONG, use: Stop < Entry < Target.")
            return
        if direction == "SHORT" and not (target < entry < stop):
            st.error("For SHORT, use: Target < Entry < Stop.")
            return

        notional = margin * leverage
        qty = notional / entry
        funding_cash = notional * funding_rate * int(funding_periods)
        dir_sign = 1 if direction == "LONG" else -1
        funding_effect = -dir_sign * funding_cash

        if direction == "LONG":
            pnl_tp = (target - entry) * qty + funding_effect
            pnl_sl = (stop - entry) * qty + funding_effect
            tp_move_pct = (target - entry) / entry * 100.0
            sl_move_pct = (stop - entry) / entry * 100.0
            if leverage > 1:
                liq_price = entry * (1.0 - 1.0 / leverage)
                liq_warn = liq_price >= stop
            else:
                liq_price = None
                liq_warn = False
        else:
            pnl_tp = (entry - target) * qty + funding_effect
            pnl_sl = (entry - stop) * qty + funding_effect
            tp_move_pct = (entry - target) / entry * 100.0
            sl_move_pct = (entry - stop) / entry * 100.0
            if leverage > 1:
                liq_price = entry * (1.0 + 1.0 / leverage)
                liq_warn = liq_price <= stop
            else:
                liq_price = None
                liq_warn = False

        liq_dist_pct = (abs((liq_price - entry) / entry) * 100.0) if liq_price is not None else None
        rr = abs(pnl_tp) / abs(pnl_sl) if pnl_sl != 0 else 0.0
        rr_color = POSITIVE if rr >= 1.5 else (WARNING if rr >= 1.0 else NEGATIVE)

        k1, k2, k3, k4 = st.columns(4)
        with k1:
            st.markdown(
                f"<div class='metric-card'><div class='metric-label'>Notional</div>"
                f"<div class='metric-value'>${notional:,.2f}</div></div>",
                unsafe_allow_html=True,
            )
        with k2:
            st.markdown(
                f"<div class='metric-card'><div class='metric-label'>Position Size (Coin)</div>"
                f"<div class='metric-value'>{qty:,.6f}</div></div>",
                unsafe_allow_html=True,
            )
        with k3:
            st.markdown(
                f"<div class='metric-card'><div class='metric-label'>R:R</div>"
                f"<div class='metric-value' style='color:{rr_color};'>1:{rr:.2f}</div></div>",
                unsafe_allow_html=True,
            )
        with k4:
            st.markdown(
                f"<div class='metric-card'><div class='metric-label'>Est. Liquidation</div>"
                f"<div class='metric-value'>{(f'${liq_price:,.2f}' if liq_price is not None else 'N/A')}</div>"
                f"<div style='color:{TEXT_MUTED};font-size:0.8rem;'>Dist: {(f'{liq_dist_pct:.2f}%' if liq_dist_pct is not None else 'N/A')}</div></div>",
                unsafe_allow_html=True,
            )

        st.markdown(
            f"<div class='panel-box' style='margin-bottom:0.8rem;'>"
            f"<b style='color:{ACCENT};'>Quick Read</b>"
            f"<div style='color:{TEXT_MUTED}; font-size:0.85rem; line-height:1.7; margin-top:6px;'>"
            f"Notional = Margin x Leverage. "
            f"PnL includes funding adjustment. "
            f"Liquidation is simplified: for x1 it is shown as N/A."
            f"</div></div>",
            unsafe_allow_html=True,
        )

        outcome_df = pd.DataFrame(
            [
                {"Scenario": "Take Profit", "Price": f"${target:,.2f}", "Move (%)": f"{tp_move_pct:+.2f}%", "PnL ($)": pnl_tp, "ROE (%)": (pnl_tp / margin) * 100.0},
                {"Scenario": "Stop Loss", "Price": f"${stop:,.2f}", "Move (%)": f"{sl_move_pct:+.2f}%", "PnL ($)": pnl_sl, "ROE (%)": (pnl_sl / margin) * 100.0},
                {"Scenario": "Funding Effect", "Price": "-", "Move (%)": "-", "PnL ($)": funding_effect, "ROE (%)": (funding_effect / margin) * 100.0},
            ]
        )
        st.dataframe(
            outcome_df.style.format({"PnL ($)": "{:+,.2f}", "ROE (%)": "{:+.2f}"}),
            width="stretch",
            hide_index=True,
        )

        if leverage == 1:
            st.info("x1 selected: liquidation is shown as N/A in this simplified model.")
        elif liq_warn:
            st.warning("Estimated liquidation is closer than your stop. Consider lower leverage or wider stop.")
        else:
            st.success("Estimated liquidation is beyond your stop.")

        if scenario_lev:
            rows = []
            for lev in sorted(set(int(x) for x in scenario_lev)):
                notional_l = margin * lev
                qty_l = notional_l / entry
                funding_l = notional_l * funding_rate * int(funding_periods)
                funding_effect_l = -dir_sign * funding_l
                if direction == "LONG":
                    tp_l = (target - entry) * qty_l + funding_effect_l
                    sl_l = (stop - entry) * qty_l + funding_effect_l
                    liq_l = (entry * (1.0 - 1.0 / lev)) if lev > 1 else None
                else:
                    tp_l = (entry - target) * qty_l + funding_effect_l
                    sl_l = (entry - stop) * qty_l + funding_effect_l
                    liq_l = (entry * (1.0 + 1.0 / lev)) if lev > 1 else None
                liq_dist_l = (abs((liq_l - entry) / entry) * 100.0) if liq_l is not None else None
                rows.append(
                    {
                        "Leverage": f"x{lev}",
                        "Notional ($)": round(notional_l, 2),
                        "TP PnL ($)": round(tp_l, 2),
                        "SL PnL ($)": round(sl_l, 2),
                        "Est. Liq Price ($)": (round(liq_l, 2) if liq_l is not None else "N/A"),
                        "Liq Dist (%)": (round(liq_dist_l, 2) if liq_dist_l is not None else "N/A"),
                    }
                )
            df_lev = pd.DataFrame(rows)
            st.markdown(f"<b style='color:{ACCENT};'>Leverage Comparison</b>", unsafe_allow_html=True)
            st.dataframe(
                df_lev.style.format(
                    {
                        "Notional ($)": "{:,.2f}",
                        "TP PnL ($)": "{:+,.2f}",
                        "SL PnL ($)": "{:+,.2f}",
                    }
                ),
                width="stretch",
                hide_index=True,
            )

            fig = go.Figure()
            fig.add_trace(go.Bar(x=df_lev["Leverage"], y=df_lev["TP PnL ($)"], name="TP PnL", marker_color=POSITIVE))
            fig.add_trace(go.Bar(x=df_lev["Leverage"], y=df_lev["SL PnL ($)"], name="SL PnL", marker_color=NEGATIVE))
            fig.update_layout(
                barmode="group",
                height=320,
                template="plotly_dark",
                margin=dict(l=20, r=20, t=30, b=20),
                yaxis_title="PnL ($)",
                xaxis_title="Leverage",
            )
            st.plotly_chart(fig, width="stretch")
