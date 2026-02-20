from ui.ctx import get_ctx

import plotly.graph_objs as go


def render(ctx: dict) -> None:
    st = get_ctx(ctx, "st")
    ACCENT = get_ctx(ctx, "ACCENT")
    TEXT_MUTED = get_ctx(ctx, "TEXT_MUTED")
    TEXT_LIGHT = get_ctx(ctx, "TEXT_LIGHT")
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
    ml_ensemble_predict = get_ctx(ctx, "ml_ensemble_predict")
    """Enhanced ensemble ML prediction."""
    st.markdown(
        f"<h2 style='color:{ACCENT};'>Ensemble AI</h2>",
        unsafe_allow_html=True,
    )
    st.markdown(
        f"<div class='panel-box'>"
        f"<b style='color:{ACCENT}; font-size:1rem;'>What does this tab show?</b>"
        f"<p style='color:{TEXT_MUTED}; font-size:0.9rem; margin-top:6px; line-height:1.6;'>"
        f"Instead of a single model, trains three different {_tip('machine learning models', 'Methods where the computer learns patterns from historical data to make predictions about the future.')} "
        f"and produces a combined prediction via weighted voting. "
        f"More reliable than a single model because the models compensate for each other's errors.</p>"
        f"<p style='color:{TEXT_MUTED}; font-size:0.85rem; margin-top:6px; line-height:1.6;'>"
        f"<b>Models:</b> "
        f"{_tip('Gradient Boosting', 'A tree-based model that iteratively corrects errors. Highest weight (45%). Captures fine details well.')} (45% weight) | "
        f"{_tip('Random Forest', 'Averages the output of hundreds of decision trees. Resistant to overfitting. 35% weight.')} (35% weight) | "
        f"{_tip('Logistic Regression', 'The simplest model. Draws a linear boundary. Acts as a stabilizer. 20% weight.')} (20% weight)</p>"
        f"<p style='color:{TEXT_MUTED}; font-size:0.85rem; margin-top:6px;'>"
        f"{_tip('Model Agreement', 'How many of the three models predict the same direction (LONG or SHORT). 100% = all three agree. 33% = only one differs.')} "
        f"shows how much the models agree with each other.</p>"
        f"</div>",
        unsafe_allow_html=True,
    )

    col1, col2 = st.columns(2)
    with col1:
        ens_coin = _normalize_coin_input(st.text_input("Coin", value="BTC", key="ens_coin"))
    with col2:
        ens_tf = st.selectbox("Timeframe", ['5m', '15m', '1h', '4h', '1d'], index=2, key="ens_tf")

    if st.button("Run Ensemble Prediction", type="primary", key="ens_run"):
        _val_err = _validate_coin_symbol(ens_coin)
        if _val_err:
            st.error(_val_err)
            return

        with st.spinner("Training 3 ML models..."):
            df = fetch_ohlcv(ens_coin, ens_tf, limit=500)
            if df is None or len(df) < 60:
                st.error("Not enough data.")
                return
            prob, direction, details = ml_ensemble_predict(df)
            if not details:
                st.error("Ensemble prediction failed.")
                return

        dir_color = POSITIVE if direction == "LONG" else (NEGATIVE if direction == "SHORT" else WARNING)
        agreement_pct = details.get('agreement', 0) * 100
        agreement_color = POSITIVE if agreement_pct >= 66 else (WARNING if agreement_pct >= 33 else NEGATIVE)

        # Big direction display
        st.markdown(
            f"<div style='text-align:center; padding:24px; background:rgba(15,22,41,0.7); "
            f"border:2px solid {dir_color}; border-radius:16px; margin:16px 0;'>"
            f"<div style='color:{TEXT_MUTED}; font-size:0.8rem; letter-spacing:2px;'>ENSEMBLE PREDICTION</div>"
            f"<div style='color:{dir_color}; font-size:3rem; font-weight:800; margin:8px 0; "
            f"text-shadow:0 0 20px {dir_color};'>{direction}</div>"
            f"<div style='color:{ACCENT}; font-size:1.3rem;'>Probability: {prob*100:.1f}%</div>"
            f"<div style='color:{agreement_color}; font-size:0.9rem; margin-top:6px;'>"
            f"Model Agreement: {agreement_pct:.0f}%</div></div>",
            unsafe_allow_html=True,
        )

        # Individual models
        st.markdown(f"<div class='god-header'><b style='color:{NEON_BLUE};'>Individual Models</b></div>",
                    unsafe_allow_html=True)
        m1, m2, m3 = st.columns(3)
        models = [
            ("Gradient Boosting", details.get('gradient_boosting', 0.5), "45%", NEON_BLUE),
            ("Random Forest", details.get('random_forest', 0.5), "35%", NEON_PURPLE),
            ("Logistic Regression", details.get('logistic_regression', 0.5), "20%", WARNING),
        ]
        for col, (name, pv, weight, color) in zip([m1, m2, m3], models):
            with col:
                # Keep thresholds consistent with core ensemble direction mapping.
                md = "LONG" if pv >= 0.58 else ("SHORT" if pv <= 0.42 else "NEUTRAL")
                mc = POSITIVE if md == "LONG" else (NEGATIVE if md == "SHORT" else WARNING)
                st.markdown(
                    f"<div style='background:rgba(15,22,41,0.7); border:1px solid {color}; "
                    f"border-radius:12px; padding:16px; text-align:center;'>"
                    f"<div style='color:{color}; font-weight:700;'>{name}</div>"
                    f"<div style='color:{TEXT_MUTED}; font-size:0.7rem;'>Weight: {weight}</div>"
                    f"<div style='color:{mc}; font-size:1.8rem; font-weight:700; margin:8px 0;'>{md}</div>"
                    f"<div style='color:{ACCENT};'>{pv*100:.1f}%</div></div>",
                    unsafe_allow_html=True,
                )

        # Gauge chart
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=prob * 100,
            gauge={
                'axis': {'range': [0, 100]},
                'bar': {'color': dir_color},
                'bgcolor': PRIMARY_BG,
                'steps': [
                    {'range': [0, 20], 'color': 'rgba(255, 51, 102, 0.3)'},
                    {'range': [20, 40], 'color': 'rgba(255, 51, 102, 0.15)'},
                    {'range': [40, 60], 'color': 'rgba(255, 209, 102, 0.15)'},
                    {'range': [60, 80], 'color': 'rgba(0, 255, 136, 0.15)'},
                    {'range': [80, 100], 'color': 'rgba(0, 255, 136, 0.3)'},
                ],
            },
            title={'text': "Ensemble Bullish Probability", 'font': {'size': 16, 'color': ACCENT}},
            number={'font': {'color': TEXT_LIGHT, 'size': 40}, 'suffix': '%'},
        ))
        fig_gauge.update_layout(height=280, margin=dict(l=30, r=30, t=60, b=20),
                                 template='plotly_dark', paper_bgcolor=PRIMARY_BG)
        st.plotly_chart(fig_gauge, use_container_width=True)

