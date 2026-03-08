from ui.ctx import get_ctx

import html
import pandas as pd
import plotly.graph_objs as go
from core.market_decision import (
    ai_vote_metrics,
    action_decision_with_reason,
    normalize_action_class,
    structure_state,
)
from core.metric_catalog import AI_LONG_THRESHOLD, AI_SHORT_THRESHOLD, direction_from_prob
from core.signal_contract import strength_bucket, strength_from_bias
from ui.primitives import render_help_details, render_intro_card, render_kpi_grid, render_page_header
from ui.snapshot_cache import live_or_snapshot


def _dir_label(raw: str) -> str:
    key = str(raw or "").upper()
    if key == "LONG":
        return "Upside"
    if key == "SHORT":
        return "Downside"
    if key == "NEUTRAL":
        return "Neutral"
    return str(raw or "Neutral")


def _dir_color_key(raw: str) -> str:
    key = str(raw or "").upper()
    if key == "LONG":
        return "LONG"
    if key == "SHORT":
        return "SHORT"
    return "NEUTRAL"


def render(ctx: dict) -> None:
    st = get_ctx(ctx, "st")
    ACCENT = get_ctx(ctx, "ACCENT")
    TEXT_MUTED = get_ctx(ctx, "TEXT_MUTED")
    POSITIVE = get_ctx(ctx, "POSITIVE")
    NEGATIVE = get_ctx(ctx, "NEGATIVE")
    WARNING = get_ctx(ctx, "WARNING")
    CARD_BG = get_ctx(ctx, "CARD_BG")
    _tip = get_ctx(ctx, "_tip")
    _normalize_coin_input = get_ctx(ctx, "_normalize_coin_input")
    _validate_coin_symbol = get_ctx(ctx, "_validate_coin_symbol")
    fetch_ohlcv = get_ctx(ctx, "fetch_ohlcv")
    ml_ensemble_predict = get_ctx(ctx, "ml_ensemble_predict")
    get_scalping_entry_target = get_ctx(ctx, "get_scalping_entry_target")
    scalp_quality_gate = get_ctx(ctx, "scalp_quality_gate")
    _calc_conviction = get_ctx(ctx, "_calc_conviction")
    analyse = get_ctx(ctx, "analyse")
    get_major_ohlcv_bundle = get_ctx(ctx, "get_major_ohlcv_bundle")
    get_market_indices = get_ctx(ctx, "get_market_indices")
    _debug = get_ctx(ctx, "_debug")

    def _adx_bucket_only(adx_value: float) -> str:
        try:
            adx_f = float(adx_value)
        except Exception:
            return ""
        if not pd.notna(adx_f):
            return ""
        if adx_f < 20:
            return "Weak"
        if adx_f < 25:
            return "Starting"
        if adx_f < 50:
            return "Strong"
        if adx_f < 75:
            return "Very Strong"
        return "Extreme"

    def _stochrsi_bucket(v: float) -> str:
        try:
            x = float(v)
        except Exception:
            return ""
        if not pd.notna(x):
            return ""
        if x < 0.2:
            return "Low"
        if x > 0.8:
            return "High"
        return "Neutral"

    def _clean_indicator_text(v: str) -> str:
        txt = str(v or "").strip()
        for token in ["🟢", "🔴", "🟡", "⚪", "🔥", "▲▲", "▲", "▼", "→", "–"]:
            txt = txt.replace(token, "")
        return " ".join(txt.split()).strip()

    def _indicator_color(v: str) -> str:
        u = str(v or "").upper()
        if "VERY STRONG" in u or "EXTREME" in u:
            return POSITIVE
        if "STRONG" in u and "NOT" not in u:
            return POSITIVE
        if "WEAK" in u:
            return NEGATIVE
        if "STARTING" in u:
            return WARNING
        if any(k in u for k in ["BULLISH", "ABOVE", "OVERSOLD", "LOW", "NEAR BOTTOM", "UP SPIKE"]):
            return POSITIVE
        if any(k in u for k in ["BEARISH", "BELOW", "OVERBOUGHT", "HIGH", "NEAR TOP", "DOWN SPIKE"]):
            return NEGATIVE
        return WARNING

    def _indicator_cell(name: str, value: str, tooltip: str) -> str:
        val = _clean_indicator_text(value)
        if not val:
            return ""
        color = _indicator_color(val)
        tip = html.escape(str(tooltip or ""), quote=True)
        return (
            f"<div class='spot-indicator-item'>"
            f"<div class='spot-indicator-name'>{name}</div>"
            f"<div class='spot-indicator-value' style='color:{color};' title='{tip}'>{val}</div>"
            f"</div>"
        )

    def _to_side_key(raw: str) -> str:
        k = str(raw or "").strip().upper()
        if k in {"UPSIDE", "LONG", "BUY", "BULLISH", "STRONG BUY"}:
            return "UPSIDE"
        if k in {"DOWNSIDE", "SHORT", "SELL", "BEARISH", "STRONG SELL"}:
            return "DOWNSIDE"
        return "NEUTRAL"

    def _to_side_label(raw: str) -> str:
        k = _to_side_key(raw)
        if k == "UPSIDE":
            return "Upside"
        if k == "DOWNSIDE":
            return "Downside"
        return "Neutral"

    def _setup_confirm_display(raw_action: str) -> str:
        cls = normalize_action_class(raw_action)
        if cls == "ENTER_TREND_AI":
            return "TREND+AI"
        if cls == "ENTER_TREND_LED":
            return "TREND-led"
        if cls == "ENTER_AI_LED":
            return "AI-led"
        if cls == "WATCH":
            return "WATCH"
        if cls == "SKIP":
            return "SKIP"
        return str(raw_action or "").strip() or "SKIP"

    def _format_delta_pct(delta: float | None) -> str:
        if delta is None:
            return "—"
        if delta > 0:
            return f"▲ {abs(delta):.2f}%"
        if delta < 0:
            return f"▼ {abs(delta):.2f}%"
        return "→ 0.00%"

    render_page_header(
        st,
        title="AI Workspace",
        intro_html=(
            "AI-focused diagnostics workspace with two modes. "
            "<b>Quick Prediction</b> gives a one-shot ensemble read for a single coin and timeframe. "
            "<b>Model & Timeframe Matrix</b> compares agreement, probability, and plan context across selected frames."
        ),
    )
    mode = st.radio(
        "Mode",
        ["Quick Prediction", "Model & Timeframe Matrix"],
        horizontal=True,
        key="ai_workspace_mode",
    )

    if mode == "Quick Prediction":
        render_intro_card(
            st,
            title="Quick Prediction Mode",
            body_html=(
                "Fast one-shot ensemble read for a single coin/timeframe. "
                "Use this mode to see immediate AI direction, probability, and model agreement."
                f"<br><br>{_tip('Direction', 'Mapped to Upside / Downside / Neutral from ensemble probability thresholds.')} | "
                f"{_tip('Agreement', 'Effective agreement: directional agreement for Upside/Downside, consensus agreement for Neutral.')} | "
                f"{_tip('Signal Certainty', 'Quick confidence class from probability distance to neutral band.')}"
            ),
        )
        qc1, qc2 = st.columns(2)
        with qc1:
            quick_coin = _normalize_coin_input(st.text_input("Coin", value="BTC", key="ai_quick_coin"))
        with qc2:
            quick_tf = st.selectbox("Timeframe", ['5m', '15m', '1h', '4h', '1d'], index=2, key="ai_quick_tf")

        if not st.button("Run Quick Prediction", type="primary", key="ai_quick_run"):
            return

        _val_err = _validate_coin_symbol(quick_coin)
        if _val_err:
            st.error(_val_err)
            return

        with st.spinner("Running ensemble models..."):
            df = fetch_ohlcv(quick_coin, quick_tf, limit=500)
            if df is None or len(df) < 60:
                st.error("Not enough data.")
                return
            df_eval = df.iloc[:-1].copy() if len(df) > 60 else df.copy()
            if df_eval is None or len(df_eval) < 55:
                st.error("Not enough closed-candle data.")
                return
            prob, direction, details = ml_ensemble_predict(df_eval)
            if not details:
                st.error("Prediction unavailable.")
                return

        status = str(details.get("status", ""))
        err_detail = str(details.get("error", "")).strip()
        if status == "single_class_window":
            st.warning("Window has one-sided history. Showing neutral fallback output for safety.")
        elif status == "model_exception":
            st.warning("Model hit unstable inputs. Showing neutral fallback output for this run.")
            if err_detail:
                st.caption(f"Reason: {err_detail}")
        elif status == "insufficient_features":
            st.warning("Indicators produced insufficient clean rows. Showing neutral fallback output.")
            if err_detail:
                st.caption(f"Reason: {err_detail}")
        elif status == "insufficient_candles":
            st.warning("Not enough candles for reliable ML. Showing neutral fallback output.")
            if err_detail:
                st.caption(f"Reason: {err_detail}")

        direction_key = _dir_color_key(direction)
        direction_label = _dir_label(direction)
        dir_color = POSITIVE if direction_key == "LONG" else (NEGATIVE if direction_key == "SHORT" else WARNING)
        directional_agreement = float(details.get("directional_agreement", details.get("agreement", 0.0)))
        consensus_agreement = float(details.get("consensus_agreement", 0.0))
        agreement_ratio = directional_agreement if direction_key in {"LONG", "SHORT"} else consensus_agreement
        agreement_pct = agreement_ratio * 100.0
        agreement_color = POSITIVE if agreement_pct >= 66 else (WARNING if agreement_pct >= 33 else NEGATIVE)
        agreement_votes = max(0, min(3, int(round((agreement_pct / 100.0) * 3.0))))
        certainty = (
            "High"
            if prob >= 0.7 or prob <= 0.3
            else ("Medium" if prob >= 0.58 or prob <= 0.42 else "Low")
        )
        render_kpi_grid(
            st,
            items=[
                {"label": "Direction", "value": direction_label, "value_color": dir_color},
                {"label": "Probability", "value": f"{prob * 100:.1f}%"},
                {
                    "label": "Agreement (Effective)",
                    "value": f"{agreement_votes}/3",
                    "value_color": agreement_color,
                },
                {"label": "Signal Certainty", "value": certainty},
            ],
        )

        m1, m2, m3 = st.columns(3)
        models = [
            ("Gradient Boosting", float(details.get("gradient_boosting", 0.5)), "45%"),
            ("Random Forest", float(details.get("random_forest", 0.5)), "35%"),
            ("Logistic Regression", float(details.get("logistic_regression", 0.5)), "20%"),
        ]
        for col, (name, pv, weight) in zip([m1, m2, m3], models):
            with col:
                mdl_key = _dir_color_key(direction_from_prob(float(pv)))
                mdl_label = _dir_label(mdl_key)
                mdl_color = POSITIVE if mdl_key == "LONG" else (NEGATIVE if mdl_key == "SHORT" else WARNING)
                render_kpi_grid(
                    st,
                    columns=1,
                    items=[
                        {
                            "label": f"{name} ({weight})",
                            "value": mdl_label,
                            "value_color": mdl_color,
                            "subtext": f"{pv * 100:.1f}%",
                        }
                    ],
                )
        return

    render_intro_card(
        st,
        title="Model & Timeframe Matrix",
        body_html=(
            "Multi-timeframe model diagnostics for one coin. Compare direction consistency, probability, "
            "agreement quality, and plan source across selected frames."
            f"<br><br>{_tip('Selected Model Prob', 'Upward probability from selected model. Higher implies Upside bias; lower implies Downside bias.')} | "
            f"{_tip('Agreement', 'Effective agreement: directional agreement for Upside/Downside, consensus agreement for Neutral.')} | "
            f"{_tip('AI Direction Bias', 'Market-wide AI bias: dominance-weighted ensemble probability over major coins for same timeframe.')}"
        ),
    )

    c1, c2, c3 = st.columns(3)
    with c1:
        coin = _normalize_coin_input(st.text_input("Coin (e.g. BTC, ETH, TAO)", value="BTC", key="ai_coin_input"))
    with c2:
        selected_model = st.selectbox(
            "Model",
            ["Ensemble", "Gradient Boosting", "Random Forest", "Logistic Regression"],
            index=0,
            key="ai_model_selector",
        )
    with c3:
        selected_timeframes = st.multiselect(
            "Select up to 3 Timeframes",
            ["1m", "3m", "5m", "15m", "1h", "4h", "1d"],
            default=["1h"],
            max_selections=3,
            key="ai_tfs",
        )

    render_help_details(
        st,
        summary="How to read quickly",
        body_html=(
            "<b>1.</b> Prefer setups where direction is consistent across timeframes.<br>"
            "<b>2.</b> If Upward Probability is near the middle band and Agreement is low, treat as fragile signal.<br>"
            "<b>3.</b> Use plan levels only as drafts; always validate in Spot/Position tabs."
        ),
    )
    st.markdown(
        f"<div style='margin:-2px 0 10px 0; color:{TEXT_MUTED}; font-size:0.84rem; line-height:1.6;'>"
        f"{_tip('Plan Entry / Plan Target', 'Primary plan shown in table. If AI and technical direction align, AI-filtered plan is used; otherwise technical fallback is shown.')}"
        f"</div>",
        unsafe_allow_html=True,
    )

    if not st.button("Predict", type="primary"):
        return

    _val_err = _validate_coin_symbol(coin)
    if _val_err:
        st.error(_val_err)
        return
    if not selected_timeframes:
        st.error("Select at least one timeframe.")
        return

    def _model_prob_direction(df_model: pd.DataFrame, model_name: str) -> tuple[float, str, dict, str]:
        prob_e, dir_e, details = ml_ensemble_predict(df_model)
        if model_name == "Ensemble":
            return float(prob_e), str(dir_e), details or {}, str(dir_e)
        key_map = {
            "Gradient Boosting": "gradient_boosting",
            "Random Forest": "random_forest",
            "Logistic Regression": "logistic_regression",
        }
        key = key_map.get(model_name)
        p = float((details or {}).get(key, prob_e))
        # Keep decision thresholds consistent with core ensemble mapping.
        d = direction_from_prob(float(p))
        return p, d, details or {}, str(dir_e)

    market_outlook_cache: dict[str, tuple[float, str]] = {}
    market_indices_cache = None

    def _market_outlook_for_tf(tf: str) -> tuple[float, str]:
        nonlocal market_indices_cache
        cached = market_outlook_cache.get(tf)
        if cached is not None:
            return cached
        try:
            bundle = get_major_ohlcv_bundle(tf, limit=500)
            probs = {}
            for sym in ["BTC/USDT", "ETH/USDT", "BNB/USDT", "SOL/USDT", "ADA/USDT", "XRP/USDT"]:
                dfx = bundle.get(sym)
                if dfx is not None and not dfx.empty and len(dfx) >= 60:
                    dfx_eval = dfx.iloc[:-1].copy() if len(dfx) > 60 else dfx.copy()
                    probs[sym], _, _ = ml_ensemble_predict(dfx_eval)
                else:
                    probs[sym] = 0.5
            if market_indices_cache is None:
                market_indices_cache = get_market_indices()
            btc_dom, eth_dom, _, _, _, bnb_dom, sol_dom, ada_dom, xrp_dom = market_indices_cache
            weights = {
                "BTC/USDT": float(btc_dom),
                "ETH/USDT": float(eth_dom),
                "BNB/USDT": float(bnb_dom),
                "SOL/USDT": float(sol_dom),
                "ADA/USDT": float(ada_dom),
                "XRP/USDT": float(xrp_dom),
            }
            wsum = sum(max(v, 0.0) for v in weights.values())
            if wsum <= 0:
                market_outlook_cache[tf] = (0.5, "NEUTRAL")
                return market_outlook_cache[tf]
            p = sum(probs[s] * max(weights[s], 0.0) for s in weights) / wsum
            d = direction_from_prob(float(p))
            market_outlook_cache[tf] = (float(p), d)
            return market_outlook_cache[tf]
        except Exception as exc:
            _debug(f"AI Workspace market outlook error ({tf}): {exc}")
            market_outlook_cache[tf] = (0.5, "NEUTRAL")
            return market_outlook_cache[tf]

    rows = []
    tf_eval_cache: dict[str, pd.DataFrame | None] = {}
    for tf in selected_timeframes:
        with st.spinner(f"Running {selected_model} for {tf}..."):
            df = fetch_ohlcv(coin, tf, limit=500)
            if df is None or len(df) < 60:
                tf_eval_cache[tf] = None
                rows.append(
                    {
                        "Timeframe": tf,
                        "Direction": "NO DATA",
                        "DirectionKey": "NO_DATA",
                        "Selected Model Prob %": 0.0,
                        "Ensemble Agree": "N/A",
                        "Ensemble Agree %": 0.0,
                        "AI Direction Bias %": 50.0,
                        "AI Direction Bias": "Neutral (50.0%)",
                        "AI Direction Bias Key": "NEUTRAL",
                        "Plan Entry": "N/A",
                        "Plan Target": "N/A",
                        "Plan Source": "N/A",
                        "AI Entry": "N/A",
                        "AI Target": "N/A",
                        "Non-AI Entry": "N/A",
                        "Non-AI Target": "N/A",
                    }
                )
                continue

            df_eval = df.iloc[:-1].copy() if len(df) > 60 else df.copy()
            if df_eval is None or len(df_eval) < 55:
                tf_eval_cache[tf] = None
                rows.append(
                    {
                        "Timeframe": tf,
                        "Direction": "NO DATA",
                        "DirectionKey": "NO_DATA",
                        "Selected Model Prob %": 0.0,
                        "Ensemble Agree": "N/A",
                        "Ensemble Agree %": 0.0,
                        "AI Direction Bias %": 50.0,
                        "AI Direction Bias": "Neutral (50.0%)",
                        "AI Direction Bias Key": "NEUTRAL",
                        "Plan Entry": "N/A",
                        "Plan Target": "N/A",
                        "Plan Source": "N/A",
                        "AI Entry": "N/A",
                        "AI Target": "N/A",
                        "Non-AI Entry": "N/A",
                        "Non-AI Target": "N/A",
                    }
                )
                continue

            tf_eval_cache[tf] = df_eval
            prob, direction, details, ensemble_direction = _model_prob_direction(df_eval, selected_model)
            direction_key = _dir_color_key(direction)
            direction_label = _dir_label(direction)
            directional_agreement = float(
                (details or {}).get("directional_agreement", (details or {}).get("agreement", 0.0))
            )
            consensus_agreement = float((details or {}).get("consensus_agreement", 0.0))
            agreement_ratio = directional_agreement if direction_key in {"LONG", "SHORT"} else consensus_agreement
            agreement_votes = max(0, min(3, int(round(agreement_ratio * 3.0))))
            agreement_pct = agreement_ratio * 100.0
            ensemble_side = _to_side_key(ensemble_direction)
            ensemble_dir_agree = float(
                (details or {}).get("directional_agreement", (details or {}).get("agreement", 0.0))
            )
            ensemble_cons_agree = float((details or {}).get("consensus_agreement", 0.0))
            ai_votes_for_decision, _unused_ratio, decision_agreement_for_decision = ai_vote_metrics(
                ensemble_side,
                ensemble_dir_agree,
                ensemble_cons_agree,
            )
            mkt_prob, mkt_dir = _market_outlook_for_tf(tf)

            ai_entry_txt = "N/A"
            ai_target_txt = "N/A"
            ta_entry_txt = "N/A"
            ta_target_txt = "N/A"
            try:
                ar = analyse(df_eval)
                scalp_dir, entry_s, target_s, stop_s, rr_ratio, _ = get_scalping_entry_target(
                    df_eval,
                    ar.bias,
                    ar.supertrend,
                    ar.ichimoku,
                    ar.vwap,
                )
                signal_dir = (
                    "LONG" if ar.signal in {"STRONG BUY", "BUY"}
                    else ("SHORT" if ar.signal in {"STRONG SELL", "SELL"} else "NEUTRAL")
                )
                conviction_lbl, _ = _calc_conviction(
                    signal_dir,
                    direction_key,
                    strength_from_bias(float(ar.bias)),
                    directional_agreement,
                )
                scalp_ok, _scalp_reason = scalp_quality_gate(
                    scalp_direction=scalp_dir,
                    signal_direction=signal_dir,
                    rr_ratio=rr_ratio,
                    adx_val=ar.adx,
                    strength=strength_from_bias(float(ar.bias)),
                    conviction_label=conviction_lbl,
                    entry=entry_s,
                    stop=stop_s,
                    target=target_s,
                )
                if scalp_ok and scalp_dir in {"LONG", "SHORT"} and entry_s and target_s:
                    ta_entry_txt = f"${entry_s:,.4f}"
                    ta_target_txt = f"${target_s:,.4f}"
                if scalp_ok and scalp_dir == direction_key and direction_key in {"LONG", "SHORT"} and entry_s and target_s:
                    ai_entry_txt = f"${entry_s:,.4f}"
                    ai_target_txt = f"${target_s:,.4f}"
            except Exception as exc:
                _debug(f"AI Workspace planning block error ({tf}): {exc}")

            rows.append(
                {
                    "Timeframe": tf,
                    "Direction": direction_label,
                    "DirectionKey": direction_key,
                    "Selected Model Prob %": round(prob * 100.0, 1),
                    "Ensemble Agree": f"{agreement_votes}/3",
                    "Ensemble Agree %": round(agreement_pct, 1),
                    "AI Direction Bias %": round(mkt_prob * 100.0, 1),
                    "AI Direction Bias": f"{_dir_label(mkt_dir)} ({mkt_prob * 100.0:.1f}%)",
                    "AI Direction Bias Key": _dir_color_key(mkt_dir),
                    "Plan Entry": ai_entry_txt if ai_entry_txt != "N/A" else ta_entry_txt,
                    "Plan Target": ai_target_txt if ai_target_txt != "N/A" else ta_target_txt,
                    "Plan Source": ("AI-Filtered" if ai_entry_txt != "N/A" else ("Technical Fallback" if ta_entry_txt != "N/A" else "No Plan")),
                    "AI Entry": ai_entry_txt,
                    "AI Target": ai_target_txt,
                    "Non-AI Entry": ta_entry_txt,
                    "Non-AI Target": ta_target_txt,
                    "__ensemble_side": ensemble_side,
                    "__ai_votes": int(ai_votes_for_decision),
                    "__decision_agreement": round(float(decision_agreement_for_decision), 4),
                }
            )

    live_valid = [r for r in rows if r.get("Direction") != "NO DATA"]
    snapshot_key = f"ailab_rows::{coin}::{selected_model}::{','.join(sorted(selected_timeframes))}"
    snapshot_sig = (coin, selected_model, tuple(sorted(selected_timeframes)))
    if len(live_valid) == 0:
        rows, from_cache, cache_ts = live_or_snapshot(
            st, snapshot_key, rows, max_age_sec=900, current_sig=snapshot_sig
        )
        if from_cache:
            st.warning(f"Live AI Workspace data unavailable. Showing cached snapshot from {cache_ts}.")
    else:
        live_or_snapshot(st, snapshot_key, rows, max_age_sec=900, current_sig=snapshot_sig)

    df_out = pd.DataFrame(rows)
    df_valid = df_out[df_out["DirectionKey"].isin(["LONG", "SHORT", "NEUTRAL"])].copy()
    dominant = "NEUTRAL"
    if not df_valid.empty:
        dominant = str(df_valid["DirectionKey"].mode().iloc[0])
    avg_prob = float(df_valid["Selected Model Prob %"].mean()) if not df_valid.empty else 0.0
    avg_agree = float(df_valid["Ensemble Agree %"].mean()) if not df_valid.empty else 0.0
    consistency = (
        (float((df_valid["DirectionKey"] == dominant).sum()) / float(len(df_valid))) * 100.0
        if not df_valid.empty
        else 0.0
    )
    tf_valid_count = int(df_valid["Timeframe"].nunique()) if not df_valid.empty else 0

    dominant_label = _dir_label(dominant)
    dom_color = POSITIVE if dominant == "LONG" else (NEGATIVE if dominant == "SHORT" else WARNING)
    dir_kpi_label = "Dominant Direction" if tf_valid_count > 1 else "Selected TF Direction"
    prob_kpi_label = "Avg Upward Prob" if tf_valid_count > 1 else "Upward Probability"
    agree_kpi_label = "Avg Ensemble Agree" if tf_valid_count > 1 else "Ensemble Agree"
    consistency_label = "TF Consistency (same-direction share)"
    consistency_value = f"{consistency:.0f}%" if tf_valid_count > 1 else "Single TF"
    consistency_color = ACCENT if tf_valid_count > 1 else TEXT_MUTED
    render_kpi_grid(
        st,
        items=[
            {
                "label": dir_kpi_label,
                "value": dominant_label,
                "value_color": dom_color,
            },
            {"label": prob_kpi_label, "value": f"{avg_prob:.1f}%"},
            {"label": agree_kpi_label, "value": f"{avg_agree:.1f}%"},
            {
                "label": consistency_label,
                "value": consistency_value,
                "value_color": consistency_color,
            },
        ],
    )

    def _dir_style(v: str) -> str:
        s = str(v).lower()
        if "upside" in s:
            return f"color:{POSITIVE}; font-weight:700;"
        if "downside" in s:
            return f"color:{NEGATIVE}; font-weight:700;"
        if "neutral" in s:
            return f"color:{WARNING}; font-weight:700;"
        return f"color:{TEXT_MUTED};"

    def _prob_style(v: float) -> str:
        try:
            x = float(v)
        except Exception:
            return f"color:{TEXT_MUTED};"
        if x >= AI_LONG_THRESHOLD * 100.0:
            return f"color:{POSITIVE}; font-weight:700;"
        if x <= AI_SHORT_THRESHOLD * 100.0:
            return f"color:{NEGATIVE}; font-weight:700;"
        return f"color:{WARNING}; font-weight:700;"

    def _agree_style(v: str) -> str:
        try:
            x = int(str(v).split("/")[0])
        except Exception:
            return f"color:{TEXT_MUTED};"
        if x >= 3:
            return f"color:{POSITIVE}; font-weight:700;"
        if x >= 2:
            return f"color:{WARNING}; font-weight:700;"
        if x >= 1:
            return f"color:{NEGATIVE}; font-weight:700;"
        return f"color:{TEXT_MUTED};"

    def _plan_source_style(v: str) -> str:
        s = str(v)
        if "AI-Filtered" in s:
            return f"color:{POSITIVE}; font-weight:700;"
        if "Technical Fallback" in s:
            return f"color:{WARNING}; font-weight:700;"
        return f"color:{TEXT_MUTED};"

    st.markdown(f"<b style='color:{ACCENT};'>AI Workspace Timeframe Matrix</b>", unsafe_allow_html=True)
    render_help_details(
        st,
        summary="Column Guide",
        body_html=(
            "<b>Direction</b>: selected model direction for this timeframe (Upside / Downside / Neutral).<br>"
            "<b>Selected Model Prob %</b>: upward probability from selected model; lower values imply Downside bias.<br>"
            "<b>Ensemble Agree</b>: effective ensemble agreement (x/3): directional for Upside/Downside, consensus for Neutral.<br>"
            "<b>AI Direction Bias</b>: same market-wide bias logic as Market tab (dominance-weighted Ensemble), shown as direction + percent.<br>"
            "<b>Plan Entry/Plan Target</b>: primary plan to watch.<br>"
            "<b>Plan Source</b>: AI-Filtered or Technical Fallback."
        ),
    )
    st.dataframe(
        df_out[
            [
                "Timeframe",
                "Direction",
                "Selected Model Prob %",
                "Ensemble Agree",
                "AI Direction Bias",
                "Plan Entry",
                "Plan Target",
                "Plan Source",
            ]
        ].style.format({"Selected Model Prob %": "{:.1f}%"})
        .map(_dir_style, subset=["Direction"])
        .map(_prob_style, subset=["Selected Model Prob %"])
        .map(_dir_style, subset=["AI Direction Bias"])
        .map(_agree_style, subset=["Ensemble Agree"])
        .map(_plan_source_style, subset=["Plan Source"]),
        width="stretch",
        hide_index=True,
    )

    with st.expander("Show Plan Debug Columns (AI vs Technical)"):
        st.dataframe(
            df_out[
                [
                    "Timeframe",
                    "AI Entry",
                    "AI Target",
                    "Non-AI Entry",
                    "Non-AI Target",
                ]
            ],
            width="stretch",
            hide_index=True,
        )

    # Probability by timeframe chart.
    df_plot = df_out[df_out["DirectionKey"].isin(["LONG", "SHORT", "NEUTRAL"])].copy()
    if not df_plot.empty:
        fig = go.Figure()
        fig.add_trace(
            go.Bar(
                x=df_plot["Timeframe"],
                y=df_plot["Selected Model Prob %"],
                marker_color=[
                    POSITIVE if d == "LONG" else (NEGATIVE if d == "SHORT" else WARNING)
                    for d in df_plot["DirectionKey"]
                ],
                name="Upward Prob %",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=df_plot["Timeframe"],
                y=df_plot["AI Direction Bias %"],
                mode="lines+markers",
                line=dict(color=ACCENT, width=2),
                name="AI Direction Bias %",
            )
        )
        fig.add_hline(
            y=AI_LONG_THRESHOLD * 100.0,
            line=dict(color=POSITIVE, dash="dot", width=1),
            annotation_text="Upside threshold",
        )
        fig.add_hline(
            y=AI_SHORT_THRESHOLD * 100.0,
            line=dict(color=NEGATIVE, dash="dot", width=1),
            annotation_text="Downside threshold",
        )
        fig.update_layout(
            height=330,
            template="plotly_dark",
            margin=dict(l=20, r=20, t=35, b=20),
            yaxis_title="Upward Probability (%)",
            paper_bgcolor=CARD_BG,
        )
        st.plotly_chart(fig, width="stretch")

    # Detailed indicator panel per timeframe (optional, collapse by default).
    tf_row_map = {str(r.get("Timeframe")): r for r in rows if r.get("Direction") != "NO DATA"}
    with st.expander("Show Technical Indicator Panels"):
        st.markdown(
            f"<style>"
            f".aiw-snap-title{{"
            f"  margin:0.35rem 0 0.25rem 0;"
            f"  text-align:center;"
            f"  color:{TEXT_MUTED};"
            f"  font-size:0.78rem;"
            f"  text-transform:uppercase;"
            f"  letter-spacing:0.45px;"
            f"}}"
            f".aiw-snap-wrap{{"
            f"  background:linear-gradient(140deg, rgba(4, 10, 18, 0.95), rgba(2, 5, 11, 0.95));"
            f"  border:1px solid rgba(0, 212, 255, 0.16);"
            f"  border-radius:12px;"
            f"  padding:10px 12px;"
            f"  margin:0.08rem 0 0.45rem 0;"
            f"}}"
            f".aiw-snap-grid{{"
            f"  display:flex;"
            f"  flex-wrap:wrap;"
            f"  justify-content:center;"
            f"  gap:0.45rem 0.75rem;"
            f"}}"
            f".aiw-snap-item{{"
            f"  flex:0 1 150px;"
            f"  min-width:130px;"
            f"  text-align:center;"
            f"  padding:4px 6px;"
            f"}}"
            f".aiw-snap-label{{"
            f"  color:{TEXT_MUTED};"
            f"  font-size:0.67rem;"
            f"  text-transform:uppercase;"
            f"  letter-spacing:0.55px;"
            f"}}"
            f".aiw-snap-value{{"
            f"  font-size:1.02rem;"
            f"  font-weight:700;"
            f"  margin-top:3px;"
            f"}}"
            f".spot-indicator-sep{{"
            f"  margin:8px 0 6px 0;"
            f"  text-align:center;"
            f"  color:{TEXT_MUTED};"
            f"  font-size:0.80rem;"
            f"  text-transform:uppercase;"
            f"  letter-spacing:0.45px;"
            f"}}"
            f".spot-indicator-wrap{{"
            f"  display:grid;"
            f"  grid-template-columns:repeat(auto-fit,minmax(250px,1fr));"
            f"  gap:0.55rem;"
            f"  margin:0.2rem 0 0.45rem 0;"
            f"}}"
            f".spot-indicator-group{{"
            f"  background:rgba(0,0,0,0.56);"
            f"  border:1px solid rgba(0, 212, 255, 0.14);"
            f"  border-radius:12px;"
            f"  padding:8px 8px 6px 8px;"
            f"}}"
            f".spot-indicator-group-title{{"
            f"  color:{ACCENT};"
            f"  text-align:center;"
            f"  font-size:0.74rem;"
            f"  text-transform:uppercase;"
            f"  letter-spacing:0.55px;"
            f"  margin-bottom:4px;"
            f"}}"
            f".spot-indicator-grid{{"
            f"  display:flex;"
            f"  flex-wrap:wrap;"
            f"  justify-content:center;"
            f"  gap:4px 10px;"
            f"  align-items:center;"
            f"}}"
            f".spot-indicator-item{{"
            f"  text-align:center;"
            f"  padding:4px 2px;"
            f"  flex:0 1 118px;"
            f"}}"
            f".spot-indicator-name{{"
            f"  color:{TEXT_MUTED};"
            f"  font-size:0.68rem;"
            f"  text-transform:uppercase;"
            f"  letter-spacing:0.5px;"
            f"}}"
            f".spot-indicator-value{{"
            f"  font-size:0.95rem;"
            f"  font-weight:700;"
            f"  margin-top:2px;"
            f"}}"
            f"</style>",
            unsafe_allow_html=True,
        )
        for tf in selected_timeframes:
            df_eval = tf_eval_cache.get(tf)
            if df_eval is None or len(df_eval) < 55:
                st.caption(f"{tf}: no sufficient data.")
                continue
            try:
                ar = analyse(df_eval)
                price_change = None
                if len(df_eval) >= 2:
                    p0 = float(df_eval["close"].iloc[-2])
                    p1 = float(df_eval["close"].iloc[-1])
                    if p0 > 0:
                        price_change = ((p1 / p0) - 1.0) * 100.0
                delta_display = _format_delta_pct(price_change)
                delta_color = (
                    POSITIVE if str(delta_display).strip().startswith("▲")
                    else (NEGATIVE if str(delta_display).strip().startswith("▼") else WARNING)
                )

                signal_side = _to_side_key(ar.signal)
                signal_side_for_conviction = signal_side if signal_side in {"UPSIDE", "DOWNSIDE"} else "WAIT"
                signal_side_label = _to_side_label(signal_side)
                signal_color = (
                    POSITIVE if signal_side == "UPSIDE"
                    else (NEGATIVE if signal_side == "DOWNSIDE" else WARNING)
                )

                row_meta = tf_row_map.get(str(tf), {})
                ensemble_side = str(row_meta.get("__ensemble_side", "")).upper()
                if ensemble_side not in {"UPSIDE", "DOWNSIDE", "NEUTRAL"}:
                    ensemble_side = _to_side_key(str(row_meta.get("Direction", "Neutral")))
                ensemble_label = _to_side_label(ensemble_side)
                ensemble_color = (
                    POSITIVE if ensemble_side == "UPSIDE"
                    else (NEGATIVE if ensemble_side == "DOWNSIDE" else WARNING)
                )
                try:
                    ai_votes = max(0, min(3, int(row_meta.get("__ai_votes", 0))))
                except Exception:
                    ai_votes = 0
                try:
                    decision_agreement = float(row_meta.get("__decision_agreement", 0.0))
                except Exception:
                    decision_agreement = 0.0
                if decision_agreement <= 0.0:
                    try:
                        decision_agreement = float(row_meta.get("Ensemble Agree %", 0.0)) / 100.0
                    except Exception:
                        decision_agreement = 0.0
                if ai_votes == 0 and "/" in str(row_meta.get("Ensemble Agree", "")):
                    try:
                        ai_votes = max(0, min(3, int(str(row_meta.get("Ensemble Agree")).split("/")[0])))
                    except Exception:
                        ai_votes = 0

                strength_score = strength_from_bias(float(ar.bias))
                strength_lbl = strength_bucket(strength_score)
                strength_display = f"{strength_score:.0f}% ({strength_lbl})"
                strength_color = (
                    POSITIVE if strength_lbl in {"STRONG", "GOOD"}
                    else (WARNING if strength_lbl == "MIXED" else NEGATIVE)
                )
                conviction_lbl, _ = _calc_conviction(
                    signal_side_for_conviction,
                    ensemble_side,
                    strength_score,
                    decision_agreement,
                )
                align_color = (
                    POSITIVE if conviction_lbl == "HIGH"
                    else (WARNING if conviction_lbl in {"MEDIUM", "TREND"} else NEGATIVE)
                )

                structure_val = structure_state(
                    signal_side_for_conviction,
                    ensemble_side,
                    strength_score,
                    decision_agreement,
                )
                action_raw, _reason_code = action_decision_with_reason(
                    signal_side_for_conviction,
                    strength_score,
                    structure_val,
                    str(conviction_lbl),
                    decision_agreement,
                    float(ar.adx) if pd.notna(ar.adx) else float("nan"),
                )
                setup_confirm = _setup_confirm_display(action_raw)
                action_cls = normalize_action_class(action_raw)
                setup_color = (
                    POSITIVE if action_cls.startswith("ENTER_")
                    else (WARNING if action_cls == "WATCH" else NEGATIVE)
                )

                st.markdown(
                    f"<div class='aiw-snap-title'>Setup Snapshot ({tf})</div>"
                    f"<div class='aiw-snap-wrap'><div class='aiw-snap-grid'>"
                    f"<div class='aiw-snap-item' title='Closed-candle move on selected timeframe.'>"
                    f"<div class='aiw-snap-label'>Δ (%)</div>"
                    f"<div class='aiw-snap-value' style='color:{delta_color};'>{delta_display}</div></div>"
                    f"<div class='aiw-snap-item' title='Execution class from market decision policy.'>"
                    f"<div class='aiw-snap-label'>Setup Confirm</div>"
                    f"<div class='aiw-snap-value' style='color:{setup_color};'>{setup_confirm}</div></div>"
                    f"<div class='aiw-snap-item' title='Technical side from closed-candle signal stack.'>"
                    f"<div class='aiw-snap-label'>Direction</div>"
                    f"<div class='aiw-snap-value' style='color:{signal_color};'>{signal_side_label}</div></div>"
                    f"<div class='aiw-snap-item' title='Direction-agnostic technical signal power (0-100).'>"
                    f"<div class='aiw-snap-label'>Strength</div>"
                    f"<div class='aiw-snap-value' style='color:{strength_color};'>{strength_display}</div></div>"
                    f"<div class='aiw-snap-item' title='Ensemble direction and effective agreement votes.'>"
                    f"<div class='aiw-snap-label'>AI Ensemble</div>"
                    f"<div class='aiw-snap-value' style='color:{ensemble_color};'>{ensemble_label} ({ai_votes}/3)</div></div>"
                    f"<div class='aiw-snap-item' title='How well technical direction and AI ensemble align.'>"
                    f"<div class='aiw-snap-label'>Tech vs AI Alignment</div>"
                    f"<div class='aiw-snap-value' style='color:{align_color};'>{conviction_lbl}</div></div>"
                    f"</div></div>",
                    unsafe_allow_html=True,
                )

                ichi_meta_parts = []
                if ar.ichimoku_tk_cross:
                    ichi_meta_parts.append(
                        f"TK Cross: {ar.ichimoku_tk_cross.replace('▲ ', '').replace('▼ ', '').replace('→ ', '')}"
                    )
                if ar.ichimoku_future_bias:
                    ichi_meta_parts.append(
                        f"Future Cloud: {ar.ichimoku_future_bias.replace('▲ ', '').replace('▼ ', '').replace('→ ', '')}"
                    )
                if ar.ichimoku_cloud_strength:
                    ichi_meta_parts.append(
                        f"Cloud Strength: {ar.ichimoku_cloud_strength.replace('▲ ', '').replace('▼ ', '').replace('→ ', '')}"
                    )
                ichimoku_hover = " | ".join(ichi_meta_parts)

                spike_label = ""
                spike_hover = ""
                if ar.volume_spike:
                    try:
                        prev_vol_avg = float(df_eval["volume"].iloc[-21:-1].mean()) if len(df_eval) >= 21 else float("nan")
                        last_vol = float(df_eval["volume"].iloc[-1]) if len(df_eval) >= 1 else float("nan")
                        vol_ratio = (
                            last_vol / prev_vol_avg
                            if pd.notna(prev_vol_avg) and prev_vol_avg > 0 and pd.notna(last_vol)
                            else float("nan")
                        )
                    except Exception:
                        vol_ratio = float("nan")
                    try:
                        o = float(df_eval["open"].iloc[-1])
                        c = float(df_eval["close"].iloc[-1])
                        candle_pct = ((c / o) - 1.0) * 100.0 if pd.notna(o) and pd.notna(c) and o > 0 else float("nan")
                        if pd.notna(o) and pd.notna(c) and c > o:
                            spike_label = "▲ Up Spike"
                        elif pd.notna(o) and pd.notna(c) and c < o:
                            spike_label = "▼ Down Spike"
                        else:
                            spike_label = "→ Spike"
                    except Exception:
                        candle_pct = float("nan")
                        spike_label = "→ Spike"
                    parts = []
                    if pd.notna(vol_ratio):
                        parts.append(f"Vol Ratio: {vol_ratio:.2f}x")
                    if pd.notna(candle_pct):
                        parts.append(f"Candle: {candle_pct:+.2f}%")
                    vwap_ctx = str(ar.vwap or "").replace("🟢 ", "").replace("🔴 ", "").replace("→ ", "").strip()
                    if vwap_ctx:
                        parts.append(f"VWAP: {vwap_ctx}")
                    spike_hover = " | ".join(parts)

                supertrend_txt = _clean_indicator_text(ar.supertrend)
                ichimoku_txt = _clean_indicator_text(ar.ichimoku)
                vwap_txt = _clean_indicator_text(ar.vwap)
                adx_txt = _clean_indicator_text(_adx_bucket_only(ar.adx))
                bollinger_txt = _clean_indicator_text(ar.bollinger)
                stochrsi_txt = _clean_indicator_text(_stochrsi_bucket(ar.stochrsi_k))
                psar_txt = _clean_indicator_text(ar.psar)
                will_txt = _clean_indicator_text(ar.williams)
                cci_txt = _clean_indicator_text(ar.cci)
                volume_txt = _clean_indicator_text(spike_label) if ar.volume_spike else ""
                volatility_txt = _clean_indicator_text(str(ar.atr_comment).replace("▲", "").replace("▼", "").replace("–", ""))
                pattern_txt = _clean_indicator_text(str(ar.candle_pattern).split(" (")[0] if ar.candle_pattern else "")

                trend_cells = "".join(
                    [
                        _indicator_cell("SuperTrend", supertrend_txt, "ATR-based trend line direction."),
                        _indicator_cell("Ichimoku", ichimoku_txt, f"Cloud trend context. {ichimoku_hover}".strip()),
                        _indicator_cell("VWAP", vwap_txt, "Price relative to volume-weighted average price."),
                        _indicator_cell("ADX", adx_txt, "Trend strength (not direction)."),
                        _indicator_cell("PSAR", psar_txt, "Parabolic SAR trend-following state."),
                    ]
                )
                momentum_cells = "".join(
                    [
                        _indicator_cell("StochRSI", stochrsi_txt, "Momentum pressure zone."),
                        _indicator_cell("Williams %R", will_txt, "Range-position momentum signal."),
                        _indicator_cell("CCI", cci_txt, "Mean-reversion momentum signal."),
                        _indicator_cell("Pattern", pattern_txt, "Latest candle pattern direction."),
                    ]
                )
                vol_cells = "".join(
                    [
                        _indicator_cell("Bollinger", bollinger_txt, "Band location (extension / pullback context)."),
                        _indicator_cell("Volatility", volatility_txt, "ATR/band-width regime."),
                        _indicator_cell("Volume", volume_txt, f"Abnormal volume event. {spike_hover}".strip()),
                    ]
                )

                if trend_cells or momentum_cells or vol_cells:
                    st.markdown(
                        f"<div class='spot-indicator-sep'>Indicators ({tf})</div>"
                        f"<div class='spot-indicator-wrap'>"
                        f"<div class='spot-indicator-group'><div class='spot-indicator-group-title'>Trend Structure</div>"
                        f"<div class='spot-indicator-grid'>{trend_cells}</div></div>"
                        f"<div class='spot-indicator-group'><div class='spot-indicator-group-title'>Momentum Signals</div>"
                        f"<div class='spot-indicator-grid'>{momentum_cells}</div></div>"
                        f"<div class='spot-indicator-group'><div class='spot-indicator-group-title'>Volatility & Volume</div>"
                        f"<div class='spot-indicator-grid'>{vol_cells}</div></div>"
                        f"</div>",
                        unsafe_allow_html=True,
                    )
            except Exception as exc:
                _debug(f"AI Workspace indicator grid error ({tf}): {exc}")
