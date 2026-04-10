from ui.ctx import get_ctx

import pandas as pd
import plotly.graph_objs as go
from core.ai_spot_bias import (
    ai_spot_bias_consensus_agreement,
    ai_spot_bias_directional_agreement,
    ai_spot_bias_display_votes,
    ai_spot_bias_probability_up,
    ai_spot_bias_status,
    build_ai_spot_bias_snapshot,
)
from core.confidence import (
    build_ai_confidence_snapshot,
    build_confidence_snapshot,
    build_execution_confidence_snapshot,
    confidence_bucket,
)
from core.market_decision import (
    ai_led_confirmation_snapshot,
    ai_vote_metrics,
    normalize_action_class,
    selected_timeframe_execution_snapshot,
    selected_timeframe_rr_ratio,
    spot_action_decision_with_reason,
    structure_state,
    trend_led_confirmation_snapshot,
)
from core.metric_catalog import AI_LONG_THRESHOLD, AI_SHORT_THRESHOLD, direction_from_prob
from core.signal_contract import bias_confidence_from_bias
from core.spot_direction import build_spot_direction_snapshot
from core.timeframe_anchors import resolve_anchor_plan
from core.trading_copy import copy_text
from ui.primitives import render_help_details, render_intro_card, render_kpi_grid, render_page_header
from ui.signal_panels import build_indicator_groups_html, build_setup_snapshot_html
from ui.signal_formatters import (
    adx_bucket_only as _adx_bucket_only,
    ai_confidence_note as _ai_confidence_note,
    ai_spot_note as _ai_spot_note,
    setup_confirm_display as _setup_confirm_display,
    spot_bias_label as _spot_bias_label,
)
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


def _prepare_closed_frame(df: pd.DataFrame | None, *, min_rows: int = 55) -> pd.DataFrame | None:
    if df is None:
        return None
    if len(df) <= int(min_rows):
        return None
    df_eval = df.iloc[:-1].copy()
    if len(df_eval) < int(min_rows):
        return None
    return df_eval


def _empty_matrix_row(tf: str) -> dict:
    return {
        "Timeframe": tf,
        "Direction": "NO DATA",
        "DirectionKey": "NO_DATA",
        "Selected Model Prob %": 0.0,
        "Ensemble Agree": "N/A",
        "Ensemble Agree %": 0.0,
        "AI Direction Bias %": 50.0,
        "AI Direction Bias": "Neutral (50.0%)",
        "AI Direction Bias Key": "NEUTRAL",
        "Reference Entry": "N/A",
        "Reference Target": "N/A",
        "Reference Source": "N/A",
        "AI Reference Entry": "N/A",
        "AI Reference Target": "N/A",
        "Technical Reference Entry": "N/A",
        "Technical Reference Target": "N/A",
    }


def _reference_plan_fields(
    ai_entry_txt: str,
    ai_target_txt: str,
    ta_entry_txt: str,
    ta_target_txt: str,
) -> dict:
    reference_entry = ai_entry_txt if ai_entry_txt != "N/A" else ta_entry_txt
    reference_target = ai_target_txt if ai_target_txt != "N/A" else ta_target_txt
    if ai_entry_txt != "N/A":
        reference_source = "AI-Aligned"
    elif ta_entry_txt != "N/A":
        reference_source = "Technical Context"
    else:
        reference_source = "No Reference"
    return {
        "Reference Entry": reference_entry,
        "Reference Target": reference_target,
        "Reference Source": reference_source,
        "AI Reference Entry": ai_entry_txt,
        "AI Reference Target": ai_target_txt,
        "Technical Reference Entry": ta_entry_txt,
        "Technical Reference Target": ta_target_txt,
    }


def _summarize_workspace_rows(rows: list[dict]) -> dict[str, float | str | int]:
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
    return {
        "dominant": dominant,
        "avg_prob": avg_prob,
        "avg_agree": avg_agree,
        "consistency": consistency,
        "tf_valid_count": tf_valid_count,
    }


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
            "<b>Model & Timeframe Matrix</b> compares agreement, probability, and reference-level context across selected frames."
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
                "Use this mode to see immediate AI direction, probability, and model agreement. "
                "Treat it as a diagnostic bias read, not a stand-alone execution decision."
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
            "agreement quality, and reference-level context across selected frames."
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
            "<b>3.</b> Treat reference levels as context only; always validate in Spot/Position tabs."
        ),
    )
    st.markdown(
        f"<div style='margin:-2px 0 10px 0; color:{TEXT_MUTED}; font-size:0.84rem; line-height:1.6;'>"
        f"{_tip('Reference Entry / Reference Target', 'Primary reference levels shown in the table. If AI and technical direction align, AI-aligned levels are preferred; otherwise technical context levels are shown.')}"
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
                rows.append(_empty_matrix_row(tf))
                continue

            df_eval = df.iloc[:-1].copy() if len(df) > 60 else df.copy()
            if df_eval is None or len(df_eval) < 55:
                tf_eval_cache[tf] = None
                rows.append(_empty_matrix_row(tf))
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
                directional_confidence = bias_confidence_from_bias(float(ar.bias))
                base_conviction_lbl, _ = _calc_conviction(
                    signal_dir,
                    direction_key,
                    directional_confidence,
                    directional_agreement,
                )
                execution_confidence = build_execution_confidence_snapshot(
                    direction=signal_dir,
                    bias_score=float(ar.bias),
                    adx_val=float(ar.adx) if pd.notna(ar.adx) else float("nan"),
                    structure_state=structure_state(
                        signal_dir,
                        direction_key,
                        directional_confidence,
                        directional_agreement,
                    ),
                    conviction_label=str(base_conviction_lbl),
                    ai_agreement=float(directional_agreement),
                )
                conviction_lbl, _ = _calc_conviction(
                    signal_dir,
                    direction_key,
                    float(execution_confidence.score),
                    directional_agreement,
                )
                scalp_ok, _scalp_reason = scalp_quality_gate(
                    scalp_direction=scalp_dir,
                    signal_direction=signal_dir,
                    rr_ratio=rr_ratio,
                    adx_val=ar.adx,
                    confidence=float(execution_confidence.score),
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
                    "__ensemble_side": ensemble_side,
                    "__ai_votes": int(ai_votes_for_decision),
                    "__decision_agreement": round(float(decision_agreement_for_decision), 4),
                    **_reference_plan_fields(ai_entry_txt, ai_target_txt, ta_entry_txt, ta_target_txt),
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
    summary = _summarize_workspace_rows(rows)
    dominant = str(summary["dominant"])
    avg_prob = float(summary["avg_prob"])
    avg_agree = float(summary["avg_agree"])
    consistency = float(summary["consistency"])
    tf_valid_count = int(summary["tf_valid_count"])

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
        if "AI-Aligned" in s:
            return f"color:{POSITIVE}; font-weight:700;"
        if "Technical Context" in s:
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
            "<b>Reference Entry/Reference Target</b>: context levels only, not a stand-alone execution instruction.<br>"
            "<b>Reference Source</b>: AI-Aligned or Technical Context."
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
                "Reference Entry",
                "Reference Target",
                "Reference Source",
            ]
        ].style.format({"Selected Model Prob %": "{:.1f}%"})
        .map(_dir_style, subset=["Direction"])
        .map(_prob_style, subset=["Selected Model Prob %"])
        .map(_dir_style, subset=["AI Direction Bias"])
        .map(_agree_style, subset=["Ensemble Agree"])
        .map(_plan_source_style, subset=["Reference Source"]),
        width="stretch",
        hide_index=True,
    )

    with st.expander("Show Reference Debug Columns (AI vs Technical)"):
        st.dataframe(
            df_out[
                [
                    "Timeframe",
                    "AI Reference Entry",
                    "AI Reference Target",
                    "Technical Reference Entry",
                    "Technical Reference Target",
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
    htf_cache: dict[str, pd.DataFrame | None] = {}
    with st.expander("Show Technical Indicator Panels"):
        for tf in selected_timeframes:
            df_eval = tf_eval_cache.get(tf)
            if df_eval is None or len(df_eval) < 55:
                st.caption(f"{tf}: no sufficient data.")
                continue
            try:
                anchor_plan = resolve_anchor_plan(tf)
                confirm_frame = htf_cache.get(anchor_plan.confirm_timeframe)
                if anchor_plan.confirm_timeframe not in htf_cache:
                    confirm_frame = _prepare_closed_frame(
                        fetch_ohlcv(coin, anchor_plan.confirm_timeframe, limit=260),
                        min_rows=81,
                    )
                    htf_cache[anchor_plan.confirm_timeframe] = confirm_frame
                lead_frame = htf_cache.get(anchor_plan.lead_timeframe)
                if anchor_plan.lead_timeframe not in htf_cache:
                    lead_frame = _prepare_closed_frame(
                        fetch_ohlcv(coin, anchor_plan.lead_timeframe, limit=260),
                        min_rows=81,
                    )
                    htf_cache[anchor_plan.lead_timeframe] = lead_frame

                spot_snapshot = build_spot_direction_snapshot(
                    df_4h=None,
                    df_1d=None,
                    confirm_df=confirm_frame,
                    lead_df=lead_frame,
                    confirm_timeframe=anchor_plan.confirm_timeframe,
                    lead_timeframe=anchor_plan.lead_timeframe,
                )
                confidence_snapshot = build_confidence_snapshot(
                    direction=spot_snapshot.direction,
                    timeframe_alignment=spot_snapshot.timeframe_alignment,
                    structure_quality=spot_snapshot.structure_quality,
                    trend_quality=spot_snapshot.trend_quality,
                    regime_quality=spot_snapshot.regime_quality,
                    location_quality=spot_snapshot.location_quality,
                    timeframe_conflict=spot_snapshot.timeframe_conflict,
                    degraded_data=spot_snapshot.degraded_data,
                    range_regime=spot_snapshot.range_regime,
                )
                ai_spot_snapshot = build_ai_spot_bias_snapshot(
                    df_4h=None,
                    df_1d=None,
                    confirm_df=confirm_frame,
                    lead_df=lead_frame,
                    confirm_timeframe=anchor_plan.confirm_timeframe,
                    lead_timeframe=anchor_plan.lead_timeframe,
                    predictor=ml_ensemble_predict,
                )
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
                signal_color = (
                    POSITIVE if str(spot_snapshot.direction).upper() == "UPSIDE"
                    else (NEGATIVE if str(spot_snapshot.direction).upper() == "DOWNSIDE" else WARNING)
                )

                row_meta = tf_row_map.get(str(tf), {})
                ensemble_side = str(row_meta.get("__ensemble_side", "")).upper()
                if ensemble_side not in {"UPSIDE", "DOWNSIDE", "NEUTRAL"}:
                    ensemble_side = _to_side_key(str(row_meta.get("Direction", "Neutral")))
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

                directional_confidence = bias_confidence_from_bias(float(ar.bias))
                confidence_display = f"{float(confidence_snapshot.score):.0f}% ({confidence_bucket(float(confidence_snapshot.score)).title()})"
                confidence_color = (
                    POSITIVE if confidence_bucket(float(confidence_snapshot.score)) == "HIGH"
                    else (WARNING if confidence_bucket(float(confidence_snapshot.score)) == "MEDIUM" else NEGATIVE)
                )
                base_conviction_lbl, _ = _calc_conviction(
                    signal_side_for_conviction,
                    ensemble_side,
                    directional_confidence,
                    decision_agreement,
                )
                execution_confidence = build_execution_confidence_snapshot(
                    direction=signal_side_for_conviction,
                    bias_score=float(ar.bias),
                    adx_val=float(ar.adx) if pd.notna(ar.adx) else float("nan"),
                    structure_state=structure_state(
                        signal_side_for_conviction,
                        ensemble_side,
                        directional_confidence,
                        decision_agreement,
                    ),
                    conviction_label=str(base_conviction_lbl),
                    ai_agreement=float(decision_agreement),
                )
                conviction_lbl, _ = _calc_conviction(
                    signal_side_for_conviction,
                    ensemble_side,
                    float(execution_confidence.score),
                    decision_agreement,
                )
                ai_spot_side = str(ai_spot_snapshot.direction).upper()
                ai_spot_votes = ai_spot_bias_display_votes(ai_spot_snapshot)
                ai_confidence_snapshot = build_ai_confidence_snapshot(
                    direction=ai_spot_snapshot.direction,
                    combined_score=float(ai_spot_snapshot.score),
                    conviction_quality=float(ai_spot_snapshot.conviction_quality),
                    timeframe_alignment=float(ai_spot_snapshot.timeframe_alignment),
                    consensus_quality=float(ai_spot_snapshot.consensus_quality),
                    support_votes=int(ai_spot_votes),
                    timeframe_conflict=bool(ai_spot_snapshot.timeframe_conflict),
                    degraded_data=bool(ai_spot_snapshot.degraded_data),
                )
                ai_spot_agreement = float(ai_spot_bias_directional_agreement(ai_spot_snapshot))
                ai_spot_consensus = float(ai_spot_bias_consensus_agreement(ai_spot_snapshot))
                ai_spot_probability_up = float(ai_spot_bias_probability_up(ai_spot_snapshot))
                ai_spot_status = str(ai_spot_bias_status(ai_spot_snapshot) or "")
                ai_conf_bucket = str(ai_confidence_snapshot.label or "LOW").upper()
                ai_conf_color = POSITIVE if ai_conf_bucket == "HIGH" else (WARNING if ai_conf_bucket == "MEDIUM" else NEGATIVE)
                df_scalp = df_eval.tail(120).copy()
                rr_ratio = float("nan")
                if df_scalp is not None and len(df_scalp) > 30:
                    try:
                        _scalp_dir, _entry_s, _target_s, _stop_s, rr_ratio, _breakout_note = get_scalping_entry_target(
                            df_scalp,
                            ar.bias,
                            ar.supertrend,
                            ar.ichimoku,
                            ar.vwap,
                        )
                    except Exception:
                        rr_ratio = float("nan")

                execution_snapshot = selected_timeframe_execution_snapshot(
                    df=df_eval,
                    direction=spot_snapshot.direction,
                    bias_score=float(ar.bias),
                    adx_val=float(ar.adx) if pd.notna(ar.adx) else float("nan"),
                    supertrend_trend=str(ar.supertrend),
                    ichimoku_trend=str(ar.ichimoku),
                    vwap_label=str(ar.vwap),
                    psar_trend=str(ar.psar),
                    bollinger_bias=str(ar.bollinger),
                    williams_label=str(ar.williams),
                    cci_label=str(ar.cci),
                )
                setup_rr_ratio = float(selected_timeframe_rr_ratio(execution_snapshot, direction=spot_snapshot.direction))
                trend_led_snapshot = trend_led_confirmation_snapshot(
                    spot_dir=spot_snapshot.direction,
                    spot_confidence=float(confidence_snapshot.score),
                    tactical_dir=signal_side,
                    adx_val=float(ar.adx) if pd.notna(ar.adx) else float("nan"),
                    structure_quality=float(execution_snapshot.structure_quality),
                    trend_quality=float(execution_snapshot.trend_quality),
                    regime_quality=float(execution_snapshot.regime_quality),
                    location_quality=float(execution_snapshot.location_quality),
                    rr_ratio=setup_rr_ratio if pd.notna(setup_rr_ratio) and float(setup_rr_ratio) > 0.0 else None,
                )
                ai_led_snapshot = ai_led_confirmation_snapshot(
                    spot_dir=spot_snapshot.direction,
                    spot_confidence=float(confidence_snapshot.score),
                    ai_dir=ai_spot_side,
                    ai_probability=float(ai_spot_probability_up),
                    directional_agreement=float(ai_spot_agreement),
                    consensus_agreement=float(ai_spot_consensus),
                    adx_val=float(ar.adx) if pd.notna(ar.adx) else float("nan"),
                    location_quality=float(execution_snapshot.location_quality),
                    rr_ratio=setup_rr_ratio if pd.notna(setup_rr_ratio) and float(setup_rr_ratio) > 0.0 else None,
                    ai_status=ai_spot_status,
                )

                action_raw, _reason_code = spot_action_decision_with_reason(
                    spot_snapshot.direction,
                    float(confidence_snapshot.score),
                    signal_side,
                    ai_spot_snapshot.direction,
                    ai_spot_agreement,
                    float(ar.adx) if pd.notna(ar.adx) else float("nan"),
                    trend_led_snapshot=trend_led_snapshot,
                    ai_led_snapshot=ai_led_snapshot,
                )
                setup_confirm = _setup_confirm_display(action_raw)
                action_cls = normalize_action_class(action_raw)
                watch_setup_color = "#7DD3FC"
                setup_color = (
                    POSITIVE if action_cls.startswith("ENTER_")
                    else (WARNING if action_cls == "PROBE" else (watch_setup_color if action_cls == "WATCH" else NEGATIVE))
                )
                ai_spot_label = _spot_bias_label(ai_spot_snapshot.direction)
                ai_spot_color = (
                    POSITIVE if ai_spot_side == "UPSIDE"
                    else (NEGATIVE if ai_spot_side == "DOWNSIDE" else WARNING)
                )
                ai_spot_note = _ai_spot_note(ai_spot_snapshot)
                ai_confidence_note = _ai_confidence_note(ai_spot_snapshot, float(ai_confidence_snapshot.score))
                setup_snapshot_html = build_setup_snapshot_html(
                    title=f"Setup Snapshot ({tf})",
                    text_muted=TEXT_MUTED,
                    items=[
                        {
                            "label": "Δ (%)",
                            "value": delta_display,
                            "color": delta_color,
                            "title": "Closed-candle move on selected timeframe.",
                        },
                        {
                            "label": "Setup Confirm",
                            "value": setup_confirm,
                            "color": setup_color,
                            "title": copy_text("ml.setup_snapshot.setup_confirm_title"),
                        },
                        {
                            "label": "Direction",
                            "value": _spot_bias_label(spot_snapshot.direction),
                            "color": signal_color,
                            "title": "Higher-timeframe spot bias from the adaptive lead/confirm anchor pair.",
                        },
                        {
                            "label": "Confidence",
                            "value": confidence_display,
                            "color": confidence_color,
                            "title": "Quality score of the spot bias from timeframe alignment, structure, trend, regime, and location.",
                        },
                        {
                            "label": "AI Ensemble",
                            "value": f"{ai_spot_label} ({ai_spot_votes}/3){' *' if ai_spot_snapshot.degraded_data else ''}",
                            "color": ai_spot_color,
                            "title": ai_spot_note,
                        },
                        {
                            "label": "AI Confidence",
                            "value": f"{float(ai_confidence_snapshot.score):.0f}% ({ai_confidence_snapshot.label.title()})",
                            "color": ai_conf_color,
                            "title": ai_confidence_note,
                        },
                    ],
                )
                st.markdown(setup_snapshot_html, unsafe_allow_html=True)

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

                indicator_groups_html = build_indicator_groups_html(
                    title=f"Indicators ({tf})",
                    accent=ACCENT,
                    text_muted=TEXT_MUTED,
                    positive=POSITIVE,
                    negative=NEGATIVE,
                    warning=WARNING,
                    groups=[
                        (
                            "Trend Structure",
                            [
                                {"name": "SuperTrend", "value": ar.supertrend, "tooltip": "ATR-based trend line direction."},
                                {
                                    "name": "Ichimoku",
                                    "value": ar.ichimoku,
                                    "tooltip": f"Cloud trend context. {ichimoku_hover}".strip(),
                                },
                                {"name": "VWAP", "value": ar.vwap, "tooltip": "Price relative to volume-weighted average price."},
                                {"name": "ADX", "value": _adx_bucket_only(ar.adx), "tooltip": "Trend strength (not direction)."},
                                {"name": "PSAR", "value": ar.psar, "tooltip": "Parabolic SAR trend-following state."},
                            ],
                        ),
                        (
                            "Momentum Signals",
                            [
                                {"name": "StochRSI", "value": _stochrsi_bucket(ar.stochrsi_k), "tooltip": "Momentum pressure zone."},
                                {"name": "Williams %R", "value": ar.williams, "tooltip": "Range-position momentum signal."},
                                {"name": "CCI", "value": ar.cci, "tooltip": "Mean-reversion momentum signal."},
                                {
                                    "name": "Pattern",
                                    "value": str(ar.candle_pattern).split(" (")[0] if ar.candle_pattern else "",
                                    "tooltip": "Latest candle pattern direction.",
                                },
                            ],
                        ),
                        (
                            "Volatility & Volume",
                            [
                                {
                                    "name": "Bollinger",
                                    "value": ar.bollinger,
                                    "tooltip": "Band location (extension / pullback context).",
                                },
                                {
                                    "name": "Volatility",
                                    "value": str(ar.atr_comment).replace("▲", "").replace("▼", "").replace("–", ""),
                                    "tooltip": "ATR/band-width regime.",
                                },
                                {
                                    "name": "Volume",
                                    "value": spike_label if ar.volume_spike else "",
                                    "tooltip": f"Abnormal volume event. {spike_hover}".strip(),
                                },
                            ],
                        ),
                    ],
                )
                if indicator_groups_html:
                    st.markdown(indicator_groups_html, unsafe_allow_html=True)
            except Exception as exc:
                _debug(f"AI Workspace indicator grid error ({tf}): {exc}")
