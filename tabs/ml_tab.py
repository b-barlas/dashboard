from ui.ctx import get_ctx

import numpy as np
import pandas as pd
import plotly.graph_objs as go
from core.metric_catalog import AI_LONG_THRESHOLD, AI_SHORT_THRESHOLD, direction_from_prob
from ui.snapshot_cache import live_or_snapshot


def render(ctx: dict) -> None:
    st = get_ctx(ctx, "st")
    ACCENT = get_ctx(ctx, "ACCENT")
    TEXT_MUTED = get_ctx(ctx, "TEXT_MUTED")
    TEXT_LIGHT = get_ctx(ctx, "TEXT_LIGHT")
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
    analyse = get_ctx(ctx, "analyse")
    _build_indicator_grid = get_ctx(ctx, "_build_indicator_grid")
    get_major_ohlcv_bundle = get_ctx(ctx, "get_major_ohlcv_bundle")
    get_market_indices = get_ctx(ctx, "get_market_indices")
    _debug = get_ctx(ctx, "_debug")
    """AI Lab: model diagnostics and multi-timeframe probability panel."""

    st.markdown(
        f"""
        <style>
        .ailab-kpi-grid {{
            display:grid;
            grid-template-columns:repeat(4,minmax(0,1fr));
            gap:10px;
            margin:8px 0 12px 0;
        }}
        .ailab-kpi {{
            border:1px solid rgba(0,212,255,0.16);
            border-radius:12px;
            padding:12px 14px;
            background:linear-gradient(140deg, rgba(0,0,0,0.72), rgba(10,18,30,0.88));
        }}
        .ailab-kpi-label {{
            color:{TEXT_MUTED};
            font-size:0.70rem;
            text-transform:uppercase;
            letter-spacing:0.8px;
        }}
        .ailab-kpi-value {{
            color:{ACCENT};
            font-size:1.2rem;
            font-weight:700;
            margin-top:4px;
        }}
        .ailab-badge {{
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

    st.markdown(f"<h2 style='color:{ACCENT};'>AI Lab</h2>", unsafe_allow_html=True)
    st.markdown(
        f"<div class='panel-box'>"
        f"<b style='color:{ACCENT}; font-size:1rem;'>What does this tab show?</b>"
        f"<p style='color:{TEXT_MUTED}; font-size:0.9rem; margin-top:6px; line-height:1.6;'>"
        f"Model diagnostics for probabilistic direction. Select a model and up to 3 timeframes to compare direction stability. "
        f"Main app signal uses Ensemble in production."
        f"</p>"
        f"<p style='color:{TEXT_MUTED}; font-size:0.85rem; margin-top:6px; line-height:1.6;'>"
        f"{_tip('Probability', 'Estimated chance of upward move from selected model.')} | "
        f"{_tip('Agreement', 'How strongly base models agree on direction in ensemble context.')} | "
        f"{_tip('Market Outlook', 'Dominance-weighted probability across BTC/ETH/BNB/SOL/ADA/XRP for same timeframe.')}"
        f"</p>"
        f"</div>",
        unsafe_allow_html=True,
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
            default=["5m"],
            max_selections=3,
            key="ai_tfs",
        )

    st.markdown(
        f"<details style='margin-bottom:0.7rem;'>"
        f"<summary style='color:{ACCENT}; cursor:pointer;'>How to read quickly (?)</summary>"
        f"<div style='color:{TEXT_MUTED}; font-size:0.85rem; line-height:1.7; margin-top:0.5rem;'>"
        f"<b>1.</b> Prefer setups where direction is consistent across timeframes.<br>"
        f"<b>2.</b> If Probability is high but Agreement is low, treat as fragile signal.<br>"
        f"<b>3.</b> Use entry/target only as plan drafts; confirm in Spot/Position tabs."
        f"</div></details>",
        unsafe_allow_html=True,
    )
    st.markdown(
        f"<div style='margin:-2px 0 10px 0; color:{TEXT_MUTED}; font-size:0.84rem; line-height:1.6;'>"
        f"{_tip('AI Entry / AI Target', 'Technical plan that is kept only when technical direction matches selected AI model direction.')}<br>"
        f"{_tip('Non-AI Entry / Non-AI Target', 'Raw technical plan from indicators and scalping rules, without AI direction filter.')}"
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

    def _model_prob_direction(df_model: pd.DataFrame, model_name: str) -> tuple[float, str, dict]:
        prob_e, dir_e, details = ml_ensemble_predict(df_model)
        if model_name == "Ensemble":
            return float(prob_e), str(dir_e), details or {}
        key_map = {
            "Gradient Boosting": "gradient_boosting",
            "Random Forest": "random_forest",
            "Logistic Regression": "logistic_regression",
        }
        key = key_map.get(model_name)
        p = float((details or {}).get(key, prob_e))
        # Keep decision thresholds consistent with core ensemble mapping.
        d = direction_from_prob(float(p))
        return p, d, details or {}

    def _pct_badge(v: float, low: float, high: float) -> str:
        if v >= high:
            return f"▲ Strong ({v:.1f}%)"
        if v <= low:
            return f"▼ Weak ({v:.1f}%)"
        return f"■ Mixed ({v:.1f}%)"

    def _market_outlook_for_tf(tf: str) -> tuple[float, str]:
        try:
            bundle = get_major_ohlcv_bundle(tf, limit=500)
            probs = {}
            for sym in ["BTC/USDT", "ETH/USDT", "BNB/USDT", "SOL/USDT", "ADA/USDT", "XRP/USDT"]:
                dfx = bundle.get(sym)
                if dfx is not None and not dfx.empty and len(dfx) >= 60:
                    probs[sym], _, _ = _model_prob_direction(dfx, selected_model)
                else:
                    probs[sym] = 0.5
            btc_dom, eth_dom, _, _, _, bnb_dom, sol_dom, ada_dom, xrp_dom = get_market_indices()
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
                return 0.5, "NEUTRAL"
            p = sum(probs[s] * max(weights[s], 0.0) for s in weights) / wsum
            d = direction_from_prob(float(p))
            return float(p), d
        except Exception as exc:
            _debug(f"AI Lab market outlook error ({tf}): {exc}")
            return 0.5, "NEUTRAL"

    rows = []
    all_probs = []
    all_agreements = []
    all_dirs = []
    for tf in selected_timeframes:
        with st.spinner(f"Running {selected_model} for {tf}..."):
            df = fetch_ohlcv(coin, tf, limit=500)
            if df is None or len(df) < 60:
                rows.append(
                    {
                        "Timeframe": tf,
                        "Direction": "NO DATA",
                        "Prob %": 0.0,
                        "Agreement %": 0.0,
                        "Market %": 50.0,
                        "Market Dir": "NEUTRAL",
                        "Entry": "N/A",
                        "Target": "N/A",
                        "Leverage": "N/A",
                        "Status": "• N/A",
                        "Signal Icon": "•",
                    }
                )
                continue

            prob, direction, details = _model_prob_direction(df, selected_model)
            agreement = float((details or {}).get("agreement", 0.0)) * 100.0
            mkt_prob, mkt_dir = _market_outlook_for_tf(tf)

            ai_entry_txt = "N/A"
            ai_target_txt = "N/A"
            ta_entry_txt = "N/A"
            ta_target_txt = "N/A"
            lev_txt = "N/A"
            try:
                ar = analyse(df)
                scalp_dir, entry_s, target_s, _, _, _ = get_scalping_entry_target(
                    df,
                    ar.confidence,
                    ar.supertrend,
                    ar.ichimoku,
                    ar.vwap,
                    ar.volume_spike,
                    strict_mode=True,
                )
                if scalp_dir in {"LONG", "SHORT"} and entry_s and target_s:
                    ta_entry_txt = f"${entry_s:,.4f}"
                    ta_target_txt = f"${target_s:,.4f}"
                if scalp_dir == direction and direction in {"LONG", "SHORT"} and entry_s and target_s:
                    ai_entry_txt = f"${entry_s:,.4f}"
                    ai_target_txt = f"${target_s:,.4f}"
                    lev_txt = f"{ar.leverage}X"
            except Exception as exc:
                _debug(f"AI Lab planning block error ({tf}): {exc}")

            if direction == "LONG":
                status = "▲ Bullish"
                icon = "↗"
            elif direction == "SHORT":
                status = "▼ Bearish"
                icon = "↘"
            else:
                status = "■ Neutral"
                icon = "→"

            rows.append(
                {
                    "Timeframe": tf,
                    "Signal Icon": icon,
                    "Direction": direction,
                    "Status": status,
                    "Prob %": round(prob * 100.0, 1),
                    "Prob Badge": _pct_badge(float(prob * 100.0), AI_SHORT_THRESHOLD * 100.0, AI_LONG_THRESHOLD * 100.0),
                    "Agreement %": round(agreement, 1),
                    "Agreement Badge": _pct_badge(float(agreement), 40.0, 65.0),
                    "Market %": round(mkt_prob * 100.0, 1),
                    "Market Badge": _pct_badge(float(mkt_prob * 100.0), AI_SHORT_THRESHOLD * 100.0, AI_LONG_THRESHOLD * 100.0),
                    "Market Dir": mkt_dir,
                    "AI Entry": ai_entry_txt,
                    "AI Target": ai_target_txt,
                    "Non-AI Entry": ta_entry_txt,
                    "Non-AI Target": ta_target_txt,
                    "Leverage": lev_txt,
                }
            )
            all_probs.append(prob * 100.0)
            all_agreements.append(agreement)
            all_dirs.append(direction)

    live_valid = [r for r in rows if r.get("Direction") != "NO DATA"]
    snapshot_key = f"ailab_rows::{coin}::{selected_model}::{','.join(sorted(selected_timeframes))}"
    if len(live_valid) == 0:
        rows, from_cache, cache_ts = live_or_snapshot(st, snapshot_key, rows)
        if from_cache:
            st.warning(f"Live AI Lab data unavailable. Showing cached snapshot from {cache_ts}.")
    else:
        live_or_snapshot(st, snapshot_key, rows)

    df_out = pd.DataFrame(rows)
    valid_dirs = [d for d in all_dirs if d in {"LONG", "SHORT", "NEUTRAL"}]
    dominant = "NEUTRAL"
    if valid_dirs:
        dominant = max(set(valid_dirs), key=valid_dirs.count)
    avg_prob = float(np.mean(all_probs)) if all_probs else 0.0
    avg_agree = float(np.mean(all_agreements)) if all_agreements else 0.0
    consistency = (sum(1 for d in valid_dirs if d == dominant) / max(len(valid_dirs), 1)) * 100.0

    dom_color = POSITIVE if dominant == "LONG" else (NEGATIVE if dominant == "SHORT" else WARNING)
    st.markdown(
        f"<div class='ailab-kpi-grid'>"
        f"<div class='ailab-kpi'><div class='ailab-kpi-label'>Dominant Direction</div><div class='ailab-kpi-value' style='color:{dom_color};'>{dominant}</div></div>"
        f"<div class='ailab-kpi'><div class='ailab-kpi-label'>Avg Probability</div><div class='ailab-kpi-value'>{avg_prob:.1f}%</div></div>"
        f"<div class='ailab-kpi'><div class='ailab-kpi-label'>Avg Agreement</div><div class='ailab-kpi-value'>{avg_agree:.1f}%</div></div>"
        f"<div class='ailab-kpi'><div class='ailab-kpi-label'>TF Consistency</div><div class='ailab-kpi-value'>{consistency:.0f}%</div></div>"
        f"</div>",
        unsafe_allow_html=True,
    )

    def _status_style(v: str) -> str:
        if "Bullish" in str(v):
            return f"color:{POSITIVE}; font-weight:700;"
        if "Bearish" in str(v):
            return f"color:{NEGATIVE}; font-weight:700;"
        if "Neutral" in str(v):
            return f"color:{WARNING}; font-weight:700;"
        return f"color:{TEXT_MUTED};"

    def _icon_style(v: str) -> str:
        if v == "↗":
            return f"color:{POSITIVE}; font-weight:700;"
        if v == "↘":
            return f"color:{NEGATIVE}; font-weight:700;"
        if v == "→":
            return f"color:{WARNING}; font-weight:700;"
        return f"color:{TEXT_MUTED};"

    def _badge_style(v: str) -> str:
        s = str(v)
        if "Strong" in s:
            return f"color:{POSITIVE}; font-weight:700;"
        if "Weak" in s:
            return f"color:{NEGATIVE}; font-weight:700;"
        if "Mixed" in s:
            return f"color:{WARNING}; font-weight:700;"
        return f"color:{TEXT_MUTED};"

    def _lev_style(v: str) -> str:
        return f"color:{TEXT_MUTED};" if str(v) == "N/A" else f"color:{TEXT_LIGHT}; font-weight:700;"

    st.markdown(f"<b style='color:{ACCENT};'>AI Lab Timeframe Matrix</b>", unsafe_allow_html=True)
    st.dataframe(
        df_out[
            [
                "Timeframe",
                "Signal Icon",
                "Direction",
                "Status",
                "Prob %",
                "Prob Badge",
                "Agreement %",
                "Agreement Badge",
                "Market %",
                "Market Badge",
                "Market Dir",
                "AI Entry",
                "AI Target",
                "Non-AI Entry",
                "Non-AI Target",
                "Leverage",
            ]
        ].style.format({"Prob %": "{:.1f}", "Agreement %": "{:.1f}", "Market %": "{:.1f}"})
        .map(_status_style, subset=["Status", "Market Dir"])
        .map(_icon_style, subset=["Signal Icon"])
        .map(_badge_style, subset=["Prob Badge", "Agreement Badge", "Market Badge"])
        .map(_lev_style, subset=["Leverage"]),
        width="stretch",
        hide_index=True,
    )

    # Probability by timeframe chart.
    df_plot = df_out[df_out["Direction"] != "NO DATA"].copy()
    if not df_plot.empty:
        fig = go.Figure()
        fig.add_trace(
            go.Bar(
                x=df_plot["Timeframe"],
                y=df_plot["Prob %"],
                marker_color=[POSITIVE if d == "LONG" else (NEGATIVE if d == "SHORT" else WARNING) for d in df_plot["Direction"]],
                name="Model Prob %",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=df_plot["Timeframe"],
                y=df_plot["Market %"],
                mode="lines+markers",
                line=dict(color=ACCENT, width=2),
                name="Market Outlook %",
            )
        )
        fig.add_hline(y=AI_LONG_THRESHOLD * 100.0, line=dict(color=POSITIVE, dash="dot", width=1), annotation_text="LONG threshold")
        fig.add_hline(y=AI_SHORT_THRESHOLD * 100.0, line=dict(color=NEGATIVE, dash="dot", width=1), annotation_text="SHORT threshold")
        fig.update_layout(
            height=330,
            template="plotly_dark",
            margin=dict(l=20, r=20, t=35, b=20),
            yaxis_title="Probability (%)",
            paper_bgcolor=CARD_BG,
        )
        st.plotly_chart(fig, width="stretch")

    # Detailed indicator panel per timeframe (optional, collapse by default).
    with st.expander("Show Technical Indicator Panels"):
        for tf in selected_timeframes:
            df = fetch_ohlcv(coin, tf, limit=250)
            if df is None or len(df) < 60:
                st.caption(f"{tf}: no sufficient data.")
                continue
            try:
                ar = analyse(df)
                grid_html = _build_indicator_grid(
                    ar.supertrend,
                    ar.ichimoku,
                    ar.vwap,
                    ar.adx,
                    ar.bollinger,
                    ar.stochrsi_k,
                    ar.psar,
                    ar.williams,
                    ar.cci,
                    ar.volume_spike,
                    ar.atr_comment,
                    ar.candle_pattern,
                )
                st.markdown(f"<b style='color:{ACCENT};'>Indicators ({tf})</b>", unsafe_allow_html=True)
                st.markdown(grid_html, unsafe_allow_html=True)
            except Exception as exc:
                _debug(f"AI Lab indicator grid error ({tf}): {exc}")
