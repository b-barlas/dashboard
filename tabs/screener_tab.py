from ui.ctx import get_ctx

from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd
import ta
from ui.snapshot_cache import live_or_snapshot


def render(ctx: dict) -> None:
    st = get_ctx(ctx, "st")
    ACCENT = get_ctx(ctx, "ACCENT")
    TEXT_MUTED = get_ctx(ctx, "TEXT_MUTED")
    POSITIVE = get_ctx(ctx, "POSITIVE")
    NEGATIVE = get_ctx(ctx, "NEGATIVE")
    WARNING = get_ctx(ctx, "WARNING")
    _tip = get_ctx(ctx, "_tip")
    fetch_ohlcv = get_ctx(ctx, "fetch_ohlcv")
    analyse = get_ctx(ctx, "analyse")
    ml_ensemble_predict = get_ctx(ctx, "ml_ensemble_predict")
    get_top_volume_usdt_symbols = get_ctx(ctx, "get_top_volume_usdt_symbols")
    """Advanced multi-condition screener."""
    st.markdown(
        f"""
        <style>
        .scr-kpi-grid {{
            display:grid;
            grid-template-columns:repeat(4,minmax(0,1fr));
            gap:10px;
            margin:8px 0 14px 0;
        }}
        .scr-kpi {{
            border:1px solid rgba(0,212,255,0.16);
            border-radius:12px;
            padding:12px 14px;
            background:linear-gradient(140deg, rgba(0,0,0,0.72), rgba(10,18,30,0.88));
        }}
        .scr-kpi-label {{
            color:{TEXT_MUTED};
            font-size:0.7rem;
            text-transform:uppercase;
            letter-spacing:0.8px;
        }}
        .scr-kpi-value {{
            color:{ACCENT};
            font-size:1.2rem;
            font-weight:700;
            margin-top:4px;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        f"<h2 style='color:{ACCENT};'>Advanced Screener</h2>",
        unsafe_allow_html=True,
    )
    st.markdown(
        f"<div class='panel-box'>"
        f"<b style='color:{ACCENT}; font-size:1rem;'>What does this tab show?</b>"
        f"<p style='color:{TEXT_MUTED}; font-size:0.9rem; margin-top:6px; line-height:1.6;'>"
        f"Scans a liquid coin universe and returns symbols that pass your technical filters. "
        f"Use it to shortlist candidates before deep validation in Spot/Position/Fibonacci.</p>"
        f"<p style='color:{TEXT_MUTED}; font-size:0.85rem; margin-top:6px; line-height:1.6;'>"
        f"<b>Filters:</b> "
        f"{_tip('Min Confidence', 'Minimum confidence score. E.g. setting 60% shows only coins with 60%+ confidence.')} | "
        f"{_tip('Signal Filter', 'Choose which signal types to display: STRONG BUY, BUY, SELL, etc.')} | "
        f"{_tip('Min ADX', 'Minimum trend strength. ADX above 20 means a trending market. 25+ is a strong trend.')} | "
        f"{_tip('RSI Range', 'Sets the RSI range. 30-70 is neutral, below 30 is oversold, above 70 is overbought.')} | "
        f"{_tip('Volume Spike Only', 'When checked, only coins showing abnormal volume increases are listed.')} | "
        f"{_tip('AI Agree', 'Agreement inside Ensemble AI models. Higher agreement means more stable AI direction.')} </p>"
        f"</div>",
        unsafe_allow_html=True,
    )

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        scr_tf = st.selectbox("Timeframe", ['5m', '15m', '1h', '4h', '1d'], index=2, key="scr_tf")
        min_confidence = st.slider("Min Confidence %", 0, 100, 60, key="scr_conf")
    with col2:
        signal_filter = st.multiselect("Signal Filter", ['STRONG BUY', 'BUY', 'WAIT', 'SELL', 'STRONG SELL'],
                                        default=['STRONG BUY', 'BUY'], key="scr_signal")
        min_adx = st.slider("Min ADX", 0, 80, 20, key="scr_adx")
    with col3:
        rsi_range = st.slider("RSI Range", 0, 100, (20, 80), key="scr_rsi")
        volume_spike_only = st.checkbox("Volume Spike Only", value=False, key="scr_volspike")
    with col4:
        universe_mode = st.selectbox("Universe", ["Dynamic Top Volume", "Core Majors"], index=0, key="scr_universe_mode")
        universe_size = st.slider("Universe Size", 10, 80, 30, step=5, key="scr_universe_size")

    st.markdown(
        f"<div class='scr-kpi-grid'>"
        f"<div class='scr-kpi'><div class='scr-kpi-label'>Timeframe</div><div class='scr-kpi-value'>{scr_tf}</div></div>"
        f"<div class='scr-kpi'><div class='scr-kpi-label'>Confidence Floor</div><div class='scr-kpi-value'>{min_confidence}%</div></div>"
        f"<div class='scr-kpi'><div class='scr-kpi-label'>ADX Floor</div><div class='scr-kpi-value'>{min_adx}</div></div>"
        f"<div class='scr-kpi'><div class='scr-kpi-label'>RSI Window</div><div class='scr-kpi-value'>{rsi_range[0]}-{rsi_range[1]}</div></div>"
        f"</div>",
        unsafe_allow_html=True,
    )

    if st.button("Run Screener", type="primary", key="scr_run"):
        with st.spinner("Scanning markets..."):
            core_majors = [
                "BTC/USDT", "ETH/USDT", "SOL/USDT", "BNB/USDT", "XRP/USDT", "ADA/USDT",
                "DOGE/USDT", "AVAX/USDT", "DOT/USDT", "LINK/USDT", "MATIC/USDT", "UNI/USDT",
                "ATOM/USDT", "NEAR/USDT", "LTC/USDT", "BCH/USDT",
            ]
            if universe_mode == "Dynamic Top Volume":
                dynamic_symbols, _raw = get_top_volume_usdt_symbols(max(universe_size, 30))
                symbols_to_scan = (dynamic_symbols or core_majors)[:universe_size]
            else:
                symbols_to_scan = core_majors[:universe_size]

            results = []
            data_hits = 0
            def _scan_one(sym: str) -> tuple[bool, dict | None]:
                try:
                    df = fetch_ohlcv(sym, scr_tf, limit=120)
                    if df is None or len(df) < 55:
                        return False, None
                    a = analyse(df)
                    if a.confidence < min_confidence:
                        return True, None
                    if signal_filter and a.signal not in signal_filter:
                        return True, None
                    if not np.isnan(a.adx) and a.adx < min_adx:
                        return True, None
                    rsi_val = ta.momentum.rsi(df['close'], window=14).iloc[-1]
                    if rsi_val < rsi_range[0] or rsi_val > rsi_range[1]:
                        return True, None
                    if volume_spike_only and not a.volume_spike:
                        return True, None
                    try:
                        ai_prob, ai_dir, ai_details = ml_ensemble_predict(df)
                    except Exception:
                        ai_prob, ai_dir, ai_details = 0.5, "NEUTRAL", {}
                    ai_agree = float(ai_details.get("agreement", 0.0)) if isinstance(ai_details, dict) else 0.0
                    row = {
                        "Symbol": sym.split("/")[0],
                        "Price": float(df["close"].iloc[-1]),
                        "Signal": a.signal,
                        "Confidence": float(a.confidence),
                        "AI Ensemble": ai_dir,
                        "AI Prob %": ai_prob * 100.0,
                        "AI Agree %": ai_agree * 100.0,
                        "RSI": round(float(rsi_val), 1),
                        "ADX": round(float(a.adx), 1) if not np.isnan(a.adx) else 0.0,
                        "Volume Spike": "Yes" if a.volume_spike else "No",
                        "Leverage": int(a.leverage),
                    }
                    return True, row
                except Exception:
                    return False, None

            with ThreadPoolExecutor(max_workers=6) as executor:
                futures = [executor.submit(_scan_one, sym) for sym in symbols_to_scan]
                for fut in as_completed(futures):
                    hit, row = fut.result()
                    if hit:
                        data_hits += 1
                    if row is not None:
                        results.append(row)

        # Fallback only for data outages (not strict-filter empty matches).
        if data_hits == 0:
            results, from_cache, cache_ts = live_or_snapshot(
                st,
                f"screener_results::{scr_tf}::{min_confidence}::{min_adx}::{rsi_range[0]}-{rsi_range[1]}::{volume_spike_only}::{universe_mode}::{universe_size}",
                results,
            )
            if from_cache:
                st.warning(f"Live screener data unavailable. Showing cached snapshot from {cache_ts}.")
        elif results:
            live_or_snapshot(
                st,
                f"screener_results::{scr_tf}::{min_confidence}::{min_adx}::{rsi_range[0]}-{rsi_range[1]}::{volume_spike_only}::{universe_mode}::{universe_size}",
                results,
            )

        if results:
            st.markdown(
                f"<div style='background:rgba(0,255,136,0.08); border:1px solid rgba(0,255,136,0.3); "
                f"border-radius:10px; padding:12px; margin:10px 0; text-align:center;'>"
                f"<span style='color:{POSITIVE}; font-weight:700; font-size:1.2rem;'>"
                f"{len(results)} coins match</span></div>",
                unsafe_allow_html=True,
            )
            st.markdown(
                f"<details style='margin-bottom:0.6rem;'>"
                f"<summary style='color:{ACCENT}; cursor:pointer;'>How to read quickly (?)</summary>"
                f"<div style='color:{TEXT_MUTED}; font-size:0.85rem; line-height:1.7; margin-top:0.5rem;'>"
                f"<b>1.</b> Start from highest <b>Confidence</b> rows.<br>"
                f"<b>2.</b> Prefer rows where <b>Signal</b> and <b>AI Ensemble</b> agree.<br>"
                f"<b>3.</b> Look for <b>AI Agree % >= 66%</b> to avoid noisy model splits.<br>"
                f"<b>4.</b> Keep <b>ADX</b> above your floor and avoid extreme RSI edges unless intentional.<br>"
                f"<b>5.</b> Treat Screener as shortlist only; validate in Spot/Position before acting."
                f"</div></details>",
                unsafe_allow_html=True,
            )
            st.markdown(
                f"<details style='margin-bottom:0.8rem;'>"
                f"<summary style='color:{ACCENT}; cursor:pointer;'>Column Guide</summary>"
                f"<div style='color:{TEXT_MUTED}; font-size:0.84rem; line-height:1.7; margin-top:0.5rem;'>"
                f"<b>Signal</b>: technical engine output (BUY/SELL/WAIT).<br>"
                f"<b>Confidence</b>: strength of technical alignment.<br>"
                f"<b>AI Ensemble</b>: 3-model combined direction.<br>"
                f"<b>AI Prob %</b>: ensemble probability of upward move.<br>"
                f"<b>AI Agree %</b>: model agreement inside the ensemble.<br>"
                f"<b>Volume Spike</b>: latest volume anomaly flag.<br>"
                f"<b>Leverage</b>: risk-based suggestion cap."
                f"</div></details>",
                unsafe_allow_html=True,
            )

            df_results = pd.DataFrame(results).sort_values(by="Confidence", ascending=False).reset_index(drop=True)
            df_show = df_results.copy()
            df_show["Price"] = df_show["Price"].map(lambda x: f"${x:,.4f}")
            df_show["Confidence"] = df_show["Confidence"].map(lambda x: f"{x:.0f}%")
            df_show["AI Prob %"] = df_show["AI Prob %"].map(lambda x: f"{x:.1f}%")
            df_show["AI Agree %"] = df_show["AI Agree %"].map(lambda x: f"{x:.0f}%")
            df_show["Leverage"] = df_show["Leverage"].map(lambda x: f"x{x}")

            def _sig_style(v: str) -> str:
                if v in {"STRONG BUY", "BUY"}:
                    return f"color:{POSITIVE}; font-weight:700;"
                if v in {"STRONG SELL", "SELL"}:
                    return f"color:{NEGATIVE}; font-weight:700;"
                return f"color:{WARNING}; font-weight:700;"

            def _ai_style(v: str) -> str:
                if v == "LONG":
                    return f"color:{POSITIVE}; font-weight:700;"
                if v == "SHORT":
                    return f"color:{NEGATIVE}; font-weight:700;"
                return f"color:{WARNING}; font-weight:700;"

            def _conf_style(v: str) -> str:
                n = float(str(v).replace("%", ""))
                if n >= 70:
                    return f"color:{POSITIVE}; font-weight:700;"
                if n >= 50:
                    return f"color:{WARNING}; font-weight:700;"
                return f"color:{NEGATIVE}; font-weight:700;"

            styled = (
                df_show.style
                .map(_sig_style, subset=["Signal"])
                .map(_ai_style, subset=["AI Ensemble"])
                .map(_conf_style, subset=["Confidence", "AI Agree %"])
            )
            st.dataframe(styled, width="stretch")
        else:
            st.warning("No coins matched. Try relaxing filters.")
