from ui.ctx import get_ctx

from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd
import ta
from core.signal_contract import strength_from_bias, strength_bucket
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

    signal_display_to_raw = {
        "Strong Upside": "STRONG BUY",
        "Upside": "BUY",
        "Wait": "WAIT",
        "Downside": "SELL",
        "Strong Downside": "STRONG SELL",
    }
    signal_raw_to_display = {v: k for k, v in signal_display_to_raw.items()}

    def _ai_display(v: str) -> str:
        s = str(v or "").strip().upper()
        if s in {"LONG", "UPSIDE", "BUY"}:
            return "Upside"
        if s in {"SHORT", "DOWNSIDE", "SELL"}:
            return "Downside"
        return "Neutral"

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
        f"{_tip('Min Strength', 'Direction-agnostic signal strength. High values can come from strong Upside or strong Downside regimes.')} | "
        f"{_tip('Signal Filter', 'Choose which signal types to display: Strong Upside, Upside, Downside, etc.')} | "
        f"{_tip('Min ADX', 'Minimum trend strength. ADX above 20 means a trending market. 25+ is a strong trend.')} | "
        f"{_tip('RSI Range', 'Sets the RSI range. 30-70 is neutral, below 30 is oversold, above 70 is overbought.')} | "
        f"{_tip('Volume Spike Only', 'When checked, only coins showing abnormal volume increases are listed.')} | "
        f"{_tip('AI Agree', 'Directional model agreement inside Ensemble AI (x/3 for final Upside/Downside direction). Neutral consensus does not count as directional agreement.')} </p>"
        f"</div>",
        unsafe_allow_html=True,
    )

    signal_options = ["Strong Upside", "Upside", "Wait", "Downside", "Strong Downside"]
    legacy_signal_map = {
        "STRONG BUY": "Strong Upside",
        "BUY": "Upside",
        "WAIT": "Wait",
        "SELL": "Downside",
        "STRONG SELL": "Strong Downside",
    }
    existing_signal_state = st.session_state.get("scr_signal")
    if isinstance(existing_signal_state, list):
        mapped = [
            legacy_signal_map.get(str(v).upper(), str(v))
            for v in existing_signal_state
        ]
        mapped = [v for v in mapped if v in signal_options]
        st.session_state["scr_signal"] = mapped or ["Strong Upside", "Upside"]

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        scr_tf = st.selectbox("Timeframe", ['5m', '15m', '1h', '4h', '1d'], index=2, key="scr_tf")
        min_strength = st.slider("Min Strength %", 0, 100, 55, key="scr_strength")
    with col2:
        signal_filter_display = st.multiselect(
            "Signal Filter",
            signal_options,
            default=["Strong Upside", "Upside"],
            key="scr_signal",
        )
        signal_filter_raw = [signal_display_to_raw[s] for s in signal_filter_display if s in signal_display_to_raw]
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
        f"<div class='scr-kpi'><div class='scr-kpi-label'>Strength Floor</div><div class='scr-kpi-value'>{min_strength}%</div></div>"
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
                    df_eval = df.iloc[:-1].copy() if len(df) > 60 else df.copy()
                    if df_eval is None or len(df_eval) < 55:
                        return False, None
                    a = analyse(df_eval)
                    bias = float(a.bias)
                    strength = float(strength_from_bias(bias))
                    if strength < min_strength:
                        return True, None
                    if signal_filter_raw and a.signal not in signal_filter_raw:
                        return True, None
                    if not np.isnan(a.adx) and a.adx < min_adx:
                        return True, None
                    rsi_val = ta.momentum.rsi(df_eval['close'], window=14).iloc[-1]
                    if rsi_val < rsi_range[0] or rsi_val > rsi_range[1]:
                        return True, None
                    if volume_spike_only and not a.volume_spike:
                        return True, None
                    try:
                        _ai_prob, ai_dir, ai_details = ml_ensemble_predict(df_eval)
                    except Exception:
                        _ai_prob, ai_dir, ai_details = 0.5, "NEUTRAL", {}
                    ai_agree = float(ai_details.get("agreement", 0.0)) if isinstance(ai_details, dict) else 0.0
                    ai_votes = max(0, min(3, int(round(ai_agree * 3.0))))
                    spike_label = ""
                    if a.volume_spike:
                        try:
                            o = float(df_eval["open"].iloc[-1])
                            c = float(df_eval["close"].iloc[-1])
                            prev_vol_avg = float(df_eval["volume"].iloc[-21:-1].mean()) if len(df_eval) >= 21 else float("nan")
                            last_vol = float(df_eval["volume"].iloc[-1]) if len(df_eval) >= 1 else float("nan")
                            vol_ratio = (
                                last_vol / prev_vol_avg
                                if pd.notna(prev_vol_avg) and prev_vol_avg > 0 and pd.notna(last_vol)
                                else float("nan")
                            )
                            if pd.notna(o) and pd.notna(c) and c > o:
                                base_lbl = "▲ Up Spike"
                            elif pd.notna(o) and pd.notna(c) and c < o:
                                base_lbl = "▼ Down Spike"
                            else:
                                base_lbl = "→ Spike"
                            spike_label = f"{base_lbl} ({vol_ratio:.2f}x)" if pd.notna(vol_ratio) else base_lbl
                        except Exception:
                            spike_label = "→ Spike"
                    row = {
                        "Symbol": sym.split("/")[0],
                        "Price": float(df_eval["close"].iloc[-1]),
                        "Signal": signal_raw_to_display.get(str(a.signal), str(a.signal)),
                        "Strength": strength,
                        "AI Ensemble": _ai_display(ai_dir),
                        "AI Agree": f"{ai_votes}/3",
                        "RSI": round(float(rsi_val), 1),
                        "ADX": round(float(a.adx), 1) if not np.isnan(a.adx) else 0.0,
                        "Volume Spike": spike_label,
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
                f"screener_results::{scr_tf}::{min_strength}::{min_adx}::{rsi_range[0]}-{rsi_range[1]}::{volume_spike_only}::{universe_mode}::{universe_size}",
                results,
                max_age_sec=900,
                current_sig=(scr_tf, min_strength, min_adx, rsi_range[0], rsi_range[1], volume_spike_only, universe_mode, universe_size),
            )
            if from_cache:
                st.warning(f"Live screener data unavailable. Showing cached snapshot from {cache_ts}.")
        elif results:
            live_or_snapshot(
                st,
                f"screener_results::{scr_tf}::{min_strength}::{min_adx}::{rsi_range[0]}-{rsi_range[1]}::{volume_spike_only}::{universe_mode}::{universe_size}",
                results,
                max_age_sec=900,
                current_sig=(scr_tf, min_strength, min_adx, rsi_range[0], rsi_range[1], volume_spike_only, universe_mode, universe_size),
            )

        if results:
            df_results = pd.DataFrame(results).sort_values(by="Strength", ascending=False).reset_index(drop=True)
            buy_sell = df_results["Signal"].astype(str).str.upper()
            ai_side = df_results["AI Ensemble"].astype(str).str.upper()
            aligned_mask = (
                ((buy_sell.isin(["STRONG UPSIDE", "UPSIDE", "STRONG BUY", "BUY"])) & (ai_side.isin(["UPSIDE", "LONG"])))
                | ((buy_sell.isin(["STRONG DOWNSIDE", "DOWNSIDE", "STRONG SELL", "SELL"])) & (ai_side.isin(["DOWNSIDE", "SHORT"])))
            )
            aligned_share = float(aligned_mask.mean() * 100.0) if len(df_results) else 0.0
            avg_strength = float(df_results["Strength"].mean()) if len(df_results) else 0.0
            spike_share = (
                float(df_results["Volume Spike"].astype(str).str.contains("Spike", na=False).mean() * 100.0)
                if len(df_results)
                else 0.0
            )
            if aligned_share >= 65 and avg_strength >= 65:
                prof = "Cleaner shortlist"
                prof_text = "Signal and AI are broadly aligned. Prioritize top-strength rows first."
            elif aligned_share >= 45:
                prof = "Mixed shortlist"
                prof_text = "Some candidates align, but quality is uneven. Validate carefully in Spot/Position."
            else:
                prof = "Noisy shortlist"
                prof_text = "Alignment is weak. Consider stricter filters (higher strength/ADX)."

            st.markdown(
                f"<div style='background:rgba(0,255,136,0.08); border:1px solid rgba(0,255,136,0.3); "
                f"border-radius:10px; padding:12px; margin:10px 0; text-align:center;'>"
                f"<span style='color:{POSITIVE}; font-weight:700; font-size:1.2rem;'>"
                f"{len(results)} coins match</span></div>",
                unsafe_allow_html=True,
            )
            st.markdown(
                f"<div class='scr-kpi-grid'>"
                f"<div class='scr-kpi'><div class='scr-kpi-label'>Avg Strength</div><div class='scr-kpi-value'>{avg_strength:.0f}%</div></div>"
                f"<div class='scr-kpi'><div class='scr-kpi-label'>AI-Technical Alignment</div><div class='scr-kpi-value'>{aligned_share:.0f}%</div></div>"
                f"<div class='scr-kpi'><div class='scr-kpi-label'>Volume Spike Share</div><div class='scr-kpi-value'>{spike_share:.0f}%</div></div>"
                f"<div class='scr-kpi'><div class='scr-kpi-label'>Screen Profile</div><div class='scr-kpi-value'>{prof}</div></div>"
                f"</div>",
                unsafe_allow_html=True,
            )
            st.markdown(
                f"<div style='border:1px solid rgba(0,212,255,0.18); border-left:4px solid {ACCENT}; border-radius:12px; "
                f"padding:10px 12px; margin:2px 0 10px 0; background:linear-gradient(140deg, rgba(0,0,0,0.72), rgba(10,18,30,0.88)); "
                f"color:{TEXT_MUTED}; font-size:0.84rem; line-height:1.55;'>"
                f"<b style='color:{ACCENT};'>Quick Interpretation:</b> {prof_text}</div>",
                unsafe_allow_html=True,
            )
            st.markdown(
                f"<details style='margin-bottom:0.6rem;'>"
                f"<summary style='color:{ACCENT}; cursor:pointer;'>How to read quickly (?)</summary>"
                f"<div style='color:{TEXT_MUTED}; font-size:0.85rem; line-height:1.7; margin-top:0.5rem;'>"
                f"<b>1.</b> Start from highest <b>Strength</b> rows.<br>"
                f"<b>2.</b> Prefer rows where <b>Signal</b> and <b>AI Ensemble</b> agree.<br>"
                f"<b>3.</b> Prefer <b>AI Agree >= 2/3</b> for stronger directional confirmation.<br>"
                f"<b>4.</b> Keep <b>ADX</b> above your floor and avoid extreme RSI edges unless intentional.<br>"
                f"<b>5.</b> Treat Screener as shortlist only; validate in Spot/Position before acting."
                f"</div></details>",
                unsafe_allow_html=True,
            )
            st.markdown(
                f"<details style='margin-bottom:0.8rem;'>"
                f"<summary style='color:{ACCENT}; cursor:pointer;'>Column Guide</summary>"
                f"<div style='color:{TEXT_MUTED}; font-size:0.84rem; line-height:1.7; margin-top:0.5rem;'>"
                f"<b>Signal</b>: technical engine output (Strong Upside/Upside/Wait/Downside/Strong Downside).<br>"
                f"<b>Strength</b>: direction-agnostic signal power from directional bias (0-100).<br>"
                f"<b>AI Ensemble</b>: 3-model combined direction (Upside/Downside/Neutral).<br>"
                f"<b>AI Agree</b>: directional vote agreement inside the ensemble (x/3).<br>"
                f"<b>Volume Spike</b>: latest volume anomaly label (Up/Down/Neutral spike, with ratio).<br>"
                f"<b>RSI</b>: momentum location in the 0-100 range."
                f"</div></details>",
                unsafe_allow_html=True,
            )

            df_show = df_results.copy()
            df_show["Price"] = df_show["Price"].map(lambda x: f"${x:,.4f}")
            df_show["Strength"] = df_show["Strength"].map(lambda x: f"{x:.0f}%")

            def _sig_style(v: str) -> str:
                s = str(v or "").strip().upper()
                if s in {"STRONG UPSIDE", "UPSIDE", "STRONG BUY", "BUY"}:
                    return f"color:{POSITIVE}; font-weight:700;"
                if s in {"STRONG DOWNSIDE", "DOWNSIDE", "STRONG SELL", "SELL"}:
                    return f"color:{NEGATIVE}; font-weight:700;"
                return f"color:{WARNING}; font-weight:700;"

            def _ai_style(v: str) -> str:
                s = str(v or "").strip().upper()
                if s in {"UPSIDE", "LONG"}:
                    return f"color:{POSITIVE}; font-weight:700;"
                if s in {"DOWNSIDE", "SHORT"}:
                    return f"color:{NEGATIVE}; font-weight:700;"
                return f"color:{WARNING}; font-weight:700;"

            def _conf_style(v: str) -> str:
                n = float(str(v).replace("%", ""))
                b = strength_bucket(n)
                if b in {"STRONG", "GOOD"}:
                    return f"color:{POSITIVE}; font-weight:700;"
                if b == "MIXED":
                    return f"color:{WARNING}; font-weight:700;"
                return f"color:{NEGATIVE}; font-weight:700;"

            def _agree_style(v: str) -> str:
                try:
                    n = int(str(v).split("/")[0])
                except Exception:
                    return f"color:{TEXT_MUTED};"
                if n >= 3:
                    return f"color:{POSITIVE}; font-weight:700;"
                if n >= 2:
                    return f"color:{WARNING}; font-weight:700;"
                return f"color:{NEGATIVE}; font-weight:700;"

            def _spike_style(v: str) -> str:
                s = str(v or "").upper()
                if "UP SPIKE" in s:
                    return f"color:{POSITIVE}; font-weight:700;"
                if "DOWN SPIKE" in s:
                    return f"color:{NEGATIVE}; font-weight:700;"
                if "SPIKE" in s:
                    return f"color:{WARNING}; font-weight:700;"
                return f"color:{TEXT_MUTED};"

            styled = (
                df_show.style
                .map(_sig_style, subset=["Signal"])
                .map(_ai_style, subset=["AI Ensemble"])
                .map(_conf_style, subset=["Strength"])
                .map(_agree_style, subset=["AI Agree"])
                .map(_spike_style, subset=["Volume Spike"])
            )
            st.dataframe(styled, width="stretch")
        else:
            st.warning("No coins matched. Try relaxing filters.")
