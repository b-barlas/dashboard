from ui.ctx import get_ctx

import numpy as np
import pandas as pd
import plotly.graph_objs as go
from ui.snapshot_cache import live_or_snapshot


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
    analyse = get_ctx(ctx, "analyse")
    format_trend = get_ctx(ctx, "format_trend")
    style_signal = get_ctx(ctx, "style_signal")
    """Multi-timeframe confluence analysis."""
    st.markdown(
        f"""
        <style>
        .mtf-kpi-grid {{
            display:grid;
            grid-template-columns:repeat(4,minmax(0,1fr));
            gap:10px;
            margin:8px 0 12px 0;
        }}
        .mtf-kpi {{
            border:1px solid rgba(0,212,255,0.16);
            border-radius:12px;
            padding:12px 14px;
            background:linear-gradient(140deg, rgba(0,0,0,0.72), rgba(10,18,30,0.88));
        }}
        .mtf-kpi-label {{
            color:{TEXT_MUTED};
            font-size:0.70rem;
            text-transform:uppercase;
            letter-spacing:0.8px;
        }}
        .mtf-kpi-value {{
            color:{ACCENT};
            font-size:1.2rem;
            font-weight:700;
            margin-top:4px;
        }}
        .mtf-badge {{
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

    def _direction_from_signal(signal: str) -> str:
        if signal in {"STRONG BUY", "BUY"}:
            return "LONG"
        if signal in {"STRONG SELL", "SELL"}:
            return "SHORT"
        return "WAIT"

    def _status(v: float, good: float, watch: float, lower_better: bool = False) -> tuple[str, str]:
        if lower_better:
            if v <= good:
                return "Healthy", POSITIVE
            if v <= watch:
                return "Watch", WARNING
            return "Risky", NEGATIVE
        if v >= good:
            return "Healthy", POSITIVE
        if v >= watch:
            return "Watch", WARNING
        return "Risky", NEGATIVE

    st.markdown(f"<h2 style='color:{ACCENT};'>Multi-Timeframe Confluence</h2>", unsafe_allow_html=True)
    st.markdown(
        f"<div class='panel-box'>"
        f"<b style='color:{ACCENT}; font-size:1rem;'>What does this tab show?</b>"
        f"<p style='color:{TEXT_MUTED}; font-size:0.9rem; margin-top:6px; line-height:1.6;'>"
        f"Runs the full technical analysis across all 5 timeframes (5m, 15m, 1h, 4h, 1d) simultaneously. "
        f"The {_tip('Confluence Score', 'Measures how many timeframes agree on the same direction. 100% = all 5 agree, 60% = 3 out of 5. Higher confluence = higher probability trade.')} "
        f"tells you how many timeframes agree. When short-term and long-term signals align, "
        f"the trade setup is much stronger.</p>"
        f"</div>",
        unsafe_allow_html=True,
    )
    coin = _normalize_coin_input(st.text_input("Coin (e.g. BTC, ETH, TAO)", value="BTC", key="mtf_coin_input"))
    if st.button("Run Multi-TF Analysis", type="primary"):
        _val_err = _validate_coin_symbol(coin)
        if _val_err:
            st.error(_val_err)
            return
        timeframes = ["5m", "15m", "1h", "4h", "1d"]
        tf_weights = {"5m": 1.0, "15m": 1.2, "1h": 1.6, "4h": 2.1, "1d": 2.6}
        with st.spinner("Analysing across all timeframes..."):
            rows = []
            for tf in timeframes:
                df = fetch_ohlcv(coin, tf, limit=200)
                if df is None or len(df) < 55:
                    rows.append({"Timeframe": tf, "Signal": "NO DATA", "Confidence": 0.0,
                                 "SuperTrend": "", "Ichimoku": "", "VWAP": "", "ADX": 0.0})
                    continue
                ar = analyse(df)
                direction = _direction_from_signal(ar.signal)
                rows.append({
                    "Timeframe": tf,
                    "Signal": direction,
                    "Confidence": round(ar.confidence, 1),
                    "SuperTrend": format_trend(ar.supertrend),
                    "Ichimoku": format_trend(ar.ichimoku),
                    "VWAP": ar.vwap,
                    "ADX": round(ar.adx, 1),
                    "Weight": tf_weights.get(tf, 1.0),
                })

            valid_live = [r for r in rows if r["Signal"] != "NO DATA"]
            if len(valid_live) == 0:
                rows, from_cache, cache_ts = live_or_snapshot(st, f"mtf_rows::{coin}", rows)
                if from_cache:
                    st.warning(f"Live multi-timeframe data unavailable. Showing cached snapshot from {cache_ts}.")
            else:
                live_or_snapshot(st, f"mtf_rows::{coin}", rows)

            # Confluence calculations
            valid = [r for r in rows if r["Signal"] != "NO DATA"]
            long_c = sum(1 for r in valid if r["Signal"] == "LONG")
            short_c = sum(1 for r in valid if r["Signal"] == "SHORT")
            total_valid = len(valid)
            avg_conf = np.mean([r["Confidence"] for r in valid]) if valid else 0.0
            wait_c = sum(1 for r in valid if r["Signal"] == "WAIT")

            if total_valid > 0:
                confluence_pct = max(long_c, short_c) / total_valid * 100
                dominant = "LONG" if long_c > short_c else ("SHORT" if short_c > long_c else "NEUTRAL")
            else:
                confluence_pct = 0
                dominant = "NEUTRAL"

            weighted_long = sum(r["Weight"] for r in valid if r["Signal"] == "LONG")
            weighted_short = sum(r["Weight"] for r in valid if r["Signal"] == "SHORT")
            weighted_total = sum(r["Weight"] for r in valid if r["Signal"] in {"LONG", "SHORT"})
            weighted_confluence = (max(weighted_long, weighted_short) / weighted_total * 100) if weighted_total > 0 else 0.0
            weighted_dominant = (
                "LONG" if weighted_long > weighted_short
                else ("SHORT" if weighted_short > weighted_long else "NEUTRAL")
            )

            # Confluence gauge
            conf_color = POSITIVE if dominant == "LONG" else (NEGATIVE if dominant == "SHORT" else WARNING)
            fig_conf = go.Figure(go.Indicator(
                mode="gauge+number",
                value=round(weighted_confluence),
                gauge={
                    "axis": {"range": [0, 100], "tickwidth": 1},
                    "bar": {"color": conf_color},
                    "bgcolor": CARD_BG,
                    "steps": [
                        {"range": [0, 40], "color": NEGATIVE},
                        {"range": [40, 60], "color": WARNING},
                        {"range": [60, 100], "color": POSITIVE},
                    ],
                },
                title={"text": f"Weighted Confluence ({weighted_dominant})", "font": {"size": 16, "color": ACCENT}},
                number={"font": {"color": ACCENT, "size": 38}, "suffix": "%"},
            ))
            fig_conf.update_layout(
                height=200, margin=dict(l=10, r=10, t=50, b=15),
                plot_bgcolor="#000000", paper_bgcolor="#000000",
            )
            st.plotly_chart(fig_conf, width="stretch")

            # Summary cards
            c1, c2, c3 = st.columns(3)
            with c1:
                st.markdown(
                    f"<div class='metric-card'><div class='metric-label'>Dominant Direction</div>"
                    f"<div class='metric-value' style='color:{conf_color};'>{weighted_dominant}</div></div>",
                    unsafe_allow_html=True,
                )
            with c2:
                st.markdown(
                    f"<div class='metric-card'><div class='metric-label'>TFs Agreeing</div>"
                    f"<div class='metric-value'>{max(long_c, short_c)}/{total_valid}</div></div>",
                    unsafe_allow_html=True,
                )
            with c3:
                st.markdown(
                    f"<div class='metric-card'><div class='metric-label'>Avg Confidence</div>"
                    f"<div class='metric-value'>{avg_conf:.1f}%</div></div>",
                    unsafe_allow_html=True,
                )
            st.markdown(
                f"<div class='mtf-kpi-grid'>"
                f"<div class='mtf-kpi'><div class='mtf-kpi-label'>Raw Confluence</div><div class='mtf-kpi-value'>{confluence_pct:.0f}%</div>"
                f"<span class='mtf-badge' style='color:{_status(confluence_pct, 70, 55)[1]}; border-color:{_status(confluence_pct, 70, 55)[1]};'><span style='color:{_status(confluence_pct, 70, 55)[1]};'>&#9679;</span>{_status(confluence_pct, 70, 55)[0]}</span></div>"
                f"<div class='mtf-kpi'><div class='mtf-kpi-label'>Weighted Confluence</div><div class='mtf-kpi-value'>{weighted_confluence:.0f}%</div>"
                f"<span class='mtf-badge' style='color:{_status(weighted_confluence, 72, 58)[1]}; border-color:{_status(weighted_confluence, 72, 58)[1]};'><span style='color:{_status(weighted_confluence, 72, 58)[1]};'>&#9679;</span>{_status(weighted_confluence, 72, 58)[0]}</span></div>"
                f"<div class='mtf-kpi'><div class='mtf-kpi-label'>WAIT Count</div><div class='mtf-kpi-value'>{wait_c}</div>"
                f"<span class='mtf-badge' style='color:{_status(wait_c, 1, 2, lower_better=True)[1]}; border-color:{_status(wait_c, 1, 2, lower_better=True)[1]};'><span style='color:{_status(wait_c, 1, 2, lower_better=True)[1]};'>&#9679;</span>{_status(wait_c, 1, 2, lower_better=True)[0]}</span></div>"
                f"<div class='mtf-kpi'><div class='mtf-kpi-label'>Data Coverage</div><div class='mtf-kpi-value'>{total_valid}/5</div>"
                f"<span class='mtf-badge' style='color:{_status(total_valid, 5, 4)[1]}; border-color:{_status(total_valid, 5, 4)[1]};'><span style='color:{_status(total_valid, 5, 4)[1]};'>&#9679;</span>{_status(total_valid, 5, 4)[0]}</span></div>"
                f"</div>",
                unsafe_allow_html=True,
            )

            st.markdown(
                f"<details style='margin-bottom:0.7rem;'>"
                f"<summary style='color:{ACCENT}; cursor:pointer;'>How to read quickly (?)</summary>"
                f"<div style='color:{TEXT_MUTED}; font-size:0.85rem; line-height:1.7; margin-top:0.5rem;'>"
                f"<b>1.</b> Prioritize setups with <b>Weighted Confluence >= 70%</b>.<br>"
                f"<b>2.</b> If 1h + 4h + 1d align, setup quality is usually higher.<br>"
                f"<b>3.</b> High WAIT count means regime uncertainty; reduce risk or stand aside.<br>"
                f"<b>4.</b> Use this tab for alignment only; confirm entries in Spot/Position tabs."
                f"</div></details>",
                unsafe_allow_html=True,
            )

            # Entry quality badge (A/B/C) from weighted confluence, confidence, and uncertainty.
            if weighted_confluence >= 80 and avg_conf >= 70 and wait_c <= 1:
                quality_grade = "A"
                quality_color = POSITIVE
                quality_text = "High-quality alignment: strong structure across higher timeframes."
            elif weighted_confluence >= 65 and avg_conf >= 60 and wait_c <= 2:
                quality_grade = "B"
                quality_color = WARNING
                quality_text = "Usable setup with moderate alignment. Keep tighter risk controls."
            else:
                quality_grade = "C"
                quality_color = NEGATIVE
                quality_text = "Weak/uncertain alignment. Prefer waiting for cleaner confirmation."

            st.markdown(
                f"<div class='panel-box' style='padding:14px 16px; margin-bottom:10px;'>"
                f"<b style='color:{ACCENT};'>Entry Quality</b> "
                f"<span style='display:inline-block; margin-left:8px; padding:2px 10px; border-radius:999px; "
                f"background:rgba(0,0,0,0.35); border:1px solid {quality_color}; color:{quality_color}; "
                f"font-weight:700;'>Grade {quality_grade}</span>"
                f"<div style='color:{TEXT_MUTED}; font-size:0.86rem; margin-top:8px;'>{quality_text}</div>"
                f"</div>",
                unsafe_allow_html=True,
            )

            # Detail table
            df_mtf = pd.DataFrame(rows).drop(columns=["Weight"])
            df_mtf["Signal Icon"] = df_mtf["Signal"].map({
                "LONG": "↗",
                "SHORT": "↘",
                "WAIT": "→",
                "NO DATA": "•",
            }).fillna("•")

            def _confidence_status(row: pd.Series) -> str:
                if row["Signal"] == "NO DATA":
                    return "• N/A"
                x = float(row["Confidence"])
                if x >= 70:
                    return "▲ Strong"
                if x >= 55:
                    return "■ Medium"
                return "▼ Weak"

            def _adx_status(row: pd.Series) -> str:
                if row["Signal"] == "NO DATA":
                    return "• N/A"
                x = float(row["ADX"])
                if x >= 25:
                    return "▲ Strong Trend"
                if x >= 18:
                    return "■ Moderate"
                return "▼ Weak Trend"

            df_mtf["Confidence Status"] = df_mtf.apply(_confidence_status, axis=1)
            df_mtf["ADX Status"] = df_mtf.apply(_adx_status, axis=1)
            df_mtf["Signal Status"] = df_mtf["Signal"].map(
                {"LONG": "▲ Bullish", "SHORT": "▼ Bearish", "WAIT": "■ Neutral", "NO DATA": "• N/A"}
            ).fillna("• N/A")

            def _style_status(v: str) -> str:
                if "Strong" in v or "Bullish" in v:
                    return f"color:{POSITIVE}; font-weight:700;"
                if "Medium" in v or "Moderate" in v or "Neutral" in v:
                    return f"color:{WARNING}; font-weight:700;"
                if "N/A" in v:
                    return f"color:{TEXT_MUTED}; font-weight:600;"
                return f"color:{NEGATIVE}; font-weight:700;"

            def _style_signal_icon(v: str) -> str:
                if v == "↗":
                    return f"color:{POSITIVE}; font-weight:700;"
                if v == "↘":
                    return f"color:{NEGATIVE}; font-weight:700;"
                if v == "→":
                    return f"color:{WARNING}; font-weight:700;"
                return f"color:{TEXT_MUTED}; font-weight:700;"

            st.markdown(
                f"<div style='color:{TEXT_MUTED}; font-size:0.83rem; margin:2px 0 8px 0; line-height:1.6;'>"
                f"<b style='color:{ACCENT};'>Table Legend:</b> "
                f"<span style='color:{POSITIVE};'>▲ Strong/Bullish</span> "
                f"<span style='color:{WARNING}; margin-left:10px;'>■ Medium/Neutral</span> "
                f"<span style='color:{NEGATIVE}; margin-left:10px;'>▼ Weak/Bearish</span> "
                f"<span style='color:{TEXT_MUTED}; margin-left:10px;'>• N/A</span>"
                f"</div>",
                unsafe_allow_html=True,
            )
            table_cols = [
                "Timeframe", "Signal Icon", "Signal", "Signal Status",
                "Confidence", "Confidence Status", "ADX", "ADX Status",
                "SuperTrend", "Ichimoku", "VWAP",
            ]

            styled_mtf = (
                df_mtf[table_cols].style
                .map(style_signal, subset=["Signal"])
                .map(_style_signal_icon, subset=["Signal Icon"])
                .map(_style_status, subset=["Signal Status", "Confidence Status", "ADX Status"])
            )
            st.dataframe(styled_mtf, width="stretch")

            export_df = pd.DataFrame(rows).copy()
            export_df["Raw Confluence %"] = round(confluence_pct, 2)
            export_df["Weighted Confluence %"] = round(weighted_confluence, 2)
            export_df["Dominant"] = weighted_dominant
            export_df["Entry Grade"] = quality_grade
            export_csv = export_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                "Export Multi-TF Results (CSV)",
                data=export_csv,
                file_name=f"{coin.replace('/', '_')}_multitf_{timeframes[0]}_{timeframes[-1]}.csv",
                mime="text/csv",
            )

            # Recommendation
            if weighted_confluence >= 80 and weighted_dominant != "NEUTRAL":
                st.success(
                    f"Strong alignment: weighted confluence {int(weighted_confluence)}% on {weighted_dominant}. "
                    f"Higher-timeframe agreement is supportive."
                )
            elif weighted_confluence >= 60 and weighted_dominant != "NEUTRAL":
                st.info(
                    f"Moderate alignment: weighted confluence {int(weighted_confluence)}% on {weighted_dominant}. "
                    f"Use tighter risk control."
                )
            else:
                st.warning("Weak alignment. Timeframes are mixed or uncertain — wait for cleaner structure.")
