from ui.ctx import get_ctx

import pandas as pd
from core.multitf import TF_SEQUENCE, TF_WEIGHTS, compute_multitf_alignment
from core.signal_contract import strength_from_bias, strength_bucket
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
    direction_key = get_ctx(ctx, "direction_key")
    direction_label = get_ctx(ctx, "direction_label")

    st.markdown(
        f"""
        <style>
        .mtf-grid {{
            display:grid;
            grid-template-columns:repeat(4,minmax(0,1fr));
            gap:12px;
            margin:12px 0 14px 0;
        }}
        .mtf-card {{
            border:1px solid rgba(0,212,255,0.16);
            border-radius:14px;
            padding:14px 16px;
            background:linear-gradient(140deg, rgba(0,0,0,0.76), rgba(8,18,30,0.9));
        }}
        .mtf-label {{
            color:{TEXT_MUTED};
            font-size:0.72rem;
            text-transform:uppercase;
            letter-spacing:0.9px;
        }}
        .mtf-value {{
            font-size:1.28rem;
            font-weight:800;
            margin-top:6px;
        }}
        .mtf-sub {{
            color:{TEXT_MUTED};
            font-size:0.84rem;
            margin-top:6px;
            line-height:1.55;
        }}
        .mtf-pill {{
            display:inline-flex;
            align-items:center;
            gap:7px;
            padding:3px 10px;
            border-radius:999px;
            border:1px solid rgba(255,255,255,0.16);
            background:rgba(0,0,0,0.28);
            font-size:0.74rem;
            font-weight:700;
            margin-right:8px;
        }}
        @media (max-width: 1100px) {{
            .mtf-grid {{
                grid-template-columns:repeat(2,minmax(0,1fr));
            }}
        }}
        @media (max-width: 720px) {{
            .mtf-grid {{
                grid-template-columns:minmax(0,1fr);
            }}
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )

    def _plain_trend_label(value: str) -> str:
        trend = str(value or "").strip().lower()
        if trend == "bullish":
            return "Bullish"
        if trend == "bearish":
            return "Bearish"
        if trend == "neutral":
            return "Neutral"
        return ""

    def _plain_vwap_label(value: str) -> str:
        label = str(value or "").strip().lower()
        if label == "above":
            return "Above"
        if label == "below":
            return "Below"
        if label == "at":
            return "At"
        return "Neutral" if not label else str(value)

    def _adx_label(adx: float) -> str:
        try:
            value = float(adx)
        except Exception:
            return ""
        if value < 20:
            return "Weak"
        if value < 25:
            return "Starting"
        if value < 50:
            return "Strong"
        if value < 75:
            return "Very Strong"
        return "Extreme"

    def _strength_label(value: float) -> str:
        bucket = strength_bucket(float(value))
        if bucket == "STRONG":
            read = "Strong"
        elif bucket == "GOOD":
            read = "Good"
        elif bucket == "MIXED":
            read = "Mixed"
        else:
            read = "Weak"
        return f"{float(value):.0f}% ({read.upper()})"

    def _style_indicator(value: str) -> str:
        text = str(value or "")
        upper = text.upper()
        if any(token in upper for token in {"UPSIDE", "BULLISH", "ABOVE", "STRONG", "VERY STRONG", "EXTREME"}):
            return f"color:{POSITIVE}; font-weight:700;"
        if any(token in upper for token in {"DOWNSIDE", "BEARISH", "BELOW", "WEAK"}):
            return f"color:{NEGATIVE}; font-weight:700;"
        return f"color:{WARNING}; font-weight:700;"

    def _style_layer(value: str) -> str:
        layer = str(value or "").strip().lower()
        if layer == "structure":
            return f"color:{ACCENT}; font-weight:700;"
        if layer == "timing":
            return f"color:{TEXT_MUTED}; font-weight:700;"
        return ""

    def _insight(metrics: dict) -> tuple[str, str, str]:
        dominant = metrics["dominant_bias"]
        higher = metrics["higher_tf_bias"]
        tactical = metrics["tactical_bias"]
        weighted = metrics["weighted_alignment_pct"]
        if dominant == "NEUTRAL":
            return (
                "Alignment Insight · Neutral Structure",
                "Higher and tactical timeframes are not producing a clean directional edge. Treat this as a structure check, not an execution prompt.",
                WARNING,
            )
        if higher == dominant and tactical == dominant and weighted >= 65:
            return (
                f"Alignment Insight · Broad {direction_label(dominant)} Agreement",
                f"Higher timeframes and timing layers are pointing {direction_label(dominant).lower()}. This supports the same-side read already seen in Market / Spot / Position.",
                POSITIVE if dominant == "UPSIDE" else NEGATIVE,
            )
        if higher == dominant and tactical != dominant:
            return (
                f"Alignment Insight · Higher-TF {direction_label(dominant)} Lead",
                f"The structural layer still leans {direction_label(dominant).lower()}, but 5m/15m timing is mixed. Wait for tactical alignment before trusting execution.",
                WARNING,
            )
        return (
            "Alignment Insight · Cross-TF Mismatch",
            "Lower and higher timeframes are pulling in different directions. Treat the coin as tactically noisy until the structure simplifies.",
            WARNING,
        )

    def _build_rows(coin: str) -> list[dict]:
        rows: list[dict] = []
        for timeframe in TF_SEQUENCE:
            df = fetch_ohlcv(coin, timeframe, limit=200)
            if df is None or len(df) < 55:
                rows.append(
                    {
                        "timeframe": timeframe,
                        "Timeframe": timeframe,
                        "Layer": "Timing" if timeframe in {"5m", "15m"} else "Structure",
                        "direction": "",
                        "Direction": "No Data",
                        "strength": 0.0,
                        "Strength": "",
                        "ADX": "",
                        "SuperTrend": "",
                        "Ichimoku": "",
                        "VWAP": "",
                        "Weight": TF_WEIGHTS.get(timeframe, 1.0),
                    }
                )
                continue
            # Multi-TF is a closed-candle diagnostic. Always drop the latest open candle when possible.
            df_eval = df.iloc[:-1].copy() if len(df) > 1 else df.copy()
            if df_eval is None or len(df_eval) < 55:
                rows.append(
                    {
                        "timeframe": timeframe,
                        "Timeframe": timeframe,
                        "Layer": "Timing" if timeframe in {"5m", "15m"} else "Structure",
                        "direction": "",
                        "Direction": "No Data",
                        "strength": 0.0,
                        "Strength": "",
                        "ADX": "",
                        "SuperTrend": "",
                        "Ichimoku": "",
                        "VWAP": "",
                        "Weight": TF_WEIGHTS.get(timeframe, 1.0),
                    }
                )
                continue
            analysis = analyse(df_eval)
            direction = direction_key(analysis.signal)
            strength = round(float(strength_from_bias(float(analysis.bias))), 1)
            rows.append(
                {
                    "timeframe": timeframe,
                    "Timeframe": timeframe,
                    "Layer": "Timing" if timeframe in {"5m", "15m"} else "Structure",
                    "direction": direction,
                    "Direction": direction_label(direction),
                    "strength": strength,
                    "Strength": _strength_label(strength),
                    "ADX": _adx_label(analysis.adx),
                    "SuperTrend": _plain_trend_label(analysis.supertrend),
                    "Ichimoku": _plain_trend_label(analysis.ichimoku),
                    "VWAP": _plain_vwap_label(analysis.vwap),
                    "Weight": TF_WEIGHTS.get(timeframe, 1.0),
                }
            )
        return rows

    st.markdown(f"<h2 style='color:{ACCENT};'>Multi-Timeframe Alignment</h2>", unsafe_allow_html=True)
    st.markdown(
        f"<div class='panel-box'>"
        f"<b style='color:{ACCENT}; font-size:1rem;'>What does this tab show?</b>"
        f"<p style='color:{TEXT_MUTED}; font-size:0.92rem; margin-top:6px; line-height:1.65;'>"
        f"Single-coin timeframe alignment view across 5m, 15m, 1h, 4h, and 1d. "
        f"Use it to check whether short-term timing is aligned with higher-timeframe structure. "
        f"Higher timeframes carry more weight because they usually define the stronger structural regime. "
        f"This tab does not create a separate trade decision; it validates the same read you already see in Market, Spot, and Position."
        f"</p></div>",
        unsafe_allow_html=True,
    )

    coin = _normalize_coin_input(st.text_input("Coin (e.g. BTC, ETH, TAO)", value="BTC", key="mtf_coin_input"))
    run = st.button("Run Alignment Check", type="primary")
    state_key = "mtf_alignment_payload"

    if run:
        validation_error = _validate_coin_symbol(coin)
        if validation_error:
            st.error(validation_error)
            return
        with st.spinner("Checking alignment across all timeframes..."):
            live_rows = _build_rows(coin)
            valid_live = [row for row in live_rows if row["direction"] in {"UPSIDE", "DOWNSIDE", "NEUTRAL"}]
            rows = live_rows
            from_cache = False
            cache_ts = None
            if not valid_live:
                rows, from_cache, cache_ts = live_or_snapshot(
                    st,
                    f"mtf_rows::{coin}",
                    live_rows,
                    max_age_sec=900,
                    current_sig=(coin,),
                )
            else:
                live_or_snapshot(
                    st,
                    f"mtf_rows::{coin}",
                    live_rows,
                    max_age_sec=900,
                    current_sig=(coin,),
                )
            metrics = compute_multitf_alignment(rows)
            st.session_state[state_key] = {
                "coin": coin,
                "rows": rows,
                "metrics": metrics,
                "from_cache": from_cache,
                "cache_ts": cache_ts,
            }

    payload = st.session_state.get(state_key)
    if not payload:
        return

    if payload.get("coin") != coin:
        st.info(f"Showing latest alignment snapshot for {payload.get('coin')}. Press Run Alignment Check to refresh {coin}.")

    rows = payload["rows"]
    metrics = payload["metrics"]

    if payload.get("from_cache"):
        st.warning(f"Live multi-timeframe data was unavailable. Showing cached snapshot from {payload.get('cache_ts')}.")
    elif metrics["coverage_count"] < metrics["coverage_total"]:
        st.warning(
            f"Only {metrics['coverage_count']}/{metrics['coverage_total']} timeframes produced usable closed-candle analysis. "
            f"Treat this as a partial alignment read."
        )
    title, body, title_color = _insight(metrics)
    st.markdown(
        f"<div class='panel-box' style='border-left:4px solid {title_color};'>"
        f"<div style='color:{title_color}; font-weight:800; font-size:1.02rem;'>{title}</div>"
        f"<div style='color:{TEXT_MUTED}; font-size:0.9rem; margin-top:6px; line-height:1.6;'>{body}</div>"
        f"<div style='color:{TEXT_MUTED}; font-size:0.82rem; margin-top:7px;'>Higher timeframes are weighted more heavily because they usually define the stronger structural regime.</div>"
        f"</div>",
        unsafe_allow_html=True,
    )

    dominant_color = POSITIVE if metrics["dominant_bias"] == "UPSIDE" else (NEGATIVE if metrics["dominant_bias"] == "DOWNSIDE" else WARNING)
    higher_color = POSITIVE if metrics["higher_tf_bias"] == "UPSIDE" else (NEGATIVE if metrics["higher_tf_bias"] == "DOWNSIDE" else WARNING)
    coverage_color = POSITIVE if metrics["coverage_count"] >= 5 else (WARNING if metrics["coverage_count"] >= 3 else NEGATIVE)

    st.markdown(
        f"<div class='mtf-grid'>"
        f"<div class='mtf-card'><div class='mtf-label'>Dominant Bias</div>"
        f"<div class='mtf-value' style='color:{dominant_color};'>{direction_label(metrics['dominant_bias'])}</div>"
        f"<div class='mtf-sub'>Bias across all valid timeframes.</div></div>"
        f"<div class='mtf-card'><div class='mtf-label'>Weighted Alignment</div>"
        f"<div class='mtf-value' style='color:{dominant_color};'>{metrics['weighted_alignment_pct']:.0f}%</div>"
        f"<div class='mtf-sub'>{metrics['alignment_read']} read after higher-TF weights are applied.</div></div>"
        f"<div class='mtf-card'><div class='mtf-label'>Higher-TF Bias</div>"
        f"<div class='mtf-value' style='color:{higher_color};'>{direction_label(metrics['higher_tf_bias'])}</div>"
        f"<div class='mtf-sub'>{metrics['higher_tf_alignment_pct']:.0f}% alignment on 1h / 4h / 1d.</div></div>"
        f"<div class='mtf-card'><div class='mtf-label'>Coverage</div>"
        f"<div class='mtf-value' style='color:{coverage_color};'>{metrics['coverage_count']}/{metrics['coverage_total']}</div>"
        f"<div class='mtf-sub'>{metrics['coverage_read']} data coverage across the five timeframes.</div></div>"
        f"</div>",
        unsafe_allow_html=True,
    )

    st.markdown(
        f"<div class='panel-box' style='padding:12px 16px;'>"
        f"<span class='mtf-pill' style='color:{higher_color}; border-color:{higher_color};'>Higher-TF · {direction_label(metrics['higher_tf_bias'])} · {metrics['higher_tf_read']}</span>"
        f"<span class='mtf-pill' style='color:{dominant_color}; border-color:{dominant_color};'>Timing Layer · {direction_label(metrics['tactical_bias'])} · {metrics['tactical_read']}</span>"
        f"<span class='mtf-pill' style='color:{ACCENT}; border-color:rgba(0,212,255,0.3);'>Avg Strength · {metrics['avg_strength']:.0f}%</span>"
        f"<span class='mtf-pill' style='color:{WARNING}; border-color:{WARNING};'>Neutral TFs · {metrics['neutral_count']}</span>"
        f"</div>",
        unsafe_allow_html=True,
    )

    st.markdown(
        f"<details style='margin:8px 0 12px 0;'>"
        f"<summary style='color:{ACCENT}; cursor:pointer;'>How to read quickly (?)</summary>"
        f"<div style='color:{TEXT_MUTED}; font-size:0.86rem; line-height:1.72; margin-top:0.55rem;'>"
        f"<b>1.</b> Start with <b>Higher-TF Bias</b>. That is the structural layer and usually matters most.<br>"
        f"<b>2.</b> Then read <b>Weighted Alignment</b>. Neutral timeframes dilute this score on purpose, so high values really mean broad agreement.<br>"
        f"<b>3.</b> A directional bias only prints when weighted agreement is broad enough. If alignment stays below 60%, the tab keeps the bias neutral on purpose.<br>"
        f"<b>4.</b> Use the <b>Timing Layer</b> as confirmation. If 5m/15m disagree with 1h/4h/1d, the structure may still be valid but entry timing is noisy.<br>"
        f"<b>5.</b> If coverage is partial, trust the result less. Missing timeframes reduce confidence even when the visible alignment looks clean.<br>"
        f"<b>6.</b> Use this tab to confirm structure, not to replace Market / Spot / Position decisions."
        f"</div></details>",
        unsafe_allow_html=True,
    )
    st.markdown(
        f"<details style='margin:0 0 10px 0;'>"
        f"<summary style='color:{ACCENT}; cursor:pointer;'>Column Guide (?)</summary>"
        f"<div style='color:{TEXT_MUTED}; font-size:0.86rem; line-height:1.72; margin-top:0.55rem;'>"
        f"<b>Timeframe</b>: the candle interval being checked.<br>"
        f"<b>Layer</b>: 5m/15m are timing; 1h/4h/1d are structural.<br>"
        f"<b>Direction</b>: Upside / Downside / Neutral technical bias for that timeframe.<br>"
        f"<b>Strength</b>: direction-agnostic signal power from the same analysis core used across the dashboard.<br>"
        f"<b>ADX</b>: trend participation quality for that timeframe.<br>"
        f"<b>SuperTrend / Ichimoku / VWAP</b>: supporting structure checks for that same candle context."
        f"</div></details>",
        unsafe_allow_html=True,
    )

    export_rows = []
    for row in rows:
        export_rows.append(
            {
                "Timeframe": row["Timeframe"],
                "Layer": row["Layer"],
                "Direction": row["Direction"],
                "Strength": row["Strength"],
                "ADX": row["ADX"],
                "SuperTrend": row["SuperTrend"],
                "Ichimoku": row["Ichimoku"],
                "VWAP": row["VWAP"],
            }
        )
    df_rows = pd.DataFrame(export_rows)
    styled = (
        df_rows.style
        .map(_style_indicator, subset=["Direction", "SuperTrend", "Ichimoku", "VWAP"])
        .map(_style_indicator, subset=["Strength", "ADX"])
        .map(_style_layer, subset=["Layer"])
    )
    st.dataframe(styled, width="stretch")

    export_df = df_rows.copy()
    export_df["Dominant Bias"] = direction_label(metrics["dominant_bias"])
    export_df["Weighted Alignment %"] = round(metrics["weighted_alignment_pct"], 2)
    export_df["Higher-TF Bias"] = direction_label(metrics["higher_tf_bias"])
    export_df["Higher-TF Alignment %"] = round(metrics["higher_tf_alignment_pct"], 2)
    export_csv = export_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Export Multi-TF Alignment (CSV)",
        data=export_csv,
        file_name=f"{payload['coin'].replace('/', '_')}_multitf_alignment.csv",
        mime="text/csv",
    )
