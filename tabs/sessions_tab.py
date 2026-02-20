from ui.ctx import get_ctx

import numpy as np
import pandas as pd
import plotly.graph_objs as go


def render(ctx: dict) -> None:
    st = get_ctx(ctx, "st")
    ACCENT = get_ctx(ctx, "ACCENT")
    TEXT_MUTED = get_ctx(ctx, "TEXT_MUTED")
    WARNING = get_ctx(ctx, "WARNING")
    POSITIVE = get_ctx(ctx, "POSITIVE")
    NEGATIVE = get_ctx(ctx, "NEGATIVE")
    _normalize_coin_input = get_ctx(ctx, "_normalize_coin_input")
    _validate_coin_symbol = get_ctx(ctx, "_validate_coin_symbol")
    fetch_ohlcv = get_ctx(ctx, "fetch_ohlcv")
    EXCHANGE = get_ctx(ctx, "EXCHANGE")
    readable_market_cap = get_ctx(ctx, "readable_market_cap")
    _tip = get_ctx(ctx, "_tip")
    """Session-based analysis: Asian, European, US sessions."""
    st.markdown(
        f"""
        <style>
        .sess-kpi-grid {{
            display:grid;
            grid-template-columns:repeat(4,minmax(0,1fr));
            gap:10px;
            margin:10px 0 12px 0;
        }}
        .sess-kpi {{
            border:1px solid rgba(0,212,255,0.16);
            border-radius:12px;
            padding:12px 14px;
            background:linear-gradient(140deg, rgba(0,0,0,0.72), rgba(10,18,30,0.88));
        }}
        .sess-kpi-label {{
            color:{TEXT_MUTED};
            font-size:0.70rem;
            text-transform:uppercase;
            letter-spacing:0.8px;
        }}
        .sess-kpi-value {{
            color:{ACCENT};
            font-size:1.2rem;
            font-weight:700;
            margin-top:4px;
        }}
        .sess-badge {{
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
    st.markdown(f"<h2 style='color:{ACCENT};'>Session Analysis</h2>", unsafe_allow_html=True)
    st.markdown(
        f"<div class='panel-box'>"
        f"<b style='color:{ACCENT}; font-size:1rem;'>What does this tab show?</b>"
        f"<p style='color:{TEXT_MUTED}; font-size:0.9rem; margin-top:6px; line-height:1.6;'>"
        f"Compares market behavior across 3 UTC sessions using 1h candles: Asian (00-08), European (08-16), US (16-00). "
        f"It measures {_tip('Liquidity', 'How much size is traded. Higher liquidity usually means tighter spreads and cleaner execution.')}, "
        f"{_tip('Volatility', 'How wide candles are. Very high volatility can improve opportunity but increases risk and slippage.')}, "
        f"and session drift to help you decide when conditions are most tradable."
        f"</p></div>",
        unsafe_allow_html=True,
    )
    c1, c2 = st.columns(2)
    with c1:
        coin_s = _normalize_coin_input(st.text_input("Coin (e.g. BTC, ETH, TAO)", value="BTC", key="session_coin_input"))
    with c2:
        lookback = st.selectbox("Lookback Candles (1h)", [240, 360, 500], index=2, key="session_lookback")

    if st.button("Analyse Sessions", type="primary"):
        _val_err = _validate_coin_symbol(coin_s)
        if _val_err:
            st.error(_val_err)
            return
        with st.spinner("Fetching hourly data for session analysis..."):
            df = fetch_ohlcv(coin_s, "1h", limit=int(lookback))
            if df is None:
                st.error(
                    f"**{coin_s}** could not be found on **{EXCHANGE.id.title()}** or CoinGecko. "
                    f"Please check the symbol and try again."
                )
                return
            if len(df) < 48:
                st.error(
                    f"Only {len(df)} hourly candles available for **{coin_s}** (need at least 48). "
                    f"This coin may have limited history."
                )
                return
            if "timestamp" not in df.columns:
                st.error("Data source returned candles without timestamp. Session analysis cannot run.")
                return

            df = df.copy()
            # Force UTC interpretation to keep session buckets consistent across user locales.
            df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
            df = df.dropna(subset=["timestamp", "open", "high", "low", "close", "volume"]).sort_values("timestamp")
            if len(df) < 48:
                st.error("Not enough clean hourly candles after data cleanup.")
                return
            df["hour"] = df["timestamp"].dt.hour
            df["range_pct"] = ((df["high"] - df["low"]) / df["close"].replace(0, np.nan)) * 100.0
            df["return_pct"] = ((df["close"] - df["open"]) / df["open"].replace(0, np.nan)) * 100.0
            df["abs_return_pct"] = df["return_pct"].abs()

            def _session(h: int) -> str:
                if 0 <= h < 8:
                    return "Asian (00-08 UTC)"
                elif 8 <= h < 16:
                    return "European (08-16 UTC)"
                else:
                    return "US (16-00 UTC)"

            df["session"] = df["hour"].apply(_session)
            session_order = ["Asian (00-08 UTC)", "European (08-16 UTC)", "US (16-00 UTC)"]

            grouped = df.groupby("session").agg(
                avg_volume=("volume", "mean"),
                total_volume=("volume", "sum"),
                avg_range=("range_pct", "mean"),
                avg_return=("return_pct", "mean"),
                avg_abs_return=("abs_return_pct", "mean"),
                candle_count=("close", "count"),
            ).reindex(session_order)
            grouped = grouped.fillna(0.0)

            vol_min, vol_max = grouped["avg_volume"].min(), grouped["avg_volume"].max()
            absret_min, absret_max = grouped["avg_abs_return"].min(), grouped["avg_abs_return"].max()
            range_median = float(grouped["avg_range"].median()) if len(grouped) else 0.0
            grouped["volume_norm"] = (
                (grouped["avg_volume"] - vol_min) / (vol_max - vol_min + 1e-9)
            )
            grouped["absret_norm"] = (
                (grouped["avg_abs_return"] - absret_min) / (absret_max - absret_min + 1e-9)
            )
            grouped["range_fit"] = (
                1.0 - (grouped["avg_range"] - range_median).abs() / (abs(range_median) + 1e-9)
            ).clip(lower=0.0, upper=1.0)
            # Tradeability score favors liquidity + directional movement with moderate volatility.
            grouped["session_score"] = (
                100.0 * (0.50 * grouped["volume_norm"] + 0.30 * grouped["range_fit"] + 0.20 * grouped["absret_norm"])
            ).round(1)

            def _score_status(score: float) -> tuple[str, str]:
                if score >= 70:
                    return "Healthy", POSITIVE
                if score >= 50:
                    return "Watch", WARNING
                return "Risky", NEGATIVE

            def _vol_status(vn: float) -> str:
                if vn >= 0.66:
                    return "● Healthy"
                if vn >= 0.33:
                    return "● Watch"
                return "● Risky"

            def _range_status(rf: float) -> str:
                if rf >= 0.66:
                    return "● Healthy"
                if rf >= 0.33:
                    return "● Watch"
                return "● Risky"

            # Session summary cards
            session_colors = [WARNING, POSITIVE, NEGATIVE]
            cols = st.columns(3)
            for idx, (sess, row) in enumerate(grouped.iterrows()):
                s_label, s_color = _score_status(float(row["session_score"]))
                with cols[idx]:
                    st.markdown(
                        f"<div class='metric-card'>"
                        f"<div class='metric-label'>{sess}</div>"
                        f"<div class='metric-value' style='font-size:1.2rem;'>"
                        f"Vol: {readable_market_cap(int(row['total_volume']))}</div>"
                        f"<div style='color:{TEXT_MUTED};font-size:0.85rem;'>"
                        f"Avg Range: {row['avg_range']:.3f}% | Avg Return: {row['avg_return']:+.3f}% | Score: {row['session_score']:.0f}"
                        f"</div>"
                        f"<span class='sess-badge' style='color:{s_color}; border-color:{s_color};'>"
                        f"<span style='color:{s_color};'>&#9679;</span>{s_label}</span>"
                        f"</div>",
                        unsafe_allow_html=True,
                    )

            best_score = grouped["session_score"].idxmax()
            best_liquidity = grouped["avg_volume"].idxmax()
            hottest = grouped["avg_range"].idxmax()
            data_coverage = int(grouped["candle_count"].sum())
            st.markdown(
                f"<div class='sess-kpi-grid'>"
                f"<div class='sess-kpi'><div class='sess-kpi-label'>Best Session (Score)</div><div class='sess-kpi-value'>{best_score}</div></div>"
                f"<div class='sess-kpi'><div class='sess-kpi-label'>Top Liquidity</div><div class='sess-kpi-value'>{best_liquidity}</div></div>"
                f"<div class='sess-kpi'><div class='sess-kpi-label'>Highest Volatility</div><div class='sess-kpi-value'>{hottest}</div></div>"
                f"<div class='sess-kpi'><div class='sess-kpi-label'>Candle Coverage</div><div class='sess-kpi-value'>{data_coverage}</div></div>"
                f"</div>",
                unsafe_allow_html=True,
            )

            st.markdown(
                f"<details style='margin-bottom:0.7rem;'>"
                f"<summary style='color:{ACCENT}; cursor:pointer;'>How to read quickly (?)</summary>"
                f"<div style='color:{TEXT_MUTED}; font-size:0.85rem; line-height:1.7; margin-top:0.5rem;'>"
                f"<b>1.</b> Start from <b>Session Score</b>; it blends liquidity + manageable volatility + movement quality.<br>"
                f"<b>2.</b> For execution quality, prioritize high <b>Avg Volume</b> sessions.<br>"
                f"<b>3.</b> If <b>Avg Range</b> is extreme, reduce leverage and widen stop assumptions.<br>"
                f"<b>4.</b> Use this as timing filter; entry direction still comes from Spot/Position tabs."
                f"</div></details>",
                unsafe_allow_html=True,
            )
            st.markdown(
                f"<div class='panel-box' style='margin-bottom:0.7rem;'>"
                f"<b style='color:{ACCENT};'>Metric Guide</b>"
                f"<div style='color:{TEXT_MUTED}; font-size:0.84rem; line-height:1.7; margin-top:6px;'>"
                f"<b>Avg Range (%)</b>: Average candle high-low width as percent of price. "
                f"Higher = more intrabar volatility and larger stop distance needed.<br>"
                f"<b>Avg Return (%)</b>: Average candle close-open drift. "
                f"Positive means upward bias in that session, negative means downward bias.<br>"
                f"<b>Avg |Return| (%)</b>: Average absolute candle move size. "
                f"Higher means faster tape and more directional movement opportunity.<br>"
                f"<b>Session Score (0-100)</b>: Composite tradeability score "
                f"(liquidity 50% + volatility-fit 30% + movement-quality 20%). "
                f"Higher generally means cleaner execution conditions."
                f"</div></div>",
                unsafe_allow_html=True,
            )

            fig_vol = go.Figure()
            for idx, sess in enumerate(session_order):
                if sess in grouped.index:
                    fig_vol.add_trace(go.Bar(
                        x=[sess], y=[grouped.loc[sess, "avg_volume"]],
                        name=sess, marker_color=session_colors[idx],
                    ))
            fig_vol.update_layout(
                title="Average Hourly Volume by Session",
                height=300, template="plotly_dark",
                margin=dict(l=20, r=20, t=40, b=30),
                showlegend=False,
            )
            st.plotly_chart(fig_vol, width="stretch")

            fig_range_ret = go.Figure()
            fig_range_ret.add_trace(
                go.Bar(
                    x=session_order,
                    y=[grouped.loc[s, "avg_range"] for s in session_order],
                    name="Avg Range (%)",
                    marker_color=[WARNING, POSITIVE, NEGATIVE],
                )
            )
            fig_range_ret.add_trace(
                go.Scatter(
                    x=session_order,
                    y=[grouped.loc[s, "avg_abs_return"] for s in session_order],
                    name="Avg |Return| (%)",
                    mode="lines+markers",
                    line=dict(color=ACCENT, width=2),
                )
            )
            fig_range_ret.update_layout(
                title="Volatility Profile by Session",
                height=320, template="plotly_dark",
                margin=dict(l=20, r=20, t=40, b=30),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
            )
            st.plotly_chart(fig_range_ret, width="stretch")

            summary_df = grouped.reset_index().rename(
                columns={
                    "session": "Session",
                    "avg_volume": "Avg Volume",
                    "total_volume": "Total Volume",
                    "avg_range": "Avg Range (%)",
                    "avg_return": "Avg Return (%)",
                    "avg_abs_return": "Avg |Return| (%)",
                    "candle_count": "Candles",
                    "session_score": "Session Score",
                }
            )
            summary_df["Volume Status"] = summary_df["volume_norm"].apply(_vol_status)
            summary_df["Volatility Status"] = summary_df["range_fit"].apply(_range_status)
            summary_df["Overall Status"] = summary_df["Session Score"].apply(
                lambda x: "● Healthy" if x >= 70 else ("● Watch" if x >= 50 else "● Risky")
            )
            summary_df["Score Icon"] = summary_df["Session Score"].apply(
                lambda x: "▲" if x >= 70 else ("■" if x >= 50 else "▼")
            )
            summary_df["Return Icon"] = summary_df["Avg Return (%)"].apply(
                lambda x: "↗" if x > 0 else ("↘" if x < 0 else "→")
            )
            summary_df["Total Volume"] = summary_df["Total Volume"].apply(lambda x: readable_market_cap(int(x)))
            st.markdown(
                f"<div style='color:{TEXT_MUTED}; font-size:0.83rem; margin:2px 0 8px 0; line-height:1.6;'>"
                f"<b style='color:{ACCENT};'>Status Legend:</b> "
                f"<span style='color:{POSITIVE};'>&#9679; Healthy</span> "
                f"<span style='color:{WARNING}; margin-left:10px;'>&#9679; Watch</span> "
                f"<span style='color:{NEGATIVE}; margin-left:10px;'>&#9679; Risky</span>"
                f"</div>",
                unsafe_allow_html=True,
            )
            st.markdown(f"<b style='color:{ACCENT};'>Session Summary Table</b>", unsafe_allow_html=True)

            def _style_status(v: str) -> str:
                if "Healthy" in v:
                    return f"color:{POSITIVE}; font-weight:700;"
                if "Watch" in v:
                    return f"color:{WARNING}; font-weight:700;"
                if "Risky" in v:
                    return f"color:{NEGATIVE}; font-weight:700;"
                return ""

            def _style_score_icon(v: str) -> str:
                if v == "▲":
                    return f"color:{POSITIVE}; font-weight:700;"
                if v == "■":
                    return f"color:{WARNING}; font-weight:700;"
                return f"color:{NEGATIVE}; font-weight:700;"

            def _style_return_icon(v: str) -> str:
                if v == "↗":
                    return f"color:{POSITIVE}; font-weight:700;"
                if v == "↘":
                    return f"color:{NEGATIVE}; font-weight:700;"
                return f"color:{WARNING}; font-weight:700;"

            st.dataframe(
                summary_df[
                    [
                        "Session", "Score Icon", "Session Score", "Avg Volume", "Total Volume",
                        "Avg Range (%)", "Return Icon", "Avg Return (%)", "Avg |Return| (%)", "Candles",
                        "Volume Status", "Volatility Status", "Overall Status",
                    ]
                ].style.format(
                    {
                        "Session Score": "{:.1f}",
                        "Avg Volume": "{:,.2f}",
                        "Avg Range (%)": "{:.3f}",
                        "Avg Return (%)": "{:+.3f}",
                        "Avg |Return| (%)": "{:.3f}",
                    }
                ).map(_style_status, subset=["Volume Status", "Volatility Status", "Overall Status"])
                .map(_style_score_icon, subset=["Score Icon"])
                .map(_style_return_icon, subset=["Return Icon"]),
                width="stretch",
                hide_index=True,
            )

            # Best session recommendation
            st.info(
                f"Best current window by composite score: **{best_score}**. "
                f"Highest liquidity: **{best_liquidity}**. "
                f"Most volatile: **{hottest}**. "
                f"Use high-liquidity + controlled-volatility sessions for cleaner entries."
            )
