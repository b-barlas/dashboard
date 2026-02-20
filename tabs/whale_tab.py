from ui.ctx import get_ctx

from datetime import datetime, timezone

import numpy as np
import pandas as pd


def render(ctx: dict) -> None:
    st = get_ctx(ctx, "st")
    ACCENT = get_ctx(ctx, "ACCENT")
    TEXT_MUTED = get_ctx(ctx, "TEXT_MUTED")
    TEXT_LIGHT = get_ctx(ctx, "TEXT_LIGHT")
    POSITIVE = get_ctx(ctx, "POSITIVE")
    NEGATIVE = get_ctx(ctx, "NEGATIVE")
    WARNING = get_ctx(ctx, "WARNING")
    NEON_BLUE = get_ctx(ctx, "NEON_BLUE")
    GOLD = get_ctx(ctx, "GOLD")
    _tip = get_ctx(ctx, "_tip")
    fetch_trending_coins = get_ctx(ctx, "fetch_trending_coins")
    fetch_top_gainers_losers = get_ctx(ctx, "fetch_top_gainers_losers")
    fetch_ohlcv = get_ctx(ctx, "fetch_ohlcv")
    get_top_volume_usdt_symbols = get_ctx(ctx, "get_top_volume_usdt_symbols")
    """Whale tracking and momentum signals."""
    st.markdown(
        f"""
        <style>
        .whale-kpi-grid {{
            display:grid;
            grid-template-columns:repeat(4,minmax(0,1fr));
            gap:10px;
            margin:8px 0 18px 0;
        }}
        .whale-kpi {{
            background:linear-gradient(145deg, rgba(0,0,0,0.72), rgba(10,18,30,0.92));
            border:1px solid rgba(0,212,255,0.18);
            border-radius:12px;
            padding:12px 14px;
            box-shadow:0 8px 24px rgba(0,0,0,0.22);
        }}
        .whale-kpi-label {{
            color:{TEXT_MUTED};
            text-transform:uppercase;
            font-size:0.70rem;
            letter-spacing:0.7px;
        }}
        .whale-kpi-value {{
            color:{ACCENT};
            font-size:1.25rem;
            font-weight:700;
            margin-top:4px;
        }}
        .whale-list-card {{
            border:1px solid rgba(0,212,255,0.14);
            border-radius:12px;
            padding:8px 10px;
            margin:4px 0;
            background:linear-gradient(120deg, rgba(255,255,255,0.01), rgba(0,212,255,0.03));
        }}
        .whale-trend-row {{
            display:grid;
            grid-template-columns:56px minmax(80px, 120px) 1fr minmax(90px, 120px);
            align-items:center;
            column-gap:10px;
        }}
        .whale-momentum-row {{
            display:flex;
            align-items:center;
            justify-content:space-between;
            gap:10px;
            padding:5px 9px;
            border-radius:8px;
            margin:3px 0;
        }}
        .whale-badge {{
            display:inline-block;
            padding:2px 8px;
            border-radius:999px;
            font-size:0.72rem;
            font-weight:700;
            letter-spacing:0.3px;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        f"<h2 style='color:{ACCENT};'>Whale Tracker</h2>",
        unsafe_allow_html=True,
    )
    st.markdown(
        f"<div class='panel-box'>"
        f"<b style='color:{ACCENT}; font-size:1rem;'>What does this tab show?</b>"
        f"<p style='color:{TEXT_MUTED}; font-size:0.9rem; margin-top:6px; line-height:1.6;'>"
        f"Tracks market attention and liquidity anomalies using public market data. "
        f"This tab is a <b>whale proxy</b> (not on-chain whale wallet tracking). "
        f"It combines search trends, 24h movers, and abnormal volume detection.</p>"
        f"<p style='color:{TEXT_MUTED}; font-size:0.85rem; margin-top:6px; line-height:1.6;'>"
        f"<b>1. Trending Coins</b> — CoinGecko search trend leaders.<br>"
        f"<b>2. Top Gainers / Losers</b> — 24h momentum leaders/laggards from CoinGecko markets feed.<br>"
        f"<b>3. Volume Anomaly Scanner</b> — dynamic high-volume universe from exchange-available pairs.<br>"
        f"Scanner uses two checks: "
        f"{_tip('Volume Ratio', 'Latest candle volume divided by previous 20-candle average.')} and "
        f"{_tip('Volume Z-Score', 'How many standard deviations the latest volume is above recent mean.')} "
        f"to reduce false positives from simple ratio-only spikes.</p>"
        f"</div>",
        unsafe_allow_html=True,
    )

    with st.spinner("Fetching whale & momentum data..."):
        trending_new = fetch_trending_coins()
        gainers_new, losers_new = fetch_top_gainers_losers(15)

    # Keep last successful payload to prevent empty UI on transient API failures.
    if trending_new:
        st.session_state["whale_trending_cache"] = trending_new
        st.session_state["whale_trending_cache_ts"] = datetime.now(timezone.utc)
    if gainers_new:
        st.session_state["whale_gainers_cache"] = gainers_new
        st.session_state["whale_gainers_cache_ts"] = datetime.now(timezone.utc)
    if losers_new:
        st.session_state["whale_losers_cache"] = losers_new
        st.session_state["whale_losers_cache_ts"] = datetime.now(timezone.utc)

    trending = trending_new or st.session_state.get("whale_trending_cache", [])
    gainers = gainers_new or st.session_state.get("whale_gainers_cache", [])
    losers = losers_new or st.session_state.get("whale_losers_cache", [])
    used_trending_cache = (not trending_new) and bool(trending)
    used_gainers_cache = (not gainers_new) and bool(gainers)
    used_losers_cache = (not losers_new) and bool(losers)

    trend_ts = st.session_state.get("whale_trending_cache_ts")
    gainers_ts = st.session_state.get("whale_gainers_cache_ts")
    losers_ts = st.session_state.get("whale_losers_cache_ts")

    def _fmt_ts(ts_obj) -> str:
        if not ts_obj:
            return "N/A"
        if isinstance(ts_obj, datetime):
            return ts_obj.astimezone(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
        return "N/A"

    top_gainers = (gainers or [])[:10]
    top_losers = (losers or [])[:10]
    def _pct24(c: dict) -> float:
        v = c.get("price_change_percentage_24h")
        if v is None:
            v = c.get("price_change_percentage_24h_in_currency")
        return float(v or 0.0)

    avg_g = np.mean([_pct24(c) for c in top_gainers]) if top_gainers else 0.0
    avg_l = np.mean([_pct24(c) for c in top_losers]) if top_losers else 0.0
    avg_g_text = f"{avg_g:+.2f}%"
    avg_l_text = f"{avg_l:+.2f}%"
    st.markdown(
        f"""
        <div class='whale-kpi-grid'>
          <div class='whale-kpi'><div class='whale-kpi-label'>Trending Count</div><div class='whale-kpi-value'>{len(trending or [])}</div></div>
          <div class='whale-kpi'><div class='whale-kpi-label'>Top Gainer Avg (24h)</div><div class='whale-kpi-value' style='color:{POSITIVE};'>{avg_g_text}</div></div>
          <div class='whale-kpi'><div class='whale-kpi-label'>Top Loser Avg (24h)</div><div class='whale-kpi-value' style='color:{NEGATIVE};'>{avg_l_text}</div></div>
          <div class='whale-kpi'><div class='whale-kpi-label'>Data Source</div><div class='whale-kpi-value' style='font-size:0.95rem;'>CoinGecko + Exchange OHLCV</div></div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Trending coins
    st.markdown(f"<div class='god-header'><b style='color:{NEON_BLUE};'>Trending Coins</b></div>",
                unsafe_allow_html=True)
    if used_trending_cache:
        st.caption(f"Live trending fetch unavailable. Showing cached snapshot from {_fmt_ts(trend_ts)}.")
    if trending:
        for i, coin in enumerate(trending[:10]):
            rank_color = GOLD if i < 3 else NEON_BLUE if i < 6 else TEXT_MUTED
            st.markdown(
                f"<div class='whale-list-card whale-trend-row'>"
                f"<span style='color:{rank_color}; font-weight:700; font-size:1.1rem;'>#{i+1}</span>"
                f"<span style='color:{ACCENT}; font-weight:700;'>{coin['symbol']}</span>"
                f"<span style='color:{TEXT_MUTED}; font-size:0.82rem; overflow:hidden; text-overflow:ellipsis; white-space:nowrap;'>{coin['name']}</span>"
                f"<span style='color:{TEXT_MUTED}; font-size:0.8rem; text-align:right;'>Rank #{coin['market_cap_rank'] or 'N/A'}</span></div>",
                unsafe_allow_html=True,
            )
    else:
        st.info("Trending data unavailable.")

    # Gainers / losers
    st.markdown(f"<div class='god-header'><b style='color:{NEON_BLUE};'>Market Momentum (24h)</b></div>",
                unsafe_allow_html=True)
    if used_gainers_cache or used_losers_cache:
        st.caption(
            f"Live momentum fetch unavailable. Showing cached snapshot "
            f"(gainers: {_fmt_ts(gainers_ts)}, losers: {_fmt_ts(losers_ts)})."
        )
    col_g, col_l = st.columns(2)
    with col_g:
        st.markdown(f"<b style='color:{POSITIVE};'>TOP GAINERS</b>", unsafe_allow_html=True)
        if gainers:
            for c in gainers[:12]:
                change = _pct24(c)
                symbol = (c.get('symbol', '') or '').upper()
                price = c.get('current_price', 0)
                st.markdown(
                    f"<div class='whale-momentum-row' style='border-left:2px solid {POSITIVE}; background:rgba(0,255,136,0.05);'>"
                    f"<span style='color:{ACCENT}; font-weight:600;'>{symbol} <span style='color:{TEXT_MUTED}; font-size:0.75rem;'>${price:,.4f}</span></span>"
                    f"<span style='color:{POSITIVE}; font-weight:700;'>+{change:.2f}%</span></div>",
                    unsafe_allow_html=True,
                )
        else:
            st.info("Top gainers data temporarily unavailable.")
    with col_l:
        st.markdown(f"<b style='color:{NEGATIVE};'>TOP LOSERS</b>", unsafe_allow_html=True)
        if losers:
            for c in losers[:12]:
                change = _pct24(c)
                symbol = (c.get('symbol', '') or '').upper()
                price = c.get('current_price', 0)
                st.markdown(
                    f"<div class='whale-momentum-row' style='border-left:2px solid {NEGATIVE}; background:rgba(255,51,102,0.05);'>"
                    f"<span style='color:{ACCENT}; font-weight:600;'>{symbol} <span style='color:{TEXT_MUTED}; font-size:0.75rem;'>${price:,.4f}</span></span>"
                    f"<span style='color:{NEGATIVE}; font-weight:700;'>{change:.2f}%</span></div>",
                    unsafe_allow_html=True,
                )
        else:
            st.info("Top losers data temporarily unavailable.")

    # Volume anomaly scanner
    st.markdown(f"<div class='god-header'><b style='color:{NEON_BLUE};'>Volume Anomaly Scanner</b></div>",
                unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        scan_tf = st.selectbox("Scan Timeframe", ['5m', '15m', '1h', '4h'], index=2, key="whale_scan_tf")
    with c2:
        universe_n = st.slider("Universe Size", min_value=10, max_value=80, value=30, step=5, key="whale_universe_n")
    with c3:
        ratio_th = st.slider("Min Volume Ratio", min_value=1.2, max_value=3.0, value=1.5, step=0.1, key="whale_ratio_th")
    with c4:
        z_th = st.slider("Min Z-Score", min_value=1.0, max_value=4.0, value=2.0, step=0.1, key="whale_z_th")

    st.markdown(
        f"<p style='color:{TEXT_MUTED}; font-size:0.82rem; margin-top:4px;'>"
        f"Trigger rule: <b>Ratio ≥ {ratio_th:.1f}</b> OR <b>Z-Score ≥ {z_th:.1f}</b>. "
        f"Higher thresholds = fewer but cleaner alerts."
        f"</p>",
        unsafe_allow_html=True,
    )

    if st.button("Run Volume Scan", key="whale_scan"):
        with st.spinner("Scanning..."):
            symbols, _raw = get_top_volume_usdt_symbols(max(universe_n, 30))
            symbols = symbols[:universe_n]
            if not symbols:
                symbols = ["BTC/USDT", "ETH/USDT", "SOL/USDT", "BNB/USDT", "XRP/USDT",
                           "ADA/USDT", "DOGE/USDT", "AVAX/USDT", "DOT/USDT", "LINK/USDT"]
            surges = []
            lookback = 20
            bars_per_24h = {"5m": 288, "15m": 96, "1h": 24, "4h": 6}.get(scan_tf, 24)
            for sym in symbols:
                try:
                    df_s = fetch_ohlcv(sym, scan_tf, limit=max(lookback + 5, bars_per_24h + 2, 80))
                    if df_s is None or len(df_s) <= lookback + 1:
                        continue

                    vol_window = df_s['volume'].iloc[-(lookback + 1):-1]
                    avg_vol = float(vol_window.mean())
                    std_vol = float(vol_window.std(ddof=0))
                    last_vol = float(df_s['volume'].iloc[-1])
                    if avg_vol <= 0:
                        continue

                    ratio = last_vol / avg_vol
                    z_score = (last_vol - avg_vol) / std_vol if std_vol > 1e-9 else 0.0
                    if ratio < ratio_th and z_score < z_th:
                        continue

                    ret_1 = ((df_s['close'].iloc[-1] / df_s['close'].iloc[-2]) - 1) * 100
                    if len(df_s) > bars_per_24h:
                        ret_24h = ((df_s['close'].iloc[-1] / df_s['close'].iloc[-(bars_per_24h + 1)]) - 1) * 100
                    else:
                        ret_24h = np.nan

                    if ratio >= 2.5 or z_score >= 3.0:
                        level = "EXTREME"
                    elif ratio >= 1.8 or z_score >= 2.0:
                        level = "HIGH"
                    else:
                        level = "MODERATE"

                    score = (min(ratio, 4.0) / 4.0) * 0.6 + (min(max(z_score, 0.0), 4.0) / 4.0) * 0.4
                    surges.append({
                        "Symbol": sym.split('/')[0],
                        "Level": level,
                        "Vol Ratio": ratio,
                        "Z-Score": z_score,
                        "Last Vol": last_vol,
                        "Avg20 Vol": avg_vol,
                        "1-Candle %": ret_1,
                        "24h %": ret_24h,
                        "Score": score,
                    })
                except Exception:
                    continue
            if surges:
                surges = sorted(surges, key=lambda x: x["Score"], reverse=True)
                st.markdown(
                    f"<details style='margin-bottom:0.5rem;'>"
                    f"<summary style='color:{ACCENT}; cursor:pointer;'>Column Guide (?)</summary>"
                    f"<div style='color:{TEXT_MUTED}; font-size:0.84rem; line-height:1.7; margin-top:0.5rem;'>"
                    f"<b>Vol Ratio</b>: latest volume / prior 20-candle average.<br>"
                    f"<b>Z-Score</b>: standardized volume shock size.<br>"
                    f"<b>1-Candle %</b>: last candle return.<br>"
                    f"<b>24h %</b>: rolling 24h return approximation by selected timeframe.<br>"
                    f"<b>Score</b>: blended anomaly strength from ratio and z-score.</div></details>",
                    unsafe_allow_html=True,
                )
                st.markdown(
                    f"<div class='panel-box' style='padding:14px 16px; margin-top:8px;'>"
                    f"<b style='color:{ACCENT};'>How to read this quickly</b>"
                    f"<ul style='color:{TEXT_MUTED}; margin:8px 0 0 18px; line-height:1.7;'>"
                    f"<li>Prioritize rows with <b>Level = EXTREME</b> and <b>Score >= 0.70</b>.</li>"
                    f"<li>Prefer anomalies where <b>1-Candle %</b> and <b>24h %</b> point the same direction.</li>"
                    f"<li>If Ratio is high but Z-Score is weak, it may be noise from already-elevated baseline volume.</li>"
                    f"<li>Use Spot/Position tabs for confirmation before acting.</li>"
                    f"</ul></div>",
                    unsafe_allow_html=True,
                )

                df_surges = pd.DataFrame(surges)
                df_show = df_surges.copy()
                df_show["Ratio Status"] = df_show["Vol Ratio"].apply(
                    lambda x: "Extreme" if x >= 2.5 else ("Elevated" if x >= 1.8 else "Mild")
                )
                df_show["Z-Status"] = df_show["Z-Score"].apply(
                    lambda x: "Extreme" if x >= 3.0 else ("Elevated" if x >= 2.0 else "Mild")
                )
                df_show["Vol Ratio"] = df_show["Vol Ratio"].map(lambda x: f"{x:.2f}x")
                df_show["Z-Score"] = df_show["Z-Score"].map(lambda x: f"{x:.2f}")
                df_show["Last Vol"] = df_show["Last Vol"].map(lambda x: f"{x:,.0f}")
                df_show["Avg20 Vol"] = df_show["Avg20 Vol"].map(lambda x: f"{x:,.0f}")
                df_show["1-Candle %"] = df_show["1-Candle %"].map(lambda x: f"{x:+.2f}%")
                df_show["24h %"] = df_show["24h %"].map(lambda x: f"{x:+.2f}%" if pd.notna(x) else "N/A")
                df_show["Score"] = df_show["Score"].map(lambda x: f"{x:.2f}")
                def _pct_style(v: str) -> str:
                    s = str(v)
                    if s == "N/A":
                        return f"color:{TEXT_MUTED};"
                    return f"color:{POSITIVE}; font-weight:700;" if s.startswith("+") else f"color:{NEGATIVE}; font-weight:700;"

                def _status_style(v: str) -> str:
                    s = str(v)
                    if "Extreme" in s:
                        return f"color:{POSITIVE}; font-weight:700;"
                    if "Elevated" in s:
                        return f"color:{WARNING}; font-weight:700;"
                    if "Mild" in s:
                        return f"color:{NEON_BLUE}; font-weight:700;"
                    return f"color:{TEXT_MUTED}; font-weight:600;"

                st.markdown(
                    f"<div style='color:{TEXT_MUTED}; font-size:0.83rem; margin:2px 0 8px 0; line-height:1.6;'>"
                    f"<b style='color:{ACCENT};'>Table Legend:</b> "
                    f"<span style='color:{POSITIVE};'>EXTREME = strongest anomaly</span> "
                    f"<span style='color:{WARNING}; margin-left:10px;'>HIGH = notable</span> "
                    f"<span style='color:{NEON_BLUE}; margin-left:10px;'>MODERATE = mild</span> "
                    f"<span style='color:{POSITIVE}; margin-left:10px;'>Ratio/Z: Extreme</span> "
                    f"<span style='color:{WARNING}; margin-left:10px;'>Elevated</span> "
                    f"<span style='color:{NEON_BLUE}; margin-left:10px;'>Mild</span>"
                    f"</div>",
                    unsafe_allow_html=True,
                )

                df_view = df_show[
                    [
                        "Symbol", "Level", "Vol Ratio", "Ratio Status",
                        "Z-Score", "Z-Status", "Score",
                        "1-Candle %", "24h %", "Last Vol", "Avg20 Vol",
                    ]
                ].rename(
                    columns={
                        "Level": "Level (?)",
                        "Vol Ratio": "Vol Ratio (?)",
                        "Ratio Status": "Ratio Status (?)",
                        "Z-Score": "Z-Score (?)",
                        "Z-Status": "Z-Status (?)",
                        "Score": "Score (?)",
                        "1-Candle %": "1-Candle % (?)",
                        "24h %": "24h % (?)",
                    }
                )

                styled = (
                    df_view.style
                    .map(
                        lambda v: (
                            f"background:rgba(0,255,136,0.16); color:{POSITIVE}; font-weight:700; border-radius:8px;"
                            if "EXTREME" in str(v)
                            else (
                                f"background:rgba(255,209,102,0.16); color:{WARNING}; font-weight:700; border-radius:8px;"
                                if "HIGH" in str(v)
                                else f"background:rgba(0,212,255,0.10); color:{NEON_BLUE}; font-weight:700; border-radius:8px;"
                            )
                        ),
                        subset=["Level (?)"],
                    )
                    .map(_pct_style, subset=["1-Candle % (?)", "24h % (?)"])
                    .map(_status_style, subset=["Ratio Status (?)", "Z-Status (?)"])
                )
                st.dataframe(styled, width="stretch")
            else:
                st.info("No significant volume anomalies detected for selected thresholds.")
