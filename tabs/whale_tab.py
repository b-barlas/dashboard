from ui.ctx import get_ctx
from ui.primitives import render_help_details, render_insight_card, render_kpi_grid, render_page_header
from ui.snapshot_cache import live_or_snapshot

import numpy as np
import pandas as pd


def render(ctx: dict) -> None:
    st = get_ctx(ctx, "st")
    ACCENT = get_ctx(ctx, "ACCENT")
    TEXT_MUTED = get_ctx(ctx, "TEXT_MUTED")
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
    render_page_header(
        st,
        title="Whale Tracker",
        intro_html=(
            "Tracks market attention and liquidity anomalies using public market data. "
            "This tab is a <b>whale proxy</b> (not on-chain whale wallet tracking). "
            "It combines search trends, 24h movers, and abnormal volume detection."
            "<br><br><b>1. Trending Coins</b> — CoinGecko search trend leaders.<br>"
            "<b>2. Top Gainers / Losers</b> — 24h momentum leaders/laggards from CoinGecko markets feed.<br>"
            "<b>3. Volume Anomaly Scanner</b> — dynamic high-volume universe from exchange-available pairs.<br>"
            "Scanner uses two checks: "
            f"{_tip('Volume Ratio', 'Latest candle volume divided by previous 20-candle average.')} and "
            f"{_tip('Volume Z-Score', 'How many standard deviations the latest volume is above recent mean.')} "
            "to reduce false positives from simple ratio-only spikes."
        ),
    )
    st.markdown(
        """
        <style>
        .whale-list-card {
            border:1px solid rgba(0,212,255,0.14);
            border-radius:12px;
            padding:8px 10px;
            margin:4px 0;
            background:linear-gradient(120deg, rgba(255,255,255,0.01), rgba(0,212,255,0.03));
        }
        .whale-trend-row {
            display:grid;
            grid-template-columns:56px minmax(80px, 120px) 1fr minmax(90px, 120px);
            align-items:center;
            column-gap:10px;
        }
        .whale-momentum-row {
            display:flex;
            align-items:center;
            justify-content:space-between;
            gap:10px;
            padding:5px 9px;
            border-radius:8px;
            margin:3px 0;
        }
        .whale-badge {
            display:inline-block;
            padding:2px 8px;
            border-radius:999px;
            font-size:0.72rem;
            font-weight:700;
            letter-spacing:0.3px;
        }
        @media (max-width: 900px) {
            .whale-trend-row {
                grid-template-columns:44px minmax(64px, 100px) 1fr;
            }
            .whale-trend-row span:last-child {
                display:none;
            }
        }
        @media (max-width: 640px) {
            .whale-momentum-row {
                flex-direction:column;
                align-items:flex-start;
                gap:4px;
            }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    with st.spinner("Fetching whale & momentum data..."):
        trending_new = fetch_trending_coins()
        gainers_new, losers_new = fetch_top_gainers_losers(15)

    trending, used_trending_cache, trend_ts = live_or_snapshot(
        st, "whale_trending", trending_new, max_age_sec=900, current_sig=("trending",)
    )
    gainers, used_gainers_cache, gainers_ts = live_or_snapshot(
        st, "whale_gainers", gainers_new, max_age_sec=900, current_sig=("gainers",)
    )
    losers, used_losers_cache, losers_ts = live_or_snapshot(
        st, "whale_losers", losers_new, max_age_sec=900, current_sig=("losers",)
    )

    def _fmt_ts(ts_obj) -> str:
        return str(ts_obj) if ts_obj else "N/A"

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
    render_kpi_grid(
        st,
        items=[
            {"label": "Trending Count", "value": len(trending or [])},
            {"label": "Top Gainer Avg (24h)", "value": avg_g_text, "value_color": POSITIVE},
            {"label": "Top Loser Avg (24h)", "value": avg_l_text, "value_color": NEGATIVE},
            {"label": "Data Source", "value": "CoinGecko + Exchange OHLCV"},
        ],
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
        if top_gainers:
            for c in top_gainers:
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
        if top_losers:
            for c in top_losers:
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

    tf_ratio_base = {"5m": 1.7, "15m": 1.6, "1h": 1.5, "4h": 1.4}.get(scan_tf, 1.5)
    tf_z_base = {"5m": 2.2, "15m": 2.0, "1h": 1.8, "4h": 1.6}.get(scan_tf, 1.8)
    ratio_gate = max(float(ratio_th), float(tf_ratio_base))
    z_gate = max(float(z_th), float(tf_z_base))
    extreme_ratio_gate = ratio_gate + 0.70
    extreme_z_gate = z_gate + 0.90

    st.markdown(
        f"<p style='color:{TEXT_MUTED}; font-size:0.82rem; margin-top:4px;'>"
        f"Adaptive trigger ({scan_tf}): <b>Ratio ≥ {ratio_gate:.2f}</b> OR <b>Z-Score ≥ {z_gate:.2f}</b>. "
        f"<b>EXTREME</b> requires both: Ratio ≥ {extreme_ratio_gate:.2f} and Z-Score ≥ {extreme_z_gate:.2f}."
        f"</p>",
        unsafe_allow_html=True,
    )

    scan_sig = (scan_tf, int(universe_n), round(ratio_gate, 3), round(z_gate, 3))
    state_key = "whale_scan_state"

    def _ratio_status(v: float) -> str:
        if v >= extreme_ratio_gate:
            return "Extreme"
        if v >= ratio_gate:
            return "Elevated"
        return "Supporting"

    def _z_status(v: float) -> str:
        if v >= extreme_z_gate:
            return "Extreme"
        if v >= z_gate:
            return "Elevated"
        return "Supporting"

    def _pct_style(v: str) -> str:
        s = str(v)
        if s == "N/A":
            return f"color:{TEXT_MUTED};"
        return f"color:{POSITIVE}; font-weight:700;" if s.startswith("+") else f"color:{NEGATIVE}; font-weight:700;"

    def _status_style(v: str) -> str:
        s = str(v)
        if "Extreme" in s:
            return f"color:{WARNING}; font-weight:700;"
        if "Elevated" in s:
            return f"color:{ACCENT}; font-weight:700;"
        if "Supporting" in s:
            return f"color:{TEXT_MUTED}; font-weight:700;"
        return f"color:{TEXT_MUTED}; font-weight:600;"

    if st.button("Run Volume Scan", type="primary", key="whale_scan"):
        with st.spinner("Scanning..."):
            symbols, _raw = get_top_volume_usdt_symbols(max(universe_n, 30))
            symbols = symbols[:universe_n]
            if not symbols:
                symbols = [
                    "BTC/USDT", "ETH/USDT", "SOL/USDT", "BNB/USDT", "XRP/USDT",
                    "ADA/USDT", "DOGE/USDT", "AVAX/USDT", "DOT/USDT", "LINK/USDT",
                ]
            surges = []
            lookback = 20
            bars_per_24h = {"5m": 288, "15m": 96, "1h": 24, "4h": 6}.get(scan_tf, 24)
            diag = {
                "symbols": len(symbols),
                "with_data": 0,
                "no_data": 0,
                "errors": 0,
                "gate_reject": 0,
                "passed": 0,
            }
            for sym in symbols:
                try:
                    df_s = fetch_ohlcv(sym, scan_tf, limit=max(lookback + 5, bars_per_24h + 2, 80))
                    if df_s is None or len(df_s) <= lookback + 1:
                        diag["no_data"] += 1
                        continue
                    # Evaluate anomalies on closed candles to avoid partial-candle volume noise.
                    df_eval = df_s.iloc[:-1].copy() if len(df_s) > (lookback + 2) else df_s.copy()
                    if df_eval is None or len(df_eval) <= lookback + 1:
                        diag["no_data"] += 1
                        continue
                    diag["with_data"] += 1

                    vol_window = df_eval["volume"].iloc[-(lookback + 1):-1]
                    avg_vol = float(vol_window.mean())
                    std_vol = float(vol_window.std(ddof=0))
                    last_vol = float(df_eval["volume"].iloc[-1])
                    if avg_vol <= 0:
                        diag["gate_reject"] += 1
                        continue

                    ratio = last_vol / avg_vol
                    z_score = (last_vol - avg_vol) / std_vol if std_vol > 1e-9 else 0.0
                    if ratio < ratio_gate and z_score < z_gate:
                        diag["gate_reject"] += 1
                        continue

                    ret_1 = ((df_eval["close"].iloc[-1] / df_eval["close"].iloc[-2]) - 1) * 100
                    if len(df_eval) > bars_per_24h:
                        ret_24h = ((df_eval["close"].iloc[-1] / df_eval["close"].iloc[-(bars_per_24h + 1)]) - 1) * 100
                    else:
                        ret_24h = np.nan

                    is_extreme = ratio >= extreme_ratio_gate and z_score >= extreme_z_gate
                    is_high = (
                        (ratio >= ratio_gate and z_score >= z_gate)
                        or ratio >= (ratio_gate + 0.35)
                        or z_score >= (z_gate + 0.50)
                    )
                    if is_extreme:
                        level = "EXTREME"
                    elif is_high:
                        level = "HIGH"
                    else:
                        level = "MODERATE"

                    ratio_norm = (ratio - ratio_gate) / max(extreme_ratio_gate - ratio_gate, 1e-9)
                    z_norm = (z_score - z_gate) / max(extreme_z_gate - z_gate, 1e-9)
                    ratio_norm = float(min(max(ratio_norm, 0.0), 1.0))
                    z_norm = float(min(max(z_norm, 0.0), 1.0))
                    base_score = 0.55 * ratio_norm + 0.45 * z_norm
                    if level == "EXTREME":
                        score = 0.85 + 0.15 * base_score
                    elif level == "HIGH":
                        score = 0.60 + 0.25 * base_score
                    else:
                        score = 0.35 + 0.20 * base_score
                    score = float(min(max(score, 0.0), 1.0))
                    diag["passed"] += 1
                    surges.append(
                        {
                            "Symbol": sym.split("/")[0],
                            "Level": level,
                            "Vol Ratio": ratio,
                            "Z-Score": z_score,
                            "Last Vol": last_vol,
                            "Avg20 Vol": avg_vol,
                            "1-Candle %": ret_1,
                            "24h %": ret_24h,
                            "Score": score,
                        }
                    )
                except Exception:
                    diag["errors"] += 1
                    continue

            st.session_state[state_key] = {
                "sig": scan_sig,
                "surges": sorted(surges, key=lambda x: x["Score"], reverse=True),
                "diag": diag,
                "scan_ts": pd.Timestamp.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC"),
            }

    state = st.session_state.get(state_key)
    if not state:
        st.info("Run Volume Scan to build the anomaly table.")
        return
    if state.get("sig") != scan_sig:
        st.info("Settings changed. Run Volume Scan again to refresh anomaly results for the current configuration.")
        return

    surges = state.get("surges", [])
    diag = state.get("diag", {})
    scan_ts = state.get("scan_ts")
    if scan_ts:
        st.caption(f"Last scan: {scan_ts} | symbols={diag.get('symbols', 0)} | with data={diag.get('with_data', 0)}")

    if not surges:
        if diag.get("with_data", 0) == 0:
            st.warning(
                "No scan rows produced because market data could not be fetched for the selected universe/timeframe. "
                "Try another timeframe or rerun shortly."
            )
        else:
            st.info(
                f"No significant volume anomalies detected for current thresholds. "
                f"Evaluated={diag.get('with_data', 0)}, gate rejects={diag.get('gate_reject', 0)}."
            )
        return

    render_help_details(
        st,
        summary="Column Guide (?)",
        body_html=(
            "<b>Vol Ratio</b>: latest volume / prior 20-candle average.<br>"
            "<b>Z-Score</b>: standardized volume shock size.<br>"
            "<b>1-Candle %</b>: last closed candle return.<br>"
            "<b>24h %</b>: rolling 24h return approximation by selected timeframe.<br>"
            "<b>Score</b>: blended anomaly strength from ratio and z-score."
        ),
    )
    render_insight_card(
        st,
        title="How to read this quickly",
        body_html=(
            "<ul style='margin:8px 0 0 18px; line-height:1.7;'>"
            "<li>Prioritize rows with <b>Level = EXTREME</b> and <b>Score >= 0.85</b> (dual-confirmed ratio + z-score).</li>"
            "<li>Prefer anomalies where <b>1-Candle %</b> and <b>24h %</b> point the same direction.</li>"
            "<li>Rows with Ratio/Z status = <b>Supporting</b> passed mainly because the other metric carried the trigger.</li>"
            "<li>Use Spot/Position tabs for execution confirmation before acting.</li>"
            "</ul>"
        ),
        tone="accent",
    )

    df_surges = pd.DataFrame(surges)
    df_show = df_surges.copy()
    df_show["Ratio Status"] = df_show["Vol Ratio"].apply(_ratio_status)
    df_show["Z-Status"] = df_show["Z-Score"].apply(_z_status)
    df_show["Vol Ratio"] = df_show["Vol Ratio"].map(lambda x: f"{x:.2f}x")
    df_show["Z-Score"] = df_show["Z-Score"].map(lambda x: f"{x:.2f}")
    df_show["Last Vol"] = df_show["Last Vol"].map(lambda x: f"{x:,.0f}")
    df_show["Avg20 Vol"] = df_show["Avg20 Vol"].map(lambda x: f"{x:,.0f}")
    df_show["1-Candle %"] = df_show["1-Candle %"].map(lambda x: f"{x:+.2f}%")
    df_show["24h %"] = df_show["24h %"].map(lambda x: f"{x:+.2f}%" if pd.notna(x) else "N/A")
    df_show["Score"] = df_show["Score"].map(lambda x: f"{x:.2f}")

    st.markdown(
        f"<div style='color:{TEXT_MUTED}; font-size:0.83rem; margin:2px 0 8px 0; line-height:1.6;'>"
        f"<b style='color:{ACCENT};'>Table Legend:</b> "
        f"<span style='color:{WARNING};'>EXTREME = strongest attention event (dual-confirmed)</span> "
        f"<span style='color:{ACCENT}; margin-left:10px;'>HIGH = notable anomaly</span> "
        f"<span style='color:{TEXT_MUTED}; margin-left:10px;'>MODERATE = mild anomaly</span> "
        f"<span style='color:{WARNING}; margin-left:10px;'>Ratio/Z: Extreme</span> "
        f"<span style='color:{ACCENT}; margin-left:10px;'>Elevated</span> "
        f"<span style='color:{TEXT_MUTED}; margin-left:10px;'>Supporting</span>"
        f"</div>",
        unsafe_allow_html=True,
    )

    st.markdown(
        f"<div style='color:{TEXT_MUTED}; font-size:0.83rem; margin:2px 0 8px 0; line-height:1.8;'>"
        f"<b style='color:{ACCENT};'>Column Tips:</b> "
        f"{_tip('Level', 'Anomaly class: MODERATE / HIGH / EXTREME. EXTREME requires dual confirmation.')} "
        f"{_tip('Vol Ratio', 'Latest closed-candle volume divided by previous 20-candle average.')} "
        f"{_tip('Z-Score', 'How far latest closed-candle volume is from recent mean in std units.')} "
        f"{_tip('Score', 'Blended anomaly strength (0-1) from ratio + z-score.')} "
        f"{_tip('1-Candle %', 'Latest closed candle return.')} "
        f"{_tip('24h %', 'Approx rolling 24h return at selected timeframe.')}"
        f"</div>",
        unsafe_allow_html=True,
    )

    show_diag_cols = st.checkbox(
        "Show diagnostic columns (Ratio/Z status, raw volumes)",
        value=False,
        key="whale_show_diag_cols",
        help="Enable this if you need deeper diagnostics. Keep OFF for cleaner scanning view.",
    )

    base_cols = ["Symbol", "Level", "Vol Ratio", "Z-Score", "Score", "1-Candle %", "24h %"]
    diag_cols = ["Ratio Status", "Z-Status", "Last Vol", "Avg20 Vol"]
    selected_cols = base_cols + diag_cols if show_diag_cols else base_cols
    df_view = df_show[selected_cols]

    styled = (
        df_view.style
        .map(
            lambda v: (
                f"background:rgba(255,209,102,0.16); color:{WARNING}; font-weight:700; border-radius:8px;"
                if "EXTREME" in str(v)
                else (
                    f"background:rgba(0,212,255,0.14); color:{ACCENT}; font-weight:700; border-radius:8px;"
                    if "HIGH" in str(v)
                    else f"background:rgba(255,255,255,0.06); color:{TEXT_MUTED}; font-weight:700; border-radius:8px;"
                )
            ),
            subset=["Level"],
        )
        .map(_pct_style, subset=["1-Candle %", "24h %"])
    )
    if show_diag_cols:
        styled = styled.map(_status_style, subset=["Ratio Status", "Z-Status"])
    st.dataframe(styled, width="stretch")
