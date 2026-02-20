from ui.ctx import get_ctx

from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
import plotly.graph_objs as go
from core.metric_catalog import (
    AI_LONG_THRESHOLD,
    AI_SHORT_THRESHOLD,
    ai_stability_bucket,
    confidence_bucket,
    direction_from_prob,
)


def render(ctx: dict) -> None:
    st = get_ctx(ctx, "st")
    ACCENT = get_ctx(ctx, "ACCENT")
    TEXT_MUTED = get_ctx(ctx, "TEXT_MUTED")
    CARD_BG = get_ctx(ctx, "CARD_BG")
    POSITIVE = get_ctx(ctx, "POSITIVE")
    NEGATIVE = get_ctx(ctx, "NEGATIVE")
    WARNING = get_ctx(ctx, "WARNING")
    get_market_indices = get_ctx(ctx, "get_market_indices")
    get_fear_greed = get_ctx(ctx, "get_fear_greed")
    get_btc_eth_prices = get_ctx(ctx, "get_btc_eth_prices")
    get_price_change = get_ctx(ctx, "get_price_change")
    _tip = get_ctx(ctx, "_tip")
    get_major_ohlcv_bundle = get_ctx(ctx, "get_major_ohlcv_bundle")
    ml_ensemble_predict = get_ctx(ctx, "ml_ensemble_predict")
    get_top_volume_usdt_symbols = get_ctx(ctx, "get_top_volume_usdt_symbols")
    fetch_ohlcv = get_ctx(ctx, "fetch_ohlcv")
    analyse = get_ctx(ctx, "analyse")
    get_scalping_entry_target = get_ctx(ctx, "get_scalping_entry_target")
    _calc_conviction = get_ctx(ctx, "_calc_conviction")
    signal_plain = get_ctx(ctx, "signal_plain")
    confidence_score_badge = get_ctx(ctx, "confidence_score_badge")
    readable_market_cap = get_ctx(ctx, "readable_market_cap")
    format_delta = get_ctx(ctx, "format_delta")
    format_trend = get_ctx(ctx, "format_trend")
    format_adx = get_ctx(ctx, "format_adx")
    format_stochrsi = get_ctx(ctx, "format_stochrsi")
    style_signal = get_ctx(ctx, "style_signal")
    style_confidence = get_ctx(ctx, "style_confidence")
    style_scalp_opp = get_ctx(ctx, "style_scalp_opp")
    style_delta = get_ctx(ctx, "style_delta")
    _debug = get_ctx(ctx, "_debug")
    """Render the Market Dashboard tab containing top‑level crypto metrics and scanning."""

    # Fetch global market data
    # Unpack market indices.  The function returns BTC/ETH dominance, market caps,
    # 24h change and dominance values for BNB, SOL, ADA and XRP.  We keep the
    # additional dominance values for use in the AI market outlook calculation.
    btc_dom, eth_dom, total_mcap, alt_mcap, mcap_24h_pct, bnb_dom, sol_dom, ada_dom, xrp_dom = get_market_indices()
    fg_value, fg_label = get_fear_greed()
    fg_value = fg_value if fg_value is not None else 0
    btc_price, eth_price = get_btc_eth_prices()
    btc_price = btc_price or 0
    eth_price = eth_price or 0

    # Compute percentage change for market cap
    delta_mcap = mcap_24h_pct

    # Compute price change percentages using ccxt
    btc_change = get_price_change("BTC/USDT")
    eth_change = get_price_change("ETH/USDT")

    # Display headline and subtitle
    st.markdown("<h1 class='title'>Crypto Command Center</h1>", unsafe_allow_html=True)
    st.markdown(
        f"<div class='panel-box'>"
        f"<b style='color:{ACCENT}; font-size:1rem;'>What does this tab show?</b>"
        f"<p style='color:{TEXT_MUTED}; font-size:0.9rem; margin-top:6px; line-height:1.6;'>"
        f"Your market overview dashboard. Shows live BTC/ETH prices, total market cap, "
        f"{_tip('Fear & Greed Index', 'A 0-100 score measuring market sentiment. 0 = Extreme Fear (buy opportunity), 100 = Extreme Greed (sell signal). Based on volatility, volume, social media, and surveys.')} "
        f"and {_tip('BTC Dominance', 'Percentage of total crypto market cap that belongs to Bitcoin. Rising dominance = money flowing into BTC (risk-off). Falling = altcoin season.')}. "
        f"Top coins are dynamically selected by 24h volume from CoinGecko and scored with real-time technical signals.</p>"
        f"</div>",
        unsafe_allow_html=True,
    )

    # Determine which timeframe to use for the market prediction.  We rely on
    # Streamlit session state to persist the selected timeframe from the
    # scanner controls.  On the first render, default to 1h.  This allows
    # the market prediction card to update automatically when the user
    # changes the timeframe in the scanner below.  We fetch BTC/USDT as a
    # proxy for the overall crypto market and compute a prediction using
    # 500 candles of history.
    selected_timeframe = st.session_state.get("market_timeframe", "1h")
    # Top row: Price and market cap metrics.
    m1, m2, m3, m4 = st.columns(4, gap="medium")
    # Bitcoin price
    with m1:
        delta_class = "metric-delta-positive" if (btc_change or 0) >= 0 else "metric-delta-negative"
        delta_text = f"({btc_change:+.2f}%)" if btc_change is not None else ""
        st.markdown(
            f"<div class='metric-card'>"
            f"  <div class='metric-label'>Bitcoin Price</div>"
            f"  <div class='metric-value'>${btc_price:,.2f}</div>"
            f"  <div class='{delta_class}'>{delta_text}</div>"
            f"</div>",
            unsafe_allow_html=True,
        )
    # Ethereum price
    with m2:
        delta_class = "metric-delta-positive" if (eth_change or 0) >= 0 else "metric-delta-negative"
        delta_text = f"({eth_change:+.2f}%)" if eth_change is not None else ""
        st.markdown(
            f"<div class='metric-card'>"
            f"  <div class='metric-label'>Ethereum Price</div>"
            f"  <div class='metric-value'>${eth_price:,.2f}</div>"
            f"  <div class='{delta_class}'>{delta_text}</div>"
            f"</div>",
            unsafe_allow_html=True,
        )
    # Total market cap
    with m3:
        delta_class = "metric-delta-positive" if delta_mcap >= 0 else "metric-delta-negative"
        st.markdown(
            f"<div class='metric-card'>"
            f"  <div class='metric-label'>Total Market Cap</div>"
            f"  <div class='metric-value'>${total_mcap / 1e12:.2f}T</div>"
            f"  <div class='{delta_class}'>({delta_mcap:+.2f}%)</div>"
            f"</div>",
            unsafe_allow_html=True,
        )
    # Fear & Greed index
    with m4:
        sentiment_color = POSITIVE if "Greed" in fg_label else (NEGATIVE if "Fear" in fg_label else WARNING)
        st.markdown(
            f"<div class='metric-card'>"
            f"  <div class='metric-label'>Fear &amp; Greed "
            f"<span title='Crypto Fear &amp; Greed Index (0-100). "
            f"0-25 = Extreme Fear (potential buy zone), 75-100 = Extreme Greed (potential sell zone).' "
            f"style='cursor:help; font-size:0.7rem;'>ℹ️</span></div>"
            f"  <div class='metric-value'>{fg_value}</div>"
            f"  <div style='color:{sentiment_color};font-size:0.9rem;'>{fg_label}</div>"
            f"</div>",
            unsafe_allow_html=True,
        )
    # Second row: Dominance gauges and AI market outlook
    # Compute AI market outlook using a dominance-weighted ML prediction across
    # BTC, ETH and major altcoins (BNB, SOL, ADA, XRP) on the selected timeframe.
    try:
        bundle_behav = get_major_ohlcv_bundle(selected_timeframe, limit=500)
        btc_df_behav = bundle_behav.get('BTC/USDT')
        eth_df_behav = bundle_behav.get('ETH/USDT')
        bnb_df_behav = bundle_behav.get('BNB/USDT')
        sol_df_behav = bundle_behav.get('SOL/USDT')
        ada_df_behav = bundle_behav.get('ADA/USDT')
        xrp_df_behav = bundle_behav.get('XRP/USDT')
        # Initialise probabilities at a neutral value of 0.5.  If data
        # retrieval or training fails for an asset, the neutral prior will
        # prevent it from skewing the combined outlook.
        btc_prob = eth_prob = bnb_prob = sol_prob = ada_prob = xrp_prob = 0.5
        if btc_df_behav is not None and not btc_df_behav.empty:
            btc_prob, _, _ = ml_ensemble_predict(btc_df_behav)
        if eth_df_behav is not None and not eth_df_behav.empty:
            eth_prob, _, _ = ml_ensemble_predict(eth_df_behav)
        if bnb_df_behav is not None and not bnb_df_behav.empty:
            bnb_prob, _, _ = ml_ensemble_predict(bnb_df_behav)
        if sol_df_behav is not None and not sol_df_behav.empty:
            sol_prob, _, _ = ml_ensemble_predict(sol_df_behav)
        if ada_df_behav is not None and not ada_df_behav.empty:
            ada_prob, _, _ = ml_ensemble_predict(ada_df_behav)
        if xrp_df_behav is not None and not xrp_df_behav.empty:
            xrp_prob, _, _ = ml_ensemble_predict(xrp_df_behav)
        # Compute a weighted probability across all assets.  Dominance values
        # reflect each coin's share of the total crypto market.  If the sum of
        # dominances is zero (unlikely), default to 1 to avoid division by zero.
        dom_sum = btc_dom + eth_dom + bnb_dom + sol_dom + ada_dom + xrp_dom
        dom_sum = dom_sum if dom_sum > 0 else 1.0
        behaviour_prob = (
            btc_prob * btc_dom
            + eth_prob * eth_dom
            + bnb_prob * bnb_dom
            + sol_prob * sol_dom
            + ada_prob * ada_dom
            + xrp_prob * xrp_dom
        ) / dom_sum
    except Exception:
        behaviour_prob = 0.5
    # Determine behaviour direction from the combined probability
    behaviour_dir = direction_from_prob(float(behaviour_prob))
    # Map behaviour direction to a label for display and choose colour.  We
    # reuse the POSITIVE/NEGATIVE/WARNING colours defined above.
    if behaviour_dir == "LONG":
        behaviour_label = "Up"
        behaviour_color = POSITIVE
    elif behaviour_dir == "SHORT":
        behaviour_label = "Down"
        behaviour_color = NEGATIVE
    else:
        behaviour_label = "Neutral"
        behaviour_color = WARNING

    g1, g2, g3 = st.columns(3, gap="medium")
    # BTC dominance gauge
    with g1:
        fig_btc = go.Figure(go.Indicator(
            mode="gauge+number",
            value=btc_dom,
            gauge={
                'axis': {'range': [0, 100], 'tickwidth': 1},
                'bar': {'color': ACCENT},
                'bgcolor': CARD_BG,
                'steps': [
                    {'range': [0, AI_SHORT_THRESHOLD * 100], 'color': NEGATIVE},
                    {'range': [AI_SHORT_THRESHOLD * 100, AI_LONG_THRESHOLD * 100], 'color': WARNING},
                    {'range': [AI_LONG_THRESHOLD * 100, 100], 'color': POSITIVE},
                ],
            },
            title={'text': 'BTC Dominance (%)', 'font': {'size': 16, 'color': ACCENT}},
            number={'font': {'color': ACCENT, 'size': 38}},
        ))
        fig_btc.update_layout(
            height=170,
            margin=dict(l=10, r=10, t=40, b=15),
            plot_bgcolor="#000000",
            paper_bgcolor="#000000",
        )
        st.plotly_chart(fig_btc, width="stretch")

    # ETH dominance gauge
    with g2:
        fig_eth = go.Figure(go.Indicator(
            mode="gauge+number",
            value=eth_dom,
            gauge={
                'axis': {'range': [0, 100], 'tickwidth': 1},
                'bar': {'color': ACCENT},
                'bgcolor': CARD_BG,
                'steps': [
                    {'range': [0, 15], 'color': NEGATIVE},
                    {'range': [15, 25], 'color': WARNING},
                    {'range': [25, 100], 'color': POSITIVE},
                ],
            },
            title={'text': 'ETH Dominance (%)', 'font': {'size': 16, 'color': ACCENT}},
            number={'font': {'color': ACCENT, 'size': 38}},
        ))
        fig_eth.update_layout(
            height=170,
            margin=dict(l=10, r=10, t=40, b=15),
            plot_bgcolor="#000000",
            paper_bgcolor="#000000",
        )
        st.plotly_chart(fig_eth, width="stretch")

    # AI market outlook gauge
    with g3:
        fig_behaviour = go.Figure(go.Indicator(
            mode="gauge+number",
            value=int(round(behaviour_prob * 100)),
            gauge={
                'axis': {'range': [0, 100], 'tickwidth': 1},
                'bar': {'color': ACCENT},
                'bgcolor': CARD_BG,
                'steps': [
                    {'range': [0, 40], 'color': NEGATIVE},
                    {'range': [40, 60], 'color': WARNING},
                    {'range': [60, 100], 'color': POSITIVE},
                ],
            },
            title={'text': 'AI Market Outlook (%)', 'font': {'size': 16, 'color': ACCENT}},
            number={'font': {'color': ACCENT, 'size': 38}},
        ))
        fig_behaviour.update_layout(
            height=170,
            margin=dict(l=10, r=10, t=40, b=15),
            plot_bgcolor="#000000",
            paper_bgcolor="#000000",
        )
        st.plotly_chart(fig_behaviour, width="stretch")
        # Direction label + tooltip explanation
        st.markdown(
            f"<div style='text-align:center; margin-top:-12px;'>"
            f"<span style='color:{behaviour_color}; font-size:0.9rem;'>{behaviour_label}</span>"
            f"<span title='Dominance-weighted ML prediction across BTC, ETH, BNB, SOL, ADA, XRP "
            f"on the selected timeframe. Each coin&apos;s prediction is weighted by its market dominance. "
            f"&ge;{int(AI_LONG_THRESHOLD*100)}% = Bullish, "
            f"&le;{int(AI_SHORT_THRESHOLD*100)}% = Bearish, "
            f"{int(AI_SHORT_THRESHOLD*100)}-{int(AI_LONG_THRESHOLD*100)}% = Neutral.' "
            f"style='cursor:help; color:{TEXT_MUTED}; font-size:0.8rem; margin-left:6px;'>ℹ️</span>"
            f"</div>",
            unsafe_allow_html=True,
        )


    # Divider
    st.markdown("\n\n")

    # Top coin scanner controls
    st.markdown(
        f"<h2 style='color:{ACCENT};margin-bottom:0.5rem;'>Coin Signal Scanner</h2>",
        unsafe_allow_html=True,
    )

    controls = st.columns([1.4, 1.4, 1, 1], gap="medium")
    with controls[0]:
        # Persist the selected timeframe in session state so the market
        # prediction card updates when this value changes.  The key ensures
        # the selection is stored under 'market_timeframe'.
        timeframe = st.selectbox(
            "Select timeframe",
            ['5m', '15m', '1h', '4h', '1d'],
            index=2,
            key="market_timeframe"
        )
    with controls[1]:
        signal_filter = st.selectbox("Signal", ['LONG', 'SHORT', 'BOTH'], index=2)
    with controls[2]:
        top_n = st.slider("Top N", min_value=3, max_value=50, value=50)
    with controls[3]:
        refresh_scan = st.button("Refresh Scan", use_container_width=True)
    # Strict scalp mode is always enabled (non-strict path removed).

    def _ai_agree_badge(v: float) -> str:
        b = ai_stability_bucket(float(v))
        if b == "Strong":
            return "🟢 Strong"
        if b == "Medium":
            return "🟡 Medium"
        return "⚪ Weak"

    def _confidence_band(v: float) -> str:
        b = confidence_bucket(float(v))
        if b == "Strong":
            return "🟢 Strong"
        if b == "Good":
            return "🟡 Good"
        if b == "Mixed":
            return "⚪ Mixed"
        return "🔴 Weak"

    def _setup_badge(scalp_dir: str, signal_dir: str, ai_dir: str) -> str:
        if scalp_dir and signal_dir in {"LONG", "SHORT"} and signal_dir == ai_dir == scalp_dir:
            return "🟢 Aligned"
        if scalp_dir and signal_dir in {"LONG", "SHORT"} and signal_dir == scalp_dir and ai_dir == "NEUTRAL":
            return "🟡 Tech-Only"
        if scalp_dir:
            return "⚪ Draft"
        return "🔴 No Setup"

    def _style_setup(v: str) -> str:
        s = str(v)
        if "Aligned" in s:
            return f"color:{POSITIVE}; font-weight:700;"
        if "Tech-Only" in s:
            return f"color:{WARNING}; font-weight:700;"
        if "No Setup" in s:
            return f"color:{NEGATIVE}; font-weight:700;"
        return f"color:{TEXT_MUTED}; font-weight:600;"

    def _style_trend_state(v: str) -> str:
        s = str(v)
        # ADX-specific states first (string comes from format_adx helper).
        if "Very Strong" in s or "Extreme" in s or "(Strong)" in s:
            return f"color:{POSITIVE}; font-weight:700;"
        if "Weak" in s or "Starting" in s:
            return f"color:{WARNING}; font-weight:700;"
        if "Bullish" in s or "Above" in s or "Oversold" in s or "Near Bottom" in s or "Low" in s:
            return f"color:{POSITIVE}; font-weight:700;"
        if "Bearish" in s or "Below" in s or "Overbought" in s or "Near Top" in s or "High" in s:
            return f"color:{NEGATIVE}; font-weight:700;"
        if "Neutral" in s or "Moderate" in s or "Near VWAP" in s or "Starting" in s:
            return f"color:{WARNING}; font-weight:700;"
        return f"color:{TEXT_MUTED};"

    def _fmt_price(v: float) -> str:
        try:
            p = float(v)
        except Exception:
            return ""
        if p >= 1000:
            return f"${p:,.2f}"
        if p >= 1:
            return f"${p:,.4f}"
        if p >= 0.01:
            return f"${p:,.6f}"
        if p >= 0.0001:
            return f"${p:,.8f}"
        return f"${p:,.10f}"

    def _action_decision(
        signal_dir: str,
        confidence: float,
        setup_badge: str,
        conviction_label: str,
        agreement: float,
        adx_val: float,
        has_plan: bool,
    ) -> str:
        if signal_dir not in {"LONG", "SHORT"}:
            return "⛔ SKIP"
        if not has_plan:
            return "⏳ WAIT"
        if "No Setup" in setup_badge or conviction_label == "CONFLICT" or confidence < 55:
            return "⛔ SKIP"

        strict_ok = (
            confidence >= 65
            and "Aligned" in setup_badge
            and conviction_label == "HIGH"
            and agreement >= 0.65
            and (pd.isna(adx_val) or adx_val >= 18)
        )
        if strict_ok:
            return "✅ ENTER"

        weak_count = 0
        if "Draft" in setup_badge or "Tech-Only" in setup_badge:
            weak_count += 1
        if conviction_label in {"LOW", "MEDIUM"}:
            weak_count += 1
        if agreement < 0.55:
            weak_count += 1
        if pd.notna(adx_val) and adx_val < 18:
            weak_count += 1
        return "⏳ WAIT" if weak_count >= 1 else "✅ ENTER"

    def _style_action(v: str) -> str:
        s = str(v)
        if "ENTER" in s:
            return f"color:{POSITIVE}; font-weight:800;"
        if "WAIT" in s:
            return f"color:{WARNING}; font-weight:800;"
        return f"color:{NEGATIVE}; font-weight:800;"

    scan_sig = (timeframe, signal_filter, int(top_n))
    last_sig = st.session_state.get("market_scan_sig")
    should_scan = refresh_scan or (last_sig != scan_sig) or ("market_scan_results" not in st.session_state)

    results: list[dict] = st.session_state.get("market_scan_results", [])

    # Fetch top coins
    if should_scan:
        with st.spinner(f"Scanning {top_n} coins ({signal_filter}) [{timeframe}] ..."):
            usdt_symbols, market_data = get_top_volume_usdt_symbols(max(top_n, 50))

            # skip "wrapped"
            seen_symbols = set()
            unique_market_data = []
            for coin in market_data:
                coin_id = (coin.get("id") or "").lower()
                symbol = (coin.get("symbol") or "").upper()
                if not symbol:
                    continue
                if "wrapped" in coin_id:
                    continue
                if symbol in seen_symbols:
                    continue
                seen_symbols.add(symbol)
                unique_market_data.append(coin)

            # Market cap map
            mcap_map = {}
            for coin in unique_market_data:
                symbol = (coin.get("symbol") or "").upper()
                mcap = int(coin.get("market_cap") or 0)
                if symbol and (symbol not in mcap_map or mcap > mcap_map[symbol]):
                    mcap_map[symbol] = mcap

            # USDT match
            valid_bases = {(c.get("symbol") or "").upper() for c in unique_market_data}
            working_symbols = [s for s in usdt_symbols if s.split("/")[0].upper() in valid_bases]
            working_symbols = working_symbols[:top_n]

            if not working_symbols:
                st.warning(
                    "No scanner symbols matched current market filters. "
                    f"Source pairs: {len(usdt_symbols)}, market rows: {len(unique_market_data)}, "
                    f"requested top_n: {top_n}."
                )

            # Analysis — parallelised data fetching for speed
            def _scan_one(sym: str) -> dict | None:
                """Analyse a single symbol for the scanner. Returns a row dict or None."""
                df = fetch_ohlcv(sym, timeframe, limit=500)
                if df is None or len(df) <= 60:
                    return None

                # Align analysis and scalp planning on same closed-candle context.
                df_eval = df.iloc[:-1].copy()
                if df_eval is None or len(df_eval) <= 55:
                    return None

                prob_up, ai_direction, ai_details = ml_ensemble_predict(df_eval)
                agreement = float(ai_details.get("agreement", 0.0)) if isinstance(ai_details, dict) else 0.0
                latest = df.iloc[-1]

                base = sym.split('/')[0].upper()
                mcap_val = mcap_map.get(base, 0)
                price = float(latest['close'])
                price_change = get_price_change(sym)

                a = analyse(df_eval)
                signal, volume_spike = a.signal, a.volume_spike
                atr_comment_v, candle_pattern_v, confidence_score_v = a.atr_comment, a.candle_pattern, a.confidence
                adx_val_v, supertrend_trend_v, ichimoku_trend_v = a.adx, a.supertrend, a.ichimoku
                stochrsi_k_val_v, bollinger_bias_v, vwap_label_v = a.stochrsi_k, a.bollinger, a.vwap
                psar_trend_v = a.psar

                scalp_direction, entry_s, target_s, stop_s, rr_ratio, breakout_note = get_scalping_entry_target(
                    df_eval, confidence_score_v, supertrend_trend_v, ichimoku_trend_v, vwap_label_v,
                    volume_spike, strict_mode=True,
                )
                entry_price = entry_s if scalp_direction else 0.0
                target_price = target_s if scalp_direction else 0.0

                include = (
                    (signal_filter == 'BOTH') or
                    (signal_filter == 'LONG' and signal in ['STRONG BUY', 'BUY']) or
                    (signal_filter == 'SHORT' and signal in ['STRONG SELL', 'SELL'])
                )
                if not include:
                    return None

                signal_direction = "LONG" if signal in ['STRONG BUY', 'BUY'] else ("SHORT" if signal in ['STRONG SELL', 'SELL'] else "NEUTRAL")

                if signal_direction == "LONG" and ai_direction == "SHORT":
                    ai_display = "⚠️ SHORT (Divergence)"
                elif signal_direction == "SHORT" and ai_direction == "LONG":
                    ai_display = "⚠️ LONG (Divergence)"
                elif signal_direction != "NEUTRAL" and ai_direction == "NEUTRAL":
                    ai_display = "NEUTRAL (Weak)"
                else:
                    ai_display = ai_direction

                _conv_lbl, _conv_clr = _calc_conviction(signal_direction, ai_direction, confidence_score_v)
                _emoji_map = {"HIGH": "🟢", "MEDIUM": "🟡", "LOW": "⚪", "CONFLICT": "🔴"}
                conviction = f"{_emoji_map.get(_conv_lbl, '')} {_conv_lbl}" if _conv_lbl else ""
                setup_badge = _setup_badge(scalp_direction or "", signal_direction, ai_direction)
                has_trade_plan = bool(entry_price and target_price and stop_s)
                action = _action_decision(
                    signal_direction,
                    float(confidence_score_v),
                    setup_badge,
                    str(_conv_lbl),
                    float(agreement),
                    float(adx_val_v) if pd.notna(adx_val_v) else float("nan"),
                    has_trade_plan,
                )

                return {
                    'Coin': base,
                    'Price ($)': _fmt_price(price),
                    'Δ (%)': format_delta(price_change) if price_change is not None else '',
                    'Action': action,
                    'Signal': signal_plain(signal),
                    'Confidence': confidence_score_badge(confidence_score_v),
                    'Confidence Band': _confidence_band(confidence_score_v),
                    'AI Ensemble': ai_display,
                    'AI Agree': f"{agreement * 100:.0f}%",
                    'AI Stability': _ai_agree_badge(agreement),
                    'Conviction': conviction,
                    'Setup': setup_badge,
                    'Scalp Opportunity': scalp_direction or "",
                    'Entry Zone': _fmt_price(entry_price) if entry_price else '',
                    'Invalidation (SL)': _fmt_price(stop_s) if stop_s else '',
                    'Target (TP)': _fmt_price(target_price) if target_price else '',
                    'Market Cap ($)': readable_market_cap(mcap_val),
                    'Spike Alert': '▲ Spike' if volume_spike else '',
                    'ADX': round(adx_val_v, 1) if pd.notna(adx_val_v) else float("nan"),
                    'SuperTrend': supertrend_trend_v,
                    'Volatility': atr_comment_v,
                    'Stochastic RSI': round(stochrsi_k_val_v, 2) if pd.notna(stochrsi_k_val_v) else float("nan"),
                    'Candle Pattern': candle_pattern_v,
                    'Ichimoku': ichimoku_trend_v,
                    'Bollinger': bollinger_bias_v,
                    'VWAP': vwap_label_v,
                    'PSAR': psar_trend_v if psar_trend_v != "Unavailable" else '',
                    'Williams %R': a.williams,
                    'CCI': a.cci,
                    '__confidence_val': confidence_score_v,
                }

            # Parallel scan using ThreadPoolExecutor (5-10x faster than sequential)
            fresh_results: list[dict] = []
            with ThreadPoolExecutor(max_workers=6) as executor:
                futures = {executor.submit(_scan_one, sym): sym for sym in working_symbols}
                for future in as_completed(futures):
                    try:
                        row = future.result()
                        if row is not None:
                            fresh_results.append(row)
                    except Exception as e:
                        _debug(f"Scanner error for {futures[future]}: {e}")

            prev_results = st.session_state.get("market_scan_results", [])
            # Sort and keep top_n
            fresh_results = sorted(fresh_results, key=lambda x: x['__confidence_val'], reverse=True)[:top_n]
            if fresh_results:
                st.session_state["market_scan_results"] = fresh_results
                st.session_state["market_scan_sig"] = scan_sig
                st.session_state["market_scan_cache_ts"] = pd.Timestamp.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")
                st.session_state["market_scan_cache_sig"] = scan_sig
                results = fresh_results
            else:
                # Keep last non-empty snapshot when APIs temporarily fail/rate-limit.
                st.session_state["market_scan_sig"] = scan_sig
                results = prev_results
                if prev_results:
                    ts = st.session_state.get("market_scan_cache_ts", "unknown time")
                    cache_sig = st.session_state.get("market_scan_cache_sig")
                    if cache_sig and tuple(cache_sig) != tuple(scan_sig):
                        st.warning(
                            f"Live scan returned no rows. Showing snapshot from {ts} "
                            f"(different filter/timeframe: {cache_sig})."
                        )
                    else:
                        st.warning(f"Live scan returned no rows. Showing last successful snapshot from {ts}.")

    # Prepare DataFrame for display
    if results:
        st.markdown(
            f"<details style='margin-bottom:0.6rem;'>"
            f"<summary style='color:{ACCENT}; cursor:pointer; font-size:0.9rem;'>"
            f"How to read quickly (?)</summary>"
            f"<div style='color:{TEXT_MUTED}; font-size:0.84rem; line-height:1.7; padding:0.4rem 0.2rem;'>"
            f"<b>1.</b> Read <b>Action</b> first: ✅ ENTER, ⏳ WAIT, ⛔ SKIP.<br>"
            f"<b>2.</b> Confirm with <b>Confidence</b> + <b>Setup</b> + <b>Invalidation (SL)</b>.<br>"
            f"<b>3.</b> Open <b>+ Show advanced columns</b> only when you need deeper diagnostics."
            f"</div></details>",
            unsafe_allow_html=True,
        )
        show_advanced = st.checkbox("+ Show advanced columns", value=False, key="market_show_adv_cols")
        st.markdown(
            f"<details style='margin-bottom:0.8rem;'>"
            f"<summary style='color:{ACCENT}; cursor:pointer; font-size:0.9rem;'>"
            f"ℹ️ Column Guide (click to expand)</summary>"
            f"<div style='color:{TEXT_MUTED}; font-size:0.84rem; line-height:1.7; padding:0.5rem;'>"
            "<b>Action</b> — The final quick decision. "
            "✅ ENTER = conditions are strong enough, ⏳ WAIT = setup is forming/unclear, ⛔ SKIP = low-quality or conflicting setup.<br>"
            "<b>Signal</b> — Technical direction only (LONG/SHORT/WAIT). "
            "This is not a full trade plan by itself.<br>"
            "<b>Confidence</b> — Strength score (0-100). Higher = stronger technical evidence.<br>"
            "<b>Confidence Band</b> — Quick quality label for confidence (Weak/Mixed/Good/Strong).<br>"
            "<b>Setup</b> — Trade-readiness check. "
            "Aligned = strongest, Tech-Only/Draft = partial, No Setup = no valid plan yet.<br>"
            "<b>Entry Zone</b> — Suggested area to enter (draft level).<br>"
            "<b>Invalidation (SL)</b> — Price where the setup is considered broken. "
            "If hit, the trade idea is wrong and should be exited.<br>"
            "<b>Target (TP)</b> — First planned take-profit area if the move works.<br>"
            "<b>AI Ensemble</b> — AI model direction (LONG/SHORT/NEUTRAL).<br>"
            "<b>AI Agree</b> — Percentage of AI model agreement. Higher = models agree more.<br>"
            "<b>AI Stability</b> — Agreement quality bucket (Strong/Medium/Weak).<br>"
            "<b>Conviction</b> — How well technical Signal + AI direction + confidence align.<br>"
            "<b>Scalp Opportunity</b> — Direction from strict entry model (if available).<br>"
            "<b>ADX</b> — Trend strength. Low = ranging, higher = stronger trend.<br>"
            "<b>SuperTrend / Ichimoku / VWAP / Bollinger / Stochastic RSI</b> — Trend and momentum context fields.<br>"
            "<b>Volatility / PSAR / Williams %R / CCI / Candle Pattern</b> — secondary confirmation fields.<br>"
            "<b>Spike Alert</b> — Abnormal volume flag. <b>Δ (%)</b> — latest 24h price change."
            "</div></details>",
            unsafe_allow_html=True,
        )

        df_results = pd.DataFrame(results)

        # indicator visual formatting
        df_results["SuperTrend"] = df_results["SuperTrend"].apply(format_trend)
        df_results["ADX"] = df_results["ADX"].apply(format_adx)
        df_results["Ichimoku"] = df_results["Ichimoku"].apply(format_trend)
        df_results["Stochastic RSI"] = df_results["Stochastic RSI"].apply(format_stochrsi)

        primary_cols = [
            "Coin",
            "Signal",
            "Confidence",
            "Setup",
            "Entry Zone",
            "Invalidation (SL)",
            "Target (TP)",
            "Action",
        ]
        all_cols = [
            "Coin",
            "Price ($)",
            "Δ (%)",
            "Action",
            "Signal",
            "Confidence",
            "Confidence Band",
            "AI Ensemble",
            "AI Agree",
            "AI Stability",
            "Conviction",
            "Setup",
            "Entry Zone",
            "Invalidation (SL)",
            "Target (TP)",
            "Scalp Opportunity",
            "Spike Alert",
            "Market Cap ($)",
            "ADX",
            "SuperTrend",
            "Ichimoku",
            "VWAP",
            "Bollinger",
            "Stochastic RSI",
            "Volatility",
            "PSAR",
            "Williams %R",
            "CCI",
            "Candle Pattern",
        ]
        display_cols = all_cols if show_advanced else primary_cols
        df_display = df_results[display_cols].copy()

        styled_summary = (
            df_display.style
            .map(style_signal, subset=[c for c in ["Signal", "AI Ensemble"] if c in df_display.columns])
            .map(style_confidence, subset=[c for c in ["Confidence"] if c in df_display.columns])
            .map(style_scalp_opp, subset=[c for c in ["Scalp Opportunity"] if c in df_display.columns])
            .map(style_delta, subset=[c for c in ["Δ (%)"] if c in df_display.columns])
            .map(_style_setup, subset=[c for c in ["Setup"] if c in df_display.columns])
            .map(_style_action, subset=[c for c in ["Action"] if c in df_display.columns])
            .map(
                _style_trend_state,
                subset=[c for c in ["SuperTrend", "Ichimoku", "VWAP", "Bollinger", "Volatility", "ADX", "Stochastic RSI"] if c in df_display.columns],
            )
        )
        st.dataframe(styled_summary, width="stretch", hide_index=True)

        csv_market = df_results.drop(columns=["__confidence_val"]).to_csv(index=False).encode("utf-8")
        st.download_button(
            label="Download Scan Results (CSV)",
            data=csv_market,
            file_name="scan_results.csv",
            mime="text/csv"
        )
    else:
        st.info("No coins matched the criteria.")
