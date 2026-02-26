from ui.ctx import get_ctx

from concurrent.futures import ThreadPoolExecutor, as_completed
import html

import pandas as pd
import plotly.graph_objs as go
from core.market_decision import action_decision, setup_badge, trade_quality
from core.signal_contract import strength_from_bias, strength_bucket
from core.metric_catalog import (
    AI_LONG_THRESHOLD,
    AI_SHORT_THRESHOLD,
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
    direction_label = get_ctx(ctx, "direction_label")
    readable_market_cap = get_ctx(ctx, "readable_market_cap")
    format_delta = get_ctx(ctx, "format_delta")
    format_trend = get_ctx(ctx, "format_trend")
    format_adx = get_ctx(ctx, "format_adx")
    format_stochrsi = get_ctx(ctx, "format_stochrsi")
    style_signal = get_ctx(ctx, "style_signal")
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
    st.markdown("<h1 class='title'>Crypto Market Intelligence Hub</h1>", unsafe_allow_html=True)
    st.markdown(
        f"<div class='panel-box'>"
        f"<b class='market-intro-title' style='color:{ACCENT};'>What does this tab show?</b>"
        f"<p class='market-intro-body' style='color:{TEXT_MUTED};'>"
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
    btc_prob = eth_prob = bnb_prob = sol_prob = ada_prob = xrp_prob = 0.5
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
    major_probs = [btc_prob, eth_prob, bnb_prob, sol_prob, ada_prob, xrp_prob]
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

    # Composite market score (0-100): Direction + Regime + Breadth + Trust
    direction_score = float(max(0.0, min(100.0, abs(float(behaviour_prob) - 0.5) * 200.0)))
    major_longs = sum(1 for p in major_probs if float(p) >= AI_LONG_THRESHOLD)
    major_shorts = sum(1 for p in major_probs if float(p) <= AI_SHORT_THRESHOLD)
    breadth_score = float(max(major_longs, major_shorts) / max(len(major_probs), 1) * 100.0)

    mcap_chg = abs(float(delta_mcap or 0.0))
    # Continuous regime scoring to avoid jumpy mode transitions.
    if mcap_chg <= 1.5:
        regime_score = 72.0 + (mcap_chg / 1.5) * 10.0
    elif mcap_chg <= 4.0:
        regime_score = 82.0 - ((mcap_chg - 1.5) / 2.5) * 24.0
    else:
        regime_score = 58.0 - (mcap_chg - 4.0) * 4.0
    regime_score = float(max(38.0, min(90.0, regime_score)))

    try:
        spread = float(pd.Series(major_probs).std())
    except Exception:
        spread = 0.18
    trust_score = float(max(0.0, min(100.0, 78.0 - spread * 100.0)))
    if direction_score < 25:
        trust_score = min(trust_score, 55.0)

    composite_score = (
        0.35 * direction_score
        + 0.20 * regime_score
        + 0.25 * breadth_score
        + 0.20 * trust_score
    )
    composite_score = float(max(0.0, min(100.0, composite_score)))
    composite_mode = (
        "Risk-On" if composite_score >= 68 else ("Selective" if composite_score >= 52 else "Risk-Off")
    )
    composite_color = POSITIVE if composite_mode == "Risk-On" else (WARNING if composite_mode == "Selective" else NEGATIVE)

    def _score_tone(v: float) -> tuple[str, str]:
        x = float(v)
        if x >= 70:
            return ("Strong", POSITIVE)
        if x >= 50:
            return ("Moderate", WARNING)
        return ("Weak", NEGATIVE)

    def _chip_center(label: str, tone_color: str, tip_text: str) -> str:
        return (
            f"<div class='market-gauge-chip-wrap'>"
            f"<span class='market-gauge-chip' style='border:1px solid {tone_color}; color:{tone_color}; "
            f"background:rgba(255,255,255,0.04);'>"
            f"{label} {_tip('', tip_text)}</span></div>"
        )

    def _dom_state(v: float) -> tuple[str, str]:
        x = float(v)
        if x >= 60:
            return ("High", POSITIVE)
        if x >= 45:
            return ("Balanced", WARNING)
        return ("Low", NEGATIVE)

    g1, g2, g3, g4 = st.columns(4, gap="medium")
    # BTC dominance gauge
    with g1:
        fig_btc = go.Figure(go.Indicator(
            mode="gauge+number",
            value=btc_dom,
            gauge={
                'axis': {'range': [0, 100], 'tickwidth': 1, 'tickvals': [0, 50, 100], 'tickfont': {'size': 12, 'color': TEXT_MUTED}},
                'bar': {'color': ACCENT},
                'bgcolor': CARD_BG,
                'steps': [
                    {'range': [0, AI_SHORT_THRESHOLD * 100], 'color': NEGATIVE},
                    {'range': [AI_SHORT_THRESHOLD * 100, AI_LONG_THRESHOLD * 100], 'color': WARNING},
                    {'range': [AI_LONG_THRESHOLD * 100, 100], 'color': POSITIVE},
                ],
            },
            title={'text': 'BTC Dominance (%)', 'font': {'size': 13, 'color': '#E5E7EB'}},
            number={'font': {'color': '#F8FAFC', 'size': 34}},
        ))
        fig_btc.update_layout(
            height=186,
            margin=dict(l=6, r=6, t=52, b=10),
            plot_bgcolor="#000000",
            paper_bgcolor="#000000",
        )
        st.plotly_chart(fig_btc, width="stretch")
        btc_state, btc_color = _dom_state(btc_dom)
        st.markdown(
            _chip_center("BTC Weight: " + btc_state, btc_color, "Bitcoin share of total market cap. High values usually indicate BTC-led market."),
            unsafe_allow_html=True,
        )

    # ETH dominance gauge
    with g2:
        fig_eth = go.Figure(go.Indicator(
            mode="gauge+number",
            value=eth_dom,
            gauge={
                'axis': {'range': [0, 100], 'tickwidth': 1, 'tickvals': [0, 50, 100], 'tickfont': {'size': 12, 'color': TEXT_MUTED}},
                'bar': {'color': ACCENT},
                'bgcolor': CARD_BG,
                'steps': [
                    {'range': [0, 15], 'color': NEGATIVE},
                    {'range': [15, 25], 'color': WARNING},
                    {'range': [25, 100], 'color': POSITIVE},
                ],
            },
            title={'text': 'ETH Dominance (%)', 'font': {'size': 13, 'color': '#E5E7EB'}},
            number={'font': {'color': '#F8FAFC', 'size': 34}},
        ))
        fig_eth.update_layout(
            height=186,
            margin=dict(l=6, r=6, t=52, b=10),
            plot_bgcolor="#000000",
            paper_bgcolor="#000000",
        )
        st.plotly_chart(fig_eth, width="stretch")
        eth_state, eth_color = _dom_state(eth_dom)
        st.markdown(
            _chip_center("ETH Weight: " + eth_state, eth_color, "Ethereum share of total market cap. Higher values show stronger ETH participation."),
            unsafe_allow_html=True,
        )

    # AI direction bias gauge
    with g3:
        fig_behaviour = go.Figure(go.Indicator(
            mode="gauge+number",
            value=int(round(behaviour_prob * 100)),
            gauge={
                'axis': {'range': [0, 100], 'tickwidth': 1, 'tickvals': [0, 50, 100], 'tickfont': {'size': 12, 'color': TEXT_MUTED}},
                'bar': {'color': ACCENT},
                'bgcolor': CARD_BG,
                'steps': [
                    {'range': [0, 40], 'color': NEGATIVE},
                    {'range': [40, 60], 'color': WARNING},
                    {'range': [60, 100], 'color': POSITIVE},
                ],
            },
            title={'text': 'AI Direction Bias (%)', 'font': {'size': 13, 'color': '#E5E7EB'}},
            number={'font': {'color': '#F8FAFC', 'size': 34}},
        ))
        fig_behaviour.update_layout(
            height=186,
            margin=dict(l=6, r=6, t=52, b=10),
            plot_bgcolor="#000000",
            paper_bgcolor="#000000",
        )
        st.plotly_chart(fig_behaviour, width="stretch")
        st.markdown(
            _chip_center(
                f"{behaviour_label} Bias",
                behaviour_color,
                "Dominance-weighted ML direction across BTC/ETH/BNB/SOL/ADA/XRP. Direction signal only.",
            ),
            unsafe_allow_html=True,
        )

    # Setup quality gauge (composite)
    with g4:
        fig_quality = go.Figure(go.Indicator(
            mode="gauge+number",
            value=int(round(composite_score)),
            gauge={
                'axis': {'range': [0, 100], 'tickwidth': 1, 'tickvals': [0, 50, 100], 'tickfont': {'size': 12, 'color': TEXT_MUTED}},
                'bar': {'color': ACCENT},
                'bgcolor': CARD_BG,
                'steps': [
                    {'range': [0, 45], 'color': NEGATIVE},
                    {'range': [45, 65], 'color': WARNING},
                    {'range': [65, 100], 'color': POSITIVE},
                ],
            },
            title={'text': 'Setup Quality (%)', 'font': {'size': 13, 'color': '#E5E7EB'}},
            number={'font': {'color': '#F8FAFC', 'size': 34}},
        ))
        fig_quality.update_layout(
            height=186,
            margin=dict(l=6, r=6, t=52, b=10),
            plot_bgcolor="#000000",
            paper_bgcolor="#000000",
        )
        st.plotly_chart(fig_quality, width="stretch")
        st.markdown(
            _chip_center(
                composite_mode,
                composite_color,
                "Setup quality combines Direction, Regime, Breadth and Trust. Measures environment quality, not direction alone.",
            ),
            unsafe_allow_html=True,
        )
        d_col = _score_tone(direction_score)[1]
        r_col = _score_tone(regime_score)[1]
        b_col = _score_tone(breadth_score)[1]
        t_col = _score_tone(trust_score)[1]
        st.markdown(
            f"<div style='display:flex; justify-content:center; gap:6px; margin-top:9px; flex-wrap:wrap;'>"
            f"<span title='Direction: strength of directional edge from AI direction bias. Higher = clearer market direction.' "
            f"style='background:rgba(255,255,255,0.05); border:1px solid {d_col}; color:{d_col}; "
            f"border-radius:999px; padding:2px 8px; font-size:0.72rem; overflow:visible; cursor:help;'>"
            f"Direction {direction_score:.0f}</span>"
            f"<span title='Regime: market environment quality proxy from total market-cap move behavior.' "
            f"style='background:rgba(255,255,255,0.05); border:1px solid {r_col}; color:{r_col}; "
            f"border-radius:999px; padding:2px 8px; font-size:0.72rem; overflow:visible; cursor:help;'>"
            f"Regime {regime_score:.0f}</span>"
            f"<span title='Breadth: how many major assets align on one side. Higher breadth means stronger participation.' "
            f"style='background:rgba(255,255,255,0.05); border:1px solid {b_col}; color:{b_col}; "
            f"border-radius:999px; padding:2px 8px; font-size:0.72rem; overflow:visible; cursor:help;'>"
            f"Breadth {breadth_score:.0f}</span>"
            f"<span title='Trust: reliability from cross-major model consistency. Lower dispersion = higher trust.' "
            f"style='background:rgba(255,255,255,0.05); border:1px solid {t_col}; color:{t_col}; "
            f"border-radius:999px; padding:2px 8px; font-size:0.72rem; overflow:visible; cursor:help;'>"
            f"Trust {trust_score:.0f}</span>"
            f"</div>",
            unsafe_allow_html=True,
        )

    # Divider
    st.markdown("\n\n")

    # Top coin scanner controls
    st.markdown(
        f"<h2 class='market-section-title' style='color:{ACCENT};'>Coin Action Scanner</h2>",
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
        direction_filter = st.selectbox("Direction", ['Upside', 'Downside', 'Both'], index=2)
    with controls[2]:
        top_n = st.slider("Top N", min_value=3, max_value=50, value=50)
    with controls[3]:
        refresh_scan = st.button("Refresh Scan", use_container_width=True)
    # Strict scalp mode is always enabled (non-strict path removed).

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
        if s in {"", "N/A", "Unavailable", "nan"}:
            return f"color:{TEXT_MUTED};"
        if s.startswith("▲"):
            return f"color:{POSITIVE}; font-weight:700;"
        if s.startswith("▼"):
            return f"color:{NEGATIVE}; font-weight:700;"
        if s.startswith("→"):
            return f"color:{WARNING}; font-weight:700;"
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

    def _style_secondary(v: str) -> str:
        s = str(v)
        if not s or s in {"N/A", "Unavailable", "nan"}:
            return f"color:{TEXT_MUTED}; opacity:0.7;"
        return f"color:{TEXT_MUTED}; opacity:0.9;"

    def _style_trend_state_soft(v: str) -> str:
        s = str(v)
        if s in {"", "N/A", "Unavailable", "nan"}:
            return f"color:{TEXT_MUTED}; opacity:0.75;"
        if s.startswith("▲"):
            return f"color:{POSITIVE}; font-weight:600; opacity:0.82;"
        if s.startswith("▼"):
            return f"color:{NEGATIVE}; font-weight:600; opacity:0.82;"
        if s.startswith("→"):
            return f"color:{WARNING}; font-weight:600; opacity:0.82;"
        if "Bullish" in s or "Above" in s or "Oversold" in s or "Near Bottom" in s or "Low" in s:
            return f"color:{POSITIVE}; font-weight:600; opacity:0.8;"
        if "Bearish" in s or "Below" in s or "Overbought" in s or "Near Top" in s or "High" in s:
            return f"color:{NEGATIVE}; font-weight:600; opacity:0.8;"
        if "Neutral" in s or "Moderate" in s or "Near VWAP" in s or "Starting" in s:
            return f"color:{WARNING}; font-weight:600; opacity:0.8;"
        return f"color:{TEXT_MUTED}; opacity:0.9;"

    def _row_action_highlight(row: pd.Series) -> list[str]:
        action = str(row.get("Action", ""))
        neutral = ""
        if "ENTER" in action:
            tone = "background-color: rgba(0,255,136,0.055);"
        elif "SKIP" in action:
            tone = "background-color: rgba(255,51,102,0.05);"
        else:
            tone = "background-color: rgba(255,209,102,0.04);"
        return [tone if col != "Action" else f"{tone} font-weight:800;" for col in row.index]

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

    def _normalize_indicator_label(v: object) -> str:
        raw = str(v or "").strip()
        if not raw or raw in {"Unavailable", "N/A", "nan"}:
            return "N/A"
        clean = (
            raw.replace("🟢 ", "")
            .replace("🔴 ", "")
            .replace("🟡 ", "")
            .replace("▲▲ ", "")
            .replace("▲ ", "")
            .replace("▼ ", "")
            .replace("→ ", "")
            .replace("– ", "")
            .strip()
        )
        if "Near Top" in clean:
            return "▼ Near Top"
        if "Near Bottom" in clean:
            return "▲ Near Bottom"
        if "Near VWAP" in clean:
            return "→ Near VWAP"
        if any(k in clean for k in ["Bullish", "Above", "Oversold", "Low"]):
            return f"▲ {clean}"
        if any(k in clean for k in ["Bearish", "Below", "Overbought", "High"]):
            return f"▼ {clean}"
        if any(k in clean for k in ["Neutral", "Moderate", "Starting"]):
            return f"→ {clean}"
        return clean

    def _style_action(v: str) -> str:
        s = str(v)
        if "ENTER" in s:
            return (
                f"color:{POSITIVE}; font-weight:800; "
                f"background-color:rgba(0,255,136,0.08); border:1px solid rgba(0,255,136,0.35);"
            )
        if "WATCH" in s:
            return (
                f"color:{WARNING}; font-weight:800; "
                f"background-color:rgba(255,209,102,0.08); border:1px solid rgba(255,209,102,0.35);"
            )
        return (
            f"color:{NEGATIVE}; font-weight:800; "
            f"background-color:rgba(255,51,102,0.08); border:1px solid rgba(255,51,102,0.35);"
        )

    def _style_trade_quality(v: str) -> str:
        s = str(v)
        if "A" in s:
            return (
                f"color:{POSITIVE}; font-weight:800; "
                f"background-color:rgba(0,255,136,0.08); border:1px solid rgba(0,255,136,0.35);"
            )
        if "B" in s:
            return (
                f"color:{WARNING}; font-weight:800; "
                f"background-color:rgba(255,209,102,0.08); border:1px solid rgba(255,209,102,0.35);"
            )
        return (
            f"color:{NEGATIVE}; font-weight:800; "
            f"background-color:rgba(255,51,102,0.08); border:1px solid rgba(255,51,102,0.35);"
        )

    def _rr_badge(rr_val: float) -> str:
        if rr_val <= 0:
            return ""
        if rr_val >= 2.0:
            return f"🟢 {rr_val:.2f}"
        if rr_val >= 1.3:
            return f"🟡 {rr_val:.2f}"
        return f"🔴 {rr_val:.2f}"

    def _style_rr(v: str) -> str:
        s = str(v)
        if s.startswith("🟢"):
            return f"color:{POSITIVE}; font-weight:700;"
        if s.startswith("🟡"):
            return f"color:{WARNING}; font-weight:700;"
        if s.startswith("🔴"):
            return f"color:{NEGATIVE}; font-weight:700;"
        return ""

    def _strength_badge(bias: float) -> str:
        strength = float(strength_from_bias(bias))
        label = strength_bucket(strength)
        return f"{strength:.0f}% ({label})"

    def _style_strength(v: str) -> str:
        s = str(v)
        if "(STRONG)" in s or "(GOOD)" in s:
            return f"color:{POSITIVE}; font-weight:700;"
        if "(MIXED)" in s:
            return f"color:{WARNING}; font-weight:700;"
        return f"color:{NEGATIVE}; font-weight:700;"

    def _tone_for_text(text: str, *, neutral_tone: str = "muted") -> str:
        s = str(text).strip().upper()
        if not s or s in {"N/A", "NA", "NAN", "UNAVAILABLE", "-"}:
            return "muted"
        s = (
            s.replace("🟢 ", "")
            .replace("🔴 ", "")
            .replace("🟡 ", "")
            .replace("⚪ ", "")
            .replace("✅ ", "")
            .replace("⏳ ", "")
            .replace("⛔ ", "")
        ).strip()
        if s.startswith("▲"):
            return "pos"
        if s.startswith("▼"):
            return "neg"
        if s.startswith("→"):
            return "warn"
        if any(
            k in s
            for k in [
                "ENTER", "LONG", "UPSIDE", "ALIGNED", "GOOD", "STRONG", "VERY STRONG", "EXTREME",
                "ABOVE", "BULLISH", "OVERSOLD", "NEAR BOTTOM",
            ]
        ):
            return "pos"
        if any(
            k in s
            for k in [
                "SKIP", "SHORT", "DOWNSIDE", "CONFLICT", "NO SETUP", "WEAK", "BEARISH",
                "OVERBOUGHT", "BELOW", "NEAR TOP",
            ]
        ):
            return "neg"
        if any(k in s for k in ["WATCH", "WAIT", "MIXED", "DRAFT", "TECH-ONLY", "NEUTRAL", "MEDIUM", "STARTING", "MODERATE", "SPIKE"]):
            return "warn"
        return neutral_tone

    def _tone_for_col(col: str, text: str) -> str:
        s = str(text or "").strip().upper()
        s = (
            s.replace("🟢 ", "")
            .replace("🔴 ", "")
            .replace("🟡 ", "")
            .replace("⚪ ", "")
            .replace("✅ ", "")
            .replace("⏳ ", "")
            .replace("⛔ ", "")
        ).strip()
        if not s or s in {"N/A", "NA", "NAN", "UNAVAILABLE", "-"}:
            return "muted"

        if col == "Action":
            if "ENTER" in s:
                return "pos"
            if "SKIP" in s:
                return "neg"
            return "warn"

        if col == "Direction":
            if "UPSIDE" in s:
                return "pos"
            if "DOWNSIDE" in s:
                return "neg"
            return "warn"

        if col == "Strength":
            if "GOOD" in s or "STRONG" in s:
                return "pos"
            if "MIXED" in s:
                return "warn"
            return "neg"

        if col == "Tech vs AI Alignment":
            if "HIGH" in s:
                return "pos"
            if "MEDIUM" in s:
                return "warn"
            if "TECH-ONLY" in s:
                return "warn"
            if "LOW" in s or "CONFLICT" in s:
                return "neg"
            return "muted"

        if col == "R:R":
            try:
                rr = float(
                    s.replace("🟢", "")
                    .replace("🟡", "")
                    .replace("🔴", "")
                    .strip()
                )
                if rr >= 1.5:
                    return "pos"
                if rr >= 1.2:
                    return "warn"
                return "neg"
            except Exception:
                return "muted"

        if col == "Scalp Opportunity":
            if "UPSIDE" in s:
                return "pos"
            if "DOWNSIDE" in s:
                return "neg"
            return "muted"

        if col == "AI Ensemble":
            if s.startswith("UPSIDE"):
                return "pos"
            if s.startswith("DOWNSIDE"):
                return "neg"
            return "warn"

        if col == "Volatility":
            if "LOW" in s:
                return "pos"
            if "MODERATE" in s or "NEUTRAL" in s:
                return "warn"
            if "HIGH" in s or "EXTREME" in s:
                return "neg"
            return "muted"

        if col == "Spike Alert":
            return "pos" if "SPIKE" in s else "muted"

        if col == "ADX":
            if "VERY STRONG" in s or "STRONG" in s:
                return "pos"
            if "STARTING" in s or "MODERATE" in s:
                return "warn"
            if "WEAK" in s:
                return "neg"
            return "muted"

        if col in {"SuperTrend", "Ichimoku", "VWAP", "PSAR"}:
            if "BULLISH" in s or "ABOVE" in s:
                return "pos"
            if "BEARISH" in s or "BELOW" in s:
                return "neg"
            return "warn"

        if col in {"Bollinger", "Stochastic RSI", "Williams %R", "CCI"}:
            if "OVERSOLD" in s or "NEAR BOTTOM" in s or "LOW" in s:
                return "pos"
            if "OVERBOUGHT" in s or "NEAR TOP" in s or "HIGH" in s:
                return "neg"
            return "warn"

        if col == "Candle Pattern":
            if "BULLISH" in s:
                return "pos"
            if "BEARISH" in s:
                return "neg"
            return "warn"

        return _tone_for_text(s)

    def _chip(text: object, tone: str | None = None) -> str:
        raw = "" if text is None else str(text).strip()
        if not raw or raw.upper() in {"N/A", "NA", "NAN", "UNAVAILABLE", "-"}:
            return ""
        tone_key = tone or _tone_for_text(raw)
        tone_map = {
            "pos": "mk-pos",
            "neg": "mk-neg",
            "warn": "mk-warn",
            "muted": "mk-muted",
            "info": "mk-info",
        }
        cls = tone_map.get(tone_key, "mk-muted")
        return f"<span class='mk-chip {cls}'>{html.escape(raw)}</span>"

    def _render_cell(col: str, row: dict) -> str:
        val = row.get(col, "")
        txt = "" if val is None else str(val).strip()
        if txt.upper() in {"N/A", "NA", "NAN", "UNAVAILABLE", "-"}:
            txt = ""
        if col == "Coin":
            return f"<span class='mk-coin'>{html.escape(txt)}</span>"
        if col in {"Action", "Direction", "Strength", "R:R", "Scalp Opportunity"}:
            return _chip(txt, _tone_for_col(col, txt))
        if col == "Tech vs AI Alignment":
            return _chip(txt, _tone_for_col(col, txt))
        if col == "AI Ensemble":
            t = _tone_for_col(col, txt)
            return _chip(txt, t)
        if col == "Δ (%)":
            if not txt:
                return ""
            if txt.startswith("▲"):
                return f"<span class='mk-delta mk-pos-t'>{html.escape(txt)}</span>"
            if txt.startswith("▼"):
                return f"<span class='mk-delta mk-neg-t'>{html.escape(txt)}</span>"
            return f"<span class='mk-delta mk-muted-t'>{html.escape(txt)}</span>"
        if col in {"Entry Price", "Stop Loss", "Target Price"}:
            return f"<span class='mk-plain'>{html.escape(txt)}</span>" if txt else ""
        if col in {"ADX", "SuperTrend", "Ichimoku", "VWAP", "Bollinger", "Stochastic RSI", "Volatility", "PSAR", "Williams %R", "CCI", "Candle Pattern", "Spike Alert"}:
            return _chip(txt, _tone_for_col(col, txt)) if txt else ""
        return f"<span class='mk-plain'>{html.escape(txt)}</span>"

    def _render_pro_table(df: pd.DataFrame, cols: list[str]) -> None:
        sticky_order: list[str] = ["Coin"]
        col_widths = {
            "Coin": 120,
            "Price ($)": 122,
            "Δ (%)": 92,
            "Action": 112,
            "Direction": 106,
            "Strength": 132,
        }
        left_offsets: dict[str, str] = {}
        running_left = 0
        for c in sticky_order:
            left_offsets[c] = f"{running_left}px"
            running_left += col_widths[c]
        sticky_cols = set(sticky_order)

        header_html = []
        for c in cols:
            sticky = ""
            width_style = ""
            if c in col_widths:
                w = col_widths[c]
                width_style = f"min-width:{w}px; max-width:{w}px; width:{w}px;"
            if c in sticky_cols:
                sticky = (
                    f"position:sticky; left:{left_offsets[c]}; z-index:7; "
                    f"background:linear-gradient(180deg, rgba(18,24,36,0.99), rgba(12,18,30,0.99)); "
                    f"box-shadow: 1px 0 0 rgba(148,163,184,0.16);"
                )
            header_html.append(f"<th style='{width_style}{sticky}'>{html.escape(c)}</th>")

        rows_html = []
        for _, r in df.iterrows():
            row_dict = r.to_dict()
            cell_html = []
            for c in cols:
                sticky = ""
                width_style = ""
                if c in col_widths:
                    w = col_widths[c]
                    width_style = f"min-width:{w}px; max-width:{w}px; width:{w}px;"
                if c in sticky_cols:
                    sticky = (
                        f"position:sticky; left:{left_offsets[c]}; z-index:6; "
                        f"background:rgba(8,12,20,1.0); box-shadow:1px 0 0 rgba(148,163,184,0.22), 2px 0 10px rgba(0,0,0,0.24);"
                    )
                cell_html.append(f"<td style='{width_style}{sticky}'>{_render_cell(c, row_dict)}</td>")
            rows_html.append("<tr>" + "".join(cell_html) + "</tr>")

        st.markdown(
            f"""
            <style>
            .scan-kpi-value {{
              color:#F8FAFC;
              font-family:'Space Grotesk','Manrope',sans-serif;
              font-size:2rem;
              font-weight:700;
              letter-spacing:0.2px;
              margin-top:4px;
              line-height:1.1;
            }}
            .scan-kpi-sub {{
              color:{TEXT_MUTED};
              font-size:0.84rem;
              margin-top:8px;
              letter-spacing:0.15px;
            }}
            .mk-wrap {{
              width:100%;
              overflow-x:auto;
              border:1px solid rgba(0,212,255,0.20);
              border-radius:12px;
              background:linear-gradient(180deg, rgba(6,10,18,0.96), rgba(4,8,14,0.96));
              box-shadow:0 10px 28px rgba(0,0,0,0.34), inset 0 0 0 1px rgba(255,255,255,0.03);
            }}
            .mk-table {{
              width:max-content;
              min-width:100%;
              border-collapse:separate;
              border-spacing:0;
              font-size:0.82rem;
              font-family:'Manrope','Segoe UI',sans-serif;
            }}
            .mk-table th {{
              text-align:left;
              padding:10px 10px;
              color:{TEXT_MUTED};
              font-weight:700;
              letter-spacing:0.25px;
              border-bottom:1px solid rgba(148,163,184,0.22);
              border-right:1px solid rgba(148,163,184,0.08);
              white-space:nowrap;
              top:0;
              position:sticky;
              z-index:4;
              background:linear-gradient(180deg, rgba(18,24,36,0.98), rgba(12,18,30,0.98));
            }}
            .mk-table td {{
              padding:8px 10px;
              color:#E5E7EB;
              border-bottom:1px solid rgba(148,163,184,0.12);
              border-right:1px solid rgba(148,163,184,0.07);
              white-space:nowrap;
              vertical-align:middle;
            }}
            .mk-table tr:hover td {{
              background-color:rgba(0,212,255,0.06);
            }}
            .mk-table tr:hover td[style*="position:sticky"] {{
              background-color:rgba(8,12,20,1.0) !important;
            }}
            .mk-chip {{
              display:inline-flex;
              align-items:center;
              gap:6px;
              padding:2px 8px;
              border-radius:999px;
              border:1px solid rgba(148,163,184,0.34);
              background:rgba(148,163,184,0.10);
              font-size:0.74rem;
              font-weight:700;
            }}
            .mk-pos {{ color:{POSITIVE}; border-color:rgba(0,255,136,0.42); background:rgba(0,255,136,0.10); }}
            .mk-neg {{ color:{NEGATIVE}; border-color:rgba(255,51,102,0.44); background:rgba(255,51,102,0.10); }}
            .mk-warn {{ color:{WARNING}; border-color:rgba(255,209,102,0.46); background:rgba(255,209,102,0.10); }}
            .mk-info {{ color:{ACCENT}; border-color:rgba(0,212,255,0.46); background:rgba(0,212,255,0.10); }}
            .mk-muted {{ color:{TEXT_MUTED}; border-color:rgba(140,161,182,0.35); background:rgba(140,161,182,0.08); }}
            .mk-coin {{ font-weight:800; letter-spacing:0.2px; color:#F8FAFC; }}
            .mk-plain {{ color:#E5E7EB; }}
            .mk-delta {{ font-weight:700; }}
            .mk-pos-t {{ color:{POSITIVE}; }}
            .mk-neg-t {{ color:{NEGATIVE}; }}
            .mk-muted-t {{ color:{TEXT_MUTED}; }}
            </style>
            <div class="mk-wrap">
              <table class="mk-table">
                <thead><tr>{''.join(header_html)}</tr></thead>
                <tbody>{''.join(rows_html)}</tbody>
              </table>
            </div>
            """,
            unsafe_allow_html=True,
        )

    scan_sig = (timeframe, direction_filter, int(top_n))
    last_sig = st.session_state.get("market_scan_sig")
    should_scan = refresh_scan or (last_sig != scan_sig) or ("market_scan_results" not in st.session_state)

    results: list[dict] = st.session_state.get("market_scan_results", [])
    source_label = st.session_state.get("market_scan_source", "LIVE")

    # Fetch top coins
    if should_scan:
        with st.spinner(f"Scanning {top_n} coins ({direction_filter}) [{timeframe}] ..."):
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
            elif len(working_symbols) < top_n:
                st.info(
                    f"Liquidity universe currently returned {len(working_symbols)} eligible symbols "
                    f"(requested {top_n}). Scanner remains strict to top-volume matched pairs."
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
                consensus_agreement = float(ai_details.get("consensus_agreement", 0.0)) if isinstance(ai_details, dict) else 0.0
                latest = df.iloc[-1]

                base = sym.split('/')[0].upper()
                mcap_val = mcap_map.get(base, 0)
                price = float(latest['close'])
                price_change = get_price_change(sym)

                a = analyse(df_eval)
                signal, volume_spike = a.signal, a.volume_spike
                atr_comment_v, candle_pattern_v, bias_score_v = a.atr_comment, a.candle_pattern, a.bias
                adx_val_v, supertrend_trend_v, ichimoku_trend_v = a.adx, a.supertrend, a.ichimoku
                stochrsi_k_val_v, bollinger_bias_v, vwap_label_v = a.stochrsi_k, a.bollinger, a.vwap
                psar_trend_v = a.psar

                scalp_direction, entry_s, target_s, stop_s, rr_ratio, breakout_note = get_scalping_entry_target(
                    df_eval, bias_score_v, supertrend_trend_v, ichimoku_trend_v, vwap_label_v,
                    volume_spike, strict_mode=True,
                )
                entry_price = entry_s if scalp_direction else 0.0
                target_price = target_s if scalp_direction else 0.0

                include = (
                    (direction_filter == 'Both') or
                    (direction_filter == 'Upside' and signal in ['STRONG BUY', 'BUY']) or
                    (direction_filter == 'Downside' and signal in ['STRONG SELL', 'SELL'])
                )
                if not include:
                    return None

                signal_direction = "LONG" if signal in ['STRONG BUY', 'BUY'] else ("SHORT" if signal in ['STRONG SELL', 'SELL'] else "NEUTRAL")

                ai_display = direction_label(ai_direction)
                vote_ratio = agreement if ai_direction in {"LONG", "SHORT"} else consensus_agreement
                consensus_votes = max(0, min(3, int(round(float(vote_ratio) * 3.0))))
                ai_display = f"{ai_display} ({consensus_votes}/3)"

                _emoji_map = {"HIGH": "🟢", "MEDIUM": "🟡", "LOW": "⚪", "CONFLICT": "🔴"}
                setup_badge_val = setup_badge(scalp_direction or "", signal_direction, ai_direction)
                has_trade_plan = bool(entry_price and target_price and stop_s)
                strength_val = float(strength_from_bias(float(bias_score_v)))
                _conv_lbl, _conv_clr = _calc_conviction(
                    signal_direction,
                    ai_direction,
                    strength_val,
                    float(agreement),
                )
                conviction = f"{_emoji_map.get(_conv_lbl, '')} {_conv_lbl}" if _conv_lbl else ""
                rr_val = float(rr_ratio) if rr_ratio else 0.0
                action = action_decision(
                    signal_direction,
                    strength_val,
                    setup_badge_val,
                    str(_conv_lbl),
                    float(agreement),
                    float(adx_val_v) if pd.notna(adx_val_v) else float("nan"),
                    rr_val,
                    has_trade_plan,
                )
                trade_quality_val = trade_quality(
                    action,
                    setup_badge_val,
                    str(_conv_lbl),
                    strength_val,
                    float(agreement),
                    rr_val,
                )

                return {
                    'Coin': base,
                    'Price ($)': _fmt_price(price),
                    'Δ (%)': format_delta(price_change) if price_change is not None else '',
                    'Action': action,
                    'Trade Quality': trade_quality_val,
                    'Direction': direction_label(signal_plain(signal)),
                    'Strength': _strength_badge(float(bias_score_v)),
                    'AI Ensemble': ai_display,
                    'Tech vs AI Alignment': conviction,
                    'Setup': setup_badge_val,
                    'Scalp Opportunity': direction_label(scalp_direction or ""),
                    'Entry Price': _fmt_price(entry_price) if entry_price else '',
                    'Stop Loss': _fmt_price(stop_s) if stop_s else '',
                    'Target Price': _fmt_price(target_price) if target_price else '',
                    'R:R': _rr_badge(rr_val),
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
                    '__strength_val': strength_val,
                }

            # Parallel scan using ThreadPoolExecutor (5-10x faster than sequential)
            fresh_results: list[dict] = []
            with ThreadPoolExecutor(max_workers=4) as executor:
                futures = {executor.submit(_scan_one, sym): sym for sym in working_symbols}
                for future in as_completed(futures):
                    try:
                        row = future.result()
                        if row is not None:
                            fresh_results.append(row)
                    except Exception as e:
                        _debug(f"Scanner error for {futures[future]}: {e}")

            prev_results = st.session_state.get("market_scan_results", [])
            # Sort by execution priority: Action > Setup > Strength
            def _action_rank(v: str) -> int:
                s = str(v or "").upper()
                if "ENTER" in s:
                    return 3
                if "WATCH" in s:
                    return 2
                if "SKIP" in s:
                    return 1
                return 0
            setup_rank = {"🟢 Aligned": 4, "🟡 Tech-Only": 3, "⚪ Draft": 2, "🔴 No Setup": 1}
            fresh_results = sorted(
                fresh_results,
                key=lambda x: (
                    _action_rank(str(x.get("Action"))),
                    setup_rank.get(str(x.get("Setup")), 0),
                    float(x.get("__strength_val", 0.0)),
                ),
                reverse=True,
            )[:top_n]
            if fresh_results:
                st.session_state["market_scan_results"] = fresh_results
                st.session_state["market_scan_sig"] = scan_sig
                st.session_state["market_scan_cache_ts"] = pd.Timestamp.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")
                st.session_state["market_scan_cache_sig"] = scan_sig
                st.session_state["market_scan_source"] = "LIVE"
                source_label = "LIVE"
                results = fresh_results
            else:
                # Keep last non-empty snapshot when APIs temporarily fail/rate-limit.
                st.session_state["market_scan_sig"] = scan_sig
                results = prev_results
                if prev_results:
                    ts = st.session_state.get("market_scan_cache_ts", "unknown time")
                    source_label = f"CACHED ({ts})"
                    st.session_state["market_scan_source"] = source_label
                    cache_sig = st.session_state.get("market_scan_cache_sig")
                    if cache_sig and tuple(cache_sig) != tuple(scan_sig):
                        st.warning(
                            f"Live scan returned no rows. Showing snapshot from {ts} "
                            f"(different filter/timeframe: {cache_sig}). "
                            f"Do not execute directly from this snapshot; refresh until LIVE rows return."
                        )
                    else:
                        st.warning(
                            f"Live scan returned no rows. Showing last successful snapshot from {ts}. "
                            f"Do not execute directly from cache-only view."
                        )

    # Prepare DataFrame for display
    if results:
        source_color = POSITIVE if source_label.startswith("LIVE") else WARNING
        source_chip = "LIVE FEED" if source_label.startswith("LIVE") else "CACHED SNAPSHOT"
        st.markdown(
            f"<div style='margin:0 0 0.45rem 0;'>"
            f"<span class='market-inline-chip' style='border:1px solid {source_color}; color:{source_color}; "
            f"background:rgba(255,255,255,0.04);'>"
            f"{source_chip} • {source_label}</span></div>",
            unsafe_allow_html=True,
        )
        if source_label.startswith("CACHED"):
            st.markdown(
                f"<div class='market-note-box' style='border:1px solid rgba(255,209,102,0.4); border-left:4px solid {WARNING}; "
                f"background:rgba(255,209,102,0.08); color:{TEXT_MUTED};'>"
                f"<b style='color:{WARNING};'>Execution Caution:</b> This table is running on cached snapshot data. "
                f"Confirm with a fresh LIVE scan before placing trades."
                f"</div>",
                unsafe_allow_html=True,
            )
        st.markdown(
            f"<details class='market-details' style='margin-bottom:0.6rem;'>"
            f"<summary style='color:{ACCENT};'>"
            f"How to read quickly (?)</summary>"
            f"<div class='market-details-body' style='color:{TEXT_MUTED};'>"
            f"<b>1.</b> Read <b>Action</b> first: ✅ ENTER (Trend+AI), 🟡 ENTER (Trend-Only / AI-Only), 👀 WATCH, ⛔ SKIP.<br>"
            f"<b>2.</b> Confirm with <b>Direction</b> + <b>Strength</b> + <b>AI Ensemble</b> + <b>Tech vs AI Alignment</b>.<br>"
            f"<b>3.</b> Use <b>R:R + Entry Price / Stop Loss / Target Price</b> only for execution planning after Action confirms.<br>"
            f"<b>4.</b> Open <b>+ Show advanced columns</b> only when you need deeper diagnostics."
            f"</div></details>",
            unsafe_allow_html=True,
        )
        st.markdown(
            f"<details class='market-details' style='margin-bottom:0.6rem;'>"
            f"<summary style='color:{ACCENT};'>Scanner logic (what drives decisions?)</summary>"
            f"<div class='market-details-body' style='color:{TEXT_MUTED};'>"
            f"<b>Step 1:</b> Build universe from top liquid USDT pairs (wrapped tokens removed).<br>"
            f"<b>Step 2:</b> Run technical analysis + AI ensemble per coin, then generate execution plan fields (Entry/SL/TP).<br>"
            f"<b>Step 3:</b> Build bias from a regime-aware engine (trend/momentum/volume/volatility), then convert to direction-agnostic <b>Strength</b> (0-100).<br>"
            f"<b>Step 4:</b> Calculate <b>Action</b> with guardrails. "
            f"ENTER uses dynamic regime gates: thresholds relax slightly in strong-trend ADX, remain strict in weak ADX. "
            f"TECH-ONLY can enter only in exceptional trend+strength conditions. "
            f"R:R and plan levels are execution aids, not Action gates.<br>"
            f"<b>Step 5:</b> Convert raw outputs into one actionable class: ENTER / WATCH / SKIP."
            f"</div></details>",
            unsafe_allow_html=True,
        )
        show_advanced = st.checkbox("+ Show advanced columns", value=False, key="market_show_adv_cols")
        st.markdown(
            f"<details class='market-details' style='margin-bottom:0.8rem;'>"
            f"<summary style='color:{ACCENT};'>"
            f"ℹ️ Column Guide (click to expand)</summary>"
            f"<div class='market-details-body' style='color:{TEXT_MUTED}; padding:0.5rem;'>"
            "<b>Decision Block (left)</b>: Action, Direction, Strength. "
            "Start here for go/no-go decision.<br>"
            "<b>Strength meaning</b>: 0-100 direction-agnostic edge force. Read Upside/Downside from Direction; "
            "use Strength as conviction power of that direction.<br>"
            "<b>Confirmation Block</b>: AI Ensemble and Tech vs AI Alignment. "
            "Confirms whether technical and AI context agree for Action.<br>"
            "<b>Plan Block</b>: Entry Price, Stop Loss, Target Price. "
            "Draft execution levels from strict setup engine.<br>"
            "<b>Diagnostics (advanced)</b>: ADX, SuperTrend, Ichimoku, VWAP first; "
            "then secondary oscillators and pattern fields.<br>"
            "<b>Note</b>: Decision columns are always kept first in table order for rapid scanning."
            "</div></details>",
            unsafe_allow_html=True,
        )
        st.markdown(
            f"<div style='display:flex; flex-wrap:wrap; gap:8px; margin:0 0 0.55rem 0;'>"
            f"<span class='market-criteria-chip' style='border:1px solid rgba(0,255,136,0.35); color:{POSITIVE}; background:rgba(0,255,136,0.08);'>"
            f"ENTER (Trend+AI): strongest class, technical and AI confirmation align</span>"
            f"<span class='market-criteria-chip' style='border:1px solid rgba(255,209,102,0.35); color:{WARNING}; background:rgba(255,209,102,0.08);'>"
            f"ENTER (Trend-Only / AI-Only): secondary class, one side leads while risk guards still pass</span>"
            f"<span class='market-criteria-chip' style='border:1px solid rgba(255,209,102,0.35); color:{WARNING}; background:rgba(255,209,102,0.08);'>"
            f"WATCH: direction exists, but confirmation is not complete yet</span>"
            f"<span class='market-criteria-chip' style='border:1px solid rgba(255,51,102,0.35); color:{NEGATIVE}; background:rgba(255,51,102,0.08);'>"
            f"SKIP: no clear direction, conflict, or very weak edge</span>"
            f"</div>",
            unsafe_allow_html=True,
        )
        st.caption("Signals and plan levels are computed on closed candles; Price ($) shows the latest live tick.")

        df_results = pd.DataFrame(results)

        # Quick scan health summary (visual-first, logic unchanged)
        action_series = df_results.get("Action", pd.Series(dtype=str)).astype(str)
        enter_count = int(action_series.str.contains("ENTER", na=False).sum())
        watch_count = int(action_series.str.contains("WATCH", na=False).sum())
        skip_count = int(action_series.str.contains("SKIP", na=False).sum())
        rr_values = (
            df_results.get("R:R", pd.Series(dtype=str))
            .astype(str)
            .str.replace("🟢", "", regex=False)
            .str.replace("🟡", "", regex=False)
            .str.replace("🔴", "", regex=False)
            .str.strip()
        )
        rr_numeric = pd.to_numeric(rr_values, errors="coerce")
        aligned_count = (
            int(df_results["Setup"].astype(str).str.contains("Aligned", na=False).sum())
            if "Setup" in df_results.columns
            else 0
        )

        best_scalp_coin = "—"
        best_scalp_sub = "No ENTER-qualified scalp setup in this scan."
        if "Scalp Opportunity" in df_results.columns:
            scalp_mask = df_results["Scalp Opportunity"].astype(str).isin(["Upside", "Downside"])
            scoped = df_results[scalp_mask].copy()
            if not scoped.empty:
                scoped["__rr"] = pd.to_numeric(
                    scoped["R:R"]
                    .astype(str)
                    .str.replace("🟢", "", regex=False)
                    .str.replace("🟡", "", regex=False)
                    .str.replace("🔴", "", regex=False)
                    .str.strip(),
                    errors="coerce",
                )
                scoped["__enter"] = scoped["Action"].astype(str).str.contains("ENTER", na=False)
                scoped = scoped.dropna(subset=["__rr"])
                scoped = scoped[scoped["__enter"]]
                if not scoped.empty:
                    best_row = scoped.sort_values(["__rr"], ascending=[False]).iloc[0]
                    best_scalp_coin = str(best_row.get("Coin", "—"))
                    best_dir = str(best_row.get("Scalp Opportunity", ""))
                    best_rr = float(best_row["__rr"])
                    best_scalp_sub = f"{best_dir} | R:R {best_rr:.2f}"

        strength_coin = "—"
        strength_sub = "No strength data available."
        if "__strength_val" in df_results.columns and len(df_results) > 0:
            strength_series = pd.to_numeric(df_results["__strength_val"], errors="coerce").dropna()
            if not strength_series.empty:
                strength_idx = strength_series.idxmax()
                row = df_results.loc[strength_idx]
                strength_coin = str(row.get("Coin", "—"))
                strength_sub = (
                    f"Strength {float(row.get('__strength_val', 0.0)):.0f}% • "
                    f"Direction {row.get('Direction', '')} • "
                    f"AI {row.get('AI Ensemble', '')}"
                )

        q1, q2, q3, q4 = st.columns(4, gap="small")
        with q1:
            status_head = f"{enter_count} Enter Ready" if enter_count > 0 else "No Enter Setup"
            status_sub = f"ENTER {enter_count} • WATCH {watch_count} • SKIP {skip_count}"
            st.markdown(
                "<div class='elite-card'>"
                "<div class='elite-label'>Action Status</div>"
                f"<div class='scan-kpi-value'>{status_head}</div>"
                f"<div class='scan-kpi-sub'>{status_sub}</div>"
                "</div>",
                unsafe_allow_html=True,
            )
        with q2:
            st.markdown(
                "<div class='elite-card'>"
                "<div class='elite-label'>Aligned Structure Count</div>"
                f"<div class='elite-value'>{aligned_count}</div>"
                "<div class='elite-sub'>coins passing full setup alignment</div>"
                "</div>",
                unsafe_allow_html=True,
            )
        with q3:
            st.markdown(
                "<div class='elite-card'>"
                "<div class='elite-label'>Best Scalp Opportunity</div>"
                f"<div class='elite-value'>{best_scalp_coin}</div>"
                f"<div class='elite-sub'>{best_scalp_sub}</div>"
                "</div>",
                unsafe_allow_html=True,
            )
        with q4:
            st.markdown(
                "<div class='elite-card'>"
                "<div class='elite-label'>Strength Leader</div>"
                f"<div class='elite-value'>{strength_coin}</div>"
                f"<div class='elite-sub'>{strength_sub}</div>"
                "</div>",
                unsafe_allow_html=True,
            )
        st.markdown("<div style='height:0.7rem;'></div>", unsafe_allow_html=True)

        # indicator visual formatting
        df_results["SuperTrend"] = df_results["SuperTrend"].apply(format_trend)
        df_results["ADX"] = df_results["ADX"].apply(format_adx)
        df_results["Ichimoku"] = (
            df_results["Ichimoku"]
            .apply(format_trend)
            .replace("–", "")
        )
        df_results["Stochastic RSI"] = df_results["Stochastic RSI"].apply(format_stochrsi).apply(_normalize_indicator_label)
        df_results["VWAP"] = df_results["VWAP"].apply(_normalize_indicator_label)
        df_results["Bollinger"] = df_results["Bollinger"].apply(_normalize_indicator_label)
        df_results["PSAR"] = df_results["PSAR"].apply(_normalize_indicator_label)
        df_results["Williams %R"] = df_results["Williams %R"].apply(_normalize_indicator_label)
        df_results["CCI"] = df_results["CCI"].apply(_normalize_indicator_label)

        primary_cols = [
            "Coin",
            "Price ($)",
            "Δ (%)",
            "Action",
            "Direction",
            "Strength",
            "AI Ensemble",
            "Tech vs AI Alignment",
            "R:R",
            "Entry Price",
            "Stop Loss",
            "Target Price",
            "Scalp Opportunity",
            "Market Cap ($)",
        ]
        advanced_extra_cols = [
            "ADX",
            "SuperTrend",
            "Ichimoku",
            "VWAP",
            "Spike Alert",
            "Bollinger",
            "Stochastic RSI",
            "Volatility",
            "PSAR",
            "Williams %R",
            "CCI",
            "Candle Pattern",
        ]
        all_cols = primary_cols + [c for c in advanced_extra_cols if c not in primary_cols]
        display_cols = all_cols if show_advanced else primary_cols
        df_display = df_results[display_cols].copy()

        _render_pro_table(df_display, display_cols)

        csv_market = df_results.drop(columns=["__strength_val"]).to_csv(index=False).encode("utf-8")
        st.download_button(
            label="Download Scan Results (CSV)",
            data=csv_market,
            file_name="scan_results.csv",
            mime="text/csv"
        )
    else:
        st.info("No coins matched the criteria.")
