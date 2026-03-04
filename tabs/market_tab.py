from ui.ctx import get_ctx

from concurrent.futures import ThreadPoolExecutor, as_completed
import html
import re
from threading import Lock

import pandas as pd
import plotly.graph_objs as go
from core.market_decision import (
    ai_vote_metrics,
    action_decision_with_reason,
    action_rank,
    action_reason_text,
    normalize_action_class,
    structure_state,
)
from core.scalping import scalp_gate_thresholds
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
    get_market_top_snapshot = get_ctx(ctx, "get_market_top_snapshot")
    get_price_change = get_ctx(ctx, "get_price_change")
    _tip = get_ctx(ctx, "_tip")
    get_major_ohlcv_bundle = get_ctx(ctx, "get_major_ohlcv_bundle")
    ml_ensemble_predict = get_ctx(ctx, "ml_ensemble_predict")
    get_top_volume_usdt_symbols = get_ctx(ctx, "get_top_volume_usdt_symbols")
    fetch_ohlcv = get_ctx(ctx, "fetch_ohlcv")
    analyse = get_ctx(ctx, "analyse")
    get_scalping_entry_target = get_ctx(ctx, "get_scalping_entry_target")
    scalp_quality_gate = get_ctx(ctx, "scalp_quality_gate")
    _calc_conviction = get_ctx(ctx, "_calc_conviction")
    signal_plain = get_ctx(ctx, "signal_plain")
    direction_key = get_ctx(ctx, "direction_key")
    direction_label = get_ctx(ctx, "direction_label")
    readable_market_cap = get_ctx(ctx, "readable_market_cap")
    format_delta = get_ctx(ctx, "format_delta")
    format_trend = get_ctx(ctx, "format_trend")
    format_adx = get_ctx(ctx, "format_adx")
    format_stochrsi = get_ctx(ctx, "format_stochrsi")
    sanitize_trading_terms = get_ctx(ctx, "sanitize_trading_terms")
    _debug = get_ctx(ctx, "_debug")
    major_fallback_symbols = [
        "BTC/USDT",
        "ETH/USDT",
        "SOL/USDT",
        "XRP/USDT",
        "BNB/USDT",
        "ADA/USDT",
        "DOGE/USDT",
        "AVAX/USDT",
        "LINK/USDT",
        "TON/USDT",
    ]

    # Unified top snapshot (provider-consistent + fallback + last-good cache).
    top_snapshot = get_market_top_snapshot()
    btc_dom_raw = top_snapshot.get("btc_dom")
    eth_dom_raw = top_snapshot.get("eth_dom")
    total_mcap_raw = top_snapshot.get("total_mcap")
    mcap_24h_pct = top_snapshot.get("mcap_24h_pct")
    bnb_dom_raw = top_snapshot.get("bnb_dom")
    sol_dom_raw = top_snapshot.get("sol_dom")
    ada_dom_raw = top_snapshot.get("ada_dom")
    xrp_dom_raw = top_snapshot.get("xrp_dom")

    def _to_num(v: object) -> float:
        try:
            f = float(v)
            return f if pd.notna(f) else 0.0
        except Exception:
            return 0.0

    btc_dom = _to_num(btc_dom_raw)
    eth_dom = _to_num(eth_dom_raw)
    bnb_dom = _to_num(bnb_dom_raw)
    sol_dom = _to_num(sol_dom_raw)
    ada_dom = _to_num(ada_dom_raw)
    xrp_dom = _to_num(xrp_dom_raw)
    total_mcap = _to_num(total_mcap_raw)
    fg_value_raw = top_snapshot.get("fg_value")
    fg_label = str(top_snapshot.get("fg_label") or "Unavailable")
    fg_value = fg_value_raw if isinstance(fg_value_raw, (int, float)) else None
    fg_available = fg_value is not None
    btc_price_raw = top_snapshot.get("btc_price")
    eth_price_raw = top_snapshot.get("eth_price")
    btc_price = float(btc_price_raw) if isinstance(btc_price_raw, (int, float)) else None
    eth_price = float(eth_price_raw) if isinstance(eth_price_raw, (int, float)) else None

    # Treat all-zero dominance payload as unavailable upstream enrichment.
    dominance_sum = (
        max(btc_dom, 0.0)
        + max(eth_dom, 0.0)
        + max(bnb_dom, 0.0)
        + max(sol_dom, 0.0)
        + max(ada_dom, 0.0)
        + max(xrp_dom, 0.0)
    )
    dominance_feed_ok = dominance_sum > 0.01
    mcap_feed_ok = total_mcap > 0

    btc_dom_display = btc_dom if dominance_feed_ok else None
    eth_dom_display = eth_dom if dominance_feed_ok else None

    # Compute percentage change for market cap
    delta_mcap = float(mcap_24h_pct) if pd.notna(mcap_24h_pct) and mcap_feed_ok else float("nan")

    # Price changes come from the same provider as price in top snapshot.
    btc_change = top_snapshot.get("btc_change")
    eth_change = top_snapshot.get("eth_change")

    # Display headline and subtitle
    st.markdown("<h1 class='title'>Crypto Market Intelligence Hub</h1>", unsafe_allow_html=True)
    st.markdown(
        f"<div class='panel-box'>"
        f"<b class='market-intro-title' style='color:{ACCENT};'>What does this tab show?</b>"
        f"<p class='market-intro-body' style='color:{TEXT_MUTED};'>"
        f"Your market overview dashboard. Shows live BTC/ETH prices, total market cap, "
        f"{_tip('Fear & Greed Index', 'A 0-100 score measuring market sentiment. 0 = Extreme Fear (potential accumulation zone), 100 = Extreme Greed (potential distribution zone). Based on volatility, volume, social media, and surveys.')} "
        f"and {_tip('BTC Dominance', 'Percentage of total crypto market cap that belongs to Bitcoin. Rising dominance = money flowing into BTC (risk-off). Falling = altcoin season.')}. "
        f"Top coins are dynamically selected by 24h volume from CoinGecko and scored with real-time technical signals.</p>"
        f"</div>",
        unsafe_allow_html=True,
    )

    # Determine which timeframe to use for market bias gauges. We rely on
    # Streamlit session state to persist the selected timeframe from the
    # scanner controls. On first render, default to 1h. Bias is computed
    # from a six-asset major bundle (BTC/ETH/BNB/SOL/ADA/XRP) on 500
    # candles, using dominance weights when available and equal-weight
    # fallback when dominance feed is unavailable.
    selected_timeframe = st.session_state.get("market_timeframe", "1h")
    # Top row: Price and market cap metrics.
    m1, m2, m3, m4 = st.columns(4, gap="medium")
    # Bitcoin price
    with m1:
        if btc_price is not None and btc_price > 0:
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
        else:
            st.markdown(
                f"<div class='metric-card'>"
                f"  <div class='metric-label'>Bitcoin Price</div>"
                f"  <div class='metric-value'>N/A</div>"
                f"  <div style='color:{TEXT_MUTED};font-size:0.85rem;'>Data unavailable</div>"
                f"</div>",
                unsafe_allow_html=True,
            )
    # Ethereum price
    with m2:
        if eth_price is not None and eth_price > 0:
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
        else:
            st.markdown(
                f"<div class='metric-card'>"
                f"  <div class='metric-label'>Ethereum Price</div>"
                f"  <div class='metric-value'>N/A</div>"
                f"  <div style='color:{TEXT_MUTED};font-size:0.85rem;'>Data unavailable</div>"
                f"</div>",
                unsafe_allow_html=True,
            )
    # Total market cap
    with m3:
        if mcap_feed_ok:
            delta_class = "metric-delta-positive" if delta_mcap >= 0 else "metric-delta-negative"
            delta_text = f"({delta_mcap:+.2f}%)" if pd.notna(delta_mcap) else ""
            st.markdown(
                f"<div class='metric-card'>"
                f"  <div class='metric-label'>Total Market Cap</div>"
                f"  <div class='metric-value'>${total_mcap / 1e12:.2f}T</div>"
                f"  <div class='{delta_class}'>{delta_text}</div>"
                f"</div>",
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f"<div class='metric-card'>"
                f"  <div class='metric-label'>Total Market Cap</div>"
                f"  <div class='metric-value'>N/A</div>"
                f"  <div style='color:{TEXT_MUTED};font-size:0.85rem;'>Data unavailable</div>"
                f"</div>",
                unsafe_allow_html=True,
            )
    # Fear & Greed index
    with m4:
        sentiment_color = POSITIVE if "Greed" in fg_label else (NEGATIVE if "Fear" in fg_label else WARNING)
        fg_value_display = f"{int(fg_value)}" if fg_available else "N/A"
        fg_label_display = fg_label if fg_available else "Unavailable"
        st.markdown(
            f"<div class='metric-card'>"
            f"  <div class='metric-label'>Fear &amp; Greed "
            f"<span title='Crypto Fear &amp; Greed Index (0-100). "
            f"0-25 = Extreme Fear (potential accumulation zone), 75-100 = Extreme Greed (potential distribution zone).' "
            f"style='cursor:help; font-size:0.7rem;'>ℹ️</span></div>"
            f"  <div class='metric-value'>{fg_value_display}</div>"
            f"  <div style='color:{sentiment_color};font-size:0.9rem;'>{fg_label_display}</div>"
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
        # Compute a weighted probability across all assets.  Prefer market-cap
        # dominance weights; if dominance enrichment is unavailable, fall back
        # to equal weights so the score remains usable in exchange-only mode.
        dominance_weights = [
            max(float(btc_dom), 0.0),
            max(float(eth_dom), 0.0),
            max(float(bnb_dom), 0.0),
            max(float(sol_dom), 0.0),
            max(float(ada_dom), 0.0),
            max(float(xrp_dom), 0.0),
        ]
        dom_sum = float(sum(dominance_weights))
        if dom_sum > 0.01:
            weights = [w / dom_sum for w in dominance_weights]
            behaviour_weight_mode = "dominance"
        else:
            weights = [1.0 / 6.0] * 6
            behaviour_weight_mode = "equal"
        behaviour_prob = (
            btc_prob * weights[0]
            + eth_prob * weights[1]
            + bnb_prob * weights[2]
            + sol_prob * weights[3]
            + ada_prob * weights[4]
            + xrp_prob * weights[5]
        )
    except Exception as e:
        _debug(f"AI market-bias fallback to neutral: {e.__class__.__name__}: {str(e).strip()}")
        behaviour_prob = 0.5
        behaviour_weight_mode = "equal"
    behaviour_prob = float(max(0.0, min(1.0, behaviour_prob)))
    major_probs = [btc_prob, eth_prob, bnb_prob, sol_prob, ada_prob, xrp_prob]
    # Determine behaviour direction from the combined probability
    behaviour_dir = direction_from_prob(float(behaviour_prob))
    # Map behaviour direction to a label for display and choose colour.  We
    # reuse the POSITIVE/NEGATIVE/WARNING colours defined above.
    behaviour_side = direction_key(behaviour_dir)
    if behaviour_side == "UPSIDE":
        behaviour_label = "Upside"
        behaviour_color = POSITIVE
    elif behaviour_side == "DOWNSIDE":
        behaviour_label = "Downside"
        behaviour_color = NEGATIVE
    else:
        behaviour_label = "Neutral"
        behaviour_color = WARNING

    # Composite market score (0-100): Direction + Regime + Breadth + Trust
    direction_score = float(max(0.0, min(100.0, abs(float(behaviour_prob) - 0.5) * 200.0)))
    major_upsides = sum(1 for p in major_probs if float(p) >= AI_LONG_THRESHOLD)
    major_downsides = sum(1 for p in major_probs if float(p) <= AI_SHORT_THRESHOLD)
    breadth_score = float(max(major_upsides, major_downsides) / max(len(major_probs), 1) * 100.0)

    if mcap_feed_ok and pd.notna(delta_mcap):
        mcap_chg = abs(float(delta_mcap))
        # Continuous regime scoring to avoid jumpy mode transitions.
        if mcap_chg <= 1.5:
            regime_score = 72.0 + (mcap_chg / 1.5) * 10.0
        elif mcap_chg <= 4.0:
            regime_score = 82.0 - ((mcap_chg - 1.5) / 2.5) * 24.0
        else:
            regime_score = 58.0 - (mcap_chg - 4.0) * 4.0
        regime_score = float(max(38.0, min(90.0, regime_score)))
        regime_score_fallback = False
    else:
        # Neutral fallback when market-cap regime input is unavailable.
        regime_score = 50.0
        regime_score_fallback = True

    try:
        spread = float(pd.Series(major_probs).std())
    except Exception as e:
        _debug(f"Trust-score spread fallback used: {e.__class__.__name__}: {str(e).strip()}")
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
    ai_bias_tip = (
        "Dominance-weighted ML direction across BTC/ETH/BNB/SOL/ADA/XRP. "
        "Direction signal only."
    )
    if behaviour_weight_mode == "equal":
        ai_bias_tip += " Dominance feed unavailable: equal-weight fallback is active."

    def _score_tone(v: float) -> tuple[str, str]:
        x = float(v)
        if x >= 68:
            return ("Strong", POSITIVE)
        if x >= 52:
            return ("Moderate", WARNING)
        return ("Weak", NEGATIVE)

    def _chip_center(label: str, tone_color: str, tip_text: str) -> str:
        return (
            f"<div class='market-gauge-chip-wrap'>"
            f"<span class='market-gauge-chip' style='border:1px solid {tone_color}; color:{tone_color}; "
            f"background:rgba(255,255,255,0.04);'>"
            f"{label} {_tip('', tip_text)}</span></div>"
        )

    def _dom_state(v: float | None, low_cut: float, high_cut: float) -> tuple[str, str]:
        if v is None or pd.isna(v):
            return ("N/A", TEXT_MUTED)
        x = float(v)
        if x >= high_cut:
            return ("High", POSITIVE)
        if x >= low_cut:
            return ("Balanced", WARNING)
        return ("Low", NEGATIVE)

    g1, g2, g3, g4 = st.columns(4, gap="medium")
    GAUGE_DOMAIN = {"x": [0.06, 0.94], "y": [0, 1]}
    GAUGE_HEIGHT = 186
    GAUGE_MARGIN = dict(l=12, r=12, t=52, b=10)
    GAUGE_NUMBER_Y = 0.16

    def _render_market_gauge(
        *,
        title: str,
        value: float | None,
        steps: list[dict[str, float | str]],
        value_text: str,
        title_hover: str | None = None,
    ) -> None:
        title_text = html.escape(title)
        if title_hover:
            title_html = (
                "<div style='text-align:center; margin-bottom:2px;'>"
                f"<span title='{html.escape(title_hover)}' style='color:#E5E7EB; font-size:13px; cursor:help;'>"
                f"{title_text}</span></div>"
            )
        else:
            title_html = (
                "<div style='text-align:center; margin-bottom:2px;'>"
                f"<span style='color:#E5E7EB; font-size:13px;'>{title_text}</span></div>"
            )
        st.markdown(title_html, unsafe_allow_html=True)

        if value is None or pd.isna(value):
            st.markdown(
                f"<div class='metric-card' style='height:{GAUGE_HEIGHT}px; display:flex; flex-direction:column; "
                f"justify-content:center; align-items:center;'>"
                f"<div class='metric-label'>{title_text}</div>"
                f"<div class='metric-value'>N/A</div>"
                f"<div style='color:{TEXT_MUTED}; font-size:0.85rem;'>Data unavailable</div>"
                f"</div>",
                unsafe_allow_html=True,
            )
            return

        safe_val = float(max(0.0, min(100.0, float(value))))
        fig = go.Figure(
            go.Indicator(
                mode="gauge",
                value=safe_val,
                domain=GAUGE_DOMAIN,
                gauge={
                    "axis": {
                        "range": [0, 100],
                        "tickwidth": 1,
                        "tickvals": [0, 50, 100],
                        "tickfont": {"size": 12, "color": TEXT_MUTED},
                    },
                    "bar": {"color": ACCENT},
                    "bgcolor": CARD_BG,
                    "steps": steps,
                },
                title={"text": "", "font": {"size": 13, "color": "#E5E7EB"}},
            )
        )
        fig.update_layout(
            height=GAUGE_HEIGHT,
            margin=GAUGE_MARGIN,
            plot_bgcolor="#000000",
            paper_bgcolor="#000000",
        )
        fig.add_annotation(
            x=0.5,
            y=GAUGE_NUMBER_Y,
            xref="paper",
            yref="paper",
            text=html.escape(value_text),
            showarrow=False,
            yanchor="middle",
            font={"color": "#F8FAFC", "size": 34},
        )
        st.plotly_chart(fig, width="stretch")

    # BTC dominance gauge
    with g1:
        _render_market_gauge(
            title="BTC Dominance (%)",
            value=btc_dom_display,
            steps=[
                {"range": [0, AI_SHORT_THRESHOLD * 100], "color": NEGATIVE},
                {"range": [AI_SHORT_THRESHOLD * 100, AI_LONG_THRESHOLD * 100], "color": WARNING},
                {"range": [AI_LONG_THRESHOLD * 100, 100], "color": POSITIVE},
            ],
            value_text=f"{float(btc_dom_display):.1f}" if btc_dom_display is not None else "N/A",
        )
        btc_state, btc_color = _dom_state(btc_dom_display, AI_SHORT_THRESHOLD * 100, AI_LONG_THRESHOLD * 100)
        st.markdown(
            _chip_center("BTC Weight: " + btc_state, btc_color, "Bitcoin share of total market cap. High values usually indicate BTC-led market."),
            unsafe_allow_html=True,
        )

    # ETH dominance gauge
    with g2:
        _render_market_gauge(
            title="ETH Dominance (%)",
            value=eth_dom_display,
            steps=[
                {"range": [0, 15], "color": NEGATIVE},
                {"range": [15, 25], "color": WARNING},
                {"range": [25, 100], "color": POSITIVE},
            ],
            value_text=f"{float(eth_dom_display):.1f}" if eth_dom_display is not None else "N/A",
        )
        eth_state, eth_color = _dom_state(eth_dom_display, 15.0, 25.0)
        st.markdown(
            _chip_center("ETH Weight: " + eth_state, eth_color, "Ethereum share of total market cap. Higher values show stronger ETH participation."),
            unsafe_allow_html=True,
        )

    # AI direction bias gauge
    with g3:
        ai_direction_hover = (
            "AI Direction Bias is the market-direction score from BTC/ETH/BNB/SOL/ADA/XRP model outputs. "
            "If dominance data is available, it uses dominance weighting; otherwise equal weighting fallback is used. "
            f"Score zones: 0-{int(AI_SHORT_THRESHOLD * 100)} = Downside bias, "
            f"{int(AI_SHORT_THRESHOLD * 100)}-{int(AI_LONG_THRESHOLD * 100)} = Neutral bias, "
            f"{int(AI_LONG_THRESHOLD * 100)}-100 = Upside bias."
        )
        if behaviour_weight_mode == "equal":
            ai_direction_hover += " Dominance feed unavailable right now, so equal-weight fallback is active."
        ai_direction_score = int(round(behaviour_prob * 100))
        _render_market_gauge(
            title="AI Direction Bias (%)",
            value=float(ai_direction_score),
            steps=[
                {"range": [0, AI_SHORT_THRESHOLD * 100], "color": NEGATIVE},
                {"range": [AI_SHORT_THRESHOLD * 100, AI_LONG_THRESHOLD * 100], "color": WARNING},
                {"range": [AI_LONG_THRESHOLD * 100, 100], "color": POSITIVE},
            ],
            value_text=f"{ai_direction_score:d}",
            title_hover=ai_direction_hover,
        )
        st.markdown(
            _chip_center(
                f"{behaviour_label} Bias",
                behaviour_color,
                ai_bias_tip,
            ),
            unsafe_allow_html=True,
        )

    # Setup quality gauge (composite)
    with g4:
        setup_quality_hover = (
            "Setup Quality formula: 35% Direction + 20% Regime + 25% Breadth + 20% Trust. "
            "Direction = AI Direction Bias strength, Regime = market-cap environment quality, "
            "Breadth = major-asset participation on one side, Trust = cross-major model consistency. "
            "This score measures market environment quality, not trade direction alone."
        )
        setup_mode_hover = {
            "Risk-On": (
                "Risk-On: broad market conditions are favorable. "
                "Direction, participation, and model trust are strong enough for active setup hunting."
            ),
            "Selective": (
                "Selective: market quality is mixed. "
                "Some setups can work, but confirmation standards should stay strict and risk should stay controlled."
            ),
            "Risk-Off": (
                "Risk-Off: market environment is weak or fragmented. "
                "Favor capital preservation and avoid forcing setups."
            ),
        }.get(composite_mode, setup_quality_hover)
        _render_market_gauge(
            title="Setup Quality (%)",
            value=float(composite_score),
            steps=[
                {"range": [0, 52], "color": NEGATIVE},
                {"range": [52, 68], "color": WARNING},
                {"range": [68, 100], "color": POSITIVE},
            ],
            value_text=f"{int(round(composite_score)):d}",
            title_hover=setup_quality_hover,
        )
        st.markdown(
            _chip_center(
                composite_mode,
                composite_color,
                setup_mode_hover,
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
            f"<span title='Regime: market environment quality proxy from total market-cap move behavior."
            f"{' Market-cap feed unavailable, neutral fallback (50) active.' if regime_score_fallback else ''}' "
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
        f"<div class='market-section-title' style='color:{ACCENT};'>Coin Setup Scanner</div>",
        unsafe_allow_html=True,
    )
    st.markdown(
        """
        <style>
        div[data-testid="stButton"] > button {
          white-space: nowrap;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    def _parse_custom_bases(raw: str, limit: int = 10) -> list[str]:
        tokens = re.split(r"[\s,;\n]+", str(raw or "").upper().strip())
        out: list[str] = []
        seen: set[str] = set()
        for tok in tokens:
            t = tok.strip()
            if not t:
                continue
            if "/" in t:
                t = t.split("/", 1)[0].strip()
            for suf in ("-USDT", "_USDT", "USDT", "-USD", "_USD", "USD"):
                if t.endswith(suf) and len(t) > len(suf):
                    t = t[: -len(suf)]
                    break
            t = re.sub(r"[^A-Z0-9]", "", t)
            if len(t) < 2 or len(t) > 15:
                continue
            if t in seen:
                continue
            seen.add(t)
            out.append(t)
            if len(out) >= int(limit):
                break
        return out

    custom_bases_applied = list(st.session_state.get("market_custom_bases_applied", []))
    custom_mode_active = bool(custom_bases_applied)

    controls = st.columns([1.08, 1.18, 0.90, 1.52, 0.92], gap="medium")
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
        direction_filter = st.selectbox(
            "Direction",
            ['Upside', 'Downside', 'Both'],
            index=2,
            format_func=lambda x: "All Directions" if x == "Both" else x,
        )
    with controls[2]:
        top_n_default = int(st.session_state.get("market_top_n", 50))
        top_n = st.slider(
            "Top N",
            min_value=3,
            max_value=50,
            value=top_n_default,
            key="market_top_n",
            disabled=custom_mode_active,
        )
    with controls[3]:
        custom_coin_input = st.text_input(
            "Custom Coins (max 10)",
            value=st.session_state.get("market_custom_coin_input", ""),
            key="market_custom_coin_input",
            placeholder="BTC, ETH, SOL",
            help="Optional watchlist mode. Enter up to 10 symbols separated by comma.",
        )
    with controls[4]:
        run_scan = st.button("Run Scan", width="stretch")
        clear_custom = st.button(
            "Clear Custom",
            width="stretch",
            disabled=not bool(st.session_state.get("market_custom_bases_applied", [])),
            key="market_clear_custom",
        )

    custom_bases_draft = _parse_custom_bases(custom_coin_input, limit=10)
    if run_scan:
        st.session_state["market_custom_bases_applied"] = custom_bases_draft
        custom_bases_applied = list(custom_bases_draft)
        custom_mode_active = bool(custom_bases_applied)
    if clear_custom:
        st.session_state["market_custom_coin_input"] = ""
        st.session_state["market_custom_bases_applied"] = []
        st.rerun()

    if custom_mode_active:
        preview = ", ".join(custom_bases_applied[:6])
        more = "" if len(custom_bases_applied) <= 6 else f" +{len(custom_bases_applied) - 6}"
        st.markdown(
            f"<div class='market-note-box' style='border:1px solid rgba(0,212,255,0.34); border-left:4px solid {ACCENT}; "
            f"background:rgba(0,212,255,0.06); color:{TEXT_MUTED}; margin-top:0.25rem;'>"
            f"<b style='color:{ACCENT};'>Custom Watchlist Mode:</b> scanning {len(custom_bases_applied)} coin(s): "
            f"{preview}{more}. Top N is disabled while custom mode is active."
            f"</div>",
            unsafe_allow_html=True,
        )
    elif custom_coin_input.strip() and custom_bases_draft:
        st.caption("Custom symbols are ready. Click Run Scan to apply watchlist mode.")

    exclude_stables = st.checkbox(
        "Exclude stablecoins",
        value=True,
        key="market_exclude_stables",
        help="Hide stable/synthetic USD-pegged coins from scanner universe.",
    )
    CACHE_TTL_MINUTES = 15
    gate_min_rr, gate_min_adx, gate_min_strength = scalp_gate_thresholds(timeframe)

    STABLE_BASES = {
        "USDT", "USDC", "BUSD", "DAI", "TUSD", "USDE", "FDUSD", "PYUSD",
        "RLUSD", "USDP", "GUSD", "EURS", "EURC",
    }

    def _is_stable_base(base: str) -> bool:
        b = str(base or "").upper().strip()
        return bool(b and b in STABLE_BASES)

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

    def _compact_adx_label(v: object) -> str:
        raw = str(v or "").strip()
        if not raw or raw.upper() in {"N/A", "NA", "NAN", "UNAVAILABLE"}:
            return ""
        # format_adx output examples:
        # "▼ 17.9 (Weak)", "→ 23.4 (Starting)", "▲ 34.8 (Strong)",
        # "▲▲ 58.1 (Very Strong)", "🔥 79.3 (Extreme)"
        m = re.search(r"\(([^)]+)\)", raw)
        if not m:
            return raw
        bucket = m.group(1).strip()
        if raw.startswith("▲▲"):
            return f"▲▲ {bucket}"
        if raw.startswith("▲"):
            return f"▲ {bucket}"
        if raw.startswith("▼"):
            return f"▼ {bucket}"
        if raw.startswith("→"):
            return f"→ {bucket}"
        if raw.startswith("🔥"):
            return f"🔥 {bucket}"
        return bucket

    def _rr_badge(rr_val: float) -> str:
        if rr_val <= 0:
            return ""
        return f"{rr_val:.2f}"

    def _strength_badge(bias: float) -> str:
        strength = float(strength_from_bias(bias))
        label = strength_bucket(strength)
        return f"{strength:.0f}% ({label})"

    def _setup_confirm_display(raw_action: str) -> str:
        cls = normalize_action_class(raw_action)
        if cls == "ENTER_TREND_AI":
            return "TREND+AI"
        if cls == "ENTER_TREND_LED":
            return "TREND-led"
        if cls == "ENTER_AI_LED":
            return "AI-led"
        if cls == "WATCH":
            return "WATCH"
        if cls == "SKIP":
            return "SKIP"
        return str(raw_action or "").strip()
    def _setup_confirm_class(value: str) -> str:
        s = str(value or "").strip().upper()
        if "TREND+AI" in s:
            return "ENTER_TREND_AI"
        if s == "TREND" or s == "TREND-LED" or "TREND-LED" in s:
            return "ENTER_TREND_LED"
        if s == "AI" or s == "AI-LED" or "AI-LED" in s:
            return "ENTER_AI_LED"
        return normalize_action_class(s)

    def _setup_confirm_rank(value: str) -> int:
        cls = _setup_confirm_class(value)
        if cls.startswith("ENTER_"):
            return 3
        if cls == "WATCH":
            return 2
        if cls == "SKIP":
            return 1
        return 0

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
                "ENTER", "UPSIDE", "ALIGNED", "GOOD", "STRONG", "VERY STRONG", "EXTREME",
                "ABOVE", "BULLISH", "OVERSOLD", "NEAR BOTTOM",
            ]
        ):
            return "pos"
        if any(
            k in s
            for k in [
                "SKIP", "DOWNSIDE", "CONFLICT", "WEAK", "BEARISH",
                "OVERBOUGHT", "BELOW", "NEAR TOP",
            ]
        ):
            return "neg"
        if any(k in s for k in ["WATCH", "WAIT", "MIXED", "EARLY", "TREND", "NEUTRAL", "MEDIUM", "STARTING", "MODERATE", "SPIKE"]):
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

        if col == "Setup Confirm":
            cls = _setup_confirm_class(s)
            if cls in {"ENTER_TREND_AI", "ENTER_TREND_LED", "ENTER_AI_LED"}:
                return "pos"
            if cls == "WATCH":
                return "warn"
            if cls == "SKIP":
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
            if "CONFLICT" in s:
                return "neg"
            if "HIGH" in s:
                return "pos"
            if "MEDIUM" in s:
                return "warn"
            if "TREND" in s:
                return "warn"
            if "WEAK" in s:
                return "warn"
            return "muted"

        if col == "R:R":
            try:
                rr = float(
                    s.replace("🟢", "")
                    .replace("🟡", "")
                    .replace("🔴", "")
                    .strip()
                )
                if rr >= 2.0:
                    return "pos"
                if rr >= 1.5:
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
            return "warn" if "SPIKE" in s else "muted"

        if col == "ADX":
            if "EXTREME" in s or "VERY STRONG" in s or "STRONG" in s:
                return "pos"
            if "STARTING" in s:
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
            if s.startswith("▲"):
                return "pos"
            if s.startswith("▼"):
                return "neg"
            if s.startswith("→") or s.startswith("-"):
                return "warn"
            # Compatibility fallback when arrow prefixes are missing.
            if "BULLISH" in s:
                return "pos"
            if "BEARISH" in s:
                return "neg"
            if "NEUTRAL" in s or "INDECISION" in s:
                return "warn"
            return "warn"

        return _tone_for_text(s)

    def _chip(
        text: object,
        tone: str | None = None,
        title: str | None = None,
        extra_class: str = "",
    ) -> str:
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
        title_attr = f" title='{html.escape(title)}'" if title else ""
        cls_full = f"mk-chip {cls} {extra_class}".strip()
        return f"<span class='{cls_full}'{title_attr}>{html.escape(raw)}</span>"

    def _render_cell(col: str, row: dict) -> str:
        val = row.get(col, "")
        txt = "" if val is None else str(val).strip()
        if txt.upper() in {"N/A", "NA", "NAN", "UNAVAILABLE", "-"}:
            txt = ""
        if col == "Coin":
            pair = str(row.get("__pair", "")).strip()
            if pair:
                return (
                    f"<span class='mk-coin-wrap'>"
                    f"<span class='mk-coin'>{html.escape(txt)}</span>"
                    f"<span class='mk-coin-tooltip'>{html.escape(pair)}</span>"
                    f"</span>"
                )
            return f"<span class='mk-coin'>{html.escape(txt)}</span>"
        if col in {"Setup Confirm", "Direction", "Strength", "R:R", "Scalp Opportunity"}:
            if col == "Setup Confirm":
                reason_code = str(row.get("__action_reason", "")).strip()
                reason_text = action_reason_text(reason_code)
                raw_action = str(row.get("__action_raw", txt))
                display_txt = _setup_confirm_display(raw_action)
                sc_cls = _setup_confirm_class(raw_action or txt)
                extra_cls = "mk-chip-action"
                if sc_cls == "ENTER_TREND_LED":
                    extra_cls += " mk-sc-trend-led"
                elif sc_cls == "ENTER_AI_LED":
                    extra_cls += " mk-sc-ai-led"
                title_txt = display_txt if not reason_text else f"{display_txt} | Reason: {reason_text}"
                return _chip(
                    display_txt,
                    _tone_for_col(col, raw_action or txt),
                    title=title_txt,
                    extra_class=extra_cls,
                )
            if col == "Direction":
                direction_note = str(row.get("__direction_note", "")).strip()
                return _chip(txt, _tone_for_col(col, txt), title=direction_note or None)
            if col == "Scalp Opportunity" and txt.upper() == "NEUTRAL":
                return ""
            if col == "Strength":
                strength_note = str(row.get("__strength_note", "")).strip()
                return _chip(txt, _tone_for_col(col, txt), title=strength_note or None)
            if col == "R:R":
                rr_note = str(row.get("__rr_note", "")).strip()
                rr_text = txt
                if rr_note:
                    rr_text = f"{txt}*"
                return _chip(rr_text, _tone_for_col(col, txt), title=rr_note or None)
            return _chip(txt, _tone_for_col(col, txt))
        if col == "Tech vs AI Alignment":
            align_note = str(row.get("__alignment_note", "")).strip()
            return _chip(txt, _tone_for_col(col, txt), title=align_note or None)
        if col == "AI Ensemble":
            t = _tone_for_col(col, txt)
            ai_note = str(row.get("__ai_note", "")).strip()
            base_txt = re.sub(r"\s*\(\s*\d+\s*/\s*3\s*\)\s*$", "", txt).strip() or txt
            votes = row.get("__ai_votes")
            try:
                votes_n = int(votes)
            except Exception:
                m = re.search(r"\((\d)\s*/\s*3\)", txt)
                votes_n = int(m.group(1)) if m else 0
            votes_n = max(0, min(3, votes_n))
            tone_map = {
                "pos": "mk-pos",
                "neg": "mk-neg",
                "warn": "mk-warn",
                "muted": "mk-muted",
                "info": "mk-info",
            }
            tone_cls = tone_map.get(t, "mk-muted")
            title_attr = f" title='{html.escape(ai_note)}'" if ai_note else ""
            dots_html = "".join(
                f"<span class='mk-ai-dot{' is-filled' if i < votes_n else ''}'></span>"
                for i in range(3)
            )
            return (
                f"<span class='mk-chip {tone_cls} mk-chip-ai'{title_attr}>"
                f"<span class='mk-ai-text'>{html.escape(base_txt)}</span>"
                f"<span class='mk-ai-dots'>{dots_html}</span>"
                f"</span>"
            )
        if col == "Spike Alert":
            if not txt:
                return ""
            spike_dir = str(row.get("__spike_dir", "")).upper()
            if spike_dir == "UP":
                spike_tone = "pos"
                spike_label = "▲ Up Spike"
            elif spike_dir == "DOWN":
                spike_tone = "neg"
                spike_label = "▼ Down Spike"
            else:
                spike_tone = "warn"
                spike_label = "→ Spike"
            detail_parts: list[str] = []
            spike_vol_ratio = row.get("__spike_vol_ratio")
            spike_candle_pct = row.get("__spike_candle_pct")
            spike_vwap_ctx = str(row.get("__spike_vwap_ctx", "")).strip()
            try:
                if pd.notna(spike_vol_ratio):
                    detail_parts.append(f"Vol Ratio: {float(spike_vol_ratio):.2f}x")
            except Exception:
                pass
            try:
                if pd.notna(spike_candle_pct):
                    detail_parts.append(f"Candle: {float(spike_candle_pct):+,.2f}%")
            except Exception:
                pass
            if spike_vwap_ctx:
                detail_parts.append(f"VWAP: {spike_vwap_ctx}")
            detail_title = " | ".join(detail_parts) if detail_parts else "Volume anomaly detected"
            return _chip(spike_label, spike_tone, title=detail_title)
        if col == "Δ (%)":
            if not txt:
                return ""
            delta_note = str(row.get("__delta_note", "")).strip()
            title_attr = f" title='{html.escape(delta_note, quote=True)}'" if delta_note else ""
            if txt.startswith("▲"):
                return f"<span class='mk-delta mk-pos-t'{title_attr}>{html.escape(txt)}</span>"
            if txt.startswith("▼"):
                return f"<span class='mk-delta mk-neg-t'{title_attr}>{html.escape(txt)}</span>"
            return f"<span class='mk-delta mk-muted-t'{title_attr}>{html.escape(txt)}</span>"
        if col == "Price ($)":
            if not txt:
                return ""
            plain = txt[1:] if txt.startswith("$") else txt
            return f"<span class='mk-plain'>{html.escape(plain)}</span>"
        if col == "Entry Price":
            if not txt:
                return ""
            entry_note = str(row.get("__entry_note", "")).strip()
            if entry_note:
                return (
                    f"<span class='mk-plain' title='{html.escape(entry_note, quote=True)}'>"
                    f"{html.escape(txt)}</span>"
                )
            return f"<span class='mk-plain'>{html.escape(txt)}</span>"
        if col == "Stop Loss":
            return f"<span class='mk-plain'>{html.escape(txt)}</span>" if txt else ""
        if col == "Target Price":
            if not txt:
                return ""
            target_note = str(row.get("__target_note", "")).strip()
            if target_note:
                return (
                    f"<span class='mk-plain' title='{html.escape(target_note, quote=True)}'>"
                    f"{html.escape(txt)}</span>"
                )
            return f"<span class='mk-plain'>{html.escape(txt)}</span>"
        if col == "Ichimoku":
            ichi_title = str(row.get("__ichimoku_detail", "")).strip()
            return _chip(txt, _tone_for_col(col, txt), title=ichi_title or None) if txt else ""
        if col == "ADX":
            adx_raw = row.get("__adx_raw")
            adx_title = None
            try:
                if pd.notna(adx_raw):
                    adx_f = float(adx_raw)
                    gate_txt = (
                        f"PASS (>={gate_min_adx:.0f})"
                        if adx_f >= float(gate_min_adx)
                        else f"LOW (<{gate_min_adx:.0f})"
                    )
                    adx_title = f"ADX {adx_f:.1f} | Scalp trend gate: {gate_txt}"
            except Exception:
                adx_title = None
            return _chip(txt, _tone_for_col(col, txt), title=adx_title) if txt else ""
        if col in {"SuperTrend", "Ichimoku", "VWAP", "Bollinger", "Stochastic RSI", "Volatility", "PSAR", "Williams %R", "CCI", "Candle Pattern"}:
            return _chip(txt, _tone_for_col(col, txt)) if txt else ""
        return f"<span class='mk-plain'>{html.escape(txt)}</span>"

    def _render_pro_table(df: pd.DataFrame, cols: list[str]) -> None:
        sticky_order: list[str] = ["Coin"]
        col_widths = {
            "Coin": 120,
            "Price ($)": 122,
            "Δ (%)": 92,
            "Setup Confirm": 160,
            "Direction": 130,
            "Strength": 132,
            "AI Ensemble": 170,
            "Tech vs AI Alignment": 190,
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
                cell_class = ""
                if c in col_widths:
                    w = col_widths[c]
                    width_style = f"min-width:{w}px; max-width:{w}px; width:{w}px;"
                if c in sticky_cols:
                    sticky = (
                        f"position:sticky; left:{left_offsets[c]}; z-index:6; "
                        f"background:rgba(8,12,20,1.0); box-shadow:1px 0 0 rgba(148,163,184,0.22), 2px 0 10px rgba(0,0,0,0.24);"
                    )
                if c == "Coin":
                    cell_class = " class='mk-coin-cell'"
                cell_html.append(f"<td{cell_class} style='{width_style}{sticky}'>{_render_cell(c, row_dict)}</td>")
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
              overflow:hidden;
              text-overflow:ellipsis;
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
              max-width:100%;
              border-radius:999px;
              border:1px solid rgba(148,163,184,0.34);
              background:rgba(148,163,184,0.10);
              font-size:0.74rem;
              font-weight:700;
              overflow:hidden;
              text-overflow:ellipsis;
              white-space:nowrap;
              box-sizing:border-box;
            }}
            .mk-chip-action {{
              font-size:0.70rem;
              padding:2px 6px;
              letter-spacing:0.1px;
              gap:4px;
            }}
            .mk-chip-ai {{
              gap:8px;
              min-height:26px;
              padding:2px 8px;
            }}
            .mk-ai-text {{
              line-height:1.15;
            }}
            .mk-ai-dots {{
              display:inline-flex;
              align-items:center;
              gap:3px;
              min-height:12px;
            }}
            .mk-ai-dot {{
              width:8px;
              height:8px;
              border-radius:999px;
              border:1px solid currentColor;
              background:transparent;
              opacity:0.55;
              flex:0 0 8px;
            }}
            .mk-ai-dot.is-filled {{
              background:currentColor;
              opacity:1;
            }}
            .mk-sc-trend-led {{
              color:#38BDF8 !important;
              border-color:rgba(56,189,248,0.52) !important;
              background:rgba(56,189,248,0.12) !important;
            }}
            .mk-sc-ai-led {{
              color:#22D3EE !important;
              border-color:rgba(34,211,238,0.52) !important;
              background:rgba(34,211,238,0.12) !important;
            }}
            .mk-pos {{ color:{POSITIVE}; border-color:rgba(0,255,136,0.42); background:rgba(0,255,136,0.10); }}
            .mk-neg {{ color:{NEGATIVE}; border-color:rgba(255,51,102,0.44); background:rgba(255,51,102,0.10); }}
            .mk-warn {{ color:{WARNING}; border-color:rgba(255,209,102,0.46); background:rgba(255,209,102,0.10); }}
            .mk-info {{ color:{ACCENT}; border-color:rgba(0,212,255,0.46); background:rgba(0,212,255,0.10); }}
            .mk-muted {{ color:{TEXT_MUTED}; border-color:rgba(140,161,182,0.35); background:rgba(140,161,182,0.08); }}
            .mk-coin {{ font-weight:800; letter-spacing:0.2px; color:#F8FAFC; }}
            .mk-coin-wrap {{
              position:relative;
              display:inline-flex;
              align-items:center;
            }}
            .mk-table td.mk-coin-cell {{
              overflow:visible !important;
              position:relative;
            }}
            .mk-coin-tooltip {{
              position:absolute;
              left:calc(100% + 8px);
              top:50%;
              transform:translateY(-50%);
              z-index:40;
              opacity:0;
              visibility:hidden;
              transition:opacity 0.14s ease, visibility 0.14s ease;
              pointer-events:none;
              white-space:nowrap;
              border:1px solid rgba(0,212,255,0.40);
              background:rgba(6,12,24,0.96);
              color:#D6E8FF;
              font-size:0.70rem;
              font-weight:700;
              line-height:1.2;
              border-radius:8px;
              padding:4px 8px;
              box-shadow:0 8px 20px rgba(0,0,0,0.35);
            }}
            .mk-coin-wrap:hover .mk-coin-tooltip {{
              opacity:1;
              visibility:visible;
            }}
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

    scan_sig = (
        timeframe,
        direction_filter,
        int(top_n),
        bool(exclude_stables),
        tuple(custom_bases_applied),
    )
    last_sig = st.session_state.get("market_scan_sig")
    should_scan = run_scan or (last_sig != scan_sig) or ("market_scan_results" not in st.session_state)

    results: list[dict] = st.session_state.get("market_scan_results", [])
    source_label = st.session_state.get("market_scan_source", "LIVE")
    data_mode = st.session_state.get("market_data_mode", "FULL MARKET MODE")

    # Fetch top coins
    if should_scan:
        spinner_label = (
            f"Scanning custom watchlist ({len(custom_bases_applied)}) ({direction_filter}) [{timeframe}] ..."
            if custom_mode_active
            else f"Scanning {top_n} coins ({direction_filter}) [{timeframe}] ..."
        )
        with st.spinner(spinner_label):
            universe_fetch_n = max(top_n, 200) if custom_mode_active else max(top_n, 50)
            usdt_symbols, market_data = get_top_volume_usdt_symbols(universe_fetch_n)

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
                raw_mcap = coin.get("market_cap")
                try:
                    if isinstance(raw_mcap, str):
                        raw_mcap = raw_mcap.replace(",", "").strip()
                    mcap_f = float(raw_mcap) if raw_mcap is not None and pd.notna(raw_mcap) else 0.0
                    mcap = int(mcap_f) if pd.notna(mcap_f) and mcap_f > 0 else 0
                except Exception:
                    mcap = 0
                if symbol and (symbol not in mcap_map or mcap > mcap_map[symbol]):
                    mcap_map[symbol] = mcap
            is_exchange_only_mode = len(unique_market_data) == 0

            requested_n = len(custom_bases_applied) if custom_mode_active else int(top_n)

            if custom_mode_active:
                working_symbols = [f"{b}/USDT" for b in custom_bases_applied]
            else:
                # USDT match
                if unique_market_data:
                    valid_bases = {(c.get("symbol") or "").upper() for c in unique_market_data}
                    working_symbols = [s for s in usdt_symbols if s.split("/")[0].upper() in valid_bases]
                else:
                    # If market rows are empty (e.g. CoinGecko rate-limit), keep scan alive
                    # using exchange pairs directly.
                    working_symbols = list(usdt_symbols)

            if exclude_stables:
                working_symbols = [s for s in working_symbols if not _is_stable_base(s.split("/")[0].upper())]

            if custom_mode_active:
                working_symbols = working_symbols[:len(custom_bases_applied)]
            else:
                working_symbols = working_symbols[:top_n]

            if not working_symbols and not custom_mode_active:
                # Hard fallback universe for temporary upstream outages.
                working_symbols = major_fallback_symbols[: min(top_n, len(major_fallback_symbols))]
                if working_symbols:
                    st.info(
                        "Primary market feed is temporarily unavailable. "
                        "Scanner switched to major fallback universe."
                    )
            data_mode = "EXCHANGE-ONLY MODE" if is_exchange_only_mode else "FULL MARKET MODE"
            st.session_state["market_data_mode"] = data_mode

            if not working_symbols:
                if custom_mode_active:
                    st.warning(
                        "Custom watchlist did not produce eligible symbols after normalization/filtering. "
                        "Check symbol spelling (e.g., BTC, ETH, SOL) or disable stablecoin exclusion."
                    )
                else:
                    st.warning(
                        "No scanner symbols matched current market filters. "
                        f"Source pairs: {len(usdt_symbols)}, market rows: {len(unique_market_data)}, "
                        f"requested top_n: {top_n}."
                        )
            elif len(working_symbols) < requested_n:
                if custom_mode_active:
                    st.info(
                        f"Custom mode active: scanning {len(working_symbols)} / {requested_n} requested symbols."
                    )
                else:
                    st.info(
                        f"Liquidity universe currently returned {len(working_symbols)} eligible symbols "
                        f"(requested {top_n}). Scanner remains strict to top-volume matched pairs."
                    )

            # Two-phase scan:
            # 1) Fetch OHLCV with a narrow lock for shared exchange safety.
            # 2) Run analysis/model pipeline in parallel on fetched frames.
            fetch_lock = Lock()

            def _fetch_ohlcv_thread_safe(sym: str) -> pd.DataFrame | None:
                with fetch_lock:
                    return fetch_ohlcv(sym, timeframe, limit=500)

            fetched_frames: list[tuple[str, pd.DataFrame]] = []
            for sym in working_symbols:
                df = _fetch_ohlcv_thread_safe(sym)
                if df is None or len(df) <= 60:
                    continue
                # Align analysis and scalp planning on same closed-candle context.
                df_eval = df.iloc[:-1].copy()
                if df_eval is None or len(df_eval) <= 55:
                    continue
                fetched_frames.append((sym, df_eval))

            def _scan_one(sym: str, df_eval: pd.DataFrame) -> dict | None:
                """Analyse a single symbol for the scanner. Returns a row dict or None."""

                _ai_prob, ai_direction, ai_details = ml_ensemble_predict(df_eval)
                agreement = float(ai_details.get("agreement", 0.0)) if isinstance(ai_details, dict) else 0.0
                directional_agreement = float(ai_details.get("directional_agreement", agreement)) if isinstance(ai_details, dict) else agreement
                consensus_agreement = float(ai_details.get("consensus_agreement", 0.0)) if isinstance(ai_details, dict) else 0.0
                model_votes = list(ai_details.get("model_votes", [])) if isinstance(ai_details, dict) else []
                latest_closed = df_eval.iloc[-1]

                base = sym.split('/')[0].upper()
                mcap_val = mcap_map.get(base)
                # Keep price semantics aligned with all decision metrics (closed-candle context).
                price = float(latest_closed["close"])
                # Delta source of truth: selected-timeframe closed candles.
                # This keeps table delta aligned with Direction/Strength calculations.
                price_change = None
                delta_note = "Source: selected-timeframe closed candles."
                try:
                    prev_close = float(df_eval["close"].iloc[-2])
                    last_closed = float(df_eval["close"].iloc[-1])
                    if pd.notna(prev_close) and prev_close > 0 and pd.notna(last_closed):
                        price_change = ((last_closed / prev_close) - 1.0) * 100.0
                except Exception as e:
                    _debug(f"Delta candle fallback for {sym} ({timeframe}): {e.__class__.__name__}: {str(e).strip()}")
                    price_change = None
                # Safety fallback (rare): if candle delta is unavailable, use ticker percentage.
                if price_change is None:
                    try:
                        # Protect shared exchange ticker fallback under the same lock.
                        with fetch_lock:
                            price_change = get_price_change(sym)
                        if price_change is not None:
                            delta_note = "Fallback source: ticker percentage (closed-candle delta unavailable)."
                    except Exception as e:
                        _debug(f"Ticker delta fallback failed for {sym} ({timeframe}): {e.__class__.__name__}: {str(e).strip()}")
                        price_change = None

                a = analyse(df_eval)
                signal, volume_spike = a.signal, a.volume_spike
                atr_comment_v, candle_pattern_v, bias_score_v = a.atr_comment, a.candle_pattern, a.bias
                adx_val_v, supertrend_trend_v, ichimoku_trend_v = a.adx, a.supertrend, a.ichimoku
                stochrsi_k_val_v, bollinger_bias_v, vwap_label_v = a.stochrsi_k, a.bollinger, a.vwap
                psar_trend_v = a.psar

                spike_dir = ""
                spike_vol_ratio = float("nan")
                spike_candle_pct = float("nan")
                spike_vwap_ctx = str(vwap_label_v or "").replace("🟢 ", "").replace("🔴 ", "").replace("→ ", "").strip()
                if volume_spike:
                    try:
                        prev_vol_avg = float(df_eval["volume"].iloc[-21:-1].mean()) if len(df_eval) >= 21 else float("nan")
                        last_vol = float(df_eval["volume"].iloc[-1]) if len(df_eval) >= 1 else float("nan")
                        if pd.notna(prev_vol_avg) and prev_vol_avg > 0 and pd.notna(last_vol):
                            spike_vol_ratio = last_vol / prev_vol_avg
                    except Exception:
                        spike_vol_ratio = float("nan")
                    try:
                        o = float(latest_closed["open"])
                        c = float(latest_closed["close"])
                        if pd.notna(o) and pd.notna(c) and o > 0:
                            spike_candle_pct = ((c / o) - 1.0) * 100.0
                        if pd.notna(o) and pd.notna(c):
                            if c > o:
                                spike_dir = "UP"
                            elif c < o:
                                spike_dir = "DOWN"
                            else:
                                spike_dir = "NEUTRAL"
                    except Exception:
                        spike_dir = "NEUTRAL"

                # Plan generator produces candidate levels; final execution decision is
                # made by a single external gate (scalp_quality_gate) below.
                scalp_direction, entry_s, target_s, stop_s, rr_ratio, breakout_note = get_scalping_entry_target(
                    df_eval, bias_score_v, supertrend_trend_v, ichimoku_trend_v, vwap_label_v,
                )
                entry_price = entry_s if scalp_direction else 0.0
                target_price = target_s if scalp_direction else 0.0

                signal_direction = direction_key(signal_plain(signal))
                include = (
                    (direction_filter == 'Both')
                    or (direction_filter == 'Upside' and signal_direction == "UPSIDE")
                    or (direction_filter == 'Downside' and signal_direction == "DOWNSIDE")
                )
                if not include:
                    return None

                signal_text = sanitize_trading_terms(signal)
                comment_text = sanitize_trading_terms(str(getattr(a, 'comment', '') or '').strip())
                direction_note = (
                    f"Source signal: {signal_text} | Bias: {float(bias_score_v):.1f} | "
                    f"Signal comment: {comment_text}"
                ).strip()

                ai_display = direction_label(ai_direction)
                ai_direction_key = direction_key(ai_direction)
                consensus_votes, _display_ratio, decision_agreement = ai_vote_metrics(
                    ai_direction_key,
                    float(directional_agreement),
                    float(consensus_agreement),
                )
                ai_display = f"{ai_display} ({consensus_votes}/3)"
                ai_note = (
                    f"AI direction: {direction_label(ai_direction)} | "
                    f"Displayed votes: {consensus_votes}/3 | "
                    f"Directional agreement: {float(directional_agreement) * 100:.0f}% | "
                    f"Consensus agreement: {float(consensus_agreement) * 100:.0f}%"
                )
                if model_votes:
                    vote_labels = [direction_label(str(v)) for v in model_votes]
                    ai_note += f" | Model votes: {', '.join(vote_labels)}"

                _emoji_map = {"HIGH": "🟢", "MEDIUM": "🟡", "TREND": "🟡", "WEAK": "⚪", "CONFLICT": "🔴"}
                strength_val = float(strength_from_bias(float(bias_score_v)))
                strength_note = (
                    f"Bias: {float(bias_score_v):.1f} | Strength: {strength_val:.1f} "
                    f"(formula: |bias-50|^0.70 scaled to 0-100) | "
                    f"Bands: Weak<40, Mixed 40-59, Good 60-74, Strong>=75"
                )
                structure_val = structure_state(
                    signal_direction,
                    ai_direction,
                    strength_val,
                    float(decision_agreement),
                )
                _conv_lbl, _ = _calc_conviction(
                    signal_direction,
                    ai_direction,
                    strength_val,
                    float(decision_agreement),
                )
                conviction = f"{_emoji_map.get(_conv_lbl, '')} {_conv_lbl}" if _conv_lbl else ""
                structure_desc = {
                    "FULL": "Trend+AI confirmed",
                    "TREND": "Trend-led structure",
                    "EARLY": "Early structure",
                    "NONE": "No structure",
                }.get(str(structure_val), str(structure_val))
                align_note = (
                    f"Tech: {direction_label(signal_direction)} | "
                    f"AI: {direction_label(ai_direction)} ({consensus_votes}/3) | "
                    f"Directional agreement: {float(directional_agreement) * 100:.0f}% | "
                    f"Consensus agreement: {float(consensus_agreement) * 100:.0f}% | "
                    f"Structure: {structure_desc}"
                )
                if str(_conv_lbl) == "CONFLICT":
                    align_note += " | Conflict: technical and AI directions are opposite."
                elif str(_conv_lbl) == "WEAK":
                    align_note += " | Weak: no hard conflict, but confirmation quality is low."
                rr_val = float(rr_ratio) if rr_ratio else 0.0
                action, action_reason_code = action_decision_with_reason(
                    signal_direction,
                    strength_val,
                    structure_val,
                    str(_conv_lbl),
                    float(decision_agreement),
                    float(adx_val_v) if pd.notna(adx_val_v) else float("nan"),
                )
                scalp_gate_pass, _ = scalp_quality_gate(
                    scalp_direction=scalp_direction,
                    signal_direction=signal_direction,
                    rr_ratio=rr_val,
                    adx_val=float(adx_val_v) if pd.notna(adx_val_v) else float("nan"),
                    strength=strength_val,
                    conviction_label=str(_conv_lbl),
                    entry=entry_s,
                    stop=stop_s,
                    target=target_s,
                    min_rr=gate_min_rr,
                    min_adx=gate_min_adx,
                    min_strength=gate_min_strength,
                )
                scalp_opportunity_label = direction_label(scalp_direction or "") if scalp_gate_pass else ""

                if not scalp_gate_pass:
                    entry_price = 0.0
                    stop_s = 0.0
                    target_price = 0.0
                    rr_val = 0.0
                entry_note = ""
                scalp_dir_key = direction_key(scalp_direction)
                if scalp_gate_pass and scalp_dir_key != "NEUTRAL" and entry_s:
                    try:
                        close_ref = float(latest_closed["close"])
                        ema5_ref = float(df_eval["close"].ewm(span=5, adjust=False).mean().iloc[-1])
                        if scalp_dir_key == "UPSIDE":
                            base_ref = max(close_ref, ema5_ref)
                            atr_used = (float(entry_s) - base_ref) / 0.20
                            rule_txt = "max(Close, EMA5) + 0.20×ATR"
                        else:
                            base_ref = min(close_ref, ema5_ref)
                            atr_used = (base_ref - float(entry_s)) / 0.20
                            rule_txt = "min(Close, EMA5) - 0.20×ATR"
                        entry_note = (
                            f"{direction_label(scalp_direction)} entry model: {rule_txt} | "
                            f"Close {_fmt_price(close_ref)} | EMA5 {_fmt_price(ema5_ref)} | "
                            f"ATR {_fmt_price(atr_used)}"
                        )
                    except Exception:
                        entry_note = ""
                target_note = str(breakout_note or "").strip() if scalp_gate_pass else ""
                rr_note = ""
                if scalp_gate_pass and target_note:
                    rr_note = f"Conditional R:R: {target_note}"

                ichimoku_cell = format_trend(ichimoku_trend_v)
                ichi_detail_parts: list[str] = []
                if a.ichimoku_tk_cross:
                    ichi_detail_parts.append(f"TK Cross: {a.ichimoku_tk_cross.replace('▲ ', '').replace('▼ ', '').replace('→ ', '')}")
                if a.ichimoku_future_bias:
                    ichi_detail_parts.append(
                        f"Future Cloud: {a.ichimoku_future_bias.replace('▲ ', '').replace('▼ ', '').replace('→ ', '')}"
                    )
                if a.ichimoku_cloud_strength:
                    ichi_detail_parts.append(
                        f"Cloud Strength: {a.ichimoku_cloud_strength.replace('▲ ', '').replace('▼ ', '').replace('→ ', '')}"
                    )
                ichimoku_detail = " | ".join(ichi_detail_parts)

                return {
                    'Coin': base,
                    '__pair': sym,
                    'Price ($)': _fmt_price(price),
                    'Δ (%)': format_delta(price_change) if price_change is not None else '',
                    '__delta_note': delta_note if price_change is not None else "",
                    'Setup Confirm': _setup_confirm_display(action),
                    '__action_raw': action,
                    '__action_reason': action_reason_code,
                    'Direction': direction_label(signal_plain(signal)),
                    '__direction_note': direction_note,
                    'Strength': _strength_badge(float(bias_score_v)),
                    '__strength_note': strength_note,
                    'AI Ensemble': ai_display,
                    '__ai_votes': consensus_votes,
                    '__ai_note': ai_note,
                    'Tech vs AI Alignment': conviction,
                    '__alignment_note': align_note,
                    '__structure_state': structure_val,
                    'Scalp Opportunity': scalp_opportunity_label,
                    'Entry Price': _fmt_price(entry_price) if entry_price else '',
                    '__entry_note': entry_note,
                    'Stop Loss': _fmt_price(stop_s) if stop_s else '',
                    'Target Price': _fmt_price(target_price) if target_price else '',
                    '__target_note': target_note,
                    '__rr_note': rr_note,
                    'R:R': _rr_badge(rr_val),
                    'Market Cap ($)': readable_market_cap(mcap_val) if mcap_val else "—",
                    '__mcap_val': int(mcap_val) if mcap_val else 0,
                    'Spike Alert': '→ Spike' if volume_spike else '',
                    '__spike_dir': spike_dir,
                    '__spike_vol_ratio': spike_vol_ratio,
                    '__spike_candle_pct': spike_candle_pct,
                    '__spike_vwap_ctx': spike_vwap_ctx,
                    'ADX': round(adx_val_v, 1) if pd.notna(adx_val_v) else float("nan"),
                    '__adx_raw': round(adx_val_v, 2) if pd.notna(adx_val_v) else float("nan"),
                    'SuperTrend': supertrend_trend_v,
                    'Volatility': atr_comment_v,
                    'Stochastic RSI': round(stochrsi_k_val_v, 2) if pd.notna(stochrsi_k_val_v) else float("nan"),
                    'Candle Pattern': candle_pattern_v,
                    'Ichimoku': ichimoku_cell,
                    '__ichimoku_detail': ichimoku_detail,
                    'Bollinger': bollinger_bias_v,
                    'VWAP': vwap_label_v,
                    'PSAR': psar_trend_v if psar_trend_v != "Unavailable" else '',
                    'Williams %R': a.williams,
                    'CCI': a.cci,
                    '__strength_val': strength_val,
                }

            # Parallel analysis pass over fetched frames
            fresh_results: list[dict] = []
            scan_errors: list[tuple[str, str]] = []
            with ThreadPoolExecutor(max_workers=4) as executor:
                futures = {executor.submit(_scan_one, sym, df_eval): sym for sym, df_eval in fetched_frames}
                for future in as_completed(futures):
                    try:
                        row = future.result()
                        if row is not None:
                            fresh_results.append(row)
                    except Exception as e:
                        sym = futures[future]
                        err = f"{e.__class__.__name__}: {str(e).strip()}".strip(": ")
                        scan_errors.append((sym, err))
                        _debug(f"Scanner error for {sym}: {err}")

            if scan_errors:
                st.session_state["market_scan_error_count"] = len(scan_errors)
                sample = ", ".join(f"{sym} ({err})" for sym, err in scan_errors[:3])
                more = "" if len(scan_errors) <= 3 else f" +{len(scan_errors) - 3} more"
                st.warning(
                    "Some symbols were skipped due to temporary fetch/analysis errors. "
                    f"Skipped: {len(scan_errors)} | Sample: {sample}{more}."
                )
            else:
                st.session_state["market_scan_error_count"] = 0

            prev_results = st.session_state.get("market_scan_results", [])
            # Sort by execution priority: Setup Confirm > Structure > Strength
            setup_rank = {"FULL": 4, "TREND": 3, "EARLY": 2, "NONE": 1}
            limit_n = len(custom_bases_applied) if custom_mode_active else int(top_n)
            fresh_results = sorted(
                fresh_results,
                key=lambda x: (
                    -action_rank(str(x.get("__action_raw", x.get("Setup Confirm", "")))),
                    -setup_rank.get(str(x.get("__structure_state")), 0),
                    -float(x.get("__strength_val", 0.0)),
                    -float(x.get("__mcap_val", 0)),
                    str(x.get("Coin", "")),
                ),
            )[:limit_n]
            if fresh_results:
                st.session_state["market_scan_results"] = fresh_results
                st.session_state["market_scan_sig"] = scan_sig
                st.session_state["market_scan_cache_ts"] = pd.Timestamp.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")
                st.session_state["market_scan_cache_sig"] = scan_sig
                st.session_state["market_scan_source"] = "LIVE"
                st.session_state["market_data_mode"] = data_mode
                source_label = "LIVE"
                results = fresh_results
            else:
                st.session_state["market_scan_sig"] = scan_sig
                cache_sig = st.session_state.get("market_scan_cache_sig")
                same_sig_cache = bool(cache_sig and tuple(cache_sig) == tuple(scan_sig))
                if prev_results and same_sig_cache:
                    # Only fallback to cache for the exact same scan signature.
                    ts = st.session_state.get("market_scan_cache_ts", "unknown time")
                    ts_parsed = pd.to_datetime(ts, utc=True, errors="coerce")
                    is_fresh_cache = False
                    if pd.notna(ts_parsed):
                        try:
                            age_minutes = (
                                pd.Timestamp.now(tz="UTC") - ts_parsed
                            ).total_seconds() / 60.0
                            is_fresh_cache = age_minutes <= float(CACHE_TTL_MINUTES)
                        except Exception:
                            is_fresh_cache = False
                    if is_fresh_cache:
                        results = prev_results
                        source_label = f"CACHED ({ts})"
                        st.session_state["market_scan_source"] = source_label
                        st.session_state["market_data_mode"] = data_mode
                        st.warning(
                            f"Live scan returned no rows. Showing last successful snapshot from {ts} "
                            f"for the same timeframe/filter. Do not execute directly from cache-only view."
                        )
                    else:
                        results = []
                        source_label = "LIVE"
                        st.session_state["market_scan_source"] = source_label
                        st.session_state["market_data_mode"] = data_mode
                        st.warning(
                            f"Live scan returned no rows and cache is older than {CACHE_TTL_MINUTES} minutes. "
                            "Stale snapshot was not used."
                        )
                else:
                    # Do not leak stale cache across timeframe/filter changes.
                    results = []
                    source_label = "LIVE"
                    st.session_state["market_scan_source"] = source_label
                    st.session_state["market_data_mode"] = data_mode
                    if cache_sig and tuple(cache_sig) != tuple(scan_sig):
                        st.warning(
                            "Live scan returned no rows for current timeframe/filter. "
                            "Stale cache from another setting was intentionally not used."
                        )

    # Prepare DataFrame for display
    if results:
        source_color = POSITIVE if source_label.startswith("LIVE") else WARNING
        source_chip = "LIVE FEED" if source_label.startswith("LIVE") else "CACHED SNAPSHOT"
        mode_color = ACCENT if data_mode.startswith("FULL") else WARNING
        if source_label.startswith("CACHED"):
            st.markdown(
                f"<div class='market-note-box' style='border:1px solid rgba(255,209,102,0.4); border-left:4px solid {WARNING}; "
                f"background:rgba(255,209,102,0.08); color:{TEXT_MUTED};'>"
                f"<b style='color:{WARNING};'>Execution Caution:</b> This table is running on cached snapshot data. "
                f"Confirm with a fresh LIVE scan before placing trades."
                f"</div>",
                unsafe_allow_html=True,
            )
        if data_mode.startswith("EXCHANGE-ONLY"):
            st.markdown(
                f"<div class='market-note-box' style='border:1px solid rgba(255,209,102,0.34); border-left:4px solid {WARNING}; "
                f"background:rgba(255,209,102,0.06); color:{TEXT_MUTED};'>"
                f"<b style='color:{WARNING};'>Data Mode:</b> Exchange-only feed is active. "
                f"Trade metrics are live from exchange candles; enrichment fields like market cap may show as —."
                f"</div>",
                unsafe_allow_html=True,
            )
        st.markdown(
            f"<details class='market-details' style='margin-bottom:0.6rem;'>"
            f"<summary style='color:{ACCENT};'>"
            f"How to read quickly (?)</summary>"
            f"<div class='market-details-body' style='color:{TEXT_MUTED};'>"
            f"<b>Decision sequence:</b> <b>Setup Confirm</b> → <b>Direction + Strength</b> → "
            f"<b>AI Ensemble + Tech vs AI Alignment</b>.<br><br>"
            f"<b>Setup Confirm classes:</b> TREND+AI (strongest), TREND-led, AI-led, WATCH, SKIP.<br>"
            f"<b>AI Ensemble:</b> direction + 3-dot agreement meter (more filled dots = stronger model agreement).<br>"
            f"<b>Scalp Opportunity:</b> separate execution gate; shown only when direction/levels/quality thresholds all pass.<br><br>"
            f"<b>Scan mode:</b> default Top N market scan. Custom Coins (max 10) scans only your watchlist until cleared."
            f"</div></details>",
            unsafe_allow_html=True,
        )
        st.markdown(
            f"<details class='market-details' style='margin-bottom:0.8rem;'>"
            f"<summary style='color:{ACCENT};'>"
            f"ℹ️ Column Guide (click to expand)</summary>"
            f"<div class='market-details-body' style='color:{TEXT_MUTED}; padding:0.5rem;'>"
            "<b>Coin</b>: asset ticker (hover shows exchange pair).<br>"
            "<b>Price ($)</b>: latest closed-candle price from active feed.<br>"
            "<b>Δ (%)</b>: change from previous closed candle to latest closed candle on selected timeframe (fallback: ticker % if candle delta unavailable).<br><br>"
            "<b>Setup Confirm</b>: final scanner confirmation class (not a direct execution order). "
            "Calculated from Direction + Strength + AI Ensemble + Alignment + ADX regime checks.<br>"
            "<b>Direction</b>: side from technical signal mapping (Upside / Downside / Neutral).<br>"
            "<b>Strength</b>: technical signal power (0-100), derived from bias distance to neutral midpoint (50).<br>"
            "<b>AI Ensemble</b>: side label plus 3-dot model-agreement meter (filled dots = stronger agreement).<br>"
            "<b>Tech vs AI Alignment</b>: confirmation quality between technical side and AI side (HIGH/MEDIUM/TREND/WEAK/CONFLICT).<br>"
            "<b>R:R</b>: reward-to-risk ratio from target distance vs stop distance.<br>"
            "<b>Entry Price</b>: model entry level (close/EMA5 with ATR buffer).<br>"
            "<b>Stop Loss</b>: risk invalidation level (support/resistance with ATR clamps).<br>"
            "<b>Target Price</b>: first take-profit level (structure level with minimum ATR extension).<br>"
            "<b>R:R marker (*)</b>: conditional plan; target may require breakout.<br>"
            "<b>Scalp Opportunity</b>: shown only when the single execution gate passes "
            "(Direction match + no CONFLICT + valid levels + timeframe-adaptive R:R/ADX/Strength thresholds).<br>"
            "<b>Market Cap ($)</b>: size/liquidity context.<br><br>"
            "<b>Advanced columns (what they mean + short calc):</b><br>"
            "<b>ADX</b>: trend strength (14-period directional movement strength; not side).<br>"
            "<b>SuperTrend</b>: trend state from ATR-based trailing bands (Bullish/Bearish/Neutral).<br>"
            "<b>Ichimoku</b>: cloud trend state from conversion/base/cloud structure.<br>"
            "<b>VWAP</b>: price position vs Volume-Weighted Average Price (Above/Below/Near).<br>"
            "<b>Spike Alert</b>: abnormal volume event (volume ratio vs recent average + spike candle direction).<br>"
            "<b>Bollinger</b>: price location vs 20-period volatility bands (Overbought/Oversold/Neutral).<br>"
            "<b>Stochastic RSI</b>: momentum position of RSI in its recent range (Low/High/Neutral).<br>"
            "<b>Volatility</b>: ATR-based volatility regime label.<br>"
            "<b>PSAR</b>: Parabolic SAR trend side (Bullish/Bearish).<br>"
            "<b>Williams %R</b>: momentum oscillator showing near-top / near-bottom conditions.<br>"
            "<b>CCI</b>: deviation of typical price from its moving average (trend pressure/mean-reversion context).<br>"
            "<b>Candle Pattern</b>: candlestick pattern classifier with directional label (bullish/bearish/neutral)."
            "</div></details>",
            unsafe_allow_html=True,
        )
        st.caption("Signals and plan levels are computed on closed candles; Price ($) shows the latest candle close.")
        controls_col, chips_col = st.columns([1.2, 2.8], gap="small")
        with controls_col:
            show_advanced = st.checkbox("+ Show advanced columns", value=False, key="market_show_adv_cols")
        with chips_col:
            st.markdown(
                f"<div style='display:flex; align-items:center; gap:10px; flex-wrap:wrap; padding-top:0.18rem;'>"
                f"<span class='market-inline-chip' style='border:1px solid {source_color}; color:{source_color}; "
                f"background:rgba(255,255,255,0.04);'>{source_chip} • {source_label}</span>"
                f"<span class='market-inline-chip' style='border:1px solid {mode_color}; color:{mode_color}; "
                f"background:rgba(255,255,255,0.04);'>{data_mode}</span>"
                f"</div>",
                unsafe_allow_html=True,
            )

        df_results = pd.DataFrame(results)

        # Quick scan health summary (visual-first, logic unchanged)
        if "__action_raw" in df_results.columns:
            action_series = df_results["__action_raw"].astype(str)
        else:
            action_series = df_results.get("Setup Confirm", pd.Series(dtype=str)).astype(str)
        action_class_series = action_series.apply(_setup_confirm_class)
        enter_count = int(action_class_series.str.startswith("ENTER_").sum())
        watch_count = int((action_class_series == "WATCH").sum())
        skip_count = int((action_class_series == "SKIP").sum())
        trend_ai_enter_count = int((action_class_series == "ENTER_TREND_AI").sum())
        trend_led_enter_count = int((action_class_series == "ENTER_TREND_LED").sum())
        ai_led_enter_count = int((action_class_series == "ENTER_AI_LED").sum())

        best_scalp_coin = "—"
        best_scalp_sub = ""
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
                scoped["__action_rank"] = (
                    scoped.get("__action_raw", scoped.get("Setup Confirm", pd.Series(dtype=str)))
                    .astype(str)
                    .apply(_setup_confirm_rank)
                )
                scoped = scoped.dropna(subset=["__rr"])
                scoped = scoped[scoped["__rr"] > 0]
                if not scoped.empty:
                    best_row = scoped.sort_values(["__rr", "__action_rank"], ascending=[False, False]).iloc[0]
                    best_coin = str(best_row.get("Coin", "—"))
                    best_rr = float(best_row["__rr"])
                    best_scalp_coin = f"{best_coin} ({best_rr:.2f})"
                    best_action = str(best_row.get("__action_raw", best_row.get("Setup Confirm", ""))).strip()
                    best_direction = str(best_row.get("Direction", "")).strip()
                    best_strength = str(best_row.get("Strength", "")).strip()
                    best_ai = str(best_row.get("AI Ensemble", "")).strip()
                    best_action_compact = _setup_confirm_display(best_action)
                    best_scalp_sub = (
                        f"Setup: {best_action_compact} • Direction: {best_direction} • "
                        f"Strength: {best_strength} • Ensemble: {best_ai}"
                    )

        strength_coin = "—"
        strength_val_head = None
        strength_sub = "No strength data available."
        if "__strength_val" in df_results.columns and len(df_results) > 0:
            strength_series = pd.to_numeric(df_results["__strength_val"], errors="coerce").dropna()
            if not strength_series.empty:
                strength_idx = strength_series.idxmax()
                row = df_results.loc[strength_idx]
                strength_coin = str(row.get("Coin", "—"))
                strength_val_head = float(row.get("__strength_val", 0.0))
                strength_sub = (
                    f"Direction: {row.get('Direction', '')} • "
                    f"Ensemble: {row.get('AI Ensemble', '')}"
                )
        strength_head = strength_coin if strength_val_head is None else f"{strength_coin} ({strength_val_head:.0f}%)"

        q1, q2, q3, q4 = st.columns(4, gap="small")
        with q1:
            status_head = "SETUP READY" if enter_count > 0 else "NO SETUP READY"
            status_sub = f"READY: {enter_count} • WATCH: {watch_count} • SKIP: {skip_count}"
            st.markdown(
                "<div class='elite-card' style='min-height:164px; display:flex; flex-direction:column; justify-content:space-between;'>"
                "<div class='elite-label'>Execution Status</div>"
                f"<div class='scan-kpi-value'>{status_head}</div>"
                f"<div class='scan-kpi-sub'>{status_sub}</div>"
                "</div>",
                unsafe_allow_html=True,
            )
        with q2:
            enter_mix_head = "NO READY CLASS" if enter_count == 0 else "CLASS BREAKDOWN"
            enter_mix_sub = (
                f"Trend+AI: {trend_ai_enter_count} • "
                f"Trend-led: {trend_led_enter_count} • "
                f"AI-led: {ai_led_enter_count}"
            )
            st.markdown(
                "<div class='elite-card' style='min-height:164px; display:flex; flex-direction:column; justify-content:space-between;'>"
                "<div class='elite-label'>Setup Class Mix</div>"
                f"<div class='scan-kpi-value'>{enter_mix_head}</div>"
                f"<div class='scan-kpi-sub'>{enter_mix_sub}</div>"
                "</div>",
                unsafe_allow_html=True,
            )
        with q3:
            st.markdown(
                "<div class='elite-card' style='min-height:164px; display:flex; flex-direction:column; justify-content:space-between;'>"
                "<div class='elite-label'>Best Scalp Opportunity</div>"
                f"<div class='elite-value'>{best_scalp_coin}</div>"
                f"<div class='elite-sub' title='{best_scalp_sub}' "
                f"style='white-space:nowrap; overflow:hidden; text-overflow:ellipsis;'>{best_scalp_sub}</div>"
                "</div>",
                unsafe_allow_html=True,
            )
        with q4:
            st.markdown(
                "<div class='elite-card' style='min-height:164px; display:flex; flex-direction:column; justify-content:space-between;'>"
                "<div class='elite-label'>Strength Leader</div>"
                f"<div class='elite-value'>{strength_head}</div>"
                f"<div class='elite-sub'>{strength_sub}</div>"
                "</div>",
                unsafe_allow_html=True,
            )
        st.markdown("<div style='height:0.7rem;'></div>", unsafe_allow_html=True)

        # indicator visual formatting
        df_results["SuperTrend"] = df_results["SuperTrend"].apply(format_trend)
        df_results["ADX"] = df_results["ADX"].apply(format_adx).apply(_compact_adx_label)
        def _format_ichimoku_cell(v: object) -> str:
            raw = str(v or "").strip()
            if not raw or raw.upper() in {"UNAVAILABLE", "N/A", "NA", "NAN"}:
                return ""
            cleaned = (
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
            up = cleaned.upper()
            if "BULLISH" in up:
                return "Bullish"
            if "BEARISH" in up:
                return "Bearish"
            if "NEUTRAL" in up:
                return "Neutral"
            return cleaned

        df_results["Ichimoku"] = df_results["Ichimoku"].apply(_format_ichimoku_cell)
        df_results["Stochastic RSI"] = (
            df_results["Stochastic RSI"]
            .apply(lambda v: format_stochrsi(v, timeframe=timeframe))
            .apply(_normalize_indicator_label)
        )
        df_results["VWAP"] = df_results["VWAP"].apply(_normalize_indicator_label)
        df_results["Bollinger"] = df_results["Bollinger"].apply(_normalize_indicator_label)
        df_results["PSAR"] = df_results["PSAR"].apply(_normalize_indicator_label)
        df_results["Williams %R"] = df_results["Williams %R"].apply(_normalize_indicator_label)
        df_results["CCI"] = df_results["CCI"].apply(_normalize_indicator_label)

        primary_cols = [
            "Coin",
            "Price ($)",
            "Δ (%)",
            "Setup Confirm",
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
        hidden_meta_cols = [
                c for c in [
                    "__action_reason",
                    "__action_raw",
                    "__entry_note",
                "__ichimoku_detail",
                "__spike_dir",
                "__spike_vol_ratio",
                "__spike_candle_pct",
                "__spike_vwap_ctx",
                "__target_note",
                "__rr_note",
                "__adx_raw",
                "__direction_note",
                "__strength_note",
                    "__ai_note",
                    "__alignment_note",
                    "__pair",
                ] if c in df_results.columns and c not in display_cols
            ]
        df_display = df_results[display_cols + hidden_meta_cols].copy()

        _render_pro_table(df_display, display_cols)

        def _csv_clean_text(v: object) -> str:
            if v is None:
                return ""
            try:
                if pd.isna(v):
                    return ""
            except Exception:
                pass
            s = str(v).strip()
            if not s or s.upper() in {"N/A", "NA", "NAN", "UNAVAILABLE", "-"}:
                return ""
            for token in (
                "✅", "🟡", "⛔", "⌛", "👀", "🟢", "🔴", "⚪", "🔥",
                "⚠️", "⚠", "🚀", "📈", "📉", "•", "*", "★",
            ):
                s = s.replace(token, "")
            s = re.sub(r"[▲▼→–]+", "", s)
            s = re.sub(r"\s{2,}", " ", s).strip()
            return s

        def _csv_clean_price(v: object) -> str:
            s = _csv_clean_text(v)
            if not s:
                return ""
            return s.replace("$", "").replace(",", "").strip()

        def _csv_clean_delta(v: object) -> str:
            s = str(v or "").strip()
            if not s:
                return ""
            sign = "+"
            if s.startswith("▼"):
                sign = "-"
            elif s.startswith("→"):
                sign = ""
            cleaned = re.sub(r"^[\s▲▼→–-]+", "", s).strip()
            if not cleaned:
                return ""
            return f"{sign}{cleaned}" if sign else cleaned

        csv_df = df_results.copy()
        for col in csv_df.columns:
            if col in {"Price ($)", "Entry Price", "Stop Loss", "Target Price"}:
                csv_df[col] = csv_df[col].apply(_csv_clean_price)
            elif col == "Δ (%)":
                csv_df[col] = csv_df[col].apply(_csv_clean_delta)
            elif csv_df[col].dtype == object or pd.api.types.is_string_dtype(csv_df[col]):
                csv_df[col] = csv_df[col].apply(_csv_clean_text)

        # Never export internal/meta columns.
        csv_df = csv_df[[c for c in csv_df.columns if not str(c).startswith("__")]]

        csv_market = csv_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="Download Scan Results (CSV)",
            data=csv_market,
            file_name="scan_results.csv",
            mime="text/csv"
        )
    else:
        st.info("No coins matched the criteria.")
