from ui.ctx import get_ctx

import numpy as np
import plotly.graph_objs as go
import ta
from core.metric_catalog import ai_stability_bucket
from ui.snapshot_cache import live_or_snapshot


def render(ctx: dict) -> None:
    st = get_ctx(ctx, "st")
    ACCENT = get_ctx(ctx, "ACCENT")
    TEXT_MUTED = get_ctx(ctx, "TEXT_MUTED")
    POSITIVE = get_ctx(ctx, "POSITIVE")
    NEGATIVE = get_ctx(ctx, "NEGATIVE")
    WARNING = get_ctx(ctx, "WARNING")
    CARD_BG = get_ctx(ctx, "CARD_BG")
    PRIMARY_BG = get_ctx(ctx, "PRIMARY_BG")
    TEXT_LIGHT = get_ctx(ctx, "TEXT_LIGHT")
    _tip = get_ctx(ctx, "_tip")
    _normalize_coin_input = get_ctx(ctx, "_normalize_coin_input")
    _validate_coin_symbol = get_ctx(ctx, "_validate_coin_symbol")
    fetch_ohlcv = get_ctx(ctx, "fetch_ohlcv")
    analyse = get_ctx(ctx, "analyse")
    signal_plain = get_ctx(ctx, "signal_plain")
    ml_ensemble_predict = get_ctx(ctx, "ml_ensemble_predict")
    _calc_conviction = get_ctx(ctx, "_calc_conviction")
    _build_indicator_grid = get_ctx(ctx, "_build_indicator_grid")
    get_social_sentiment = get_ctx(ctx, "get_social_sentiment")
    _wma = get_ctx(ctx, "_wma")
    _sr_lookback = get_ctx(ctx, "_sr_lookback")
    _debug = get_ctx(ctx, "_debug")
    """Render the Spot Trading tab which allows instant analysis of a selected coin."""
    st.markdown(
        f"<h2 style='color:{ACCENT};margin-bottom:0.5rem;'>Spot Trading</h2>",
        unsafe_allow_html=True,
    )
    st.markdown(
        f"<div class='panel-box'>"
        f"<b style='color:{ACCENT}; font-size:1rem;'>What does this tab show?</b>"
        f"<p style='color:{TEXT_MUTED}; font-size:0.9rem; margin-top:6px; line-height:1.6;'>"
        f"In-depth technical analysis for any coin. Combines "
        f"{_tip('Trend', 'EMA crossovers, SuperTrend, Ichimoku Cloud, Parabolic SAR, and ADX indicators.')} (40%), "
        f"{_tip('Momentum', 'RSI, MACD, Stochastic RSI, Williams %R, and CCI indicators.')} (30%), "
        f"{_tip('Volume', 'OBV direction, volume spikes, and VWAP positioning.')} (20%), and "
        f"{_tip('Volatility', 'Bollinger Band width, ATR level, and Keltner Channel breakouts.')} (10%) "
        f"into a single signal (BUY / SELL / WAIT) with a confidence score from 0-100%. "
        f"Designed for spot trading without leverage.</p>"
        f"</div>",
        unsafe_allow_html=True,
    )
    st.markdown(
        f"<details style='margin-bottom:0.7rem;'>"
        f"<summary style='color:{ACCENT}; cursor:pointer;'>How to read quickly (?)</summary>"
        f"<div style='color:{TEXT_MUTED}; font-size:0.85rem; line-height:1.7; margin-top:0.45rem;'>"
        f"<b>1.</b> Start with Signal + Confidence + AI Ensemble + Conviction.<br>"
        f"<b>2.</b> If technical Signal and AI disagree, treat setup as weaker.<br>"
        f"<b>3.</b> Use indicator grid and snapshot for context, not standalone entry triggers."
        f"</div></details>",
        unsafe_allow_html=True,
    )
    coin = _normalize_coin_input(st.text_input(
        "Coin (e.g. BTC, ETH, TAO)",
        value="BTC",
        key="spot_coin_input",
    ))
    timeframe = st.selectbox("Timeframe", ['1m', '3m', '5m', '15m', '1h', '4h', '1d'], index=4)
    if st.button("Analyse", type="primary"):
        _val_err = _validate_coin_symbol(coin)
        if _val_err:
            st.error(_val_err)
            return
        df_live = fetch_ohlcv(coin, timeframe)
        df, used_cache, cache_ts = live_or_snapshot(st, f"spot_df::{coin}::{timeframe}", df_live)
        if used_cache:
            st.warning(f"Live data unavailable. Showing cached snapshot from {cache_ts}.")
        if df is None or len(df) < 60:
            st.error(f"Could not fetch data for **{coin}** on {timeframe}. The coin may not be listed on supported exchanges. Try a major pair (BTC, ETH) or check the symbol.")
            return
        # Keep analysis on closed candles for consistency with signal engine.
        df_eval = df.iloc[:-1].copy() if len(df) > 60 else df.copy()
        if df_eval is None or len(df_eval) < 55:
            st.error("Not enough closed-candle data for a stable analysis.")
            return

        a = analyse(df_eval)
        signal, lev, comment, volume_spike = a.signal, a.leverage, a.comment, a.volume_spike
        atr_comment, candle_pattern, confidence_score = a.atr_comment, a.candle_pattern, a.confidence
        adx_val, supertrend_trend, ichimoku_trend = a.adx, a.supertrend, a.ichimoku
        stochrsi_k_val, bollinger_bias, vwap_label = a.stochrsi_k, a.bollinger, a.vwap
        psar_trend, williams_label, cci_label = a.psar, a.williams, a.cci

        current_price = df['close'].iloc[-1]

        # Display summary grid
        signal_clean = signal_plain(signal)
        try:
            _ai_prob_s, ai_dir_s, _ai_details_s = ml_ensemble_predict(df_eval)
            ai_agree = float((_ai_details_s or {}).get("agreement", 0.0)) * 100.0
        except Exception:
            ai_dir_s = "NEUTRAL"
            ai_agree = 0.0

        sig_dir_s = "LONG" if signal in ['STRONG BUY', 'BUY'] else ("SHORT" if signal in ['STRONG SELL', 'SELL'] else "WAIT")
        conv_lbl_s, conv_c_s = _calc_conviction(sig_dir_s, ai_dir_s, confidence_score)
        ai_stability = ai_stability_bucket(ai_agree / 100.0)

        sig_c_s = POSITIVE if "LONG" in signal_clean else (NEGATIVE if "SHORT" in signal_clean else WARNING)
        ai_c_s = POSITIVE if ai_dir_s == "LONG" else (NEGATIVE if ai_dir_s == "SHORT" else WARNING)
        conf_c_s = POSITIVE if confidence_score >= 70 else (WARNING if confidence_score >= 50 else NEGATIVE)
        ai_stab_c = POSITIVE if ai_stability == "Strong" else (WARNING if ai_stability == "Medium" else TEXT_MUTED)
        st.markdown(
            f"<div style='display:grid; grid-template-columns:repeat(auto-fit, minmax(120px, 1fr)); "
            f"gap:4px; background:{CARD_BG}; border-radius:8px; padding:10px; margin:8px 0;'>"
            f"<div style='text-align:center; padding:6px;'>"
            f"<div style='color:{TEXT_MUTED}; font-size:0.7rem; text-transform:uppercase;'>Signal</div>"
            f"<div style='color:{sig_c_s}; font-size:0.85rem; font-weight:600;'>{signal_clean}</div></div>"
            f"<div style='text-align:center; padding:6px;'>"
            f"<div style='color:{TEXT_MUTED}; font-size:0.7rem; text-transform:uppercase;'>Confidence</div>"
            f"<div style='color:{conf_c_s}; font-size:0.85rem; font-weight:600;'>{confidence_score:.0f}%</div></div>"
            f"<div style='text-align:center; padding:6px;'>"
            f"<div style='color:{TEXT_MUTED}; font-size:0.7rem; text-transform:uppercase;'>AI Ensemble</div>"
            f"<div style='color:{ai_c_s}; font-size:0.85rem; font-weight:600;'>{ai_dir_s}</div></div>"
            f"<div style='text-align:center; padding:6px;'>"
            f"<div style='color:{TEXT_MUTED}; font-size:0.7rem; text-transform:uppercase;'>AI Stability</div>"
            f"<div style='color:{ai_stab_c}; font-size:0.85rem; font-weight:600;'>{ai_stability} ({ai_agree:.0f}%)</div></div>"
            f"<div style='text-align:center; padding:6px;'>"
            f"<div style='color:{TEXT_MUTED}; font-size:0.7rem; text-transform:uppercase;'>Conviction</div>"
            f"<div style='color:{conv_c_s}; font-size:0.85rem; font-weight:600;'>{conv_lbl_s}</div></div>"
            f"</div>",
            unsafe_allow_html=True,
        )
        st.markdown(
            f"<div style='color:{TEXT_MUTED}; font-size:0.82rem; margin:0.15rem 0 0.55rem 0;'>"
            f"<b>Signal</b> = technical direction, <b>Confidence</b> = score strength, "
            f"<b>AI Stability</b> = model agreement quality, <b>Conviction</b> = technical+AI alignment."
            f"</div>",
            unsafe_allow_html=True,
        )

        st.markdown(f"<p style='color:{TEXT_MUTED}; font-size:0.88rem;'>{comment}</p>", unsafe_allow_html=True)

        # Market regime warning (ADX < 20 = ranging market)
        if not np.isnan(adx_val) and adx_val < 20:
            st.markdown(
                f"<div style='background:#2D1B00; border-left:4px solid {WARNING}; "
                f"padding:8px 12px; border-radius:4px; margin:6px 0;'>"
                f"<span style='color:{WARNING}; font-weight:600;'>Market Ranging (ADX {adx_val:.0f})</span>"
                f"<span style='color:{TEXT_MUTED}; font-size:0.82rem;'> — Trend signals are less reliable. "
                f"Confidence discounted. Consider smaller positions or waiting for a clear trend.</span></div>",
                unsafe_allow_html=True,
            )

        # Risk alert: model risk profile suggests aggression while confidence is weak.
        if lev >= 6 and confidence_score < 55:
            st.markdown(
                f"<div style='background:#2D0A0A; border-left:4px solid {NEGATIVE}; "
                f"padding:8px 12px; border-radius:4px; margin:6px 0;'>"
                f"<span style='color:{NEGATIVE}; font-weight:600;'>Risk Warning</span>"
                f"<span style='color:{TEXT_MUTED}; font-size:0.82rem;'> — The model risk profile is aggressive "
                f"(x{lev}) but confidence is only {confidence_score:.0f}%. For spot, treat this as unstable conditions "
                f"and avoid oversized entries.</span></div>",
                unsafe_allow_html=True,
            )

        # Indicator grid (professional card layout)
        _grid_html = _build_indicator_grid(
            supertrend_trend, ichimoku_trend, vwap_label, adx_val, bollinger_bias,
            stochrsi_k_val, psar_trend, williams_label, cci_label,
            volume_spike, atr_comment, candle_pattern,
        )
        if _grid_html:
            st.markdown(_grid_html, unsafe_allow_html=True,
            )

        # Price box
        st.markdown(f"<div class='metric-card'><div class='metric-label'>Current Price</div><div class='metric-value'>${current_price:,.2f}</div></div>", unsafe_allow_html=True)

        # Action plan (beginner-friendly, spot-only, conditional playbook)
        try:
            atr14 = float(ta.volatility.average_true_range(df_eval["high"], df_eval["low"], df_eval["close"], window=14).iloc[-1])
        except Exception:
            atr14 = 0.0
        plan_lookback = _sr_lookback(timeframe)
        plan_recent = df_eval.tail(plan_lookback)
        plan_support = float(plan_recent["low"].min())
        plan_resistance = float(plan_recent["high"].max())
        pullback_low = max(0.0, plan_support - 0.2 * atr14)
        pullback_high = plan_support + 0.4 * atr14
        breakout_trigger = plan_resistance + 0.2 * atr14
        invalidation_level = max(0.0, plan_support - 0.8 * atr14)

        if signal_clean == "LONG":
            plan_status = "Bullish Playbook"
            plan_color = POSITIVE
            plan_lines = (
                f"1) <b>Buy zone:</b> ${pullback_low:,.2f} - ${pullback_high:,.2f} (pullback near support).<br>"
                f"2) <b>Breakout add:</b> only if price closes above ${breakout_trigger:,.2f}.<br>"
                f"3) <b>Risk line:</b> if price closes below ${invalidation_level:,.2f}, setup is invalid."
            )
        elif signal_clean == "SHORT":
            plan_status = "Defensive Playbook"
            plan_color = NEGATIVE
            plan_lines = (
                f"1) <b>No fresh spot buy</b> while bearish pressure continues.<br>"
                f"2) <b>Re-entry watch:</b> wait reclaim above ${plan_resistance:,.2f} or clear base near support.<br>"
                f"3) <b>Risk line:</b> protect capital if price loses ${invalidation_level:,.2f} with momentum."
            )
        else:
            plan_status = "Wait Playbook"
            plan_color = WARNING
            plan_lines = (
                f"1) <b>No-trade zone:</b> mixed signals, avoid forcing entries.<br>"
                f"2) <b>Bull trigger:</b> acceptance above ${breakout_trigger:,.2f}.<br>"
                f"3) <b>Pullback watch:</b> reaction around ${pullback_low:,.2f} - ${pullback_high:,.2f}."
            )

        st.markdown(
            f"<div class='panel-box' style='border-left:4px solid {plan_color};'>"
            f"<b style='color:{plan_color}; font-size:1rem;'>Spot Action Plan ({plan_status})</b>"
            f"<div style='color:{TEXT_MUTED}; font-size:0.86rem; line-height:1.7; margin-top:6px;'>"
            f"{plan_lines}"
            f"<br><span style='color:{TEXT_MUTED}; font-size:0.78rem;'>Guide only, not financial advice. "
            f"Always confirm with your own risk plan.</span>"
            f"</div></div>",
            unsafe_allow_html=True,
        )

        sent_score, sent_label = get_social_sentiment(coin)
        gauge_sent = go.Figure(go.Indicator(
            mode="gauge+number",
            value=sent_score,
            gauge={
                'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': TEXT_MUTED},
                'bar': {'color': ACCENT},
                'bgcolor': PRIMARY_BG,
                'steps': [
                    {'range': [0, 25], 'color': NEGATIVE},
                    {'range': [25, 45], 'color': WARNING},
                    {'range': [45, 55], 'color': TEXT_MUTED},
                    {'range': [55, 75], 'color': POSITIVE},
                    {'range': [75, 100], 'color': POSITIVE},
                ],
            },
            title={'text': f"Price Bias Proxy ({sent_label})", 'font': {'size': 16, 'color': ACCENT}},
            number={'font': {'color': TEXT_LIGHT, 'size': 36}}
        ))
        gauge_sent.update_layout(
            height=170,
            margin=dict(l=10, r=10, t=40, b=15),
            template='plotly_dark',
            paper_bgcolor=PRIMARY_BG
        )
        st.plotly_chart(gauge_sent, width="stretch")
        st.caption(
            "Price Bias Proxy is derived from 24h price change (not social-media/NLP sentiment). "
            "Use as a light context signal only."
        )
        # Plot candlestick with EMAs
        fig = go.Figure()
        fig.add_trace(go.Candlestick(
            x=df['timestamp'], open=df['open'], high=df['high'], low=df['low'], close=df['close'],
            increasing_line_color=POSITIVE, decreasing_line_color=NEGATIVE, name="Price"
        ))
        # Plot EMAs
        for window, color in [(5, '#F472B6'), (9, '#60A5FA'), (21, '#FBBF24'), (50, '#FCD34D')]:
            ema_series = ta.trend.ema_indicator(df['close'], window=window)
            fig.add_trace(go.Scatter(x=df['timestamp'], y=ema_series, mode='lines',
                                     name=f"EMA{window}", line=dict(color=color, width=1.5)))
        # Plot weighted moving averages (WMA) for additional insight.  The WMA gives
        # more weight to recent prices and can help identify trend shifts earlier.
        try:
            wma20 = _wma(df['close'], length=20)
            wma50 = _wma(df['close'], length=50)
            fig.add_trace(go.Scatter(x=df['timestamp'], y=wma20, mode='lines',
                                     name="WMA20", line=dict(color='#34D399', width=1, dash='dot')))
            fig.add_trace(go.Scatter(x=df['timestamp'], y=wma50, mode='lines',
                                     name="WMA50", line=dict(color='#10B981', width=1, dash='dash')))
        except Exception as e:
            _debug(f"WMA chart overlay error: {e}")
        # Place legend at top left for candlestick chart
        fig.update_layout(
            height=380,
            template='plotly_dark',
            margin=dict(l=20, r=20, t=30, b=30),
            xaxis_rangeslider_visible=False,
            showlegend=True,
            legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='left', x=0)
        )
        st.plotly_chart(fig, width="stretch")
        # RSI chart
        rsi_fig = go.Figure()
        for period, color in [(6, '#D8B4FE'), (14, '#A78BFA'), (24, '#818CF8')]:
            rsi_series = ta.momentum.rsi(df['close'], window=period)
            rsi_fig.add_trace(go.Scatter(
                x=df['timestamp'], y=rsi_series, mode='lines', name=f"RSI {period}",
                line=dict(color=color, width=2)
            ))
        # Add overbought/oversold bands
        rsi_fig.add_hline(y=70, line=dict(color=NEGATIVE, dash='dot', width=1), name="Overbought")
        rsi_fig.add_hline(y=30, line=dict(color=POSITIVE, dash='dot', width=1), name="Oversold")
        rsi_fig.update_layout(
            height=180,
            template='plotly_dark',
            margin=dict(l=20, r=20, t=20, b=30),
            yaxis=dict(title="RSI"),
            showlegend=True,
            legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='left', x=0)
        )
        st.plotly_chart(rsi_fig, width="stretch")

        # MACD chart
        macd_ind = ta.trend.MACD(df['close'])
        df['macd'] = macd_ind.macd()
        df['macd_signal'] = macd_ind.macd_signal()
        df['macd_diff'] = macd_ind.macd_diff()
        macd_fig = go.Figure()
        macd_fig.add_trace(go.Scatter(
            x=df['timestamp'], y=df['macd'], name="MACD",
            line=dict(color=ACCENT, width=2)
        ))
        macd_fig.add_trace(go.Scatter(
            x=df['timestamp'], y=df['macd_signal'], name="Signal",
            line=dict(color=WARNING, width=2, dash='dot')
        ))
        macd_fig.add_trace(go.Bar(
            x=df['timestamp'], y=df['macd_diff'], name="Histogram",
            marker_color=CARD_BG
        ))
        macd_fig.update_layout(
            height=200,
            template='plotly_dark',
            margin=dict(l=20, r=20, t=20, b=30),
            yaxis=dict(title="MACD"),
            showlegend=True,
            legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='left', x=0)
        )
        st.plotly_chart(macd_fig, width="stretch")

        # Volume & OBV chart
        df['obv'] = ta.volume.on_balance_volume(df['close'], df['volume'])
        volume_fig = go.Figure()
        volume_fig.add_trace(go.Bar(
            x=df['timestamp'], y=df['volume'], name="Volume", marker_color="#6B7280"
        ))
        volume_fig.add_trace(go.Scatter(
            x=df['timestamp'], y=df['obv'], name="OBV",
            line=dict(color=WARNING, width=1.5, dash='dot'),
            yaxis='y2'
        ))
        volume_fig.update_layout(
            height=180,
            template='plotly_dark',
            margin=dict(l=20, r=20, t=20, b=30),
            yaxis=dict(title="Volume"),
            yaxis2=dict(overlaying='y', side='right', title='OBV', showgrid=False),
            showlegend=True,
            legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='left', x=0)
        )
        st.plotly_chart(volume_fig, width="stretch")

        # Technical snapshot
        # Compute indicators for snapshot
        df['ema9'] = ta.trend.ema_indicator(df['close'], window=9)
        df['ema21'] = ta.trend.ema_indicator(df['close'], window=21)
        df['rsi14'] = ta.momentum.rsi(df['close'], window=14)
        latest = df.iloc[-1]
        ema9 = latest['ema9']
        ema21 = latest['ema21']
        macd_val = df['macd'].iloc[-1]
        rsi_val = latest['rsi14']
        _obv_back_pos = min(5, len(df) - 1)
        obv_change = ((df['obv'].iloc[-1] - df['obv'].iloc[-_obv_back_pos]) / abs(df['obv'].iloc[-_obv_back_pos]) * 100) if (_obv_back_pos > 0 and df['obv'].iloc[-_obv_back_pos] != 0) else 0
        recent = df.tail(_sr_lookback(timeframe))
        support = recent['low'].min()
        resistance = recent['high'].max()
        current_price = latest['close']
        support_dist = abs(current_price - support) / current_price * 100
        resistance_dist = abs(current_price - resistance) / current_price * 100
        # Build snapshot HTML
        snapshot_html = f"""
        <div class='panel-box'>
          <b style='color:{ACCENT}; font-size:1.05rem;'>📊 Technical Snapshot</b><br>
          <ul style='color:{TEXT_MUTED}; font-size:0.9rem; line-height:1.5; list-style-position:inside; margin-top:6px;'>
            <li>EMA Trend (9 vs 21): <b>{ema9:.2f}</b> vs <b>{ema21:.2f}</b> {('🟢' if ema9 > ema21 else '🔴')} — When EMA9 is above EMA21 the short‑term trend is bullish; otherwise bearish.</li>
            <li>MACD: <b>{macd_val:.2f}</b> {('🟢' if macd_val > 0 else '🔴')} — Positive MACD indicates upward momentum; negative values suggest downward pressure.</li>
            <li>RSI (14): <b>{rsi_val:.2f}</b> {('🟢' if rsi_val > 55 else ('🟠' if 45 <= rsi_val <= 55 else '🔴'))} — Above 70 may signal overbought, below 30 oversold. Values above 50 favour bulls.</li>
            <li>OBV change (last 5 candles): <b>{obv_change:+.2f}%</b> {('🟢' if obv_change > 0 else '🔴')} — Rising OBV supports the price move; falling OBV warns against continuation.</li>
            <li>Support / Resistance: support at <b>${support:,.2f}</b> ({support_dist:.2f}% away), resistance at <b>${resistance:,.2f}</b> ({resistance_dist:.2f}% away).</li>
          </ul>
        </div>
        """
        st.markdown(snapshot_html, unsafe_allow_html=True)
