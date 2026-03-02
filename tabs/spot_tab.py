from ui.ctx import get_ctx

import numpy as np
import pandas as pd
import plotly.graph_objs as go
import ta
from core.market_decision import (
    ai_vote_metrics,
    action_decision_with_reason,
    action_reason_text,
    normalize_action_class,
    structure_state,
)
from core.signal_contract import strength_from_bias, strength_bucket
from ui.snapshot_cache import live_or_snapshot


def render(ctx: dict) -> None:
    """Render the Spot Trading tab."""
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
    signal_plain = get_ctx(ctx, "signal_plain")
    direction_key = get_ctx(ctx, "direction_key")
    direction_label = get_ctx(ctx, "direction_label")
    format_delta = get_ctx(ctx, "format_delta")
    ml_ensemble_predict = get_ctx(ctx, "ml_ensemble_predict")
    get_price_change = get_ctx(ctx, "get_price_change")
    _calc_conviction = get_ctx(ctx, "_calc_conviction")
    _build_indicator_grid = get_ctx(ctx, "_build_indicator_grid")
    _wma = get_ctx(ctx, "_wma")
    _sr_lookback = get_ctx(ctx, "_sr_lookback")
    _debug = get_ctx(ctx, "_debug")

    def _adx_bucket_only(adx_value: float) -> str:
        try:
            adx_f = float(adx_value)
        except Exception:
            return ""
        if not np.isfinite(adx_f):
            return ""
        if adx_f < 20:
            return "Weak"
        if adx_f < 25:
            return "Starting"
        if adx_f < 50:
            return "Strong"
        if adx_f < 75:
            return "Very Strong"
        return "Extreme"

    def _spot_cache_ttl(tf: str) -> int:
        return {
            "1m": 120,
            "3m": 180,
            "5m": 300,
            "15m": 600,
            "1h": 900,
            "4h": 1800,
            "1d": 3600,
        }.get(str(tf or "").strip(), 900)

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
        f"into a single direction output (Upside / Downside / Neutral) with a direction-agnostic strength score from 0-100%. "
        f"Designed for spot trading without leverage.</p>"
        f"</div>",
        unsafe_allow_html=True,
    )
    st.markdown(
        f"<details style='margin-bottom:0.7rem;'>"
        f"<summary style='color:{ACCENT}; cursor:pointer;'>How to read quickly (?)</summary>"
        f"<div style='color:{TEXT_MUTED}; font-size:0.85rem; line-height:1.7; margin-top:0.45rem;'>"
        f"<b>1.</b> Start with Δ (%) + Setup Confirm + Direction + Strength.<br>"
        f"<b>2.</b> Validate with AI Ensemble + Tech vs AI Alignment before acting.<br>"
        f"<b>3.</b> Use indicator grid and execution levels for context, not standalone triggers."
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
        df, used_cache, cache_ts = live_or_snapshot(
            st,
            f"spot_df::{coin}::{timeframe}",
            df_live,
            max_age_sec=_spot_cache_ttl(timeframe),
            current_sig=(coin, timeframe),
        )
        if used_cache:
            st.warning(f"Live data unavailable. Showing cached snapshot from {cache_ts}.")
        if df is None or len(df) < 60:
            st.error(f"Could not fetch data for **{coin}** on {timeframe}. The coin may not be listed on supported exchanges. Try a major pair (BTC, ETH) or check the symbol.")
            return
        # Keep analysis on closed candles for consistency with signal engine.
        # Always analyse on closed candles; live tick may still be shown separately.
        df_eval = df.iloc[:-1].copy()
        if df_eval is None or len(df_eval) < 55:
            st.error("Not enough closed-candle data for a stable analysis.")
            return

        a = analyse(df_eval)
        signal, _comment, volume_spike = a.signal, a.comment, a.volume_spike
        atr_comment, candle_pattern, bias_score = a.atr_comment, a.candle_pattern, a.bias
        strength_score = float(strength_from_bias(float(bias_score)))
        adx_val, supertrend_trend, ichimoku_trend = a.adx, a.supertrend, a.ichimoku
        stochrsi_k_val, bollinger_bias, vwap_label = a.stochrsi_k, a.bollinger, a.vwap
        psar_trend, williams_label, cci_label = a.psar, a.williams, a.cci

        # Keep displayed reference price aligned with decision inputs (closed candles).
        current_price = df_eval['close'].iloc[-1]
        price_change = None
        delta_note = "Source: selected-timeframe closed candles."
        try:
            prev_close = float(df_eval["close"].iloc[-2])
            last_closed = float(df_eval["close"].iloc[-1])
            if pd.notna(prev_close) and prev_close > 0 and pd.notna(last_closed):
                price_change = ((last_closed / prev_close) - 1.0) * 100.0
        except Exception:
            price_change = None
        if price_change is None:
            try:
                # `coin` is already normalized as BASE/QUOTE (e.g. BTC/USDT).
                fallback = get_price_change(coin)
                if fallback is not None:
                    price_change = float(fallback)
                    delta_note = "Fallback source: ticker percentage (closed-candle delta unavailable)."
            except Exception:
                price_change = None

        # Display summary grid
        signal_dir_raw = signal_plain(signal)
        signal_dir = direction_key(signal_dir_raw)
        signal_clean = direction_label(signal_dir)
        try:
            _ai_prob_s, ai_dir_s, _ai_details_s = ml_ensemble_predict(df_eval)
            agreement = float((_ai_details_s or {}).get("agreement", 0.0))
            directional_agree = float((_ai_details_s or {}).get("directional_agreement", agreement))
            consensus_agree = float((_ai_details_s or {}).get("consensus_agreement", 0.0))
            ai_dir_key = direction_key(ai_dir_s)
            ai_votes, _display_ratio, decision_agreement = ai_vote_metrics(
                ai_dir_key,
                directional_agree,
                consensus_agree,
            )
        except Exception:
            ai_dir_key = "NEUTRAL"
            ai_votes = 0
            decision_agreement = 0.0

        sig_dir_s = signal_dir if signal_dir in {"UPSIDE", "DOWNSIDE"} else "WAIT"
        conv_lbl_s, _conv_c_s = _calc_conviction(sig_dir_s, ai_dir_key, strength_score, decision_agreement)
        structure_val = structure_state(sig_dir_s, ai_dir_key, strength_score, decision_agreement)
        action_raw, action_reason_code = action_decision_with_reason(
            sig_dir_s,
            strength_score,
            structure_val,
            str(conv_lbl_s),
            decision_agreement,
            float(adx_val) if pd.notna(adx_val) else float("nan"),
        )

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

        setup_confirm = _setup_confirm_display(action_raw)
        setup_reason = action_reason_text(action_reason_code)

        sig_c_s = POSITIVE if signal_dir == "UPSIDE" else (NEGATIVE if signal_dir == "DOWNSIDE" else WARNING)
        ai_c_s = POSITIVE if ai_dir_key == "UPSIDE" else (NEGATIVE if ai_dir_key == "DOWNSIDE" else WARNING)
        _s_bucket = strength_bucket(strength_score)
        strength_display = f"{strength_score:.0f}% ({_s_bucket})"
        conf_c_s = POSITIVE if _s_bucket in {"STRONG", "GOOD"} else (WARNING if _s_bucket == "MIXED" else NEGATIVE)
        conv_c_s = POSITIVE if conv_lbl_s == "HIGH" else (WARNING if conv_lbl_s in {"MEDIUM", "TREND"} else NEGATIVE)
        if normalize_action_class(action_raw).startswith("ENTER_"):
            setup_c_s = POSITIVE
        elif normalize_action_class(action_raw) == "WATCH":
            setup_c_s = WARNING
        else:
            setup_c_s = NEGATIVE
        delta_display = format_delta(price_change) if price_change is not None else ""
        delta_c_s = (
            POSITIVE if str(delta_display).strip().startswith("▲")
            else (NEGATIVE if str(delta_display).strip().startswith("▼") else WARNING)
        )
        st.markdown(
            f"<div style='display:grid; grid-template-columns:repeat(auto-fit, minmax(120px, 1fr)); "
            f"gap:4px; background:{CARD_BG}; border-radius:8px; padding:10px; margin:8px 0;'>"
            f"<div style='text-align:center; padding:6px;' title='{delta_note}'>"
            f"<div style='color:{TEXT_MUTED}; font-size:0.7rem; text-transform:uppercase;'>Δ (%)</div>"
            f"<div style='color:{delta_c_s}; font-size:0.85rem; font-weight:600;'>{delta_display or '—'}</div></div>"
            f"<div style='text-align:center; padding:6px;' title='{setup_reason}'>"
            f"<div style='color:{TEXT_MUTED}; font-size:0.7rem; text-transform:uppercase;'>Setup Confirm</div>"
            f"<div style='color:{setup_c_s}; font-size:0.85rem; font-weight:600;'>{setup_confirm}</div></div>"
            f"<div style='text-align:center; padding:6px;' title='Technical side from closed candles (Upside/Downside/Neutral).'>"
            f"<div style='color:{TEXT_MUTED}; font-size:0.7rem; text-transform:uppercase;'>Direction</div>"
            f"<div style='color:{sig_c_s}; font-size:0.85rem; font-weight:600;'>{signal_clean}</div></div>"
            f"<div style='text-align:center; padding:6px;' title='Direction-agnostic signal power from the technical stack (0-100).'>"
            f"<div style='color:{TEXT_MUTED}; font-size:0.7rem; text-transform:uppercase;'>Strength</div>"
            f"<div style='color:{conf_c_s}; font-size:0.85rem; font-weight:600;'>{strength_display}</div></div>"
            f"<div style='text-align:center; padding:6px;' title='Model vote direction and vote count out of 3 models.'>"
            f"<div style='color:{TEXT_MUTED}; font-size:0.7rem; text-transform:uppercase;'>AI Ensemble</div>"
            f"<div style='color:{ai_c_s}; font-size:0.85rem; font-weight:600;'>{direction_label(ai_dir_key)} ({ai_votes}/3)</div></div>"
            f"<div style='text-align:center; padding:6px;' title='How well technical direction and AI direction agree.'>"
            f"<div style='color:{TEXT_MUTED}; font-size:0.7rem; text-transform:uppercase;'>Tech vs AI Alignment</div>"
            f"<div style='color:{conv_c_s}; font-size:0.85rem; font-weight:600;'>{conv_lbl_s}</div></div>"
            f"</div>",
            unsafe_allow_html=True,
        )

        ichi_meta_parts = []
        if a.ichimoku_tk_cross:
            ichi_meta_parts.append(
                f"TK Cross: {a.ichimoku_tk_cross.replace('▲ ', '').replace('▼ ', '').replace('→ ', '')}"
            )
        if a.ichimoku_future_bias:
            ichi_meta_parts.append(
                f"Future Cloud: {a.ichimoku_future_bias.replace('▲ ', '').replace('▼ ', '').replace('→ ', '')}"
            )
        if a.ichimoku_cloud_strength:
            ichi_meta_parts.append(
                f"Cloud Strength: {a.ichimoku_cloud_strength.replace('▲ ', '').replace('▼ ', '').replace('→ ', '')}"
            )
        ichimoku_hover = " | ".join(ichi_meta_parts)

        spike_label = ""
        spike_hover = ""
        if volume_spike:
            try:
                prev_vol_avg = float(df_eval["volume"].iloc[-21:-1].mean()) if len(df_eval) >= 21 else float("nan")
                last_vol = float(df_eval["volume"].iloc[-1]) if len(df_eval) >= 1 else float("nan")
                vol_ratio = (last_vol / prev_vol_avg) if pd.notna(prev_vol_avg) and prev_vol_avg > 0 and pd.notna(last_vol) else float("nan")
            except Exception:
                vol_ratio = float("nan")
            try:
                o = float(df_eval["open"].iloc[-1])
                c = float(df_eval["close"].iloc[-1])
                candle_pct = ((c / o) - 1.0) * 100.0 if pd.notna(o) and pd.notna(c) and o > 0 else float("nan")
                if pd.notna(o) and pd.notna(c) and c > o:
                    spike_label = "▲ Up Spike"
                elif pd.notna(o) and pd.notna(c) and c < o:
                    spike_label = "▼ Down Spike"
                else:
                    spike_label = "→ Spike"
            except Exception:
                candle_pct = float("nan")
                spike_label = "→ Spike"
            parts = []
            if pd.notna(vol_ratio):
                parts.append(f"Vol Ratio: {vol_ratio:.2f}x")
            if pd.notna(candle_pct):
                parts.append(f"Candle: {candle_pct:+.2f}%")
            vwap_ctx = str(vwap_label or "").replace("🟢 ", "").replace("🔴 ", "").replace("→ ", "").strip()
            if vwap_ctx:
                parts.append(f"VWAP: {vwap_ctx}")
            spike_hover = " | ".join(parts)

        # Indicator grid (professional card layout)
        _grid_html = _build_indicator_grid(
            supertrend_trend, ichimoku_trend, vwap_label, adx_val, bollinger_bias,
            stochrsi_k_val, psar_trend, williams_label, cci_label,
            volume_spike, atr_comment, candle_pattern,
            spike_label=spike_label, spike_hover=spike_hover,
            timeframe=timeframe,
            ichimoku_hover=ichimoku_hover,
            adx_label_override=_adx_bucket_only(adx_val),
        )
        if _grid_html:
            st.markdown(_grid_html, unsafe_allow_html=True,
            )

        # Execution levels (spot-only): separate pullback and breakout paths.
        try:
            atr14 = float(ta.volatility.average_true_range(df_eval["high"], df_eval["low"], df_eval["close"], window=14).iloc[-1])
        except Exception:
            atr14 = 0.0
        plan_lookback = _sr_lookback(timeframe)
        plan_recent = df_eval.tail(plan_lookback)
        plan_support = float(plan_recent["low"].min())
        plan_resistance = float(plan_recent["high"].max())
        atr_unit = float(max(atr14, current_price * 0.003))

        # Pullback path
        pullback_low = max(0.0, plan_support - 0.25 * atr_unit)
        pullback_high = plan_support + 0.35 * atr_unit
        pullback_invalidation = max(0.0, plan_support - 0.90 * atr_unit)
        pullback_tp_low = max(plan_support, plan_resistance - 0.10 * atr_unit)
        pullback_tp_high = max(pullback_tp_low, plan_resistance + 0.70 * atr_unit)

        # Breakout path
        breakout_trigger = plan_resistance + 0.20 * atr_unit
        breakout_invalidation = max(0.0, plan_resistance - 0.60 * atr_unit)
        if breakout_invalidation >= breakout_trigger:
            breakout_invalidation = max(0.0, breakout_trigger - 0.60 * atr_unit)
        breakout_tp_low = breakout_trigger + 0.80 * atr_unit
        breakout_tp_high = breakout_trigger + 1.60 * atr_unit

        pullback_zone_text = f"{_fmt_price(pullback_low)}-{_fmt_price(pullback_high)}"
        pullback_tp_text = f"{_fmt_price(pullback_tp_low)}-{_fmt_price(pullback_tp_high)}"
        breakout_tp_text = f"{_fmt_price(breakout_tp_low)}-{_fmt_price(breakout_tp_high)}"

        st.markdown(
            f"<style>"
            f".spot-kpi-row {{"
            f"  display:grid;"
            f"  grid-template-columns:repeat(7, minmax(0, 1fr));"
            f"  gap:0.45rem;"
            f"  margin:0.25rem 0 0.5rem 0;"
            f"}}"
            f".spot-kpi-card {{"
            f"  background:linear-gradient(140deg, rgba(4, 10, 18, 0.96), rgba(2, 5, 11, 0.96));"
            f"  border:1px solid rgba(0, 212, 255, 0.16);"
            f"  border-radius:14px;"
            f"  padding:10px 11px;"
            f"  box-shadow:0 10px 30px rgba(0, 0, 0, 0.35);"
            f"  min-width:0;"
            f"}}"
            f".spot-kpi-label {{"
            f"  color:{TEXT_MUTED};"
            f"  font-size:0.62rem;"
            f"  letter-spacing:0.7px;"
            f"  text-transform:uppercase;"
            f"  font-weight:650;"
            f"  white-space:nowrap;"
            f"  overflow:hidden;"
            f"  text-overflow:ellipsis;"
            f"}}"
            f".spot-kpi-value {{"
            f"  color:{ACCENT};"
            f"  font-family:'Space Grotesk','Manrope',sans-serif;"
            f"  font-size:clamp(0.86rem, 0.95vw, 1.35rem);"
            f"  font-weight:700;"
            f"  margin-top:5px;"
            f"  line-height:1.15;"
            f"  white-space:nowrap;"
            f"  overflow:visible;"
            f"  text-overflow:clip;"
            f"  font-variant-numeric:tabular-nums;"
            f"}}"
            f".spot-kpi-value-range {{"
            f"  font-size:clamp(0.70rem, 0.72vw, 0.96rem);"
            f"}}"
            f".spot-kpi-guide {{"
            f"  border:1px solid rgba(0, 212, 255, 0.14);"
            f"  background:rgba(0, 0, 0, 0.62);"
            f"  border-radius:12px;"
            f"  padding:10px 12px;"
            f"  margin:0 0 0.7rem 0;"
            f"  color:{TEXT_MUTED};"
            f"  font-size:0.85rem;"
            f"  line-height:1.64;"
            f"}}"
            f".spot-kpi-guide b {{ color:{ACCENT}; }}"
            f"</style>"
            f"<div class='spot-kpi-row'>"
            f"<div class='spot-kpi-card' title='Latest close of selected timeframe (not live tick).'>"
            f"<div class='spot-kpi-label'>Latest Candle Close</div><div class='spot-kpi-value'>{_fmt_price(current_price)}</div></div>"
            f"<div class='spot-kpi-card'><div class='spot-kpi-label'>Primary Accumulation Zone</div><div class='spot-kpi-value spot-kpi-value-range'>{pullback_zone_text}</div></div>"
            f"<div class='spot-kpi-card'><div class='spot-kpi-label'>Breakout Entry</div><div class='spot-kpi-value'>{_fmt_price(breakout_trigger)}</div></div>"
            f"<div class='spot-kpi-card'><div class='spot-kpi-label'>Pullback Exit If Broken</div><div class='spot-kpi-value'>{_fmt_price(pullback_invalidation)}</div></div>"
            f"<div class='spot-kpi-card'><div class='spot-kpi-label'>Breakout Exit If Broken</div><div class='spot-kpi-value'>{_fmt_price(breakout_invalidation)}</div></div>"
            f"<div class='spot-kpi-card'><div class='spot-kpi-label'>Pullback TP Zone</div><div class='spot-kpi-value spot-kpi-value-range'>{pullback_tp_text}</div></div>"
            f"<div class='spot-kpi-card'><div class='spot-kpi-label'>Breakout TP Zone</div><div class='spot-kpi-value spot-kpi-value-range'>{breakout_tp_text}</div></div>"
            f"</div>"
            f"<div class='spot-kpi-guide'>"
            f"<b>KPI Quick Guide (Beginner):</b><br>"
            f"1) <b>Latest Candle Close</b>: your current reference price for this timeframe (closed candle).<br>"
            f"2) <b>Primary Accumulation Zone</b>: first buy area. If price revisits this zone and holds, it is your pullback entry area.<br>"
            f"3) <b>Breakout Entry</b>: second buy trigger. If price closes above this level, momentum confirmation appears.<br>"
            f"4) <b>Pullback Exit If Broken</b>: stop level for pullback entry. If price closes below it, pullback setup is invalid.<br>"
            f"5) <b>Breakout Exit If Broken</b>: stop level for breakout entry. If price falls back below it after breakout, momentum setup weakens.<br>"
            f"6) <b>Pullback TP Zone</b>: partial/full take-profit area for pullback entries.<br>"
            f"7) <b>Breakout TP Zone</b>: partial/full take-profit area for breakout entries.<br>"
            f"<span style='font-size:0.78rem;'>Simple workflow: wait for a valid entry trigger, define the matching exit level first, then use the matching TP zone.</span>"
            f"</div>",
            unsafe_allow_html=True,
        )

        setup_cls = normalize_action_class(action_raw)
        setup_label = setup_confirm
        if setup_cls == "SKIP":
            plan_status = "No-Trade"
            plan_color = NEGATIVE
            plan_lines = (
                f"1) <b>Setup Confirm is SKIP:</b> do not open a new spot position on this structure.<br>"
                f"2) <b>Wait for regime improvement:</b> setup should move to WATCH or a confirmed class (TREND+AI / TREND-led / AI-led) before re-evaluation.<br>"
                f"3) <b>If already holding:</b> reduce risk into strength and enforce pullback risk line ({_fmt_price(pullback_invalidation)}).<br>"
                f"4) <b>Keep both paths prepared:</b> Pullback zone ({pullback_zone_text}) and breakout trigger ({_fmt_price(breakout_trigger)}).<br>"
                f"5) <b>Profit maps:</b> Pullback TP ({pullback_tp_text}) / Breakout TP ({breakout_tp_text})."
            )
        elif setup_cls == "WATCH":
            plan_status = "Watch"
            plan_color = WARNING
            if signal_dir == "UPSIDE":
                plan_lines = (
                    f"1) <b>Setup Confirm is WATCH:</b> confirmation is partial; monitor, do not force entry.<br>"
                    f"2) <b>Primary trigger path:</b> reaction quality in Primary Accumulation Zone ({pullback_zone_text}).<br>"
                    f"3) <b>Momentum trigger path:</b> candle close above Breakout Entry ({_fmt_price(breakout_trigger)}).<br>"
                    f"4) <b>Risk discipline:</b> pullback invalidation {_fmt_price(pullback_invalidation)}, breakout invalidation {_fmt_price(breakout_invalidation)}.<br>"
                    f"5) <b>Profit discipline:</b> Pullback TP ({pullback_tp_text}) / Breakout TP ({breakout_tp_text})."
                )
            elif signal_dir == "DOWNSIDE":
                plan_lines = (
                    f"1) <b>Setup Confirm is WATCH with Downside direction:</b> avoid fresh spot buys until reclaim confirmation.<br>"
                    f"2) <b>Reclaim trigger:</b> wait for a close back above Breakout Entry ({_fmt_price(breakout_trigger)}).<br>"
                    f"3) <b>If already holding:</b> reduce risk into strength and protect with pullback invalidation ({_fmt_price(pullback_invalidation)}).<br>"
                    f"4) <b>If reclaim confirms:</b> use breakout invalidation ({_fmt_price(breakout_invalidation)}) and breakout TP ({breakout_tp_text})."
                )
            else:
                plan_lines = (
                    f"1) <b>Setup Confirm is WATCH with Neutral direction:</b> no-force zone until a side confirms.<br>"
                    f"2) <b>Range decision levels:</b> monitor Primary Accumulation Zone ({pullback_zone_text}) "
                    f"and Breakout Entry ({_fmt_price(breakout_trigger)}).<br>"
                    f"3) <b>Execution only after side confirmation:</b> map risk to pullback/breakout invalidation lines.<br>"
                    f"4) <b>Keep exits pre-defined:</b> Pullback TP ({pullback_tp_text}) and Breakout TP ({breakout_tp_text})."
                )
        elif signal_dir == "UPSIDE":
            plan_status = "Bullish Confirmed"
            plan_color = POSITIVE
            plan_lines = (
                f"1) <b>Setup Confirm is {setup_label}:</b> execution-ready upside context.<br>"
                f"2) <b>Pullback path:</b> accumulate in {pullback_zone_text}, invalidate below {_fmt_price(pullback_invalidation)}.<br>"
                f"3) <b>Breakout path:</b> execute on close above {_fmt_price(breakout_trigger)}, invalidate below {_fmt_price(breakout_invalidation)}.<br>"
                f"4) <b>Take profit map:</b> Pullback TP ({pullback_tp_text}), Breakout TP ({breakout_tp_text}).<br>"
                f"5) <b>Risk management:</b> take partials at TP-low, trail remainder only while structure stays intact."
            )
        else:
            plan_status = "Defensive Confirmed"
            plan_color = NEGATIVE
            plan_lines = (
                f"1) <b>Setup Confirm is {setup_label}, but direction is Downside:</b> spot mode stays defensive.<br>"
                f"2) <b>No fresh spot buy</b> until direction recovers and closes above Breakout Entry ({_fmt_price(breakout_trigger)}).<br>"
                f"3) <b>If already holding:</b> de-risk into rallies and protect downside via pullback invalidation ({_fmt_price(pullback_invalidation)}).<br>"
                f"4) <b>Only after reclaim confirmation:</b> use breakout invalidation ({_fmt_price(breakout_invalidation)}) and breakout TP ({breakout_tp_text})."
            )

        st.markdown(
            f"<div class='panel-box' style='border-left:4px solid {plan_color};'>"
            f"<b style='color:{plan_color}; font-size:1rem;'>Spot Execution Plan</b>"
            f"<div style='color:{plan_color}; font-size:0.82rem; margin-top:4px;'><b>Mode:</b> {plan_status}</div>"
            f"<div style='color:{plan_color}; font-size:0.82rem; margin-top:2px;'><b>Setup Confirm:</b> {setup_label}</div>"
            f"<div style='color:{TEXT_MUTED}; font-size:0.86rem; line-height:1.7; margin-top:6px;'>"
            f"{plan_lines}"
            f"<br><span style='color:{TEXT_MUTED}; font-size:0.78rem;'>Guide only, not financial advice. "
            f"Always confirm with your own risk plan.</span>"
            f"</div></div>",
            unsafe_allow_html=True,
        )

        # Use the same closed-candle frame for charts to avoid visual/decision drift.
        chart_df = df_eval.copy()

        # Plot candlestick with EMAs
        fig = go.Figure()
        fig.add_trace(go.Candlestick(
            x=chart_df['timestamp'], open=chart_df['open'], high=chart_df['high'], low=chart_df['low'], close=chart_df['close'],
            increasing_line_color=POSITIVE, decreasing_line_color=NEGATIVE, name="Price"
        ))
        # Plot EMAs
        for window, color in [(5, '#F472B6'), (9, '#60A5FA'), (21, '#FBBF24'), (50, '#FCD34D')]:
            ema_series = ta.trend.ema_indicator(chart_df['close'], window=window)
            fig.add_trace(go.Scatter(x=chart_df['timestamp'], y=ema_series, mode='lines',
                                     name=f"EMA{window}", line=dict(color=color, width=1.5)))
        # Plot weighted moving averages (WMA) for additional insight.  The WMA gives
        # more weight to recent prices and can help identify trend shifts earlier.
        try:
            wma20 = _wma(chart_df['close'], length=20)
            wma50 = _wma(chart_df['close'], length=50)
            fig.add_trace(go.Scatter(x=chart_df['timestamp'], y=wma20, mode='lines',
                                     name="WMA20", line=dict(color='#34D399', width=1, dash='dot')))
            fig.add_trace(go.Scatter(x=chart_df['timestamp'], y=wma50, mode='lines',
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
        with st.expander("Advanced Chart Pack"):
            # RSI chart
            rsi_fig = go.Figure()
            for period, color in [(6, '#D8B4FE'), (14, '#A78BFA'), (24, '#818CF8')]:
                rsi_series = ta.momentum.rsi(chart_df['close'], window=period)
                rsi_fig.add_trace(go.Scatter(
                    x=chart_df['timestamp'], y=rsi_series, mode='lines', name=f"RSI {period}",
                    line=dict(color=color, width=2)
                ))
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
            macd_ind = ta.trend.MACD(chart_df['close'])
            chart_df['macd'] = macd_ind.macd()
            chart_df['macd_signal'] = macd_ind.macd_signal()
            chart_df['macd_diff'] = macd_ind.macd_diff()
            macd_fig = go.Figure()
            macd_fig.add_trace(go.Scatter(
                x=chart_df['timestamp'], y=chart_df['macd'], name="MACD",
                line=dict(color=ACCENT, width=2)
            ))
            macd_fig.add_trace(go.Scatter(
                x=chart_df['timestamp'], y=chart_df['macd_signal'], name="Signal",
                line=dict(color=WARNING, width=2, dash='dot')
            ))
            macd_fig.add_trace(go.Bar(
                x=chart_df['timestamp'], y=chart_df['macd_diff'], name="Histogram",
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
            chart_df['obv'] = ta.volume.on_balance_volume(chart_df['close'], chart_df['volume'])
            volume_fig = go.Figure()
            volume_fig.add_trace(go.Bar(
                x=chart_df['timestamp'], y=chart_df['volume'], name="Volume", marker_color="#6B7280"
            ))
            volume_fig.add_trace(go.Scatter(
                x=chart_df['timestamp'], y=chart_df['obv'], name="OBV",
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
            snap_df = df_eval.copy()
            snap_df['ema9'] = ta.trend.ema_indicator(snap_df['close'], window=9)
            snap_df['ema21'] = ta.trend.ema_indicator(snap_df['close'], window=21)
            snap_df['rsi14'] = ta.momentum.rsi(snap_df['close'], window=14)
            snap_macd = ta.trend.MACD(snap_df['close'])
            snap_df['macd'] = snap_macd.macd()
            snap_df['obv'] = ta.volume.on_balance_volume(snap_df['close'], snap_df['volume'])
            latest = snap_df.iloc[-1]
            ema9 = latest['ema9']
            ema21 = latest['ema21']
            macd_val = snap_df['macd'].iloc[-1]
            rsi_val = latest['rsi14']
            _obv_back_pos = min(5, len(snap_df) - 1)
            obv_change = ((snap_df['obv'].iloc[-1] - snap_df['obv'].iloc[-_obv_back_pos]) / abs(snap_df['obv'].iloc[-_obv_back_pos]) * 100) if (_obv_back_pos > 0 and snap_df['obv'].iloc[-_obv_back_pos] != 0) else 0
            recent = snap_df.tail(_sr_lookback(timeframe))
            support = recent['low'].min()
            resistance = recent['high'].max()
            snapshot_price = latest['close']
            if np.isfinite(snapshot_price) and snapshot_price > 0:
                support_dist = abs(snapshot_price - support) / snapshot_price * 100
                resistance_dist = abs(snapshot_price - resistance) / snapshot_price * 100
            else:
                support_dist = 0.0
                resistance_dist = 0.0
            snapshot_html = f"""
            <div class='panel-box'>
              <b style='color:{ACCENT}; font-size:1.05rem;'>📊 Technical Snapshot</b><br>
              <ul style='color:{TEXT_MUTED}; font-size:0.9rem; line-height:1.5; list-style-position:inside; margin-top:6px;'>
                <li>EMA Trend (9 vs 21): <b>{ema9:.2f}</b> vs <b>{ema21:.2f}</b> {('🟢' if ema9 > ema21 else '🔴')} — When EMA9 is above EMA21 the near-term trend is bullish; otherwise bearish.</li>
                <li>MACD: <b>{macd_val:.2f}</b> {('🟢' if macd_val > 0 else '🔴')} — Positive MACD indicates upward momentum; negative values suggest downward pressure.</li>
                <li>RSI (14): <b>{rsi_val:.2f}</b> {('🟢' if rsi_val > 55 else ('🟠' if 45 <= rsi_val <= 55 else '🔴'))} — Above 70 may signal overbought, below 30 oversold. Values above 50 favour bulls.</li>
                <li>OBV change (last 5 candles): <b>{obv_change:+.2f}%</b> {('🟢' if obv_change > 0 else '🔴')} — Rising OBV supports the price move; falling OBV warns against continuation.</li>
                <li>Support / Resistance: support at <b>${support:,.2f}</b> ({support_dist:.2f}% away), resistance at <b>${resistance:,.2f}</b> ({resistance_dist:.2f}% away).</li>
              </ul>
            </div>
            """
            st.markdown(snapshot_html, unsafe_allow_html=True)
