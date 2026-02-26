from ui.ctx import get_ctx

from datetime import datetime, timezone
import io

import numpy as np
import pandas as pd
import plotly.graph_objs as go
import ta
from core.signal_contract import strength_from_bias, strength_bucket
from core.position_metrics import (
    compute_hard_invalidation,
    compute_health_decision,
    compute_position_pnl,
    estimate_liquidation,
)
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
    _tip = get_ctx(ctx, "_tip")
    _normalize_coin_input = get_ctx(ctx, "_normalize_coin_input")
    _validate_coin_symbol = get_ctx(ctx, "_validate_coin_symbol")
    _symbol_variants = get_ctx(ctx, "_symbol_variants")
    EXCHANGE = get_ctx(ctx, "EXCHANGE")
    fetch_ohlcv = get_ctx(ctx, "fetch_ohlcv")
    analyse = get_ctx(ctx, "analyse")
    signal_plain = get_ctx(ctx, "signal_plain")
    direction_label = get_ctx(ctx, "direction_label")
    ml_ensemble_predict = get_ctx(ctx, "ml_ensemble_predict")
    _calc_conviction = get_ctx(ctx, "_calc_conviction")
    _build_indicator_grid = get_ctx(ctx, "_build_indicator_grid")
    _sr_lookback = get_ctx(ctx, "_sr_lookback")
    _wma = get_ctx(ctx, "_wma")
    _debug = get_ctx(ctx, "_debug")
    get_scalping_entry_target = get_ctx(ctx, "get_scalping_entry_target")
    """Render the Position Analyser tab for evaluating open positions."""
    st.markdown(
        f"<h2 style='color:{ACCENT};margin-bottom:0.5rem;'>Position Analyser</h2>",
        unsafe_allow_html=True,
    )
    st.markdown(
        f"<div class='panel-box'>"
        f"<b style='color:{ACCENT}; font-size:1rem;'>What does this tab show?</b>"
        f"<p style='color:{TEXT_MUTED}; font-size:0.9rem; margin-top:6px; line-height:1.6;'>"
        f"Track and manage open positions. Enter your entry price, leverage, and direction to see "
        f"{_tip('PnL', 'Profit and Loss — your current gain or loss percentage based on entry price vs current price, multiplied by leverage.')} in real-time, "
        f"{_tip('Stop-Loss / Take-Profit', 'Automatically calculated based on ATR (Average True Range). Stop-loss protects against excessive loss, take-profit locks in gains.')} levels, "
        f"and {_tip('liquidation distance', 'How far the price needs to move against you before your position is liquidated at the selected position settings.')}. "
        f"Also shows updated technical signals for the coin while your position is open.</p>"
        f"</div>",
        unsafe_allow_html=True,
    )
    st.markdown(
        f"<details style='margin:0.15rem 0 0.7rem 0;'>"
        f"<summary style='color:{ACCENT}; cursor:pointer; font-size:0.9rem;'>How to read quickly</summary>"
        f"<div style='color:{TEXT_MUTED}; font-size:0.85rem; line-height:1.65; margin-top:0.4rem;'>"
        f"1) Confirm <b>PnL + Liquidation Distance</b> first. "
        f"2) Check <b>Direction / Strength / AI / Alignment</b>. "
        f"3) Respect <b>Technical Invalidation</b> as hard risk line. "
        f"4) Follow the <b>Decision Model</b> action (HOLD / REDUCE / EXIT style)."
        f"</div></details>",
        unsafe_allow_html=True,
    )
    prefill = st.session_state.get("position_prefill_plan")
    if isinstance(prefill, dict):
        try:
            plan_coin = str(prefill.get("coin") or "").strip().upper()
            plan_dir = str(prefill.get("direction") or "LONG").strip().upper()
            plan_entry = float(prefill.get("entry") or 0.0)
            if plan_coin:
                st.session_state["position_coin_input"] = plan_coin
            if plan_dir in {"LONG", "SHORT"}:
                st.session_state["position_direction_input"] = plan_dir
            if plan_entry > 0:
                st.session_state["position_entry_input"] = plan_entry
            st.session_state["position_prefill_note"] = (
                f"Loaded from {prefill.get('source', 'Rapid')}: "
                f"{plan_coin} {plan_dir} | Entry {plan_entry:.6f} | "
                f"SL {prefill.get('sl', 'N/A')} | TP1 {prefill.get('tp1', 'N/A')}"
            )
        except Exception:
            pass
        finally:
            st.session_state.pop("position_prefill_plan", None)

    if st.session_state.get("position_prefill_note"):
        st.info(st.session_state.get("position_prefill_note"))
        st.session_state.pop("position_prefill_note", None)

    # Assign a unique key to avoid StreamlitDuplicateElementId errors
    coin = _normalize_coin_input(st.text_input(
        "Coin (e.g. BTC, ETH, TAO)",
        value="BTC",
        key="position_coin_input",
    ))
    selected_timeframes = st.multiselect("Select up to 3 Timeframes", ['1m', '3m', '5m', '15m', '1h', '4h', '1d'], default=['3m'], max_selections=3)

    default_entry_price: float = 0.0
    for _v in _symbol_variants(coin):
        try:
            ticker = EXCHANGE.fetch_ticker(_v)
            default_entry_price = float(ticker.get('last', 0) or 0)
            break
        except Exception:
            continue

    # Auto-refresh entry field when coin changes (unless value was prefilled from Rapid plan).
    prev_coin = str(st.session_state.get("position_last_coin", "")).strip().upper()
    if coin and coin != prev_coin and default_entry_price > 0:
        st.session_state["position_entry_input"] = float(default_entry_price)
        st.session_state["position_last_coin"] = coin
    elif coin and not prev_coin:
        st.session_state["position_last_coin"] = coin

    entry_price = st.number_input(
        "Entry Price",
        min_value=0.0,
        format="%.4f",
        value=float(st.session_state.get("position_entry_input", default_entry_price)),
        key="position_entry_input",
    )
    leverage = st.number_input("Leverage (x)", min_value=1, max_value=125, value=5, step=1)
    direction = st.selectbox("Position Direction", ["LONG", "SHORT"], key="position_direction_input")
    p1, p2 = st.columns(2)
    with p1:
        margin_used = st.number_input("Margin Used ($)", min_value=0.0, value=1000.0, step=100.0)
    with p2:
        funding_impact_pct = st.number_input(
            "Funding Impact (%)",
            min_value=-5.0,
            max_value=5.0,
            value=0.0,
            step=0.00001,
            format="%.5f",
            help="Signed value on notional. Use negative when you pay funding.",
        )

    # Strict scalp mode is always enabled (non-strict path removed).
   
    if st.button("Analyse Position", type="primary"):
        st.session_state["position_analysis_active"] = True

    if st.session_state.get("position_analysis_active", False):
        _val_err = _validate_coin_symbol(coin)
        if _val_err:
            st.error(_val_err)
            return
        if not selected_timeframes:
            st.error("Select at least one timeframe.")
            return
        report_rows: list[dict] = []
        tf_order = {'1m': 1, '3m': 2, '5m': 3, '15m': 4, '1h': 5, '4h': 6, '1d': 7}
        largest_tf = max(selected_timeframes, key=lambda tf: tf_order[tf])

        tf_tabs = st.tabs(selected_timeframes)

        for idx, tf in enumerate(selected_timeframes):
            with tf_tabs[idx]:
                df_live = fetch_ohlcv(coin, tf, limit=200)
                df, used_cache, cache_ts = live_or_snapshot(st, f"position_df::{coin}::{tf}::200", df_live)
                if used_cache:
                    st.warning(f"{tf}: live data unavailable, using cached snapshot from {cache_ts}.")
                if df is None or len(df) < 55:
                    st.error(f"Not enough data to analyse position for {tf}.")
                    continue

                # Use closed-candle context for stable live decisions.
                df_eval = df.iloc[:-1].copy() if len(df) > 60 else df.copy()
                if df_eval is None or len(df_eval) < 55:
                    st.error(f"Not enough closed-candle data to analyse position for {tf}.")
                    continue

                a = analyse(df_eval)
                signal, comment, volume_spike = a.signal, a.comment, a.volume_spike
                atr_comment, candle_pattern, bias_score = a.atr_comment, a.candle_pattern, a.bias
                strength_score = float(strength_from_bias(float(bias_score)))
                adx_val, supertrend_trend, ichimoku_trend = a.adx, a.supertrend, a.ichimoku
                stochrsi_k_val, bollinger_bias, vwap_label = a.stochrsi_k, a.bollinger, a.vwap
                psar_trend, williams_label, cci_label = a.psar, a.williams, a.cci

                current_price = df['close'].iloc[-1]
                pnl_pack = compute_position_pnl(
                    entry_price=float(entry_price),
                    current_price=float(current_price),
                    direction=direction,
                    leverage=float(leverage),
                    margin_used=float(margin_used),
                    funding_impact_pct=float(funding_impact_pct),
                )
                pnl_percent_raw = float(pnl_pack["raw_pct"])
                pnl_percent = float(pnl_pack["levered_pct"])
                notional = float(pnl_pack["notional"])
                gross_pnl_usd = float(pnl_pack["gross_usd"])
                funding_usd = float(pnl_pack["funding_usd"])
                net_pnl_usd = float(pnl_pack["net_usd"])

                liq_pack = estimate_liquidation(
                    entry_price=float(entry_price),
                    current_price=float(current_price),
                    direction=direction,
                    leverage=float(leverage),
                )
                liq_price = liq_pack["liq_price"]
                liq_dist_pct = liq_pack["distance_pct"]
                liq_note = "Simple estimate"

                col = POSITIVE if pnl_percent > 0 else (WARNING if abs(pnl_percent) < 1 else NEGATIVE)
                icon = '🟢' if pnl_percent > 0 else ('🟠' if abs(pnl_percent) < 1 else '🔴')

                st.markdown(
                    f"<div class='panel-box' style='border-left:4px solid {col}; padding:16px 18px;'>"
                    f"  <div style='display:flex; justify-content:space-between; flex-wrap:wrap; gap:8px;'>"
                    f"    <span style='color:{col}; font-weight:800;'>{icon} {direction} Position ({tf})</span>"
                    f"    <span style='color:{col}; font-weight:800;'>Levered {pnl_percent:+.2f}%</span>"
                    f"  </div>"
                    f"  <div style='color:{TEXT_MUTED}; font-size:0.9rem; margin-top:6px;'>"
                    f"    Entry: <b style='color:{ACCENT};'>${entry_price:,.4f}</b> | "
                    f"Current: <b style='color:{ACCENT};'>${current_price:,.4f}</b> | "
                    f"Raw Move: <b style='color:{ACCENT};'>{pnl_percent_raw:+.2f}%</b>"
                    f"  </div>"
                    f"</div>",
                    unsafe_allow_html=True,
                )
                liq_txt = f"${liq_price:,.4f}" if liq_price is not None else "N/A"
                liq_dist_txt = f"{liq_dist_pct:.2f}%" if liq_dist_pct is not None else "N/A"
                st.markdown(
                    f"<div style='color:{TEXT_MUTED}; font-size:0.82rem; margin-top:-4px; margin-bottom:6px;'>"
                    f"Est. liquidation: <b>{liq_txt}</b> | Distance from current: <b>{liq_dist_txt}</b>"
                    f" | Model: <b>{liq_note}</b>"
                    f"</div>",
                    unsafe_allow_html=True,
                )
                st.markdown(
                    f"<div style='color:{TEXT_MUTED}; font-size:0.82rem; margin-top:-4px; margin-bottom:6px;'>"
                    f"Notional: <b>${notional:,.2f}</b> | Gross PnL: <b>${gross_pnl_usd:,.2f}</b> | "
                    f"Funding: <b>${funding_usd:,.2f}</b> | "
                    f"Net PnL: <b>${net_pnl_usd:,.2f}</b>"
                    f"</div>",
                    unsafe_allow_html=True,
                )
                if liq_dist_pct is not None and leverage >= 10 and liq_dist_pct < 5:
                    st.markdown(
                        f"<div style='background:#2D0A0A; border-left:4px solid {NEGATIVE}; "
                        f"padding:6px 10px; border-radius:4px; margin:4px 0; font-size:0.82rem;'>"
                        f"<span style='color:{NEGATIVE}; font-weight:600;'>Liquidation Risk</span>"
                        f"<span style='color:{TEXT_MUTED};'> — With current settings, liquidation is only "
                        f"{liq_dist_pct:.2f}% away. Tighten risk controls.</span></div>",
                        unsafe_allow_html=True,
                    )

                signal_dir_display = signal_plain(signal)
                signal_clean = direction_label(signal_dir_display)

                # -- AI ensemble prediction for this coin/timeframe --
                try:
                    _ai_prob, ai_dir, _ai_details = ml_ensemble_predict(df_eval)
                except Exception:
                    ai_dir = "NEUTRAL"

                # Alignment: alignment of Direction + AI + Strength
                sig_direction = "LONG" if signal in ['STRONG BUY', 'BUY'] else ("SHORT" if signal in ['STRONG SELL', 'SELL'] else "WAIT")
                conviction_lbl, conviction_c = _calc_conviction(sig_direction, ai_dir, strength_score)

                # Direction / Strength / AI / Alignment summary grid
                sig_color = POSITIVE if signal_dir_display == "LONG" else (NEGATIVE if signal_dir_display == "SHORT" else WARNING)
                ai_color = POSITIVE if ai_dir == "LONG" else (NEGATIVE if ai_dir == "SHORT" else WARNING)
                _s_bucket = strength_bucket(strength_score)
                conf_color = POSITIVE if _s_bucket in {"STRONG", "GOOD"} else (WARNING if _s_bucket == "MIXED" else NEGATIVE)
                summary_row = (
                    f"<div style='text-align:center; padding:6px;'>"
                    f"<div style='color:{TEXT_MUTED}; font-size:0.7rem; text-transform:uppercase;'>Direction</div>"
                    f"<div style='color:{sig_color}; font-size:0.85rem; font-weight:600;'>{signal_clean}</div></div>"
                    f"<div style='text-align:center; padding:6px;'>"
                    f"<div style='color:{TEXT_MUTED}; font-size:0.7rem; text-transform:uppercase;'>Strength</div>"
                    f"<div style='color:{conf_color}; font-size:0.85rem; font-weight:600;'>{strength_score:.0f}%</div></div>"
                    f"<div style='text-align:center; padding:6px;'>"
                    f"<div style='color:{TEXT_MUTED}; font-size:0.7rem; text-transform:uppercase;'>AI Ensemble</div>"
                    f"<div style='color:{ai_color}; font-size:0.85rem; font-weight:600;'>{direction_label(ai_dir)}</div></div>"
                    f"<div style='text-align:center; padding:6px;'>"
                    f"<div style='color:{TEXT_MUTED}; font-size:0.7rem; text-transform:uppercase;'>Alignment</div>"
                    f"<div style='color:{conviction_c}; font-size:0.85rem; font-weight:600;'>{conviction_lbl}</div></div>"
                )
                st.markdown(
                    f"<div style='display:grid; grid-template-columns:repeat(auto-fit, minmax(120px, 1fr)); "
                    f"gap:4px; background:{CARD_BG}; border-radius:8px; padding:10px; margin:8px 0;'>"
                    f"{summary_row}</div>",
                    unsafe_allow_html=True,
                )

                st.markdown(f"<p style='color:{TEXT_MUTED}; font-size:0.88rem;'>{comment}</p>", unsafe_allow_html=True)

                # Market regime warning (ADX < 20)
                if not np.isnan(adx_val) and adx_val < 20:
                    st.markdown(
                        f"<div style='background:#2D1B00; border-left:4px solid {WARNING}; "
                        f"padding:6px 10px; border-radius:4px; margin:4px 0; font-size:0.82rem;'>"
                        f"<span style='color:{WARNING}; font-weight:600;'>Ranging (ADX {adx_val:.0f})</span>"
                        f"<span style='color:{TEXT_MUTED};'> — Signals less reliable.</span></div>",
                        unsafe_allow_html=True,
                    )

                # Risk alert for position
                if direction == sig_direction and strength_score < 50:
                    st.markdown(
                        f"<div style='background:#2D0A0A; border-left:4px solid {NEGATIVE}; "
                        f"padding:6px 10px; border-radius:4px; margin:4px 0; font-size:0.82rem;'>"
                        f"<span style='color:{NEGATIVE}; font-weight:600;'>Low Strength</span>"
                        f"<span style='color:{TEXT_MUTED};'> — Position direction matches signal but strength "
                        f"is only {strength_score:.0f}%. Consider tightening stop-loss.</span></div>",
                        unsafe_allow_html=True,
                    )
                elif direction != sig_direction and sig_direction != "WAIT":
                    st.markdown(
                        f"<div style='background:#2D0A0A; border-left:4px solid {NEGATIVE}; "
                        f"padding:6px 10px; border-radius:4px; margin:4px 0; font-size:0.82rem;'>"
                        f"<span style='color:{NEGATIVE}; font-weight:600;'>Direction Conflict</span>"
                        f"<span style='color:{TEXT_MUTED};'> — Your {direction} position conflicts with "
                        f"the current {sig_direction} signal. Review position validity.</span></div>",
                        unsafe_allow_html=True,
                    )

                df['ema5'] = ta.trend.ema_indicator(df['close'], window=5)
                df['ema13'] = ta.trend.ema_indicator(df['close'], window=13)
                df['ema9'] = ta.trend.ema_indicator(df['close'], window=9)
                df['ema21'] = ta.trend.ema_indicator(df['close'], window=21)
                macd_ind = ta.trend.MACD(df['close'])
                df['macd'] = macd_ind.macd()
                df['macd_signal'] = macd_ind.macd_signal()
                df['macd_diff'] = macd_ind.macd_diff()
                df['rsi6'] = ta.momentum.rsi(df['close'], window=6)
                df['rsi14'] = ta.momentum.rsi(df['close'], window=14)
                df['rsi24'] = ta.momentum.rsi(df['close'], window=24)
                df['obv'] = ta.volume.on_balance_volume(df['close'], df['volume'])

                recent_sr = df.tail(_sr_lookback(tf))
                support_sr = recent_sr['low'].min()
                resistance_sr = recent_sr['high'].max()
                atr14_sr = float(ta.volatility.average_true_range(df_eval['high'], df_eval['low'], df_eval['close'], window=14).iloc[-1])
                inv_pack = compute_hard_invalidation(
                    direction=direction,
                    support=float(support_sr),
                    resistance=float(resistance_sr),
                    atr14=float(atr14_sr),
                    buffer_mult=0.5,
                    current_price=float(current_price),
                )
                invalidation = float(inv_pack["level"])
                invalidated = bool(inv_pack["invalidated"])
                inv_buffer = float(inv_pack["buffer"])

                inv_color = NEGATIVE if invalidated else WARNING
                inv_state = "BROKEN" if invalidated else "ACTIVE"
                inv_action = "EXIT / HEDGE immediately." if invalidated else "Keep as hard risk line."
                inv_side = "above entry" if invalidation > float(entry_price) else "below entry"
                if direction == "LONG":
                    inv_profile = "profit-protect (trailing)" if invalidation > float(entry_price) else "loss-control"
                else:
                    inv_profile = "profit-protect (trailing)" if invalidation < float(entry_price) else "loss-control"
                st.markdown(
                    f"<div class='panel-box' style='border-left:4px solid {inv_color};'>"
                    f"<b style='color:{inv_color};'>Technical Invalidation Line ({tf}): {inv_state}</b><br>"
                    f"<span style='color:{TEXT_MUTED}; font-size:0.84rem;'>"
                    f"Level: <b>${invalidation:,.4f}</b> (ATR buffer {inv_buffer:,.4f}). Action: <b>{inv_action}</b>"
                    f"<br><span style='color:{TEXT_MUTED};'>Relative to entry: <b>{inv_side}</b> "
                    f"({inv_profile}).</span>"
                    f"</span></div>",
                    unsafe_allow_html=True,
                )

                health_pack = compute_health_decision(
                    direction=direction,
                    signal_direction=sig_direction,
                    strength=float(strength_score),
                    conviction_label=conviction_lbl,
                    liq_distance_pct=liq_dist_pct,
                    invalidated=invalidated,
                    levered_pnl_pct=float(pnl_percent),
                )
                health_score = int(health_pack["score"])
                health_label = str(health_pack["label"])
                health_action = str(health_pack["action"])
                health_notes = list(health_pack["notes"])
                health_color = POSITIVE if health_label == "HOLD" else (WARNING if health_label == "REDUCE" else NEGATIVE)

                notes_txt = ", ".join(health_notes) if health_notes else "no major risk flags"
                report_rows.append(
                    {
                        "Timestamp (UTC)": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
                        "Coin": coin,
                        "Timeframe": tf,
                        "Direction": direction,
                        "Entry Price": round(float(entry_price), 6),
                        "Current Price": round(float(current_price), 6),
                        "Leverage (x)": int(leverage),
                        "Margin Used ($)": round(float(margin_used), 2),
                        "Funding Impact (%)": round(float(funding_impact_pct), 5),
                        "PnL Raw (%)": round(float(pnl_percent_raw), 4),
                        "PnL Levered (%)": round(float(pnl_percent), 4),
                        "Notional ($)": round(float(notional), 2),
                        "Gross PnL ($)": round(float(gross_pnl_usd), 2),
                        "Funding ($)": round(float(funding_usd), 2),
                        "Net PnL ($)": round(float(net_pnl_usd), 2),
                        "Est. Liquidation": (round(float(liq_price), 6) if liq_price is not None else None),
                        "Liq Distance (%)": (round(float(liq_dist_pct), 4) if liq_dist_pct is not None else None),
                        "Direction": signal_clean,
                        "Strength (%)": round(float(strength_score), 2),
                        "AI Direction": ai_dir,
                        "Alignment": conviction_lbl,
                        "Technical Invalidation": round(float(invalidation), 6),
                        "Invalidation State": inv_state,
                        "Health Label": health_label,
                        "Health Score": int(health_score),
                        "Health Action": health_action,
                    }
                )
                suggestion = ""

                if direction == "LONG":
                    if current_price < support_sr:
                        suggestion = (
                            f"🔻 Price has broken below the key support at <b>${support_sr:,.4f}</b>.<br>"
                            f"This invalidates the bullish setup. <b>Consider closing the position (stop-out).</b>"
                        )
                    elif current_price < entry_price:
                        suggestion = (
                            f"⚠️ Price is trading below the entry level.<br>"
                            f"Monitor support at <b>${support_sr:,.4f}</b>. If it fails, risk increases significantly.<br>"
                            f"<i>Maintain caution unless support holds and momentum returns.</i>"
                        )
                    elif current_price < resistance_sr:
                        suggestion = (
                            f"📈 Price is above entry but below resistance at <b>${resistance_sr:,.4f}</b>.<br>"
                            f"<i>Consider holding the position. A breakout may offer further upside.</i>"
                        )
                    else:
                        suggestion = (
                            f"🟢 Price has broken above resistance at <b>${resistance_sr:,.4f}</b>.<br>"
                            f"<b>Consider taking partial profits or trailing your stop.</b>"
                        )
                else:
                    if current_price > resistance_sr:
                        suggestion = (
                            f"🔺 Price has broken above key resistance at <b>${resistance_sr:,.4f}</b>.<br>"
                            f"This invalidates the bearish case. <b>Consider closing the position (stop-out).</b>"
                        )
                    elif current_price > entry_price:
                        suggestion = (
                            f"⚠️ Price is above the short entry level.<br>"
                            f"Watch resistance at <b>${resistance_sr:,.4f}</b>. If it holds, the trade may still be valid.<br>"
                            f"<i>Remain cautious—trend may be reversing.</i>"
                        )
                    elif current_price > support_sr:
                        suggestion = (
                            f"📉 Price is below entry, approaching support at <b>${support_sr:,.4f}</b>.<br>"
                            f"<i>Consider holding. Breakdown of support could validate the short setup further.</i>"
                        )
                    else:
                        suggestion = (
                            f"🟢 Price has broken below support at <b>${support_sr:,.4f}</b>.<br>"
                            f"<b>Consider taking partial profits or holding to maximise gain.</b>"
                        )

                st.caption(
                    f"{tf} decision model: {health_label} ({health_score}/100) — {health_action} "
                    f"Drivers: {notes_txt}."
                )

                # === Scalping Setup ===
                df_scalp = df_eval.tail(120).copy()
                
                if df_scalp is not None and len(df_scalp) > 30:
                
                    # Technical calculations
                    df_scalp['ema5'] = df_scalp['close'].ewm(span=5).mean()
                    df_scalp['ema13'] = df_scalp['close'].ewm(span=13).mean()
                    df_scalp['ema21'] = df_scalp['close'].ewm(span=21).mean()
                    df_scalp['atr'] = ta.volatility.average_true_range(df_scalp['high'], df_scalp['low'], df_scalp['close'], window=14)
                    df_scalp['rsi'] = ta.momentum.rsi(df_scalp['close'], window=14)
                    _macd_scalp = ta.trend.MACD(df_scalp['close'])
                    df_scalp['macd'] = _macd_scalp.macd()
                    df_scalp['macd_signal'] = _macd_scalp.macd_signal()
                    df_scalp['macd_diff'] = _macd_scalp.macd_diff()
                    df_scalp['obv'] = ta.volume.on_balance_volume(df_scalp['close'], df_scalp['volume'])
                
                    latest = df_scalp.iloc[-1]
                    close_price = latest['close']
                    ema5_val = latest['ema5']
                    ema13_val = latest['ema13']
                    macd_hist_s = latest['macd_diff']
                    rsi14_val = latest['rsi']
                    _obv_back_scalp = min(5, len(df_scalp) - 1)
                    obv5 = df_scalp['obv'].iloc[-_obv_back_scalp] if _obv_back_scalp > 0 else df_scalp['obv'].iloc[-1]
                    obv_change_s = ((latest['obv'] - obv5) / abs(obv5) * 100) if obv5 != 0 else 0
                    _sr_scalp = _sr_lookback(tf)
                    support_s = df_scalp['low'].tail(_sr_scalp).min()
                    resistance_s = df_scalp['high'].tail(_sr_scalp).max()
                    support_dist_s = abs(close_price - support_s) / close_price * 100
                    resistance_dist_s = abs(close_price - resistance_s) / close_price * 100
                
                    scalping_snapshot_html = f"""
                    <div class='panel-box'>
                      <b style='color:{ACCENT}; font-size:1.02rem;'>Technical Snapshot ({tf})</b><br>
                      <div style='color:{TEXT_MUTED}; font-size:0.86rem; line-height:1.65; margin-top:6px;'>
                        EMA 5/13: <b>${ema5_val:,.2f}</b> / <b>${ema13_val:,.2f}</b> {('🟢' if ema5_val > ema13_val else '🔴')}<br>
                        MACD Hist: <b>{macd_hist_s:.2f}</b> {('🟢' if macd_hist_s > 0 else '🔴')} |
                        RSI14: <b>{rsi14_val:.2f}</b> {('🟢' if rsi14_val > 50 else '🔴')} |
                        OBV Δ: <b>{obv_change_s:+.2f}%</b> {('🟢' if obv_change_s > 0 else '🔴')}<br>
                        S/R: <b>${support_s:,.4f}</b> ({support_dist_s:.2f}%) / <b>${resistance_s:,.4f}</b> ({resistance_dist_s:.2f}%)
                      </div>
                    </div>
                    """
                
                
                    # === Scalping Strategy Call ===
                    scalp_direction, entry_s, target_s, stop_s, rr_ratio, breakout_note = get_scalping_entry_target(
                        df_scalp,
                        bias_score,
                        supertrend_trend,
                        ichimoku_trend,
                        vwap_label,
                        volume_spike,
                        strict_mode=True
                    )
                
                    # === Display Scalping Result ===
                    if scalp_direction:
                        color = POSITIVE if scalp_direction == "LONG" else NEGATIVE
                        icon = "🟢" if scalp_direction == "LONG" else "🔴"
                        st.markdown(
                            f"""
                            <div class='panel-box' style='border-left:4px solid {color};'>
                              <div style='display:flex; justify-content:space-between; gap:8px; flex-wrap:wrap;'>
                                <span style='color:{color}; font-weight:800;'>{icon} Scalping {scalp_direction}</span>
                                <span style='color:{color}; font-weight:700;'>R:R {rr_ratio:.2f}</span>
                              </div>
                              <div style='color:{TEXT_MUTED}; font-size:0.88rem; margin-top:6px; line-height:1.65;'>
                                Your Entry <b style='color:{ACCENT};'>${entry_price:,.4f}</b> |
                                Model Entry <b style='color:{ACCENT};'>${entry_s:,.4f}</b><br>
                                Stop <b style='color:{NEGATIVE};'>${stop_s:,.4f}</b> |
                                Target <b style='color:{POSITIVE};'>${target_s:,.4f}</b><br>
                                {'Good setup quality (R:R ≥ 1.5).' if rr_ratio >= 1.5 else 'R:R is below ideal threshold (1.5).'}
                              </div>
                            </div>
                            """,
                            unsafe_allow_html=True
                        )
                    else:
                        msg = breakout_note or "No valid scalping setup with current filters."
                        if msg in {"Invalid plan levels", "Invalid ATR/price"}:
                            msg = "No valid plan on current candle structure."
                        st.markdown(
                            f"<div class='panel-box' style='border-left:4px solid {WARNING};'>"
                            f"<b style='color:{WARNING};'>Scalping Opportunity: Not Available</b><br>"
                            f"<span style='color:{TEXT_MUTED}; font-size:0.86rem;'>Reason: {msg}</span>"
                            f"</div>",
                            unsafe_allow_html=True,
                        )
                    with st.expander("Advanced Context"):
                        # -- Indicator grid (professional card layout) --
                        _grid_html = _build_indicator_grid(
                            supertrend_trend, ichimoku_trend, vwap_label, adx_val, bollinger_bias,
                            stochrsi_k_val, psar_trend, williams_label, cci_label,
                            volume_spike, atr_comment, candle_pattern,
                        )
                        if _grid_html:
                            st.markdown(_grid_html, unsafe_allow_html=True)
                        st.markdown(scalping_snapshot_html, unsafe_allow_html=True)
                else:
                    st.markdown(
                        f"<div class='panel-box' style='border-left:4px solid {WARNING};'>"
                        f"<b style='color:{WARNING};'>Scalping Opportunity: Not Available</b><br>"
                        f"<span style='color:{TEXT_MUTED}; font-size:0.86rem;'>Reason: Not enough recent candles for scalping model.</span>"
                        f"</div>",
                        unsafe_allow_html=True,
                    )

        df_candle_live = fetch_ohlcv(coin, largest_tf, limit=100)
        df_candle, used_candle_cache, candle_cache_ts = live_or_snapshot(
            st, f"position_chart::{coin}::{largest_tf}::100", df_candle_live
        )
        if used_candle_cache:
            st.warning(f"Candlestick live data unavailable. Showing cached snapshot from {candle_cache_ts}.")
        if df_candle is not None and len(df_candle) >= 30:
            fig_candle = go.Figure()
            fig_candle.add_trace(go.Candlestick(
                x=df_candle['timestamp'], open=df_candle['open'], high=df_candle['high'],
                low=df_candle['low'], close=df_candle['close'],
                increasing_line_color=POSITIVE, decreasing_line_color=NEGATIVE, name="Price"
            ))
            # Plot EMAs
            for window, color in [(5, '#F472B6'), (9, '#60A5FA'), (13, '#A78BFA'), (21, '#FBBF24'), (50, '#FCD34D')]:
                ema_series = ta.trend.ema_indicator(df_candle['close'], window=window)
                fig_candle.add_trace(go.Scatter(x=df_candle['timestamp'], y=ema_series, mode='lines',
                                                name=f"EMA{window}", line=dict(color=color, width=1.5)))
            # Plot weighted moving averages (WMA) for deeper trend insight
            try:
                wma20_c = _wma(df_candle['close'], length=20)
                wma50_c = _wma(df_candle['close'], length=50)
                fig_candle.add_trace(go.Scatter(x=df_candle['timestamp'], y=wma20_c, mode='lines',
                                                name="WMA20", line=dict(color='#34D399', width=1, dash='dot')))
                fig_candle.add_trace(go.Scatter(x=df_candle['timestamp'], y=wma50_c, mode='lines',
                                                name="WMA50", line=dict(color='#10B981', width=1, dash='dash')))
            except Exception as e:
                _debug(f"WMA candlestick overlay error: {e}")
            fig_candle.update_layout(
                height=380,
                template='plotly_dark',
                margin=dict(l=20, r=20, t=30, b=30),
                xaxis_rangeslider_visible=False,
                showlegend=True,
                legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='left', x=0)
            )
            st.markdown(f"<h4 style='color:{ACCENT};'>📈 Candlestick Chart – {largest_tf}</h4>", unsafe_allow_html=True)
            st.plotly_chart(fig_candle, width="stretch")
        else:
            st.warning(f"Not enough data to display candlestick chart for {largest_tf}.")

        if report_rows:
            report_df = pd.DataFrame(report_rows)
            try:
                buffer = io.BytesIO()
                report_df.to_excel(buffer, index=False, sheet_name="PositionReport")
                st.download_button(
                    label="Download Decision Report (Excel)",
                    data=buffer.getvalue(),
                    file_name=f"position_report_{coin.replace('/', '_')}.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    on_click="ignore",
                )
            except Exception:
                csv_bytes = report_df.to_csv(index=False).encode("utf-8")
                st.download_button(
                    label="Download Decision Report (CSV fallback)",
                    data=csv_bytes,
                    file_name=f"position_report_{coin.replace('/', '_')}.csv",
                    mime="text/csv",
                    on_click="ignore",
                )
