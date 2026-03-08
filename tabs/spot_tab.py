from ui.ctx import get_ctx

import html
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
from ui.primitives import render_help_details, render_kpi_grid, render_page_header
from ui.snapshot_cache import live_or_snapshot


def format_spot_price(v: float) -> str:
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


def _spot_ai_fallback_note(details: dict | None) -> str:
    status = str((details or {}).get("status") or "").strip().lower()
    if status == "insufficient_candles":
        return "AI fallback active: not enough candles for a reliable model vote."
    if status == "insufficient_features":
        return "AI fallback active: not enough clean indicator data after warm-up."
    if status == "single_class_window":
        return "AI fallback active: recent training window had only one class."
    if status == "model_exception":
        return "AI fallback active: ensemble model output was unavailable."
    return ""


def _spot_ai_display_value(direction_fn, ai_dir_key: str, ai_votes: int, fallback_note: str) -> str:
    base = f"{direction_fn(ai_dir_key)} ({ai_votes}/3)"
    if fallback_note:
        return f"{direction_fn(ai_dir_key)}* ({ai_votes}/3)"
    return base


def _spot_execution_map_copy(signal_dir: str) -> dict[str, str]:
    key = str(signal_dir or "").strip().upper()
    if key == "DOWNSIDE":
        return {
            "section_title": "Reclaim Map (Defensive)",
            "left_path": "Support Watch",
            "right_path": "Reclaim Path",
            "left_zone": "Watch Zone",
            "right_trigger": "Reclaim Trigger",
            "left_tp": "Recovery TP",
            "right_tp": "Recovery TP",
        }
    if key == "NEUTRAL":
        return {
            "section_title": "Decision Map",
            "left_path": "Support Watch",
            "right_path": "Breakout Watch",
            "left_zone": "Watch Zone",
            "right_trigger": "Breakout Trigger",
            "left_tp": "TP Zone",
            "right_tp": "TP Zone",
        }
    return {
        "section_title": "Execution Map",
        "left_path": "Buy Zone Path",
        "right_path": "Breakout Path",
        "left_zone": "Buy Zone",
        "right_trigger": "Breakout Trigger",
        "left_tp": "TP Zone",
        "right_tp": "TP Zone",
    }


def _spot_axis_tickformat(reference_price: float) -> str:
    try:
        p = float(reference_price)
    except Exception:
        return ",.2f"
    if p >= 1000:
        return ",.2f"
    if p >= 1:
        return ",.4f"
    if p >= 0.01:
        return ",.6f"
    if p >= 0.0001:
        return ",.8f"
    return ",.10f"


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
    format_stochrsi = get_ctx(ctx, "format_stochrsi")
    ml_ensemble_predict = get_ctx(ctx, "ml_ensemble_predict")
    get_price_change = get_ctx(ctx, "get_price_change")
    _calc_conviction = get_ctx(ctx, "_calc_conviction")
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

    _fmt_price = format_spot_price
    render_page_header(
        st,
        title="Spot Trading",
        intro_html=(
            f"Spot-focused decision workspace for a single coin/timeframe. Uses the same core signal engine as Market tab, then maps it into "
            f"<b>Setup Confirm</b> classes (TREND+AI / TREND-led / AI-led / WATCH / SKIP) for non-leverage execution. Combines "
            f"{_tip('Trend', 'EMA crossovers, SuperTrend, Ichimoku Cloud, Parabolic SAR, and ADX indicators.')} (40%), "
            f"{_tip('Momentum', 'RSI, MACD, Stochastic RSI, Williams %R, and CCI indicators.')} (30%), "
            f"{_tip('Volume', 'OBV direction, volume spikes, and VWAP positioning.')} (20%), and "
            f"{_tip('Volatility', 'Bollinger Band width, ATR level, and Keltner Channel breakouts.')} (10%) "
            f"into <b>Direction</b> (Upside / Downside / Neutral) and a direction-agnostic <b>Strength</b> score (0-100). "
            f"All setup/level outputs are based on closed candles."
        ),
    )
    render_help_details(
        st,
        summary="How to read quickly",
        body_html=(
            "<b>1.</b> Start with <b>Setup Snapshot</b>: Δ (%) + Setup Confirm + Direction + Strength.<br>"
            "<b>2.</b> Read <b>Setup Confirm</b> first: TREND+AI strongest, TREND-led/AI-led conditional, WATCH monitor, SKIP no-trade.<br>"
            "<b>3.</b> Validate with <b>AI Ensemble</b> + <b>Tech vs AI Alignment</b>. A <b>*</b> after AI Ensemble means the model fell back to neutral because the current frame did not have reliable ML context.<br>"
            "<b>4.</b> Use <b>Technical Regime Breakdown</b> only as confirmation context (not standalone trigger).<br>"
            "<b>5.</b> In <b>Execution Levels</b>, choose one path (support path or trigger path), define the matching stop first, then the matching TP zone."
        ),
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
        # Keep Spot analysis history depth aligned with Market scanner (500 candles).
        # This avoids setup-confirm drift between tabs for the same coin/timeframe.
        spot_limit = 500
        df_live = fetch_ohlcv(coin, timeframe, limit=spot_limit)
        df, used_cache, cache_ts = live_or_snapshot(
            st,
            f"spot_df::{coin}::{timeframe}::{spot_limit}",
            df_live,
            max_age_sec=_spot_cache_ttl(timeframe),
            current_sig=(coin, timeframe, spot_limit),
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
        signal, volume_spike = a.signal, a.volume_spike
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
        except Exception as e:
            _debug(
                f"Spot delta candle path failed for {coin} ({timeframe}): "
                f"{e.__class__.__name__}: {str(e).strip()}"
            )
            price_change = None
        if price_change is None:
            try:
                # `coin` is already normalized as BASE/QUOTE (e.g. BTC/USDT).
                fallback = get_price_change(coin)
                if fallback is not None:
                    price_change = float(fallback)
                    delta_note = "Fallback source: ticker percentage (closed-candle delta unavailable)."
            except Exception as e:
                _debug(
                    f"Spot delta ticker fallback failed for {coin} ({timeframe}): "
                    f"{e.__class__.__name__}: {str(e).strip()}"
                )
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
            ai_fallback_note = _spot_ai_fallback_note(_ai_details_s)
        except Exception:
            ai_dir_key = "NEUTRAL"
            ai_votes = 0
            decision_agreement = 0.0
            ai_fallback_note = "AI fallback active: ensemble model output was unavailable."

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
            f"<style>"
            f".spot-summary-title{{"
            f"  margin:0.40rem 0 0.28rem 0;"
            f"  text-align:center;"
            f"  color:{TEXT_MUTED};"
            f"  font-size:0.78rem;"
            f"  text-transform:uppercase;"
            f"  letter-spacing:0.45px;"
            f"}}"
            f".spot-summary-wrap{{"
            f"  background:linear-gradient(140deg, rgba(4, 10, 18, 0.95), rgba(2, 5, 11, 0.95));"
            f"  border:1px solid rgba(0, 212, 255, 0.16);"
            f"  border-radius:12px;"
            f"  padding:10px 12px;"
            f"  margin:0.1rem 0 0.48rem 0;"
            f"}}"
            f".spot-summary-grid{{"
            f"  display:flex;"
            f"  flex-wrap:wrap;"
            f"  justify-content:center;"
            f"  gap:0.45rem 0.75rem;"
            f"}}"
            f".spot-summary-item{{"
            f"  flex:0 1 150px;"
            f"  min-width:130px;"
            f"  text-align:center;"
            f"  padding:4px 6px;"
            f"}}"
            f".spot-summary-label{{"
            f"  color:{TEXT_MUTED};"
            f"  font-size:0.67rem;"
            f"  text-transform:uppercase;"
            f"  letter-spacing:0.55px;"
            f"}}"
            f".spot-summary-value{{"
            f"  font-size:1.03rem;"
            f"  font-weight:700;"
            f"  margin-top:3px;"
            f"}}"
            f"</style>"
            f"<div class='spot-summary-title'>Setup Snapshot</div>"
            f"<div class='spot-summary-wrap'><div class='spot-summary-grid'>"
            f"<div class='spot-summary-item' title='{delta_note}'>"
            f"<div class='spot-summary-label'>Δ (%)</div>"
            f"<div class='spot-summary-value' style='color:{delta_c_s};'>{delta_display or '—'}</div></div>"
            f"<div class='spot-summary-item' title='{setup_reason}'>"
            f"<div class='spot-summary-label'>Setup Confirm</div>"
            f"<div class='spot-summary-value' style='color:{setup_c_s};'>{setup_confirm}</div></div>"
            f"<div class='spot-summary-item' title='Technical side from closed candles (Upside/Downside/Neutral).'>"
            f"<div class='spot-summary-label'>Direction</div>"
            f"<div class='spot-summary-value' style='color:{sig_c_s};'>{signal_clean}</div></div>"
            f"<div class='spot-summary-item' title='Direction-agnostic signal power from the technical stack (0-100).'>"
            f"<div class='spot-summary-label'>Strength</div>"
            f"<div class='spot-summary-value' style='color:{conf_c_s};'>{strength_display}</div></div>"
            f"<div class='spot-summary-item' title='{html.escape(ai_fallback_note or 'Model vote direction and vote count out of 3 models.', quote=True)}'>"
            f"<div class='spot-summary-label'>AI Ensemble</div>"
            f"<div class='spot-summary-value' style='color:{ai_c_s};'>{_spot_ai_display_value(direction_label, ai_dir_key, ai_votes, ai_fallback_note)}</div></div>"
            f"<div class='spot-summary-item' title='How well technical direction and AI direction agree.'>"
            f"<div class='spot-summary-label'>Tech vs AI Alignment</div>"
            f"<div class='spot-summary-value' style='color:{conv_c_s};'>{conv_lbl_s}</div></div>"
            f"</div></div>",
            unsafe_allow_html=True,
        )
        render_help_details(
            st,
            summary="Setup Snapshot Guide",
            body_html=(
                "<b>Δ (%)</b> = last closed-candle move on this timeframe.<br>"
                "<b>Setup Confirm</b> = setup quality class, not a standalone buy command.<br>"
                "<b>Direction</b> = technical direction from closed candles only.<br>"
                "<b>Strength</b> = direction-agnostic signal power from the technical stack.<br>"
                "<b>AI Ensemble</b> = 3-model vote summary. A <b>*</b> means the model fell back to neutral because ML context was not reliable.<br>"
                "<b>Tech vs AI Alignment</b> = how well technical direction and AI direction agree."
            ),
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

        def _clean_indicator_text(v: str) -> str:
            txt = str(v or "").strip()
            for token in ["🟢", "🔴", "🟡", "⚪", "🔥", "▲▲", "▲", "▼", "→", "–"]:
                txt = txt.replace(token, "")
            return " ".join(txt.split()).strip()

        def _indicator_color(v: str) -> str:
            u = str(v or "").upper()
            if "VERY STRONG" in u or "EXTREME" in u:
                return POSITIVE
            if "STRONG" in u and "NOT" not in u:
                return POSITIVE
            if "WEAK" in u:
                return NEGATIVE
            if "STARTING" in u:
                return WARNING
            if any(k in u for k in ["BULLISH", "ABOVE", "OVERSOLD", "LOW", "NEAR BOTTOM", "UP SPIKE"]):
                return POSITIVE
            if any(k in u for k in ["BEARISH", "BELOW", "OVERBOUGHT", "HIGH", "NEAR TOP", "DOWN SPIKE"]):
                return NEGATIVE
            return WARNING

        def _indicator_cell(name: str, value: str, tooltip: str) -> str:
            val = _clean_indicator_text(value)
            if not val:
                return ""
            color = _indicator_color(val)
            tip = html.escape(tooltip, quote=True)
            return (
                f"<div class='spot-indicator-item'>"
                f"<div class='spot-indicator-name'>{name}</div>"
                f"<div class='spot-indicator-value' style='color:{color};' title='{tip}'>{val}</div>"
                f"</div>"
            )

        supertrend_txt = _clean_indicator_text(supertrend_trend)
        ichimoku_txt = _clean_indicator_text(ichimoku_trend)
        vwap_txt = _clean_indicator_text(vwap_label)
        adx_txt = _clean_indicator_text(_adx_bucket_only(adx_val))
        bollinger_txt = _clean_indicator_text(bollinger_bias)
        stochrsi_txt = _clean_indicator_text(format_stochrsi(stochrsi_k_val, timeframe=timeframe))
        psar_txt = _clean_indicator_text(psar_trend)
        will_txt = _clean_indicator_text(williams_label)
        cci_txt = _clean_indicator_text(cci_label)
        volume_txt = _clean_indicator_text(spike_label) if volume_spike else ""
        volatility_txt = _clean_indicator_text(atr_comment.replace("▲", "").replace("▼", "").replace("–", ""))
        pattern_txt = _clean_indicator_text(candle_pattern.split(" (")[0] if candle_pattern else "")

        trend_cells = "".join(
            [
                _indicator_cell("SuperTrend", supertrend_txt, "ATR-based trend line direction."),
                _indicator_cell("Ichimoku", ichimoku_txt, f"Cloud trend context. {ichimoku_hover}".strip()),
                _indicator_cell("VWAP", vwap_txt, "Price relative to volume-weighted average price."),
                _indicator_cell("ADX", adx_txt, "Trend strength (not direction)."),
                _indicator_cell("PSAR", psar_txt, "Parabolic SAR trend-following state."),
            ]
        )
        momentum_cells = "".join(
            [
                _indicator_cell("StochRSI", stochrsi_txt, "Momentum pressure zone."),
                _indicator_cell("Williams %R", will_txt, "Range-position momentum signal."),
                _indicator_cell("CCI", cci_txt, "Mean-reversion momentum signal."),
                _indicator_cell("Pattern", pattern_txt, "Latest candle pattern direction."),
            ]
        )
        vol_cells = "".join(
            [
                _indicator_cell("Bollinger", bollinger_txt, "Band location (extension / pullback context)."),
                _indicator_cell("Volatility", volatility_txt, "ATR/band-width regime."),
                _indicator_cell("Volume", volume_txt, f"Abnormal volume event. {spike_hover}".strip()),
            ]
        )

        if trend_cells or momentum_cells or vol_cells:
            st.markdown(
                f"<style>"
                f".spot-indicator-sep{{"
                f"  margin:8px 0 6px 0;"
                f"  text-align:center;"
                f"  color:{TEXT_MUTED};"
                f"  font-size:0.80rem;"
                f"  text-transform:uppercase;"
                f"  letter-spacing:0.45px;"
                f"}}"
                f".spot-indicator-wrap{{"
                f"  display:grid;"
                f"  grid-template-columns:repeat(auto-fit,minmax(250px,1fr));"
                f"  gap:0.55rem;"
                f"  margin:0.2rem 0 0.45rem 0;"
                f"}}"
                f".spot-indicator-group{{"
                f"  background:rgba(0,0,0,0.56);"
                f"  border:1px solid rgba(0, 212, 255, 0.14);"
                f"  border-radius:12px;"
                f"  padding:8px 8px 6px 8px;"
                f"}}"
                f".spot-indicator-group-title{{"
                f"  color:{ACCENT};"
                f"  text-align:center;"
                f"  font-size:0.74rem;"
                f"  text-transform:uppercase;"
                f"  letter-spacing:0.55px;"
                f"  margin-bottom:4px;"
                f"}}"
                f".spot-indicator-grid{{"
                f"  display:flex;"
                f"  flex-wrap:wrap;"
                f"  justify-content:center;"
                f"  gap:4px 10px;"
                f"  align-items:center;"
                f"}}"
                f".spot-indicator-item{{"
                f"  text-align:center;"
                f"  padding:4px 2px;"
                f"  flex:0 1 118px;"
                f"}}"
                f".spot-indicator-name{{"
                f"  color:{TEXT_MUTED};"
                f"  font-size:0.68rem;"
                f"  text-transform:uppercase;"
                f"  letter-spacing:0.5px;"
                f"}}"
                f".spot-indicator-value{{"
                f"  font-size:0.95rem;"
                f"  font-weight:700;"
                f"  margin-top:2px;"
                f"}}"
                f"</style>"
                f"<div class='spot-indicator-sep'>Technical Regime Breakdown (closed-candle context)</div>"
                f"<div class='spot-indicator-wrap'>"
                f"<div class='spot-indicator-group'><div class='spot-indicator-group-title'>Trend Structure</div>"
                f"<div class='spot-indicator-grid'>{trend_cells}</div></div>"
                f"<div class='spot-indicator-group'><div class='spot-indicator-group-title'>Momentum Signals</div>"
                f"<div class='spot-indicator-grid'>{momentum_cells}</div></div>"
                f"<div class='spot-indicator-group'><div class='spot-indicator-group-title'>Volatility & Volume</div>"
                f"<div class='spot-indicator-grid'>{vol_cells}</div></div>"
                f"</div>",
                unsafe_allow_html=True,
            )
            render_help_details(
                st,
                summary="Indicator Guide",
                body_html=(
                    "<b>Trend Structure</b><br>"
                    "SuperTrend = ATR-based trend line bias. Ichimoku = cloud trend context. VWAP = price relative to average traded price. ADX = trend strength only. PSAR = trailing trend state.<br><br>"
                    "<b>Momentum Signals</b><br>"
                    "StochRSI = short-term momentum pressure. Williams %R = range-position momentum. CCI = mean-reversion pressure. Pattern = latest candle pattern context.<br><br>"
                    "<b>Volatility & Volume</b><br>"
                    "Bollinger = band location (extension or pullback), Volatility = ATR/band-width regime, Volume = abnormal volume spike context."
                ),
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
        map_copy = _spot_execution_map_copy(signal_dir)
        left_path_label = str(map_copy.get("left_path") or "Buy Zone Path")
        left_zone_label = str(map_copy.get("left_zone") or "Buy Zone")
        trigger_label = str(map_copy.get("right_trigger") or "Breakout Trigger")
        right_path_label = str(map_copy.get("right_path") or "Breakout Path")
        left_tp_label = str(map_copy.get("left_tp") or "TP Zone")
        right_tp_label = str(map_copy.get("right_tp") or "TP Zone")
        left_stop_context = left_zone_label
        right_stop_context = trigger_label.replace(" Trigger", "").strip() or "Breakout"
        trigger_guide = (
            "upper trigger if a candle closes above this level."
            if trigger_label == "Breakout Trigger"
            else "confirmation level that price must reclaim before a fresh spot buy is considered."
        )

        st.markdown(
            f"<div style='margin:0.3rem 0 0.45rem 0; text-align:center; color:{TEXT_MUTED}; font-size:0.78rem; text-transform:uppercase; letter-spacing:0.45px;'>"
            "Execution Levels (spot-only)"
            "</div>",
            unsafe_allow_html=True,
        )
        render_kpi_grid(
            st,
            items=[
                {
                    "label": "Reference Price",
                    "label_title": "Latest close of selected timeframe (not live tick).",
                    "value": _fmt_price(current_price),
                },
                {
                    "label": left_zone_label,
                    "value": pullback_zone_text,
                    "subtext": f"raw support: {_fmt_price(plan_support)}",
                },
                {
                    "label": trigger_label,
                    "value": _fmt_price(breakout_trigger),
                    "subtext": f"raw resistance: {_fmt_price(plan_resistance)}",
                },
                {"label": f"Stop ({left_stop_context})", "value": _fmt_price(pullback_invalidation)},
                {"label": f"Stop ({right_stop_context})", "value": _fmt_price(breakout_invalidation)},
                {"label": f"{left_tp_label} ({left_zone_label})", "value": pullback_tp_text},
                {"label": f"{right_tp_label} ({right_stop_context})", "value": breakout_tp_text},
            ],
            columns=4,
            align="center",
            card_min_height="132px",
            center_last_row=True,
        )
        render_help_details(
            st,
            summary="Info | KPI Quick Guide",
            body_html=(
                "<ul>"
                "<li><b>Reference Price</b>: latest closed candle for this timeframe.</li>"
                f"<li><b>{html.escape(left_zone_label)}</b>: primary reaction area for the {html.escape(left_path_label.lower())}.</li>"
                f"<li><b>{html.escape(trigger_label)}</b>: {html.escape(trigger_guide)}</li>"
                "<li><b>Stop levels</b>: if price breaks stop, that specific path is invalid.</li>"
                f"<li><b>{html.escape(left_tp_label)}</b> / <b>{html.escape(right_tp_label)}</b>: scale out in the matching path.</li>"
                "</ul>"
                f"<span>Workflow: choose one path ({html.escape(left_path_label)} or {html.escape(right_path_label)}), define the matching stop first, then use the matching take-profit zone.</span>"
            ),
        )
        st.markdown(
            f"<div style='margin:0.15rem 0 0.45rem 0; text-align:center; color:{TEXT_MUTED}; font-size:0.78rem; text-transform:uppercase; letter-spacing:0.45px;'>"
            f"{_spot_execution_map_copy(signal_dir).get('section_title', 'Execution Map')}"
            "</div>",
            unsafe_allow_html=True,
        )
        map_levels = [
            pullback_invalidation,
            breakout_invalidation,
            pullback_low,
            pullback_high,
            plan_support,
            current_price,
            plan_resistance,
            breakout_trigger,
            pullback_tp_low,
            pullback_tp_high,
            breakout_tp_low,
            breakout_tp_high,
        ]
        map_min = min(map_levels)
        map_max = max(map_levels)
        map_span = max(map_max - map_min, current_price * 0.01)
        map_pad = max(map_span * 0.14, current_price * 0.0025)
        map_y0 = max(0.0, map_min - map_pad)
        map_y1 = map_max + map_pad
        left_x0, left_x1 = 10, 44
        right_x0, right_x1 = 56, 90
        left_path_color = POSITIVE if signal_dir == "UPSIDE" else WARNING
        right_path_color = ACCENT if signal_dir == "UPSIDE" else WARNING
        left_fill_color = "rgba(0, 255, 136, 0.18)" if signal_dir == "UPSIDE" else "rgba(255, 209, 102, 0.14)"
        left_tp_fill = "rgba(0, 255, 136, 0.12)" if signal_dir == "UPSIDE" else "rgba(255, 209, 102, 0.08)"
        right_tp_fill = "rgba(34, 211, 238, 0.12)" if signal_dir == "UPSIDE" else "rgba(255, 209, 102, 0.08)"
        trigger_color = ACCENT if signal_dir == "UPSIDE" else WARNING
        left_label_color = "#E8FFF7" if signal_dir == "UPSIDE" else "#FFF3D0"
        right_label_color = "#D8F9FF" if signal_dir == "UPSIDE" else "#FFF3D0"

        exec_fig = go.Figure()

        exec_fig.update_layout(
            height=360,
            template="plotly_dark",
            margin=dict(l=24, r=24, t=18, b=18),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(3,8,15,0.96)",
            xaxis=dict(
                range=[0, 100],
                showgrid=False,
                zeroline=False,
                showticklabels=False,
                fixedrange=True,
            ),
            yaxis=dict(
                range=[map_y0, map_y1],
                showgrid=True,
                gridcolor="rgba(255,255,255,0.06)",
                zeroline=False,
                tickfont=dict(color=TEXT_MUTED, size=11),
                tickprefix="$",
                tickformat=_spot_axis_tickformat(current_price),
                fixedrange=True,
            ),
            showlegend=False,
            shapes=[
                dict(
                    type="rect",
                    xref="x",
                    yref="paper",
                    x0=left_x0 - 2,
                    x1=left_x1 + 2,
                    y0=0.04,
                    y1=0.96,
                    fillcolor="rgba(0, 212, 255, 0.035)",
                    line=dict(color="rgba(0, 212, 255, 0.10)", width=1),
                    layer="below",
                ),
                dict(
                    type="rect",
                    xref="x",
                    yref="paper",
                    x0=right_x0 - 2,
                    x1=right_x1 + 2,
                    y0=0.04,
                    y1=0.96,
                    fillcolor="rgba(124, 58, 237, 0.04)",
                    line=dict(color="rgba(124, 58, 237, 0.12)", width=1),
                    layer="below",
                ),
                dict(
                    type="rect",
                    xref="x",
                    yref="y",
                    x0=left_x0,
                    x1=left_x1,
                    y0=pullback_low,
                    y1=pullback_high,
                    fillcolor=left_fill_color,
                    line=dict(color=left_path_color, width=1.6),
                    layer="below",
                ),
                dict(
                    type="rect",
                    xref="x",
                    yref="y",
                    x0=left_x0,
                    x1=left_x1,
                    y0=pullback_tp_low,
                    y1=pullback_tp_high,
                    fillcolor=left_tp_fill,
                    line=dict(color=left_path_color, width=1.2, dash="dot"),
                    layer="below",
                ),
                dict(
                    type="rect",
                    xref="x",
                    yref="y",
                    x0=right_x0,
                    x1=right_x1,
                    y0=breakout_tp_low,
                    y1=breakout_tp_high,
                    fillcolor=right_tp_fill,
                    line=dict(color=right_path_color, width=1.2, dash="dot"),
                    layer="below",
                ),
                dict(
                    type="line",
                    xref="x",
                    yref="y",
                    x0=left_x0,
                    x1=left_x1,
                    y0=pullback_invalidation,
                    y1=pullback_invalidation,
                    line=dict(color=NEGATIVE, width=2.2),
                    layer="below",
                ),
                dict(
                    type="line",
                    xref="x",
                    yref="y",
                    x0=right_x0,
                    x1=right_x1,
                    y0=breakout_trigger,
                    y1=breakout_trigger,
                    line=dict(color=trigger_color, width=2.2),
                    layer="below",
                ),
                dict(
                    type="line",
                    xref="x",
                    yref="y",
                    x0=right_x0,
                    x1=right_x1,
                    y0=breakout_invalidation,
                    y1=breakout_invalidation,
                    line=dict(color=NEGATIVE, width=2.2),
                    layer="below",
                ),
                dict(
                    type="line",
                    xref="x",
                    yref="y",
                    x0=6,
                    x1=94,
                    y0=plan_support,
                    y1=plan_support,
                    line=dict(color="rgba(0, 212, 255, 0.45)", width=1.2, dash="dot"),
                    layer="below",
                ),
                dict(
                    type="line",
                    xref="x",
                    yref="y",
                    x0=6,
                    x1=94,
                    y0=plan_resistance,
                    y1=plan_resistance,
                    line=dict(color="rgba(0, 212, 255, 0.45)", width=1.2, dash="dot"),
                    layer="below",
                ),
                dict(
                    type="line",
                    xref="x",
                    yref="y",
                    x0=6,
                    x1=94,
                    y0=current_price,
                    y1=current_price,
                    line=dict(color="rgba(255, 255, 255, 0.72)", width=1.6, dash="dash"),
                    layer="below",
                ),
            ],
            annotations=[
                dict(
                    x=(left_x0 + left_x1) / 2,
                    y=1.02,
                    xref="x",
                    yref="paper",
                    text=map_copy["left_path"],
                    showarrow=False,
                    font=dict(color=left_path_color, size=12, family="Space Grotesk, Manrope, sans-serif"),
                ),
                dict(
                    x=(right_x0 + right_x1) / 2,
                    y=1.02,
                    xref="x",
                    yref="paper",
                    text=map_copy["right_path"],
                    showarrow=False,
                    font=dict(color=right_path_color, size=12, family="Space Grotesk, Manrope, sans-serif"),
                ),
                dict(
                    x=(left_x0 + left_x1) / 2,
                    y=(pullback_low + pullback_high) / 2,
                    xref="x",
                    yref="y",
                    text=f"<b>{map_copy['left_zone']}</b><br><span style='font-size:11px'>{pullback_zone_text}</span>",
                    showarrow=False,
                    align="center",
                    font=dict(color=left_label_color, size=12, family="Manrope, Segoe UI, sans-serif"),
                    bgcolor="rgba(8, 14, 24, 0.70)",
                    bordercolor="rgba(255,255,255,0.08)",
                    borderpad=5,
                ),
                dict(
                    x=(left_x0 + left_x1) / 2,
                    y=(pullback_tp_low + pullback_tp_high) / 2,
                    xref="x",
                    yref="y",
                    text=f"<b>{map_copy['left_tp']}</b><br><span style='font-size:11px'>{pullback_tp_text}</span>",
                    showarrow=False,
                    align="center",
                    font=dict(color=left_label_color, size=12, family="Manrope, Segoe UI, sans-serif"),
                    bgcolor="rgba(8, 14, 24, 0.70)",
                    bordercolor="rgba(255,255,255,0.08)",
                    borderpad=5,
                ),
                dict(
                    x=right_x1 - 2,
                    y=breakout_trigger,
                    xref="x",
                    yref="y",
                    text=f"<b>{map_copy['right_trigger']}</b><br><span style='font-size:11px'>{_fmt_price(breakout_trigger)}</span>",
                    showarrow=False,
                    xanchor="right",
                    yshift=18,
                    align="center",
                    font=dict(color=right_label_color, size=12, family="Manrope, Segoe UI, sans-serif"),
                    bgcolor="rgba(8, 14, 24, 0.76)",
                    bordercolor="rgba(255,255,255,0.08)",
                    borderpad=5,
                ),
                dict(
                    x=(right_x0 + right_x1) / 2,
                    y=(breakout_tp_low + breakout_tp_high) / 2,
                    xref="x",
                    yref="y",
                    text=f"<b>{map_copy['right_tp']}</b><br><span style='font-size:11px'>{breakout_tp_text}</span>",
                    showarrow=False,
                    align="center",
                    font=dict(color=right_label_color, size=12, family="Manrope, Segoe UI, sans-serif"),
                    bgcolor="rgba(8, 14, 24, 0.82)",
                    bordercolor="rgba(255,255,255,0.10)",
                    borderpad=5,
                ),
                dict(
                    x=left_x0 + 2,
                    y=pullback_invalidation,
                    xref="x",
                    yref="y",
                    text=f"<b>Stop</b><br><span style='font-size:11px'>{_fmt_price(pullback_invalidation)}</span>",
                    showarrow=False,
                    xanchor="left",
                    yshift=-18,
                    font=dict(color="#FFD6DF", size=11, family="Manrope, Segoe UI, sans-serif"),
                    bgcolor="rgba(8, 14, 24, 0.74)",
                    bordercolor="rgba(255,255,255,0.08)",
                    borderpad=5,
                ),
                dict(
                    x=right_x1 - 2,
                    y=breakout_invalidation,
                    xref="x",
                    yref="y",
                    text=f"<b>Stop</b><br><span style='font-size:11px'>{_fmt_price(breakout_invalidation)}</span>",
                    showarrow=False,
                    xanchor="right",
                    yshift=-18,
                    font=dict(color="#FFD6DF", size=11, family="Manrope, Segoe UI, sans-serif"),
                    bgcolor="rgba(8, 14, 24, 0.74)",
                    bordercolor="rgba(255,255,255,0.08)",
                    borderpad=5,
                ),
                dict(
                    x=left_x0 - 1,
                    y=plan_support,
                    xref="x",
                    yref="y",
                    text=f"Raw Support {_fmt_price(plan_support)}",
                    showarrow=False,
                    xanchor="left",
                    yshift=12,
                    font=dict(color=TEXT_MUTED, size=10, family="Manrope, Segoe UI, sans-serif"),
                    bgcolor="rgba(8, 14, 24, 0.62)",
                    bordercolor="rgba(255,255,255,0.06)",
                    borderpad=4,
                ),
                dict(
                    x=right_x1 + 1,
                    y=plan_resistance,
                    xref="x",
                    yref="y",
                    text=f"Raw Resistance {_fmt_price(plan_resistance)}",
                    showarrow=False,
                    xanchor="right",
                    yshift=12,
                    font=dict(color=TEXT_MUTED, size=10, family="Manrope, Segoe UI, sans-serif"),
                    bgcolor="rgba(8, 14, 24, 0.62)",
                    bordercolor="rgba(255,255,255,0.06)",
                    borderpad=4,
                ),
                dict(
                    x=50,
                    y=current_price,
                    xref="x",
                    yref="y",
                    text=f"Reference {_fmt_price(current_price)}",
                    showarrow=False,
                    xanchor="center",
                    yshift=18,
                    font=dict(color="#E6EDF7", size=10, family="Manrope, Segoe UI, sans-serif"),
                    bgcolor="rgba(8, 14, 24, 0.66)",
                    bordercolor="rgba(255,255,255,0.07)",
                    borderpad=4,
                ),
            ],
        )
        st.plotly_chart(exec_fig, width="stretch")

        setup_cls = normalize_action_class(action_raw)
        setup_label = setup_confirm
        if setup_cls == "SKIP":
            plan_status = "No-Trade"
            plan_color = NEGATIVE
            plan_lines = (
                f"1) <b>Setup Confirm is SKIP:</b> do not open a new spot position on this structure.<br>"
                f"2) <b>Wait for regime improvement:</b> setup should move to WATCH or a confirmed class (TREND+AI / TREND-led / AI-led) before re-evaluation.<br>"
                f"3) <b>If already holding:</b> reduce risk and keep stop ({left_stop_context}) at {_fmt_price(pullback_invalidation)}.<br>"
                f"4) <b>Keep both paths prepared:</b> {left_zone_label} ({pullback_zone_text}) and {trigger_label} ({_fmt_price(breakout_trigger)}).<br>"
                f"5) <b>Take-profit maps:</b> {left_tp_label} ({pullback_tp_text}) / {right_tp_label} ({breakout_tp_text})."
            )
        elif setup_cls == "WATCH":
            plan_status = "Watch"
            plan_color = WARNING
            if signal_dir == "UPSIDE":
                plan_lines = (
                    f"1) <b>Setup Confirm is WATCH:</b> confirmation is partial; monitor, do not force entry.<br>"
                    f"2) <b>Primary trigger path:</b> reaction quality in {left_zone_label} ({pullback_zone_text}).<br>"
                    f"3) <b>Momentum trigger path:</b> candle close above the {trigger_label} ({_fmt_price(breakout_trigger)}).<br>"
                    f"4) <b>Risk discipline:</b> stop ({left_stop_context}) {_fmt_price(pullback_invalidation)}, stop ({right_stop_context}) {_fmt_price(breakout_invalidation)}.<br>"
                    f"5) <b>Take-profit discipline:</b> {left_tp_label} ({pullback_tp_text}) / {right_tp_label} ({breakout_tp_text})."
                )
            elif signal_dir == "DOWNSIDE":
                plan_lines = (
                    f"1) <b>Setup Confirm is WATCH with Downside direction:</b> avoid fresh spot buys until reclaim confirmation.<br>"
                    f"2) <b>{trigger_label}:</b> wait for a close back above the {trigger_label} ({_fmt_price(breakout_trigger)}).<br>"
                    f"3) <b>If already holding:</b> reduce risk and protect with stop ({left_stop_context}) {_fmt_price(pullback_invalidation)}.<br>"
                    f"4) <b>If reclaim confirms:</b> use stop ({right_stop_context}) {_fmt_price(breakout_invalidation)} and {right_tp_label} ({breakout_tp_text})."
                )
            else:
                plan_lines = (
                    f"1) <b>Setup Confirm is WATCH with Neutral direction:</b> no-force zone until a side confirms.<br>"
                    f"2) <b>Range decision levels:</b> monitor {left_zone_label} ({pullback_zone_text}) and the {trigger_label} ({_fmt_price(breakout_trigger)}).<br>"
                    f"3) <b>Execution only after side confirmation:</b> map risk to stop ({left_stop_context} / {right_stop_context}).<br>"
                    f"4) <b>Keep exits pre-defined:</b> {left_tp_label} ({pullback_tp_text}) and {right_tp_label} ({breakout_tp_text})."
                )
        elif signal_dir == "UPSIDE":
            plan_status = "Bullish Confirmed"
            plan_color = POSITIVE
            plan_lines = (
                f"1) <b>Setup Confirm is {setup_label}:</b> execution-ready upside context.<br>"
                f"2) <b>{left_path_label}:</b> accumulate in {pullback_zone_text}, stop at {_fmt_price(pullback_invalidation)}.<br>"
                f"3) <b>{right_path_label}:</b> execute on close above the {trigger_label} ({_fmt_price(breakout_trigger)}), stop at {_fmt_price(breakout_invalidation)}.<br>"
                f"4) <b>Take-profit map:</b> {left_tp_label} ({pullback_tp_text}), {right_tp_label} ({breakout_tp_text}).<br>"
                f"5) <b>Risk management:</b> take partials at TP-low, trail remainder only while structure stays intact."
            )
        else:
            plan_status = "Defensive Confirmed"
            plan_color = NEGATIVE
            plan_lines = (
                f"1) <b>Setup Confirm is {setup_label}, but direction is Downside:</b> spot mode stays defensive.<br>"
                f"2) <b>No fresh spot buy</b> until direction recovers and closes above the {trigger_label} ({_fmt_price(breakout_trigger)}).<br>"
                f"3) <b>If already holding:</b> de-risk into rallies and protect downside with stop ({left_stop_context}) {_fmt_price(pullback_invalidation)}.<br>"
                f"4) <b>Only after reclaim confirmation:</b> use stop ({right_stop_context}) {_fmt_price(breakout_invalidation)} and {right_tp_label} ({breakout_tp_text})."
            )

        st.markdown(
            f"<div class='panel-box' style='border-left:4px solid {plan_color};'>"
            f"<b style='color:{plan_color}; font-size:1rem;'>Spot Execution Plan</b>"
            f"<div style='color:{plan_color}; font-family:\"Manrope\",\"Segoe UI\",sans-serif; font-size:0.86rem; font-weight:500; line-height:1.7; margin-top:4px;'><b>Mode:</b> {plan_status}</div>"
            f"<div style='color:{plan_color}; font-family:\"Manrope\",\"Segoe UI\",sans-serif; font-size:0.86rem; font-weight:500; line-height:1.7; margin-top:2px;'><b>Setup Confirm:</b> {setup_label}</div>"
            f"<div style='color:{TEXT_MUTED}; font-family:\"Manrope\",\"Segoe UI\",sans-serif; font-size:0.86rem; font-weight:500; line-height:1.7; margin-top:6px;'>"
            f"{plan_lines}"
            f"<br><span style='color:{TEXT_MUTED}; font-family:\"Manrope\",\"Segoe UI\",sans-serif; font-size:0.86rem; font-weight:500; line-height:1.7;'>Guide only, not financial advice. "
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
