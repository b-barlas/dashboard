from ui.ctx import get_ctx

from datetime import datetime, timezone
import html
import io

import numpy as np
import pandas as pd
import plotly.graph_objs as go
import ta
from core.ai_spot_bias import (
    ai_spot_bias_consensus_agreement,
    ai_spot_bias_directional_agreement,
    ai_spot_bias_display_votes,
    ai_spot_bias_probability_up,
    ai_spot_bias_status,
    build_ai_spot_bias_snapshot,
)
from core.confidence import (
    ai_confidence_bucket,
    build_ai_confidence_snapshot,
    build_confidence_snapshot,
    build_execution_confidence_snapshot,
    confidence_bucket,
)
from core.market_decision import (
    ai_led_confirmation_snapshot,
    action_reason_text,
    normalize_action_class,
    selected_timeframe_execution_snapshot,
    selected_timeframe_rr_ratio,
    spot_action_decision_with_reason,
    structure_state,
    trend_led_confirmation_snapshot,
)
from core.market_decision import ai_vote_metrics
from core.scalping import scalp_gate_thresholds
from core.signal_contract import bias_confidence_from_bias
from core.position_metrics import (
    compute_hard_invalidation,
    compute_health_decision,
    compute_position_pnl,
    estimate_liquidation,
)
from core.spot_direction import build_spot_direction_snapshot
from ui.primitives import render_help_details, render_page_header
from ui.snapshot_cache import live_or_snapshot


def _prepare_closed_frame(df: pd.DataFrame | None, *, min_rows: int = 55) -> pd.DataFrame | None:
    if df is None:
        return None
    if len(df) <= int(min_rows):
        return None
    df_eval = df.iloc[:-1].copy()
    if len(df_eval) < int(min_rows):
        return None
    return df_eval


def _direction_fetch_symbol(symbol: str, actual_symbol: str, source_provider: str) -> str:
    # Use the canonical requested symbol for HTF context fetches so spot
    # Direction/AI semantics do not drift with selected-timeframe provider
    # resolution.
    return str(symbol or actual_symbol or "").strip()


def _confidence_display(score: float) -> str:
    score_f = max(0.0, min(100.0, float(score)))
    return f"{score_f:.0f}% ({confidence_bucket(score_f).title()})"


def _ai_confidence_display(snapshot, score: float) -> str:
    score_f = max(0.0, min(100.0, float(score)))
    label = ai_confidence_bucket(
        score_f,
        direction=str(snapshot.direction or ""),
        support_votes=int(ai_spot_bias_display_votes(snapshot)),
        timeframe_conflict=bool(snapshot.timeframe_conflict),
        degraded_data=bool(snapshot.degraded_data),
    )
    return f"{score_f:.0f}% ({label.title()})"


def _spot_bias_label(direction: str) -> str:
    raw = str(direction or "").strip().upper()
    if raw == "UPSIDE":
        return "Upside"
    if raw == "DOWNSIDE":
        return "Downside"
    return "Neutral"


def _spot_tf_note(snapshot) -> str:
    return (
        f"{str(snapshot.timeframe).upper()}: {_spot_bias_label(snapshot.direction)} | "
        f"Score {float(snapshot.score):.1f} | "
        f"Structure {snapshot.structure_label} ({float(snapshot.structure_score):.0f}) | "
        f"Trend {float(snapshot.trend_score):.0f} | "
        f"Regime {snapshot.regime_label} ({float(snapshot.regime_quality):.0f}) | "
        f"Location {float(snapshot.location_quality):.0f}"
    )


def _ai_spot_tf_note(snapshot) -> str:
    status = str(getattr(snapshot, "status", "") or "").strip()
    note = str(getattr(snapshot, "note", "") or "").strip()
    suffix_parts = []
    if status:
        suffix_parts.append(f"Status {status}")
    if note:
        suffix_parts.append(note)
    suffix = f" | {' | '.join(suffix_parts)}" if suffix_parts else ""
    return (
        f"{str(snapshot.timeframe).upper()}: {_spot_bias_label(snapshot.direction)} | "
        f"Score {float(snapshot.score):.1f} | "
        f"Prob Up {float(snapshot.probability_up) * 100:.0f}% | "
        f"Directional agreement {float(snapshot.directional_agreement) * 100:.0f}% | "
        f"Consensus {float(snapshot.consensus_agreement) * 100:.0f}%{suffix}"
    )


def _ai_spot_note(snapshot) -> str:
    dots = ai_spot_bias_display_votes(snapshot)
    return (
        f"AI spot bias (1D + 4H): {_spot_bias_label(snapshot.direction)} | "
        f"Combined score {float(snapshot.score):.1f} | "
        f"Conviction quality {float(snapshot.conviction_quality):.0f} | "
        f"Timeframe alignment {float(snapshot.timeframe_alignment):.0f} | "
        f"Displayed model-support dots {dots}/3 | "
        f"{str(snapshot.note or '').strip()} | "
        f"{_ai_spot_tf_note(snapshot.one_day)} | "
        f"{_ai_spot_tf_note(snapshot.four_hour)}"
    )


def _ai_confidence_note(snapshot, score: float) -> str:
    dots = ai_spot_bias_display_votes(snapshot)
    caps: list[str] = []
    if str(snapshot.direction or "").strip().upper() == "NEUTRAL":
        caps.append("neutral-verdict cap <=58")
    if bool(snapshot.timeframe_conflict):
        caps.append("timeframe-conflict cap <=30")
    if bool(snapshot.degraded_data):
        caps.append("degraded-data cap <=35")
    if str(snapshot.direction or "").strip().upper() != "NEUTRAL" and int(dots) <= 1:
        caps.append("low-model-support cap <=59")
    cap_text = f" | Active caps: {', '.join(caps)}" if caps else ""
    return (
        f"AI confidence: {float(score):.1f}% ({ai_confidence_bucket(float(score), direction=str(snapshot.direction or ''), support_votes=int(dots), timeframe_conflict=bool(snapshot.timeframe_conflict), degraded_data=bool(snapshot.degraded_data)).title()}) | "
        f"HTF AI verdict {_spot_bias_label(snapshot.direction)} | "
        f"Combined score {float(snapshot.score):.1f} | "
        f"Conviction quality {float(snapshot.conviction_quality):.0f} | "
        f"Timeframe alignment {float(snapshot.timeframe_alignment):.0f} | "
        f"Consensus quality {float(snapshot.consensus_quality):.0f} | "
        f"Model support {int(dots)}/3{cap_text}"
    )


def render(ctx: dict) -> None:
    """Render the Position Analyser tab for evaluating open positions."""
    st = get_ctx(ctx, "st")
    ACCENT = get_ctx(ctx, "ACCENT")
    TEXT_MUTED = get_ctx(ctx, "TEXT_MUTED")
    POSITIVE = get_ctx(ctx, "POSITIVE")
    NEGATIVE = get_ctx(ctx, "NEGATIVE")
    WARNING = get_ctx(ctx, "WARNING")
    _tip = get_ctx(ctx, "_tip")
    _normalize_coin_input = get_ctx(ctx, "_normalize_coin_input")
    _validate_coin_symbol = get_ctx(ctx, "_validate_coin_symbol")
    _symbol_variants = get_ctx(ctx, "_symbol_variants")
    EXCHANGE = get_ctx(ctx, "EXCHANGE")
    fetch_ohlcv = get_ctx(ctx, "fetch_ohlcv")
    analyse = get_ctx(ctx, "analyse")
    signal_plain = get_ctx(ctx, "signal_plain")
    direction_key = get_ctx(ctx, "direction_key")
    direction_label = get_ctx(ctx, "direction_label")
    format_delta = get_ctx(ctx, "format_delta")
    format_stochrsi = get_ctx(ctx, "format_stochrsi")
    ml_ensemble_predict = get_ctx(ctx, "ml_ensemble_predict")
    _calc_conviction = get_ctx(ctx, "_calc_conviction")
    _sr_lookback = get_ctx(ctx, "_sr_lookback")
    _wma = get_ctx(ctx, "_wma")
    _debug = get_ctx(ctx, "_debug")
    get_scalping_entry_target = get_ctx(ctx, "get_scalping_entry_target")
    scalp_quality_gate = get_ctx(ctx, "scalp_quality_gate")
    render_page_header(
        st,
        title="Position Analyser",
        intro_html=(
            f"Track and manage open positions. Enter your entry price, leverage, and direction to see "
            f"{_tip('PnL', 'Profit and Loss — your current gain or loss percentage based on entry price vs current price, multiplied by leverage.')} in real-time, "
            f"{_tip('Stop-Loss / Take-Profit', 'Automatically calculated based on ATR (Average True Range). Stop-loss protects against excessive loss, take-profit locks in gains.')} levels, "
            f"and {_tip('liquidation distance', 'How far the price needs to move against you before your position is liquidated at the selected position settings.')}. "
            f"Also shows updated technical signals for the coin while your position is open."
        ),
    )
    render_help_details(
        st,
        summary="How to read quickly",
        body_html=(
            "1) Confirm <b>PnL + Liquidation Distance</b> first. "
            "2) Check <b>Direction / Confidence / HTF AI / AI Confidence</b>. "
            "3) Respect <b>Technical Invalidation</b> as hard risk line. "
            "4) Follow the <b>Decision Model</b> action (HOLD / REDUCE / EXIT style)."
        ),
    )

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
        tip = str(tooltip or "").replace("'", "&#39;")
        return (
            f"<div class='spot-indicator-item'>"
            f"<div class='spot-indicator-name'>{name}</div>"
            f"<div class='spot-indicator-value' style='color:{color};' title='{tip}'>{val}</div>"
            f"</div>"
        )
    # Assign a unique key to avoid StreamlitDuplicateElementId errors
    coin = _normalize_coin_input(st.text_input(
        "Coin (e.g. BTC, ETH, TAO)",
        value="BTC",
        key="position_coin_input",
    ))
    selected_timeframes = st.multiselect(
        "Select up to 3 Timeframes",
        ['1m', '3m', '5m', '15m', '1h', '4h', '1d'],
        default=['15m'],
        max_selections=3,
        key="position_timeframes",
    )

    default_entry_price: float = 0.0
    for _v in _symbol_variants(coin):
        try:
            ticker = EXCHANGE.fetch_ticker(_v)
            default_entry_price = float(ticker.get('last', 0) or 0)
            break
        except Exception:
            continue

    # Auto-refresh entry field when coin changes.
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
    _raw_dir = str(st.session_state.get("position_direction_input", "Upside")).strip()
    if _raw_dir.upper() == "LONG":
        _raw_dir = "Upside"
    elif _raw_dir.upper() == "SHORT":
        _raw_dir = "Downside"
    if _raw_dir not in {"Upside", "Downside"}:
        _raw_dir = "Upside"
    st.session_state["position_direction_input"] = _raw_dir
    direction_ui = st.selectbox("Position Direction", ["Upside", "Downside"], key="position_direction_input")
    direction = "LONG" if direction_ui == "Upside" else "SHORT"
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
        live_position_price: float | None = None
        for _v in _symbol_variants(coin):
            try:
                _ticker = EXCHANGE.fetch_ticker(_v)
                _last = float(_ticker.get("last", 0) or 0)
                if _last > 0:
                    live_position_price = _last
                    break
            except Exception:
                continue

        tf_tabs = st.tabs(selected_timeframes)

        for idx, tf in enumerate(selected_timeframes):
            with tf_tabs[idx]:
                df_live = fetch_ohlcv(coin, tf, limit=200)
                df, used_cache, cache_ts = live_or_snapshot(
                    st,
                    f"position_df::{coin}::{tf}::200",
                    df_live,
                    max_age_sec=900,
                    current_sig=(coin, tf, 200),
                )
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

                actual_symbol = str(df.attrs.get("source_symbol") or "").strip() or coin
                source_provider = str(df.attrs.get("source_provider") or "").strip() or "exchange"
                direction_fetch_symbol = _direction_fetch_symbol(coin, actual_symbol, source_provider)
                # Keep HTF context stable across selected-timeframe changes.
                df_4h_raw = fetch_ohlcv(direction_fetch_symbol, "4h", limit=260)
                df_1d_raw = fetch_ohlcv(direction_fetch_symbol, "1d", limit=260)
                df_direction_4h = _prepare_closed_frame(df_4h_raw, min_rows=81)
                df_direction_1d = _prepare_closed_frame(df_1d_raw, min_rows=81)
                spot_snapshot = build_spot_direction_snapshot(
                    df_4h=df_direction_4h,
                    df_1d=df_direction_1d,
                )
                confidence_snapshot = build_confidence_snapshot(
                    direction=spot_snapshot.direction,
                    timeframe_alignment=spot_snapshot.timeframe_alignment,
                    structure_quality=spot_snapshot.structure_quality,
                    trend_quality=spot_snapshot.trend_quality,
                    regime_quality=spot_snapshot.regime_quality,
                    location_quality=spot_snapshot.location_quality,
                    timeframe_conflict=spot_snapshot.timeframe_conflict,
                    degraded_data=spot_snapshot.degraded_data,
                    range_regime=spot_snapshot.range_regime,
                )
                ai_spot_snapshot = build_ai_spot_bias_snapshot(
                    df_4h=df_direction_4h,
                    df_1d=df_direction_1d,
                    predictor=ml_ensemble_predict,
                )

                a = analyse(df_eval)
                signal, volume_spike = a.signal, a.volume_spike
                atr_comment, candle_pattern, bias_score = a.atr_comment, a.candle_pattern, a.bias
                directional_confidence = float(bias_confidence_from_bias(float(bias_score)))
                adx_val, supertrend_trend, ichimoku_trend = a.adx, a.supertrend, a.ichimoku
                stochrsi_k_val, bollinger_bias, vwap_label = a.stochrsi_k, a.bollinger, a.vwap
                psar_trend, williams_label, cci_label = a.psar, a.williams, a.cci

                # Keep open-position PnL/liquidation on one live price source across TF tabs.
                current_price = float(live_position_price) if live_position_price and live_position_price > 0 else float(df["close"].iloc[-1])
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

                pos_side_label = direction_label(direction)
                st.markdown(
                    f"<div class='panel-box' style='border-left:4px solid {col}; padding:16px 18px;'>"
                    f"  <div style='display:flex; justify-content:space-between; flex-wrap:wrap; gap:8px;'>"
                    f"    <span style='color:{col}; font-weight:800;'>{pos_side_label} Position ({tf})</span>"
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

                signal_dir = direction_key(signal_plain(signal))
                signal_clean = direction_label(signal_dir)
                signal_dir_legacy = (
                    "LONG" if signal_dir == "UPSIDE" else ("SHORT" if signal_dir == "DOWNSIDE" else "WAIT")
                )

                # -- AI ensemble prediction for this coin/timeframe --
                ai_votes = 0
                try:
                    _ai_prob, ai_dir_raw, ai_details = ml_ensemble_predict(df_eval)
                    ai_dir = direction_key(ai_dir_raw)
                    directional_agreement = float(
                        (ai_details or {}).get(
                            "directional_agreement",
                            (ai_details or {}).get("agreement", 0.0),
                        )
                    )
                    consensus_agreement = float(
                        (ai_details or {}).get("consensus_agreement", directional_agreement)
                    )
                    ai_votes, _, decision_agreement = ai_vote_metrics(
                        ai_dir,
                        directional_agreement,
                        consensus_agreement,
                    )
                except Exception:
                    ai_dir = "NEUTRAL"
                    decision_agreement = 0.0

                base_conviction_lbl, _ = _calc_conviction(
                    signal_dir,
                    ai_dir,
                    directional_confidence,
                    decision_agreement,
                )
                tactical_structure = structure_state(signal_dir, ai_dir, directional_confidence, decision_agreement)
                execution_confidence = build_execution_confidence_snapshot(
                    direction=signal_dir,
                    bias_score=float(bias_score),
                    adx_val=float(adx_val) if pd.notna(adx_val) else float("nan"),
                    structure_state=tactical_structure,
                    conviction_label=str(base_conviction_lbl),
                    ai_agreement=float(decision_agreement),
                )
                conviction_lbl, _ = _calc_conviction(signal_dir, ai_dir, float(execution_confidence.score), decision_agreement)
                ai_spot_direction = direction_key(ai_spot_snapshot.direction)
                ai_spot_votes = ai_spot_bias_display_votes(ai_spot_snapshot)
                ai_confidence_snapshot = build_ai_confidence_snapshot(
                    direction=ai_spot_snapshot.direction,
                    combined_score=float(ai_spot_snapshot.score),
                    conviction_quality=float(ai_spot_snapshot.conviction_quality),
                    timeframe_alignment=float(ai_spot_snapshot.timeframe_alignment),
                    consensus_quality=float(ai_spot_snapshot.consensus_quality),
                    support_votes=int(ai_spot_votes),
                    timeframe_conflict=bool(ai_spot_snapshot.timeframe_conflict),
                    degraded_data=bool(ai_spot_snapshot.degraded_data),
                )
                ai_spot_agreement = float(ai_spot_bias_directional_agreement(ai_spot_snapshot))
                ai_spot_consensus = float(ai_spot_bias_consensus_agreement(ai_spot_snapshot))
                ai_spot_probability_up = float(ai_spot_bias_probability_up(ai_spot_snapshot))
                ai_spot_status = str(ai_spot_bias_status(ai_spot_snapshot) or "")
                df_scalp = df_eval.tail(120).copy()
                scalp_direction = None
                entry_s = target_s = stop_s = rr_ratio = 0.0
                breakout_note = ""
                if df_scalp is not None and len(df_scalp) > 30:
                    try:
                        scalp_direction, entry_s, target_s, stop_s, rr_ratio, breakout_note = get_scalping_entry_target(
                            df_scalp,
                            bias_score,
                            supertrend_trend,
                            ichimoku_trend,
                            vwap_label,
                        )
                    except Exception:
                        scalp_direction = None
                        entry_s = target_s = stop_s = rr_ratio = 0.0
                        breakout_note = ""

                execution_snapshot = selected_timeframe_execution_snapshot(
                    df=df_eval,
                    direction=spot_snapshot.direction,
                    bias_score=float(bias_score),
                    adx_val=float(adx_val) if pd.notna(adx_val) else float("nan"),
                    supertrend_trend=str(supertrend_trend),
                    ichimoku_trend=str(ichimoku_trend),
                    vwap_label=str(vwap_label),
                    psar_trend=str(psar_trend),
                    bollinger_bias=str(bollinger_bias),
                    williams_label=str(williams_label),
                    cci_label=str(cci_label),
                )
                setup_rr_ratio = float(selected_timeframe_rr_ratio(execution_snapshot, direction=spot_snapshot.direction))
                trend_led_snapshot = trend_led_confirmation_snapshot(
                    spot_dir=spot_snapshot.direction,
                    spot_confidence=float(confidence_snapshot.score),
                    tactical_dir=signal_dir,
                    adx_val=float(adx_val) if pd.notna(adx_val) else float("nan"),
                    structure_quality=float(execution_snapshot.structure_quality),
                    trend_quality=float(execution_snapshot.trend_quality),
                    regime_quality=float(execution_snapshot.regime_quality),
                    location_quality=float(execution_snapshot.location_quality),
                    rr_ratio=setup_rr_ratio if np.isfinite(setup_rr_ratio) and setup_rr_ratio > 0.0 else None,
                )
                ai_led_snapshot = ai_led_confirmation_snapshot(
                    spot_dir=spot_snapshot.direction,
                    spot_confidence=float(confidence_snapshot.score),
                    ai_dir=ai_spot_direction,
                    ai_probability=float(ai_spot_probability_up),
                    directional_agreement=float(ai_spot_agreement),
                    consensus_agreement=float(ai_spot_consensus),
                    adx_val=float(adx_val) if pd.notna(adx_val) else float("nan"),
                    location_quality=float(execution_snapshot.location_quality),
                    rr_ratio=setup_rr_ratio if np.isfinite(setup_rr_ratio) and setup_rr_ratio > 0.0 else None,
                    ai_status=ai_spot_status,
                )

                # Setup confirm mirrors market/spot decision policy.
                action_raw, action_reason_code = spot_action_decision_with_reason(
                    spot_snapshot.direction,
                    float(confidence_snapshot.score),
                    signal_dir,
                    ai_spot_snapshot.direction,
                    float(ai_spot_agreement),
                    float(adx_val),
                    trend_led_snapshot=trend_led_snapshot,
                    ai_led_snapshot=ai_led_snapshot,
                )

                # Spot-style setup snapshot (selected coin/timeframe).
                price_change = None
                if len(df_eval) >= 2:
                    p0 = float(df_eval["close"].iloc[-2])
                    p1 = float(df_eval["close"].iloc[-1])
                    if p0 > 0:
                        price_change = ((p1 / p0) - 1.0) * 100.0
                delta_note = "Closed-candle move on selected timeframe."
                delta_display = format_delta(price_change) if price_change is not None else ""
                delta_c = (
                    POSITIVE if str(delta_display).strip().startswith("▲")
                    else (NEGATIVE if str(delta_display).strip().startswith("▼") else WARNING)
                )

                sig_color = POSITIVE if spot_snapshot.direction == "UPSIDE" else (NEGATIVE if spot_snapshot.direction == "DOWNSIDE" else WARNING)
                ai_color = POSITIVE if ai_spot_direction == "UPSIDE" else (NEGATIVE if ai_spot_direction == "DOWNSIDE" else WARNING)
                confidence_display = _confidence_display(float(confidence_snapshot.score))
                conf_bucket = confidence_bucket(float(confidence_snapshot.score))
                conf_color = POSITIVE if conf_bucket == "HIGH" else (WARNING if conf_bucket == "MEDIUM" else NEGATIVE)
                ai_conf_bucket = str(ai_confidence_snapshot.label or "LOW").upper()
                ai_conf_color = POSITIVE if ai_conf_bucket == "HIGH" else (WARNING if ai_conf_bucket == "MEDIUM" else NEGATIVE)
                setup_confirm = _setup_confirm_display(action_raw)
                setup_reason = action_reason_text(action_reason_code)
                action_class = normalize_action_class(action_raw)
                if action_class.startswith("ENTER_"):
                    setup_color = POSITIVE
                elif action_class == "WATCH":
                    setup_color = WARNING
                else:
                    setup_color = NEGATIVE

                direction_note = (
                    f"Spot bias (1D + 4H): {_spot_bias_label(spot_snapshot.direction)} | "
                    f"Combined score {float(spot_snapshot.score):.1f} | {str(spot_snapshot.note or '').strip()} | "
                    f"{_spot_tf_note(spot_snapshot.one_day)} | "
                    f"{_spot_tf_note(spot_snapshot.four_hour)} | "
                    f"Tactical ({tf}): {direction_label(signal_dir)} | Bias {float(bias_score):.1f}"
                )
                confidence_note = (
                    f"Spot confidence: {float(confidence_snapshot.score):.1f}% ({confidence_snapshot.label.title()}) | "
                    f"Timeframe alignment {float(spot_snapshot.timeframe_alignment):.0f} | "
                    f"Structure quality {float(spot_snapshot.structure_quality):.0f} | "
                    f"Trend quality {float(spot_snapshot.trend_quality):.0f} | "
                    f"Regime quality {float(spot_snapshot.regime_quality):.0f} | "
                    f"Location quality {float(spot_snapshot.location_quality):.0f}"
                )
                ai_note = _ai_spot_note(ai_spot_snapshot)
                ai_confidence_note = _ai_confidence_note(ai_spot_snapshot, float(ai_confidence_snapshot.score))
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
                    f"<div class='spot-summary-value' style='color:{delta_c};'>{delta_display or '—'}</div></div>"
                    f"<div class='spot-summary-item' title='{setup_reason}'>"
                    f"<div class='spot-summary-label'>Setup Confirm</div>"
                    f"<div class='spot-summary-value' style='color:{setup_color};'>{setup_confirm}</div></div>"
                    f"<div class='spot-summary-item' title='{html.escape(direction_note, quote=True)}'>"
                    f"<div class='spot-summary-label'>Direction</div>"
                    f"<div class='spot-summary-value' style='color:{sig_color};'>{direction_label(spot_snapshot.direction)}</div></div>"
                    f"<div class='spot-summary-item' title='{html.escape(confidence_note, quote=True)}'>"
                    f"<div class='spot-summary-label'>Confidence</div>"
                    f"<div class='spot-summary-value' style='color:{conf_color};'>{confidence_display}</div></div>"
                    f"<div class='spot-summary-item' title='{html.escape(ai_note, quote=True)}'>"
                    f"<div class='spot-summary-label'>AI Ensemble</div>"
                    f"<div class='spot-summary-value' style='color:{ai_color};'>{direction_label(ai_spot_snapshot.direction)} ({ai_spot_votes}/3){' *' if ai_spot_snapshot.degraded_data else ''}</div></div>"
                    f"<div class='spot-summary-item' title='{html.escape(ai_confidence_note, quote=True)}'>"
                    f"<div class='spot-summary-label'>AI Confidence</div>"
                    f"<div class='spot-summary-value' style='color:{ai_conf_color};'>{_ai_confidence_display(ai_spot_snapshot, float(ai_confidence_snapshot.score))}</div></div>"
                    f"</div></div>",
                    unsafe_allow_html=True,
                )

                supertrend_txt = _clean_indicator_text(supertrend_trend)
                ichimoku_txt = _clean_indicator_text(ichimoku_trend)
                vwap_txt = _clean_indicator_text(vwap_label)
                adx_txt = _clean_indicator_text(_adx_bucket_only(adx_val))
                bollinger_txt = _clean_indicator_text(bollinger_bias)
                stochrsi_txt = _clean_indicator_text(format_stochrsi(stochrsi_k_val, timeframe=tf))
                psar_txt = _clean_indicator_text(psar_trend)
                will_txt = _clean_indicator_text(williams_label)
                cci_txt = _clean_indicator_text(cci_label)
                volume_txt = ""
                if volume_spike:
                    try:
                        o = float(df_eval["open"].iloc[-1])
                        c = float(df_eval["close"].iloc[-1])
                        if pd.notna(o) and pd.notna(c) and c > o:
                            volume_txt = "▲ Up Spike"
                        elif pd.notna(o) and pd.notna(c) and c < o:
                            volume_txt = "▼ Down Spike"
                        else:
                            volume_txt = "→ Spike"
                    except Exception:
                        volume_txt = "→ Spike"
                volatility_txt = _clean_indicator_text(str(atr_comment).replace("▲", "").replace("▼", "").replace("–", ""))
                pattern_txt = _clean_indicator_text(candle_pattern.split(" (")[0] if candle_pattern else "")

                trend_cells = "".join(
                    [
                        _indicator_cell("SuperTrend", supertrend_txt, "ATR-based trend line direction."),
                        _indicator_cell("Ichimoku", ichimoku_txt, "Cloud trend context."),
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
                        _indicator_cell("Volume", volume_txt, "Abnormal volume event."),
                    ]
                )
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

                # Risk alert for position
                pos_dir = direction_key(direction)
                if pos_dir == direction_key(spot_snapshot.direction) and float(confidence_snapshot.score) < 50:
                    st.markdown(
                        f"<div style='background:#2D0A0A; border-left:4px solid {NEGATIVE}; "
                        f"padding:6px 10px; border-radius:4px; margin:4px 0; font-size:0.82rem;'>"
                        f"<span style='color:{NEGATIVE}; font-weight:600;'>Low Confidence</span>"
                        f"<span style='color:{TEXT_MUTED};'> — Position direction matches the spot bias but confidence "
                        f"is only {float(confidence_snapshot.score):.0f}%. Consider tighter risk control.</span></div>",
                        unsafe_allow_html=True,
                    )
                elif pos_dir != direction_key(spot_snapshot.direction) and direction_key(spot_snapshot.direction) != "NEUTRAL":
                    st.markdown(
                        f"<div style='background:#2D0A0A; border-left:4px solid {NEGATIVE}; "
                        f"padding:6px 10px; border-radius:4px; margin:4px 0; font-size:0.82rem;'>"
                        f"<span style='color:{NEGATIVE}; font-weight:600;'>Direction Conflict</span>"
                        f"<span style='color:{TEXT_MUTED};'> — Your {direction_label(direction)} position conflicts with "
                        f"the current spot bias {direction_label(spot_snapshot.direction)}. Review position validity.</span></div>",
                        unsafe_allow_html=True,
                    )

                recent_sr = df_eval.tail(_sr_lookback(tf))
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
                    signal_direction=signal_dir_legacy,
                    confidence=float(execution_confidence.score),
                    conviction_label=conviction_lbl,
                    liq_distance_pct=liq_dist_pct,
                    invalidated=invalidated,
                    levered_pnl_pct=float(pnl_percent),
                )
                health_score = int(health_pack["score"])
                health_label = str(health_pack["label"])
                health_action = str(health_pack["action"])
                health_notes = [str(x) for x in list(health_pack.get("notes", []))]
                report_rows.append(
                    {
                        "Timestamp (UTC)": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
                        "Coin": coin,
                        "Timeframe": tf,
                        "Position Side": direction_label(direction),
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
                        "Spot Direction": direction_label(spot_snapshot.direction),
                        "Tactical Direction": signal_clean,
                        "Confidence (%)": round(float(confidence_snapshot.score), 2),
                        "Confidence (Selected TF) (%)": round(float(execution_confidence.score), 2),
                        "AI Direction": direction_label(ai_spot_snapshot.direction),
                        "AI Confidence (%)": round(float(ai_confidence_snapshot.score), 2),
                        "Technical Invalidation": round(float(invalidation), 6),
                        "Invalidation State": inv_state,
                        "Health Label": health_label,
                        "Health Score": int(health_score),
                        "Health Action": health_action,
                    }
                )
                reason_map = {
                    "signal conflict": "signal conflict",
                    "no clear technical edge": "no clear edge",
                    "low confidence": "low confidence",
                    "medium confidence": "medium confidence",
                    "AI conflict": "AI conflict",
                    "low conviction": "weak conviction",
                    "liquidation too close": "liquidation too close",
                    "liquidation moderately close": "liquidation moderately close",
                    "hard invalidation broken": "invalidation broken",
                    "deep drawdown": "deep drawdown",
                }
                why_items = health_notes[:3] if health_notes else ["no major risk flag"]
                why_text = ", ".join(reason_map.get(n, n) for n in why_items)
                health_meaning_map = {
                    "HOLD": "Meaning: structure is acceptable; hold with discipline.",
                    "REDUCE": "Meaning: edge is weakened; reduce size and avoid adding.",
                    "EXIT": "Meaning: risk breach is high; close or hedge immediately.",
                }
                health_meaning = health_meaning_map.get(health_label, "")
                st.markdown(
                    f"<div class='panel-box' style='padding:10px 12px;'>"
                    f"<div style='font-size:0.92rem;'>"
                    f"<span style='color:{TEXT_MUTED};'>{tf} Position Risk Status:</span> "
                    f"<b style='color:{ACCENT};'>{health_label} ({health_score}/100)</b>"
                    f"</div>"
                    f"<div style='color:{TEXT_MUTED}; font-size:0.86rem; margin-top:4px;'>"
                    f"Why: {why_text}"
                    f"</div>"
                    f"<div style='color:{TEXT_MUTED}; font-size:0.86rem; margin-top:2px;'>"
                    f"{health_meaning}"
                    f"</div>"
                    f"<div style='color:{TEXT_MUTED}; font-size:0.86rem; margin-top:2px;'>"
                    f"Next: {health_action} Keep risk line at "
                    f"<b style='color:{WARNING};'>{invalidation:,.4f}</b>."
                    f"</div>"
                    f"</div>",
                    unsafe_allow_html=True,
                )

                # === Scalping Setup ===
                if df_scalp is not None and len(df_scalp) > 30:
                    gate_min_rr, gate_min_adx, gate_min_confidence = scalp_gate_thresholds(tf)
                    scalp_ok, scalp_reason = scalp_quality_gate(
                        scalp_direction=scalp_direction,
                        signal_direction=signal_dir,
                        rr_ratio=rr_ratio,
                        adx_val=adx_val,
                        confidence=float(execution_confidence.score),
                        conviction_label=conviction_lbl,
                        entry=entry_s,
                        stop=stop_s,
                        target=target_s,
                        min_rr=gate_min_rr,
                        min_adx=gate_min_adx,
                        min_confidence=gate_min_confidence,
                    )
                
                    # === Display Scalping Result ===
                    if scalp_ok and scalp_direction:
                        color = POSITIVE if scalp_direction == "LONG" else NEGATIVE
                        st.markdown(
                            f"""
                            <div class='panel-box' style='border-left:4px solid {color};'>
                              <div style='display:flex; justify-content:space-between; gap:8px; flex-wrap:wrap;'>
                                <span style='color:{color}; font-weight:800;'>Scalping {direction_label(scalp_direction)}</span>
                                <span style='color:{color}; font-weight:700;'>R:R {rr_ratio:.2f}</span>
                              </div>
                              <div style='color:{TEXT_MUTED}; font-size:0.88rem; margin-top:6px; line-height:1.65;'>
                                Your Entry <b style='color:{ACCENT};'>${entry_price:,.4f}</b> |
                                Model Entry <b style='color:{ACCENT};'>${entry_s:,.4f}</b><br>
                                Stop <b style='color:{NEGATIVE};'>${stop_s:,.4f}</b> |
                                Target <b style='color:{POSITIVE};'>${target_s:,.4f}</b><br>
                                {'Good setup quality (R:R gate passed).' if rr_ratio >= gate_min_rr else f'R:R is below gate ({gate_min_rr:.2f}).'}
                              </div>
                            </div>
                            """,
                            unsafe_allow_html=True
                        )
                    else:
                        reason_map = {
                            "NO_SCALP_DIRECTION": "No directional scalp setup on current structure.",
                            "SIGNAL_DIRECTION_NEUTRAL": "Signal direction is neutral; scalp setup is blocked.",
                            "DIRECTION_MISMATCH": "Scalp side does not match direction; setup filtered.",
                            "CONFLICT": "Technical vs AI alignment is in conflict.",
                            "RR_TOO_LOW": f"R:R below required threshold ({gate_min_rr:.2f}).",
                            "ADX_TOO_LOW": f"ADX below required trend threshold ({gate_min_adx:.0f}).",
                            "CONFIDENCE_TOO_LOW": f"Confidence below required threshold ({gate_min_confidence:.0f}).",
                            "INVALID_LEVELS": "Invalid Entry/Stop/Target levels.",
                        }
                        msg = breakout_note or reason_map.get(scalp_reason, "No valid scalping setup with current filters.")
                        if msg in {"Invalid plan levels", "Invalid ATR/price"}:
                            msg = "No valid plan on current candle structure."
                        st.markdown(
                            f"<div class='panel-box' style='border-left:4px solid {WARNING};'>"
                            f"<b style='color:{WARNING};'>Scalping Opportunity: Not Available</b><br>"
                            f"<span style='color:{TEXT_MUTED}; font-size:0.86rem;'>Reason: {msg}</span>"
                            f"</div>",
                            unsafe_allow_html=True,
                        )
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
            st,
            f"position_chart::{coin}::{largest_tf}::100",
            df_candle_live,
            max_age_sec=900,
            current_sig=(coin, largest_tf, 100),
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
            st.markdown(f"<h4 style='color:{ACCENT};'>Candlestick Chart – {largest_tf}</h4>", unsafe_allow_html=True)
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
