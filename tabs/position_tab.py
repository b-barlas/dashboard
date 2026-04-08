from ui.ctx import get_ctx

from datetime import datetime, timezone
import io

import numpy as np
import pandas as pd
import plotly.graph_objs as go
import ta
from core.session_utils import session_bucket_for_timestamp
from core.ai_spot_bias import (
    ai_spot_bias_consensus_agreement,
    ai_spot_bias_directional_agreement,
    ai_spot_bias_display_votes,
    ai_spot_bias_probability_up,
    ai_spot_bias_status,
    build_ai_spot_bias_snapshot,
)
from core.confidence import (
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
from core.position_management import build_position_management_snapshot
from core.signal_tracker import build_actual_exit_quality_profile, build_actual_trade_hold_profile
from core.spot_direction import build_spot_direction_snapshot
from ui.primitives import render_help_details, render_page_header
from ui.signal_panels import build_indicator_groups_html, build_learned_edge_banner_html, build_setup_snapshot_html
from ui.signal_formatters import (
    adx_bucket_only as _adx_bucket_only,
    ai_confidence_display as _ai_confidence_display,
    ai_confidence_note as _ai_confidence_note,
    ai_spot_note as _ai_spot_note,
    execution_read_note as _execution_read_note,
    context_fit_snapshot as _context_fit_snapshot,
    setup_confirm_display as _setup_confirm_display,
    spot_bias_label as _spot_bias_label,
    spot_confidence_display as _confidence_display,
    trade_gate_display_label as _trade_gate_display_label,
)
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


def _spot_tf_note(snapshot) -> str:
    return (
        f"{str(snapshot.timeframe).upper()}: {_spot_bias_label(snapshot.direction)} | "
        f"Score {float(snapshot.score):.1f} | "
        f"Structure {snapshot.structure_label} ({float(snapshot.structure_score):.0f}) | "
        f"Trend {float(snapshot.trend_score):.0f} | "
        f"Regime {snapshot.regime_label} ({float(snapshot.regime_quality):.0f}) | "
        f"Location {float(snapshot.location_quality):.0f}"
    )


def _learned_edge_tone(adaptive_label: str, execution_fit_label: str, archive_guardrail_label: str = "") -> str:
    adaptive_key = str(adaptive_label or "").strip().upper()
    execution_key = str(execution_fit_label or "").strip().upper()
    guardrail_key = str(archive_guardrail_label or "").strip().upper()
    if "GUARDRAIL" in guardrail_key or "CAUTION" in guardrail_key:
        return "warning"
    if "WEAK" in adaptive_key or "FRAGILE" in execution_key:
        return "negative"
    if "FAVORED" in adaptive_key and "PROVEN" in execution_key:
        return "positive"
    if "FAVORED" in adaptive_key:
        return "info"
    if "MIXED" in execution_key:
        return "warning"
    return "neutral"


def _tone_color(tone: str, *, accent: str, positive: str, negative: str, warning: str) -> str:
    key = str(tone or "").strip().lower()
    if key == "positive":
        return positive
    if key == "negative":
        return negative
    if key == "warning":
        return warning
    return accent


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
    get_signal_tracker_db_path = get_ctx(ctx, "get_signal_tracker_db_path")
    init_signal_tracker_db = get_ctx(ctx, "init_signal_tracker_db")
    fetch_signal_events_df = get_ctx(ctx, "fetch_signal_events_df")
    build_adaptive_context_model = get_ctx(ctx, "build_adaptive_context_model")
    build_live_signal_adaptive_snapshot = get_ctx(ctx, "build_live_signal_adaptive_snapshot")
    build_recent_market_context_snapshot = get_ctx(ctx, "build_recent_market_context_snapshot")
    build_recent_symbol_market_signal_snapshot = get_ctx(ctx, "build_recent_symbol_market_signal_snapshot")
    classify_symbol_sector = get_ctx(ctx, "classify_symbol_sector")
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
        signal_tracker_db_path = init_signal_tracker_db(get_signal_tracker_db_path())
        adaptive_history_df = fetch_signal_events_df(
            limit=2000,
            status="RESOLVED",
            source="Market",
            db_path=signal_tracker_db_path,
        )
        recent_market_events_df = fetch_signal_events_df(
            limit=240,
            source="Market",
            db_path=signal_tracker_db_path,
        )
        recent_market_context = build_recent_market_context_snapshot(recent_market_events_df)
        adaptive_model = build_adaptive_context_model(adaptive_history_df)
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
                watch_setup_color = "#7DD3FC"
                if action_class.startswith("ENTER_"):
                    setup_color = POSITIVE
                elif action_class == "PROBE":
                    setup_color = WARNING
                elif action_class == "WATCH":
                    setup_color = watch_setup_color
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
                current_session_bucket = session_bucket_for_timestamp()
                current_sector_tag = str(classify_symbol_sector(coin) or "").strip()
                recent_symbol_market_signal = build_recent_symbol_market_signal_snapshot(
                    recent_market_events_df,
                    symbol=coin,
                    timeframe=tf,
                )
                adaptive_snapshot = build_live_signal_adaptive_snapshot(
                    adaptive_model,
                    signal={
                        "Setup Confirm": str(action_raw or ""),
                        "Lead": str(recent_symbol_market_signal.get("Lead") or "No LEAD"),
                        "AI Alignment": "Aligned" if direction_key(spot_snapshot.direction) == ai_spot_direction else "Not aligned",
                        "Market Lead": str(recent_market_context.get("Market Lead") or "No Clear Lead"),
                        "Market Regime": str(recent_market_context.get("Market Regime") or "Unknown"),
                        "Playbook": str(recent_market_context.get("Playbook") or "Unknown"),
                        "Trade Gate": str(recent_market_context.get("Trade Gate") or "Unknown"),
                        "Sector Rotation": str(recent_market_context.get("Sector Rotation") or "Unknown"),
                        "Catalyst State": str(recent_market_context.get("Catalyst State") or "Unknown"),
                        "Catalyst Window": str(recent_market_context.get("Catalyst Window") or "Unknown"),
                        "Catalyst Scope": str(recent_market_context.get("Catalyst Scope") or "Unknown"),
                        "Catalyst Targeting": str(recent_market_context.get("Catalyst Targeting") or "Unknown"),
                        "Flow Proxy": str(recent_market_context.get("Flow Proxy") or "Unknown"),
                        "Session": current_session_bucket,
                        "Timeframe": str(tf or "Unknown"),
                    },
                )
                market_context_note = str(recent_market_context.get("Context Note") or "").strip()
                scanner_signal_note = str(recent_symbol_market_signal.get("Signal Note") or "").strip()
                context_fit = _context_fit_snapshot(
                    adaptive_snapshot,
                    market_context=recent_market_context,
                    recent_symbol_market_signal=recent_symbol_market_signal,
                )
                hold_profile = build_actual_trade_hold_profile(
                    adaptive_history_df,
                    symbol=coin,
                    timeframe=tf,
                    direction=direction,
                    sector_tag=current_sector_tag,
                    playbook=str(recent_market_context.get("Playbook") or ""),
                    session_bucket=current_session_bucket,
                    trade_gate=str(recent_market_context.get("Trade Gate") or ""),
                    catalyst_window=str(recent_market_context.get("Catalyst Window") or ""),
                )
                exit_quality = build_actual_exit_quality_profile(
                    adaptive_history_df,
                    symbol=coin,
                    timeframe=tf,
                    direction=direction,
                    sector_tag=current_sector_tag,
                    playbook=str(recent_market_context.get("Playbook") or ""),
                    session_bucket=current_session_bucket,
                    trade_gate=str(recent_market_context.get("Trade Gate") or ""),
                    catalyst_window=str(recent_market_context.get("Catalyst Window") or ""),
                )
                confidence_note = (
                    f"{confidence_note} | Historical read: {adaptive_snapshot.note} | "
                    f"Execution fit: {adaptive_snapshot.execution_fit_note} | "
                    f"Session fit: {adaptive_snapshot.session_fit_note}"
                )
                setup_snapshot_html = build_setup_snapshot_html(
                    title="Setup Snapshot",
                    text_muted=TEXT_MUTED,
                    items=[
                        {"label": "Δ (%)", "value": delta_display or "—", "color": delta_c, "title": delta_note},
                        {
                            "label": "Setup Confirm",
                            "value": setup_confirm,
                            "color": setup_color,
                            "title": f"{setup_reason} | Execution fit: {adaptive_snapshot.execution_fit_note}",
                        },
                        {
                            "label": "Direction",
                            "value": direction_label(spot_snapshot.direction),
                            "color": sig_color,
                            "title": direction_note,
                        },
                        {
                            "label": "Confidence",
                            "value": confidence_display,
                            "color": conf_color,
                            "title": confidence_note,
                        },
                        {
                            "label": "AI Ensemble",
                            "value": f"{direction_label(ai_spot_snapshot.direction)} ({ai_spot_votes}/3){' *' if ai_spot_snapshot.degraded_data else ''}",
                            "color": ai_color,
                            "title": ai_note,
                        },
                        {
                            "label": "AI Confidence",
                            "value": _ai_confidence_display(ai_spot_snapshot, float(ai_confidence_snapshot.score)),
                            "color": ai_conf_color,
                            "title": ai_confidence_note,
                        },
                    ],
                )
                st.markdown(setup_snapshot_html, unsafe_allow_html=True)
                st.markdown(
                    build_learned_edge_banner_html(
                        title="Execution Read",
                        label=(
                            f"{_trade_gate_display_label(context_fit['label'])} • "
                            f"{adaptive_snapshot.execution_fit_label}"
                        ),
                        note=_execution_read_note(
                            adaptive_snapshot,
                            context_fit=context_fit,
                            market_context_note=market_context_note,
                            scanner_signal_note=scanner_signal_note,
                        ),
                        tone=_learned_edge_tone(
                            adaptive_snapshot.label,
                            adaptive_snapshot.execution_fit_label,
                            adaptive_snapshot.archive_guardrail_label,
                        ),
                        text_muted=TEXT_MUTED,
                        positive=POSITIVE,
                        negative=NEGATIVE,
                        warning=WARNING,
                        accent=ACCENT,
                    ),
                    unsafe_allow_html=True,
                )

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
                indicator_groups_html = build_indicator_groups_html(
                    title="Technical Regime Breakdown (closed-candle context)",
                    accent=ACCENT,
                    text_muted=TEXT_MUTED,
                    positive=POSITIVE,
                    negative=NEGATIVE,
                    warning=WARNING,
                    groups=[
                        (
                            "Trend Structure",
                            [
                                {"name": "SuperTrend", "value": supertrend_trend, "tooltip": "ATR-based trend line direction."},
                                {"name": "Ichimoku", "value": ichimoku_trend, "tooltip": "Cloud trend context."},
                                {"name": "VWAP", "value": vwap_label, "tooltip": "Price relative to volume-weighted average price."},
                                {"name": "ADX", "value": _adx_bucket_only(adx_val), "tooltip": "Trend strength (not direction)."},
                                {"name": "PSAR", "value": psar_trend, "tooltip": "Parabolic SAR trend-following state."},
                            ],
                        ),
                        (
                            "Momentum Signals",
                            [
                                {
                                    "name": "StochRSI",
                                    "value": format_stochrsi(stochrsi_k_val, timeframe=tf),
                                    "tooltip": "Momentum pressure zone.",
                                },
                                {"name": "Williams %R", "value": williams_label, "tooltip": "Range-position momentum signal."},
                                {"name": "CCI", "value": cci_label, "tooltip": "Mean-reversion momentum signal."},
                                {
                                    "name": "Pattern",
                                    "value": candle_pattern.split(" (")[0] if candle_pattern else "",
                                    "tooltip": "Latest candle pattern direction.",
                                },
                            ],
                        ),
                        (
                            "Volatility & Volume",
                            [
                                {
                                    "name": "Bollinger",
                                    "value": bollinger_bias,
                                    "tooltip": "Band location (extension / pullback context).",
                                },
                                {
                                    "name": "Volatility",
                                    "value": str(atr_comment).replace("▲", "").replace("▼", "").replace("–", ""),
                                    "tooltip": "ATR/band-width regime.",
                                },
                                {"name": "Volume", "value": volume_txt, "tooltip": "Abnormal volume event."},
                            ],
                        ),
                    ],
                )
                if indicator_groups_html:
                    st.markdown(indicator_groups_html, unsafe_allow_html=True)

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
                invalidation_distance_pct = (
                    abs(float(current_price) - float(invalidation)) / float(current_price) * 100.0
                    if float(current_price) > 0
                    else None
                )

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
                management_snapshot = build_position_management_snapshot(
                    direction=direction,
                    health_label=health_label,
                    health_score=float(health_score),
                    health_notes=health_notes,
                    levered_pnl_pct=float(pnl_percent),
                    liq_distance_pct=float(liq_dist_pct) if liq_dist_pct is not None else None,
                    leverage=float(leverage),
                    invalidated=invalidated,
                    invalidation_distance_pct=invalidation_distance_pct,
                    spot_direction=str(spot_snapshot.direction or ""),
                    tactical_direction=str(signal_dir or ""),
                    ai_direction=str(ai_spot_snapshot.direction or ""),
                    selected_confidence=float(execution_confidence.score),
                    context_fit_label=str(context_fit["label"] or ""),
                    context_fit_aggression=str(context_fit["aggression"] or ""),
                    adaptive_label=str(adaptive_snapshot.label or ""),
                    execution_fit_label=str(adaptive_snapshot.execution_fit_label or ""),
                    session_fit_label=str(adaptive_snapshot.session_fit_label or ""),
                    archive_guardrail_label=str(adaptive_snapshot.archive_guardrail_label or ""),
                    catalyst_window=str(recent_market_context.get("Catalyst Window") or ""),
                    trade_gate=str(recent_market_context.get("Trade Gate") or ""),
                    playbook=str(recent_market_context.get("Playbook") or ""),
                    flow_proxy=str(recent_market_context.get("Flow Proxy") or ""),
                    volatility_regime=str(atr_comment or ""),
                    short_term_move_pct=float(price_change) if price_change is not None else None,
                    volume_spike_label=str(volume_txt or ""),
                    hold_profile_label=str(hold_profile.get("label") or ""),
                    hold_profile_note=str(hold_profile.get("note") or ""),
                    exit_quality_label=str(exit_quality.get("label") or ""),
                    exit_quality_note=str(exit_quality.get("note") or ""),
                )
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
                        "Management Label": management_snapshot.label,
                        "Management Score": int(management_snapshot.score),
                        "Management Size": management_snapshot.size_guidance,
                        "Management Adds": management_snapshot.adds_guidance,
                        "Management Risk": management_snapshot.risk_guidance,
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
                management_color = _tone_color(
                    management_snapshot.tone,
                    accent=ACCENT,
                    positive=POSITIVE,
                    negative=NEGATIVE,
                    warning=WARNING,
                )
                st.markdown(
                    f"<div class='panel-box' style='padding:10px 12px; border-left:4px solid {management_color};'>"
                    f"<div style='display:flex; justify-content:space-between; gap:8px; flex-wrap:wrap; font-size:0.92rem;'>"
                    f"<span style='color:{TEXT_MUTED};'>{tf} Position Management:</span> "
                    f"<b style='color:{management_color};'>{management_snapshot.label} ({management_snapshot.score}/100)</b>"
                    f"</div>"
                    f"<div style='color:{TEXT_MUTED}; font-size:0.86rem; margin-top:4px;'>"
                    f"Execution stance: <b>{context_fit['label']}</b> — {context_fit['aggression']}"
                    f"</div>"
                    f"<div style='color:{TEXT_MUTED}; font-size:0.86rem; margin-top:4px;'>"
                    f"Hold profile: <b>{hold_profile.get('label') or 'Archive Building'}</b>"
                    f" • Exit quality: <b>{exit_quality.get('label') or 'Archive Building'}</b>"
                    f"</div>"
                    f"<div style='color:{TEXT_MUTED}; font-size:0.86rem; margin-top:4px;'>"
                    f"Size: <b>{management_snapshot.size_guidance}</b>"
                    f"</div>"
                    f"<div style='color:{TEXT_MUTED}; font-size:0.86rem; margin-top:4px;'>"
                    f"Adds: <b>{management_snapshot.adds_guidance}</b>"
                    f"</div>"
                    f"<div style='color:{TEXT_MUTED}; font-size:0.86rem; margin-top:4px;'>"
                    f"Why: {management_snapshot.note} Base health: <b>{health_label}</b> ({health_score}/100); {why_text}."
                    f"</div>"
                    f"<div style='color:{TEXT_MUTED}; font-size:0.86rem; margin-top:2px;'>"
                    f"Next: {management_snapshot.risk_guidance} Hard risk line stays at "
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
