from ui.ctx import get_ctx

import pandas as pd
import plotly.graph_objs as go
import ta
from core.adaptive_weighting import (
    build_ai_confidence_calibration_model,
    build_ai_confidence_calibration_snapshot,
    build_confidence_calibration_model,
    build_setup_calibration_model,
    build_setup_calibration_snapshot,
)
from core.confidence import build_ai_confidence_snapshot
from core.trading_copy import copy_text, playbook_key, trade_gate_key
from core.session_utils import session_bucket_for_timestamp
from core.confidence import confidence_bucket
from core.market_decision import apply_setup_archive_calibration, action_reason_text, normalize_action_class
from core.signal_tracker import prefer_current_decision_version_slice
from core.spot_execution_pipeline import build_spot_execution_pipeline, direction_fetch_symbol
from ui.primitives import render_help_details, render_kpi_grid, render_page_header
from ui.signal_panels import (
    build_indicator_groups_html,
    build_learned_edge_banner_html,
    build_setup_snapshot_html,
)
from ui.signal_formatters import (
    adx_bucket_only as _adx_bucket_only,
    ai_confidence_display as _spot_ai_confidence_display,
    ai_confidence_note as _spot_ai_confidence_note,
    ai_spot_note as _spot_ai_bias_note,
    compact_note_parts as _compact_note_parts,
    context_fit_snapshot as _context_fit_snapshot,
    setup_confirm_display as _setup_confirm_display,
    spot_bias_label as _spot_bias_label,
    spot_confidence_display as _spot_confidence_display,
    trade_gate_display_label as _trade_gate_display_label,
)
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


def _spot_direction_fetch_symbol(symbol: str, actual_symbol: str, source_provider: str) -> str:
    return direction_fetch_symbol(symbol, actual_symbol, source_provider)


def _spot_tf_note(snapshot) -> str:
    return (
        f"{str(snapshot.timeframe).upper()}: {_spot_bias_label(snapshot.direction)} | "
        f"Score {float(snapshot.score):.1f} | "
        f"Structure {snapshot.structure_label} ({float(snapshot.structure_score):.0f}) | "
        f"Trend {float(snapshot.trend_score):.0f} | "
        f"Regime {snapshot.regime_label} ({float(snapshot.regime_quality):.0f}) | "
        f"Location {float(snapshot.location_quality):.0f}"
    )


def _spot_lead_snapshot(snapshot):
    return getattr(snapshot, "lead_snapshot", getattr(snapshot, "one_day", None))


def _spot_confirm_snapshot(snapshot):
    return getattr(snapshot, "confirm_snapshot", getattr(snapshot, "four_hour", None))


def _spot_anchor_pair_label(snapshot) -> str:
    label = str(getattr(snapshot, "anchor_pair_label", "") or "").strip()
    if label:
        return label
    lead = _spot_lead_snapshot(snapshot)
    confirm = _spot_confirm_snapshot(snapshot)
    if lead is None or confirm is None:
        return "HTF"
    return f"{str(lead.timeframe).upper()} + {str(confirm.timeframe).upper()}"


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


def _spot_archive_status_label(adaptive_snapshot) -> str:
    archive_label = str(getattr(adaptive_snapshot, "archive_guardrail_label", "") or "").strip()
    adaptive_label = str(getattr(adaptive_snapshot, "label", "") or "").strip()
    session_label = str(getattr(adaptive_snapshot, "session_fit_label", "") or "").strip()

    if archive_label == "Archive Guardrail":
        return copy_text("spot.archive.status.guardrail")
    if archive_label == "Archive Caution":
        return copy_text("spot.archive.status.caution")
    if adaptive_label == "Historically Favored":
        return copy_text("spot.archive.status.supportive")
    if adaptive_label == "Historically Weak":
        return copy_text("spot.archive.status.fragile")
    if session_label == "Session Supportive":
        return copy_text("spot.archive.status.session_supportive")
    if session_label == "Session Fragile":
        return copy_text("spot.archive.status.session_fragile")
    return copy_text("spot.archive.status.mixed")


def _spot_archive_banner_note(
    adaptive_snapshot,
    *,
    context_fit: dict[str, str],
    market_context: dict[str, str] | None = None,
    market_context_note: str = "",
    scanner_signal_note: str = "",
) -> str:
    market_context = dict(market_context or {})
    stance = _trade_gate_display_label(str((context_fit or {}).get("label") or ""))
    aggression = str((context_fit or {}).get("aggression") or "").strip()
    stance_summary = f"{stance}: {aggression}." if stance and aggression else ""

    archive_note = str(getattr(adaptive_snapshot, "archive_guardrail_note", "") or "").strip()
    session_label = str(getattr(adaptive_snapshot, "session_fit_label", "") or "").strip()
    session_note = str(getattr(adaptive_snapshot, "session_fit_note", "") or "").strip()
    adaptive_label = str(getattr(adaptive_snapshot, "label", "") or "").strip()
    history_note = str(getattr(adaptive_snapshot, "note", "") or "").strip()
    trade_gate = str(market_context.get("Trade Gate") or "").strip()
    trade_gate_value_key = trade_gate_key(market_context.get("Trade Gate Key") or trade_gate)
    playbook = str(market_context.get("Playbook") or "").strip()
    catalyst_window = str(market_context.get("Catalyst Window") or "").strip()
    flow_proxy = str(market_context.get("Flow Proxy") or "").strip()

    # In defensive/no-trade reads, lead with the stance and avoid a misleading
    # "history favored" opener unless the archive verdict is actually actionable.
    if trade_gate_key((context_fit or {}).get("gate_key") or stance) in {"NO_TRADE", "DEFENSIVE_ONLY"} and adaptive_label == "Historically Favored":
        history_note = ""
    if session_label in {"Session Mixed", "Session Unproven"}:
        session_note = ""

    plain_archive_note = ""
    if archive_label := str(getattr(adaptive_snapshot, "archive_guardrail_label", "") or "").strip():
        if archive_label == "Archive Guardrail":
            plain_archive_note = copy_text("spot.archive.history.guardrail")
        elif archive_label == "Archive Caution":
            plain_archive_note = copy_text("spot.archive.history.caution")

    plain_history_note = ""
    if not plain_archive_note:
        if adaptive_label == "Historically Favored":
            plain_history_note = copy_text("spot.archive.history.supportive")
        elif adaptive_label == "Historically Weak":
            plain_history_note = copy_text("spot.archive.history.fragile")
        elif adaptive_label == "Historically Neutral":
            plain_history_note = copy_text("spot.archive.history.neutral")

    plain_session_note = ""
    if session_label == "Session Supportive":
        plain_session_note = copy_text("spot.archive.session.supportive")
    elif session_label == "Session Fragile":
        plain_session_note = copy_text("spot.archive.session.fragile")

    plain_context_note = ""
    context_bits: list[str] = []
    if trade_gate_value_key == "NO_TRADE":
        context_bits.append(copy_text("spot.archive.context.trade_gate.no_trade"))
    elif trade_gate_value_key == "SELECTIVE_ONLY":
        context_bits.append(copy_text("spot.archive.context.trade_gate.selective"))
    elif trade_gate_value_key == "TRADEABLE":
        context_bits.append(copy_text("spot.archive.context.trade_gate.tradeable"))
    if catalyst_window.startswith("Far"):
        context_bits.append(copy_text("spot.archive.context.catalyst.far"))
    elif catalyst_window.startswith("Near"):
        context_bits.append(copy_text("spot.archive.context.catalyst.near"))
    elif catalyst_window.startswith("Blocking"):
        context_bits.append(copy_text("spot.archive.context.catalyst.blocking"))
    if flow_proxy == "Flow Balanced":
        context_bits.append(copy_text("spot.archive.context.flow.balanced"))
    elif flow_proxy in {"Shorts Crowded", "Longs Crowded"}:
        context_bits.append(copy_text("spot.archive.context.flow.crowded"))
    if context_bits:
        plain_context_note = " • ".join(
            [context_bits[0].capitalize() + "."] + [bit.capitalize() + "." for bit in context_bits[1:2]]
        )
    elif playbook_key(playbook) == "WAIT_CONFIRMATION":
        plain_context_note = copy_text("spot.archive.context.playbook.wait")

    return _compact_note_parts(
        [
            stance_summary,
            plain_archive_note or archive_note,
            plain_session_note or session_note,
            plain_history_note or history_note,
            plain_context_note or market_context_note or scanner_signal_note,
        ],
        limit=3,
    )


def render(ctx: dict) -> None:
    """Render the Spot Trading tab."""
    st = get_ctx(ctx, "st")
    ACCENT = get_ctx(ctx, "ACCENT")
    TEXT_MUTED = get_ctx(ctx, "TEXT_MUTED")
    POSITIVE = get_ctx(ctx, "POSITIVE")
    NEGATIVE = get_ctx(ctx, "NEGATIVE")
    WARNING = get_ctx(ctx, "WARNING")
    CARD_BG = get_ctx(ctx, "CARD_BG")
    _normalize_coin_input = get_ctx(ctx, "_normalize_coin_input")
    _validate_coin_symbol = get_ctx(ctx, "_validate_coin_symbol")
    fetch_ohlcv = get_ctx(ctx, "fetch_ohlcv")
    analyse = get_ctx(ctx, "analyse")
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
    get_signal_tracker_db_path = get_ctx(ctx, "get_signal_tracker_db_path")
    init_signal_tracker_db = get_ctx(ctx, "init_signal_tracker_db")
    fetch_signal_events_df = get_ctx(ctx, "fetch_signal_events_df")
    build_adaptive_context_model = get_ctx(ctx, "build_adaptive_context_model")
    build_live_signal_adaptive_snapshot = get_ctx(ctx, "build_live_signal_adaptive_snapshot")
    build_recent_market_context_snapshot = get_ctx(ctx, "build_recent_market_context_snapshot")
    build_recent_symbol_market_signal_snapshot = get_ctx(ctx, "build_recent_symbol_market_signal_snapshot")
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
            "Spot-focused decision workspace for a single coin. "
            "<b>Direction</b> and <b>Confidence</b> use adaptive higher-timeframe anchors "
            "(for example <b>4H + 1H</b> or <b>1D + 4H</b>) to show the main spot bias. "
            "<b>Setup Confirm</b> then checks whether the selected timeframe trend and/or AI confirm that spot bias for execution. "
            "Selected-timeframe technical layers still drive entry/stop/target timing, while the headline direction stays anchored to spot context."
        ),
    )
    render_help_details(
        st,
        summary="How to read quickly",
        body_html=copy_text("spot.help.quick_html"),
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
        signal_tracker_db_path = init_signal_tracker_db(get_signal_tracker_db_path())
        adaptive_history_df = fetch_signal_events_df(
            limit=2000,
            status="RESOLVED",
            source="Market",
            db_path=signal_tracker_db_path,
        )
        adaptive_history_df = prefer_current_decision_version_slice(
            adaptive_history_df,
            source="Market",
        )
        recent_market_events_df = fetch_signal_events_df(
            limit=240,
            source="Market",
            db_path=signal_tracker_db_path,
        )
        confidence_calibration_model = build_confidence_calibration_model(adaptive_history_df)

        actual_symbol = str(df.attrs.get("source_symbol") or "").strip() or coin
        source_provider = str(df.attrs.get("source_provider") or "").strip() or "exchange"
        pipeline = build_spot_execution_pipeline(
            symbol=coin,
            actual_symbol=actual_symbol,
            source_provider=source_provider,
            timeframe=timeframe,
            df_eval=df_eval,
            fetch_ohlcv=fetch_ohlcv,
            analyse_fn=analyse,
            predictor=ml_ensemble_predict,
            conviction_fn=_calc_conviction,
            confidence_calibration_model=confidence_calibration_model,
            scan_focus="Unknown",
        )
        a = pipeline.analysis
        spot_snapshot = pipeline.spot_snapshot
        confidence_snapshot = pipeline.confidence_snapshot
        ai_spot_snapshot = pipeline.ai_spot_snapshot
        ai_confidence_snapshot = pipeline.ai_confidence_snapshot
        signal = pipeline.signal_raw
        volume_spike = bool(getattr(a, "volume_spike", False))
        atr_comment = str(getattr(a, "atr_comment", "") or "")
        candle_pattern = str(getattr(a, "candle_pattern", "") or "")
        bias_score = float(pipeline.bias_score)
        adx_val = float(pipeline.adx_val)
        supertrend_trend = str(pipeline.supertrend_trend)
        ichimoku_trend = str(pipeline.ichimoku_trend)
        stochrsi_k_val = float(pipeline.stochrsi_k_val)
        bollinger_bias = str(pipeline.bollinger_bias)
        vwap_label = str(pipeline.vwap_label)
        psar_trend = str(pipeline.psar_trend)
        williams_label = str(pipeline.williams_label)
        cci_label = str(pipeline.cci_label)

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
        tactical_signal_dir = pipeline.signal_direction
        spot_direction_key = direction_key(spot_snapshot.direction)
        signal_clean = direction_label(spot_snapshot.direction)
        ai_spot_direction_key = pipeline.ai_spot_direction
        ai_spot_votes = int(pipeline.ai_spot_votes)
        ai_spot_note = _spot_ai_bias_note(ai_spot_snapshot)
        action_raw, action_reason_code = pipeline.action_raw, pipeline.action_reason_code

        setup_confirm = _setup_confirm_display(action_raw, action_reason=action_reason_code)
        setup_reason = action_reason_text(action_reason_code)

        sig_c_s = POSITIVE if spot_snapshot.direction == "UPSIDE" else (NEGATIVE if spot_snapshot.direction == "DOWNSIDE" else WARNING)
        ai_c_s = POSITIVE if ai_spot_direction_key == "UPSIDE" else (NEGATIVE if ai_spot_direction_key == "DOWNSIDE" else WARNING)
        confidence_display = _spot_confidence_display(float(confidence_snapshot.score))
        conf_bucket = confidence_bucket(float(confidence_snapshot.score))
        conf_c_s = POSITIVE if conf_bucket == "HIGH" else (WARNING if conf_bucket == "MEDIUM" else NEGATIVE)
        action_class = normalize_action_class(action_raw)
        watch_setup_color = "#7DD3FC"
        if action_class.startswith("ENTER_"):
            setup_c_s = POSITIVE
        elif action_class == "PROBE":
            setup_c_s = WARNING
        elif action_class == "WATCH":
            setup_c_s = watch_setup_color
        else:
            setup_c_s = NEGATIVE
        delta_display = format_delta(price_change) if price_change is not None else ""
        delta_c_s = (
            POSITIVE if str(delta_display).strip().startswith("▲")
            else (NEGATIVE if str(delta_display).strip().startswith("▼") else WARNING)
        )
        direction_note = (
            f"Spot bias ({_spot_anchor_pair_label(spot_snapshot)}): {_spot_bias_label(spot_snapshot.direction)} | "
            f"Combined score {float(spot_snapshot.score):.1f} | {str(spot_snapshot.note or '').strip()} | "
            f"{_spot_tf_note(_spot_lead_snapshot(spot_snapshot))} | "
            f"{_spot_tf_note(_spot_confirm_snapshot(spot_snapshot))} | "
            f"Tactical ({timeframe}): {direction_label(tactical_signal_dir)} | "
            f"Signal {str(signal or '').strip()} | Bias {float(bias_score):.1f}"
        )
        confidence_note = (
            f"Spot confidence: {float(confidence_snapshot.score):.1f}% ({confidence_snapshot.label.title()}) | "
            f"Timeframe alignment {float(spot_snapshot.timeframe_alignment):.0f} | "
            f"Structure quality {float(spot_snapshot.structure_quality):.0f} | "
            f"Trend quality {float(spot_snapshot.trend_quality):.0f} | "
            f"Regime quality {float(spot_snapshot.regime_quality):.0f} | "
            f"Location quality {float(spot_snapshot.location_quality):.0f}"
        )
        if str(getattr(confidence_snapshot, "note", "") or "").strip():
            confidence_note = (
                f"{confidence_note} | {str(getattr(confidence_snapshot, 'note', '')).strip()}"
            )
        recent_market_context = build_recent_market_context_snapshot(recent_market_events_df)
        recent_symbol_market_signal = build_recent_symbol_market_signal_snapshot(
            recent_market_events_df,
            symbol=coin,
            timeframe=timeframe,
        )
        adaptive_model = build_adaptive_context_model(adaptive_history_df)
        setup_calibration_model = build_setup_calibration_model(adaptive_history_df)
        setup_calibration_snapshot = build_setup_calibration_snapshot(
            setup_calibration_model,
            signal={
                "Setup Confirm": str(action_raw or ""),
                "AI Alignment": "Aligned" if direction_key(spot_snapshot.direction) == ai_spot_direction_key else "Not aligned",
                "Timeframe": str(timeframe or "Unknown"),
                "Scan Focus": "Unknown",
                "Direction": str(spot_snapshot.direction or ""),
            },
        )
        action_raw, action_reason_code = apply_setup_archive_calibration(
            action_raw,
            action_reason_code,
            calibration_delta=float(getattr(setup_calibration_snapshot, "delta", 0.0) or 0.0),
        )
        setup_confirm = _setup_confirm_display(action_raw, action_reason=action_reason_code)
        setup_reason = action_reason_text(action_reason_code)
        action_class = normalize_action_class(action_raw)
        if action_class.startswith("ENTER_"):
            setup_c_s = POSITIVE
        elif action_class == "PROBE":
            setup_c_s = WARNING
        elif action_class == "WATCH":
            setup_c_s = watch_setup_color
        else:
            setup_c_s = NEGATIVE
        ai_confidence_calibration_model = build_ai_confidence_calibration_model(adaptive_history_df)
        ai_confidence_calibration_snapshot = build_ai_confidence_calibration_snapshot(
            ai_confidence_calibration_model,
            signal={
                "Setup Confirm": str(action_raw or ""),
                "AI Alignment": "Aligned" if direction_key(spot_snapshot.direction) == ai_spot_direction_key else "Not aligned",
                "Timeframe": str(timeframe or "Unknown"),
                "Scan Focus": "Unknown",
                "Direction": str(spot_snapshot.direction or ""),
            },
        )
        ai_confidence_snapshot = build_ai_confidence_snapshot(
            direction=ai_spot_snapshot.direction,
            combined_score=float(ai_spot_snapshot.score),
            conviction_quality=float(ai_spot_snapshot.conviction_quality),
            timeframe_alignment=float(ai_spot_snapshot.timeframe_alignment),
            consensus_quality=float(ai_spot_snapshot.consensus_quality),
            support_votes=int(ai_spot_votes),
            timeframe_conflict=bool(ai_spot_snapshot.timeframe_conflict),
            degraded_data=bool(ai_spot_snapshot.degraded_data),
            archive_calibration_delta=float(getattr(ai_confidence_calibration_snapshot, "delta", 0.0) or 0.0),
            archive_calibration_note=str(getattr(ai_confidence_calibration_snapshot, "note", "") or ""),
        )
        ai_conf_bucket = str(ai_confidence_snapshot.label or "LOW").upper()
        ai_conf_c_s = POSITIVE if ai_conf_bucket == "HIGH" else (WARNING if ai_conf_bucket == "MEDIUM" else NEGATIVE)
        ai_confidence_note = _spot_ai_confidence_note(
            ai_spot_snapshot,
            float(ai_confidence_snapshot.score),
            ai_confidence_snapshot,
        )
        current_session_bucket = session_bucket_for_timestamp()
        adaptive_snapshot = build_live_signal_adaptive_snapshot(
            adaptive_model,
            signal={
                "Setup Confirm": str(action_raw or ""),
                "Lead": str(recent_symbol_market_signal.get("Lead") or "No LEAD"),
                "AI Alignment": "Aligned" if direction_key(spot_snapshot.direction) == ai_spot_direction_key else "Not aligned",
                "Market Lead": str(recent_market_context.get("Market Lead") or "No Clear Lead"),
                "Market Regime": str(recent_market_context.get("Market Regime") or "Unknown"),
                "Playbook Key": str(recent_market_context.get("Playbook Key") or "Unknown"),
                "Playbook": str(recent_market_context.get("Playbook") or "Unknown"),
                "Trade Gate Key": str(recent_market_context.get("Trade Gate Key") or "Unknown"),
                "Trade Gate": str(recent_market_context.get("Trade Gate") or "Unknown"),
                "Sector Rotation": str(recent_market_context.get("Sector Rotation") or "Unknown"),
                "Catalyst State": str(recent_market_context.get("Catalyst State") or "Unknown"),
                "Catalyst Window": str(recent_market_context.get("Catalyst Window") or "Unknown"),
                "Catalyst Scope": str(recent_market_context.get("Catalyst Scope") or "Unknown"),
                "Catalyst Targeting": str(recent_market_context.get("Catalyst Targeting") or "Unknown"),
                "Flow Proxy": str(recent_market_context.get("Flow Proxy") or "Unknown"),
                "Session": current_session_bucket,
                "Timeframe": str(timeframe or "Unknown"),
            },
        )
        market_context_note = str(recent_market_context.get("Context Note") or "").strip()
        scanner_signal_note = str(recent_symbol_market_signal.get("Signal Note") or "").strip()
        context_fit = _context_fit_snapshot(
            adaptive_snapshot,
            market_context=recent_market_context,
            recent_symbol_market_signal=recent_symbol_market_signal,
        )
        confidence_note = (
            f"{confidence_note} | Historical read: {adaptive_snapshot.note} | "
            f"Session fit: {adaptive_snapshot.session_fit_note}"
        )
        if str(getattr(setup_calibration_snapshot, "note", "") or "").strip():
            confidence_note = (
                f"{confidence_note} | Setup calibration: {str(getattr(setup_calibration_snapshot, 'note', '')).strip()}"
            )
        setup_snapshot_html = build_setup_snapshot_html(
            title="Setup Snapshot",
            text_muted=TEXT_MUTED,
            items=[
                {"label": "Δ (%)", "value": delta_display or "—", "color": delta_c_s, "title": delta_note},
                {
                    "label": "Setup Confirm",
                    "value": setup_confirm,
                    "color": setup_c_s,
                    "title": setup_reason,
                },
                {"label": "Direction", "value": signal_clean, "color": sig_c_s, "title": direction_note},
                {"label": "Confidence", "value": confidence_display, "color": conf_c_s, "title": confidence_note},
                {
                    "label": "AI Ensemble",
                    "value": _spot_ai_display_value(
                        direction_label,
                        ai_spot_direction_key,
                        ai_spot_votes,
                        "fallback" if ai_spot_snapshot.degraded_data else "",
                    ),
                    "color": ai_c_s,
                    "title": ai_spot_note,
                },
                {
                    "label": "AI Confidence",
                    "value": _spot_ai_confidence_display(ai_spot_snapshot, float(ai_confidence_snapshot.score)),
                    "color": ai_conf_c_s,
                    "title": ai_confidence_note,
                },
            ],
        )
        st.markdown(setup_snapshot_html, unsafe_allow_html=True)
        st.markdown(
            build_learned_edge_banner_html(
                title="Market Archive Read",
                label=(
                    f"{_trade_gate_display_label(context_fit['label'])} • "
                    f"{_spot_archive_status_label(adaptive_snapshot)}"
                ),
                note=_spot_archive_banner_note(
                    adaptive_snapshot,
                    context_fit=context_fit,
                    market_context=recent_market_context,
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
                        {
                            "name": "Ichimoku",
                            "value": ichimoku_trend,
                            "tooltip": f"Cloud trend context. {ichimoku_hover}".strip(),
                        },
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
                            "value": format_stochrsi(stochrsi_k_val, timeframe=timeframe),
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
                            "value": atr_comment.replace("▲", "").replace("▼", "").replace("–", ""),
                            "tooltip": "ATR/band-width regime.",
                        },
                        {
                            "name": "Volume",
                            "value": spike_label if volume_spike else "",
                            "tooltip": f"Abnormal volume event. {spike_hover}".strip(),
                        },
                    ],
                ),
            ],
        )
        if indicator_groups_html:
            st.markdown(indicator_groups_html, unsafe_allow_html=True)

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
        map_copy = _spot_execution_map_copy(spot_direction_key)
        left_path_label = str(map_copy.get("left_path") or "Buy Zone Path")
        left_zone_label = str(map_copy.get("left_zone") or "Buy Zone")
        trigger_label = str(map_copy.get("right_trigger") or "Breakout Trigger")
        right_path_label = str(map_copy.get("right_path") or "Breakout Path")
        left_tp_label = str(map_copy.get("left_tp") or "TP Zone")
        right_tp_label = str(map_copy.get("right_tp") or "TP Zone")
        left_stop_context = left_zone_label
        right_stop_context = trigger_label.replace(" Trigger", "").strip() or "Breakout"

        setup_cls = normalize_action_class(action_raw)
        setup_label = setup_confirm
        if setup_cls == "SKIP":
            plan_status = copy_text("spot.plan.mode.no_trade")
            plan_color = NEGATIVE
            plan_now = copy_text("spot.plan.skip.now")
            plan_entry = copy_text(
                "spot.plan.skip.entry",
                left_zone_label=left_zone_label,
                pullback_zone_text=pullback_zone_text,
                trigger_label=trigger_label,
                breakout_trigger=_fmt_price(breakout_trigger),
            )
            plan_protection = copy_text(
                "spot.plan.skip.protection",
                pullback_invalidation=_fmt_price(pullback_invalidation),
            )
            plan_next = copy_text("spot.plan.skip.next")
        elif setup_cls == "PROBE":
            plan_status = copy_text("spot.plan.mode.probe")
            plan_color = WARNING
            if spot_direction_key == "UPSIDE":
                plan_now = copy_text("spot.plan.probe.upside.now")
                plan_entry = copy_text(
                    "spot.plan.probe.upside.entry",
                    left_zone_label=left_zone_label,
                    pullback_zone_text=pullback_zone_text,
                    trigger_label=trigger_label,
                    breakout_trigger=_fmt_price(breakout_trigger),
                )
                plan_protection = copy_text(
                    "spot.plan.probe.upside.protection",
                    pullback_invalidation=_fmt_price(pullback_invalidation),
                    breakout_invalidation=_fmt_price(breakout_invalidation),
                )
                plan_next = copy_text(
                    "spot.plan.probe.upside.next",
                    left_tp_label=left_tp_label,
                    pullback_tp_text=pullback_tp_text,
                    right_tp_label=right_tp_label,
                    breakout_tp_text=breakout_tp_text,
                )
            elif spot_direction_key == "DOWNSIDE":
                plan_now = copy_text("spot.plan.probe.downside.now")
                plan_entry = copy_text("spot.plan.probe.downside.entry")
                plan_protection = copy_text(
                    "spot.plan.probe.downside.protection",
                    trigger_label=trigger_label,
                    breakout_trigger=_fmt_price(breakout_trigger),
                    pullback_invalidation=_fmt_price(pullback_invalidation),
                )
                plan_next = copy_text("spot.plan.probe.downside.next")
            else:
                plan_now = copy_text("spot.plan.probe.neutral.now")
                plan_entry = copy_text(
                    "spot.plan.probe.neutral.entry",
                    left_zone_label=left_zone_label,
                    pullback_zone_text=pullback_zone_text,
                    trigger_label=trigger_label,
                    breakout_trigger=_fmt_price(breakout_trigger),
                )
                plan_protection = copy_text("spot.plan.probe.neutral.protection")
                plan_next = copy_text("spot.plan.probe.neutral.next")
        elif setup_cls == "WATCH":
            plan_status = copy_text("spot.plan.mode.watch")
            plan_color = WARNING
            if spot_direction_key == "UPSIDE":
                plan_now = copy_text("spot.plan.watch.upside.now")
                plan_entry = copy_text(
                    "spot.plan.watch.upside.entry",
                    left_zone_label=left_zone_label,
                    pullback_zone_text=pullback_zone_text,
                    trigger_label=trigger_label,
                    breakout_trigger=_fmt_price(breakout_trigger),
                )
                plan_protection = copy_text(
                    "spot.plan.watch.upside.protection",
                    pullback_invalidation=_fmt_price(pullback_invalidation),
                    breakout_invalidation=_fmt_price(breakout_invalidation),
                )
                plan_next = copy_text(
                    "spot.plan.watch.upside.next",
                    left_tp_label=left_tp_label,
                    pullback_tp_text=pullback_tp_text,
                    right_tp_label=right_tp_label,
                    breakout_tp_text=breakout_tp_text,
                )
            elif spot_direction_key == "DOWNSIDE":
                plan_now = copy_text("spot.plan.watch.downside.now")
                plan_entry = copy_text(
                    "spot.plan.watch.downside.entry",
                    trigger_label=trigger_label,
                    breakout_trigger=_fmt_price(breakout_trigger),
                )
                plan_protection = copy_text(
                    "spot.plan.watch.downside.protection",
                    pullback_invalidation=_fmt_price(pullback_invalidation),
                )
                plan_next = copy_text(
                    "spot.plan.watch.downside.next",
                    breakout_invalidation=_fmt_price(breakout_invalidation),
                    right_tp_label=right_tp_label,
                    breakout_tp_text=breakout_tp_text,
                )
            else:
                plan_now = copy_text("spot.plan.watch.neutral.now")
                plan_entry = copy_text(
                    "spot.plan.watch.neutral.entry",
                    left_zone_label=left_zone_label,
                    pullback_zone_text=pullback_zone_text,
                    trigger_label=trigger_label,
                    breakout_trigger=_fmt_price(breakout_trigger),
                )
                plan_protection = copy_text(
                    "spot.plan.watch.neutral.protection",
                    pullback_invalidation=_fmt_price(pullback_invalidation),
                    breakout_invalidation=_fmt_price(breakout_invalidation),
                )
                plan_next = copy_text(
                    "spot.plan.watch.neutral.next",
                    left_tp_label=left_tp_label,
                    pullback_tp_text=pullback_tp_text,
                    right_tp_label=right_tp_label,
                    breakout_tp_text=breakout_tp_text,
                )
        elif spot_direction_key == "UPSIDE":
            plan_status = copy_text("spot.plan.mode.bullish_confirmed")
            plan_color = POSITIVE
            plan_now = copy_text("spot.plan.confirmed.upside.now")
            plan_entry = copy_text(
                "spot.plan.confirmed.upside.entry",
                left_path_label=left_path_label,
                pullback_zone_text=pullback_zone_text,
                right_path_label=right_path_label,
                trigger_label=trigger_label,
                breakout_trigger=_fmt_price(breakout_trigger),
            )
            plan_protection = copy_text(
                "spot.plan.confirmed.upside.protection",
                pullback_invalidation=_fmt_price(pullback_invalidation),
                breakout_invalidation=_fmt_price(breakout_invalidation),
            )
            plan_next = copy_text(
                "spot.plan.confirmed.upside.next",
                left_tp_label=left_tp_label,
                pullback_tp_text=pullback_tp_text,
                right_tp_label=right_tp_label,
                breakout_tp_text=breakout_tp_text,
            )
        else:
            plan_status = copy_text("spot.plan.mode.defensive_confirmed")
            plan_color = NEGATIVE
            plan_now = copy_text("spot.plan.confirmed.downside.now")
            plan_entry = copy_text(
                "spot.plan.confirmed.downside.entry",
                trigger_label=trigger_label,
                breakout_trigger=_fmt_price(breakout_trigger),
            )
            plan_protection = copy_text(
                "spot.plan.confirmed.downside.protection",
                pullback_invalidation=_fmt_price(pullback_invalidation),
            )
            plan_next = copy_text(
                "spot.plan.confirmed.downside.next",
                breakout_invalidation=_fmt_price(breakout_invalidation),
                right_tp_label=right_tp_label,
                breakout_tp_text=breakout_tp_text,
            )

        st.markdown(
            f"<div class='panel-box' style='border-left:4px solid {plan_color};'>"
            f"<b style='color:{plan_color}; font-size:1rem;'>{copy_text('spot.plan.title')}</b>"
            f"<div style='color:{TEXT_MUTED}; font-size:0.86rem; margin-top:4px;'>"
            f"{copy_text('spot.plan.label.mode')}: <b style='color:{plan_color};'>{plan_status}</b>"
            f"</div>"
            f"<div style='color:{TEXT_MUTED}; font-size:0.86rem; margin-top:4px;'>"
            f"{copy_text('spot.plan.label.setup')}: <b>{setup_label}</b>"
            f"</div>"
            f"<div style='color:{TEXT_MUTED}; font-size:0.86rem; margin-top:4px;'>"
            f"{copy_text('spot.plan.label.market_stance')}: <b>{context_fit['label']}</b> — {context_fit['aggression']}"
            f"</div>"
            f"<div style='color:{TEXT_MUTED}; font-size:0.86rem; margin-top:4px;'>"
            f"{copy_text('spot.plan.label.now')}: {plan_now}"
            f"</div>"
            f"<div style='color:{TEXT_MUTED}; font-size:0.86rem; margin-top:4px;'>"
            f"{copy_text('spot.plan.label.entry')}: {plan_entry}"
            f"</div>"
            f"<div style='color:{TEXT_MUTED}; font-size:0.86rem; margin-top:4px;'>"
            f"{copy_text('spot.plan.label.protection')}: {plan_protection}"
            f"</div>"
            f"<div style='color:{TEXT_MUTED}; font-size:0.86rem; margin-top:4px;'>"
            f"{copy_text('spot.plan.label.next')}: {plan_next}"
            f"</div>"
            f"<div style='color:{TEXT_MUTED}; font-size:0.86rem; margin-top:6px;'>"
            f"{copy_text('spot.plan.disclaimer')}</span>"
            f"</div></div>",
            unsafe_allow_html=True,
        )

        show_actionable_paths = not (setup_cls == "SKIP" or spot_direction_key == "DOWNSIDE")
        levels_section_label = (
            copy_text("spot.levels.actionable_title")
            if show_actionable_paths
            else copy_text("spot.levels.defensive_title")
        )

        st.markdown(
            f"<div style='margin:0.3rem 0 0.45rem 0; text-align:center; color:{TEXT_MUTED}; font-size:0.78rem; text-transform:uppercase; letter-spacing:0.45px;'>"
            f"{levels_section_label}"
            "</div>",
            unsafe_allow_html=True,
        )
        if show_actionable_paths:
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
            st.markdown(
                f"<div style='margin:0.15rem 0 0.45rem 0; text-align:center; color:{TEXT_MUTED}; font-size:0.78rem; text-transform:uppercase; letter-spacing:0.45px;'>"
                f"{_spot_execution_map_copy(spot_direction_key).get('section_title', 'Execution Map')}"
                "</div>",
                unsafe_allow_html=True,
            )
        else:
            render_kpi_grid(
                st,
                items=[
                    {
                        "label": "Reference Price",
                        "label_title": "Latest close of selected timeframe (not live tick).",
                        "value": _fmt_price(current_price),
                    },
                    {"label": "Raw Support", "value": _fmt_price(plan_support)},
                    {"label": "Raw Resistance", "value": _fmt_price(plan_resistance)},
                    {"label": trigger_label, "value": _fmt_price(breakout_trigger)},
                    {"label": f"Protective Stop ({left_stop_context})", "value": _fmt_price(pullback_invalidation)},
                    {"label": f"{right_tp_label} (after reclaim)", "value": breakout_tp_text},
                ],
                columns=3,
                align="center",
                card_min_height="128px",
                center_last_row=True,
            )
            st.caption(copy_text("spot.levels.defensive_caption"))

        if show_actionable_paths:
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
            left_path_color = POSITIVE if spot_direction_key == "UPSIDE" else WARNING
            right_path_color = ACCENT if spot_direction_key == "UPSIDE" else WARNING
            left_fill_color = "rgba(0, 255, 136, 0.18)" if spot_direction_key == "UPSIDE" else "rgba(255, 209, 102, 0.14)"
            left_tp_fill = "rgba(0, 255, 136, 0.12)" if spot_direction_key == "UPSIDE" else "rgba(255, 209, 102, 0.08)"
            right_tp_fill = "rgba(34, 211, 238, 0.12)" if spot_direction_key == "UPSIDE" else "rgba(255, 209, 102, 0.08)"
            trigger_color = ACCENT if spot_direction_key == "UPSIDE" else WARNING
            left_label_color = "#E8FFF7" if spot_direction_key == "UPSIDE" else "#FFF3D0"
            right_label_color = "#D8F9FF" if spot_direction_key == "UPSIDE" else "#FFF3D0"

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
