from ui.ctx import get_ctx

from datetime import datetime, timezone
import io

import pandas as pd
import plotly.graph_objs as go
import ta
from core.archive_policy import ARCHIVE_LEARNING_WINDOW_ROWS, ARCHIVE_RECENT_CONTEXT_ROWS
from core.archive_decision import (
    archive_symbol_key as _archive_symbol_key,
    build_archive_signal_decision_snapshot,
    select_archive_signal_scope_events,
)
from core.adaptive_weighting import (
    build_ai_confidence_calibration_model,
    build_ai_confidence_calibration_snapshot,
    build_confidence_calibration_model,
    build_scalp_calibration_model,
    build_scalp_calibration_snapshot,
    build_setup_calibration_model,
    build_setup_calibration_snapshot,
)
from core.confidence import build_ai_confidence_snapshot
from core.trading_copy import copy_text, playbook_key, trade_gate_key
from core.session_utils import session_bucket_for_timestamp
from core.confidence import confidence_bucket
from core.market_decision import apply_setup_archive_calibration, action_reason_text, normalize_action_class
from core.scalping import apply_scalp_archive_calibration, scalp_gate_thresholds, scalp_reason_text
from core.position_metrics import (
    compute_hard_invalidation,
    compute_health_decision,
    compute_position_pnl,
    estimate_liquidation,
)
from core.position_management import build_position_management_snapshot
from core.signal_tracker import (
    build_actual_exit_quality_profile,
    build_actual_trade_hold_profile,
    prefer_current_decision_version_slice,
)
from core.spot_execution_pipeline import build_spot_execution_pipeline
from ui.primitives import render_help_details, render_insight_card, render_page_header
from ui.signal_panels import build_indicator_groups_html, build_learned_edge_banner_html, build_setup_snapshot_html
from ui.signal_formatters import (
    adx_bucket_only as _adx_bucket_only,
    ai_confidence_display as _ai_confidence_display,
    ai_confidence_note as _ai_confidence_note,
    ai_spot_note as _ai_spot_note,
    compact_note_parts as _compact_note_parts,
    context_fit_snapshot as _context_fit_snapshot,
    setup_confirm_display as _setup_confirm_display,
    spot_bias_label as _spot_bias_label,
    spot_confidence_display as _confidence_display,
    trade_gate_display_label as _trade_gate_display_label,
)
from ui.snapshot_cache import live_or_snapshot


def _position_hold_archive_slice(
    df_events: pd.DataFrame,
    *,
    symbol: str,
    timeframe: str,
    direction: str,
    setup_confirm: str = "",
) -> tuple[pd.DataFrame, str]:
    return select_archive_signal_scope_events(
        df_events,
        symbol=symbol,
        timeframe=timeframe,
        direction=direction,
        setup_confirm=setup_confirm,
    )


def _position_hold_window_note(snapshot: dict[str, object], *, scope_label: str) -> tuple[str, str]:
    scope = str(scope_label or "archive slice").strip()
    resolved_signals = int(snapshot.get("resolved_signals") or 0)
    if resolved_signals <= 0:
        return (
            f"No resolved archive pocket yet for <b>{scope}</b>. Historical hold guidance is still building.",
            "neutral",
        )
    if not bool(snapshot.get("available")):
        return (
            (
                f"Historical hold guidance is still building for <b>{scope}</b>. "
                f"We have <b>{resolved_signals}</b> resolved signals, but the checkpoint sample is not deep enough yet."
            ),
            "neutral",
        )
    best_bar = int(snapshot.get("best_bar") or 0)
    best_label = str(snapshot.get("best_label") or "").strip() or "around 0 bars"
    best_style = str(snapshot.get("best_style") or "Standard Hold").strip()
    sample = int(snapshot.get("sample") or 0)
    avg_dir_return_pct = float(snapshot.get("avg_dir_return_pct") or 0.0)
    follow_through_pct = float(snapshot.get("follow_through_pct") or 0.0)
    fade_after_bar = int(snapshot.get("fade_after_bar") or 0)
    fade_text = (
        f"Edge fades after roughly <b>{fade_after_bar}</b> bars."
        if fade_after_bar > 0
        else "Edge has not clearly faded inside the measured checkpoint ladder yet."
    )
    tone = "positive" if avg_dir_return_pct > 0.0 else "warning"
    lead_text = (
        f"<b>Best at: {best_bar} bars</b><br>"
        if best_bar > 0
        else f"<b>Suggested hold: {best_label}</b><br>"
    )
    body_text = lead_text + (
        f"Scope: <b>{scope}</b>.<br>"
        f"Style: <b>{best_style}</b>. Sample at that checkpoint: <b>{sample}</b> resolved signals.<br>"
        f"Avg directional return: <b>{avg_dir_return_pct:+.2f}%</b>, follow-through: <b>{follow_through_pct:.1f}%</b>. "
        f"{fade_text}"
    )
    return (
        body_text,
        tone,
    )


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
        return "Higher-TF"
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


def _tone_color(tone: str, *, accent: str, positive: str, negative: str, warning: str) -> str:
    key = str(tone or "").strip().lower()
    if key == "positive":
        return positive
    if key == "negative":
        return negative
    if key == "warning":
        return warning
    return accent


def _position_archive_status_label(adaptive_snapshot) -> str:
    archive_label = str(getattr(adaptive_snapshot, "archive_guardrail_label", "") or "").strip()
    adaptive_label = str(getattr(adaptive_snapshot, "label", "") or "").strip()
    session_label = str(getattr(adaptive_snapshot, "session_fit_label", "") or "").strip()

    if archive_label == "Archive Guardrail":
        return copy_text("position.archive.status.guardrail")
    if archive_label == "Archive Caution":
        return copy_text("position.archive.status.caution")
    if adaptive_label == "Historically Favored":
        return copy_text("position.archive.status.supportive")
    if adaptive_label == "Historically Weak":
        return copy_text("position.archive.status.fragile")
    if session_label == "Session Supportive":
        return copy_text("position.archive.status.session_supportive")
    if session_label == "Session Fragile":
        return copy_text("position.archive.status.session_fragile")
    return copy_text("position.archive.status.mixed")


def _position_archive_banner_note(
    adaptive_snapshot,
    *,
    context_fit: dict[str, str],
    market_context: dict[str, str] | None = None,
    market_context_note: str = "",
    scanner_signal_note: str = "",
) -> str:
    market_context = dict(market_context or {})
    stance = _trade_gate_display_label(str((context_fit or {}).get("label") or ""))
    stance_key = trade_gate_key((context_fit or {}).get("gate_key") or stance)
    adaptive_label = str(getattr(adaptive_snapshot, "label", "") or "").strip()
    archive_label = str(getattr(adaptive_snapshot, "archive_guardrail_label", "") or "").strip()
    archive_note = str(getattr(adaptive_snapshot, "archive_guardrail_note", "") or "").strip()
    session_label = str(getattr(adaptive_snapshot, "session_fit_label", "") or "").strip()
    session_note = str(getattr(adaptive_snapshot, "session_fit_note", "") or "").strip()
    history_note = str(getattr(adaptive_snapshot, "note", "") or "").strip()
    trade_gate = str(market_context.get("Trade Gate") or "").strip()
    trade_gate_value_key = trade_gate_key(market_context.get("Trade Gate Key") or trade_gate)
    playbook = str(market_context.get("Playbook") or "").strip()
    catalyst_window = str(market_context.get("Catalyst Window") or "").strip()
    flow_proxy = str(market_context.get("Flow Proxy") or "").strip()

    stance_summary = ""
    if stance_key == "NO_TRADE":
        stance_summary = copy_text("position.archive.stance.stand_aside")
    elif stance_key == "DEFENSIVE_ONLY":
        stance_summary = copy_text("position.archive.stance.defensive")
    elif stance_key == "TRADEABLE":
        stance_summary = copy_text("position.archive.stance.tradeable")
    elif stance_key == "SELECTIVE_ONLY":
        stance_summary = copy_text("position.archive.stance.selective")

    plain_archive_note = ""
    if archive_label == "Archive Guardrail":
        plain_archive_note = copy_text("position.archive.history.guardrail")
    elif archive_label == "Archive Caution":
        plain_archive_note = copy_text("position.archive.history.caution")

    plain_history_note = ""
    if not plain_archive_note:
        if adaptive_label == "Historically Favored":
            plain_history_note = copy_text("position.archive.history.supportive")
        elif adaptive_label == "Historically Weak":
            plain_history_note = copy_text("position.archive.history.fragile")
        elif adaptive_label == "Historically Neutral":
            plain_history_note = copy_text("position.archive.history.neutral")

    plain_session_note = ""
    if session_label == "Session Supportive":
        plain_session_note = copy_text("position.archive.session.supportive")
    elif session_label == "Session Fragile":
        plain_session_note = copy_text("position.archive.session.fragile")

    context_bits: list[str] = []
    if trade_gate_value_key == "NO_TRADE":
        context_bits.append(copy_text("position.archive.context.trade_gate.no_trade"))
    elif trade_gate_value_key == "SELECTIVE_ONLY":
        context_bits.append(copy_text("position.archive.context.trade_gate.selective"))
    elif trade_gate_value_key == "TRADEABLE":
        context_bits.append(copy_text("position.archive.context.trade_gate.tradeable"))
    if catalyst_window.startswith("Far"):
        context_bits.append(copy_text("position.archive.context.catalyst.far"))
    elif catalyst_window.startswith(("Near", "High Impact")):
        context_bits.append(copy_text("position.archive.context.catalyst.near"))
    elif catalyst_window.startswith("Blocking"):
        context_bits.append(copy_text("position.archive.context.catalyst.blocking"))
    if flow_proxy == "Flow Balanced":
        context_bits.append(copy_text("position.archive.context.flow.balanced"))
    elif flow_proxy in {"Shorts Crowded", "Longs Crowded"}:
        context_bits.append(copy_text("position.archive.context.flow.crowded"))

    plain_context_note = ""
    if context_bits:
        plain_context_note = " • ".join(
            [context_bits[0].capitalize() + "."] + [bit.capitalize() + "." for bit in context_bits[1:2]]
        )
    elif playbook_key(playbook) == "WAIT_CONFIRMATION":
        plain_context_note = copy_text("position.archive.context.playbook.wait")

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
    fetch_signal_forward_windows_df = get_ctx(ctx, "fetch_signal_forward_windows_df")
    build_adaptive_context_model = get_ctx(ctx, "build_adaptive_context_model")
    build_live_signal_adaptive_snapshot = get_ctx(ctx, "build_live_signal_adaptive_snapshot")
    build_hold_window_intelligence = get_ctx(ctx, "build_hold_window_intelligence")
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
            "2) Respect the <b>anchor timeframe</b> and its <b>Technical Invalidation</b> as your hard risk line. "
            "3) Follow <b>Position Management</b> for the immediate action. "
            "4) Use <b>Setup Snapshot</b> and <b>Market Archive Read</b> as context, not as reasons to ignore the risk line."
        ),
    )

    # Assign a unique key to avoid StreamlitDuplicateElementId errors
    coin = _normalize_coin_input(st.text_input(
        "Coin (e.g. BTC, ETH, TAO)",
        value="BTC",
        key="position_coin_input",
    ))
    archive_symbol = _archive_symbol_key(coin) or coin
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
            limit=ARCHIVE_LEARNING_WINDOW_ROWS,
            status="RESOLVED",
            source="Market",
            db_path=signal_tracker_db_path,
        )
        adaptive_history_df = prefer_current_decision_version_slice(
            adaptive_history_df,
            source="Market",
        )
        recent_market_events_df = fetch_signal_events_df(
            limit=ARCHIVE_RECENT_CONTEXT_ROWS,
            source="Market",
            db_path=signal_tracker_db_path,
        )
        adaptive_forward_windows_df = fetch_signal_forward_windows_df(
            signal_keys=adaptive_history_df["signal_key"].fillna("").astype(str).tolist()
            if "signal_key" in adaptive_history_df.columns
            else [],
            db_path=signal_tracker_db_path,
        )
        recent_market_context = build_recent_market_context_snapshot(recent_market_events_df)
        adaptive_model = build_adaptive_context_model(adaptive_history_df)
        confidence_calibration_model = build_confidence_calibration_model(adaptive_history_df)
        setup_calibration_model = build_setup_calibration_model(adaptive_history_df)
        ai_confidence_calibration_model = build_ai_confidence_calibration_model(adaptive_history_df)
        scalp_calibration_model = build_scalp_calibration_model(adaptive_history_df)
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

        if len(selected_timeframes) > 1:
            st.markdown(
                f"<div class='panel-box' style='padding:10px 12px; margin-bottom:10px;'>"
                f"<b style='color:{ACCENT};'>Anchor timeframe: {largest_tf}</b><br>"
                f"<span style='color:{TEXT_MUTED}; font-size:0.88rem;'>"
                f"Use {largest_tf} for the main risk line and core management call. "
                f"Smaller frames are timing and context only.</span></div>",
                unsafe_allow_html=True,
            )

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
                pipeline = build_spot_execution_pipeline(
                    symbol=coin,
                    actual_symbol=actual_symbol,
                    source_provider=source_provider,
                    timeframe=tf,
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
                    f"Base Move: <b style='color:{ACCENT};'>{pnl_percent_raw:+.2f}%</b>"
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
                if len(selected_timeframes) > 1:
                    frame_role = "Anchor timeframe" if tf == largest_tf else "Timing / context frame"
                    frame_role_note = (
                        "Use this frame for the main risk line and the core management call."
                        if tf == largest_tf
                        else f"Use this frame to fine-tune timing only. Do not let it overrule {largest_tf} on its own."
                    )
                    st.markdown(
                        f"<div style='color:{TEXT_MUTED}; font-size:0.84rem; margin:4px 0 10px 0;'>"
                        f"<b>{frame_role}:</b> {frame_role_note}</div>",
                        unsafe_allow_html=True,
                    )

                signal_dir = pipeline.signal_direction
                signal_clean = direction_label(signal_dir)
                signal_dir_legacy = pipeline.signal_direction_legacy
                execution_confidence = pipeline.execution_confidence_snapshot
                conviction_lbl = pipeline.execution_conviction_label
                ai_spot_direction = pipeline.ai_spot_direction
                ai_spot_votes = int(pipeline.ai_spot_votes)
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
                            timeframe=tf,
                            execution_snapshot=pipeline.execution_snapshot,
                            trend_led_snapshot=pipeline.trend_led_snapshot,
                            ai_led_snapshot=pipeline.ai_led_snapshot,
                            spot_direction=spot_snapshot.direction,
                            ai_direction=ai_spot_snapshot.direction,
                        )
                    except Exception:
                        scalp_direction = None
                        entry_s = target_s = stop_s = rr_ratio = 0.0
                        breakout_note = ""

                # Setup confirm mirrors market/spot decision policy.
                action_raw, action_reason_code = pipeline.action_raw, pipeline.action_reason_code

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
                setup_confirm = _setup_confirm_display(
                    action_raw,
                    action_reason=action_reason_code,
                    direction=str(spot_snapshot.direction or ""),
                )
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
                    f"Spot bias ({_spot_anchor_pair_label(spot_snapshot)}): {_spot_bias_label(spot_snapshot.direction)} | "
                    f"Combined score {float(spot_snapshot.score):.1f} | {str(spot_snapshot.note or '').strip()} | "
                    f"{_spot_tf_note(_spot_lead_snapshot(spot_snapshot))} | "
                    f"{_spot_tf_note(_spot_confirm_snapshot(spot_snapshot))} | "
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
                if str(getattr(confidence_snapshot, "note", "") or "").strip():
                    confidence_note = (
                        f"{confidence_note} | {str(getattr(confidence_snapshot, 'note', '')).strip()}"
                    )
                ai_note = _ai_spot_note(ai_spot_snapshot)
                setup_calibration_snapshot = build_setup_calibration_snapshot(
                    setup_calibration_model,
                    signal={
                        "Setup Confirm": str(action_raw or ""),
                        "AI Alignment": "Aligned" if direction_key(spot_snapshot.direction) == ai_spot_direction else "Not aligned",
                        "Timeframe": str(tf or "Unknown"),
                        "Scan Focus": "Unknown",
                        "Direction": str(spot_snapshot.direction or ""),
                    },
                )
                action_raw, action_reason_code = apply_setup_archive_calibration(
                    action_raw,
                    action_reason_code,
                    calibration_delta=float(getattr(setup_calibration_snapshot, "delta", 0.0) or 0.0),
                )
                setup_confirm = _setup_confirm_display(
                    action_raw,
                    action_reason=action_reason_code,
                    direction=str(spot_snapshot.direction or ""),
                )
                setup_reason = action_reason_text(action_reason_code)
                action_class = normalize_action_class(action_raw)
                if action_class.startswith("ENTER_"):
                    setup_color = POSITIVE
                elif action_class == "PROBE":
                    setup_color = WARNING
                elif action_class == "WATCH":
                    setup_color = watch_setup_color
                else:
                    setup_color = NEGATIVE
                ai_confidence_calibration_snapshot = build_ai_confidence_calibration_snapshot(
                    ai_confidence_calibration_model,
                    signal={
                        "Setup Confirm": str(action_raw or ""),
                        "AI Alignment": "Aligned" if direction_key(spot_snapshot.direction) == ai_spot_direction else "Not aligned",
                        "Timeframe": str(tf or "Unknown"),
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
                    support_votes=int(pipeline.ai_spot_votes),
                    timeframe_conflict=bool(ai_spot_snapshot.timeframe_conflict),
                    degraded_data=bool(ai_spot_snapshot.degraded_data),
                    archive_calibration_delta=float(getattr(ai_confidence_calibration_snapshot, "delta", 0.0) or 0.0),
                    archive_calibration_note=str(getattr(ai_confidence_calibration_snapshot, "note", "") or ""),
                )
                ai_conf_bucket = str(ai_confidence_snapshot.label or "LOW").upper()
                ai_conf_color = POSITIVE if ai_conf_bucket == "HIGH" else (WARNING if ai_conf_bucket == "MEDIUM" else NEGATIVE)
                ai_confidence_note = _ai_confidence_note(
                    ai_spot_snapshot,
                    float(ai_confidence_snapshot.score),
                    ai_confidence_snapshot,
                )
                if str(getattr(setup_calibration_snapshot, "note", "") or "").strip():
                    confidence_note = (
                        f"{confidence_note} | Setup calibration: {str(getattr(setup_calibration_snapshot, 'note', '')).strip()}"
                    )
                current_session_bucket = session_bucket_for_timestamp()
                current_sector_tag = str(classify_symbol_sector(coin) or "").strip()
                recent_symbol_market_signal = build_recent_symbol_market_signal_snapshot(
                    recent_market_events_df,
                    symbol=archive_symbol,
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
                    symbol=archive_symbol,
                    timeframe=tf,
                    direction=direction,
                    sector_tag=current_sector_tag,
                    playbook=str(recent_market_context.get("Playbook") or ""),
                    session_bucket=current_session_bucket,
                    trade_gate=str(recent_market_context.get("Trade Gate") or ""),
                    catalyst_window=str(recent_market_context.get("Catalyst Window") or ""),
                )
                archive_signal_decision = build_archive_signal_decision_snapshot(
                    df_events=adaptive_history_df,
                    df_forward_windows=adaptive_forward_windows_df,
                    symbol=archive_symbol,
                    timeframe=tf,
                    direction=direction,
                    setup_confirm=action_raw,
                    build_hold_window_intelligence_fn=build_hold_window_intelligence,
                )
                hold_window_snapshot = archive_signal_decision.hold_window
                hold_window_note, hold_window_tone = _position_hold_window_note(
                    hold_window_snapshot,
                    scope_label=archive_signal_decision.scope_label,
                )
                exit_quality = build_actual_exit_quality_profile(
                    adaptive_history_df,
                    symbol=archive_symbol,
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
                            "title": setup_reason,
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
                archive_banner_html = build_learned_edge_banner_html(
                    title="Market Archive Read",
                    label=(
                        f"{_trade_gate_display_label(context_fit['label'])} • "
                        f"{_position_archive_status_label(adaptive_snapshot)}"
                    ),
                    note=_position_archive_banner_note(
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
                                    "value": candle_pattern or "",
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
                # Risk alert for position
                pos_dir = direction_key(direction)
                risk_alert_html = ""
                if pos_dir == direction_key(spot_snapshot.direction) and float(confidence_snapshot.score) < 50:
                    risk_alert_html = (
                        f"<div style='background:#2D0A0A; border-left:4px solid {NEGATIVE}; "
                        f"padding:6px 10px; border-radius:4px; margin:4px 0; font-size:0.82rem;'>"
                        f"<span style='color:{NEGATIVE}; font-weight:600;'>Low Confidence</span>"
                        f"<span style='color:{TEXT_MUTED};'> — Position direction matches the spot bias but confidence "
                        f"is only {float(confidence_snapshot.score):.0f}%. Consider tighter risk control.</span></div>"
                    )
                elif pos_dir != direction_key(spot_snapshot.direction) and direction_key(spot_snapshot.direction) != "NEUTRAL":
                    risk_alert_html = (
                        f"<div style='background:#2D0A0A; border-left:4px solid {NEGATIVE}; "
                        f"padding:6px 10px; border-radius:4px; margin:4px 0; font-size:0.82rem;'>"
                        f"<span style='color:{NEGATIVE}; font-weight:600;'>Direction Conflict</span>"
                        f"<span style='color:{TEXT_MUTED};'> — Your {direction_label(direction)} position conflicts with "
                        f"the current spot bias {direction_label(spot_snapshot.direction)}. Review position validity.</span></div>"
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
                        "PnL Base (%)": round(float(pnl_percent_raw), 4),
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
                management_color = _tone_color(
                    management_snapshot.tone,
                    accent=ACCENT,
                    positive=POSITIVE,
                    negative=NEGATIVE,
                    warning=WARNING,
                )
                management_scope = "Anchor" if tf == largest_tf else "Timing"
                st.markdown(
                    f"<div class='panel-box' style='padding:10px 12px; border-left:4px solid {management_color};'>"
                    f"<div style='display:flex; justify-content:space-between; gap:8px; flex-wrap:wrap; font-size:0.92rem;'>"
                    f"<span style='color:{TEXT_MUTED};'>{tf} {copy_text('position.panel.title')} ({copy_text('position.panel.scope.anchor') if management_scope == 'Anchor' else copy_text('position.panel.scope.timing')})</span> "
                    f"<b style='color:{management_color};'>{management_snapshot.label} ({management_snapshot.score}/100)</b>"
                    f"</div>"
                    f"<div style='color:{TEXT_MUTED}; font-size:0.86rem; margin-top:4px;'>"
                    f"{copy_text('position.panel.label.now')}: <b>{management_snapshot.size_guidance}</b>"
                    f"</div>"
                    f"<div style='color:{TEXT_MUTED}; font-size:0.86rem; margin-top:4px;'>"
                    f"{copy_text('position.panel.label.adds')}: <b>{management_snapshot.adds_guidance}</b>"
                    f"</div>"
                    f"<div style='color:{TEXT_MUTED}; font-size:0.86rem; margin-top:4px;'>"
                    f"{copy_text('position.panel.label.market_stance')}: <b>{_trade_gate_display_label(context_fit['label'])}</b> — {context_fit['aggression']}"
                    f"</div>"
                    f"<div style='color:{TEXT_MUTED}; font-size:0.86rem; margin-top:4px;'>"
                    f"{copy_text('position.panel.label.hard_risk_line')}: <b style='color:{WARNING};'>{invalidation:,.4f}</b>"
                    f"</div>"
                    f"<div style='color:{TEXT_MUTED}; font-size:0.86rem; margin-top:4px;'>"
                    f"{management_snapshot.note}"
                    f"</div>"
                    f"<div style='color:{TEXT_MUTED}; font-size:0.86rem; margin-top:2px;'>"
                    f"{copy_text('position.panel.label.next')}: {management_snapshot.risk_guidance}"
                    f"</div>"
                    f"</div>",
                    unsafe_allow_html=True,
                )
                if risk_alert_html:
                    st.markdown(risk_alert_html, unsafe_allow_html=True)

                st.markdown(setup_snapshot_html, unsafe_allow_html=True)
                st.markdown(archive_banner_html, unsafe_allow_html=True)
                render_insight_card(
                    st,
                    title="Historical Hold Window",
                    body_html=hold_window_note,
                    tone=hold_window_tone,
                )
                if indicator_groups_html:
                    st.markdown(indicator_groups_html, unsafe_allow_html=True)

                if tf in {"1m", "3m", "5m", "15m", "1h"}:
                    with st.expander("Optional Scalp Read", expanded=False):
                        st.markdown(
                            f"<div style='color:{TEXT_MUTED}; font-size:0.84rem; margin-bottom:8px;'>"
                            f"This is a separate short-term lens. It should not override the core position plan above."
                            f"</div>",
                            unsafe_allow_html=True,
                        )
                        if df_scalp is not None and len(df_scalp) > 30:
                            gate_min_rr, gate_min_adx, gate_min_confidence = scalp_gate_thresholds(tf)
                            scalp_calibration_snapshot = build_scalp_calibration_snapshot(
                                scalp_calibration_model,
                                signal={
                                    "Setup Confirm": str(action_raw or ""),
                                    "AI Alignment": (
                                        "Aligned"
                                        if direction_key(spot_snapshot.direction) == ai_spot_direction
                                        else "Not aligned"
                                    ),
                                    "Timeframe": str(tf or "Unknown"),
                                    "Scan Focus": "Unknown",
                                    "Direction": str(spot_snapshot.direction or signal_dir or ""),
                                },
                            )
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
                                timeframe=tf,
                                setup_confirm=action_raw,
                                market_trade_gate_key=str(
                                    recent_market_context.get("Trade Gate Key")
                                    or recent_market_context.get("Trade Gate")
                                    or ""
                                ),
                                archive_guardrail_penalty=float(
                                    getattr(adaptive_snapshot, "archive_guardrail_penalty", 0.0) or 0.0
                                ),
                                archive_guardrail_label=str(
                                    getattr(adaptive_snapshot, "archive_guardrail_label", "") or ""
                                ),
                            )
                            scalp_ok, scalp_reason = apply_scalp_archive_calibration(
                                scalp_ok,
                                scalp_reason,
                                calibration_delta=float(getattr(scalp_calibration_snapshot, "delta", 0.0) or 0.0),
                                rr_ratio=rr_ratio,
                                adx_val=adx_val,
                                confidence=float(execution_confidence.score),
                                timeframe=tf,
                            )
                            scalp_note = scalp_reason_text(
                                scalp_reason,
                                timeframe=tf,
                                min_rr=gate_min_rr,
                                min_adx=gate_min_adx,
                                min_confidence=gate_min_confidence,
                            ) or "No valid scalping setup with current filters."
                            scalp_calibration_note = str(getattr(scalp_calibration_snapshot, "note", "") or "").strip()
                            if scalp_calibration_note:
                                scalp_note = f"{scalp_note} {scalp_calibration_note}".strip()
                            if scalp_note in {"Invalid plan levels", "Invalid ATR/price"}:
                                scalp_note = "No valid plan on current candle structure."
                            scalp_show_conditional = bool(scalp_direction) and str(scalp_reason or "").upper() not in {
                                "NO_SCALP_DIRECTION",
                                "SIGNAL_DIRECTION_NEUTRAL",
                                "UNSUPPORTED_TIMEFRAME",
                            }
                            show_levels = bool(entry_s and stop_s and target_s)
                            breakout_context = str(breakout_note or "").strip()

                            if scalp_ok and scalp_direction:
                                color = POSITIVE if scalp_direction == "LONG" else NEGATIVE
                                context_line = (
                                    f"<br><span style='color:{TEXT_MUTED};'>Trigger context: {breakout_context}</span>"
                                    if breakout_context
                                    else ""
                                )
                                st.markdown(
                                    f"""
                                    <div class='panel-box' style='border-left:4px solid {color};'>
                                      <div style='display:flex; justify-content:space-between; gap:8px; flex-wrap:wrap;'>
                                        <span style='color:{color}; font-weight:800;'>Scalp {direction_label(scalp_direction)}</span>
                                        <span style='color:{color}; font-weight:700;'>R:R {rr_ratio:.2f}</span>
                                      </div>
                                      <div style='color:{TEXT_MUTED}; font-size:0.88rem; margin-top:6px; line-height:1.65;'>
                                        Your Entry <b style='color:{ACCENT};'>${entry_price:,.4f}</b> |
                                        Model Entry <b style='color:{ACCENT};'>${entry_s:,.4f}</b><br>
                                        Stop <b style='color:{NEGATIVE};'>${stop_s:,.4f}</b> |
                                        Target <b style='color:{POSITIVE};'>${target_s:,.4f}</b><br>
                                        {'Good setup quality (R:R requirement passed).' if rr_ratio >= gate_min_rr else f'R:R is below requirement ({gate_min_rr:.2f}).'}
                                        {context_line}
                                      </div>
                                    </div>
                                    """,
                                    unsafe_allow_html=True,
                                )
                            elif scalp_show_conditional and scalp_direction:
                                color = POSITIVE if scalp_direction == "LONG" else NEGATIVE
                                levels_html = ""
                                if show_levels:
                                    levels_html = (
                                        f"Reference Entry <b style='color:{ACCENT};'>${entry_s:,.4f}</b> | "
                                        f"Stop <b style='color:{NEGATIVE};'>${stop_s:,.4f}</b> | "
                                        f"Target <b style='color:{POSITIVE};'>${target_s:,.4f}</b><br>"
                                        f"R:R <b>{float(rr_ratio or 0.0):.2f}</b><br>"
                                    )
                                trigger_html = (
                                    f"<span style='color:{TEXT_MUTED};'>Trigger context: {breakout_context}</span><br>"
                                    if breakout_context
                                    else ""
                                )
                                st.markdown(
                                    f"""
                                    <div class='panel-box' style='border-left:4px solid {WARNING};'>
                                      <div style='display:flex; justify-content:space-between; gap:8px; flex-wrap:wrap;'>
                                        <span style='color:{color}; font-weight:800;'>Scalp {direction_label(scalp_direction)} (Conditional)</span>
                                        <span style='color:{WARNING}; font-weight:700;'>Reference Only</span>
                                      </div>
                                      <div style='color:{TEXT_MUTED}; font-size:0.88rem; margin-top:6px; line-height:1.65;'>
                                        {levels_html}
                                        {trigger_html}
                                        {scalp_note}
                                      </div>
                                    </div>
                                    """,
                                    unsafe_allow_html=True,
                                )
                            else:
                                st.markdown(
                                    f"<div class='panel-box' style='border-left:4px solid {WARNING};'>"
                                    f"<b style='color:{WARNING};'>Scalp Read: Not Available</b><br>"
                                    f"<span style='color:{TEXT_MUTED}; font-size:0.86rem;'>Reason: {scalp_note}</span>"
                                    f"</div>",
                                    unsafe_allow_html=True,
                                )
                        else:
                            st.markdown(
                                f"<div class='panel-box' style='border-left:4px solid {WARNING};'>"
                                f"<b style='color:{WARNING};'>Scalp Read: Not Available</b><br>"
                                f"<span style='color:{TEXT_MUTED}; font-size:0.86rem;'>Reason: Not enough recent candles for the scalp model.</span>"
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
                    label="Download Decision Report (CSV backup)",
                    data=csv_bytes,
                    file_name=f"position_report_{coin.replace('/', '_')}.csv",
                    mime="text/csv",
                    on_click="ignore",
                )
