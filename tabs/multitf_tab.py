from ui.ctx import get_ctx

import html

import pandas as pd
from core.archive_policy import ARCHIVE_LEARNING_WINDOW_ROWS
from core.adaptive_weighting import (
    build_ai_confidence_calibration_model,
    build_ai_confidence_calibration_snapshot,
    build_confidence_calibration_model,
    build_setup_calibration_model,
    build_setup_calibration_snapshot,
)
from core.confidence import (
    build_ai_confidence_snapshot,
    build_confidence_snapshot,
    confidence_bucket,
)
from core.market_decision import (
    apply_setup_archive_calibration,
    normalize_action_class,
)
from core.multitf import HIGHER_TFS, TF_SEQUENCE, TF_WEIGHTS, compute_multitf_alignment, summarize_scope_bias
from core.signal_tracker import (
    fetch_signal_events_df as _fetch_signal_events_df_fallback,
    get_signal_tracker_db_path as _get_signal_tracker_db_path_fallback,
    init_signal_tracker_db as _init_signal_tracker_db_fallback,
    prefer_current_decision_version_slice,
)
from core.spot_execution_pipeline import build_spot_execution_pipeline
from core.spot_direction import build_spot_direction_snapshot
from core.trading_copy import copy_text
from ui.primitives import render_help_details, render_page_header
from ui.signal_panels import normalize_indicator_label
from ui.signal_formatters import (
    ai_confidence_display as _shared_ai_confidence_display,
    setup_confirm_display as _shared_setup_confirm_display,
    spot_confidence_display as _shared_spot_confidence_display,
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


def render(ctx: dict) -> None:
    st = get_ctx(ctx, "st")
    ACCENT = get_ctx(ctx, "ACCENT")
    TEXT_MUTED = get_ctx(ctx, "TEXT_MUTED")
    POSITIVE = get_ctx(ctx, "POSITIVE")
    NEGATIVE = get_ctx(ctx, "NEGATIVE")
    WARNING = get_ctx(ctx, "WARNING")
    _normalize_coin_input = get_ctx(ctx, "_normalize_coin_input")
    _validate_coin_symbol = get_ctx(ctx, "_validate_coin_symbol")
    fetch_ohlcv = get_ctx(ctx, "fetch_ohlcv")
    fetch_signal_events_df = ctx.get("fetch_signal_events_df", _fetch_signal_events_df_fallback)
    analyse = get_ctx(ctx, "analyse")
    direction_key = get_ctx(ctx, "direction_key")
    direction_label = get_ctx(ctx, "direction_label")
    format_delta = get_ctx(ctx, "format_delta")
    format_stochrsi = get_ctx(ctx, "format_stochrsi")
    get_signal_tracker_db_path = ctx.get("get_signal_tracker_db_path", _get_signal_tracker_db_path_fallback)
    init_signal_tracker_db = ctx.get("init_signal_tracker_db", _init_signal_tracker_db_fallback)
    ml_ensemble_predict = get_ctx(ctx, "ml_ensemble_predict")
    _calc_conviction = get_ctx(ctx, "_calc_conviction")

    def _adx_label(adx: float) -> str:
        try:
            value = float(adx)
        except Exception:
            return ""
        if value < 20:
            return "Weak"
        if value < 25:
            return "Starting"
        if value < 50:
            return "Strong"
        if value < 75:
            return "Very Strong"
        return "Extreme"

    def _setup_confirm_display(raw_action: str, action_reason: str | None = None, direction: str | None = None) -> str:
        return _shared_setup_confirm_display(raw_action, action_reason=action_reason, direction=direction)

    def _ai_fallback_note(status: str) -> str:
        key = str(status or "").strip().lower()
        if not key:
            return ""
        if key == "insufficient_candles":
            return "AI shown as Neutral for safety (not enough candles)"
        if key == "insufficient_features":
            return "AI shown as Neutral for safety (not enough clean data)"
        if key == "single_class_window":
            return "AI shown as Neutral for safety (training window had one class)"
        if key == "model_exception":
            return "AI shown as Neutral for safety (model exception)"
        return f"AI shown as Neutral for safety ({key.replace('_', ' ')})"

    def _spot_ai_display_value(ai_dir_key: str, ai_votes: int, fallback_note: str) -> str:
        base = f"{direction_label(ai_dir_key)} ({ai_votes}/3)"
        if fallback_note:
            return f"{direction_label(ai_dir_key)}* ({ai_votes}/3)"
        return base

    def _conviction_quality(rows: list[dict], dominant_bias: str) -> tuple[str, str]:
        if dominant_bias not in {"UPSIDE", "DOWNSIDE"}:
            return "Neutral", WARNING
        confidences = [
            float(row.get("confidence", 0.0) or 0.0)
            for row in rows
            if str(row.get("direction", "")).upper() == dominant_bias
        ]
        if not confidences:
            return "No confirming TFs", WARNING
        avg_confidence = sum(confidences) / len(confidences)
        bucket = confidence_bucket(avg_confidence)
        color = POSITIVE if bucket == "HIGH" else (WARNING if bucket == "MEDIUM" else NEGATIVE)
        return f"{avg_confidence:.0f}% ({bucket.title()})", color

    def _directional_alignment_copy(pct: float, scope_label: str, bias: str, row_count: int) -> str:
        pct_txt = f"{pct:.0f}%"
        if row_count <= 0:
            return f"No usable {scope_label.lower()} data."
        if bias == "NEUTRAL" and pct <= 0:
            return f"No directional consensus on {scope_label.lower()}."
        if bias == "NEUTRAL":
            return f"{pct_txt} directional alignment on {scope_label.lower()}."
        return f"{pct_txt} directional alignment on {scope_label.lower()}."

    def _kpi_card(label: str, value: str, subtext: str, value_color: str, label_title: str) -> str:
        title_attr = f" title='{html.escape(label_title)}'" if label_title else ""
        return (
            "<div class='mtf-card'>"
            f"<div class='mtf-label'{title_attr}>{html.escape(label)}</div>"
            f"<div class='mtf-value' style='color:{value_color};'>{html.escape(value)}</div>"
            f"<div class='mtf-sub'>{html.escape(subtext)}</div>"
            "</div>"
        )

    def _pill(text: str, color: str, title: str) -> str:
        title_attr = f" title='{html.escape(title)}'" if title else ""
        return (
            f"<span class='mtf-pill' style='color:{color}; border-color:{color};'{title_attr}>"
            f"{html.escape(text)}</span>"
        )

    def _advanced_help_items(columns: list[str]) -> list[str]:
        help_map = {
            "ADX": "Trend strength only. It does not tell direction by itself.",
            "SuperTrend": "Trend-side read from the SuperTrend overlay.",
            "Ichimoku": "Cloud-based trend context for that timeframe.",
            "VWAP": "Price relative to VWAP on that timeframe: Above, Below, or Near VWAP.",
            "Spike Alert": "Flags unusual volume activity on the latest closed candle.",
            "Bollinger": "Where price sits versus the Bollinger Bands.",
            "Stochastic RSI": "Momentum location: high, low, or neutral.",
            "Volatility": "ATR-style volatility regime for that timeframe.",
            "PSAR": "Parabolic SAR trend-side confirmation.",
            "Williams %R": "Momentum location near top / bottom of range.",
            "CCI": "Momentum/mean-reversion pressure indicator.",
            "Candle Pattern": "Latest closed-candle pattern label.",
        }
        return [f"<b>{html.escape(col)}</b>: {html.escape(help_map[col])}" for col in columns if col in help_map]

    def _style_indicator(value: str) -> str:
        text = str(value or "").strip()
        upper = text.upper()
        if upper in {"", "NO DATA", "N/A"}:
            return f"color:{TEXT_MUTED}; font-weight:600;"
        if upper.startswith("▲") or upper.startswith("🔥"):
            return f"color:{POSITIVE}; font-weight:700;"
        if upper.startswith("▼"):
            return f"color:{NEGATIVE}; font-weight:700;"
        if upper.startswith("-"):
            return f"color:{TEXT_MUTED}; font-weight:600;"
        if any(token in upper for token in {"UPSIDE", "BULLISH", "STRONG", "VERY STRONG", "EXTREME"}):
            return f"color:{POSITIVE}; font-weight:700;"
        if any(token in upper for token in {"DOWNSIDE", "BEARISH", "WEAK", "VERY LOW", "LOW"}):
            return f"color:{NEGATIVE}; font-weight:700;"
        if any(token in upper for token in {"NEUTRAL", "MODERATE", "STARTING", "MIXED", "INDECISION", "SPIKE"}):
            return f"color:{TEXT_MUTED}; font-weight:600;"
        return f"color:{WARNING}; font-weight:700;"

    def _style_layer(value: str) -> str:
        layer = str(value or "").strip().lower()
        if layer == "structure":
            return f"color:{ACCENT}; font-weight:700;"
        if layer == "timing":
            return f"color:{TEXT_MUTED}; font-weight:700;"
        return ""

    def _style_delta(value: str) -> str:
        text = str(value or "").strip()
        if not text or text == "—":
            return f"color:{TEXT_MUTED}; font-weight:600;"
        if text.startswith("▲"):
            return f"color:{POSITIVE}; font-weight:700;"
        if text.startswith("▼"):
            return f"color:{NEGATIVE}; font-weight:700;"
        return f"color:{WARNING}; font-weight:700;"

    def _style_setup_confirm(value: str) -> str:
        cls = normalize_action_class(str(value or ""))
        if cls.startswith("ENTER_"):
            return f"color:{POSITIVE}; font-weight:700;"
        if cls == "PROBE":
            return f"color:{WARNING}; font-weight:700;"
        if cls == "WATCH":
            return "color:#7DD3FC; font-weight:700;"
        if cls == "SKIP":
            return f"color:{NEGATIVE}; font-weight:700;"
        return f"color:{TEXT_MUTED}; font-weight:600;"

    def _insight(metrics: dict) -> tuple[str, str, str]:
        dominant = metrics["dominant_bias"]
        higher = metrics["higher_tf_bias"]
        tactical = metrics["tactical_bias"]
        weighted = metrics["weighted_alignment_pct"]
        if dominant == "NEUTRAL":
            return (
                "Alignment Read · Mixed",
                "Higher and shorter timeframes are not lining up cleanly. Use this as a structure check, not an execution prompt.",
                WARNING,
            )
        if higher == dominant and tactical == dominant and weighted >= 65:
            return (
                f"Alignment Read · {direction_label(dominant)} Agreement",
                f"Higher timeframes and short-term timing are both pointing {direction_label(dominant).lower()}. That supports the same-side read you already saw in Market, Spot, or Position.",
                POSITIVE if dominant == "UPSIDE" else NEGATIVE,
            )
        if higher == dominant and tactical != dominant:
            return (
                "Alignment Read · Structure Leads, Timing Mixed",
                f"The higher-timeframe structure still leans {direction_label(dominant).lower()}, but short-term timing is mixed. Wait for timing to catch up before trusting execution.",
                WARNING,
            )
        return (
            "Alignment Read · Timeframes Mismatch",
            "Higher and lower timeframes are pulling in different directions. Treat this coin as tactically noisy until the picture simplifies.",
            WARNING,
        )

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
    confidence_calibration_model = build_confidence_calibration_model(adaptive_history_df)
    setup_calibration_model = build_setup_calibration_model(adaptive_history_df)
    ai_confidence_calibration_model = build_ai_confidence_calibration_model(adaptive_history_df)

    def _build_rows(coin: str) -> list[dict]:
        rows: list[dict] = []
        for timeframe in TF_SEQUENCE:
            df = fetch_ohlcv(coin, timeframe, limit=500)
            if df is None or len(df) < 55:
                rows.append(
                    {
                        "timeframe": timeframe,
                        "Timeframe": timeframe,
                        "Layer": "Timing" if timeframe in {"5m", "15m"} else "Structure",
                        "direction": "",
                        "Delta": "—",
                        "Setup Confirm": "N/A",
                        "Direction": "No Data",
                        "confidence": 0.0,
                        "Confidence": "",
                        "AI Ensemble": "N/A",
                        "AI Confidence": "",
                        "ADX": "",
                        "SuperTrend": "",
                        "Ichimoku": "",
                        "VWAP": "",
                        "Spike Alert": "",
                        "PSAR": "",
                        "Stochastic RSI": "",
                        "Williams %R": "",
                        "CCI": "",
                        "Candle Pattern": "",
                        "Bollinger": "",
                        "Volatility": "",
                        "Weight": TF_WEIGHTS.get(timeframe, 1.0),
                    }
                )
                continue
            df_eval = _prepare_closed_frame(df, min_rows=55)
            if df_eval is None:
                rows.append(
                    {
                        "timeframe": timeframe,
                        "Timeframe": timeframe,
                        "Layer": "Timing" if timeframe in {"5m", "15m"} else "Structure",
                        "direction": "",
                        "Delta": "—",
                        "Setup Confirm": "N/A",
                        "Direction": "No Data",
                        "confidence": 0.0,
                        "Confidence": "",
                        "AI Ensemble": "N/A",
                        "AI Confidence": "",
                        "ADX": "",
                        "SuperTrend": "",
                        "Ichimoku": "",
                        "VWAP": "",
                        "Spike Alert": "",
                        "PSAR": "",
                        "Stochastic RSI": "",
                        "Williams %R": "",
                        "CCI": "",
                        "Candle Pattern": "",
                        "Bollinger": "",
                        "Volatility": "",
                        "Weight": TF_WEIGHTS.get(timeframe, 1.0),
                    }
                )
                continue
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
            analysis = pipeline.analysis
            spot_snapshot = pipeline.spot_snapshot
            confidence_snapshot = pipeline.confidence_snapshot
            ai_spot_snapshot = pipeline.ai_spot_snapshot
            ai_spot_direction_key = pipeline.ai_spot_direction
            ai_spot_votes = int(pipeline.ai_spot_votes)
            try:
                price_change = ((float(df_eval["close"].iloc[-1]) / float(df_eval["close"].iloc[-2])) - 1.0) * 100.0
            except Exception:
                price_change = None
            ai_fallback_note = _ai_fallback_note(pipeline.ai_spot_status)
            ai_display = _spot_ai_display_value(ai_spot_direction_key, ai_spot_votes, ai_fallback_note)
            spike_label = ""
            if bool(analysis.volume_spike):
                try:
                    last_open = float(df_eval["open"].iloc[-1])
                    last_close = float(df_eval["close"].iloc[-1])
                    if pd.notna(last_open) and pd.notna(last_close) and last_close > last_open:
                        spike_label = "Up Spike"
                    elif pd.notna(last_open) and pd.notna(last_close) and last_close < last_open:
                        spike_label = "Down Spike"
                    else:
                        spike_label = "Spike"
                except Exception:
                    spike_label = "Spike"
            action_raw, action_reason_code = pipeline.action_raw, pipeline.action_reason_code
            setup_calibration_snapshot = build_setup_calibration_snapshot(
                setup_calibration_model,
                signal={
                    "Setup Confirm": str(action_raw or ""),
                    "AI Alignment": (
                        "Aligned"
                        if direction_key(spot_snapshot.direction) == ai_spot_direction_key
                        else "Not aligned"
                    ),
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
            ai_confidence_calibration_snapshot = build_ai_confidence_calibration_snapshot(
                ai_confidence_calibration_model,
                signal={
                    "Setup Confirm": str(action_raw or ""),
                    "AI Alignment": (
                        "Aligned"
                        if direction_key(spot_snapshot.direction) == ai_spot_direction_key
                        else "Not aligned"
                    ),
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
            rows.append(
                {
                    "timeframe": timeframe,
                    "Timeframe": timeframe,
                    "Layer": "Timing" if timeframe in {"5m", "15m"} else "Structure",
                    "direction": str(spot_snapshot.direction or "NEUTRAL"),
                    "Delta": format_delta(price_change) if price_change is not None else "—",
                    "Setup Confirm": _setup_confirm_display(
                        action_raw,
                        action_reason_code,
                        direction=str(spot_snapshot.direction or ""),
                    ),
                    "Direction": direction_label(spot_snapshot.direction),
                    "confidence": float(confidence_snapshot.score),
                    "Confidence": _shared_spot_confidence_display(float(confidence_snapshot.score)),
                    "AI Ensemble": ai_display,
                    "AI Confidence": _shared_ai_confidence_display(
                        ai_spot_snapshot,
                        float(ai_confidence_snapshot.score),
                    ),
                    "Spot Bias": direction_label(spot_snapshot.direction),
                    "Spot Confidence": f"{float(confidence_snapshot.score):.0f}%",
                    "ADX": normalize_indicator_label(_adx_label(analysis.adx), name="ADX"),
                    "SuperTrend": normalize_indicator_label(analysis.supertrend, name="SuperTrend"),
                    "Ichimoku": normalize_indicator_label(analysis.ichimoku, name="Ichimoku"),
                    "VWAP": normalize_indicator_label(analysis.vwap, name="VWAP"),
                    "Spike Alert": normalize_indicator_label(spike_label, name="Spike Alert"),
                    "PSAR": normalize_indicator_label(analysis.psar, name="PSAR"),
                    "Stochastic RSI": normalize_indicator_label(
                        format_stochrsi(analysis.stochrsi_k, timeframe=timeframe),
                        name="Stochastic RSI",
                    ),
                    "Williams %R": normalize_indicator_label(analysis.williams, name="Williams %R"),
                    "CCI": normalize_indicator_label(analysis.cci, name="CCI"),
                    "Candle Pattern": normalize_indicator_label(
                        str(analysis.candle_pattern or ""),
                        name="Candle Pattern",
                    ),
                    "Bollinger": normalize_indicator_label(analysis.bollinger, name="Bollinger"),
                    "Volatility": normalize_indicator_label(analysis.atr_comment, name="Volatility"),
                    "Weight": TF_WEIGHTS.get(timeframe, 1.0),
                }
            )
        return rows

    render_page_header(
        st,
        title="Multi-Timeframe Alignment",
        intro_html=copy_text("multitf.intro_html"),
    )
    st.markdown(
        f"""
        <style>
        .mtf-grid {{
            display:grid;
            grid-template-columns:repeat(4,minmax(0,1fr));
            gap:12px;
            margin:12px 0 14px 0;
        }}
        .mtf-card {{
            border:1px solid rgba(0,212,255,0.16);
            border-radius:14px;
            padding:14px 16px;
            background:linear-gradient(140deg, rgba(0,0,0,0.76), rgba(8,18,30,0.9));
        }}
        .mtf-label {{
            color:{TEXT_MUTED};
            font-size:0.72rem;
            text-transform:uppercase;
            letter-spacing:0.9px;
        }}
        .mtf-value {{
            font-size:1.28rem;
            font-weight:800;
            margin-top:6px;
        }}
        .mtf-sub {{
            color:{TEXT_MUTED};
            font-size:0.84rem;
            margin-top:6px;
            line-height:1.55;
        }}
        .mtf-pill {{
            display:inline-flex;
            align-items:center;
            gap:7px;
            padding:3px 10px;
            border-radius:999px;
            border:1px solid rgba(255,255,255,0.16);
            background:rgba(0,0,0,0.28);
            font-size:0.74rem;
            font-weight:700;
            margin-right:8px;
        }}
        @media (max-width: 1100px) {{
            .mtf-grid {{
                grid-template-columns:repeat(2,minmax(0,1fr));
            }}
        }}
        @media (max-width: 720px) {{
            .mtf-grid {{
                grid-template-columns:minmax(0,1fr);
            }}
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )

    coin = _normalize_coin_input(st.text_input("Coin (e.g. BTC, ETH, TAO)", value="BTC", key="mtf_coin_input"))
    state_key = "mtf_alignment_payload"
    sig_key = "mtf_alignment_sig"
    validation_error = _validate_coin_symbol(coin)
    if validation_error:
        st.error(validation_error)
        payload = st.session_state.get(state_key)
        if not payload:
            return
    else:
        current_sig = (coin,)
        if st.session_state.get(sig_key) != current_sig:
            with st.spinner("Checking alignment across all timeframes..."):
                df_spot_4h = fetch_ohlcv(coin, "4h", limit=260)
                df_spot_1d = fetch_ohlcv(coin, "1d", limit=260)
                spot_snapshot = build_spot_direction_snapshot(
                    df_4h=_prepare_closed_frame(df_spot_4h, min_rows=81),
                    df_1d=_prepare_closed_frame(df_spot_1d, min_rows=81),
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
                live_rows = _build_rows(coin)
                valid_live = [row for row in live_rows if row["direction"] in {"UPSIDE", "DOWNSIDE", "NEUTRAL"}]
                rows = live_rows
                from_cache = False
                cache_ts = None
                if not valid_live:
                    rows, from_cache, cache_ts = live_or_snapshot(
                        st,
                        f"mtf_rows::{coin}",
                        live_rows,
                        max_age_sec=900,
                        current_sig=current_sig,
                    )
                else:
                    live_or_snapshot(
                        st,
                        f"mtf_rows::{coin}",
                        live_rows,
                        max_age_sec=900,
                        current_sig=current_sig,
                    )
                metrics = compute_multitf_alignment(rows)
                st.session_state[state_key] = {
                    "coin": coin,
                    "rows": rows,
                    "metrics": metrics,
                    "spot_direction": str(spot_snapshot.direction),
                    "spot_confidence": float(confidence_snapshot.score),
                    "from_cache": from_cache,
                    "cache_ts": cache_ts,
                }
                st.session_state[sig_key] = current_sig

    payload = st.session_state.get(state_key)
    if not payload:
        return

    rows = payload["rows"]
    metrics = payload["metrics"]
    spot_direction = str(payload.get("spot_direction") or "NEUTRAL")
    spot_confidence = float(payload.get("spot_confidence") or 0.0)

    if payload.get("from_cache"):
        st.warning(f"Live multi-timeframe data was unavailable. Showing cached snapshot from {payload.get('cache_ts')}.")
    elif metrics["coverage_count"] < metrics["coverage_total"]:
        st.warning(
            f"Only {metrics['coverage_count']}/{metrics['coverage_total']} timeframes produced usable closed-candle analysis. "
            f"Treat this as a partial alignment read."
        )
    title, body, title_color = _insight(metrics)
    st.markdown(
        f"<div class='panel-box' style='border-left:4px solid {title_color};'>"
        f"<div style='color:{title_color}; font-weight:800; font-size:1.02rem;'>{title}</div>"
        f"<div style='color:{TEXT_MUTED}; font-size:0.9rem; margin-top:6px; line-height:1.6;'>{body}</div>"
        f"<div style='color:{TEXT_MUTED}; font-size:0.82rem; margin-top:7px;'>Higher timeframes are weighted more heavily because they usually define the stronger structural regime.</div>"
        f"</div>",
        unsafe_allow_html=True,
    )

    dominant_color = POSITIVE if metrics["dominant_bias"] == "UPSIDE" else (NEGATIVE if metrics["dominant_bias"] == "DOWNSIDE" else WARNING)
    higher_color = POSITIVE if metrics["higher_tf_bias"] == "UPSIDE" else (NEGATIVE if metrics["higher_tf_bias"] == "DOWNSIDE" else WARNING)
    tactical_color = POSITIVE if metrics["tactical_bias"] == "UPSIDE" else (NEGATIVE if metrics["tactical_bias"] == "DOWNSIDE" else WARNING)
    coverage_color = POSITIVE if metrics["coverage_count"] >= 5 else (WARNING if metrics["coverage_count"] >= 3 else NEGATIVE)
    conviction_quality_label, conviction_quality_color = _conviction_quality(rows, metrics["dominant_bias"])
    overall_alignment_copy = _directional_alignment_copy(
        metrics["weighted_alignment_pct"],
        "all valid timeframes",
        metrics["dominant_bias"],
        metrics["coverage_count"],
    )
    higher_bias_copy = summarize_scope_bias(rows, HIGHER_TFS, "higher timeframes", metrics["higher_tf_bias"])

    st.markdown(
        "<div class='mtf-grid'>"
        + _kpi_card(
            "Dominant Bias",
            direction_label(metrics["dominant_bias"]),
            "Overall direction across usable timeframes.",
            dominant_color,
            "The direction carrying the strongest weighted support across all valid timeframes.",
        )
        + _kpi_card(
            "Directional Alignment",
            f"{metrics['weighted_alignment_pct']:.0f}%",
            f"{overall_alignment_copy} {metrics['alignment_read']} direction-only read after higher-timeframe weights are applied.",
            dominant_color,
            "Measures weighted agreement of non-neutral direction only. Neutral rows do not count as directional agreement.",
        )
        + _kpi_card(
            "Higher-TF Bias",
            direction_label(metrics["higher_tf_bias"]),
            higher_bias_copy,
            higher_color,
            "Direction read using only 1h, 4h, and 1d. This is the structural layer.",
        )
        + _kpi_card(
            "Coverage",
            f"{metrics['coverage_count']}/{metrics['coverage_total']}",
            f"{metrics['coverage_read']} data coverage across the five timeframes.",
            coverage_color,
            "How many of the five timeframes produced usable closed-candle analysis.",
        )
        + "</div>",
        unsafe_allow_html=True,
    )

    st.markdown(
        "<div class='panel-box' style='padding:12px 16px;'>"
        + _pill(
            f"Spot Bias · {direction_label(spot_direction)} · {spot_confidence:.0f}%",
            POSITIVE if spot_direction == "UPSIDE" else (NEGATIVE if spot_direction == "DOWNSIDE" else WARNING),
            "Higher-timeframe spot bias summary with confidence score.",
        )
        + _pill(
            f"Higher-TF · {direction_label(metrics['higher_tf_bias'])} · {metrics['higher_tf_read']}",
            higher_color,
            "Structure-only read from 1h / 4h / 1d.",
        )
        + _pill(
            f"Short-TF Timing · {direction_label(metrics['tactical_bias'])} · {metrics['tactical_read']}",
            tactical_color,
            "Timing read from 5m / 15m. Useful for entry timing, not structure by itself.",
        )
        + _pill(
            f"Confirming Confidence · {conviction_quality_label}",
            conviction_quality_color,
            "Average confidence of only the timeframes that support the current directional bias.",
        )
        + "</div>",
        unsafe_allow_html=True,
    )

    render_help_details(
        st,
        summary="How to use this tab (?)",
        body_html=copy_text("multitf.help.quick_html"),
    )

    show_advanced_columns = False
    base_columns = [
        "Timeframe",
        "Role",
        "Δ (%)",
        "Setup Confirm",
        "Direction",
        "Confidence",
        "AI Ensemble",
        "AI Confidence",
    ]
    advanced_columns = [
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
    advanced_column_groups = {
        "All": advanced_columns,
        "Trend": ["ADX", "SuperTrend", "Ichimoku", "VWAP", "PSAR"],
        "Momentum": ["Stochastic RSI", "Williams %R", "CCI", "Candle Pattern"],
        "Volatility & Volume": ["Spike Alert", "Bollinger", "Volatility"],
    }
    advanced_view = "All"
    with st.expander("Table Options & Export", expanded=False):
        st.markdown(copy_text("multitf.table.confirmation_note"))
        show_advanced_columns = st.toggle("Show advanced columns", value=False, key="mtf_show_advanced_columns")
        if show_advanced_columns:
            advanced_view = st.radio(
                "Advanced view",
                list(advanced_column_groups.keys()),
                index=0,
                horizontal=True,
                key="mtf_advanced_view",
            )
            render_help_details(
                st,
                summary="Column guide (?)",
                body_html=copy_text("multitf.table.column_guide_html"),
            )
            render_help_details(
                st,
                summary=f"{advanced_view} advanced help (?)",
                body_html="<br>".join(_advanced_help_items(advanced_column_groups.get(advanced_view, advanced_columns))),
            )
    table_rows = []
    export_rows = []
    for row in rows:
        table_row = {
            "Timeframe": row["Timeframe"],
            "Role": row["Layer"],
            "Δ (%)": row["Delta"],
            "Setup Confirm": row["Setup Confirm"],
            "Direction": row["Direction"],
            "Confidence": row["Confidence"],
            "AI Ensemble": row["AI Ensemble"],
            "AI Confidence": row["AI Confidence"],
        }
        if show_advanced_columns:
            table_row.update(
                {
                    "ADX": row["ADX"],
                    "SuperTrend": row["SuperTrend"],
                    "Ichimoku": row["Ichimoku"],
                    "VWAP": row["VWAP"],
                    "Spike Alert": row["Spike Alert"],
                    "Bollinger": row["Bollinger"],
                    "Stochastic RSI": row["Stochastic RSI"],
                    "Volatility": row["Volatility"],
                    "PSAR": row["PSAR"],
                    "Williams %R": row["Williams %R"],
                    "CCI": row["CCI"],
                    "Candle Pattern": row["Candle Pattern"],
                }
            )
        table_rows.append(table_row)
        export_rows.append(
            {
                "Timeframe": row["Timeframe"],
                "Role": row["Layer"],
                "Δ (%)": row["Delta"],
                "Setup Confirm": row["Setup Confirm"],
                "Direction": row["Direction"],
                "Confidence": row["Confidence"],
                "AI Ensemble": row["AI Ensemble"],
                "AI Confidence": row["AI Confidence"],
                "ADX": row["ADX"],
                "SuperTrend": row["SuperTrend"],
                "Ichimoku": row["Ichimoku"],
                "VWAP": row["VWAP"],
                "Spike Alert": row["Spike Alert"],
                "Bollinger": row["Bollinger"],
                "Stochastic RSI": row["Stochastic RSI"],
                "Volatility": row["Volatility"],
                "PSAR": row["PSAR"],
                "Williams %R": row["Williams %R"],
                "CCI": row["CCI"],
                "Candle Pattern": row["Candle Pattern"],
            }
        )
    visible_advanced_columns = advanced_column_groups.get(advanced_view, advanced_columns) if show_advanced_columns else []
    display_columns = base_columns + visible_advanced_columns
    df_rows = pd.DataFrame(table_rows)
    df_rows = df_rows[[col for col in display_columns if col in df_rows.columns]]
    styled = (
        df_rows.style
        .map(_style_delta, subset=["Δ (%)"])
        .map(_style_setup_confirm, subset=["Setup Confirm"])
        .map(_style_indicator, subset=["Direction", "Confidence", "AI Ensemble", "AI Confidence"])
        .map(_style_layer, subset=["Role"])
    )
    if show_advanced_columns:
        if "ADX" in df_rows.columns:
            styled = styled.map(_style_indicator, subset=["ADX"])
        trend_cols = [col for col in ["SuperTrend", "Ichimoku", "VWAP", "PSAR"] if col in df_rows.columns]
        if trend_cols:
            styled = styled.map(_style_indicator, subset=trend_cols)
        if "Spike Alert" in df_rows.columns:
            styled = styled.map(_style_indicator, subset=["Spike Alert"])
        momentum_cols = [col for col in ["Bollinger", "Stochastic RSI", "Williams %R", "CCI"] if col in df_rows.columns]
        if momentum_cols:
            styled = styled.map(_style_indicator, subset=momentum_cols)
        if "Volatility" in df_rows.columns:
            styled = styled.map(_style_indicator, subset=["Volatility"])
        if "Candle Pattern" in df_rows.columns:
            styled = styled.map(_style_indicator, subset=["Candle Pattern"])
    styled = styled.hide(axis="index")
    st.dataframe(styled, width="stretch", hide_index=True)

    export_df = pd.DataFrame(export_rows)
    export_df = export_df[[col for col in base_columns + advanced_columns if col in export_df.columns]]
    export_df["Dominant Bias"] = direction_label(metrics["dominant_bias"])
    export_df["Spot Bias"] = direction_label(spot_direction)
    export_df["Spot Confidence %"] = round(float(spot_confidence), 2)
    export_df["Directional Alignment %"] = round(metrics["weighted_alignment_pct"], 2)
    export_df["Directional Alignment Summary"] = overall_alignment_copy
    export_df["Higher-TF Bias"] = direction_label(metrics["higher_tf_bias"])
    export_df["Higher-TF Alignment %"] = round(metrics["higher_tf_alignment_pct"], 2)
    export_df["Higher-TF Bias Summary"] = higher_bias_copy
    export_df["Short-TF Timing"] = direction_label(metrics["tactical_bias"])
    export_df["Short-TF Timing Read"] = metrics["tactical_read"]
    export_df["Confirming Confidence"] = conviction_quality_label
    export_df["Coverage Summary"] = f"{metrics['coverage_count']}/{metrics['coverage_total']} ({metrics['coverage_read']})"
    export_csv = export_df.to_csv(index=False).encode("utf-8")
    with st.expander("Export Snapshot", expanded=False):
        st.download_button(
            "Export Multi-TF Alignment (CSV)",
            data=export_csv,
            file_name=f"{payload['coin'].replace('/', '_')}_multitf_alignment.csv",
            mime="text/csv",
        )
