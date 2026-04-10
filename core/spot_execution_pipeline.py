"""Shared HTF-anchor + selected-timeframe execution pipeline for workspace tabs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import numpy as np
import pandas as pd

from core.adaptive_weighting import build_confidence_calibration_snapshot
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
)
from core.market_decision import (
    ai_led_confirmation_snapshot,
    ai_vote_metrics,
    selected_timeframe_execution_snapshot,
    selected_timeframe_rr_ratio,
    spot_action_decision_with_reason,
    structure_state,
    trend_led_confirmation_snapshot,
)
from core.signal_contract import bias_confidence_from_bias
from core.spot_direction import build_spot_direction_snapshot
from core.timeframe_anchors import choose_anchor_context


def prepare_closed_frame(df: pd.DataFrame | None, *, min_rows: int = 55) -> pd.DataFrame | None:
    if df is None:
        return None
    if len(df) <= int(min_rows):
        return None
    df_eval = df.iloc[:-1].copy()
    if len(df_eval) < int(min_rows):
        return None
    return df_eval


def direction_fetch_symbol(symbol: str, actual_symbol: str, source_provider: str) -> str:
    """Keep HTF context anchored to the requested symbol across provider variants."""
    del source_provider
    return str(symbol or actual_symbol or "").strip()


def _signal_plain(signal: str) -> str:
    s = str(signal or "").strip().upper()
    if s in {
        "STRONG BUY",
        "BUY",
        "LONG",
        "UPSIDE",
        "STRONG UPSIDE",
        "BULLISH",
        "STRONG BULLISH",
    }:
        return "LONG"
    if s in {
        "STRONG SELL",
        "SELL",
        "SHORT",
        "DOWNSIDE",
        "STRONG DOWNSIDE",
        "BEARISH",
        "STRONG BEARISH",
    }:
        return "SHORT"
    return "WAIT"


def _direction_key(direction: str) -> str:
    d = str(direction or "").strip().upper()
    if d in {"UPSIDE", "LONG", "BUY", "BULLISH", "STRONG BUY"}:
        return "UPSIDE"
    if d in {"DOWNSIDE", "SHORT", "SELL", "BEARISH", "STRONG SELL"}:
        return "DOWNSIDE"
    return "NEUTRAL"


@dataclass(frozen=True)
class SpotExecutionPipelineSnapshot:
    anchor_plan: Any
    analysis: Any
    spot_snapshot: Any
    confidence_snapshot: Any
    ai_spot_snapshot: Any
    ai_confidence_snapshot: Any
    signal_raw: str
    signal_direction: str
    signal_direction_legacy: str
    bias_score: float
    directional_confidence: float
    adx_val: float
    supertrend_trend: str
    ichimoku_trend: str
    stochrsi_k_val: float
    bollinger_bias: str
    vwap_label: str
    psar_trend: str
    williams_label: str
    cci_label: str
    selected_ai_direction: str
    selected_ai_votes: int
    selected_ai_directional_agreement: float
    selected_ai_consensus_agreement: float
    selected_ai_decision_agreement: float
    execution_confidence_snapshot: Any
    execution_conviction_label: str
    execution_snapshot: Any
    setup_rr_ratio: float
    trend_led_snapshot: Any
    ai_led_snapshot: Any
    ai_spot_direction: str
    ai_spot_votes: int
    ai_spot_agreement: float
    ai_spot_consensus: float
    ai_spot_probability_up: float
    ai_spot_status: str
    action_raw: str
    action_reason_code: str


def build_spot_execution_pipeline(
    *,
    symbol: str,
    actual_symbol: str,
    source_provider: str,
    timeframe: str,
    df_eval: pd.DataFrame,
    fetch_ohlcv: Callable[..., pd.DataFrame | None],
    analyse_fn: Callable[[pd.DataFrame], Any],
    predictor: Callable[[pd.DataFrame], tuple[Any, Any, dict | None]],
    conviction_fn: Callable[[str, str, float, float], tuple[str, Any]],
    confidence_calibration_model: dict[str, object] | None = None,
    scan_focus: str = "Unknown",
    htf_limit: int = 260,
    min_context_rows: int = 81,
) -> SpotExecutionPipelineSnapshot:
    symbol_for_context = direction_fetch_symbol(symbol, actual_symbol, source_provider)
    frame_cache: dict[str, pd.DataFrame | None] = {}

    def _fetch_anchor_frame(anchor_timeframe: str) -> pd.DataFrame | None:
        if anchor_timeframe in frame_cache:
            return frame_cache[anchor_timeframe]
        raw = fetch_ohlcv(symbol_for_context, anchor_timeframe, limit=htf_limit)
        prepared = prepare_closed_frame(raw, min_rows=min_context_rows)
        frame_cache[anchor_timeframe] = prepared
        return prepared

    anchor_plan, df_direction_lead, df_direction_confirm = choose_anchor_context(
        timeframe,
        _fetch_anchor_frame,
    )

    spot_snapshot = build_spot_direction_snapshot(
        df_4h=None,
        df_1d=None,
        lead_df=df_direction_lead,
        confirm_df=df_direction_confirm,
        lead_timeframe=anchor_plan.lead_timeframe,
        confirm_timeframe=anchor_plan.confirm_timeframe,
    )
    ai_spot_snapshot = build_ai_spot_bias_snapshot(
        df_4h=None,
        df_1d=None,
        lead_df=df_direction_lead,
        confirm_df=df_direction_confirm,
        lead_timeframe=anchor_plan.lead_timeframe,
        confirm_timeframe=anchor_plan.confirm_timeframe,
        predictor=predictor,
    )
    confidence_calibration_snapshot = build_confidence_calibration_snapshot(
        dict(confidence_calibration_model or {}),
        signal={
            "Direction": str(spot_snapshot.direction or ""),
            "AI Alignment": (
                "Aligned"
                if _direction_key(spot_snapshot.direction) in {"UPSIDE", "DOWNSIDE"}
                and _direction_key(spot_snapshot.direction) == _direction_key(ai_spot_snapshot.direction)
                else "Not aligned"
            ),
            "Timeframe": str(timeframe or "Unknown"),
            "Scan Focus": str(scan_focus or "Unknown"),
        },
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
        archive_calibration_delta=float(getattr(confidence_calibration_snapshot, "delta", 0.0) or 0.0),
        archive_calibration_note=str(getattr(confidence_calibration_snapshot, "note", "") or ""),
    )

    analysis = analyse_fn(df_eval)
    signal_raw = str(getattr(analysis, "signal", "") or "")
    bias_score = float(getattr(analysis, "bias", 0.0) or 0.0)
    directional_confidence = float(bias_confidence_from_bias(bias_score))
    adx_val = float(getattr(analysis, "adx", float("nan")) or float("nan"))
    supertrend_trend = str(getattr(analysis, "supertrend", "") or "")
    ichimoku_trend = str(getattr(analysis, "ichimoku", "") or "")
    stochrsi_k_val = float(getattr(analysis, "stochrsi_k", 0.0) or 0.0)
    bollinger_bias = str(getattr(analysis, "bollinger", "") or "")
    vwap_label = str(getattr(analysis, "vwap", "") or "")
    psar_trend = str(getattr(analysis, "psar", "") or "")
    williams_label = str(getattr(analysis, "williams", "") or "")
    cci_label = str(getattr(analysis, "cci", "") or "")

    signal_direction = _direction_key(_signal_plain(signal_raw))
    signal_direction_legacy = (
        "LONG" if signal_direction == "UPSIDE" else ("SHORT" if signal_direction == "DOWNSIDE" else "WAIT")
    )

    selected_ai_votes = 0
    try:
        _selected_ai_prob, selected_ai_raw, selected_ai_details = predictor(df_eval)
        selected_ai_direction = _direction_key(selected_ai_raw)
        selected_ai_directional_agreement = float(
            (selected_ai_details or {}).get(
                "directional_agreement",
                (selected_ai_details or {}).get("agreement", 0.0),
            )
        )
        selected_ai_consensus_agreement = float(
            (selected_ai_details or {}).get("consensus_agreement", selected_ai_directional_agreement)
        )
        selected_ai_votes, _, selected_ai_decision_agreement = ai_vote_metrics(
            selected_ai_direction,
            selected_ai_directional_agreement,
            selected_ai_consensus_agreement,
        )
    except Exception:
        selected_ai_direction = "NEUTRAL"
        selected_ai_directional_agreement = 0.0
        selected_ai_consensus_agreement = 0.0
        selected_ai_decision_agreement = 0.0

    base_conviction_lbl, _ = conviction_fn(
        signal_direction,
        selected_ai_direction,
        directional_confidence,
        selected_ai_decision_agreement,
    )
    tactical_structure = structure_state(
        signal_direction,
        selected_ai_direction,
        directional_confidence,
        selected_ai_decision_agreement,
    )
    execution_confidence_snapshot = build_execution_confidence_snapshot(
        direction=signal_direction,
        bias_score=bias_score,
        adx_val=adx_val if np.isfinite(adx_val) else float("nan"),
        structure_state=tactical_structure,
        conviction_label=str(base_conviction_lbl),
        ai_agreement=float(selected_ai_decision_agreement),
    )
    execution_conviction_label, _ = conviction_fn(
        signal_direction,
        selected_ai_direction,
        float(execution_confidence_snapshot.score),
        selected_ai_decision_agreement,
    )

    execution_snapshot = selected_timeframe_execution_snapshot(
        df=df_eval,
        direction=spot_snapshot.direction,
        bias_score=bias_score,
        adx_val=adx_val if np.isfinite(adx_val) else float("nan"),
        supertrend_trend=supertrend_trend,
        ichimoku_trend=ichimoku_trend,
        vwap_label=vwap_label,
        psar_trend=psar_trend,
        bollinger_bias=bollinger_bias,
        williams_label=williams_label,
        cci_label=cci_label,
    )
    setup_rr_ratio = float(selected_timeframe_rr_ratio(execution_snapshot, direction=spot_snapshot.direction))

    trend_led_snapshot = trend_led_confirmation_snapshot(
        spot_dir=spot_snapshot.direction,
        spot_confidence=float(confidence_snapshot.score),
        tactical_dir=signal_direction,
        adx_val=adx_val if np.isfinite(adx_val) else float("nan"),
        structure_quality=float(execution_snapshot.structure_quality),
        trend_quality=float(execution_snapshot.trend_quality),
        regime_quality=float(execution_snapshot.regime_quality),
        location_quality=float(execution_snapshot.location_quality),
        rr_ratio=setup_rr_ratio if np.isfinite(setup_rr_ratio) and setup_rr_ratio > 0.0 else None,
    )

    ai_spot_direction = _direction_key(ai_spot_snapshot.direction)
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
    ai_led_snapshot = ai_led_confirmation_snapshot(
        spot_dir=spot_snapshot.direction,
        spot_confidence=float(confidence_snapshot.score),
        ai_dir=ai_spot_direction,
        ai_probability=float(ai_spot_probability_up),
        directional_agreement=float(ai_spot_agreement),
        consensus_agreement=float(ai_spot_consensus),
        adx_val=adx_val if np.isfinite(adx_val) else float("nan"),
        location_quality=float(execution_snapshot.location_quality),
        rr_ratio=setup_rr_ratio if np.isfinite(setup_rr_ratio) and setup_rr_ratio > 0.0 else None,
        ai_status=ai_spot_status,
    )

    action_raw, action_reason_code = spot_action_decision_with_reason(
        spot_snapshot.direction,
        float(confidence_snapshot.score),
        signal_direction,
        ai_spot_snapshot.direction,
        ai_spot_agreement,
        adx_val if np.isfinite(adx_val) else float("nan"),
        trend_led_snapshot=trend_led_snapshot,
        ai_led_snapshot=ai_led_snapshot,
    )

    return SpotExecutionPipelineSnapshot(
        anchor_plan=anchor_plan,
        analysis=analysis,
        spot_snapshot=spot_snapshot,
        confidence_snapshot=confidence_snapshot,
        ai_spot_snapshot=ai_spot_snapshot,
        ai_confidence_snapshot=ai_confidence_snapshot,
        signal_raw=signal_raw,
        signal_direction=signal_direction,
        signal_direction_legacy=signal_direction_legacy,
        bias_score=bias_score,
        directional_confidence=directional_confidence,
        adx_val=adx_val,
        supertrend_trend=supertrend_trend,
        ichimoku_trend=ichimoku_trend,
        stochrsi_k_val=stochrsi_k_val,
        bollinger_bias=bollinger_bias,
        vwap_label=vwap_label,
        psar_trend=psar_trend,
        williams_label=williams_label,
        cci_label=cci_label,
        selected_ai_direction=selected_ai_direction,
        selected_ai_votes=selected_ai_votes,
        selected_ai_directional_agreement=selected_ai_directional_agreement,
        selected_ai_consensus_agreement=selected_ai_consensus_agreement,
        selected_ai_decision_agreement=selected_ai_decision_agreement,
        execution_confidence_snapshot=execution_confidence_snapshot,
        execution_conviction_label=execution_conviction_label,
        execution_snapshot=execution_snapshot,
        setup_rr_ratio=setup_rr_ratio,
        trend_led_snapshot=trend_led_snapshot,
        ai_led_snapshot=ai_led_snapshot,
        ai_spot_direction=ai_spot_direction,
        ai_spot_votes=ai_spot_votes,
        ai_spot_agreement=ai_spot_agreement,
        ai_spot_consensus=ai_spot_consensus,
        ai_spot_probability_up=ai_spot_probability_up,
        ai_spot_status=ai_spot_status,
        action_raw=action_raw,
        action_reason_code=action_reason_code,
    )
