from __future__ import annotations

from collections import Counter
from typing import Any, Callable, Protocol, Tuple

import numpy as np
import pandas as pd
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
    action_decision_with_reason,
    ai_led_confirmation_snapshot,
    ai_vote_metrics,
    normalize_action_class,
    selected_timeframe_execution_snapshot,
    selected_timeframe_rr_ratio,
    spot_action_decision_with_reason,
    structure_state,
    trend_led_confirmation_snapshot,
)
from core.scalping import scalp_gate_thresholds
from core.signal_contract import bias_confidence_from_bias
from core.spot_direction import build_spot_direction_snapshot
from core.trading_copy import setup_class_display


class AnalysisLike(Protocol):
    signal: str
    bias: float


class MLEnsembleLike(Protocol):
    def __call__(self, df: pd.DataFrame) -> tuple[float, str, dict]:
        ...


class ConvictionLike(Protocol):
    def __call__(
        self,
        signal_dir: str,
        ai_dir: str,
        confidence: float,
        ai_agreement: float = 0.0,
    ) -> tuple[str, Any]:
        ...


def _read_bias_like(result: AnalysisLike) -> float:
    """Read bias score from analysis result."""
    try:
        return float(getattr(result, "bias"))
    except Exception:
        return 50.0


def _normalize_direction_signal(raw_signal: object) -> str:
    """Map heterogeneous signal labels to LONG/SHORT/WAIT for backtest routing.

    Supports legacy BUY/SELL labels and current Upside/Downside wording.
    """
    s = str(raw_signal or "").strip().upper()
    if not s:
        return "WAIT"

    long_exact = {
        "BUY",
        "STRONG BUY",
        "LONG",
        "UPSIDE",
        "STRONG UPSIDE",
        "BULLISH",
        "STRONG BULLISH",
    }
    short_exact = {
        "SELL",
        "STRONG SELL",
        "SHORT",
        "DOWNSIDE",
        "STRONG DOWNSIDE",
        "BEARISH",
        "STRONG BEARISH",
    }
    if s in long_exact:
        return "LONG"
    if s in short_exact:
        return "SHORT"

    # Lenient fallback for prefixed/suffixed variants.
    if ("UPSIDE" in s or "BUY" in s) and "DOWNSIDE" not in s and "SELL" not in s:
        return "LONG"
    if ("DOWNSIDE" in s or "SELL" in s) and "UPSIDE" not in s and "BUY" not in s:
        return "SHORT"
    return "WAIT"


def _setup_confirm_label(action_class: str) -> str:
    return setup_class_display(action_class, audience="trader")


def _direction_label_from_key(value: str) -> str:
    key = str(value or "").strip().upper()
    if key == "UPSIDE":
        return "Upside"
    if key == "DOWNSIDE":
        return "Downside"
    return "Neutral"


def _timeframe_delta(timeframe: str | None) -> pd.Timedelta | None:
    tf = str(timeframe or "").strip().lower()
    mapping = {
        "1m": pd.Timedelta(minutes=1),
        "3m": pd.Timedelta(minutes=3),
        "5m": pd.Timedelta(minutes=5),
        "15m": pd.Timedelta(minutes=15),
        "1h": pd.Timedelta(hours=1),
        "4h": pd.Timedelta(hours=4),
        "1d": pd.Timedelta(days=1),
    }
    return mapping.get(tf)


def _event_close_time(timestamp: object, timeframe: str | None) -> pd.Timestamp | None:
    ts = pd.to_datetime(timestamp, utc=True, errors="coerce")
    delta = _timeframe_delta(timeframe)
    if pd.isna(ts) or delta is None:
        return None
    return ts + delta


def _closed_context_frame(
    df: pd.DataFrame | None,
    *,
    event_close_time: pd.Timestamp | None,
    timeframe: str,
) -> pd.DataFrame | None:
    if df is None or event_close_time is None or "timestamp" not in df.columns:
        return None
    delta = _timeframe_delta(timeframe)
    if delta is None:
        return None
    out = df.copy()
    out["timestamp"] = pd.to_datetime(out["timestamp"], utc=True, errors="coerce")
    out = out.dropna(subset=["timestamp"])
    if out.empty:
        return None
    closed_mask = (out["timestamp"] + delta) <= event_close_time
    out = out.loc[closed_mask].copy()
    if len(out) < 80:
        return None
    return out.reset_index(drop=True)


def _resolve_setup_context(
    *,
    event_timestamp: object,
    timeframe: str | None,
    df_slice: pd.DataFrame | None,
    df_4h: pd.DataFrame | None,
    df_1d: pd.DataFrame | None,
    analysis_obj: AnalysisLike | None,
    tactical_direction: str,
    ai_key: str,
    agreement: float,
    adx_val: float,
    conviction_label: str,
    directional_confidence: float,
    bias_score: float,
    rr_ratio: float | None = None,
    ai_probability: float = float("nan"),
    directional_agreement: float = 0.0,
    consensus_agreement: float = 0.0,
    ai_status: str = "",
) -> dict[str, object]:
    event_close = _event_close_time(event_timestamp, timeframe)
    closed_4h = _closed_context_frame(df_4h, event_close_time=event_close, timeframe="4h")
    closed_1d = _closed_context_frame(df_1d, event_close_time=event_close, timeframe="1d")

    if closed_4h is not None or closed_1d is not None:
        spot_snapshot = build_spot_direction_snapshot(df_4h=closed_4h, df_1d=closed_1d)
        ai_spot_snapshot = build_ai_spot_bias_snapshot(df_4h=closed_4h, df_1d=closed_1d)
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
        trend_led_snapshot = None
        ai_spot_direction = str(ai_spot_snapshot.direction)
        ai_spot_agreement = float(ai_spot_bias_directional_agreement(ai_spot_snapshot))
        ai_spot_consensus = float(ai_spot_bias_consensus_agreement(ai_spot_snapshot))
        ai_spot_probability = float(ai_spot_bias_probability_up(ai_spot_snapshot))
        ai_spot_status = str(ai_spot_bias_status(ai_spot_snapshot) or "")
        ai_spot_votes = int(ai_spot_bias_display_votes(ai_spot_snapshot))
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
        execution_snapshot = selected_timeframe_execution_snapshot(
            df=df_slice,
            direction=spot_snapshot.direction,
            bias_score=float(bias_score),
            adx_val=adx_val,
            supertrend_trend=str(getattr(analysis_obj, "supertrend", "") or ""),
            ichimoku_trend=str(getattr(analysis_obj, "ichimoku", "") or ""),
            vwap_label=str(getattr(analysis_obj, "vwap", "") or ""),
            psar_trend=str(getattr(analysis_obj, "psar", "") or ""),
            bollinger_bias=str(getattr(analysis_obj, "bollinger", "") or ""),
            williams_label=str(getattr(analysis_obj, "williams", "") or ""),
            cci_label=str(getattr(analysis_obj, "cci", "") or ""),
        )
        setup_rr_ratio = float(selected_timeframe_rr_ratio(execution_snapshot, direction=spot_snapshot.direction))
        trend_led_snapshot = trend_led_confirmation_snapshot(
            spot_dir=spot_snapshot.direction,
            spot_confidence=float(confidence_snapshot.score),
            tactical_dir=tactical_direction,
            adx_val=adx_val,
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
            ai_probability=float(ai_spot_probability),
            directional_agreement=float(ai_spot_agreement),
            consensus_agreement=float(ai_spot_consensus),
            adx_val=adx_val,
            location_quality=float(execution_snapshot.location_quality),
            rr_ratio=setup_rr_ratio if np.isfinite(setup_rr_ratio) and setup_rr_ratio > 0.0 else None,
            ai_status=ai_spot_status,
        )
        action_raw, action_reason = spot_action_decision_with_reason(
            spot_snapshot.direction,
            float(confidence_snapshot.score),
            tactical_direction,
            ai_spot_direction,
            float(ai_spot_agreement),
            adx_val,
            trend_led_snapshot=trend_led_snapshot,
            ai_led_snapshot=ai_led_snapshot,
        )
        return {
            "action_raw": action_raw,
            "action_reason": action_reason,
            "direction_key": str(spot_snapshot.direction),
            "confidence": float(confidence_snapshot.score),
            "spot_snapshot": spot_snapshot,
            "ai_spot_snapshot": ai_spot_snapshot,
            "ai_direction_key": ai_spot_direction,
            "ai_votes": f"{ai_spot_votes}/3",
            "ai_confidence": float(ai_confidence_snapshot.score),
            "ai_timeframe_conflict": bool(ai_spot_snapshot.timeframe_conflict),
            "ai_degraded_data": bool(ai_spot_snapshot.degraded_data),
            "directional_confidence": float(directional_confidence),
            "bias": float(bias_score),
        }

    structure = structure_state(tactical_direction, ai_key, directional_confidence, float(agreement))
    action_raw, action_reason = action_decision_with_reason(
        tactical_direction,
        directional_confidence,
        structure,
        str(conviction_label or ""),
        float(agreement),
        adx_val,
    )
    return {
        "action_raw": action_raw,
        "action_reason": action_reason,
        "direction_key": str(tactical_direction),
        "confidence": float("nan"),
        "spot_snapshot": None,
        "ai_spot_snapshot": None,
        "ai_direction_key": str(ai_key),
        "ai_votes": f"{ai_vote_metrics(ai_key, directional_agreement, consensus_agreement)[0]}/3",
        "ai_confidence": float("nan"),
        "ai_timeframe_conflict": False,
        "ai_degraded_data": False,
        "directional_confidence": float(directional_confidence),
        "bias": float(bias_score),
    }


def _resolve_rr_ratio_from_plan(
    *,
    get_scalping_entry_target_fn: Callable[..., tuple] | None,
    sr_lookback_fn: Callable[[str | None], int] | None,
    df_slice: pd.DataFrame,
    analysis_obj: AnalysisLike,
    bias_score: float,
) -> float | None:
    if not callable(get_scalping_entry_target_fn):
        return None
    try:
        try:
            if callable(sr_lookback_fn):
                _scalp_direction, _entry, _target, _stop, rr_ratio, _breakout_note = get_scalping_entry_target_fn(
                    df_slice,
                    bias_score,
                    getattr(analysis_obj, "supertrend", ""),
                    getattr(analysis_obj, "ichimoku", ""),
                    getattr(analysis_obj, "vwap", ""),
                    sr_lookback_fn=sr_lookback_fn,
                )
            else:
                _scalp_direction, _entry, _target, _stop, rr_ratio, _breakout_note = get_scalping_entry_target_fn(
                    df_slice,
                    bias_score,
                    getattr(analysis_obj, "supertrend", ""),
                    getattr(analysis_obj, "ichimoku", ""),
                    getattr(analysis_obj, "vwap", ""),
                )
        except TypeError as te:
            if "sr_lookback_fn" not in str(te):
                raise
            _scalp_direction, _entry, _target, _stop, rr_ratio, _breakout_note = get_scalping_entry_target_fn(
                df_slice,
                bias_score,
                getattr(analysis_obj, "supertrend", ""),
                getattr(analysis_obj, "ichimoku", ""),
                getattr(analysis_obj, "vwap", ""),
            )
        rr_f = float(rr_ratio)
        if np.isfinite(rr_f) and rr_f > 0.0:
            return rr_f
    except Exception:
        return None
    return None


def _infer_regime(df_slice: pd.DataFrame, analysis_obj: AnalysisLike) -> tuple[str, float]:
    """Classify local market regime at entry time.

    Priority:
    1) ADX from analysis object (if available)
    2) Price-structure fallback using drift/noise ratio
    """
    adx_raw = getattr(analysis_obj, "adx", np.nan)
    try:
        adx_val = float(adx_raw)
    except Exception:
        adx_val = np.nan

    if np.isfinite(adx_val):
        if adx_val >= 25:
            return "TREND", float(np.clip((adx_val - 20.0) * 4.0 + 60.0, 0.0, 100.0))
        if adx_val <= 18:
            return "RANGE", float(np.clip(70.0 - (adx_val - 10.0) * 3.0, 0.0, 100.0))
        return "MIXED", 50.0

    close = pd.to_numeric(df_slice["close"], errors="coerce").dropna()
    if len(close) < 15:
        return "MIXED", 50.0

    lookback = close.iloc[-30:] if len(close) >= 30 else close
    rets = lookback.pct_change().dropna()
    if rets.empty:
        return "MIXED", 50.0

    drift = abs(float(lookback.iloc[-1] / lookback.iloc[0] - 1.0))
    noise = float(rets.std()) * np.sqrt(len(rets))
    trend_ratio = drift / (noise + 1e-9)

    if trend_ratio >= 1.2:
        return "TREND", float(np.clip(60.0 + (trend_ratio - 1.2) * 35.0, 0.0, 100.0))
    if trend_ratio <= 0.6:
        return "RANGE", float(np.clip(70.0 + (0.6 - trend_ratio) * 40.0, 0.0, 100.0))
    return "MIXED", 50.0


def run_setup_confirm_backtest(
    df: pd.DataFrame,
    analyzer: Callable[[pd.DataFrame], AnalysisLike],
    ml_predictor: MLEnsembleLike,
    conviction_fn: ConvictionLike,
    signal_plain_fn: Callable[[str], str],
    direction_key_fn: Callable[[str], str],
    *,
    timeframe: str | None = None,
    df_4h: pd.DataFrame | None = None,
    df_1d: pd.DataFrame | None = None,
    get_scalping_entry_target_fn: Callable[..., tuple] | None = None,
    sr_lookback_fn: Callable[[str | None], int] | None = None,
    exit_after: int = 5,
    commission: float = 0.001,
    slippage: float = 0.0005,
) -> pd.DataFrame:
    """Backtest Setup Confirm classes used by the market scanner.

    Trades are opened only when setup class is one of:
    - ENTER_TREND_AI
    - ENTER_TREND_LED
    - ENTER_AI_LED
    """
    exit_after = max(1, int(exit_after))
    commission = max(0.0, float(commission))
    slippage = max(0.0, float(slippage))

    results: list[dict] = []
    equity = 10000.0
    window_size = 200
    i = 55

    while i < len(df) - exit_after - 1:
        start_idx = max(0, i - window_size)
        df_slice = df.iloc[start_idx : i + 1]
        if len(df_slice) < 55:
            i += 1
            continue

        try:
            result = analyzer(df_slice)
            bias_score = _read_bias_like(result)
            directional_confidence = float(bias_confidence_from_bias(bias_score))
            _p, ai_direction, ai_details = ml_predictor(df_slice)
        except Exception:
            i += 1
            continue

        raw_signal = str(getattr(result, "signal", "") or "")
        signal_side = str(direction_key_fn(signal_plain_fn(raw_signal)))
        if signal_side not in {"UPSIDE", "DOWNSIDE"}:
            i += 1
            continue

        ai_key = str(direction_key_fn(ai_direction))
        agreement = float(ai_details.get("agreement", 0.0)) if isinstance(ai_details, dict) else 0.0
        directional_agreement = (
            float(ai_details.get("directional_agreement", agreement)) if isinstance(ai_details, dict) else agreement
        )
        consensus_agreement = (
            float(ai_details.get("consensus_agreement", 0.0)) if isinstance(ai_details, dict) else 0.0
        )
        ai_status = str(ai_details.get("status", "")) if isinstance(ai_details, dict) else ""
        votes, _, decision_agreement = ai_vote_metrics(
            ai_key,
            directional_agreement,
            consensus_agreement,
        )

        structure = structure_state(signal_side, ai_key, directional_confidence, float(decision_agreement))
        conviction_label, _ = conviction_fn(signal_side, ai_key, directional_confidence, float(decision_agreement))
        adx_raw = getattr(result, "adx", np.nan)
        try:
            adx_val = float(adx_raw)
        except Exception:
            adx_val = float("nan")
        execution_confidence = build_execution_confidence_snapshot(
            direction=signal_side,
            bias_score=bias_score,
            adx_val=adx_val,
            structure_state=structure,
            conviction_label=str(conviction_label),
            ai_agreement=float(decision_agreement),
        )
        rr_ratio = _resolve_rr_ratio_from_plan(
            get_scalping_entry_target_fn=get_scalping_entry_target_fn,
            sr_lookback_fn=sr_lookback_fn,
            df_slice=df_slice,
            analysis_obj=result,
            bias_score=bias_score,
        )

        setup_ctx = _resolve_setup_context(
            event_timestamp=df["timestamp"].iloc[i],
            timeframe=timeframe,
            df_slice=df_slice,
            df_4h=df_4h,
            df_1d=df_1d,
            analysis_obj=result,
            tactical_direction=signal_side,
            ai_key=ai_key,
            agreement=float(decision_agreement),
            ai_probability=float(_p),
            directional_agreement=float(directional_agreement),
            consensus_agreement=float(consensus_agreement),
            ai_status=ai_status,
            adx_val=adx_val,
            conviction_label=str(conviction_label),
            directional_confidence=directional_confidence,
            bias_score=bias_score,
            rr_ratio=rr_ratio,
        )
        action_raw = str(setup_ctx["action_raw"])
        action_reason = str(setup_ctx["action_reason"])
        action_class = normalize_action_class(action_raw)
        if action_class not in {"ENTER_TREND_AI", "ENTER_TREND_LED", "ENTER_AI_LED"}:
            i += 1
            continue

        entry_idx = i + 1
        exit_idx = entry_idx + exit_after
        if exit_idx >= len(df):
            break

        entry_open = float(df["open"].iloc[entry_idx]) if "open" in df.columns else float(df["close"].iloc[entry_idx])
        exit_open = float(df["open"].iloc[exit_idx]) if "open" in df.columns else float(df["close"].iloc[exit_idx])
        if entry_open <= 0 or exit_open <= 0:
            i += 1
            continue

        trade_direction = str(setup_ctx.get("direction_key") or signal_side)
        is_upside = trade_direction == "UPSIDE"
        if is_upside:
            entry_exec = entry_open * (1.0 + slippage)
            exit_exec = exit_open * (1.0 - slippage)
            gross_ret = (exit_exec - entry_exec) / entry_exec
        else:
            entry_exec = entry_open * (1.0 - slippage)
            exit_exec = exit_open * (1.0 + slippage)
            gross_ret = (entry_exec - exit_exec) / entry_exec

        net_ret = gross_ret - 2.0 * commission
        pnl = net_ret * 100.0
        equity = equity * (1.0 + (pnl / 100.0))
        regime, regime_score = _infer_regime(df_slice, result)

        results.append(
            {
                "Date": df["timestamp"].iloc[i],
                "Signal Time": df["timestamp"].iloc[i],
                "Entry Time": df["timestamp"].iloc[entry_idx],
                "Exit Time": df["timestamp"].iloc[exit_idx],
                "Setup Confirm": _setup_confirm_label(action_class),
                "Setup Class": action_class,
                "Action Reason": action_reason,
                "Direction": _direction_label_from_key(trade_direction),
                "AI Direction": _direction_label_from_key(str(setup_ctx.get("ai_direction_key") or ai_key)),
                "AI Votes": str(setup_ctx.get("ai_votes") or f"{votes}/3"),
                "Confidence": round(float(setup_ctx.get("confidence", float("nan"))), 1),
                "Selected-TF Confidence": round(float(execution_confidence.score), 1),
                "Bias": round(bias_score, 1),
                "Entry": entry_exec,
                "Exit": exit_exec,
                "PnL (%)": round(pnl, 2),
                "Equity": round(equity, 2),
                "Regime": regime,
                "Regime Score": round(regime_score, 1),
                "Holding Bars": int(exit_after),
            }
        )

        # Single-position mode: re-evaluate only after this trade exits.
        i = exit_idx

    return pd.DataFrame(results)


def summarize_setup_confirm_performance(df_results: pd.DataFrame) -> pd.DataFrame:
    """Summarize Setup Confirm performance by class."""
    if df_results is None or df_results.empty or "Setup Confirm" not in df_results.columns:
        return pd.DataFrame()

    d = df_results.copy()
    d["PnL (%)"] = pd.to_numeric(d.get("PnL (%)", pd.Series(dtype=float)), errors="coerce")
    d["is_win"] = (d["PnL (%)"] > 0).astype(int)
    grouped = (
        d.groupby("Setup Confirm", dropna=False)
        .agg(
            Trades=("PnL (%)", "count"),
            WinRate=("is_win", "mean"),
            AvgPnL=("PnL (%)", "mean"),
            MedianPnL=("PnL (%)", "median"),
            TotalPnL=("PnL (%)", "sum"),
            GrossProfit=("PnL (%)", lambda s: float(pd.to_numeric(s, errors="coerce")[pd.to_numeric(s, errors="coerce") > 0].sum())),
            GrossLoss=("PnL (%)", lambda s: abs(float(pd.to_numeric(s, errors="coerce")[pd.to_numeric(s, errors="coerce") <= 0].sum()))),
            AvgWin=("PnL (%)", lambda s: float(pd.to_numeric(s, errors="coerce")[pd.to_numeric(s, errors="coerce") > 0].mean()) if (pd.to_numeric(s, errors="coerce") > 0).any() else 0.0),
            AvgLoss=("PnL (%)", lambda s: abs(float(pd.to_numeric(s, errors="coerce")[pd.to_numeric(s, errors="coerce") <= 0].mean())) if (pd.to_numeric(s, errors="coerce") <= 0).any() else 0.0),
        )
        .reset_index()
    )
    grouped["WinRate"] = grouped["WinRate"] * 100.0
    grouped["Expectancy"] = grouped["AvgPnL"]
    grouped["PayoffRatio"] = grouped.apply(
        lambda r: (float(r["AvgWin"]) / float(r["AvgLoss"])) if float(r["AvgLoss"]) > 0 else (float("inf") if float(r["AvgWin"]) > 0 else 0.0),
        axis=1,
    )
    grouped["ProfitFactor"] = grouped.apply(
        lambda r: (float(r["GrossProfit"]) / float(r["GrossLoss"])) if float(r["GrossLoss"]) > 0 else float("inf"),
        axis=1,
    )
    grouped[["QualityGrade", "QualityNote"]] = grouped.apply(
        lambda r: pd.Series(
            grade_setup_class_quality(
                occurrences=float(r["Trades"]),
                expectancy=float(r["Expectancy"]),
                profit_factor=float(r["ProfitFactor"]),
                payoff_ratio=float(r["PayoffRatio"]),
                win_rate=float(r["WinRate"]),
            )
        ),
        axis=1,
    )
    grouped = grouped.sort_values(by=["Expectancy", "ProfitFactor", "Trades"], ascending=[False, False, False]).reset_index(drop=True)
    return grouped[[
        "Setup Confirm",
        "Trades",
        "WinRate",
        "Expectancy",
        "AvgWin",
        "AvgLoss",
        "PayoffRatio",
        "MedianPnL",
        "TotalPnL",
        "ProfitFactor",
        "QualityGrade",
        "QualityNote",
        "AvgPnL",
    ]]


def _allowed_setup_classes(setup_filter: str) -> set[str]:
    s = str(setup_filter or "").strip().upper()
    if s in {"TREND+AI", "ENTER_TREND_AI"}:
        return {"ENTER_TREND_AI"}
    if s in {"TREND-LED", "TREND_LED", "ENTER_TREND_LED"}:
        return {"ENTER_TREND_LED"}
    if s in {"AI-LED", "AI_LED", "ENTER_AI_LED"}:
        return {"ENTER_AI_LED"}
    return {"ENTER_TREND_AI", "ENTER_TREND_LED", "ENTER_AI_LED"}


def _return_distribution_metrics(values: pd.Series | pd.Index | list[float] | tuple[float, ...]) -> dict[str, float]:
    series = pd.to_numeric(pd.Series(values, dtype=float), errors="coerce").dropna()
    if series.empty:
        return {
            "win_rate": 0.0,
            "avg_win": 0.0,
            "avg_loss": 0.0,
            "expectancy": 0.0,
            "payoff_ratio": 0.0,
            "profit_factor": 0.0,
            "median": 0.0,
        }

    wins = series[series > 0]
    losses = series[series <= 0]
    win_rate = float((len(wins) / len(series)) * 100.0)
    avg_win = float(wins.mean()) if not wins.empty else 0.0
    avg_loss_abs = abs(float(losses.mean())) if not losses.empty else 0.0
    expectancy = float(series.mean())
    gross_profit = float(wins.sum()) if not wins.empty else 0.0
    gross_loss_abs = abs(float(losses.sum())) if not losses.empty else 0.0
    payoff_ratio = (avg_win / avg_loss_abs) if avg_loss_abs > 0 else (float("inf") if avg_win > 0 else 0.0)
    profit_factor = (gross_profit / gross_loss_abs) if gross_loss_abs > 0 else (float("inf") if gross_profit > 0 else 0.0)
    median = float(np.nanmedian(series.values)) if len(series) else 0.0
    return {
        "win_rate": win_rate,
        "avg_win": avg_win,
        "avg_loss": avg_loss_abs,
        "expectancy": expectancy,
        "payoff_ratio": payoff_ratio,
        "profit_factor": profit_factor,
        "median": median,
    }


def grade_setup_class_quality(
    *,
    occurrences: float,
    expectancy: float,
    profit_factor: float,
    payoff_ratio: float,
    win_rate: float,
) -> tuple[str, str]:
    occ = max(0.0, float(occurrences))
    exp = float(expectancy)
    pf = float(profit_factor)
    payoff = float(payoff_ratio)
    wr = float(win_rate)

    if occ < 8:
        return "Unrated", "Too few events to trust the class."

    elite = (
        occ >= 35
        and exp >= 0.75
        and pf >= 1.80
        and payoff >= 1.20
        and wr >= 52.0
    )
    if elite:
        return "Elite", "Strong expectancy with enough sample and robust payoff structure."

    validated = (
        occ >= 25
        and exp >= 0.30
        and pf >= 1.25
        and payoff >= 1.00
        and wr >= 48.0
    )
    if validated:
        return "Validated", "Positive edge is present and sample size is becoming credible."

    fragile = (
        occ >= 12
        and exp > 0.0
        and pf >= 1.00
        and payoff >= 0.90
        and wr >= 44.0
    )
    if fragile:
        return "Fragile", "Edge exists, but it is still thin or unstable."

    return "Weak", "Current sample does not show a robust class-level edge."


def _scalp_gate_thresholds(timeframe: str | None) -> tuple[float, float, float]:
    # Single source of truth shared with market/spot scalp gate policy.
    return scalp_gate_thresholds(timeframe)


def build_scalp_outcome_study(
    df: pd.DataFrame,
    analyzer: Callable[[pd.DataFrame], AnalysisLike],
    ml_predictor: MLEnsembleLike,
    conviction_fn: ConvictionLike,
    signal_plain_fn: Callable[[str], str],
    direction_key_fn: Callable[[str], str],
    get_scalping_entry_target_fn: Callable[..., tuple],
    scalp_quality_gate_fn: Callable[..., tuple[bool, str]],
    sr_lookback_fn: Callable[[str | None], int],
    *,
    timeframe: str = "1h",
    df_4h: pd.DataFrame | None = None,
    df_1d: pd.DataFrame | None = None,
    forward_bars: int = 10,
    window_size: int = 200,
    min_history: int = 55,
) -> pd.DataFrame:
    """Event-study engine for scalp opportunities generated by market scalp logic."""
    forward_bars = max(1, int(forward_bars))
    min_history = max(30, int(min_history))
    gate_min_rr, gate_min_adx, gate_min_confidence = _scalp_gate_thresholds(timeframe)

    rows: list[dict] = []
    diagnostics: dict[str, Any] = {
        "timeframe": str(timeframe),
        "forward_bars": int(forward_bars),
        "gate_thresholds": {
            "min_rr": float(gate_min_rr),
            "min_adx": float(gate_min_adx),
            "min_confidence": float(gate_min_confidence),
        },
        "bars_evaluated": 0,
        "analysis_fail": 0,
        "signal_side_reject": 0,
        "plan_fail": 0,
        "plan_fail_counts": {},
        "gate_reject_counts": {},
        "gate_pass_candidates": 0,
        "side_key_reject": 0,
        "price_level_reject": 0,
        "forward_window_reject": 0,
    }
    gate_reject_counter: Counter[str] = Counter()
    plan_fail_counter: Counter[str] = Counter()
    max_idx = len(df) - forward_bars - 1
    if max_idx <= min_history:
        out = pd.DataFrame()
        out.attrs["diagnostics"] = diagnostics
        return out

    high_col = "high" if "high" in df.columns else "close"
    low_col = "low" if "low" in df.columns else "close"

    for i in range(min_history, max_idx + 1):
        diagnostics["bars_evaluated"] += 1
        start_idx = max(0, i - max(60, int(window_size)))
        df_slice = df.iloc[start_idx : i + 1]
        if len(df_slice) < min_history:
            continue

        try:
            analysis = analyzer(df_slice)
            bias_score = _read_bias_like(analysis)
            directional_confidence = float(bias_confidence_from_bias(bias_score))
            _prob, ai_direction, ai_details = ml_predictor(df_slice)
        except Exception:
            diagnostics["analysis_fail"] += 1
            continue

        raw_signal = str(getattr(analysis, "signal", "") or "")
        signal_side = str(direction_key_fn(signal_plain_fn(raw_signal)))
        if signal_side not in {"UPSIDE", "DOWNSIDE"}:
            diagnostics["signal_side_reject"] += 1
            continue

        ai_key = str(direction_key_fn(ai_direction))
        agreement = float(ai_details.get("agreement", 0.0)) if isinstance(ai_details, dict) else 0.0
        directional_agreement = (
            float(ai_details.get("directional_agreement", agreement)) if isinstance(ai_details, dict) else agreement
        )
        consensus_agreement = (
            float(ai_details.get("consensus_agreement", 0.0)) if isinstance(ai_details, dict) else 0.0
        )
        ai_status = str(ai_details.get("status", "")) if isinstance(ai_details, dict) else ""
        votes, _, decision_agreement = ai_vote_metrics(
            ai_key,
            directional_agreement,
            consensus_agreement,
        )

        structure = structure_state(signal_side, ai_key, directional_confidence, float(decision_agreement))
        conviction_label, _ = conviction_fn(signal_side, ai_key, directional_confidence, float(decision_agreement))
        adx_raw = getattr(analysis, "adx", np.nan)
        try:
            adx_val = float(adx_raw)
        except Exception:
            adx_val = float("nan")
        execution_confidence = build_execution_confidence_snapshot(
            direction=signal_side,
            bias_score=bias_score,
            adx_val=adx_val,
            structure_state=structure,
            conviction_label=str(conviction_label),
            ai_agreement=float(decision_agreement),
        )
        execution_snapshot = selected_timeframe_execution_snapshot(
            df=df_slice,
            direction=signal_side,
            bias_score=float(bias_score),
            adx_val=adx_val,
            supertrend_trend=str(getattr(analysis, "supertrend", "") or ""),
            ichimoku_trend=str(getattr(analysis, "ichimoku", "") or ""),
            vwap_label=str(getattr(analysis, "vwap", "") or ""),
            psar_trend=str(getattr(analysis, "psar", "") or ""),
            bollinger_bias=str(getattr(analysis, "bollinger", "") or ""),
            williams_label=str(getattr(analysis, "williams", "") or ""),
            cci_label=str(getattr(analysis, "cci", "") or ""),
        )
        setup_rr_ratio = float(selected_timeframe_rr_ratio(execution_snapshot, direction=signal_side))
        trend_led_snapshot = trend_led_confirmation_snapshot(
            spot_dir=signal_side,
            spot_confidence=float(directional_confidence),
            tactical_dir=signal_side,
            adx_val=adx_val,
            structure_quality=float(execution_snapshot.structure_quality),
            trend_quality=float(execution_snapshot.trend_quality),
            regime_quality=float(execution_snapshot.regime_quality),
            location_quality=float(execution_snapshot.location_quality),
            rr_ratio=setup_rr_ratio if pd.notna(setup_rr_ratio) and setup_rr_ratio > 0.0 else None,
        )
        ai_led_snapshot = ai_led_confirmation_snapshot(
            spot_dir=signal_side,
            spot_confidence=float(directional_confidence),
            ai_dir=ai_key,
            ai_probability=float(_prob),
            directional_agreement=float(directional_agreement),
            consensus_agreement=float(consensus_agreement),
            adx_val=adx_val,
            location_quality=float(execution_snapshot.location_quality),
            rr_ratio=setup_rr_ratio if pd.notna(setup_rr_ratio) and setup_rr_ratio > 0.0 else None,
            ai_status=ai_status,
        )

        action_raw, _action_reason = action_decision_with_reason(
            signal_side,
            directional_confidence,
            structure,
            str(conviction_label),
            float(decision_agreement),
            adx_val,
        )
        action_class = normalize_action_class(action_raw)

        try:
            try:
                # Core engine signature supports explicit sr_lookback_fn.
                scalp_direction, entry, target, stop, rr_ratio, breakout_note = get_scalping_entry_target_fn(
                    df_slice,
                    bias_score,
                    getattr(analysis, "supertrend", ""),
                    getattr(analysis, "ichimoku", ""),
                    getattr(analysis, "vwap", ""),
                    sr_lookback_fn=sr_lookback_fn,
                    timeframe=timeframe,
                    execution_snapshot=execution_snapshot,
                    trend_led_snapshot=trend_led_snapshot,
                    ai_led_snapshot=ai_led_snapshot,
                    spot_direction=signal_side,
                    ai_direction=ai_key,
                )
            except TypeError as te:
                # Service-layer wrapper does not take sr_lookback_fn; retry without it.
                if "sr_lookback_fn" not in str(te) and "execution_snapshot" not in str(te):
                    raise
                scalp_direction, entry, target, stop, rr_ratio, breakout_note = get_scalping_entry_target_fn(
                    df_slice,
                    bias_score,
                    getattr(analysis, "supertrend", ""),
                    getattr(analysis, "ichimoku", ""),
                    getattr(analysis, "vwap", ""),
                    timeframe=timeframe,
                    execution_snapshot=execution_snapshot,
                    trend_led_snapshot=trend_led_snapshot,
                    ai_led_snapshot=ai_led_snapshot,
                    spot_direction=signal_side,
                    ai_direction=ai_key,
                )
        except Exception as plan_exc:
            diagnostics["plan_fail"] += 1
            plan_fail_counter[type(plan_exc).__name__] += 1
            continue

        gate_pass, gate_reason = scalp_quality_gate_fn(
            scalp_direction=scalp_direction,
            signal_direction=signal_side,
            rr_ratio=rr_ratio,
            adx_val=adx_val,
            confidence=float(execution_confidence.score),
            conviction_label=str(conviction_label),
            entry=entry,
            stop=stop,
            target=target,
            min_rr=gate_min_rr,
            min_adx=gate_min_adx,
            min_confidence=gate_min_confidence,
            timeframe=timeframe,
            setup_confirm=action_raw,
        )
        if not gate_pass:
            gate_reject_counter[str(gate_reason or "UNKNOWN_GATE_REJECT")] += 1
            continue

        diagnostics["gate_pass_candidates"] += 1
        setup_ctx = _resolve_setup_context(
            event_timestamp=df["timestamp"].iloc[i],
            timeframe=timeframe,
            df_slice=df_slice,
            df_4h=df_4h,
            df_1d=df_1d,
            analysis_obj=analysis,
            tactical_direction=signal_side,
            ai_key=ai_key,
            agreement=float(decision_agreement),
            ai_probability=float(_prob),
            directional_agreement=float(directional_agreement),
            consensus_agreement=float(consensus_agreement),
            ai_status=ai_status,
            adx_val=adx_val,
            conviction_label=str(conviction_label),
            directional_confidence=directional_confidence,
            bias_score=bias_score,
            rr_ratio=float(rr_ratio) if pd.notna(rr_ratio) else None,
        )
        side_key = str(direction_key_fn(str(scalp_direction)))
        if side_key not in {"UPSIDE", "DOWNSIDE"}:
            diagnostics["side_key_reject"] += 1
            continue

        try:
            event_price = float(entry)
            target_f = float(target)
            stop_f = float(stop)
            rr_f = float(rr_ratio)
        except Exception:
            diagnostics["price_level_reject"] += 1
            continue
        if not np.isfinite(event_price) or not np.isfinite(target_f) or not np.isfinite(stop_f):
            diagnostics["price_level_reject"] += 1
            continue
        if event_price <= 0 or target_f <= 0 or stop_f <= 0:
            diagnostics["price_level_reject"] += 1
            continue

        f_close = pd.to_numeric(df["close"].iloc[i + 1 : i + 1 + forward_bars], errors="coerce")
        f_high = pd.to_numeric(df[high_col].iloc[i + 1 : i + 1 + forward_bars], errors="coerce")
        f_low = pd.to_numeric(df[low_col].iloc[i + 1 : i + 1 + forward_bars], errors="coerce")
        if len(f_close) < forward_bars or f_close.isna().any():
            diagnostics["forward_window_reject"] += 1
            continue

        end_price = float(f_close.iloc[-1])
        end_ret_raw = ((end_price / event_price) - 1.0) * 100.0
        directional_end = end_ret_raw if side_key == "UPSIDE" else (-end_ret_raw)

        max_high = float(f_high.max()) if not f_high.empty and f_high.notna().any() else end_price
        min_low = float(f_low.min()) if not f_low.empty and f_low.notna().any() else end_price
        max_up_pct = ((max_high / event_price) - 1.0) * 100.0
        max_down_pct = ((min_low / event_price) - 1.0) * 100.0
        favorable_exc_pct = max(0.0, max_up_pct) if side_key == "UPSIDE" else max(0.0, -max_down_pct)
        adverse_exc_pct = max(0.0, -max_down_pct) if side_key == "UPSIDE" else max(0.0, max_up_pct)

        tp_hit_bar = None
        sl_hit_bar = None
        for step in range(1, forward_bars + 1):
            high_step = float(f_high.iloc[step - 1]) if pd.notna(f_high.iloc[step - 1]) else float("nan")
            low_step = float(f_low.iloc[step - 1]) if pd.notna(f_low.iloc[step - 1]) else float("nan")
            if not np.isfinite(high_step) or not np.isfinite(low_step):
                continue
            if side_key == "UPSIDE":
                tp_hit = high_step >= target_f
                sl_hit = low_step <= stop_f
            else:
                tp_hit = low_step <= target_f
                sl_hit = high_step >= stop_f
            if tp_hit and tp_hit_bar is None:
                tp_hit_bar = step
            if sl_hit and sl_hit_bar is None:
                sl_hit_bar = step

        tp_return = abs((target_f - event_price) / event_price) * 100.0
        sl_return = abs((event_price - stop_f) / event_price) * 100.0
        if tp_hit_bar is not None and sl_hit_bar is not None:
            if tp_hit_bar < sl_hit_bar:
                outcome = "TP"
                realized = tp_return
                hit_bar = tp_hit_bar
            elif sl_hit_bar < tp_hit_bar:
                outcome = "SL"
                realized = -sl_return
                hit_bar = sl_hit_bar
            else:
                outcome = "BOTH"
                realized = -sl_return
                hit_bar = tp_hit_bar
        elif tp_hit_bar is not None:
            outcome = "TP"
            realized = tp_return
            hit_bar = tp_hit_bar
        elif sl_hit_bar is not None:
            outcome = "SL"
            realized = -sl_return
            hit_bar = sl_hit_bar
        else:
            outcome = "TIMEOUT"
            realized = directional_end
            hit_bar = forward_bars

        realized_col = f"Realized Return @+{forward_bars} (%)"
        close_dir_col = f"Close Directional Return @+{forward_bars} (%)"
        row = {
            "Event Time": df["timestamp"].iloc[i],
            "Setup Confirm": _setup_confirm_label(action_class),
            "Setup Class": action_class,
            "Direction": _direction_label_from_key(side_key),
            "AI Direction": _direction_label_from_key(str(setup_ctx.get("ai_direction_key") or ai_key)),
            "AI Votes": str(setup_ctx.get("ai_votes") or f"{votes}/3"),
            "Confidence": round(float(setup_ctx.get("confidence", float(execution_confidence.score))), 1),
            "AI Confidence": round(float(setup_ctx.get("ai_confidence", float("nan"))), 1),
            "AI Timeframe Conflict": bool(setup_ctx.get("ai_timeframe_conflict", False)),
            "AI Data Partial": bool(setup_ctx.get("ai_degraded_data", False)),
            "Event Price": event_price,
            "Target": target_f,
            "Stop": stop_f,
            "R:R": rr_f,
            "Plan Check": str(gate_reason or "PASS"),
            "Breakout Note": str(breakout_note or "").strip(),
            "Outcome": outcome,
            "Hit Bar": int(hit_bar),
            f"End Price (+{forward_bars})": end_price,
            realized_col: float(realized),
            # Backward-compat alias for older consumers/tests.
            f"Return @+{forward_bars} (%)": float(realized),
            close_dir_col: float(directional_end),
            "Favorable Excursion (%)": float(favorable_exc_pct),
            "Adverse Excursion (%)": float(adverse_exc_pct),
        }

        for step in range(1, forward_bars + 1):
            p = float(f_close.iloc[step - 1])
            r = ((p / event_price) - 1.0) * 100.0
            row[f"Price +{step}"] = p
            row[f"Return +{step} (%)"] = r
            row[f"Directional Return +{step} (%)"] = r if side_key == "UPSIDE" else (-r)

        rows.append(row)

    out = pd.DataFrame(rows)
    diagnostics["plan_fail_counts"] = dict(plan_fail_counter)
    diagnostics["gate_reject_counts"] = dict(gate_reject_counter)
    diagnostics["events"] = int(len(out))
    out.attrs["diagnostics"] = diagnostics
    if out.empty:
        return out
    return out.sort_values("Event Time").reset_index(drop=True)


def summarize_scalp_outcome_study(df_events: pd.DataFrame, forward_bars: int) -> dict[str, float]:
    if df_events is None or df_events.empty:
        return {
            "occurrences": 0.0,
            "tp_rate": 0.0,
            "sl_rate": 0.0,
            "timeout_rate": 0.0,
            "avg_outcome": 0.0,
            "median_outcome": 0.0,
            "avg_favorable_exc": 0.0,
            "avg_adverse_exc": 0.0,
        }

    n = float(len(df_events))
    realized_col = f"Realized Return @+{int(forward_bars)} (%)"
    legacy_col = f"Return @+{int(forward_bars)} (%)"
    returns = pd.to_numeric(
        df_events.get(realized_col, df_events.get(legacy_col, pd.Series(dtype=float))),
        errors="coerce",
    )
    outcomes = df_events.get("Outcome", pd.Series(dtype=object)).astype(str).str.upper()
    tp_rate = float((outcomes == "TP").mean() * 100.0) if n > 0 else 0.0
    sl_rate = float(outcomes.isin({"SL", "BOTH"}).mean() * 100.0) if n > 0 else 0.0
    timeout_rate = float((outcomes == "TIMEOUT").mean() * 100.0) if n > 0 else 0.0
    return {
        "occurrences": n,
        "tp_rate": tp_rate,
        "sl_rate": sl_rate,
        "timeout_rate": timeout_rate,
        "avg_outcome": float(returns.mean()) if len(returns) else 0.0,
        "median_outcome": float(np.nanmedian(returns.values)) if len(returns) else 0.0,
        "avg_favorable_exc": float(
            pd.to_numeric(df_events.get("Favorable Excursion (%)", pd.Series(dtype=float)), errors="coerce").mean()
        ),
        "avg_adverse_exc": float(
            pd.to_numeric(df_events.get("Adverse Excursion (%)", pd.Series(dtype=float)), errors="coerce").mean()
        ),
    }


def summarize_scalp_outcome_by_class(df_events: pd.DataFrame, forward_bars: int) -> pd.DataFrame:
    if df_events is None or df_events.empty or "Setup Confirm" not in df_events.columns:
        return pd.DataFrame()

    realized_col = f"Realized Return @+{int(forward_bars)} (%)"
    legacy_col = f"Return @+{int(forward_bars)} (%)"
    d = df_events.copy()
    d[realized_col] = pd.to_numeric(
        d.get(realized_col, d.get(legacy_col, pd.Series(dtype=float))),
        errors="coerce",
    )
    outcome = d.get("Outcome", pd.Series(dtype=object)).astype(str).str.upper()
    d["is_tp"] = (outcome == "TP").astype(int)
    d["is_sl"] = (outcome == "SL").astype(int)
    d["is_timeout"] = (outcome == "TIMEOUT").astype(int)
    d["Favorable Excursion (%)"] = pd.to_numeric(
        d.get("Favorable Excursion (%)", pd.Series(dtype=float)),
        errors="coerce",
    )
    d["Adverse Excursion (%)"] = pd.to_numeric(
        d.get("Adverse Excursion (%)", pd.Series(dtype=float)),
        errors="coerce",
    )
    grouped = (
        d.groupby("Setup Confirm", dropna=False)
        .agg(
            Occurrences=("Setup Confirm", "count"),
            TPRate=("is_tp", "mean"),
            SLRate=("is_sl", "mean"),
            TimeoutRate=("is_timeout", "mean"),
            MedianOutcome=(realized_col, "median"),
            AvgOutcome=(realized_col, "mean"),
            AvgFavorableExcursion=("Favorable Excursion (%)", "mean"),
            AvgAdverseExcursion=("Adverse Excursion (%)", "mean"),
        )
        .reset_index()
    )
    grouped["TPRate"] = grouped["TPRate"] * 100.0
    grouped["SLRate"] = grouped["SLRate"] * 100.0
    grouped["TimeoutRate"] = grouped["TimeoutRate"] * 100.0
    return grouped.sort_values(by=["Occurrences", "AvgOutcome"], ascending=[False, False]).reset_index(drop=True)


def build_setup_outcome_study(
    df: pd.DataFrame,
    analyzer: Callable[[pd.DataFrame], AnalysisLike],
    ml_predictor: MLEnsembleLike,
    conviction_fn: ConvictionLike,
    signal_plain_fn: Callable[[str], str],
    direction_key_fn: Callable[[str], str],
    *,
    timeframe: str | None = None,
    df_4h: pd.DataFrame | None = None,
    df_1d: pd.DataFrame | None = None,
    get_scalping_entry_target_fn: Callable[..., tuple] | None = None,
    sr_lookback_fn: Callable[[str | None], int] | None = None,
    setup_filter: str = "ALL",
    forward_bars: int = 10,
    window_size: int = 200,
    min_history: int = 55,
) -> pd.DataFrame:
    """Event study for Setup Confirm classes.

    For each matching setup event at bar i (closed candle):
    - event price is close[i]
    - future path is close[i+1 .. i+forward_bars]
    - MFE/MAE style excursions are derived from future highs/lows
    """
    forward_bars = max(1, int(forward_bars))
    min_history = max(30, int(min_history))
    allowed_classes = _allowed_setup_classes(setup_filter)

    rows: list[dict] = []
    max_idx = len(df) - forward_bars - 1
    if max_idx <= min_history:
        return pd.DataFrame()

    for i in range(min_history, max_idx + 1):
        start_idx = max(0, i - max(60, int(window_size)))
        df_slice = df.iloc[start_idx : i + 1]
        if len(df_slice) < min_history:
            continue

        try:
            analysis = analyzer(df_slice)
            bias_score = _read_bias_like(analysis)
            directional_confidence = float(bias_confidence_from_bias(bias_score))
            _prob, ai_direction, ai_details = ml_predictor(df_slice)
        except Exception:
            continue

        raw_signal = str(getattr(analysis, "signal", "") or "")
        signal_side = str(direction_key_fn(signal_plain_fn(raw_signal)))
        if signal_side not in {"UPSIDE", "DOWNSIDE"}:
            continue

        ai_key = str(direction_key_fn(ai_direction))
        agreement = float(ai_details.get("agreement", 0.0)) if isinstance(ai_details, dict) else 0.0
        directional_agreement = (
            float(ai_details.get("directional_agreement", agreement)) if isinstance(ai_details, dict) else agreement
        )
        consensus_agreement = (
            float(ai_details.get("consensus_agreement", 0.0)) if isinstance(ai_details, dict) else 0.0
        )
        ai_status = str(ai_details.get("status", "")) if isinstance(ai_details, dict) else ""
        votes, _, decision_agreement = ai_vote_metrics(
            ai_key,
            directional_agreement,
            consensus_agreement,
        )

        structure = structure_state(signal_side, ai_key, directional_confidence, float(decision_agreement))
        conviction_label, _ = conviction_fn(signal_side, ai_key, directional_confidence, float(decision_agreement))
        adx_raw = getattr(analysis, "adx", np.nan)
        try:
            adx_val = float(adx_raw)
        except Exception:
            adx_val = float("nan")
        execution_confidence = build_execution_confidence_snapshot(
            direction=signal_side,
            bias_score=bias_score,
            adx_val=adx_val,
            structure_state=structure,
            conviction_label=str(conviction_label),
            ai_agreement=float(decision_agreement),
        )
        rr_ratio = _resolve_rr_ratio_from_plan(
            get_scalping_entry_target_fn=get_scalping_entry_target_fn,
            sr_lookback_fn=sr_lookback_fn,
            df_slice=df_slice,
            analysis_obj=analysis,
            bias_score=bias_score,
        )

        setup_ctx = _resolve_setup_context(
            event_timestamp=df["timestamp"].iloc[i],
            timeframe=timeframe,
            df_slice=df_slice,
            df_4h=df_4h,
            df_1d=df_1d,
            analysis_obj=analysis,
            tactical_direction=signal_side,
            ai_key=ai_key,
            agreement=float(decision_agreement),
            ai_probability=float(_prob),
            directional_agreement=float(directional_agreement),
            consensus_agreement=float(consensus_agreement),
            ai_status=ai_status,
            adx_val=adx_val,
            conviction_label=str(conviction_label),
            directional_confidence=directional_confidence,
            bias_score=bias_score,
            rr_ratio=rr_ratio,
        )
        action_raw = str(setup_ctx["action_raw"])
        action_reason = str(setup_ctx["action_reason"])
        action_class = normalize_action_class(action_raw)
        if action_class not in allowed_classes:
            continue

        event_price = float(df["close"].iloc[i])
        if not np.isfinite(event_price) or event_price <= 0:
            continue

        high_col = "high" if "high" in df.columns else "close"
        low_col = "low" if "low" in df.columns else "close"
        f_close = pd.to_numeric(df["close"].iloc[i + 1 : i + 1 + forward_bars], errors="coerce")
        f_high = pd.to_numeric(df[high_col].iloc[i + 1 : i + 1 + forward_bars], errors="coerce")
        f_low = pd.to_numeric(df[low_col].iloc[i + 1 : i + 1 + forward_bars], errors="coerce")
        if len(f_close) < forward_bars or f_close.isna().any():
            continue

        end_price = float(f_close.iloc[-1])
        end_return_pct = ((end_price / event_price) - 1.0) * 100.0
        max_high = float(f_high.max()) if not f_high.empty and f_high.notna().any() else end_price
        min_low = float(f_low.min()) if not f_low.empty and f_low.notna().any() else end_price
        max_up_pct = ((max_high / event_price) - 1.0) * 100.0
        max_down_pct = ((min_low / event_price) - 1.0) * 100.0

        trade_direction = str(setup_ctx.get("direction_key") or signal_side)
        is_upside = trade_direction == "UPSIDE"
        directional_return_pct = end_return_pct if is_upside else (-end_return_pct)
        favorable_exc_pct = max(0.0, max_up_pct) if is_upside else max(0.0, -max_down_pct)
        adverse_exc_pct = max(0.0, -max_down_pct) if is_upside else max(0.0, max_up_pct)

        row = {
            "Event Time": df["timestamp"].iloc[i],
            "Setup Confirm": _setup_confirm_label(action_class),
            "Setup Class": action_class,
            "Action Reason": action_reason,
            "Direction": _direction_label_from_key(trade_direction),
            "AI Direction": _direction_label_from_key(str(setup_ctx.get("ai_direction_key") or ai_key)),
            "AI Votes": str(setup_ctx.get("ai_votes") or f"{votes}/3"),
            "Confidence": round(float(setup_ctx.get("confidence", float("nan"))), 1),
            "AI Confidence": round(float(setup_ctx.get("ai_confidence", float("nan"))), 1),
            "AI Timeframe Conflict": bool(setup_ctx.get("ai_timeframe_conflict", False)),
            "AI Data Partial": bool(setup_ctx.get("ai_degraded_data", False)),
            "Selected-TF Confidence": round(float(execution_confidence.score), 1),
            "Bias": round(bias_score, 1),
            "Event Price": event_price,
            f"End Price (+{forward_bars})": end_price,
            f"Return @+{forward_bars} (%)": directional_return_pct,
            "Base Return (%)": end_return_pct,
            "Max Up (%)": max_up_pct,
            "Max Down (%)": max_down_pct,
            "Favorable Excursion (%)": favorable_exc_pct,
            "Adverse Excursion (%)": adverse_exc_pct,
        }

        for step in range(1, forward_bars + 1):
            p = float(f_close.iloc[step - 1])
            r = ((p / event_price) - 1.0) * 100.0
            row[f"Price +{step}"] = p
            row[f"Return +{step} (%)"] = r
            row[f"Directional Return +{step} (%)"] = r if is_upside else (-r)

        rows.append(row)

    out = pd.DataFrame(rows)
    if out.empty:
        return out
    return out.sort_values("Event Time").reset_index(drop=True)


def summarize_setup_outcome_study(df_events: pd.DataFrame, forward_bars: int) -> dict[str, float]:
    """Aggregate high-level KPIs for setup outcome study."""
    if df_events is None or df_events.empty:
        return {
            "occurrences": 0.0,
            "favorable_rate": 0.0,
            "median_dir_return": 0.0,
            "expectancy": 0.0,
            "avg_win": 0.0,
            "avg_loss": 0.0,
            "payoff_ratio": 0.0,
            "profit_factor": 0.0,
            "avg_favorable_exc": 0.0,
            "avg_adverse_exc": 0.0,
        }

    n = float(len(df_events))
    ret_col = f"Return @+{int(forward_bars)} (%)"
    returns = pd.to_numeric(df_events.get(ret_col, pd.Series(dtype=float)), errors="coerce")
    metrics = _return_distribution_metrics(returns)
    favorable_rate = metrics["win_rate"] if n > 0 else 0.0
    median_dir_return = metrics["median"]
    avg_favorable_exc = float(
        pd.to_numeric(df_events.get("Favorable Excursion (%)", pd.Series(dtype=float)), errors="coerce").mean()
    )
    avg_adverse_exc = float(
        pd.to_numeric(df_events.get("Adverse Excursion (%)", pd.Series(dtype=float)), errors="coerce").mean()
    )
    return {
        "occurrences": n,
        "favorable_rate": favorable_rate,
        "median_dir_return": median_dir_return,
        "expectancy": metrics["expectancy"],
        "avg_win": metrics["avg_win"],
        "avg_loss": metrics["avg_loss"],
        "payoff_ratio": metrics["payoff_ratio"],
        "profit_factor": metrics["profit_factor"],
        "avg_favorable_exc": avg_favorable_exc,
        "avg_adverse_exc": avg_adverse_exc,
    }


def summarize_setup_outcome_by_class(df_events: pd.DataFrame, forward_bars: int) -> pd.DataFrame:
    """Class-level summary for setup outcome study."""
    if df_events is None or df_events.empty or "Setup Confirm" not in df_events.columns:
        return pd.DataFrame()

    ret_col = f"Return @+{int(forward_bars)} (%)"
    d = df_events.copy()
    d[ret_col] = pd.to_numeric(d.get(ret_col, pd.Series(dtype=float)), errors="coerce")
    d["is_favorable"] = (d[ret_col] > 0).astype(int)
    d["Favorable Excursion (%)"] = pd.to_numeric(
        d.get("Favorable Excursion (%)", pd.Series(dtype=float)),
        errors="coerce",
    )
    d["Adverse Excursion (%)"] = pd.to_numeric(
        d.get("Adverse Excursion (%)", pd.Series(dtype=float)),
        errors="coerce",
    )

    grouped = (
        d.groupby("Setup Confirm", dropna=False)
        .agg(
            Occurrences=("Setup Confirm", "count"),
            FavorableRate=("is_favorable", "mean"),
            MedianDirectionalReturn=(ret_col, "median"),
            AvgDirectionalReturn=(ret_col, "mean"),
            AvgWin=(ret_col, lambda s: float(pd.to_numeric(s, errors="coerce")[pd.to_numeric(s, errors="coerce") > 0].mean()) if (pd.to_numeric(s, errors="coerce") > 0).any() else 0.0),
            AvgLoss=(ret_col, lambda s: abs(float(pd.to_numeric(s, errors="coerce")[pd.to_numeric(s, errors="coerce") <= 0].mean())) if (pd.to_numeric(s, errors="coerce") <= 0).any() else 0.0),
            AvgFavorableExcursion=("Favorable Excursion (%)", "mean"),
            AvgAdverseExcursion=("Adverse Excursion (%)", "mean"),
        )
        .reset_index()
    )
    grouped["FavorableRate"] = grouped["FavorableRate"] * 100.0
    grouped["Expectancy"] = grouped["AvgDirectionalReturn"]
    grouped["PayoffRatio"] = grouped.apply(
        lambda r: (float(r["AvgWin"]) / float(r["AvgLoss"])) if float(r["AvgLoss"]) > 0 else (float("inf") if float(r["AvgWin"]) > 0 else 0.0),
        axis=1,
    )
    grouped["ProfitFactor"] = grouped.apply(
        lambda r: (
            float(r["AvgWin"]) * (float(r["FavorableRate"]) / 100.0)
        ) / (
            float(r["AvgLoss"]) * max(1e-9, 1.0 - (float(r["FavorableRate"]) / 100.0))
        ) if float(r["AvgLoss"]) > 0 and float(r["FavorableRate"]) < 100.0 else (float("inf") if float(r["AvgWin"]) > 0 else 0.0),
        axis=1,
    )
    grouped[["QualityGrade", "QualityNote"]] = grouped.apply(
        lambda r: pd.Series(
            grade_setup_class_quality(
                occurrences=float(r["Occurrences"]),
                expectancy=float(r["Expectancy"]),
                profit_factor=float(r["ProfitFactor"]),
                payoff_ratio=float(r["PayoffRatio"]),
                win_rate=float(r["FavorableRate"]),
            )
        ),
        axis=1,
    )
    return grouped.sort_values(by=["Expectancy", "ProfitFactor", "Occurrences"], ascending=[False, False, False]).reset_index(drop=True)


def run_backtest(
    df: pd.DataFrame,
    analyzer: Callable[[pd.DataFrame], AnalysisLike],
    threshold: float = 70,
    exit_after: int = 5,
    commission: float = 0.001,
    slippage: float = 0.0005,
) -> Tuple[pd.DataFrame, str]:
    """Run single-position backtest over a price series.

    Entry filter is direction-agnostic:
    - Compute execution confidence from directional bias (bias score)
    - Enter LONG/SHORT only when confidence >= threshold
    """
    exit_after = max(1, int(exit_after))
    commission = max(0.0, float(commission))
    slippage = max(0.0, float(slippage))
    results = []
    equity_curve = [10000.0]
    peak = 10000.0
    max_drawdown = 0.0
    consecutive_losses = 0
    max_consecutive_losses = 0
    window_size = 200

    i = 55
    # Signal is computed on bar i close. Execution starts on next bar open.
    while i < len(df) - exit_after - 1:
        start_idx = max(0, i - window_size)
        df_slice = df.iloc[start_idx : i + 1]
        if len(df_slice) < 55:
            i += 1
            continue

        try:
            result = analyzer(df_slice)
            raw_signal = result.signal
            bias_score = _read_bias_like(result)
            directional_confidence = float(bias_confidence_from_bias(bias_score))
        except Exception:
            i += 1
            continue

        sig_plain = _normalize_direction_signal(raw_signal)
        tactical_structure = structure_state(
            "UPSIDE" if sig_plain == "LONG" else ("DOWNSIDE" if sig_plain == "SHORT" else "NEUTRAL"),
            "NEUTRAL",
            directional_confidence,
            0.0,
        )
        tactical_conviction = (
            "TREND" if directional_confidence >= 70.0 else ("WEAK" if directional_confidence < 55.0 else "MEDIUM")
        )
        execution_confidence = build_execution_confidence_snapshot(
            direction="UPSIDE" if sig_plain == "LONG" else ("DOWNSIDE" if sig_plain == "SHORT" else "NEUTRAL"),
            bias_score=bias_score,
            adx_val=getattr(result, "adx", np.nan),
            structure_state=tactical_structure,
            conviction_label=tactical_conviction,
            ai_agreement=0.0,
        )

        long_ok = sig_plain == "LONG" and float(execution_confidence.score) >= threshold
        short_ok = sig_plain == "SHORT" and float(execution_confidence.score) >= threshold
        if not (long_ok or short_ok):
            i += 1
            continue

        entry_idx = i + 1
        exit_idx = entry_idx + exit_after
        if exit_idx >= len(df):
            break

        entry_open = float(df["open"].iloc[entry_idx]) if "open" in df.columns else float(df["close"].iloc[entry_idx])
        exit_open = float(df["open"].iloc[exit_idx]) if "open" in df.columns else float(df["close"].iloc[exit_idx])
        if entry_open <= 0 or exit_open <= 0:
            i += 1
            continue

        if sig_plain == "LONG":
            entry_exec = entry_open * (1.0 + slippage)
            exit_exec = exit_open * (1.0 - slippage)
            gross_ret = (exit_exec - entry_exec) / entry_exec
        else:
            entry_exec = entry_open * (1.0 - slippage)
            exit_exec = exit_open * (1.0 + slippage)
            gross_ret = (entry_exec - exit_exec) / entry_exec

        net_ret = gross_ret - 2.0 * commission
        pnl = net_ret * 100.0
        regime, regime_score = _infer_regime(df_slice, result)

        equity = equity_curve[-1] * (1 + pnl / 100)
        equity_curve.append(equity)

        peak = max(peak, equity)
        drawdown = (peak - equity) / peak * 100
        max_drawdown = max(max_drawdown, drawdown)

        if pnl <= 0:
            consecutive_losses += 1
            max_consecutive_losses = max(max_consecutive_losses, consecutive_losses)
        else:
            consecutive_losses = 0

        results.append(
            {
                "Date": df["timestamp"].iloc[i],
                "Signal Time": df["timestamp"].iloc[i],
                "Confidence": round(float(execution_confidence.score), 1),
                "Bias": round(bias_score, 1),
                "Signal": sig_plain,
                "Entry": entry_exec,
                "Exit": exit_exec,
                "PnL (%)": round(pnl, 2),
                "Equity": round(equity, 2),
                "Regime": regime,
                "Regime Score": round(regime_score, 1),
                "Holding Bars": int(exit_after),
            }
        )

        # Single-position mode: next signal evaluation starts at exit bar.
        i = exit_idx

    df_results = pd.DataFrame(results)
    if df_results.empty:
        return (
            df_results,
            "<div style='color:#FFB000;margin-top:1rem;'>"
            "<p><b>⚠️ No Signals:</b> No trades met the threshold criteria</p>"
            "<p>Try lowering the confidence threshold or using more data</p>"
            "</div>",
        )

    wins = int((df_results["PnL (%)"] > 0).sum())
    losses = int((df_results["PnL (%)"] <= 0).sum())
    total_trades = wins + losses
    winrate = (wins / total_trades) * 100 if total_trades > 0 else 0.0

    gross_profit = float(df_results[df_results["PnL (%)"] > 0]["PnL (%)"].sum())
    gross_loss = abs(float(df_results[df_results["PnL (%)"] <= 0]["PnL (%)"].sum()))
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")

    avg_win = (
        float(df_results[df_results["PnL (%)"] > 0]["PnL (%)"].mean()) if wins > 0 else 0.0
    )
    avg_loss = (
        float(df_results[df_results["PnL (%)"] <= 0]["PnL (%)"].mean()) if losses > 0 else 0.0
    )

    total_return = (equity_curve[-1] - equity_curve[0]) / equity_curve[0] * 100
    returns = df_results["PnL (%)"].astype(float) / 100.0
    mean_return = float(returns.mean())
    std_return = float(returns.std())
    # Approximate annualization by timeframe implied from candle timestamps.
    ann_factor = 365.0
    if len(df) >= 2 and "timestamp" in df.columns:
        try:
            dt_hours = (df["timestamp"].iloc[1] - df["timestamp"].iloc[0]).total_seconds() / 3600.0
            if dt_hours > 0:
                periods_per_day = 24.0 / dt_hours
                ann_factor = periods_per_day * 365.0 / exit_after
        except Exception:
            ann_factor = 365.0 / exit_after
    else:
        ann_factor = 365.0 / exit_after

    sharpe_ratio = (mean_return / (std_return + 1e-9)) * np.sqrt(ann_factor) if std_return > 0 else 0.0

    summary_html = f"""
    <div style='margin-top:1rem; background-color:#16213E; padding:20px; border-radius:10px;'>
        <h3 style='color:#06D6A0; margin-top:0;'>📊 Backtest Results</h3>
        <p style='color:#8CA1B6; margin:0;'>Trades: {total_trades} | Win Rate: {winrate:.1f}% | Profit Factor: {profit_factor:.2f}</p>
        <p style='color:#8CA1B6; margin:6px 0 0 0;'>Return: {total_return:+.2f}% | Max DD: {max_drawdown:.2f}% | Sharpe: {sharpe_ratio:.2f}</p>
        <p style='color:#8CA1B6; margin:6px 0 0 0;'>Avg Win: {avg_win:+.2f}% | Avg Loss: {avg_loss:.2f}% | Max Consecutive Losses: {max_consecutive_losses}</p>
    </div>
    """
    return df_results, summary_html
