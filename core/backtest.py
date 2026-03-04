from __future__ import annotations

from collections import Counter
from typing import Any, Callable, Protocol, Tuple

import numpy as np
import pandas as pd
from core.market_decision import (
    action_decision_with_reason,
    ai_vote_metrics,
    normalize_action_class,
    structure_state,
)
from core.scalping import scalp_gate_thresholds
from core.signal_contract import strength_from_bias


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
        strength: float,
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
    cls = str(action_class or "").strip().upper()
    if cls == "ENTER_TREND_AI":
        return "TREND+AI"
    if cls == "ENTER_TREND_LED":
        return "TREND-led"
    if cls == "ENTER_AI_LED":
        return "AI-led"
    return cls or "UNKNOWN"


def _direction_label_from_key(value: str) -> str:
    key = str(value or "").strip().upper()
    if key == "UPSIDE":
        return "Upside"
    if key == "DOWNSIDE":
        return "Downside"
    return "Neutral"


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
            strength = float(strength_from_bias(bias_score))
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
        votes, _display_ratio, decision_agreement = ai_vote_metrics(
            ai_key,
            directional_agreement,
            consensus_agreement,
        )

        structure = structure_state(signal_side, ai_key, strength, float(decision_agreement))
        conviction_label, _ = conviction_fn(signal_side, ai_key, strength, float(decision_agreement))
        adx_raw = getattr(result, "adx", np.nan)
        try:
            adx_val = float(adx_raw)
        except Exception:
            adx_val = float("nan")

        action_raw, action_reason = action_decision_with_reason(
            signal_side,
            strength,
            structure,
            str(conviction_label),
            float(decision_agreement),
            adx_val,
        )
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

        is_upside = signal_side == "UPSIDE"
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
                "Direction": _direction_label_from_key(signal_side),
                "AI Direction": _direction_label_from_key(ai_key),
                "AI Votes": f"{votes}/3",
                "Strength": round(strength, 1),
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
    d["is_win"] = (pd.to_numeric(d["PnL (%)"], errors="coerce") > 0).astype(int)
    grouped = (
        d.groupby("Setup Confirm", dropna=False)
        .agg(
            Trades=("PnL (%)", "count"),
            WinRate=("is_win", "mean"),
            AvgPnL=("PnL (%)", "mean"),
            TotalPnL=("PnL (%)", "sum"),
            GrossProfit=("PnL (%)", lambda s: float(pd.to_numeric(s, errors="coerce")[pd.to_numeric(s, errors="coerce") > 0].sum())),
            GrossLoss=("PnL (%)", lambda s: abs(float(pd.to_numeric(s, errors="coerce")[pd.to_numeric(s, errors="coerce") <= 0].sum()))),
        )
        .reset_index()
    )
    grouped["WinRate"] = grouped["WinRate"] * 100.0
    grouped["ProfitFactor"] = grouped.apply(
        lambda r: (float(r["GrossProfit"]) / float(r["GrossLoss"])) if float(r["GrossLoss"]) > 0 else float("inf"),
        axis=1,
    )
    grouped = grouped.sort_values(by=["Trades", "TotalPnL"], ascending=[False, False]).reset_index(drop=True)
    return grouped[["Setup Confirm", "Trades", "WinRate", "AvgPnL", "TotalPnL", "ProfitFactor"]]


def _allowed_setup_classes(setup_filter: str) -> set[str]:
    s = str(setup_filter or "").strip().upper()
    if s in {"TREND+AI", "ENTER_TREND_AI"}:
        return {"ENTER_TREND_AI"}
    if s in {"TREND-LED", "TREND_LED", "ENTER_TREND_LED"}:
        return {"ENTER_TREND_LED"}
    if s in {"AI-LED", "AI_LED", "ENTER_AI_LED"}:
        return {"ENTER_AI_LED"}
    return {"ENTER_TREND_AI", "ENTER_TREND_LED", "ENTER_AI_LED"}


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
    forward_bars: int = 10,
    window_size: int = 200,
    min_history: int = 55,
) -> pd.DataFrame:
    """Event-study engine for scalp opportunities generated by market scalp logic."""
    forward_bars = max(1, int(forward_bars))
    min_history = max(30, int(min_history))
    gate_min_rr, gate_min_adx, gate_min_strength = _scalp_gate_thresholds(timeframe)

    rows: list[dict] = []
    diagnostics: dict[str, Any] = {
        "timeframe": str(timeframe),
        "forward_bars": int(forward_bars),
        "gate_thresholds": {
            "min_rr": float(gate_min_rr),
            "min_adx": float(gate_min_adx),
            "min_strength": float(gate_min_strength),
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
            strength = float(strength_from_bias(bias_score))
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
        votes, _display_ratio, decision_agreement = ai_vote_metrics(
            ai_key,
            directional_agreement,
            consensus_agreement,
        )

        structure = structure_state(signal_side, ai_key, strength, float(decision_agreement))
        conviction_label, _ = conviction_fn(signal_side, ai_key, strength, float(decision_agreement))
        adx_raw = getattr(analysis, "adx", np.nan)
        try:
            adx_val = float(adx_raw)
        except Exception:
            adx_val = float("nan")

        action_raw, _action_reason = action_decision_with_reason(
            signal_side,
            strength,
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
                )
            except TypeError as te:
                # Service-layer wrapper does not take sr_lookback_fn; retry without it.
                if "sr_lookback_fn" not in str(te):
                    raise
                scalp_direction, entry, target, stop, rr_ratio, breakout_note = get_scalping_entry_target_fn(
                    df_slice,
                    bias_score,
                    getattr(analysis, "supertrend", ""),
                    getattr(analysis, "ichimoku", ""),
                    getattr(analysis, "vwap", ""),
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
            strength=strength,
            conviction_label=str(conviction_label),
            entry=entry,
            stop=stop,
            target=target,
            min_rr=gate_min_rr,
            min_adx=gate_min_adx,
            min_strength=gate_min_strength,
        )
        if not gate_pass:
            gate_reject_counter[str(gate_reason or "UNKNOWN_GATE_REJECT")] += 1
            continue

        diagnostics["gate_pass_candidates"] += 1
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
            "AI Direction": _direction_label_from_key(ai_key),
            "AI Votes": f"{votes}/3",
            "Tech vs AI Alignment": str(conviction_label or "").strip().upper(),
            "Strength": round(strength, 1),
            "Event Price": event_price,
            "Target": target_f,
            "Stop": stop_f,
            "R:R": rr_f,
            "Gate": str(gate_reason or "PASS"),
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
            strength = float(strength_from_bias(bias_score))
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
        votes, _display_ratio, decision_agreement = ai_vote_metrics(
            ai_key,
            directional_agreement,
            consensus_agreement,
        )

        structure = structure_state(signal_side, ai_key, strength, float(decision_agreement))
        conviction_label, _ = conviction_fn(signal_side, ai_key, strength, float(decision_agreement))
        adx_raw = getattr(analysis, "adx", np.nan)
        try:
            adx_val = float(adx_raw)
        except Exception:
            adx_val = float("nan")

        action_raw, action_reason = action_decision_with_reason(
            signal_side,
            strength,
            structure,
            str(conviction_label),
            float(decision_agreement),
            adx_val,
        )
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

        is_upside = signal_side == "UPSIDE"
        directional_return_pct = end_return_pct if is_upside else (-end_return_pct)
        favorable_exc_pct = max(0.0, max_up_pct) if is_upside else max(0.0, -max_down_pct)
        adverse_exc_pct = max(0.0, -max_down_pct) if is_upside else max(0.0, max_up_pct)

        row = {
            "Event Time": df["timestamp"].iloc[i],
            "Setup Confirm": _setup_confirm_label(action_class),
            "Setup Class": action_class,
            "Action Reason": action_reason,
            "Direction": _direction_label_from_key(signal_side),
            "AI Direction": _direction_label_from_key(ai_key),
            "AI Votes": f"{votes}/3",
            "Tech vs AI Alignment": str(conviction_label or "").strip().upper(),
            "Strength": round(strength, 1),
            "Bias": round(bias_score, 1),
            "Event Price": event_price,
            f"End Price (+{forward_bars})": end_price,
            f"Return @+{forward_bars} (%)": directional_return_pct,
            "Raw Return (%)": end_return_pct,
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
            "avg_favorable_exc": 0.0,
            "avg_adverse_exc": 0.0,
        }

    n = float(len(df_events))
    ret_col = f"Return @+{int(forward_bars)} (%)"
    returns = pd.to_numeric(df_events.get(ret_col, pd.Series(dtype=float)), errors="coerce")
    fav = returns[returns > 0]
    favorable_rate = (len(fav) / n) * 100.0 if n > 0 else 0.0
    median_dir_return = float(np.nanmedian(returns.values)) if len(returns) else 0.0
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
            AvgFavorableExcursion=("Favorable Excursion (%)", "mean"),
            AvgAdverseExcursion=("Adverse Excursion (%)", "mean"),
        )
        .reset_index()
    )
    grouped["FavorableRate"] = grouped["FavorableRate"] * 100.0
    return grouped.sort_values(by=["Occurrences", "AvgDirectionalReturn"], ascending=[False, False]).reset_index(drop=True)


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
    - Compute strength from directional bias (bias score)
    - Enter LONG/SHORT only when strength >= threshold
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
            strength_score = float(strength_from_bias(bias_score))
        except Exception:
            i += 1
            continue

        sig_plain = _normalize_direction_signal(raw_signal)

        long_ok = sig_plain == "LONG" and strength_score >= threshold
        short_ok = sig_plain == "SHORT" and strength_score >= threshold
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
                "Strength": round(strength_score, 1),
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
            "<p>Try lowering the strength threshold or using more data</p>"
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
