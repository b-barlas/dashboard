from __future__ import annotations

import math

import numpy as np
import pandas as pd


MAX_PORTFOLIO_COINS = 10
MIN_MATCHED_BARS = 40
RETURN_FLOOR = -0.95
MAX_SCENARIO_HORIZON_BARS = 48
MIN_TYPICAL_BAR_MOVE = 0.0005


def sanitize_holdings_rows(
    raw_df: pd.DataFrame | list[dict],
    normalize_coin_input,
    *,
    max_items: int = MAX_PORTFOLIO_COINS,
    return_meta: bool = False,
) -> list[dict]:
    if raw_df is None:
        empty_meta = {
            "coin_rows": 0,
            "kept_rows": 0,
            "duplicate_rows": 0,
            "invalid_value_rows": 0,
            "truncated_rows": 0,
            "duplicate_symbols": [],
        }
        return ([], empty_meta) if return_meta else []
    df = raw_df if isinstance(raw_df, pd.DataFrame) else pd.DataFrame(raw_df)
    if df.empty:
        empty_meta = {
            "coin_rows": 0,
            "kept_rows": 0,
            "duplicate_rows": 0,
            "invalid_value_rows": 0,
            "truncated_rows": 0,
            "duplicate_symbols": [],
        }
        return ([], empty_meta) if return_meta else []

    cleaned: list[dict] = []
    seen: set[str] = set()
    duplicate_rows = 0
    invalid_value_rows = 0
    truncated_rows = 0
    duplicate_symbols: list[str] = []
    coin_rows = 0
    for _, row in df.iterrows():
        raw_coin = str(row.get("Coin", "")).strip()
        raw_value = row.get("Current Value ($)")
        if not raw_coin:
            continue
        coin_rows += 1
        symbol = normalize_coin_input(raw_coin)
        if not symbol:
            invalid_value_rows += 1
            continue
        if symbol in seen:
            duplicate_rows += 1
            duplicate_symbols.append(symbol.split("/")[0])
            continue
        try:
            current_value = float(raw_value)
        except Exception:
            invalid_value_rows += 1
            continue
        if not np.isfinite(current_value) or current_value <= 0:
            invalid_value_rows += 1
            continue
        if len(cleaned) >= max_items:
            truncated_rows += 1
            continue
        seen.add(symbol)
        cleaned.append(
            {
                "coin": symbol.split("/")[0],
                "symbol": symbol,
                "current_value": current_value,
            }
        )
    meta = {
        "coin_rows": coin_rows,
        "kept_rows": len(cleaned),
        "duplicate_rows": duplicate_rows,
        "invalid_value_rows": invalid_value_rows,
        "truncated_rows": truncated_rows,
        "duplicate_symbols": sorted(set(duplicate_symbols)),
    }
    return (cleaned, meta) if return_meta else cleaned


def returns_series(df: pd.DataFrame, label: str, *, horizon_bars: int = 1) -> pd.Series | None:
    if df is None or len(df) <= 10 or "close" not in df.columns:
        return None
    horizon_bars = max(1, int(horizon_bars))
    prices = pd.to_numeric(df["close"], errors="coerce")
    rets = ((prices.shift(-horizon_bars) / prices) - 1.0).dropna()
    if rets.empty:
        return None
    if "timestamp" in df.columns:
        ts_vals = df.loc[rets.index, "timestamp"]
        if pd.api.types.is_numeric_dtype(ts_vals):
            idx = pd.to_datetime(ts_vals, unit="ms", errors="coerce", utc=True)
        else:
            idx = pd.to_datetime(ts_vals, errors="coerce", utc=True)
    else:
        idx = pd.to_datetime(df.index, errors="coerce", utc=True)
    series = pd.Series(rets.values, index=idx, name=label).dropna()
    series = series[~series.index.duplicated(keep="last")]
    series = series[np.isfinite(series)]
    if len(series) < MIN_MATCHED_BARS:
        return None
    return series


def estimate_anchor_horizon(anchor_df: pd.DataFrame, anchor_move: float) -> dict:
    one_bar = returns_series(anchor_df, "anchor_1bar", horizon_bars=1)
    if one_bar is None or one_bar.empty:
        return {
            "horizon_bars": 1,
            "typical_bar_move": MIN_TYPICAL_BAR_MOVE,
            "raw_horizon_bars": 1,
            "horizon_capped": False,
            "horizon_cap_reason": None,
        }

    typical_bar_move = float(one_bar.abs().median())
    if not math.isfinite(typical_bar_move) or typical_bar_move <= 0:
        typical_bar_move = float(one_bar.abs().mean())
    if not math.isfinite(typical_bar_move) or typical_bar_move <= 0:
        typical_bar_move = MIN_TYPICAL_BAR_MOVE
    typical_bar_move = max(typical_bar_move, MIN_TYPICAL_BAR_MOVE)

    raw_horizon = int(math.ceil(abs(anchor_move) / typical_bar_move)) if abs(anchor_move) > 0 else 1
    max_by_data = max(1, len(anchor_df) - MIN_MATCHED_BARS - 1)
    bounded_raw_horizon = max(1, raw_horizon)
    horizon_bars = min(bounded_raw_horizon, MAX_SCENARIO_HORIZON_BARS, max_by_data)
    cap_reason = None
    if horizon_bars < bounded_raw_horizon:
        if horizon_bars == MAX_SCENARIO_HORIZON_BARS:
            cap_reason = "stability_cap"
        elif horizon_bars == max_by_data:
            cap_reason = "history_limit"
    return {
        "horizon_bars": horizon_bars,
        "typical_bar_move": typical_bar_move,
        "raw_horizon_bars": bounded_raw_horizon,
        "horizon_capped": cap_reason is not None,
        "horizon_cap_reason": cap_reason,
    }


def regression_relationship(anchor_series: pd.Series, coin_series: pd.Series) -> dict | None:
    if anchor_series is None or coin_series is None:
        return None
    aligned = pd.concat([anchor_series.rename("anchor"), coin_series.rename("coin")], axis=1, join="inner").dropna()
    if len(aligned) < MIN_MATCHED_BARS:
        return None

    x = aligned["anchor"].to_numpy(dtype=float)
    y = aligned["coin"].to_numpy(dtype=float)
    x_var = float(np.var(x))
    y_var = float(np.var(y))
    if x_var <= 0 or y_var <= 0:
        return None

    beta = float(np.cov(x, y, ddof=0)[0, 1] / x_var)
    alpha = float(y.mean() - beta * x.mean())
    y_hat = alpha + beta * x
    ss_tot = float(((y - y.mean()) ** 2).sum())
    ss_res = float(((y - y_hat) ** 2).sum())
    r2 = 0.0 if ss_tot <= 0 else max(0.0, min(1.0, 1.0 - (ss_res / ss_tot)))
    residual_std = float(np.std(y - y_hat, ddof=1)) if len(y) > 2 else 0.0
    corr = float(np.corrcoef(x, y)[0, 1]) if len(x) > 2 else 0.0

    return {
        "alpha": alpha,
        "beta": beta,
        "r2": r2,
        "corr": corr,
        "residual_std": residual_std,
        "matched_bars": int(len(aligned)),
    }


def relationship_read(beta: float, r2: float) -> str:
    if r2 < 0.12:
        return "Loose Link"
    if beta <= -0.35 and r2 >= 0.18:
        return "Hedge vs Anchor"
    if beta >= 1.2 and r2 >= 0.35:
        return "High Beta"
    if beta >= 0.6:
        return "Follows Anchor"
    if beta >= 0.25:
        return "Partial Link"
    return "Loose Link"


def fit_read(r2: float) -> str:
    if r2 >= 0.60:
        return "Strong"
    if r2 >= 0.35:
        return "Usable"
    if r2 >= 0.15:
        return "Weak"
    return "Fragile"


def verdict_read(weighted_abs_beta: float, weighted_r2: float, coverage_pct: float) -> tuple[str, str]:
    if coverage_pct < 60:
        return (
            "Partial Coverage",
            "Several holdings could not be modeled cleanly. Treat the basket projection as incomplete until coverage improves.",
        )
    if weighted_abs_beta >= 0.90 and weighted_r2 >= 0.35:
        return (
            "Anchor-Heavy Response",
            "This basket is tightly tied to the anchor. If the anchor reaches target, most holdings are likely to react in the same theme.",
        )
    if weighted_abs_beta <= 0.40 and weighted_r2 < 0.25:
        return (
            "Loose Response Basket",
            "The basket is only weakly linked to the anchor. Treat the scenario as a rough macro influence check, not a strong projection map.",
        )
    return (
        "Balanced Response",
        "Some names follow the anchor closely while others react more loosely. Use the per-coin fit column before relying on the basket total.",
    )


def _safe_price(df: pd.DataFrame) -> float | None:
    if df is None or df.empty or "close" not in df.columns:
        return None
    try:
        px = float(pd.to_numeric(df["close"], errors="coerce").dropna().iloc[-1])
    except Exception:
        return None
    if not math.isfinite(px) or px <= 0:
        return None
    return px


def _clamp_modeled_return(value: float) -> float:
    if not math.isfinite(value):
        return RETURN_FLOOR
    return max(RETURN_FLOOR, value)


def build_portfolio_scenario(
    holdings: list[dict],
    anchor_symbol: str,
    anchor_target_price: float,
    ohlcv_map: dict[str, pd.DataFrame | None],
) -> dict:
    anchor_df = ohlcv_map.get(anchor_symbol)
    anchor_price = _safe_price(anchor_df)
    if anchor_price is None or anchor_target_price <= 0:
        raise ValueError("Anchor pricing is unavailable.")

    anchor_move = (float(anchor_target_price) / anchor_price) - 1.0
    horizon_info = estimate_anchor_horizon(anchor_df, anchor_move)
    horizon_bars = int(horizon_info["horizon_bars"])
    typical_bar_move = float(horizon_info["typical_bar_move"])
    anchor_series = returns_series(anchor_df, anchor_symbol, horizon_bars=horizon_bars)
    if anchor_series is None:
        raise ValueError("Anchor history is not sufficient for the estimated scenario horizon.")
    rows: list[dict] = []
    current_total = float(sum(item["current_value"] for item in holdings))
    modeled_value = 0.0
    projected_total = 0.0
    weighted_r2 = 0.0
    weighted_abs_beta = 0.0
    skipped_coins: list[str] = []
    capped_coins: list[str] = []

    for item in holdings:
        symbol = item["symbol"]
        df = ohlcv_map.get(symbol)
        current_price = _safe_price(df)
        if current_price is None:
            skipped_coins.append(item["coin"])
            projected_total += item["current_value"]
            continue

        units = item["current_value"] / current_price
        if symbol == anchor_symbol:
            model = {
                "alpha": 0.0,
                "beta": 1.0,
                "r2": 1.0,
                "corr": 1.0,
                "residual_std": 0.0,
                "matched_bars": len(anchor_series) if anchor_series is not None else 0,
            }
            link_read = "Anchor"
            fit = "Direct"
        else:
            coin_series = returns_series(df, symbol, horizon_bars=horizon_bars)
            model = regression_relationship(anchor_series, coin_series)
            if model is None:
                skipped_coins.append(item["coin"])
                projected_total += item["current_value"]
                continue
            link_read = relationship_read(model["beta"], model["r2"])
            fit = fit_read(model["r2"])

        raw_projected_return = float(model["alpha"] + (model["beta"] * anchor_move))
        projected_return = _clamp_modeled_return(raw_projected_return)
        low_return = _clamp_modeled_return(raw_projected_return - model["residual_std"])
        high_return = _clamp_modeled_return(raw_projected_return + model["residual_std"])
        if projected_return != raw_projected_return:
            capped_coins.append(item["coin"])
        projected_price = max(0.0, current_price * (1.0 + projected_return))
        low_price = max(0.0, current_price * (1.0 + low_return))
        high_price = max(0.0, current_price * (1.0 + high_return))
        projected_value = units * projected_price

        modeled_value += item["current_value"]
        projected_total += projected_value
        weighted_r2 += model["r2"] * item["current_value"]
        weighted_abs_beta += abs(model["beta"]) * item["current_value"]

        rows.append(
            {
                "Coin": item["coin"],
                "Current Price ($)": current_price,
                "Current Value ($)": item["current_value"],
                "Link Read": link_read,
                "Beta vs Anchor": model["beta"],
                "Fit": fit,
                "Scenario Return (%)": projected_return * 100.0,
                "Projected Price ($)": projected_price,
                "Projected Value ($)": projected_value,
                "Scenario Low ($)": low_price,
                "Scenario High ($)": high_price,
                "Matched Bars": model["matched_bars"],
                "R2": model["r2"],
            }
        )

    coverage_pct = 0.0 if current_total <= 0 else (modeled_value / current_total) * 100.0
    weighted_r2 = 0.0 if modeled_value <= 0 else weighted_r2 / modeled_value
    weighted_abs_beta = 0.0 if modeled_value <= 0 else weighted_abs_beta / modeled_value
    verdict, verdict_body = verdict_read(weighted_abs_beta, weighted_r2, coverage_pct)

    result_df = pd.DataFrame(rows)
    if not result_df.empty:
        result_df = result_df.sort_values(["Projected Value ($)", "Current Value ($)"], ascending=False).reset_index(drop=True)

    current_total_safe = current_total if current_total > 0 else 1.0
    projected_delta_pct = ((projected_total / current_total_safe) - 1.0) * 100.0

    return {
        "anchor_symbol": anchor_symbol,
        "anchor_price": anchor_price,
        "anchor_target_price": float(anchor_target_price),
        "anchor_move_pct": anchor_move * 100.0,
        "horizon_bars": horizon_bars,
        "raw_horizon_bars": int(horizon_info["raw_horizon_bars"]),
        "horizon_capped": bool(horizon_info["horizon_capped"]),
        "horizon_cap_reason": horizon_info["horizon_cap_reason"],
        "typical_bar_move_pct": typical_bar_move * 100.0,
        "current_total": current_total,
        "projected_total": projected_total,
        "projected_delta_pct": projected_delta_pct,
        "coverage_pct": coverage_pct,
        "weighted_r2": weighted_r2,
        "weighted_abs_beta": weighted_abs_beta,
        "verdict": verdict,
        "verdict_body": verdict_body,
        "skipped_coins": skipped_coins,
        "capped_coins": capped_coins,
        "rows": result_df,
    }
