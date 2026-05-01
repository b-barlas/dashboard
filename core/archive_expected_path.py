"""Archive-based expected path calculations shared outside the UI."""

from __future__ import annotations

from collections.abc import Mapping

import pandas as pd

from core.archive_intelligence import archive_direction_key


def projection_float(value: object, default: float = 0.0) -> float:
    try:
        out = float(value)
    except Exception:
        return float(default)
    if pd.isna(out):
        return float(default)
    return float(out)


def expected_path_read_quality(sample: int) -> str:
    if int(sample) >= 32:
        return "Strong"
    if int(sample) >= 16:
        return "Good"
    return "Thin"


def _direction_key(value: object) -> str:
    return archive_direction_key(value)


def _format_archive_reference_label(ts: object, *, stale: bool) -> str:
    prefix = "last archived price" if stale else "latest archived price"
    parsed = pd.to_datetime(ts, utc=True, errors="coerce")
    if pd.isna(parsed):
        return prefix
    return f"{prefix}, {pd.Timestamp(parsed).strftime('%b %d')}"


def _reference_price_from_events(
    df_events: pd.DataFrame,
    *,
    timeframe: str,
    direction: str,
    now: object | None = None,
    max_age_hours: int = 48,
) -> tuple[float | None, str]:
    if df_events is None or df_events.empty or "price" not in df_events.columns:
        return None, ""
    d = df_events.copy()
    if "timeframe" in d.columns:
        d = d[d["timeframe"].fillna("").astype(str).str.strip().str.lower().eq(str(timeframe).lower())].copy()
    if "direction" in d.columns:
        d["__direction"] = d["direction"].map(_direction_key)
        d = d[d["__direction"].eq(str(direction).upper())].copy()
    d["__price"] = pd.to_numeric(d.get("price"), errors="coerce")
    d = d[d["__price"].notna() & (d["__price"] > 0)].copy()
    if d.empty:
        return None, ""
    if "event_time" in d.columns:
        d["__event_ts"] = pd.to_datetime(d["event_time"], utc=True, errors="coerce")
        d = d.sort_values("__event_ts", ascending=False)
        latest = d.iloc[0]
        ts = latest.get("__event_ts")
        if pd.notna(ts):
            now_ts = pd.to_datetime(now, utc=True, errors="coerce") if now is not None else pd.Timestamp.now(tz="UTC")
            if pd.notna(now_ts):
                age_hours = max(0.0, float((pd.Timestamp(now_ts) - pd.Timestamp(ts)).total_seconds()) / 3600.0)
                return (
                    float(latest["__price"]),
                    _format_archive_reference_label(ts, stale=age_hours > float(max_age_hours)),
                )
    latest_price = float(d.iloc[0]["__price"])
    return latest_price, "latest archived price"


def with_expected_path_reference_price(
    snapshot: Mapping[str, object],
    reference_price: object,
    reference_label: str = "latest close",
) -> dict[str, object]:
    out = dict(snapshot)
    ref_price = projection_float(reference_price, 0.0)
    if ref_price <= 0:
        return out

    direction_key = str(out.get("direction") or "").strip().upper()
    lower_return = abs(projection_float(out.get("best_zone_low_pct"), 0.0))
    upper_return = abs(projection_float(out.get("best_zone_high_pct"), 0.0))
    lower_return, upper_return = sorted([lower_return, upper_return])
    normal_pullback = max(0.0, abs(projection_float(out.get("normal_pullback_pct"), 0.0)))
    caution_pullback = max(normal_pullback, abs(projection_float(out.get("caution_pullback_pct"), normal_pullback)))

    if direction_key == "DOWNSIDE":
        p1 = ref_price * (1.0 - lower_return / 100.0)
        p2 = ref_price * (1.0 - upper_return / 100.0)
        pullback_price = ref_price * (1.0 + normal_pullback / 100.0)
        caution_price = ref_price * (1.0 + caution_pullback / 100.0)
    else:
        p1 = ref_price * (1.0 + lower_return / 100.0)
        p2 = ref_price * (1.0 + upper_return / 100.0)
        pullback_price = ref_price * (1.0 - normal_pullback / 100.0)
        caution_price = ref_price * (1.0 - caution_pullback / 100.0)

    price_low, price_high = sorted([p1, p2])
    out.update(
        {
            "reference_price": ref_price,
            "reference_price_label": str(reference_label or "latest close").strip() or "latest close",
            "price_zone_low": float(price_low),
            "price_zone_high": float(price_high),
            "pullback_price": float(pullback_price),
            "caution_price": float(caution_price),
        }
    )
    return out


def _empty_snapshot(*, symbol_filter: str, timeframe: str = "", direction: str = "") -> dict[str, object]:
    return {
        "available": False,
        "symbol": str(symbol_filter or "").strip().upper(),
        "timeframe": str(timeframe or "").strip().lower(),
        "direction": str(direction or "").strip().upper(),
        "sample": 0,
    }


def _expected_path_snapshot_for_group(
    *,
    events_df: pd.DataFrame,
    windows_df: pd.DataFrame,
    symbol_filter: str,
    timeframe: str,
    direction: str,
    min_samples: int,
    now: object | None = None,
) -> dict[str, object]:
    empty = _empty_snapshot(symbol_filter=symbol_filter, timeframe=timeframe, direction=direction)
    if events_df is None or events_df.empty or windows_df is None or windows_df.empty:
        return empty
    required_event_cols = {"signal_key", "timeframe", "direction"}
    required_window_cols = {"signal_key", "bars_ahead", "directional_return_pct", "adverse_excursion_pct"}
    if not required_event_cols.issubset(events_df.columns) or not required_window_cols.issubset(windows_df.columns):
        return empty

    e = events_df.copy()
    e["signal_key"] = e["signal_key"].fillna("").astype(str).str.strip()
    e["timeframe"] = e["timeframe"].fillna("").astype(str).str.strip().str.lower()
    e["__direction"] = e["direction"].map(_direction_key)
    status = e.get("status", pd.Series(index=e.index, dtype=object)).fillna("").astype(str).str.upper()
    e = e[
        e["signal_key"].ne("")
        & e["timeframe"].eq(str(timeframe).strip().lower())
        & e["__direction"].eq(str(direction).strip().upper())
        & status.eq("RESOLVED")
    ].copy()
    if e.empty:
        return empty

    w = windows_df.copy()
    w["signal_key"] = w["signal_key"].fillna("").astype(str).str.strip()
    w["bars_ahead"] = pd.to_numeric(w["bars_ahead"], errors="coerce")
    w["directional_return_pct"] = pd.to_numeric(w["directional_return_pct"], errors="coerce")
    w["adverse_excursion_pct"] = pd.to_numeric(w["adverse_excursion_pct"], errors="coerce")
    if "favorable_excursion_pct" in w.columns:
        w["favorable_excursion_pct"] = pd.to_numeric(w["favorable_excursion_pct"], errors="coerce")
    w = w[
        w["signal_key"].ne("")
        & w["bars_ahead"].notna()
        & w["directional_return_pct"].notna()
        & w["adverse_excursion_pct"].notna()
    ].copy()
    if w.empty:
        return empty

    merge_cols = ["signal_key", "timeframe", "__direction"]
    if {"price", "event_time"}.issubset(e.columns):
        merge_cols.extend(["price", "event_time"])
    merged = w.merge(e[merge_cols], on="signal_key", how="inner")
    if merged.empty:
        return empty
    sample = int(merged["signal_key"].nunique())
    if sample < int(min_samples):
        out = dict(empty)
        out.update({"sample": sample, "read_quality": "Building"})
        return out

    by_bar_rows: list[dict[str, object]] = []
    for bar, group in merged.groupby("bars_ahead", dropna=True):
        bar_sample = int(group["signal_key"].nunique())
        if bar_sample < max(3, min(int(min_samples), 8)):
            continue
        returns = pd.to_numeric(group["directional_return_pct"], errors="coerce").dropna()
        adverse = pd.to_numeric(group["adverse_excursion_pct"], errors="coerce").dropna()
        if returns.empty:
            continue
        median_return = float(returns.median())
        if median_return <= 0.0:
            continue
        adverse_median = float(adverse.median()) if not adverse.empty else 0.0
        follow_through = float((returns > 0.0).mean() * 100.0)
        path_score = median_return - (0.30 * adverse_median) + ((follow_through - 50.0) / 100.0)
        by_bar_rows.append(
            {
                "bars_ahead": int(bar),
                "sample": bar_sample,
                "median_return": median_return,
                "lower_return": float(returns.quantile(0.40)),
                "upper_return": float(returns.quantile(0.75)),
                "normal_pullback": max(0.0, float(adverse.quantile(0.65)) if not adverse.empty else 0.0),
                "caution_pullback": max(0.0, float(adverse.quantile(0.85)) if not adverse.empty else 0.0),
                "follow_through": follow_through,
                "path_score": path_score,
            }
        )
    if not by_bar_rows:
        out = dict(empty)
        out.update({"sample": sample, "read_quality": expected_path_read_quality(sample)})
        return out

    by_bar_rows.sort(key=lambda row: (float(row["path_score"]), float(row["median_return"]), int(row["sample"])), reverse=True)
    best = by_bar_rows[0]
    best_bar = int(best["bars_ahead"])
    best_median = float(best["median_return"])
    fade_after_bar = 0
    for row in sorted([r for r in by_bar_rows if int(r["bars_ahead"]) > best_bar], key=lambda r: int(r["bars_ahead"])):
        if float(row["median_return"]) <= max(0.05, best_median * 0.65) or float(row["follow_through"]) < 50.0:
            fade_after_bar = int(row["bars_ahead"])
            break

    ref_price, ref_label = _reference_price_from_events(
        merged,
        timeframe=str(timeframe).strip().lower(),
        direction=str(direction).strip().upper(),
        now=now,
    )
    lower_return = float(min(best["lower_return"], best["upper_return"]))
    upper_return = float(max(best["lower_return"], best["upper_return"]))
    normal_pullback = float(best["normal_pullback"])
    caution_pullback = max(normal_pullback, float(best.get("caution_pullback") or normal_pullback))
    best_eval = merged[merged["bars_ahead"].astype(int).eq(best_bar)].copy()
    best_eval["__return"] = pd.to_numeric(best_eval.get("directional_return_pct"), errors="coerce")
    if "favorable_excursion_pct" in best_eval.columns:
        best_eval["__zone_basis"] = pd.to_numeric(best_eval.get("favorable_excursion_pct"), errors="coerce")
    else:
        best_eval["__zone_basis"] = best_eval["__return"]
    best_eval["__zone_basis"] = best_eval["__zone_basis"].fillna(best_eval["__return"])
    best_eval["__adverse"] = pd.to_numeric(best_eval.get("adverse_excursion_pct"), errors="coerce")
    best_eval = best_eval.dropna(subset=["__zone_basis", "__adverse"])
    archive_check_sample = int(best_eval["signal_key"].nunique()) if not best_eval.empty else 0
    if archive_check_sample > 0:
        zone_reached = best_eval["__zone_basis"] >= abs(lower_return)
        clean_path = zone_reached & (best_eval["__adverse"] <= normal_pullback)
        caution_broken = best_eval["__adverse"] > caution_pullback
        zone_hit_rate = float(zone_reached.mean() * 100.0)
        clean_path_rate = float(clean_path.mean() * 100.0)
        caution_break_rate = float(caution_broken.mean() * 100.0)
    else:
        zone_hit_rate = 0.0
        clean_path_rate = 0.0
        caution_break_rate = 0.0

    score = (
        float(best["path_score"])
        + min(1.5, sample / 24.0)
        + (0.25 if expected_path_read_quality(sample) == "Strong" else 0.0)
        + ((clean_path_rate - 50.0) / 120.0)
        - (caution_break_rate / 160.0)
    )
    snapshot = {
        "available": True,
        "symbol": str(symbol_filter or "").strip().upper(),
        "timeframe": str(timeframe or "").strip().lower(),
        "direction": str(direction or "").strip().upper(),
        "sample": sample,
        "read_quality": expected_path_read_quality(sample),
        "best_bar": best_bar,
        "fade_after_bar": fade_after_bar,
        "best_zone_low_pct": lower_return,
        "best_zone_high_pct": upper_return,
        "normal_pullback_pct": normal_pullback,
        "caution_pullback_pct": caution_pullback,
        "follow_through_pct": float(best["follow_through"]),
        "archive_check_sample": archive_check_sample,
        "zone_hit_rate_pct": zone_hit_rate,
        "clean_path_rate_pct": clean_path_rate,
        "caution_break_rate_pct": caution_break_rate,
        "score": float(score),
        "reference_price": ref_price,
        "reference_price_label": ref_label,
        "price_zone_low": None,
        "price_zone_high": None,
        "pullback_price": None,
        "caution_price": None,
    }
    return with_expected_path_reference_price(snapshot, ref_price, ref_label)


def build_archive_expected_path_projection(
    *,
    df_events: pd.DataFrame,
    df_forward_windows: pd.DataFrame,
    symbol_filter: str,
    timeframe_filter: str,
    timeframe_order: tuple[str, ...] = ("5m", "15m", "1h", "4h", "1d"),
    min_samples: int = 8,
    now: object | None = None,
) -> dict[str, object]:
    empty = _empty_snapshot(symbol_filter=symbol_filter)
    if df_events is None or df_events.empty or df_forward_windows is None or df_forward_windows.empty:
        return empty
    if "timeframe" not in df_events.columns or "direction" not in df_events.columns:
        return empty
    timeframe_text = str(timeframe_filter or "").strip().lower()
    available_timeframes = set(df_events["timeframe"].fillna("").astype(str).str.strip().str.lower().tolist())
    available_timeframes.discard("")
    if timeframe_text and timeframe_text != "all":
        timeframes = [timeframe_text]
    else:
        extras = sorted(available_timeframes.difference(timeframe_order))
        timeframes = [tf for tf in timeframe_order if not available_timeframes or tf in available_timeframes]
        timeframes.extend(extras)
    snapshots: list[dict[str, object]] = []
    for timeframe in timeframes:
        for direction in ("UPSIDE", "DOWNSIDE"):
            snap = _expected_path_snapshot_for_group(
                events_df=df_events,
                windows_df=df_forward_windows,
                symbol_filter=symbol_filter,
                timeframe=timeframe,
                direction=direction,
                min_samples=min_samples,
                now=now,
            )
            if bool(snap.get("available")):
                snapshots.append(snap)
    if not snapshots:
        return empty
    snapshots.sort(
        key=lambda snap: (
            float(snap.get("score") or 0.0),
            float(snap.get("follow_through_pct") or 0.0),
            int(snap.get("sample") or 0),
        ),
        reverse=True,
    )
    primary = dict(snapshots[0])
    alternate = dict(snapshots[1]) if len(snapshots) > 1 else {}
    if alternate:
        primary["alternate_path"] = {
            "timeframe": str(alternate.get("timeframe") or ""),
            "direction": str(alternate.get("direction") or ""),
            "score": float(alternate.get("score") or 0.0),
            "sample": int(alternate.get("sample") or 0),
        }
        primary_score = abs(float(primary.get("score") or 0.0))
        alt_score = float(alternate.get("score") or 0.0)
        primary["path_conflict"] = bool(primary_score > 0 and alt_score >= primary_score * 0.92)
    else:
        primary["alternate_path"] = {}
        primary["path_conflict"] = False
    return primary
