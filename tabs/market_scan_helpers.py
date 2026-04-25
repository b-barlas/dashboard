from __future__ import annotations

import math

import pandas as pd

from core.symbols import canonical_base_symbol, is_stable_base_symbol

SCAN_MODE_BROAD = "Broad Market"
SCAN_MODE_ACTIONABLE = "Actionable Setups"
SCAN_MODE_EMERGING = "Breakout Radar"


def _sortable_float(value: object) -> float:
    try:
        return float(value or 0.0)
    except Exception:
        return 0.0


def _canonical_pair_base(symbol: str) -> str:
    raw = str(symbol or "").strip()
    base = raw.split("/", 1)[0] if "/" in raw else raw
    return canonical_base_symbol(base)


def _signal_tracker_direction_key(value: object) -> str:
    d = str(value or "").strip().upper()
    if d in {"UPSIDE", "LONG", "BUY", "BULLISH"}:
        return "UPSIDE"
    if d in {"DOWNSIDE", "SHORT", "SELL", "BEARISH"}:
        return "DOWNSIDE"
    return "NEUTRAL"


def _normalize_scan_mode(value: object) -> str:
    raw = str(value or "").strip().lower()
    if raw in {SCAN_MODE_ACTIONABLE.lower(), "setup hunter"}:
        return SCAN_MODE_ACTIONABLE
    if raw in {SCAN_MODE_EMERGING.lower(), "early momentum", "emerging", "emerging movers", "breakout radar"}:
        return SCAN_MODE_EMERGING
    return SCAN_MODE_BROAD


def _valid_market_bases(market_rows: list[dict]) -> set[str]:
    out: set[str] = set()
    for row in market_rows:
        base = canonical_base_symbol((row or {}).get("symbol") or "")
        if base:
            out.add(base)
    return out


def _filter_scan_symbols(usdt_symbols: list[str], market_rows: list[dict]) -> list[str]:
    valid_bases = _valid_market_bases(market_rows)
    if not valid_bases:
        return list(usdt_symbols)
    return [pair for pair in usdt_symbols if _canonical_pair_base(pair) in valid_bases]


def _market_row_pct_change(row: dict) -> float:
    for key in ("price_change_percentage_24h", "price_change_percentage_24h_in_currency"):
        try:
            value = float((row or {}).get(key) or 0.0)
        except Exception:
            value = 0.0
        if math.isfinite(value):
            return value
    return 0.0


def _dedupe_market_rows(market_data: list[dict]) -> list[dict]:
    seen_symbols: set[str] = set()
    unique_market_data: list[dict] = []
    for coin in market_data:
        coin_id = (coin.get("id") or "").lower()
        symbol = (coin.get("symbol") or "").upper()
        if not symbol:
            continue
        if "wrapped" in coin_id:
            continue
        if symbol in seen_symbols:
            continue
        seen_symbols.add(symbol)
        unique_market_data.append(coin)
    return unique_market_data


def _actionable_universe_profile(timeframe: str) -> tuple[float, float, float, float, float, float]:
    tf = str(timeframe or "").strip().lower()
    if tf == "1m":
        return (0.34, 0.52, 0.14, 0.8, 6.8, 15.5)
    if tf == "3m":
        return (0.35, 0.51, 0.14, 0.75, 6.4, 14.8)
    if tf == "5m":
        return (0.36, 0.50, 0.14, 0.7, 6.0, 14.0)
    if tf == "15m":
        return (0.38, 0.46, 0.16, 0.8, 5.0, 13.0)
    if tf == "1h":
        return (0.40, 0.40, 0.20, 0.6, 4.0, 10.5)
    if tf == "4h":
        return (0.44, 0.28, 0.28, 0.6, 3.5, 10.0)
    if tf == "1d":
        return (0.46, 0.22, 0.32, 0.5, 2.8, 8.0)
    return (0.42, 0.36, 0.22, 0.7, 4.2, 11.0)


def _actionable_aligned_move(
    pct_change_24h: float,
    *,
    direction_filter: str,
) -> tuple[float, float]:
    pct = float(pct_change_24h or 0.0)
    direction = str(direction_filter or "").strip().upper()
    if direction == "UPSIDE":
        return max(pct, 0.0), max(-pct, 0.0)
    if direction == "DOWNSIDE":
        return max(-pct, 0.0), max(pct, 0.0)
    move = abs(pct)
    return move, 0.0


def _actionable_universe_movement_score(
    pct_change_24h: float,
    *,
    timeframe: str,
    direction_filter: str,
) -> float:
    _volume_w, _move_w, _mcap_w, min_move, sweet_move, stretch_move = _actionable_universe_profile(timeframe)
    aligned_move, counter_move = _actionable_aligned_move(
        pct_change_24h,
        direction_filter=direction_filter,
    )
    if aligned_move < min_move:
        score = 0.16 + 0.22 * (aligned_move / max(min_move, 1e-9))
    elif aligned_move < sweet_move:
        score = 0.35 + 0.55 * ((aligned_move - min_move) / max(sweet_move - min_move, 1e-9))
    elif aligned_move <= stretch_move:
        score = 1.0 - 0.30 * ((aligned_move - sweet_move) / max(stretch_move - sweet_move, 1e-9))
    else:
        score = 0.55 - min(0.30, 0.02 * (aligned_move - stretch_move))
    if counter_move > 0:
        score -= min(0.45, 0.12 + 0.04 * counter_move)
    return max(0.0, min(1.0, score))


def _actionable_universe_market_cap_score(market_cap: object) -> float:
    mcap = _sortable_float(market_cap)
    if mcap >= 20_000_000_000:
        return 1.0
    if mcap >= 5_000_000_000:
        return 0.9
    if mcap >= 1_000_000_000:
        return 0.78
    if mcap >= 300_000_000:
        return 0.62
    if mcap >= 100_000_000:
        return 0.45
    if mcap > 0:
        return 0.2
    return 0.1


def _emerging_universe_profile(timeframe: str) -> tuple[float, float, float, float, float, float]:
    tf = str(timeframe or "").strip().lower()
    if tf == "1m":
        return (0.20, 0.66, 0.06, 0.4, 3.2, 9.2)
    if tf == "3m":
        return (0.20, 0.66, 0.06, 0.4, 3.2, 9.0)
    if tf == "5m":
        return (0.22, 0.62, 0.06, 0.35, 2.8, 8.5)
    if tf == "15m":
        return (0.20, 0.62, 0.08, 0.4, 3.0, 8.8)
    if tf == "1h":
        return (0.20, 0.62, 0.08, 0.45, 3.2, 9.0)
    if tf == "4h":
        return (0.24, 0.54, 0.10, 0.45, 2.8, 8.0)
    if tf == "1d":
        return (0.26, 0.48, 0.12, 0.4, 2.4, 6.5)
    return (0.20, 0.62, 0.08, 0.4, 3.0, 8.5)


def _emerging_universe_movement_score(
    pct_change_24h: float,
    *,
    timeframe: str,
    direction_filter: str,
) -> float:
    _volume_w, _move_w, _mcap_w, min_move, sweet_move, stretch_move = _emerging_universe_profile(timeframe)
    aligned_move, counter_move = _actionable_aligned_move(
        pct_change_24h,
        direction_filter=direction_filter,
    )
    if aligned_move < min_move:
        score = 0.18 + 0.28 * (aligned_move / max(min_move, 1e-9))
    elif aligned_move < sweet_move:
        score = 0.46 + 0.46 * ((aligned_move - min_move) / max(sweet_move - min_move, 1e-9))
    elif aligned_move <= stretch_move:
        score = 0.92 - 0.22 * ((aligned_move - sweet_move) / max(stretch_move - sweet_move, 1e-9))
    else:
        score = 0.62 - min(0.30, 0.015 * (aligned_move - stretch_move))
    if counter_move > 0:
        score -= min(0.38, 0.10 + 0.035 * counter_move)
    return max(0.0, min(1.0, score))


def _emerging_universe_market_cap_score(market_cap: object) -> float:
    mcap = _sortable_float(market_cap)
    if mcap >= 20_000_000_000:
        return 1.0
    if mcap >= 5_000_000_000:
        return 0.95
    if mcap >= 1_000_000_000:
        return 0.88
    if mcap >= 300_000_000:
        return 0.76
    if mcap >= 100_000_000:
        return 0.62
    if mcap > 0:
        return 0.45
    return 0.25


def _breakout_late_chase_penalty(
    *,
    timeframe: str,
    aligned_move: float,
    radar_freshness_score: float = 0.0,
    radar_memory_score: float = 0.0,
) -> float:
    _volume_w, _move_w, _mcap_w, _min_move, _sweet_move, stretch_move = _emerging_universe_profile(timeframe)
    move = max(0.0, _sortable_float(aligned_move))
    if move <= stretch_move:
        return 0.0
    extension = move - stretch_move
    base_penalty = min(13.0, 2.5 + 0.72 * extension)
    if move >= stretch_move * 1.85:
        base_penalty += min(4.0, 0.22 * (move - stretch_move * 1.85))
    freshness_shield = max(
        _sortable_float(radar_freshness_score),
        _sortable_float(radar_memory_score),
    )
    shield = min(0.70, max(0.0, freshness_shield) * 0.70)
    return max(0.0, min(16.0, base_penalty * (1.0 - shield)))


def _breakout_row_base(row: dict) -> str:
    return canonical_base_symbol((row or {}).get("symbol") or (row or {}).get("Coin") or "")


def _breakout_row_quote_volume(row: dict) -> float:
    return max(
        _sortable_float((row or {}).get("_quote_volume_24h")),
        _sortable_float((row or {}).get("total_volume")),
        _sortable_float((row or {}).get("_volume_24h")),
        _sortable_float((row or {}).get("quote_volume_24h")),
    )


def _breakout_memory_history_rows(history_rows: object) -> list[dict]:
    if history_rows is None:
        return []
    if isinstance(history_rows, pd.DataFrame):
        if history_rows.empty:
            return []
        return [dict(row) for row in history_rows.to_dict("records")]
    if isinstance(history_rows, list):
        return [dict(row) for row in history_rows if isinstance(row, dict)]
    return []


def _breakout_memory_score_for_row(
    current_row: dict,
    history_rows: list[dict],
    *,
    direction_filter: str,
) -> dict[str, float]:
    if not history_rows:
        return {}
    current_source = _sortable_float((current_row or {}).get("_radar_source_score"))
    current_fresh = _sortable_float((current_row or {}).get("_radar_freshness_score"))
    current_move, _current_counter = _actionable_aligned_move(
        _market_row_pct_change(current_row),
        direction_filter=direction_filter,
    )
    current_volume = _breakout_row_quote_volume(current_row)

    prior = history_rows[0]
    prior_source = _sortable_float(prior.get("radar_source_score") or prior.get("_radar_source_score"))
    prior_fresh = _sortable_float(prior.get("radar_freshness_score") or prior.get("_radar_freshness_score"))
    prior_move_raw = (
        prior.get("pct_change_24h")
        if "pct_change_24h" in prior
        else prior.get("price_change_percentage_24h")
    )
    prior_move, _prior_counter = _actionable_aligned_move(
        _sortable_float(prior_move_raw),
        direction_filter=direction_filter,
    )
    prior_volume = _sortable_float(
        prior.get("quote_volume_24h")
        or prior.get("_quote_volume_24h")
        or prior.get("total_volume")
        or prior.get("_volume_24h")
    )

    source_jump = min(1.0, max(0.0, current_source - prior_source) / 0.30)
    freshness_jump = min(1.0, max(0.0, current_fresh - prior_fresh) / 0.35)
    move_jump = min(1.0, max(0.0, current_move - prior_move) / 4.0)
    volume_jump = 0.0
    if current_volume > 0.0 and prior_volume > 0.0:
        ratio = current_volume / max(prior_volume, 1e-9)
        if ratio > 1.12:
            volume_jump = min(1.0, math.log(max(ratio, 1.0)) / math.log(4.0))

    pressure_hits = sum(
        1
        for row in history_rows[:6]
        if max(
            _sortable_float(row.get("radar_source_score") or row.get("_radar_source_score")),
            _sortable_float(row.get("radar_freshness_score") or row.get("_radar_freshness_score")),
        )
        >= 0.42
    )
    persistence = min(1.0, float(pressure_hits) / 3.0)
    current_pressure = min(1.0, 0.52 * current_source + 0.48 * current_fresh)
    score = (
        0.24 * source_jump
        + 0.22 * freshness_jump
        + 0.20 * volume_jump
        + 0.18 * move_jump
        + 0.10 * persistence
        + 0.06 * current_pressure
    )
    score = max(0.0, min(1.0, score))
    if score <= 0.04:
        return {}
    return {
        "_radar_memory_score": score,
        "_radar_memory_source_jump": source_jump,
        "_radar_memory_freshness_jump": freshness_jump,
        "_radar_memory_volume_jump": volume_jump,
        "_radar_memory_move_jump": move_jump,
        "_radar_memory_hits": float(pressure_hits),
    }


def _apply_breakout_memory_to_market_rows(
    market_rows: list[dict],
    history_rows: object,
    *,
    direction_filter: str,
) -> list[dict]:
    history_list = _breakout_memory_history_rows(history_rows)
    if not market_rows or not history_list:
        return list(market_rows or [])

    history_by_base: dict[str, list[dict]] = {}
    for row in history_list:
        base = _breakout_row_base(row)
        if not base:
            continue
        history_by_base.setdefault(base, []).append(row)

    for rows in history_by_base.values():
        rows.sort(key=lambda row: str(row.get("observed_at") or ""), reverse=True)

    enriched: list[dict] = []
    for row in list(market_rows or []):
        out = dict(row)
        base = _breakout_row_base(out)
        memory = _breakout_memory_score_for_row(
            out,
            history_by_base.get(base, []),
            direction_filter=direction_filter,
        )
        out.update(memory)
        enriched.append(out)
    return enriched


def _build_breakout_archive_feedback_map(
    events_df: pd.DataFrame,
    *,
    timeframe: str,
    direction_filter: str,
    min_resolved: int = 3,
) -> dict[str, dict[str, float]]:
    if events_df is None or events_df.empty:
        return {}
    required = {"symbol", "timeframe", "directional_return_pct"}
    if not required.issubset(set(events_df.columns)):
        return {}
    d = events_df.copy()
    if "scan_focus" in d.columns:
        focus_mode = d["scan_focus"].fillna("").astype(str).map(_normalize_scan_mode)
        d = d[focus_mode.eq(SCAN_MODE_EMERGING)]
    if d.empty:
        return {}
    tf_key = str(timeframe or "").strip().lower()
    if tf_key:
        d = d[d["timeframe"].fillna("").astype(str).str.strip().str.lower().eq(tf_key)]
    direction_key = str(direction_filter or "").strip().upper()
    if direction_key in {"UPSIDE", "DOWNSIDE"} and "direction" in d.columns:
        d = d[d["direction"].fillna("").astype(str).str.strip().str.upper().eq(direction_key)]
    if d.empty:
        return {}
    d["__base"] = d["symbol"].fillna("").astype(str).map(canonical_base_symbol)
    d["__dir_return"] = pd.to_numeric(d["directional_return_pct"], errors="coerce")
    d = d[d["__base"].astype(bool) & d["__dir_return"].notna()].copy()
    if d.empty:
        return {}

    out: dict[str, dict[str, float]] = {}
    for base, group in d.groupby("__base"):
        resolved = int(len(group))
        if resolved <= 0:
            continue
        returns = pd.to_numeric(group["__dir_return"], errors="coerce").dropna()
        if returns.empty:
            continue
        follow_through_pct = float((returns > 0).mean() * 100.0)
        avg_dir_return_pct = float(returns.mean())
        if resolved < int(min_resolved):
            edge_score = 0.0
        else:
            sample_strength = min(1.0, max(0.0, (resolved - int(min_resolved) + 1) / 22.0))
            follow_edge = max(-1.0, min(1.0, (follow_through_pct - 50.0) / 35.0))
            return_edge = max(-1.0, min(1.0, avg_dir_return_pct / 4.0))
            edge_score = sample_strength * ((0.65 * follow_edge) + (0.35 * return_edge))
            edge_score = max(-1.0, min(1.0, edge_score))
        out[str(base)] = {
            "radar_archive_edge_score": float(edge_score),
            "radar_archive_resolved": float(resolved),
            "radar_archive_follow_through_pct": follow_through_pct,
            "radar_archive_avg_dir_return_pct": avg_dir_return_pct,
        }
    return out


def _apply_breakout_archive_feedback_to_market_rows(
    market_rows: list[dict],
    feedback_map: dict[str, dict[str, float]],
) -> list[dict]:
    if not market_rows or not feedback_map:
        return list(market_rows or [])
    enriched: list[dict] = []
    for row in list(market_rows or []):
        out = dict(row)
        base = _breakout_row_base(out)
        feedback = feedback_map.get(base)
        if feedback:
            out["_radar_archive_edge_score"] = float(feedback.get("radar_archive_edge_score") or 0.0)
            out["_radar_archive_resolved"] = float(feedback.get("radar_archive_resolved") or 0.0)
            out["_radar_archive_follow_through_pct"] = float(
                feedback.get("radar_archive_follow_through_pct") or 0.0
            )
            out["_radar_archive_avg_dir_return_pct"] = float(
                feedback.get("radar_archive_avg_dir_return_pct") or 0.0
            )
        enriched.append(out)
    return enriched


def _emerging_sector_bias_weight(timeframe: str) -> float:
    tf = str(timeframe or "").strip().lower()
    if tf in {"4h", "1d"}:
        return 0.10
    if tf in {"1m", "3m", "5m", "15m"}:
        return 0.08
    return 0.09


def _emerging_exploration_ratio(timeframe: str) -> float:
    tf = str(timeframe or "").strip().lower()
    if tf in {"1m", "3m"}:
        return 0.42
    if tf in {"5m", "15m"}:
        return 0.36
    if tf == "1h":
        return 0.28
    if tf in {"4h", "1d"}:
        return 0.18
    return 0.24


def _actionable_sector_bias_weight(timeframe: str) -> float:
    tf = str(timeframe or "").strip().lower()
    if tf in {"4h", "1d"}:
        return 0.16
    if tf in {"1m", "3m", "5m", "15m"}:
        return 0.10
    return 0.13


def _actionable_exploration_ratio(timeframe: str) -> float:
    tf = str(timeframe or "").strip().lower()
    if tf in {"1m", "3m"}:
        return 0.34
    if tf in {"5m", "15m"}:
        return 0.28
    if tf == "1h":
        return 0.20
    if tf in {"4h", "1d"}:
        return 0.12
    return 0.18


def _actionable_sector_bias_scores(
    market_rows: list[dict],
    *,
    direction_filter: str,
    classify_symbol_sector=None,
) -> dict[str, float]:
    if not callable(classify_symbol_sector) or not market_rows:
        return {}
    sector_totals: dict[str, float] = {}
    sector_counts: dict[str, int] = {}
    direction_key = str(direction_filter or "").strip().upper()
    for row in market_rows:
        base = canonical_base_symbol((row or {}).get("symbol") or "")
        if not base:
            continue
        sector = str(classify_symbol_sector(base) or "").strip()
        if not sector or sector.lower() in {"unknown", "other"}:
            continue
        pct = _market_row_pct_change(row)
        if direction_key == "UPSIDE":
            aligned = max(pct, 0.0) - 0.45 * max(-pct, 0.0)
        elif direction_key == "DOWNSIDE":
            aligned = max(-pct, 0.0) - 0.45 * max(pct, 0.0)
        else:
            aligned = abs(pct)
        sector_totals[sector] = sector_totals.get(sector, 0.0) + float(aligned)
        sector_counts[sector] = sector_counts.get(sector, 0) + 1
    if not sector_totals:
        return {}
    averages = {
        sector: float(total) / max(1, int(sector_counts.get(sector, 0)))
        for sector, total in sector_totals.items()
    }
    low = min(averages.values())
    high = max(averages.values())
    if math.isclose(low, high, rel_tol=1e-9, abs_tol=1e-9):
        return {sector: 0.5 for sector in averages}
    return {
        sector: max(0.0, min(1.0, (value - low) / max(high - low, 1e-9)))
        for sector, value in averages.items()
    }


def _actionable_adx_score(adx_val: float) -> float:
    adx = _sortable_float(adx_val)
    if adx <= 10.0:
        return 0.0
    if adx >= 30.0:
        return 100.0
    return max(0.0, min(100.0, ((adx - 10.0) / 20.0) * 100.0))


def _actionable_rr_score(rr_ratio: float | None) -> float:
    rr = _sortable_float(rr_ratio)
    if rr <= 1.0:
        return 0.0
    if rr >= 2.5:
        return 100.0
    return max(0.0, min(100.0, ((rr - 1.0) / 1.5) * 100.0))


def _actionable_spike_adjustment(
    *,
    signal_direction: str,
    volume_spike: bool,
    spike_dir: str,
) -> float:
    if not bool(volume_spike):
        return 0.0
    direction_key = _signal_tracker_direction_key(signal_direction)
    spike_key = str(spike_dir or "").strip().upper()
    if direction_key == "UPSIDE":
        if spike_key == "UP":
            return 6.0
        if spike_key == "DOWN":
            return -4.0
    if direction_key == "DOWNSIDE":
        if spike_key == "DOWN":
            return 6.0
        if spike_key == "UP":
            return -4.0
    return 1.5


def _actionable_delta_profile(timeframe: str) -> tuple[float, float, float]:
    tf = str(timeframe or "").strip().lower()
    if tf == "5m":
        return (0.12, 0.65, 1.8)
    if tf == "15m":
        return (0.18, 0.95, 2.4)
    if tf == "4h":
        return (0.45, 2.4, 6.5)
    if tf == "1d":
        return (0.9, 4.5, 11.0)
    return (0.28, 1.45, 3.8)


def _actionable_delta_score(
    *,
    delta_pct: float | None,
    signal_direction: str,
    timeframe: str,
) -> float:
    raw_delta = _sortable_float(delta_pct)
    direction_key = _signal_tracker_direction_key(signal_direction)
    if direction_key == "UPSIDE":
        aligned = max(raw_delta, 0.0)
        counter = max(-raw_delta, 0.0)
    elif direction_key == "DOWNSIDE":
        aligned = max(-raw_delta, 0.0)
        counter = max(raw_delta, 0.0)
    else:
        aligned = abs(raw_delta)
        counter = 0.0
    min_move, sweet_move, stretch_move = _actionable_delta_profile(timeframe)
    if aligned < min_move:
        score = 15.0 * (aligned / max(min_move, 1e-9))
    elif aligned < sweet_move:
        score = 30.0 + 55.0 * ((aligned - min_move) / max(sweet_move - min_move, 1e-9))
    elif aligned <= stretch_move:
        score = 85.0 - 25.0 * ((aligned - sweet_move) / max(stretch_move - sweet_move, 1e-9))
    else:
        score = 52.0 - min(22.0, 2.0 * (aligned - stretch_move))
    if counter > 0:
        score -= min(30.0, 8.0 + 6.0 * counter)
    return max(0.0, min(100.0, score))


def _actionable_range_position_score(
    *,
    close_position: float,
    signal_direction: str,
) -> float:
    pos = max(0.0, min(1.0, _sortable_float(close_position)))
    direction_key = _signal_tracker_direction_key(signal_direction)
    if direction_key == "UPSIDE":
        ideal = 0.82
        edge_penalty = 14.0 if pos >= 0.985 else 0.0
    elif direction_key == "DOWNSIDE":
        ideal = 0.18
        edge_penalty = 14.0 if pos <= 0.015 else 0.0
    else:
        ideal = 0.50
        edge_penalty = 0.0
    distance = abs(pos - ideal)
    score = 100.0 * max(0.0, 1.0 - (distance / max(ideal, 1.0 - ideal, 1e-9)))
    return max(0.0, min(100.0, score - edge_penalty))


def _actionable_compression_score(
    *,
    recent_range_pct: float,
    base_range_pct: float,
) -> float:
    recent = max(0.0, _sortable_float(recent_range_pct))
    base = max(0.0, _sortable_float(base_range_pct))
    if recent <= 0.0 or base <= 0.0:
        return 0.0
    ratio = recent / max(base, 1e-9)
    if ratio <= 0.18:
        score = 18.0 + 30.0 * (ratio / 0.18)
    elif ratio <= 0.72:
        score = 48.0 + 42.0 * ((ratio - 0.18) / 0.54)
    elif ratio <= 1.20:
        score = 90.0 - 20.0 * ((ratio - 0.72) / 0.48)
    elif ratio <= 2.10:
        score = 70.0 - 38.0 * ((ratio - 1.20) / 0.90)
    else:
        score = max(12.0, 28.0 - 6.0 * (ratio - 2.10))
    return max(0.0, min(100.0, score))


def _actionable_volume_impulse_score(
    *,
    volume_ratio: float,
    delta_pct: float,
    signal_direction: str,
) -> float:
    ratio = max(0.0, _sortable_float(volume_ratio))
    raw_delta = _sortable_float(delta_pct)
    direction_key = _signal_tracker_direction_key(signal_direction)
    if direction_key == "UPSIDE":
        aligned = raw_delta > 0.0
    elif direction_key == "DOWNSIDE":
        aligned = raw_delta < 0.0
    else:
        aligned = abs(raw_delta) > 0.0
    if ratio <= 0.55:
        score = 18.0
    elif ratio <= 1.0:
        score = 18.0 + 34.0 * ((ratio - 0.55) / 0.45)
    elif ratio <= 1.8:
        score = 52.0 + 34.0 * ((ratio - 1.0) / 0.8)
    else:
        score = min(96.0, 86.0 + 6.0 * min(2.0, ratio - 1.8))
    if aligned:
        score += 8.0
    else:
        score -= 18.0
    return max(0.0, min(100.0, score))


def _actionable_frame_hunt_score(
    *,
    df_eval: pd.DataFrame | None,
    timeframe: str,
    direction_filter: str,
) -> float:
    if df_eval is None or len(df_eval) < 26:
        return 0.0
    try:
        close = pd.to_numeric(df_eval["close"], errors="coerce")
        high = pd.to_numeric(df_eval["high"], errors="coerce")
        low = pd.to_numeric(df_eval["low"], errors="coerce")
    except Exception:
        return 0.0
    if close.isna().all() or high.isna().all() or low.isna().all():
        return 0.0

    last_close = _sortable_float(close.iloc[-1])
    prev_close = _sortable_float(close.iloc[-2])
    if last_close <= 0.0 or prev_close <= 0.0:
        return 0.0
    delta_pct = ((last_close / prev_close) - 1.0) * 100.0

    recent_window = min(12, len(df_eval))
    base_window = min(32, len(df_eval))
    recent_high = _sortable_float(high.tail(recent_window).max())
    recent_low = _sortable_float(low.tail(recent_window).min())
    base_high = _sortable_float(high.tail(base_window).max())
    base_low = _sortable_float(low.tail(base_window).min())
    recent_range = max(0.0, recent_high - recent_low)
    base_range = max(0.0, base_high - base_low)
    close_position = (last_close - recent_low) / recent_range if recent_range > 0.0 else 0.5
    recent_range_pct = (recent_range / last_close) * 100.0 if last_close > 0.0 else 0.0
    base_range_pct = (base_range / last_close) * 100.0 if last_close > 0.0 else 0.0

    volume_ratio = 1.0
    try:
        volume = pd.to_numeric(df_eval["volume"], errors="coerce")
        if not volume.isna().all():
            last_vol = _sortable_float(volume.iloc[-1])
            base_vol = (
                _sortable_float(volume.iloc[-21:-1].mean())
                if len(volume) >= 21
                else _sortable_float(volume.iloc[:-1].mean())
            )
            if last_vol > 0.0 and base_vol > 0.0:
                volume_ratio = last_vol / max(base_vol, 1e-9)
    except Exception:
        volume_ratio = 1.0

    def _score_for(signal_direction: str) -> float:
        return (
            0.34
            * _actionable_delta_score(
                delta_pct=delta_pct,
                signal_direction=signal_direction,
                timeframe=timeframe,
            )
            + 0.24
            * _actionable_range_position_score(
                close_position=close_position,
                signal_direction=signal_direction,
            )
            + 0.22
            * _actionable_compression_score(
                recent_range_pct=recent_range_pct,
                base_range_pct=base_range_pct,
            )
            + 0.20
            * _actionable_volume_impulse_score(
                volume_ratio=volume_ratio,
                delta_pct=delta_pct,
                signal_direction=signal_direction,
            )
        )

    direction_key = str(direction_filter or "").strip().upper()
    if direction_key == "UPSIDE":
        return max(0.0, min(100.0, _score_for("UPSIDE")))
    if direction_key == "DOWNSIDE":
        return max(0.0, min(100.0, _score_for("DOWNSIDE")))
    return max(0.0, min(100.0, max(_score_for("UPSIDE"), _score_for("DOWNSIDE"))))


def _actionable_label_context_score(
    *,
    timeframe: str,
    signal_direction: str,
    volatility_label: str,
    vwap_label: str,
    bollinger_bias: str,
) -> float:
    score = 50.0
    tf = str(timeframe or "").strip().lower()
    vol_key = str(volatility_label or "").strip()
    if "Moderate" in vol_key:
        score += 12.0
    elif "High" in vol_key:
        score += 10.0 if tf in {"5m", "15m"} else 4.0
    elif "Low" in vol_key:
        score += 6.0 if tf in {"4h", "1d"} else -4.0

    direction_key = _signal_tracker_direction_key(signal_direction)
    vwap_key = str(vwap_label or "").strip().upper()
    boll_key = str(bollinger_bias or "").strip().upper()

    if direction_key == "UPSIDE":
        if "ABOVE" in vwap_key:
            score += 12.0
        elif "NEAR VWAP" in vwap_key:
            score += 4.0
        elif "BELOW" in vwap_key:
            score -= 10.0

        if "OVERSOLD" in boll_key:
            score += 14.0
        elif "NEAR BOTTOM" in boll_key:
            score += 8.0
        elif "NEAR TOP" in boll_key:
            score -= 4.0
        elif "OVERBOUGHT" in boll_key:
            score -= 12.0
    elif direction_key == "DOWNSIDE":
        if "BELOW" in vwap_key:
            score += 12.0
        elif "NEAR VWAP" in vwap_key:
            score += 4.0
        elif "ABOVE" in vwap_key:
            score -= 10.0

        if "OVERBOUGHT" in boll_key:
            score += 14.0
        elif "NEAR TOP" in boll_key:
            score += 8.0
        elif "NEAR BOTTOM" in boll_key:
            score -= 4.0
        elif "OVERSOLD" in boll_key:
            score -= 12.0
    return max(0.0, min(100.0, score))


def _actionable_setup_score(
    *,
    timeframe: str,
    execution_structure_quality: float,
    execution_trend_quality: float,
    execution_regime_quality: float,
    execution_location_quality: float,
    trend_led_score: float,
    ai_led_score: float,
    rr_ratio: float | None,
    adx_val: float,
    delta_pct: float | None,
    volatility_label: str,
    vwap_label: str,
    bollinger_bias: str,
    signal_direction: str,
    volume_spike: bool,
    spike_dir: str,
    frame_hunt_score: float | None = None,
) -> float:
    base_score = (
        0.18 * _sortable_float(execution_structure_quality)
        + 0.16 * _sortable_float(execution_trend_quality)
        + 0.09 * _sortable_float(execution_regime_quality)
        + 0.15 * _sortable_float(execution_location_quality)
        + 0.08 * _actionable_adx_score(adx_val)
        + 0.10 * _actionable_rr_score(rr_ratio)
        + 0.07 * _sortable_float(trend_led_score)
        + 0.05 * _sortable_float(ai_led_score)
        + 0.12 * _actionable_delta_score(
            delta_pct=delta_pct,
            signal_direction=signal_direction,
            timeframe=timeframe,
        )
        + 0.10 * _actionable_label_context_score(
            timeframe=timeframe,
            signal_direction=signal_direction,
            volatility_label=volatility_label,
            vwap_label=vwap_label,
            bollinger_bias=bollinger_bias,
        )
    )
    score = 0.88 * base_score + 0.12 * _sortable_float(frame_hunt_score if frame_hunt_score is not None else 50.0)
    score += _actionable_spike_adjustment(
        signal_direction=signal_direction,
        volume_spike=volume_spike,
        spike_dir=spike_dir,
    )
    return max(0.0, min(100.0, score))


def _actionable_context_score(
    *,
    adaptive_edge_score: float,
    session_fit_score: float,
    archive_guardrail_penalty: float,
    direction: str,
    market_lead_state: str,
    symbol: str,
    classify_symbol_sector,
    sector_rotation_snapshot,
) -> float:
    direction_key = _signal_tracker_direction_key(direction)
    lead_key = _signal_tracker_direction_key(market_lead_state)
    lead_align = 1.0 if direction_key in {"UPSIDE", "DOWNSIDE"} and direction_key == lead_key else 0.0
    leader_sector = str(getattr(sector_rotation_snapshot, "leader_sector", "") or "").strip().lower()
    sector_tag = str(classify_symbol_sector(str(symbol or "")) or "").strip().lower()
    sector_align = 1.0 if leader_sector and sector_tag and sector_tag == leader_sector else 0.0
    score = (
        0.55 * _sortable_float(adaptive_edge_score)
        + 0.20 * _sortable_float(session_fit_score)
        + 12.0 * lead_align
        + 8.0 * sector_align
        - 0.60 * _sortable_float(archive_guardrail_penalty)
    )
    return max(0.0, min(100.0, score))


def _actionable_tactical_candidate_score(
    *,
    spot_direction: str,
    signal_direction: str,
    ai_direction: str,
    ai_agreement: float,
    frame_hunt_score: float,
    execution_structure_quality: float,
    execution_trend_quality: float,
    execution_location_quality: float,
    rr_ratio: float | None,
    adx_val: float,
) -> float:
    spot_key = _signal_tracker_direction_key(spot_direction)
    signal_key = _signal_tracker_direction_key(signal_direction)
    ai_key = _signal_tracker_direction_key(ai_direction)
    if signal_key not in {"UPSIDE", "DOWNSIDE"}:
        return 0.0
    if spot_key != "NEUTRAL":
        return 0.0
    if ai_key not in {"NEUTRAL", signal_key} and float(ai_agreement or 0.0) >= 0.67:
        return 0.0
    score = (
        0.34 * _sortable_float(frame_hunt_score)
        + 0.22 * _sortable_float(execution_structure_quality)
        + 0.18 * _sortable_float(execution_trend_quality)
        + 0.16 * _sortable_float(execution_location_quality)
        + 0.06 * _actionable_rr_score(rr_ratio)
        + 0.04 * _actionable_adx_score(adx_val)
    )
    if ai_key == signal_key:
        score += 6.0 * max(0.0, min(1.0, float(ai_agreement or 0.0)))
    return max(0.0, min(100.0, score))


def _actionable_direction_include(
    *,
    direction_filter: str,
    scan_mode: str,
    spot_direction: str,
    signal_direction: str,
    tactical_candidate_score: float,
    emerging_direction: str = "",
    frame_hunt_score: float = 0.0,
    radar_source_score: float = 0.0,
) -> bool:
    spot_key = _signal_tracker_direction_key(spot_direction)
    if (
        str(direction_filter or "").strip() == "Both"
        or (direction_filter == "Upside" and spot_key == "UPSIDE")
        or (direction_filter == "Downside" and spot_key == "DOWNSIDE")
    ):
        return True
    normalized_mode = _normalize_scan_mode(scan_mode)
    if normalized_mode not in {SCAN_MODE_ACTIONABLE, SCAN_MODE_EMERGING}:
        return False
    signal_key = _signal_tracker_direction_key(signal_direction)
    emerging_key = _signal_tracker_direction_key(emerging_direction)
    if direction_filter == "Upside" and signal_key != "UPSIDE":
        if normalized_mode != SCAN_MODE_EMERGING or emerging_key != "UPSIDE":
            return False
    if direction_filter == "Downside" and signal_key != "DOWNSIDE":
        if normalized_mode != SCAN_MODE_EMERGING or emerging_key != "DOWNSIDE":
            return False
    if normalized_mode == SCAN_MODE_EMERGING:
        if direction_filter == "Upside" and emerging_key == "UPSIDE":
            return True
        if direction_filter == "Downside" and emerging_key == "DOWNSIDE":
            return True
        if _sortable_float(radar_source_score) >= 0.64 and signal_key in {"UPSIDE", "DOWNSIDE"}:
            if direction_filter == "Both":
                return True
            if direction_filter == "Upside" and signal_key == "UPSIDE":
                return True
            if direction_filter == "Downside" and signal_key == "DOWNSIDE":
                return True
        return _sortable_float(tactical_candidate_score) >= 64.0 or _sortable_float(frame_hunt_score) >= 68.0
    return _sortable_float(tactical_candidate_score) >= 72.0


def _actionable_universe_ordered_symbols(
    usdt_symbols: list[str],
    market_rows: list[dict],
    *,
    timeframe: str,
    direction_filter: str,
    classify_symbol_sector=None,
) -> list[str]:
    if not usdt_symbols or not market_rows:
        return list(usdt_symbols)
    ranked_rows = _dedupe_market_rows(market_rows)
    if not ranked_rows:
        return list(usdt_symbols)
    volume_weight, move_weight, mcap_weight, _min_move, _sweet_move, _stretch_move = _actionable_universe_profile(timeframe)
    sector_weight = _actionable_sector_bias_weight(timeframe)
    sector_scores = _actionable_sector_bias_scores(
        ranked_rows,
        direction_filter=direction_filter,
        classify_symbol_sector=classify_symbol_sector,
    )
    total = max(len(ranked_rows) - 1, 1)
    score_map: dict[str, float] = {}
    for idx, row in enumerate(ranked_rows):
        base = canonical_base_symbol((row or {}).get("symbol") or "")
        if not base:
            continue
        volume_rank_score = max(0.0, 1.0 - (idx / total))
        move_score = _actionable_universe_movement_score(
            _market_row_pct_change(row),
            timeframe=timeframe,
            direction_filter=direction_filter,
        )
        mcap_score = _actionable_universe_market_cap_score((row or {}).get("market_cap"))
        sector = str(classify_symbol_sector(base) or "").strip() if callable(classify_symbol_sector) else ""
        sector_score = _sortable_float(sector_scores.get(sector, 0.5 if sector_scores else 0.0))
        score_map[base] = (
            volume_weight * volume_rank_score
            + move_weight * move_score
            + mcap_weight * mcap_score
            + sector_weight * sector_score
        )

    indexed_pairs = list(enumerate(usdt_symbols))
    ordered_pairs = sorted(
        indexed_pairs,
        key=lambda item: (
            -score_map.get(_canonical_pair_base(item[1]), -1.0),
            item[0],
        ),
    )
    return [pair for _idx, pair in ordered_pairs]


def _emerging_universe_ordered_symbols(
    usdt_symbols: list[str],
    market_rows: list[dict],
    *,
    timeframe: str,
    direction_filter: str,
    classify_symbol_sector=None,
) -> list[str]:
    if not usdt_symbols or not market_rows:
        return list(usdt_symbols)
    ranked_rows = _dedupe_market_rows(market_rows)
    if not ranked_rows:
        return list(usdt_symbols)
    volume_weight, move_weight, mcap_weight, _min_move, _sweet_move, _stretch_move = _emerging_universe_profile(timeframe)
    sector_weight = _emerging_sector_bias_weight(timeframe)
    sector_scores = _actionable_sector_bias_scores(
        ranked_rows,
        direction_filter=direction_filter,
        classify_symbol_sector=classify_symbol_sector,
    )
    score_map: dict[str, float] = {}

    def _emerging_universe_volume_score(row: dict) -> float:
        quote_volume = max(
            _sortable_float((row or {}).get("_quote_volume_24h")),
            _sortable_float((row or {}).get("total_volume")),
            _sortable_float((row or {}).get("_volume_24h")),
        )
        if quote_volume > 0:
            return max(0.0, min(1.0, (math.log10(max(quote_volume, 1.0)) - 5.2) / 3.1))
        radar_hint = _sortable_float((row or {}).get("_radar_source_score", 0.0))
        if radar_hint > 0:
            return max(0.16, min(0.62, 0.18 + 0.44 * radar_hint))
        return 0.12

    for row in ranked_rows:
        base = canonical_base_symbol((row or {}).get("symbol") or "")
        if not base:
            continue
        volume_rank_score = _emerging_universe_volume_score(row)
        move_score = _emerging_universe_movement_score(
            _market_row_pct_change(row),
            timeframe=timeframe,
            direction_filter=direction_filter,
        )
        mcap_score = _emerging_universe_market_cap_score((row or {}).get("market_cap"))
        sector = str(classify_symbol_sector(base) or "").strip() if callable(classify_symbol_sector) else ""
        sector_score = _sortable_float(sector_scores.get(sector, 0.5 if sector_scores else 0.0))
        radar_source_score = _sortable_float((row or {}).get("_radar_source_score", 0.0))
        freshness_score = _sortable_float((row or {}).get("_radar_freshness_score", 0.0))
        memory_score = _sortable_float((row or {}).get("_radar_memory_score", 0.0))
        archive_edge_score = _sortable_float((row or {}).get("_radar_archive_edge_score", 0.0))
        score_map[base] = (
            volume_weight * volume_rank_score
            + move_weight * move_score
            + mcap_weight * mcap_score
            + sector_weight * sector_score
            + 0.18 * radar_source_score
            + 0.20 * freshness_score
            + 0.28 * memory_score
            + 0.18 * archive_edge_score
        )

    indexed_pairs = list(enumerate(usdt_symbols))
    ordered_pairs = sorted(
        indexed_pairs,
        key=lambda item: (
            -score_map.get(_canonical_pair_base(item[1]), -1.0),
            item[0],
        ),
    )
    return [pair for _idx, pair in ordered_pairs]


def _candidate_scan_symbols(
    *,
    usdt_symbols: list[str],
    market_rows: list[dict],
    exclude_stables: bool,
    custom_bases_applied: list[str],
    timeframe: str = "1h",
    direction_filter: str = "Both",
    scan_mode: str = SCAN_MODE_BROAD,
    classify_symbol_sector=None,
) -> list[str]:
    if custom_bases_applied:
        candidates = [f"{b}/USDT" for b in custom_bases_applied]
    else:
        candidates = _filter_scan_symbols(usdt_symbols, market_rows)
        normalized_mode = _normalize_scan_mode(scan_mode)
        if normalized_mode == SCAN_MODE_ACTIONABLE:
            candidates = _actionable_universe_ordered_symbols(
                candidates,
                market_rows,
                timeframe=timeframe,
                direction_filter=direction_filter,
                classify_symbol_sector=classify_symbol_sector,
            )
        elif normalized_mode == SCAN_MODE_EMERGING:
            candidates = _emerging_universe_ordered_symbols(
                candidates,
                market_rows,
                timeframe=timeframe,
                direction_filter=direction_filter,
                classify_symbol_sector=classify_symbol_sector,
            )
    if exclude_stables:
        candidates = [
            s
            for s in candidates
            if "/" in s and not is_stable_base_symbol(s.split("/")[0].upper())
        ]
    return candidates


def _scan_candidate_pool_size(
    requested_n: int,
    *,
    custom_mode_active: bool,
    scan_mode: str = SCAN_MODE_BROAD,
    max_pool_n: int = 250,
) -> int:
    requested = max(1, int(requested_n))
    if custom_mode_active:
        return requested
    normalized_mode = _normalize_scan_mode(scan_mode)
    if normalized_mode == SCAN_MODE_ACTIONABLE:
        extra = min(140, max(35, requested * 3))
    elif normalized_mode == SCAN_MODE_EMERGING:
        extra = min(180, max(45, requested * 4))
    else:
        extra = min(25, max(10, requested // 2))
    return min(int(max_pool_n), requested + extra)


def _next_scan_pool_target(
    current_pool_n: int,
    *,
    requested_n: int,
    produced_n: int,
    custom_mode_active: bool,
    used_major_fallback: bool,
    scan_mode: str = SCAN_MODE_BROAD,
    max_pool_n: int = 250,
) -> int:
    if custom_mode_active or used_major_fallback:
        return int(current_pool_n)
    if int(produced_n) >= int(requested_n):
        return int(current_pool_n)
    if int(current_pool_n) >= int(max_pool_n):
        return int(current_pool_n)
    shortfall = max(1, int(requested_n) - int(produced_n))
    next_pool_n = max(
        int(current_pool_n * 1.5),
        int(current_pool_n)
        + max(
            shortfall,
            55 if _normalize_scan_mode(scan_mode) == SCAN_MODE_EMERGING else 40 if _normalize_scan_mode(scan_mode) == SCAN_MODE_ACTIONABLE else 25,
        ),
    )
    return min(int(max_pool_n), int(next_pool_n))


def _initial_scan_batch_size(
    requested_n: int,
    *,
    scan_pool_n: int,
    custom_mode_active: bool,
    scan_mode: str = SCAN_MODE_BROAD,
    max_initial_n: int = 80,
) -> int:
    requested = max(1, int(requested_n))
    normalized_mode = _normalize_scan_mode(scan_mode)
    if custom_mode_active or normalized_mode not in {SCAN_MODE_ACTIONABLE, SCAN_MODE_EMERGING}:
        return requested
    if normalized_mode == SCAN_MODE_EMERGING:
        return min(int(scan_pool_n), max(int(max_initial_n), 96), max(requested + 18, requested * 3))
    return min(int(scan_pool_n), int(max_initial_n), max(requested + 10, requested * 2))


def _initial_scan_symbols(
    *,
    candidate_pool: list[str],
    market_rows: list[dict] | None = None,
    requested_n: int,
    scan_pool_n: int,
    custom_mode_active: bool,
    scan_mode: str = SCAN_MODE_BROAD,
    timeframe: str = "1h",
) -> list[str]:
    initial_batch_n = _initial_scan_batch_size(
        requested_n,
        scan_pool_n=scan_pool_n,
        custom_mode_active=custom_mode_active,
        scan_mode=scan_mode,
    )
    normalized_mode = _normalize_scan_mode(scan_mode)
    if custom_mode_active or normalized_mode not in {SCAN_MODE_ACTIONABLE, SCAN_MODE_EMERGING}:
        return list(candidate_pool[:initial_batch_n])
    if initial_batch_n >= len(candidate_pool):
        return list(candidate_pool[:initial_batch_n])

    if normalized_mode == SCAN_MODE_EMERGING:
        exploration_ratio = _emerging_exploration_ratio(timeframe)
    else:
        exploration_ratio = _actionable_exploration_ratio(timeframe)
    exploration_n = max(2, int(round(initial_batch_n * exploration_ratio)))
    exploration_n = min(exploration_n, max(0, initial_batch_n - int(requested_n)))
    primary_n = max(int(requested_n), int(initial_batch_n) - int(exploration_n))

    primary_slice = list(candidate_pool[:primary_n])
    remainder = list(candidate_pool[primary_n:])
    if not remainder or exploration_n <= 0:
        return list(candidate_pool[:initial_batch_n])

    protected_slice: list[str] = []
    if normalized_mode == SCAN_MODE_EMERGING and market_rows:
        radar_map: dict[str, float] = {}
        for row in list(market_rows or []):
            if not isinstance(row, dict):
                continue
            base = canonical_base_symbol((row or {}).get("symbol") or "")
            if not base:
                continue
            radar_signal = max(
                _sortable_float((row or {}).get("_radar_source_score", 0.0)),
                _sortable_float((row or {}).get("_radar_freshness_score", 0.0)),
                _sortable_float((row or {}).get("_radar_memory_score", 0.0)),
            )
            if radar_signal <= 0.0:
                continue
            combined_signal = (
                radar_signal
                + 0.20 * _sortable_float((row or {}).get("_radar_freshness_score", 0.0))
                + 0.25 * _sortable_float((row or {}).get("_radar_memory_score", 0.0))
                + 0.18 * max(0.0, _sortable_float((row or {}).get("_radar_archive_edge_score", 0.0)))
            )
            radar_map[base] = max(combined_signal, radar_map.get(base, 0.0))

        if radar_map:
            remainder_order = {symbol: idx for idx, symbol in enumerate(remainder)}
            protected_n = min(
                max(2, int(round(initial_batch_n * 0.12))),
                exploration_n,
            )
            ranked_remainder = sorted(
                remainder,
                key=lambda symbol: (
                    -radar_map.get(_canonical_pair_base(symbol), 0.0),
                    remainder_order.get(symbol, 0),
                ),
            )
            for symbol in ranked_remainder:
                if len(protected_slice) >= protected_n:
                    break
                if radar_map.get(_canonical_pair_base(symbol), 0.0) < 0.58:
                    break
                protected_slice.append(symbol)

    exploratory_slice: list[str] = []
    protected_set = set(protected_slice)
    remaining_remainder = [symbol for symbol in remainder if symbol not in protected_set]
    remaining_exploration_n = max(0, exploration_n - len(protected_slice))
    if remaining_exploration_n >= len(remaining_remainder):
        exploratory_slice = remaining_remainder
    else:
        step = max(1.0, len(remaining_remainder) / float(max(remaining_exploration_n, 1)))
        cursor = step / 2.0
        while len(exploratory_slice) < remaining_exploration_n and remaining_remainder:
            idx = min(len(remaining_remainder) - 1, int(cursor))
            exploratory_slice.append(remaining_remainder[idx])
            cursor += step

    out: list[str] = []
    seen: set[str] = set()
    for symbol in [*primary_slice, *protected_slice, *exploratory_slice, *remaining_remainder]:
        if symbol in seen:
            continue
        seen.add(symbol)
        out.append(symbol)
        if len(out) >= initial_batch_n:
            break
    return out


def _actionable_analysis_batch_size(
    requested_n: int,
    *,
    fetched_n: int,
    scan_mode: str,
) -> int:
    normalized_mode = _normalize_scan_mode(scan_mode)
    if normalized_mode == SCAN_MODE_EMERGING:
        target_n = max(int(requested_n) * 3, int(requested_n) + 24, 40)
        return min(int(fetched_n), min(target_n, 96))
    if normalized_mode != SCAN_MODE_ACTIONABLE:
        return int(fetched_n)
    target_n = max(int(requested_n) * 2, int(requested_n) + 12, 28)
    return min(int(fetched_n), min(target_n, 72))


def _emerging_candidate_score(
    *,
    timeframe: str,
    direction_filter: str,
    spot_direction: str,
    signal_direction: str,
    emerging_direction: str,
    emerging_active: bool,
    frame_hunt_score: float,
    tactical_candidate_score: float,
    execution_structure_quality: float,
    execution_trend_quality: float,
    execution_location_quality: float,
    tech_confidence_score: float,
    ai_confidence_score: float,
    market_cap: object,
    market_pct_change_24h: float = 0.0,
    volume_spike: bool,
    spike_dir: str,
    radar_source_score: float = 0.0,
    radar_freshness_score: float = 0.0,
    radar_memory_score: float = 0.0,
    radar_archive_edge_score: float = 0.0,
) -> float:
    signal_key = _signal_tracker_direction_key(signal_direction)
    if signal_key not in {"UPSIDE", "DOWNSIDE"}:
        return 0.0
    focus_key = str(direction_filter or "").strip().upper()
    emerging_key = _signal_tracker_direction_key(emerging_direction)
    focus_aligned = focus_key == "BOTH" or (
        focus_key == "UPSIDE" and signal_key == "UPSIDE"
    ) or (
        focus_key == "DOWNSIDE" and signal_key == "DOWNSIDE"
    )
    lead_aligned = emerging_active and emerging_key in {"UPSIDE", "DOWNSIDE"} and (
        focus_key == "BOTH"
        or (focus_key == "UPSIDE" and emerging_key == "UPSIDE")
        or (focus_key == "DOWNSIDE" and emerging_key == "DOWNSIDE")
    )
    score = (
        0.38 * _sortable_float(frame_hunt_score)
        + 0.20 * _sortable_float(tactical_candidate_score)
        + 0.10 * _sortable_float(execution_structure_quality)
        + 0.08 * _sortable_float(execution_trend_quality)
        + 0.06 * _sortable_float(execution_location_quality)
        + 0.10 * _sortable_float(tech_confidence_score)
        + 0.04 * _sortable_float(ai_confidence_score)
        + 5.0 * _emerging_universe_market_cap_score(market_cap)
        + 9.0 * _sortable_float(radar_source_score)
        + 11.0 * _sortable_float(radar_freshness_score)
        + 10.0 * _sortable_float(radar_memory_score)
        + 7.0 * _sortable_float(radar_archive_edge_score)
    )
    aligned_move = (
        max(0.0, _sortable_float(market_pct_change_24h))
        if signal_key == "UPSIDE"
        else max(0.0, -_sortable_float(market_pct_change_24h))
    )
    if 0.8 <= aligned_move <= 8.5:
        score += 3.0
    score -= _breakout_late_chase_penalty(
        timeframe=timeframe,
        aligned_move=aligned_move,
        radar_freshness_score=radar_freshness_score,
        radar_memory_score=radar_memory_score,
    )
    if lead_aligned:
        score += 12.0
    elif bool(emerging_active):
        score += 6.0
    if focus_aligned:
        score += 5.0
    if _signal_tracker_direction_key(spot_direction) == "NEUTRAL":
        score += 4.0
    score += 0.15 * _actionable_spike_adjustment(
        signal_direction=signal_direction,
        volume_spike=volume_spike,
        spike_dir=spike_dir,
    )
    return max(0.0, min(100.0, score))


def _next_refill_candidate_batch(
    *,
    candidate_pool: list[str],
    attempted_symbols: set[str],
    requested_n: int,
    produced_n: int,
    custom_mode_active: bool,
    used_major_fallback: bool,
    scan_mode: str = SCAN_MODE_BROAD,
) -> list[str]:
    if custom_mode_active or used_major_fallback:
        return []
    if int(produced_n) >= int(requested_n):
        return []
    remaining = [symbol for symbol in candidate_pool if symbol not in attempted_symbols]
    if not remaining:
        return []
    refill_target = _scan_candidate_pool_size(
        max(1, int(requested_n) - int(produced_n)),
        custom_mode_active=False,
        scan_mode=scan_mode,
    )
    return remaining[:refill_target]
