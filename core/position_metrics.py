from __future__ import annotations


def compute_position_pnl(
    *,
    entry_price: float,
    current_price: float,
    direction: str,
    leverage: float,
    margin_used: float,
    funding_impact_pct: float,
) -> dict:
    if entry_price <= 0:
        return {
            "raw_pct": 0.0,
            "levered_pct": 0.0,
            "notional": 0.0,
            "gross_usd": 0.0,
            "funding_usd": 0.0,
            "net_usd": 0.0,
        }

    pnl_abs = (entry_price - current_price) if direction == "SHORT" else (current_price - entry_price)
    raw_pct = (pnl_abs / entry_price) * 100.0
    levered_pct = raw_pct * float(leverage)
    notional = max(0.0, float(margin_used) * float(leverage))
    gross_usd = notional * (raw_pct / 100.0)
    funding_usd = notional * (float(funding_impact_pct) / 100.0)
    net_usd = gross_usd + funding_usd
    return {
        "raw_pct": raw_pct,
        "levered_pct": levered_pct,
        "notional": notional,
        "gross_usd": gross_usd,
        "funding_usd": funding_usd,
        "net_usd": net_usd,
    }


def estimate_liquidation(
    *,
    entry_price: float,
    current_price: float,
    direction: str,
    leverage: float,
) -> dict:
    if entry_price <= 0 or leverage <= 1:
        return {"liq_price": None, "distance_pct": None}

    eff = 1.0 / float(leverage)
    liq_price = entry_price * (1.0 - eff) if direction == "LONG" else entry_price * (1.0 + eff)
    if current_price > 0:
        distance_pct = abs(current_price - liq_price) / current_price * 100.0
    else:
        distance_pct = None
    return {"liq_price": liq_price, "distance_pct": distance_pct}


def compute_hard_invalidation(
    *,
    direction: str,
    support: float,
    resistance: float,
    atr14: float,
    buffer_mult: float = 0.5,
    current_price: float,
) -> dict:
    inv_buffer = max(0.0, float(buffer_mult) * max(0.0, float(atr14)))
    invalidation = (support - inv_buffer) if direction == "LONG" else (resistance + inv_buffer)
    invalidated = (current_price < invalidation) if direction == "LONG" else (current_price > invalidation)
    return {"level": invalidation, "invalidated": bool(invalidated), "buffer": inv_buffer}


def compute_health_decision(
    *,
    direction: str,
    signal_direction: str,
    confidence: float,
    conviction_label: str,
    liq_distance_pct: float | None,
    invalidated: bool,
    levered_pnl_pct: float,
) -> dict:
    health_score = 100
    notes: list[str] = []

    if direction != signal_direction and signal_direction != "WAIT":
        health_score -= 35
        notes.append("signal conflict")
    elif signal_direction == "WAIT":
        health_score -= 15
        notes.append("no clear technical edge")

    if confidence < 50:
        health_score -= 20
        notes.append("low confidence")
    elif confidence < 60:
        health_score -= 10
        notes.append("medium confidence")

    if conviction_label == "CONFLICT":
        health_score -= 25
        notes.append("AI conflict")
    elif conviction_label == "LOW":
        health_score -= 15
        notes.append("low conviction")
    elif conviction_label == "MEDIUM":
        health_score -= 5

    if liq_distance_pct is not None and liq_distance_pct < 5:
        health_score -= 25
        notes.append("liquidation too close")
    elif liq_distance_pct is not None and liq_distance_pct < 10:
        health_score -= 10
        notes.append("liquidation moderately close")

    if invalidated:
        health_score -= 35
        notes.append("hard invalidation broken")

    if levered_pnl_pct < -10:
        health_score -= 10
        notes.append("deep drawdown")
    elif levered_pnl_pct < 0:
        health_score -= 5

    health_score = max(0, min(100, int(round(health_score))))

    if invalidated or health_score < 40:
        label, action = "EXIT", "Close or hedge now."
    elif health_score < 65:
        label, action = "REDUCE", "De-risk, tighten stop, cut size."
    else:
        label, action = "HOLD", "Position can be held with discipline."

    return {"score": health_score, "label": label, "action": action, "notes": notes}
