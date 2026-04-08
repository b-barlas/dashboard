"""Context-only flow proxy helpers from public derivatives market data."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class MarketFlowProxySnapshot:
    state: str
    label: str
    note: str
    score: float
    leader_symbol: str
    funding_bps: float
    oi_change_pct: float
    long_short_ratio: float


def build_market_flow_proxy_snapshot(rows: list[dict] | None) -> MarketFlowProxySnapshot:
    normalized: list[dict[str, float | str]] = []
    for row in list(rows or []):
        symbol = str((row or {}).get("symbol") or "").strip().upper()
        if not symbol:
            continue
        try:
            funding_rate = float((row or {}).get("funding_rate"))
            oi_change_pct = float((row or {}).get("oi_change_pct"))
            long_short_ratio = float((row or {}).get("long_short_ratio"))
        except Exception:
            continue
        funding_bps = funding_rate * 10000.0
        score = 0.0
        if funding_bps <= -1.0:
            score += min(22.0, abs(funding_bps) * 6.0)
        elif funding_bps >= 1.0:
            score -= min(22.0, abs(funding_bps) * 6.0)

        if oi_change_pct >= 2.0:
            if long_short_ratio <= 0.90:
                score += 22.0
            elif long_short_ratio >= 1.10:
                score -= 22.0
        elif oi_change_pct <= -2.0:
            score *= 0.5

        if long_short_ratio <= 0.80:
            score += 8.0
        elif long_short_ratio >= 1.25:
            score -= 8.0

        normalized.append(
            {
                "symbol": symbol,
                "funding_bps": funding_bps,
                "oi_change_pct": oi_change_pct,
                "long_short_ratio": long_short_ratio,
                "score": score,
            }
        )

    if not normalized:
        return MarketFlowProxySnapshot(
            state="BALANCED",
            label="Flow Balanced",
            note="Public positioning proxies are not stretched enough to change the playbook.",
            score=0.0,
            leader_symbol="",
            funding_bps=0.0,
            oi_change_pct=0.0,
            long_short_ratio=1.0,
        )

    normalized.sort(key=lambda item: abs(float(item["score"])), reverse=True)
    leader = normalized[0]
    aggregate = float(sum(float(item["score"]) for item in normalized) / len(normalized))
    avg_funding = float(sum(float(item["funding_bps"]) for item in normalized) / len(normalized))
    avg_oi = float(sum(float(item["oi_change_pct"]) for item in normalized) / len(normalized))
    avg_ratio = float(sum(float(item["long_short_ratio"]) for item in normalized) / len(normalized))

    if aggregate >= 18.0:
        return MarketFlowProxySnapshot(
            state="SHORT_CROWDING",
            label="Shorts Crowded",
            note="Public funding, open interest, and account-ratio proxies suggest shorts are leaning too hard. Upside squeezes become more likely if price keeps pressing higher.",
            score=aggregate,
            leader_symbol=str(leader["symbol"]),
            funding_bps=avg_funding,
            oi_change_pct=avg_oi,
            long_short_ratio=avg_ratio,
        )

    if aggregate <= -18.0:
        return MarketFlowProxySnapshot(
            state="LONG_CROWDING",
            label="Longs Crowded",
            note="Public funding, open interest, and account-ratio proxies suggest longs are leaning too hard. Downside flushes become more likely if support breaks.",
            score=aggregate,
            leader_symbol=str(leader["symbol"]),
            funding_bps=avg_funding,
            oi_change_pct=avg_oi,
            long_short_ratio=avg_ratio,
        )

    return MarketFlowProxySnapshot(
        state="BALANCED",
        label="Flow Balanced",
        note="Public positioning proxies are not stretched enough to create a strong contrarian read.",
        score=aggregate,
        leader_symbol=str(leader["symbol"]),
        funding_bps=avg_funding,
        oi_change_pct=avg_oi,
        long_short_ratio=avg_ratio,
    )
