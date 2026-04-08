"""Lightweight sector/theme rotation helpers for market scanning."""

from __future__ import annotations

from dataclasses import dataclass


_SECTOR_BY_SYMBOL = {
    "BTC": "Majors",
    "ETH": "Majors",
    "SOL": "Infra / L1",
    "ADA": "Infra / L1",
    "AVAX": "Infra / L1",
    "DOT": "Infra / L1",
    "ATOM": "Infra / L1",
    "NEAR": "Infra / L1",
    "SUI": "Infra / L1",
    "APT": "Infra / L1",
    "SEI": "Infra / L1",
    "TIA": "Infra / L1",
    "TON": "Infra / L1",
    "TRX": "Payments",
    "XRP": "Payments",
    "XLM": "Payments",
    "HBAR": "Payments",
    "LTC": "Payments",
    "BCH": "Payments",
    "BNB": "Exchange",
    "OKB": "Exchange",
    "UNI": "DeFi",
    "AAVE": "DeFi",
    "MKR": "DeFi",
    "CRV": "DeFi",
    "LDO": "DeFi",
    "JUP": "DeFi",
    "PENDLE": "DeFi",
    "SUSHI": "DeFi",
    "LINK": "Oracle / Data",
    "PYTH": "Oracle / Data",
    "DOGE": "Meme",
    "SHIB": "Meme",
    "PEPE": "Meme",
    "WIF": "Meme",
    "BONK": "Meme",
    "FLOKI": "Meme",
    "FET": "AI",
    "RENDER": "AI",
    "RNDR": "AI",
    "TAO": "AI",
    "AGIX": "AI",
    "ONDO": "RWA",
    "ENA": "RWA",
    "ARB": "Layer 2",
    "OP": "Layer 2",
    "STRK": "Layer 2",
    "ZK": "Layer 2",
}


@dataclass(frozen=True)
class SectorRotationSnapshot:
    state: str
    label: str
    note: str
    leader_sector: str
    leader_score: float


def classify_symbol_sector(symbol: object) -> str:
    key = str(symbol or "").strip().upper()
    return _SECTOR_BY_SYMBOL.get(key, "Other")


def _direction_key(value: object) -> str:
    d = str(value or "").strip().upper()
    if d in {"UPSIDE", "LONG", "BUY", "BULLISH"}:
        return "UPSIDE"
    if d in {"DOWNSIDE", "SHORT", "SELL", "BEARISH"}:
        return "DOWNSIDE"
    return "NEUTRAL"


def build_sector_rotation_snapshot(rows: list[dict]) -> SectorRotationSnapshot:
    sector_scores: dict[str, float] = {}
    for row in list(rows or []):
        symbol = str((row or {}).get("Coin") or "").strip().upper()
        if not symbol:
            continue
        sector = classify_symbol_sector(symbol)
        confidence = float(row.get("__confidence_val") or 0.0)
        direction = _direction_key(row.get("Direction"))
        lead_label = str((row or {}).get("__emerging_label") or "").strip()

        weight = 1.0
        if confidence >= 80.0:
            weight += 0.50
        elif confidence >= 65.0:
            weight += 0.25

        score = 0.0
        if lead_label == "Emerging Upside":
            score += 2.0 * weight
        elif lead_label == "Emerging Downside":
            score -= 2.0 * weight
        elif direction == "UPSIDE" and confidence >= 60.0:
            score += 1.0 * weight
        elif direction == "DOWNSIDE" and confidence >= 60.0:
            score -= 1.0 * weight

        if score == 0.0:
            continue
        sector_scores[sector] = float(sector_scores.get(sector, 0.0) + score)

    if not sector_scores:
        return SectorRotationSnapshot(
            state="NONE",
            label="No Clear Sector Lead",
            note="No sector is clustering enough upside or downside leadership yet.",
            leader_sector="None",
            leader_score=0.0,
        )

    positive_scores = [(sector, score) for sector, score in sector_scores.items() if score > 0.0]
    negative_scores = [(sector, score) for sector, score in sector_scores.items() if score < 0.0]

    if positive_scores:
        top_up_sector, top_up_score = max(positive_scores, key=lambda kv: kv[1])
    else:
        top_up_sector, top_up_score = "None", 0.0

    if negative_scores:
        top_down_sector, top_down_score = min(negative_scores, key=lambda kv: kv[1])
    else:
        top_down_sector, top_down_score = "None", 0.0

    if top_up_score >= 2.5 and top_up_score >= abs(top_down_score) + 1.0:
        return SectorRotationSnapshot(
            state="UPSIDE",
            label=f"{top_up_sector} Rotation",
            note=f"{top_up_sector} is clustering the strongest upside leadership right now.",
            leader_sector=top_up_sector,
            leader_score=float(top_up_score),
        )

    if abs(top_down_score) >= 2.5 and abs(top_down_score) >= top_up_score + 1.0:
        return SectorRotationSnapshot(
            state="DOWNSIDE",
            label=f"{top_down_sector} Pressure",
            note=f"{top_down_sector} is clustering the heaviest downside pressure right now.",
            leader_sector=top_down_sector,
            leader_score=float(abs(top_down_score)),
        )

    return SectorRotationSnapshot(
        state="BALANCED",
        label="Mixed Sector Rotation",
        note="Leadership is split across sectors, so no single theme owns the tape yet.",
        leader_sector=top_up_sector if top_up_score >= abs(top_down_score) else top_down_sector,
        leader_score=float(max(top_up_score, abs(top_down_score))),
    )
