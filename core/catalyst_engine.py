"""Event / catalyst helpers for low-noise market risk windows."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import pandas as pd


@dataclass(frozen=True)
class MarketCatalystSnapshot:
    state: str
    label: str
    note: str
    blocking: bool
    caution: bool
    severity: str
    next_event: str
    hours_to_event: float | None
    size_cap_fraction: float
    category: str = ""
    scope: str = "market"
    tag: str = ""
    gate_bias: str = "NONE"
    targeted_only: bool = False
    targeted_size_cap_fraction: float = 1.0


def _normalize_scope(value: object) -> str:
    scope = str(value or "market").strip().lower()
    if scope in {"token", "asset", "symbol", "coin"}:
        return "token"
    if scope in {"sector", "theme"}:
        return "sector"
    return "market"


def _normalize_symbol_tag(value: object) -> str:
    text = str(value or "").strip().upper()
    if "/" in text:
        text = text.split("/", 1)[0].strip()
    return text


def _normalize_sector_tag(value: object) -> str:
    return str(value or "").strip().lower().replace("_", " ").replace("-", " ")


def _event_priority(event: dict[str, object]) -> tuple[int, int, float]:
    severity_rank = {"high": 3, "medium": 2, "low": 1}.get(str(event.get("severity") or "medium"), 2)
    scope_rank = {"market": 3, "sector": 2, "token": 2}.get(str(event.get("scope") or "market"), 3)
    hours = float(event.get("hours_to_event") or 1e9)
    urgency_rank = 3 if hours <= 6.0 else (2 if hours <= 24.0 else (1 if hours <= 48.0 else 0))
    return (severity_rank, scope_rank + urgency_rank, -hours)


def load_manual_catalyst_events(path: str | Path) -> list[dict[str, object]]:
    file_path = Path(path)
    if not file_path.exists():
        return []
    try:
        data = json.loads(file_path.read_text(encoding="utf-8"))
    except Exception:
        return []
    if not isinstance(data, list):
        return []
    rows: list[dict[str, object]] = []
    for item in data:
        if not isinstance(item, dict):
            continue
        rows.append(dict(item))
    return rows


def normalize_catalyst_event(raw: dict[str, object]) -> dict[str, object] | None:
    if not isinstance(raw, dict):
        return None
    title = str(raw.get("title") or raw.get("event") or "").strip()
    when_raw = raw.get("event_time", raw.get("date"))
    if not title or when_raw is None:
        return None
    event_time = pd.to_datetime(when_raw, utc=True, errors="coerce")
    if pd.isna(event_time):
        return None
    severity = str(raw.get("severity") or raw.get("impact") or "medium").strip().lower()
    if severity not in {"high", "medium", "low"}:
        severity = "medium"
    category = str(raw.get("category") or "macro").strip().lower() or "macro"
    scope = _normalize_scope(raw.get("scope") or "market")
    source = str(raw.get("source") or "manual").strip() or "manual"
    tag = str(raw.get("tag") or raw.get("country") or "").strip()
    return {
        "title": title,
        "event_time": event_time,
        "severity": severity,
        "category": category,
        "scope": scope,
        "source": source,
        "tag": tag,
    }


def build_market_catalyst_snapshot(
    events: list[dict[str, object]] | None,
    *,
    now: object | None = None,
) -> MarketCatalystSnapshot:
    ts_now = pd.Timestamp.utcnow() if now is None else pd.to_datetime(now, utc=True, errors="coerce")
    if pd.isna(ts_now):
        ts_now = pd.Timestamp.utcnow()

    normalized: list[dict[str, object]] = []
    for raw in list(events or []):
        event = normalize_catalyst_event(raw)
        if event is None:
            continue
        horizon_hours = float((event["event_time"] - ts_now).total_seconds() / 3600.0)
        if horizon_hours < -2.0 or horizon_hours > 96.0:
            continue
        event["hours_to_event"] = horizon_hours
        normalized.append(event)

    if not normalized:
        return MarketCatalystSnapshot(
            state="CLEAR",
            label="No Near Catalyst",
            note="No high-impact catalyst is close enough to justify changing the playbook.",
            blocking=False,
            caution=False,
            severity="none",
            next_event="",
            hours_to_event=None,
            size_cap_fraction=1.0,
        )

    normalized.sort(key=_event_priority, reverse=True)
    next_event = normalized[0]
    title = str(next_event["title"])
    severity = str(next_event["severity"])
    category = str(next_event.get("category") or "").strip()
    scope = _normalize_scope(next_event.get("scope"))
    hours_to_event = float(next_event["hours_to_event"])
    tag = str(next_event.get("tag") or "").strip()
    prefix = f"{tag} " if tag else ""
    targeted_only = scope in {"sector", "token"}

    def _snapshot(
        *,
        state: str,
        label: str,
        note: str,
        blocking: bool,
        caution: bool,
        size_cap_fraction: float,
        gate_bias: str,
        targeted_size_cap_fraction: float | None = None,
    ) -> MarketCatalystSnapshot:
        return MarketCatalystSnapshot(
            state=state,
            label=label,
            note=note,
            blocking=blocking,
            caution=caution,
            severity=severity,
            next_event=title,
            hours_to_event=hours_to_event,
            size_cap_fraction=float(size_cap_fraction),
            category=category,
            scope=scope,
            tag=tag,
            gate_bias=gate_bias,
            targeted_only=targeted_only,
            targeted_size_cap_fraction=(
                float(targeted_size_cap_fraction)
                if targeted_size_cap_fraction is not None
                else float(size_cap_fraction)
            ),
        )

    if targeted_only and severity == "high" and hours_to_event <= 6.0:
        target_txt = f"{tag} " if tag else ""
        return _snapshot(
            state="TARGETED_BLOCK",
            label="Targeted Catalyst Risk",
            note=(
                f"{target_txt}{title} is very close. Affected names should be treated as stand-aside until the event passes."
            ),
            blocking=False,
            caution=True,
            size_cap_fraction=1.0,
            gate_bias="NONE",
            targeted_size_cap_fraction=0.0,
        )

    if targeted_only and severity == "high" and hours_to_event <= 24.0:
        target_txt = f"{tag} " if tag else ""
        return _snapshot(
            state="TARGETED_CAUTION",
            label="Targeted Catalyst",
            note=(
                f"{target_txt}{title} is within the next day. Keep affected names smaller and demand cleaner confirmation."
            ),
            blocking=False,
            caution=True,
            size_cap_fraction=1.0,
            gate_bias="NONE",
            targeted_size_cap_fraction=0.25,
        )

    if targeted_only and severity == "medium" and hours_to_event <= 12.0:
        target_txt = f"{tag} " if tag else ""
        return _snapshot(
            state="TARGETED_CAUTION",
            label="Targeted Catalyst",
            note=(
                f"{target_txt}{title} is close enough to add localized noise. Treat affected names more selectively."
            ),
            blocking=False,
            caution=True,
            size_cap_fraction=1.0,
            gate_bias="NONE",
            targeted_size_cap_fraction=0.5,
        )

    if severity == "high" and hours_to_event <= 6.0:
        return _snapshot(
            state="BLOCKING",
            label="Catalyst Risk High",
            note=f"{prefix}{title} is too close. Reduce new risk and wait for the event to pass before trusting fresh setups.",
            blocking=True,
            caution=True,
            size_cap_fraction=0.0,
            gate_bias="NO_TRADE",
        )

    if severity == "high" and hours_to_event <= 24.0:
        return _snapshot(
            state="CAUTION",
            label="Catalyst Caution",
            note=f"{prefix}{title} is within the next day. Keep size smaller, stay selective, and favor post-event confirmation over forcing early entries.",
            blocking=False,
            caution=True,
            size_cap_fraction=0.25,
            gate_bias="SELECTIVE_ONLY",
        )

    if severity == "medium" and hours_to_event <= 12.0:
        return _snapshot(
            state="CAUTION",
            label="Catalyst On Radar",
            note=f"{prefix}{title} is close enough to add noise. Favor selective setups, lighter size, and avoid forcing marginal breaks.",
            blocking=False,
            caution=True,
            size_cap_fraction=0.5,
            gate_bias="SELECTIVE_ONLY",
        )

    if severity == "low" and hours_to_event <= 8.0:
        return _snapshot(
            state="RADAR",
            label="Catalyst On Radar",
            note=f"{prefix}{title} is on the radar. It should not control the tape, but avoid getting careless into the print.",
            blocking=False,
            caution=True,
            size_cap_fraction=0.75,
            gate_bias="NONE",
        )

    return _snapshot(
        state="CLEAR",
        label="Catalyst Clear",
        note=f"Next scheduled catalyst is {prefix}{title}, but it is far enough away that normal rules can stay in control.",
        blocking=False,
        caution=False,
        size_cap_fraction=1.0,
        gate_bias="NONE",
    )


def catalyst_event_matches_signal(
    snapshot: MarketCatalystSnapshot | None,
    *,
    symbol: str = "",
    sector_tag: str = "",
) -> bool:
    if snapshot is None:
        return False
    scope = _normalize_scope(getattr(snapshot, "scope", "market"))
    if scope == "market":
        return True
    tag = str(getattr(snapshot, "tag", "") or "").strip()
    if not tag:
        return False
    if scope == "token":
        return _normalize_symbol_tag(symbol) == _normalize_symbol_tag(tag)
    if scope == "sector":
        return _normalize_sector_tag(sector_tag) == _normalize_sector_tag(tag)
    return False


def catalyst_signal_size_cap(
    snapshot: MarketCatalystSnapshot | None,
    *,
    symbol: str = "",
    sector_tag: str = "",
) -> float:
    if snapshot is None:
        return 1.0
    market_cap = float(getattr(snapshot, "size_cap_fraction", 1.0) or 1.0)
    if not bool(getattr(snapshot, "targeted_only", False)):
        return market_cap
    if catalyst_event_matches_signal(snapshot, symbol=symbol, sector_tag=sector_tag):
        return float(getattr(snapshot, "targeted_size_cap_fraction", market_cap) or market_cap)
    return market_cap


def catalyst_signal_note(
    snapshot: MarketCatalystSnapshot | None,
    *,
    symbol: str = "",
    sector_tag: str = "",
) -> str:
    if snapshot is None:
        return ""
    note = str(getattr(snapshot, "note", "") or "").strip()
    if not note:
        return ""
    if not bool(getattr(snapshot, "targeted_only", False)):
        return note
    if catalyst_event_matches_signal(snapshot, symbol=symbol, sector_tag=sector_tag):
        return note
    return ""


def catalyst_window_label(snapshot: MarketCatalystSnapshot | None) -> str:
    if snapshot is None:
        return "Unknown"
    state = str(getattr(snapshot, "state", "") or "").strip().upper()
    severity = str(getattr(snapshot, "severity", "") or "").strip().lower()
    hours = getattr(snapshot, "hours_to_event", None)
    if state == "BLOCKING":
        return "Blocking (<6h)"
    if state == "CAUTION" and severity == "high":
        return "High Impact (6-24h)"
    if state == "CAUTION" and severity == "medium":
        return "Medium Impact (<12h)"
    if state == "RADAR":
        return "Low Impact Radar"
    if state == "TARGETED_BLOCK":
        return "Targeted Block"
    if state == "TARGETED_CAUTION":
        return "Targeted Caution"
    if state == "CLEAR":
        if hours is not None and float(hours) <= 48.0:
            return "Clear (>24h)"
        return "Far / Clear"
    return "Unknown"
