"""High-signal market alert helpers for the dashboard."""

from __future__ import annotations

from dataclasses import dataclass

from core.market_decision import normalize_action_class
from core.trading_copy import copy_text, trade_gate_display


@dataclass(frozen=True)
class MarketAlert:
    alert_key: str
    state_signature: str
    severity: str
    tone: str
    title: str
    note: str


def _direction_key(value: object) -> str:
    d = str(value or "").strip().upper()
    if d in {"UPSIDE", "LONG", "BUY", "BULLISH"}:
        return "UPSIDE"
    if d in {"DOWNSIDE", "SHORT", "SELL", "BEARISH"}:
        return "DOWNSIDE"
    return "NEUTRAL"


def _severity_rank(value: str) -> int:
    key = str(value or "").strip().upper()
    if key == "HIGH":
        return 3
    if key == "MEDIUM":
        return 2
    return 1


def _alert_priority(alert_key: str) -> int:
    key = str(alert_key or "").strip().upper()
    order = {
        "CATALYST_BLOCK": 60,
        "TRADE_GATE": 55,
        "MARKET_LEAD": 45,
        "LEARNED_EDGE": 42,
        "ACTIONABLE_CLUSTER": 40,
        "ARCHIVE_GUARDRAIL": 38,
        "EXECUTION_STANCE": 37,
        "PLAYBOOK_WINDOW": 34,
        "SECTOR_ROTATION": 30,
        "SESSION_FIT": 25,
        "FLOW_PROXY": 20,
        "CATALYST_CAUTION": 15,
    }
    return order.get(key, 0)


def _top_setup_cluster_alert(rows: list[dict], market_lead_snapshot, market_trade_gate_snapshot) -> MarketAlert | None:
    lead_direction = _direction_key(getattr(market_lead_snapshot, "state", ""))
    if lead_direction not in {"UPSIDE", "DOWNSIDE"}:
        return None
    gate_key = str(getattr(market_trade_gate_snapshot, "gate_key", "") or "").strip().upper()
    if gate_key not in {"TRADEABLE", "SELECTIVE_ONLY", "DEFENSIVE_ONLY"}:
        return None

    candidates: list[tuple[float, str, str]] = []
    for row in list(rows or []):
        symbol = str((row or {}).get("Coin") or "").strip().upper()
        if not symbol:
            continue
        action_raw = str((row or {}).get("__action_raw", (row or {}).get("Setup Confirm", "")) or "")
        action_class = normalize_action_class(action_raw)
        if action_class == "SKIP":
            continue
        direction = _direction_key((row or {}).get("Direction"))
        if direction != lead_direction:
            continue
        confidence = float((row or {}).get("__confidence_val") or 0.0)
        rr_ratio = float((row or {}).get("__rr_val") or 0.0)
        lead_active = _direction_key((row or {}).get("__emerging_direction")) in {"UPSIDE", "DOWNSIDE"}

        score = confidence
        if action_class == "ENTER_TREND_AI":
            score += 24.0
        elif action_class in {"ENTER_TREND_LED", "ENTER_AI_LED"}:
            score += 18.0
        elif action_class == "WATCH":
            score += 8.0
        if lead_active:
            score += 10.0
        if rr_ratio >= 2.0:
            score += 8.0
        elif rr_ratio >= 1.5:
            score += 4.0
        candidates.append((score, symbol, action_class))

    if len(candidates) < 2:
        return None

    candidates.sort(key=lambda item: item[0], reverse=True)
    top_symbols = [symbol for _, symbol, _ in candidates[:3]]
    enter_count = sum(1 for _, _, action_class in candidates if action_class != "WATCH")
    severity = "MEDIUM" if enter_count >= 1 else "INFO"
    tone = "positive" if lead_direction == "UPSIDE" else "negative"
    direction_txt = "upside" if lead_direction == "UPSIDE" else "downside"
    title = f"Clean {direction_txt} cluster"
    note = (
        f"The current market is concentrating cleaner {direction_txt} names. "
        f"Best on-screen cluster: {', '.join(top_symbols)}."
    )
    return MarketAlert(
        alert_key="ACTIONABLE_CLUSTER",
        state_signature=f"{lead_direction}|{'-'.join(top_symbols)}|{enter_count}",
        severity=severity,
        tone=tone,
        title=title,
        note=note,
    )


def _learned_edge_alert(rows: list[dict]) -> MarketAlert | None:
    favored: list[tuple[float, str]] = []
    weak: list[tuple[float, str]] = []
    for row in list(rows or []):
        symbol = str((row or {}).get("Coin") or "").strip().upper()
        if not symbol:
            continue
        score = float((row or {}).get("__adaptive_edge_score") or 50.0)
        label = str((row or {}).get("__adaptive_edge_label") or "").strip().upper()
        if label == "HISTORICALLY FAVORED" and score >= 60.0:
            favored.append((score, symbol))
        elif label == "HISTORICALLY WEAK" and score <= 40.0:
            weak.append((score, symbol))

    if len(favored) >= 2:
        favored.sort(key=lambda item: item[0], reverse=True)
        names = [symbol for _, symbol in favored[:3]]
        avg_score = sum(score for score, _ in favored[:3]) / float(min(len(favored), 3))
        return MarketAlert(
            alert_key="LEARNED_EDGE",
            state_signature=f"FAVORED|{'-'.join(names)}|{round(avg_score, 1)}",
            severity="MEDIUM" if avg_score >= 66.0 else "INFO",
            tone="positive",
            title="History favors the current cluster",
            note=(
                f"Live candidates like {', '.join(names)} line up with the strongest resolved history, "
                "including real outcome feedback where available."
            ),
        )

    if len(weak) >= 2:
        weak.sort(key=lambda item: item[0])
        names = [symbol for _, symbol in weak[:3]]
        avg_score = sum(score for score, _ in weak[:3]) / float(min(len(weak), 3))
        return MarketAlert(
            alert_key="LEARNED_EDGE",
            state_signature=f"WEAK|{'-'.join(names)}|{round(avg_score, 1)}",
            severity="INFO",
            tone="warning",
            title="History is weak for the current cluster",
            note=(
                f"Names like {', '.join(names)} are showing up, but their matched history has been soft. "
                "That usually argues for patience or smaller size."
            ),
        )
    return None


def _session_fit_alert(session_fit_snapshot) -> MarketAlert | None:
    label = str(getattr(session_fit_snapshot, "label", "") or "").strip()
    note = str(getattr(session_fit_snapshot, "note", "") or "").strip()
    score = float(getattr(session_fit_snapshot, "score", 0.0) or 0.0)
    if label == "Session Supportive":
        return MarketAlert(
            alert_key="SESSION_FIT",
            state_signature=f"SUPPORTIVE|{round(score, 2)}",
            severity="INFO" if score < 4.0 else "MEDIUM",
            tone="positive",
            title="Current session has been supportive",
            note=note or "The current UTC session has been one of the cleaner windows in recent history.",
        )
    if label == "Session Fragile":
        return MarketAlert(
            alert_key="SESSION_FIT",
            state_signature=f"FRAGILE|{round(score, 2)}",
            severity="INFO",
            tone="warning",
            title="Current session has been fragile",
            note=note or "The current UTC session has been a weaker conversion window lately.",
        )
    return None


def _archive_guardrail_alert(rows: list[dict]) -> MarketAlert | None:
    guarded: list[tuple[float, str]] = []
    cautioned: list[tuple[float, str]] = []
    for row in list(rows or []):
        symbol = str((row or {}).get("Coin") or "").strip().upper()
        if not symbol:
            continue
        label = str((row or {}).get("__archive_guardrail_label") or "").strip().upper()
        penalty = float((row or {}).get("__archive_guardrail_penalty") or 0.0)
        if label == "ARCHIVE GUARDRAIL" and penalty >= 5.0:
            guarded.append((penalty, symbol))
        elif label == "ARCHIVE CAUTION" and penalty >= 3.0:
            cautioned.append((penalty, symbol))

    if len(guarded) >= 2:
        guarded.sort(key=lambda item: item[0], reverse=True)
        names = [symbol for _, symbol in guarded[:3]]
        return MarketAlert(
            alert_key="ARCHIVE_GUARDRAIL",
            state_signature=f"GUARDRAIL|{'-'.join(names)}|{round(sum(score for score, _ in guarded[:3]), 1)}",
            severity="MEDIUM",
            tone="warning",
            title="History cautions are clustering",
            note=(
                f"Names like {', '.join(names)} are lining up with historically weak session, catalyst, "
                "or market-stance windows. Stay smaller and more selective."
            ),
        )

    if len(cautioned) >= 2:
        cautioned.sort(key=lambda item: item[0], reverse=True)
        names = [symbol for _, symbol in cautioned[:3]]
        return MarketAlert(
            alert_key="ARCHIVE_GUARDRAIL",
            state_signature=f"CAUTION|{'-'.join(names)}|{round(sum(score for score, _ in cautioned[:3]), 1)}",
            severity="INFO",
            tone="warning",
            title="History caution is building",
            note=(
                f"Names like {', '.join(names)} are appearing in softer historical windows. "
                "They may still work, but they usually deserve tighter standards."
            ),
        )
    return None


def _execution_stance_alert(rows: list[dict], market_trade_gate_snapshot, market_catalyst_snapshot, session_fit_snapshot) -> MarketAlert | None:
    gate_key = str(getattr(market_trade_gate_snapshot, "gate_key", "") or "").strip().upper()
    if gate_key == "NO_TRADE":
        return None

    catalyst_label = str(getattr(market_catalyst_snapshot, "label", "") or "").strip()
    catalyst_blocking = bool(getattr(market_catalyst_snapshot, "blocking", False))
    session_label = str(getattr(session_fit_snapshot, "label", "") or "").strip()

    supportive: list[tuple[float, str]] = []
    fragile: list[tuple[float, str]] = []
    for row in list(rows or []):
        symbol = str((row or {}).get("Coin") or "").strip().upper()
        if not symbol:
            continue
        adaptive_label = str((row or {}).get("__adaptive_edge_label") or "").strip().upper()
        adaptive_score = float((row or {}).get("__adaptive_edge_score") or 50.0)
        guardrail_label = str((row or {}).get("__archive_guardrail_label") or "").strip().upper()
        guardrail_penalty = float((row or {}).get("__archive_guardrail_penalty") or 0.0)
        risk_fraction = float((row or {}).get("__risk_unit_fraction") or 0.0)

        if (
            adaptive_label == "HISTORICALLY FAVORED"
            and adaptive_score >= 62.0
            and guardrail_penalty < 3.0
            and risk_fraction >= 0.75
        ):
            supportive.append((adaptive_score + (risk_fraction * 10.0), symbol))
        if (
            guardrail_label == "ARCHIVE GUARDRAIL"
            or guardrail_penalty >= 5.0
            or risk_fraction <= 0.45
            or adaptive_label == "HISTORICALLY WEAK"
        ):
            fragile.append((max(guardrail_penalty, 50.0 - adaptive_score, (1.0 - risk_fraction) * 10.0), symbol))

    if (
        gate_key in {"TRADEABLE", "SELECTIVE_ONLY"}
        and not catalyst_blocking
        and catalyst_label == "Catalyst Clear"
        and session_label == "Session Supportive"
        and len(supportive) >= 2
    ):
        supportive.sort(key=lambda item: item[0], reverse=True)
        names = [symbol for _, symbol in supportive[:3]]
        return MarketAlert(
            alert_key="EXECUTION_STANCE",
            state_signature=f"SUPPORTIVE|{gate_key}|{'-'.join(names)}",
            severity="INFO" if gate_key == "SELECTIVE_ONLY" else "MEDIUM",
            tone="positive",
            title="Current market stance is supportive",
            note=(
                f"Names like {', '.join(names)} are lining up with favorable history, cleaner size support, "
                "and a session/catalyst window that has converted well."
            ),
        )

    if len(fragile) >= 2 and (
        gate_key == "DEFENSIVE_ONLY"
        or session_label == "Session Fragile"
        or catalyst_label == "Catalyst Caution"
    ):
        fragile.sort(key=lambda item: item[0], reverse=True)
        names = [symbol for _, symbol in fragile[:3]]
        return MarketAlert(
            alert_key="EXECUTION_STANCE",
            state_signature=f"FRAGILE|{gate_key}|{'-'.join(names)}",
            severity="INFO",
            tone="warning",
            title="Current market stance is fragile",
            note=(
                f"Names like {', '.join(names)} are showing up in a weaker history window. "
                "Even if setup quality looks fine, this usually argues for less aggression."
            ),
        )
    return None


def _playbook_window_alert(rows: list[dict], market_regime_snapshot, market_catalyst_snapshot, session_fit_snapshot) -> MarketAlert | None:
    playbook = str(getattr(market_regime_snapshot, "playbook", "") or "").strip()
    if not playbook or playbook.lower() == "unknown":
        return None
    session_label = str(getattr(session_fit_snapshot, "label", "") or "").strip()
    catalyst_label = str(getattr(market_catalyst_snapshot, "label", "") or "").strip()
    favored: list[str] = []
    fragile: list[str] = []
    for row in list(rows or []):
        symbol = str((row or {}).get("Coin") or "").strip().upper()
        if not symbol:
            continue
        adaptive_label = str((row or {}).get("__adaptive_edge_label") or "").strip().upper()
        adaptive_score = float((row or {}).get("__adaptive_edge_score") or 50.0)
        archive_label = str((row or {}).get("__archive_guardrail_label") or "").strip().upper()
        archive_penalty = float((row or {}).get("__archive_guardrail_penalty") or 0.0)
        if adaptive_label == "HISTORICALLY FAVORED" and adaptive_score >= 62.0:
            favored.append(symbol)
        if archive_label == "ARCHIVE GUARDRAIL" and archive_penalty >= 5.0:
            fragile.append(symbol)

    if session_label == "Session Supportive" and catalyst_label == "Catalyst Clear" and len(favored) >= 2:
        names = favored[:3]
        return MarketAlert(
            alert_key="PLAYBOOK_WINDOW",
            state_signature=f"SUPPORTIVE|{playbook}|{'-'.join(names)}",
            severity="INFO",
            tone="positive",
            title=copy_text("alert.playbook.supportive.title"),
            note=copy_text("alert.playbook.supportive.note", playbook=playbook, names=", ".join(names)),
        )

    if (session_label == "Session Fragile" or catalyst_label == "Catalyst Caution") and len(fragile) >= 2:
        names = fragile[:3]
        return MarketAlert(
            alert_key="PLAYBOOK_WINDOW",
            state_signature=f"FRAGILE|{playbook}|{'-'.join(names)}",
            severity="INFO",
            tone="warning",
            title=copy_text("alert.playbook.fragile.title"),
            note=copy_text("alert.playbook.fragile.note", playbook=playbook, names=", ".join(names)),
        )
    return None


def build_market_alerts(
    *,
    market_lead_snapshot,
    market_regime_snapshot,
    market_trade_gate_snapshot,
    market_catalyst_snapshot,
    market_flow_snapshot,
    sector_rotation_snapshot,
    session_fit_snapshot=None,
    rows: list[dict],
    max_alerts: int = 3,
) -> list[MarketAlert]:
    alerts: list[MarketAlert] = []

    catalyst_state = str(getattr(market_catalyst_snapshot, "state", "") or "").strip().upper()
    catalyst_title = str(getattr(market_catalyst_snapshot, "next_event", "") or "").strip()
    catalyst_note = str(getattr(market_catalyst_snapshot, "note", "") or "").strip()
    targeted_only = bool(getattr(market_catalyst_snapshot, "targeted_only", False))
    catalyst_tag = str(getattr(market_catalyst_snapshot, "tag", "") or "").strip()
    if bool(getattr(market_catalyst_snapshot, "blocking", False)):
        alerts.append(
            MarketAlert(
                alert_key="CATALYST_BLOCK",
                state_signature=f"{catalyst_state}|{catalyst_title}",
                severity="HIGH",
                tone="negative",
                title=copy_text("alert.catalyst.block.title", title=catalyst_title or "the catalyst"),
                note=catalyst_note or copy_text("alert.catalyst.block.note"),
            )
        )
    elif targeted_only and bool(getattr(market_catalyst_snapshot, "caution", False)):
        target_txt = f"{catalyst_tag} " if catalyst_tag else ""
        alerts.append(
            MarketAlert(
                alert_key="CATALYST_CAUTION",
                state_signature=f"{catalyst_state}|{catalyst_title}|{catalyst_tag}",
                severity="INFO",
                tone="warning",
                title=copy_text(
                    "alert.catalyst.targeted.title",
                    target=f"{target_txt}{catalyst_title}".strip(),
                ),
                note=catalyst_note or copy_text("alert.catalyst.targeted.note"),
            )
        )
    elif bool(getattr(market_catalyst_snapshot, "caution", False)):
        alerts.append(
            MarketAlert(
                alert_key="CATALYST_CAUTION",
                state_signature=f"{catalyst_state}|{catalyst_title}",
                severity="MEDIUM",
                tone="warning",
                title=copy_text("alert.catalyst.window.title", title=catalyst_title or "market event"),
                note=catalyst_note or copy_text("alert.catalyst.window.note"),
            )
        )

    gate_key = str(getattr(market_trade_gate_snapshot, "gate_key", "") or "").strip().upper()
    gate_reason = str(getattr(market_trade_gate_snapshot, "reason_code", "") or "").strip().upper()
    gate_label = str(getattr(market_trade_gate_snapshot, "label", "") or "").strip()
    gate_note = str(getattr(market_trade_gate_snapshot, "note", "") or "").strip()
    if bool(getattr(market_trade_gate_snapshot, "no_trade", False)):
        alerts.append(
            MarketAlert(
                alert_key="TRADE_GATE",
                state_signature=f"{gate_key}|{gate_reason}",
                severity="HIGH",
                tone="negative",
                title=f"{gate_label or trade_gate_display('NO_TRADE')} active",
                note=gate_note or copy_text("alert.trade_gate.no_trade.note"),
            )
        )
    elif gate_key == "DEFENSIVE_ONLY":
        alerts.append(
            MarketAlert(
                alert_key="TRADE_GATE",
                state_signature=f"{gate_key}|{gate_reason}",
                severity="MEDIUM",
                tone="warning",
                title=copy_text("alert.trade_gate.defensive.title"),
                note=gate_note or copy_text("alert.trade_gate.defensive.note"),
            )
        )

    lead_state = str(getattr(market_lead_snapshot, "state", "") or "").strip().upper()
    lead_score = float(getattr(market_lead_snapshot, "score", 50.0) or 50.0)
    lead_note = str(getattr(market_lead_snapshot, "note", "") or "").strip()
    if lead_state == "UPSIDE" and lead_score >= 65.0:
        alerts.append(
            MarketAlert(
                alert_key="MARKET_LEAD",
                state_signature=f"{lead_state}|{round(lead_score)}",
                severity="MEDIUM",
                tone="positive",
                title="Upside pressure is building",
                note=lead_note or "Early market internals are leaning higher before full confirmation.",
            )
        )
    elif lead_state == "DOWNSIDE" and lead_score <= 35.0:
        alerts.append(
            MarketAlert(
                alert_key="MARKET_LEAD",
                state_signature=f"{lead_state}|{round(lead_score)}",
                severity="MEDIUM",
                tone="negative",
                title="Downside pressure is building",
                note=lead_note or "Early market internals are leaning lower before full confirmation.",
            )
        )

    flow_state = str(getattr(market_flow_snapshot, "state", "") or "").strip().upper()
    flow_label = str(getattr(market_flow_snapshot, "label", "") or "").strip()
    flow_note = str(getattr(market_flow_snapshot, "note", "") or "").strip()
    flow_symbol = str(getattr(market_flow_snapshot, "leader_symbol", "") or "").strip()
    flow_score = float(getattr(market_flow_snapshot, "score", 0.0) or 0.0)
    if flow_state == "SHORT_CROWDING":
        alerts.append(
            MarketAlert(
                alert_key="FLOW_PROXY",
                state_signature=f"{flow_state}|{flow_symbol}|{round(flow_score, 1)}",
                severity="INFO",
                tone="positive",
                title=f"{flow_label} in {flow_symbol or 'majors'}",
                note=flow_note,
            )
        )
    elif flow_state == "LONG_CROWDING":
        alerts.append(
            MarketAlert(
                alert_key="FLOW_PROXY",
                state_signature=f"{flow_state}|{flow_symbol}|{round(flow_score, 1)}",
                severity="INFO",
                tone="negative",
                title=f"{flow_label} in {flow_symbol or 'majors'}",
                note=flow_note,
            )
        )

    sector_state = str(getattr(sector_rotation_snapshot, "state", "") or "").strip().upper()
    sector_label = str(getattr(sector_rotation_snapshot, "leader_sector", "") or "").strip()
    sector_note = str(getattr(sector_rotation_snapshot, "note", "") or "").strip()
    sector_score = float(getattr(sector_rotation_snapshot, "leader_score", 0.0) or 0.0)
    if sector_state == "UPSIDE" and sector_score >= 3.0:
        alerts.append(
            MarketAlert(
                alert_key="SECTOR_ROTATION",
                state_signature=f"{sector_state}|{sector_label}|{round(sector_score, 1)}",
                severity="INFO",
                tone="positive",
                title=f"{sector_label} is leading",
                note=sector_note or "A single sector is clustering the strongest upside leadership.",
            )
        )
    elif sector_state == "DOWNSIDE" and sector_score >= 3.0:
        alerts.append(
            MarketAlert(
                alert_key="SECTOR_ROTATION",
                state_signature=f"{sector_state}|{sector_label}|{round(sector_score, 1)}",
                severity="INFO",
                tone="negative",
                title=f"{sector_label} is under pressure",
                note=sector_note or "A single sector is clustering the strongest downside pressure.",
            )
        )

    cluster_alert = _top_setup_cluster_alert(rows, market_lead_snapshot, market_trade_gate_snapshot)
    if cluster_alert is not None:
        alerts.append(cluster_alert)
    learned_edge_alert = _learned_edge_alert(rows)
    if learned_edge_alert is not None:
        alerts.append(learned_edge_alert)
    archive_guardrail_alert = _archive_guardrail_alert(rows)
    if archive_guardrail_alert is not None:
        alerts.append(archive_guardrail_alert)
    execution_stance_alert = _execution_stance_alert(
        rows,
        market_trade_gate_snapshot,
        market_catalyst_snapshot,
        session_fit_snapshot,
    )
    if execution_stance_alert is not None:
        alerts.append(execution_stance_alert)
    playbook_window_alert = _playbook_window_alert(
        rows,
        market_regime_snapshot,
        market_catalyst_snapshot,
        session_fit_snapshot,
    )
    if playbook_window_alert is not None:
        alerts.append(playbook_window_alert)
    session_fit_alert = _session_fit_alert(session_fit_snapshot)
    if session_fit_alert is not None:
        alerts.append(session_fit_alert)

    alerts.sort(
        key=lambda item: (
            -_severity_rank(item.severity),
            -_alert_priority(item.alert_key),
            item.title,
        )
    )
    return alerts[: max(0, int(max_alerts))]
