"""Shared UI render helpers for setup snapshots and indicator groups."""

from __future__ import annotations

import html


def clean_indicator_text(value: str) -> str:
    text = str(value or "").strip()
    for token in ["🟢", "🔴", "🟡", "⚪", "🔥", "▲▲", "▲", "▼", "→", "–"]:
        text = text.replace(token, "")
    return " ".join(text.split()).strip()


def indicator_color(value: str, *, positive: str, negative: str, warning: str) -> str:
    upper = str(value or "").upper()
    if "VERY STRONG" in upper or "EXTREME" in upper:
        return positive
    if "STRONG" in upper and "NOT" not in upper:
        return positive
    if "WEAK" in upper:
        return negative
    if "STARTING" in upper:
        return warning
    if any(key in upper for key in ["BULLISH", "ABOVE", "OVERSOLD", "LOW", "NEAR BOTTOM", "UP SPIKE"]):
        return positive
    if any(key in upper for key in ["BEARISH", "BELOW", "OVERBOUGHT", "HIGH", "NEAR TOP", "DOWN SPIKE"]):
        return negative
    return warning


def build_setup_snapshot_html(
    *,
    title: str,
    items: list[dict[str, str]],
    text_muted: str,
) -> str:
    item_html = []
    for item in items:
        tooltip = html.escape(str(item.get("title") or ""), quote=True)
        label = html.escape(str(item.get("label") or ""))
        value = str(item.get("value") or "—")
        color = str(item.get("color") or "")
        item_html.append(
            f"<div class='signal-snapshot-item' title='{tooltip}'>"
            f"<div class='signal-snapshot-label'>{label}</div>"
            f"<div class='signal-snapshot-value' style='color:{color};'>{value}</div>"
            f"</div>"
        )

    return (
        f"<style>"
        f".signal-snapshot-title{{"
        f"  color:{text_muted};"
        f"  font-size:0.80rem;"
        f"  text-align:center;"
        f"  text-transform:uppercase;"
        f"  letter-spacing:0.45px;"
        f"  margin:8px 0 6px 0;"
        f"}}"
        f".signal-snapshot-wrap{{"
        f"  background:linear-gradient(140deg, rgba(4, 10, 18, 0.95), rgba(2, 5, 11, 0.95));"
        f"  border:1px solid rgba(0, 212, 255, 0.16);"
        f"  border-radius:12px;"
        f"  padding:10px 12px;"
        f"  margin:0.1rem 0 0.48rem 0;"
        f"}}"
        f".signal-snapshot-grid{{"
        f"  display:flex;"
        f"  flex-wrap:wrap;"
        f"  justify-content:center;"
        f"  gap:0.45rem 0.75rem;"
        f"}}"
        f".signal-snapshot-item{{"
        f"  flex:0 1 150px;"
        f"  min-width:130px;"
        f"  text-align:center;"
        f"  padding:4px 6px;"
        f"}}"
        f".signal-snapshot-label{{"
        f"  color:{text_muted};"
        f"  font-size:0.67rem;"
        f"  text-transform:uppercase;"
        f"  letter-spacing:0.55px;"
        f"}}"
        f".signal-snapshot-value{{"
        f"  font-size:1.03rem;"
        f"  font-weight:700;"
        f"  margin-top:3px;"
        f"}}"
        f"</style>"
        f"<div class='signal-snapshot-title'>{html.escape(title)}</div>"
        f"<div class='signal-snapshot-wrap'><div class='signal-snapshot-grid'>"
        f"{''.join(item_html)}"
        f"</div></div>"
    )


def build_indicator_groups_html(
    *,
    title: str,
    groups: list[tuple[str, list[dict[str, str]]]],
    accent: str,
    text_muted: str,
    positive: str,
    negative: str,
    warning: str,
) -> str:
    group_html: list[str] = []
    for group_title, items in groups:
        cells: list[str] = []
        for item in items:
            raw_value = str(item.get("value") or "")
            cleaned = clean_indicator_text(raw_value)
            if not cleaned:
                continue
            tooltip = html.escape(str(item.get("tooltip") or ""), quote=True)
            name = html.escape(str(item.get("name") or ""))
            color = indicator_color(cleaned, positive=positive, negative=negative, warning=warning)
            cells.append(
                f"<div class='signal-indicator-item'>"
                f"<div class='signal-indicator-name'>{name}</div>"
                f"<div class='signal-indicator-value' style='color:{color};' title='{tooltip}'>{cleaned}</div>"
                f"</div>"
            )
        if not cells:
            continue
        group_html.append(
            f"<div class='signal-indicator-group'>"
            f"<div class='signal-indicator-group-title'>{html.escape(group_title)}</div>"
            f"<div class='signal-indicator-grid'>{''.join(cells)}</div>"
            f"</div>"
        )

    if not group_html:
        return ""

    return (
        f"<style>"
        f".signal-indicator-sep{{"
        f"  margin:8px 0 6px 0;"
        f"  text-align:center;"
        f"  color:{text_muted};"
        f"  font-size:0.80rem;"
        f"  text-transform:uppercase;"
        f"  letter-spacing:0.45px;"
        f"}}"
        f".signal-indicator-wrap{{"
        f"  display:grid;"
        f"  grid-template-columns:repeat(auto-fit,minmax(250px,1fr));"
        f"  gap:0.55rem;"
        f"  margin:0.2rem 0 0.45rem 0;"
        f"}}"
        f".signal-indicator-group{{"
        f"  background:rgba(0,0,0,0.56);"
        f"  border:1px solid rgba(0, 212, 255, 0.14);"
        f"  border-radius:12px;"
        f"  padding:8px 8px 6px 8px;"
        f"}}"
        f".signal-indicator-group-title{{"
        f"  color:{accent};"
        f"  text-align:center;"
        f"  font-size:0.74rem;"
        f"  text-transform:uppercase;"
        f"  letter-spacing:0.55px;"
        f"  margin-bottom:4px;"
        f"}}"
        f".signal-indicator-grid{{"
        f"  display:flex;"
        f"  flex-wrap:wrap;"
        f"  justify-content:center;"
        f"  gap:4px 10px;"
        f"  align-items:center;"
        f"}}"
        f".signal-indicator-item{{"
        f"  text-align:center;"
        f"  padding:4px 2px;"
        f"  flex:0 1 118px;"
        f"}}"
        f".signal-indicator-name{{"
        f"  color:{text_muted};"
        f"  font-size:0.68rem;"
        f"  text-transform:uppercase;"
        f"  letter-spacing:0.5px;"
        f"}}"
        f".signal-indicator-value{{"
        f"  font-size:0.95rem;"
        f"  font-weight:700;"
        f"  margin-top:2px;"
        f"}}"
        f"</style>"
        f"<div class='signal-indicator-sep'>{html.escape(title)}</div>"
        f"<div class='signal-indicator-wrap'>"
        f"{''.join(group_html)}"
        f"</div>"
    )
