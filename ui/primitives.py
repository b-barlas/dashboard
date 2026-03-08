"""Shared UI primitives for consistent page and sidebar rendering."""

from __future__ import annotations

import html
from collections.abc import Sequence


def _safe(text: object) -> str:
    return html.escape("" if text is None else str(text))


def render_page_header(
    st,
    *,
    title: str,
    intro_html: str | None = None,
    intro_title: str = "What does this tab show?",
    subtitle: str | None = None,
    hero: bool = False,
) -> None:
    title_tag = "h1" if hero else "h2"
    title_class = "app-page-title app-page-title--hero" if hero else "app-page-title"
    parts = ["<div class='app-page-header'>"]
    parts.append(f"<{title_tag} class='{title_class}'>{_safe(title)}</{title_tag}>")
    if subtitle:
        parts.append(f"<div class='app-page-subtitle'>{_safe(subtitle)}</div>")
    if intro_html:
        parts.append(
            "<div class='app-intro-card'>"
            f"<div class='app-intro-title'>{_safe(intro_title)}</div>"
            f"<div class='app-intro-body'>{intro_html}</div>"
            "</div>"
        )
    parts.append("</div>")
    st.markdown("".join(parts), unsafe_allow_html=True)


def render_intro_card(st, *, title: str, body_html: str) -> None:
    st.markdown(
        "<div class='app-intro-card'>"
        f"<div class='app-intro-title'>{_safe(title)}</div>"
        f"<div class='app-intro-body'>{body_html}</div>"
        "</div>",
        unsafe_allow_html=True,
    )


def render_help_details(st, *, summary: str, body_html: str) -> None:
    st.markdown(
        "<details class='app-help-details'>"
        f"<summary>{_safe(summary)}</summary>"
        f"<div class='app-help-body'>{body_html}</div>"
        "</details>",
        unsafe_allow_html=True,
    )


def render_sidebar_title(st, title: str) -> None:
    st.markdown(
        "<div class='app-sidebar-title-wrap'>"
        f"<div class='app-sidebar-title'>{_safe(title)}</div>"
        "</div>",
        unsafe_allow_html=True,
    )


def render_sidebar_panel(st, *, title: str, body_html: str, tone: str = "neutral") -> None:
    tone_key = str(tone or "neutral").strip().lower()
    tone_class = {
        "accent": "app-sidebar-panel--accent",
        "positive": "app-sidebar-panel--positive",
        "warning": "app-sidebar-panel--warning",
        "neutral": "app-sidebar-panel--neutral",
    }.get(tone_key, "app-sidebar-panel--neutral")
    st.markdown(
        f"<div class='app-sidebar-panel {tone_class}'>"
        f"<div class='app-sidebar-panel-title'>{_safe(title)}</div>"
        f"<div class='app-sidebar-panel-body'>{body_html}</div>"
        "</div>",
        unsafe_allow_html=True,
    )


def render_status_pill(st, *, text: str, tone: str = "positive") -> None:
    tone_key = str(tone or "positive").strip().lower()
    tone_class = {
        "positive": "app-status-pill--positive",
        "warning": "app-status-pill--warning",
        "neutral": "app-status-pill--neutral",
    }.get(tone_key, "app-status-pill--positive")
    st.markdown(
        f"<div class='app-status-pill {tone_class} pulse'>{_safe(text)}</div>",
        unsafe_allow_html=True,
    )


def _chip_tone_class(tone: str | None) -> str:
    tone_key = str(tone or "neutral").strip().lower()
    return {
        "accent": "app-chip--accent",
        "positive": "app-chip--positive",
        "warning": "app-chip--warning",
        "negative": "app-chip--negative",
        "neutral": "app-chip--neutral",
    }.get(tone_key, "app-chip--neutral")


def _chip_html(
    *,
    text: str,
    tone: str = "neutral",
    color: str | None = None,
    leading_dot: bool = False,
) -> str:
    custom_style = ""
    if color:
        custom_style = (
            f" style='color:{_safe(color)}; border-color:{_safe(color)};"
            " background:rgba(0,0,0,0.28);'"
        )
    dot = f"<span class='app-chip-dot' style='color:{_safe(color)};'>&#9679;</span>" if leading_dot else ""
    return (
        f"<span class='app-chip {_chip_tone_class(tone)}'{custom_style}>"
        f"{dot}{_safe(text)}</span>"
    )


def render_badge_row(
    st,
    *,
    badges: Sequence[str | dict[str, object]],
) -> None:
    parts = ["<div class='app-badge-row'>"]
    for badge in badges:
        if isinstance(badge, str):
            parts.append(_chip_html(text=badge))
            continue
        parts.append(
            _chip_html(
                text=str(badge.get("text") or ""),
                tone=str(badge.get("tone") or "neutral"),
                color=str(badge.get("color")) if badge.get("color") else None,
                leading_dot=bool(badge.get("leading_dot")),
            )
        )
    parts.append("</div>")
    st.markdown("".join(parts), unsafe_allow_html=True)


def render_kpi_grid(
    st,
    *,
    items: Sequence[dict[str, object]],
    columns: int = 4,
    align: str = "left",
    card_min_height: str | None = None,
    center_last_row: bool = False,
) -> None:
    safe_columns = max(1, int(columns))
    align_key = str(align or "left").strip().lower()
    if align_key not in {"left", "center"}:
        align_key = "left"
    style_vars = [f"--app-kpi-columns:{safe_columns}", f"--app-kpi-align:{align_key}"]
    if card_min_height:
        style_vars.append(f"--app-kpi-card-min-height:{_safe(card_min_height)}")
    grid_class = "app-kpi-grid app-kpi-grid--center-last" if center_last_row else "app-kpi-grid"
    parts = [f"<div class='{grid_class}' style='{' ; '.join(style_vars)};'>"]
    for item in items:
        label = _safe(item.get("label"))
        value = _safe(item.get("value"))
        value_style = ""
        if item.get("value_color"):
            value_style = f" style='color:{_safe(item.get('value_color'))};'"
        label_attrs = ""
        if item.get("label_title"):
            label_attrs = f" title='{_safe(item.get('label_title'))}'"
        parts.append("<div class='app-kpi-card'>")
        parts.append(f"<div class='app-kpi-label'{label_attrs}>{label}</div>")
        parts.append(f"<div class='app-kpi-value'{value_style}>{value}</div>")
        subtext = item.get("subtext")
        if subtext:
            parts.append(f"<div class='app-kpi-sub'>{_safe(subtext)}</div>")
        badge_text = item.get("badge_text")
        if badge_text:
            parts.append(
                _chip_html(
                    text=str(badge_text),
                    tone=str(item.get("badge_tone") or "neutral"),
                    color=str(item.get("badge_color")) if item.get("badge_color") else None,
                    leading_dot=bool(item.get("badge_dot")),
                )
            )
        parts.append("</div>")
    parts.append("</div>")
    st.markdown("".join(parts), unsafe_allow_html=True)


def render_insight_card(
    st,
    *,
    title: str,
    body_html: str,
    badges: Sequence[str | dict[str, object]] | None = None,
    tone: str = "accent",
) -> None:
    tone_key = str(tone or "accent").strip().lower()
    tone_class = {
        "accent": "app-insight-card--accent",
        "positive": "app-insight-card--positive",
        "warning": "app-insight-card--warning",
        "negative": "app-insight-card--negative",
        "neutral": "app-insight-card--neutral",
    }.get(tone_key, "app-insight-card--accent")
    parts = [f"<div class='app-insight-card {tone_class}'>"]
    parts.append(f"<div class='app-insight-title'>{_safe(title)}</div>")
    parts.append(f"<div class='app-insight-body'>{body_html}</div>")
    if badges:
        parts.append("<div class='app-insight-badges'>")
        for badge in badges:
            if isinstance(badge, str):
                parts.append(_chip_html(text=badge))
                continue
            parts.append(
                _chip_html(
                    text=str(badge.get("text") or ""),
                    tone=str(badge.get("tone") or "neutral"),
                    color=str(badge.get("color")) if badge.get("color") else None,
                    leading_dot=bool(badge.get("leading_dot")),
                )
            )
        parts.append("</div>")
    parts.append("</div>")
    st.markdown("".join(parts), unsafe_allow_html=True)
