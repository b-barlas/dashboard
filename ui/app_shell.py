"""Application shell: sidebar, tab routing, and UI context wiring."""

from __future__ import annotations

import time

from ui.ctx import get_ctx, get_ctx_callable
from ui.primitives import render_sidebar_panel, render_sidebar_title, render_status_pill
from ui.tab_registry import TAB_TITLES, build_tab_specs
from core.telemetry import snapshot_summary


def render_app(deps: dict) -> None:
    st = get_ctx(deps, "st", scope="app_shell.deps")

    with st.sidebar:
        render_sidebar_title(st, "Crypto Market Intelligence Hub")
        render_sidebar_panel(
            st,
            title="Core Workflow",
            body_html="1) Market: scan universe<br>2) Position: execute/manage",
            tone="accent",
        )
        t = snapshot_summary(st)
        render_sidebar_panel(
            st,
            title="System Health",
            body_html=(
                f"Events: {t['total_events']} | Errors: {t['error_events']} ({t['error_rate']:.1f}%)<br>"
                f"Cache hit-rate: {t['cache_hit_rate']:.1f}% | HTTP failures: {t['http_failures']}"
            ),
            tone="neutral",
        )

        auto_refresh = st.checkbox("Auto-Refresh", value=False, key="auto_refresh")
        if auto_refresh:
            refresh_interval = st.slider("Refresh Interval (sec)", 30, 300, 60, key="refresh_interval")
            render_status_pill(st, text=f"LIVE — Refreshing every {refresh_interval}s", tone="positive")
            time.sleep(refresh_interval)
            st.rerun()

    tabs = st.tabs(TAB_TITLES)

    tab_specs = build_tab_specs(deps)

    for idx, (renderer, extra_ctx) in enumerate(tab_specs):
        with tabs[idx]:
            render_fn = get_ctx_callable({"renderer": renderer}, "renderer", scope=f"tab:{TAB_TITLES[idx]}")
            render_fn({"st": st, **extra_ctx})
