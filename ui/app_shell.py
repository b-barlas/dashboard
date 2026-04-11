"""Application shell: sidebar, tab routing, and UI context wiring."""

from __future__ import annotations

from ui.ctx import get_ctx, get_ctx_callable
from ui.primitives import render_sidebar_panel, render_sidebar_title, render_status_pill
from ui.tab_registry import TAB_GROUPS, TAB_TITLES, build_tab_specs
from core.telemetry import snapshot_summary
from core.trading_copy import get_copy_audience, set_copy_audience


def render_app(deps: dict) -> None:
    st = get_ctx(deps, "st", scope="app_shell.deps")
    requested_audience = st.session_state.get("dashboard_copy_audience", get_copy_audience())
    set_copy_audience(requested_audience)

    with st.sidebar:
        render_sidebar_title(st, "Crypto Market Intelligence Hub")
        render_sidebar_panel(
            st,
            title="Core Workflow",
            body_html=(
                "1) Market: scan and prioritize<br>"
                "2) Spot: confirm the setup<br>"
                "3) Position: manage open risk<br>"
                "4) Sessions: time the entry<br>"
                "5) Signal Archive: learn and calibrate"
            ),
            tone="accent",
        )
        render_sidebar_panel(
            st,
            title="Navigation Map",
            body_html="<br>".join(
                f"<b>{group}</b>: {', '.join(titles)}"
                for group, titles in TAB_GROUPS
            ),
            tone="neutral",
        )
        render_sidebar_panel(
            st,
            title="What Is Secondary",
            body_html=(
                "Multi-TF and the research tabs are there to support the core flow. "
                "You should not need them on every pass through the dashboard."
            ),
            tone="neutral",
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

    def _render_tabs_once() -> None:
        tabs = st.tabs(TAB_TITLES)
        tab_specs = build_tab_specs(deps)
        for idx, (renderer, extra_ctx) in enumerate(tab_specs):
            with tabs[idx]:
                render_fn = get_ctx_callable({"renderer": renderer}, "renderer", scope=f"tab:{TAB_TITLES[idx]}")
                render_fn({"st": st, **extra_ctx})

    if auto_refresh:
        @st.fragment(run_every=int(st.session_state.get("refresh_interval", 60)))
        def _render_live_tabs() -> None:
            _render_tabs_once()

        _render_live_tabs()
    else:
        _render_tabs_once()
