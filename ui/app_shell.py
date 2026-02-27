"""Application shell: sidebar, tab routing, and UI context wiring."""

from __future__ import annotations

import time

from ui.ctx import get_ctx, get_ctx_callable
from ui.tab_registry import TAB_TITLES, build_tab_specs
from core.telemetry import snapshot_summary


def render_app(deps: dict) -> None:
    st = get_ctx(deps, "st", scope="app_shell.deps")
    ACCENT = get_ctx(deps, "ACCENT", scope="app_shell.deps")
    POSITIVE = get_ctx(deps, "POSITIVE", scope="app_shell.deps")

    with st.sidebar:
        st.markdown(
            f"<div style='text-align:center; margin:8px 0;'>"
            f"<span style='color:{ACCENT}; font-size:1.1rem; font-weight:700;'>"
            f"Crypto Market Intelligence Hub</span></div>",
            unsafe_allow_html=True,
        )
        st.markdown(
            "<div style='font-size:0.78rem; color:#8CA1B6; line-height:1.55; "
            "border:1px solid rgba(0,212,255,0.25); border-radius:8px; padding:8px; margin-bottom:8px;'>"
            "<b style='color:#00D4FF;'>Core Workflow</b><br>"
            "1) Market: scan universe<br>"
            "2) Position: execute/manage"
            "</div>",
            unsafe_allow_html=True,
        )
        t = snapshot_summary(st)
        st.markdown(
            "<div style='font-size:0.76rem; color:#8CA1B6; line-height:1.5; "
            "border:1px solid rgba(255,255,255,0.12); border-radius:8px; padding:8px; margin-bottom:8px;'>"
            "<b style='color:#7FE7FF;'>System Health</b><br>"
            f"Events: {t['total_events']} | Errors: {t['error_events']} ({t['error_rate']:.1f}%)<br>"
            f"Cache hit-rate: {t['cache_hit_rate']:.1f}% | HTTP failures: {t['http_failures']}"
            "</div>",
            unsafe_allow_html=True,
        )

        auto_refresh = st.checkbox("Auto-Refresh", value=False, key="auto_refresh")
        if auto_refresh:
            refresh_interval = st.slider("Refresh Interval (sec)", 30, 300, 60, key="refresh_interval")
            st.markdown(
                f"<div class='pulse' style='text-align:center; color:{POSITIVE}; font-size:0.8rem;'>"
                f"LIVE — Refreshing every {refresh_interval}s</div>",
                unsafe_allow_html=True,
            )
            time.sleep(refresh_interval)
            st.rerun()

    tabs = st.tabs(TAB_TITLES)

    tab_specs = build_tab_specs(deps)

    for idx, (renderer, extra_ctx) in enumerate(tab_specs):
        with tabs[idx]:
            render_fn = get_ctx_callable({"renderer": renderer}, "renderer", scope=f"tab:{TAB_TITLES[idx]}")
            render_fn({"st": st, **extra_ctx})
