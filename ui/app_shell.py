"""Application shell: sidebar, tab routing, and UI context wiring."""

from __future__ import annotations

import time

from ui.ctx import get_ctx, get_ctx_callable
from ui.tab_registry import TAB_TITLES, build_tab_specs


def render_app(deps: dict) -> None:
    st = get_ctx(deps, "st", scope="app_shell.deps")
    ACCENT = get_ctx(deps, "ACCENT", scope="app_shell.deps")
    POSITIVE = get_ctx(deps, "POSITIVE", scope="app_shell.deps")

    with st.sidebar:
        st.markdown(
            f"<div style='text-align:center; margin:8px 0;'>"
            f"<span style='color:{ACCENT}; font-size:1.1rem; font-weight:700;'>"
            f"Crypto Command Center</span></div>",
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
