from __future__ import annotations

from tabs.scalp_backtest_tab import render as render_scalp_lab
from tabs.setup_backtest_tab import render as render_setup_lab
from ui.ctx import get_ctx
from ui.primitives import render_insight_card, render_page_header


def render(ctx: dict) -> None:
    st = get_ctx(ctx, "st")

    render_page_header(
        st,
        title="Labs",
        intro_html=(
            "Focused research workspace for setup and scalp policy. "
            "Setup Lab stays historical; Scalp Lab now pairs live scalp archive truth with closed-candle study."
        ),
    )

    top_cols = st.columns(2)
    with top_cols[0]:
        render_insight_card(
            st,
            title="What Lives Here",
            body_html=(
                "<b>Setup Lab</b> compares historical setup-class behavior.<br>"
                "<b>Scalp Lab</b> combines live scalp archive truth with historical planner study."
            ),
            tone="accent",
        )
    with top_cols[1]:
        render_insight_card(
            st,
            title="What This Is Not",
            body_html=(
                "This is not a replacement for <b>Signal Archive</b>. "
                "Use Signal Archive for the dashboard-wide live archive; use Labs for tighter policy research."
            ),
            tone="neutral",
        )

    lab_view = st.selectbox(
        "Lab View",
        ["Setup Lab", "Scalp Lab"],
        key="labs_active_view",
        help="Only the selected lab is rendered, so the other study does not run in the background.",
    )
    if lab_view == "Setup Lab":
        render_setup_lab(ctx)
    else:
        render_scalp_lab(ctx)
