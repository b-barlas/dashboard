from __future__ import annotations

from tabs.backtest_tab import render as render_model_lab
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
            "Research and simulation workspace. These labs help us test policy ideas on historical candles "
            "before we change live scanner behavior. They are not live archive screens."
        ),
    )

    top_cols = st.columns(2)
    with top_cols[0]:
        render_insight_card(
            st,
            title="What Lives Here",
            body_html=(
                "<b>Model Lab</b> diagnoses the raw signal engine on historical candles.<br>"
                "<b>Setup Lab</b> compares historical setup-class behavior.<br>"
                "<b>Scalp Lab</b> tests historical scalp planner and gate behavior."
            ),
            tone="accent",
        )
    with top_cols[1]:
        render_insight_card(
            st,
            title="What This Is Not",
            body_html=(
                "This is not the live tracker archive. "
                "Use <b>Signal Archive</b> for real logged outcomes, journaled execution, and live learning."
            ),
            tone="neutral",
        )

    model_tab, setup_tab, scalp_tab = st.tabs(["Model Lab", "Setup Lab", "Scalp Lab"])
    with model_tab:
        render_model_lab(ctx)
    with setup_tab:
        render_setup_lab(ctx)
    with scalp_tab:
        render_scalp_lab(ctx)
