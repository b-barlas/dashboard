from __future__ import annotations

import math

import pandas as pd
import plotly.graph_objs as go

from core.portfolio_scenario import (
    MAX_PORTFOLIO_COINS,
    build_portfolio_scenario,
    sanitize_holdings_rows,
)
from ui.ctx import get_ctx
from ui.snapshot_cache import live_or_snapshot


SNAPSHOT_TTL_SEC = 1800
DEFAULT_EDITOR_ROWS = pd.DataFrame(
    {
        "Coin": ["", "", "", ""],
        "Current Value ($)": [math.nan, math.nan, math.nan, math.nan],
    }
)


def _format_money(value: float | None) -> str:
    if value is None or not math.isfinite(float(value)):
        return "—"
    value = float(value)
    if abs(value) >= 1000:
        return f"${value:,.0f}"
    if abs(value) >= 1:
        return f"${value:,.2f}"
    if abs(value) >= 0.1:
        return f"${value:,.4f}"
    return f"${value:,.6f}"


def _format_pct(value: float | None) -> str:
    if value is None or not math.isfinite(float(value)):
        return "—"
    return f"{float(value):+,.2f}%"


def _kpi_card(label: str, value: str, sub: str, accent: str) -> str:
    return (
        "<div class='scenario-kpi'>"
        f"<div class='scenario-kpi-label'>{label}</div>"
        f"<div class='scenario-kpi-value' style='color:{accent};'>{value}</div>"
        f"<div class='scenario-kpi-sub'>{sub}</div>"
        "</div>"
    )


def _display_table(result_df: pd.DataFrame) -> pd.DataFrame:
    if result_df.empty:
        return result_df
    display = result_df.copy()
    display["Current Price ($)"] = display["Current Price ($)"].map(_format_money)
    display["Current Value ($)"] = display["Current Value ($)"].map(_format_money)
    display["Beta vs Anchor"] = display["Beta vs Anchor"].map(lambda v: f"{v:.2f}")
    display["Scenario Return (%)"] = display["Scenario Return (%)"].map(_format_pct)
    display["Projected Price ($)"] = display["Projected Price ($)"].map(_format_money)
    display["Projected Value ($)"] = display["Projected Value ($)"].map(_format_money)
    display["Scenario Range ($)"] = (
        display["Scenario Low ($)"].map(_format_money) + " to " + display["Scenario High ($)"].map(_format_money)
    )
    display["Matched Bars"] = display["Matched Bars"].map(lambda v: f"{int(v)}")
    display["Fit"] = display["Fit"] + " fit"
    return display[
        [
            "Coin",
            "Current Price ($)",
            "Current Value ($)",
            "Link Read",
            "Beta vs Anchor",
            "Fit",
            "Scenario Return (%)",
            "Projected Price ($)",
            "Projected Value ($)",
            "Scenario Range ($)",
            "Matched Bars",
        ]
    ]


def _scenario_chart(result_df: pd.DataFrame, accent: str, positive: str, negative: str) -> go.Figure:
    chart_df = result_df.copy()
    chart_df = chart_df.sort_values("Current Value ($)", ascending=False)
    projected_colors = [positive if val >= 0 else negative for val in chart_df["Scenario Return (%)"]]
    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            name="Current Value",
            x=chart_df["Coin"],
            y=chart_df["Current Value ($)"],
            marker_color="rgba(120, 144, 180, 0.42)",
        )
    )
    fig.add_trace(
        go.Bar(
            name="Projected Value",
            x=chart_df["Coin"],
            y=chart_df["Projected Value ($)"],
            marker_color=projected_colors,
            error_y={
                "type": "data",
                "array": (chart_df["Scenario High ($)"] - chart_df["Projected Price ($)"]) * (chart_df["Current Value ($)"] / chart_df["Current Price ($)"]),
                "arrayminus": (chart_df["Projected Price ($)"] - chart_df["Scenario Low ($)"]) * (chart_df["Current Value ($)"] / chart_df["Current Price ($)"]),
                "thickness": 1.0,
                "width": 2,
                "color": accent,
            },
        )
    )
    fig.update_layout(
        template="plotly_dark",
        barmode="group",
        margin=dict(l=20, r=20, t=40, b=20),
        height=420,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        legend=dict(orientation="h", y=1.08, x=0),
        title="Current vs Projected Basket Value",
        yaxis_title="Value ($)",
    )
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(gridcolor="rgba(148,163,184,0.14)")
    return fig


def render(ctx: dict) -> None:
    st = get_ctx(ctx, "st")
    ACCENT = get_ctx(ctx, "ACCENT")
    TEXT_MUTED = get_ctx(ctx, "TEXT_MUTED")
    POSITIVE = get_ctx(ctx, "POSITIVE")
    NEGATIVE = get_ctx(ctx, "NEGATIVE")
    WARNING = get_ctx(ctx, "WARNING")
    _tip = get_ctx(ctx, "_tip")
    _normalize_coin_input = get_ctx(ctx, "_normalize_coin_input")
    _validate_coin_symbol = get_ctx(ctx, "_validate_coin_symbol")
    fetch_ohlcv = get_ctx(ctx, "fetch_ohlcv")

    st.markdown(
        f"""
        <style>
        .scenario-kpi-grid {{
            display:grid;
            grid-template-columns:repeat(4, minmax(0, 1fr));
            gap:12px;
            margin:10px 0 12px 0;
        }}
        .scenario-kpi {{
            border:1px solid rgba(0,212,255,0.16);
            border-radius:14px;
            padding:14px 16px;
            background:linear-gradient(140deg, rgba(0,0,0,0.74), rgba(9,16,30,0.92));
            min-height:118px;
        }}
        .scenario-kpi-label {{
            color:{TEXT_MUTED};
            font-size:0.72rem;
            letter-spacing:0.8px;
            text-transform:uppercase;
        }}
        .scenario-kpi-value {{
            font-size:1.52rem;
            font-weight:800;
            margin-top:8px;
        }}
        .scenario-kpi-sub {{
            color:{TEXT_MUTED};
            font-size:0.82rem;
            line-height:1.5;
            margin-top:6px;
        }}
        .scenario-insight {{
            border:1px solid rgba(0,212,255,0.18);
            border-left:4px solid {ACCENT};
            border-radius:14px;
            padding:16px 18px;
            background:linear-gradient(135deg, rgba(0,0,0,0.78), rgba(8,18,32,0.94));
            margin:10px 0 16px 0;
        }}
        .scenario-insight-title {{
            color:{ACCENT};
            font-size:1.02rem;
            font-weight:800;
            margin-bottom:6px;
        }}
        .scenario-insight-body {{
            color:{TEXT_MUTED};
            font-size:0.9rem;
            line-height:1.65;
            margin-bottom:10px;
        }}
        .scenario-insight-badges {{
            display:flex;
            flex-wrap:wrap;
            gap:8px;
        }}
        .scenario-insight-badge {{
            border:1px solid rgba(255,255,255,0.14);
            border-radius:999px;
            padding:5px 10px;
            color:{TEXT_MUTED};
            background:rgba(255,255,255,0.03);
            font-size:0.78rem;
        }}
        @media (max-width: 1100px) {{
            .scenario-kpi-grid {{
                grid-template-columns:repeat(2, minmax(0, 1fr));
            }}
        }}
        @media (max-width: 720px) {{
            .scenario-kpi-grid {{
                grid-template-columns:1fr;
            }}
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(f"<h2 style='color:{ACCENT};'>Portfolio Scenario</h2>", unsafe_allow_html=True)
    st.markdown(
        f"<div class='panel-box'>"
        f"<b style='color:{ACCENT}; font-size:1rem;'>What does this tab show?</b>"
        f"<p style='color:{TEXT_MUTED}; font-size:0.9rem; margin-top:6px; line-height:1.65;'>"
        f"Builds an anchor-driven scenario for your basket. You enter the coins you hold and their current dollar values, "
        f"then choose an {_tip('anchor coin', 'The reference coin you want to stress-test, such as BTC or ETH.')} and a target price. "
        f"The model estimates a typical holding horizon from the anchor distance and the anchor's usual bar speed, "
        f"then uses same-timeframe return relationships over that horizon to estimate how each holding may react "
        f"if the anchor reaches that level. This is a horizon-aware linear scenario approximation, not a path-by-path forecast, promise, or prediction."
        f"</p></div>",
        unsafe_allow_html=True,
    )

    with st.form("portfolio_scenario_form"):
        scen_c1, scen_c2, scen_c3, scen_c4 = st.columns([0.9, 1.0, 0.72, 0.6])
        with scen_c1:
            anchor_raw = st.text_input("Anchor Coin", value="BTC")
        with scen_c2:
            anchor_target_price = st.number_input("Anchor Target Price ($)", min_value=0.000001, value=80000.0, step=100.0)
        with scen_c3:
            timeframe = st.selectbox("Timeframe", ["15m", "1h", "4h", "1d"], index=2)
        with scen_c4:
            lookback = st.slider("Lookback Candles", min_value=200, max_value=1500, value=500, step=100)

        st.caption(
            "Enter each holding once. Use tickers like DOGE, TAO, SOL. Current Value is the total USD value of that holding. "
            f"Up to {MAX_PORTFOLIO_COINS} valid holdings are modeled."
        )

        editor_df = st.data_editor(
            DEFAULT_EDITOR_ROWS.copy(),
            key="portfolio_scenario_editor",
            num_rows="dynamic",
            width="stretch",
            column_config={
                "Coin": st.column_config.TextColumn(
                    "Coin (e.g. DOGE, TAO, SOL)",
                    help="Ticker only, for example BTC, ETH, SOL.",
                ),
                "Current Value ($)": st.column_config.NumberColumn(
                    "Current Value ($) · enter total coin value",
                    help="Current total portfolio value for this holding in USD.",
                    min_value=0.0,
                    step=100.0,
                    format="$%.2f",
                ),
            },
            hide_index=True,
        )

        with st.expander("How to read quickly (?)", expanded=False):
            st.markdown(
                """
1. Start with the **Anchor Scenario** card. It tells you how far the anchor must move to hit your target.
2. Check the **Estimated Horizon** badge. That is the bar window used to build the scenario relationship.
3. Read **Projected Basket** and **Projected Delta** together. That is the base-case scenario for the full portfolio.
4. Check **Model Coverage** before trusting the result. Low coverage means several holdings could not be modeled cleanly.
5. Use **Link Read** and **Fit** in the table before trusting any single-coin projection.
6. Treat the **Scenario Range** as uncertainty, not optional noise. Large anchor jumps are still rougher approximations than smaller moves.
                """
            )

        with st.expander("Column Guide (?)", expanded=False):
            st.markdown(
                """
- **Link Read**: how strongly that coin has historically reacted to the anchor.
- **Beta vs Anchor**: rough sensitivity. `1.00` means the coin tended to move close to the anchor's move, `2.00` means roughly double, negative means hedge-like.
- **Fit**: how reliable that historical relationship has been over the estimated scenario horizon.
- **Scenario Return (%)**: modeled move if the anchor reaches target, using the horizon-aware linear relationship on the selected timeframe.
- **Scenario Range ($)**: a low/high band using historical residual noise around that same horizon model. Wider band = weaker certainty.
- **Matched Bars**: how many aligned candles were available to fit the relationship.
                """
            )

        run_scenario = st.form_submit_button("Run Scenario", type="primary", width="stretch")

    holdings, holdings_meta = sanitize_holdings_rows(
        editor_df,
        _normalize_coin_input,
        max_items=MAX_PORTFOLIO_COINS,
        return_meta=True,
    )
    anchor_symbol = _normalize_coin_input(anchor_raw.strip())
    anchor_validation_error = _validate_coin_symbol(anchor_symbol)
    if anchor_validation_error:
        anchor_symbol = None
    current_sig = (
        tuple((item["symbol"], round(float(item["current_value"]), 2)) for item in holdings),
        anchor_symbol,
        round(float(anchor_target_price), 6),
        timeframe,
        int(lookback),
    )

    if run_scenario:
        if not holdings:
            st.warning("Enter at least one portfolio row with Coin and Current Value.")
        elif anchor_symbol is None:
            st.warning(anchor_validation_error or "Anchor coin is not available on the current data stack.")
        else:
            symbols = sorted({item["symbol"] for item in holdings} | {anchor_symbol})
            ohlcv_map: dict[str, pd.DataFrame | None] = {}
            cached_hits: list[str] = []
            missing_symbols: list[str] = []
            for symbol in symbols:
                live_df = fetch_ohlcv(symbol, timeframe, limit=lookback)
                frame, from_cache, ts = live_or_snapshot(
                    st,
                    f"scenario_ohlcv::{symbol}::{timeframe}::{lookback}",
                    live_df,
                    max_age_sec=SNAPSHOT_TTL_SEC,
                    current_sig=(symbol, timeframe, lookback),
                )
                ohlcv_map[symbol] = frame
                if from_cache and ts:
                    cached_hits.append(f"{symbol} ({ts})")
                if frame is None or frame.empty:
                    missing_symbols.append(symbol)

            if anchor_symbol in missing_symbols:
                st.error("Anchor coin data could not be loaded for this scenario.")
            else:
                result = build_portfolio_scenario(holdings, anchor_symbol, float(anchor_target_price), ohlcv_map)
                st.session_state["portfolio_scenario_result"] = {
                    "sig": current_sig,
                    "data": result,
                    "cached_hits": cached_hits,
                    "missing_symbols": missing_symbols,
                    "holdings_meta": holdings_meta,
                }

    stored = st.session_state.get("portfolio_scenario_result")
    if stored and stored.get("sig") == current_sig:
        result = stored["data"]
        cached_hits = stored.get("cached_hits", [])
        missing_symbols = stored.get("missing_symbols", [])
        holdings_meta = stored.get("holdings_meta", holdings_meta)

        if cached_hits:
            st.info("Snapshot fallback used for: " + ", ".join(cached_hits))
        if missing_symbols:
            st.warning("Some symbols could not be modeled and were held flat in basket totals: " + ", ".join(missing_symbols))
        if holdings_meta.get("duplicate_rows", 0):
            dupes = ", ".join(holdings_meta.get("duplicate_symbols", [])) or "duplicate symbols"
            st.info(
                f"Duplicate holding rows were merged by symbol before modeling: {dupes}. "
                "Only the first valid entry for each coin is used."
            )
        if holdings_meta.get("invalid_value_rows", 0):
            st.warning(
                f"Ignored {holdings_meta['invalid_value_rows']} holding row(s) with missing, invalid, or non-positive value."
            )
        if holdings_meta.get("truncated_rows", 0):
            st.warning(
                f"Only the first {MAX_PORTFOLIO_COINS} valid holdings were modeled. "
                f"{holdings_meta['truncated_rows']} extra row(s) were excluded from the basket projection."
            )
        if result.get("capped_coins"):
            st.warning(
                "Large downside scenario exceeded physical limits for: "
                + ", ".join(result["capped_coins"])
                + ". Those projections were capped for consistency."
            )
        if result.get("horizon_capped"):
            reason = result.get("horizon_cap_reason")
            if reason == "stability_cap":
                st.warning(
                    f"Estimated horizon was {result['raw_horizon_bars']} bars, but the model capped it at "
                    f"{result['horizon_bars']} bars for stability. Large target moves are still approximations."
                )
            elif reason == "history_limit":
                st.warning(
                    f"Estimated horizon was {result['raw_horizon_bars']} bars, but available history limited the model to "
                    f"{result['horizon_bars']} bars. Treat long-distance scenarios with extra caution."
                )

        kpi_html = (
            "<div class='scenario-kpi-grid'>"
            + _kpi_card(
                "Anchor Scenario",
                f"{result['anchor_symbol'].split('/')[0]} { _format_pct(result['anchor_move_pct']) }",
                f"Current {_format_money(result['anchor_price'])} -> target {_format_money(result['anchor_target_price'])}",
                ACCENT,
            )
            + _kpi_card(
                "Current Basket",
                _format_money(result["current_total"]),
                "Current portfolio value across entered holdings.",
                ACCENT,
            )
            + _kpi_card(
                "Projected Basket",
                _format_money(result["projected_total"]),
                f"Base-case scenario using {result['horizon_bars']} modeled bars on {timeframe}.",
                POSITIVE if result["projected_delta_pct"] >= 0 else NEGATIVE,
            )
            + _kpi_card(
                "Projected Delta",
                _format_pct(result["projected_delta_pct"]),
                f"Coverage {result['coverage_pct']:.0f}% | weighted fit {result['weighted_r2']:.2f}",
                POSITIVE if result["projected_delta_pct"] >= 0 else NEGATIVE,
            )
            + "</div>"
        )
        st.markdown(kpi_html, unsafe_allow_html=True)

        badges = [
            f"Anchor move {_format_pct(result['anchor_move_pct'])}",
            f"Estimated horizon {result['horizon_bars']} bars",
            (
                f"Raw horizon {result['raw_horizon_bars']} bars"
                + (" (capped)" if result.get("horizon_capped") else "")
            ),
            f"Typical anchor bar move {result['typical_bar_move_pct']:.2f}%",
            f"Modeled coverage {result['coverage_pct']:.0f}%",
            f"Weighted fit {result['weighted_r2']:.2f}",
            f"Avg |beta| {result['weighted_abs_beta']:.2f}",
        ]
        if result["skipped_coins"]:
            badges.append("Held flat: " + ", ".join(result["skipped_coins"]))

        st.markdown(
            "<div class='scenario-insight'>"
            f"<div class='scenario-insight-title'>Basket Insight · {result['verdict']}</div>"
            f"<div class='scenario-insight-body'>{result['verdict_body']}</div>"
            "<div class='scenario-insight-badges'>"
            + "".join(f"<span class='scenario-insight-badge'>{badge}</span>" for badge in badges)
            + "</div></div>",
            unsafe_allow_html=True,
        )

        rows = result["rows"]
        if rows.empty:
            st.warning("No holdings had enough aligned history to produce a scenario.")
        else:
            fig = _scenario_chart(rows, ACCENT, POSITIVE, NEGATIVE)
            st.plotly_chart(fig, width="stretch")
            st.markdown("#### Scenario Table")
            st.dataframe(_display_table(rows), hide_index=True, width="stretch")
    elif stored and stored.get("sig") != current_sig:
        st.info("Inputs changed. Run Scenario again to refresh the basket projection.")
