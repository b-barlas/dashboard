from __future__ import annotations

import pandas as pd
import plotly.express as px
from core.symbols import is_stable_base_symbol
from ui.ctx import get_ctx
from ui.primitives import render_badge_row, render_help_details, render_insight_card, render_kpi_grid, render_page_header


def _safe_float(value, default: float = 0.0) -> float:
    try:
        f = float(value)
        if f != f:  # NaN
            return default
        return f
    except Exception:
        return default


def _is_stablecoin(symbol: str) -> bool:
    return is_stable_base_symbol(symbol)


def _prepare_heatmap_frames(
    rows: list[dict],
    *,
    exclude_stablecoins: bool,
    map_limit: int = 100,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    df_all = pd.DataFrame(rows)
    if df_all.empty:
        return df_all, df_all
    if "Stablecoin" not in df_all.columns:
        df_all["Stablecoin"] = df_all["Symbol"].apply(_is_stablecoin)
    if "TreemapKey" not in df_all.columns:
        df_all["TreemapKey"] = [
            f"legacy:{sym}:{i}" for i, sym in enumerate(df_all["Symbol"].astype(str), start=1)
        ]
    if exclude_stablecoins:
        df_all = df_all[~df_all["Stablecoin"]].copy()
    if df_all.empty:
        return df_all, df_all
    df_all = df_all.sort_values("Market Cap", ascending=False).copy()
    df_map = df_all.head(map_limit).copy()
    return df_all, df_map


def _build_top_movers_tables(df_sample: pd.DataFrame, top_n: int = 10) -> tuple[pd.DataFrame, pd.DataFrame]:
    if df_sample.empty:
        return df_sample.copy(), df_sample.copy()
    df_sorted = df_sample.sort_values("Change 24h (%)", ascending=False).copy()
    top_g = df_sorted.head(top_n).copy()
    top_l = df_sorted.tail(top_n).iloc[::-1].copy()
    top_g["Status"] = top_g["Change 24h (%)"].apply(
        lambda x: "▲ Strong" if x >= 8 else ("■ Moderate" if x >= 3 else "• Mild")
    )
    top_l["Status"] = top_l["Change 24h (%)"].apply(
        lambda x: "▼ Heavy" if x <= -8 else ("■ Moderate" if x <= -3 else "• Mild")
    )
    top_g["Move"] = top_g["Change 24h (%)"]
    top_l["Move"] = top_l["Change 24h (%)"]
    return top_g, top_l


def render(ctx: dict) -> None:
    st = get_ctx(ctx, "st")
    ACCENT = get_ctx(ctx, "ACCENT")
    TEXT_MUTED = get_ctx(ctx, "TEXT_MUTED")
    POSITIVE = get_ctx(ctx, "POSITIVE")
    NEGATIVE = get_ctx(ctx, "NEGATIVE")
    WARNING = get_ctx(ctx, "WARNING")
    PRIMARY_BG = get_ctx(ctx, "PRIMARY_BG")
    NEON_BLUE = get_ctx(ctx, "NEON_BLUE")
    _tip = get_ctx(ctx, "_tip")
    get_heatmap_rows = get_ctx(ctx, "get_heatmap_rows")

    render_page_header(
        st,
        title="Market Heatmap",
        intro_html=(
            "Top market-cap coins in one map, with breadth and movers read from the broader tracked sample. "
            f"{_tip('Tile Size', 'Bigger tile means larger market cap.')} "
            f"{_tip('Tile Color', 'Green = positive 24h return, red = negative. Intensity shows move strength.')} "
            "Use this tab to read breadth and leadership quickly."
        ),
    )
    with st.spinner("Loading market data for heatmap..."):
        rows, source, feed_mode, asof_utc = get_heatmap_rows(limit=180, live_ttl_sec=90)

    if not rows:
        st.error("Could not fetch market heatmap data from live providers.")
        return

    exclude_stablecoins = st.checkbox(
        "Exclude stablecoins",
        value=True,
        key="heatmap_exclude_stablecoins",
    )

    feed_color = POSITIVE if feed_mode == "LIVE" else WARNING
    render_badge_row(
        st,
        badges=[
            {"text": f"{feed_mode} FEED", "color": feed_color},
            {"text": f"SOURCE: {source}"},
            {"text": f"AS OF: {asof_utc or 'N/A'}"},
            {
                "text": (
                    "EXCLUDE STABLECOINS"
                    if exclude_stablecoins
                    else "INCLUDE STABLECOINS"
                )
            },
        ],
    )
    if feed_mode == "CACHED":
        st.warning(
            "Live provider temporarily unavailable. Showing last successful heatmap snapshot."
        )
    elif source == "CoinPaprika":
        st.info("CoinGecko backup active: data is currently sourced from CoinPaprika.")

    df_all, df_map = _prepare_heatmap_frames(
        rows,
        exclude_stablecoins=exclude_stablecoins,
        map_limit=100,
    )
    if df_all.empty:
        st.warning("No valid data for heatmap.")
        return
    if df_map.empty:
        st.warning("No non-stablecoin rows left after filter. Disable stablecoin exclusion to view full map.")
        return

    flat_eps = 0.05
    chg = df_all["Change 24h (%)"]
    adv = int((chg > flat_eps).sum())
    dec = int((chg < -flat_eps).sum())
    flat = int(len(df_all) - adv - dec)
    total = max(len(df_all), 1)
    avg_chg = float(df_all["Change 24h (%)"].mean())
    mcap_sum = float(df_all["Market Cap"].sum())
    cap_weighted = float(
        (df_all["Change 24h (%)"] * df_all["Market Cap"]).sum() / mcap_sum
    ) if mcap_sum > 0 else 0.0
    breadth = (adv / total) * 100.0

    breadth_status = (
        ("Healthy", POSITIVE)
        if breadth >= 58
        else (("Watch", WARNING) if breadth >= 45 else ("Risky", NEGATIVE))
    )
    avg_status = ("Healthy", POSITIVE) if avg_chg > 0 else ("Risky", NEGATIVE)
    cap_status = ("Healthy", POSITIVE) if cap_weighted > 0 else ("Risky", NEGATIVE)
    ad_ratio = adv / max(dec, 1)
    adr_status = (
        ("Healthy", POSITIVE)
        if ad_ratio >= 1.2
        else (("Watch", WARNING) if ad_ratio >= 0.85 else ("Risky", NEGATIVE))
    )
    market_regime = (
        "Risk-On"
        if (breadth >= 55 and cap_weighted > 0)
        else ("Risk-Off" if (breadth < 45 and cap_weighted < 0) else "Selective")
    )
    regime_color = (
        POSITIVE if market_regime == "Risk-On" else (NEGATIVE if market_regime == "Risk-Off" else WARNING)
    )
    divergence = (breadth >= 50 and cap_weighted < 0) or (breadth < 50 and cap_weighted > 0)
    divergence_text = (
        "Large-cap / broad-market divergence detected"
        if divergence
        else "Broad participation and cap-weighted move are aligned"
    )
    divergence_color = WARNING if divergence else POSITIVE

    render_kpi_grid(
        st,
        items=[
            {
                "label": "Breadth (Advancers)",
                "value": f"{breadth:.1f}%",
                "badge_text": breadth_status[0],
                "badge_color": breadth_status[1],
                "badge_dot": True,
            },
            {
                "label": "A/D Ratio",
                "value": f"{ad_ratio:.2f}",
                "badge_text": adr_status[0],
                "badge_color": adr_status[1],
                "badge_dot": True,
            },
            {
                "label": "Avg 24h Change",
                "value": f"{avg_chg:+.2f}%",
                "badge_text": avg_status[0],
                "badge_color": avg_status[1],
                "badge_dot": True,
            },
            {
                "label": "Cap-Weighted Change",
                "value": f"{cap_weighted:+.2f}%",
                "badge_text": cap_status[0],
                "badge_color": cap_status[1],
                "badge_dot": True,
            },
        ],
    )
    render_insight_card(
        st,
        title=f"Market Regime · {market_regime}",
        body_html=(
            "Interpretation: "
            f"<span style='color:{divergence_color}; font-weight:700;'>{divergence_text}</span>"
        ),
        badges=[
            {"text": market_regime, "color": regime_color},
            {"text": divergence_text, "color": divergence_color},
        ],
        tone=(
            "positive"
            if regime_color == POSITIVE
            else ("negative" if regime_color == NEGATIVE else "warning")
        ),
    )

    render_help_details(
        st,
        summary="How to read quickly (?)",
        body_html=(
            "<b>1.</b> Start with Breadth and A/D Ratio for market participation quality.<br>"
            "<b>2.</b> Compare Cap-weighted move vs Breadth; conflict means large caps and broad market diverge.<br>"
            "<b>3.</b> Toggle stablecoin exclusion depending on your read (risk view vs pure crypto beta view).<br>"
            "<b>4.</b> Use Top Movers to find leadership, then confirm execution quality in Market/Spot/Position."
        ),
    )
    st.markdown(
        f"<div style='color:{TEXT_MUTED}; font-size:0.83rem; margin:2px 0 8px 0; line-height:1.6;'>"
        f"<b style='color:{ACCENT};'>KPI Guide:</b> "
        f"{_tip('Breadth (Advancers)', 'Share of displayed coins with move above +0.05%. Higher means broader strength.')} | "
        f"{_tip('A/D Ratio', 'Advancers divided by decliners. Above 1 means winners outnumber losers.')} | "
        f"{_tip('Avg 24h Change', 'Simple average 24h move across displayed coins.')} | "
        f"{_tip('Cap-Weighted Change', '24h move weighted by market cap; emphasizes large-cap behavior.')}"
        f"</div>",
        unsafe_allow_html=True,
    )

    max_abs = float(df_map["Change 24h (%)"].abs().quantile(0.95))
    color_bound = max(3.0, min(15.0, max_abs))
    fig = px.treemap(
        df_map,
        path=["Sector", "TreemapKey"],
        values="Market Cap",
        color="Change 24h (%)",
        color_continuous_scale=["#8A001F", "#D4143A", "#2A3345", "#008A4F", "#00D084"],
        color_continuous_midpoint=0,
        range_color=[-color_bound, color_bound],
        custom_data=["Symbol", "Name", "Price", "Change 24h (%)", "Market Cap", "Provider"],
    )
    fig.update_traces(
        hovertemplate=(
            "<b>%{customdata[1]} (%{customdata[0]})</b><br>"
            "Price: $%{customdata[2]:,.6f}<br>"
            "24h Change: %{customdata[3]:+.2f}%<br>"
            "Market Cap: $%{customdata[4]:,.0f}<br>"
            "Provider: %{customdata[5]}<extra></extra>"
        ),
        textinfo="text+percent root",
        texttemplate="<b>%{customdata[0]}</b><br>%{customdata[3]:+.2f}%",
    )
    fig.update_layout(
        height=650,
        template="plotly_dark",
        margin=dict(l=5, r=5, t=30, b=5),
        paper_bgcolor=PRIMARY_BG,
        font=dict(size=12),
    )
    st.plotly_chart(fig, width="stretch")
    st.caption(
        f"Treemap shows the top {len(df_map)} names by market cap. Breadth and movers use the broader tracked sample of {len(df_all)} coins."
    )

    st.markdown(
        f"<div class='god-header'><b style='color:{NEON_BLUE};'>Tracked-Sample Movers (24h)</b></div>",
        unsafe_allow_html=True,
    )
    top_g, top_l = _build_top_movers_tables(df_all, top_n=10)

    def _status_style(v: str) -> str:
        s = str(v)
        if "▲" in s:
            return f"color:{POSITIVE}; font-weight:700;"
        if "▼" in s:
            return f"color:{NEGATIVE}; font-weight:700;"
        if "Moderate" in s:
            return f"color:{WARNING}; font-weight:700;"
        return f"color:{NEON_BLUE}; font-weight:700;"

    cg, cl = st.columns(2)
    with cg:
        st.markdown(f"<b style='color:{POSITIVE};'>Top Gainers</b>", unsafe_allow_html=True)
        st.dataframe(
            top_g[["Symbol", "Move", "Status"]]
            .style.format({"Move": "{:+.2f}%"})
            .map(_status_style, subset=["Status"]),
            width="stretch",
            hide_index=True,
        )
    with cl:
        st.markdown(f"<b style='color:{NEGATIVE};'>Top Losers</b>", unsafe_allow_html=True)
        st.dataframe(
            top_l[["Symbol", "Move", "Status"]]
            .style.format({"Move": "{:+.2f}%"})
            .map(_status_style, subset=["Status"]),
            width="stretch",
            hide_index=True,
        )

    st.caption(
        f"Tracked sample: {len(df_all)} coins | Treemap: top {len(df_map)} by market cap | Advancers: {adv} | Decliners: {dec} | Flat: {flat} (|move| < {flat_eps:.2f}%)"
    )
