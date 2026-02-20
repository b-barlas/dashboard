from ui.ctx import get_ctx

import pandas as pd
import plotly.express as px
import requests


def render(ctx: dict) -> None:
    st = get_ctx(ctx, "st")
    ACCENT = get_ctx(ctx, "ACCENT")
    TEXT_MUTED = get_ctx(ctx, "TEXT_MUTED")
    POSITIVE = get_ctx(ctx, "POSITIVE")
    NEGATIVE = get_ctx(ctx, "NEGATIVE")
    WARNING = get_ctx(ctx, "WARNING")
    TEXT_LIGHT = get_ctx(ctx, "TEXT_LIGHT")
    PRIMARY_BG = get_ctx(ctx, "PRIMARY_BG")
    NEON_BLUE = get_ctx(ctx, "NEON_BLUE")
    _tip = get_ctx(ctx, "_tip")
    """Market heatmap with breadth and mover diagnostics."""

    st.markdown(
        f"""
        <style>
        .hm-kpi-grid {{
            display:grid;
            grid-template-columns:repeat(4,minmax(0,1fr));
            gap:10px;
            margin:8px 0 14px 0;
        }}
        .hm-kpi {{
            border:1px solid rgba(0,212,255,0.16);
            border-radius:12px;
            padding:12px 14px;
            background:linear-gradient(140deg, rgba(0,0,0,0.72), rgba(10,18,30,0.88));
        }}
        .hm-kpi-label {{
            color:{TEXT_MUTED};
            font-size:0.70rem;
            text-transform:uppercase;
            letter-spacing:0.8px;
        }}
        .hm-kpi-value {{
            color:{ACCENT};
            font-size:1.2rem;
            font-weight:700;
            margin-top:4px;
        }}
        .hm-badge {{
            display:inline-flex;
            align-items:center;
            gap:6px;
            margin-top:7px;
            padding:2px 9px;
            border-radius:999px;
            font-size:0.72rem;
            font-weight:700;
            border:1px solid rgba(255,255,255,0.18);
            background:rgba(0,0,0,0.28);
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(f"<h2 style='color:{ACCENT};'>Market Heatmap</h2>", unsafe_allow_html=True)
    st.markdown(
        f"<div class='panel-box'>"
        f"<b style='color:{ACCENT}; font-size:1rem;'>What does this tab show?</b>"
        f"<p style='color:{TEXT_MUTED}; font-size:0.9rem; margin-top:6px; line-height:1.6;'>"
        f"Top 100 coins in one map. "
        f"{_tip('Tile Size', 'Bigger tile = larger market cap.')} "
        f"{_tip('Tile Color', 'Green = positive 24h return, red = negative. Intensity shows move size.')} "
        f"Use it to read market breadth and leadership quickly."
        f"</p></div>",
        unsafe_allow_html=True,
    )

    with st.spinner("Loading market data for heatmap..."):
        coins = None
        url = "https://api.coingecko.com/api/v3/coins/markets"
        params = {
            "vs_currency": "usd",
            "order": "market_cap_desc",
            "per_page": 100,
            "page": 1,
            "sparkline": False,
            "price_change_percentage": "24h",
        }
        for _ in range(3):
            try:
                resp = requests.get(url, params=params, timeout=15)
                if resp.status_code == 200:
                    coins = resp.json()
                    break
            except Exception:
                continue

    if not isinstance(coins, list) or not coins:
        st.error("Could not fetch market heatmap data (CoinGecko temporarily unavailable).")
        return

    rows = []
    for c in coins:
        mcap = float(c.get("market_cap") or 0.0)
        pct = c.get("price_change_percentage_24h")
        if pct is None:
            pct = c.get("price_change_percentage_24h_in_currency")
        if pct is None:
            pct = 0.0
        price = float(c.get("current_price") or 0.0)
        symbol = (c.get("symbol") or "").upper()
        name = c.get("name") or ""
        if mcap <= 0 or not symbol:
            continue
        rows.append(
            {
                "Symbol": symbol,
                "Name": name,
                "Market Cap": mcap,
                "Change 24h (%)": float(pct),
                "Price": price,
                "Sector": "Crypto",
            }
        )

    if not rows:
        st.warning("No valid data for heatmap.")
        return

    df_hm = pd.DataFrame(rows)
    adv = int((df_hm["Change 24h (%)"] > 0).sum())
    dec = int((df_hm["Change 24h (%)"] < 0).sum())
    flat = int((df_hm["Change 24h (%)"] == 0).sum())
    avg_chg = float(df_hm["Change 24h (%)"].mean())
    cap_weighted = float((df_hm["Change 24h (%)"] * df_hm["Market Cap"]).sum() / df_hm["Market Cap"].sum())
    breadth = (adv / max(adv + dec, 1)) * 100

    breadth_status = ("Healthy", POSITIVE) if breadth >= 58 else (("Watch", WARNING) if breadth >= 45 else ("Risky", NEGATIVE))
    avg_status = ("Healthy", POSITIVE) if avg_chg > 0 else ("Risky", NEGATIVE)
    cap_status = ("Healthy", POSITIVE) if cap_weighted > 0 else ("Risky", NEGATIVE)
    ad_ratio = adv / max(dec, 1)
    adr_status = ("Healthy", POSITIVE) if ad_ratio >= 1.2 else (("Watch", WARNING) if ad_ratio >= 0.85 else ("Risky", NEGATIVE))

    st.markdown(
        f"<div class='hm-kpi-grid'>"
        f"<div class='hm-kpi'><div class='hm-kpi-label'>Breadth (Advancers)</div><div class='hm-kpi-value'>{breadth:.1f}%</div>"
        f"<span class='hm-badge' style='color:{breadth_status[1]}; border-color:{breadth_status[1]};'><span>&#9679;</span>{breadth_status[0]}</span></div>"
        f"<div class='hm-kpi'><div class='hm-kpi-label'>A/D Ratio</div><div class='hm-kpi-value'>{ad_ratio:.2f}</div>"
        f"<span class='hm-badge' style='color:{adr_status[1]}; border-color:{adr_status[1]};'><span>&#9679;</span>{adr_status[0]}</span></div>"
        f"<div class='hm-kpi'><div class='hm-kpi-label'>Avg 24h Change</div><div class='hm-kpi-value'>{avg_chg:+.2f}%</div>"
        f"<span class='hm-badge' style='color:{avg_status[1]}; border-color:{avg_status[1]};'><span>&#9679;</span>{avg_status[0]}</span></div>"
        f"<div class='hm-kpi'><div class='hm-kpi-label'>Cap-Weighted Change</div><div class='hm-kpi-value'>{cap_weighted:+.2f}%</div>"
        f"<span class='hm-badge' style='color:{cap_status[1]}; border-color:{cap_status[1]};'><span>&#9679;</span>{cap_status[0]}</span></div>"
        f"</div>",
        unsafe_allow_html=True,
    )

    st.markdown(
        f"<details style='margin-bottom:0.7rem;'>"
        f"<summary style='color:{ACCENT}; cursor:pointer;'>How to read quickly (?)</summary>"
        f"<div style='color:{TEXT_MUTED}; font-size:0.85rem; line-height:1.7; margin-top:0.5rem;'>"
        f"<b>1.</b> Start with Breadth and A/D Ratio for market direction quality.<br>"
        f"<b>2.</b> If cap-weighted change conflicts with breadth, large caps and alts are diverging.<br>"
        f"<b>3.</b> Use Top Movers to identify leadership, then validate setup quality in Spot/Position tabs."
        f"</div></details>",
        unsafe_allow_html=True,
    )
    st.markdown(
        f"<div style='color:{TEXT_MUTED}; font-size:0.83rem; margin:2px 0 8px 0; line-height:1.6;'>"
        f"<b style='color:{ACCENT};'>KPI Guide:</b> "
        f"{_tip('Breadth (Advancers)', 'Percentage of coins with positive 24h return. Higher means broader market strength.')} | "
        f"{_tip('A/D Ratio', 'Advancers divided by decliners. >1 means more winners than losers.')} | "
        f"{_tip('Avg 24h Change', 'Simple average of 24h returns across displayed coins.')} | "
        f"{_tip('Cap-Weighted Change', '24h return weighted by market cap. Shows what large caps are doing.')}"
        f"</div>",
        unsafe_allow_html=True,
    )

    # Use a tighter symmetric range so normal daily moves are visually distinguishable.
    max_abs = float(df_hm["Change 24h (%)"].abs().quantile(0.95))
    color_bound = max(3.0, min(15.0, max_abs))
    fig = px.treemap(
        df_hm,
        path=["Sector", "Symbol"],
        values="Market Cap",
        color="Change 24h (%)",
        color_continuous_scale=["#8A001F", "#D4143A", "#253041", "#008A4F", "#00D084"],
        color_continuous_midpoint=0,
        range_color=[-color_bound, color_bound],
        custom_data=["Name", "Price", "Change 24h (%)", "Market Cap"],
    )
    fig.update_traces(
        hovertemplate=(
            "<b>%{customdata[0]} (%{label})</b><br>"
            "Price: $%{customdata[1]:,.4f}<br>"
            "24h Change: %{customdata[2]:+.2f}%<br>"
            "Market Cap: $%{customdata[3]:,.0f}<extra></extra>"
        ),
        textinfo="label+text+percent root",
        texttemplate="<b>%{label}</b><br>%{customdata[2]:+.2f}%",
    )
    fig.update_layout(
        height=650,
        template="plotly_dark",
        margin=dict(l=5, r=5, t=30, b=5),
        paper_bgcolor=PRIMARY_BG,
        font=dict(size=12),
    )
    st.plotly_chart(fig, use_container_width=True)

    st.markdown(f"<div class='god-header'><b style='color:{NEON_BLUE};'>Top Movers (24h)</b></div>", unsafe_allow_html=True)
    df_sorted = df_hm.sort_values("Change 24h (%)", ascending=False).copy()
    top_g = df_sorted.head(10).copy()
    top_l = df_sorted.tail(10).iloc[::-1].copy()
    top_g["Status"] = top_g["Change 24h (%)"].apply(lambda x: "▲ Strong" if x >= 8 else ("■ Moderate" if x >= 3 else "• Mild"))
    top_l["Status"] = top_l["Change 24h (%)"].apply(lambda x: "▼ Heavy" if x <= -8 else ("■ Moderate" if x <= -3 else "• Mild"))
    top_g["Move"] = top_g["Change 24h (%)"]
    top_l["Move"] = top_l["Change 24h (%)"]

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
            top_g[["Symbol", "Move", "Status"]].style.format({"Move": "{:+.2f}%"}).map(_status_style, subset=["Status"]),
            width="stretch",
            hide_index=True,
        )
    with cl:
        st.markdown(f"<b style='color:{NEGATIVE};'>Top Losers</b>", unsafe_allow_html=True)
        st.dataframe(
            top_l[["Symbol", "Move", "Status"]].style.format({"Move": "{:+.2f}%"}).map(_status_style, subset=["Status"]),
            width="stretch",
            hide_index=True,
        )

    st.caption(
        f"Coverage: {len(df_hm)} coins | Advancers: {adv} | Decliners: {dec} | Flat: {flat}"
    )
