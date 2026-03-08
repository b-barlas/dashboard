from __future__ import annotations

from datetime import datetime, timezone
import time

import pandas as pd
import plotly.express as px
import requests
from ui.ctx import get_ctx
from ui.primitives import render_badge_row, render_help_details, render_insight_card, render_kpi_grid, render_page_header

STABLE_SYMBOLS = {
    "USDT",
    "USDC",
    "DAI",
    "TUSD",
    "FDUSD",
    "PYUSD",
    "USDE",
    "USDD",
    "USDP",
    "GUSD",
    "LUSD",
    "FRAX",
    "EURS",
    "EURC",
    "SUSDE",
}


def _safe_float(value, default: float = 0.0) -> float:
    try:
        f = float(value)
        if f != f:  # NaN
            return default
        return f
    except Exception:
        return default


def _is_stablecoin(symbol: str) -> bool:
    return str(symbol or "").upper().strip() in STABLE_SYMBOLS


def _http_get_json(
    url: str,
    params: dict | None = None,
    timeout: int = 8,
    retries: int = 2,
):
    headers = {
        "Accept": "application/json",
        "User-Agent": "crypto-market-dashboard/1.0",
    }
    for attempt in range(retries):
        try:
            resp = requests.get(url, params=params, timeout=timeout, headers=headers)
            if resp.status_code == 200:
                return resp.json()
            if resp.status_code not in {429, 500, 502, 503, 504}:
                break
        except Exception:
            pass
        if attempt < retries - 1:
            time.sleep(0.35 * (attempt + 1))
    return None


def _fetch_coingecko_heatmap_rows(limit: int = 180) -> list[dict]:
    payload = _http_get_json(
        "https://api.coingecko.com/api/v3/coins/markets",
        params={
            "vs_currency": "usd",
            "order": "market_cap_desc",
            "per_page": min(max(limit, 100), 250),
            "page": 1,
            "sparkline": False,
            "price_change_percentage": "24h",
        },
        timeout=8,
        retries=2,
    )
    if not isinstance(payload, list):
        return []

    rows: list[dict] = []
    for c in payload:
        symbol = str(c.get("symbol") or "").upper().strip()
        if not symbol:
            continue
        mcap = _safe_float(c.get("market_cap"))
        if mcap <= 0:
            continue
        pct = c.get("price_change_percentage_24h")
        if pct is None:
            pct = c.get("price_change_percentage_24h_in_currency")
        cid = str(c.get("id") or "").strip()
        key = f"cg:{cid}" if cid else f"cg:{symbol}:{str(c.get('name') or symbol).strip()}"
        rows.append(
            {
                "Symbol": symbol,
                "Name": str(c.get("name") or symbol),
                "TreemapKey": key,
                "Market Cap": mcap,
                "Change 24h (%)": _safe_float(pct, 0.0),
                "Price": _safe_float(c.get("current_price"), 0.0),
                "Sector": "Crypto",
                "Stablecoin": _is_stablecoin(symbol),
                "Provider": "CoinGecko",
            }
        )

    rows.sort(key=lambda r: r["Market Cap"], reverse=True)
    return rows[:limit]


def _fetch_coinpaprika_heatmap_rows(limit: int = 180) -> list[dict]:
    payload = _http_get_json(
        "https://api.coinpaprika.com/v1/tickers",
        timeout=8,
        retries=2,
    )
    if not isinstance(payload, list):
        return []

    rows: list[dict] = []
    for c in payload:
        symbol = str(c.get("symbol") or "").upper().strip()
        if not symbol:
            continue
        quotes = c.get("quotes") if isinstance(c, dict) else {}
        usd_q = quotes.get("USD") if isinstance(quotes, dict) else {}
        mcap = _safe_float(usd_q.get("market_cap"))
        if mcap <= 0:
            continue
        pid = str(c.get("id") or "").strip()
        key = f"cp:{pid}" if pid else f"cp:{symbol}:{str(c.get('name') or symbol).strip()}"
        rows.append(
            {
                "Symbol": symbol,
                "Name": str(c.get("name") or symbol),
                "TreemapKey": key,
                "Market Cap": mcap,
                "Change 24h (%)": _safe_float(usd_q.get("percent_change_24h"), 0.0),
                "Price": _safe_float(usd_q.get("price"), 0.0),
                "Sector": "Crypto",
                "Stablecoin": _is_stablecoin(symbol),
                "Provider": "CoinPaprika",
            }
        )

    rows.sort(key=lambda r: r["Market Cap"], reverse=True)
    return rows[:limit]


def _load_heatmap_rows(
    st,
    limit: int = 180,
    live_ttl_sec: int = 90,
) -> tuple[list[dict], str, str, str | None]:
    cache_key = "heatmap_last_good_v3"
    live_key = "heatmap_live_cache_v1"
    now_epoch = time.time()
    now_utc = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

    live_cached = st.session_state.get(live_key, {})
    if isinstance(live_cached, dict):
        live_rows = live_cached.get("rows")
        live_at = _safe_float(live_cached.get("fetched_at"), 0.0)
        if isinstance(live_rows, list) and live_rows and (now_epoch - live_at) <= live_ttl_sec:
            return (
                live_rows,
                str(live_cached.get("source") or "Live cache"),
                "LIVE",
                str(live_cached.get("ts") or now_utc),
            )

    rows = _fetch_coingecko_heatmap_rows(limit=limit)
    if rows:
        payload = {"rows": rows, "source": "CoinGecko", "ts": now_utc, "fetched_at": now_epoch}
        st.session_state[live_key] = payload
        st.session_state[cache_key] = payload
        return rows, "CoinGecko", "LIVE", now_utc

    rows = _fetch_coinpaprika_heatmap_rows(limit=limit)
    if rows:
        payload = {"rows": rows, "source": "CoinPaprika", "ts": now_utc, "fetched_at": now_epoch}
        st.session_state[live_key] = payload
        st.session_state[cache_key] = payload
        return rows, "CoinPaprika", "LIVE", now_utc

    cached = st.session_state.get(cache_key, {})
    cached_rows = cached.get("rows") if isinstance(cached, dict) else None
    if isinstance(cached_rows, list) and cached_rows:
        return (
            cached_rows,
            str(cached.get("source") or "Cached"),
            "CACHED",
            str(cached.get("ts") or ""),
        )

    return [], "Unavailable", "EMPTY", None


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

    render_page_header(
        st,
        title="Market Heatmap",
        intro_html=(
            "Top market-cap coins in one map. "
            f"{_tip('Tile Size', 'Bigger tile means larger market cap.')} "
            f"{_tip('Tile Color', 'Green = positive 24h return, red = negative. Intensity shows move strength.')} "
            "Use this tab to read breadth and leadership quickly."
        ),
    )
    with st.spinner("Loading market data for heatmap..."):
        rows, source, feed_mode, asof_utc = _load_heatmap_rows(st, limit=180, live_ttl_sec=90)

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
        st.info("CoinGecko fallback active: data is currently sourced from CoinPaprika.")

    df_hm = pd.DataFrame(rows)
    if df_hm.empty:
        st.warning("No valid data for heatmap.")
        return
    if "Stablecoin" not in df_hm.columns:
        df_hm["Stablecoin"] = df_hm["Symbol"].apply(_is_stablecoin)
    if "TreemapKey" not in df_hm.columns:
        df_hm["TreemapKey"] = [
            f"legacy:{sym}:{i}" for i, sym in enumerate(df_hm["Symbol"].astype(str), start=1)
        ]
    if exclude_stablecoins:
        df_hm = df_hm[~df_hm["Stablecoin"]].copy()
    if df_hm.empty:
        st.warning("No non-stablecoin rows left after filter. Disable stablecoin exclusion to view full map.")
        return
    df_hm = df_hm.sort_values("Market Cap", ascending=False).head(100).copy()

    flat_eps = 0.05
    chg = df_hm["Change 24h (%)"]
    adv = int((chg > flat_eps).sum())
    dec = int((chg < -flat_eps).sum())
    flat = int(len(df_hm) - adv - dec)
    total = max(len(df_hm), 1)
    avg_chg = float(df_hm["Change 24h (%)"].mean())
    mcap_sum = float(df_hm["Market Cap"].sum())
    cap_weighted = float(
        (df_hm["Change 24h (%)"] * df_hm["Market Cap"]).sum() / mcap_sum
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

    max_abs = float(df_hm["Change 24h (%)"].abs().quantile(0.95))
    color_bound = max(3.0, min(15.0, max_abs))
    fig = px.treemap(
        df_hm,
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

    st.markdown(
        f"<div class='god-header'><b style='color:{NEON_BLUE};'>Top Movers (24h)</b></div>",
        unsafe_allow_html=True,
    )
    df_sorted = df_hm.sort_values("Change 24h (%)", ascending=False).copy()
    top_g = df_sorted.head(10).copy()
    top_l = df_sorted.tail(10).iloc[::-1].copy()
    top_g["Status"] = top_g["Change 24h (%)"].apply(
        lambda x: "▲ Strong" if x >= 8 else ("■ Moderate" if x >= 3 else "• Mild")
    )
    top_l["Status"] = top_l["Change 24h (%)"].apply(
        lambda x: "▼ Heavy" if x <= -8 else ("■ Moderate" if x <= -3 else "• Mild")
    )
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
        f"Coverage: {len(df_hm)} coins | Advancers: {adv} | Decliners: {dec} | Flat: {flat} (|move| < {flat_eps:.2f}%)"
    )
