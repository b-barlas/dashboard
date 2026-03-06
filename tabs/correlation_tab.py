from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objs as go

from ui.ctx import get_ctx
from ui.snapshot_cache import live_or_snapshot


DEFAULT_SYMBOLS = ["BTC/USDT", "ETH/USDT", "BNB/USDT", "SOL/USDT", "ADA/USDT", "XRP/USDT"]
MAX_CUSTOM_COINS = 10
SNAPSHOT_TTL_SEC = 1800


def _parse_custom_coins(raw: str, normalize_coin_input) -> list[str]:
    seen: set[str] = set()
    parsed: list[str] = []
    for coin in [c.strip() for c in raw.split(",") if c.strip()]:
        normalized = normalize_coin_input(coin)
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        parsed.append(normalized)
        if len(parsed) >= MAX_CUSTOM_COINS:
            break
    return parsed


def _pair_regime(corr_value: float) -> str:
    if corr_value >= 0.8:
        return "High Co-Move"
    if corr_value <= -0.3:
        return "Inverse / Defensive"
    if abs(corr_value) < 0.3:
        return "Low Co-Move"
    return "Medium Co-Move"


def _pair_read(corr_value: float) -> str:
    if corr_value >= 0.8:
        return "Tight"
    if corr_value >= 0.5:
        return "Linked"
    if corr_value > 0.2:
        return "Mild"
    if corr_value > -0.2:
        return "Neutral"
    if corr_value > -0.5:
        return "Hedge"
    return "Inverse"


def _corr_style(value: float) -> str:
    if pd.isna(value):
        return "color:#94a3b8;"
    if value >= 0.8:
        return "color:#ff4d6d; font-weight:700;"
    if value >= 0.5:
        return "color:#ffd166; font-weight:700;"
    if value <= -0.3:
        return "color:#00e5a8; font-weight:700;"
    return "color:#e2e8f0;"


def _pair_read_style(label: str) -> str:
    if label == "Tight":
        return "color:#ff4d6d; font-weight:700;"
    if label in {"Linked", "Mild", "Neutral"}:
        return "color:#ffd166; font-weight:700;"
    if label in {"Hedge", "Inverse"}:
        return "color:#00e5a8; font-weight:700;"
    return "color:#e2e8f0;"


def _returns_series(df: pd.DataFrame, label: str) -> pd.Series | None:
    if df is None or len(df) <= 10:
        return None
    rets = df["close"].pct_change().dropna()
    if rets.empty:
        return None
    if "timestamp" in df.columns:
        ts_vals = df.loc[rets.index, "timestamp"]
        if pd.api.types.is_numeric_dtype(ts_vals):
            idx = pd.to_datetime(ts_vals, unit="ms", errors="coerce", utc=True)
        else:
            idx = pd.to_datetime(ts_vals, errors="coerce", utc=True)
    else:
        idx = pd.to_datetime(df.index, errors="coerce", utc=True)
    series = pd.Series(rets.values, index=idx, name=label).dropna()
    series = series[~series.index.duplicated(keep="last")]
    if len(series) < 10:
        return None
    return series


def _basket_insight_payload(
    pair_df: pd.DataFrame,
    avg_abs_corr: float,
    base: str,
    diversifier: str,
    diversifier_val: float,
) -> tuple[str, list[str]]:
    total_pairs = max(len(pair_df), 1)
    tight_pairs = int((pair_df["AbsCorr"] >= 0.8).sum())
    low_pairs = int((pair_df["AbsCorr"] < 0.3).sum())
    hedge_pairs = int((pair_df["Correlation"] <= -0.3).sum())
    tight_share = tight_pairs / total_pairs
    low_share = low_pairs / total_pairs

    badges = [
        f"Tight Pairs {tight_pairs}/{total_pairs}",
        f"Low Co-Move {low_pairs}/{total_pairs}",
        f"Hedge Pairs {hedge_pairs}",
    ]

    diversifier_text = (
        f"Best diversifier vs {base}: {diversifier} ({diversifier_val:.2f})"
        if diversifier != "N/A" and not np.isnan(diversifier_val)
        else f"Best diversifier vs {base}: N/A"
    )
    badges.append(diversifier_text)

    if avg_abs_corr >= 0.65 or tight_share >= 0.45:
        body = (
            "Most pairs are moving together. This basket behaves like one crowded theme, "
            "so adding another coin here may increase size more than diversification."
        )
    elif hedge_pairs >= 1 and avg_abs_corr < 0.55:
        body = (
            "The basket still has linked names, but at least one pair is acting as a hedge. "
            "This reduces overlap better than a pure majors stack."
        )
    elif avg_abs_corr < 0.40 and low_share >= 0.45:
        body = (
            "Several pairs have low co-movement. This basket is less likely to behave like a single trade, "
            "so diversification quality is cleaner."
        )
    else:
        body = (
            "Some names are linked, some are not. The basket is usable, but overlap should still be checked "
            "before adding another coin from the same theme."
        )

    return body, badges


def render(ctx: dict) -> None:
    st = get_ctx(ctx, "st")
    ACCENT = get_ctx(ctx, "ACCENT")
    TEXT_MUTED = get_ctx(ctx, "TEXT_MUTED")
    _tip = get_ctx(ctx, "_tip")
    _normalize_coin_input = get_ctx(ctx, "_normalize_coin_input")
    fetch_ohlcv = get_ctx(ctx, "fetch_ohlcv")
    EXCHANGE = get_ctx(ctx, "EXCHANGE")

    st.markdown(
        f"""
        <style>
        .corr-kpi-grid {{
            display:grid;
            grid-template-columns:repeat(4,minmax(0,1fr));
            gap:10px;
            margin:10px 0 12px 0;
        }}
        .corr-kpi {{
            border:1px solid rgba(0,212,255,0.16);
            border-radius:12px;
            padding:12px 14px;
            background:linear-gradient(140deg, rgba(0,0,0,0.72), rgba(10,18,30,0.88));
        }}
        .corr-kpi-label {{
            color:{TEXT_MUTED};
            font-size:0.70rem;
            text-transform:uppercase;
            letter-spacing:0.8px;
        }}
        .corr-kpi-value {{
            color:{ACCENT};
            font-size:1.2rem;
            font-weight:700;
            margin-top:4px;
        }}
        .corr-kpi-sub {{
            color:{TEXT_MUTED};
            font-size:0.80rem;
            margin-top:4px;
            line-height:1.45;
        }}
        .corr-insight {{
            border:1px solid rgba(0,212,255,0.20);
            border-left:4px solid {ACCENT};
            border-radius:12px;
            padding:14px 16px;
            background:linear-gradient(140deg, rgba(0,0,0,0.76), rgba(8,18,32,0.92));
            margin:10px 0 14px 0;
        }}
        .corr-insight-title {{
            color:{ACCENT};
            font-size:1rem;
            font-weight:700;
            margin-bottom:6px;
        }}
        .corr-insight-body {{
            color:{TEXT_MUTED};
            font-size:0.87rem;
            line-height:1.6;
            margin-bottom:8px;
        }}
        .corr-insight-badges {{
            display:flex;
            flex-wrap:wrap;
            gap:8px;
        }}
        .corr-insight-badge {{
            border:1px solid rgba(255,255,255,0.14);
            border-radius:999px;
            padding:5px 10px;
            color:{TEXT_MUTED};
            font-size:0.78rem;
            background:rgba(255,255,255,0.03);
        }}
        @media (max-width: 1100px) {{
            .corr-kpi-grid {{
                grid-template-columns:repeat(2,minmax(0,1fr));
            }}
        }}
        @media (max-width: 720px) {{
            .corr-kpi-grid {{
                grid-template-columns:1fr;
            }}
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(f"<h2 style='color:{ACCENT};'>Correlation Matrix</h2>", unsafe_allow_html=True)
    st.markdown(
        f"<div class='panel-box'>"
        f"<b style='color:{ACCENT}; font-size:1rem;'>What does this tab show?</b>"
        f"<p style='color:{TEXT_MUTED}; font-size:0.9rem; margin-top:6px; line-height:1.6;'>"
        f"Builds a return-based {_tip('correlation matrix', 'A grid showing how closely two assets move together. +1.0 = perfectly correlated, -1.0 = inverse, 0 = unrelated.')} "
        f"for the basket you actually hold or plan to hold. Returns are time-aligned by candle timestamp before correlation is computed, "
        f"which avoids false relationships from misaligned series. Use this for "
        f"{_tip('portfolio diversification', 'Combining lower-correlation assets reduces concentration risk when one coin dumps.')} "
        f"and to spot over-crowded exposure. This tab is most useful when you enter your own basket, "
        f"not when you just load default majors."
        f"</p></div>",
        unsafe_allow_html=True,
    )

    if "corr_custom_coin" not in st.session_state:
        st.session_state["corr_custom_coin"] = ""

    def _load_corr_example() -> None:
        st.session_state["corr_custom_coin"] = ", ".join([s.split("/")[0] for s in DEFAULT_SYMBOLS])

    corr_c1, corr_c2, corr_c3, corr_c4 = st.columns([0.8, 0.8, 1.55, 0.55])
    with corr_c1:
        tf_corr = st.selectbox("Timeframe", ["15m", "1h", "4h", "1d"], index=3, key="corr_tf")
    with corr_c2:
        corr_method = st.selectbox(
            "Method",
            ["pearson", "spearman"],
            index=0,
            key="corr_method",
            help=(
                "Pearson is the default method for direct co-movement. "
                "Spearman is a secondary cross-check that is more robust to outliers."
            ),
        )
    with corr_c3:
        custom_coins_raw = st.text_input(
            f"Custom Coins (optional, max {MAX_CUSTOM_COINS})",
            placeholder="e.g. DOGE, TAO, LINK, FET",
            key="corr_custom_coin",
        ).upper().strip()
    with corr_c4:
        st.markdown("<div style='height:1.95rem;'></div>", unsafe_allow_html=True)
        st.button(
            "Load Majors Example",
            key="corr_load_example",
            on_click=_load_corr_example,
            help=(
                "If you leave Custom Coins empty, the tab will wait for your basket. "
                "Use Load Majors Example only as a demo."
            ),
            width="stretch",
        )
    st.caption(
        f"Enter the coins you already hold or plan to hold. "
        f"Add up to {MAX_CUSTOM_COINS} symbols to test your real portfolio overlap. "
        f"Method note: use Pearson as your main read; use Spearman only if you want a rank-based cross-check."
    )

    current_sig = (tf_corr, corr_method, custom_coins_raw)
    state_key = "correlation_analysis_result"

    if st.button("Generate Correlation Matrix", type="primary"):
        custom_symbols = _parse_custom_coins(custom_coins_raw, _normalize_coin_input)
        if not custom_symbols:
            st.info(
                f"Enter up to {MAX_CUSTOM_COINS} coins from your basket, or click **Load Majors Example** if you want a demo basket."
            )
            return
        symbols = list(custom_symbols)
        labels = [s.split("/")[0] for s in symbols]

        with st.spinner("Fetching data for correlation analysis..."):
            returns_dict: dict[str, pd.Series] = {}
            cached_symbols: list[str] = []
            failed_coins: list[str] = []
            for sym, label in zip(symbols, labels):
                live_df = fetch_ohlcv(sym, tf_corr, limit=200)
                snapshot_key = f"corr_ohlcv::{sym}::{tf_corr}::200"
                df, used_cache, _cache_ts = live_or_snapshot(
                    st,
                    snapshot_key,
                    live_df,
                    max_age_sec=SNAPSHOT_TTL_SEC,
                    current_sig=(sym, tf_corr, 200),
                )
                if used_cache:
                    cached_symbols.append(label)
                series = _returns_series(df, label)
                if series is None:
                    failed_coins.append(label)
                    continue
                returns_dict[label] = series

            if len(returns_dict) < 2:
                st.error("Not enough usable price series to compute correlations.")
                return

            df_ret = pd.concat(returns_dict.values(), axis=1, join="inner").dropna()
            if len(df_ret) < 20:
                st.error(
                    "Matched return history is too short to produce a stable matrix. Try fewer custom coins or a higher timeframe."
                )
                return

            corr = df_ret.corr(method=corr_method)
            np.fill_diagonal(corr.values, np.nan)
            pair_count = int((len(corr.columns) * (len(corr.columns) - 1)) / 2)

            pairs = []
            cols_list = list(corr.columns)
            for i in range(len(cols_list)):
                for j in range(i + 1, len(cols_list)):
                    pairs.append((cols_list[i], cols_list[j], corr.iloc[i, j]))
            pairs = [p for p in pairs if not pd.isna(p[2])]
            if not pairs:
                st.error("No valid pair correlations after timestamp alignment.")
                return

            most_corr = max(pairs, key=lambda x: x[2])
            least_corr = min(pairs, key=lambda x: x[2])
            abs_vals = [abs(p[2]) for p in pairs]
            avg_abs_corr = float(np.mean(abs_vals)) if abs_vals else 0.0

            if avg_abs_corr >= 0.65:
                corr_profile = "Crowded Basket"
                corr_action = "Most names are moving together. Reduce overlap if you want true diversification."
            elif avg_abs_corr >= 0.40:
                corr_profile = "Balanced Basket"
                corr_action = "Co-movement is moderate. Diversification exists, but basket risk is still theme-sensitive."
            else:
                corr_profile = "Diversified Basket"
                corr_action = "Co-movement is relatively low. Cross-asset concentration risk is lower."

            base = "BTC" if "BTC" in corr.columns else corr.columns[0]
            base_rel = corr[base].drop(labels=[base], errors="ignore").dropna().sort_values()
            diversifier = base_rel.index[0] if not base_rel.empty else "N/A"
            diversifier_val = float(base_rel.iloc[0]) if not base_rel.empty else np.nan

            summary_df = pd.DataFrame(
                [
                    {
                        "Insight": "Most Correlated Pair",
                        "Value": f"{most_corr[0]} - {most_corr[1]}",
                        "Correlation": round(float(most_corr[2]), 2),
                    },
                    {
                        "Insight": "Least Correlated Pair",
                        "Value": f"{least_corr[0]} - {least_corr[1]}",
                        "Correlation": round(float(least_corr[2]), 2),
                    },
                    {
                        "Insight": f"Best Diversifier vs {base}",
                        "Value": diversifier,
                        "Correlation": round(diversifier_val, 2) if not np.isnan(diversifier_val) else "N/A",
                    },
                ]
            )

            pair_df = (
                pd.DataFrame(pairs, columns=["Coin A", "Coin B", "Correlation"])
                .assign(
                    AbsCorr=lambda d: d["Correlation"].abs(),
                    Regime=lambda d: d["Correlation"].apply(_pair_regime),
                    **{"Pair Read": lambda d: d["Correlation"].apply(_pair_read)},
                )
                .sort_values("AbsCorr", ascending=False)
            )

            st.session_state[state_key] = {
                "sig": current_sig,
                "corr": corr,
                "pair_df": pair_df,
                "summary_df": summary_df,
                "pair_count": pair_count,
                "matched_samples": len(df_ret),
                "avg_abs_corr": avg_abs_corr,
                "corr_profile": corr_profile,
                "corr_action": corr_action,
                "base": base,
                "most_corr": most_corr,
                "least_corr": least_corr,
                "diversifier": diversifier,
                "diversifier_val": diversifier_val,
                "cached_symbols": cached_symbols,
                "failed_coins": failed_coins,
                "symbols": list(returns_dict.keys()),
            }

    result = st.session_state.get(state_key)
    if not result or result.get("sig") != current_sig:
        return

    corr = result["corr"]
    pair_df = result["pair_df"]
    summary_df = result["summary_df"]
    pair_count = int(result["pair_count"])
    matched_samples = int(result["matched_samples"])
    avg_abs_corr = float(result["avg_abs_corr"])
    corr_profile = str(result["corr_profile"])
    corr_action = str(result["corr_action"])
    base = str(result["base"])
    cached_symbols = list(result.get("cached_symbols", []))
    failed_coins = list(result.get("failed_coins", []))
    insight_body, insight_badges = _basket_insight_payload(
        pair_df,
        avg_abs_corr,
        base,
        str(result.get("diversifier", "N/A")),
        float(result.get("diversifier_val", np.nan)),
    )

    if cached_symbols:
        st.warning(
            f"Live feed was unavailable for **{', '.join(cached_symbols)}**. "
            "Using fresh cached snapshots for those symbols."
        )
    if failed_coins:
        st.warning(
            f"No usable aligned return series for: **{', '.join(failed_coins)}**. "
            f"These symbols were excluded from the matrix on **{EXCHANGE.id.title()} ({tf_corr})**."
        )

    fig_corr = go.Figure(
        data=go.Heatmap(
            z=corr.values,
            x=corr.columns.tolist(),
            y=corr.index.tolist(),
            colorscale="RdBu",
            zmin=-1,
            zmax=1,
            text=np.round(corr.values, 2),
            texttemplate="%{text}",
            textfont={"size": 13},
            colorbar=dict(title="Corr"),
        )
    )
    fig_corr.update_layout(
        height=520,
        template="plotly_dark",
        margin=dict(l=60, r=20, t=45, b=60),
        xaxis=dict(side="bottom"),
        title=f"{corr_method.title()} Correlation ({tf_corr})",
    )
    st.plotly_chart(fig_corr, width="stretch")

    st.markdown(
        f"<div class='corr-kpi-grid'>"
        f"<div class='corr-kpi'><div class='corr-kpi-label'>Method</div><div class='corr-kpi-value'>{corr_method.title()}</div></div>"
        f"<div class='corr-kpi'><div class='corr-kpi-label'>Matched Samples</div><div class='corr-kpi-value'>{matched_samples}</div><div class='corr-kpi-sub'>Shared return bars after timestamp alignment</div></div>"
        f"<div class='corr-kpi'><div class='corr-kpi-label'>Pair Count</div><div class='corr-kpi-value'>{pair_count}</div><div class='corr-kpi-sub'>Valid relationships inside the current basket</div></div>"
        f"<div class='corr-kpi'><div class='corr-kpi-label'>Avg |Corr|</div><div class='corr-kpi-value'>{avg_abs_corr:.2f}</div><div class='corr-kpi-sub'>Lower = cleaner diversification</div></div>"
        f"</div>",
        unsafe_allow_html=True,
    )

    st.markdown(
        "<div class='corr-insight'>"
        f"<div class='corr-insight-title'>Basket Insight · {corr_profile}</div>"
        f"<div class='corr-insight-body'>{corr_action} {insight_body}</div>"
        "<div class='corr-insight-badges'>"
        + "".join(f"<span class='corr-insight-badge'>{badge}</span>" for badge in insight_badges)
        + "</div></div>",
        unsafe_allow_html=True,
    )

    st.markdown(
        f"<details style='margin-bottom:0.7rem;'>"
        f"<summary style='color:{ACCENT}; cursor:pointer;'>How to read quickly (?)</summary>"
        f"<div style='color:{TEXT_MUTED}; font-size:0.85rem; line-height:1.7; margin-top:0.5rem;'>"
        f"<b>1.</b> Start with <b>Avg |Corr|</b>: lower value means better diversification across the basket.<br>"
        f"<b>2.</b> Avoid stacking names already <b>> 0.80</b> correlated with your core holding unless you intentionally want concentrated exposure.<br>"
        f"<b>Method check:</b> "
        f"{_tip('Pearson (main method)', 'Use this first. It tells you how directly two coins move together bar by bar.')} | "
        f"{_tip('Spearman (cross-check)', 'Use this as a second look. It checks whether the ranking or general relationship stays similar even if price moves are noisy or uneven.')}<br>"
        f"<b>3.</b> Use the <b>Best Diversifier vs {base}</b> row to find the cleanest non-crowded add-on.<br>"
        f"<b>4.</b> Re-check on 4h/1d before swing decisions; lower timeframes are noisier and more temporary."
        f"</div></details>",
        unsafe_allow_html=True,
    )

    st.dataframe(summary_df, width="stretch", hide_index=True)

    st.markdown(
        f"<details style='margin-bottom:0.45rem;'>"
        f"<summary style='color:{ACCENT}; cursor:pointer;'>Column Guide (?)</summary>"
        f"<div style='color:{TEXT_MUTED}; font-size:0.85rem; line-height:1.7; margin-top:0.5rem;'>"
        f"<b>Correlation</b>: signed co-movement between pair returns (-1 to +1).<br>"
        f"<b>AbsCorr</b>: strength only (ignores direction). Higher = stronger linkage.<br>"
        f"<b>Pair Read</b>: quick interpretation of pair behavior (Tight, Linked, Mild, Hedge, Inverse).<br>"
        f"<b>Regime</b>: portfolio interpretation of the pair relation."
        f"</div></details>",
        unsafe_allow_html=True,
    )
    st.markdown(f"<b style='color:{ACCENT};'>Pair Table</b>", unsafe_allow_html=True)
    st.dataframe(
        pair_df.style.format({"Correlation": "{:.2f}", "AbsCorr": "{:.2f}"})
        .map(_corr_style, subset=["Correlation", "AbsCorr"])
        .map(_pair_read_style, subset=["Pair Read"]),
        width="stretch",
        hide_index=True,
    )
