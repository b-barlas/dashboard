from ui.ctx import get_ctx

import numpy as np
import pandas as pd
import plotly.graph_objs as go


def render(ctx: dict) -> None:
    st = get_ctx(ctx, "st")
    ACCENT = get_ctx(ctx, "ACCENT")
    TEXT_MUTED = get_ctx(ctx, "TEXT_MUTED")
    _tip = get_ctx(ctx, "_tip")
    _normalize_coin_input = get_ctx(ctx, "_normalize_coin_input")
    fetch_ohlcv = get_ctx(ctx, "fetch_ohlcv")
    EXCHANGE = get_ctx(ctx, "EXCHANGE")
    """Render a correlation matrix for major crypto assets."""
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
        f"for major + custom coins. Returns are time-aligned by candle timestamp before correlation is computed, "
        f"which avoids false relationships from misaligned series. Use this for "
        f"{_tip('portfolio diversification', 'Combining lower-correlation assets reduces concentration risk when one coin dumps.')} "
        f"and to spot over-crowded exposure.</p>"
        f"</div>",
        unsafe_allow_html=True,
    )
    corr_c1, corr_c2, corr_c3 = st.columns(3)
    with corr_c1:
        tf_corr = st.selectbox("Timeframe", ["15m", "1h", "4h", "1d"], index=3, key="corr_tf")
    with corr_c2:
        corr_method = st.selectbox(
            "Method",
            ["pearson", "spearman"],
            index=0,
            key="corr_method",
            help=(
                "Pearson: linear co-movement strength. "
                "Spearman: rank-based monotonic relationship, more robust to outliers/non-linear scaling."
            ),
        )
    with corr_c3:
        custom_coins_raw = st.text_input(
            "Add coins (up to 4, comma-separated)",
            value="",
            placeholder="e.g. DOGE, TAO, LINK, FET",
            key="corr_custom_coin",
        ).upper().strip()
    if st.button("Generate Correlation Matrix", type="primary"):
        symbols = ["BTC/USDT", "ETH/USDT", "BNB/USDT", "SOL/USDT", "ADA/USDT", "XRP/USDT"]
        # Parse comma-separated custom coins (up to 4)
        if custom_coins_raw:
            custom_list = [c.strip() for c in custom_coins_raw.split(",") if c.strip()][:4]
            for cc in custom_list:
                normalized = _normalize_coin_input(cc)
                if normalized and normalized not in symbols:
                    symbols.append(normalized)
        labels = [s.split("/")[0] for s in symbols]
        with st.spinner("Fetching data for correlation analysis..."):
            returns_dict: dict[str, pd.Series] = {}
            failed_coins = []
            for sym, label in zip(symbols, labels):
                df = fetch_ohlcv(sym, tf_corr, limit=200)
                if df is not None and len(df) > 10:
                    rets = df["close"].pct_change().dropna()
                    if rets.empty:
                        failed_coins.append(sym)
                        continue
                    if "timestamp" in df.columns:
                        idx = pd.to_datetime(df.loc[rets.index, "timestamp"], unit="ms", errors="coerce")
                    else:
                        idx = pd.to_datetime(df.index, errors="coerce")
                    s = pd.Series(rets.values, index=idx, name=label).dropna()
                    # Remove duplicated timestamps to ensure clean alignment.
                    s = s[~s.index.duplicated(keep="last")]
                    if len(s) < 10:
                        failed_coins.append(sym)
                        continue
                    returns_dict[label] = s
                else:
                    failed_coins.append(sym)
            if failed_coins:
                st.warning(
                    f"Could not fetch data for: **{', '.join(failed_coins)}**. "
                    f"These coins were not found on {EXCHANGE.id.title()} or CoinGecko. "
                    f"Please check the symbol is correct."
                )
            if len(returns_dict) < 2:
                st.error("Not enough data to compute correlations.")
                return
            # Time-align by candle timestamp (inner join) for statistically consistent comparison.
            df_ret = pd.concat(returns_dict.values(), axis=1, join="inner").dropna()
            if len(df_ret) < 20:
                st.error(
                    "Aligned data is too short to produce a stable matrix. Try fewer custom coins or a higher timeframe."
                )
                return

            corr = df_ret.corr(method=corr_method)
            np.fill_diagonal(corr.values, np.nan)
            pair_count = int((len(corr.columns) * (len(corr.columns) - 1)) / 2)

            # Heatmap
            fig_corr = go.Figure(data=go.Heatmap(
                z=corr.values,
                x=corr.columns.tolist(),
                y=corr.index.tolist(),
                colorscale="RdBu",
                zmin=-1, zmax=1,
                text=np.round(corr.values, 2),
                texttemplate="%{text}",
                textfont={"size": 13},
                colorbar=dict(title="Corr"),
            ))
            fig_corr.update_layout(
                height=520,
                template="plotly_dark",
                margin=dict(l=60, r=20, t=45, b=60),
                xaxis=dict(side="bottom"),
                title=f"{corr_method.title()} Correlation ({tf_corr})",
            )
            st.plotly_chart(fig_corr, width="stretch")

            # Insights
            pairs = []
            cols_list = list(corr.columns)
            for i in range(len(cols_list)):
                for j in range(i + 1, len(cols_list)):
                    pairs.append((cols_list[i], cols_list[j], corr.iloc[i, j]))
            pairs = [p for p in pairs if not pd.isna(p[2])]
            if not pairs:
                st.error("No valid pair correlations after alignment.")
                return
            pairs.sort(key=lambda x: x[2])
            most_corr = max(pairs, key=lambda x: x[2])
            least_corr = min(pairs, key=lambda x: x[2])
            abs_vals = [abs(p[2]) for p in pairs]
            avg_abs_corr = float(np.mean(abs_vals)) if abs_vals else 0.0

            base = "BTC" if "BTC" in corr.columns else corr.columns[0]
            base_rel = (
                corr[base].drop(labels=[base], errors="ignore").dropna().sort_values()
                if base in corr.columns else pd.Series(dtype=float)
            )
            diversifier = base_rel.index[0] if not base_rel.empty else "N/A"
            diversifier_val = float(base_rel.iloc[0]) if not base_rel.empty else np.nan

            st.markdown(
                f"<div class='corr-kpi-grid'>"
                f"<div class='corr-kpi'><div class='corr-kpi-label'>Method</div><div class='corr-kpi-value'>{corr_method.title()}</div></div>"
                f"<div class='corr-kpi'><div class='corr-kpi-label'>Aligned Samples</div><div class='corr-kpi-value'>{len(df_ret)}</div></div>"
                f"<div class='corr-kpi'><div class='corr-kpi-label'>Pair Count</div><div class='corr-kpi-value'>{pair_count}</div></div>"
                f"<div class='corr-kpi'><div class='corr-kpi-label'>Avg |Corr|</div><div class='corr-kpi-value'>{avg_abs_corr:.2f}</div></div>"
                f"</div>",
                unsafe_allow_html=True,
            )

            st.markdown(
                f"<details style='margin-bottom:0.7rem;'>"
                f"<summary style='color:{ACCENT}; cursor:pointer;'>How to read quickly (?)</summary>"
                f"<div style='color:{TEXT_MUTED}; font-size:0.85rem; line-height:1.7; margin-top:0.5rem;'>"
                f"<b>1.</b> Start with <b>Avg |Corr|</b>: lower value means better diversification across the basket.<br>"
                f"<b>2.</b> For new positions, avoid adding coins that are already <b>> 0.80</b> correlated with your core holding.<br>"
                f"<b>3.</b> Use the <b>Best Diversifier vs {base}</b> insight to reduce single-theme exposure.<br>"
                f"<b>4.</b> Re-check on 4h/1d before swing decisions; lower timeframes are noisier."
                f"</div></details>",
                unsafe_allow_html=True,
            )

            summary_df = pd.DataFrame([
                {"Insight": "Most Correlated Pair", "Value": f"{most_corr[0]} - {most_corr[1]}", "Correlation": round(float(most_corr[2]), 2)},
                {"Insight": "Least Correlated Pair", "Value": f"{least_corr[0]} - {least_corr[1]}", "Correlation": round(float(least_corr[2]), 2)},
                {"Insight": f"Best Diversifier vs {base}", "Value": diversifier, "Correlation": (round(diversifier_val, 2) if not np.isnan(diversifier_val) else "N/A")},
            ])
            st.dataframe(summary_df, width="stretch", hide_index=True)

            pair_df = (
                pd.DataFrame(pairs, columns=["Coin A", "Coin B", "Correlation"])
                .assign(
                    AbsCorr=lambda d: d["Correlation"].abs(),
                    Regime=lambda d: np.select(
                        [d["Correlation"] >= 0.8, d["Correlation"] <= -0.3, d["Correlation"].abs() < 0.3],
                        ["High Co-Move", "Inverse/Defensive", "Low Co-Move"],
                        default="Medium Co-Move",
                    ),
                )
                .sort_values("AbsCorr", ascending=False)
            )
            st.markdown(f"<b style='color:{ACCENT};'>Pair Table</b>", unsafe_allow_html=True)
            st.dataframe(
                pair_df.style.format({"Correlation": "{:.2f}", "AbsCorr": "{:.2f}"}),
                width="stretch",
                hide_index=True,
            )
