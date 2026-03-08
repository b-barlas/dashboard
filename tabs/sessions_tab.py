from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objs as go

from ui.ctx import get_ctx
from ui.primitives import render_help_details, render_insight_card, render_kpi_grid, render_page_header
from ui.snapshot_cache import live_or_snapshot


SESSION_ORDER = ["Asian (00-08 UTC)", "European (08-16 UTC)", "US (16-00 UTC)"]
SNAPSHOT_TTL_SEC = 1800
DRIFT_BIAS_DEADBAND_PCT = 0.02


def _format_volume_compact(value: float) -> str:
    value = float(value)
    if value >= 1_000_000_000:
        return f"{value / 1_000_000_000:.2f}B"
    if value >= 1_000_000:
        return f"{value / 1_000_000:.2f}M"
    if value >= 1_000:
        return f"{value / 1_000:.2f}K"
    return f"{value:,.0f}"


def _session_bucket(hour: int) -> str:
    if 0 <= hour < 8:
        return SESSION_ORDER[0]
    if 8 <= hour < 16:
        return SESSION_ORDER[1]
    return SESSION_ORDER[2]


def _relative_quality_label(score: float) -> str:
    if score >= 68:
        return "Leading"
    if score >= 45:
        return "Balanced"
    return "Lagging"


def _liquidity_label(norm: float) -> str:
    if norm >= 0.66:
        return "Deep"
    if norm >= 0.33:
        return "Average"
    return "Thin"


def _range_profile_label(fit: float) -> str:
    if fit >= 0.66:
        return "Controlled"
    if fit >= 0.33:
        return "Tradable"
    return "Stretched"


def _drift_bias_label(avg_return: float) -> str:
    if avg_return > DRIFT_BIAS_DEADBAND_PCT:
        return "Up Tilt"
    if avg_return < -DRIFT_BIAS_DEADBAND_PCT:
        return "Down Tilt"
    return "Flat"


def _compute_session_metrics(df: pd.DataFrame) -> pd.DataFrame:
    clean = df.copy()
    clean["timestamp"] = pd.to_datetime(clean["timestamp"], utc=True, errors="coerce")
    clean = clean.dropna(subset=["timestamp", "open", "high", "low", "close", "volume"]).sort_values("timestamp")
    clean["hour"] = clean["timestamp"].dt.hour
    clean["range_pct"] = ((clean["high"] - clean["low"]) / clean["close"].replace(0, np.nan)) * 100.0
    clean["return_pct"] = ((clean["close"] - clean["open"]) / clean["open"].replace(0, np.nan)) * 100.0
    clean["abs_return_pct"] = clean["return_pct"].abs()
    clean["session"] = clean["hour"].apply(_session_bucket)

    grouped = clean.groupby("session").agg(
        avg_volume=("volume", "mean"),
        total_volume=("volume", "sum"),
        avg_range=("range_pct", "mean"),
        avg_return=("return_pct", "mean"),
        avg_abs_return=("abs_return_pct", "mean"),
        candle_count=("close", "count"),
    ).reindex(SESSION_ORDER)
    grouped = grouped.fillna(0.0)

    vol_min, vol_max = grouped["avg_volume"].min(), grouped["avg_volume"].max()
    absret_min, absret_max = grouped["avg_abs_return"].min(), grouped["avg_abs_return"].max()
    range_median = float(grouped["avg_range"].median()) if len(grouped) else 0.0

    grouped["volume_norm"] = ((grouped["avg_volume"] - vol_min) / (vol_max - vol_min + 1e-9)).clip(0.0, 1.0)
    grouped["absret_norm"] = (
        (grouped["avg_abs_return"] - absret_min) / (absret_max - absret_min + 1e-9)
    ).clip(0.0, 1.0)
    grouped["range_fit"] = (
        1.0 - (grouped["avg_range"] - range_median).abs() / (abs(range_median) + 1e-9)
    ).clip(lower=0.0, upper=1.0)
    grouped["relative_quality"] = (
        100.0
        * (
            0.50 * grouped["volume_norm"]
            + 0.30 * grouped["range_fit"]
            + 0.20 * grouped["absret_norm"]
        )
    ).round(1)
    grouped["relative_label"] = grouped["relative_quality"].apply(_relative_quality_label)
    grouped["liquidity_label"] = grouped["volume_norm"].apply(_liquidity_label)
    grouped["range_profile_label"] = grouped["range_fit"].apply(_range_profile_label)
    grouped["drift_bias_label"] = grouped["avg_return"].apply(_drift_bias_label)
    return grouped


def render(ctx: dict) -> None:
    st = get_ctx(ctx, "st")
    ACCENT = get_ctx(ctx, "ACCENT")
    TEXT_MUTED = get_ctx(ctx, "TEXT_MUTED")
    WARNING = get_ctx(ctx, "WARNING")
    POSITIVE = get_ctx(ctx, "POSITIVE")
    NEGATIVE = get_ctx(ctx, "NEGATIVE")
    _normalize_coin_input = get_ctx(ctx, "_normalize_coin_input")
    _validate_coin_symbol = get_ctx(ctx, "_validate_coin_symbol")
    fetch_ohlcv = get_ctx(ctx, "fetch_ohlcv")
    EXCHANGE = get_ctx(ctx, "EXCHANGE")
    _tip = get_ctx(ctx, "_tip")

    def _label_color(label: str) -> str:
        if label in {"Leading", "Deep", "Controlled", "Up Tilt"}:
            return POSITIVE
        if label in {"Balanced", "Average", "Tradable", "Flat"}:
            return WARNING
        return NEGATIVE

    render_page_header(
        st,
        title="Session Analysis",
        intro_html=(
            "Compares market behavior across 3 UTC sessions using 1h candles: Asian (00-08), European (08-16), and US (16-00). "
            "It is an <b>execution timing filter</b>, not a standalone entry signal. "
            f"Use it to see which session is relatively {_tip('Deeper', 'Higher average volume means cleaner fills and usually tighter spreads.')}, "
            f"{_tip('Controlled', 'A balanced range profile is easier to execute than a session that is completely dead or overstretched.')}, "
            "and directionally tilted. All labels here are <b>relative across these 3 sessions only</b>."
        ),
    )
    with st.form("sessions_analysis_form"):
        c1, c2 = st.columns(2)
        with c1:
            coin_s = _normalize_coin_input(
                st.text_input("Coin (e.g. BTC, ETH, TAO)", value="BTC", key="session_coin_input")
            )
        with c2:
            lookback = st.selectbox("Lookback Candles (1h)", [240, 360, 500], index=2, key="session_lookback")
        submitted = st.form_submit_button("Analyse Sessions", type="primary")

    current_sig = (coin_s, int(lookback), "1h")
    state_key = "sessions_analysis_result"

    if submitted:
        val_err = _validate_coin_symbol(coin_s)
        if val_err:
            st.error(val_err)
            return

        with st.spinner("Fetching hourly data for session analysis..."):
            live_df = fetch_ohlcv(coin_s, "1h", limit=int(lookback))
            snapshot_key = f"sessions_ohlcv::{coin_s}::1h::{int(lookback)}"
            df, used_cache, cache_ts = live_or_snapshot(
                st,
                snapshot_key,
                live_df,
                max_age_sec=SNAPSHOT_TTL_SEC,
                current_sig=current_sig,
            )
            if df is None or len(df) == 0:
                st.error(
                    f"Unable to fetch 1h candles for **{coin_s}** from **{EXCHANGE.id.title()}** and no fresh snapshot is available."
                )
                return
            grouped = _compute_session_metrics(df)
            if int(grouped["candle_count"].sum()) < 48:
                st.error("Not enough clean hourly candles after data cleanup (need at least 48).")
                return

            st.session_state[state_key] = {
                "sig": current_sig,
                "grouped": grouped,
                "used_cache": used_cache,
                "cache_ts": cache_ts,
                "coin": coin_s,
            }

    result = st.session_state.get(state_key)
    if not result or result.get("sig") != current_sig:
        return

    grouped = result["grouped"]
    used_cache = bool(result.get("used_cache"))
    cache_ts = result.get("cache_ts")

    if used_cache and cache_ts:
        st.warning(
            f"Live session feed was unavailable. Showing cached 1h snapshot from **{cache_ts}** for **{coin_s}**."
        )

    session_colors = [WARNING, POSITIVE, NEGATIVE]
    session_cols = st.columns(3)
    for idx, (session_name, row) in enumerate(grouped.iterrows()):
        rel_label = str(row["relative_label"])
        rel_color = _label_color(rel_label)
        drift_label = str(row["drift_bias_label"])
        drift_color = _label_color(drift_label)
        with session_cols[idx]:
            st.markdown(
                f"<div class='metric-card'>"
                f"<div class='metric-label'>{session_name}</div>"
                f"<div class='metric-value' style='font-size:1.2rem;'>Relative Quality {row['relative_quality']:.0f}</div>"
                f"<div style='color:{TEXT_MUTED};font-size:0.84rem; line-height:1.55;'>"
                f"Avg Vol: {_format_volume_compact(row['avg_volume'])} | "
                f"Range: {row['avg_range']:.2f}% | "
                f"Avg |Move|: {row['avg_abs_return']:.2f}%"
                f"</div>"
                f"<span class='app-chip' style='color:{rel_color}; border-color:{rel_color}; background:rgba(0,0,0,0.28); margin-top:8px; margin-right:6px;'>{rel_label}</span>"
                f"<span class='app-chip' style='color:{drift_color}; border-color:{drift_color}; background:rgba(0,0,0,0.28); margin-top:8px; margin-right:6px;'>{drift_label}</span>"
                f"</div>",
                unsafe_allow_html=True,
            )

    best_score = grouped["relative_quality"].idxmax()
    best_liquidity = grouped["avg_volume"].idxmax()
    fastest_tape = grouped["avg_abs_return"].idxmax()
    data_coverage = int(grouped["candle_count"].sum())
    leader_label = str(grouped.loc[best_score, "relative_label"])
    if leader_label == "Leading":
        profile_title = "Leading Execution Window"
        profile_color = POSITIVE
        profile_note = "This session currently offers the cleanest relative mix of liquidity, movement and usable range."
    elif leader_label == "Balanced":
        profile_title = "Mixed Execution Window"
        profile_color = WARNING
        profile_note = "Conditions are usable, but the edge is only moderate relative to the other sessions."
    else:
        profile_title = "Weak Relative Window"
        profile_color = NEGATIVE
        profile_note = "All sessions look soft; the best one is only less weak than the others."

    render_kpi_grid(
        st,
        items=[
            {
                "label": "Relative Leader",
                "value": best_score,
                "subtext": (
                    f"Quality {grouped.loc[best_score, 'relative_quality']:.1f} · "
                    f"{grouped.loc[best_score, 'relative_label']}"
                ),
            },
            {
                "label": "Deepest Liquidity",
                "value": best_liquidity,
                "subtext": (
                    f"Avg Vol {_format_volume_compact(grouped.loc[best_liquidity, 'avg_volume'])} · "
                    f"{grouped.loc[best_liquidity, 'liquidity_label']}"
                ),
            },
            {
                "label": "Fastest Tape",
                "value": fastest_tape,
                "subtext": (
                    f"Avg |Move| {grouped.loc[fastest_tape, 'avg_abs_return']:.2f}% · "
                    f"Drift {grouped.loc[fastest_tape, 'drift_bias_label']}"
                ),
            },
            {
                "label": "Candle Coverage",
                "value": data_coverage,
                "subtext": "1h candles across all session buckets",
            },
        ],
    )
    render_insight_card(
        st,
        title=f"Execution Timing Read · {profile_title}",
        body_html=f"<span style='color:{profile_color}; font-weight:700;'>{profile_note}</span>",
        tone=(
            "positive"
            if profile_color == POSITIVE
            else ("negative" if profile_color == NEGATIVE else "warning")
        ),
    )

    render_help_details(
        st,
        summary="How to read quickly (?)",
        body_html=(
            "<b>1.</b> Start with <b>Relative Leader</b>: this is the best session among Asia / Europe / US <b>for this sample only</b>.<br>"
            "<b>2.</b> Use <b>Deepest Liquidity</b> when your setup already exists and you want cleaner fills.<br>"
            "<b>3.</b> Use <b>Fastest Tape</b> when you want movement, but remember fast tape can also mean harder execution.<br>"
            "<b>4.</b> <b>Up Tilt / Down Tilt</b> is timing context, not a buy/sell signal by itself.<br>"
            "<b>5.</b> Use this tab to choose <b>when</b> to execute; use Market / Spot / Position to decide <b>what</b> to trade."
        ),
    )

    render_help_details(
        st,
        summary="Metric Guide",
        body_html=(
            "<b>Session</b>: The UTC block being evaluated (Asia / Europe / US).<br>"
            "<b>Relative Quality (0-100)</b>: A relative timing score across the 3 sessions only. "
            "It blends liquidity 50%, usable range 30%, and movement quality 20%. "
            "It does <b>not</b> mean a guaranteed edge by itself.<br>"
            "<b>Relative Read</b>: Quick label for the score band. Leading = best of the three right now, Balanced = usable, Lagging = weakest.<br>"
            "<b>Avg Volume</b>: Average hourly traded size inside that session. Higher usually means cleaner fills.<br>"
            "<b>Avg Range (%)</b>: Average candle high-low width. Higher means wider tape and usually bigger stop distance.<br>"
            "<b>Avg Return (%)</b>: Average hourly drift. Positive = mild upward tilt, negative = mild downward tilt.<br>"
            "<b>Avg |Move| (%)</b>: Average absolute candle move size. Higher means faster tape.<br>"
            "<b>Candles</b>: Number of hourly candles in the sample for that session bucket.<br>"
            "<b>Liquidity</b>: Deep / Average / Thin volume context <b>relative to the other 2 sessions in this sample</b>.<br>"
            "<b>Range Profile</b>: Controlled / Tradable / Stretched range balance for execution, also <b>relative across the 3 sessions</b>.<br>"
            f"<b>Drift Bias</b>: Up Tilt / Down Tilt / Flat hourly drift context. Tiny drift inside +/-{DRIFT_BIAS_DEADBAND_PCT:.2f}% is treated as Flat to avoid noise."
        ),
    )

    with st.expander("Advanced Session Charts (optional diagnostics)"):
        st.caption(
            "Use these only as visual support: the first chart ranks liquidity by session; "
            "the second shows whether movement is coming from usable range expansion or just noisy tape."
        )
        fig_vol = go.Figure()
        for idx, session_name in enumerate(SESSION_ORDER):
            if session_name in grouped.index:
                fig_vol.add_trace(
                    go.Bar(
                        x=[session_name],
                        y=[grouped.loc[session_name, "avg_volume"]],
                        name=session_name,
                        marker_color=session_colors[idx],
                    )
                )
        fig_vol.update_layout(
            title="Liquidity Depth by Session",
            height=300,
            template="plotly_dark",
            margin=dict(l=20, r=20, t=40, b=30),
            showlegend=False,
        )
        st.plotly_chart(fig_vol, width="stretch")

        fig_range_ret = go.Figure()
        fig_range_ret.add_trace(
            go.Bar(
                x=SESSION_ORDER,
                y=[grouped.loc[s, "avg_range"] for s in SESSION_ORDER],
                name="Avg Range (%)",
                marker_color=[WARNING, POSITIVE, NEGATIVE],
            )
        )
        fig_range_ret.add_trace(
            go.Scatter(
                x=SESSION_ORDER,
                y=[grouped.loc[s, "avg_abs_return"] for s in SESSION_ORDER],
                name="Avg |Move| (%)",
                mode="lines+markers",
                line=dict(color=ACCENT, width=2),
            )
        )
        fig_range_ret.update_layout(
            title="Range Profile vs Tape Speed",
            height=320,
            template="plotly_dark",
            margin=dict(l=20, r=20, t=40, b=30),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        )
        st.plotly_chart(fig_range_ret, width="stretch")

    summary_df = grouped.reset_index().rename(
        columns={
            "session": "Session",
            "avg_volume": "Avg Volume",
            "avg_range": "Avg Range (%)",
            "avg_return": "Avg Return (%)",
            "avg_abs_return": "Avg |Move| (%)",
            "candle_count": "Candles",
            "relative_quality": "Relative Quality",
            "relative_label": "Relative Read",
            "liquidity_label": "Liquidity",
            "range_profile_label": "Range Profile",
            "drift_bias_label": "Drift Bias",
        }
    )
    summary_df["Avg Volume"] = summary_df["Avg Volume"].apply(_format_volume_compact)

    def _style_label(value: str) -> str:
        color = _label_color(str(value))
        return f"color:{color}; font-weight:700;"

    st.markdown(f"<b style='color:{ACCENT};'>Session Summary Table</b>", unsafe_allow_html=True)
    st.dataframe(
        summary_df[
            [
                "Session",
                "Relative Quality",
                "Relative Read",
                "Avg Volume",
                "Avg Range (%)",
                "Avg Return (%)",
                "Avg |Move| (%)",
                "Candles",
                "Liquidity",
                "Range Profile",
                "Drift Bias",
            ]
        ].style.format(
            {
                "Relative Quality": "{:.1f}",
                "Avg Range (%)": "{:.3f}",
                "Avg Return (%)": "{:+.3f}",
                "Avg |Move| (%)": "{:.3f}",
                "Candles": "{:.0f}",
            }
        ).map(_style_label, subset=["Relative Read", "Liquidity", "Range Profile", "Drift Bias"]),
        width="stretch",
        hide_index=True,
    )

    st.info(
        f"Use **{best_score}** as the preferred execution window when your setup already exists. "
        f"If you need clean fills, prioritize **{best_liquidity}**. "
        f"If you need movement, monitor **{fastest_tape}**, but expect harder execution."
    )
