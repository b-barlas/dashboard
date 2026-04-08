from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objs as go
from core.session_utils import SESSION_ORDER, session_bucket_for_timestamp

from ui.ctx import get_ctx
from ui.primitives import render_help_details, render_insight_card, render_kpi_grid, render_page_header
from ui.snapshot_cache import live_or_snapshot

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
    clean["session"] = clean["timestamp"].apply(session_bucket_for_timestamp)

    grouped = clean.groupby("session").agg(
        avg_volume=("volume", "mean"),
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


def _prepare_tracked_session_archive(df_events: pd.DataFrame) -> pd.DataFrame:
    if df_events is None or df_events.empty:
        return pd.DataFrame()
    d = df_events.copy()
    d["Session"] = (
        d.get("session_bucket", pd.Series(index=d.index, dtype=object))
        .replace("", "Unknown")
        .fillna("Unknown")
    )
    d["Playbook"] = (
        d.get("market_playbook", pd.Series(index=d.index, dtype=object))
        .replace("", "Unknown")
        .fillna("Unknown")
    )
    return d


def _build_playbook_session_archive(df_events: pd.DataFrame, playbook: str) -> pd.DataFrame:
    normalized_playbook = str(playbook or "").strip()
    if not normalized_playbook or normalized_playbook == "Unknown":
        return pd.DataFrame()
    archive = _prepare_tracked_session_archive(df_events)
    if archive.empty:
        return pd.DataFrame()
    scoped = archive[archive["Playbook"].astype(str).str.strip() == normalized_playbook].copy()
    if scoped.empty:
        return pd.DataFrame()
    scoped["directional_return_pct"] = pd.to_numeric(scoped.get("directional_return_pct"), errors="coerce")
    scoped["actual_pnl_pct"] = pd.to_numeric(scoped.get("actual_pnl_pct"), errors="coerce")
    scoped["status"] = scoped.get("status", pd.Series(index=scoped.index, dtype=object)).fillna("").astype(str).str.upper()
    scoped["actual_trade_status"] = (
        scoped.get("actual_trade_status", pd.Series(index=scoped.index, dtype=object)).fillna("").astype(str).str.upper()
    )
    summary = (
        scoped.groupby("Session", dropna=False)
        .agg(
            Signals=("symbol", "count"),
            Resolved=("status", lambda s: int((pd.Series(s).astype(str).str.upper() == "RESOLVED").sum())),
            FollowThroughPct=("directional_return_pct", lambda s: float((pd.to_numeric(s, errors="coerce") > 0).mean() * 100.0) if len(pd.Series(s).dropna()) else 0.0),
            ClosedTradeCount=("actual_trade_status", lambda s: int((pd.Series(s).astype(str).str.upper() == "CLOSED").sum())),
            ActualWinRatePct=("actual_pnl_pct", lambda s: float((pd.to_numeric(s, errors="coerce") > 0).mean() * 100.0) if len(pd.Series(s).dropna()) else 0.0),
            AvgActualPnlPct=("actual_pnl_pct", "mean"),
        )
        .reset_index()
    )
    if summary.empty:
        return summary
    return summary.sort_values(by=["FollowThroughPct", "Signals"], ascending=[False, False]).reset_index(drop=True)


def _playbook_timing_read(
    playbook_archive: pd.DataFrame,
    *,
    current_session: str,
    trade_gate: str,
    catalyst_window: str,
) -> dict[str, str]:
    if playbook_archive is None or playbook_archive.empty:
        return {
            "title": "Mixed",
            "tone": "neutral",
            "note": "The current playbook does not have enough tracked history yet to favor a specific session window.",
        }

    best_follow = playbook_archive.sort_values(["FollowThroughPct", "Signals"], ascending=[False, False]).iloc[0]
    best_exec = playbook_archive.sort_values(
        ["ActualWinRatePct", "ClosedTradeCount", "Signals"], ascending=[False, False, False]
    ).iloc[0]
    follow_session = str(best_follow["Session"])
    exec_session = str(best_exec["Session"])

    if str(trade_gate or "").strip() == "No-Trade" or str(catalyst_window or "").startswith("Blocking"):
        title = "Stand Aside"
        tone = "negative"
    elif current_session in {follow_session, exec_session}:
        title = "Supportive"
        tone = "positive"
    elif current_session == "Unknown":
        title = "Mixed"
        tone = "neutral"
    else:
        title = "Cautious"
        tone = "warning"

    note = (
        f"Current session is <b>{current_session}</b>. For this playbook, best signal follow-through has clustered in "
        f"<b>{follow_session}</b> ({float(best_follow['FollowThroughPct']):.1f}% across {int(best_follow['Resolved'])} resolved signals), "
        f"while best real execution has clustered in <b>{exec_session}</b> "
        f"({float(best_exec['ActualWinRatePct']):.1f}% across {int(best_exec['ClosedTradeCount'])} closed trades)."
    )
    return {"title": title, "tone": tone, "note": note}


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
    get_signal_tracker_db_path = get_ctx(ctx, "get_signal_tracker_db_path")
    init_signal_tracker_db = get_ctx(ctx, "init_signal_tracker_db")
    fetch_signal_events_df = get_ctx(ctx, "fetch_signal_events_df")
    build_signal_cohort_summary = get_ctx(ctx, "build_signal_cohort_summary")
    build_recent_market_context_snapshot = get_ctx(ctx, "build_recent_market_context_snapshot")
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
    c1, c2 = st.columns(2)
    with c1:
        coin_s = _normalize_coin_input(
            st.text_input("Coin (e.g. BTC, ETH, TAO)", value="BTC", key="session_coin_input")
        )
    with c2:
        lookback = st.selectbox("Lookback Candles (1h)", [240, 360, 500], index=2, key="session_lookback")

    current_sig = (coin_s, int(lookback), "1h")
    state_key = "sessions_analysis_result"

    result = st.session_state.get(state_key)
    if not result or result.get("sig") != current_sig:
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
    if not result:
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
        profile_title = "Best Relative Timing Window"
        profile_color = POSITIVE
        profile_note = "This session currently offers the cleanest relative mix of liquidity, movement and usable range."
    elif leader_label == "Balanced":
        profile_title = "Mixed Timing Window"
        profile_color = WARNING
        profile_note = "Conditions are usable, but the edge is only moderate relative to the other sessions."
    else:
        profile_title = "Weak Timing Window"
        profile_color = NEGATIVE
        profile_note = "All sessions look soft; the best one is only less weak than the others."

    render_kpi_grid(
        st,
        items=[
            {
                "label": "Best Relative Session",
                "value": best_score,
                "subtext": (
                    f"Quality {grouped.loc[best_score, 'relative_quality']:.1f} · "
                    f"{grouped.loc[best_score, 'relative_label']}"
                ),
            },
            {
                "label": "Cleanest Liquidity",
                "value": best_liquidity,
                "subtext": (
                    f"Avg Vol {_format_volume_compact(grouped.loc[best_liquidity, 'avg_volume'])} · "
                    f"{grouped.loc[best_liquidity, 'liquidity_label']}"
                ),
            },
            {
                "label": "Most Active Tape",
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
    tracker_db_path = init_signal_tracker_db(get_signal_tracker_db_path())
    tracked_events = fetch_signal_events_df(limit=2000, source="Market", db_path=tracker_db_path)
    tracked_session_summary = pd.DataFrame()
    current_playbook = "Unknown"
    current_trade_gate = "Unknown"
    current_catalyst_window = "Unknown"
    current_session = session_bucket_for_timestamp()
    playbook_session_archive = pd.DataFrame()
    timing_read: dict[str, str] | None = None

    if tracked_events is not None and not tracked_events.empty:
        tracked_events = _prepare_tracked_session_archive(tracked_events)
        tracked_session_summary = build_signal_cohort_summary(tracked_events, "Session")
        recent_market_context = build_recent_market_context_snapshot(tracked_events)
        current_playbook = str(recent_market_context.get("Playbook") or "Unknown")
        current_trade_gate = str(recent_market_context.get("Trade Gate") or "Unknown")
        current_catalyst_window = str(recent_market_context.get("Catalyst Window") or "Unknown")
        playbook_session_archive = _build_playbook_session_archive(tracked_events, current_playbook)
        if not playbook_session_archive.empty:
            timing_read = _playbook_timing_read(
                playbook_session_archive,
                current_session=current_session,
                trade_gate=current_trade_gate,
                catalyst_window=current_catalyst_window,
            )
            render_insight_card(
                st,
                title=f"Current Timing Read · {timing_read['title']}",
                body_html=(
                    f"<span style='font-weight:700;'>Current session:</span> {current_session}. "
                    f"<span style='font-weight:700;'>Playbook:</span> {current_playbook}. "
                    f"<span style='font-weight:700;'>Catalyst window:</span> {current_catalyst_window}. "
                    f"{timing_read['note']}"
                ),
                tone=timing_read["tone"],
            )

    render_help_details(
        st,
        summary="How to read quickly (?)",
        body_html=(
            "<b>1.</b> Start with <b>Best Relative Session</b>: this is the best session among Asia / Europe / US <b>for this sample only</b>.<br>"
            "<b>2.</b> Use <b>Cleanest Liquidity</b> when your setup already exists and you want cleaner fills.<br>"
            "<b>3.</b> Use <b>Most Active Tape</b> when you want movement, but remember fast tape can also mean harder execution.<br>"
            "<b>4.</b> <b>Up Tilt / Down Tilt</b> is timing context, not a buy/sell signal by itself.<br>"
            "<b>5.</b> Use this tab to choose <b>when</b> to execute; use Market / Spot / Position to decide <b>what</b> to trade."
        ),
    )

    with st.expander("Archive & Diagnostics", expanded=False):
        st.markdown(
            f"<div style='color:{TEXT_MUTED}; font-size:0.88rem; margin-bottom:10px;'>"
            f"Use this section only when you want the deeper archive view or raw diagnostics."
            f"</div>",
            unsafe_allow_html=True,
        )
        st.markdown(
            f"<div style='color:{TEXT_MUTED}; font-size:0.86rem; line-height:1.65; margin-bottom:12px;'>"
            f"<b>Metric Guide:</b> Relative Quality is a relative timing score across the 3 sessions only. "
            f"It blends liquidity 50%, usable range 30%, and movement quality 20%. "
            f"Up Tilt / Down Tilt is context, not a standalone buy or sell signal."
            f"</div>",
            unsafe_allow_html=True,
        )
        if not tracked_session_summary.empty:
            best_archive_follow = tracked_session_summary.sort_values(
                ["FollowThroughPct", "Signals"], ascending=[False, False]
            ).iloc[0]
            best_archive_execution = tracked_session_summary.sort_values(
                ["ActualWinRatePct", "ClosedTradeCount", "Signals"], ascending=[False, False, False]
            ).iloc[0]
            render_insight_card(
                st,
                title="Tracked Session Archive",
                body_html=(
                    f"Best system follow-through has clustered in <b>{best_archive_follow['Session']}</b> "
                    f"({float(best_archive_follow['FollowThroughPct']):.1f}% across {int(best_archive_follow['Resolved'])} resolved signals). "
                    f"Best real execution is currently <b>{best_archive_execution['Session']}</b> "
                    f"({float(best_archive_execution['ActualWinRatePct']):.1f}% across {int(best_archive_execution['ClosedTradeCount'])} closed trades)."
                ),
                tone="positive" if float(best_archive_execution["ActualWinRatePct"]) >= 55.0 else "neutral",
            )
            tracked_session_view = tracked_session_summary[
                [
                    "Session",
                    "Signals",
                    "Resolved",
                    "FollowThroughPct",
                    "TakenPct",
                    "ClosedTradeCount",
                    "ActualWinRatePct",
                    "AvgActualPnlPct",
                ]
            ].copy()
            st.markdown(f"<b style='color:{ACCENT};'>Tracked Session Archive</b>", unsafe_allow_html=True)
            st.dataframe(tracked_session_view.round(2), hide_index=True, width="stretch")

        if timing_read is not None and not playbook_session_archive.empty:
            best_playbook_follow = playbook_session_archive.sort_values(
                ["FollowThroughPct", "Signals"], ascending=[False, False]
            ).iloc[0]
            best_playbook_execution = playbook_session_archive.sort_values(
                ["ActualWinRatePct", "ClosedTradeCount", "Signals"], ascending=[False, False, False]
            ).iloc[0]
            render_kpi_grid(
                st,
                items=[
                    {"label": "Current Session", "value": current_session, "subtext": "UTC bucket right now"},
                    {"label": "Current Playbook", "value": current_playbook, "subtext": f"Gate {current_trade_gate}"},
                    {
                        "label": "Best Follow-Through",
                        "value": str(best_playbook_follow["Session"]),
                        "subtext": f"{float(best_playbook_follow['FollowThroughPct']):.1f}% on matched signals",
                    },
                    {
                        "label": "Best Execution",
                        "value": str(best_playbook_execution["Session"]),
                        "subtext": f"{float(best_playbook_execution['ActualWinRatePct']):.1f}% realized win rate",
                    },
                ],
            )
            playbook_view = playbook_session_archive[
                ["Session", "Signals", "Resolved", "FollowThroughPct", "ClosedTradeCount", "ActualWinRatePct", "AvgActualPnlPct"]
            ].copy()
            st.markdown(
                f"<b style='color:{ACCENT};'>Playbook Timing Archive · {current_playbook}</b>",
                unsafe_allow_html=True,
            )
            st.dataframe(playbook_view.round(2), hide_index=True, width="stretch")

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
