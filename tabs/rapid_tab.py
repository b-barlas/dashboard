from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd

from ui.ctx import get_ctx


def render(ctx: dict) -> None:
    st = get_ctx(ctx, "st")
    ACCENT = get_ctx(ctx, "ACCENT")
    TEXT_MUTED = get_ctx(ctx, "TEXT_MUTED")
    CARD_BG = get_ctx(ctx, "CARD_BG")
    POSITIVE = get_ctx(ctx, "POSITIVE")
    NEGATIVE = get_ctx(ctx, "NEGATIVE")
    WARNING = get_ctx(ctx, "WARNING")
    _tip = get_ctx(ctx, "_tip")
    get_top_volume_usdt_symbols = get_ctx(ctx, "get_top_volume_usdt_symbols")
    fetch_ohlcv = get_ctx(ctx, "fetch_ohlcv")
    analyse = get_ctx(ctx, "analyse")
    get_scalping_entry_target = get_ctx(ctx, "get_scalping_entry_target")
    ml_ensemble_predict = get_ctx(ctx, "ml_ensemble_predict")
    _calc_conviction = get_ctx(ctx, "_calc_conviction")
    signal_plain = get_ctx(ctx, "signal_plain")
    readable_market_cap = get_ctx(ctx, "readable_market_cap")
    _debug = get_ctx(ctx, "_debug")

    def _fmt_price(v: float) -> str:
        p = float(v)
        if p >= 1000:
            return f"${p:,.2f}"
        if p >= 1:
            return f"${p:,.4f}"
        if p >= 0.01:
            return f"${p:,.6f}"
        if p >= 0.0001:
            return f"${p:,.8f}"
        return f"${p:,.10f}"

    def _setup_badge(scalp_dir: str, signal_dir: str, ai_dir: str) -> str:
        if scalp_dir and signal_dir in {"LONG", "SHORT"} and signal_dir == ai_dir == scalp_dir:
            return "Aligned"
        if scalp_dir and signal_dir in {"LONG", "SHORT"} and signal_dir == scalp_dir and ai_dir == "NEUTRAL":
            return "Tech-Only"
        if scalp_dir:
            return "Draft"
        return "No Setup"

    def _grade(score: float) -> str:
        if score >= 82:
            return "A+"
        if score >= 74:
            return "A"
        if score >= 66:
            return "B"
        return "C"

    def _score_row(
        signal_dir: str,
        confidence: float,
        setup: str,
        conviction_label: str,
        ai_dir: str,
        agreement: float,
        adx: float,
        rr: float,
        has_plan: bool,
    ) -> float:
        if signal_dir not in {"LONG", "SHORT"}:
            return 0.0
        if not has_plan:
            return 5.0

        confidence_score = max(0.0, min(100.0, float(confidence)))
        setup_score = {"Aligned": 100.0, "Tech-Only": 72.0, "Draft": 45.0}.get(setup, 0.0)

        ai_align = 100.0 if ai_dir == signal_dir else (60.0 if ai_dir == "NEUTRAL" else 15.0)
        ai_score = 0.55 * ai_align + 0.45 * max(0.0, min(100.0, agreement * 100.0))

        if pd.isna(adx):
            trend_score = 55.0
        else:
            adx_f = float(adx)
            if adx_f < 18:
                trend_score = 35.0
            elif adx_f < 25:
                trend_score = 60.0
            elif adx_f < 40:
                trend_score = 82.0
            else:
                trend_score = 92.0

        rr_score = 0.0
        if rr >= 2.0:
            rr_score = 100.0
        elif rr >= 1.5:
            rr_score = 82.0
        elif rr >= 1.2:
            rr_score = 66.0
        elif rr > 0:
            rr_score = 50.0

        conv_penalty = 0.0
        if conviction_label == "CONFLICT":
            conv_penalty = -18.0
        elif conviction_label == "LOW":
            conv_penalty = -8.0

        score = (
            0.30 * confidence_score
            + 0.20 * setup_score
            + 0.20 * ai_score
            + 0.15 * trend_score
            + 0.15 * rr_score
            + conv_penalty
        )
        return max(0.0, min(100.0, score))

    def _action_label(
        signal_dir: str,
        confidence: float,
        setup: str,
        conviction_label: str,
        ai_dir: str,
        score: float,
        has_plan: bool,
    ) -> str:
        if signal_dir not in {"LONG", "SHORT"}:
            return "SKIP"
        if not has_plan:
            return "WAIT"
        if setup == "No Setup" or conviction_label == "CONFLICT":
            return "SKIP"
        if ai_dir not in {signal_dir, "NEUTRAL"}:
            return "WAIT"
        if confidence >= 62 and score >= 76:
            return "READY"
        if score >= 64:
            return "WAIT"
        return "SKIP"

    def _action_badge(v: str) -> str:
        if v == "READY":
            return "✅ READY"
        if v == "WAIT":
            return "⏳ WAIT"
        return "⛔ SKIP"

    def _dir_badge(v: str) -> str:
        if v == "LONG":
            return "🟢 LONG"
        if v == "SHORT":
            return "🔴 SHORT"
        return "⚪ NEUTRAL"

    def _setup_icon(v: str) -> str:
        if v == "Aligned":
            return "🟢 Aligned"
        if v == "Tech-Only":
            return "🟡 Tech-Only"
        if v == "Draft":
            return "⚪ Draft"
        return "🔴 No Setup"

    def _conv_icon(v: str) -> str:
        if v == "HIGH":
            return "🟢 HIGH"
        if v == "MEDIUM":
            return "🟡 MEDIUM"
        if v == "LOW":
            return "⚪ LOW"
        if v == "CONFLICT":
            return "🔴 CONFLICT"
        return v

    def _adx_icon(v) -> str:
        if v is None or pd.isna(v):
            return "N/A"
        x = float(v)
        if x >= 25:
            return f"🟢 {x:.1f}"
        if x >= 18:
            return f"🟡 {x:.1f}"
        return f"🔴 {x:.1f}"

    def _grade_icon(g: str) -> str:
        if g == "A+":
            return "🏆 A+"
        if g == "A":
            return "✅ A"
        if g == "B":
            return "🟡 B"
        return "⚪ C"

    st.markdown(
        f"<h2 style='color:{ACCENT};margin-bottom:0.35rem;'>Rapid ⚡</h2>",
        unsafe_allow_html=True,
    )
    st.markdown(
        f"<div class='panel-box' style='border:1px solid rgba(255,59,92,0.25);'>"
        f"<b style='color:{ACCENT};'>What does this tab show?</b>"
        f"<p style='color:{TEXT_MUTED}; font-size:0.9rem; margin-top:6px; line-height:1.6;'>"
        f"Fast decision feed for short-horizon setups. Rapid scans the live universe, ranks candidates, "
        f"and shows ready-to-act plans with {_tip('Entry / SL / TP', 'Entry zone, invalidation stop-loss level, and first take-profit level.')}."
        f"</p></div>",
        unsafe_allow_html=True,
    )
    st.markdown(
        f"<details style='margin:0.35rem 0 0.7rem 0;'>"
        f"<summary style='color:{ACCENT}; cursor:pointer;'>How Rapid Score is calculated (?)</summary>"
        f"<div style='color:{TEXT_MUTED}; font-size:0.84rem; line-height:1.7; margin-top:0.45rem;'>"
        f"<b>Score (0-100)</b> is a weighted setup-quality score:<br>"
        f"• Confidence: <b>30%</b><br>"
        f"• Setup quality (Aligned/Tech-Only/Draft): <b>20%</b><br>"
        f"• AI quality (direction fit + agreement): <b>20%</b><br>"
        f"• Trend quality (ADX): <b>15%</b><br>"
        f"• Execution quality (R:R): <b>15%</b><br>"
        f"• Conviction conflict/low gets penalty.<br>"
        f"Higher score means cleaner and more actionable setup quality."
        f"</div></details>",
        unsafe_allow_html=True,
    )

    c1, c2, c3, c4 = st.columns([1.2, 1.1, 1, 1], gap="medium")
    with c1:
        timeframe = st.selectbox("Timeframe", ["5m", "15m", "1h", "4h"], index=1, key="rapid_tf")
    with c2:
        universe_size = st.slider("Universe", min_value=8, max_value=40, value=20, step=2, key="rapid_universe")
    with c3:
        min_score = st.slider("Min Score", min_value=55, max_value=90, value=68, step=1, key="rapid_min_score")
    with c4:
        refresh = st.button("Refresh Rapid", use_container_width=True)

    scan_sig = (timeframe, int(universe_size), int(min_score))
    prev_sig = st.session_state.get("rapid_sig")
    should_scan = refresh or ("rapid_rows" not in st.session_state) or (scan_sig != prev_sig)

    rows: list[dict] = st.session_state.get("rapid_rows", [])
    if should_scan:
        with st.spinner(f"Running rapid scan on {universe_size} symbols ({timeframe})..."):
            usdt_symbols, market_data = get_top_volume_usdt_symbols(max(universe_size, 50))

            seen_symbols = set()
            mcap_map: dict[str, int] = {}
            for coin in market_data:
                coin_id = str(coin.get("id") or "").lower()
                symbol = str(coin.get("symbol") or "").upper()
                if not symbol or "wrapped" in coin_id or symbol in seen_symbols:
                    continue
                seen_symbols.add(symbol)
                mcap_map[symbol] = int(coin.get("market_cap") or 0)

            valid_bases = set(mcap_map.keys())
            working_symbols = [s for s in usdt_symbols if s.split("/")[0].upper() in valid_bases][:universe_size]

            def _scan_one(sym: str) -> dict | None:
                df = fetch_ohlcv(sym, timeframe, limit=500)
                if df is None or len(df) <= 80:
                    return None

                df_eval = df.iloc[:-1].copy()
                if len(df_eval) <= 55:
                    return None

                a = analyse(df_eval)
                signal = a.signal
                confidence = float(a.confidence)
                adx = float(a.adx) if pd.notna(a.adx) else float("nan")
                signal_dir = signal_plain(signal)
                if signal_dir not in {"LONG", "SHORT"}:
                    return None

                prob, ai_dir, ai_details = ml_ensemble_predict(df_eval)
                agreement = float((ai_details or {}).get("agreement", 0.0))
                _conv_lbl, _conv_color = _calc_conviction(signal_dir, ai_dir, confidence)

                scalp_dir, entry, target, stop, rr_ratio, note = get_scalping_entry_target(
                    df_eval,
                    confidence,
                    a.supertrend,
                    a.ichimoku,
                    a.vwap,
                    a.volume_spike,
                    strict_mode=True,
                )
                setup = _setup_badge(scalp_dir or "", signal_dir, ai_dir)
                has_plan = bool(entry and target and stop and rr_ratio)
                rr = float(rr_ratio or 0.0)

                score = _score_row(
                    signal_dir=signal_dir,
                    confidence=confidence,
                    setup=setup,
                    conviction_label=str(_conv_lbl),
                    ai_dir=ai_dir,
                    agreement=agreement,
                    adx=adx,
                    rr=rr,
                    has_plan=has_plan,
                )
                action = _action_label(
                    signal_dir=signal_dir,
                    confidence=confidence,
                    setup=setup,
                    conviction_label=str(_conv_lbl),
                    ai_dir=ai_dir,
                    score=score,
                    has_plan=has_plan,
                )
                base = sym.split("/")[0].upper()
                last_price = float(df["close"].iloc[-1])
                why_now = []
                if setup == "Aligned":
                    why_now.append("Setup is fully aligned")
                if agreement >= 0.65:
                    why_now.append(f"AI agreement is strong ({agreement * 100:.0f}%)")
                if pd.notna(adx) and adx >= 25:
                    why_now.append(f"Trend strength is healthy (ADX {adx:.1f})")
                if rr >= 1.5:
                    why_now.append(f"Risk/reward is acceptable ({rr:.2f})")
                if not why_now:
                    why_now.append("Partial alignment only; requires tighter confirmation")

                return {
                    "Symbol": sym,
                    "Coin": base,
                    "Price": _fmt_price(last_price),
                    "Action": action,
                    "Direction": signal_dir,
                    "Score": round(score, 1),
                    "Grade": _grade(score),
                    "Confidence": round(confidence, 1),
                    "Setup": setup,
                    "Conviction": str(_conv_lbl),
                    "AI Direction": ai_dir,
                    "AI Agree": f"{agreement * 100:.0f}%",
                    "ADX": (round(adx, 1) if pd.notna(adx) else None),
                    "Entry": _fmt_price(float(entry)) if entry else "N/A",
                    "SL": _fmt_price(float(stop)) if stop else "N/A",
                    "TP1": _fmt_price(float(target)) if target else "N/A",
                    "R:R": (f"{rr:.2f}" if rr > 0 else "N/A"),
                    "Market Cap": readable_market_cap(int(mcap_map.get(base, 0))),
                    "Why": why_now[:3],
                    "Note": note or "",
                }

            fresh: list[dict] = []
            with ThreadPoolExecutor(max_workers=6) as executor:
                futures = {executor.submit(_scan_one, sym): sym for sym in working_symbols}
                for future in as_completed(futures):
                    try:
                        row = future.result()
                        if row:
                            fresh.append(row)
                    except Exception as exc:
                        _debug(f"Rapid scan error {futures[future]}: {exc}")

            fresh = sorted(fresh, key=lambda x: (x["Action"] == "READY", x["Score"]), reverse=True)
            qualified = [r for r in fresh if float(r["Score"]) >= float(min_score)]
            watchlist = [r for r in fresh if float(r["Score"]) < float(min_score)][:3]
            st.session_state["rapid_watchlist"] = watchlist
            if qualified:
                st.session_state["rapid_rows"] = qualified
                st.session_state["rapid_sig"] = scan_sig
                st.session_state["rapid_cache_ts"] = pd.Timestamp.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")
                rows = qualified
            else:
                old_rows = st.session_state.get("rapid_rows", [])
                rows = old_rows
                if old_rows:
                    ts = st.session_state.get("rapid_cache_ts", "unknown time")
                    st.warning(f"Rapid live scan returned no candidates. Showing previous snapshot from {ts}.")

    if not rows:
        watchlist = st.session_state.get("rapid_watchlist", [])
        st.info("No rapid candidates currently match the quality threshold.")
        if watchlist:
            st.markdown(
                f"<div style='color:{TEXT_MUTED}; font-size:0.84rem; margin:0.25rem 0 0.6rem 0;'>"
                f"Showing <b>near-miss watchlist</b> (best below current Min Score).</div>",
                unsafe_allow_html=True,
            )
            wdf = pd.DataFrame(watchlist)[
                ["Coin", "Direction", "Score", "Grade", "Confidence", "Setup", "Conviction", "AI Direction", "AI Agree", "ADX", "Entry", "SL", "TP1", "R:R"]
            ].copy()
            wdf["Direction"] = wdf["Direction"].map(_dir_badge)
            wdf["Grade"] = wdf["Grade"].map(_grade_icon)
            wdf["Setup"] = wdf["Setup"].map(_setup_icon)
            wdf["Conviction"] = wdf["Conviction"].map(_conv_icon)
            wdf["AI Direction"] = wdf["AI Direction"].map(_dir_badge)
            wdf["ADX"] = wdf["ADX"].map(_adx_icon)
            st.dataframe(wdf, width="stretch", hide_index=True)
            st.markdown(
                f"<details style='margin-top:0.55rem;'>"
                f"<summary style='color:{ACCENT}; cursor:pointer;'>Column Guide (?)</summary>"
                f"<div style='color:{TEXT_MUTED}; font-size:0.83rem; line-height:1.7; margin-top:0.45rem;'>"
                f"<b>Direction</b>: trade side from technical signal.<br>"
                f"<b>Score</b>: overall setup quality (0-100).<br>"
                f"<b>Grade</b>: compressed score class (A+ best).<br>"
                f"<b>Setup</b>: plan alignment quality; Aligned is strongest.<br>"
                f"<b>Conviction</b>: technical + AI + confidence alignment quality.<br>"
                f"<b>AI Direction / AI Agree</b>: AI side and model agreement percentage.<br>"
                f"<b>ADX</b>: trend strength (green stronger trend, red weak trend).<br>"
                f"<b>Entry / SL / TP1</b>: draft execution prices. SL is invalidation level.<br>"
                f"<b>R:R</b>: reward-to-risk ratio. Higher is generally better."
                f"</div></details>",
                unsafe_allow_html=True,
            )
        return

    best = rows[0]
    action_color = POSITIVE if best["Action"] == "READY" else (WARNING if best["Action"] == "WAIT" else NEGATIVE)
    st.markdown(
        f"<div style='background:linear-gradient(135deg, rgba(0,0,0,0.92), rgba(18,28,43,0.95)); "
        f"border:1px solid {action_color}; border-radius:12px; padding:14px; margin:10px 0 12px 0;'>"
        f"<div style='display:flex; justify-content:space-between; align-items:center; gap:12px;'>"
        f"<div><div style='color:{TEXT_MUTED}; font-size:0.78rem; text-transform:uppercase;'>Best Now</div>"
        f"<div style='color:{ACCENT}; font-size:1.25rem; font-weight:700;'>{best['Coin']} • {best['Direction']}</div></div>"
        f"<div style='text-align:right;'>"
        f"<div style='color:{action_color}; font-size:1.05rem; font-weight:800;'>{best['Action']} ({best['Grade']})</div>"
        f"<div style='color:{TEXT_MUTED}; font-size:0.8rem;'>Score {best['Score']:.1f}</div></div></div>"
        f"<div style='color:{TEXT_MUTED}; font-size:0.86rem; line-height:1.65; margin-top:8px;'>"
        f"Entry <b>{best['Entry']}</b> | SL <b>{best['SL']}</b> | TP1 <b>{best['TP1']}</b> | "
        f"R:R <b>{best['R:R']}</b> | Confidence <b>{best['Confidence']:.0f}%</b></div>"
        f"</div>",
        unsafe_allow_html=True,
    )

    st.markdown(
        f"<details style='margin-bottom:0.7rem;'>"
        f"<summary style='color:{ACCENT}; cursor:pointer;'>How to act quickly (?)</summary>"
        f"<div style='color:{TEXT_MUTED}; font-size:0.84rem; line-height:1.7; margin-top:0.45rem;'>"
        f"1. Use <b>READY</b> rows first, then verify the plan in Position tab before execution.<br>"
        f"2. Respect <b>SL</b> as invalidation. If invalidated, the setup is considered broken.<br>"
        f"3. Prefer rows with <b>Setup=Aligned</b>, <b>Conviction=HIGH</b>, and <b>R:R >= 1.5</b>."
        f"</div></details>",
        unsafe_allow_html=True,
    )
    st.markdown(
        f"<details style='margin-bottom:0.7rem;'>"
        f"<summary style='color:{ACCENT}; cursor:pointer;'>Column Guide (?)</summary>"
        f"<div style='color:{TEXT_MUTED}; font-size:0.84rem; line-height:1.7; margin-top:0.45rem;'>"
        f"<b>Action</b>: READY / WAIT / SKIP quick decision.<br>"
        f"<b>Score</b>: weighted quality score (0-100).<br>"
        f"<b>Grade</b>: score class (A+ / A / B / C).<br>"
        f"<b>Setup</b>: structural alignment quality of plan and direction.<br>"
        f"<b>Conviction</b>: alignment quality of technical + AI + confidence.<br>"
        f"<b>AI Direction / AI Agree</b>: AI side and confidence through model agreement.<br>"
        f"<b>Entry / SL / TP1</b>: draft levels; SL is invalidation.<br>"
        f"<b>R:R</b>: reward-to-risk ratio; prefer >= 1.5."
        f"</div></details>",
        unsafe_allow_html=True,
    )

    top_rows = rows[:5]
    cols = st.columns(len(top_rows), gap="small")
    for idx, row in enumerate(top_rows):
        with cols[idx]:
            c = POSITIVE if row["Action"] == "READY" else (WARNING if row["Action"] == "WAIT" else NEGATIVE)
            why_html = "".join(f"<li>{w}</li>" for w in row["Why"])
            st.markdown(
                f"<div style='background:{CARD_BG}; border:1px solid {c}; border-radius:10px; padding:10px; min-height:270px;'>"
                f"<div style='display:flex; justify-content:space-between; align-items:center;'>"
                f"<b style='color:{ACCENT};'>{row['Coin']}</b>"
                f"<span style='color:{c}; font-weight:700; font-size:0.8rem;'>{_action_badge(row['Action'])}</span></div>"
                f"<div style='color:{TEXT_MUTED}; font-size:0.78rem; margin-top:4px;'>"
                f"{_dir_badge(row['Direction'])} | Score {row['Score']:.1f} ({_grade_icon(row['Grade'])})</div>"
                f"<div style='color:{TEXT_MUTED}; font-size:0.8rem; margin-top:8px; line-height:1.6;'>"
                f"Price <b>{row['Price']}</b><br>"
                f"Entry <b>{row['Entry']}</b><br>"
                f"SL <b>{row['SL']}</b><br>"
                f"TP1 <b>{row['TP1']}</b><br>"
                f"R:R <b>{row['R:R']}</b><br>"
                f"Setup <b>{_setup_icon(row['Setup'])}</b><br>"
                f"Conviction <b>{_conv_icon(row['Conviction'])}</b><br>"
                f"AI <b>{_dir_badge(row['AI Direction'])}</b> ({row['AI Agree']})<br>"
                f"ADX <b>{_adx_icon(row['ADX'])}</b><br>"
                f"Market Cap <b>{row['Market Cap']}</b></div>"
                f"<div style='color:{TEXT_MUTED}; font-size:0.76rem; margin-top:7px;'>Why now:</div>"
                f"<ul style='color:{TEXT_MUTED}; font-size:0.76rem; padding-left:16px; margin:4px 0 0 0; line-height:1.5;'>{why_html}</ul>"
                f"</div>",
                unsafe_allow_html=True,
            )

    table_df = pd.DataFrame(rows).copy()
    table_df = table_df[
        ["Coin", "Action", "Direction", "Score", "Grade", "Confidence", "Setup", "Conviction", "AI Direction", "AI Agree", "ADX", "Entry", "SL", "TP1", "R:R"]
    ]
    table_df["Action"] = table_df["Action"].map(_action_badge)
    table_df["Direction"] = table_df["Direction"].map(_dir_badge)
    table_df["Grade"] = table_df["Grade"].map(_grade_icon)
    table_df["Setup"] = table_df["Setup"].map(_setup_icon)
    table_df["Conviction"] = table_df["Conviction"].map(_conv_icon)
    table_df["AI Direction"] = table_df["AI Direction"].map(_dir_badge)
    table_df["ADX"] = table_df["ADX"].map(_adx_icon)
    st.markdown(f"<h4 style='color:{ACCENT}; margin-top:0.8rem;'>Rapid Table</h4>", unsafe_allow_html=True)
    st.dataframe(table_df, width="stretch", hide_index=True)
