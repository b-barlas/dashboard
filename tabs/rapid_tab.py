from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd

from core.rapid_config import DEFAULT_RAPID_CONFIG
from core.signal_contract import strength_from_bias
from core.rapid_engine import (
    compute_rapid_score,
    decide_action,
    grade_from_score,
    setup_badge,
    summarize_quality_history,
)
from ui.ctx import get_ctx


def render(ctx: dict) -> None:
    st = get_ctx(ctx, "st")
    ACCENT = get_ctx(ctx, "ACCENT")
    TEXT_LIGHT = str(ctx.get("TEXT_LIGHT", ACCENT))
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
    direction_label = get_ctx(ctx, "direction_label")
    readable_market_cap = get_ctx(ctx, "readable_market_cap")
    _debug = get_ctx(ctx, "_debug")
    cfg = DEFAULT_RAPID_CONFIG
    major_fallback_symbols = [
        "BTC/USDT",
        "ETH/USDT",
        "SOL/USDT",
        "XRP/USDT",
        "BNB/USDT",
        "ADA/USDT",
        "DOGE/USDT",
        "AVAX/USDT",
        "LINK/USDT",
        "TON/USDT",
    ]

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

    def _action_badge(v: str) -> str:
        if v == "READY":
            return "ENTER"
        if v == "WAIT":
            return "WATCH"
        return "SKIP"

    def _dir_badge(v: str) -> str:
        return direction_label(v)

    def _setup_icon(v: str) -> str:
        if v == "Aligned":
            return "Aligned"
        if v == "Tech-Only":
            return "Tech-Only"
        if v == "Draft":
            return "Draft"
        return "No Setup"

    def _conv_icon(v: str) -> str:
        if v == "HIGH":
            return "High"
        if v == "MEDIUM":
            return "Medium"
        if v == "LOW":
            return "Low"
        if v == "CONFLICT":
            return "Conflict"
        return v

    def _adx_icon(v) -> str:
        if v is None or pd.isna(v):
            return "N/A"
        return f"{float(v):.1f}"

    def _grade_icon(g: str) -> str:
        return str(g)

    def _chip_class(value: str) -> str:
        if value in {"READY", "LONG", "Aligned", "High"}:
            return "elite-chip-positive"
        if value in {"WAIT", "Tech-Only", "Medium"}:
            return "elite-chip-warning"
        return "elite-chip-negative"

    def _render_candidate_cards(source_rows: list[dict], *, title: str) -> None:
        if not source_rows:
            return
        st.markdown(f"<h4 style='color:{ACCENT}; margin:0.3rem 0 0.45rem 0;'>{title}</h4>", unsafe_allow_html=True)
        for i in range(0, len(source_rows), 3):
            chunk = source_rows[i:i + 3]
            cols = st.columns(3, gap="small")
            for j, row in enumerate(chunk):
                action = str(row.get("Action", "SKIP"))
                direction = str(row.get("Direction", "NEUTRAL"))
                setup = str(row.get("Setup", "No Setup"))
                align = str(row.get("Alignment", "Low"))
                score = float(row.get("Score", 0.0))
                strength = float(row.get("Strength", 0.0))
                ai_agree = str(row.get("AI Agree", "0/3"))
                adx_val = row.get("ADX")
                price = str(row.get("Price", "N/A"))
                entry = str(row.get("Entry", "N/A"))
                stop = str(row.get("SL", "N/A"))
                target = str(row.get("TP1", "N/A"))
                rr = str(row.get("R:R", "N/A"))
                coin = str(row.get("Coin", "N/A"))
                why = row.get("Why", []) or []
                why_line = " | ".join([str(x) for x in why[:2]]) if why else "Await stronger alignment confirmation."

                action_chip = _chip_class(action)
                direction_chip = "elite-chip-positive" if direction == "LONG" else ("elite-chip-negative" if direction == "SHORT" else "elite-chip-warning")
                setup_chip = _chip_class(setup)
                align_chip = _chip_class(align.title())
                score_color = POSITIVE if score >= 75 else (WARNING if score >= 62 else NEGATIVE)
                score_fill = max(0, min(100, int(round(score))))
                try:
                    rr_num = float(rr)
                except Exception:
                    rr_num = None
                rr_color = POSITIVE if rr_num is not None and rr_num >= 1.5 else (WARNING if rr_num is not None else TEXT_MUTED)
                sl_val = stop
                tp_val = target

                with cols[j]:
                    st.markdown(
                        f"<div class='rapid-watch-card'>"
                        f"<div class='rapid-watch-head'>"
                        f"<div class='rapid-watch-symbol'>{coin}</div>"
                        f"<span class='elite-chip {action_chip}'>{_action_badge(action)}</span>"
                        f"</div>"
                        f"<div class='rapid-watch-chip-row'>"
                        f"<span class='elite-chip {direction_chip}'>{_dir_badge(direction)}</span>"
                        f"<span class='elite-chip {setup_chip}'>{setup}</span>"
                        f"<span class='elite-chip {align_chip}'>{align}</span>"
                        f"</div>"
                        f"<div style='display:flex; justify-content:space-between; color:{TEXT_MUTED}; font-size:0.78rem;'>"
                        f"<span>Score {score:.0f}</span><span>Strength {strength:.0f}%</span>"
                        f"</div>"
                        f"<div class='rapid-watch-bar'><span style='width:{score_fill}%; background:{score_color};'></span></div>"
                        f"<div class='rapid-watch-grid'>"
                        f"<div><div class='rapid-watch-k'>Price</div><div class='rapid-watch-v'>{price}</div></div>"
                        f"<div><div class='rapid-watch-k'>Entry</div><div class='rapid-watch-v'>{entry}</div></div>"
                        f"<div><div class='rapid-watch-k'>Stop Loss</div><div class='rapid-watch-v' style='color:{NEGATIVE};'>{sl_val}</div></div>"
                        f"<div><div class='rapid-watch-k'>Target</div><div class='rapid-watch-v' style='color:{POSITIVE};'>{tp_val}</div></div>"
                        f"<div><div class='rapid-watch-k'>R:R</div><div class='rapid-watch-v' style='color:{rr_color};'>{rr}</div></div>"
                        f"<div><div class='rapid-watch-k'>AI / ADX</div><div class='rapid-watch-v'>{ai_agree} / {_adx_icon(adx_val)}</div></div>"
                        f"</div>"
                        f"<div class='rapid-watch-note'>{why_line}</div>"
                        f"</div>",
                        unsafe_allow_html=True,
                    )

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
        f"<summary style='color:{ACCENT}; cursor:pointer;'>How Rapid Score Works</summary>"
        f"<div style='color:{TEXT_MUTED}; font-size:0.84rem; line-height:1.7; margin-top:0.45rem;'>"
        f"<b>Score (0-100)</b> is a weighted setup-quality score:<br>"
        f"• Strength: <b>30%</b><br>"
        f"• Setup quality (Aligned/Tech-Only/Draft): <b>20%</b><br>"
        f"• AI quality (direction fit + agreement): <b>20%</b><br>"
        f"• Trend quality (ADX): <b>15%</b><br>"
        f"• Execution quality (R:R): <b>15%</b><br>"
        f"• Alignment conflict/low gets penalty.<br>"
        f"Higher score means cleaner and more actionable setup quality."
        f"</div></details>",
        unsafe_allow_html=True,
    )

    c1, c2, c3, c4 = st.columns([1.2, 1.1, 1, 1], gap="medium")
    with c1:
        timeframe = st.selectbox("Timeframe", ["5m", "15m", "1h", "4h"], index=1, key="rapid_tf")
    with c2:
        universe_size = st.slider(
            "Universe",
            min_value=8,
            max_value=40,
            value=int(cfg.default_universe),
            step=2,
            key="rapid_universe",
        )
    with c3:
        min_score = st.slider(
            "Min Score",
            min_value=55,
            max_value=90,
            value=int(cfg.default_min_score),
            step=1,
            key="rapid_min_score",
        )
    with c4:
        refresh = st.button("Refresh Rapid", use_container_width=True)

    scan_sig = (timeframe, int(universe_size), int(min_score))
    prev_sig = st.session_state.get("rapid_sig")
    should_scan = refresh or ("rapid_rows" not in st.session_state) or (scan_sig != prev_sig)

    rows: list[dict] = st.session_state.get("rapid_rows", [])
    if should_scan:
        with st.spinner(f"Running rapid scan on {universe_size} symbols ({timeframe})..."):
            fresh: list[dict] = []
            mcap_map: dict[str, int] = {}
            # Retry once in the same render to reduce first-load empty-state caused by transient API hiccups.
            for attempt in range(2):
                usdt_symbols, market_data = get_top_volume_usdt_symbols(max(universe_size, 50))

                seen_symbols = set()
                mcap_map = {}
                for coin in market_data:
                    coin_id = str(coin.get("id") or "").lower()
                    symbol = str(coin.get("symbol") or "").upper()
                    if not symbol or "wrapped" in coin_id or symbol in seen_symbols:
                        continue
                    seen_symbols.add(symbol)
                    mcap_map[symbol] = int(coin.get("market_cap") or 0)

                valid_bases = set(mcap_map.keys())
                working_symbols = [s for s in usdt_symbols if s.split("/")[0].upper() in valid_bases][:universe_size]
                if not working_symbols:
                    _debug(f"Rapid scan attempt {attempt+1}: no working symbols for {timeframe}.")
                    # Fallback to a stable major list when market feed returns empty.
                    working_symbols = major_fallback_symbols[: min(universe_size, len(major_fallback_symbols))]
                    if not working_symbols:
                        continue

                def _scan_one(sym: str) -> dict | None:
                    df = fetch_ohlcv(sym, timeframe, limit=500)
                    if df is None or len(df) <= 80:
                        return None

                    df_eval = df.iloc[:-1].copy()
                    if len(df_eval) <= 55:
                        return None

                    a = analyse(df_eval)
                    signal = a.signal
                    bias_raw = float(a.confidence)
                    strength = float(strength_from_bias(bias_raw))
                    adx = float(a.adx) if pd.notna(a.adx) else float("nan")
                    signal_dir = signal_plain(signal)

                    prob, ai_dir, ai_details = ml_ensemble_predict(df_eval)
                    agreement = float((ai_details or {}).get("agreement", 0.0))
                    trade_dir = signal_dir if signal_dir in {"LONG", "SHORT"} else (ai_dir if ai_dir in {"LONG", "SHORT"} else "WAIT")
                    _conv_lbl, _conv_color = _calc_conviction(signal_dir, ai_dir, strength)

                    scalp_dir, entry, target, stop, rr_ratio, note = get_scalping_entry_target(
                        df_eval,
                        bias_raw,
                        a.supertrend,
                        a.ichimoku,
                        a.vwap,
                        a.volume_spike,
                        strict_mode=True,
                    )
                    setup = setup_badge(scalp_dir or "", signal_dir, ai_dir)
                    has_plan = bool(entry and target and stop and rr_ratio)
                    rr = float(rr_ratio or 0.0)

                    if trade_dir in {"LONG", "SHORT"}:
                        score = compute_rapid_score(
                            signal_dir=trade_dir,
                            strength=strength,
                            setup=setup,
                            conviction_label=str(_conv_lbl),
                            ai_dir=ai_dir,
                            agreement=agreement,
                            adx=adx,
                            rr=rr,
                            has_plan=has_plan,
                            cfg=cfg,
                        )
                        action = decide_action(
                            signal_dir=trade_dir,
                            strength=strength,
                            setup=setup,
                            conviction_label=str(_conv_lbl),
                            ai_dir=ai_dir,
                            score=score,
                            has_plan=has_plan,
                            cfg=cfg,
                        )
                    else:
                        # Relaxed scoring path: keep near-miss candidates visible instead of blank screen.
                        setup_boost = 18.0 if setup == "Aligned" else (10.0 if setup in {"Tech-Only", "Draft"} else 0.0)
                        trend_boost = 10.0 if (pd.notna(adx) and float(adx) >= cfg.trend_adx_weak) else 0.0
                        neutral_score = (
                            0.55 * float(strength)
                            + 0.35 * float(agreement * 100.0)
                            + setup_boost
                            + trend_boost
                        )
                        score = max(0.0, min(59.0, neutral_score))
                        action = "SKIP"
                    base = sym.split("/")[0].upper()
                    last_price = float(df["close"].iloc[-1])
                    why_now = []
                    if setup == "Aligned":
                        why_now.append("Setup is fully aligned")
                    votes = max(0, min(3, int(round(float(agreement) * 3.0))))
                    if agreement >= 0.65:
                        why_now.append(f"AI agreement is strong ({votes}/3)")
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
                        "Direction": trade_dir,
                        "Score": round(score, 1),
                        "Grade": grade_from_score(score),
                        "Strength": round(strength, 1),
                        "Setup": setup,
                        "Alignment": str(_conv_lbl),
                        "AI Direction": ai_dir,
                        "AI Agree": f"{votes}/3",
                        "ADX": (round(adx, 1) if pd.notna(adx) else None),
                        "Entry": _fmt_price(float(entry)) if entry else "N/A",
                        "SL": _fmt_price(float(stop)) if stop else "N/A",
                        "TP1": _fmt_price(float(target)) if target else "N/A",
                        "Entry Raw": (float(entry) if entry else None),
                        "SL Raw": (float(stop) if stop else None),
                        "TP1 Raw": (float(target) if target else None),
                        "R:R": (f"{rr:.2f}" if rr > 0 else "N/A"),
                        "Market Cap": readable_market_cap(int(mcap_map.get(base, 0))),
                        "Why": why_now[:3],
                        "Note": note or "",
                    }

                attempt_rows: list[dict] = []
                with ThreadPoolExecutor(max_workers=6) as executor:
                    futures = {executor.submit(_scan_one, sym): sym for sym in working_symbols}
                    for future in as_completed(futures):
                        try:
                            row = future.result()
                            if row:
                                attempt_rows.append(row)
                        except Exception as exc:
                            _debug(f"Rapid scan error {futures[future]}: {exc}")
                if attempt_rows:
                    fresh = attempt_rows
                    break
                _debug(f"Rapid scan attempt {attempt+1}: no rows produced for {timeframe}.")

            fresh = sorted(fresh, key=lambda x: (x["Action"] == "READY", x["Action"] == "WAIT", x["Score"]), reverse=True)
            qualified = [
                r for r in fresh
                if float(r["Score"]) >= float(min_score) and str(r.get("Action")) in {"READY", "WAIT"}
            ]
            watchlist = [r for r in fresh if r not in qualified][:3]
            if not watchlist and fresh:
                watchlist = fresh[:3]
            st.session_state["rapid_watchlist"] = watchlist

            quality_hist = list(st.session_state.get("rapid_quality_history", []))
            if fresh:
                strong_adx_count = sum(
                    1 for r in fresh if r.get("ADX") is not None and float(r.get("ADX")) >= float(cfg.trend_adx_starting)
                )
                quality_hist.append(
                    {
                        "ts": pd.Timestamp.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC"),
                        "best_action": str(fresh[0].get("Action", "SKIP")),
                        "best_score": float(fresh[0].get("Score", 0.0)),
                        "qualified_count": int(len(qualified)),
                        "strong_adx_share": float(strong_adx_count / max(len(fresh), 1)),
                    }
                )
                st.session_state["rapid_quality_history"] = quality_hist[-int(cfg.history_max_items):]

            # Mark current scan signature as evaluated, even when no qualified rows.
            st.session_state["rapid_sig"] = scan_sig
            if qualified:
                st.session_state["rapid_rows"] = qualified
                st.session_state["rapid_cache_ts"] = pd.Timestamp.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")
                st.session_state["rapid_auto_retry_done"] = False
                rows = qualified
            else:
                old_rows = st.session_state.get("rapid_rows", [])
                old_sig = st.session_state.get("rapid_sig")
                same_context = tuple(old_sig) == tuple(scan_sig) if old_sig else False
                rows = old_rows if same_context else []
                if rows:
                    ts = st.session_state.get("rapid_cache_ts", "unknown time")
                    st.warning(f"Rapid live scan returned no candidates. Showing previous snapshot from {ts}.")

    hist_summary = summarize_quality_history(st.session_state.get("rapid_quality_history", []))
    k1, k2, k3, k4 = st.columns(4, gap="small")
    with k1:
        st.markdown(
            "<div class='elite-card'>"
            "<div class='elite-label'>Tracked Scans</div>"
            f"<div class='elite-value'>{int(hist_summary['scans'])}</div>"
            "</div>",
            unsafe_allow_html=True,
        )
    with k2:
        st.markdown(
            "<div class='elite-card'>"
            "<div class='elite-label'>ENTER Rate (50)</div>"
            f"<div class='elite-value'>{hist_summary['ready_rate']:.1f}%</div>"
            "</div>",
            unsafe_allow_html=True,
        )
    with k3:
        st.markdown(
            "<div class='elite-card'>"
            "<div class='elite-label'>Avg Best Score (50)</div>"
            f"<div class='elite-value'>{hist_summary['avg_best_score']:.1f}</div>"
            "</div>",
            unsafe_allow_html=True,
        )
    with k4:
        st.markdown(
            "<div class='elite-card'>"
            "<div class='elite-label'>Trend-Friendly Share</div>"
            f"<div class='elite-value'>{hist_summary['trend_share']:.1f}%</div>"
            "</div>",
            unsafe_allow_html=True,
        )
    st.markdown(
        f"<div style='color:{TEXT_MUTED}; font-size:0.79rem; margin:0.2rem 0 0.55rem 0;'>"
        f"{_tip('Tracked Scans', 'Number of recent Rapid scans stored for quality tracking (up to 50).')} | "
        f"{_tip('ENTER Rate (50)', 'Percentage of recent scans where the best candidate was ENTER.')} | "
        f"{_tip('Avg Best Score (50)', 'Average top-candidate score across recent scans. Higher is better.')} | "
        f"{_tip('Trend-Friendly Share', 'Share of scans where candidate set had supportive ADX trend strength.')}"
        f"</div>",
        unsafe_allow_html=True,
    )

    def _to_display_df(source_rows: list[dict]) -> pd.DataFrame:
        df = pd.DataFrame(source_rows).copy()
        if df.empty:
            return df
        df = df[
            ["Coin", "Action", "Direction", "Score", "Grade", "Strength", "Setup", "Alignment", "AI Direction", "AI Agree", "ADX", "Entry", "SL", "TP1", "R:R"]
        ]
        df["Action"] = df["Action"].map(_action_badge)
        df["Direction"] = df["Direction"].map(_dir_badge)
        df["Grade"] = df["Grade"].map(_grade_icon)
        df["Setup"] = df["Setup"].map(_setup_icon)
        df["Alignment"] = df["Alignment"].map(_conv_icon)
        df["AI Direction"] = df["AI Direction"].map(_dir_badge)
        df["ADX"] = df["ADX"].map(_adx_icon)
        return df

    if not rows:
        watchlist = st.session_state.get("rapid_watchlist", [])
        st.markdown(
            f"<div class='elite-hero' style='border-color:rgba(255,209,102,0.35);'>"
            f"<div class='elite-hero-title' style='color:{WARNING};'>Watch Mode</div>"
            f"<div class='elite-sub'>No qualified rapid setup right now. Monitor near-miss candidates and wait for stronger alignment.</div>"
            f"</div>",
            unsafe_allow_html=True,
        )
        if should_scan and not watchlist and not st.session_state.get("rapid_auto_retry_done", False):
            st.session_state["rapid_auto_retry_done"] = True
            st.rerun()
        if watchlist:
            _render_candidate_cards(watchlist, title="Near-Miss Watchlist")
            with st.expander("Near-Miss Table (Compact)"):
                st.dataframe(_to_display_df(watchlist), width="stretch", hide_index=True)
        st.markdown(
            f"<details style='margin-top:0.45rem;'><summary style='color:{ACCENT}; cursor:pointer;'>Column Guide</summary>"
            f"<div style='color:{TEXT_MUTED}; font-size:0.83rem; line-height:1.7; margin-top:0.45rem;'>"
            f"<b>Action</b>: final quick class (ENTER/WATCH/SKIP). <b>Score</b>: setup quality 0-100. <b>Setup</b>/<b>Alignment</b>: structural alignment quality. "
            f"<b>Entry/SL/TP1</b>: draft trade levels; SL is invalidation.</div></details>",
            unsafe_allow_html=True,
        )
        return

    best = rows[0]
    action_color = POSITIVE if best["Action"] == "READY" else WARNING
    action_chip = "elite-chip-positive" if best["Action"] == "READY" else "elite-chip-warning"
    direction_chip = "elite-chip-positive" if best["Direction"] == "LONG" else ("elite-chip-negative" if best["Direction"] == "SHORT" else "elite-chip-warning")
    setup_chip = _chip_class(str(best["Setup"]))
    align_chip = _chip_class(str(best["Alignment"]))
    st.markdown(
        f"<div class='elite-hero' style='margin-top:0.55rem; border-color:rgba(0,212,255,0.24);'>"
        f"<div style='display:flex; justify-content:space-between; gap:8px; align-items:center; flex-wrap:wrap;'>"
        f"<div class='elite-hero-title'>Best Candidate: {best['Coin']}</div>"
        f"<div style='display:flex; gap:8px; flex-wrap:wrap;'>"
        f"<span class='elite-chip {action_chip}'>{_action_badge(best['Action'])}</span>"
        f"<span class='elite-chip {direction_chip}'>{_dir_badge(best['Direction'])}</span>"
        f"<span class='elite-chip {setup_chip}'>{_setup_icon(best['Setup'])}</span>"
        f"<span class='elite-chip {align_chip}'>{_conv_icon(best['Alignment'])}</span>"
        f"</div></div>"
        f"<div class='elite-grid'>"
        f"<div class='elite-mini'><div class='elite-label'>Score</div><div style='color:{action_color}; font-weight:700;'>{best['Score']:.1f} ({_grade_icon(best['Grade'])})</div></div>"
        f"<div class='elite-mini'><div class='elite-label'>Strength</div><div style='color:{ACCENT}; font-weight:700;'>{best['Strength']:.0f}%</div></div>"
        f"<div class='elite-mini'><div class='elite-label'>Entry</div><div style='color:{ACCENT}; font-weight:700;'>{best['Entry']}</div></div>"
        f"<div class='elite-mini'><div class='elite-label'>Stop Loss</div><div style='color:{NEGATIVE}; font-weight:700;'>{best['SL']}</div></div>"
        f"<div class='elite-mini'><div class='elite-label'>Target</div><div style='color:{POSITIVE}; font-weight:700;'>{best['TP1']}</div></div>"
        f"<div class='elite-mini'><div class='elite-label'>R:R</div><div style='color:{WARNING}; font-weight:700;'>{best['R:R']}</div></div>"
        f"</div></div>",
        unsafe_allow_html=True,
    )
    if st.button("Send Best Plan to Position", key="rapid_send_best", use_container_width=True):
        st.session_state["position_prefill_plan"] = {
            "coin": str(best["Coin"]),
            "direction": str(best["Direction"]) if str(best["Direction"]) in {"LONG", "SHORT"} else "LONG",
            "entry": float(best.get("Entry Raw") or 0.0),
            "sl": best.get("SL"),
            "tp1": best.get("TP1"),
            "source": "Rapid Best",
        }
        st.success("Plan sent to Position tab.")

    st.markdown(
        f"<details style='margin:0.55rem 0;'><summary style='color:{ACCENT}; cursor:pointer;'>How to Act Quickly</summary>"
        f"<div style='color:{TEXT_MUTED}; font-size:0.84rem; line-height:1.7; margin-top:0.45rem;'>"
        f"1. Start with ENTER setups. 2. Respect SL as invalidation. 3. WATCH rows are monitor-only until quality improves.</div></details>",
        unsafe_allow_html=True,
    )
    near_miss = [r for r in rows[1:4] if str(r.get("Action")) != "READY"]
    if near_miss:
        _render_candidate_cards(near_miss, title="Also Watch")
    st.markdown(f"<h4 style='color:{ACCENT}; margin-top:0.35rem;'>Rapid Tactical Table</h4>", unsafe_allow_html=True)
    st.dataframe(_to_display_df(rows), width="stretch", hide_index=True)
