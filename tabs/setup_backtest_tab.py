from __future__ import annotations

import html
import re

import numpy as np
import pandas as pd
import plotly.graph_objs as go

from core.backtest import (
    build_setup_outcome_study,
    summarize_setup_outcome_by_class,
    summarize_setup_outcome_study,
)
from core.signal_contract import strength_bucket
from ui.ctx import get_ctx
from ui.snapshot_cache import live_or_snapshot


def _setup_filter_key(value: str) -> str:
    s = str(value or "").strip().upper()
    if "TREND+AI" in s:
        return "TREND+AI"
    if "TREND-LED" in s:
        return "TREND-LED"
    if "AI-LED" in s:
        return "AI-LED"
    return "ALL"


def _best_hold_window(df_events: pd.DataFrame, forward_bars: int) -> tuple[str, float]:
    if df_events is None or df_events.empty:
        return "N/A", float("nan")
    means: list[float] = []
    for step in range(1, forward_bars + 1):
        col = f"Directional Return +{step} (%)"
        vals = pd.to_numeric(df_events.get(col, pd.Series(dtype=float)), errors="coerce").dropna()
        means.append(float(vals.mean()) if not vals.empty else float("nan"))
    arr = np.array(means, dtype=float)
    if not np.isfinite(arr).any():
        return "N/A", float("nan")
    best_idx = int(np.nanargmax(arr)) + 1
    best_val = float(arr[best_idx - 1])
    if not np.isfinite(best_val) or best_val <= 0:
        return "No positive edge", best_val
    left = max(1, best_idx - 1)
    right = min(forward_bars, best_idx + 1)
    if left == right:
        return f"+{left}", best_val
    return f"+{left} to +{right}", best_val


def _risk_balance(
    avg_fav: float,
    avg_adv: float,
    avg_outcome: float = float("nan"),
    win_rate: float = float("nan"),
) -> tuple[str, str]:
    fav = max(0.0, float(avg_fav) if np.isfinite(avg_fav) else 0.0)
    adv = max(0.0, float(avg_adv) if np.isfinite(avg_adv) else 0.0)
    if adv <= 1e-9:
        return "N/A", "Unknown"
    ratio = fav / adv
    if ratio >= 1.8:
        note = "Low"
    elif ratio >= 1.2:
        note = "Medium"
    else:
        note = "High"
    try:
        avg_outcome_f = float(avg_outcome)
    except Exception:
        avg_outcome_f = float("nan")
    if np.isfinite(avg_outcome_f):
        if avg_outcome_f < 0:
            note = "High"
        elif avg_outcome_f < 0.25 and note == "Low":
            note = "Medium"
    try:
        win_rate_f = float(win_rate)
    except Exception:
        win_rate_f = float("nan")
    if np.isfinite(win_rate_f):
        if win_rate_f < 40:
            note = "High"
        elif win_rate_f < 50 and note == "Low":
            note = "Medium"
    return f"{ratio:.2f}x", note


def _parse_coin_inputs(raw: str, normalize_fn, limit: int = 10) -> list[str]:
    tokens = re.split(r"[\s,;\n]+", str(raw or "").strip())
    out: list[str] = []
    seen: set[str] = set()
    for tok in tokens:
        t = str(tok or "").strip()
        if not t:
            continue
        norm = str(normalize_fn(t)).strip()
        if not norm or norm in seen:
            continue
        seen.add(norm)
        out.append(norm)
        if len(out) >= int(limit):
            break
    return out


def render(ctx: dict) -> None:
    st = get_ctx(ctx, "st")
    ACCENT = get_ctx(ctx, "ACCENT")
    TEXT_MUTED = get_ctx(ctx, "TEXT_MUTED")
    POSITIVE = get_ctx(ctx, "POSITIVE")
    WARNING = get_ctx(ctx, "WARNING")
    _normalize_coin_input = get_ctx(ctx, "_normalize_coin_input")
    _validate_coin_symbol = get_ctx(ctx, "_validate_coin_symbol")
    fetch_ohlcv = get_ctx(ctx, "fetch_ohlcv")
    analyse = get_ctx(ctx, "analyse")
    ml_ensemble_predict = get_ctx(ctx, "ml_ensemble_predict")
    signal_plain = get_ctx(ctx, "signal_plain")
    direction_key = get_ctx(ctx, "direction_key")
    _calc_conviction = get_ctx(ctx, "_calc_conviction")

    def _chip(text: str, tone: str, extra_class: str = "", title: str | None = None) -> str:
        if not str(text or "").strip():
            return ""
        tone_class = {
            "pos": "eb-pos",
            "neg": "eb-neg",
            "warn": "eb-warn",
            "muted": "eb-muted",
            "info": "eb-info",
        }.get(tone, "eb-muted")
        title_attr = f" title='{html.escape(title)}'" if title else ""
        return (
            f"<span class='eb-chip {tone_class} {extra_class}'{title_attr}>"
            f"{html.escape(str(text))}</span>"
        )

    def _setup_chip(value: str) -> str:
        v = str(value or "").strip()
        u = v.upper()
        if "TREND+AI" in u:
            return _chip(v, "pos")
        if "TREND-LED" in u:
            return _chip(v, "info", "eb-sc-trend-led")
        if "AI-LED" in u:
            return _chip(v, "info", "eb-sc-ai-led")
        return _chip(v, "warn")

    def _direction_chip(value: str) -> str:
        v = str(value or "").strip().upper()
        if "UPSIDE" in v:
            return _chip("Upside", "pos")
        if "DOWNSIDE" in v:
            return _chip("Downside", "neg")
        return _chip("Neutral", "warn")

    def _strength_chip(value: object) -> str:
        try:
            s = float(value)
        except Exception:
            return ""
        label = str(strength_bucket(s) or "MIXED").upper()
        text = f"{s:.0f}% ({label})"
        if label in {"STRONG", "VERY STRONG", "EXTREME", "GOOD"}:
            tone = "pos"
        elif label in {"MIXED", "STARTING"}:
            tone = "warn"
        else:
            tone = "neg"
        return _chip(text, tone)

    def _alignment_chip(value: str) -> str:
        v = str(value or "").strip().upper()
        if v == "HIGH":
            return _chip("HIGH", "pos")
        if v in {"MEDIUM", "TREND"}:
            return _chip(v, "warn")
        if v == "CONFLICT":
            return _chip("CONFLICT", "neg")
        if v == "WEAK":
            return _chip("WEAK", "warn")
        return _chip(v or "N/A", "muted")

    def _ai_ensemble_chip(direction: str, votes_text: str) -> str:
        d = str(direction or "").strip()
        tone = "warn"
        if str(direction or "").strip().upper().startswith("UPSIDE"):
            tone = "pos"
        elif str(direction or "").strip().upper().startswith("DOWNSIDE"):
            tone = "neg"
        m = re.search(r"(\d)\s*/\s*3", str(votes_text or ""))
        votes_n = int(m.group(1)) if m else 0
        votes_n = max(0, min(3, votes_n))
        dots = "".join(
            f"<span class='eb-ai-dot{' is-filled' if i < votes_n else ''}'></span>"
            for i in range(3)
        )
        tone_class = {
            "pos": "eb-pos",
            "neg": "eb-neg",
            "warn": "eb-warn",
            "muted": "eb-muted",
            "info": "eb-info",
        }.get(tone, "eb-muted")
        return (
            f"<span class='eb-chip {tone_class} eb-chip-ai'>"
            f"<span class='eb-ai-text'>{html.escape(d or 'Neutral')}</span>"
            f"<span class='eb-ai-dots'>{dots}</span>"
            f"</span>"
        )

    def _price_step_html(price_val: object, event_price: float, direction_value: object) -> str:
        try:
            p = float(price_val)
            e = float(event_price)
            if not np.isfinite(p) or not np.isfinite(e) or e <= 0:
                return ""
            raw_pct = ((p / e) - 1.0) * 100.0
            is_downside = str(direction_value or "").strip().upper().startswith("DOWNSIDE")
            directional_pct = (-raw_pct) if is_downside else raw_pct
            if directional_pct > 0:
                pct_html = f"<span class='eb-step-pos'>(+{directional_pct:.2f}%)</span>"
            elif directional_pct < 0:
                pct_html = f"<span class='eb-step-neg'>({directional_pct:.2f}%)</span>"
            else:
                pct_html = "<span class='eb-step-neu'>(0.00%)</span>"
            return f"<span class='eb-step-price'>${p:,.6f}</span> {pct_html}"
        except Exception:
            return ""

    def _render_event_table_html(df_events: pd.DataFrame, n_forward: int) -> None:
        cols = [
            "Event Time",
            "Coin",
            "Setup Confirm",
            "Direction",
            "Strength",
            "AI Ensemble",
            "Tech vs AI Alignment",
            "Event Price",
        ] + [f"Price +{i}" for i in range(1, n_forward + 1)] + [
            f"End Price (+{n_forward})",
            f"Return @+{n_forward} (%)",
        ]

        header_html = "".join(f"<th>{html.escape(c)}</th>" for c in cols)
        row_html = []
        for _, r in df_events.iterrows():
            event_price = pd.to_numeric(r.get("Event Price"), errors="coerce")
            cells = []
            for c in cols:
                if c == "Event Time":
                    ts = pd.to_datetime(r.get(c), errors="coerce")
                    txt = ts.strftime("%Y-%m-%d %H:%M") if pd.notna(ts) else ""
                    cells.append(f"<td><span class='eb-plain'>{html.escape(txt)}</span></td>")
                elif c == "Coin":
                    cells.append(f"<td><span class='eb-plain'>{html.escape(str(r.get(c, '')))}</span></td>")
                elif c == "Setup Confirm":
                    cells.append(f"<td>{_setup_chip(str(r.get(c, '')))}</td>")
                elif c == "Direction":
                    cells.append(f"<td>{_direction_chip(str(r.get(c, '')))}</td>")
                elif c == "Strength":
                    cells.append(f"<td>{_strength_chip(r.get(c))}</td>")
                elif c == "AI Ensemble":
                    cells.append(
                        f"<td>{_ai_ensemble_chip(str(r.get('AI Direction', 'Neutral')), str(r.get('AI Votes', '0/3')))}</td>"
                    )
                elif c == "Tech vs AI Alignment":
                    cells.append(f"<td>{_alignment_chip(str(r.get(c, '')))}</td>")
                elif c == "Event Price":
                    v = pd.to_numeric(r.get(c), errors="coerce")
                    txt = f"${float(v):,.6f}" if pd.notna(v) else ""
                    cells.append(f"<td><span class='eb-plain'>{html.escape(txt)}</span></td>")
                elif c.startswith("Price +"):
                    cells.append(
                        f"<td>{_price_step_html(r.get(c), float(event_price), r.get('Direction', ''))}</td>"
                    )
                elif c.startswith("End Price (+"):
                    v = pd.to_numeric(r.get(c), errors="coerce")
                    txt = f"${float(v):,.6f}" if pd.notna(v) else ""
                    cells.append(f"<td><span class='eb-plain'>{html.escape(txt)}</span></td>")
                elif c.startswith("Return @+"):
                    v = pd.to_numeric(r.get(c), errors="coerce")
                    if pd.notna(v):
                        cells.append(f"<td>{_chip(f'{float(v):+.2f}%', 'pos' if float(v) > 0 else ('neg' if float(v) < 0 else 'warn'))}</td>")
                    else:
                        cells.append("<td></td>")
                else:
                    cells.append(f"<td><span class='eb-plain'>{html.escape(str(r.get(c, '')))}</span></td>")
            row_html.append("<tr>" + "".join(cells) + "</tr>")

        st.markdown(
            f"""
            <style>
              .eb-wrap {{
                width:100%;
                overflow-x:auto;
                border:1px solid rgba(0,212,255,0.20);
                border-radius:12px;
                background:linear-gradient(180deg, rgba(6,10,18,0.96), rgba(4,8,14,0.96));
              }}
              .eb-table {{
                width:max-content;
                min-width:100%;
                border-collapse:separate;
                border-spacing:0;
                font-size:0.81rem;
                font-family:'Manrope','Segoe UI',sans-serif;
              }}
              .eb-table th {{
                text-align:left;
                padding:10px 10px;
                color:{TEXT_MUTED};
                font-weight:700;
                border-bottom:1px solid rgba(148,163,184,0.22);
                border-right:1px solid rgba(148,163,184,0.08);
                white-space:nowrap;
                position:sticky;
                top:0;
                z-index:2;
                background:linear-gradient(180deg, rgba(18,24,36,0.98), rgba(12,18,30,0.98));
              }}
              .eb-table td {{
                padding:8px 10px;
                color:#E5E7EB;
                border-bottom:1px solid rgba(148,163,184,0.12);
                border-right:1px solid rgba(148,163,184,0.07);
                white-space:nowrap;
              }}
              .eb-table tr:hover td {{ background-color:rgba(0,212,255,0.06); }}
              .eb-chip {{
                display:inline-flex;
                align-items:center;
                gap:6px;
                padding:2px 8px;
                border-radius:999px;
                border:1px solid rgba(148,163,184,0.34);
                background:rgba(148,163,184,0.10);
                font-size:0.74rem;
                font-weight:700;
                white-space:nowrap;
              }}
              .eb-chip-ai {{ gap:8px; min-height:26px; padding:2px 8px; }}
              .eb-ai-dots {{ display:inline-flex; align-items:center; gap:3px; }}
              .eb-ai-dot {{
                width:8px; height:8px; border-radius:999px; border:1px solid currentColor;
                background:transparent; opacity:0.55; flex:0 0 8px;
              }}
              .eb-ai-dot.is-filled {{ background:currentColor; opacity:1; }}
              .eb-pos {{ color:{POSITIVE}; border-color:rgba(0,255,136,0.42); background:rgba(0,255,136,0.10); }}
              .eb-neg {{ color:#FF3366; border-color:rgba(255,51,102,0.44); background:rgba(255,51,102,0.10); }}
              .eb-warn {{ color:{WARNING}; border-color:rgba(255,209,102,0.46); background:rgba(255,209,102,0.10); }}
              .eb-info {{ color:{ACCENT}; border-color:rgba(0,212,255,0.46); background:rgba(0,212,255,0.10); }}
              .eb-muted {{ color:{TEXT_MUTED}; border-color:rgba(140,161,182,0.35); background:rgba(140,161,182,0.08); }}
              .eb-sc-trend-led {{ color:#38BDF8 !important; border-color:rgba(56,189,248,0.52) !important; background:rgba(56,189,248,0.12) !important; }}
              .eb-sc-ai-led {{ color:#22D3EE !important; border-color:rgba(34,211,238,0.52) !important; background:rgba(34,211,238,0.12) !important; }}
              .eb-step-price {{ color:#E5E7EB; }}
              .eb-step-pos {{ color:{POSITIVE}; font-weight:700; }}
              .eb-step-neg {{ color:#FF3366; font-weight:700; }}
              .eb-step-neu {{ color:{TEXT_MUTED}; font-weight:700; }}
              .eb-plain {{ color:#E5E7EB; }}
            </style>
            <div class="eb-wrap">
              <table class="eb-table">
                <thead><tr>{header_html}</tr></thead>
                <tbody>{''.join(row_html)}</tbody>
              </table>
            </div>
            """,
            unsafe_allow_html=True,
        )

    def _render_kpi_cards(kpis: list[tuple[str, str, str]]) -> None:
        cards_html = []
        for label, value, tip in kpis:
            cards_html.append(
                "<div class='sob-kpi-card'>"
                f"<div class='sob-kpi-label' title='{html.escape(tip, quote=True)}'>"
                f"{html.escape(label)} <span class='sob-help'>?</span>"
                "</div>"
                f"<div class='sob-kpi-value'>{html.escape(value)}</div>"
                "</div>"
            )
        st.markdown(
            f"""
            <style>
              .sob-kpi-grid {{
                display:grid;
                grid-template-columns:repeat(5, minmax(150px, 1fr));
                gap:10px;
                width:100%;
              }}
              .sob-kpi-card {{
                border:1px solid rgba(0,212,255,0.24);
                border-radius:12px;
                padding:10px 12px;
                background:linear-gradient(180deg, rgba(8,14,24,0.94), rgba(5,10,18,0.94));
              }}
              .sob-kpi-label {{
                color:{TEXT_MUTED};
                font-size:0.77rem;
                font-weight:700;
                letter-spacing:0.04em;
                text-transform:uppercase;
                display:inline-flex;
                align-items:center;
                gap:6px;
                cursor:help;
              }}
              .sob-help {{
                display:inline-flex;
                align-items:center;
                justify-content:center;
                width:14px;
                height:14px;
                border-radius:999px;
                border:1px solid rgba(0,212,255,0.45);
                color:{ACCENT};
                font-size:0.64rem;
                font-weight:800;
                line-height:1;
              }}
              .sob-kpi-value {{
                margin-top:8px;
                color:#E5E7EB;
                font-size:1.85rem;
                font-weight:800;
                line-height:1.15;
                letter-spacing:-0.01em;
              }}
              @media (max-width: 980px) {{
                .sob-kpi-grid {{
                  grid-template-columns:repeat(2, minmax(140px, 1fr));
                }}
              }}
            </style>
            <div class='sob-kpi-grid'>
              {''.join(cards_html)}
            </div>
            """,
            unsafe_allow_html=True,
        )

    def _render_hover_table(
        df_table: pd.DataFrame,
        header_tips: dict[str, str],
        table_id: str,
    ) -> None:
        if df_table.empty:
            return
        cols = list(df_table.columns)
        header_html = "".join(
            "<th>"
            f"<span class='sob-hdr' title='{html.escape(header_tips.get(c, ''), quote=True)}'>"
            f"{html.escape(str(c))} <span class='sob-help'>?</span>"
            "</span>"
            "</th>"
            for c in cols
        )
        row_html: list[str] = []
        for _, row in df_table.iterrows():
            cells = []
            for c in cols:
                v = row.get(c, "")
                txt = "" if pd.isna(v) else str(v)
                cells.append(f"<td>{html.escape(txt)}</td>")
            row_html.append("<tr>" + "".join(cells) + "</tr>")
        st.markdown(
            f"""
            <style>
              .sob-wrap-{table_id} {{
                width:100%;
                overflow-x:auto;
                border:1px solid rgba(0,212,255,0.20);
                border-radius:12px;
                background:linear-gradient(180deg, rgba(10,16,26,0.94), rgba(6,11,20,0.94));
              }}
              .sob-table-{table_id} {{
                width:max-content;
                min-width:100%;
                border-collapse:separate;
                border-spacing:0;
                font-size:0.83rem;
                font-family:'Manrope','Segoe UI',sans-serif;
              }}
              .sob-table-{table_id} th {{
                text-align:left;
                padding:10px 10px;
                color:{TEXT_MUTED};
                font-weight:700;
                border-bottom:1px solid rgba(148,163,184,0.22);
                border-right:1px solid rgba(148,163,184,0.08);
                white-space:nowrap;
                background:linear-gradient(180deg, rgba(19,26,38,0.98), rgba(12,19,31,0.98));
              }}
              .sob-table-{table_id} td {{
                padding:9px 10px;
                color:#E5E7EB;
                border-bottom:1px solid rgba(148,163,184,0.12);
                border-right:1px solid rgba(148,163,184,0.07);
                white-space:nowrap;
              }}
              .sob-table-{table_id} tr:hover td {{
                background-color:rgba(0,212,255,0.05);
              }}
              .sob-hdr {{
                display:inline-flex;
                align-items:center;
                gap:6px;
                cursor:help;
              }}
            </style>
            <div class='sob-wrap-{table_id}'>
              <table class='sob-table-{table_id}'>
                <thead><tr>{header_html}</tr></thead>
                <tbody>{''.join(row_html)}</tbody>
              </table>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown(f"<h2 style='color:{ACCENT};'>Setup Backtest</h2>", unsafe_allow_html=True)
    st.markdown(
        f"<div class='panel-box' style='margin-bottom:1rem;'>"
        f"<b style='color:{ACCENT};'>Setup Outcome Study</b>"
        f"<ul style='color:{TEXT_MUTED}; font-size:0.88rem; line-height:1.7; margin-top:0.5rem;'>"
        f"<li>Select a Setup Confirm class (TREND+AI / TREND-led / AI-led / ALL).</li>"
        f"<li>The engine scans each closed candle and records every matching setup event.</li>"
        f"<li>For each event, it stores event price and forward path for the next N bars.</li>"
        f"<li>Use this to measure setup quality by occurrence frequency and forward outcome behavior.</li>"
        f"</ul></div>",
        unsafe_allow_html=True,
    )

    c1, c2 = st.columns(2)
    with c1:
        coins_raw = st.text_input(
            "Coins (max 10, comma-separated)",
            value="BTC",
            key="setup_bt_coins",
            placeholder="BTC, ETH, SOL",
            help="You can enter up to 10 symbols. Example: BTC, ETH, SOL",
        )
        timeframe = st.selectbox(
            "Timeframe",
            ["3m", "5m", "15m", "1h", "4h", "1d"],
            index=3,
            key="setup_bt_timeframe",
        )
        lookback_candles = st.slider(
            "Lookback Candles",
            200,
            3000,
            step=100,
            value=1000,
            key="setup_bt_lookback",
        )
    with c2:
        setup_filter = st.selectbox(
            "Setup Filter",
            ["ALL Setup Confirmations", "TREND+AI", "TREND-led", "AI-led"],
            index=0,
            key="setup_bt_filter",
        )
        forward_bars = st.slider(
            "Forward Bars (Outcome Window)",
            3,
            30,
            step=1,
            value=10,
            key="setup_bt_forward",
        )
        st.caption("Example: Forward Bars = 10 means event price vs next 10 bars outcome.")

    state_key = "setup_bt_last_result"
    run_now = st.button("Run Setup Outcome Study", type="primary", key="setup_bt_run")
    current_inputs_signature = {
        "coins_raw": str(coins_raw or "").strip(),
        "timeframe": str(timeframe),
        "lookback_candles": int(lookback_candles),
        "setup_filter": str(setup_filter),
        "forward_bars": int(forward_bars),
    }

    events_df: pd.DataFrame
    summary: dict
    by_class: pd.DataFrame
    setup_key = _setup_filter_key(setup_filter)
    cache_used: list[str] = []
    no_data: list[str] = []
    failed: list[str] = []
    coins: list[str] = []
    events_coin_count = 0

    if run_now:
        coins = _parse_coin_inputs(coins_raw, _normalize_coin_input, limit=10)
        if not coins:
            st.error("Please enter at least one valid coin symbol.")
            return
        invalid: list[str] = []
        for coin in coins:
            err = _validate_coin_symbol(coin)
            if err:
                invalid.append(str(coin))
        if invalid:
            st.error(f"Invalid symbol(s): {', '.join(invalid)}")
            return

        st.info(f"Fetching candles and scanning setup events across {len(coins)} coin(s)...")
        fetch_limit = int(lookback_candles)

        all_events: list[pd.DataFrame] = []
        for coin in coins:
            df_live = fetch_ohlcv(coin, timeframe, limit=fetch_limit)
            df, used_cache, cache_ts = live_or_snapshot(
                st,
                f"setup_outcome::{coin}::{timeframe}::{fetch_limit}::{forward_bars}::{setup_filter}",
                df_live,
                max_age_sec=1800,
                current_sig=(coin, timeframe, fetch_limit, forward_bars, setup_filter),
            )
            if used_cache:
                cache_used.append(f"{coin} ({cache_ts})")
            if df is None or df.empty:
                no_data.append(coin)
                continue
            try:
                ev = build_setup_outcome_study(
                    df=df,
                    analyzer=analyse,
                    ml_predictor=ml_ensemble_predict,
                    conviction_fn=_calc_conviction,
                    signal_plain_fn=signal_plain,
                    direction_key_fn=direction_key,
                    setup_filter=setup_key,
                    forward_bars=forward_bars,
                )
                if ev is None or ev.empty:
                    continue
                ev = ev.copy()
                ev["Coin"] = coin.split("/", 1)[0]
                all_events.append(ev)
            except Exception:
                failed.append(coin)

        if no_data and len(no_data) == len(coins):
            st.error("No data returned for selected coins/timeframe.")
            return
        if failed and len(failed) == len(coins):
            st.error("Setup outcome study failed for all selected coins.")
            return

        if all_events:
            events_df = pd.concat(all_events, ignore_index=True)
        else:
            events_df = pd.DataFrame()

        if events_df.empty:
            st.warning(
                "No matching setup events found in this window. "
                "Try ALL Setup Confirmations, another timeframe, or more lookback candles."
            )
            return

        summary = summarize_setup_outcome_study(events_df, forward_bars)
        by_class = summarize_setup_outcome_by_class(events_df, forward_bars)
        events_coin_count = len(all_events)

        st.session_state[state_key] = {
            "events_df": events_df,
            "summary": summary,
            "by_class": by_class,
            "forward_bars": int(forward_bars),
            "setup_key": setup_key,
            "coins": list(coins),
            "timeframe": str(timeframe),
            "cache_used": list(cache_used),
            "no_data": list(no_data),
            "failed": list(failed),
            "events_coin_count": int(events_coin_count),
            "inputs_signature": current_inputs_signature,
        }
    else:
        saved = st.session_state.get(state_key)
        if not isinstance(saved, dict):
            return
        saved_inputs_signature = saved.get("inputs_signature", {})
        if isinstance(saved_inputs_signature, dict) and saved_inputs_signature and saved_inputs_signature != current_inputs_signature:
            st.warning(
                "Inputs changed since the last run. Click 'Run Setup Outcome Study' to refresh results "
                "for current filters/timeframe."
            )
            return
        events_df = saved.get("events_df", pd.DataFrame())
        if not isinstance(events_df, pd.DataFrame) or events_df.empty:
            return
        forward_bars = int(saved.get("forward_bars", forward_bars))
        summary = saved.get("summary", summarize_setup_outcome_study(events_df, forward_bars))
        by_class = saved.get("by_class", summarize_setup_outcome_by_class(events_df, forward_bars))
        setup_key = str(saved.get("setup_key", setup_key))
        coins = list(saved.get("coins", []))
        timeframe = str(saved.get("timeframe", timeframe))
        cache_used = list(saved.get("cache_used", []))
        no_data = list(saved.get("no_data", []))
        failed = list(saved.get("failed", []))
        events_coin_count = int(saved.get("events_coin_count", 0))
        st.caption("Showing last run results. Click 'Run Setup Outcome Study' to refresh.")

    if cache_used:
        st.warning(
            f"Live data unavailable for some symbols. Using cache: {', '.join(cache_used[:4])}"
            + (" ..." if len(cache_used) > 4 else "")
        )
    if no_data:
        st.warning(
            f"No data for {len(no_data)} coin(s): {', '.join(no_data[:6])}"
            + (" ..." if len(no_data) > 6 else "")
        )
    if failed:
        st.warning(
            f"Analysis failed for {len(failed)} coin(s): {', '.join(failed[:6])}"
            + (" ..." if len(failed) > 6 else "")
        )

    st.success(
        f"Study complete. {int(summary['occurrences'])} events matched ({setup_key}) "
        f"across {events_coin_count}/{len(coins)} coin(s)."
    )

    ret_col = f"Return @+{forward_bars} (%)"
    best_window = _best_hold_window(events_df, forward_bars)[0]
    risk_balance_value, risk_note = _risk_balance(
        float(summary["avg_favorable_exc"]),
        float(summary["avg_adverse_exc"]),
        avg_outcome=float(summary.get("median_dir_return", float("nan"))),
        win_rate=float(summary.get("favorable_rate", float("nan"))),
    )

    _render_kpi_cards(
        [
            (
                "Events Found",
                f"{int(summary['occurrences'])}",
                "How many setup events matched your filters in this sample window.",
            ),
            (
                f"Win Chance @+{forward_bars}",
                f"{summary['favorable_rate']:.1f}%",
                "Share of events where directional return at the selected horizon is positive.",
            ),
            (
                "Best Hold Window",
                best_window,
                "Forward-bar range where mean directional edge is strongest. "
                "'No positive edge' means none of the tested windows had positive mean edge.",
            ),
            (
                "Risk Balance",
                risk_balance_value,
                "Average favorable excursion divided by average adverse excursion. "
                "Higher is better because upside movement dominates downside movement.",
            ),
            (
                "Risk Note",
                risk_note,
                "Qualitative risk bucket combining excursion balance, win-rate, and directional outcome. "
                "Low = favorable structure; High = weaker or unstable structure.",
            ),
        ]
    )

    st.caption(
        "Simple read: Win Chance shows how often the setup moved in expected direction at the selected horizon; "
        "Best Hold Window shows where edge peaks (or No positive edge if all windows are weak); "
        "Risk Balance compares favorable move vs adverse move."
    )

    if int(summary["occurrences"]) < 30:
        st.warning("Sample size is small (<30 events). Treat this as exploratory, not definitive.")

    st.markdown("### Setup Class Summary")
    class_rows: list[dict] = []
    for setup_class, grp in events_df.groupby("Setup Confirm", dropna=False):
        class_window, _ = _best_hold_window(grp, forward_bars)
        class_returns = pd.to_numeric(grp.get(ret_col, pd.Series(dtype=float)), errors="coerce")
        class_returns_valid = class_returns.dropna()
        class_fav = pd.to_numeric(grp.get("Favorable Excursion (%)", pd.Series(dtype=float)), errors="coerce").mean()
        class_adv = pd.to_numeric(grp.get("Adverse Excursion (%)", pd.Series(dtype=float)), errors="coerce").mean()
        class_win_rate = (
            float((class_returns_valid > 0).mean() * 100.0)
            if not class_returns_valid.empty
            else float("nan")
        )
        class_median = (
            float(np.nanmedian(class_returns_valid.values))
            if not class_returns_valid.empty
            else float("nan")
        )
        class_mean = float(class_returns_valid.mean()) if not class_returns_valid.empty else float("nan")
        class_ratio, class_risk = _risk_balance(
            float(class_fav),
            float(class_adv),
            avg_outcome=class_median,
            win_rate=class_win_rate,
        )
        class_rows.append(
            {
                "Setup Class": str(setup_class),
                "Events": int(len(grp)),
                "Win %": f"{class_win_rate:.1f}%" if np.isfinite(class_win_rate) else "N/A",
                "Best Window": class_window,
                "Avg Outcome": f"{class_mean:+.2f}%" if np.isfinite(class_mean) else "N/A",
                "Risk Note": class_risk,
                "Risk Balance": class_ratio,
            }
        )

    class_view = pd.DataFrame(class_rows)
    if class_view.empty:
        st.info("No class-level summary available.")
    else:
        class_view = class_view.sort_values(by=["Events"], ascending=False).reset_index(drop=True)
        class_display = class_view[
            ["Setup Class", "Events", "Win %", "Best Window", "Avg Outcome", "Risk Note"]
        ].copy()
        _render_hover_table(
            class_display,
            {
                "Setup Class": "Which Setup Confirm bucket produced the events (TREND+AI / TREND-led / AI-led).",
                "Events": "How many times this setup class appeared in the selected sample.",
                "Win %": "Percent of events with positive directional return at the selected horizon (+N bars).",
                "Best Window": "Forward-bar range where this class had the strongest mean directional edge.",
                "Avg Outcome": "Average directional return at the selected horizon (+N bars).",
                "Risk Note": "Risk bucket from excursion balance + win-rate + directional outcome quality.",
            },
            table_id="class-summary",
        )
    with st.expander("Advanced metrics and event details", expanded=False):
        st.markdown("#### Raw class breakdown")
        if by_class.empty:
            st.info("No raw class-level metrics available.")
        else:
            b = by_class.copy()
            b["FavorableRate"] = b["FavorableRate"].map(lambda v: f"{float(v):.1f}%")
            b["MedianDirectionalReturn"] = b["MedianDirectionalReturn"].map(lambda v: f"{float(v):+.2f}%")
            b["AvgDirectionalReturn"] = b["AvgDirectionalReturn"].map(lambda v: f"{float(v):+.2f}%")
            b["AvgFavorableExcursion"] = b["AvgFavorableExcursion"].map(lambda v: f"{float(v):+.2f}%")
            b["AvgAdverseExcursion"] = b["AvgAdverseExcursion"].map(lambda v: f"{float(v):+.2f}%")
            _render_hover_table(
                b,
                {
                    "Setup Confirm": "Raw setup class key used in aggregation.",
                    "Occurrences": "Number of detected events for this class.",
                    "FavorableRate": "Percent of events with positive directional return at the selected horizon.",
                    "MedianDirectionalReturn": "Median directional return at selected horizon; robust against outliers.",
                    "AvgDirectionalReturn": "Mean directional return at selected horizon.",
                    "AvgFavorableExcursion": "Average maximum favorable move reached within the forward window.",
                    "AvgAdverseExcursion": "Average maximum adverse move reached within the forward window.",
                },
                table_id="raw-breakdown",
            )

        st.markdown(f"#### Holding Edge by Bar (next {forward_bars} bars)")

        def _mean_path(df_slice: pd.DataFrame) -> list[float]:
            points: list[float] = []
            for step in range(1, forward_bars + 1):
                col = f"Directional Return +{step} (%)"
                vals = pd.to_numeric(df_slice.get(col, pd.Series(dtype=float)), errors="coerce").dropna()
                points.append(float(vals.mean()) if not vals.empty else float("nan"))
            return points

        fig = go.Figure()
        x_axis = list(range(1, forward_bars + 1))

        if setup_key == "ALL":
            st.caption(
                "ALL mode overlays each setup class separately (no blended average). "
                "Above 0 means class edge is positive at that bar; below 0 means edge weakens/reverses."
            )
            class_specs = [
                ("TREND+AI", POSITIVE, "solid", "circle"),
                ("TREND-led", "#38BDF8", "solid", "square"),
                ("AI-led", "#A78BFA", "dot", "diamond"),
            ]
            plotted = 0
            for cls_name, cls_color, cls_dash, cls_symbol in class_specs:
                cls_df = events_df[events_df["Setup Confirm"].astype(str) == cls_name]
                if cls_df.empty:
                    continue
                path_points = _mean_path(cls_df)
                cls_ret_col = f"Return @+{forward_bars} (%)"
                cls_ret = pd.to_numeric(cls_df.get(cls_ret_col, pd.Series(dtype=float)), errors="coerce")
                cls_win = float((cls_ret > 0).mean() * 100.0) if cls_ret.notna().any() else 0.0
                fig.add_trace(
                    go.Scatter(
                        x=x_axis,
                        y=path_points,
                        mode="lines+markers",
                        name=f"{cls_name} (n={len(cls_df)}, win {cls_win:.1f}%)",
                        line=dict(color=cls_color, width=2, dash=cls_dash),
                        marker=dict(size=5, symbol=cls_symbol),
                    )
                )
                plotted += 1
            if plotted == 0:
                st.info("No class path available for the selected sample.")
        else:
            st.caption(
                "This chart shows average directional edge for the selected setup class. "
                "Above 0 means the setup is still moving in expected direction on average; "
                "below 0 means edge weakens or reverses."
            )
            path_points = _mean_path(events_df)
            cls_ret = pd.to_numeric(events_df.get(ret_col, pd.Series(dtype=float)), errors="coerce")
            cls_win = float((cls_ret > 0).mean() * 100.0) if cls_ret.notna().any() else 0.0
            fig.add_trace(
                go.Scatter(
                    x=x_axis,
                    y=path_points,
                    mode="lines+markers",
                    name=f"{setup_key} (n={len(events_df)}, win {cls_win:.1f}%)",
                    line=dict(color=POSITIVE, width=2),
                    marker=dict(size=5),
                    fill="tozeroy",
                    fillcolor="rgba(6,214,160,0.12)",
                )
            )

        fig.add_hline(y=0.0, line=dict(color=WARNING, dash="dash", width=1))
        fig.update_layout(
            template="plotly_dark",
            height=320,
            xaxis_title="Bars After Event",
            yaxis_title="Directional Return (%)",
            showlegend=True,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0.0),
        )
        st.plotly_chart(fig, width="stretch")

        st.markdown("#### Event table")
        _render_event_table_html(events_df, forward_bars)

    csv_data = events_df.to_csv(index=False).encode("utf-8")
    if len(coins) == 1:
        filename = f"{coins[0].replace('/', '_')}_{timeframe}_setup_outcome.csv"
    else:
        filename = f"multi_{len(coins)}_{timeframe}_setup_outcome.csv"
    st.download_button(
        "Download Setup Outcome Study (CSV)",
        data=csv_data,
        file_name=filename,
        mime="text/csv",
    )
