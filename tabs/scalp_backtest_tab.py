from __future__ import annotations

from collections import Counter
import html
import re
import time

import numpy as np
import pandas as pd
import plotly.graph_objs as go

from core.backtest import (
    build_scalp_outcome_study,
    summarize_scalp_outcome_study,
)
from core.signal_contract import strength_bucket
from ui.ctx import get_ctx
from ui.primitives import render_help_details, render_kpi_grid, render_page_header
from ui.snapshot_cache import live_or_snapshot


def _safe_float(v: object, default: float = float("nan")) -> float:
    try:
        n = float(v)
        return n if np.isfinite(n) else default
    except Exception:
        return default


def _avg_hit_bar(df_events: pd.DataFrame) -> float:
    if df_events is None or df_events.empty:
        return float("nan")
    hit = pd.to_numeric(df_events.get("Hit Bar", pd.Series(dtype=float)), errors="coerce")
    hit = hit.replace([np.inf, -np.inf], np.nan).dropna()
    return float(hit.mean()) if not hit.empty else float("nan")


def _parse_custom_bases(raw: str, limit: int = 10) -> list[str]:
    tokens = re.split(r"[\s,;\n]+", str(raw or "").upper().strip())
    out: list[str] = []
    seen: set[str] = set()
    for tok in tokens:
        t = tok.strip()
        if not t:
            continue
        if "/" in t:
            t = t.split("/", 1)[0].strip()
        for suf in ("-USDT", "_USDT", "USDT", "-USD", "_USD", "USD"):
            if t.endswith(suf) and len(t) > len(suf):
                t = t[: -len(suf)]
                break
        t = re.sub(r"[^A-Z0-9]", "", t)
        if len(t) < 2 or len(t) > 15:
            continue
        if t in seen:
            continue
        seen.add(t)
        out.append(t)
        if len(out) >= int(limit):
            break
    return out


def render(ctx: dict) -> None:
    st = get_ctx(ctx, "st")
    ACCENT = get_ctx(ctx, "ACCENT")
    TEXT_MUTED = get_ctx(ctx, "TEXT_MUTED")
    POSITIVE = get_ctx(ctx, "POSITIVE")
    WARNING = get_ctx(ctx, "WARNING")
    fetch_ohlcv = get_ctx(ctx, "fetch_ohlcv")
    get_top_volume_usdt_symbols = get_ctx(ctx, "get_top_volume_usdt_symbols")
    analyse = get_ctx(ctx, "analyse")
    ml_ensemble_predict = get_ctx(ctx, "ml_ensemble_predict")
    signal_plain = get_ctx(ctx, "signal_plain")
    direction_key = get_ctx(ctx, "direction_key")
    _calc_conviction = get_ctx(ctx, "_calc_conviction")
    get_scalping_entry_target = get_ctx(ctx, "get_scalping_entry_target")
    scalp_quality_gate = get_ctx(ctx, "scalp_quality_gate")
    _sr_lookback = get_ctx(ctx, "_sr_lookback")

    STABLE_BASES = {
        "USDT", "USDC", "BUSD", "DAI", "TUSD", "USDE", "FDUSD", "PYUSD",
        "RLUSD", "USDP", "GUSD", "EURS", "EURC",
    }

    def _chip(text: str, tone: str, extra_class: str = "", title: str | None = None) -> str:
        if not str(text or "").strip():
            return ""
        tone_class = {
            "pos": "sb-pos",
            "neg": "sb-neg",
            "warn": "sb-warn",
            "muted": "sb-muted",
            "info": "sb-info",
        }.get(tone, "sb-muted")
        title_attr = f" title='{html.escape(title)}'" if title else ""
        return (
            f"<span class='sb-chip {tone_class} {extra_class}'{title_attr}>"
            f"{html.escape(str(text))}</span>"
        )

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
            f"<span class='sb-ai-dot{' is-filled' if i < votes_n else ''}'></span>"
            for i in range(3)
        )
        tone_class = {
            "pos": "sb-pos",
            "neg": "sb-neg",
            "warn": "sb-warn",
            "muted": "sb-muted",
            "info": "sb-info",
        }.get(tone, "sb-muted")
        return (
            f"<span class='sb-chip {tone_class} sb-chip-ai'>"
            f"<span class='sb-ai-text'>{html.escape(d or 'Neutral')}</span>"
            f"<span class='sb-ai-dots'>{dots}</span>"
            f"</span>"
        )

    def _outcome_chip(value: str) -> str:
        v = str(value or "").strip().upper()
        if v == "TP":
            return _chip("TP", "pos")
        if v in {"SL", "BOTH"}:
            return _chip(v, "neg")
        if v == "TIMEOUT":
            return _chip("TIMEOUT", "warn")
        return _chip(v or "N/A", "muted")

    def _gate_reason_label(code: str) -> str:
        key = str(code or "").strip().upper()
        labels = {
            "NO_SCALP_DIRECTION": "No scalp direction",
            "SIGNAL_DIRECTION_NEUTRAL": "Signal neutral",
            "DIRECTION_MISMATCH": "Direction mismatch",
            "CONFLICT": "Tech/AI conflict",
            "RR_TOO_LOW": "R:R too low",
            "ADX_TOO_LOW": "ADX too low",
            "STRENGTH_TOO_LOW": "Strength too low",
            "INVALID_LEVELS": "Invalid levels",
            "UNKNOWN_GATE_REJECT": "Unknown reject",
        }
        return labels.get(key, key.title())

    def _price_step_html(price_val: object, event_price: float) -> str:
        try:
            p = float(price_val)
            e = float(event_price)
            if not np.isfinite(p) or not np.isfinite(e) or e <= 0:
                return ""
            pct = ((p / e) - 1.0) * 100.0
            if pct > 0:
                pct_html = f"<span class='sb-step-pos'>(+{pct:.2f}%)</span>"
            elif pct < 0:
                pct_html = f"<span class='sb-step-neg'>({pct:.2f}%)</span>"
            else:
                pct_html = "<span class='sb-step-neu'>(0.00%)</span>"
            return f"<span class='sb-step-price'>${p:,.6f}</span> {pct_html}"
        except Exception:
            return ""

    def _render_kpi_cards(kpis: list[tuple[str, str, str]]) -> None:
        render_kpi_grid(
            st,
            columns=5,
            items=[
                {
                    "label": label,
                    "label_title": tip,
                    "value": value,
                }
                for label, value, tip in kpis
            ],
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
            f"<span class='sb-hdr' title='{html.escape(header_tips.get(c, ''), quote=True)}'>"
            f"{html.escape(str(c))} <span class='sb-help'>?</span>"
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
              .sb-wrap-{table_id} {{
                width:100%;
                overflow-x:auto;
                border:1px solid rgba(0,212,255,0.20);
                border-radius:12px;
                background:linear-gradient(180deg, rgba(10,16,26,0.94), rgba(6,11,20,0.94));
              }}
              .sb-table-{table_id} {{
                width:max-content;
                min-width:100%;
                border-collapse:separate;
                border-spacing:0;
                font-size:0.83rem;
                font-family:'Manrope','Segoe UI',sans-serif;
              }}
              .sb-table-{table_id} th {{
                text-align:left;
                padding:10px 10px;
                color:{TEXT_MUTED};
                font-weight:700;
                border-bottom:1px solid rgba(148,163,184,0.22);
                border-right:1px solid rgba(148,163,184,0.08);
                white-space:nowrap;
                background:linear-gradient(180deg, rgba(19,26,38,0.98), rgba(12,19,31,0.98));
              }}
              .sb-table-{table_id} td {{
                padding:9px 10px;
                color:#E5E7EB;
                border-bottom:1px solid rgba(148,163,184,0.12);
                border-right:1px solid rgba(148,163,184,0.07);
                white-space:nowrap;
              }}
              .sb-table-{table_id} tr:hover td {{
                background-color:rgba(0,212,255,0.05);
              }}
              .sb-hdr {{
                display:inline-flex;
                align-items:center;
                gap:6px;
                cursor:help;
              }}
            </style>
            <div class='sb-wrap-{table_id}'>
              <table class='sb-table-{table_id}'>
                <thead><tr>{header_html}</tr></thead>
                <tbody>{''.join(row_html)}</tbody>
              </table>
            </div>
            """,
            unsafe_allow_html=True,
        )

    def _render_event_table_html(df_events: pd.DataFrame, n_forward: int) -> None:
        cols = [
            "Event Time",
            "Coin",
            "Pair",
            "Setup Confirm",
            "Direction",
            "Strength",
            "AI Ensemble",
            "Tech vs AI Alignment",
            "Event Price",
            "Target",
            "Stop",
            "R:R",
            "Outcome",
            "Hit Bar",
        ] + [f"Price +{i}" for i in range(1, n_forward + 1)] + [
            f"End Price (+{n_forward})",
            f"Realized Return @+{n_forward} (%)",
            f"Close Directional Return @+{n_forward} (%)",
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
                    cells.append(f"<td><span class='sb-plain'>{html.escape(txt)}</span></td>")
                elif c in {"Coin", "Pair", "Setup Confirm"}:
                    cells.append(f"<td><span class='sb-plain'>{html.escape(str(r.get(c, '')))}</span></td>")
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
                elif c in {"Event Price", "Target", "Stop"}:
                    v = pd.to_numeric(r.get(c), errors="coerce")
                    txt = f"${float(v):,.6f}" if pd.notna(v) else ""
                    cells.append(f"<td><span class='sb-plain'>{html.escape(txt)}</span></td>")
                elif c == "R:R":
                    v = pd.to_numeric(r.get(c), errors="coerce")
                    if pd.notna(v):
                        tone = "pos" if float(v) >= 1.8 else ("warn" if float(v) >= 1.4 else "neg")
                        cells.append(f"<td>{_chip(f'{float(v):.2f}', tone)}</td>")
                    else:
                        cells.append("<td></td>")
                elif c == "Outcome":
                    cells.append(f"<td>{_outcome_chip(str(r.get(c, '')))}</td>")
                elif c == "Hit Bar":
                    v = pd.to_numeric(r.get(c), errors="coerce")
                    txt = f"+{int(v)}" if pd.notna(v) else ""
                    cells.append(f"<td><span class='sb-plain'>{html.escape(txt)}</span></td>")
                elif c.startswith("Price +"):
                    cells.append(f"<td>{_price_step_html(r.get(c), float(event_price))}</td>")
                elif c.startswith("End Price (+"):
                    v = pd.to_numeric(r.get(c), errors="coerce")
                    txt = f"${float(v):,.6f}" if pd.notna(v) else ""
                    cells.append(f"<td><span class='sb-plain'>{html.escape(txt)}</span></td>")
                elif c.startswith("Realized Return @+") or c.startswith("Return @+"):
                    v = pd.to_numeric(r.get(c), errors="coerce")
                    if pd.isna(v) and c.startswith("Realized Return @+"):
                        legacy_col = c.replace("Realized Return @+", "Return @+")
                        v = pd.to_numeric(r.get(legacy_col), errors="coerce")
                    if pd.notna(v):
                        cells.append(
                            f"<td>{_chip(f'{float(v):+.2f}%', 'pos' if float(v) > 0 else ('neg' if float(v) < 0 else 'warn'))}</td>"
                        )
                    else:
                        cells.append("<td></td>")
                elif c.startswith("Close Directional Return @+"):
                    v = pd.to_numeric(r.get(c), errors="coerce")
                    if pd.notna(v):
                        cells.append(
                            f"<td>{_chip(f'{float(v):+.2f}%', 'pos' if float(v) > 0 else ('neg' if float(v) < 0 else 'warn'))}</td>"
                        )
                    else:
                        cells.append("<td></td>")
                else:
                    cells.append(f"<td><span class='sb-plain'>{html.escape(str(r.get(c, '')))}</span></td>")
            row_html.append("<tr>" + "".join(cells) + "</tr>")

        st.markdown(
            f"""
            <style>
              .sb-wrap {{
                width:100%;
                overflow-x:auto;
                border:1px solid rgba(0,212,255,0.20);
                border-radius:12px;
                background:linear-gradient(180deg, rgba(6,10,18,0.96), rgba(4,8,14,0.96));
              }}
              .sb-table {{
                width:max-content;
                min-width:100%;
                border-collapse:separate;
                border-spacing:0;
                font-size:0.81rem;
                font-family:'Manrope','Segoe UI',sans-serif;
              }}
              .sb-table th {{
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
              .sb-table td {{
                padding:8px 10px;
                color:#E5E7EB;
                border-bottom:1px solid rgba(148,163,184,0.12);
                border-right:1px solid rgba(148,163,184,0.07);
                white-space:nowrap;
              }}
              .sb-table tr:hover td {{ background-color:rgba(0,212,255,0.06); }}
              .sb-chip {{
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
              .sb-chip-ai {{ gap:8px; min-height:26px; padding:2px 8px; }}
              .sb-ai-dots {{ display:inline-flex; align-items:center; gap:3px; }}
              .sb-ai-dot {{
                width:8px; height:8px; border-radius:999px; border:1px solid currentColor;
                background:transparent; opacity:0.55; flex:0 0 8px;
              }}
              .sb-ai-dot.is-filled {{ background:currentColor; opacity:1; }}
              .sb-pos {{ color:{POSITIVE}; border-color:rgba(0,255,136,0.42); background:rgba(0,255,136,0.10); }}
              .sb-neg {{ color:#FF3366; border-color:rgba(255,51,102,0.44); background:rgba(255,51,102,0.10); }}
              .sb-warn {{ color:{WARNING}; border-color:rgba(255,209,102,0.46); background:rgba(255,209,102,0.10); }}
              .sb-info {{ color:{ACCENT}; border-color:rgba(0,212,255,0.46); background:rgba(0,212,255,0.10); }}
              .sb-muted {{ color:{TEXT_MUTED}; border-color:rgba(140,161,182,0.35); background:rgba(140,161,182,0.08); }}
              .sb-step-price {{ color:#E5E7EB; }}
              .sb-step-pos {{ color:{POSITIVE}; font-weight:700; }}
              .sb-step-neg {{ color:#FF3366; font-weight:700; }}
              .sb-step-neu {{ color:{TEXT_MUTED}; font-weight:700; }}
              .sb-plain {{ color:#E5E7EB; }}
            </style>
            <div class="sb-wrap">
              <table class="sb-table">
                <thead><tr>{header_html}</tr></thead>
                <tbody>{''.join(row_html)}</tbody>
              </table>
            </div>
            """,
            unsafe_allow_html=True,
        )

    render_page_header(
        st,
        title="Scalp Backtest",
        intro_html=(
            "Outcome study for the scalp engine used in Market tab. "
            "It validates entry/stop/target behavior and the scalp quality gate over historical events, "
            "so you can see how often setups hit TP first, SL first, or time out across a study window."
        ),
    )
    render_help_details(
        st,
        summary="How to read quickly",
        body_html=(
            "1. This tab validates <b>scalp setup behavior</b> using the same scalp engine as Market tab "
            "(entry/stop/target + quality gate).<br>"
            "2. If <b>Custom Coins</b> is empty, scan runs on the top-volume universe (controlled by Universe Size).<br>"
            "3. If <b>Custom Coins</b> has symbols, scan runs <b>only</b> on those symbols (Universe Size is ignored).<br>"
            "4. Each event is tracked for the next N bars and labeled as <b>TP-first</b>, <b>SL-first</b>, or <b>TIMEOUT</b>.<br>"
            "5. Use Direction/Coin Scoreboards to decide which side or symbols are currently worth testing in live flow."
        ),
    )

    c1, c2 = st.columns(2)
    with c1:
        timeframe = st.selectbox(
            "Timeframe",
            ["5m", "15m", "1h", "4h", "1d"],
            index=1,
            key="scalp_bt_timeframe",
        )
        lookback_candles = st.slider("Lookback Candles", 200, 3000, step=100, value=1000, key="scalp_bt_lookback")
    with c2:
        universe_size = st.slider(
            "Universe Size (Top Volume Pairs)",
            10,
            120,
            step=5,
            value=40,
            key="scalp_bt_universe",
        )
        forward_bars = st.slider(
            "Tracking Window (bars after event)",
            3,
            30,
            step=1,
            value=10,
            key="scalp_bt_forward",
        )
        st.caption("Example: 10 means each scalp event is tracked for 10 bars after trigger.")

    cc1, cc2 = st.columns([0.86, 0.14])
    with cc1:
        custom_coin_input = st.text_input(
            "Custom Coins (max 10)",
            value=st.session_state.get("scalp_bt_custom_coin_input", ""),
            key="scalp_bt_custom_coin_input",
            placeholder="BTC, ETH, SOL",
            help="Optional. If filled, study runs only on these symbols.",
        )
    with cc2:
        # Align clear button with custom input field row.
        st.markdown("<div style='height:1.9rem;'></div>", unsafe_allow_html=True)
        clear_custom = st.button("Clear", width="stretch", key="scalp_bt_clear_custom")
    if clear_custom:
        st.session_state["scalp_bt_custom_coin_input"] = ""
        st.rerun()

    custom_bases = _parse_custom_bases(custom_coin_input, limit=10)
    custom_mode_active = bool(custom_bases)
    if custom_mode_active:
        preview = ", ".join(custom_bases[:6])
        more = "" if len(custom_bases) <= 6 else f" +{len(custom_bases) - 6}"
        st.caption(
            f"Custom mode active: scanning {len(custom_bases)} symbol(s): {preview}{more}. "
            "Universe Size is ignored while custom mode is active."
        )

    if hasattr(st, "checkbox"):
        exclude_stables = st.checkbox(
            "Exclude stablecoins",
            value=True,
            key="scalp_bt_exclude_stables",
            help="Enabled by default to keep scalp study focused on directional assets.",
        )
    else:
        # Contract-test dummy Streamlit object fallback.
        exclude_stables = True
    est_symbols = len(custom_bases) if custom_mode_active else int(universe_size)
    est_workload = int(lookback_candles) * int(est_symbols)
    st.caption(
        f"Estimated workload: {int(est_symbols)} symbols x {int(lookback_candles)} candles = {est_workload:,} rows to scan."
    )

    if not st.button("Run Scalp Outcome Study", type="primary", key="scalp_bt_run"):
        return

    fetch_limit = int(lookback_candles)
    pairs, _provider_rows = get_top_volume_usdt_symbols(top_n=int(universe_size), vs_currency="usd")
    pairs = [str(p).upper() for p in pairs if isinstance(p, str) and "/" in p]
    pairs = list(dict.fromkeys(pairs))[: int(universe_size)]
    if custom_mode_active:
        pairs = [f"{b}/USDT" for b in custom_bases]
    if exclude_stables:
        before_n = len(pairs)
        pairs = [p for p in pairs if p.split("/", 1)[0] not in STABLE_BASES]
        removed_n = max(0, before_n - len(pairs))
        if removed_n:
            st.caption(f"Stablecoin filter removed {removed_n} pair(s) from study universe.")
    if not pairs:
        st.error("Universe is empty. No exchange pairs available for scalp study.")
        return

    # Guardrail to keep UI responsive on large universe*lookback combinations.
    max_rows_budget = 90000
    estimated_rows = len(pairs) * int(fetch_limit)
    if estimated_rows > max_rows_budget and fetch_limit > 0:
        capped_n = max(10, int(max_rows_budget // int(fetch_limit)))
        capped_n = min(capped_n, len(pairs))
        if capped_n < len(pairs):
            st.warning(
                f"Workload capped for responsiveness: scanning {capped_n}/{len(pairs)} symbols "
                f"(budget {max_rows_budget:,} rows)."
            )
            pairs = pairs[:capped_n]

    st.info(
        f"Fetching candles and scanning scalp-qualified events across {len(pairs)} symbols..."
    )
    fetch_limit = int(lookback_candles)
    all_events: list[pd.DataFrame] = []
    symbols_with_events = 0
    no_data_symbols = 0
    failed_symbols = 0
    cache_hits = 0
    diag_bars_evaluated = 0
    diag_analysis_fail = 0
    diag_signal_side_reject = 0
    diag_plan_fail = 0
    diag_gate_pass_candidates = 0
    diag_side_key_reject = 0
    diag_price_level_reject = 0
    diag_forward_window_reject = 0
    diag_gate_reject_counts: Counter[str] = Counter()
    diag_plan_fail_counts: Counter[str] = Counter()
    processed_symbols = 0
    truncated_by_time = False
    runtime_budget_sec = 75.0
    started_at = time.time()
    progress_bar = st.progress(0, text="Starting scalp scan...")
    for pair in pairs:
        elapsed = time.time() - started_at
        if elapsed > runtime_budget_sec:
            truncated_by_time = True
            break
        try:
            df_live = fetch_ohlcv(pair, timeframe, limit=fetch_limit)
            df, used_cache, _cache_ts = live_or_snapshot(
                st,
                f"scalp_outcome::{pair}::{timeframe}::{fetch_limit}::{forward_bars}",
                df_live,
                max_age_sec=1800,
                current_sig=(pair, timeframe, fetch_limit, forward_bars),
            )
            if used_cache:
                cache_hits += 1
            if df is None or df.empty:
                no_data_symbols += 1
                continue

            # Keep parity with Market tab: compute decisions on closed-candle context.
            df_eval = df.iloc[:-1].copy() if len(df) > 1 else df.copy()
            if df_eval is None or len(df_eval) <= 55:
                no_data_symbols += 1
                continue

            events_sym = build_scalp_outcome_study(
                df=df_eval,
                analyzer=analyse,
                ml_predictor=ml_ensemble_predict,
                conviction_fn=_calc_conviction,
                signal_plain_fn=signal_plain,
                direction_key_fn=direction_key,
                get_scalping_entry_target_fn=get_scalping_entry_target,
                scalp_quality_gate_fn=scalp_quality_gate,
                sr_lookback_fn=_sr_lookback,
                timeframe=timeframe,
                forward_bars=forward_bars,
            )
            diag = {}
            try:
                diag = dict(getattr(events_sym, "attrs", {}).get("diagnostics", {}) or {})
            except Exception:
                diag = {}
            diag_bars_evaluated += int(diag.get("bars_evaluated", 0) or 0)
            diag_analysis_fail += int(diag.get("analysis_fail", 0) or 0)
            diag_signal_side_reject += int(diag.get("signal_side_reject", 0) or 0)
            diag_plan_fail += int(diag.get("plan_fail", 0) or 0)
            diag_gate_pass_candidates += int(diag.get("gate_pass_candidates", 0) or 0)
            diag_side_key_reject += int(diag.get("side_key_reject", 0) or 0)
            diag_price_level_reject += int(diag.get("price_level_reject", 0) or 0)
            diag_forward_window_reject += int(diag.get("forward_window_reject", 0) or 0)
            diag_gate_reject_counts.update(
                {
                    str(k): int(v)
                    for k, v in dict(diag.get("gate_reject_counts", {}) or {}).items()
                    if str(k).strip()
                }
            )
            diag_plan_fail_counts.update(
                {
                    str(k): int(v)
                    for k, v in dict(diag.get("plan_fail_counts", {}) or {}).items()
                    if str(k).strip()
                }
            )
            if events_sym is None or events_sym.empty:
                continue
            events_sym = events_sym.copy()
            events_sym["Pair"] = pair
            events_sym["Coin"] = pair.split("/", 1)[0]
            all_events.append(events_sym)
            symbols_with_events += 1
        except Exception:
            failed_symbols += 1
        finally:
            processed_symbols += 1
            progress_bar.progress(
                min(100, int((processed_symbols / max(1, len(pairs))) * 100)),
                text=f"Scanning {processed_symbols}/{len(pairs)} symbols...",
            )

    progress_bar.empty()

    if not all_events:
        st.warning(
            "No scalp-qualified events were found in this window. "
            "Try increasing lookback or using a faster timeframe."
        )
        if diag_gate_reject_counts:
            reject_parts = []
            for reason_code, count in diag_gate_reject_counts.most_common(4):
                reject_parts.append(f"{_gate_reason_label(reason_code)} {int(count)}")
            if reject_parts:
                st.caption(f"Top gate rejects: {' • '.join(reject_parts)}")
        if diag_plan_fail_counts:
            plan_parts = [f"{reason} {int(count)}" for reason, count in diag_plan_fail_counts.most_common(3)]
            if plan_parts:
                st.caption(f"Top plan failures: {' • '.join(plan_parts)}")
        if (
            no_data_symbols
            or failed_symbols
            or diag_bars_evaluated
            or diag_signal_side_reject
            or diag_plan_fail
            or diag_analysis_fail
        ):
            st.caption(
                f"Diagnostics: no-data symbols={no_data_symbols}, failed symbols={failed_symbols}, "
                f"cache hits={cache_hits}, bars evaluated={diag_bars_evaluated}, "
                f"signal-neutral rejects={diag_signal_side_reject}, plan-fail={diag_plan_fail}, "
                f"analysis-fail={diag_analysis_fail}, gate-pass candidates={diag_gate_pass_candidates}."
            )
        return

    events_df = pd.concat(all_events, ignore_index=True)
    events_df = events_df.sort_values("Event Time", ascending=False).reset_index(drop=True)

    summary = summarize_scalp_outcome_study(events_df, forward_bars)

    st.success(
        f"Study complete. {int(summary['occurrences'])} scalp-qualified events across "
        f"{symbols_with_events}/{len(pairs)} symbols."
    )
    if no_data_symbols or failed_symbols or cache_hits:
        st.caption(
            f"Universe diagnostics: cache hits={cache_hits}, no-data symbols={no_data_symbols}, "
            f"failed symbols={failed_symbols}, bars evaluated={diag_bars_evaluated}, "
            f"gate-pass candidates={diag_gate_pass_candidates}."
        )
    if truncated_by_time:
        st.warning(
            f"Runtime cap reached (~{int(runtime_budget_sec)}s). Showing partial results "
            f"from {processed_symbols}/{len(pairs)} scanned symbols."
        )

    ret_col = f"Realized Return @+{forward_bars} (%)"
    if ret_col not in events_df.columns:
        ret_col = f"Return @+{forward_bars} (%)"
    tp_rate = _safe_float(summary.get("tp_rate"), 0.0)
    sl_rate = _safe_float(summary.get("sl_rate"), 0.0)
    timeout_rate = _safe_float(summary.get("timeout_rate"), 0.0)
    avg_outcome = _safe_float(summary.get("avg_outcome"), 0.0)
    median_outcome = _safe_float(summary.get("median_outcome"), 0.0)
    avg_hit = _avg_hit_bar(events_df)

    _render_kpi_cards(
        [
            (
                "Universe Scanned",
                f"{len(pairs)}",
                "How many top-volume symbols were scanned in this run.",
            ),
            (
                "Symbols with Events",
                f"{symbols_with_events}",
                "How many symbols produced at least one scalp-qualified event.",
            ),
            (
                "Scalp Events",
                f"{int(summary['occurrences'])}",
                "How many scalp candidates passed the scalp quality gate.",
            ),
            (
                "TP First Rate",
                f"{tp_rate:.1f}%",
                "Percent of events where target was hit before stop inside the selected bar window.",
            ),
            (
                f"Expectancy @+{forward_bars}",
                f"{avg_outcome:+.2f}%",
                "Average realized directional outcome at the selected horizon for gate-passing scalp events.",
            ),
        ]
    )

    avg_hit_text = f"+{avg_hit:.1f} bars" if np.isfinite(avg_hit) else "N/A"
    median_text = f"{median_outcome:+.2f}%"
    st.caption(
        f"Scalp read: TP {tp_rate:.1f}% • SL {sl_rate:.1f}% • Timeout {timeout_rate:.1f}% • "
        f"Avg hit speed {avg_hit_text} • Median outcome {median_text}."
    )

    if int(summary["occurrences"]) < 30:
        st.warning("Sample size is small (<30 events). Treat this as exploratory, not definitive.")

    st.markdown("### Direction Scoreboard")
    dir_rows: list[dict] = []
    for dir_name, grp in events_df.groupby("Direction", dropna=False):
        dir_returns = pd.to_numeric(grp.get(ret_col, pd.Series(dtype=float)), errors="coerce")
        dir_outcome = grp.get("Outcome", pd.Series(dtype=object)).astype(str).str.upper()
        dir_hit = _avg_hit_bar(grp)
        dir_rr = pd.to_numeric(grp.get("R:R", pd.Series(dtype=float)), errors="coerce").mean()
        tp_pct = float((dir_outcome == "TP").mean() * 100.0) if len(grp) else 0.0
        sl_pct = float(dir_outcome.isin({"SL", "BOTH"}).mean() * 100.0) if len(grp) else 0.0
        to_pct = float((dir_outcome == "TIMEOUT").mean() * 100.0) if len(grp) else 0.0
        avg_ret = float(dir_returns.mean()) if not dir_returns.empty else float("nan")
        med_ret = float(dir_returns.median()) if not dir_returns.empty else float("nan")
        dir_rows.append(
            {
                "Direction": str(dir_name),
                "Events": int(len(grp)),
                "TP First": f"{tp_pct:.1f}%",
                "SL First": f"{sl_pct:.1f}%",
                "Timeout": f"{to_pct:.1f}%",
                "Avg Hit Bar": f"+{dir_hit:.1f}" if np.isfinite(dir_hit) else "N/A",
                "Avg R:R": f"{float(dir_rr):.2f}" if np.isfinite(dir_rr) else "N/A",
                "Avg Outcome": f"{avg_ret:+.2f}%" if np.isfinite(avg_ret) else "N/A",
                "Median Outcome": f"{med_ret:+.2f}%" if np.isfinite(med_ret) else "N/A",
                "TPPctNum": tp_pct,
                "AvgOutcomeNum": avg_ret if np.isfinite(avg_ret) else -999.0,
            }
        )

    dir_view = pd.DataFrame(dir_rows)
    if dir_view.empty:
        st.info("No direction-level summary available.")
    else:
        dir_view = dir_view.sort_values(
            by=["TPPctNum", "AvgOutcomeNum", "Events"],
            ascending=[False, False, False],
        ).reset_index(drop=True)
        dir_display = dir_view[
            [
                "Direction",
                "Events",
                "TP First",
                "SL First",
                "Timeout",
                "Avg Hit Bar",
                "Avg R:R",
                "Avg Outcome",
                "Median Outcome",
            ]
        ].copy()
        _render_hover_table(
            dir_display,
            {
                "Direction": "Scalp side of event (Upside/Downside).",
                "Events": "Number of scalp-qualified events in this direction.",
                "TP First": "Target hit before stop, within tracking window.",
                "SL First": "Stop hit before target, within tracking window.",
                "Timeout": "Neither TP nor SL hit before the tracking window ended.",
                "Avg Hit Bar": "Average bar index where outcome was resolved. Lower is faster.",
                "Avg R:R": "Average planned reward-to-risk of recorded scalp entries.",
                "Avg Outcome": "Average realized directional return at selected horizon.",
                "Median Outcome": "Median realized directional return at selected horizon.",
            },
            table_id="scalp-direction-summary",
        )

    st.markdown("### Coin Scoreboard")
    coin_rows: list[dict] = []
    for coin_name, grp in events_df.groupby("Coin", dropna=False):
        coin_returns = pd.to_numeric(grp.get(ret_col, pd.Series(dtype=float)), errors="coerce")
        coin_outcome = grp.get("Outcome", pd.Series(dtype=object)).astype(str).str.upper()
        tp_pct = float((coin_outcome == "TP").mean() * 100.0) if len(grp) else 0.0
        sl_pct = float(coin_outcome.isin({"SL", "BOTH"}).mean() * 100.0) if len(grp) else 0.0
        to_pct = float((coin_outcome == "TIMEOUT").mean() * 100.0) if len(grp) else 0.0
        avg_ret = float(coin_returns.mean()) if not coin_returns.empty else float("nan")
        med_ret = float(coin_returns.median()) if not coin_returns.empty else float("nan")
        avg_rr = pd.to_numeric(grp.get("R:R", pd.Series(dtype=float)), errors="coerce").mean()
        coin_rows.append(
            {
                "Coin": str(coin_name),
                "Events": int(len(grp)),
                "TP First": f"{tp_pct:.1f}%",
                "SL First": f"{sl_pct:.1f}%",
                "Timeout": f"{to_pct:.1f}%",
                "Avg R:R": f"{float(avg_rr):.2f}" if np.isfinite(avg_rr) else "N/A",
                "Avg Outcome": f"{avg_ret:+.2f}%" if np.isfinite(avg_ret) else "N/A",
                "Median Outcome": f"{med_ret:+.2f}%" if np.isfinite(med_ret) else "N/A",
                "TPPctNum": tp_pct,
                "AvgOutcomeNum": avg_ret if np.isfinite(avg_ret) else -999.0,
            }
        )
    coin_view = pd.DataFrame(coin_rows)
    if coin_view.empty:
        st.info("No coin-level summary available.")
    else:
        coin_view = coin_view.sort_values(
            by=["Events", "TPPctNum", "AvgOutcomeNum"],
            ascending=[False, False, False],
        ).reset_index(drop=True)
        _render_hover_table(
            coin_view[["Coin", "Events", "TP First", "SL First", "Timeout", "Avg R:R", "Avg Outcome", "Median Outcome"]],
            {
                "Coin": "Symbol base (quote is USDT/USD in this universe).",
                "Events": "Number of scalp-qualified events for this coin.",
                "TP First": "Target hit before stop.",
                "SL First": "Stop hit before target (includes BOTH).",
                "Timeout": "Neither TP nor SL hit inside tracking window.",
                "Avg R:R": "Average planned reward-to-risk across events.",
                "Avg Outcome": "Average realized directional return at selected horizon.",
                "Median Outcome": "Median realized directional return at selected horizon.",
            },
            table_id="scalp-coin-summary",
        )

    with st.expander("Advanced scalp diagnostics and event details", expanded=False):
        st.markdown("#### Outcome mix")
        outcome = events_df.get("Outcome", pd.Series(dtype=object)).astype(str).str.upper()
        total_n = max(1, len(events_df))
        outcome_df = pd.DataFrame(
            {
                "Outcome": ["TP", "SL/BOTH", "TIMEOUT"],
                "Count": [
                    int((outcome == "TP").sum()),
                    int(outcome.isin({"SL", "BOTH"}).sum()),
                    int((outcome == "TIMEOUT").sum()),
                ],
            }
        )
        outcome_df["Share"] = outcome_df["Count"].map(lambda n: f"{(float(n) / total_n) * 100.0:.1f}%")
        _render_hover_table(
            outcome_df,
            {
                "Outcome": "Scalp exit type.",
                "Count": "How many events ended with this outcome.",
                "Share": "Outcome share in current filtered sample.",
            },
            table_id="scalp-outcome-mix",
        )

        st.markdown(f"#### TP/SL Hit Timing (next {forward_bars} bars)")

        def _cum_rate(df_slice: pd.DataFrame, outcome_keys: set[str]) -> list[float]:
            n = len(df_slice)
            if n <= 0:
                return [float("nan")] * forward_bars
            hit = pd.to_numeric(df_slice.get("Hit Bar", pd.Series(dtype=float)), errors="coerce")
            out = df_slice.get("Outcome", pd.Series(dtype=object)).astype(str).str.upper()
            out_mask = out.isin(outcome_keys)
            vals: list[float] = []
            for step in range(1, forward_bars + 1):
                vals.append(float(((out_mask) & (hit <= step)).mean() * 100.0))
            return vals

        fig = go.Figure()
        x_axis = list(range(1, forward_bars + 1))
        st.caption("TP-first timing by scalp direction across the scanned universe.")
        dir_specs = [("Upside", POSITIVE, "solid"), ("Downside", "#FF3366", "dot")]
        plotted = 0
        for dir_name, dir_color, dir_dash in dir_specs:
            dir_df = events_df[events_df["Direction"].astype(str) == dir_name]
            if dir_df.empty:
                continue
            tp_path = _cum_rate(dir_df, {"TP"})
            tp_rate_now = float((dir_df["Outcome"].astype(str).str.upper() == "TP").mean() * 100.0)
            fig.add_trace(
                go.Scatter(
                    x=x_axis,
                    y=tp_path,
                    mode="lines+markers",
                    name=f"{dir_name} TP-first (n={len(dir_df)}, total {tp_rate_now:.1f}%)",
                    line=dict(color=dir_color, width=2, dash=dir_dash),
                    marker=dict(size=5),
                )
            )
            plotted += 1

        sl_path = _cum_rate(events_df, {"SL", "BOTH"})
        timeout_path = _cum_rate(events_df, {"TIMEOUT"})
        fig.add_trace(
            go.Scatter(
                x=x_axis,
                y=sl_path,
                mode="lines",
                name="SL-first (all)",
                line=dict(color="#FF3366", width=1.6, dash="dash"),
            )
        )
        fig.add_trace(
            go.Scatter(
                x=x_axis,
                y=timeout_path,
                mode="lines",
                name="Timeout (all)",
                line=dict(color=WARNING, width=1.4, dash="dot"),
            )
        )
        if plotted == 0:
            st.info("No direction path available for the selected sample.")

        fig.update_layout(
            template="plotly_dark",
            height=320,
            xaxis_title="Bars After Event",
            yaxis_title="Cumulative Event Share (%)",
            showlegend=True,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0.0),
            yaxis=dict(range=[0, 100]),
        )
        st.plotly_chart(fig, width="stretch")

        st.markdown("#### Event table")
        _render_event_table_html(events_df, forward_bars)

    csv_data = events_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download Scalp Outcome Study (CSV)",
        data=csv_data,
        file_name=f"scalp_universe_{timeframe}_n{len(pairs)}_lb{fetch_limit}_fwd{forward_bars}.csv",
        mime="text/csv",
    )
