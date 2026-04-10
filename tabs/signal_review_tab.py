from __future__ import annotations

import pandas as pd
from core.decision_version import current_decision_version
from core.session_utils import session_bucket_for_timestamp
from core.signal_tracker import prefer_current_decision_version_slice
from core.trading_copy import copy_text, playbook_display, playbook_key, trade_gate_display, trade_gate_key

from ui.ctx import get_ctx
from ui.signal_formatters import archived_execution_stance_label
from ui.primitives import render_insight_card, render_kpi_grid, render_page_header


_MIN_SIGNAL_ARCHIVE_ROWS = 8
_MIN_EXECUTION_ARCHIVE_ROWS = 3


def _display_trade_direction(value: object) -> str:
    side = str(value or "").strip().upper()
    if side in {"LONG", "UPSIDE", "BUY"}:
        return "Upside"
    if side in {"SHORT", "DOWNSIDE", "SELL"}:
        return "Downside"
    return ""


def _annotate_actual_hold_style(df_events: pd.DataFrame) -> pd.DataFrame:
    if df_events is None or df_events.empty:
        return pd.DataFrame()
    d = df_events.copy()
    if "actual_trade_status" not in d.columns:
        return d
    d["actual_trade_status"] = d["actual_trade_status"].fillna("").astype(str).str.upper()
    d["actual_entry_at"] = pd.to_datetime(d.get("actual_entry_at"), utc=True, errors="coerce")
    d["actual_exit_at"] = pd.to_datetime(d.get("actual_exit_at"), utc=True, errors="coerce")
    d["Actual Hold Hours"] = (
        (d["actual_exit_at"] - d["actual_entry_at"]).dt.total_seconds() / 3600.0
    )
    d["Actual Hold Hours"] = pd.to_numeric(d["Actual Hold Hours"], errors="coerce")

    def _style_for_row(row: pd.Series) -> str:
        status = str(row.get("actual_trade_status") or "").strip().upper()
        hold_hours = row.get("Actual Hold Hours")
        if status != "CLOSED":
            return "Open / Unjournaled"
        if pd.isna(hold_hours):
            return "Unknown Hold"
        if float(hold_hours) <= 6.0:
            return "Quick Follow-Through"
        if float(hold_hours) >= 18.0:
            return "Needs Room"
        return "Standard Hold"

    d["Hold Style"] = d.apply(_style_for_row, axis=1)
    return d


def _annotate_actual_exit_quality(df_events: pd.DataFrame) -> pd.DataFrame:
    if df_events is None or df_events.empty:
        return pd.DataFrame()
    d = df_events.copy()
    if "actual_trade_status" not in d.columns:
        return d
    d["actual_trade_status"] = d["actual_trade_status"].fillna("").astype(str).str.upper()
    d["actual_pnl_pct"] = pd.to_numeric(d.get("actual_pnl_pct"), errors="coerce")
    d["actual_exit_reason"] = d.get("actual_exit_reason", pd.Series(dtype=object)).fillna("").astype(str).str.strip().str.upper()

    def _quality_for_row(row: pd.Series) -> str:
        status = str(row.get("actual_trade_status") or "").strip().upper()
        if status != "CLOSED":
            return "Open / Unjournaled"
        pnl = row.get("actual_pnl_pct")
        if pd.isna(pnl):
            return "Unknown Exit"
        reason = str(row.get("actual_exit_reason") or "").strip().upper()
        if float(pnl) > 0.0:
            if reason == "TARGET":
                return "Target Winner"
            if reason in {"MANUAL EXIT", "TIME EXIT"}:
                return "Manual Winner Exit"
            return "Winner Other Exit"
        if reason in {"STOP", "INVALIDATION"}:
            return "Protected Loss Exit"
        if reason in {"MANUAL EXIT", "TIME EXIT"}:
            return "Late Manual Loss"
        return "Loss Other Exit"

    d["Exit Quality"] = d.apply(_quality_for_row, axis=1)
    return d


def _review_scope_note(
    status_filter: str,
    timeframe_filter: str,
    limit: int,
    rows_in_slice: int,
    version_filter: str = "All",
) -> str:
    status_text = "all statuses" if str(status_filter) == "All" else f"{str(status_filter).lower()} signals"
    timeframe_text = "all timeframes" if str(timeframe_filter) == "All" else f"{str(timeframe_filter).upper()} only"
    version_text = "all decision versions" if str(version_filter) == "All" else str(version_filter)
    return (
        f"Current review slice: {rows_in_slice} rows shown from the latest up to {int(limit)} Market signals, "
        f"filtered to {status_text}, {timeframe_text}, and {version_text}. Archive cards and cohorts below are computed from this slice, "
        "not the full tracker history."
    )


def _decision_cohort_note(*, mode: str, target_version: str, current_rows: int, total_rows: int) -> tuple[str, str]:
    target_label = str(target_version or "Current Version").strip() or "Current Version"
    if mode == "current_only":
        return (
            (
                f"<b>{target_label}</b> now has enough resolved archive depth to drive adaptive calibration directly "
                f"({int(current_rows)} resolved rows, out of {int(total_rows)} recent resolved rows loaded)."
            ),
            "positive",
        )
    if mode == "mixed_fallback":
        return (
            (
                f"<b>{target_label}</b> has started building ({int(current_rows)} resolved rows), but adaptive calibration "
                f"is still falling back to the mixed archive until the current-version cohort is deep enough. "
                f"Current loaded resolved pool: {int(total_rows)} rows."
            ),
            "warning",
        )
    if mode == "unversioned_fallback":
        return (
            (
                "This review slice still includes legacy or unversioned archive rows, so adaptive calibration cannot isolate "
                "the current scanner logic yet."
            ),
            "warning",
        )
    if mode == "empty":
        return (
            "No resolved archive rows are available yet for decision-version calibration.",
            "neutral",
        )
    return (
        (
            f"Adaptive calibration is currently reading a mixed archive slice. Current version: <b>{target_label}</b>. "
            f"Resolved current rows: {int(current_rows)} of {int(total_rows)} loaded."
        ),
        "neutral",
    )


def _qualified_summary_rows(
    summary_df: pd.DataFrame,
    *,
    count_field: str,
    min_count: int,
) -> pd.DataFrame:
    if summary_df is None or summary_df.empty or count_field not in summary_df.columns:
        return pd.DataFrame()
    d = summary_df.copy()
    d[count_field] = pd.to_numeric(d[count_field], errors="coerce").fillna(0.0)
    return d[d[count_field] >= float(min_count)].copy()


def _prefer_known_summary_rows(summary_df: pd.DataFrame, *, label_field: str) -> pd.DataFrame:
    if summary_df is None or summary_df.empty or label_field not in summary_df.columns:
        return pd.DataFrame()
    d = summary_df.copy()
    labels = d[label_field].fillna("").astype(str).str.strip()
    known_mask = ~labels.str.contains(r"\bUnknown\b", case=False, na=False)
    if bool(known_mask.any()):
        return d.loc[known_mask].copy()
    return pd.DataFrame()


def _archive_building_card(title: str, body_html: str) -> dict[str, str]:
    return {
        "title": title,
        "body_html": body_html,
        "tone": "neutral",
        "kind": "building",
    }


def _prepare_section_cards(
    cards: list[dict[str, str]],
    *,
    max_actionable: int = 3,
) -> list[dict[str, str]]:
    visible_cards = [card for card in list(cards or []) if card]
    if not visible_cards:
        return []
    actionable = [card for card in visible_cards if str(card.get("kind") or "").strip().lower() != "building"]
    building = [card for card in visible_cards if str(card.get("kind") or "").strip().lower() == "building"]
    prepared = actionable[: max(1, int(max_actionable))]
    if building:
        titles = [str(card.get("title") or "").strip() for card in building if str(card.get("title") or "").strip()]
        preview = ", ".join(titles[:3])
        extra = "" if len(titles) <= 3 else f" +{len(titles) - 3} more"
        prepared.append(
            {
                "title": "Archive Status",
                "body_html": (
                    f"Still building: <b>{preview}</b>{extra}. "
                    "These need more resolved signals or journaled closed trades before they become trustworthy."
                ),
                "tone": "neutral",
            }
        )
    return prepared if prepared else visible_cards[:1]


def _execution_vs_system_note(execution_snapshot: dict[str, float]) -> tuple[str, str]:
    taken = float(execution_snapshot.get("taken", 0.0) or 0.0)
    taken_resolved = float(execution_snapshot.get("taken_resolved", 0.0) or 0.0)
    actual_closed = float(execution_snapshot.get("actual_closed", 0.0) or 0.0)
    if taken <= 0.0 and actual_closed <= 0.0:
        return (
            "Execution journal is still building. The system archive is live, but you have not journaled any taken trades yet.",
            "neutral",
        )
    if taken_resolved < _MIN_EXECUTION_ARCHIVE_ROWS or actual_closed < _MIN_EXECUTION_ARCHIVE_ROWS:
        return (
            "Execution archive is still thin. Use this section directionally for now, but wait for more journaled trades before trusting hard conclusions.",
            "neutral",
        )
    return (
        (
            f"Taken setups had <b>{execution_snapshot['taken_follow_through_rate']:.1f}%</b> signal follow-through, "
            f"while closed real trades finished with <b>{execution_snapshot['actual_win_rate']:.1f}%</b> win rate. "
            f"Execution gap is <b>{execution_snapshot['execution_gap_pct']:+.2f}%</b>. "
            f"Skipped winners: <b>{int(execution_snapshot['skipped_winners'])}</b>."
        ),
        "positive" if float(execution_snapshot.get("execution_gap_pct", 0.0) or 0.0) >= 0.0 else "warning",
    )


def _render_insight_card_grid(st, cards: list[dict[str, str]], *, columns: int = 3) -> None:
    visible_cards = [card for card in cards if card]
    if not visible_cards:
        return
    cols = st.columns(columns, gap="medium")
    for idx, card in enumerate(visible_cards):
        with cols[idx % columns]:
            render_insight_card(
                st,
                title=str(card.get("title") or ""),
                body_html=str(card.get("body_html") or ""),
                tone=str(card.get("tone") or "neutral"),
            )


def _render_compact_cohort_tables(
    st,
    *,
    df_events: pd.DataFrame,
    build_signal_cohort_summary,
    specs: list[tuple[str, str]],
) -> None:
    visible_specs: list[tuple[str, str, pd.DataFrame]] = []
    for group_field, title in specs:
        if group_field not in df_events.columns:
            continue
        summary_df = build_signal_cohort_summary(df_events, group_field)
        if summary_df is None or summary_df.empty:
            continue
        known_summary_df = _prefer_known_summary_rows(summary_df, label_field=group_field)
        if known_summary_df is not None and not known_summary_df.empty:
            summary_df = known_summary_df
        visible_specs.append((group_field, title, summary_df))
    if not visible_specs:
        st.caption("No cohort data is available in this slice yet.")
        return
    cols = st.columns(2, gap="medium")
    for idx, (_, title, summary_df) in enumerate(visible_specs):
        with cols[idx % 2]:
            st.markdown(f"##### {title}")
            st.dataframe(summary_df.round(2), hide_index=True, width="stretch")


def _render_execution_review_section(
    *,
    st,
    df_events: pd.DataFrame,
    db_path: str,
    save_signal_trade_overlay,
    save_signal_trade_journal,
    positive_color: str,
) -> None:
    st.markdown("### Execution Review")
    st.caption("Use this section after the top-level read. It separates system quality from your own execution decisions.")
    review_cols = st.columns([1.2, 1.0], gap="medium")
    trade_overlay_count = int(df_events.get("trade_decision", pd.Series(dtype=object)).fillna("").astype(str).str.strip().ne("").sum())
    taken_count = int(
        df_events.get("trade_decision", pd.Series(dtype=object))
        .fillna("")
        .astype(str)
        .str.upper()
        .eq("TAKEN")
        .sum()
    )
    closed_trade_count = int(
        df_events.get("actual_trade_status", pd.Series(dtype=object))
        .fillna("")
        .astype(str)
        .str.upper()
        .eq("CLOSED")
        .sum()
    )
    with review_cols[1]:
        st.markdown(
            (
                f"<div class='market-note-box' style='border:1px solid rgba(0,212,255,0.26); border-left:4px solid {positive_color};"
                " background:rgba(0,212,255,0.04);'>"
                f"<b style='color:{positive_color};'>Execution Overlay:</b> "
                f"{trade_overlay_count} signals have manual execution tags. "
                f"{taken_count} were actually taken, and {closed_trade_count} now have a full execution journal."
                "</div>"
            ),
            unsafe_allow_html=True,
        )
    signal_options: dict[str, str] = {}
    for _, row in df_events.iterrows():
        signal_key = str(row.get("signal_key") or "").strip()
        if not signal_key:
            continue
        event_time = pd.to_datetime(row.get("event_time"), errors="coerce")
        ts_label = event_time.strftime("%Y-%m-%d %H:%M") if pd.notna(event_time) else "Unknown time"
        label = (
            f"{ts_label} • {str(row.get('symbol') or '')} • "
            f"{str(row.get('timeframe') or '')} • {str(row.get('setup_confirm') or '')}"
        )
        signal_options[label] = signal_key

    with review_cols[0]:
        if signal_options:
            with st.expander("Journal a tracked setup", expanded=False):
                selected_label = st.selectbox(
                    "Trade overlay",
                    list(signal_options.keys()),
                    index=0,
                    key="signal_review_trade_overlay_pick",
                    help="Use this to mark whether you actually took a tracked setup or passed on it.",
                )
                selected_key = signal_options[selected_label]
                selected_row = df_events[df_events["signal_key"].astype(str) == selected_key].iloc[0]
                current_decision = str(selected_row.get("trade_decision") or "").strip()
                current_note = str(selected_row.get("trade_note") or "").strip()
                current_side = str(selected_row.get("actual_trade_side") or "").strip().upper()
                current_entry_price = selected_row.get("actual_entry_price")
                current_entry_at = str(selected_row.get("actual_entry_at") or "").strip()
                current_exit_price = selected_row.get("actual_exit_price")
                current_exit_at = str(selected_row.get("actual_exit_at") or "").strip()
                current_exit_reason = str(selected_row.get("actual_exit_reason") or "").strip()
                current_trade_status = str(selected_row.get("actual_trade_status") or "").strip().upper()
                decision_options = ["Taken", "Skipped", "Observed", "Clear"]
                default_idx = decision_options.index(current_decision) if current_decision in decision_options else 0
                with st.form("signal_trade_overlay_form", clear_on_submit=False):
                    chosen_decision = st.selectbox(
                        "Decision",
                        decision_options,
                        index=default_idx,
                        key="signal_trade_overlay_decision",
                    )
                    trade_note = st.text_input(
                        "Note",
                        value=current_note,
                        key="signal_trade_overlay_note",
                        placeholder="Optional execution note",
                    )
                    submitted = st.form_submit_button("Save overlay", use_container_width=False)
                if submitted:
                    decision_value = "" if chosen_decision == "Clear" else chosen_decision
                    saved = save_signal_trade_overlay(
                        selected_key,
                        trade_decision=decision_value,
                        trade_note=trade_note,
                        db_path=db_path,
                    )
                    if saved:
                        st.success("Trade overlay saved.")
                        st.rerun()
                    else:
                        st.error("Trade overlay could not be saved for that signal.")

                has_journal = any(
                    [
                        current_side,
                        current_entry_at,
                        current_exit_at,
                        pd.notna(current_entry_price),
                        pd.notna(current_exit_price),
                        current_exit_reason,
                    ]
                )
                if current_decision == "Taken" or has_journal:
                    signal_direction = str(selected_row.get("direction") or "").strip().upper()
                    suggested_side = "Upside" if signal_direction == "UPSIDE" else "Downside"
                    side_options = ["Upside", "Downside"]
                    default_side = _display_trade_direction(current_side) or suggested_side
                    default_side_idx = side_options.index(default_side)
                    exit_reason_options = ["Open", "Target", "Stop", "Manual Exit", "Time Exit", "Invalidation", "Clear"]
                    mapped_exit_reason = current_exit_reason if current_exit_reason in exit_reason_options else ("Open" if current_trade_status != "CLOSED" else "Manual Exit")
                    with st.form("signal_trade_journal_form", clear_on_submit=False):
                        st.markdown("#### Actual Trade Journal")
                        st.caption("Use this only for trades you really took. It keeps system signal quality separate from your actual execution.")
                        chosen_side = st.selectbox(
                            "Trade direction",
                            side_options,
                            index=default_side_idx,
                            key="signal_trade_journal_side",
                        )
                        entry_price_text = st.text_input(
                            "Entry price",
                            value="" if pd.isna(current_entry_price) else f"{float(current_entry_price):.8f}".rstrip("0").rstrip("."),
                            key="signal_trade_journal_entry_price",
                            placeholder="Example: 102.45",
                        )
                        entry_time_text = st.text_input(
                            "Entry time (UTC)",
                            value=current_entry_at,
                            key="signal_trade_journal_entry_at",
                            placeholder="2026-04-04T12:00:00Z",
                        )
                        exit_price_text = st.text_input(
                            "Exit price",
                            value="" if pd.isna(current_exit_price) else f"{float(current_exit_price):.8f}".rstrip("0").rstrip("."),
                            key="signal_trade_journal_exit_price",
                            placeholder="Leave blank if still open",
                        )
                        exit_time_text = st.text_input(
                            "Exit time (UTC)",
                            value=current_exit_at,
                            key="signal_trade_journal_exit_at",
                            placeholder="Leave blank if still open",
                        )
                        chosen_exit_reason = st.selectbox(
                            "Exit reason",
                            exit_reason_options,
                            index=exit_reason_options.index(mapped_exit_reason),
                            key="signal_trade_journal_exit_reason",
                        )
                        journal_submitted = st.form_submit_button("Save journal", use_container_width=False)
                    if journal_submitted:
                        if chosen_exit_reason == "Clear":
                            journal_saved = save_signal_trade_journal(selected_key, db_path=db_path)
                        else:
                            journal_saved = save_signal_trade_journal(
                                selected_key,
                                actual_trade_side=chosen_side,
                                actual_entry_price=entry_price_text,
                                actual_entry_at=entry_time_text,
                                actual_exit_price="" if chosen_exit_reason == "Open" else exit_price_text,
                                actual_exit_at="" if chosen_exit_reason == "Open" else exit_time_text,
                                actual_exit_reason="" if chosen_exit_reason == "Open" else chosen_exit_reason,
                                db_path=db_path,
                            )
                        if journal_saved:
                            st.success("Actual trade journal saved.")
                            st.rerun()
                        else:
                            st.error("Trade journal could not be saved. Entry price and trade direction are required.")
        else:
            st.caption("No tracked signals are available in this slice yet for journaling.")


def _render_tracker_backup_restore(
    *,
    st,
    db_path: str,
    storage_snapshot,
    read_signal_tracker_db_bytes,
    backup_signal_tracker_db,
    restore_signal_tracker_db_bytes,
    fetch_signal_events_df,
    fetch_market_alerts_df,
) -> None:
    with st.expander("Storage, Backup & Restore", expanded=False):
        render_insight_card(
            st,
            title="Tracker Storage",
            body_html=(
                f"<b>{storage_snapshot.label}</b><br>"
                f"{storage_snapshot.note}<br><br>"
                f"<span style='color:#8B949E;'>Durability:</span> {storage_snapshot.durability_label}<br>"
                f"<span style='color:#8B949E;'>Durability note:</span> {storage_snapshot.durability_note}<br>"
                f"<span style='color:#8B949E;'>Archive:</span> {int(storage_snapshot.size_bytes):,} bytes<br>"
                f"<span style='color:#8B949E;'>Recovery:</span> {storage_snapshot.recovery_status}<br>"
                f"<span style='color:#8B949E;'>Recovery note:</span> {storage_snapshot.recovery_note}<br>"
                f"<span style='color:#8B949E;'>Mirror rail:</span> "
                f"{storage_snapshot.mirror_dir if storage_snapshot.mirror_enabled else 'Not configured'}<br>"
                f"<span style='color:#8B949E;'>Mirror snapshots:</span> {int(storage_snapshot.mirror_count):,}<br>"
                f"<span style='color:#8B949E;'>Path:</span> {storage_snapshot.path}"
            ),
            tone=str(storage_snapshot.tone or "neutral"),
        )
        st.caption(
            "Use this area to protect the learning archive before risky changes, or move it cleanly between machines."
        )
        if st.button("Create Local Restore Point", key="signal_review_local_restore_point"):
            backup_path = str(backup_signal_tracker_db(db_path) or "").strip()
            if backup_path:
                st.session_state["signal_review_tracker_notice"] = f"Local restore point created at {backup_path}"
                st.session_state["signal_review_tracker_notice_tone"] = "success"
                st.rerun()
            st.session_state["signal_review_tracker_notice"] = "Tracker DB does not exist yet, so there is nothing to back up."
            st.session_state["signal_review_tracker_notice_tone"] = "warning"
            st.rerun()
        db_bytes = read_signal_tracker_db_bytes(db_path)
        st.download_button(
            "Download Tracker DB",
            data=db_bytes,
            file_name=str(storage_snapshot.filename or "signal_tracker.sqlite3"),
            mime="application/x-sqlite3",
            disabled=not bool(db_bytes),
            on_click="ignore",
        )
        full_events_csv = fetch_signal_events_df(limit=100000, source="Market", db_path=db_path)
        full_alerts_csv = fetch_market_alerts_df(limit=5000, source="Market", db_path=db_path)
        st.download_button(
            "Download Signal Events CSV",
            data=full_events_csv.to_csv(index=False).encode("utf-8"),
            file_name="signal_events_backup.csv",
            mime="text/csv",
            on_click="ignore",
        )
        st.download_button(
            "Download Market Alerts CSV",
            data=full_alerts_csv.to_csv(index=False).encode("utf-8"),
            file_name="market_alerts_backup.csv",
            mime="text/csv",
            on_click="ignore",
        )
        st.markdown("---")
        st.caption(
            "Restore replaces the current tracker DB after first creating a local backup copy. Only upload tracker snapshots created from this dashboard."
        )
        uploaded_tracker_db = st.file_uploader(
            "Upload tracker snapshot",
            type=["sqlite3", "db", "sqlite"],
            key="signal_review_tracker_restore_upload",
            help="Accepted files are SQLite tracker snapshots previously exported from Signal Review.",
        )
        if st.button("Restore Uploaded Tracker DB", key="signal_review_restore_uploaded_db"):
            if uploaded_tracker_db is None:
                st.session_state["signal_review_tracker_notice"] = "Choose a tracker snapshot file before running restore."
                st.session_state["signal_review_tracker_notice_tone"] = "warning"
                st.rerun()
            try:
                restore_result = restore_signal_tracker_db_bytes(
                    uploaded_tracker_db.getvalue(),
                    db_path=db_path,
                    backup_existing=True,
                )
            except Exception as exc:
                st.session_state["signal_review_tracker_notice"] = f"Restore failed: {exc}"
                st.session_state["signal_review_tracker_notice_tone"] = "warning"
                st.rerun()
            backup_msg = (
                f" Previous DB backed up to {restore_result.backup_path}."
                if str(restore_result.backup_path or "").strip()
                else ""
            )
            st.session_state["signal_review_tracker_notice"] = (
                f"Tracker DB restored to {restore_result.path} ({int(restore_result.restored_size):,} bytes).{backup_msg}"
            )
            st.session_state["signal_review_tracker_notice_tone"] = "success"
            st.rerun()


def render(ctx: dict) -> None:
    st = get_ctx(ctx, "st")
    POSITIVE = get_ctx(ctx, "POSITIVE")
    NEGATIVE = get_ctx(ctx, "NEGATIVE")
    WARNING = get_ctx(ctx, "WARNING")
    TEXT_MUTED = get_ctx(ctx, "TEXT_MUTED")
    fetch_ohlcv = get_ctx(ctx, "fetch_ohlcv")
    get_signal_tracker_db_path = get_ctx(ctx, "get_signal_tracker_db_path")
    init_signal_tracker_db = get_ctx(ctx, "init_signal_tracker_db")
    fetch_market_alerts_df = get_ctx(ctx, "fetch_market_alerts_df")
    resolve_open_signal_events_via_fetch = get_ctx(ctx, "resolve_open_signal_events_via_fetch")
    fetch_signal_events_df = get_ctx(ctx, "fetch_signal_events_df")
    save_signal_trade_overlay = get_ctx(ctx, "save_signal_trade_overlay")
    save_signal_trade_journal = get_ctx(ctx, "save_signal_trade_journal")
    build_signal_review_snapshot = get_ctx(ctx, "build_signal_review_snapshot")
    build_execution_overlay_snapshot = get_ctx(ctx, "build_execution_overlay_snapshot")
    build_signal_cohort_summary = get_ctx(ctx, "build_signal_cohort_summary")
    annotate_alert_footprint = get_ctx(ctx, "annotate_alert_footprint")
    build_alert_effectiveness_summary = get_ctx(ctx, "build_alert_effectiveness_summary")
    build_adaptive_context_model = get_ctx(ctx, "build_adaptive_context_model")
    build_learning_edge_table = get_ctx(ctx, "build_learning_edge_table")
    build_tracker_storage_snapshot = get_ctx(ctx, "build_tracker_storage_snapshot")
    read_signal_tracker_db_bytes = get_ctx(ctx, "read_signal_tracker_db_bytes")
    backup_signal_tracker_db = get_ctx(ctx, "backup_signal_tracker_db")
    restore_signal_tracker_db_bytes = get_ctx(ctx, "restore_signal_tracker_db_bytes")

    db_path = init_signal_tracker_db(get_signal_tracker_db_path())
    storage_snapshot = build_tracker_storage_snapshot(db_path)

    render_page_header(
        st,
        title="Signal Review",
        intro_html=(
            "This is the review and calibration page. Use it to learn what is actually working, what is failing, "
            "and whether the issue is the system or your execution, not as a live entry screen."
        ),
    )

    with st.spinner("Refreshing recent signal outcomes..."):
        resolved_now = int(
            resolve_open_signal_events_via_fetch(
                fetch_ohlcv=fetch_ohlcv,
                source="Market",
                db_path=db_path,
                limit_pairs=12,
                candle_limit=260,
            )
        )

    status_filter = st.selectbox("Status", ["All", "Open", "Resolved"], index=0, key="signal_review_status")
    timeframe_filter = st.selectbox("Timeframe", ["All", "15m", "1h", "4h", "1d"], index=0, key="signal_review_tf")
    limit = st.slider("Rows in Review Slice", 50, 500, 200, 25, key="signal_review_limit")

    df_events = fetch_signal_events_df(
        limit=int(limit),
        status=None if status_filter == "All" else status_filter.upper(),
        source="Market",
        timeframe=None if timeframe_filter == "All" else timeframe_filter,
        db_path=db_path,
    )
    current_market_version = current_decision_version("Market")
    version_series = (
        df_events.get("decision_version", pd.Series(index=df_events.index, dtype=object))
        .fillna("")
        .astype(str)
        .str.strip()
        .replace("", "Legacy / Unversioned")
    )
    explicit_versions = [
        value
        for value in version_series.drop_duplicates().tolist()
        if str(value).strip() and str(value).strip() != current_market_version
    ]
    version_options = ["All", "Current Version Only", "Legacy / Unversioned", *explicit_versions]
    deduped_version_options: list[str] = []
    for option in version_options:
        if option not in deduped_version_options:
            deduped_version_options.append(option)
    version_filter = st.selectbox(
        "Decision Version",
        deduped_version_options,
        index=0,
        key="signal_review_decision_version",
    )
    if version_filter == "Current Version Only":
        df_events = df_events.loc[version_series == current_market_version].copy()
    elif version_filter == "Legacy / Unversioned":
        df_events = df_events.loc[version_series == "Legacy / Unversioned"].copy()
    elif version_filter != "All":
        df_events = df_events.loc[version_series == version_filter].copy()
    df_alerts = fetch_market_alerts_df(limit=100, source="Market", db_path=db_path)
    df_active_alerts = fetch_market_alerts_df(limit=25, active_only=True, source="Market", db_path=db_path)
    adaptive_archive_df = fetch_signal_events_df(
        limit=2000,
        status="RESOLVED",
        source="Market",
        db_path=db_path,
    )
    adaptive_archive_df = prefer_current_decision_version_slice(
        adaptive_archive_df,
        source="Market",
    )
    adaptive_mode = str(adaptive_archive_df.attrs.get("decision_version_mode") or "mixed_fallback")
    adaptive_target = str(adaptive_archive_df.attrs.get("decision_version_target") or current_market_version)
    adaptive_rows = int(adaptive_archive_df.attrs.get("decision_version_rows") or 0)
    adaptive_total_rows = int(adaptive_archive_df.attrs.get("decision_version_total_rows") or len(adaptive_archive_df))
    adaptive_note, adaptive_tone = _decision_cohort_note(
        mode=adaptive_mode,
        target_version=adaptive_target,
        current_rows=adaptive_rows,
        total_rows=adaptive_total_rows,
    )

    top_insight_cols = st.columns(3, gap="medium")
    with top_insight_cols[0]:
        render_insight_card(
            st,
            title="Review Scope",
            body_html=_review_scope_note(
                status_filter=status_filter,
                timeframe_filter=timeframe_filter,
                version_filter=version_filter,
                limit=int(limit),
                rows_in_slice=int(len(df_events)),
            ),
            tone="neutral",
        )
    with top_insight_cols[1]:
        render_insight_card(
            st,
            title="Tracker Memory",
            body_html=(
                f"<b>{storage_snapshot.durability_label}</b><br>"
                f"{storage_snapshot.durability_note}<br><br>"
                f"<span style='color:#8B949E;'>DB:</span> {storage_snapshot.filename}<br>"
                f"<span style='color:#8B949E;'>Recovery:</span> {storage_snapshot.recovery_status}"
            ),
            tone=str(storage_snapshot.durability_tone or "neutral"),
        )
    with top_insight_cols[2]:
        render_insight_card(
            st,
            title="Calibration Cohort",
            body_html=adaptive_note,
            tone=adaptive_tone,
        )
    tracker_notice = st.session_state.pop("signal_review_tracker_notice", None)
    tracker_notice_tone = str(st.session_state.pop("signal_review_tracker_notice_tone", "info") or "info")
    if tracker_notice:
        if tracker_notice_tone == "success":
            st.success(str(tracker_notice))
        elif tracker_notice_tone == "warning":
            st.warning(str(tracker_notice))
        else:
            st.info(str(tracker_notice))

    snapshot = build_signal_review_snapshot(df_events)
    execution_snapshot = build_execution_overlay_snapshot(df_events)
    taken_count = int(snapshot["taken"])
    actual_closed = int(snapshot["actual_closed"])
    tone_follow = POSITIVE if snapshot["follow_through_rate"] >= 55.0 else (WARNING if snapshot["follow_through_rate"] >= 45.0 else NEGATIVE)
    tone_dir = POSITIVE if snapshot["avg_dir_return"] >= 0.0 else NEGATIVE
    st.markdown("### Overview")
    st.caption("Start here. This is the quick health read for the signal engine and your real execution.")
    render_kpi_grid(
        st,
        items=[
            {
                "label": "Follow-Through",
                "value": f"{snapshot['follow_through_rate']:.1f}%",
                "value_color": tone_follow,
                "subtext": "Resolved signals finishing in their intended direction",
            },
            {
                "label": "Avg Dir Return",
                "value": f"{snapshot['avg_dir_return']:+.2f}%",
                "value_color": tone_dir,
                "subtext": "Directional move after the signal horizon",
            },
            {
                "label": "Logged Signals",
                "value": int(snapshot["total"]),
                "subtext": f"DB: {db_path.split('/')[-1]}",
            },
            {
                "label": "Resolved",
                "value": int(snapshot["resolved"]),
                "subtext": f"Open: {int(snapshot['open'])}",
                "badge_text": f"+{resolved_now} refreshed" if resolved_now > 0 else "Up to date",
                "badge_color": POSITIVE if resolved_now > 0 else TEXT_MUTED,
                "badge_tone": "positive" if resolved_now > 0 else "neutral",
            },
        ],
        columns=4,
    )
    render_kpi_grid(
        st,
        items=[
            {
                "label": "Taken Trades",
                "value": taken_count,
                "value_color": POSITIVE if taken_count > 0 else TEXT_MUTED,
                "subtext": "Signals you actually marked as taken",
            },
            {
                "label": "Journal Closed",
                "value": actual_closed,
                "value_color": POSITIVE if actual_closed > 0 else TEXT_MUTED,
                "subtext": "Taken trades with a recorded exit",
            },
            {
                "label": "Avg MAE",
                "value": f"{snapshot['avg_adverse_excursion']:.2f}%",
                "value_color": NEGATIVE,
                "subtext": "Average adverse excursion after signal",
            },
            {
                "label": "Active Alerts",
                "value": int(len(df_active_alerts)),
                "value_color": WARNING if len(df_active_alerts) else TEXT_MUTED,
                "subtext": "Live market alerts still active now",
            },
        ],
        columns=4,
    )
    execution_vs_system_note, execution_vs_system_tone = _execution_vs_system_note(execution_snapshot)
    render_insight_card(
        st,
        title="Execution vs System",
        body_html=execution_vs_system_note,
        tone=execution_vs_system_tone,
    )

    if df_events.empty:
        st.info("No tracked signals yet. Open Market and let the scanner generate signals first.")
        _render_tracker_backup_restore(
            st=st,
            db_path=db_path,
            storage_snapshot=storage_snapshot,
            read_signal_tracker_db_bytes=read_signal_tracker_db_bytes,
            backup_signal_tracker_db=backup_signal_tracker_db,
            restore_signal_tracker_db_bytes=restore_signal_tracker_db_bytes,
            fetch_signal_events_df=fetch_signal_events_df,
            fetch_market_alerts_df=fetch_market_alerts_df,
        )
        return

    adaptive_model = build_adaptive_context_model(df_events)
    learned_edges_df = build_learning_edge_table(adaptive_model, limit=12)

    if "lead_active" in df_events.columns:
        df_events["Lead"] = df_events["lead_active"].fillna(0).astype(int).map({1: "LEAD", 0: "No LEAD"})
    if "ai_aligned" in df_events.columns:
        df_events["AI Alignment"] = df_events["ai_aligned"].fillna(0).astype(int).map({1: "Aligned", 0: "Not aligned"})
    if "market_lead_label" in df_events.columns:
        df_events["Market Lead"] = df_events["market_lead_label"].replace("", "No Clear Lead").fillna("No Clear Lead")
    if "market_regime" in df_events.columns:
        df_events["Market Regime"] = df_events["market_regime"].replace("", "Unknown").fillna("Unknown")
    if "scan_focus" in df_events.columns:
        df_events["Scan Focus"] = df_events["scan_focus"].replace("", "Unknown").fillna("Unknown")
    if "decision_version" in df_events.columns:
        df_events["Decision Version"] = (
            df_events["decision_version"].replace("", "Legacy / Unversioned").fillna("Legacy / Unversioned")
        )
    if "created_decision_version" in df_events.columns:
        df_events["Created Decision Version"] = (
            df_events["created_decision_version"].replace("", "Legacy / Unversioned").fillna("Legacy / Unversioned")
        )
    if "market_playbook_key" in df_events.columns or "market_playbook" in df_events.columns:
        playbook_keys = df_events.get("market_playbook_key", pd.Series(index=df_events.index, dtype=object))
        playbook_display_values = df_events.get("market_playbook", pd.Series(index=df_events.index, dtype=object))
        df_events["Playbook"] = pd.Series(playbook_keys, index=df_events.index).fillna("").astype(str).str.strip().map(
            lambda value: playbook_display(value) if value else ""
        )
        fallback_playbook = pd.Series(playbook_display_values, index=df_events.index).fillna("").astype(str).str.strip()
        df_events["Playbook"] = df_events["Playbook"].where(df_events["Playbook"].ne(""), fallback_playbook)
        df_events["Playbook"] = df_events["Playbook"].replace("", "Unknown").fillna("Unknown")
        df_events["Playbook Key"] = pd.Series(playbook_keys, index=df_events.index).fillna("").astype(str).str.strip()
        df_events["Playbook Key"] = df_events["Playbook Key"].where(
            df_events["Playbook Key"].ne(""),
            fallback_playbook.map(playbook_key),
        )
        df_events["Playbook Key"] = df_events["Playbook Key"].replace("", "Unknown").fillna("Unknown")
    if "market_trade_gate_key" in df_events.columns or "market_trade_gate" in df_events.columns:
        trade_gate_keys = df_events.get("market_trade_gate_key", pd.Series(index=df_events.index, dtype=object))
        trade_gate_display_values = df_events.get("market_trade_gate", pd.Series(index=df_events.index, dtype=object))
        df_events["Trade Gate"] = pd.Series(trade_gate_keys, index=df_events.index).fillna("").astype(str).str.strip().map(
            lambda value: trade_gate_display(value) if value else ""
        )
        fallback_trade_gate = pd.Series(trade_gate_display_values, index=df_events.index).fillna("").astype(str).str.strip()
        df_events["Trade Gate"] = df_events["Trade Gate"].where(df_events["Trade Gate"].ne(""), fallback_trade_gate)
        df_events["Trade Gate"] = df_events["Trade Gate"].replace("", "Unknown").fillna("Unknown")
        df_events["Trade Gate Key"] = pd.Series(trade_gate_keys, index=df_events.index).fillna("").astype(str).str.strip()
        df_events["Trade Gate Key"] = df_events["Trade Gate Key"].where(
            df_events["Trade Gate Key"].ne(""),
            fallback_trade_gate.map(trade_gate_key),
        )
        df_events["Trade Gate Key"] = df_events["Trade Gate Key"].replace("", "Unknown").fillna("Unknown")
    if "market_no_trade_reason" in df_events.columns:
        df_events[copy_text("review.label.no_trade_reason")] = (
            df_events["market_no_trade_reason"]
            .replace("", "None")
            .fillna("None")
            .astype(str)
            .str.replace("_", " ", regex=False)
            .str.title()
        )
    if "risk_tier" in df_events.columns:
        df_events["Risk Tier"] = df_events["risk_tier"].replace("", "Unknown").fillna("Unknown")
    if "sector_tag" in df_events.columns:
        df_events["Sector"] = df_events["sector_tag"].replace("", "Other").fillna("Other")
    if "market_sector_rotation" in df_events.columns:
        df_events["Sector Rotation"] = (
            df_events["market_sector_rotation"].replace("", "Unknown").fillna("Unknown")
        )
    if "market_catalyst_state" in df_events.columns:
        df_events["Catalyst State"] = (
            df_events["market_catalyst_state"].replace("", "Unknown").fillna("Unknown")
        )
    if "market_catalyst_event" in df_events.columns:
        df_events["Catalyst Event"] = (
            df_events["market_catalyst_event"].replace("", "None").fillna("None")
        )
    if "market_catalyst_category" in df_events.columns:
        df_events["Catalyst Category"] = (
            df_events["market_catalyst_category"].replace("", "Unknown").fillna("Unknown")
        )
    if "market_catalyst_scope" in df_events.columns:
        df_events["Catalyst Scope"] = (
            df_events["market_catalyst_scope"].replace("", "Unknown").fillna("Unknown")
        )
    if "market_catalyst_tag" in df_events.columns:
        df_events["Catalyst Tag"] = (
            df_events["market_catalyst_tag"].replace("", "None").fillna("None")
        )
    if "market_catalyst_targeted" in df_events.columns:
        df_events["Catalyst Targeting"] = (
            df_events["market_catalyst_targeted"].fillna(0).astype(int).map({1: "Targeted", 0: "Market-Wide"})
        )
    if "market_catalyst_window" in df_events.columns:
        df_events["Catalyst Window"] = (
            df_events["market_catalyst_window"].replace("", "Unknown").fillna("Unknown")
        )
    if "market_flow_state" in df_events.columns:
        df_events["Flow Proxy"] = (
            df_events["market_flow_state"].replace("", "Unknown").fillna("Unknown")
        )
    if "session_bucket" in df_events.columns:
        df_events["Session"] = df_events["session_bucket"].replace("", "Unknown").fillna("Unknown")
    elif "event_time" in df_events.columns:
        df_events["Session"] = pd.to_datetime(df_events["event_time"], utc=True, errors="coerce").map(
            lambda ts: session_bucket_for_timestamp(ts) if pd.notna(ts) else "Unknown"
        )
    if "Playbook" in df_events.columns and "Session" in df_events.columns:
        df_events["Playbook x Session"] = (
            df_events["Playbook"].astype(str).str.strip().replace("", "Unknown")
            + " | "
            + df_events["Session"].astype(str).str.strip().replace("", "Unknown")
        )
    if "Playbook" in df_events.columns and "Catalyst Window" in df_events.columns:
        df_events["Playbook x Catalyst Window"] = (
            df_events["Playbook"].astype(str).str.strip().replace("", "Unknown")
            + " | "
            + df_events["Catalyst Window"].astype(str).str.strip().replace("", "Unknown")
        )
    if "adaptive_edge_label" in df_events.columns:
        df_events["Adaptive Edge"] = (
            df_events["adaptive_edge_label"].replace("", "Unknown").fillna("Unknown")
        )
    if "archive_guardrail_label" in df_events.columns:
        df_events["Archive Guardrail"] = (
            df_events["archive_guardrail_label"].replace("", "Archive Clear").fillna("Archive Clear")
        )
    if "archive_guardrail_penalty" in df_events.columns:
        penalty_series = pd.to_numeric(df_events["archive_guardrail_penalty"], errors="coerce").fillna(0.0)
        df_events["Archive Guardrail Severity"] = penalty_series.map(
            lambda value: (
                "Guardrail"
                if float(value) >= 5.0
                else ("Caution" if float(value) >= 3.0 else "Clear")
            )
        )
    if "Trade Gate" in df_events.columns:
        df_events["Execution Stance"] = df_events.apply(
            lambda row: archived_execution_stance_label(
                trade_gate=str(row.get("Trade Gate") or ""),
                adaptive_edge=str(row.get("Adaptive Edge") or ""),
                archive_guardrail_severity=str(row.get("Archive Guardrail Severity") or ""),
            ),
            axis=1,
        )
    if "trade_decision" in df_events.columns:
        df_events["Trade Decision"] = (
            df_events["trade_decision"].replace("", "Unmarked").fillna("Unmarked")
        )
    if "actual_trade_status" in df_events.columns:
        df_events["Actual Trade Status"] = (
            df_events["actual_trade_status"].replace("", "Unjournaled").fillna("Unjournaled")
        )
    if "actual_exit_reason" in df_events.columns:
        df_events["Actual Exit Reason"] = (
            df_events["actual_exit_reason"].replace("", "Open / Unset").fillna("Open / Unset")
        )
    df_events = annotate_alert_footprint(df_events)
    df_events = _annotate_actual_hold_style(df_events)
    df_events = _annotate_actual_exit_quality(df_events)

    session_summary_df = build_signal_cohort_summary(df_events, "Session") if "Session" in df_events.columns else pd.DataFrame()
    catalyst_window_summary_df = (
        build_signal_cohort_summary(df_events, "Catalyst Window") if "Catalyst Window" in df_events.columns else pd.DataFrame()
    )
    archive_guardrail_summary_df = (
        build_signal_cohort_summary(df_events, "Archive Guardrail") if "Archive Guardrail" in df_events.columns else pd.DataFrame()
    )
    execution_stance_summary_df = (
        build_signal_cohort_summary(df_events, "Execution Stance") if "Execution Stance" in df_events.columns else pd.DataFrame()
    )
    scan_focus_summary_df = (
        build_signal_cohort_summary(df_events, "Scan Focus") if "Scan Focus" in df_events.columns else pd.DataFrame()
    )
    hold_style_summary_df = build_signal_cohort_summary(df_events, "Hold Style") if "Hold Style" in df_events.columns else pd.DataFrame()
    exit_quality_summary_df = (
        build_signal_cohort_summary(df_events, "Exit Quality") if "Exit Quality" in df_events.columns else pd.DataFrame()
    )
    playbook_session_summary_df = (
        build_signal_cohort_summary(df_events, "Playbook x Session") if "Playbook x Session" in df_events.columns else pd.DataFrame()
    )
    playbook_catalyst_summary_df = (
        build_signal_cohort_summary(df_events, "Playbook x Catalyst Window")
        if "Playbook x Catalyst Window" in df_events.columns
        else pd.DataFrame()
    )
    primary_alert_summary_df = build_alert_effectiveness_summary(df_events, primary_only=True)
    alert_effectiveness_df = build_alert_effectiveness_summary(df_events, primary_only=False)

    works_cards: list[dict[str, str]] = []
    fail_cards: list[dict[str, str]] = []
    current_market_version = current_decision_version("Market")

    if "Decision Version" in df_events.columns:
        version_counts = (
            df_events["Decision Version"]
            .fillna("Legacy / Unversioned")
            .astype(str)
            .str.strip()
            .replace("", "Legacy / Unversioned")
            .value_counts()
        )
        current_count = int(version_counts.get(current_market_version, 0))
        if len(version_counts) > 1:
            dominant_version = str(version_counts.index[0])
            dominant_count = int(version_counts.iloc[0])
            fail_cards.append(
                {
                    "title": "Mixed Decision Versions",
                    "body_html": (
                        f"This archive currently mixes <b>{len(version_counts)}</b> decision versions. "
                        f"The largest slice is <b>{dominant_version}</b> ({dominant_count} signals), while the current "
                        f"scanner version <b>{current_market_version}</b> has {current_count} signals so far."
                    ),
                    "tone": "warning",
                }
            )
        elif len(version_counts) == 1:
            only_version = str(version_counts.index[0])
            only_count = int(version_counts.iloc[0])
            works_cards.append(
                {
                    "title": "Decision Version",
                    "body_html": (
                        f"Current archive slice is running on <b>{only_version}</b> "
                        f"({only_count} logged signals in this review window)."
                    ),
                    "tone": "positive" if only_version == current_market_version else "neutral",
                }
            )

    qualified_session_execution_df = _qualified_summary_rows(
        session_summary_df,
        count_field="ClosedTradeCount",
        min_count=_MIN_EXECUTION_ARCHIVE_ROWS,
    )
    qualified_session_signal_df = _qualified_summary_rows(
        session_summary_df,
        count_field="Resolved",
        min_count=_MIN_SIGNAL_ARCHIVE_ROWS,
    )
    qualified_known_session_execution_df = _prefer_known_summary_rows(
        qualified_session_execution_df,
        label_field="Session",
    )
    qualified_known_session_signal_df = _prefer_known_summary_rows(
        qualified_session_signal_df,
        label_field="Session",
    )
    if not qualified_known_session_execution_df.empty:
        best_execution_row = qualified_known_session_execution_df.sort_values(
            ["ActualWinRatePct", "ClosedTradeCount", "Signals"], ascending=[False, False, False]
        ).iloc[0]
        works_cards.append(
            {
                "title": "Best Session",
                "body_html": (
                    f"<b>{best_execution_row['Session']}</b> is currently converting best in real execution "
                    f"({float(best_execution_row['ActualWinRatePct']):.1f}% across "
                    f"{int(best_execution_row['ClosedTradeCount'])} closed trades)."
                ),
                "tone": "positive" if float(best_execution_row["ActualWinRatePct"]) >= 55.0 else "neutral",
            }
        )
    elif not qualified_known_session_signal_df.empty:
        best_follow_row = qualified_known_session_signal_df.sort_values(
            ["FollowThroughPct", "Resolved", "Signals"], ascending=[False, False, False]
        ).iloc[0]
        works_cards.append(
            {
                "title": "Best Session",
                "body_html": (
                    f"Execution archive is still building. On the signal side, <b>{best_follow_row['Session']}</b> "
                    f"is currently the cleanest session ({float(best_follow_row['FollowThroughPct']):.1f}% follow-through "
                    f"across {int(best_follow_row['Resolved'])} resolved signals)."
                ),
                "tone": "neutral",
            }
        )
    else:
        works_cards.append(
            _archive_building_card(
                "Best Session",
                "Session archive is still building. We need more resolved signals or journaled trades before trusting timing rankings.",
            )
        )

    qualified_execution_stance_df = _qualified_summary_rows(
        execution_stance_summary_df,
        count_field="Resolved",
        min_count=_MIN_SIGNAL_ARCHIVE_ROWS,
    )
    if len(qualified_execution_stance_df) >= 2:
        strongest_execution_stance = qualified_execution_stance_df.sort_values(
            ["FollowThroughPct", "Resolved", "Signals"], ascending=[False, False, False]
        ).iloc[0]
        weakest_execution_stance = qualified_execution_stance_df.sort_values(
            ["FollowThroughPct", "Resolved", "Signals"], ascending=[True, False, False]
        ).iloc[0]
        works_cards.append(
            {
                "title": "Best Execution Stance",
                "body_html": (
                    f"<b>{strongest_execution_stance['Execution Stance']}</b> is the cleanest current stance "
                    f"({float(strongest_execution_stance['FollowThroughPct']):.1f}% follow-through across "
                    f"{int(strongest_execution_stance['Resolved'])} resolved signals)."
                ),
                "tone": "positive"
                if trade_gate_key(strongest_execution_stance["Execution Stance"]) == "TRADEABLE"
                else "neutral",
            }
        )
        fail_cards.append(
            {
                "title": "Weakest Execution Stance",
                "body_html": (
                    f"<b>{weakest_execution_stance['Execution Stance']}</b> is the weakest current stance "
                    f"({float(weakest_execution_stance['FollowThroughPct']):.1f}% follow-through across "
                    f"{int(weakest_execution_stance['Resolved'])} resolved signals)."
                ),
                "tone": "warning",
            }
        )
    elif len(qualified_execution_stance_df) == 1:
        only_stance = qualified_execution_stance_df.iloc[0]
        works_cards.append(
            _archive_building_card(
                "Execution Stance Archive",
                (
                    f"Only <b>{only_stance['Execution Stance']}</b> has enough resolved history in this slice so far "
                    f"({int(only_stance['Resolved'])} resolved signals). We need more stance variety before ranking strongest vs weakest."
                ),
            )
        )
    else:
        works_cards.append(
            _archive_building_card(
                "Execution Stance Archive",
                "Execution stance archive is still building. We need more resolved signals before trusting stance rankings.",
            )
        )

    qualified_primary_alert_df = _qualified_summary_rows(
        primary_alert_summary_df,
        count_field="Resolved",
        min_count=_MIN_SIGNAL_ARCHIVE_ROWS,
    )
    if not qualified_primary_alert_df.empty:
        strongest_primary_alert = qualified_primary_alert_df.sort_values(
            ["FollowThroughPct", "Resolved", "Signals"], ascending=[False, False, False]
        ).iloc[0]
        works_cards.append(
            {
                "title": "Best Primary Alert",
                "body_html": (
                    f"<b>{strongest_primary_alert['Primary Alert']}</b> is converting best "
                    f"({float(strongest_primary_alert['FollowThroughPct']):.1f}% follow-through across "
                    f"{int(strongest_primary_alert['Resolved'])} resolved signals)."
                ),
                "tone": "positive" if float(strongest_primary_alert["FollowThroughPct"]) >= 55.0 else "neutral",
            }
        )
    else:
        works_cards.append(
            _archive_building_card(
                "Primary Alert Archive",
                "Primary alert archive is still building. We need more resolved signals before trusting alert rankings.",
            )
        )

    qualified_scan_focus_df = _qualified_summary_rows(
        scan_focus_summary_df,
        count_field="Resolved",
        min_count=_MIN_SIGNAL_ARCHIVE_ROWS,
    )
    qualified_known_scan_focus_df = _prefer_known_summary_rows(
        qualified_scan_focus_df,
        label_field="Scan Focus",
    )
    if len(qualified_known_scan_focus_df) >= 2:
        strongest_scan_focus = qualified_known_scan_focus_df.sort_values(
            ["FollowThroughPct", "Resolved", "Signals"], ascending=[False, False, False]
        ).iloc[0]
        weakest_scan_focus = qualified_known_scan_focus_df.sort_values(
            ["FollowThroughPct", "Resolved", "Signals"], ascending=[True, False, False]
        ).iloc[0]
        works_cards.append(
            {
                "title": "Best Scan Focus",
                "body_html": (
                    f"<b>{strongest_scan_focus['Scan Focus']}</b> is converting best in this slice "
                    f"({float(strongest_scan_focus['FollowThroughPct']):.1f}% follow-through across "
                    f"{int(strongest_scan_focus['Resolved'])} resolved signals)."
                ),
                "tone": "positive" if float(strongest_scan_focus["FollowThroughPct"]) >= 55.0 else "neutral",
            }
        )
        fail_cards.append(
            {
                "title": "Weakest Scan Focus",
                "body_html": (
                    f"<b>{weakest_scan_focus['Scan Focus']}</b> is converting weakest in this slice "
                    f"({float(weakest_scan_focus['FollowThroughPct']):.1f}% follow-through across "
                    f"{int(weakest_scan_focus['Resolved'])} resolved signals)."
                ),
                "tone": "warning" if float(weakest_scan_focus["FollowThroughPct"]) < 45.0 else "neutral",
            }
        )
    elif len(qualified_known_scan_focus_df) == 1:
        only_focus = qualified_known_scan_focus_df.iloc[0]
        works_cards.append(
            _archive_building_card(
                "Scan Focus Archive",
                (
                    f"Only <b>{only_focus['Scan Focus']}</b> has enough resolved history in this slice so far "
                    f"({int(only_focus['Resolved'])} resolved signals). We need both Broad Market and Actionable Setups "
                    "in the archive before comparing focus quality."
                ),
            )
        )
    else:
        works_cards.append(
            _archive_building_card(
                "Scan Focus Archive",
                "Scan-focus archive is still building. We need more resolved signals before comparing Broad Market vs Actionable Setups.",
            )
        )

    qualified_playbook_session_df = _qualified_summary_rows(
        playbook_session_summary_df,
        count_field="Resolved",
        min_count=_MIN_SIGNAL_ARCHIVE_ROWS,
    )
    qualified_known_playbook_session_df = _prefer_known_summary_rows(
        qualified_playbook_session_df,
        label_field="Playbook x Session",
    )
    if not qualified_known_playbook_session_df.empty:
        best_playbook_session = qualified_known_playbook_session_df.sort_values(
            ["FollowThroughPct", "Resolved", "Signals"], ascending=[False, False, False]
        ).iloc[0]
        works_cards.append(
            {
                "title": "Best Playbook Timing",
                "body_html": (
                    f"<b>{best_playbook_session['Playbook x Session']}</b> is the strongest timing combo "
                    f"({float(best_playbook_session['FollowThroughPct']):.1f}% follow-through across "
                    f"{int(best_playbook_session['Resolved'])} resolved signals)."
                ),
                "tone": "positive" if float(best_playbook_session["FollowThroughPct"]) >= 55.0 else "neutral",
            }
        )
    else:
        works_cards.append(
            _archive_building_card(
                "Playbook Timing",
                "Playbook timing archive is still building. We need more resolved signals before trusting timing combos.",
            )
        )

    qualified_hold_style_df = _qualified_summary_rows(
        hold_style_summary_df,
        count_field="ClosedTradeCount",
        min_count=_MIN_EXECUTION_ARCHIVE_ROWS,
    )
    if not qualified_hold_style_df.empty:
        best_hold_style = qualified_hold_style_df.sort_values(
            ["ActualWinRatePct", "ClosedTradeCount", "Signals"], ascending=[False, False, False]
        ).iloc[0]
        works_cards.append(
            {
                "title": "Best Hold Profile",
                "body_html": (
                    f"<b>{best_hold_style['Hold Style']}</b> is the healthiest hold profile so far "
                    f"({float(best_hold_style['ActualWinRatePct']):.1f}% closed-trade win rate across "
                    f"{int(best_hold_style['ClosedTradeCount'])} journaled trades)."
                ),
                "tone": "positive" if float(best_hold_style["ActualWinRatePct"]) >= 55.0 else "neutral",
            }
        )
    else:
        works_cards.append(
            _archive_building_card(
                "Hold Profile Archive",
                "Hold-profile coaching is still building. We need more journaled closed trades before trusting this read.",
            )
        )

    qualified_catalyst_window_df = _qualified_summary_rows(
        catalyst_window_summary_df,
        count_field="Resolved",
        min_count=_MIN_SIGNAL_ARCHIVE_ROWS,
    )
    qualified_known_catalyst_window_df = _prefer_known_summary_rows(
        qualified_catalyst_window_df,
        label_field="Catalyst Window",
    )
    if len(qualified_known_catalyst_window_df) >= 2:
        weakest_catalyst_row = qualified_known_catalyst_window_df.sort_values(
            ["FollowThroughPct", "Resolved", "Signals"], ascending=[True, False, False]
        ).iloc[0]
        fail_cards.append(
            {
                "title": "Weakest Event Window",
                "body_html": (
                    f"<b>{weakest_catalyst_row['Catalyst Window']}</b> is the weakest event window "
                    f"({float(weakest_catalyst_row['FollowThroughPct']):.1f}% follow-through across "
                    f"{int(weakest_catalyst_row['Resolved'])} resolved signals)."
                ),
                "tone": "warning" if float(weakest_catalyst_row["FollowThroughPct"]) < 45.0 else "neutral",
            }
        )
    else:
        fail_cards.append(
            _archive_building_card(
                "Event Window Archive",
                "Event-window archive is still too thin or too one-sided to trust weakest-window rankings.",
            )
        )

    qualified_guardrail_df = _qualified_summary_rows(
        archive_guardrail_summary_df,
        count_field="Resolved",
        min_count=_MIN_SIGNAL_ARCHIVE_ROWS,
    )
    qualified_known_guardrail_df = _prefer_known_summary_rows(
        qualified_guardrail_df,
        label_field="Archive Guardrail",
    )
    if len(qualified_known_guardrail_df) >= 2:
        strongest_guardrail_row = qualified_known_guardrail_df.sort_values(
            ["FollowThroughPct", "Resolved", "Signals"], ascending=[True, False, False]
        ).iloc[0]
        fail_cards.append(
            {
                "title": "Weakest Guardrail Cluster",
                "body_html": (
                    f"<b>{strongest_guardrail_row['Archive Guardrail']}</b> is the weakest matched archive cluster "
                    f"({float(strongest_guardrail_row['FollowThroughPct']):.1f}% across "
                    f"{int(strongest_guardrail_row['Resolved'])} resolved signals)."
                ),
                "tone": "warning",
            }
        )
    else:
        fail_cards.append(
            _archive_building_card(
                "Guardrail Archive",
                "Guardrail archive is still too thin to rank weak clusters confidently.",
            )
        )

    qualified_playbook_catalyst_df = _qualified_summary_rows(
        playbook_catalyst_summary_df,
        count_field="Resolved",
        min_count=_MIN_SIGNAL_ARCHIVE_ROWS,
    )
    qualified_known_playbook_catalyst_df = _prefer_known_summary_rows(
        qualified_playbook_catalyst_df,
        label_field="Playbook x Catalyst Window",
    )
    if len(qualified_known_playbook_catalyst_df) >= 2:
        weakest_playbook_catalyst = qualified_known_playbook_catalyst_df.sort_values(
            ["FollowThroughPct", "Resolved", "Signals"], ascending=[True, False, False]
        ).iloc[0]
        fail_cards.append(
            {
                "title": "Weakest Playbook/Event Fit",
                "body_html": (
                    f"<b>{weakest_playbook_catalyst['Playbook x Catalyst Window']}</b> is the weakest combo "
                    f"({float(weakest_playbook_catalyst['FollowThroughPct']):.1f}% follow-through across "
                    f"{int(weakest_playbook_catalyst['Resolved'])} resolved signals)."
                ),
                "tone": "warning" if float(weakest_playbook_catalyst["FollowThroughPct"]) < 45.0 else "neutral",
            }
        )
    else:
        fail_cards.append(
            _archive_building_card(
                "Playbook/Event Archive",
                "Playbook/event archive is still too thin to rank weak combinations confidently.",
            )
        )

    qualified_exit_quality_df = _qualified_summary_rows(
        exit_quality_summary_df,
        count_field="ClosedTradeCount",
        min_count=_MIN_EXECUTION_ARCHIVE_ROWS,
    )
    if len(qualified_exit_quality_df) >= 2:
        weakest_exit_quality = qualified_exit_quality_df.sort_values(
            ["ActualWinRatePct", "ClosedTradeCount", "Signals"], ascending=[True, False, False]
        ).iloc[0]
        fail_cards.append(
            {
                "title": "Weakest Exit Discipline",
                "body_html": (
                    f"<b>{weakest_exit_quality['Exit Quality']}</b> is the weakest realized exit pattern "
                    f"({float(weakest_exit_quality['ActualWinRatePct']):.1f}% across "
                    f"{int(weakest_exit_quality['ClosedTradeCount'])} closed trades)."
                ),
                "tone": "warning",
            }
        )
    else:
        fail_cards.append(
            _archive_building_card(
                "Exit Discipline Archive",
                "Exit-discipline archive is still building. We need more journaled closed trades before trusting weakest-exit rankings.",
            )
        )

    works_cards = _prepare_section_cards(works_cards, max_actionable=3)
    fail_cards = _prepare_section_cards(fail_cards, max_actionable=3)

    st.markdown("### What Works")
    st.caption("Read this as the current trust list for the selected review slice.")
    _render_insight_card_grid(st, works_cards, columns=3)

    st.markdown("### What Fails")
    st.caption("Read this as the current brake list: where we should be more selective or stand aside.")
    _render_insight_card_grid(st, fail_cards, columns=3)

    _render_execution_review_section(
        st=st,
        df_events=df_events,
        db_path=db_path,
        save_signal_trade_overlay=save_signal_trade_overlay,
        save_signal_trade_journal=save_signal_trade_journal,
        positive_color=POSITIVE,
    )

    st.markdown("### Deep Dives")
    st.caption("Open these only when you want the detail behind the top-level read.")
    with st.expander("Market, Setup & Timing Deep Dive", expanded=False):
        _render_compact_cohort_tables(
            st,
            df_events=df_events,
            build_signal_cohort_summary=build_signal_cohort_summary,
            specs=[
                ("setup_confirm", "By Setup Confirm"),
                ("Lead", "By LEAD"),
                ("Market Lead", "By Market Lead"),
                ("Session", "By Session"),
                ("timeframe", "By Timeframe"),
                ("Decision Version", "By Decision Version"),
                ("Market Regime", "By Market Regime"),
                ("Scan Focus", "By Scan Focus"),
                ("Playbook", "By Playbook"),
                ("Trade Gate", "By Trade Gate"),
                ("AI Alignment", "By AI Alignment"),
                ("Adaptive Edge", "By Adaptive Edge"),
            ],
        )

    with st.expander("Alerts, Events & Context Deep Dive", expanded=False):
        _render_compact_cohort_tables(
            st,
            df_events=df_events,
            build_signal_cohort_summary=build_signal_cohort_summary,
            specs=[
                ("Primary Alert", "By Primary Alert"),
                ("Catalyst State", "By Catalyst State"),
                ("Catalyst Window", "By Catalyst Window"),
                ("Catalyst Scope", "By Catalyst Scope"),
                ("Catalyst Targeting", "By Catalyst Targeting"),
                ("Catalyst Category", "By Catalyst Category"),
                ("Catalyst Tag", "By Catalyst Tag"),
                ("Flow Proxy", "By Flow Proxy"),
                ("Playbook x Session", "By Playbook x Session"),
                ("Playbook x Catalyst Window", "By Playbook x Catalyst Window"),
                ("Sector Rotation", "By Sector Rotation"),
                ("Sector", "By Sector"),
            ],
        )
        if not alert_effectiveness_df.empty:
            st.markdown("##### By Alert Key")
            st.dataframe(alert_effectiveness_df.round(2), hide_index=True, width="stretch")

    with st.expander("Execution & Journal Deep Dive", expanded=False):
        _render_compact_cohort_tables(
            st,
            df_events=df_events,
            build_signal_cohort_summary=build_signal_cohort_summary,
            specs=[
                ("Execution Stance", "By Execution Stance"),
                ("Archive Guardrail", "By Archive Guardrail"),
                ("Archive Guardrail Severity", "By Guardrail Severity"),
                ("Risk Tier", "By Risk Tier"),
                (copy_text("review.label.no_trade_reason"), copy_text("review.group.no_trade_reason")),
                ("Trade Decision", "By Trade Decision"),
                ("Actual Trade Status", "By Actual Trade Status"),
                ("Hold Style", "By Hold Style"),
                ("Exit Quality", "By Exit Quality"),
                ("Actual Exit Reason", "By Actual Exit Reason"),
            ],
        )

    recent_cols = [
        "event_time",
        "symbol",
        "timeframe",
        "Primary Alert",
        "Alert Footprint",
        "Scan Focus",
        "Decision Version",
        "Created Decision Version",
        "setup_confirm",
        "direction",
        "Lead",
        "Market Lead",
        "Sector",
        "Sector Rotation",
        "Catalyst State",
        "Catalyst Window",
        "Catalyst Scope",
        "Catalyst Targeting",
        "Catalyst Category",
        "Catalyst Tag",
        "Catalyst Event",
        "Flow Proxy",
        "Session",
        "Adaptive Edge",
        "Execution Stance",
        "Market Regime",
        "Trade Gate",
        "Trade Decision",
        "Actual Trade Status",
        "Hold Style",
        "Actual Hold Hours",
        "Exit Quality",
        "Actual Exit Reason",
        "Risk Tier",
        copy_text("review.label.no_trade_reason"),
        "Playbook",
        "Playbook x Session",
        "Playbook x Catalyst Window",
        "adaptive_edge_score",
        "actionable_frame_score",
        "actionable_setup_score",
        "actionable_context_score",
        "actionable_tactical_score",
        "trade_note",
        "actual_trade_side",
        "actual_entry_price",
        "actual_entry_at",
        "actual_exit_price",
        "actual_exit_at",
        "actual_pnl_pct",
        "confidence",
        "ai_confidence",
        "plan_outcome",
        "directional_return_pct",
        "favorable_excursion_pct",
        "adverse_excursion_pct",
        "status",
    ]
    recent_df = df_events[[c for c in recent_cols if c in df_events.columns]].copy()
    if "actual_trade_side" in recent_df.columns:
        recent_df["actual_trade_side"] = recent_df["actual_trade_side"].map(_display_trade_direction).replace("", "—")
    if "event_time" in recent_df.columns:
        recent_df["event_time"] = pd.to_datetime(recent_df["event_time"], errors="coerce").dt.strftime("%Y-%m-%d %H:%M")
    rename_map = {
        "event_time": "Signal Time",
        "symbol": "Coin",
        "timeframe": "TF",
        "setup_confirm": "Setup Confirm",
        "direction": "Direction",
        "confidence": "Confidence",
        "ai_confidence": "AI Confidence",
        "plan_outcome": "Plan Outcome",
        "directional_return_pct": "Dir Return %",
        "favorable_excursion_pct": "MFE %",
        "adverse_excursion_pct": "MAE %",
        "status": "Status",
        "adaptive_edge_score": "Adaptive Score",
        "actionable_frame_score": "Actionable Hunt Score",
        "actionable_setup_score": "Actionable Setup Score",
        "actionable_context_score": "Actionable Context Score",
        "actionable_tactical_score": "Actionable Tactical Score",
        "trade_note": "Trade Note",
        "actual_trade_side": "Trade Direction",
        "actual_entry_price": "Actual Entry",
        "actual_entry_at": "Entry Time",
        "actual_exit_price": "Actual Exit",
        "actual_exit_at": "Exit Time",
        "actual_pnl_pct": "Actual PnL %",
    }
    recent_df = recent_df.rename(columns=rename_map)
    with st.expander("Raw Tables", expanded=False):
        st.markdown("#### Recent Signals")
        st.dataframe(recent_df.round(2), hide_index=True, width="stretch")

        if not df_alerts.empty:
            alerts_df = df_alerts.copy()
            if "last_seen_at" in alerts_df.columns:
                alerts_df["last_seen_at"] = pd.to_datetime(alerts_df["last_seen_at"], errors="coerce").dt.strftime("%Y-%m-%d %H:%M")
            if "first_seen_at" in alerts_df.columns:
                alerts_df["first_seen_at"] = pd.to_datetime(alerts_df["first_seen_at"], errors="coerce").dt.strftime("%Y-%m-%d %H:%M")
            if "active" in alerts_df.columns:
                alerts_df["Active"] = alerts_df["active"].fillna(0).astype(int).map({1: "Yes", 0: "No"})
            alerts_df = alerts_df.rename(
                columns={
                    "severity": "Severity",
                    "title": "Title",
                    "note": "Note",
                    "last_seen_at": "Last Seen",
                    "first_seen_at": "First Seen",
                    "times_seen": "Times Seen",
                    "alert_key": "Alert Key",
                }
            )
            st.markdown("#### Recent Market Alerts")
            st.dataframe(
                alerts_df[
                    [
                        c
                        for c in ["Last Seen", "Active", "Severity", "Title", "Note", "Times Seen", "Alert Key"]
                        if c in alerts_df.columns
                    ]
                ],
                hide_index=True,
                width="stretch",
            )

        if not learned_edges_df.empty:
            st.markdown("#### Learned Edges")
            st.dataframe(learned_edges_df.round(2), hide_index=True, width="stretch")

    _render_tracker_backup_restore(
        st=st,
        db_path=db_path,
        storage_snapshot=storage_snapshot,
        read_signal_tracker_db_bytes=read_signal_tracker_db_bytes,
        backup_signal_tracker_db=backup_signal_tracker_db,
        restore_signal_tracker_db_bytes=restore_signal_tracker_db_bytes,
        fetch_signal_events_df=fetch_signal_events_df,
        fetch_market_alerts_df=fetch_market_alerts_df,
    )
