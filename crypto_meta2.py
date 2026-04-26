import streamlit as st
import pandas as pd
import numpy as np
import threading
import core.services as _services
import core.signal_tracker as _signal_tracker
from core.alert_engine import build_market_alerts
from core.adaptive_weighting import (
    build_adaptive_context_model,
    build_learning_edge_table,
    build_live_signal_adaptive_snapshot,
)
from core.signal_tracker import (
    annotate_alert_footprint,
    backfill_signal_forward_windows_via_fetch,
    build_alert_effectiveness_summary,
    build_hold_window_cohort_summary,
    build_recent_market_context_snapshot,
    build_recent_symbol_market_signal_snapshot,
    build_hold_window_intelligence,
    build_signal_cohort_summary,
    build_execution_overlay_snapshot,
    build_scanner_trace_summary,
    build_signal_review_snapshot,
    fetch_market_alerts_df,
    fetch_breakout_radar_snapshots_df,
    fetch_signal_forward_windows_df,
    fetch_scanner_trace_events_df,
    fetch_signal_events_df,
    get_signal_tracker_db_path,
    init_signal_tracker_db,
    log_breakout_radar_snapshots,
    log_market_alerts,
    log_scanner_trace_events,
    log_signal_events,
    resolve_open_signal_events_for_frame,
    resolve_open_signal_events_via_fetch,
    save_signal_trade_journal,
    save_signal_trade_overlay,
)
from core.tracker_store import (
    backup_signal_tracker_db,
    build_tracker_storage_snapshot,
    read_signal_tracker_db_bytes,
    restore_signal_tracker_db_bytes,
)
from core.catalyst_engine import build_market_catalyst_snapshot
from core.flow_proxy_engine import build_market_flow_proxy_snapshot
from core.no_trade_engine import build_market_trade_gate
from core.regime_engine import build_market_regime_snapshot
from core.risk_sizing_engine import build_signal_risk_sizing, market_default_risk_budget
from core.sector_rotation import build_sector_rotation_snapshot, classify_symbol_sector
from core.backtest import run_backtest as run_backtest_core
from core.services import (
    EXCHANGE,
    _normalize_coin_input,
    _sr_lookback,
    _symbol_variants,
    _validate_coin_symbol,
    analyse,
    calculate_fibonacci_levels,
    calculate_risk_metrics,
    calculate_volume_profile,
    detect_divergence,
    detect_market_regime,
    fetch_ohlcv,
    fetch_exchange_tickers_snapshot,
    get_market_flow_proxy_rows,
    get_market_catalyst_events,
    get_market_cap_rows_for_symbols,
    fetch_top_gainers_losers,
    fetch_trending_coins,
    get_major_ohlcv_bundle,
    get_market_indices,
    get_market_top_snapshot,
    get_price_change,
    get_scalping_entry_target,
    scalp_quality_gate,
    get_top_volume_usdt_symbols,
    ml_ensemble_predict,
    monte_carlo_simulation,
)
import ui.helpers as _ui_helpers
from ui.theme import (
    ACCENT,
    CARD_BG,
    GOLD,
    NEGATIVE,
    NEON_BLUE,
    NEON_PURPLE,
    POSITIVE,
    PRIMARY_BG,
    TEXT_LIGHT,
    TEXT_MUTED,
    WARNING,
    calc_conviction as _calc_conviction,
    tip as _tip,
)
from ui.app_shell import render_app
import ui.deps_factory as _deps_factory
from ui.styles import app_css

build_app_deps = _deps_factory.build_app_deps
count_market_alerts = getattr(
    _signal_tracker,
    "count_market_alerts",
    lambda *, active_only=False, source=None, db_path=None: len(
        fetch_market_alerts_df(
            limit=100000,
            active_only=active_only,
            source=source,
            db_path=db_path,
        )
    ),
)
direction_key_fallback = getattr(
    _deps_factory,
    "direction_key_fallback",
    lambda direction: str(direction or "").strip().upper(),
)
direction_label_fallback = getattr(
    _deps_factory,
    "direction_label_fallback",
    lambda direction: str(direction or "Neutral"),
)
missing_fetch_coingecko_ohlcv_by_coin_id = getattr(
    _deps_factory,
    "missing_fetch_coingecko_ohlcv_by_coin_id",
    lambda *_args, **_kwargs: None,
)
sanitize_trading_terms_fallback = getattr(
    _deps_factory,
    "sanitize_trading_terms_fallback",
    lambda text: "" if text is None else str(text),
)
signal_plain_fallback = getattr(
    _deps_factory,
    "signal_plain_fallback",
    lambda signal: "WAIT" if not signal else str(signal),
)
style_delta_fallback = getattr(_deps_factory, "style_delta_fallback", lambda *_args, **_kwargs: "")
style_scalp_opp_fallback = getattr(_deps_factory, "style_scalp_opp_fallback", lambda *_args, **_kwargs: "")
style_signal_fallback = getattr(_deps_factory, "style_signal_fallback", lambda *_args, **_kwargs: "")


def _fallback_bias_score_badge(bias_score: float) -> str:
    try:
        return f"{round(float(bias_score))}"
    except Exception:
        return "N/A"


def _fallback_format_adx(adx: float) -> str:
    try:
        v = float(adx)
    except Exception:
        return "N/A"
    return f"{v:.1f}"


def _fallback_format_delta(delta) -> str:
    try:
        v = float(delta)
    except Exception:
        return ""
    return f"{v:+.2f}%"


def _fallback_format_stochrsi(value, timeframe=None) -> str:
    try:
        v = float(value)
    except Exception:
        return "N/A"
    return f"{v:.2f}"


def _fallback_format_trend(trend: str) -> str:
    t = str(trend or "").strip()
    return t if t else "Neutral"


def _fallback_leverage_badge(lev: int) -> str:
    try:
        return f"x{int(lev)}"
    except Exception:
        return "x1"


def _fallback_readable_market_cap(value) -> str:
    try:
        v = float(value)
    except Exception:
        return "N/A"
    if v >= 1_000_000_000_000:
        return f"{v / 1_000_000_000_000:.2f}T"
    if v >= 1_000_000_000:
        return f"{v / 1_000_000_000:.2f}B"
    if v >= 1_000_000:
        return f"{v / 1_000_000:.2f}M"
    return f"{v:,.0f}"


def _missing_get_heatmap_rows(*_args, **_kwargs):
    return [], "Unavailable", "EMPTY", None


_missing_get_heatmap_rows._codex_missing_dep = True
_missing_get_heatmap_rows._codex_missing_dep_reason = (
    "core.services heatmap helper unavailable during app bootstrap"
)


bias_score_badge = getattr(_ui_helpers, "bias_score_badge", _fallback_bias_score_badge)
format_adx = getattr(_ui_helpers, "format_adx", _fallback_format_adx)
format_delta = getattr(_ui_helpers, "format_delta", _fallback_format_delta)
format_stochrsi = getattr(_ui_helpers, "format_stochrsi", _fallback_format_stochrsi)
format_trend = getattr(_ui_helpers, "format_trend", _fallback_format_trend)
leverage_badge = getattr(_ui_helpers, "leverage_badge", _fallback_leverage_badge)
readable_market_cap = getattr(_ui_helpers, "readable_market_cap", _fallback_readable_market_cap)
direction_key = getattr(_ui_helpers, "direction_key", direction_key_fallback)
direction_label = getattr(_ui_helpers, "direction_label", direction_label_fallback)
signal_plain = getattr(_ui_helpers, "signal_plain", signal_plain_fallback)
sanitize_trading_terms = getattr(_ui_helpers, "sanitize_trading_terms", sanitize_trading_terms_fallback)
style_delta = getattr(_ui_helpers, "style_delta", style_delta_fallback)
style_scalp_opp = getattr(_ui_helpers, "style_scalp_opp", style_scalp_opp_fallback)
style_signal = getattr(_ui_helpers, "style_signal", style_signal_fallback)
fetch_coingecko_ohlcv_by_coin_id = getattr(
    _services,
    "fetch_coingecko_ohlcv_by_coin_id",
    missing_fetch_coingecko_ohlcv_by_coin_id,
)
get_heatmap_rows = getattr(_services, "get_heatmap_rows", _missing_get_heatmap_rows)


def _wma(series: pd.Series, length: int) -> pd.Series:
    """Weighted Moving Average – gives more weight to recent prices."""
    weights = np.arange(1, length + 1, dtype=float)
    return series.rolling(window=length).apply(
        lambda x: np.dot(x, weights) / weights.sum(), raw=True
    )


# Set up page title, icon and wide layout
st.set_page_config(
    page_title="Crypto Market Intelligence Hub",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# === Debug / diagnostics ===
if 'debug_mode' not in st.session_state:
    st.session_state['debug_mode'] = False
with st.sidebar:
    with st.expander("Developer Tools", expanded=False):
        st.session_state['debug_mode'] = st.toggle('Debug mode', value=st.session_state['debug_mode'])
        if "market_show_diagnostics" not in st.session_state:
            st.session_state["market_show_diagnostics"] = False
        st.toggle(
            "Market diagnostics",
            value=bool(st.session_state.get("market_show_diagnostics", False)),
            key="market_show_diagnostics",
        )

def _debug(msg: str) -> None:
    """Emit a debug message only when Debug mode is enabled."""
    if not st.session_state.get('debug_mode', False):
        return
    if threading.current_thread() is threading.main_thread():
        st.sidebar.write(msg)
        return
    print(f"[debug] {msg}")



st.markdown(app_css(), unsafe_allow_html=True)


def run_backtest(df: pd.DataFrame, threshold: float = 70, exit_after: int = 5,
                 commission: float = 0.001, slippage: float = 0.0005) -> tuple[pd.DataFrame, str]:
    """Backward-compatible wrapper to the core backtest engine."""
    return run_backtest_core(
        df,
        analyzer=analyse,
        threshold=threshold,
        exit_after=exit_after,
        commission=commission,
        slippage=slippage,
    )


APP_DEPS = {
    "ACCENT": ACCENT,
    "CARD_BG": CARD_BG,
    "EXCHANGE": EXCHANGE,
    "GOLD": GOLD,
    "NEGATIVE": NEGATIVE,
    "NEON_BLUE": NEON_BLUE,
    "NEON_PURPLE": NEON_PURPLE,
    "POSITIVE": POSITIVE,
    "PRIMARY_BG": PRIMARY_BG,
    "TEXT_LIGHT": TEXT_LIGHT,
    "TEXT_MUTED": TEXT_MUTED,
    "WARNING": WARNING,
    "_calc_conviction": _calc_conviction,
    "_debug": _debug,
    "_normalize_coin_input": _normalize_coin_input,
    "_sr_lookback": _sr_lookback,
    "_symbol_variants": _symbol_variants,
    "_tip": _tip,
    "_validate_coin_symbol": _validate_coin_symbol,
    "_wma": _wma,
    "analyse": analyse,
    "annotate_alert_footprint": annotate_alert_footprint,
    "backfill_signal_forward_windows_via_fetch": backfill_signal_forward_windows_via_fetch,
    "backup_signal_tracker_db": backup_signal_tracker_db,
    "bias_score_badge": bias_score_badge,
    "build_adaptive_context_model": build_adaptive_context_model,
    "build_alert_effectiveness_summary": build_alert_effectiveness_summary,
    "build_execution_overlay_snapshot": build_execution_overlay_snapshot,
    "build_hold_window_cohort_summary": build_hold_window_cohort_summary,
    "build_hold_window_intelligence": build_hold_window_intelligence,
    "build_learning_edge_table": build_learning_edge_table,
    "build_live_signal_adaptive_snapshot": build_live_signal_adaptive_snapshot,
    "build_market_alerts": build_market_alerts,
    "build_market_catalyst_snapshot": build_market_catalyst_snapshot,
    "build_market_flow_proxy_snapshot": build_market_flow_proxy_snapshot,
    "build_market_regime_snapshot": build_market_regime_snapshot,
    "build_market_trade_gate": build_market_trade_gate,
    "build_recent_market_context_snapshot": build_recent_market_context_snapshot,
    "build_recent_symbol_market_signal_snapshot": build_recent_symbol_market_signal_snapshot,
    "build_sector_rotation_snapshot": build_sector_rotation_snapshot,
    "build_signal_cohort_summary": build_signal_cohort_summary,
    "build_scanner_trace_summary": build_scanner_trace_summary,
    "build_signal_review_snapshot": build_signal_review_snapshot,
    "build_signal_risk_sizing": build_signal_risk_sizing,
    "build_tracker_storage_snapshot": build_tracker_storage_snapshot,
    "calculate_fibonacci_levels": calculate_fibonacci_levels,
    "calculate_risk_metrics": calculate_risk_metrics,
    "calculate_volume_profile": calculate_volume_profile,
    "classify_symbol_sector": classify_symbol_sector,
    "count_market_alerts": count_market_alerts,
    "detect_divergence": detect_divergence,
    "detect_market_regime": detect_market_regime,
    "direction_key": direction_key,
    "direction_label": direction_label,
    "fetch_coingecko_ohlcv_by_coin_id": fetch_coingecko_ohlcv_by_coin_id,
    "fetch_breakout_radar_snapshots_df": fetch_breakout_radar_snapshots_df,
    "fetch_exchange_tickers_snapshot": fetch_exchange_tickers_snapshot,
    "fetch_market_alerts_df": fetch_market_alerts_df,
    "fetch_ohlcv": fetch_ohlcv,
    "fetch_scanner_trace_events_df": fetch_scanner_trace_events_df,
    "fetch_signal_events_df": fetch_signal_events_df,
    "fetch_signal_forward_windows_df": fetch_signal_forward_windows_df,
    "fetch_top_gainers_losers": fetch_top_gainers_losers,
    "fetch_trending_coins": fetch_trending_coins,
    "format_adx": format_adx,
    "format_delta": format_delta,
    "format_stochrsi": format_stochrsi,
    "format_trend": format_trend,
    "get_heatmap_rows": get_heatmap_rows,
    "get_major_ohlcv_bundle": get_major_ohlcv_bundle,
    "get_market_cap_rows_for_symbols": get_market_cap_rows_for_symbols,
    "get_market_catalyst_events": get_market_catalyst_events,
    "get_market_flow_proxy_rows": get_market_flow_proxy_rows,
    "get_market_indices": get_market_indices,
    "get_market_top_snapshot": get_market_top_snapshot,
    "get_price_change": get_price_change,
    "get_scalping_entry_target": get_scalping_entry_target,
    "get_signal_tracker_db_path": get_signal_tracker_db_path,
    "get_top_volume_usdt_symbols": get_top_volume_usdt_symbols,
    "init_signal_tracker_db": init_signal_tracker_db,
    "leverage_badge": leverage_badge,
    "log_breakout_radar_snapshots": log_breakout_radar_snapshots,
    "log_market_alerts": log_market_alerts,
    "log_scanner_trace_events": log_scanner_trace_events,
    "log_signal_events": log_signal_events,
    "market_default_risk_budget": market_default_risk_budget,
    "ml_ensemble_predict": ml_ensemble_predict,
    "monte_carlo_simulation": monte_carlo_simulation,
    "read_signal_tracker_db_bytes": read_signal_tracker_db_bytes,
    "readable_market_cap": readable_market_cap,
    "resolve_open_signal_events_for_frame": resolve_open_signal_events_for_frame,
    "resolve_open_signal_events_via_fetch": resolve_open_signal_events_via_fetch,
    "restore_signal_tracker_db_bytes": restore_signal_tracker_db_bytes,
    "run_backtest": run_backtest,
    "sanitize_trading_terms": sanitize_trading_terms,
    "save_signal_trade_journal": save_signal_trade_journal,
    "save_signal_trade_overlay": save_signal_trade_overlay,
    "scalp_quality_gate": scalp_quality_gate,
    "signal_plain": signal_plain,
    "style_delta": style_delta,
    "style_scalp_opp": style_scalp_opp,
    "style_signal": style_signal,
}


def main():
    """Entry point for the Streamlit app."""

    render_app(build_app_deps(APP_DEPS, st=st))


main()
