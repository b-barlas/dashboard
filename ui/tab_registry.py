"""Registry for tab renderers and their dependency keys."""

from __future__ import annotations

from ui.ctx import get_ctx, require_keys

from tabs.correlation_tab import render as render_correlation_tab_ui
from tabs.fibonacci_tab import render as render_fibonacci_tab_ui
from tabs.guide_tab import render as render_guide_tab_ui
from tabs.heatmap_tab import render as render_heatmap_tab_ui
from tabs.labs_tab import render as render_labs_tab_ui
from tabs.market_tab import render as render_market_tab_ui
from tabs.ml_tab import render as render_ml_tab_ui
from tabs.monte_carlo_tab import render as render_monte_carlo_tab_ui
from tabs.multitf_tab import render as render_multitf_tab_ui
from tabs.portfolio_scenario_tab import render as render_portfolio_scenario_tab_ui
from tabs.position_tab import render as render_position_tab_ui
from tabs.risk_tab import render as render_risk_analytics_tab_ui
from tabs.sessions_tab import render as render_sessions_tab_ui
from tabs.signal_review_tab import render as render_signal_review_tab_ui
from tabs.spot_tab import render as render_spot_tab_ui
from tabs.tools_tab import render as render_tools_tab_ui
from tabs.whale_tab import render as render_whale_tab_ui


TAB_TITLES = [
    "Market", "Spot", "Position", "Sessions", "Signal Archive",
    "Multi-TF", "AI Workspace", "Heatmap", "Whale Tracker", "Risk Analytics",
    "Monte Carlo", "Fibonacci", "Correlation", "Portfolio Scenario",
    "Tools", "Labs", "Analysis Guide",
]

TAB_GROUPS = [
    (
        "Core Trading Flow",
        ["Market", "Spot", "Position", "Sessions", "Signal Archive"],
    ),
    (
        "Research & Labs",
        [
            "Multi-TF", "AI Workspace", "Heatmap", "Whale Tracker", "Risk Analytics",
            "Monte Carlo", "Fibonacci", "Correlation", "Portfolio Scenario", "Labs",
        ],
    ),
    (
        "Tools & Reference",
        ["Tools", "Analysis Guide"],
    ),
]

_STYLE_DEPS = ["ACCENT", "TEXT_MUTED", "POSITIVE", "NEGATIVE", "WARNING"]
_CARD_STYLE_DEPS = [*_STYLE_DEPS, "CARD_BG"]
_TRACKER_DEPS = ["get_signal_tracker_db_path", "init_signal_tracker_db", "fetch_signal_events_df"]
_ADAPTIVE_DEPS = ["build_adaptive_context_model", "build_live_signal_adaptive_snapshot"]
_RECENT_MARKET_CONTEXT_DEPS = ["build_recent_market_context_snapshot", "build_recent_symbol_market_signal_snapshot"]
_SIGNAL_REVIEW_STORAGE_DEPS = [
    "build_tracker_storage_snapshot", "read_signal_tracker_db_bytes",
    "backup_signal_tracker_db", "restore_signal_tracker_db_bytes",
]


_TAB_DEPS = [
    (
        render_market_tab_ui,
        [
            *_CARD_STYLE_DEPS,
            "get_market_top_snapshot", "get_price_change",
            "_tip", "get_major_ohlcv_bundle", "ml_ensemble_predict", "get_top_volume_usdt_symbols",
            "fetch_top_gainers_losers", "fetch_trending_coins", "fetch_exchange_tickers_snapshot",
            "get_market_cap_rows_for_symbols",
            "build_market_regime_snapshot",
            "build_market_trade_gate",
            "build_signal_risk_sizing", "market_default_risk_budget",
            "build_sector_rotation_snapshot", "classify_symbol_sector",
            "build_market_flow_proxy_snapshot", "get_market_flow_proxy_rows",
            "build_market_alerts", "log_market_alerts",
            "build_market_catalyst_snapshot", "get_market_catalyst_events",
            "fetch_coingecko_ohlcv_by_coin_id",
            "fetch_ohlcv", "analyse", "get_scalping_entry_target", "scalp_quality_gate", "_calc_conviction",
            "signal_plain", "direction_key", "direction_label", "bias_score_badge", "readable_market_cap", "leverage_badge",
            "format_delta", "format_trend", "format_adx", "format_stochrsi",
            "style_signal", "style_scalp_opp", "style_delta", "sanitize_trading_terms", "_debug",
            "log_signal_events", "resolve_open_signal_events_for_frame",
            *_TRACKER_DEPS, *_ADAPTIVE_DEPS,
        ],
    ),
    (
        render_spot_tab_ui,
        [
            *_CARD_STYLE_DEPS,
            "_tip", "_normalize_coin_input", "_validate_coin_symbol", "fetch_ohlcv", "analyse",
            "signal_plain", "direction_key", "direction_label", "format_delta",
            "format_stochrsi",
            "ml_ensemble_predict", "get_price_change", "_calc_conviction",
            "_wma", "_sr_lookback", "_debug",
            *_TRACKER_DEPS, *_ADAPTIVE_DEPS, *_RECENT_MARKET_CONTEXT_DEPS,
        ],
    ),
    (
        render_position_tab_ui,
        [
            *_CARD_STYLE_DEPS, "PRIMARY_BG",
            "_tip", "_normalize_coin_input", "_validate_coin_symbol", "_symbol_variants", "EXCHANGE",
            "fetch_ohlcv", "analyse", "signal_plain", "direction_key", "direction_label", "ml_ensemble_predict", "_calc_conviction",
            "format_delta", "format_stochrsi",
            "_sr_lookback", "_wma", "_debug", "get_scalping_entry_target", "scalp_quality_gate",
            "sanitize_trading_terms",
            *_TRACKER_DEPS, *_ADAPTIVE_DEPS, *_RECENT_MARKET_CONTEXT_DEPS,
            "fetch_signal_forward_windows_df", "build_hold_window_intelligence",
            "classify_symbol_sector",
        ],
    ),
    (
        render_sessions_tab_ui,
        [
            "ACCENT", "TEXT_MUTED", "WARNING", "POSITIVE", "NEGATIVE", "_normalize_coin_input",
            "_validate_coin_symbol", "fetch_ohlcv", "EXCHANGE", "readable_market_cap", "_tip",
            "get_signal_tracker_db_path", "init_signal_tracker_db", "fetch_signal_events_df",
            "build_signal_cohort_summary", "build_recent_market_context_snapshot",
        ],
    ),
    (
        render_signal_review_tab_ui,
        [
            *_STYLE_DEPS, "_tip", "fetch_ohlcv", "resolve_open_signal_events_via_fetch", "backfill_signal_forward_windows_via_fetch",
            *_TRACKER_DEPS, "fetch_market_alerts_df", "count_market_alerts",
            "build_signal_review_snapshot", "build_execution_overlay_snapshot", "build_signal_cohort_summary", "build_adaptive_context_model",
            "fetch_signal_forward_windows_df", "build_hold_window_intelligence", "build_hold_window_cohort_summary",
            "annotate_alert_footprint", "build_alert_effectiveness_summary",
            "build_learning_edge_table", "save_signal_trade_overlay", "save_signal_trade_journal",
            *_SIGNAL_REVIEW_STORAGE_DEPS,
        ],
    ),
    (
        render_multitf_tab_ui,
        [
            "ACCENT", "TEXT_MUTED", "POSITIVE", "NEGATIVE", "WARNING",
            "_normalize_coin_input", "_validate_coin_symbol", "fetch_ohlcv", "analyse",
            "direction_key", "direction_label", "format_delta", "format_stochrsi",
            "ml_ensemble_predict", "_calc_conviction",
            *_TRACKER_DEPS,
        ],
    ),
    (
        render_ml_tab_ui,
        [
            "ACCENT", "TEXT_MUTED", "POSITIVE", "NEGATIVE", "WARNING", "CARD_BG", "_tip",
            "_normalize_coin_input", "_validate_coin_symbol", "fetch_ohlcv", "ml_ensemble_predict",
            "get_scalping_entry_target", "scalp_quality_gate", "_calc_conviction", "analyse",
            "_debug",
            "get_major_ohlcv_bundle", "get_market_indices",
        ],
    ),
    (
        render_heatmap_tab_ui,
        [
            "ACCENT", "TEXT_MUTED", "POSITIVE", "NEGATIVE", "WARNING",
            "TEXT_LIGHT", "PRIMARY_BG", "NEON_BLUE", "_tip", "get_heatmap_rows",
        ],
    ),
    (
        render_whale_tab_ui,
        [
            "ACCENT", "TEXT_MUTED", "TEXT_LIGHT", "POSITIVE", "NEGATIVE", "WARNING", "NEON_BLUE",
            "GOLD", "_tip", "fetch_trending_coins", "fetch_top_gainers_losers", "fetch_ohlcv",
            "get_top_volume_usdt_symbols",
        ],
    ),
    (
        render_risk_analytics_tab_ui,
        [
            "ACCENT", "TEXT_MUTED", "POSITIVE", "NEGATIVE", "WARNING", "NEON_BLUE", "NEON_PURPLE",
            "PRIMARY_BG", "_tip", "_normalize_coin_input", "_validate_coin_symbol", "fetch_ohlcv",
            "calculate_risk_metrics",
        ],
    ),
    (
        render_monte_carlo_tab_ui,
        [
            "ACCENT", "TEXT_MUTED", "TEXT_LIGHT", "POSITIVE", "NEGATIVE", "WARNING", "NEON_BLUE",
            "NEON_PURPLE", "PRIMARY_BG", "_tip", "_normalize_coin_input", "_validate_coin_symbol",
            "fetch_ohlcv", "monte_carlo_simulation",
        ],
    ),
    (
        render_fibonacci_tab_ui,
        [
            "ACCENT", "TEXT_MUTED", "POSITIVE", "NEGATIVE", "WARNING", "GOLD", "PRIMARY_BG", "_tip",
            "_normalize_coin_input", "_validate_coin_symbol", "fetch_ohlcv", "calculate_fibonacci_levels",
            "detect_divergence", "calculate_volume_profile", "detect_market_regime",
        ],
    ),
    (
        render_correlation_tab_ui,
        ["ACCENT", "TEXT_MUTED", "_tip", "_normalize_coin_input", "fetch_ohlcv", "EXCHANGE"],
    ),
    (
        render_portfolio_scenario_tab_ui,
        [
            "ACCENT", "TEXT_MUTED", "POSITIVE", "NEGATIVE", "WARNING", "_tip",
            "_normalize_coin_input", "_validate_coin_symbol", "fetch_ohlcv",
        ],
    ),
    (
        render_tools_tab_ui,
        ["ACCENT", "TEXT_MUTED", "POSITIVE", "NEGATIVE", "WARNING", "_tip"],
    ),
    (
        render_labs_tab_ui,
        [
            "ACCENT", "TEXT_MUTED", "POSITIVE", "WARNING",
            "_normalize_coin_input", "_validate_coin_symbol", "fetch_ohlcv", "analyse",
            "ml_ensemble_predict", "signal_plain", "direction_key", "_calc_conviction",
            "get_scalping_entry_target", "scalp_quality_gate", "_sr_lookback",
            "get_top_volume_usdt_symbols", "run_backtest", "_tip",
            "get_signal_tracker_db_path", "init_signal_tracker_db", "fetch_signal_events_df",
            "fetch_signal_forward_windows_df", "build_hold_window_intelligence",
            "build_signal_review_snapshot", "build_execution_overlay_snapshot", "build_signal_cohort_summary",
            "save_signal_trade_overlay", "save_signal_trade_journal",
        ],
    ),
    (render_guide_tab_ui, []),
]


def build_tab_specs(deps: dict) -> list[tuple]:
    """Build render specs as (renderer, context) from dependency keys."""
    require_keys(deps, required_dep_keys(), scope="tab_registry.deps")
    specs = []
    for renderer, keys in _TAB_DEPS:
        specs.append((renderer, {k: get_ctx(deps, k, scope="tab_registry.deps") for k in keys}))
    return specs


def build_tab_spec(deps: dict, title: str) -> tuple:
    """Build one render spec for the active dashboard tab."""
    title_text = str(title or "").strip()
    if title_text not in TAB_TITLES:
        title_text = TAB_TITLES[0]
    idx = TAB_TITLES.index(title_text)
    renderer, keys = _TAB_DEPS[idx]
    require_keys(deps, set(keys), scope=f"tab_registry.deps.{title_text}")
    return renderer, {k: get_ctx(deps, k, scope=f"tab_registry.deps.{title_text}") for k in keys}


def required_dep_keys() -> set[str]:
    """Return all dependency keys needed by the tab registry."""
    keys: set[str] = set()
    for _, dep_keys in _TAB_DEPS:
        keys.update(dep_keys)
    return keys
