"""Registry for tab renderers and their dependency keys."""

from __future__ import annotations

from ui.ctx import get_ctx, require_keys

from tabs.backtest_tab import render as render_backtest_tab_ui
from tabs.correlation_tab import render as render_correlation_tab_ui
from tabs.fibonacci_tab import render as render_fibonacci_tab_ui
from tabs.guide_tab import render as render_guide_tab_ui
from tabs.heatmap_tab import render as render_heatmap_tab_ui
from tabs.market_tab import render as render_market_tab_ui
from tabs.ml_tab import render as render_ml_tab_ui
from tabs.monte_carlo_tab import render as render_monte_carlo_tab_ui
from tabs.multitf_tab import render as render_multitf_tab_ui
from tabs.portfolio_scenario_tab import render as render_portfolio_scenario_tab_ui
from tabs.position_tab import render as render_position_tab_ui
from tabs.risk_tab import render as render_risk_analytics_tab_ui
from tabs.scalp_backtest_tab import render as render_scalp_backtest_tab_ui
from tabs.sessions_tab import render as render_sessions_tab_ui
from tabs.spot_tab import render as render_spot_tab_ui
from tabs.setup_backtest_tab import render as render_setup_backtest_tab_ui
from tabs.tools_tab import render as render_tools_tab_ui
from tabs.whale_tab import render as render_whale_tab_ui


TAB_TITLES = [
    "Market", "Spot", "Position", "AI Workspace",
    "Heatmap", "Monte Carlo",
    "Fibonacci", "Risk Analytics", "Whale Tracker",
    "Multi-TF", "Correlation", "Portfolio Scenario",
    "Sessions", "Tools", "Model Lab", "Setup Backtest", "Scalp Backtest", "Analysis Guide",
]


_TAB_DEPS = [
    (
        render_market_tab_ui,
        [
            "ACCENT", "TEXT_MUTED", "CARD_BG", "POSITIVE", "NEGATIVE", "WARNING",
            "get_market_top_snapshot", "get_price_change",
            "_tip", "get_major_ohlcv_bundle", "ml_ensemble_predict", "get_top_volume_usdt_symbols",
            "fetch_ohlcv", "analyse", "get_scalping_entry_target", "scalp_quality_gate", "_calc_conviction",
            "signal_plain", "direction_key", "direction_label", "bias_score_badge", "readable_market_cap", "leverage_badge",
            "format_delta", "format_trend", "format_adx", "format_stochrsi",
            "style_signal", "style_scalp_opp", "style_delta", "sanitize_trading_terms", "_debug",
        ],
    ),
    (
        render_spot_tab_ui,
        [
            "ACCENT", "TEXT_MUTED", "POSITIVE", "NEGATIVE", "WARNING", "CARD_BG",
            "_tip", "_normalize_coin_input", "_validate_coin_symbol", "fetch_ohlcv", "analyse",
            "signal_plain", "direction_key", "direction_label", "format_delta",
            "format_stochrsi",
            "ml_ensemble_predict", "get_price_change", "_calc_conviction",
            "_wma", "_sr_lookback", "_debug",
        ],
    ),
    (
        render_position_tab_ui,
        [
            "ACCENT", "TEXT_MUTED", "POSITIVE", "NEGATIVE", "WARNING", "CARD_BG", "PRIMARY_BG",
            "_tip", "_normalize_coin_input", "_validate_coin_symbol", "_symbol_variants", "EXCHANGE",
            "fetch_ohlcv", "analyse", "signal_plain", "direction_key", "direction_label", "ml_ensemble_predict", "_calc_conviction",
            "format_delta", "format_stochrsi",
            "_sr_lookback", "_wma", "_debug", "get_scalping_entry_target", "scalp_quality_gate",
            "sanitize_trading_terms",
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
        ["ACCENT", "TEXT_MUTED", "POSITIVE", "NEGATIVE", "WARNING", "TEXT_LIGHT", "PRIMARY_BG", "NEON_BLUE", "_tip"],
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
        render_risk_analytics_tab_ui,
        [
            "ACCENT", "TEXT_MUTED", "POSITIVE", "NEGATIVE", "WARNING", "NEON_BLUE", "NEON_PURPLE",
            "PRIMARY_BG", "_tip", "_normalize_coin_input", "_validate_coin_symbol", "fetch_ohlcv",
            "calculate_risk_metrics",
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
        render_multitf_tab_ui,
        [
            "ACCENT", "TEXT_MUTED", "POSITIVE", "NEGATIVE", "WARNING", "CARD_BG", "_tip",
            "_normalize_coin_input", "_validate_coin_symbol", "fetch_ohlcv", "analyse", "signal_plain",
            "format_trend", "style_signal",
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
        render_sessions_tab_ui,
        [
            "ACCENT", "TEXT_MUTED", "WARNING", "POSITIVE", "NEGATIVE", "_normalize_coin_input",
            "_validate_coin_symbol", "fetch_ohlcv", "EXCHANGE", "readable_market_cap", "_tip",
        ],
    ),
    (
        render_tools_tab_ui,
        ["ACCENT", "TEXT_MUTED", "POSITIVE", "NEGATIVE", "WARNING", "_tip"],
    ),
    (
        render_backtest_tab_ui,
        [
            "ACCENT", "TEXT_MUTED", "POSITIVE", "WARNING", "_normalize_coin_input", "_validate_coin_symbol",
            "fetch_ohlcv", "run_backtest", "analyse",
        ],
    ),
    (
        render_setup_backtest_tab_ui,
        [
            "ACCENT", "TEXT_MUTED", "POSITIVE", "WARNING",
            "_normalize_coin_input", "_validate_coin_symbol", "fetch_ohlcv", "analyse",
            "ml_ensemble_predict", "signal_plain", "direction_key", "_calc_conviction",
        ],
    ),
    (
        render_scalp_backtest_tab_ui,
        [
            "ACCENT", "TEXT_MUTED", "POSITIVE", "WARNING",
            "fetch_ohlcv", "analyse",
            "ml_ensemble_predict", "signal_plain", "direction_key", "_calc_conviction",
            "get_scalping_entry_target", "scalp_quality_gate", "_sr_lookback",
            "get_top_volume_usdt_symbols",
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


def required_dep_keys() -> set[str]:
    """Return all dependency keys needed by the tab registry."""
    keys: set[str] = set()
    for _, dep_keys in _TAB_DEPS:
        keys.update(dep_keys)
    return keys
