import streamlit as st
import pandas as pd
import numpy as np
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
    fetch_top_gainers_losers,
    fetch_trending_coins,
    get_btc_eth_prices,
    get_fear_greed,
    get_major_ohlcv_bundle,
    get_market_indices,
    get_market_top_snapshot,
    get_price_change,
    get_scalping_entry_target,
    scalp_quality_gate,
    get_social_sentiment,
    get_top_volume_usdt_symbols,
    ml_ensemble_predict,
    ml_predict_direction,
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
    build_indicator_grid as _build_indicator_grid,
    calc_conviction as _calc_conviction,
    tip as _tip,
)
from ui.app_shell import render_app
from ui.deps_factory import build_app_deps
from ui.styles import app_css


def _fallback_direction_key(direction: str) -> str:
    d = str(direction or "").strip().upper()
    if d in {"UPSIDE", "LONG", "BUY", "BULLISH", "STRONG BUY"}:
        return "UPSIDE"
    if d in {"DOWNSIDE", "SHORT", "SELL", "BEARISH", "STRONG SELL"}:
        return "DOWNSIDE"
    return "NEUTRAL"


def _fallback_direction_label(direction: str) -> str:
    d = _fallback_direction_key(direction)
    if d == "UPSIDE":
        return "Upside"
    if d == "DOWNSIDE":
        return "Downside"
    return "Neutral"


def _fallback_signal_plain(signal: str) -> str:
    s = str(signal or "").strip().upper()
    if s in {"STRONG BUY", "BUY"}:
        return "LONG"
    if s in {"STRONG SELL", "SELL"}:
        return "SHORT"
    return "WAIT"


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


bias_score_badge = getattr(_ui_helpers, "bias_score_badge", _fallback_bias_score_badge)
format_adx = getattr(_ui_helpers, "format_adx", _fallback_format_adx)
format_delta = getattr(_ui_helpers, "format_delta", _fallback_format_delta)
format_stochrsi = getattr(_ui_helpers, "format_stochrsi", _fallback_format_stochrsi)
format_trend = getattr(_ui_helpers, "format_trend", _fallback_format_trend)
leverage_badge = getattr(_ui_helpers, "leverage_badge", _fallback_leverage_badge)
readable_market_cap = getattr(_ui_helpers, "readable_market_cap", _fallback_readable_market_cap)
direction_key = getattr(_ui_helpers, "direction_key", _fallback_direction_key)
direction_label = getattr(_ui_helpers, "direction_label", _fallback_direction_label)
signal_plain = getattr(_ui_helpers, "signal_plain", _fallback_signal_plain)
sanitize_trading_terms = getattr(_ui_helpers, "sanitize_trading_terms", lambda t: "" if t is None else str(t))
style_delta = getattr(_ui_helpers, "style_delta", lambda *_args, **_kwargs: "")
style_scalp_opp = getattr(_ui_helpers, "style_scalp_opp", lambda *_args, **_kwargs: "")
style_signal = getattr(_ui_helpers, "style_signal", lambda *_args, **_kwargs: "")


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

def _debug(msg: str) -> None:
    """Emit a debug message only when Debug mode is enabled."""
    if st.session_state.get('debug_mode', False):
        st.sidebar.write(msg)



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


def main():
    """Entry point for the Streamlit app."""

    render_app(build_app_deps(globals(), st=st))


main()
