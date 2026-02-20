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
    get_price_change,
    get_scalping_entry_target,
    get_social_sentiment,
    get_top_volume_usdt_symbols,
    ml_ensemble_predict,
    ml_predict_direction,
    monte_carlo_simulation,
)
from ui.helpers import (
    confidence_score_badge,
    format_adx,
    format_delta,
    format_stochrsi,
    format_trend,
    leverage_badge,
    readable_market_cap,
    signal_plain,
    style_confidence,
    style_delta,
    style_scalp_opp,
    style_signal,
)
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


def _wma(series: pd.Series, length: int) -> pd.Series:
    """Weighted Moving Average – gives more weight to recent prices."""
    weights = np.arange(1, length + 1, dtype=float)
    return series.rolling(window=length).apply(
        lambda x: np.dot(x, weights) / weights.sum(), raw=True
    )


# Set up page title, icon and wide layout
st.set_page_config(
    page_title="Crypto Command Center",
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
