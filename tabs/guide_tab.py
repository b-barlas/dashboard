"""Comprehensive beginner-friendly analysis guide."""

from __future__ import annotations

from typing import Iterable

from ui.ctx import get_ctx


SECTION_STYLE = {
    "core": "border-left:4px solid #06D6A0; background:rgba(6,214,160,0.06);",
    "risk": "border-left:4px solid #FFD166; background:rgba(255,209,102,0.08);",
    "warn": "border-left:4px solid #EF476F; background:rgba(239,71,111,0.08);",
    "info": "border-left:4px solid #00D4FF; background:rgba(0,212,255,0.08);",
}


def _panel(st, title: str, body: str, tone: str = "core") -> None:
    style = SECTION_STYLE.get(tone, SECTION_STYLE["core"])
    st.markdown(
        f"""
        <div class='panel-box' style='{style}'>
          <b style='color:#E5E7EB; font-size:1.15rem;'>{title}</b>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown(body)


def _render_sections(st, sections: Iterable[tuple[str, str, str]]) -> None:
    for title, body, tone in sections:
        _panel(st, title, body, tone)


def render(ctx: dict) -> None:
    st = get_ctx(ctx, "st")

    st.markdown("## Analysis Guide")
    st.caption(
        "Full user manual: what each tab does, how calculations are made, and how to interpret outputs without guesswork."
    )

    sections = [
        (
            "1) What this dashboard is",
            """
This app is a **decision-support dashboard** for crypto markets. It combines:
- Technical analysis (trend, momentum, volume, volatility)
- Machine learning signals (production default: Ensemble)
- Risk and backtest tooling
- Multi-tab workflow for scanning, validation, and execution planning

It does **not** execute trades. Final decision stays with the user.
            """,
            "info",
        ),
        (
            "2) Core signal engine (how BUY/SELL/WAIT is computed)",
            """
The signal engine computes 4 category scores in range **[-1, +1]**:
- Trend
- Momentum
- Volume
- Volatility

Weighted score:
`final_score = trend*0.40 + momentum*0.30 + volume*0.20 + volatility*0.10`

Confidence conversion:
`confidence = (final_score + 1) / 2 * 100`

Meaning:
- Near 100: strong bullish alignment
- Near 0: strong bearish alignment
- Around 50: mixed/no edge
            """,
            "core",
        ),
        (
            "3) Quality gates and adaptive thresholds",
            """
High confidence alone is not enough. Signals pass extra filters:
- Trend confirmation (avoid counter-trend forcing)
- Momentum sanity checks
- Volume confirmation
- Volatility/risk gating

Adaptive threshold logic (stricter in harder regimes):
- Normal market: less strict
- High volatility: stricter
- Weak trend (low ADX): strictest

If filters fail, output becomes **WAIT**.
            """,
            "risk",
        ),
        (
            "4) Indicator families used in analysis",
            """
Main indicators used across tabs:
- **Trend**: EMA stack, SuperTrend, Ichimoku, PSAR, ADX strength
- **Momentum**: RSI, MACD, StochRSI, Williams %R, CCI
- **Volume**: OBV, volume spike checks, VWAP context
- **Volatility**: ATR and Bollinger width

These are combined into category scores, then into final confidence.
            """,
            "core",
        ),
        (
            "5) AI architecture (important)",
            """
Production AI signal in main tabs uses **Ensemble ML**:
- Gradient Boosting (weight 45%)
- Random Forest (weight 35%)
- Logistic Regression (weight 20%)

Ensemble probability:
`p_ens = p_gb*0.45 + p_rf*0.35 + p_lr*0.20`

Direction mapping (ensemble):
- LONG if probability >= 0.58
- SHORT if probability <= 0.42
- NEUTRAL in the 0.42-0.58 zone

`AI Agree` = model agreement ratio inside the ensemble.
            """,
            "core",
        ),
        (
            "6) AI Lab tab (model diagnostics)",
            """
The **AI Lab** tab is for model comparison and diagnostics.
You can select:
- Ensemble
- Gradient Boosting
- Random Forest
- Logistic Regression

Use case:
- Compare model behavior across timeframes
- Inspect stability before trusting edge
- Keep production trading decisions aligned with Ensemble in main tabs
            """,
            "info",
        ),
        (
            "7) Market tab (scanner + macro view)",
            """
Market tab includes:
- Live BTC/ETH, market cap, Fear & Greed
- Dominance gauges (BTC/ETH)
- AI market outlook (dominance-weighted)
- Coin scanner table

Scanner table key columns:
- Signal
- Confidence
- Confidence Band
- AI Ensemble
- AI Agree
- AI Stability
- Conviction
- Setup
- Indicator snapshots (ADX, Ichimoku, StochRSI, VWAP, Candle Pattern, etc.)

Conviction is based on Signal + AI alignment + confidence quality.
            """,
            "core",
        ),
        (
            "8) Rapid tab (fast decision feed)",
            """
**Rapid** is a speed-focused tab for short-horizon opportunities.
It scans a liquid universe and ranks candidates with a single **Rapid Score (0-100)**.

Rapid Score combines:
- Confidence (30%)
- Setup quality (20%)
- AI quality: direction fit + agreement (20%)
- Trend quality via ADX (15%)
- Execution quality via R:R (15%)
- Penalty for weak/conflicting conviction

Outputs:
- Action: READY / WAIT / SKIP
- Direction, Score, Grade, Entry / SL / TP1, R:R
- "Why now?" bullets for quick context

Use Rapid for speed, then verify final execution discipline in Position tab.
            """,
            "core",
        ),
        (
            "9) Spot and Position tabs",
            """
**Spot tab**:
- Single-coin deep analysis for non-leveraged decisions
- Full indicator grid, chart overlays, sentiment proxy, technical snapshot

**Position tab**:
- Open-position context with direction-aware commentary
- Raw and levered PnL context + current signal regime
- Scalping setup, support/resistance distance, risk warnings
- Estimated liquidation price/distance (simple estimate)
- Net PnL view (funding impact) and a Hard Invalidation line
- Position Health Score with action bias (HOLD / REDUCE / EXIT)

Both use Ensemble AI for directional confirmation.
            """,
            "core",
        ),
        (
            "10) Screener tab",
            """
Screener scans predefined liquid symbols with filters:
- Min confidence
- Signal types
- Min ADX
- RSI range
- Optional volume-spike-only

Each matching row includes technical signal and Ensemble AI direction.
Use this to shortlist candidates, then validate in Spot/Position/Fibonacci.
            """,
            "core",
        ),
        (
            "11) Fibonacci, Monte Carlo, Risk Analytics",
            """
**Fibonacci tab**:
- Retracement/extension levels
- Divergence detection
- Volume profile (POC/value area)
- Market regime classification

**Monte Carlo tab**:
- Simulated future paths from historical return distribution
- Probability bands, expected return, VaR-like downside view

**Risk Analytics tab**:
- Sharpe, Sortino, Calmar
- Max Drawdown
- VaR and CVaR
- Distribution shape (skew/kurtosis)
            """,
            "core",
        ),
        (
            "12) Backtest tab",
            """
Backtest replays strategy logic on historical candles:
- Entry by signal+confidence rules
- Exit after configured hold window
- Commission/slippage assumptions

Read together, not in isolation:
- Win rate
- Profit factor
- Max drawdown
- Sharpe (risk-adjusted)

A high win rate with bad drawdown is not necessarily good.
            """,
            "risk",
        ),
        (
            "13) Tools tab (risk/reward + liquidation)",
            """
Tools help pre-trade planning:
- Risk/Reward calculator
- Position sizing context
- Leverage impact table
- Liquidation distance estimation

Always define risk before entry:
1. Stop-loss
2. Risk amount
3. Position size
4. Leverage only if needed
            """,
            "risk",
        ),
        (
            "14) Correlation, Sessions, Heatmap, Whale",
            """
- **Correlation**: co-movement matrix for diversification
- **Sessions**: behavior by Asia/Europe/US market windows
- **Heatmap**: market-wide breadth by cap and 24h move
- **Whale**: trending, gainers/losers, volume-pressure context

These tabs help with **context**, not standalone entries.
            """,
            "info",
        ),
        (
            "15) Leverage guidance",
            """
Leverage suggestions are ceilings, not targets.
Use lower leverage when:
- Confidence is medium
- ADX is weak
- Volatility is high
- Signal/AI disagree

Best practice:
- Risk 1-2% account per trade
- Avoid over-sizing from high-confidence overtrust
            """,
            "risk",
        ),
        (
            "16) Data sources and UK-safe policy",
            """
Primary exchange fallback list is intentionally UK-safe for this setup:
- Kraken
- Coinbase
- Bitstamp

If one source fails, app falls back to the next source.
Some market-wide datasets use CoinGecko.

When a live request temporarily fails/rate-limits in analysis tabs,
the UI may show the **last successful cached snapshot** with a UTC timestamp.
This is intentional to avoid blank panels during transient outages.
            """,
            "info",
        ),
        (
            "17) How to run with Streamlit",
            """
Run locally:
1. Install dependencies from `requirements.txt`
2. Start app: `streamlit run crypto_meta2.py`
3. Open local URL shown by Streamlit (usually `http://localhost:8501`)

If data looks stale:
- Use Streamlit "Clear cache"
- Refresh page
            """,
            "info",
        ),
        (
            "18) Practical workflow (recommended)",
            """
Recommended daily flow:
1. Market tab: check regime + scanner shortlist
2. Rapid tab: check READY candidates and pre-built plans
3. Spot: validate setup and read the Action Plan
4. Position: if already in trade, follow Hard Invalidation + Position Health first
5. Fibonacci/Risk: validate structure and downside risk
6. Tools: confirm R:R and liquidation distance
7. Backtest: validate settings before using new setup live

Quick rule:
- If Signal/AI conflict and Health says REDUCE or EXIT, reduce risk first.
- If setup is aligned and Health says HOLD, manage with invalidation discipline.
            """,
            "core",
        ),
        (
            "19) Limitations and responsibility",
            """
No model can predict news shocks, listing events, outages, or sudden regime breaks.
Treat all outputs as probabilistic guidance.

Non-negotiables:
- Use stop-loss
- Cap per-trade risk
- Avoid revenge/forced trades
- Respect invalidation levels

This dashboard is **not financial advice**.
            """,
            "warn",
        ),
        (
            "20) Quick Smoke Checklist (before daily use)",
            """
Use this 60-second checklist:

1. **Market tab**
- Scanner table loads and shows multiple rows
- Signal / AI / Setup columns are not empty

2. **Spot tab**
- Analyse runs and shows Signal + Confidence + AI + Conviction
- Spot Action Plan appears

3. **Rapid tab**
- Rapid table loads with Action / Score / Entry-SL-TP columns
- If no qualified rows, near-miss watchlist appears

4. **Position tab**
- Raw/Levered PnL and Net PnL render correctly
- Hard Invalidation and Position Health are visible
- Excel report downloads without resetting analysis view

5. **AI Lab**
- Predict fills timeframe matrix
- AI Entry and Non-AI Entry columns populate logically

6. **Fallback check**
- If any live endpoint fails, cached snapshot warning with UTC timestamp appears
            """,
            "info",
        ),
    ]

    _render_sections(st, sections)
