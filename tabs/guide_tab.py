"""Comprehensive beginner-friendly analysis guide."""

from __future__ import annotations

from typing import Iterable

from core.trading_copy import copy_text
from ui.ctx import get_ctx
from ui.primitives import render_page_header


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

    render_page_header(
        st,
        title="Analysis Guide",
        intro_html=(
            "Full user manual in dashboard tab order: what each tab does, how core calculations work, and how to read outputs clearly."
        ),
    )

    sections = [
        (
            "0) Core Engine (used across tabs)",
            """
This dashboard is a decision-support system (it does **not** place trades).

Technical engine computes 4 category blocks in range **[-1, +1]**:
- Trend
- Momentum
- Volume
- Volatility

Those are combined into a final score, then converted:
- `bias = (final_score + 1) / 2 * 100`  (0-100)
- `bias_confidence = (abs(bias - 50) / 50) ^ 0.70 * 100`  (0-100, direction-agnostic quality score)

Execution confidence buckets:
- 0-39: Very Low
- 40-59: Low
- 60-79: Medium
- 80-100: High

Spot-facing tabs now separate:
- `Direction`: higher-timeframe technical bias from closed adaptive lead/confirm anchors
- `Confidence`: quality score for that spot direction
- `AI Ensemble`: higher-timeframe AI bias from a separate AI engine
- `AI Confidence`: quality score for that higher-timeframe AI verdict

Ensemble AI (3 models) is used as confirmation:
- Gradient Boosting 45%
- Random Forest 35%
- Logistic Regression 20%
- Direction thresholds: Upside if prob >= 0.58, Downside if prob <= 0.42, else Neutral
            """,
            "core",
        ),
        (
            "1) Market tab (Coin Setup Scanner)",
            copy_text("guide.section.market"),
            "core",
        ),
        (
            "2) Spot tab",
            copy_text("guide.section.spot"),
            "core",
        ),
        (
            "3) Position tab",
            copy_text("guide.section.position"),
            "core",
        ),
        (
            "4) Sessions tab",
            copy_text("guide.section.sessions"),
            "info",
        ),
        (
            "5) Signal Archive tab",
            copy_text("guide.section.signal_archive"),
            "info",
        ),
        (
            "6) Multi-TF tab",
            """
Single-coin timeframe alignment view across 5m / 15m / 1h / 4h / 1d.
Use it to:
- check whether short-term timing agrees with higher-timeframe structure
- see which layer is leading (timing vs higher timeframe)
- confirm the same spot-bias read already seen in Market / Spot / Position

Read order:
- Higher-TF Bias first (1h / 4h / 1d)
- Weighted Alignment second
- Timing Layer third (5m / 15m)
- Confidence last, as the timeframe-level quality read for that row

Higher timeframes carry more weight because they usually define the stronger structural regime.
Neutral timeframes intentionally dilute the weighted score, so high alignment really means broad agreement.
A directional bias only prints when weighted agreement is broad enough. Below 60%, the tab stays neutral on purpose.
If coverage is partial, treat the read as lower-confidence even when the visible timeframes agree.
This tab validates alignment; it is not a separate trade decision engine.
            """,
            "core",
        ),
        (
            "7) AI Workspace tab",
            """
Single AI tab with 2 modes:
- Quick Prediction: one coin/timeframe fast ensemble output
- Model & Timeframe Matrix: compare direction/probability/agreement across up to 3 timeframes

Shows:
- Direction (Upside/Downside/Neutral)
- Probability
- Effective agreement (x/3)
- AI Direction Bias (same market-wide bias logic as Market tab)
- Plan Entry/Target + Plan Source (AI-filtered vs technical fallback)

Use this tab for model diagnostics and confirmation quality checks, not as a standalone execution trigger.
            """,
            "core",
        ),
        (
            "8) Heatmap tab",
            """
Top-coin market breadth view.
Shows:
- Cap-weighted heatmap
- Breadth (advancers share), A/D ratio, average change
- Quick read of market leadership concentration
- Provider-aware feed state (LIVE/CACHED) with fallback chain
- Stablecoin exclusion toggle (default ON for cleaner beta read)
- Flat classification with a ±0.05% dead-band to reduce micro-noise

Provider order:
- CoinGecko (primary)
- CoinPaprika (fallback)
- Last-good snapshot (temporary cache fallback)
            """,
            "info",
        ),
        (
            "9) Whale Tracker tab",
            """
Attention/liquidity proxy tab (not on-chain wallet tracking).
Combines:
- Trending coins feed
- Top gainers / losers snapshots
- Volume anomaly scanner with adaptive ratio + z-score logic

Scanner notes:
- Uses closed candles (reduces partial-candle noise)
- Trigger = Ratio OR Z-Score threshold; EXTREME = dual confirmation
- Distinguishes no-data scan from no-anomaly scan

Use it to find attention shifts, then validate entries in Market/Spot/Position.
            """,
            "info",
        ),
        (
            "10) Risk Analytics tab",
            """
Portfolio-style risk metrics from recent return series.
Includes:
- Sharpe / Sortino / Calmar
- Max drawdown
- VaR / CVaR
- Skew / Kurtosis
- Timeframe-aware risk regime labels (1h / 4h / 1d)
- Closed-candle evaluation (reduces live-candle noise)

Use this tab to understand risk shape, not to create entry signals alone.
            """,
            "risk",
        ),
        (
            "11) Monte Carlo tab",
            """
Scenario-risk simulation from historical return behavior.
Use it for:
- Distribution-aware expectation
- Tail-risk stress context (VaR/CVaR)
- Probability-band interpretation (not certainty)

How it works:
- Horizon is converted into timeframe steps (1h/4h/1d)
- Path generation blends empirical bootstrap shocks + Gaussian shocks
- Metric bands are horizon-aware (short vs long horizon risk context)

Primary outputs:
- Profit Probability
- Expected Return
- VaR 95%
- CVaR 95%
- Median Target

Read this tab as a risk-planning layer (position sizing / scenario quality),
not as a direct entry trigger.
            """,
            "risk",
        ),
        (
            "12) Fibonacci tab",
            """
Decision-oriented structure/zone map tab.
Main outputs:
- Setup Quality score (Execution-level Distance + Regime + POC - Net Divergence)
- Bands: STRONG >= 68, MODERATE 45-67, WEAK < 45
- Retracement/extension execution levels (38.2/50/61.8 first, 100/161.8 as extension map)
- Divergence warnings (deduplicated + direction-aware impact: conflict penalizes more than supportive)
- Volume-profile context (POC distance)

Important:
- Uses closed-candle context for both calculations and chart (reduced live-candle noise)
- Use this tab to validate structure quality before triggering execution in Spot/Position
            """,
            "core",
        ),
        (
            "13) Correlation tab",
            """
Co-movement matrix for selected symbols.
Use it to:
- avoid accidental concentration
- separate highly correlated positions
- build better diversification logic
            """,
            "info",
        ),
        (
            "14) Portfolio Scenario tab",
            """
Anchor-based basket scenario engine.
Use it to:
- enter the coins you already hold and their current dollar values
- choose an anchor coin (for example BTC or ETH) plus a target price
- see how the basket may react if that anchor reaches target

It is a scenario projection layer, not a guarantee or prediction.
The engine estimates a typical holding horizon from the anchor distance and the anchor's usual bar speed,
then uses historical same-timeframe return relationships over that horizon as a linear scenario approximation.
It still does not simulate the full path or exact timing of the move.
Only the first 10 valid holdings are modeled.
Duplicate rows are merged by symbol, and invalid or non-positive rows are ignored.
If the estimated horizon is too long for stable modeling or longer than the available history,
the engine caps that horizon and shows a warning in the tab.
            """,
            "info",
        ),
        (
            "15) Tools tab",
            """
Beginner-friendly pre-trade calculator.
Inputs:
- Entry, Stop Loss, Take Profit
- Margin used, leverage, funding rate

Outputs:
- Notional, position size, R:R
- TP/SL PnL projections
- simplified liquidation estimate and leverage comparison
            """,
            "risk",
        ),
        (
            "16) Labs tab",
            """
Research and simulation workspace with two sub-tabs:
- Setup Lab
- Scalp Lab

Use Setup Lab to compare historical setup-class behavior.
Use Scalp Lab to read live scalp archive truth first, then test the scalp planner on historical candles.
Use Signal Archive for the real logged tracker history and execution journal.
            """,
            "risk",
        ),
        (
            "17) Analysis Guide tab (this page)",
            """
This guide mirrors the live dashboard behavior.
Use it as:
- quick reference for each tab
- plain-language explanation for key columns and calculations
- consistency check before production use
            """,
            "info",
        ),
        (
            "18) Data sources, fallback policy, and UK-safe exchanges",
            """
Primary exchange fallback list is intentionally UK-safe for this setup:
- Kraken
- Coinbase
- Bitstamp

Scanner/enrichment provider order (when building market universe):
1. CoinGecko markets feed (volume-ranked symbols + enrichment)
2. CoinPaprika tickers fallback (volume-ranked symbols + enrichment)
3. Exchange-pair fallback (symbols only)

Execution/indicator data order:
1. Exchange OHLCV/ticker from the active UK-safe exchange
2. If exchange fails, app falls back to the next UK-safe exchange

This means trade-critical fields (price, candles, indicators, Setup Confirm/Direction/Confidence)
remain exchange-driven even when enrichment providers are rate-limited.
Enrichment fields (for example Market Cap) may show as `—` in exchange-only mode.

Cache policy:
- Market tab: cached snapshot is used only for the **same timeframe/filter signature**
  (stale cache from another setting is intentionally blocked).
- Other analysis tabs: live-or-snapshot fallback is used with TTL guards.
  Typical cache TTL is 15 minutes for live analysis tabs and 30 minutes for backtest frames.
            """,
            "info",
        ),
        (
            "19) How to run with Streamlit",
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
            "20) Practical workflow (recommended)",
            copy_text("guide.section.workflow"),
            "core",
        ),
        (
            "21) Limitations and responsibility",
            copy_text("guide.section.limitations"),
            "warn",
        ),
        (
            "22) Quick Smoke Checklist (before daily use)",
            copy_text("guide.section.smoke_checklist"),
            "info",
        ),
    ]

    _render_sections(st, sections)
