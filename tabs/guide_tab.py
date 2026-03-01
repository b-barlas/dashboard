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
        "Full user manual in dashboard tab order: what each tab does, how core calculations work, and how to read outputs clearly."
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
- `strength = (abs(bias - 50) / 50) ^ 0.70 * 100`  (0-100, direction-agnostic)

Strength buckets used in UI:
- 0-39: Weak
- 40-59: Mixed
- 60-74: Good
- 75-100: Strong

Direction comes from technical signal mapping:
- Upside / Downside / Neutral

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
            """
This is the primary scan tab. Start here.

Main table columns:
- Coin, Price ($), Delta (%)
- Setup Confirm
- Direction
- Strength
- AI Ensemble
- Tech vs AI Alignment
- R:R, Entry/Stop/Target, Scalp Opportunity
- Optional advanced indicator columns

Scanner input modes:
- Default mode: Top N liquidity scan (market-wide)
- Custom mode: enter up to 10 symbols in Custom Coins, click Run Scan, and scanner analyzes only that watchlist
- Top N control is disabled while custom mode is active
- Selected timeframe controls candle context used for Direction/Strength, levels, and Delta

Setup Confirm classes (final confirmation class):
- CONFIRMED (Trend+AI): strongest class (trend and AI align)
- CONFIRMED (Trend-Led): trend leads; AI works as guardrail
- CONFIRMED (AI-Led): AI leads; trend works as guardrail
- WATCH: setup exists but confirmation incomplete
- SKIP: no direction, conflict, or weak edge

Scalp Opportunity is separate from Setup Confirm.
It appears only if all execution gates pass:
- Direction match
- Timeframe-adaptive R:R / ADX / Strength thresholds
- No tech/AI conflict
- Valid entry/stop/target

Timeframe gate matrix:
- 5m / 15m: R:R >= 1.30, ADX >= 18, Strength >= 52
- 1h: R:R >= 1.40, ADX >= 18, Strength >= 52
- 4h / 1d: R:R >= 1.70, ADX >= 22, Strength >= 60

Important consistency rule:
- Direction/Strength and plan levels are computed from **closed candles**
- Price column also reflects closed-candle close in scanner context

Mode badges:
- FULL MARKET MODE: exchange + enrichment providers active
- EXCHANGE-ONLY MODE: enrichment unavailable, core trade fields still active
            """,
            "core",
        ),
        (
            "2) Spot tab",
            """
Single-coin analysis for spot execution planning.
What to read:
- Direction + Strength + AI Ensemble + Tech vs AI Alignment (signal quality)
- Indicator strip (SuperTrend, Ichimoku, VWAP, ADX, Bollinger, StochRSI, PSAR, Williams %R, CCI, Volatility)
- Current Price / Breakout Entry / Exit If Broken cards
- Spot Execution Plan text (scenario-based execution steps)
            """,
            "core",
        ),
        (
            "3) Position tab",
            """
Live position management tab.
Main outputs:
- Raw PnL, levered PnL, funding effect, net PnL
- Estimated liquidation distance (simple model)
- Direction / Strength / AI Ensemble / Alignment summary
- Technical Invalidation Line (hard risk line)
- Position health decision block (HOLD / REDUCE / EXIT style guidance)
- Optional scalp setup block with gate reasons when unavailable
            """,
            "core",
        ),
        (
            "4) AI Lab tab",
            """
Model diagnostics across multiple timeframes for one coin.
Shows:
- Selected model direction/probability
- Ensemble agreement (x/3)
- AI Direction Bias (same market-wide bias logic as Market tab)
- Plan Entry/Target + plan source (AI-filtered vs technical fallback)

Use AI Lab for diagnostics and model behavior checks, not as a standalone execution trigger.
            """,
            "core",
        ),
        (
            "5) Ensemble AI tab",
            """
Pure ensemble prediction view for one coin/timeframe.
Shows:
- Ensemble direction and probability
- Effective agreement (directional or consensus-aware)
- Individual model outputs
- Confidence gauge

If training window is unstable/insufficient, the tab intentionally returns a neutral fallback output.
            """,
            "core",
        ),
        (
            "6) Heatmap tab",
            """
Top-coin market breadth view.
Shows:
- Cap-weighted heatmap
- Breadth (advancers share), A/D ratio, average change
- Quick read of market leadership concentration
            """,
            "info",
        ),
        (
            "7) Monte Carlo tab",
            """
Scenario simulation from historical return behavior.
Use it for:
- Distribution-aware expectation
- Downside path stress context
- Probability-band interpretation (not certainty)
            """,
            "risk",
        ),
        (
            "8) Fibonacci tab",
            """
Structure and zone-mapping tab.
Main outputs:
- Retracement/extension levels
- Divergence checks
- Volume-profile context (including POC area)
- Regime hints and action hints
            """,
            "core",
        ),
        (
            "9) Risk Analytics tab",
            """
Portfolio-style risk metrics from recent return series.
Includes:
- Sharpe / Sortino / Calmar
- Max drawdown
- VaR / CVaR
- Skew / Kurtosis

Use this tab to understand risk shape, not to create entry signals alone.
            """,
            "risk",
        ),
        (
            "10) Whale Tracker tab",
            """
Attention/liquidity proxy tab (not on-chain wallet tracking).
Combines:
- Trending coins feed
- Top gainers / losers snapshots
- Volume anomaly scanner with ratio + z-score logic

Use it to find attention shifts, then validate entries in Market/Spot/Position.
            """,
            "info",
        ),
        (
            "11) Screener tab",
            """
Rule-based shortlist tab over liquid symbols.
Typical filters:
- Minimum strength
- Direction type
- ADX floor
- RSI window
- Optional spike-only mode

Use Screener as pre-filter, then confirm execution context in Market/Spot/Position.
            """,
            "core",
        ),
        (
            "12) Multi-TF tab",
            """
Cross-timeframe alignment view for a single coin.
Shows:
- Direction status by timeframe
- Strength status by timeframe
- ADX status by timeframe

Purpose: avoid taking a low-timeframe setup fully against higher-timeframe structure.
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
            "14) Sessions tab",
            """
Session behavior split (Asia / Europe / US windows).
Shows:
- session-level return and volatility context
- volume profile by session
- quick quality cues
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
            "16) Backtest tab",
            """
Historical replay for strategy validation.
Core reading order:
1) trade count and sample quality
2) win rate + profit factor
3) drawdown and risk-adjusted metrics

A high win rate alone is not enough if drawdown/risk quality is poor.
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

This means trade-critical fields (price, candles, indicators, Setup Confirm/Direction/Strength)
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
            """
Recommended daily flow:
1. Market tab: check regime + scanner shortlist
2. Spot: validate setup and read the Spot Execution Plan
3. Position: if already in trade, follow Technical Invalidation + decision model first
4. Fibonacci/Risk: validate structure and downside risk
5. Tools: confirm R:R and liquidation distance
6. Backtest: validate settings before using new setup live

Quick rule:
- If Direction/AI conflict and Health says REDUCE or EXIT, reduce risk first.
- If setup is aligned and Health says HOLD, manage with invalidation discipline.
            """,
            "core",
        ),
        (
            "21) Limitations and responsibility",
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
            "22) Quick Smoke Checklist (before daily use)",
            """
Use this 60-second checklist:

1. **Market tab**
- Scanner table loads and shows multiple rows
- Setup Confirm / Direction / AI columns are not empty

2. **Spot tab**
- Analyse runs and shows Direction + Strength + AI + Tech vs AI Alignment
- Spot Execution Plan appears

3. **Position tab**
- Raw/Levered PnL and Net PnL render correctly
- Technical Invalidation line is visible
- Excel report downloads without resetting analysis view

4. **AI Lab**
- Predict fills timeframe matrix
- Plan Entry / Plan Target / Plan Source columns populate logically
- Debug expander shows AI vs non-AI plan levels

5. **Fallback check**
- If enrichment fails, Market shows **EXCHANGE-ONLY MODE** and core trade columns still render
- If live endpoint fails, cached snapshot warning with UTC timestamp appears
- Market should not reuse stale snapshot from a different timeframe/filter
            """,
            "info",
        ),
    ]

    _render_sections(st, sections)
