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
- TREND+AI: strongest class (trend and AI align)
- TREND-led: trend leads; AI works as guardrail
- AI-led: AI leads; trend works as guardrail
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
Single-coin spot decision workspace (non-leverage), synchronized with Market tab decision logic.

Read in this order:
1) Setup Snapshot:
- Delta (%)
- Setup Confirm
- Direction
- Strength
- AI Ensemble
- Tech vs AI Alignment

Setup Confirm classes:
- TREND+AI: strongest confirmation class
- TREND-led: trend leads, AI acts as guardrail
- AI-led: AI leads, trend acts as guardrail
- WATCH: setup exists, confirmation incomplete
- SKIP: no actionable setup

2) Technical Regime Breakdown:
- Trend Structure: SuperTrend, Ichimoku, VWAP, ADX, PSAR
- Momentum Signals: StochRSI, Williams %R, CCI, Pattern
- Volatility & Volume: Bollinger, Volatility, Volume spike context

3) Execution Levels (spot-only):
- Reference Price
- Buy Zone + Buy Above (Breakout)
- Stop (Buy Zone) + Stop (Breakout)
- Take-Profit (Buy Zone) + Take-Profit (Breakout)

4) Spot Execution Plan:
- Scenario-specific action text driven by Setup Confirm + Direction context
- Use it as execution workflow guidance, not as a guaranteed outcome

Important:
- Decision fields and levels are based on closed candles.
- Spot tab uses the same core signal/decision engine as Market tab.
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
            "4) AI Workspace tab",
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
            "5) Heatmap tab",
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
            "6) Monte Carlo tab",
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
            "7) Fibonacci tab",
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
            "8) Risk Analytics tab",
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
            "10) Screener tab",
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
            "11) Multi-TF tab",
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
            "12) Correlation tab",
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
            "13) Sessions tab",
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
            "14) Tools tab",
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
            "15) Model Lab tab",
            """
Signal-engine diagnostics backtest.
Use it to calibrate:
- strength threshold
- fixed holding bars
- fee/slippage assumptions

This mode validates the raw Direction + Strength engine, not the full setup class model.
            """,
            "risk",
        ),
        (
            "16) Setup Backtest tab",
            """
Setup outcome study for Setup Confirm:
- TREND+AI
- TREND-led
- AI-led

Inputs:
- setup class filter
- timeframe
- lookback candles
- forward bars (for outcome window)

Outputs:
- how many events were generated
- event price and forward path over next N bars
- class-level favorable rate and directional return quality
            """,
            "risk",
        ),
        (
            "17) Scalp Backtest tab",
            """
Scalp outcome study for execution-ready scalp events.

Uses the same market scalp pipeline:
- top-volume multi-coin scan (stablecoins excluded by default)
- scalp gate (timeframe-adaptive R:R / ADX / Strength thresholds)
- generated scalp levels (entry / stop / target)

Outputs:
- event count and TP-hit behavior
- direction-level and coin-level scalp outcome quality
- forward path table with event-relative price movement

Use this tab to validate scalp execution behavior before relying on scalp labels live.
            """,
            "risk",
        ),
        (
            "18) Analysis Guide tab (this page)",
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
            "19) Data sources, fallback policy, and UK-safe exchanges",
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
            "20) How to run with Streamlit",
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
            "21) Practical workflow (recommended)",
            """
Recommended daily flow:
1. Market tab: check regime + scanner shortlist
2. Spot: validate setup and read the Spot Execution Plan
3. Position: if already in trade, follow Technical Invalidation + decision model first
4. Fibonacci/Risk: validate structure and downside risk
5. Tools: confirm R:R and liquidation distance
6. Setup Backtest: validate Setup Confirm class edge before using new setup live
7. Scalp Backtest: validate scalp gate behavior and TP/SL outcome profile
8. Model Lab: tune raw signal threshold/holding parameters

Quick rule:
- If Direction/AI conflict and Health says REDUCE or EXIT, reduce risk first.
- If setup is aligned and Health says HOLD, manage with invalidation discipline.
            """,
            "core",
        ),
        (
            "22) Limitations and responsibility",
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
            "23) Quick Smoke Checklist (before daily use)",
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

4. **AI Workspace**
- Quick Prediction mode returns Direction / Probability / Agreement in one run
- Model & Timeframe Matrix mode fills timeframe matrix with plan source fields
- Debug expander shows AI vs non-AI plan levels

5. **Fallback check**
- If enrichment fails, Market shows **EXCHANGE-ONLY MODE** and core trade columns still render
- If live endpoint fails, cached snapshot warning with UTC timestamp appears
- Market should not reuse stale snapshot from a different timeframe/filter

6. **Model Lab + Setup/Scalp Backtest**
- Model Lab runs and returns trades/metrics for raw signal diagnostics
- Setup Backtest returns setup-event count, forward-bar outcome table, and class breakdown
- Scalp Backtest returns scalp-qualified event count, class outcome table, and TP/SL behavior
            """,
            "info",
        ),
    ]

    _render_sections(st, sections)
