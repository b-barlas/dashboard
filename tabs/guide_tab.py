"""Comprehensive beginner-friendly analysis guide."""

from __future__ import annotations

from typing import Iterable

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
- `Direction`: higher-timeframe technical bias from closed `1D + 4H` candles
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
            """
This is the primary scan tab. Start here.

Main table columns:
- Coin, Price ($), Delta (%)
- Setup Confirm
- Direction
- Confidence
- AI Ensemble
- AI Confidence
- R:R, Entry/Stop/Target, Scalp Opportunity
- Optional advanced indicator columns

Scanner input modes:
- Default mode: Top N liquidity scan (market-wide)
- Custom mode: enter up to 10 symbols in Custom Coins, click Run Scan, and scanner analyzes only that watchlist
- Top N control is disabled while custom mode is active
- Custom watchlist mode does not depend on the live top-volume provider universe; it scans requested symbols directly
- Selected timeframe controls tactical candle context, levels, scalp gating, and Delta
- Visible `Direction` + `Confidence` come from closed `1D + 4H` spot bias
- Visible `AI Ensemble` comes from a separate closed `1D + 4H` AI bias engine
- Visible `AI Confidence` scores the quality of that HTF AI verdict

How the 5 key columns are calculated:

1. `Direction` (main spot direction)
- Uses only closed `1D + 4H` candles
- Technical engine builds a score for each timeframe from:
  - structure
  - trend
  - momentum
  - regime / location
- `1D` leads and `4H` confirms
- Final logic is intentionally strict:
  - if `1D` is Neutral, final Direction becomes Neutral
  - if `1D` and `4H` conflict, final Direction becomes Neutral
  - otherwise final output is `Upside`, `Downside`, or `Neutral`

2. `Confidence` (quality of Direction)
- This is **not** a second direction column
- It answers: “How trustworthy is the current technical Direction?”
- Formula:
  - 30% timeframe alignment
  - 25% structure quality
  - 20% trend quality
  - 15% regime quality
  - 10% location quality
- Confidence is capped lower on purpose when:
  - Direction is Neutral
  - timeframes conflict
  - structure is weak
  - data is degraded
  - regime is range/chop

3. `AI Ensemble` (AI version of spot direction)
- Uses a separate AI engine on closed `1D + 4H` candles
- It does **not** copy the technical Direction formula
- For each timeframe, AI quality comes from:
  - 35% probability edge
  - 25% agreement quality
  - 25% persistence across recent closed bars
  - 15% stability (not flip-flopping)
- `1D` leads and `4H` confirms
- Final output is again `Upside`, `Downside`, or `Neutral`
- The 3 dots show how many of the 3 AI models support that final HTF AI verdict

4. `AI Confidence` (quality of AI Ensemble)
- This answers: “How trustworthy is the current HTF AI verdict?”
- Formula:
  - 40% conviction quality
  - 25% combined HTF AI score
  - 15% timeframe alignment
  - 10% consensus quality
  - 10% model support
- AI Confidence is capped lower when:
  - AI verdict is Neutral
  - `1D` / `4H` AI conflict
  - AI data degraded/fallback
- directional call has weak model support

Coin inline badge:
- `LEAD ↗` / `LEAD ↘` can appear next to the ticker when 4H technical structure is already leading, while confirmed HTF Direction is still Neutral or still low-confidence
- AI can confirm the same side or stay neutral; a strong opposite AI blocks the badge
- `1D` must not be opposing yet
- Treat this as an early-attention signal, not as a replacement for confirmed Direction

5. `Setup Confirm` (final confirmation class)
- This is **not** the main direction
- It answers: “Given the main spot direction, is the selected timeframe good enough right now?”
- First, spot `Direction + Confidence` must be valid
- Then selected timeframe execution is checked from:
  - local structure quality
  - local trend quality
  - local regime quality
  - local location quality
  - local spot-style risk/reward from support / resistance / EMA21 / ATR
- `TREND-led` = pure technical selected-timeframe confirmation
- `AI-led` = pure AI confirmation, but it still must pass the same execution safety gates
- `TREND+AI` = both motors are independently strong and also elite together
- `PROBE` = not fully confirmed yet, but clean enough for starter-risk only
- `WATCH` = the idea is alive, but timing is not clean yet
- `SKIP` = edge is too weak, too conflicted, or badly located right now

Setup Confirm classes (final confirmation class):
- TREND+AI: both technicals and AI agree, and both are very strong
- TREND-led: the main direction exists, and the selected timeframe technical picture now supports it
- AI-led: the main direction exists, and AI support is strong enough for a setup
- PROBE: the setup is close enough for small starter risk, but not full size
- WATCH: the idea is still alive, but timing is not clean yet
- SKIP: the edge is too weak or the setup is not worth tracking now

Scalp Opportunity is separate from Setup Confirm.
It appears only if all execution gates pass:
- Direction match
- Timeframe-adaptive R:R / ADX / Confidence thresholds
- No tech/AI conflict
- Valid entry/stop/target

Timeframe gate matrix:
- 5m / 15m: R:R >= 1.30, ADX >= 18, Confidence >= 52
- 1h: R:R >= 1.40, ADX >= 18, Confidence >= 52
- 4h / 1d: R:R >= 1.70, ADX >= 22, Confidence >= 60

Important consistency rule:
- Spot Direction/Confidence and tactical plan levels are computed from **closed candles**
- Price column also reflects closed-candle close in scanner context

Mode badges:
- FULL MARKET MODE: exchange + enrichment providers active
- CUSTOM WATCHLIST MODE: scanner is running the requested watchlist directly
- CUSTOM WATCHLIST MODE (PARTIAL ENRICHMENT): watchlist scan is live, but market-cap enrichment is only available for some symbols
- CUSTOM WATCHLIST MODE (EXCHANGE-ONLY): watchlist scan is live from exchange candles, but enrichment is unavailable
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
- Confidence
- AI Ensemble
- AI Confidence

Meaning:
- `Direction`: higher-timeframe technical spot bias from `1D + 4H`
- `AI Ensemble`: higher-timeframe AI bias from `1D + 4H`; the 3 dots show how many ensemble models support that final AI direction
- `AI Confidence`: quality score of that higher-timeframe AI verdict

These 5 columns use the same core logic as Market tab:
- `Direction`: HTF technical bias (`1D + 4H`)
- `Confidence`: quality of that HTF technical bias
- `AI Ensemble`: HTF AI bias (`1D + 4H`)
- `AI Confidence`: quality of that HTF AI verdict
- `Setup Confirm`: selected-timeframe confirmation layer built on top of those HTF reads

Setup Confirm classes:
- TREND+AI: technicals and AI both agree, and both are strong
- TREND-led: technicals support the main direction
- AI-led: AI support is strong enough for the main direction
- PROBE: starter-risk only; not full confirmation yet
- WATCH: keep it on the radar, but do not rush
- SKIP: leave it alone for now

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
- Visible `Direction` + `Confidence` are the same higher-timeframe spot read used in Market.
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
- Direction / Confidence / AI Ensemble / AI Confidence summary
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
            "10) Multi-TF tab",
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
            "11) Correlation tab",
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
            "12) Portfolio Scenario tab",
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
            "13) Sessions tab",
            """
Execution timing filter across Asia / Europe / US windows.
Shows:
- relative session quality (not an absolute signal)
- liquidity depth by session
- range profile and drift bias

Use it to decide when execution conditions look cleaner after a setup already exists elsewhere.
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
- confidence threshold
- fixed holding bars
- fee/slippage assumptions

This mode validates the raw Direction + Confidence engine, not only the higher-timeframe spot Direction model.
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
- scalp gate (timeframe-adaptive R:R / ADX / Confidence thresholds)
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
2. Correlation: check whether your planned basket is over-crowded
3. Portfolio Scenario: stress-test how your basket may react if BTC/ETH or another anchor reaches target
4. Spot: validate setup and read the Spot Execution Plan
5. Position: if already in trade, follow Technical Invalidation + decision model first
6. Fibonacci/Risk: validate structure and downside risk
7. Tools: confirm R:R and liquidation distance
8. Setup Backtest: validate Setup Confirm class edge before using new setup live
9. Scalp Backtest: validate scalp gate behavior and TP/SL outcome profile
10. Model Lab: tune raw signal threshold/holding parameters

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
- Setup Confirm / Direction / Confidence / AI Ensemble / AI Confidence are not empty

2. **Spot tab**
- Analyse runs and shows Direction + Confidence + HTF AI + AI Confidence
- Spot Execution Plan appears

3. **Position tab**
- Raw/Levered PnL and Net PnL render correctly
- Technical Invalidation line is visible
- Excel report downloads without resetting analysis view

4. **AI Workspace**
- Quick Prediction mode returns Direction / Probability / Agreement in one run
- Model & Timeframe Matrix mode fills timeframe matrix with plan source fields
- Debug expander shows AI vs non-AI plan levels

5. **Portfolio Scenario**
- Basket editor accepts your holdings and anchor target
- Current basket / projected basket / scenario table render in one run
- Basket Insight card explains whether anchor response is concentrated, balanced, or loose

6. **Fallback check**
- If enrichment fails, Market shows **EXCHANGE-ONLY MODE** and core trade columns still render
- If live endpoint fails, cached snapshot warning with UTC timestamp appears
- Market should not reuse stale snapshot from a different timeframe/filter

7. **Model Lab + Setup/Scalp Backtest**
- Model Lab runs and returns trades/metrics for raw signal diagnostics
- Setup Backtest returns setup-event count, forward-bar outcome table, and class breakdown
- Scalp Backtest returns scalp-qualified event count, class outcome table, and TP/SL behavior
            """,
            "info",
        ),
    ]

    _render_sections(st, sections)
