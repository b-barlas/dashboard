from __future__ import annotations

import os


DEFAULT_AUDIENCE = "trader"
_VALID_AUDIENCES = {"trader", "neutral"}
_ACTIVE_AUDIENCE = DEFAULT_AUDIENCE


_COPY = {
    "stance.display.stand_aside": {
        "trader": "Stand Aside",
        "neutral": "Low Alignment",
    },
    "stance.display.defensive_only": {
        "trader": "Defensive Only",
        "neutral": "Cautious",
    },
    "stance.display.selective_only": {
        "trader": "Selective Only",
        "neutral": "Selective",
    },
    "stance.display.tradeable": {
        "trader": "Tradeable",
        "neutral": "Supportive",
    },
    "context_fit.aggression.no_fresh_risk": {
        "trader": "No fresh risk",
        "neutral": "Low alignment",
    },
    "context_fit.aggression.normal_aggression": {
        "trader": "Normal aggression",
        "neutral": "Supportive backdrop",
    },
    "context_fit.aggression.selective_adds_only": {
        "trader": "Selective adds only",
        "neutral": "Selective follow-through only",
    },
    "context_fit.aggression.reduced_aggression": {
        "trader": "Reduced aggression",
        "neutral": "Use extra caution",
    },
    "context_fit.aggression.probe_only": {
        "trader": "Early-entry only",
        "neutral": "Early confirmation only",
    },
    "spot.archive.status.guardrail": {
        "trader": "Archive Guardrail",
        "neutral": "Archive Guardrail",
    },
    "spot.archive.status.caution": {
        "trader": "Archive Caution",
        "neutral": "Archive Caution",
    },
    "spot.archive.status.supportive": {
        "trader": "History Supportive",
        "neutral": "History Supportive",
    },
    "spot.archive.status.fragile": {
        "trader": "History Fragile",
        "neutral": "History Fragile",
    },
    "spot.archive.status.session_supportive": {
        "trader": "Session Supportive",
        "neutral": "Session Supportive",
    },
    "spot.archive.status.session_fragile": {
        "trader": "Session Fragile",
        "neutral": "Session Fragile",
    },
    "spot.archive.status.mixed": {
        "trader": "History Mixed",
        "neutral": "History Mixed",
    },
    "spot.archive.history.guardrail": {
        "trader": "Similar setups have struggled enough here to avoid fresh risk.",
        "neutral": "Similar setups have struggled enough here to avoid fresh risk.",
    },
    "spot.archive.history.caution": {
        "trader": "Similar setups have been softer in this kind of market window.",
        "neutral": "Similar setups have been softer in this kind of market window.",
    },
    "spot.archive.history.supportive": {
        "trader": "Similar setups have generally held up well.",
        "neutral": "Similar setups have generally held up well.",
    },
    "spot.archive.history.fragile": {
        "trader": "Similar setups have had weak follow-through.",
        "neutral": "Similar setups have had weak follow-through.",
    },
    "spot.archive.history.neutral": {
        "trader": "History is mixed here, so cleaner confirmation matters more.",
        "neutral": "History is mixed here, so cleaner confirmation matters more.",
    },
    "spot.archive.session.supportive": {
        "trader": "This session has been a cleaner trading window lately.",
        "neutral": "This session has been a cleaner trading window lately.",
    },
    "spot.archive.session.fragile": {
        "trader": "This session has been less reliable lately.",
        "neutral": "This session has been less reliable lately.",
    },
    "spot.archive.context.trade_gate.no_trade": {
        "trader": "The market is not in a clean entry window right now",
        "neutral": "The market is not in a clean setup window right now",
    },
    "spot.archive.context.trade_gate.selective": {
        "trader": "Only the cleanest setups deserve attention right now",
        "neutral": "Only the clearest setups deserve attention right now",
    },
    "spot.archive.context.trade_gate.tradeable": {
        "trader": "The market is open enough for clean setups",
        "neutral": "The market backdrop is supportive enough for cleaner setups",
    },
    "spot.archive.context.catalyst.far": {
        "trader": "there is no strong catalyst nearby",
        "neutral": "there is no strong catalyst nearby",
    },
    "spot.archive.context.catalyst.near": {
        "trader": "a nearby catalyst could speed up the move",
        "neutral": "a nearby catalyst could speed up the move",
    },
    "spot.archive.context.catalyst.blocking": {
        "trader": "a nearby catalyst is adding event risk",
        "neutral": "a nearby catalyst is adding event risk",
    },
    "spot.archive.context.flow.balanced": {
        "trader": "positioning looks balanced rather than stretched",
        "neutral": "positioning looks balanced rather than stretched",
    },
    "spot.archive.context.flow.crowded": {
        "trader": "positioning looks stretched enough to watch for squeeze risk",
        "neutral": "positioning looks stretched enough to watch for squeeze risk",
    },
    "spot.archive.context.playbook.wait": {
        "trader": "The market still needs cleaner confirmation before pressing a fresh spot setup.",
        "neutral": "The market still needs cleaner confirmation before this setup looks stronger.",
    },
    "spot.plan.title": {
        "trader": "Spot Execution Plan",
        "neutral": "Spot Setup Read",
    },
    "spot.plan.label.mode": {
        "trader": "Mode",
        "neutral": "Read",
    },
    "spot.plan.label.setup": {
        "trader": "Setup Confirm",
        "neutral": "System Class",
    },
    "spot.plan.label.market_stance": {
        "trader": "Market stance",
        "neutral": "Market backdrop",
    },
    "spot.plan.label.now": {
        "trader": "Now",
        "neutral": "Current read",
    },
    "spot.plan.label.entry": {
        "trader": "Entry path",
        "neutral": "Reference path",
    },
    "spot.plan.label.protection": {
        "trader": "Protection",
        "neutral": "Protection",
    },
    "spot.plan.label.next": {
        "trader": "Next",
        "neutral": "What would improve",
    },
    "spot.plan.mode.no_trade": {
        "trader": "Stand Aside",
        "neutral": "Low Alignment",
    },
    "spot.plan.mode.probe": {
        "trader": "Early",
        "neutral": "Early Setup",
    },
    "spot.plan.mode.watch": {
        "trader": "Watch",
        "neutral": "Developing",
    },
    "spot.plan.mode.bullish_confirmed": {
        "trader": "Bullish Confirmed",
        "neutral": "Supportive Setup",
    },
    "spot.plan.mode.defensive_confirmed": {
        "trader": "Defensive Confirmed",
        "neutral": "Defensive Setup",
    },
    "spot.plan.skip.now": {
        "trader": "Do not open a new spot position on this structure.",
        "neutral": "This structure is not aligned for fresh spot risk right now.",
    },
    "spot.plan.skip.entry": {
        "trader": "Reference only: keep {left_zone_label} ({pullback_zone_text}) and {trigger_label} ({breakout_trigger}) on the map.",
        "neutral": "Reference only: keep {left_zone_label} ({pullback_zone_text}) and {trigger_label} ({breakout_trigger}) on the map.",
    },
    "spot.plan.skip.protection": {
        "trader": "If already holding, keep protection at {pullback_invalidation}.",
        "neutral": "If already holding, keep protection at {pullback_invalidation}.",
    },
    "spot.plan.skip.next": {
        "trader": "Wait for WATCH, EARLY, or a confirmed class before treating this as active risk.",
        "neutral": "Wait for stronger confirmation before treating this as an active setup.",
    },
    "spot.plan.probe.upside.now": {
        "trader": "Starter-risk only. Keep size small.",
        "neutral": "This is still an early setup. Keep risk small.",
    },
    "spot.plan.probe.upside.entry": {
        "trader": "Starter entry can come from {left_zone_label} ({pullback_zone_text}) or a clean close above {trigger_label} ({breakout_trigger}).",
        "neutral": "Early setup interest sits around {left_zone_label} ({pullback_zone_text}) or a clean close above {trigger_label} ({breakout_trigger}).",
    },
    "spot.plan.probe.upside.protection": {
        "trader": "Stops stay at {pullback_invalidation} / {breakout_invalidation}.",
        "neutral": "Risk references stay at {pullback_invalidation} / {breakout_invalidation}.",
    },
    "spot.plan.probe.upside.next": {
        "trader": "Add only after stronger confirmation. Targets: {left_tp_label} ({pullback_tp_text}) / {right_tp_label} ({breakout_tp_text}).",
        "neutral": "Stronger confirmation would improve this setup. Upside references: {left_tp_label} ({pullback_tp_text}) / {right_tp_label} ({breakout_tp_text}).",
    },
    "spot.plan.probe.downside.now": {
        "trader": "Stay defensive. Do not add fresh spot size here.",
        "neutral": "Stay defensive here. This is not aligned for fresh spot size.",
    },
    "spot.plan.probe.downside.entry": {
        "trader": "Treat this as an early warning, not a buy trigger.",
        "neutral": "Treat this as an early warning, not a buy setup.",
    },
    "spot.plan.probe.downside.protection": {
        "trader": "Wait for a reclaim above {trigger_label} ({breakout_trigger}) and keep protection at {pullback_invalidation}.",
        "neutral": "A reclaim above {trigger_label} ({breakout_trigger}) would improve the picture. Keep protection at {pullback_invalidation}.",
    },
    "spot.plan.probe.downside.next": {
        "trader": "Only reconsider upside risk after direction repairs.",
        "neutral": "Reconsider this only after direction improves.",
    },
    "spot.plan.probe.neutral.now": {
        "trader": "Attention only. The structure is interesting, but not ready for committed spot risk.",
        "neutral": "The structure is interesting, but not ready for committed spot risk.",
    },
    "spot.plan.probe.neutral.entry": {
        "trader": "Use it as a starter-watch zone around {left_zone_label} ({pullback_zone_text}) and {trigger_label} ({breakout_trigger}).",
        "neutral": "Use it as an early watch zone around {left_zone_label} ({pullback_zone_text}) and {trigger_label} ({breakout_trigger}).",
    },
    "spot.plan.probe.neutral.protection": {
        "trader": "Do not commit full spot size while direction is still neutral.",
        "neutral": "Avoid full spot size while direction is still neutral.",
    },
    "spot.plan.probe.neutral.next": {
        "trader": "Wait for one side to confirm and close with follow-through.",
        "neutral": "Wait for one side to confirm with follow-through.",
    },
    "spot.plan.watch.upside.now": {
        "trader": "Monitor only. Confirmation is partial.",
        "neutral": "This setup is still developing.",
    },
    "spot.plan.watch.upside.entry": {
        "trader": "Watch reaction quality in {left_zone_label} ({pullback_zone_text}) or a close above {trigger_label} ({breakout_trigger}).",
        "neutral": "Watch reaction quality in {left_zone_label} ({pullback_zone_text}) or a close above {trigger_label} ({breakout_trigger}).",
    },
    "spot.plan.watch.upside.protection": {
        "trader": "Stops stay at {pullback_invalidation} / {breakout_invalidation}.",
        "neutral": "Risk references stay at {pullback_invalidation} / {breakout_invalidation}.",
    },
    "spot.plan.watch.upside.next": {
        "trader": "If confirmation improves, targets sit at {left_tp_label} ({pullback_tp_text}) / {right_tp_label} ({breakout_tp_text}).",
        "neutral": "If confirmation improves, upside references sit at {left_tp_label} ({pullback_tp_text}) / {right_tp_label} ({breakout_tp_text}).",
    },
    "spot.plan.watch.downside.now": {
        "trader": "Stay defensive until reclaim confirmation appears.",
        "neutral": "Stay defensive until reclaim confirmation appears.",
    },
    "spot.plan.watch.downside.entry": {
        "trader": "Wait for a close back above {trigger_label} ({breakout_trigger}).",
        "neutral": "A close back above {trigger_label} ({breakout_trigger}) would improve the setup.",
    },
    "spot.plan.watch.downside.protection": {
        "trader": "If already holding, protect with {pullback_invalidation}.",
        "neutral": "If already holding, protect with {pullback_invalidation}.",
    },
    "spot.plan.watch.downside.next": {
        "trader": "Only after reclaim confirmation should you use {breakout_invalidation} and {right_tp_label} ({breakout_tp_text}).",
        "neutral": "Only after reclaim confirmation should you lean on {breakout_invalidation} and {right_tp_label} ({breakout_tp_text}).",
    },
    "spot.plan.watch.neutral.now": {
        "trader": "No-force zone until one side confirms.",
        "neutral": "No-force zone until one side confirms.",
    },
    "spot.plan.watch.neutral.entry": {
        "trader": "Monitor {left_zone_label} ({pullback_zone_text}) and {trigger_label} ({breakout_trigger}).",
        "neutral": "Monitor {left_zone_label} ({pullback_zone_text}) and {trigger_label} ({breakout_trigger}).",
    },
    "spot.plan.watch.neutral.protection": {
        "trader": "Map risk to {pullback_invalidation} / {breakout_invalidation}.",
        "neutral": "Use {pullback_invalidation} / {breakout_invalidation} as risk references.",
    },
    "spot.plan.watch.neutral.next": {
        "trader": "Keep targets pre-defined at {left_tp_label} ({pullback_tp_text}) / {right_tp_label} ({breakout_tp_text}).",
        "neutral": "Keep upside references at {left_tp_label} ({pullback_tp_text}) / {right_tp_label} ({breakout_tp_text}).",
    },
    "spot.plan.confirmed.upside.now": {
        "trader": "Execution-ready upside setup.",
        "neutral": "This is the strongest upside setup state.",
    },
    "spot.plan.confirmed.upside.entry": {
        "trader": "{left_path_label}: {pullback_zone_text} or {right_path_label}: close above {trigger_label} ({breakout_trigger}).",
        "neutral": "{left_path_label}: {pullback_zone_text} or {right_path_label}: close above {trigger_label} ({breakout_trigger}).",
    },
    "spot.plan.confirmed.upside.protection": {
        "trader": "Stops stay at {pullback_invalidation} / {breakout_invalidation}.",
        "neutral": "Risk references stay at {pullback_invalidation} / {breakout_invalidation}.",
    },
    "spot.plan.confirmed.upside.next": {
        "trader": "Targets: {left_tp_label} ({pullback_tp_text}) / {right_tp_label} ({breakout_tp_text}). Take partials at TP-low and trail the remainder.",
        "neutral": "Upside references: {left_tp_label} ({pullback_tp_text}) / {right_tp_label} ({breakout_tp_text}).",
    },
    "spot.plan.confirmed.downside.now": {
        "trader": "Confirmed read, but spot stays defensive.",
        "neutral": "The read is defensive, so spot should stay cautious.",
    },
    "spot.plan.confirmed.downside.entry": {
        "trader": "No fresh spot buy until direction recovers and reclaims {trigger_label} ({breakout_trigger}).",
        "neutral": "Avoid fresh spot buy until direction recovers and reclaims {trigger_label} ({breakout_trigger}).",
    },
    "spot.plan.confirmed.downside.protection": {
        "trader": "If already holding, de-risk into rallies and protect with {pullback_invalidation}.",
        "neutral": "If already holding, de-risk into rallies and protect with {pullback_invalidation}.",
    },
    "spot.plan.confirmed.downside.next": {
        "trader": "Only after reclaim confirmation should you use {breakout_invalidation} and {right_tp_label} ({breakout_tp_text}).",
        "neutral": "Only after reclaim confirmation should you lean on {breakout_invalidation} and {right_tp_label} ({breakout_tp_text}).",
    },
    "spot.plan.disclaimer": {
        "trader": "Guide only, not financial advice. Always confirm with your own risk plan.",
        "neutral": "Guide only. Always confirm with your own risk plan.",
    },
    "spot.levels.actionable_title": {
        "trader": "Execution Levels (spot-only)",
        "neutral": "Reference Levels",
    },
    "spot.levels.defensive_title": {
        "trader": "Defensive Reference Levels",
        "neutral": "Defensive Reference Levels",
    },
    "spot.levels.defensive_caption": {
        "trader": "Spot stays defensive here. Use these as reclaim and protection references, not as an active two-path entry map.",
        "neutral": "Spot stays defensive here. Use these as reclaim and protection references, not as an active two-path setup map.",
    },
    "market.trade_gate.summary.tradeable": {
        "trader": "Market conditions are open for normal-quality setups.",
        "neutral": "Market conditions are supportive enough for normal-quality setups.",
    },
    "market.trade_gate.summary.selective_probe": {
        "trader": "Nothing is fully ready yet, but early-entry setups are live.",
        "neutral": "Nothing is fully ready yet, but early setups are live.",
    },
    "market.trade_gate.summary.selective_clean": {
        "trader": "Conditions are selective; only the cleanest setups deserve fresh risk.",
        "neutral": "Conditions are selective; only the clearest setups deserve attention.",
    },
    "market.trade_gate.summary.defensive": {
        "trader": "Conditions are defensive; fresh risk should stay small.",
        "neutral": "Conditions are defensive; new exposure should stay small.",
    },
    "market.trade_gate.summary.stand_aside": {
        "trader": "Stand aside until market conditions improve.",
        "neutral": "Wait until market conditions improve.",
    },
    "market.trade_gate.summary.size_cap": {
        "trader": "Size cap: {size_label}.",
        "neutral": "Size cap: {size_label}.",
    },
    "market.trade_gate.summary.catalyst": {
        "trader": "Catalyst: {catalyst_label}.",
        "neutral": "Catalyst: {catalyst_label}.",
    },
    "market.trade_gate.summary.flow": {
        "trader": "Flow: {flow_label}.",
        "neutral": "Flow: {flow_label}.",
    },
    "market.trade_gate.summary.session": {
        "trader": "Session: {session_label}.",
        "neutral": "Session: {session_label}.",
    },
    "market.status.head.cached_ready": {
        "trader": "CACHED SETUPS",
        "neutral": "CACHED SETUPS",
    },
    "market.status.head.cached_probe": {
        "trader": "CACHED EARLY SETUPS",
        "neutral": "CACHED EARLY SETUPS",
    },
    "market.status.head.cached_none": {
        "trader": "NO LIVE SETUP",
        "neutral": "NO LIVE SETUP",
    },
    "market.status.sub.cached": {
        "trader": "CACHED READY: {enter_count} • EARLY: {probe_count} • WATCH: {watch_count} • SKIP: {skip_count}",
        "neutral": "CACHED HIGH-QUALITY: {enter_count} • EARLY: {probe_count} • DEVELOPING: {watch_count} • NOT ALIGNED: {skip_count}",
    },
    "market.status.head.degraded_ready": {
        "trader": "PARTIAL-DATA SETUPS",
        "neutral": "PARTIAL-DATA SETUPS",
    },
    "market.status.head.degraded_probe": {
        "trader": "PARTIAL-DATA EARLY SETUPS",
        "neutral": "PARTIAL-DATA EARLY SETUPS",
    },
    "market.status.head.degraded_none": {
        "trader": "NO CLEAN SETUP",
        "neutral": "NO CLEAN SETUP",
    },
    "market.status.sub.degraded": {
        "trader": "PARTIAL-DATA READY: {enter_count} • EARLY: {probe_count} • WATCH: {watch_count} • SKIP: {skip_count}",
        "neutral": "PARTIAL-DATA HIGH-QUALITY: {enter_count} • EARLY: {probe_count} • DEVELOPING: {watch_count} • NOT ALIGNED: {skip_count}",
    },
    "market.status.head.live_ready": {
        "trader": "SETUPS READY",
        "neutral": "SETUPS READY",
    },
    "market.status.head.live_probe": {
        "trader": "EARLY SETUPS LIVE",
        "neutral": "EARLY SETUPS LIVE",
    },
    "market.status.head.live_none": {
        "trader": "NO SETUP READY",
        "neutral": "NO SETUP READY",
    },
    "market.status.sub.live": {
        "trader": "READY: {enter_count} • EARLY: {probe_count} • WATCH: {watch_count} • SKIP: {skip_count}",
        "neutral": "HIGH-QUALITY: {enter_count} • EARLY: {probe_count} • DEVELOPING: {watch_count} • NOT ALIGNED: {skip_count}",
    },
    "market.status.label_title": {
        "trader": "Quick count of how many shown rows are ready, early, watch-only, or pass.",
        "neutral": "Quick count of how many shown rows are high-quality, early, developing, or not aligned.",
    },
    "market.audit.probe_heavy": {
        "trader": "Early-entry heavy table: setups are close enough for small risk, but full confirmation is still scarce.",
        "neutral": "Early-setup heavy table: many candidates are close, but stronger confirmation is still scarce.",
    },
    "market.audit.watch_heavy": {
        "trader": "WATCH-heavy table: selected-timeframe execution checks are filtering most names.",
        "neutral": "Developing-setup heavy table: selected-timeframe execution checks are filtering most names.",
    },
    "market.audit.skip_neutral": {
        "trader": "SKIP-heavy table: most candidates are being rejected because the higher-timeframe Direction is still Neutral.",
        "neutral": "Not-aligned heavy table: most candidates are being filtered out because the higher-timeframe direction is still neutral.",
    },
    "market.audit.skip_risk": {
        "trader": "SKIP-heavy table: location/risk filters are rejecting most candidates.",
        "neutral": "Not-aligned heavy table: location/risk filters are rejecting most candidates.",
    },
    "review.label.no_trade_reason": {
        "trader": "Stand-Aside Reason",
        "neutral": "Low-Alignment Reason",
    },
    "review.group.no_trade_reason": {
        "trader": "By Stand-Aside Reason",
        "neutral": "By Low-Alignment Reason",
    },
    "market.hero.fear_greed_tip": {
        "trader": "Quick sentiment gauge. Lower means fear; higher means greed. Use it as background context, not as a trade trigger.",
        "neutral": "Quick sentiment gauge. Lower means fear; higher means greed. Use it as background context, not as a standalone trigger.",
    },
    "market.tooltip.setup_confirm": {
        "trader": "Final table read showing whether the setup is ready, early-only, watch-only, or a pass for now.",
        "neutral": "Final table read showing whether the setup looks high-quality, early but interesting, still developing, or not aligned yet.",
    },
    "market.tooltip.entry_price": {
        "trader": "Suggested model entry level for the intraday timing lens. If the scalp read is conditional, treat this as a reference level only.",
        "neutral": "Model reference entry level for the intraday timing lens. If the scalp is conditional, treat this as a reference only.",
    },
    "market.tooltip.stop_loss": {
        "trader": "Risk invalidation level for the intraday timing lens. Conditional scalp rows show this as a reference, not a live trigger.",
        "neutral": "Risk reference level showing where the intraday timing setup would break. Conditional rows are reference-only.",
    },
    "market.tooltip.target_price": {
        "trader": "First target level for the intraday timing lens. Conditional scalp rows show this as a reference target only.",
        "neutral": "First reference target for the intraday timing lens. Conditional rows stay reference-only.",
    },
    "market.tooltip.scalp_opportunity": {
        "trader": "Separate intraday timing lens. Live rows passed all scalp checks. Conditional rows found a local scalp structure, but a broader veto is still active.",
        "neutral": "Separate intraday timing lens. Live rows passed the scalp checks. Conditional rows found a local setup, but a broader veto is still active.",
    },
    "market.help.scanner_guide_html": {
        "trader": (
            "<b>Read order:</b> <b>Setup Confirm</b> -> <b>Direction + Confidence</b> -> <b>AI Ensemble + AI Confidence</b>.<br>"
            "<b>Scan modes:</b> Broad Market = cleaner liquid universe, Breakout Radar = early acceleration hunt, Trending Coins = attention + momentum + volume anomaly candidates, Custom Coins = your watchlist. For Custom Coins, type symbols and press Enter or Scan.<br>"
            "<b>Setup Confirm:</b> ENTER = ready, EARLY = small-risk only, WATCH = monitor only, SKIP = leave it alone for now.<br>"
            "Price ($) shows the latest candle close.<br>"
            "Δ (%) shows the change from previous closed candle to latest closed candle on selected timeframe.<br>"
            "<b>Tip:</b> use column-header and cell hovers for detailed definitions. Advanced columns are optional."
        ),
        "neutral": (
            "<b>Read order:</b> <b>Setup Confirm</b> -> <b>Direction + Confidence</b> -> <b>AI Ensemble + AI Confidence</b>.<br>"
            "<b>Scan modes:</b> Broad Market = cleaner liquid universe, Breakout Radar = early acceleration read, Trending Coins = attention + momentum + volume anomaly candidates, Custom Coins = your watchlist. For Custom Coins, type symbols and press Enter or Scan.<br>"
            "<b>Setup Confirm:</b> High-quality = strongest class, Early setup = interesting but not fully confirmed, Developing = monitor, Not aligned = pass.<br>"
            "Price ($) shows the latest candle close.<br>"
            "Δ (%) shows the change from previous closed candle to latest closed candle on selected timeframe.<br>"
            "<b>Tip:</b> use column-header and cell hovers for detailed definitions. Advanced columns are optional."
        ),
    },
    "spot.help.quick_html": {
        "trader": (
            "<b>1.</b> Start with <b>Setup Snapshot</b>: Δ (%) + Setup Confirm + Direction + Confidence.<br>"
            "<b>2.</b> Read <b>Setup Confirm</b> first: TREND+AI = strongest confirmation, TREND-led = technicals support the move, AI-led = AI support is strong enough, EARLY = small-risk only, WATCH = idea is alive but early, SKIP = leave it alone for now. This uses selected-timeframe execution quality plus a local spot risk model, not the scalp planner.<br>"
            "<b>3.</b> <b>Direction</b> = higher-timeframe spot bias from the adaptive lead/confirm anchor pair. <b>Confidence</b> = quality of that bias.<br>"
            "<b>4.</b> Validate with <b>AI Ensemble</b> + <b>AI Confidence</b>. AI Ensemble is the higher-timeframe AI bias from the same adaptive anchor pair; AI Confidence scores how reliable that higher-timeframe AI verdict is.<br>"
            "<b>5.</b> <b>Market Archive Read</b> is a market-history fit check, not a coin-specific proof card. Use it to size aggression, not to override price structure.<br>"
            "<b>6.</b> Use <b>Technical Regime Breakdown</b> only as selected-timeframe confirmation context, not as the main direction engine.<br>"
            "<b>7.</b> If the plan is defensive or SKIP, treat the lower section as reference/reclaim levels, not as an active two-path entry map."
        ),
        "neutral": (
            "<b>1.</b> Start with <b>Setup Snapshot</b>: Δ (%) + Setup Confirm + Direction + Confidence.<br>"
            "<b>2.</b> Read <b>Setup Confirm</b> first: Strong = highest-quality confirmation, Early setup = interesting but still early, Developing = monitor, Not aligned = leave it alone for now. This uses selected-timeframe execution quality plus a local spot risk model, not the scalp planner.<br>"
            "<b>3.</b> <b>Direction</b> = higher-timeframe spot bias from the adaptive lead/confirm anchor pair. <b>Confidence</b> = quality of that bias.<br>"
            "<b>4.</b> Validate with <b>AI Ensemble</b> + <b>AI Confidence</b>. AI Ensemble is the higher-timeframe AI bias from the same adaptive anchor pair; AI Confidence scores how reliable that higher-timeframe AI verdict is.<br>"
            "<b>5.</b> <b>Market Archive Read</b> is a market-history fit check, not a coin-specific proof card. Use it to size aggression, not to override price structure.<br>"
            "<b>6.</b> Use <b>Technical Regime Breakdown</b> only as selected-timeframe confirmation context, not as the main direction engine.<br>"
            "<b>7.</b> If the plan is defensive or low alignment, treat the lower section as reference/reclaim levels, not as an active two-path setup map."
        ),
    },
    "multitf.intro_html": {
        "trader": (
            "Check one coin across 5m, 15m, 1h, 4h, and 1d to see whether short-term timing agrees with higher-timeframe structure. "
            "Use it after Market or Spot when you want extra confirmation. "
            "Higher timeframes carry more weight here because they usually define the stronger structural regime. "
            "This tab is a confirmation view, not a standalone trade command."
        ),
        "neutral": (
            "Check one coin across 5m, 15m, 1h, 4h, and 1d to see whether short-term timing agrees with higher-timeframe structure. "
            "Use it after Market or Spot when you want extra confirmation. "
            "Higher timeframes carry more weight here because they usually define the stronger structural regime. "
            "This tab is a confirmation view, not a standalone action command."
        ),
    },
    "multitf.help.quick_html": {
        "trader": (
            "<b>1.</b> Start with <b>Higher-TF Bias</b>. That is the structural layer and usually matters most.<br>"
            "<b>2.</b> Then read <b>Directional Alignment</b>. It measures weighted direction agreement, not full setup quality by itself.<br>"
            "<b>3.</b> Check <b>Confirming Confidence</b>. That tells you how strong the supporting timeframes look on their own confidence model.<br>"
            "<b>4.</b> Use <b>Short-TF Timing</b> as confirmation. If 5m/15m disagree with 1h/4h/1d, the structure may still be valid but entry timing is noisy.<br>"
            "<b>5.</b> If coverage is partial, trust the result less. Missing timeframes reduce confidence even when the visible alignment looks clean.<br>"
            "<b>6.</b> A `*` in <b>AI Ensemble</b> means the ML layer used a neutral safety read on that timeframe.<br>"
            "<b>7.</b> <b>Setup Confirm</b>, <b>Direction</b>, <b>Confidence</b>, <b>AI Ensemble</b>, and <b>AI Confidence</b> use the same spot execution engine family as Market/Spot for that timeframe, with archive calibration when enough history exists. They still support the main read; they are not standalone trade commands."
        ),
        "neutral": (
            "<b>1.</b> Start with <b>Higher-TF Bias</b>. That is the structural layer and usually matters most.<br>"
            "<b>2.</b> Then read <b>Directional Alignment</b>. It measures weighted direction agreement, not full setup quality by itself.<br>"
            "<b>3.</b> Check <b>Confirming Confidence</b>. That tells you how strong the supporting timeframes look on their own confidence model.<br>"
            "<b>4.</b> Use <b>Short-TF Timing</b> as confirmation. If 5m/15m disagree with 1h/4h/1d, the structure may still be valid but timing is noisy.<br>"
            "<b>5.</b> If coverage is partial, trust the result less. Missing timeframes reduce confidence even when the visible alignment looks clean.<br>"
            "<b>6.</b> A `*` in <b>AI Ensemble</b> means the ML layer used a neutral safety read on that timeframe.<br>"
            "<b>7.</b> <b>Setup Confirm</b>, <b>Direction</b>, <b>Confidence</b>, <b>AI Ensemble</b>, and <b>AI Confidence</b> use the same spot execution engine family as Market/Spot for that timeframe, with archive calibration when enough history exists. They still support the main read; they are not standalone action commands."
        ),
    },
    "multitf.table.confirmation_note": {
        "trader": "This table is for confirmation, not a standalone trade trigger. Open advanced columns only when you want the full regime breakdown.",
        "neutral": "This table is for confirmation, not a standalone action trigger. Open advanced columns only when you want the full regime breakdown.",
    },
    "multitf.table.column_guide_html": {
        "trader": (
            "<b>Timeframe</b>: the candle interval being checked.<br>"
            "<b>Role</b>: 5m/15m are timing; 1h/4h/1d are structural.<br>"
            "<b>Δ (%)</b>: last closed-candle change for that timeframe.<br>"
            "<b>Setup Confirm</b>: the same setup-confirm engine family the Market tab uses for that timeframe, then archive-calibrated when enough live history exists.<br>"
            "<b>Direction</b>: the same higher-timeframe spot-bias direction family used in Market/Spot, but built for this row's timeframe.<br>"
            "<b>Confidence</b>: the same spot-confidence family used in Market/Spot for that timeframe, including archive calibration when available.<br>"
            "<b>AI Ensemble</b>: the same higher-timeframe AI bias family used in Market/Spot for that timeframe. `*` means the AI layer used a neutral safety read.<br>"
            "<b>AI Confidence</b>: the same AI-confidence family used in Market/Spot for that timeframe, including archive calibration when available.<br>"
            "<b>Advanced view</b>: filters advanced columns into Trend, Momentum, or Volatility & Volume subsets."
        ),
        "neutral": (
            "<b>Timeframe</b>: the candle interval being checked.<br>"
            "<b>Role</b>: 5m/15m are timing; 1h/4h/1d are structural.<br>"
            "<b>Δ (%)</b>: last closed-candle change for that timeframe.<br>"
            "<b>Setup Confirm</b>: the same setup-confirm engine family the Market tab uses for that timeframe, then archive-calibrated when enough live history exists.<br>"
            "<b>Direction</b>: the same higher-timeframe spot-bias direction family used in Market/Spot, but built for this row's timeframe.<br>"
            "<b>Confidence</b>: the same spot-confidence family used in Market/Spot for that timeframe, including archive calibration when available.<br>"
            "<b>AI Ensemble</b>: the same higher-timeframe AI bias family used in Market/Spot for that timeframe. `*` means the AI layer used a neutral safety read.<br>"
            "<b>AI Confidence</b>: the same AI-confidence family used in Market/Spot for that timeframe, including archive calibration when available.<br>"
            "<b>Advanced view</b>: filters advanced columns into Trend, Momentum, or Volatility & Volume subsets."
        ),
    },
    "ml.setup_snapshot.setup_confirm_title": {
        "trader": "Execution class from market decision policy.",
        "neutral": "System setup class from the market classification policy.",
    },
    "fib.action_hint.strong": {
        "trader": "Action: execution-ready zone. Wait only for trigger confirmation.",
        "neutral": "Read: high-quality zone. Wait only for trigger confirmation.",
    },
    "fib.action_hint.moderate": {
        "trader": "Action: confirmation-first, smaller size, strict invalidation.",
        "neutral": "Read: confirmation-first, smaller size, strict invalidation.",
    },
    "fib.action_hint.weak": {
        "trader": "Action: stand aside until quality improves.",
        "neutral": "Read: low alignment until quality improves.",
    },
    "no_trade.note.degraded_scan": {
        "trader": "The live market read is partial, so the picture is incomplete. Wait for full data before taking fresh risk.",
        "neutral": "The live market read is partial, so the picture is incomplete. Wait for full data before acting on new setups.",
    },
    "no_trade.note.catalyst_block_default": {
        "trader": "A market-wide catalyst is too close to trust fresh risk.",
        "neutral": "A market-wide catalyst is too close to trust new setup exposure.",
    },
    "no_trade.note.probe_only": {
        "trader": "Nothing is fully ready yet, but there are early-entry setups worth small risk. Stay selective and keep size light until stronger confirmation appears.",
        "neutral": "Nothing is fully ready yet, but a few early setups are interesting. Stay selective and keep size small until stronger confirmation appears.",
    },
    "no_trade.note.no_ready_setups": {
        "trader": "Nothing is truly ready right now. Most names are still being filtered out, so patience is the cleaner decision.",
        "neutral": "Nothing is truly aligned right now. Most names are still being filtered out, so patience is the cleaner decision.",
    },
    "no_trade.note.weak_participation": {
        "trader": "Participation is too weak and leadership is unclear. This is a poor environment for pressing new ideas.",
        "neutral": "Participation is too weak and leadership is unclear. This is a poor environment for stretching standards.",
    },
    "no_trade.note.archive_cluster_prefix": {
        "trader": "This looks like one of the historically weak cluster windows.",
        "neutral": "This looks like one of the historically weak cluster windows.",
    },
    "no_trade.note.risk_off_defensive": {
        "trader": "Downside pressure is still leading. Stay defensive and avoid forcing fresh upside exposure.",
        "neutral": "Downside pressure is still leading. Stay defensive and avoid forcing fresh upside exposure.",
    },
    "no_trade.note.risk_off_weakness": {
        "trader": "The market is still defensive without a clean directional read worth acting on. Capital preservation comes first.",
        "neutral": "The market is still defensive without a clean directional read worth leaning on. Capital preservation comes first.",
    },
    "no_trade.note.selective_clean": {
        "trader": "Setups can work, but only the cleanest aligned leaders deserve attention. Treat everything else as noise.",
        "neutral": "Setups can work, but only the clearest aligned leaders deserve attention. Treat everything else as noise.",
    },
    "no_trade.note.selective_probe": {
        "trader": "This is still a selective market, but early-entry setups are showing up. Small risk is fine; save normal size for cleaner confirmation.",
        "neutral": "This is still a selective market, but early setups are showing up. Keep size small and wait for cleaner confirmation before doing more.",
    },
    "no_trade.note.selective_catalyst_default": {
        "trader": "Setups can work, but a market catalyst is close enough that we should stay selective.",
        "neutral": "Setups can work, but a market catalyst is close enough that we should stay selective.",
    },
    "no_trade.note.selective_session_default": {
        "trader": "Setups can work, but the current session archive has been weak enough to keep us extra selective.",
        "neutral": "Setups can work, but the current session archive has been weak enough to keep us extra selective.",
    },
    "no_trade.note.selective_archive_default": {
        "trader": "Setups can work, but matched archive history is weak enough to keep us extra selective.",
        "neutral": "Setups can work, but matched archive history is weak enough to keep us extra selective.",
    },
    "no_trade.note.tradeable_catalyst_default": {
        "trader": "A nearby market catalyst is close enough to keep conditions selective instead of fully tradeable.",
        "neutral": "A nearby market catalyst is close enough to keep conditions selective instead of fully supportive.",
    },
    "no_trade.note.tradeable_session_prefix": {
        "trader": "The broader market is supportive, but the current session archive is weak enough to avoid full aggression.",
        "neutral": "The broader market is supportive, but the current session archive is weak enough to avoid full aggression.",
    },
    "no_trade.note.tradeable_archive_default": {
        "trader": "Market conditions look open, but matched playbook history is weak enough to avoid full aggression.",
        "neutral": "Market conditions look open, but matched playbook history is weak enough to avoid full aggression.",
    },
    "no_trade.note.tradeable_base": {
        "trader": "Conditions are supportive enough to hunt normal-quality setups. Keep confirmation discipline, but the market is open for business.",
        "neutral": "Conditions are supportive enough for normal-quality setups. Keep confirmation discipline, but the market is open enough to explore.",
    },
    "no_trade.note.filter_harder_base": {
        "trader": "There is some opportunity, but not enough to widen standards. Filter hard and take only clearly aligned names.",
        "neutral": "There is some opportunity, but not enough to widen standards. Stay selective and focus only on clearly aligned names.",
    },
    "no_trade.note.filter_harder_catalyst_default": {
        "trader": "There is some opportunity, but a near catalyst means we should keep filtering hard.",
        "neutral": "There is some opportunity, but a near catalyst means we should keep filtering hard.",
    },
    "no_trade.note.filter_harder_archive_default": {
        "trader": "There is some opportunity, but matched archive history is weak enough to keep filtering hard.",
        "neutral": "There is some opportunity, but matched archive history is weak enough to keep filtering hard.",
    },
    "risk_sizing.note.full": {
        "trader": "Top-tier setup. Normal size is justified.",
        "neutral": "Highest-quality setup. Normal size is reasonable.",
    },
    "risk_sizing.note.half": {
        "trader": "Good setup, but keep size controlled.",
        "neutral": "Good setup, but keep size controlled.",
    },
    "risk_sizing.note.probe": {
        "trader": "Early or selective setup. Use only probe-size risk.",
        "neutral": "Early or selective setup. Keep size small.",
    },
    "risk_sizing.note.flat": {
        "trader": "Do not allocate fresh risk to this setup.",
        "neutral": "Do not add new exposure to this setup.",
    },
    "risk_sizing.note.catalyst_cap": {
        "trader": "Catalyst risk is capping size.",
        "neutral": "Catalyst risk is capping size.",
    },
    "risk_sizing.note.market_gate_cap": {
        "trader": "Current market stance is capping size.",
        "neutral": "Current market filter is capping size.",
    },
    "risk_sizing.note.probe_only": {
        "trader": "Probe setup: starter size only until full confirmation appears.",
        "neutral": "Early setup: keep size small until stronger confirmation appears.",
    },
    "risk_sizing.note.learned_support": {
        "trader": "Learned execution history is supporting size.",
        "neutral": "Learned history is supporting size.",
    },
    "risk_sizing.note.learned_trim": {
        "trader": "Learned execution history is trimming size.",
        "neutral": "Learned history is trimming size.",
    },
    "risk_sizing.note.session_support": {
        "trader": "Current session archive is supporting size.",
        "neutral": "Current session archive is supporting size.",
    },
    "risk_sizing.note.session_trim": {
        "trader": "Current session archive is trimming size.",
        "neutral": "Current session archive is trimming size.",
    },
    "risk_sizing.note.weak_cluster_probe": {
        "trader": "Historically weak alert/playbook timing cluster is forcing probe-only size.",
        "neutral": "Historically weak alert/playbook timing cluster is forcing smaller size.",
    },
    "risk_sizing.note.archive_guardrail_trim": {
        "trader": "History caution is trimming size.",
        "neutral": "History caution is trimming size.",
    },
    "risk_sizing.note.archive_caution_trim": {
        "trader": "History caution is keeping size smaller.",
        "neutral": "History caution is keeping size smaller.",
    },
    "alert.playbook.supportive.title": {
        "trader": "Active playbook window is lining up",
        "neutral": "Active playbook window is lining up",
    },
    "alert.playbook.supportive.note": {
        "trader": "{playbook} is getting support from the current session and catalyst window. Live names like {names} are lining up with that archive read.",
        "neutral": "{playbook} is getting support from the current session and catalyst window. Live names like {names} are lining up with that archive read.",
    },
    "alert.playbook.fragile.title": {
        "trader": "Active playbook window looks fragile",
        "neutral": "Active playbook window looks fragile",
    },
    "alert.playbook.fragile.note": {
        "trader": "{playbook} is active, but the current timing window has been weak for names like {names}. Stay more selective than the setup alone suggests.",
        "neutral": "{playbook} is active, but the current timing window has been weak for names like {names}. Stay more selective than the setup alone suggests.",
    },
    "alert.catalyst.block.title": {
        "trader": "Stand aside into {title}",
        "neutral": "Caution into {title}",
    },
    "alert.catalyst.block.note": {
        "trader": "A high-impact catalyst is too close to trust fresh risk.",
        "neutral": "A high-impact catalyst is too close to trust new setup exposure.",
    },
    "alert.catalyst.targeted.title": {
        "trader": "Targeted catalyst active: {target}",
        "neutral": "Targeted catalyst active: {target}",
    },
    "alert.catalyst.targeted.note": {
        "trader": "A targeted catalyst is active. Affected names deserve smaller size and cleaner confirmation.",
        "neutral": "A targeted catalyst is active. Affected names deserve smaller size and cleaner confirmation.",
    },
    "alert.catalyst.window.title": {
        "trader": "Catalyst window active: {title}",
        "neutral": "Catalyst window active: {title}",
    },
    "alert.catalyst.window.note": {
        "trader": "A known market catalyst is close enough to justify smaller size.",
        "neutral": "A known market catalyst is close enough to justify smaller size.",
    },
    "alert.trade_gate.no_trade.note": {
        "trader": "The current market state does not justify fresh risk.",
        "neutral": "The current market state does not justify new setup exposure.",
    },
    "alert.trade_gate.defensive.title": {
        "trader": "Defensive mode",
        "neutral": "Defensive mode",
    },
    "alert.trade_gate.defensive.note": {
        "trader": "Conditions still favor smaller, more selective positioning.",
        "neutral": "Conditions still favor smaller, more selective positioning.",
    },
    "sessions.intro_html": {
        "trader": (
            "Compares market behavior across 3 UTC sessions using 1h candles: Asian (00-08), European (08-16), and US (16-00). "
            "It is an <b>execution timing filter</b>, not a standalone entry signal. "
            "Use it to see which session is relatively {deeper}, {controlled}, "
            "and directionally tilted. All labels here are <b>relative across these 3 sessions only</b>."
        ),
        "neutral": (
            "Compares market behavior across 3 UTC sessions using 1h candles: Asian (00-08), European (08-16), and US (16-00). "
            "It is a timing filter, not a standalone signal. "
            "Use it to see which session is relatively {deeper}, {controlled}, "
            "and directionally tilted. All labels here are <b>relative across these 3 sessions only</b>."
        ),
    },
    "position.archive.status.guardrail": {
        "trader": "Archive Guardrail",
        "neutral": "Archive Guardrail",
    },
    "position.archive.status.caution": {
        "trader": "Archive Caution",
        "neutral": "Archive Caution",
    },
    "position.archive.status.supportive": {
        "trader": "History Supportive",
        "neutral": "History Supportive",
    },
    "position.archive.status.fragile": {
        "trader": "History Fragile",
        "neutral": "History Fragile",
    },
    "position.archive.status.session_supportive": {
        "trader": "Session Supportive",
        "neutral": "Session Supportive",
    },
    "position.archive.status.session_fragile": {
        "trader": "Session Fragile",
        "neutral": "Session Fragile",
    },
    "position.archive.status.mixed": {
        "trader": "History Mixed",
        "neutral": "History Mixed",
    },
    "position.archive.stance.stand_aside": {
        "trader": "Stand Aside: avoid fresh adds or new risk here.",
        "neutral": "Low alignment: avoid fresh adds or new risk here.",
    },
    "position.archive.stance.defensive": {
        "trader": "Defensive Only: protect the position and avoid pressing size.",
        "neutral": "Cautious: protect the position and avoid pressing size.",
    },
    "position.archive.stance.tradeable": {
        "trader": "Tradeable: the backdrop is supportive enough to manage this like a normal hold.",
        "neutral": "Supportive: the backdrop is supportive enough to manage this like a normal hold.",
    },
    "position.archive.stance.selective": {
        "trader": "Selective Only: keep the position clean and add only on clear confirmation.",
        "neutral": "Selective: keep the position clean and wait for clear confirmation before adding.",
    },
    "position.archive.history.guardrail": {
        "trader": "Similar setups have struggled enough here to stay defensive.",
        "neutral": "Similar setups have struggled enough here to stay defensive.",
    },
    "position.archive.history.caution": {
        "trader": "Similar setups have been softer in this kind of market window.",
        "neutral": "Similar setups have been softer in this kind of market window.",
    },
    "position.archive.history.supportive": {
        "trader": "Similar setups have generally held up better.",
        "neutral": "Similar setups have generally held up better.",
    },
    "position.archive.history.fragile": {
        "trader": "Similar setups have had weaker follow-through.",
        "neutral": "Similar setups have had weaker follow-through.",
    },
    "position.archive.history.neutral": {
        "trader": "History is mixed here, so clean confirmation matters more.",
        "neutral": "History is mixed here, so clean confirmation matters more.",
    },
    "position.archive.session.supportive": {
        "trader": "This session has been a cleaner management window lately.",
        "neutral": "This session has been a cleaner management window lately.",
    },
    "position.archive.session.fragile": {
        "trader": "This session has been less reliable lately.",
        "neutral": "This session has been less reliable lately.",
    },
    "position.archive.context.trade_gate.no_trade": {
        "trader": "The market is not in a clean add window right now",
        "neutral": "The market is not in a clean add window right now",
    },
    "position.archive.context.trade_gate.selective": {
        "trader": "Only the cleanest adds deserve attention right now",
        "neutral": "Only the clearest adds deserve attention right now",
    },
    "position.archive.context.trade_gate.tradeable": {
        "trader": "The market is open enough for cleaner management",
        "neutral": "The market is supportive enough for cleaner management",
    },
    "position.archive.context.catalyst.far": {
        "trader": "there is no major catalyst nearby",
        "neutral": "there is no major catalyst nearby",
    },
    "position.archive.context.catalyst.near": {
        "trader": "a nearby catalyst could speed up the move",
        "neutral": "a nearby catalyst could speed up the move",
    },
    "position.archive.context.catalyst.blocking": {
        "trader": "a nearby event is adding risk",
        "neutral": "a nearby event is adding risk",
    },
    "position.archive.context.flow.balanced": {
        "trader": "positioning looks balanced rather than stretched",
        "neutral": "positioning looks balanced rather than stretched",
    },
    "position.archive.context.flow.crowded": {
        "trader": "positioning looks stretched enough that squeeze risk matters more",
        "neutral": "positioning looks stretched enough that squeeze risk matters more",
    },
    "position.archive.context.playbook.wait": {
        "trader": "The market still needs cleaner confirmation before pressing this position.",
        "neutral": "The market still needs cleaner confirmation before pressing this position.",
    },
    "position.panel.title": {
        "trader": "Position Management",
        "neutral": "Position Read",
    },
    "position.panel.scope.anchor": {
        "trader": "Anchor",
        "neutral": "Anchor",
    },
    "position.panel.scope.timing": {
        "trader": "Timing",
        "neutral": "Timing",
    },
    "position.panel.label.now": {
        "trader": "Now",
        "neutral": "Current read",
    },
    "position.panel.label.adds": {
        "trader": "Adds",
        "neutral": "Add posture",
    },
    "position.panel.label.market_stance": {
        "trader": "Market stance",
        "neutral": "Market backdrop",
    },
    "position.panel.label.hard_risk_line": {
        "trader": "Hard risk line",
        "neutral": "Hard risk line",
    },
    "position.panel.label.next": {
        "trader": "Next",
        "neutral": "What would improve",
    },
    "position.mgmt.exit.label": {
        "trader": "Exit Now",
        "neutral": "Exit",
    },
    "position.mgmt.exit.size": {
        "trader": "Flat or hedge immediately",
        "neutral": "Flat or hedge immediately",
    },
    "position.mgmt.exit.adds": {
        "trader": "No fresh adds",
        "neutral": "No fresh adds",
    },
    "position.mgmt.exit.risk": {
        "trader": "Close or hedge now and reset only after structure repairs.",
        "neutral": "Close or hedge now and reset only after structure repairs.",
    },
    "position.mgmt.exit.note": {
        "trader": "Hard invalidation or cluster risk is already broken. This is no longer a manageable hold.",
        "neutral": "Hard invalidation or cluster risk is already broken. This is no longer a manageable hold.",
    },
    "position.mgmt.reduce.label": {
        "trader": "Reduce Risk",
        "neutral": "Reduce Risk",
    },
    "position.mgmt.reduce.size": {
        "trader": "Trim 25-50% and protect the core",
        "neutral": "Trim 25-50% and protect the core",
    },
    "position.mgmt.reduce.adds": {
        "trader": "No fresh adds",
        "neutral": "No fresh adds",
    },
    "position.mgmt.reduce.risk": {
        "trader": "Trim size, tighten execution, and keep the hard invalidation non-negotiable.",
        "neutral": "Trim size, tighten execution, and keep the hard invalidation non-negotiable.",
    },
    "position.mgmt.reduce.note": {
        "trader": "{reason}. Treat this as protection mode, not a window to press.{extra}",
        "neutral": "{reason}. Treat this as protection mode, not a window to press.{extra}",
    },
    "position.mgmt.press.label": {
        "trader": "Press on Strength",
        "neutral": "Manage Like a Winner",
    },
    "position.mgmt.press.size": {
        "trader": "Keep core size; add only on confirmation",
        "neutral": "Keep core size; add only on confirmation",
    },
    "position.mgmt.press.adds": {
        "trader": "Add only on fresh confirmation",
        "neutral": "Add only on fresh confirmation",
    },
    "position.mgmt.press.risk": {
        "trader": "Let winners work, but add only if price confirms again above your active structure.",
        "neutral": "Let winners work, but add only if price confirms again above your active structure.",
    },
    "position.mgmt.hold.label": {
        "trader": "Hold Only",
        "neutral": "Hold",
    },
    "position.mgmt.hold.size": {
        "trader": "Keep size stable",
        "neutral": "Keep size stable",
    },
    "position.mgmt.hold.risk": {
        "trader": "Hold with discipline, trail against the hard invalidation, and avoid forcing new size.",
        "neutral": "Hold with discipline, trail against the hard invalidation, and avoid forcing new size.",
    },
    "guide.section.market": {
        "trader": """
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

Market scan modes:
- Broad Market: cleaner liquid-universe scan
- Breakout Radar: broader early-acceleration scan using liquidity, momentum, volume, and archive feedback
- Trending Coins: attention + momentum + volume-anomaly candidates shown in the same table
- Watchlist: enter up to 10 symbols in Custom Coins and press Enter or Scan; the table analyzes only that watchlist
- Top N control is disabled while custom mode is active
- Watchlist mode reads requested symbols directly and does not depend on the broad provider universe
- Selected timeframe controls tactical candle context, levels, scalp timing checks, and Delta
- Visible `Direction` + `Confidence` come from closed higher-timeframe anchor candles
- Visible `AI Ensemble` comes from a separate closed AI bias engine using the same adaptive anchors
- Visible `AI Confidence` scores the quality of that higher-timeframe AI verdict

How the 5 key columns are calculated:

1. `Direction` (main spot direction)
- Uses only closed higher-timeframe anchor candles
- Technical engine builds a score for each timeframe from:
  - structure
  - trend
  - momentum
  - regime / location
- A slower anchor sets the broader bias and a faster anchor confirms it
- Final logic is intentionally strict:
  - if `1D` is Neutral, final Direction becomes Neutral
  - if `1D` is strong and `4H` is aligned, direction is confirmed
  - if `1D` and `4H` conflict, Direction becomes Neutral

2. `Confidence`
- This is the quality score of the same higher-timeframe technical spot direction
- It scores:
  - higher-timeframe alignment
  - structure quality
  - trend quality
  - regime quality
  - location quality
  - and penalties for conflict / range / partial data

3. `AI Ensemble`
- Separate AI bias engine, also using the closed adaptive anchor pair
- 3-model ensemble:
  - Gradient Boosting
  - Random Forest
  - Logistic Regression
- The 3 dots show how many models support the final AI direction

4. `AI Confidence`
- Quality score of the higher-timeframe AI verdict
- Lower when:
  - AI verdict is Neutral
  - `1D` / `4H` AI conflict exists
  - AI data is partial or using a neutral safety read
  - model support is weak

5. `Setup Confirm`
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
- `AI-led` = pure AI confirmation, but it still must pass the same execution safety checks
- `TREND+AI` = both motors are independently strong and also elite together
- `EARLY` = not fully confirmed yet, but clean enough for small-risk only
- `WATCH` = the idea is alive, but timing is not clean yet
- `SKIP` = quality is too weak, too conflicted, or badly located right now

Scalp Opportunity is separate from Setup Confirm.
It appears only if all execution checks pass:
- Direction match
- Timeframe-adaptive R:R / ADX / Confidence thresholds
- No technical/AI conflict
- Valid entry/stop/target
""",
        "neutral": """
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

Market scan modes:
- Broad Market: cleaner liquid-universe scan
- Breakout Radar: broader early-acceleration scan using liquidity, momentum, volume, and archive feedback
- Trending Coins: attention + momentum + volume-anomaly candidates shown in the same table
- Watchlist: enter up to 10 symbols in Custom Coins and press Enter or Scan; the table analyzes only that watchlist
- Top N control is disabled while custom mode is active
- Watchlist mode reads requested symbols directly and does not depend on the broad provider universe
- Selected timeframe controls tactical candle context, levels, scalp timing checks, and Delta
- Visible `Direction` + `Confidence` come from closed higher-timeframe anchor candles
- Visible `AI Ensemble` comes from a separate closed AI bias engine using the same adaptive anchors
- Visible `AI Confidence` scores the quality of that higher-timeframe AI verdict

How the 5 key columns are calculated:

1. `Direction` (main spot direction)
- Uses only closed higher-timeframe anchor candles
- Technical engine builds a score for each timeframe from:
  - structure
  - trend
  - momentum
  - regime / location
- A slower anchor sets the broader bias and a faster anchor confirms it

2. `Confidence`
- This is the quality score of the same higher-timeframe technical spot direction

3. `AI Ensemble`
- Separate AI bias engine, also using the closed adaptive anchor pair
- The 3 dots show how many models support the final AI direction

4. `AI Confidence`
- Quality score of the higher-timeframe AI verdict

5. `Setup Confirm`
- This is **not** the main direction
- It answers: “Given the main spot direction, does the selected timeframe look good enough right now?”
- `TREND-led` = technical selected-timeframe confirmation
- `AI-led` = AI selected-timeframe confirmation
- `TREND+AI` = both are strong together
- `EARLY` = early setup
- `WATCH` = developing setup
- `SKIP` = not aligned right now

Scalp Opportunity is separate from Setup Confirm.
It appears only if all execution checks pass:
- Direction match
- Timeframe-adaptive R:R / ADX / Confidence thresholds
- No technical/AI conflict
- Valid entry/stop/target
""",
    },
    "guide.section.spot": {
        "trader": """
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
- `Direction`: higher-timeframe technical spot bias from the adaptive lead/confirm anchor pair
- `AI Ensemble`: higher-timeframe AI bias from the same adaptive anchor pair; the 3 dots show how many ensemble models support that final AI direction
- `AI Confidence`: quality score of that higher-timeframe AI verdict

These 5 columns use the same core logic as Market tab.

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
- Use it as workflow guidance, not as a guaranteed outcome
""",
        "neutral": """
Single-coin spot decision workspace (non-leverage), synchronized with Market tab decision logic.

Read in this order:
1) Setup Snapshot:
- Delta (%)
- Setup Confirm
- Direction
- Confidence
- AI Ensemble
- AI Confidence

These 5 columns use the same core logic as Market tab.

2) Technical Regime Breakdown:
- Trend Structure
- Momentum Signals
- Volatility & Volume

3) Reference Levels:
- Reference Price
- Zone / Trigger
- Risk references
- Upside references

4) Spot Setup Read:
- Scenario-specific guidance driven by Setup Confirm + Direction context
- Use it as setup guidance, not as a guaranteed outcome
""",
    },
    "guide.section.position": {
        "trader": """
Live position management tab.
Main outputs:
- Base PnL, levered PnL, funding effect, net PnL
- Estimated liquidation distance (simple model)
- Direction / Confidence / AI Ensemble / AI Confidence summary
- Technical Invalidation Line (hard risk line)
- Position health decision block (HOLD / REDUCE / EXIT style guidance)
- Optional scalp setup block with context reasons when unavailable
""",
        "neutral": """
Live position management tab.
Main outputs:
- Base PnL, levered PnL, funding effect, net PnL
- Estimated liquidation distance
- Direction / Confidence / AI summary
- Technical Invalidation Line (hard risk line)
- Position read block
- Optional scalp block with context reasons when unavailable
""",
    },
    "guide.section.sessions": {
        "trader": """
Execution timing filter across Asia / Europe / US windows.
Shows:
- relative session quality (not an absolute signal)
- liquidity depth by session
- range profile and drift bias

Use it to decide when execution conditions look cleaner after a setup already exists elsewhere.
""",
        "neutral": """
Timing filter across Asia / Europe / US windows.
Shows:
- relative session quality (not an absolute signal)
- liquidity depth by session
- range profile and drift bias

Use it to decide when conditions look cleaner after a setup already exists elsewhere.
""",
    },
    "guide.section.signal_archive": {
        "trader": """
Smart signal archive and execution review.

Use it to:
- review what the dashboard actually logged in production
- compare the market scan with what you actually took
- inspect a coin directly or let Best Signal surface the strongest learned leader
- study resolved quality by timeframe, session, execution, and hold timing

This tab is not a live entry screen.
It is the system's memory and review surface.
""",
        "neutral": """
Smart signal archive and execution review.

Use it to:
- review what the dashboard actually logged
- compare the market scan with what was actually executed
- inspect a coin directly or let Best Signal surface the strongest learned leader
- study resolved quality by timeframe, session, execution, and hold timing

This tab is not a live action screen.
It is the system's memory and review surface.
""",
    },
    "guide.section.setup_backtest": {
        "trader": """
Setup Lab is a historical simulation layer, not a live archive.

What it does:
- replays the current setup engine on closed candles
- compares ENTER setup families (TREND+AI, TREND-led, AI-led)
- measures forward outcome over the next N bars

Use it for:
- class-level edge checks before changing live policy
- hold-window and expectancy comparisons

Use Signal Archive for:
- what the dashboard actually logged live
- what you actually took or skipped
- journaled execution outcomes
""",
        "neutral": """
Setup Lab is a historical simulation layer, not a live archive.

What it does:
- replays the current setup engine on closed candles
- compares ENTER setup families (TREND+AI, TREND-led, AI-led)
- measures forward outcome over the next N bars

Use it for:
- class-level behavior checks before changing live policy
- hold-window and expectancy comparisons

Use Signal Archive for:
- what the dashboard actually logged live
- what you actually took or skipped
- journaled execution outcomes
""",
    },
    "guide.section.scalp_backtest": {
        "trader": """
Scalp Lab has 2 layers: live scalp archive truth on top, historical study below.

What it does:
- reviews what the live dashboard already logged as scalp
- compares taken / skipped / observed execution outcomes
- replays the current scalp planner and scalp checks on closed candles
- tracks TP-first, SL-first, and timeout behavior by side and by coin

Use it for:
- checking whether live scalp behavior is actually clean
- stress-testing scalp policy before changing thresholds

Use Signal Archive for:
- broader dashboard-wide tracker history
- non-scalp setup learning and archive review
""",
        "neutral": """
Scalp Lab has 2 layers: live scalp archive truth on top, historical study below.

What it does:
- reviews what the live dashboard already logged as scalp
- compares taken / skipped / observed execution outcomes
- replays the current scalp planner and scalp checks on closed candles
- tracks TP-first, SL-first, and timeout behavior by side and by coin

Use it for:
- checking whether live scalp behavior is actually clean
- stress-testing scalp policy before changing thresholds

Use Signal Archive for:
- broader dashboard-wide tracker history
- non-scalp setup learning and archive review
""",
    },
    "guide.section.workflow": {
        "trader": """
Recommended daily flow:
1. Market tab: check regime + market shortlist
2. Spot: validate setup and read the Spot Execution Plan
3. Position: if already in trade, follow Technical Invalidation + decision model first
4. Sessions: check whether the timing window is actually favorable
5. Signal Archive: verify what the system is learning from before trusting archive-driven tilt
6. Correlation / Portfolio Scenario: stress-test basket overlap and anchor risk
7. Fibonacci / Risk Analytics: validate structure and downside risk
8. Tools: confirm R:R and liquidation distance
9. Labs: use Setup / Scalp only when validating policy changes or edge hypotheses

Quick rule:
- If Direction/AI conflict and Health says REDUCE or EXIT, reduce risk first.
- If setup is aligned and Health says HOLD, manage with invalidation discipline.
""",
        "neutral": """
Recommended daily flow:
1. Market tab: check regime + market shortlist
2. Spot: validate setup and read the Spot Setup Read
3. Position: if already in trade, follow Technical Invalidation + decision model first
4. Sessions: check whether the timing window is actually favorable
5. Signal Archive: verify what the system is learning from before trusting archive-driven tilt
6. Correlation / Portfolio Scenario: stress-test basket overlap and anchor risk
7. Fibonacci / Risk Analytics: validate structure and downside risk
8. Tools: confirm R:R and liquidation distance
9. Labs: use Setup / Scalp only when validating research ideas or threshold changes
""",
    },
    "guide.section.limitations": {
        "trader": """
No model can predict news shocks, listing events, outages, or sudden regime breaks.
Treat all outputs as probabilistic guidance.

Non-negotiables:
- Use stop-loss
- Cap per-trade risk
- Avoid revenge/forced trades
- Respect invalidation levels

This dashboard is **not financial advice**.
""",
        "neutral": """
No model can predict news shocks, listing events, outages, or sudden regime breaks.
Treat all outputs as probabilistic guidance.

Non-negotiables:
- Cap risk
- Avoid forced trades
- Respect invalidation levels

This dashboard is **not financial advice**.
""",
    },
    "guide.section.smoke_checklist": {
        "trader": """
Use this 60-second checklist:

1. **Market tab**
- Market table loads and shows multiple rows
- Setup Confirm / Direction / Confidence / AI Ensemble / AI Confidence are not empty

2. **Spot tab**
- Analyse runs and shows Direction + Confidence + higher-timeframe AI + AI Confidence
- Spot setup panel appears

3. **Position tab**
- Base/Levered PnL and Net PnL render correctly
- Technical Invalidation line is visible
- Excel report downloads without resetting analysis view

4. **AI Workspace**
- Quick Prediction mode returns Direction / Probability / Agreement in one run
- Model & Timeframe Matrix mode fills timeframe matrix with plan source fields
- Details panel shows AI vs non-AI plan levels

5. **Portfolio Scenario**
- Basket editor accepts your holdings and anchor target
- Current basket / projected basket / scenario table render in one run
""",
        "neutral": """
Use this 60-second checklist:

1. **Market tab**
- Market table loads and shows multiple rows
- Setup Confirm / Direction / Confidence / AI Ensemble / AI Confidence are not empty

2. **Spot tab**
- Analyse runs and shows Direction + Confidence + higher-timeframe AI + AI Confidence
- Spot setup panel appears

3. **Position tab**
- Base/Levered PnL and Net PnL render correctly
- Technical Invalidation line is visible
- Excel report downloads without resetting analysis view

4. **AI Workspace**
- Quick Prediction mode returns Direction / Probability / Agreement in one run
- Model & Timeframe Matrix mode fills timeframe matrix with plan source fields
- Details panel shows AI vs non-AI plan levels

5. **Portfolio Scenario**
- Basket editor accepts your holdings and anchor target
- Current basket / projected basket / scenario table render in one run
""",
    },
    "advanced_analysis.regime.strong_trend": {
        "trader": "Powerful directional move. Trend-following strategies look favorable.",
        "neutral": "Powerful directional move. Trend-following conditions look favorable.",
    },
    "advanced_analysis.regime.trending": {
        "trader": "Clear directional bias. EMAs and MACD look reliable.",
        "neutral": "Clear directional bias. EMAs and MACD look reliable.",
    },
    "advanced_analysis.regime.compression": {
        "trader": "Extreme low volatility. Breakout potential is building.",
        "neutral": "Extreme low volatility. Breakout potential is building.",
    },
    "advanced_analysis.regime.high_volatility": {
        "trader": "Choppy conditions. Keep size smaller and expect wider risk bands.",
        "neutral": "Choppy conditions. Keep size smaller and expect wider risk bands.",
    },
    "advanced_analysis.regime.ranging": {
        "trader": "No clear trend. Mean-reversion conditions are more likely than clean breakouts.",
        "neutral": "No clear trend. Mean-reversion conditions are more likely than clean breakouts.",
    },
    "advanced_analysis.regime.transitioning": {
        "trader": "Market shifting between regimes. Cleaner confirmation is still needed.",
        "neutral": "Market shifting between regimes. Cleaner confirmation is still needed.",
    },
}


def _normalize_audience(value: object) -> str:
    raw = str(value or "").strip().lower()
    return raw if raw in _VALID_AUDIENCES else DEFAULT_AUDIENCE


def set_copy_audience(audience: object) -> str:
    global _ACTIVE_AUDIENCE
    _ACTIVE_AUDIENCE = _normalize_audience(audience)
    return _ACTIVE_AUDIENCE


def get_copy_audience() -> str:
    env_value = os.getenv("TRADING_COPY_AUDIENCE", "")
    if env_value:
        return _normalize_audience(env_value)
    return _ACTIVE_AUDIENCE


def copy_text(key: str, *, audience: str | None = None, **kwargs: object) -> str:
    entry = _COPY.get(str(key))
    if not entry:
        raise KeyError(f"Unknown presentation copy key: {key}")
    selected_audience = _normalize_audience(audience) if audience is not None else get_copy_audience()
    template = entry.get(selected_audience) or entry.get(DEFAULT_AUDIENCE) or ""
    return str(template).format(**kwargs)


def setup_class_key(value: object) -> str:
    raw = str(value or "").strip().upper().replace("-", "_").replace(" ", "_")
    if raw in {"ENTER_TREND_AI", "TREND+AI", "TREND_AI", "TREND_AND_AI"}:
        return "ENTER_TREND_AI"
    if raw in {"ENTER_TREND_LED", "TREND_LED"}:
        return "ENTER_TREND_LED"
    if raw in {"ENTER_AI_LED", "AI_LED"}:
        return "ENTER_AI_LED"
    if raw in {"PROBE", "EARLY", "EARLY_ENTRY", "EARLY_SETUP"} or raw.startswith("EARLY_"):
        return "PROBE"
    if raw == "WATCH":
        return "WATCH"
    if raw == "SKIP":
        return "SKIP"
    if raw == "ALL":
        return "ALL"
    return "UNKNOWN"


def setup_class_display(value: object, *, audience: str | None = None) -> str:
    key = setup_class_key(value)
    selected_audience = _normalize_audience(audience) if audience is not None else get_copy_audience()
    mapping = {
        "ENTER_TREND_AI": {
            "trader": "TREND+AI",
            "neutral": "Trend + Model Aligned",
        },
        "ENTER_TREND_LED": {
            "trader": "TREND-led",
            "neutral": "Trend-Led Setup",
        },
        "ENTER_AI_LED": {
            "trader": "AI-led",
            "neutral": "Model-Led Setup",
        },
        "PROBE": {
            "trader": "EARLY",
            "neutral": "Early Setup",
        },
        "WATCH": {
            "trader": "WATCH",
            "neutral": "Developing Setup",
        },
        "SKIP": {
            "trader": "SKIP",
            "neutral": "Not Aligned",
        },
        "ALL": {
            "trader": "ALL Setup Confirmations",
            "neutral": "All Setup Classes",
        },
    }
    profile = mapping.get(key)
    if profile:
        return str(profile.get(selected_audience) or profile.get(DEFAULT_AUDIENCE) or key)
    raw = str(value or "").strip()
    return raw or "Unknown"


def playbook_key(value: object) -> str:
    raw = str(value or "").strip().lower()
    if not raw:
        return "UNKNOWN"
    if "wait for confirmation" in raw or "needs confirmation" in raw:
        return "WAIT_CONFIRMATION"
    if "trend continuation" in raw:
        return "TREND_CONTINUATION"
    if "selective upside rotation" in raw:
        return "SELECTIVE_UPSIDE_ROTATION"
    if "mean reversion" in raw or "range/reversion" in raw or "stand aside" in raw:
        return "MEAN_REVERSION_OR_STAND_ASIDE"
    if "selective only" in raw:
        return "SELECTIVE_ONLY"
    if "defensive" in raw or "downside only" in raw or "risk-off" in raw:
        return "DEFENSIVE_DOWNSIDE_ONLY"
    return "OTHER"


def playbook_display(value: object, *, audience: str | None = None) -> str:
    raw_key = str(value or "").strip().upper()
    selected_audience = _normalize_audience(audience) if audience is not None else get_copy_audience()
    mapping = {
        "WAIT_CONFIRMATION": {
            "trader": "Wait for confirmation",
            "neutral": "Needs confirmation",
        },
        "TREND_CONTINUATION": {
            "trader": "Trend continuation",
            "neutral": "Trend continuation",
        },
        "SELECTIVE_UPSIDE_ROTATION": {
            "trader": "Selective upside rotation",
            "neutral": "Selective upside rotation",
        },
        "MEAN_REVERSION_OR_STAND_ASIDE": {
            "trader": "Mean reversion or stand aside",
            "neutral": "Range/reversion or stand aside",
        },
        "SELECTIVE_ONLY": {
            "trader": "Selective only",
            "neutral": "Selective only",
        },
        "DEFENSIVE_DOWNSIDE_ONLY": {
            "trader": "Defensive / downside only",
            "neutral": "Defensive downside context",
        },
    }
    key = raw_key if raw_key in mapping else playbook_key(value)
    profile = mapping.get(key)
    if profile:
        return str(profile.get(selected_audience) or profile.get(DEFAULT_AUDIENCE) or key)
    raw = str(value or "").strip()
    return raw or "Unknown"


def trade_gate_key(value: object) -> str:
    raw = str(value or "").strip().upper().replace("-", "_").replace(" ", "_")
    if raw in {"NO_TRADE", "STAND_ASIDE", "LOW_ALIGNMENT"}:
        return "NO_TRADE"
    if raw in {"DEFENSIVE_ONLY", "CAUTIOUS"}:
        return "DEFENSIVE_ONLY"
    if raw in {"SELECTIVE_ONLY", "SELECTIVE"}:
        return "SELECTIVE_ONLY"
    if raw in {"TRADEABLE", "SUPPORTIVE"}:
        return "TRADEABLE"
    return "UNKNOWN"


def trade_gate_display(value: object, *, audience: str | None = None) -> str:
    key = trade_gate_key(value)
    selected_audience = _normalize_audience(audience) if audience is not None else get_copy_audience()
    mapping = {
        "NO_TRADE": {
            "trader": "Stand Aside",
            "neutral": "Low Alignment",
        },
        "DEFENSIVE_ONLY": {
            "trader": "Defensive Only",
            "neutral": "Cautious",
        },
        "SELECTIVE_ONLY": {
            "trader": "Selective Only",
            "neutral": "Selective",
        },
        "TRADEABLE": {
            "trader": "Tradeable",
            "neutral": "Supportive",
        },
    }
    profile = mapping.get(key)
    if profile:
        return str(profile.get(selected_audience) or profile.get(DEFAULT_AUDIENCE) or key)
    raw = str(value or "").strip()
    return raw or "Unknown"
