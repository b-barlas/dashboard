import unittest

import numpy as np
import pandas as pd

from core.market_decision import (
    ACTION_ENTER_AI_LED,
    ACTION_ENTER_TREND_AI,
    ACTION_ENTER_TREND_LED,
    ACTION_SKIP,
    ACTION_WATCH,
    ai_led_confirmation_snapshot,
    ai_vote_metrics,
    action_decision_with_reason,
    action_rank,
    action_reason_text,
    compact_action_label,
    emerging_bias_snapshot,
    normalize_action_class,
    selected_timeframe_execution_snapshot,
    selected_timeframe_rr_ratio,
    spot_action_decision_with_reason,
    spot_structure_state,
    structure_state,
    trend_ai_confirmation_snapshot,
    trend_led_confirmation_snapshot,
)
from core.ai_spot_bias import AISpotBiasSnapshot, TimeframeAIBiasSnapshot
from core.spot_direction import SpotDirectionSnapshot, TimeframeDirectionSnapshot


class MarketDecisionContractTests(unittest.TestCase):
    @staticmethod
    def _tf_direction_snapshot(
        *,
        timeframe: str,
        direction: str = "NEUTRAL",
        score: float = 0.0,
        raw_score: float = 0.0,
        structure_label: str = "RANGE",
        trend_score: float = 0.0,
        momentum_score: float = 0.0,
    ) -> TimeframeDirectionSnapshot:
        return TimeframeDirectionSnapshot(
            timeframe=timeframe,
            direction=direction,
            score=score,
            raw_score=raw_score,
            structure_score=abs(raw_score),
            structure_label=structure_label,
            trend_score=trend_score,
            momentum_score=momentum_score,
            regime_quality=60.0,
            regime_label="MIXED",
            location_quality=60.0,
            support=90.0,
            resistance=110.0,
            close=100.0,
            degraded=False,
        )

    @staticmethod
    def _spot_snapshot(
        *,
        direction: str = "NEUTRAL",
        one_day: TimeframeDirectionSnapshot | None = None,
        four_hour: TimeframeDirectionSnapshot | None = None,
        degraded_data: bool = False,
        timeframe_conflict: bool = False,
    ) -> SpotDirectionSnapshot:
        one_day = one_day or MarketDecisionContractTests._tf_direction_snapshot(timeframe="1d")
        four_hour = four_hour or MarketDecisionContractTests._tf_direction_snapshot(timeframe="4h")
        return SpotDirectionSnapshot(
            direction=direction,
            score=0.0,
            timeframe_alignment=0.0,
            structure_quality=55.0,
            trend_quality=55.0,
            regime_quality=60.0,
            location_quality=60.0,
            timeframe_conflict=timeframe_conflict,
            degraded_data=degraded_data,
            range_regime=False,
            note="",
            four_hour=four_hour,
            one_day=one_day,
        )

    @staticmethod
    def _tf_ai_snapshot(
        *,
        timeframe: str,
        direction: str = "NEUTRAL",
    ) -> TimeframeAIBiasSnapshot:
        return TimeframeAIBiasSnapshot(
            timeframe=timeframe,
            direction=direction,
            score=70.0 if direction != "NEUTRAL" else 0.0,
            probability_up=0.7 if direction == "UPSIDE" else (0.3 if direction == "DOWNSIDE" else 0.5),
            directional_agreement=2.0 / 3.0 if direction != "NEUTRAL" else 0.0,
            consensus_agreement=2.0 / 3.0 if direction != "NEUTRAL" else 0.0,
            conviction_quality=78.0 if direction != "NEUTRAL" else 0.0,
            consensus_quality=72.0 if direction != "NEUTRAL" else 0.0,
            degraded=False,
            status="ok",
            note="",
            model_votes=(direction, direction, "NEUTRAL"),
        )

    @staticmethod
    def _ai_spot_snapshot(
        *,
        direction: str = "NEUTRAL",
        support_votes: int = 0,
        degraded_data: bool = False,
        timeframe_conflict: bool = False,
    ) -> AISpotBiasSnapshot:
        return AISpotBiasSnapshot(
            direction=direction,
            score=70.0 if direction != "NEUTRAL" else 0.0,
            timeframe_alignment=100.0 if direction != "NEUTRAL" else 0.0,
            conviction_quality=82.0 if direction != "NEUTRAL" else 0.0,
            consensus_quality=74.0 if direction != "NEUTRAL" else 0.0,
            timeframe_conflict=timeframe_conflict,
            degraded_data=degraded_data,
            note="",
            four_hour=MarketDecisionContractTests._tf_ai_snapshot(timeframe="4h", direction=direction),
            one_day=MarketDecisionContractTests._tf_ai_snapshot(timeframe="1d", direction=direction),
            support_votes=support_votes,
        )

    @staticmethod
    def _trend_df(*, n: int = 80, spike_last: bool = False) -> pd.DataFrame:
        ts = pd.date_range("2025-01-01", periods=n, freq="h")
        base = np.linspace(100.0, 118.0, n)
        close = base + np.sin(np.linspace(0, 8, n)) * 0.25
        if spike_last:
            close[-1] = close[-2] + 2.8
        open_ = close - 0.2
        high = close + 0.45
        low = close - 0.45
        volume = np.full(n, 1000.0)
        return pd.DataFrame(
            {
                "timestamp": ts,
                "open": open_,
                "high": high,
                "low": low,
                "close": close,
                "volume": volume,
            }
        )

    @staticmethod
    def _decision(
        signal_dir: str,
        confidence: float,
        structure_state_val: str,
        conviction_label: str,
        agreement: float,
        adx_val: float,
    ) -> str:
        action, _ = action_decision_with_reason(
            signal_dir,
            confidence,
            structure_state_val,
            conviction_label,
            agreement,
            adx_val,
        )
        return action

    @staticmethod
    def _reason(
        signal_dir: str,
        confidence: float,
        structure_state_val: str,
        conviction_label: str,
        agreement: float,
        adx_val: float,
    ) -> str:
        _, reason = action_decision_with_reason(
            signal_dir,
            confidence,
            structure_state_val,
            conviction_label,
            agreement,
            adx_val,
        )
        return reason

    @staticmethod
    def _spot_decision(
        spot_dir: str,
        confidence: float,
        tactical_dir: str,
        ai_dir: str,
        agreement: float,
        adx_val: float,
    ) -> str:
        action, _ = spot_action_decision_with_reason(
            spot_dir,
            confidence,
            tactical_dir,
            ai_dir,
            agreement,
            adx_val,
        )
        return action

    @staticmethod
    def _spot_reason(
        spot_dir: str,
        confidence: float,
        tactical_dir: str,
        ai_dir: str,
        agreement: float,
        adx_val: float,
    ) -> str:
        _, reason = spot_action_decision_with_reason(
            spot_dir,
            confidence,
            tactical_dir,
            ai_dir,
            agreement,
            adx_val,
        )
        return reason

    def test_ai_vote_metrics_uses_consensus_only_for_neutral_display(self):
        votes, display_ratio, decision_agree = ai_vote_metrics("NEUTRAL", 0.0, 2.0 / 3.0)
        self.assertEqual(votes, 2)
        self.assertAlmostEqual(display_ratio, 2.0 / 3.0, places=6)
        self.assertEqual(decision_agree, 0.0)

    def test_ai_vote_metrics_uses_directional_agreement_for_directional_ai(self):
        votes, display_ratio, decision_agree = ai_vote_metrics("UPSIDE", 2.0 / 3.0, 1.0 / 3.0)
        self.assertEqual(votes, 2)
        self.assertAlmostEqual(display_ratio, 2.0 / 3.0, places=6)
        self.assertAlmostEqual(decision_agree, 2.0 / 3.0, places=6)

    def test_structure_state_full(self):
        self.assertEqual(structure_state("LONG", "LONG", 68, 0.8), "FULL")

    def test_structure_state_trend(self):
        self.assertEqual(structure_state("SHORT", "NEUTRAL", 72, 0.0), "TREND")

    def test_structure_state_early(self):
        self.assertEqual(structure_state("LONG", "NEUTRAL", 56, 0.0), "EARLY")

    def test_spot_structure_state_full(self):
        self.assertEqual(spot_structure_state("UPSIDE", "UPSIDE", "UPSIDE", 82, 0.8), "FULL")

    def test_spot_structure_state_trend_led(self):
        self.assertEqual(spot_structure_state("UPSIDE", "UPSIDE", "NEUTRAL", 71, 0.0), "TREND")

    def test_spot_structure_state_ai_led(self):
        self.assertEqual(spot_structure_state("UPSIDE", "NEUTRAL", "UPSIDE", 72, 0.85), "EARLY")

    def test_action_requires_valid_direction(self):
        out = self._decision("NEUTRAL", 70, "FULL", "HIGH", 0.8, 30.0)
        self.assertEqual(out, "⛔ SKIP")

    def test_action_enter_from_core_quality_inputs(self):
        out = self._decision("LONG", 70, "FULL", "HIGH", 0.8, 30.0)
        self.assertEqual(out, "✅ ENTER (Trend+AI)")

    def test_action_skip_on_conflict(self):
        out = self._decision("LONG", 70, "FULL", "CONFLICT", 0.8, 30.0)
        self.assertEqual(out, "⛔ SKIP")

    def test_action_skip_on_low_confidence(self):
        out = self._decision("SHORT", 34, "TREND", "MEDIUM", 0.75, 29.0)
        self.assertEqual(out, "⛔ SKIP")

    def test_action_enter_on_strict_gate(self):
        out = self._decision("SHORT", 62, "FULL", "HIGH", 0.7, 26.0)
        self.assertEqual(out, "✅ ENTER (Trend+AI)")

    def test_action_enter_on_trend_led_path(self):
        out = self._decision("LONG", 58, "TREND", "WEAK", 0.40, 23.0)
        self.assertEqual(out, "🟡 ENTER (Trend-Led)")

    def test_action_enter_trend_led_when_trend_is_exceptional(self):
        out = self._decision("LONG", 74, "TREND", "TREND", 0.0, 27.0)
        self.assertEqual(out, "🟡 ENTER (Trend-Led)")

    def test_action_enter_trend_led_when_structure_is_early(self):
        out = self._decision("LONG", 58, "EARLY", "WEAK", 0.20, 22.0)
        self.assertEqual(out, "🟡 ENTER (Trend-Led)")

    def test_action_enter_ai_led_when_agreement_is_exceptional(self):
        out = self._decision("LONG", 46, "EARLY", "WEAK", 0.85, 20.0)
        self.assertEqual(out, "🟡 ENTER (AI-Led)")

    def test_action_prefers_ai_led_when_agreement_is_exceptional(self):
        out = self._decision("LONG", 58, "EARLY", "WEAK", 0.85, 22.0)
        self.assertEqual(out, "🟡 ENTER (AI-Led)")

    def test_action_watch_when_adx_unknown(self):
        out = self._decision("LONG", 74, "FULL", "HIGH", 0.8, float("nan"))
        self.assertEqual(out, "WATCH")

    def test_action_watch_when_adx_is_weak_even_if_quality_is_high(self):
        out = self._decision("LONG", 74, "FULL", "HIGH", 0.8, 19.9)
        self.assertEqual(out, "WATCH")

    def test_action_reason_conflict(self):
        out = self._reason("LONG", 70, "FULL", "CONFLICT", 0.8, 30.0)
        self.assertEqual(out, "TECH_AI_CONFLICT")

    def test_action_reason_enter_trend_ai(self):
        out = self._reason("LONG", 70, "FULL", "HIGH", 0.8, 30.0)
        self.assertEqual(out, "ENTER_TREND_AI")

    def test_emerging_bias_snapshot_flags_bullish_4h_leader_with_ai_confirmation(self):
        snap = emerging_bias_snapshot(
            spot_snapshot=self._spot_snapshot(
                direction="NEUTRAL",
                one_day=self._tf_direction_snapshot(
                    timeframe="1d",
                    direction="NEUTRAL",
                    score=6.0,
                    raw_score=5.0,
                    structure_label="RANGE",
                    trend_score=6.0,
                ),
                four_hour=self._tf_direction_snapshot(
                    timeframe="4h",
                    direction="UPSIDE",
                    score=18.0,
                    raw_score=15.0,
                    structure_label="BREAKOUT_UP",
                    trend_score=12.0,
                    momentum_score=8.0,
                ),
            ),
            ai_spot_snapshot=self._ai_spot_snapshot(direction="UPSIDE", support_votes=3),
            ai_confidence_score=76.0,
        )
        self.assertTrue(snap.active)
        self.assertEqual(snap.direction, "UPSIDE")
        self.assertEqual(snap.label, "Emerging Upside")

    def test_emerging_bias_snapshot_allows_neutral_ai_when_4h_is_leading(self):
        snap = emerging_bias_snapshot(
            spot_snapshot=self._spot_snapshot(
                direction="NEUTRAL",
                one_day=self._tf_direction_snapshot(
                    timeframe="1d",
                    direction="NEUTRAL",
                    score=6.0,
                    raw_score=5.0,
                    structure_label="RANGE",
                    trend_score=6.0,
                ),
                four_hour=self._tf_direction_snapshot(
                    timeframe="4h",
                    direction="UPSIDE",
                    score=18.0,
                    raw_score=15.0,
                    structure_label="BREAKOUT_UP",
                    trend_score=12.0,
                    momentum_score=8.0,
                ),
            ),
            ai_spot_snapshot=self._ai_spot_snapshot(direction="NEUTRAL", support_votes=0),
            ai_confidence_score=42.0,
            tech_confidence_score=35.0,
        )
        self.assertTrue(snap.active)
        self.assertEqual(snap.direction, "UPSIDE")

    def test_emerging_bias_snapshot_flags_directional_4h_leader_even_without_breakout_label(self):
        snap = emerging_bias_snapshot(
            spot_snapshot=self._spot_snapshot(
                direction="NEUTRAL",
                one_day=self._tf_direction_snapshot(
                    timeframe="1d",
                    direction="NEUTRAL",
                    score=4.0,
                    raw_score=3.0,
                    structure_label="RANGE",
                    trend_score=4.0,
                ),
                four_hour=self._tf_direction_snapshot(
                    timeframe="4h",
                    direction="UPSIDE",
                    score=16.0,
                    raw_score=12.0,
                    structure_label="RANGE",
                    trend_score=8.0,
                    momentum_score=6.0,
                ),
            ),
            ai_spot_snapshot=self._ai_spot_snapshot(direction="UPSIDE", support_votes=2),
            ai_confidence_score=64.0,
            tech_confidence_score=35.0,
        )
        self.assertTrue(snap.active)
        self.assertEqual(snap.direction, "UPSIDE")
        self.assertEqual(snap.label, "Emerging Upside")

    def test_emerging_bias_snapshot_blocks_on_strong_opposite_ai(self):
        snap = emerging_bias_snapshot(
            spot_snapshot=self._spot_snapshot(
                direction="NEUTRAL",
                one_day=self._tf_direction_snapshot(
                    timeframe="1d",
                    direction="NEUTRAL",
                    score=4.0,
                    raw_score=3.0,
                    structure_label="RANGE",
                    trend_score=4.0,
                ),
                four_hour=self._tf_direction_snapshot(
                    timeframe="4h",
                    direction="UPSIDE",
                    score=16.0,
                    raw_score=12.0,
                    structure_label="RANGE",
                    trend_score=8.0,
                    momentum_score=6.0,
                ),
            ),
            ai_spot_snapshot=self._ai_spot_snapshot(direction="DOWNSIDE", support_votes=3),
            ai_confidence_score=78.0,
            tech_confidence_score=35.0,
        )
        self.assertFalse(snap.active)
        self.assertEqual(snap.direction, "NEUTRAL")

    def test_emerging_bias_snapshot_can_show_when_direction_exists_but_confidence_is_still_low(self):
        snap = emerging_bias_snapshot(
            spot_snapshot=self._spot_snapshot(
                direction="UPSIDE",
                one_day=self._tf_direction_snapshot(
                    timeframe="1d",
                    direction="NEUTRAL",
                    score=4.0,
                    raw_score=3.0,
                    structure_label="RANGE",
                    trend_score=4.0,
                ),
                four_hour=self._tf_direction_snapshot(
                    timeframe="4h",
                    direction="UPSIDE",
                    score=16.0,
                    raw_score=12.0,
                    structure_label="RANGE",
                    trend_score=8.0,
                    momentum_score=6.0,
                ),
            ),
            ai_spot_snapshot=self._ai_spot_snapshot(direction="NEUTRAL", support_votes=0),
            ai_confidence_score=42.0,
            tech_confidence_score=45.0,
        )
        self.assertTrue(snap.active)
        self.assertEqual(snap.direction, "UPSIDE")

    def test_spot_action_requires_valid_direction(self):
        out = self._spot_decision("NEUTRAL", 80, "UPSIDE", "UPSIDE", 0.8, 25.0)
        self.assertEqual(out, ACTION_SKIP)

    def test_spot_action_skips_on_low_confidence(self):
        out = self._spot_decision("UPSIDE", 44, "UPSIDE", "UPSIDE", 0.8, 25.0)
        self.assertEqual(out, ACTION_SKIP)

    def test_spot_action_enters_trend_plus_ai(self):
        out = self._spot_decision("UPSIDE", 82, "UPSIDE", "UPSIDE", 0.82, 26.0)
        self.assertEqual(out, ACTION_ENTER_TREND_AI)

    def test_spot_action_enters_trend_led_when_ai_is_neutral(self):
        out = self._spot_decision("UPSIDE", 72, "UPSIDE", "NEUTRAL", 0.0, 24.0)
        self.assertEqual(out, ACTION_ENTER_TREND_LED)

    def test_spot_action_enters_ai_led_when_ai_confirms_and_trend_is_neutral(self):
        out = self._spot_decision("UPSIDE", 71, "NEUTRAL", "UPSIDE", 0.84, 23.0)
        self.assertEqual(out, ACTION_ENTER_AI_LED)

    def test_spot_action_watches_when_tactical_trend_opposes_spot(self):
        out = self._spot_decision("UPSIDE", 78, "DOWNSIDE", "NEUTRAL", 0.0, 24.0)
        self.assertEqual(out, ACTION_WATCH)

    def test_spot_action_reason_reports_ai_spot_conflict(self):
        out = self._spot_reason("UPSIDE", 78, "UPSIDE", "DOWNSIDE", 0.8, 24.0)
        self.assertEqual(out, "AI_SPOT_CONFLICT")

    def test_spot_action_watches_when_trend_motor_is_only_blocked_by_low_adx(self):
        trend_snap = trend_led_confirmation_snapshot(
            spot_dir="UPSIDE",
            spot_confidence=78.0,
            tactical_dir="UPSIDE",
            adx_val=16.0,
            structure_quality=86.0,
            trend_quality=84.0,
            regime_quality=80.0,
            location_quality=78.0,
            rr_ratio=2.2,
        )
        action, reason = spot_action_decision_with_reason(
            "UPSIDE",
            78.0,
            "UPSIDE",
            "NEUTRAL",
            0.0,
            16.0,
            trend_led_snapshot=trend_snap,
        )
        self.assertEqual(action, ACTION_WATCH)
        self.assertEqual(reason, "ADX_TOO_LOW")

    def test_spot_action_skips_when_rr_is_too_low_for_all_paths(self):
        trend_snap = trend_led_confirmation_snapshot(
            spot_dir="UPSIDE",
            spot_confidence=78.0,
            tactical_dir="UPSIDE",
            adx_val=26.0,
            structure_quality=86.0,
            trend_quality=84.0,
            regime_quality=80.0,
            location_quality=78.0,
            rr_ratio=1.45,
        )
        ai_snap = ai_led_confirmation_snapshot(
            spot_dir="UPSIDE",
            spot_confidence=78.0,
            ai_dir="UPSIDE",
            ai_probability=0.74,
            directional_agreement=1.0,
            consensus_agreement=1.0,
            adx_val=26.0,
            location_quality=78.0,
            rr_ratio=1.45,
            ai_status="",
        )
        action, reason = spot_action_decision_with_reason(
            "UPSIDE",
            78.0,
            "UPSIDE",
            "UPSIDE",
            0.84,
            26.0,
            trend_led_snapshot=trend_snap,
            ai_led_snapshot=ai_snap,
        )
        self.assertEqual(action, ACTION_SKIP)
        self.assertEqual(reason, "RR_TOO_LOW")

    def test_trend_led_snapshot_ready_when_technical_continuation_is_clean(self):
        snap = trend_led_confirmation_snapshot(
            spot_dir="UPSIDE",
            spot_confidence=78.0,
            tactical_dir="UPSIDE",
            adx_val=26.0,
            structure_quality=86.0,
            trend_quality=84.0,
            regime_quality=80.0,
            location_quality=78.0,
            rr_ratio=2.2,
        )
        self.assertEqual(snap.state, "READY")
        self.assertEqual(snap.reason_code, "ENTER_TREND_LED")

    def test_trend_led_snapshot_watches_when_location_is_poor(self):
        snap = trend_led_confirmation_snapshot(
            spot_dir="UPSIDE",
            spot_confidence=78.0,
            tactical_dir="UPSIDE",
            adx_val=26.0,
            structure_quality=82.0,
            trend_quality=80.0,
            regime_quality=74.0,
            location_quality=40.0,
            rr_ratio=2.0,
        )
        self.assertEqual(snap.state, "WATCH")
        self.assertEqual(snap.reason_code, "POOR_LOCATION")

    def test_trend_led_snapshot_watches_when_tactical_direction_is_neutral(self):
        snap = trend_led_confirmation_snapshot(
            spot_dir="UPSIDE",
            spot_confidence=78.0,
            tactical_dir="NEUTRAL",
            adx_val=26.0,
            structure_quality=82.0,
            trend_quality=80.0,
            regime_quality=74.0,
            location_quality=72.0,
            rr_ratio=2.0,
        )
        self.assertEqual(snap.state, "WATCH")
        self.assertEqual(snap.reason_code, "TACTICAL_NEUTRAL")

    def test_trend_led_snapshot_watches_when_selected_timeframe_trend_strength_is_too_low(self):
        snap = trend_led_confirmation_snapshot(
            spot_dir="UPSIDE",
            spot_confidence=78.0,
            tactical_dir="UPSIDE",
            adx_val=16.0,
            structure_quality=82.0,
            trend_quality=80.0,
            regime_quality=74.0,
            location_quality=72.0,
            rr_ratio=2.0,
        )
        self.assertEqual(snap.state, "WATCH")
        self.assertEqual(snap.reason_code, "ADX_TOO_LOW")

    def test_trend_led_snapshot_watches_when_execution_score_is_too_low(self):
        snap = trend_led_confirmation_snapshot(
            spot_dir="UPSIDE",
            spot_confidence=78.0,
            tactical_dir="UPSIDE",
            adx_val=26.0,
            structure_quality=62.0,
            trend_quality=58.0,
            regime_quality=45.0,
            location_quality=58.0,
            rr_ratio=1.9,
        )
        self.assertEqual(snap.state, "WATCH")
        self.assertEqual(snap.reason_code, "TREND_SCORE_TOO_LOW")

    def test_selected_timeframe_execution_snapshot_rewards_clean_continuation(self):
        snap = selected_timeframe_execution_snapshot(
            df=self._trend_df(),
            direction="UPSIDE",
            bias_score=88.0,
            adx_val=30.0,
            supertrend_trend="Bullish",
            ichimoku_trend="Bullish",
            vwap_label="🟢 Above",
            psar_trend="▲ Bullish",
            bollinger_bias="→ Neutral",
            williams_label="🟡 Neutral",
            cci_label="🟡 Neutral",
        )
        self.assertGreaterEqual(float(snap.structure_quality), 80.0)
        self.assertGreaterEqual(float(snap.trend_quality), 75.0)
        self.assertGreaterEqual(float(snap.regime_quality), 70.0)

    def test_selected_timeframe_execution_snapshot_penalizes_late_location(self):
        snap = selected_timeframe_execution_snapshot(
            df=self._trend_df(spike_last=True),
            direction="UPSIDE",
            bias_score=86.0,
            adx_val=28.0,
            supertrend_trend="Bullish",
            ichimoku_trend="Bullish",
            vwap_label="🟢 Above",
            psar_trend="▲ Bullish",
            bollinger_bias="🔴 Overbought",
            williams_label="🔴 Overbought",
            cci_label="🔴 Overbought",
        )
        self.assertLess(float(snap.location_quality), 55.0)

    def test_selected_timeframe_rr_ratio_rewards_clean_upside_room(self):
        snap = selected_timeframe_execution_snapshot(
            df=self._trend_df(),
            direction="UPSIDE",
            bias_score=88.0,
            adx_val=30.0,
            supertrend_trend="Bullish",
            ichimoku_trend="Bullish",
            vwap_label="🟢 Above",
            psar_trend="▲ Bullish",
            bollinger_bias="→ Neutral",
            williams_label="🟡 Neutral",
            cci_label="🟡 Neutral",
        )
        rr = selected_timeframe_rr_ratio(snap, direction="UPSIDE")
        self.assertGreater(rr, 1.0)

    def test_selected_timeframe_rr_ratio_penalizes_late_upside_extension(self):
        snap = selected_timeframe_execution_snapshot(
            df=self._trend_df(spike_last=True),
            direction="UPSIDE",
            bias_score=86.0,
            adx_val=28.0,
            supertrend_trend="Bullish",
            ichimoku_trend="Bullish",
            vwap_label="🟢 Above",
            psar_trend="▲ Bullish",
            bollinger_bias="🔴 Overbought",
            williams_label="🔴 Overbought",
            cci_label="🔴 Overbought",
        )
        rr = selected_timeframe_rr_ratio(snap, direction="UPSIDE")
        self.assertLess(rr, 1.7)

    def test_ai_led_snapshot_ready_when_ai_edge_is_clean(self):
        snap = ai_led_confirmation_snapshot(
            spot_dir="UPSIDE",
            spot_confidence=78.0,
            ai_dir="UPSIDE",
            ai_probability=0.74,
            directional_agreement=1.0,
            consensus_agreement=1.0,
            adx_val=26.0,
            location_quality=78.0,
            rr_ratio=2.2,
            ai_status="",
        )
        self.assertEqual(snap.state, "READY")
        self.assertEqual(snap.reason_code, "ENTER_AI_LED")

    def test_ai_led_snapshot_watches_when_ai_is_unavailable(self):
        snap = ai_led_confirmation_snapshot(
            spot_dir="UPSIDE",
            spot_confidence=78.0,
            ai_dir="UPSIDE",
            ai_probability=0.74,
            directional_agreement=1.0,
            consensus_agreement=1.0,
            adx_val=26.0,
            location_quality=78.0,
            rr_ratio=2.2,
            ai_status="insufficient_features",
        )
        self.assertEqual(snap.state, "WATCH")
        self.assertEqual(snap.reason_code, "AI_UNAVAILABLE")

    def test_ai_led_snapshot_watches_when_selected_timeframe_trend_strength_is_too_low(self):
        snap = ai_led_confirmation_snapshot(
            spot_dir="UPSIDE",
            spot_confidence=78.0,
            ai_dir="UPSIDE",
            ai_probability=0.74,
            directional_agreement=1.0,
            consensus_agreement=1.0,
            adx_val=16.0,
            location_quality=78.0,
            rr_ratio=2.2,
            ai_status="",
        )
        self.assertEqual(snap.state, "WATCH")
        self.assertEqual(snap.reason_code, "ADX_TOO_LOW")

    def test_ai_led_snapshot_watches_when_ai_is_neutral(self):
        snap = ai_led_confirmation_snapshot(
            spot_dir="UPSIDE",
            spot_confidence=78.0,
            ai_dir="NEUTRAL",
            ai_probability=0.50,
            directional_agreement=0.0,
            consensus_agreement=0.67,
            adx_val=26.0,
            location_quality=78.0,
            rr_ratio=2.2,
            ai_status="",
        )
        self.assertEqual(snap.state, "WATCH")
        self.assertEqual(snap.reason_code, "AI_NEUTRAL")

    def test_ai_led_snapshot_watches_when_ai_edge_is_weak(self):
        snap = ai_led_confirmation_snapshot(
            spot_dir="UPSIDE",
            spot_confidence=78.0,
            ai_dir="UPSIDE",
            ai_probability=0.60,
            directional_agreement=0.55,
            consensus_agreement=0.67,
            adx_val=26.0,
            location_quality=78.0,
            rr_ratio=2.2,
            ai_status="",
        )
        self.assertEqual(snap.state, "WATCH")
        self.assertEqual(snap.reason_code, "AI_EDGE_WEAK")

    def test_trend_ai_snapshot_ready_only_when_both_motors_are_elite(self):
        trend_snap = trend_led_confirmation_snapshot(
            spot_dir="UPSIDE",
            spot_confidence=82.0,
            tactical_dir="UPSIDE",
            adx_val=30.0,
            structure_quality=90.0,
            trend_quality=88.0,
            regime_quality=84.0,
            location_quality=78.0,
            rr_ratio=2.4,
        )
        ai_snap = ai_led_confirmation_snapshot(
            spot_dir="UPSIDE",
            spot_confidence=82.0,
            ai_dir="UPSIDE",
            ai_probability=0.76,
            directional_agreement=1.0,
            consensus_agreement=1.0,
            adx_val=30.0,
            location_quality=78.0,
            rr_ratio=2.4,
            ai_status="",
        )
        dual = trend_ai_confirmation_snapshot(
            spot_confidence=82.0,
            trend_led_snapshot=trend_snap,
            ai_led_snapshot=ai_snap,
        )
        self.assertEqual(dual.state, "READY")
        self.assertEqual(dual.reason_code, "ENTER_TREND_AI")

    def test_trend_ai_snapshot_watches_when_both_motors_are_only_borderline_ready(self):
        trend_snap = trend_led_confirmation_snapshot(
            spot_dir="UPSIDE",
            spot_confidence=80.0,
            tactical_dir="UPSIDE",
            adx_val=30.0,
            structure_quality=85.0,
            trend_quality=82.0,
            regime_quality=78.0,
            location_quality=74.0,
            rr_ratio=2.0,
        )
        ai_snap = ai_led_confirmation_snapshot(
            spot_dir="UPSIDE",
            spot_confidence=80.0,
            ai_dir="UPSIDE",
            ai_probability=0.70,
            directional_agreement=0.78,
            consensus_agreement=1.0,
            adx_val=30.0,
            location_quality=74.0,
            rr_ratio=2.0,
            ai_status="",
        )
        self.assertEqual(trend_snap.state, "READY")
        self.assertEqual(ai_snap.state, "READY")
        dual = trend_ai_confirmation_snapshot(
            spot_confidence=80.0,
            trend_led_snapshot=trend_snap,
            ai_led_snapshot=ai_snap,
        )
        self.assertEqual(dual.state, "WATCH")
        self.assertEqual(dual.reason_code, "DUAL_NOT_ELITE")

    def test_spot_action_trend_led_ignores_ai_opposition_when_trend_snapshot_is_ready(self):
        trend_snap = trend_led_confirmation_snapshot(
            spot_dir="UPSIDE",
            spot_confidence=80.0,
            tactical_dir="UPSIDE",
            adx_val=27.0,
            structure_quality=85.0,
            trend_quality=82.0,
            regime_quality=76.0,
            location_quality=74.0,
            rr_ratio=2.1,
        )
        action, reason = spot_action_decision_with_reason(
            "UPSIDE",
            80.0,
            "UPSIDE",
            "DOWNSIDE",
            0.82,
            27.0,
            trend_led_snapshot=trend_snap,
        )
        self.assertEqual(action, ACTION_ENTER_TREND_LED)
        self.assertEqual(reason, "ENTER_TREND_LED")

    def test_spot_action_falls_back_to_trend_led_when_dual_confirmation_is_not_elite(self):
        trend_snap = trend_led_confirmation_snapshot(
            spot_dir="UPSIDE",
            spot_confidence=80.0,
            tactical_dir="UPSIDE",
            adx_val=30.0,
            structure_quality=85.0,
            trend_quality=82.0,
            regime_quality=78.0,
            location_quality=74.0,
            rr_ratio=2.0,
        )
        ai_snap = ai_led_confirmation_snapshot(
            spot_dir="UPSIDE",
            spot_confidence=80.0,
            ai_dir="UPSIDE",
            ai_probability=0.70,
            directional_agreement=0.78,
            consensus_agreement=1.0,
            adx_val=30.0,
            location_quality=74.0,
            rr_ratio=2.0,
            ai_status="",
        )
        action, reason = spot_action_decision_with_reason(
            "UPSIDE",
            80.0,
            "UPSIDE",
            "UPSIDE",
            0.78,
            30.0,
            trend_led_snapshot=trend_snap,
            ai_led_snapshot=ai_snap,
        )
        self.assertEqual(action, ACTION_ENTER_TREND_LED)
        self.assertEqual(reason, "ENTER_TREND_LED")

    def test_spot_action_ai_led_ignores_trend_opposition_when_ai_is_exceptional(self):
        trend_snap = trend_led_confirmation_snapshot(
            spot_dir="UPSIDE",
            spot_confidence=80.0,
            tactical_dir="DOWNSIDE",
            adx_val=27.0,
            structure_quality=85.0,
            trend_quality=82.0,
            regime_quality=76.0,
            location_quality=74.0,
            rr_ratio=2.1,
        )
        ai_snap = ai_led_confirmation_snapshot(
            spot_dir="UPSIDE",
            spot_confidence=80.0,
            ai_dir="UPSIDE",
            ai_probability=0.76,
            directional_agreement=1.0,
            consensus_agreement=1.0,
            adx_val=27.0,
            location_quality=74.0,
            rr_ratio=2.1,
            ai_status="",
        )
        action, reason = spot_action_decision_with_reason(
            "UPSIDE",
            80.0,
            "DOWNSIDE",
            "UPSIDE",
            0.84,
            27.0,
            trend_led_snapshot=trend_snap,
            ai_led_snapshot=ai_snap,
        )
        self.assertEqual(action, ACTION_ENTER_AI_LED)
        self.assertEqual(reason, "ENTER_AI_LED")

    def test_spot_action_watches_when_ai_snapshot_reports_unavailable(self):
        ai_snap = ai_led_confirmation_snapshot(
            spot_dir="UPSIDE",
            spot_confidence=80.0,
            ai_dir="UPSIDE",
            ai_probability=0.80,
            directional_agreement=1.0,
            consensus_agreement=1.0,
            adx_val=27.0,
            location_quality=76.0,
            rr_ratio=2.0,
            ai_status="model_exception",
        )
        action, reason = spot_action_decision_with_reason(
            "UPSIDE",
            80.0,
            "NEUTRAL",
            "UPSIDE",
            0.90,
            27.0,
            ai_led_snapshot=ai_snap,
        )
        self.assertEqual(action, ACTION_WATCH)
        self.assertEqual(reason, "AI_UNAVAILABLE")

    def test_action_decision_with_reason_contract(self):
        action, reason = action_decision_with_reason("LONG", 70, "FULL", "HIGH", 0.8, 30.0)
        self.assertEqual(action, ACTION_ENTER_TREND_AI)
        self.assertEqual(reason, "ENTER_TREND_AI")

    def test_action_class_normalization_contract(self):
        self.assertEqual(normalize_action_class(ACTION_ENTER_TREND_AI), "ENTER_TREND_AI")
        self.assertEqual(normalize_action_class(ACTION_ENTER_TREND_LED), "ENTER_TREND_LED")
        self.assertEqual(normalize_action_class(ACTION_ENTER_AI_LED), "ENTER_AI_LED")
        self.assertEqual(normalize_action_class(ACTION_WATCH), "WATCH")
        self.assertEqual(normalize_action_class(ACTION_SKIP), "SKIP")

    def test_action_rank_contract(self):
        self.assertEqual(action_rank(ACTION_ENTER_TREND_AI), 3)
        self.assertEqual(action_rank(ACTION_WATCH), 2)
        self.assertEqual(action_rank(ACTION_SKIP), 1)

    def test_compact_action_label_contract(self):
        self.assertEqual(compact_action_label(ACTION_ENTER_TREND_AI), "ENTER T+AI")
        self.assertEqual(compact_action_label(ACTION_ENTER_TREND_LED), "ENTER Trend")
        self.assertEqual(compact_action_label(ACTION_ENTER_AI_LED), "ENTER AI")
        self.assertEqual(compact_action_label(ACTION_WATCH), "WATCH")
        self.assertEqual(compact_action_label(ACTION_SKIP), "SKIP")

    def test_action_reason_text_contract(self):
        self.assertIn("Trend and AI align", action_reason_text("ENTER_TREND_AI"))
        self.assertIn("not strong enough yet", action_reason_text("AI_EDGE_WEAK"))


if __name__ == "__main__":
    unittest.main()
