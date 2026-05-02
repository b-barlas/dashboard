from __future__ import annotations

import unittest

from core.position_management import build_position_management_snapshot


class PositionManagementTests(unittest.TestCase):
    def test_invalidated_position_forces_exit_now(self) -> None:
        out = build_position_management_snapshot(
            direction="LONG",
            health_label="EXIT",
            health_score=18.0,
            health_notes=["hard invalidation broken"],
            levered_pnl_pct=-12.0,
            liq_distance_pct=3.2,
            leverage=8.0,
            invalidated=True,
            invalidation_distance_pct=0.1,
            spot_direction="DOWNSIDE",
            tactical_direction="DOWNSIDE",
            ai_direction="DOWNSIDE",
            selected_confidence=34.0,
            context_fit_label="Stand Aside",
            context_fit_aggression="No fresh risk",
            adaptive_label="Historically Weak",
            execution_fit_label="Execution Fragile",
            session_fit_label="Session Fragile",
            archive_guardrail_label="Archive Guardrail",
            catalyst_window="Blocking (<6h)",
            trade_gate="No-Trade",
            playbook="Stand aside / mean reversion only",
            flow_proxy="Longs Crowded",
        )
        self.assertEqual(out.action_key, "EXIT")
        self.assertEqual(out.label, "Exit Now")

    def test_supportive_winner_stack_can_press_on_strength(self) -> None:
        out = build_position_management_snapshot(
            direction="LONG",
            health_label="HOLD",
            health_score=82.0,
            health_notes=[],
            levered_pnl_pct=9.5,
            liq_distance_pct=18.0,
            leverage=4.0,
            invalidated=False,
            invalidation_distance_pct=4.5,
            spot_direction="UPSIDE",
            tactical_direction="UPSIDE",
            ai_direction="UPSIDE",
            selected_confidence=74.0,
            context_fit_label="Tradeable",
            context_fit_aggression="Normal aggression",
            adaptive_label="Historically Favored",
            execution_fit_label="Execution Proven",
            session_fit_label="Session Supportive",
            archive_guardrail_label="",
            catalyst_window="Far / Clear",
            trade_gate="Tradeable",
            playbook="Trend continuation",
            flow_proxy="Shorts Crowded",
            volatility_regime="▼ Low",
        )
        self.assertEqual(out.action_key, "PRESS")
        self.assertEqual(out.label, "Press on Strength")
        self.assertIn("add only", out.size_guidance.lower())

    def test_defensive_context_reduces_risk_even_without_full_exit(self) -> None:
        out = build_position_management_snapshot(
            direction="LONG",
            health_label="HOLD",
            health_score=68.0,
            health_notes=["signal conflict"],
            levered_pnl_pct=1.0,
            liq_distance_pct=9.0,
            leverage=6.0,
            invalidated=False,
            invalidation_distance_pct=1.8,
            spot_direction="DOWNSIDE",
            tactical_direction="UPSIDE",
            ai_direction="UPSIDE",
            selected_confidence=58.0,
            context_fit_label="Defensive Only",
            context_fit_aggression="Reduced aggression",
            adaptive_label="Historically Mixed",
            execution_fit_label="Execution Mixed",
            session_fit_label="Session Fragile",
            archive_guardrail_label="Archive Caution",
            catalyst_window="High Impact (6-24h)",
            trade_gate="Selective Only",
            playbook="Selective breakout",
            flow_proxy="Longs Crowded",
            volatility_regime="▲ High",
            exit_quality_label="Late Loss Risk",
            exit_quality_note="This cluster often exits losers late. Cut faster when structure breaks.",
        )
        self.assertEqual(out.action_key, "REDUCE")
        self.assertEqual(out.label, "Reduce Risk")
        self.assertIn("late", out.note.lower())

    def test_hot_volatility_and_high_leverage_block_press_behavior(self) -> None:
        out = build_position_management_snapshot(
            direction="LONG",
            health_label="HOLD",
            health_score=84.0,
            health_notes=[],
            levered_pnl_pct=7.0,
            liq_distance_pct=14.0,
            leverage=10.0,
            invalidated=False,
            invalidation_distance_pct=4.0,
            spot_direction="UPSIDE",
            tactical_direction="UPSIDE",
            ai_direction="UPSIDE",
            selected_confidence=76.0,
            context_fit_label="Tradeable",
            context_fit_aggression="Normal aggression",
            adaptive_label="Historically Favored",
            execution_fit_label="Execution Proven",
            session_fit_label="Session Supportive",
            archive_guardrail_label="",
            catalyst_window="Far / Clear",
            trade_gate="Tradeable",
            playbook="Trend continuation",
            flow_proxy="Shorts Crowded",
            volatility_regime="▲ High",
        )
        self.assertNotEqual(out.action_key, "PRESS")
        self.assertIn(out.action_key, {"REDUCE", "HOLD"})

    def test_fast_adverse_spike_pushes_position_into_reduce_mode(self) -> None:
        out = build_position_management_snapshot(
            direction="LONG",
            health_label="HOLD",
            health_score=77.0,
            health_notes=[],
            levered_pnl_pct=2.0,
            liq_distance_pct=12.0,
            leverage=9.0,
            invalidated=False,
            invalidation_distance_pct=2.6,
            spot_direction="UPSIDE",
            tactical_direction="UPSIDE",
            ai_direction="UPSIDE",
            selected_confidence=73.0,
            context_fit_label="Tradeable",
            context_fit_aggression="Normal aggression",
            adaptive_label="Historically Favored",
            execution_fit_label="Execution Proven",
            session_fit_label="Session Supportive",
            archive_guardrail_label="",
            catalyst_window="Far / Clear",
            trade_gate="Tradeable",
            playbook="Trend continuation",
            flow_proxy="Shorts Crowded",
            volatility_regime="▲ High",
            short_term_move_pct=-3.4,
            volume_spike_label="▼ Down Spike",
        )
        self.assertEqual(out.action_key, "REDUCE")
        self.assertIn("volume spike", out.note.lower())

    def test_signal_archive_hold_window_fills_when_actual_trade_profile_is_building(self) -> None:
        out = build_position_management_snapshot(
            direction="LONG",
            health_label="HOLD",
            health_score=68.0,
            health_notes=[],
            levered_pnl_pct=1.0,
            liq_distance_pct=14.0,
            leverage=4.0,
            invalidated=False,
            invalidation_distance_pct=3.0,
            spot_direction="UPSIDE",
            tactical_direction="UPSIDE",
            ai_direction="UPSIDE",
            selected_confidence=66.0,
            context_fit_label="Tradeable",
            context_fit_aggression="Selective adds only",
            adaptive_label="Historically Mixed",
            execution_fit_label="Execution Mixed",
            session_fit_label="Session Supportive",
            archive_guardrail_label="",
            catalyst_window="Far / Clear",
            trade_gate="Tradeable",
            playbook="Trend continuation",
            flow_proxy="Balanced",
            hold_profile_label="Archive Building",
            hold_profile_note="Closed trade profile is still building.",
            archive_hold_profile_label="Quick Follow-Through",
            archive_hold_profile_note="Signal Archive hold window: best around 4 bars, fades after roughly 8 bars.",
        )

        self.assertIn("Signal Archive hold window", out.note)

    def test_actual_trade_hold_profile_takes_priority_over_signal_archive_fallback(self) -> None:
        out = build_position_management_snapshot(
            direction="LONG",
            health_label="HOLD",
            health_score=68.0,
            health_notes=[],
            levered_pnl_pct=1.0,
            liq_distance_pct=14.0,
            leverage=4.0,
            invalidated=False,
            invalidation_distance_pct=3.0,
            spot_direction="UPSIDE",
            tactical_direction="UPSIDE",
            ai_direction="UPSIDE",
            selected_confidence=66.0,
            context_fit_label="Tradeable",
            context_fit_aggression="Selective adds only",
            adaptive_label="Historically Mixed",
            execution_fit_label="Execution Mixed",
            session_fit_label="Session Supportive",
            archive_guardrail_label="",
            catalyst_window="Far / Clear",
            trade_gate="Tradeable",
            playbook="Trend continuation",
            flow_proxy="Balanced",
            hold_profile_label="Needs Room",
            hold_profile_note="Actual closed trades say winners need room.",
            archive_hold_profile_label="Quick Follow-Through",
            archive_hold_profile_note="Signal Archive hold window should not override actual trade profile.",
        )

        self.assertIn("Actual closed trades say winners need room", out.note)
        self.assertNotIn("Signal Archive hold window should not override", out.note)


if __name__ == "__main__":
    unittest.main()
