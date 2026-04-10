import unittest

from core.confidence import (
    ConfidenceSnapshot,
    ai_confidence_bucket,
    ai_confidence_from_components,
    build_confidence_snapshot,
    build_ai_confidence_snapshot,
    build_execution_confidence_snapshot,
    confidence_bucket,
    confidence_from_components,
    execution_alignment_quality,
    execution_confidence_from_components,
    execution_structure_quality,
    execution_trend_quality,
    normalize_direction,
)


class ConfidenceContractTests(unittest.TestCase):
    def test_normalize_direction_accepts_common_aliases(self) -> None:
        self.assertEqual(normalize_direction("buy"), "UPSIDE")
        self.assertEqual(normalize_direction("SHORT"), "DOWNSIDE")
        self.assertEqual(normalize_direction("neutral"), "NEUTRAL")

    def test_confidence_uses_weighted_component_formula_without_caps(self) -> None:
        score = confidence_from_components(
            direction="Upside",
            timeframe_alignment=100,
            structure_quality=80,
            trend_quality=60,
            regime_quality=40,
            location_quality=20,
        )
        self.assertAlmostEqual(score, 70.0, places=6)

    def test_neutral_direction_caps_confidence(self) -> None:
        score = confidence_from_components(
            direction="Neutral",
            timeframe_alignment=100,
            structure_quality=100,
            trend_quality=100,
            regime_quality=100,
            location_quality=100,
        )
        self.assertEqual(score, 35.0)

    def test_timeframe_conflict_caps_confidence(self) -> None:
        score = confidence_from_components(
            direction="Upside",
            timeframe_alignment=100,
            structure_quality=100,
            trend_quality=100,
            regime_quality=100,
            location_quality=100,
            timeframe_conflict=True,
        )
        self.assertEqual(score, 15.0)

    def test_weak_structure_caps_confidence(self) -> None:
        score = confidence_from_components(
            direction="Downside",
            timeframe_alignment=100,
            structure_quality=20,
            trend_quality=100,
            regime_quality=100,
            location_quality=100,
        )
        self.assertEqual(score, 45.0)

    def test_degraded_data_caps_confidence(self) -> None:
        score = confidence_from_components(
            direction="Upside",
            timeframe_alignment=100,
            structure_quality=100,
            trend_quality=100,
            regime_quality=100,
            location_quality=100,
            degraded_data=True,
        )
        self.assertEqual(score, 55.0)

    def test_range_regime_caps_confidence(self) -> None:
        score = confidence_from_components(
            direction="Downside",
            timeframe_alignment=100,
            structure_quality=100,
            trend_quality=100,
            regime_quality=100,
            location_quality=100,
            range_regime=True,
        )
        self.assertEqual(score, 35.0)

    def test_lowest_cap_wins_when_multiple_risk_flags_exist(self) -> None:
        score = confidence_from_components(
            direction="Neutral",
            timeframe_alignment=100,
            structure_quality=10,
            trend_quality=100,
            regime_quality=100,
            location_quality=100,
            timeframe_conflict=True,
            degraded_data=True,
            range_regime=True,
        )
        self.assertEqual(score, 15.0)

    def test_confidence_bucket_contract(self) -> None:
        self.assertEqual(confidence_bucket(20), "VERY LOW")
        self.assertEqual(confidence_bucket(50), "LOW")
        self.assertEqual(confidence_bucket(70), "MEDIUM")
        self.assertEqual(confidence_bucket(85), "HIGH")

    def test_build_confidence_snapshot_returns_score_and_label(self) -> None:
        snap = build_confidence_snapshot(
            direction="Upside",
            timeframe_alignment=90,
            structure_quality=85,
            trend_quality=80,
            regime_quality=75,
            location_quality=70,
        )
        self.assertIsInstance(snap, ConfidenceSnapshot)
        self.assertGreater(snap.score, 0.0)
        self.assertEqual(snap.label, confidence_bucket(snap.score))

    def test_build_confidence_snapshot_applies_archive_delta_but_respects_caps(self) -> None:
        supportive = build_confidence_snapshot(
            direction="Upside",
            timeframe_alignment=90,
            structure_quality=85,
            trend_quality=80,
            regime_quality=75,
            location_quality=70,
            archive_calibration_delta=4.0,
            archive_calibration_note="Archive confidence calibration is modestly supportive here.",
        )
        neutral_capped = build_confidence_snapshot(
            direction="Neutral",
            timeframe_alignment=100,
            structure_quality=90,
            trend_quality=90,
            regime_quality=90,
            location_quality=90,
            archive_calibration_delta=8.0,
        )
        self.assertGreater(supportive.score, float(supportive.base_score))
        self.assertEqual(supportive.calibration_delta, 4.0)
        self.assertIn("Archive confidence calibration", supportive.note)
        self.assertEqual(neutral_capped.score, 35.0)

    def test_ai_confidence_uses_weighted_htf_quality_formula(self) -> None:
        score = ai_confidence_from_components(
            direction="Upside",
            combined_score=82.0,
            conviction_quality=88.0,
            timeframe_alignment=100.0,
            consensus_quality=72.0,
            support_votes=3,
        )
        self.assertGreater(score, 80.0)

    def test_ai_confidence_caps_neutral_conflict_and_degraded_states(self) -> None:
        neutral_score = ai_confidence_from_components(
            direction="Neutral",
            combined_score=70.0,
            conviction_quality=80.0,
            timeframe_alignment=100.0,
            consensus_quality=90.0,
            support_votes=3,
        )
        conflict_score = ai_confidence_from_components(
            direction="Neutral",
            combined_score=75.0,
            conviction_quality=90.0,
            timeframe_alignment=0.0,
            consensus_quality=90.0,
            support_votes=3,
            timeframe_conflict=True,
        )
        degraded_score = ai_confidence_from_components(
            direction="Upside",
            combined_score=90.0,
            conviction_quality=95.0,
            timeframe_alignment=100.0,
            consensus_quality=95.0,
            support_votes=3,
            degraded_data=True,
        )
        self.assertEqual(neutral_score, 58.0)
        self.assertEqual(conflict_score, 30.0)
        self.assertEqual(degraded_score, 35.0)

    def test_ai_confidence_caps_low_model_support_for_directional_calls(self) -> None:
        score = ai_confidence_from_components(
            direction="Upside",
            combined_score=88.0,
            conviction_quality=92.0,
            timeframe_alignment=100.0,
            consensus_quality=88.0,
            support_votes=1,
        )
        self.assertEqual(score, 59.0)

    def test_build_ai_confidence_snapshot_returns_score_and_label(self) -> None:
        snap = build_ai_confidence_snapshot(
            direction="Downside",
            combined_score=-76.0,
            conviction_quality=81.0,
            timeframe_alignment=100.0,
            consensus_quality=72.0,
            support_votes=2,
        )
        self.assertIsInstance(snap, ConfidenceSnapshot)
        self.assertGreater(snap.score, 0.0)
        self.assertEqual(snap.label, "MEDIUM")
        self.assertEqual(snap.base_score, snap.score)

    def test_build_ai_confidence_snapshot_applies_archive_delta_but_respects_caps(self) -> None:
        supportive = build_ai_confidence_snapshot(
            direction="Upside",
            combined_score=76.0,
            conviction_quality=80.0,
            timeframe_alignment=100.0,
            consensus_quality=72.0,
            support_votes=2,
            archive_calibration_delta=4.0,
            archive_calibration_note="Archive calibration is modestly supportive here.",
        )
        degraded = build_ai_confidence_snapshot(
            direction="Upside",
            combined_score=90.0,
            conviction_quality=95.0,
            timeframe_alignment=100.0,
            consensus_quality=95.0,
            support_votes=3,
            degraded_data=True,
            archive_calibration_delta=8.0,
        )
        self.assertGreater(supportive.score, float(supportive.base_score))
        self.assertEqual(supportive.calibration_delta, 4.0)
        self.assertIn("Archive calibration", supportive.note)
        self.assertEqual(degraded.score, 35.0)

    def test_ai_confidence_bucket_uses_ai_specific_semantics(self) -> None:
        self.assertEqual(
            ai_confidence_bucket(
                92.0,
                direction="Upside",
                support_votes=3,
            ),
            "HIGH",
        )
        self.assertEqual(
            ai_confidence_bucket(
                81.0,
                direction="Upside",
                support_votes=2,
            ),
            "MEDIUM",
        )
        self.assertEqual(
            ai_confidence_bucket(
                58.0,
                direction="Neutral",
                support_votes=3,
            ),
            "MEDIUM",
        )
        self.assertEqual(
            ai_confidence_bucket(
                30.0,
                direction="Neutral",
                support_votes=1,
                timeframe_conflict=True,
            ),
            "VERY LOW",
        )

    def test_execution_trend_quality_contract(self) -> None:
        self.assertLess(execution_trend_quality(10.0), execution_trend_quality(20.0))
        self.assertLess(execution_trend_quality(20.0), execution_trend_quality(35.0))

    def test_execution_structure_quality_contract(self) -> None:
        self.assertGreater(execution_structure_quality("FULL"), execution_structure_quality("TREND"))
        self.assertGreater(execution_structure_quality("TREND"), execution_structure_quality("EARLY"))
        self.assertGreater(execution_structure_quality("EARLY"), execution_structure_quality("NONE"))

    def test_execution_alignment_quality_conflict_is_zero(self) -> None:
        self.assertEqual(execution_alignment_quality("CONFLICT", 0.9), 0.0)
        self.assertGreater(execution_alignment_quality("HIGH", 0.8), execution_alignment_quality("MEDIUM", 0.8))

    def test_execution_confidence_uses_weighted_components(self) -> None:
        score = execution_confidence_from_components(
            direction="Upside",
            bias_score=80.0,
            adx_val=28.0,
            structure_state="TREND",
            conviction_label="MEDIUM",
            ai_agreement=0.7,
        )
        self.assertGreater(score, 60.0)

    def test_execution_confidence_caps_conflict_and_neutral(self) -> None:
        neutral_score = execution_confidence_from_components(
            direction="Neutral",
            bias_score=80.0,
            adx_val=30.0,
            structure_state="TREND",
            conviction_label="MEDIUM",
            ai_agreement=0.8,
        )
        conflict_score = execution_confidence_from_components(
            direction="Upside",
            bias_score=95.0,
            adx_val=40.0,
            structure_state="FULL",
            conviction_label="CONFLICT",
            ai_agreement=0.9,
        )
        self.assertEqual(neutral_score, 35.0)
        self.assertEqual(conflict_score, 15.0)

    def test_build_execution_confidence_snapshot_returns_score_and_label(self) -> None:
        snap = build_execution_confidence_snapshot(
            direction="Downside",
            bias_score=12.0,
            adx_val=32.0,
            structure_state="FULL",
            conviction_label="HIGH",
            ai_agreement=0.8,
        )
        self.assertIsInstance(snap, ConfidenceSnapshot)
        self.assertGreater(snap.score, 0.0)
        self.assertEqual(snap.label, confidence_bucket(snap.score))


if __name__ == "__main__":
    unittest.main()
