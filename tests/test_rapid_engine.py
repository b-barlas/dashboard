from __future__ import annotations

import unittest

from core.rapid_engine import (
    compute_rapid_score,
    decide_action,
    grade_from_score,
    structure_state,
    summarize_quality_history,
)


class RapidEngineTests(unittest.TestCase):
    def test_structure_state(self) -> None:
        self.assertEqual(structure_state("LONG", "LONG", "LONG"), "FULL")
        self.assertEqual(structure_state("LONG", "LONG", "NEUTRAL"), "TREND")
        self.assertEqual(structure_state("LONG", "SHORT", "LONG"), "EARLY")
        self.assertEqual(structure_state("", "LONG", "LONG"), "NONE")

    def test_score_prefers_high_structure_quality(self) -> None:
        hi = compute_rapid_score(
            signal_dir="LONG",
            strength=78,
            structure="FULL",
            conviction_label="HIGH",
            ai_dir="LONG",
            agreement=0.80,
            adx=28.0,
            rr=1.8,
            has_plan=True,
        )
        lo = compute_rapid_score(
            signal_dir="LONG",
            strength=55,
            structure="EARLY",
            conviction_label="WEAK",
            ai_dir="SHORT",
            agreement=0.30,
            adx=14.0,
            rr=1.0,
            has_plan=True,
        )
        self.assertGreater(hi, lo)

    def test_action_rules(self) -> None:
        self.assertEqual(
            decide_action(
                signal_dir="LONG",
                strength=72,
                structure="FULL",
                conviction_label="HIGH",
                ai_dir="LONG",
                score=82,
                has_plan=True,
            ),
            "READY",
        )
        self.assertEqual(
            decide_action(
                signal_dir="LONG",
                strength=58,
                structure="TREND",
                conviction_label="MEDIUM",
                ai_dir="NEUTRAL",
                score=68,
                has_plan=True,
            ),
            "WAIT",
        )
        self.assertEqual(
            decide_action(
                signal_dir="LONG",
                strength=75,
                structure="NONE",
                conviction_label="CONFLICT",
                ai_dir="SHORT",
                score=80,
                has_plan=False,
            ),
            "WAIT",
        )

    def test_grade_and_history_summary(self) -> None:
        self.assertEqual(grade_from_score(85.0), "A+")
        self.assertEqual(grade_from_score(70.0), "B")

        s = summarize_quality_history(
            [
                {"best_action": "READY", "best_score": 80, "qualified_count": 4, "strong_adx_share": 0.6},
                {"best_action": "WAIT", "best_score": 68, "qualified_count": 2, "strong_adx_share": 0.4},
            ]
        )
        self.assertEqual(int(s["scans"]), 2)
        self.assertAlmostEqual(s["ready_rate"], 50.0, places=2)
        self.assertGreater(s["avg_best_score"], 70.0)


if __name__ == "__main__":
    unittest.main()
