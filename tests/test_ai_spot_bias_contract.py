import unittest

import numpy as np
import pandas as pd

from core.ai_spot_bias import (
    AISpotBiasSnapshot,
    TimeframeAIBiasSnapshot,
    ai_spot_bias_display_votes,
    analyze_timeframe_ai_bias,
    build_ai_spot_bias_snapshot,
)


def _frame(
    *,
    rows: int = 260,
    start: float = 100.0,
    end: float = 130.0,
) -> pd.DataFrame:
    close = np.linspace(start, end, rows)
    open_ = close - np.sign(end - start or 1.0) * 0.15
    high = close + 0.65
    low = close - 0.65
    volume = np.linspace(100.0, 200.0, rows)
    return pd.DataFrame(
        {
            "timestamp": pd.date_range("2026-01-01", periods=rows, freq="h"),
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
        }
    )


def _predictor(*outputs):
    queue = list(outputs)

    def _run(_df):
        if not queue:
            raise AssertionError("predictor queue exhausted")
        return queue.pop(0)

    return _run


def _htf_outputs(
    four_hour_output,
    one_day_output,
):
    return (
        *(four_hour_output for _ in range(4)),
        *(one_day_output for _ in range(3)),
    )


class AISpotBiasContractTests(unittest.TestCase):
    def test_analyze_timeframe_ai_bias_reads_directional_upside(self) -> None:
        snap = analyze_timeframe_ai_bias(
            _frame(),
            timeframe="4h",
            predictor=_predictor(
                (
                    0.72,
                    "LONG",
                    {
                        "directional_agreement": 1.0,
                        "consensus_agreement": 1.0,
                    },
                )
            ),
        )
        self.assertIsInstance(snap, TimeframeAIBiasSnapshot)
        self.assertEqual(snap.direction, "UPSIDE")
        self.assertGreater(snap.score, 60.0)
        self.assertFalse(snap.degraded)

    def test_build_ai_spot_bias_snapshot_prefers_aligned_higher_timeframes(self) -> None:
        out = build_ai_spot_bias_snapshot(
            df_4h=_frame(start=105.0, end=130.0),
            df_1d=_frame(start=95.0, end=150.0),
            predictor=_predictor(
                *_htf_outputs(
                    (
                        0.68,
                        "LONG",
                        {
                            "directional_agreement": 2.0 / 3.0,
                            "consensus_agreement": 2.0 / 3.0,
                        },
                    ),
                    (
                        0.74,
                        "LONG",
                        {
                            "directional_agreement": 1.0,
                            "consensus_agreement": 1.0,
                        },
                    ),
                )
            ),
        )
        self.assertIsInstance(out, AISpotBiasSnapshot)
        self.assertEqual(out.direction, "UPSIDE")
        self.assertEqual(out.timeframe_alignment, 100.0)
        self.assertFalse(out.timeframe_conflict)
        self.assertIn("1D AI bias", out.note)

    def test_build_ai_spot_bias_snapshot_returns_neutral_on_conflict(self) -> None:
        out = build_ai_spot_bias_snapshot(
            df_4h=_frame(start=140.0, end=100.0),
            df_1d=_frame(start=95.0, end=150.0),
            predictor=_predictor(
                *_htf_outputs(
                    (
                        0.24,
                        "SHORT",
                        {
                            "directional_agreement": 1.0,
                            "consensus_agreement": 1.0,
                        },
                    ),
                    (
                        0.76,
                        "LONG",
                        {
                            "directional_agreement": 1.0,
                            "consensus_agreement": 1.0,
                        },
                    ),
                )
            ),
        )
        self.assertEqual(out.direction, "NEUTRAL")
        self.assertTrue(out.timeframe_conflict)
        self.assertEqual(out.timeframe_alignment, 0.0)

    def test_build_ai_spot_bias_snapshot_returns_neutral_when_daily_bias_is_neutral(self) -> None:
        out = build_ai_spot_bias_snapshot(
            df_4h=_frame(start=105.0, end=130.0),
            df_1d=_frame(start=100.0, end=101.0),
            predictor=_predictor(
                *_htf_outputs(
                    (
                        0.68,
                        "LONG",
                        {
                            "directional_agreement": 2.0 / 3.0,
                            "consensus_agreement": 2.0 / 3.0,
                        },
                    ),
                    (
                        0.53,
                        "NEUTRAL",
                        {
                            "directional_agreement": 0.0,
                            "consensus_agreement": 2.0 / 3.0,
                        },
                    ),
                )
            ),
        )
        self.assertEqual(out.direction, "NEUTRAL")
        self.assertFalse(out.degraded_data)
        self.assertIn("1D AI bias", out.note)

    def test_build_ai_spot_bias_snapshot_marks_missing_or_fallback_context_as_degraded(self) -> None:
        out = build_ai_spot_bias_snapshot(
            df_4h=_frame(start=105.0, end=130.0),
            df_1d=_frame(start=95.0, end=150.0),
            predictor=_predictor(
                *_htf_outputs(
                    (
                        0.50,
                        "NEUTRAL",
                        {
                            "directional_agreement": 0.0,
                            "consensus_agreement": 0.0,
                            "status": "insufficient_features",
                        },
                    ),
                    (
                        0.74,
                        "LONG",
                        {
                            "directional_agreement": 1.0,
                            "consensus_agreement": 1.0,
                        },
                    ),
                )
            ),
        )
        self.assertEqual(out.direction, "NEUTRAL")
        self.assertTrue(out.degraded_data)
        self.assertEqual(out.note, "Higher-timeframe AI context is incomplete.")

    def test_analyze_timeframe_ai_bias_can_promote_soft_direction_from_probability(self) -> None:
        snap = analyze_timeframe_ai_bias(
            _frame(),
            timeframe="1d",
            predictor=_predictor(
                (
                    0.56,
                    "NEUTRAL",
                    {
                        "directional_agreement": 2.0 / 3.0,
                        "consensus_agreement": 2.0 / 3.0,
                    },
                )
            ),
        )
        self.assertEqual(snap.direction, "UPSIDE")
        self.assertGreaterEqual(snap.score, 52.0)

    def test_analyze_timeframe_ai_bias_uses_trace_weighted_consensus_quality(self) -> None:
        snap = analyze_timeframe_ai_bias(
            _frame(),
            timeframe="4h",
            predictor=_predictor(
                (
                    0.72,
                    "LONG",
                    {
                        "directional_agreement": 1.0,
                        "consensus_agreement": 0.0,
                    },
                ),
                (
                    0.72,
                    "LONG",
                    {
                        "directional_agreement": 1.0,
                        "consensus_agreement": 1.0,
                    },
                ),
                (
                    0.72,
                    "LONG",
                    {
                        "directional_agreement": 1.0,
                        "consensus_agreement": 1.0,
                    },
                ),
                (
                    0.72,
                    "LONG",
                    {
                        "directional_agreement": 1.0,
                        "consensus_agreement": 1.0,
                    },
                ),
            ),
        )
        self.assertEqual(snap.direction, "UPSIDE")
        self.assertAlmostEqual(snap.consensus_quality, 68.52, places=2)

    def test_ai_spot_bias_display_votes_reflects_how_many_models_support_final_htf_bias(self) -> None:
        aligned = build_ai_spot_bias_snapshot(
            df_4h=_frame(start=105.0, end=130.0),
            df_1d=_frame(start=95.0, end=150.0),
            predictor=_predictor(
                *_htf_outputs(
                    (
                        0.70,
                        "LONG",
                        {
                            "directional_agreement": 1.0,
                            "consensus_agreement": 1.0,
                            "model_votes": ["LONG", "LONG", "LONG"],
                        },
                    ),
                    (
                        0.76,
                        "LONG",
                        {
                            "directional_agreement": 1.0,
                            "consensus_agreement": 1.0,
                            "model_votes": ["LONG", "LONG", "LONG"],
                        },
                    ),
                )
            ),
        )
        self.assertEqual(ai_spot_bias_display_votes(aligned), 3)

        partial = build_ai_spot_bias_snapshot(
            df_4h=_frame(start=105.0, end=130.0),
            df_1d=_frame(start=95.0, end=150.0),
            predictor=_predictor(
                *_htf_outputs(
                    (
                        0.66,
                        "LONG",
                        {
                            "directional_agreement": 2.0 / 3.0,
                            "consensus_agreement": 2.0 / 3.0,
                            "model_votes": ["LONG", "LONG", "SHORT"],
                        },
                    ),
                    (
                        0.72,
                        "LONG",
                        {
                            "directional_agreement": 2.0 / 3.0,
                            "consensus_agreement": 2.0 / 3.0,
                            "model_votes": ["LONG", "LONG", "NEUTRAL"],
                        },
                    ),
                )
            ),
        )
        self.assertEqual(partial.direction, "UPSIDE")
        self.assertEqual(ai_spot_bias_display_votes(partial), 2)

    def test_build_ai_spot_bias_snapshot_supports_custom_anchor_pair(self) -> None:
        out = build_ai_spot_bias_snapshot(
            df_4h=None,
            df_1d=None,
            confirm_df=_frame(start=105.0, end=130.0),
            lead_df=_frame(start=95.0, end=150.0),
            confirm_timeframe="1h",
            lead_timeframe="4h",
            predictor=_predictor(
                *_htf_outputs(
                    (
                        0.68,
                        "LONG",
                        {
                            "directional_agreement": 2.0 / 3.0,
                            "consensus_agreement": 2.0 / 3.0,
                        },
                    ),
                    (
                        0.74,
                        "LONG",
                        {
                            "directional_agreement": 1.0,
                            "consensus_agreement": 1.0,
                        },
                    ),
                )
            ),
        )
        self.assertEqual(out.direction, "UPSIDE")
        self.assertEqual(out.lead_timeframe, "4h")
        self.assertEqual(out.confirm_timeframe, "1h")
        self.assertEqual(out.anchor_pair_label, "4H + 1H")
        self.assertIn("4H", out.note)

        conflicted = build_ai_spot_bias_snapshot(
            df_4h=_frame(start=140.0, end=100.0),
            df_1d=_frame(start=95.0, end=150.0),
            predictor=_predictor(
                *_htf_outputs(
                    (
                        0.24,
                        "SHORT",
                        {
                            "directional_agreement": 1.0,
                            "consensus_agreement": 1.0,
                            "model_votes": ["SHORT", "SHORT", "SHORT"],
                        },
                    ),
                    (
                        0.76,
                        "LONG",
                        {
                            "directional_agreement": 1.0,
                            "consensus_agreement": 1.0,
                            "model_votes": ["LONG", "LONG", "LONG"],
                        },
                    ),
                )
            ),
        )
        self.assertEqual(conflicted.direction, "NEUTRAL")
        self.assertEqual(ai_spot_bias_display_votes(conflicted), 3)

        neutral_supported = build_ai_spot_bias_snapshot(
            df_4h=_frame(start=105.0, end=130.0),
            df_1d=_frame(start=100.0, end=101.0),
            predictor=_predictor(
                *_htf_outputs(
                    (
                        0.67,
                        "LONG",
                        {
                            "directional_agreement": 2.0 / 3.0,
                            "consensus_agreement": 2.0 / 3.0,
                            "model_votes": ["LONG", "LONG", "LONG"],
                        },
                    ),
                    (
                        0.53,
                        "NEUTRAL",
                        {
                            "directional_agreement": 0.0,
                            "consensus_agreement": 2.0 / 3.0,
                            "model_votes": ["NEUTRAL", "NEUTRAL", "LONG"],
                        },
                    ),
                )
            ),
        )
        self.assertEqual(neutral_supported.direction, "NEUTRAL")
        self.assertEqual(ai_spot_bias_display_votes(neutral_supported), 2)

    def test_ai_spot_bias_display_votes_use_trace_level_model_support_not_just_latest_votes(self) -> None:
        out = build_ai_spot_bias_snapshot(
            df_4h=_frame(start=105.0, end=130.0),
            df_1d=_frame(start=95.0, end=150.0),
            predictor=_predictor(
                (
                    0.67,
                    "LONG",
                    {
                        "directional_agreement": 2.0 / 3.0,
                        "consensus_agreement": 2.0 / 3.0,
                        "model_votes": ["LONG", "LONG", "SHORT"],
                    },
                ),
                (
                    0.68,
                    "LONG",
                    {
                        "directional_agreement": 1.0,
                        "consensus_agreement": 1.0,
                        "model_votes": ["LONG", "LONG", "LONG"],
                    },
                ),
                (
                    0.69,
                    "LONG",
                    {
                        "directional_agreement": 1.0,
                        "consensus_agreement": 1.0,
                        "model_votes": ["LONG", "LONG", "LONG"],
                    },
                ),
                (
                    0.70,
                    "LONG",
                    {
                        "directional_agreement": 1.0,
                        "consensus_agreement": 1.0,
                        "model_votes": ["LONG", "LONG", "LONG"],
                    },
                ),
                (
                    0.74,
                    "LONG",
                    {
                        "directional_agreement": 1.0,
                        "consensus_agreement": 1.0,
                        "model_votes": ["LONG", "LONG", "LONG"],
                    },
                ),
                (
                    0.75,
                    "LONG",
                    {
                        "directional_agreement": 1.0,
                        "consensus_agreement": 1.0,
                        "model_votes": ["LONG", "LONG", "LONG"],
                    },
                ),
                (
                    0.76,
                    "LONG",
                    {
                        "directional_agreement": 1.0,
                        "consensus_agreement": 1.0,
                        "model_votes": ["LONG", "LONG", "LONG"],
                    },
                ),
            ),
        )
        self.assertEqual(out.direction, "UPSIDE")
        self.assertEqual(out.four_hour.model_votes, ("UPSIDE", "UPSIDE", "UPSIDE"))
        self.assertEqual(ai_spot_bias_display_votes(out), 3)


if __name__ == "__main__":
    unittest.main()
