import unittest
from types import SimpleNamespace
from unittest.mock import patch

import pandas as pd

from core.spot_execution_pipeline import build_spot_execution_pipeline, prepare_closed_frame


class SpotExecutionPipelineTests(unittest.TestCase):
    def _sample_df(self, rows: int = 120) -> pd.DataFrame:
        timestamps = pd.date_range("2026-01-01", periods=rows, freq="h", tz="UTC")
        base = pd.Series(range(rows), dtype=float) + 100.0
        return pd.DataFrame(
            {
                "timestamp": timestamps,
                "open": base,
                "high": base + 1.5,
                "low": base - 1.5,
                "close": base + 0.5,
                "volume": pd.Series(range(rows), dtype=float) + 10.0,
            }
        )

    def test_prepare_closed_frame_drops_last_candle(self) -> None:
        df = self._sample_df(70)
        out = prepare_closed_frame(df, min_rows=55)
        self.assertIsNotNone(out)
        self.assertEqual(len(out), 69)
        self.assertEqual(out["timestamp"].iloc[-1], df["timestamp"].iloc[-2])

    def test_build_spot_execution_pipeline_returns_shared_decision_bundle(self) -> None:
        df_eval = self._sample_df(90)
        htf_df = self._sample_df(200)
        fetch_calls: list[tuple[str, str, int]] = []

        def fetch_ohlcv(symbol: str, timeframe: str, limit: int):
            fetch_calls.append((symbol, timeframe, limit))
            return htf_df

        def analyse_fn(_df: pd.DataFrame):
            return SimpleNamespace(
                signal="Upside",
                bias=82.0,
                adx=22.0,
                supertrend="Bullish",
                ichimoku="Bullish",
                stochrsi_k=0.55,
                bollinger="→ Neutral",
                vwap="🟢 Above",
                psar="Bullish",
                williams="Neutral",
                cci="Neutral",
            )

        def predictor(_df: pd.DataFrame):
            return 0.72, "Upside", {"directional_agreement": 0.82, "consensus_agreement": 0.74}

        def conviction_fn(_signal_dir: str, _ai_dir: str, _confidence: float, _agree: float = 0.0):
            return "HIGH", None

        spot_snapshot = SimpleNamespace(
            direction="UPSIDE",
            timeframe_alignment=88.0,
            structure_quality=84.0,
            trend_quality=81.0,
            regime_quality=77.0,
            location_quality=79.0,
            timeframe_conflict=False,
            degraded_data=False,
            range_regime=False,
        )
        confidence_snapshot = SimpleNamespace(score=76.0, label="MEDIUM")
        ai_spot_snapshot = SimpleNamespace(
            direction="UPSIDE",
            score=74.0,
            conviction_quality=71.0,
            timeframe_alignment=78.0,
            consensus_quality=69.0,
            support_votes=2,
            timeframe_conflict=False,
            degraded_data=False,
            one_day=SimpleNamespace(
                probability_up=0.78,
                directional_agreement=0.80,
                consensus_agreement=0.72,
                status="",
            ),
            four_hour=SimpleNamespace(
                probability_up=0.70,
                directional_agreement=0.76,
                consensus_agreement=0.70,
                status="",
            ),
        )
        ai_confidence_snapshot = SimpleNamespace(score=68.0, label="MEDIUM")
        execution_confidence_snapshot = SimpleNamespace(score=73.0, label="MEDIUM")
        execution_snapshot = SimpleNamespace(
            structure_quality=82.0,
            trend_quality=79.0,
            regime_quality=74.0,
            location_quality=76.0,
        )
        trend_led_snapshot = SimpleNamespace(score=81.0, state="READY", reason_code="CONFIRMED")
        ai_led_snapshot = SimpleNamespace(score=74.0, state="WATCH", reason_code="EARLY_SUPPORT")

        with patch(
            "core.spot_execution_pipeline.build_spot_direction_snapshot",
            return_value=spot_snapshot,
        ), patch(
            "core.spot_execution_pipeline.build_confidence_snapshot",
            return_value=confidence_snapshot,
        ), patch(
            "core.spot_execution_pipeline.build_ai_spot_bias_snapshot",
            return_value=ai_spot_snapshot,
        ), patch(
            "core.spot_execution_pipeline.build_ai_confidence_snapshot",
            return_value=ai_confidence_snapshot,
        ), patch(
            "core.spot_execution_pipeline.build_execution_confidence_snapshot",
            return_value=execution_confidence_snapshot,
        ), patch(
            "core.spot_execution_pipeline.selected_timeframe_execution_snapshot",
            return_value=execution_snapshot,
        ), patch(
            "core.spot_execution_pipeline.selected_timeframe_rr_ratio",
            return_value=1.9,
        ), patch(
            "core.spot_execution_pipeline.trend_led_confirmation_snapshot",
            return_value=trend_led_snapshot,
        ), patch(
            "core.spot_execution_pipeline.ai_led_confirmation_snapshot",
            return_value=ai_led_snapshot,
        ), patch(
            "core.spot_execution_pipeline.spot_action_decision_with_reason",
            return_value=("PROBE", "TACTICAL_WATCH"),
        ):
            out = build_spot_execution_pipeline(
                symbol="BTC/USDT",
                actual_symbol="BTC/USDT",
                source_provider="exchange",
                timeframe="1h",
                df_eval=df_eval,
                fetch_ohlcv=fetch_ohlcv,
                analyse_fn=analyse_fn,
                predictor=predictor,
                conviction_fn=conviction_fn,
            )

        self.assertEqual(fetch_calls[0], ("BTC/USDT", "4h", 260))
        self.assertEqual(fetch_calls[1], ("BTC/USDT", "1d", 260))
        self.assertEqual(out.anchor_plan.pair_label, "1D + 4H")
        self.assertEqual(out.signal_direction, "UPSIDE")
        self.assertEqual(out.signal_direction_legacy, "LONG")
        self.assertEqual(out.ai_spot_direction, "UPSIDE")
        self.assertEqual(out.action_raw, "PROBE")
        self.assertEqual(out.action_reason_code, "TACTICAL_WATCH")
        self.assertAlmostEqual(float(out.setup_rr_ratio), 1.9, places=4)
        self.assertEqual(out.execution_conviction_label, "HIGH")

    def test_build_spot_execution_pipeline_uses_intraday_anchor_plan_for_15m(self) -> None:
        df_eval = self._sample_df(90)
        htf_df = self._sample_df(200)
        fetch_calls: list[tuple[str, str, int]] = []

        def fetch_ohlcv(symbol: str, timeframe: str, limit: int):
            fetch_calls.append((symbol, timeframe, limit))
            return htf_df

        def analyse_fn(_df: pd.DataFrame):
            return SimpleNamespace(
                signal="Upside",
                bias=82.0,
                adx=22.0,
                supertrend="Bullish",
                ichimoku="Bullish",
                stochrsi_k=0.55,
                bollinger="→ Neutral",
                vwap="🟢 Above",
                psar="Bullish",
                williams="Neutral",
                cci="Neutral",
            )

        def predictor(_df: pd.DataFrame):
            return 0.72, "Upside", {"directional_agreement": 0.82, "consensus_agreement": 0.74}

        def conviction_fn(_signal_dir: str, _ai_dir: str, _confidence: float, _agree: float = 0.0):
            return "HIGH", None

        with patch(
            "core.spot_execution_pipeline.build_spot_direction_snapshot",
            return_value=SimpleNamespace(
                direction="UPSIDE",
                timeframe_alignment=88.0,
                structure_quality=84.0,
                trend_quality=81.0,
                regime_quality=77.0,
                location_quality=79.0,
                timeframe_conflict=False,
                degraded_data=False,
                range_regime=False,
            ),
        ), patch(
            "core.spot_execution_pipeline.build_confidence_snapshot",
            return_value=SimpleNamespace(score=76.0, label="MEDIUM"),
        ), patch(
            "core.spot_execution_pipeline.build_ai_spot_bias_snapshot",
            return_value=SimpleNamespace(
                direction="UPSIDE",
                score=74.0,
                conviction_quality=71.0,
                timeframe_alignment=78.0,
                consensus_quality=69.0,
                support_votes=2,
                timeframe_conflict=False,
                degraded_data=False,
                lead_snapshot=SimpleNamespace(probability_up=0.75, directional_agreement=0.8, consensus_agreement=0.7, status=""),
                confirm_snapshot=SimpleNamespace(probability_up=0.70, directional_agreement=0.76, consensus_agreement=0.70, status=""),
            ),
        ), patch(
            "core.spot_execution_pipeline.build_ai_confidence_snapshot",
            return_value=SimpleNamespace(score=68.0, label="MEDIUM"),
        ), patch(
            "core.spot_execution_pipeline.build_execution_confidence_snapshot",
            return_value=SimpleNamespace(score=73.0, label="MEDIUM"),
        ), patch(
            "core.spot_execution_pipeline.selected_timeframe_execution_snapshot",
            return_value=SimpleNamespace(
                structure_quality=82.0,
                trend_quality=79.0,
                regime_quality=74.0,
                location_quality=76.0,
            ),
        ), patch(
            "core.spot_execution_pipeline.selected_timeframe_rr_ratio",
            return_value=1.9,
        ), patch(
            "core.spot_execution_pipeline.trend_led_confirmation_snapshot",
            return_value=SimpleNamespace(score=81.0, state="READY", reason_code="CONFIRMED"),
        ), patch(
            "core.spot_execution_pipeline.ai_led_confirmation_snapshot",
            return_value=SimpleNamespace(score=74.0, state="WATCH", reason_code="EARLY_SUPPORT"),
        ), patch(
            "core.spot_execution_pipeline.spot_action_decision_with_reason",
            return_value=("PROBE", "TACTICAL_WATCH"),
        ):
            out = build_spot_execution_pipeline(
                symbol="BTC/USDT",
                actual_symbol="BTC/USDT",
                source_provider="exchange",
                timeframe="15m",
                df_eval=df_eval,
                fetch_ohlcv=fetch_ohlcv,
                analyse_fn=analyse_fn,
                predictor=predictor,
                conviction_fn=conviction_fn,
            )

        self.assertEqual(fetch_calls[0], ("BTC/USDT", "1h", 260))
        self.assertEqual(fetch_calls[1], ("BTC/USDT", "4h", 260))
        self.assertEqual(out.anchor_plan.pair_label, "4H + 1H")

    def test_build_spot_execution_pipeline_falls_back_from_weekly_to_daily_ladder(self) -> None:
        df_eval = self._sample_df(90)
        htf_df = self._sample_df(200)
        fetch_calls: list[tuple[str, str, int]] = []

        def fetch_ohlcv(symbol: str, timeframe: str, limit: int):
            fetch_calls.append((symbol, timeframe, limit))
            if timeframe == "1w":
                return None
            return htf_df

        def analyse_fn(_df: pd.DataFrame):
            return SimpleNamespace(
                signal="Upside",
                bias=82.0,
                adx=22.0,
                supertrend="Bullish",
                ichimoku="Bullish",
                stochrsi_k=0.55,
                bollinger="→ Neutral",
                vwap="🟢 Above",
                psar="Bullish",
                williams="Neutral",
                cci="Neutral",
            )

        def predictor(_df: pd.DataFrame):
            return 0.72, "Upside", {"directional_agreement": 0.82, "consensus_agreement": 0.74}

        def conviction_fn(_signal_dir: str, _ai_dir: str, _confidence: float, _agree: float = 0.0):
            return "HIGH", None

        with patch(
            "core.spot_execution_pipeline.build_spot_direction_snapshot",
            return_value=SimpleNamespace(
                direction="UPSIDE",
                timeframe_alignment=88.0,
                structure_quality=84.0,
                trend_quality=81.0,
                regime_quality=77.0,
                location_quality=79.0,
                timeframe_conflict=False,
                degraded_data=False,
                range_regime=False,
            ),
        ), patch(
            "core.spot_execution_pipeline.build_confidence_snapshot",
            return_value=SimpleNamespace(score=76.0, label="MEDIUM"),
        ), patch(
            "core.spot_execution_pipeline.build_ai_spot_bias_snapshot",
            return_value=SimpleNamespace(
                direction="UPSIDE",
                score=74.0,
                conviction_quality=71.0,
                timeframe_alignment=78.0,
                consensus_quality=69.0,
                support_votes=2,
                timeframe_conflict=False,
                degraded_data=False,
                lead_snapshot=SimpleNamespace(probability_up=0.75, directional_agreement=0.8, consensus_agreement=0.7, status=""),
                confirm_snapshot=SimpleNamespace(probability_up=0.70, directional_agreement=0.76, consensus_agreement=0.70, status=""),
            ),
        ), patch(
            "core.spot_execution_pipeline.build_ai_confidence_snapshot",
            return_value=SimpleNamespace(score=68.0, label="MEDIUM"),
        ), patch(
            "core.spot_execution_pipeline.build_execution_confidence_snapshot",
            return_value=SimpleNamespace(score=73.0, label="MEDIUM"),
        ), patch(
            "core.spot_execution_pipeline.selected_timeframe_execution_snapshot",
            return_value=SimpleNamespace(
                structure_quality=82.0,
                trend_quality=79.0,
                regime_quality=74.0,
                location_quality=76.0,
            ),
        ), patch(
            "core.spot_execution_pipeline.selected_timeframe_rr_ratio",
            return_value=1.9,
        ), patch(
            "core.spot_execution_pipeline.trend_led_confirmation_snapshot",
            return_value=SimpleNamespace(score=81.0, state="READY", reason_code="CONFIRMED"),
        ), patch(
            "core.spot_execution_pipeline.ai_led_confirmation_snapshot",
            return_value=SimpleNamespace(score=74.0, state="WATCH", reason_code="EARLY_SUPPORT"),
        ), patch(
            "core.spot_execution_pipeline.spot_action_decision_with_reason",
            return_value=("PROBE", "TACTICAL_WATCH"),
        ):
            out = build_spot_execution_pipeline(
                symbol="BTC/USDT",
                actual_symbol="BTC/USDT",
                source_provider="exchange",
                timeframe="4h",
                df_eval=df_eval,
                fetch_ohlcv=fetch_ohlcv,
                analyse_fn=analyse_fn,
                predictor=predictor,
                conviction_fn=conviction_fn,
            )

        self.assertEqual(fetch_calls[0], ("BTC/USDT", "1d", 260))
        self.assertEqual(fetch_calls[1], ("BTC/USDT", "1w", 260))
        self.assertEqual(fetch_calls[2], ("BTC/USDT", "4h", 260))
        self.assertEqual(out.anchor_plan.pair_label, "1D + 4H")


if __name__ == "__main__":
    unittest.main()
