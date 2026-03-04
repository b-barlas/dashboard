from __future__ import annotations

import unittest
from dataclasses import dataclass
from unittest.mock import patch

try:
    import numpy as np
    import pandas as pd
    from core.backtest import (
        build_setup_outcome_study,
        _normalize_direction_signal,
        run_backtest,
        run_setup_confirm_backtest,
        summarize_setup_outcome_by_class,
        summarize_setup_outcome_study,
        summarize_setup_confirm_performance,
    )
    from core.ml import ml_ensemble_predict, ml_predict_direction
    from core.risk import annualization_factor, calculate_risk_metrics
    from core.signals import analyse as analyse_core
    DEPS_OK = True
except Exception:
    DEPS_OK = False


@dataclass
class DummyAnalysis:
    signal: str
    bias: float
    adx: float = 30.0


@unittest.skipUnless(DEPS_OK, "Missing dependencies for core engine tests")
class CoreEngineTests(unittest.TestCase):
    def _sample_df(self, n: int = 180) -> pd.DataFrame:
        ts = pd.date_range("2025-01-01", periods=n, freq="h")
        base = np.linspace(100.0, 120.0, n)
        wiggle = np.sin(np.linspace(0, 10, n)) * 0.6
        close = base + wiggle
        open_ = close - 0.15
        high = close + 0.35
        low = close - 0.35
        vol = np.full(n, 1000.0)
        return pd.DataFrame(
            {
                "timestamp": ts,
                "open": open_,
                "high": high,
                "low": low,
                "close": close,
                "volume": vol,
            }
        )

    def test_annualization_factor(self) -> None:
        self.assertEqual(annualization_factor("1h"), 24 * 365)
        self.assertEqual(annualization_factor("4h"), 6 * 365)
        self.assertEqual(annualization_factor("1d"), 365)
        self.assertEqual(annualization_factor("unknown"), 365)

    def test_risk_metrics_contains_expected_keys(self) -> None:
        df = self._sample_df()
        metrics = calculate_risk_metrics(df, timeframe="1h")
        self.assertIn("sharpe", metrics)
        self.assertIn("max_drawdown", metrics)
        self.assertIn("ann_volatility", metrics)
        self.assertIsInstance(metrics["sharpe"], float)

    def test_backtest_long_entries_work(self) -> None:
        df = self._sample_df()

        def analyzer(_slice: pd.DataFrame) -> DummyAnalysis:
            return DummyAnalysis(signal="BUY", bias=95.0)

        out, _summary = run_backtest(df, analyzer=analyzer, threshold=70, exit_after=5)
        self.assertGreater(len(out), 0)
        self.assertTrue((out["Signal"] == "LONG").all())

    def test_backtest_short_entries_use_strength_threshold(self) -> None:
        df = self._sample_df()

        def analyzer(_slice: pd.DataFrame) -> DummyAnalysis:
            return DummyAnalysis(signal="SELL", bias=10.0)

        out, _summary = run_backtest(df, analyzer=analyzer, threshold=70, exit_after=5)
        self.assertGreater(len(out), 0)
        self.assertTrue((out["Signal"] == "SHORT").all())

    def test_backtest_short_is_blocked_when_strength_is_low(self) -> None:
        df = self._sample_df()

        def analyzer(_slice: pd.DataFrame) -> DummyAnalysis:
            return DummyAnalysis(signal="SELL", bias=40.0)

        out, _summary = run_backtest(df, analyzer=analyzer, threshold=70, exit_after=5)
        self.assertEqual(len(out), 0)

    def test_backtest_accepts_upside_downside_labels(self) -> None:
        df = self._sample_df()

        def analyzer_up(_slice: pd.DataFrame) -> DummyAnalysis:
            return DummyAnalysis(signal="Upside", bias=95.0)

        out_up, _summary_up = run_backtest(df, analyzer=analyzer_up, threshold=70, exit_after=5)
        self.assertGreater(len(out_up), 0)
        self.assertTrue((out_up["Signal"] == "LONG").all())

        def analyzer_down(_slice: pd.DataFrame) -> DummyAnalysis:
            return DummyAnalysis(signal="Downside", bias=5.0)

        out_down, _summary_down = run_backtest(df, analyzer=analyzer_down, threshold=70, exit_after=5)
        self.assertGreater(len(out_down), 0)
        self.assertTrue((out_down["Signal"] == "SHORT").all())

    def test_signal_normalizer_handles_current_and_legacy_labels(self) -> None:
        self.assertEqual(_normalize_direction_signal("BUY"), "LONG")
        self.assertEqual(_normalize_direction_signal("Upside"), "LONG")
        self.assertEqual(_normalize_direction_signal("SELL"), "SHORT")
        self.assertEqual(_normalize_direction_signal("Downside"), "SHORT")
        self.assertEqual(_normalize_direction_signal("WAIT"), "WAIT")

    def test_setup_confirm_backtest_generates_class_entries(self) -> None:
        df = self._sample_df()

        def analyzer(_slice: pd.DataFrame) -> DummyAnalysis:
            return DummyAnalysis(signal="Upside", bias=90.0, adx=30.0)

        def ml_predictor(_slice: pd.DataFrame):
            return 0.80, "Upside", {"directional_agreement": 0.80, "consensus_agreement": 0.80}

        def conviction_fn(_signal_dir: str, _ai_dir: str, _strength: float, _agree: float = 0.0):
            return "HIGH", None

        def signal_plain_fn(signal: str) -> str:
            return signal

        def direction_key_fn(direction: str) -> str:
            s = str(direction or "").strip().upper()
            if s in {"UPSIDE", "LONG", "BUY"}:
                return "UPSIDE"
            if s in {"DOWNSIDE", "SHORT", "SELL"}:
                return "DOWNSIDE"
            return "NEUTRAL"

        out = run_setup_confirm_backtest(
            df=df,
            analyzer=analyzer,
            ml_predictor=ml_predictor,
            conviction_fn=conviction_fn,
            signal_plain_fn=signal_plain_fn,
            direction_key_fn=direction_key_fn,
            exit_after=5,
        )
        self.assertGreater(len(out), 0)
        self.assertIn("Setup Confirm", out.columns)
        self.assertTrue((out["Setup Confirm"] == "TREND+AI").all())
        self.assertTrue((out["Direction"] == "Upside").all())

    def test_setup_confirm_summary_has_expected_columns(self) -> None:
        df = pd.DataFrame(
            {
                "Setup Confirm": ["TREND+AI", "TREND+AI", "AI-led"],
                "PnL (%)": [2.0, -1.0, 1.5],
            }
        )
        summary = summarize_setup_confirm_performance(df)
        self.assertGreater(len(summary), 0)
        self.assertIn("Setup Confirm", summary.columns)
        self.assertIn("Trades", summary.columns)
        self.assertIn("ProfitFactor", summary.columns)

    def test_setup_outcome_study_generates_forward_path(self) -> None:
        df = self._sample_df(240)

        def analyzer(_slice: pd.DataFrame) -> DummyAnalysis:
            return DummyAnalysis(signal="Upside", bias=88.0, adx=30.0)

        def ml_predictor(_slice: pd.DataFrame):
            return 0.82, "Upside", {"directional_agreement": 0.80, "consensus_agreement": 0.80}

        def conviction_fn(_signal_dir: str, _ai_dir: str, _strength: float, _agree: float = 0.0):
            return "HIGH", None

        def signal_plain_fn(signal: str) -> str:
            return signal

        def direction_key_fn(direction: str) -> str:
            s = str(direction or "").strip().upper()
            if s in {"UPSIDE", "LONG", "BUY"}:
                return "UPSIDE"
            if s in {"DOWNSIDE", "SHORT", "SELL"}:
                return "DOWNSIDE"
            return "NEUTRAL"

        out = build_setup_outcome_study(
            df=df,
            analyzer=analyzer,
            ml_predictor=ml_predictor,
            conviction_fn=conviction_fn,
            signal_plain_fn=signal_plain_fn,
            direction_key_fn=direction_key_fn,
            setup_filter="TREND+AI",
            forward_bars=10,
        )
        self.assertGreater(len(out), 0)
        self.assertIn("Event Price", out.columns)
        self.assertIn("Price +1", out.columns)
        self.assertIn("Price +10", out.columns)
        self.assertIn("Return @+10 (%)", out.columns)
        self.assertTrue((out["Setup Confirm"] == "TREND+AI").all())

    def test_setup_outcome_summaries_have_expected_fields(self) -> None:
        df = pd.DataFrame(
            {
                "Setup Confirm": ["TREND+AI", "TREND+AI", "AI-led"],
                "Return @+10 (%)": [1.2, -0.4, 0.8],
                "Favorable Excursion (%)": [2.0, 1.1, 1.5],
                "Adverse Excursion (%)": [0.6, 1.3, 0.9],
            }
        )
        top = summarize_setup_outcome_study(df, 10)
        self.assertIn("occurrences", top)
        self.assertIn("favorable_rate", top)
        self.assertIn("median_dir_return", top)

        by_class = summarize_setup_outcome_by_class(df, 10)
        self.assertGreater(len(by_class), 0)
        self.assertIn("Setup Confirm", by_class.columns)
        self.assertIn("Occurrences", by_class.columns)
        self.assertIn("FavorableRate", by_class.columns)

    def test_backtest_positions_do_not_overlap(self) -> None:
        df = self._sample_df()

        def analyzer(_slice: pd.DataFrame) -> DummyAnalysis:
            return DummyAnalysis(signal="BUY", bias=95.0)

        out, _summary = run_backtest(df, analyzer=analyzer, threshold=70, exit_after=5)
        self.assertGreater(len(out), 1)
        step_hours = out["Date"].diff().dropna().dt.total_seconds() / 3600
        self.assertTrue((step_hours >= 5).all())
        self.assertIn("Regime", out.columns)
        self.assertIn("Regime Score", out.columns)
        self.assertTrue(out["Regime"].isin({"TREND", "RANGE", "MIXED"}).all())

    def test_signals_engine_returns_expected_fields(self) -> None:
        df = self._sample_df()
        result = analyse_core(df)
        self.assertTrue(hasattr(result, "signal"))
        self.assertTrue(hasattr(result, "bias"))
        self.assertTrue(hasattr(result, "leverage"))

    def test_signals_engine_does_not_crash_when_adx_fails(self) -> None:
        df = self._sample_df()
        with patch("core.signals.ta.trend.adx", side_effect=RuntimeError("adx failed")):
            result = analyse_core(df)
        self.assertTrue(hasattr(result, "signal"))

    def test_ml_engine_shapes(self) -> None:
        df = self._sample_df(220)
        prob, direction = ml_predict_direction(df)
        self.assertIsInstance(prob, float)
        self.assertIn(direction, {"LONG", "SHORT", "NEUTRAL"})
        p2, d2, details = ml_ensemble_predict(df)
        self.assertIsInstance(p2, float)
        self.assertIn(d2, {"LONG", "SHORT", "NEUTRAL"})
        self.assertIsInstance(details, dict)


if __name__ == "__main__":
    unittest.main()
