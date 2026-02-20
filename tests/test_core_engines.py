from __future__ import annotations

import unittest
from dataclasses import dataclass
from unittest.mock import patch

try:
    import numpy as np
    import pandas as pd
    from core.backtest import run_backtest
    from core.ml import ml_ensemble_predict, ml_predict_direction
    from core.risk import annualization_factor, calculate_risk_metrics
    from core.signals import analyse as analyse_core
    DEPS_OK = True
except Exception:
    DEPS_OK = False


@dataclass
class DummyAnalysis:
    signal: str
    confidence: float


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
            return DummyAnalysis(signal="BUY", confidence=80.0)

        out, _summary = run_backtest(df, analyzer=analyzer, threshold=70, exit_after=5)
        self.assertGreater(len(out), 0)
        self.assertTrue((out["Signal"] == "LONG").all())

    def test_backtest_short_entries_use_inverse_threshold(self) -> None:
        df = self._sample_df()

        def analyzer(_slice: pd.DataFrame) -> DummyAnalysis:
            return DummyAnalysis(signal="SELL", confidence=20.0)

        out, _summary = run_backtest(df, analyzer=analyzer, threshold=70, exit_after=5)
        self.assertGreater(len(out), 0)
        self.assertTrue((out["Signal"] == "SHORT").all())

    def test_backtest_short_is_blocked_when_confidence_is_high(self) -> None:
        df = self._sample_df()

        def analyzer(_slice: pd.DataFrame) -> DummyAnalysis:
            return DummyAnalysis(signal="SELL", confidence=80.0)

        out, _summary = run_backtest(df, analyzer=analyzer, threshold=70, exit_after=5)
        self.assertEqual(len(out), 0)

    def test_backtest_positions_do_not_overlap(self) -> None:
        df = self._sample_df()

        def analyzer(_slice: pd.DataFrame) -> DummyAnalysis:
            return DummyAnalysis(signal="BUY", confidence=80.0)

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
        self.assertTrue(hasattr(result, "confidence"))
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
