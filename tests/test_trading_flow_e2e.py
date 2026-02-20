from __future__ import annotations

import unittest

try:
    import numpy as np
    import pandas as pd
    from core.ml import ml_ensemble_predict
    from core.position_metrics import (
        compute_hard_invalidation,
        compute_health_decision,
        compute_position_pnl,
        estimate_liquidation,
    )
    from core.scalping import get_scalping_entry_target
    from core.signals import analyse

    DEPS_OK = True
except Exception:
    DEPS_OK = False


@unittest.skipUnless(DEPS_OK, "Missing dependencies for trading flow e2e tests")
class TradingFlowE2ETests(unittest.TestCase):
    def _sample_df(self, n: int = 220) -> pd.DataFrame:
        ts = pd.date_range("2025-01-01", periods=n, freq="h")
        trend = np.linspace(100.0, 130.0, n)
        wave = np.sin(np.linspace(0, 16, n)) * 0.8
        close = trend + wave
        open_ = close - 0.12
        high = close + 0.45
        low = close - 0.45
        volume = 1000.0 + (np.cos(np.linspace(0, 12, n)) * 60.0)
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

    def test_market_spot_position_flow_is_consistent(self) -> None:
        df = self._sample_df()
        df_eval = df.iloc[:-1].copy()

        # Spot/position core engines should produce valid outputs on same closed-candle frame.
        a = analyse(df_eval)
        self.assertIn(a.signal, {"STRONG BUY", "BUY", "WAIT", "SELL", "STRONG SELL", "NO DATA"})
        self.assertGreaterEqual(float(a.confidence), 0.0)
        self.assertLessEqual(float(a.confidence), 100.0)

        p, ai_dir, details = ml_ensemble_predict(df_eval)
        self.assertIsInstance(p, float)
        self.assertIn(ai_dir, {"LONG", "SHORT", "NEUTRAL"})
        self.assertIsInstance(details, dict)

        scalp = get_scalping_entry_target(
            df_eval,
            float(a.confidence),
            str(a.supertrend),
            str(a.ichimoku),
            str(a.vwap),
            bool(a.volume_spike),
            strict_mode=True,
            sr_lookback_fn=lambda _tf=None: 30,
        )
        self.assertEqual(len(scalp), 6)

        current_price = float(df["close"].iloc[-1])
        pnl_pack = compute_position_pnl(
            entry_price=current_price * 0.98,
            current_price=current_price,
            direction="LONG",
            leverage=5.0,
            margin_used=1000.0,
            funding_impact_pct=0.0,
        )
        self.assertGreaterEqual(float(pnl_pack["notional"]), 0.0)

        liq_pack = estimate_liquidation(
            entry_price=current_price * 0.98,
            current_price=current_price,
            direction="LONG",
            leverage=5.0,
        )
        self.assertIsNotNone(liq_pack["liq_price"])

        recent = df.tail(30)
        inv_pack = compute_hard_invalidation(
            direction="LONG",
            support=float(recent["low"].min()),
            resistance=float(recent["high"].max()),
            atr14=1.0,
            current_price=current_price,
        )
        h_pack = compute_health_decision(
            direction="LONG",
            signal_direction="LONG",
            confidence=float(a.confidence),
            conviction_label="MEDIUM",
            liq_distance_pct=float(liq_pack["distance_pct"]) if liq_pack["distance_pct"] is not None else None,
            invalidated=bool(inv_pack["invalidated"]),
            levered_pnl_pct=float(pnl_pack["levered_pct"]),
        )
        self.assertIn(h_pack["label"], {"HOLD", "REDUCE", "EXIT"})
        self.assertGreaterEqual(int(h_pack["score"]), 0)
        self.assertLessEqual(int(h_pack["score"]), 100)


if __name__ == "__main__":
    unittest.main()
