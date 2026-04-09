import unittest

import pandas as pd

from tabs.whale_tab import (
    _build_exchange_momentum_fallback,
    _compute_scan_thresholds,
    _fmt_momentum_price,
    _run_volume_anomaly_scan,
)


class WhaleLogicTests(unittest.TestCase):
    def test_fmt_momentum_price_scales_small_values(self) -> None:
        self.assertEqual(_fmt_momentum_price(None), "N/A")
        self.assertEqual(_fmt_momentum_price(0), "N/A")
        self.assertEqual(_fmt_momentum_price(150.0), "$150.00")
        self.assertEqual(_fmt_momentum_price(0.4235068), "$0.4235")
        self.assertEqual(_fmt_momentum_price(0.00350558), "$0.003506")

    def test_compute_scan_thresholds_applies_timeframe_floors(self) -> None:
        ratio_gate, z_gate, extreme_ratio_gate, extreme_z_gate = _compute_scan_thresholds("1h", 1.2, 1.0)
        self.assertEqual(ratio_gate, 1.5)
        self.assertEqual(z_gate, 1.8)
        self.assertEqual(extreme_ratio_gate, 2.2)
        self.assertEqual(extreme_z_gate, 2.7)

    def test_run_volume_anomaly_scan_uses_closed_candle_and_sorts_by_score(self) -> None:
        base_ts = pd.date_range("2026-01-01", periods=90, freq="h", tz="UTC")

        def _frame(base_volume: float, spike_volume: float, closes: list[float]) -> pd.DataFrame:
            volumes = [base_volume + ((i % 5) * 50.0) for i in range(88)] + [spike_volume, spike_volume * 0.2]
            return pd.DataFrame(
                {
                    "timestamp": base_ts,
                    "open": closes,
                    "high": closes,
                    "low": closes,
                    "close": closes,
                    "volume": volumes,
                }
            )

        btc_closes = [100.0 + i * 0.1 for i in range(90)]
        sol_closes = [50.0 + i * 0.05 for i in range(90)]
        frames = {
            "BTC/USDT": _frame(1000.0, 9000.0, btc_closes),
            "SOL/USDT": _frame(1000.0, 1800.0, sol_closes),
        }

        surges, diag = _run_volume_anomaly_scan(
            ["BTC/USDT", "SOL/USDT"],
            fetch_ohlcv=lambda symbol, *_args, **_kwargs: frames[symbol],
            scan_tf="1h",
            ratio_gate=1.5,
            z_gate=1.8,
            extreme_ratio_gate=2.2,
            extreme_z_gate=2.7,
        )

        self.assertEqual(diag["symbols"], 2)
        self.assertEqual(diag["with_data"], 2)
        self.assertEqual(diag["passed"], 2)
        self.assertEqual([row["Symbol"] for row in surges], ["BTC", "SOL"])
        self.assertEqual(surges[0]["Level"], "EXTREME")
        self.assertGreater(surges[0]["Score"], surges[1]["Score"])

    def test_exchange_momentum_fallback_builds_ranked_gainers_and_losers(self) -> None:
        base_ts = pd.date_range("2026-01-01", periods=30, freq="h", tz="UTC")

        def _frame(start: float, end: float) -> pd.DataFrame:
            closes = [start + (end - start) * (i / 29.0) for i in range(30)]
            return pd.DataFrame(
                {
                    "timestamp": base_ts,
                    "open": closes,
                    "high": closes,
                    "low": closes,
                    "close": closes,
                    "volume": [1000.0] * 30,
                }
            )

        frames = {
            "BTC/USDT": _frame(100.0, 120.0),
            "ETH/USDT": _frame(100.0, 90.0),
            "SOL/USDT": _frame(100.0, 130.0),
        }

        gainers, losers = _build_exchange_momentum_fallback(
            fetch_ohlcv=lambda symbol, *_args, **_kwargs: frames.get(symbol),
            get_top_volume_usdt_symbols=lambda top_n=20: (["BTC/USDT", "ETH/USDT", "SOL/USDT"], []),
            limit=2,
        )

        self.assertEqual([row["symbol"] for row in gainers], ["sol", "btc"])
        self.assertEqual([row["symbol"] for row in losers], ["eth", "btc"])


if __name__ == "__main__":
    unittest.main()
