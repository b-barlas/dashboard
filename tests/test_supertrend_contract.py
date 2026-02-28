import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from core.signals import analyse


def _sample_df(n: int = 180) -> pd.DataFrame:
    ts = pd.date_range("2026-01-01", periods=n, freq="h")
    base = np.linspace(100.0, 120.0, n)
    wiggle = np.sin(np.linspace(0, 8, n)) * 0.5
    close = base + wiggle
    open_ = close - 0.12
    high = close + 0.30
    low = close - 0.30
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


def _supertrend_factory(level: float):
    def _factory(high, low, close, length=10, multiplier=3.0):
        idx = close.index
        return pd.DataFrame(
            {
                f"SUPERT_{length}_{multiplier}": pd.Series(level, index=idx),
                f"SUPERTd_{length}_{multiplier}": pd.Series(1.0, index=idx),
            },
            index=idx,
        )

    return _factory


class SuperTrendContractTests(unittest.TestCase):
    def test_supertrend_bullish_when_close_above_level(self) -> None:
        df = _sample_df()
        with patch("core.signals.supertrend", side_effect=_supertrend_factory(90.0)):
            out = analyse(df)
        self.assertEqual(out.supertrend, "Bullish")

    def test_supertrend_bearish_when_close_below_level(self) -> None:
        df = _sample_df()
        with patch("core.signals.supertrend", side_effect=_supertrend_factory(140.0)):
            out = analyse(df)
        self.assertEqual(out.supertrend, "Bearish")

    def test_supertrend_neutral_on_exact_touch(self) -> None:
        df = _sample_df()
        last_close = float(df["close"].iloc[-1])
        with patch("core.signals.supertrend", side_effect=_supertrend_factory(last_close)):
            out = analyse(df)
        self.assertEqual(out.supertrend, "Neutral")

    def test_supertrend_unavailable_when_level_missing(self) -> None:
        df = _sample_df()
        with patch("core.signals.supertrend", side_effect=_supertrend_factory(np.nan)):
            out = analyse(df)
        self.assertEqual(out.supertrend, "Unavailable")


if __name__ == "__main__":
    unittest.main()

