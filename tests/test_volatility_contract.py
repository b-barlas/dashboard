import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from core.signals import analyse


def _sample_df(n: int = 180) -> pd.DataFrame:
    ts = pd.date_range("2026-01-01", periods=n, freq="h")
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


class VolatilityContractTests(unittest.TestCase):
    def test_atr_pandas_na_does_not_crash_and_returns_empty_label(self) -> None:
        df = _sample_df()

        def _atr_pdna(_high, _low, _close, window=14):
            return pd.Series([pd.NA] * len(df), index=df.index)

        with patch("core.signals.ta.volatility.average_true_range", side_effect=_atr_pdna):
            out = analyse(df)

        self.assertEqual(out.atr_comment, "")

    def test_atr_low_returns_low_label(self) -> None:
        df = _sample_df()

        def _atr_low(_high, _low, _close, window=14):
            return pd.Series(1.0, index=df.index)

        with patch("core.signals.ta.volatility.average_true_range", side_effect=_atr_low):
            out = analyse(df)

        self.assertEqual(out.atr_comment, "▼ Low")

    def test_atr_high_returns_high_label(self) -> None:
        df = _sample_df()

        def _atr_high(_high, _low, _close, window=14):
            return pd.Series(10.0, index=df.index)

        with patch("core.signals.ta.volatility.average_true_range", side_effect=_atr_high):
            out = analyse(df)

        self.assertEqual(out.atr_comment, "▲ High")


if __name__ == "__main__":
    unittest.main()
