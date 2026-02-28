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


class CCIContractTests(unittest.TestCase):
    def test_cci_nan_returns_empty_label(self) -> None:
        df = _sample_df()

        def _cci_nan(_high, _low, _close, window=20):
            return pd.Series(np.nan, index=df.index)

        with patch("core.signals.ta.trend.cci", side_effect=_cci_nan):
            out = analyse(df)

        self.assertEqual(out.cci, "")

    def test_cci_high_returns_overbought_label(self) -> None:
        df = _sample_df()

        def _cci_high(_high, _low, _close, window=20):
            return pd.Series(150.0, index=df.index)

        with patch("core.signals.ta.trend.cci", side_effect=_cci_high):
            out = analyse(df)

        self.assertEqual(out.cci, "🔴 Overbought")

    def test_cci_low_returns_oversold_label(self) -> None:
        df = _sample_df()

        def _cci_low(_high, _low, _close, window=20):
            return pd.Series(-150.0, index=df.index)

        with patch("core.signals.ta.trend.cci", side_effect=_cci_low):
            out = analyse(df)

        self.assertEqual(out.cci, "🟢 Oversold")


if __name__ == "__main__":
    unittest.main()
