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


class _StubPSAR:
    def __init__(self, up: pd.Series, down: pd.Series) -> None:
        self._up = up
        self._down = down

    def psar_up(self):
        return self._up

    def psar_down(self):
        return self._down


class PSARContractTests(unittest.TestCase):
    def test_psar_unavailable_returns_empty_label(self) -> None:
        df = _sample_df()

        with patch("core.signals.ta.trend.PSARIndicator", side_effect=RuntimeError("psar failed")):
            out = analyse(df)

        self.assertEqual(out.psar, "")

    def test_psar_bullish_when_close_above_psar(self) -> None:
        df = _sample_df()
        up = pd.Series(np.nan, index=df.index)
        # keep PSAR below price on the last candle
        down = pd.Series(df["close"].values - 1.0, index=df.index)

        with patch("core.signals.ta.trend.PSARIndicator", return_value=_StubPSAR(up, down)):
            out = analyse(df)

        self.assertEqual(out.psar, "▲ Bullish")

    def test_psar_bearish_when_close_below_psar(self) -> None:
        df = _sample_df()
        up = pd.Series(df["close"].values + 1.0, index=df.index)
        down = pd.Series(np.nan, index=df.index)

        with patch("core.signals.ta.trend.PSARIndicator", return_value=_StubPSAR(up, down)):
            out = analyse(df)

        self.assertEqual(out.psar, "▼ Bearish")


if __name__ == "__main__":
    unittest.main()
