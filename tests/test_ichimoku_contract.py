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


class _FakeIchimoku:
    def __init__(
        self,
        index: pd.Index,
        senkou_a,
        senkou_b,
        tenkan=0.0,
        kijun=0.0,
    ) -> None:
        self._index = index
        self._sa = senkou_a
        self._sb = senkou_b
        self._tenkan = tenkan
        self._kijun = kijun

    def _series(self, value) -> pd.Series:
        if isinstance(value, pd.Series):
            return value.reindex(self._index)
        if isinstance(value, (list, tuple, np.ndarray)):
            arr = np.asarray(value)
            if arr.shape[0] != len(self._index):
                raise ValueError("Fake Ichimoku series length mismatch")
            return pd.Series(arr, index=self._index)
        return pd.Series(value, index=self._index)

    def ichimoku_conversion_line(self) -> pd.Series:
        return self._series(self._tenkan)

    def ichimoku_base_line(self) -> pd.Series:
        return self._series(self._kijun)

    def ichimoku_a(self) -> pd.Series:
        return self._series(self._sa)

    def ichimoku_b(self) -> pd.Series:
        return self._series(self._sb)


def _ichimoku_factory(sa: float, sb: float, tenkan: float = 0.0, kijun: float = 0.0):
    def _factory(high, low, window1=9, window2=26, window3=52, visual=False):
        return _FakeIchimoku(high.index, sa, sb, tenkan=tenkan, kijun=kijun)

    return _factory


class IchimokuContractTests(unittest.TestCase):
    def test_ichimoku_bullish_when_close_above_cloud(self) -> None:
        df = _sample_df()
        with patch("core.signals.ta.trend.IchimokuIndicator", side_effect=_ichimoku_factory(110.0, 108.0, tenkan=111.0, kijun=109.0)):
            out = analyse(df)
        self.assertEqual(out.ichimoku, "Bullish")
        self.assertEqual(out.ichimoku_tk_cross, "▲ Bullish")
        self.assertEqual(out.ichimoku_future_bias, "▲ Bullish")
        self.assertIn("Thick", out.ichimoku_cloud_strength)

    def test_ichimoku_bearish_when_close_below_cloud(self) -> None:
        df = _sample_df()
        with patch("core.signals.ta.trend.IchimokuIndicator", side_effect=_ichimoku_factory(130.0, 132.0, tenkan=129.0, kijun=131.0)):
            out = analyse(df)
        self.assertEqual(out.ichimoku, "Bearish")
        self.assertEqual(out.ichimoku_tk_cross, "▼ Bearish")
        self.assertEqual(out.ichimoku_future_bias, "▼ Bearish")
        self.assertIn("Thick", out.ichimoku_cloud_strength)

    def test_ichimoku_neutral_when_close_inside_cloud(self) -> None:
        df = _sample_df()
        with patch("core.signals.ta.trend.IchimokuIndicator", side_effect=_ichimoku_factory(121.0, 119.0, tenkan=120.0, kijun=120.0)):
            out = analyse(df)
        self.assertEqual(out.ichimoku, "Neutral")
        self.assertEqual(out.ichimoku_tk_cross, "→ Neutral")
        self.assertEqual(out.ichimoku_future_bias, "▲ Bullish")

    def test_ichimoku_unavailable_when_cloud_missing(self) -> None:
        df = _sample_df()
        with patch("core.signals.ta.trend.IchimokuIndicator", side_effect=_ichimoku_factory(np.nan, np.nan)):
            out = analyse(df)
        self.assertEqual(out.ichimoku, "Unavailable")
        self.assertEqual(out.ichimoku_tk_cross, "")
        self.assertEqual(out.ichimoku_future_bias, "")
        self.assertEqual(out.ichimoku_cloud_strength, "")

    def test_ichimoku_current_vs_future_cloud_are_separated(self) -> None:
        df = _sample_df()
        n = len(df.index)
        # Current cloud (shifted by 26) should be bearish:
        # value taken from t-26 => 140 / 130, while close is near 120.
        # Future cloud (raw latest) is bullish: 80 / 70.
        sa = np.concatenate([np.full(n - 26, 140.0), np.full(26, 80.0)])
        sb = np.concatenate([np.full(n - 26, 130.0), np.full(26, 70.0)])
        with patch(
            "core.signals.ta.trend.IchimokuIndicator",
            side_effect=_ichimoku_factory(sa, sb, tenkan=111.0, kijun=109.0),
        ):
            out = analyse(df)
        self.assertEqual(out.ichimoku, "Bearish")
        self.assertEqual(out.ichimoku_future_bias, "▲ Bullish")


if __name__ == "__main__":
    unittest.main()
