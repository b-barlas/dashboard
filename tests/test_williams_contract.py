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


class WilliamsContractTests(unittest.TestCase):
    def test_williams_nan_returns_empty_label(self) -> None:
        df = _sample_df()

        def _wr_nan(_high, _low, _close, lbp=14):
            return pd.Series(np.nan, index=df.index)

        with patch("core.signals.ta.momentum.williams_r", side_effect=_wr_nan):
            out = analyse(df)

        self.assertEqual(out.williams, "")

    def test_williams_pandas_na_returns_empty_label(self) -> None:
        df = _sample_df()

        def _wr_pdna(_high, _low, _close, lbp=14):
            return pd.Series([pd.NA] * len(df), index=df.index)

        with patch("core.signals.ta.momentum.williams_r", side_effect=_wr_pdna):
            out = analyse(df)

        self.assertEqual(out.williams, "")

    def test_williams_oversold_and_overbought_labels(self) -> None:
        df = _sample_df()

        def _wr_oversold(_high, _low, _close, lbp=14):
            return pd.Series(-95.0, index=df.index)

        with patch("core.signals.ta.momentum.williams_r", side_effect=_wr_oversold):
            out_os = analyse(df)
        self.assertEqual(out_os.williams, "🟢 Oversold")

        def _wr_overbought(_high, _low, _close, lbp=14):
            return pd.Series(-5.0, index=df.index)

        with patch("core.signals.ta.momentum.williams_r", side_effect=_wr_overbought):
            out_ob = analyse(df)
        self.assertEqual(out_ob.williams, "🔴 Overbought")


if __name__ == "__main__":
    unittest.main()
