import unittest

import pandas as pd

from core.signals import detect_candle_pattern


def _df_from_ohlc(rows):
    data = []
    ts = pd.Timestamp("2026-01-01")
    for i, (o, h, l, c) in enumerate(rows):
        data.append(
            {
                "timestamp": ts + pd.Timedelta(minutes=i),
                "open": float(o),
                "high": float(h),
                "low": float(l),
                "close": float(c),
                "volume": 1000.0,
            }
        )
    return pd.DataFrame(data)


class CandlePatternContractTests(unittest.TestCase):
    def test_detect_returns_neutral_doji_with_arrow(self):
        df = _df_from_ohlc(
            [
                (10.0, 10.6, 9.8, 10.4),
                (10.4, 10.7, 10.0, 10.1),
                (10.1, 10.3, 9.9, 10.2),
                (10.2, 10.4, 10.0, 10.2),
                # tiny body vs wide range -> doji
                (10.20, 10.80, 9.60, 10.21),
            ]
        )
        out = detect_candle_pattern(df)
        self.assertTrue(out.startswith("→ Doji"), out)

    def test_detect_returns_bullish_harami(self):
        df = _df_from_ohlc(
            [
                (10.5, 10.8, 10.2, 10.6),
                (10.6, 10.9, 10.3, 10.4),
                (10.4, 10.6, 10.1, 10.2),
                # strong bearish previous candle
                (10.0, 10.1, 8.9, 9.0),
                # small bullish body inside previous body
                (9.2, 9.6, 9.1, 9.5),
            ]
        )
        out = detect_candle_pattern(df)
        self.assertTrue(out.startswith("▲ Bullish Harami"), out)

    def test_detect_returns_bearish_marubozu(self):
        df = _df_from_ohlc(
            [
                (10.0, 10.4, 9.8, 10.2),
                (10.2, 10.3, 10.0, 10.1),
                (10.1, 10.4, 10.0, 10.2),
                (10.3, 10.4, 10.0, 10.05),
                # dominant bearish body, very small shadows
                (10.5, 10.52, 8.80, 8.82),
            ]
        )
        out = detect_candle_pattern(df)
        self.assertTrue(out.startswith("▼ Bearish Marubozu"), out)

    def test_detect_returns_shooting_star_for_bearish_upper_wick(self):
        df = _df_from_ohlc(
            [
                (10.0, 10.3, 9.9, 10.2),
                (10.2, 10.4, 10.0, 10.3),
                (10.3, 10.6, 10.2, 10.4),
                (10.4, 10.6, 10.3, 10.5),
                # bearish small body near low with long upper shadow
                (10.50, 11.20, 10.45, 10.46),
            ]
        )
        out = detect_candle_pattern(df)
        self.assertTrue(out.startswith("▼ Shooting Star"), out)

    def test_detect_returns_hanging_man_in_uptrend(self):
        df = _df_from_ohlc(
            [
                (10.0, 10.3, 9.9, 10.2),
                (10.2, 10.7, 10.1, 10.5),
                (10.5, 10.9, 10.4, 10.8),
                (10.8, 11.2, 10.7, 11.0),
                # small real body near top, long lower wick after uptrend
                (11.05, 11.08, 10.30, 10.98),
            ]
        )
        out = detect_candle_pattern(df)
        self.assertTrue(out.startswith("▼ Hanging Man"), out)


if __name__ == "__main__":
    unittest.main()
