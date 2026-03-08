import unittest
from unittest.mock import patch

import pandas as pd

from core import data


class _Resp:
    def __init__(self, status_code: int, payload):
        self.status_code = status_code
        self._payload = payload

    def json(self):
        return self._payload


class DataFallbackTests(unittest.TestCase):
    def test_symbol_variants_include_btc_xbt_alias_path(self):
        variants = data.symbol_variants("BTC/USDT")
        self.assertIn("BTC/USDT", variants)
        self.assertIn("BTC/USD", variants)
        self.assertIn("XBT/USD", variants)

    def test_coingecko_coin_id_uses_override_without_http(self):
        with patch("core.data.requests.get") as mocked_get:
            out = data.coingecko_coin_id("BTC")
        self.assertEqual(out, "bitcoin")
        mocked_get.assert_not_called()

    def test_coingecko_coin_id_returns_unique_exact_match(self):
        payload = {"coins": [{"id": "coin-xyz", "symbol": "xyz", "market_cap_rank": 45}]}
        with patch("core.data.requests.get", return_value=_Resp(200, payload)):
            out = data.coingecko_coin_id("XYZZ")
        self.assertIsNone(out)

        with patch("core.data.requests.get", return_value=_Resp(200, payload)):
            out = data.coingecko_coin_id("XYZ")
        self.assertEqual(out, "coin-xyz")

    def test_coingecko_coin_id_rejects_ambiguous_ranked_matches(self):
        payload = {
            "coins": [
                {"id": "coin-a", "symbol": "abc", "market_cap_rank": 120},
                {"id": "coin-b", "symbol": "ABC", "market_cap_rank": 450},
            ]
        }
        with patch("core.data.requests.get", return_value=_Resp(200, payload)):
            out = data.coingecko_coin_id("ABC")
        self.assertIsNone(out)

    def test_market_chart_resample_builds_real_ohlcv_when_granularity_is_sufficient(self):
        prices = [
            [1735689600000, 100.0],
            [1735693200000, 101.0],
            [1735696800000, 99.0],
            [1735700400000, 102.0],
            [1735704000000, 103.0],
            [1735707600000, 104.0],
            [1735711200000, 105.0],
            [1735714800000, 106.0],
        ]
        volumes = [
            [1735689600000, 10.0],
            [1735693200000, 11.0],
            [1735696800000, 12.0],
            [1735700400000, 13.0],
            [1735704000000, 14.0],
            [1735707600000, 15.0],
            [1735711200000, 16.0],
            [1735714800000, 17.0],
        ]

        def _fake_get(url, **kwargs):
            if url.endswith("/ohlc"):
                return _Resp(404, {})
            if url.endswith("/market_chart"):
                return _Resp(200, {"prices": prices, "total_volumes": volumes})
            raise AssertionError(f"unexpected url: {url}")

        with patch("core.data.requests.get", side_effect=_fake_get):
            out = data.coingecko_market_chart("bitcoin", 7, "4h")

        self.assertIsNotNone(out)
        self.assertEqual(list(out.columns), ["timestamp", "open", "high", "low", "close", "volume"])
        self.assertEqual(len(out), 2)
        first = out.iloc[0]
        second = out.iloc[1]
        self.assertEqual(first["open"], 100.0)
        self.assertEqual(first["high"], 102.0)
        self.assertEqual(first["low"], 99.0)
        self.assertEqual(first["close"], 102.0)
        self.assertEqual(second["open"], 103.0)
        self.assertEqual(second["high"], 106.0)
        self.assertEqual(second["low"], 103.0)
        self.assertEqual(second["close"], 106.0)
        self.assertTrue(bool(out.attrs.get("volume_is_24h_aggregate")))

    def test_market_chart_resample_rejects_too_coarse_data(self):
        prices = [
            [1735689600000, 100.0],
            [1735693200000, 101.0],
            [1735696800000, 99.0],
            [1735700400000, 102.0],
        ]

        def _fake_get(url, **kwargs):
            if url.endswith("/ohlc"):
                return _Resp(404, {})
            if url.endswith("/market_chart"):
                return _Resp(200, {"prices": prices, "total_volumes": []})
            raise AssertionError(f"unexpected url: {url}")

        with patch("core.data.requests.get", side_effect=_fake_get):
            out = data.coingecko_market_chart("bitcoin", 7, "15m")

        self.assertIsNone(out)

    def test_ohlc_endpoint_rejects_coarse_granularity_and_uses_market_chart_fallback(self):
        coarse_ohlc = [
            [1735689600000, 100.0, 102.0, 99.0, 101.0],
            [1735704000000, 101.0, 103.0, 100.0, 102.0],
            [1735718400000, 102.0, 104.0, 101.0, 103.0],
            [1735732800000, 103.0, 105.0, 102.0, 104.0],
            [1735747200000, 104.0, 106.0, 103.0, 105.0],
            [1735761600000, 105.0, 107.0, 104.0, 106.0],
        ]
        prices = [
            [1735689600000, 100.0],
            [1735693200000, 101.0],
            [1735696800000, 99.0],
            [1735700400000, 102.0],
            [1735704000000, 103.0],
            [1735707600000, 104.0],
            [1735711200000, 105.0],
            [1735714800000, 106.0],
        ]
        volumes = [
            [1735689600000, 10.0],
            [1735693200000, 11.0],
            [1735696800000, 12.0],
            [1735700400000, 13.0],
            [1735704000000, 14.0],
            [1735707600000, 15.0],
            [1735711200000, 16.0],
            [1735714800000, 17.0],
        ]

        def _fake_get(url, **kwargs):
            if url.endswith("/ohlc"):
                return _Resp(200, coarse_ohlc)
            if url.endswith("/market_chart"):
                return _Resp(200, {"prices": prices, "total_volumes": volumes})
            raise AssertionError(f"unexpected url: {url}")

        with patch("core.data.requests.get", side_effect=_fake_get):
            out = data.coingecko_market_chart("bitcoin", 7, "1h")

        self.assertIsNotNone(out)
        self.assertEqual(len(out), 8)
        self.assertEqual(float(out["open"].iloc[0]), 100.0)
        self.assertEqual(float(out["close"].iloc[-1]), 106.0)
        self.assertTrue(bool(out.attrs.get("volume_is_24h_aggregate")))

    def test_coingecko_market_chart_reuses_market_chart_payload_after_coarse_ohlc(self):
        coarse_ohlc = [
            [1735689600000, 100.0, 102.0, 99.0, 101.0],
            [1735704000000, 101.0, 103.0, 100.0, 102.0],
            [1735718400000, 102.0, 104.0, 101.0, 103.0],
            [1735732800000, 103.0, 105.0, 102.0, 104.0],
            [1735747200000, 104.0, 106.0, 103.0, 105.0],
            [1735761600000, 105.0, 107.0, 104.0, 106.0],
        ]
        prices = [
            [1735689600000, 100.0],
            [1735693200000, 101.0],
            [1735696800000, 99.0],
            [1735700400000, 102.0],
            [1735704000000, 103.0],
            [1735707600000, 104.0],
            [1735711200000, 105.0],
            [1735714800000, 106.0],
        ]
        volumes = [
            [1735689600000, 10.0],
            [1735693200000, 11.0],
            [1735696800000, 12.0],
            [1735700400000, 13.0],
            [1735704000000, 14.0],
            [1735707600000, 15.0],
            [1735711200000, 16.0],
            [1735714800000, 17.0],
        ]
        calls: list[str] = []

        def _fake_get(url, **kwargs):
            if url.endswith("/ohlc"):
                calls.append("ohlc")
                return _Resp(200, coarse_ohlc)
            if url.endswith("/market_chart"):
                calls.append("market_chart")
                return _Resp(200, {"prices": prices, "total_volumes": volumes})
            raise AssertionError(f"unexpected url: {url}")

        with patch("core.data.requests.get", side_effect=_fake_get):
            out = data.coingecko_market_chart("bitcoin", 7, "1h")

        self.assertIsNotNone(out)
        self.assertEqual(calls.count("market_chart"), 1)

    def test_ohlc_endpoint_still_merges_volume_when_available(self):
        ohlc = [
            [1735689600000, 100.0, 102.0, 99.0, 101.0],
            [1735693200000, 101.0, 103.0, 100.0, 102.0],
            [1735696800000, 102.0, 104.0, 101.0, 103.0],
            [1735700400000, 103.0, 105.0, 102.0, 104.0],
            [1735704000000, 104.0, 106.0, 103.0, 105.0],
            [1735707600000, 105.0, 107.0, 104.0, 106.0],
        ]
        volumes = [
            [1735689600000, 10.0],
            [1735693200000, 11.0],
            [1735696800000, 12.0],
            [1735700400000, 13.0],
            [1735704000000, 14.0],
            [1735707600000, 15.0],
        ]

        def _fake_get(url, **kwargs):
            if url.endswith("/ohlc"):
                return _Resp(200, ohlc)
            if url.endswith("/market_chart"):
                return _Resp(200, {"total_volumes": volumes})
            raise AssertionError(f"unexpected url: {url}")

        with patch("core.data.requests.get", side_effect=_fake_get):
            out = data.coingecko_market_chart("bitcoin", 7, "1h")

        self.assertIsInstance(out, pd.DataFrame)
        self.assertIn("volume", out.columns)
        self.assertEqual(float(out["volume"].iloc[0]), 10.0)
        self.assertTrue(bool(out.attrs.get("volume_is_24h_aggregate")))

    def test_fetch_ohlcv_marks_actual_exchange_pair_in_attrs(self):
        frame = pd.DataFrame(
            {
                "timestamp": pd.date_range("2026-01-01", periods=3, freq="h"),
                "open": [1.0, 2.0, 3.0],
                "high": [1.0, 2.0, 3.0],
                "low": [1.0, 2.0, 3.0],
                "close": [1.0, 2.0, 3.0],
                "volume": [10.0, 11.0, 12.0],
            }
        )

        calls: list[str] = []

        def _fake_cached_fetcher(symbol, timeframe, limit):
            calls.append(symbol)
            if symbol == "XBT/USD":
                return frame.copy()
            raise RuntimeError("missing")

        with patch("core.data.fetch_coingecko_ohlcv", return_value=None):
            out = data.fetch_ohlcv(object(), "BTC/USDT", "1h", 100, _fake_cached_fetcher)
        self.assertIsNotNone(out)
        self.assertEqual(calls, ["BTC/USDT", "BTC/USD", "XBT/USDT", "XBT/USD"])
        self.assertEqual(out.attrs.get("source_symbol"), "XBT/USD")
        self.assertEqual(out.attrs.get("source_provider"), "exchange")

    def test_fetch_ohlcv_does_not_call_coingecko_when_exchange_history_is_sufficient(self):
        frame = pd.DataFrame(
            {
                "timestamp": pd.date_range("2026-01-01", periods=100, freq="h"),
                "open": range(100),
                "high": range(100),
                "low": range(100),
                "close": range(100),
                "volume": range(100),
            }
        )

        with patch("core.data.fetch_coingecko_ohlcv") as mocked_cg:
            out = data.fetch_ohlcv(
                object(),
                "BTC/USDT",
                "1h",
                100,
                lambda *_args, **_kwargs: frame.copy(),
            )
        self.assertIsNotNone(out)
        mocked_cg.assert_not_called()
        self.assertEqual(out.attrs.get("source_provider"), "exchange")

    def test_fetch_ohlcv_keeps_trying_exchange_variants_after_short_history(self):
        short_frame = pd.DataFrame(
            {
                "timestamp": pd.date_range("2026-01-01", periods=10, freq="h"),
                "open": range(10),
                "high": range(10),
                "low": range(10),
                "close": range(10),
                "volume": range(10),
            }
        )
        long_frame = pd.DataFrame(
            {
                "timestamp": pd.date_range("2026-01-01", periods=80, freq="h"),
                "open": range(80),
                "high": range(80),
                "low": range(80),
                "close": range(80),
                "volume": range(80),
            }
        )
        calls: list[str] = []

        def _fake_cached_fetcher(symbol, timeframe, limit):
            calls.append(symbol)
            if symbol == "BTC/USDT":
                return short_frame.copy()
            if symbol == "BTC/USD":
                return long_frame.copy()
            raise RuntimeError("missing")

        with patch("core.data.fetch_coingecko_ohlcv", return_value=None):
            out = data.fetch_ohlcv(object(), "BTC/USDT", "1h", 100, _fake_cached_fetcher)
        self.assertIsNotNone(out)
        self.assertEqual(calls[:2], ["BTC/USDT", "BTC/USD"])
        self.assertEqual(len(out), 80)
        self.assertEqual(out.attrs.get("source_symbol"), "BTC/USD")
        self.assertEqual(out.attrs.get("source_provider"), "exchange")

    def test_fetch_coingecko_ohlcv_uses_timeframe_safe_day_window(self):
        with patch("core.data.coingecko_coin_id", return_value="bitcoin"):
            with patch("core.data.coingecko_market_chart", return_value=None) as mocked_market_chart:
                data.fetch_coingecko_ohlcv("BTC/USDT", "4h", 500)
        mocked_market_chart.assert_called_once_with("bitcoin", 85, "4h")

    def test_fetch_ohlcv_marks_coingecko_fallback_in_attrs(self):
        frame = pd.DataFrame(
            {
                "timestamp": pd.date_range("2026-01-01", periods=3, freq="h"),
                "open": [1.0, 2.0, 3.0],
                "high": [1.0, 2.0, 3.0],
                "low": [1.0, 2.0, 3.0],
                "close": [1.0, 2.0, 3.0],
                "volume": [10.0, 11.0, 12.0],
            }
        )

        with patch("core.data.fetch_coingecko_ohlcv", return_value=frame.copy()):
            out = data.fetch_ohlcv(
                object(),
                "BTC/USDT",
                "1h",
                100,
                lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("missing")),
            )
        self.assertIsNotNone(out)
        self.assertEqual(out.attrs.get("source_symbol"), "BTC/USDT")
        self.assertEqual(out.attrs.get("source_provider"), "coingecko")


if __name__ == "__main__":
    unittest.main()
