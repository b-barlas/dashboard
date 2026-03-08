import unittest
from unittest.mock import patch

try:
    from core.policy import UK_SAFE_EXCHANGE_FALLBACKS
    import core.services as svc
    DEPS_OK = True
except Exception:
    DEPS_OK = False


@unittest.skipUnless(DEPS_OK, "Missing dependencies for services tests")
class ServicesHttpTests(unittest.TestCase):
    def _fake_st(self, session_state: dict | None = None):
        class FakeSt:
            def __init__(self, state: dict | None = None) -> None:
                self.session_state = state or {}

        return FakeSt(session_state)

    def test_exchange_fallback_order_is_uk_safe(self):
        self.assertEqual(
            [name for name, _ in svc._EXCHANGE_CONFIGS],
            [name for name, _ in UK_SAFE_EXCHANGE_FALLBACKS],
        )

    def test_http_get_json_success_first_try(self):
        class Resp:
            status_code = 200

            @staticmethod
            def json():
                return {"ok": True}

        with patch("core.services.requests.get", return_value=Resp()):
            out = svc._http_get_json("https://example.com", retries=3)
            self.assertEqual(out, {"ok": True})

    def test_http_get_json_retries_then_returns_none(self):
        class Resp:
            status_code = 500

            @staticmethod
            def json():
                return {}

        with patch("core.services.requests.get", return_value=Resp()) as mocked_get:
            with patch("core.services.time.sleep", return_value=None):
                out = svc._http_get_json("https://example.com", retries=3)
        self.assertIsNone(out)
        self.assertEqual(mocked_get.call_count, 3)

    def test_get_top_volume_usdt_symbols_filters_by_available_markets(self):
        fake_data = [
            {"symbol": "btc"},
            {"symbol": "eth"},
            {"symbol": "xrp"},
            {"symbol": "btc"},  # duplicate symbol should be ignored
        ]
        with patch("core.services._http_get_json", return_value=fake_data):
            with patch("core.services.MARKETS", {"BTC/USDT": {}, "ETH/USD": {}}):
                pairs, raw = svc.get_top_volume_usdt_symbols(top_n=10)
        self.assertEqual(raw, fake_data)
        self.assertEqual(pairs, ["BTC/USDT", "ETH/USD"])

    def test_fetch_top_gainers_losers_sorts_and_limits(self):
        sample = [
            {"symbol": "a", "price_change_percentage_24h": 2.0},
            {"symbol": "b", "price_change_percentage_24h": -5.0},
            {"symbol": "c", "price_change_percentage_24h": 8.0},
            {"symbol": "d", "price_change_percentage_24h": 0.0},
        ]
        with patch("core.services._http_get_json", return_value=sample):
            gainers, losers = svc.fetch_top_gainers_losers(limit=2)
        self.assertEqual([g["symbol"] for g in gainers], ["c", "a"])
        self.assertEqual([l["symbol"] for l in losers], ["b", "d"])

    def test_market_top_snapshot_drops_stale_fields_even_if_other_live_fields_arrive(self):
        fake_st = self._fake_st(
            {
                svc._TOP_SNAPSHOT_KEY: {
                    "total_mcap": 3_100_000_000_000,
                    "btc_dom": 55.0,
                },
                svc._TOP_SNAPSHOT_META_KEY: {
                    "total_mcap": 0.0,
                    "btc_dom": 0.0,
                },
            }
        )
        with patch("core.services.st", fake_st):
            with patch("core.services.time.time", return_value=10_000.0):
                with patch(
                    "core.services._fetch_pair_price_change_from_exchange",
                    side_effect=[(90_000.0, 1.2), (2_200.0, -0.5)],
                ):
                    with patch("core.services._fetch_btc_eth_from_coingecko_with_change", return_value={}):
                        with patch("core.services.get_market_indices", return_value=(0.0, 0.0, 0, 0, 0.0, 0.0, 0.0, 0.0, 0.0)):
                            with patch("core.services._fetch_market_indices_coinpaprika", return_value=None):
                                with patch("core.services.get_fear_greed_core", side_effect=RuntimeError("down")):
                                    out = svc.get_market_top_snapshot()
        self.assertEqual(out["btc_price"], 90_000.0)
        self.assertIsNone(out["total_mcap"])
        self.assertNotIn("total_mcap", fake_st.session_state[svc._TOP_SNAPSHOT_KEY])
        self.assertNotIn("btc_dom", fake_st.session_state[svc._TOP_SNAPSHOT_META_KEY])

    def test_market_top_snapshot_keeps_fresh_previous_field_within_ttl(self):
        fake_st = self._fake_st(
            {
                svc._TOP_SNAPSHOT_KEY: {
                    "fg_value": 68.0,
                    "fg_label": "Greed",
                },
                svc._TOP_SNAPSHOT_META_KEY: {
                    "fg_value": 9_000.0,
                    "fg_label": 9_000.0,
                },
            }
        )
        with patch("core.services.st", fake_st):
            with patch("core.services.time.time", return_value=10_000.0):
                with patch("core.services._fetch_pair_price_change_from_exchange", side_effect=[(None, None), (None, None)]):
                    with patch("core.services._fetch_btc_eth_from_coingecko_with_change", return_value={}):
                        with patch("core.services.get_market_indices", return_value=(0.0, 0.0, 0, 0, 0.0, 0.0, 0.0, 0.0, 0.0)):
                            with patch("core.services._fetch_market_indices_coinpaprika", return_value=None):
                                with patch("core.services.get_fear_greed_core", side_effect=RuntimeError("down")):
                                    out = svc.get_market_top_snapshot()
        self.assertEqual(out["fg_value"], 68.0)
        self.assertEqual(out["fg_label"], "Greed")

    def test_debug_uses_stdout_from_worker_threads(self):
        fake_st = self._fake_st({"debug_mode": True})

        class Sidebar:
            def __init__(self) -> None:
                self.calls: list[str] = []

            def write(self, msg: str) -> None:
                self.calls.append(msg)

        fake_st.sidebar = Sidebar()
        worker_thread = object()
        main_thread = object()
        with patch("core.services.st", fake_st):
            with patch("core.services.threading.current_thread", return_value=worker_thread):
                with patch("core.services.threading.main_thread", return_value=main_thread):
                    with patch("builtins.print") as mocked_print:
                        svc._debug("hello")
        self.assertEqual(fake_st.sidebar.calls, [])
        mocked_print.assert_called_once()

    def test_fetch_exchange_tickers_snapshot_retries_uncached_when_cached_empty(self):
        with patch("core.services._fetch_exchange_tickers_snapshot_cached", return_value={}):
            with patch(
                "core.services._fetch_exchange_tickers_snapshot_uncached",
                return_value={"BTC/USD": {"last": 90_000.0}},
            ) as mocked_uncached:
                out = svc.fetch_exchange_tickers_snapshot()
        self.assertEqual(out, {"BTC/USD": {"last": 90_000.0}})
        mocked_uncached.assert_called_once()

    def test_get_market_cap_rows_for_symbols_normalizes_aliases_for_cache_key(self):
        with patch(
            "core.services._fetch_market_cap_rows_for_symbols_cached",
            return_value=[{"symbol": "btc", "market_cap": 22}],
        ) as mocked_cached:
            out = svc.get_market_cap_rows_for_symbols(["XBT", "BTC/USDT"])
        self.assertEqual(out, [{"symbol": "btc", "market_cap": 22}])
        mocked_cached.assert_called_once_with(("BTC",), vs_currency="usd")

    def test_get_market_cap_rows_for_symbols_uses_order_independent_cache_key(self):
        with patch(
            "core.services._fetch_market_cap_rows_for_symbols_cached",
            return_value=[{"symbol": "btc", "market_cap": 22}, {"symbol": "eth", "market_cap": 11}],
        ) as mocked_cached:
            svc.get_market_cap_rows_for_symbols(["ETH", "BTC"])
        mocked_cached.assert_called_once_with(("BTC", "ETH"), vs_currency="usd")

    def test_get_market_cap_rows_for_symbols_retries_uncached_when_cached_empty(self):
        with patch("core.services._fetch_market_cap_rows_for_symbols_cached", return_value=[]):
            with patch(
                "core.services.fetch_market_cap_rows_for_symbols_core",
                return_value=[{"symbol": "doge", "market_cap": 11}],
            ) as mocked_core:
                out = svc.get_market_cap_rows_for_symbols(["DOGE"])
        self.assertEqual(out, [{"symbol": "doge", "market_cap": 11}])
        mocked_core.assert_called_once()

    def test_get_fear_greed_uses_cached_fetch_before_uncached_retry(self):
        with patch("core.services._fetch_fear_greed_cached", return_value=(61, "Greed")) as mocked_cached:
            with patch("core.services.get_fear_greed_core", side_effect=AssertionError("uncached retry should not run")):
                out = svc.get_fear_greed()
        self.assertEqual(out, (61.0, "Greed"))
        mocked_cached.assert_called_once()

    def test_market_top_snapshot_uses_wrapped_fear_greed_fetch(self):
        fake_st = self._fake_st()
        with patch("core.services.st", fake_st):
            with patch("core.services.time.time", return_value=10_000.0):
                with patch(
                    "core.services._fetch_pair_price_change_from_exchange",
                    side_effect=[(90_000.0, 1.2), (2_200.0, -0.5)],
                ):
                    with patch("core.services._fetch_btc_eth_from_coingecko_with_change", return_value={}):
                        with patch("core.services.get_market_indices", return_value=(54.0, 18.0, 3_000_000_000_000, 2_000_000_000_000, 1.2, 3.0, 2.5, 1.0, 1.5)):
                            with patch("core.services.get_fear_greed", return_value=(61.0, "Greed")) as mocked_fg:
                                out = svc.get_market_top_snapshot()
        self.assertEqual(out["fg_value"], 61.0)
        self.assertEqual(out["fg_label"], "Greed")
        mocked_fg.assert_called_once()

    def test_exchange_pair_price_change_direct_fetch_is_cached(self):
        svc._fetch_pair_price_change_from_exchange_direct.clear()
        with patch.object(svc.EXCHANGE, "fetch_ticker", return_value={"last": 90_000.0, "percentage": 1.2}) as mocked_fetch:
            first = svc._fetch_pair_price_change_from_exchange_direct("BTC/USDT")
            second = svc._fetch_pair_price_change_from_exchange_direct("BTC/USDT")
        self.assertEqual(first, (90_000.0, 1.2))
        self.assertEqual(second, (90_000.0, 1.2))
        mocked_fetch.assert_called_once()


if __name__ == "__main__":
    unittest.main()
