import unittest

from core import market_data


class MarketDataHelpersTests(unittest.TestCase):
    def test_fetch_trending_coins_maps_fields(self):
        payload = {
            "coins": [
                {"item": {"name": "Bitcoin", "symbol": "btc", "market_cap_rank": 1, "price_btc": 1.0, "score": 0}}
            ]
        }
        out = market_data.fetch_trending_coins(lambda *args, **kwargs: payload)
        self.assertEqual(len(out), 1)
        self.assertEqual(out[0]["symbol"], "BTC")

    def test_fetch_top_gainers_losers_orders(self):
        payload = [
            {"symbol": "a", "price_change_percentage_24h": -2.0},
            {"symbol": "b", "price_change_percentage_24h": 8.0},
            {"symbol": "c", "price_change_percentage_24h": 3.0},
        ]
        gainers, losers = market_data.fetch_top_gainers_losers(lambda *args, **kwargs: payload, limit=2)
        self.assertEqual([g["symbol"] for g in gainers], ["b", "c"])
        self.assertEqual([l["symbol"] for l in losers], ["a", "c"])

    def test_get_top_volume_usdt_symbols_filters_markets(self):
        payload = [{"symbol": "btc"}, {"symbol": "eth"}, {"symbol": "btc"}]
        pairs, raw = market_data.get_top_volume_usdt_symbols(
            lambda *args, **kwargs: payload,
            {"BTC/USDT": {}, "ETH/USD": {}},
            lambda _msg: None,
            top_n=10,
        )
        self.assertEqual(raw, payload)
        self.assertEqual(pairs, ["BTC/USDT", "ETH/USD"])

    def test_get_top_volume_usdt_symbols_supports_btc_xbt_alias(self):
        payload = [{"symbol": "btc"}]
        pairs, _ = market_data.get_top_volume_usdt_symbols(
            lambda *args, **kwargs: payload,
            {"XBT/USD": {}},
            lambda _msg: None,
            top_n=10,
        )
        self.assertIn("XBT/USD", pairs)

    def test_get_top_volume_usdt_symbols_prefers_more_liquid_mapped_pair(self):
        payload = [{"id": "btc-bitcoin", "symbol": "btc", "market_cap": 10}]
        pairs, _ = market_data.get_top_volume_usdt_symbols(
            lambda *args, **kwargs: payload,
            {"BTC/USD": {}, "BTC/USDT": {}},
            lambda _msg: None,
            top_n=10,
            exchange_tickers_fetcher=lambda: {
                "BTC/USD": {"quoteVolume": 9_000_000},
                "BTC/USDT": {"quoteVolume": 4_000_000},
            },
        )
        self.assertEqual(pairs, ["BTC/USD"])

    def test_get_top_volume_usdt_symbols_skips_ambiguous_mapped_pair_without_ranked_tickers(self):
        debug_msgs: list[str] = []
        payload = [{"id": "btc-bitcoin", "symbol": "btc", "market_cap": 10}]
        pairs, raw = market_data.get_top_volume_usdt_symbols(
            lambda *args, **kwargs: payload,
            {"BTC/USD": {}, "BTC/USDT": {}},
            debug_msgs.append,
            top_n=10,
            exchange_tickers_fetcher=lambda: {},
        )
        self.assertEqual(pairs, [])
        self.assertEqual(raw, payload)
        self.assertTrue(any("Strict pair ranking skipped BTC" in msg for msg in debug_msgs))
        self.assertTrue(any("strict pair ranking could not resolve exchange feeds" in msg.lower() for msg in debug_msgs))

    def test_get_top_volume_usdt_symbols_skips_ambiguous_provider_tickers_with_multiple_ids(self):
        debug_msgs: list[str] = []
        payload = [
            {"id": "alpha-foo", "symbol": "foo", "market_cap": 100},
            {"id": "beta-foo", "symbol": "foo", "market_cap": 200},
            {"id": "bar-coin", "symbol": "bar", "market_cap": 50},
        ]
        pairs, raw = market_data.get_top_volume_usdt_symbols(
            lambda *args, **kwargs: payload,
            {"FOO/USDT": {}, "BAR/USDT": {}},
            debug_msgs.append,
            top_n=10,
        )
        self.assertEqual(pairs, ["BAR/USDT"])
        self.assertEqual(raw, [{"id": "bar-coin", "symbol": "bar", "market_cap": 50}])
        self.assertTrue(any("duplicate ticker ambiguity" in msg for msg in debug_msgs))

    def test_get_top_volume_usdt_symbols_skips_exchange_fallback_without_ranked_tickers(self):
        payload = [{"symbol": "unknown"}]
        pairs, raw = market_data.get_top_volume_usdt_symbols(
            lambda *args, **kwargs: payload,
            {"ADA/USD": {}, "SOL/USDT": {}, "ETH/BTC": {}},
            lambda _msg: None,
            top_n=10,
        )
        self.assertEqual(pairs, [])
        self.assertEqual(raw, [])

    def test_get_top_volume_usdt_symbols_ranks_exchange_fallback_by_ticker_volume(self):
        payload = [{"symbol": "unknown"}]
        pairs, raw = market_data.get_top_volume_usdt_symbols(
            lambda *args, **kwargs: payload,
            {
                "ADA/USD": {},
                "SOL/USDT": {},
                "BTC/USD": {},
                "XBT/USDT": {},
            },
            lambda _msg: None,
            top_n=10,
            exchange_tickers_fetcher=lambda: {
                "ADA/USD": {"quoteVolume": 2_000_000},
                "SOL/USDT": {"quoteVolume": 8_000_000},
                "BTC/USD": {"quoteVolume": 5_000_000},
                "XBT/USDT": {"quoteVolume": 9_000_000},
            },
        )
        self.assertEqual(pairs[:3], ["XBT/USDT", "SOL/USDT", "ADA/USD"])
        self.assertEqual(raw, [])

    def test_get_top_volume_usdt_symbols_uses_coinpaprika_when_coingecko_empty(self):
        def _http_get_json(url, **kwargs):
            if "coingecko" in url:
                return []
            if "coinpaprika" in url:
                return [
                    {
                        "id": "btc-bitcoin",
                        "symbol": "BTC",
                        "quotes": {"USD": {"volume_24h": 1_000_000.0, "market_cap": 10}},
                    },
                    {
                        "id": "eth-ethereum",
                        "symbol": "ETH",
                        "quotes": {"USD": {"volume_24h": 900_000.0, "market_cap": 9}},
                    },
                ]
            return []

        pairs, raw = market_data.get_top_volume_usdt_symbols(
            _http_get_json,
            {"BTC/USDT": {}, "ETH/USD": {}},
            lambda _msg: None,
            top_n=10,
        )
        self.assertEqual(pairs, ["BTC/USDT", "ETH/USD"])
        self.assertIsInstance(raw, list)
        self.assertGreaterEqual(len(raw), 2)

    def test_get_top_volume_usdt_symbols_uses_exchange_fallback_when_both_providers_empty(self):
        def _http_get_json(url, **kwargs):
            return []

        pairs, raw = market_data.get_top_volume_usdt_symbols(
            _http_get_json,
            {"ADA/USD": {}, "SOL/USDT": {}, "ETH/BTC": {}},
            lambda _msg: None,
            top_n=10,
        )
        self.assertEqual(pairs, [])
        self.assertEqual(raw, [])

    def test_get_top_volume_usdt_symbols_ignores_zero_volume_exchange_snapshot(self):
        payload = [{"symbol": "unknown"}]
        pairs, raw = market_data.get_top_volume_usdt_symbols(
            lambda *args, **kwargs: payload,
            {"ADA/USD": {}, "SOL/USDT": {}, "BTC/USD": {}},
            lambda _msg: None,
            top_n=10,
            exchange_tickers_fetcher=lambda: {
                "ADA/USD": {"quoteVolume": 0},
                "SOL/USDT": {"quoteVolume": 0},
                "BTC/USD": {"quoteVolume": 0},
            },
        )
        self.assertEqual(pairs, [])
        self.assertEqual(raw, [])

    def test_fetch_market_cap_rows_for_symbols_uses_targeted_coingecko_ids(self):
        requested_urls: list[str] = []

        def _http_get_json(url, **kwargs):
            requested_urls.append(url)
            if "coins/markets" in url:
                self.assertEqual(kwargs["params"]["ids"], "dogecoin,bitcoin")
                return [
                    {"id": "dogecoin", "symbol": "doge", "market_cap": 11},
                    {"id": "bitcoin", "symbol": "btc", "market_cap": 22},
                ]
            return []

        rows = market_data.fetch_market_cap_rows_for_symbols(
            _http_get_json,
            lambda symbol: {"DOGE": "dogecoin", "BTC": "bitcoin"}.get(symbol),
            ["DOGE", "BTC/USDT"],
            lambda _msg: None,
        )
        self.assertEqual(
            rows,
            [
                {"id": "dogecoin", "symbol": "doge", "market_cap": 11},
                {"id": "bitcoin", "symbol": "btc", "market_cap": 22},
            ],
        )
        self.assertTrue(any("coins/markets" in url for url in requested_urls))

    def test_fetch_market_cap_rows_for_symbols_falls_back_to_coinpaprika_for_unresolved_symbols(self):
        def _http_get_json(url, **kwargs):
            if "coinpaprika" not in url:
                return []
            return [
                {
                    "id": "doge-dogecoin",
                    "symbol": "DOGE",
                    "quotes": {"USD": {"market_cap": 1234}},
                },
                {
                    "id": "btc-bitcoin",
                    "symbol": "BTC",
                    "quotes": {"USD": {"market_cap": 9999}},
                },
            ]

        rows = market_data.fetch_market_cap_rows_for_symbols(
            _http_get_json,
            lambda _symbol: None,
            ["DOGE"],
            lambda _msg: None,
        )
        self.assertEqual(
            rows,
            [{"id": "doge-dogecoin", "symbol": "doge", "market_cap": 1234.0}],
        )

    def test_fetch_market_cap_rows_for_symbols_skips_ambiguous_coinpaprika_tickers(self):
        debug_msgs: list[str] = []

        def _http_get_json(url, **kwargs):
            if "coinpaprika" not in url:
                return []
            return [
                {
                    "id": "alpha-foo",
                    "symbol": "FOO",
                    "quotes": {"USD": {"market_cap": 100}},
                },
                {
                    "id": "beta-foo",
                    "symbol": "FOO",
                    "quotes": {"USD": {"market_cap": 200}},
                },
            ]

        rows = market_data.fetch_market_cap_rows_for_symbols(
            _http_get_json,
            lambda _symbol: None,
            ["FOO"],
            debug_msgs.append,
        )
        self.assertEqual(rows, [])
        self.assertTrue(any("ambiguous ticker match" in msg.lower() for msg in debug_msgs))


if __name__ == "__main__":
    unittest.main()
