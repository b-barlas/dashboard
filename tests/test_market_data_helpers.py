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

    def test_get_top_volume_usdt_symbols_falls_back_to_exchange_pairs(self):
        payload = [{"symbol": "unknown"}]
        pairs, _ = market_data.get_top_volume_usdt_symbols(
            lambda *args, **kwargs: payload,
            {"ADA/USD": {}, "SOL/USDT": {}, "ETH/BTC": {}},
            lambda _msg: None,
            top_n=10,
        )
        self.assertEqual(pairs[:2], ["ADA/USD", "SOL/USDT"])


if __name__ == "__main__":
    unittest.main()
