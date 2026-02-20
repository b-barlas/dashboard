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


if __name__ == "__main__":
    unittest.main()
