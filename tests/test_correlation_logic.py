import unittest

from tabs.correlation_tab import (
    _pair_read,
    _pair_regime,
    _parse_custom_coins,
)


class CorrelationLogicTests(unittest.TestCase):
    def test_pair_read_bands(self):
        self.assertEqual(_pair_read(0.85), "Tight")
        self.assertEqual(_pair_read(0.60), "Linked")
        self.assertEqual(_pair_read(0.25), "Mild")
        self.assertEqual(_pair_read(0.00), "Neutral")
        self.assertEqual(_pair_read(-0.30), "Hedge")
        self.assertEqual(_pair_read(-0.70), "Inverse")

    def test_pair_regime_bands(self):
        self.assertEqual(_pair_regime(0.85), "High Co-Move")
        self.assertEqual(_pair_regime(-0.40), "Inverse / Defensive")
        self.assertEqual(_pair_regime(0.10), "Low Co-Move")
        self.assertEqual(_pair_regime(0.45), "Medium Co-Move")

    def test_custom_coin_parser_dedupes_and_limits(self):
        normalize = lambda coin: f"{coin.upper()}/USDT"
        parsed = _parse_custom_coins(
            "btc, eth, btc, sol, link, doge, tao, fet, render, xrp, ada, bnb",
            normalize,
        )
        self.assertEqual(
            parsed,
            [
                "BTC/USDT",
                "ETH/USDT",
                "SOL/USDT",
                "LINK/USDT",
                "DOGE/USDT",
                "TAO/USDT",
                "FET/USDT",
                "RENDER/USDT",
                "XRP/USDT",
                "ADA/USDT",
            ],
        )


if __name__ == "__main__":
    unittest.main()
