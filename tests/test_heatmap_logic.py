import unittest

from tabs.heatmap_tab import _build_top_movers_tables, _prepare_heatmap_frames


class HeatmapLogicTests(unittest.TestCase):
    def test_prepare_heatmap_frames_keeps_broader_sample_and_trims_map_slice(self) -> None:
        rows = []
        for i in range(120):
            rows.append(
                {
                    "Symbol": f"C{i}",
                    "Name": f"Coin {i}",
                    "Market Cap": float(10_000 - i),
                    "Change 24h (%)": float(i % 7),
                    "Price": 1.0 + i,
                    "Sector": "Crypto",
                    "Stablecoin": False,
                    "Provider": "CoinGecko",
                }
            )

        df_all, df_map = _prepare_heatmap_frames(rows, exclude_stablecoins=False, map_limit=100)

        self.assertEqual(len(df_all), 120)
        self.assertEqual(len(df_map), 100)
        self.assertEqual(df_map.iloc[0]["Symbol"], "C0")
        self.assertEqual(df_map.iloc[-1]["Symbol"], "C99")

    def test_prepare_heatmap_frames_excludes_stablecoins_from_sample_and_map(self) -> None:
        rows = [
            {
                "Symbol": "USDT",
                "Name": "Tether",
                "Market Cap": 100.0,
                "Change 24h (%)": 0.01,
                "Price": 1.0,
                "Sector": "Crypto",
                "Stablecoin": True,
                "Provider": "CoinGecko",
            },
            {
                "Symbol": "BTC",
                "Name": "Bitcoin",
                "Market Cap": 90.0,
                "Change 24h (%)": 2.5,
                "Price": 90_000.0,
                "Sector": "Crypto",
                "Stablecoin": False,
                "Provider": "CoinGecko",
            },
        ]

        df_all, df_map = _prepare_heatmap_frames(rows, exclude_stablecoins=True, map_limit=100)

        self.assertEqual(df_all["Symbol"].tolist(), ["BTC"])
        self.assertEqual(df_map["Symbol"].tolist(), ["BTC"])

    def test_build_top_movers_tables_uses_broader_sample_not_map_slice(self) -> None:
        rows = []
        for i in range(110):
            rows.append(
                {
                    "Symbol": f"C{i}",
                    "Name": f"Coin {i}",
                    "Market Cap": float(10_000 - i),
                    "Change 24h (%)": -1.0,
                    "Price": 1.0,
                    "Sector": "Crypto",
                    "Stablecoin": False,
                    "Provider": "CoinGecko",
                }
            )
        rows[105]["Change 24h (%)"] = 25.0
        rows[108]["Change 24h (%)"] = -18.0

        df_all, _df_map = _prepare_heatmap_frames(rows, exclude_stablecoins=False, map_limit=100)
        top_g, top_l = _build_top_movers_tables(df_all, top_n=5)

        self.assertEqual(top_g.iloc[0]["Symbol"], "C105")
        self.assertEqual(top_l.iloc[0]["Symbol"], "C108")


if __name__ == "__main__":
    unittest.main()
