import unittest

import numpy as np
import pandas as pd

from core.portfolio_scenario import (
    build_portfolio_scenario,
    estimate_anchor_horizon,
    regression_relationship,
    returns_series,
    sanitize_holdings_rows,
)


def _frame_from_returns(returns, start_price=100.0):
    prices = [start_price]
    for ret in returns:
        prices.append(prices[-1] * (1.0 + ret))
    prices = prices[1:]
    idx = pd.date_range("2026-01-01", periods=len(prices), freq="h", tz="UTC")
    return pd.DataFrame(
        {
            "timestamp": (idx.view("int64") // 10**6),
            "close": prices,
        }
    )


class PortfolioScenarioLogicTests(unittest.TestCase):
    def test_returns_series_supports_forward_horizon(self):
        base = np.array([0.01] * 80)
        df = _frame_from_returns(base, start_price=100.0)
        series = returns_series(df, "BTC/USDT", horizon_bars=3)

        self.assertIsNotNone(series)
        self.assertGreater(len(series), 40)
        self.assertAlmostEqual(float(series.iloc[0]), (1.01**3) - 1.0, places=6)

    def test_estimate_anchor_horizon_expands_with_target_distance(self):
        base = np.array([0.01, -0.004, 0.008, -0.003] * 40)
        anchor_df = _frame_from_returns(base, start_price=100.0)

        short_info = estimate_anchor_horizon(anchor_df, 0.02)
        long_info = estimate_anchor_horizon(anchor_df, 0.15)

        self.assertGreater(short_info["typical_bar_move"], 0.0)
        self.assertGreaterEqual(short_info["horizon_bars"], 1)
        self.assertGreater(long_info["horizon_bars"], short_info["horizon_bars"])

    def test_sanitize_holdings_rows_dedupes_and_limits(self):
        normalize = lambda coin: f"{coin.upper()}/USDT"
        raw = pd.DataFrame(
            {
                "Coin": ["btc", "eth", "btc", "sol", "xrp", "ada", "doge", "link", "fet", "tao", "render", "inj"],
                "Current Value ($)": [1000] * 12,
            }
        )
        rows = sanitize_holdings_rows(raw, normalize)
        self.assertEqual(len(rows), 10)
        self.assertEqual(rows[0]["symbol"], "BTC/USDT")
        self.assertEqual(rows[1]["symbol"], "ETH/USDT")
        self.assertEqual(rows[-1]["symbol"], "RENDER/USDT")

    def test_sanitize_holdings_rows_returns_meta_for_dropped_rows(self):
        normalize = lambda coin: f"{coin.upper()}/USDT"
        raw = pd.DataFrame(
            {
                "Coin": ["btc", "eth", "btc", "sol", "doge"],
                "Current Value ($)": [1000, 0, 900, "bad", 500],
            }
        )
        rows, meta = sanitize_holdings_rows(raw, normalize, max_items=2, return_meta=True)
        self.assertEqual(len(rows), 2)
        self.assertEqual(meta["coin_rows"], 5)
        self.assertEqual(meta["kept_rows"], 2)
        self.assertEqual(meta["duplicate_rows"], 1)
        self.assertEqual(meta["invalid_value_rows"], 2)
        self.assertEqual(meta["truncated_rows"], 0)
        self.assertIn("BTC", meta["duplicate_symbols"])

    def test_regression_relationship_captures_positive_and_negative_beta(self):
        anchor = pd.Series(np.linspace(-0.01, 0.012, 80), index=pd.date_range("2026-01-01", periods=80, freq="h", tz="UTC"))
        follower = 0.001 + (1.5 * anchor)
        hedge = -0.0005 - (0.6 * anchor)

        pos_model = regression_relationship(anchor, follower)
        neg_model = regression_relationship(anchor, hedge)

        self.assertIsNotNone(pos_model)
        self.assertIsNotNone(neg_model)
        self.assertGreater(pos_model["beta"], 1.4)
        self.assertLess(neg_model["beta"], -0.5)

    def test_build_portfolio_scenario_projects_anchor_and_follower(self):
        base = np.array([0.006, -0.003, 0.004, 0.002, -0.001, 0.005, -0.002, 0.003] * 10)
        anchor_df = _frame_from_returns(base, start_price=100.0)
        follower_df = _frame_from_returns(0.0008 + (1.3 * base), start_price=50.0)
        hedge_df = _frame_from_returns(-0.0004 - (0.5 * base), start_price=30.0)

        holdings = [
            {"coin": "BTC", "symbol": "BTC/USDT", "current_value": 1000.0},
            {"coin": "ETH", "symbol": "ETH/USDT", "current_value": 800.0},
            {"coin": "ATOM", "symbol": "ATOM/USDT", "current_value": 500.0},
        ]
        anchor_price = float(anchor_df["close"].iloc[-1])
        result = build_portfolio_scenario(
            holdings,
            "BTC/USDT",
            anchor_price * 1.10,
            {
                "BTC/USDT": anchor_df,
                "ETH/USDT": follower_df,
                "ATOM/USDT": hedge_df,
            },
        )

        rows = result["rows"].set_index("Coin")
        self.assertAlmostEqual(rows.loc["BTC", "Scenario Return (%)"], 10.0, places=3)
        self.assertGreater(rows.loc["ETH", "Scenario Return (%)"], 10.0)
        self.assertLess(rows.loc["ATOM", "Scenario Return (%)"], 0.0)
        self.assertGreater(result["coverage_pct"], 99.0)
        self.assertGreaterEqual(result["horizon_bars"], 1)
        self.assertGreater(result["typical_bar_move_pct"], 0.0)

    def test_build_portfolio_scenario_caps_impossible_negative_return(self):
        base = np.array([0.01, -0.005, 0.008, -0.004] * 30)
        anchor_df = _frame_from_returns(base, start_price=100.0)
        crash_df = _frame_from_returns(0.001 + (2.5 * base), start_price=50.0)

        holdings = [
            {"coin": "BTC", "symbol": "BTC/USDT", "current_value": 1000.0},
            {"coin": "ALT", "symbol": "ALT/USDT", "current_value": 1000.0},
        ]
        result = build_portfolio_scenario(
            holdings,
            "BTC/USDT",
            10.0,
            {
                "BTC/USDT": anchor_df,
                "ALT/USDT": crash_df,
            },
        )

        rows = result["rows"].set_index("Coin")
        self.assertGreaterEqual(rows.loc["ALT", "Scenario Return (%)"], -95.0)
        self.assertGreaterEqual(rows.loc["ALT", "Projected Price ($)"], 0.0)
        self.assertGreaterEqual(result["horizon_bars"], 1)

    def test_build_portfolio_scenario_exposes_horizon_cap(self):
        base = np.array([0.004, -0.002, 0.003, -0.001] * 80)
        anchor_df = _frame_from_returns(base, start_price=100.0)
        follower_df = _frame_from_returns(0.0005 + (1.1 * base), start_price=50.0)
        holdings = [
            {"coin": "BTC", "symbol": "BTC/USDT", "current_value": 1000.0},
            {"coin": "ETH", "symbol": "ETH/USDT", "current_value": 800.0},
        ]
        result = build_portfolio_scenario(
            holdings,
            "BTC/USDT",
            float(anchor_df["close"].iloc[-1]) * 3.5,
            {"BTC/USDT": anchor_df, "ETH/USDT": follower_df},
        )
        self.assertTrue(result["horizon_capped"])
        self.assertGreater(result["raw_horizon_bars"], result["horizon_bars"])
        self.assertIn(result["horizon_cap_reason"], {"stability_cap", "history_limit"})


if __name__ == "__main__":
    unittest.main()
