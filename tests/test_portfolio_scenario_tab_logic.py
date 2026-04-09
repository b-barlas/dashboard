import unittest

import pandas as pd

from tabs.portfolio_scenario_tab import _display_table, _format_money, _format_pct


class PortfolioScenarioTabLogicTests(unittest.TestCase):
    def test_format_money_handles_small_and_large_values(self) -> None:
        self.assertEqual(_format_money(1500.0), "$1,500")
        self.assertEqual(_format_money(12.3456), "$12.35")
        self.assertEqual(_format_money(0.123456), "$0.1235")

    def test_format_pct_handles_sign(self) -> None:
        self.assertEqual(_format_pct(3.25), "+3.25%")
        self.assertEqual(_format_pct(-1.5), "-1.50%")

    def test_display_table_builds_range_and_fit_columns(self) -> None:
        df = pd.DataFrame(
            [
                {
                    "Coin": "ETH",
                    "Current Price ($)": 2500.0,
                    "Current Value ($)": 800.0,
                    "Link Read": "Linked",
                    "Beta vs Anchor": 1.2,
                    "Fit": "Strong",
                    "Scenario Return (%)": 12.0,
                    "Projected Price ($)": 2800.0,
                    "Projected Value ($)": 896.0,
                    "Scenario Low ($)": 2600.0,
                    "Scenario High ($)": 3000.0,
                    "Matched Bars": 180,
                }
            ]
        )
        out = _display_table(df)
        self.assertEqual(out.iloc[0]["Fit"], "Strong fit")
        self.assertEqual(out.iloc[0]["Scenario Range ($)"], "$2,600 to $3,000")


if __name__ == "__main__":
    unittest.main()
