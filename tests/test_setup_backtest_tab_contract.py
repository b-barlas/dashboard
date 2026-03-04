from __future__ import annotations

import unittest
from unittest.mock import Mock, patch

try:
    import pandas as pd
    from tabs.setup_backtest_tab import render

    DEPS_OK = True
except Exception:
    DEPS_OK = False


class _DummySt:
    def __init__(self):
        self.session_state = {}

    def markdown(self, *args, **kwargs):
        return None

    def columns(self, n):
        return [self for _ in range(int(n))]

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def expander(self, *args, **kwargs):
        return self

    def text_input(self, *args, **kwargs):
        return "BTC"

    def selectbox(self, *args, **kwargs):
        options = args[1] if len(args) > 1 else kwargs.get("options", [])
        return options[0] if options else "ALL Setup Confirmations"

    def slider(self, *args, **kwargs):
        return kwargs.get("value", args[3] if len(args) > 3 else 0)

    def caption(self, *args, **kwargs):
        return None

    def button(self, *args, **kwargs):
        return True

    def info(self, *args, **kwargs):
        return None

    def warning(self, *args, **kwargs):
        return None

    def error(self, *args, **kwargs):
        return None

    def success(self, *args, **kwargs):
        return None

    def metric(self, *args, **kwargs):
        return None

    def plotly_chart(self, *args, **kwargs):
        return None

    def dataframe(self, *args, **kwargs):
        return None

    def download_button(self, *args, **kwargs):
        return None


@unittest.skipUnless(DEPS_OK, "Missing deps for setup backtest tab contract test")
class SetupBacktestTabContractTests(unittest.TestCase):
    @patch("tabs.setup_backtest_tab.live_or_snapshot")
    def test_render_calls_setup_outcome_builder_with_expected_contract(self, live_or_snapshot_mock):
        st = _DummySt()
        ts = pd.date_range("2025-01-01", periods=200, freq="h")
        price = [100 + i * 0.1 for i in range(200)]
        df = pd.DataFrame(
            {
                "timestamp": ts,
                "open": price,
                "high": [p + 0.3 for p in price],
                "low": [p - 0.3 for p in price],
                "close": price,
                "volume": [1000.0] * 200,
            }
        )

        ev = {
            "Event Time": [ts[100], ts[120]],
            "Setup Confirm": ["TREND-led", "AI-led"],
            "Direction": ["Upside", "Downside"],
            "Strength": [61.0, 58.0],
            "AI Votes": ["2/3", "3/3"],
            "Event Price": [110.0, 112.0],
            "End Price (+10)": [112.0, 110.0],
            "Return @+10 (%)": [1.8, 1.2],
            "Favorable Excursion (%)": [3.5, 2.9],
            "Adverse Excursion (%)": [1.1, 0.8],
        }
        for i in range(1, 11):
            ev[f"Price +{i}"] = [110.0 + i * 0.1, 112.0 - i * 0.1]
            ev[f"Directional Return +{i} (%)"] = [0.2 * i, 0.15 * i]
        events_df = pd.DataFrame(ev)

        build_setup_outcome_study = Mock(return_value=events_df)
        summarize_setup_outcome_study = Mock(
            return_value={
                "occurrences": 2.0,
                "favorable_rate": 100.0,
                "median_dir_return": 1.5,
                "avg_favorable_exc": 3.2,
                "avg_adverse_exc": 0.95,
            }
        )
        summarize_setup_outcome_by_class = Mock(
            return_value=pd.DataFrame(
                {
                    "Setup Confirm": ["TREND-led", "AI-led"],
                    "Occurrences": [1, 1],
                    "FavorableRate": [100.0, 100.0],
                    "MedianDirectionalReturn": [1.8, 1.2],
                    "AvgDirectionalReturn": [1.8, 1.2],
                    "AvgFavorableExcursion": [3.5, 2.9],
                    "AvgAdverseExcursion": [1.1, 0.8],
                }
            )
        )

        live_or_snapshot_mock.return_value = (df, False, None)

        ctx = {
            "st": st,
            "ACCENT": "#00E5FF",
            "TEXT_MUTED": "#8CA1B6",
            "POSITIVE": "#06D6A0",
            "WARNING": "#FFB000",
            "_normalize_coin_input": lambda x: "BTC/USDT",
            "_validate_coin_symbol": lambda _x: None,
            "fetch_ohlcv": lambda *_a, **_k: df,
            "analyse": object(),
            "ml_ensemble_predict": object(),
            "signal_plain": object(),
            "direction_key": object(),
            "_calc_conviction": object(),
        }

        with patch("tabs.setup_backtest_tab.build_setup_outcome_study", build_setup_outcome_study), patch(
            "tabs.setup_backtest_tab.summarize_setup_outcome_study", summarize_setup_outcome_study
        ), patch("tabs.setup_backtest_tab.summarize_setup_outcome_by_class", summarize_setup_outcome_by_class):
            render(ctx)

        self.assertTrue(build_setup_outcome_study.called)
        kwargs = build_setup_outcome_study.call_args.kwargs
        self.assertEqual(kwargs["setup_filter"], "ALL")
        self.assertEqual(kwargs["forward_bars"], 10)
        self.assertIn("df", kwargs)
        self.assertIn("analyzer", kwargs)
        self.assertIn("ml_predictor", kwargs)
        self.assertIn("conviction_fn", kwargs)
        self.assertIn("signal_plain_fn", kwargs)
        self.assertIn("direction_key_fn", kwargs)


if __name__ == "__main__":
    unittest.main()
