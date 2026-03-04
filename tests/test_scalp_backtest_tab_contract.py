from __future__ import annotations

import unittest
from unittest.mock import Mock, patch

try:
    import pandas as pd
    from tabs.scalp_backtest_tab import render

    DEPS_OK = True
except Exception:
    DEPS_OK = False


class _DummySt:
    def __init__(self):
        self.session_state = {}

    def markdown(self, *args, **kwargs):
        return None

    def columns(self, n):
        if isinstance(n, (list, tuple)):
            return [self for _ in range(len(n))]
        return [self for _ in range(int(n))]

    def progress(self, *args, **kwargs):
        return self

    def empty(self):
        return None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def expander(self, *args, **kwargs):
        return self

    def text_input(self, *args, **kwargs):
        label = str(args[0]) if args else str(kwargs.get("label", ""))
        if "Custom Coins" in label:
            return ""
        return "BTC"

    def selectbox(self, *args, **kwargs):
        options = args[1] if len(args) > 1 else kwargs.get("options", [])
        return options[0] if options else "ALL Setup Confirmations"

    def slider(self, *args, **kwargs):
        return kwargs.get("value", args[3] if len(args) > 3 else 0)

    def caption(self, *args, **kwargs):
        return None

    def button(self, *args, **kwargs):
        label = str(args[0]) if args else str(kwargs.get("label", ""))
        if "Clear" in label:
            return False
        return True

    def rerun(self):
        return None

    def info(self, *args, **kwargs):
        return None

    def warning(self, *args, **kwargs):
        return None

    def error(self, *args, **kwargs):
        return None

    def success(self, *args, **kwargs):
        return None

    def plotly_chart(self, *args, **kwargs):
        return None

    def download_button(self, *args, **kwargs):
        return None


@unittest.skipUnless(DEPS_OK, "Missing deps for scalp backtest tab contract test")
class ScalpBacktestTabContractTests(unittest.TestCase):
    @patch("tabs.scalp_backtest_tab.live_or_snapshot")
    def test_render_calls_scalp_outcome_builder_with_expected_contract(self, live_or_snapshot_mock):
        st = _DummySt()
        ts = pd.date_range("2025-01-01", periods=220, freq="h")
        price = [100 + i * 0.1 for i in range(220)]
        df = pd.DataFrame(
            {
                "timestamp": ts,
                "open": price,
                "high": [p + 0.3 for p in price],
                "low": [p - 0.3 for p in price],
                "close": price,
                "volume": [1000.0] * 220,
            }
        )

        ev = {
            "Event Time": [ts[100], ts[120]],
            "Setup Confirm": ["TREND-led", "AI-led"],
            "Direction": ["Upside", "Downside"],
            "Strength": [61.0, 58.0],
            "AI Votes": ["2/3", "3/3"],
            "Event Price": [110.0, 112.0],
            "Target": [111.0, 110.5],
            "Stop": [109.0, 113.0],
            "R:R": [1.6, 1.7],
            "Outcome": ["TP", "SL"],
            "Hit Bar": [4, 3],
            "End Price (+10)": [112.0, 110.0],
            "Return @+10 (%)": [1.8, -1.2],
            "Favorable Excursion (%)": [3.5, 2.9],
            "Adverse Excursion (%)": [1.1, 0.8],
            "AI Direction": ["Upside", "Downside"],
            "Tech vs AI Alignment": ["HIGH", "MEDIUM"],
        }
        for i in range(1, 11):
            ev[f"Price +{i}"] = [110.0 + i * 0.1, 112.0 - i * 0.1]
            ev[f"Directional Return +{i} (%)"] = [0.2 * i, 0.15 * i]
        events_df = pd.DataFrame(ev)

        build_scalp_outcome_study = Mock(return_value=events_df)
        summarize_scalp_outcome_study = Mock(
            return_value={
                "occurrences": 2.0,
                "tp_rate": 50.0,
                "sl_rate": 50.0,
                "timeout_rate": 0.0,
                "avg_outcome": 0.3,
                "median_outcome": 0.3,
                "avg_favorable_exc": 3.2,
                "avg_adverse_exc": 0.95,
            }
        )
        live_or_snapshot_mock.return_value = (df, False, None)

        ctx = {
            "st": st,
            "ACCENT": "#00E5FF",
            "TEXT_MUTED": "#8CA1B6",
            "POSITIVE": "#06D6A0",
            "WARNING": "#FFB000",
            "get_top_volume_usdt_symbols": lambda top_n=40, vs_currency="usd": (["BTC/USDT", "ETH/USDT"], []),
            "fetch_ohlcv": lambda *_a, **_k: df,
            "analyse": object(),
            "ml_ensemble_predict": object(),
            "signal_plain": object(),
            "direction_key": object(),
            "_calc_conviction": object(),
            "get_scalping_entry_target": object(),
            "scalp_quality_gate": object(),
            "_sr_lookback": object(),
        }

        with patch("tabs.scalp_backtest_tab.build_scalp_outcome_study", build_scalp_outcome_study), patch(
            "tabs.scalp_backtest_tab.summarize_scalp_outcome_study", summarize_scalp_outcome_study
        ):
            render(ctx)

        self.assertTrue(build_scalp_outcome_study.called)
        self.assertEqual(build_scalp_outcome_study.call_count, 2)
        kwargs = build_scalp_outcome_study.call_args_list[0].kwargs
        self.assertEqual(kwargs["timeframe"], "5m")
        self.assertEqual(kwargs["forward_bars"], 10)
        self.assertIn("df", kwargs)
        self.assertIn("analyzer", kwargs)
        self.assertIn("ml_predictor", kwargs)
        self.assertIn("conviction_fn", kwargs)
        self.assertIn("signal_plain_fn", kwargs)
        self.assertIn("direction_key_fn", kwargs)
        self.assertIn("get_scalping_entry_target_fn", kwargs)
        self.assertIn("scalp_quality_gate_fn", kwargs)
        self.assertIn("sr_lookback_fn", kwargs)


if __name__ == "__main__":
    unittest.main()
