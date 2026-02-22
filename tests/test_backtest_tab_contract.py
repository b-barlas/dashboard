from __future__ import annotations

import unittest
from unittest.mock import Mock

try:
    import pandas as pd
    from tabs.backtest_tab import render

    DEPS_OK = True
except Exception:
    DEPS_OK = False


class _DummySt:
    def __init__(self):
        self.session_state = {}

    def markdown(self, *args, **kwargs):
        return None

    def columns(self, n):
        return [self for _ in range(n)]

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def text_input(self, *args, **kwargs):
        return "BTC"

    def selectbox(self, *args, **kwargs):
        options = args[1] if len(args) > 1 else kwargs.get("options", [])
        return options[0] if options else "1h"

    def slider(self, *args, **kwargs):
        return kwargs.get("value", args[3] if len(args) > 3 else 0)

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


@unittest.skipUnless(DEPS_OK, "Missing deps for backtest tab contract test")
class BacktestTabContractTests(unittest.TestCase):
    def test_render_calls_run_backtest_with_named_args(self):
        st = _DummySt()

        ts = pd.date_range("2025-01-01", periods=120, freq="h")
        df = pd.DataFrame(
            {
                "timestamp": ts,
                "open": [100 + i * 0.1 for i in range(120)],
                "high": [100 + i * 0.1 + 0.3 for i in range(120)],
                "low": [100 + i * 0.1 - 0.3 for i in range(120)],
                "close": [100 + i * 0.1 for i in range(120)],
                "volume": [1000.0] * 120,
            }
        )

        def _run_backtest(data, **kwargs):
            out = pd.DataFrame(
                {
                    "Date": [data["timestamp"].iloc[60]],
                    "Confidence": [70.0],
                    "Signal": ["LONG"],
                    "Entry": [float(data["close"].iloc[60])],
                    "Exit": [float(data["close"].iloc[65])],
                    "PnL (%)": [1.0],
                    "Equity": [10100.0],
                }
            )
            return out, "<html/>"

        run_bt = Mock(side_effect=_run_backtest)

        ctx = {
            "ACCENT": "#0ff",
            "TEXT_MUTED": "#999",
            "POSITIVE": "#0f0",
            "WARNING": "#ff0",
            "_normalize_coin_input": lambda x: "BTC/USDT",
            "_validate_coin_symbol": lambda _x: None,
            "fetch_ohlcv": lambda *_a, **_k: df,
            "run_backtest": run_bt,
        }

        render(ctx | {"st": st})
        self.assertTrue(run_bt.called)
        kwargs = run_bt.call_args.kwargs
        self.assertIn("threshold", kwargs)
        self.assertIn("exit_after", kwargs)
        self.assertIn("commission", kwargs)
        self.assertIn("slippage", kwargs)


if __name__ == "__main__":
    unittest.main()
