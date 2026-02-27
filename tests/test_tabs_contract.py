import importlib
import inspect
import unittest

try:
    import streamlit  # noqa: F401
    import pandas  # noqa: F401
    import plotly  # noqa: F401
    import ta  # noqa: F401
    DEPS_OK = True
except Exception:
    DEPS_OK = False

TAB_MODULES = [
    "tabs.backtest_tab",
    "tabs.correlation_tab",
    "tabs.ensemble_ml_tab",
    "tabs.fibonacci_tab",
    "tabs.guide_tab",
    "tabs.heatmap_tab",
    "tabs.market_tab",
    "tabs.ml_tab",
    "tabs.monte_carlo_tab",
    "tabs.multitf_tab",
    "tabs.position_tab",
    "tabs.risk_tab",
    "tabs.screener_tab",
    "tabs.sessions_tab",
    "tabs.spot_tab",
    "tabs.tools_tab",
    "tabs.whale_tab",
]


@unittest.skipUnless(DEPS_OK, "Missing UI dependencies for tab contract tests")
class TabsContractTests(unittest.TestCase):
    def test_tabs_expose_render_ctx(self):
        for module_name in TAB_MODULES:
            with self.subTest(module=module_name):
                module = importlib.import_module(module_name)

                self.assertTrue(hasattr(module, "render"), f"{module_name} missing render")
                render_fn = getattr(module, "render")
                self.assertTrue(callable(render_fn), f"{module_name}.render is not callable")

                sig = inspect.signature(render_fn)
                self.assertEqual(len(sig.parameters), 1, f"{module_name}.render must accept ctx")


if __name__ == "__main__":
    unittest.main()
