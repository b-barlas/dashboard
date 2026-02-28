import unittest

try:
    import numpy as np
    import pandas as pd
    from core.scalping import get_scalping_entry_target

    DEPS_OK = True
except Exception:
    DEPS_OK = False


@unittest.skipUnless(DEPS_OK, "Missing dependencies for scalping tests")
class ScalpingContractTests(unittest.TestCase):
    def test_returns_default_for_short_dataframe(self):
        df = pd.DataFrame(
            {
                "open": [1.0, 2.0],
                "high": [1.1, 2.1],
                "low": [0.9, 1.9],
                "close": [1.0, 2.0],
                "volume": [10.0, 20.0],
            }
        )
        out = get_scalping_entry_target(
            df,
            70.0,
            "Bullish",
            "Bullish",
            "🟢 Above",
            True,
            sr_lookback_fn=lambda _tf=None: 30,
        )
        self.assertEqual(out, (None, 0.0, 0.0, 0.0, 0.0, ""))

    def test_level_ordering_for_long_plan(self):
        n = 220
        close = pd.Series(np.linspace(100, 130, n) + np.sin(np.linspace(0, 20, n)) * 0.3)
        df = pd.DataFrame(
            {
                "open": close - 0.1,
                "high": close + 0.4,
                "low": close - 0.4,
                "close": close,
                "volume": [1000.0] * n,
            }
        )
        direction, entry, target, stop, rr, _ = get_scalping_entry_target(
            df,
            70.0,
            "Bullish",
            "Bullish",
            "🟢 Above",
            True,
            strict_mode=False,
            sr_lookback_fn=lambda _tf=None: 30,
        )
        self.assertEqual(direction, "LONG")
        self.assertGreater(entry, 0.0)
        self.assertGreater(target, 0.0)
        self.assertGreater(stop, 0.0)
        self.assertGreater(rr, 0.0)
        self.assertLess(stop, entry)
        self.assertLess(entry, target)

    def test_level_ordering_for_short_plan(self):
        n = 220
        close = pd.Series(np.linspace(130, 100, n) + np.sin(np.linspace(0, 20, n)) * 0.3)
        df = pd.DataFrame(
            {
                "open": close + 0.1,
                "high": close + 0.4,
                "low": close - 0.4,
                "close": close,
                "volume": [1000.0] * n,
            }
        )
        direction, entry, target, stop, rr, _ = get_scalping_entry_target(
            df,
            30.0,
            "Bearish",
            "Bearish",
            "🔴 Below",
            True,
            strict_mode=False,
            sr_lookback_fn=lambda _tf=None: 30,
        )
        self.assertEqual(direction, "SHORT")
        self.assertGreater(entry, 0.0)
        self.assertGreater(target, 0.0)
        self.assertGreater(stop, 0.0)
        self.assertGreater(rr, 0.0)
        self.assertLess(target, entry)
        self.assertLess(entry, stop)


if __name__ == "__main__":
    unittest.main()
