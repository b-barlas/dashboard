import unittest

try:
    import numpy as np
    import pandas as pd
    from core import advanced_analysis as aa

    DEPS_OK = True
except Exception:
    DEPS_OK = False


@unittest.skipUnless(DEPS_OK, "Missing dependencies for advanced analysis tests")
class AdvancedAnalysisHelpersTests(unittest.TestCase):
    def _sample_df(self, n: int = 120):
        idx = np.arange(n, dtype=float)
        close = 100 + np.sin(idx / 8.0) * 3 + idx * 0.05
        high = close + 1.5
        low = close - 1.5
        volume = np.full(n, 1000.0)
        return pd.DataFrame({"open": close, "high": high, "low": low, "close": close, "volume": volume})

    def test_fibonacci_contains_key_levels(self):
        levels = aa.calculate_fibonacci_levels(self._sample_df(), lookback=100)
        self.assertIn("61.8%", levels)
        self.assertIn("161.8%", levels)

    def test_volume_profile_contains_poc(self):
        profile = aa.calculate_volume_profile(self._sample_df(), num_bins=20)
        self.assertIn("poc_price", profile)
        self.assertIn("volumes", profile)


if __name__ == "__main__":
    unittest.main()
