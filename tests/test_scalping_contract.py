import unittest

try:
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


if __name__ == "__main__":
    unittest.main()
