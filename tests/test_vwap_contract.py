import unittest

from core.signals import classify_vwap_bias


class VWAPContractTests(unittest.TestCase):
    def test_vwap_bias_above_below_near(self) -> None:
        self.assertEqual(classify_vwap_bias(101.0, 100.0, None), "🟢 Above")
        self.assertEqual(classify_vwap_bias(99.0, 100.0, None), "🔴 Below")
        self.assertEqual(classify_vwap_bias(100.03, 100.0, None), "→ Near VWAP")

    def test_vwap_bias_uses_atr_deadband(self) -> None:
        # base tol=0.08, atr tol=0.16 => effective tol=0.16
        self.assertEqual(classify_vwap_bias(100.10, 100.0, 2.0), "→ Near VWAP")
        self.assertEqual(classify_vwap_bias(100.20, 100.0, 2.0), "🟢 Above")
        self.assertEqual(classify_vwap_bias(99.80, 100.0, 2.0), "🔴 Below")

    def test_vwap_bias_invalid_inputs_return_empty(self) -> None:
        self.assertEqual(classify_vwap_bias(float("nan"), 100.0, None), "")
        self.assertEqual(classify_vwap_bias(100.0, float("nan"), None), "")


if __name__ == "__main__":
    unittest.main()
