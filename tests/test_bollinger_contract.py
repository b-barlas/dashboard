import unittest

from core.signals import classify_bollinger_bias


class BollingerContractTests(unittest.TestCase):
    def test_outer_extremes(self) -> None:
        # width=20, 1% buffer=0.2
        self.assertEqual(classify_bollinger_bias(120.3, 90.0, 110.0), "🔴 Overbought")
        self.assertEqual(classify_bollinger_bias(89.7, 90.0, 110.0), "🟢 Oversold")

    def test_inner_band_zones(self) -> None:
        # band_pos=0.90 -> Near Top
        self.assertEqual(classify_bollinger_bias(108.0, 90.0, 110.0), "→ Near Top")
        # band_pos=0.10 -> Near Bottom
        self.assertEqual(classify_bollinger_bias(92.0, 90.0, 110.0), "→ Near Bottom")
        # middle zone -> Neutral
        self.assertEqual(classify_bollinger_bias(100.0, 90.0, 110.0), "→ Neutral")

    def test_invalid_inputs_return_empty(self) -> None:
        self.assertEqual(classify_bollinger_bias(float("nan"), 90.0, 110.0), "")
        self.assertEqual(classify_bollinger_bias(100.0, float("nan"), 110.0), "")
        self.assertEqual(classify_bollinger_bias(100.0, 110.0, 110.0), "")


if __name__ == "__main__":
    unittest.main()
