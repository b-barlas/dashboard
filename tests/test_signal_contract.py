import unittest

from core.signal_contract import bias_confidence_from_bias, direction_from_bias


class SignalContractTests(unittest.TestCase):
    def test_direction_from_bias(self) -> None:
        self.assertEqual(direction_from_bias(80), "LONG")
        self.assertEqual(direction_from_bias(20), "SHORT")
        self.assertEqual(direction_from_bias(50), "NEUTRAL")

    def test_bias_confidence_from_bias_is_symmetric(self) -> None:
        self.assertEqual(bias_confidence_from_bias(80), bias_confidence_from_bias(20))
        self.assertEqual(bias_confidence_from_bias(50), 0.0)
        self.assertEqual(bias_confidence_from_bias(100), 100.0)


if __name__ == "__main__":
    unittest.main()
