from __future__ import annotations

import unittest

from core.metric_catalog import ai_stability_bucket, confidence_bucket, direction_from_prob


class MetricCatalogTests(unittest.TestCase):
    def test_direction_thresholds(self) -> None:
        self.assertEqual(direction_from_prob(0.70), "LONG")
        self.assertEqual(direction_from_prob(0.30), "SHORT")
        self.assertEqual(direction_from_prob(0.50), "NEUTRAL")

    def test_ai_stability_bucket(self) -> None:
        self.assertEqual(ai_stability_bucket(0.80), "Strong")
        self.assertEqual(ai_stability_bucket(0.65), "Medium")
        self.assertEqual(ai_stability_bucket(0.40), "Weak")

    def test_confidence_bucket(self) -> None:
        self.assertEqual(confidence_bucket(85), "Strong")
        self.assertEqual(confidence_bucket(70), "Good")
        self.assertEqual(confidence_bucket(55), "Mixed")
        self.assertEqual(confidence_bucket(35), "Weak")


if __name__ == "__main__":
    unittest.main()
