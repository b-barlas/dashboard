import unittest

from core.scalping import scalp_quality_gate


class ScalpGateContractTests(unittest.TestCase):
    def test_pass_when_all_quality_gates_pass(self):
        ok, reason = scalp_quality_gate(
            scalp_direction="LONG",
            signal_direction="LONG",
            rr_ratio=1.8,
            adx_val=24.0,
            strength=61.0,
            conviction_label="MEDIUM",
            entry=100.0,
            stop=98.5,
            target=103.0,
        )
        self.assertTrue(ok)
        self.assertEqual(reason, "PASS")

    def test_fail_on_direction_mismatch(self):
        ok, reason = scalp_quality_gate(
            scalp_direction="SHORT",
            signal_direction="LONG",
            rr_ratio=1.9,
            adx_val=28.0,
            strength=72.0,
            conviction_label="HIGH",
            entry=100.0,
            stop=102.0,
            target=96.0,
        )
        self.assertFalse(ok)
        self.assertEqual(reason, "DIRECTION_MISMATCH")

    def test_fail_when_signal_direction_is_not_tradeable(self):
        ok, reason = scalp_quality_gate(
            scalp_direction="LONG",
            signal_direction="NEUTRAL",
            rr_ratio=2.0,
            adx_val=30.0,
            strength=75.0,
            conviction_label="HIGH",
            entry=100.0,
            stop=98.0,
            target=104.0,
        )
        self.assertFalse(ok)
        self.assertEqual(reason, "SIGNAL_DIRECTION_NEUTRAL")

    def test_fail_on_conflict(self):
        ok, reason = scalp_quality_gate(
            scalp_direction="LONG",
            signal_direction="LONG",
            rr_ratio=1.7,
            adx_val=22.0,
            strength=59.0,
            conviction_label="CONFLICT",
            entry=100.0,
            stop=98.2,
            target=103.2,
        )
        self.assertFalse(ok)
        self.assertEqual(reason, "CONFLICT")

    def test_fail_on_low_rr(self):
        ok, reason = scalp_quality_gate(
            scalp_direction="LONG",
            signal_direction="LONG",
            rr_ratio=1.2,
            adx_val=25.0,
            strength=66.0,
            conviction_label="HIGH",
            entry=100.0,
            stop=99.0,
            target=101.2,
        )
        self.assertFalse(ok)
        self.assertEqual(reason, "RR_TOO_LOW")

    def test_fail_on_low_adx(self):
        ok, reason = scalp_quality_gate(
            scalp_direction="SHORT",
            signal_direction="SHORT",
            rr_ratio=1.6,
            adx_val=18.0,
            strength=63.0,
            conviction_label="MEDIUM",
            entry=100.0,
            stop=101.0,
            target=97.8,
        )
        self.assertFalse(ok)
        self.assertEqual(reason, "ADX_TOO_LOW")

    def test_fail_on_low_strength(self):
        ok, reason = scalp_quality_gate(
            scalp_direction="LONG",
            signal_direction="LONG",
            rr_ratio=1.7,
            adx_val=21.0,
            strength=45.0,
            conviction_label="MEDIUM",
            entry=100.0,
            stop=99.0,
            target=102.0,
        )
        self.assertFalse(ok)
        self.assertEqual(reason, "STRENGTH_TOO_LOW")

    def test_fail_on_invalid_levels(self):
        ok, reason = scalp_quality_gate(
            scalp_direction="LONG",
            signal_direction="LONG",
            rr_ratio=2.0,
            adx_val=30.0,
            strength=70.0,
            conviction_label="HIGH",
            entry=0.0,
            stop=99.0,
            target=104.0,
        )
        self.assertFalse(ok)
        self.assertEqual(reason, "INVALID_LEVELS")


if __name__ == "__main__":
    unittest.main()
