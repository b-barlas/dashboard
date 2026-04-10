import unittest

from core.timeframe_anchors import choose_anchor_context, resolve_anchor_plan, resolve_anchor_plan_candidates


class TimeframeAnchorTests(unittest.TestCase):
    def test_intraday_frames_use_4h_1h_anchor_ladder(self) -> None:
        for timeframe in ("1m", "3m", "5m", "15m"):
            plan = resolve_anchor_plan(timeframe)
            self.assertEqual(plan.lead_timeframe, "4h")
            self.assertEqual(plan.confirm_timeframe, "1h")

    def test_one_hour_uses_daily_led_anchor_ladder(self) -> None:
        plan = resolve_anchor_plan("1h")
        self.assertEqual(plan.lead_timeframe, "1d")
        self.assertEqual(plan.confirm_timeframe, "4h")

    def test_slower_frames_prefer_weekly_led_ladder(self) -> None:
        for timeframe in ("4h", "1d"):
            plan = resolve_anchor_plan(timeframe)
            self.assertEqual(plan.lead_timeframe, "1w")
            self.assertEqual(plan.confirm_timeframe, "1d")

    def test_slower_frames_keep_daily_led_fallback_candidate(self) -> None:
        for timeframe in ("4h", "1d"):
            plans = resolve_anchor_plan_candidates(timeframe)
            self.assertEqual(plans[1].lead_timeframe, "1d")
            self.assertEqual(plans[1].confirm_timeframe, "4h")

    def test_choose_anchor_context_falls_back_when_weekly_context_is_missing(self) -> None:
        frames = {
            "1w": None,
            "1d": "daily",
            "4h": "four_hour",
        }

        plan, lead_frame, confirm_frame = choose_anchor_context("4h", lambda tf: frames.get(tf))
        self.assertEqual(plan.lead_timeframe, "1d")
        self.assertEqual(plan.confirm_timeframe, "4h")
        self.assertEqual(lead_frame, "daily")
        self.assertEqual(confirm_frame, "four_hour")

    def test_choose_anchor_context_uses_weekly_plan_when_available(self) -> None:
        frames = {
            "1w": "weekly",
            "1d": "daily",
            "4h": "four_hour",
        }

        plan, lead_frame, confirm_frame = choose_anchor_context("4h", lambda tf: frames.get(tf))
        self.assertEqual(plan.lead_timeframe, "1w")
        self.assertEqual(plan.confirm_timeframe, "1d")
        self.assertEqual(lead_frame, "weekly")
        self.assertEqual(confirm_frame, "daily")


if __name__ == "__main__":
    unittest.main()
