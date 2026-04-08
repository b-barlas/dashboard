from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from core.catalyst_engine import (
    build_market_catalyst_snapshot,
    catalyst_event_matches_signal,
    catalyst_signal_size_cap,
    catalyst_window_label,
    load_manual_catalyst_events,
    normalize_catalyst_event,
)


class CatalystEngineTests(unittest.TestCase):
    def test_load_manual_catalyst_events_reads_list_payload(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "market_catalysts.json"
            path.write_text(
                '[{"title":"US CPI","event_time":"2026-04-05T12:30:00Z","severity":"high"}]',
                encoding="utf-8",
            )
            rows = load_manual_catalyst_events(path)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["title"], "US CPI")

    def test_normalize_catalyst_event_rejects_invalid_rows(self) -> None:
        self.assertIsNone(normalize_catalyst_event({"title": "", "event_time": "bad"}))

    def test_blocking_high_impact_event_caps_size(self) -> None:
        snap = build_market_catalyst_snapshot(
            [
                {
                    "title": "US CPI",
                    "event_time": "2026-04-05T12:30:00Z",
                    "severity": "high",
                    "category": "macro",
                    "scope": "market",
                }
            ],
            now="2026-04-05T08:00:00Z",
        )
        self.assertTrue(snap.blocking)
        self.assertEqual(snap.size_cap_fraction, 0.0)

    def test_medium_event_becomes_caution(self) -> None:
        snap = build_market_catalyst_snapshot(
            [
                {
                    "title": "Token Unlock",
                    "event_time": "2026-04-05T18:00:00Z",
                    "severity": "medium",
                    "category": "unlock",
                    "scope": "market",
                }
            ],
            now="2026-04-05T10:00:00Z",
        )
        self.assertTrue(snap.caution)
        self.assertFalse(snap.blocking)
        self.assertEqual(snap.size_cap_fraction, 0.5)

    def test_no_events_returns_clear(self) -> None:
        snap = build_market_catalyst_snapshot([], now="2026-04-05T10:00:00Z")
        self.assertEqual(snap.state, "CLEAR")
        self.assertEqual(snap.size_cap_fraction, 1.0)

    def test_targeted_sector_event_only_caps_matching_sector(self) -> None:
        snap = build_market_catalyst_snapshot(
            [
                {
                    "title": "AI Token Unlock Wave",
                    "event_time": "2026-04-05T18:00:00Z",
                    "severity": "high",
                    "category": "unlock",
                    "scope": "sector",
                    "tag": "AI",
                }
            ],
            now="2026-04-05T10:00:00Z",
        )
        self.assertEqual(snap.state, "TARGETED_CAUTION")
        self.assertTrue(snap.targeted_only)
        self.assertTrue(catalyst_event_matches_signal(snap, symbol="FET", sector_tag="AI"))
        self.assertFalse(catalyst_event_matches_signal(snap, symbol="ETH", sector_tag="L1"))
        self.assertAlmostEqual(catalyst_signal_size_cap(snap, symbol="FET", sector_tag="AI"), 0.25, places=4)
        self.assertAlmostEqual(catalyst_signal_size_cap(snap, symbol="ETH", sector_tag="L1"), 1.0, places=4)
        self.assertEqual(catalyst_window_label(snap), "Targeted Caution")


if __name__ == "__main__":
    unittest.main()
