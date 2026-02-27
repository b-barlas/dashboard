from __future__ import annotations

import unittest

import pandas as pd

from ui.snapshot_cache import live_or_snapshot


class _St:
    def __init__(self):
        self.session_state = {}


class SnapshotCacheTests(unittest.TestCase):
    def test_live_value_updates_snapshot(self) -> None:
        st = _St()
        out, from_cache, ts = live_or_snapshot(st, "k", [1, 2, 3])
        self.assertFalse(from_cache)
        self.assertEqual(out, [1, 2, 3])
        self.assertTrue(isinstance(ts, str) and "UTC" in ts)

    def test_empty_live_uses_cached(self) -> None:
        st = _St()
        live_or_snapshot(st, "k", [1])
        out, from_cache, _ = live_or_snapshot(st, "k", [])
        self.assertTrue(from_cache)
        self.assertEqual(out, [1])

    def test_dataframe_handling(self) -> None:
        st = _St()
        df = pd.DataFrame({"x": [1]})
        out, from_cache, _ = live_or_snapshot(st, "dfk", df)
        self.assertFalse(from_cache)
        self.assertFalse(out.empty)
        out2, from_cache2, _ = live_or_snapshot(st, "dfk", pd.DataFrame())
        self.assertTrue(from_cache2)
        self.assertFalse(out2.empty)

    def test_signature_mismatch_blocks_cache(self) -> None:
        st = _St()
        live_or_snapshot(st, "k", [1], current_sig=("BTC", "1h"))
        out, from_cache, ts = live_or_snapshot(st, "k", [], current_sig=("BTC", "4h"))
        self.assertFalse(from_cache)
        self.assertEqual(out, [])
        self.assertIsNone(ts)

    def test_ttl_blocks_stale_cache(self) -> None:
        st = _St()
        live_or_snapshot(st, "k", [1], current_sig=("BTC",))
        # Force old cache age.
        st.session_state["k_ts_epoch"] = 0.0
        out, from_cache, ts = live_or_snapshot(st, "k", [], max_age_sec=60, current_sig=("BTC",))
        self.assertFalse(from_cache)
        self.assertEqual(out, [])
        self.assertIsNone(ts)


if __name__ == "__main__":
    unittest.main()
