import unittest

try:
    import ui.tab_registry as reg

    DEPS_OK = True
except Exception:
    DEPS_OK = False


class _FallbackDict(dict):
    def __contains__(self, _key):
        return True

    def __missing__(self, key):
        return f"<{key}>"


@unittest.skipUnless(DEPS_OK, "Missing UI dependencies for tab registry tests")
class TabRegistryContractTests(unittest.TestCase):
    def test_titles_and_specs_count_match(self):
        deps = _FallbackDict()
        specs = reg.build_tab_specs(deps)
        self.assertGreaterEqual(len(reg.TAB_TITLES), 15)
        self.assertEqual(len(reg.TAB_TITLES), len(specs))
        for renderer, ctx in specs:
            self.assertTrue(callable(renderer))
            self.assertIsInstance(ctx, dict)

    def test_build_tab_spec_returns_only_requested_context(self):
        deps = _FallbackDict()
        renderer, ctx = reg.build_tab_spec(deps, "Signal Archive")
        self.assertTrue(callable(renderer))
        self.assertIsInstance(ctx, dict)
        self.assertIn("fetch_signal_events_df", ctx)
        self.assertNotIn("get_market_top_snapshot", ctx)


if __name__ == "__main__":
    unittest.main()
