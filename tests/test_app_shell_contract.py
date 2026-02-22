import unittest

try:
    import ui.app_shell as app_shell
    DEPS_OK = True
except Exception:
    DEPS_OK = False


@unittest.skipUnless(DEPS_OK, "Missing UI dependencies for app shell tests")
class AppShellContractTests(unittest.TestCase):
    def test_tab_titles_shape(self):
        self.assertEqual(len(app_shell.TAB_TITLES), 18)
        self.assertEqual(len(set(app_shell.TAB_TITLES)), 18)
        self.assertEqual(app_shell.TAB_TITLES[0], "Market")
        self.assertEqual(app_shell.TAB_TITLES[-1], "Analysis Guide")


if __name__ == "__main__":
    unittest.main()
