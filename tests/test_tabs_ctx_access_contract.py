import pathlib
import unittest


TABS_DIR = pathlib.Path("/Users/burakbarlas/Documents/Claude/Web_ready_2002/tabs")


class TabsCtxAccessContractTests(unittest.TestCase):
    def test_tabs_use_shared_ctx_helper(self):
        tab_files = sorted(TABS_DIR.glob("*_tab.py"))
        self.assertGreater(len(tab_files), 0)
        for path in tab_files:
            text = path.read_text(encoding="utf-8")
            self.assertIn("from ui.ctx import get_ctx", text, msg=f"{path.name} must import get_ctx")
            self.assertNotIn('ctx["', text, msg=f"{path.name} should not use direct ctx indexing")


if __name__ == "__main__":
    unittest.main()
