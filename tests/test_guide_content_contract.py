import pathlib
import unittest

GUIDE_PATH = pathlib.Path("/Users/burakbarlas/Documents/Claude/Web_ready_2002/tabs/guide_tab.py")


class GuideContentContractTests(unittest.TestCase):
    def test_guide_mentions_streamlit_and_uk_safe_exchanges(self):
        text = GUIDE_PATH.read_text(encoding="utf-8").lower()

        for token in ["streamlit", "kraken", "coinbase", "bitstamp", "beginner-friendly"]:
            self.assertIn(token, text)


if __name__ == "__main__":
    unittest.main()
