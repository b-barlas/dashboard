import unittest

from ui.styles import app_css


class UiStylesTests(unittest.TestCase):
    def test_app_css_contains_style_tag(self):
        css = app_css()
        self.assertIn("<style>", css)
        self.assertIn("</style>", css)
        self.assertIn(".stApp", css)


if __name__ == "__main__":
    unittest.main()
