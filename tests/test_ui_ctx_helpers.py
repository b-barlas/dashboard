import unittest

from ui.ctx import get_ctx, get_ctx_callable, get_ctx_typed, require_keys


class UiCtxHelpersTests(unittest.TestCase):
    def test_require_keys_raises_when_missing(self):
        with self.assertRaises(KeyError):
            require_keys({"a": 1}, ["a", "b"], scope="test")

    def test_get_ctx_typed_returns_value(self):
        out = get_ctx_typed({"x": 42}, "x", int, scope="test")
        self.assertEqual(out, 42)

    def test_get_ctx_callable_rejects_non_callable(self):
        with self.assertRaises(TypeError):
            get_ctx_callable({"fn": 42}, "fn", scope="test")

    def test_get_ctx_missing_key(self):
        with self.assertRaises(KeyError):
            get_ctx({}, "missing", scope="test")


if __name__ == "__main__":
    unittest.main()
