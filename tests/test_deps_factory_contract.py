import ast
from pathlib import Path
import unittest

try:
    from ui.deps_factory import (
        build_app_deps,
        missing_fetch_coingecko_ohlcv_by_coin_id,
    )
    from ui.tab_registry import required_dep_keys

    DEPS_OK = True
except Exception:
    DEPS_OK = False


@unittest.skipUnless(DEPS_OK, "Missing UI dependencies for deps factory tests")
class DepsFactoryContractTests(unittest.TestCase):
    def test_build_app_deps_contains_required_keys(self):
        source = {k: object() for k in required_dep_keys()}
        source.update({"st": object(), "ACCENT": object(), "POSITIVE": object()})
        out = build_app_deps(source)
        self.assertIn("st", out)
        for k in required_dep_keys():
            self.assertIn(k, out)

    def test_build_app_deps_raises_on_missing(self):
        with self.assertRaises(KeyError):
            build_app_deps({})

    def test_missing_coingecko_fallback_is_marked(self):
        self.assertTrue(getattr(missing_fetch_coingecko_ohlcv_by_coin_id, "_codex_missing_dep", False))
        self.assertIn(
            "dependency injection",
            str(getattr(missing_fetch_coingecko_ohlcv_by_coin_id, "_codex_missing_dep_reason", "")),
        )

    def test_crypto_meta_app_deps_match_tab_registry_contract(self):
        module = ast.parse(Path("crypto_meta2.py").read_text())
        app_dep_keys = None
        for node in module.body:
            if not isinstance(node, ast.Assign):
                continue
            if not any(isinstance(target, ast.Name) and target.id == "APP_DEPS" for target in node.targets):
                continue
            self.assertIsInstance(node.value, ast.Dict)
            app_dep_keys = {
                key.value
                for key in node.value.keys
                if isinstance(key, ast.Constant) and isinstance(key.value, str)
            }
            break

        self.assertIsNotNone(app_dep_keys)
        self.assertEqual(app_dep_keys, required_dep_keys())


if __name__ == "__main__":
    unittest.main()
