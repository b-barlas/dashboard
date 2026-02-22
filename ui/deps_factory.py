"""Dependency factory for app-shell injection."""

from __future__ import annotations

from collections.abc import Mapping

from ui.tab_registry import required_dep_keys


def build_app_deps(source: Mapping[str, object], **overrides: object) -> dict:
    """Build the dependency dictionary expected by `ui.app_shell.render_app`."""
    required = {"st", "ACCENT", "POSITIVE"} | required_dep_keys()
    merged = dict(source)
    merged.update(overrides)

    missing = sorted(k for k in required if k not in merged)
    if missing:
        raise KeyError(f"Missing app dependencies: {', '.join(missing)}")

    return {k: merged[k] for k in required}

