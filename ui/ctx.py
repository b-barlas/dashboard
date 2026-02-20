"""Shared context helpers for tab render functions."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import TypeVar


T = TypeVar("T")


def require_keys(ctx: Mapping[str, object], keys: list[str] | tuple[str, ...] | set[str], *, scope: str = "ctx") -> None:
    missing = [k for k in keys if k not in ctx]
    if missing:
        raise KeyError(f"Missing keys in {scope}: {', '.join(sorted(missing))}")


def get_ctx(ctx: Mapping[str, object], key: str, *, scope: str = "ctx") -> object:
    if key not in ctx:
        raise KeyError(f"Missing key '{key}' in {scope}")
    return ctx[key]


def get_ctx_typed(ctx: Mapping[str, object], key: str, expected_type: type[T], *, scope: str = "ctx") -> T:
    value = get_ctx(ctx, key, scope=scope)
    if not isinstance(value, expected_type):
        raise TypeError(
            f"Key '{key}' in {scope} must be {expected_type.__name__}, got {type(value).__name__}"
        )
    return value


def get_ctx_callable(ctx: Mapping[str, object], key: str, *, scope: str = "ctx") -> Callable:
    value = get_ctx(ctx, key, scope=scope)
    if not callable(value):
        raise TypeError(f"Key '{key}' in {scope} must be callable")
    return value

