"""Version stamps for archive-facing decision logic."""

from __future__ import annotations


# Bump this when scanner decision semantics change enough that old and new
# archive cohorts should be considered different populations.
MARKET_SCANNER_DECISION_VERSION = "market-scanner-2026-04-10-v1"


def current_decision_version(source: str | None = None) -> str:
    normalized = str(source or "").strip().lower()
    if normalized in {"", "market"}:
        return MARKET_SCANNER_DECISION_VERSION
    return ""
