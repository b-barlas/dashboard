"""Centralized runtime policy constants."""

from __future__ import annotations

# UK-safe exchange fallback order for this dashboard.
# Keep this list aligned with product/legal requirements.
UK_SAFE_EXCHANGE_FALLBACKS = (
    ("kraken", {}),
    ("coinbase", {}),
    ("bitstamp", {}),
)

