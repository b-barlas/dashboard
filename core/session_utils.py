"""Shared UTC session helpers used across timing and review surfaces."""

from __future__ import annotations

import pandas as pd

SESSION_ORDER = ["Asian (00-08 UTC)", "European (08-16 UTC)", "US (16-00 UTC)"]


def session_bucket_for_hour(hour: int) -> str:
    hour_int = int(hour)
    if 0 <= hour_int < 8:
        return SESSION_ORDER[0]
    if 8 <= hour_int < 16:
        return SESSION_ORDER[1]
    return SESSION_ORDER[2]


def session_bucket_for_timestamp(value: object | None = None) -> str:
    ts = pd.Timestamp.utcnow() if value is None else pd.to_datetime(value, utc=True, errors="coerce")
    if pd.isna(ts):
        ts = pd.Timestamp.utcnow()
    return session_bucket_for_hour(int(ts.hour))
