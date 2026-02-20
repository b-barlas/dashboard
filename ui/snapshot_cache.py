from __future__ import annotations

from datetime import datetime, timezone

import pandas as pd


def _utc_now_str() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")


def _is_nonempty_payload(value) -> bool:
    if value is None:
        return False
    if isinstance(value, pd.DataFrame):
        return not value.empty
    if isinstance(value, (list, tuple, set, dict, str, bytes)):
        return len(value) > 0
    return True


def set_snapshot(st, key: str, value) -> None:
    st.session_state[f"{key}_value"] = value
    st.session_state[f"{key}_ts"] = _utc_now_str()


def get_snapshot(st, key: str):
    return st.session_state.get(f"{key}_value"), st.session_state.get(f"{key}_ts")


def live_or_snapshot(st, key: str, live_value):
    """Return (value, from_cache, ts). Uses snapshot when live value is empty."""
    if _is_nonempty_payload(live_value):
        set_snapshot(st, key, live_value)
        return live_value, False, st.session_state.get(f"{key}_ts")
    cached_value, cached_ts = get_snapshot(st, key)
    if _is_nonempty_payload(cached_value):
        return cached_value, True, cached_ts
    return live_value, False, None
