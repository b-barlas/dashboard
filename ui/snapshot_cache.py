from __future__ import annotations

from datetime import datetime, timezone
import time

import pandas as pd
from core.telemetry import record_event


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


def set_snapshot(st, key: str, value, *, sig=None) -> None:
    st.session_state[f"{key}_value"] = value
    st.session_state[f"{key}_ts"] = _utc_now_str()
    st.session_state[f"{key}_ts_epoch"] = time.time()
    st.session_state[f"{key}_sig"] = sig


def get_snapshot(st, key: str):
    return (
        st.session_state.get(f"{key}_value"),
        st.session_state.get(f"{key}_ts"),
        st.session_state.get(f"{key}_ts_epoch"),
        st.session_state.get(f"{key}_sig"),
    )


def live_or_snapshot(st, key: str, live_value, *, max_age_sec: int | None = 900, current_sig=None):
    """Return (value, from_cache, ts). Uses snapshot when live value is empty.

    Guards:
    - TTL guard via ``max_age_sec`` (None disables age check)
    - Optional signature guard via ``current_sig``
    """
    if _is_nonempty_payload(live_value):
        set_snapshot(st, key, live_value, sig=current_sig)
        record_event(st, "cache_miss", status="ok", source="snapshot", detail=key)
        return live_value, False, st.session_state.get(f"{key}_ts")
    cached_value, cached_ts, cached_epoch, cached_sig = get_snapshot(st, key)
    is_fresh = True
    if max_age_sec is not None:
        try:
            if cached_epoch is None:
                is_fresh = False
            else:
                is_fresh = (time.time() - float(cached_epoch)) <= float(max_age_sec)
        except Exception:
            is_fresh = False
    same_sig = True
    if current_sig is not None:
        same_sig = cached_sig == current_sig
    if _is_nonempty_payload(cached_value) and is_fresh and same_sig:
        record_event(st, "cache_hit", status="ok", source="snapshot", detail=key)
        return cached_value, True, cached_ts
    miss_detail = f"{key}:empty"
    if _is_nonempty_payload(cached_value) and not is_fresh:
        miss_detail = f"{key}:stale"
    elif _is_nonempty_payload(cached_value) and not same_sig:
        miss_detail = f"{key}:sig_mismatch"
    record_event(st, "cache_miss", status="ok", source="snapshot", detail=miss_detail)
    return live_value, False, None
