from __future__ import annotations

from datetime import datetime, timezone
from typing import Any


_KEY = "__telemetry_store"
_MAX_EVENTS = 120


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")


def _ensure_store(st) -> dict[str, Any]:
    store = st.session_state.get(_KEY)
    if isinstance(store, dict):
        return store
    store = {
        "started_at": _utc_now(),
        "total_events": 0,
        "error_events": 0,
        "cache_hits": 0,
        "cache_misses": 0,
        "http_failures": 0,
        "recent": [],
    }
    st.session_state[_KEY] = store
    return store


def record_event(st, event: str, *, status: str = "ok", source: str = "", detail: str = "") -> None:
    """Record a lightweight telemetry event in session_state."""
    try:
        store = _ensure_store(st)
        store["total_events"] = int(store.get("total_events", 0)) + 1
        if status.lower() in {"error", "fail", "failed"}:
            store["error_events"] = int(store.get("error_events", 0)) + 1
        if event == "cache_hit":
            store["cache_hits"] = int(store.get("cache_hits", 0)) + 1
        elif event == "cache_miss":
            store["cache_misses"] = int(store.get("cache_misses", 0)) + 1
        elif event == "http_failure":
            store["http_failures"] = int(store.get("http_failures", 0)) + 1

        recent = list(store.get("recent", []))
        recent.append(
            {
                "ts": _utc_now(),
                "event": str(event),
                "status": str(status),
                "source": str(source),
                "detail": str(detail)[:220],
            }
        )
        if len(recent) > _MAX_EVENTS:
            recent = recent[-_MAX_EVENTS:]
        store["recent"] = recent
        st.session_state[_KEY] = store
    except Exception:
        # Telemetry must never break user flow.
        return


def snapshot_summary(st) -> dict[str, Any]:
    store = _ensure_store(st)
    total = int(store.get("total_events", 0))
    errors = int(store.get("error_events", 0))
    cache_hits = int(store.get("cache_hits", 0))
    cache_misses = int(store.get("cache_misses", 0))
    http_failures = int(store.get("http_failures", 0))
    cache_total = cache_hits + cache_misses
    cache_hit_rate = (cache_hits / cache_total * 100.0) if cache_total > 0 else 0.0
    error_rate = (errors / total * 100.0) if total > 0 else 0.0
    return {
        "started_at": store.get("started_at", ""),
        "total_events": total,
        "error_events": errors,
        "cache_hits": cache_hits,
        "cache_misses": cache_misses,
        "http_failures": http_failures,
        "cache_hit_rate": cache_hit_rate,
        "error_rate": error_rate,
        "recent": list(store.get("recent", []))[-10:],
    }
