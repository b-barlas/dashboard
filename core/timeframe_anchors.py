"""Adaptive higher-timeframe anchor plans for decision engines."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, TypeVar


_T = TypeVar("_T")


@dataclass(frozen=True)
class AnchorPlan:
    selected_timeframe: str
    lead_timeframe: str
    confirm_timeframe: str

    @property
    def pair_label(self) -> str:
        return f"{self.lead_timeframe.upper()} + {self.confirm_timeframe.upper()}"

    @property
    def role_label(self) -> str:
        return f"{self.lead_timeframe.upper()} lead + {self.confirm_timeframe.upper()} confirm"


def resolve_anchor_plan(selected_timeframe: str) -> AnchorPlan:
    return resolve_anchor_plan_candidates(selected_timeframe)[0]


def resolve_anchor_plan_candidates(selected_timeframe: str) -> tuple[AnchorPlan, ...]:
    tf = str(selected_timeframe or "").strip().lower()
    if tf in {"1m", "3m", "5m", "15m"}:
        return (
            AnchorPlan(
                selected_timeframe=tf or "15m",
                lead_timeframe="4h",
                confirm_timeframe="1h",
            ),
        )
    if tf == "1h":
        return (
            AnchorPlan(
                selected_timeframe=tf,
                lead_timeframe="1d",
                confirm_timeframe="4h",
            ),
        )
    if tf in {"4h", "1d"}:
        return (
            AnchorPlan(
                selected_timeframe=tf,
                lead_timeframe="1w",
                confirm_timeframe="1d",
            ),
            AnchorPlan(
                selected_timeframe=tf,
                lead_timeframe="1d",
                confirm_timeframe="4h",
            ),
        )
    return (
        AnchorPlan(
            selected_timeframe=tf or "1h",
            lead_timeframe="1d",
            confirm_timeframe="4h",
        ),
    )


def choose_anchor_context(
    selected_timeframe: str,
    fetch_frame: Callable[[str], _T | None],
) -> tuple[AnchorPlan, _T | None, _T | None]:
    candidates = resolve_anchor_plan_candidates(selected_timeframe)
    best_plan = candidates[0]
    best_lead = None
    best_confirm = None
    best_score = -1

    for plan in candidates:
        confirm_frame = fetch_frame(plan.confirm_timeframe)
        lead_frame = fetch_frame(plan.lead_timeframe)
        score = int(confirm_frame is not None) + int(lead_frame is not None)
        if score > best_score:
            best_plan = plan
            best_lead = lead_frame
            best_confirm = confirm_frame
            best_score = score
        if score == 2:
            return plan, lead_frame, confirm_frame

    return best_plan, best_lead, best_confirm
