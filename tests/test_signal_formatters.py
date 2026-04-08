from types import SimpleNamespace

from ui.signal_formatters import (
    archived_execution_stance_label,
    execution_read_note,
    context_fit_snapshot,
    trade_gate_display_label,
)


def _adaptive(**kwargs):
    base = {
        "label": "Historically Neutral",
        "execution_fit_label": "Execution Mixed",
        "session_fit_label": "Session Mixed",
        "archive_guardrail_label": "",
    }
    base.update(kwargs)
    return SimpleNamespace(**base)


def test_context_fit_snapshot_returns_aligned_window_for_supportive_stack() -> None:
    snap = context_fit_snapshot(
        _adaptive(
            label="Historically Favored",
            execution_fit_label="Execution Proven",
            session_fit_label="Session Supportive",
        ),
        market_context={
            "Playbook": "Selective upside rotation",
            "Trade Gate": "Selective Only",
            "Catalyst Window": "Far / Clear",
            "Flow Proxy": "Shorts Crowded",
        },
        recent_symbol_market_signal={
            "Lead": "LEAD",
            "Signal Note": "Recent Market scanner read: 1H | Emerging Upside | Tier 1.",
        },
    )
    assert snap["label"] == "Tradeable"
    assert snap["aggression"] == "Normal aggression"
    assert "Selective upside rotation" in snap["note"]


def test_context_fit_snapshot_returns_stand_aside_for_guardrailed_window() -> None:
    snap = context_fit_snapshot(
        _adaptive(
            label="Historically Weak",
            execution_fit_label="Execution Fragile",
            session_fit_label="Session Fragile",
            archive_guardrail_label="Archive Guardrail",
        ),
        market_context={
            "Trade Gate": "No-Trade",
            "Catalyst Window": "Blocking (<6h)",
        },
        recent_symbol_market_signal={},
    )
    assert snap["label"] == "Stand Aside"
    assert snap["aggression"] == "No fresh risk"


def test_trade_gate_display_label_maps_no_trade_to_user_facing_language() -> None:
    assert trade_gate_display_label("No-Trade") == "Stand Aside"
    assert trade_gate_display_label("Selective Only") == "Selective Only"


def test_archived_execution_stance_label_derives_unified_review_bucket() -> None:
    assert archived_execution_stance_label(
        trade_gate="Tradeable",
        adaptive_edge="Historically Favored",
        archive_guardrail_severity="Clear",
    ) == "Tradeable"
    assert archived_execution_stance_label(
        trade_gate="No-Trade",
        adaptive_edge="Historically Favored",
        archive_guardrail_severity="Guardrail",
    ) == "Stand Aside"


def test_execution_read_note_compacts_duplicate_context_into_shorter_line() -> None:
    adaptive = _adaptive(
        label="Historically Favored",
        execution_fit_label="Execution Proven",
        session_fit_label="Session Supportive",
    )
    adaptive.note = "Selective upside rotation"
    adaptive.execution_fit_note = "Execution Proven"
    adaptive.session_fit_note = "Session Supportive"
    note = execution_read_note(
        adaptive,
        context_fit={
            "label": "Tradeable",
            "aggression": "Normal aggression",
            "note": "Playbook: Selective upside rotation | Gate: Selective Only",
        },
        market_context_note="Recent market archive: Selective upside rotation | Gate: Selective Only",
        scanner_signal_note="Recent Market scanner read: 1H | Emerging Upside | Tier 1.",
    )
    assert "Selective upside rotation" in note
    assert "Stance: Tradeable — Normal aggression." in note
    assert note.count("Selective upside rotation") == 1
