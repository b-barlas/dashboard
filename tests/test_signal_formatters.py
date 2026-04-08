from types import SimpleNamespace

from core.trading_copy import (
    get_copy_audience,
    playbook_key,
    set_copy_audience,
    setup_class_display,
    setup_class_key,
    trade_gate_key,
)
from ui.signal_formatters import (
    action_family,
    archived_execution_stance_label,
    context_fit_snapshot,
    execution_read_note,
    setup_confirm_display,
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
    assert snap["gate_key"] == "TRADEABLE"
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
    assert snap["gate_key"] == "NO_TRADE"
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


def test_action_family_groups_internal_classes_into_stable_ui_buckets() -> None:
    assert action_family("ENTER_TREND_AI") == "enter"
    assert action_family("PROBE") == "probe"
    assert action_family("WATCH") == "watch"
    assert action_family("SKIP") == "skip"


def test_setup_confirm_display_supports_future_neutral_presentation_mode() -> None:
    assert setup_confirm_display("ENTER_TREND_LED") == "TREND-led"
    assert setup_confirm_display("ENTER_TREND_LED", audience="neutral") == "High-Quality Setup"
    assert setup_confirm_display("PROBE", audience="neutral") == "Early Setup"


def test_playbook_key_normalizes_visible_labels_into_stable_logic_keys() -> None:
    assert playbook_key("Wait for confirmation") == "WAIT_CONFIRMATION"
    assert playbook_key("Stand aside / mean reversion only") == "MEAN_REVERSION_OR_STAND_ASIDE"
    assert playbook_key("Defensive / downside only") == "DEFENSIVE_DOWNSIDE_ONLY"


def test_setup_confirm_display_uses_global_copy_audience_when_not_overridden() -> None:
    previous = get_copy_audience()
    try:
        set_copy_audience("neutral")
        assert setup_confirm_display("PROBE") == "Early Setup"
    finally:
        set_copy_audience(previous)


def test_trade_gate_key_normalizes_neutral_labels_into_stable_logic_keys() -> None:
    assert trade_gate_key("Low Alignment") == "NO_TRADE"
    assert trade_gate_key("Cautious") == "DEFENSIVE_ONLY"
    assert trade_gate_key("Selective") == "SELECTIVE_ONLY"
    assert trade_gate_key("Supportive") == "TRADEABLE"


def test_setup_class_display_preserves_distinct_backtest_class_labels() -> None:
    assert setup_class_key("TREND+AI") == "ENTER_TREND_AI"
    assert setup_class_display("ENTER_TREND_AI") == "TREND+AI"
    assert setup_class_display("ENTER_TREND_AI", audience="neutral") == "Trend + Model Aligned"
    assert setup_class_display("ENTER_TREND_LED", audience="neutral") == "Trend-Led Setup"
    assert setup_class_display("ENTER_AI_LED", audience="neutral") == "Model-Led Setup"
