from tabs.spot_tab import (
    _spot_ai_display_value,
    _spot_ai_fallback_note,
    _spot_axis_tickformat,
    _spot_execution_map_copy,
    format_spot_price,
)


def test_spot_ai_fallback_note_maps_known_statuses() -> None:
    assert _spot_ai_fallback_note({"status": "insufficient_features"}).startswith("AI fallback active")
    assert _spot_ai_fallback_note({"status": "single_class_window"}).startswith("AI fallback active")
    assert _spot_ai_fallback_note({}) == ""


def test_spot_ai_display_value_marks_fallback() -> None:
    def _dir_label(value: str) -> str:
        return {"UPSIDE": "Upside", "DOWNSIDE": "Downside", "NEUTRAL": "Neutral"}.get(value, value)

    assert _spot_ai_display_value(_dir_label, "NEUTRAL", 0, "") == "Neutral (0/3)"
    assert _spot_ai_display_value(_dir_label, "NEUTRAL", 0, "fallback") == "Neutral* (0/3)"


def test_format_spot_price_preserves_low_price_precision() -> None:
    assert format_spot_price(0.000123456) == "$0.00012346"
    assert format_spot_price(0.0000123456) == "$0.0000123456"


def test_spot_execution_map_copy_downgrade_for_downside() -> None:
    copy = _spot_execution_map_copy("DOWNSIDE")
    assert copy["section_title"] == "Reclaim Map (Defensive)"
    assert copy["right_trigger"] == "Reclaim Trigger"
    assert copy["left_zone"] == "Watch Zone"


def test_spot_axis_tickformat_scales_with_price() -> None:
    assert _spot_axis_tickformat(65000.0) == ",.2f"
    assert _spot_axis_tickformat(0.12) == ",.6f"
    assert _spot_axis_tickformat(0.00012) == ",.8f"
