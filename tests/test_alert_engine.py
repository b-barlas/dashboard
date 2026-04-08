from __future__ import annotations

import unittest
from types import SimpleNamespace

from core.alert_engine import build_market_alerts


class AlertEngineTests(unittest.TestCase):
    def test_build_market_alerts_includes_blocking_catalyst_and_trade_gate(self) -> None:
        alerts = build_market_alerts(
            market_lead_snapshot=SimpleNamespace(state="UPSIDE", score=69.0, note="Upside pressure is building."),
            market_regime_snapshot=SimpleNamespace(regime_key="RISK_ON_TREND", label="Risk-On Trend", playbook="Trend continuation"),
            market_trade_gate_snapshot=SimpleNamespace(
                gate_key="NO_TRADE",
                label="No-Trade",
                reason_code="REGIME_NO_TRADE",
                note="The market regime does not justify fresh risk.",
                no_trade=True,
            ),
            market_catalyst_snapshot=SimpleNamespace(
                state="BLOCKING",
                next_event="US CPI",
                note="US CPI is too close.",
                blocking=True,
                caution=True,
            ),
            market_flow_snapshot=SimpleNamespace(state="BALANCED", label="Flow Balanced", note="Balanced.", leader_symbol="", score=0.0),
            sector_rotation_snapshot=SimpleNamespace(state="NONE", leader_sector="None", leader_score=0.0, note=""),
            rows=[],
        )
        keys = [alert.alert_key for alert in alerts]
        self.assertIn("CATALYST_BLOCK", keys)
        self.assertIn("TRADE_GATE", keys)

    def test_build_market_alerts_finds_actionable_cluster(self) -> None:
        alerts = build_market_alerts(
            market_lead_snapshot=SimpleNamespace(state="UPSIDE", score=72.0, note="Upside pressure is building."),
            market_regime_snapshot=SimpleNamespace(regime_key="ALT_ROTATION", label="Alt Rotation", playbook="Selective upside rotation"),
            market_trade_gate_snapshot=SimpleNamespace(
                gate_key="SELECTIVE_ONLY",
                label="Selective Only",
                reason_code="SELECTIVE_FILTER",
                note="Take only aligned names.",
                no_trade=False,
            ),
            market_catalyst_snapshot=SimpleNamespace(
                state="CLEAR",
                next_event="",
                note="No near catalyst.",
                blocking=False,
                caution=False,
            ),
            market_flow_snapshot=SimpleNamespace(
                state="SHORT_CROWDING",
                label="Shorts Crowded",
                note="Shorts are leaning too hard.",
                leader_symbol="BTCUSDT",
                score=24.0,
            ),
            sector_rotation_snapshot=SimpleNamespace(
                state="UPSIDE",
                leader_sector="AI",
                leader_score=3.5,
                note="AI is clustering the strongest upside leadership.",
            ),
            session_fit_snapshot=SimpleNamespace(
                label="Session Supportive",
                note="European session has been one of the cleaner execution windows lately.",
                score=3.4,
            ),
            rows=[
                {
                    "Coin": "FET",
                    "__action_raw": "✅ ENTER (Trend+AI)",
                    "Direction": "Upside",
                    "__confidence_val": 84.0,
                    "__rr_val": 2.2,
                    "__emerging_direction": "Upside",
                    "__adaptive_edge_score": 68.0,
                    "__adaptive_edge_label": "Historically Favored",
                },
                {
                    "Coin": "RENDER",
                    "__action_raw": "👀 WATCH",
                    "Direction": "Upside",
                    "__confidence_val": 78.0,
                    "__rr_val": 1.8,
                    "__emerging_direction": "Upside",
                    "__adaptive_edge_score": 64.0,
                    "__adaptive_edge_label": "Historically Favored",
                },
            ],
            max_alerts=4,
        )
        keys = [alert.alert_key for alert in alerts]
        self.assertIn("ACTIONABLE_CLUSTER", keys)
        self.assertIn("MARKET_LEAD", keys)
        self.assertIn("LEARNED_EDGE", keys)
        self.assertIn("SECTOR_ROTATION", keys)

    def test_build_market_alerts_can_surface_session_fit_when_room_exists(self) -> None:
        alerts = build_market_alerts(
            market_lead_snapshot=SimpleNamespace(state="BALANCED", score=50.0, note="Mixed tape."),
            market_regime_snapshot=SimpleNamespace(regime_key="SELECTIVE_BALANCE", label="Selective Balance", playbook="Stay selective"),
            market_trade_gate_snapshot=SimpleNamespace(
                gate_key="SELECTIVE_ONLY",
                label="Selective Only",
                reason_code="SELECTIVE_FILTER",
                note="Take only aligned names.",
                no_trade=False,
            ),
            market_catalyst_snapshot=SimpleNamespace(
                state="CLEAR",
                next_event="",
                note="No near catalyst.",
                blocking=False,
                caution=False,
            ),
            market_flow_snapshot=SimpleNamespace(
                state="BALANCED",
                label="Flow Balanced",
                note="Balanced.",
                leader_symbol="",
                score=0.0,
            ),
            sector_rotation_snapshot=SimpleNamespace(
                state="NONE",
                leader_sector="",
                leader_score=0.0,
                note="",
            ),
            session_fit_snapshot=SimpleNamespace(
                label="Session Supportive",
                note="European session has been one of the cleaner execution windows lately.",
                score=3.4,
            ),
            rows=[],
            max_alerts=4,
        )
        keys = [alert.alert_key for alert in alerts]
        self.assertIn("SESSION_FIT", keys)

    def test_build_market_alerts_can_surface_archive_guardrail_cluster(self) -> None:
        alerts = build_market_alerts(
            market_lead_snapshot=SimpleNamespace(state="BALANCED", score=50.0, note="Mixed tape."),
            market_regime_snapshot=SimpleNamespace(regime_key="SELECTIVE_BALANCE", label="Selective Balance", playbook="Stay selective"),
            market_trade_gate_snapshot=SimpleNamespace(
                gate_key="SELECTIVE_ONLY",
                label="Selective Only",
                reason_code="SELECTIVE_FILTER",
                note="Take only aligned names.",
                no_trade=False,
            ),
            market_catalyst_snapshot=SimpleNamespace(
                state="CLEAR",
                next_event="",
                note="No near catalyst.",
                blocking=False,
                caution=False,
                targeted_only=False,
                tag="",
            ),
            market_flow_snapshot=SimpleNamespace(
                state="BALANCED",
                label="Flow Balanced",
                note="Balanced.",
                leader_symbol="",
                score=0.0,
            ),
            sector_rotation_snapshot=SimpleNamespace(
                state="NONE",
                leader_sector="",
                leader_score=0.0,
                note="",
            ),
            session_fit_snapshot=SimpleNamespace(
                label="Session Mixed",
                note="Mixed session archive.",
                score=0.0,
            ),
            rows=[
                {
                    "Coin": "ARB",
                    "__archive_guardrail_label": "Archive Guardrail",
                    "__archive_guardrail_penalty": 6.4,
                },
                {
                    "Coin": "OP",
                    "__archive_guardrail_label": "Archive Guardrail",
                    "__archive_guardrail_penalty": 5.7,
                },
            ],
            max_alerts=4,
        )
        keys = [alert.alert_key for alert in alerts]
        self.assertIn("ARCHIVE_GUARDRAIL", keys)

    def test_build_market_alerts_can_surface_playbook_window_fit(self) -> None:
        alerts = build_market_alerts(
            market_lead_snapshot=SimpleNamespace(state="BALANCED", score=50.0, note="Mixed tape."),
            market_regime_snapshot=SimpleNamespace(
                regime_key="ALT_ROTATION",
                label="Alt Rotation",
                playbook="Selective upside rotation",
            ),
            market_trade_gate_snapshot=SimpleNamespace(
                gate_key="SELECTIVE_ONLY",
                label="Selective Only",
                reason_code="SELECTIVE_FILTER",
                note="Take only aligned names.",
                no_trade=False,
            ),
            market_catalyst_snapshot=SimpleNamespace(
                state="CLEAR",
                label="Catalyst Clear",
                next_event="",
                note="No near catalyst.",
                blocking=False,
                caution=False,
                targeted_only=False,
                tag="",
            ),
            market_flow_snapshot=SimpleNamespace(
                state="BALANCED",
                label="Flow Balanced",
                note="Balanced.",
                leader_symbol="",
                score=0.0,
            ),
            sector_rotation_snapshot=SimpleNamespace(
                state="NONE",
                leader_sector="",
                leader_score=0.0,
                note="",
            ),
            session_fit_snapshot=SimpleNamespace(
                label="Session Supportive",
                note="European session has been one of the cleaner execution windows lately.",
                score=3.4,
            ),
            rows=[
                {
                    "Coin": "FET",
                    "__adaptive_edge_label": "Historically Favored",
                    "__adaptive_edge_score": 68.0,
                },
                {
                    "Coin": "RENDER",
                    "__adaptive_edge_label": "Historically Favored",
                    "__adaptive_edge_score": 64.0,
                },
            ],
            max_alerts=4,
        )
        keys = [alert.alert_key for alert in alerts]
        self.assertIn("PLAYBOOK_WINDOW", keys)

    def test_build_market_alerts_can_surface_supportive_execution_stance(self) -> None:
        alerts = build_market_alerts(
            market_lead_snapshot=SimpleNamespace(state="UPSIDE", score=68.0, note="Upside pressure is building."),
            market_regime_snapshot=SimpleNamespace(
                regime_key="RISK_ON_TREND",
                label="Risk-On Trend",
                playbook="Trend continuation",
            ),
            market_trade_gate_snapshot=SimpleNamespace(
                gate_key="TRADEABLE",
                label="Tradeable",
                reason_code="ALIGNED",
                note="Conditions are aligned.",
                no_trade=False,
            ),
            market_catalyst_snapshot=SimpleNamespace(
                state="CLEAR",
                label="Catalyst Clear",
                next_event="",
                note="No near catalyst.",
                blocking=False,
                caution=False,
                targeted_only=False,
                tag="",
            ),
            market_flow_snapshot=SimpleNamespace(
                state="BALANCED",
                label="Flow Balanced",
                note="Balanced.",
                leader_symbol="",
                score=0.0,
            ),
            sector_rotation_snapshot=SimpleNamespace(
                state="UPSIDE",
                leader_sector="AI",
                leader_score=3.2,
                note="AI leadership is improving.",
            ),
            session_fit_snapshot=SimpleNamespace(
                label="Session Supportive",
                note="European session has been one of the cleaner execution windows lately.",
                score=3.8,
            ),
            rows=[
                {
                    "Coin": "FET",
                    "__adaptive_edge_label": "Historically Favored",
                    "__adaptive_edge_score": 69.0,
                    "__archive_guardrail_label": "Archive Clear",
                    "__archive_guardrail_penalty": 1.2,
                    "__risk_unit_fraction": 0.95,
                },
                {
                    "Coin": "RENDER",
                    "__adaptive_edge_label": "Historically Favored",
                    "__adaptive_edge_score": 66.0,
                    "__archive_guardrail_label": "Archive Clear",
                    "__archive_guardrail_penalty": 1.6,
                    "__risk_unit_fraction": 0.82,
                },
            ],
            max_alerts=6,
        )
        keys = [alert.alert_key for alert in alerts]
        self.assertIn("EXECUTION_STANCE", keys)

    def test_build_market_alerts_can_surface_fragile_execution_stance(self) -> None:
        alerts = build_market_alerts(
            market_lead_snapshot=SimpleNamespace(state="BALANCED", score=49.0, note="Mixed tape."),
            market_regime_snapshot=SimpleNamespace(
                regime_key="SELECTIVE_BALANCE",
                label="Selective Balance",
                playbook="Stay selective",
            ),
            market_trade_gate_snapshot=SimpleNamespace(
                gate_key="DEFENSIVE_ONLY",
                label="Defensive Only",
                reason_code="FRAGILE_WINDOW",
                note="Conditions favor smaller size.",
                no_trade=False,
            ),
            market_catalyst_snapshot=SimpleNamespace(
                state="CAUTION",
                label="Catalyst Caution",
                next_event="Fed Speaker",
                note="A catalyst window is active.",
                blocking=False,
                caution=True,
                targeted_only=False,
                tag="",
            ),
            market_flow_snapshot=SimpleNamespace(
                state="BALANCED",
                label="Flow Balanced",
                note="Balanced.",
                leader_symbol="",
                score=0.0,
            ),
            sector_rotation_snapshot=SimpleNamespace(
                state="NONE",
                leader_sector="",
                leader_score=0.0,
                note="",
            ),
            session_fit_snapshot=SimpleNamespace(
                label="Session Fragile",
                note="This session has been a weaker conversion window lately.",
                score=-2.1,
            ),
            rows=[
                {
                    "Coin": "ARB",
                    "__adaptive_edge_label": "Historically Weak",
                    "__adaptive_edge_score": 34.0,
                    "__archive_guardrail_label": "Archive Guardrail",
                    "__archive_guardrail_penalty": 6.1,
                    "__risk_unit_fraction": 0.35,
                },
                {
                    "Coin": "OP",
                    "__adaptive_edge_label": "Historically Weak",
                    "__adaptive_edge_score": 38.0,
                    "__archive_guardrail_label": "Archive Guardrail",
                    "__archive_guardrail_penalty": 5.4,
                    "__risk_unit_fraction": 0.42,
                },
            ],
            max_alerts=6,
        )
        keys = [alert.alert_key for alert in alerts]
        self.assertIn("EXECUTION_STANCE", keys)


if __name__ == "__main__":
    unittest.main()
