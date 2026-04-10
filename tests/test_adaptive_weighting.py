from __future__ import annotations

import unittest

import pandas as pd

from core.adaptive_weighting import (
    build_actionable_ranking_model,
    build_actionable_ranking_snapshot,
    build_ai_confidence_calibration_model,
    build_ai_confidence_calibration_snapshot,
    build_adaptive_context_model,
    build_archive_guardrail_snapshot,
    build_confidence_calibration_model,
    build_confidence_calibration_snapshot,
    build_learning_edge_table,
    build_live_signal_adaptive_snapshot,
    build_risk_sizing_calibration_model,
    build_risk_sizing_calibration_snapshot,
    build_scalp_calibration_model,
    build_scalp_calibration_snapshot,
    build_setup_calibration_model,
    build_setup_calibration_snapshot,
    build_session_fit_snapshot,
    build_trade_gate_calibration_model,
    build_trade_gate_calibration_snapshot,
)


class AdaptiveWeightingTests(unittest.TestCase):
    def _resolved_history(self) -> pd.DataFrame:
        rows = []
        for idx in range(12):
            rows.append(
                {
                    "symbol": f"AI{idx}",
                    "status": "RESOLVED",
                    "directional_return_pct": 2.5 if idx < 9 else -1.0,
                    "direction": "Upside",
                    "scan_focus": "Actionable Setups",
                    "lead_active": 1,
                    "ai_aligned": 1,
                    "market_lead_label": "Upside",
                    "market_regime": "Alt Rotation",
                    "market_playbook": "Selective upside rotation",
                    "market_trade_gate": "Selective Only",
                    "market_primary_alert": "EXECUTION_STANCE",
                    "market_sector_rotation": "AI Rotation",
                    "market_catalyst_state": "Catalyst Clear",
                    "market_catalyst_window": "Far / Clear",
                    "market_catalyst_scope": "Market",
                    "market_catalyst_targeted": 0,
                    "market_flow_state": "Shorts Crowded",
                    "session_bucket": "European (08-16 UTC)",
                    "timeframe": "1h",
                    "setup_confirm": "✅ ENTER (Trend+AI)",
                    "has_plan": 1,
                    "plan_outcome": "TP" if idx < 9 else "SL",
                    "rr_ratio": 1.85 if idx < 9 else 1.25,
                    "adaptive_edge_label": "Historically Favored",
                    "archive_guardrail_label": "Archive Clear",
                    "actual_trade_status": "CLOSED" if idx < 8 else "",
                    "actual_pnl_pct": 3.5 if idx < 8 else None,
                }
            )
        for idx in range(12):
            rows.append(
                {
                    "symbol": f"BAD{idx}",
                    "status": "RESOLVED",
                    "directional_return_pct": -2.0 if idx < 9 else 1.0,
                    "direction": "Downside",
                    "scan_focus": "Broad Market",
                    "lead_active": 0,
                    "ai_aligned": 0,
                    "market_lead_label": "Balanced",
                    "market_regime": "Range / Chop",
                    "market_playbook": "Stand aside / mean reversion only",
                    "market_trade_gate": "No-Trade",
                    "market_primary_alert": "TRADE_GATE",
                    "market_sector_rotation": "Mixed Sector Rotation",
                    "market_catalyst_state": "Catalyst Caution",
                    "market_catalyst_window": "Blocking (<6h)",
                    "market_catalyst_scope": "Market",
                    "market_catalyst_targeted": 0,
                    "market_flow_state": "Longs Crowded",
                    "session_bucket": "Asian (00-08 UTC)",
                    "timeframe": "1h",
                    "setup_confirm": "⛔ SKIP",
                    "has_plan": 1,
                    "plan_outcome": "SL" if idx < 9 else "TP",
                    "rr_ratio": 1.20 if idx < 9 else 1.55,
                    "adaptive_edge_label": "Historically Weak",
                    "archive_guardrail_label": "Archive Guardrail",
                    "actual_trade_status": "CLOSED" if idx < 8 else "",
                    "actual_pnl_pct": -2.5 if idx < 8 else None,
                }
            )
        return pd.DataFrame(rows)

    def test_live_signal_adaptive_snapshot_favors_historically_strong_mix(self) -> None:
        model = build_adaptive_context_model(self._resolved_history())
        snap = build_live_signal_adaptive_snapshot(
            model,
            signal={
                "Setup Confirm": "✅ ENTER (Trend+AI)",
                "Lead": "LEAD",
                "AI Alignment": "Aligned",
                "Market Lead": "Upside",
                "Market Regime": "Alt Rotation",
                "Playbook": "Selective upside rotation",
                "Trade Gate": "Selective Only",
                "Sector Rotation": "AI Rotation",
                "Catalyst State": "Catalyst Clear",
                "Catalyst Window": "Far / Clear",
                "Catalyst Scope": "Market",
                "Catalyst Targeting": "Market-Wide",
                "Flow Proxy": "Shorts Crowded",
                "Session": "European (08-16 UTC)",
                "Timeframe": "1h",
            },
        )
        self.assertGreaterEqual(snap.score, 58.0)
        self.assertEqual(snap.label, "Historically Favored")
        self.assertGreaterEqual(snap.actual_trade_sample, 8)
        self.assertEqual(snap.execution_fit_label, "Execution Proven")

    def test_learning_edge_table_returns_ranked_rows(self) -> None:
        model = build_adaptive_context_model(self._resolved_history())
        table = build_learning_edge_table(model, limit=20)
        self.assertFalse(table.empty)
        self.assertIn("EdgeScore", table.columns)
        self.assertIn("ActualWinPct", table.columns)
        self.assertIn("AvgActualPnlPct", table.columns)
        self.assertIn("Session", set(table["Lens"]))
        self.assertIn("Playbook", set(table["Lens"]))
        self.assertIn("Execution Stance", set(table["Lens"]))
        self.assertIn("Primary Alert", set(table["Lens"]))
        self.assertIn("Primary Alert x Playbook", set(table["Lens"]))
        self.assertIn("Playbook x Session", set(table["Lens"]))
        self.assertIn("Playbook x Catalyst Window", set(table["Lens"]))

    def test_session_fit_snapshot_reads_supportive_bucket(self) -> None:
        model = build_adaptive_context_model(self._resolved_history())
        session_snap = build_session_fit_snapshot(model, "European (08-16 UTC)")
        self.assertEqual(session_snap.label, "Session Supportive")
        self.assertGreater(session_snap.score, 0.0)

    def test_archive_guardrail_snapshot_flags_weak_trade_gate_window(self) -> None:
        model = build_adaptive_context_model(self._resolved_history())
        guardrail = build_archive_guardrail_snapshot(
            model,
            signal={
                "Trade Gate": "No-Trade",
                "Catalyst Window": "Blocking (<6h)",
                "Session": "Asian (00-08 UTC)",
                "Market Regime": "Range / Chop",
                "Catalyst Scope": "Market",
            },
        )
        self.assertEqual(guardrail.label, "Archive Guardrail")
        self.assertGreaterEqual(guardrail.penalty, 5.0)
        self.assertIn("Trade Gate", guardrail.note)

    def test_recent_history_outweighs_stale_history_in_adaptive_snapshot(self) -> None:
        rows = []
        for idx in range(10):
            rows.append(
                {
                    "symbol": f"OLD{idx}",
                    "status": "RESOLVED",
                    "event_time": f"2026-01-{idx + 1:02d}T12:00:00Z",
                    "directional_return_pct": 3.0,
                    "lead_active": 1,
                    "ai_aligned": 1,
                    "market_lead_label": "Upside",
                    "market_regime": "Alt Rotation",
                    "market_playbook": "Selective upside rotation",
                    "market_trade_gate": "Selective Only",
                    "market_sector_rotation": "AI Rotation",
                    "market_catalyst_state": "Catalyst Clear",
                    "market_catalyst_window": "Far / Clear",
                    "market_catalyst_scope": "Market",
                    "market_catalyst_targeted": 0,
                    "market_flow_state": "Shorts Crowded",
                    "session_bucket": "European (08-16 UTC)",
                    "timeframe": "1h",
                    "setup_confirm": "✅ ENTER (Trend+AI)",
                }
            )
        for idx in range(10):
            rows.append(
                {
                    "symbol": f"NEW{idx}",
                    "status": "RESOLVED",
                    "event_time": f"2026-04-{idx + 1:02d}T12:00:00Z",
                    "directional_return_pct": -2.5,
                    "lead_active": 1,
                    "ai_aligned": 1,
                    "market_lead_label": "Upside",
                    "market_regime": "Alt Rotation",
                    "market_playbook": "Selective upside rotation",
                    "market_trade_gate": "Selective Only",
                    "market_sector_rotation": "AI Rotation",
                    "market_catalyst_state": "Catalyst Clear",
                    "market_catalyst_window": "Far / Clear",
                    "market_catalyst_scope": "Market",
                    "market_catalyst_targeted": 0,
                    "market_flow_state": "Shorts Crowded",
                    "session_bucket": "European (08-16 UTC)",
                    "timeframe": "1h",
                    "setup_confirm": "✅ ENTER (Trend+AI)",
                }
            )
        for idx in range(12):
            rows.append(
                {
                    "symbol": f"CTRL{idx}",
                    "status": "RESOLVED",
                    "event_time": f"2026-04-{idx + 1:02d}T16:00:00Z",
                    "directional_return_pct": 2.8,
                    "lead_active": 1,
                    "ai_aligned": 1,
                    "market_lead_label": "Upside",
                    "market_regime": "Risk-On Trend",
                    "market_playbook": "Trend continuation",
                    "market_trade_gate": "Tradeable",
                    "market_sector_rotation": "AI Rotation",
                    "market_catalyst_state": "Catalyst Clear",
                    "market_catalyst_window": "Far / Clear",
                    "market_catalyst_scope": "Market",
                    "market_catalyst_targeted": 0,
                    "market_flow_state": "Shorts Crowded",
                    "session_bucket": "European (08-16 UTC)",
                    "timeframe": "1h",
                    "setup_confirm": "✅ ENTER (Trend+AI)",
                }
            )
        model = build_adaptive_context_model(pd.DataFrame(rows))
        snap = build_live_signal_adaptive_snapshot(
            model,
            signal={
                "Setup Confirm": "✅ ENTER (Trend+AI)",
                "Lead": "LEAD",
                "AI Alignment": "Aligned",
                "Market Lead": "Upside",
                "Market Regime": "Alt Rotation",
                "Playbook": "Selective upside rotation",
                "Trade Gate": "Selective Only",
                "Sector Rotation": "AI Rotation",
                "Catalyst State": "Catalyst Clear",
                "Catalyst Window": "Far / Clear",
                "Catalyst Scope": "Market",
                "Catalyst Targeting": "Market-Wide",
                "Flow Proxy": "Shorts Crowded",
                "Session": "European (08-16 UTC)",
                "Timeframe": "1h",
            },
        )
        self.assertLessEqual(snap.score, 48.0)

    def test_ai_confidence_calibration_supports_strong_aligned_enter_cohort(self) -> None:
        enriched = pd.concat([self._resolved_history(), self._resolved_history()], ignore_index=True)
        model = build_ai_confidence_calibration_model(enriched)
        snap = build_ai_confidence_calibration_snapshot(
            model,
            signal={
                "Setup Confirm": "ENTER_TREND_AI",
                "AI Alignment": "Aligned",
                "Timeframe": "1h",
                "Scan Focus": "Actionable Setups",
                "Direction": "Upside",
            },
        )
        self.assertGreater(snap.delta, 0.0)
        self.assertIn("Archive calibration", snap.note)

    def test_ai_confidence_calibration_stays_off_when_archive_is_thin(self) -> None:
        thin = self._resolved_history().head(12).copy()
        model = build_ai_confidence_calibration_model(thin)
        snap = build_ai_confidence_calibration_snapshot(
            model,
            signal={
                "Setup Confirm": "ENTER_TREND_AI",
                "AI Alignment": "Aligned",
                "Timeframe": "1h",
                "Scan Focus": "Actionable Setups",
                "Direction": "Upside",
            },
        )
        self.assertEqual(snap.delta, 0.0)
        self.assertEqual(snap.note, "")

    def test_confidence_calibration_supports_strong_directional_cohort(self) -> None:
        enriched = pd.concat([self._resolved_history(), self._resolved_history()], ignore_index=True)
        model = build_confidence_calibration_model(enriched)
        snap = build_confidence_calibration_snapshot(
            model,
            signal={
                "Direction": "Upside",
                "AI Alignment": "Aligned",
                "Timeframe": "1h",
                "Scan Focus": "Actionable Setups",
            },
        )
        self.assertGreater(snap.delta, 0.0)
        self.assertIn("Archive confidence calibration", snap.note)

    def test_confidence_calibration_softens_weak_directional_cohort(self) -> None:
        enriched = pd.concat([self._resolved_history(), self._resolved_history()], ignore_index=True)
        model = build_confidence_calibration_model(enriched)
        snap = build_confidence_calibration_snapshot(
            model,
            signal={
                "Direction": "Downside",
                "AI Alignment": "Not aligned",
                "Timeframe": "1h",
                "Scan Focus": "Broad Market",
            },
        )
        self.assertLess(snap.delta, 0.0)
        self.assertIn("cautious", snap.note.lower())

    def test_setup_calibration_supports_strong_enter_cohort(self) -> None:
        enriched = pd.concat([self._resolved_history(), self._resolved_history()], ignore_index=True)
        model = build_setup_calibration_model(enriched)
        snap = build_setup_calibration_snapshot(
            model,
            signal={
                "Setup Confirm": "ENTER_TREND_AI",
                "AI Alignment": "Aligned",
                "Timeframe": "1h",
                "Scan Focus": "Actionable Setups",
                "Direction": "Upside",
            },
        )
        self.assertGreater(snap.delta, 0.0)
        self.assertIn("Archive setup calibration", snap.note)

    def test_setup_calibration_stays_off_when_archive_is_thin(self) -> None:
        thin = self._resolved_history().head(16).copy()
        model = build_setup_calibration_model(thin)
        snap = build_setup_calibration_snapshot(
            model,
            signal={
                "Setup Confirm": "PROBE",
                "AI Alignment": "Aligned",
                "Timeframe": "1h",
                "Scan Focus": "Actionable Setups",
                "Direction": "Upside",
            },
        )
        self.assertEqual(snap.delta, 0.0)
        self.assertEqual(snap.note, "")

    def test_actionable_ranking_snapshot_supports_strong_probe_cohort(self) -> None:
        enriched = pd.concat([self._resolved_history(), self._resolved_history()], ignore_index=True)
        model = build_actionable_ranking_model(enriched)
        snap = build_actionable_ranking_snapshot(
            model,
            signal={
                "Setup Confirm": "PROBE",
                "AI Alignment": "Aligned",
                "Timeframe": "1h",
                "Scan Focus": "Actionable Setups",
                "Direction": "Upside",
            },
        )
        self.assertGreater(snap.delta, 0.0)
        self.assertIn("Archive ranking", snap.note)

    def test_actionable_ranking_stays_off_when_archive_is_thin(self) -> None:
        thin = self._resolved_history().head(18).copy()
        model = build_actionable_ranking_model(thin)
        snap = build_actionable_ranking_snapshot(
            model,
            signal={
                "Setup Confirm": "PROBE",
                "AI Alignment": "Aligned",
                "Timeframe": "1h",
                "Scan Focus": "Actionable Setups",
                "Direction": "Upside",
            },
        )
        self.assertEqual(snap.delta, 0.0)
        self.assertEqual(snap.note, "")

    def test_risk_sizing_calibration_supports_strong_probe_cohort(self) -> None:
        enriched = pd.concat([self._resolved_history(), self._resolved_history()], ignore_index=True)
        model = build_risk_sizing_calibration_model(enriched)
        snap = build_risk_sizing_calibration_snapshot(
            model,
            signal={
                "Setup Confirm": "PROBE",
                "AI Alignment": "Aligned",
                "Timeframe": "1h",
                "Scan Focus": "Actionable Setups",
                "Direction": "Upside",
            },
        )
        self.assertGreater(snap.delta, 0.0)
        self.assertIn("Archive sizing calibration", snap.note)

    def test_risk_sizing_calibration_softens_weak_skip_cohort(self) -> None:
        enriched = pd.concat([self._resolved_history(), self._resolved_history()], ignore_index=True)
        model = build_risk_sizing_calibration_model(enriched)
        snap = build_risk_sizing_calibration_snapshot(
            model,
            signal={
                "Setup Confirm": "SKIP",
                "AI Alignment": "Not aligned",
                "Timeframe": "1h",
                "Scan Focus": "Broad Market",
                "Direction": "Downside",
            },
        )
        self.assertLess(snap.delta, 0.0)
        self.assertIn("cautious", snap.note.lower())

    def test_trade_gate_calibration_supports_strong_tradeable_context(self) -> None:
        enriched = pd.concat([self._resolved_history(), self._resolved_history()], ignore_index=True)
        model = build_trade_gate_calibration_model(enriched)
        snap = build_trade_gate_calibration_snapshot(
            model,
            signal={
                "Trade Gate": "Tradeable",
                "Playbook": "Selective upside rotation",
                "Market Regime": "Alt Rotation",
                "Session": "European (08-16 UTC)",
                "Catalyst Window": "Far / Clear",
            },
        )
        self.assertGreater(snap.delta, 0.0)
        self.assertIn("Archive gate calibration", snap.note)

    def test_scalp_calibration_supports_strong_intraday_probe_cohort(self) -> None:
        enriched = pd.concat([self._resolved_history(), self._resolved_history()], ignore_index=True)
        model = build_scalp_calibration_model(enriched)
        snap = build_scalp_calibration_snapshot(
            model,
            signal={
                "Setup Confirm": "PROBE",
                "AI Alignment": "Aligned",
                "Timeframe": "1h",
                "Scan Focus": "Actionable Setups",
                "Direction": "Upside",
            },
        )
        self.assertGreater(snap.delta, 0.0)
        self.assertIn("Archive scalp calibration", snap.note)

    def test_scalp_calibration_stays_off_when_archive_is_thin(self) -> None:
        thin = self._resolved_history().head(18).copy()
        model = build_scalp_calibration_model(thin)
        snap = build_scalp_calibration_snapshot(
            model,
            signal={
                "Setup Confirm": "PROBE",
                "AI Alignment": "Aligned",
                "Timeframe": "1h",
                "Scan Focus": "Actionable Setups",
                "Direction": "Upside",
            },
        )
        self.assertEqual(snap.delta, 0.0)
        self.assertEqual(snap.note, "")

    def test_trade_gate_calibration_softens_weak_no_trade_context(self) -> None:
        enriched = pd.concat([self._resolved_history(), self._resolved_history()], ignore_index=True)
        model = build_trade_gate_calibration_model(enriched)
        snap = build_trade_gate_calibration_snapshot(
            model,
            signal={
                "Trade Gate": "No-Trade",
                "Playbook": "Stand aside / mean reversion only",
                "Market Regime": "Range / Chop",
                "Session": "Asian (00-08 UTC)",
                "Catalyst Window": "Blocking (<6h)",
            },
        )
        self.assertLess(snap.delta, 0.0)
        self.assertIn("cautious", snap.note.lower())


if __name__ == "__main__":
    unittest.main()
