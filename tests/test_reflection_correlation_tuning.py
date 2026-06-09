"""Unit tests for analyze_correlation_tuning.

The pure function that implements per-rule EWMA confidence with
hysteresis thresholds. All tests use deterministic fabricated
feedback records and explicit expected values.
"""

from __future__ import annotations

import dataclasses

from blackboard.config import AugurConfig
from reasoning.correlator import DEFAULT_ESCALATION_MATRIX
from reasoning.reflection_engine import analyze_correlation_tuning

CONFIG = AugurConfig()


def _advice(
    rule_key: str | None,
    explicit: str = "no_response",
    behavioral: float = 0.0,
    correlation_found: bool = True,
    escalation_rule: str | None = None,
    behavioral_finalized: bool = True,
    unmeasurable: bool = False,
) -> dict:
    return {
        "advice_id": "adv",
        "domain": "multi",
        "entity": "chess+typing",
        "severity": "medium",
        "explicit_rating": explicit,
        "behavioral_score": behavioral,
        "behavioral_finalized": behavioral_finalized,
        "unmeasurable": unmeasurable,
        "think_times_after": [],
        "baseline_mean_at_time": 5.0,
        "timestamp": "2026-04-09T12:00:00+00:00",
        "correlation_found": correlation_found,
        "correlated_domains": ["typing"] if correlation_found else [],
        "rule_key": rule_key,
        "escalation_rule": escalation_rule,
    }


def _feedback(events: list[dict]) -> dict:
    return {
        "session_id": "sess-test",
        "advice_events": events,
        "session_summary": {
            "total_advice": len(events),
            "explicit_positive": sum(1 for e in events if e["explicit_rating"] == "y"),
            "explicit_negative": sum(1 for e in events if e["explicit_rating"] == "n"),
        },
    }


class TestEmptyAndNonCorrelated:
    def test_empty_feedback_produces_no_updates(self) -> None:
        result = analyze_correlation_tuning(
            _feedback([]), DEFAULT_ESCALATION_MATRIX, {}, CONFIG
        )
        assert result["rules_evaluated"] == 0
        assert result["per_rule"] == {}
        assert result["new_matrix"] is None
        assert result["new_confidence_state"] == {}

    def test_single_domain_only_session_produces_no_updates(self) -> None:
        events = [
            _advice(rule_key=None, correlation_found=False, explicit="n"),
            _advice(rule_key=None, correlation_found=False, explicit="n"),
        ]
        result = analyze_correlation_tuning(
            _feedback(events), DEFAULT_ESCALATION_MATRIX, {}, CONFIG
        )
        assert result["rules_evaluated"] == 0
        assert result["new_matrix"] is None


class TestFirstObservation:
    def test_first_observation_starts_at_one_and_drops_with_bad_feedback(self) -> None:
        # One LOW+LOW event, negative rating, no behavioral data (unfinalized →
        # excluded, behavioral_avg defaults to 0.5) → utility = 0.2
        # EWMA: (1-0.2)*1.0 + 0.2*0.2 = 0.84
        events = [
            _advice(
                rule_key="LOW+LOW",
                explicit="n",
                behavioral=0.0,
                behavioral_finalized=False,
            )
        ]
        result = analyze_correlation_tuning(
            _feedback(events), DEFAULT_ESCALATION_MATRIX, {}, CONFIG
        )
        assert result["rules_evaluated"] == 1
        per = result["per_rule"]["LOW+LOW"]
        assert per["event_count"] == 1
        assert per["confidence_before"] == 1.0
        assert per["confidence_after"] == 0.84
        assert per["target_before"] == "MEDIUM"
        assert per["target_after"] == "MEDIUM"
        assert per["action"] == "tracked"
        assert result["new_matrix"] is None
        assert result["new_confidence_state"]["LOW+LOW"]["confidence"] == 0.84
        assert result["new_confidence_state"]["LOW+LOW"]["restore_target"] == "MEDIUM"

    def test_good_first_observation_keeps_confidence_at_one(self) -> None:
        # utility = 0.6*1.0 + 0.4*1.0 = 1.0; EWMA: 0.8*1.0 + 0.2*1.0 = 1.0
        events = [_advice(rule_key="LOW+LOW", explicit="y", behavioral=1.0)]
        result = analyze_correlation_tuning(
            _feedback(events), DEFAULT_ESCALATION_MATRIX, {}, CONFIG
        )
        per = result["per_rule"]["LOW+LOW"]
        assert per["confidence_after"] == 1.0
        assert per["target_after"] == "MEDIUM"


class TestMultipleFiringsAveraged:
    def test_three_events_one_ewma_update(self) -> None:
        # Three events with explicit [n, no_response, n] and behavioral [0.2, 0.3, 0.1]
        # explicit_avg = (0.0 + 0.5 + 0.0) / 3 = 0.1667
        # behavioral_avg = (0.2 + 0.3 + 0.1) / 3 = 0.2 (all > 0 so all counted)
        # session_utility = 0.6*0.1667 + 0.4*0.2 = 0.1 + 0.08 = 0.18
        # EWMA: (1-0.2)*1.0 + 0.2*0.18 = 0.836
        events = [
            _advice(rule_key="LOW+LOW", explicit="n", behavioral=0.2),
            _advice(rule_key="LOW+LOW", explicit="no_response", behavioral=0.3),
            _advice(rule_key="LOW+LOW", explicit="n", behavioral=0.1),
        ]
        result = analyze_correlation_tuning(
            _feedback(events), DEFAULT_ESCALATION_MATRIX, {}, CONFIG
        )
        per = result["per_rule"]["LOW+LOW"]
        assert per["event_count"] == 3
        assert per["confidence_after"] == 0.836


class TestHysteresisBand:
    def test_confidence_in_band_stays_at_current_target(self) -> None:
        # Confidence 0.5, session utility 0.5 → new conf = 0.5 (unchanged)
        # 0.3 <= 0.5 < 0.6 → hysteresis band, target stays
        matrix = {"version": "1.0", "rules": {"LOW+LOW": "MEDIUM"}}
        confidence_state = {"LOW+LOW": {"confidence": 0.5, "restore_target": "MEDIUM"}}
        events = [_advice(rule_key="LOW+LOW", explicit="no_response", behavioral=0.5)]
        result = analyze_correlation_tuning(
            _feedback(events), matrix, confidence_state, CONFIG
        )
        per = result["per_rule"]["LOW+LOW"]
        # explicit_avg=0.5, behavioral_avg=0.5, session_utility=0.5
        # EWMA: (1-0.2)*0.5 + 0.2*0.5 = 0.5
        assert per["confidence_after"] == 0.5
        assert per["target_before"] == "MEDIUM"
        assert per["target_after"] == "MEDIUM"
        assert per["action"] == "tracked"
        assert result["new_matrix"] is None
        assert per["restore_target_after"] == "MEDIUM"


class TestCrossingDisableThreshold:
    def test_crosses_disable_and_flips_target_to_low(self) -> None:
        # conf 0.32, utility 0.1 → new = (1-0.2)*0.32 + 0.2*0.1 = 0.276, below 0.3
        matrix = {"version": "1.0", "rules": {"LOW+LOW": "MEDIUM"}}
        confidence_state = {"LOW+LOW": {"confidence": 0.32, "restore_target": "MEDIUM"}}
        # utility = 0.6*0 + 0.4*0.25 = 0.1
        events = [_advice(rule_key="LOW+LOW", explicit="n", behavioral=0.25)]
        result = analyze_correlation_tuning(
            _feedback(events), matrix, confidence_state, CONFIG
        )
        per = result["per_rule"]["LOW+LOW"]
        assert per["confidence_after"] == 0.276
        assert per["target_before"] == "MEDIUM"
        assert per["target_after"] == "LOW"
        assert per["action"] == "disabled"
        assert per["restore_target_after"] == "MEDIUM"
        assert result["new_matrix"] is not None
        assert result["new_matrix"]["rules"]["LOW+LOW"] == "LOW"


class TestCrossingEnableThresholdRestoresSnapshot:
    def test_disabled_rule_recovers_to_snapshot_not_default(self) -> None:
        # Codex concern #2 correctness test:
        # A rule was manually set to HIGH via MCP, then disabled by bad feedback.
        # When confidence recovers, it should restore to HIGH (the snapshot),
        # NOT to MEDIUM (the hardcoded default).
        matrix = {"version": "1.0", "rules": {"LOW+LOW": "LOW"}}
        confidence_state = {"LOW+LOW": {"confidence": 0.55, "restore_target": "HIGH"}}
        # utility = 0.6*1.0 + 0.4*1.0 = 1.0
        # EWMA: (1-0.2)*0.55 + 0.2*1.0 = 0.64 ≥ 0.6 → enabled
        events = [_advice(rule_key="LOW+LOW", explicit="y", behavioral=1.0)]
        result = analyze_correlation_tuning(
            _feedback(events), matrix, confidence_state, CONFIG
        )
        per = result["per_rule"]["LOW+LOW"]
        assert per["confidence_after"] == 0.64
        assert per["target_before"] == "LOW"
        assert per["target_after"] == "HIGH"
        assert per["action"] == "re-enabled"
        assert result["new_matrix"]["rules"]["LOW+LOW"] == "HIGH"


class TestManualEditRefreshesRestoreTarget:
    def test_healthy_rule_refreshes_restore_target_to_current_matrix_value(
        self,
    ) -> None:
        # Rule is healthy (conf 0.9), matrix currently has LOW+LOW→HIGH (manual edit).
        # Positive session keeps confidence high. restore_target should refresh to HIGH.
        matrix = {"version": "1.0", "rules": {"LOW+LOW": "HIGH"}}
        confidence_state = {"LOW+LOW": {"confidence": 0.9, "restore_target": "MEDIUM"}}
        events = [_advice(rule_key="LOW+LOW", explicit="y", behavioral=1.0)]
        result = analyze_correlation_tuning(
            _feedback(events), matrix, confidence_state, CONFIG
        )
        per = result["per_rule"]["LOW+LOW"]
        # EWMA: 0.8*0.9 + 0.2*1.0 = 0.92
        assert per["confidence_after"] == 0.92
        assert per["target_after"] == "HIGH"
        assert per["restore_target_after"] == "HIGH"


class TestMatrixMissAttribution:
    def test_matrix_miss_event_still_attributed_via_rule_key(self) -> None:
        # Codex concern #4 correctness test:
        # Event with escalation_rule=None but valid rule_key → STILL included.
        matrix = {"version": "1.0", "rules": {}}
        events = [
            _advice(
                rule_key="LOW+LOW",
                explicit="n",
                behavioral=0.0,
                escalation_rule=None,
            ),
        ]
        result = analyze_correlation_tuning(_feedback(events), matrix, {}, CONFIG)
        assert result["rules_evaluated"] == 1
        assert "LOW+LOW" in result["per_rule"]


class TestNullRuleKey:
    def test_null_rule_key_events_skipped(self) -> None:
        events = [
            _advice(rule_key=None, explicit="n"),
            _advice(rule_key=None, explicit="n"),
        ]
        result = analyze_correlation_tuning(
            _feedback(events), DEFAULT_ESCALATION_MATRIX, {}, CONFIG
        )
        assert result["rules_evaluated"] == 0


class TestConfigDisabled:
    def test_short_circuit_when_disabled(self) -> None:
        cfg = dataclasses.replace(AugurConfig(), correlation_tuning_enabled=False)
        events = [_advice(rule_key="LOW+LOW", explicit="n")]
        result = analyze_correlation_tuning(
            _feedback(events), DEFAULT_ESCALATION_MATRIX, {}, cfg
        )
        assert result.get("disabled") is True
        assert result["analysis"] == "correlation_tuning"


class TestEwmaProgressionPinnedAlpha02:
    def test_six_bad_sessions_crosses_disable(self) -> None:
        # Expected rounded progression: 1.0 → 0.8 → 0.64 → 0.512 → 0.41 → 0.328 → 0.262
        # Use behavioral=0.01 to make behavioral_avg ≈ 0 (avoiding the 0.5 default)
        # utility ≈ 0.6*0 + 0.4*0.01 = 0.004 ≈ 0
        matrix = {"version": "1.0", "rules": {"LOW+LOW": "MEDIUM"}}
        state: dict = {}
        expected = [0.8, 0.64, 0.512, 0.41, 0.328, 0.262]
        events = [_advice(rule_key="LOW+LOW", explicit="n", behavioral=0.01)]
        for i, exp_conf in enumerate(expected):
            result = analyze_correlation_tuning(
                _feedback(events), matrix, state, CONFIG
            )
            per = result["per_rule"]["LOW+LOW"]
            assert abs(per["confidence_after"] - exp_conf) < 0.01, (
                f"Session {i + 1}: expected ~{exp_conf}, got {per['confidence_after']}"
            )
            state = result["new_confidence_state"]
            if result["new_matrix"] is not None:
                matrix = result["new_matrix"]

        # After 6 sessions, rule should be disabled
        assert matrix["rules"]["LOW+LOW"] == "LOW"


class TestTwoRulesPartialUpdate:
    def test_rule_with_no_data_is_copied_unchanged(self) -> None:
        matrix = {
            "version": "1.0",
            "rules": {"LOW+LOW": "MEDIUM", "MEDIUM+MEDIUM": "HIGH"},
        }
        state = {
            "LOW+LOW": {"confidence": 1.0, "restore_target": "MEDIUM"},
            "MEDIUM+MEDIUM": {"confidence": 0.7, "restore_target": "HIGH"},
        }
        events = [_advice(rule_key="LOW+LOW", explicit="y", behavioral=1.0)]
        result = analyze_correlation_tuning(_feedback(events), matrix, state, CONFIG)
        assert result["rules_evaluated"] == 1
        assert "LOW+LOW" in result["per_rule"]
        assert "MEDIUM+MEDIUM" not in result["per_rule"]
        # MEDIUM+MEDIUM state copied through byte-for-byte
        assert result["new_confidence_state"]["MEDIUM+MEDIUM"] == state["MEDIUM+MEDIUM"]


class TestDisableCapturesRestoreIfNone:
    def test_first_session_disable_captures_current_target(self) -> None:
        matrix = {"version": "1.0", "rules": {"LOW+LOW": "MEDIUM"}}
        state = {"LOW+LOW": {"confidence": 0.32, "restore_target": None}}
        # utility = 0.6*0 + 0.4*0.5 = 0.2 (no behavioral outcome — unfinalized →
        # excluded, behavioral_avg defaults to 0.5)
        # conf: 0.8*0.32 + 0.2*0.2 = 0.296, below 0.3 → disable
        events = [
            _advice(
                rule_key="LOW+LOW",
                explicit="n",
                behavioral=0.0,
                behavioral_finalized=False,
            )
        ]
        result = analyze_correlation_tuning(_feedback(events), matrix, state, CONFIG)
        per = result["per_rule"]["LOW+LOW"]
        assert per["confidence_after"] == 0.296
        assert per["target_after"] == "LOW"
        assert per["action"] == "disabled"
        # restore_target captured from current matrix target since prev was None
        assert per["restore_target_after"] == "MEDIUM"
