"""Coverage pins surfaced during the deep-review fix loop.

Each test below exists specifically to prevent a future refactor from
silently reintroducing a class of bug that the Round 1 reviewers flagged.
IDs reference the ISSUES.md audit trail.
"""

from __future__ import annotations

import dataclasses
import json
from unittest.mock import MagicMock


from tabula.config import AugurConfig
from nexus.correlator import (
    DEFAULT_ESCALATION_MATRIX,
    _build_correlation_payload,
    _build_passthrough_payload,
    correlate,
)
from reasoning.reflection_engine import (
    analyze_correlation_tuning,
    analyze_utility,
)

CONFIG = AugurConfig()


# ---------------------------------------------------------------------------
# Helpers (duplicated from the existing test files to keep this file
# self-contained; a future refactor could extract them into a shared
# conftest fixture).
# ---------------------------------------------------------------------------


def _advice(
    rule_key: str | None,
    explicit: str = "no_response",
    behavioral: float = 0.0,
    correlation_found: bool = True,
    escalation_rule: str | None = None,
) -> dict:
    return {
        "advice_id": "adv",
        "domain": "multi",
        "entity": "chess+typing",
        "severity": "medium",
        "explicit_rating": explicit,
        "behavioral_score": behavioral,
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
        "session_summary": {"total_advice": len(events)},
    }


def _make_anomaly(
    domain: str,
    entity: str,
    severity: str,
    ts_iso: str,
    value: float = 1.0,
) -> dict:
    return {
        "domain": domain,
        "stream_id": f"{domain}_stream",
        "entity": entity,
        "event_type": "test",
        "value": value,
        "unit": "s",
        "context": {},
        "session_id": "sess-cov",
        "baseline_mean": 0.0,
        "baseline_std": 1.0,
        "deviation_score": 2.5,
        "anomaly_score": 0.5,
        "severity": severity,
        "timestamp": ts_iso,
    }


# ---------------------------------------------------------------------------
# COV-01: analyze_correlation_tuning disabled-path shape
# ---------------------------------------------------------------------------


class TestCOV01DisabledPathShape:
    """Pin the disabled return shape so a future refactor cannot silently
    add ``new_confidence_state`` or ``rules_evaluated`` keys that would
    change the caller contract in run_reflection."""

    def test_disabled_result_has_no_new_confidence_state(self) -> None:
        cfg = dataclasses.replace(AugurConfig(), correlation_tuning_enabled=False)
        result = analyze_correlation_tuning(
            _feedback([_advice(rule_key="LOW+LOW", explicit="y")]),
            DEFAULT_ESCALATION_MATRIX,
            {},
            cfg,
        )
        assert result.get("disabled") is True
        assert "new_confidence_state" not in result
        assert "rules_evaluated" not in result
        assert "per_rule" not in result
        assert "new_matrix" not in result


# ---------------------------------------------------------------------------
# COV-02: rule_key missing from the dict entirely, not just set to None
# ---------------------------------------------------------------------------


class TestCOV02MissingRuleKeyAttribute:
    """A feedback record that predates the Phase 3B+ fields may have no
    ``rule_key`` key at all. The filter uses .get("rule_key") which
    returns None for missing keys, so the event should be skipped the
    same way an explicit None is."""

    def test_event_without_rule_key_field_is_skipped(self) -> None:
        ev = _advice(rule_key=None)
        del ev["rule_key"]  # key not present at all
        result = analyze_correlation_tuning(
            _feedback([ev]), DEFAULT_ESCALATION_MATRIX, {}, CONFIG
        )
        assert result["rules_evaluated"] == 0


# ---------------------------------------------------------------------------
# COV-03: feedback dict without "advice_events" key at all
# ---------------------------------------------------------------------------


class TestCOV03FeedbackMissingAdviceEvents:
    """run_reflection can pass through a feedback record that has no
    advice_events key (e.g., a structurally-minimal fabrication). The
    function should produce a rules_evaluated=0 no-op, not crash."""

    def test_missing_advice_events_key(self) -> None:
        feedback = {"session_id": "s"}  # no advice_events
        result = analyze_correlation_tuning(
            feedback, DEFAULT_ESCALATION_MATRIX, {}, CONFIG
        )
        assert result["rules_evaluated"] == 0
        assert result["new_matrix"] is None


# ---------------------------------------------------------------------------
# COV-04: current_target already "LOW" + prev_restore None
# ---------------------------------------------------------------------------


class TestCOV04DisableWithCurrentLowNoSnapshot:
    """Pathological case: the matrix has LOW+LOW→LOW (rule was manually
    disabled via MCP) AND there is no prior restore_target snapshot.
    Bad feedback must not crash, and must leave restore_target=None so
    the caller knows recovery needs explicit intervention."""

    def test_disable_of_already_low_with_no_snapshot(self) -> None:
        matrix = {"version": "1.0", "rules": {"LOW+LOW": "LOW"}}
        state = {"LOW+LOW": {"confidence": 0.32, "restore_target": None}}
        events = [_advice(rule_key="LOW+LOW", explicit="n", behavioral=0.0)]
        result = analyze_correlation_tuning(_feedback(events), matrix, state, CONFIG)

        per = result["per_rule"]["LOW+LOW"]
        # Utility = 0.6*0 + 0.4*0.5 = 0.2; EWMA 0.8*0.32 + 0.2*0.2 = 0.296 < 0.3
        assert per["confidence_after"] == 0.296
        # Still "LOW" → "LOW" is not a state transition; action is "tracked"
        assert per["target_before"] == "LOW"
        assert per["target_after"] == "LOW"
        assert per["action"] == "tracked"
        # No snapshot to preserve, no new snapshot captured (can't snapshot
        # LOW as a restore target — that would be an oxymoron).
        assert per["restore_target_after"] is None
        # Matrix unchanged
        assert result["new_matrix"] is None


# ---------------------------------------------------------------------------
# COV-05: analyze_utility uses the filtered count for `total`, not
# session_summary["total_advice"]
# ---------------------------------------------------------------------------


class TestCOV05UtilityTotalUsesFilteredCount:
    """Guard against a future refactor that accidentally restores the old
    ``total = summary.get("total_advice")`` denominator. With 1 standalone
    event mixed with 3 correlated events, the mutation-threshold guard
    (``total >= 2``) must use the filtered count (1), not the summary
    count (4)."""

    def test_mutation_threshold_uses_filtered_count(self) -> None:
        feedback = {
            "advice_events": [
                # 3 correlated events — all filtered out
                {
                    "explicit_rating": "n",
                    "behavioral_score": 0.0,
                    "correlation_found": True,
                },
                {
                    "explicit_rating": "n",
                    "behavioral_score": 0.0,
                    "correlation_found": True,
                },
                {
                    "explicit_rating": "n",
                    "behavioral_score": 0.0,
                    "correlation_found": True,
                },
                # 1 standalone event — the only one that contributes
                {
                    "explicit_rating": "n",
                    "behavioral_score": 0.0,
                    "correlation_found": False,
                },
            ],
            "session_summary": {"total_advice": 4},  # intentionally stale
        }
        result = analyze_utility(feedback, CONFIG)
        # With 1 filtered event, total < 2, needs_mutation must be False
        # regardless of how poor the utility is.
        assert result["needs_prompt_mutation"] is False


# ---------------------------------------------------------------------------
# COV-07: rule_key driver-vs-closest divergence
# ---------------------------------------------------------------------------


class TestCOV07RuleKeyUsesDriverNotClosest:
    """When the correlator's sliding window contains multiple events of
    different severities at different temporal distances, rule_key must
    include all surviving correlated events (N-way) and temporal_lag must
    reflect the *closest* correlated event."""

    def test_nway_rule_key_and_closest_temporal_lag(self) -> None:
        # Primary: LOW at T=30s
        primary = _make_anomaly("chess", "white", "low", "2026-03-17T14:30:00+00:00")
        # Closest: LOW at T=29:55 (5s lag) — same severity as primary
        closest_low = _make_anomaly(
            "typing", "user", "low", "2026-03-17T14:29:55+00:00"
        )
        # Higher severity event at T=29:35 (25s lag) — both survive
        # pairwise filter (both < 30s default window).
        driver_medium = _make_anomaly(
            "focus", "app", "medium", "2026-03-17T14:29:35+00:00"
        )

        mock_redis = MagicMock()
        mock_redis.zrangebyscore.return_value = [
            json.dumps(closest_low).encode(),
            json.dumps(driver_medium).encode(),
            json.dumps(primary).encode(),
        ]

        result = correlate(primary, mock_redis, DEFAULT_ESCALATION_MATRIX, CONFIG)
        assert result is not None
        assert result["correlation_found"] is True
        # N-way: primary=LOW + typing=LOW + focus=MEDIUM → "LOW+LOW+MEDIUM"
        assert result["rule_key"] == "LOW+LOW+MEDIUM"
        # temporal_lag = min(5s, 25s) = closest = ~5s
        assert 4.0 < result["temporal_lag_seconds"] < 6.0


# ---------------------------------------------------------------------------
# COV-10: behavioral_avg default (0.5) when no positive behavioral scores
# ---------------------------------------------------------------------------


class TestCOV10BehavioralAvgDefault:
    """Pin the default behavioral_avg=0.5 when every event has
    behavioral_score <= 0 (excluded by the ``> 0`` filter). Separate
    from the EWMA progression test (which uses behavioral=0.01 to avoid
    the default)."""

    def test_zero_behavioral_scores_produce_default_half(self) -> None:
        # One event with explicit="y" (1.0) and behavioral=0.0 (excluded)
        # session_utility = 0.6 * 1.0 + 0.4 * 0.5 = 0.8
        events = [_advice(rule_key="LOW+LOW", explicit="y", behavioral=0.0)]
        result = analyze_correlation_tuning(
            _feedback(events), DEFAULT_ESCALATION_MATRIX, {}, CONFIG
        )
        per = result["per_rule"]["LOW+LOW"]
        assert per["session_utility"] == 0.8


# ---------------------------------------------------------------------------
# COV-11: primary_anomaly is passed through the payload unchanged (dict eq)
# ---------------------------------------------------------------------------


class TestCOV11PrimaryAnomalyFullPassthrough:
    """The correlator's _build_correlation_payload must not mutate or
    reconstruct primary_anomaly — downstream consumers rely on every
    field of the original anomaly being present, not just the common
    ones (domain, entity, severity)."""

    def test_build_correlation_payload_full_passthrough(self) -> None:
        primary = _make_anomaly("chess", "white", "low", "2026-03-17T14:30:00+00:00")
        # Add some extra fields to make sure they survive the passthrough
        primary["custom_context"] = {"deep": {"nested": 42}}
        primary["baseline_std"] = 1.234
        correlated = _make_anomaly("typing", "user", "low", "2026-03-17T14:29:48+00:00")

        result = _build_correlation_payload(
            primary, [correlated], DEFAULT_ESCALATION_MATRIX, CONFIG
        )
        # Full dict equality — every field preserved
        assert result["primary_anomaly"] == primary

    def test_build_passthrough_payload_full_passthrough(self) -> None:
        primary = _make_anomaly("chess", "white", "medium", "2026-03-17T14:30:00+00:00")
        primary["extra_field"] = "preserved"
        result = _build_passthrough_payload(primary)
        assert result["primary_anomaly"] == primary


# ---------------------------------------------------------------------------
# COV-13: feedback collector on_advice field extraction shape
# ---------------------------------------------------------------------------


class TestCOV13FeedbackCollectorAdvicePayloadKeys:
    """Pin the exact field names the feedback collector reads from the
    advice payload so a future advisor schema change gets caught by a
    unit test instead of silently breaking matrix tuning attribution."""

    def test_pending_advice_fields_match_advisor_payload_keys(self) -> None:
        # This is the set of keys the advisor is contractually required
        # to put on the advice payload for the feedback collector to
        # correctly populate PendingAdvice. If any key name drifts,
        # matrix tuning attribution silently breaks.
        from responsum.feedback_collector import PendingAdvice

        advisor_payload = {
            "player": "white",
            "severity": "medium",
            "move": "Nf3",
            "think_time": 12.5,
            "correlation_found": True,
            "correlated_domains": ["typing"],
            "rule_key": "LOW+LOW",
            "escalation_rule": "LOW+LOW\u2192MEDIUM",
        }

        # Mirror on_advice's extraction
        pending = PendingAdvice(
            advice_id="adv-1",
            domain="chess",
            entity=advisor_payload.get("player", "?"),
            severity=advisor_payload.get("severity", "?"),
            baseline_mean=10.0,
            timestamp="2026-04-09T12:00:00+00:00",
            correlation_found=bool(advisor_payload.get("correlation_found", False)),
            correlated_domains=advisor_payload.get("correlated_domains") or [],
            rule_key=advisor_payload.get("rule_key"),
            escalation_rule=advisor_payload.get("escalation_rule"),
        )
        record = pending.to_record()

        assert record["correlation_found"] is True
        assert record["correlated_domains"] == ["typing"]
        assert record["rule_key"] == "LOW+LOW"
        assert record["escalation_rule"] == "LOW+LOW\u2192MEDIUM"
