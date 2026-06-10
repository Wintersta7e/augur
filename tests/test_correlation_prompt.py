"""Tests for advisor correlation-path functions.

describe_signal, build_correlation_prompt, and the routing change on
payload.correlation_found.
"""

from __future__ import annotations

from consilium.advisor import (
    SEVERITY_GATE,
    SUBSCRIBE_SUBJECT,
    build_correlation_prompt,
    describe_signal,
)


def _chess_event() -> dict:
    return {
        "domain": "chess",
        "entity": "white",
        "event_type": "move",
        "value": 47.2,
        "unit": "seconds",
        "context": {"move_san": "Nf3", "move_number": 12},
        "baseline_mean": 8.2,
        "baseline_std": 2.1,
        "deviation_score": 5.7,
        "severity": "low",
    }


def _typing_event() -> dict:
    return {
        "domain": "typing",
        "entity": "user",
        "event_type": "pause",
        "value": 18.0,
        "unit": "seconds",
        "context": {"wpm_before": 65, "wpm_after": 39, "avg_wpm": 52},
        "baseline_mean": 3.5,
        "baseline_std": 1.2,
        "deviation_score": 4.1,
        "severity": "low",
    }


class TestDescribeSignal:
    def test_chess_mentions_move_thinktime_baseline_deviation(self) -> None:
        line = describe_signal("chess", _chess_event())
        assert "Nf3" in line
        assert "47" in line  # think time
        assert "8.2" in line  # baseline
        assert "5.7" in line  # deviation

    def test_chess_one_line(self) -> None:
        line = describe_signal("chess", _chess_event())
        assert "\n" not in line

    def test_typing_mentions_pause_and_wpm(self) -> None:
        line = describe_signal("typing", _typing_event())
        assert "18" in line  # pause seconds
        assert "52" in line or "wpm" in line.lower()

    def test_typing_one_line(self) -> None:
        line = describe_signal("typing", _typing_event())
        assert "\n" not in line

    def test_unknown_domain_fallback(self) -> None:
        event = {
            "domain": "focus",
            "entity": "app",
            "event_type": "switch",
            "value": 5,
            "unit": "count",
            "context": {},
            "baseline_mean": 1,
            "deviation_score": 3.0,
            "severity": "low",
        }
        line = describe_signal("focus", event)
        assert "focus" in line.lower()
        assert "\n" not in line


def _correlation_payload() -> dict:
    return {
        "primary_anomaly": _chess_event() | {"timestamp": "2026-03-17T14:30:00+00:00"},
        "correlated_events": [
            _typing_event() | {"timestamp": "2026-03-17T14:29:48+00:00"}
        ],
        "correlation_found": True,
        "temporal_lag_seconds": 12.0,
        "combined_severity": "MEDIUM",
        "severity_escalated": True,
        "escalation_rule": "LOW+LOW→MEDIUM",
        "escalation_matrix_version": "1.0",
        "timestamp": "2026-03-17T14:30:00+00:00",
    }


class TestBuildCorrelationPrompt:
    def test_contains_both_domain_one_liners(self) -> None:
        prompt = build_correlation_prompt(_correlation_payload())
        assert "CHESS" in prompt
        assert "TYPING" in prompt

    def test_mentions_temporal_lag(self) -> None:
        prompt = build_correlation_prompt(_correlation_payload())
        assert "12" in prompt

    def test_mentions_combined_severity(self) -> None:
        prompt = build_correlation_prompt(_correlation_payload())
        assert "MEDIUM" in prompt

    def test_mentions_escalation_rule(self) -> None:
        prompt = build_correlation_prompt(_correlation_payload())
        assert "LOW" in prompt  # escalated from LOW+LOW

    def test_asks_for_relational_reasoning_not_sum(self) -> None:
        prompt = build_correlation_prompt(_correlation_payload())
        # The correlation prompt's value is forcing the LLM to reason about
        # the combination — not the two signals in isolation.
        assert "combination" in prompt.lower() or "combined" in prompt.lower()

    def test_handles_three_domain_correlation(self) -> None:
        payload = _correlation_payload()
        payload["correlated_events"].append(
            {
                "domain": "focus",
                "entity": "app",
                "event_type": "switch",
                "value": 5,
                "unit": "count",
                "context": {},
                "baseline_mean": 1,
                "deviation_score": 3.0,
                "severity": "low",
                "timestamp": "2026-03-17T14:29:40+00:00",
            }
        )
        prompt = build_correlation_prompt(payload)
        assert "CHESS" in prompt
        assert "TYPING" in prompt
        assert "FOCUS" in prompt


class TestAdvisorSubscriptionAndGate:
    def test_subscribes_to_correlation_subject(self) -> None:
        assert SUBSCRIBE_SUBJECT == "augur.nexus.detected"

    def test_severity_gate_is_lowercase(self) -> None:
        # The gate itself stays lowercase; the advisor lowercases
        # the uppercase combined_severity before comparing.
        assert "medium" in SEVERITY_GATE
        assert "high" in SEVERITY_GATE
        assert "low" not in SEVERITY_GATE


class TestResolveAdvisorPath:
    """Tests the routing branch used by the advisor's on_message callback.

    We test a pure helper — resolve_advisor_path(payload) — that returns
    either 'correlation' or 'single' so the async callback logic is unit-
    testable without NATS.
    """

    def test_correlation_found_true_routes_to_correlation(self) -> None:
        from consilium.advisor import resolve_advisor_path

        assert (
            resolve_advisor_path({"correlation_found": True, "primary_anomaly": {}})
            == "correlation"
        )

    def test_correlation_found_false_routes_to_single(self) -> None:
        from consilium.advisor import resolve_advisor_path

        assert (
            resolve_advisor_path({"correlation_found": False, "primary_anomaly": {}})
            == "single"
        )

    def test_missing_flag_defaults_to_single(self) -> None:
        from consilium.advisor import resolve_advisor_path

        assert resolve_advisor_path({"primary_anomaly": {}}) == "single"
