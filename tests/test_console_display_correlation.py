"""Tests for output/console_display.py correlation rendering.

The correlation event enters via augur.correlation.detected and must:
- Render a distinct MEDIUM/HIGH correlation block
- Suppress a previously rendered low-severity one-liner for the primary
  anomaly if the same event is now surfacing as correlated
"""

from __future__ import annotations


from output.console_display import (
    SUBJECT_CORRELATION,
    dedup_should_suppress,
    render_correlation,
    update_last_rendered,
)


def _chess_anomaly() -> dict:
    return {
        "domain": "chess",
        "entity": "white",
        "severity": "low",
        "value": 47.2,
        "baseline_mean": 8.2,
        "deviation_score": 5.7,
        "context": {"move_san": "Nf3", "move_number": 12},
        "timestamp": "2026-03-17T14:30:00+00:00",
        "player": "white",
        "move": "Nf3",
        "think_time": 47.2,
    }


def _typing_anomaly() -> dict:
    return {
        "domain": "typing",
        "entity": "user",
        "severity": "low",
        "value": 18.0,
        "baseline_mean": 3.5,
        "deviation_score": 4.1,
        "context": {"avg_wpm": 52},
        "timestamp": "2026-03-17T14:29:48+00:00",
    }


def _correlation_payload() -> dict:
    return {
        "primary_anomaly": _chess_anomaly(),
        "correlated_events": [_typing_anomaly()],
        "correlation_found": True,
        "temporal_lag_seconds": 12.0,
        "combined_severity": "MEDIUM",
        "severity_escalated": True,
        "escalation_rule": "LOW+LOW→MEDIUM",
        "escalation_matrix_version": "1.0",
        "timestamp": "2026-03-17T14:30:00+00:00",
    }


class TestSubject:
    def test_correlation_subject_constant(self) -> None:
        assert SUBJECT_CORRELATION == "augur.correlation.detected"


class TestRenderCorrelation:
    def test_contains_both_domains(self) -> None:
        rendered = render_correlation(_correlation_payload())
        assert "chess" in rendered.lower()
        assert "typing" in rendered.lower()

    def test_contains_combined_severity(self) -> None:
        rendered = render_correlation(_correlation_payload())
        assert "MEDIUM" in rendered

    def test_contains_escalation_rule_label(self) -> None:
        rendered = render_correlation(_correlation_payload())
        assert "LOW+LOW" in rendered or "escalated" in rendered.lower()

    def test_pass_through_renders_single_domain_block(self) -> None:
        payload = {
            "primary_anomaly": _chess_anomaly() | {"severity": "high"},
            "correlated_events": [],
            "correlation_found": False,
            "temporal_lag_seconds": None,
            "combined_severity": "HIGH",
            "severity_escalated": False,
            "escalation_rule": None,
            "escalation_matrix_version": None,
            "timestamp": "2026-03-17T14:30:00+00:00",
        }
        rendered = render_correlation(payload)
        assert "chess" in rendered.lower()
        assert "HIGH" in rendered


class TestDedup:
    def test_suppress_when_primary_matches_last_rendered_for_domain(self) -> None:
        last: dict = {}
        update_last_rendered(last, _chess_anomaly())
        assert dedup_should_suppress(last, _chess_anomaly()) is True

    def test_no_suppress_when_timestamp_differs(self) -> None:
        last: dict = {}
        update_last_rendered(last, _chess_anomaly())
        newer = _chess_anomaly() | {"timestamp": "2026-03-17T14:30:05+00:00"}
        assert dedup_should_suppress(last, newer) is False

    def test_no_suppress_when_entity_differs(self) -> None:
        last: dict = {}
        update_last_rendered(last, _chess_anomaly())
        other_entity = _chess_anomaly() | {"entity": "black"}
        assert dedup_should_suppress(last, other_entity) is False

    def test_no_suppress_when_domain_never_rendered(self) -> None:
        last: dict = {}
        assert dedup_should_suppress(last, _typing_anomaly()) is False
