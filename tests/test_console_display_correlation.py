"""Tests for vox/console_display.py correlation rendering.

The correlation event enters via augur.nexus.detected and must:
- Render a distinct MEDIUM/HIGH correlation block
- Suppress a previously rendered low-severity one-liner for the primary
  anomaly if the same event is now surfacing as correlated
"""

from __future__ import annotations


from vox.console_display import (
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
        assert SUBJECT_CORRELATION == "augur.nexus.detected"


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


# ---------------------------------------------------------------------------
# render_reflection — Phase 3 per-domain shape + legacy fallback
# ---------------------------------------------------------------------------

from vox.console_display import render_reflection  # noqa: E402


def test_render_reflection_handles_per_domain_precision():
    data = {
        "session_id": "s1",
        "analyses": {
            "precision": {
                "per_domain": {
                    "chess": {"precision_ratio": 0.9, "action": "lower_sigma"},
                    "typing": {"precision_ratio": 0.4, "action": "none"},
                },
                "domains_evaluated": ["chess", "typing"],
            },
            "utility": {"utility_score": 0.8},
            "counterfactual": {"recommendation": "OK"},
            "correlation_tuning": {"rules_evaluated": 0},
            "correlation_window_tuning": {"rules_evaluated": 0},
        },
        "adjustments": {
            "sigma_adjusted": True,
            "sigma_values": {"chess": 1.9, "typing": 2.0},
            "prompt_mutated": False,
            "matrix_mutated": False,
            "windows_tuned": False,
        },
    }
    rendered = render_reflection(data)
    assert "chess" in rendered
    assert "typing" in rendered
    assert "1.9" in rendered  # the per-domain sigma value


def test_render_reflection_handles_empty_per_domain():
    data = {
        "session_id": "s1",
        "analyses": {
            "precision": {"per_domain": {}, "domains_evaluated": []},
            "utility": {"utility_score": 1.0},
            "counterfactual": {"recommendation": ""},
            "correlation_tuning": {"rules_evaluated": 0},
            "correlation_window_tuning": {"rules_evaluated": 0},
        },
        "adjustments": {
            "sigma_adjusted": False,
            "sigma_values": {},
        },
    }
    rendered = render_reflection(data)
    # No precision data → no precision lines in output, but still renders
    assert "session" in rendered.lower() or "reflection" in rendered.lower()


def test_render_reflection_shows_windows_tuned_flag():
    data = {
        "session_id": "s1",
        "analyses": {
            "precision": {"per_domain": {}, "domains_evaluated": []},
            "utility": {"utility_score": 1.0},
            "counterfactual": {"recommendation": ""},
            "correlation_tuning": {"rules_evaluated": 0},
            "correlation_window_tuning": {
                "rules_evaluated": 1,
                "per_rule": {
                    "LOW+LOW": {
                        "action": "tuned",
                        "window_before": 30.0,
                        "window_after": 25.0,
                    }
                },
            },
        },
        "adjustments": {
            "sigma_adjusted": False,
            "sigma_values": {},
            "windows_tuned": True,
        },
    }
    rendered = render_reflection(data)
    assert "LOW+LOW" in rendered
    assert "25" in rendered  # tuned window value


def test_render_reflection_old_single_domain_shape_falls_back_gracefully():
    """Old reflection records (pre-Phase-3) had top-level precision_ratio
    and adjustments[sigma_value]. Renderer should not crash on them."""
    data = {
        "session_id": "old",
        "analyses": {
            "precision": {"precision_ratio": 0.7, "action": "none"},  # OLD shape
            "utility": {"utility_score": 0.8},
            "counterfactual": {"recommendation": ""},
        },
        "adjustments": {
            "sigma_adjusted": True,
            "sigma_value": 2.1,  # OLD singular field
        },
    }
    rendered = render_reflection(data)
    assert isinstance(rendered, str)
    assert len(rendered) > 0
