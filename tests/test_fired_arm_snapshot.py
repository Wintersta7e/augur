"""Fired-arm decision-time snapshot wiring (spec §1A): _build_advice_event
threads the detector snapshot into the advice payload, and PendingAdvice freezes
+ serializes it."""

from perception.feedback_collector import PendingAdvice
from reasoning.augur_advisor import _build_advice_event


def test_advice_event_carries_snapshot():
    payload = {
        "primary_anomaly": {
            "domain": "typing",
            "entity": "keyboard",
            "value": 1.8,
            "severity": "medium",
            "baseline_mean": 0.9,
            "baseline_std": 0.3,
            "deviation_score": 3.0,
            "baseline_observation_count": 40,
            "timestamp": "2026-06-09T00:00:00+00:00",
        },
        "combined_severity": "medium",
    }
    ev = _build_advice_event(payload, "advice", "qwen2.5:32b")
    assert ev["baseline_std"] == 0.3
    assert ev["deviation_score"] == 3.0
    assert ev["baseline_observation_count"] == 40


def test_pending_advice_freezes_snapshot():
    p = PendingAdvice(
        advice_id="a1",
        domain="typing",
        entity="keyboard",
        severity="medium",
        baseline_mean=0.9,
        timestamp="t",
        baseline_std=0.3,
        deviation_at_decision=3.0,
        baseline_observation_count=40,
    )
    assert p.baseline_std == 0.3
    assert p.deviation_at_decision == 3.0
    rec = p.to_record()
    assert rec["baseline_std_at_time"] == 0.3
    assert rec["deviation_at_decision"] == 3.0
    assert rec["baseline_observation_count"] == 40
    assert rec["outcome_metric_version"] == 2
    assert rec["unmeasurable"] is False
