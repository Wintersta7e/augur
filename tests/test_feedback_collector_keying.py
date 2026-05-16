"""Behavioral test: active_tracking is keyed by (domain, entity).

Verifies that PendingAdvice records under (activity_focus, code) and
(activity_intensity, code) do not collide and each advances independently
when perception events for their domain arrive.
"""

from __future__ import annotations

from blackboard.contracts import PerceptionEvent


def test_active_tracking_handles_overlapping_entities_across_domains():
    """Two domains, same entity name. The (domain, entity) keying must
    keep their PendingAdvice records distinct."""
    from perception.feedback_collector import PendingAdvice

    # Hand-construct the active_tracking dict the way the run() closure does.
    active_tracking: dict[tuple[str, str], PendingAdvice] = {}

    a = PendingAdvice(
        advice_id="adv-focus",
        domain="activity_focus",
        entity="code",
        severity="HIGH",
        baseline_mean=3.2,
        timestamp="2026-05-16T12:00:00+00:00",
        correlation_found=False,
        correlated_domains=[],
        rule_key=None,
        escalation_rule=None,
        involved_domains=["activity_focus"],
        temporal_lag_seconds=None,
        correlation_span_s=None,
        rule_window_s=None,
    )
    b = PendingAdvice(
        advice_id="adv-intensity",
        domain="activity_intensity",
        entity="code",
        severity="HIGH",
        baseline_mean=60.0,
        timestamp="2026-05-16T12:00:01+00:00",
        correlation_found=False,
        correlated_domains=[],
        rule_key=None,
        escalation_rule=None,
        involved_domains=["activity_intensity"],
        temporal_lag_seconds=None,
        correlation_span_s=None,
        rule_window_s=None,
    )

    active_tracking[("activity_focus", "code")] = a
    active_tracking[("activity_intensity", "code")] = b

    # Two distinct records exist — no overwrite.
    assert len(active_tracking) == 2
    assert active_tracking[("activity_focus", "code")] is a
    assert active_tracking[("activity_intensity", "code")] is b

    # A perception event with domain=activity_focus advances only `a`.
    focus_event = PerceptionEvent(
        domain="activity_focus",
        stream_id="activity_focus",
        entity="code",
        event_type="focus_change",
        value=5.5,
        unit="log1p_seconds",
        context={},
        timestamp="2026-05-16T12:00:02+00:00",
        session_id="sess-1",
    )

    pending_a = active_tracking.get((focus_event.domain, focus_event.entity))
    assert pending_a is a
    pending_a.add_post_move(focus_event.value)
    assert len(a.think_times_after) == 1
    assert len(b.think_times_after) == 0

    # A perception event with domain=activity_intensity advances only `b`.
    intensity_event = PerceptionEvent(
        domain="activity_intensity",
        stream_id="activity_intensity",
        entity="code",
        event_type="intensity_sample",
        value=80.0,
        unit="ipm",
        context={},
        timestamp="2026-05-16T12:00:03+00:00",
        session_id="sess-1",
    )

    pending_b = active_tracking.get((intensity_event.domain, intensity_event.entity))
    assert pending_b is b
    pending_b.add_post_move(intensity_event.value)
    assert len(a.think_times_after) == 1  # unchanged
    assert len(b.think_times_after) == 1  # advanced
