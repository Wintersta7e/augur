"""Praefectus supervises Praesagium (spec 2026-07-09 §7).

1. "praesagium" joins REQUIRED_FACULTIES: seeded by initial_states, heartbeats
   accepted, liveness tracked like any other required faculty.
2. Consilium's stall deficit must be blind to Praesagium-attributable
   terminals: an anticipatory advice/suppressed/delivery-failure/advice-surface
   violation is a terminal of PRAESAGIUM work, not of a nexus detection, and
   must not paper over a genuinely stalled real detection in the same window.

Carrier fields (pinned from the real publisher payloads, consilium/advisor.py
+ conscientia/screens.py):
  - augur.consilium.advice   -> event["domain"] == "praesagium"
    (advisor.py _build_advice_event: `"domain": domain,` where domain is
    primary_anomaly["domain"] on the single path; PR1b's _clamp_foreseen forces
    primary_anomaly["domain"] == "praesagium" and path == "single".)
  - augur.limen.suppressed / augur.limen.delivery_failure ->
    event["state_key"].startswith("single:praesagium:")
    (advisor.py publish_suppressed_event/publish_delivery_failure_event both
    carry `"state_key": signature.state_key,`; limen/gate.py build_signature
    sets `state_key = f"single:{domain}:{entity}"` on the single path.)
  - augur.conscientia.violation (surface == "advice") ->
    event["domain"] == "praesagium"
    (conscientia/screens.py make_violation: `"domain": domain,`; advisor.py's
    conscientia_finalize_text threads the same `domain` used to build the
    advice/suppressed/delivery_failure events.)
"""

from praefectus import health as H


class _Cfg:
    """Minimal config stand-in for the pure functions (mirrors
    tests/test_praefectus_health.py's _Cfg)."""

    ollama_timeout = 120
    praefectus_heartbeat_interval_s = 10.0
    praefectus_stale_after_s = 30.0
    praefectus_dead_after_s = 90.0
    praefectus_warmup_s = 30.0
    praefectus_stall_tolerance = 1
    praefectus_stall_min_events = 2
    praefectus_delivery_failure_spike = 3
    effective_stall_window_s = 300.0
    effective_reflection_window_s = 300.0


def test_praesagium_is_a_required_faculty():
    assert "praesagium" in H.REQUIRED_FACULTIES


def test_initial_states_seeds_praesagium():
    states = H.initial_states(1000.0)
    assert "praesagium" in states
    assert states["praesagium"].required is True
    assert states["praesagium"].seen is False


def test_record_heartbeat_accepts_praesagium():
    states = H.initial_states(1000.0)
    H.record_heartbeat(states, "praesagium", 1005.0)
    assert states["praesagium"].seen is True
    assert states["praesagium"].last_heartbeat == 1005.0


# ---------------------------------------------------------------------------
# Exclusion matrix: for each terminal type, a praesagium-attributable event
# leaves the window/deficit unchanged, while the equivalent typing-domain
# event still counts. Paired assertions per event type.
# ---------------------------------------------------------------------------


def test_advice_event_praesagium_domain_excluded_typing_domain_counts():
    cfg = _Cfg()

    states_p = H.initial_states(1000.0)
    window_p = H.ActivityWindow()
    H.record_activity(
        states_p,
        window_p,
        "augur.consilium.advice",
        {"domain": "praesagium"},
        1012.0,
        cfg,
    )
    assert window_p.advice == []

    states_t = H.initial_states(1000.0)
    window_t = H.ActivityWindow()
    H.record_activity(
        states_t,
        window_t,
        "augur.consilium.advice",
        {"domain": "typing"},
        1012.0,
        cfg,
    )
    assert window_t.advice == [1012.0]


def test_suppressed_event_praesagium_state_key_excluded_typing_counts():
    cfg = _Cfg()

    states_p = H.initial_states(1000.0)
    window_p = H.ActivityWindow()
    H.record_activity(
        states_p,
        window_p,
        "augur.limen.suppressed",
        {"state_key": "single:praesagium:typing_wpm", "domain": "praesagium"},
        1013.0,
        cfg,
    )
    assert window_p.suppressed == []

    states_t = H.initial_states(1000.0)
    window_t = H.ActivityWindow()
    H.record_activity(
        states_t,
        window_t,
        "augur.limen.suppressed",
        {"state_key": "single:typing:wpm", "domain": "typing"},
        1013.0,
        cfg,
    )
    assert window_t.suppressed == [1013.0]


def test_delivery_failure_event_praesagium_state_key_excluded_typing_counts():
    cfg = _Cfg()

    states_p = H.initial_states(1000.0)
    window_p = H.ActivityWindow()
    H.record_activity(
        states_p,
        window_p,
        "augur.limen.delivery_failure",
        {"state_key": "single:praesagium:typing_wpm", "domain": "praesagium"},
        1014.0,
        cfg,
    )
    assert window_p.delivery_failure == []

    states_t = H.initial_states(1000.0)
    window_t = H.ActivityWindow()
    H.record_activity(
        states_t,
        window_t,
        "augur.limen.delivery_failure",
        {"state_key": "single:typing:wpm", "domain": "typing"},
        1014.0,
        cfg,
    )
    assert window_t.delivery_failure == [1014.0]


def test_advice_surface_violation_praesagium_domain_excluded_typing_counts():
    cfg = _Cfg()

    states_p = H.initial_states(1000.0)
    window_p = H.ActivityWindow()
    H.record_activity(
        states_p,
        window_p,
        "augur.conscientia.violation",
        {"surface": "advice", "domain": "praesagium"},
        1012.0,
        cfg,
    )
    assert window_p.conscientia_block == []

    states_t = H.initial_states(1000.0)
    window_t = H.ActivityWindow()
    H.record_activity(
        states_t,
        window_t,
        "augur.conscientia.violation",
        {"surface": "advice", "domain": "typing"},
        1012.0,
        cfg,
    )
    assert window_t.conscientia_block == [1012.0]


def test_stall_deficit_unchanged_by_praesagium_terminals():
    # End-to-end: two MEDIUM nexus.detected (real work, aged past the servicing
    # grace so they land in the deficit numerator) answered ONLY by
    # praesagium-attributable terminals must still read as a genuine stall --
    # the anticipatory terminal must not mask the real, unserviced detection.
    cfg = _Cfg()
    states = H.initial_states(1000.0)
    window = H.ActivityWindow()
    H.record_activity(
        states,
        window,
        "augur.nexus.detected",
        {"combined_severity": "MEDIUM"},
        750.0,
        cfg,
    )
    H.record_activity(
        states,
        window,
        "augur.consilium.advice",
        {"domain": "praesagium"},
        751.0,
        cfg,
    )
    H.record_activity(
        states,
        window,
        "augur.nexus.detected",
        {"combined_severity": "MEDIUM"},
        760.0,
        cfg,
    )
    H.record_activity(
        states,
        window,
        "augur.limen.suppressed",
        {"state_key": "single:praesagium:typing_wpm"},
        761.0,
        cfg,
    )
    verdict = H.stall_signal(window, 1000.0, cfg)
    assert verdict.degraded and "consilium_stall" in verdict.reasons


def test_stall_deficit_cleared_by_equivalent_typing_terminals():
    # Control: the same shape, but with real (typing-domain) terminals instead
    # of praesagium ones -- must clear the stall. Confirms the exclusion above
    # is domain-specific, not a blanket "advice/suppressed never count" bug.
    cfg = _Cfg()
    states = H.initial_states(1000.0)
    window = H.ActivityWindow()
    H.record_activity(
        states,
        window,
        "augur.nexus.detected",
        {"combined_severity": "MEDIUM"},
        750.0,
        cfg,
    )
    H.record_activity(
        states, window, "augur.consilium.advice", {"domain": "typing"}, 751.0, cfg
    )
    H.record_activity(
        states,
        window,
        "augur.nexus.detected",
        {"combined_severity": "MEDIUM"},
        760.0,
        cfg,
    )
    H.record_activity(
        states,
        window,
        "augur.limen.suppressed",
        {"state_key": "single:typing:wpm"},
        761.0,
        cfg,
    )
    verdict = H.stall_signal(window, 1000.0, cfg)
    assert verdict.degraded is False
    assert "consilium_stall" not in verdict.reasons


def test_praesagium_foreseen_event_not_counted_as_pending_work():
    # Invariant: pending work is ONLY ever built from augur.nexus.detected; a
    # praesagium foreseen-lane event was never pending in the first place. It
    # still stamps liveness (classify_event routes augur.praesagium.* to the
    # "praesagium" faculty) but never touches detected_mh.
    cfg = _Cfg()
    states = H.initial_states(1000.0)
    window = H.ActivityWindow()
    H.record_activity(
        states,
        window,
        "augur.praesagium.foreseen",
        {"combined_severity": "MEDIUM"},
        1010.0,
        cfg,
    )
    assert window.detected_mh == []
    assert states["praesagium"].last_event_ts == 1010.0
