"""Pure Praefectus health engine."""

from praefectus import health as H


def test_classify_event_routing():
    assert H.classify_event("augur.system.heartbeat") == ("heartbeat", None)
    assert H.classify_event("augur.vigil.anomaly") == ("activity", "vigil")
    assert H.classify_event("augur.nexus.detected") == ("activity", "nexus")
    assert H.classify_event("augur.praefectus.health") == ("ignore", None)
    assert H.classify_event("augur.session.end") == ("ignore", None)


def test_initial_states_seeds_required():
    states = H.initial_states(1000.0)
    assert set(states) == set(H.REQUIRED_FACULTIES)
    assert states["vigil"].required is True
    assert states["vigil"].seen is False
    assert states["vigil"].last_heartbeat is None


def test_record_heartbeat_required_and_optional():
    states = H.initial_states(1000.0)
    H.record_heartbeat(states, "vigil", 1005.0)
    assert states["vigil"].seen is True
    assert states["vigil"].last_heartbeat == 1005.0
    # optional component registers on first beat
    H.record_heartbeat(states, "sensus.chess", 1006.0)
    assert states["sensus.chess"].required is False
    assert states["sensus.chess"].seen is True


def test_record_heartbeat_unknown_ignored():
    states = H.initial_states(1000.0)
    H.record_heartbeat(states, "bogus", 1005.0)
    assert "bogus" not in states


def test_record_activity_window_and_last_event():
    states = H.initial_states(1000.0)
    window = H.ActivityWindow()
    cfg = _Cfg()
    H.record_activity(
        states,
        window,
        "augur.nexus.detected",
        {"combined_severity": "HIGH"},
        1010.0,
        cfg,
    )
    assert window.detected_mh == [1010.0]
    assert states["nexus"].last_event_ts == 1010.0
    # LOW is excluded from detected_mh
    H.record_activity(
        states,
        window,
        "augur.nexus.detected",
        {"combined_severity": "LOW"},
        1011.0,
        cfg,
    )
    assert window.detected_mh == [1010.0]
    # terminal outcomes recorded
    H.record_activity(states, window, "augur.consilium.advice", {}, 1012.0, cfg)
    H.record_activity(states, window, "augur.limen.suppressed", {}, 1013.0, cfg)
    H.record_activity(states, window, "augur.limen.delivery_failure", {}, 1014.0, cfg)
    assert window.advice and window.suppressed and window.delivery_failure


class _Cfg:
    """Minimal config stand-in for the pure functions."""

    praefectus_heartbeat_interval_s = 10.0
    praefectus_stale_after_s = 30.0
    praefectus_dead_after_s = 90.0
    praefectus_warmup_s = 30.0
    praefectus_stall_tolerance = 1
    praefectus_stall_min_events = 2
    praefectus_delivery_failure_spike = 3
    effective_stall_window_s = 300.0
    effective_reflection_window_s = 300.0
