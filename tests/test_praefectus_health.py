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


def test_liveness_alive_stale_dead():
    cfg = _Cfg()
    st = H.FacultyHealth("vigil", required=True, seen=True, last_heartbeat=1000.0)
    assert H.liveness(st, 1010.0, 900.0, cfg) == "alive"  # age 10 <= 30
    assert H.liveness(st, 1050.0, 900.0, cfg) == "stale"  # 30 < age 50 <= 90
    assert H.liveness(st, 1200.0, 900.0, cfg) == "dead"  # age 200 > 90 (lost)


def test_liveness_never_started_required():
    cfg = _Cfg()
    st = H.FacultyHealth("vox", required=True)  # never seen
    started = 1000.0
    assert H.liveness(st, 1010.0, started, cfg) == "warming_up"  # within warmup (1030)
    assert (
        H.liveness(st, 1100.0, started, cfg) == "unknown"
    )  # past warmup, pre-dead horizon (1120)
    assert (
        H.liveness(st, 1200.0, started, cfg) == "dead"
    )  # past horizon → never_started


def test_liveness_absent_optional():
    cfg = _Cfg()
    st = H.FacultyHealth("sensus.chess", required=False)  # never seen, optional
    assert H.liveness(st, 9999.0, 1000.0, cfg) == "absent"


def test_stall_signal_deficit():
    cfg = _Cfg()
    w = H.ActivityWindow(detected_mh=[1.0, 2.0, 3.0])  # 3 MEDIUM/HIGH, 0 terminals
    v = H.stall_signal(w, 100.0, cfg)
    assert v.degraded and "consilium_stall" in v.reasons


def test_stall_signal_no_false_trigger_on_coalescing():
    cfg = _Cfg()
    # 3 detected, 2 terminals — within tolerance(1) → NOT degraded
    w = H.ActivityWindow(detected_mh=[1.0, 2.0, 3.0], advice=[1.5, 2.5])
    assert H.stall_signal(w, 100.0, cfg).degraded is False


def test_stall_signal_delivery_failure_spike():
    cfg = _Cfg()
    w = H.ActivityWindow(delivery_failure=[1.0, 2.0, 3.0])  # >= spike(3)
    v = H.stall_signal(w, 100.0, cfg)
    assert v.degraded and "delivery_failures" in v.reasons


def test_evaluate_entered_then_cleared():
    cfg = _Cfg()
    states = H.initial_states(1000.0)
    for f in H.REQUIRED_FACULTIES:  # all alive
        H.record_heartbeat(states, f, 1000.0)
    w = H.ActivityWindow(detected_mh=[1000.0, 1000.0, 1000.0])  # consilium stall
    r1 = H.evaluate(states, w, 1000.0, 1000.0, cfg)
    assert ("consilium", "consilium_stall") in r1.entered
    # clear the stall → recovery
    w2 = H.ActivityWindow()
    r2 = H.evaluate(states, w2, 1001.0, 1000.0, cfg)
    assert ("consilium", "consilium_stall") in r2.cleared


def test_evaluate_never_started_dead_entered():
    cfg = _Cfg()
    states = H.initial_states(1000.0)  # nobody heartbeats
    r = H.evaluate(states, H.ActivityWindow(), 1200.0, 1000.0, cfg)  # past horizon
    assert ("vox", "never_started") in r.entered
    assert states["vox"].overall_state == "dead"


def test_summarize_shape():
    cfg = _Cfg()
    states = H.initial_states(1000.0)
    H.record_heartbeat(states, "vigil", 1000.0)
    r = H.evaluate(states, H.ActivityWindow(), 1005.0, 1000.0, cfg)
    out = H.summarize(r)
    assert out["faculties"]["vigil"]["liveness"] == "alive"
    assert "uptime_s" in out and "ts" in out


def test_reflection_lag_flag_observability_only():
    cfg = _Cfg()
    states = H.initial_states(1000.0)
    for f in H.REQUIRED_FACULTIES:
        H.record_heartbeat(states, f, 2000.0)
    states["responsum"].last_event_ts = 1000.0  # completed long ago
    states["disciplina"].last_event_ts = None  # no reflection followed
    r = H.evaluate(
        states, H.ActivityWindow(), 2000.0, 1000.0, cfg
    )  # 1000s > 300 window
    assert "reflection_lag" in states["disciplina"].flags
    assert all(reason != "reflection_lag" for _, reason in r.entered)  # never an alert


def test_no_reflection_lag_when_disciplina_followed():
    cfg = _Cfg()
    states = H.initial_states(1000.0)
    for f in H.REQUIRED_FACULTIES:
        H.record_heartbeat(states, f, 2000.0)
    states["responsum"].last_event_ts = 1000.0
    states["disciplina"].last_event_ts = 1100.0  # followed after responsum
    H.evaluate(states, H.ActivityWindow(), 2000.0, 1000.0, cfg)
    assert states["disciplina"].flags == []


def test_no_masking_between_sensors():
    cfg = _Cfg()
    states = H.initial_states(1000.0)
    H.record_heartbeat(states, "sensus.chess", 1000.0)
    H.record_heartbeat(states, "sensus.typing", 2000.0)
    # at now=2000: chess 1000s stale (dead), typing fresh (alive) — distinct ids, no masking
    assert H.liveness(states["sensus.chess"], 2000.0, 1000.0, cfg) == "dead"
    assert H.liveness(states["sensus.typing"], 2000.0, 1000.0, cfg) == "alive"


def test_evaluate_never_started_fires_once_then_silent_then_clears():
    # T1: debounce — a never_started death ENTERS once, stays silent on the
    # next tick (no re-fire), and CLEARS exactly once on recovery. Same states
    # dict mutated across ticks (not fresh dicts) so reason-set diffing applies.
    cfg = _Cfg()
    states = H.initial_states(1000.0)  # nobody heartbeats
    # tick1: past the never_started horizon → enters
    r1 = H.evaluate(states, H.ActivityWindow(), 1200.0, 1000.0, cfg)
    assert ("vox", "never_started") in r1.entered
    # tick2: later now, still no heartbeat → no re-fire
    r2 = H.evaluate(states, H.ActivityWindow(), 1300.0, 1000.0, cfg)
    assert r2.entered == []
    # recovery: vox heartbeats → clears exactly once
    H.record_heartbeat(states, "vox", 1305.0)
    r3 = H.evaluate(states, H.ActivityWindow(), 1306.0, 1000.0, cfg)
    cleared_vox = [c for c in r3.cleared if c == ("vox", "never_started")]
    assert cleared_vox == [("vox", "never_started")]


def test_record_activity_prunes_old_entries():
    # T2: window pruning — an entry older than the stall cutoff is dropped on
    # the next record_activity, while the fresh entry remains.
    states = H.initial_states(1000.0)
    window = H.ActivityWindow(detected_mh=[0.0])  # far older than cutoff
    cfg = _Cfg()  # effective_stall_window_s = 300.0
    H.record_activity(
        states,
        window,
        "augur.nexus.detected",
        {"combined_severity": "HIGH"},
        1000.0,
        cfg,
    )
    # cutoff = 1000 - 300 = 700; 0.0 dropped, 1000.0 kept
    assert window.detected_mh == [1000.0]


def test_stall_signal_below_min_events_floor():
    # T3a: the min_events floor is the SOLE gate here. With tolerance=0 the deficit
    # clause alone WOULD fire on 1 detected/0 terminals (0 < 1-0), so a non-degraded
    # verdict isolates the floor as the blocker; crossing to 2 detected then degrades.
    cfg = _Cfg()
    cfg.praefectus_stall_tolerance = 0
    below = H.ActivityWindow(detected_mh=[1.0])  # 1 < min_events(2) → floor blocks
    assert H.stall_signal(below, 100.0, cfg).degraded is False
    at_floor = H.ActivityWindow(detected_mh=[1.0, 2.0])  # 2 ≥ floor, deficit 2>0
    assert H.stall_signal(at_floor, 100.0, cfg).degraded is True


def test_stall_signal_deficit_equals_tolerance_plus_one():
    # T3b: 3 detected, 1 terminal — deficit (2) == tolerance(1)+1 → degraded.
    cfg = _Cfg()
    w = H.ActivityWindow(detected_mh=[1.0, 2.0, 3.0], advice=[1.5])
    v = H.stall_signal(w, 100.0, cfg)
    assert v.degraded and "consilium_stall" in v.reasons


def test_evaluate_overall_state_masking_and_isolation():
    # T4: a stall degrades consilium's overall_state, and the populated window
    # does NOT leak into a non-consilium faculty's activity_state.
    cfg = _Cfg()
    states = H.initial_states(1000.0)
    for f in H.REQUIRED_FACULTIES:  # everyone alive
        H.record_heartbeat(states, f, 1000.0)
    w = H.ActivityWindow(detected_mh=[1000.0, 1000.0, 1000.0])  # consilium stall
    H.evaluate(states, w, 1000.0, 1000.0, cfg)
    assert states["consilium"].overall_state == "degraded"
    assert states["nexus"].activity_state == "ok"


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
