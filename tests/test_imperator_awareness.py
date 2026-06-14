from imperator import awareness


def test_allowlist_only_two_subjects():
    assert awareness.consumed("augur.nexus.detected") is True
    assert awareness.consumed("augur.vigil.anomaly") is True
    assert awareness.consumed("augur.consilium.advice") is False
    assert awareness.consumed("augur.limen.suppressed") is False
    assert awareness.consumed("augur.imperator.auspices") is False
    assert awareness.consumed("augur.session.end") is False


def test_apply_event_nexus_sets_tier_and_correlation():
    state = {}
    awareness.apply_event(
        state,
        "augur.nexus.detected",
        {
            "combined_severity": "HIGH",
            "correlation_found": True,
            "involved_domains": ["typing", "activity_focus"],
            "correlation_span_s": 12.0,
        },
        now=100.0,
    )
    assert state["escalation_tier"] == "HIGH"
    assert state["has_active_correlation"] is True
    assert state["correlation_span_s"] == 12.0


def test_apply_event_anomaly_ema():
    state = {}
    awareness.apply_event(state, "augur.vigil.anomaly", {"severity": "high"}, now=1.0)
    assert state["anomaly_load"] == 0.3 * 3


def test_decay_stream_decays_and_expires():
    state = {
        "anomaly_load": 3.0,
        "anomaly_load_ts": 0.0,
        "has_active_correlation": True,
        "active_correlations": {"x": 1},
        "active_correlations_ts": 0.0,
        "correlation_span_s": 10.0,
        "escalation_tier": "HIGH",
        "escalation_tier_ts": 0.0,
    }
    cfg = type("C", (), {"imperator_salience_window_s": 300.0})()
    awareness.decay_stream(state, now=600.0, cfg=cfg)
    assert state["anomaly_load"] < 3.0
    assert state["has_active_correlation"] is False
    assert state["escalation_tier"] == "quiescent"


def test_materially_changed_ignores_decay_jitter():
    a = {"generated_at": 1.0, "salience": {"value": 0.500, "as_of": 1.0}}
    b = {"generated_at": 2.0, "salience": {"value": 0.503, "as_of": 2.0}}
    assert awareness.materially_changed(a, b) is False
    assert awareness.materially_changed(a, {"salience": {"value": 0.70}}) is True
