from imperator.auspices import _TIER_INT, compute_auspices, salience


def _inputs(**over):
    base = {
        "activity": "ide",
        "intensity_ewma": 90.0,
        "anomaly_load": 1.5,
        "escalation_tier": "MEDIUM",
        "has_active_correlation": True,
        "active_correlations": {"involved_domains": ["typing", "activity_focus"]},
        "last_advice": {"decision_id": "d1", "advice": "x"},
        "reception": {"explicit_rating": "n"},
        "latest_decision": {"decision": "fired", "decision_as_of": 1.0},
        "pipeline_health_rollup": "healthy",
        "session_id": "s1",
    }
    base.update(over)
    return base


def test_salience_bounded_and_weighted():
    s = salience(
        anomaly_load=1.5,
        escalation_tier="MEDIUM",
        has_active_correlation=True,
        intensity_ewma=90.0,
    )
    assert 0.0 <= s <= 1.0
    assert round(s, 3) == round(
        0.35 * 0.5 + 0.30 * (2 / 3) + 0.20 * 1.0 + 0.15 * min(90.0 / 300.0, 1.0), 3
    )


def test_salience_quiescent_floor():
    assert salience(0.0, "quiescent", False, 0.0) == 0.0


def test_compute_auspices_shape_and_freshness():
    snap = compute_auspices(_inputs(), now=100.0, prev={}, cfg=None)
    assert snap["schema_version"] == 1
    assert snap["generated_at"] == 100.0
    assert snap["session_id"] == "s1"
    assert snap["activity"] == {"value": "ide", "fresh": True, "as_of": 100.0}
    assert (
        snap["last_advice_and_reception"]["value"]["latest_decision"]["decision"]
        == "fired"
    )
    assert snap["salience"]["value"] == salience(1.5, "MEDIUM", True, 90.0)


def test_compute_auspices_missing_inputs_not_fresh():
    snap = compute_auspices({"session_id": None}, now=5.0, prev={}, cfg=None)
    assert snap["activity"] == {"value": None, "fresh": False, "as_of": 5.0}
    assert snap["salience"]["value"] == 0.0


def test_salience_not_fresh_during_warmup():
    # No stream signals yet: salience computes to the quiescent floor, but must
    # NOT advertise itself fresh — else the reasoner reads a warmup 0.0 as a
    # current "all calm" attention reading (it only trusts fresh cells).
    snap = compute_auspices({"session_id": None}, now=5.0, prev={}, cfg=None)
    assert snap["salience"]["fresh"] is False


def test_salience_fresh_when_any_attention_signal_present():
    # Any genuine attention signal makes the fused salience a current reading.
    for key, val in (
        ("anomaly_load", 1.5),
        ("escalation_tier", "MEDIUM"),
        ("intensity_ewma", 90.0),
    ):
        snap = compute_auspices(
            {"session_id": "s1", key: val}, now=6.0, prev={}, cfg=None
        )
        assert snap["salience"]["fresh"] is True, key


def test_salience_fresh_with_full_inputs():
    snap = compute_auspices(_inputs(), now=100.0, prev={}, cfg=None)
    assert snap["salience"]["fresh"] is True


def test_tier_int_omits_low_per_spec():
    # Spec §5.5: tier_int = {quiescent:0, MEDIUM:2, HIGH:3}. LOW is never a
    # published escalation_tier (Nexus drops standalone LOW), so it must fall
    # through the default branch, not carry a redundant identity entry.
    assert "LOW" not in _TIER_INT
    assert _TIER_INT == {"quiescent": 0, "MEDIUM": 2, "HIGH": 3}


def test_salience_unknown_tier_defaults_to_zero():
    # Default branch: an unmapped tier (incl. LOW, should it ever leak) scores 0
    # on the escalation term, identical to quiescent.
    base = salience(0.0, "quiescent", False, 0.0)
    assert salience(0.0, "LOW", False, 0.0) == base
    assert salience(0.0, "totally-unknown", False, 0.0) == base
    assert base == 0.0
