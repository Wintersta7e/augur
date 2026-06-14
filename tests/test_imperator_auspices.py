from imperator.auspices import compute_auspices, salience


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
