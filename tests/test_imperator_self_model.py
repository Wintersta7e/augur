from imperator.self_model import compute_self_model, competence


def _inputs(**over):
    base = {
        "precision": 0.8,
        "utility": 0.6,
        "utility_no_data": False,
        "mrt": {"excursion": 0.1, "directional": True},
        "dismissal_rate": 0.2,
        "suppression_rate": 0.3,
        "advice_volume": {"delivered": 5},
        "pipeline_health_full": {"faculties": {}},
        "health_score": 1.0,
        "coverage": {"coverage_depth": 0.75},
        "blind_spots": [],
        "recent_self_tuning": {},
        "session_id": "s1",
    }
    base.update(over)
    return base


def test_competence_no_data_utility_is_neutral():
    c = competence(
        precision=1.0,
        utility=1.0,
        utility_no_data=True,
        dismissal_rate=0.0,
        coverage_depth=1.0,
        health_score=1.0,
        n_blind_spots=0,
    )
    expected = 0.30 * 1.0 + 0.25 * 0.5 + 0.20 * 1.0 + 0.15 * 1.0 + 0.10 * 1.0 - 0.0
    assert round(c, 6) == round(expected, 6)


def test_competence_blind_spot_penalty_caps_at_five():
    low = competence(0.5, 0.5, False, 0.5, 0.5, 1.0, n_blind_spots=10)
    five = competence(0.5, 0.5, False, 0.5, 0.5, 1.0, n_blind_spots=5)
    assert low == five


def test_competence_coverage_no_data_neutral():
    base = competence(0.5, 0.5, False, 0.5, 0.0, 1.0, 0, coverage_no_data=False)
    neutral = competence(0.5, 0.5, False, 0.5, 0.0, 1.0, 0, coverage_no_data=True)
    assert neutral > base  # no-data coverage uses neutral 0.5, not 0.0


def test_compute_self_model_carries_directional_and_blind_spots():
    snap = compute_self_model(
        _inputs(blind_spots=[{"kind": "low_confidence_rule", "detail": "r"}]),
        now=10.0,
        prev={},
        cfg=None,
    )
    assert snap["schema_version"] == 1
    assert snap["mrt"]["value"]["directional"] is True
    assert snap["blind_spots"]["value"][0]["kind"] == "low_confidence_rule"
    assert snap["competence"]["value"] == competence(0.8, 0.6, False, 0.2, 0.75, 1.0, 1)


def test_compute_self_model_warming_up_when_no_report():
    snap = compute_self_model({"session_id": None}, now=1.0, prev={}, cfg=None)
    assert snap["precision"] == {"value": None, "fresh": False, "as_of": 1.0}


def test_compute_self_model_surfaces_reflection_ts():
    # The folded reflection's epoch is exposed for II's freshness content-check.
    snap = compute_self_model(
        _inputs(reflection_ts=1234.5), now=10.0, prev={}, cfg=None
    )
    assert snap["reflection_ts"] == 1234.5
    # Absent -> 0.0, so a warming-up self-model is never spuriously "fresh".
    snap2 = compute_self_model({"session_id": None}, now=1.0, prev={}, cfg=None)
    assert snap2["reflection_ts"] == 0.0
