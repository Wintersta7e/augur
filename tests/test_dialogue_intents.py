import pytest
from imperator.dialogue import intents as I


def test_valid_intent_passes():
    out = I.validate_intent(
        {
            "kind": "correct_silence",
            "target": "single:typing:user",
            "action": {},
            "rationale": "speak up",
        }
    )
    assert out["kind"] == "correct_silence"


def test_unknown_kind_raises():
    with pytest.raises(ValueError):
        I.validate_intent(
            {"kind": "delete_everything", "target": "x", "action": {}, "rationale": "r"}
        )


def test_missing_target_raises():
    # Must use a VALID kind from _REQUIRE_TARGET: an unknown kind (the brief's
    # original "sigma") trips the kind check first and never reaches the
    # missing-target branch.
    for kind in ("tune_rule", "correct_silence"):
        with pytest.raises(ValueError, match="requires a target"):
            I.validate_intent({"kind": kind, "action": {}, "rationale": "r"})


def test_unknown_kind_rejected():
    # "sigma" is a PROPOSAL kind, not an intent kind — must be rejected as unknown.
    with pytest.raises(ValueError, match="unknown intent kind"):
        I.validate_intent({"kind": "sigma", "action": {}, "rationale": "r"})


def test_affirmative_and_heavy():
    assert I.is_affirmative("yes") and I.is_affirmative("do it")
    assert not I.is_affirmative("no")
    assert I.matches_heavy_phrase("yes, change the matrix", "change the matrix")
    assert not I.matches_heavy_phrase("yes", "change the matrix")


# ── numeric-field validation (sigma + habituation floor) ─────────────────────
# Bounds mirror Disciplina's own tuning writer: sigma clamped into
# [sigma_min=1.5, sigma_max=5.0], habituation floor into [0.0, GATE_FLOOR_MAX=0.6].


def _sigma_intent(value):
    return {
        "kind": "tune_rule",
        "target": "typing",
        "action": {"sigma": value},
        "rationale": "r",
    }


def _floor_intent(value):
    action = {"op": "floor_set", "state_key": "single:typing:user"}
    if value is not None:
        action["value"] = value
    return {
        "kind": "correct_silence",
        "target": "single:typing:user",
        "action": action,
        "rationale": "r",
    }


def test_sigma_non_finite_rejected():
    for bad in (float("nan"), float("inf"), float("-inf")):
        with pytest.raises(ValueError):
            I.validate_intent(_sigma_intent(bad))


def test_sigma_out_of_range_rejected():
    for bad in (0.0, 1.49, 5.01, -2.0):
        with pytest.raises(ValueError):
            I.validate_intent(_sigma_intent(bad))


def test_sigma_non_numeric_rejected():
    for bad in ("very high", True, [3.0]):
        with pytest.raises(ValueError):
            I.validate_intent(_sigma_intent(bad))


def test_sigma_in_range_normalized():
    for ok, expected in ((3, 3.0), (1.5, 1.5), (5.0, 5.0)):
        out = I.validate_intent(_sigma_intent(ok))
        assert out["action"]["sigma"] == expected


def test_floor_non_finite_rejected():
    for bad in (float("nan"), float("inf"), float("-inf")):
        with pytest.raises(ValueError):
            I.validate_intent(_floor_intent(bad))


def test_floor_out_of_range_rejected():
    for bad in (-0.1, 0.61, 1.5):
        with pytest.raises(ValueError):
            I.validate_intent(_floor_intent(bad))


def test_floor_set_missing_value_rejected():
    with pytest.raises(ValueError):
        I.validate_intent(_floor_intent(None))


def test_floor_bounds_inclusive():
    for ok in (0.0, 0.6):
        out = I.validate_intent(_floor_intent(ok))
        assert out["action"]["value"] == ok
