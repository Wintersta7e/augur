"""Tests for Task 11: SAFE handlers — sigma + gate_calibration.

Success paths assert through the REAL consumer contracts (vigil's
load_domain_thresholds merge for sigma; PersistenceManager hash/set state for
gate calibration), and every validation-failure path asserts BOTH the terminal
status AND that the anti-thrash gate was never armed (validation failure must
not arm).
"""

import fakeredis

from imperator import apply as A, proposals as P
from tabula.persistence import PersistenceManager
from vigil.anomaly_detector import load_domain_thresholds


def _pm():
    return PersistenceManager(fakeredis.FakeStrictRedis(decode_responses=False))


class _Cfg:
    dialogue_confirmed_apply_enabled = True
    imperator_ii_dedupe_staleness_s = 86400.0


def _confirmed(pm, p):
    return A.apply_proposal(pm, p, cfg=_Cfg(), session_id="d1", confirmed=True)


# ── sigma ────────────────────────────────────────────────────────────────────


def test_sigma_apply_and_anchor():
    pm = _pm()
    pm.save_thresholds("typing", {"sigma_threshold": 2.0})
    p = P.make_proposal(
        kind="sigma",
        target="typing",
        action={"domain": "typing", "sigma": 3.0},
        rationale="quieter",
        source="dialogue",
    )
    out = _confirmed(pm, p)
    assert out["status"] == "applied"
    # Assert through the REAL consumer read path: vigil merges stored thresholds
    # over DEFAULT_THRESHOLDS and gates on the "sigma_threshold" key.
    th = load_domain_thresholds(pm, "typing")
    assert th["sigma_threshold"] == 3.0
    assert out["action"]["prior_sigma"] == 2.0
    assert pm.is_proposal_applied(p["dedupe_key"]) is True  # armed on success


def test_sigma_preserves_other_threshold_fields():
    pm = _pm()
    pm.save_thresholds("typing", {"sigma_threshold": 2.0, "hst_threshold": 0.9})
    p = P.make_proposal(
        kind="sigma",
        target="typing",
        action={"domain": "typing", "sigma": 2.5},
        rationale="quieter",
        source="dialogue",
    )
    out = _confirmed(pm, p)
    assert out["status"] == "applied"
    stored = pm.load_thresholds("typing")
    assert stored == {"sigma_threshold": 2.5, "hst_threshold": 0.9}


def test_sigma_missing_value_logged_and_not_armed():
    pm = _pm()
    p = P.make_proposal(
        kind="sigma",
        target="typing",
        action={"domain": "typing"},  # no "sigma"
        rationale="bad",
        source="dialogue",
    )
    out = _confirmed(pm, p)
    assert out["status"] == "logged"
    assert pm.is_proposal_applied(p["dedupe_key"]) is False  # never armed
    assert pm.load_thresholds("typing") is None  # never wrote


def test_sigma_non_numeric_value_logged_and_not_armed():
    pm = _pm()
    p = P.make_proposal(
        kind="sigma",
        target="typing",
        action={"domain": "typing", "sigma": "very high"},
        rationale="bad",
        source="dialogue",
    )
    out = _confirmed(pm, p)
    assert out["status"] == "logged"
    assert pm.is_proposal_applied(p["dedupe_key"]) is False
    assert pm.load_thresholds("typing") is None


def test_sigma_non_finite_value_logged_and_not_armed():
    # NaN sigma would make vigil's `deviation >= sigma_threshold` always-False,
    # silently disabling anomaly detection for the domain. Must fail closed.
    pm = _pm()
    pm.save_thresholds("typing", {"sigma_threshold": 2.0})
    for bad in (float("nan"), float("inf"), float("-inf")):
        p = P.make_proposal(
            kind="sigma",
            target="typing",
            action={"domain": "typing", "sigma": bad},
            rationale="bad",
            source="dialogue",
        )
        out = _confirmed(pm, p)
        assert out["status"] == "logged"
        assert pm.is_proposal_applied(p["dedupe_key"]) is False  # never armed
    assert pm.load_thresholds("typing") == {"sigma_threshold": 2.0}  # never wrote


def test_sigma_out_of_range_value_logged_and_not_armed():
    # Bounds mirror Disciplina's tuning clamps [sigma_min=1.5, sigma_max=5.0].
    pm = _pm()
    for bad in (0.0, 1.49, 99.0, -2.0):
        p = P.make_proposal(
            kind="sigma",
            target="typing",
            action={"domain": "typing", "sigma": bad},
            rationale="bad",
            source="dialogue",
        )
        out = _confirmed(pm, p)
        assert out["status"] == "logged"
        assert pm.is_proposal_applied(p["dedupe_key"]) is False
    assert pm.load_thresholds("typing") is None


# ── gate_calibration ─────────────────────────────────────────────────────────


def test_gate_calibration_self_tolerance_add():
    pm = _pm()
    p = P.make_proposal(
        kind="gate_calibration",
        target="single:typing:user",
        action={"op": "self_tolerance_add", "state_key": "single:typing:user"},
        rationale="stop flagging",
        source="dialogue",
    )
    out = _confirmed(pm, p)
    assert out["status"] == "applied"
    assert pm.is_self_tolerant("single:typing:user") is True
    assert out["action"]["prior"] is False  # was not a member
    assert pm.is_proposal_applied(p["dedupe_key"]) is True  # armed on success


def test_gate_calibration_self_tolerance_remove():
    pm = _pm()
    pm.add_self_tolerance("single:typing:user")
    p = P.make_proposal(
        kind="gate_calibration",
        target="single:typing:user",
        action={"op": "self_tolerance_remove", "state_key": "single:typing:user"},
        rationale="flag again",
        source="dialogue",
    )
    out = _confirmed(pm, p)
    assert out["status"] == "applied"
    assert pm.is_self_tolerant("single:typing:user") is False  # actually removed
    assert out["action"]["prior"] is True  # was a member


def test_gate_calibration_floor_set_shape_and_merge():
    pm = _pm()
    # Pre-existing floor entry for a DIFFERENT signature must survive the write.
    pm.save_habituation_floor("other:chess:board", {"floor": 0.2, "last_ts": "s0"})
    p = P.make_proposal(
        kind="gate_calibration",
        target="single:typing:user",
        action={
            "op": "floor_set",
            "state_key": "single:typing:user",
            "value": 0.5,
        },
        rationale="habituate faster",
        source="dialogue",
    )
    out = _confirmed(pm, p)
    assert out["status"] == "applied"
    entry = pm.load_habituation_floor("single:typing:user")
    assert entry["floor"] == 0.5
    assert isinstance(entry["last_ts"], float)  # refreshed alongside the floor
    assert out["action"]["prior"] == {}  # no prior entry for this signature
    # Merge semantics: the other signature's entry is untouched.
    assert pm.load_habituation_floor("other:chess:board") == {
        "floor": 0.2,
        "last_ts": "s0",
    }


def test_gate_calibration_unknown_op_logged_and_not_armed():
    pm = _pm()
    p = P.make_proposal(
        kind="gate_calibration",
        target="single:typing:user",
        action={"op": "bogus", "state_key": "single:typing:user"},
        rationale="bad",
        source="dialogue",
    )
    out = _confirmed(pm, p)
    assert out["status"] == "logged"
    assert pm.is_proposal_applied(p["dedupe_key"]) is False  # never armed


def test_gate_calibration_floor_set_missing_value_logged_and_not_armed():
    pm = _pm()
    p = P.make_proposal(
        kind="gate_calibration",
        target="single:typing:user",
        action={"op": "floor_set", "state_key": "single:typing:user"},  # no value
        rationale="bad",
        source="dialogue",
    )
    out = _confirmed(pm, p)
    assert out["status"] == "logged"
    assert pm.is_proposal_applied(p["dedupe_key"]) is False
    assert pm.load_habituation_floor("single:typing:user") == {}  # never wrote


def test_gate_calibration_floor_set_non_numeric_value_logged_and_not_armed():
    # False is the load-bearing bool case: float(False) == 0.0 is in-range, so
    # only an explicit bool rejection keeps it out (True is out-of-range anyway).
    pm = _pm()
    for bad in ("lots", False, True):
        p = P.make_proposal(
            kind="gate_calibration",
            target="single:typing:user",
            action={
                "op": "floor_set",
                "state_key": "single:typing:user",
                "value": bad,
            },
            rationale="bad",
            source="dialogue",
        )
        out = _confirmed(pm, p)
        assert out["status"] == "logged"
        assert pm.is_proposal_applied(p["dedupe_key"]) is False
    assert pm.load_habituation_floor("single:typing:user") == {}


def test_gate_calibration_floor_set_out_of_range_logged_and_not_armed():
    # A floor > 1.0 produces a negative habituation cap downstream; bounds
    # mirror Disciplina's floor sweep [0.0, GATE_FLOOR_MAX=0.6].
    pm = _pm()
    for bad in (1.5, 0.61, -0.1):
        p = P.make_proposal(
            kind="gate_calibration",
            target="single:typing:user",
            action={
                "op": "floor_set",
                "state_key": "single:typing:user",
                "value": bad,
            },
            rationale="bad",
            source="dialogue",
        )
        out = _confirmed(pm, p)
        assert out["status"] == "logged"
        assert pm.is_proposal_applied(p["dedupe_key"]) is False  # never armed
    assert pm.load_habituation_floor("single:typing:user") == {}  # never wrote


def test_gate_calibration_floor_set_non_finite_logged_and_not_armed():
    pm = _pm()
    for bad in (float("nan"), float("inf"), float("-inf")):
        p = P.make_proposal(
            kind="gate_calibration",
            target="single:typing:user",
            action={
                "op": "floor_set",
                "state_key": "single:typing:user",
                "value": bad,
            },
            rationale="bad",
            source="dialogue",
        )
        out = _confirmed(pm, p)
        assert out["status"] == "logged"
        assert pm.is_proposal_applied(p["dedupe_key"]) is False
    assert pm.load_habituation_floor("single:typing:user") == {}
