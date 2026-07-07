"""Tests for Task 11: SAFE handlers — sigma + gate_calibration.

Success paths assert through the REAL consumer contracts (vigil's
load_domain_thresholds merge for sigma; PersistenceManager hash/set state for
gate calibration), and every validation-failure path asserts BOTH the terminal
status AND that the anti-thrash gate was never armed (validation failure must
not arm).
"""

import fakeredis

from imperator import apply as A, proposals as P
from imperator.dialogue import router as R
from tabula.persistence import PersistenceManager
from vigil.anomaly_detector import load_domain_thresholds


def _pm():
    return PersistenceManager(fakeredis.FakeStrictRedis(decode_responses=False))


class _Cfg:
    dialogue_confirmed_apply_enabled = True
    imperator_ii_dedupe_staleness_s = 86400.0
    # FSRS review knobs, read only on a semantic_fact re-teach (matches
    # tabula.config.AugurConfig's defaults).
    memory_s_growth_factor = 0.5
    memory_s_max = 365


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


# ── context_directive ────────────────────────────────────────────────────────


def test_context_directive_apply_and_remove():
    pm = _pm()
    p = P.make_proposal(
        kind="context_directive",
        target="appX",
        action={
            "predicate": {"context": "focused_app", "match": "appX"},
            "action": "suppress",
            "scope": "all",
        },
        rationale="deep work",
        source="dialogue",
    )
    out = _confirmed(pm, p)
    assert out["status"] == "applied"
    did = out["action"]["directive_id"]
    assert any(d["directive_id"] == did for d in pm.load_dialogue_directives())
    assert pm.is_proposal_applied(p["dedupe_key"]) is True  # create arms
    # Fresh directive_id: no pre-existing content, so the restore anchor is None.
    assert out["action"]["prior_directive"] is None
    # remove path (undo)
    rem = P.make_proposal(
        kind="context_directive",
        target="appX",
        action={"op": "remove", "directive_id": did},
        rationale="undo",
        source="dialogue",
    )
    out_rem = _confirmed(pm, rem)
    assert out_rem["status"] == "applied"
    assert pm.load_dialogue_directives() == []
    # The removed content is captured as the restore anchor (Task 20 decision A).
    assert out_rem["action"]["prior_directive"]["directive_id"] == did
    assert out_rem["action"]["prior_directive"]["predicate"] == {
        "context": "focused_app",
        "match": "appX",
    }


def test_context_directive_remove_arms():
    # Removal arms too, consistent with every other dispatched apply
    # (self_tolerance_remove arms; removal ops are not a no-arm precedent).
    # Seeded directly (no prior create proposal) so the dedupe key for
    # (context_directive, appY) is untouched before the remove runs.
    pm = _pm()
    assert (
        pm.add_dialogue_directive(
            {
                "directive_id": "d-seeded",
                "predicate": {},
                "action": "suppress",
                "scope": "all",
            }
        )
        is True
    )
    p = P.make_proposal(
        kind="context_directive",
        target="appY",
        action={"op": "remove", "directive_id": "d-seeded"},
        rationale="undo",
        source="dialogue",
    )
    out = _confirmed(pm, p)
    assert out["status"] == "applied"
    assert pm.load_dialogue_directives() == []
    assert pm.is_proposal_applied(p["dedupe_key"]) is True  # removal arms


def test_context_directive_remove_missing_id_not_armed():
    pm = _pm()
    p = P.make_proposal(
        kind="context_directive",
        target="appX",
        action={"op": "remove"},  # no directive_id
        rationale="bad",
        source="dialogue",
    )
    out = _confirmed(pm, p)
    assert out["status"] == "logged"
    assert pm.is_proposal_applied(p["dedupe_key"]) is False  # never armed


def test_context_directive_invalid_action_enum_logged_and_not_armed():
    # The directive's inner "action" field (suppress|downgrade) must fail
    # closed on anything else, like every other enum/range check in this
    # module (Task 20 decision C).
    pm = _pm()
    p = P.make_proposal(
        kind="context_directive",
        target="appX",
        action={
            "predicate": {"context": "focused_app", "match": "appX"},
            "action": "delete_everything",  # not in {suppress, downgrade}
            "scope": "all",
        },
        rationale="bad",
        source="dialogue",
    )
    out = _confirmed(pm, p)
    assert out["status"] == "logged"
    assert pm.is_proposal_applied(p["dedupe_key"]) is False  # never armed
    assert pm.load_dialogue_directives() == []


def test_context_directive_at_cap_refusal():
    pm = _pm()
    # Create and store up to the cap
    from tabula.persistence import MAX_DIALOGUE_DIRECTIVES

    directives = []
    for i in range(MAX_DIALOGUE_DIRECTIVES):
        d = {
            "directive_id": f"directive_{i}",
            "predicate": {"context": "test"},
            "action": "suppress",
            "scope": "all",
        }
        assert pm.add_dialogue_directive(d) is True
        directives.append(d)

    # Try to add one more (new id) — should be refused
    p = P.make_proposal(
        kind="context_directive",
        target="appX",
        action={
            "predicate": {"context": "focused_app", "match": "appX"},
            "action": "suppress",
            "scope": "all",
        },
        rationale="over cap",
        source="dialogue",
    )
    out = _confirmed(pm, p)
    assert out["status"] == "logged"
    # Verify the cap is still at max (no new directive was added)
    all_directives = pm.load_dialogue_directives()
    assert len(all_directives) == MAX_DIALOGUE_DIRECTIVES
    # Nothing was stored, so NO rollback anchor may be recorded: an anchor on a
    # refused write would let a follow-on undo hdel a never-stored id and
    # report a false "Reversed."
    assert "directive_id" not in out["action"]
    # The undo path must report unavailable, not build a bogus remove-inverse.
    assert R.build_inverse({"proposal": out}) is None


def test_context_directive_upsert_at_cap():
    pm = _pm()
    # Create and store up to the cap
    from tabula.persistence import MAX_DIALOGUE_DIRECTIVES

    for i in range(MAX_DIALOGUE_DIRECTIVES):
        d = {
            "directive_id": f"directive_{i}",
            "predicate": {"context": "test"},
            "action": "suppress",
            "scope": "all",
        }
        pm.add_dialogue_directive(d)

    # Upsert an existing id at cap — should succeed
    existing_id = "directive_0"
    p = P.make_proposal(
        kind="context_directive",
        target="appX",
        action={
            "directive_id": existing_id,
            "predicate": {"context": "updated"},
            "action": "downgrade",
            "scope": "single",
        },
        rationale="update existing",
        source="dialogue",
    )
    out = _confirmed(pm, p)
    assert out["status"] == "applied"
    # Verify the directive was updated
    all_directives = pm.load_dialogue_directives()
    assert len(all_directives) == MAX_DIALOGUE_DIRECTIVES
    updated = next(
        (d for d in all_directives if d["directive_id"] == existing_id), None
    )
    assert updated is not None
    assert updated["predicate"] == {"context": "updated"}
    # The pre-upsert content is captured as the restore anchor (decision A).
    prior = out["action"]["prior_directive"]
    assert prior["directive_id"] == existing_id
    assert prior["predicate"] == {"context": "test"}
    assert prior["action"] == "suppress"
    assert prior["scope"] == "all"


# ── semantic_fact ─────────────────────────────────────────────────────────────


def test_semantic_fact_apply():
    pm = _pm()
    p = P.make_proposal(
        kind="semantic_fact",
        target="chess+typing",
        action={
            "pattern": {
                "kind": "semantic",
                "domains": ["chess", "typing"],
                "rule_key": "HIGH+HIGH",
                "severity": "MEDIUM",
            }
        },
        rationale="stress",
        source="dialogue",
    )
    out = _confirmed(pm, p)
    assert out["status"] == "applied" and out["action"]["memory_id"]
    assert pm.load_taught_facts()
    assert pm.is_proposal_applied(p["dedupe_key"]) is True  # create arms
    # Fresh memory_id: no pre-existing state, so the restore anchor is None.
    assert out["action"]["prior_fact"] is None


def test_semantic_fact_missing_pattern_not_armed():
    pm = _pm()
    p = P.make_proposal(
        kind="semantic_fact",
        target="chess",
        action={},  # no pattern
        rationale="bad",
        source="dialogue",
    )
    out = _confirmed(pm, p)
    assert out["status"] == "logged"
    assert pm.is_proposal_applied(p["dedupe_key"]) is False  # never armed
    assert pm.load_taught_facts() == []


def test_semantic_fact_invalid_pattern_kind_not_armed():
    # Apply-layer defense-in-depth mirror of the create_user_taught_memory
    # persistence-layer check (Task 20 decision C).
    pm = _pm()
    p = P.make_proposal(
        kind="semantic_fact",
        target="chess",
        action={
            "pattern": {
                "kind": "episodic",  # not "semantic"
                "domains": ["chess"],
                "rule_key": None,
                "severity": "LOW",
            }
        },
        rationale="bad",
        source="dialogue",
    )
    out = _confirmed(pm, p)
    assert out["status"] == "logged"
    assert pm.is_proposal_applied(p["dedupe_key"]) is False  # never armed
    assert pm.load_taught_facts() == []


def test_semantic_fact_remove_missing_id_not_armed():
    pm = _pm()
    p = P.make_proposal(
        kind="semantic_fact",
        target="chess",
        action={"op": "remove"},  # no memory_id
        rationale="bad",
        source="dialogue",
    )
    out = _confirmed(pm, p)
    assert out["status"] == "logged"
    assert pm.is_proposal_applied(p["dedupe_key"]) is False  # never armed


def test_semantic_fact_remove_archives_and_anchors_prior():
    pm = _pm()
    p = P.make_proposal(
        kind="semantic_fact",
        target="chess+typing",
        action={
            "pattern": {
                "kind": "semantic",
                "domains": ["chess", "typing"],
                "rule_key": "HIGH+HIGH",
                "severity": "MEDIUM",
            }
        },
        rationale="stress",
        source="dialogue",
    )
    out = _confirmed(pm, p)
    assert out["status"] == "applied"
    mid = out["action"]["memory_id"]

    rem = P.make_proposal(
        kind="semantic_fact",
        target="chess+typing",
        action={"op": "remove", "memory_id": mid},
        rationale="undo",
        source="dialogue",
    )
    out_rem = _confirmed(pm, rem)
    assert out_rem["status"] == "applied"
    st = pm.load_memory_state(mid)
    assert st is not None and st["status"] == "archived"
    # The pre-removal state is captured as the restore anchor (decision A).
    prior = out_rem["action"]["prior_fact"]
    assert prior is not None and prior["memory_id"] == mid
    assert prior["status"] == "active"


def test_semantic_fact_remove_unknown_id_anchors_none():
    pm = _pm()
    p = P.make_proposal(
        kind="semantic_fact",
        target="chess",
        action={"op": "remove", "memory_id": "nonexistent"},
        rationale="bad",
        source="dialogue",
    )
    out = _confirmed(pm, p)
    assert out["status"] == "applied"  # arms + no-ops: nothing to archive
    assert out["action"]["prior_fact"] is None
    assert R.build_inverse({"proposal": out}) is None  # unavailable: no prior


def test_semantic_fact_reteach_strengthens_not_overwrites():
    # Task 20 decision B, exercised through the full confirmed-apply handler
    # (not just the persistence layer): re-teaching the SAME pattern via a
    # second confirmed proposal must review, not reset, FSRS decay.
    pm = _pm()
    pattern = {
        "kind": "semantic",
        "domains": ["typing"],
        "rule_key": None,
        "severity": "LOW",
    }
    p1 = P.make_proposal(
        kind="semantic_fact",
        target="typing",
        action={"pattern": pattern},
        rationale="left-handed",
        source="dialogue",
    )
    out1 = _confirmed(pm, p1)
    assert out1["status"] == "applied"
    mid = out1["action"]["memory_id"]
    first = pm.load_memory_state(mid)
    assert first["S"] == 1.0

    p2 = P.make_proposal(
        kind="semantic_fact",
        target="typing",
        action={"pattern": pattern},
        rationale="left-handed again",
        source="dialogue",
    )
    out2 = _confirmed(pm, p2)
    assert out2["status"] == "applied"
    assert out2["action"]["memory_id"] == mid  # same deterministic id
    second = pm.load_memory_state(mid)
    assert second["S"] > first["S"], "re-teach must strengthen S, not reset it"
    # The pre-re-teach state is captured as the restore anchor.
    assert out2["action"]["prior_fact"]["S"] == 1.0
