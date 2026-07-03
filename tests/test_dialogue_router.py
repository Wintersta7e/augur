"""Tests for Task 13: dialogue router -- intent -> pending -> apply -> undo."""

import fakeredis
import pytest

from imperator import proposals as P
from imperator.dialogue import context as C
from imperator.dialogue import router as R
from tabula.persistence import PersistenceManager


def _pm():
    return PersistenceManager(fakeredis.FakeStrictRedis(decode_responses=False))


class _Cfg:
    dialogue_confirmed_apply_enabled = True
    imperator_ii_dedupe_staleness_s = 86400.0
    sigma_min = 1.5
    sigma_max = 5.0
    min_prompt_len = 20
    prompt_forbidden_patterns = ()
    # FSRS review knobs, read only on a semantic_fact re-teach (matches
    # tabula.config.AugurConfig's defaults).
    memory_s_growth_factor = 0.5
    memory_s_max = 365


# ── route(): tier + kind mapping, reason-aware correct_silence ─────────────


def test_correct_silence_targets_habituation_arm():
    ctx = C.DialogueContext(
        recent_suppressions=[
            {
                "state_key": "single:typing:user",
                "arm": "habituation",
                "reason": "habituated",
            }
        ]
    )
    pending = R.route(
        {
            "kind": "correct_silence",
            "target": "single:typing:user",
            "action": {},
            "rationale": "speak up",
        },
        ctx,
        pm=None,
        cfg=_Cfg(),
    )
    assert pending["tier"] == "light"
    p = pending["proposal"]
    assert p["kind"] == "gate_calibration"
    assert p["action"]["op"] in {"floor_set", "self_tolerance_remove"}
    assert p["source"] == "dialogue"


def test_correct_silence_habituation_arm_sets_floor_zero():
    ctx = C.DialogueContext(
        recent_suppressions=[{"state_key": "single:typing:user", "arm": "habituation"}]
    )
    pending = R.route(
        {
            "kind": "correct_silence",
            "target": "single:typing:user",
            "action": {},
            "rationale": "speak up",
        },
        ctx,
        pm=None,
        cfg=_Cfg(),
    )
    assert pending["proposal"]["action"] == {
        "op": "floor_set",
        "state_key": "single:typing:user",
        "value": 0.0,
    }


def test_correct_silence_central_tolerance_arm_removes_self_tolerance():
    ctx = C.DialogueContext(
        recent_suppressions=[
            {"state_key": "single:typing:user", "arm": "central_tolerance"}
        ]
    )
    pending = R.route(
        {
            "kind": "correct_silence",
            "target": "single:typing:user",
            "action": {},
            "rationale": "speak up",
        },
        ctx,
        pm=None,
        cfg=_Cfg(),
    )
    assert pending["proposal"]["action"] == {
        "op": "self_tolerance_remove",
        "state_key": "single:typing:user",
    }


def test_correct_silence_no_suppression_record_falls_back_to_self_tolerance_remove():
    pending = R.route(
        {
            "kind": "correct_silence",
            "target": "single:typing:user",
            "action": {},
            "rationale": "speak up",
        },
        C.DialogueContext(),
        pm=None,
        cfg=_Cfg(),
    )
    assert pending["proposal"]["action"] == {
        "op": "self_tolerance_remove",
        "state_key": "single:typing:user",
    }


def test_correct_silence_unmatched_arm_falls_back_to_self_tolerance_remove():
    ctx = C.DialogueContext(
        recent_suppressions=[
            {"state_key": "single:typing:user", "arm": "novelty_prediction_error"}
        ]
    )
    pending = R.route(
        {
            "kind": "correct_silence",
            "target": "single:typing:user",
            "action": {},
            "rationale": "speak up",
        },
        ctx,
        pm=None,
        cfg=_Cfg(),
    )
    assert pending["proposal"]["action"] == {
        "op": "self_tolerance_remove",
        "state_key": "single:typing:user",
    }


def test_correct_noise_is_light_self_tolerance_add():
    pending = R.route(
        {
            "kind": "correct_noise",
            "target": "single:typing:user",
            "action": {},
            "rationale": "too chatty",
        },
        C.DialogueContext(),
        pm=None,
        cfg=_Cfg(),
    )
    assert pending["tier"] == "light"
    assert pending["proposal"]["action"] == {
        "op": "self_tolerance_add",
        "state_key": "single:typing:user",
    }


def test_tune_rule_is_heavy():
    pending = R.route(
        {
            "kind": "tune_rule",
            "target": "chess+typing",
            "action": {"target": "MEDIUM"},
            "rationale": "treat as medium",
        },
        C.DialogueContext(),
        pm=None,
        cfg=_Cfg(),
    )
    assert pending["tier"] == "heavy"
    assert (
        pending["confirm_phrase"] and pending["proposal"]["kind"] == "escalation_rule"
    )


def test_teach_context_directive_routes():
    pending = R.route(
        {
            "kind": "teach_context_directive",
            "target": "night_mode",
            "action": {"op": "add"},
            "rationale": "quiet at night",
        },
        C.DialogueContext(),
        pm=None,
        cfg=_Cfg(),
    )
    assert pending["tier"] == "light"
    assert pending["proposal"]["kind"] == "context_directive"
    assert pending["proposal"]["action"] == {"op": "add"}


def test_teach_semantic_fact_routes():
    pending = R.route(
        {
            "kind": "teach_semantic_fact",
            "target": "user",
            "action": {
                "domains": ["typing"],
                "rule_key": "HIGH",
                "severity": "MEDIUM",
            },
            "rationale": "left-handed",
        },
        C.DialogueContext(),
        pm=None,
        cfg=_Cfg(),
    )
    assert pending["proposal"]["kind"] == "semantic_fact"
    assert "left-handed" in pending["echo"]
    # route() builds a pattern dict from the intent's action fields.
    assert pending["proposal"]["action"] == {
        "pattern": {
            "kind": "semantic",
            "domains": ["typing"],
            "rule_key": "HIGH",
            "severity": "MEDIUM",
        }
    }


def test_teach_semantic_fact_routes_defaults_with_empty_action():
    pending = R.route(
        {
            "kind": "teach_semantic_fact",
            "target": "user",
            "action": {},
            "rationale": "left-handed",
        },
        C.DialogueContext(),
        pm=None,
        cfg=_Cfg(),
    )
    assert pending["proposal"]["action"] == {
        "pattern": {
            "kind": "semantic",
            "domains": [],
            "rule_key": None,
            "severity": "LOW",
        }
    }


def test_teach_semantic_fact_routes_passes_through_explicit_pattern():
    explicit = {
        "kind": "semantic",
        "domains": ["chess"],
        "rule_key": None,
        "severity": "LOW",
    }
    pending = R.route(
        {
            "kind": "teach_semantic_fact",
            "target": "user",
            "action": {"pattern": explicit},
            "rationale": "castling habit",
        },
        C.DialogueContext(),
        pm=None,
        cfg=_Cfg(),
    )
    assert pending["proposal"]["action"] == {"pattern": explicit}


def test_correct_advice_quality_light_without_rewrite():
    pending = R.route(
        {
            "kind": "correct_advice_quality",
            "target": "typing",
            "action": {},
            "rationale": "bad advice",
        },
        C.DialogueContext(),
        pm=None,
        cfg=_Cfg(),
    )
    assert pending["tier"] == "light"
    assert pending["proposal"]["kind"] == "gate_calibration"
    assert pending["proposal"]["action"]["op"] == "self_tolerance_add"


def test_correct_advice_quality_heavy_with_rewrite():
    pending = R.route(
        {
            "kind": "correct_advice_quality",
            "target": "typing",
            "action": {"rewrite": True, "text": "be terser"},
            "rationale": "too verbose",
        },
        C.DialogueContext(),
        pm=None,
        cfg=_Cfg(),
    )
    assert pending["tier"] == "heavy"
    assert pending["confirm_phrase"] == "rewrite the prompt"
    assert pending["proposal"]["kind"] == "prompt_strategy"


def test_unknown_kind_raises():
    with pytest.raises(ValueError):
        R.route(
            {"kind": "query", "action": {}, "rationale": ""},
            C.DialogueContext(),
            pm=None,
            cfg=_Cfg(),
        )


# ── apply_confirmed(): confirmed dialogue apply ─────────────────────────────


def test_apply_confirmed_applies_escalation_rule_and_returns_echo():
    pm = _pm()
    pm.save_escalation_matrix({"version": "v1", "rules": {"LOW+LOW": "LOW"}})
    pending = R.route(
        {
            "kind": "tune_rule",
            "target": "LOW+LOW",
            "action": {"target": "MEDIUM"},
            "rationale": "medium",
        },
        C.DialogueContext(),
        pm=pm,
        cfg=_Cfg(),
    )
    out = R.apply_confirmed(pending, pm=pm, cfg=_Cfg(), session_id="d1")
    assert out["status"] == "applied"
    assert out["echo"] == pending["echo"]
    assert pm.load_escalation_matrix()["rules"]["LOW+LOW"] == "MEDIUM"


def test_apply_confirmed_logged_when_apply_disabled():
    pm = _pm()
    pm.save_escalation_matrix({"version": "v1", "rules": {"LOW+LOW": "LOW"}})
    pending = R.route(
        {
            "kind": "tune_rule",
            "target": "LOW+LOW",
            "action": {"target": "MEDIUM"},
            "rationale": "medium",
        },
        C.DialogueContext(),
        pm=pm,
        cfg=_Cfg(),
    )

    class _Off(_Cfg):
        dialogue_confirmed_apply_enabled = False

    out = R.apply_confirmed(pending, pm=pm, cfg=_Off(), session_id="d1")
    assert out["status"] == "logged"


# ── build_inverse(): rollback-anchor inversion ──────────────────────────────


def test_build_inverse_escalation_rule():
    p = P.make_proposal(
        kind="escalation_rule",
        target="LOW+LOW",
        action={"target": "MEDIUM", "prior_target": "LOW"},
        rationale="taught",
        source="dialogue",
    )
    inv = R.build_inverse({"proposal": p})
    assert inv["kind"] == "escalation_rule"
    assert inv["target"] == "LOW+LOW"
    assert inv["action"]["target"] == "LOW"
    assert inv["source"] == "dialogue"


def test_build_inverse_escalation_rule_no_prior_is_none():
    # A brand-new rule (nothing existed before) can't be automatically
    # un-added -- matrix_ops has no "remove key" op, only "set value".
    p = P.make_proposal(
        kind="escalation_rule",
        target="NEW+RULE",
        action={"target": "MEDIUM", "prior_target": None},
        rationale="taught",
        source="dialogue",
    )
    assert R.build_inverse({"proposal": p}) is None


def test_build_inverse_sigma():
    p = P.make_proposal(
        kind="sigma",
        target="typing",
        action={"domain": "typing", "sigma": 3.0, "prior_sigma": 2.0},
        rationale="quieter",
        source="dialogue",
    )
    inv = R.build_inverse({"proposal": p})
    assert inv["kind"] == "sigma"
    assert inv["action"] == {"domain": "typing", "sigma": 2.0}


def test_build_inverse_sigma_no_prior_is_none():
    p = P.make_proposal(
        kind="sigma",
        target="typing",
        action={"domain": "typing", "sigma": 3.0, "prior_sigma": None},
        rationale="quieter",
        source="dialogue",
    )
    assert R.build_inverse({"proposal": p}) is None


def test_build_inverse_prompt_strategy():
    p = P.make_proposal(
        kind="prompt_strategy",
        target="typing",
        action={"domain": "typing", "text": "new", "prior_text": "old"},
        rationale="rewrite",
        source="dialogue",
    )
    inv = R.build_inverse({"proposal": p})
    assert inv["kind"] == "prompt_strategy"
    assert inv["action"] == {"domain": "typing", "text": "old"}


def test_build_inverse_context_directive():
    p = P.make_proposal(
        kind="context_directive",
        target="night_mode",
        action={"directive_id": "abc123"},
        rationale="taught",
        source="dialogue",
    )
    inv = R.build_inverse({"proposal": p})
    assert inv["kind"] == "context_directive"
    assert inv["action"] == {"op": "remove", "directive_id": "abc123"}


def test_build_inverse_semantic_fact():
    p = P.make_proposal(
        kind="semantic_fact",
        target="user",
        action={"memory_id": "m1"},
        rationale="taught",
        source="dialogue",
    )
    inv = R.build_inverse({"proposal": p})
    assert inv["kind"] == "semantic_fact"
    assert inv["action"] == {"op": "remove", "memory_id": "m1"}


def test_build_inverse_self_tolerance_add_changed():
    p = P.make_proposal(
        kind="gate_calibration",
        target="single:typing:user",
        action={
            "op": "self_tolerance_add",
            "state_key": "single:typing:user",
            "prior": False,
        },
        rationale="stop flagging",
        source="dialogue",
    )
    inv = R.build_inverse({"proposal": p})
    assert inv["action"] == {
        "op": "self_tolerance_remove",
        "state_key": "single:typing:user",
    }


def test_build_inverse_self_tolerance_add_noop_is_none():
    p = P.make_proposal(
        kind="gate_calibration",
        target="single:typing:user",
        action={
            "op": "self_tolerance_add",
            "state_key": "single:typing:user",
            "prior": True,
        },
        rationale="stop flagging",
        source="dialogue",
    )
    assert R.build_inverse({"proposal": p}) is None


def test_build_inverse_self_tolerance_remove_changed():
    # Regression: a naive "not prior" check inverts the WRONG way for
    # remove (prior=True means membership WAS removed -- a real change).
    p = P.make_proposal(
        kind="gate_calibration",
        target="single:typing:user",
        action={
            "op": "self_tolerance_remove",
            "state_key": "single:typing:user",
            "prior": True,
        },
        rationale="flag again",
        source="dialogue",
    )
    inv = R.build_inverse({"proposal": p})
    assert inv["action"] == {
        "op": "self_tolerance_add",
        "state_key": "single:typing:user",
    }


def test_build_inverse_self_tolerance_remove_noop_is_none():
    p = P.make_proposal(
        kind="gate_calibration",
        target="single:typing:user",
        action={
            "op": "self_tolerance_remove",
            "state_key": "single:typing:user",
            "prior": False,
        },
        rationale="flag again",
        source="dialogue",
    )
    assert R.build_inverse({"proposal": p}) is None


def test_build_inverse_floor_set_restores_prior_floor():
    p = P.make_proposal(
        kind="gate_calibration",
        target="single:typing:user",
        action={
            "op": "floor_set",
            "state_key": "single:typing:user",
            "value": 0.5,
            "prior": {"floor": 0.1, "last_ts": 1.0},
        },
        rationale="habituate faster",
        source="dialogue",
    )
    inv = R.build_inverse({"proposal": p})
    assert inv["action"] == {
        "op": "floor_set",
        "state_key": "single:typing:user",
        "value": 0.1,
    }


def test_build_inverse_floor_set_no_prior_entry_restores_zero():
    p = P.make_proposal(
        kind="gate_calibration",
        target="single:typing:user",
        action={
            "op": "floor_set",
            "state_key": "single:typing:user",
            "value": 0.5,
            "prior": {},
        },
        rationale="habituate faster",
        source="dialogue",
    )
    inv = R.build_inverse({"proposal": p})
    assert inv["action"]["value"] == 0.0


def test_build_inverse_unknown_kind_is_none():
    p = P.make_proposal(
        kind="observe_more", target="x", action={}, rationale="", source="dialogue"
    )
    assert R.build_inverse({"proposal": p}) is None


# ── apply_undo(): reason-aware undo, distinct out-of-bounds outcome ────────


def test_apply_undo_no_inverse_available():
    p = P.make_proposal(
        kind="observe_more", target="x", action={}, rationale="", source="dialogue"
    )
    out = R.apply_undo({"proposal": p}, pm=None, cfg=_Cfg(), session_id="d1")
    assert out["status"] == "unavailable"
    assert out["proposal"] is None
    assert "no inverse" in out["reason"]


def test_apply_undo_blocked_when_prior_sigma_outside_current_bounds():
    # A prior_sigma of 10.0 might have been valid under a wider historical
    # [sigma_min, sigma_max] window; the CURRENT cfg only allows [1.5, 5.0].
    p = P.make_proposal(
        kind="sigma",
        target="typing",
        action={"domain": "typing", "sigma": 2.0, "prior_sigma": 10.0},
        rationale="quieter",
        source="dialogue",
    )
    out = R.apply_undo({"proposal": p}, pm=None, cfg=_Cfg(), session_id="d1")
    assert out["status"] == "blocked"
    assert out["reason"] == "cannot restore: prior value outside current bounds"
    assert out["proposal"]["action"]["sigma"] == 10.0  # inverse built, not applied


def test_apply_undo_blocked_when_prior_floor_outside_current_bounds():
    p = P.make_proposal(
        kind="gate_calibration",
        target="single:typing:user",
        action={
            "op": "floor_set",
            "state_key": "single:typing:user",
            "value": 0.5,
            "prior": {"floor": 0.9},
        },
        rationale="habituate",
        source="dialogue",
    )
    out = R.apply_undo({"proposal": p}, pm=None, cfg=_Cfg(), session_id="d1")
    assert out["status"] == "blocked"
    assert out["reason"] == "cannot restore: prior value outside current bounds"


def test_apply_undo_applies_when_in_bounds():
    pm = _pm()
    pm.save_thresholds("typing", {"sigma_threshold": 2.0})
    p = P.make_proposal(
        kind="sigma",
        target="typing",
        action={"domain": "typing", "sigma": 3.0, "prior_sigma": 2.0},
        rationale="quieter",
        source="dialogue",
    )
    out = R.apply_undo({"proposal": p}, pm=pm, cfg=_Cfg(), session_id="d1")
    assert out["status"] == "applied"
    assert out["reason"] is None
    assert pm.load_thresholds("typing")["sigma_threshold"] == 2.0


def test_apply_undo_escalation_rule_full_path():
    pm = _pm()
    pm.save_escalation_matrix({"version": "v1", "rules": {"LOW+LOW": "MEDIUM"}})
    p = P.make_proposal(
        kind="escalation_rule",
        target="LOW+LOW",
        action={"target": "MEDIUM", "prior_target": "LOW"},
        rationale="taught",
        source="dialogue",
    )
    out = R.apply_undo({"proposal": p}, pm=pm, cfg=_Cfg(), session_id="d1")
    assert out["status"] == "applied"
    assert pm.load_escalation_matrix()["rules"]["LOW+LOW"] == "LOW"


def test_apply_undo_logged_when_confirmed_apply_disabled():
    # In-bounds inverse, but the confirmed-apply flag is off downstream: the
    # bounds pre-check passes and apply_proposal itself returns "logged" --
    # the documented fourth outcome, distinct from unavailable/blocked.
    pm = _pm()
    pm.save_thresholds("typing", {"sigma_threshold": 2.0})
    forward = P.make_proposal(
        kind="sigma",
        target="typing",
        action={"domain": "typing", "sigma": 3.0},
        rationale="quieter",
        source="dialogue",
    )
    applied = R.apply_confirmed(
        {"proposal": forward, "echo": "e"}, pm=pm, cfg=_Cfg(), session_id="d1"
    )
    assert applied["status"] == "applied"  # real anchor recorded by the handler

    class _Off(_Cfg):
        dialogue_confirmed_apply_enabled = False

    out = R.apply_undo(
        {"proposal": applied["proposal"]}, pm=pm, cfg=_Off(), session_id="d1"
    )
    assert out["status"] == "logged"
    assert out["reason"] == "could not apply undo"
    assert pm.load_thresholds("typing")["sigma_threshold"] == 3.0  # untouched


def test_apply_confirmed_logged_on_handler_failure():
    # A proposal that fails INSIDE the handler (unrecognized gate_calibration
    # op), not at the flag/klass/kind gate: "logged" must propagate through
    # the router's return, and the anti-thrash gate must never arm.
    pm = _pm()
    p = P.make_proposal(
        kind="gate_calibration",
        target="single:typing:user",
        action={"op": "credibility_nudge", "state_key": "single:typing:user"},
        rationale="bad op",
        source="dialogue",
    )
    out = R.apply_confirmed(
        {"proposal": p, "echo": "e"}, pm=pm, cfg=_Cfg(), session_id="d1"
    )
    assert out["status"] == "logged"
    assert out["echo"] == "e"
    assert pm.is_proposal_applied(p["dedupe_key"]) is False  # never armed


# ── end-to-end round-trips: REAL forward apply -> inverse -> undo ───────────
#
# Every rollback anchor here is recorded by the real apply.py handler against
# fakeredis, never hand-authored -- this is what protects against anchor
# field-name drift between apply.py (writer) and router.build_inverse (reader).


def test_roundtrip_self_tolerance_add_undo_restores_membership():
    pm = _pm()
    assert pm.is_self_tolerant("single:typing:user") is False
    pending = R.route(
        {
            "kind": "correct_noise",
            "target": "single:typing:user",
            "action": {},
            "rationale": "too chatty",
        },
        C.DialogueContext(),
        pm=pm,
        cfg=_Cfg(),
    )
    applied = R.apply_confirmed(pending, pm=pm, cfg=_Cfg(), session_id="d1")
    assert applied["status"] == "applied"
    assert pm.is_self_tolerant("single:typing:user") is True

    inv = R.build_inverse({"proposal": applied["proposal"]})
    assert inv is not None and inv["action"]["op"] == "self_tolerance_remove"
    out = R.apply_undo(
        {"proposal": applied["proposal"]}, pm=pm, cfg=_Cfg(), session_id="d1"
    )
    assert out["status"] == "applied"
    assert pm.is_self_tolerant("single:typing:user") is False  # restored


def test_roundtrip_sigma_undo_restores_threshold():
    # route() has no sigma-producing intent kind, so the pending is built
    # around a real make_proposal -- the forward apply, anchor recording, and
    # undo all run through the genuine apply.py path.
    pm = _pm()
    pm.save_thresholds("typing", {"sigma_threshold": 2.0, "hst_threshold": 0.9})
    forward = P.make_proposal(
        kind="sigma",
        target="typing",
        action={"domain": "typing", "sigma": 3.5},
        rationale="quieter",
        source="dialogue",
    )
    applied = R.apply_confirmed(
        {"proposal": forward, "echo": "e"}, pm=pm, cfg=_Cfg(), session_id="d1"
    )
    assert applied["status"] == "applied"
    assert pm.load_thresholds("typing")["sigma_threshold"] == 3.5

    out = R.apply_undo(
        {"proposal": applied["proposal"]}, pm=pm, cfg=_Cfg(), session_id="d1"
    )
    assert out["status"] == "applied"
    # Restored, and the unrelated threshold field survived both writes.
    assert pm.load_thresholds("typing") == {
        "sigma_threshold": 2.0,
        "hst_threshold": 0.9,
    }


def test_roundtrip_floor_set_undo_restores_prior_floor():
    pm = _pm()
    pm.save_habituation_floor("single:typing:user", {"floor": 0.3, "last_ts": 1.0})
    ctx = C.DialogueContext(
        recent_suppressions=[{"state_key": "single:typing:user", "arm": "habituation"}]
    )
    pending = R.route(
        {
            "kind": "correct_silence",
            "target": "single:typing:user",
            "action": {},
            "rationale": "speak up",
        },
        ctx,
        pm=pm,
        cfg=_Cfg(),
    )
    assert pending["proposal"]["action"]["op"] == "floor_set"
    applied = R.apply_confirmed(pending, pm=pm, cfg=_Cfg(), session_id="d1")
    assert applied["status"] == "applied"
    assert pm.load_habituation_floor("single:typing:user")["floor"] == 0.0

    out = R.apply_undo(
        {"proposal": applied["proposal"]}, pm=pm, cfg=_Cfg(), session_id="d1"
    )
    assert out["status"] == "applied"
    assert pm.load_habituation_floor("single:typing:user")["floor"] == 0.3


def test_roundtrip_escalation_rule_undo_restores_rule():
    pm = _pm()
    pm.save_escalation_matrix({"version": "v1", "rules": {"LOW+LOW": "LOW"}})
    pending = R.route(
        {
            "kind": "tune_rule",
            "target": "LOW+LOW",
            "action": {"target": "MEDIUM"},
            "rationale": "medium",
        },
        C.DialogueContext(),
        pm=pm,
        cfg=_Cfg(),
    )
    applied = R.apply_confirmed(pending, pm=pm, cfg=_Cfg(), session_id="d1")
    assert applied["status"] == "applied"
    assert pm.load_escalation_matrix()["rules"]["LOW+LOW"] == "MEDIUM"

    out = R.apply_undo(
        {"proposal": applied["proposal"]}, pm=pm, cfg=_Cfg(), session_id="d1"
    )
    assert out["status"] == "applied"
    assert pm.load_escalation_matrix()["rules"]["LOW+LOW"] == "LOW"


def test_roundtrip_prompt_strategy_undo_restores_text():
    pm = _pm()
    old_text = "You are a helpful typing coach. Be concise and specific."
    new_text = "You are a terse typing coach. One sentence max, no fluff."
    pm.save_prompt("typing", old_text)
    pending = R.route(
        {
            "kind": "correct_advice_quality",
            "target": "typing",
            "action": {"rewrite": True, "domain": "typing", "text": new_text},
            "rationale": "too verbose",
        },
        C.DialogueContext(),
        pm=pm,
        cfg=_Cfg(),
    )
    assert pending["tier"] == "heavy"
    applied = R.apply_confirmed(pending, pm=pm, cfg=_Cfg(), session_id="d1")
    assert applied["status"] == "applied"
    assert pm.load_prompt("typing") == new_text

    out = R.apply_undo(
        {"proposal": applied["proposal"]}, pm=pm, cfg=_Cfg(), session_id="d1"
    )
    assert out["status"] == "applied"
    assert pm.load_prompt("typing") == old_text


def test_roundtrip_context_directive_undo_of_undo_restores_directive():
    # teach -> confirm-apply (create) -> undo (removes, records the removed
    # content as prior_directive) -> undo of THAT undo re-adds it: Task 20
    # decision A's complete restore semantics replaces the earlier
    # "unavailable" stopgap for a removal that captured a real prior.
    pm = _pm()
    forward = P.make_proposal(
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
    applied = R.apply_confirmed(
        {"proposal": forward, "echo": "e"}, pm=pm, cfg=_Cfg(), session_id="d1"
    )
    assert applied["status"] == "applied"
    assert len(pm.load_dialogue_directives()) == 1
    did = applied["proposal"]["action"]["directive_id"]

    undo = R.apply_undo(
        {"proposal": applied["proposal"]}, pm=pm, cfg=_Cfg(), session_id="d1"
    )
    assert undo["status"] == "applied"
    assert pm.load_dialogue_directives() == []
    # The removed content was captured as the restore anchor.
    assert undo["proposal"]["action"]["prior_directive"]["directive_id"] == did

    # The audited record of the undo is an applied REMOVE-WITH-PRIOR: its
    # inverse re-adds the exact removed content.
    inv = R.build_inverse({"proposal": undo["proposal"]})
    assert inv is not None
    assert inv["action"] == {
        "directive_id": did,
        "predicate": {"context": "focused_app", "match": "appX"},
        "action": "suppress",
        "scope": "all",
        "rationale": "deep work",
    }

    out = R.apply_undo(
        {"proposal": undo["proposal"]}, pm=pm, cfg=_Cfg(), session_id="d1"
    )
    assert out["status"] == "applied"
    restored = pm.load_dialogue_directives()
    assert len(restored) == 1
    assert restored[0]["directive_id"] == did
    assert restored[0]["predicate"] == {"context": "focused_app", "match": "appX"}
    # The original user-facing rationale is restored too, not overwritten
    # with the generic "undo" audit rationale.
    assert restored[0]["rationale"] == "deep work"


def test_roundtrip_context_directive_upsert_undo_restores_prior_content():
    # teach directive A -> apply (fresh create, no prior) -> teach directive A
    # AGAIN with the SAME directive_id but a DIFFERENT predicate (upsert) ->
    # apply records the pre-upsert content as prior_directive -> undo
    # RESTORES that prior content (Task 20 decision A), rather than deleting
    # the directive outright.
    pm = _pm()
    first = P.make_proposal(
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
    applied1 = R.apply_confirmed(
        {"proposal": first, "echo": "e"}, pm=pm, cfg=_Cfg(), session_id="d1"
    )
    assert applied1["status"] == "applied"
    did = applied1["proposal"]["action"]["directive_id"]

    second = P.make_proposal(
        kind="context_directive",
        target="appX",
        action={
            "directive_id": did,
            "predicate": {"context": "focused_app", "match": "appX_v2"},
            "action": "downgrade",
            "scope": "single",
        },
        rationale="narrower scope",
        source="dialogue",
    )
    applied2 = R.apply_confirmed(
        {"proposal": second, "echo": "e"}, pm=pm, cfg=_Cfg(), session_id="d1"
    )
    assert applied2["status"] == "applied"
    assert applied2["proposal"]["action"]["prior_directive"]["predicate"] == {
        "context": "focused_app",
        "match": "appX",
    }
    current = pm.load_dialogue_directives()
    assert len(current) == 1
    assert current[0]["predicate"] == {"context": "focused_app", "match": "appX_v2"}

    out = R.apply_undo(
        {"proposal": applied2["proposal"]}, pm=pm, cfg=_Cfg(), session_id="d1"
    )
    assert out["status"] == "applied"
    restored = pm.load_dialogue_directives()
    assert len(restored) == 1  # restored in place, NOT deleted
    assert restored[0]["directive_id"] == did
    assert restored[0]["predicate"] == {"context": "focused_app", "match": "appX"}
    assert restored[0]["action"] == "suppress"
    assert restored[0]["scope"] == "all"
    assert restored[0]["rationale"] == "deep work"


def test_roundtrip_semantic_fact_remove_undo_readds_content():
    # teach -> confirm-apply (fresh create) -> undo (archives, records the
    # pre-archive state as prior_fact) -> undo of THAT undo re-teaches the
    # removed content (Task 20 decision A), reactivating it.
    pm = _pm()
    pattern = {
        "kind": "semantic",
        "domains": ["chess", "typing"],
        "rule_key": "HIGH+HIGH",
        "severity": "MEDIUM",
    }
    forward = P.make_proposal(
        kind="semantic_fact",
        target="chess+typing",
        action={"pattern": pattern},
        rationale="taught",
        source="dialogue",
    )
    applied = R.apply_confirmed(
        {"proposal": forward, "echo": "e"}, pm=pm, cfg=_Cfg(), session_id="d1"
    )
    assert applied["status"] == "applied"
    mid = applied["proposal"]["action"]["memory_id"]
    assert applied["proposal"]["action"]["prior_fact"] is None  # fresh create
    assert pm.load_memory_state(mid)["status"] == "active"

    undo = R.apply_undo(
        {"proposal": applied["proposal"]}, pm=pm, cfg=_Cfg(), session_id="d1"
    )
    assert undo["status"] == "applied"
    assert pm.load_memory_state(mid)["status"] == "archived"
    assert undo["proposal"]["action"]["prior_fact"] is not None  # captured

    inv = R.build_inverse({"proposal": undo["proposal"]})
    assert inv is not None and inv["action"] == {"pattern": pattern}

    readd = R.apply_undo(
        {"proposal": undo["proposal"]}, pm=pm, cfg=_Cfg(), session_id="d1"
    )
    assert readd["status"] == "applied"
    assert pm.load_memory_state(mid)["status"] == "active"  # reactivated
    assert mid in [f["memory_id"] for f in pm.load_taught_facts()]


def test_roundtrip_semantic_fact_reteach_undo_restores_prior_pattern():
    # A re-teach (upsert-with-prior) undo restores the PRIOR pattern content
    # by re-teaching it -- itself a review, per decision B: even an undo is
    # forward decay, never a raw state rollback.
    pm = _pm()
    pattern = {
        "kind": "semantic",
        "domains": ["typing"],
        "rule_key": None,
        "severity": "LOW",
    }
    first = P.make_proposal(
        kind="semantic_fact",
        target="typing",
        action={"pattern": pattern},
        rationale="left-handed",
        source="dialogue",
    )
    applied1 = R.apply_confirmed(
        {"proposal": first, "echo": "e"}, pm=pm, cfg=_Cfg(), session_id="d1"
    )
    assert applied1["status"] == "applied"
    mid = applied1["proposal"]["action"]["memory_id"]
    first_state = pm.load_memory_state(mid)
    assert first_state["S"] == 1.0

    # Re-teach the SAME pattern (upsert-with-prior: same deterministic id).
    second = P.make_proposal(
        kind="semantic_fact",
        target="typing",
        action={"pattern": pattern},
        rationale="left-handed again",
        source="dialogue",
    )
    applied2 = R.apply_confirmed(
        {"proposal": second, "echo": "e"}, pm=pm, cfg=_Cfg(), session_id="d1"
    )
    assert applied2["status"] == "applied"
    assert applied2["proposal"]["action"]["prior_fact"]["S"] == 1.0
    strengthened = pm.load_memory_state(mid)
    assert strengthened["S"] > first_state["S"]

    inv = R.build_inverse({"proposal": applied2["proposal"]})
    assert inv is not None and inv["action"] == {"pattern": pattern}

    # A different session for the undo: review()'s idempotency is per
    # session, so re-using "d1" here would be a no-op review, not a
    # meaningful demonstration of forward-only decay.
    out = R.apply_undo(
        {"proposal": applied2["proposal"]}, pm=pm, cfg=_Cfg(), session_id="d2"
    )
    assert out["status"] == "applied"
    # The undo itself re-teaches (reviews) the prior pattern -- S strengthens
    # further rather than rolling back to the pre-re-teach value; this is the
    # intended decision-B semantics (forward-only decay, even on undo).
    assert pm.load_memory_state(mid)["S"] > strengthened["S"]
    assert pm.load_memory_state(mid)["pattern"] == pattern
