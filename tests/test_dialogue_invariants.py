"""Property invariants I1–I4 for the dialogue write-path.

- I1: a proposal with klass="gated" never applies, even confirmed — pinned
  against kinds that HAVE a confirmed dispatch case, so the klass check in
  apply._apply_confirmed is the only thing standing between them and a write.
- I2: every applied proposal carries its reversibility anchor.
- I3: an intent-PARSING turn never applies — engine.handle_turn only saves a
  pending record; state moves on a separate confirmation turn.
- I4: the confirmed (dialogue) and autonomous (Imperator II) paths are gated
  independently, and the autonomous path's is_auto_applicable floor holds.

Each test's docstring records the sabotage run that proved its teeth: the
actual guard was locally broken, the test failed, the guard was reverted.
"""

from __future__ import annotations

import asyncio

import fakeredis
import pytest

from tabula.persistence import PersistenceManager
from imperator import apply as A, proposals as P
from imperator.dialogue import engine as E, router as R


def _pm() -> PersistenceManager:
    """Return a fresh PersistenceManager backed by fakeredis."""
    return PersistenceManager(fakeredis.FakeStrictRedis(decode_responses=False))


class _Cfg:
    """Test config: dialogue confirmed apply enabled, II apply disabled."""

    dialogue_confirmed_apply_enabled = True
    imperator_ii_apply_enabled = False
    # Read directly (no getattr default) by apply._arm_gate; without it the
    # anti-thrash marker write raises and every apply fails closed.
    imperator_ii_dedupe_staleness_s = 86400.0
    min_prompt_len = 20
    prompt_forbidden_patterns = ()
    sigma_min = 1.5
    sigma_max = 5.0
    # Engine/context/persona knobs for the handle_turn-level I3 test.
    dialogue_num_predict = 512
    dialogue_context_max_turns = 12
    dialogue_context_token_budget = 2048
    dialogue_pending_ttl_s = 300.0


class _NC:
    """NATS spy: records published subjects."""

    def __init__(self):
        self.published = []

    async def publish(self, subj, data=b""):
        self.published.append(subj)


_OLD_PROMPT = "Old prompt text long enough to satisfy the minimum length."
_NEW_PROMPT = "New prompt text long enough to satisfy the minimum length."


def _gated_variant(pm, kind: str):
    """Build a proposal of a kind that HAS a confirmed dispatch case, with
    klass force-set to "gated" AFTER construction (bypassing normalize_klass),
    plus a zero-arg checker returning True iff its target surface is unchanged.

    State is set up so the apply WOULD succeed if the klass guard vanished —
    that is what gives the I1 test teeth.
    """
    if kind == "escalation_rule":
        pm.save_escalation_matrix({"version": "v", "rules": {"LOW+LOW": "LOW"}})
        p = P.make_proposal(
            kind=kind,
            target="LOW+LOW",
            action={"target": "MEDIUM"},
            rationale="r",
            source="dialogue",
        )

        def unchanged():
            return pm.load_escalation_matrix()["rules"]["LOW+LOW"] == "LOW"

    elif kind == "sigma":
        pm.save_thresholds("typing", {"sigma_threshold": 2.0})
        p = P.make_proposal(
            kind=kind,
            target="typing",
            action={"domain": "typing", "sigma": 3.0},
            rationale="r",
            source="dialogue",
        )

        def unchanged():
            return pm.load_thresholds("typing")["sigma_threshold"] == 2.0

    elif kind == "prompt_strategy":
        pm.save_prompt("typing", _OLD_PROMPT)
        p = P.make_proposal(
            kind=kind,
            target="typing",
            action={"domain": "typing", "text": _NEW_PROMPT},
            rationale="r",
            source="dialogue",
        )

        def unchanged():
            return pm.load_prompt("typing") == _OLD_PROMPT

    elif kind == "gate_calibration":
        p = P.make_proposal(
            kind=kind,
            target="k1",
            action={"op": "self_tolerance_add", "state_key": "k1"},
            rationale="r",
            source="dialogue",
        )

        def unchanged():
            return pm.is_self_tolerant("k1") is False

    else:
        raise AssertionError(f"unexpected kind {kind}")
    p["klass"] = "gated"  # forced AFTER construction: only the guard blocks it
    return p, unchanged


# ============================================================================
# I1: klass="gated" never applies, even confirmed
# ============================================================================


@pytest.mark.parametrize(
    "kind", ["escalation_rule", "sigma", "prompt_strategy", "gate_calibration"]
)
def test_I1_gated_klass_never_applies_confirmed(kind: str):
    """I1: for every kind with a confirmed dispatch case, a klass="gated"
    proposal stays logged and its target surface stays untouched. The kinds
    are all in P._CONFIRMED_APPLY_KINDS and fully dispatchable, so the klass
    check in apply._apply_confirmed is the ONLY preventer here.

    Sabotage run: dropped `p.get("klass") != "safe" or` from the guard in
    _apply_confirmed -> all four parametrizations failed, e.g.
    `AssertionError: gated escalation_rule proposal applied via confirmed
    path — assert 'applied' == 'logged'`. Reverted.
    """
    pm = _pm()
    p, unchanged = _gated_variant(pm, kind)
    result = A.apply_proposal(pm, p, cfg=_Cfg(), session_id="d", confirmed=True)
    assert result["status"] == "logged", (
        f"gated {kind} proposal applied via confirmed path"
    )
    assert unchanged(), f"gated {kind} proposal mutated its target surface"


def test_I1_code_structural_normalize_to_gated():
    """I1 support: normalize_klass deterministically classifies code and
    structural kinds as "gated", overriding any LLM-claimed "safe" klass —
    the classification the pipeline relies on before apply_proposal ever
    sees a proposal.

    (code/structural additionally have no dispatch case in apply.py, so they
    are doubly blocked; the empirical apply-level pin for the klass guard
    itself is test_I1_gated_klass_never_applies_confirmed above.)

    Sabotage run: set `"code": "safe"` in P._KIND_KLASS -> failed with
    `assert 'safe' == 'gated'`. Reverted.
    """
    for kind in ("code", "structural"):
        p = P.make_proposal(
            kind=kind, target="x", action={}, rationale="r", klass="safe"
        )  # LLM claims safe
        assert P.normalize_klass(p)["klass"] == "gated"


# ============================================================================
# I2: Applied proposals carry reversibility anchors
# ============================================================================


def test_I2_applied_sigma_has_anchor():
    """I2: an applied sigma proposal records prior_sigma for rollback.

    Sabotage run: commented out `a["prior_sigma"] = current.get(...)` in
    apply._apply_sigma -> failed with `AssertionError: applied sigma proposal
    must carry prior_sigma anchor`. Reverted.
    """
    pm = _pm()
    pm.save_thresholds("typing", {"sigma_threshold": 2.0})
    p = P.make_proposal(
        kind="sigma",
        target="typing",
        action={"domain": "typing", "sigma": 3.0},
        rationale="r",
        source="dialogue",
    )
    result = A.apply_proposal(pm, p, cfg=_Cfg(), session_id="d", confirmed=True)
    assert result["status"] == "applied"
    assert "prior_sigma" in result["action"], (
        "applied sigma proposal must carry prior_sigma anchor"
    )
    assert result["action"]["prior_sigma"] == 2.0


def test_I2_applied_escalation_rule_has_anchor():
    """I2: an applied escalation_rule proposal records prior_target.

    Sabotage run: replaced the prior_target assignment in
    apply._apply_escalation_rule with `pass` -> failed with
    `AssertionError: applied escalation_rule proposal must carry prior_target
    anchor`. Reverted.
    """
    pm = _pm()
    pm.save_escalation_matrix({"version": "v", "rules": {"LOW+LOW": "LOW"}})
    p = P.make_proposal(
        kind="escalation_rule",
        target="LOW+LOW",
        action={"target": "MEDIUM"},
        rationale="r",
        source="dialogue",
    )
    result = A.apply_proposal(pm, p, cfg=_Cfg(), session_id="d", confirmed=True)
    assert result["status"] == "applied"
    assert "prior_target" in result["action"], (
        "applied escalation_rule proposal must carry prior_target anchor"
    )
    assert result["action"]["prior_target"] == "LOW"


def test_I2_applied_prompt_strategy_has_anchor():
    """I2: an applied prompt_strategy proposal records prior_text.

    Sabotage run: commented out `action["prior_text"] = current` in
    apply._apply_prompt_strategy -> failed with `AssertionError: applied
    prompt_strategy proposal must carry prior_text anchor`. Reverted.
    """
    pm = _pm()
    pm.save_prompt("typing", _OLD_PROMPT)
    p = P.make_proposal(
        kind="prompt_strategy",
        target="typing",
        action={"domain": "typing", "text": _NEW_PROMPT},
        rationale="r",
        source="dialogue",
    )
    result = A.apply_proposal(pm, p, cfg=_Cfg(), session_id="d", confirmed=True)
    assert result["status"] == "applied"
    assert "prior_text" in result["action"], (
        "applied prompt_strategy proposal must carry prior_text anchor"
    )
    assert result["action"]["prior_text"] == _OLD_PROMPT


def test_I2_applied_gate_calibration_has_anchor():
    """I2: applied gate_calibration proposals record the prior state for both
    self_tolerance_add and floor_set ops.

    Sabotage run: commented out `a["prior"] = prior` (tolerance branch) in
    apply._apply_gate_calibration -> failed with `AssertionError: applied
    gate_calibration (self_tolerance_add) must carry prior`; likewise
    `a["prior"] = prior_entry` (floor branch) -> the floor_set assertion
    failed. Reverted both.
    """
    pm = _pm()
    p_add = P.make_proposal(
        kind="gate_calibration",
        target="test_key",
        action={"op": "self_tolerance_add", "state_key": "test_key"},
        rationale="r",
        source="dialogue",
    )
    out_add = A.apply_proposal(pm, p_add, cfg=_Cfg(), session_id="d", confirmed=True)
    assert out_add["status"] == "applied"
    assert "prior" in out_add["action"], (
        "applied gate_calibration (self_tolerance_add) must carry prior"
    )

    p_floor = P.make_proposal(
        kind="gate_calibration",
        target="test_key",
        action={"op": "floor_set", "state_key": "test_key", "value": 0.3},
        rationale="r",
        source="dialogue",
    )
    out_floor = A.apply_proposal(
        pm, p_floor, cfg=_Cfg(), session_id="d", confirmed=True
    )
    assert out_floor["status"] == "applied"
    assert "prior" in out_floor["action"], (
        "applied gate_calibration (floor_set) must carry prior"
    )


# ============================================================================
# I3: an intent-parsing turn never applies (engine level)
# ============================================================================


def test_I3_intent_turn_never_applies():
    """I3: when handle_turn parses a teach intent, it ONLY saves a pending
    record — no apply, no audit entry, no mutation of the intent's actual
    target surface (here: the self-tolerance set a correct_silence confirm
    would shrink), no apply/trigger events published.

    Sabotage run: added an eager
    `R.apply_confirmed(new_pending, pm=pm, cfg=cfg, session_id=session_id)`
    right after `new_pending = R.route(...)` in engine.handle_turn's intent
    branch -> failed with `AssertionError: intent turn mutated its target
    surface — assert False is True` (self-tolerance was removed on the parse
    turn). Reverted.
    """
    pm, nc = _pm(), _NC()
    pm.add_self_tolerance("single:typing:user")  # the surface a confirm would touch

    async def llm_intent(prompt, system, client, cfg):
        return (
            '{"reply": "I will speak up.", "needs_clarification": false,'
            ' "question": null,'
            ' "intent": {"kind": "correct_silence", "target": "single:typing:user",'
            ' "action": {}, "rationale": "speak up"}}'
        )

    turn = asyncio.run(
        E.handle_turn(
            "s1",
            "you should've spoken",
            pm=pm,
            nc=nc,
            http_client=None,
            cfg=_Cfg(),
            query_fn=llm_intent,
        )
    )

    assert turn.intent is not None and turn.pending is not None
    assert turn.applied is None, "intent-parsing turn must not apply"
    pending = pm.load_dialogue_pending("s1")
    assert pending is not None, "pending record must be saved"
    assert pending["proposal"]["status"] == "logged"
    assert pm.load_dialogue_audit(limit=5) == [], "no audit entry before confirm"
    assert pm.is_self_tolerant("single:typing:user") is True, (
        "intent turn mutated its target surface"
    )
    assert "augur.imperator.dialogue.applied" not in nc.published
    assert "augur.imperator.ii.trigger" not in nc.published


def test_route_is_pure_no_state_writes():
    """Supplementary (NOT the I3 guard): router.route() is a pure translation
    layer — it takes pm/cfg for interface symmetry but writes nothing. This
    pins route() staying deterministic; the engine-level I3 pin is
    test_I3_intent_turn_never_applies above.
    """
    pm = _pm()
    pm.save_thresholds("typing", {"sigma_threshold": 2.0})
    pm.save_escalation_matrix({"version": "v", "rules": {"LOW+LOW": "LOW"}})
    matrix_before = pm.load_escalation_matrix()
    thresholds_before = pm.load_thresholds("typing")

    ctx = type("Ctx", (), {"recent_suppressions": [], "focused_app": "test_app"})()
    intent = {
        "kind": "teach_context_directive",
        "target": "test_directive",
        "action": {"action": "suppress", "scope": "all"},
        "rationale": "teaching test",
    }
    pending = R.route(intent, ctx, pm=pm, cfg=_Cfg())

    assert pm.load_escalation_matrix() == matrix_before
    assert pm.load_thresholds("typing") == thresholds_before
    assert pending["proposal"]["kind"] == "context_directive"
    assert pending["proposal"]["status"] == "logged"


# ============================================================================
# I4: Confirmed and autonomous paths are independent
# ============================================================================


def test_I4_paths_independent():
    """I4: with imperator_ii_apply_enabled=False and
    dialogue_confirmed_apply_enabled=True, the SAME safe escalation_rule
    change logs on the autonomous path but applies on the confirmed path —
    the two flags gate their paths orthogonally.

    Sabotage run: flipped `if confirmed:` to `if False:` in
    apply.apply_proposal (killing the confirmed routing) -> failed with
    `AssertionError: dialogue proposal should apply when
    dialogue_confirmed_apply_enabled=True — assert 'logged' == 'applied'`.
    Reverted.
    """
    pm = _pm()
    pm.save_escalation_matrix({"version": "v", "rules": {"LOW+LOW": "LOW"}})

    p_ii = P.make_proposal(
        kind="escalation_rule",
        target="LOW+LOW",
        action={"target": "MEDIUM"},
        rationale="r",
        source="imperator_ii",
    )
    p_dlg = P.make_proposal(
        kind="escalation_rule",
        target="LOW+LOW",
        action={"target": "MEDIUM"},
        rationale="r",
        source="dialogue",
    )

    out_ii = A.apply_proposal(pm, p_ii, cfg=_Cfg(), session_id="d", confirmed=False)
    assert out_ii["status"] == "logged", (
        "II proposal should log when imperator_ii_apply_enabled=False"
    )
    out_dlg = A.apply_proposal(pm, p_dlg, cfg=_Cfg(), session_id="d", confirmed=True)
    assert out_dlg["status"] == "applied", (
        "dialogue proposal should apply when dialogue_confirmed_apply_enabled=True"
    )


def test_I4_is_auto_applicable_membership():
    """I4: P.is_auto_applicable (the autonomous path's floor) is True for
    exactly the AUTO kinds (escalation_rule, prompt_strategy) with safe klass
    and logged status — and False for every other kind in the taxonomy,
    including the confirmed-only safe kinds (sigma, gate_calibration,
    context_directive, semantic_fact, observe_more) and the gated ones.

    Sabotage run: added "sigma" to P._AUTO_APPLY_KINDS -> failed with
    `AssertionError: sigma must not be auto-applicable`. Reverted.
    """
    auto = {"escalation_rule", "prompt_strategy"}
    # Independent expected-klass table (NOT derived from P._KIND_KLASS): the
    # prior `assert p["klass"] == klass` compared two values both read from
    # _KIND_KLASS, so it could never catch a wrong classification. This pins
    # normalize_klass against a hardcoded ground truth instead.
    expected_klass = {
        "escalation_rule": "safe",
        "prompt_strategy": "safe",
        "sigma": "safe",
        "gate_calibration": "safe",
        "observe_more": "safe",
        "context_directive": "safe",
        "semantic_fact": "safe",
        "code": "gated",
        "structural": "gated",
    }
    assert set(P._KIND_KLASS) == set(expected_klass), (
        "taxonomy changed — update the independent expected_klass table"
    )
    for kind, want_klass in expected_klass.items():
        p = P.make_proposal(kind=kind, target="t", action={}, rationale="r")
        P.normalize_klass(p)
        assert p["klass"] == want_klass, f"{kind} klass regressed"
        if kind in auto:
            assert P.is_auto_applicable(p), f"{kind} must be auto-applicable"
        else:
            assert not P.is_auto_applicable(p), f"{kind} must not be auto-applicable"
    # status floor: an already-applied AUTO proposal is not auto-applicable
    p = P.make_proposal(kind="escalation_rule", target="t", action={}, rationale="r")
    p["status"] = "applied"
    assert not P.is_auto_applicable(p)


def test_I4_autonomous_gate_blocks_non_auto_applicable():
    """I4: on the autonomous path with imperator_ii_apply_enabled=True, the
    is_auto_applicable check is the ONLY preventer for a klass="gated"
    escalation_rule — the kind is fully dispatchable there and the matrix
    write would succeed, so this empirically pins the gate itself (not a
    missing dispatch case).

    Sabotage run: dropped `or not P.is_auto_applicable(p)` from
    apply.apply_proposal -> failed with `AssertionError: autonomous path must
    honor is_auto_applicable — assert 'applied' == 'logged'`. Reverted.
    """
    pm = _pm()
    pm.save_escalation_matrix({"version": "v", "rules": {"LOW+LOW": "LOW"}})

    class _CfgIIOn(_Cfg):
        imperator_ii_apply_enabled = True

    p = P.make_proposal(
        kind="escalation_rule",
        target="LOW+LOW",
        action={"target": "MEDIUM"},
        rationale="r",
    )
    p["klass"] = "gated"  # not auto-applicable; everything else would succeed
    out = A.apply_proposal(pm, p, cfg=_CfgIIOn(), session_id="d", confirmed=False)
    assert out["status"] == "logged", "autonomous path must honor is_auto_applicable"
    assert pm.load_escalation_matrix()["rules"]["LOW+LOW"] == "LOW"
