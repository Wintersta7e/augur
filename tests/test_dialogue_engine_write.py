import asyncio

import fakeredis

from tabula.config import AugurConfig
from tabula.persistence import PersistenceManager
from imperator.dialogue import engine as E
from limen import gate as G
from tests.conftest import SINGLE_MEDIUM_TYPING

_GATE_CFG = AugurConfig()  # real gate_* bounds for the taught-directive round-trip


class _Cfg:
    dialogue_num_predict = 512
    dialogue_context_max_turns = 12
    dialogue_context_token_budget = 2048
    dialogue_pending_ttl_s = 300.0
    dialogue_confirmed_apply_enabled = True
    min_prompt_len = 20
    prompt_forbidden_patterns = ()
    # Real cfg default (tabula/config.py); the brief's fixture omitted it, but
    # imperator/apply.py's _arm_gate reads it directly (no getattr default) --
    # without it, the anti-thrash marker write raises AttributeError and
    # _arm_gate fails closed, so nothing in these tests would ever apply.
    imperator_ii_dedupe_staleness_s = 86400.0
    sigma_min = 1.5
    sigma_max = 5.0


def _pm():
    pm = PersistenceManager(fakeredis.FakeStrictRedis(decode_responses=False))
    pm.add_self_tolerance(
        "single:typing:user"
    )  # so correct_silence has something to undo
    return pm


class _NC:
    def __init__(self):
        self.published = []

    async def publish(self, subj, data=b""):
        self.published.append(subj)


class _CallSpy:
    """Records whether query_fn was invoked, without relying on exception
    propagation -- handle_turn's LLM-call try/except would swallow a raised
    AssertionError as an ordinary fail-soft error, hiding a real bug where
    the pending short-circuit fails to short-circuit."""

    def __init__(self):
        self.calls = 0

    async def __call__(self, prompt, system, client, cfg):
        self.calls += 1
        return (
            '{"reply": "unexpected LLM call", "needs_clarification": false,'
            ' "question": null, "intent": null}'
        )


def test_teach_then_confirm_applies_and_audits():
    pm, nc = _pm(), _NC()

    async def llm_intent(prompt, system, client, cfg):
        return (
            '{"reply": "I will speak up.", "needs_clarification": false,'
            ' "question": null,'
            ' "intent": {"kind": "correct_silence", "target": "single:typing:user",'
            ' "action": {}, "rationale": "speak up"}}'
        )

    t1 = asyncio.run(
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
    assert t1.pending is not None and t1.applied is None  # awaiting confirm

    # second turn: a bare "yes" confirms (LLM not consulted on a pending confirm)
    t2 = asyncio.run(
        E.handle_turn(
            "s1", "yes", pm=pm, nc=nc, http_client=None, cfg=_Cfg(), query_fn=llm_intent
        )
    )
    assert t2.applied is not None and t2.applied["status"] == "applied"
    assert pm.is_self_tolerant("single:typing:user") is False  # arm reversed
    audit = pm.load_dialogue_audit(limit=5)
    assert audit and audit[0]["status"] == "applied"
    # audit provenance: who confirmed (session_id) and what turn confirmed it
    assert audit[0]["session_id"] == "s1"
    assert audit[0]["confirming_text"] == "yes"
    assert "augur.imperator.dialogue.applied" in nc.published
    assert "augur.imperator.ii.trigger" in nc.published

    # a THIRD bare "yes" must not re-apply -- pending was cleared on t2, so
    # this hits the "nothing pending" truthful reply, not a second apply.
    spy = _CallSpy()
    t3 = asyncio.run(
        E.handle_turn(
            "s1", "yes", pm=pm, nc=nc, http_client=None, cfg=_Cfg(), query_fn=spy
        )
    )
    assert t3.applied is None
    assert spy.calls == 0  # short-circuited before any LLM call
    assert len(pm.load_dialogue_audit(limit=5)) == 1  # no new audit entry


def test_null_reply_from_llm_fails_truthful_not_crash():
    """F5 regression: an LLM emitting {"reply": null} (valid JSON, no schema)
    must be caught by the fail-truthful path -- not crash the turn with a
    TypeError nor leak the literal "None" into the reply (invariant 7)."""
    pm, nc = _pm(), _NC()

    async def llm_null(prompt, system, client, cfg):
        return (
            '{"reply": null, "intent": null, "needs_clarification": false,'
            ' "question": null}'
        )

    turn = asyncio.run(
        E.handle_turn(
            "s1", "hi", pm=pm, nc=nc, http_client=None, cfg=_Cfg(), query_fn=llm_null
        )
    )
    assert turn.error is not None
    assert "None" not in turn.reply
    assert turn.reply == "I can't reason about that right now."


def test_publish_failure_after_commit_still_reports_applied():
    """F16 regression: a NATS publish failure AFTER a confirmed apply commits
    must not surface the committed change as "turn failed" (invariant 7 in
    reverse). The apply is truthful; the dropped event is logged, not fatal."""

    class _FailNC:
        async def publish(self, subj, data=b""):
            raise RuntimeError("nats down")

    pm = _pm()
    pm.save_escalation_matrix({"version": "v", "rules": {"LOW+LOW": "LOW"}})

    async def llm_tune(prompt, system, client, cfg):
        return (
            '{"reply": "ok", "intent": {"kind": "tune_rule", "target": "LOW+LOW",'
            ' "action": {"target": "MEDIUM"}}, "needs_clarification": false,'
            ' "question": null}'
        )

    asyncio.run(
        E.handle_turn(
            "s1",
            "treat low+low as medium",
            pm=pm,
            nc=_NC(),
            http_client=None,
            cfg=_Cfg(),
            query_fn=llm_tune,
        )
    )
    turn = asyncio.run(
        E.handle_turn(
            "s1",
            "change the matrix",
            pm=pm,
            nc=_FailNC(),
            http_client=None,
            cfg=_Cfg(),
            query_fn=llm_tune,
        )
    )
    assert turn.error is None
    assert turn.applied is not None and turn.applied["status"] == "applied"
    assert pm.load_escalation_matrix()["rules"]["LOW+LOW"] == "MEDIUM"


def test_undo_of_failed_confirm_reports_nothing_to_undo():
    """F1 regression: a confirmed apply that ended "logged" (nothing written)
    must NOT be offered for undo -- undoing it would reply a false "Reversed."
    for a change that never happened (invariant 7)."""
    pm, nc = _pm(), _NC()
    pm.append_dialogue_audit(
        {
            "ts": 1.0,
            "session_id": "s1",
            "kind": "gate_calibration",
            "target": "single:typing:user",
            "proposal": {
                "kind": "gate_calibration",
                "target": "single:typing:user",
                "action": {
                    "op": "self_tolerance_add",
                    "state_key": "single:typing:user",
                    "prior": False,
                },
            },
            "status": "logged",
        }
    )

    async def llm_undo(prompt, system, client, cfg):
        return (
            '{"reply": "sure", "intent": {"kind": "undo"},'
            ' "needs_clarification": false, "question": null}'
        )

    turn = asyncio.run(
        E.handle_turn(
            "s1",
            "undo that",
            pm=pm,
            nc=nc,
            http_client=None,
            cfg=_Cfg(),
            query_fn=llm_undo,
        )
    )
    assert turn.reply == "There's nothing recent to undo."
    assert turn.pending is None


def test_correct_silence_reverses_taught_directive_and_gate_stops_suppressing():
    """A silence caused by a taught directive (limen/gate.py's Stage-0.5
    pre-check) is reversible via correct_silence: teach -> confirm removes
    the directive, and a subsequent gate evaluation on the same channel no
    longer suppresses via that directive."""
    pm, nc = _pm(), _NC()
    pm.load_focused_app = lambda **_k: "appX"
    pm.add_dialogue_directive(
        {
            "directive_id": "d1",
            "predicate": {"context": "focused_app", "match": "appX"},
            "action": "suppress",
            "scope": "all",
        }
    )
    sig = G.build_signature(SINGLE_MEDIUM_TYPING)
    gate = G.Gate()
    before = gate.evaluate(sig, pm, _GATE_CFG, now=500.0)
    assert before.action == "suppress"
    assert before.deciding_arm == "taught_directive"
    assert gate.record_suppression(before, sig, pm, 500.0) is True

    async def correct(prompt, system, client, cfg):
        return (
            '{"reply":"Noted.", "needs_clarification":false,"question":null,'
            '"intent":{"kind":"correct_silence","target":"single:typing:user",'
            '"action":{},"rationale":"you should have spoken up"}}'
        )

    t1 = asyncio.run(
        E.handle_turn(
            "s12",
            "you should've spoken up about typing",
            pm=pm,
            nc=nc,
            http_client=None,
            cfg=_GATE_CFG,
            query_fn=correct,
        )
    )
    assert t1.pending is not None
    assert t1.pending["proposal"]["kind"] == "context_directive"
    assert t1.pending["proposal"]["action"] == {"op": "remove", "directive_id": "d1"}

    t2 = asyncio.run(
        E.handle_turn(
            "s12",
            "yes",
            pm=pm,
            nc=nc,
            http_client=None,
            cfg=_GATE_CFG,
            query_fn=correct,
        )
    )
    assert t2.applied is not None and t2.applied["status"] == "applied"
    assert pm.get_dialogue_directive("d1") is None  # the taught rule is gone

    after = G.Gate().evaluate(sig, pm, _GATE_CFG, now=501.0)
    assert after.action != "suppress" or after.deciding_arm != "taught_directive"


def test_heavy_requires_phrase():
    pm, nc = _pm(), _NC()

    async def llm_heavy(prompt, system, client, cfg):
        return (
            '{"reply":"set rule","needs_clarification":false,"question":null,'
            '"intent":{"kind":"tune_rule","target":"LOW+LOW",'
            '"action":{"target":"MEDIUM"},"rationale":"medium"}}'
        )

    pm.save_escalation_matrix({"version": "v1", "rules": {"LOW+LOW": "LOW"}})
    asyncio.run(
        E.handle_turn(
            "s2",
            "treat low+low as medium",
            pm=pm,
            nc=nc,
            http_client=None,
            cfg=_Cfg(),
            query_fn=llm_heavy,
        )
    )
    bad = asyncio.run(
        E.handle_turn(
            "s2", "yes", pm=pm, nc=nc, http_client=None, cfg=_Cfg(), query_fn=llm_heavy
        )
    )
    assert bad.applied is None  # plain yes not enough
    good = asyncio.run(
        E.handle_turn(
            "s2",
            "yes, change the matrix",
            pm=pm,
            nc=nc,
            http_client=None,
            cfg=_Cfg(),
            query_fn=llm_heavy,
        )
    )
    assert good.applied and pm.load_escalation_matrix()["rules"]["LOW+LOW"] == "MEDIUM"


def test_bare_yes_with_no_pending_is_truthful():
    """Constraint: expired/absent pending on a 'yes' -> truthful 'nothing
    pending' reply, without wasting (or being misled by) an LLM call."""
    pm, nc = _pm(), _NC()  # fresh session, nothing ever proposed
    spy = _CallSpy()
    out = asyncio.run(
        E.handle_turn(
            "s9", "yes", pm=pm, nc=nc, http_client=None, cfg=_Cfg(), query_fn=spy
        )
    )
    assert out.applied is None and out.pending is None
    assert spy.calls == 0  # short-circuited before any LLM call
    assert "nothing" in out.reply.lower()


def test_undo_with_no_audit_history_is_truthful():
    """No audit trail at all -> _handle_undo (the request phase) reports
    immediately: no pending is created, nothing is audited or published."""
    pm, nc = _pm(), _NC()
    out = asyncio.run(E._handle_undo("s1", "undo that", "ok", pm=pm, cfg=_Cfg()))
    assert out.applied is None
    assert out.pending is None
    assert "nothing" in out.reply.lower()
    assert nc.published == []  # no event for a no-op


def test_undo_round_trip_restores_state_and_audits():
    """Full path: teach -> confirm -> undo request -> confirm undo, via
    handle_turn's intent kind=undo (spec §9: undo is a light-tier pending
    like every other light intent, never applied on the same turn it's
    requested), exercising router.apply_undo (not a hand-built
    inverse+apply_proposal)."""
    pm, nc = _pm(), _NC()

    async def llm_intent(prompt, system, client, cfg):
        return (
            '{"reply": "I will speak up.", "needs_clarification": false,'
            ' "question": null,'
            ' "intent": {"kind": "correct_silence", "target": "single:typing:user",'
            ' "action": {}, "rationale": "speak up"}}'
        )

    asyncio.run(
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
    asyncio.run(
        E.handle_turn(
            "s1", "yes", pm=pm, nc=nc, http_client=None, cfg=_Cfg(), query_fn=llm_intent
        )
    )
    assert pm.is_self_tolerant("single:typing:user") is False

    async def llm_undo(prompt, system, client, cfg):
        return (
            '{"reply": "Okay.", "needs_clarification": false, "question": null,'
            ' "intent": {"kind": "undo", "target": null, "action": {},'
            ' "rationale": "undo"}}'
        )

    t3 = asyncio.run(
        E.handle_turn(
            "s1",
            "undo that",
            pm=pm,
            nc=nc,
            http_client=None,
            cfg=_Cfg(),
            query_fn=llm_undo,
        )
    )
    assert t3.pending is not None and t3.applied is None  # awaiting confirm
    assert pm.is_self_tolerant("single:typing:user") is False  # not reversed yet
    assert len(pm.load_dialogue_audit(limit=5)) == 1  # no new audit entry yet

    t4 = asyncio.run(
        E.handle_turn(
            "s1", "yes", pm=pm, nc=nc, http_client=None, cfg=_Cfg(), query_fn=llm_undo
        )
    )
    assert t4.applied is not None and t4.applied["status"] == "applied"
    assert t4.applied["undo"] is True
    assert t4.applied["confirming_text"] == "yes"  # the turn that confirmed it
    assert pm.is_self_tolerant("single:typing:user") is True  # restored
    audit = pm.load_dialogue_audit(limit=5)
    assert len(audit) == 2 and audit[0]["undo"] is True
    assert nc.published.count("augur.imperator.dialogue.applied") == 2
    assert nc.published.count("augur.imperator.ii.trigger") == 2


def test_undo_unavailable_when_no_inverse_exists():
    pm, nc = _pm(), _NC()
    pm.append_dialogue_audit(
        {
            "ts": 1.0,
            "session_id": "s4",
            "kind": "gate_calibration",
            "target": "x",
            "proposal": {
                "kind": "gate_calibration",
                "target": "x",
                "action": {"op": "self_tolerance_add", "state_key": "x", "prior": True},
            },
            "status": "applied",
        }
    )
    requested = asyncio.run(E._handle_undo("s4", "undo that", "ok", pm=pm, cfg=_Cfg()))
    assert requested.pending is not None and requested.applied is None
    out = asyncio.run(
        E._finish_undo(
            requested.pending["prior"], "s4", "yes", pm=pm, nc=nc, cfg=_Cfg()
        )
    )
    assert out.applied is not None and out.applied["status"] == "unavailable"
    assert "can't be undone" in out.reply.lower() or "cannot" in out.reply.lower()


def test_confirmed_apply_disabled_persists_truthful_applied_flag():
    """Fix round: the dialogue-log turn record must not claim applied=True
    when the confirmed apply actually resolved 'logged' (e.g. the
    dialogue_confirmed_apply_enabled kill switch is off). The audit log
    keeps the true status either way."""

    class _CfgDisabled(_Cfg):
        dialogue_confirmed_apply_enabled = False

    pm, nc = _pm(), _NC()

    async def llm_intent(prompt, system, client, cfg):
        return (
            '{"reply": "I will speak up.", "needs_clarification": false,'
            ' "question": null,'
            ' "intent": {"kind": "correct_silence", "target": "single:typing:user",'
            ' "action": {}, "rationale": "speak up"}}'
        )

    asyncio.run(
        E.handle_turn(
            "s8",
            "you should've spoken",
            pm=pm,
            nc=nc,
            http_client=None,
            cfg=_CfgDisabled(),
            query_fn=llm_intent,
        )
    )
    t2 = asyncio.run(
        E.handle_turn(
            "s8",
            "yes",
            pm=pm,
            nc=nc,
            http_client=None,
            cfg=_CfgDisabled(),
            query_fn=llm_intent,
        )
    )
    assert t2.applied is not None and t2.applied["status"] == "logged"
    assert "couldn't apply" in t2.reply.lower()
    assert pm.load_dialogue_audit(limit=1)[0]["status"] == "logged"
    log = pm.load_dialogue_log(limit=1, session_id="s8")
    assert log and log[0]["applied"] is False  # truthful, matches the reply


def test_invalid_intent_needs_clarification_with_reason():
    """Pin the except-ValueError path around validate_intent/route: a
    target-less correct_silence intent must produce a needs-clarification
    reply carrying the validation reason, persist the turn, and mutate
    nothing."""
    pm, nc = _pm(), _NC()

    async def llm_bad_intent(prompt, system, client, cfg):
        return (
            '{"reply": "ok", "needs_clarification": false, "question": null,'
            ' "intent": {"kind": "correct_silence", "target": null,'
            ' "action": {}, "rationale": "x"}}'
        )

    out = asyncio.run(
        E.handle_turn(
            "s10",
            "fix it",
            pm=pm,
            nc=nc,
            http_client=None,
            cfg=_Cfg(),
            query_fn=llm_bad_intent,
        )
    )
    assert out.needs_clarification is True
    assert out.applied is None and out.pending is None
    assert "requires a target" in out.reply
    assert pm.load_dialogue_pending("s10") is None  # nothing routed
    assert pm.load_dialogue_audit(limit=5) == []  # nothing applied
    log = pm.load_dialogue_log(limit=1, session_id="s10")
    assert log and log[0]["reply"] == out.reply  # turn persisted


def test_non_affirmative_turn_drops_pending_with_notice():
    """Fix round: a fresh (non-affirmative, non-heavy-phrase) turn while a
    pending exists must still drop the pending, but say so instead of
    silently discarding the user's un-confirmed proposal."""
    pm, nc = _pm(), _NC()

    async def llm_intent(prompt, system, client, cfg):
        return (
            '{"reply": "I will speak up.", "needs_clarification": false,'
            ' "question": null,'
            ' "intent": {"kind": "correct_silence", "target": "single:typing:user",'
            ' "action": {}, "rationale": "speak up"}}'
        )

    asyncio.run(
        E.handle_turn(
            "s11",
            "you should've spoken",
            pm=pm,
            nc=nc,
            http_client=None,
            cfg=_Cfg(),
            query_fn=llm_intent,
        )
    )
    assert pm.load_dialogue_pending("s11") is not None

    async def llm_plain(prompt, system, client, cfg):
        return (
            '{"reply": "All quiet.", "needs_clarification": false,'
            ' "question": null, "intent": null}'
        )

    out = asyncio.run(
        E.handle_turn(
            "s11",
            "what do you see right now?",
            pm=pm,
            nc=nc,
            http_client=None,
            cfg=_Cfg(),
            query_fn=llm_plain,
        )
    )
    assert pm.load_dialogue_pending("s11") is None  # dropped
    assert "dropped the pending" in out.reply.lower()  # ...and said so
    assert "All quiet." in out.reply  # fresh turn still answered
    assert pm.load_dialogue_audit(limit=5) == []  # nothing applied
    assert out.applied is None


def test_undo_after_unavailable_undo_does_not_crash():
    """A prior undo attempt that itself resolved 'unavailable' audits a
    record with proposal=None. router.build_inverse assumes a real proposal
    dict (`p.get("action")`), so a chained "undo that" REQUEST on top of that
    record would AttributeError without the precondition guard in
    _handle_undo -- the second request must instead hit the immediate
    nothing-to-undo reply (no new pending)."""
    pm, nc = _pm(), _NC()
    pm.append_dialogue_audit(
        {
            "ts": 1.0,
            "session_id": "s7",
            "kind": "gate_calibration",
            "target": "x",
            "proposal": {
                "kind": "gate_calibration",
                "target": "x",
                "action": {"op": "self_tolerance_add", "state_key": "x", "prior": True},
            },
            "status": "applied",
        }
    )
    requested = asyncio.run(E._handle_undo("s7", "undo that", "ok", pm=pm, cfg=_Cfg()))
    assert requested.pending is not None
    first = asyncio.run(
        E._finish_undo(
            requested.pending["prior"], "s7", "yes", pm=pm, nc=nc, cfg=_Cfg()
        )
    )
    assert first.applied["status"] == "unavailable"

    second = asyncio.run(E._handle_undo("s7", "undo that", "ok", pm=pm, cfg=_Cfg()))
    assert second.applied is None
    assert second.pending is None
    assert "nothing" in second.reply.lower()


def test_undo_blocked_when_prior_value_outside_current_bounds():
    """apply_undo's bounds pre-check: a rollback anchor recorded before the
    floor bounds tightened must report 'blocked', not silently fail or
    (worse) claim success."""
    pm, nc = _pm(), _NC()
    pm.append_dialogue_audit(
        {
            "ts": 1.0,
            "session_id": "s5",
            "kind": "gate_calibration",
            "target": "single:typing:user",
            "proposal": {
                "kind": "gate_calibration",
                "target": "single:typing:user",
                "action": {
                    "op": "floor_set",
                    "state_key": "single:typing:user",
                    "value": 0.3,
                    "prior": {"floor": 0.9},  # outside current [0.0, 0.6]
                },
            },
            "status": "applied",
        }
    )
    requested = asyncio.run(E._handle_undo("s5", "undo that", "ok", pm=pm, cfg=_Cfg()))
    assert requested.pending is not None
    out = asyncio.run(
        E._finish_undo(
            requested.pending["prior"], "s5", "yes", pm=pm, nc=nc, cfg=_Cfg()
        )
    )
    assert out.applied is not None and out.applied["status"] == "blocked"
    # the reason surfaces once, cleanly -- no "can't undo -- cannot restore"
    # double negation.
    assert "I can't restore that: prior value outside current bounds." in out.reply
    assert "cannot restore" not in out.reply


def test_undo_logged_when_confirmed_apply_disabled():
    class _CfgDisabled(_Cfg):
        dialogue_confirmed_apply_enabled = False

    pm, nc = _pm(), _NC()
    pm.append_dialogue_audit(
        {
            "ts": 1.0,
            "session_id": "s6",
            "kind": "gate_calibration",
            "target": "single:typing:user",
            "proposal": {
                "kind": "gate_calibration",
                "target": "single:typing:user",
                "action": {
                    "op": "self_tolerance_remove",
                    "state_key": "single:typing:user",
                    "prior": True,
                },
            },
            "status": "applied",
        }
    )
    requested = asyncio.run(
        E._handle_undo("s6", "undo that", "ok", pm=pm, cfg=_CfgDisabled())
    )
    assert requested.pending is not None
    out = asyncio.run(
        E._finish_undo(
            requested.pending["prior"], "s6", "yes", pm=pm, nc=nc, cfg=_CfgDisabled()
        )
    )
    assert out.applied is not None and out.applied["status"] == "logged"
    assert "didn't take" in out.reply.lower()
