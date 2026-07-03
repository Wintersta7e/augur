"""Capstone showcase scenarios for Imperator III — Dialogue (spec §14, Task 22).

Each scenario tells a story and asserts a *behavior change* against real
Redis + NATS (the ``real_pm``/``real_nc``/``dialogue_cfg`` fixtures in
``tests/integration/conftest.py``) — not just that a reply was produced, but
that a downstream faculty (Limen, Consilium, Memoria) actually behaves
differently afterward. The dialogue LLM call is stubbed via the engine's
``query_fn`` seam for determinism (per spec §14: "the dialogue *behavior* is
what's asserted, not Ollama prose"); scenario 4 additionally crosses into
Consilium's advice path, where the LLM is stubbed via the advisor's own
``query_ollama`` seam (mirroring ``tests/test_dialogue_advice_injection.py``).

Conventions copied from existing suites:
  * ``async def test_...`` + ``await`` (not ``asyncio.run``) — every other
    file in this package drives async code this way
    (``tests/integration/test_gate_integration.py``,
    ``tests/integration/test_imperator_integration.py``); nesting a second
    ``asyncio.run`` event loop around a fixture-bound live NATS connection
    risks cross-loop errors, so this file departs from the task brief's
    literal ``asyncio.run`` sketch for that reason.
  * Gate mechanics (arm order, ``central_tolerance``/reservoir priming,
    ``build_signature`` state keys) — ``tests/integration/test_gate_integration.py``.
  * Taught-directive / focused-app seeding — ``tests/test_dialogue_gate_directive.py``.
  * Correction/undo/heavy-confirm turn shapes — ``tests/test_dialogue_engine_write.py``.
  * Semantic-fact teach + real Memoria sweep — ``tests/test_dialogue_memoria_teach.py``.
  * Advice-prompt capture harness (``_run``/``_scheduler``) —
    ``tests/test_dialogue_advice_injection.py`` / ``tests/test_advisor_gate_flow.py``.
  * GATED klass invariant — ``tests/test_dialogue_apply_confirmed.py`` /
    ``tests/test_dialogue_invariants.py``.

Varied baseline values are not needed here (no anomaly-detector baselines are
exercised). Gate ``state_key``s are mostly unique per scenario; scenarios 3
and 7 both reuse ``single:chess:user`` (each independently driving that
channel to central_tolerance suppression on the same illustrative "chess"
domain), but no state actually bleeds across them -- each test gets a
freshly flushed Redis via the ``redis_client`` fixture, so the shared key
never carries state between scenarios.
"""

from __future__ import annotations

from dataclasses import replace
from unittest.mock import AsyncMock, MagicMock

import pytest

from imperator import proposals as P
from imperator.dialogue import engine as E
from imperator.dialogue import router as R
from limen import gate as G
from memoria.tiers import plan_sweep
from tests.conftest import CORRELATION_MEDIUM, SINGLE_MEDIUM, SINGLE_MEDIUM_TYPING
from tests.test_advisor_gate_flow import _run, _scheduler

pytestmark = [pytest.mark.integration, pytest.mark.asyncio]


# ── shared helpers ───────────────────────────────────────────────────────────


def _sig(payload: dict) -> G.Signature:
    return G.build_signature(payload)


def _focus_app(pm, app: str, ts: str | None = None) -> None:
    """Seed the real activity-focus stream ``load_focused_app`` reads (spec
    §7.2) — mirrors ``tests/test_dialogue_gate_directive.py``'s
    ``test_load_focused_app_falls_back_to_focus_stream``. Avoids the
    nonexistent ``pm.save_focused_app`` the task brief's sketch hedges around
    with ``hasattr``: no such method exists, so this is the real write path.

    ``ts`` defaults to the current UTC time so the focus reads as FRESH under
    load_focused_app's staleness bound (``focused_app_max_age_s``): the teach
    path assembles context with wall-clock ``now``, so a fixed past timestamp
    would be treated as a stale (absent) focus. Pass an explicit ts only when a
    test deliberately needs a specific age."""
    from datetime import datetime, timezone

    from tabula.contracts import PerceptionEvent

    if ts is None:
        ts = datetime.now(timezone.utc).isoformat()
    pm.append_event(
        PerceptionEvent(
            domain="activity_focus",
            stream_id="activity_focus",
            entity=app,
            event_type="focus_change",
            value=0.0,
            unit="none",
            context={"new_app": app},
            timestamp=ts,
            session_id="showcase",
        )
    )


# ── Scenario 1 (spec §14.1): stay silent in deep work, then undo ───────────


async def test_stay_silent_in_appx_then_undo(real_pm, real_nc, dialogue_cfg):
    """Teach a context directive for the focused app -> Limen SUPPRESSES an
    anomaly while that app is focused (previously it FIRED) -> undo restores
    firing. The task brief's representative scenario, adapted to this
    package's async convention and a real focused-app write."""

    async def teach(prompt, system, client, cfg):
        return (
            '{"reply":"Understood.","needs_clarification":false,"question":null,'
            '"intent":{"kind":"teach_context_directive","target":"appX",'
            '"action":{"predicate":{"context":"focused_app","match":"appX"},'
            '"action":"suppress","scope":"all"},"rationale":"deep work"}}'
        )

    # appX is the real focused app AT TEACH TIME (Redis-backed activity
    # history): route() now fills the directive's predicate.match from the LIVE
    # focused app (spec §7.2 / F14), so it must be present before the teach turn
    # -- otherwise the teach is truthfully refused ("I can't tell which app...").
    # Fresh timestamp (default): the teach assembles context with wall-clock now,
    # so the focus must be recent to pass load_focused_app's staleness bound.
    _focus_app(real_pm, "appX")

    t1 = await E.handle_turn(
        "sc1",
        "stay quiet in appX",
        pm=real_pm,
        nc=real_nc,
        http_client=None,
        cfg=dialogue_cfg,
        query_fn=teach,
    )
    assert t1.pending is not None and t1.applied is None

    t2 = await E.handle_turn(
        "sc1",
        "yes",
        pm=real_pm,
        nc=real_nc,
        http_client=None,
        cfg=dialogue_cfg,
        query_fn=teach,
    )
    assert t2.applied is not None and t2.applied["status"] == "applied"

    dec = G.Gate().evaluate(
        _sig(SINGLE_MEDIUM_TYPING), real_pm, dialogue_cfg, now=100.0
    )
    assert dec.action == "suppress" and dec.reason.startswith("taught_directive")

    async def undo(prompt, system, client, cfg):
        return (
            '{"reply":"ok","needs_clarification":false,"question":null,'
            '"intent":{"kind":"undo","target":null,"action":{},"rationale":"undo"}}'
        )

    t3 = await E.handle_turn(
        "sc1",
        "undo that",
        pm=real_pm,
        nc=real_nc,
        http_client=None,
        cfg=dialogue_cfg,
        query_fn=undo,
    )
    # spec §9: undo is a LIGHT-tier intent, confirmed by a plain affirmative
    # like every other light intent -- not applied on the same turn it's
    # requested.
    assert t3.pending is not None and t3.applied is None

    t4 = await E.handle_turn(
        "sc1",
        "yes",
        pm=real_pm,
        nc=real_nc,
        http_client=None,
        cfg=dialogue_cfg,
        query_fn=undo,
    )
    assert t4.applied is not None and t4.applied["status"] == "applied"

    dec2 = G.Gate().evaluate(
        _sig(SINGLE_MEDIUM_TYPING), real_pm, dialogue_cfg, now=101.0
    )
    assert dec2.action != "suppress" or not dec2.reason.startswith("taught_directive")


# ── Scenario 2 (spec §14.2): "you should've spoken up" -> it speaks next time


async def test_correct_silence_reverses_arm_and_fires(real_pm, real_nc, dialogue_cfg):
    """Build a real suppression driven by a known arm (central_tolerance) ->
    teach correct_silence -> the exact arm is reversed (self-tolerance
    removed, not a blanket floor reset) -> an equivalent event now FIRES."""
    state_key = "single:typing:user"
    sig = _sig(SINGLE_MEDIUM_TYPING)
    gate = G.Gate()

    real_pm.add_self_tolerance(state_key)
    before = gate.evaluate(sig, real_pm, dialogue_cfg, now=200.0)
    assert before.action == "suppress"
    assert before.deciding_arm == "central_tolerance"
    assert before.reason == "central_tolerance_learned_self"
    # Authoritative silence write (invariant A) — this is what
    # imperator/dialogue/router.py's _arm_for_silence reads via
    # ctx.recent_suppressions to pick which arm to reverse.
    assert gate.record_suppression(before, sig, real_pm, 200.0) is True

    async def correct(prompt, system, client, cfg):
        return (
            '{"reply":"Noted -- I will speak up next time.",'
            '"needs_clarification":false,"question":null,'
            '"intent":{"kind":"correct_silence","target":"single:typing:user",'
            '"action":{},"rationale":"you should have spoken up"}}'
        )

    await E.handle_turn(
        "sc2",
        "you should've spoken up about typing",
        pm=real_pm,
        nc=real_nc,
        http_client=None,
        cfg=dialogue_cfg,
        query_fn=correct,
    )
    t2 = await E.handle_turn(
        "sc2",
        "yes",
        pm=real_pm,
        nc=real_nc,
        http_client=None,
        cfg=dialogue_cfg,
        query_fn=correct,
    )
    assert t2.applied is not None and t2.applied["status"] == "applied"
    assert t2.applied["proposal"]["action"]["op"] == "self_tolerance_remove"
    assert real_pm.is_self_tolerant(state_key) is False  # the exact arm reversed

    # Commit the reservoir (Arm 5, checked after central_tolerance) so this
    # first-ever eval of the channel isn't itself suppressed by reservoir
    # insufficiency — isolating the assertion to "central_tolerance no longer
    # fires" rather than a fresh-channel artifact of a different arm.
    real_pm.save_reservoir(
        state_key,
        {
            "count": dialogue_cfg.gate_reservoir_on_count,
            "last_ts": 201.0,
            "suppressing": False,
        },
    )
    after = G.Gate().evaluate(sig, real_pm, dialogue_cfg, now=201.0)
    assert after.action != "suppress"
    assert after.reason != "central_tolerance_learned_self"


# ── Scenario 3 (spec §14.3): "stop flagging this" -> it stops ──────────────


async def test_correct_noise_raises_tolerance_and_suppresses(
    real_pm, real_nc, dialogue_cfg
):
    """A firing (chatty) channel -> teach correct_noise -> self-tolerance is
    raised -> the channel now suppresses via central_tolerance."""
    state_key = "single:chess:user"
    sig = _sig(SINGLE_MEDIUM)

    # Commit the reservoir so the "before" eval genuinely fires (a chatty
    # channel), isolating the "after" suppression to the taught correction.
    real_pm.save_reservoir(
        state_key,
        {
            "count": dialogue_cfg.gate_reservoir_on_count,
            "last_ts": 300.0,
            "suppressing": False,
        },
    )
    before = G.Gate().evaluate(sig, real_pm, dialogue_cfg, now=300.0)
    assert before.action != "suppress"

    async def correct(prompt, system, client, cfg):
        return (
            '{"reply":"Understood -- I will stop flagging that.",'
            '"needs_clarification":false,"question":null,'
            '"intent":{"kind":"correct_noise","target":"single:chess:user",'
            '"action":{},"rationale":"too chatty"}}'
        )

    await E.handle_turn(
        "sc3",
        "stop flagging my chess moves",
        pm=real_pm,
        nc=real_nc,
        http_client=None,
        cfg=dialogue_cfg,
        query_fn=correct,
    )
    t2 = await E.handle_turn(
        "sc3",
        "yes",
        pm=real_pm,
        nc=real_nc,
        http_client=None,
        cfg=dialogue_cfg,
        query_fn=correct,
    )
    assert t2.applied is not None and t2.applied["status"] == "applied"
    assert real_pm.is_self_tolerant(state_key) is True  # tolerance raised

    after = G.Gate().evaluate(sig, real_pm, dialogue_cfg, now=301.0)
    assert after.action == "suppress"
    assert after.reason == "central_tolerance_learned_self"


# ── Scenario 4 (spec §14.4): "chess+typing spike = stress" -> it knows it ──


async def test_semantic_fact_persists_and_shapes_advice(real_pm, real_nc, dialogue_cfg):
    """Teach a semantic fact -> (a) the Memoria entry survives a real sweep,
    (b) it's injected into Consilium's advice context next time chess+typing
    are active together."""

    async def teach(prompt, system, client, cfg):
        return (
            '{"reply":"I will remember that.","needs_clarification":false,'
            '"question":null,"intent":{"kind":"teach_semantic_fact",'
            '"target":"chess_typing_stress","action":{"domains":["chess","typing"],'
            '"rule_key":"HIGH+HIGH","severity":"MEDIUM"},'
            '"rationale":"chess and typing spiking together means stress"}}'
        )

    await E.handle_turn(
        "sc4",
        "when chess and typing spike together that means stress",
        pm=real_pm,
        nc=real_nc,
        http_client=None,
        cfg=dialogue_cfg,
        query_fn=teach,
    )
    t2 = await E.handle_turn(
        "sc4",
        "yes",
        pm=real_pm,
        nc=real_nc,
        http_client=None,
        cfg=dialogue_cfg,
        query_fn=teach,
    )
    assert t2.applied is not None and t2.applied["status"] == "applied"
    memory_id = t2.applied["proposal"]["action"]["memory_id"]

    facts = real_pm.load_taught_facts()
    taught = next((f for f in facts if f["memory_id"] == memory_id), None)
    assert taught is not None
    assert sorted(taught["pattern"]["domains"]) == ["chess", "typing"]
    assert taught["pattern"]["rule_key"] == "HIGH+HIGH"

    # (a) Survives a real Memoria sweep (Task 20 decision A: protect=True ->
    # origin_severity=HIGH -> tiers.is_floor_protected -> never pruned).
    plan = plan_sweep(
        real_pm.load_all_memory_states(), [], 10, "sc4-sweep", dialogue_cfg
    )
    assert real_pm.apply_memory_sweep("sc4-sweep", plan)
    assert plan.prunes == []
    assert any(f["memory_id"] == memory_id for f in real_pm.load_taught_facts())

    # (b) Shapes Consilium's advice context next time chess+typing correlate.
    cfg2 = replace(dialogue_cfg, gate_cost_tier_enabled=False)
    gate = G.Gate(arms=[], config=cfg2)  # passes all arms -> fire
    query_ollama = AsyncMock(return_value=("advice text", 12.3))
    await _run(
        payload=CORRELATION_MEDIUM,
        gate=gate,
        scheduler=_scheduler(),
        pm=real_pm,
        nc=real_nc,
        http_client=MagicMock(),
        config=cfg2,
        query_ollama=query_ollama,
        # redis_client intentionally omitted -- _run defaults it to a
        # throwaway fakeredis instance (mirrors the unit harness in
        # tests/test_dialogue_advice_injection.py's _run_and_capture_prompt).
        # process_message only reads it for active-session lookup + the
        # prompt builder's own context, unrelated to the taught-fact
        # injection this assertion checks, which lives entirely on real_pm.
    )
    assert query_ollama.await_count == 1, "advice LLM was not called"
    prompt = query_ollama.await_args.args[0]
    assert "Known facts (taught by the user):" in prompt
    assert "HIGH+HIGH" in prompt


# ── Scenario 5 (spec §14.5): heavy-confirm gate ─────────────────────────────


async def test_heavy_confirm_requires_exact_phrase(real_pm, real_nc, dialogue_cfg):
    """tune_rule does NOT apply on a plain 'yes' -- only on the explicit heavy
    confirm phrase. A plain 'yes' falls through to a fresh turn (per
    engine.handle_turn's tier-mismatch path); since the stub is content-
    agnostic it re-teaches the same intent, so a live heavy pending is always
    present for the real phrase to confirm (matches
    tests/test_dialogue_engine_write.py::test_heavy_requires_phrase)."""
    real_pm.save_escalation_matrix({"version": "v1", "rules": {"LOW+LOW": "LOW"}})

    async def llm_heavy(prompt, system, client, cfg):
        return (
            '{"reply":"I will set that rule.","needs_clarification":false,'
            '"question":null,"intent":{"kind":"tune_rule","target":"LOW+LOW",'
            '"action":{"target":"MEDIUM"},"rationale":"low+low should be medium"}}'
        )

    await E.handle_turn(
        "sc5",
        "treat low+low as medium",
        pm=real_pm,
        nc=real_nc,
        http_client=None,
        cfg=dialogue_cfg,
        query_fn=llm_heavy,
    )
    bad = await E.handle_turn(
        "sc5",
        "yes",
        pm=real_pm,
        nc=real_nc,
        http_client=None,
        cfg=dialogue_cfg,
        query_fn=llm_heavy,
    )
    assert bad.applied is None  # plain "yes" is not enough for a heavy confirm
    assert real_pm.load_escalation_matrix()["rules"]["LOW+LOW"] == "LOW"  # unchanged

    good = await E.handle_turn(
        "sc5",
        "yes, change the matrix",
        pm=real_pm,
        nc=real_nc,
        http_client=None,
        cfg=dialogue_cfg,
        query_fn=llm_heavy,
    )
    assert good.applied is not None and good.applied["status"] == "applied"
    assert real_pm.load_escalation_matrix()["rules"]["LOW+LOW"] == "MEDIUM"


# ── Scenario 6 (spec §14.6): GATED under pressure ───────────────────────────


async def test_code_change_request_never_applies_logged_gated(
    real_pm, real_nc, dialogue_cfg
):
    """Asking (conversationally) for a code/structural change never reaches
    an applicable surface, proven two ways against real Redis:

    1. Engine level -- even an LLM that ignores the persona's fixed intent
       taxonomy and hallucinates an out-of-taxonomy "code" kind fails
       validate_intent closed: no pending is stored, nothing is audited, and
       the persona's reply explains it did not understand the request.
    2. Apply-layer level -- even a hand-built "code" proposal pushed straight
       at the confirmed-apply entry point (bypassing the dialogue taxonomy
       entirely) is refused: normalize_klass forces klass="gated", and the
       confirmed path refuses any gated klass unconditionally -- status
       "logged", the dedupe/anti-thrash marker never armed, no write.
    """

    async def hallucinate_code_change(prompt, system, client, cfg):
        return (
            '{"reply":"I understand you want a code change.",'
            '"needs_clarification":false,"question":null,'
            '"intent":{"kind":"code","target":"advisor.py",'
            '"action":{"patch":"..."},"rationale":"fix the bug directly"}}'
        )

    turn = await E.handle_turn(
        "sc6",
        "just rewrite the code yourself and fix this bug",
        pm=real_pm,
        nc=real_nc,
        http_client=None,
        cfg=dialogue_cfg,
        query_fn=hallucinate_code_change,
    )
    assert turn.needs_clarification is True
    assert turn.applied is None
    assert "unknown intent kind" in turn.reply.lower()
    assert real_pm.load_dialogue_pending("sc6") is None
    assert real_pm.load_dialogue_audit(limit=5) == []

    p = P.normalize_klass(
        P.make_proposal(
            kind="code",
            target="advisor.py",
            action={"patch": "..."},
            rationale="fix the bug directly",
            source="dialogue",
        )
    )
    assert p["klass"] == "gated"
    out = R.apply_confirmed(
        {"proposal": p, "echo": "n/a"}, pm=real_pm, cfg=dialogue_cfg, session_id="sc6"
    )
    assert out["status"] == "logged"
    assert real_pm.is_proposal_applied(p["dedupe_key"]) is False  # never armed


# ── Scenario 7 (spec §14.7): honest introspection ───────────────────────────


async def test_introspection_references_real_suppression_record(
    real_pm, real_nc, dialogue_cfg
):
    """Cause a real, known suppression -> "why did you stay silent?" -> the
    reply the wiring supports references the actual arm/reason from the
    record, proven by capturing the system prompt the (stubbed) LLM sees:
    it must carry the REAL silence record's state_key/arm, assembled by
    imperator/dialogue/context.py from pm.load_silence_records — the same
    record record_suppression just wrote against real Redis."""
    state_key = "single:chess:user"
    sig = _sig(SINGLE_MEDIUM)
    gate = G.Gate()

    real_pm.add_self_tolerance(state_key)
    decision = gate.evaluate(sig, real_pm, dialogue_cfg, now=400.0)
    assert decision.action == "suppress"
    assert gate.record_suppression(decision, sig, real_pm, 400.0) is True

    captured: dict[str, str] = {}

    async def introspect(prompt, system, client, cfg):
        # Deliberately no assert here: an exception raised inside query_fn is
        # swallowed by handle_turn's fail-soft try/except (see
        # tests/test_dialogue_engine_write.py's _CallSpy docstring) and would
        # hide a real wiring bug behind a generic "can't reason" reply. Just
        # capture, and assert on it after handle_turn returns.
        captured["system"] = system
        return (
            '{"reply":"I stayed quiet on chess because central_tolerance '
            'judged it a chronic, already-dismissed channel.",'
            '"needs_clarification":false,"question":null,"intent":null}'
        )

    turn = await E.handle_turn(
        "sc7",
        "why did you stay silent on chess?",
        pm=real_pm,
        nc=real_nc,
        http_client=None,
        cfg=dialogue_cfg,
        query_fn=introspect,
    )
    assert turn.needs_clarification is False
    assert f"{state_key} arm=central_tolerance" in captured["system"]
    assert "central_tolerance" in turn.reply
