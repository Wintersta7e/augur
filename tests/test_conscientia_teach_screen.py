"""S3a — teach-time value screen refuses charter-violating teachings.

Adapted from the task brief's guessed shape: ``router.route()`` has no
``session_id`` parameter and is a pure translation layer -- ``pm``/``cfg``
are accepted only for interface symmetry with apply_confirmed/apply_undo and
it performs no Redis writes (pinned by
tests/test_dialogue_invariants.py::test_route_is_pure_no_state_writes, plus
route()'s own docstring: "this deterministic mapping needs neither"). The
screen + best-effort violation write + refusal therefore live in
imperator/dialogue/engine.py::_handle_intent instead -- the earliest point
where the validated intent fields, ``pm``, and ``session_id`` are all in
scope together without writing state from inside route(). Tests drive the
public ``E.handle_turn()`` entry point, mirroring
tests/test_dialogue_engine_write.py's conventions (stub ``query_fn``
returning raw LLM-shaped JSON, ``_NC`` NATS stub, fakeredis-backed
PersistenceManager) rather than calling ``R.route()`` directly.
"""

import asyncio
import json

import fakeredis

from imperator.dialogue import engine as E
from tabula.persistence import PersistenceManager


def _pm():
    return PersistenceManager(fakeredis.FakeStrictRedis(decode_responses=False))


class _NC:
    def __init__(self):
        self.published = []

    async def publish(self, subj, data=b""):
        self.published.append(subj)


class _Cfg:
    # Baseline engine-level fixture (mirrors tests/test_dialogue_engine_write.py's
    # _Cfg, proven to satisfy context.assemble/persona's full handle_turn() path).
    dialogue_num_predict = 512
    dialogue_context_max_turns = 12
    dialogue_context_token_budget = 2048
    dialogue_pending_ttl_s = 300.0
    dialogue_confirmed_apply_enabled = True
    min_prompt_len = 20
    prompt_forbidden_patterns = ("take a break",)
    imperator_ii_dedupe_staleness_s = 86400.0
    sigma_min = 1.5
    sigma_max = 5.0
    # Task 7 conscientia fields the teach screen reads (conscientia.charter.
    # teach_patterns + screens.screen_taught_content's self-gate).
    conscientia_enabled = True
    conscientia_teach_screen_enabled = True
    conscientia_teach_extra_patterns = ()


def _cfg(**overrides):
    cfg = _Cfg()
    for k, v in overrides.items():
        setattr(cfg, k, v)
    return cfg


def _fact_query_fn(rationale, rule_key="rk"):
    """Stub query_fn returning a teach_semantic_fact intent with the given
    rationale, in the raw LLM JSON shape handle_turn's _parse expects."""
    intent = {
        "kind": "teach_semantic_fact",
        "target": "typing",
        "action": {"domains": ["typing"], "rule_key": rule_key, "severity": "LOW"},
        "rationale": rationale,
    }
    payload = json.dumps(
        {
            "reply": "noted",
            "needs_clarification": False,
            "question": None,
            "intent": intent,
        }
    )

    async def query_fn(prompt, system, client, cfg):
        return payload

    return query_fn


def _directive_query_fn(rationale):
    """Stub query_fn returning a teach_context_directive intent. No live
    focused_app is set up in these tests -- a violating rationale must be
    refused by the screen before route()'s own focused-app check ever runs,
    so none is needed."""
    intent = {
        "kind": "teach_context_directive",
        "target": "test_directive",
        "action": {"action": "suppress", "scope": "all"},
        "rationale": rationale,
    }
    payload = json.dumps(
        {
            "reply": "noted",
            "needs_clarification": False,
            "question": None,
            "intent": intent,
        }
    )

    async def query_fn(prompt, system, client, cfg):
        return payload

    return query_fn


def _turn(pm, nc, cfg, query_fn, session_id="d1", user_text="teach me something"):
    return asyncio.run(
        E.handle_turn(
            session_id,
            user_text,
            pm=pm,
            nc=nc,
            http_client=None,
            cfg=cfg,
            query_fn=query_fn,
        )
    )


def test_violating_fact_rationale_refused():
    pm, nc = _pm(), _NC()
    out = _turn(pm, nc, _cfg(), _fact_query_fn("remind me to take a break every hour"))
    assert out.needs_clarification is True
    assert "won't store" in out.reply
    assert out.pending is None
    viols = pm.load_conscientia_violations(limit=5)
    assert viols and viols[0]["surface"] == "teach"
    assert pm.load_dialogue_pending("d1") is None  # nothing routed
    assert pm.load_taught_facts() == []  # nothing stored


def test_clean_fact_routes():
    pm, nc = _pm(), _NC()
    out = _turn(pm, nc, _cfg(), _fact_query_fn("deep work happens in the mornings"))
    assert out.needs_clarification is False
    assert out.pending is not None  # a pending proposal comes back
    assert out.pending["proposal"]["kind"] == "semantic_fact"
    assert pm.load_conscientia_violations(limit=5) == []


def test_screen_disabled_stores_like_before():
    pm, nc = _pm(), _NC()
    out = _turn(
        pm,
        nc,
        _cfg(conscientia_enabled=False),
        _fact_query_fn("take a break hourly"),
    )
    assert out.pending is not None
    assert pm.load_conscientia_violations(limit=5) == []


def test_violating_directive_rationale_also_refused():
    """Interfaces section names both branches ("the teach_semantic_fact /
    teach_context_directive branches") -- pin that teach_context_directive
    is screened too, not just the fact path, since both share the single
    insertion point in _handle_intent."""
    pm, nc = _pm(), _NC()
    out = _turn(pm, nc, _cfg(), _directive_query_fn("just take a break and stay quiet"))
    assert out.needs_clarification is True
    assert "won't store" in out.reply
    viols = pm.load_conscientia_violations(limit=5)
    assert viols and viols[0]["surface"] == "teach"
    assert pm.load_dialogue_pending("d1") is None
