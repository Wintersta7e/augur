import fakeredis

from imperator import proposals as P
from imperator.dialogue import context as C
from imperator.dialogue import router as R
from tabula.persistence import PersistenceManager


class _PM:
    def load_auspices(self):
        return {"salience": {"value": 0.7}}

    def load_self_model(self):
        return {
            "precision": {"value": 0.5},
            "blind_spots": {
                "value": [{"kind": "low_precision_domain"}],
                "fresh": True,
                "as_of": 0.0,
            },
        }

    def load_silence_records(self, limit=50):
        return [
            {
                "state_key": "single:typing:user",
                "arm": "habituation",
                "reason": "habituated",
            }
        ]

    def load_emissions(self, limit=50):
        return []

    def load_dialogue_log(self, limit=12):
        return []


class _Cfg:
    dialogue_context_max_turns = 12
    dialogue_context_token_budget = 2048


def test_assemble_pulls_salience_and_suppression_reason():
    ctx = C.assemble(_PM(), now=100.0, cfg=_Cfg())
    assert ctx.salience == 0.7
    assert ctx.recent_suppressions[0]["arm"] == "habituation"


def test_render_is_bounded_and_mentions_reason():
    ctx = C.assemble(_PM(), now=100.0, cfg=_Cfg())
    block = C.render(ctx, _Cfg())
    assert "habituation" in block
    assert "low_precision_domain" in block
    assert len(block) <= _Cfg().dialogue_context_token_budget * 8  # ~chars budget guard


class _ApplyCfg(_Cfg):
    # Knobs for the real confirmed-apply path (matches the other dialogue
    # test configs) + FSRS review knobs read on a semantic_fact re-teach.
    dialogue_confirmed_apply_enabled = True
    imperator_ii_dedupe_staleness_s = 86400.0
    memory_s_growth_factor = 0.5
    memory_s_max = 365


def test_removed_taught_fact_absent_from_context_until_undo():
    """End-to-end against the REAL persistence + apply + router surfaces:
    a user-confirmed "forget X" (semantic_fact remove) must drop X from the
    assembled dialogue context and its rendered LLM block on the very next
    turn -- and the undo of that remove (the decision-A re-add restore path)
    must bring it back."""
    pm = PersistenceManager(fakeredis.FakeStrictRedis(decode_responses=False))
    pattern = {
        "kind": "semantic",
        "domains": ["typing"],
        "rule_key": None,
        "severity": "LOW",
    }
    teach = P.make_proposal(
        kind="semantic_fact",
        target="typing",
        action={"pattern": pattern},
        rationale="left-handed",
        source="dialogue",
    )
    applied = R.apply_confirmed(
        {"proposal": teach, "echo": "e"}, pm=pm, cfg=_ApplyCfg(), session_id="d1"
    )
    assert applied["status"] == "applied"
    mid = applied["proposal"]["action"]["memory_id"]

    ctx = C.assemble(pm, now=100.0, cfg=_Cfg())
    assert [f["memory_id"] for f in ctx.taught_facts] == [mid]
    assert mid in C.render(ctx, _Cfg())

    # Confirmed remove ("forget X"): the fact must vanish from the context.
    removed = R.apply_undo(
        {"proposal": applied["proposal"]}, pm=pm, cfg=_ApplyCfg(), session_id="d1"
    )
    assert removed["status"] == "applied"
    ctx = C.assemble(pm, now=100.0, cfg=_Cfg())
    assert ctx.taught_facts == []
    assert mid not in C.render(ctx, _Cfg())

    # Undo of the remove (re-add restore): visible again.
    readded = R.apply_undo(
        {"proposal": removed["proposal"]}, pm=pm, cfg=_ApplyCfg(), session_id="d2"
    )
    assert readded["status"] == "applied"
    ctx = C.assemble(pm, now=100.0, cfg=_Cfg())
    assert [f["memory_id"] for f in ctx.taught_facts] == [mid]
    assert mid in C.render(ctx, _Cfg())
