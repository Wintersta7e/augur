import asyncio
from imperator import reasoner

SELF_MODEL = {
    "schema_version": 1,
    "generated_at": 100.0,
    "session_id": "s1",
    "competence": {"value": 0.4, "fresh": True},
    "blind_spots": {
        "value": [
            {"kind": "low_confidence_rule", "detail": "x", "evidence": "LOW+LOW"}
        ],
        "fresh": True,
    },
    "recent_self_tuning": {"value": {}, "fresh": True},
}
CANNED = '[{"kind":"escalation_rule","target":"LOW+LOW","action":{"target":"MEDIUM"},"rationale":"helps you","rank":1}]'


class _Cfg:
    ollama_url = "x"
    ollama_model = "m"
    ollama_timeout = 1.0
    imperator_ii_num_predict = 64
    imperator_ii_max_proposals_per_cycle = 5


def test_build_prompt_includes_signals():
    p = reasoner.build_reasoning_prompt(SELF_MODEL)
    assert "low_confidence_rule" in p and "LOW+LOW" in p and "0.4" in p


def test_generate_parses_stub():
    async def stub(prompt, client, config):
        return CANNED, 1.0

    out = asyncio.run(
        reasoner.generate_proposals(
            SELF_MODEL, client=None, config=_Cfg(), now=100.0, query_ollama_fn=stub
        )
    )
    assert len(out) == 1 and out[0]["target"] == "LOW+LOW"


def test_parser_tolerates_bad_rank_and_action():
    bad = (
        '[{"kind":"escalation_rule","target":"A","action":"notdict","rank":"x"},'
        '{"kind":"escalation_rule","target":"B","action":{"target":"LOW"},"rank":2}]'
    )
    out = reasoner.parse_proposals(bad, now=0.0, max_n=5)
    assert [p["target"] for p in out] == ["B"]


def test_parser_accepts_gated_kinds():
    g = '[{"kind":"code","target":"imperator/apply.py","action":{"note":"refactor"},"rationale":"r","rank":3}]'
    out = reasoner.parse_proposals(g, now=0.0, max_n=5)
    assert len(out) == 1 and out[0]["kind"] == "code"


def test_generate_garbage_empty():
    async def stub(prompt, client, config):
        return "no json", 1.0

    assert (
        asyncio.run(
            reasoner.generate_proposals(
                SELF_MODEL, client=None, config=_Cfg(), now=0.0, query_ollama_fn=stub
            )
        )
        == []
    )


def test_generate_failure_raises_classified():
    import pytest

    async def stub(prompt, client, config):
        raise TimeoutError("slow")

    with pytest.raises(reasoner.ReasonerError) as e:
        asyncio.run(
            reasoner.generate_proposals(
                SELF_MODEL, client=None, config=_Cfg(), now=0.0, query_ollama_fn=stub
            )
        )
    assert e.value.reason in ("ollama_timeout", "ollama_error")
