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


def test_parser_extracts_array_with_trailing_prose():
    # A valid array followed by prose containing a bracket must still parse;
    # a greedy regex would over-capture to the trailing ']' and fail.
    txt = (
        '[{"kind":"escalation_rule","target":"LOW+LOW","action":{"target":"MEDIUM"},"rank":1}]'
        "\n\nNote: see item [1] above for the rationale."
    )
    out = reasoner.parse_proposals(txt, now=0.0, max_n=5)
    assert [p["target"] for p in out] == ["LOW+LOW"]


def test_parser_extracts_object_wrapped_array():
    # Models often wrap the array in {"proposals":[...]} (or "items"/"results");
    # the parser must dig out the array-valued field, not silently drop everything.
    for key in ("proposals", "items", "results"):
        txt = (
            '{"%s":[{"kind":"escalation_rule","target":"LOW+LOW",'
            '"action":{"target":"MEDIUM"},"rank":1}]}' % key
        )
        out = reasoner.parse_proposals(txt, now=0.0, max_n=5)
        assert [p["target"] for p in out] == ["LOW+LOW"], key


def test_parser_object_wrapped_with_trailing_prose():
    # raw_decode tolerance must survive for the object-wrapped form too.
    txt = (
        '{"proposals":[{"kind":"escalation_rule","target":"A+B",'
        '"action":{"target":"HIGH"},"rank":1}]}'
        "\n\nThat is my recommendation."
    )
    out = reasoner.parse_proposals(txt, now=0.0, max_n=5)
    assert [p["target"] for p in out] == ["A+B"]


def test_parser_object_without_array_field_is_empty():
    # An object whose values are all scalars (no array field) yields nothing.
    txt = '{"note":"no proposals this cycle","count":0}'
    assert reasoner.parse_proposals(txt, now=0.0, max_n=5) == []


def test_parser_truncated_array_is_empty():
    # A truncated/malformed array (model cut off mid-token) is intentionally [].
    txt = '[{"kind":"escalation_rule","target":"A","action":{"target":"LOW"'
    assert reasoner.parse_proposals(txt, now=0.0, max_n=5) == []


def test_parser_caps_valid_survivors_not_raw_candidates():
    # Two malformed items precede two valid ones; max_n=2 must yield the two
    # VALID proposals, not stop after consuming two raw (invalid) candidates.
    items = (
        '[{"bad":1},{"also":"bad"},'
        '{"kind":"escalation_rule","target":"A+B","action":{"target":"LOW"},"rank":1},'
        '{"kind":"escalation_rule","target":"C+D","action":{"target":"HIGH"},"rank":2}]'
    )
    out = reasoner.parse_proposals(items, now=0.0, max_n=2)
    assert [p["target"] for p in out] == ["A+B", "C+D"]


def test_parser_drops_prompt_strategy_with_non_string_text():
    # Non-string action.text would crash prompt-safety string ops downstream.
    bad = '[{"kind":"prompt_strategy","target":"typing","action":{"text":{"oops":1}},"rank":1}]'
    assert reasoner.parse_proposals(bad, now=0.0, max_n=5) == []


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
