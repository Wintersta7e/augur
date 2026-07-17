import asyncio
from imperator.dialogue import engine as E


class _PM:
    def load_auspices(self):
        return {"salience": {"value": 0.3}}

    def load_self_model(self):
        return {"blind_spots": []}

    def load_silence_records(self, limit=10):
        return []

    def load_emissions(self, limit=10):
        return []

    def load_dialogue_log(self, limit=12):
        return []

    def load_dialogue_pending(self, session_id):
        return None

    def resolve_learn_context(self, sid):
        from tabula.provenance import LearnContext

        return LearnContext(sid, True, "real")

    def save_dialogue_turn(self, turn, *, ctx=None):
        self.saved = turn


class _Cfg:
    dialogue_num_predict = 512
    dialogue_context_max_turns = 12
    dialogue_context_token_budget = 2048
    dialogue_pending_ttl_s = 300.0


def test_query_turn_returns_reply_and_persists():
    pm = _PM()

    async def fake_llm(prompt, system, client, cfg):
        return '{"reply": "I am quiet now.", "intent": null, "needs_clarification": false, "question": null}'

    out = asyncio.run(
        E.handle_turn(
            "s1",
            "what are you seeing?",
            pm=pm,
            nc=None,
            http_client=None,
            cfg=_Cfg(),
            query_fn=fake_llm,
        )
    )
    assert out.reply == "I am quiet now."
    assert out.intent is None
    assert pm.saved["reply"] == "I am quiet now."


def test_malformed_json_fails_soft():
    pm = _PM()

    async def bad_llm(prompt, system, client, cfg):
        return "not json"

    out = asyncio.run(
        E.handle_turn(
            "s1", "hi", pm=pm, nc=None, http_client=None, cfg=_Cfg(), query_fn=bad_llm
        )
    )
    assert out.error is not None and out.intent is None  # no guessed mutation


def test_query_fn_receives_register_scaled_num_predict():
    """persona.num_predict_for_register must actually reach the LLM call
    (spec §6): a low-salience (terse) turn and a high-salience (urgent) turn
    reach query_fn with a different cfg.dialogue_num_predict."""

    class _LowPM(_PM):
        def load_auspices(self):
            return {"salience": {"value": 0.05}}  # terse register

    class _HighPM(_PM):
        def load_auspices(self):
            return {"salience": {"value": 0.95}}  # urgent register

    captured: list[int] = []

    async def capture_llm(prompt, system, client, cfg):
        captured.append(cfg.dialogue_num_predict)
        return (
            '{"reply": "ok", "intent": null, "needs_clarification": false,'
            ' "question": null}'
        )

    asyncio.run(
        E.handle_turn(
            "s1",
            "hi",
            pm=_LowPM(),
            nc=None,
            http_client=None,
            cfg=_Cfg(),
            query_fn=capture_llm,
        )
    )
    asyncio.run(
        E.handle_turn(
            "s2",
            "hi",
            pm=_HighPM(),
            nc=None,
            http_client=None,
            cfg=_Cfg(),
            query_fn=capture_llm,
        )
    )
    assert len(captured) == 2
    assert captured[0] != captured[1]
    assert captured[0] < captured[1]  # terse budget < urgent budget
    assert _Cfg.dialogue_num_predict == 512  # the shared class attr is untouched


def test_dialogue_disabled_short_circuits_before_llm_or_persistence():
    pm = _PM()

    class _DisabledCfg(_Cfg):
        dialogue_enabled = False

    async def unreachable_llm(prompt, system, client, cfg):
        raise AssertionError("LLM must not be called when dialogue is disabled")

    out = asyncio.run(
        E.handle_turn(
            "s1",
            "hello",
            pm=pm,
            nc=None,
            http_client=None,
            cfg=_DisabledCfg(),
            query_fn=unreachable_llm,
        )
    )
    assert out.error == "dialogue_disabled"
    assert not hasattr(pm, "saved")  # no turn logged on the disabled path
