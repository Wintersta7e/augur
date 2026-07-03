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

    def save_dialogue_turn(self, turn):
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
