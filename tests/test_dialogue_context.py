from imperator.dialogue import context as C


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
