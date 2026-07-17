def test_dialogue_session_is_learnable() -> None:
    import fakeredis
    from tabula.persistence import PersistenceManager
    from imperator.dialogue.console import register_dialogue_session

    r = fakeredis.FakeStrictRedis(decode_responses=True)
    pm = PersistenceManager(r)
    register_dialogue_session(pm, "dialogue-abcd1234")
    assert pm.is_learnable_session("dialogue-abcd1234") is True
