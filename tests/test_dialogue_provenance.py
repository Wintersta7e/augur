def test_dialogue_session_is_learnable() -> None:
    import fakeredis
    from tabula.persistence import PersistenceManager
    from imperator.dialogue.console import register_dialogue_session

    r = fakeredis.FakeStrictRedis(decode_responses=True)
    pm = PersistenceManager(r)
    register_dialogue_session(pm, "dialogue-abcd1234")
    assert pm.is_learnable_session("dialogue-abcd1234") is True


def test_registration_is_best_effort_never_crashes_dialogue() -> None:
    # Provenance registration is inert; a Redis failure at dialogue startup must
    # NOT crash the user's session. It degrades to non-learnable, not a crash.
    from unittest.mock import Mock

    from imperator.dialogue.console import register_dialogue_session

    pm = Mock()
    pm.save_session_meta.side_effect = RuntimeError("redis down")
    register_dialogue_session(pm, "dialogue-deadbeef")  # must not raise
    pm.save_session_meta.assert_called_once()
