import fakeredis

from tabula.persistence import PersistenceManager


def _pm():
    return PersistenceManager(fakeredis.FakeStrictRedis(decode_responses=False))


def test_dialogue_log_roundtrip_newest_first():
    pm = _pm()
    pm.save_dialogue_turn({"ts": 1.0, "user_text": "a", "reply": "x"})
    pm.save_dialogue_turn({"ts": 2.0, "user_text": "b", "reply": "y"})
    log = pm.load_dialogue_log(limit=10)
    assert [t["user_text"] for t in log] == ["b", "a"]


def test_dialogue_log_filters_by_session_id():
    pm = _pm()
    pm.save_dialogue_turn({"ts": 1.0, "session_id": "A", "user_text": "a1"})
    pm.save_dialogue_turn({"ts": 2.0, "session_id": "B", "user_text": "b1"})
    pm.save_dialogue_turn({"ts": 3.0, "session_id": "A", "user_text": "a2"})
    pm.save_dialogue_turn({"ts": 4.0, "session_id": "A", "user_text": "a3"})

    only_a = pm.load_dialogue_log(limit=10, session_id="A")
    assert [t["user_text"] for t in only_a] == ["a3", "a2", "a1"]

    # limit applies to the filtered result, still newest-first
    only_a_capped = pm.load_dialogue_log(limit=2, session_id="A")
    assert [t["user_text"] for t in only_a_capped] == ["a3", "a2"]

    # session_id=None keeps the unfiltered behavior
    everything = pm.load_dialogue_log(limit=10)
    assert [t["user_text"] for t in everything] == ["a3", "a2", "b1", "a1"]


def test_dialogue_pending_roundtrip_and_clear():
    pm = _pm()
    assert pm.load_dialogue_pending("s1") is None
    pm.save_dialogue_pending("s1", {"tier": "light", "echo": "do X"}, ttl=300.0)
    assert pm.load_dialogue_pending("s1")["echo"] == "do X"
    pm.clear_dialogue_pending("s1")
    assert pm.load_dialogue_pending("s1") is None


def test_dialogue_audit_append_and_load():
    pm = _pm()
    pm.append_dialogue_audit({"ts": 1.0, "kind": "sigma"})
    assert pm.load_dialogue_audit(limit=5)[0]["kind"] == "sigma"
