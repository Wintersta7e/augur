import json

import fakeredis
import pytest

from tabula.persistence import MAX_DIALOGUE_DIRECTIVES, PersistenceManager


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


def test_directives_add_remove_load():
    pm = _pm()
    pm.add_dialogue_directive(
        {
            "directive_id": "d1",
            "predicate": {"context": "focused_app", "match": "appX"},
            "action": "suppress",
        }
    )
    assert any(d["directive_id"] == "d1" for d in pm.load_dialogue_directives())
    pm.remove_dialogue_directive("d1")
    assert pm.load_dialogue_directives() == []


def test_directives_load_skips_corrupt_entry():
    pm = _pm()
    pm.add_dialogue_directive({"directive_id": "good", "action": "suppress"})
    # Seed a corrupt hash field directly (bypassing the PM write path)
    pm._r.hset("augur:imperator:dialogue:directives", "bad", b"{not json")
    loaded = pm.load_dialogue_directives()
    assert [d["directive_id"] for d in loaded] == ["good"]


def test_directives_multi_entry_load():
    pm = _pm()
    pm.add_dialogue_directive({"directive_id": "d1", "action": "suppress"})
    pm.add_dialogue_directive({"directive_id": "d2", "action": "boost"})
    loaded = pm.load_dialogue_directives()
    assert {d["directive_id"] for d in loaded} == {"d1", "d2"}


def test_directives_upsert_same_id_overwrites():
    pm = _pm()
    pm.add_dialogue_directive({"directive_id": "d1", "action": "suppress"})
    pm.add_dialogue_directive({"directive_id": "d1", "action": "boost"})
    loaded = pm.load_dialogue_directives()
    assert len(loaded) == 1
    assert loaded[0]["action"] == "boost"


def test_directives_refused_at_cap_existing_still_updates():
    pm = _pm()
    # Fill the hash to the cap directly (fast path; same key/encoding as PM)
    mapping = {
        f"d{i}": json.dumps({"directive_id": f"d{i}"})
        for i in range(MAX_DIALOGUE_DIRECTIVES)
    }
    pm._r.hset("augur:imperator:dialogue:directives", mapping=mapping)

    # New id at cap → refused (returns False, not stored)
    assert pm.add_dialogue_directive({"directive_id": "overflow"}) is False
    assert len(pm.load_dialogue_directives()) == MAX_DIALOGUE_DIRECTIVES
    assert not any(
        d["directive_id"] == "overflow" for d in pm.load_dialogue_directives()
    )

    # Existing id at cap → still updates (returns True)
    assert pm.add_dialogue_directive({"directive_id": "d0", "action": "boost"}) is True
    updated = [d for d in pm.load_dialogue_directives() if d["directive_id"] == "d0"]
    assert updated[0]["action"] == "boost"


def test_directive_missing_id_raises_value_error():
    pm = _pm()
    with pytest.raises(ValueError, match="directive_id"):
        pm.add_dialogue_directive({"action": "suppress"})
