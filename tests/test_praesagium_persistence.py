"""Praesagium persistence — episode lists, mined-pattern blob, open-prediction
hash, and the resolved-prediction log (WATCH/MULTI atomic resolve, PR6)."""

from __future__ import annotations

import fakeredis
import pytest

from tabula.persistence import PersistenceManager


def _pm():
    return PersistenceManager(fakeredis.FakeStrictRedis(decode_responses=False))


# -- episodes -----------------------------------------------------------------


def test_episode_append_sets_ttl():
    pm = _pm()
    pm.append_praesagium_episode("s1", {"kind": "anomaly"})
    assert pm._r.ttl("augur:praesagium:episodes:s1") > 0


def test_episode_append_trims_to_cap_keeping_newest():
    pm = _pm()
    for i in range(10):
        pm.append_praesagium_episode("s1", {"i": i}, cap=5)
    got = pm.load_praesagium_episodes("s1")
    assert [e["i"] for e in got] == [5, 6, 7, 8, 9]


def test_episode_index_registered_once_per_session():
    pm = _pm()
    pm.append_praesagium_episode("s1", {"i": 0})
    pm.append_praesagium_episode("s1", {"i": 1})
    pm.append_praesagium_episode("s2", {"i": 0})
    sessions = pm.list_praesagium_episode_sessions(limit=10)
    assert sessions == ["s2", "s1"]  # newest session first, s1 not duplicated


def test_episode_loader_skips_corrupt_entries():
    pm = _pm()
    pm.append_praesagium_episode("s1", {"i": 0})
    pm._r.rpush("augur:praesagium:episodes:s1", b"{not json")
    pm.append_praesagium_episode("s1", {"i": 1})
    got = pm.load_praesagium_episodes("s1")
    assert [e["i"] for e in got] == [0, 1]


def test_load_episodes_empty_for_unknown_session():
    pm = _pm()
    assert pm.load_praesagium_episodes("nope") == []


def test_list_episode_sessions_nonpositive_limit_returns_empty():
    pm = _pm()
    pm.append_praesagium_episode("s1", {"i": 0})
    assert pm.list_praesagium_episode_sessions(limit=0) == []


def test_list_episode_sessions_is_keyword_only():
    pm = _pm()
    with pytest.raises(TypeError):
        pm.list_praesagium_episode_sessions(5)  # type: ignore[misc]


# -- patterns -------------------------------------------------------------------


def test_patterns_roundtrip():
    pm = _pm()
    pm.save_praesagium_patterns({"rules": [{"a": 1}]})
    assert pm.load_praesagium_patterns() == {"rules": [{"a": 1}]}


def test_patterns_none_when_absent():
    pm = _pm()
    assert pm.load_praesagium_patterns() is None


# -- open predictions -------------------------------------------------------------


def test_open_prediction_save_load_update():
    pm = _pm()
    assert (
        pm.save_praesagium_open_prediction({"prediction_id": "p1", "pattern_id": "x"})
        is True
    )
    got = pm.load_praesagium_open_predictions()
    assert got == [{"prediction_id": "p1", "pattern_id": "x"}]
    pm.update_praesagium_open_prediction("p1", {"status": "armed"})
    got2 = pm.load_praesagium_open_predictions()
    assert got2[0]["status"] == "armed"
    assert got2[0]["pattern_id"] == "x"  # merge, not overwrite


def test_open_prediction_requires_prediction_id():
    pm = _pm()
    with pytest.raises(ValueError):
        pm.save_praesagium_open_prediction({"pattern_id": "x"})


def test_open_prediction_cap_refusal_returns_false():
    pm = _pm()
    for i in range(3):
        assert pm.save_praesagium_open_prediction({"prediction_id": f"p{i}"}, cap=3)
    assert pm.save_praesagium_open_prediction({"prediction_id": "p3"}, cap=3) is False
    # existing id keeps updating even at cap
    assert (
        pm.save_praesagium_open_prediction({"prediction_id": "p0", "x": 1}, cap=3)
        is True
    )


def test_update_open_prediction_noop_if_gone():
    pm = _pm()
    pm.update_praesagium_open_prediction("ghost", {"status": "armed"})  # no raise
    assert pm.load_praesagium_open_predictions() == []


def test_open_predictions_loader_skips_corrupt_entries():
    pm = _pm()
    pm.save_praesagium_open_prediction({"prediction_id": "p1"})
    pm._r.hset("augur:praesagium:predictions:open", "bad", b"{not json")
    got = pm.load_praesagium_open_predictions()
    assert [r["prediction_id"] for r in got] == ["p1"]


class _HdelBetweenReadWrite:
    """Wraps a real pipeline; fires a concurrent HDEL right after the FIRST
    ``hget`` (inside the WATCH), simulating a miner expiry sweep deleting a
    born-expired prediction between the CAS read and its write. Under a blind
    hget->hset RMW this would RESURRECT the resolved record; under WATCH/CAS the
    execute() aborts (WatchError) and the retry re-reads the now-absent field."""

    def __init__(self, real, client, key, pid):
        self._real = real
        self._client = client
        self._key = key
        self._pid = pid
        self._armed = True

    def __enter__(self):
        self._real.__enter__()
        return self

    def __exit__(self, *a):
        return self._real.__exit__(*a)

    def hget(self, *a, **k):
        val = self._real.hget(*a, **k)
        if self._armed:
            self._armed = False
            self._client.hdel(self._key, self._pid)  # concurrent expiry sweep
        return val

    def __getattr__(self, name):
        return getattr(self._real, name)


def test_update_open_prediction_cas_no_resurrect_on_concurrent_hdel():
    # A concurrent HDEL between the read and the write must NOT resurrect
    # a resolved record. The blind RMW would hset the merged record back.
    pm = _pm()
    key = "augur:praesagium:predictions:open"
    pm.save_praesagium_open_prediction({"prediction_id": "x1", "pattern_id": "p"})
    real_pipeline = pm._r.pipeline
    pm._r.pipeline = lambda *a, **k: _HdelBetweenReadWrite(
        real_pipeline(*a, **k), pm._r, key, "x1"
    )
    pm.update_praesagium_open_prediction("x1", {"status": "armed"})
    pm._r.pipeline = real_pipeline  # restore
    assert pm.load_praesagium_open_predictions() == []
    assert pm._r.hexists(key, "x1") in (False, 0)


def test_update_open_prediction_corrupt_field_is_dropped(caplog):
    pm = _pm()
    key = "augur:praesagium:predictions:open"
    pm._r.hset(key, "bad", b"{not json")
    pm.update_praesagium_open_prediction("bad", {"status": "armed"})  # no raise
    # The corrupt field is left untouched, not overwritten with the merge.
    assert pm._r.hget(key, "bad") == b"{not json"


# -- resolve (atomic, exactly-once — PR6) ----------------------------------------


def test_resolve_is_exactly_once():
    pm = _pm()
    pm.save_praesagium_open_prediction({"prediction_id": "p1", "pattern_id": "x"})
    rec = {"prediction_id": "p1", "outcome": "fulfilled", "resolved_ts": 5.0}
    assert pm.resolve_praesagium_prediction("p1", rec) is True
    assert pm.resolve_praesagium_prediction("p1", rec) is False  # replay = no-op
    assert [r["prediction_id"] for r in pm.load_praesagium_resolved(limit=10)] == ["p1"]
    assert pm.load_praesagium_open_predictions() == []


def test_resolve_missing_prediction_returns_false():
    pm = _pm()
    assert (
        pm.resolve_praesagium_prediction("ghost", {"prediction_id": "ghost"}) is False
    )
    assert pm.load_praesagium_resolved(limit=10) == []


def test_resolve_log_capped_by_ltrim():
    pm = _pm()
    for i in range(10):
        pid = f"p{i}"
        pm.save_praesagium_open_prediction({"prediction_id": pid})
        pm.resolve_praesagium_prediction(pid, {"prediction_id": pid}, cap=5)
    got = pm.load_praesagium_resolved(limit=100)
    assert len(got) == 5
    assert got[0]["prediction_id"] == "p9"  # newest first


def test_resolved_loader_nonpositive_limit_returns_empty():
    pm = _pm()
    assert pm.load_praesagium_resolved(limit=0) == []


def test_resolved_loader_is_keyword_only():
    pm = _pm()
    with pytest.raises(TypeError):
        pm.load_praesagium_resolved(5)  # type: ignore[misc]


def test_resolved_loader_corrupt_entry_degrades_to_empty():
    pm = _pm()
    pm.save_praesagium_open_prediction({"prediction_id": "p1"})
    pm.resolve_praesagium_prediction("p1", {"prediction_id": "p1"})
    pm._r.lpush("augur:praesagium:predictions:log", b"{not json")
    assert pm.load_praesagium_resolved(limit=10) == []


class _CountingPipe:
    """Wraps a real pipeline, counting execute() calls (mirrors the
    _FlakyPipe pattern in tests/test_nexus_matrix_ops.py)."""

    def __init__(self, real, counter):
        self._real, self._counter = real, counter

    def __enter__(self):
        self._real.__enter__()
        return self

    def __exit__(self, *a):
        return self._real.__exit__(*a)

    def execute(self):
        self._counter[0] += 1
        return self._real.execute()

    def __getattr__(self, name):
        return getattr(self._real, name)


def test_resolve_is_a_single_transactional_pipeline_execute():
    """HDEL + LPUSH + LTRIM must commit in ONE execute() call (MULTI/EXEC),
    not as three separate round-trips — the transactional half of PR6."""
    pm = _pm()
    pm.save_praesagium_open_prediction({"prediction_id": "p1"})
    counter = [0]
    real_pipeline = pm._r.pipeline
    pm._r.pipeline = lambda *a, **kw: _CountingPipe(real_pipeline(*a, **kw), counter)
    assert pm.resolve_praesagium_prediction("p1", {"prediction_id": "p1"}) is True
    assert counter[0] == 1
