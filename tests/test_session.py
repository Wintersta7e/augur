"""Unit tests for SessionManager — session start/end and provenance."""

from __future__ import annotations


def test_start_writes_learnable_meta_for_real_session() -> None:
    import fakeredis
    from tabula.persistence import PersistenceManager
    from tabula.session import SessionManager

    r = fakeredis.FakeStrictRedis(decode_responses=True)
    sid = SessionManager(r).start()
    assert PersistenceManager(r).is_learnable_session(sid) is True


def test_start_synthetic_origin_is_not_learnable() -> None:
    import fakeredis
    from tabula.persistence import PersistenceManager
    from tabula.session import SessionManager

    r = fakeredis.FakeStrictRedis(decode_responses=True)
    sid = SessionManager(r).start(origin="synthetic", created_by="test")
    assert PersistenceManager(r).is_learnable_session(sid) is False


def test_start_meta_started_at_matches_current() -> None:
    import json
    import fakeredis
    from tabula.session import REDIS_KEY_CURRENT, REDIS_KEY_META, SessionManager

    r = fakeredis.FakeStrictRedis(decode_responses=True)
    sid = SessionManager(r).start()
    current = json.loads(r.get(REDIS_KEY_CURRENT))
    meta = json.loads(r.get(REDIS_KEY_META.format(sid=sid)))
    assert meta["started_at"] == current["started_at"]
