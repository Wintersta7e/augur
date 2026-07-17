"""Provenance predicate: is_learnable_session fails closed."""

from __future__ import annotations

import json

import fakeredis

from tabula.persistence import PersistenceManager
from tabula.session import REDIS_KEY_META, build_session_meta


def _pm() -> PersistenceManager:
    return PersistenceManager(fakeredis.FakeStrictRedis(decode_responses=True))


class TestBuildSessionMeta:
    def test_real_is_learnable(self) -> None:
        m = build_session_meta("s1", origin="real", created_by="x", started_at="t")
        assert m["learnable"] is True
        assert m["origin"] == "real"

    def test_synthetic_is_not_learnable(self) -> None:
        m = build_session_meta("s1", origin="synthetic", created_by="x", started_at="t")
        assert m["learnable"] is False

    def test_unattributed_is_not_learnable(self) -> None:
        m = build_session_meta(
            "s1", origin="unattributed", created_by="x", started_at="t"
        )
        assert m["learnable"] is False


class TestIsLearnableSession:
    def test_real_session_is_learnable(self) -> None:
        pm = _pm()
        pm._r.set(
            REDIS_KEY_META.format(sid="s1"),
            json.dumps(
                build_session_meta("s1", origin="real", created_by="x", started_at="t")
            ),
        )
        assert pm.is_learnable_session("s1") is True

    def test_synthetic_session_is_not_learnable(self) -> None:
        pm = _pm()
        pm._r.set(
            REDIS_KEY_META.format(sid="s1"),
            json.dumps(
                build_session_meta(
                    "s1", origin="synthetic", created_by="x", started_at="t"
                )
            ),
        )
        assert pm.is_learnable_session("s1") is False

    def test_unknown_session_fails_closed(self) -> None:
        assert _pm().is_learnable_session("never-written") is False

    def test_none_fails_closed(self) -> None:
        assert _pm().is_learnable_session(None) is False

    def test_corrupt_json_fails_closed(self) -> None:
        pm = _pm()
        pm._r.set(REDIS_KEY_META.format(sid="s1"), "{not json")
        assert pm.is_learnable_session("s1") is False

    def test_non_dict_fails_closed(self) -> None:
        pm = _pm()
        pm._r.set(REDIS_KEY_META.format(sid="s1"), json.dumps([1, 2, 3]))
        assert pm.is_learnable_session("s1") is False

    def test_missing_learnable_field_fails_closed(self) -> None:
        pm = _pm()
        pm._r.set(REDIS_KEY_META.format(sid="s1"), json.dumps({"session_id": "s1"}))
        assert pm.is_learnable_session("s1") is False

    def test_redis_error_fails_closed(self) -> None:
        class Boom:
            def get(self, *_a, **_k):
                raise RuntimeError("redis down")

        pm = PersistenceManager(Boom())
        assert pm.is_learnable_session("s1") is False


def test_provenance_ttl_outlives_reflection_reports() -> None:
    # A shorter provenance TTL would make a real session's late reflection fail
    # closed and silently stop it training. Pin the relationship executably.
    from tabula.persistence import PROVENANCE_TTL_S, SESSION_KEY_TTL_S

    assert PROVENANCE_TTL_S >= SESSION_KEY_TTL_S
