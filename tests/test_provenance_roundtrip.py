"""The provenance contract across all three session minters."""

from __future__ import annotations

import fakeredis

from tabula.persistence import PersistenceManager
from tabula.session import SessionManager


def _fresh():
    r = fakeredis.FakeStrictRedis(decode_responses=True)
    return r, PersistenceManager(r)


def test_real_perception_session_is_learnable() -> None:
    r, pm = _fresh()
    sid = SessionManager(r).start()
    assert pm.is_learnable_session(sid) is True


def test_synthetic_session_is_not_learnable() -> None:
    r, pm = _fresh()
    sid = SessionManager(r).start(origin="synthetic", created_by="loop_test")
    assert pm.is_learnable_session(sid) is False


def test_dialogue_session_is_learnable() -> None:
    r, pm = _fresh()
    from imperator.dialogue.console import register_dialogue_session

    register_dialogue_session(pm, "dialogue-deadbeef")
    assert pm.is_learnable_session("dialogue-deadbeef") is True


def test_orphan_fallback_is_not_learnable() -> None:
    r, pm = _fresh()
    from responsum.feedback_collector import resolve_or_fabricate_session

    sid = resolve_or_fabricate_session(r, pm)
    assert pm.is_learnable_session(sid) is False


def test_unknown_id_fails_closed() -> None:
    _, pm = _fresh()
    assert pm.is_learnable_session("made-up-id") is False
    assert pm.is_learnable_session(None) is False
