"""Lane-1 persistence: MRT rating-session count (1B) + prompt realized-score
pair (1E)."""

import fakeredis

from tabula.persistence import PersistenceManager


def _pm() -> PersistenceManager:
    return PersistenceManager(fakeredis.FakeStrictRedis())


def test_mrt_rating_session_count_dedups():
    pm = _pm()
    assert pm.count_mrt_rating_sessions() == 0
    pm.mark_mrt_rating_session("s1")
    pm.mark_mrt_rating_session("s1")  # set semantics → idempotent
    pm.mark_mrt_rating_session("s2")
    assert pm.count_mrt_rating_sessions() == 2


def test_prompt_score_pair_and_update():
    pm = _pm()
    pm.save_prompt("typing", "prompt A", score=0.7)  # A current
    pm.save_prompt("typing", "prompt B", score=0.4)  # B current, A → history
    cur, prev = pm.get_prompt_score_pair("typing")
    assert cur == 0.4 and prev == 0.7
    pm.update_current_prompt_score("typing", 0.55)  # restamp B's realized score
    cur, prev = pm.get_prompt_score_pair("typing")
    assert cur == 0.55 and prev == 0.7


def test_prompt_score_pair_missing_returns_none():
    pm = _pm()
    assert pm.get_prompt_score_pair("nodomain") == (None, None)
    # update on a missing current key is a no-op (no crash)
    pm.update_current_prompt_score("nodomain", 0.9)
    assert pm.get_prompt_score_pair("nodomain") == (None, None)
