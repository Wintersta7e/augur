def test_fabricated_fallback_is_not_learnable() -> None:
    import fakeredis
    from tabula.persistence import PersistenceManager
    from responsum.feedback_collector import resolve_or_fabricate_session

    r = fakeredis.FakeStrictRedis(decode_responses=True)  # no augur:session:current
    pm = PersistenceManager(r)
    sid = resolve_or_fabricate_session(r, pm)
    assert sid  # a uuid was produced
    assert pm.is_learnable_session(sid) is False


def test_real_current_session_is_used_and_not_overwritten() -> None:
    import fakeredis
    from tabula.persistence import PersistenceManager
    from tabula.session import SessionManager
    from responsum.feedback_collector import resolve_or_fabricate_session

    r = fakeredis.FakeStrictRedis(decode_responses=True)
    real = SessionManager(r).start()  # writes current + learnable meta
    pm = PersistenceManager(r)
    sid = resolve_or_fabricate_session(r, pm)
    assert sid == real
    assert pm.is_learnable_session(sid) is True  # untouched
