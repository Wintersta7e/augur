"""PersistenceManager Praefectus health snapshot round-trip."""

import fakeredis

from tabula.persistence import PersistenceManager


def _pm():
    return PersistenceManager(fakeredis.FakeStrictRedis())


def test_snapshot_round_trip():
    pm = _pm()
    snap = {"ts": 1.0, "faculties": {"vigil": {"overall": "alive"}}}
    pm.save_health_snapshot(snap)
    assert pm.load_health_snapshot() == snap


def test_load_missing_returns_none():
    assert _pm().load_health_snapshot() is None


def test_snapshot_uses_praefectus_key():
    pm = _pm()
    pm.save_health_snapshot({"ts": 2.0})
    assert pm._r.exists("augur:praefectus:health")
