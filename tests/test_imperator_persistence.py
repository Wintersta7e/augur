import json

import fakeredis
from tabula.persistence import PersistenceManager


def _pm() -> PersistenceManager:
    return PersistenceManager(fakeredis.FakeStrictRedis(decode_responses=False))


def test_auspices_round_trip():
    pm = _pm()
    assert pm.load_auspices() is None
    snap = {"schema_version": 1, "generated_at": 1.0, "salience": {"value": 0.5}}
    pm.save_auspices(snap)
    assert pm.load_auspices() == snap


def test_self_model_round_trip():
    pm = _pm()
    assert pm.load_self_model() is None
    snap = {"schema_version": 1, "competence": {"value": 0.7}}
    pm.save_self_model(snap)
    assert pm.load_self_model() == snap


def test_scan_baseline_maturity_tallies_trained():
    pm = _pm()
    r = pm._r
    r.set("augur:vigil:profile:typing:alice", json.dumps({"observation_count": 20}))
    r.set("augur:vigil:profile:typing:bob", json.dumps({"observation_count": 5}))
    r.set(
        "augur:vigil:profile:activity_focus:ide", json.dumps({"observation_count": 15})
    )
    out = pm.scan_baseline_maturity(trained_obs=15)
    assert out["total"] == 3
    assert out["trained"] == 2
    assert out["untrained"] == 1
    assert out["by_domain"]["typing"] == {"total": 2, "trained": 1}


def test_load_all_channel_stats_returns_every_field():
    pm = _pm()
    pm._r.hset(
        "augur:limen:channel_stats", "k1", json.dumps({"consecutive_suppressions": 3})
    )
    pm._r.hset(
        "augur:limen:channel_stats", "k2", json.dumps({"consecutive_suppressions": 0})
    )
    out = pm.load_all_channel_stats()
    assert out["k1"]["consecutive_suppressions"] == 3
    assert out["k2"]["consecutive_suppressions"] == 0


def test_load_all_channel_stats_empty():
    assert _pm().load_all_channel_stats() == {}
