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
    r.set(
        "augur:vigil:profile:typing:sample:alice", json.dumps({"observation_count": 20})
    )
    r.set("augur:vigil:profile:typing:pause:bob", json.dumps({"observation_count": 5}))
    r.set(
        "augur:vigil:profile:activity_focus:focus_change:ide",
        json.dumps({"observation_count": 15}),
    )
    # Legacy pre-series key: unattributable to a series, so it must not count.
    r.set("augur:vigil:profile:typing:legacy", json.dumps({"observation_count": 99}))
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


def test_proposals_round_trip_and_cap():
    pm = _pm()
    assert pm.load_proposals() == []
    for i in range(3):
        pm.save_proposal({"proposal_id": f"p{i}", "status": "logged"})
    assert [r["proposal_id"] for r in pm.load_proposals(limit=10)] == ["p2", "p1", "p0"]


def test_proposals_corrupt_entry_guard():
    pm = _pm()
    pm._r.lpush("augur:imperator:proposals", b"{not json")
    assert pm.load_proposals() == []


def test_applied_dedup_marker():
    pm = _pm()
    assert pm.is_proposal_applied("k1") is False
    pm.mark_proposal_applied("k1", ttl_s=3600)
    assert pm.is_proposal_applied("k1") is True
