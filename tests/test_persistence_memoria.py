"""Memoria storage primitives on PersistenceManager (fakeredis)."""

import fakeredis
import pytest

from tabula.persistence import PersistenceManager
from memoria.tiers import SweepPlan


@pytest.fixture
def pm():
    return PersistenceManager(fakeredis.FakeStrictRedis(decode_responses=True))


def _state(mid, tier="warm", **kw):
    base = {
        "memory_id": mid,
        "pattern": {
            "kind": "episodic",
            "domains": ["chess"],
            "rule_key": None,
            "severity": "MEDIUM",
        },
        "S": 1.0,
        "D": 5.0,
        "last_review_session": 1,
        "tier": tier,
        "status": "active",
        "origin_severity": "MEDIUM",
        "memory_kind": "episodic",
        "source_sessions": ["s1"],
    }
    base.update(kw)
    base["tier"] = tier
    return base


def test_save_load_and_tier_index(pm):
    pm.save_memory_state("m1", _state("m1", tier="warm"))
    assert pm.load_memory_state("m1")["S"] == 1.0
    assert pm.list_memory_ids("warm") == ["m1"]
    assert pm.load_memory_state("absent") is None


def test_load_all_memory_states(pm):
    pm.save_memory_state("m1", _state("m1", tier="warm"))
    pm.save_memory_state("m2", _state("m2", tier="cold"))
    ids = {s["memory_id"] for s in pm.load_all_memory_states()}
    assert ids == {"m1", "m2"}


def test_states_with_bytes_redis():
    """Live connect_redis has NO decode_responses → smembers returns bytes;
    list_memory_ids/load_all_memory_states must decode set members."""
    pm_b = PersistenceManager(fakeredis.FakeStrictRedis())  # bytes mode
    pm_b.save_memory_state("mb", _state("mb", tier="warm"))
    assert pm_b.list_memory_ids("warm") == ["mb"]
    assert {s["memory_id"] for s in pm_b.load_all_memory_states()} == {"mb"}


def test_save_memory_state_single_tier_membership(pm):
    """Re-saving with a changed tier must not leave a stale index entry."""
    pm.save_memory_state("m1", _state("m1", tier="warm"))
    pm.save_memory_state("m1", _state("m1", tier="cold"))
    assert pm.list_memory_ids("warm") == []
    assert pm.list_memory_ids("cold") == ["m1"]


def test_processed_sessions_clock(pm):
    assert pm.active_session_count() == 0
    assert pm.is_session_processed("s1") is False
    pm.apply_memory_sweep("s1", SweepPlan())
    assert pm.active_session_count() == 1
    assert pm.is_session_processed("s1") is True


def test_apply_memory_sweep_atomic_and_idempotent(pm):
    plan = SweepPlan(creates=[_state("m1", tier="warm")])
    assert pm.apply_memory_sweep("s1", plan) is True
    assert pm.load_memory_state("m1") is not None
    assert pm.list_memory_ids("warm") == ["m1"]
    plan2 = SweepPlan(creates=[_state("m2", tier="warm")])
    assert pm.apply_memory_sweep("s1", plan2) is False
    assert pm.load_memory_state("m2") is None
    assert pm.active_session_count() == 1


def test_apply_memory_sweep_promote_demote_prune(pm):
    pm.save_memory_state("w", _state("w", tier="warm"))
    pm.save_memory_state("c", _state("c", tier="cold"))
    pm.save_memory_state("p", _state("p", tier="warm"))
    plan = SweepPlan(
        promotions=[_state("w", tier="cold")],
        demotions=[_state("c", tier="warm")],
        prunes=[_state("p", tier="warm")],
    )
    assert pm.apply_memory_sweep("s2", plan) is True
    assert pm.list_memory_ids("cold") == ["w"]
    assert pm.list_memory_ids("warm") == ["c"]
    assert pm.load_memory_state("p") is None
    arch = pm.load_archived_memory("p")
    assert arch is not None and arch["status"] == "archived"


def test_record_memory_review_hook(pm):
    pm.save_memory_state("m1", _state("m1", S=1.0, source_sessions=["s1"]))
    pm.apply_memory_sweep("s2", SweepPlan())  # advance clock to 1
    pm.record_memory_review("m1", "s3", pm.active_session_count() + 1)
    assert pm.load_memory_state("m1")["S"] == 1.5
