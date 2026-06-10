"""Tests for gate persistence methods — Tasks 1.1, 1.2, 1.3, and 1.4."""

from __future__ import annotations

import fakeredis
import pytest

from tabula.persistence import PersistenceManager, MAX_GATE_SILENCES


def _pm() -> PersistenceManager:
    return PersistenceManager(fakeredis.FakeStrictRedis(decode_responses=True))


# ── silence records ──────────────────────────────────────────────────────────


def test_silence_record_roundtrip() -> None:
    pm = _pm()
    rec = {
        "ts": "t",
        "decision_id": "d1",
        "state_key": "single:typing:user",
        "domain": "typing",
        "entity": "user",
        "severity": "medium",
        "arm": "habituation",
        "reason": "habituated",
        "metrics": {"h_eff": 0.9},
        "mrt_eligible": False,
        "p_withhold": None,
    }
    pm.save_silence_record(rec)
    out = pm.load_silence_records(limit=10)
    assert out[0]["decision_id"] == "d1" and out[0]["reason"] == "habituated"


def test_silences_capped() -> None:
    pm = _pm()
    for i in range(MAX_GATE_SILENCES + 50):
        pm.save_silence_record(
            {
                "ts": "t",
                "decision_id": f"d{i}",
                "state_key": "k",
                "domain": "d",
                "entity": "e",
                "severity": "medium",
                "arm": "a",
                "reason": "r",
                "metrics": {},
                "mrt_eligible": False,
                "p_withhold": None,
            }
        )
    assert len(pm.load_silence_records(limit=10_000)) <= MAX_GATE_SILENCES


def test_load_silences_corrupt_returns_empty() -> None:
    pm = _pm()
    pm._r.lpush("augur:limen:silences", "{not json")
    assert pm.load_silence_records(limit=10) == []  # guarded


# ── emissions ────────────────────────────────────────────────────────────────


def test_emission_roundtrip() -> None:

    pm = _pm()
    rec = {
        "ts": "t2",
        "decision_id": "e1",
        "state_key": "single:chess:user",
        "severity": "medium",
        "tier": 2,
        "probe": False,
        "audit_only": False,
        "withheld_reason": None,
        "mrt_eligible": False,
        "p_fire": None,
    }
    pm.save_emission(rec)
    out = pm.load_emissions(limit=10)
    assert out[0]["decision_id"] == "e1"
    assert out[0]["probe"] is False


def test_emissions_capped() -> None:
    from tabula.persistence import MAX_GATE_EMISSIONS

    pm = _pm()
    for i in range(MAX_GATE_EMISSIONS + 50):
        pm.save_emission(
            {
                "ts": "t",
                "decision_id": f"e{i}",
                "state_key": "k",
                "severity": "medium",
                "tier": 2,
                "probe": False,
                "audit_only": False,
                "withheld_reason": None,
                "mrt_eligible": False,
                "p_fire": None,
            }
        )
    assert len(pm.load_emissions(limit=10_000)) <= MAX_GATE_EMISSIONS


def test_load_emissions_corrupt_returns_empty() -> None:
    pm = _pm()
    pm._r.lpush("augur:limen:emissions", "{bad")
    assert pm.load_emissions(limit=10) == []


# ── observed ─────────────────────────────────────────────────────────────────


def test_observed_roundtrip() -> None:
    pm = _pm()
    pm.save_observed(
        {
            "ts": "t3",
            "state_key": "single:chess:user",
            "value": 3.1,
            "severity": "medium",
        }
    )
    out = pm.load_observed("single:chess:user", limit=10)
    assert len(out) == 1
    assert out[0]["value"] == 3.1


def test_observed_filtered_by_state_key() -> None:
    pm = _pm()
    pm.save_observed(
        {
            "ts": "t",
            "state_key": "single:chess:user",
            "value": 1.0,
            "severity": "medium",
        }
    )
    pm.save_observed(
        {
            "ts": "t",
            "state_key": "single:typing:user",
            "value": 2.0,
            "severity": "medium",
        }
    )
    out = pm.load_observed("single:chess:user", limit=10)
    assert all(r["state_key"] == "single:chess:user" for r in out)
    assert len(out) == 1


def test_observed_capped() -> None:
    from tabula.persistence import MAX_GATE_OBSERVED

    pm = _pm()
    for i in range(MAX_GATE_OBSERVED + 50):
        pm.save_observed(
            {"ts": "t", "state_key": "k", "value": float(i), "severity": "medium"}
        )
    assert len(pm.load_observed("k", limit=10_000)) <= MAX_GATE_OBSERVED


def test_load_observed_corrupt_returns_empty() -> None:
    pm = _pm()
    pm._r.lpush("augur:limen:observed", "{bad")
    assert pm.load_observed("k", limit=10) == []


# ── delivery failures ─────────────────────────────────────────────────────────


def test_delivery_failure_roundtrip() -> None:
    pm = _pm()

    class _FakeSig:
        state_key = "single:chess:user"
        domain = "chess"
        entity = "user"

    pm.save_delivery_failure(
        _FakeSig(), "advisor_busy_skipped", "2026-06-07T00:00:00Z", "did1"
    )
    out = pm.load_delivery_failures(limit=10)
    assert out[0]["decision_id"] == "did1"
    assert out[0]["reason"] == "advisor_busy_skipped"
    assert out[0]["state_key"] == "single:chess:user"
    assert out[0]["domain"] == "chess"
    assert out[0]["entity"] == "user"
    assert "ts" in out[0]


def test_delivery_failures_capped() -> None:
    from tabula.persistence import MAX_GATE_DELIVERY_FAILURES

    pm = _pm()

    class _FakeSig:
        state_key = "k"
        domain = "d"
        entity = "e"

    for i in range(MAX_GATE_DELIVERY_FAILURES + 50):
        pm.save_delivery_failure(_FakeSig(), "busy", f"ts{i}", f"d{i}")
    assert len(pm.load_delivery_failures(limit=10_000)) <= MAX_GATE_DELIVERY_FAILURES


def test_load_delivery_failures_corrupt_returns_empty() -> None:
    pm = _pm()
    pm._r.lpush("augur:limen:delivery_failures", "{bad")
    assert pm.load_delivery_failures(limit=10) == []


# ── Task 1.2: per-field hash stores ─────────────────────────────────────────


def test_habituation_per_field_no_clobber() -> None:
    pm = _pm()
    pm.save_habituation("single:a:b", {"h": 0.5, "last_event_ts": 1.0, "count": 3})
    pm.save_habituation_floor(
        "single:a:b", {"floor": 0.1, "last_ts": 1.0}
    )  # separate key
    assert pm.load_habituation("single:a:b")["h"] == 0.5
    assert pm.load_habituation_floor("single:a:b")["floor"] == 0.1


def test_channel_stats_refuse_at_cap(monkeypatch: object) -> None:
    import tabula.persistence as P

    monkeypatch.setattr(P, "MAX_GATE_STATE_KEYS", 2)
    pm = _pm()
    assert pm.save_channel_stats("k1", {"seen": 1}) is True
    assert pm.save_channel_stats("k2", {"seen": 1}) is True
    assert pm.save_channel_stats("k3", {"seen": 1}) is False  # refused at cap (new key)
    assert (
        pm.save_channel_stats("k1", {"seen": 2}) is True
    )  # existing key still updates


def test_load_habituation_corrupt_field_returns_default() -> None:
    pm = _pm()
    pm._r.hset("augur:limen:habituation", "k", "{bad")
    assert pm.load_habituation("k") == {}  # guarded → unseen


# ── refuse-at-cap for all six hashes ────────────────────────────────────────


def test_habituation_refuse_at_cap(monkeypatch: object) -> None:
    import tabula.persistence as P

    monkeypatch.setattr(P, "MAX_GATE_STATE_KEYS", 1)
    pm = _pm()
    assert (
        pm.save_habituation("k1", {"h": 0.1, "last_event_ts": 1.0, "count": 1}) is True
    )
    assert (
        pm.save_habituation("k2", {"h": 0.2, "last_event_ts": 2.0, "count": 1}) is False
    )
    assert (
        pm.save_habituation("k1", {"h": 0.9, "last_event_ts": 3.0, "count": 2}) is True
    )  # existing


def test_habituation_floor_refuse_at_cap(monkeypatch: object) -> None:
    import tabula.persistence as P

    monkeypatch.setattr(P, "MAX_GATE_STATE_KEYS", 1)
    pm = _pm()
    assert pm.save_habituation_floor("k1", {"floor": 0.2, "last_ts": 1.0}) is True
    assert pm.save_habituation_floor("k2", {"floor": 0.3, "last_ts": 2.0}) is False


def test_credibility_refuse_at_cap(monkeypatch: object) -> None:
    import tabula.persistence as P

    monkeypatch.setattr(P, "MAX_GATE_STATE_KEYS", 1)
    pm = _pm()
    assert pm.save_credibility("c1", {"cred": 0.7, "n": 4, "last_fb_ts": 1.0}) is True
    assert pm.save_credibility("c2", {"cred": 0.5, "n": 2, "last_fb_ts": 2.0}) is False


def test_reservoir_refuse_at_cap(monkeypatch: object) -> None:
    import tabula.persistence as P

    monkeypatch.setattr(P, "MAX_GATE_STATE_KEYS", 1)
    pm = _pm()
    assert pm.save_reservoir("k1", {"count": 1.0, "last_ts": 1.0}) is True
    assert pm.save_reservoir("k2", {"count": 2.0, "last_ts": 2.0}) is False


def test_cost_tier_memory_refuse_at_cap(monkeypatch: object) -> None:
    import tabula.persistence as P

    monkeypatch.setattr(P, "MAX_GATE_STATE_KEYS", 1)
    pm = _pm()
    assert (
        pm.save_cost_tier_memory(
            "k1", {"earned_tier2": False, "helped": 0, "count": 1, "last_ts": 1.0}
        )
        is True
    )
    assert (
        pm.save_cost_tier_memory(
            "k2", {"earned_tier2": False, "helped": 0, "count": 1, "last_ts": 2.0}
        )
        is False
    )


# ── load roundtrips for all hashes ──────────────────────────────────────────


def test_habituation_floor_roundtrip() -> None:
    pm = _pm()
    pm.save_habituation_floor("k", {"floor": 0.25, "last_ts": 42.0})
    result = pm.load_habituation_floor("k")
    assert result["floor"] == 0.25
    assert result["last_ts"] == 42.0


def test_credibility_roundtrip() -> None:
    pm = _pm()
    pm.save_credibility("chess:medium", {"cred": 0.8, "n": 10, "last_fb_ts": 99.0})
    result = pm.load_credibility("chess:medium")
    assert result["cred"] == 0.8
    assert result["n"] == 10


def test_reservoir_roundtrip() -> None:
    pm = _pm()
    pm.save_reservoir("single:typing:user", {"count": 2.5, "last_ts": 50.0})
    result = pm.load_reservoir("single:typing:user")
    assert result["count"] == 2.5


def test_cost_tier_memory_roundtrip() -> None:
    pm = _pm()
    pm.save_cost_tier_memory(
        "k", {"earned_tier2": True, "helped": 3, "count": 5, "last_ts": 1.0}
    )
    result = pm.load_cost_tier_memory("k")
    assert result["earned_tier2"] is True
    assert result["count"] == 5


def test_channel_stats_roundtrip() -> None:
    pm = _pm()
    pm.save_channel_stats(
        "k",
        {
            "seen": 5,
            "consecutive_suppressions": 2,
            "suppression_streak_started_ts": 100.0,
            "last_delivery_ts": 200.0,
            "last_ts": 250.0,
        },
    )
    result = pm.load_channel_stats("k")
    assert result["seen"] == 5
    assert result["consecutive_suppressions"] == 2


# ── load returns {} for missing ──────────────────────────────────────────────


def test_load_missing_returns_empty_dict() -> None:
    pm = _pm()
    assert pm.load_habituation("nonexistent") == {}
    assert pm.load_habituation_floor("nonexistent") == {}
    assert pm.load_credibility("nonexistent") == {}
    assert pm.load_reservoir("nonexistent") == {}
    assert pm.load_cost_tier_memory("nonexistent") == {}
    assert pm.load_channel_stats("nonexistent") == {}


# ── corrupt field guards for all six hashes ──────────────────────────────────


def test_habituation_floor_corrupt_returns_empty() -> None:
    pm = _pm()
    pm._r.hset("augur:limen:habituation_floor", "k", "{bad")
    assert pm.load_habituation_floor("k") == {}


def test_credibility_corrupt_returns_empty() -> None:
    pm = _pm()
    pm._r.hset("augur:limen:credibility", "c", "{bad")
    assert pm.load_credibility("c") == {}


def test_reservoir_corrupt_returns_empty() -> None:
    pm = _pm()
    pm._r.hset("augur:limen:reservoir", "k", "{bad")
    assert pm.load_reservoir("k") == {}


def test_cost_tier_memory_corrupt_returns_empty() -> None:
    pm = _pm()
    pm._r.hset("augur:limen:cost_tier_memory", "k", "{bad")
    assert pm.load_cost_tier_memory("k") == {}


def test_channel_stats_corrupt_returns_empty() -> None:
    pm = _pm()
    pm._r.hset("augur:limen:channel_stats", "k", "{bad")
    assert pm.load_channel_stats("k") == {}


# ── advice_rate (string key) ─────────────────────────────────────────────────


def test_advice_rate_roundtrip() -> None:
    pm = _pm()
    pm.save_advice_rate({"rate_ewma": 0.05, "last_ts": 123.0})
    result = pm.load_advice_rate()
    assert result["rate_ewma"] == 0.05
    assert result["last_ts"] == 123.0


def test_advice_rate_missing_returns_empty() -> None:
    pm = _pm()
    assert pm.load_advice_rate() == {}


def test_advice_rate_corrupt_returns_empty() -> None:
    pm = _pm()
    pm._r.set("augur:limen:advice_rate", "{bad")
    assert pm.load_advice_rate() == {}


# ── self_tolerance set ops ───────────────────────────────────────────────────


def test_self_tolerance_add_is_member() -> None:
    pm = _pm()
    pm.add_self_tolerance("single:typing:user")
    assert pm.is_self_tolerant("single:typing:user") is True
    assert pm.is_self_tolerant("single:chess:user") is False


def test_self_tolerance_remove() -> None:
    pm = _pm()
    pm.add_self_tolerance("single:typing:user")
    pm.remove_self_tolerance("single:typing:user")
    assert pm.is_self_tolerant("single:typing:user") is False


def test_load_self_tolerance_returns_set() -> None:
    pm = _pm()
    pm.add_self_tolerance("k1")
    pm.add_self_tolerance("k2")
    result = pm.load_self_tolerance()
    assert isinstance(result, set)
    assert "k1" in result and "k2" in result


def test_load_self_tolerance_empty() -> None:
    pm = _pm()
    assert pm.load_self_tolerance() == set()


# ── can_track_gate_state probe ───────────────────────────────────────────────


def test_can_track_gate_state_empty_hash() -> None:
    pm = _pm()
    # No entries yet — new key can be tracked
    assert pm.can_track_gate_state("augur:limen:channel_stats", "new:key") is True


def test_can_track_gate_state_existing_at_cap(monkeypatch: object) -> None:
    import tabula.persistence as P

    monkeypatch.setattr(P, "MAX_GATE_STATE_KEYS", 2)
    pm = _pm()
    pm.save_channel_stats("k1", {"seen": 1})
    pm.save_channel_stats("k2", {"seen": 1})
    # At cap — new key cannot be tracked
    assert pm.can_track_gate_state("augur:limen:channel_stats", "k3") is False
    # Existing key CAN still be tracked even at cap
    assert pm.can_track_gate_state("augur:limen:channel_stats", "k1") is True


# ── Task 1.3: save_gate_tuning_state (atomic) + pass_name idempotency ────────


def test_gate_tuning_state_atomic_and_independent_marker() -> None:
    pm = _pm()
    pm.save_gate_tuning_state(
        floors={"k": {"floor": 0.2, "last_ts": 1.0}},
        credibility={"typing:medium": {"cred": 0.7, "n": 4, "last_fb_ts": 1.0}},
        tolerance_add=["single:x:y"],
        advice_rate={"rate_ewma": 0.1, "last_ts": 1.0},
    )
    assert pm.load_habituation_floor("k")["floor"] == 0.2
    assert pm.load_credibility("typing:medium")["cred"] == 0.7
    assert pm.is_self_tolerant("single:x:y")
    assert pm.load_advice_rate()["rate_ewma"] == 0.1

    pm.mark_tuning_applied("sess1", pass_name="gate")
    assert pm.is_tuning_applied("sess1", pass_name="gate")
    assert not pm.is_tuning_applied("sess1", pass_name="correlation")  # independent


def test_mark_tuning_applied_default_pass_name_preserves_behavior() -> None:
    """Default pass_name should match the original 'correlation' key behavior."""
    pm = _pm()
    pm.mark_tuning_applied("sess2", pass_name="correlation")
    assert pm.is_tuning_applied("sess2", pass_name="correlation")
    assert not pm.is_tuning_applied("sess2", pass_name="gate")


# ── Task 1.4: parameterized corrupt-read guard for EVERY gate loader ─────────
# Invariant C (spec §2C/§6/§11): every load_gate_* must return its safe
# default ([] for lists, {} for dicts) when the underlying Redis value is
# corrupt — not just the loaders covered by Tasks 1.1-1.3 individually.


@pytest.mark.parametrize(
    "seed, loader, expected",
    [
        (
            lambda pm: pm._r.lpush("augur:limen:emissions", "{bad"),
            lambda pm: pm.load_emissions(limit=10),
            [],
        ),
        (
            lambda pm: pm._r.lpush("augur:limen:observed", "{bad"),
            lambda pm: pm.load_observed("k", limit=10),
            [],
        ),
        (
            lambda pm: pm._r.lpush("augur:limen:delivery_failures", "{bad"),
            lambda pm: pm.load_delivery_failures(limit=10),
            [],
        ),
        (
            lambda pm: pm._r.hset("augur:limen:channel_stats", "k", "{bad"),
            lambda pm: pm.load_channel_stats("k"),
            {},
        ),
        (
            lambda pm: pm._r.hset("augur:limen:reservoir", "k", "{bad"),
            lambda pm: pm.load_reservoir("k"),
            {},
        ),
        (
            lambda pm: pm._r.hset("augur:limen:credibility", "c", "{bad"),
            lambda pm: pm.load_credibility("c"),
            {},
        ),
        (
            lambda pm: pm._r.hset("augur:limen:cost_tier_memory", "k", "{bad"),
            lambda pm: pm.load_cost_tier_memory("k"),
            {},
        ),
        (
            lambda pm: pm._r.hset("augur:limen:habituation_floor", "k", "{bad"),
            lambda pm: pm.load_habituation_floor("k"),
            {},
        ),
        (
            lambda pm: pm._r.set("augur:limen:advice_rate", "{bad"),
            lambda pm: pm.load_advice_rate(),
            {},
        ),
    ],
)
def test_every_gate_loader_guards_corruption(
    seed: object, loader: object, expected: object
) -> None:
    pm = _pm()
    seed(pm)
    assert loader(pm) == expected
