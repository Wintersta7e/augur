"""Tests for gate persistence methods — Task 1.1 (append-log save/load)."""

from __future__ import annotations

import fakeredis

from blackboard.persistence import PersistenceManager, MAX_GATE_SILENCES


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
    pm._r.lpush("augur:gate:silences", "{not json")
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
    from blackboard.persistence import MAX_GATE_EMISSIONS

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
    pm._r.lpush("augur:gate:emissions", "{bad")
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
    from blackboard.persistence import MAX_GATE_OBSERVED

    pm = _pm()
    for i in range(MAX_GATE_OBSERVED + 50):
        pm.save_observed(
            {"ts": "t", "state_key": "k", "value": float(i), "severity": "medium"}
        )
    assert len(pm.load_observed("k", limit=10_000)) <= MAX_GATE_OBSERVED


def test_load_observed_corrupt_returns_empty() -> None:
    pm = _pm()
    pm._r.lpush("augur:gate:observed", "{bad")
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
    from blackboard.persistence import MAX_GATE_DELIVERY_FAILURES

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
    pm._r.lpush("augur:gate:delivery_failures", "{bad")
    assert pm.load_delivery_failures(limit=10) == []
