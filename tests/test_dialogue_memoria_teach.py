"""Tests for Memoria teach API — user-taught semantic memory storage.

Includes FSRS-shape compatibility coverage: taught states run through the
REAL memoria.tiers sweep pipeline (plan_sweep + apply_memory_sweep), mirroring
how disciplina.run_memory_sweep invokes it.
"""

import fakeredis
import pytest
from memoria.fsrs import retrievability
from memoria.tiers import classify, is_floor_protected, plan_sweep
from tabula.config import AugurConfig
from tabula.persistence import PersistenceManager

CFG = AugurConfig()


def _pm():
    return PersistenceManager(fakeredis.FakeStrictRedis(decode_responses=False))


def _sweep(pm, active_session, session_id):
    """Mirror disciplina.run_memory_sweep's plan+commit (no observed patterns)."""
    plan = plan_sweep(pm.load_all_memory_states(), [], active_session, session_id, CFG)
    assert pm.apply_memory_sweep(session_id, plan)
    return plan


def test_create_and_load_taught_fact():
    pm = _pm()
    mid = pm.create_user_taught_memory(
        {
            "kind": "semantic",
            "domains": ["chess", "typing"],
            "rule_key": "HIGH+HIGH",
            "severity": "MEDIUM",
        },
        source="user",
        protect=True,
    )
    facts = pm.load_taught_facts()
    assert any(f["memory_id"] == mid for f in facts)
    assert facts[0]["pattern"]["kind"] == "semantic"


def test_load_for_domains_filters():
    pm = _pm()
    pm.create_user_taught_memory(
        {"kind": "semantic", "domains": ["chess"], "rule_key": None, "severity": "LOW"},
        source="user",
    )
    assert pm.load_taught_facts_for_domains(["chess"])
    assert pm.load_taught_facts_for_domains(["typing"]) == []


def test_protected_taught_fact_survives_real_sweeps():
    """protect=True → origin_severity=HIGH engages tiers.is_floor_protected,
    so the fact is never pruned no matter how far the session clock advances.

    rule_key is deliberately None (no "HIGH" substring): protection must come
    SOLELY from the origin_severity field the teach API writes.
    """
    pm = _pm()
    mid = pm.create_user_taught_memory(
        {
            "kind": "semantic",
            "domains": ["typing"],
            "rule_key": None,
            "severity": "LOW",
        },
        source="user",  # protect defaults to True
    )
    st = pm.load_memory_state(mid)
    assert is_floor_protected(st)

    # Two sweeps far past the S=1.0 prune horizon (R = 0.9^t < 0.05 at t ≈ 29).
    for active, sid in ((40, "sweep-a"), (80, "sweep-b")):
        plan = _sweep(pm, active, sid)
        assert plan.prunes == []

    assert [f["memory_id"] for f in pm.load_taught_facts()] == [mid]
    # classify keeps it even though retrievability is below the prune floor
    st = pm.load_memory_state(mid)
    assert retrievability(st, 80, CFG) < CFG.memory_prune_r
    assert classify(st, 80, CFG) == "keep"


def test_unprotected_taught_fact_decays_per_fsrs():
    """protect=False → the taught fact rides normal FSRS decay: retrievable
    from the warm tier while R is above the prune floor, archived by the real
    sweep once the active-session clock passes the prune horizon."""
    pm = _pm()
    mid = pm.create_user_taught_memory(
        {
            "kind": "semantic",
            "domains": ["typing"],
            "rule_key": None,
            "severity": "LOW",
        },
        source="user",
        protect=False,
    )
    st = pm.load_memory_state(mid)
    assert not is_floor_protected(st)

    # Recent sweep (t=5 → R ≈ 0.59 > prune_r 0.05): kept, warm, retrievable.
    plan = _sweep(pm, 5, "sweep-early")
    assert plan.prunes == []
    assert mid in pm.list_memory_ids("warm")
    assert pm.load_taught_facts_for_domains(["typing"])

    # Past the prune horizon (t=50 → R ≈ 0.005 < 0.05): pruned + archived.
    plan = _sweep(pm, 50, "sweep-late")
    assert [p["memory_id"] for p in plan.prunes] == [mid]
    assert pm.load_taught_facts() == []
    archived = pm.load_archived_memory(mid)
    assert archived is not None and archived["status"] == "archived"


# ── Task 20 decision B/C: re-teach = FSRS review, not overwrite ────────────


def test_create_user_taught_memory_rejects_non_semantic_kind():
    # make_memory_id's docstring says "recurrence == review": the Memoria
    # teach API only owns semantic facts, so a non-semantic pattern must fail
    # closed rather than silently create a mis-shaped memory record.
    pm = _pm()
    with pytest.raises(ValueError):
        pm.create_user_taught_memory(
            {
                "kind": "episodic",
                "domains": ["chess"],
                "rule_key": None,
                "severity": "LOW",
            },
            source="user",
        )


def test_reteach_reviews_via_fsrs_not_overwrite():
    """Re-teaching an existing taught fact (same deterministic memory_id)
    strengthens it via the real memoria.fsrs.review path -- S grows and
    last_review_session advances -- instead of resetting S/D/last_review to
    the fresh-create defaults. The stored pattern is refreshed (content is
    updatable) even though non-identity fields don't affect the memory_id.
    """
    pm = _pm()
    pattern = {
        "kind": "semantic",
        "domains": ["chess", "typing"],
        "rule_key": "HIGH+HIGH",
        "severity": "MEDIUM",
    }
    mid = pm.create_user_taught_memory(pattern, source="user", session_id="s0", cfg=CFG)
    first = pm.load_memory_state(mid)
    assert first["S"] == 1.0
    assert first["last_review_session"] == 0
    assert first["source_sessions"] == []

    # Advance the active-session clock via a real (empty) sweep commit, so
    # last_review_session has somewhere to advance TO.
    _sweep(pm, 1, "sweep-0")

    mid2 = pm.create_user_taught_memory(
        {**pattern, "note": "extra"}, source="user", session_id="s1", cfg=CFG
    )
    assert mid2 == mid  # deterministic id: unaffected by non-identity fields

    second = pm.load_memory_state(mid)
    assert second["S"] > first["S"], "re-teach must strengthen S, not reset it"
    assert second["last_review_session"] == 1, "last_review_session must advance"
    assert second["source_sessions"] == ["s1"]
    assert second["pattern"]["note"] == "extra"  # content updatable
    assert second["status"] == "active"


def test_reteach_is_idempotent_per_session():
    # review()'s own idempotency contract: re-teaching twice in the SAME
    # session must not double-strengthen.
    pm = _pm()
    pattern = {
        "kind": "semantic",
        "domains": ["typing"],
        "rule_key": None,
        "severity": "LOW",
    }
    mid = pm.create_user_taught_memory(pattern, source="user", session_id="s0", cfg=CFG)
    _sweep(pm, 1, "sweep-0")
    pm.create_user_taught_memory(pattern, source="user", session_id="s1", cfg=CFG)
    once = pm.load_memory_state(mid)

    pm.create_user_taught_memory(pattern, source="user", session_id="s1", cfg=CFG)
    twice = pm.load_memory_state(mid)
    assert twice["S"] == once["S"]
    assert twice["source_sessions"] == once["source_sessions"]


def test_reteach_reactivates_archived_fact():
    # Mirrors apply.py's semantic_fact remove handler, which archives via a
    # status flip on the SAME dsr: record (not a move to augur:memoria:archive).
    # Re-teaching that record must bring it back to "active".
    pm = _pm()
    pattern = {
        "kind": "semantic",
        "domains": ["typing"],
        "rule_key": None,
        "severity": "LOW",
    }
    mid = pm.create_user_taught_memory(pattern, source="user", session_id="s0", cfg=CFG)
    st = pm.load_memory_state(mid)
    st["status"] = "archived"
    pm.save_memory_state(mid, st)
    assert pm.load_memory_state(mid)["status"] == "archived"

    pm.create_user_taught_memory(pattern, source="user", session_id="s1", cfg=CFG)
    assert pm.load_memory_state(mid)["status"] == "active"


def test_removed_fact_hidden_from_taught_fact_loads():
    # apply.py's semantic_fact remove handler archives via a status flip on
    # the live dsr record; the taught-fact loaders must hide such records or
    # a user-confirmed "forget X" keeps feeding X into every dialogue turn's
    # LLM context. Re-teaching (the decision-A re-add restore path)
    # reactivates it and it becomes visible again.
    pm = _pm()
    pattern = {
        "kind": "semantic",
        "domains": ["typing"],
        "rule_key": None,
        "severity": "LOW",
    }
    mid = pm.create_user_taught_memory(pattern, source="user", session_id="s0", cfg=CFG)
    assert [f["memory_id"] for f in pm.load_taught_facts()] == [mid]

    st = pm.load_memory_state(mid)
    pm.save_memory_state(mid, {**st, "status": "archived"})
    assert pm.load_taught_facts() == []
    assert pm.load_taught_facts_for_domains(["typing"]) == []
    # The record itself still exists (undo anchors read it directly).
    assert pm.load_memory_state(mid)["status"] == "archived"

    pm.create_user_taught_memory(pattern, source="user", session_id="s1", cfg=CFG)
    assert [f["memory_id"] for f in pm.load_taught_facts()] == [mid]
    assert pm.load_taught_facts_for_domains(["typing"])


def test_legacy_fact_without_status_still_loads():
    # Backward compatibility: taught facts stored before the status field
    # existed carry no "status" key at all -- treat missing as active.
    pm = _pm()
    state = {
        "memory_id": "legacy-1",
        "pattern": {
            "kind": "semantic",
            "domains": ["chess"],
            "rule_key": None,
            "severity": "LOW",
        },
        "S": 1.0,
        "D": 5.0,
        "last_review_session": 0,
        "tier": "warm",
        # no "status" key
        "origin_severity": "LOW",
        "memory_kind": "semantic",
        "source_sessions": [],
        "taught_by": "user",
    }
    pm.save_memory_state("legacy-1", state)
    assert [f["memory_id"] for f in pm.load_taught_facts()] == ["legacy-1"]
    assert pm.load_taught_facts_for_domains(["chess"])
