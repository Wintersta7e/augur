"""Integration tests: reflection engine matrix tuning loop.

Exercises the full persistence-to-reflection loop against real Redis + NATS.
No Ollama — we construct feedback records directly instead of running the
full pipeline, because:
  (a) interactive stdin prompts in feedback_collector are not automatable;
  (b) we want deterministic utility values, not whatever qwen2.5 produces.

The correlator, advisor, and feedback collector subprocesses are NOT started.
run_reflection is called in-process with a fabricated feedback record.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone

import httpx
import pytest

from blackboard.config import AugurConfig
from blackboard.persistence import PersistenceManager
from reasoning.correlator import DEFAULT_ESCALATION_MATRIX, ensure_matrix_seeded
from reasoning.reflection_engine import (
    TUNING_APPLIED_KEY_PREFIX,
    run_reflection,
)

pytestmark = pytest.mark.asyncio


def _correlated_advice_event(
    rule_key: str,
    explicit: str,
    behavioral: float,
) -> dict:
    return {
        "advice_id": f"adv-{uuid.uuid4().hex[:8]}",
        "domain": "multi",
        "entity": "chess+typing",
        "severity": "medium",
        "explicit_rating": explicit,
        "behavioral_score": behavioral,
        "think_times_after": [],
        "baseline_mean_at_time": 5.0,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "correlation_found": True,
        "correlated_domains": ["typing"],
        "rule_key": rule_key,
        "escalation_rule": f"{rule_key}\u2192MEDIUM",
    }


def _standalone_advice_event() -> dict:
    return {
        "advice_id": f"adv-{uuid.uuid4().hex[:8]}",
        "domain": "chess",
        "entity": "white",
        "severity": "medium",
        "explicit_rating": "y",
        "behavioral_score": 0.8,
        "think_times_after": [],
        "baseline_mean_at_time": 5.0,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "correlation_found": False,
        "correlated_domains": [],
        "rule_key": None,
        "escalation_rule": None,
    }


def _fabricate_feedback(session_id: str, events: list[dict]) -> dict:
    return {
        "session_id": session_id,
        "advice_events": events,
        "session_summary": {
            "total_advice": len(events),
            "explicit_positive": sum(1 for e in events if e["explicit_rating"] == "y"),
            "explicit_negative": sum(1 for e in events if e["explicit_rating"] == "n"),
            "avg_behavioral_score": (
                sum(e["behavioral_score"] for e in events) / len(events)
                if events
                else 0.0
            ),
        },
    }


async def _run_reflection_with_feedback(
    session_id: str,
    feedback: dict,
    pm: PersistenceManager,
    redis_client,
    nats_conn,
) -> dict:
    """Helper to drive run_reflection with injected feedback."""
    pm.save_feedback(session_id, feedback)
    config = AugurConfig.from_env()
    http_client = httpx.AsyncClient()
    try:
        report = await run_reflection(
            session_id,
            feedback,
            pm,
            redis_client,
            http_client,
            nats_conn,
            config,
        )
    finally:
        await http_client.aclose()
    return report


async def test_bad_feedback_lowers_confidence_but_stays_in_band(
    redis_client,
    nats_conn,
) -> None:
    """One session of 3 bad LOW+LOW events drops confidence to 0.84 (still
    above 0.6 enable threshold at alpha=0.2), matrix unchanged."""
    pm = PersistenceManager(redis_client)
    ensure_matrix_seeded(pm)
    session_id = str(uuid.uuid4())

    feedback = _fabricate_feedback(
        session_id,
        [_correlated_advice_event("LOW+LOW", "n", 0.0) for _ in range(3)],
    )

    report = await _run_reflection_with_feedback(
        session_id, feedback, pm, redis_client, nats_conn
    )

    # Matrix unchanged
    matrix = pm.load_escalation_matrix()
    assert matrix["rules"]["LOW+LOW"] == "MEDIUM"

    # Confidence state persisted
    state = pm.load_rule_confidence()
    assert state is not None
    assert "LOW+LOW" in state
    # First observation from 1.0 with utility=0.2 (explicit=0, behavioral=0.5 default):
    # (1-0.2)*1.0 + 0.2*0.2 = 0.84
    assert state["LOW+LOW"]["confidence"] == 0.84
    assert state["LOW+LOW"]["restore_target"] == "MEDIUM"

    # Report contains the tuning section
    assert "correlation_tuning" in report["analyses"]
    assert report["adjustments"]["matrix_mutated"] is False


async def test_sustained_bad_feedback_disables_rule(
    redis_client,
    nats_conn,
) -> None:
    """Six consecutive bad sessions cross the disable threshold and flip
    the matrix rule to LOW."""
    pm = PersistenceManager(redis_client)
    ensure_matrix_seeded(pm)

    # Six sessions, each with bad LOW+LOW feedback (behavioral 0.01 so
    # behavioral_avg doesn't default to 0.5)
    final_state = None
    for _ in range(6):
        session_id = str(uuid.uuid4())
        feedback = _fabricate_feedback(
            session_id,
            [_correlated_advice_event("LOW+LOW", "n", 0.01)],
        )
        await _run_reflection_with_feedback(
            session_id, feedback, pm, redis_client, nats_conn
        )
        final_state = pm.load_rule_confidence()

    # After 6 sessions, rule should be disabled
    matrix = pm.load_escalation_matrix()
    assert matrix["rules"]["LOW+LOW"] == "LOW"
    assert final_state is not None
    assert final_state["LOW+LOW"]["confidence"] < 0.3
    assert final_state["LOW+LOW"]["restore_target"] == "MEDIUM"


async def test_run_reflection_is_idempotent_on_same_session(
    redis_client,
    nats_conn,
) -> None:
    """Re-running run_reflection on the same session does not double-count
    the EWMA update."""
    pm = PersistenceManager(redis_client)
    ensure_matrix_seeded(pm)
    session_id = str(uuid.uuid4())

    feedback = _fabricate_feedback(
        session_id,
        [_correlated_advice_event("LOW+LOW", "n", 0.0) for _ in range(3)],
    )

    # First pass
    await _run_reflection_with_feedback(
        session_id, feedback, pm, redis_client, nats_conn
    )
    state_after_first = pm.load_rule_confidence()
    assert state_after_first["LOW+LOW"]["confidence"] == 0.84

    # Verify idempotency marker is set
    assert redis_client.exists(f"{TUNING_APPLIED_KEY_PREFIX}{session_id}") > 0

    # Second pass on the same session_id + same feedback
    await _run_reflection_with_feedback(
        session_id, feedback, pm, redis_client, nats_conn
    )
    state_after_second = pm.load_rule_confidence()

    # State should be unchanged
    assert state_after_second["LOW+LOW"]["confidence"] == 0.84
    assert state_after_second == state_after_first


async def test_run_reflection_no_correlated_advice_no_writes(
    redis_client,
    nats_conn,
) -> None:
    """A session with only standalone advice leaves confidence state and
    idempotency marker untouched."""
    pm = PersistenceManager(redis_client)
    ensure_matrix_seeded(pm)
    session_id = str(uuid.uuid4())

    feedback = _fabricate_feedback(
        session_id,
        [_standalone_advice_event() for _ in range(3)],
    )

    await _run_reflection_with_feedback(
        session_id, feedback, pm, redis_client, nats_conn
    )

    # No confidence state written
    assert pm.load_rule_confidence() is None

    # No idempotency marker
    assert redis_client.exists(f"{TUNING_APPLIED_KEY_PREFIX}{session_id}") == 0

    # Matrix unchanged
    matrix = pm.load_escalation_matrix()
    assert matrix == DEFAULT_ESCALATION_MATRIX


async def test_manual_matrix_edit_preserved_through_disable_and_recovery(
    redis_client,
    nats_conn,
) -> None:
    """An operator sets LOW+LOW to HIGH via the MCP path. Six bad sessions
    disable the rule. Then good feedback restores it — to HIGH, not the
    hardcoded MEDIUM default."""
    pm = PersistenceManager(redis_client)
    ensure_matrix_seeded(pm)

    # Simulate manual MCP edit
    manual_matrix = {
        "version": "1.0",
        "rules": {
            **DEFAULT_ESCALATION_MATRIX["rules"],
            "LOW+LOW": "HIGH",  # operator override
        },
    }
    pm.save_escalation_matrix(manual_matrix)

    # First, one good session so restore_target gets captured as HIGH while healthy
    session_id = str(uuid.uuid4())
    feedback = _fabricate_feedback(
        session_id,
        [_correlated_advice_event("LOW+LOW", "y", 1.0)],
    )
    await _run_reflection_with_feedback(
        session_id, feedback, pm, redis_client, nats_conn
    )

    state = pm.load_rule_confidence()
    assert state["LOW+LOW"]["restore_target"] == "HIGH"
    assert pm.load_escalation_matrix()["rules"]["LOW+LOW"] == "HIGH"

    # Now six bad sessions to disable the rule
    for _ in range(6):
        sid = str(uuid.uuid4())
        bad_feedback = _fabricate_feedback(
            sid,
            [_correlated_advice_event("LOW+LOW", "n", 0.01)],
        )
        await _run_reflection_with_feedback(
            sid, bad_feedback, pm, redis_client, nats_conn
        )

    disabled_matrix = pm.load_escalation_matrix()
    assert disabled_matrix["rules"]["LOW+LOW"] == "LOW"
    state = pm.load_rule_confidence()
    assert state["LOW+LOW"]["confidence"] < 0.3
    assert state["LOW+LOW"]["restore_target"] == "HIGH"  # still HIGH, not reset

    # Now good sessions until recovery
    for _ in range(10):
        sid = str(uuid.uuid4())
        good_feedback = _fabricate_feedback(
            sid,
            [_correlated_advice_event("LOW+LOW", "y", 1.0)],
        )
        await _run_reflection_with_feedback(
            sid, good_feedback, pm, redis_client, nats_conn
        )
        final_matrix = pm.load_escalation_matrix()
        if final_matrix["rules"]["LOW+LOW"] != "LOW":
            break

    # After recovery, rule should be HIGH again (the manual edit), not MEDIUM (default)
    final_matrix = pm.load_escalation_matrix()
    assert final_matrix["rules"]["LOW+LOW"] == "HIGH"
