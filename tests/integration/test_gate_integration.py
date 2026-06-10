"""Integration tests: advisor gate end-to-end against real Redis + NATS.

Exercises the advisor gate (spec §11 "Integration") against the live
``augur-redis-1`` / ``augur-nats-1`` containers — NO fakeredis, NO Ollama.

Following the established ``test_matrix_tuning_loop.py`` pattern, the LLM
(``query_ollama``) is injected as a deterministic stub and the interactive
feedback subprocess is not started (its stdin prompt is not automatable); the
gate's per-message control flow (``reasoning.augur_advisor.process_message``)
is driven in-process so that every Redis write goes through a real
``PersistenceManager`` and every NATS publish traverses a real connection,
verified by a real subscriber.

Scenarios (Task 12.1 / spec §11):
  1. a gate SUPPRESS writes the silence log AND publishes
     ``augur.advisor.suppressed`` AND the console dedups it;
  2. an exempt high+correlated event fires end-to-end even when the
     reasoning_lock is held;
  3. a Tier-1 note is published on ``augur.reasoning.advice`` (tier=1) and
     reaches the feedback collector;
  4. hot-reload — ``analyze_gate`` writes tuned params that a subsequent
     ``evaluate`` reads;
  5. refuse-at-cap behavior;
  6. ``anti_starvation`` releases a saturated channel.

Varied baselines are used everywhere (zero-variance gotcha).
"""

from __future__ import annotations

import asyncio
import json
import uuid
from datetime import datetime, timezone
from typing import Any

import pytest
from unittest.mock import AsyncMock, MagicMock

from tabula.config import AugurConfig
from tabula.persistence import PersistenceManager
from vox.console_display import dedup_should_suppress, update_last_rendered
from perception.feedback_collector import PendingAdvice, _resolve_primary_domain
from reasoning.advisor_gate import Gate, GateDecision, build_signature
from reasoning.advisor_gate_scheduler import MustFireScheduler
from reasoning.augur_advisor import (
    PUBLISH_SUBJECT,
    SUBJECT_SUPPRESSED,
    process_message,
)
from reasoning.reflection_engine import analyze_gate

pytestmark = pytest.mark.asyncio

# A frozen "now" so gate decisions are deterministic (the gate takes an explicit
# `now`; never reads the wall clock inside decision logic).
NOW = 1_000_000.0

# Channel-stats key the gate's cap probe checks (advisor_gate._CHANNEL_STATS_KEY).
_CHANNEL_STATS_KEY = "augur:gate:channel_stats"


def _pm(redis_client: Any) -> PersistenceManager:
    return PersistenceManager(redis_client)


def _scheduler(lock: asyncio.Lock | None = None) -> MustFireScheduler:
    return MustFireScheduler(
        lock or asyncio.Lock(),
        max_release_wait_s=30,
        max_release_overtake=5,
        now=lambda: NOW,
    )


async def _collect(
    nc: Any, subject: str, holder: list[dict], stop: asyncio.Event
) -> Any:
    async def _cb(msg: Any) -> None:
        try:
            holder.append(json.loads(msg.data.decode()))
        except (json.JSONDecodeError, UnicodeDecodeError):
            return
        stop.set()

    sub = await nc.subscribe(subject, cb=_cb)
    return sub


async def _run_message(
    *,
    payload: dict,
    gate: Gate,
    scheduler: MustFireScheduler,
    pm: PersistenceManager,
    nc: Any,
    config: AugurConfig,
    redis_client: Any,
    query_ollama: Any | None = None,
) -> None:
    if query_ollama is None:
        query_ollama = AsyncMock(return_value=("synthetic advice", 5.0))
    await process_message(
        payload=payload,
        gate=gate,
        scheduler=scheduler,
        pm=pm,
        nc=nc,
        http_client=MagicMock(),
        redis_client=redis_client,
        classifier_lane=MagicMock(),
        config=config,
        now=NOW,
        query_ollama=query_ollama,
    )


# ── Scenario 1: suppress → silence log + suppressed event + console dedup ──────


async def test_suppress_logs_silence_publishes_suppressed_and_console_dedups(
    redis_client, nats_conn
) -> None:
    """A central-tolerance SUPPRESS writes one silence record, publishes
    ``augur.advisor.suppressed`` (received over real NATS), and the console
    contract dedups the originating anomaly via that payload."""
    pm = _pm(redis_client)
    config = AugurConfig()

    # central_tolerance suppresses a non-high single event whose state_key is in
    # the offline-learned self-tolerance set — a deterministic, real suppressor.
    state_key = "single:chess:white"
    pm.add_self_tolerance(state_key)

    received: list[dict] = []
    stop = asyncio.Event()
    sub = await _collect(nats_conn, SUBJECT_SUPPRESSED, received, stop)

    origin_ts = datetime.now(timezone.utc).isoformat()
    payload = {
        "combined_severity": "MEDIUM",
        "correlation_found": False,
        "primary_anomaly": {
            "domain": "chess",
            "entity": "white",
            "value": 2.3,  # varied
            "severity": "medium",
            "baseline_mean": 1.7,  # varied baseline (zero-variance gotcha)
            "timestamp": origin_ts,
        },
    }

    gate = Gate(config=config)
    await _run_message(
        payload=payload,
        gate=gate,
        scheduler=_scheduler(),
        pm=pm,
        nc=nats_conn,
        config=config,
        redis_client=redis_client,
    )

    # (A) exactly one silence record, with the central_tolerance reason.
    silences = pm.load_silence_records(limit=10)
    assert len(silences) == 1
    assert silences[0]["reason"] == "central_tolerance_learned_self"
    assert silences[0]["state_key"] == state_key

    # No advice was published (suppress path returns before any fire).
    advice = redis_client.get("augur:reasoning:last_advice")
    assert advice is None

    # The suppressed event reached a real NATS subscriber.
    await asyncio.wait_for(stop.wait(), timeout=5.0)
    await sub.unsubscribe()
    assert len(received) == 1
    event = received[0]
    assert event["reason"] == "central_tolerance_learned_self"
    assert event["domain"] == "chess" and event["entity"] == "white"
    assert event["timestamp"] == origin_ts  # ORIGINATING ts → console dedup

    # Console contract: rendering the suppressed event + update_last_rendered
    # makes dedup_should_suppress fire for the same originating anomaly.
    last_rendered: dict[str, tuple[str, str]] = {}
    update_last_rendered(last_rendered, event)
    assert dedup_should_suppress(last_rendered, payload["primary_anomaly"]) is True


# ── Scenario 2: exempt fires end-to-end even when the reasoning_lock is held ───


async def test_exempt_fires_through_scheduler_with_lock_held(
    redis_client, nats_conn
) -> None:
    """An exempt high+correlated event awaits and fires via the scheduler even
    while another delivery holds the reasoning_lock (invariant B at the hook)."""
    pm = _pm(redis_client)
    config = AugurConfig()
    scheduler = _scheduler()

    received: list[dict] = []
    stop = asyncio.Event()

    async def _cb(msg: Any) -> None:
        try:
            data = json.loads(msg.data.decode())
        except (json.JSONDecodeError, UnicodeDecodeError):
            return
        received.append(data)
        stop.set()

    sub = await nats_conn.subscribe(PUBLISH_SUBJECT, cb=_cb)

    exempt_payload = {
        "combined_severity": "HIGH",
        "correlation_found": True,
        "involved_domains": ["typing", "chess"],
        "correlated_events": [{"domain": "chess"}],
        "primary_anomaly": {
            "domain": "typing",
            "entity": "user",
            "value": 4.8,  # varied
            "severity": "high",
            "baseline_mean": 1.1,  # varied baseline
            "timestamp": datetime.now(timezone.utc).isoformat(),
        },
    }
    sig = build_signature(exempt_payload)
    assert sig.exempt is True  # precondition

    gate = Gate(config=config)

    # Hold the reasoning_lock the way a real in-flight ordinary delivery does —
    # through the scheduler — so the exempt must-fire queues behind it.  Release
    # it shortly (via the scheduler, which re-dispatches the queued must-fire).
    assert scheduler.try_acquire_ordinary() is True

    async def _release_soon() -> None:
        await asyncio.sleep(0.3)
        scheduler.release_ordinary()

    releaser = asyncio.create_task(_release_soon())

    await _run_message(
        payload=exempt_payload,
        gate=gate,
        scheduler=scheduler,
        pm=pm,
        nc=nats_conn,
        config=config,
        redis_client=redis_client,
    )
    await releaser

    await asyncio.wait_for(stop.wait(), timeout=5.0)
    await sub.unsubscribe()
    # The exempt event fired (advice published) despite the held lock.  A
    # correlation fires on the "multi" advice domain (build_advice_event path);
    # the linkage is the decision_id, which is what threads to feedback/MRT.
    assert len(received) == 1
    assert received[0]["domain"] == "multi"
    assert received[0]["correlation_found"] is True
    assert received[0]["decision_id"] is not None

    # Exempt path is audit-only: no gating-visible silence, no channel_stats.
    assert pm.load_silence_records(limit=10) == []
    assert pm.load_channel_stats(sig.state_key) == {}


# ── Scenario 3: Tier-1 note on augur.reasoning.advice (tier=1) reaches feedback


async def test_tier1_note_published_and_reaches_feedback(
    redis_client, nats_conn
) -> None:
    """A cost-tier downgrade publishes a Tier-1 note on the advice subject with
    ``tier=1`` + ``decision_id``; the feedback collector's ingestion contract
    accepts it (builds a PendingAdvice carrying the decision_id)."""
    pm = _pm(redis_client)
    config = AugurConfig()

    received: list[dict] = []
    stop = asyncio.Event()

    async def _cb(msg: Any) -> None:
        try:
            received.append(json.loads(msg.data.decode()))
        except (json.JSONDecodeError, UnicodeDecodeError):
            return
        stop.set()

    sub = await nats_conn.subscribe(PUBLISH_SUBJECT, cb=_cb)

    payload = {
        "combined_severity": "MEDIUM",
        "correlation_found": False,
        "primary_anomaly": {
            "domain": "typing",
            "entity": "user",
            "value": 2.6,  # varied
            "severity": "medium",
            "baseline_mean": 1.4,  # varied baseline
            "timestamp": datetime.now(timezone.utc).isoformat(),
        },
    }

    # Force a downgrade to Tier-1 via a single-arm gate so the test targets the
    # cost-tier-router → downgrade(tier=1) path deterministically.
    def _downgrade(gate, sig, state, config, now, rng):  # noqa: ANN001
        return GateDecision.downgrade(
            "cost_tier_downgrade_note", deciding_arm="cost_tier_router", tier=1
        )

    gate = Gate(arms=[_downgrade], config=config)
    await _run_message(
        payload=payload,
        gate=gate,
        scheduler=_scheduler(),
        pm=pm,
        nc=nats_conn,
        config=config,
        redis_client=redis_client,
    )

    await asyncio.wait_for(stop.wait(), timeout=5.0)
    await sub.unsubscribe()
    assert len(received) == 1
    note = received[0]
    assert note["tier"] == 1
    assert note["domain"] == "typing"
    assert note["decision_id"] is not None

    # Feedback ingestion contract: on_advice reads exactly these fields and
    # builds a PendingAdvice — the note "reaches" feedback faithfully.
    assert _resolve_primary_domain(note) == "typing"
    pending = PendingAdvice(
        advice_id="t1",
        domain=_resolve_primary_domain(note),
        entity=note.get("player", "?"),
        severity=note.get("severity", "?"),
        baseline_mean=note.get("think_time", 0),
        timestamp=datetime.now(timezone.utc).isoformat(),
        decision_id=note.get("decision_id"),
        probe=bool(note.get("probe", False)),
        mrt_eligible=bool(note.get("mrt_eligible", False)),
        p_fire=note.get("p_fire"),
    )
    rec = pending.to_record()
    assert rec["decision_id"] == note["decision_id"]


# ── Scenario 4: hot-reload — analyze_gate writes tuned params evaluate reads ───


async def test_hot_reload_analyze_gate_tunes_self_tolerance_read_by_evaluate(
    redis_client, nats_conn
) -> None:
    """analyze_gate (offline) adds a chronically-dismissed channel to the
    self-tolerance set; a subsequent in-process evaluate reads it and suppresses
    — no restart (hot-reload via the shared Redis state)."""
    pm = _pm(redis_client)
    config = AugurConfig()

    domain, entity = "chess", "white"
    state_key = f"single:{domain}:{entity}"

    # Precondition: the channel is NOT yet self-tolerant, so central_tolerance
    # (Arm 1) is not the deciding arm — a fresh single+medium is instead handled
    # by a later suppressor (reservoir), never central_tolerance.
    sig = build_signature(
        {
            "combined_severity": "MEDIUM",
            "correlation_found": False,
            "primary_anomaly": {
                "domain": domain,
                "entity": entity,
                "value": 2.2,
                "severity": "medium",
            },
        }
    )
    before = Gate(config=config).evaluate(sig, pm, config, now=NOW)
    assert before.reason != "central_tolerance_learned_self"
    assert not pm.is_self_tolerant(state_key)

    # Fabricate a session's feedback: 6 advice events on this channel (chronic),
    # 4 explicitly dismissed ("n") — exceeds GATE_CHRONIC_MIN_PRESENCE=5 and
    # GATE_DISMISSAL_MIN=3, so analyze_gate adds it to self_tolerance.
    session_id = str(uuid.uuid4())
    advice_events = []
    for i in range(6):
        advice_events.append(
            {
                "advice_id": f"adv-{i}",
                "domain": domain,
                "entity": entity,
                "severity": "medium",
                "explicit_rating": "n" if i < 4 else "y",
                "behavioral_score": 0.5,
                "baseline_mean_at_time": 1.5 + i * 0.1,  # varied
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "correlation_found": False,
                "correlated_domains": [],
                "rule_key": None,
                "escalation_rule": None,
                "involved_domains": [domain],
                "behavioral_finalized": True,
            }
        )
    pm.save_feedback(session_id, {"advice_events": advice_events})

    report = analyze_gate(session_id, pm, config)
    assert state_key in report["tolerance_added"]
    assert pm.is_self_tolerant(state_key)

    # Hot-reload: a fresh evaluate reads the tuned self-tolerance set and now
    # suppresses the same medium event via central_tolerance.
    after = Gate(config=config).evaluate(sig, pm, config, now=NOW)
    assert after.action == "suppress"
    assert after.reason == "central_tolerance_learned_self"

    # Idempotent: a second pass over the same session does not re-apply.
    again = analyze_gate(session_id, pm, config)
    assert again.get("skipped") is True


# ── Scenario 5: refuse-at-cap behavior (cap fail-open) ────────────────────────


async def test_refuse_at_cap_fails_open_to_fire(
    redis_client, nats_conn, monkeypatch
) -> None:
    """When the channel_stats hash is at MAX_GATE_STATE_KEYS, a new state_key
    that WOULD be suppressed cannot be tracked → the gate fails open to
    FIRE("cap_fail_open") rather than silence it indefinitely (invariant D)."""
    import tabula.persistence as P

    # Shrink the cap so we can fill the hash cheaply with real Redis writes.
    monkeypatch.setattr(P, "MAX_GATE_STATE_KEYS", 2)
    pm = _pm(redis_client)
    config = AugurConfig()

    # Fill channel_stats to the cap with two existing (different) keys.
    assert pm.save_channel_stats("single:chess:a", {"seen": 1}) is True
    assert pm.save_channel_stats("single:typing:b", {"seen": 1}) is True
    # A brand-new key is now untrackable.
    new_key = "single:activity:newuser"
    assert pm.can_track_gate_state(_CHANNEL_STATS_KEY, new_key) is False

    # This new-key event would be suppressed by central_tolerance (we add it to
    # the tolerance set), but at cap it cannot be tracked → cap_fail_open.
    pm.add_self_tolerance(new_key)
    sig = build_signature(
        {
            "combined_severity": "MEDIUM",
            "correlation_found": False,
            "primary_anomaly": {
                "domain": "activity",
                "entity": "newuser",
                "value": 2.0,
                "severity": "medium",
            },
        }
    )
    assert sig.state_key == new_key
    decision = Gate(config=config).evaluate(sig, pm, config, now=NOW)
    assert decision.action == "fire"
    assert decision.reason == "cap_fail_open"

    # An EXISTING key at cap still suppresses normally (cap only blocks new keys).
    existing_key = "single:chess:a"
    pm.add_self_tolerance(existing_key)
    sig_existing = build_signature(
        {
            "combined_severity": "MEDIUM",
            "correlation_found": False,
            "primary_anomaly": {
                "domain": "chess",
                "entity": "a",
                "value": 2.0,
                "severity": "medium",
            },
        }
    )
    d2 = Gate(config=config).evaluate(sig_existing, pm, config, now=NOW)
    assert d2.action == "suppress"


# ── Scenario 6: anti_starvation releases a saturated channel ──────────────────


async def test_anti_starvation_releases_saturated_channel(
    redis_client, nats_conn
) -> None:
    """A channel saturated past gate_max_consecutive_suppressions is force-fired
    by anti_starvation even though a suppressor would otherwise silence it, and
    the release fires end-to-end through the must-fire scheduler (invariant D)."""
    pm = _pm(redis_client)
    config = AugurConfig()

    domain, entity = "typing", "user"
    state_key = f"single:{domain}:{entity}"

    # Make the channel suppressable (central_tolerance) AND saturate it.
    pm.add_self_tolerance(state_key)
    pm.save_channel_stats(
        state_key,
        {
            "seen": 20,
            "consecutive_suppressions": config.gate_max_consecutive_suppressions,
            "suppression_streak_started_ts": NOW - 10.0,
            "last_ts": NOW - 1.0,
        },
    )

    sig = build_signature(
        {
            "combined_severity": "MEDIUM",
            "correlation_found": False,
            "primary_anomaly": {
                "domain": domain,
                "entity": entity,
                "value": 2.4,
                "severity": "medium",
            },
        }
    )

    # Pure-decision check: anti_starvation converts the suppress to a release.
    decision = Gate(config=config).evaluate(sig, pm, config, now=NOW)
    assert decision.action == "fire"
    assert decision.reason == "anti_starvation_release"

    # End-to-end: the release is a must-fire and reaches NATS as published advice.
    received: list[dict] = []
    stop = asyncio.Event()

    async def _cb(msg: Any) -> None:
        try:
            received.append(json.loads(msg.data.decode()))
        except (json.JSONDecodeError, UnicodeDecodeError):
            return
        stop.set()

    sub = await nats_conn.subscribe(PUBLISH_SUBJECT, cb=_cb)

    payload = {
        "combined_severity": "MEDIUM",
        "correlation_found": False,
        "primary_anomaly": {
            "domain": domain,
            "entity": entity,
            "value": 2.4,  # varied
            "severity": "medium",
            "baseline_mean": 1.2,  # varied baseline
            "timestamp": datetime.now(timezone.utc).isoformat(),
        },
    }
    await _run_message(
        payload=payload,
        gate=Gate(config=config),
        scheduler=_scheduler(),
        pm=pm,
        nc=nats_conn,
        config=config,
        redis_client=redis_client,
    )

    await asyncio.wait_for(stop.wait(), timeout=5.0)
    await sub.unsubscribe()
    assert len(received) == 1
    assert received[0]["domain"] == domain

    # The delivery reset the saturation streak (no longer starved).
    stats = pm.load_channel_stats(state_key)
    assert stats.get("consecutive_suppressions") == 0
    assert stats.get("suppression_streak_started_ts") is None
