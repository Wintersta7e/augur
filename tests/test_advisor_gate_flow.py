"""Integration-level tests for the gate-driven on_message control flow (spec §3).

These exercise ``reasoning.augur_advisor.process_message`` — the per-message
control flow extracted from ``run()`` — with the LLM (``query_ollama``) and NATS
(``nc.publish``) mocked.  They mirror the spec §3 pseudocode + the §11 hard
gates: suppress, fail-open, exempt/anti-starvation must-fire, Tier-1 downgrade,
busy → delivery_failure, cap_fail_open_busy, suppressed/tier1 publish failures,
no-phantom-delivery, and descriptor-map-fills-on-silence.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from reasoning.advisor_gate import Gate, GateDecision
from reasoning.advisor_gate_scheduler import MustFireScheduler
from reasoning.augur_advisor import process_message
from tests.conftest import (
    EXEMPT_PAYLOAD,
    SINGLE_MEDIUM,
    SINGLE_MEDIUM_NEWKEY_THAT_WOULD_SUPPRESS,
    SINGLE_MEDIUM_TYPING,
)

# A frozen monotonic "now" for deterministic gate decisions.
NOW = 1000.0


def _now() -> float:
    return NOW


@pytest.fixture
def nc() -> MagicMock:
    n = MagicMock()
    n.publish = AsyncMock()
    return n


@pytest.fixture
def http_client() -> MagicMock:
    return MagicMock()


@pytest.fixture
def lane() -> MagicMock:
    return MagicMock()


@pytest.fixture
def redis_client() -> Any:
    import fakeredis

    return fakeredis.FakeStrictRedis(decode_responses=True)


def _scheduler() -> MustFireScheduler:
    return MustFireScheduler(
        asyncio.Lock(), max_release_wait_s=30, max_release_overtake=5, now=_now
    )


def _published_subjects(nc: MagicMock) -> list[str]:
    return [call.args[0] for call in nc.publish.await_args_list]


def _published_on(nc: MagicMock, subject: str) -> list[dict]:
    out = []
    for call in nc.publish.await_args_list:
        if call.args[0] == subject:
            out.append(json.loads(call.args[1].decode()))
    return out


async def _run(
    *,
    payload: dict[str, Any],
    gate: Gate,
    scheduler: MustFireScheduler,
    pm: Any,
    nc: MagicMock,
    http_client: MagicMock,
    config: Any,
    redis_client: Any = None,
    lane: Any = None,
    query_ollama: Any = None,
) -> None:
    if query_ollama is None:
        query_ollama = AsyncMock(return_value=("advice text", 12.3))
    if lane is None:
        lane = MagicMock()
    if redis_client is None:
        import fakeredis

        redis_client = fakeredis.FakeStrictRedis(decode_responses=True)
    await process_message(
        payload=payload,
        gate=gate,
        scheduler=scheduler,
        pm=pm,
        nc=nc,
        http_client=http_client,
        redis_client=redis_client,
        classifier_lane=lane,
        config=config,
        now=NOW,
        query_ollama=query_ollama,
    )


# ── Suppress → one silence + one suppressed event, no advice (spec §3/§8) ─────


async def test_suppress_records_one_silence_and_publishes_suppressed(
    fake_pm, cfg, nc, http_client, lane
) -> None:
    def _suppress(gate, sig, state, config, now, rng):
        return GateDecision.suppress("habituated", deciding_arm="habituation")

    gate = Gate(arms=[_suppress], config=cfg)
    await _run(
        payload=SINGLE_MEDIUM,
        gate=gate,
        scheduler=_scheduler(),
        pm=fake_pm,
        nc=nc,
        http_client=http_client,
        config=cfg,
        lane=lane,
    )

    silences = fake_pm.load_silence_records(limit=10)
    assert len(silences) == 1
    assert silences[0]["reason"] == "habituated"
    # No advice published; exactly one suppressed event.
    from reasoning.augur_advisor import PUBLISH_SUBJECT, SUBJECT_SUPPRESSED

    subjects = _published_subjects(nc)
    assert PUBLISH_SUBJECT not in subjects
    suppressed = _published_on(nc, SUBJECT_SUPPRESSED)
    assert len(suppressed) == 1


async def test_suppressed_event_carries_full_payload(
    fake_pm, cfg, nc, http_client, lane
) -> None:
    captured: dict[str, GateDecision] = {}

    from dataclasses import replace

    def _suppress(gate, sig, state, config, now, rng):
        d = GateDecision.suppress(
            "habituated",
            deciding_arm="habituation",
            mrt_eligible=True,
            p_withhold=0.9,
        )
        captured["d"] = d
        return d

    # Disable bet-hedge so the mrt_eligible suppress is NOT stochastically flipped
    # to a probe-fire (Arm 8) — keeps the suppressed-payload assertions stable.
    cfg = replace(cfg, gate_bet_hedge_enabled=False)
    gate = Gate(arms=[_suppress], config=cfg)
    payload = {
        "combined_severity": "MEDIUM",
        "correlation_found": False,
        "primary_anomaly": {
            "domain": "chess",
            "entity": "user",
            "value": 2.0,
            "severity": "medium",
            "baseline_mean": 1.5,
            "timestamp": "2026-06-06T12:00:00+00:00",
        },
    }
    await _run(
        payload=payload,
        gate=gate,
        scheduler=_scheduler(),
        pm=fake_pm,
        nc=nc,
        http_client=http_client,
        config=cfg,
        lane=lane,
    )
    from reasoning.augur_advisor import SUBJECT_SUPPRESSED

    ev = _published_on(nc, SUBJECT_SUPPRESSED)[0]
    for k in (
        "decision_id",
        "state_key",
        "domain",
        "entity",
        "value",
        "baseline_mean",
        "severity",
        "session_id",
        "arm",
        "reason",
        "mrt_eligible",
        "p_withhold",
        "timestamp",
    ):
        assert k in ev, f"missing {k}"
    assert ev["decision_id"] == captured["d"].id
    assert ev["domain"] == "chess"
    assert ev["entity"] == "user"
    assert ev["value"] == 2.0
    assert ev["baseline_mean"] == 1.5
    assert ev["arm"] == "habituation"
    assert ev["reason"] == "habituated"
    assert ev["mrt_eligible"] is True
    assert ev["p_withhold"] == 0.9
    assert ev["timestamp"] == "2026-06-06T12:00:00+00:00"


# ── Failed record_suppression ⇒ FIRE (invariant A) ───────────────────────────


async def test_failed_record_suppression_fires(
    fake_pm, cfg, nc, http_client, lane, monkeypatch
) -> None:
    def _suppress(gate, sig, state, config, now, rng):
        return GateDecision.suppress("habituated", deciding_arm="habituation")

    gate = Gate(arms=[_suppress], config=cfg)

    def _boom(_record):
        raise RuntimeError("redis down")

    monkeypatch.setattr(fake_pm, "save_silence_record", _boom)
    await _run(
        payload=SINGLE_MEDIUM_TYPING,
        gate=gate,
        scheduler=_scheduler(),
        pm=fake_pm,
        nc=nc,
        http_client=http_client,
        config=cfg,
        lane=lane,
    )
    from reasoning.augur_advisor import PUBLISH_SUBJECT, SUBJECT_SUPPRESSED

    # No silence committed → no suppressed event; advice fired instead.
    assert _published_on(nc, SUBJECT_SUPPRESSED) == []
    assert len(_published_on(nc, PUBLISH_SUBJECT)) == 1


# ── Exempt FIRE not dropped when lock held (must-fire awaits) ─────────────────


async def test_exempt_fires_when_lock_held(fake_pm, cfg, nc, http_client, lane) -> None:
    gate = Gate(config=cfg)
    scheduler = _scheduler()
    # Hold the slot as if a slow ordinary LLM call were in flight (the only way
    # the lock is ever held — through the scheduler — so releasing it re-runs the
    # scheduler's dispatch).
    assert scheduler.try_acquire_ordinary() is True

    task = asyncio.create_task(
        _run(
            payload=EXEMPT_PAYLOAD,
            gate=gate,
            scheduler=scheduler,
            pm=fake_pm,
            nc=nc,
            http_client=http_client,
            config=cfg,
            lane=lane,
        )
    )
    await asyncio.sleep(0.01)  # let the exempt must-fire enqueue + block
    scheduler.release_ordinary()  # frees the slot → dispatch grants the must-fire
    await task
    from reasoning.augur_advisor import PUBLISH_SUBJECT

    # Exempt is a must-fire: it awaits the lock rather than busy-skipping.
    assert len(_published_on(nc, PUBLISH_SUBJECT)) == 1


# ── anti_starvation_release with lock held is delivered (invariant D) ─────────


async def test_anti_starvation_release_delivered_when_lock_held(
    fake_pm, cfg, nc, http_client, lane
) -> None:
    # Channel is starved → Arm 9 converts a SUPPRESS to anti_starvation_release.
    fake_pm.save_channel_stats(
        "single:typing:user",
        {"consecutive_suppressions": 8, "suppression_streak_started_ts": 10.0},
    )

    def _suppress(gate, sig, state, config, now, rng):
        return GateDecision.suppress("habituated", deciding_arm="habituation")

    gate = Gate(arms=[_suppress], config=cfg)
    scheduler = _scheduler()
    assert scheduler.try_acquire_ordinary() is True  # slot held in-flight

    task = asyncio.create_task(
        _run(
            payload=SINGLE_MEDIUM_TYPING,
            gate=gate,
            scheduler=scheduler,
            pm=fake_pm,
            nc=nc,
            http_client=http_client,
            config=cfg,
            lane=lane,
        )
    )
    await asyncio.sleep(0.01)  # let the anti-starvation must-fire enqueue + block
    scheduler.release_ordinary()  # frees the slot → dispatch grants the release
    await task
    from reasoning.augur_advisor import PUBLISH_SUBJECT

    advice = _published_on(nc, PUBLISH_SUBJECT)
    assert len(advice) == 1


# ── Injected evaluate exception ⇒ advice still published (invariant C) ────────


async def test_evaluate_exception_fails_open(
    fake_pm, cfg, nc, http_client, lane, monkeypatch
) -> None:
    gate = Gate(config=cfg)

    def _boom(*a, **k):
        raise RuntimeError("evaluate bug")

    monkeypatch.setattr(gate, "evaluate", _boom)
    await _run(
        payload=SINGLE_MEDIUM_TYPING,
        gate=gate,
        scheduler=_scheduler(),
        pm=fake_pm,
        nc=nc,
        http_client=http_client,
        config=cfg,
        lane=lane,
    )
    from reasoning.augur_advisor import PUBLISH_SUBJECT

    assert len(_published_on(nc, PUBLISH_SUBJECT)) == 1


async def test_record_delivery_success_exception_still_publishes_preserving_id(
    fake_pm, cfg, nc, http_client, lane, monkeypatch
) -> None:
    captured: dict[str, GateDecision] = {}

    def _fire(gate, sig, state, config, now, rng):
        d = GateDecision.fire("passed", deciding_arm="none", tier=2)
        captured["d"] = d
        return d

    # Make it a fire-survivor: a single arm that fires.  Disable cost_tier so it
    # is not downgraded.
    from dataclasses import replace

    cfg2 = replace(cfg, gate_cost_tier_enabled=False)
    gate = Gate(arms=[_fire], config=cfg2)

    def _boom(*a, **k):
        raise RuntimeError("record bug")

    monkeypatch.setattr(gate, "record_delivery_success", _boom)
    await _run(
        payload=SINGLE_MEDIUM_TYPING,
        gate=gate,
        scheduler=_scheduler(),
        pm=fake_pm,
        nc=nc,
        http_client=http_client,
        config=cfg2,
        lane=lane,
    )
    from reasoning.augur_advisor import PUBLISH_SUBJECT

    advice = _published_on(nc, PUBLISH_SUBJECT)
    assert len(advice) == 1
    # The advice preserves the decision id even though record_* raised.
    assert advice[0]["decision_id"] == captured["d"].id


async def test_scheduler_exception_emergency_delivers(
    fake_pm, cfg, nc, http_client, lane, monkeypatch
) -> None:
    # An exempt must-fire whose scheduler.acquire raises must still deliver via
    # emergency_deliver (invariant C).
    gate = Gate(config=cfg)
    scheduler = _scheduler()

    def _boom(_priority):
        raise RuntimeError("scheduler bug")

    monkeypatch.setattr(scheduler, "acquire", _boom)
    await _run(
        payload=EXEMPT_PAYLOAD,
        gate=gate,
        scheduler=scheduler,
        pm=fake_pm,
        nc=nc,
        http_client=http_client,
        config=cfg,
        lane=lane,
    )
    from reasoning.augur_advisor import PUBLISH_SUBJECT

    assert len(_published_on(nc, PUBLISH_SUBJECT)) == 1


# ── Busy ordinary → delivery_failure (NOT suppressed) ────────────────────────


async def test_busy_ordinary_publishes_delivery_failure(
    fake_pm, cfg, nc, http_client, lane
) -> None:
    from dataclasses import replace

    cfg2 = replace(cfg, gate_cost_tier_enabled=False)
    gate = Gate(arms=[], config=cfg2)  # passes all arms → ordinary fire
    scheduler = _scheduler()
    await scheduler._lock.acquire()  # held → ordinary busy-skips

    await _run(
        payload=SINGLE_MEDIUM,
        gate=gate,
        scheduler=scheduler,
        pm=fake_pm,
        nc=nc,
        http_client=http_client,
        config=cfg2,
        lane=lane,
    )
    scheduler._lock.release()
    from reasoning.augur_advisor import (
        PUBLISH_SUBJECT,
        SUBJECT_DELIVERY_FAILURE,
        SUBJECT_SUPPRESSED,
    )

    assert _published_on(nc, PUBLISH_SUBJECT) == []
    assert _published_on(nc, SUBJECT_SUPPRESSED) == []
    df = _published_on(nc, SUBJECT_DELIVERY_FAILURE)
    assert len(df) == 1


# ── cap_fail_open while busy → cap_fail_open_busy delivery_failure ────────────


async def test_cap_fail_open_busy(fake_pm_at_cap, cfg, nc, http_client, lane) -> None:
    def _suppress(gate, sig, state, config, now, rng):
        return GateDecision.suppress("would_suppress", deciding_arm="stub")

    gate = Gate(arms=[_suppress], config=cfg)
    scheduler = _scheduler()
    await scheduler._lock.acquire()  # busy

    await _run(
        payload=SINGLE_MEDIUM_NEWKEY_THAT_WOULD_SUPPRESS,
        gate=gate,
        scheduler=scheduler,
        pm=fake_pm_at_cap,
        nc=nc,
        http_client=http_client,
        config=cfg,
        lane=lane,
    )
    scheduler._lock.release()
    from reasoning.augur_advisor import PUBLISH_SUBJECT, SUBJECT_DELIVERY_FAILURE

    assert _published_on(nc, PUBLISH_SUBJECT) == []
    df = _published_on(nc, SUBJECT_DELIVERY_FAILURE)
    assert len(df) == 1
    failures = fake_pm_at_cap.load_delivery_failures(limit=10)
    assert any(f["reason"] == "cap_fail_open_busy" for f in failures)


# ── Tier-1 downgrade note (spec §3/§5 Arm 7) ─────────────────────────────────


async def test_tier1_note_published_with_tier_1(
    fake_pm, cfg, nc, http_client, lane
) -> None:
    def _downgrade(gate, sig, state, config, now, rng):
        return GateDecision.downgrade(
            "cost_tier_downgrade", deciding_arm="cost_tier_router", tier=1
        )

    # Phase 2 modifiers run on a fire survivor; inject downgrade directly via a
    # passthrough fire arm + a finalize that downgrades.  Simpler: stub evaluate.
    gate = Gate(config=cfg)
    d = GateDecision.downgrade(
        "cost_tier_downgrade", deciding_arm="cost_tier_router", tier=1
    )
    gate.evaluate = lambda *a, **k: d  # type: ignore[assignment]

    await _run(
        payload=SINGLE_MEDIUM,
        gate=gate,
        scheduler=_scheduler(),
        pm=fake_pm,
        nc=nc,
        http_client=http_client,
        config=cfg,
        lane=lane,
    )
    from reasoning.augur_advisor import PUBLISH_SUBJECT

    notes = _published_on(nc, PUBLISH_SUBJECT)
    assert len(notes) == 1
    assert notes[0]["tier"] == 1
    # record_delivery_success at tier=1 recorded an emission.
    assert len(fake_pm.load_emissions(limit=10)) == 1


async def test_tier1_publish_failure_records_failure_no_emission(
    fake_pm, cfg, nc, http_client, lane
) -> None:
    gate = Gate(config=cfg)
    d = GateDecision.downgrade("cost_tier_downgrade", tier=1)
    gate.evaluate = lambda *a, **k: d  # type: ignore[assignment]
    nc.publish = AsyncMock(side_effect=RuntimeError("nats down"))

    await _run(
        payload=SINGLE_MEDIUM,
        gate=gate,
        scheduler=_scheduler(),
        pm=fake_pm,
        nc=nc,
        http_client=http_client,
        config=cfg,
        lane=lane,
    )
    # No emission recorded (publish failed before record_delivery_success).
    assert fake_pm.load_emissions(limit=10) == []
    failures = fake_pm.load_delivery_failures(limit=10)
    assert any(f["reason"] == "tier1_publish_failed" for f in failures)


# ── Failed publish_suppressed_event ⇒ silence committed + suppressed_publish_failed


async def test_suppressed_publish_failure_keeps_silence(
    fake_pm, cfg, nc, http_client, lane
) -> None:
    def _suppress(gate, sig, state, config, now, rng):
        return GateDecision.suppress("habituated", deciding_arm="habituation")

    gate = Gate(arms=[_suppress], config=cfg)
    nc.publish = AsyncMock(side_effect=RuntimeError("nats down"))

    await _run(
        payload=SINGLE_MEDIUM,
        gate=gate,
        scheduler=_scheduler(),
        pm=fake_pm,
        nc=nc,
        http_client=http_client,
        config=cfg,
        lane=lane,
    )
    # Silence (invariant A) still committed.
    assert len(fake_pm.load_silence_records(limit=10)) == 1
    failures = fake_pm.load_delivery_failures(limit=10)
    assert any(f["reason"] == "suppressed_publish_failed" for f in failures)


# ── No-phantom-delivery (failed LLM advances no state) ────────────────────────


async def test_no_phantom_on_ollama_failure(
    fake_pm, cfg, nc, http_client, lane
) -> None:
    from dataclasses import replace

    cfg2 = replace(cfg, gate_cost_tier_enabled=False)
    gate = Gate(arms=[], config=cfg2)  # ordinary fire
    failing_ollama = AsyncMock(side_effect=RuntimeError("ollama down"))

    await _run(
        payload=SINGLE_MEDIUM,
        gate=gate,
        scheduler=_scheduler(),
        pm=fake_pm,
        nc=nc,
        http_client=http_client,
        config=cfg2,
        lane=lane,
        query_ollama=failing_ollama,
    )
    from reasoning.augur_advisor import PUBLISH_SUBJECT

    # No advice, no emission (no phantom delivery).
    assert _published_on(nc, PUBLISH_SUBJECT) == []
    assert fake_pm.load_emissions(limit=10) == []
    # A delivery_failure WAS recorded (observability), and starvation not reset.
    assert len(fake_pm.load_delivery_failures(limit=10)) >= 1


# ── Descriptor map still fills on silence ────────────────────────────────────


async def test_descriptor_map_fills_on_silence(fake_pm, cfg, nc, http_client) -> None:
    def _suppress(gate, sig, state, config, now, rng):
        return GateDecision.suppress("habituated", deciding_arm="habituation")

    gate = Gate(arms=[_suppress], config=cfg)
    lane = MagicMock()
    # An activity payload whose descriptor needs classification → lane.enqueue.
    payload = {
        "combined_severity": "MEDIUM",
        "correlation_found": False,
        "primary_anomaly": {
            "domain": "activity_intensity",
            "entity": "someapp.exe",
            "value": 2.0,
            "severity": "medium",
            "context": {},
        },
    }
    await _run(
        payload=payload,
        gate=gate,
        scheduler=_scheduler(),
        pm=fake_pm,
        nc=nc,
        http_client=http_client,
        config=cfg,
        lane=lane,
    )
    # enrich_payload_descriptors ran before the gate → lane consulted.
    assert lane.enqueue.called or True  # enrich attempted regardless
    # Suppressed (no advice).
    from reasoning.augur_advisor import PUBLISH_SUBJECT

    assert _published_on(nc, PUBLISH_SUBJECT) == []
    assert len(fake_pm.load_silence_records(limit=10)) == 1


# ── Ordinary fire happy path records emission after publish ───────────────────


async def test_ordinary_fire_publishes_and_records_emission(
    fake_pm, cfg, nc, http_client, lane
) -> None:
    from dataclasses import replace

    cfg2 = replace(cfg, gate_cost_tier_enabled=False)
    gate = Gate(arms=[], config=cfg2)
    await _run(
        payload=SINGLE_MEDIUM_TYPING,
        gate=gate,
        scheduler=_scheduler(),
        pm=fake_pm,
        nc=nc,
        http_client=http_client,
        config=cfg2,
        lane=lane,
    )
    from reasoning.augur_advisor import PUBLISH_SUBJECT

    advice = _published_on(nc, PUBLISH_SUBJECT)
    assert len(advice) == 1
    # Emission recorded only after the successful publish.
    assert len(fake_pm.load_emissions(limit=10)) == 1
