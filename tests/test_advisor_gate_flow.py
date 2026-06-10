"""Integration-level tests for the gate-driven on_message control flow (spec §3).

These exercise ``consilium.advisor.process_message`` — the per-message
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
from consilium.advisor import process_message
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
    from consilium.advisor import PUBLISH_SUBJECT, SUBJECT_SUPPRESSED

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
    from consilium.advisor import SUBJECT_SUPPRESSED

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
    from consilium.advisor import PUBLISH_SUBJECT, SUBJECT_SUPPRESSED

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
    from consilium.advisor import PUBLISH_SUBJECT

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
    from consilium.advisor import PUBLISH_SUBJECT

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
    from consilium.advisor import PUBLISH_SUBJECT

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
    from consilium.advisor import PUBLISH_SUBJECT

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
    from consilium.advisor import PUBLISH_SUBJECT

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
    from consilium.advisor import (
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
    from consilium.advisor import PUBLISH_SUBJECT, SUBJECT_DELIVERY_FAILURE

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
    from consilium.advisor import PUBLISH_SUBJECT

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
    from consilium.advisor import PUBLISH_SUBJECT

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
    # enrich_payload_descriptors ran before the gate → the unclassified entity
    # (empty descriptor map, no app_identity) was enqueued for classification.
    lane.enqueue.assert_called_once_with("someapp.exe")
    # Suppressed (no advice).
    from consilium.advisor import PUBLISH_SUBJECT

    assert _published_on(nc, PUBLISH_SUBJECT) == []
    assert len(fake_pm.load_silence_records(limit=10)) == 1


# ── record_busy_skip raises (lock held) ⇒ fail-open must-fire (invariant C) ───


async def test_record_busy_skip_exception_fails_open_and_fires(
    fake_pm, cfg, nc, http_client, lane, monkeypatch
) -> None:
    # An ordinary fire is busy-skipped because the lock is held; if
    # record_busy_skip itself *raises*, the code must convert to
    # gate_error_fail_open and deliver via the scheduler (never a silent drop).
    from dataclasses import replace

    cfg2 = replace(cfg, gate_cost_tier_enabled=False)
    gate = Gate(arms=[], config=cfg2)  # passes all arms → ordinary fire

    def _boom(*a, **k):
        raise RuntimeError("record_busy_skip bug")

    monkeypatch.setattr(gate, "record_busy_skip", _boom)
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
            config=cfg2,
            lane=lane,
        )
    )
    await asyncio.sleep(0.01)  # let the fail-open must-fire enqueue + block
    scheduler.release_ordinary()  # frees the slot → dispatch grants the fail-open
    await task
    from consilium.advisor import PUBLISH_SUBJECT, SUBJECT_SUPPRESSED

    # The dropped ordinary fire is delivered (fail open), never silenced.
    assert len(_published_on(nc, PUBLISH_SUBJECT)) == 1
    assert _published_on(nc, SUBJECT_SUPPRESSED) == []


# ── record_suppression itself raises ⇒ FIRE (invariant A try/except) ──────────


async def test_record_suppression_raises_fires(
    fake_pm, cfg, nc, http_client, lane, monkeypatch
) -> None:
    # save_silence_record-raising is caught *inside* record_suppression (returns
    # False); to exercise process_message's own try/except around the call we
    # make record_suppression itself raise.
    def _suppress(gate, sig, state, config, now, rng):
        return GateDecision.suppress("habituated", deciding_arm="habituation")

    gate = Gate(arms=[_suppress], config=cfg)

    def _boom(*a, **k):
        raise RuntimeError("record_suppression bug")

    monkeypatch.setattr(gate, "record_suppression", _boom)
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
    from consilium.advisor import PUBLISH_SUBJECT, SUBJECT_SUPPRESSED

    # No suppressed event; advice fired instead (fail open at the call site).
    assert _published_on(nc, SUBJECT_SUPPRESSED) == []
    assert len(_published_on(nc, PUBLISH_SUBJECT)) == 1


# ── Distinct Ollama failure reason labels (ollama_unreachable / _timeout) ──────


async def test_ollama_connect_error_records_unreachable(
    fake_pm, cfg, nc, http_client, lane
) -> None:
    import httpx

    from dataclasses import replace

    cfg2 = replace(cfg, gate_cost_tier_enabled=False)
    gate = Gate(arms=[], config=cfg2)  # ordinary fire
    failing = AsyncMock(side_effect=httpx.ConnectError("refused"))

    await _run(
        payload=SINGLE_MEDIUM_TYPING,
        gate=gate,
        scheduler=_scheduler(),
        pm=fake_pm,
        nc=nc,
        http_client=http_client,
        config=cfg2,
        lane=lane,
        query_ollama=failing,
    )
    from consilium.advisor import PUBLISH_SUBJECT

    assert _published_on(nc, PUBLISH_SUBJECT) == []
    failures = fake_pm.load_delivery_failures(limit=10)
    assert any(f["reason"] == "ollama_unreachable" for f in failures)


async def test_ollama_timeout_records_timeout(
    fake_pm, cfg, nc, http_client, lane
) -> None:
    import httpx

    from dataclasses import replace

    cfg2 = replace(cfg, gate_cost_tier_enabled=False)
    gate = Gate(arms=[], config=cfg2)  # ordinary fire
    failing = AsyncMock(side_effect=httpx.TimeoutException("slow"))

    await _run(
        payload=SINGLE_MEDIUM_TYPING,
        gate=gate,
        scheduler=_scheduler(),
        pm=fake_pm,
        nc=nc,
        http_client=http_client,
        config=cfg2,
        lane=lane,
        query_ollama=failing,
    )
    from consilium.advisor import PUBLISH_SUBJECT

    assert _published_on(nc, PUBLISH_SUBJECT) == []
    failures = fake_pm.load_delivery_failures(limit=10)
    assert any(f["reason"] == "ollama_timeout" for f in failures)


# ── Anti-starvation coalesce + under-lock re-check (concurrency branches) ──────


async def test_anti_starvation_coalesced_when_in_flight(
    fake_pm, cfg, nc, http_client, lane
) -> None:
    # A release whose state_key is already in-flight returns immediately
    # (coalesce: one in-flight anti-starvation release per channel).
    fake_pm.save_channel_stats(
        "single:typing:user",
        {"consecutive_suppressions": 8, "suppression_streak_started_ts": 10.0},
    )

    def _suppress(gate, sig, state, config, now, rng):
        return GateDecision.suppress("habituated", deciding_arm="habituation")

    gate = Gate(arms=[_suppress], config=cfg)
    scheduler = _scheduler()
    # Pre-arm the in-flight token for this channel → the new release coalesces.
    scheduler._release_in_flight.add("single:typing:user")

    await _run(
        payload=SINGLE_MEDIUM_TYPING,
        gate=gate,
        scheduler=scheduler,
        pm=fake_pm,
        nc=nc,
        http_client=http_client,
        config=cfg,
        lane=lane,
    )
    from consilium.advisor import PUBLISH_SUBJECT

    # Coalesced → nothing published, no emission.
    assert _published_on(nc, PUBLISH_SUBJECT) == []
    assert fake_pm.load_emissions(limit=10) == []


async def test_anti_starvation_recheck_skips_when_not_starved(
    fake_pm, cfg, nc, http_client, lane, monkeypatch
) -> None:
    # After acquiring the lock the release re-checks still_starved; if a
    # concurrent delivery already served the channel (still_starved False), it
    # skips the redundant release.
    fake_pm.save_channel_stats(
        "single:typing:user",
        {"consecutive_suppressions": 8, "suppression_streak_started_ts": 10.0},
    )

    def _suppress(gate, sig, state, config, now, rng):
        return GateDecision.suppress("habituated", deciding_arm="habituation")

    gate = Gate(arms=[_suppress], config=cfg)
    monkeypatch.setattr(gate, "still_starved", lambda *a, **k: False)

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
    from consilium.advisor import PUBLISH_SUBJECT

    # Re-check returned not-starved → skipped, no advice, no emission.
    assert _published_on(nc, PUBLISH_SUBJECT) == []
    assert fake_pm.load_emissions(limit=10) == []


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
    from consilium.advisor import PUBLISH_SUBJECT

    advice = _published_on(nc, PUBLISH_SUBJECT)
    assert len(advice) == 1
    # Emission recorded only after the successful publish.
    assert len(fake_pm.load_emissions(limit=10)) == 1
