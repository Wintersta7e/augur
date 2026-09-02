"""Integration tests: perception events create and update Redis baselines."""

from __future__ import annotations

import asyncio
import json

import pytest

from tests.integration.conftest import (
    inject_perception_event,
    wait_for_redis_pattern,
)


@pytest.mark.parametrize("pipeline", [["vigil"]], indirect=True)
class TestEventInjection:
    """Verify that injected PerceptionEvents are processed by the detector."""

    @pytest.mark.asyncio
    async def test_inject_creates_baseline(
        self, pipeline, redis_client, nats_conn
    ) -> None:
        """Single event creates a baseline entry in Redis."""
        await inject_perception_event(
            nats_conn,
            domain="inttest",
            entity="player1",
            event_type="move",
            value=5.0,
            unit="seconds",
            context={},
            session_id="inject-test-1",
        )

        found = await wait_for_redis_pattern(
            redis_client, "augur:vigil:profile:inttest:move:player1"
        )
        assert found, "Baseline key not created within timeout"

        raw = redis_client.get("augur:vigil:profile:inttest:move:player1")
        assert raw is not None
        state = json.loads(raw)
        assert state["observation_count"] == 1
        assert state["ewma_mean"] == 5.0

    @pytest.mark.asyncio
    async def test_two_event_types_on_one_entity_stay_separate(
        self, pipeline, redis_client, nats_conn
    ) -> None:
        """Two streams on one entity must not share a baseline.

        The live typing sensor publishes ``sample`` in ms (~170) and ``pause``
        in seconds (~7) for the same ``user`` entity. When baselines were keyed
        on (domain, entity) alone, both folded into one EWMA: the stored mean
        oscillated between the two scales and every switch between the streams
        was reported as a multi-sigma anomaly. This drives the REAL detector,
        not the key helper, so a regression in either the keying or the sensor
        contract fails here.
        """
        for _ in range(3):
            await inject_perception_event(
                nats_conn,
                domain="inttest_mix",
                entity="user",
                event_type="sample",
                value=170.0,
                unit="ms",
                context={},
                session_id="mixed-units-1",
            )
        for _ in range(3):
            await inject_perception_event(
                nats_conn,
                domain="inttest_mix",
                entity="user",
                event_type="pause",
                value=7.0,
                unit="seconds",
                context={},
                session_id="mixed-units-1",
            )

        sample_key = "augur:vigil:profile:inttest_mix:sample:user"
        pause_key = "augur:vigil:profile:inttest_mix:pause:user"
        assert await wait_for_redis_pattern(redis_client, sample_key)
        assert await wait_for_redis_pattern(redis_client, pause_key)

        sample = json.loads(redis_client.get(sample_key))
        pause = json.loads(redis_client.get(pause_key))
        # Each series keeps its own scale — neither mean was dragged toward the
        # other, which is exactly what the shared baseline used to do.
        assert sample["ewma_mean"] == pytest.approx(170.0)
        assert pause["ewma_mean"] == pytest.approx(7.0)
        assert sample["unit"] == "ms"
        assert pause["unit"] == "seconds"
        # And the pre-fix key must not exist at all.
        assert redis_client.get("augur:vigil:profile:inttest_mix:user") is None

    @pytest.mark.asyncio
    async def test_unit_change_within_a_series_is_refused(
        self, pipeline, redis_client, nats_conn
    ) -> None:
        """A series has one unit; a sensor changing it must not corrupt the EWMA."""
        for _ in range(3):
            await inject_perception_event(
                nats_conn,
                domain="inttest_unit",
                entity="probe",
                event_type="sample",
                value=200.0,
                unit="ms",
                context={},
                session_id="unit-guard-1",
            )
        key = "augur:vigil:profile:inttest_unit:sample:probe"
        assert await wait_for_redis_pattern(redis_client, key)
        before = json.loads(redis_client.get(key))

        # Same series, different unit and scale — must be refused outright.
        await inject_perception_event(
            nats_conn,
            domain="inttest_unit",
            entity="probe",
            event_type="sample",
            value=3.0,
            unit="seconds",
            context={},
            session_id="unit-guard-1",
        )
        await asyncio.sleep(2.0)
        after = json.loads(redis_client.get(key))
        assert after == before, "a mismatched-unit event must not touch the baseline"

    @pytest.mark.asyncio
    async def test_multiple_events_update_baseline(
        self, pipeline, redis_client, nats_conn
    ) -> None:
        """Multiple events update the EWMA baseline."""
        for _ in range(5):
            await inject_perception_event(
                nats_conn,
                domain="inttest",
                entity="player2",
                event_type="move",
                value=5.0,
                unit="seconds",
                context={},
                session_id="inject-test-2",
            )
            await asyncio.sleep(0.1)

        found = await wait_for_redis_pattern(
            redis_client, "augur:vigil:profile:inttest:move:player2"
        )
        assert found, "Baseline key not created within timeout"

        raw = redis_client.get("augur:vigil:profile:inttest:move:player2")
        assert raw is not None
        state = json.loads(raw)
        assert state["observation_count"] == 5
        assert abs(state["ewma_mean"] - 5.0) < 0.01

    @pytest.mark.asyncio
    async def test_new_domain_creates_baseline(
        self, pipeline, redis_client, nats_conn
    ) -> None:
        """Events from unknown domains create baselines without errors."""
        await inject_perception_event(
            nats_conn,
            domain="newdomain",
            entity="entity1",
            event_type="event",
            value=42.0,
            unit="units",
            context={},
            session_id="inject-test-3",
        )

        found = await wait_for_redis_pattern(
            redis_client, "augur:vigil:profile:newdomain:event:entity1"
        )
        assert found, "Baseline key for new domain not created within timeout"

    @pytest.mark.asyncio
    async def test_event_persisted_to_history(
        self, pipeline, redis_client, nats_conn
    ) -> None:
        """Injected events appear in domain history."""
        await inject_perception_event(
            nats_conn,
            domain="inttest",
            entity="player3",
            event_type="move",
            value=7.0,
            unit="seconds",
            context={},
            session_id="inject-test-4",
        )

        # Poll rather than sleep-and-index: the detector is a subprocess on a
        # slow mount, and other events share this domain's history list, so a
        # fixed wait plus raw_list[0] is a race on both timing and ordering.
        def _find() -> dict | None:
            for raw in redis_client.lrange("augur:vigil:history:inttest", 0, -1):
                event_data = json.loads(raw)
                if event_data.get("value") == 7.0:
                    return event_data
            return None

        found = None
        for _ in range(40):
            found = _find()
            if found is not None:
                break
            await asyncio.sleep(0.25)
        assert found is not None, "injected event never reached domain history"
        assert found["domain"] == "inttest"
        assert found["entity"] == "player3"


@pytest.mark.parametrize(
    "pipeline",
    [{"components": ["vigil"], "env": {"AUGUR_BASELINE_ENTITY_IDLE_EVICT_S": "2"}}],
    indirect=True,
)
@pytest.mark.asyncio
async def test_idle_eviction_does_not_reset_a_trained_baseline(
    pipeline, redis_client, nats_conn
) -> None:
    """A series idle longer than the eviction TTL must keep its observations.

    Eviction drops the in-memory model and leaves the Redis profile alone, but
    the detector used to rebuild a fresh EntityBaseline on the cache miss and
    persist it unconditionally — so the first re-sighted event overwrote a
    trained profile with a one-observation EWMA. Any series whose inter-event
    gap exceeded the TTL could therefore never accumulate, however long the
    process ran; on the real activity domains most apps are that shape.

    Runs against the REAL detector subprocess with a 2s TTL (the sweep cadence
    is min(60, ttl)), because the two previous baseline defects both lived in
    wiring that unit tests did not execute.
    """
    key = "augur:vigil:profile:inttest_evict:sample:probe"
    for i in range(4):
        await inject_perception_event(
            nats_conn,
            domain="inttest_evict",
            entity="probe",
            event_type="sample",
            value=100.0 + i,
            unit="ms",
            context={},
            session_id="evict-1",
        )
    assert await wait_for_redis_pattern(redis_client, key)

    def _obs() -> int:
        raw = redis_client.get(key)
        return json.loads(raw)["observation_count"] if raw else 0

    for _ in range(40):
        if _obs() >= 4:
            break
        await asyncio.sleep(0.25)
    assert _obs() == 4, "detector did not record all four warm-up events"

    # Idle past the TTL so the sweep evicts the in-memory model.
    await asyncio.sleep(6.0)

    await inject_perception_event(
        nats_conn,
        domain="inttest_evict",
        entity="probe",
        event_type="sample",
        value=104.0,
        unit="ms",
        context={},
        session_id="evict-1",
    )
    for _ in range(40):
        if _obs() != 4:
            break
        await asyncio.sleep(0.25)
    assert _obs() == 5, (
        f"re-sighted event reset the durable profile to {_obs()} observations "
        "instead of extending it"
    )
