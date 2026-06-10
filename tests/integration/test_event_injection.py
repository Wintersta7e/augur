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
            redis_client, "augur:vigil:profile:inttest:player1"
        )
        assert found, "Baseline key not created within timeout"

        raw = redis_client.get("augur:vigil:profile:inttest:player1")
        assert raw is not None
        state = json.loads(raw)
        assert state["observation_count"] == 1
        assert state["ewma_mean"] == 5.0

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
            redis_client, "augur:vigil:profile:inttest:player2"
        )
        assert found, "Baseline key not created within timeout"

        raw = redis_client.get("augur:vigil:profile:inttest:player2")
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
            redis_client, "augur:vigil:profile:newdomain:entity1"
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
        await asyncio.sleep(1.0)

        raw_list = redis_client.lrange("augur:vigil:history:inttest", 0, -1)
        assert len(raw_list) >= 1
        event_data = json.loads(raw_list[0])
        assert event_data["domain"] == "inttest"
        assert event_data["value"] == 7.0
