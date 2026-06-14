import asyncio
import json
import pytest
from tests.integration.conftest import inject_perception_event, wait_for_redis_key


@pytest.mark.parametrize("pipeline", [["vigil", "imperator"]], indirect=True)
@pytest.mark.asyncio
async def test_imperator_publishes_and_persists(pipeline, nats_conn, redis_client):
    seen = {}

    async def cap(msg):
        seen[msg.subject] = json.loads(msg.data.decode())

    await nats_conn.subscribe("augur.imperator.*", cb=cap)

    # Varied activity_intensity so Vigil has signal (zero-variance gotcha).
    for v in (10.0, 95.0, 12.0, 98.0):
        await inject_perception_event(
            nats_conn,
            domain="activity_intensity",
            entity="ide",
            event_type="activity_intensity",
            value=v,
            unit="ipm",
            context={"focused_app": "ide", "idle_seconds": 0.0},
            session_id="itest",
        )
        await asyncio.sleep(0.3)

    assert await wait_for_redis_key(
        redis_client, "augur:imperator:auspices", timeout=15.0
    )
    assert redis_client.get("augur:imperator:self_model") is not None
    await asyncio.sleep(1)
    assert "augur.imperator.auspices" in seen
