"""Real Redis/NATS: praefectus boots, heartbeats, and surfaces a snapshot."""

import asyncio
import json

import pytest

pytestmark = pytest.mark.integration


@pytest.mark.parametrize("pipeline", [["praefectus", "nexus"]], indirect=True)
@pytest.mark.asyncio
async def test_praefectus_snapshot_and_heartbeat(pipeline, redis_client):
    # Poll until praefectus + nexus are confirmed ALIVE (proves heartbeat receipt,
    # not just that a snapshot exists). redis_client is a sync client (conftest).
    deadline = 25.0
    waited = 0.0
    snap = None
    while waited < deadline:
        raw = redis_client.get("augur:praefectus:health")
        if raw is not None:
            snap = json.loads(raw)
            facs = snap.get("faculties", {})
            p = facs.get("praefectus", {})
            n = facs.get("nexus", {})
            if p.get("liveness") == "alive" and n.get("liveness") == "alive":
                break
        await asyncio.sleep(1.0)
        waited += 1.0

    assert snap is not None, "praefectus wrote no health snapshot"
    assert snap["faculties"]["praefectus"]["liveness"] == "alive"
    assert snap["faculties"]["praefectus"]["last_heartbeat"] is not None
    assert snap["faculties"]["nexus"]["liveness"] == "alive"
    assert snap["faculties"]["nexus"]["last_heartbeat"] is not None
