"""Integration tests: full session reflection cycle (requires Ollama)."""

from __future__ import annotations

import asyncio
import json

import pytest

from tabula.session import SessionManager
from tests.integration.conftest import (
    inject_perception_event,
    requires_ollama,
    wait_for_redis_key,
)


@requires_ollama
@pytest.mark.parametrize(
    "pipeline",
    [["vigil", "consilium", "responsum", "disciplina"]],
    indirect=True,
)
class TestReflection:
    """Verify end-to-end session reflection after an anomaly + advice cycle."""

    @pytest.mark.asyncio
    async def test_full_session_reflection_cycle(
        self, pipeline, redis_client, nats_conn
    ) -> None:
        """A complete session ending triggers a reflection report in Redis."""
        sm = SessionManager(redis_client)
        session_id = sm.start()

        await nats_conn.publish(
            "augur.session.start",
            json.dumps({"session_id": session_id}).encode(),
        )

        for _ in range(10):
            await inject_perception_event(
                nats_conn,
                domain="refltest",
                entity="subject",
                event_type="move",
                value=5.0,
                unit="seconds",
                context={},
                session_id=session_id,
            )
            await asyncio.sleep(0.05)
        await asyncio.sleep(1.5)

        # Inject outlier to trigger anomaly + advice
        await inject_perception_event(
            nats_conn,
            domain="refltest",
            entity="subject",
            event_type="move",
            value=100.0,
            unit="seconds",
            context={},
            session_id=session_id,
        )
        await asyncio.sleep(15.0)

        sm.end()
        await nats_conn.publish(
            "augur.session.end",
            json.dumps({"session_id": session_id}).encode(),
        )
        await nats_conn.publish(
            "augur.disciplina.trigger",
            json.dumps({"session_id": session_id}).encode(),
        )

        found = await wait_for_redis_key(
            redis_client, f"augur:disciplina:{session_id}", timeout=60.0
        )
        assert found, "No reflection report generated within timeout"

        raw = redis_client.get(f"augur:disciplina:{session_id}")
        assert raw is not None
        report = json.loads(raw)
        assert "session_id" in report
