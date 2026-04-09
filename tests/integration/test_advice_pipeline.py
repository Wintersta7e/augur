"""Integration tests: anomalies produce LLM advice (requires Ollama)."""

from __future__ import annotations

import asyncio
import json

import pytest

from tests.integration.conftest import (
    inject_perception_event,
    requires_ollama,
    wait_for_redis_key,
)


@requires_ollama
@pytest.mark.parametrize(
    "pipeline", [["detector", "correlator", "advisor"]], indirect=True
)
class TestAdvicePipeline:
    """Verify that anomalies trigger advice generation via the LLM advisor."""

    @pytest.mark.asyncio
    async def test_high_anomaly_produces_advice(
        self, pipeline, redis_client, nats_conn
    ) -> None:
        """A high-value outlier causes the advisor to write advice to Redis."""
        sid = "advice-test"
        for _ in range(10):
            await inject_perception_event(
                nats_conn,
                domain="advicetest",
                entity="subject",
                event_type="move",
                value=5.0,
                unit="seconds",
                context={},
                session_id=sid,
            )
            await asyncio.sleep(0.05)
        await asyncio.sleep(1.0)

        redis_client.delete("augur:reasoning:last_advice")

        await inject_perception_event(
            nats_conn,
            domain="advicetest",
            entity="subject",
            event_type="move",
            value=200.0,
            unit="seconds",
            context={},
            session_id=sid,
        )

        found = await wait_for_redis_key(
            redis_client, "augur:reasoning:last_advice", timeout=130.0
        )
        assert found, "No advice generated within timeout"

        raw = redis_client.get("augur:reasoning:last_advice")
        assert raw is not None
        advice = json.loads(raw)
        assert "text" in advice or "advice" in advice

    @pytest.mark.asyncio
    async def test_advice_contains_domain_context(
        self, pipeline, redis_client, nats_conn
    ) -> None:
        """Advice generated for a chess anomaly carries the chess domain tag."""
        sid = "context-test"
        for _ in range(10):
            await inject_perception_event(
                nats_conn,
                domain="chess",
                entity="white",
                event_type="move",
                value=3.0,
                unit="seconds",
                context={"move_san": "e4", "move_number": 1},
                session_id=sid,
            )
            await asyncio.sleep(0.05)
        await asyncio.sleep(1.0)

        redis_client.delete("augur:reasoning:last_advice")

        await inject_perception_event(
            nats_conn,
            domain="chess",
            entity="white",
            event_type="move",
            value=120.0,
            unit="seconds",
            context={"move_san": "Nf3", "move_number": 11},
            session_id=sid,
        )

        found = await wait_for_redis_key(
            redis_client, "augur:reasoning:last_advice", timeout=130.0
        )
        assert found, "No advice generated within timeout"

        raw = redis_client.get("augur:reasoning:last_advice")
        assert raw is not None
        advice = json.loads(raw)
        assert advice.get("domain") == "chess"
