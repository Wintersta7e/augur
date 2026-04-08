"""Integration tests: outlier values trigger anomaly detection."""

from __future__ import annotations

import asyncio
import json

import pytest

from tests.integration.conftest import (
    inject_perception_event,
    wait_for_redis_key,
)


@pytest.mark.parametrize("pipeline", [["detector"]], indirect=True)
class TestAnomalyPipeline:
    """Verify outlier detection via the live detector subprocess."""

    @pytest.mark.asyncio
    async def test_outlier_triggers_anomaly(
        self, pipeline, redis_client, nats_conn
    ) -> None:
        """After building a baseline, an extreme outlier triggers anomaly detection."""
        sid = "anomaly-test"
        # Use varied values to build a baseline with non-zero variance
        # (identical values → std≈0 → deviation forced to 0 by detector)
        baseline_values = [4.5, 5.2, 4.8, 5.5, 4.3, 5.1, 4.7, 5.3, 4.9, 5.0]
        for val in baseline_values:
            await inject_perception_event(
                nats_conn,
                domain="anomtest",
                entity="subject",
                event_type="move",
                value=val,
                unit="seconds",
                context={},
                session_id=sid,
            )
            await asyncio.sleep(0.05)
        await asyncio.sleep(1.0)

        # Inject extreme outlier (10x baseline mean)
        await inject_perception_event(
            nats_conn,
            domain="anomtest",
            entity="subject",
            event_type="move",
            value=50.0,
            unit="seconds",
            context={},
            session_id=sid,
        )

        found = await wait_for_redis_key(
            redis_client, "augur:detection:last_anomaly", timeout=10.0
        )
        assert found, "No anomaly detected within timeout"

        raw = redis_client.get("augur:detection:last_anomaly")
        assert raw is not None
        anomaly = json.loads(raw)
        assert anomaly["domain"] == "anomtest"
        assert anomaly["entity"] == "subject"
        assert anomaly["severity"] in ("low", "medium", "high")

    @pytest.mark.asyncio
    async def test_normal_value_no_anomaly(
        self, pipeline, redis_client, nats_conn
    ) -> None:
        """Normal values after baseline should not trigger anomaly."""
        sid = "no-anomaly-test"
        for _ in range(10):
            await inject_perception_event(
                nats_conn,
                domain="normtest",
                entity="subject",
                event_type="move",
                value=5.0,
                unit="seconds",
                context={},
                session_id=sid,
            )
            await asyncio.sleep(0.05)
        await asyncio.sleep(1.0)

        redis_client.delete("augur:detection:last_anomaly")

        await inject_perception_event(
            nats_conn,
            domain="normtest",
            entity="subject",
            event_type="move",
            value=5.1,
            unit="seconds",
            context={},
            session_id=sid,
        )
        await asyncio.sleep(2.0)

        raw = redis_client.get("augur:detection:last_anomaly")
        if raw is not None:
            anomaly = json.loads(raw)
            # If something was written, it must not be from our normtest/subject
            assert (
                anomaly.get("domain") != "normtest"
                or anomaly.get("entity") != "subject"
            )

    @pytest.mark.asyncio
    async def test_anomaly_contains_required_fields(
        self, pipeline, redis_client, nats_conn
    ) -> None:
        """Anomaly events contain all expected fields."""
        sid = "fields-test"
        baseline_values = [2.8, 3.2, 3.0, 3.5, 2.7, 3.1, 2.9, 3.3, 3.0, 3.4]
        for val in baseline_values:
            await inject_perception_event(
                nats_conn,
                domain="fieldtest",
                entity="player",
                event_type="move",
                value=val,
                unit="seconds",
                context={},
                session_id=sid,
            )
            await asyncio.sleep(0.05)
        await asyncio.sleep(1.0)

        redis_client.delete("augur:detection:last_anomaly")

        await inject_perception_event(
            nats_conn,
            domain="fieldtest",
            entity="player",
            event_type="move",
            value=100.0,
            unit="seconds",
            context={},
            session_id=sid,
        )

        found = await wait_for_redis_key(
            redis_client, "augur:detection:last_anomaly", timeout=10.0
        )
        assert found, "No anomaly detected within timeout"

        raw = redis_client.get("augur:detection:last_anomaly")
        assert raw is not None
        anomaly = json.loads(raw)
        if anomaly.get("domain") == "fieldtest":
            required = {"domain", "entity", "severity", "value", "deviation_score"}
            assert required.issubset(anomaly.keys())
