"""Integration test: Imperator II self-improvement cycle (requires Ollama)."""

from __future__ import annotations

import json

import pytest

from tests.integration.conftest import (
    requires_ollama,
    wait_for_redis_key,
)


@requires_ollama
@pytest.mark.parametrize("pipeline", [["imperator_ii"]], indirect=True)
@pytest.mark.asyncio
async def test_imperator_ii_emits_proposal(pipeline, nats_conn, redis_client):
    """Disciplina completion triggers Imperator II to emit a self-improvement proposal."""
    # Pre-seed a self-model that is already fresh relative to the (past) reflection
    # timestamp we will publish — so the freshness gate passes immediately.
    redis_client.set(
        "augur:imperator:self_model",
        json.dumps(
            {
                "schema_version": 1,
                "generated_at": 4102444800.0,  # year 2100 — always newer than past ts
                "session_id": "itest",
                "blind_spots": {
                    "value": [
                        {
                            "kind": "low_confidence_rule",
                            "detail": "rule LOW+LOW low conf",
                            "evidence": "LOW+LOW",
                        }
                    ],
                    "fresh": True,
                },
                "recent_self_tuning": {"value": {}, "fresh": True},
                "competence": {"value": 0.4, "fresh": True},
            }
        ),
    )

    seen: list[str] = []

    async def _capture(msg):  # type: ignore[no-untyped-def]
        seen.append(msg.subject)

    await nats_conn.subscribe("augur.imperator.proposal", cb=_capture)

    # Publish a past-dated disciplina.complete to trigger the improver.
    await nats_conn.publish(
        "augur.disciplina.complete",
        json.dumps(
            {"session_id": "itest", "timestamp": "2020-01-01T00:00:00+00:00"}
        ).encode(),
    )

    assert await wait_for_redis_key(
        redis_client, "augur:imperator:proposals", timeout=30.0
    )
