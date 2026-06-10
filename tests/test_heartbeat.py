# tests/test_heartbeat.py
"""Heartbeat emission — async helper (this task) + sync NatsPublisher (Task 6)."""

import asyncio
import json
from unittest.mock import AsyncMock

import pytest

from tabula import heartbeat as HB


@pytest.mark.asyncio
async def test_async_heartbeat_publishes():
    nc = AsyncMock()
    task = HB.start_heartbeat(nc, "vigil", 0.01)
    await asyncio.sleep(0.05)
    task.cancel()
    assert nc.publish.await_count >= 1
    subject, payload = nc.publish.await_args_list[0].args
    assert subject == HB.HEARTBEAT_SUBJECT
    data = json.loads(payload)
    assert data["faculty"] == "vigil" and "ts" in data


@pytest.mark.asyncio
async def test_async_heartbeat_survives_publish_error():
    nc = AsyncMock()
    nc.publish.side_effect = RuntimeError("nats down")
    task = HB.start_heartbeat(nc, "vox", 0.01)
    await asyncio.sleep(0.05)
    task.cancel()
    assert nc.publish.await_count >= 1  # kept looping despite errors


def test_chess_natspublisher_heartbeat(monkeypatch):
    import threading
    from sensus.chess_board import NatsPublisher
    from tabula.config import AugurConfig
    from tabula.heartbeat import HEARTBEAT_SUBJECT

    pub = NatsPublisher(AugurConfig())
    calls: list[tuple] = []
    # avoid real NATS: record publishes instead of driving the private loop
    monkeypatch.setattr(
        pub, "publish", lambda subject, payload: calls.append((subject, payload))
    )

    pub.start_heartbeat("sensus.chess", 0.01)
    threading.Event().wait(0.05)
    pub.stop_heartbeat()

    assert any(
        s == HEARTBEAT_SUBJECT and p["faculty"] == "sensus.chess" for s, p in calls
    )


def test_chess_publish_has_lock():
    import threading
    from sensus.chess_board import NatsPublisher
    from tabula.config import AugurConfig

    pub = NatsPublisher(AugurConfig())
    assert isinstance(pub._pub_lock, type(threading.Lock()))
