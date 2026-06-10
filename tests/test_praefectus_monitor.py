# tests/test_praefectus_monitor.py
"""Monitor tick: drives the pure engine, snapshots to Redis, alerts on transitions."""

import json

import fakeredis

from praefectus import health as H
from praefectus import monitor as M
from tabula.config import AugurConfig
from tabula.persistence import PersistenceManager


class FakeNC:
    def __init__(self):
        self.published = []

    async def publish(self, subject, payload):
        self.published.append((subject, json.loads(payload)))


def test_tick_snapshots_and_alerts_never_started():
    cfg = AugurConfig()
    pm = PersistenceManager(fakeredis.FakeStrictRedis())
    states = H.initial_states(1000.0)
    window = H.ActivityWindow()
    nc = FakeNC()
    # past the never_started horizon, nobody heartbeats
    import asyncio

    asyncio.run(
        M.tick(nc, pm, states, window, now=1200.0, started_at=1000.0, config=cfg)
    )

    snap = pm.load_health_snapshot()
    assert snap["faculties"]["vox"]["overall"] == "dead"
    health = [d for s, d in nc.published if s == H.HEALTH_SUBJECT]
    assert any(
        m["transition"] == "dead" and m["reason"] == "never_started" for m in health
    )


def test_tick_clears_then_silent():
    cfg = AugurConfig()
    pm = PersistenceManager(fakeredis.FakeStrictRedis())
    states = H.initial_states(1000.0)
    for f in H.REQUIRED_FACULTIES:
        H.record_heartbeat(states, f, 1000.0)
    window = H.ActivityWindow()
    nc = FakeNC()
    import asyncio

    asyncio.run(
        M.tick(nc, pm, states, window, now=1001.0, started_at=1000.0, config=cfg)
    )
    # all alive, no transitions → no health publishes
    assert not [s for s, _ in nc.published if s == H.HEALTH_SUBJECT]
