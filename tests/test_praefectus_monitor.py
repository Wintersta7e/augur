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


def test_tick_publishes_recovered_when_required_faculty_recovers():
    # T1 (monitor): a previously-dead required faculty recovering publishes a
    # HEALTH_SUBJECT message with transition 'recovered'. Same states across ticks.
    import asyncio

    cfg = AugurConfig()
    pm = PersistenceManager(fakeredis.FakeStrictRedis())
    states = H.initial_states(1000.0)
    window = H.ActivityWindow()
    nc = FakeNC()
    # tick1: past horizon, vox never started → dead
    asyncio.run(
        M.tick(nc, pm, states, window, now=1200.0, started_at=1000.0, config=cfg)
    )
    assert any(
        d["transition"] == "dead" and d["faculty"] == "vox"
        for s, d in nc.published
        if s == H.HEALTH_SUBJECT
    )
    nc.published.clear()
    # tick2: vox heartbeats → recovered publish
    H.record_heartbeat(states, "vox", 1205.0)
    asyncio.run(
        M.tick(nc, pm, states, window, now=1206.0, started_at=1000.0, config=cfg)
    )
    health = [d for s, d in nc.published if s == H.HEALTH_SUBJECT]
    assert any(m["transition"] == "recovered" and m["faculty"] == "vox" for m in health)


def test_tick_publishes_lost_for_stopped_required_heartbeat():
    # T6 (M2): a required faculty whose heartbeat is far in the past shows dead
    # in the snapshot AND publishes a 'dead'/reason 'lost' transition.
    import asyncio

    cfg = AugurConfig()
    pm = PersistenceManager(fakeredis.FakeStrictRedis())
    states = H.initial_states(1000.0)
    for f in H.REQUIRED_FACULTIES:
        H.record_heartbeat(states, f, 1000.0)
    # vox's heartbeat is now stale-then-dead (age 1000 > dead_after 90)
    states["vox"].last_heartbeat = 1000.0
    window = H.ActivityWindow()
    nc = FakeNC()
    asyncio.run(
        M.tick(nc, pm, states, window, now=2000.0, started_at=900.0, config=cfg)
    )
    snap = pm.load_health_snapshot()
    assert snap["faculties"]["vox"]["overall"] == "dead"
    health = [d for s, d in nc.published if s == H.HEALTH_SUBJECT]
    assert any(
        m["transition"] == "dead" and m["reason"] == "lost" and m["faculty"] == "vox"
        for m in health
    )


def test_tick_no_masking_between_sensors():
    # T6 (M2): chess dead while typing alive → snapshot shows distinct states.
    import asyncio

    cfg = AugurConfig()
    pm = PersistenceManager(fakeredis.FakeStrictRedis())
    states = H.initial_states(1000.0)
    for f in H.REQUIRED_FACULTIES:
        H.record_heartbeat(states, f, 2000.0)
    H.record_heartbeat(states, "sensus.chess", 1000.0)  # stale → dead at now=2000
    H.record_heartbeat(states, "sensus.typing", 2000.0)  # fresh → alive
    window = H.ActivityWindow()
    nc = FakeNC()
    asyncio.run(
        M.tick(nc, pm, states, window, now=2000.0, started_at=1000.0, config=cfg)
    )
    snap = pm.load_health_snapshot()
    assert snap["faculties"]["sensus.chess"]["liveness"] == "dead"
    assert snap["faculties"]["sensus.typing"]["liveness"] == "alive"


def test_run_early_exits_when_disabled_without_connecting(monkeypatch):
    # T5 (M11): praefectus_enabled=False → run() returns None WITHOUT connecting.
    # connect_redis / nats.connect are patched to blow up if reached.
    import asyncio

    import nats

    disabled = AugurConfig(praefectus_enabled=False)
    monkeypatch.setattr(M.AugurConfig, "from_env", classmethod(lambda cls: disabled))

    def _boom(*a, **k):
        raise AssertionError("must not connect when praefectus disabled")

    monkeypatch.setattr(M, "connect_redis", _boom)
    monkeypatch.setattr(nats, "connect", _boom)
    assert asyncio.run(M.run()) is None


def test_on_msg_tolerates_malformed_heartbeat_payloads():
    # T5 (M11): the on_msg heartbeat branch is a closure inside run() and not
    # directly importable, so this exercises the same H3-hardened json/dict/ts
    # guard the closure uses against non-JSON and valid-but-non-dict payloads.
    import asyncio

    cfg = AugurConfig()
    states = H.initial_states(1000.0)
    window = H.ActivityWindow()

    async def on_msg(msg) -> None:
        kind, _ = H.classify_event(msg.subject)
        now = 1000.0
        if kind == "heartbeat":
            try:
                data = json.loads(msg.data)
                if not isinstance(data, dict):
                    return
                ts = float(data.get("ts", now))
            except (ValueError, TypeError):
                return
            H.record_heartbeat(states, data.get("faculty"), ts)
        elif kind == "activity":
            try:
                data = json.loads(msg.data)
            except (ValueError, TypeError):
                data = {}
            H.record_activity(states, window, msg.subject, data, now, cfg)

    class _Msg:
        def __init__(self, subject, data):
            self.subject = subject
            self.data = data

    # non-JSON bytes and valid-JSON-non-dict (b"5") must not raise
    asyncio.run(on_msg(_Msg(H.HEARTBEAT_SUBJECT, b"not json")))
    asyncio.run(on_msg(_Msg(H.HEARTBEAT_SUBJECT, b"5")))
    # well-formed payload still records normally
    asyncio.run(
        on_msg(
            _Msg(
                H.HEARTBEAT_SUBJECT,
                json.dumps({"faculty": "vox", "ts": 999.0}).encode(),
            )
        )
    )
    assert states["vox"].last_heartbeat == 999.0
