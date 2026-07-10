"""Integration test: Praesagium's live mining + matcher flow (fast tier —
real Redis + NATS, no subprocesses, no Ollama).

Drives the "library faculty" surfaces directly (Memoria/Conscientia
precedent, spec 2026-07-09 §13): ``run_praesagium_mining`` is called
in-process against a real ``PersistenceManager``, and the matcher's
``augur.vigil.anomaly`` callback (``make_on_anomaly``) is invoked directly
with the real ``nats_conn.publish`` as its publish function — no
``praesagium.matcher`` subprocess is spawned, so ``AUGUR_TEST_STARTUP_WAIT_S``
does not apply to this test (it still governs the rest of the suite).

Episodes are seeded directly via ``append_praesagium_episode`` rather than
produced by a live anomaly detector, so the zero-variance gotcha (identical
baseline values → forced-zero deviation) does not apply here — the corpus is
authored, not detected.

Flow:
1. Seed two synthetic sessions with an antecedent A → consequent B episode
   pair each (same ~60s lag, well inside the default lag window) — enough
   for ``praesagium_support_min_sessions=2``.
2. Mine once: the pair passes every Sec 4.4 promotion test and enters
   probation (``status="provisional"``, ``created_at`` = mine 1's wall
   clock).
3. Seed a third session with unrelated filler episodes timestamped AFTER
   mine 1 (real wall clock only moves forward), then mine again: the fresh
   corpus timestamp exceeds the pattern's ``created_at``, which is the
   provisional → active promotion trigger (Sec 4.6-2).
4. With ``praesagium_emit_enabled=True``, drive the matcher callback with a
   real antecedent anomaly: the pattern arms and a foreseen envelope is
   published on ``augur.praesagium.foreseen`` (subscribed before injecting).
5. Drive the matcher callback with the matching consequent (medium severity,
   inside the window): the open prediction resolves ``fulfilled`` — both a
   persisted record (``load_praesagium_resolved``) and an
   ``augur.praesagium.resolved`` event.

Hermetic via the ``redis_client`` fixture's per-test ``augur:*`` flush
(mirrors ``test_conscientia_live.py`` — no manual FLUSHALL needed inside the
test body).
"""

from __future__ import annotations

import asyncio
import json
import time
from datetime import datetime, timezone

import pytest

from praesagium.matcher import SUBJECT_FORESEEN, SUBJECT_RESOLVED, make_on_anomaly
from praesagium.miner import run_praesagium_mining
from tabula.config import AugurConfig
from tabula.persistence import PersistenceManager

_ANTE_DOMAIN = "typing"
_ANTE_ENTITY = "latency_spike"
_CONS_DOMAIN = "activity"
_CONS_ENTITY = "context_switch"
_ANTE_KEY = f"{_ANTE_DOMAIN}:{_ANTE_ENTITY}"
_CONS_KEY = f"{_CONS_DOMAIN}:{_CONS_ENTITY}"

_SESSION_1 = "praesagium-live-s1"
_SESSION_2 = "praesagium-live-s2"
_SESSION_3 = "praesagium-live-s3"
_RUN_SESSION = "praesagium-live-run"

_MINE_MARKER_1 = "praesagium-live-mine-1"
_MINE_MARKER_2 = "praesagium-live-mine-2"


class _Msg:
    """Minimal NATS-message stand-in: only ``.data`` (bytes) is read (mirrors
    ``tests/test_praesagium_matcher.py``'s ``_Msg``)."""

    def __init__(self, data: bytes) -> None:
        self.data = data


def _episode(key: str, severity: str, t: float) -> dict:
    return {"k": key, "s": severity, "t": t}


def _iso(ts: float) -> str:
    return datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()


def _config() -> AugurConfig:
    """Permissive mining thresholds (task spec) plus the two knobs the
    two-mine flow additionally needs: ``mine_min_interval_s=0.0`` so the
    second mine isn't rate-limited seconds after the first, and
    ``hit_rate_retire_below`` pulled below ``conf_lower_min`` to satisfy
    AugurConfig's cross-field validation. Emit armed for the matcher steps —
    mining itself never reads ``praesagium_emit_enabled``, so one config
    serves both halves of the test."""
    return AugurConfig(
        praesagium_enabled=True,
        praesagium_emit_enabled=True,
        praesagium_support_min_sessions=2,
        praesagium_conf_lower_min=0.1,
        praesagium_lift_min=1.0,
        praesagium_hit_rate_retire_below=0.05,
        praesagium_mine_min_interval_s=0.0,
    )


@pytest.mark.asyncio
async def test_praesagium_live_mining_and_foreseen_flow(
    redis_client, nats_conn
) -> None:
    """Two live mines (provisional → active) then a real foreseen/resolved
    round trip through the matcher callback."""
    pm = PersistenceManager(redis_client)
    cfg = _config()

    base = time.time() - 7200.0  # comfortably before either mine's wall clock

    # -- Step 1: seed two synthetic sessions, each one A occurrence followed
    # by a medium-severity B occurrence ~60s later (inside the default
    # (lag_min=10s, lag_max=900s] discovery window). --
    pm.append_praesagium_episode(_SESSION_1, _episode(_ANTE_KEY, "low", base))
    pm.append_praesagium_episode(_SESSION_1, _episode(_CONS_KEY, "medium", base + 60.0))
    pm.append_praesagium_episode(_SESSION_2, _episode(_ANTE_KEY, "low", base + 300.0))
    pm.append_praesagium_episode(
        _SESSION_2, _episode(_CONS_KEY, "medium", base + 360.0)
    )

    # -- Step 2 (first mine): the pair passes every Sec 4.4 promotion test
    # and enters probation. --
    report1 = run_praesagium_mining(_MINE_MARKER_1, pm, cfg)
    assert report1["corpus_sessions"] == 2
    assert report1["provisional"] == 1
    assert report1["active"] == 0

    blob1 = pm.load_praesagium_patterns()
    assert blob1 is not None
    patterns1 = blob1["patterns"]
    assert len(patterns1) == 1
    pid = next(iter(patterns1))
    pattern1 = patterns1[pid]
    assert pattern1["antecedent"] == _ANTE_KEY
    assert pattern1["consequent"] == _CONS_KEY
    assert pattern1["status"] == "provisional"
    assert pattern1["support_sessions"] == 2

    # Seed a THIRD session, timestamped after mine 1 (real wall clock only
    # moves forward — the sleep just guards against same-tick float ties).
    # Its keys are unrelated to the pattern; only its timestamp matters: the
    # corpus's newest episode must postdate the pattern's created_at (=mine
    # 1's wall clock) for provisional -> active promotion (Sec 4.6-2).
    await asyncio.sleep(0.05)
    fresh_ts = time.time()
    pm.append_praesagium_episode(_SESSION_3, _episode("typing:filler", "low", fresh_ts))
    pm.append_praesagium_episode(
        _SESSION_3, _episode("activity:filler", "low", fresh_ts + 5.0)
    )

    # -- Step 2 (second mine): the corpus now contains a session newer than
    # the pattern's created_at -> promoted to active. --
    report2 = run_praesagium_mining(_MINE_MARKER_2, pm, cfg)
    assert report2["corpus_sessions"] == 3
    assert report2["active"] == 1
    assert report2["promoted"] == 1

    blob2 = pm.load_praesagium_patterns()
    assert blob2 is not None
    pattern2 = blob2["patterns"][pid]
    assert pattern2["status"] == "active"
    assert pattern2["created_at"] == pattern1["created_at"]  # belief preserved

    # -- Step 3: drive the matcher callback directly, publishing over the
    # real NATS connection. Subscribe BEFORE injecting (NATS core has no
    # persistence, same discipline as test_conscientia_live.py). --
    received_foreseen: list[dict] = []
    received_resolved: list[dict] = []

    async def _capture_foreseen(msg) -> None:  # type: ignore[no-untyped-def]
        received_foreseen.append(json.loads(msg.data.decode()))

    async def _capture_resolved(msg) -> None:  # type: ignore[no-untyped-def]
        received_resolved.append(json.loads(msg.data.decode()))

    sub_foreseen = await nats_conn.subscribe(SUBJECT_FORESEEN, cb=_capture_foreseen)
    sub_resolved = await nats_conn.subscribe(SUBJECT_RESOLVED, cb=_capture_resolved)

    cooldowns: dict[str, float] = {}
    on_anomaly = make_on_anomaly(pm, cfg, nats_conn.publish, cooldowns)

    ante_ts = time.time()
    ante_payload = {
        "domain": _ANTE_DOMAIN,
        "entity": _ANTE_ENTITY,
        "severity": "low",
        "timestamp": _iso(ante_ts),
        "session_id": _RUN_SESSION,
    }
    await on_anomaly(_Msg(json.dumps(ante_payload).encode()))

    for _ in range(50):  # up to ~10s on the WSL/Windows-mount box
        if received_foreseen:
            break
        await asyncio.sleep(0.2)
    assert received_foreseen, "No augur.praesagium.foreseen event received"
    foreseen = received_foreseen[0]
    assert foreseen["source"] == "anticipatory"
    assert foreseen["primary_anomaly"]["domain"] == "praesagium"
    assert foreseen["primary_anomaly"]["entity"] == pid
    assert foreseen["anticipatory"]["pattern_id"] == pid
    assert foreseen["anticipatory"]["antecedent"] == _ANTE_KEY
    assert foreseen["anticipatory"]["consequent"] == _CONS_KEY

    opens = pm.load_praesagium_open_predictions()
    assert any(r.get("pattern_id") == pid for r in opens)

    # -- Step 4: the matching consequent, inside the armed window ->
    # fulfilled resolution. --
    cons_ts = ante_ts + 60.0  # < pattern2["window_s"] (~75s)
    assert cons_ts - ante_ts < pattern2["window_s"]
    cons_payload = {
        "domain": _CONS_DOMAIN,
        "entity": _CONS_ENTITY,
        "severity": "medium",
        "timestamp": _iso(cons_ts),
        "session_id": _RUN_SESSION,
    }
    await on_anomaly(_Msg(json.dumps(cons_payload).encode()))

    for _ in range(50):
        if received_resolved:
            break
        await asyncio.sleep(0.2)
    assert received_resolved, "No augur.praesagium.resolved event received"
    resolved_event = received_resolved[0]
    assert resolved_event["pattern_id"] == pid
    assert resolved_event["outcome"] == "fulfilled"

    resolved_records = pm.load_praesagium_resolved()
    matches = [
        r
        for r in resolved_records
        if r.get("pattern_id") == pid and r.get("outcome") == "fulfilled"
    ]
    assert matches, "No fulfilled resolution persisted via load_praesagium_resolved"

    await sub_foreseen.unsubscribe()
    await sub_resolved.unsubscribe()
