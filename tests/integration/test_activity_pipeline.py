"""Fast integration tests for the activity_focus + activity_intensity pipeline.

Requires Redis + NATS running (docker compose up -d). The tests publish
hand-crafted PerceptionEvents directly to NATS — the Windows daemon is
not exercised; that's manual-verification territory.

Marked NOT slow; the Ollama-gated case lives in a separate file with
@pytest.mark.slow.
"""

from __future__ import annotations

import asyncio
import json
import time
import uuid

import pytest
import redis

from tabula.config import AugurConfig
from tabula.connections import connect_redis
from tabula.contracts import PerceptionEvent


async def _publish(nc, subject: str, event: PerceptionEvent) -> None:
    await nc.publish(subject, event.to_bytes())


async def _wait_until(
    condition_fn, timeout_s: float = 10.0, interval_s: float = 0.1
) -> bool:
    """Poll condition_fn until it returns truthy or timeout (in seconds)."""
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if condition_fn():
            return True
        await asyncio.sleep(interval_s)
    return False


def _focus_event(
    session_id: str,
    entity: str = "code",
    value: float = 4.8,
    prev_app: str = "code",
    new_app: str = "chrome",
    active_dwell_s: float = 25.0,
) -> PerceptionEvent:
    return PerceptionEvent(
        domain="activity_focus",
        stream_id="activity_focus",
        entity=entity,
        event_type="focus_change",
        value=value,
        unit="log1p_seconds",
        context={
            "prev_app": prev_app,
            "new_app": new_app,
            "prev_title": None,
            "new_title": None,
            "active_dwell_s": active_dwell_s,
            "idle_dwell_s": 5.0,
            "total_dwell_s": active_dwell_s + 5.0,
            "source_id": "test-host",
            "span_id": str(uuid.uuid4()),
        },
        timestamp="2026-05-16T12:00:00+00:00",
        session_id=session_id,
    )


def _intensity_event(
    session_id: str,
    entity: str = "code",
    value: float = 240.0,
    keystrokes: int = 30,
    mouse_events: int = 10,
) -> PerceptionEvent:
    return PerceptionEvent(
        domain="activity_intensity",
        stream_id="activity_intensity",
        entity=entity,
        event_type="intensity_sample",
        value=value,
        unit="ipm",
        context={
            "focused_app": entity,
            "title": None,
            "keystroke_count": keystrokes,
            "mouse_event_count": mouse_events,
            "idle_seconds": 0.5,
            "window_duration_s": 10.0,
            "source_id": "test-host",
            "span_id": str(uuid.uuid4()),
        },
        timestamp="2026-05-16T12:00:01+00:00",
        session_id=session_id,
    )


def _typing_event(session_id: str, value: float = 3.5) -> PerceptionEvent:
    return PerceptionEvent(
        domain="typing",
        stream_id="typing_pause",
        entity="user",
        event_type="pause",
        value=value,
        unit="seconds",
        context={"avg_wpm": 45, "keypress_count": 1200},
        timestamp="2026-05-16T12:00:02+00:00",
        session_id=session_id,
    )


@pytest.fixture
def clean_redis(session_id):
    """Wipe baselines/state so each test starts from a clean baseline."""
    config = AugurConfig.from_env()
    r = connect_redis(config)
    # Clean activity-related keys before the test
    for pattern in [
        "augur:vigil:profile:activity_*",
        "augur:vigil:history:activity_*",
        "augur:vigil:*",
        "augur:nexus:*",
        "augur:responsum:*",
    ]:
        keys = r.keys(pattern) or []
        for k in keys:
            try:
                r.delete(k)
            except redis.exceptions.ResponseError:
                # Some keys may be lists/sets, skip
                pass
    yield r
    # Clean up after the test
    for pattern in [
        "augur:vigil:profile:activity_*",
        "augur:vigil:history:activity_*",
        "augur:vigil:*",
        "augur:nexus:*",
        "augur:responsum:*",
    ]:
        keys = r.keys(pattern) or []
        for k in keys:
            try:
                r.delete(k)
            except redis.exceptions.ResponseError:
                # Some keys may be lists/sets, skip
                pass


@pytest.fixture
def clean_redis_sync():
    """Sync Redis fixture for non-async tests (T17)."""
    config = AugurConfig.from_env()
    r = connect_redis(config)
    # Clean up session-related keys for T17
    r.delete("augur:session:current")
    yield r
    r.delete("augur:session:current")


# ---------------------------------------------------------------------------
# T12: activity_focus baseline creation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("pipeline", [["vigil"]], indirect=True)
@pytest.mark.asyncio
async def test_activity_focus_event_creates_baseline(pipeline, clean_redis, session_id):
    """Publishing focus events through NATS creates a per-(domain, entity) baseline."""
    import nats

    config = AugurConfig.from_env()
    nc = await nats.connect(config.nats_url)
    try:
        for i in range(max(3, config.min_observations) + 1):
            ev = _focus_event(session_id=session_id, value=3.0 + 0.1 * i)
            await _publish(nc, "augur.sensus.activity_focus", ev)
        await nc.flush()
        assert await _wait_until(
            lambda: (
                clean_redis.get("augur:vigil:profile:activity_focus:focus_change:code")
                is not None
            ),
            timeout_s=5.0,
        ), "baseline not created within timeout"
    finally:
        await nc.drain()

    raw = clean_redis.get("augur:vigil:profile:activity_focus:focus_change:code")
    assert raw is not None, "baseline for (activity_focus, code) was not created"
    bl = json.loads(raw)
    assert "ewma_mean" in bl
    assert bl["ewma_mean"] > 0


# ---------------------------------------------------------------------------
# T13: activity_intensity baseline creation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("pipeline", [["vigil"]], indirect=True)
@pytest.mark.asyncio
async def test_activity_intensity_event_creates_baseline(
    pipeline, clean_redis, session_id
):
    import nats

    config = AugurConfig.from_env()
    nc = await nats.connect(config.nats_url)
    try:
        for i in range(max(3, config.min_observations) + 1):
            ev = _intensity_event(session_id=session_id, value=50.0 + 10.0 * i)
            await _publish(nc, "augur.sensus.activity_intensity", ev)
        await nc.flush()
        assert await _wait_until(
            lambda: (
                clean_redis.get(
                    "augur:vigil:profile:activity_intensity:intensity_sample:code"
                )
                is not None
            ),
            timeout_s=5.0,
        ), "baseline not created within timeout"
    finally:
        await nc.drain()

    raw = clean_redis.get(
        "augur:vigil:profile:activity_intensity:intensity_sample:code"
    )
    assert raw is not None, "baseline for (activity_intensity, code) was not created"
    bl = json.loads(raw)
    assert bl["ewma_mean"] > 0


# ---------------------------------------------------------------------------
# T14: cross-domain activity_focus + typing
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("pipeline", [["vigil", "nexus"]], indirect=True)
@pytest.mark.asyncio
async def test_activity_focus_and_typing_correlate_cross_domain(
    pipeline, clean_redis, session_id
):
    """Co-occurring activity_focus + typing anomalies should produce a
    correlation event with both domains in involved_domains."""
    import nats

    config = AugurConfig.from_env()
    nc = await nats.connect(config.nats_url)

    received: list[dict] = []

    async def _capture(msg):
        received.append(json.loads(msg.data.decode()))

    sub = await nc.subscribe("augur.nexus.detected", cb=_capture)
    try:
        # Build baselines with varied normal values. Size adapts to
        # config.min_observations so future bumps don't break this test.
        _warm = config.min_observations + 5
        _focus_pool = [4.5, 5.2, 4.8, 5.5, 4.3, 5.1, 4.7, 5.3, 4.9, 5.0]
        _typing_pool = [3.2, 3.8, 3.5, 4.0, 3.1, 3.9, 3.3, 3.7, 3.4, 3.6]
        baseline_focus = [_focus_pool[i % len(_focus_pool)] for i in range(_warm)]
        baseline_typing = [_typing_pool[i % len(_typing_pool)] for i in range(_warm)]

        for val_f, val_t in zip(baseline_focus, baseline_typing):
            await _publish(
                nc,
                "augur.sensus.activity_focus",
                _focus_event(session_id, value=val_f),
            )
            await _publish(
                nc, "augur.sensus.typing", _typing_event(session_id, value=val_t)
            )
        await nc.flush()
        assert await _wait_until(
            lambda: (
                clean_redis.get("augur:vigil:profile:activity_focus:focus_change:code")
                is not None
            ),
            timeout_s=5.0,
        ), "typing baseline not created within timeout"

        # Publish extreme outliers in both domains
        await _publish(
            nc,
            "augur.sensus.activity_focus",
            _focus_event(session_id, value=50.0, active_dwell_s=600.0),
        )
        await asyncio.sleep(0.5)
        await _publish(nc, "augur.sensus.typing", _typing_event(session_id, value=50.0))
        await nc.flush()
        await _wait_until(
            lambda: any(
                set(c.get("involved_domains", [])) >= {"activity_focus", "typing"}
                for c in received
            ),
            timeout_s=10.0,
        )
    finally:
        await sub.unsubscribe()
        await nc.drain()

    cross = [
        c
        for c in received
        if set(c.get("involved_domains", [])) >= {"activity_focus", "typing"}
    ]
    assert cross, (
        "expected a correlation event involving both activity_focus and typing; "
        f"got {[set(c.get('involved_domains', [])) for c in received]}"
    )


# ---------------------------------------------------------------------------
# T15: cross-stream activity_focus + activity_intensity
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("pipeline", [["vigil", "nexus"]], indirect=True)
@pytest.mark.asyncio
async def test_activity_focus_and_intensity_correlate_cross_domain(
    pipeline, clean_redis, session_id
):
    """activity_focus and activity_intensity are SEPARATE domains by design,
    so the correlator's same-domain filter does NOT exclude them — they
    should correlate when their anomalies co-occur."""
    import nats

    config = AugurConfig.from_env()
    nc = await nats.connect(config.nats_url)

    received: list[dict] = []

    async def _capture(msg):
        received.append(json.loads(msg.data.decode()))

    sub = await nc.subscribe("augur.nexus.detected", cb=_capture)
    try:
        # Build baselines with varied normal values. Adaptive sizing so
        # future config.min_observations bumps don't break this test.
        _warm = config.min_observations + 5
        _focus_pool = [4.5, 5.2, 4.8, 5.5, 4.3, 5.1, 4.7, 5.3, 4.9, 5.0]
        _intensity_pool = [60.0, 70.0, 65.0, 75.0, 62.0, 72.0, 68.0, 74.0, 66.0, 73.0]
        baseline_focus = [_focus_pool[i % len(_focus_pool)] for i in range(_warm)]
        baseline_intensity = [
            _intensity_pool[i % len(_intensity_pool)] for i in range(_warm)
        ]

        for val_f, val_i in zip(baseline_focus, baseline_intensity):
            await _publish(
                nc,
                "augur.sensus.activity_focus",
                _focus_event(session_id, value=val_f),
            )
            await _publish(
                nc,
                "augur.sensus.activity_intensity",
                _intensity_event(session_id, value=val_i),
            )
        await nc.flush()
        assert await _wait_until(
            lambda: (
                clean_redis.get(
                    "augur:vigil:profile:activity_intensity:intensity_sample:code"
                )
                is not None
            ),
            timeout_s=5.0,
        ), "intensity baseline not created within timeout"

        # Publish extreme outliers in both domains
        await _publish(
            nc,
            "augur.sensus.activity_focus",
            _focus_event(session_id, value=50.0, active_dwell_s=500.0),
        )
        await asyncio.sleep(0.5)
        await _publish(
            nc,
            "augur.sensus.activity_intensity",
            _intensity_event(session_id, value=350.0),
        )
        await nc.flush()
        await _wait_until(
            lambda: any(
                set(c.get("involved_domains", []))
                >= {"activity_focus", "activity_intensity"}
                for c in received
            ),
            timeout_s=10.0,
        )
    finally:
        await sub.unsubscribe()
        await nc.drain()

    cross = [
        c
        for c in received
        if set(c.get("involved_domains", []))
        >= {"activity_focus", "activity_intensity"}
    ]
    assert cross, (
        "expected a correlation event involving both activity_focus and "
        f"activity_intensity; got {[set(c.get('involved_domains', [])) for c in received]}"
    )


# ---------------------------------------------------------------------------
# T16: feedback collector keying isolation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("pipeline", [["responsum"]], indirect=True)
@pytest.mark.asyncio
async def test_feedback_keying_separates_focus_and_intensity_tracking(
    pipeline, clean_redis, session_id, nats_conn
):
    """Two advice events for the same entity 'code' but different domains
    (activity_focus vs activity_intensity) must create independent
    PendingAdvice records in the feedback collector."""
    nc = nats_conn
    advice_focus = {
        "advice_id": "adv-focus",
        "player": "code",
        "severity": "HIGH",
        "move": "n/a",
        "think_time": 0,
        "domain": "activity_focus",
        "primary_domain": "activity_focus",
        "involved_domains": ["activity_focus"],
    }
    advice_intensity = {
        "advice_id": "adv-intensity",
        "player": "code",
        "severity": "HIGH",
        "move": "n/a",
        "think_time": 0,
        "domain": "activity_intensity",
        "primary_domain": "activity_intensity",
        "involved_domains": ["activity_intensity"],
    }
    await nc.publish("augur.consilium.advice", json.dumps(advice_focus).encode())
    await nc.publish("augur.consilium.advice", json.dumps(advice_intensity).encode())
    await nc.flush()
    assert await _wait_until(
        lambda: len(clean_redis.keys("augur:responsum:*") or []) > 0,
        timeout_s=10.0,
    ), "feedback keys not created within timeout"

    # Strengthened: confirm both activity_focus and activity_intensity advice
    # appear as DISTINCT entries in the same feedback session.
    all_keys = clean_redis.keys("augur:responsum:*")
    seen_domains: set[str] = set()  # domains seen in advice_events
    for k in all_keys:
        # Skip the _index key which is typically a list
        k_str = k.decode() if isinstance(k, bytes) else k
        if "_index" in k_str:
            continue
        try:
            v = clean_redis.get(k)
        except redis.exceptions.ResponseError:
            # Some keys are lists/sets, skip
            continue
        if not v:
            continue
        try:
            text = v.decode() if isinstance(v, bytes) else v
            data = json.loads(text)
        except (json.JSONDecodeError, TypeError):
            continue
        if not isinstance(data, dict):
            continue
        for entry in data.get("advice_events", []):
            if not isinstance(entry, dict):
                continue
            dom = entry.get("domain")
            if dom in ("activity_focus", "activity_intensity"):
                seen_domains.add(dom)

    assert "activity_focus" in seen_domains, (
        f"activity_focus advice not found in feedback; domains seen={seen_domains}"
    )
    assert "activity_intensity" in seen_domains, (
        f"activity_intensity advice not found in feedback; domains seen={seen_domains}"
    )


# ---------------------------------------------------------------------------
# T17: get_active_session against real Redis
# ---------------------------------------------------------------------------


def test_get_active_session_helper_returns_none_for_ended_session(clean_redis_sync):
    """Direct check of the validity helper using real Redis."""
    from datetime import datetime, timezone

    from tabula.session import get_active_session

    clean_redis_sync.set(
        "augur:session:current",
        json.dumps(
            {
                "session_id": "sess-stale",
                "started_at": datetime.now(timezone.utc).isoformat(),
                "status": "ended",
            }
        ),
    )
    assert get_active_session(clean_redis_sync, max_age_h=12.0) is None


def test_get_active_session_helper_returns_id_for_active_fresh(clean_redis_sync):
    from datetime import datetime, timezone

    from tabula.session import get_active_session

    clean_redis_sync.set(
        "augur:session:current",
        json.dumps(
            {
                "session_id": "sess-live",
                "started_at": datetime.now(timezone.utc).isoformat(),
                "status": "active",
            }
        ),
    )
    assert get_active_session(clean_redis_sync, max_age_h=12.0) == "sess-live"


# ---------------------------------------------------------------------------
# T18: Slow Ollama-gated activity_focus advice
# ---------------------------------------------------------------------------


@pytest.mark.slow
@pytest.mark.parametrize("pipeline", [["vigil", "nexus", "consilium"]], indirect=True)
@pytest.mark.asyncio
async def test_activity_focus_advisor_returns_advice_via_ollama(
    pipeline, clean_redis, session_id
):
    """Activity-only outlier → advisor → Ollama. Asserts shape only,
    not content. Gated on AUGUR_OLLAMA_URL being set."""
    import os

    if not os.environ.get("AUGUR_OLLAMA_URL"):
        pytest.skip("AUGUR_OLLAMA_URL not set")

    import nats

    config = AugurConfig.from_env()
    nc = await nats.connect(config.nats_url)

    received: list[dict] = []

    async def _capture(msg):
        received.append(json.loads(msg.data.decode()))

    sub = await nc.subscribe("augur.consilium.advice", cb=_capture)
    try:
        for i in range(max(6, config.min_observations + 3)):
            await _publish(
                nc,
                "augur.sensus.activity_focus",
                _focus_event(session_id, value=3.0),
            )
        await nc.flush()
        await asyncio.sleep(2.0)
        # Outlier.
        await _publish(
            nc,
            "augur.sensus.activity_focus",
            _focus_event(session_id, value=10.0, active_dwell_s=600.0),
        )
        await nc.flush()
        # Ollama cold start can be 60s+; give generous slack.
        await asyncio.sleep(90.0)
    finally:
        await sub.unsubscribe()
        await nc.drain()

    activity_advice = [a for a in received if a.get("domain") == "activity_focus"]
    assert activity_advice, "expected at least one activity_focus advice from Ollama"
    msg = activity_advice[0]
    assert msg.get("advice"), "advice field is empty"
    assert isinstance(msg["advice"], str)
