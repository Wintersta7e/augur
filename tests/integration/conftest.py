"""Integration test fixtures requiring real Redis and NATS infrastructure."""

from __future__ import annotations

import asyncio
import signal
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import AsyncIterator
from uuid import uuid4

import nats as nats_client
import pytest
import pytest_asyncio
import redis

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from blackboard.config import AugurConfig  # noqa: E402
from blackboard.contracts import PerceptionEvent  # noqa: E402

_config = AugurConfig.from_env()


# ---------------------------------------------------------------------------
# Infrastructure availability probes (run at module load time)
# ---------------------------------------------------------------------------


def _redis_available() -> bool:
    try:
        r = redis.Redis.from_url(_config.redis_url, socket_connect_timeout=2)
        r.ping()
        r.close()
        return True
    except Exception:
        return False


def _nats_available() -> bool:
    try:

        async def _check() -> bool:
            nc = await nats_client.connect(_config.nats_url, connect_timeout=2)
            await nc.close()
            return True

        return asyncio.run(_check())
    except Exception:
        return False


def _ollama_available() -> bool:
    try:
        import httpx

        resp = httpx.get(f"{_config.ollama_url}/api/tags", timeout=3)
        return resp.status_code == 200
    except Exception:
        return False


_infra_ok: bool = _redis_available() and _nats_available()

pytestmark = pytest.mark.skipif(
    not _infra_ok,
    reason="Redis or NATS not reachable — skipping integration tests",
)

# ---------------------------------------------------------------------------
# Marker / decorator for tests that require Ollama
# ---------------------------------------------------------------------------

requires_ollama = pytest.mark.skipif(
    not _ollama_available(),
    reason="Ollama not reachable — skipping LLM-dependent test",
)
requires_ollama = pytest.mark.slow(requires_ollama)  # type: ignore[assignment]


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest_asyncio.fixture
async def redis_client() -> AsyncIterator[redis.Redis]:  # type: ignore[type-arg]
    """Synchronous Redis client with augur:* key cleanup before each test."""
    r: redis.Redis = redis.Redis.from_url(  # type: ignore[type-arg]
        _config.redis_url,
        socket_connect_timeout=_config.redis_connect_timeout,
        decode_responses=True,
    )
    # Clean up any leftover augur keys from previous runs
    keys = r.keys("augur:*")
    if keys:
        r.delete(*keys)
    yield r
    r.close()


@pytest_asyncio.fixture
async def nats_conn() -> AsyncIterator[nats_client.NATS]:
    """Real NATS connection, closed after the test."""
    nc = await nats_client.connect(
        _config.nats_url,
        connect_timeout=_config.nats_connect_timeout,
    )
    yield nc
    if not nc.is_closed:
        await nc.close()


@pytest.fixture
def session_id() -> str:
    """Fresh UUID session identifier."""
    return str(uuid4())


@pytest_asyncio.fixture
async def pipeline(
    request: pytest.FixtureRequest,
) -> AsyncIterator[dict[str, asyncio.subprocess.Process]]:
    """Start one or more pipeline components as asyncio subprocesses.

    Parametrize with a list of component names, e.g.:
        @pytest.mark.parametrize("pipeline", [["detector", "advisor"]], indirect=True)

    Available components: detector, advisor, feedback, reflection, display.
    """
    component_commands: dict[str, list[str]] = {
        "detector": [sys.executable, "-m", "detection.anomaly_detector"],
        "advisor": [sys.executable, "-m", "reasoning.augur_advisor"],
        "feedback": [sys.executable, "-m", "perception.feedback_collector"],
        "reflection": [sys.executable, "-m", "reasoning.reflection_engine"],
        "display": [sys.executable, "-m", "output.console_display"],
    }

    requested: list[str] = getattr(request, "param", [])
    procs: dict[str, asyncio.subprocess.Process] = {}

    for name in requested:
        cmd = component_commands[name]
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            cwd=str(PROJECT_ROOT),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        procs[name] = proc

    # Give components time to connect to Redis/NATS and subscribe
    await asyncio.sleep(3.0)

    yield procs

    # Teardown: SIGTERM then SIGKILL if needed
    for name, proc in procs.items():
        if proc.returncode is None:
            try:
                proc.send_signal(signal.SIGTERM)
                try:
                    await asyncio.wait_for(proc.wait(), timeout=3.0)
                except asyncio.TimeoutError:
                    proc.kill()
                    await proc.wait()
            except ProcessLookupError:
                pass  # already exited


# ---------------------------------------------------------------------------
# Helper functions (not fixtures)
# ---------------------------------------------------------------------------


async def inject_perception_event(
    nc: nats_client.NATS,
    domain: str,
    entity: str,
    event_type: str,
    value: float,
    unit: str,
    context: dict,
    session_id: str,
) -> PerceptionEvent:
    """Create a PerceptionEvent and publish it to the NATS perception subject."""
    event = PerceptionEvent(
        domain=domain,
        stream_id=f"{domain}_stream",
        entity=entity,
        event_type=event_type,
        value=value,
        unit=unit,
        context=context,
        timestamp=datetime.now(timezone.utc).isoformat(),
        session_id=session_id,
    )
    subject = f"augur.perception.{domain}"
    await nc.publish(subject, event.to_bytes())
    return event


async def wait_for_redis_key(
    r: redis.Redis,  # type: ignore[type-arg]
    key: str,
    timeout: float = 10.0,
    poll_interval: float = 0.2,
) -> bool:
    """Poll Redis until *key* appears or *timeout* seconds elapse.

    Returns True if the key was found, False on timeout.
    """
    elapsed = 0.0
    while elapsed < timeout:
        if r.exists(key):
            return True
        await asyncio.sleep(poll_interval)
        elapsed += poll_interval
    return False


async def wait_for_redis_pattern(
    r: redis.Redis,  # type: ignore[type-arg]
    pattern: str,
    timeout: float = 10.0,
    poll_interval: float = 0.2,
) -> bool:
    """Poll Redis scan_iter until at least one key matching *pattern* appears.

    Returns True if a matching key was found, False on timeout.
    """
    elapsed = 0.0
    while elapsed < timeout:
        for _ in r.scan_iter(pattern):
            return True
        await asyncio.sleep(poll_interval)
        elapsed += poll_interval
    return False
