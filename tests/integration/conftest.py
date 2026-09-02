"""Integration test fixtures requiring real Redis and NATS infrastructure."""

from __future__ import annotations

import asyncio
import os
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

from tabula.config import AugurConfig  # noqa: E402
from tabula.contracts import PerceptionEvent  # noqa: E402
from tabula.persistence import PersistenceManager  # noqa: E402
from tests.integration.cell_guard import check_test_cell  # noqa: E402

_config = AugurConfig.from_env()


@pytest.fixture(scope="session", autouse=True)
def _require_test_cell() -> None:
    """Abort the whole suite unless we are in the test cell.

    autouse + session scope so this runs before any fixture writes. The suite
    deletes every augur:* key; against the live cell that is data loss.
    """
    reason = check_test_cell(_config)
    if reason is not None:
        pytest.exit(f"REFUSING TO RUN: {reason}", returncode=2)


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
    leftover = r.keys("augur:*")
    if leftover:
        r.delete(*leftover)
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
    """Fresh session identifier, with its provenance recorded.

    A live sensor mints provenance before its first event; the harness does the
    same, so anything driven by this fixture can train (see
    :func:`mint_session_provenance`).
    """
    return learnable_session()


@pytest_asyncio.fixture
async def real_pm(redis_client: redis.Redis) -> PersistenceManager:  # type: ignore[type-arg]
    """A real PersistenceManager over the flushed-per-test Redis client —
    the dialogue showcase scenarios' single source of durable cross-faculty
    state (used by ``tests/integration/test_dialogue_showcase.py``)."""
    return PersistenceManager(redis_client)


@pytest_asyncio.fixture
async def real_nc(nats_conn: nats_client.NATS) -> nats_client.NATS:
    """Alias of ``nats_conn`` under the dialogue showcase scenarios' own
    fixture name — the same real NATS connection ``engine.handle_turn``
    publishes ``augur.imperator.dialogue.*`` events over."""
    return nats_conn


@pytest.fixture
def dialogue_cfg() -> AugurConfig:
    """AugurConfig with dialogue defaults: confirmed-apply ON (the human-
    confirmed dialogue path), watch-first Imperator II apply OFF (default) —
    the showcase scenarios exercise only the confirmed path, never the
    autonomous one."""
    return AugurConfig()


@pytest_asyncio.fixture
async def pipeline(
    request: pytest.FixtureRequest,
) -> AsyncIterator[dict[str, asyncio.subprocess.Process]]:
    """Start one or more pipeline components as asyncio subprocesses.

    Parametrize with a list of component names, e.g.:
        @pytest.mark.parametrize("pipeline", [["vigil", "nexus", "consilium"]], indirect=True)

    Available components: vigil, nexus, consilium, responsum, disciplina, vox.
    """
    component_commands: dict[str, list[str]] = {
        "vigil": [sys.executable, "-m", "vigil.anomaly_detector"],
        "nexus": [sys.executable, "-m", "nexus.correlator"],
        "consilium": [sys.executable, "-m", "consilium.advisor"],
        "responsum": [sys.executable, "-m", "responsum.feedback_collector"],
        "disciplina": [sys.executable, "-m", "disciplina.reflection_engine"],
        "vox": [sys.executable, "-m", "vox.console_display"],
        "praefectus": [sys.executable, "-m", "praefectus.monitor"],
        "imperator": [sys.executable, "-m", "imperator.awareness"],
        "imperator_ii": [sys.executable, "-m", "imperator.improver"],
    }

    # param is either a plain list of component names, or a dict
    # {"components": [...], "env": {...}} when a test needs a component started
    # with a specific AUGUR_* override (the components read config at import,
    # so it has to be in the subprocess environment, not patched in-process).
    param = getattr(request, "param", [])
    if isinstance(param, dict):
        requested = list(param.get("components", []))
        extra_env = dict(param.get("env", {}))
    else:
        requested = list(param)
        extra_env = {}
    procs: dict[str, asyncio.subprocess.Process] = {}

    cell_env = {
        **os.environ,
        "AUGUR_REDIS_URL": _config.redis_url,
        "AUGUR_NATS_URL": _config.nats_url,
        **extra_env,
    }

    for name in requested:
        cmd = component_commands[name]
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            cwd=str(PROJECT_ROOT),
            env=cell_env,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        procs[name] = proc

    # Give components time to connect to Redis/NATS and subscribe. The wait is
    # AUGUR_TEST_STARTUP_WAIT_S-overridable: native-Linux CI is fast (~3s), but a
    # WSL/Windows-mount dev box imports river/numba off a slow filesystem and
    # needs longer or it injects events before the detector has subscribed (NATS
    # core has no persistence → the event is dropped → no baseline).
    await asyncio.sleep(float(os.environ.get("AUGUR_TEST_STARTUP_WAIT_S", "3.0")))

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


_provenance_pm: PersistenceManager | None = None


def mint_session_provenance(session_id: str, *, origin: str = "real") -> None:
    """Record a test-cell session's provenance (spec §4.4).

    Sessions in the test cell are real *within that cell* — nothing is withheld
    there — but "real" still has to be RECORDED, because provenance is only ever
    a durable record this system wrote, never an inference from a session id. So
    the harness mints it exactly like a live sensor does, rather than the
    provenance layer learning about cells.

    Without this, ENFORCE correctly withholds every learned write the suite
    triggers and the tuning assertions fail. Pass ``origin="synthetic"`` to
    exercise the withholding path on purpose.
    """
    global _provenance_pm
    if _provenance_pm is None:
        _provenance_pm = PersistenceManager(
            redis.Redis.from_url(_config.redis_url, decode_responses=True)
        )
    _provenance_pm.save_session_meta(
        session_id, origin=origin, created_by="integration"
    )


def learnable_session(session_id: str | None = None, *, origin: str = "real") -> str:
    """Mint a session id whose provenance is recorded, and return it.

    The one-liner replacement for a bare ``str(uuid4())`` in a test that then
    drives a learned write.
    """
    sid = session_id or str(uuid4())
    mint_session_provenance(sid, origin=origin)
    return sid


async def inject_perception_event(
    nc: nats_client.NATS,
    domain: str,
    entity: str,
    event_type: str,
    value: float,
    unit: str,
    context: dict,
    session_id: str,
    origin: str = "real",
) -> PerceptionEvent:
    """Create a PerceptionEvent and publish it to the NATS perception subject.

    Mints the session's provenance first (see :func:`mint_session_provenance`),
    so the components that consume the event can resolve it — the live sensors
    do the same thing before their first event.
    """
    mint_session_provenance(session_id, origin=origin)
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
    subject = f"augur.sensus.{domain}"
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
