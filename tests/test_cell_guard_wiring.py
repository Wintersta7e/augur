"""Proves the live-cell guard is wired into the integration suite, not just
correct as a standalone predicate.

``tests/test_cell_guard.py`` covers ``check_test_cell()`` as a pure function.
That leaves a gap: nothing proves the predicate is actually consulted before
any integration fixture runs. Deleting ``autouse=True`` from
``tests/integration/conftest.py``'s ``_require_test_cell`` fixture would leave
every other test in this repository green while the interlock silently
stopped protecting the live cell (Redis db 0, NATS port 4222) that real
sensors write to. Two checks close that gap:

* ``test_live_cell_aborts_before_any_test_runs`` replays the fixture's exact
  wiring shape — session-scoped, autouse, calling ``check_test_cell`` and
  ``pytest.exit`` on refusal — inside a throwaway pytest session built by the
  ``pytester`` fixture, and asserts a live-cell config aborts with exit code
  2 before any test body executes. It calls the real ``check_test_cell``, so
  it also catches regressions in the predicate itself.
* ``test_real_fixture_is_session_scoped_autouse`` inspects the actual
  ``_require_test_cell`` fixture object in ``tests/integration/conftest.py``
  and asserts its pytest fixture marker carries session scope and
  ``autouse=True``. This is the assertion that fails the moment someone
  deletes ``autouse=True`` from the production fixture. The pytester replay
  above cannot exercise that module directly: importing it runs real
  Redis/NATS/Ollama reachability probes at import time as plain function
  calls (not fixtures pytest could substitute), so this test neutralizes
  those probes with mocks before importing — no socket is ever opened.
"""

from __future__ import annotations

import importlib
from unittest.mock import patch

import httpx
import nats
import pytest
import redis

pytest_plugins = ["pytester"]

# Config spellings the integration suite must refuse to run against: real
# Redis db 0, and ways the live NATS port (4222 — nats-py's own default when
# a URL carries no explicit port) can be spelled, each paired with an
# otherwise-safe partner value so the case isolates which check fires.
LIVE_CELL_CONFIGS = [
    pytest.param("redis://127.0.0.1:6379/0", "nats://127.0.0.1:4223", id="redis-db0"),
    pytest.param(
        "redis://127.0.0.1:6379/1", "nats://127.0.0.1:4222", id="nats-explicit-port"
    ),
    pytest.param(
        "redis://127.0.0.1:6379/1", "nats://localhost", id="nats-portless-host"
    ),
    pytest.param("redis://127.0.0.1:6379/1", "nats://127.0.0.1", id="nats-portless-ip"),
    pytest.param("redis://127.0.0.1:6379/0", "nats://127.0.0.1:4222", id="both-live"),
]

TEST_CELL_REDIS_URL = "redis://127.0.0.1:6379/1"
TEST_CELL_NATS_URL = "nats://127.0.0.1:4223"

# The wiring shape under test, reproduced verbatim from
# tests/integration/conftest.py's _require_test_cell fixture. It imports the
# REAL check_test_cell (a pure function with no I/O) so a regression in the
# predicate's logic fails this test too, not just a regression in the wiring.
_CONFTEST_SOURCE = """
import pytest
from tabula.config import AugurConfig
from tests.integration.cell_guard import check_test_cell

_config = AugurConfig(redis_url={redis_url!r}, nats_url={nats_url!r})


@pytest.fixture(scope="session", autouse=True)
def _require_test_cell():
    reason = check_test_cell(_config)
    if reason is not None:
        pytest.exit(f"REFUSING TO RUN: {{reason}}", returncode=2)
"""

_TEST_SOURCE = """
def test_would_touch_live_state():
    assert False, "this test body must never execute against a live cell"
"""


@pytest.mark.parametrize(
    "redis_url,nats_url",
    [(p.values[0], p.values[1]) for p in LIVE_CELL_CONFIGS],
    ids=[p.id for p in LIVE_CELL_CONFIGS],
)
def test_live_cell_aborts_before_any_test_runs(
    pytester: pytest.Pytester, redis_url: str, nats_url: str
) -> None:
    """A pytest session wired like the real one must exit(2) with zero tests
    run when pointed at any live-cell spelling — before the test body (which
    stands in for a fixture that would delete augur:* keys) ever executes."""
    pytester.makeconftest(
        _CONFTEST_SOURCE.format(redis_url=redis_url, nats_url=nats_url)
    )
    pytester.makepyfile(test_would_touch_live_state=_TEST_SOURCE)

    result = pytester.runpytest()

    assert result.ret == 2, f"expected abort exit code 2, got {result.ret}"
    result.stdout.fnmatch_lines(["*REFUSING TO RUN*"])
    outcomes = result.parseoutcomes()
    assert outcomes.get("passed", 0) == 0
    assert outcomes.get("failed", 0) == 0
    assert outcomes.get("error", 0) == 0


def test_test_cell_config_runs_normally(pytester: pytest.Pytester) -> None:
    """Control case: the identical wiring shape, pointed at the real test
    cell (db 1, port 4223), must let the test body run. Without this, a bug
    that made the fixture unconditionally call pytest.exit would still pass
    the abort assertions above for the wrong reason."""
    pytester.makeconftest(
        _CONFTEST_SOURCE.format(
            redis_url=TEST_CELL_REDIS_URL, nats_url=TEST_CELL_NATS_URL
        )
    )
    pytester.makepyfile(test_would_touch_live_state="def test_ok(): assert True")

    result = pytester.runpytest()

    assert result.ret == 0
    outcomes = result.parseoutcomes()
    assert outcomes.get("passed", 0) == 1


def _fixture_marker(fixture_obj: object) -> object:
    """Return the scope/autouse-bearing marker for a ``@pytest.fixture``
    function, independent of pytest's internal fixture representation.

    Newer pytest wraps fixtures in ``FixtureFunctionDefinition`` and stores
    the marker at ``_fixture_function_marker``; older pytest attached
    ``_pytestfixturefunction`` directly to the function. Neither being
    present is itself a signal worth failing loudly on, rather than silently
    skipping the check.
    """
    marker = getattr(fixture_obj, "_fixture_function_marker", None)
    if marker is None:
        marker = getattr(fixture_obj, "_pytestfixturefunction", None)
    assert marker is not None, (
        f"{fixture_obj!r} does not look like a @pytest.fixture-decorated "
        "function under this pytest version — update _fixture_marker()"
    )
    return marker


def test_real_fixture_is_session_scoped_autouse() -> None:
    """The production _require_test_cell fixture must genuinely be
    session-scoped and autouse — the two properties that make it run before
    any other integration fixture gets a chance to write. Importing
    tests.integration.conftest executes real Redis/NATS/Ollama reachability
    probes as plain function calls at module load time; each client
    call is patched to raise immediately so this test never opens a socket."""
    with (
        patch.object(
            redis.Redis,
            "from_url",
            side_effect=RuntimeError("blocked: no real connections in tests"),
        ),
        patch.object(
            nats,
            "connect",
            side_effect=RuntimeError("blocked: no real connections in tests"),
        ),
        patch.object(
            httpx,
            "get",
            side_effect=RuntimeError("blocked: no real connections in tests"),
        ),
    ):
        module = importlib.import_module("tests.integration.conftest")

    marker = _fixture_marker(module._require_test_cell)
    assert marker.scope == "session"
    assert marker.autouse is True
