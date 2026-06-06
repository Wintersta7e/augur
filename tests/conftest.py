"""Shared fixtures for Augur test suite."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import fakeredis
import pytest

# Ensure project root is on sys.path so imports work without installation
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from blackboard.config import AugurConfig  # noqa: E402
from blackboard.persistence import PersistenceManager  # noqa: E402

# ── Gate test constants (plain dicts — advisor_gate.py does not exist yet) ──

SINGLE_MEDIUM: dict[str, Any] = {
    "combined_severity": "MEDIUM",
    "correlation_found": False,
    "primary_anomaly": {
        "domain": "chess",
        "entity": "user",
        "value": 2.0,
        "severity": "medium",
    },
}

SINGLE_MEDIUM_TYPING: dict[str, Any] = {
    "combined_severity": "MEDIUM",
    "correlation_found": False,
    "primary_anomaly": {
        "domain": "typing",
        "entity": "user",
        "value": 2.0,
        "severity": "medium",
    },
}

SINGLE_HIGH_TYPING: dict[str, Any] = {
    "combined_severity": "HIGH",
    "correlation_found": False,
    "primary_anomaly": {
        "domain": "typing",
        "entity": "user",
        "value": 4.5,
        "severity": "high",
    },
}

EXEMPT_PAYLOAD: dict[str, Any] = {
    "combined_severity": "HIGH",
    "correlation_found": True,
    "involved_domains": ["typing", "chess"],
    "correlated_events": [{"domain": "chess"}],
    "primary_anomaly": {
        "domain": "typing",
        "entity": "user",
        "value": 4.5,
        "severity": "high",
    },
}

CORRELATION_MEDIUM: dict[str, Any] = {
    "combined_severity": "MEDIUM",
    "correlation_found": True,
    "involved_domains": ["typing", "chess"],
    "correlated_events": [{"domain": "chess"}],
    "primary_anomaly": {
        "domain": "typing",
        "entity": "user",
        "value": 2.5,
        "severity": "medium",
    },
}

# A single+medium event whose state_key is new — used by the cap-fail-open test.
# The state_key would be "single:activity:newuser"; when channel_stats hash is
# at MAX_GATE_STATE_KEYS, a new key cannot be tracked → fail open to FIRE.
SINGLE_MEDIUM_NEWKEY_THAT_WOULD_SUPPRESS: dict[str, Any] = {
    "combined_severity": "MEDIUM",
    "correlation_found": False,
    "primary_anomaly": {
        "domain": "activity",
        "entity": "newuser",
        "value": 2.0,
        "severity": "medium",
    },
}


# ── Instrumented Redis wrapper ───────────────────────────────────────────────

# Redis command categories for call counting.
_WRITE_CMDS = frozenset(
    [
        "set",
        "hset",
        "hsetnx",
        "lpush",
        "rpush",
        "sadd",
        "srem",
        "ltrim",
        "delete",
        "expire",
    ]
)
_READ_CMDS = frozenset(
    ["get", "hget", "hgetall", "hlen", "hexists", "lrange", "smembers", "sismember"]
)


class _CountingFakeRedis(fakeredis.FakeStrictRedis):
    """FakeStrictRedis that counts read and write commands."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.write_calls: int = 0
        self.read_calls: int = 0

    def execute_command(self, *args: Any, **options: Any) -> Any:
        cmd = args[0].lower() if args else ""
        if cmd in _WRITE_CMDS:
            self.write_calls += 1
        elif cmd in _READ_CMDS:
            self.read_calls += 1
        return super().execute_command(*args, **options)


class _InstrumentedPM(PersistenceManager):
    """PersistenceManager backed by _CountingFakeRedis.

    Exposes ``write_calls`` and ``read_calls`` attributes that mirror the
    underlying Redis client counters — used by gate unit tests that assert
    ``evaluate()`` performs no Redis writes (read-only contract).
    """

    def __init__(self) -> None:
        self._redis = _CountingFakeRedis(decode_responses=True)
        super().__init__(self._redis)

    @property
    def write_calls(self) -> int:
        return self._redis.write_calls

    @property
    def read_calls(self) -> int:
        return self._redis.read_calls


@pytest.fixture
def mock_persistence_manager() -> MagicMock:
    """PersistenceManager stub — no Redis required.

    COV-08: explicit return values for every save_*/load_* method exposed
    on PersistenceManager so a test using this fixture gets predictable
    ``None`` / ``[]`` results rather than a truthy MagicMock default.
    The truthy default was a subtle footgun: ``pm.load_rule_confidence()
    or {}`` would have returned the MagicMock instance instead of ``{}``,
    silently bypassing first-observation initialization in
    analyze_correlation_tuning and similar.
    """
    pm = MagicMock()
    # Pre-existing load methods
    pm.load_thresholds.return_value = None
    pm.load_baseline.return_value = None
    pm.get_history.return_value = []
    pm.get_feedback.return_value = None
    pm.get_all_feedback.return_value = []
    pm.load_prompt.return_value = None
    pm.get_prompt_history.return_value = []
    # Phase 3B additions
    pm.load_escalation_matrix.return_value = None
    # Option A1 additions
    pm.load_correlation_graph.return_value = None
    pm.list_correlation_graphs.return_value = []
    # Deep-review extensions
    pm.load_rule_confidence.return_value = None
    pm.load_reflection.return_value = None
    pm.load_last_anomaly.return_value = None
    pm.load_last_advice.return_value = None
    pm.is_tuning_applied.return_value = False
    # Task-10 additions: rule_window_state + atomic save_tuning_state
    pm.load_rule_window_state.return_value = {}
    pm.save_rule_window_state.return_value = None
    pm.save_tuning_state.return_value = None
    # Save/mark methods return None explicitly (MagicMock default is a
    # new MagicMock, not None — we want callers to see None so a naive
    # `if pm.save_*(...)` check doesn't behave unexpectedly.)
    pm.save_baseline.return_value = None
    pm.save_feedback.return_value = None
    pm.save_prompt.return_value = None
    pm.save_thresholds.return_value = None
    pm.save_escalation_matrix.return_value = None
    pm.save_correlation_graph.return_value = None
    pm.save_rule_confidence.return_value = None
    pm.save_reflection.return_value = None
    pm.save_last_anomaly.return_value = None
    pm.save_last_advice.return_value = None
    pm.mark_tuning_applied.return_value = None
    return pm


# ── Gate fixtures ────────────────────────────────────────────────────────────


@pytest.fixture
def fake_pm() -> _InstrumentedPM:
    """PersistenceManager over FakeStrictRedis with read/write call counters.

    Used by gate unit tests that verify ``evaluate()`` is read-only and that
    the record_* methods perform the expected number of writes.
    """
    return _InstrumentedPM()


@pytest.fixture
def cfg() -> AugurConfig:
    """Default AugurConfig (all gate knobs at spec defaults)."""
    return AugurConfig()


@pytest.fixture
def cfg_disabled() -> AugurConfig:
    """AugurConfig with gate_enabled=False — all gate arms bypassed."""
    return AugurConfig(gate_enabled=False)


@pytest.fixture
def fake_pm_at_cap(monkeypatch: pytest.MonkeyPatch) -> _InstrumentedPM:
    """PersistenceManager whose channel_stats hash is pre-filled to cap.

    Monkeypatches ``blackboard.persistence.MAX_GATE_STATE_KEYS`` to 3 so the
    cap is easy to reach, then pre-fills the hash with 3 distinct keys.  Any
    subsequent attempt to track a *new* key must fail (cap-fail-open path).

    Note: ``MAX_GATE_STATE_KEYS`` is added to persistence.py in Phase 1 Task
    1.1.  Until then this fixture skips the monkeypatch silently (the constant
    does not exist yet) so test collection remains unbroken.
    """
    import blackboard.persistence as _P

    _CAP = 3

    pm = _InstrumentedPM()

    # Monkeypatch the cap constant if it exists (Phase 1+); no-op otherwise.
    if hasattr(_P, "MAX_GATE_STATE_KEYS"):
        monkeypatch.setattr(_P, "MAX_GATE_STATE_KEYS", _CAP)

    # Pre-fill channel_stats to the cap by writing directly to the fake Redis
    # client.  This works before Phase 1 persistence methods exist because we
    # bypass PersistenceManager entirely for setup only.
    import json

    for i in range(_CAP):
        key = f"single:chess:user{i}"
        pm._r.hset("augur:gate:channel_stats", key, json.dumps({"seen": i + 1}))

    return pm
