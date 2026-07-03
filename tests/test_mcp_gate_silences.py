"""Tests for get_gate_silences MCP tool (Task 10.2).

Spec §8: new read-only MCP tool ``get_gate_silences`` returns the recent
silence log + per-arm counts.  Tool count goes from 22 → 23.
"""

from __future__ import annotations

import asyncio
from contextlib import contextmanager
from unittest.mock import MagicMock, patch


import augur_mcp.augur_server as server


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------


def _fake_ctx(pm: object):
    @contextmanager
    def _ctx():
        yield pm

    return _ctx


async def _list_tools():
    return await server.mcp.list_tools()


# ---------------------------------------------------------------------------
# Tool count assertion (31 tools)
# ---------------------------------------------------------------------------


def test_mcp_tool_count_is_31():
    """Tool count must be 31 after adding dialogue tools (dialogue_turn/history/pending)."""
    tools = asyncio.run(_list_tools())
    assert len(tools) == 31, (
        f"Expected 31 MCP tools, got {len(tools)}: {[t.name for t in tools]}"
    )


def test_get_gate_silences_tool_exists():
    """The get_gate_silences tool must be registered."""
    tools = asyncio.run(_list_tools())
    names = {t.name for t in tools}
    assert "get_gate_silences" in names


# ---------------------------------------------------------------------------
# Functional tests
# ---------------------------------------------------------------------------


def test_get_gate_silences_returns_records_and_arm_counts():
    """Returns recent silence log + per-arm counts derived from the records."""
    silences = [
        {
            "ts": "2026-01-01T00:00:00Z",
            "decision_id": "abc",
            "state_key": "single:typing:user",
            "domain": "typing",
            "entity": "user",
            "severity": "medium",
            "arm": "habituation",
            "reason": "habituated",
            "metrics": {"h_eff": 0.9},
            "mrt_eligible": False,
            "p_withhold": None,
        },
        {
            "ts": "2026-01-01T00:01:00Z",
            "decision_id": "def",
            "state_key": "single:chess:user",
            "domain": "chess",
            "entity": "user",
            "severity": "medium",
            "arm": "refractory_burden",
            "reason": "absolute_refractory",
            "metrics": {"remaining_s": 12},
            "mrt_eligible": False,
            "p_withhold": None,
        },
        {
            "ts": "2026-01-01T00:02:00Z",
            "decision_id": "ghi",
            "state_key": "single:typing:user",
            "domain": "typing",
            "entity": "user",
            "severity": "medium",
            "arm": "habituation",
            "reason": "habituated",
            "metrics": {"h_eff": 0.95},
            "mrt_eligible": False,
            "p_withhold": None,
        },
    ]

    pm = MagicMock()
    pm.load_silence_records.return_value = silences

    with patch.object(server, "_persistence_ctx", _fake_ctx(pm)):
        result = server.get_gate_silences()

    assert "silences" in result
    assert "arm_counts" in result
    assert "total" in result
    assert result["total"] == 3
    assert result["silences"] == silences
    # Per-arm counts derived from the records
    assert result["arm_counts"]["habituation"] == 2
    assert result["arm_counts"]["refractory_burden"] == 1


def test_get_gate_silences_empty_returns_empty():
    """Empty silence log returns empty lists and zero counts."""
    pm = MagicMock()
    pm.load_silence_records.return_value = []

    with patch.object(server, "_persistence_ctx", _fake_ctx(pm)):
        result = server.get_gate_silences()

    assert result["silences"] == []
    assert result["arm_counts"] == {}
    assert result["total"] == 0


def test_get_gate_silences_limit_is_passed_to_pm():
    """The limit param is forwarded to load_silence_records."""
    pm = MagicMock()
    pm.load_silence_records.return_value = []

    with patch.object(server, "_persistence_ctx", _fake_ctx(pm)):
        server.get_gate_silences(limit=50)

    pm.load_silence_records.assert_called_once_with(limit=50)


def test_get_gate_silences_default_limit():
    """Default limit of 100 is used when not specified."""
    pm = MagicMock()
    pm.load_silence_records.return_value = []

    with patch.object(server, "_persistence_ctx", _fake_ctx(pm)):
        server.get_gate_silences()

    pm.load_silence_records.assert_called_once_with(limit=100)


def test_get_gate_silences_is_readonly():
    """get_gate_silences must not call any save_* / write methods."""
    pm = MagicMock()
    pm.load_silence_records.return_value = []

    with patch.object(server, "_persistence_ctx", _fake_ctx(pm)):
        server.get_gate_silences()

    # Verify no save_* was called
    for call in pm.mock_calls:
        method_name = call[0]
        assert not method_name.startswith("save_"), (
            f"get_gate_silences called write method: {method_name}"
        )


def test_get_gate_silences_redis_error_returns_error_dict():
    """Redis errors are caught and returned as an error dict."""

    @contextmanager
    def boom_ctx():
        raise RuntimeError("redis down")
        yield  # pragma: no cover

    with patch.object(server, "_persistence_ctx", boom_ctx):
        result = server.get_gate_silences()

    assert "error" in result
