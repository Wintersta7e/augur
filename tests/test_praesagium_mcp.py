"""Praesagium MCP tools (Task 11).

Mirrors tests/test_conscientia_mcp.py's harness: monkeypatch
augur_mcp.augur_server._new_redis onto a fakeredis client, exercise the
tools through the real _persistence_ctx()/PersistenceManager path.
"""

from __future__ import annotations

import json

import fakeredis
import pytest

import augur_mcp.augur_server as server
from tabula.persistence import PersistenceManager


@pytest.fixture()
def fake_r(monkeypatch):
    r = fakeredis.FakeStrictRedis(decode_responses=False)
    monkeypatch.setattr(server, "_new_redis", lambda: r)
    return r


# ---------------------------------------------------------------------------
# Empty stores
# ---------------------------------------------------------------------------


def test_patterns_tool_empty_store(fake_r):
    out = server.get_praesagium_patterns()
    assert out == {"patterns": [], "count": 0, "mined_at": None}
    assert "error" not in out


def test_predictions_tool_empty_store(fake_r):
    out = server.get_praesagium_predictions()
    assert out["open"] == []
    assert out["resolved"] == []
    assert out["counts"] == {"open": 0, "resolved_returned": 0}
    assert "error" not in out


# ---------------------------------------------------------------------------
# Patterns: sorting + clamping + counts
# ---------------------------------------------------------------------------


def test_patterns_tool_sorts_active_before_provisional_before_retired(fake_r):
    pm = PersistenceManager(fake_r)
    blob = {
        "version": 1,
        "mined_at": 1234.5,
        "hit_rate_watermark": 0.0,
        "patterns": {
            "p_retired": {
                "pattern_id": "p_retired",
                "status": "retired",
                "conf_lower": 0.99,
            },
            "p_active_low": {
                "pattern_id": "p_active_low",
                "status": "active",
                "conf_lower": 0.5,
            },
            "p_active_high": {
                "pattern_id": "p_active_high",
                "status": "active",
                "conf_lower": 0.8,
            },
            "p_provisional": {
                "pattern_id": "p_provisional",
                "status": "provisional",
                "conf_lower": 0.9,
            },
        },
    }
    pm.save_praesagium_patterns(blob)

    out = server.get_praesagium_patterns(limit=50)

    ids = [p["pattern_id"] for p in out["patterns"]]
    # Group order (active, provisional, retired) wins over raw conf_lower --
    # p_retired has the highest conf_lower of all four but sorts last.
    assert ids == [
        "p_active_high",
        "p_active_low",
        "p_provisional",
        "p_retired",
    ]
    assert out["count"] == 4
    assert out["mined_at"] == 1234.5
    assert "error" not in out


def test_patterns_tool_limit_clamps_to_1_and_200(fake_r):
    pm = PersistenceManager(fake_r)
    patterns = {
        f"p{i:03d}": {
            "pattern_id": f"p{i:03d}",
            "status": "provisional",
            "conf_lower": i / 1000.0,
        }
        for i in range(210)
    }
    pm.save_praesagium_patterns(
        {
            "version": 1,
            "mined_at": 42.0,
            "hit_rate_watermark": 0.0,
            "patterns": patterns,
        }
    )

    out_zero = server.get_praesagium_patterns(limit=0)
    assert out_zero["count"] == 1  # clamped to max(1, ...)

    out_negative = server.get_praesagium_patterns(limit=-5)
    assert out_negative["count"] == 1

    out_big = server.get_praesagium_patterns(limit=999)
    assert out_big["count"] == 200  # clamped to <= 200 even with 210 available
    assert len(out_big["patterns"]) == 200


# ---------------------------------------------------------------------------
# Predictions: open + resolved, clamping + counts
# ---------------------------------------------------------------------------


def test_predictions_tool_reports_open_and_resolved(fake_r):
    pm = PersistenceManager(fake_r)
    for i in range(3):
        pm.save_praesagium_open_prediction(
            {"prediction_id": f"open{i}", "pattern_id": "p1"}
        )
    pm.resolve_praesagium_prediction(
        "open0", {"prediction_id": "open0", "outcome": "fulfilled"}
    )

    out = server.get_praesagium_predictions(limit=20)

    assert out["counts"]["open"] == 2
    assert out["counts"]["resolved_returned"] == 1
    assert len(out["open"]) == 2
    assert out["resolved"][0]["prediction_id"] == "open0"
    assert "error" not in out


def test_predictions_tool_resolved_limit_clamps_to_1_and_200(fake_r):
    pm = PersistenceManager(fake_r)
    for i in range(210):
        pm._r.lpush(
            "augur:praesagium:predictions:log",
            json.dumps({"prediction_id": f"r{i}", "outcome": "expired"}),
        )

    out_zero = server.get_praesagium_predictions(limit=0)
    assert out_zero["counts"]["resolved_returned"] == 1

    out_big = server.get_praesagium_predictions(limit=999)
    assert out_big["counts"]["resolved_returned"] == 200
    assert len(out_big["resolved"]) == 200
    # open is unaffected by the resolved-log clamp
    assert out_big["counts"]["open"] == 0
