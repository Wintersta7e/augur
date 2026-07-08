"""Conscientia MCP tools."""

import fakeredis
import pytest

import augur_mcp.augur_server as server
from conscientia import charter
from tabula.persistence import PersistenceManager


@pytest.fixture()
def fake_r(monkeypatch):
    r = fakeredis.FakeStrictRedis(decode_responses=False)
    monkeypatch.setattr(server, "_new_redis", lambda: r)
    return r


def test_charter_tool_needs_no_redis():
    doc = server.get_conscientia_charter()
    assert doc["version"] == charter.CHARTER_VERSION
    assert [p["pid"] for p in doc["principles"]][0] == "pietas"


def test_verdicts_tool(fake_r):
    PersistenceManager(fake_r).save_conscientia_verdict(
        {"proposal_id": "p1", "recommendation": "needs_human"}
    )
    out = server.get_conscientia_verdicts(limit=5)
    assert out["count"] == 1 and out["verdicts"][0]["proposal_id"] == "p1"


def test_violations_tool_clamps(fake_r):
    pm = PersistenceManager(fake_r)
    for i in range(3):
        pm.save_conscientia_violation({"surface": "advice", "code": f"c{i}"})
    out = server.get_conscientia_violations(limit=0)  # clamped to 1
    assert out["count"] == 1
