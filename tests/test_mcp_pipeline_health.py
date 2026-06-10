# tests/test_mcp_pipeline_health.py
"""MCP get_pipeline_health reads the snapshot via _persistence_ctx."""

from unittest import mock

import augur_mcp.augur_server as srv


def test_get_pipeline_health_returns_snapshot():
    snap = {"ts": 1.0, "faculties": {"vigil": {"overall": "alive"}}}
    fake_pm = mock.MagicMock()
    fake_pm.load_health_snapshot.return_value = snap
    ctx = mock.MagicMock()
    ctx.__enter__.return_value = fake_pm
    with mock.patch.object(srv, "_persistence_ctx", return_value=ctx):
        assert srv.get_pipeline_health() == snap


def test_get_pipeline_health_warming_up_when_empty():
    fake_pm = mock.MagicMock()
    fake_pm.load_health_snapshot.return_value = None
    ctx = mock.MagicMock()
    ctx.__enter__.return_value = fake_pm
    with mock.patch.object(srv, "_persistence_ctx", return_value=ctx):
        out = srv.get_pipeline_health()
    assert out["status"] == "warming_up"


def test_praefectus_in_component_commands():
    assert srv.COMPONENT_COMMANDS["praefectus"][-1] == "praefectus.monitor"
