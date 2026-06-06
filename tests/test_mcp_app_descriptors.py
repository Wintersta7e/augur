"""The read-only get_app_descriptors MCP tool returns the decoded map."""

from contextlib import contextmanager
from unittest.mock import MagicMock, patch

import augur_mcp.augur_server as server


def test_get_app_descriptors_returns_map():
    pm = MagicMock()
    pm.load_app_descriptors.return_value = {"alpha_app": "Alpha Browser"}

    @contextmanager
    def fake_ctx():
        yield pm

    with patch.object(server, "_persistence_ctx", fake_ctx):
        result = server.get_app_descriptors()
    assert result == {"descriptors": {"alpha_app": "Alpha Browser"}}


def test_get_app_descriptors_error_is_caught():
    @contextmanager
    def boom_ctx():
        raise RuntimeError("redis down")
        yield  # pragma: no cover

    with patch.object(server, "_persistence_ctx", boom_ctx):
        result = server.get_app_descriptors()
    assert "error" in result
