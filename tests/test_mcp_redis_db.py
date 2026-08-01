"""The MCP server must honour the configured Redis db (cell boundary)."""

from __future__ import annotations

from unittest.mock import MagicMock, patch


class TestMcpRedisDb:
    def test_new_redis_passes_db(self) -> None:
        import augur_mcp.augur_server as srv

        with patch.object(
            srv, "_config", srv.AugurConfig(redis_url="redis://h:6379/1")
        ):
            with patch.object(srv.redis, "Redis") as mock_redis:
                mock_redis.return_value = MagicMock()
                srv._new_redis()
        assert mock_redis.call_args.kwargs["db"] == 1
