"""Unit tests for the shared Redis connection helper."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from tabula.config import AugurConfig
from tabula.connections import connect_redis


class TestConnectRedisDb:
    def test_passes_configured_db_to_client(self) -> None:
        cfg = AugurConfig(redis_url="redis://127.0.0.1:6379/1")
        with patch("tabula.connections.redis.Redis") as mock_redis:
            mock_redis.return_value = MagicMock()
            connect_redis(cfg)
        assert mock_redis.call_args.kwargs["db"] == 1

    def test_defaults_to_db_zero(self) -> None:
        cfg = AugurConfig(redis_url="redis://127.0.0.1:6379")
        with patch("tabula.connections.redis.Redis") as mock_redis:
            mock_redis.return_value = MagicMock()
            connect_redis(cfg)
        assert mock_redis.call_args.kwargs["db"] == 0
