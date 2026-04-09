"""Shared Redis connection helpers.

ARCH-11: Previously each component defined its own ``connect_redis``
helper (or inlined ``redis.Redis(...)``), creating a drift risk where a
change to Redis connection semantics (pooling, TLS, retry policy, etc.)
would have to be applied to every copy individually. ``perception/chess_board.py``
already had a hardcoded ``localhost`` copy that ignored ``AugurConfig``
until the deep-review fix-loop migrated it; the duplication was actively
drifting.

Callers should use ``connect_redis(config)`` from this module instead
of constructing their own client.
"""

from __future__ import annotations

import logging

import redis

from blackboard.config import AugurConfig

log = logging.getLogger("augur.connections")


def connect_redis(config: AugurConfig) -> redis.Redis:
    """Create a Redis client from AugurConfig and verify connectivity.

    Raises ``redis.ConnectionError`` if the ping fails so the caller can
    decide whether to log-and-exit or log-and-continue.
    """
    client = redis.Redis(
        host=config.redis_host,
        port=config.redis_port,
        socket_connect_timeout=config.redis_connect_timeout,
    )
    client.ping()
    log.info("Redis connected (%s)", config.redis_url)
    return client
