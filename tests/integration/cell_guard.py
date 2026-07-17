"""Refuse to run the integration suite against the live cell.

A *cell* is an isolated Augur world: a Redis database plus a NATS endpoint,
and the processes bound to them. The blackboard spans both store and bus, so
a Redis-only split leaks — live faculties would still consume test events off
the shared bus. Two cells exist: the live one (db 0, NATS 4222) and the test
one (db 1, NATS 4223).

The suite deletes every ``augur:*`` key before each test. Against the live
cell that destroys real learned state, so this is a safety interlock, not a
lint: the run must abort BEFORE any fixture writes.
"""

from __future__ import annotations

from typing import Any

LIVE_NATS_URLS: frozenset[str] = frozenset(
    {
        "nats://127.0.0.1:4222",
        "nats://localhost:4222",
        "nats://nats:4222",
    }
)


def check_test_cell(config: Any) -> str | None:
    """Return why *config* is not a test cell, or None when it is safe."""
    if config.redis_db == 0:
        return (
            "integration suite points at Redis db 0 (the live cell); it would "
            "delete every augur:* key. Set AUGUR_REDIS_URL=redis://127.0.0.1:6379/1"
        )
    if config.nats_url in LIVE_NATS_URLS:
        return (
            f"integration suite points at the live NATS ({config.nats_url}); live "
            "faculties would consume test events. Set "
            "AUGUR_NATS_URL=nats://127.0.0.1:4223"
        )
    return None
