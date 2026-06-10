"""Shared liveness heartbeat for Praefectus supervision. Every managed faculty
publishes augur.system.heartbeat {faculty, ts}; Praefectus derives liveness.
See docs/superpowers/specs/2026-06-10-praefectus-supervision-health-design.md §4.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time

log = logging.getLogger(__name__)

HEARTBEAT_SUBJECT = "augur.system.heartbeat"


async def heartbeat_loop(nc, faculty: str, interval_s: float) -> None:
    """Publish a heartbeat every interval_s until cancelled. Best-effort:
    a publish error is logged at debug and the loop continues — a heartbeat
    failure must never crash the host faculty."""
    while True:
        try:
            payload = json.dumps({"faculty": faculty, "ts": time.time()}).encode()
            await nc.publish(HEARTBEAT_SUBJECT, payload)
        except Exception as exc:  # noqa: BLE001 - heartbeat is best-effort
            log.debug("heartbeat publish failed (%s): %s", faculty, exc)
        await asyncio.sleep(interval_s)


def start_heartbeat(nc, faculty: str, interval_s: float) -> asyncio.Task:
    """Fire-and-forget the heartbeat loop; return the Task so the caller cancels on shutdown."""
    return asyncio.create_task(heartbeat_loop(nc, faculty, interval_s))
