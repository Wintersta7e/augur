"""Best-effort violation bookkeeping — the one Conscientia module that
touches persistence (screens stay pure: no Redis, no NATS). Call sites hand
in their PersistenceManager; a bookkeeping failure is logged and swallowed,
never raised — the enforcement outcome (block/refuse) is already decided by
the screen, and recording must not alter it.
"""

from __future__ import annotations

import logging

log = logging.getLogger("conscientia.recording")


def record_violation_best_effort(pm, record: dict) -> None:
    """Persist *record* via ``pm.save_conscientia_violation``, logging (never
    raising) on failure."""
    try:
        pm.save_conscientia_violation(record)
    except Exception:
        log.warning("conscientia violation record failed (non-fatal)", exc_info=True)
