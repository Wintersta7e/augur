"""Session manager — tracks active session in Redis."""

from __future__ import annotations

import json
import logging
import uuid
from datetime import datetime, timezone

import redis

log = logging.getLogger("session")

REDIS_KEY_CURRENT = "augur:session:current"
REDIS_KEY_COUNT = "augur:session:count"
REDIS_KEY_META = "augur:session:meta:{sid}"
# Provenance must outlive the longest session-scoped record (reflection reports,
# 30 days) or a late reflection would find no provenance and fail closed,
# silently stopping a real session from training. A test pins this relationship.
SESSION_META_TTL_S = 30 * 24 * 3600


def build_session_meta(
    session_id: str, *, origin: str, created_by: str, started_at: str
) -> dict:
    """Provenance record for a session.

    ``learnable`` is derived from ``origin`` here so the two can never drift:
    only an ``origin`` of ``"real"`` may train the system; ``"synthetic"`` and
    ``"unattributed"`` never do.
    """
    return {
        "session_id": session_id,
        "origin": origin,
        "learnable": origin == "real",
        "created_by": created_by,
        "started_at": started_at,
    }


class SessionManager:
    """Generates a session ID, stores it in Redis, and tracks start/end."""

    def __init__(self, r: redis.Redis) -> None:
        self._redis = r
        self.session_id = str(uuid.uuid4())

    def start(self) -> str:
        """Record session start in Redis and return the session_id."""
        self._redis.incr(REDIS_KEY_COUNT)
        self._redis.set(
            REDIS_KEY_CURRENT,
            json.dumps(
                {
                    "session_id": self.session_id,
                    "started_at": datetime.now(timezone.utc).isoformat(),
                    "status": "active",
                }
            ),
        )
        log.info("Session started: %s", self.session_id)
        return self.session_id

    def end(self) -> None:
        """Record session end in Redis."""
        raw = self._redis.get(REDIS_KEY_CURRENT)
        if raw:
            data = json.loads(raw)
            data["ended_at"] = datetime.now(timezone.utc).isoformat()
            data["status"] = "ended"
            self._redis.set(REDIS_KEY_CURRENT, json.dumps(data))
        log.info("Session ended: %s", self.session_id)


def get_active_session(
    r: redis.Redis,
    max_age_h: float = 12.0,
) -> str | None:
    """Return current session_id if active and not stale; else None.

    Validity rules:
      - augur:session:current must exist
      - record must parse as JSON with a session_id
      - status must be "active" (not "ended")
      - if started_at is present, age must be <= max_age_h hours
        (older records without started_at are accepted; they predate the field)
    """
    try:
        raw = r.get(REDIS_KEY_CURRENT)
    except Exception:
        # Redis connection error or other transport failure.
        # Treat as "no valid session" rather than propagating.
        return None
    if not raw:
        return None
    try:
        data = json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return None
    if not isinstance(data, dict):
        return None
    if data.get("status") != "active":
        return None
    session_id = data.get("session_id")
    if not session_id:
        return None
    started_at = data.get("started_at")
    if started_at:
        try:
            started = datetime.fromisoformat(started_at)
            age = datetime.now(timezone.utc) - started
        except (ValueError, TypeError):
            return None
        if age.total_seconds() > max_age_h * 3600:
            return None
    return session_id
