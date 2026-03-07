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


class SessionManager:
    """Generates a session ID, stores it in Redis, and tracks start/end."""

    def __init__(self, r: redis.Redis) -> None:
        self._redis = r
        self.session_id = str(uuid.uuid4())

    def start(self) -> str:
        """Record session start in Redis and return the session_id."""
        self._redis.incr(REDIS_KEY_COUNT)
        self._redis.set(REDIS_KEY_CURRENT, json.dumps({
            "session_id": self.session_id,
            "started_at": datetime.now(timezone.utc).isoformat(),
            "status": "active",
        }))
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
