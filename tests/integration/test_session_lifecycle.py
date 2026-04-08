"""Integration tests: SessionManager against real Redis."""

from __future__ import annotations

import json


from blackboard.session import SessionManager


class TestSessionLifecycle:
    """Verify session start/end state written to Redis."""

    def test_session_start_writes_to_redis(self, redis_client) -> None:
        """Starting a session writes an active record to augur:session:current."""
        sm = SessionManager(redis_client)
        sid = sm.start()

        raw = redis_client.get("augur:session:current")
        assert raw is not None
        data = json.loads(raw)
        assert data["session_id"] == sid
        assert data["status"] == "active"

    def test_session_end_updates_status(self, redis_client) -> None:
        """Ending a session marks status as 'ended' and records ended_at."""
        sm = SessionManager(redis_client)
        sm.start()
        sm.end()

        raw = redis_client.get("augur:session:current")
        assert raw is not None
        data = json.loads(raw)
        assert data["status"] == "ended"
        assert "ended_at" in data

    def test_session_count_increments(self, redis_client) -> None:
        """Each new session start increments the session counter."""
        sm1 = SessionManager(redis_client)
        sm1.start()
        count1 = int(redis_client.get("augur:session:count") or 0)

        sm2 = SessionManager(redis_client)
        sm2.start()
        count2 = int(redis_client.get("augur:session:count") or 0)

        assert count2 == count1 + 1

    def test_multiple_sessions_have_unique_ids(self, redis_client) -> None:
        """Two independently constructed SessionManagers have different IDs."""
        sm1 = SessionManager(redis_client)
        sm2 = SessionManager(redis_client)
        assert sm1.session_id != sm2.session_id
